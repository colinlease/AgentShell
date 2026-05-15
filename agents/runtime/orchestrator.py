from __future__ import annotations

from typing import Any
from uuid import uuid4

from agents.core.agent_runner import AgentRunner
from agents.core.models import AgentRequest, ChatMessage
from agents.core.results import AgentResult
from agents.core.runtime_config import AgentRuntimeConfig, RuntimeBudgetConfig
from agents.core.runtime_protocol import BaseAgentRuntime
from app.state.chat_context_state import (
    clear_compacted_summary,
    get_compacted_message_count,
    get_compacted_summary,
    set_compacted_summary,
)
from agents.prompts.app_prompts import get_app_prompt
from agents.prompts.runtime_prompts import build_execution_system_prompt
from agents.providers.base import BaseLLMProvider
from agents.notes.store import RuntimeNoteStore
from agents.notes.tools import build_notes_read_registry, build_notes_reflection_registry
from agents.runtime.critique import CritiqueService
from agents.runtime.catalog import build_tool_catalog_text
from agents.runtime.compaction import ConversationCompactionService
from agents.runtime.context_builder import format_conversation_context
from agents.runtime.models import (
    ConversationContextArtifact,
    CritiqueArtifact,
    OrchestratedRunState,
    PlanningArtifact,
    ReflectionArtifact,
    ReflectionDecision,
    TriageDecision,
)
from agents.runtime.planning import PlanningService
from agents.runtime.reflection import ReflectionService, decide_reflection
from agents.runtime.triage import TriageService
from agents.tools.registry import ToolRegistry


class OrchestratedAgentRuntime(BaseAgentRuntime):
    """
    Future-ready orchestration wrapper around the existing execution runner.

    In Phase 3 this runtime is intentionally transparent: it preserves the
    current execution behavior and delegates all work to the underlying
    `AgentRunner`. Later phases can add hidden triage, planning, critique,
    reflection, and compaction logic here without forcing another factory or
    adapter rewrite.
    """

    def __init__(
        self,
        *,
        execution_runner: AgentRunner,
        runtime_config: AgentRuntimeConfig,
        triage_provider: BaseLLMProvider | None = None,
        planning_provider: BaseLLMProvider | None = None,
        critique_provider: BaseLLMProvider | None = None,
        reflection_provider: BaseLLMProvider | None = None,
        note_store: RuntimeNoteStore | None = None,
    ) -> None:
        self.execution_runner = execution_runner
        self.runtime_config = runtime_config
        self.triage_provider = triage_provider or execution_runner.provider
        self.planning_provider = planning_provider or execution_runner.provider
        self.critique_provider = critique_provider or execution_runner.provider
        self.reflection_provider = reflection_provider or execution_runner.provider
        self.note_store = note_store

    def run(self, request: AgentRequest) -> AgentResult:
        """
        Execute a request using the orchestration flow.
        """
        run_state = self.create_run_state(request)
        while run_state.phase not in {"completed", "failed"}:
            self.advance_run(run_state)
        return self.finalize_run(run_state)

    def create_run_state(
        self,
        request: AgentRequest,
        *,
        run_id: str | None = None,
        target_message_id: str | None = None,
        surface: str | None = None,
    ):
        """
        Create a resumable outer run state.
        """
        provider_name = self.execution_runner.provider.__class__.__name__
        provider_model = getattr(self.execution_runner.provider, "model", None)
        return OrchestratedRunState(
            run_id=run_id or f"orun_{uuid4().hex}",
            user_input=request.user_input,
            request_messages=list(request.messages),
            request_context=dict(request.context),
            target_message_id=target_message_id,
            surface=surface,
            provider=provider_name,
            model=provider_model,
            phase="compaction_pending" if self.runtime_config.features.compaction_enabled else "triage_pending",
        )

    def advance_run(self, run_state):
        """
        Advance a resumable run using the orchestration flow.
        """
        if run_state.phase in {"completed", "failed"}:
            return run_state

        if run_state.phase == "compaction_pending":
            return self._advance_compaction(run_state)

        if run_state.phase == "triage_pending":
            return self._advance_triage(run_state)

        if run_state.phase == "planning_pending":
            return self._advance_planning(run_state)

        if run_state.phase == "critique_pending":
            return self._advance_critique(run_state)

        if run_state.phase == "execution_running":
            return self._advance_execution(run_state)

        if run_state.phase == "reflection_pending":
            return self._advance_reflection(run_state)

        run_state.phase = "failed"
        run_state.stop_reason = "invalid_state"
        run_state.error = f"Unknown orchestration phase '{run_state.phase}'."
        run_state.final_text = "I ran into an internal orchestration error before finishing."
        return run_state

    def finalize_run(self, run_state) -> AgentResult:
        """
        Finalize a run, merging orchestration trace into the final result.
        """
        execution_state = getattr(run_state, "execution_state", None)
        if execution_state is None:
            result = AgentResult(
                final_text=run_state.final_text,
                blocks=[],
                tool_results=list(run_state.tool_results),
                trace=list(run_state.trace),
                stop_reason=run_state.stop_reason,
                error=run_state.error,
                metadata={
                    "provider": run_state.provider,
                    "model": run_state.model,
                    "run_id": run_state.run_id,
                    "surface": run_state.surface or str(run_state.request_context.get("surface", "") or ""),
                    "logging_enabled": bool(run_state.request_context.get("logging_enabled", False)),
                    "session_id": str(run_state.request_context.get("session_id", "") or ""),
                    "started_at_utc": str(run_state.request_context.get("run_started_at_utc", "") or ""),
                    "started_at_ms": int(run_state.request_context.get("run_started_at_ms", 0) or 0),
                },
            )
            if run_state.stop_reason == "stopped":
                result.metadata["stopped"] = True
                stopped_reason = str(getattr(run_state, "stopped_reason", "") or "").strip()
                if stopped_reason:
                    result.metadata["stopped_reason"] = stopped_reason
            self.execution_runner.record_result(user_input=run_state.user_input, result=result)
            return result

        base_result = self.execution_runner.build_result(execution_state)
        base_result.trace = [*run_state.trace, *base_result.trace]
        if run_state.triage is not None:
            base_result.metadata["triage_reason"] = run_state.triage.reason
            base_result.metadata["triage_source"] = run_state.triage.source
        if run_state.plan is not None:
            base_result.metadata["plan_summary"] = run_state.plan.summary
            base_result.metadata["plan_notes_count"] = len(run_state.plan.notes_context)
        if run_state.critique is not None:
            base_result.metadata["critique_summary"] = run_state.critique.summary
        if run_state.reflection is not None:
            base_result.metadata["reflection_summary"] = run_state.reflection.summary
            base_result.metadata["reflection_mutations_applied"] = run_state.reflection.mutations_applied
        base_result.metadata["run_id"] = run_state.run_id
        base_result.metadata["surface"] = run_state.surface or str(run_state.request_context.get("surface", "") or "")
        base_result.metadata["logging_enabled"] = bool(run_state.request_context.get("logging_enabled", False))
        base_result.metadata["session_id"] = str(run_state.request_context.get("session_id", "") or "")
        base_result.metadata["started_at_utc"] = str(run_state.request_context.get("run_started_at_utc", "") or "")
        base_result.metadata["started_at_ms"] = int(run_state.request_context.get("run_started_at_ms", 0) or 0)
        self.execution_runner.record_result(user_input=run_state.user_input, result=base_result)
        return base_result

    def _advance_compaction(self, run_state: OrchestratedRunState) -> OrchestratedRunState:
        visible_messages = self._get_visible_history_messages(run_state)
        compaction_result = self._prepare_conversation_context(
            run_state,
            visible_messages=visible_messages,
            app_prompt=self._get_app_prompt(),
            allow_compaction=bool(self.runtime_config.features.compaction_enabled),
        )
        run_state.trace.append(
            {
                "stage": "compaction_check",
                "provider": run_state.provider,
                "model": getattr(self.execution_runner.provider, "model", None),
                "compaction": {
                    "enabled": bool(self.runtime_config.features.compaction_enabled),
                    "message_count": len(visible_messages),
                    "compacted_message_count": compaction_result["compacted_message_count"],
                    "summary_present": bool(compaction_result["summary"]),
                    "did_compact": bool(compaction_result["did_compact"]),
                },
            }
        )
        if compaction_result["did_compact"]:
            run_state.trace.append(
                {
                    "stage": "compaction",
                    "provider": run_state.provider,
                    "model": getattr(self.execution_runner.provider, "model", None),
                    "compaction": {
                        "compacted_message_count": compaction_result["compacted_message_count"],
                        "recent_message_count": len(compaction_result["recent_messages"]),
                        "summary_chars": len(compaction_result["summary"] or ""),
                    },
                }
            )
            run_state.status_text = "Compacting Context"
            run_state.status_state = "running"
        else:
            run_state.status_text = ""
            run_state.status_state = "running"

        run_state.phase = "triage_pending"
        return run_state

    def _advance_triage(self, run_state: OrchestratedRunState) -> OrchestratedRunState:
        if run_state.conversation_context is None:
            self._prepare_conversation_context(
                run_state,
                visible_messages=self._get_visible_history_messages(run_state),
                app_prompt=self._get_app_prompt(),
                allow_compaction=False,
            )
        request = self._build_request(run_state)
        app_prompt = self._get_app_prompt()
        tool_catalog_text = build_tool_catalog_text(self.execution_runner.tool_registry)
        planning_enabled = bool(self.runtime_config.features.planning_enabled)

        if not planning_enabled:
            decision = TriageDecision(
                should_plan=False,
                planning_mode="skip",
                reason="planning_disabled",
                source="fallback",
            )
        else:
            triage_service = TriageService(provider=self.triage_provider)
            decision = triage_service.decide(
                request=request,
                app_prompt=app_prompt,
                tool_catalog_text=tool_catalog_text,
                conversation_context_text=self._get_conversation_context_text(run_state),
                max_model_calls=self.runtime_config.budgets.triage_model_calls,
            )

        run_state.triage = decision
        run_state.trace.append(
            {
                "stage": "triage",
                "provider": run_state.provider,
                "model": getattr(self.triage_provider, "model", None),
                "decision": {
                    "should_plan": decision.should_plan,
                    "planning_mode": decision.planning_mode,
                    "reason": decision.reason,
                    "source": decision.source,
                },
            }
        )

        if decision.should_plan and decision.planning_mode != "skip":
            run_state.status_text = "Planning"
            run_state.status_state = "running"
            run_state.phase = "planning_pending"
            return run_state

        self._start_execution(run_state, plan=None, critique=None)
        return run_state

    def _advance_planning(self, run_state: OrchestratedRunState) -> OrchestratedRunState:
        request = self._build_request(run_state)
        app_prompt = self._get_app_prompt()
        tool_catalog_text = build_tool_catalog_text(self.execution_runner.tool_registry)
        planning_service = PlanningService(provider=self.planning_provider)
        notes_tool_registry = None
        if self.runtime_config.features.reflection_enabled and self.note_store is not None:
            notes_tool_registry = build_notes_read_registry(self.note_store)
        plan = planning_service.plan(
            request=request,
            app_prompt=app_prompt,
            tool_catalog_text=tool_catalog_text,
            conversation_context_text=self._get_conversation_context_text(run_state),
            notes_tool_registry=notes_tool_registry,
            max_model_calls=self.runtime_config.budgets.planning_model_calls,
            max_note_reads=self.runtime_config.budgets.planning_note_reads,
            execution_provider_turn_budget=self._get_execution_provider_turn_limit(),
            execution_tool_call_budget=self._get_execution_tool_call_limit(),
            execution_note_read_budget=self.runtime_config.budgets.execution_note_reads
            if notes_tool_registry is not None
            else None,
        )
        run_state.plan = plan
        run_state.trace.append(
            {
                "stage": "planning",
                "provider": run_state.provider,
                "model": getattr(self.planning_provider, "model", None),
                "notes_enabled": notes_tool_registry is not None,
                "plan": {
                    "summary": plan.summary,
                    "execution_guidance": plan.execution_guidance,
                    "missing_context": plan.missing_context,
                    "notes_context": plan.notes_context,
                    "notes_tool_activity": plan.notes_tool_activity,
                },
            }
        )
        if run_state.triage is not None and run_state.triage.planning_mode == "deep":
            run_state.status_text = "Critiquing"
            run_state.status_state = "running"
            run_state.phase = "critique_pending"
            return run_state

        self._start_execution(run_state, plan=plan, critique=None)
        return run_state

    def _advance_critique(self, run_state: OrchestratedRunState) -> OrchestratedRunState:
        if run_state.plan is None:
            run_state.phase = "failed"
            run_state.stop_reason = "invalid_state"
            run_state.error = "Critique phase started without a plan."
            run_state.final_text = "I ran into an internal critique error before finishing."
            return run_state

        tool_catalog_text = build_tool_catalog_text(self.execution_runner.tool_registry)
        critique_service = CritiqueService(provider=self.critique_provider)
        critique = critique_service.critique(
            request=self._build_request(run_state),
            plan=run_state.plan,
            app_prompt=self._get_app_prompt(),
            tool_catalog_text=tool_catalog_text,
            conversation_context_text=self._get_conversation_context_text(run_state),
            max_model_calls=self.runtime_config.budgets.critique_model_calls,
        )
        run_state.critique = critique
        run_state.trace.append(
            {
                "stage": "critique",
                "provider": run_state.provider,
                "model": getattr(self.critique_provider, "model", None),
                "notes_enabled": bool(run_state.plan.notes_context),
                "critique": {
                    "summary": critique.summary,
                    "issues": critique.issues,
                    "revised_execution_guidance": critique.revised_execution_guidance,
                    "notes_context": run_state.plan.notes_context,
                },
            }
        )
        self._start_execution(run_state, plan=run_state.plan, critique=critique)
        return run_state

    def _advance_execution(self, run_state: OrchestratedRunState) -> OrchestratedRunState:
        execution_runtime = getattr(run_state, "execution_runtime", None)
        execution_state = run_state.execution_state
        if execution_state is None:
            run_state.phase = "failed"
            run_state.stop_reason = "invalid_state"
            run_state.error = "Execution state was missing."
            run_state.final_text = "I ran into an internal execution error before finishing."
            return run_state

        if execution_runtime is None:
            run_state.phase = "failed"
            run_state.stop_reason = "invalid_state"
            run_state.error = "Execution runtime was missing."
            run_state.final_text = "I ran into an internal execution error before finishing."
            return run_state

        execution_runtime.advance_run(execution_state)
        run_state.tool_results = list(execution_state.tool_results)
        run_state.status_text = str(getattr(execution_state, "status_text", "") or "")
        run_state.status_state = str(getattr(execution_state, "status_state", "running") or "running")

        if execution_state.phase in {"completed", "failed"}:
            run_state.execution_terminal_phase = execution_state.phase
            run_state.stop_reason = execution_state.stop_reason
            run_state.final_text = execution_state.final_text
            run_state.error = execution_state.error
            reflection_decision = self._decide_reflection(run_state)
            run_state.reflection_decision = reflection_decision
            run_state.trace.append(
                {
                    "stage": "reflection_gate",
                    "provider": run_state.provider,
                    "model": getattr(self.reflection_provider, "model", None),
                    "decision": {
                        "should_reflect": reflection_decision.should_reflect,
                        "forced": reflection_decision.forced,
                        "reason": reflection_decision.reason,
                        "source": reflection_decision.source,
                        "tool_use_threshold": self.runtime_config.gates.reflection_tool_use_threshold,
                    },
                    "tool_count": len(run_state.tool_results),
                }
            )
            if reflection_decision.should_reflect:
                run_state.status_text = "Reflecting"
                run_state.status_state = "running"
                run_state.phase = "reflection_pending"
                return run_state

            run_state.phase = execution_state.phase
            run_state.status_text = ""
            return run_state

        run_state.phase = "execution_running"
        return run_state

    def _advance_reflection(self, run_state: OrchestratedRunState) -> OrchestratedRunState:
        execution_state = run_state.execution_state
        if execution_state is None:
            run_state.phase = run_state.execution_terminal_phase or "failed"
            run_state.status_text = ""
            return run_state

        if not self.runtime_config.features.reflection_enabled or self.note_store is None:
            run_state.phase = run_state.execution_terminal_phase or "completed"
            run_state.status_text = ""
            return run_state

        try:
            reflection_service = ReflectionService(provider=self.reflection_provider)
            reflection = reflection_service.reflect(
                request=self._build_request(run_state),
                result=self.execution_runner.build_result(execution_state),
                plan=run_state.plan,
                critique=run_state.critique,
                app_prompt=self._get_app_prompt(),
                conversation_context_text=self._get_conversation_context_text(run_state),
                active_app_id=self._get_active_app_id(),
                app_specific_tools_used=self._get_app_specific_tools_used(execution_state),
                orchestration_trace=list(run_state.trace),
                execution_trace=list(execution_state.trace),
                notes_tool_registry=build_notes_reflection_registry(self.note_store),
                max_model_calls=self.runtime_config.budgets.reflection_model_calls,
                max_tool_calls=self.runtime_config.budgets.reflection_tool_calls,
                max_note_writes=self.runtime_config.budgets.note_writes,
                max_note_deletes=self.runtime_config.budgets.note_deletes,
            )
        except Exception as exc:  # noqa: BLE001
            reflection = ReflectionArtifact(
                summary="Reflection failed internally and was skipped.",
                lessons=[],
                tool_activity=[
                    {
                        "tool_name": "reflection_internal_error",
                        "status": "error",
                        "file_name": "",
                        "note_id": "",
                        "replaced": False,
                        "deleted": False,
                        "message": str(exc),
                    }
                ],
                note_files_touched=[],
                mutations_applied=0,
            )

        run_state.reflection = reflection
        run_state.trace.append(
            {
                "stage": "reflection",
                "provider": run_state.provider,
                "model": getattr(self.reflection_provider, "model", None),
                "reflection": {
                    "summary": reflection.summary,
                    "lessons": reflection.lessons,
                    "tool_activity": reflection.tool_activity,
                    "note_files_touched": reflection.note_files_touched,
                    "mutations_applied": reflection.mutations_applied,
                    "active_app_id": self._get_active_app_id(),
                },
            }
        )
        run_state.status_text = ""
        run_state.phase = run_state.execution_terminal_phase or "completed"
        return run_state

    def _build_request(self, run_state: OrchestratedRunState) -> AgentRequest:
        return AgentRequest(
            user_input=run_state.user_input,
            messages=list(run_state.request_messages),
            context=dict(run_state.request_context),
        )

    def _get_app_prompt(self) -> str:
        from app.components.workspace_host import get_active_workspace_app

        active_workspace_app = get_active_workspace_app()
        active_app_id = getattr(active_workspace_app, "app_id", None)
        return get_app_prompt(active_app_id)

    def _get_active_app_id(self) -> str | None:
        from app.components.workspace_host import get_active_workspace_app

        active_workspace_app = get_active_workspace_app()
        active_app_id = getattr(active_workspace_app, "app_id", None)
        return str(active_app_id).strip() if active_app_id else None

    def _start_execution(
        self,
        run_state: OrchestratedRunState,
        *,
        plan: PlanningArtifact | None,
        critique: CritiqueArtifact | None,
    ) -> None:
        planning_guidance = (
            critique.revised_execution_guidance
            if critique is not None and critique.revised_execution_guidance
            else plan.execution_guidance if plan is not None
            else None
        )
        system_prompt = build_execution_system_prompt(
            app_prompt=self._get_app_prompt(),
            context_summary=self._get_conversation_context_text(run_state) or None,
            planning_guidance=planning_guidance,
            notes_enabled=self.runtime_config.features.reflection_enabled and self.note_store is not None,
            max_note_reads=self.runtime_config.budgets.execution_note_reads,
            max_execution_provider_turns=self._get_execution_provider_turn_limit(),
            max_execution_tool_calls=self._get_execution_tool_call_limit(),
        )
        execution_runtime = AgentRunner(
            provider=self.execution_runner.provider,
            tool_registry=self._build_execution_tool_registry(),
            system_prompt=system_prompt,
            max_steps=self._get_execution_provider_turn_limit(),
            max_tool_calls=self._get_execution_tool_call_limit(),
            tool_scope_limits={"runtime": self.runtime_config.budgets.execution_note_reads}
            if self.runtime_config.features.reflection_enabled and self.note_store is not None
            else None,
        )
        request = self._build_request(run_state)
        execution_state = execution_runtime.create_run_state(
            request,
            run_id=run_state.run_id,
            target_message_id=run_state.target_message_id,
            surface=run_state.surface,
        )
        run_state.execution_runtime = execution_runtime
        run_state.execution_state = execution_state
        run_state.tool_results = list(execution_state.tool_results)
        run_state.status_text = ""
        run_state.status_state = "running"
        run_state.phase = "execution_running"

    def _decide_reflection(self, run_state: OrchestratedRunState) -> ReflectionDecision:
        if not self.runtime_config.features.reflection_enabled:
            return ReflectionDecision(
                should_reflect=False,
                forced=False,
                reason="reflection_disabled",
                source="fallback",
            )

        return decide_reflection(
            user_input=run_state.user_input,
            stop_reason=run_state.stop_reason,
            tool_count=len(run_state.tool_results),
            tool_failures=sum(1 for tool_result in run_state.tool_results if not bool(getattr(tool_result, "success", False))),
            used_planning=run_state.plan is not None,
            used_critique=run_state.critique is not None,
            reflection_tool_use_threshold=self.runtime_config.gates.reflection_tool_use_threshold,
        )

    def _get_execution_provider_turn_limit(self) -> int:
        default_value = RuntimeBudgetConfig().execution_provider_turns
        configured_value = self.runtime_config.budgets.execution_provider_turns
        if int(configured_value) != int(default_value):
            return int(configured_value)
        return int(self.execution_runner.max_steps)

    def _get_execution_tool_call_limit(self) -> int | None:
        default_value = RuntimeBudgetConfig().execution_tool_calls
        configured_value = self.runtime_config.budgets.execution_tool_calls
        if int(configured_value) != int(default_value):
            return int(configured_value)
        return getattr(self.execution_runner, "max_tool_calls", None)

    def _get_app_specific_tools_used(self, execution_state) -> list[str]:
        if execution_state is None:
            return []

        app_tool_names = {
            str(tool.name).strip()
            for tool in self.execution_runner.tool_registry.list_tools_by_scope("app")
            if str(tool.name).strip()
        }
        if not app_tool_names:
            return []

        used: list[str] = []
        for tool_result in getattr(execution_state, "tool_results", []):
            tool_name = str(getattr(tool_result, "tool_name", "") or "").strip()
            if not tool_name or tool_name not in app_tool_names or tool_name in used:
                continue
            used.append(tool_name)
        return used

    def _prepare_conversation_context(
        self,
        run_state: OrchestratedRunState,
        *,
        visible_messages: list[ChatMessage],
        app_prompt: str,
        allow_compaction: bool,
    ) -> dict[str, Any]:
        summary: str | None = None
        compacted_message_count = 0
        did_compact = False

        if allow_compaction:
            summary = get_compacted_summary()
            compacted_message_count = min(get_compacted_message_count(), len(visible_messages))
            uncompacted_messages = list(visible_messages[compacted_message_count:])
            compaction_service = ConversationCompactionService(provider=self.execution_runner.provider)
            if compaction_service.should_compact(
                messages=uncompacted_messages,
                trigger_message_count=self.runtime_config.budgets.compaction_message_trigger,
                keep_recent=self.runtime_config.budgets.compaction_keep_recent_messages,
                trigger_character_count=self.runtime_config.budgets.compaction_char_trigger,
            ):
                older_messages, recent_messages = compaction_service.split_for_compaction(
                    uncompacted_messages,
                    keep_recent=self.runtime_config.budgets.compaction_keep_recent_messages,
                )
                if older_messages:
                    summary = compaction_service.compact(
                        older_messages=older_messages,
                        existing_summary=summary,
                        app_prompt=app_prompt,
                        max_summary_chars=self.runtime_config.budgets.compaction_max_summary_chars,
                        max_model_calls=self.runtime_config.budgets.compaction_max_model_calls,
                    )
                    compacted_message_count += len(older_messages)
                    set_compacted_summary(summary, compacted_message_count)
                    did_compact = True
            else:
                recent_messages = uncompacted_messages[-self.runtime_config.budgets.compaction_keep_recent_messages :]
                if summary is None and compacted_message_count == 0:
                    clear_compacted_summary()
        else:
            recent_messages = visible_messages[-self.runtime_config.budgets.compaction_keep_recent_messages :]

        context_text = format_conversation_context(
            summary=summary if allow_compaction else None,
            recent_messages=recent_messages,
        )
        run_state.conversation_context = ConversationContextArtifact(
            summary=summary if allow_compaction else None,
            recent_messages=list(recent_messages),
            compacted_message_count=compacted_message_count if allow_compaction else 0,
            context_text=context_text,
        )
        return {
            "summary": run_state.conversation_context.summary,
            "recent_messages": list(run_state.conversation_context.recent_messages),
            "compacted_message_count": run_state.conversation_context.compacted_message_count,
            "did_compact": did_compact,
        }

    def _get_visible_history_messages(self, run_state: OrchestratedRunState) -> list[ChatMessage]:
        messages = list(run_state.request_messages)
        if (
            messages
            and messages[-1].role == "user"
            and " ".join(str(messages[-1].content or "").split()).strip()
            == " ".join(str(run_state.user_input or "").split()).strip()
        ):
            return messages[:-1]
        return messages

    @staticmethod
    def _get_conversation_context_text(run_state: OrchestratedRunState) -> str:
        artifact = getattr(run_state, "conversation_context", None)
        if artifact is None:
            return ""
        return str(artifact.context_text or "").strip()

    def _build_execution_tool_registry(self) -> ToolRegistry:
        if not self.runtime_config.features.reflection_enabled or self.note_store is None:
            return self.execution_runner.tool_registry

        merged_tools = [
            *self.execution_runner.tool_registry.list_tools(),
            *build_notes_read_registry(self.note_store).list_tools(),
        ]
        return ToolRegistry(merged_tools)
