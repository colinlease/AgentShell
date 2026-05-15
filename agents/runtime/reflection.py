from __future__ import annotations

from typing import Any

from agents.core.models import AgentRequest, ChatMessage
from agents.core.results import AgentResult
from agents.core.runtime_config import validate_runtime_budget_value
from agents.prompts.runtime_prompts import build_reflection_prompt
from agents.providers.base import BaseLLMProvider
from agents.runtime.json_utils import parse_json_object
from agents.runtime.models import CritiqueArtifact, PlanningArtifact, ReflectionArtifact, ReflectionDecision
from agents.runtime.tool_dialogue import run_bounded_tool_dialogue
from agents.runtime.triage import detect_obvious_low_complexity
from agents.tools.registry import ToolRegistry


def decide_reflection(
    *,
    user_input: str,
    stop_reason: str,
    tool_count: int,
    tool_failures: int = 0,
    used_planning: bool,
    used_critique: bool,
    reflection_tool_use_threshold: int = 4,
) -> ReflectionDecision:
    """
    Decide whether reflection is worth running for the finished turn.
    """
    tool_use_threshold = max(0, int(reflection_tool_use_threshold))

    if stop_reason == "max_steps":
        return ReflectionDecision(should_reflect=True, forced=True, reason="max_steps")

    if stop_reason == "provider_error":
        return ReflectionDecision(should_reflect=True, forced=True, reason="provider_error")

    if int(tool_count) > tool_use_threshold:
        return ReflectionDecision(
            should_reflect=True,
            forced=True,
            reason=f"tool_usage_gt_{tool_use_threshold}",
        )

    trivial_turn = detect_obvious_low_complexity(user_input)
    if trivial_turn is not None and stop_reason == "completed" and int(tool_count) == 0:
        return ReflectionDecision(
            should_reflect=False,
            forced=False,
            reason=trivial_turn.reason,
        )

    if int(tool_failures) > 0:
        return ReflectionDecision(should_reflect=True, forced=False, reason="tool_failures_present")

    return ReflectionDecision(should_reflect=False, forced=False, reason="low_value_completed_turn")


class ReflectionService:
    """
    Hidden bounded reflection phase with note-only tool access.
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    def reflect(
        self,
        *,
        request: AgentRequest,
        result: AgentResult,
        plan: PlanningArtifact | None,
        critique: CritiqueArtifact | None,
        app_prompt: str,
        conversation_context_text: str | None,
        active_app_id: str | None,
        app_specific_tools_used: list[str],
        orchestration_trace: list[dict[str, Any]],
        execution_trace: list[dict[str, Any]],
        notes_tool_registry: ToolRegistry,
        max_model_calls: int = 8,
        max_tool_calls: int = 8,
        max_note_writes: int = 4,
        max_note_deletes: int = 2,
    ) -> ReflectionArtifact:
        validate_runtime_budget_value("reflection_model_calls", max_model_calls)
        validate_runtime_budget_value("reflection_tool_calls", max_tool_calls)
        validate_runtime_budget_value("note_writes", max_note_writes)
        validate_runtime_budget_value("note_deletes", max_note_deletes)
        tool_catalog_text = self._build_tool_catalog_text(notes_tool_registry)
        reflection_prompt = build_reflection_prompt(
            app_prompt=app_prompt,
            tool_catalog_text=tool_catalog_text,
            max_tool_calls=max_tool_calls,
        )
        reflection_user_message = self._build_reflection_user_message(
            request=request,
            result=result,
            plan=plan,
            critique=critique,
            conversation_context_text=conversation_context_text,
            active_app_id=active_app_id,
            orchestration_trace=orchestration_trace,
            execution_trace=execution_trace,
        )

        working_messages = [ChatMessage(role="user", content=reflection_user_message)]
        remaining_note_writes_ref = [int(max_note_writes)]
        remaining_note_deletes_ref = [int(max_note_deletes)]
        tool_activity: list[dict[str, Any]] = []

        def execute_tool_call(tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
            return self._execute_reflection_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                notes_tool_registry=notes_tool_registry,
                active_app_id=active_app_id,
                app_specific_tools_used=app_specific_tools_used,
                remaining_note_writes_ref=remaining_note_writes_ref,
                remaining_note_deletes_ref=remaining_note_deletes_ref,
            )

        dialogue = run_bounded_tool_dialogue(
            provider=self.provider,
            working_messages=working_messages,
            system_prompt=reflection_prompt,
            tool_registry=notes_tool_registry,
            max_model_calls=max_model_calls,
            max_tool_calls=max_tool_calls,
            execute_tool_call=execute_tool_call,
        )

        for step in dialogue.steps:
            normalized_arguments = self._normalize_note_tool_arguments(
                tool_name=step.tool_name,
                arguments=step.arguments,
                active_app_id=active_app_id,
                app_specific_tools_used=app_specific_tools_used,
            )
            tool_activity.append(
                self._build_tool_activity(
                    tool_name=step.tool_name or "unknown",
                    arguments=normalized_arguments,
                    output=step.tool_output,
                )
            )

        parsed = parse_json_object(dialogue.final_text)

        if not parsed:
            return ReflectionArtifact(
                summary="Reflection completed without a structured summary.",
                lessons=[],
                tool_activity=tool_activity,
                note_files_touched=self._collect_note_files_touched(tool_activity),
                mutations_applied=self._count_applied_mutations(tool_activity),
            )

        lessons_raw = parsed.get("lessons", [])
        lessons = (
            [str(item).strip() for item in lessons_raw if str(item).strip()]
            if isinstance(lessons_raw, list)
            else []
        )

        return ReflectionArtifact(
            summary=str(parsed.get("summary", "")).strip() or "Reflection completed.",
            lessons=lessons,
            tool_activity=tool_activity,
            note_files_touched=self._collect_note_files_touched(tool_activity),
            mutations_applied=self._count_applied_mutations(tool_activity),
        )

    @staticmethod
    def _build_reflection_user_message(
        *,
        request: AgentRequest,
        result: AgentResult,
        plan: PlanningArtifact | None,
        critique: CritiqueArtifact | None,
        conversation_context_text: str | None,
        active_app_id: str | None,
        orchestration_trace: list[dict[str, Any]],
        execution_trace: list[dict[str, Any]],
    ) -> str:
        execution_summary = ReflectionService._format_execution_trace(execution_trace)
        orchestration_summary = ReflectionService._format_orchestration_trace(orchestration_trace)
        plan_summary = plan.summary if plan is not None else "none"
        critique_summary = critique.summary if critique is not None else "none"

        return (
            "Reflect on the just-finished run and maintain the runtime notes.\n"
            "Do not re-solve the task for the user. Use note tools only if they help improve future agent behavior.\n"
            "The notes are for future LLM/runtime effectiveness, not for user education. "
            "If no high-value note maintenance is warranted, do not make note changes.\n"
            "Return JSON only with keys: summary (string), lessons (array of strings).\n\n"
            f"Hidden prior conversation context:\n{conversation_context_text or '- none'}\n\n"
            f"Latest user turn:\n{request.user_input}\n\n"
            f"Final assistant response:\n{result.final_text}\n\n"
            f"Stop reason:\n{result.stop_reason}\n\n"
            f"Active workspace app id:\n{active_app_id or 'none'}\n\n"
            f"Plan summary:\n{plan_summary}\n\n"
            f"Critique summary:\n{critique_summary}\n\n"
            f"Orchestration summary:\n{orchestration_summary}\n\n"
            f"Execution summary:\n{execution_summary}"
        )

    @staticmethod
    def _format_orchestration_trace(trace: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for item in trace[-10:]:
            if not isinstance(item, dict):
                continue
            stage = str(item.get("stage") or "unknown")
            if stage == "triage":
                decision = item.get("decision", {}) or {}
                lines.append(
                    f"- triage: {decision.get('planning_mode', 'unknown')} ({decision.get('reason', 'no_reason')})"
                )
            elif stage == "planning":
                plan = item.get("plan", {}) or {}
                lines.append(f"- planning: {str(plan.get('summary') or '').strip()[:180]}")
            elif stage == "critique":
                critique = item.get("critique", {}) or {}
                lines.append(f"- critique: {str(critique.get('summary') or '').strip()[:180]}")

        return "\n".join(line for line in lines if line.strip()) or "- none"

    @staticmethod
    def _format_execution_trace(trace: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for item in trace[-16:]:
            if not isinstance(item, dict):
                continue
            stage = str(item.get("stage") or "unknown")
            step = item.get("step", "—")
            if stage == "provider_error":
                lines.append(f"- step {step}: provider_error")
            elif stage == "tool_requested":
                lines.append(f"- step {step}: requested {str(item.get('tool_name') or 'unknown')}")
            elif stage == "tool_result":
                status = "success" if bool(item.get("success", False)) else "failed"
                lines.append(f"- step {step}: {status} {str(item.get('tool_name') or 'unknown')}")
            elif stage == "provider_response":
                response = item.get("response", {}) or {}
                if response.get("tool_call"):
                    lines.append(f"- step {step}: provider requested a tool")
                else:
                    lines.append(f"- step {step}: provider returned final text")

        return "\n".join(lines) if lines else "- none"

    @staticmethod
    def _build_tool_catalog_text(tool_registry: ToolRegistry) -> str:
        lines: list[str] = []
        for schema in tool_registry.list_tool_schemas():
            if not isinstance(schema, dict):
                continue
            lines.append(
                f"- {str(schema.get('name') or '').strip()}: {str(schema.get('description') or '').strip()}"
            )
        return "\n".join(line for line in lines if line.strip())

    @staticmethod
    def _tool_budget_error(
        *,
        tool_name: str,
        remaining_note_writes: int,
        remaining_note_deletes: int,
    ) -> str | None:
        if tool_name == "upsert_runtime_note" and remaining_note_writes <= 0:
            return "Reflection note-write budget exhausted."
        if tool_name == "delete_runtime_note" and remaining_note_deletes <= 0:
            return "Reflection note-delete budget exhausted."
        return None

    @staticmethod
    def _execute_reflection_tool_call(
        *,
        tool_name: str,
        arguments: dict[str, Any],
        notes_tool_registry: ToolRegistry,
        active_app_id: str | None,
        app_specific_tools_used: list[str],
        remaining_note_writes_ref: list[int],
        remaining_note_deletes_ref: list[int],
    ) -> tuple[dict[str, Any], bool]:
        normalized_arguments = ReflectionService._normalize_note_tool_arguments(
            tool_name=tool_name,
            arguments=arguments,
            active_app_id=active_app_id,
            app_specific_tools_used=app_specific_tools_used,
        )

        budget_error = ReflectionService._tool_budget_error(
            tool_name=tool_name,
            remaining_note_writes=remaining_note_writes_ref[0],
            remaining_note_deletes=remaining_note_deletes_ref[0],
        )
        if budget_error is not None:
            return (
                {
                    "status": "error",
                    "message": budget_error,
                },
                False,
            )

        if not notes_tool_registry.has(tool_name):
            return (
                {
                    "status": "error",
                    "message": f"Reflection-phase note tool '{tool_name}' is not available.",
                },
                False,
            )

        if tool_name == "upsert_runtime_note":
            remaining_note_writes_ref[0] -= 1
        elif tool_name == "delete_runtime_note":
            remaining_note_deletes_ref[0] -= 1

        tool = notes_tool_registry.get(tool_name)
        try:
            return tool.run(**normalized_arguments), True
        except Exception as exc:  # noqa: BLE001
            return (
                {
                    "status": "error",
                    "message": str(exc),
                },
                True,
            )

    @staticmethod
    def _build_tool_activity(
        *,
        tool_name: str,
        arguments: dict[str, Any],
        output: dict[str, Any],
    ) -> dict[str, Any]:
        file_name = str(output.get("file_name") or arguments.get("file_name") or "").strip()
        return {
            "tool_name": tool_name,
            "status": str(output.get("status") or "ok"),
            "file_name": file_name,
            "scope": str(output.get("scope") or arguments.get("scope") or "").strip(),
            "app_id": str(output.get("app_id") or arguments.get("app_id") or "").strip(),
            "note_id": str(output.get("note_id") or arguments.get("note_id") or "").strip(),
            "replaced": bool(output.get("replaced", False)),
            "deleted": bool(output.get("deleted", False)),
            "message": str(output.get("message") or "").strip(),
            "statement": str((output.get("note") or {}).get("statement") or arguments.get("statement") or "").strip(),
            "confidence": (output.get("note") or {}).get("confidence", arguments.get("confidence")),
        }

    @staticmethod
    def _collect_note_files_touched(tool_activity: list[dict[str, Any]]) -> list[str]:
        touched: list[str] = []
        for item in tool_activity:
            file_name = str(item.get("file_name") or "").strip()
            if not file_name or file_name in touched:
                continue
            touched.append(file_name)
        return touched

    @staticmethod
    def _count_applied_mutations(tool_activity: list[dict[str, Any]]) -> int:
        count = 0
        for item in tool_activity:
            if str(item.get("status") or "").strip().lower() != "ok":
                continue
            if str(item.get("tool_name") or "") in {"upsert_runtime_note", "delete_runtime_note"}:
                count += 1
        return count

    @staticmethod
    def _normalize_note_tool_arguments(
        *,
        tool_name: str,
        arguments: dict[str, Any],
        active_app_id: str | None,
        app_specific_tools_used: list[str],
    ) -> dict[str, Any]:
        if tool_name != "upsert_runtime_note":
            return arguments

        normalized = dict(arguments)
        if not active_app_id:
            return normalized

        requested_file = " ".join(str(normalized.get("file_name", "")).strip().split()) or "general"
        requested_scope = " ".join(str(normalized.get("scope", "")).strip().split()) or "general"
        if requested_file != "general" and requested_scope != "general":
            return normalized

        if not ReflectionService._should_route_note_to_active_app(
            arguments=normalized,
            active_app_id=active_app_id,
            app_specific_tools_used=app_specific_tools_used,
        ):
            return normalized

        normalized["file_name"] = active_app_id
        normalized["scope"] = "app"
        normalized["app_id"] = active_app_id
        return normalized

    @staticmethod
    def _should_route_note_to_active_app(
        *,
        arguments: dict[str, Any],
        active_app_id: str,
        app_specific_tools_used: list[str],
    ) -> bool:
        text_parts = [
            str(arguments.get("note_id") or ""),
            str(arguments.get("title") or ""),
            str(arguments.get("statement") or ""),
            " ".join(str(item) for item in (arguments.get("tags") or []) if str(item).strip()),
            " ".join(str(item) for item in (arguments.get("keywords") or []) if str(item).strip()),
        ]
        normalized_blob = " ".join(" ".join(part.strip().lower().split()) for part in text_parts if part).strip()
        if not normalized_blob:
            return bool(app_specific_tools_used)

        if "cross-app" in normalized_blob or "across apps" in normalized_blob or "agentshell" in normalized_blob:
            return False

        if active_app_id.strip().lower() in normalized_blob:
            return True

        return any(tool_name.strip().lower() in normalized_blob for tool_name in app_specific_tools_used)
