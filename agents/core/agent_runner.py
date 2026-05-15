from __future__ import annotations

import json
from typing import Any
from uuid import uuid4
import streamlit as st

from agents.core.models import AgentRequest, AgentRunState, ChatMessage, ToolCall, ToolResult
from agents.core.runtime_protocol import BaseAgentRuntime
from agents.core.results import AgentResult
from agents.observability.run_logger import persist_completed_run_log
from agents.providers.base import BaseLLMProvider
from agents.tools.registry import ToolRegistry


class AgentRunner(BaseAgentRuntime):
    """
    Lightweight orchestration loop for a single-agent, tool-using workflow.

    This is intentionally simple for the first version of the app. It supports:
    - provider-agnostic model calls
    - optional tool use
    - bounded step execution
    - structured tracing

    The runner is designed so it can later be replaced or upgraded with a more
    advanced orchestration layer, including LangGraph, without forcing major
    changes to the rest of the app.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        tool_registry: ToolRegistry,
        system_prompt: str,
        *,
        max_steps: int = 15,
        max_tool_calls: int | None = None,
        tool_scope_limits: dict[str, int] | None = None,
    ) -> None:
        self.provider = provider
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tool_calls = None if max_tool_calls is None else max(0, int(max_tool_calls))
        self.tool_scope_limits = dict(tool_scope_limits or {})

    def run(self, request: AgentRequest) -> AgentResult:
        """
        Execute a bounded agent run.
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
    ) -> AgentRunState:
        """
        Create resumable state for a new agent run.
        """
        provider_name = self.provider.__class__.__name__
        provider_model = getattr(self.provider, "model", None)

        working_messages = list(request.messages)
        working_messages.append(ChatMessage(role="user", content=request.user_input))

        return AgentRunState(
            run_id=run_id or f"run_{uuid4().hex}",
            user_input=request.user_input,
            working_messages=working_messages,
            context=dict(request.context),
            max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls,
            target_message_id=target_message_id,
            surface=surface,
            provider=provider_name,
            model=provider_model,
            status_text="",
            status_state="running",
        )

    def advance_run(self, run_state: AgentRunState) -> AgentRunState:
        """
        Advance a resumable run by one provider or tool step.
        """
        if run_state.phase in {"completed", "failed"}:
            return run_state

        if run_state.phase == "provider_pending":
            return self._advance_provider_step(run_state)

        if run_state.phase == "tool_pending":
            return self._advance_tool_step(run_state)

        unknown_phase = run_state.phase
        run_state.phase = "failed"
        run_state.stop_reason = "invalid_state"
        run_state.error = f"Unknown run phase '{unknown_phase}'."
        run_state.final_text = "I ran into an internal state error before finishing."
        return run_state

    def finalize_run(self, run_state: AgentRunState) -> AgentResult:
        """
        Convert a completed run state into a final AgentResult and record it.
        """
        result = self.build_result(run_state)
        self.record_result(user_input=run_state.user_input, result=result)
        return result

    def build_result(self, run_state: AgentRunState) -> AgentResult:
        """
        Convert a completed run state into a final AgentResult without recording it.
        """
        response_blocks = self._extract_response_blocks(run_state.tool_results)

        metadata = {
            "steps_used": run_state.step_index,
            "provider": run_state.provider,
            "model": run_state.model,
            "run_id": run_state.run_id,
            "surface": run_state.surface or str(run_state.context.get("surface", "") or ""),
            "logging_enabled": bool(run_state.context.get("logging_enabled", False)),
            "session_id": str(run_state.context.get("session_id", "") or ""),
            "started_at_utc": str(run_state.context.get("run_started_at_utc", "") or ""),
            "started_at_ms": int(run_state.context.get("run_started_at_ms", 0) or 0),
        }
        if run_state.stop_reason == "provider_error" and run_state.error:
            metadata["provider_error"] = run_state.error
        if run_state.stop_reason == "stopped":
            metadata["stopped"] = True
            stopped_reason = str(getattr(run_state, "stopped_reason", "") or "").strip()
            if stopped_reason:
                metadata["stopped_reason"] = stopped_reason

        result = AgentResult(
            final_text=run_state.final_text,
            blocks=response_blocks,
            tool_results=run_state.tool_results,
            trace=run_state.trace,
            stop_reason=run_state.stop_reason,
            error=run_state.error,
            metadata=metadata,
        )
        return result

    def record_result(self, *, user_input: str, result: AgentResult) -> None:
        """
        Record a completed AgentResult in session activity state.
        """
        self._record_agent_run(user_input=user_input, result=result)
        try:
            persist_completed_run_log(user_input=user_input, result=result)
        except Exception:
            pass

    def _advance_provider_step(self, run_state: AgentRunState) -> AgentRunState:
        if self._restore_pending_tool_call_from_messages(run_state):
            return run_state

        if run_state.step_index >= run_state.max_steps:
            run_state.phase = "completed"
            run_state.stop_reason = "max_steps"
            run_state.final_text = "I reached the step limit before finishing."
            return run_state

        next_step = run_state.step_index + 1

        try:
            provider_response = self.provider.generate(
                messages=run_state.working_messages,
                system_prompt=self.system_prompt,
                tools=self.tool_registry.list_tool_schemas(),
            )
        except Exception as exc:  # noqa: BLE001
            run_state.trace.append(
                {
                    "step": next_step,
                    "stage": "provider_error",
                    "provider": run_state.provider,
                    "model": run_state.model,
                    "error": str(exc),
                }
            )
            run_state.step_index = next_step
            run_state.phase = "failed"
            run_state.stop_reason = "provider_error"
            run_state.error = str(exc)
            run_state.final_text = self._provider_error_to_user_text(str(exc))
            return run_state

        run_state.step_index = next_step
        run_state.trace.append(
            {
                "step": next_step,
                "stage": "provider_response",
                "provider": run_state.provider,
                "model": run_state.model,
                "response": provider_response,
            }
        )

        tool_call_payload = provider_response.get("tool_call")
        if tool_call_payload:
            tool_call = self._prepare_tool_call(
                tool_call_payload,
                run_state.trace,
                step=next_step,
            )
            run_state.working_messages.append(
                ChatMessage(
                    role="assistant",
                    content=self._normalize_final_text(provider_response.get("text", "")),
                    name=tool_call.tool_name,
                    tool_call_id=tool_call.tool_call_id,
                    tool_arguments=dict(tool_call.arguments),
                )
            )
            run_state.pending_tool_call = tool_call
            run_state.pending_tool_payload = tool_call_payload
            run_state.current_tool_name = tool_call.tool_name
            run_state.status_text = f"Running {tool_call.tool_name}"
            run_state.status_state = "running"
            run_state.phase = "tool_pending"
            return run_state

        run_state.final_text = self._normalize_final_text(provider_response.get("text", ""))
        run_state.status_text = ""
        run_state.phase = "completed"
        run_state.stop_reason = "completed"
        return run_state

    def _advance_tool_step(self, run_state: AgentRunState) -> AgentRunState:
        tool_call = run_state.pending_tool_call
        raw_tool_call = run_state.pending_tool_payload

        if tool_call is None:
            restored = self._restore_pending_tool_call_from_messages(run_state)
            if restored:
                tool_call = run_state.pending_tool_call
                raw_tool_call = run_state.pending_tool_payload

        if tool_call is None:
            run_state.phase = "provider_pending"
            run_state.current_tool_name = None
            run_state.pending_tool_payload = None
            return run_state

        tool_result = self._execute_parsed_tool(
            tool_call,
            run_state.trace,
            step=run_state.step_index,
            raw_tool_call=raw_tool_call,
            prior_tool_results=run_state.tool_results,
            max_tool_calls=run_state.max_tool_calls,
        )
        run_state.tool_results.append(tool_result)

        tool_message_content = self._tool_result_to_message_content(tool_result)
        run_state.working_messages.append(
            ChatMessage(
                role="tool",
                name=tool_result.tool_name,
                content=tool_message_content,
                tool_call_id=tool_call.tool_call_id,
            )
        )

        run_state.pending_tool_call = None
        run_state.pending_tool_payload = None
        run_state.current_tool_name = None
        run_state.status_text = ""
        run_state.phase = "provider_pending"
        return run_state

    def _execute_tool(
        self,
        tool_call_payload: dict[str, Any],
        trace: list[dict[str, Any]],
        *,
        step: int,
    ) -> ToolResult:
        """
        Execute a single tool call from a normalized provider response.

        Additional trace fields are recorded here to make provider/tool-call
        debugging easier in the Admin panel.
        """
        tool_call = self._prepare_tool_call(tool_call_payload, trace, step=step)
        return self._execute_parsed_tool(
            tool_call,
            trace,
            step=step,
            raw_tool_call=tool_call_payload,
            prior_tool_results=None,
            max_tool_calls=None,
        )

    def _record_agent_run(self, user_input: str, result: AgentResult) -> None:
        """
        Record per-session agent-run activity in Streamlit session state.

        This keeps higher-level run tracing separate from tool usage metrics so
        the Admin UI can later render step-by-step agent turns, provider/model
        information, and stop reasons without changing the core execution flow.
        """
        activity = st.session_state.setdefault(
            "agent_activity",
            {
                "total_runs": 0,
                "recent_runs": [],
            },
        )

        activity["total_runs"] += 1
        activity["recent_runs"].append(
            {
                "user_input": user_input,
                "stop_reason": result.stop_reason,
                "final_text": result.final_text,
                "blocks": result.blocks,
                "tool_count": len(result.tool_results),
                "trace": result.trace,
                "tool_results": [tool_result.__dict__ for tool_result in result.tool_results],
                "metadata": result.metadata,
            }
        )
        activity["recent_runs"] = activity["recent_runs"][-10:]

    def _prepare_tool_call(
        self,
        tool_call_payload: dict[str, Any],
        trace: list[dict[str, Any]],
        *,
        step: int,
    ) -> ToolCall:
        """
        Parse and trace a requested tool call before execution.
        """
        tool_call = self._parse_tool_call(tool_call_payload)
        trace.append(
            {
                "step": step,
                "stage": "tool_requested",
                "raw_tool_call": tool_call_payload,
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
            }
        )
        return tool_call

    def _execute_parsed_tool(
        self,
        tool_call: ToolCall,
        trace: list[dict[str, Any]],
        *,
        step: int,
        raw_tool_call: dict[str, Any] | None = None,
        prior_tool_results: list[ToolResult] | None = None,
        max_tool_calls: int | None = None,
    ) -> ToolResult:
        """
        Execute a previously parsed tool call and record the result trace.
        """
        execution_limit_error = self._check_execution_tool_limit(
            prior_tool_results=prior_tool_results or [],
            max_tool_calls=max_tool_calls,
        )
        if execution_limit_error is not None:
            result = ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output=None,
                error=execution_limit_error,
            )
            self._record_tool_usage(result)
            trace.append(
                {
                    "step": step,
                    "stage": "tool_result",
                    "raw_tool_call": raw_tool_call,
                    "tool_name": result.tool_name,
                    "arguments": tool_call.arguments,
                    "success": result.success,
                    "error": result.error,
                }
            )
            return result

        if not self.tool_registry.has(tool_call.tool_name):
            result = ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output=None,
                error=f"Tool '{tool_call.tool_name}' is not registered.",
            )
            trace.append(
                {
                    "step": step,
                    "stage": "tool_result",
                    "raw_tool_call": raw_tool_call,
                    "tool_name": result.tool_name,
                    "arguments": tool_call.arguments,
                    "success": result.success,
                    "error": result.error,
                }
            )
            return result

        tool = self.tool_registry.get(tool_call.tool_name)
        scope_limit_error = self._check_tool_scope_limit(
            tool_name=tool_call.tool_name,
            prior_tool_results=prior_tool_results or [],
        )
        if scope_limit_error is not None:
            result = ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output=None,
                error=scope_limit_error,
            )
            self._record_tool_usage(result)
            trace.append(
                {
                    "step": step,
                    "stage": "tool_result",
                    "raw_tool_call": raw_tool_call,
                    "tool_name": result.tool_name,
                    "arguments": tool_call.arguments,
                    "success": result.success,
                    "error": result.error,
                }
            )
            return result

        try:
            output = tool.run(**tool_call.arguments)
            result = ToolResult(
                tool_name=tool_call.tool_name,
                success=True,
                output=output,
            )
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output=None,
                error=str(exc),
            )

        self._record_tool_usage(result)
        trace.append(
            {
                "step": step,
                "stage": "tool_result",
                "raw_tool_call": raw_tool_call,
                "tool_name": result.tool_name,
                "arguments": tool_call.arguments,
                "success": result.success,
                "error": result.error,
                "output": result.output if result.success else None,
            }
        )
        return result

    def _record_tool_usage(self, tool_result: ToolResult) -> None:
        """
        Record per-session tool usage in Streamlit session state.

        This keeps the source of truth for tool execution in the runner, while
        allowing UI components such as an Admin tab to render usage statistics
        without owning backend counting logic.
        """
        usage = st.session_state.setdefault(
            "tool_usage",
            {
                "total_calls": 0,
                "by_tool": {},
                "recent_events": [],
            },
        )

        usage["total_calls"] += 1
        usage["by_tool"][tool_result.tool_name] = (
            usage["by_tool"].get(tool_result.tool_name, 0) + 1
        )

        usage["recent_events"].append(
            {
                "tool_name": tool_result.tool_name,
                "success": tool_result.success,
                "error": tool_result.error,
            }
        )

        usage["recent_events"] = usage["recent_events"][-10:]

    @staticmethod
    def _check_execution_tool_limit(
        *,
        prior_tool_results: list[ToolResult],
        max_tool_calls: int | None,
    ) -> str | None:
        if max_tool_calls is None:
            return None

        allowed_calls = max(0, int(max_tool_calls))
        if len(prior_tool_results) >= allowed_calls:
            return "Execution tool budget exhausted for this run."

        return None

    def _check_tool_scope_limit(
        self,
        *,
        tool_name: str,
        prior_tool_results: list[ToolResult],
    ) -> str | None:
        if not self.tool_scope_limits or not self.tool_registry.has(tool_name):
            return None

        tool = self.tool_registry.get(tool_name)
        tool_scope = str(getattr(tool, "scope", "")).strip().lower()
        if not tool_scope or tool_scope not in self.tool_scope_limits:
            return None

        allowed_calls = max(0, int(self.tool_scope_limits[tool_scope]))
        used_calls = 0
        for tool_result in prior_tool_results:
            prior_tool_name = str(getattr(tool_result, "tool_name", "") or "").strip()
            if not prior_tool_name or not self.tool_registry.has(prior_tool_name):
                continue
            prior_tool = self.tool_registry.get(prior_tool_name)
            if str(getattr(prior_tool, "scope", "")).strip().lower() == tool_scope:
                used_calls += 1

        if used_calls >= allowed_calls:
            return f"{tool_scope.capitalize()} tool budget exhausted for this run."

        return None

    def _restore_pending_tool_call_from_messages(self, run_state: AgentRunState) -> bool:
        if run_state.pending_tool_call is not None:
            return False

        unmatched_message = self._find_unmatched_tool_call_message(run_state.working_messages)
        if unmatched_message is None:
            return False

        tool_call_payload = {
            "tool_name": unmatched_message.name,
            "arguments": dict(unmatched_message.tool_arguments or {}),
            "tool_call_id": unmatched_message.tool_call_id,
        }
        tool_call = self._parse_tool_call(tool_call_payload)
        run_state.pending_tool_call = tool_call
        run_state.pending_tool_payload = tool_call_payload
        run_state.current_tool_name = tool_call.tool_name
        run_state.status_text = f"Running {tool_call.tool_name}"
        run_state.status_state = "running"
        run_state.phase = "tool_pending"
        return True

    @staticmethod
    def _find_unmatched_tool_call_message(working_messages: list[ChatMessage]) -> ChatMessage | None:
        satisfied_tool_call_ids: set[str] = set()

        for message in reversed(working_messages):
            tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
            if not tool_call_id:
                continue

            if message.role == "tool":
                satisfied_tool_call_ids.add(tool_call_id)
                continue

            if message.role == "assistant" and message.name and tool_call_id not in satisfied_tool_call_ids:
                return message

        return None

    @staticmethod
    def _parse_tool_call(tool_call_payload: dict[str, Any]) -> ToolCall:
        """
        Normalize a tool call payload into a ToolCall object.
        """
        tool_name = str(tool_call_payload.get("tool_name", "")).strip()
        arguments = tool_call_payload.get("arguments", {})
        tool_call_id = str(tool_call_payload.get("tool_call_id", "")).strip() or None

        if not isinstance(arguments, dict):
            arguments = {}

        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            tool_call_id=tool_call_id,
        )

    @staticmethod
    def _tool_result_to_message_content(tool_result: ToolResult) -> str:
        """
        Convert a tool result into a string suitable for a tool-role message.

        Structured outputs are serialized as formatted JSON so providers see a
        stable, markdown-friendly representation instead of Python repr text.
        Plain scalar outputs still pass through as simple strings.
        """
        if tool_result.success:
            return AgentRunner._serialize_message_content(tool_result.output)

        error_payload = {
            "tool_name": tool_result.tool_name,
            "success": tool_result.success,
            "error": tool_result.error,
        }
        return AgentRunner._serialize_message_content(error_payload)

    @staticmethod
    def _extract_response_blocks(tool_results: list[ToolResult]) -> list[dict[str, Any]]:
        """
        Extract renderable assistant response blocks from successful tool results.

        For v1, this supports at most one chart block per assistant response.
        """
        blocks: list[dict[str, Any]] = []

        for tool_result in tool_results:
            if not tool_result.success or not isinstance(tool_result.output, dict):
                continue

            rendering_payload = tool_result.output.get("tool_rendering")
            if not isinstance(rendering_payload, dict):
                continue

            block_type = str(rendering_payload.get("type", "")).strip().lower()
            if block_type != "chart":
                continue

            blocks.append(rendering_payload)
            break

        return blocks

    @staticmethod
    def _normalize_final_text(value: Any) -> str:
        """
        Normalize final provider text into a stable markdown-friendly string.
        """
        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        return AgentRunner._serialize_message_content(value).strip()

    @staticmethod
    def _serialize_message_content(value: Any) -> str:
        """
        Serialize message content into a stable text format.

        - strings stay as strings
        - scalars become simple strings
        - dict/list payloads become formatted JSON
        - non-JSON-serializable objects fall back to `str(...)`
        """
        if value is None:
            return "null"

        if isinstance(value, str):
            return value

        if isinstance(value, (int, float, bool)):
            return str(value)

        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, indent=2, ensure_ascii=False, default=str)
            except TypeError:
                return str(value)

        try:
            return json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)

    @staticmethod
    def _provider_error_to_user_text(error_message: str) -> str:
        normalized_error = str(error_message or "").strip().lower()
        if (
            "invalid_request_error" in normalized_error
            or "tool_calls" in normalized_error
            or "malformed" in normalized_error
        ):
            return (
                "The selected model provider rejected the request because the tool-call "
                "message history became invalid. Please retry or switch providers while "
                "this provider integration is corrected."
            )

        return (
            "The selected model provider is temporarily unavailable or exceeded its "
            "current quota. Please try again shortly, switch providers, or check "
            "the provider configuration and billing details."
        )
