from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from agents.core.models import ChatMessage
from agents.providers.base import BaseLLMProvider
from agents.tools.registry import ToolRegistry


@dataclass
class ToolDialogueStep:
    """
    One provider-requested tool interaction inside a hidden orchestration phase.
    """

    tool_name: str
    arguments: dict[str, Any]
    tool_output: dict[str, Any]
    response_text: str
    tool_call_id: str | None = None
    consumed_tool_budget: bool = False


@dataclass
class ToolDialogueResult:
    """
    Result of a bounded provider/tool dialogue for one hidden phase.
    """

    final_text: str
    working_messages: list[ChatMessage] = field(default_factory=list)
    steps: list[ToolDialogueStep] = field(default_factory=list)
    remaining_model_calls: int = 0
    remaining_tool_calls: int = 0


def run_bounded_tool_dialogue(
    *,
    provider: BaseLLMProvider,
    working_messages: list[ChatMessage],
    system_prompt: str,
    tool_registry: ToolRegistry | None,
    max_model_calls: int,
    max_tool_calls: int,
    execute_tool_call: Callable[[str, dict[str, Any]], tuple[dict[str, Any], bool]],
) -> ToolDialogueResult:
    """
    Run a bounded provider/tool loop while preserving structured tool-call history.

    The caller keeps ownership of phase-specific budgets, tool registries, and
    output interpretation. This helper only standardizes how provider-requested
    tool calls are threaded back into message history.
    """

    messages = list(working_messages)
    steps: list[ToolDialogueStep] = []
    remaining_model_calls = int(max_model_calls)
    if remaining_model_calls < 1:
        raise ValueError("max_model_calls must be at least 1.")
    remaining_tool_calls = max(0, int(max_tool_calls))

    while remaining_model_calls > 0:
        remaining_model_calls -= 1
        response = provider.generate(
            messages=messages,
            system_prompt=system_prompt,
            tools=tool_registry.list_tool_schemas() if tool_registry is not None else None,
        )

        response_text = str(response.get("text", "") or "").strip()
        tool_call = response.get("tool_call")
        if tool_call and tool_registry is not None and remaining_tool_calls > 0:
            tool_name = str(tool_call.get("tool_name", "")).strip() or "unknown"
            arguments = tool_call.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            tool_call_id = str(tool_call.get("tool_call_id", "") or "").strip() or None

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response_text,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    tool_arguments=dict(arguments),
                )
            )

            tool_output, consumed_tool_budget = execute_tool_call(tool_name, dict(arguments))
            if consumed_tool_budget:
                remaining_tool_calls -= 1

            messages.append(
                ChatMessage(
                    role="tool",
                    name=tool_name,
                    content=json.dumps(tool_output, indent=2, ensure_ascii=False, default=str),
                    tool_call_id=tool_call_id,
                )
            )
            steps.append(
                ToolDialogueStep(
                    tool_name=tool_name,
                    arguments=dict(arguments),
                    tool_output=tool_output,
                    response_text=response_text,
                    tool_call_id=tool_call_id,
                    consumed_tool_budget=consumed_tool_budget,
                )
            )
            continue

        return ToolDialogueResult(
            final_text=response_text,
            working_messages=messages,
            steps=steps,
            remaining_model_calls=remaining_model_calls,
            remaining_tool_calls=remaining_tool_calls,
        )

    return ToolDialogueResult(
        final_text="",
        working_messages=messages,
        steps=steps,
        remaining_model_calls=remaining_model_calls,
        remaining_tool_calls=remaining_tool_calls,
    )
