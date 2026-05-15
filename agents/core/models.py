

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    """
    Standard internal message format used across providers and agent logic.
    """
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_arguments: dict[str, Any] | None = None


@dataclass
class ToolCall:
    """
    Represents a request from the model to execute a named tool.
    """
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None


@dataclass
class ToolResult:
    """
    Represents the result returned by a tool execution.
    """
    tool_name: str
    success: bool
    output: Any
    error: str | None = None


@dataclass
class AgentRequest:
    """
    Container for a single agent run request.
    """
    user_input: str
    messages: list[ChatMessage] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunState:
    """
    Mutable state for a resumable agent run across Streamlit reruns.
    """
    run_id: str
    user_input: str
    working_messages: list[ChatMessage] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    step_index: int = 0
    max_steps: int = 15
    max_tool_calls: int | None = None
    phase: str = "provider_pending"
    stop_reason: str = "in_progress"
    final_text: str = ""
    error: str | None = None
    target_message_id: str | None = None
    surface: str | None = None
    provider: str | None = None
    model: str | None = None
    pending_tool_call: ToolCall | None = None
    pending_tool_payload: dict[str, Any] | None = None
    current_tool_name: str | None = None
    status_text: str = ""
    status_state: str = "running"
    stopped_reason: str | None = None
