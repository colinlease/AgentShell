from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.core.models import ToolResult


@dataclass
class AgentResult:
    """
    Structured return object for a completed agent run.
    """
    final_text: str
    blocks: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "completed"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)