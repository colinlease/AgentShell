from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agents.core.models import AgentRunState, ChatMessage, ToolResult


PlanningMode = Literal["skip", "light", "deep"]
OrchestrationPhase = Literal[
    "compaction_pending",
    "triage_pending",
    "planning_pending",
    "critique_pending",
    "execution_running",
    "reflection_pending",
    "completed",
    "failed",
]


@dataclass
class TriageDecision:
    """
    Minimal planning decision used by the orchestration wrapper.
    """

    should_plan: bool
    planning_mode: PlanningMode
    reason: str
    source: Literal["deterministic", "model", "fallback"] = "fallback"


@dataclass
class PlanningArtifact:
    """
    Compact plan output used to guide the execution runner.
    """

    summary: str
    execution_guidance: list[str] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)
    notes_context: list[dict[str, Any]] = field(default_factory=list)
    notes_tool_activity: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CritiqueArtifact:
    """
    Compact critique output that can refine execution guidance before execution.
    """

    summary: str
    issues: list[str] = field(default_factory=list)
    revised_execution_guidance: list[str] = field(default_factory=list)


@dataclass
class ReflectionDecision:
    """
    Deterministic reflection gate used after execution completes.
    """

    should_reflect: bool
    forced: bool
    reason: str
    source: Literal["deterministic", "fallback"] = "deterministic"


@dataclass
class ReflectionArtifact:
    """
    Compact reflection output plus bounded note-maintenance activity.
    """

    summary: str
    lessons: list[str] = field(default_factory=list)
    tool_activity: list[dict[str, Any]] = field(default_factory=list)
    note_files_touched: list[str] = field(default_factory=list)
    mutations_applied: int = 0


@dataclass
class ConversationContextArtifact:
    """
    Hidden shared conversation context used across orchestration phases.
    """

    summary: str | None = None
    recent_messages: list[ChatMessage] = field(default_factory=list)
    compacted_message_count: int = 0
    context_text: str = ""


@dataclass
class OrchestratedRunState:
    """
    Resumable outer runtime state for the orchestration wrapper.
    """

    run_id: str
    user_input: str
    request_messages: list[ChatMessage] = field(default_factory=list)
    request_context: dict[str, Any] = field(default_factory=dict)
    target_message_id: str | None = None
    surface: str | None = None
    provider: str | None = None
    model: str | None = None
    phase: OrchestrationPhase = "triage_pending"
    stop_reason: str = "in_progress"
    final_text: str = ""
    error: str | None = None
    status_text: str = ""
    status_state: str = "running"
    trace: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    triage: TriageDecision | None = None
    plan: PlanningArtifact | None = None
    critique: CritiqueArtifact | None = None
    reflection_decision: ReflectionDecision | None = None
    reflection: ReflectionArtifact | None = None
    conversation_context: ConversationContextArtifact | None = None
    execution_runtime: Any | None = None
    execution_state: AgentRunState | None = None
    execution_terminal_phase: Literal["completed", "failed"] | None = None
    stopped_reason: str | None = None
