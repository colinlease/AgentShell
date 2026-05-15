from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


STRICT_POSITIVE_BUDGET_DISABLE_HINTS: dict[str, str] = {
    "triage_model_calls": "Disable planning with planning_enabled=False instead.",
    "planning_model_calls": "Disable planning with planning_enabled=False instead.",
    "critique_model_calls": "Disable planning with planning_enabled=False instead.",
    "reflection_model_calls": "Disable reflection with reflection_enabled=False instead.",
    "reflection_tool_calls": "Disable reflection with reflection_enabled=False instead.",
    "note_writes": "Disable reflection with reflection_enabled=False instead.",
    "note_deletes": "Disable reflection with reflection_enabled=False instead.",
    "compaction_max_model_calls": "Disable compaction with compaction_enabled=False instead.",
    "compaction_message_trigger": "Disable compaction with compaction_enabled=False instead.",
    "compaction_keep_recent_messages": "Disable compaction with compaction_enabled=False instead.",
    "compaction_max_summary_chars": "Disable compaction with compaction_enabled=False instead.",
    "compaction_char_trigger": "Disable compaction with compaction_enabled=False instead.",
}


def validate_runtime_budget_value(field_name: str, value: object) -> None:
    """
    Validate a single runtime budget value whose zero value would partially
    disable one of the major hidden runtime modes.
    """
    hint = STRICT_POSITIVE_BUDGET_DISABLE_HINTS.get(str(field_name))
    if hint is None:
        return

    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer. {hint}") from exc

    if int_value < 1:
        raise ValueError(f"{field_name} must be at least 1. {hint}")


@dataclass(frozen=True)
class RuntimeFeatureFlags:
    """
    Feature toggles for optional orchestration capabilities.

    All flags default to disabled so the application can preserve the current
    agent behavior unless features are explicitly enabled later.
    """

    planning_enabled: bool = False
    reflection_enabled: bool = False
    compaction_enabled: bool = False


@dataclass(frozen=True)
class RuntimeBudgetConfig:
    """
    Tunable per-phase runtime limits for future orchestration work.
    """

    triage_model_calls: int = 1
    planning_model_calls: int = 2
    planning_note_reads: int = 3
    critique_model_calls: int = 1
    execution_provider_turns: int = 15
    execution_tool_calls: int = 15
    execution_note_reads: int = 2
    reflection_model_calls: int = 8
    reflection_tool_calls: int = 8
    note_reads: int = 4
    note_writes: int = 4
    note_deletes: int = 2
    preflight_actions: int = 3
    compaction_max_model_calls: int = 1
    compaction_message_trigger: int = 10
    compaction_keep_recent_messages: int = 5
    compaction_max_summary_chars: int = 2000
    compaction_char_trigger: int = 24000

    def __post_init__(self) -> None:
        for field_name in STRICT_POSITIVE_BUDGET_DISABLE_HINTS:
            validate_runtime_budget_value(field_name, getattr(self, field_name))


@dataclass(frozen=True)
class RuntimeGateConfig:
    """
    Tunable deterministic thresholds that decide whether optional phases run.
    """

    reflection_tool_use_threshold: int = 4


@dataclass(frozen=True)
class RuntimePaths:
    """
    Filesystem locations used by future runtime extensions.
    """

    notes_root: Path = field(default_factory=lambda: Path("runtime_notes"))


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """
    Aggregate configuration for the extended agent runtime.
    """

    features: RuntimeFeatureFlags = field(default_factory=RuntimeFeatureFlags)
    budgets: RuntimeBudgetConfig = field(default_factory=RuntimeBudgetConfig)
    gates: RuntimeGateConfig = field(default_factory=RuntimeGateConfig)
    paths: RuntimePaths = field(default_factory=RuntimePaths)
