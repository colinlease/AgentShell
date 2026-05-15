from __future__ import annotations

from dataclasses import fields
from typing import Any

from agents.core.runtime_config import RuntimeBudgetConfig, RuntimeGateConfig


DEFAULT_RUNTIME_BUDGETS = RuntimeBudgetConfig()
DEFAULT_RUNTIME_GATES = RuntimeGateConfig()


RUNTIME_LIMIT_GROUPS: list[dict[str, Any]] = [
    {
        "title": "Planning",
        "subtitle": "Limits used before execution begins.",
        "fields": [
            {
                "name": "triage_model_calls",
                "label": "Triage Model Calls",
                "min": 1,
                "max": 10,
                "step": 1,
                "help": "Maximum hidden model calls allowed for triage.",
            },
            {
                "name": "planning_model_calls",
                "label": "Planning Model Calls",
                "min": 1,
                "max": 12,
                "step": 1,
                "help": "Maximum hidden model calls allowed during planning.",
            },
            {
                "name": "planning_note_reads",
                "label": "Planning Note Reads",
                "min": 0,
                "max": 20,
                "step": 1,
                "help": "Maximum runtime-note reads allowed during planning.",
            },
            {
                "name": "critique_model_calls",
                "label": "Critique Model Calls",
                "min": 1,
                "max": 10,
                "step": 1,
                "help": "Maximum hidden model calls allowed for critique after deep planning.",
            },
        ],
    },
    {
        "title": "Execution",
        "subtitle": "Per-run limits while the assistant is actively solving the task.",
        "fields": [
            {
                "name": "execution_provider_turns",
                "label": "Provider Turns",
                "min": 1,
                "max": 50,
                "step": 1,
                "help": "Maximum provider turns allowed during the visible execution run.",
            },
            {
                "name": "execution_tool_calls",
                "label": "Tool Calls",
                "min": 0,
                "max": 50,
                "step": 1,
                "help": "Maximum execution-phase tool calls across framework, app, and runtime-note tools.",
            },
            {
                "name": "execution_note_reads",
                "label": "Note Reads",
                "min": 0,
                "max": 20,
                "step": 1,
                "help": "Maximum runtime-note reads available during execution.",
            },
        ],
    },
    {
        "title": "Reflection",
        "subtitle": "Post-run reflection limits, including how many notes reflection may update or delete.",
        "fields": [
            {
                "name": "reflection_model_calls",
                "label": "Reflection Model Calls",
                "min": 1,
                "max": 20,
                "step": 1,
                "help": "Maximum hidden model calls allowed during reflection.",
            },
            {
                "name": "reflection_tool_calls",
                "label": "Reflection Tool Calls",
                "min": 1,
                "max": 20,
                "step": 1,
                "help": "Maximum note-tool calls allowed during reflection.",
            },
            {
                "name": "note_writes",
                "label": "Reflection Note Writes",
                "min": 1,
                "max": 20,
                "step": 1,
                "help": "Maximum notes reflection may create or update in one run.",
            },
            {
                "name": "note_deletes",
                "label": "Reflection Note Deletes",
                "min": 1,
                "max": 20,
                "step": 1,
                "help": "Maximum notes reflection may delete in one run.",
            },
        ],
    },
]


COMPACTION_LIMIT_GROUP: dict[str, Any] = {
    "title": "Compaction Limits",
    "subtitle": "Additional compaction constraints beyond the thresholds shown above.",
    "fields": [
        {
            "name": "compaction_max_model_calls",
            "label": "Compaction Model Calls",
            "min": 1,
            "max": 10,
            "step": 1,
            "help": "Maximum hidden model calls allowed during compaction.",
        },
        {
            "name": "compaction_char_trigger",
            "label": "Compaction Char Trigger",
            "min": 1000,
            "max": 120000,
            "step": 1000,
            "help": "Optional character-count threshold that can trigger compaction logic.",
        },
    ],
}


COMPACTION_CONTROL_FIELDS: list[dict[str, Any]] = [
    {
        "name": "compaction_message_trigger",
        "label": "Trigger After Messages",
        "min": 6,
        "max": 100,
        "step": 1,
        "default": 10,
    },
    {
        "name": "compaction_keep_recent_messages",
        "label": "Keep Recent Raw Messages",
        "min": 1,
        "max": 20,
        "step": 1,
        "default": 5,
    },
    {
        "name": "compaction_max_summary_chars",
        "label": "Max Summary Chars",
        "min": 200,
        "max": 6000,
        "step": 100,
        "default": 2000,
    },
]


REFLECTION_GATE_FIELDS: list[dict[str, Any]] = [
    {
        "name": "reflection_tool_use_threshold",
        "label": "Reflect After Tool Calls",
        "min": 0,
        "max": 50,
        "step": 1,
        "default": DEFAULT_RUNTIME_GATES.reflection_tool_use_threshold,
        "help": "Force reflection only when a completed run uses more than this many tools.",
    },
]


def iter_agent_runtime_budget_fields() -> list[dict[str, Any]]:
    """
    Return the runtime budget fields exposed in the Admin Agent subtab.
    """
    fields_metadata: list[dict[str, Any]] = []
    for group in [*RUNTIME_LIMIT_GROUPS, COMPACTION_LIMIT_GROUP]:
        fields_metadata.extend(dict(field) for field in group.get("fields", []) or [])
    fields_metadata.extend(dict(field) for field in COMPACTION_CONTROL_FIELDS)
    return fields_metadata


def iter_agent_runtime_gate_fields() -> list[dict[str, Any]]:
    """
    Return deterministic gate fields exposed in the Admin Agent subtab.
    """
    return [dict(field) for field in REFLECTION_GATE_FIELDS]


def get_agent_runtime_budget_field_names() -> set[str]:
    """
    Return the persisted budget names supported by the Admin Agent subtab.
    """
    runtime_budget_names = {field.name for field in fields(RuntimeBudgetConfig)}
    return {
        str(field.get("name"))
        for field in iter_agent_runtime_budget_fields()
        if str(field.get("name")) in runtime_budget_names
    }


def get_agent_runtime_gate_field_names() -> set[str]:
    """
    Return the persisted gate setting names supported by the Admin Agent subtab.
    """
    runtime_gate_names = {field.name for field in fields(RuntimeGateConfig)}
    return {
        str(field.get("name"))
        for field in iter_agent_runtime_gate_fields()
        if str(field.get("name")) in runtime_gate_names
    }


def get_agent_runtime_budget_bounds() -> dict[str, tuple[int, int]]:
    """
    Return inclusive integer bounds for persisted Admin Agent budget fields.
    """
    return {
        str(field["name"]): (int(field["min"]), int(field["max"]))
        for field in iter_agent_runtime_budget_fields()
        if "name" in field and "min" in field and "max" in field
    }


def get_agent_runtime_gate_bounds() -> dict[str, tuple[int, int]]:
    """
    Return inclusive integer bounds for persisted Admin Agent gate settings.
    """
    return {
        str(field["name"]): (int(field["min"]), int(field["max"]))
        for field in iter_agent_runtime_gate_fields()
        if "name" in field and "min" in field and "max" in field
    }
