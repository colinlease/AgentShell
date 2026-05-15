from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from agents.core.results import AgentResult
from agents.core.runtime_config import RuntimeBudgetConfig
from agents.observability.metrics_aggregator import update_daily_summary
from app.components.workspace_host import get_workspace_host_snapshot
from app.state.agent_runtime_state import get_runtime_budget_overrides, get_runtime_state_snapshot
from config.settings import PROJECT_ROOT


RUN_LOG_SCHEMA_VERSION = 1
RUN_LOG_ROOT = PROJECT_ROOT / "logs"
RUN_LOG_SESSION_KEY = "persistent_run_log_session_id"
MAX_TEXT_PREVIEW_CHARS = 600
MAX_TRACE_STRING_CHARS = 1200
MAX_TRACE_ITEMS = 25
MAX_TRACE_DEPTH = 6


def persist_completed_run_log(*, user_input: str, result: AgentResult) -> None:
    """
    Persist one completed run to the daily JSONL log and update the daily summary.

    This function is best-effort and should be called from a protective try/except
    so any logging failure cannot affect the user-visible run.
    """
    if not _is_file_logging_enabled(result):
        return

    finished_at = datetime.now(timezone.utc)
    run_record = build_run_record(
        user_input=user_input,
        result=result,
        finished_at=finished_at,
    )
    run_log_path = get_daily_run_log_path(finished_at=finished_at)
    summary_path = get_daily_summary_path(finished_at=finished_at)

    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with run_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(run_record, sort_keys=True, ensure_ascii=True))
        log_file.write("\n")

    update_daily_summary(summary_path=summary_path, run_record=run_record)


def build_run_record(
    *,
    user_input: str,
    result: AgentResult,
    finished_at: datetime,
) -> dict[str, Any]:
    """
    Normalize one finalized AgentResult into a compact JSON-safe run record.
    """
    metadata = dict(result.metadata or {})
    started_at_iso = str(metadata.get("started_at_utc", "") or "")
    started_at_ms = int(metadata.get("started_at_ms", 0) or 0)
    finished_at_iso = finished_at.astimezone(timezone.utc).isoformat()
    finished_at_ms = int(finished_at.timestamp() * 1000)
    duration_ms = max(0, finished_at_ms - started_at_ms) if started_at_ms > 0 else 0

    workspace_snapshot = get_workspace_host_snapshot()
    runtime_snapshot = _build_runtime_snapshot(metadata)
    orchestration_summary = _build_orchestration_summary(result.trace)
    usage_summary, tool_name_counts = _build_usage_summary(result)
    trace_summary = _build_trace_summary(result.trace)
    stopped = str(result.stop_reason or "") == "stopped"
    stopped_reason = str(metadata.get("stopped_reason", "") or "").strip() or None

    run_record = {
        "schema_version": RUN_LOG_SCHEMA_VERSION,
        "run_id": str(metadata.get("run_id", "") or ""),
        "session_id": str(metadata.get("session_id", "") or _get_persistent_session_id()),
        "timestamp_started": started_at_iso,
        "timestamp_finished": finished_at_iso,
        "duration_ms": duration_ms,
        "surface": str(metadata.get("surface", "") or "chat"),
        "workspace_app": {
            "app_id": workspace_snapshot.get("active_workspace_app_id"),
            "app_label": workspace_snapshot.get("active_workspace_app_label"),
            "app_type": workspace_snapshot.get("active_workspace_app_type"),
        },
        "user": {
            "input_preview": _trim_string(user_input, max_chars=MAX_TEXT_PREVIEW_CHARS),
            "input_chars": len(str(user_input or "")),
        },
        "runtime": runtime_snapshot,
        "outcome": {
            "stop_reason": str(result.stop_reason or ""),
            "stopped": stopped,
            "stopped_reason": stopped_reason if stopped else None,
            "error": str(result.error) if result.error else None,
            "final_text_preview": _trim_string(result.final_text, max_chars=MAX_TEXT_PREVIEW_CHARS),
            "final_text_chars": len(str(result.final_text or "")),
            "blocks_count": len(list(result.blocks or [])),
        },
        "usage": usage_summary,
        "orchestration": orchestration_summary,
        "trace_summary": trace_summary,
        "metadata": _trim_value(metadata),
        "trace": [_trim_value(item) for item in list(result.trace or [])],
        "tool_results": [_summarize_tool_result(item) for item in list(result.tool_results or [])],
        "tool_name_counts": dict(tool_name_counts),
    }
    return run_record


def get_daily_run_log_path(*, finished_at: datetime) -> Path:
    local_date = finished_at.astimezone().date().isoformat()
    return RUN_LOG_ROOT / "agent_runs" / f"{local_date}.jsonl"


def get_daily_summary_path(*, finished_at: datetime) -> Path:
    local_date = finished_at.astimezone().date().isoformat()
    return RUN_LOG_ROOT / "agent_metrics" / f"{local_date}.summary.json"


def _is_file_logging_enabled(result: AgentResult) -> bool:
    return bool(dict(result.metadata or {}).get("logging_enabled", False))


def _build_runtime_snapshot(metadata: dict[str, Any]) -> dict[str, Any]:
    runtime_state = get_runtime_state_snapshot()
    budgets = _build_budget_snapshot()
    return {
        "provider": str(metadata.get("provider", "") or ""),
        "model": str(metadata.get("model", "") or ""),
        "feature_flags": {
            "planning_enabled": bool(runtime_state.get("planning_enabled", False)),
            "reflection_enabled": bool(runtime_state.get("reflection_enabled", False)),
            "compaction_enabled": bool(runtime_state.get("compaction_enabled", False)),
        },
        "budgets": budgets,
    }


def _build_budget_snapshot() -> dict[str, Any]:
    defaults = asdict(RuntimeBudgetConfig())
    overrides = get_runtime_budget_overrides()
    for key, value in dict(overrides or {}).items():
        if key in defaults:
            defaults[key] = int(value)
    return defaults


def _build_orchestration_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    latest_by_stage: dict[str, dict[str, Any]] = {}
    for item in list(trace or []):
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "")).strip()
        if not stage:
            continue
        latest_by_stage[stage] = item

    compaction_item = latest_by_stage.get("compaction", {})
    compaction_payload = dict(compaction_item.get("compaction", {}) or {})
    triage_item = latest_by_stage.get("triage", {})
    planning_item = latest_by_stage.get("planning", {})
    plan_payload = dict(planning_item.get("plan", {}) or {})
    critique_item = latest_by_stage.get("critique", {})
    critique_payload = dict(critique_item.get("critique", {}) or {})
    reflection_gate_item = latest_by_stage.get("reflection_gate", {})
    reflection_gate_payload = dict(reflection_gate_item.get("decision", {}) or {})
    reflection_item = latest_by_stage.get("reflection", {})
    reflection_payload = dict(reflection_item.get("reflection", {}) or {})

    return {
        "used_compaction": "compaction" in latest_by_stage,
        "compacted_message_count": int(compaction_payload.get("compacted_message_count", 0) or 0),
        "compaction_summary_chars": int(compaction_payload.get("summary_chars", 0) or 0),
        "triage": _trim_value(dict(triage_item.get("decision", {}) or {})),
        "planning": {
            "used": "planning" in latest_by_stage,
            "summary": _trim_string(str(plan_payload.get("summary", "") or ""), max_chars=MAX_TEXT_PREVIEW_CHARS),
            "guidance_count": len(list(plan_payload.get("execution_guidance", []) or [])),
            "missing_context_count": len(list(plan_payload.get("missing_context", []) or [])),
            "notes_context_count": len(list(plan_payload.get("notes_context", []) or [])),
            "notes_tool_activity_count": len(list(plan_payload.get("notes_tool_activity", []) or [])),
        },
        "critique": {
            "used": "critique" in latest_by_stage,
            "summary": _trim_string(str(critique_payload.get("summary", "") or ""), max_chars=MAX_TEXT_PREVIEW_CHARS),
            "issues_count": len(list(critique_payload.get("issues", []) or [])),
        },
        "reflection_gate": _trim_value(reflection_gate_payload),
        "reflection": {
            "used": "reflection" in latest_by_stage,
            "summary": _trim_string(str(reflection_payload.get("summary", "") or ""), max_chars=MAX_TEXT_PREVIEW_CHARS),
            "lessons_count": len(list(reflection_payload.get("lessons", []) or [])),
            "tool_activity_count": len(list(reflection_payload.get("tool_activity", []) or [])),
            "note_files_touched": [str(item) for item in list(reflection_payload.get("note_files_touched", []) or [])],
            "mutations_applied": int(reflection_payload.get("mutations_applied", 0) or 0),
        },
    }


def _build_usage_summary(result: AgentResult) -> tuple[dict[str, Any], Counter[str]]:
    tool_name_counts: Counter[str] = Counter()
    failed_tools: Counter[tuple[str, str]] = Counter()
    tool_results = list(result.tool_results or [])

    for tool_result in tool_results:
        tool_name = str(getattr(tool_result, "tool_name", "") or "").strip()
        if not tool_name:
            continue
        tool_name_counts[tool_name] += 1
        if not bool(getattr(tool_result, "success", False)):
            failed_tools[(tool_name, str(getattr(tool_result, "error", "") or ""))] += 1

    failed_tool_rows = [
        {
            "tool_name": tool_name,
            "error": error_text,
            "count": count,
        }
        for (tool_name, error_text), count in failed_tools.most_common()
    ]

    provider_steps = int(dict(result.metadata or {}).get("steps_used", 0) or 0)
    return (
        {
            "provider_steps": provider_steps,
            "tool_calls": len(tool_results),
            "tool_failures": sum(1 for item in tool_results if not bool(getattr(item, "success", False))),
            "tools_used": list(tool_name_counts.keys()),
            "failed_tools": failed_tool_rows,
        },
        tool_name_counts,
    )


def _build_trace_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts: Counter[str] = Counter()
    for item in list(trace or []):
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "")).strip()
        if stage:
            stage_counts[stage] += 1
    return {"stage_counts": dict(stage_counts)}


def _summarize_tool_result(tool_result: Any) -> dict[str, Any]:
    tool_name = str(getattr(tool_result, "tool_name", "") or "").strip()
    success = bool(getattr(tool_result, "success", False))
    error = str(getattr(tool_result, "error", "") or "") or None

    record = {
        "tool_name": tool_name,
        "success": success,
        "error": error,
    }
    if success:
        output_text = _serialize_preview_value(getattr(tool_result, "output", None))
        record["output_preview"] = _trim_string(output_text, max_chars=MAX_TEXT_PREVIEW_CHARS)
        record["output_chars"] = len(output_text)
    return record


def _serialize_preview_value(value: Any) -> str:
    normalized = _trim_value(value)
    if isinstance(normalized, str):
        return normalized
    try:
        return json.dumps(normalized, sort_keys=True, ensure_ascii=True)
    except TypeError:
        return repr(normalized)


def _trim_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _trim_string(value, max_chars=MAX_TRACE_STRING_CHARS)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _trim_value(asdict(value), depth=depth)
    if isinstance(value, dict):
        if depth >= MAX_TRACE_DEPTH:
            return {
                "trimmed": True,
                "type": "dict",
                "original_items": len(value),
            }
        trimmed_dict: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_TRACE_ITEMS:
                trimmed_dict["__truncated__"] = {
                    "trimmed": True,
                    "original_items": len(value),
                }
                break
            trimmed_dict[str(key)] = _trim_value(item, depth=depth + 1)
        return trimmed_dict
    if isinstance(value, (list, tuple, set)):
        normalized_items = list(value)
        if depth >= MAX_TRACE_DEPTH:
            return {
                "trimmed": True,
                "type": "list",
                "original_items": len(normalized_items),
            }
        trimmed_items = [
            _trim_value(item, depth=depth + 1)
            for item in normalized_items[:MAX_TRACE_ITEMS]
        ]
        if len(normalized_items) > MAX_TRACE_ITEMS:
            trimmed_items.append(
                {
                    "trimmed": True,
                    "original_items": len(normalized_items),
                }
            )
        return trimmed_items
    return _trim_string(repr(value), max_chars=MAX_TRACE_STRING_CHARS)


def _trim_string(value: str, *, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [trimmed {len(text) - max_chars} chars]"


def _get_persistent_session_id() -> str:
    session_id = str(st.session_state.get(RUN_LOG_SESSION_KEY, "") or "").strip()
    if session_id:
        return session_id

    session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    st.session_state[RUN_LOG_SESSION_KEY] = session_id
    return session_id
