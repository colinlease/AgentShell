from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


SUMMARY_SCHEMA_VERSION = 1
MAX_TOP_ERRORS = 10


def update_daily_summary(*, summary_path: Path, run_record: dict[str, Any]) -> None:
    """
    Incrementally update one daily metrics summary from a normalized run record.
    """
    summary = _load_summary(summary_path)
    _apply_run_record(summary, run_record)
    _write_summary(summary_path, summary)


def _load_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return _new_summary(date=str(summary_path.stem).replace(".summary", ""))

    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _new_summary(date=str(summary_path.stem).replace(".summary", ""))

    if not isinstance(loaded, dict):
        return _new_summary(date=str(summary_path.stem).replace(".summary", ""))

    return _ensure_summary_shape(loaded)


def _new_summary(*, date: str) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "date": date,
        "updated_at": "",
        "runs": {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "stopped": 0,
            "max_steps": 0,
            "provider_error": 0,
        },
        "runtime_features": {
            "planning_used": 0,
            "planning_deep": 0,
            "critique_used": 0,
            "reflection_used": 0,
            "reflection_forced": 0,
            "compaction_used": 0,
        },
        "providers": {},
        "performance": {
            "duration_ms_total": 0,
            "duration_ms_count": 0,
            "avg_duration_ms": 0.0,
            "provider_steps_total": 0,
            "provider_steps_count": 0,
            "avg_provider_steps": 0.0,
            "tool_calls_total": 0,
            "tool_calls_count": 0,
            "avg_tool_calls": 0.0,
            "compacted_message_count_total": 0,
            "compacted_message_count_count": 0,
            "avg_compacted_message_count": 0.0,
        },
        "tools": {
            "total_calls": 0,
            "total_failures": 0,
            "by_tool": {},
            "failures_by_tool": {},
            "error_counts": {},
            "top_errors": [],
        },
        "reflection": {
            "mutations_applied_total": 0,
            "note_files_touched": {},
        },
    }


def _ensure_summary_shape(summary: dict[str, Any]) -> dict[str, Any]:
    expected = _new_summary(date=str(summary.get("date", "")))
    merged = dict(expected)
    for key, value in summary.items():
        if key in {"runs", "runtime_features", "providers", "performance", "tools", "reflection"} and isinstance(value, dict):
            nested = dict(expected[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _apply_run_record(summary: dict[str, Any], run_record: dict[str, Any]) -> None:
    summary["updated_at"] = str(run_record.get("timestamp_finished", ""))

    runs = summary["runs"]
    outcome = dict(run_record.get("outcome", {}) or {})
    usage = dict(run_record.get("usage", {}) or {})
    orchestration = dict(run_record.get("orchestration", {}) or {})
    runtime = dict(run_record.get("runtime", {}) or {})

    runs["total"] = int(runs.get("total", 0)) + 1
    stop_reason = str(outcome.get("stop_reason", ""))
    if stop_reason == "completed":
        runs["completed"] = int(runs.get("completed", 0)) + 1
    elif stop_reason == "stopped":
        runs["stopped"] = int(runs.get("stopped", 0)) + 1
    else:
        runs["failed"] = int(runs.get("failed", 0)) + 1
    if stop_reason == "max_steps":
        runs["max_steps"] = int(runs.get("max_steps", 0)) + 1
    if stop_reason == "provider_error":
        runs["provider_error"] = int(runs.get("provider_error", 0)) + 1

    runtime_features = summary["runtime_features"]
    planning = dict(orchestration.get("planning", {}) or {})
    critique = dict(orchestration.get("critique", {}) or {})
    reflection_gate = dict(orchestration.get("reflection_gate", {}) or {})
    reflection = dict(orchestration.get("reflection", {}) or {})

    if bool(planning.get("used")):
        runtime_features["planning_used"] = int(runtime_features.get("planning_used", 0)) + 1
    if str(dict(orchestration.get("triage", {}) or {}).get("planning_mode", "")) == "deep":
        runtime_features["planning_deep"] = int(runtime_features.get("planning_deep", 0)) + 1
    if bool(critique.get("used")):
        runtime_features["critique_used"] = int(runtime_features.get("critique_used", 0)) + 1
    if bool(reflection.get("used")):
        runtime_features["reflection_used"] = int(runtime_features.get("reflection_used", 0)) + 1
    if bool(reflection_gate.get("forced")):
        runtime_features["reflection_forced"] = int(runtime_features.get("reflection_forced", 0)) + 1
    if bool(orchestration.get("used_compaction")):
        runtime_features["compaction_used"] = int(runtime_features.get("compaction_used", 0)) + 1

    provider_name = str(runtime.get("provider", "")).strip()
    model_name = str(runtime.get("model", "")).strip()
    provider_key = f"{provider_name}:{model_name}" if provider_name or model_name else "unknown"
    summary["providers"][provider_key] = int(summary["providers"].get(provider_key, 0)) + 1

    _increment_average(
        summary["performance"],
        value=int(run_record.get("duration_ms", 0) or 0),
        total_key="duration_ms_total",
        count_key="duration_ms_count",
        average_key="avg_duration_ms",
    )
    _increment_average(
        summary["performance"],
        value=int(usage.get("provider_steps", 0) or 0),
        total_key="provider_steps_total",
        count_key="provider_steps_count",
        average_key="avg_provider_steps",
    )
    _increment_average(
        summary["performance"],
        value=int(usage.get("tool_calls", 0) or 0),
        total_key="tool_calls_total",
        count_key="tool_calls_count",
        average_key="avg_tool_calls",
    )
    compacted_message_count = int(orchestration.get("compacted_message_count", 0) or 0)
    if compacted_message_count > 0:
        _increment_average(
            summary["performance"],
            value=compacted_message_count,
            total_key="compacted_message_count_total",
            count_key="compacted_message_count_count",
            average_key="avg_compacted_message_count",
        )

    tools = summary["tools"]
    tools["total_calls"] = int(tools.get("total_calls", 0)) + int(usage.get("tool_calls", 0) or 0)
    tools["total_failures"] = int(tools.get("total_failures", 0)) + int(usage.get("tool_failures", 0) or 0)

    for tool_name in run_record.get("tool_name_counts", {}) or {}:
        tools["by_tool"][tool_name] = int(tools["by_tool"].get(tool_name, 0)) + int(run_record["tool_name_counts"][tool_name])

    for failed_tool in list(usage.get("failed_tools", []) or []):
        tool_name = str(failed_tool.get("tool_name", "")).strip()
        error = str(failed_tool.get("error", "") or "").strip()
        count = int(failed_tool.get("count", 0) or 0)
        if not tool_name or count <= 0:
            continue

        tools["failures_by_tool"][tool_name] = int(tools["failures_by_tool"].get(tool_name, 0)) + count
        error_key = f"{tool_name}||{error}" if error else tool_name
        tools["error_counts"][error_key] = int(tools["error_counts"].get(error_key, 0)) + count

    tools["top_errors"] = _build_top_errors(tools["error_counts"])

    reflection_summary = summary["reflection"]
    reflection_summary["mutations_applied_total"] = (
        int(reflection_summary.get("mutations_applied_total", 0))
        + int(reflection.get("mutations_applied", 0) or 0)
    )
    for file_name in list(reflection.get("note_files_touched", []) or []):
        normalized_name = str(file_name).strip()
        if not normalized_name:
            continue
        reflection_summary["note_files_touched"][normalized_name] = (
            int(reflection_summary["note_files_touched"].get(normalized_name, 0)) + 1
        )


def _increment_average(
    performance: dict[str, Any],
    *,
    value: int,
    total_key: str,
    count_key: str,
    average_key: str,
) -> None:
    performance[total_key] = int(performance.get(total_key, 0)) + int(value)
    performance[count_key] = int(performance.get(count_key, 0)) + 1
    count = int(performance[count_key])
    performance[average_key] = round(int(performance[total_key]) / count, 2) if count else 0.0


def _build_top_errors(error_counts: dict[str, Any]) -> list[dict[str, Any]]:
    counter = Counter()
    for error_key, count in dict(error_counts or {}).items():
        counter[str(error_key)] = int(count or 0)

    top_errors: list[dict[str, Any]] = []
    for error_key, count in counter.most_common(MAX_TOP_ERRORS):
        tool_name, separator, error_text = error_key.partition("||")
        top_errors.append(
            {
                "tool_name": tool_name,
                "error": error_text if separator else "",
                "count": count,
            }
        )
    return top_errors


def _write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, sort_keys=True, ensure_ascii=True, indent=2)

    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=summary_path.parent) as temp_file:
        temp_file.write(payload)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)

    temp_path.replace(summary_path)
