from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable


DATE_RANGE_TODAY = "Today"
DATE_RANGE_LAST_7_DAYS = "Last 7 days"
DATE_RANGE_THIS_WEEK = "This week"
DATE_RANGE_THIS_MONTH = "This month"
DATE_RANGE_ALL_TIME = "All time"
DATE_RANGE_OPTIONS = (
    DATE_RANGE_TODAY,
    DATE_RANGE_LAST_7_DAYS,
    DATE_RANGE_THIS_WEEK,
    DATE_RANGE_THIS_MONTH,
    DATE_RANGE_ALL_TIME,
)


@dataclass(frozen=True)
class RateRow:
    label: str
    count: int
    rate: float | None


@dataclass(frozen=True)
class ToolIssueRow:
    tool_name: str
    call_count: int
    issue_count: int
    issue_rate: float | None


@dataclass(frozen=True)
class PerformanceReport:
    date_range: str
    start_date: date | None
    end_date: date | None
    files_loaded: int
    latest_updated_at: str
    total_runs: int
    completed_runs: int
    failed_runs: int
    stopped_runs: int
    provider_errors: int
    max_steps: int
    duration_seconds_total: float
    avg_duration_seconds: float | None
    total_tool_calls: int
    avg_tools_per_run: float | None
    tool_issue_count: int
    tool_issue_rate: float | None
    tool_success_rate: float | None
    planning_used: int
    planning_deep: int
    planning_light: int
    planning_skipped: int
    reflection_used: int
    reflection_forced: int
    compaction_used: int
    run_outcome_rows: list[RateRow] = field(default_factory=list)
    planning_rows: list[RateRow] = field(default_factory=list)
    tool_issue_rows: list[ToolIssueRow] = field(default_factory=list)


def build_performance_report(
    *,
    summary_dir: Path,
    date_range: str,
    registered_tool_names: Iterable[str] = (),
    today: date | None = None,
) -> PerformanceReport:
    """
    Build a summary-only Admin performance report from daily metrics files.
    """
    current_date = today or date.today()
    normalized_range = _normalize_date_range(date_range)
    start_date, end_date = resolve_date_range(normalized_range, today=current_date)
    summaries = _load_summaries(summary_dir=summary_dir, start_date=start_date, end_date=end_date)

    total_runs = _sum_nested_int(summaries, "runs", "total")
    completed_runs = _sum_nested_int(summaries, "runs", "completed")
    failed_runs = _sum_nested_int(summaries, "runs", "failed")
    stopped_runs = _sum_nested_int(summaries, "runs", "stopped")
    provider_errors = _sum_nested_int(summaries, "runs", "provider_error")
    max_steps = _sum_nested_int(summaries, "runs", "max_steps")
    duration_ms_total = _sum_nested_int(summaries, "performance", "duration_ms_total")
    duration_seconds_total = round(duration_ms_total / 1000, 3)
    total_tool_calls = _sum_nested_int(summaries, "tools", "total_calls")
    total_tool_failures = _sum_nested_int(summaries, "tools", "total_failures")

    planning_used = _sum_nested_int(summaries, "runtime_features", "planning_used")
    planning_deep = _sum_nested_int(summaries, "runtime_features", "planning_deep")
    planning_light = max(0, planning_used - planning_deep)
    planning_skipped = max(0, total_runs - planning_used)
    reflection_used = _sum_nested_int(summaries, "runtime_features", "reflection_used")
    reflection_forced = _sum_nested_int(summaries, "runtime_features", "reflection_forced")
    compaction_used = _sum_nested_int(summaries, "runtime_features", "compaction_used")

    tool_calls_by_name = _sum_counter(summaries, section="tools", key="by_tool")
    tool_failures_by_name = _sum_counter(summaries, section="tools", key="failures_by_tool")
    tool_errors_by_name = _sum_error_counts_by_tool(summaries)
    total_logged_errors = sum(tool_errors_by_name.values())
    tool_issue_count = max(total_tool_failures, total_logged_errors)

    latest_updated_at = max(
        (str(summary.get("updated_at") or "") for summary in summaries),
        default="",
    )

    run_outcome_rows = [
        RateRow("Completed", completed_runs, _rate(completed_runs, total_runs)),
        RateRow("Failed", failed_runs, _rate(failed_runs, total_runs)),
        RateRow("Stopped", stopped_runs, _rate(stopped_runs, total_runs)),
        RateRow("Provider Errors", provider_errors, _rate(provider_errors, total_runs)),
        RateRow("Max Steps", max_steps, _rate(max_steps, total_runs)),
    ]
    planning_rows = [
        RateRow("Skip Planning", planning_skipped, _rate(planning_skipped, total_runs)),
        RateRow("Light Planning", planning_light, _rate(planning_light, total_runs)),
        RateRow("Deep Planning", planning_deep, _rate(planning_deep, total_runs)),
        RateRow("Reflection", reflection_used, _rate(reflection_used, total_runs)),
        RateRow("Compaction", compaction_used, _rate(compaction_used, total_runs)),
    ]

    tool_issue_rows = _build_tool_issue_rows(
        registered_tool_names=registered_tool_names,
        tool_calls_by_name=tool_calls_by_name,
        tool_failures_by_name=tool_failures_by_name,
        tool_errors_by_name=tool_errors_by_name,
    )

    return PerformanceReport(
        date_range=normalized_range,
        start_date=start_date,
        end_date=end_date,
        files_loaded=len(summaries),
        latest_updated_at=latest_updated_at,
        total_runs=total_runs,
        completed_runs=completed_runs,
        failed_runs=failed_runs,
        stopped_runs=stopped_runs,
        provider_errors=provider_errors,
        max_steps=max_steps,
        duration_seconds_total=duration_seconds_total,
        avg_duration_seconds=_rate(duration_seconds_total, total_runs),
        total_tool_calls=total_tool_calls,
        avg_tools_per_run=_rate(total_tool_calls, total_runs),
        tool_issue_count=tool_issue_count,
        tool_issue_rate=_rate(tool_issue_count, total_tool_calls),
        tool_success_rate=_rate(total_tool_calls - tool_issue_count, total_tool_calls),
        planning_used=planning_used,
        planning_deep=planning_deep,
        planning_light=planning_light,
        planning_skipped=planning_skipped,
        reflection_used=reflection_used,
        reflection_forced=reflection_forced,
        compaction_used=compaction_used,
        run_outcome_rows=run_outcome_rows,
        planning_rows=planning_rows,
        tool_issue_rows=tool_issue_rows,
    )


def resolve_date_range(date_range: str, *, today: date) -> tuple[date | None, date | None]:
    normalized_range = _normalize_date_range(date_range)
    if normalized_range == DATE_RANGE_TODAY:
        return today, today
    if normalized_range == DATE_RANGE_LAST_7_DAYS:
        return today - timedelta(days=6), today
    if normalized_range == DATE_RANGE_THIS_WEEK:
        return today - timedelta(days=today.weekday()), today
    if normalized_range == DATE_RANGE_THIS_MONTH:
        return date(today.year, today.month, 1), today
    return None, None


def _normalize_date_range(date_range: str) -> str:
    candidate = str(date_range or "").strip()
    return candidate if candidate in DATE_RANGE_OPTIONS else DATE_RANGE_LAST_7_DAYS


def _load_summaries(
    *,
    summary_dir: Path,
    start_date: date | None,
    end_date: date | None,
) -> list[dict[str, Any]]:
    if not summary_dir.exists():
        return []

    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(summary_dir.glob("*.summary.json")):
        summary_date = _date_from_summary_path(summary_path)
        if summary_date is None:
            continue
        if start_date is not None and summary_date < start_date:
            continue
        if end_date is not None and summary_date > end_date:
            continue

        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict):
            summaries.append(loaded)
    return summaries


def _date_from_summary_path(summary_path: Path) -> date | None:
    date_text = summary_path.name.removesuffix(".summary.json")
    try:
        return date.fromisoformat(date_text)
    except ValueError:
        return None


def _sum_nested_int(summaries: list[dict[str, Any]], section: str, key: str) -> int:
    total = 0
    for summary in summaries:
        values = summary.get(section, {}) or {}
        if isinstance(values, dict):
            total += int(values.get(key, 0) or 0)
    return total


def _sum_counter(summaries: list[dict[str, Any]], *, section: str, key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for summary in summaries:
        section_values = summary.get(section, {}) or {}
        if not isinstance(section_values, dict):
            continue
        values = section_values.get(key, {}) or {}
        if not isinstance(values, dict):
            continue
        for name, count in values.items():
            normalized_name = str(name).strip()
            if normalized_name:
                counter[normalized_name] += int(count or 0)
    return counter


def _sum_error_counts_by_tool(summaries: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    error_counts = _sum_counter(summaries, section="tools", key="error_counts")
    for error_key, count in error_counts.items():
        tool_name, _, _ = str(error_key).partition("||")
        normalized_name = tool_name.strip()
        if normalized_name:
            counter[normalized_name] += int(count or 0)
    return counter


def _build_tool_issue_rows(
    *,
    registered_tool_names: Iterable[str],
    tool_calls_by_name: Counter[str],
    tool_failures_by_name: Counter[str],
    tool_errors_by_name: Counter[str],
) -> list[ToolIssueRow]:
    tool_names = set(tool_calls_by_name)
    tool_names.update(tool_failures_by_name)
    tool_names.update(tool_errors_by_name)
    tool_names.update(str(name).strip() for name in registered_tool_names if str(name).strip())

    rows = [
        ToolIssueRow(
            tool_name=tool_name,
            call_count=int(tool_calls_by_name.get(tool_name, 0)),
            issue_count=max(
                int(tool_failures_by_name.get(tool_name, 0)),
                int(tool_errors_by_name.get(tool_name, 0)),
            ),
            issue_rate=_rate(
                max(
                    int(tool_failures_by_name.get(tool_name, 0)),
                    int(tool_errors_by_name.get(tool_name, 0)),
                ),
                int(tool_calls_by_name.get(tool_name, 0)),
            ),
        )
        for tool_name in tool_names
    ]
    return sorted(
        rows,
        key=lambda row: (
            row.call_count == 0,
            -(row.issue_rate or 0.0),
            -row.issue_count,
            row.tool_name.lower(),
        ),
    )


def _rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)
