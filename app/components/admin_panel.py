from __future__ import annotations

import html
import re

import streamlit as st
from agents.factory import build_note_store, build_runtime_config, build_tool_registry
from agents.notes.tools import build_notes_reflection_registry
from agents.observability.performance_summary import (
    DATE_RANGE_OPTIONS,
    DATE_RANGE_LAST_7_DAYS,
    PerformanceReport,
    RateRow,
    ToolIssueRow,
    build_performance_report,
)
from app.state.agent_runtime_metadata import (
    COMPACTION_CONTROL_FIELDS,
    COMPACTION_LIMIT_GROUP,
    DEFAULT_RUNTIME_BUDGETS,
    REFLECTION_GATE_FIELDS,
    RUNTIME_LIMIT_GROUPS,
)
from app.state.agent_runtime_state import (
    clear_runtime_budget_override,
    get_runtime_gate_setting,
    get_runtime_budget_overrides,
    is_compaction_enabled,
    is_planning_enabled,
    is_reflection_enabled,
    set_compaction_enabled,
    set_planning_enabled,
    set_reflection_enabled,
    set_runtime_gate_setting,
    set_runtime_budget_override,
)
from app.state.chat_context_state import get_chat_context_snapshot
from app.components.workspace_host import (
    get_active_workspace_app_id,
    set_active_workspace_app_id,
)
from app.state.provider_state import (
    get_configured_provider_names,
    get_models_for_provider,
    get_provider_label,
    get_selected_model_name,
    get_selected_provider_name,
    initialize_provider_state,
    set_selected_model_name,
    set_selected_provider_name,
)
from app.workspace_apps.registry import list_workspace_app_metadata
from config.settings import PROJECT_ROOT


ADMIN_SUBTAB_KEY = "admin_active_subtab"
ADMIN_SUBTABS = ["Agent", "Tools", "Notes", "Trace", "Performance"]
ADMIN_PROVIDER_WIDGET_KEY = "admin_selected_provider_widget"
ADMIN_MODEL_WIDGET_KEY = "admin_selected_model_widget"
ADMIN_PERFORMANCE_RANGE_KEY = "admin_performance_date_range"
AGENT_METRICS_DIR = PROJECT_ROOT / "logs" / "agent_metrics"


def render_admin_panel() -> None:
    """
    Render the Admin view for current-session assistant operations.

    The Admin panel is intentionally organized into small rendering helpers so
    sections can be reordered, expanded, or restyled later without rewriting the
    whole page. All content is derived from session-state structures populated by
    the runner and tool usage tracking.
    """
    initialize_provider_state()
    usage = st.session_state.get(
        "tool_usage",
        {
            "total_calls": 0,
            "by_tool": {},
            "recent_events": [],
        },
    )
    activity = st.session_state.get(
        "agent_activity",
        {
            "total_runs": 0,
            "recent_runs": [],
        },
    )

    total_calls = int(usage.get("total_calls", 0))
    by_tool: dict[str, int] = usage.get("by_tool", {}) or {}
    recent_events: list[dict] = usage.get("recent_events", []) or []

    total_runs = int(activity.get("total_runs", 0))
    recent_runs: list[dict] = activity.get("recent_runs", []) or []

    available_tools = build_tool_registry().list_tools()
    available_tools = sorted(available_tools, key=lambda tool: tool.name.lower())
    runtime_config = build_runtime_config()
    note_store = build_note_store(runtime_config)
    runtime_note_tools = build_notes_reflection_registry(note_store).list_tools()
    runtime_note_tools = sorted(runtime_note_tools, key=lambda tool: tool.name.lower())

    st.markdown(
        """
        <div class="app-panel">
            <div class="panel-title">Admin</div>
            <div class="panel-subtitle">
                Session-level operational visibility for the assistant.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_subtab = _render_admin_subtab_selector()

    if active_subtab == "Agent":
        _render_agent_subtab()
        return

    if active_subtab == "Performance":
        _render_performance_subtab(
            available_tools=available_tools,
            runtime_note_tools=runtime_note_tools,
        )
        return

    if active_subtab == "Tools":
        _render_tools_subtab(
            total_calls=total_calls,
            by_tool=by_tool,
            recent_events=recent_events,
            total_runs=total_runs,
            available_tools=available_tools,
            runtime_note_tools=runtime_note_tools,
        )
        return

    if active_subtab == "Notes":
        _render_notes_subtab(note_store=note_store, recent_runs=recent_runs)
        return

    _render_trace_subtab(recent_runs=recent_runs)


def _render_admin_subtab_selector() -> str:
    active_subtab = str(st.session_state.get(ADMIN_SUBTAB_KEY, ADMIN_SUBTABS[0])).strip()
    if active_subtab not in ADMIN_SUBTABS:
        active_subtab = ADMIN_SUBTABS[0]
        st.session_state[ADMIN_SUBTAB_KEY] = active_subtab

    tab_cols = st.columns(len(ADMIN_SUBTABS))
    for index, subtab_name in enumerate(ADMIN_SUBTABS):
        with tab_cols[index]:
            if subtab_name == active_subtab:
                st.button(
                    subtab_name,
                    key=f"admin_subtab_button_{subtab_name.lower()}",
                    type="secondary",
                    use_container_width=True,
                    disabled=True,
                )
                continue

            if st.button(
                subtab_name,
                key=f"admin_subtab_button_{subtab_name.lower()}",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state[ADMIN_SUBTAB_KEY] = subtab_name
                st.rerun()

    _render_section_divider()
    return str(st.session_state.get(ADMIN_SUBTAB_KEY, active_subtab))


def _render_agent_subtab() -> None:
    _render_agent_selection_row()
    _render_runtime_mode_controls()
    _render_runtime_limits_section()
    _render_compacted_summary_viewer()


def _render_tools_subtab(
    *,
    total_calls: int,
    by_tool: dict[str, int],
    recent_events: list[dict],
    total_runs: int,
    available_tools: list,
    runtime_note_tools: list,
) -> None:
    _render_metrics(
        total_calls=total_calls,
        tool_count=len(by_tool),
        event_count=len(recent_events),
        total_runs=total_runs,
    )
    _render_calls_by_tool(by_tool)
    _render_recent_events(recent_events)
    _render_available_tools(available_tools, runtime_note_tools=runtime_note_tools)


def _render_notes_subtab(*, note_store, recent_runs: list[dict]) -> None:
    _render_runtime_notes_section(note_store=note_store, recent_runs=recent_runs)


def _render_trace_subtab(*, recent_runs: list[dict]) -> None:
    _render_recent_runs(recent_runs)


def _render_performance_subtab(*, available_tools: list, runtime_note_tools: list) -> None:
    date_range = _render_performance_date_range_selector()
    report = build_performance_report(
        summary_dir=AGENT_METRICS_DIR,
        date_range=date_range,
        registered_tool_names=_collect_registered_tool_names(
            available_tools=available_tools,
            runtime_note_tools=runtime_note_tools,
        ),
    )

    _render_performance_summary_caption(report)
    _render_performance_metric_cards(report)

    if report.files_loaded == 0:
        _render_empty_state("No daily performance summary logs are available for this date range.")

    outcome_col, runtime_col = st.columns(2)
    with outcome_col:
        _render_compact_rate_section(
            title="Run Outcomes",
            subtitle="",
            rows=report.run_outcome_rows,
        )
    with runtime_col:
        _render_compact_rate_section(
            title="Orchestration",
            subtitle="",
            rows=report.planning_rows,
        )

    _render_tool_issue_rate_section(report.tool_issue_rows)


def _render_performance_date_range_selector() -> str:
    options = list(DATE_RANGE_OPTIONS)
    selected = str(st.session_state.get(ADMIN_PERFORMANCE_RANGE_KEY, DATE_RANGE_LAST_7_DAYS)).strip()
    if selected not in options:
        selected = DATE_RANGE_LAST_7_DAYS
        st.session_state[ADMIN_PERFORMANCE_RANGE_KEY] = selected

    selector_col, _ = st.columns([0.28, 0.72])
    with selector_col:
        return str(
            st.selectbox(
                "Date range",
                options=options,
                index=options.index(selected),
                key=ADMIN_PERFORMANCE_RANGE_KEY,
            )
        )


def _collect_registered_tool_names(*, available_tools: list, runtime_note_tools: list) -> list[str]:
    tool_names = {
        str(getattr(tool, "name", "") or "").strip()
        for tool in list(available_tools or []) + list(runtime_note_tools or [])
        if str(getattr(tool, "name", "") or "").strip()
    }
    return sorted(tool_names, key=lambda value: value.lower())


def _render_performance_summary_caption(report: PerformanceReport) -> None:
    summary_count = f"{report.files_loaded} daily summar{'y' if report.files_loaded == 1 else 'ies'}"
    st.markdown(
        f'<div class="panel-subtitle">Includes {html.escape(summary_count)} in the selected date range.</div>',
        unsafe_allow_html=True,
    )


def _format_performance_date_span(report: PerformanceReport) -> str:
    if report.start_date is None or report.end_date is None:
        return "All available dates"
    if report.start_date == report.end_date:
        return report.start_date.isoformat()
    return f"{report.start_date.isoformat()} to {report.end_date.isoformat()}"


def _render_performance_metric_cards(report: PerformanceReport) -> None:
    max_steps_rate = _find_rate(report.run_outcome_rows, "Max Steps")
    completed_rate = _find_rate(report.run_outcome_rows, "Completed")
    failed_rate = _find_rate(report.run_outcome_rows, "Failed")
    stopped_rate = _find_rate(report.run_outcome_rows, "Stopped")
    provider_error_rate = _find_rate(report.run_outcome_rows, "Provider Errors")
    reflection_rate = _find_rate(report.planning_rows, "Reflection")
    planning_rate = _rate_from_counts(report.planning_used, report.total_runs)

    st.markdown(
        f"""
        <div class="metric-grid" style="grid-template-columns:repeat(auto-fit,minmax(180px,1fr));">
            <div class="m-card">
                <div class="m-val">{report.total_runs}</div>
                <div class="m-label">Runs</div>
                {_build_performance_card_detail_rows_html([
                    ("Completed", _format_percent(completed_rate)),
                    ("Failed", _format_percent(failed_rate)),
                    ("Stopped", _format_percent(stopped_rate)),
                ])}
            </div>
            <div class="m-card">
                <div class="m-val">{_format_seconds(report.avg_duration_seconds)}</div>
                <div class="m-label">Speed</div>
                {_build_performance_card_detail_rows_html([
                    ("Average/run", _format_seconds(report.avg_duration_seconds)),
                    ("Total runtime", _format_seconds(report.duration_seconds_total)),
                ])}
            </div>
            <div class="m-card">
                <div class="m-val">{_format_percent(report.tool_success_rate)}</div>
                <div class="m-label">Tool Health</div>
                {_build_performance_card_detail_rows_html([
                    ("Total calls", str(report.total_tool_calls)),
                    ("Tools/run", _format_decimal(report.avg_tools_per_run)),
                    ("Issue rate", _format_percent(report.tool_issue_rate)),
                ])}
            </div>
            <div class="m-card">
                <div class="m-val">{_format_percent(max_steps_rate)}</div>
                <div class="m-label">Runtime</div>
                {_build_performance_card_detail_rows_html([
                    ("Max steps", _format_percent(max_steps_rate)),
                    ("Provider errors", _format_percent(provider_error_rate)),
                    ("Reflection", _format_percent(reflection_rate)),
                    ("Planning", _format_percent(planning_rate)),
                ])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_performance_card_detail_rows_html(items: list[tuple[str, str]]) -> str:
    rows_html = "".join(
        '<div style="display:flex;align-items:center;justify-content:space-between;gap:0.75rem;padding:0.16rem 0;">'
        f'<span style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--slate);font-size:0.78rem;font-weight:600;">{html.escape(label)}</span>'
        f'<span style="flex:0 0 auto;color:var(--text);font-family:var(--font-mono);font-size:0.78rem;font-weight:600;">{html.escape(value)}</span>'
        "</div>"
        for label, value in items
    )
    return (
        '<div style="margin-top:0.8rem;padding-top:0.65rem;border-top:1px solid var(--border);text-align:left;">'
        f"{rows_html}"
        "</div>"
    )


def _render_compact_rate_section(*, title: str, subtitle: str, rows: list[RateRow]) -> None:
    _render_section_header(title=title, subtitle=subtitle)
    if not rows:
        _render_empty_state("No performance rows are available.")
        return
    st.markdown(
        _build_compact_rate_rows_html(
            [(row.label, _format_percent(row.rate)) for row in rows],
        ),
        unsafe_allow_html=True,
    )


def _render_tool_issue_rate_section(rows: list[ToolIssueRow]) -> None:
    _render_section_header(
        title="Tool Issue Rates",
        subtitle="",
    )
    if not rows:
        _render_empty_state("No tools are registered or present in selected summaries.")
        return
    st.markdown(
        _build_compact_rate_rows_html(
            [(row.tool_name, _format_percent(row.issue_rate)) for row in rows],
            max_height_px=360,
        ),
        unsafe_allow_html=True,
    )


def _build_compact_rate_rows_html(rows: list[tuple[str, str]], *, max_height_px: int | None = None) -> str:
    max_height_style = f"max-height:{max_height_px}px;overflow-y:auto;" if max_height_px else ""
    row_html = "".join(
        '<div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:0.45rem 0;border-bottom:1px solid var(--border);">'
        f'<div style="min-width:0;color:var(--text);font-size:0.88rem;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{html.escape(label)}</div>'
        f'<div style="flex:0 0 auto;color:var(--slate);font-family:var(--font-mono);font-size:0.86rem;">{html.escape(value)}</div>'
        "</div>"
        for label, value in rows
    )
    return f'<div style="background:var(--card);border:1px solid var(--border);border-radius:14px;padding:0.35rem 0.85rem;margin-bottom:1rem;{max_height_style}">{row_html}</div>'


def _find_rate(rows: list[RateRow], label: str) -> float | None:
    for row in rows:
        if row.label == label:
            return row.rate
    return None


def _rate_from_counts(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.1f}%"


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}s"


def _format_decimal(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}"


def _render_agent_selection_row() -> None:
    """
    Render workspace, provider, and model selectors in one Agent subtab row.
    """
    available_apps = list_workspace_app_metadata()
    app_options = [
        str(app.get("app_id", "")).strip()
        for app in available_apps
        if str(app.get("app_id", "")).strip()
    ]
    app_labels = {
        str(app.get("app_id", "")).strip(): (
            str(app.get("app_label", "")).strip()
            or str(app.get("app_id", "")).strip()
        )
        for app in available_apps
        if str(app.get("app_id", "")).strip()
    }
    configured_provider_names = get_configured_provider_names()
    current_provider = get_selected_provider_name()
    current_model = get_selected_model_name()
    provider_options = configured_provider_names

    if provider_options and current_provider not in provider_options:
        current_provider = provider_options[0]
        set_selected_provider_name(current_provider)
        current_model = get_selected_model_name()

    provider_labels = {
        provider_name: get_provider_label(provider_name)
        for provider_name in provider_options
    }

    if provider_options:
        _sync_admin_provider_model_widget_state(
            provider_options=provider_options,
            current_provider=current_provider,
            current_model=current_model,
        )

    selected_app_id: str | None = None
    current_app_id = get_active_workspace_app_id()
    if app_options:
        if current_app_id not in app_options:
            current_app_id = app_options[0]
        selected_app_id = current_app_id

    selected_provider = (
        str(st.session_state.get(ADMIN_PROVIDER_WIDGET_KEY, current_provider)).strip().lower()
        if provider_options
        else ""
    )
    model_options = get_models_for_provider(selected_provider) if selected_provider else []

    workspace_col, provider_col, model_col = st.columns(3)
    with workspace_col:
        if app_options and selected_app_id is not None:
            selected_app_id = st.selectbox(
                "Choose workspace app",
                options=app_options,
                index=app_options.index(selected_app_id),
                format_func=lambda app_id: app_labels.get(app_id, app_id),
                key="admin_selected_workspace_app_widget",
            )
        else:
            _render_empty_state("No workspace apps are currently registered.")

    with provider_col:
        if provider_options:
            st.selectbox(
                "Choose provider",
                options=provider_options,
                format_func=lambda provider_name: provider_labels.get(provider_name, provider_name),
                key=ADMIN_PROVIDER_WIDGET_KEY,
                on_change=_handle_admin_provider_change,
            )
        else:
            _render_empty_state(
                "No configured model providers are available. Add provider API keys to your environment settings."
            )

    with model_col:
        if model_options:
            st.selectbox(
                "Choose model",
                options=model_options,
                key=ADMIN_MODEL_WIDGET_KEY,
                on_change=_handle_admin_model_change,
            )

    if selected_app_id is not None and selected_app_id != get_active_workspace_app_id():
        set_active_workspace_app_id(selected_app_id)
        st.rerun()

    status_badges = []
    if selected_app_id is not None:
        status_badges.append(
            _build_badge_html(
                label=f"Active workspace app · {app_labels.get(selected_app_id, selected_app_id)}",
                variant="info",
            )
        )
    if provider_options:
        status_badges.append(
            _build_badge_html(
                label=f"Active provider · {get_provider_label(get_selected_provider_name())}",
                variant="info",
            )
        )
        status_badges.append(
            _build_badge_html(
                label=f"Active model · {get_selected_model_name()}",
                variant="info",
            )
        )

    if status_badges:
        st.markdown(
            f'<div class="status-row">{"".join(status_badges)}</div>',
            unsafe_allow_html=True,
        )


# Provider/model selection section for the Admin panel
def _sync_admin_provider_model_widget_state(
    *,
    provider_options: list[str],
    current_provider: str,
    current_model: str,
) -> None:
    """
    Reconcile Admin widget session state against the canonical provider/model
    state before rendering the widgets.

    This keeps the Admin controls aligned with changes coming from the control
    rail while still allowing widget callbacks to update the canonical state
    first on direct Admin interactions.
    """
    if st.session_state.get(ADMIN_PROVIDER_WIDGET_KEY) != current_provider:
        st.session_state[ADMIN_PROVIDER_WIDGET_KEY] = current_provider

    active_provider = str(st.session_state.get(ADMIN_PROVIDER_WIDGET_KEY, current_provider)).strip().lower()
    if active_provider not in provider_options:
        active_provider = current_provider
        st.session_state[ADMIN_PROVIDER_WIDGET_KEY] = current_provider

    valid_model_options = get_models_for_provider(active_provider)
    widget_model = str(st.session_state.get(ADMIN_MODEL_WIDGET_KEY, ""))
    if widget_model not in valid_model_options or widget_model != current_model:
        st.session_state[ADMIN_MODEL_WIDGET_KEY] = current_model


def _handle_admin_provider_change() -> None:
    """Persist Admin provider widget changes into the canonical provider state."""
    selected_provider = str(st.session_state.get(ADMIN_PROVIDER_WIDGET_KEY, "")).strip().lower()
    if not selected_provider:
        return

    set_selected_provider_name(selected_provider)
    st.session_state[ADMIN_MODEL_WIDGET_KEY] = get_selected_model_name()


def _handle_admin_model_change() -> None:
    """Persist Admin model widget changes into the canonical model state."""
    selected_model = str(st.session_state.get(ADMIN_MODEL_WIDGET_KEY, "")).strip()
    if not selected_model:
        return

    set_selected_model_name(selected_model)


def _render_runtime_mode_controls() -> None:
    planning_enabled = is_planning_enabled()
    reflection_enabled = is_reflection_enabled()
    compaction_enabled = is_compaction_enabled()

    _render_section_header(
        title="Runtime Modes",
        subtitle="Live session toggles for planning, reflection, and context compaction.",
    )

    planning_col, reflection_col, compaction_col = st.columns(3)
    with planning_col:
        planning_value = st.toggle(
            "Enable Planning",
            value=planning_enabled,
            key="admin_planning_enabled_toggle",
        )
    with reflection_col:
        reflection_value = st.toggle(
            "Enable Reflection",
            value=reflection_enabled,
            key="admin_reflection_enabled_toggle",
        )
    with compaction_col:
        compaction_value = st.toggle(
            "Enable Context Compaction",
            value=compaction_enabled,
            key="admin_compaction_enabled_toggle",
        )

    if planning_value != planning_enabled:
        set_planning_enabled(planning_value)
        st.rerun()
    if reflection_value != reflection_enabled:
        set_reflection_enabled(reflection_value)
        st.rerun()
    if compaction_value != compaction_enabled:
        set_compaction_enabled(compaction_value)
        st.rerun()

    badges = "".join(
        [
            _build_badge_html(label=f"Planning · {'On' if planning_value else 'Off'}", variant="info" if planning_value else "neutral"),
            _build_badge_html(label=f"Reflection · {'On' if reflection_value else 'Off'}", variant="info" if reflection_value else "neutral"),
            _build_badge_html(label=f"Compaction · {'On' if compaction_value else 'Off'}", variant="info" if compaction_value else "neutral"),
        ]
    )
    st.markdown(
        f'<div class="status-row">{badges}</div>',
        unsafe_allow_html=True,
    )


def _render_runtime_limits_section() -> None:
    _render_section_header(
        title="Runtime Limits",
        subtitle="Adjust the per-phase model, tool, note, and execution budgets already supported by the runtime.",
    )

    left_col, right_col = st.columns(2)
    with left_col:
        _render_runtime_limit_group(_get_runtime_limit_group("Planning"))
        _render_agent_group_spacer()
        _render_runtime_limit_group(_get_runtime_limit_group("Execution"))
        _render_agent_group_spacer()
        _render_runtime_limit_group(COMPACTION_LIMIT_GROUP)
    with right_col:
        _render_reflection_runtime_group()
        _render_agent_group_spacer()
        _render_compaction_controls()


def _get_runtime_limit_group(title: str) -> dict:
    normalized_title = str(title or "").strip().lower()
    for group in RUNTIME_LIMIT_GROUPS:
        if str(group.get("title") or "").strip().lower() == normalized_title:
            return group
    raise ValueError(f"Unknown runtime limit group: {title}")


def _render_agent_group_spacer() -> None:
    st.markdown(
        '<div style="height: 0.9rem;"></div>',
        unsafe_allow_html=True,
    )


def _render_runtime_limit_group(group: dict) -> None:
    title = str(group.get("title") or "").strip()
    subtitle = str(group.get("subtitle") or "").strip()
    fields = group.get("fields", []) or []

    if not fields:
        return

    input_col, _ = st.columns([0.72, 0.28])
    with input_col:
        _render_runtime_group_header(title=title, subtitle=subtitle)
        for field in fields:
            _render_runtime_limit_input(field)


def _render_reflection_runtime_group() -> None:
    group = _get_runtime_limit_group("Reflection")
    title = str(group.get("title") or "").strip()
    subtitle = str(group.get("subtitle") or "").strip()
    budget_fields = group.get("fields", []) or []

    if not REFLECTION_GATE_FIELDS and not budget_fields:
        return

    input_col, _ = st.columns([0.72, 0.28])
    with input_col:
        _render_runtime_group_header(title=title, subtitle=subtitle)
        for field in REFLECTION_GATE_FIELDS:
            _render_runtime_gate_input(field)
        for field in budget_fields:
            _render_runtime_limit_input(field)


def _render_runtime_gate_input(field: dict) -> None:
    field_name = str(field.get("name") or "").strip()
    if not field_name:
        return

    current_value = get_runtime_gate_setting(field_name)
    label = str(field.get("label") or field_name).strip()
    min_value = int(field.get("min", 0))
    max_value = int(field.get("max", 100))
    step_value = int(field.get("step", 1))
    help_text = str(field.get("help") or "").strip() or None

    selected_value = st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=current_value,
        step=step_value,
        help=help_text,
        key=f"admin_runtime_gate_{field_name}",
    )
    set_runtime_gate_setting(field_name, int(selected_value))


def _render_runtime_limit_input(field: dict) -> None:
    field_name = str(field.get("name") or "").strip()
    if not field_name:
        return

    budget_overrides = get_runtime_budget_overrides()
    default_value = int(getattr(DEFAULT_RUNTIME_BUDGETS, field_name))
    current_value = int(budget_overrides.get(field_name, default_value))
    label = str(field.get("label") or field_name).strip()
    min_value = int(field.get("min", 0))
    max_value = int(field.get("max", 100))
    step_value = int(field.get("step", 1))
    help_text = str(field.get("help") or "").strip() or None

    selected_value = st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=current_value,
        step=step_value,
        help=help_text,
        key=f"admin_runtime_limit_{field_name}",
    )
    _sync_budget_override(field_name, int(selected_value), default_value=default_value)


def _render_compaction_controls() -> None:
    budget_overrides = get_runtime_budget_overrides()

    selected_values: dict[str, int] = {}
    input_col, _ = st.columns([0.72, 0.28])
    with input_col:
        _render_runtime_group_header(
            title="Context Compaction",
            subtitle="Control when older conversation history is summarized into hidden runtime context.",
        )
        for field in COMPACTION_CONTROL_FIELDS:
            field_name = str(field.get("name") or "").strip()
            if not field_name:
                continue
            default_value = int(field.get("default", getattr(DEFAULT_RUNTIME_BUDGETS, field_name)))
            current_value = int(budget_overrides.get(field_name, default_value))
            selected_values[field_name] = int(
                st.number_input(
                    str(field.get("label") or field_name),
                    min_value=int(field.get("min", 0)),
                    max_value=int(field.get("max", 100)),
                    value=current_value,
                    step=int(field.get("step", 1)),
                    key=f"admin_{field_name}_input",
                )
            )

    for field in COMPACTION_CONTROL_FIELDS:
        field_name = str(field.get("name") or "").strip()
        if not field_name:
            continue
        default_value = int(field.get("default", getattr(DEFAULT_RUNTIME_BUDGETS, field_name)))
        _sync_budget_override(field_name, selected_values[field_name], default_value=default_value)


def _render_compacted_summary_viewer() -> None:
    runtime_snapshot = get_chat_context_snapshot()
    summary = str(runtime_snapshot.get("summary") or "").strip()

    _render_section_header(
        title="Compacted Context Summary",
        subtitle="Read the hidden summary retained from older conversation history when context compaction has run.",
    )
    st.markdown(
        f'<div class="status-row">{_build_compaction_status_badges(runtime_snapshot)}</div>',
        unsafe_allow_html=True,
    )

    if not summary:
        _render_empty_state("No compacted context summary is available in this session yet.")
        return

    st.markdown(
        _build_code_panel_html(summary, label="Summary"),
        unsafe_allow_html=True,
    )




def _render_section_divider() -> None:
    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )


def _render_runtime_group_header(*, title: str, subtitle: str) -> None:
    if title:
        _render_group_label(title)
    if subtitle:
        _render_empty_state(subtitle)


def _sync_budget_override(name: str, value: int, *, default_value: int) -> None:
    if int(value) == int(default_value):
        clear_runtime_budget_override(name)
        return
    set_runtime_budget_override(name, int(value))


def _render_group_label(title: str) -> None:
    st.markdown(
        f"""
        <div
            style="
                margin-top: 0.4rem;
                margin-bottom: 0.35rem;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #64748b;
            "
        >
            {html.escape(title)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_compaction_status_badges(runtime_snapshot: dict) -> str:
    enabled = is_compaction_enabled()
    summary = runtime_snapshot.get("summary")
    compacted_message_count = int(runtime_snapshot.get("compacted_message_count", 0))
    metrics = runtime_snapshot.get("metrics", {}) or {}

    return "".join(
        [
            _build_badge_html(
                label=f"Compaction · {'On' if enabled else 'Off'}",
                variant="info" if enabled else "neutral",
            ),
            _build_badge_html(
                label=f"Compacted Messages · {compacted_message_count}",
                variant="info",
            ),
            _build_badge_html(
                label=f"Visible Messages · {int(metrics.get('message_count', 0))}",
                variant="info",
            ),
            _build_badge_html(
                label=f"Summary · {'Present' if summary else 'None'}",
                variant="info" if summary else "neutral",
            ),
        ]
    )


def _render_metrics(
    *,
    total_calls: int,
    tool_count: int,
    event_count: int,
    total_runs: int,
) -> None:
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="m-card">
                <div class="m-val">{total_calls}</div>
                <div class="m-label">Total Tool Calls</div>
            </div>
            <div class="m-card">
                <div class="m-val">{tool_count}</div>
                <div class="m-label">Tools Used</div>
            </div>
            <div class="m-card">
                <div class="m-val">{event_count}</div>
                <div class="m-label">Recent Events</div>
            </div>
            <div class="m-card">
                <div class="m-val">{total_runs}</div>
                <div class="m-label">Agent Runs</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _render_calls_by_tool(by_tool: dict[str, int]) -> None:
    _render_section_header(
        title="Calls by Tool",
        subtitle="Current-session counts for each registered tool that has been used.",
    )

    if not by_tool:
        _render_empty_state("No tools have been called in this session yet.")
        return

    badges = "".join(
        _build_badge_html(
            label=f"{tool_name} · {count} call{'s' if count != 1 else ''}",
            variant="info",
        )
        for tool_name, count in sorted(by_tool.items(), key=lambda item: item[0])
    )

    st.markdown(
        f'<div class="status-row">{badges}</div>',
        unsafe_allow_html=True,
    )



def _render_recent_events(recent_events: list[dict]) -> None:
    _render_section_header(
        title="Recent Tool Events",
        subtitle="Most recent tool execution events from this session.",
    )

    if not recent_events:
        _render_empty_state("No recent tool events are available yet.")
        return

    badges = "".join(
        _build_event_badge_html(event)
        for event in reversed(recent_events)
    )

    st.markdown(
        f'<div class="status-row">{badges}</div>',
        unsafe_allow_html=True,
    )



def _render_recent_runs(recent_runs: list[dict]) -> None:
    _render_section_header(
        title="Recent Agent Runs",
        subtitle=(
            "High-level outcomes for recent assistant runs, including provider, "
            "model, stop reason, and step count."
        ),
    )

    if not recent_runs:
        _render_empty_state("No agent runs have been recorded in this session yet.")
        return

    indexed_runs = list(enumerate(recent_runs))
    for run_index, run in reversed(indexed_runs):
        _render_run_card(run, run_index=run_index)



def _render_available_tools(tools: list, *, runtime_note_tools: list) -> None:
    _render_section_header(
        title="Available Tools",
        subtitle="All tools currently registered and available to the assistant in this app session. Hover over a tool for more information.",
    )

    if not tools:
        _render_empty_state("No tools are currently registered.")
        return

    general_tools = [
        tool for tool in tools
        if str(getattr(tool, "scope", "framework")).strip().lower() == "framework"
    ]
    app_tools = [
        tool for tool in tools
        if str(getattr(tool, "scope", "")).strip().lower() == "app"
    ]

    _render_tool_group(
        title="General Tools",
        subtitle="Shell-level tools available across workspace apps.",
        tools=general_tools,
        empty_message="No general tools are currently registered.",
    )
    _render_tool_group(
        title="App-Specific Tools",
        subtitle="Tools exposed by the active workspace app.",
        tools=app_tools,
        empty_message="No app-specific tools are currently registered for the active workspace app.",
    )
    _render_tool_group(
        title="Runtime Notes Tools",
        subtitle="Hidden runtime tools used during planning/reflection note access and maintenance.",
        tools=runtime_note_tools,
        empty_message="No runtime note tools are currently registered.",
    )


def _render_runtime_notes_section(*, note_store, recent_runs: list[dict]) -> None:
    _render_section_header(
        title="Runtime Notes",
        subtitle="Persistent runtime heuristics used by planning and reflection, with recent note-maintenance activity.",
    )

    note_files = note_store.list_note_files()
    total_files = len(note_files)
    total_notes = sum(int(item.get("note_count", 0)) for item in note_files)
    general_notes = sum(
        int(item.get("note_count", 0))
        for item in note_files
        if str(item.get("scope") or "").strip().lower() == "general"
    )
    app_notes = total_notes - general_notes
    recent_mutations = _collect_recent_note_mutations(recent_runs)

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="m-card">
                <div class="m-val">{total_files}</div>
                <div class="m-label">Note Files</div>
            </div>
            <div class="m-card">
                <div class="m-val">{total_notes}</div>
                <div class="m-label">Total Notes</div>
            </div>
            <div class="m-card">
                <div class="m-val">{general_notes}</div>
                <div class="m-label">General Notes</div>
            </div>
            <div class="m-card">
                <div class="m-val">{app_notes}</div>
                <div class="m-label">App Notes</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div
            style="
                margin-top: 0.9rem;
                margin-bottom: 0.45rem;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #64748b;
            "
        >
            Note Files
        </div>
        """,
        unsafe_allow_html=True,
    )
    if note_files:
        badges = "".join(
            _build_badge_html(
                label=f"{str(item.get('file_name') or '').strip()} · {int(item.get('note_count', 0))} notes",
                variant="info",
            )
            for item in note_files
        )
        st.markdown(
            f'<div class="status-row">{badges}</div>',
            unsafe_allow_html=True,
        )
    else:
        _render_empty_state("No runtime note files have been created yet.")

    st.markdown(
        """
        <div
            style="
                margin-top: 0.9rem;
                margin-bottom: 0.45rem;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #64748b;
            "
        >
            Recent Note Activity
        </div>
        """,
        unsafe_allow_html=True,
    )

    if recent_mutations:
        badges = "".join(_build_note_mutation_badge_html(mutation) for mutation in recent_mutations)
        st.markdown(
            f'<div class="status-row">{badges}</div>',
            unsafe_allow_html=True,
        )
    else:
        _render_empty_state("No runtime note mutations have been recorded in recent runs.")


def _render_tool_group(
    *,
    title: str,
    subtitle: str,
    tools: list,
    empty_message: str,
) -> None:
    st.markdown(
        f"""
        <div
            style="
                margin-top: 0.7rem;
                margin-bottom: 0.4rem;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #64748b;
            "
        >
            {html.escape(title)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not tools:
        st.markdown(
            f"""
            <div
                style="
                    margin: 0 0 0.65rem 0;
                    font-size: 0.82rem;
                    color: #64748b;
                "
            >
                {html.escape(empty_message)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    badges = "".join(
        _build_tool_badge_html(tool_name=tool.name, description=tool.description)
        for tool in tools
    )

    st.markdown(
        f'<div class="status-row">{badges}</div>',
        unsafe_allow_html=True,
    )


def _build_tool_badge_html(*, tool_name: str, description: str | None) -> str:
    safe_tool_name = html.escape(tool_name)
    safe_description = html.escape(description or "No description available.")
    return (
        '<div class="badge-tooltip-wrap">'
        f'<div class="badge info">{safe_tool_name}</div>'
        f'<div class="badge-tooltip">{safe_description}</div>'
        '</div>'
    )


def _build_note_mutation_badge_html(mutation: dict) -> str:
    file_name = str(mutation.get("file_name") or "unknown").strip()
    tool_name = str(mutation.get("tool_name") or "note_tool").strip()
    note_id = str(mutation.get("note_id") or "").strip()
    scope = str(mutation.get("scope") or "").strip()
    statement = str(mutation.get("statement") or "").strip()
    confidence = mutation.get("confidence")
    action_label = "delete" if tool_name == "delete_runtime_note" else "upsert"
    label = f"{file_name} · {action_label}"
    if note_id:
        label = f"{label} · {note_id}"

    tooltip_lines = [
        f"Scope: {scope or 'unknown'}",
        f"Tool: {tool_name}",
    ]
    if statement:
        tooltip_lines.append(f"Statement: {statement}")
    if confidence not in (None, ""):
        tooltip_lines.append(f"Confidence: {confidence}")
    tooltip_lines.append(f"Status: {str(mutation.get('status') or 'unknown')}")
    message = str(mutation.get("message") or "").strip()
    if message:
        tooltip_lines.append(f"Message: {message}")

    safe_label = html.escape(label)
    safe_tooltip = html.escape("\n".join(tooltip_lines))
    return (
        '<div class="badge-tooltip-wrap">'
        f'<div class="badge info">{safe_label}</div>'
        f'<div class="badge-tooltip">{safe_tooltip}</div>'
        '</div>'
    )



def _build_run_tools_badge_html(run: dict) -> str:
    tool_results = run.get("tool_results", []) or []
    tool_counts = _get_run_tool_counts(tool_results)
    tool_count = int(run.get("tool_count", 0))

    if not tool_counts:
        tooltip_text = "No tools recorded."
    else:
        tooltip_text = "\n".join(
            f"{tool_name} ({count})"
            for tool_name, count in sorted(tool_counts.items(), key=lambda item: item[0])
        )

    safe_label = html.escape(f"Tools · {tool_count}")
    safe_tooltip = html.escape(tooltip_text)
    return (
        '<div class="badge-tooltip-wrap">'
        f'<div class="badge info">{safe_label}</div>'
        f'<div class="badge-tooltip">{safe_tooltip}</div>'
        '</div>'
    )


# New helper function for run steps badge
def _build_run_steps_badge_html(run: dict) -> str:
    trace = run.get("trace", []) or []
    metadata = run.get("metadata", {}) or {}
    steps_used = metadata.get("steps_used", "—")

    step_lines: list[str] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        step = item.get("step", "—")
        stage = str(item.get("stage") or "unknown")

        if stage == "tool_requested":
            tool_name = str(item.get("tool_name") or "unknown")
            step_lines.append(f"Step {step} · tool requested · {tool_name}")
        elif stage == "tool_result":
            tool_name = str(item.get("tool_name") or "unknown")
            success = bool(item.get("success", False))
            status = "success" if success else "failed"
            step_lines.append(f"Step {step} · tool {status} · {tool_name}")
        elif stage == "provider_error":
            step_lines.append(f"Step {step} · provider error")
        elif stage == "provider_response":
            response = item.get("response", {}) or {}
            tool_call = response.get("tool_call")
            if tool_call:
                step_lines.append(f"Step {step} · provider response · tool call")
            else:
                step_lines.append(f"Step {step} · provider response")
        else:
            step_lines.append(f"Step {step} · {stage}")

    tooltip_text = "\n".join(step_lines) if step_lines else "No steps recorded."
    safe_label = html.escape(f"Steps · {steps_used}")
    safe_tooltip = html.escape(tooltip_text)
    return (
        '<div class="badge-tooltip-wrap">'
        f'<div class="badge info">{safe_label}</div>'
        f'<div class="badge-tooltip">{safe_tooltip}</div>'
        '</div>'
    )


def _get_run_tool_counts(tool_results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tool_result in tool_results:
        if not isinstance(tool_result, dict):
            continue
        tool_name = str(tool_result.get("tool_name") or "").strip()
        if not tool_name:
            continue
        counts[tool_name] = counts.get(tool_name, 0) + 1
    return counts


def _collect_recent_note_mutations(recent_runs: list[dict]) -> list[dict]:
    mutations: list[dict] = []
    for run in reversed(recent_runs):
        trace = run.get("trace", []) or []
        for item in trace:
            if not isinstance(item, dict) or str(item.get("stage") or "") != "reflection":
                continue
            reflection = item.get("reflection", {}) or {}
            tool_activity = reflection.get("tool_activity", [])
            if not isinstance(tool_activity, list):
                continue
            for activity in tool_activity:
                if not isinstance(activity, dict):
                    continue
                tool_name = str(activity.get("tool_name") or "").strip()
                if tool_name not in {"upsert_runtime_note", "delete_runtime_note"}:
                    continue
                mutations.append(activity)
                if len(mutations) >= 8:
                    return mutations
    return mutations

def _render_section_header(*, title: str, subtitle: str) -> None:
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)
    st.markdown(
        f"""
        <div class="panel-title">{safe_title}</div>
        <div class="panel-subtitle">{safe_subtitle}</div>
        """,
        unsafe_allow_html=True,
    )



def _render_empty_state(message: str) -> None:
    safe_message = html.escape(message)
    st.markdown(
        f'<div class="panel-subtitle">{safe_message}</div>',
        unsafe_allow_html=True,
    )



def _build_badge_html(*, label: str, variant: str = "info") -> str:
    safe_label = html.escape(label)
    return f'<div class="badge {variant}">{safe_label}</div>'



def _build_event_badge_html(event: dict) -> str:
    tool_name = html.escape(str(event.get("tool_name", "")))
    success = bool(event.get("success", False))

    status_label = "success" if success else "failed"
    status_variant = "success" if success else "error"

    return _build_badge_html(
        label=f"{tool_name} - {status_label}",
        variant=status_variant,
    )


def _extract_run_phase_snapshot(run: dict) -> dict[str, object]:
    planning_mode = "none"
    planning_used = False
    planning_skipped = False
    reflection_happened = False
    compaction_happened = False

    for item in run.get("trace", []) or []:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "").strip().lower()
        if stage == "triage":
            decision = item.get("decision", {}) or {}
            candidate_mode = str(decision.get("planning_mode") or "").strip().lower()
            if candidate_mode in {"skip", "light", "deep"}:
                planning_mode = candidate_mode
                if candidate_mode == "skip" and str(decision.get("reason") or "").strip().lower() != "planning_disabled":
                    planning_skipped = True
        elif stage == "planning":
            planning_used = True
        elif stage == "reflection":
            reflection_happened = True
        elif stage == "compaction":
            compaction_happened = True

    if planning_used and planning_mode not in {"light", "deep"}:
        planning_mode = "used"

    return {
        "planning_mode": planning_mode,
        "planning_used": planning_used,
        "planning_skipped": planning_skipped,
        "reflection_happened": reflection_happened,
        "compaction_happened": compaction_happened,
    }


def _planning_mode_variant(planning_mode: str) -> str:
    normalized = str(planning_mode or "").strip().lower()
    if normalized == "deep":
        return "warning"
    if normalized == "light":
        return "info"
    if normalized == "skip":
        return "neutral"
    if normalized == "used":
        return "info"
    return "neutral"


def _format_planning_status_label(snapshot: dict[str, object]) -> str:
    planning_mode = str(snapshot.get("planning_mode") or "none").strip().lower()
    planning_used = bool(snapshot.get("planning_used", False))
    planning_skipped = bool(snapshot.get("planning_skipped", False))

    if planning_used and planning_mode in {"light", "deep"}:
        return f"Planning · {planning_mode.title()}"
    if planning_used:
        return "Planning · Used"
    if planning_skipped:
        return "Planning · Skipped"
    return "Planning · None"


def _build_run_phase_badges_html(run: dict) -> str:
    snapshot = _extract_run_phase_snapshot(run)
    reflection_happened = bool(snapshot.get("reflection_happened", False))
    compaction_happened = bool(snapshot.get("compaction_happened", False))
    planning_mode = str(snapshot.get("planning_mode") or "unknown")
    planning_used = bool(snapshot.get("planning_used", False))
    planning_skipped = bool(snapshot.get("planning_skipped", False))

    return "".join(
        [
            _build_badge_html(
                label=_format_planning_status_label(snapshot),
                variant=_planning_mode_variant(planning_mode) if planning_used or planning_skipped else "neutral",
            ),
            _build_badge_html(
                label=f"Reflection · {'Yes' if reflection_happened else 'No'}",
                variant="info" if reflection_happened else "neutral",
            ),
            _build_badge_html(
                label=f"Compaction · {'Yes' if compaction_happened else 'No'}",
                variant="info" if compaction_happened else "neutral",
            ),
        ]
    )




def _render_run_card(run: dict, *, run_index: int) -> None:
    """
    Render a single agent run card with an expandable trace section.

    The summary content remains visible by default, while the trace can be
    expanded per run using session-state-backed UI controls. This keeps the
    layout compact without changing the existing visual language.
    """
    trace_toggle_key = f"admin_run_trace_expanded_{run_index}"
    is_expanded = bool(st.session_state.get(trace_toggle_key, False))

    with st.container():
        _render_run_summary(run)

        toggle_col_left, toggle_col_right = st.columns([0.82, 0.18])
        with toggle_col_right:
            toggle_label = "▾ Trace" if is_expanded else "▸ Trace"
            if st.button(toggle_label, key=f"{trace_toggle_key}_button", use_container_width=True):
                st.session_state[trace_toggle_key] = not is_expanded
                st.rerun()

        if st.session_state.get(trace_toggle_key, False):
            trace_blocks = _build_trace_sections_html(run.get("trace", []) or [], run=run)
            trace_html = (
                trace_blocks
                if trace_blocks
                else '<div class="panel-subtitle">No trace events recorded for this run.</div>'
            )

            st.markdown(
                f"""
                <div class="app-panel">
                    <div class="panel-title">Trace</div>
                    <div class="trace-stack">{trace_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



def _render_run_summary(run: dict) -> None:
    metadata = run.get("metadata", {}) or {}
    provider = str(metadata.get("provider") or "Unknown provider")
    model = str(metadata.get("model") or "Unknown model")
    stop_reason = str(run.get("stop_reason") or "unknown")
    user_input = str(run.get("user_input") or "")
    final_text = str(run.get("final_text") or "").strip()

    stop_variant = _stop_reason_variant(stop_reason)
    stop_badge = _build_badge_html(label=f"Stop · {stop_reason}", variant=stop_variant)
    steps_badge = _build_run_steps_badge_html(run)
    tools_badge = _build_run_tools_badge_html(run)
    provider_badge = _build_badge_html(label=f"Provider · {provider}", variant="info")
    model_badge = _build_badge_html(label=f"Model · {model}", variant="info")
    phase_badges = _build_run_phase_badges_html(run)

    response_html = (
        _build_plain_response_panel_html(final_text, max_chars=3000)
        if final_text
        else '<div class="panel-subtitle">No final text recorded.</div>'
    )

    st.markdown(
        f"""
        <div class="app-panel">
            <div class="panel-title">Run</div>
            <div class="panel-subtitle">{html.escape(user_input) if user_input else 'No user input recorded.'}</div>
            <div class="status-row">{provider_badge}{model_badge}{steps_badge}{tools_badge}{stop_badge}</div>
            <div class="status-row">{phase_badges}</div>
            {response_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_run_summary_html(run: dict) -> str:
    metadata = run.get("metadata", {}) or {}
    provider = str(metadata.get("provider") or "Unknown provider")
    model = str(metadata.get("model") or "Unknown model")
    stop_reason = str(run.get("stop_reason") or "unknown")
    user_input = str(run.get("user_input") or "")
    final_text = str(run.get("final_text") or "").strip()

    stop_variant = _stop_reason_variant(stop_reason)
    stop_badge = _build_badge_html(label=f"Stop · {stop_reason}", variant=stop_variant)
    steps_badge = _build_run_steps_badge_html(run)
    tools_badge = _build_run_tools_badge_html(run)
    provider_badge = _build_badge_html(label=f"Provider · {provider}", variant="info")
    model_badge = _build_badge_html(label=f"Model · {model}", variant="info")
    phase_badges = _build_run_phase_badges_html(run)

    return f"""
    <div class="app-panel">
        <div class="panel-title">Run</div>
        <div class="panel-subtitle">{html.escape(user_input) if user_input else 'No user input recorded.'}</div>
        <div class="status-row">{provider_badge}{model_badge}{steps_badge}{tools_badge}{stop_badge}</div>
        <div class="status-row">{phase_badges}</div>
        {_build_plain_response_panel_html(final_text, max_chars=3000) if final_text else ""}
    </div>
    """


def _trace_stage_display_priority(stage: str) -> int:
    """
    Order trace events to match the real runtime flow in the Admin panel.
    """
    normalized = str(stage or "").strip().lower()
    if normalized in {"compaction_check", "compaction"}:
        return 0
    if normalized in {"triage", "planning", "critique"}:
        return 1
    if normalized in {"provider_response", "tool_requested", "tool_result", "provider_error"}:
        return 2
    if normalized in {"reflection_gate", "reflection"}:
        return 3
    return 4


def _trace_stage_step_priority(stage: str) -> int:
    normalized = str(stage or "").strip().lower()
    if normalized == "provider_response":
        return 0
    if normalized == "tool_requested":
        return 1
    if normalized == "tool_result":
        return 2
    if normalized == "provider_error":
        return 3
    return 4


def _trace_step_sort_value(step: object) -> int:
    try:
        return int(step)
    except (TypeError, ValueError):
        return -1


def _order_trace_for_display(trace: list[dict]) -> list[dict]:
    """
    Reorder mixed orchestration/execution traces so the Admin panel reflects runtime order.
    """
    indexed_trace = list(enumerate(trace))
    ordered_pairs = sorted(
        indexed_trace,
        key=lambda pair: (
            _trace_stage_display_priority(str((pair[1] or {}).get("stage") or "")),
            _trace_step_sort_value((pair[1] or {}).get("step")),
            _trace_stage_step_priority(str((pair[1] or {}).get("stage") or "")),
            pair[0],
        ),
    )
    return [item for _, item in ordered_pairs]


def _build_trace_sections_html(trace: list[dict], *, run: dict | None = None) -> str:
    """
    Build the grouped trace UI without changing the recorded trace payload.
    """
    view_model = _build_trace_view_model(trace, run=run or {})
    sections: list[str] = []

    plan_items = view_model.get("plan", [])
    if plan_items:
        sections.append(
            _build_trace_section_html(
                title="Planning",
                badges=_build_plan_section_badges_html(plan_items),
                body_html="".join(_build_trace_block_html(item) for item in plan_items),
            )
        )

    execution_items = view_model.get("execution", [])
    if execution_items:
        sections.append(
            _build_trace_section_html(
                title="Execution",
                badges=_build_execution_section_badges_html(execution_items, run=run or {}),
                body_html="".join(_build_trace_block_html(item) for item in execution_items),
            )
        )

    reflection_items = view_model.get("reflection", [])
    if reflection_items:
        sections.append(
            _build_trace_section_html(
                title="Reflection",
                badges=_build_reflection_section_badges_html(reflection_items),
                body_html="".join(_build_trace_block_html(item) for item in reflection_items),
            )
        )

    return "".join(sections)


def _build_trace_view_model(trace: list[dict], *, run: dict | None = None) -> dict[str, list[dict]]:
    """
    Convert raw trace events into display groups while preserving raw event shape.
    """
    ordered_trace = _order_trace_for_display(trace)
    tool_requested_steps = {
        str(item.get("step"))
        for item in ordered_trace
        if isinstance(item, dict) and str(item.get("stage") or "") == "tool_requested"
    }
    plan_candidates: list[dict] = []
    execution_items: list[dict] = []
    reflection_candidates: list[dict] = []

    for item in ordered_trace:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "").strip()

        if stage == "provider_response":
            response = item.get("response", {}) or {}
            if response.get("tool_call") and str(item.get("step")) in tool_requested_steps:
                continue

        if stage in {"triage", "planning", "critique"}:
            plan_candidates.append(item)
            continue

        if stage in {"reflection_gate", "reflection"}:
            reflection_candidates.append(item)
            continue

        execution_items.append(item)

    return {
        "plan": plan_candidates if _has_plan_section(plan_candidates) else [],
        "execution": execution_items,
        "reflection": reflection_candidates if _has_reflection_section(reflection_candidates) else [],
    }


def _has_plan_section(items: list[dict]) -> bool:
    return any(
        isinstance(item, dict) and str(item.get("stage") or "") in {"planning", "critique"}
        for item in items
    )


def _has_reflection_section(items: list[dict]) -> bool:
    return any(
        isinstance(item, dict) and str(item.get("stage") or "") == "reflection"
        for item in items
    )


def _build_trace_section_html(*, title: str, badges: str, body_html: str) -> str:
    safe_title = html.escape(title)
    return (
        '<details class="trace-section">'
        '<summary class="trace-section-summary">'
        f'<span class="trace-section-title">{safe_title}</span>'
        f'<span class="trace-section-meta">{badges}</span>'
        '</summary>'
        f'<div class="trace-section-body">{body_html}</div>'
        '</details>'
    )


def _build_plan_section_badges_html(items: list[dict]) -> str:
    mode = "Used"
    provider_model = _first_provider_model_label(items)

    for item in items:
        if not isinstance(item, dict) or str(item.get("stage") or "") != "triage":
            continue
        decision = item.get("decision", {}) or {}
        candidate_mode = str(decision.get("planning_mode") or "").strip().lower()
        if candidate_mode in {"light", "deep"}:
            mode = candidate_mode.title()
            break

    return "".join(
        [
            _build_badge_html(label=f"Plan · {mode}", variant=_planning_mode_variant(mode)),
            _build_badge_html(label=provider_model, variant="info") if provider_model else "",
        ]
    )


def _build_execution_section_badges_html(items: list[dict], *, run: dict) -> str:
    step_values = {
        str(item.get("step"))
        for item in items
        if isinstance(item, dict) and str(item.get("step") or "").strip() not in {"", "—"}
    }
    tool_count = sum(
        1
        for item in items
        if isinstance(item, dict) and str(item.get("stage") or "") == "tool_requested"
    )
    stop_reason = str((run or {}).get("stop_reason") or "").strip()

    badges = [
        _build_badge_html(label=f"Execution · {len(step_values)} steps", variant="info"),
        _build_badge_html(label=f"{tool_count} tools", variant="info"),
    ]
    if stop_reason:
        badges.append(_build_badge_html(label=f"Stop · {stop_reason}", variant=_stop_reason_variant(stop_reason)))
    return "".join(badges)


def _build_reflection_section_badges_html(items: list[dict]) -> str:
    reflection_item = next(
        (
            item for item in items
            if isinstance(item, dict) and str(item.get("stage") or "") == "reflection"
        ),
        {},
    )
    reflection = reflection_item.get("reflection", {}) if isinstance(reflection_item, dict) else {}
    mutations_applied = int((reflection or {}).get("mutations_applied", 0) or 0)
    provider_model = _first_provider_model_label(items)

    if mutations_applied == 1:
        label = "Reflection · Applied 1 note"
    elif mutations_applied > 1:
        label = f"Reflection · Applied {mutations_applied} notes"
    else:
        label = "Reflection · Used"

    return "".join(
        [
            _build_badge_html(label=label, variant="info"),
            _build_badge_html(label=provider_model, variant="info") if provider_model else "",
        ]
    )


def _first_provider_model_label(items: list[dict]) -> str:
    for item in items:
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider") or "").strip()
        model = str(item.get("model") or "").strip()
        if provider and model:
            return f"{provider} · {model}"
        if provider:
            return provider
        if model:
            return model
    return ""


def _build_markdown_response_panel_html(
    text: str,
    *,
    label: str | None = None,
    max_chars: int = 2000,
) -> str:
    normalized_text = str(text or "").strip()
    if len(normalized_text) > max_chars:
        normalized_text = f"{normalized_text[:max_chars]}... [trimmed {len(normalized_text) - max_chars} chars]"

    label_html = (
        f'<div class="panel-subtitle">{html.escape(label)}</div>'
        if label
        else ""
    )
    return (
        f'{label_html}<div class="trace-markdown-panel">'
        f'<div class="trace-markdown">{_build_markdown_text_html(normalized_text or "No text returned.")}</div>'
        '</div>'
    )


def _build_plain_response_panel_html(
    text: str,
    *,
    label: str | None = None,
    max_chars: int = 2000,
) -> str:
    normalized_text = _format_assistant_text_for_trace(text)
    if len(normalized_text) > max_chars:
        normalized_text = f"{normalized_text[:max_chars]}... [trimmed {len(normalized_text) - max_chars} chars]"

    label_html = (
        f'<div class="panel-subtitle">{html.escape(label)}</div>'
        if label
        else ""
    )
    return (
        f'{label_html}<div class="trace-plain-text-panel">'
        f'<div class="trace-plain-text">{html.escape(normalized_text or "No text returned.")}</div>'
        '</div>'
    )


def _format_assistant_text_for_trace(text: str) -> str:
    """
    Convert assistant markdown into compact diagnostic plain text for Admin.
    """
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    in_code_block = False

    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if not in_code_block and _is_table_separator_line(stripped):
            continue

        if not in_code_block:
            line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
            if "|" in line:
                table_cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if len(table_cells) > 1:
                    line = " | ".join(table_cells)
            line = _strip_inline_markdown_for_trace(line)

        lines.append(line)

    compact_lines = [line for line in lines if line.strip()]

    return "\n".join(compact_lines).strip()


def _strip_inline_markdown_for_trace(text: str) -> str:
    rendered = re.sub(r"`([^`]+)`", r"\1", str(text or ""))
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"\1", rendered)
    rendered = re.sub(r"__([^_]+)__", r"\1", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", rendered)
    rendered = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", rendered)
    return rendered


def _build_markdown_text_html(text: str) -> str:
    """
    Render the Admin trace's compact assistant-markdown subset safely.
    """
    normalized_content = str(text or "").replace("\r\n", "\n")
    lines = normalized_content.split("\n")
    blocks: list[str] = []
    paragraph_lines: list[str] = []
    list_items: list[str] = []
    list_type: str | None = None
    in_code_block = False
    code_lines: list[str] = []
    table_lines: list[str] = []
    skip_table_separator_index: int | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph_html = _apply_inline_markdown("<br>".join(html.escape(line) for line in paragraph_lines))
        blocks.append(f"<p>{paragraph_html}</p>")
        paragraph_lines = []

    def flush_list() -> None:
        nonlocal list_items, list_type
        if not list_items or not list_type:
            return
        items_html = "".join(
            f"<li>{_apply_inline_markdown(html.escape(item))}</li>"
            for item in list_items
        )
        blocks.append(f"<{list_type}>{items_html}</{list_type}>")
        list_items = []
        list_type = None

    def flush_code_block() -> None:
        nonlocal code_lines
        blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
        code_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return
        table_html = _build_markdown_table_html(table_lines)
        if table_html:
            blocks.append(table_html)
        else:
            paragraph_lines.extend(table_lines)
            flush_paragraph()
        table_lines = []

    for index, line in enumerate(lines):
        if skip_table_separator_index == index:
            skip_table_separator_index = None
            continue

        trimmed = line.strip()

        if trimmed.startswith("```"):
            flush_paragraph()
            flush_list()
            flush_table()
            if in_code_block:
                flush_code_block()
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if not trimmed:
            flush_paragraph()
            flush_list()
            flush_table()
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.*)$", trimmed)
        if heading_match:
            flush_paragraph()
            flush_list()
            flush_table()
            level = len(heading_match.group(1))
            heading_text = _apply_inline_markdown(html.escape(heading_match.group(2)))
            blocks.append(f'<div class="trace-markdown-heading heading-{level}">{heading_text}</div>')
            continue

        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if not table_lines and _is_potential_table_line(line) and _is_table_separator_line(next_line):
            flush_paragraph()
            flush_list()
            table_lines.extend([line, next_line])
            skip_table_separator_index = index + 1
            continue

        if table_lines:
            if _is_potential_table_line(line):
                table_lines.append(line)
                continue
            flush_table()

        ordered_match = re.match(r"^\d+\.\s+(.*)$", trimmed)
        if ordered_match:
            flush_paragraph()
            if list_type and list_type != "ol":
                flush_list()
            list_type = "ol"
            list_items.append(ordered_match.group(1))
            continue

        unordered_match = re.match(r"^[-*]\s+(.*)$", trimmed)
        if unordered_match:
            flush_paragraph()
            if list_type and list_type != "ul":
                flush_list()
            list_type = "ul"
            list_items.append(unordered_match.group(1))
            continue

        flush_list()
        paragraph_lines.append(line)

    if in_code_block:
        flush_code_block()
    flush_paragraph()
    flush_list()
    flush_table()
    return "".join(blocks)


def _apply_inline_markdown(escaped_text: str) -> str:
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped_text)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", rendered)
    return rendered


def _is_table_separator_line(line: str) -> bool:
    trimmed = str(line or "").strip()
    if "|" not in trimmed:
        return False
    normalized = trimmed.strip("|")
    cells = [cell.strip() for cell in normalized.split("|")]
    return bool(cells) and all(re.match(r"^:?-{3,}:?$", cell) for cell in cells)


def _is_potential_table_line(line: str) -> bool:
    return "|" in str(line or "").strip()


def _split_table_cells(line: str) -> list[str]:
    trimmed = str(line or "").strip().strip("|")
    return [
        _apply_inline_markdown(html.escape(cell.strip()))
        for cell in trimmed.split("|")
    ]


def _build_markdown_table_html(table_lines: list[str]) -> str:
    if len(table_lines) < 2:
        return ""

    header_cells = _split_table_cells(table_lines[0])
    body_lines = table_lines[2:]
    thead_html = "<thead><tr>" + "".join(f"<th>{cell}</th>" for cell in header_cells) + "</tr></thead>"
    tbody_html = ""
    if body_lines:
        rows_html = []
        for line in body_lines:
            cells = _split_table_cells(line)
            rows_html.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        tbody_html = "<tbody>" + "".join(rows_html) + "</tbody>"

    return f'<div class="trace-markdown-table-wrapper"><table>{thead_html}{tbody_html}</table></div>'


# --- Inserted helper functions for trace rendering ---

def _extract_tool_call_parts(tool_call: object) -> tuple[str, object | None]:
    """
    Normalize a provider-returned tool call object into a tool name and arguments.

    This keeps trace rendering compact and stage-aware without requiring any
    upstream changes to how trace events are recorded.
    """
    if not isinstance(tool_call, dict):
        return ("unknown", None)

    tool_name = str(tool_call.get("tool_name") or tool_call.get("name") or "unknown")
    arguments = tool_call.get("arguments")
    return (tool_name, arguments)


def _build_trace_item_header_html(title: str, *, variant: str = "info") -> str:
    safe_title = html.escape(str(title or "").strip())
    safe_variant = html.escape(str(variant or "info").strip().lower())
    return f'<div class="trace-item-header {safe_variant}">{safe_title}</div>'


def _build_trace_kv_row_html(pairs: list[tuple[str, object]]) -> str:
    items: list[str] = []
    for label, value in pairs:
        safe_label = html.escape(str(label or "").strip())
        safe_value = html.escape(str(value if value is not None else "").strip())
        if not safe_label or not safe_value:
            continue
        items.append(
            '<span class="trace-kv">'
            f'<span class="trace-kv-label">{safe_label}</span>'
            f'<span class="trace-kv-value">{safe_value}</span>'
            '</span>'
        )
    if not items:
        return ""
    return f'<div class="trace-kv-row">{"".join(items)}</div>'


def _format_trace_stage_title(*, step: object, stage_label: str) -> str:
    label = " ".join(str(stage_label or "").replace("_", " ").split()).strip()
    title = label.title() if label else "Trace Event"
    normalized_step = str(step).strip()
    if normalized_step and normalized_step != "—":
        return f"Step {normalized_step} · {title}"
    return title


def _build_provider_tool_request_trace_html(item: dict, *, step: object) -> str:
    tool_call = item.get("response", {}) or {}
    extracted_tool_call = tool_call.get("tool_call")
    tool_name, _arguments = _extract_tool_call_parts(extracted_tool_call)

    header = _build_trace_item_header_html(
        _format_trace_stage_title(step=step, stage_label="provider requested tool")
    )
    detail_html = _build_code_panel_html(tool_name)
    return f'<div class="trace-item">{header}{detail_html}</div>'


def _build_stage_badge_label(*, step: object, stage_label: str) -> str:
    """
    Build a trace badge label, omitting the placeholder step marker when absent.
    """
    normalized_step = str(step).strip()
    if not normalized_step or normalized_step == "—":
        return stage_label
    return f"Step {normalized_step} · {stage_label}"


def _build_tool_requested_trace_html(item: dict, *, step: object) -> str:
    tool_name = str(item.get("tool_name") or "unknown")
    raw_tool_call = item.get("raw_tool_call")

    header = _build_trace_item_header_html(
        _format_trace_stage_title(step=step, stage_label="tool requested")
    )

    detail_lines: list[str] = [
        _build_code_panel_html(tool_name)
    ]

    if raw_tool_call not in (None, {}, []):
        detail_lines.append(
            _build_code_panel_html(str(raw_tool_call), label="Raw tool call")
        )

    return f'<div class="trace-item">{header}{"".join(detail_lines)}</div>'


def _build_tool_result_trace_html(item: dict, *, step: object) -> str:
    success = bool(item.get("success", False))
    tool_name = str(item.get("tool_name") or "unknown")
    output = item.get("output")
    error = item.get("error")

    header = _build_trace_item_header_html(
        _format_trace_stage_title(
            step=step,
            stage_label=f"tool {'success' if success else 'failed'}",
        ),
        variant="success" if success else "error",
    )

    detail_lines: list[str] = [
        _build_code_panel_html(tool_name)
    ]

    if error:
        detail_lines.append(
            _build_code_panel_html(str(error), label="Error")
        )

    if output not in (None, {}, []):
        detail_lines.append(
            _build_code_panel_html(str(output)[:2000], label="Output")
        )

    return f'<div class="trace-item">{header}{"".join(detail_lines)}</div>'


def _build_provider_text_response_trace_html(item: dict, *, step: object) -> str:
    response = item.get("response", {}) or {}
    text_preview = str(response.get("text") or "").strip()

    header = _build_trace_item_header_html(
        _format_trace_stage_title(step=step, stage_label="provider response")
    )
    detail_html = _build_plain_response_panel_html(text_preview, max_chars=2000)
    return f'<div class="trace-item">{header}{detail_html}</div>'


def _format_planning_notes_tool_activity(activity: dict) -> str:
    tool_name = str(activity.get("tool_name") or "unknown").strip()
    status = str(activity.get("status") or "unknown").strip()
    if not tool_name:
        return ""

    line = f"- {tool_name} [{status}]"
    query = str(activity.get("query") or "").strip()
    note_id = str(activity.get("note_id") or "").strip()
    file_name = str(activity.get("file_name") or "").strip()
    result_count = activity.get("result_count")
    file_count = activity.get("file_count")
    found = activity.get("found")
    message = str(activity.get("message") or "").strip()

    if query:
        line += f" · query={query}"
    if note_id:
        line += f" · note_id={note_id}"
    if file_name:
        line += f" · file={file_name}"
    if isinstance(result_count, int):
        line += f" · {result_count} result{'s' if result_count != 1 else ''}"
    if isinstance(file_count, int):
        line += f" · {file_count} file{'s' if file_count != 1 else ''}"
    if isinstance(found, bool):
        line += f" · {'found' if found else 'not found'}"
    if message:
        line += f" · {message}"

    return line


def _build_triage_trace_html(item: dict, *, step: object) -> str:
    decision = item.get("decision", {}) or {}
    reason = str(decision.get("reason") or "No reason recorded.")
    source = str(decision.get("source") or "unknown")

    header = _build_trace_item_header_html(_format_trace_stage_title(step=step, stage_label="triage"))

    detail_lines: list[str] = [
        _build_trace_kv_row_html(
            [
                ("Source", source),
            ]
        ),
        _build_code_panel_html(reason, label="Reason"),
    ]

    return f'<div class="trace-item">{header}{"".join(detail_lines)}</div>'


def _build_planning_trace_html(item: dict, *, step: object) -> str:
    plan = item.get("plan", {}) or {}
    summary = str(plan.get("summary") or "No plan summary recorded.").strip()
    execution_guidance = plan.get("execution_guidance", [])
    missing_context = plan.get("missing_context", [])
    notes_tool_activity = plan.get("notes_tool_activity", [])

    if isinstance(execution_guidance, list):
        guidance_text = "\n".join(
            f"- {str(line).strip()}"
            for line in execution_guidance
            if str(line).strip()
        ).strip()
    else:
        guidance_text = ""

    if isinstance(missing_context, list):
        missing_context_text = "\n".join(
            f"- {str(line).strip()}"
            for line in missing_context
            if str(line).strip()
        ).strip()
    else:
        missing_context_text = ""

    notes_context = plan.get("notes_context", [])
    if isinstance(notes_context, list):
        notes_context_text = "\n".join(
            f"- {str(note.get('file_name') or '').strip()}/{str(note.get('note_id') or '').strip()}: {str(note.get('statement') or '').strip()}"
            for note in notes_context
            if isinstance(note, dict)
            and str(note.get("note_id") or "").strip()
            and str(note.get("statement") or "").strip()
        ).strip()
    else:
        notes_context_text = ""

    if isinstance(notes_tool_activity, list):
        notes_tool_activity_text = "\n".join(
            _format_planning_notes_tool_activity(activity)
            for activity in notes_tool_activity
            if isinstance(activity, dict) and _format_planning_notes_tool_activity(activity)
        ).strip()
    else:
        notes_tool_activity_text = ""

    header = _build_trace_item_header_html(_format_trace_stage_title(step=step, stage_label="planning"))

    detail_lines: list[str] = [
        _build_code_panel_html(summary, label="Summary"),
    ]

    if guidance_text:
        detail_lines.append(
            _build_code_panel_html(guidance_text, label="Execution guidance")
        )

    if missing_context_text:
        detail_lines.append(
            _build_code_panel_html(missing_context_text, label="Missing context")
        )

    if notes_context_text:
        detail_lines.append(
            _build_code_panel_html(notes_context_text, label="Notes context")
        )

    if notes_tool_activity_text:
        detail_lines.append(
            _build_code_panel_html(notes_tool_activity_text, label="Notes tool activity")
        )

    return f'<div class="trace-item">{header}{"".join(detail_lines)}</div>'


def _build_critique_trace_html(item: dict, *, step: object) -> str:
    critique = item.get("critique", {}) or {}
    summary = str(critique.get("summary") or "No critique summary recorded.").strip()
    issues = critique.get("issues", [])
    revised_guidance = critique.get("revised_execution_guidance", [])
    notes_context = critique.get("notes_context", [])

    if isinstance(issues, list):
        issues_text = "\n".join(
            f"- {str(line).strip()}"
            for line in issues
            if str(line).strip()
        ).strip()
    else:
        issues_text = ""

    if isinstance(revised_guidance, list):
        revised_guidance_text = "\n".join(
            f"- {str(line).strip()}"
            for line in revised_guidance
            if str(line).strip()
        ).strip()
    else:
        revised_guidance_text = ""

    if isinstance(notes_context, list):
        notes_context_text = "\n".join(
            f"- {str(note.get('file_name') or '').strip()}/{str(note.get('note_id') or '').strip()}: {str(note.get('statement') or '').strip()}"
            for note in notes_context
            if isinstance(note, dict)
            and str(note.get("note_id") or "").strip()
            and str(note.get("statement") or "").strip()
        ).strip()
    else:
        notes_context_text = ""

    header = _build_trace_item_header_html(_format_trace_stage_title(step=step, stage_label="critique"))

    detail_lines: list[str] = [
        _build_code_panel_html(summary, label="Summary"),
    ]

    if issues_text:
        detail_lines.append(
            _build_code_panel_html(issues_text, label="Issues")
        )

    if revised_guidance_text:
        detail_lines.append(
            _build_code_panel_html(revised_guidance_text, label="Revised guidance")
        )

    if notes_context_text:
        detail_lines.append(
            _build_code_panel_html(notes_context_text, label="Notes context")
        )

    return f'<div class="trace-item">{header}{"".join(detail_lines)}</div>'


def _build_reflection_gate_trace_html(item: dict, *, step: object) -> str:
    decision = item.get("decision", {}) or {}
    should_reflect = bool(decision.get("should_reflect", False))
    forced = bool(decision.get("forced", False))
    reason = str(decision.get("reason") or "No reason recorded.").strip()
    source = str(decision.get("source") or "unknown").strip()
    tool_count = int(item.get("tool_count", 0))

    header = _build_trace_item_header_html(_format_trace_stage_title(step=step, stage_label="reflection gate"))

    detail_lines: list[str] = [
        _build_trace_kv_row_html(
            [
                ("Decision", "reflect" if should_reflect else "skip"),
                ("Mode", "forced" if forced else "optional"),
                ("Tools", tool_count),
            ]
        ),
        _build_trace_kv_row_html(
            [
                ("Reason", reason),
                ("Source", source),
            ]
        ),
    ]

    return f'<div class="trace-item">{header}{"".join(detail_lines)}</div>'


def _build_reflection_trace_html(item: dict, *, step: object) -> str:
    reflection = item.get("reflection", {}) or {}
    summary = str(reflection.get("summary") or "No reflection summary recorded.").strip()
    lessons = reflection.get("lessons", [])
    tool_activity = reflection.get("tool_activity", [])
    note_files_touched = reflection.get("note_files_touched", [])
    mutations_applied = int(reflection.get("mutations_applied", 0))

    if isinstance(lessons, list):
        lessons_text = "\n".join(
            f"- {str(line).strip()}"
            for line in lessons
            if str(line).strip()
        ).strip()
    else:
        lessons_text = ""

    if isinstance(tool_activity, list):
        tool_activity_lines: list[str] = []
        for activity in tool_activity:
            if not isinstance(activity, dict):
                continue
            line = (
                f"- {str(activity.get('tool_name') or 'unknown')}"
                f" [{str(activity.get('status') or 'unknown')}]"
            )
            file_name = str(activity.get("file_name") or "").strip()
            message = str(activity.get("message") or "").strip()
            if file_name:
                line += f" · {file_name}"
            if message:
                line += f" · {message}"
            tool_activity_lines.append(line)
        tool_activity_text = "\n".join(tool_activity_lines).strip()
    else:
        tool_activity_text = ""

    if isinstance(note_files_touched, list):
        touched_text = "\n".join(
            f"- {str(file_name).strip()}"
            for file_name in note_files_touched
            if str(file_name).strip()
        ).strip()
    else:
        touched_text = ""

    header = _build_trace_item_header_html(_format_trace_stage_title(step=step, stage_label="reflection"))

    detail_lines: list[str] = [
        _build_code_panel_html(summary, label="Summary"),
        _build_code_panel_html(str(mutations_applied), label="Mutations applied"),
    ]

    if lessons_text:
        detail_lines.append(
            _build_code_panel_html(lessons_text, label="Lessons")
        )

    if touched_text:
        detail_lines.append(
            _build_code_panel_html(touched_text, label="Note files touched")
        )

    if tool_activity_text:
        detail_lines.append(
            _build_code_panel_html(tool_activity_text, label="Note tool activity")
        )

    return f'<div class="trace-item">{header}{"".join(detail_lines)}</div>'




def _build_trace_block_html(item: dict) -> str:
    step = item.get("step", "—")
    stage = str(item.get("stage") or "unknown")

    if stage == "provider_response":
        response = item.get("response", {}) or {}
        if response.get("tool_call"):
            return _build_provider_tool_request_trace_html(item, step=step)
        return _build_provider_text_response_trace_html(item, step=step)

    if stage == "tool_requested":
        return _build_tool_requested_trace_html(item, step=step)

    if stage == "tool_result":
        return _build_tool_result_trace_html(item, step=step)

    if stage == "triage":
        return _build_triage_trace_html(item, step=step)

    if stage == "planning":
        return _build_planning_trace_html(item, step=step)

    if stage == "critique":
        return _build_critique_trace_html(item, step=step)

    if stage == "reflection_gate":
        return _build_reflection_gate_trace_html(item, step=step)

    if stage == "reflection":
        return _build_reflection_trace_html(item, step=step)

    title = _format_trace_stage_title(step=step, stage_label=stage)
    primary = stage
    variant = _trace_stage_variant(stage)

    if stage == "provider_error":
        title = _format_trace_stage_title(step=step, stage_label="provider error")
        primary = str(item.get("error") or "provider error")

    header = _build_trace_item_header_html(title, variant=variant)

    raw_tool_call = item.get("raw_tool_call")
    error = item.get("error")
    output = item.get("output")

    detail_lines: list[str] = []

    if stage == "provider_error":
        detail_lines.append(_build_code_panel_html(primary))

    if raw_tool_call not in (None, {}, []):
        detail_lines.append(
            _build_code_panel_html(str(raw_tool_call), label="Raw tool call")
        )

    if error:
        detail_lines.append(
            _build_code_panel_html(str(error), label="Error")
        )

    if output not in (None, {}, []):
        detail_lines.append(
            _build_code_panel_html(str(output)[:2000], label="Output")
        )

    return f'<div class="trace-item">{header}{"".join(detail_lines)}</div>'



def _build_code_panel_html(value: str, *, label: str | None = None) -> str:
    """
    Render a reusable code-style panel using fully custom HTML containers.

    This intentionally avoids native pre/code styling so the app theme can own
    the visual treatment consistently across light and dark modes.
    """
    safe_value = html.escape(value)
    label_html = (
        f'<div class="panel-subtitle">{html.escape(label)}</div>'
        if label
        else ""
    )
    return (
        f'{label_html}<div class="code-panel"><div class="code-panel-text">{safe_value}</div></div>'
    )



def _stop_reason_variant(stop_reason: str) -> str:
    normalized = stop_reason.strip().lower()
    if normalized == "completed":
        return "success"
    if normalized == "stopped":
        return "neutral"
    if normalized in {"provider_error", "max_steps"}:
        return "error"
    return "info"



def _trace_stage_variant(stage: str) -> str:
    normalized = stage.strip().lower()
    if normalized in {"tool_result"}:
        return "success"
    if normalized in {"provider_error"}:
        return "error"
    return "info"
