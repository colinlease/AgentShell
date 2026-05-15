

"""Results panel for the ML Workbench app."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from app.workspace_apps.ml_workbench.services.export_service import (
    generate_excel_export_workbook,
)
from app.workspace_apps.ml_workbench.services.modeling_service import (
    build_results_comparison_summary,
    get_modeling_problem_type,
)
from app.workspace_apps.ml_workbench.ui.layout import (
    create_surface_panel,
    render_badge_row,
    render_status_message,
)


RESULTS_EMPTY_STATE_MESSAGE = (
    "Train at least one candidate model in the Model tab to see comparison results here."
)


_METRIC_EXCLUSION_NAMES = {
    "row_count",
    "train_row_count",
    "test_row_count",
    "cv_folds",
    "split_strategy",
    "test_size",
    "random_seed",
    "rebalancing_applied",
    "train_class_counts_original",
    "train_class_counts_rebalanced",
    "cv_rebalancing_summary",
    "positive_class_label",
    "classification_threshold_policy",
    "classification_threshold_objective",
    "classification_threshold_source",
    "classification_threshold_used",
    "classification_threshold_manual_value",
    "classification_threshold_optimization_details",
    "cv_classification_threshold_summary",
}


def _escape(value: object) -> str:
    """Return an HTML-safe string."""
    return html.escape("" if value is None else str(value))



def _get_completed_candidates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only completed candidate summaries."""
    candidates = summary.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if str(candidate.get("status", "")).strip().lower() == "completed"]



def _get_best_candidate(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Return the best candidate summary, if available."""
    for candidate in _get_completed_candidates(summary):
        if bool(candidate.get("is_best_candidate", False)):
            return candidate
    return None



def _format_metric_value(value: object) -> str:
    """Return a compact display string for metric values."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)



def _metric_label(metric_name: object) -> str:
    """Return a user-friendly metric label."""
    return str(metric_name or "Metric").replace("_", " ").title()

def _ordered_candidates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return completed candidates ordered for display."""
    return sorted(
        _get_completed_candidates(summary),
        key=lambda candidate: (
            not bool(candidate.get("is_best_candidate", False)),
            str(candidate.get("candidate_label", "")),
        ),
    )



def _comparison_metric_names(candidates: list[dict[str, Any]]) -> list[str]:
    """Return a stable ordered list of comparison metrics to show across cards."""
    ordered_metric_names: list[str] = []
    seen_metric_names: set[str] = set()

    for candidate in candidates:
        primary_metric_name = str(candidate.get("primary_metric_name") or "").strip()
        if primary_metric_name and primary_metric_name not in seen_metric_names:
            ordered_metric_names.append(primary_metric_name)
            seen_metric_names.add(primary_metric_name)

        metrics = candidate.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric_name in metrics:
            metric_name_text = str(metric_name).strip()
            if (
                not metric_name_text
                or metric_name_text in seen_metric_names
                or metric_name_text in _METRIC_EXCLUSION_NAMES
            ):
                continue
            ordered_metric_names.append(metric_name_text)
            seen_metric_names.add(metric_name_text)
    return ordered_metric_names



def _candidate_metric_rows(candidate: dict[str, Any], metric_names: list[str]) -> list[tuple[str, str]]:
    """Return display-ready metric rows for one candidate."""
    metrics = candidate.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    primary_metric_name = str(candidate.get("primary_metric_name") or "").strip()
    primary_metric_value = candidate.get("primary_metric_value")
    rows: list[tuple[str, str]] = []
    for metric_name in metric_names:
        if metric_name == primary_metric_name:
            metric_value = primary_metric_value
        else:
            metric_value = metrics.get(metric_name)
        if metric_value is None:
            continue
        rows.append((_metric_label(metric_name), _format_metric_value(metric_value)))
    return rows



def _overall_evaluation_mode(summary: dict[str, Any]) -> str:
    """Return one friendly evaluation-mode summary for the header."""
    completed_candidates = _get_completed_candidates(summary)
    modes = {
        str(candidate.get("evaluation_mode", "")).strip()
        for candidate in completed_candidates
        if str(candidate.get("evaluation_mode", "")).strip()
    }
    if not modes:
        return "Not available"
    if len(modes) == 1:
        return next(iter(modes))
    return "Mixed"



def _render_results_header(summary: dict[str, Any]) -> None:
    """Render the top results summary panel."""
    completed_candidates = _get_completed_candidates(summary)
    best_candidate = _get_best_candidate(summary)
    problem_type = str(get_modeling_problem_type() or "Not set").replace("_", " ").title()
    best_metric_name = _metric_label(best_candidate.get("primary_metric_name")) if best_candidate else "Not available"
    evaluation_mode = _overall_evaluation_mode(summary)

    panel = create_surface_panel(
        title="Results",
        subtitle="Compare completed candidate runs and review the strongest model.",
    )
    with panel:
        badges = [
            f"Problem · {problem_type}",
            f"Completed Candidates · {len(completed_candidates)}",
            f"Best Metric · {best_metric_name}",
            f"Evaluation · {evaluation_mode}",
        ]
        render_badge_row(badges, variant="info")



def _render_candidate_comparison_cards(summary: dict[str, Any]) -> None:
    """Render one unified comparison band for completed candidates."""
    ordered_candidates = _ordered_candidates(summary)
    if not ordered_candidates:
        return

    metric_names = _comparison_metric_names(ordered_candidates)

    panel = create_surface_panel(
        title="Candidate Comparison",
        subtitle="Review all completed candidate runs. The strongest candidate is highlighted and shown first.",
    )
    with panel:
        cards_html: list[str] = []
        for candidate in ordered_candidates:
            candidate_label = _escape(candidate.get("candidate_label", "Candidate"))
            evaluation_mode = _escape(candidate.get("evaluation_mode", "Not available"))
            status_text = _escape(str(candidate.get("status", "unknown")).replace("_", " ").title())
            predictor_count = _escape(int(candidate.get("predictor_count", 0)))
            best_badge_html = ""
            best_class_name = ""
            if bool(candidate.get("is_best_candidate", False)):
                best_badge_html = '<div class="mlw-badge success">Best Candidate</div>'
                best_class_name = " mlw-result-card--best"

            metric_rows = _candidate_metric_rows(candidate, metric_names)
            metrics_html = ""
            if metric_rows:
                metrics_html = "".join(
                    (
                        '<div class="mlw-result-card__metric">'
                        f'<div class="mlw-result-card__metric-label">{_escape(metric_label)}</div>'
                        f'<div class="mlw-result-card__metric-value">{_escape(metric_value)}</div>'
                        "</div>"
                    )
                    for metric_label, metric_value in metric_rows
                )
                metrics_html = f'<div class="mlw-result-card__metrics">{metrics_html}</div>'

            cards_html.append(
                (
                    f'<div class="mlw-result-card{best_class_name}">'
                    '<div class="mlw-result-card__header">'
                    '<div class="mlw-result-card__title-group">'
                    f'<div class="mlw-result-card__title">{candidate_label}</div>'
                    f'<div class="mlw-result-card__subtitle">{evaluation_mode}</div>'
                    "</div>"
                    f"{best_badge_html}"
                    "</div>"
                    '<div class="mlw-result-card__meta">'
                    f'<div class="mlw-badge info">Status · {status_text}</div>'
                    f'<div class="mlw-badge info">Predictors · {predictor_count}</div>'
                    "</div>"
                    f"{metrics_html}"
                    "</div>"
                )
            )

        st.markdown(
            f'<div class="mlw-results-band">{"".join(cards_html)}</div>',
            unsafe_allow_html=True,
        )



def _render_results_export_panel(summary: dict[str, Any]) -> None:
    """Render the results export panel below the candidate comparison."""
    completed_candidates = _get_completed_candidates(summary)
    workbook_bytes, workbook_filename = generate_excel_export_workbook()
    comparison_metric_name = str(summary.get("comparison_metric_name") or "").strip()
    comparison_metric_label = _metric_label(comparison_metric_name) if comparison_metric_name else "Not available"

    panel = create_surface_panel(
        title="Export Results",
    )
    with panel:
        render_badge_row(
            [
                f"Completed Candidates · {len(completed_candidates)}",
                f"Comparison Metric · {comparison_metric_label}",
                "Format · Excel Workbook (.xlsx)",
            ],
            variant="info",
        )
        st.download_button(
            label="Download Results",
            data=workbook_bytes,
            file_name=workbook_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False,
            key="mlw_results_download_excel",
        )
        st.caption(
            "Includes summary, candidate comparison, candidate settings, predictor columns, "
            "shared preprocessing, feature engineering, and raw JSON sheets."
        )


def render_results_panel() -> None:
    """Render the results stage panel for the ML Workbench app."""
    summary = build_results_comparison_summary()
    completed_candidates = _get_completed_candidates(summary)

    if not completed_candidates:
        panel = create_surface_panel(
            title="No results yet",
            subtitle="Run at least one candidate model to populate the Results tab.",
        )
        with panel:
            render_status_message(RESULTS_EMPTY_STATE_MESSAGE, variant="info")
        return

    _render_results_header(summary)
    _render_candidate_comparison_cards(summary)
    _render_results_export_panel(summary)
