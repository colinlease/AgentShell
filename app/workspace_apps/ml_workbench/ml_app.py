"""Shell-facing ML Workbench workspace app.

This module provides the AgentShell workspace app adapter for the ML
Workbench. It is intentionally thin: it initializes app state, exposes
published shell context, exposes named dataset access for shell tools, and
renders a compact v1 scaffold UI that can be expanded with modular panels.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.workspace_apps.base import BaseWorkspaceApp
from app.workspace_apps.ml_workbench.constants import (
    APP_ID,
    APP_LABEL,
    APP_TYPE,
    STAGE_FEATURES,
    STAGE_MODELING,
    STAGE_PREPROCESS,
    STAGE_PROFILE,
    STAGE_RESULTS,
)
from app.workspace_apps.ml_workbench.manifest import get_app_description, get_app_manifest

DISPLAY_STAGE_ORDER = [
    STAGE_PROFILE,
    STAGE_PREPROCESS,
    STAGE_FEATURES,
    STAGE_MODELING,
    STAGE_RESULTS,
]

DISPLAY_STAGE_LABELS = {
    STAGE_PROFILE: "Data",
    STAGE_PREPROCESS: "Prepare",
    STAGE_FEATURES: "Features",
    STAGE_MODELING: "Models",
    STAGE_RESULTS: "Results",
}
from app.workspace_apps.ml_workbench.services.context_service import (
    build_published_data_context,
    build_published_ui_state,
)
from app.workspace_apps.ml_workbench.assets.css import inject_ml_workbench_css
from app.workspace_apps.ml_workbench.ui.layout import (
    render_badge_row,
    render_compact_title,
    render_info_card,
    render_metric_row,
    render_section_header,
)
from app.workspace_apps.ml_workbench.ui.profile_panel import (
    render_dataset_preview_panel,
    render_modeling_setup_panel,
)
from app.workspace_apps.ml_workbench.ui.preprocess_panel import render_preprocess_panel
from app.workspace_apps.ml_workbench.ui.features_panel import render_features_panel
from app.workspace_apps.ml_workbench.ui.modeling_panel import render_modeling_panel
from app.workspace_apps.ml_workbench.ui.results_panel import render_results_panel
from app.workspace_apps.ml_workbench.ui.upload_panel import render_upload_panel
from app.workspace_apps.ml_workbench.services.dataset_service import (
    get_workspace_dataset_object,
    has_loaded_dataset,
)
from app.workspace_apps.ml_workbench.state import get_app_state, initialize_state
from app.workspace_apps.ml_workbench.tools.factory import build_ml_workbench_tools


class MLWorkbenchApp(BaseWorkspaceApp):
    """Workspace app adapter for the ML Workbench."""

    def __init__(self, *, hosted: bool = False) -> None:
        self._hosted = bool(hosted)

    @property
    def app_id(self) -> str:
        return APP_ID

    @property
    def app_label(self) -> str:
        return APP_LABEL

    @property
    def app_type(self) -> str:
        return APP_TYPE

    def initialize_state(self) -> None:
        """Initialize app session state and artifact containers."""
        initialize_state()

    def render(self) -> None:
        """Render the standalone ML Workbench experience."""
        self.initialize_state()
        inject_ml_workbench_css(hosted=self._hosted)

        if not has_loaded_dataset():
            self._render_landing_view()
            return

        self._ensure_valid_loaded_stage()
        self._render_header()
        self._render_stage_selector()
        self._render_stage_panel()

    def get_ui_state(self) -> dict[str, Any]:
        """Return compact shell-facing UI state."""
        return build_published_ui_state()

    def get_data_context(self) -> dict[str, Any]:
        """Return shell-facing data context for the current workspace."""
        return build_published_data_context()

    def get_dataset_object(self, dataset_name: str | None = None) -> Any | None:
        """Return the full underlying dataset object for shell/tool access."""
        return get_workspace_dataset_object(dataset_name=dataset_name)

    def get_tools(self) -> list[Any]:
        """Return ML Workbench app-specific tools for the active workspace app."""
        return build_ml_workbench_tools()

    def describe(self) -> str:
        """Return a short app description for future shell use."""
        return get_app_description()

    def get_manifest(self) -> dict[str, Any]:
        """Return the canonical app manifest."""
        return get_app_manifest()

    def _render_header(self) -> None:
        """Render a compact workspace header for the loaded state."""
        state = get_app_state()
        data_context = build_published_data_context()
        active_dataset_name = data_context.get("active_dataset_name") or "Not set"

        rows_value = "Not set"
        columns_value = "Not set"
        for dataset in data_context.get("datasets", []):
            if dataset.get("name") == active_dataset_name:
                rows_value = f"{int(dataset.get('rows', 0)):,}"
                columns_value = f"{int(dataset.get('columns', 0)):,}"
                break

        problem_type = state.get("problem_type") or "Not set"
        target_column = state.get("target_column") or "Not set"

        render_compact_title(self.describe())
        render_metric_row(
            [
                {
                    "label": "Dataset",
                    "value": self._display_dataset_name(active_dataset_name),
                },
                {
                    "label": "Rows",
                    "value": rows_value,
                },
                {
                    "label": "Columns",
                    "value": columns_value,
                },
                {
                    "label": "Problem",
                    "value": str(problem_type).title()
                    if problem_type != "Not set"
                    else "Not set",
                },
                {
                    "label": "Target",
                    "value": str(target_column),
                },
            ]
        )
        badge_labels = [
            f"Current Section · {DISPLAY_STAGE_LABELS.get(state.get('app_stage', STAGE_PROFILE), 'Data')}",
            f"Active Dataset · {self._display_dataset_name(active_dataset_name)}",
        ]
        if state.get("loaded_file_name"):
            badge_labels.append(f"File · {state['loaded_file_name']}")
        render_badge_row(badge_labels, variant="info")

    def _render_stage_selector(self) -> None:
        """Render a compact horizontal workflow navigator."""
        state = get_app_state()
        current_stage = state.get("app_stage", STAGE_PROFILE)
        if current_stage not in DISPLAY_STAGE_ORDER:
            current_stage = STAGE_PROFILE
            state["app_stage"] = current_stage

        render_section_header(
            title="Workflow",
            subtitle="Move freely between sections as you prepare data, engineer features, train models, and review results.",
        )
        st.markdown(
            f'<div class="mlw-stage-selector mlw-stage-selector--{current_stage}"></div>',
            unsafe_allow_html=True,
        )
        columns = st.columns(len(DISPLAY_STAGE_ORDER))
        selected_stage = current_stage

        for column, stage in zip(columns, DISPLAY_STAGE_ORDER):
            label = DISPLAY_STAGE_LABELS[stage]
            button_type = "primary" if stage == selected_stage else "secondary"
            with column:
                if st.button(
                    label,
                    key=f"ml_workbench_stage_button_{stage}",
                    use_container_width=True,
                    type=button_type,
                ):
                    selected_stage = stage

        if selected_stage != current_stage:
            state["app_stage"] = selected_stage
            st.rerun()

    def _render_stage_panel(self) -> None:
        """Render the current section of the loaded workflow."""
        current_stage = get_app_state().get("app_stage", STAGE_PROFILE)

        if current_stage == STAGE_PROFILE:
            data_context = build_published_data_context()
            active_dataset_name = data_context.get("active_dataset_name")
            active_summary: dict[str, Any] | None = None
            for dataset in data_context.get("datasets", []):
                if dataset.get("name") == active_dataset_name:
                    active_summary = dataset
                    break

            if active_summary is not None:
                render_modeling_setup_panel(active_summary)
            render_dataset_preview_panel()
        elif current_stage == STAGE_PREPROCESS:
            render_preprocess_panel()
        elif current_stage == STAGE_FEATURES:
            render_features_panel()
        elif current_stage == STAGE_MODELING:
            render_modeling_panel()
        elif current_stage == STAGE_RESULTS:
            render_results_panel()
        else:
            data_context = build_published_data_context()
            active_dataset_name = data_context.get("active_dataset_name")
            active_summary: dict[str, Any] | None = None
            for dataset in data_context.get("datasets", []):
                if dataset.get("name") == active_dataset_name:
                    active_summary = dataset
                    break

            if active_summary is not None:
                render_modeling_setup_panel(active_summary)
            render_dataset_preview_panel()

    def _render_landing_view(self) -> None:
        """Render the standalone landing state before a dataset is loaded."""
        render_compact_title(self.describe())
        render_upload_panel()

    def _ensure_valid_loaded_stage(self) -> None:
        """Normalize the active stage after a dataset has been loaded."""
        state = get_app_state()
        current_stage = state.get("app_stage")
        if current_stage not in DISPLAY_STAGE_ORDER:
            state["app_stage"] = STAGE_PROFILE

    @staticmethod
    def _display_dataset_name(dataset_name: str) -> str:
        """Convert internal dataset artifact names into user-friendly labels."""
        display_map = {
            "raw_dataset": "Raw Data",
            "working_dataset": "Working Data",
            "model_input_dataset": "Model Input Data",
        }
        return display_map.get(dataset_name, dataset_name.replace("_", " ").title())
