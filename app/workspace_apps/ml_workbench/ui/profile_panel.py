"""Profile and dataset preview panel helpers for the ML Workbench UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from app.workspace_apps.ml_workbench.ui.layout import (
    create_plain_group,
    create_surface_panel,
    render_status_message,
)

from app.workspace_apps.ml_workbench.constants import (
    DEFAULT_PREVIEW_ROW_LIMIT,
    EMPTY_STATE_NO_DATASET,
    PROBLEM_TYPE_CLASSIFICATION,
    PROBLEM_TYPE_REGRESSION,
)
from app.workspace_apps.ml_workbench.services.dataset_service import (
    DatasetLoadError,
    get_dataset_copy,
    dataset_summary,
    get_active_dataset_name,
    get_available_dataset_names,
    get_dataset_preview,
    has_loaded_dataset,
)
from app.workspace_apps.ml_workbench.state import (
    get_app_state,
    reconcile_column_state,
    set_active_dataset_name,
    set_state_value,
    update_ui_state,
)


# Helper mappings and functions
DATASET_DISPLAY_NAMES = {
    "raw_dataset": "Raw Data",
    "working_dataset": "Working Data",
    "model_input_dataset": "Model Input Data",
}

PROBLEM_TYPE_OPTIONS = {
    "Not set": None,
    "Classification": PROBLEM_TYPE_CLASSIFICATION,
    "Regression": PROBLEM_TYPE_REGRESSION,
}


def _display_dataset_name(dataset_name: str) -> str:
    """Convert an internal dataset artifact name into a user-friendly label."""
    return DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name.replace("_", " ").title())


def _positive_class_options(*, dataset_name: str | None, target_column: str | None) -> list[object]:
    """Return binary target label options for positive-class selection."""
    if not dataset_name or not target_column:
        return []
    try:
        source_df = get_dataset_copy(dataset_name)
    except Exception:
        return []
    if target_column not in source_df.columns:
        return []
    unique_values = list(pd.Series(source_df[target_column]).dropna().unique())
    if len(unique_values) != 2:
        return []
    return unique_values


def render_modeling_setup_panel(summary: dict) -> None:
    """Render the modeling setup controls as a standalone panel."""
    modeling_panel = create_surface_panel(
        title="Modeling Setup",
        subtitle=(
            "Define the modeling objective and assign roles to fields."
        ),
    )
    with modeling_panel:
        state = get_app_state()
        column_names = list(summary.get("column_names", []))
        active_dataset_name = get_active_dataset_name(default_to_working=True)

        current_problem_type = state.get("problem_type")
        current_problem_label = "Not set"
        for label, value in PROBLEM_TYPE_OPTIONS.items():
            if value == current_problem_type:
                current_problem_label = label
                break

        changed = False

        top_group = create_plain_group()
        with top_group:
            top_left_col, top_middle_col, top_right_col = st.columns(3, gap="large", border=False)

            with top_left_col:
                problem_type_label = st.selectbox(
                    "Problem type",
                    options=list(PROBLEM_TYPE_OPTIONS.keys()),
                    index=list(PROBLEM_TYPE_OPTIONS.keys()).index(current_problem_label),
                    key="ml_workbench_problem_type_select",
                    help="Choose whether you are trying to predict a category or a numeric value.",
                )

            target_options = ["Not set"] + column_names
            current_target = state.get("target_column") or "Not set"
            if current_target not in target_options:
                current_target = "Not set"

            with top_middle_col:
                target_selection = st.selectbox(
                    "Target column",
                    options=target_options,
                    index=target_options.index(current_target),
                    key="ml_workbench_target_column_select",
                    help="This is the column the models will try to predict.",
                )

            selected_problem_type = PROBLEM_TYPE_OPTIONS[problem_type_label]
            selected_target = None if target_selection == "Not set" else target_selection
            positive_class_options = _positive_class_options(
                dataset_name=active_dataset_name,
                target_column=selected_target,
            )
            current_positive_class = state.get("positive_class_label")
            valid_positive_class = (
                current_positive_class if current_positive_class in positive_class_options else None
            )

            with top_right_col:
                if selected_problem_type == PROBLEM_TYPE_CLASSIFICATION and positive_class_options:
                    selected_positive_class = st.selectbox(
                        "Positive class label",
                        options=positive_class_options,
                        index=positive_class_options.index(valid_positive_class)
                        if valid_positive_class in positive_class_options
                        else 0,
                        key="ml_workbench_positive_class_label_select",
                        help="This label will be treated as the positive class for binary classification metrics and thresholding.",
                    )
                else:
                    selected_positive_class = None
                    st.selectbox(
                        "Positive class label",
                        options=["Not available"],
                        index=0,
                        disabled=True,
                        key="ml_workbench_positive_class_label_select_disabled",
                        help="Available only for binary classification targets.",
                    )

        bottom_group = create_plain_group()
        with bottom_group:
            bottom_left_col, bottom_middle_col, _ = st.columns(3, gap="large", border=False)

            current_id_columns = list(state.get("id_columns", []))
            valid_id_columns = [
                column for column in current_id_columns if column in column_names
            ]
            with bottom_left_col:
                selected_id_columns = st.multiselect(
                    "Identifier columns (optional)",
                    options=column_names,
                    default=valid_id_columns,
                    key="ml_workbench_id_columns_select",
                    help="Mark columns that identify records and are usually not useful as predictive features.",
                )

            current_ignored_columns = list(state.get("ignored_columns", []))
            valid_ignored_columns = [
                column for column in current_ignored_columns if column in column_names
            ]
            with bottom_middle_col:
                selected_ignored_columns = st.multiselect(
                    "Ignore columns (optional)",
                    options=column_names,
                    default=valid_ignored_columns,
                    key="ml_workbench_ignored_columns_select",
                    help="These columns will be excluded when building the model-input dataset.",
                )

        if state.get("problem_type") != selected_problem_type:
            set_state_value("problem_type", selected_problem_type)
            changed = True

        if state.get("target_column") != selected_target:
            set_state_value("target_column", selected_target)
            changed = True

        if state.get("positive_class_label") != selected_positive_class:
            set_state_value("positive_class_label", selected_positive_class)
            changed = True

        if state.get("id_columns") != selected_id_columns:
            set_state_value("id_columns", selected_id_columns)
            changed = True

        if state.get("ignored_columns") != selected_ignored_columns:
            set_state_value("ignored_columns", selected_ignored_columns)
            changed = True

        if changed:
            st.rerun()


def render_dataset_preview_panel() -> None:
    """Render a preview of the currently active dataset artifact."""
    if not has_loaded_dataset():
        render_status_message(EMPTY_STATE_NO_DATASET, variant="info")
        return

    dataset_names = get_available_dataset_names()
    if not dataset_names:
        render_status_message("No dataset artifacts are currently available for preview.", variant="info")
        return

    active_dataset_name = get_active_dataset_name(default_to_working=True)
    if active_dataset_name not in dataset_names:
        active_dataset_name = dataset_names[0]

    preview_panel = create_surface_panel(title="Data Preview")
    with preview_panel:
        selected_dataset = st.selectbox(
            "Dataset to preview",
            options=dataset_names,
            index=dataset_names.index(active_dataset_name),
            key="ml_workbench_preview_dataset_select",
            format_func=_display_dataset_name,
        )
        set_active_dataset_name(selected_dataset)

        summary = dataset_summary(selected_dataset)
        reconcile_column_state(list(summary.get("column_names", [])))

        preview_limit = st.slider(
            "Preview rows",
            min_value=5,
            max_value=200,
            value=int(
                get_app_state().get("ui", {}).get(
                    "preview_row_limit", DEFAULT_PREVIEW_ROW_LIMIT
                )
            ),
            step=5,
            key="ml_workbench_preview_limit_slider",
        )
        update_ui_state(preview_row_limit=preview_limit)

        try:
            preview_df = get_dataset_preview(selected_dataset, row_limit=preview_limit)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
        except DatasetLoadError as exc:
            render_status_message(str(exc), variant="error")
            return

    dtype_panel = create_surface_panel(title="Columns and Types")
    with dtype_panel:
        dtype_summary = summary.get("dtype_summary", {})
        if dtype_summary:
            dtype_df = pd.DataFrame(
                {
                    "column": list(dtype_summary.keys()),
                    "dtype": list(dtype_summary.values()),
                }
            )
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)
        else:
            render_status_message("No column-type summary is available for this dataset.", variant="info")

    missing_panel = create_surface_panel(title="Missing Values")
    with missing_panel:
        missing_summary = summary.get("missing_summary", {})
        if missing_summary:
            missing_df = pd.DataFrame(
                {
                    "column": list(missing_summary.keys()),
                    "missing_count": list(missing_summary.values()),
                }
            )
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            render_status_message("No missing values were detected in this dataset.", variant="success")
