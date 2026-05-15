"""Features tab UI for the ML Workbench app.

This first pass establishes the page structure and shared visual language for
engineered features. It intentionally focuses on summary, current-feature
visibility, and a clear builder scaffold before full feature creation and
execution are wired in.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.workspace_apps.ml_workbench.services.feature_service import (
    apply_feature_spec_to_dataframe,
    build_feature_collection_plan,
    create_and_store_feature_spec,
    get_feature_specs,
    preview_feature_spec,
    remove_stored_feature_specs,
)
from app.workspace_apps.ml_workbench.services.dataset_service import dataset_summary, get_dataset_copy
from app.workspace_apps.ml_workbench.services.preprocessing_service import (
    PreprocessingServiceError,
    rebuild_working_data_from_shared_rules,
)
from app.workspace_apps.ml_workbench.state import get_app_state
from app.workspace_apps.ml_workbench.ui.layout import (
    create_surface_panel,
    render_badge_row,
    render_info_card,
    render_status_message,
)

DATASET_DISPLAY_NAMES = {
    "raw_dataset": "Raw Data",
    "working_dataset": "Working Data",
}

FEATURES_SOURCE_DATASET_NAME = "raw_dataset"
FEATURES_TARGET_DATASET_NAME = "working_dataset"

FEATURE_BUILDER_MODES = ["Guided Builder", "Advanced Expression"]
GUIDED_FEATURE_FAMILIES = [
    "Arithmetic",
    "Transformation",
    "Flag",
]

GUIDED_FAMILY_OPERATIONS = {
    "Arithmetic": ["add", "subtract", "multiply", "divide", "ratio"],
    "Transformation": ["log", "log1p", "square", "cube"],
    "Flag": ["flag_gt", "flag_gte", "flag_lt", "flag_lte", "flag_eq", "flag_is_missing"],
}


GUIDED_OPERATION_LABELS = {
    "add": "Add",
    "subtract": "Subtract",
    "multiply": "Multiply",
    "divide": "Divide",
    "ratio": "Ratio",
    "log": "Log transform",
    "log1p": "Log(1 + x)",
    "square": "Square",
    "cube": "Cube",
    "flag_gt": "Greater than",
    "flag_gte": "Greater than or equal to",
    "flag_lt": "Less than",
    "flag_lte": "Less than or equal to",
    "flag_eq": "Equals",
    "flag_is_missing": "Is missing",
}


GUIDED_OPERATION_FEATURE_TYPES = {
    "add": "Arithmetic",
    "subtract": "Arithmetic",
    "multiply": "Arithmetic",
    "divide": "Arithmetic",
    "ratio": "Arithmetic",
    "log": "Transformation",
    "log1p": "Transformation",
    "square": "Transformation",
    "cube": "Transformation",
    "interaction": "Arithmetic",
    "flag_gt": "Flag",
    "flag_gte": "Flag",
    "flag_lt": "Flag",
    "flag_lte": "Flag",
    "flag_eq": "Flag",
    "flag_is_missing": "Flag",
}

BINARY_OPERATIONS = {"add", "subtract", "multiply", "divide", "ratio", "interaction"}
UNARY_OPERATIONS = {"log", "log1p", "square", "cube", "flag_is_missing"}
THRESHOLD_FLAG_OPERATIONS = {"flag_gt", "flag_gte", "flag_lt", "flag_lte"}
COMPARE_VALUE_FLAG_OPERATIONS = {"flag_eq"}



def _set_feature_preview_message(message: str, *, is_error: bool = False) -> None:
    """Persist the latest feature-builder status message for the preview panel."""
    st.session_state["ml_workbench_feature_preview_message"] = message
    st.session_state["ml_workbench_feature_preview_is_error"] = is_error



def _set_feature_preview_warnings(warnings: list[str]) -> None:
    """Persist the latest feature-builder warnings for the preview panel."""
    st.session_state["ml_workbench_feature_preview_warnings"] = list(warnings)



def _set_feature_preview_table(preview_df: pd.DataFrame | None) -> None:
    """Persist the latest feature preview dataframe for display."""
    st.session_state["ml_workbench_feature_preview_table"] = preview_df



def _clear_feature_preview_state() -> None:
    """Clear the latest feature preview state."""
    st.session_state["ml_workbench_feature_preview_message"] = ""
    st.session_state["ml_workbench_feature_preview_is_error"] = False
    st.session_state["ml_workbench_feature_preview_warnings"] = []
    st.session_state["ml_workbench_feature_preview_table"] = None
    _clear_pending_feature_spec()


def _set_pending_feature_spec(feature_spec: dict | None) -> None:
    """Persist the current unsaved preview feature spec."""
    st.session_state["ml_workbench_pending_feature_spec"] = feature_spec


def _get_pending_feature_spec() -> dict | None:
    """Return the current unsaved preview feature spec, if present."""
    pending_feature_spec = st.session_state.get("ml_workbench_pending_feature_spec")
    return pending_feature_spec if isinstance(pending_feature_spec, dict) else None


def _clear_pending_feature_spec() -> None:
    """Clear the current unsaved preview feature spec."""
    st.session_state["ml_workbench_pending_feature_spec"] = None



def _rebuild_working_data_after_feature_change() -> None:
    """Rebuild Working Data after engineered features change."""
    try:
        rebuild_working_data_from_shared_rules()
    except PreprocessingServiceError as exc:
        raise RuntimeError(str(exc)) from exc



def _normalize_operation_family(feature_family: str) -> str:
    """Convert the guided UI family label into the shared contract value."""
    return feature_family.strip().lower()


def _build_guided_feature_spec_from_form(
    *,
    feature_name: str,
    feature_family: str,
    operation: str,
    input_columns: list[str],
    threshold_value: str = "",
    compare_value: str = "",
) -> tuple[dict | None, str | None]:
    """Build an unsaved guided feature spec from the current form state."""
    parameters: dict[str, object] = {}
    if operation in THRESHOLD_FLAG_OPERATIONS:
        if threshold_value.strip() == "":
            return None, "Threshold value is required for this flag operation."
        parameters["threshold"] = threshold_value.strip()
    elif operation in COMPARE_VALUE_FLAG_OPERATIONS:
        if compare_value.strip() == "":
            return None, "Compare value is required for this flag operation."
        parameters["compare_value"] = compare_value.strip()

    if operation in BINARY_OPERATIONS and len(input_columns) != 2:
        return None, "This operation requires exactly two input columns."
    if operation in UNARY_OPERATIONS.union(THRESHOLD_FLAG_OPERATIONS).union(COMPARE_VALUE_FLAG_OPERATIONS) and len(input_columns) != 1:
        return None, "This operation requires exactly one input column."

    feature_spec = {
        "feature_id": "preview_feature",
        "feature_name": feature_name.strip(),
        "feature_type": GUIDED_OPERATION_FEATURE_TYPES.get(operation, feature_family),
        "builder_mode": "guided",
        "operation_family": _normalize_operation_family(feature_family),
        "operation": operation,
        "expression": "",
        "expression_language": None,
        "parameters": parameters,
        "source_columns": list(input_columns),
        "dependencies": [
            {
                "dependency_kind": "column",
                "dependency_name": column_name,
            }
            for column_name in input_columns
        ],
        "enabled": True,
        "execution_scope": "shared",
        "apply_order": 0,
        "created_by": "user",
        "status": "draft",
    }
    return feature_spec, None


def _build_expression_feature_spec_from_form(
    *,
    feature_name: str,
    expression: str,
) -> tuple[dict | None, str | None]:
    """Build an unsaved expression-based feature spec from the current form state."""
    if not feature_name.strip():
        return None, "Feature name is required."
    if not expression.strip():
        return None, "Expression is required."

    feature_spec = {
        "feature_id": "preview_feature",
        "feature_name": feature_name.strip(),
        "feature_type": "Expression",
        "builder_mode": "expression",
        "operation_family": "expression",
        "operation": "expression",
        "expression": expression.strip(),
        "expression_language": "mlw_expr_v1",
        "parameters": {},
        "source_columns": [],
        "dependencies": [],
        "enabled": True,
        "execution_scope": "shared",
        "apply_order": 0,
        "created_by": "user",
        "status": "draft",
    }
    return feature_spec, None


def _display_dataset_name(dataset_name: str) -> str:
    """Return a friendly dataset display label."""
    return DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name.replace("_", " ").title())



def _problem_display(problem_type: object) -> str:
    """Return a friendly problem-type label."""
    text = str(problem_type or "").strip()
    return text.title() if text else "Not set"



def _build_selectable_features_table_df(feature_specs: list[dict]) -> pd.DataFrame:
    """Build a selectable current-features table."""
    rows: list[dict[str, object]] = []
    for feature_spec in sorted(feature_specs, key=lambda feature: int(feature.get("apply_order", 0))):
        rows.append(
            {
                "selected": False,
                "feature_id": str(feature_spec.get("feature_id", "")),
                "feature_name": str(feature_spec.get("feature_name", "")),
                "type": str(feature_spec.get("feature_type", "")),
                "enabled": bool(feature_spec.get("enabled", True)),
                "order": int(feature_spec.get("apply_order", 0)),
            }
        )
    return pd.DataFrame(rows)


def _features_editor_state_key() -> str:
    """Return the session-state key used for the features table dataframe."""
    return "ml_workbench_features_overview_df"


def _features_editor_signature(feature_specs: list[dict]) -> tuple[tuple[str, str, bool, int], ...]:
    """Return a compact signature for the current features-table contents."""
    return tuple(
        (
            str(feature_spec.get("feature_id", "")),
            str(feature_spec.get("feature_name", "")),
            bool(feature_spec.get("enabled", True)),
            int(feature_spec.get("apply_order", 0)),
        )
        for feature_spec in sorted(feature_specs, key=lambda feature: int(feature.get("apply_order", 0)))
    )


def _features_editor_signature_key() -> str:
    """Return the session-state key used for the features table signature."""
    return "ml_workbench_features_overview_signature"


def _features_editor_version_key() -> str:
    """Return the session-state key used to version/reset the features data editor."""
    return "ml_workbench_features_overview_version"


def _features_editor_widget_key() -> str:
    """Return the widget key for the features data editor."""
    version = int(st.session_state.get(_features_editor_version_key(), 0))
    return f"ml_workbench_features_overview_editor_v{version}"


def _get_or_initialize_features_editor_df(feature_specs: list[dict]) -> pd.DataFrame:
    """Return a stable dataframe for the features data editor."""
    editor_key = _features_editor_state_key()
    signature_key = _features_editor_signature_key()
    current_signature = _features_editor_signature(feature_specs)
    stored_df = st.session_state.get(editor_key)
    stored_signature = st.session_state.get(signature_key)

    if not isinstance(stored_df, pd.DataFrame) or stored_signature != current_signature:
        initialized_df = _build_selectable_features_table_df(feature_specs).copy()
        st.session_state[editor_key] = initialized_df
        st.session_state[signature_key] = current_signature
        st.session_state[_features_editor_version_key()] = 0
        return initialized_df

    return stored_df.copy()


def _set_features_editor_df(editor_df: pd.DataFrame) -> None:
    """Persist the latest features data-editor dataframe into session state."""
    st.session_state[_features_editor_state_key()] = editor_df.copy()


def _bump_features_editor_version() -> None:
    """Force the features data editor to rebuild from the cached dataframe."""
    current_version = int(st.session_state.get(_features_editor_version_key(), 0))
    st.session_state[_features_editor_version_key()] = current_version + 1


def _build_features_table_df(feature_specs: list[dict]) -> pd.DataFrame:
    """Build a lightweight current-features table for the first-pass UI."""
    rows: list[dict[str, object]] = []
    for feature_spec in sorted(feature_specs, key=lambda feature: int(feature.get("apply_order", 0))):
        rows.append(
            {
                "feature_name": str(feature_spec.get("feature_name", "")),
                "type": str(feature_spec.get("feature_type", "")),
                "enabled": bool(feature_spec.get("enabled", True)),
                "order": int(feature_spec.get("apply_order", 0)),
            }
        )
    return pd.DataFrame(rows)



def _render_feature_summary_panel(*, state: dict, feature_specs: list[dict]) -> None:
    """Render the top summary pills for the Features tab."""
    collection_plan = build_feature_collection_plan(
        feature_specs,
        available_columns=list(state.get("working_dataset_columns", [])),
    )
    badges = [
        f"Source · {_display_dataset_name(FEATURES_SOURCE_DATASET_NAME)}",
        f"Target · {_display_dataset_name(FEATURES_TARGET_DATASET_NAME)}",
        f"Active Features · {collection_plan.feature_count}",
        f"Enabled · {collection_plan.enabled_feature_count}",
        f"Problem · {_problem_display(state.get('problem_type'))}",
    ]
    render_badge_row(badges, variant="info")



def _render_current_engineered_features_panel(feature_specs: list[dict], *, state: dict) -> None:
    """Render the current engineered-features panel."""
    panel = create_surface_panel(
        title="Current Engineered Features",
        subtitle="Review the shared engineered features that will be applied during Working Data rebuilds.",
    )
    with panel:
        if not feature_specs:
            render_info_card(
                title="No engineered features yet",
                message="Create your first shared engineered feature below. It will be saved once and shared across candidate models.",
                border=False,
            )
            return

        features_df = _get_or_initialize_features_editor_df(feature_specs)
        edited_features_df = st.data_editor(
            features_df,
            use_container_width=True,
            hide_index=True,
            disabled=["feature_id", "feature_name", "type", "enabled", "order"],
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select engineered features here for the next action.",
                    default=False,
                    width="small",
                ),
                "feature_id": None,
                "feature_name": st.column_config.TextColumn("Feature Name"),
                "type": st.column_config.TextColumn("Type"),
                "enabled": st.column_config.CheckboxColumn("Enabled", width="small"),
                "order": st.column_config.NumberColumn("Order", width="small"),
            },
            key=_features_editor_widget_key(),
        )
        selected_feature_ids = edited_features_df.loc[
            edited_features_df["selected"], "feature_id"
        ].tolist()

        if st.button(
            "Delete selected feature(s)",
            key="ml_workbench_delete_selected_features",
            disabled=not selected_feature_ids,
        ):
            removed_feature_specs = remove_stored_feature_specs(selected_feature_ids)
            if not removed_feature_specs:
                render_status_message("No engineered features were deleted.", variant="warning")
                return
            try:
                _rebuild_working_data_after_feature_change()
            except RuntimeError as exc:
                render_status_message(str(exc), variant="error")
                st.rerun()
            updated_editor_df = edited_features_df.loc[~edited_features_df["selected"]].copy()
            if "selected" in updated_editor_df.columns:
                updated_editor_df["selected"] = False
            _set_features_editor_df(updated_editor_df)
            _bump_features_editor_version()
            removed_feature_names = [
                str(feature_spec.get("feature_name", "")).strip()
                for feature_spec in removed_feature_specs
                if str(feature_spec.get("feature_name", "")).strip()
            ]
            feature_label = ", ".join(removed_feature_names) if removed_feature_names else "selected features"
            render_status_message(
                f"Deleted {len(removed_feature_specs)} engineered feature(s): {feature_label}.",
                variant="success",
            )
            st.rerun()

        collection_plan = build_feature_collection_plan(
            feature_specs,
            available_columns=list(state.get("working_dataset_columns", [])),
        )
        if collection_plan.warnings:
            st.caption("Validation notes")
            for warning in collection_plan.warnings[:5]:
                st.write(f"- {warning}")



def _render_feature_builder_panel(*, state: dict) -> None:
    """Render the first-pass feature builder scaffold."""
    panel = create_surface_panel(
        title="Feature Builder",
        subtitle="Create shared engineered features. Guided Builder is for common patterns, while Advanced Expression will support a lightweight formula language later.",
    )
    with panel:
        builder_mode = st.radio(
            "Builder mode",
            options=FEATURE_BUILDER_MODES,
            horizontal=True,
            key="ml_workbench_feature_builder_mode",
        )

        if builder_mode == "Guided Builder":
            top_left_col, top_right_col = st.columns(2)
            with top_left_col:
                feature_name = st.text_input(
                    "Feature name",
                    key="ml_workbench_feature_name",
                    placeholder="Example: debt_to_income",
                )
            with top_right_col:
                feature_family = st.selectbox(
                    "Feature family",
                    options=GUIDED_FEATURE_FAMILIES,
                    key="ml_workbench_feature_family",
                    help="Choose the broad category of feature logic you want to build.",
                )

            bottom_left_col, bottom_right_col = st.columns(2)
            with bottom_left_col:
                operation = st.selectbox(
                    "Operation",
                    options=GUIDED_FAMILY_OPERATIONS.get(feature_family, []),
                    key="ml_workbench_feature_operation",
                    help="Choose the specific operation within the selected feature family.",
                    format_func=lambda op: GUIDED_OPERATION_LABELS.get(op, op.replace("_", " ").title()),
                )
            with bottom_right_col:
                input_columns = st.multiselect(
                    "Input columns",
                    options=list(state.get("working_dataset_columns", [])),
                    key="ml_workbench_feature_input_columns",
                    help="Pick the columns used to define this engineered feature.",
                )

            threshold_value: str = ""
            compare_value: str = ""
            if operation in THRESHOLD_FLAG_OPERATIONS:
                threshold_value = st.text_input(
                    "Threshold value",
                    key="ml_workbench_feature_threshold_value",
                    placeholder="Example: 0.5",
                    help="Used by comparison flag features such as greater than or less than.",
                )
            elif operation in COMPARE_VALUE_FLAG_OPERATIONS:
                compare_value = st.text_input(
                    "Compare value",
                    key="ml_workbench_feature_compare_value",
                    placeholder="Example: Approved",
                    help="Used by equality flag features.",
                )

            if operation in BINARY_OPERATIONS:
                st.caption("This operation expects two input columns.")
            elif operation in UNARY_OPERATIONS or operation in THRESHOLD_FLAG_OPERATIONS or operation in COMPARE_VALUE_FLAG_OPERATIONS:
                st.caption("This operation expects one input column.")

            st.caption(
                "Guided Builder supports arithmetic features, transformations, and flags using the shared engineered-feature contract."
            )

            preview_feature = st.button(
                "Preview feature",
                key="ml_workbench_preview_engineered_feature",
                disabled=not feature_name.strip() or not input_columns,
                use_container_width=False,
            )
            if preview_feature:
                candidate_feature_spec, validation_error = _build_guided_feature_spec_from_form(
                    feature_name=feature_name,
                    feature_family=feature_family,
                    operation=operation,
                    input_columns=input_columns,
                    threshold_value=threshold_value,
                    compare_value=compare_value,
                )
                if candidate_feature_spec is None:
                    _set_pending_feature_spec(None)
                    _set_feature_preview_message(validation_error or "Unable to preview feature.", is_error=True)
                    _set_feature_preview_warnings([])
                    _set_feature_preview_table(None)
                    st.rerun()

                source_df = get_dataset_copy(FEATURES_TARGET_DATASET_NAME)
                preview_result = preview_feature_spec(candidate_feature_spec, source_df=source_df)
                preview_df = preview_result.preview_dataframe
                executed_preview_df, execution_step = apply_feature_spec_to_dataframe(candidate_feature_spec, source_df.head(10).copy())
                preview_columns = [
                    column
                    for column in candidate_feature_spec.get("source_columns", [])
                    if column in executed_preview_df.columns
                ]
                output_column = str(candidate_feature_spec.get("feature_name", "")).strip()
                if output_column and output_column in executed_preview_df.columns:
                    preview_columns.append(output_column)
                preview_table = executed_preview_df[preview_columns].copy() if preview_columns else preview_df

                combined_warnings = list(preview_result.warnings) + list(execution_step.warnings)
                if execution_step.error_message:
                    combined_warnings.append(execution_step.error_message)

                _set_pending_feature_spec(candidate_feature_spec)
                _set_feature_preview_message(
                    f"Feature '{feature_name.strip()}' is ready to save. Review the preview below.",
                    is_error=not bool(preview_result.validation.get("is_valid")) or execution_step.status == "error",
                )
                _set_feature_preview_warnings(combined_warnings)
                _set_feature_preview_table(preview_table)
                st.rerun()

        else:
            feature_name = st.text_input(
                "Feature name",
                key="ml_workbench_expression_feature_name",
                placeholder="Example: debt_to_income",
            )
            expression = st.text_area(
                "Expression",
                key="ml_workbench_expression_text",
                placeholder="Examples:\nincome / debt\n[Loan Amount] / [Annual Income]\n([field1] - [field2]) * field3\nlog(balance)\nscore^2",
                height=160,
            )
            st.caption(
                "Use simple field names directly, or wrap multi-word columns in brackets like [Loan Amount]. Supported operators include +, -, *, /, parentheses, log(...), log1p(...), square(...), cube(...), and ^2/^3."
            )
            preview_expression = st.button(
                "Preview expression",
                key="ml_workbench_preview_expression_feature",
                disabled=not feature_name.strip() or not expression.strip(),
                use_container_width=False,
            )
            if preview_expression:
                candidate_feature_spec, validation_error = _build_expression_feature_spec_from_form(
                    feature_name=feature_name,
                    expression=expression,
                )
                if candidate_feature_spec is None:
                    _set_pending_feature_spec(None)
                    _set_feature_preview_message(validation_error or "Unable to preview expression.", is_error=True)
                    _set_feature_preview_warnings([])
                    _set_feature_preview_table(None)
                    st.rerun()

                source_df = get_dataset_copy(FEATURES_TARGET_DATASET_NAME)
                preview_result = preview_feature_spec(candidate_feature_spec, source_df=source_df)
                executed_preview_df, execution_step = apply_feature_spec_to_dataframe(candidate_feature_spec, source_df.head(10).copy())
                preview_columns = [
                    column
                    for column in candidate_feature_spec.get("source_columns", [])
                    if column in executed_preview_df.columns
                ]
                output_column = str(candidate_feature_spec.get("feature_name", "")).strip()
                if output_column and output_column in executed_preview_df.columns:
                    preview_columns.append(output_column)
                preview_table = executed_preview_df[preview_columns].copy() if preview_columns else None

                combined_warnings = list(preview_result.warnings) + list(execution_step.warnings)
                if execution_step.error_message:
                    combined_warnings.append(execution_step.error_message)

                _set_pending_feature_spec(candidate_feature_spec)
                _set_feature_preview_message(
                    f"Expression feature '{feature_name.strip()}' is ready to save. Review the preview below.",
                    is_error=not bool(preview_result.validation.get("is_valid")) or execution_step.status == "error",
                )
                _set_feature_preview_warnings(combined_warnings)
                _set_feature_preview_table(preview_table)
                st.rerun()


def _render_feature_preview_panel() -> None:
    """Render the first-pass preview and validation panel."""
    panel = create_surface_panel(
        title="Validation & Preview",
        subtitle="Preview dependencies, validation status, and sample output before saving a feature.",
    )
    with panel:
        preview_message = str(st.session_state.get("ml_workbench_feature_preview_message", "")).strip()
        preview_is_error = bool(st.session_state.get("ml_workbench_feature_preview_is_error", False))
        preview_warnings = list(st.session_state.get("ml_workbench_feature_preview_warnings", []))
        preview_table = st.session_state.get("ml_workbench_feature_preview_table")

        if not preview_message and not preview_warnings and preview_table is None:
            render_info_card(
                title="Preview not connected yet",
                message="Create a guided engineered feature to see validation notes and a lightweight source-column preview here.",
                border=False,
            )
            return

        if preview_message:
            if preview_is_error:
                render_status_message(preview_message, variant="error")
            else:
                render_status_message(preview_message, variant="success")

        if preview_warnings:
            st.caption("Validation notes")
            for warning in preview_warnings:
                st.write(f"- {warning}")

        if isinstance(preview_table, pd.DataFrame) and not preview_table.empty:
            st.caption("Feature preview")
            st.dataframe(preview_table, use_container_width=True, hide_index=True)

        create_feature = st.button(
            "Create engineered feature",
            key="ml_workbench_create_engineered_feature",
            disabled=_get_pending_feature_spec() is None,
            use_container_width=False,
        )
        if create_feature:
            pending_feature_spec = _get_pending_feature_spec()
            if pending_feature_spec is None:
                _set_feature_preview_message("Preview the feature before saving it.", is_error=True)
                st.rerun()

            feature_spec = create_and_store_feature_spec(
                feature_name=str(pending_feature_spec.get("feature_name", "")),
                feature_type=str(pending_feature_spec.get("feature_type", "")),
                operation_family=str(pending_feature_spec.get("operation_family", "")),
                operation=str(pending_feature_spec.get("operation", "")),
                source_columns=list(pending_feature_spec.get("source_columns", [])),
                expression=str(pending_feature_spec.get("expression", "")),
                parameters=dict(pending_feature_spec.get("parameters", {})),
                builder_mode=str(pending_feature_spec.get("builder_mode", "guided")),
                expression_language=pending_feature_spec.get("expression_language"),
                created_by="user",
            )

            try:
                _rebuild_working_data_after_feature_change()
            except RuntimeError as exc:
                _set_feature_preview_message(str(exc), is_error=True)
                st.rerun()

            _clear_pending_feature_spec()
            _set_feature_preview_message(
                f"Engineered feature '{feature_spec.get('feature_name', '')}' was created and Working Data was rebuilt.",
                is_error=False,
            )
            st.rerun()


def render_features_panel() -> None:
    """Render the Features tab for the ML Workbench app."""
    state = get_app_state()
    feature_specs = get_feature_specs()

    if "ml_workbench_feature_preview_message" not in st.session_state:
        _clear_feature_preview_state()
    if "ml_workbench_pending_feature_spec" not in st.session_state:
        _clear_pending_feature_spec()

    working_summary = dataset_summary(FEATURES_TARGET_DATASET_NAME)
    working_dataset_columns = list(working_summary.get("column_names", []))

    state_for_page = {
        **state,
        "working_dataset_columns": working_dataset_columns,
    }

    summary_panel = create_surface_panel(
        title="Features",
        subtitle="Build shared engineered features that are applied during Working Data rebuilds from Raw Data and shared across candidate models.",
    )
    with summary_panel:
        _render_feature_summary_panel(state=state_for_page, feature_specs=feature_specs)

    _render_current_engineered_features_panel(feature_specs, state=state_for_page)
    _render_feature_builder_panel(state=state_for_page)
    _render_feature_preview_panel()
