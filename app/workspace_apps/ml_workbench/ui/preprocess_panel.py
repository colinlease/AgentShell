"""Preprocessing panel helpers for the ML Workbench UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.workspace_apps.ml_workbench.constants import (
    EMPTY_STATE_NO_DATASET,
    PROBLEM_TYPE_CLASSIFICATION,
)
from app.workspace_apps.ml_workbench.services.dataset_service import (
    dataset_summary,
    get_active_dataset_name,
    has_loaded_dataset,
)
from app.workspace_apps.ml_workbench.services.preprocessing_service import (
    PreprocessingServiceError,
    rebuild_working_data_from_shared_rules,
)
from app.workspace_apps.ml_workbench.state import get_app_state, get_preprocessing_config
from app.workspace_apps.ml_workbench.ui.layout import (
    create_surface_panel,
    render_badge_row,
    render_info_card,
    render_status_message,
)


DATASET_DISPLAY_NAMES = {
    "raw_dataset": "Raw Data",
    "working_dataset": "Working Data",
    "model_input_dataset": "Model Input Data",
}

PREPARE_SOURCE_DATASET_NAME = "raw_dataset"
PREPARE_TARGET_DATASET_NAME = "working_dataset"

ACTION_FAMILY_OPTIONS = [
    "Column removal",
    "Imputation",
    "Datetime handling",
]

IMPUTATION_KIND_OPTIONS = ["Numeric", "Category/Text"]
NUMERIC_IMPUTATION_METHOD_OPTIONS = ["mean", "median", "constant"]
CATEGORICAL_IMPUTATION_METHOD_OPTIONS = ["mode", "constant"]
RULE_APPLY_MODE_OPTIONS = ["Add to existing rule", "Replace existing rule"]



def _display_dataset_name(dataset_name: str) -> str:
    """Convert an internal dataset artifact name into a user-friendly label."""
    return DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name.replace("_", " ").title())



def _problem_display(problem_type: object) -> str:
    """Return a user-friendly problem-type label."""
    if problem_type == PROBLEM_TYPE_CLASSIFICATION:
        return "Classification"
    if problem_type is None:
        return "Not set"
    return str(problem_type).title()




def _build_column_overview_df(summary: dict, state: dict) -> pd.DataFrame:
    """Build a compact column overview table for the prepare tab."""
    column_names = list(summary.get("column_names", []))
    dtype_summary = summary.get("dtype_summary", {})
    missing_summary = summary.get("missing_summary", {})
    target_column = state.get("target_column")
    id_columns = set(state.get("id_columns", []))
    ignored_columns = set(state.get("ignored_columns", []))

    rows: list[dict[str, object]] = []
    for column_name in column_names:
        role_parts: list[str] = []
        if column_name == target_column:
            role_parts.append("Target")
        if column_name in id_columns:
            role_parts.append("Identifier")
        if column_name in ignored_columns:
            role_parts.append("Ignored")

        rows.append(
            {
                "column": column_name,
                "dtype": dtype_summary.get(column_name, "—"),
                "missing_count": int(missing_summary.get(column_name, 0)),
                "role": ", ".join(role_parts) if role_parts else "—",
            }
        )

    return pd.DataFrame(rows)



def _build_selectable_column_overview_df(summary: dict, state: dict) -> pd.DataFrame:
    """Build a selectable column overview table for the prepare tab.

    The selection state is still owned separately in session state so this table
    renderer can be replaced later without changing downstream rule logic.
    """
    overview_df = _build_column_overview_df(summary, state).copy()
    overview_df.insert(
        0,
        "selected",
        [False for _ in overview_df["column"].tolist()],
    )
    return overview_df


# ---- Prepare Table State Helpers ----

def _prepare_editor_state_key() -> str:
    """Return the session-state key used for the prepare table dataframe."""
    return "ml_workbench_prepare_column_overview_df"


def _column_signature(summary: dict, state: dict) -> tuple[tuple[str, ...], str, tuple[str, ...], tuple[str, ...]]:
    """Return a compact signature for the current prepare-table schema context."""
    return (
        tuple(summary.get("column_names", [])),
        str(state.get("target_column") or ""),
        tuple(state.get("id_columns", [])),
        tuple(state.get("ignored_columns", [])),
    )


def _prepare_editor_signature_key() -> str:
    """Return the session-state key used for the prepare table signature."""
    return "ml_workbench_prepare_column_overview_signature"


def _prepare_editor_version_key() -> str:
    """Return the session-state key used to version/reset the prepare data editor."""
    return "ml_workbench_prepare_column_overview_version"


def _prepare_editor_widget_key() -> str:
    """Return the widget key for the prepare data editor.

    The key is versioned so explicit actions like Select all / Clear selection can
    intentionally reset the editor, while ordinary checkbox interaction can stay
    fully owned by the widget without a second source of truth fighting it.
    """
    version = int(st.session_state.get(_prepare_editor_version_key(), 0))
    return f"ml_workbench_prepare_column_overview_editor_v{version}"


def _get_or_initialize_prepare_editor_df(summary: dict, state: dict) -> pd.DataFrame:
    """Return a stable dataframe for the prepare data editor.

    The editor dataframe is cached in session state so checkbox interaction does
    not require rebuilding the table from scratch on each rerun.
    """
    editor_key = _prepare_editor_state_key()
    signature_key = _prepare_editor_signature_key()
    current_signature = _column_signature(summary, state)
    stored_df = st.session_state.get(editor_key)
    stored_signature = st.session_state.get(signature_key)

    if not isinstance(stored_df, pd.DataFrame) or stored_signature != current_signature:
        initialized_df = _build_selectable_column_overview_df(summary, state).copy()
        st.session_state[editor_key] = initialized_df
        st.session_state[signature_key] = current_signature
        st.session_state[_prepare_editor_version_key()] = 0
        return initialized_df

    return stored_df.copy()


def _set_prepare_editor_df(editor_df: pd.DataFrame) -> None:
    """Persist the latest prepare data-editor dataframe into session state."""
    st.session_state[_prepare_editor_state_key()] = editor_df.copy()


def _bump_prepare_editor_version() -> None:
    """Force the prepare data editor to rebuild from the cached dataframe."""
    current_version = int(st.session_state.get(_prepare_editor_version_key(), 0))
    st.session_state[_prepare_editor_version_key()] = current_version + 1




def _union_preserve_order(existing: list[str], selected: list[str]) -> list[str]:
    """Append new selected columns while preserving prior order."""
    combined = list(existing)
    for column in selected:
        if column not in combined:
            combined.append(column)
    return combined


def _replace_preserve_order(selected: list[str]) -> list[str]:
    """Return a clean ordered copy of selected columns."""
    cleaned: list[str] = []
    for column in selected:
        if column not in cleaned:
            cleaned.append(column)
    return cleaned


def _resolve_rule_columns(
    *,
    existing: list[str],
    selected: list[str],
    apply_mode: str,
) -> list[str]:
    """Resolve the next stored column list for a preparation rule."""
    if apply_mode == "Replace existing rule":
        return _replace_preserve_order(selected)
    return _union_preserve_order(existing, selected)


def _remove_selected_columns_from_rule(existing: list[str], selected: list[str]) -> list[str]:
    """Remove the selected columns from an existing stored rule."""
    selected_set = set(selected)
    return [column for column in existing if column not in selected_set]


def _get_prepare_selected_columns() -> list[str]:
    """Return the current prepare-tab selected columns from session state."""
    return list(st.session_state.get("ml_workbench_prepare_selected_columns", []))


def _set_prepare_selected_columns(selected_columns: list[str]) -> None:
    """Persist the current prepare-tab selected columns into session state."""
    st.session_state["ml_workbench_prepare_selected_columns"] = list(selected_columns)



def _clear_prepare_selected_columns() -> None:
    """Clear the current prepare-tab selected columns from session state."""
    st.session_state["ml_workbench_prepare_selected_columns"] = []


def _rebuild_working_data_after_rule_change() -> None:
    """Rebuild Working Data after a shared preprocessing rule changes."""
    try:
        rebuild_working_data_from_shared_rules()
    except PreprocessingServiceError as exc:
        render_status_message(str(exc), variant="error")
        return




def _render_current_rules_panel(preprocessing_config: dict, selected_columns: list[str]) -> None:
    """Render a compact view of the currently configured preparation rules."""
    current_rules_panel = create_surface_panel(
        title="Current Preparation Rules",
        subtitle="Review, trim, or clear the shared cleanup rules that rebuild Working Data from Raw Data.",
    )
    with current_rules_panel:
        badges: list[str] = []

        drop_columns = list(preprocessing_config.get("drop_columns", []))
        if drop_columns:
            badges.append(f"Drop · {len(drop_columns)} column(s)")

        numeric_imputation = preprocessing_config.get("numeric_imputation", {})
        if numeric_imputation.get("strategy") and numeric_imputation.get("columns"):
            badges.append(
                f"Numeric imputation · {numeric_imputation.get('strategy')} · {len(numeric_imputation.get('columns', []))} column(s)"
            )

        categorical_imputation = preprocessing_config.get("categorical_imputation", {})
        if categorical_imputation.get("strategy") and categorical_imputation.get("columns"):
            badges.append(
                f"Category/text imputation · {categorical_imputation.get('strategy')} · {len(categorical_imputation.get('columns', []))} column(s)"
            )

        # Remove encoding, scaling, and rebalancing badges
        # encoding_config = preprocessing_config.get("encoding", {})
        # if encoding_config.get("strategy") and encoding_config.get("columns"):
        #     badges.append(
        #         f"Encoding · {encoding_config.get('strategy')} · {len(encoding_config.get('columns', []))} column(s)"
        #     )
        # scaling_config = preprocessing_config.get("scaling", {})
        # if scaling_config.get("strategy") not in (None, "none") and scaling_config.get("columns"):
        #     badges.append(
        #         f"Scaling · {scaling_config.get('strategy')} · {len(scaling_config.get('columns', []))} column(s)"
        #     )

        datetime_config = preprocessing_config.get("datetime_handling", {})
        if datetime_config.get("expanded_columns"):
            badges.append(
                f"Datetime expansion · {len(datetime_config.get('expanded_columns', []))} column(s)"
            )

        # rebalancing_config = preprocessing_config.get("class_rebalancing", {})
        # if bool(rebalancing_config.get("enabled", False)):
        #     badges.append(
        #         f"Class rebalancing · {rebalancing_config.get('strategy', 'none')}"
        #     )

        if badges:
            render_badge_row(badges, variant="info")
        else:
            render_info_card(
                title="No rules yet",
                message="No preparation rules have been applied yet. Select columns above, choose an action family, and apply your first rule.",
                border=False,
            )

        if drop_columns:
            st.markdown("**Dropped from prepared data**")
            st.markdown(" ".join(f"`{column}`" for column in drop_columns))
            drop_remove_col, drop_clear_col = st.columns([1, 1])
            with drop_remove_col:
                if st.button(
                    "Remove selected from drop rule",
                    key="ml_workbench_remove_selected_from_drop_rule",
                    disabled=not selected_columns,
                ):
                    preprocessing_config["drop_columns"] = _remove_selected_columns_from_rule(
                        drop_columns,
                        selected_columns,
                    )
                    _rebuild_working_data_after_rule_change()
                    st.rerun()
            with drop_clear_col:
                if st.button(
                    "Clear drop rule",
                    key="ml_workbench_clear_drop_rule",
                ):
                    preprocessing_config["drop_columns"] = []
                    _rebuild_working_data_after_rule_change()
                    st.rerun()

        if numeric_imputation.get("strategy") and numeric_imputation.get("columns"):
            st.markdown("**Numeric imputation**")
            st.markdown(
                f"Strategy: `{numeric_imputation.get('strategy')}` | Columns: "
                + " ".join(f"`{column}`" for column in numeric_imputation.get("columns", []))
            )
            num_remove_col, num_clear_col = st.columns([1, 1])
            with num_remove_col:
                if st.button(
                    "Remove selected from numeric rule",
                    key="ml_workbench_remove_selected_from_numeric_rule",
                    disabled=not selected_columns,
                ):
                    numeric_imputation["columns"] = _remove_selected_columns_from_rule(
                        list(numeric_imputation.get("columns", [])),
                        selected_columns,
                    )
                    if not numeric_imputation["columns"]:
                        numeric_imputation["strategy"] = None
                        numeric_imputation["fill_value"] = None
                    _rebuild_working_data_after_rule_change()
                    st.rerun()
            with num_clear_col:
                if st.button(
                    "Clear numeric rule",
                    key="ml_workbench_clear_numeric_rule",
                ):
                    numeric_imputation["strategy"] = None
                    numeric_imputation["columns"] = []
                    numeric_imputation["fill_value"] = None
                    _rebuild_working_data_after_rule_change()
                    st.rerun()

        if categorical_imputation.get("strategy") and categorical_imputation.get("columns"):
            st.markdown("**Category/text imputation**")
            st.markdown(
                f"Strategy: `{categorical_imputation.get('strategy')}` | Columns: "
                + " ".join(f"`{column}`" for column in categorical_imputation.get("columns", []))
            )
            cat_remove_col, cat_clear_col = st.columns([1, 1])
            with cat_remove_col:
                if st.button(
                    "Remove selected from category/text rule",
                    key="ml_workbench_remove_selected_from_categorical_rule",
                    disabled=not selected_columns,
                ):
                    categorical_imputation["columns"] = _remove_selected_columns_from_rule(
                        list(categorical_imputation.get("columns", [])),
                        selected_columns,
                    )
                    if not categorical_imputation["columns"]:
                        categorical_imputation["strategy"] = None
                        categorical_imputation["fill_value"] = None
                    _rebuild_working_data_after_rule_change()
                    st.rerun()
            with cat_clear_col:
                if st.button(
                    "Clear category/text rule",
                    key="ml_workbench_clear_categorical_rule",
                ):
                    categorical_imputation["strategy"] = None
                    categorical_imputation["columns"] = []
                    categorical_imputation["fill_value"] = None
                    _rebuild_working_data_after_rule_change()
                    st.rerun()

        # Remove encoding and scaling UI sections

        if datetime_config.get("expanded_columns"):
            st.markdown("**Datetime handling**")
            st.markdown(
                "Expanded columns: "
                + " ".join(f"`{column}`" for column in datetime_config.get("expanded_columns", []))
            )
            dt_remove_col, dt_clear_col = st.columns([1, 1])
            with dt_remove_col:
                if st.button(
                    "Remove selected from datetime rule",
                    key="ml_workbench_remove_selected_from_datetime_rule",
                    disabled=not selected_columns,
                ):
                    datetime_config["expanded_columns"] = _remove_selected_columns_from_rule(
                        list(datetime_config.get("expanded_columns", [])),
                        selected_columns,
                    )
                    _rebuild_working_data_after_rule_change()
                    st.rerun()
            with dt_clear_col:
                if st.button(
                    "Clear datetime rule",
                    key="ml_workbench_clear_datetime_rule",
                ):
                    datetime_config["expanded_columns"] = []
                    _rebuild_working_data_after_rule_change()
                    st.rerun()

        # Remove class rebalancing UI section



def render_preprocess_panel() -> None:
    """Render the preprocessing configuration panel.

    Phase 1 shifts the prepare tab from a long operation-first form into a
    column-first workflow:
    1. Review columns and select the ones you want to work on.
    2. Choose an action family and method.
    3. Apply that rule and review the currently stored preparation rules.
    """
    if not has_loaded_dataset():
        render_status_message(EMPTY_STATE_NO_DATASET, variant="info")
        return

    active_dataset_name = get_active_dataset_name(default_to_working=True)
    if active_dataset_name is None:
        render_status_message("No dataset is currently available for preprocessing.", variant="info")
        return

    summary = dataset_summary(PREPARE_TARGET_DATASET_NAME)
    column_names = list(summary.get("column_names", []))
    state = get_app_state()
    preprocessing_config = get_preprocessing_config()
    selected_columns: list[str] = []

    overview_panel = create_surface_panel(
        title="Prepare",
        subtitle="Review the current Working Data view, select columns, and build reusable preparation rules that rebuild Working Data from Raw Data.",
    )
    with overview_panel:
        summary_badges = [
            f"Source · {_display_dataset_name(PREPARE_SOURCE_DATASET_NAME)}",
            f"Target · {_display_dataset_name(PREPARE_TARGET_DATASET_NAME)}",
            f"Rows · {int(summary.get('rows', 0)):,}",
            f"Columns · {int(summary.get('columns', 0)):,}",
            f"Problem · {_problem_display(state.get('problem_type'))}",
        ]
        render_badge_row(summary_badges, variant="info")

        column_overview_df = _get_or_initialize_prepare_editor_df(summary, state)
        edited_column_overview_df = st.data_editor(
            column_overview_df,
            use_container_width=True,
            hide_index=True,
            disabled=["column", "dtype", "missing_count", "role"],
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select columns here for the next preparation action.",
                    default=False,
                    width="small",
                ),
                "column": st.column_config.TextColumn("Column"),
                "dtype": st.column_config.TextColumn("Type"),
                "missing_count": st.column_config.NumberColumn("Missing"),
                "role": st.column_config.TextColumn("Role"),
            },
            key=_prepare_editor_widget_key(),
        )
        selected_columns = edited_column_overview_df.loc[
            edited_column_overview_df["selected"], "column"
        ].tolist()



        selected_columns_col, selected_actions_col = st.columns([1, 2])
        with selected_columns_col:
            selected_count = len(selected_columns)
            selected_label = (
                f"{selected_count} column selected"
                if selected_count == 1
                else f"{selected_count} columns selected"
            )
            st.markdown(f"**Current selection:** {selected_label}")
            if selected_columns:
                st.markdown(" ".join(f"`{column}`" for column in selected_columns))
            else:
                st.caption("Use the table checkboxes above to select columns for the next action.")

        with selected_actions_col:
            st.markdown("**Selection actions**")
            selection_button_col, clear_selection_col = st.columns([1, 1])
            with selection_button_col:
                if st.button(
                    "Select all columns",
                    key="ml_workbench_prepare_select_all_columns",
                ):
                    updated_editor_df = column_overview_df.copy()
                    updated_editor_df["selected"] = updated_editor_df["column"].isin(column_names)
                    _set_prepare_editor_df(updated_editor_df)
                    _bump_prepare_editor_version()
                    st.rerun()
            with clear_selection_col:
                if st.button(
                    "Clear selection",
                    key="ml_workbench_prepare_clear_selected_columns",
                ):
                    updated_editor_df = column_overview_df.copy()
                    updated_editor_df["selected"] = False
                    _set_prepare_editor_df(updated_editor_df)
                    _bump_prepare_editor_version()
                    st.rerun()

    action_builder_panel = create_surface_panel(
        title="Action Builder",
        subtitle="Choose an action family, then a method and settings that apply to the currently selected columns.",
    )
    with action_builder_panel:
        action_family_col, _ = st.columns([1, 2])
        with action_family_col:
            action_family = st.selectbox(
                "Action family",
                options=ACTION_FAMILY_OPTIONS,
                key="ml_workbench_prepare_action_family",
            )
        apply_mode_col, _ = st.columns([1, 2])
        with apply_mode_col:
            apply_mode = st.selectbox(
                "Rule update mode",
                options=RULE_APPLY_MODE_OPTIONS,
                key="ml_workbench_prepare_rule_apply_mode",
                help="Add selected columns to an existing rule or replace that rule's current column list entirely.",
            )
        st.caption("This action updates shared preprocessing and rebuilds Working Data from Raw Data.")

        if action_family == "Column removal":
            st.caption(
                "This adds the selected columns to the preparation rule that removes them from prepared data. Raw Data stays unchanged."
            )
            apply_drop = st.button(
                "Apply column removal",
                key="ml_workbench_apply_drop_columns",
                use_container_width=False,
                disabled=not selected_columns,
            )
            if apply_drop:
                preprocessing_config["drop_columns"] = _resolve_rule_columns(
                    existing=list(preprocessing_config.get("drop_columns", [])),
                    selected=selected_columns,
                    apply_mode=apply_mode,
                )
                _rebuild_working_data_after_rule_change()
                render_status_message("Column-removal rule updated.", variant="success")
                st.rerun()

        elif action_family == "Imputation":
            imputation_kind_col, _ = st.columns([1, 2])
            with imputation_kind_col:
                imputation_kind = st.selectbox(
                    "Imputation type",
                    options=IMPUTATION_KIND_OPTIONS,
                    key="ml_workbench_prepare_imputation_kind",
                )

            if imputation_kind == "Numeric":
                numeric_method_col, numeric_fill_col = st.columns(2)
                with numeric_method_col:
                    imputation_method = st.selectbox(
                        "Imputation method",
                        options=NUMERIC_IMPUTATION_METHOD_OPTIONS,
                        key="ml_workbench_prepare_numeric_imputation_method",
                    )
                with numeric_fill_col:
                    fill_value = st.text_input(
                        "Constant fill value",
                        key="ml_workbench_prepare_numeric_fill_value",
                        help="Only used when the method is constant.",
                        disabled=imputation_method != "constant",
                    )
                apply_numeric = st.button(
                    "Apply numeric imputation",
                    key="ml_workbench_apply_numeric_imputation",
                    use_container_width=False,
                    disabled=not selected_columns,
                )
                if apply_numeric:
                    numeric_imputation = preprocessing_config.get("numeric_imputation", {})
                    numeric_imputation["strategy"] = imputation_method
                    numeric_imputation["columns"] = _resolve_rule_columns(
                        existing=list(numeric_imputation.get("columns", [])),
                        selected=selected_columns,
                        apply_mode=apply_mode,
                    )
                    numeric_imputation["fill_value"] = (
                        None if imputation_method != "constant" or fill_value.strip() == "" else fill_value.strip()
                    )
                    _rebuild_working_data_after_rule_change()
                    render_status_message("Numeric imputation rule updated.", variant="success")
                    st.rerun()
            else:
                categorical_method_col, categorical_fill_col = st.columns(2)
                with categorical_method_col:
                    imputation_method = st.selectbox(
                        "Imputation method",
                        options=CATEGORICAL_IMPUTATION_METHOD_OPTIONS,
                        key="ml_workbench_prepare_categorical_imputation_method",
                    )
                with categorical_fill_col:
                    fill_value = st.text_input(
                        "Constant fill value",
                        key="ml_workbench_prepare_categorical_fill_value",
                        help="Only used when the method is constant.",
                        disabled=imputation_method != "constant",
                    )
                apply_categorical = st.button(
                    "Apply category/text imputation",
                    key="ml_workbench_apply_categorical_imputation",
                    use_container_width=False,
                    disabled=not selected_columns,
                )
                if apply_categorical:
                    categorical_imputation = preprocessing_config.get("categorical_imputation", {})
                    categorical_imputation["strategy"] = imputation_method
                    categorical_imputation["columns"] = _resolve_rule_columns(
                        existing=list(categorical_imputation.get("columns", [])),
                        selected=selected_columns,
                        apply_mode=apply_mode,
                    )
                    categorical_imputation["fill_value"] = (
                        None if imputation_method != "constant" or fill_value.strip() == "" else fill_value.strip()
                    )
                    _rebuild_working_data_after_rule_change()
                    render_status_message("Category/text imputation rule updated.", variant="success")
                    st.rerun()

        # Remove Encoding and Scaling action families

        elif action_family == "Datetime handling":
            auto_detect = st.checkbox(
                "Automatically detect date/time columns",
                value=bool(preprocessing_config.get("datetime_handling", {}).get("auto_detect", True)),
                key="ml_workbench_prepare_datetime_auto_detect",
            )
            apply_datetime = st.button(
                "Apply datetime handling rule",
                key="ml_workbench_apply_datetime_handling",
                use_container_width=False,
                disabled=not selected_columns,
            )
            if apply_datetime:
                datetime_config = preprocessing_config.get("datetime_handling", {})
                datetime_config["auto_detect"] = auto_detect
                datetime_config["expanded_columns"] = _resolve_rule_columns(
                    existing=list(datetime_config.get("expanded_columns", [])),
                    selected=selected_columns,
                    apply_mode=apply_mode,
                )
                _rebuild_working_data_after_rule_change()
                render_status_message("Datetime handling rule updated.", variant="success")
                st.rerun()

        # Remove class rebalancing action family

    _render_current_rules_panel(preprocessing_config, selected_columns)
