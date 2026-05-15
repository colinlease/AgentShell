

"""Modeling tab UI for the ML Workbench app.

This first pass focuses on a clean, extensible training-configuration workflow:
- summary pills at the top
- lightweight candidate model selection
- shared training settings
- one shared model-specific settings area with model-switch buttons

The goal is to keep the page visually consistent with the other tabs while
avoiding a giant stacked form.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.workspace_apps.ml_workbench.constants import STAGE_RESULTS

from app.workspace_apps.ml_workbench.models.registry import (
    get_model_specs_for_problem_type,
)
from app.workspace_apps.ml_workbench.services.dataset_service import dataset_summary, get_dataset_copy
from app.workspace_apps.ml_workbench.services.feature_service import get_feature_specs
from app.workspace_apps.ml_workbench.services.modeling_service import (
    ModelingServiceError,
    build_candidate_run_plan,
    create_candidate_model,
    get_candidate_model,
    remove_candidate_model,
    select_best_candidate_from_latest_results,
    set_best_candidate_id,
    train_candidate_model,
    update_candidate_model_config,
    update_model_comparison_settings,
)
from app.workspace_apps.ml_workbench.state import get_app_state, set_state_value
from app.workspace_apps.ml_workbench.ui.layout import (
    create_surface_panel,
    render_badge_row,
    render_info_card,
    render_status_message,
)


MODEL_INPUT_DATASET_NAME = "working_dataset"
MAX_ONE_HOT_CARDINALITY = 100


def _training_notice_key() -> str:
    """Return the session-state key for the latest post-training notice."""
    return "ml_workbench_post_training_notice"


def _set_training_notice(
    message: str,
    *,
    variant: str,
    show_results_button: bool = False,
) -> None:
    """Persist the latest post-training notice for subsequent reruns."""
    st.session_state[_training_notice_key()] = {
        "message": str(message),
        "variant": str(variant),
        "show_results_button": bool(show_results_button),
    }


def _clear_training_notice() -> None:
    """Clear any persisted post-training notice."""
    st.session_state.pop(_training_notice_key(), None)


def _get_training_notice() -> dict[str, Any] | None:
    """Return the latest persisted post-training notice, if any."""
    notice = st.session_state.get(_training_notice_key())
    return notice if isinstance(notice, dict) else None



def _render_training_status_with_results_action(
    message: str,
    *,
    variant: str,
    show_results_button: bool = False,
) -> None:
    """Render the post-training message with an optional shortcut to Results."""
    message_col, action_col, _ = st.columns([3.2, 1.35, 7.45], gap="small")
    with message_col:
        render_status_message(message, variant=variant)
    with action_col:
        if show_results_button:
            if st.button(
                "View Results",
                key="ml_workbench_view_results_after_training",
                use_container_width=False,
            ):
                set_state_value("app_stage", STAGE_RESULTS)
                st.rerun()


def _render_training_progress(progress_placeholder: Any, value: int) -> None:
    """Render one stable horizontal training-progress bar."""
    bounded_value = max(0, min(int(value), 100))
    progress_placeholder.markdown(
        f"""
        <div class="mlw-training-progress" aria-label="Training progress">
          <div class="mlw-training-progress__track">
            <div class="mlw-training-progress__fill" style="width: {bounded_value}%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _problem_display(problem_type: object) -> str:
    """Return a friendly problem-type label."""
    text = str(problem_type or "").strip()
    return text.title() if text else "Not set"



def _get_working_dataset_columns() -> list[str]:
    """Return the current Working Data columns."""
    summary = dataset_summary(MODEL_INPUT_DATASET_NAME)
    return list(summary.get("column_names", []))



def _get_engineered_feature_count() -> int:
    """Return the number of currently saved engineered features."""
    return len(get_feature_specs())



def _get_eligible_model_input_columns(state: dict[str, Any]) -> list[str]:
    """Return Working Data columns that are eligible as predictors."""
    working_columns = _get_working_dataset_columns()
    target_column = str(state.get("target_column") or "").strip()
    ignored_columns = set(state.get("ignored_columns", []))
    identifier_columns = set(state.get("id_columns", []))
    return [
        column
        for column in working_columns
        if column != target_column and column not in ignored_columns and column not in identifier_columns
    ]


def _get_low_cardinality_encoding_columns(eligible_columns: list[str]) -> list[str]:
    """Return eligible predictor columns with one-hot-safe cardinality."""
    source_df = get_dataset_copy(MODEL_INPUT_DATASET_NAME)
    encoding_candidates: list[str] = []
    for column_name in eligible_columns:
        if column_name not in source_df.columns:
            continue
        unique_count = int(source_df[column_name].dropna().nunique())
        if unique_count <= MAX_ONE_HOT_CARDINALITY:
            encoding_candidates.append(column_name)
    return encoding_candidates


def _get_high_cardinality_encoding_columns(eligible_columns: list[str]) -> list[str]:
    """Return eligible predictor columns that exceed the one-hot cardinality limit."""
    source_df = get_dataset_copy(MODEL_INPUT_DATASET_NAME)
    blocked_columns: list[str] = []
    for column_name in eligible_columns:
        if column_name not in source_df.columns:
            continue
        unique_count = int(source_df[column_name].dropna().nunique())
        if unique_count > MAX_ONE_HOT_CARDINALITY:
            blocked_columns.append(column_name)
    return blocked_columns



def _ensure_modeling_session_defaults(problem_type: str, available_model_ids: list[str]) -> None:
    """Initialize modeling-panel session state used by the first-pass UI."""
    if "ml_workbench_model_selected_ids" not in st.session_state:
        st.session_state["ml_workbench_model_selected_ids"] = []

    selected_ids = [
        model_id
        for model_id in st.session_state.get("ml_workbench_model_selected_ids", [])
        if model_id in available_model_ids
    ]
    st.session_state["ml_workbench_model_selected_ids"] = selected_ids

    if "ml_workbench_active_model_settings_id" not in st.session_state:
        st.session_state["ml_workbench_active_model_settings_id"] = selected_ids[0] if selected_ids else None

    active_model_id = st.session_state.get("ml_workbench_active_model_settings_id")
    if active_model_id not in selected_ids:
        st.session_state["ml_workbench_active_model_settings_id"] = selected_ids[0] if selected_ids else None

    if "ml_workbench_model_training_settings" not in st.session_state:
        st.session_state["ml_workbench_model_training_settings"] = {
            "evaluation_mode": "Cross Validation",
            "cv_folds": 5,
            "test_size": 0.2,
            "random_seed": 42,
            "classification_threshold_policy": "Use model default",
            "classification_threshold_manual_value": 0.5,
            "classification_threshold_objective": "F1",
        }

    if "ml_workbench_model_candidate_drafts" not in st.session_state:
        st.session_state["ml_workbench_model_candidate_drafts"] = {}




def _toggle_model_selection(model_id: str) -> None:
    """Add or remove a model from the current candidate selection."""
    selected_ids = list(st.session_state.get("ml_workbench_model_selected_ids", []))
    if model_id in selected_ids:
        selected_ids = [candidate_id for candidate_id in selected_ids if candidate_id != model_id]
    else:
        selected_ids.append(model_id)
    st.session_state["ml_workbench_model_selected_ids"] = selected_ids

    active_model_id = st.session_state.get("ml_workbench_active_model_settings_id")
    if active_model_id not in selected_ids:
        st.session_state["ml_workbench_active_model_settings_id"] = selected_ids[0] if selected_ids else None


def _chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    """Split a list into evenly sized chunks."""
    if chunk_size <= 0:
        return [items]
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _candidate_id_map_key() -> str:
    """Return the session-state key for model-id to candidate-id mapping."""
    return "ml_workbench_model_candidate_id_map"


def _get_candidate_id_map() -> dict[str, str]:
    """Return the current model-id to candidate-id mapping."""
    stored_value = st.session_state.get(_candidate_id_map_key(), {})
    if not isinstance(stored_value, dict):
        stored_value = {}
        st.session_state[_candidate_id_map_key()] = stored_value
    return stored_value



def _set_active_model_settings(model_id: str) -> None:
    """Set the currently visible model-specific settings section."""
    st.session_state["ml_workbench_active_model_settings_id"] = model_id



def _get_candidate_draft(model_spec: Any, eligible_columns: list[str], problem_type: str) -> dict[str, Any]:
    """Return a mutable first-pass draft config for one candidate model."""
    drafts = st.session_state["ml_workbench_model_candidate_drafts"]
    model_id = str(model_spec.get("model_id"))
    if model_id not in drafts:
        drafts[model_id] = {
            "feature_subset_mode": "Use all eligible predictors",
            "included_columns": [],
            "excluded_columns": [],
            "encoding_strategy": "None",
            "encoding_columns": [],
            "scaling_strategy": "None",
            "class_rebalancing_strategy": "None",
            "hyperparameters": {
                param_name: param_config.get("default")
                for param_name, param_config in dict(model_spec.get("param_schema", {})).items()
            },
        }
    return drafts[model_id]


def _ensure_model_widget_defaults(model_id: str, draft: dict[str, Any]) -> None:
    """Initialize model-specific widget state once so controls do not require double clicks."""
    widget_defaults = {
        f"ml_workbench_subset_mode_{model_id}": draft.get("feature_subset_mode", "Use all eligible predictors"),
        f"ml_workbench_excluded_columns_{model_id}": list(draft.get("excluded_columns", [])),
        f"ml_workbench_included_columns_{model_id}": list(draft.get("included_columns", [])),
        f"ml_workbench_encoding_strategy_{model_id}": draft.get("encoding_strategy", "None"),
        f"ml_workbench_encoding_columns_{model_id}": list(draft.get("encoding_columns", [])),
        f"ml_workbench_scaling_strategy_{model_id}": draft.get("scaling_strategy", "None"),
        f"ml_workbench_rebalancing_strategy_{model_id}": draft.get("class_rebalancing_strategy", "None"),
    }
    for widget_key, widget_value in widget_defaults.items():
        if widget_key not in st.session_state:
            st.session_state[widget_key] = widget_value



def _normalize_encoding_strategy_for_backend(value: object) -> str:
    """Normalize encoding strategy labels for the modeling service contract."""
    text = str(value or "none").strip().lower().replace("-", "_")
    if text in {"onehot", "one_hot"}:
        return "one_hot"
    if text in {"", "none"}:
        return "none"
    return text



def _normalize_scaling_strategy_for_backend(value: object) -> str:
    """Normalize scaling strategy labels for the modeling service contract."""
    text = str(value or "none").strip().lower().replace("-", "_")
    if text in {"minmax", "min_max", "min-max"}:
        return "minmax"
    if text in {"", "none"}:
        return "none"
    return text



def _normalize_rebalancing_strategy_for_backend(value: object) -> str:
    """Normalize class-rebalancing labels for the modeling service contract."""
    text = str(value or "none").strip().lower().replace("-", "_")
    if text in {"", "none"}:
        return "none"
    return text



def _comparison_config_from_training_settings(training_settings: dict[str, Any]) -> dict[str, Any]:
    """Translate panel training settings into the modeling-service config shape."""
    evaluation_mode = str(training_settings.get("evaluation_mode", "Cross Validation"))
    if evaluation_mode == "Train / Test Split":
        split_strategy = "train_test_split"
    else:
        split_strategy = "cross_validation"

    return {
        "split_strategy": split_strategy,
        "cv_folds": int(training_settings.get("cv_folds", 5)),
        "test_size": float(training_settings.get("test_size", 0.2)),
        "random_seed": int(training_settings.get("random_seed", 42)),
        "classification_threshold_policy": str(
            training_settings.get("classification_threshold_policy", "Use model default")
        ),
        "classification_threshold_manual_value": float(
            training_settings.get("classification_threshold_manual_value", 0.5)
        ),
        "classification_threshold_objective": str(
            training_settings.get("classification_threshold_objective", "F1")
        ),
    }



def _candidate_config_from_draft(model_id: str, model_spec: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    """Translate one model-tab draft into the modeling-service candidate config shape."""
    hyperparameters = {
        param_name: draft.get("hyperparameters", {}).get(param_name, param_config.get("default"))
        for param_name, param_config in dict(model_spec.get("param_schema", {})).items()
    }
    feature_subset_mode = str(draft.get("feature_subset_mode", "Use all eligible predictors"))
    included_columns = list(draft.get("included_columns", []))
    excluded_columns = list(draft.get("excluded_columns", []))

    if feature_subset_mode == "Include only specific predictors":
        excluded_columns = []
    elif feature_subset_mode == "Exclude specific predictors":
        included_columns = []
    else:
        included_columns = []
        excluded_columns = []

    return {
        "candidate_label": str(model_spec.get("label", model_id)),
        "enabled": True,
        "model_id": model_id,
        "preprocessing": {
            "use_shared_preprocessing": True,
            "feature_subset_mode": feature_subset_mode,
            "included_columns": included_columns,
            "excluded_columns": excluded_columns,
            "encoding_strategy": _normalize_encoding_strategy_for_backend(draft.get("encoding_strategy", "None")),
            "encoding_columns": list(draft.get("encoding_columns", [])),
            "scaling_strategy": _normalize_scaling_strategy_for_backend(draft.get("scaling_strategy", "None")),
            "class_rebalancing_strategy": _normalize_rebalancing_strategy_for_backend(
                draft.get("class_rebalancing_strategy", "None")
            ),
        },
        "hyperparameters": hyperparameters,
    }



def _sync_ui_drafts_to_modeling_service(model_specs_by_id: dict[str, Any]) -> list[str]:
    """Persist the currently selected model drafts into the modeling-service state.

    Returns the candidate ids that are enabled for the current run.
    """
    selected_model_ids = list(st.session_state.get("ml_workbench_model_selected_ids", []))
    training_settings = dict(st.session_state.get("ml_workbench_model_training_settings", {}))
    drafts = st.session_state.get("ml_workbench_model_candidate_drafts", {})
    candidate_id_map = _get_candidate_id_map()

    update_model_comparison_settings(**_comparison_config_from_training_settings(training_settings))

    enabled_candidate_ids: list[str] = []
    kept_model_ids: set[str] = set()

    for model_id in selected_model_ids:
        model_spec = model_specs_by_id.get(model_id)
        if not model_spec:
            continue
        draft = drafts.get(model_id, {})
        candidate_payload = _candidate_config_from_draft(model_id, model_spec, draft)

        candidate_id = str(candidate_id_map.get(model_id, "")).strip()
        existing_candidate = get_candidate_model(candidate_id) if candidate_id else None
        if existing_candidate is None:
            created_candidate = create_candidate_model(model_id=model_id)
            candidate_id = str(created_candidate.get("candidate_id"))
            candidate_id_map[model_id] = candidate_id

        update_candidate_model_config(candidate_id, **candidate_payload)
        enabled_candidate_ids.append(candidate_id)
        kept_model_ids.add(model_id)

    for model_id, candidate_id in list(candidate_id_map.items()):
        if model_id in kept_model_ids:
            continue
        existing_candidate = get_candidate_model(candidate_id)
        if existing_candidate is not None:
            remove_candidate_model(candidate_id)
        candidate_id_map.pop(model_id, None)

    st.session_state[_candidate_id_map_key()] = candidate_id_map
    return enabled_candidate_ids






def _render_candidate_selection_panel(problem_type: str, model_specs: list[Any]) -> None:
    """Render candidate-model selection using simple toggle buttons."""
    if not problem_type:
        render_info_card(
            title="Problem type required",
            message="Set the problem type in the Data tab before configuring candidate models.",
            border=False,
        )
        return

    if not model_specs:
        render_info_card(
            title="No models available",
            message="No model specs are currently registered for this problem type.",
            border=False,
        )
        return

    selected_ids = list(st.session_state.get("ml_workbench_model_selected_ids", []))
    model_chunks = _chunk_list([str(spec.get("model_id")) for spec in model_specs], 4)
    model_specs_by_id = {str(spec.get("model_id")): spec for spec in model_specs}

    for model_chunk in model_chunks:
        selector_columns = st.columns([1] * len(model_chunk) + [6], gap="small")
        for col_index, model_id in enumerate(model_chunk):
            model_spec = model_specs_by_id.get(model_id, {})
            model_label = str(model_spec.get("label", model_id))
            is_selected = model_id in selected_ids
            button_label = f"✓ {model_label}" if is_selected else model_label
            with selector_columns[col_index]:
                if st.button(
                    button_label,
                    key=f"ml_workbench_select_model_{model_id}",
                    use_container_width=False,
                    type="primary" if is_selected else "secondary",
                ):
                    _toggle_model_selection(model_id)
                    st.rerun()

    selected_count = len(st.session_state.get("ml_workbench_model_selected_ids", []))
    st.caption(f"Selected candidate models: {selected_count}")


def _has_binary_classification_target() -> bool:
    """Return True when the current workspace target is binary classification."""
    state = get_app_state()
    problem_type = str(state.get("problem_type") or "").strip().lower()
    target_column = str(state.get("target_column") or "").strip()
    if problem_type != "classification" or not target_column:
        return False
    try:
        source_df = get_dataset_copy(MODEL_INPUT_DATASET_NAME)
    except Exception:
        return False
    if target_column not in source_df.columns:
        return False
    return len(source_df[target_column].dropna().unique().tolist()) == 2


def _render_classification_threshold_settings(training_settings: dict[str, Any]) -> None:
    """Render shared threshold controls for binary classification only."""
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(123, 129, 144, 0.28); margin: 0.85rem 0 0.95rem 0;'>", unsafe_allow_html=True)
    st.markdown("**Classification Decision Threshold**")

    threshold_policy_key = "ml_workbench_training_threshold_policy"
    manual_threshold_key = "ml_workbench_training_manual_threshold"
    threshold_objective_key = "ml_workbench_training_threshold_objective"

    if threshold_policy_key not in st.session_state:
        st.session_state[threshold_policy_key] = str(
            training_settings.get("classification_threshold_policy", "Use model default")
        )
    if manual_threshold_key not in st.session_state:
        st.session_state[manual_threshold_key] = float(
            training_settings.get("classification_threshold_manual_value", 0.5)
        )
    if threshold_objective_key not in st.session_state:
        st.session_state[threshold_objective_key] = str(
            training_settings.get("classification_threshold_objective", "F1")
        )

    threshold_policy = st.radio(
        "Threshold strategy",
        options=["Use model default", "Set manual threshold", "Optimize threshold"],
        horizontal=True,
        key=threshold_policy_key,
        help="Binary classification only. This threshold setting is shared across the selected candidate models for the current run.",
    )
    training_settings["classification_threshold_policy"] = threshold_policy

    threshold_columns = st.columns([1, 6], gap="small")
    if threshold_policy == "Set manual threshold":
        with threshold_columns[0]:
            training_settings["classification_threshold_manual_value"] = st.number_input(
                "Manual threshold",
                min_value=0.01,
                max_value=0.99,
                step=0.01,
                value=float(st.session_state[manual_threshold_key]),
                key=manual_threshold_key,
            )
    elif threshold_policy == "Optimize threshold":
        with threshold_columns[0]:
            training_settings["classification_threshold_objective"] = st.selectbox(
                "Optimization target",
                options=["F1", "Precision", "Recall"],
                key=threshold_objective_key,
            )


def _render_training_settings_panel(problem_type: str, model_specs_by_id: dict[str, Any]) -> None:
    """Render shared training/evaluation settings."""
    panel = create_surface_panel(
        title="Training Settings",
        subtitle="These settings apply to the current training run across the selected candidate models.",
    )
    with panel:
        training_settings = st.session_state["ml_workbench_model_training_settings"]
        evaluation_mode_key = "ml_workbench_training_evaluation_mode"
        if evaluation_mode_key not in st.session_state:
            st.session_state[evaluation_mode_key] = str(training_settings.get("evaluation_mode", "Cross Validation"))

        training_settings["evaluation_mode"] = st.radio(
            "Evaluation method",
            options=["Cross Validation", "Train / Test Split"],
            horizontal=True,
            key=evaluation_mode_key,
        )

        evaluation_mode = str(training_settings.get("evaluation_mode", "Cross Validation"))
        input_columns = st.columns([1, 1, 6], gap="small")
        if evaluation_mode == "Cross Validation":
            with input_columns[0]:
                training_settings["cv_folds"] = st.number_input(
                    "CV folds",
                    min_value=2,
                    max_value=10,
                    step=1,
                    value=int(training_settings.get("cv_folds", 5)),
                    key="ml_workbench_training_cv_folds",
                )
        else:
            with input_columns[0]:
                training_settings["test_size"] = st.number_input(
                    "Test size",
                    min_value=0.1,
                    max_value=0.4,
                    step=0.05,
                    value=float(training_settings.get("test_size", 0.2)),
                    key="ml_workbench_training_test_size",
                )
        with input_columns[1]:
            training_settings["random_seed"] = st.number_input(
                "Random seed",
                min_value=0,
                max_value=999999,
                step=1,
                value=int(training_settings.get("random_seed", 42)),
                key="ml_workbench_training_random_seed",
            )

        if problem_type == "classification" and _has_binary_classification_target():
            _render_classification_threshold_settings(training_settings)




def _render_model_specific_settings_panel(problem_type: str, eligible_columns: list[str], model_specs_by_id: dict[str, Any]) -> None:
    """Render one shared settings area that switches between selected models."""
    panel = create_surface_panel(
        title="Model-Specific Settings",
        subtitle="Use the model buttons below to move between candidate-specific preprocessing and hyperparameter settings.",
    )
    with panel:
        selected_ids = list(st.session_state.get("ml_workbench_model_selected_ids", []))
        if not selected_ids:
            render_info_card(
                title="No candidate models selected",
                message="Select at least one model above to configure candidate-specific settings.",
                border=False,
            )
            return

        tab_labels = [str(model_specs_by_id[model_id].get("label")) for model_id in selected_ids if model_id in model_specs_by_id]
        if not tab_labels:
            render_info_card(
                title="Model settings unavailable",
                message="Choose a candidate model above to configure its settings.",
                border=False,
            )
            return

        model_tabs = st.tabs(tab_labels)
        for tab_index, model_id in enumerate([model_id for model_id in selected_ids if model_id in model_specs_by_id]):
            active_model_spec = model_specs_by_id[model_id]
            draft = _get_candidate_draft(active_model_spec, eligible_columns, problem_type)
            _ensure_model_widget_defaults(model_id, draft)
            encoding_candidate_columns = _get_low_cardinality_encoding_columns(eligible_columns)
            blocked_encoding_columns = _get_high_cardinality_encoding_columns(eligible_columns)

            with model_tabs[tab_index]:
                st.markdown(f"**{str(active_model_spec.get('label'))}**")

                subset_left_col, subset_right_col = st.columns(2, border=False)
                with subset_left_col:
                    subset_options = [
                        "Use all eligible predictors",
                        "Exclude specific predictors",
                        "Include only specific predictors",
                    ]
                    subset_key = f"ml_workbench_subset_mode_{model_id}"
                    draft["feature_subset_mode"] = st.radio(
                        "Predictor subset",
                        options=subset_options,
                        key=subset_key,
                    )
                with subset_right_col:
                    excluded_key = f"ml_workbench_excluded_columns_{model_id}"
                    included_key = f"ml_workbench_included_columns_{model_id}"
                    if draft["feature_subset_mode"] == "Exclude specific predictors":
                        draft["excluded_columns"] = st.multiselect(
                            "Excluded predictors",
                            options=eligible_columns,
                            key=excluded_key,
                        )
                    elif draft["feature_subset_mode"] == "Include only specific predictors":
                        draft["included_columns"] = st.multiselect(
                            "Included predictors",
                            options=eligible_columns,
                            key=included_key,
                        )
                    else:
                        st.caption("All eligible Working Data predictors will be used for this candidate.")

                st.markdown("<hr style='border: none; border-top: 1px solid rgba(123, 129, 144, 0.28); margin: 0.85rem 0 0.95rem 0;'>", unsafe_allow_html=True)

                preprocess_columns = st.columns([1, 1, 1, 5], border=False, gap="small")
                with preprocess_columns[0]:
                    draft["encoding_strategy"] = st.radio(
                        "Encoding",
                        options=["None", "One-hot"],
                        horizontal=False,
                        key=f"ml_workbench_encoding_strategy_{model_id}",
                    )
                    if draft["encoding_strategy"] == "One-hot":
                        draft["encoding_columns"] = st.multiselect(
                            "Columns to one-hot encode",
                            options=encoding_candidate_columns,
                            key=f"ml_workbench_encoding_columns_{model_id}",
                            help=(
                                f"Only predictors with {MAX_ONE_HOT_CARDINALITY} or fewer distinct non-null values are shown here. "
                                "Columns above that limit are hidden to avoid creating an excessive number of one-hot features."
                            ) if blocked_encoding_columns else None,
                        )
                with preprocess_columns[1]:
                    if problem_type == "classification":
                        draft["class_rebalancing_strategy"] = st.radio(
                            "Class rebalancing",
                            options=["None", "Oversample", "Undersample"],
                            horizontal=True,
                            help=(
                                "Binary classification only. Non-numeric labels such as GOOD and BAD are supported. "
                                "Rebalancing applies to training data only."
                            ),
                            key=f"ml_workbench_rebalancing_strategy_{model_id}",
                        )
                    else:
                        st.caption("Class rebalancing is only shown for classification problems.")
                with preprocess_columns[2]:
                    draft["scaling_strategy"] = st.radio(
                        "Scaling",
                        options=["None", "Standard", "Min-Max"],
                        horizontal=False,
                        key=f"ml_workbench_scaling_strategy_{model_id}",
                    )

                st.markdown("<hr style='border: none; border-top: 1px solid rgba(123, 129, 144, 0.28); margin: 0.85rem 0 0.95rem 0;'>", unsafe_allow_html=True)
                st.markdown("**Hyperparameters**")
                hyperparameter_columns = st.columns([1, 1, 6], border=False, gap="small")
                param_items = list(dict(active_model_spec.get("param_schema", {})).items())
                for index, (param_name, param_config) in enumerate(param_items):
                    with hyperparameter_columns[index % len(hyperparameter_columns)]:
                        current_value = draft["hyperparameters"].get(param_name, param_config.get("default"))
                        param_type = param_config.get("type")
                        if param_type == "int":
                            draft["hyperparameters"][param_name] = st.number_input(
                                param_name.replace("_", " ").title(),
                                min_value=int(param_config.get("min", 0)),
                                max_value=int(param_config.get("max", 1000000)),
                                step=int(param_config.get("step", 1)),
                                value=int(current_value),
                                help=None if param_name in {"penalty", "solver"} else param_config.get("help_text"),
                                key=f"ml_workbench_param_{model_id}_{param_name}",
                            )
                        elif param_type == "float":
                            draft["hyperparameters"][param_name] = st.number_input(
                                param_name.replace("_", " ").title(),
                                min_value=float(param_config.get("min", 0.0)),
                                max_value=float(param_config.get("max", 1000000.0)),
                                step=float(param_config.get("step", 0.01)),
                                value=float(current_value),
                                help=None if param_name in {"penalty", "solver"} else param_config.get("help_text"),
                                key=f"ml_workbench_param_{model_id}_{param_name}",
                            )
                        elif param_type == "bool":
                            draft["hyperparameters"][param_name] = st.checkbox(
                                param_name.replace("_", " ").title(),
                                value=bool(current_value),
                                help=None if param_name in {"penalty", "solver"} else param_config.get("help_text"),
                                key=f"ml_workbench_param_{model_id}_{param_name}",
                            )
                        else:
                            options = list(param_config.get("options", []))
                            if not options:
                                draft["hyperparameters"][param_name] = st.text_input(
                                    param_name.replace("_", " ").title(),
                                    value="" if current_value is None else str(current_value),
                                    help=None if param_name in {"penalty", "solver"} else param_config.get("help_text"),
                                    key=f"ml_workbench_param_{model_id}_{param_name}",
                                )
                            else:
                                selected_option = current_value if current_value in options else options[0]
                                draft["hyperparameters"][param_name] = st.radio(
                                    param_name.replace("_", " ").title(),
                                    options=options,
                                    horizontal=False,
                                    index=options.index(selected_option),
                                    help=None if param_name in {"penalty", "solver"} else param_config.get("help_text"),
                                    key=f"ml_workbench_param_{model_id}_{param_name}",
                                )



def _render_train_panel(model_specs_by_id: dict[str, Any]) -> None:
    """Render the bottom training action section."""
    panel = create_surface_panel(
        title="Train & Evaluate",
        subtitle="Start a candidate training run using the selected models and settings above.",
    )
    with panel:
        action_col, _ = st.columns([1.35, 6.65], gap="small")
        with action_col:
            train_requested = st.button(
                "Train and Evaluate",
                key="ml_workbench_train_and_evaluate",
                use_container_width=False,
            )

        if train_requested:
            try:
                _clear_training_notice()
                enabled_candidate_ids = _sync_ui_drafts_to_modeling_service(model_specs_by_id)
                if not enabled_candidate_ids:
                    render_status_message("Select at least one candidate model before training.", variant="error")
                    return

                progress_text = st.empty()
                progress_bar_placeholder = st.empty()

                progress_text.caption("Preparing candidate training run...")
                _render_training_progress(progress_bar_placeholder, 5)

                execution_results = []
                total_candidates = len(enabled_candidate_ids)
                for index, candidate_id in enumerate(enabled_candidate_ids, start=1):
                    run_plan = build_candidate_run_plan(candidate_id)
                    progress_text.caption(
                        f"Training candidate {index} of {total_candidates}: {run_plan.candidate_label}"
                    )
                    progress_value = 15 + int(((index - 1) / max(total_candidates, 1)) * 70)
                    _render_training_progress(progress_bar_placeholder, progress_value)

                    try:
                        result = train_candidate_model(candidate_id)
                    except ModelingServiceError:
                        raise
                    except Exception as exc:
                        message = f"Training failed for {run_plan.candidate_label}: {exc}"
                        from app.workspace_apps.ml_workbench.services.modeling_service import _build_model_run_record, CandidateExecutionResult
                        failed_run_record = _build_model_run_record(
                            run_plan=run_plan,
                            metrics={},
                            status="failed",
                            message=message,
                        )
                        result = CandidateExecutionResult(
                            candidate_id=run_plan.candidate_id,
                            candidate_label=run_plan.candidate_label,
                            model_id=run_plan.model_id,
                            status="failed",
                            message=message,
                            metrics={},
                            run_record=failed_run_record,
                            run_plan=run_plan,
                        )

                    if result.run_record is not None:
                        update_candidate_model_config(
                            candidate_id,
                            latest_run_id=result.run_record.get("run_id"),
                            latest_run_record=result.run_record,
                        )
                    execution_results.append(result)
                    _render_training_progress(
                        progress_bar_placeholder,
                        15 + int((index / max(total_candidates, 1)) * 70),
                    )
                    if str(result.status).strip().lower() == "failed":
                        progress_text.caption(f"Candidate failed: {result.candidate_label}")

                progress_text.caption("Finalizing results...")
                _render_training_progress(progress_bar_placeholder, 95)
                best_candidate_id = select_best_candidate_from_latest_results()
                set_best_candidate_id(best_candidate_id)
                _render_training_progress(progress_bar_placeholder, 100)
                progress_text.caption("Training complete.")

                completed_results = [
                    result for result in execution_results if str(result.status).strip().lower() == "completed"
                ]
                failed_results = [
                    result for result in execution_results if str(result.status).strip().lower() == "failed"
                ]

                if completed_results and not failed_results:
                    _set_training_notice(
                        f"Completed {len(completed_results)} candidate run(s).",
                        variant="success",
                        show_results_button=True,
                    )
                elif completed_results and failed_results:
                    failure_messages = "; ".join(result.message for result in failed_results[:2])
                    _set_training_notice(
                        f"Completed {len(completed_results)} candidate run(s), but {len(failed_results)} candidate run(s) failed. {failure_messages}",
                        variant="warning",
                        show_results_button=True,
                    )
                elif failed_results:
                    failure_messages = "; ".join(result.message for result in failed_results[:2])
                    _set_training_notice(
                        f"All candidate runs failed. {failure_messages}",
                        variant="error",
                        show_results_button=False,
                    )
                    render_status_message(
                        f"All candidate runs failed. {failure_messages}",
                        variant="error",
                    )
                else:
                    _set_training_notice(
                        "No candidate runs were started.",
                        variant="warning",
                        show_results_button=False,
                    )
                    render_status_message("No candidate runs were started.", variant="warning")
            except ModelingServiceError as exc:
                _clear_training_notice()
                render_status_message(str(exc), variant="error")

        training_notice = _get_training_notice()
        if training_notice is not None:
            _render_training_status_with_results_action(
                str(training_notice.get("message", "")),
                variant=str(training_notice.get("variant", "info")),
                show_results_button=bool(training_notice.get("show_results_button", False)),
            )



def render_modeling_panel() -> None:
    """Render the Model tab for the ML Workbench app."""
    state = get_app_state()
    problem_type = str(state.get("problem_type") or "").strip().lower()
    eligible_columns = _get_eligible_model_input_columns(state)
    model_specs = get_model_specs_for_problem_type(problem_type) if problem_type else []
    model_specs_by_id = {str(spec.get("model_id")): spec for spec in model_specs}

    _ensure_modeling_session_defaults(
        problem_type=problem_type,
        available_model_ids=list(model_specs_by_id.keys()) if problem_type else [],
    )

    candidate_panel = create_surface_panel(
        title="Candidate Models",
        subtitle="Choose one or more model types to configure and train using the current Working Data.",
    )
    with candidate_panel:
        badges = [
            f"Problem · {_problem_display(state.get('problem_type'))}",
            f"Target · {str(state.get('target_column') or 'Not set')}",
            f"Engineered Features · {_get_engineered_feature_count()}",
            f"Model Input Columns · {len(eligible_columns)}",
        ]
        render_badge_row(badges, variant="info")
        st.markdown("<hr style='border: none; border-top: 1px solid rgba(123, 129, 144, 0.28); margin: 0.8rem 0 0.9rem 0;'>", unsafe_allow_html=True)
        _render_candidate_selection_panel(problem_type, model_specs)
    _render_training_settings_panel(problem_type, model_specs_by_id)
    _render_model_specific_settings_panel(problem_type, eligible_columns, model_specs_by_id)
    _render_train_panel(model_specs_by_id)
