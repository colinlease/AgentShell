"""Session-backed state helpers for the ML Workbench app."""

from __future__ import annotations

from copy import deepcopy

import streamlit as st

from app.workspace_apps.ml_workbench.constants import (
    DEFAULT_CV_FOLDS,
    DEFAULT_PREVIEW_ROW_LIMIT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    ML_WORKBENCH_ARTIFACTS_KEY,
    ML_WORKBENCH_STATE_KEY,
    PARAMETER_MODE_DEFAULT,
    REBALANCING_NONE,
    SCALING_NONE,
    STAGE_UPLOAD,
)
from app.workspace_apps.ml_workbench.schemas import (
    AppState,
    ArtifactRecord,
    CandidateModelConfig,
    CandidatePreprocessingConfig,
    CategoricalImputationConfig,
    ClassRebalancingConfig,
    DateTimeHandlingConfig,
    EncodingConfig,
    FeatureSpec,
    ModelComparisonConfig,
    NumericImputationConfig,
    PreprocessingConfig,
    ScalingConfig,
    SplitConfig,
    StatusFlags,
    TuningConfig,
    UIStateConfig,
)


def build_default_numeric_imputation_config() -> NumericImputationConfig:
    """Return the default numeric imputation config."""
    return {
        "strategy": None,
        "fill_value": None,
        "columns": [],
    }


def build_default_categorical_imputation_config() -> CategoricalImputationConfig:
    """Return the default categorical imputation config."""
    return {
        "strategy": None,
        "fill_value": None,
        "columns": [],
    }


def build_default_encoding_config() -> EncodingConfig:
    """Return the default encoding config."""
    return {
        "strategy": None,
        "columns": [],
    }


def build_default_scaling_config() -> ScalingConfig:
    """Return the default scaling config."""
    return {
        "strategy": SCALING_NONE,
        "columns": [],
    }


def build_default_class_rebalancing_config() -> ClassRebalancingConfig:
    """Return the default class rebalancing config."""
    return {
        "enabled": False,
        "strategy": REBALANCING_NONE,
    }


def build_default_datetime_handling_config() -> DateTimeHandlingConfig:
    """Return the default datetime handling config."""
    return {
        "auto_detect": True,
        "expanded_columns": [],
    }


def build_default_preprocessing_config() -> PreprocessingConfig:
    """Return the default preprocessing config."""
    return {
        "drop_columns": [],
        "numeric_imputation": build_default_numeric_imputation_config(),
        "categorical_imputation": build_default_categorical_imputation_config(),
        "datetime_handling": build_default_datetime_handling_config(),
    }


def build_default_split_config() -> SplitConfig:
    """Return the default train/test split config."""
    return {
        "enabled": True,
        "test_size": DEFAULT_TEST_SIZE,
        "random_state": DEFAULT_RANDOM_STATE,
        "stratify": True,
    }


def build_default_tuning_config() -> TuningConfig:
    """Return the default tuning config."""
    return {
        "enabled": False,
        "search_type": None,
        "n_iter": None,
        "scoring": None,
        "per_model_search_space": {},
    }


def build_default_candidate_preprocessing_config() -> CandidatePreprocessingConfig:
    """Return the default candidate-level preprocessing overrides."""
    return {
        "use_shared_preprocessing": True,
        "numeric_imputation": build_default_numeric_imputation_config(),
        "categorical_imputation": build_default_categorical_imputation_config(),
        "encoding": build_default_encoding_config(),
        "scaling": build_default_scaling_config(),
        "class_rebalancing": build_default_class_rebalancing_config(),
        "selected_feature_columns": [],
        "excluded_feature_columns": [],
    }


def build_default_candidate_model_config(
    *,
    candidate_id: str,
    candidate_label: str,
    model_id: str,
) -> CandidateModelConfig:
    """Return the default config for one candidate model."""
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_label,
        "model_id": model_id,
        "enabled": True,
        "preprocessing": build_default_candidate_preprocessing_config(),
        "train_test_split_enabled": True,
        "custom_params": {},
        "classification_threshold": None,
        "tuning": build_default_tuning_config(),
        "notes": "",
        "latest_run_id": None,
        "latest_run_record": None,
    }


def build_default_model_comparison_config() -> ModelComparisonConfig:
    """Return the default model comparison config."""
    return {
        "evaluation_metric": None,
        "cv_folds": DEFAULT_CV_FOLDS,
        "split_strategy": "cross_validation",
        "test_size": DEFAULT_TEST_SIZE,
        "random_seed": DEFAULT_RANDOM_STATE,
        "random_state": DEFAULT_RANDOM_STATE,
        "use_cross_validation": True,
        "classification_threshold_policy": "Use model default",
        "classification_threshold_manual_value": 0.5,
        "classification_threshold_objective": "F1",
        "default_parameter_mode": PARAMETER_MODE_DEFAULT,
        "candidate_models": [],
    }


def build_default_status_flags() -> StatusFlags:
    """Return the default workflow status flags."""
    return {
        "dataset_loaded": False,
        "profile_ready": False,
        "preprocessing_applied": False,
        "features_applied": False,
        "model_input_ready": False,
        "models_trained": False,
        "results_ready": False,
    }


def build_default_ui_state() -> UIStateConfig:
    """Return the default UI-only state."""
    return {
        "selected_profile_column": None,
        "selected_chart_type": None,
        "selected_model_ids": [],
        "show_advanced_options": False,
        "preview_row_limit": DEFAULT_PREVIEW_ROW_LIMIT,
    }


def build_default_app_state() -> AppState:
    """Return the full default app state."""
    return {
        "app_stage": STAGE_UPLOAD,
        "loaded_file_name": None,
        "problem_type": None,
        "target_column": None,
        "positive_class_label": None,
        "id_columns": [],
        "ignored_columns": [],
        "selected_feature_columns": [],
        "active_dataset_name": None,
        "active_candidate_id": None,
        "active_model_run_id": None,
        "best_candidate_id": None,
        "best_model_run_id": None,
        "preprocessing_config": build_default_preprocessing_config(),
        "feature_specs": [],
        "split_config": build_default_split_config(),
        "model_comparison_config": build_default_model_comparison_config(),
        "export_config": {},
        "status": build_default_status_flags(),
        "ui": build_default_ui_state(),
    }


def ensure_app_state() -> AppState:
    """Ensure the ML Workbench app state exists in Streamlit session state."""
    if ML_WORKBENCH_STATE_KEY not in st.session_state:
        st.session_state[ML_WORKBENCH_STATE_KEY] = build_default_app_state()
    return st.session_state[ML_WORKBENCH_STATE_KEY]


def ensure_artifact_registry() -> dict[str, ArtifactRecord]:
    """Ensure the ML Workbench artifact registry exists in session state."""
    if ML_WORKBENCH_ARTIFACTS_KEY not in st.session_state:
        st.session_state[ML_WORKBENCH_ARTIFACTS_KEY] = {}
    return st.session_state[ML_WORKBENCH_ARTIFACTS_KEY]


def initialize_state() -> None:
    """Initialize all ML Workbench session-backed state containers."""
    ensure_app_state()
    ensure_artifact_registry()


def get_app_state() -> AppState:
    """Return the authoritative ML Workbench app state."""
    return ensure_app_state()


def set_app_state(state: AppState) -> None:
    """Replace the full ML Workbench app state."""
    st.session_state[ML_WORKBENCH_STATE_KEY] = state


def get_artifact_registry() -> dict[str, ArtifactRecord]:
    """Return the named artifact registry."""
    return ensure_artifact_registry()


def set_artifact_registry(registry: dict[str, ArtifactRecord]) -> None:
    """Replace the full artifact registry."""
    st.session_state[ML_WORKBENCH_ARTIFACTS_KEY] = registry


def reset_app_state(preserve_loaded_file_name: bool = False) -> AppState:
    """Reset app state to defaults.

    Parameters
    ----------
    preserve_loaded_file_name:
        When True, keep the current loaded file name in the new state.
    """
    current_state = ensure_app_state()
    new_state = build_default_app_state()
    if preserve_loaded_file_name:
        new_state["loaded_file_name"] = current_state.get("loaded_file_name")
    set_app_state(new_state)
    return new_state


def clear_artifact_registry() -> None:
    """Remove all named artifacts from session state."""
    set_artifact_registry({})


def reset_all_state() -> None:
    """Reset both app state and artifact registry to their defaults."""
    reset_app_state()
    clear_artifact_registry()


def get_state_value(key: str, default: object | None = None) -> object | None:
    """Return a top-level state value by key."""
    state = ensure_app_state()
    return state.get(key, default)


def set_state_value(key: str, value: object) -> None:
    """Set a top-level state value by key."""
    state = ensure_app_state()
    state[key] = value


def set_active_dataset_name(dataset_name: str | None) -> None:
    """Set the active dataset artifact name."""
    set_state_value("active_dataset_name", dataset_name)




def reconcile_column_state(valid_columns: list[str]) -> AppState:
    """Reconcile column-based modeling state against the currently valid columns.

    This keeps modeling-oriented selections usable when the active dataset
    changes, without deleting shared preprocessing rules that are meant to be
    evaluated against Raw Data during Working Data rebuilds.
    """
    state = ensure_app_state()
    valid_column_set = set(valid_columns)

    target_column = state.get("target_column")
    if target_column not in valid_column_set:
        state["target_column"] = None

    state["id_columns"] = [
        column for column in state.get("id_columns", []) if column in valid_column_set
    ]
    state["ignored_columns"] = [
        column for column in state.get("ignored_columns", []) if column in valid_column_set
    ]
    state["selected_feature_columns"] = [
        column
        for column in state.get("selected_feature_columns", [])
        if column in valid_column_set
    ]

    return state


def update_state_values(**updates: object) -> AppState:
    """Apply multiple top-level updates to the app state."""
    state = ensure_app_state()
    state.update(updates)
    return state


def get_status_flags() -> StatusFlags:
    """Return the workflow status flags."""
    return ensure_app_state()["status"]


def update_status_flags(**updates: bool) -> StatusFlags:
    """Update one or more workflow status flags."""
    status = ensure_app_state()["status"]
    status.update(updates)
    return status


def get_ui_state() -> UIStateConfig:
    """Return the UI-only state block."""
    return ensure_app_state()["ui"]


def update_ui_state(**updates: object) -> UIStateConfig:
    """Update one or more UI-only state values."""
    ui_state = ensure_app_state()["ui"]
    ui_state.update(updates)
    return ui_state


def get_preprocessing_config() -> PreprocessingConfig:
    """Return the preprocessing config block."""
    return ensure_app_state()["preprocessing_config"]


def set_preprocessing_config(config: PreprocessingConfig) -> None:
    """Replace the preprocessing config block."""
    ensure_app_state()["preprocessing_config"] = config


def update_preprocessing_config(**updates: object) -> PreprocessingConfig:
    """Update the top-level preprocessing config."""
    config = ensure_app_state()["preprocessing_config"]
    config.update(updates)
    return config


def get_feature_specs() -> list[FeatureSpec]:
    """Return the current list of engineered feature specs."""
    return ensure_app_state()["feature_specs"]


def set_feature_specs(feature_specs: list[FeatureSpec]) -> None:
    """Replace the full feature spec list."""
    ensure_app_state()["feature_specs"] = feature_specs


def append_feature_spec(feature_spec: FeatureSpec) -> list[FeatureSpec]:
    """Append one engineered feature spec to state."""
    feature_specs = ensure_app_state()["feature_specs"]
    feature_specs.append(feature_spec)
    return feature_specs


def get_feature_spec(feature_id: str) -> FeatureSpec | None:
    """Return one engineered feature spec by feature_id, if present."""
    for feature_spec in ensure_app_state()["feature_specs"]:
        if feature_spec.get("feature_id") == feature_id:
            return feature_spec
    return None


def update_feature_spec(feature_id: str, **updates: object) -> FeatureSpec | None:
    """Update one engineered feature spec by feature_id."""
    feature_spec = get_feature_spec(feature_id)
    if feature_spec is None:
        return None
    feature_spec.update(updates)
    return feature_spec


def remove_feature_spec(feature_id: str) -> list[FeatureSpec]:
    """Remove a feature spec by feature_id."""
    feature_specs = ensure_app_state()["feature_specs"]
    filtered = [spec for spec in feature_specs if spec["feature_id"] != feature_id]
    ensure_app_state()["feature_specs"] = filtered
    return filtered


def get_split_config() -> SplitConfig:
    """Return the split config block."""
    return ensure_app_state()["split_config"]


def set_split_config(config: SplitConfig) -> None:
    """Replace the split config block."""
    ensure_app_state()["split_config"] = config


def update_split_config(**updates: object) -> SplitConfig:
    """Update one or more split config values."""
    config = ensure_app_state()["split_config"]
    config.update(updates)
    return config


def get_model_comparison_config() -> ModelComparisonConfig:
    """Return the model comparison config block."""
    return ensure_app_state()["model_comparison_config"]


def set_model_comparison_config(config: ModelComparisonConfig) -> None:
    """Replace the model comparison config block."""
    ensure_app_state()["model_comparison_config"] = config


def update_model_comparison_config(**updates: object) -> ModelComparisonConfig:
    """Update one or more top-level model comparison config values."""
    config = ensure_app_state()["model_comparison_config"]
    config.update(updates)
    return config


def get_candidate_models() -> list[CandidateModelConfig]:
    """Return the list of configured candidate models."""
    return ensure_app_state()["model_comparison_config"]["candidate_models"]


def set_candidate_models(candidate_models: list[CandidateModelConfig]) -> None:
    """Replace the full candidate model list."""
    ensure_app_state()["model_comparison_config"]["candidate_models"] = candidate_models


def append_candidate_model(candidate_model: CandidateModelConfig) -> list[CandidateModelConfig]:
    """Append one candidate model config to the comparison state."""
    candidate_models = get_candidate_models()
    candidate_models.append(candidate_model)
    return candidate_models


def get_candidate_model(candidate_id: str) -> CandidateModelConfig | None:
    """Return one candidate model by id, if present."""
    for candidate_model in get_candidate_models():
        if candidate_model.get("candidate_id") == candidate_id:
            return candidate_model
    return None


def update_candidate_model(
    candidate_id: str,
    **updates: object,
) -> CandidateModelConfig | None:
    """Update one candidate model config by id."""
    candidate_model = get_candidate_model(candidate_id)
    if candidate_model is None:
        return None
    candidate_model.update(updates)
    return candidate_model


def remove_candidate_model(candidate_id: str) -> list[CandidateModelConfig]:
    """Remove one candidate model config by id."""
    filtered = [
        candidate_model
        for candidate_model in get_candidate_models()
        if candidate_model.get("candidate_id") != candidate_id
    ]
    set_candidate_models(filtered)

    state = ensure_app_state()
    if state.get("active_candidate_id") == candidate_id:
        state["active_candidate_id"] = None
    if state.get("best_candidate_id") == candidate_id:
        state["best_candidate_id"] = None

    return filtered


def get_active_candidate_id() -> str | None:
    """Return the currently active candidate id."""
    return ensure_app_state().get("active_candidate_id")


def set_active_candidate_id(candidate_id: str | None) -> None:
    """Set the currently active candidate id."""
    ensure_app_state()["active_candidate_id"] = candidate_id


def get_active_candidate() -> CandidateModelConfig | None:
    """Return the currently active candidate model, if present."""
    active_candidate_id = get_active_candidate_id()
    if active_candidate_id is None:
        return None
    return get_candidate_model(active_candidate_id)


def get_best_candidate_id() -> str | None:
    """Return the best candidate id currently stored in state."""
    return ensure_app_state().get("best_candidate_id")


def set_best_candidate_id(candidate_id: str | None) -> None:
    """Set the best candidate id currently stored in state."""
    ensure_app_state()["best_candidate_id"] = candidate_id


def clone_app_state() -> AppState:
    """Return a deep copy of the current app state."""
    return deepcopy(ensure_app_state())


def clone_artifact_registry() -> dict[str, ArtifactRecord]:
    """Return a deep copy of the current artifact registry."""
    return deepcopy(ensure_artifact_registry())
