"""Typed schema definitions for the ML Workbench workspace app.

These schemas provide a lightweight, explicit contract for the app's core
state, artifacts, feature specs, model registry entries, training runs, and
published context. They are intentionally dependency-light and use TypedDict
so the rest of the app can stay modular without adding runtime validation
libraries at this stage.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class ArtifactMetadata(TypedDict, total=False):
    """Metadata describing a named artifact stored by the app."""

    rows: int
    columns: int
    column_names: list[str]
    dtype_summary: dict[str, str]
    missing_summary: dict[str, int]
    target_column: str | None
    problem_type: str | None
    note: str
    source_file_name: str | None
    created_from_stage: str | None
    ready_for_modeling: bool


class ArtifactRecord(TypedDict):
    """Registry entry for any stored artifact."""

    name: str
    kind: str
    role: str
    object: Any
    object_type: str
    created_at: str
    updated_at: str
    source_artifact: str | None
    metadata: ArtifactMetadata


class NumericImputationConfig(TypedDict):
    """Configuration for numeric missing value handling."""

    strategy: str | None
    fill_value: float | int | None
    columns: list[str]


class CategoricalImputationConfig(TypedDict):
    """Configuration for categorical missing value handling."""

    strategy: str | None
    fill_value: str | None
    columns: list[str]


class EncodingConfig(TypedDict):
    """Configuration for categorical encoding."""

    strategy: str | None
    columns: list[str]


class ScalingConfig(TypedDict):
    """Configuration for numeric scaling."""

    strategy: str | None
    columns: list[str]


class ClassRebalancingConfig(TypedDict):
    """Configuration for class rebalancing in classification workflows."""

    enabled: bool
    strategy: str | None


class DateTimeHandlingConfig(TypedDict):
    """Configuration for datetime detection and expansion."""

    auto_detect: bool
    expanded_columns: list[str]


class PreprocessingConfig(TypedDict):
    """Top-level shared preprocessing configuration for Working Data rebuilds."""

    drop_columns: list[str]
    numeric_imputation: NumericImputationConfig
    categorical_imputation: CategoricalImputationConfig
    datetime_handling: DateTimeHandlingConfig


class FeatureDependencySpec(TypedDict):
    """Dependency reference for one engineered feature input."""

    dependency_kind: str
    dependency_name: str


class FeatureValidationResult(TypedDict, total=False):
    """Validation metadata for one engineered feature definition."""

    is_valid: bool
    warnings: list[str]
    error_message: str | None


class FeatureSpec(TypedDict):
    """Definition of one engineered feature.

    This schema is intentionally broad enough to support both a guided builder
    and a future expression-editor workflow using the same underlying contract.
    """

    feature_id: str
    feature_name: str
    feature_type: str
    builder_mode: str
    operation_family: str
    operation: str
    expression: str
    expression_language: str | None
    parameters: dict[str, Any]
    source_columns: list[str]
    dependencies: list[FeatureDependencySpec]
    enabled: bool
    execution_scope: str
    apply_order: int
    created_by: str
    status: str
    error_message: NotRequired[str | None]
    description: NotRequired[str | None]
    output_dtype: NotRequired[str | None]
    validation: NotRequired[FeatureValidationResult]


class SplitConfig(TypedDict):
    """Configuration for train/test splitting."""

    enabled: bool
    test_size: float
    random_state: int
    stratify: bool


class ParamSchemaField(TypedDict, total=False):
    """Describes one configurable model parameter."""

    type: Literal["int", "float", "str", "bool", "list"]
    default: Any
    min: int | float
    max: int | float
    options: list[Any]
    description: str


class TuningSchema(TypedDict, total=False):
    """Describes how a model can support future tuning."""

    enabled: bool
    search_type_supported: list[str]
    search_space_defaults: dict[str, list[Any]]



class TuningConfig(TypedDict):
    """User-selected tuning configuration for training workflows."""

    enabled: bool
    search_type: str | None
    n_iter: int | None
    scoring: str | None
    per_model_search_space: dict[str, dict[str, list[Any]]]


class ModelSpec(TypedDict):
    """Registry definition for one supported model type."""

    model_id: str
    label: str
    problem_type: str
    family: str
    estimator_class_path: str
    default_metrics: list[str]
    param_schema: dict[str, ParamSchemaField]
    tuning_schema: TuningSchema
    description: NotRequired[str]


class CandidatePreprocessingConfig(TypedDict):
    """Model-specific preprocessing overrides applied on top of Working Data."""

    use_shared_preprocessing: bool
    numeric_imputation: NumericImputationConfig
    categorical_imputation: CategoricalImputationConfig
    encoding: EncodingConfig
    scaling: ScalingConfig
    class_rebalancing: ClassRebalancingConfig
    selected_feature_columns: list[str]
    excluded_feature_columns: list[str]


class CandidateModelConfig(TypedDict):
    """Configuration for one candidate model in the comparison workflow."""

    candidate_id: str
    candidate_label: str
    model_id: str
    enabled: bool
    preprocessing: CandidatePreprocessingConfig
    train_test_split_enabled: bool
    custom_params: dict[str, Any]
    classification_threshold: float | None
    tuning: TuningConfig
    notes: str
    latest_run_id: str | None


class ModelComparisonConfig(TypedDict):
    """Top-level comparison settings shared across candidate model runs."""

    evaluation_metric: str | None
    cv_folds: int
    split_strategy: str
    test_size: float
    random_seed: int
    random_state: int
    use_cross_validation: bool
    classification_threshold_policy: str
    classification_threshold_manual_value: float
    classification_threshold_objective: str
    default_parameter_mode: str
    candidate_models: list[CandidateModelConfig]


class ModelPlotData(TypedDict, total=False):
    """Plot-ready outputs associated with a trained model run."""

    roc_curve_data: dict[str, list[float]]
    confusion_matrix: list[list[int]]
    residual_plot_data: dict[str, list[float]]


class ModelRunRecord(TypedDict, total=False):
    """Artifact-like record describing a completed or attempted model run."""

    run_id: str
    candidate_id: str
    model_id: str
    model_label: str
    problem_type: str
    status: str
    training_mode: str
    input_artifact_name: str
    target_column: str
    feature_columns: list[str]
    fitted_object: Any
    preprocessing_summary: dict[str, Any]
    params_used: dict[str, Any]
    tuning_result: dict[str, Any] | None
    metrics: dict[str, Any]
    plots: ModelPlotData
    artifacts: dict[str, Any]
    notes: str
    started_at: str
    completed_at: str
    created_at: str
    split_strategy: str
    random_seed: int
    positive_class_label: object | None
    classification_threshold_policy: str
    classification_threshold_objective: str
    classification_threshold_source: str
    classification_threshold_used: float
    classification_threshold_manual_value: float
    classification_threshold_optimization_details: dict[str, Any]
    cv_classification_threshold_summary: list[dict[str, Any]]
    test_size: float
    train_row_count_original: int
    train_row_count_after_rebalancing: int
    test_row_count: int
    cv_folds: int
    error_message: NotRequired[str | None]


class StatusFlags(TypedDict):
    """Boolean workflow progress flags."""

    dataset_loaded: bool
    profile_ready: bool
    preprocessing_applied: bool
    features_applied: bool
    model_input_ready: bool
    models_trained: bool
    results_ready: bool


class UIStateConfig(TypedDict):
    """UI-only controls and selections for the Streamlit interface."""

    selected_profile_column: str | None
    selected_chart_type: str | None
    selected_model_ids: list[str]
    show_advanced_options: bool
    preview_row_limit: int


class AppState(TypedDict):
    """Authoritative session-backed state for the ML Workbench app."""

    app_stage: str
    loaded_file_name: str | None
    problem_type: str | None
    target_column: str | None
    positive_class_label: object | None
    id_columns: list[str]
    ignored_columns: list[str]
    selected_feature_columns: list[str]
    active_dataset_name: str | None
    active_candidate_id: str | None
    active_model_run_id: str | None
    best_candidate_id: str | None
    best_model_run_id: str | None
    preprocessing_config: PreprocessingConfig
    feature_specs: list[FeatureSpec]
    split_config: SplitConfig
    model_comparison_config: ModelComparisonConfig
    export_config: dict[str, Any]
    status: StatusFlags
    ui: UIStateConfig


class PublishedUIState(TypedDict, total=False):
    """Compact shell-facing UI state returned by get_ui_state()."""

    app_id: str
    app_label: str
    app_stage: str
    dataset_loaded: bool
    loaded_file_name: str | None
    problem_type: str | None
    target_column: str | None
    active_dataset_name: str | None
    active_candidate_id: str | None
    active_model_ids: list[str]
    models_trained: bool
    best_model_run_id: str | None


class PublishedDatasetContext(TypedDict, total=False):
    """Shell-facing summary for one published dataset artifact."""

    name: str
    type: str
    role: str
    rows: int
    columns: int
    column_names: list[str]
    dtype_summary: dict[str, str]
    missing_summary: dict[str, int]
    target_column: str | None
    problem_type: str | None
    ready_for_modeling: bool
    note: str


class PublishedModelingContext(TypedDict, total=False):
    """High-level modeling summary for shell/agent context."""

    problem_type: str | None
    target_column: str | None
    positive_class_label: object | None
    active_candidate_id: str | None
    active_model_ids: list[str]
    best_candidate_id: str | None
    candidate_models: list[CandidateModelConfig]
    best_model_run_id: str | None
    results_ready: bool
    split_strategy: str
    evaluation_mode: str
    test_size: float
    cv_folds: int
    random_seed: int
    classification_threshold_policy: str
    classification_threshold_manual_value: float
    classification_threshold_objective: str


class PublishedDataContext(TypedDict, total=False):
    """Shell-facing data context returned by get_data_context()."""

    has_data: bool
    active_dataset_name: str | None
    datasets: list[PublishedDatasetContext]
    modeling_context: PublishedModelingContext
