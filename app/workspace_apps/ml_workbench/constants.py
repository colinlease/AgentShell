

"""Constants for the ML Workbench workspace app.

This module centralizes stable identifiers, default configuration values,
and user-facing section names so the rest of the app can share a consistent
contract.
"""

from __future__ import annotations

# App identity
APP_ID = "ml_workbench"
APP_LABEL = "ML Workbench"
APP_TYPE = "streamlit"
APP_DESCRIPTION = (
    "A no-code machine learning workbench for structured tabular data. "
    "Users can upload data, profile it, preprocess it, engineer features, "
    "train candidate models, compare results, and export outputs."
)

# Session state keys
ML_WORKBENCH_STATE_KEY = "ml_workbench_state"
ML_WORKBENCH_ARTIFACTS_KEY = "ml_workbench_artifacts"
ML_WORKBENCH_BOOTSTRAP_KEY = "ml_workbench_bootstrapped"

# Workflow stages
STAGE_UPLOAD = "upload"
STAGE_PROFILE = "profile"
STAGE_PREPROCESS = "preprocess"
STAGE_FEATURES = "features"
STAGE_MODELING = "modeling"
STAGE_RESULTS = "results"
STAGE_EXPORT = "export"

WORKFLOW_STAGES = [
    STAGE_UPLOAD,
    STAGE_PROFILE,
    STAGE_PREPROCESS,
    STAGE_FEATURES,
    STAGE_MODELING,
    STAGE_RESULTS,
    STAGE_EXPORT,
]

WORKFLOW_STAGE_LABELS = {
    STAGE_UPLOAD: "Upload",
    STAGE_PROFILE: "Profile",
    STAGE_PREPROCESS: "Preprocess",
    STAGE_FEATURES: "Features",
    STAGE_MODELING: "Models",
    STAGE_RESULTS: "Results",
    STAGE_EXPORT: "Export",
}

# Artifact names
ARTIFACT_RAW_DATASET = "raw_dataset"
ARTIFACT_WORKING_DATASET = "working_dataset"
ARTIFACT_MODEL_INPUT_DATASET = "model_input_dataset"
ARTIFACT_TRAIN_DATASET = "train_dataset"
ARTIFACT_TEST_DATASET = "test_dataset"
ARTIFACT_MODEL_RESULTS_SUMMARY = "model_results_summary"

# Artifact kinds
ARTIFACT_KIND_DATASET = "dataset"
ARTIFACT_KIND_SPLIT = "split"
ARTIFACT_KIND_MODEL_RUN = "model_run"
ARTIFACT_KIND_RESULTS_TABLE = "results_table"
ARTIFACT_KIND_REPORT = "report"

# Artifact roles
ARTIFACT_ROLE_RAW = "raw"
ARTIFACT_ROLE_WORKING = "working"
ARTIFACT_ROLE_MODEL_INPUT = "model_input"
ARTIFACT_ROLE_TRAIN = "train"
ARTIFACT_ROLE_TEST = "test"
ARTIFACT_ROLE_RESULTS = "results"

# Problem types
PROBLEM_TYPE_CLASSIFICATION = "classification"
PROBLEM_TYPE_REGRESSION = "regression"
SUPPORTED_PROBLEM_TYPES = [
    PROBLEM_TYPE_CLASSIFICATION,
    PROBLEM_TYPE_REGRESSION,
]

# Training modes
TRAINING_MODE_DEFAULT = "default"
TRAINING_MODE_CUSTOM = "custom"
TRAINING_MODE_TUNED = "tuned"
SUPPORTED_TRAINING_MODES = [
    TRAINING_MODE_DEFAULT,
    TRAINING_MODE_CUSTOM,
    TRAINING_MODE_TUNED,
]

# Parameter modes
PARAMETER_MODE_DEFAULT = "default"
PARAMETER_MODE_CUSTOM = "custom"
PARAMETER_MODE_TUNED = "tuned"
SUPPORTED_PARAMETER_MODES = [
    PARAMETER_MODE_DEFAULT,
    PARAMETER_MODE_CUSTOM,
    PARAMETER_MODE_TUNED,
]

# Feature operations
FEATURE_OPERATION_FORMULA = "formula"
FEATURE_OPERATION_BINNING = "binning"
FEATURE_OPERATION_DATETIME_PART = "datetime_part"
FEATURE_OPERATION_CONDITION_FLAG = "condition_flag"
SUPPORTED_FEATURE_OPERATIONS = [
    FEATURE_OPERATION_FORMULA,
    FEATURE_OPERATION_BINNING,
    FEATURE_OPERATION_DATETIME_PART,
    FEATURE_OPERATION_CONDITION_FLAG,
]

# Feature types
FEATURE_TYPE_NUMERIC = "numeric"
FEATURE_TYPE_CATEGORICAL = "categorical"
FEATURE_TYPE_BINARY = "binary"
FEATURE_TYPE_DATETIME_PART = "datetime_part"
SUPPORTED_FEATURE_TYPES = [
    FEATURE_TYPE_NUMERIC,
    FEATURE_TYPE_CATEGORICAL,
    FEATURE_TYPE_BINARY,
    FEATURE_TYPE_DATETIME_PART,
]

# Feature provenance
CREATED_BY_USER = "user"
CREATED_BY_AGENT = "agent"
CREATED_BY_SYSTEM = "system"
SUPPORTED_CREATED_BY_VALUES = [
    CREATED_BY_USER,
    CREATED_BY_AGENT,
    CREATED_BY_SYSTEM,
]

# Feature / run status values
STATUS_ACTIVE = "active"
STATUS_REMOVED = "removed"
STATUS_ERROR = "error"
STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Missing value strategies
NUMERIC_IMPUTE_MEAN = "mean"
NUMERIC_IMPUTE_MEDIAN = "median"
NUMERIC_IMPUTE_CONSTANT = "constant"
SUPPORTED_NUMERIC_IMPUTATION_STRATEGIES = [
    NUMERIC_IMPUTE_MEAN,
    NUMERIC_IMPUTE_MEDIAN,
    NUMERIC_IMPUTE_CONSTANT,
]

CATEGORICAL_IMPUTE_MODE = "mode"
CATEGORICAL_IMPUTE_CONSTANT = "constant"
SUPPORTED_CATEGORICAL_IMPUTATION_STRATEGIES = [
    CATEGORICAL_IMPUTE_MODE,
    CATEGORICAL_IMPUTE_CONSTANT,
]

# Encoding strategies
ENCODING_ONE_HOT = "one_hot"
SUPPORTED_ENCODING_STRATEGIES = [ENCODING_ONE_HOT]

# Scaling strategies
SCALING_NONE = "none"
SCALING_STANDARD = "standard"
SCALING_MINMAX = "minmax"
SUPPORTED_SCALING_STRATEGIES = [
    SCALING_NONE,
    SCALING_STANDARD,
    SCALING_MINMAX,
]

# Rebalancing strategies
REBALANCING_NONE = "none"
REBALANCING_OVERSAMPLE = "oversample"
REBALANCING_UNDERSAMPLE = "undersample"
SUPPORTED_REBALANCING_STRATEGIES = [
    REBALANCING_NONE,
    REBALANCING_OVERSAMPLE,
    REBALANCING_UNDERSAMPLE,
]

# Tuning search types
TUNING_SEARCH_GRID = "grid"
TUNING_SEARCH_RANDOM = "random"
SUPPORTED_TUNING_SEARCH_TYPES = [
    TUNING_SEARCH_GRID,
    TUNING_SEARCH_RANDOM,
]

# Defaults
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
DEFAULT_PREVIEW_ROW_LIMIT = 50
DEFAULT_SAMPLE_ROW_LIMIT = 20
DEFAULT_MAX_CATEGORICAL_LEVELS = 20
DEFAULT_MAX_NUMERIC_SUMMARY_COLUMNS = 25
DEFAULT_MAX_DISPLAY_COLUMNS = 50

# Export file names
DEFAULT_PROCESSED_DATA_FILENAME = "processed_dataset.csv"
DEFAULT_RESULTS_SUMMARY_FILENAME = "model_results_summary.csv"
DEFAULT_TRANSFORMATION_REPORT_FILENAME = "transformation_report.json"

# User-facing empty-state messages
EMPTY_STATE_NO_DATASET = "Upload a CSV or Excel file to begin."
EMPTY_STATE_NO_TARGET = "Select a target column to configure the modeling workflow."
EMPTY_STATE_NO_MODELS = "Choose one or more candidate models to train."
EMPTY_STATE_NO_RESULTS = "Train candidate models to see evaluation results."