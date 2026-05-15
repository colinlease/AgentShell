

"""Modeling orchestration service for the ML Workbench app.

Phase 1 establishes the backend contract for candidate-model workflows without
fully implementing training yet. The goal is to provide a clean, reusable
service layer that both the standalone UI and future AgentShell tools can call.

This module is intentionally organized around orchestration functions rather
than Streamlit UI concerns. The UI should call these functions, and future
AgentShell tools should call these same functions as well.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from app.workspace_apps.ml_workbench.schemas import (
    CandidateModelConfig,
    ModelComparisonConfig,
    ModelRunRecord,
)

from app.workspace_apps.ml_workbench.models.registry import (
    get_model_spec,
    get_model_specs_for_problem_type,
    is_supported_model,
    require_model_spec,
)

from app.workspace_apps.ml_workbench.services.dataset_service import (
    dataset_summary,
    get_dataset_copy,
)
from app.workspace_apps.ml_workbench.state import (
    append_candidate_model,
    get_active_candidate,
    get_active_candidate_id,
    get_app_state,
    get_candidate_model,
    get_candidate_models,
    get_model_comparison_config,
    remove_candidate_model,
    set_active_candidate_id,
    set_best_candidate_id,
    update_candidate_model,
    update_model_comparison_config,
)


DEFAULT_EVALUATION_METRIC_BY_PROBLEM_TYPE = {
    "classification": "roc_auc",
    "regression": "rmse",
}

COMPARISON_METRIC_PRIORITY_BY_PROBLEM_TYPE = {
    "classification": ["roc_auc", "f1", "accuracy"],
    "regression": ["rmse", "mae", "r2"],
}

MAX_ONE_HOT_CARDINALITY = 100

WORKING_DATASET_NAME = "working_dataset"

RESULTS_BUNDLE_EXPORT_VERSION = "1"


@dataclass(frozen=True)
class CandidateDatasetPlan:
    """Resolved plan for building one candidate's model input dataset.

    This is a design-time structure used to make the data dependencies explicit
    before full model-input construction is implemented.
    """

    candidate_id: str
    candidate_label: str
    model_id: str
    source_dataset_name: str
    use_shared_preprocessing: bool
    shared_preprocessing_applies: bool
    candidate_preprocessing: dict[str, Any]
    available_feature_columns: list[str]
    resolved_feature_columns: list[str]
    selected_feature_columns: list[str]
    excluded_feature_columns: list[str]
    encoding_columns: list[str]
    target_column: str | None
    identifier_columns: list[str]
    ignored_columns: list[str]


@dataclass(frozen=True)
class CandidateRunPlan:
    """Resolved training plan for one candidate model."""

    candidate_id: str
    candidate_label: str
    model_id: str
    source_dataset_name: str
    evaluation_metric: str | None
    split_strategy: str
    test_size: float
    cv_folds: int
    random_state: int
    use_cross_validation: bool
    train_test_split_enabled: bool
    positive_class_label: Any
    classification_threshold_policy: str
    classification_threshold_manual_value: float
    classification_threshold_objective: str
    classification_threshold: float | None
    custom_params: dict[str, Any]
    tuning: dict[str, Any]
    dataset_plan: CandidateDatasetPlan


@dataclass(frozen=True)
class CandidateExecutionResult:
    """Compact result object returned by phase-1 orchestration functions.

    In phase 1, this returns structured metadata and a status message rather
    than a trained sklearn object.
    """

    candidate_id: str
    candidate_label: str
    model_id: str
    status: str
    message: str
    metrics: dict[str, Any]
    run_record: ModelRunRecord | None
    run_plan: CandidateRunPlan


class ModelingServiceError(RuntimeError):
    """Raised when a modeling workflow request cannot be completed."""


@dataclass(frozen=True)
class RebalancingSummary:
    """Execution summary for one train-only class-rebalancing step."""

    applied: bool
    strategy: str
    original_class_counts: dict[str, int]
    rebalanced_class_counts: dict[str, int]


def _normalize_problem_type(problem_type: object) -> str | None:
    """Return a normalized problem-type string when possible."""
    if problem_type is None:
        return None
    value = str(problem_type).strip().lower()
    return value or None


def _normalize_rebalancing_strategy(strategy: object) -> str:
    """Return a normalized class-rebalancing strategy."""
    value = str(strategy or "none").strip().lower().replace("-", "_")
    return value or "none"



def _default_metric_for_problem_type(problem_type: object) -> str | None:
    """Return the default evaluation metric for a problem type."""
    return DEFAULT_EVALUATION_METRIC_BY_PROBLEM_TYPE.get(_normalize_problem_type(problem_type))


def _resolve_candidate_encoding_columns(
    candidate_preprocessing: dict[str, Any],
    column_names: list[str],
) -> list[str]:
    """Return the candidate-selected encoding columns that exist in the source dataset."""
    encoding_columns = candidate_preprocessing.get("encoding_columns", [])
    if not isinstance(encoding_columns, list):
        return []
    return [str(column) for column in encoding_columns if str(column) in column_names]


def _validate_one_hot_encoding_columns(
    *,
    source_dataset_name: str,
    encoding_strategy: object,
    encoding_columns: list[str],
) -> None:
    """Validate one-hot encoding selections before training.

    Numeric-coded categorical fields are allowed. The only hard guardrail in
    phase 1 is cardinality: one-hot encoding is blocked when a selected column
    has more than MAX_ONE_HOT_CARDINALITY distinct non-null values.
    """
    if str(encoding_strategy or "none").strip().lower() != "one_hot":
        return
    if not encoding_columns:
        return

    source_df = get_dataset_copy(source_dataset_name)
    for column_name in encoding_columns:
        if column_name not in source_df.columns:
            continue
        unique_count = int(source_df[column_name].dropna().nunique())
        if unique_count > MAX_ONE_HOT_CARDINALITY:
            raise ModelingServiceError(
                f"Column '{column_name}' has {unique_count} distinct non-null values, which exceeds the one-hot encoding limit of {MAX_ONE_HOT_CARDINALITY}. Choose a different column or remove it from the one-hot encoding selection."
            )


def _working_dataset_name() -> str:
    """Return the fixed dataset name used for modeling workflows."""
    return WORKING_DATASET_NAME



def _normalize_string_list(value: object) -> list[str]:
    """Return a cleaned list of strings from a candidate config field."""
    if not isinstance(value, list):
        return []
    cleaned_values: list[str] = []
    for item in value:
        item_text = str(item).strip()
        if item_text and item_text not in cleaned_values:
            cleaned_values.append(item_text)
    return cleaned_values



def _eligible_workspace_predictor_columns(column_names: list[str]) -> list[str]:
    """Return the workspace-level predictor pool from Working Data."""
    target_column = get_modeling_target_column()
    identifier_columns = set(get_modeling_identifier_columns())
    ignored_columns = set(get_modeling_ignored_columns())
    return [
        column_name
        for column_name in column_names
        if column_name != target_column and column_name not in identifier_columns and column_name not in ignored_columns
    ]



def _normalize_candidate_preprocessing_config(
    candidate_preprocessing: dict[str, Any],
    column_names: list[str],
) -> dict[str, Any]:
    """Normalize candidate preprocessing into one stable internal structure."""
    normalized_preprocessing = deepcopy(candidate_preprocessing)

    feature_subset_mode = str(
        normalized_preprocessing.get("feature_subset_mode", "Use all eligible predictors")
    ).strip()
    if feature_subset_mode not in {
        "Use all eligible predictors",
        "Exclude specific predictors",
        "Include only specific predictors",
    }:
        feature_subset_mode = "Use all eligible predictors"

    selected_feature_columns = [
        column for column in _normalize_string_list(normalized_preprocessing.get("selected_feature_columns", []))
        if column in column_names
    ]
    if not selected_feature_columns:
        selected_feature_columns = [
            column for column in _normalize_string_list(normalized_preprocessing.get("included_columns", []))
            if column in column_names
        ]

    excluded_feature_columns = [
        column for column in _normalize_string_list(normalized_preprocessing.get("excluded_feature_columns", []))
        if column in column_names
    ]
    if not excluded_feature_columns:
        excluded_feature_columns = [
            column for column in _normalize_string_list(normalized_preprocessing.get("excluded_columns", []))
            if column in column_names
        ]

    encoding_strategy = normalized_preprocessing.get("encoding_strategy")
    if encoding_strategy is None and isinstance(normalized_preprocessing.get("encoding"), dict):
        encoding_strategy = normalized_preprocessing.get("encoding", {}).get("strategy")
    encoding_strategy_text = str(encoding_strategy or "none").strip().lower().replace("-", "_")
    if encoding_strategy_text in {"onehot", "one_hot"}:
        encoding_strategy_text = "one_hot"
    elif encoding_strategy_text in {"", "none"}:
        encoding_strategy_text = "none"

    scaling_strategy = normalized_preprocessing.get("scaling_strategy")
    if scaling_strategy is None and isinstance(normalized_preprocessing.get("scaling"), dict):
        scaling_strategy = normalized_preprocessing.get("scaling", {}).get("strategy")
    scaling_strategy_text = str(scaling_strategy or "none").strip().lower().replace("-", "_") or "none"

    class_rebalancing_strategy = normalized_preprocessing.get("class_rebalancing_strategy")
    if class_rebalancing_strategy is None and isinstance(normalized_preprocessing.get("class_rebalancing"), dict):
        class_rebalancing_strategy = normalized_preprocessing.get("class_rebalancing", {}).get("strategy")
    class_rebalancing_strategy_text = (
        str(class_rebalancing_strategy or "none").strip().lower().replace("-", "_") or "none"
    )

    encoding_columns = _resolve_candidate_encoding_columns(normalized_preprocessing, column_names)

    if feature_subset_mode == "Include only specific predictors":
        excluded_feature_columns = []
    elif feature_subset_mode == "Exclude specific predictors":
        selected_feature_columns = []
    elif selected_feature_columns:
        feature_subset_mode = "Include only specific predictors"
        excluded_feature_columns = []
    elif excluded_feature_columns:
        feature_subset_mode = "Exclude specific predictors"
        selected_feature_columns = []

    normalized_preprocessing["feature_subset_mode"] = feature_subset_mode
    normalized_preprocessing["selected_feature_columns"] = selected_feature_columns
    normalized_preprocessing["excluded_feature_columns"] = excluded_feature_columns
    normalized_preprocessing["encoding_strategy"] = encoding_strategy_text
    normalized_preprocessing["encoding_columns"] = encoding_columns
    normalized_preprocessing["scaling_strategy"] = scaling_strategy_text
    normalized_preprocessing["class_rebalancing_strategy"] = class_rebalancing_strategy_text
    return normalized_preprocessing



def _resolve_candidate_feature_columns(
    *,
    available_feature_columns: list[str],
    feature_subset_mode: str,
    selected_feature_columns: list[str],
    excluded_feature_columns: list[str],
) -> list[str]:
    """Resolve the final predictor columns for one candidate."""
    if feature_subset_mode == "Include only specific predictors":
        resolved_columns = [
            column_name for column_name in available_feature_columns if column_name in selected_feature_columns
        ]
    else:
        resolved_columns = list(available_feature_columns)

    if feature_subset_mode == "Exclude specific predictors" and excluded_feature_columns:
        excluded_set = set(excluded_feature_columns)
        resolved_columns = [column_name for column_name in resolved_columns if column_name not in excluded_set]

    return resolved_columns



def _validate_candidate_dataset_plan_inputs(
    *,
    candidate_label: str,
    target_column: str | None,
    available_feature_columns: list[str],
    resolved_feature_columns: list[str],
    encoding_columns: list[str],
    source_dataset_name: str,
    encoding_strategy: str,
) -> None:
    """Validate the resolved candidate dataset inputs before training."""
    if not target_column:
        raise ModelingServiceError(
            f"Candidate '{candidate_label}' cannot be prepared because no target column is set in the Data tab."
        )
    if not available_feature_columns:
        raise ModelingServiceError(
            f"Candidate '{candidate_label}' cannot be prepared because Working Data has no eligible predictor columns after removing the target, ignored columns, and identifier columns."
        )
    if not resolved_feature_columns:
        raise ModelingServiceError(
            f"Candidate '{candidate_label}' cannot be prepared because no predictor columns remain after applying the selected include/exclude settings."
        )
    invalid_encoding_columns = [
        column_name for column_name in encoding_columns if column_name not in resolved_feature_columns
    ]
    if invalid_encoding_columns:
        invalid_columns_text = ", ".join(invalid_encoding_columns)
        raise ModelingServiceError(
            f"Candidate '{candidate_label}' has one-hot encoding columns that are not part of the resolved predictor set: {invalid_columns_text}."
        )
    _validate_one_hot_encoding_columns(
        source_dataset_name=source_dataset_name,
        encoding_strategy=encoding_strategy,
        encoding_columns=encoding_columns,
    )


def _load_candidate_training_dataframe(run_plan: CandidateRunPlan) -> pd.DataFrame:
    """Load the source dataframe for one candidate run."""
    source_df = get_dataset_copy(run_plan.source_dataset_name)
    if source_df.empty:
        raise ModelingServiceError("Working Data is empty and cannot be used for training.")
    return source_df


def _build_candidate_xy(
    source_df: pd.DataFrame,
    run_plan: CandidateRunPlan,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build X and y from the resolved candidate run plan."""
    target_column = run_plan.dataset_plan.target_column
    if not target_column or target_column not in source_df.columns:
        raise ModelingServiceError("The target column is missing from Working Data.")

    feature_columns = list(run_plan.dataset_plan.resolved_feature_columns)
    if not feature_columns:
        raise ModelingServiceError("No resolved predictor columns are available for training.")

    missing_feature_columns = [column for column in feature_columns if column not in source_df.columns]
    if missing_feature_columns:
        missing_columns_text = ", ".join(missing_feature_columns)
        raise ModelingServiceError(
            f"Working Data is missing one or more resolved predictor columns: {missing_columns_text}."
        )

    X = source_df[feature_columns].copy()
    y = source_df[target_column].copy()

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    if X.empty or y.empty:
        raise ModelingServiceError("No training rows remain after removing records with missing target values.")

    return X, y


def _resolve_encoded_vs_passthrough_columns(
    run_plan: CandidateRunPlan,
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Split resolved features into encoded and passthrough sets."""
    encoded_columns = [
        column_name for column_name in run_plan.dataset_plan.encoding_columns if column_name in X.columns
    ]
    passthrough_columns = [
        column_name for column_name in list(X.columns) if column_name not in set(encoded_columns)
    ]
    return encoded_columns, passthrough_columns


def _validate_unencoded_feature_types(
    run_plan: CandidateRunPlan,
    X: pd.DataFrame,
    encoded_columns: list[str],
) -> None:
    """Ensure non-encoded features are numeric-compatible for sklearn estimators."""
    invalid_columns: list[str] = []
    for column_name in X.columns:
        if column_name in encoded_columns:
            continue
        series = X[column_name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        invalid_columns.append(column_name)

    if invalid_columns:
        invalid_columns_text = ", ".join(invalid_columns)
        raise ModelingServiceError(
            f"Candidate '{run_plan.candidate_label}' includes non-numeric predictor columns that are not selected for one-hot encoding: {invalid_columns_text}. Add these columns to the one-hot encoding selection or remove them from the predictor set."
        )



def _build_column_transformer(run_plan: CandidateRunPlan, X: pd.DataFrame) -> ColumnTransformer | str:
    """Build the sklearn column transformer for one candidate run."""
    candidate_preprocessing = run_plan.dataset_plan.candidate_preprocessing
    encoded_columns, passthrough_columns = _resolve_encoded_vs_passthrough_columns(run_plan, X)
    _validate_unencoded_feature_types(run_plan, X, encoded_columns)

    scaling_strategy = str(candidate_preprocessing.get("scaling_strategy", "none"))
    transformers: list[tuple[str, Any, list[str]]] = []

    if encoded_columns:
        transformers.append(
            (
                "encoded",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                encoded_columns,
            )
        )

    if passthrough_columns:
        if scaling_strategy == "standard":
            transformers.append(
                (
                    "scaled_numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    passthrough_columns,
                )
            )
        elif scaling_strategy == "minmax":
            transformers.append(
                (
                    "scaled_numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", MinMaxScaler()),
                        ]
                    ),
                    passthrough_columns,
                )
            )
        else:
            transformers.append(
                (
                    "numeric",
                    Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                    passthrough_columns,
                )
            )

    if not transformers:
        return "passthrough"

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _transformed_feature_count(preprocessor: ColumnTransformer | str, X: pd.DataFrame) -> int:
    """Return the effective feature count after candidate preprocessing."""
    if preprocessor == "passthrough":
        return int(X.shape[1])
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "shape") and len(transformed.shape) >= 2:
        return int(transformed.shape[1])
    return int(X.shape[1])


def _adjusted_r2_value(
    r2_value: float,
    sample_count: int,
    predictor_count: int,
) -> float | None:
    """Return adjusted R-squared when sample size supports it."""
    n = int(sample_count)
    p = int(predictor_count)
    if n <= p + 1:
        return None
    return float(1.0 - ((1.0 - float(r2_value)) * (n - 1) / (n - p - 1)))


def _adjusted_r2_scorer(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """Score one fitted regression fold using adjusted R-squared."""
    y_pred = estimator.predict(X)
    r2_value = float(r2_score(y, y_pred))
    preprocessor = estimator.named_steps.get("preprocessor", "passthrough")
    predictor_count = _transformed_feature_count(preprocessor, X)
    adjusted_r2 = _adjusted_r2_value(r2_value, len(y), predictor_count)
    if adjusted_r2 is None:
        return float("nan")
    return adjusted_r2


# --- Helper functions for estimator and metrics parameter normalization and scoring ---

def _sanitize_estimator_params(model_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Normalize UI-provided hyperparameters into sklearn-compatible values."""
    normalized_params = deepcopy(params)

    if model_id in {"random_forest_classifier", "random_forest_regressor"}:
        if normalized_params.get("max_depth", None) == "":
            normalized_params["max_depth"] = None
        elif normalized_params.get("max_depth", None) is not None:
            try:
                normalized_params["max_depth"] = int(normalized_params["max_depth"])
            except (TypeError, ValueError):
                normalized_params["max_depth"] = None

        for integer_param in {"n_estimators", "min_samples_split", "min_samples_leaf"}:
            if integer_param not in normalized_params:
                continue
            try:
                normalized_params[integer_param] = int(normalized_params[integer_param])
            except (TypeError, ValueError):
                pass

    if model_id == "logistic_regression":
        if "C" in normalized_params:
            try:
                normalized_params["C"] = float(normalized_params["C"])
            except (TypeError, ValueError):
                pass
        if "max_iter" in normalized_params:
            try:
                normalized_params["max_iter"] = int(normalized_params["max_iter"])
            except (TypeError, ValueError):
                pass

    return normalized_params


def _classification_positive_label(y: pd.Series) -> Any:
    """Return the positive label to use for binary classification metrics."""
    unique_values = [value for value in list(pd.Series(y).dropna().unique())]
    preferred_labels = ["Good", "good", True, 1, "1", "yes", "Yes", "TRUE", "true"]
    for preferred_label in preferred_labels:
        if preferred_label in unique_values:
            return preferred_label
    if len(unique_values) == 2:
        try:
            return sorted(unique_values, key=lambda value: str(value))[-1]
        except Exception:
            return unique_values[-1]
    return unique_values[0] if unique_values else None


def _configured_positive_class_label(run_plan: CandidateRunPlan, y: pd.Series) -> Any:
    """Return the configured positive class label when it is valid for the target."""
    configured_label = run_plan.positive_class_label
    labels = _binary_target_labels(y)
    if configured_label in labels:
        return configured_label
    return None


def _validate_selected_positive_class_label(
    run_plan: CandidateRunPlan,
    y: pd.Series,
    *,
    context: str,
) -> Any:
    """Require an explicit valid positive-class label for binary classification."""
    labels = _binary_target_labels(y)
    configured_label = run_plan.positive_class_label
    if configured_label in labels:
        return configured_label
    labels_text = ", ".join(str(label) for label in labels)
    raise ModelingServiceError(
        f"Binary classification requires selecting a valid positive class label in the Data tab. "
        f"The selected target in {context} has classes: {labels_text}."
    )


def _resolved_positive_class_label(run_plan: CandidateRunPlan, y: pd.Series) -> Any:
    """Return the effective positive class label for one binary classification target."""
    configured_label = _configured_positive_class_label(run_plan, y)
    if configured_label is not None:
        return configured_label
    return _classification_positive_label(y)


def _binary_target_labels(y: pd.Series) -> list[Any]:
    """Return the two non-null target labels when the target is binary."""
    return [value for value in list(pd.Series(y).dropna().unique())]


def _stringify_class_counts(class_counts: dict[Any, int]) -> dict[str, int]:
    """Convert label-count mappings into JSON-friendly string-keyed payloads."""
    return {str(label): int(count) for label, count in class_counts.items()}


def _class_count_summary(y: pd.Series) -> dict[str, int]:
    """Return raw target label counts using original label values."""
    return _stringify_class_counts(pd.Series(y).value_counts(dropna=True).to_dict())


def _classification_rebalancing_strategy(run_plan: CandidateRunPlan) -> str:
    """Return the configured classification rebalancing strategy for one run."""
    preprocessing = run_plan.dataset_plan.candidate_preprocessing
    return _normalize_rebalancing_strategy(preprocessing.get("class_rebalancing_strategy", "none"))


def _classification_threshold_policy(run_plan: CandidateRunPlan) -> str:
    """Return the normalized shared classification-threshold policy."""
    value = str(run_plan.classification_threshold_policy or "Use model default").strip().lower()
    if value in {"use model default", "default"}:
        return "use_model_default"
    if value in {"set manual threshold", "manual"}:
        return "manual"
    if value in {"optimize threshold", "optimize"}:
        return "optimize"
    return "use_model_default"


def _classification_threshold_source(run_plan: CandidateRunPlan) -> str:
    """Return the user-facing threshold source for one run."""
    policy = _classification_threshold_policy(run_plan)
    if policy == "manual":
        return "manual"
    if policy == "optimize":
        return "optimized"
    return "default"


def _classification_threshold_objective(run_plan: CandidateRunPlan) -> str:
    """Return the normalized threshold-optimization objective."""
    value = str(run_plan.classification_threshold_objective or "F1").strip().lower()
    if value == "precision":
        return "precision"
    if value == "recall":
        return "recall"
    return "f1"


def _classification_threshold_value(run_plan: CandidateRunPlan) -> float:
    """Return the fixed decision threshold for default/manual policies."""
    policy = _classification_threshold_policy(run_plan)
    if policy == "optimize":
        raise ModelingServiceError(
            "Threshold optimization is not implemented yet. Use model default or set a manual threshold for now."
        )
    if policy == "manual":
        threshold_value = float(run_plan.classification_threshold_manual_value)
        if not 0.0 < threshold_value < 1.0:
            raise ModelingServiceError(
                "Manual classification threshold must be greater than 0 and less than 1."
            )
        return threshold_value
    return 0.5


def _classification_negative_label(y: pd.Series, positive_label: Any) -> Any:
    """Return the negative binary label paired with the resolved positive label."""
    labels = _binary_target_labels(y)
    for label in labels:
        if label != positive_label:
            return label
    return None


def _threshold_tuning_validation_fraction(y: pd.Series) -> float:
    """Return a safe inner-validation fraction for threshold tuning."""
    class_counts = pd.Series(y).value_counts(dropna=True)
    if class_counts.empty or int(class_counts.min()) < 2:
        raise ModelingServiceError(
            "Threshold optimization requires at least 2 rows in each target class in the training partition."
        )
    minimum_class_count = int(class_counts.min())
    return max(0.2, 1.0 / float(minimum_class_count))


def _threshold_predictions_from_scores(
    y_score: pd.Series,
    *,
    threshold: float,
    positive_label: Any,
    negative_label: Any,
) -> pd.Series:
    """Convert positive-class scores into label predictions using a fixed threshold."""
    return pd.Series(
        [positive_label if float(score) >= float(threshold) else negative_label for score in y_score],
        index=y_score.index,
    )


def _threshold_objective_value(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    positive_label: Any,
    objective: str,
) -> float:
    """Return the optimization score for one threshold candidate."""
    if objective == "precision":
        return float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    if objective == "recall":
        return float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    return float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0))


def _optimize_classification_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    positive_label: Any,
    negative_label: Any,
    objective: str,
) -> tuple[float, float]:
    """Return the best threshold and objective value for binary classification."""
    candidate_thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_score = float("-inf")
    best_distance = float("inf")
    for threshold in candidate_thresholds:
        y_pred = _threshold_predictions_from_scores(
            y_score,
            threshold=float(threshold),
            positive_label=positive_label,
            negative_label=negative_label,
        )
        objective_score = _threshold_objective_value(
            y_true,
            y_pred,
            positive_label=positive_label,
            objective=objective,
        )
        threshold_distance = abs(float(threshold) - 0.5)
        if (
            objective_score > best_score
            or (objective_score == best_score and threshold_distance < best_distance)
        ):
            best_score = objective_score
            best_threshold = float(threshold)
            best_distance = threshold_distance
    return best_threshold, best_score


def _tuned_classification_threshold_for_training_partition(
    run_plan: CandidateRunPlan,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    positive_label: Any,
    partition_label: str,
    random_state: int,
) -> tuple[float, dict[str, Any]]:
    """Tune a binary classification threshold on an inner validation split."""
    validation_fraction = _threshold_tuning_validation_fraction(y_train)
    X_inner_train, X_inner_validation, y_inner_train, y_inner_validation = train_test_split(
        X_train,
        y_train,
        test_size=float(validation_fraction),
        random_state=random_state,
        stratify=y_train,
    )
    _validate_training_partition_labels(y_inner_train, partition_label=f"{partition_label} inner training split")

    X_inner_fit = X_inner_train
    y_inner_fit = y_inner_train
    if _normalize_problem_type(get_modeling_problem_type()) == "classification":
        X_inner_fit, y_inner_fit, _ = _rebalance_training_partition(
            X_inner_train,
            y_inner_train,
            strategy=_classification_rebalancing_strategy(run_plan),
            random_state=random_state,
        )

    tuning_pipeline = _build_training_pipeline(run_plan, X_inner_fit)
    tuning_pipeline.fit(X_inner_fit, y_inner_fit)
    y_score = _positive_class_score_from_pipeline(tuning_pipeline, X_inner_validation, positive_label)
    if y_score is None:
        raise ModelingServiceError(
            "Threshold optimization requires a classifier that exposes positive-class probabilities."
        )
    negative_label = _classification_negative_label(y_train, positive_label)
    threshold_objective = _classification_threshold_objective(run_plan)
    threshold_value, objective_score = _optimize_classification_threshold(
        y_inner_validation,
        y_score,
        positive_label=positive_label,
        negative_label=negative_label,
        objective=threshold_objective,
    )
    return threshold_value, {
        "partition": partition_label,
        "objective": threshold_objective,
        "objective_score": float(objective_score),
        "validation_fraction": float(validation_fraction),
    }


def _validate_binary_classification_target(
    y: pd.Series,
    *,
    run_plan: CandidateRunPlan,
    context: str,
) -> list[Any]:
    """Validate that a classification target is strictly binary."""
    problem_type = _normalize_problem_type(get_modeling_problem_type())
    rebalancing_strategy = _classification_rebalancing_strategy(run_plan)
    if problem_type != "classification":
        if rebalancing_strategy != "none":
            raise ModelingServiceError(
                "Class rebalancing is only supported for binary classification problems."
            )
        return []

    labels = _binary_target_labels(y)
    if len(labels) != 2:
        raise ModelingServiceError(
            f"Binary classification is required for this workflow. The target has {len(labels)} distinct non-null classes in {context}. Non-numeric binary labels such as 'GOOD' and 'BAD' are supported."
        )
    _validate_selected_positive_class_label(run_plan, y, context=context)
    return labels


def _validate_training_partition_labels(
    y: pd.Series,
    *,
    partition_label: str,
) -> None:
    """Ensure a training partition still contains both target classes."""
    labels = _binary_target_labels(y)
    if len(labels) < 2:
        raise ModelingServiceError(
            f"Training cannot continue because the {partition_label} contains only one target class after splitting."
        )


def _validate_classification_cv_feasibility(y: pd.Series, *, cv_folds: int) -> None:
    """Ensure stratified cross-validation can be built for a binary target."""
    class_counts = pd.Series(y).value_counts(dropna=True)
    if class_counts.empty:
        raise ModelingServiceError("Binary classification requires two target classes.")
    minimum_class_count = int(class_counts.min())
    if minimum_class_count < int(cv_folds):
        raise ModelingServiceError(
            f"Cross-validation requires at least {cv_folds} rows in each target class, but the smallest class has {minimum_class_count} rows."
        )


def _rebalance_training_partition(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    strategy: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, RebalancingSummary]:
    """Apply train-only class rebalancing for binary targets."""
    normalized_strategy = _normalize_rebalancing_strategy(strategy)
    original_counts_raw = pd.Series(y_train).value_counts(dropna=True).to_dict()
    original_counts = _stringify_class_counts(original_counts_raw)
    if normalized_strategy == "none":
        return (
            X_train,
            y_train,
            RebalancingSummary(
                applied=False,
                strategy=normalized_strategy,
                original_class_counts=original_counts,
                rebalanced_class_counts=original_counts,
            ),
        )

    _validate_training_partition_labels(y_train, partition_label="training partition")
    sorted_counts = sorted(
        original_counts_raw.items(),
        key=lambda item: (item[1], str(item[0])),
    )
    minority_label, minority_count = sorted_counts[0]
    majority_label, majority_count = sorted_counts[-1]
    if minority_count == majority_count:
        return (
            X_train,
            y_train,
            RebalancingSummary(
                applied=False,
                strategy=normalized_strategy,
                original_class_counts=original_counts,
                rebalanced_class_counts=original_counts,
            ),
        )

    train_df = X_train.copy()
    target_name = y_train.name if y_train.name is not None else "__mlw_target__"
    train_df[target_name] = y_train

    minority_rows = train_df[train_df[target_name] == minority_label]
    majority_rows = train_df[train_df[target_name] == majority_label]

    if normalized_strategy == "oversample":
        sampled_minority_rows = minority_rows.sample(
            n=int(majority_count),
            replace=True,
            random_state=random_state,
        )
        rebalanced_df = pd.concat([majority_rows, sampled_minority_rows], axis=0)
    elif normalized_strategy == "undersample":
        sampled_majority_rows = majority_rows.sample(
            n=int(minority_count),
            replace=False,
            random_state=random_state,
        )
        rebalanced_df = pd.concat([sampled_majority_rows, minority_rows], axis=0)
    else:
        raise ModelingServiceError(
            f"Class rebalancing strategy '{normalized_strategy}' is not supported."
        )

    rebalanced_df = rebalanced_df.sample(frac=1.0, random_state=random_state)
    rebalanced_y = rebalanced_df[target_name].copy()
    rebalanced_X = rebalanced_df.drop(columns=[target_name]).copy()
    rebalanced_counts = _class_count_summary(rebalanced_y)
    return (
        rebalanced_X,
        rebalanced_y,
        RebalancingSummary(
            applied=True,
            strategy=normalized_strategy,
            original_class_counts=original_counts,
            rebalanced_class_counts=rebalanced_counts,
        ),
    )


def _positive_class_score_from_pipeline(pipeline: Pipeline, X: pd.DataFrame, positive_label: Any) -> pd.Series | None:
    """Return positive-class scores from a fitted classifier pipeline when available."""
    estimator = pipeline.named_steps["estimator"]
    if not hasattr(estimator, "predict_proba"):
        return None

    probabilities = pipeline.predict_proba(X)
    classes = list(getattr(estimator, "classes_", []))
    if positive_label in classes:
        positive_index = classes.index(positive_label)
    elif len(classes) == 2:
        positive_index = 1
    else:
        return None
    return pd.Series(probabilities[:, positive_index], index=X.index)


def _build_estimator(run_plan: CandidateRunPlan) -> Any:
    """Build the sklearn estimator for one candidate run."""
    params = _sanitize_estimator_params(run_plan.model_id, deepcopy(run_plan.custom_params))
    model_id = run_plan.model_id

    if model_id == "logistic_regression":
        return LogisticRegression(**params)
    if model_id == "random_forest_classifier":
        return RandomForestClassifier(random_state=run_plan.random_state, **params)
    if model_id == "linear_regression":
        return LinearRegression(**params)
    if model_id == "random_forest_regressor":
        return RandomForestRegressor(random_state=run_plan.random_state, **params)

    raise ModelingServiceError(f"Training is not implemented for model '{model_id}'.")


def _build_training_pipeline(run_plan: CandidateRunPlan, X: pd.DataFrame) -> Pipeline:
    """Build the sklearn preprocessing + estimator pipeline for one run."""
    preprocessor = _build_column_transformer(run_plan, X)
    estimator = _build_estimator(run_plan)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ]
    )


def _classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    positive_label: Any | None = None,
    y_score: pd.Series | None = None,
) -> dict[str, float]:
    """Return standard classification metrics for one fitted run."""
    resolved_positive_label = positive_label if positive_label is not None else _classification_positive_label(y_true)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    precision_kwargs = {"zero_division": 0}
    recall_kwargs = {"zero_division": 0}
    f1_kwargs = {"zero_division": 0}
    if resolved_positive_label is not None and len(pd.Series(y_true).dropna().unique()) == 2:
        precision_kwargs["pos_label"] = resolved_positive_label
        recall_kwargs["pos_label"] = resolved_positive_label
        f1_kwargs["pos_label"] = resolved_positive_label

    metrics["precision"] = float(precision_score(y_true, y_pred, **precision_kwargs))
    metrics["recall"] = float(recall_score(y_true, y_pred, **recall_kwargs))
    metrics["f1"] = float(f1_score(y_true, y_pred, **f1_kwargs))

    if y_score is not None and resolved_positive_label is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    return metrics


def _classification_scoring_summary(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    positive_label: Any,
    y_score: pd.Series | None = None,
) -> dict[str, float]:
    """Return classification scoring metrics for one evaluation partition."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def _regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    predictor_count: int,
) -> dict[str, float]:
    """Return standard regression metrics for one fitted run."""
    rmse_value = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2_value = float(r2_score(y_true, y_pred))
    metrics = {
        "rmse": rmse_value,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2_value,
    }
    adjusted_r2 = _adjusted_r2_value(r2_value, len(y_true), predictor_count)
    if adjusted_r2 is not None:
        metrics["adjusted_r2"] = adjusted_r2
    return metrics


def _run_train_test_evaluation(
    run_plan: CandidateRunPlan,
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
) -> dict[str, Any]:
    """Fit/evaluate one candidate using a single train/test split."""
    problem_type = _normalize_problem_type(get_modeling_problem_type())
    if problem_type == "classification":
        _validate_binary_classification_target(y, run_plan=run_plan, context="the selected target column")
    stratify = y if problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(run_plan.test_size),
        random_state=run_plan.random_state,
        stratify=stratify,
    )

    threshold_value: float | None = None
    threshold_details: dict[str, Any] | None = None
    if problem_type == "classification" and _classification_threshold_policy(run_plan) == "optimize":
        positive_label = _resolved_positive_class_label(run_plan, y_train)
        threshold_value, threshold_details = _tuned_classification_threshold_for_training_partition(
            run_plan,
            X_train,
            y_train,
            positive_label=positive_label,
            partition_label="training split",
            random_state=run_plan.random_state,
        )

    original_train_row_count = int(len(X_train))
    rebalanced_train_row_count = original_train_row_count
    rebalancing_summary: RebalancingSummary | None = None
    if problem_type == "classification":
        _validate_training_partition_labels(y_train, partition_label="training split")
        rebalancing_strategy = _classification_rebalancing_strategy(run_plan)
        X_train, y_train, rebalancing_summary = _rebalance_training_partition(
            X_train,
            y_train,
            strategy=rebalancing_strategy,
            random_state=run_plan.random_state,
        )
        rebalanced_train_row_count = int(len(X_train))

    pipeline.fit(X_train, y_train)

    if problem_type == "classification":
        positive_label = _resolved_positive_class_label(run_plan, y_train)
        y_score = _positive_class_score_from_pipeline(pipeline, X_test, positive_label)
        if y_score is None and _classification_threshold_policy(run_plan) in {"manual", "optimize"}:
            raise ModelingServiceError(
                "Thresholding requires a classifier that exposes positive-class probabilities."
            )
        if y_score is not None:
            negative_label = _classification_negative_label(y_train, positive_label)
            if _classification_threshold_policy(run_plan) != "optimize":
                threshold_value = _classification_threshold_value(run_plan)
            y_pred = _threshold_predictions_from_scores(
                y_score,
                threshold=float(threshold_value),
                positive_label=positive_label,
                negative_label=negative_label,
            )
        else:
            y_pred = pipeline.predict(X_test)
        metrics = _classification_metrics(y_test, y_pred, positive_label=positive_label, y_score=y_score)
    else:
        y_pred = pipeline.predict(X_test)
        preprocessor = pipeline.named_steps.get("preprocessor", "passthrough")
        predictor_count = _transformed_feature_count(preprocessor, X_test)
        metrics = _regression_metrics(y_test, y_pred, predictor_count=predictor_count)

    metrics["row_count"] = int(len(X))
    metrics["train_row_count"] = original_train_row_count
    metrics["test_row_count"] = int(len(X_test))
    metrics["split_strategy"] = run_plan.split_strategy
    metrics["test_size"] = float(run_plan.test_size)
    metrics["random_seed"] = int(run_plan.random_state)
    metrics["train_row_count_after_rebalancing"] = rebalanced_train_row_count
    if problem_type == "classification":
        metrics["classification_threshold_source"] = _classification_threshold_source(run_plan)
        metrics["classification_threshold_policy"] = run_plan.classification_threshold_policy
        metrics["classification_threshold_objective"] = run_plan.classification_threshold_objective
        if threshold_value is not None:
            metrics["classification_threshold_used"] = float(threshold_value)
        if threshold_details is not None:
            metrics["classification_threshold_optimization_details"] = threshold_details
    if rebalancing_summary is not None:
        metrics["rebalancing_applied"] = rebalancing_summary.applied
        metrics["rebalancing_strategy"] = rebalancing_summary.strategy
        metrics["train_class_counts_original"] = rebalancing_summary.original_class_counts
        metrics["train_class_counts_rebalanced"] = rebalancing_summary.rebalanced_class_counts
    return metrics


def _run_classification_cross_validation_with_optional_rebalancing(
    run_plan: CandidateRunPlan,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    """Fit/evaluate one classification candidate using explicit CV folds."""
    _validate_binary_classification_target(y, run_plan=run_plan, context="the selected target column")
    cv_folds = max(2, int(run_plan.cv_folds))
    _validate_classification_cv_feasibility(y, cv_folds=cv_folds)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=run_plan.random_state)
    positive_label = _resolved_positive_class_label(run_plan, y)
    if positive_label is None:
        raise ModelingServiceError("Binary classification requires two target classes.")

    fold_metrics: list[dict[str, float]] = []
    rebalancing_summaries: list[RebalancingSummary] = []
    fold_threshold_summaries: list[dict[str, Any]] = []

    for fold_index, (train_index, validation_index) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_index].copy()
        X_validation = X.iloc[validation_index].copy()
        y_train = y.iloc[train_index].copy()
        y_validation = y.iloc[validation_index].copy()
        _validate_training_partition_labels(y_train, partition_label=f"training fold {fold_index}")

        threshold_value: float | None = None
        threshold_details: dict[str, Any] | None = None
        if _classification_threshold_policy(run_plan) == "optimize":
            threshold_value, threshold_details = _tuned_classification_threshold_for_training_partition(
                run_plan,
                X_train,
                y_train,
                positive_label=positive_label,
                partition_label=f"training fold {fold_index}",
                random_state=run_plan.random_state + fold_index - 1,
            )

        X_train_rebalanced, y_train_rebalanced, rebalancing_summary = _rebalance_training_partition(
            X_train,
            y_train,
            strategy=_classification_rebalancing_strategy(run_plan),
            random_state=run_plan.random_state + fold_index - 1,
        )
        rebalancing_summaries.append(rebalancing_summary)

        pipeline = _build_training_pipeline(run_plan, X_train_rebalanced)
        pipeline.fit(X_train_rebalanced, y_train_rebalanced)
        y_score = _positive_class_score_from_pipeline(pipeline, X_validation, positive_label)
        if y_score is None and _classification_threshold_policy(run_plan) in {"manual", "optimize"}:
            raise ModelingServiceError(
                "Thresholding requires a classifier that exposes positive-class probabilities."
            )
        if y_score is not None:
            negative_label = _classification_negative_label(y_train_rebalanced, positive_label)
            if _classification_threshold_policy(run_plan) != "optimize":
                threshold_value = _classification_threshold_value(run_plan)
            y_pred = _threshold_predictions_from_scores(
                y_score,
                threshold=float(threshold_value),
                positive_label=positive_label,
                negative_label=negative_label,
            )
        else:
            y_pred = pipeline.predict(X_validation)
        fold_metrics.append(
            _classification_scoring_summary(
                y_validation,
                y_pred,
                positive_label=positive_label,
                y_score=y_score,
            )
        )
        if threshold_value is not None:
            fold_summary = {
                "fold": fold_index,
                "threshold_used": float(threshold_value),
                "source": _classification_threshold_source(run_plan),
            }
            if threshold_details is not None:
                fold_summary.update(threshold_details)
            fold_threshold_summaries.append(fold_summary)

    metrics: dict[str, Any] = {}
    for metric_name in {"accuracy", "precision", "recall", "f1", "roc_auc"}:
        values = [fold_metric[metric_name] for fold_metric in fold_metrics if metric_name in fold_metric]
        if values:
            metrics[metric_name] = float(mean(values))

    metrics["row_count"] = int(len(X))
    metrics["cv_folds"] = int(cv_folds)
    metrics["split_strategy"] = run_plan.split_strategy
    metrics["random_seed"] = int(run_plan.random_state)
    metrics["rebalancing_applied"] = any(summary.applied for summary in rebalancing_summaries)
    metrics["rebalancing_strategy"] = _classification_rebalancing_strategy(run_plan)
    metrics["classification_threshold_source"] = _classification_threshold_source(run_plan)
    metrics["classification_threshold_policy"] = run_plan.classification_threshold_policy
    metrics["classification_threshold_objective"] = run_plan.classification_threshold_objective
    if fold_threshold_summaries:
        metrics["cv_classification_threshold_summary"] = fold_threshold_summaries
        metrics["classification_threshold_used"] = float(
            mean(summary["threshold_used"] for summary in fold_threshold_summaries)
        )
    metrics["cv_rebalancing_summary"] = [
        {
            "fold": fold_index,
            "applied": summary.applied,
            "strategy": summary.strategy,
            "train_class_counts_original": summary.original_class_counts,
            "train_class_counts_rebalanced": summary.rebalanced_class_counts,
        }
        for fold_index, summary in enumerate(rebalancing_summaries, start=1)
    ]
    return metrics


def _run_cross_validation_evaluation(
    run_plan: CandidateRunPlan,
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
) -> dict[str, Any]:
    """Fit/evaluate one candidate using cross validation."""
    problem_type = _normalize_problem_type(get_modeling_problem_type())
    cv_folds = max(2, int(run_plan.cv_folds))

    if problem_type == "classification":
        if _classification_rebalancing_strategy(run_plan) != "none":
            return _run_classification_cross_validation_with_optional_rebalancing(run_plan, X, y)
        if _classification_threshold_policy(run_plan) in {"manual", "optimize"}:
            return _run_classification_cross_validation_with_optional_rebalancing(run_plan, X, y)
        _validate_binary_classification_target(y, run_plan=run_plan, context="the selected target column")
        _validate_classification_cv_feasibility(y, cv_folds=cv_folds)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=run_plan.random_state)
        positive_label = _resolved_positive_class_label(run_plan, y)
        if positive_label is None:
            raise ModelingServiceError("Binary classification requires two target classes.")
        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, pos_label=positive_label, zero_division=0),
            "recall": make_scorer(recall_score, pos_label=positive_label, zero_division=0),
            "f1": make_scorer(f1_score, pos_label=positive_label, zero_division=0),
            "roc_auc": "roc_auc",
        }
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=run_plan.random_state)
        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
            "adjusted_r2": _adjusted_r2_scorer,
        }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        error_score="raise",
    )

    metrics: dict[str, float] = {}
    for metric_name in scoring:
        values = cv_results.get(f"test_{metric_name}")
        if values is None:
            continue
        metric_value = float(pd.Series(values).mean())
        if pd.isna(metric_value):
            continue
        if metric_name in {"rmse", "mae"}:
            metric_value = abs(metric_value)
        metrics[metric_name] = metric_value

    metrics["row_count"] = int(len(X))
    metrics["cv_folds"] = int(cv_folds)
    metrics["split_strategy"] = run_plan.split_strategy
    metrics["random_seed"] = int(run_plan.random_state)
    return metrics


def _build_model_run_record(
    run_plan: CandidateRunPlan,
    metrics: dict[str, Any],
    status: str,
    message: str,
) -> ModelRunRecord:
    """Build a standardized run-record payload for one execution result."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    training_mode = "cross_validation" if run_plan.use_cross_validation else "train_test_split"
    rebalancing_strategy = _classification_rebalancing_strategy(run_plan)
    run_record: ModelRunRecord = {
        "run_id": f"run_{run_plan.candidate_id}_{timestamp.replace(':', '').replace('-', '')}",
        "candidate_id": run_plan.candidate_id,
        "model_id": run_plan.model_id,
        "model_label": run_plan.candidate_label,
        "problem_type": get_modeling_problem_type() or "unknown",
        "status": status,
        "training_mode": training_mode,
        "split_strategy": run_plan.split_strategy,
        "random_seed": int(run_plan.random_state),
        "positive_class_label": run_plan.positive_class_label,
        "classification_threshold_policy": run_plan.classification_threshold_policy,
        "classification_threshold_objective": run_plan.classification_threshold_objective,
        "started_at": timestamp,
        "completed_at": timestamp,
        "input_artifact_name": run_plan.source_dataset_name,
        "target_column": run_plan.dataset_plan.target_column or "",
        "feature_columns": run_plan.dataset_plan.resolved_feature_columns,
        "fitted_object": None,
        "preprocessing_summary": {
            "encoding_strategy": run_plan.dataset_plan.candidate_preprocessing.get("encoding_strategy"),
            "encoding_columns": list(run_plan.dataset_plan.encoding_columns),
            "scaling_strategy": run_plan.dataset_plan.candidate_preprocessing.get("scaling_strategy"),
            "class_rebalancing_strategy": rebalancing_strategy,
            "class_rebalancing_applied": bool(metrics.get("rebalancing_applied", False)),
            "train_class_counts_original": deepcopy(metrics.get("train_class_counts_original")),
            "train_class_counts_rebalanced": deepcopy(metrics.get("train_class_counts_rebalanced")),
            "cv_rebalancing_summary": deepcopy(metrics.get("cv_rebalancing_summary")),
        },
        "params_used": deepcopy(run_plan.custom_params),
        "tuning_result": None,
        "metrics": metrics,
        "plots": {},
        "artifacts": {},
        "notes": message,
        "created_at": timestamp,
        "error_message": None if status == "completed" else message,
    }
    threshold_source = _classification_threshold_source(run_plan)
    run_record["classification_threshold_source"] = str(
        metrics.get("classification_threshold_source", threshold_source)
    )
    if "classification_threshold_used" in metrics:
        run_record["classification_threshold_used"] = float(metrics["classification_threshold_used"])
    elif threshold_source in {"default", "manual"}:
        try:
            run_record["classification_threshold_used"] = _classification_threshold_value(run_plan)
        except ModelingServiceError:
            pass
    if threshold_source == "manual":
        run_record["classification_threshold_manual_value"] = float(run_plan.classification_threshold_manual_value)
    if "classification_threshold_optimization_details" in metrics:
        run_record["classification_threshold_optimization_details"] = deepcopy(
            metrics.get("classification_threshold_optimization_details")
        )
    if "cv_classification_threshold_summary" in metrics:
        run_record["cv_classification_threshold_summary"] = deepcopy(
            metrics.get("cv_classification_threshold_summary")
        )
    if training_mode == "train_test_split":
        run_record["test_size"] = float(run_plan.test_size)
        if "train_row_count" in metrics:
            run_record["train_row_count_original"] = int(metrics["train_row_count"])
        if "train_row_count_after_rebalancing" in metrics:
            run_record["train_row_count_after_rebalancing"] = int(metrics["train_row_count_after_rebalancing"])
        if "test_row_count" in metrics:
            run_record["test_row_count"] = int(metrics["test_row_count"])
    elif training_mode == "cross_validation":
        run_record["cv_folds"] = int(run_plan.cv_folds)
    return run_record


def _run_record_training_mode(run_record: object) -> str | None:
    """Return the normalized training mode encoded by a run record."""
    if not isinstance(run_record, dict):
        return None
    training_mode = str(run_record.get("training_mode", "")).strip().lower()
    if training_mode in {"cross_validation", "train_test_split"}:
        return training_mode
    split_strategy = str(run_record.get("split_strategy", "")).strip().lower()
    if split_strategy in {"cross_validation", "train_test_split"}:
        return split_strategy
    return None


def _shared_export_value(run_records: list[ModelRunRecord], key: str) -> object | None:
    """Return the shared value for one run-record field when all runs agree."""
    values = [run_record.get(key) for run_record in run_records if isinstance(run_record, dict) and key in run_record]
    if not values:
        return None
    first_value = values[0]
    for value in values[1:]:
        if value != first_value:
            return None
    return deepcopy(first_value)


def _candidate_export_config(
    candidate: CandidateModelConfig,
    run_record: ModelRunRecord | None,
    dataset_plan_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return export config normalized to the latest executed run when available."""
    export_config = deepcopy(candidate)
    if not isinstance(run_record, dict):
        return export_config

    export_config["latest_run_id"] = run_record.get("run_id")
    training_mode = _run_record_training_mode(run_record)
    if training_mode is not None:
        export_config["train_test_split_enabled"] = training_mode == "train_test_split"

    preprocessing_summary = (
        dict(run_record.get("preprocessing_summary", {}))
        if isinstance(run_record.get("preprocessing_summary"), dict)
        else {}
    )
    dataset_preprocessing = (
        dict(dataset_plan_summary.get("candidate_preprocessing", {}))
        if isinstance(dataset_plan_summary, dict)
        and isinstance(dataset_plan_summary.get("candidate_preprocessing"), dict)
        else {}
    )
    resolved_preprocessing = deepcopy(export_config.get("preprocessing", {}))
    if not isinstance(resolved_preprocessing, dict):
        resolved_preprocessing = {}
    resolved_preprocessing.update(dataset_preprocessing)
    for key in ("encoding_strategy", "encoding_columns", "scaling_strategy", "class_rebalancing_strategy"):
        if key in preprocessing_summary:
            resolved_preprocessing[key] = deepcopy(preprocessing_summary.get(key))
    export_config["preprocessing"] = resolved_preprocessing

    params_used = run_record.get("params_used")
    if isinstance(params_used, dict):
        export_config["custom_params"] = deepcopy(params_used)
        export_config["hyperparameters"] = deepcopy(params_used)

    return export_config


def _build_export_comparison_settings() -> dict[str, Any]:
    """Return canonical comparison settings for export payloads.

    When run records exist, evaluation settings come from the actual latest run
    records rather than mutable current UI state. This keeps exports accurate
    even when the user changes modeling settings after training.
    """
    current_config = deepcopy(get_model_comparison_config())
    export_settings: dict[str, Any] = {
        "evaluation_metric": current_config.get("evaluation_metric"),
        "default_parameter_mode": current_config.get("default_parameter_mode"),
    }

    run_records = [
        candidate.get("latest_run_record")
        for candidate in get_candidate_models()
        if isinstance(candidate.get("latest_run_record"), dict)
    ]
    if not run_records:
        export_settings.update(
            {
                "split_strategy": current_config.get("split_strategy"),
                "use_cross_validation": current_config.get("use_cross_validation"),
                "random_seed": current_config.get("random_seed", current_config.get("random_state")),
            }
        )
        if current_config.get("split_strategy") == "cross_validation":
            export_settings["cv_folds"] = current_config.get("cv_folds")
        elif current_config.get("split_strategy") == "train_test_split":
            export_settings["test_size"] = current_config.get("test_size")
        return export_settings

    training_modes = [_run_record_training_mode(run_record) for run_record in run_records]
    unique_training_modes = {mode for mode in training_modes if mode}
    random_seed = _shared_export_value(run_records, "random_seed")
    if random_seed is not None:
        export_settings["random_seed"] = random_seed

    if len(unique_training_modes) != 1:
        export_settings["split_strategy"] = "mixed"
        return export_settings

    training_mode = next(iter(unique_training_modes))
    export_settings["split_strategy"] = training_mode
    export_settings["use_cross_validation"] = training_mode == "cross_validation"
    if training_mode == "cross_validation":
        cv_folds = _shared_export_value(run_records, "cv_folds")
        if cv_folds is not None:
            export_settings["cv_folds"] = cv_folds
    elif training_mode == "train_test_split":
        test_size = _shared_export_value(run_records, "test_size")
        if test_size is not None:
            export_settings["test_size"] = test_size
    return export_settings



def _next_candidate_id(existing_candidates: list[CandidateModelConfig]) -> str:
    """Generate the next stable candidate id."""
    existing_ids = {str(candidate.get("candidate_id", "")) for candidate in existing_candidates}
    index = 1
    while True:
        candidate_id = f"candidate_{index:03d}"
        if candidate_id not in existing_ids:
            return candidate_id
        index += 1



def _default_candidate_label(model_name: str, existing_candidates: list[CandidateModelConfig]) -> str:
    """Generate a readable default candidate label."""
    base_label = model_name.strip() or "Model"
    existing_labels = {str(candidate.get("candidate_label", "")) for candidate in existing_candidates}
    if base_label not in existing_labels:
        return base_label

    index = 2
    while True:
        label = f"{base_label} {index}"
        if label not in existing_labels:
            return label
        index += 1


def _metric_direction(metric_name: object) -> str | None:
    """Return whether a metric should be maximized or minimized."""
    metric_key = str(metric_name or "").strip().lower()
    if metric_key in {"accuracy", "precision", "recall", "f1", "roc_auc", "r2"}:
        return "maximize"
    if metric_key in {"rmse", "mae"}:
        return "minimize"
    return None


def _is_completed_run_record(run_record: object) -> bool:
    """Return True when a stored run record represents a completed run."""
    return isinstance(run_record, dict) and str(run_record.get("status", "")).strip().lower() == "completed"


def _completed_candidate_run_records() -> list[tuple[CandidateModelConfig, str, ModelRunRecord]]:
    """Return completed candidates paired with their persisted run records."""
    completed_records: list[tuple[CandidateModelConfig, str, ModelRunRecord]] = []
    for candidate in get_candidate_models():
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        run_record = candidate.get("latest_run_record")
        if not candidate_id or not _is_completed_run_record(run_record):
            continue
        completed_records.append((candidate, candidate_id, run_record))
    return completed_records


def _run_record_has_metric(run_record: ModelRunRecord | None, metric_name: object) -> bool:
    """Return True when a run record contains a metric key."""
    if not isinstance(run_record, dict):
        return False
    metric_key = str(metric_name or "").strip()
    if not metric_key:
        return False
    return metric_key in dict(run_record.get("metrics", {}))


def _run_record_metric_value(run_record: ModelRunRecord | None, metric_name: object) -> float | None:
    """Return a numeric metric value from a run record when available."""
    if not isinstance(run_record, dict):
        return None
    metric_key = str(metric_name or "").strip()
    if not metric_key:
        return None
    raw_value = dict(run_record.get("metrics", {})).get(metric_key)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _resolve_shared_comparison_metric_name(
    completed_records: list[tuple[CandidateModelConfig, str, ModelRunRecord]] | None = None,
) -> str | None:
    """Return the single metric name that can be used across completed candidates."""
    completed_records = completed_records if completed_records is not None else _completed_candidate_run_records()
    if not completed_records:
        return None

    configured_metric = str(get_model_comparison_config().get("evaluation_metric") or "").strip()
    if configured_metric and all(
        _run_record_has_metric(run_record, configured_metric) for _, _, run_record in completed_records
    ):
        return configured_metric

    problem_type = _normalize_problem_type(get_modeling_problem_type())
    for metric_name in COMPARISON_METRIC_PRIORITY_BY_PROBLEM_TYPE.get(problem_type or "", []):
        if all(_run_record_has_metric(run_record, metric_name) for _, _, run_record in completed_records):
            return metric_name
    return None


def _primary_metric_name_for_candidate(candidate: CandidateModelConfig, run_record: ModelRunRecord | None) -> str | None:
    """Return the best metric name to use for candidate-level display fallback."""
    if not isinstance(run_record, dict):
        return None

    comparison_metric = get_model_comparison_config().get("evaluation_metric")
    if comparison_metric and comparison_metric in dict(run_record.get("metrics", {})):
        return str(comparison_metric)

    model_spec = get_model_spec(str(candidate.get("model_id", "")))
    if model_spec is not None:
        for metric_name in list(model_spec.get("default_metrics", [])):
            if metric_name in dict(run_record.get("metrics", {})):
                return str(metric_name)

    default_metric = _default_metric_for_problem_type(get_modeling_problem_type())
    if default_metric and default_metric in dict(run_record.get("metrics", {})):
        return str(default_metric)

    metric_names = list(dict(run_record.get("metrics", {})).keys())
    return str(metric_names[0]) if metric_names else None


def _score_candidate_run_record(candidate: CandidateModelConfig, run_record: ModelRunRecord | None) -> tuple[str | None, float | None]:
    """Return the primary metric name and numeric score for a run record."""
    if not isinstance(run_record, dict):
        return None, None
    metric_name = _primary_metric_name_for_candidate(candidate, run_record)
    if metric_name is None:
        return None, None
    raw_value = dict(run_record.get("metrics", {})).get(metric_name)
    if raw_value is None:
        return metric_name, None
    try:
        return metric_name, float(raw_value)
    except (TypeError, ValueError):
        return metric_name, None


def _persist_candidate_run_record(candidate_id: str, run_record: ModelRunRecord) -> CandidateModelConfig:
    """Persist the latest run record on the candidate config."""
    return update_candidate_model_config(
        candidate_id,
        latest_run_id=run_record.get("run_id"),
        latest_run_record=run_record,
    )


def list_available_model_specs(problem_type: str | None = None) -> list[dict[str, Any]]:
    """Return registered model specs, optionally filtered by problem type."""
    if problem_type is None:
        classification_specs = get_model_specs_for_problem_type("classification")
        regression_specs = get_model_specs_for_problem_type("regression")
        classification_ids = {str(spec.get("model_id")) for spec in classification_specs}
        specs = classification_specs + [
            spec
            for spec in regression_specs
            if str(spec.get("model_id")) not in classification_ids
        ]
    else:
        specs = get_model_specs_for_problem_type(problem_type)
    return [deepcopy(spec) for spec in specs]




def _require_supported_model_id(model_id: str) -> None:
    """Raise a clear error when a model id is not registered."""
    if not is_supported_model(model_id):
        raise ModelingServiceError(f"Model '{model_id}' is not registered.")



def get_model_comparison_settings() -> ModelComparisonConfig:
    """Return the top-level model comparison config.

    This is the shared comparison-level configuration that applies across
    candidate models.
    """
    return get_model_comparison_config()



def update_model_comparison_settings(**updates: object) -> ModelComparisonConfig:
    """Update top-level model comparison settings."""
    return update_model_comparison_config(**updates)





def create_candidate_model(
    *,
    model_id: str,
    candidate_label: str | None = None,
) -> CandidateModelConfig:
    """Create and register a new candidate model config.

    This mutates app state by appending a new candidate model and, when no
    candidate is currently active, setting the new candidate as active.
    """
    _require_supported_model_id(model_id)
    existing_candidates = get_candidate_models()
    candidate_id = _next_candidate_id(existing_candidates)
    model_spec = require_model_spec(model_id)
    resolved_label = candidate_label or _default_candidate_label(
        str(model_spec.get("label") or model_id),
        existing_candidates,
    )

    candidate: CandidateModelConfig = {
        "candidate_id": candidate_id,
        "candidate_label": resolved_label,
        "model_id": model_id,
        "enabled": False,
        "preprocessing": {},
        "hyperparameters": {},
        "custom_params": {},
        "tuning": {},
        "classification_threshold": None,
        "latest_run_id": None,
        "latest_run_record": None,
    }
    append_candidate_model(candidate)

    if get_active_candidate_id() is None:
        set_active_candidate_id(candidate_id)

    return candidate



def duplicate_candidate_model(candidate_id: str) -> CandidateModelConfig:
    """Duplicate an existing candidate model config and register the copy."""
    source_candidate = get_candidate_model(candidate_id)
    if source_candidate is None:
        raise ModelingServiceError(f"Candidate '{candidate_id}' was not found.")

    existing_candidates = get_candidate_models()
    new_candidate_id = _next_candidate_id(existing_candidates)
    copied_candidate = deepcopy(source_candidate)
    copied_candidate["candidate_id"] = new_candidate_id
    source_model_id = str(source_candidate.get("model_id", "model"))
    source_model_spec = get_model_spec(source_model_id)
    copied_candidate["candidate_label"] = _default_candidate_label(
        str(source_model_spec.get("label")) if source_model_spec is not None else source_model_id,
        existing_candidates,
    )
    copied_candidate["latest_run_id"] = None
    copied_candidate["latest_run_record"] = None

    append_candidate_model(copied_candidate)
    return copied_candidate



def remove_candidate_model_config(candidate_id: str) -> list[CandidateModelConfig]:
    """Remove a candidate model config from state."""
    return remove_candidate_model(candidate_id)



def set_active_candidate_model(candidate_id: str | None) -> CandidateModelConfig | None:
    """Set the active candidate model by id and return it."""
    if candidate_id is None:
        set_active_candidate_id(None)
        return None

    candidate = get_candidate_model(candidate_id)
    if candidate is None:
        raise ModelingServiceError(f"Candidate '{candidate_id}' was not found.")

    set_active_candidate_id(candidate_id)
    return candidate



def get_active_candidate_model() -> CandidateModelConfig | None:
    """Return the active candidate model config, if any."""
    return get_active_candidate()


def get_latest_candidate_run_record(candidate_id: str) -> ModelRunRecord | None:
    """Return the latest persisted run record for one candidate, if available."""
    candidate = get_candidate_model(candidate_id)
    if candidate is None:
        raise ModelingServiceError(f"Candidate '{candidate_id}' was not found.")
    run_record = candidate.get("latest_run_record")
    return deepcopy(run_record) if isinstance(run_record, dict) else None


def get_all_latest_candidate_run_records() -> dict[str, ModelRunRecord]:
    """Return the latest persisted run records keyed by candidate id."""
    run_records: dict[str, ModelRunRecord] = {}
    for candidate in get_candidate_models():
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        run_record = candidate.get("latest_run_record")
        if candidate_id and isinstance(run_record, dict):
            run_records[candidate_id] = deepcopy(run_record)
    return run_records



def update_candidate_model_config(
    candidate_id: str,
    **updates: object,
) -> CandidateModelConfig:
    """Update one candidate model config and return the updated object."""
    candidate = update_candidate_model(candidate_id, **updates)
    if candidate is None:
        raise ModelingServiceError(f"Candidate '{candidate_id}' was not found.")
    return candidate



def build_candidate_dataset_plan(candidate_id: str) -> CandidateDatasetPlan:
    """Resolve the dataset plan for one candidate model.

    Phase 1 does not yet construct a transformed dataframe. Instead, it returns
    a fully resolved plan describing which dataset and preprocessing layers the
    candidate would use.
    """
    candidate = get_candidate_model(candidate_id)
    if candidate is None:
        raise ModelingServiceError(f"Candidate '{candidate_id}' was not found.")
    _require_supported_model_id(str(candidate.get("model_id")))

    source_dataset_name = _working_dataset_name()
    summary = dataset_summary(source_dataset_name)
    column_names = list(summary.get("column_names", []))
    if not column_names:
        raise ModelingServiceError("Working Data is not currently available for modeling.")

    normalized_preprocessing = _normalize_candidate_preprocessing_config(
        deepcopy(candidate.get("preprocessing", {})),
        column_names,
    )
    use_shared_preprocessing = bool(normalized_preprocessing.get("use_shared_preprocessing", True))
    available_feature_columns = _eligible_workspace_predictor_columns(column_names)
    feature_subset_mode = str(normalized_preprocessing.get("feature_subset_mode", "Use all eligible predictors"))
    selected_feature_columns = list(normalized_preprocessing.get("selected_feature_columns", []))
    excluded_feature_columns = list(normalized_preprocessing.get("excluded_feature_columns", []))
    resolved_feature_columns = _resolve_candidate_feature_columns(
        available_feature_columns=available_feature_columns,
        feature_subset_mode=feature_subset_mode,
        selected_feature_columns=selected_feature_columns,
        excluded_feature_columns=excluded_feature_columns,
    )
    encoding_columns = [
        column_name
        for column_name in list(normalized_preprocessing.get("encoding_columns", []))
        if column_name in resolved_feature_columns
    ]
    normalized_preprocessing["encoding_columns"] = encoding_columns

    target_column = get_modeling_target_column()
    identifier_columns = get_modeling_identifier_columns()
    ignored_columns = get_modeling_ignored_columns()

    _validate_candidate_dataset_plan_inputs(
        candidate_label=str(candidate.get("candidate_label")),
        target_column=target_column,
        available_feature_columns=available_feature_columns,
        resolved_feature_columns=resolved_feature_columns,
        encoding_columns=encoding_columns,
        source_dataset_name=source_dataset_name,
        encoding_strategy=str(normalized_preprocessing.get("encoding_strategy", "none")),
    )

    return CandidateDatasetPlan(
        candidate_id=str(candidate.get("candidate_id")),
        candidate_label=str(candidate.get("candidate_label")),
        model_id=str(candidate.get("model_id")),
        source_dataset_name=source_dataset_name,
        use_shared_preprocessing=use_shared_preprocessing,
        shared_preprocessing_applies=use_shared_preprocessing,
        candidate_preprocessing=normalized_preprocessing,
        available_feature_columns=available_feature_columns,
        resolved_feature_columns=resolved_feature_columns,
        selected_feature_columns=selected_feature_columns,
        excluded_feature_columns=excluded_feature_columns,
        encoding_columns=encoding_columns,
        target_column=target_column,
        identifier_columns=identifier_columns,
        ignored_columns=ignored_columns,
    )



def build_candidate_run_plan(candidate_id: str) -> CandidateRunPlan:
    """Build the resolved training/evaluation plan for one candidate."""
    candidate = get_candidate_model(candidate_id)
    if candidate is None:
        raise ModelingServiceError(f"Candidate '{candidate_id}' was not found.")
    model_spec = require_model_spec(str(candidate.get("model_id")))

    comparison_config = get_model_comparison_config()
    problem_type = get_modeling_problem_type()
    dataset_plan = build_candidate_dataset_plan(candidate_id)
    evaluation_metric = comparison_config.get("evaluation_metric") or _default_metric_for_problem_type(problem_type)
    if evaluation_metric is None:
        model_metrics = list(model_spec.get("default_metrics", []))
        evaluation_metric = model_metrics[0] if model_metrics else None

    split_strategy = str(comparison_config.get("split_strategy", "cross_validation"))
    use_cross_validation = split_strategy == "cross_validation"
    train_test_split_enabled = split_strategy == "train_test_split"
    resolved_test_size = float(comparison_config.get("test_size", 0.2))
    resolved_random_seed = int(comparison_config.get("random_seed", comparison_config.get("random_state", 42)))

    return CandidateRunPlan(
        candidate_id=str(candidate.get("candidate_id")),
        candidate_label=str(candidate.get("candidate_label")),
        model_id=str(candidate.get("model_id")),
        source_dataset_name=dataset_plan.source_dataset_name,
        evaluation_metric=evaluation_metric,
        split_strategy=split_strategy,
        test_size=resolved_test_size,
        cv_folds=int(comparison_config.get("cv_folds", 5)),
        random_state=resolved_random_seed,
        use_cross_validation=use_cross_validation,
        train_test_split_enabled=train_test_split_enabled,
        positive_class_label=get_app_state().get("positive_class_label"),
        classification_threshold_policy=str(
            comparison_config.get("classification_threshold_policy", "Use model default")
        ),
        classification_threshold_manual_value=float(
            comparison_config.get("classification_threshold_manual_value", 0.5)
        ),
        classification_threshold_objective=str(
            comparison_config.get("classification_threshold_objective", "F1")
        ),
        classification_threshold=candidate.get("classification_threshold"),
        custom_params=deepcopy(candidate.get("hyperparameters", candidate.get("custom_params", {}))),
        tuning=deepcopy(candidate.get("tuning", {})),
        dataset_plan=dataset_plan,
    )



def train_candidate_model(candidate_id: str) -> CandidateExecutionResult:
    """Train and evaluate one candidate model."""
    run_plan = build_candidate_run_plan(candidate_id)
    source_df = _load_candidate_training_dataframe(run_plan)
    X, y = _build_candidate_xy(source_df, run_plan)
    pipeline = _build_training_pipeline(run_plan, X)

    if run_plan.use_cross_validation:
        metrics = _run_cross_validation_evaluation(run_plan, X, y, pipeline)
        message = (
            f"Completed cross-validation for {run_plan.candidate_label} using "
            f"{len(run_plan.dataset_plan.resolved_feature_columns)} predictor columns."
        )
    else:
        metrics = _run_train_test_evaluation(run_plan, X, y, pipeline)
        message = (
            f"Completed train/test evaluation for {run_plan.candidate_label} using "
            f"{len(run_plan.dataset_plan.resolved_feature_columns)} predictor columns."
        )

    run_record = _build_model_run_record(
        run_plan=run_plan,
        metrics=metrics,
        status="completed",
        message=message,
    )
    return CandidateExecutionResult(
        candidate_id=run_plan.candidate_id,
        candidate_label=run_plan.candidate_label,
        model_id=run_plan.model_id,
        status="completed",
        message=message,
        metrics=metrics,
        run_record=run_record,
        run_plan=run_plan,
    )



def train_enabled_candidate_models() -> list[CandidateExecutionResult]:
    """Train all enabled candidate models and return structured results."""
    results: list[CandidateExecutionResult] = []
    for candidate in get_candidate_models():
        if not bool(candidate.get("enabled", False)):
            continue
        candidate_id = str(candidate.get("candidate_id"))
        try:
            result = train_candidate_model(candidate_id)
            if result.run_record is not None:
                _persist_candidate_run_record(candidate_id, result.run_record)
            results.append(result)
        except ModelingServiceError:
            raise
        except Exception as exc:
            run_plan = build_candidate_run_plan(candidate_id)
            message = f"Training failed for {run_plan.candidate_label}: {exc}"
            failed_run_record = _build_model_run_record(
                run_plan=run_plan,
                metrics={},
                status="failed",
                message=message,
            )
            _persist_candidate_run_record(candidate_id, failed_run_record)
            results.append(
                CandidateExecutionResult(
                    candidate_id=run_plan.candidate_id,
                    candidate_label=run_plan.candidate_label,
                    model_id=run_plan.model_id,
                    status="failed",
                    message=message,
                    metrics={},
                    run_record=failed_run_record,
                    run_plan=run_plan,
                )
            )

    best_candidate_id = select_best_candidate_from_latest_results()
    set_best_candidate_id(best_candidate_id)
    return results












def get_modeling_problem_type() -> str | None:
    """Return the current workspace-level problem type."""
    return get_app_state().get("problem_type")



def get_modeling_target_column() -> str | None:
    """Return the current workspace-level target column."""
    return get_app_state().get("target_column")



def get_modeling_identifier_columns() -> list[str]:
    """Return the current workspace-level identifier columns."""
    return list(get_app_state().get("id_columns", []))



def get_modeling_ignored_columns() -> list[str]:
    """Return the current workspace-level ignored columns."""
    return list(get_app_state().get("ignored_columns", []))


def select_best_candidate_from_latest_results() -> str | None:
    """Return the best candidate id based on persisted completed results."""
    completed_records = _completed_candidate_run_records()
    comparison_metric_name = _resolve_shared_comparison_metric_name(completed_records)
    if comparison_metric_name is None:
        return None

    direction = _metric_direction(comparison_metric_name)
    if direction is None:
        return None

    best_candidate_id: str | None = None
    best_metric_value: float | None = None
    for _, candidate_id, run_record in completed_records:
        metric_value = _run_record_metric_value(run_record, comparison_metric_name)
        if metric_value is None:
            return None

        if best_candidate_id is None:
            best_candidate_id = candidate_id
            best_metric_value = metric_value
            continue

        if direction == "maximize" and metric_value > float(best_metric_value):
            best_candidate_id = candidate_id
            best_metric_value = metric_value
        elif direction == "minimize" and metric_value < float(best_metric_value):
            best_candidate_id = candidate_id
            best_metric_value = metric_value

    return best_candidate_id


def build_results_comparison_summary() -> dict[str, Any]:
    """Return a UI-friendly summary of persisted candidate results."""
    candidates_summary: list[dict[str, Any]] = []
    completed_records = _completed_candidate_run_records()
    comparison_metric_name = _resolve_shared_comparison_metric_name(completed_records)
    best_candidate_id = select_best_candidate_from_latest_results()

    for candidate in get_candidate_models():
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        run_record = candidate.get("latest_run_record")
        metric_name, metric_value = _score_candidate_run_record(candidate, run_record)
        comparison_metric_value = _run_record_metric_value(run_record, comparison_metric_name)
        is_comparison_eligible = bool(
            _is_completed_run_record(run_record)
            and comparison_metric_name
            and comparison_metric_value is not None
        )
        feature_columns = []
        if isinstance(run_record, dict):
            feature_columns = list(run_record.get("feature_columns", []))

        if isinstance(run_record, dict):
            training_mode = str(run_record.get("training_mode", "")).strip().lower()
        else:
            training_mode = ""
        if training_mode == "train_test_split":
            evaluation_mode = "Train / Test Split"
        elif training_mode == "cross_validation":
            evaluation_mode = "Cross Validation"
        else:
            split_strategy = str(get_model_comparison_config().get("split_strategy", "cross_validation"))
            evaluation_mode = "Train / Test Split" if split_strategy == "train_test_split" else "Cross Validation"

        candidates_summary.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": str(candidate.get("candidate_label", candidate_id)),
                "model_id": str(candidate.get("model_id", "")),
                "status": str(run_record.get("status", "not_run")) if isinstance(run_record, dict) else "not_run",
                "primary_metric_name": metric_name,
                "primary_metric_value": metric_value,
                "comparison_metric_name": comparison_metric_name,
                "comparison_metric_value": comparison_metric_value,
                "is_comparison_eligible": is_comparison_eligible,
                "metrics": deepcopy(dict(run_record.get("metrics", {}))) if isinstance(run_record, dict) else {},
                "predictor_count": int(len(feature_columns)),
                "resolved_feature_columns": feature_columns,
                "evaluation_mode": evaluation_mode,
                "latest_run_id": run_record.get("run_id") if isinstance(run_record, dict) else None,
                "notes": str(run_record.get("notes", "")) if isinstance(run_record, dict) else "",
                "is_best_candidate": candidate_id == best_candidate_id,
            }
        )

    return {
        "active_candidate_id": get_active_candidate_id(),
        "best_candidate_id": best_candidate_id,
        "comparison_metric_name": comparison_metric_name,
        "candidate_count": len(candidates_summary),
        "candidates": candidates_summary,
    }


def build_results_bundle() -> dict[str, Any]:
    """Return the canonical structured results bundle for UI and export use."""
    app_state = get_app_state()
    feature_specs = []
    try:
        from app.workspace_apps.ml_workbench.services.feature_service import get_feature_specs

        feature_specs = deepcopy(get_feature_specs())
    except Exception:
        feature_specs = []

    candidates_bundle: list[dict[str, Any]] = []
    completed_records = _completed_candidate_run_records()
    comparison_metric_name = _resolve_shared_comparison_metric_name(completed_records)
    best_candidate_id = select_best_candidate_from_latest_results()

    for candidate in get_candidate_models():
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        run_record = candidate.get("latest_run_record")
        comparison_metric_value = _run_record_metric_value(run_record, comparison_metric_name)
        dataset_plan_summary: dict[str, Any] | None = None
        try:
            dataset_plan = build_candidate_dataset_plan(candidate_id)
            dataset_plan_summary = {
                "source_dataset_name": dataset_plan.source_dataset_name,
                "available_feature_columns": list(dataset_plan.available_feature_columns),
                "resolved_feature_columns": list(dataset_plan.resolved_feature_columns),
                "selected_feature_columns": list(dataset_plan.selected_feature_columns),
                "excluded_feature_columns": list(dataset_plan.excluded_feature_columns),
                "encoding_columns": list(dataset_plan.encoding_columns),
                "target_column": dataset_plan.target_column,
                "identifier_columns": list(dataset_plan.identifier_columns),
                "ignored_columns": list(dataset_plan.ignored_columns),
                "candidate_preprocessing": deepcopy(dataset_plan.candidate_preprocessing),
            }
        except Exception:
            dataset_plan_summary = None

        export_config = _candidate_export_config(candidate, run_record, dataset_plan_summary)

        candidates_bundle.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": str(candidate.get("candidate_label", candidate_id)),
                "model_id": str(candidate.get("model_id", "")),
                "enabled": bool(candidate.get("enabled", False)),
                "config": export_config,
                "current_config": deepcopy(candidate),
                "dataset_plan": dataset_plan_summary,
                "run_record": deepcopy(run_record) if isinstance(run_record, dict) else None,
                "comparison_metric_name": comparison_metric_name,
                "comparison_metric_value": comparison_metric_value,
                "is_comparison_eligible": bool(
                    _is_completed_run_record(run_record)
                    and comparison_metric_name
                    and comparison_metric_value is not None
                ),
                "is_best_candidate": candidate_id == best_candidate_id,
            }
        )

    return {
        "workspace": {
            "problem_type": app_state.get("problem_type"),
            "target_column": app_state.get("target_column"),
            "identifier_columns": list(app_state.get("id_columns", [])),
            "ignored_columns": list(app_state.get("ignored_columns", [])),
            "source_dataset_name": WORKING_DATASET_NAME,
        },
        "shared_preprocessing": deepcopy(app_state.get("preprocessing_config", {})),
        "feature_engineering": {
            "feature_specs": feature_specs,
        },
        "comparison_settings": _build_export_comparison_settings(),
        "comparison_metric_name": comparison_metric_name,
        "best_candidate_id": best_candidate_id,
        "active_candidate_id": get_active_candidate_id(),
        "candidates": candidates_bundle,
    }


def build_export_payload() -> dict[str, Any]:
    """Return the canonical structured export payload for downstream consumers."""
    return {
        "exported_at": datetime.utcnow().isoformat(timespec="seconds"),
        "export_version": RESULTS_BUNDLE_EXPORT_VERSION,
        "results_bundle": build_results_bundle(),
    }
