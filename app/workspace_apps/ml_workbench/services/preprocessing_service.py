

"""Shared preprocessing execution service for the ML Workbench app.

Phase 1A establishes the backend contract for rebuilding Working Data from Raw
Data plus the current shared preprocessing rules. This module is intentionally
UI-agnostic so both the standalone app and future AgentShell tools can call the
same functions.

At this stage, the service focuses on planning and structured result objects.
The actual dataframe transformation execution will be added in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.workspace_apps.ml_workbench.schemas import PreprocessingConfig
from app.workspace_apps.ml_workbench.services.feature_service import (
    apply_feature_specs_to_dataframe,
)
from app.workspace_apps.ml_workbench.services.dataset_service import (
    dataset_summary,
    get_active_dataset_name,
    get_dataset_copy,
    set_working_dataset,
)
from app.workspace_apps.ml_workbench.state import get_preprocessing_config

RAW_DATASET_NAME = "raw_dataset"
WORKING_DATASET_NAME = "working_dataset"


@dataclass(frozen=True)
class PreprocessingRuleStatus:
    """Compact status record for one preprocessing rule family."""

    rule_name: str
    enabled: bool
    configured_columns: list[str] = field(default_factory=list)
    valid_columns: list[str] = field(default_factory=list)
    invalid_columns: list[str] = field(default_factory=list)
    strategy: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class SharedPreprocessingPlan:
    """Resolved plan for rebuilding Working Data from shared preprocessing rules."""

    source_dataset_name: str
    output_dataset_name: str
    available_columns: list[str]
    rule_statuses: list[PreprocessingRuleStatus]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkingDataRebuildPreview:
    """Phase-1A preview result for a Working Data rebuild request.

    This does not yet execute dataframe transformations. It returns the
    resolved plan that later execution phases will use.
    """

    status: str
    message: str
    plan: SharedPreprocessingPlan


@dataclass(frozen=True)
class AppliedPreprocessingStep:
    """Execution summary for one applied preprocessing step."""

    rule_name: str
    strategy: str | None
    applied_columns: list[str] = field(default_factory=list)
    skipped_columns: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass(frozen=True)
class WorkingDataExecutionResult:
    """Result of applying shared preprocessing rules to a dataframe."""

    status: str
    message: str
    dataframe: pd.DataFrame
    plan: SharedPreprocessingPlan
    applied_steps: list[AppliedPreprocessingStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)



@dataclass(frozen=True)
class AppliedFeatureEngineeringStep:
    """Execution summary for shared engineered features during rebuild."""

    applied_feature_count: int
    created_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: str | None = None


class PreprocessingServiceError(RuntimeError):
    """Raised when a preprocessing workflow request cannot be completed."""



def _valid_and_invalid_columns(
    configured_columns: list[str],
    available_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Split configured columns into valid and invalid groups."""
    available_column_set = set(available_columns)
    valid_columns = [column for column in configured_columns if column in available_column_set]
    invalid_columns = [column for column in configured_columns if column not in available_column_set]
    return valid_columns, invalid_columns



def _build_drop_rule_status(
    preprocessing_config: PreprocessingConfig,
    available_columns: list[str],
) -> PreprocessingRuleStatus:
    """Build the status record for the shared drop-columns rule."""
    configured_columns = list(preprocessing_config.get("drop_columns", []))
    valid_columns, invalid_columns = _valid_and_invalid_columns(configured_columns, available_columns)
    return PreprocessingRuleStatus(
        rule_name="drop_columns",
        enabled=bool(valid_columns),
        configured_columns=configured_columns,
        valid_columns=valid_columns,
        invalid_columns=invalid_columns,
        strategy="drop",
        notes="Shared rule that removes columns from Working Data during rebuild.",
    )



def _build_numeric_imputation_rule_status(
    preprocessing_config: PreprocessingConfig,
    available_columns: list[str],
) -> PreprocessingRuleStatus:
    """Build the status record for numeric imputation."""
    config = preprocessing_config.get("numeric_imputation", {})
    configured_columns = list(config.get("columns", []))
    valid_columns, invalid_columns = _valid_and_invalid_columns(configured_columns, available_columns)
    strategy = config.get("strategy")
    return PreprocessingRuleStatus(
        rule_name="numeric_imputation",
        enabled=bool(strategy and valid_columns),
        configured_columns=configured_columns,
        valid_columns=valid_columns,
        invalid_columns=invalid_columns,
        strategy=strategy,
        notes="Shared rule for filling missing numeric values in Working Data.",
    )



def _build_categorical_imputation_rule_status(
    preprocessing_config: PreprocessingConfig,
    available_columns: list[str],
) -> PreprocessingRuleStatus:
    """Build the status record for category/text imputation."""
    config = preprocessing_config.get("categorical_imputation", {})
    configured_columns = list(config.get("columns", []))
    valid_columns, invalid_columns = _valid_and_invalid_columns(configured_columns, available_columns)
    strategy = config.get("strategy")
    return PreprocessingRuleStatus(
        rule_name="categorical_imputation",
        enabled=bool(strategy and valid_columns),
        configured_columns=configured_columns,
        valid_columns=valid_columns,
        invalid_columns=invalid_columns,
        strategy=strategy,
        notes="Shared rule for filling missing category/text values in Working Data.",
    )



def _build_datetime_rule_status(
    preprocessing_config: PreprocessingConfig,
    available_columns: list[str],
) -> PreprocessingRuleStatus:
    """Build the status record for datetime expansion."""
    config = preprocessing_config.get("datetime_handling", {})
    configured_columns = list(config.get("expanded_columns", []))
    valid_columns, invalid_columns = _valid_and_invalid_columns(configured_columns, available_columns)
    return PreprocessingRuleStatus(
        rule_name="datetime_handling",
        enabled=bool(valid_columns),
        configured_columns=configured_columns,
        valid_columns=valid_columns,
        invalid_columns=invalid_columns,
        strategy="expand_datetime_parts",
        notes="Shared rule for expanding selected datetime columns during Working Data rebuild.",
    )


def _safe_mode(series: pd.Series) -> Any:
    """Return a stable mode value or None when no mode is available."""
    non_null_series = series.dropna()
    if non_null_series.empty:
        return None
    mode_values = non_null_series.mode(dropna=True)
    if mode_values.empty:
        return None
    return mode_values.iloc[0]



def _coerce_constant_numeric_fill(fill_value: Any) -> float | int | None:
    """Coerce a constant numeric fill value into a numeric scalar when possible."""
    if fill_value is None:
        return None
    if isinstance(fill_value, (int, float)):
        return fill_value
    text = str(fill_value).strip()
    if text == "":
        return None
    try:
        numeric_value = float(text)
    except ValueError:
        return None
    return int(numeric_value) if numeric_value.is_integer() else numeric_value



def _apply_drop_columns_step(
    df: pd.DataFrame,
    rule_status: PreprocessingRuleStatus,
) -> tuple[pd.DataFrame, AppliedPreprocessingStep]:
    """Apply the shared drop-columns rule to a dataframe."""
    valid_columns = list(rule_status.valid_columns)
    skipped_columns = list(rule_status.invalid_columns)
    if not valid_columns:
        return df, AppliedPreprocessingStep(
            rule_name=rule_status.rule_name,
            strategy=rule_status.strategy,
            applied_columns=[],
            skipped_columns=skipped_columns,
            notes="No valid columns were available to drop.",
        )

    updated_df = df.drop(columns=valid_columns, errors="ignore")
    return updated_df, AppliedPreprocessingStep(
        rule_name=rule_status.rule_name,
        strategy=rule_status.strategy,
        applied_columns=valid_columns,
        skipped_columns=skipped_columns,
        notes="Dropped configured columns from the Working Data rebuild.",
    )



def _apply_numeric_imputation_step(
    df: pd.DataFrame,
    rule_status: PreprocessingRuleStatus,
    preprocessing_config: PreprocessingConfig,
) -> tuple[pd.DataFrame, AppliedPreprocessingStep]:
    """Apply the shared numeric-imputation rule to a dataframe."""
    config = preprocessing_config.get("numeric_imputation", {})
    strategy = rule_status.strategy
    valid_columns = [column for column in rule_status.valid_columns if column in df.columns]
    skipped_columns = list(rule_status.invalid_columns)
    updated_df = df.copy()
    applied_columns: list[str] = []

    if not strategy or not valid_columns:
        return df, AppliedPreprocessingStep(
            rule_name=rule_status.rule_name,
            strategy=strategy,
            applied_columns=[],
            skipped_columns=skipped_columns,
            notes="Numeric imputation was configured but no valid columns were available.",
        )

    constant_fill = _coerce_constant_numeric_fill(config.get("fill_value"))

    for column in valid_columns:
        series = updated_df[column]
        fill_value: Any = None
        if strategy == "mean":
            non_null_series = pd.to_numeric(series, errors="coerce")
            fill_value = None if non_null_series.dropna().empty else float(non_null_series.mean())
        elif strategy == "median":
            non_null_series = pd.to_numeric(series, errors="coerce")
            fill_value = None if non_null_series.dropna().empty else float(non_null_series.median())
        elif strategy == "constant":
            fill_value = constant_fill

        if fill_value is None:
            skipped_columns.append(column)
            continue

        updated_df[column] = series.fillna(fill_value)
        applied_columns.append(column)

    return updated_df, AppliedPreprocessingStep(
        rule_name=rule_status.rule_name,
        strategy=strategy,
        applied_columns=applied_columns,
        skipped_columns=skipped_columns,
        notes="Filled missing numeric values using the configured shared strategy.",
    )



def _apply_categorical_imputation_step(
    df: pd.DataFrame,
    rule_status: PreprocessingRuleStatus,
    preprocessing_config: PreprocessingConfig,
) -> tuple[pd.DataFrame, AppliedPreprocessingStep]:
    """Apply the shared category/text-imputation rule to a dataframe."""
    config = preprocessing_config.get("categorical_imputation", {})
    strategy = rule_status.strategy
    valid_columns = [column for column in rule_status.valid_columns if column in df.columns]
    skipped_columns = list(rule_status.invalid_columns)
    updated_df = df.copy()
    applied_columns: list[str] = []

    if not strategy or not valid_columns:
        return df, AppliedPreprocessingStep(
            rule_name=rule_status.rule_name,
            strategy=strategy,
            applied_columns=[],
            skipped_columns=skipped_columns,
            notes="Category/text imputation was configured but no valid columns were available.",
        )

    constant_fill = config.get("fill_value")
    if constant_fill is not None:
        constant_fill = str(constant_fill)

    for column in valid_columns:
        series = updated_df[column]
        fill_value: Any = None
        if strategy == "mode":
            fill_value = _safe_mode(series)
        elif strategy == "constant":
            fill_value = constant_fill

        if fill_value is None or (isinstance(fill_value, str) and fill_value == ""):
            skipped_columns.append(column)
            continue

        updated_df[column] = series.fillna(fill_value)
        applied_columns.append(column)

    return updated_df, AppliedPreprocessingStep(
        rule_name=rule_status.rule_name,
        strategy=strategy,
        applied_columns=applied_columns,
        skipped_columns=skipped_columns,
        notes="Filled missing category/text values using the configured shared strategy.",
    )



def _apply_datetime_expansion_step(
    df: pd.DataFrame,
    rule_status: PreprocessingRuleStatus,
) -> tuple[pd.DataFrame, AppliedPreprocessingStep]:
    """Apply shared datetime expansion to selected columns.

    Phase 1B keeps this lightweight by adding year/month/day columns where a
    datetime conversion succeeds while preserving the original source column.
    """
    valid_columns = [column for column in rule_status.valid_columns if column in df.columns]
    skipped_columns = list(rule_status.invalid_columns)
    updated_df = df.copy()
    applied_columns: list[str] = []

    if not valid_columns:
        return df, AppliedPreprocessingStep(
            rule_name=rule_status.rule_name,
            strategy=rule_status.strategy,
            applied_columns=[],
            skipped_columns=skipped_columns,
            notes="No valid datetime columns were available for expansion.",
        )

    for column in valid_columns:
        parsed = pd.to_datetime(updated_df[column], errors="coerce")
        if parsed.notna().sum() == 0:
            skipped_columns.append(column)
            continue
        updated_df[f"{column}__year"] = parsed.dt.year
        updated_df[f"{column}__month"] = parsed.dt.month
        updated_df[f"{column}__day"] = parsed.dt.day
        applied_columns.append(column)

    return updated_df, AppliedPreprocessingStep(
        rule_name=rule_status.rule_name,
        strategy=rule_status.strategy,
        applied_columns=applied_columns,
        skipped_columns=skipped_columns,
        notes="Expanded configured datetime columns into year/month/day components.",
    )


def _apply_shared_feature_engineering_step(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, AppliedFeatureEngineeringStep]:
    """Apply shared engineered features before final shared column removal.

    Shared feature rules are executed after shared imputations/datetime handling
    so engineered features can use cleaned source columns, and before final
    drop-columns removal so users can still engineer from base columns they plan
    to remove from Working Data afterward.
    """
    result = apply_feature_specs_to_dataframe(df)
    created_columns = [
        step.output_column
        for step in result.steps
        if step.status == "applied" and step.output_column and step.output_column in result.dataframe.columns
    ]
    return result.dataframe, AppliedFeatureEngineeringStep(
        applied_feature_count=sum(1 for step in result.steps if step.status == "applied"),
        created_columns=created_columns,
        warnings=list(result.warnings),
        notes="Applied shared engineered features before final shared drop-column removal.",
    )



def get_shared_preprocessing_config() -> PreprocessingConfig:
    """Return the current shared preprocessing config from app state."""
    return get_preprocessing_config()



def resolve_preprocessing_source_dataset_name() -> str:
    """Return the dataset artifact that should source shared preprocessing.

    Shared preprocessing should always rebuild Working Data from Raw Data so rule
    deletion or modification can cleanly revert the derived artifact.
    """
    active_dataset_name = get_active_dataset_name(default_to_working=False)
    if active_dataset_name is None:
        return RAW_DATASET_NAME
    return RAW_DATASET_NAME



def build_shared_preprocessing_plan() -> SharedPreprocessingPlan:
    """Build the resolved shared preprocessing plan for Working Data.

    Phase 1A does not yet execute transformations. It resolves the current
    config against the available source columns and produces structured rule
    status records plus warnings.
    """
    source_dataset_name = resolve_preprocessing_source_dataset_name()
    try:
        summary = dataset_summary(source_dataset_name)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise PreprocessingServiceError(
            f"Unable to summarize source dataset '{source_dataset_name}'."
        ) from exc

    available_columns = list(summary.get("column_names", []))
    preprocessing_config = get_shared_preprocessing_config()

    rule_statuses = [
        _build_drop_rule_status(preprocessing_config, available_columns),
        _build_numeric_imputation_rule_status(preprocessing_config, available_columns),
        _build_categorical_imputation_rule_status(preprocessing_config, available_columns),
        _build_datetime_rule_status(preprocessing_config, available_columns),
    ]

    warnings: list[str] = []
    for status in rule_statuses:
        if status.invalid_columns:
            warnings.append(
                f"Rule '{status.rule_name}' references missing columns: {', '.join(status.invalid_columns)}"
            )

    return SharedPreprocessingPlan(
        source_dataset_name=source_dataset_name,
        output_dataset_name=WORKING_DATASET_NAME,
        available_columns=available_columns,
        rule_statuses=rule_statuses,
        warnings=warnings,
    )



def preview_working_data_rebuild() -> WorkingDataRebuildPreview:
    """Return a phase-1A preview of the next Working Data rebuild.

    This preview is the contract that later execution phases will use when they
    actually materialize the rebuilt Working Data artifact.
    """
    plan = build_shared_preprocessing_plan()
    enabled_rule_count = sum(1 for status in plan.rule_statuses if status.enabled)
    message = (
        f"Working Data would be rebuilt from '{plan.source_dataset_name}' using "
        f"{enabled_rule_count} active shared preprocessing rule"
        f"{'s' if enabled_rule_count != 1 else ''}."
    )
    return WorkingDataRebuildPreview(
        status="preview_ready",
        message=message,
        plan=plan,
    )


def apply_shared_preprocessing_to_dataframe(
    source_df: pd.DataFrame,
    preprocessing_config: PreprocessingConfig | None = None,
) -> WorkingDataExecutionResult:
    """Apply shared preprocessing rules to a dataframe.

    Phase 1B executes drop-columns, numeric imputation, category/text
    imputation, and datetime expansion in a deterministic order.
    """
    if not isinstance(source_df, pd.DataFrame):
        raise PreprocessingServiceError("Source dataframe must be a pandas DataFrame.")

    resolved_preprocessing_config = preprocessing_config or get_shared_preprocessing_config()
    available_columns = list(source_df.columns)
    plan = SharedPreprocessingPlan(
        source_dataset_name=RAW_DATASET_NAME,
        output_dataset_name=WORKING_DATASET_NAME,
        available_columns=available_columns,
        rule_statuses=[
            _build_drop_rule_status(resolved_preprocessing_config, available_columns),
            _build_numeric_imputation_rule_status(resolved_preprocessing_config, available_columns),
            _build_categorical_imputation_rule_status(resolved_preprocessing_config, available_columns),
            _build_datetime_rule_status(resolved_preprocessing_config, available_columns),
        ],
        warnings=[],
    )

    warnings: list[str] = []
    working_df = source_df.copy()
    applied_steps: list[AppliedPreprocessingStep] = []

    numeric_status = PreprocessingRuleStatus(
        **{**plan.rule_statuses[1].__dict__, "valid_columns": [c for c in plan.rule_statuses[1].valid_columns if c in working_df.columns]}
    )
    working_df, numeric_step = _apply_numeric_imputation_step(
        working_df,
        numeric_status,
        resolved_preprocessing_config,
    )
    applied_steps.append(numeric_step)

    categorical_status = PreprocessingRuleStatus(
        **{**plan.rule_statuses[2].__dict__, "valid_columns": [c for c in plan.rule_statuses[2].valid_columns if c in working_df.columns]}
    )
    working_df, categorical_step = _apply_categorical_imputation_step(
        working_df,
        categorical_status,
        resolved_preprocessing_config,
    )
    applied_steps.append(categorical_step)

    datetime_status = PreprocessingRuleStatus(
        **{**plan.rule_statuses[3].__dict__, "valid_columns": [c for c in plan.rule_statuses[3].valid_columns if c in working_df.columns]}
    )
    working_df, datetime_step = _apply_datetime_expansion_step(working_df, datetime_status)
    applied_steps.append(datetime_step)

    working_df, feature_step = _apply_shared_feature_engineering_step(working_df)
    warnings.extend(feature_step.warnings)

    drop_status = PreprocessingRuleStatus(
        **{**plan.rule_statuses[0].__dict__, "valid_columns": [c for c in plan.rule_statuses[0].valid_columns if c in working_df.columns]}
    )
    working_df, drop_step = _apply_drop_columns_step(working_df, drop_status)
    applied_steps.append(drop_step)

    for step in applied_steps:
        if step.skipped_columns:
            warnings.append(
                f"Step '{step.rule_name}' skipped columns: {', '.join(step.skipped_columns)}"
            )

    applied_rule_count = sum(1 for step in applied_steps if step.applied_columns) + feature_step.applied_feature_count
    message = (
        f"Applied {applied_rule_count} shared preprocessing/feature step"
        f"{'s' if applied_rule_count != 1 else ''} to the Working Data rebuild."
    )

    return WorkingDataExecutionResult(
        status="execution_complete",
        message=message,
        dataframe=working_df,
        plan=plan,
        applied_steps=applied_steps,
        warnings=warnings,
    )



def execute_preview_plan_on_dataframe(source_df: pd.DataFrame) -> WorkingDataExecutionResult:
    """Execute the currently configured shared preprocessing plan on a dataframe."""
    return apply_shared_preprocessing_to_dataframe(source_df)


def rebuild_working_data_from_shared_rules() -> WorkingDataExecutionResult:
    """Rebuild the Working Data artifact from Raw Data plus shared rules.

    This is the main orchestration entrypoint for shared preprocessing actions in
    the Prepare tab. It always starts from Raw Data so rule changes can cleanly
    revert the derived Working Data artifact.
    """
    try:
        source_df = get_dataset_copy(RAW_DATASET_NAME)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise PreprocessingServiceError(
            f"Unable to load source dataset '{RAW_DATASET_NAME}' for preprocessing."
        ) from exc

    result = apply_shared_preprocessing_to_dataframe(source_df)
    set_working_dataset(result.dataframe, source_name=RAW_DATASET_NAME)
    return result