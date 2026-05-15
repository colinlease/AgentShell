

"""Shared engineered-feature service for the ML Workbench app.

Phase 1A establishes the backend contract for engineered features without yet
executing them inside the shared rebuild pipeline. The service is intentionally
UI-agnostic so both the standalone app and future AgentShell tools can use the
same feature definitions, validation rules, and preview contracts.

The design assumes engineered features are shared by default across candidate
models. Candidate-specific encoding, scaling, and rebalancing will happen later
in the model-input pipeline, while shared engineered features will eventually be
applied during the Working Data rebuild before final shared column removal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import ast
import math
import re

import pandas as pd

from app.workspace_apps.ml_workbench.schemas import (
    FeatureDependencySpec,
    FeatureSpec,
    FeatureValidationResult,
)
from app.workspace_apps.ml_workbench.state import append_feature_spec, get_app_state, remove_feature_spec

DEFAULT_FEATURE_EXECUTION_SCOPE = "shared"
DEFAULT_FEATURE_BUILDER_MODE = "guided"
DEFAULT_FEATURE_STATUS = "draft"
DEFAULT_FEATURE_EXPRESSION_LANGUAGE = "mlw_expr_v1"

SUPPORTED_FEATURE_OPERATION_FAMILIES = {
    "arithmetic",
    "transformation",
    "interaction",
    "polynomial",
    "flag",
    "expression",
}

SUPPORTED_FEATURE_OPERATIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "ratio",
    "log",
    "log1p",
    "interaction",
    "square",
    "cube",
    "flag_gt",
    "flag_gte",
    "flag_lt",
    "flag_lte",
    "flag_eq",
    "flag_is_missing",
    "expression",
}


ALLOWED_EXPRESSION_FUNCTIONS = {"log", "log1p", "square", "cube"}
_EXPRESSION_FIELD_TOKEN_PREFIX = "__mlw_field_"


class _SafeExpressionEvaluator(ast.NodeVisitor):
    """Safely evaluate a constrained feature expression against a dataframe."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.dependencies: list[str] = []

    def _track_dependency(self, dependency_name: str) -> None:
        if dependency_name not in self.dependencies:
            self.dependencies.append(dependency_name)

    def visit_Expression(self, node: ast.Expression) -> pd.Series | float:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> pd.Series:
        token_to_column = getattr(self, "token_to_column", {})
        if node.id in token_to_column:
            column_name = token_to_column[node.id]
        else:
            column_name = node.id
        if column_name not in self.df.columns:
            raise FeatureServiceError(f"Referenced source column '{column_name}' is not available.")
        self._track_dependency(column_name)
        return _coerce_numeric_series(self.df[column_name])

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise FeatureServiceError("Only numeric constants are allowed in expressions.")

    def visit_Num(self, node: ast.Num) -> float:
        return float(node.n)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> pd.Series | float:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise FeatureServiceError("Only unary plus and unary minus are allowed in expressions.")

    def visit_BinOp(self, node: ast.BinOp) -> pd.Series | float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if isinstance(right, pd.Series):
                right = right.replace(0, pd.NA)
            elif right == 0:
                return pd.Series([pd.NA] * len(self.df), index=self.df.index, dtype="Float64")
            return left / right
        if isinstance(node.op, ast.Pow):
            if isinstance(right, pd.Series):
                raise FeatureServiceError("Power expressions must use a numeric exponent.")
            exponent = float(right)
            if exponent == 2:
                return left**2
            if exponent == 3:
                return left**3
            raise FeatureServiceError("Only exponents of 2 or 3 are supported in expressions.")
        raise FeatureServiceError("Only +, -, *, /, and exponents of 2 or 3 are allowed in expressions.")

    def visit_Call(self, node: ast.Call) -> pd.Series:
        if not isinstance(node.func, ast.Name):
            raise FeatureServiceError("Only simple function names are allowed in expressions.")
        function_name = node.func.id
        if function_name not in ALLOWED_EXPRESSION_FUNCTIONS:
            raise FeatureServiceError(f"Function '{function_name}' is not supported in expressions.")
        if len(node.args) != 1:
            raise FeatureServiceError(f"Function '{function_name}' requires exactly one argument.")
        operand = self.visit(node.args[0])
        if function_name == "log":
            return operand.where(operand > 0).map(lambda value: pd.NA if pd.isna(value) else float(math.log(value)))
        if function_name == "log1p":
            return operand.where(operand > -1).map(lambda value: pd.NA if pd.isna(value) else float(math.log1p(value)))
        if function_name == "square":
            return operand**2
        if function_name == "cube":
            return operand**3
        raise FeatureServiceError(f"Function '{function_name}' is not supported in expressions.")

    def generic_visit(self, node: ast.AST) -> Any:
        raise FeatureServiceError(f"Unsupported expression element: {node.__class__.__name__}.")


@dataclass(frozen=True)
class FeaturePlan:
    """Resolved execution plan for one engineered feature."""

    feature_id: str
    feature_name: str
    feature_type: str
    builder_mode: str
    operation_family: str
    operation: str
    expression: str
    expression_language: str | None
    source_columns: list[str]
    dependencies: list[FeatureDependencySpec]
    execution_scope: str
    apply_order: int
    enabled: bool
    parameters: dict[str, Any] = field(default_factory=dict)
    validation: FeatureValidationResult = field(default_factory=dict)


@dataclass(frozen=True)
class FeaturePreviewResult:
    """Structured preview result for one engineered feature definition."""

    status: str
    message: str
    feature_spec: FeatureSpec
    validation: FeatureValidationResult
    preview_dataframe: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureCollectionPlan:
    """Resolved plan for a collection of engineered features."""

    feature_count: int
    enabled_feature_count: int
    sorted_feature_ids: list[str]
    validation_results: dict[str, FeatureValidationResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureExecutionStep:
    """Execution summary for one engineered feature."""

    feature_id: str
    feature_name: str
    operation: str
    status: str
    output_column: str | None = None
    dependencies: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None


@dataclass(frozen=True)
class FeatureExecutionResult:
    """Result of applying engineered features to a dataframe."""

    status: str
    message: str
    dataframe: pd.DataFrame
    steps: list[FeatureExecutionStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FeatureServiceError(RuntimeError):
    """Raised when a feature workflow request cannot be completed."""



def _next_feature_id(existing_features: list[FeatureSpec]) -> str:
    """Generate the next stable feature id."""
    existing_ids = {str(feature.get("feature_id", "")) for feature in existing_features}
    index = 1
    while True:
        feature_id = f"feature_{index:03d}"
        if feature_id not in existing_ids:
            return feature_id
        index += 1



def _next_apply_order(existing_features: list[FeatureSpec]) -> int:
    """Return the next default apply order for a new feature."""
    if not existing_features:
        return 1
    return max(int(feature.get("apply_order", 0)) for feature in existing_features) + 1



def get_feature_specs() -> list[FeatureSpec]:
    """Return the current engineered feature specs from app state."""
    return list(get_app_state().get("feature_specs", []))




def _normalize_dependencies(
    source_columns: list[str],
    dependencies: list[FeatureDependencySpec] | None = None,
) -> list[FeatureDependencySpec]:
    """Return normalized dependency specs for a feature definition."""
    if dependencies:
        normalized_dependencies: list[FeatureDependencySpec] = []
        for dependency in dependencies:
            dependency_kind = str(dependency.get("dependency_kind", "column")).strip() or "column"
            dependency_name = str(dependency.get("dependency_name", "")).strip()
            if not dependency_name:
                continue
            normalized_dependencies.append(
                {
                    "dependency_kind": dependency_kind,
                    "dependency_name": dependency_name,
                }
            )
        return normalized_dependencies

    normalized_source_columns: list[str] = []
    for column in source_columns:
        column_name = str(column).strip()
        if column_name and column_name not in normalized_source_columns:
            normalized_source_columns.append(column_name)

    return [
        {
            "dependency_kind": "column",
            "dependency_name": column_name,
        }
        for column_name in normalized_source_columns
    ]


def _normalize_expression_field_tokens(expression: str) -> tuple[str, list[str]]:
    """Replace bracketed field references with safe parser tokens."""
    dependencies: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        field_name = match.group(1).strip()
        if not field_name:
            raise FeatureServiceError("Bracketed field references cannot be empty.")
        if field_name not in dependencies:
            dependencies.append(field_name)
        safe_field_name = re.sub(r"\W", "_", field_name)
        return f"{_EXPRESSION_FIELD_TOKEN_PREFIX}{safe_field_name}"

    normalized_expression = re.sub(r"\[([^\[\]]+)\]", _replace, expression)
    normalized_expression = normalized_expression.replace("^", "**")
    return normalized_expression, dependencies



def _extract_simple_expression_dependencies(expression: str) -> list[str]:
    """Extract likely single-word field references from an expression."""
    bracketless_expression = re.sub(r"\[([^\[\]]+)\]", " ", expression)
    candidates = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", bracketless_expression)
    dependencies: list[str] = []
    for candidate in candidates:
        if candidate in ALLOWED_EXPRESSION_FUNCTIONS:
            continue
        if candidate not in dependencies:
            dependencies.append(candidate)
    return dependencies



def _parse_expression(expression: str) -> tuple[ast.Expression, dict[str, str], list[str]]:
    """Parse a constrained feature expression into an AST."""
    token_to_column: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        field_name = match.group(1).strip()
        if not field_name:
            raise FeatureServiceError("Bracketed field references cannot be empty.")
        safe_field_name = re.sub(r"\W", "_", field_name)
        token_name = f"{_EXPRESSION_FIELD_TOKEN_PREFIX}{safe_field_name}"
        token_to_column[token_name] = field_name
        return token_name

    normalized_expression = re.sub(r"\[([^\[\]]+)\]", _replace, expression)
    normalized_expression = normalized_expression.replace("^", "**")
    try:
        parsed_expression = ast.parse(normalized_expression, mode="eval")
    except SyntaxError as exc:
        raise FeatureServiceError("Expression could not be parsed. Check parentheses and operators.") from exc

    dependencies = list(token_to_column.values())
    for dependency_name in _extract_simple_expression_dependencies(expression):
        if dependency_name not in dependencies:
            dependencies.append(dependency_name)
    return parsed_expression, token_to_column, dependencies



def _evaluate_expression_to_series(expression: str, df: pd.DataFrame) -> tuple[pd.Series, list[str], list[str]]:
    """Safely evaluate a constrained expression against a dataframe."""
    parsed_expression, token_to_column, dependencies = _parse_expression(expression)
    evaluator = _SafeExpressionEvaluator(df)
    evaluator.token_to_column = token_to_column
    result = evaluator.visit(parsed_expression)
    combined_dependencies = list(dependencies)
    for dependency_name in evaluator.dependencies:
        if dependency_name not in combined_dependencies:
            combined_dependencies.append(dependency_name)
    warnings: list[str] = []
    compact_expression = expression.replace(" ", "")
    if "/" in compact_expression:
        warnings.append("Division uses null output where a denominator is zero.")
    if "log(" in compact_expression:
        warnings.append("Log transform returns null for non-positive input values.")
    if "log1p(" in compact_expression:
        warnings.append("Log1p transform returns null for values less than or equal to -1.")
    if isinstance(result, pd.Series):
        return result, combined_dependencies, warnings
    return pd.Series([result] * len(df), index=df.index), combined_dependencies, warnings



def build_default_feature_spec(
    *,
    feature_name: str,
    feature_type: str,
    operation_family: str,
    operation: str,
    source_columns: list[str] | None = None,
    expression: str = "",
    parameters: dict[str, Any] | None = None,
    builder_mode: str = DEFAULT_FEATURE_BUILDER_MODE,
    expression_language: str | None = DEFAULT_FEATURE_EXPRESSION_LANGUAGE,
    created_by: str = "user",
    description: str | None = None,
    output_dtype: str | None = None,
    dependencies: list[FeatureDependencySpec] | None = None,
) -> FeatureSpec:
    """Build a default engineered-feature spec.

    This shared contract can support both a guided builder and a future
    expression editor by changing the builder mode and expression fields while
    preserving the same core schema.
    """
    existing_features = get_feature_specs()
    resolved_source_columns = [str(column).strip() for column in (source_columns or []) if str(column).strip()]
    if builder_mode.strip() == "expression" and expression.strip() and not resolved_source_columns:
        resolved_source_columns = _extract_simple_expression_dependencies(expression.strip())
    normalized_dependencies = _normalize_dependencies(resolved_source_columns, dependencies)

    feature_spec: FeatureSpec = {
        "feature_id": _next_feature_id(existing_features),
        "feature_name": feature_name.strip(),
        "feature_type": feature_type.strip(),
        "builder_mode": builder_mode.strip() or DEFAULT_FEATURE_BUILDER_MODE,
        "operation_family": operation_family.strip(),
        "operation": operation.strip(),
        "expression": expression.strip(),
        "expression_language": expression_language,
        "parameters": dict(parameters or {}),
        "source_columns": resolved_source_columns,
        "dependencies": normalized_dependencies,
        "enabled": True,
        "execution_scope": DEFAULT_FEATURE_EXECUTION_SCOPE,
        "apply_order": _next_apply_order(existing_features),
        "created_by": created_by.strip() or "user",
        "status": DEFAULT_FEATURE_STATUS,
    }

    if description is not None:
        feature_spec["description"] = description
    if output_dtype is not None:
        feature_spec["output_dtype"] = output_dtype

    return feature_spec


def create_and_store_feature_spec(
    *,
    feature_name: str,
    feature_type: str,
    operation_family: str,
    operation: str,
    source_columns: list[str] | None = None,
    expression: str = "",
    parameters: dict[str, Any] | None = None,
    builder_mode: str = DEFAULT_FEATURE_BUILDER_MODE,
    expression_language: str | None = DEFAULT_FEATURE_EXPRESSION_LANGUAGE,
    created_by: str = "user",
    description: str | None = None,
    output_dtype: str | None = None,
    dependencies: list[FeatureDependencySpec] | None = None,
) -> FeatureSpec:
    """Build and persist one engineered feature spec in app state."""
    feature_spec = build_default_feature_spec(
        feature_name=feature_name,
        feature_type=feature_type,
        operation_family=operation_family,
        operation=operation,
        source_columns=source_columns,
        expression=expression,
        parameters=parameters,
        builder_mode=builder_mode,
        expression_language=expression_language,
        created_by=created_by,
        description=description,
        output_dtype=output_dtype,
        dependencies=dependencies,
    )
    append_feature_spec(feature_spec)
    return feature_spec


def remove_stored_feature_specs(feature_ids: list[str]) -> list[FeatureSpec]:
    """Remove one or more engineered feature specs from app state."""
    removed_feature_specs: list[FeatureSpec] = []
    seen_feature_ids: set[str] = set()
    existing_features = list(get_feature_specs())
    features_by_id = {
        str(feature_spec.get("feature_id", "")): feature_spec
        for feature_spec in existing_features
        if str(feature_spec.get("feature_id", "")).strip()
    }
    for feature_id in feature_ids:
        normalized_feature_id = str(feature_id).strip()
        if not normalized_feature_id or normalized_feature_id in seen_feature_ids:
            continue
        seen_feature_ids.add(normalized_feature_id)
        feature_spec = features_by_id.get(normalized_feature_id)
        if feature_spec is None:
            continue
        removed_feature_specs.append(feature_spec)
        remove_feature_spec(normalized_feature_id)
    return removed_feature_specs



def _validate_operation_family(feature_spec: FeatureSpec) -> list[str]:
    """Validate the feature operation family."""
    warnings: list[str] = []
    operation_family = str(feature_spec.get("operation_family", "")).strip()
    if operation_family not in SUPPORTED_FEATURE_OPERATION_FAMILIES:
        warnings.append(
            f"Operation family '{operation_family}' is not currently supported."
        )
    return warnings



def _validate_operation(feature_spec: FeatureSpec) -> list[str]:
    """Validate the feature operation."""
    warnings: list[str] = []
    operation = str(feature_spec.get("operation", "")).strip()
    if operation not in SUPPORTED_FEATURE_OPERATIONS:
        warnings.append(f"Operation '{operation}' is not currently supported.")
    return warnings



def _validate_feature_name(feature_spec: FeatureSpec) -> list[str]:
    """Validate the feature name."""
    warnings: list[str] = []
    feature_name = str(feature_spec.get("feature_name", "")).strip()
    if not feature_name:
        warnings.append("Feature name is required.")
    return warnings



def _validate_dependencies(
    feature_spec: FeatureSpec,
    available_columns: list[str],
    available_feature_names: list[str],
) -> list[str]:
    """Validate dependencies against available inputs."""
    warnings: list[str] = []
    available_column_set = set(available_columns)
    available_feature_name_set = set(available_feature_names)

    for dependency in feature_spec.get("dependencies", []):
        dependency_kind = str(dependency.get("dependency_kind", "column")).strip() or "column"
        dependency_name = str(dependency.get("dependency_name", "")).strip()
        if not dependency_name:
            warnings.append("Feature dependency is missing a dependency name.")
            continue
        if dependency_kind == "column" and dependency_name not in available_column_set:
            warnings.append(
                f"Referenced source column '{dependency_name}' is not available."
            )
        elif dependency_kind == "feature" and dependency_name not in available_feature_name_set:
            warnings.append(
                f"Referenced engineered feature '{dependency_name}' is not available."
            )

    return warnings



def validate_feature_spec(
    feature_spec: FeatureSpec,
    *,
    available_columns: list[str],
    available_feature_names: list[str] | None = None,
) -> FeatureValidationResult:
    """Validate one engineered feature definition.

    Phase 1A focuses on structural validation rather than full dataframe
    execution. The result is stored in the shared contract so both UI and future
    tools can reason about the feature definition consistently.
    """
    warnings: list[str] = []
    warnings.extend(_validate_feature_name(feature_spec))
    warnings.extend(_validate_operation_family(feature_spec))
    warnings.extend(_validate_operation(feature_spec))

    if str(feature_spec.get("builder_mode", "")).strip() == "expression":
        expression = str(feature_spec.get("expression", "")).strip()
        if not expression:
            warnings.append("Expression is required for expression-based features.")
        else:
            try:
                _, _, expression_dependencies = _parse_expression(expression)
                feature_spec["source_columns"] = expression_dependencies
                feature_spec["dependencies"] = _normalize_dependencies(expression_dependencies)
            except FeatureServiceError as exc:
                warnings.append(str(exc))

    warnings.extend(
        _validate_dependencies(
            feature_spec,
            available_columns=available_columns,
            available_feature_names=list(available_feature_names or []),
        )
    )

    validation_result: FeatureValidationResult = {
        "is_valid": len(warnings) == 0,
        "warnings": warnings,
        "error_message": None if len(warnings) == 0 else warnings[0],
    }
    return validation_result



def build_feature_plan(
    feature_spec: FeatureSpec,
    *,
    available_columns: list[str],
    available_feature_names: list[str] | None = None,
) -> FeaturePlan:
    """Build the resolved plan for one engineered feature."""
    validation = validate_feature_spec(
        feature_spec,
        available_columns=available_columns,
        available_feature_names=available_feature_names,
    )
    return FeaturePlan(
        feature_id=str(feature_spec.get("feature_id", "")),
        feature_name=str(feature_spec.get("feature_name", "")),
        feature_type=str(feature_spec.get("feature_type", "")),
        builder_mode=str(feature_spec.get("builder_mode", DEFAULT_FEATURE_BUILDER_MODE)),
        operation_family=str(feature_spec.get("operation_family", "")),
        operation=str(feature_spec.get("operation", "")),
        expression=str(feature_spec.get("expression", "")),
        expression_language=feature_spec.get("expression_language"),
        source_columns=list(feature_spec.get("source_columns", [])),
        dependencies=list(feature_spec.get("dependencies", [])),
        execution_scope=str(feature_spec.get("execution_scope", DEFAULT_FEATURE_EXECUTION_SCOPE)),
        apply_order=int(feature_spec.get("apply_order", 0)),
        enabled=bool(feature_spec.get("enabled", True)),
        parameters=dict(feature_spec.get("parameters", {})),
        validation=validation,
    )



def build_feature_collection_plan(
    feature_specs: list[FeatureSpec] | None = None,
    *,
    available_columns: list[str],
) -> FeatureCollectionPlan:
    """Build a collection-wide plan for engineered features."""
    resolved_feature_specs = list(feature_specs or get_feature_specs())
    sorted_specs = sorted(
        resolved_feature_specs,
        key=lambda feature: int(feature.get("apply_order", 0)),
    )

    validation_results: dict[str, FeatureValidationResult] = {}
    warnings: list[str] = []
    known_feature_names: list[str] = []

    for feature_spec in sorted_specs:
        feature_id = str(feature_spec.get("feature_id", ""))
        validation = validate_feature_spec(
            feature_spec,
            available_columns=available_columns,
            available_feature_names=known_feature_names,
        )
        validation_results[feature_id] = validation
        for warning in validation.get("warnings", []):
            warnings.append(f"{feature_id}: {warning}")
        if validation.get("is_valid"):
            feature_name = str(feature_spec.get("feature_name", "")).strip()
            if feature_name:
                known_feature_names.append(feature_name)

    return FeatureCollectionPlan(
        feature_count=len(sorted_specs),
        enabled_feature_count=sum(1 for feature in sorted_specs if bool(feature.get("enabled", True))),
        sorted_feature_ids=[str(feature.get("feature_id", "")) for feature in sorted_specs],
        validation_results=validation_results,
        warnings=warnings,
    )



def preview_feature_spec(
    feature_spec: FeatureSpec,
    *,
    source_df: pd.DataFrame | None = None,
) -> FeaturePreviewResult:
    """Return a structured preview result for one feature definition.

    Phase 1A validates the definition and optionally returns a lightweight input
    preview. Full feature execution will be added in later phases.
    """
    available_columns = list(source_df.columns) if isinstance(source_df, pd.DataFrame) else []
    validation = validate_feature_spec(
        feature_spec,
        available_columns=available_columns,
        available_feature_names=[],
    )

    preview_dataframe: pd.DataFrame | None = None
    if isinstance(source_df, pd.DataFrame) and available_columns:
        preview_columns = [
            column
            for column in feature_spec.get("source_columns", [])
            if column in source_df.columns
        ]
        if preview_columns:
            preview_dataframe = source_df[preview_columns].head(10).copy()

    message = (
        f"Feature '{feature_spec.get('feature_name', '')}' is valid."
        if validation.get("is_valid")
        else f"Feature '{feature_spec.get('feature_name', '')}' has validation issues."
    )

    return FeaturePreviewResult(
        status="preview_ready" if validation.get("is_valid") else "preview_invalid",
        message=message,
        feature_spec=feature_spec,
        validation=validation,
        preview_dataframe=preview_dataframe,
        warnings=list(validation.get("warnings", [])),
    )


def _get_required_source_columns(feature_spec: FeatureSpec) -> list[str]:
    """Return the normalized source columns required for a feature."""
    return [str(column).strip() for column in feature_spec.get("source_columns", []) if str(column).strip()]



def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Return a numeric version of a series using pandas coercion rules."""
    return pd.to_numeric(series, errors="coerce")


def _normalize_engineered_feature_output_dtype(
    feature_spec: FeatureSpec,
    series: pd.Series,
) -> pd.Series:
    """Normalize one engineered feature output to a stable pandas dtype.

    Numeric feature operations should remain numeric even when the computation
    introduces missing values. Pandas may otherwise promote mixed float/NA
    results to ``object``, which later model validators correctly reject.
    """
    operation = str(feature_spec.get("operation", "")).strip()
    if operation in {
        "add",
        "subtract",
        "multiply",
        "interaction",
        "divide",
        "ratio",
        "log",
        "log1p",
        "square",
        "cube",
        "expression",
    }:
        return pd.to_numeric(series, errors="coerce").astype("Float64")
    if operation in {
        "flag_gt",
        "flag_gte",
        "flag_lt",
        "flag_lte",
        "flag_eq",
        "flag_is_missing",
    }:
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    return series


def _coerce_compare_value_for_series(series: pd.Series, compare_value: object) -> object:
    """Coerce a compare value to match a source series when possible."""
    if compare_value is None:
        return None

    numeric_series = pd.to_numeric(series, errors="coerce")
    has_numeric_values = numeric_series.notna().any()
    if has_numeric_values:
        numeric_compare_value = pd.to_numeric(pd.Series([compare_value]), errors="coerce").iloc[0]
        if pd.notna(numeric_compare_value):
            return numeric_compare_value.item() if hasattr(numeric_compare_value, "item") else numeric_compare_value

    return compare_value


def _build_feature_step(
    *,
    feature_spec: FeatureSpec,
    status: str,
    dependencies: list[str],
    warnings: list[str] | None = None,
    error_message: str | None = None,
) -> FeatureExecutionStep:
    """Build one feature execution step summary."""
    return FeatureExecutionStep(
        feature_id=str(feature_spec.get("feature_id", "")),
        feature_name=str(feature_spec.get("feature_name", "")),
        operation=str(feature_spec.get("operation", "")),
        status=status,
        output_column=str(feature_spec.get("feature_name", "")).strip() or None,
        dependencies=dependencies,
        warnings=list(warnings or []),
        error_message=error_message,
    )


def _validate_required_source_columns(feature_spec: FeatureSpec, df: pd.DataFrame) -> list[str]:
    """Return missing source columns required by a feature spec."""
    required_columns = _get_required_source_columns(feature_spec)
    return [column for column in required_columns if column not in df.columns]


def _resolve_binary_numeric_operands(feature_spec: FeatureSpec, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, list[str]]:
    """Resolve two numeric operands for binary arithmetic operations."""
    source_columns = _get_required_source_columns(feature_spec)
    if len(source_columns) < 2:
        raise FeatureServiceError(
            f"Feature '{feature_spec.get('feature_name', '')}' requires two source columns."
        )
    left_name, right_name = source_columns[0], source_columns[1]
    return _coerce_numeric_series(df[left_name]), _coerce_numeric_series(df[right_name]), [left_name, right_name]


def _resolve_unary_numeric_operand(feature_spec: FeatureSpec, df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Resolve one numeric operand for unary transformations."""
    source_columns = _get_required_source_columns(feature_spec)
    if len(source_columns) < 1:
        raise FeatureServiceError(
            f"Feature '{feature_spec.get('feature_name', '')}' requires one source column."
        )
    source_name = source_columns[0]
    return _coerce_numeric_series(df[source_name]), [source_name]


def apply_feature_spec_to_dataframe(
    feature_spec: FeatureSpec,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, FeatureExecutionStep]:
    """Apply one engineered feature spec to a dataframe.

    Phase 1B implements the first guided-builder operations for shared features.
    """
    if not isinstance(df, pd.DataFrame):
        raise FeatureServiceError("Feature execution requires a pandas DataFrame.")

    feature_name = str(feature_spec.get("feature_name", "")).strip()
    if not feature_name:
        raise FeatureServiceError("Feature execution requires a non-empty feature name.")

    if not bool(feature_spec.get("enabled", True)):
        return df, _build_feature_step(
            feature_spec=feature_spec,
            status="skipped_disabled",
            dependencies=_get_required_source_columns(feature_spec),
            warnings=["Feature is disabled and was not applied."],
        )

    missing_columns = _validate_required_source_columns(feature_spec, df)
    if missing_columns:
        return df, _build_feature_step(
            feature_spec=feature_spec,
            status="skipped_missing_dependencies",
            dependencies=_get_required_source_columns(feature_spec),
            warnings=[f"Missing source columns: {', '.join(missing_columns)}"],
            error_message=f"Missing source columns: {', '.join(missing_columns)}",
        )

    updated_df = df.copy()
    operation = str(feature_spec.get("operation", "")).strip()
    warnings: list[str] = []
    dependencies = _get_required_source_columns(feature_spec)

    # Expression support
    if operation == "expression":
        expression = str(feature_spec.get("expression", "")).strip()
        if not expression:
            return df, _build_feature_step(
                feature_spec=feature_spec,
                status="error",
                dependencies=dependencies,
                error_message="Expression is required for expression-based features.",
            )
        try:
            evaluated_series, dependencies, warnings = _evaluate_expression_to_series(expression, updated_df)
            updated_df[feature_name] = _normalize_engineered_feature_output_dtype(
                feature_spec,
                evaluated_series,
            )
            return updated_df, _build_feature_step(
                feature_spec=feature_spec,
                status="applied",
                dependencies=dependencies,
                warnings=warnings,
            )
        except Exception as exc:
            return df, _build_feature_step(
                feature_spec=feature_spec,
                status="error",
                dependencies=dependencies,
                warnings=warnings,
                error_message=str(exc),
            )

    try:
        if operation == "add":
            left, right, dependencies = _resolve_binary_numeric_operands(feature_spec, updated_df)
            updated_df[feature_name] = left + right
        elif operation == "subtract":
            left, right, dependencies = _resolve_binary_numeric_operands(feature_spec, updated_df)
            updated_df[feature_name] = left - right
        elif operation in {"multiply", "interaction"}:
            left, right, dependencies = _resolve_binary_numeric_operands(feature_spec, updated_df)
            updated_df[feature_name] = left * right
        elif operation in {"divide", "ratio"}:
            left, right, dependencies = _resolve_binary_numeric_operands(feature_spec, updated_df)
            safe_denominator = right.replace(0, pd.NA)
            updated_df[feature_name] = left / safe_denominator
            warnings.append("Division uses null output where the denominator is zero.")
        elif operation == "log":
            operand, dependencies = _resolve_unary_numeric_operand(feature_spec, updated_df)
            updated_df[feature_name] = operand.where(operand > 0).map(lambda value: pd.NA if pd.isna(value) else float(__import__('math').log(value)))
            warnings.append("Log transform returns null for non-positive input values.")
        elif operation == "log1p":
            operand, dependencies = _resolve_unary_numeric_operand(feature_spec, updated_df)
            updated_df[feature_name] = operand.where(operand > -1).map(lambda value: pd.NA if pd.isna(value) else float(__import__('math').log1p(value)))
            warnings.append("Log1p transform returns null for values less than or equal to -1.")
        elif operation == "square":
            operand, dependencies = _resolve_unary_numeric_operand(feature_spec, updated_df)
            updated_df[feature_name] = operand ** 2
        elif operation == "cube":
            operand, dependencies = _resolve_unary_numeric_operand(feature_spec, updated_df)
            updated_df[feature_name] = operand ** 3
        elif operation == "flag_gt":
            operand, dependencies = _resolve_unary_numeric_operand(feature_spec, updated_df)
            threshold = float(feature_spec.get("parameters", {}).get("threshold"))
            updated_df[feature_name] = (operand > threshold).astype("Int64")
        elif operation == "flag_gte":
            operand, dependencies = _resolve_unary_numeric_operand(feature_spec, updated_df)
            threshold = float(feature_spec.get("parameters", {}).get("threshold"))
            updated_df[feature_name] = (operand >= threshold).astype("Int64")
        elif operation == "flag_lt":
            operand, dependencies = _resolve_unary_numeric_operand(feature_spec, updated_df)
            threshold = float(feature_spec.get("parameters", {}).get("threshold"))
            updated_df[feature_name] = (operand < threshold).astype("Int64")
        elif operation == "flag_lte":
            operand, dependencies = _resolve_unary_numeric_operand(feature_spec, updated_df)
            threshold = float(feature_spec.get("parameters", {}).get("threshold"))
            updated_df[feature_name] = (operand <= threshold).astype("Int64")
        elif operation == "flag_eq":
            source_columns = _get_required_source_columns(feature_spec)
            source_name = source_columns[0]
            source_series = updated_df[source_name]
            compare_value = feature_spec.get("parameters", {}).get("compare_value")
            compare_value = _coerce_compare_value_for_series(source_series, compare_value)
            updated_df[feature_name] = (source_series == compare_value).astype("Int64")
            dependencies = [source_name]
        elif operation == "flag_is_missing":
            source_columns = _get_required_source_columns(feature_spec)
            source_name = source_columns[0]
            updated_df[feature_name] = updated_df[source_name].isna().astype("Int64")
            dependencies = [source_name]
        else:
            return df, _build_feature_step(
                feature_spec=feature_spec,
                status="skipped_unsupported_operation",
                dependencies=dependencies,
                warnings=[f"Operation '{operation}' is not implemented yet."],
                error_message=f"Operation '{operation}' is not implemented yet.",
            )
    except Exception as exc:
        return df, _build_feature_step(
            feature_spec=feature_spec,
            status="error",
            dependencies=dependencies,
            warnings=warnings,
            error_message=str(exc),
        )

    updated_df[feature_name] = _normalize_engineered_feature_output_dtype(
        feature_spec,
        updated_df[feature_name],
    )

    return updated_df, _build_feature_step(
        feature_spec=feature_spec,
        status="applied",
        dependencies=dependencies,
        warnings=warnings,
    )


def apply_feature_specs_to_dataframe(
    source_df: pd.DataFrame,
    feature_specs: list[FeatureSpec] | None = None,
) -> FeatureExecutionResult:
    """Apply enabled engineered features to a dataframe in apply-order sequence."""
    if not isinstance(source_df, pd.DataFrame):
        raise FeatureServiceError("Feature execution requires a pandas DataFrame.")

    resolved_feature_specs = list(feature_specs or get_feature_specs())
    sorted_feature_specs = sorted(
        resolved_feature_specs,
        key=lambda feature: int(feature.get("apply_order", 0)),
    )

    working_df = source_df.copy()
    steps: list[FeatureExecutionStep] = []
    warnings: list[str] = []

    for feature_spec in sorted_feature_specs:
        working_df, step = apply_feature_spec_to_dataframe(feature_spec, working_df)
        steps.append(step)
        for warning in step.warnings:
            warnings.append(f"{step.feature_name}: {warning}")
        if step.error_message:
            warnings.append(f"{step.feature_name}: {step.error_message}")

    applied_count = sum(1 for step in steps if step.status == "applied")
    message = (
        f"Applied {applied_count} engineered feature"
        f"{'s' if applied_count != 1 else ''} to the dataframe."
    )

    return FeatureExecutionResult(
        status="execution_complete",
        message=message,
        dataframe=working_df,
        steps=steps,
        warnings=warnings,
    )
