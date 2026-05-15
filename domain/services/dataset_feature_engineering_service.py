from __future__ import annotations

import importlib
import math
from copy import deepcopy
from typing import Any

import pandas as pd
import streamlit as st

from config.settings import AppSettings, get_settings
from domain.services.data_context_service import DataContextService
from domain.services.derived_dataset_store import (
    get_active_derived_metadata,
    get_derived_dataset_object,
    set_active_derived_dataset,
)


MAX_FEATURES_PER_CALL = 10
SUPPORTED_BINARY_OPERATIONS = {"add", "subtract", "multiply", "divide", "ratio"}
SUPPORTED_UNARY_OPERATIONS = {"log", "log1p", "square", "cube"}
SUPPORTED_FLAG_OPERATIONS = {"flag_gt", "flag_gte", "flag_lt", "flag_lte", "flag_eq", "flag_is_missing"}
SUPPORTED_DATETIME_PARTS = {"year", "quarter", "month", "week", "day", "dayofweek", "hour"}


class DatasetFeatureEngineeringService:
    """
    Framework-level feature derivation over app-published DataFrames.

    The service never mutates app-owned datasets. Successful derivations are
    stored as one session-scoped framework dataset that becomes active for
    later general data tools.
    """

    def __init__(
        self,
        *,
        data_context_service: DataContextService | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.data_context_service = data_context_service or DataContextService()
        self.settings = settings or get_settings()

    def derive_dataset_features(
        self,
        *,
        dataset_name: str | None = None,
        output_dataset_name: str | None = None,
        features: list[dict[str, Any]] | None = None,
        preview_rows: int = 10,
    ) -> dict[str, Any]:
        normalized_preview_rows = max(1, min(int(preview_rows or 10), 20))
        normalized_features = self._normalize_features(features)
        if isinstance(normalized_features, dict):
            return normalized_features

        source = self._resolve_source_dataset(dataset_name)
        if source["status"] != "ok":
            return source

        source_name = str(source["dataset_name"])
        source_df = source["dataframe"]
        if not isinstance(source_df, pd.DataFrame):
            return self._error("The requested dataset is not available as a pandas DataFrame.")

        validation_error = self._validate_feature_collection(source_df, normalized_features)
        if validation_error is not None:
            return validation_error

        derived_name = self._build_output_dataset_name(
            source_dataset_name=source_name,
            output_dataset_name=output_dataset_name,
            source_is_derived=bool(source.get("is_derived", False)),
        )
        estimated_memory_bytes = self._estimate_materialized_memory_bytes(
            source_df=source_df,
            feature_count=len(normalized_features),
        )
        max_memory_bytes = max(1, int(self.settings.derived_dataset_max_memory_mb)) * 1024 * 1024
        warnings: list[str] = []

        if estimated_memory_bytes > max_memory_bytes:
            preview_df, preview_warnings = self._apply_features(
                source_df.head(normalized_preview_rows).copy(deep=True),
                normalized_features,
            )
            warnings.extend(preview_warnings)
            warnings.append(
                "Derived dataset was previewed only because the estimated in-memory copy "
                f"({estimated_memory_bytes} bytes) exceeds the configured budget ({max_memory_bytes} bytes)."
            )
            return {
                "status": "preview_only",
                "message": "Derived features were previewed but not stored because the memory budget would be exceeded.",
                "dataset_name": derived_name,
                "source_dataset_name": source_name,
                "is_active_dataset": False,
                "created_columns": [feature["name"] for feature in normalized_features],
                "estimated_memory_bytes": estimated_memory_bytes,
                "max_memory_bytes": max_memory_bytes,
                "warnings": warnings,
                "preview_rows": self._serialize_records(preview_df.head(normalized_preview_rows)),
                "guidance": "Future framework dataset tools still default to the previous active dataset because no derived dataset was stored.",
            }

        derived_df, execution_warnings = self._apply_features(source_df.copy(deep=True), normalized_features)
        warnings.extend(execution_warnings)
        if int(derived_df.shape[0]) != int(source_df.shape[0]):
            return self._error("Feature derivation changed the dataset row count, so no derived dataset was stored.")

        existing_feature_specs: list[dict[str, Any]] = []
        existing_feature_columns: list[str] = []
        active_derived_metadata = get_active_derived_metadata()
        if bool(source.get("is_derived", False)) and isinstance(active_derived_metadata, dict):
            if str(active_derived_metadata.get("name") or "") == source_name:
                existing_feature_specs = [
                    dict(feature)
                    for feature in active_derived_metadata.get("feature_specs", []) or []
                    if isinstance(feature, dict)
                ]
                existing_feature_columns = [
                    str(column)
                    for column in active_derived_metadata.get("feature_columns", []) or []
                    if str(column).strip()
                ]

        created_columns = [feature["name"] for feature in normalized_features]
        feature_columns = [*existing_feature_columns]
        for column in created_columns:
            if column not in feature_columns:
                feature_columns.append(column)

        metadata = set_active_derived_dataset(
            dataframe=derived_df,
            name=derived_name,
            source_dataset_name=source_name,
            feature_specs=[*existing_feature_specs, *deepcopy(normalized_features)],
            created_columns=created_columns,
            feature_columns=feature_columns,
            estimated_memory_bytes=estimated_memory_bytes,
        )

        return {
            "status": "ok",
            "message": "Derived dataset was created and is now the active framework dataset.",
            "dataset_name": str(metadata.get("name", derived_name)),
            "source_dataset_name": source_name,
            "is_active_dataset": True,
            "created_columns": created_columns,
            "rows": int(derived_df.shape[0]),
            "columns": int(derived_df.shape[1]),
            "estimated_memory_bytes": estimated_memory_bytes,
            "warnings": warnings,
            "preview_rows": self._serialize_records(derived_df.head(normalized_preview_rows)),
            "guidance": "Future framework dataset tools default to this derived dataset when dataset_name is omitted.",
        }

    def _normalize_features(self, features: list[dict[str, Any]] | None) -> list[dict[str, Any]] | dict[str, Any]:
        if not isinstance(features, list) or not features:
            return self._error("At least one feature specification is required.")
        if len(features) > MAX_FEATURES_PER_CALL:
            return self._error(f"At most {MAX_FEATURES_PER_CALL} features can be derived in one call.")

        normalized: list[dict[str, Any]] = []
        for index, feature in enumerate(features, start=1):
            if not isinstance(feature, dict):
                return self._error(f"Feature specification {index} must be an object.")

            name = str(feature.get("name") or "").strip()
            operation = str(feature.get("operation") or "").strip().lower()
            source_columns_raw = feature.get("source_columns")
            source_columns = [
                str(column).strip()
                for column in source_columns_raw
                if str(column).strip()
            ] if isinstance(source_columns_raw, list) else []
            parameters = feature.get("parameters")
            normalized.append({
                "name": name,
                "operation": operation,
                "source_columns": source_columns,
                "parameters": deepcopy(parameters) if isinstance(parameters, dict) else {},
            })

        return normalized

    def _validate_feature_collection(
        self,
        source_df: pd.DataFrame,
        features: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        existing_columns = {str(column) for column in source_df.columns}
        available_columns = set(existing_columns)
        feature_names: set[str] = set()

        for feature in features:
            name = str(feature["name"])
            operation = str(feature["operation"])
            source_columns = list(feature["source_columns"])

            if not name:
                return self._error("Each feature must include a non-empty name.")
            if name in existing_columns:
                return self._error(f"Feature name '{name}' already exists in the source dataset.")
            if name in feature_names:
                return self._error(f"Feature name '{name}' is duplicated in this request.")

            unsupported = operation not in (
                SUPPORTED_BINARY_OPERATIONS
                | SUPPORTED_UNARY_OPERATIONS
                | SUPPORTED_FLAG_OPERATIONS
                | {"datetime_part"}
            )
            if unsupported:
                return self._error(f"Unsupported feature operation '{operation}'.")

            missing_columns = [column for column in source_columns if column not in available_columns]
            if missing_columns:
                return self._error(
                    "One or more source columns were not found in the dataset.",
                    errors=[{"field": "source_columns", "reason": ", ".join(missing_columns)}],
                )

            expected_count = 2 if operation in SUPPORTED_BINARY_OPERATIONS else 1
            if len(source_columns) != expected_count:
                return self._error(
                    f"Operation '{operation}' requires exactly {expected_count} source column"
                    f"{'s' if expected_count != 1 else ''}."
                )

            if operation in {"flag_gt", "flag_gte", "flag_lt", "flag_lte"}:
                if not self._is_number(feature["parameters"].get("threshold")):
                    return self._error(f"Operation '{operation}' requires numeric parameters.threshold.")
            if operation == "flag_eq" and "compare_value" not in feature["parameters"]:
                return self._error("Operation 'flag_eq' requires parameters.compare_value.")
            if operation == "datetime_part":
                part = str(feature["parameters"].get("part") or "").strip().lower()
                if part not in SUPPORTED_DATETIME_PARTS:
                    return self._error(
                        "Operation 'datetime_part' requires parameters.part to be one of: "
                        + ", ".join(sorted(SUPPORTED_DATETIME_PARTS))
                        + "."
                    )

            feature_names.add(name)
            available_columns.add(name)

        return None

    def _apply_features(
        self,
        dataframe: pd.DataFrame,
        features: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, list[str]]:
        df = dataframe.copy(deep=True)
        warnings: list[str] = []
        for feature in features:
            name = str(feature["name"])
            operation = str(feature["operation"])
            source_columns = list(feature["source_columns"])
            parameters = dict(feature["parameters"])

            if operation in SUPPORTED_BINARY_OPERATIONS:
                left = self._numeric_series(df[source_columns[0]], source_columns[0], warnings)
                right = self._numeric_series(df[source_columns[1]], source_columns[1], warnings)
                if operation == "add":
                    df[name] = left + right
                elif operation == "subtract":
                    df[name] = left - right
                elif operation == "multiply":
                    df[name] = left * right
                else:
                    zero_count = int((right == 0).sum())
                    if zero_count > 0:
                        warnings.append(f"{name}: {zero_count} row(s) have a zero denominator and return null.")
                    df[name] = left / right.mask(right == 0)
                continue

            if operation in SUPPORTED_UNARY_OPERATIONS:
                value = self._numeric_series(df[source_columns[0]], source_columns[0], warnings)
                if operation == "log":
                    invalid_count = int((value <= 0).sum())
                    if invalid_count > 0:
                        warnings.append(f"{name}: {invalid_count} row(s) are non-positive and return null.")
                    df[name] = value.where(value > 0).map(lambda item: math.log(item) if pd.notna(item) else pd.NA)
                elif operation == "log1p":
                    invalid_count = int((value <= -1).sum())
                    if invalid_count > 0:
                        warnings.append(f"{name}: {invalid_count} row(s) are <= -1 and return null.")
                    df[name] = value.where(value > -1).map(lambda item: math.log1p(item) if pd.notna(item) else pd.NA)
                elif operation == "square":
                    df[name] = value**2
                else:
                    df[name] = value**3
                continue

            if operation in SUPPORTED_FLAG_OPERATIONS:
                source = df[source_columns[0]]
                if operation == "flag_is_missing":
                    df[name] = source.isna().astype("Int64")
                elif operation == "flag_eq":
                    df[name] = (source == parameters.get("compare_value")).astype("Int64")
                else:
                    value = self._numeric_series(source, source_columns[0], warnings)
                    threshold = float(parameters.get("threshold"))
                    if operation == "flag_gt":
                        df[name] = (value > threshold).astype("Int64")
                    elif operation == "flag_gte":
                        df[name] = (value >= threshold).astype("Int64")
                    elif operation == "flag_lt":
                        df[name] = (value < threshold).astype("Int64")
                    else:
                        df[name] = (value <= threshold).astype("Int64")
                continue

            if operation == "datetime_part":
                source_name = source_columns[0]
                parsed = pd.to_datetime(df[source_name], errors="coerce")
                failed_count = int(df[source_name].notna().sum() - parsed.notna().sum())
                if failed_count > 0:
                    warnings.append(f"{name}: {failed_count} non-null row(s) could not be parsed as datetime.")
                part = str(parameters.get("part") or "").strip().lower()
                if part == "year":
                    df[name] = parsed.dt.year.astype("Int64")
                elif part == "quarter":
                    df[name] = parsed.dt.quarter.astype("Int64")
                elif part == "month":
                    df[name] = parsed.dt.month.astype("Int64")
                elif part == "week":
                    df[name] = parsed.dt.isocalendar().week.astype("Int64")
                elif part == "day":
                    df[name] = parsed.dt.day.astype("Int64")
                elif part == "dayofweek":
                    df[name] = parsed.dt.dayofweek.astype("Int64")
                else:
                    df[name] = parsed.dt.hour.astype("Int64")

        return df, warnings

    def _numeric_series(self, series: pd.Series, column_name: str, warnings: list[str]) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        failed_count = int(series.notna().sum() - numeric.notna().sum())
        if failed_count > 0:
            warnings.append(f"Column '{column_name}' had {failed_count} non-null value(s) coerced to null for numeric operations.")
        return numeric

    def _resolve_source_dataset(self, dataset_name: str | None) -> dict[str, Any]:
        loaded_context = self.data_context_service.get_loaded_data_context()
        datasets = loaded_context.get("datasets", [])
        available_names = [
            str(dataset.get("name"))
            for dataset in datasets
            if isinstance(dataset, dict) and dataset.get("name")
        ]
        target_name = str(dataset_name).strip() if dataset_name not in (None, "") else str(loaded_context.get("active_dataset_name") or "")
        if not target_name:
            return self._error("No active dataset is available.")
        if available_names and target_name not in available_names:
            return self._error(
                "Requested dataset was not found in the current app context.",
                errors=[{"field": "dataset_name", "reason": f"Available datasets: {', '.join(available_names)}"}],
            )

        derived_dataset = get_derived_dataset_object(target_name)
        if derived_dataset is not None:
            return {
                "status": "ok",
                "dataset_name": target_name,
                "dataframe": derived_dataset,
                "is_derived": True,
            }

        app = self._resolve_active_workspace_app()
        dataset_object = self._resolve_app_dataset_object(app=app, dataset_name=target_name)
        if not isinstance(dataset_object, pd.DataFrame):
            return self._error("The requested dataset is not available as a pandas DataFrame.")
        return {
            "status": "ok",
            "dataset_name": target_name,
            "dataframe": dataset_object,
            "is_derived": False,
        }

    def _build_output_dataset_name(
        self,
        *,
        source_dataset_name: str,
        output_dataset_name: str | None,
        source_is_derived: bool,
    ) -> str:
        requested_name = str(output_dataset_name or "").strip()
        if requested_name:
            return requested_name
        if source_is_derived:
            return source_dataset_name
        return f"{source_dataset_name}__derived"

    def _estimate_materialized_memory_bytes(self, *, source_df: pd.DataFrame, feature_count: int) -> int:
        try:
            source_memory = int(source_df.memory_usage(deep=True).sum())
        except Exception:
            source_memory = int(source_df.shape[0] * max(1, source_df.shape[1]) * 8)
        feature_memory = int(source_df.shape[0] * max(1, feature_count) * 8)
        return source_memory + feature_memory

    def _resolve_active_workspace_app(self) -> Any | None:
        for module_name in (
            "app.components.workspace_host",
            "app.components.workspace",
            "app.pages.workspace",
            "app.workspace",
        ):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            getter = getattr(module, "get_active_workspace_app", None)
            if callable(getter):
                try:
                    app = getter()
                except Exception:
                    app = None
                if app is not None:
                    return app

        for state_key in (
            "active_workspace_app",
            "workspace_active_app",
            "mounted_workspace_app",
            "current_workspace_app",
        ):
            app = st.session_state.get(state_key)
            if app is not None:
                return app
        return None

    def _resolve_app_dataset_object(self, *, app: Any, dataset_name: str) -> Any | None:
        getter = getattr(app, "get_dataset_object", None)
        if not callable(getter):
            return None
        try:
            return getter(dataset_name=dataset_name)
        except TypeError:
            try:
                return getter(dataset_name)
            except TypeError:
                try:
                    return getter()
                except Exception:
                    return None
        except Exception:
            return None

    def _serialize_records(self, dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for record in dataframe.to_dict(orient="records"):
            serialized: dict[str, Any] = {}
            for key, value in record.items():
                if pd.isna(value):
                    serialized[str(key)] = None
                elif hasattr(value, "isoformat"):
                    try:
                        serialized[str(key)] = value.isoformat()
                    except Exception:
                        serialized[str(key)] = str(value)
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    serialized[str(key)] = value
                else:
                    serialized[str(key)] = str(value)
            records.append(serialized)
        return records

    def _error(self, message: str, *, errors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "error",
            "message": message,
        }
        if errors:
            payload["errors"] = errors
        return payload

    def _is_number(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True
