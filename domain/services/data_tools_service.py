from __future__ import annotations

import importlib
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import streamlit as st

from domain.services.data_context_service import DataContextService
from domain.services.derived_dataset_store import get_derived_dataset_object


class DataToolsService:
    """
    Deterministic service for dataset-level read-only data utilities.

    This service is intentionally deeper than `DataContextService`.
    - `DataContextService` answers what datasets are available.
    - `DataToolsService` answers what is analytically important about one
      specific dataset.
    """

    def __init__(self, data_context_service: DataContextService | None = None) -> None:
        self.data_context_service = data_context_service or DataContextService()

    def get_dataset_profile(
        self,
        dataset_name: str | None = None,
        focus_column: str | None = None,
    ) -> dict[str, Any]:
        """
        Return a structured analysis-oriented profile for one dataset.
        """
        loaded_data_context = self.data_context_service.get_loaded_data_context()
        datasets = loaded_data_context.get("datasets", [])

        if not loaded_data_context.get("has_data") or not isinstance(datasets, list) or not datasets:
            return {
                "status": "error",
                "message": "No structured datasets are currently available.",
                "dataset_name": dataset_name,
            }

        active_dataset_name = loaded_data_context.get("active_dataset_name")
        target_dataset_name = dataset_name or active_dataset_name

        available_dataset_names = {
            str(dataset.get("name"))
            for dataset in datasets
            if isinstance(dataset, dict) and dataset.get("name")
        }

        if not target_dataset_name:
            return {
                "status": "error",
                "message": "No active dataset is available to profile.",
                "dataset_name": None,
                "available_datasets": sorted(available_dataset_names),
            }

        if target_dataset_name not in available_dataset_names:
            return {
                "status": "error",
                "message": "Requested dataset was not found in the current app context.",
                "dataset_name": target_dataset_name,
                "available_datasets": sorted(available_dataset_names),
            }

        app = self._resolve_active_workspace_app()
        dataset_object = self._resolve_dataset_object(app=app, dataset_name=target_dataset_name)
        if not isinstance(dataset_object, pd.DataFrame):
            return {
                "status": "error",
                "message": "The requested dataset is not available as a pandas DataFrame.",
                "dataset_name": target_dataset_name,
            }

        if focus_column is not None and str(focus_column) not in {str(column) for column in dataset_object.columns}:
            return {
                "status": "error",
                "message": "Requested focus column was not found in the dataset.",
                "dataset_name": target_dataset_name,
                "focus_column": focus_column,
                "available_columns": [str(column) for column in dataset_object.columns],
            }

        df = dataset_object
        selected_columns = [str(focus_column)] if focus_column is not None else [str(column) for column in df.columns]
        column_profiles = self._build_column_profiles(df, selected_columns=selected_columns)
        numeric_summary = self._build_numeric_summary(df, selected_columns=selected_columns)
        duplicate_row_count = int(df.duplicated().sum())
        constant_columns = [
            str(column)
            for column in selected_columns
            if int(df[column].nunique(dropna=False)) <= 1
        ]
        rows_with_any_missing = int(df[selected_columns].isna().any(axis=1).sum()) if selected_columns else 0
        low_cardinality_distributions = self._build_low_cardinality_distributions(
            df,
            selected_columns=selected_columns,
        )

        warnings: list[str] = []
        if duplicate_row_count > 0:
            warnings.append(f"Dataset contains {duplicate_row_count} duplicate row(s).")
        if constant_columns:
            warnings.append(
                "Constant or single-valued columns detected: " + ", ".join(constant_columns)
            )
        if rows_with_any_missing > 0:
            warnings.append(f"{rows_with_any_missing} row(s) contain at least one missing value across the profiled columns.")
        if focus_column is not None and low_cardinality_distributions.get(str(focus_column)):
            warnings.append(f"Low-cardinality value distribution included for focus column '{focus_column}'.")

        focus_profile = column_profiles.get(str(focus_column)) if focus_column is not None else None
        datetime_profile = focus_profile.get("datetime_profile") if isinstance(focus_profile, dict) else None
        if isinstance(datetime_profile, dict):
            warnings.append(
                f"Focus column '{focus_column}' was detected as {datetime_profile.get('detected_as')} with a parsed date range included."
            )

        return {
            "status": "ok",
            "dataset_name": target_dataset_name,
            "focus_column": str(focus_column) if focus_column is not None else None,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "profiled_column_count": len(selected_columns),
            "duplicate_row_count": duplicate_row_count,
            "rows_with_any_missing": rows_with_any_missing,
            "constant_columns": constant_columns,
            "column_profiles": column_profiles,
            "numeric_summary": numeric_summary,
            "low_cardinality_distributions": low_cardinality_distributions,
            "warnings": warnings,
        }

    def get_dataset_sample(
        self,
        dataset_name: str | None = None,
        sample_type: str = "head",
        row_count: int = 10,
    ) -> dict[str, Any]:
        """
        Return a compact but meaningful sample for one dataset.
        """
        loaded_data_context = self.data_context_service.get_loaded_data_context()
        datasets = loaded_data_context.get("datasets", [])

        if not loaded_data_context.get("has_data") or not isinstance(datasets, list) or not datasets:
            return {
                "status": "error",
                "message": "No structured datasets are currently available.",
                "dataset_name": dataset_name,
            }

        active_dataset_name = loaded_data_context.get("active_dataset_name")
        target_dataset_name = dataset_name or active_dataset_name

        available_dataset_names = {
            str(dataset.get("name"))
            for dataset in datasets
            if isinstance(dataset, dict) and dataset.get("name")
        }

        if not target_dataset_name:
            return {
                "status": "error",
                "message": "No active dataset is available to sample.",
                "dataset_name": None,
                "available_datasets": sorted(available_dataset_names),
            }

        if target_dataset_name not in available_dataset_names:
            return {
                "status": "error",
                "message": "Requested dataset was not found in the current app context.",
                "dataset_name": target_dataset_name,
                "available_datasets": sorted(available_dataset_names),
            }

        app = self._resolve_active_workspace_app()
        dataset_object = self._resolve_dataset_object(app=app, dataset_name=target_dataset_name)
        if not isinstance(dataset_object, pd.DataFrame):
            return {
                "status": "error",
                "message": "The requested dataset is not available as a pandas DataFrame.",
                "dataset_name": target_dataset_name,
            }

        df = dataset_object
        normalized_sample_type = str(sample_type).strip().lower() if sample_type else "head"
        if normalized_sample_type not in {"head", "random"}:
            normalized_sample_type = "head"

        capped_row_count = max(1, min(int(row_count), 12))
        effective_row_count = min(capped_row_count, int(df.shape[0]))

        if normalized_sample_type == "random":
            sample_df = df.sample(n=effective_row_count, random_state=42) if effective_row_count > 0 else df.head(0)
        else:
            sample_df = df.head(effective_row_count)

        sample_records = self._serialize_sample_records(sample_df)

        return {
            "status": "ok",
            "dataset_name": target_dataset_name,
            "sample_type": normalized_sample_type,
            "requested_row_count": int(row_count),
            "returned_row_count": len(sample_records),
            "total_rows": int(df.shape[0]),
            "columns": [str(column) for column in df.columns],
            "rows": sample_records,
        }

    def get_dataset_aggregation(
        self,
        dataset_name: str | None = None,
        filters: list[dict[str, Any]] | None = None,
        group_by: list[str] | None = None,
        time_bucket: dict[str, Any] | None = None,
        metrics: list[dict[str, Any]] | None = None,
        sort_by: str | None = None,
        sort_direction: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Return a compact grouped/filtered aggregation for one dataset.
        """
        loaded_data_context = self.data_context_service.get_loaded_data_context()
        datasets = loaded_data_context.get("datasets", [])

        if not loaded_data_context.get("has_data") or not isinstance(datasets, list) or not datasets:
            return {
                "status": "error",
                "message": "No structured datasets are currently available.",
                "dataset_name": dataset_name,
            }

        active_dataset_name = loaded_data_context.get("active_dataset_name")
        target_dataset_name = dataset_name or active_dataset_name

        available_dataset_names = {
            str(dataset.get("name"))
            for dataset in datasets
            if isinstance(dataset, dict) and dataset.get("name")
        }

        if not target_dataset_name:
            return {
                "status": "error",
                "message": "No active dataset is available to aggregate.",
                "dataset_name": None,
                "available_datasets": sorted(available_dataset_names),
            }

        if target_dataset_name not in available_dataset_names:
            return {
                "status": "error",
                "message": "Requested dataset was not found in the current app context.",
                "dataset_name": target_dataset_name,
                "available_datasets": sorted(available_dataset_names),
            }

        if not metrics:
            return {
                "status": "error",
                "message": "At least one aggregation metric is required.",
                "dataset_name": target_dataset_name,
            }

        app = self._resolve_active_workspace_app()
        dataset_object = self._resolve_dataset_object(app=app, dataset_name=target_dataset_name)
        if not isinstance(dataset_object, pd.DataFrame):
            return {
                "status": "error",
                "message": "The requested dataset is not available as a pandas DataFrame.",
                "dataset_name": target_dataset_name,
            }

        df = dataset_object.copy()
        warnings: list[str] = []
        normalized_group_by = [str(column) for column in (group_by or []) if str(column).strip()]

        if len(normalized_group_by) > 2:
            return {
                "status": "error",
                "message": "At most 2 group_by columns are supported.",
                "dataset_name": target_dataset_name,
            }

        if len(metrics) > 3:
            return {
                "status": "error",
                "message": "At most 3 aggregation metrics are supported.",
                "dataset_name": target_dataset_name,
            }

        missing_group_columns = [column for column in normalized_group_by if column not in {str(col) for col in df.columns}]
        if missing_group_columns:
            return {
                "status": "error",
                "message": "One or more group_by columns were not found in the dataset.",
                "dataset_name": target_dataset_name,
                "missing_columns": missing_group_columns,
                "available_columns": [str(column) for column in df.columns],
            }

        try:
            df = self._apply_filters(df, filters=filters)
        except ValueError as exc:
            return {
                "status": "error",
                "message": str(exc),
                "dataset_name": target_dataset_name,
            }

        row_count_after_filters = int(df.shape[0])

        derived_time_bucket_label: str | None = None
        if time_bucket:
            try:
                df, derived_time_bucket_label, time_bucket_warning = self._apply_time_bucket(df, time_bucket=time_bucket)
            except ValueError as exc:
                return {
                    "status": "error",
                    "message": str(exc),
                    "dataset_name": target_dataset_name,
                }

            if time_bucket_warning:
                warnings.append(time_bucket_warning)
            if derived_time_bucket_label:
                normalized_group_by = normalized_group_by + [derived_time_bucket_label]

        try:
            named_aggregations = self._build_named_aggregations(df, metrics=metrics)
        except ValueError as exc:
            return {
                "status": "error",
                "message": str(exc),
                "dataset_name": target_dataset_name,
            }

        if normalized_group_by:
            aggregated_df = (
                df.groupby(normalized_group_by, dropna=False)
                .agg(**named_aggregations)
                .reset_index()
            )
        else:
            aggregated_row: dict[str, Any] = {}
            for output_name, aggregation_spec in named_aggregations.items():
                target_column, operation = aggregation_spec
                if operation == "size":
                    aggregated_row[output_name] = int(df.shape[0])
                elif operation == "nunique":
                    aggregated_row[output_name] = int(df[target_column].nunique(dropna=True))
                else:
                    aggregated_row[output_name] = df[target_column].agg(operation)

            aggregated_df = pd.DataFrame([aggregated_row])

        normalized_sort_direction = str(sort_direction or "asc").strip().lower()
        if normalized_sort_direction not in {"asc", "desc"}:
            normalized_sort_direction = "asc"

        if sort_by:
            sort_key = str(sort_by)
            if sort_key not in aggregated_df.columns:
                return {
                    "status": "error",
                    "message": "Requested sort_by column was not found in the aggregation output.",
                    "dataset_name": target_dataset_name,
                    "available_sort_columns": [str(column) for column in aggregated_df.columns],
                }
            aggregated_df = aggregated_df.sort_values(by=sort_key, ascending=(normalized_sort_direction == "asc"))
        elif derived_time_bucket_label and derived_time_bucket_label in aggregated_df.columns:
            aggregated_df = aggregated_df.sort_values(by=derived_time_bucket_label, ascending=True)

        normalized_limit = 20 if limit is None else int(limit)
        normalized_limit = max(1, min(normalized_limit, 100))
        result_df = aggregated_df.head(normalized_limit).copy()

        return {
            "status": "ok",
            "dataset_name": target_dataset_name,
            "filters_applied": filters or [],
            "group_by": normalized_group_by,
            "time_bucket": time_bucket or None,
            "metrics": metrics,
            "sort_by": str(sort_by) if sort_by is not None else None,
            "sort_direction": normalized_sort_direction if sort_by is not None else None,
            "row_count_after_filters": row_count_after_filters,
            "result_row_count": int(result_df.shape[0]),
            "total_result_row_count": int(aggregated_df.shape[0]),
            "rows": self._serialize_sample_records(result_df),
            "warnings": warnings,
        }

    def _apply_filters(
        self,
        df: pd.DataFrame,
        *,
        filters: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """
        Apply a compact AND-combined filter list to a dataframe.
        """
        if not filters:
            return df

        filtered_df = df
        available_columns = {str(column) for column in df.columns}

        for filter_spec in filters:
            if not isinstance(filter_spec, dict):
                raise ValueError("Each filter must be an object with column, operator, and optional value.")

            column = str(filter_spec.get("column") or "").strip()
            operator = str(filter_spec.get("operator") or "").strip()
            value = filter_spec.get("value")

            if not column:
                raise ValueError("Each filter must include a column name.")
            if column not in available_columns:
                raise ValueError(f"Filter column '{column}' was not found in the dataset.")
            if not operator:
                raise ValueError(f"Filter for column '{column}' is missing an operator.")

            series = filtered_df[column]

            if operator == "==":
                mask = series == value
            elif operator == "!=":
                mask = series != value
            elif operator == ">":
                mask = series > value
            elif operator == ">=":
                mask = series >= value
            elif operator == "<":
                mask = series < value
            elif operator == "<=":
                mask = series <= value
            elif operator == "in":
                if not isinstance(value, list):
                    raise ValueError(f"Filter operator 'in' for column '{column}' requires a list value.")
                mask = series.isin(value)
            elif operator == "not_in":
                if not isinstance(value, list):
                    raise ValueError(f"Filter operator 'not_in' for column '{column}' requires a list value.")
                mask = ~series.isin(value)
            elif operator == "contains":
                mask = series.astype(str).str.contains(str(value), case=False, na=False)
            elif operator == "not_contains":
                mask = ~series.astype(str).str.contains(str(value), case=False, na=False)
            elif operator == "is_null":
                mask = series.isna()
            elif operator == "is_not_null":
                mask = series.notna()
            else:
                raise ValueError(f"Unsupported filter operator '{operator}' for column '{column}'.")

            filtered_df = filtered_df.loc[mask].copy()

        return filtered_df

    def _apply_time_bucket(
        self,
        df: pd.DataFrame,
        *,
        time_bucket: dict[str, Any],
    ) -> tuple[pd.DataFrame, str, str | None]:
        """
        Parse a date/datetime column and add a compact derived time bucket.
        """
        if not isinstance(time_bucket, dict):
            raise ValueError("time_bucket must be an object with column and unit.")

        source_column = str(time_bucket.get("column") or "").strip()
        unit = str(time_bucket.get("unit") or "").strip().lower()
        label = str(time_bucket.get("label") or "").strip() or f"{source_column}_{unit}"

        if not source_column:
            raise ValueError("time_bucket.column is required.")
        if source_column not in {str(column) for column in df.columns}:
            raise ValueError(f"time_bucket column '{source_column}' was not found in the dataset.")
        if unit not in {"year", "quarter", "month", "week", "day", "hour"}:
            raise ValueError("time_bucket.unit must be one of: year, quarter, month, week, day, hour.")

        parsed = pd.to_datetime(df[source_column], errors="coerce", utc=True)
        non_null_count = int(df[source_column].notna().sum())
        parsed_count = int(parsed.notna().sum())
        parse_success_rate = (parsed_count / non_null_count) if non_null_count > 0 else 0.0

        if parsed_count == 0 or parse_success_rate < 0.8:
            raise ValueError(
                f"time_bucket column '{source_column}' could not be reliably parsed as datetime values."
            )

        parsed_naive = parsed.dt.tz_convert(None)

        if unit == "year":
            bucket_series = parsed_naive.dt.strftime("%Y")
        elif unit == "quarter":
            bucket_series = parsed_naive.dt.to_period("Q").astype(str)
        elif unit == "month":
            bucket_series = parsed_naive.dt.strftime("%Y-%m")
        elif unit == "week":
            bucket_series = parsed_naive.dt.to_period("W").apply(lambda value: value.start_time.strftime("%Y-%m-%d"))
        elif unit == "day":
            bucket_series = parsed_naive.dt.strftime("%Y-%m-%d")
        else:
            bucket_series = parsed_naive.dt.strftime("%Y-%m-%d %H:00")

        derived_df = df.copy()
        derived_df[label] = bucket_series.where(parsed.notna(), None)

        warning = None
        if parse_success_rate < 1.0:
            warning = (
                f"time_bucket column '{source_column}' was partially parsed as datetime values "
                f"(parse success rate: {round(parse_success_rate, 4)})."
            )

        return derived_df, label, warning

    def _build_named_aggregations(
        self,
        df: pd.DataFrame,
        *,
        metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Build pandas named aggregations from a compact metric specification.
        """
        named_aggregations: dict[str, Any] = {}
        available_columns = {str(column) for column in df.columns}
        supported_operations = {"count", "sum", "mean", "median", "min", "max", "nunique"}

        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                raise ValueError("Each metric must be an object with operation and optional target_column.")

            operation = str(metric.get("operation") or "").strip().lower()
            target_column_raw = metric.get("target_column")
            target_column = str(target_column_raw).strip() if target_column_raw is not None else None
            alias_raw = metric.get("alias")
            alias = str(alias_raw).strip() if alias_raw is not None and str(alias_raw).strip() else None

            if operation not in supported_operations:
                raise ValueError(
                    "Unsupported metric operation. Supported operations are: count, sum, mean, median, min, max, nunique."
                )

            if operation == "count":
                output_name = alias or (f"count_{target_column}" if target_column else "count")
                if target_column:
                    if target_column not in available_columns:
                        raise ValueError(f"Metric target_column '{target_column}' was not found in the dataset.")
                    named_aggregations[output_name] = (target_column, "count")
                else:
                    named_aggregations[output_name] = (df.columns[0], "size")
                continue

            if not target_column:
                raise ValueError(f"Metric operation '{operation}' requires a target_column.")
            if target_column not in available_columns:
                raise ValueError(f"Metric target_column '{target_column}' was not found in the dataset.")

            output_name = alias or f"{operation}_{target_column}"
            named_aggregations[output_name] = (target_column, operation)

        if len(named_aggregations) != len(metrics):
            raise ValueError("Metric output names must be unique.")

        return named_aggregations

    def _resolve_active_workspace_app(self) -> Any | None:
        """
        Resolve the active mounted workspace app using the shell's canonical
        helper when possible, with safe fallbacks for older wiring paths.
        """
        candidate_modules = [
            "app.components.workspace_host",
            "app.components.workspace",
            "app.pages.workspace",
            "app.workspace",
        ]

        for module_name in candidate_modules:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue

            get_active_workspace_app = getattr(module, "get_active_workspace_app", None)
            if not callable(get_active_workspace_app):
                continue

            try:
                app = get_active_workspace_app()
            except Exception:
                continue

            if app is not None:
                return app

        candidate_keys = [
            "active_workspace_app",
            "workspace_active_app",
            "mounted_workspace_app",
            "current_workspace_app",
        ]

        for key in candidate_keys:
            app = st.session_state.get(key)
            if app is not None:
                return app

        return None

    def _resolve_dataset_object(self, *, app: Any, dataset_name: str | None) -> Any | None:
        """
        Ask the active app for the actual dataset object if it supports the
        standardized dataset-object accessor.
        """
        derived_dataset = get_derived_dataset_object(dataset_name)
        if derived_dataset is not None:
            return derived_dataset

        get_dataset_object = getattr(app, "get_dataset_object", None)
        if not callable(get_dataset_object):
            return None

        try:
            return get_dataset_object(dataset_name=dataset_name)
        except TypeError:
            try:
                return get_dataset_object(dataset_name)
            except TypeError:
                try:
                    return get_dataset_object()
                except Exception:
                    return None
        except Exception:
            return None

    def _build_column_profiles(
        self,
        df: pd.DataFrame,
        *,
        selected_columns: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Build per-column completeness, cardinality, and lightweight type-aware
        profile information.
        """
        total_rows = int(df.shape[0])
        profiles: dict[str, dict[str, Any]] = {}

        columns_to_profile = selected_columns or [str(column) for column in df.columns]

        for column in columns_to_profile:
            series = df[column]
            missing_count = int(series.isna().sum())
            non_null_count = int(total_rows - missing_count)
            distinct_count = int(series.nunique(dropna=True))

            profile: dict[str, Any] = {
                "dtype": str(series.dtype),
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_pct": round((missing_count / total_rows) * 100, 4) if total_rows > 0 else 0.0,
                "distinct_count": distinct_count,
            }

            datetime_profile = self._build_datetime_profile(series)
            if datetime_profile is not None:
                profile["datetime_profile"] = datetime_profile

            profiles[str(column)] = profile

        return profiles
    
    def _build_datetime_profile(self, series: pd.Series) -> dict[str, Any] | None:
        """
        Detect true datetime columns and object/string columns that are mostly
        parseable as datetimes, then return a compact date-range profile.
        """
        non_null_series = series.dropna()
        if non_null_series.empty:
            return None

        parsed_series: pd.Series | None = None
        detected_as = ""

        if is_datetime64_any_dtype(series):
            parsed_series = pd.to_datetime(non_null_series, errors="coerce")
            detected_as = "datetime"
        elif series.dtype == object or pd.api.types.is_string_dtype(series):
            parsed_candidate = pd.to_datetime(non_null_series, errors="coerce")
            parsed_non_null = parsed_candidate.notna().sum()
            parse_success_rate = parsed_non_null / len(non_null_series) if len(non_null_series) > 0 else 0.0

            if parsed_non_null > 0 and parse_success_rate >= 0.8:
                parsed_series = parsed_candidate
                detected_as = "datetime_like"
        else:
            return None

        if parsed_series is None:
            return None

        parsed_series = parsed_series.dropna()
        if parsed_series.empty:
            return None

        min_value = parsed_series.min()
        max_value = parsed_series.max()

        return {
            "detected_as": detected_as,
            "min": min_value.isoformat() if hasattr(min_value, "isoformat") else str(min_value),
            "max": max_value.isoformat() if hasattr(max_value, "isoformat") else str(max_value),
            "parseable_non_null_count": int(parsed_series.shape[0]),
            "parse_success_rate": round(parsed_series.shape[0] / len(non_null_series), 4) if len(non_null_series) > 0 else 0.0,
        }

    def _build_numeric_summary(
        self,
        df: pd.DataFrame,
        *,
        selected_columns: list[str] | None = None,
    ) -> dict[str, dict[str, float | int | None]]:
        """
        Build numeric summary statistics for numeric columns only.
        """
        if selected_columns is not None:
            numeric_df = df[selected_columns].select_dtypes(include="number")
        else:
            numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return {}

        numeric_summary: dict[str, dict[str, float | int | None]] = {}
        describe = numeric_df.describe().transpose()

        for column in describe.index:
            row = describe.loc[column]
            numeric_summary[str(column)] = {
                "count": float(row.get("count")) if pd.notna(row.get("count")) else None,
                "mean": float(row.get("mean")) if pd.notna(row.get("mean")) else None,
                "std": float(row.get("std")) if pd.notna(row.get("std")) else None,
                "min": float(row.get("min")) if pd.notna(row.get("min")) else None,
                "p25": float(row.get("25%")) if pd.notna(row.get("25%")) else None,
                "median": float(row.get("50%")) if pd.notna(row.get("50%")) else None,
                "p75": float(row.get("75%")) if pd.notna(row.get("75%")) else None,
                "max": float(row.get("max")) if pd.notna(row.get("max")) else None,
            }

        return numeric_summary

    def _build_low_cardinality_distributions(
        self,
        df: pd.DataFrame,
        *,
        selected_columns: list[str] | None = None,
        max_distinct_values: int = 10,
        max_returned_values: int = 10,
    ) -> dict[str, dict[str, Any]]:
        """
        Build compact value distributions for low-cardinality columns only.
        """
        total_rows = int(df.shape[0])
        columns_to_profile = selected_columns or [str(column) for column in df.columns]
        distributions: dict[str, dict[str, Any]] = {}

        for column in columns_to_profile:
            distinct_count = int(df[column].nunique(dropna=False))
            if distinct_count > max_distinct_values:
                continue

            value_counts = df[column].value_counts(dropna=False).head(max_returned_values)
            top_values: list[dict[str, Any]] = []
            for value, count in value_counts.items():
                if pd.isna(value):
                    serialized_value: Any = None
                elif hasattr(value, "isoformat"):
                    try:
                        serialized_value = value.isoformat()
                    except Exception:
                        serialized_value = str(value)
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    serialized_value = value
                else:
                    serialized_value = str(value)

                top_values.append(
                    {
                        "value": serialized_value,
                        "count": int(count),
                        "pct": round((int(count) / total_rows) * 100, 4) if total_rows > 0 else 0.0,
                    }
                )

            distributions[str(column)] = {
                "distinct_count": distinct_count,
                "top_values": top_values,
            }

        return distributions

    def _serialize_sample_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Convert a sampled dataframe into JSON-friendly row records.
        """
        if df.empty:
            return []

        serialized_records: list[dict[str, Any]] = []
        records = df.to_dict(orient="records")

        for record in records:
            serialized_row: dict[str, Any] = {}
            for key, value in record.items():
                if pd.isna(value):
                    serialized_row[str(key)] = None
                elif hasattr(value, "isoformat"):
                    try:
                        serialized_row[str(key)] = value.isoformat()
                    except Exception:
                        serialized_row[str(key)] = str(value)
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    serialized_row[str(key)] = value
                else:
                    serialized_row[str(key)] = str(value)
            serialized_records.append(serialized_row)

        return serialized_records
