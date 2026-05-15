

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import importlib
import numpy as np
import pandas as pd
import streamlit as st


from domain.services.data_context_service import DataContextService
from domain.services.derived_dataset_store import get_derived_dataset_object

_DEFAULT_HISTOGRAM_BINS = 12
_MAX_HISTOGRAM_BINS = 20

_MAX_BAR_CHART_BARS = 20
_MAX_SCATTER_PLOT_POINTS = 1000


@dataclass
class _DatasetHandle:
    name: str
    dataframe: pd.DataFrame


class VisualizationToolsService:
    def __init__(self, data_context_service: DataContextService | None = None) -> None:
        self.data_context_service = data_context_service or DataContextService()
    """
    Service layer for standardized dataset visualizations used by agent tools.

    This service prepares chart-ready data and a lightweight rendering payload for
    the chat widget. It does not render charts directly; instead, it returns a
    structured, deterministic payload that can be displayed inline by the UI and
    reasoned over by the agent in the same turn.
    """

    def get_dataset_chart(
        self,
        *,
        dataset_name: str | None,
        chart_type: str,
        x_column: str,
        y_column: str | None = None,
        aggregation: str | None = None,
        filters: list[dict[str, Any]] | None = None,
        time_bucket: str | None = None,
        bins: int | None = None,
        limit: int | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        try:
            handle = self._resolve_dataset_handle(dataset_name)
            df = handle.dataframe.copy()

            chart_type = str(chart_type or "").strip().lower()
            x_column = str(x_column or "").strip()
            y_column = str(y_column).strip() if y_column is not None else None
            aggregation = str(aggregation).strip().lower() if aggregation else None
            time_bucket = str(time_bucket).strip().lower() if time_bucket else None
            limit = int(limit) if limit is not None else None
            bins = int(bins) if bins is not None else None
            effective_filters = filters or []

            if chart_type not in {"bar", "line", "scatter", "histogram"}:
                return self._error(
                    dataset_name=handle.name,
                    message="Unsupported chart_type. Supported chart types are bar, line, scatter, and histogram.",
                )

            if not x_column:
                return self._error(
                    dataset_name=handle.name,
                    message="x_column is required.",
                )

            self._require_column(df, x_column)
            if y_column:
                self._require_column(df, y_column)

            filtered_df = self._apply_filters(df, effective_filters)
            source_rows = int(len(filtered_df))
            if source_rows == 0:
                return self._error(
                    dataset_name=handle.name,
                    message="No rows remain after applying filters.",
                )

            prepared_x_column = x_column
            if chart_type == "line" and time_bucket is None:
                inferred_time_bucket = self._infer_line_time_bucket(filtered_df[x_column])
                if inferred_time_bucket is not None:
                    time_bucket = inferred_time_bucket

            if time_bucket and chart_type == "line":
                filtered_df, prepared_x_column = self._apply_time_bucket(
                    filtered_df, x_column, time_bucket
                )
            elif chart_type != "line":
                time_bucket = None

            prepared = self._prepare_chart_data(
                filtered_df=filtered_df,
                chart_type=chart_type,
                x_column=prepared_x_column,
                y_column=y_column,
                aggregation=aggregation,
                bins=bins,
                limit=limit,
            )
            if prepared["status"] == "error":
                prepared["dataset_name"] = handle.name
                return prepared

            effective_title = title or self._build_default_title(
                chart_type=chart_type,
                x_column=x_column,
                y_column=y_column,
                aggregation=aggregation,
                time_bucket=time_bucket,
            )
            subtitle = self._build_subtitle(
                chart_type=chart_type,
                dataset_name=handle.name,
                source_rows=source_rows,
                plotted_rows=prepared["plotted_rows"],
                sampling_applied=bool(prepared.get("sampling_applied", False)),
                bin_count=prepared.get("bin_count"),
                filters=effective_filters,
                aggregation=aggregation,
                time_bucket=time_bucket,
            )
            summary = self._build_summary(
                chart_type=chart_type,
                data=prepared["data"],
                x_label=prepared["x_label"],
                y_label=prepared["y_label"],
            )

            rendering_payload: dict[str, Any] = {
                "type": "chart",
                "chart_type": chart_type,
                "title": effective_title,
                "subtitle": subtitle,
                "summary": summary,
                "data": prepared["data"],
                "x_field": prepared["x_field"],
                "y_field": prepared["y_field"],
                "x_label": prepared["x_label"],
                "y_label": prepared["y_label"],
            }

            return {
                "status": "success",
                "dataset_name": handle.name,
                "chart_type": chart_type,
                "title": effective_title,
                "subtitle": subtitle,
                "summary": summary,
                "source_rows": source_rows,
                "plotted_rows": prepared["plotted_rows"],
                "sampling_applied": bool(prepared.get("sampling_applied", False)),
                "sampling_method": prepared.get("sampling_method"),
                "x_field": prepared["x_field"],
                "y_field": prepared["y_field"],
                "x_label": prepared["x_label"],
                "y_label": prepared["y_label"],
                "aggregation": aggregation,
                "time_bucket": time_bucket,
                "filters_applied": effective_filters,
                "data": prepared["data"],
                "tool_rendering": rendering_payload,
            }
        except Exception as exc:
            return self._error(dataset_name=dataset_name, message=str(exc))

    def _prepare_chart_data(
        self,
        *,
        filtered_df: pd.DataFrame,
        chart_type: str,
        x_column: str,
        y_column: str | None,
        aggregation: str | None,
        bins: int | None,
        limit: int | None,
    ) -> dict[str, Any]:
        if chart_type == "bar":
            return self._prepare_bar_chart(
                filtered_df=filtered_df,
                x_column=x_column,
                y_column=y_column,
                aggregation=aggregation,
                limit=limit,
            )
        if chart_type == "line":
            return self._prepare_line_chart(
                filtered_df=filtered_df,
                x_column=x_column,
                y_column=y_column,
                aggregation=aggregation,
                limit=limit,
            )
        if chart_type == "scatter":
            return self._prepare_scatter_chart(
                filtered_df=filtered_df,
                x_column=x_column,
                y_column=y_column,
                limit=limit,
            )
        if chart_type == "histogram":
            return self._prepare_histogram_chart(
                filtered_df=filtered_df,
                x_column=x_column,
                bins=bins,
            )
        return {"status": "error", "message": "Unsupported chart type."}

    def _prepare_bar_chart(
        self,
        *,
        filtered_df: pd.DataFrame,
        x_column: str,
        y_column: str | None,
        aggregation: str | None,
        limit: int | None,
    ) -> dict[str, Any]:
        grouped: pd.DataFrame
        value_label: str

        if y_column:
            if aggregation is None:
                aggregation = "mean"
            if aggregation == "count":
                grouped = (
                    filtered_df.groupby(x_column, dropna=False)
                    .size()
                    .reset_index(name="value")
                )
                value_label = "Count"
            else:
                self._require_numeric_series(filtered_df[y_column], y_column)
                grouped = (
                    filtered_df.groupby(x_column, dropna=False)[y_column]
                    .agg(aggregation)
                    .reset_index(name="value")
                )
                value_label = f"{aggregation.title()} of {y_column}"
        else:
            grouped = (
                filtered_df.groupby(x_column, dropna=False)
                .size()
                .reset_index(name="value")
            )
            aggregation = "count"
            value_label = "Count"

        grouped = self._sort_bar_chart_groups(grouped=grouped, x_column=x_column)
        total_bars = int(len(grouped))

        if limit is not None and limit > _MAX_BAR_CHART_BARS:
            return {
                "status": "error",
                "message": (
                    f"This bar chart would produce {total_bars} bars, which exceeds the maximum supported "
                    f"{_MAX_BAR_CHART_BARS} bars. Apply filters or set limit to "
                    f"{_MAX_BAR_CHART_BARS} or fewer."
                ),
            }

        if limit is None and total_bars > _MAX_BAR_CHART_BARS:
            return {
                "status": "error",
                "message": (
                    f"This bar chart would produce {total_bars} bars, which exceeds the maximum supported "
                    f"{_MAX_BAR_CHART_BARS} bars. Apply filters or set limit to "
                    f"{_MAX_BAR_CHART_BARS} or fewer."
                ),
            }

        if limit is not None:
            grouped = grouped.head(limit)

        data = [
            {x_column: self._to_json_scalar(row[x_column]), "value": self._to_json_scalar(row["value"])}
            for _, row in grouped.iterrows()
        ]
        return {
            "status": "success",
            "data": data,
            "plotted_rows": len(data),
            "x_field": x_column,
            "y_field": y_column if aggregation != "count" else "value",
            "x_label": x_column,
            "y_label": value_label,
        }

    def _sort_bar_chart_groups(
        self,
        *,
        grouped: pd.DataFrame,
        x_column: str,
    ) -> pd.DataFrame:
        x_series = grouped[x_column]

        if self._series_is_ordered_numeric(x_series):
            numeric_sort = pd.to_numeric(x_series, errors="coerce")
            return grouped.assign(_sort_key=numeric_sort).sort_values(
                "_sort_key", ascending=True, kind="stable"
            ).drop(columns=["_sort_key"])

        if self._series_is_ordered_datetime(x_series):
            datetime_sort = pd.to_datetime(x_series, errors="coerce")
            return grouped.assign(_sort_key=datetime_sort).sort_values(
                "_sort_key", ascending=True, kind="stable"
            ).drop(columns=["_sort_key"])

        return grouped.sort_values("value", ascending=False, kind="stable")

    def _series_is_ordered_numeric(self, series: pd.Series) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return False
        numeric = pd.to_numeric(non_null, errors="coerce")
        return numeric.notna().all()

    def _series_is_ordered_datetime(self, series: pd.Series) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return False
        datetime_values = pd.to_datetime(non_null, errors="coerce")
        return datetime_values.notna().all()


    def _prepare_line_chart(
        self,
        *,
        filtered_df: pd.DataFrame,
        x_column: str,
        y_column: str | None,
        aggregation: str | None,
        limit: int | None,
    ) -> dict[str, Any]:
        if not y_column:
            return {
                "status": "error",
                "message": "y_column is required for line charts.",
            }

        if aggregation is None:
            aggregation = "mean"

        if aggregation == "count":
            grouped = (
                filtered_df.groupby(x_column, dropna=False)
                .size()
                .reset_index(name="value")
            )
            value_label = "Count"
        else:
            self._require_numeric_series(filtered_df[y_column], y_column)
            grouped = (
                filtered_df.groupby(x_column, dropna=False)[y_column]
                .agg(aggregation)
                .reset_index(name="value")
            )
            value_label = f"{aggregation.title()} of {y_column}"

        grouped = grouped.sort_values(x_column, kind="stable")
        if limit is not None:
            grouped = grouped.head(limit)

        data = [
            {x_column: self._to_json_scalar(row[x_column]), "value": self._to_json_scalar(row["value"])}
            for _, row in grouped.iterrows()
        ]
        return {
            "status": "success",
            "data": data,
            "plotted_rows": len(data),
            "x_field": x_column,
            "y_field": y_column if aggregation != "count" else "value",
            "x_label": x_column,
            "y_label": value_label,
        }

    def _infer_line_time_bucket(self, series: pd.Series) -> str | None:
        non_null = series.dropna()
        if non_null.empty:
            return None

        timestamps = pd.to_datetime(non_null, errors="coerce")
        if not timestamps.notna().all():
            return None

        unique_points = int(timestamps.nunique())
        if unique_points <= 180:
            return None
        if unique_points <= 1500:
            return "week"
        if unique_points <= 5000:
            return "month"
        if unique_points <= 12000:
            return "quarter"
        return "year"

    def _prepare_scatter_chart(
        self,
        *,
        filtered_df: pd.DataFrame,
        x_column: str,
        y_column: str | None,
        limit: int | None,
    ) -> dict[str, Any]:
        if not y_column:
            return {
                "status": "error",
                "message": "y_column is required for scatter charts.",
            }

        self._require_numeric_series(filtered_df[x_column], x_column)
        self._require_numeric_series(filtered_df[y_column], y_column)

        plot_df = filtered_df[[x_column, y_column]].dropna().copy()
        total_points = int(len(plot_df))

        if limit is not None and limit > _MAX_SCATTER_PLOT_POINTS:
            return {
                "status": "error",
                "message": (
                    f"This scatter chart would plot {limit} points, which exceeds the maximum supported "
                    f"{_MAX_SCATTER_PLOT_POINTS} points. Apply filters or set limit to "
                    f"{_MAX_SCATTER_PLOT_POINTS} or fewer."
                ),
            }

        if limit is not None:
            sample_size = min(int(limit), total_points)
            if total_points > sample_size:
                plot_df = plot_df.sample(n=sample_size, random_state=42)
        elif total_points > _MAX_SCATTER_PLOT_POINTS:
            plot_df = plot_df.sample(n=_MAX_SCATTER_PLOT_POINTS, random_state=42)

        data = [
            {
                x_column: self._to_json_scalar(row[x_column]),
                y_column: self._to_json_scalar(row[y_column]),
            }
            for _, row in plot_df.iterrows()
        ]
        return {
            "status": "success",
            "data": data,
            "plotted_rows": len(data),
            "sampling_applied": len(data) < total_points,
            "sampling_method": "random_sample_fixed_seed" if len(data) < total_points else None,
            "x_field": x_column,
            "y_field": y_column,
            "x_label": x_column,
            "y_label": y_column,
        }

    def _prepare_histogram_chart(
        self,
        *,
        filtered_df: pd.DataFrame,
        x_column: str,
        bins: int | None,
    ) -> dict[str, Any]:
        self._require_numeric_series(filtered_df[x_column], x_column)
        values = pd.to_numeric(filtered_df[x_column], errors="coerce").dropna()
        if values.empty:
            return {
                "status": "error",
                "message": f"Column '{x_column}' has no usable numeric values for a histogram.",
            }

        effective_bins = bins or _DEFAULT_HISTOGRAM_BINS
        if effective_bins <= 0:
            return {
                "status": "error",
                "message": "Histogram bins must be greater than 0.",
            }
        if effective_bins > _MAX_HISTOGRAM_BINS:
            return {
                "status": "error",
                "message": f"At most {_MAX_HISTOGRAM_BINS} bins are supported for histogram charts.",
            }

        counts, edges = np.histogram(values.to_numpy(dtype=float), bins=effective_bins)
        data: list[dict[str, Any]] = []
        for index, count in enumerate(counts.tolist()):
            x0 = float(edges[index])
            x1 = float(edges[index + 1])
            data.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "count": int(count),
                    "label": self._format_histogram_bin_label(x0, x1),
                }
            )

        return {
            "status": "success",
            "data": data,
            "plotted_rows": len(data),
            "bin_count": len(data),
            "x_field": x_column,
            "y_field": "count",
            "x_label": x_column,
            "y_label": "Count",
        }

    def _format_histogram_bin_label(self, x0: float, x1: float) -> str:
        decimals = self._histogram_bin_decimals(x0, x1)
        return f"{self._format_bin_edge(x0, decimals)}–{self._format_bin_edge(x1, decimals)}"

    def _histogram_bin_decimals(self, x0: float, x1: float) -> int:
        width = abs(float(x1) - float(x0))
        max_abs = max(abs(float(x0)), abs(float(x1)))

        if max_abs >= 1000:
            return 0
        if width >= 10:
            return 0
        if width >= 1:
            return 1
        if width >= 0.1:
            return 2
        return 3

    def _format_bin_edge(self, value: float, decimals: int) -> str:
        rounded = round(float(value), decimals)
        if decimals == 0:
            return f"{rounded:,.0f}"
        return f"{rounded:,.{decimals}f}"

    def _resolve_dataset_handle(self, dataset_name: str | None) -> _DatasetHandle:
        loaded_data_context = self.data_context_service.get_loaded_data_context()
        if not loaded_data_context.get("has_data", False):
            raise ValueError("No dataset is currently loaded.")

        datasets = loaded_data_context.get("datasets", [])
        dataset_names = [
            item.get("name")
            for item in datasets
            if isinstance(item, dict) and item.get("name")
        ]
        active_dataset_name = loaded_data_context.get("active_dataset_name")
        target_dataset_name = dataset_name or active_dataset_name

        if not target_dataset_name:
            raise ValueError("No dataset is currently loaded.")

        if target_dataset_name not in dataset_names:
            available = ", ".join(dataset_names) or "none"
            raise ValueError(
                f"Dataset '{target_dataset_name}' is not loaded. Available datasets: {available}."
            )

        app = self._resolve_active_workspace_app()
        dataset_object = self._resolve_dataset_object(
            app=app,
            dataset_name=target_dataset_name,
        )
        if dataset_object is None:
            raise ValueError(
                f"Dataset '{target_dataset_name}' is not available from the active workspace app."
            )

        dataframe = self._coerce_dataframe(dataset_object, target_dataset_name)
        return _DatasetHandle(name=target_dataset_name, dataframe=dataframe)

    def _resolve_active_workspace_app(self) -> Any | None:
        module_candidates = (
            "app.components.workspace_host",
            "app.components.workspace",
            "app.pages.workspace",
            "app.workspace",
        )
        for module_name in module_candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            getter = getattr(module, "get_active_workspace_app", None)
            if callable(getter):
                try:
                    app = getter()
                except TypeError:
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

    def _resolve_dataset_object(self, *, app: Any, dataset_name: str) -> Any | None:
        derived_dataset = get_derived_dataset_object(dataset_name)
        if derived_dataset is not None:
            return derived_dataset

        getter = getattr(app, "get_dataset_object", None)
        if callable(getter):
            try:
                return getter(dataset_name=dataset_name)
            except TypeError:
                try:
                    return getter(dataset_name)
                except TypeError:
                    try:
                        return getter()
                    except TypeError:
                        return None
        return None

    def _coerce_dataframe(self, dataset_obj: Any, dataset_name: str) -> pd.DataFrame:
        if isinstance(dataset_obj, pd.DataFrame):
            return dataset_obj

        for attr_name in ("df", "dataframe", "data"):
            value = getattr(dataset_obj, attr_name, None)
            if isinstance(value, pd.DataFrame):
                return value

        raise ValueError(f"Dataset '{dataset_name}' is not backed by a pandas DataFrame.")

    def _require_column(self, df: pd.DataFrame, column_name: str) -> None:
        if column_name not in df.columns:
            available = ", ".join(map(str, df.columns.tolist()[:30]))
            raise ValueError(
                f"Column '{column_name}' was not found in the dataset. Available columns include: {available}."
            )

    def _require_numeric_series(self, series: pd.Series, column_name: str) -> None:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0:
            raise ValueError(f"Column '{column_name}' must contain numeric values for this chart type.")

    def _apply_filters(
        self,
        df: pd.DataFrame,
        filters: list[dict[str, Any]],
    ) -> pd.DataFrame:
        filtered = df.copy()
        for filter_item in filters:
            column = str(filter_item.get("column", "")).strip()
            operator = str(filter_item.get("operator", "")).strip()
            value = filter_item.get("value")

            self._require_column(filtered, column)
            series = filtered[column]

            if operator == "==":
                mask = series == value
            elif operator == "!=":
                mask = series != value
            elif operator in {">", ">=", "<", "<="}:
                numeric_series = pd.to_numeric(series, errors="coerce")
                numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.isna(numeric_value):
                    raise ValueError(
                        f"Filter value for operator '{operator}' on column '{column}' must be numeric."
                    )
                if operator == ">":
                    mask = numeric_series > numeric_value
                elif operator == ">=":
                    mask = numeric_series >= numeric_value
                elif operator == "<":
                    mask = numeric_series < numeric_value
                else:
                    mask = numeric_series <= numeric_value
            elif operator == "contains":
                mask = series.astype(str).str.contains(str(value), case=False, na=False)
            elif operator == "starts_with":
                mask = series.astype(str).str.startswith(str(value), na=False)
            elif operator == "ends_with":
                mask = series.astype(str).str.endswith(str(value), na=False)
            elif operator == "in":
                values = value if isinstance(value, list) else [value]
                mask = series.isin(values)
            elif operator == "not_in":
                values = value if isinstance(value, list) else [value]
                mask = ~series.isin(values)
            elif operator == "is_null":
                mask = series.isna()
            elif operator == "is_not_null":
                mask = series.notna()
            else:
                raise ValueError(f"Unsupported filter operator '{operator}'.")

            filtered = filtered.loc[mask.fillna(False) if hasattr(mask, "fillna") else mask]

        return filtered

    def _apply_time_bucket(
        self,
        df: pd.DataFrame,
        column_name: str,
        time_bucket: str,
    ) -> tuple[pd.DataFrame, str]:
        timestamps = pd.to_datetime(df[column_name], errors="coerce")
        if timestamps.notna().sum() == 0:
            raise ValueError(
                f"Column '{column_name}' must contain date or datetime values to use time_bucket."
            )

        bucketed_column = f"{column_name}_{time_bucket}"
        bucketed_df = df.copy()

        if time_bucket == "day":
            bucketed_series = timestamps.dt.strftime("%Y-%m-%d")
        elif time_bucket == "week":
            period = timestamps.dt.to_period("W")
            bucketed_series = period.apply(lambda item: str(item.start_time.date()) if pd.notna(item) else None)
        elif time_bucket == "month":
            bucketed_series = timestamps.dt.strftime("%Y-%m")
        elif time_bucket == "quarter":
            bucketed_series = timestamps.dt.to_period("Q").astype(str)
        elif time_bucket == "year":
            bucketed_series = timestamps.dt.strftime("%Y")
        else:
            raise ValueError(f"Unsupported time_bucket '{time_bucket}'.")

        bucketed_df[bucketed_column] = bucketed_series
        bucketed_df = bucketed_df.loc[bucketed_df[bucketed_column].notna()].copy()
        return bucketed_df, bucketed_column

    def _build_default_title(
        self,
        *,
        chart_type: str,
        x_column: str,
        y_column: str | None,
        aggregation: str | None,
        time_bucket: str | None,
    ) -> str:
        if chart_type == "histogram":
            return f"Distribution of {x_column}"
        if chart_type == "scatter":
            return f"{y_column} vs {x_column}"
        if chart_type == "line":
            if aggregation and y_column:
                prefix = f"{aggregation.title()} {y_column}"
            elif y_column:
                prefix = y_column
            else:
                prefix = "Trend"
            if time_bucket:
                return f"{prefix} by {time_bucket.title()}"
            return f"{prefix} by {x_column}"
        if chart_type == "bar":
            if aggregation and y_column and aggregation != "count":
                return f"{aggregation.title()} {y_column} by {x_column}"
            if y_column and not aggregation:
                return f"Mean {y_column} by {x_column}"
            return f"Count by {x_column}"
        return "Chart"

    def _build_subtitle(
        self,
        *,
        chart_type: str,
        dataset_name: str,
        source_rows: int,
        plotted_rows: int,
        sampling_applied: bool,
        bin_count: int | None,
        filters: list[dict[str, Any]],
        aggregation: str | None,
        time_bucket: str | None,
    ) -> str:
        parts = [f"Dataset: {dataset_name}", f"Source rows: {source_rows}"]

        if chart_type == "scatter":
            if sampling_applied:
                parts.append(f"Sampled points: {plotted_rows}")
        elif chart_type == "line":
            parts.append(f"Plotted points: {plotted_rows}")
        elif chart_type == "histogram":
            if bin_count is not None:
                parts.append(f"Bins: {bin_count}")

        if aggregation and chart_type in {"bar", "line"}:
            parts.append(f"Aggregation: {aggregation}")
        if time_bucket and chart_type == "line":
            parts.append(f"Time bucket: {time_bucket}")
        if filters:
            parts.append(f"Filters: {len(filters)}")
        return " • ".join(parts)

    def _build_summary(
        self,
        *,
        chart_type: str,
        data: list[dict[str, Any]],
        x_label: str,
        y_label: str,
    ) -> str:
        if not data:
            return "No chart points were produced."

        if chart_type in {"bar", "line"}:
            top_item = max(
                data,
                key=lambda item: pd.to_numeric(pd.Series([item.get("value")]), errors="coerce").iloc[0]
                if item.get("value") is not None
                else float("-inf"),
            )
            category = top_item.get(x_label, top_item.get("x", ""))
            value = top_item.get("value")
            return f"Highest plotted value: {category} at {self._format_number(value)} {y_label.lower()}."

        if chart_type == "scatter":
            return f"Scatter chart with {len(data)} plotted points across {x_label} and {y_label}."

        if chart_type == "histogram":
            top_bin = max(data, key=lambda item: item.get("count", 0))
            return f"Most populated bin: {top_bin.get('label', '')} with {int(top_bin.get('count', 0))} rows."

        return f"Chart contains {len(data)} plotted points."

    def _to_json_scalar(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (int, float, str, bool)):
            return value
        return str(value)

    def _format_number(self, value: Any) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return str(value)
        numeric_value = float(numeric)
        if abs(numeric_value) >= 1000:
            return f"{numeric_value:,.0f}"
        if abs(numeric_value) >= 100:
            return f"{numeric_value:,.1f}"
        return f"{numeric_value:,.2f}"

    def _error(self, *, dataset_name: str | None, message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "message": message,
            "dataset_name": dataset_name,
        }
