

from __future__ import annotations

from io import BytesIO
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from app.workspace_apps.base import BaseWorkspaceApp


DEMO_APP_STATE_KEY = "demo_app_state"
DEMO_APP_DATAFRAME_KEY = "demo_app_dataframe"
DEMO_APP_UPLOADER_KEY = "demo_app_file_uploader"


class DemoApp(BaseWorkspaceApp):
    app_id = "demo_app"
    app_label = "Demo App"
    app_type = "streamlit"

    def initialize_state(self) -> None:
        state = st.session_state.setdefault(DEMO_APP_STATE_KEY, {})
        state.setdefault("active_internal_tab", "Data")
        state.setdefault("uploaded_file_name", None)
        state.setdefault("dataset_loaded", False)
        state.setdefault("row_count", 0)
        state.setdefault("column_count", 0)
        state.setdefault("selected_column", None)
        state.setdefault("chart_type", "Auto")
        state.setdefault("max_preview_rows", 25)
        state.setdefault("notes", "")
        state.setdefault("status", "Idle")
        state.setdefault("show_schema", False)
        state.setdefault("show_missing_values", False)
        state.setdefault("show_filtered_preview", True)
        state.setdefault("filters", {})
        state.setdefault("filter_column", None)

    def render(self) -> None:
        self.initialize_state()
        state = self._state
        theme_name = self._theme_name

        st.markdown("## Demo App")
        st.caption(
            "A simple workspace app for testing file upload, nested UI state, filters, charts, and notes."
        )

        self._render_theme_banner(theme_name)

        uploader_col, action_col = st.columns([4, 1])
        with uploader_col:
            uploaded_file = st.file_uploader(
                "Upload a CSV or Excel file",
                type=["csv", "xlsx", "xls"],
                key=DEMO_APP_UPLOADER_KEY,
            )
        with action_col:
            st.write("")
            if st.button("Clear data", key="demo_app_clear_data"):
                self._clear_loaded_data()
                st.rerun()

        df = self._load_uploaded_dataframe(uploaded_file)
        self._update_dataset_metadata(df=df, uploaded_file=uploaded_file)

        selected_tab = st.segmented_control(
            "Demo App section",
            options=["Data", "Explore", "Filters", "Notes"],
            selection_mode="single",
            default=str(state.get("active_internal_tab", "Data") or "Data"),
            key="demo_app_active_internal_tab",
        )
        state["active_internal_tab"] = str(selected_tab or "Data")

        if state["active_internal_tab"] == "Data":
            self._render_data_tab(df)
        elif state["active_internal_tab"] == "Explore":
            self._render_explore_tab(df)
        elif state["active_internal_tab"] == "Filters":
            self._render_filters_tab(df)
        else:
            self._render_notes_tab(df)

        st.divider()
        self._render_debug_state_card(df)

    def get_ui_state(self) -> dict[str, Any]:
        state = self._state
        df = st.session_state.get("demo_app_dataframe")
        filters = state.get("filters", {})
        active_filters = {key: value for key, value in filters.items() if value not in (None, "", [], {})}

        return {
            "app_loaded": True,
            "app_id": self.app_id,
            "app_label": self.app_label,
            "app_type": self.app_type,
            "active_internal_tab": state.get("active_internal_tab"),
            "selected_column": state.get("selected_column"),
            "chart_type": state.get("chart_type"),
            "status": state.get("status"),
            "notes_present": bool(str(state.get("notes", "")).strip()),
            "open_sections": {
                "show_schema": bool(state.get("show_schema")),
                "show_missing_values": bool(state.get("show_missing_values")),
                "show_filtered_preview": bool(state.get("show_filtered_preview")),
            },
            "active_filters": active_filters,
            "raw": {
                "active_internal_tab": state.get("active_internal_tab"),
                "selected_column": state.get("selected_column"),
                "chart_type": state.get("chart_type"),
                "status": state.get("status"),
                "show_schema": bool(state.get("show_schema")),
                "show_missing_values": bool(state.get("show_missing_values")),
                "show_filtered_preview": bool(state.get("show_filtered_preview")),
                "filters": dict(filters),
                "filter_column": state.get("filter_column"),
                "notes_present": bool(str(state.get("notes", "")).strip()),
                "dataset_available": isinstance(df, pd.DataFrame),
            },
        }

    def get_data_context(self) -> dict[str, Any]:
        state = self._state
        df = st.session_state.get(DEMO_APP_DATAFRAME_KEY)

        if not isinstance(df, pd.DataFrame):
            return {
                "has_data": False,
                "dataset_count": 0,
                "active_dataset_name": None,
                "datasets": [],
            }

        missing_by_column = {
            str(column): int(df[column].isna().sum())
            for column in df.columns
            if int(df[column].isna().sum()) > 0
        }
        missing_summary = {
            "columns_with_missing_count": len(missing_by_column),
            "columns_with_missing": missing_by_column,
        }

        dtype_summary = {
            "numeric": [str(column) for column in df.select_dtypes(include="number").columns],
            "datetime": [str(column) for column in df.select_dtypes(include=["datetime", "datetimetz"]).columns],
            "boolean": [str(column) for column in df.select_dtypes(include="bool").columns],
            "other": [
                str(column)
                for column in df.columns
                if str(column)
                not in set(df.select_dtypes(include="number").columns)
                and str(column)
                not in set(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
                and str(column) not in set(df.select_dtypes(include="bool").columns)
            ],
        }

        dataset_name = str(state.get("uploaded_file_name") or "demo_app_dataset")

        return {
            "has_data": True,
            "dataset_count": 1,
            "active_dataset_name": dataset_name,
            "datasets": [
                {
                    "name": dataset_name,
                    "type": "dataframe",
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "column_names": [str(column) for column in df.columns],
                    "dtype_summary": dtype_summary,
                    "missing_summary": missing_summary,
                }
            ],
        }

    def get_dataset_object(self, dataset_name: str | None = None) -> Any | None:
        """
        Return the actual dataset object for the requested dataset name.

        If no dataset name is provided, return the active dataset object.
        """
        df = st.session_state.get(DEMO_APP_DATAFRAME_KEY)
        if not isinstance(df, pd.DataFrame):
            return None

        active_dataset_name = str(self._state.get("uploaded_file_name") or "demo_app_dataset")
        if dataset_name in (None, "", active_dataset_name):
            return df

        return None

    @property
    def _state(self) -> dict[str, Any]:
        return st.session_state.setdefault(DEMO_APP_STATE_KEY, {})

    @property
    def _theme_name(self) -> str:
        theme_name = st.session_state.get("theme_name", "light")
        return str(theme_name) if theme_name else "light"

    def _render_theme_banner(self, theme_name: str) -> None:
        if theme_name == "light":
            st.info("Shell theme detected: Light")
        else:
            st.info("Shell theme detected: Dark")

    def _load_uploaded_dataframe(self, uploaded_file) -> pd.DataFrame | None:
        existing_df = st.session_state.get(DEMO_APP_DATAFRAME_KEY)

        if uploaded_file is None:
            return existing_df if isinstance(existing_df, pd.DataFrame) else None

        file_name = str(uploaded_file.name)
        file_bytes = uploaded_file.getvalue()
        suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""

        try:
            if suffix == "csv":
                df = pd.read_csv(BytesIO(file_bytes))
            elif suffix in {"xlsx", "xls"}:
                df = pd.read_excel(BytesIO(file_bytes))
            else:
                st.error("Unsupported file type.")
                return existing_df if isinstance(existing_df, pd.DataFrame) else None
        except Exception as exc:
            st.error(f"Failed to load file: {exc}")
            return existing_df if isinstance(existing_df, pd.DataFrame) else None

        st.session_state[DEMO_APP_DATAFRAME_KEY] = df
        return df

    def _update_dataset_metadata(self, df: pd.DataFrame | None, uploaded_file) -> None:
        state = self._state
        if df is None:
            state["uploaded_file_name"] = None
            state["dataset_loaded"] = False
            state["row_count"] = 0
            state["column_count"] = 0
            state["selected_column"] = None
            state["filters"] = {}
            state["filter_column"] = None
            return

        if uploaded_file is not None:
            state["uploaded_file_name"] = str(uploaded_file.name)
        elif not state.get("uploaded_file_name"):
            state["uploaded_file_name"] = "Persisted dataset"

        state["dataset_loaded"] = True
        state["row_count"] = int(df.shape[0])
        state["column_count"] = int(df.shape[1])

        if state.get("selected_column") not in df.columns:
            state["selected_column"] = str(df.columns[0]) if len(df.columns) else None

        filter_column = state.get("filter_column")
        if filter_column not in df.columns:
            state["filter_column"] = str(df.columns[0]) if len(df.columns) else None
            state["filters"] = {}
    def _clear_loaded_data(self) -> None:
        state = self._state
        st.session_state.pop(DEMO_APP_DATAFRAME_KEY, None)
        st.session_state.pop(DEMO_APP_UPLOADER_KEY, None)
        state["uploaded_file_name"] = None
        state["dataset_loaded"] = False
        state["row_count"] = 0
        state["column_count"] = 0
        state["selected_column"] = None
        state["filters"] = {}
        state["filter_column"] = None

    def _render_data_tab(self, df: pd.DataFrame | None) -> None:
        state = self._state

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", state.get("row_count", 0))
        col2.metric("Columns", state.get("column_count", 0))
        col3.metric("Dataset Loaded", "Yes" if state.get("dataset_loaded") else "No")

        if df is None:
            st.warning("Upload a CSV or Excel file to begin.")
            return

        max_rows = st.slider(
            "Preview row limit",
            min_value=5,
            max_value=min(100, max(len(df), 5)),
            value=min(int(state.get("max_preview_rows", 25)), min(100, max(len(df), 5))),
            key="demo_app_preview_row_limit",
        )
        state["max_preview_rows"] = int(max_rows)

        with st.expander("Dataset Preview", expanded=True):
            st.dataframe(df.head(max_rows), use_container_width=True)

        state["show_schema"] = st.checkbox(
            "Show schema",
            value=bool(state.get("show_schema", False)),
            key="demo_app_show_schema_toggle",
        )
        if state["show_schema"]:
            schema_df = pd.DataFrame(
                {
                    "column": df.columns,
                    "dtype": [str(dtype) for dtype in df.dtypes],
                    "non_null_count": [int(df[col].notna().sum()) for col in df.columns],
                }
            )
            st.dataframe(schema_df, use_container_width=True)

        state["show_missing_values"] = st.checkbox(
            "Show missing values",
            value=bool(state.get("show_missing_values", False)),
            key="demo_app_show_missing_values_toggle",
        )
        if state["show_missing_values"]:
            missing_df = pd.DataFrame(
                {
                    "column": df.columns,
                    "missing_count": [int(df[col].isna().sum()) for col in df.columns],
                    "missing_pct": [
                        float((df[col].isna().mean() * 100.0) if len(df) else 0.0)
                        for col in df.columns
                    ],
                }
            )
            st.dataframe(missing_df, use_container_width=True)

    def _render_explore_tab(self, df: pd.DataFrame | None) -> None:
        state = self._state
        if df is None:
            st.warning("Upload a dataset to explore columns and charts.")
            return

        selected_column = st.selectbox(
            "Select a column",
            options=list(df.columns),
            index=list(df.columns).index(state.get("selected_column"))
            if state.get("selected_column") in df.columns
            else 0,
            key="demo_app_selected_column",
        )
        state["selected_column"] = str(selected_column)

        series = df[selected_column]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        chart_type = st.selectbox(
            "Chart type",
            options=["Auto", "Histogram", "Bar"],
            index=["Auto", "Histogram", "Bar"].index(state.get("chart_type", "Auto")),
            key="demo_app_chart_type",
        )
        state["chart_type"] = str(chart_type)

        if is_numeric:
            stats = {
                "mean": float(series.dropna().mean()) if series.dropna().shape[0] else None,
                "median": float(series.dropna().median()) if series.dropna().shape[0] else None,
                "min": float(series.dropna().min()) if series.dropna().shape[0] else None,
                "max": float(series.dropna().max()) if series.dropna().shape[0] else None,
            }
            st.json(stats)
        else:
            top_values = (
                series.astype(str)
                .fillna("<missing>")
                .value_counts(dropna=False)
                .head(10)
                .rename_axis("value")
                .reset_index(name="count")
            )
            st.dataframe(top_values, use_container_width=True)

        chart = self._build_chart(df=df, column_name=selected_column, chart_type=chart_type)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)

    def _render_filters_tab(self, df: pd.DataFrame | None) -> None:
        state = self._state
        if df is None:
            st.warning("Upload a dataset to test filters.")
            state["filters"] = {}
            state["filter_column"] = None
            return

        column_options = list(df.columns)
        current_filter_column = state.get("filter_column")
        column_name = st.selectbox(
            "Filter column",
            options=column_options,
            index=column_options.index(current_filter_column)
            if current_filter_column in column_options
            else 0,
            key="demo_app_filter_column",
        )
        state["filter_column"] = str(column_name)
        series = df[column_name]
        filters: dict[str, Any] = {}

        if pd.api.types.is_numeric_dtype(series):
            min_value = float(series.min()) if len(series.dropna()) else 0.0
            max_value = float(series.max()) if len(series.dropna()) else 0.0
            selected_range = st.slider(
                "Numeric range",
                min_value=min_value,
                max_value=max_value,
                value=(min_value, max_value),
                key="demo_app_numeric_filter_range",
            )
            filters = {
                "column": str(column_name),
                "type": "numeric_range",
                "min": float(selected_range[0]),
                "max": float(selected_range[1]),
            }
            filtered_df = df[(df[column_name] >= selected_range[0]) & (df[column_name] <= selected_range[1])]
        else:
            options = sorted(series.astype(str).fillna("<missing>").unique().tolist())
            selected_values = st.multiselect(
                "Category values",
                options=options,
                default=options,
                key="demo_app_category_filter_values",
            )
            filters = {
                "column": str(column_name),
                "type": "categorical_values",
                "values": list(selected_values),
            }
            compare_series = series.astype(str).fillna("<missing>")
            filtered_df = df[compare_series.isin(selected_values)]

        state["filters"] = filters

        st.metric("Filtered rows", int(filtered_df.shape[0]))

        show_filtered_preview = st.checkbox(
            "Show filtered preview",
            value=bool(state.get("show_filtered_preview", True)),
            key="demo_app_show_filtered_preview",
        )
        state["show_filtered_preview"] = bool(show_filtered_preview)

        if show_filtered_preview:
            st.dataframe(filtered_df.head(25), use_container_width=True)

        if st.button("Reset filters", key="demo_app_reset_filters"):
            state["filters"] = {}
            st.rerun()

    def _render_notes_tab(self, df: pd.DataFrame | None) -> None:
        state = self._state

        status = st.selectbox(
            "Review status",
            options=["Idle", "Reviewing", "Ready", "Needs Attention"],
            index=["Idle", "Reviewing", "Ready", "Needs Attention"].index(
                state.get("status", "Idle")
            ),
            key="demo_app_status",
        )
        state["status"] = str(status)

        notes = st.text_area(
            "Notes",
            value=str(state.get("notes", "")),
            height=180,
            key="demo_app_notes",
        )
        state["notes"] = str(notes)

        st.caption(
            "Use this section to test freeform UI state that an agent may need to understand later."
        )

        if df is not None:
            st.write(
                f"Current dataset: **{state.get('uploaded_file_name')}** with **{state.get('row_count', 0)}** rows."
            )

    def _render_debug_state_card(self, df: pd.DataFrame | None) -> None:
        state = self._state
        st.markdown("### Current App State")
        st.json(
            {
                "theme_name": self._theme_name,
                "ui_state": self.get_ui_state(),
                "data_context": self.get_data_context(),
            }
        )

    def _build_chart(
        self,
        *,
        df: pd.DataFrame,
        column_name: str,
        chart_type: str,
    ):
        plot_df = df[[column_name]].copy()
        plot_df = plot_df.dropna()
        if plot_df.empty:
            return None

        series = plot_df[column_name]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        resolved_chart_type = chart_type
        if resolved_chart_type == "Auto":
            resolved_chart_type = "Histogram" if is_numeric else "Bar"

        if resolved_chart_type == "Histogram" and is_numeric:
            return alt.Chart(plot_df).mark_bar().encode(x=alt.X(f"{column_name}:Q", bin=True), y="count()")

        if resolved_chart_type == "Bar":
            top_values = (
                series.astype(str)
                .value_counts()
                .head(20)
                .rename_axis(column_name)
                .reset_index(name="count")
            )
            return alt.Chart(top_values).mark_bar().encode(
                x=alt.X(f"{column_name}:N", sort="-y"),
                y=alt.Y("count:Q"),
            )

        return None
