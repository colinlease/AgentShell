from __future__ import annotations

from typing import Any
import importlib

import pandas as pd
import streamlit as st

from domain.services.derived_dataset_store import get_active_derived_dataset_summary


class DataContextService:
    """
    Deterministic service for summarizing structured data currently loaded in the app.

    Preferred behavior:
    - Ask the active mounted workspace app for its published `get_data_context()`
      payload if that interface is available.

    Backward-compatible fallback:
    - If no mounted app publishes structured data context yet, summarize legacy
      dataframe objects stored directly in Streamlit session state using the
      older `loaded_dataframes` / `active_dataframe_name` convention.

    The goal is to give the agent grounded visibility into what structured data
    is available before it decides whether deeper analysis is needed.
    """

    def get_loaded_data_context(self) -> dict[str, Any]:
        """
        Return a structured summary of data currently available to the shell.
        """
        published_context = self._get_app_published_data_context()
        if published_context is not None:
            return self._with_active_derived_dataset(
                self._normalize_published_data_context(published_context)
            )

        loaded_dataframes = st.session_state.get("loaded_dataframes", {}) or {}
        active_name = st.session_state.get("active_dataframe_name")

        if not isinstance(loaded_dataframes, dict):
            loaded_dataframes = {}

        datasets: list[dict[str, Any]] = []

        for name, value in loaded_dataframes.items():
            dataset_summary = self._summarize_dataset(name=name, value=value)
            if dataset_summary is not None:
                datasets.append(dataset_summary)

        return self._with_active_derived_dataset({
            "has_data": bool(datasets),
            "dataset_count": len(datasets),
            "active_dataset_name": active_name if active_name in loaded_dataframes else None,
            "datasets": datasets,
        })

    def _get_app_published_data_context(self) -> dict[str, Any] | None:
        """
        Return a standardized data context published by the active mounted app.

        Preferred path:
        - Resolve the active mounted workspace app through the shell's canonical
          workspace host helper, if available.

        Backward-compatible fallback:
        - Fall back to a few legacy session-state keys that may directly hold
          the mounted app object.
        """
        app = self._resolve_active_workspace_app()
        if app is None:
            return None

        get_data_context = getattr(app, "get_data_context", None)
        if not callable(get_data_context):
            return None

        try:
            context = get_data_context()
        except Exception:
            return None

        return context if isinstance(context, dict) else None

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

    def _normalize_published_data_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize app-published data context to the shell's expected structure.
        """
        datasets_raw = context.get("datasets", [])
        datasets = datasets_raw if isinstance(datasets_raw, list) else []
        active_dataset_name = context.get("active_dataset_name")

        normalized_datasets: list[dict[str, Any]] = []
        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            dtype_summary_raw = dataset.get("dtype_summary", {})
            missing_summary_raw = dataset.get("missing_summary", {})

            dtype_summary = dtype_summary_raw if isinstance(dtype_summary_raw, dict) else {}
            missing_summary = missing_summary_raw if isinstance(missing_summary_raw, dict) else {}

            normalized_dataset = {
                "name": str(dataset.get("name") or "unnamed_dataset"),
                "type": str(dataset.get("type") or dataset.get("object_type") or "unknown"),
                "rows": int(dataset.get("rows", 0) or 0),
                "columns": int(dataset.get("columns", 0) or 0),
                "column_names": [str(column) for column in dataset.get("column_names", []) or []],
                "dtype_summary": {
                    "numeric": [str(column) for column in dtype_summary.get("numeric", []) or []],
                    "datetime": [str(column) for column in dtype_summary.get("datetime", []) or []],
                    "boolean": [str(column) for column in dtype_summary.get("boolean", []) or []],
                    "other": [str(column) for column in dtype_summary.get("other", []) or []],
                },
                "missing_summary": {
                    "columns_with_missing_count": int(missing_summary.get("columns_with_missing_count", 0) or 0),
                    "columns_with_missing": {
                        str(column): int(count)
                        for column, count in (missing_summary.get("columns_with_missing", {}) or {}).items()
                    },
                },
            }
            for optional_key in (
                "is_derived",
                "source_dataset_name",
                "feature_columns",
                "created_columns",
                "feature_specs",
                "estimated_memory_bytes",
            ):
                if optional_key in dataset:
                    normalized_dataset[optional_key] = dataset.get(optional_key)
            normalized_datasets.append(normalized_dataset)

        if active_dataset_name not in {dataset["name"] for dataset in normalized_datasets}:
            active_dataset_name = normalized_datasets[0]["name"] if normalized_datasets else None

        return {
            "has_data": bool(normalized_datasets),
            "dataset_count": len(normalized_datasets),
            "active_dataset_name": active_dataset_name,
            "datasets": normalized_datasets,
        }

    def _with_active_derived_dataset(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Add the session-scoped framework derived dataset, if present, and make
        it the active dataset for general data tools.
        """
        derived_summary = get_active_derived_dataset_summary()
        if derived_summary is None:
            return context

        datasets = [
            dataset
            for dataset in list(context.get("datasets", []) or [])
            if isinstance(dataset, dict) and dataset.get("name") != derived_summary.get("name")
        ]
        datasets.append(derived_summary)

        updated_context = dict(context)
        updated_context["has_data"] = True
        updated_context["dataset_count"] = len(datasets)
        updated_context["active_dataset_name"] = derived_summary.get("name")
        updated_context["datasets"] = datasets
        return updated_context

    def _summarize_dataset(self, *, name: str, value: Any) -> dict[str, Any] | None:
        """
        Build a structured summary for one loaded dataframe-like object.
        """
        if not isinstance(value, pd.DataFrame):
            return {
                "name": str(name),
                "type": type(value).__name__,
                "rows": 0,
                "columns": 0,
                "column_names": [],
                "dtype_summary": {
                    "numeric": [],
                    "datetime": [],
                    "boolean": [],
                    "other": [],
                },
                "missing_summary": {
                    "columns_with_missing_count": 0,
                    "columns_with_missing": {},
                },
                "supported": False,
                "note": "Object is loaded but is not a pandas DataFrame.",
            }

        rows, columns = value.shape
        column_names = [str(col) for col in value.columns.tolist()]
        dtype_summary = self._build_dtype_summary(value)
        missing_summary = self._build_missing_summary(value)

        return {
            "name": str(name),
            "type": "dataframe",
            "supported": True,
            "rows": int(rows),
            "columns": int(columns),
            "column_names": column_names,
            "dtype_summary": dtype_summary,
            "missing_summary": missing_summary,
        }

    @staticmethod
    def _build_dtype_summary(df: pd.DataFrame) -> dict[str, list[str]]:
        """
        Group columns into broad dtype categories useful for agent reasoning.
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

        categorized = set(numeric_cols) | set(datetime_cols) | set(bool_cols)
        other_cols = [col for col in df.columns.tolist() if col not in categorized]

        return {
            "numeric": [str(col) for col in numeric_cols],
            "datetime": [str(col) for col in datetime_cols],
            "boolean": [str(col) for col in bool_cols],
            "other": [str(col) for col in other_cols],
        }

    @staticmethod
    def _build_missing_summary(df: pd.DataFrame) -> dict[str, Any]:
        """
        Return a compact missing-data summary for the dataframe.
        """
        missing_counts = df.isna().sum()
        columns_with_missing = {
            str(col): int(count)
            for col, count in missing_counts.items()
            if int(count) > 0
        }

        return {
            "columns_with_missing_count": len(columns_with_missing),
            "columns_with_missing": columns_with_missing,
        }
