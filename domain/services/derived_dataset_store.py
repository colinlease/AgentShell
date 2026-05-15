from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import streamlit as st


DERIVED_DATASET_KEY = "agentshell_active_derived_dataset"
DERIVED_DATASET_METADATA_KEY = "agentshell_active_derived_dataset_metadata"


def get_active_derived_dataset() -> pd.DataFrame | None:
    dataset = st.session_state.get(DERIVED_DATASET_KEY)
    return dataset if isinstance(dataset, pd.DataFrame) else None


def get_active_derived_metadata() -> dict[str, Any] | None:
    metadata = st.session_state.get(DERIVED_DATASET_METADATA_KEY)
    return deepcopy(metadata) if isinstance(metadata, dict) else None


def get_derived_dataset_object(dataset_name: str | None = None) -> pd.DataFrame | None:
    dataset = get_active_derived_dataset()
    metadata = get_active_derived_metadata()
    if dataset is None or metadata is None:
        return None

    active_name = str(metadata.get("name") or "")
    requested_name = str(dataset_name or active_name).strip()
    if requested_name and requested_name == active_name:
        return dataset
    return None


def get_active_derived_dataset_summary() -> dict[str, Any] | None:
    dataset = get_active_derived_dataset()
    metadata = get_active_derived_metadata()
    if dataset is None or metadata is None:
        return None

    summary = _build_dataset_summary(
        dataframe=dataset,
        name=str(metadata.get("name") or "derived_dataset"),
        source_dataset_name=str(metadata.get("source_dataset_name") or ""),
        feature_specs=list(metadata.get("feature_specs", []) or []),
        estimated_memory_bytes=int(metadata.get("estimated_memory_bytes", 0) or 0),
    )
    summary.update({
        "created_columns": [str(column) for column in metadata.get("created_columns", []) or []],
        "feature_columns": [str(column) for column in metadata.get("feature_columns", []) or []],
    })
    return summary


def set_active_derived_dataset(
    *,
    dataframe: pd.DataFrame,
    name: str,
    source_dataset_name: str,
    feature_specs: list[dict[str, Any]],
    created_columns: list[str],
    feature_columns: list[str] | None = None,
    estimated_memory_bytes: int | None = None,
) -> dict[str, Any]:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Derived dataset must be a pandas DataFrame.")

    memory_bytes = (
        int(estimated_memory_bytes)
        if estimated_memory_bytes is not None
        else _estimate_dataframe_memory_bytes(dataframe)
    )
    summary = _build_dataset_summary(
        dataframe=dataframe,
        name=name,
        source_dataset_name=source_dataset_name,
        feature_specs=feature_specs,
        estimated_memory_bytes=memory_bytes,
    )
    summary["created_columns"] = [str(column) for column in created_columns]
    summary["feature_columns"] = [
        str(column)
        for column in (feature_columns if feature_columns is not None else created_columns)
    ]

    st.session_state[DERIVED_DATASET_KEY] = dataframe
    st.session_state[DERIVED_DATASET_METADATA_KEY] = deepcopy(summary)
    return deepcopy(summary)


def clear_active_derived_dataset() -> None:
    st.session_state.pop(DERIVED_DATASET_KEY, None)
    st.session_state.pop(DERIVED_DATASET_METADATA_KEY, None)


def _build_dataset_summary(
    *,
    dataframe: pd.DataFrame,
    name: str,
    source_dataset_name: str,
    feature_specs: list[dict[str, Any]],
    estimated_memory_bytes: int,
) -> dict[str, Any]:
    missing_by_column = {
        str(column): int(dataframe[column].isna().sum())
        for column in dataframe.columns
        if int(dataframe[column].isna().sum()) > 0
    }
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    datetime_columns = dataframe.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    boolean_columns = dataframe.select_dtypes(include="bool").columns.tolist()
    categorized = set(numeric_columns) | set(datetime_columns) | set(boolean_columns)
    other_columns = [column for column in dataframe.columns.tolist() if column not in categorized]

    return {
        "name": str(name),
        "type": "dataframe",
        "supported": True,
        "is_derived": True,
        "source_dataset_name": str(source_dataset_name),
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": [str(column) for column in dataframe.columns],
        "dtype_summary": {
            "numeric": [str(column) for column in numeric_columns],
            "datetime": [str(column) for column in datetime_columns],
            "boolean": [str(column) for column in boolean_columns],
            "other": [str(column) for column in other_columns],
        },
        "missing_summary": {
            "columns_with_missing_count": len(missing_by_column),
            "columns_with_missing": missing_by_column,
        },
        "feature_specs": deepcopy(feature_specs),
        "estimated_memory_bytes": int(estimated_memory_bytes),
    }


def _estimate_dataframe_memory_bytes(dataframe: pd.DataFrame) -> int:
    try:
        return int(dataframe.memory_usage(deep=True).sum())
    except Exception:
        return int(dataframe.shape[0] * max(1, dataframe.shape[1]) * 8)
