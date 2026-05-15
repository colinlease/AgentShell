from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from app.workspace_apps.local_knowledge.constants import DATASET_EXTENSIONS
from app.workspace_apps.local_knowledge.services.index_store import LocalKnowledgeIndexStore
from app.workspace_apps.local_knowledge.state import get_app_state, update_state_values


MAX_DATASET_FILE_BYTES = 100 * 1024 * 1024


class LocalKnowledgeDatasetError(RuntimeError):
    """Raised when a Local Knowledge dataset file cannot be loaded."""


def load_dataset_from_inventory(*, root_path: str, relative_path: str) -> dict[str, Any]:
    normalized_path = _normalize_relative_path(relative_path)
    if not normalized_path:
        raise LocalKnowledgeDatasetError("path is required.")

    store = LocalKnowledgeIndexStore(root_path=str(root_path))
    record = store.get_file(normalized_path)
    if record is None:
        raise LocalKnowledgeDatasetError("Dataset file is not present in the current Local Knowledge inventory.")

    extension = str(record.get("extension") or "").lower()
    if extension not in DATASET_EXTENSIONS:
        raise LocalKnowledgeDatasetError("File is not a CSV or Excel dataset candidate.")

    size_bytes = int(record.get("size_bytes") or 0)
    if size_bytes > MAX_DATASET_FILE_BYTES:
        raise LocalKnowledgeDatasetError(f"Dataset file is too large to load ({size_bytes} bytes).")

    file_path = _resolve_inventory_path(root_path=str(root_path), relative_path=normalized_path)
    dataframe = _read_dataframe(file_path=file_path, extension=extension)
    dataset_name = _build_dataset_name(normalized_path)
    metadata = _build_dataset_metadata(
        dataset_name=dataset_name,
        relative_path=normalized_path,
        record=record,
        dataframe=dataframe,
    )

    state = get_app_state()
    loaded_datasets = dict(state.get("loaded_datasets") or {})
    loaded_metadata = dict(state.get("loaded_dataset_metadata") or {})
    loaded_datasets[dataset_name] = dataframe
    loaded_metadata[dataset_name] = metadata
    update_state_values(
        loaded_datasets=loaded_datasets,
        loaded_dataset_metadata=loaded_metadata,
        dataset_loaded=True,
        active_dataset_name=dataset_name,
        active_dataset_path=normalized_path,
        status_message=f"Dataset loaded: {dataset_name} ({dataframe.shape[0]} rows, {dataframe.shape[1]} columns).",
        status_variant="success",
    )

    return {
        "status": "ok",
        "dataset_name": dataset_name,
        "path": normalized_path,
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": [str(column) for column in dataframe.columns],
        "metadata": metadata,
    }


def get_dataset_object(dataset_name: str | None = None) -> Any | None:
    state = get_app_state()
    loaded_datasets = state.get("loaded_datasets") or {}
    if not isinstance(loaded_datasets, dict) or not loaded_datasets:
        return None
    active_dataset_name = state.get("active_dataset_name")
    requested_name = str(dataset_name) if dataset_name not in (None, "") else str(active_dataset_name or "")
    if not requested_name:
        return None
    dataset = loaded_datasets.get(requested_name)
    return dataset if isinstance(dataset, pd.DataFrame) else None


def build_loaded_dataset_context() -> dict[str, Any]:
    state = get_app_state()
    loaded_metadata = state.get("loaded_dataset_metadata") or {}
    loaded_datasets = state.get("loaded_datasets") or {}
    if not isinstance(loaded_metadata, dict) or not isinstance(loaded_datasets, dict):
        return {
            "has_data": False,
            "dataset_count": 0,
            "active_dataset_name": None,
            "datasets": [],
        }

    datasets: list[dict[str, Any]] = []
    for dataset_name, dataframe in loaded_datasets.items():
        if not isinstance(dataframe, pd.DataFrame):
            continue
        metadata = loaded_metadata.get(dataset_name)
        if isinstance(metadata, dict):
            datasets.append(metadata)
        else:
            datasets.append(_build_dataset_metadata(
                dataset_name=str(dataset_name),
                relative_path=str(dataset_name),
                record={},
                dataframe=dataframe,
            ))

    active_dataset_name = state.get("active_dataset_name")
    if active_dataset_name not in {dataset["name"] for dataset in datasets}:
        active_dataset_name = datasets[0]["name"] if datasets else None

    return {
        "has_data": bool(datasets),
        "dataset_count": len(datasets),
        "active_dataset_name": active_dataset_name,
        "datasets": datasets,
    }


def clear_loaded_dataset_if_missing(*, root_path: str) -> None:
    state = get_app_state()
    active_path = state.get("active_dataset_path")
    if not active_path:
        return
    store = LocalKnowledgeIndexStore(root_path=str(root_path))
    record = store.get_file(str(active_path))
    if record is not None:
        return
    update_state_values(
        dataset_loaded=False,
        active_dataset_name=None,
        active_dataset_path=None,
        loaded_datasets={},
        loaded_dataset_metadata={},
        status_message="Active dataset was removed from the mounted folder and has been unloaded.",
        status_variant="info",
    )


def _read_dataframe(*, file_path: Path, extension: str) -> pd.DataFrame:
    try:
        if extension == ".csv":
            return _read_csv(file_path)
        if extension in {".xlsx", ".xlsm", ".xls"}:
            return pd.read_excel(file_path)
    except Exception as exc:
        raise LocalKnowledgeDatasetError(f"Dataset could not be loaded: {exc}") from exc
    raise LocalKnowledgeDatasetError(f"Unsupported dataset extension: {extension or 'none'}.")


def _read_csv(file_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(file_path)


def _build_dataset_metadata(
    *,
    dataset_name: str,
    relative_path: str,
    record: dict[str, Any],
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    missing_by_column = {
        str(column): int(dataframe[column].isna().sum())
        for column in dataframe.columns
        if int(dataframe[column].isna().sum()) > 0
    }
    dtype_summary = {
        "numeric": [str(column) for column in dataframe.select_dtypes(include="number").columns],
        "datetime": [str(column) for column in dataframe.select_dtypes(include=["datetime", "datetimetz"]).columns],
        "boolean": [str(column) for column in dataframe.select_dtypes(include="bool").columns],
        "other": [
            str(column)
            for column in dataframe.columns
            if str(column)
            not in set(dataframe.select_dtypes(include="number").columns)
            and str(column) not in set(dataframe.select_dtypes(include=["datetime", "datetimetz"]).columns)
            and str(column) not in set(dataframe.select_dtypes(include="bool").columns)
        ],
    }
    return {
        "name": dataset_name,
        "type": "dataframe",
        "source": "local_knowledge",
        "path": relative_path,
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": [str(column) for column in dataframe.columns],
        "dtype_summary": dtype_summary,
        "missing_summary": {
            "columns_with_missing_count": len(missing_by_column),
            "columns_with_missing": missing_by_column,
        },
        "source_file": {
            "relative_path": relative_path,
            "extension": str(record.get("extension") or Path(relative_path).suffix.lower()),
            "size_bytes": int(record.get("size_bytes") or 0),
            "mtime_ns": int(record.get("mtime_ns") or 0),
            "read_only": True,
        },
    }


def _build_dataset_name(relative_path: str) -> str:
    path = Path(relative_path)
    stem = path.stem.strip() or "dataset"
    readable = "".join(character if character.isalnum() else "_" for character in stem).strip("_")
    readable = readable or "dataset"
    suffix = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:8]
    return f"local_knowledge_{readable}_{suffix}"


def _resolve_inventory_path(*, root_path: str, relative_path: str) -> Path:
    root = Path(root_path).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise LocalKnowledgeDatasetError("path must stay inside the mounted folder.") from exc
    if not candidate.exists() or not candidate.is_file():
        raise LocalKnowledgeDatasetError("Dataset file is not available on disk.")
    return candidate


def _normalize_relative_path(path: str | None) -> str:
    value = str(path or "").strip().replace("\\", "/").strip("/")
    if value in {"", "."}:
        return ""
    parts = [part for part in value.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise LocalKnowledgeDatasetError("path must be a folder-relative file path without '.' or '..' segments.")
    if parts and ":" in parts[0]:
        raise LocalKnowledgeDatasetError("path must be relative to the mounted folder, not an absolute path.")
    return "/".join(parts)
