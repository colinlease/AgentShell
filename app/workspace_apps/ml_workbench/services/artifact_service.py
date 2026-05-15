

"""Artifact registry helpers for the ML Workbench app.

This service owns creation, update, lookup, deletion, and lightweight
summarization of named artifacts stored in the ML Workbench session-backed
artifact registry.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.workspace_apps.ml_workbench.schemas import ArtifactMetadata, ArtifactRecord
from app.workspace_apps.ml_workbench.state import get_artifact_registry, set_artifact_registry


UTC = timezone.utc



def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()



def _safe_dtype_summary(df: pd.DataFrame) -> dict[str, str]:
    """Return a compact dtype summary for a dataframe."""
    return {column: str(dtype) for column, dtype in df.dtypes.items()}



def _safe_missing_summary(df: pd.DataFrame) -> dict[str, int]:
    """Return missing-value counts by column for a dataframe."""
    missing = df.isna().sum()
    return {column: int(count) for column, count in missing.items() if int(count) > 0}



def _infer_object_type(obj: Any) -> str:
    """Infer a stable object_type string for an artifact object."""
    if isinstance(obj, pd.DataFrame):
        return "dataframe"
    if isinstance(obj, pd.Series):
        return "series"
    return type(obj).__name__



def _build_default_metadata(
    obj: Any,
    *,
    target_column: str | None = None,
    problem_type: str | None = None,
    note: str = "",
    source_file_name: str | None = None,
    created_from_stage: str | None = None,
    ready_for_modeling: bool = False,
) -> ArtifactMetadata:
    """Build default artifact metadata based on the provided object."""
    metadata: ArtifactMetadata = {
        "target_column": target_column,
        "problem_type": problem_type,
        "note": note,
        "source_file_name": source_file_name,
        "created_from_stage": created_from_stage,
        "ready_for_modeling": ready_for_modeling,
    }

    if isinstance(obj, pd.DataFrame):
        metadata.update(
            {
                "rows": int(obj.shape[0]),
                "columns": int(obj.shape[1]),
                "column_names": list(obj.columns),
                "dtype_summary": _safe_dtype_summary(obj),
                "missing_summary": _safe_missing_summary(obj),
            }
        )

    return metadata



def build_artifact_record(
    *,
    name: str,
    kind: str,
    role: str,
    obj: Any,
    source_artifact: str | None = None,
    metadata: ArtifactMetadata | None = None,
) -> ArtifactRecord:
    """Build a normalized artifact record.

    Parameters
    ----------
    name:
        Stable artifact name.
    kind:
        High-level artifact kind, such as dataset or results_table.
    role:
        Semantic role, such as raw, working, or model_input.
    obj:
        Underlying Python object stored in the registry.
    source_artifact:
        Optional name of the parent/source artifact.
    metadata:
        Optional pre-built metadata block. When omitted, lightweight defaults
        are inferred from the object.
    """
    timestamp = _utc_now_iso()
    artifact_metadata = metadata or _build_default_metadata(obj)
    return {
        "name": name,
        "kind": kind,
        "role": role,
        "object": obj,
        "object_type": _infer_object_type(obj),
        "created_at": timestamp,
        "updated_at": timestamp,
        "source_artifact": source_artifact,
        "metadata": artifact_metadata,
    }



def register_artifact(
    *,
    name: str,
    kind: str,
    role: str,
    obj: Any,
    source_artifact: str | None = None,
    metadata: ArtifactMetadata | None = None,
) -> ArtifactRecord:
    """Create or replace a named artifact in the registry."""
    registry = get_artifact_registry()
    existing = registry.get(name)
    record = build_artifact_record(
        name=name,
        kind=kind,
        role=role,
        obj=obj,
        source_artifact=source_artifact,
        metadata=metadata,
    )
    if existing is not None:
        record["created_at"] = existing["created_at"]
    registry[name] = record
    set_artifact_registry(registry)
    return record



def update_artifact(
    name: str,
    *,
    obj: Any | None = None,
    kind: str | None = None,
    role: str | None = None,
    source_artifact: str | None = None,
    metadata_updates: dict[str, Any] | None = None,
) -> ArtifactRecord:
    """Update an existing artifact in place.

    Parameters
    ----------
    name:
        Artifact name to update.
    obj:
        Optional replacement object.
    kind:
        Optional replacement kind.
    role:
        Optional replacement role.
    source_artifact:
        Optional replacement source artifact reference.
    metadata_updates:
        Partial metadata updates to merge into the artifact metadata.
    """
    registry = get_artifact_registry()
    if name not in registry:
        raise KeyError(f"Artifact '{name}' was not found.")

    record = registry[name]
    current_obj = record["object"] if obj is None else obj

    if obj is not None:
        record["object"] = obj
        record["object_type"] = _infer_object_type(obj)

    if kind is not None:
        record["kind"] = kind
    if role is not None:
        record["role"] = role
    if source_artifact is not None:
        record["source_artifact"] = source_artifact

    refreshed_metadata = deepcopy(record["metadata"])
    if isinstance(current_obj, pd.DataFrame):
        refreshed_metadata.update(
            {
                "rows": int(current_obj.shape[0]),
                "columns": int(current_obj.shape[1]),
                "column_names": list(current_obj.columns),
                "dtype_summary": _safe_dtype_summary(current_obj),
                "missing_summary": _safe_missing_summary(current_obj),
            }
        )
    if metadata_updates:
        refreshed_metadata.update(metadata_updates)

    record["metadata"] = refreshed_metadata
    record["updated_at"] = _utc_now_iso()
    registry[name] = record
    set_artifact_registry(registry)
    return record



def get_artifact(name: str) -> ArtifactRecord | None:
    """Return one named artifact record, if present."""
    return get_artifact_registry().get(name)



def get_artifact_object(name: str) -> Any | None:
    """Return the stored object for one named artifact, if present."""
    record = get_artifact(name)
    return None if record is None else record["object"]



def list_artifacts(kind: str | None = None, role: str | None = None) -> list[ArtifactRecord]:
    """Return artifact records filtered by optional kind and/or role."""
    records = list(get_artifact_registry().values())
    if kind is not None:
        records = [record for record in records if record["kind"] == kind]
    if role is not None:
        records = [record for record in records if record["role"] == role]
    return records



def artifact_exists(name: str) -> bool:
    """Return True when the named artifact exists in the registry."""
    return name in get_artifact_registry()



def delete_artifact(name: str) -> bool:
    """Delete one artifact by name.

    Returns True if the artifact existed and was removed, otherwise False.
    """
    registry = get_artifact_registry()
    if name not in registry:
        return False
    del registry[name]
    set_artifact_registry(registry)
    return True



def clear_artifacts(kind: str | None = None, role: str | None = None) -> int:
    """Delete artifacts filtered by optional kind and/or role.

    Returns the number of artifacts removed.
    """
    registry = get_artifact_registry()
    names_to_remove: list[str] = []
    for name, record in registry.items():
        if kind is not None and record["kind"] != kind:
            continue
        if role is not None and record["role"] != role:
            continue
        names_to_remove.append(name)

    for name in names_to_remove:
        del registry[name]

    if names_to_remove:
        set_artifact_registry(registry)
    return len(names_to_remove)



def summarize_artifact(name: str) -> dict[str, Any] | None:
    """Return a lightweight summary for one artifact, excluding the raw object."""
    record = get_artifact(name)
    if record is None:
        return None
    return {
        "name": record["name"],
        "kind": record["kind"],
        "role": record["role"],
        "object_type": record["object_type"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "source_artifact": record["source_artifact"],
        "metadata": deepcopy(record["metadata"]),
    }



def summarize_all_artifacts(kind: str | None = None, role: str | None = None) -> list[dict[str, Any]]:
    """Return lightweight summaries for all matching artifacts."""
    return [
        {
            "name": record["name"],
            "kind": record["kind"],
            "role": record["role"],
            "object_type": record["object_type"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "source_artifact": record["source_artifact"],
            "metadata": deepcopy(record["metadata"]),
        }
        for record in list_artifacts(kind=kind, role=role)
    ]