from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from app.workspace_apps.local_knowledge.constants import (
    DATASET_EXTENSIONS,
    IGNORED_DIRECTORY_NAMES,
    IGNORED_FILE_NAMES,
    SUPPORTED_EXTENSIONS,
)
from app.workspace_apps.local_knowledge.services.index_store import LocalKnowledgeIndexStore


class LocalKnowledgeInventoryError(RuntimeError):
    """Raised when a local knowledge inventory cannot be completed."""


def refresh_inventory(root_path: str) -> dict[str, Any]:
    """
    Scan a mounted folder and persist lightweight file metadata.

    Phase 2 intentionally records filesystem metadata only. Content extraction,
    embeddings, and dataset publication attach to this persisted inventory in
    later phases.
    """
    root = Path(str(root_path or "")).expanduser()
    if not root.exists() or not root.is_dir():
        raise LocalKnowledgeInventoryError("Mounted folder is no longer available.")
    try:
        resolved_root = root.resolve()
    except OSError as exc:
        raise LocalKnowledgeInventoryError(f"Folder could not be resolved: {exc}") from exc

    records = list(_iter_file_records(resolved_root))
    store = LocalKnowledgeIndexStore(root_path=str(resolved_root))
    store.initialize()
    store.upsert_root()
    return store.replace_inventory(records)


def get_inventory_summary(root_path: str) -> dict[str, Any]:
    root = Path(str(root_path or "")).expanduser()
    try:
        resolved_root = root.resolve()
    except OSError:
        resolved_root = root
    store = LocalKnowledgeIndexStore(root_path=str(resolved_root))
    return store.get_summary()


def _iter_file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for current_dir_raw, dir_names, file_names in os.walk(root):
        current_dir = Path(current_dir_raw)
        dir_names[:] = sorted(
            name
            for name in dir_names
            if name not in IGNORED_DIRECTORY_NAMES and not _is_hidden_generated_dir(name)
        )
        for file_name in sorted(file_names):
            if file_name in IGNORED_FILE_NAMES:
                continue
            file_path = current_dir / file_name
            record = _build_file_record(root=root, file_path=file_path)
            if record is not None:
                records.append(record)
    return records


def _build_file_record(*, root: Path, file_path: Path) -> dict[str, Any] | None:
    try:
        stat = file_path.stat()
    except OSError:
        return None
    if not file_path.is_file():
        return None

    try:
        relative_path = file_path.relative_to(root).as_posix()
    except ValueError:
        return None

    extension = file_path.suffix.lower()
    support_status = "supported" if extension in SUPPORTED_EXTENSIONS else "unsupported"
    dataset_candidate = extension in DATASET_EXTENSIONS
    return {
        "relative_path": relative_path,
        "absolute_path": str(file_path),
        "name": file_path.name,
        "extension": extension,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "content_hash": _metadata_fingerprint(relative_path, stat),
        "kind": _classify_file_kind(extension),
        "support_status": support_status,
        "parse_status": "not_parsed",
        "dataset_candidate": dataset_candidate,
        "error_message": "",
    }


def _metadata_fingerprint(relative_path: str, stat: Any) -> str:
    payload = f"{relative_path}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _classify_file_kind(extension: str) -> str:
    if extension in DATASET_EXTENSIONS:
        return "dataset"
    if extension in {".md", ".txt", ".py", ".json"}:
        return "text"
    if extension in {".pdf", ".docx", ".pptx"}:
        return "document"
    return "unsupported"


def _is_hidden_generated_dir(name: str) -> bool:
    return name.startswith(".") and name.endswith("_cache")
