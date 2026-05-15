from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT
from app.workspace_apps.local_knowledge.services.index_store import INDEX_ROOT


RECENT_FOLDERS_PATH = PROJECT_ROOT / "runtime" / "local_knowledge" / "recent_folders.json"
MAX_RECENT_FOLDERS_STORED = 10
DEFAULT_RECENT_FOLDERS_LIMIT = 5


def load_recent_folders(
    *,
    limit: int | None = DEFAULT_RECENT_FOLDERS_LIMIT,
    storage_path: Path | None = None,
    index_root_path: Path | None = None,
) -> list[dict[str, Any]]:
    records = _merge_recent_folder_records(
        [
            *_read_recent_folder_records(storage_path or RECENT_FOLDERS_PATH),
            *_read_index_root_records(index_root_path or INDEX_ROOT),
        ]
    )
    visible_records: list[dict[str, Any]] = []
    for record in records:
        path = str(record.get("path") or "").strip()
        if not path:
            continue
        folder_path = Path(path)
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        visible_records.append(
            {
                "path": path,
                "label": _folder_label(path),
                "last_used_at": str(record.get("last_used_at") or ""),
            }
        )
    return visible_records[: _normalize_limit(limit)]


def record_recent_folder(
    path: str,
    *,
    storage_path: Path | None = None,
    max_records: int = MAX_RECENT_FOLDERS_STORED,
) -> None:
    normalized_path = _normalize_existing_folder_path(path)
    if not normalized_path:
        return

    target_path = storage_path or RECENT_FOLDERS_PATH
    records = [
        record
        for record in _read_recent_folder_records(target_path)
        if str(record.get("path") or "") != normalized_path
    ]
    records.insert(
        0,
        {
            "path": normalized_path,
            "label": _folder_label(normalized_path),
            "last_used_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_recent_folder_records(target_path, records[: max(1, int(max_records or MAX_RECENT_FOLDERS_STORED))])


def _read_recent_folder_records(storage_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(storage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_records = payload.get("folders") if isinstance(payload, dict) else payload
    if not isinstance(raw_records, list):
        return []
    return [record for record in raw_records if isinstance(record, dict)]


def _read_index_root_records(index_root_path: Path) -> list[dict[str, Any]]:
    if not index_root_path.exists() or not index_root_path.is_dir():
        return []

    records: list[dict[str, Any]] = []
    for db_path in index_root_path.glob("*/local_knowledge.sqlite"):
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                rows = conn.execute(
                    """
                    SELECT root_path, last_seen_at, last_inventory_at
                    FROM workspace_roots
                    """
                ).fetchall()
        except (OSError, sqlite3.DatabaseError):
            continue
        for root_path, last_seen_at, last_inventory_at in rows:
            path = str(root_path or "").strip()
            if not path:
                continue
            records.append(
                {
                    "path": path,
                    "label": _folder_label(path),
                    "last_used_at": str(last_seen_at or last_inventory_at or ""),
                }
            )
    return records


def _merge_recent_folder_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in records:
        path = str(record.get("path") or "").strip()
        if not path:
            continue
        timestamp = str(record.get("last_used_at") or "")
        existing = merged.get(path)
        if existing is None or timestamp >= str(existing.get("last_used_at") or ""):
            merged[path] = {
                "path": path,
                "label": str(record.get("label") or _folder_label(path)),
                "last_used_at": timestamp,
            }
    return sorted(merged.values(), key=lambda record: str(record.get("last_used_at") or ""), reverse=True)


def _write_recent_folder_records(storage_path: Path, records: list[dict[str, Any]]) -> None:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"folders": records}
    storage_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _normalize_existing_folder_path(path: str) -> str:
    raw_path = str(path or "").strip()
    if not raw_path:
        return ""
    folder_path = Path(raw_path)
    if not folder_path.exists() or not folder_path.is_dir():
        return ""
    try:
        return str(folder_path.resolve())
    except OSError:
        return str(folder_path)


def _folder_label(path: str) -> str:
    folder_path = Path(path)
    return folder_path.name or str(folder_path)


def _normalize_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_RECENT_FOLDERS_LIMIT
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_RECENT_FOLDERS_LIMIT
    return max(1, min(value, MAX_RECENT_FOLDERS_STORED))
