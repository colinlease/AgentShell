from __future__ import annotations

from pathlib import Path
from typing import Any

from app.workspace_apps.local_knowledge.constants import DATASET_EXTENSIONS, SUPPORTED_EXTENSIONS
from app.workspace_apps.local_knowledge.services.content_extraction_service import (
    LocalKnowledgeContentError,
    read_file_excerpt,
)
from app.workspace_apps.local_knowledge.services.dataset_service import (
    LocalKnowledgeDatasetError,
    load_dataset_from_inventory,
)
from app.workspace_apps.local_knowledge.services.embedding_config_service import (
    get_local_knowledge_embedding_backend,
)
from app.workspace_apps.local_knowledge.services.embedding_index_service import (
    LocalKnowledgeEmbeddingIndexError,
    index_embeddings,
)
from app.workspace_apps.local_knowledge.services.index_store import LocalKnowledgeIndexStore
from app.workspace_apps.local_knowledge.services.search_service import (
    LocalKnowledgeSearchError,
    get_search_index_status,
    index_content,
    search_content,
)
from app.workspace_apps.local_knowledge.services.semantic_retrieval_service import (
    LocalKnowledgeSemanticSearchError,
    semantic_search_content,
)
from app.workspace_apps.local_knowledge.state import get_app_state, update_state_values


class LocalKnowledgeToolService:
    """Read-only agent-facing service for Local Knowledge tools."""

    def get_context(self) -> dict[str, Any]:
        state = get_app_state()
        mounted_root_path = state.get("mounted_root_path")
        summary = self._summary_for_root(mounted_root_path)
        search_index = self._search_index_for_root(mounted_root_path)
        not_available_yet = _not_available_yet(search_index)
        return {
            "status": "ok",
            "folder_mounted": bool(state.get("folder_mounted", False)),
            "mounted_root_path": str(mounted_root_path) if mounted_root_path else None,
            "index_ready": bool(state.get("index_ready", False)),
            "read_only_source_files": True,
            "dataset_loaded": bool(state.get("dataset_loaded", False)),
            "active_dataset_name": state.get("active_dataset_name"),
            "summary": summary,
            "search_index": search_index,
            "embedding_backend": search_index.get("embedding_backend"),
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "dataset_extensions": sorted(DATASET_EXTENSIONS),
            "available_local_knowledge_tools": [
                "get_local_knowledge_context",
                "get_local_knowledge_index_status",
                "index_local_knowledge_content",
                "index_local_knowledge_embeddings",
                "list_local_knowledge_files",
                "load_local_knowledge_dataset",
                "read_local_knowledge_file",
                "search_local_knowledge",
                "semantic_search_local_knowledge",
            ],
            "not_available_yet": not_available_yet,
        }

    def get_index_status(self) -> dict[str, Any]:
        state = get_app_state()
        mounted_root_path = state.get("mounted_root_path")
        if not mounted_root_path:
            return {
                "status": "not_mounted",
                "message": "No Local Knowledge folder is mounted.",
                "folder_mounted": False,
                "index_ready": False,
            }
        summary = self._summary_for_root(mounted_root_path)
        search_index = self._search_index_for_root(mounted_root_path)
        return {
            "status": "ok",
            "folder_mounted": True,
            "mounted_root_path": str(mounted_root_path),
            "index_ready": bool(state.get("index_ready", False)),
            "summary": summary,
            "search_index": search_index,
        }

    def list_files(
        self,
        *,
        path: str | None = None,
        depth: int | None = 1,
        include_files: bool | None = True,
        include_folders: bool | None = True,
        limit: int | None = 200,
    ) -> dict[str, Any]:
        state = get_app_state()
        mounted_root_path = state.get("mounted_root_path")
        if not mounted_root_path:
            return {
                "status": "not_mounted",
                "message": "No Local Knowledge folder is mounted.",
                "entries": [],
            }

        normalized_path = _normalize_relative_path(path)
        normalized_depth = _normalize_depth(depth)
        normalized_limit = _normalize_limit(limit)
        store = LocalKnowledgeIndexStore(root_path=str(mounted_root_path))
        records = store.list_files(path=normalized_path, include_missing=False, limit=None)
        entries = _build_directory_entries(
            records=records,
            path=normalized_path,
            depth=normalized_depth,
            include_files=include_files is not False,
            include_folders=include_folders is not False,
        )
        truncated = len(entries) > normalized_limit
        visible_entries = entries[:normalized_limit]
        return {
            "status": "ok",
            "mounted_root_path": str(mounted_root_path),
            "path": normalized_path or ".",
            "depth": normalized_depth,
            "total_entries": len(entries),
            "returned_entries": len(visible_entries),
            "truncated": truncated,
            "entries": visible_entries,
        }

    def read_file(self, *, path: str | None, max_chars: int | None = None) -> dict[str, Any]:
        state = get_app_state()
        mounted_root_path = state.get("mounted_root_path")
        if not mounted_root_path:
            return {
                "status": "not_mounted",
                "message": "No Local Knowledge folder is mounted.",
            }
        try:
            return read_file_excerpt(
                root_path=str(mounted_root_path),
                relative_path=str(path or ""),
                max_chars=max_chars,
            )
        except LocalKnowledgeContentError as exc:
            return {
                "status": "error",
                "message": str(exc),
                "path": str(path or ""),
                "read_only": True,
            }

    def load_dataset(self, *, path: str | None) -> dict[str, Any]:
        state = get_app_state()
        mounted_root_path = state.get("mounted_root_path")
        if not mounted_root_path:
            return {
                "status": "not_mounted",
                "message": "No Local Knowledge folder is mounted.",
            }
        try:
            return load_dataset_from_inventory(
                root_path=str(mounted_root_path),
                relative_path=str(path or ""),
            )
        except LocalKnowledgeDatasetError as exc:
            return {
                "status": "error",
                "message": str(exc),
                "path": str(path or ""),
                "read_only_source_files": True,
            }

    def index_content(self, *, path: str | None = None, limit: int | None = None) -> dict[str, Any]:
        state = get_app_state()
        mounted_root_path = state.get("mounted_root_path")
        if not mounted_root_path:
            return {
                "status": "not_mounted",
                "message": "No Local Knowledge folder is mounted.",
            }
        try:
            result = index_content(
                root_path=str(mounted_root_path),
                path=path,
                limit=limit,
            )
            summary = result.get("summary", {})
            search_index = result.get("search_index", {})
            if isinstance(summary, dict):
                update_state_values(
                    searchable_file_count=int(search_index.get("searchable_file_count", 0) or 0),
                    content_indexed_file_count=int(search_index.get("indexed_searchable_file_count", 0) or 0),
                    content_chunk_count=int(summary.get("content_chunk_count", 0) or 0),
                    unindexed_searchable_file_count=int(
                        search_index.get("unindexed_searchable_file_count", 0) or 0
                    ),
                    status_message=(
                        f"Search index updated: {result.get('indexed_files', 0)} files "
                        f"and {result.get('indexed_chunks', 0)} chunks added."
                    ),
                    status_variant="success",
                )
            return result
        except LocalKnowledgeSearchError as exc:
            return {
                "status": "error",
                "message": str(exc),
            }

    def index_embeddings(self, *, path: str | None = None, limit: int | None = None) -> dict[str, Any]:
        state = get_app_state()
        mounted_root_path = state.get("mounted_root_path")
        if not mounted_root_path:
            return {
                "status": "not_mounted",
                "message": "No Local Knowledge folder is mounted.",
            }
        try:
            result = index_embeddings(
                root_path=str(mounted_root_path),
                path=path,
                limit=limit,
            )
            if result.get("status") == "ok":
                update_state_values(
                    status_message=(
                        f"Embedding index updated: {result.get('embedded_files', 0)} files "
                        f"and {result.get('embedded_chunks', 0)} chunks embedded."
                    ),
                    status_variant="success",
                )
            elif result.get("status") == "unavailable":
                update_state_values(
                    status_message=str(result.get("message") or "Embedding generation is unavailable."),
                    status_variant="error",
                )
            return result
        except LocalKnowledgeEmbeddingIndexError as exc:
            return {
                "status": "error",
                "message": str(exc),
            }

    def search_content(self, *, query: str | None, path: str | None = None, limit: int | None = None) -> dict[str, Any]:
        state = get_app_state()
        mounted_root_path = state.get("mounted_root_path")
        if not mounted_root_path:
            return {
                "status": "not_mounted",
                "message": "No Local Knowledge folder is mounted.",
                "results": [],
            }
        try:
            return search_content(
                root_path=str(mounted_root_path),
                query=str(query or ""),
                path=path,
                limit=limit,
            )
        except LocalKnowledgeSearchError as exc:
            return {
                "status": "error",
                "message": str(exc),
                "results": [],
            }

    def semantic_search_content(
        self,
        *,
        query: str | None,
        path: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        state = get_app_state()
        mounted_root_path = state.get("mounted_root_path")
        if not mounted_root_path:
            return {
                "status": "not_mounted",
                "message": "No Local Knowledge folder is mounted.",
                "results": [],
            }
        try:
            return semantic_search_content(
                root_path=str(mounted_root_path),
                query=str(query or ""),
                path=path,
                limit=limit,
            )
        except LocalKnowledgeSemanticSearchError as exc:
            return {
                "status": "error",
                "message": str(exc),
                "results": [],
            }

    def _summary_for_root(self, mounted_root_path: Any) -> dict[str, Any]:
        if not mounted_root_path:
            return {
                "root_id": None,
                "file_count": 0,
                "supported_file_count": 0,
                "unsupported_file_count": 0,
                "dataset_file_count": 0,
                "deleted_file_count": 0,
                "embedding_indexed_file_count": 0,
                "embedding_count": 0,
                "last_inventory_at": "",
            }
        store = LocalKnowledgeIndexStore(root_path=str(mounted_root_path))
        return store.get_summary()

    def _search_index_for_root(self, mounted_root_path: Any) -> dict[str, Any]:
        embedding_backend = get_local_knowledge_embedding_backend().to_dict()
        if not mounted_root_path:
            return {
                "path": ".",
                "searchable_file_count": 0,
                "indexed_searchable_file_count": 0,
                "unindexed_searchable_file_count": 0,
                "content_chunk_count": 0,
                "index_complete": True,
                "sample_unindexed_paths": [],
                "embedding_backend": embedding_backend,
                "embedding_index": {
                    "embedded_searchable_file_count": 0,
                    "unembedded_searchable_file_count": 0,
                    "embedding_count": 0,
                    "semantic_index_complete": False,
                    "semantic_search_available": False,
                    "sample_unembedded_paths": [],
                },
            }
        try:
            return get_search_index_status(root_path=str(mounted_root_path))
        except LocalKnowledgeSearchError:
            return {
                "path": ".",
                "searchable_file_count": 0,
                "indexed_searchable_file_count": 0,
                "unindexed_searchable_file_count": 0,
                "content_chunk_count": 0,
                "index_complete": False,
                "sample_unindexed_paths": [],
                "embedding_backend": embedding_backend,
                "embedding_index": {
                    "embedded_searchable_file_count": 0,
                    "unembedded_searchable_file_count": 0,
                    "embedding_count": 0,
                    "semantic_index_complete": False,
                    "semantic_search_available": False,
                    "sample_unembedded_paths": [],
                },
            }


def _build_directory_entries(
    *,
    records: list[dict[str, Any]],
    path: str,
    depth: int,
    include_files: bool,
    include_folders: bool,
) -> list[dict[str, Any]]:
    folder_entries: dict[str, dict[str, Any]] = {}
    file_entries: list[dict[str, Any]] = []
    prefix = f"{path}/" if path else ""

    for record in records:
        relative_path = str(record.get("relative_path") or "")
        visible_path = relative_path[len(prefix) :] if prefix and relative_path.startswith(prefix) else relative_path
        if not visible_path or visible_path == path:
            visible_path = Path(relative_path).name
        parts = [part for part in visible_path.split("/") if part]
        if not parts:
            continue

        folder_depth = min(depth, max(len(parts) - 1, 0))
        if include_folders:
            for index in range(folder_depth):
                folder_relative = "/".join(parts[: index + 1])
                full_folder_path = f"{prefix}{folder_relative}".strip("/")
                entry = folder_entries.setdefault(
                    full_folder_path,
                    {
                        "type": "folder",
                        "path": full_folder_path,
                        "name": parts[index],
                        "file_count": 0,
                        "supported_file_count": 0,
                        "dataset_file_count": 0,
                        "total_size_bytes": 0,
                    },
                )
                entry["file_count"] += 1
                entry["supported_file_count"] += 1 if record.get("support_status") == "supported" else 0
                entry["dataset_file_count"] += 1 if bool(record.get("dataset_candidate")) else 0
                entry["total_size_bytes"] += int(record.get("size_bytes") or 0)

        if include_files and len(parts) <= depth:
            file_entries.append(_compact_file_entry(record))

    entries = list(folder_entries.values()) + file_entries
    entries.sort(key=lambda item: (0 if item["type"] == "folder" else 1, str(item["path"]).lower()))
    return entries


def _compact_file_entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "file",
        "path": str(record.get("relative_path") or ""),
        "name": str(record.get("name") or ""),
        "extension": str(record.get("extension") or ""),
        "kind": str(record.get("kind") or ""),
        "support_status": str(record.get("support_status") or ""),
        "parse_status": str(record.get("parse_status") or ""),
        "dataset_candidate": bool(record.get("dataset_candidate")),
        "size_bytes": int(record.get("size_bytes") or 0),
        "mtime_ns": int(record.get("mtime_ns") or 0),
    }


def _not_available_yet(search_index: dict[str, Any]) -> list[str]:
    unavailable = []
    if not _semantic_search_available(search_index):
        unavailable.append("semantic/vector retrieval")
    unavailable.append("source file writes")
    return unavailable


def _semantic_search_available(search_index: dict[str, Any]) -> bool:
    embedding_index = search_index.get("embedding_index")
    if isinstance(embedding_index, dict):
        return bool(embedding_index.get("semantic_search_available", False))
    embedding_backend = search_index.get("embedding_backend")
    if isinstance(embedding_backend, dict):
        return bool(embedding_backend.get("semantic_search_available", False))
    return False


def _normalize_relative_path(path: str | None) -> str:
    value = str(path or "").strip().replace("\\", "/").strip("/")
    if value in {"", "."}:
        return ""
    parts = [part for part in value.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise ValueError("path must be a folder-relative path without '.' or '..' segments.")
    if ":" in parts[0]:
        raise ValueError("path must be relative to the mounted folder, not an absolute path.")
    return "/".join(parts)


def _normalize_depth(depth: int | None) -> int:
    if depth is None:
        return 1
    try:
        value = int(depth)
    except (TypeError, ValueError):
        return 1
    return max(1, min(value, 5))


def _normalize_limit(limit: int | None) -> int:
    if limit is None:
        return 200
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return 200
    return max(1, min(value, 1000))
