from __future__ import annotations

from typing import Any

from app.workspace_apps.local_knowledge.constants import APP_ID, APP_LABEL
from app.workspace_apps.local_knowledge.services.dataset_service import build_loaded_dataset_context
from app.workspace_apps.local_knowledge.services.embedding_config_service import (
    get_local_knowledge_embedding_backend,
)
from app.workspace_apps.local_knowledge.state import get_app_state


def build_published_ui_state() -> dict[str, Any]:
    state = get_app_state()
    embedding_backend = get_local_knowledge_embedding_backend().to_dict()
    return {
        "app_id": APP_ID,
        "app_label": APP_LABEL,
        "folder_mounted": bool(state.get("folder_mounted", False)),
        "mounted_root_path": state.get("mounted_root_path"),
        "index_ready": bool(state.get("index_ready", False)),
        "root_id": state.get("root_id"),
        "file_count": int(state.get("file_count", 0) or 0),
        "supported_file_count": int(state.get("supported_file_count", 0) or 0),
        "unsupported_file_count": int(state.get("unsupported_file_count", 0) or 0),
        "dataset_file_count": int(state.get("dataset_file_count", 0) or 0),
        "searchable_file_count": int(state.get("searchable_file_count", 0) or 0),
        "content_indexed_file_count": int(state.get("content_indexed_file_count", 0) or 0),
        "content_chunk_count": int(state.get("content_chunk_count", 0) or 0),
        "unindexed_searchable_file_count": int(state.get("unindexed_searchable_file_count", 0) or 0),
        "deleted_file_count": int(state.get("deleted_file_count", 0) or 0),
        "last_inventory_at": state.get("last_inventory_at"),
        "dataset_loaded": bool(state.get("dataset_loaded", False)),
        "active_dataset_name": state.get("active_dataset_name"),
        "active_dataset_path": state.get("active_dataset_path"),
        "embedding_backend": embedding_backend,
        "status": str(state.get("status_message", "")),
    }


def build_published_data_context() -> dict[str, Any]:
    state = get_app_state()
    mounted_root_path = state.get("mounted_root_path")
    dataset_context = build_loaded_dataset_context()
    embedding_backend = get_local_knowledge_embedding_backend().to_dict()
    return {
        "has_data": bool(dataset_context.get("has_data", False)),
        "dataset_count": int(dataset_context.get("dataset_count", 0) or 0),
        "active_dataset_name": dataset_context.get("active_dataset_name"),
        "datasets": list(dataset_context.get("datasets", []) or []),
        "local_knowledge": {
            "folder_mounted": bool(state.get("folder_mounted", False)),
            "mounted_root_path": str(mounted_root_path) if mounted_root_path else None,
            "index_ready": bool(state.get("index_ready", False)),
            "file_count": int(state.get("file_count", 0) or 0),
            "supported_file_count": int(state.get("supported_file_count", 0) or 0),
            "unsupported_file_count": int(state.get("unsupported_file_count", 0) or 0),
            "dataset_file_count": int(state.get("dataset_file_count", 0) or 0),
            "searchable_file_count": int(state.get("searchable_file_count", 0) or 0),
            "content_indexed_file_count": int(state.get("content_indexed_file_count", 0) or 0),
            "content_chunk_count": int(state.get("content_chunk_count", 0) or 0),
            "unindexed_searchable_file_count": int(state.get("unindexed_searchable_file_count", 0) or 0),
            "deleted_file_count": int(state.get("deleted_file_count", 0) or 0),
            "last_inventory_at": state.get("last_inventory_at"),
            "supported_file_indexing": True,
            "bounded_file_read_available": True,
            "content_search_available": True,
            "vector_search_available": bool(embedding_backend.get("semantic_search_available", False)),
            "embedding_backend": embedding_backend,
            "dataset_publication_available": True,
            "read_only_source_files": True,
        },
    }
