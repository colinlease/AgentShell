from __future__ import annotations

from typing import Any

from app.workspace_apps.local_knowledge.constants import APP_ID, APP_LABEL, APP_TYPE


def get_app_description() -> str:
    return "Mount a local folder so the assistant can inspect its structure and use indexed files as grounded local context."


def get_app_manifest() -> dict[str, Any]:
    return {
        "app_id": APP_ID,
        "app_label": APP_LABEL,
        "app_type": APP_TYPE,
        "description": get_app_description(),
        "capabilities": [
            "folder_mount",
            "read_only_source_files",
            "lightweight_file_inventory",
            "inventory_refresh",
            "file_structure_tools",
            "bounded_file_read",
            "dataset_publication",
            "keyword_content_search",
            "semantic_content_search",
            "separate_embedding_backend_config",
            "guarded_embedding_backend",
            "embedding_indexing",
            "embedding_index_metadata",
        ],
    }
