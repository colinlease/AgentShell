from __future__ import annotations

from typing import Any

import streamlit as st

from app.workspace_apps.local_knowledge.constants import LOCAL_KNOWLEDGE_STATE_KEY


def initialize_state() -> None:
    state = get_app_state()
    state.setdefault("folder_path_input", "")
    state.setdefault("pending_folder_path_input", "")
    state.setdefault("mounted_root_path", None)
    state.setdefault("folder_mounted", False)
    state.setdefault("index_ready", False)
    state.setdefault("root_id", None)
    state.setdefault("dataset_loaded", False)
    state.setdefault("active_dataset_name", None)
    state.setdefault("active_dataset_path", None)
    state.setdefault("loaded_datasets", {})
    state.setdefault("loaded_dataset_metadata", {})
    state.setdefault("file_count", 0)
    state.setdefault("supported_file_count", 0)
    state.setdefault("unsupported_file_count", 0)
    state.setdefault("dataset_file_count", 0)
    state.setdefault("searchable_file_count", 0)
    state.setdefault("content_indexed_file_count", 0)
    state.setdefault("content_chunk_count", 0)
    state.setdefault("unindexed_searchable_file_count", 0)
    state.setdefault("stale_file_count", 0)
    state.setdefault("deleted_file_count", 0)
    state.setdefault("last_inventory_at", None)
    state.setdefault("status_message", "Choose a local folder to prepare this workspace.")
    state.setdefault("status_variant", "info")
    state.setdefault("last_refresh_message", "")


def get_app_state() -> dict[str, Any]:
    return st.session_state.setdefault(LOCAL_KNOWLEDGE_STATE_KEY, {})


def update_state_values(**values: Any) -> None:
    state = get_app_state()
    for key, value in values.items():
        state[key] = value
