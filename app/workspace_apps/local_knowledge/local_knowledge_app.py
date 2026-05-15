from __future__ import annotations

import html
from typing import Any

import streamlit as st

from app.workspace_apps.base import BaseWorkspaceApp
from app.workspace_apps.local_knowledge.assets.css import inject_local_knowledge_css
from app.workspace_apps.local_knowledge.constants import APP_ID, APP_LABEL, APP_TYPE
from app.workspace_apps.local_knowledge.manifest import get_app_description, get_app_manifest
from app.workspace_apps.local_knowledge.services.context_service import (
    build_published_data_context,
    build_published_ui_state,
)
from app.workspace_apps.local_knowledge.services.dataset_service import (
    clear_loaded_dataset_if_missing,
    get_dataset_object,
)
from app.workspace_apps.local_knowledge.services.inventory_service import (
    LocalKnowledgeInventoryError,
    refresh_inventory,
)
from app.workspace_apps.local_knowledge.services.path_service import validate_folder_path
from app.workspace_apps.local_knowledge.services.recent_folders_service import (
    load_recent_folders,
    record_recent_folder,
)
from app.workspace_apps.local_knowledge.services.search_service import LocalKnowledgeSearchError, get_search_index_status
from app.workspace_apps.local_knowledge.state import get_app_state, initialize_state, update_state_values
from app.workspace_apps.local_knowledge.tools.factory import build_local_knowledge_tools


class LocalKnowledgeApp(BaseWorkspaceApp):
    app_id = APP_ID
    app_label = APP_LABEL
    app_type = APP_TYPE

    def __init__(self, *, hosted: bool = False) -> None:
        self._hosted = bool(hosted)

    def initialize_state(self) -> None:
        initialize_state()

    def render(self) -> None:
        self.initialize_state()
        inject_local_knowledge_css()
        self._render_hero()
        self._render_mount_panel()
        self._render_overview_cards()

    def get_ui_state(self) -> dict[str, Any]:
        return build_published_ui_state()

    def get_data_context(self) -> dict[str, Any]:
        return build_published_data_context()

    def get_dataset_object(self, dataset_name: str | None = None) -> Any | None:
        return get_dataset_object(dataset_name)

    def get_tools(self) -> list[Any]:
        return build_local_knowledge_tools()

    def describe(self) -> str:
        return get_app_description()

    def get_manifest(self) -> dict[str, Any]:
        return get_app_manifest()

    @property
    def _state(self) -> dict[str, Any]:
        return get_app_state()

    def _render_hero(self) -> None:
        st.markdown(
            """
            <div class="lkw-hero">
                <div class="lkw-eyebrow">Workspace App</div>
                <div class="lkw-title">Local Knowledge</div>
                <div class="lkw-subtitle">
                    Mount a local folder so the assistant can inventory files, read supported documents,
                    search indexed text, and use CSV or Excel files through AgentShell's data tools.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_mount_panel(self) -> None:
        state = self._state
        self._sync_folder_path_input_widget(state)
        st.markdown(
            """
            <div class="lkw-folder-copy">
                <div class="lkw-folder-title">Folder</div>
                <div class="lkw-folder-subtitle">
                    Enter a local folder path. macOS, Linux, and Windows-style paths are accepted when the app is running on that machine.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        input_col, mount_col, refresh_col = st.columns([0.62, 0.19, 0.19])
        with input_col:
            folder_path = st.text_input(
                "Folder path",
                placeholder="/Users/name/Documents/project or C:\\Users\\name\\Documents\\project",
                label_visibility="collapsed",
                key="local_knowledge_folder_path_input",
            )
            state["folder_path_input"] = folder_path
        with mount_col:
            if st.button("Mount Folder", key="local_knowledge_mount_folder", use_container_width=True):
                self._mount_folder(folder_path)
                st.rerun()
        with refresh_col:
            refresh_disabled = not bool(state.get("folder_mounted", False))
            if st.button(
                "Refresh",
                key="local_knowledge_refresh_index",
                use_container_width=True,
                disabled=refresh_disabled,
            ):
                self._refresh_inventory()
                st.rerun()

        self._render_recent_folders()
        self._render_status_message()

    def _sync_folder_path_input_widget(self, state: dict[str, Any]) -> None:
        widget_key = "local_knowledge_folder_path_input"
        pending_path = str(state.get("pending_folder_path_input") or "")
        if pending_path:
            st.session_state[widget_key] = pending_path
            state["folder_path_input"] = pending_path
            state["pending_folder_path_input"] = ""
            return
        if widget_key not in st.session_state:
            st.session_state[widget_key] = str(state.get("folder_path_input") or "")

    def _render_recent_folders(self) -> None:
        recent_folders = load_recent_folders(limit=5)
        if not recent_folders:
            return
        st.markdown('<div class="lkw-recent-label">Recent folders</div>', unsafe_allow_html=True)
        columns = st.columns([1, 1, 1, 1, 1])
        for index, folder in enumerate(recent_folders):
            path = str(folder.get("path") or "")
            label = str(folder.get("label") or path)
            with columns[index]:
                if st.button(
                    label,
                    key=f"local_knowledge_recent_folder_{index}",
                    help=path,
                    use_container_width=True,
                ):
                    update_state_values(
                        folder_path_input=path,
                        pending_folder_path_input=path,
                        status_message=f"Recent folder selected: {label}. Click Mount Folder to load it.",
                        status_variant="info",
                    )
                    st.rerun()

    def _mount_folder(self, folder_path: str) -> None:
        result = validate_folder_path(folder_path)
        if result["status"] != "ok":
            update_state_values(
                folder_path_input=result.get("normalized_path") or folder_path,
                mounted_root_path=None,
                folder_mounted=False,
                index_ready=False,
                root_id=None,
                dataset_loaded=False,
                active_dataset_name=None,
                active_dataset_path=None,
                loaded_datasets={},
                loaded_dataset_metadata={},
                file_count=0,
                supported_file_count=0,
                unsupported_file_count=0,
                dataset_file_count=0,
                searchable_file_count=0,
                content_indexed_file_count=0,
                content_chunk_count=0,
                unindexed_searchable_file_count=0,
                deleted_file_count=0,
                last_inventory_at=None,
                status_message=str(result.get("message") or "Folder could not be mounted."),
                status_variant="error",
            )
            return

        mounted_path = str(result.get("normalized_path") or "")
        record_recent_folder(mounted_path)
        update_state_values(
            folder_path_input=mounted_path,
            pending_folder_path_input="",
            mounted_root_path=mounted_path,
            folder_mounted=True,
            index_ready=False,
            dataset_loaded=False,
            active_dataset_name=None,
            active_dataset_path=None,
            loaded_datasets={},
            loaded_dataset_metadata={},
            searchable_file_count=0,
            content_indexed_file_count=0,
            content_chunk_count=0,
            unindexed_searchable_file_count=0,
            status_message="Folder mounted. Building lightweight inventory...",
            status_variant="info",
        )
        self._refresh_inventory()

    def _refresh_inventory(self) -> None:
        state = self._state
        mounted_path = str(state.get("mounted_root_path") or "")
        if not mounted_path:
            update_state_values(
                index_ready=False,
                status_message="Mount a folder before refreshing Local Knowledge.",
                status_variant="error",
                last_refresh_message="",
            )
            return
        try:
            summary = refresh_inventory(mounted_path)
        except LocalKnowledgeInventoryError as exc:
            update_state_values(
                index_ready=False,
                status_message=str(exc),
                status_variant="error",
                last_refresh_message="Inventory refresh failed.",
            )
            return

        changed_total = (
            int(summary.get("added_files", 0))
            + int(summary.get("changed_files", 0))
            + int(summary.get("restored_files", 0))
            + int(summary.get("deleted_files", 0))
        )
        previously_loaded_dataset_path = state.get("active_dataset_path")
        clear_loaded_dataset_if_missing(root_path=mounted_path)
        state = self._state
        active_dataset_was_removed = bool(previously_loaded_dataset_path) and not state.get("active_dataset_path")
        search_index_status = self._get_search_index_status(mounted_path)
        status_message = (
            "Active dataset was removed from the mounted folder and has been unloaded."
            if active_dataset_was_removed
            else (
                f"Inventory refreshed: {summary.get('file_count', 0)} files tracked"
                f" ({changed_total} changed since the previous scan)."
            )
        )
        update_state_values(
            root_id=summary.get("root_id"),
            folder_mounted=True,
            index_ready=True,
            file_count=int(summary.get("file_count", 0)),
            supported_file_count=int(summary.get("supported_file_count", 0)),
            unsupported_file_count=int(summary.get("unsupported_file_count", 0)),
            dataset_file_count=int(summary.get("dataset_file_count", 0)),
            searchable_file_count=int(search_index_status.get("searchable_file_count", 0)),
            content_indexed_file_count=int(search_index_status.get("indexed_searchable_file_count", 0)),
            content_chunk_count=int(summary.get("content_chunk_count", 0)),
            unindexed_searchable_file_count=int(search_index_status.get("unindexed_searchable_file_count", 0)),
            deleted_file_count=int(summary.get("deleted_file_count", 0)),
            last_inventory_at=summary.get("last_inventory_at") or None,
            status_message=status_message,
            status_variant="info" if active_dataset_was_removed else "success",
            last_refresh_message="Inventory refreshed.",
        )

    def _get_search_index_status(self, mounted_path: str) -> dict[str, Any]:
        try:
            return get_search_index_status(root_path=mounted_path)
        except LocalKnowledgeSearchError:
            return {
                "searchable_file_count": 0,
                "indexed_searchable_file_count": 0,
                "unindexed_searchable_file_count": 0,
            }

    def _render_status_message(self) -> None:
        state = self._state
        message = str(state.get("status_message") or "")
        variant = str(state.get("status_variant") or "info")
        if not message:
            return
        st.markdown(
            f'<div class="lkw-status {html.escape(variant)}">{html.escape(message)}</div>',
            unsafe_allow_html=True,
        )

    def _render_overview_cards(self) -> None:
        state = self._state
        mounted_label = "Yes" if bool(state.get("folder_mounted", False)) else "No"
        index_label = "Ready" if bool(state.get("index_ready", False)) else "Not built"
        dataset_label = "Yes" if bool(state.get("dataset_loaded", False)) else "No"
        file_count = int(state.get("file_count", 0) or 0)
        dataset_file_count = int(state.get("dataset_file_count", 0) or 0)
        unsupported_file_count = int(state.get("unsupported_file_count", 0) or 0)
        searchable_file_count = int(state.get("searchable_file_count", 0) or 0)
        indexed_file_count = int(state.get("content_indexed_file_count", 0) or 0)
        pending_search_count = int(state.get("unindexed_searchable_file_count", 0) or 0)
        st.markdown(
            f"""
            <div class="lkw-metric-grid">
                <div class="lkw-metric-card">
                    <div class="lkw-metric-value">{mounted_label}</div>
                    <div class="lkw-metric-label">Folder Mounted</div>
                </div>
                <div class="lkw-metric-card">
                    <div class="lkw-metric-value">{file_count}</div>
                    <div class="lkw-metric-label">Files</div>
                </div>
                <div class="lkw-metric-card">
                    <div class="lkw-metric-value">{dataset_file_count}</div>
                    <div class="lkw-metric-label">Dataset Files</div>
                </div>
                <div class="lkw-metric-card">
                    <div class="lkw-metric-value">{unsupported_file_count}</div>
                    <div class="lkw-metric-label">Unsupported</div>
                </div>
            </div>
            <div class="lkw-mini-status">
                Index: {html.escape(index_label)} | Dataset loaded: {html.escape(dataset_label)} | Search index: {indexed_file_count}/{searchable_file_count} files indexed, {pending_search_count} pending | Source files: read only
            </div>
            """,
            unsafe_allow_html=True,
        )
