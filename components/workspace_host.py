

from __future__ import annotations

import streamlit as st

from app.workspace_apps.base import BaseWorkspaceApp
from app.workspace_apps.registry import (
    WorkspaceAppRegistryError,
    get_default_workspace_app,
    get_default_workspace_app_id,
    get_workspace_app,
    has_registered_workspace_apps,
    list_workspace_app_metadata,
)


ACTIVE_WORKSPACE_APP_ID_KEY = "active_workspace_app_id"
WORKSPACE_HOST_MOUNT_KEY = "workspace_host_mount"


def initialize_workspace_host_state() -> None:
    """
    Ensure the workspace host has a valid active workspace app id in session state.
    """
    if not has_registered_workspace_apps():
        return

    active_app_id = str(st.session_state.get(ACTIVE_WORKSPACE_APP_ID_KEY, "") or "").strip()
    if active_app_id:
        try:
            get_workspace_app(active_app_id)
            return
        except WorkspaceAppRegistryError:
            pass

    st.session_state[ACTIVE_WORKSPACE_APP_ID_KEY] = get_default_workspace_app_id()



def get_active_workspace_app_id() -> str | None:
    """
    Return the current active workspace app id, if available.
    """
    active_app_id = str(st.session_state.get(ACTIVE_WORKSPACE_APP_ID_KEY, "") or "").strip()
    return active_app_id or None



def set_active_workspace_app_id(app_id: str) -> None:
    """
    Set the active workspace app id after validating that it is registered.
    """
    normalized_app_id = str(app_id or "").strip()
    if not normalized_app_id:
        raise WorkspaceAppRegistryError("Active workspace app id cannot be blank.")

    get_workspace_app(normalized_app_id)
    st.session_state[ACTIVE_WORKSPACE_APP_ID_KEY] = normalized_app_id



def get_active_workspace_app() -> BaseWorkspaceApp | None:
    """
    Return the active workspace app instance, or None if no apps are registered.
    """
    if not has_registered_workspace_apps():
        return None

    initialize_workspace_host_state()
    active_app_id = get_active_workspace_app_id()
    if not active_app_id:
        return get_default_workspace_app()

    return get_workspace_app(active_app_id)



def get_workspace_host_snapshot() -> dict[str, object]:
    """
    Return host-level metadata about the mounted workspace app.
    """
    active_app = get_active_workspace_app()
    available_apps = list_workspace_app_metadata()

    if active_app is None:
        return {
            "workspace_loaded": False,
            "active_workspace_app_id": None,
            "active_workspace_app_label": None,
            "active_workspace_app_type": None,
            "available_apps": available_apps,
            "available_app_count": 0,
        }

    return {
        "workspace_loaded": True,
        "active_workspace_app_id": str(active_app.app_id),
        "active_workspace_app_label": str(active_app.app_label),
        "active_workspace_app_type": str(active_app.app_type),
        "available_apps": available_apps,
        "available_app_count": len(available_apps),
    }



def render_workspace_host() -> BaseWorkspaceApp | None:
    """
    Render the currently active workspace app inside the shell Workspace tab.

    The host owns which app is mounted. The app itself owns its internal UI
    state and rendering implementation.
    """
    if not has_registered_workspace_apps():
        st.info("No workspace app has been registered yet.")
        return None

    initialize_workspace_host_state()
    active_app = get_active_workspace_app()
    if active_app is None:
        st.info("No active workspace app is available.")
        return None

    mount = st.container(border=False, key=WORKSPACE_HOST_MOUNT_KEY)
    with mount:
        active_app.initialize_state()
        active_app.render()
    return active_app
