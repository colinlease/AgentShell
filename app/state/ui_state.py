from __future__ import annotations

from typing import Any

import streamlit as st

from app.components.workspace_host import (
    get_active_workspace_app,
    get_workspace_host_snapshot,
)


APP_SECTIONS_KEY = "app_sections"
ACTIVE_SHELL_SECTION_KEY = "active_shell_section"
WORKSPACE_UI_STATE_KEY = "workspace_ui_state"
THEME_NAME_KEY = "theme_name"
ASSISTANT_OPEN_KEY = "assistant_popup_open"


DEFAULT_WORKSPACE_UI_STATE: dict[str, Any] = {}


def initialize_ui_state(*, shell_sections: list[str] | None = None) -> None:
    """
    Ensure the shared UI-state keys exist in Streamlit session state.

    Parameters
    ----------
    shell_sections:
        Optional list of top-level shell sections. If provided, this becomes the
        default value for the shell section list unless the session already has
        one.
    """
    if APP_SECTIONS_KEY not in st.session_state:
        st.session_state[APP_SECTIONS_KEY] = list(shell_sections or [])

    if ACTIVE_SHELL_SECTION_KEY not in st.session_state:
        sections = st.session_state.get(APP_SECTIONS_KEY, [])
        st.session_state[ACTIVE_SHELL_SECTION_KEY] = sections[0] if sections else None

    if WORKSPACE_UI_STATE_KEY not in st.session_state:
        st.session_state[WORKSPACE_UI_STATE_KEY] = dict(DEFAULT_WORKSPACE_UI_STATE)



def set_shell_sections(sections: list[str]) -> None:
    """
    Persist the top-level shell sections shown by the AgentShell app.
    """
    normalized_sections = [str(section) for section in sections]
    st.session_state[APP_SECTIONS_KEY] = normalized_sections

    current_active = st.session_state.get(ACTIVE_SHELL_SECTION_KEY)
    if current_active not in normalized_sections:
        st.session_state[ACTIVE_SHELL_SECTION_KEY] = (
            normalized_sections[0] if normalized_sections else None
        )



def set_active_shell_section(section: str | None) -> None:
    """
    Persist the currently active top-level shell section.
    """
    st.session_state[ACTIVE_SHELL_SECTION_KEY] = section



def set_workspace_ui_state(**kwargs: Any) -> None:
    """
    Merge workspace-specific host/framework UI state into the shared session-state object.

    This store remains available for shell-owned workspace metadata, but the
    mounted workspace app is now the preferred source for app-specific UI state.
    """
    current_state = get_workspace_ui_state()
    current_state.update(kwargs)
    st.session_state[WORKSPACE_UI_STATE_KEY] = current_state



def replace_workspace_ui_state(state: dict[str, Any]) -> None:
    """
    Replace the workspace UI state with a new normalized dictionary.
    """
    normalized_state = dict(DEFAULT_WORKSPACE_UI_STATE)
    normalized_state.update(state)
    st.session_state[WORKSPACE_UI_STATE_KEY] = normalized_state



def clear_workspace_ui_state() -> None:
    """
    Reset workspace UI state back to the framework defaults.
    """
    st.session_state[WORKSPACE_UI_STATE_KEY] = dict(DEFAULT_WORKSPACE_UI_STATE)



def get_shell_sections() -> list[str]:
    """
    Return the normalized list of top-level shell sections.
    """
    sections = st.session_state.get(APP_SECTIONS_KEY, [])
    if not isinstance(sections, list):
        return []
    return [str(section) for section in sections]



def get_active_shell_section() -> str | None:
    """
    Return the active top-level shell section, if known.
    """
    active_section = st.session_state.get(ACTIVE_SHELL_SECTION_KEY)
    return str(active_section) if active_section is not None else None



def get_workspace_ui_state() -> dict[str, Any]:
    """
    Return the normalized shell-owned workspace UI state.
    """
    workspace_state = st.session_state.get(WORKSPACE_UI_STATE_KEY, {})
    if not isinstance(workspace_state, dict):
        workspace_state = {}

    normalized_state = dict(DEFAULT_WORKSPACE_UI_STATE)
    normalized_state.update(workspace_state)
    return normalized_state



def get_ui_state_snapshot() -> dict[str, Any]:
    """
    Return a normalized full UI-state snapshot for the current session.

    This is the canonical framework-facing representation of the shell,
    workspace host, and mounted workspace app UI state.
    """
    shell_sections = get_shell_sections()
    active_shell_section = get_active_shell_section()
    if not active_shell_section and shell_sections:
        active_shell_section = shell_sections[0]

    workspace_host_state = get_workspace_host_snapshot()
    workspace_shell_state = get_workspace_ui_state()
    active_workspace_app = get_active_workspace_app()

    embedded_app_state: dict[str, Any] = {
        "app_loaded": False,
        "app_id": None,
        "app_label": None,
        "app_type": None,
        "state_available": False,
        "state": {},
    }

    if active_workspace_app is not None:
        try:
            raw_app_state = active_workspace_app.get_ui_state()
        except Exception as exc:
            raw_app_state = {
                "error": f"Failed to read workspace app UI state: {exc}",
            }

        if not isinstance(raw_app_state, dict):
            raw_app_state = {"value": raw_app_state}

        embedded_app_state = {
            "app_loaded": True,
            "app_id": str(active_workspace_app.app_id),
            "app_label": str(active_workspace_app.app_label),
            "app_type": str(active_workspace_app.app_type),
            "state_available": bool(raw_app_state),
            "state": raw_app_state,
        }

    return {
        "theme_name": st.session_state.get(THEME_NAME_KEY, "light"),
        "assistant_open": bool(st.session_state.get(ASSISTANT_OPEN_KEY, False)),
        "shell": {
            "active_section": active_shell_section,
            "sections": shell_sections,
        },
        "workspace_host": {
            **workspace_host_state,
            "raw": workspace_shell_state,
        },
        "embedded_app": embedded_app_state,
    }
