from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from app.state.ui_state import (
    get_active_shell_section,
    initialize_ui_state,
    set_active_shell_section,
    set_shell_sections,
)


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "shell_tabs"
COMPONENT_HEIGHT = 76

_COMPONENT_NAME = "shell_tabs_component"
LAST_SHELL_TABS_EVENT_KEY = "_last_shell_tabs_event_counter"

shell_tabs_component = components.declare_component(
    _COMPONENT_NAME,
    path=str(FRONTEND_DIR),
)


def render_shell_tabs(tab_names: list[str]) -> str:
    """
    Render the custom shell tab bar using the declared shell-tabs component.

    Python remains the source of truth for the active shell section in session
    state. The frontend sends structured tab-change events back to Python
    without touching the browser URL, which preserves the current Streamlit
    session state and theme state.
    """
    normalized_tabs = [str(tab_name) for tab_name in tab_names]
    initialize_ui_state(shell_sections=normalized_tabs)
    set_shell_sections(normalized_tabs)

    if not normalized_tabs:
        return ""

    active_section = get_active_shell_section()
    if active_section not in normalized_tabs:
        active_section = normalized_tabs[0]
        set_active_shell_section(active_section)

    result = shell_tabs_component(
        tabs=normalized_tabs,
        activeTab=active_section,
        themeName=components_value_theme_name(),
        key="shell_tabs_component",
        default=None,
    )

    selected_tab = result.get("activeTab") if isinstance(result, dict) else None
    event_counter = result.get("eventCounter") if isinstance(result, dict) else None
    last_event_counter = st.session_state.get(LAST_SHELL_TABS_EVENT_KEY)

    if (
        isinstance(selected_tab, str)
        and selected_tab in normalized_tabs
        and selected_tab != active_section
        and event_counter != last_event_counter
    ):
        st.session_state[LAST_SHELL_TABS_EVENT_KEY] = event_counter
        set_active_shell_section(selected_tab)
        st.rerun()

    return get_active_shell_section()


def components_value_theme_name() -> str:
    theme_name = st.session_state.get("theme_name", "light")
    return str(theme_name) if theme_name else "light"