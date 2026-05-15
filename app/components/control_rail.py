from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from app.components.workspace_host import (
    get_active_workspace_app_id,
    set_active_workspace_app_id,
)
from app.state.agent_runtime_state import (
    initialize_agent_runtime_state,
    is_planning_enabled,
    is_reflection_enabled,
    set_planning_enabled,
    set_reflection_enabled,
)
from app.state.provider_state import (
    build_control_rail_model_options,
    get_active_control_rail_model_option_value,
    initialize_provider_state,
    set_selected_model_name,
    set_selected_provider_name,
)
from app.workspace_apps.registry import list_workspace_app_metadata


THEME_STATE_KEY = "theme_name"
ASSISTANT_OPEN_STATE_KEY = "assistant_popup_open"
LAST_CONTROL_RAIL_EVENT_KEY = "control_rail_last_event_id"

_COMPONENT_NAME = "control_rail_component"
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "control_rail"


control_rail_component = components.declare_component(
    _COMPONENT_NAME,
    path=str(_FRONTEND_DIR),
)


def render_control_rail() -> None:
    """
    Render the shared floating control rail.

    This wrapper is the Python bridge for a future unified control area that can
    contain multiple small floating controls such as:
    - theme toggle
    - assistant open/close toggle
    - future mode or permission toggles

    The frontend component owns the visual layout and click targets. Python owns
    app-global state transitions and reruns.
    """
    _initialize_state()
    initialize_provider_state()
    initialize_agent_runtime_state()

    current_theme = st.session_state.get(THEME_STATE_KEY, "light")
    assistant_open = bool(st.session_state.get(ASSISTANT_OPEN_STATE_KEY, False))
    planning_enabled = is_planning_enabled()
    reflection_enabled = is_reflection_enabled()
    available_workspace_apps = list_workspace_app_metadata()
    available_model_options = build_control_rail_model_options()

    result = control_rail_component(
        theme=current_theme,
        assistant_open=assistant_open,
        planning_enabled=planning_enabled,
        reflection_enabled=reflection_enabled,
        available_workspace_apps=available_workspace_apps,
        active_workspace_app_id=get_active_workspace_app_id(),
        available_model_options=available_model_options,
        active_model_option_value=get_active_control_rail_model_option_value(),
        key="control_rail_component",
        default=None,
    )

    event_id = result.get("event_id") if isinstance(result, dict) else None
    event_type = result.get("type") if isinstance(result, dict) else None

    if (
        event_id is not None
        and event_id != st.session_state.get(LAST_CONTROL_RAIL_EVENT_KEY)
    ):
        st.session_state[LAST_CONTROL_RAIL_EVENT_KEY] = event_id

        if event_type == "toggle_theme":
            st.session_state[THEME_STATE_KEY] = (
                "light" if current_theme == "dark" else "dark"
            )
            st.rerun()

        if event_type == "toggle_assistant":
            st.session_state[ASSISTANT_OPEN_STATE_KEY] = not assistant_open
            st.rerun()

        if event_type == "toggle_planning":
            set_planning_enabled(not planning_enabled)
            st.rerun()

        if event_type == "toggle_reflection":
            set_reflection_enabled(not reflection_enabled)
            st.rerun()

        if event_type == "toggle_assistant_close":
            st.session_state[ASSISTANT_OPEN_STATE_KEY] = False
            st.rerun()

        if event_type == "set_workspace_app":
            selected_app_id = str(result.get("app_id", "")).strip()
            if selected_app_id and selected_app_id != get_active_workspace_app_id():
                set_active_workspace_app_id(selected_app_id)
                st.rerun()

        if event_type == "set_model_selection":
            provider_name = str(result.get("provider_name", "")).strip().lower()
            model_name = str(result.get("model_name", "")).strip()
            if provider_name and model_name:
                set_selected_provider_name(provider_name)
                set_selected_model_name(model_name)
                st.rerun()
    elif result is None:
        st.session_state[LAST_CONTROL_RAIL_EVENT_KEY] = None



def _initialize_state() -> None:
    """Initialize shared control-rail state defaults."""
    if THEME_STATE_KEY not in st.session_state:
        st.session_state[THEME_STATE_KEY] = "light"

    if ASSISTANT_OPEN_STATE_KEY not in st.session_state:
        st.session_state[ASSISTANT_OPEN_STATE_KEY] = False

    if LAST_CONTROL_RAIL_EVENT_KEY not in st.session_state:
        st.session_state[LAST_CONTROL_RAIL_EVENT_KEY] = None
