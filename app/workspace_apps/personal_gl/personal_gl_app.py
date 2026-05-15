"""Workspace-app adapter for Personal GL."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.workspace_apps.base import BaseWorkspaceApp
from app.workspace_apps.personal_gl.app import render_personal_gl
from app.workspace_apps.personal_gl.constants import APP_ID, APP_LABEL, APP_TYPE
from app.workspace_apps.personal_gl.runtime import build_runtime
from app.workspace_apps.personal_gl.shell_contracts import (
    get_data_context as build_data_context,
    get_dataset_object as build_dataset_object,
    get_ui_state as build_ui_state,
)
from app.workspace_apps.personal_gl.tools.factory import build_personal_gl_tools


class PersonalGLApp(BaseWorkspaceApp):
    """Thin AgentShell adapter for the Personal GL workspace app."""

    def __init__(self, *, hosted: bool = True) -> None:
        self._hosted = bool(hosted)

    @property
    def app_id(self) -> str:
        return APP_ID

    @property
    def app_label(self) -> str:
        return APP_LABEL

    @property
    def app_type(self) -> str:
        return APP_TYPE

    def initialize_state(self) -> None:
        build_runtime(st.session_state)

    def render(self) -> None:
        render_personal_gl(hosted=self._hosted)

    def get_ui_state(self) -> dict[str, Any]:
        runtime = build_runtime(st.session_state)
        return build_ui_state(runtime)

    def get_data_context(self) -> dict[str, Any]:
        runtime = build_runtime(st.session_state)
        return build_data_context(runtime)

    def get_dataset_object(self, dataset_name: str | None = None) -> Any | None:
        runtime = build_runtime(st.session_state)
        return build_dataset_object(runtime, dataset_name=dataset_name)

    def get_tools(self) -> list[Any]:
        return build_personal_gl_tools()
