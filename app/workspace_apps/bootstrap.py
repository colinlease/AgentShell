

from __future__ import annotations

import streamlit as st

from app.workspace_apps.demo_app.demo_app import DemoApp
from app.workspace_apps.local_knowledge.local_knowledge_app import LocalKnowledgeApp
from app.workspace_apps.ml_workbench.ml_app import MLWorkbenchApp
from app.workspace_apps.personal_gl.personal_gl_app import PersonalGLApp
from app.workspace_apps.registry import (
    WorkspaceAppRegistryError,
    has_registered_workspace_apps,
    register_workspace_app,
)


WORKSPACE_APPS_BOOTSTRAPPED_KEY = "_workspace_apps_bootstrapped"


def bootstrap_workspace_apps() -> None:
    """
    Register all workspace apps available to the shell.

    This function is designed to be safe across Streamlit reruns. It registers
    apps once per session and skips duplicate registration attempts.
    """
    if st.session_state.get(WORKSPACE_APPS_BOOTSTRAPPED_KEY):
        return

    if has_registered_workspace_apps():
        st.session_state[WORKSPACE_APPS_BOOTSTRAPPED_KEY] = True
        return

    apps_to_register = [
        (DemoApp(), True),
        (MLWorkbenchApp(hosted=True), False),
        (PersonalGLApp(hosted=True), False),
        (LocalKnowledgeApp(hosted=True), False),
    ]

    for app, is_default in apps_to_register:
        try:
            register_workspace_app(app, is_default=is_default)
        except WorkspaceAppRegistryError as exc:
            if "already registered" not in str(exc):
                raise

    st.session_state[WORKSPACE_APPS_BOOTSTRAPPED_KEY] = True
