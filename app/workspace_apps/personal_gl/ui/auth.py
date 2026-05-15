"""Authentication and top controls."""

from __future__ import annotations

import streamlit as st

from app.workspace_apps.GLA import get_app_access_password, get_connection
from app.workspace_apps.personal_gl.runtime import AppRuntime


def load_stored_password() -> str | None:
    conn = get_connection()
    try:
        try:
            return get_app_access_password(conn)
        except Exception:
            return None
    finally:
        conn.close()


def render_lock_gate(runtime: AppRuntime, stored_password: str | None) -> None:
    if stored_password and not runtime.state.get_app_unlocked():
        st.info("This application is locked. Enter the password to continue.")
        pw_input = st.text_input("Password", type="password", key="app_access_pw_input")
        if st.button("Unlock", key="app_access_pw_button"):
            if pw_input == stored_password:
                runtime.state.set_app_unlocked(True)
                runtime.state.append_ui_event("unlock")
                st.success("Unlocked.")
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        st.stop()

    if not stored_password:
        runtime.state.set_app_unlocked(True)


def render_global_controls(runtime: AppRuntime, stored_password: str | None) -> None:
    col_refresh, col_spacer, col_lock = st.columns([2, 6, 2])
    with col_refresh:
        if st.button("Update"):
            runtime.state.append_ui_event("refresh")
            st.rerun()
    with col_lock:
        if st.button("Lock"):
            if stored_password:
                runtime.state.set_app_unlocked(False)
                runtime.state.append_ui_event("lock")
                st.rerun()
            else:
                st.info("No app access password is set; the app cannot be locked.")
