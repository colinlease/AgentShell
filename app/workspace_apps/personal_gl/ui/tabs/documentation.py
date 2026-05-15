"""Documentation tab renderer."""

from __future__ import annotations

import streamlit as st

from app.workspace_apps.personal_gl.runtime import AppRuntime
from app.workspace_apps.personal_gl.ui.docs import build_docs_pdf_bytes, docs_fingerprint, render_doc_sections


def render_documentation_tab(runtime: AppRuntime) -> None:
    del runtime
    col_doc_h1, col_doc_btn = st.columns([6, 2])
    with col_doc_h1:
        st.subheader("Documentation")
    with col_doc_btn:
        pdf_bytes = build_docs_pdf_bytes(docs_fingerprint())
        st.download_button(
            label="Download Documentation (PDF)",
            data=pdf_bytes if pdf_bytes else b"",
            file_name="Personal_GL_Documentation.pdf",
            mime="application/pdf",
            disabled=(not bool(pdf_bytes)),
        )
    render_doc_sections()
