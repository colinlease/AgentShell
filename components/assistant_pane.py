from __future__ import annotations

import streamlit as st

from app.components.assistant_chat import render_assistant_chat


ASSISTANT_PANE_CHAT_HEIGHT = 760


def render_assistant_pane() -> None:
    """Render the docked assistant pane used by the shell-level AI toggle."""
    st.markdown('<div class="assistant-pane-wrap">', unsafe_allow_html=True)
    render_assistant_chat(
        intro_message=(
            "Hello. I’m the docked assistant. I stay visible while you work in the app."
        ),
        height=ASSISTANT_PANE_CHAT_HEIGHT,
        component_key="assistant_pane_chat_widget",
        last_processed_key="assistant_pane_last_processed_message",
        placeholder="Ask the assistant...",
        send_label="Send",
        surface="pane",
        mode="pane",
        header_title="Assistant",
        header_subtitle="Context-aware, tool-enabled chat",
    )
    st.markdown("</div>", unsafe_allow_html=True)
