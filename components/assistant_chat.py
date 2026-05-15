from __future__ import annotations

from hashlib import sha256
from typing import Any

import streamlit as st

from agents.adapters.streamlit_chat import build_basic_app_context
from agents.factory import build_streamlit_chat_adapter
from app.components.custom_chat_component import render_custom_chat_component
from app.state.chat_run_state import (
    has_active_chat_run,
    request_stop_active_chat_run,
    start_chat_run,
)


def render_assistant_chat(
    *,
    intro_message: str,
    height: int,
    component_key: str,
    last_processed_key: str,
    placeholder: str,
    send_label: str,
    surface: str,
    mode: str = "full",
    header_title: str | None = None,
    header_subtitle: str | None = None,
) -> None:
    """
    Render the shared assistant chat surface and start runs on new submissions.

    The surrounding shell owns layout and framing. This helper owns only the
    reusable chat-history rendering and submission flow.
    """
    chat_history = st.session_state.get("chat_history", [])
    if last_processed_key not in st.session_state:
        st.session_state[last_processed_key] = None

    chat_adapter = build_streamlit_chat_adapter()
    is_running = has_active_chat_run()
    widget_messages = chat_history or [
        {
            "role": "assistant",
            "content": intro_message,
        }
    ]

    submitted_message = render_custom_chat_component(
        widget_messages,
        height=height,
        theme=st.session_state.get("theme_name", "light"),
        mode=mode,
        header_title=header_title,
        header_subtitle=header_subtitle,
        is_running=is_running,
        stop_label="Stop",
        placeholder=placeholder,
        send_label=send_label,
        key=component_key,
    )

    submitted_text, submit_signature = _extract_submit_event(submitted_message)

    if isinstance(submitted_message, dict) and submitted_text is None:
        event_type = str(submitted_message.get("type", "") or "").strip()
        event_id = submitted_message.get("event_id")
        control_event_key = f"{last_processed_key}_control_event"
        event_signature = f"{event_type}:{event_id}"
        if (
            event_type == "stop_active_run"
            and event_signature != st.session_state.get(control_event_key)
        ):
            st.session_state[control_event_key] = event_signature
            if request_stop_active_chat_run():
                st.rerun()
        return

    if (
        submitted_text
        and submit_signature != st.session_state.get(last_processed_key)
    ):
        st.session_state[last_processed_key] = submit_signature

        if is_running:
            return

        message_id = chat_adapter.append_user_message(submitted_text)
        start_chat_run(
            chat_adapter,
            user_input=submitted_text,
            context=build_basic_app_context(),
            target_message_id=message_id,
            surface=surface,
        )
        st.rerun()


def _extract_submit_event(submitted_message: str | dict[str, Any] | None) -> tuple[str | None, str | None]:
    if isinstance(submitted_message, str):
        text = submitted_message.strip()
        if not text:
            return None, None
        return text, _build_submit_signature(event_type="submit_message_legacy", event_id=None, text=text)

    if not isinstance(submitted_message, dict):
        return None, None

    event_type = str(submitted_message.get("type", "") or "").strip()
    if event_type != "submit_message":
        return None, None

    text = str(submitted_message.get("value", "") or "").strip()
    if not text:
        return None, None

    return text, _build_submit_signature(
        event_type=event_type,
        event_id=submitted_message.get("event_id"),
        text=text,
    )


def _build_submit_signature(*, event_type: str, event_id: Any, text: str) -> str:
    text_digest = sha256(text.encode("utf-8")).hexdigest()
    normalized_event_id = str(event_id or "").strip()
    if normalized_event_id:
        return f"{event_type}:{normalized_event_id}:{text_digest}"
    return f"{event_type}:{text_digest}"
