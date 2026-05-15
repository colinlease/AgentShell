from __future__ import annotations

from typing import Any

import streamlit as st

from agents.core.models import ChatMessage


COMPACTED_SUMMARY_KEY = "chat_compacted_summary"
COMPACTED_MESSAGE_COUNT_KEY = "chat_compacted_message_count"


def initialize_chat_context_state() -> None:
    """
    Ensure hidden chat-context state exists separately from visible chat history.
    """
    defaults = {
        COMPACTED_SUMMARY_KEY: None,
        COMPACTED_MESSAGE_COUNT_KEY: 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_compacted_summary() -> str | None:
    initialize_chat_context_state()
    summary = st.session_state.get(COMPACTED_SUMMARY_KEY)
    if summary in (None, ""):
        return None
    return str(summary)


def set_compacted_summary(summary: str, compacted_message_count: int) -> None:
    initialize_chat_context_state()
    normalized_summary = " ".join(str(summary or "").split()).strip()
    st.session_state[COMPACTED_SUMMARY_KEY] = normalized_summary or None
    st.session_state[COMPACTED_MESSAGE_COUNT_KEY] = max(0, int(compacted_message_count))


def clear_compacted_summary() -> None:
    initialize_chat_context_state()
    st.session_state[COMPACTED_SUMMARY_KEY] = None
    st.session_state[COMPACTED_MESSAGE_COUNT_KEY] = 0


def get_compacted_message_count() -> int:
    initialize_chat_context_state()
    return max(0, int(st.session_state.get(COMPACTED_MESSAGE_COUNT_KEY, 0)))


def get_recent_message_metrics() -> dict[str, int]:
    initialize_chat_context_state()
    messages = build_visible_chat_messages()
    return {
        "message_count": len(messages),
        "character_count": sum(len(message.content) for message in messages),
        "compacted_message_count": get_compacted_message_count(),
    }


def build_visible_chat_messages() -> list[ChatMessage]:
    """
    Convert the visible Streamlit chat transcript into internal chat messages.
    """
    history = st.session_state.get("chat_history", [])
    messages: list[ChatMessage] = []
    if not isinstance(history, list):
        return messages

    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "assistant")).strip().lower()
        content = str(item.get("content", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            role = "assistant"
        messages.append(ChatMessage(role=role, content=content))

    return messages


def get_chat_context_snapshot() -> dict[str, Any]:
    initialize_chat_context_state()
    return {
        "summary": get_compacted_summary(),
        "compacted_message_count": get_compacted_message_count(),
        "metrics": get_recent_message_metrics(),
    }
