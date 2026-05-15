from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


# Resolve the frontend directory relative to this file so the component continues
# to work even if the project folder is renamed later.
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "chat_widget"


# Register the component. This expects the frontend widget files to live in:
# app/frontend/chat_widget/
_custom_chat_component = components.declare_component(
    "custom_chat_component",
    path=str(_FRONTEND_DIR),
)


DEFAULT_HEIGHT = 620


def render_custom_chat_component(
    messages: list[dict[str, Any]],
    *,
    height: int = DEFAULT_HEIGHT,
    theme: str = "light",
    mode: str = "full",
    header_title: str | None = None,
    header_subtitle: str | None = None,
    is_running: bool = False,
    stop_label: str = "Stop",
    is_open: bool = False,
    launcher_mode: str = "internal",
    placeholder: str = "Ask the assistant something...",
    send_label: str = "Send",
    key: str = "custom_chat_component",
) -> str | dict[str, Any] | None:
    """
    Render the custom chat widget and return a newly submitted user message.

    Parameters
    ----------
    messages:
        Chat history to render in the widget. Each item should contain:
        - role: "user" or "assistant"
        - content: message text
        - optional blocks: structured renderable UI blocks for rich assistant output
    height:
        Pixel height of the chat widget container.
    theme:
        Theme name passed to the frontend widget. Expected values are currently
        "dark" or "light".
    mode:
        Display mode for the component. Supported values are currently
        "full", "floating", and "pane".
    header_title:
        Optional title rendered by pane mode.
    header_subtitle:
        Optional subtitle rendered by pane mode.
    is_running:
        Whether an assistant run is currently active. When true, the frontend
        keeps the input read-only and turns the send action into a stop action.
    stop_label:
        Button label used for the stop action while a run is active.
    is_open:
        Whether the floating popup should render in its open state. Ignored
        by the frontend when mode is "full" or "pane".
    launcher_mode:
        Controls whether the floating chat uses its own built-in launcher or
        an external launcher. Supported values are currently "internal" and
        "external".
    placeholder:
        Placeholder text shown in the input field.
    send_label:
        Button label for message submission.
    key:
        Stable Streamlit component key.

    Returns
    -------
    str | dict[str, Any] | None
        The newly submitted user message, a structured frontend control event,
        or None if no new message/event was sent.
    """
    normalized_messages = _normalize_messages(messages)

    submitted_message = _custom_chat_component(
        messages=normalized_messages,
        height=height,
        theme=theme,
        mode=mode,
        header_title=header_title,
        header_subtitle=header_subtitle,
        is_running=bool(is_running),
        stop_label=stop_label,
        is_open=is_open,
        launcher_mode=launcher_mode,
        placeholder=placeholder,
        send_label=send_label,
        key=key,
        default=None,
    )

    if isinstance(submitted_message, str):
        submitted_message = submitted_message.strip()
        return submitted_message or None

    if isinstance(submitted_message, dict):
        event_type = str(submitted_message.get("type", "") or "").strip()
        value = submitted_message.get("value")
        if event_type == "submit_message":
            if isinstance(value, str):
                value = value.strip()
            return {
                "type": event_type,
                "event_id": submitted_message.get("event_id"),
                "value": value or "",
            }
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return submitted_message

    return None


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ensure the frontend always receives a clean, predictable message payload.
    """
    normalized: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "assistant")).strip().lower()
        if role not in {"user", "assistant"}:
            role = "assistant"

        content = str(message.get("content", ""))
        blocks = message.get("blocks", [])
        if not isinstance(blocks, list):
            blocks = []

        normalized_message: dict[str, Any] = {
            "role": role,
            "content": content,
        }

        message_id = message.get("id")
        if isinstance(message_id, str) and message_id.strip():
            normalized_message["id"] = message_id.strip()

        meta = message.get("meta")
        if isinstance(meta, dict) and meta:
            normalized_message["meta"] = dict(meta)

        if blocks:
            normalized_message["blocks"] = blocks

        normalized.append(normalized_message)

    return normalized
