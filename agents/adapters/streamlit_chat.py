

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

import streamlit as st

from agents.core.runtime_protocol import BaseAgentRuntime
from agents.core.models import AgentRequest, ChatMessage
from agents.core.results import AgentResult


RUN_LOG_SESSION_KEY = "persistent_run_log_session_id"


class StreamlitChatAdapter:
    """
    Thin adapter that connects Streamlit session state and app context to the
    reusable AgentRunner.

    This layer keeps Streamlit-specific concerns out of the runner itself. Its
    job is to:
    - read chat history from session state
    - convert history into internal ChatMessage objects
    - build an AgentRequest
    - call the agent runner
    - return a structured AgentResult
    """

    def __init__(self, agent_runner: BaseAgentRuntime) -> None:
        self.agent_runner = agent_runner

    def run(self, user_input: str, *, context: dict[str, Any] | None = None) -> AgentResult:
        """
        Run the agent using the current Streamlit chat history and optional app context.
        """
        request = self.build_request(user_input, context=context)
        return self.agent_runner.run(request)

    def build_request(
        self,
        user_input: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> AgentRequest:
        """
        Build an AgentRequest from current Streamlit chat history and optional context.
        """
        merged_context = dict(context or {})
        started_at = datetime.now(timezone.utc)
        merged_context["logging_enabled"] = True
        merged_context["session_id"] = self._get_or_create_logging_session_id()
        merged_context["surface"] = str(merged_context.get("surface", "") or "chat")
        merged_context["run_started_at_utc"] = started_at.isoformat()
        merged_context["run_started_at_ms"] = int(started_at.timestamp() * 1000)

        return AgentRequest(
            user_input=user_input,
            messages=self._get_chat_messages_from_session_state(),
            context=merged_context,
        )

    @staticmethod
    def append_assistant_response(result: AgentResult) -> None:
        """
        Append the final assistant response from an AgentResult into Streamlit chat history.
        Preserves optional rich UI blocks when present.
        """
        if not result.final_text and not result.blocks:
            return

        history = st.session_state.setdefault("chat_history", [])
        message: dict[str, Any] = {
            "id": StreamlitChatAdapter.generate_message_id(),
            "role": "assistant",
            "content": result.final_text,
        }
        if result.blocks:
            message["blocks"] = result.blocks
        history.append(message)

    @staticmethod
    def append_user_message(
        user_input: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """
        Append a user message into Streamlit chat history and return its id.
        """
        history = st.session_state.setdefault("chat_history", [])
        message_id = StreamlitChatAdapter.generate_message_id()
        message: dict[str, Any] = {
            "id": message_id,
            "role": "user",
            "content": user_input,
        }
        if isinstance(meta, dict) and meta:
            message["meta"] = dict(meta)
        history.append(message)
        return message_id

    @staticmethod
    def update_message(
        message_id: str,
        *,
        content: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update a chat-history message by id.

        This helper is intentionally partial-update only so callers can mutate
        message metadata without needing to rebuild the full message payload.
        """
        history = st.session_state.get("chat_history", [])
        if not isinstance(history, list):
            return False

        for item in history:
            if str(item.get("id", "")) != message_id:
                continue

            if content is not None:
                item["content"] = content

            if blocks is not None:
                item["blocks"] = blocks

            if meta is not None:
                item["meta"] = dict(meta)

            return True

        return False

    @staticmethod
    def update_message_meta(message_id: str, meta_updates: dict[str, Any]) -> bool:
        """
        Merge metadata updates into a message by id.
        """
        if not isinstance(meta_updates, dict) or not meta_updates:
            return False

        history = st.session_state.get("chat_history", [])
        if not isinstance(history, list):
            return False

        for item in history:
            if str(item.get("id", "")) != message_id:
                continue

            current_meta = item.get("meta", {})
            if not isinstance(current_meta, dict):
                current_meta = {}

            merged_meta = dict(current_meta)
            merged_meta.update(meta_updates)
            item["meta"] = merged_meta
            return True

        return False

    @staticmethod
    def clear_message_meta(message_id: str, keys: list[str] | None = None) -> bool:
        """
        Remove metadata keys from a message, or clear all metadata when no keys are supplied.
        """
        history = st.session_state.get("chat_history", [])
        if not isinstance(history, list):
            return False

        for item in history:
            if str(item.get("id", "")) != message_id:
                continue

            current_meta = item.get("meta", {})
            if not isinstance(current_meta, dict):
                return False

            if not keys:
                item.pop("meta", None)
                return True

            next_meta = dict(current_meta)
            for key in keys:
                next_meta.pop(str(key), None)

            if next_meta:
                item["meta"] = next_meta
            else:
                item.pop("meta", None)

            return True

        return False

    @staticmethod
    def set_user_run_status(
        message_id: str,
        *,
        text: str,
        state: str,
        tool_count: int = 0,
    ) -> bool:
        """
        Store tool-status metadata on a user message by id.
        """
        return StreamlitChatAdapter.update_message_meta(
            message_id,
            {
                "run_status_text": text,
                "run_status_state": state,
                "tool_status_text": text,
                "tool_status_state": state,
                "tool_count": max(0, int(tool_count)),
            },
        )

    @staticmethod
    def set_user_run_status_lines(
        message_id: str,
        *,
        lines: list[str],
        state: str,
        tool_count: int = 0,
        display_text: str | None = None,
    ) -> bool:
        """
        Store one or more run-status lines on a user message.

        The existing single-line metadata is preserved for the current frontend.
        The line list gives the chat widget a forward-compatible way to render
        statuses such as "Ran X tools" followed by "Stopped".
        """
        normalized_lines = [str(line).strip() for line in lines if str(line).strip()]
        if not normalized_lines:
            return StreamlitChatAdapter.clear_user_run_status(message_id)

        visible_text = str(display_text or "").strip() or normalized_lines[-1]
        return StreamlitChatAdapter.update_message_meta(
            message_id,
            {
                "run_status_text": visible_text,
                "run_status_lines": normalized_lines,
                "run_status_state": state,
                "tool_status_text": visible_text,
                "tool_status_lines": normalized_lines,
                "tool_status_state": state,
                "tool_count": max(0, int(tool_count)),
            },
        )

    @staticmethod
    def clear_user_run_status(message_id: str) -> bool:
        """
        Remove run-status metadata from a user message by id.
        """
        return StreamlitChatAdapter.clear_message_meta(
            message_id,
            keys=[
                "run_status_text",
                "run_status_lines",
                "run_status_state",
                "tool_status_text",
                "tool_status_lines",
                "tool_status_state",
                "tool_count",
            ],
        )

    @staticmethod
    def set_user_tool_status(
        message_id: str,
        *,
        text: str,
        state: str,
        tool_count: int = 0,
    ) -> bool:
        """
        Compatibility wrapper for the existing tool-status API.
        """
        return StreamlitChatAdapter.set_user_run_status(
            message_id,
            text=text,
            state=state,
            tool_count=tool_count,
        )

    @staticmethod
    def clear_user_tool_status(message_id: str) -> bool:
        """
        Compatibility wrapper for the existing tool-status API.
        """
        return StreamlitChatAdapter.clear_user_run_status(message_id)

    @staticmethod
    def generate_message_id() -> str:
        """Return a stable unique id for a chat-history message."""
        return f"msg_{uuid4().hex}"

    @staticmethod
    def _get_or_create_logging_session_id() -> str:
        session_id = str(st.session_state.get(RUN_LOG_SESSION_KEY, "") or "").strip()
        if session_id:
            return session_id

        session_id = f"sess_{uuid4().hex}"
        st.session_state[RUN_LOG_SESSION_KEY] = session_id
        return session_id

    @staticmethod
    def _get_chat_messages_from_session_state() -> list[ChatMessage]:
        """
        Convert the Streamlit session chat history into internal ChatMessage objects.
        """
        history = st.session_state.get("chat_history", [])
        messages: list[ChatMessage] = []

        for item in history:
            role = str(item.get("role", "assistant")).strip().lower()
            content = str(item.get("content", ""))

            if role not in {"system", "user", "assistant", "tool"}:
                role = "assistant"

            messages.append(ChatMessage(role=role, content=content))

        return messages


def build_basic_app_context() -> dict[str, Any]:
    """
    Build a simple placeholder app context from current Streamlit session state.

    This is intentionally lightweight for now. Later, this can be expanded to
    include page name, loaded data summaries, active filters, selected records,
    or any other domain-specific context the agent should be aware of.
    """
    return {
        "theme_name": st.session_state.get("theme_name", "light"),
        "has_chat_history": bool(st.session_state.get("chat_history", [])),
        "chat_message_count": len(st.session_state.get("chat_history", [])),
    }
