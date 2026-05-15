

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agents.core.models import ChatMessage
from agents.providers.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI implementation of the provider-agnostic model interface.

    This first version is intentionally lightweight. It supports:
    - plain text responses
    - optional tool definitions
    - normalized tool-call parsing

    It returns a normalized dictionary so the rest of the app does not depend on
    OpenAI SDK response formats.
    """

    def __init__(self, *, api_key: str, model: str = "gpt-4.1-mini") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a normalized response using the OpenAI Responses API.
        """
        input_payload = self._build_input_payload(messages, system_prompt)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_payload,
        }

        if tools:
            request_kwargs["tools"] = [self._to_openai_tool_schema(tool) for tool in tools]

        response = self.client.responses.create(**request_kwargs)
        return self._normalize_response(response)

    @staticmethod
    def _build_input_payload(
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """
        Convert internal ChatMessage objects into OpenAI Responses API input.
        """
        payload: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            }
        ]

        for message in messages:
            if message.role == "assistant" and message.tool_call_id and message.name:
                normalized_text = str(message.content or "").strip()
                if normalized_text:
                    payload.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": normalized_text}],
                        }
                    )

                payload.append(
                    {
                        "type": "function_call",
                        "call_id": message.tool_call_id,
                        "name": message.name,
                        "arguments": OpenAIProvider._serialize_tool_arguments(message.tool_arguments),
                    }
                )
                continue

            if message.role == "tool":
                if message.tool_call_id:
                    payload.append(
                        {
                            "type": "function_call_output",
                            "call_id": message.tool_call_id,
                            "output": message.content,
                        }
                    )
                    continue

                payload.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": f"Tool {message.name or 'unknown'} returned: {message.content}",
                            }
                        ],
                    }
                )
                continue

            content_type = "output_text" if message.role == "assistant" else "input_text"

            payload.append(
                {
                    "role": message.role,
                    "content": [{"type": content_type, "text": message.content}],
                }
            )

        return payload

    @staticmethod
    def _serialize_tool_arguments(arguments: dict[str, Any] | None) -> str:
        if not isinstance(arguments, dict):
            return "{}"

        try:
            return json.dumps(arguments, ensure_ascii=False, default=str)
        except TypeError:
            return "{}"

    @staticmethod
    def _to_openai_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a generic provider tool schema into OpenAI tool format.
        """
        return {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        }

    def _normalize_response(self, response: Any) -> dict[str, Any]:
        """
        Convert an OpenAI response into the app's normalized response shape.
        """
        tool_call = None
        text_fragments: list[str] = []

        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)

            if item_type == "function_call":
                raw_arguments = getattr(item, "arguments", "{}") or "{}"
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    parsed_arguments = {}

                tool_call = {
                    "tool_name": getattr(item, "name", ""),
                    "arguments": parsed_arguments,
                    "tool_call_id": (
                        str(getattr(item, "call_id", "") or "").strip()
                        or str(getattr(item, "id", "") or "").strip()
                        or None
                    ),
                }
                continue

            if item_type == "message":
                for content_item in getattr(item, "content", []) or []:
                    if getattr(content_item, "type", None) in {"output_text", "text"}:
                        text_value = getattr(content_item, "text", "")
                        if isinstance(text_value, str) and text_value.strip():
                            text_fragments.append(text_value)
                        elif hasattr(text_value, "value") and str(text_value.value).strip():
                            text_fragments.append(str(text_value.value))

        return {
            "text": "\n".join(text_fragments).strip(),
            "tool_call": tool_call,
        }
