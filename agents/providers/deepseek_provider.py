

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agents.providers.base import BaseLLMProvider
from agents.core.models import ChatMessage


class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, *, api_key: str, model: str = "deepseek-chat") -> None:
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model

    def generate(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_messages(messages=messages, system_prompt=system_prompt)
        self._validate_message_sequence(payload)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
        }

        if tools:
            request_kwargs["tools"] = [self._to_openai_tool_schema(tool) for tool in tools]
            request_kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**request_kwargs)
        return self._normalize_response(response)

    def _build_messages(
        self,
        *,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        for message in messages:
            if message.role == "assistant" and message.tool_call_id and message.name:
                normalized_text = str(message.content or "").strip()
                if normalized_text:
                    payload.append(
                        {
                            "role": "assistant",
                            "content": normalized_text,
                        }
                    )

                payload.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": message.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": message.name,
                                    "arguments": self._serialize_tool_arguments(message.tool_arguments),
                                },
                            }
                        ],
                    }
                )
                continue

            if message.role == "tool":
                if message.tool_call_id:
                    payload.append(
                        {
                            "role": "tool",
                            "tool_call_id": message.tool_call_id,
                            "content": message.content,
                        }
                    )
                    continue

                payload.append(
                    {
                        "role": "assistant",
                        "content": f"Tool {message.name or 'unknown'} returned: {message.content}",
                    }
                )
                continue

            payload.append(
                {
                    "role": message.role,
                    "content": message.content,
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

    def _to_openai_tool_schema(self, tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _validate_message_sequence(payload: list[dict[str, Any]]) -> None:
        pending_tool_call_ids: list[str] = []

        for message in payload:
            role = str(message.get("role", "") or "").strip()
            if role == "assistant":
                raw_tool_calls = message.get("tool_calls", []) or []
                for tool_call in raw_tool_calls:
                    tool_call_id = str(tool_call.get("id", "") or "").strip()
                    if tool_call_id:
                        pending_tool_call_ids.append(tool_call_id)
                continue

            if role == "tool":
                tool_call_id = str(message.get("tool_call_id", "") or "").strip()
                if tool_call_id and tool_call_id in pending_tool_call_ids:
                    pending_tool_call_ids.remove(tool_call_id)

        if pending_tool_call_ids:
            missing_ids = ", ".join(pending_tool_call_ids)
            raise ValueError(
                "Malformed DeepSeek message history: assistant tool calls are missing matching "
                f"tool messages for tool_call_id(s): {missing_ids}"
            )

    def _normalize_response(self, response: Any) -> dict[str, Any]:
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)

        text = getattr(message, "content", "") or ""
        tool_call = None

        raw_tool_calls = getattr(message, "tool_calls", None) or []
        if raw_tool_calls:
            first_tool_call = raw_tool_calls[0]
            function = getattr(first_tool_call, "function", None)
            raw_arguments = getattr(function, "arguments", "{}") or "{}"

            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed_arguments = {}

            tool_call = {
                "tool_name": getattr(function, "name", ""),
                "arguments": parsed_arguments,
                "tool_call_id": str(getattr(first_tool_call, "id", "") or "").strip() or None,
            }

        return {
            "text": text.strip(),
            "tool_call": tool_call,
        }
