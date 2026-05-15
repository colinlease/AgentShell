from __future__ import annotations

from copy import deepcopy
from typing import Any

from agents.providers.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """
    Gemini provider wrapper that mirrors the same high-level interface used by
    the OpenAI provider.

    Responsibilities:
    - initialize the Gemini client
    - convert app messages into Gemini contents/parts
    - convert provider-agnostic tool schemas into Gemini function declarations
    - normalize Gemini responses into the common provider return shape
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise ValueError("Gemini API key is required.")

        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The google-genai package is required to use GeminiProvider. "
                "Install it with: pip install google-genai"
            ) from exc

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(
        self,
        messages: list[Any],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a model response using the Gemini API and normalize the result.
        """
        contents = self._build_contents_payload(messages)
        config = self._build_generation_config(system_prompt=system_prompt, tools=tools)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return self._normalize_response(response)

    def _build_contents_payload(self, messages: list[Any]) -> list[dict[str, Any]]:
        """
        Convert provider-agnostic messages into Gemini contents.
        """
        contents: list[dict[str, Any]] = []

        for message in messages:
            role = self._get_message_attr(message, "role", "user")
            name = self._get_message_attr(message, "name")
            content = self._get_message_attr(message, "content", "")

            if role == "system":
                # System instructions are passed separately through the Gemini
                # generation config, so skip system messages here.
                continue

            if role == "tool":
                tool_name = name or "tool"
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": self._normalize_tool_response_content(content),
                                }
                            }
                        ],
                    }
                )
                continue

            text = self._normalize_content_to_text(content)
            gemini_role = "model" if role == "assistant" else "user"

            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": text}],
                }
            )

        return contents

    def _get_message_attr(self, message: Any, key: str, default: Any = None) -> Any:
        """
        Read a message field from either a dict-like message or an object-like
        message (for example a ChatMessage instance).
        """
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    def _normalize_content_to_text(self, content: Any) -> str:
        """
        Normalize message content into plain text for Gemini parts.
        """
        if isinstance(content, list):
            text_chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text") or item.get("content") or item.get("value")
                    if isinstance(text_value, str):
                        text_chunks.append(text_value)
                elif isinstance(item, str):
                    text_chunks.append(item)
                else:
                    text_chunks.append(str(item))
            return "\n".join(chunk for chunk in text_chunks if chunk)

        if isinstance(content, str):
            return content

        return str(content)

    def _normalize_tool_response_content(self, content: Any) -> dict[str, Any]:
        """
        Normalize tool output into a Gemini-native function response payload.
        """
        if isinstance(content, dict):
            return content

        if isinstance(content, list):
            return {
                "items": [
                    item if isinstance(item, dict) else {"value": str(item)}
                    for item in content
                ]
            }

        if isinstance(content, str):
            return {"content": content}

        return {"content": str(content)}

    def _build_generation_config(
        self,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        """
        Build a Gemini generation config with optional function declarations.
        """
        try:
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The google-genai package is required to use GeminiProvider. "
                "Install it with: pip install google-genai"
            ) from exc

        function_declarations = []
        for tool in tools or []:
            function_declarations.append(self._to_gemini_function_declaration(tool, types))

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
        }

        if function_declarations:
            config_kwargs["tools"] = [
                types.Tool(function_declarations=function_declarations)
            ]

        return types.GenerateContentConfig(**config_kwargs)

    def _to_gemini_function_declaration(self, tool: dict[str, Any], types: Any) -> Any:
        """
        Convert a provider-agnostic tool schema into a Gemini-compatible
        function declaration schema.
        """
        raw_parameters = tool.get("parameters", {}) or {"type": "object", "properties": {}}
        parameters = self._normalize_schema_for_gemini(raw_parameters)

        return types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=parameters,
        )

    def _normalize_schema_for_gemini(self, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize a provider-agnostic JSON-schema-like dictionary into the
        stricter schema shape expected by Gemini function declarations.

        Main adaptations:
        - convert lowercase JSON Schema types to Gemini/OpenAPI-style uppercase
          types
        - collapse nullable union types like ["string", "null"] into a
          single non-null type
        - remove None values from enums
        - recurse through object properties and array items

        Optional fields remain optional by omission from `required`; explicit
        null typing is intentionally dropped for Gemini compatibility.
        """
        normalized = deepcopy(schema)
        return self._normalize_schema_node_for_gemini(normalized)

    def _normalize_schema_node_for_gemini(self, node: Any) -> Any:
        """
        Recursively normalize an individual schema node.
        """
        if isinstance(node, list):
            return [self._normalize_schema_node_for_gemini(item) for item in node]

        if not isinstance(node, dict):
            return node

        normalized: dict[str, Any] = {}
        unsupported_keys = {
            "additional_properties",
            "additionalProperties",
            "default",
            "examples",
            "title",
            "patternProperties",
            "$schema",
            "$defs",
            "defs",
            "anyOf",
            "oneOf",
            "allOf",
        }

        for key, value in node.items():
            if key in unsupported_keys:
                continue

            if key == "type":
                normalized_type = self._normalize_type_for_gemini(value)
                if normalized_type is not None:
                    normalized["type"] = normalized_type
                continue

            if key == "enum" and isinstance(value, list):
                cleaned_enum = [item for item in value if item is not None]
                normalized["enum"] = cleaned_enum
                continue

            if key == "properties" and isinstance(value, dict):
                normalized["properties"] = {
                    prop_name: self._normalize_schema_node_for_gemini(prop_schema)
                    for prop_name, prop_schema in value.items()
                }
                continue

            if key == "items":
                normalized["items"] = self._normalize_schema_node_for_gemini(value)
                continue

            normalized[key] = self._normalize_schema_node_for_gemini(value)

        return normalized

    def _normalize_type_for_gemini(self, value: Any) -> str | None:
        """
        Convert provider-agnostic schema type definitions into the single-type
        enum format expected by Gemini function declarations.
        """
        type_map = {
            "object": "OBJECT",
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "null": "NULL",
        }

        if isinstance(value, str):
            return type_map.get(value.lower(), value)

        if isinstance(value, list):
            non_null_types = [item for item in value if isinstance(item, str) and item.lower() != "null"]
            if non_null_types:
                first_type = non_null_types[0]
                return type_map.get(first_type.lower(), first_type)

            if any(isinstance(item, str) and item.lower() == "null" for item in value):
                return "NULL"

        return None

    def _normalize_response(self, response: Any) -> dict[str, Any]:
        """
        Normalize the Gemini response into the common provider return shape:

        {
            "text": str,
            "tool_call": dict | None,
        }
        """
        text_parts: list[str] = []
        tool_call: dict[str, Any] | None = None
        raw_function_calls = getattr(response, "function_calls", None) or []

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            first_candidate = candidates[0]
            content = getattr(first_candidate, "content", None)
            parts = getattr(content, "parts", None) or []

            for part in parts:
                function_call = getattr(part, "function_call", None)
                if function_call is not None and tool_call is None:
                    args = getattr(function_call, "args", None)
                    if hasattr(args, "to_dict"):
                        args = args.to_dict()
                    elif args is None:
                        args = {}

                    tool_call = {
                        "tool_name": getattr(function_call, "name", ""),
                        "arguments": args if isinstance(args, dict) else dict(args or {}),
                    }

                text_value = getattr(part, "text", None)
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value)

        if tool_call is None and raw_function_calls:
            first_call = raw_function_calls[0]
            args = getattr(first_call, "args", None)
            if hasattr(args, "to_dict"):
                args = args.to_dict()
            elif args is None:
                args = {}

            tool_call = {
                "tool_name": getattr(first_call, "name", ""),
                "arguments": args if isinstance(args, dict) else dict(args or {}),
            }

        if not text_parts:
            response_text = getattr(response, "text", None)
            if isinstance(response_text, str) and response_text.strip():
                text_parts.append(response_text)

        return {
            "text": "\n".join(text_parts).strip(),
            "tool_call": tool_call,
        }