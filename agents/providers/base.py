

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agents.core.models import ChatMessage


class BaseLLMProvider(ABC):
    """
    Provider-agnostic interface for model backends.

    Concrete providers (for example OpenAI or Gemini) should implement this
    interface and return a normalized response dictionary so the rest of the
    agent framework does not depend on provider-specific SDK formats.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a normalized model response.

        Expected normalized response shapes include examples like:

        Plain text response:
        {
            "text": "Here is what I found.",
            "tool_call": None,
        }

        Tool request response:
        {
            "text": "",
            "tool_call": {
                "tool_name": "run_basic_analysis",
                "arguments": {"columns": ["sales", "profit"]},
            },
        }
        """
        raise NotImplementedError