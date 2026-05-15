from __future__ import annotations

from agents.core.models import ChatMessage
from agents.core.runtime_config import validate_runtime_budget_value
from agents.prompts.runtime_prompts import build_compaction_prompt
from agents.providers.base import BaseLLMProvider
from agents.runtime.context_builder import format_chat_messages, split_messages_for_compaction


class ConversationCompactionService:
    """
    Hidden compaction service for summarizing older conversation history.
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    def should_compact(
        self,
        *,
        messages: list[ChatMessage],
        trigger_message_count: int,
        keep_recent: int,
        trigger_character_count: int | None = None,
    ) -> bool:
        trigger = max(1, int(trigger_message_count))
        keep = max(0, int(keep_recent))
        message_threshold_reached = len(messages) >= trigger and len(messages) > keep

        if trigger_character_count is None or int(trigger_character_count) <= 0:
            return message_threshold_reached

        character_threshold_reached = (
            sum(len(str(message.content or "")) for message in messages) >= int(trigger_character_count)
            and len(messages) > keep
        )
        return message_threshold_reached or character_threshold_reached

    def split_for_compaction(
        self,
        messages: list[ChatMessage],
        *,
        keep_recent: int,
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        return split_messages_for_compaction(messages, keep_recent=keep_recent)

    def compact(
        self,
        *,
        older_messages: list[ChatMessage],
        existing_summary: str | None,
        app_prompt: str,
        max_summary_chars: int | None = None,
        max_model_calls: int | None = None,
    ) -> str:
        if max_model_calls is not None:
            validate_runtime_budget_value("compaction_max_model_calls", max_model_calls)

        prompt = build_compaction_prompt(app_prompt=app_prompt)
        user_message = self._build_compaction_user_message(
            older_messages=older_messages,
            existing_summary=existing_summary,
            max_summary_chars=max_summary_chars,
        )
        response = self.provider.generate(
            messages=[ChatMessage(role="user", content=user_message)],
            system_prompt=prompt,
            tools=None,
        )
        summary = " ".join(str(response.get("text", "")).split()).strip()
        if max_summary_chars is not None and max_summary_chars > 0:
            return summary[: int(max_summary_chars)].strip()
        return summary

    @staticmethod
    def _build_compaction_user_message(
        *,
        older_messages: list[ChatMessage],
        existing_summary: str | None,
        max_summary_chars: int | None,
    ) -> str:
        instructions = [
            "Summarize the older conversation history into one compact hidden context block.",
            "Preserve user preferences, established facts, important names/ids/columns/entities, constraints, unresolved questions, and overall objectives.",
            "Keep the overall flow summary brief.",
            "Do not include low-value repetition or social filler.",
        ]
        if max_summary_chars is not None and max_summary_chars > 0:
            instructions.append(f"Keep the summary under approximately {int(max_summary_chars)} characters.")

        existing = " ".join(str(existing_summary or "").split()).strip()
        existing_block = existing or "none"
        older_block = format_chat_messages(older_messages) or "none"
        return (
            "\n".join(instructions)
            + f"\n\nExisting summary:\n{existing_block}\n\nOlder messages to compact:\n{older_block}"
        )
