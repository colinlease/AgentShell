from __future__ import annotations

from agents.core.models import ChatMessage


def select_recent_messages(messages: list[ChatMessage], keep_recent: int) -> list[ChatMessage]:
    """
    Return the most recent messages that should remain un-compacted.
    """
    keep = max(0, int(keep_recent))
    if keep == 0:
        return []
    return list(messages[-keep:])


def split_messages_for_compaction(
    messages: list[ChatMessage],
    *,
    keep_recent: int,
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """
    Split messages into an older prefix for compaction and a recent raw suffix.
    """
    keep = max(0, int(keep_recent))
    if keep == 0:
        return list(messages), []
    if len(messages) <= keep:
        return [], list(messages)
    return list(messages[:-keep]), list(messages[-keep:])


def format_chat_messages(messages: list[ChatMessage]) -> str:
    """
    Format chat messages into a compact role-labeled text block.
    """
    normalized_lines: list[str] = []
    for message in messages:
        role = str(message.role or "assistant").strip().lower() or "assistant"
        content = " ".join(str(message.content or "").split()).strip()
        if not content:
            continue
        normalized_lines.append(f"{role}: {content}")
    return "\n".join(normalized_lines)


def format_conversation_context(
    *,
    summary: str | None,
    recent_messages: list[ChatMessage],
) -> str:
    """
    Build one hidden conversation-context payload shared by all phases.
    """
    sections: list[str] = []
    normalized_summary = " ".join(str(summary or "").split()).strip()
    if normalized_summary:
        sections.append("Prior compacted context:\n" + normalized_summary)

    recent_block = format_chat_messages(recent_messages)
    if recent_block:
        sections.append("Recent raw conversation:\n" + recent_block)

    return "\n\n".join(sections).strip()
