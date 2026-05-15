from __future__ import annotations

from agents.notes.models import (
    DEFAULT_NOTE_CONFIDENCE,
    MAX_NOTES_PER_FILE,
    MAX_NOTE_KEYWORDS,
    MAX_NOTE_STATEMENT_LENGTH,
    MAX_NOTE_TAGS,
    MAX_NOTE_TITLE_LENGTH,
    RuntimeNote,
)


def _normalize_string_list(values: object, *, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []

    normalized: list[str] = []
    for value in values:
        item = " ".join(str(value).strip().split())
        if not item or item in normalized:
            continue
        normalized.append(item[:64])
        if len(normalized) >= limit:
            break

    return normalized


def _normalize_confidence(value: object) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = DEFAULT_NOTE_CONFIDENCE

    return round(max(0.0, min(normalized, 1.0)), 2)


def build_note_fingerprint(*, title: str, statement: str) -> str:
    normalized_title = " ".join(str(title).strip().lower().split())
    normalized_statement = " ".join(str(statement).strip().lower().split())
    return normalized_statement or normalized_title


def normalize_note_payload(note_payload: object) -> RuntimeNote | None:
    """
    Normalize an untrusted note payload into a bounded RuntimeNote.
    """
    if not isinstance(note_payload, dict):
        return None

    note_id = " ".join(str(note_payload.get("note_id", "")).strip().split())
    title = " ".join(str(note_payload.get("title", "")).strip().split())
    statement = " ".join(str(note_payload.get("statement", "")).strip().split())
    updated_at = " ".join(str(note_payload.get("updated_at", "")).strip().split())

    if not note_id or not title or not statement:
        return None

    return RuntimeNote(
        note_id=note_id[:64],
        title=title[:MAX_NOTE_TITLE_LENGTH],
        statement=statement[:MAX_NOTE_STATEMENT_LENGTH],
        tags=_normalize_string_list(note_payload.get("tags", []), limit=MAX_NOTE_TAGS),
        keywords=_normalize_string_list(note_payload.get("keywords", []), limit=MAX_NOTE_KEYWORDS),
        confidence=_normalize_confidence(note_payload.get("confidence")),
        updated_at=updated_at[:64],
    )


def validate_note_count(note_count: int) -> None:
    """
    Enforce the per-file note cap.
    """
    if int(note_count) > MAX_NOTES_PER_FILE:
        raise ValueError(
            f"Notes files may contain at most {MAX_NOTES_PER_FILE} notes."
        )
