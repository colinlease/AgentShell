from __future__ import annotations

from dataclasses import dataclass, field


MAX_NOTES_PER_FILE = 20
MAX_NOTE_TITLE_LENGTH = 80
MAX_NOTE_STATEMENT_LENGTH = 200
MAX_NOTE_TAGS = 8
MAX_NOTE_KEYWORDS = 12
DEFAULT_NOTE_CONFIDENCE = 0.5


@dataclass(frozen=True)
class RuntimeNote:
    """
    Compact persistent heuristic note used by the hidden runtime.
    """

    note_id: str
    title: str
    statement: str
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    confidence: float = DEFAULT_NOTE_CONFIDENCE
    updated_at: str = ""


@dataclass(frozen=True)
class RuntimeNoteFile:
    """
    One bounded note container stored on disk.
    """

    scope: str
    app_id: str | None
    version: int
    notes: list[RuntimeNote] = field(default_factory=list)
