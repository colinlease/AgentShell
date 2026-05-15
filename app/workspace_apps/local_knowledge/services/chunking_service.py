from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_CHUNK_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200
DEFAULT_CHUNKER_VERSION = "keyword-window-v1"


class LocalKnowledgeChunker(Protocol):
    version: str

    def build_chunks(self, text: str) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class FixedWindowChunker:
    """
    Preserve the original Local Knowledge chunking behavior behind a swappable
    interface so later RAG chunkers can be added without changing search tools.
    """

    chunk_chars: int = DEFAULT_CHUNK_CHARS
    overlap_chars: int = CHUNK_OVERLAP_CHARS
    version: str = DEFAULT_CHUNKER_VERSION

    def build_chunks(self, text: str) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        start = 0
        chunk_index = 0
        text_length = len(text)
        chunk_chars = max(1, int(self.chunk_chars))
        overlap_chars = max(0, int(self.overlap_chars))

        while start < text_length:
            end = min(text_length, start + chunk_chars)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "chunk_text": chunk_text,
                        "char_start": start,
                        "char_end": end,
                    }
                )
                chunk_index += 1
            if end >= text_length:
                break
            start = max(end - overlap_chars, start + 1)
        return chunks


def get_default_chunker() -> LocalKnowledgeChunker:
    return FixedWindowChunker()


def build_chunks(text: str, *, chunker: LocalKnowledgeChunker | None = None) -> list[dict[str, Any]]:
    active_chunker = chunker or get_default_chunker()
    return active_chunker.build_chunks(str(text or ""))
