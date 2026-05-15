from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import get_settings


@dataclass(frozen=True)
class LocalKnowledgeEmbeddingBackend:
    provider: str
    model: str
    chunker_version: str
    max_texts_per_call: int
    max_chars_per_text: int
    max_total_chars_per_call: int
    source: str = "local_knowledge_config"
    follows_shell_model: bool = False
    generation_available: bool = False
    semantic_search_available: bool = False
    availability_status: str = "unavailable"
    availability_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "chunker_version": self.chunker_version,
            "follows_shell_model": self.follows_shell_model,
            "generation_available": self.generation_available,
            "semantic_search_available": self.semantic_search_available,
            "availability_status": self.availability_status,
            "availability_message": self.availability_message,
            "requires_reindex_if_changed": True,
            "guardrails": {
                "max_texts_per_call": self.max_texts_per_call,
                "max_chars_per_text": self.max_chars_per_text,
                "max_total_chars_per_call": self.max_total_chars_per_call,
                "estimated_chars_per_token": 4,
            },
        }


def get_local_knowledge_embedding_backend() -> LocalKnowledgeEmbeddingBackend:
    """
    Return the embedding backend identity for Local Knowledge.

    The shell's top-level provider/model controls chat reasoning. Local Knowledge
    embeddings are configured separately because embedding models have different
    price/performance tradeoffs and are not interchangeable with chat models.
    """
    settings = get_settings()
    provider = _normalize_setting(getattr(settings, "local_knowledge_embedding_provider", ""), default="local").lower()
    model = _normalize_setting(getattr(settings, "local_knowledge_embedding_model", ""), default="hashing-v1")
    availability = _embedding_availability(provider=provider, settings=settings)
    return LocalKnowledgeEmbeddingBackend(
        provider=provider,
        model=model,
        chunker_version=_normalize_setting(
            getattr(settings, "local_knowledge_embedding_chunker_version", ""),
            default="keyword-window-v1",
        ),
        max_texts_per_call=_normalize_positive_int(
            getattr(settings, "local_knowledge_embedding_max_texts_per_call", None),
            default=32,
        ),
        max_chars_per_text=_normalize_positive_int(
            getattr(settings, "local_knowledge_embedding_max_chars_per_text", None),
            default=8000,
        ),
        max_total_chars_per_call=_normalize_positive_int(
            getattr(settings, "local_knowledge_embedding_max_total_chars_per_call", None),
            default=64000,
        ),
        generation_available=availability["generation_available"],
        availability_status=availability["availability_status"],
        availability_message=availability["availability_message"],
    )


def _normalize_setting(value: Any, *, default: str) -> str:
    normalized = str(value or "").strip()
    return normalized or default


def _normalize_positive_int(value: Any, *, default: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, normalized)


def _embedding_availability(*, provider: str, settings: Any) -> dict[str, Any]:
    if provider == "openai":
        if not getattr(settings, "openai_api_key", None):
            return {
                "generation_available": False,
                "availability_status": "missing_api_key",
                "availability_message": "OPENAI_API_KEY is required for Local Knowledge OpenAI embeddings.",
            }
        return {
            "generation_available": True,
            "availability_status": "available",
            "availability_message": "OpenAI embeddings are configured separately from the shell chat model.",
        }
    return {
        "generation_available": False,
        "availability_status": "unsupported_provider",
        "availability_message": (
            "Only the Local Knowledge OpenAI embedding backend is currently implemented; "
            "keyword search remains available."
        ),
    }
