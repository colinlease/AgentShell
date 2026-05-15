from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Protocol

from openai import OpenAI

from app.workspace_apps.local_knowledge.services.embedding_config_service import (
    LocalKnowledgeEmbeddingBackend,
    get_local_knowledge_embedding_backend,
)
from config.settings import get_settings


class LocalKnowledgeEmbeddingError(RuntimeError):
    """Raised when Local Knowledge embeddings cannot be generated."""


class LocalKnowledgeEmbeddingLimitError(LocalKnowledgeEmbeddingError):
    """Raised when an embedding request exceeds configured guardrails."""


class LocalKnowledgeEmbeddingProvider(Protocol):
    backend: LocalKnowledgeEmbeddingBackend

    def embed_texts(self, texts: list[str]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PreparedEmbeddingTexts:
    texts: list[str]
    input_count: int
    total_chars: int
    estimated_tokens: int


class UnavailableEmbeddingProvider:
    def __init__(self, *, backend: LocalKnowledgeEmbeddingBackend, reason: str) -> None:
        self.backend = backend
        self.reason = str(reason or "Embedding generation is unavailable.")

    def embed_texts(self, texts: list[str]) -> dict[str, Any]:
        raise LocalKnowledgeEmbeddingError(self.reason)


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        backend: LocalKnowledgeEmbeddingBackend,
        api_key: str,
        client: Any | None = None,
    ) -> None:
        self.backend = backend
        self.client = client or OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> dict[str, Any]:
        prepared = prepare_embedding_texts(texts=texts, backend=self.backend)
        response = self.client.embeddings.create(
            model=self.backend.model,
            input=prepared.texts,
        )
        vectors = [list(getattr(item, "embedding", []) or []) for item in getattr(response, "data", []) or []]
        if len(vectors) != len(prepared.texts):
            raise LocalKnowledgeEmbeddingError("Embedding provider returned an unexpected number of vectors.")
        embedding_dim = len(vectors[0]) if vectors else 0
        if embedding_dim <= 0 or any(len(vector) != embedding_dim for vector in vectors):
            raise LocalKnowledgeEmbeddingError("Embedding provider returned malformed vectors.")
        return {
            "status": "ok",
            "provider": self.backend.provider,
            "model": self.backend.model,
            "chunker_version": self.backend.chunker_version,
            "embedding_count": len(vectors),
            "embedding_dim": embedding_dim,
            "estimated_input_tokens": prepared.estimated_tokens,
            "input_count": prepared.input_count,
            "total_chars": prepared.total_chars,
            "vectors": vectors,
        }


def get_configured_embedding_provider() -> LocalKnowledgeEmbeddingProvider:
    backend = get_local_knowledge_embedding_backend()
    settings = get_settings()
    if backend.provider == "openai":
        api_key = getattr(settings, "openai_api_key", None)
        if not api_key:
            return UnavailableEmbeddingProvider(
                backend=backend,
                reason="OPENAI_API_KEY is required for Local Knowledge OpenAI embeddings.",
            )
        return OpenAIEmbeddingProvider(backend=backend, api_key=str(api_key))
    return UnavailableEmbeddingProvider(
        backend=backend,
        reason=f"Unsupported Local Knowledge embedding provider: {backend.provider}.",
    )


def prepare_embedding_texts(*, texts: list[str], backend: LocalKnowledgeEmbeddingBackend) -> PreparedEmbeddingTexts:
    normalized_texts = [str(text or "").strip() for text in texts if str(text or "").strip()]
    if not normalized_texts:
        raise LocalKnowledgeEmbeddingLimitError("At least one non-empty text is required for embedding.")
    if len(normalized_texts) > backend.max_texts_per_call:
        raise LocalKnowledgeEmbeddingLimitError(
            f"Embedding request has {len(normalized_texts)} texts; limit is {backend.max_texts_per_call}. "
            "Index a smaller folder or lower the indexing limit."
        )

    total_chars = 0
    for index, text in enumerate(normalized_texts, start=1):
        text_chars = len(text)
        if text_chars > backend.max_chars_per_text:
            raise LocalKnowledgeEmbeddingLimitError(
                f"Embedding text {index} has {text_chars} characters; limit is {backend.max_chars_per_text}. "
                "Use smaller chunks before embedding."
            )
        total_chars += text_chars

    if total_chars > backend.max_total_chars_per_call:
        raise LocalKnowledgeEmbeddingLimitError(
            f"Embedding request has {total_chars} total characters; limit is {backend.max_total_chars_per_call}. "
            "Index a smaller folder or lower the indexing limit."
        )

    return PreparedEmbeddingTexts(
        texts=normalized_texts,
        input_count=len(normalized_texts),
        total_chars=total_chars,
        estimated_tokens=estimate_embedding_tokens(normalized_texts),
    )


def estimate_embedding_tokens(texts: list[str]) -> int:
    total_chars = sum(len(str(text or "")) for text in texts)
    return max(1, ceil(total_chars / 4))
