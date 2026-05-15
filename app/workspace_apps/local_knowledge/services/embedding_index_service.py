from __future__ import annotations

from typing import Any

from app.workspace_apps.local_knowledge.services.embedding_provider_service import (
    LocalKnowledgeEmbeddingError,
    LocalKnowledgeEmbeddingLimitError,
    LocalKnowledgeEmbeddingProvider,
    get_configured_embedding_provider,
)
from app.workspace_apps.local_knowledge.services.index_store import LocalKnowledgeIndexStore


MAX_EMBED_FILES_PER_CALL = 50


class LocalKnowledgeEmbeddingIndexError(RuntimeError):
    """Raised when Local Knowledge embedding indexing cannot be completed."""


def index_embeddings(
    *,
    root_path: str,
    path: str | None = None,
    limit: int | None = 10,
    provider: LocalKnowledgeEmbeddingProvider | None = None,
) -> dict[str, Any]:
    normalized_path = _normalize_relative_path(path)
    active_provider = provider or get_configured_embedding_provider()
    backend = active_provider.backend
    store = LocalKnowledgeIndexStore(root_path=str(root_path))
    capped_limit = _normalize_limit(limit, default=10, maximum=MAX_EMBED_FILES_PER_CALL)

    if not backend.generation_available:
        return {
            "status": "unavailable",
            "message": backend.availability_message or "Embedding generation is not available.",
            "path": normalized_path or ".",
            "embedding_backend": backend.to_dict(),
            "embedded_files": 0,
            "embedded_chunks": 0,
            "skipped_files": [],
            "remaining_unembedded_chunks": store.count_chunks_missing_embeddings(
                path=normalized_path,
                embedding_provider=backend.provider,
                embedding_model=backend.model,
                chunker_version=backend.chunker_version,
            ),
        }

    candidate_paths = store.list_paths_missing_embeddings(
        path=normalized_path,
        embedding_provider=backend.provider,
        embedding_model=backend.model,
        chunker_version=backend.chunker_version,
        limit=capped_limit,
    )
    embedded_files = 0
    embedded_chunks = 0
    skipped_files: list[dict[str, Any]] = []
    estimated_input_tokens = 0
    total_chars = 0

    for relative_path in candidate_paths:
        chunks = store.get_content_chunks_for_path(relative_path)
        if not chunks:
            skipped_files.append({"path": relative_path, "reason": "No indexed chunks were found."})
            continue
        try:
            embeddings, metrics = _embed_file_chunks(provider=active_provider, chunks=chunks)
        except LocalKnowledgeEmbeddingLimitError as exc:
            skipped_files.append({"path": relative_path, "reason": str(exc)})
            continue
        except LocalKnowledgeEmbeddingError as exc:
            skipped_files.append({"path": relative_path, "reason": str(exc)})
            continue

        if not embeddings:
            skipped_files.append({"path": relative_path, "reason": "No embeddings were generated."})
            continue

        content_hash = str(chunks[0].get("content_hash") or "")
        result = store.replace_content_embeddings(
            relative_path=relative_path,
            content_hash=content_hash,
            embedding_provider=backend.provider,
            embedding_model=backend.model,
            chunker_version=backend.chunker_version,
            embeddings=embeddings,
        )
        stored_count = int(result.get("embedded_chunk_count", 0) or 0)
        if stored_count <= 0:
            skipped_files.append({"path": relative_path, "reason": "Generated embeddings could not be stored."})
            continue
        embedded_files += 1
        embedded_chunks += stored_count
        estimated_input_tokens += int(metrics.get("estimated_input_tokens", 0) or 0)
        total_chars += int(metrics.get("total_chars", 0) or 0)

    remaining_unembedded_chunks = store.count_chunks_missing_embeddings(
        path=normalized_path,
        embedding_provider=backend.provider,
        embedding_model=backend.model,
        chunker_version=backend.chunker_version,
    )
    return {
        "status": "ok",
        "path": normalized_path or ".",
        "embedding_backend": backend.to_dict(),
        "candidate_file_count": len(candidate_paths),
        "embedded_files": embedded_files,
        "embedded_chunks": embedded_chunks,
        "skipped_files": skipped_files,
        "remaining_unembedded_chunks": remaining_unembedded_chunks,
        "estimated_input_tokens": estimated_input_tokens,
        "total_chars": total_chars,
        "embedding_index_complete": remaining_unembedded_chunks == 0,
    }


def _embed_file_chunks(
    *,
    provider: LocalKnowledgeEmbeddingProvider,
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    embeddings: list[dict[str, Any]] = []
    estimated_input_tokens = 0
    total_chars = 0
    for batch in _build_embedding_batches(provider=provider, chunks=chunks):
        texts = [str(chunk.get("chunk_text") or "") for chunk in batch]
        result = provider.embed_texts(texts)
        vectors = list(result.get("vectors", []) or [])
        if len(vectors) != len(batch):
            raise LocalKnowledgeEmbeddingError("Embedding provider returned an unexpected number of vectors.")
        estimated_input_tokens += int(result.get("estimated_input_tokens", 0) or 0)
        total_chars += int(result.get("total_chars", 0) or 0)
        for chunk, vector in zip(batch, vectors):
            embeddings.append(
                {
                    "chunk_index": int(chunk.get("chunk_index") or 0),
                    "embedding_vector": list(vector),
                    "metadata": {
                        "source": "local_knowledge_embedding_index",
                        "chunk_id": str(chunk.get("chunk_id") or ""),
                    },
                }
            )
    return embeddings, {
        "estimated_input_tokens": estimated_input_tokens,
        "total_chars": total_chars,
    }


def _build_embedding_batches(
    *,
    provider: LocalKnowledgeEmbeddingProvider,
    chunks: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    backend = provider.backend
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    max_texts = max(1, int(backend.max_texts_per_call))
    max_total_chars = max(1, int(backend.max_total_chars_per_call))
    max_chars_per_text = max(1, int(backend.max_chars_per_text))

    for chunk in chunks:
        text = str(chunk.get("chunk_text") or "").strip()
        if not text:
            continue
        text_chars = len(text)
        if text_chars > max_chars_per_text:
            raise LocalKnowledgeEmbeddingLimitError(
                f"Chunk {chunk.get('chunk_index')} has {text_chars} characters; "
                f"limit is {max_chars_per_text}."
            )
        if current and (len(current) >= max_texts or current_chars + text_chars > max_total_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += text_chars
    if current:
        batches.append(current)
    return batches


def _normalize_limit(limit: int | None, *, default: int, maximum: int) -> int:
    if limit is None:
        return default
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(value, maximum))


def _normalize_relative_path(path: str | None) -> str:
    value = str(path or "").strip().replace("\\", "/").strip("/")
    if value in {"", "."}:
        return ""
    parts = [part for part in value.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise LocalKnowledgeEmbeddingIndexError("path must be a folder-relative path without '.' or '..' segments.")
    if parts and ":" in parts[0]:
        raise LocalKnowledgeEmbeddingIndexError("path must be relative to the mounted folder, not an absolute path.")
    return "/".join(parts)
