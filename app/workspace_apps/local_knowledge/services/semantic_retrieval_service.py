from __future__ import annotations

import math
from typing import Any

from app.workspace_apps.local_knowledge.services.embedding_provider_service import (
    LocalKnowledgeEmbeddingError,
    LocalKnowledgeEmbeddingProvider,
    get_configured_embedding_provider,
)
from app.workspace_apps.local_knowledge.services.index_store import LocalKnowledgeIndexStore
from app.workspace_apps.local_knowledge.services.retrieval_service import build_snippet, query_terms


MAX_SEMANTIC_RESULTS = 50
DEFAULT_SEMANTIC_CANDIDATES = 5000


class LocalKnowledgeSemanticSearchError(RuntimeError):
    """Raised when Local Knowledge semantic retrieval cannot be completed."""


def semantic_search_content(
    *,
    root_path: str,
    query: str,
    path: str | None = None,
    limit: int | None = 10,
    candidate_limit: int | None = DEFAULT_SEMANTIC_CANDIDATES,
    provider: LocalKnowledgeEmbeddingProvider | None = None,
) -> dict[str, Any]:
    normalized_path = _normalize_relative_path(path)
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {
            "status": "error",
            "message": "Enter a non-empty semantic search query.",
            "results": [],
        }

    active_provider = provider or get_configured_embedding_provider()
    backend = active_provider.backend
    store = LocalKnowledgeIndexStore(root_path=str(root_path))
    normalized_limit = _normalize_limit(limit, default=10, maximum=MAX_SEMANTIC_RESULTS)
    normalized_candidate_limit = _normalize_limit(
        candidate_limit,
        default=DEFAULT_SEMANTIC_CANDIDATES,
        maximum=DEFAULT_SEMANTIC_CANDIDATES,
    )

    if not backend.generation_available:
        return {
            "status": "unavailable",
            "message": backend.availability_message or "Embedding generation is not available.",
            "query": normalized_query,
            "path": normalized_path or ".",
            "embedding_backend": backend.to_dict(),
            "result_count": 0,
            "candidate_count": 0,
            "results": [],
        }

    try:
        query_vector = _embed_query(provider=active_provider, query=normalized_query)
    except LocalKnowledgeEmbeddingError as exc:
        return {
            "status": "error",
            "message": str(exc),
            "query": normalized_query,
            "path": normalized_path or ".",
            "embedding_backend": backend.to_dict(),
            "result_count": 0,
            "candidate_count": 0,
            "results": [],
        }

    candidates = store.list_semantic_candidate_chunks(
        path=normalized_path,
        embedding_provider=backend.provider,
        embedding_model=backend.model,
        chunker_version=backend.chunker_version,
        limit=normalized_candidate_limit,
    )
    ranked = _rank_semantic_candidates(query_vector=query_vector, candidates=candidates)
    terms = query_terms(normalized_query)
    results = [
        {
            "path": str(row.get("relative_path") or ""),
            "chunk_index": int(row.get("chunk_index") or 0),
            "score": float(row.get("semantic_score") or 0.0),
            "semantic_score": float(row.get("semantic_score") or 0.0),
            "snippet": build_snippet(str(row.get("chunk_text") or ""), terms=terms),
            "char_start": int(row.get("char_start") or 0),
            "char_end": int(row.get("char_end") or 0),
            "chunk_id": str(row.get("chunk_id") or ""),
            "retrieval_backend": "semantic_cosine",
        }
        for row in ranked[:normalized_limit]
    ]
    return {
        "status": "ok",
        "query": normalized_query,
        "path": normalized_path or ".",
        "embedding_backend": backend.to_dict(),
        "result_count": len(results),
        "candidate_count": len(candidates),
        "results": results,
    }


def _embed_query(*, provider: LocalKnowledgeEmbeddingProvider, query: str) -> list[float]:
    result = provider.embed_texts([query])
    vectors = list(result.get("vectors", []) or [])
    if len(vectors) != 1:
        raise LocalKnowledgeEmbeddingError("Embedding provider returned an unexpected number of query vectors.")
    vector = _normalize_vector(vectors[0])
    if not vector:
        raise LocalKnowledgeEmbeddingError("Embedding provider returned an empty query vector.")
    return vector


def _rank_semantic_candidates(
    *,
    query_vector: list[float],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_vector = _normalize_vector(candidate.get("embedding_vector"))
        score = _cosine_similarity(query_vector, candidate_vector)
        if score is None:
            continue
        row = dict(candidate)
        row["semantic_score"] = score
        ranked.append(row)
    ranked.sort(
        key=lambda row: (
            -float(row.get("semantic_score") or 0.0),
            str(row.get("relative_path") or "").lower(),
            int(row.get("chunk_index") or 0),
        )
    )
    return ranked


def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return None
    return dot / (left_norm * right_norm)


def _normalize_vector(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    vector: list[float] = []
    for item in value:
        try:
            vector.append(float(item))
        except (TypeError, ValueError):
            return []
    return vector


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
        raise LocalKnowledgeSemanticSearchError("path must be a folder-relative path without '.' or '..' segments.")
    if parts and ":" in parts[0]:
        raise LocalKnowledgeSemanticSearchError("path must be relative to the mounted folder, not an absolute path.")
    return "/".join(parts)
