from __future__ import annotations

import re
from typing import Any


def query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for term in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_'-]{1,}", str(query or "").lower()):
        if term not in terms:
            terms.append(term)
    return terms[:8]


def search_keyword_chunks(
    *,
    store: Any,
    query_terms: list[str],
    path: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    normalized_limit = max(1, int(limit))
    ranked = _search_keyword_chunks_fts(
        store=store,
        query_terms=query_terms,
        path=path,
        limit=normalized_limit,
    )
    fallback_ranked = _search_keyword_chunks_like(
        store=store,
        query_terms=query_terms,
        path=path,
    )
    if not ranked:
        return fallback_ranked[:normalized_limit]

    seen_chunk_ids = {str(row.get("chunk_id") or "") for row in ranked}
    for row in fallback_ranked:
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        row["retrieval_backend"] = "like"
        ranked.append(row)
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        if len(ranked) >= normalized_limit:
            break
    return ranked[:normalized_limit]


def build_snippet(text: str, *, terms: list[str]) -> str:
    lowered = text.lower()
    hit_positions = [lowered.find(term.lower()) for term in terms if lowered.find(term.lower()) >= 0]
    if not hit_positions:
        return text[:360].strip()
    start = max(0, min(hit_positions) - 120)
    end = min(len(text), start + 420)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _rank_chunk(row: dict[str, Any], *, query_terms: list[str]) -> dict[str, Any]:
    text = str(row.get("chunk_text") or "")
    lowered = text.lower()
    score = 0
    for term in query_terms:
        score += lowered.count(term.lower())
    row["score"] = score
    return row


def _search_keyword_chunks_fts(
    *,
    store: Any,
    query_terms: list[str],
    path: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    search_fts = getattr(store, "search_content_chunks_fts", None)
    if not callable(search_fts):
        return []
    rows = search_fts(query_terms=query_terms, path=path, limit=limit)
    ranked: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = _rank_chunk(dict(row), query_terms=query_terms)
        if int(payload.get("score") or 0) <= 0:
            payload["score"] = max(1, len(rows) - index)
        payload.setdefault("retrieval_backend", "fts5")
        ranked.append(payload)
    return ranked


def _search_keyword_chunks_like(
    *,
    store: Any,
    query_terms: list[str],
    path: str | None,
) -> list[dict[str, Any]]:
    rows = store.find_content_chunks_by_terms(query_terms=query_terms, path=path)
    ranked = [_rank_chunk(dict(row), query_terms=query_terms) for row in rows]
    ranked.sort(key=lambda row: (-int(row["score"]), str(row["relative_path"]).lower(), int(row["chunk_index"])))
    for row in ranked:
        row.setdefault("retrieval_backend", "like")
    return ranked
