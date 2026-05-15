from __future__ import annotations

import re
from typing import Any

from app.workspace_apps.local_knowledge.services.chunking_service import (
    build_chunks as build_text_chunks,
)
from app.workspace_apps.local_knowledge.services.content_extraction_service import (
    LocalKnowledgeContentError,
    extract_searchable_text,
)
from app.workspace_apps.local_knowledge.services.embedding_config_service import (
    get_local_knowledge_embedding_backend,
)
from app.workspace_apps.local_knowledge.services.index_store import LocalKnowledgeIndexStore
from app.workspace_apps.local_knowledge.services.retrieval_service import (
    build_snippet as build_search_snippet,
    query_terms as parse_query_terms,
    search_keyword_chunks,
)


MAX_INDEX_FILES_PER_CALL = 200
SEARCHABLE_KINDS = {"text", "document"}
SEARCHABLE_DATASET_EXTENSIONS = {".csv"}


class LocalKnowledgeSearchError(RuntimeError):
    """Raised when Local Knowledge search indexing cannot be completed."""


def index_content(
    *,
    root_path: str,
    path: str | None = None,
    limit: int | None = 50,
) -> dict[str, Any]:
    normalized_path = _normalize_relative_path(path)
    store = LocalKnowledgeIndexStore(root_path=str(root_path))
    files = store.list_files(path=normalized_path, include_missing=False, limit=None)
    indexed_paths = store.get_indexed_content_paths()
    candidates = [
        record
        for record in files
        if _is_searchable_record(record) and str(record.get("relative_path") or "") not in indexed_paths
    ]
    capped_limit = _normalize_limit(limit, default=50, maximum=MAX_INDEX_FILES_PER_CALL)
    candidates = candidates[:capped_limit]

    indexed_files = 0
    indexed_chunks = 0
    skipped_files: list[dict[str, Any]] = []

    for record in candidates:
        relative_path = str(record.get("relative_path") or "")
        try:
            extraction = extract_searchable_text(root_path=root_path, relative_path=relative_path)
        except LocalKnowledgeContentError as exc:
            skipped_files.append({"path": relative_path, "reason": str(exc)})
            continue

        text = _normalize_search_text(str(extraction.get("content") or ""))
        if not text:
            skipped_files.append({"path": relative_path, "reason": "No searchable text was extracted."})
            continue

        chunks = _build_chunks(text)
        result = store.replace_content_chunks(
            relative_path=relative_path,
            content_hash=str(extraction.get("content_hash") or record.get("content_hash") or ""),
            chunks=chunks,
        )
        indexed_files += 1
        indexed_chunks += int(result.get("indexed_chunk_count", 0))

    summary = store.get_summary()
    return {
        "status": "ok",
        "indexed_files": indexed_files,
        "indexed_chunks": indexed_chunks,
        "skipped_files": skipped_files,
        "remaining_unindexed_candidates": len(_remaining_searchable_records(store, path=normalized_path)),
        "search_index": get_search_index_status(root_path=root_path, path=normalized_path),
        "summary": summary,
    }


def search_content(
    *,
    root_path: str,
    query: str,
    path: str | None = None,
    limit: int | None = 10,
) -> dict[str, Any]:
    normalized_path = _normalize_relative_path(path)
    terms = _query_terms(query)
    if not terms:
        return {
            "status": "error",
            "message": "Enter a non-empty search query.",
            "results": [],
        }
    store = LocalKnowledgeIndexStore(root_path=str(root_path))
    index_status = get_search_index_status(root_path=root_path, path=normalized_path)
    rows = search_keyword_chunks(
        store=store,
        query_terms=terms,
        path=normalized_path,
        limit=_normalize_limit(limit, default=10, maximum=50),
    )
    results = [
        {
            "path": str(row.get("relative_path") or ""),
            "chunk_index": int(row.get("chunk_index") or 0),
            "score": int(row.get("score") or 0),
            "snippet": _build_snippet(str(row.get("chunk_text") or ""), terms=terms),
            "char_start": int(row.get("char_start") or 0),
            "char_end": int(row.get("char_end") or 0),
        }
        for row in rows
    ]
    return {
        "status": "ok",
        "query": str(query),
        "path": normalized_path or ".",
        "terms": terms,
        "result_count": len(results),
        "index_recommended": len(results) == 0 and int(index_status.get("unindexed_searchable_file_count", 0)) > 0,
        "search_index": index_status,
        "results": results,
    }


def get_search_index_status(*, root_path: str, path: str | None = None) -> dict[str, Any]:
    normalized_path = _normalize_relative_path(path)
    store = LocalKnowledgeIndexStore(root_path=str(root_path))
    searchable_records = [
        record
        for record in store.list_files(path=normalized_path, include_missing=False, limit=None)
        if _is_searchable_record(record)
    ]
    searchable_paths = {str(record.get("relative_path") or "") for record in searchable_records}
    indexed_paths = store.get_indexed_content_paths()
    indexed_searchable_paths = searchable_paths & indexed_paths
    unindexed_paths = sorted(searchable_paths - indexed_paths)
    summary = store.get_summary()
    embedding_backend = get_local_knowledge_embedding_backend()
    embedded_paths = store.get_embedded_content_paths(
        path=normalized_path,
        embedding_provider=embedding_backend.provider,
        embedding_model=embedding_backend.model,
        chunker_version=embedding_backend.chunker_version,
    )
    embedded_searchable_paths = searchable_paths & embedded_paths
    unembedded_paths = sorted(searchable_paths - embedded_paths)
    embedding_summary = store.get_embedding_summary(
        path=normalized_path,
        embedding_provider=embedding_backend.provider,
        embedding_model=embedding_backend.model,
        chunker_version=embedding_backend.chunker_version,
    )
    embedding_count = int(embedding_summary.get("embedding_count", 0) or 0)
    semantic_search_available = bool(embedding_backend.generation_available and embedding_count > 0)
    embedding_backend_status = embedding_backend.to_dict()
    embedding_backend_status["semantic_search_available"] = semantic_search_available
    return {
        "path": normalized_path or ".",
        "searchable_file_count": len(searchable_paths),
        "indexed_searchable_file_count": len(indexed_searchable_paths),
        "unindexed_searchable_file_count": len(unindexed_paths),
        "content_chunk_count": int(summary.get("content_chunk_count", 0) or 0),
        "index_complete": len(unindexed_paths) == 0,
        "sample_unindexed_paths": unindexed_paths[:10],
        "embedding_backend": embedding_backend_status,
        "embedding_index": {
            "embedded_searchable_file_count": len(embedded_searchable_paths),
            "unembedded_searchable_file_count": len(unembedded_paths),
            "embedding_count": embedding_count,
            "semantic_index_complete": len(unembedded_paths) == 0 and bool(searchable_paths),
            "semantic_search_available": semantic_search_available,
            "sample_unembedded_paths": unembedded_paths[:10],
        },
    }


def _remaining_searchable_records(store: LocalKnowledgeIndexStore, *, path: str | None) -> list[dict[str, Any]]:
    indexed_paths = store.get_indexed_content_paths()
    return [
        record
        for record in store.list_files(path=path, include_missing=False, limit=None)
        if _is_searchable_record(record) and str(record.get("relative_path") or "") not in indexed_paths
    ]


def _is_searchable_record(record: dict[str, Any]) -> bool:
    if str(record.get("support_status") or "") != "supported":
        return False
    extension = str(record.get("extension") or "")
    if extension in SEARCHABLE_DATASET_EXTENSIONS:
        return True
    return str(record.get("kind") or "") in SEARCHABLE_KINDS


def _build_chunks(text: str) -> list[dict[str, Any]]:
    return build_text_chunks(text)


def _query_terms(query: str) -> list[str]:
    return parse_query_terms(query)


def _build_snippet(text: str, *, terms: list[str]) -> str:
    return build_search_snippet(text, terms=terms)


def _normalize_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


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
        raise LocalKnowledgeSearchError("path must be a folder-relative path without '.' or '..' segments.")
    if parts and ":" in parts[0]:
        raise LocalKnowledgeSearchError("path must be relative to the mounted folder, not an absolute path.")
    return "/".join(parts)
