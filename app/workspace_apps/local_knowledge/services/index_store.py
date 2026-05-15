from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from config.settings import PROJECT_ROOT


INDEX_ROOT = PROJECT_ROOT / "runtime" / "local_knowledge" / "indexes"


def build_root_id(root_path: str) -> str:
    normalized = str(root_path or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def get_index_db_path(root_path: str) -> Path:
    root_id = build_root_id(root_path)
    return INDEX_ROOT / root_id / "local_knowledge.sqlite"


class LocalKnowledgeIndexStore:
    def __init__(self, *, root_path: str) -> None:
        self.root_path = str(root_path)
        self.root_id = build_root_id(self.root_path)
        self.db_path = get_index_db_path(self.root_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_roots (
                    root_id TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    last_inventory_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    root_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    absolute_path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    support_status TEXT NOT NULL,
                    parse_status TEXT NOT NULL,
                    dataset_candidate INTEGER NOT NULL,
                    indexed_at TEXT,
                    missing INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    UNIQUE(root_id, relative_path)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lkw_files_root_path ON files(root_id, relative_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lkw_files_root_missing ON files(root_id, missing)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS content_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    root_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL,
                    UNIQUE(root_id, relative_path, chunk_index)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lkw_chunks_root_path ON content_chunks(root_id, relative_path)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lkw_chunks_root_hash ON content_chunks(root_id, content_hash)"
            )
            fts_preexisting = _table_exists(conn, "content_chunks_fts")
            _ensure_content_chunks_fts(conn)
            if not fts_preexisting and _table_exists(conn, "content_chunks_fts"):
                _rebuild_content_chunks_fts(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS content_embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    root_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding_provider TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_vector TEXT NOT NULL,
                    chunker_version TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedded_at TEXT NOT NULL,
                    UNIQUE(root_id, chunk_id, embedding_provider, embedding_model, chunker_version)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lkw_embeddings_root_path ON content_embeddings(root_id, relative_path)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lkw_embeddings_root_chunk ON content_embeddings(root_id, chunk_id)"
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_lkw_embeddings_provider_model
                ON content_embeddings(root_id, embedding_provider, embedding_model, chunker_version)
                """
            )

    def upsert_root(self) -> None:
        now = _utc_now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT root_id FROM workspace_roots WHERE root_id = ?",
                (self.root_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE workspace_roots SET root_path = ?, last_seen_at = ? WHERE root_id = ?",
                    (self.root_path, now, self.root_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO workspace_roots (root_id, root_path, created_at, last_seen_at, last_inventory_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (self.root_id, self.root_path, now, now, None),
                )

    def replace_inventory(self, file_records: list[dict[str, Any]]) -> dict[str, Any]:
        now = _utc_now()
        seen_paths = {str(record["relative_path"]) for record in file_records}
        existing_records = {
            str(row["relative_path"]): dict(row)
            for row in self.list_files(include_missing=True, limit=None)
        }

        added = 0
        changed = 0
        unchanged = 0
        restored = 0
        deleted = 0

        with self._connect() as conn:
            for record in file_records:
                relative_path = str(record["relative_path"])
                existing = existing_records.get(relative_path)
                file_id = _build_file_id(self.root_id, relative_path)
                values = (
                    file_id,
                    self.root_id,
                    relative_path,
                    str(record["absolute_path"]),
                    str(record["name"]),
                    str(record["extension"]),
                    int(record["size_bytes"]),
                    int(record["mtime_ns"]),
                    str(record["content_hash"]),
                    str(record["kind"]),
                    str(record["support_status"]),
                    str(record["parse_status"]),
                    1 if bool(record["dataset_candidate"]) else 0,
                    now,
                    0,
                    str(record.get("error_message") or ""),
                )
                conn.execute(
                    """
                    INSERT INTO files (
                        file_id, root_id, relative_path, absolute_path, name, extension, size_bytes,
                        mtime_ns, content_hash, kind, support_status, parse_status, dataset_candidate,
                        indexed_at, missing, error_message
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(root_id, relative_path) DO UPDATE SET
                        absolute_path = excluded.absolute_path,
                        name = excluded.name,
                        extension = excluded.extension,
                        size_bytes = excluded.size_bytes,
                        mtime_ns = excluded.mtime_ns,
                        content_hash = excluded.content_hash,
                        kind = excluded.kind,
                        support_status = excluded.support_status,
                        parse_status = excluded.parse_status,
                        dataset_candidate = excluded.dataset_candidate,
                        indexed_at = excluded.indexed_at,
                        missing = excluded.missing,
                        error_message = excluded.error_message
                    """,
                    values,
                )
                if existing is None:
                    added += 1
                elif int(existing.get("missing", 0)) == 1:
                    restored += 1
                    self._delete_indexed_content_for_path(conn, relative_path)
                elif _file_changed(existing, record):
                    changed += 1
                    self._delete_indexed_content_for_path(conn, relative_path)
                else:
                    unchanged += 1

            for relative_path, existing in existing_records.items():
                if relative_path in seen_paths or int(existing.get("missing", 0)) == 1:
                    continue
                deleted += 1
                conn.execute(
                    "UPDATE files SET missing = 1, indexed_at = ? WHERE root_id = ? AND relative_path = ?",
                    (now, self.root_id, relative_path),
                )
                self._delete_indexed_content_for_path(conn, relative_path)

            conn.execute(
                "UPDATE workspace_roots SET last_inventory_at = ?, last_seen_at = ? WHERE root_id = ?",
                (now, now, self.root_id),
            )

        summary = self.get_summary()
        summary.update(
            {
                "added_files": added,
                "changed_files": changed,
                "unchanged_files": unchanged,
                "restored_files": restored,
                "deleted_files": deleted,
                "last_inventory_at": now,
            }
        )
        return summary

    def get_summary(self) -> dict[str, Any]:
        self.initialize()
        self.upsert_root()
        with self._connect() as conn:
            root = conn.execute(
                "SELECT last_inventory_at FROM workspace_roots WHERE root_id = ?",
                (self.root_id,),
            ).fetchone()
            counts = conn.execute(
                """
                SELECT
                    COUNT(CASE WHEN missing = 0 THEN 1 END) AS file_count,
                    COUNT(CASE WHEN missing = 0 AND support_status = 'supported' THEN 1 END) AS supported_file_count,
                    COUNT(CASE WHEN missing = 0 AND support_status = 'unsupported' THEN 1 END) AS unsupported_file_count,
                    COUNT(CASE WHEN missing = 0 AND dataset_candidate = 1 THEN 1 END) AS dataset_file_count,
                    COUNT(CASE WHEN missing = 1 THEN 1 END) AS deleted_file_count
                FROM files
                WHERE root_id = ?
                """,
                (self.root_id,),
            ).fetchone()
            chunk_counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS content_chunk_count,
                    COUNT(DISTINCT relative_path) AS content_indexed_file_count
                FROM content_chunks
                WHERE root_id = ?
                """,
                (self.root_id,),
            ).fetchone()
            embedding_counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS embedding_count,
                    COUNT(DISTINCT relative_path) AS embedding_indexed_file_count
                FROM content_embeddings
                WHERE root_id = ?
                """,
                (self.root_id,),
            ).fetchone()
        return {
            "root_id": self.root_id,
            "root_path": self.root_path,
            "file_count": int(counts["file_count"] or 0) if counts else 0,
            "supported_file_count": int(counts["supported_file_count"] or 0) if counts else 0,
            "unsupported_file_count": int(counts["unsupported_file_count"] or 0) if counts else 0,
            "dataset_file_count": int(counts["dataset_file_count"] or 0) if counts else 0,
            "deleted_file_count": int(counts["deleted_file_count"] or 0) if counts else 0,
            "content_indexed_file_count": int(chunk_counts["content_indexed_file_count"] or 0) if chunk_counts else 0,
            "content_chunk_count": int(chunk_counts["content_chunk_count"] or 0) if chunk_counts else 0,
            "embedding_indexed_file_count": int(embedding_counts["embedding_indexed_file_count"] or 0)
            if embedding_counts
            else 0,
            "embedding_count": int(embedding_counts["embedding_count"] or 0) if embedding_counts else 0,
            "last_inventory_at": str(root["last_inventory_at"] or "") if root else "",
        }

    def list_files(
        self,
        *,
        path: str | None = None,
        include_missing: bool = False,
        limit: int | None = 200,
    ) -> list[dict[str, Any]]:
        self.initialize()
        path_prefix = _normalize_relative_path(path)
        clauses = ["root_id = ?"]
        params: list[Any] = [self.root_id]
        if not include_missing:
            clauses.append("missing = 0")
        if path_prefix:
            clauses.append("(relative_path = ? OR relative_path LIKE ?)")
            params.extend([path_prefix, f"{path_prefix}/%"])
        sql = (
            "SELECT * FROM files WHERE "
            + " AND ".join(clauses)
            + " ORDER BY relative_path COLLATE NOCASE"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_file(self, relative_path: str) -> dict[str, Any] | None:
        self.initialize()
        normalized_path = _normalize_relative_path(relative_path)
        if not normalized_path:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM files
                WHERE root_id = ? AND relative_path = ? AND missing = 0
                """,
                (self.root_id, normalized_path),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def replace_content_chunks(
        self,
        *,
        relative_path: str,
        content_hash: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.initialize()
        normalized_path = _normalize_relative_path(relative_path)
        if not normalized_path:
            return {"indexed_chunk_count": 0}
        now = _utc_now()
        with self._connect() as conn:
            self._delete_indexed_content_for_path(conn, normalized_path)
            fts_available = _table_exists(conn, "content_chunks_fts")
            for chunk in chunks:
                chunk_index = int(chunk["chunk_index"])
                chunk_id = _build_chunk_id(self.root_id, normalized_path, chunk_index)
                chunk_text = str(chunk["chunk_text"])
                conn.execute(
                    """
                    INSERT INTO content_chunks (
                        chunk_id, root_id, relative_path, chunk_index, content_hash,
                        chunk_text, char_start, char_end, indexed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        self.root_id,
                        normalized_path,
                        chunk_index,
                        str(content_hash),
                        chunk_text,
                        int(chunk["char_start"]),
                        int(chunk["char_end"]),
                        now,
                    ),
                )
                if fts_available:
                    _insert_content_chunk_fts(
                        conn,
                        root_id=self.root_id,
                        relative_path=normalized_path,
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                        chunk_text=chunk_text,
                    )
        return {
            "indexed_chunk_count": len(chunks),
            "indexed_at": now,
        }

    def replace_content_embeddings(
        self,
        *,
        relative_path: str,
        content_hash: str,
        embedding_provider: str,
        embedding_model: str,
        chunker_version: str,
        embeddings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.initialize()
        normalized_path = _normalize_relative_path(relative_path)
        if not normalized_path:
            return {"embedded_chunk_count": 0}

        provider = str(embedding_provider or "").strip()
        model = str(embedding_model or "").strip()
        chunker = str(chunker_version or "").strip()
        if not provider or not model or not chunker:
            raise ValueError("embedding_provider, embedding_model, and chunker_version are required.")

        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM content_embeddings
                WHERE root_id = ?
                    AND relative_path = ?
                    AND embedding_provider = ?
                    AND embedding_model = ?
                    AND chunker_version = ?
                """,
                (self.root_id, normalized_path, provider, model, chunker),
            )
            chunk_rows = conn.execute(
                """
                SELECT chunk_id, chunk_index
                FROM content_chunks
                WHERE root_id = ? AND relative_path = ? AND content_hash = ?
                """,
                (self.root_id, normalized_path, str(content_hash)),
            ).fetchall()
            chunk_ids_by_index = {int(row["chunk_index"]): str(row["chunk_id"]) for row in chunk_rows}
            embedded_count = 0
            for embedding in embeddings:
                chunk_index = int(embedding["chunk_index"])
                chunk_id = chunk_ids_by_index.get(chunk_index)
                vector = _normalize_embedding_vector(embedding.get("embedding_vector"))
                if not chunk_id or not vector:
                    continue
                metadata = embedding.get("metadata") if isinstance(embedding.get("metadata"), dict) else {}
                embedding_id = _build_embedding_id(self.root_id, chunk_id, provider, model, chunker)
                conn.execute(
                    """
                    INSERT INTO content_embeddings (
                        embedding_id, root_id, relative_path, chunk_id, chunk_index, content_hash,
                        embedding_provider, embedding_model, embedding_dim, embedding_vector,
                        chunker_version, metadata_json, embedded_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(root_id, chunk_id, embedding_provider, embedding_model, chunker_version)
                    DO UPDATE SET
                        relative_path = excluded.relative_path,
                        chunk_index = excluded.chunk_index,
                        content_hash = excluded.content_hash,
                        embedding_dim = excluded.embedding_dim,
                        embedding_vector = excluded.embedding_vector,
                        metadata_json = excluded.metadata_json,
                        embedded_at = excluded.embedded_at
                    """,
                    (
                        embedding_id,
                        self.root_id,
                        normalized_path,
                        chunk_id,
                        chunk_index,
                        str(content_hash),
                        provider,
                        model,
                        len(vector),
                        serialize_json(vector),
                        chunker,
                        serialize_json(metadata),
                        now,
                    ),
                )
                embedded_count += 1
        return {
            "embedded_chunk_count": embedded_count,
            "embedded_at": now,
            "embedding_provider": provider,
            "embedding_model": model,
            "chunker_version": chunker,
        }

    def get_indexed_content_paths(self) -> set[str]:
        self.initialize()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT relative_path FROM content_chunks WHERE root_id = ?",
                (self.root_id,),
            ).fetchall()
        return {str(row["relative_path"]) for row in rows}

    def get_embedded_content_paths(
        self,
        *,
        path: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        chunker_version: str | None = None,
    ) -> set[str]:
        self.initialize()
        path_prefix = _normalize_relative_path(path)
        clauses = ["root_id = ?"]
        params: list[Any] = [self.root_id]
        if path_prefix:
            clauses.append("(relative_path = ? OR relative_path LIKE ?)")
            params.extend([path_prefix, f"{path_prefix}/%"])
        if embedding_provider:
            clauses.append("embedding_provider = ?")
            params.append(str(embedding_provider))
        if embedding_model:
            clauses.append("embedding_model = ?")
            params.append(str(embedding_model))
        if chunker_version:
            clauses.append("chunker_version = ?")
            params.append(str(chunker_version))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT relative_path FROM content_embeddings WHERE " + " AND ".join(clauses),
                params,
            ).fetchall()
        return {str(row["relative_path"]) for row in rows}

    def get_embedding_summary(
        self,
        *,
        path: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        chunker_version: str | None = None,
    ) -> dict[str, Any]:
        self.initialize()
        path_prefix = _normalize_relative_path(path)
        clauses = ["root_id = ?"]
        params: list[Any] = [self.root_id]
        if path_prefix:
            clauses.append("(relative_path = ? OR relative_path LIKE ?)")
            params.extend([path_prefix, f"{path_prefix}/%"])
        if embedding_provider:
            clauses.append("embedding_provider = ?")
            params.append(str(embedding_provider))
        if embedding_model:
            clauses.append("embedding_model = ?")
            params.append(str(embedding_model))
        if chunker_version:
            clauses.append("chunker_version = ?")
            params.append(str(chunker_version))
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS embedding_count,
                    COUNT(DISTINCT relative_path) AS embedding_indexed_file_count
                FROM content_embeddings
                WHERE """ + " AND ".join(clauses),
                params,
            ).fetchone()
        return {
            "embedding_count": int(row["embedding_count"] or 0) if row else 0,
            "embedding_indexed_file_count": int(row["embedding_indexed_file_count"] or 0) if row else 0,
        }

    def list_content_embeddings(
        self,
        *,
        path: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        chunker_version: str | None = None,
        limit: int | None = 200,
    ) -> list[dict[str, Any]]:
        self.initialize()
        path_prefix = _normalize_relative_path(path)
        clauses = ["root_id = ?"]
        params: list[Any] = [self.root_id]
        if path_prefix:
            clauses.append("(relative_path = ? OR relative_path LIKE ?)")
            params.extend([path_prefix, f"{path_prefix}/%"])
        if embedding_provider:
            clauses.append("embedding_provider = ?")
            params.append(str(embedding_provider))
        if embedding_model:
            clauses.append("embedding_model = ?")
            params.append(str(embedding_model))
        if chunker_version:
            clauses.append("chunker_version = ?")
            params.append(str(chunker_version))
        sql = (
            "SELECT * FROM content_embeddings WHERE "
            + " AND ".join(clauses)
            + " ORDER BY relative_path COLLATE NOCASE, chunk_index"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_embedding_row_to_dict(row) for row in rows]

    def list_semantic_candidate_chunks(
        self,
        *,
        embedding_provider: str,
        embedding_model: str,
        chunker_version: str,
        path: str | None = None,
        limit: int | None = 5000,
    ) -> list[dict[str, Any]]:
        self.initialize()
        path_prefix = _normalize_relative_path(path)
        clauses = [
            "e.root_id = ?",
            "e.embedding_provider = ?",
            "e.embedding_model = ?",
            "e.chunker_version = ?",
        ]
        params: list[Any] = [
            self.root_id,
            str(embedding_provider),
            str(embedding_model),
            str(chunker_version),
        ]
        if path_prefix:
            clauses.append("(e.relative_path = ? OR e.relative_path LIKE ?)")
            params.extend([path_prefix, f"{path_prefix}/%"])
        sql = (
            """
            SELECT
                e.embedding_id,
                e.embedding_provider,
                e.embedding_model,
                e.embedding_dim,
                e.embedding_vector,
                e.chunker_version,
                e.metadata_json,
                e.embedded_at,
                c.chunk_id,
                c.root_id,
                c.relative_path,
                c.chunk_index,
                c.content_hash,
                c.chunk_text,
                c.char_start,
                c.char_end,
                c.indexed_at
            FROM content_embeddings e
            JOIN content_chunks c
              ON c.root_id = e.root_id
             AND c.chunk_id = e.chunk_id
            WHERE """
            + " AND ".join(clauses)
            + """
            ORDER BY e.relative_path COLLATE NOCASE, e.chunk_index
            """
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_semantic_candidate_row_to_dict(row) for row in rows]

    def list_content_chunks(
        self,
        *,
        path: str | None = None,
        limit: int | None = 200,
    ) -> list[dict[str, Any]]:
        self.initialize()
        path_prefix = _normalize_relative_path(path)
        clauses = ["root_id = ?"]
        params: list[Any] = [self.root_id]
        if path_prefix:
            clauses.append("(relative_path = ? OR relative_path LIKE ?)")
            params.extend([path_prefix, f"{path_prefix}/%"])
        sql = (
            "SELECT * FROM content_chunks WHERE "
            + " AND ".join(clauses)
            + " ORDER BY relative_path COLLATE NOCASE, chunk_index"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_content_chunks_for_path(self, relative_path: str) -> list[dict[str, Any]]:
        self.initialize()
        normalized_path = _normalize_relative_path(relative_path)
        if not normalized_path:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM content_chunks
                WHERE root_id = ? AND relative_path = ?
                ORDER BY chunk_index
                """,
                (self.root_id, normalized_path),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_paths_missing_embeddings(
        self,
        *,
        embedding_provider: str,
        embedding_model: str,
        chunker_version: str,
        path: str | None = None,
        limit: int | None = 50,
    ) -> list[str]:
        self.initialize()
        path_prefix = _normalize_relative_path(path)
        clauses = ["c.root_id = ?", "e.embedding_id IS NULL"]
        params: list[Any] = [
            str(embedding_provider),
            str(embedding_model),
            str(chunker_version),
            self.root_id,
        ]
        if path_prefix:
            clauses.append("(c.relative_path = ? OR c.relative_path LIKE ?)")
            params.extend([path_prefix, f"{path_prefix}/%"])
        sql = (
            """
            SELECT DISTINCT c.relative_path
            FROM content_chunks c
            LEFT JOIN content_embeddings e
              ON e.root_id = c.root_id
             AND e.chunk_id = c.chunk_id
             AND e.embedding_provider = ?
             AND e.embedding_model = ?
             AND e.chunker_version = ?
            WHERE """
            + " AND ".join(clauses)
            + """
            ORDER BY c.relative_path COLLATE NOCASE
            """
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [str(row["relative_path"]) for row in rows]

    def count_chunks_missing_embeddings(
        self,
        *,
        embedding_provider: str,
        embedding_model: str,
        chunker_version: str,
        path: str | None = None,
    ) -> int:
        self.initialize()
        path_prefix = _normalize_relative_path(path)
        clauses = ["c.root_id = ?", "e.embedding_id IS NULL"]
        params: list[Any] = [
            str(embedding_provider),
            str(embedding_model),
            str(chunker_version),
            self.root_id,
        ]
        if path_prefix:
            clauses.append("(c.relative_path = ? OR c.relative_path LIKE ?)")
            params.extend([path_prefix, f"{path_prefix}/%"])
        sql = (
            """
            SELECT COUNT(*) AS missing_count
            FROM content_chunks c
            LEFT JOIN content_embeddings e
              ON e.root_id = c.root_id
             AND e.chunk_id = c.chunk_id
             AND e.embedding_provider = ?
             AND e.embedding_model = ?
             AND e.chunker_version = ?
            WHERE """
            + " AND ".join(clauses)
        )
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["missing_count"] or 0) if row else 0

    def content_fts_available(self) -> bool:
        self.initialize()
        with self._connect() as conn:
            return _table_exists(conn, "content_chunks_fts")

    def search_content_chunks_fts(
        self,
        *,
        query_terms: list[str],
        path: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        self.initialize()
        match_query = _build_fts_match_query(query_terms)
        if not match_query:
            return []
        path_prefix = _normalize_relative_path(path)
        clauses = ["c.root_id = ?"]
        params: list[Any] = [match_query, self.root_id]
        if path_prefix:
            clauses.append("(c.relative_path = ? OR c.relative_path LIKE ?)")
            params.extend([path_prefix, f"{path_prefix}/%"])
        sql = (
            """
            SELECT c.*, bm25(content_chunks_fts) AS bm25_score
            FROM content_chunks_fts
            JOIN content_chunks c ON c.chunk_id = content_chunks_fts.chunk_id
            WHERE content_chunks_fts MATCH ?
              AND """
            + " AND ".join(clauses)
            + """
            ORDER BY bm25_score ASC, c.relative_path COLLATE NOCASE, c.chunk_index
            LIMIT ?
            """
        )
        params.append(max(1, int(limit)))
        try:
            with self._connect() as conn:
                if not _table_exists(conn, "content_chunks_fts"):
                    return []
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            payload = _row_to_dict(row)
            payload["bm25_score"] = float(row["bm25_score"] or 0.0)
            payload["keyword_rank"] = rank
            payload["retrieval_backend"] = "fts5"
            results.append(payload)
        return results

    def find_content_chunks_by_terms(
        self,
        *,
        query_terms: list[str],
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        self.initialize()
        path_prefix = _normalize_relative_path(path)
        clauses = ["root_id = ?"]
        params: list[Any] = [self.root_id]
        if path_prefix:
            clauses.append("(relative_path = ? OR relative_path LIKE ?)")
            params.extend([path_prefix, f"{path_prefix}/%"])
        if query_terms:
            term_clauses: list[str] = []
            for term in query_terms:
                term_clauses.append("LOWER(chunk_text) LIKE ?")
                params.append(f"%{term.lower()}%")
            clauses.append("(" + " OR ".join(term_clauses) + ")")
        sql = (
            "SELECT * FROM content_chunks WHERE "
            + " AND ".join(clauses)
            + " ORDER BY relative_path COLLATE NOCASE, chunk_index"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def search_content_chunks(
        self,
        *,
        query_terms: list[str],
        path: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        rows = self.find_content_chunks_by_terms(query_terms=query_terms, path=path)
        ranked = [_rank_chunk(row, query_terms=query_terms) for row in rows]
        ranked.sort(key=lambda row: (-int(row["score"]), str(row["relative_path"]).lower(), int(row["chunk_index"])))
        return ranked[: max(1, int(limit))]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _delete_indexed_content_for_path(self, conn: sqlite3.Connection, relative_path: str) -> None:
        if _table_exists(conn, "content_chunks_fts"):
            conn.execute(
                "DELETE FROM content_chunks_fts WHERE root_id = ? AND relative_path = ?",
                (self.root_id, relative_path),
            )
        conn.execute(
            "DELETE FROM content_embeddings WHERE root_id = ? AND relative_path = ?",
            (self.root_id, relative_path),
        )
        conn.execute(
            "DELETE FROM content_chunks WHERE root_id = ? AND relative_path = ?",
            (self.root_id, relative_path),
        )


def _build_file_id(root_id: str, relative_path: str) -> str:
    return hashlib.sha256(f"{root_id}:{relative_path}".encode("utf-8")).hexdigest()[:24]


def _build_chunk_id(root_id: str, relative_path: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{root_id}:{relative_path}:{chunk_index}".encode("utf-8")).hexdigest()[:24]


def _build_embedding_id(
    root_id: str,
    chunk_id: str,
    embedding_provider: str,
    embedding_model: str,
    chunker_version: str,
) -> str:
    key = f"{root_id}:{chunk_id}:{embedding_provider}:{embedding_model}:{chunker_version}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _normalize_embedding_vector(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    vector: list[float] = []
    for item in value:
        try:
            vector.append(float(item))
        except (TypeError, ValueError):
            return []
    return vector


def _ensure_content_chunks_fts(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS content_chunks_fts
            USING fts5(
                root_id UNINDEXED,
                relative_path UNINDEXED,
                chunk_id UNINDEXED,
                chunk_index UNINDEXED,
                chunk_text,
                tokenize='unicode61'
            )
            """
        )
    except sqlite3.OperationalError:
        return


def _rebuild_content_chunks_fts(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("DELETE FROM content_chunks_fts")
        rows = conn.execute(
            """
            SELECT root_id, relative_path, chunk_id, chunk_index, chunk_text
            FROM content_chunks
            ORDER BY root_id, relative_path COLLATE NOCASE, chunk_index
            """
        ).fetchall()
        for row in rows:
            _insert_content_chunk_fts(
                conn,
                root_id=str(row["root_id"]),
                relative_path=str(row["relative_path"]),
                chunk_id=str(row["chunk_id"]),
                chunk_index=int(row["chunk_index"]),
                chunk_text=str(row["chunk_text"]),
            )
    except sqlite3.OperationalError:
        return


def _insert_content_chunk_fts(
    conn: sqlite3.Connection,
    *,
    root_id: str,
    relative_path: str,
    chunk_id: str,
    chunk_index: int,
    chunk_text: str,
) -> None:
    try:
        conn.execute(
            """
            INSERT INTO content_chunks_fts (root_id, relative_path, chunk_id, chunk_index, chunk_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (root_id, relative_path, chunk_id, int(chunk_index), chunk_text),
        )
    except sqlite3.OperationalError:
        return


def _build_fts_match_query(query_terms: list[str]) -> str:
    terms = [_escape_fts_term(term) for term in query_terms if _escape_fts_term(term)]
    return " OR ".join(terms[:8])


def _escape_fts_term(term: str) -> str:
    normalized = str(term or "").strip().replace('"', '""')
    if not normalized:
        return ""
    return f'"{normalized}"'


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _rank_chunk(row: dict[str, Any], *, query_terms: list[str]) -> dict[str, Any]:
    text = str(row.get("chunk_text") or "")
    lowered = text.lower()
    score = 0
    for term in query_terms:
        score += lowered.count(term.lower())
    row["score"] = score
    return row


def _file_changed(existing: dict[str, Any], record: dict[str, Any]) -> bool:
    return (
        int(existing.get("size_bytes", 0)) != int(record.get("size_bytes", 0))
        or int(existing.get("mtime_ns", 0)) != int(record.get("mtime_ns", 0))
        or str(existing.get("content_hash", "")) != str(record.get("content_hash", ""))
    )


def _normalize_relative_path(path: str | None) -> str:
    value = str(path or "").strip().replace("\\", "/").strip("/")
    if value in {"", "."}:
        return ""
    return value


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["dataset_candidate"] = bool(payload.get("dataset_candidate", 0))
    payload["missing"] = bool(payload.get("missing", 0))
    return payload


def _embedding_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["embedding_vector"] = json.loads(str(payload.get("embedding_vector") or "[]"))
    payload["metadata"] = json.loads(str(payload.get("metadata_json") or "{}"))
    payload.pop("metadata_json", None)
    return payload


def _semantic_candidate_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["embedding_vector"] = json.loads(str(payload.get("embedding_vector") or "[]"))
    payload["metadata"] = json.loads(str(payload.get("metadata_json") or "{}"))
    payload.pop("metadata_json", None)
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)
