from __future__ import annotations

from typing import Any

from agents.notes.store import RuntimeNoteStore
from agents.tools.base import BaseTool
from agents.tools.registry import ToolRegistry


class SearchRuntimeNotesTool(BaseTool):
    name = "search_runtime_notes"
    description = (
        "Search runtime heuristic notes by keyword across all note files by default, "
        "or within a single named note file when file_name is provided. Notes are "
        "heuristics only and may be stale."
    )
    category = "notes"
    scope = "runtime"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword or short phrase to search for.",
            },
            "file_name": {
                "type": ["string", "null"],
                "description": "Optional single note file name such as 'general' or an app id like 'ml_workbench'.",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "Optional maximum number of matches to return.",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, store: RuntimeNoteStore) -> None:
        self.store = store

    def run(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs.get("query", "")).strip()
        file_name = kwargs.get("file_name")
        file_name = str(file_name).strip() if file_name not in (None, "") else None
        limit = kwargs.get("limit")
        try:
            limit_value = int(limit) if limit not in (None, "") else 5
        except (TypeError, ValueError):
            limit_value = 5

        return {
            "status": "ok",
            "query": query,
            "file_name": file_name,
            "results": self.store.search_notes(
                query=query,
                file_name=file_name,
                limit=limit_value,
            ),
        }


class GetRuntimeNoteTool(BaseTool):
    name = "get_runtime_note"
    description = "Return one runtime heuristic note by note_id, optionally from a single named note file."
    category = "notes"
    scope = "runtime"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "note_id": {
                "type": "string",
                "description": "Unique note identifier.",
            },
            "file_name": {
                "type": ["string", "null"],
                "description": "Optional note file name such as 'general' or an app id like 'ml_workbench'.",
            },
        },
        "required": ["note_id"],
    }

    def __init__(self, store: RuntimeNoteStore) -> None:
        self.store = store

    def run(self, **kwargs: Any) -> dict[str, Any]:
        note_id = str(kwargs.get("note_id", "")).strip()
        file_name = kwargs.get("file_name")
        file_name = str(file_name).strip() if file_name not in (None, "") else None
        note = self.store.get_note(note_id=note_id, file_name=file_name)
        return {
            "status": "ok" if note is not None else "error",
            "note_id": note_id,
            "file_name": file_name,
            "note": note,
        }


class ListRuntimeNoteFilesTool(BaseTool):
    name = "list_runtime_note_files"
    description = "List available runtime note files that may be searched or read."
    category = "notes"
    scope = "runtime"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, store: RuntimeNoteStore) -> None:
        self.store = store

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "files": self.store.list_note_files(),
        }


class UpsertRuntimeNoteTool(BaseTool):
    name = "upsert_runtime_note"
    description = (
        "Create or update one runtime heuristic note in a named note file. "
        "Use this to refine, simplify, reclassify, or replace notes during reflection."
    )
    category = "notes"
    scope = "runtime"
    is_read_only = False
    permission_level = "reflection"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_name": {
                "type": "string",
                "description": "Target note file such as 'general' or an app id like 'ml_workbench'.",
            },
            "scope": {
                "type": "string",
                "description": "Note scope: 'general' or 'app'.",
                "enum": ["general", "app"],
            },
            "app_id": {
                "type": ["string", "null"],
                "description": "App id when writing an app-scoped note, otherwise null.",
            },
            "note_id": {
                "type": "string",
                "description": "Unique note identifier.",
            },
            "title": {
                "type": "string",
                "description": "Short descriptive title.",
            },
            "statement": {
                "type": "string",
                "description": "Compact operational note statement.",
            },
            "tags": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Optional short tags.",
            },
            "keywords": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Optional search keywords.",
            },
            "confidence": {
                "type": ["number", "null"],
                "description": "Confidence from 0.0 to 1.0.",
                "minimum": 0,
                "maximum": 1,
            },
            "updated_at": {
                "type": ["string", "null"],
                "description": "Optional ISO timestamp. If omitted, the store will fill it automatically.",
            },
        },
        "required": ["file_name", "scope", "note_id", "title", "statement"],
    }

    def __init__(self, store: RuntimeNoteStore) -> None:
        self.store = store

    def run(self, **kwargs: Any) -> dict[str, Any]:
        result = self.store.upsert_note(
            file_name=str(kwargs.get("file_name", "")).strip(),
            scope=str(kwargs.get("scope", "")).strip() or "general",
            app_id=str(kwargs.get("app_id", "")).strip() or None,
            note_payload={
                "note_id": kwargs.get("note_id"),
                "title": kwargs.get("title"),
                "statement": kwargs.get("statement"),
                "tags": kwargs.get("tags"),
                "keywords": kwargs.get("keywords"),
                "confidence": kwargs.get("confidence"),
                "updated_at": kwargs.get("updated_at"),
            },
        )
        result["scope"] = str(kwargs.get("scope", "")).strip() or "general"
        result["app_id"] = str(kwargs.get("app_id", "")).strip() or None
        return result


class DeleteRuntimeNoteTool(BaseTool):
    name = "delete_runtime_note"
    description = "Delete one runtime heuristic note from a named note file during bounded reflection cleanup."
    category = "notes"
    scope = "runtime"
    is_read_only = False
    permission_level = "reflection"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "note_id": {
                "type": "string",
                "description": "Note id to delete.",
            },
            "file_name": {
                "type": "string",
                "description": "Source note file such as 'general' or an app id like 'ml_workbench'.",
            },
        },
        "required": ["note_id", "file_name"],
    }

    def __init__(self, store: RuntimeNoteStore) -> None:
        self.store = store

    def run(self, **kwargs: Any) -> dict[str, Any]:
        result = self.store.delete_note(
            note_id=str(kwargs.get("note_id", "")).strip(),
            file_name=str(kwargs.get("file_name", "")).strip(),
        )
        result["scope"] = "general" if result.get("file_name") == "general" else "app"
        result["app_id"] = None if result.get("file_name") == "general" else result.get("file_name")
        return result


def build_notes_read_registry(store: RuntimeNoteStore) -> ToolRegistry:
    """
    Build the planning-safe read-only notes tool registry.
    """
    return ToolRegistry(
        tools=[
            SearchRuntimeNotesTool(store),
            GetRuntimeNoteTool(store),
            ListRuntimeNoteFilesTool(store),
        ]
    )


def build_notes_reflection_registry(store: RuntimeNoteStore) -> ToolRegistry:
    """
    Build the bounded note-only registry used during reflection.
    """
    return ToolRegistry(
        tools=[
            SearchRuntimeNotesTool(store),
            GetRuntimeNoteTool(store),
            ListRuntimeNoteFilesTool(store),
            UpsertRuntimeNoteTool(store),
            DeleteRuntimeNoteTool(store),
        ]
    )
