from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from app.workspace_apps.local_knowledge.services.agent_tool_service import LocalKnowledgeToolService


class GetLocalKnowledgeContextTool(BaseTool):
    name = "get_local_knowledge_context"
    description = "Return compact Local Knowledge app context, including mounted folder status, inventory counts, supported file types, dataset candidate extensions, and currently unavailable capabilities."
    category = "local_knowledge"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, service: LocalKnowledgeToolService | None = None) -> None:
        self.service = service or LocalKnowledgeToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.get_context()


class GetLocalKnowledgeIndexStatusTool(BaseTool):
    name = "get_local_knowledge_index_status"
    description = "Return read-only Local Knowledge inventory status and file counts for the mounted folder."
    category = "local_knowledge"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, service: LocalKnowledgeToolService | None = None) -> None:
        self.service = service or LocalKnowledgeToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.get_index_status()


class ListLocalKnowledgeFilesTool(BaseTool):
    name = "list_local_knowledge_files"
    description = "List files and folders from the mounted Local Knowledge folder inventory. Use folder-relative paths only. Returns metadata, support status, dataset-candidate flags, sizes, and modified timestamps, but not file contents."
    category = "local_knowledge"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": ["string", "null"],
                "description": "Folder-relative path to list. Use null or '.' for the mounted folder root.",
            },
            "depth": {
                "type": ["integer", "null"],
                "description": "Folder depth to return, from 1 to 5. Defaults to 1.",
                "minimum": 1,
                "maximum": 5,
            },
            "include_files": {
                "type": ["boolean", "null"],
                "description": "Whether file entries should be included. Defaults to true.",
            },
            "include_folders": {
                "type": ["boolean", "null"],
                "description": "Whether folder entries should be included. Defaults to true.",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "Maximum entries to return, from 1 to 1000. Defaults to 200.",
                "minimum": 1,
                "maximum": 1000,
            },
        },
        "required": [],
    }

    def __init__(self, service: LocalKnowledgeToolService | None = None) -> None:
        self.service = service or LocalKnowledgeToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        try:
            return self.service.list_files(
                path=kwargs.get("path"),
                depth=kwargs.get("depth"),
                include_files=kwargs.get("include_files"),
                include_folders=kwargs.get("include_folders"),
                limit=kwargs.get("limit"),
            )
        except ValueError as exc:
            return {
                "status": "error",
                "message": str(exc),
                "entries": [],
            }


class ReadLocalKnowledgeFileTool(BaseTool):
    name = "read_local_knowledge_file"
    description = "Read a bounded, read-only text excerpt from one supported file in the mounted Local Knowledge inventory. Use after listing files and pass a folder-relative file path. Returns extracted text for text, CSV, XLSX/XLSM, DOCX, PPTX, and PDF only when the environment has a PDF text extractor."
    category = "local_knowledge"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Folder-relative file path from the Local Knowledge inventory.",
            },
            "max_chars": {
                "type": ["integer", "null"],
                "description": "Maximum characters to return, from 500 to 20000. Defaults to 6000.",
                "minimum": 500,
                "maximum": 20000,
            },
        },
        "required": ["path"],
    }

    def __init__(self, service: LocalKnowledgeToolService | None = None) -> None:
        self.service = service or LocalKnowledgeToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.read_file(
            path=kwargs.get("path"),
            max_chars=kwargs.get("max_chars"),
        )


class LoadLocalKnowledgeDatasetTool(BaseTool):
    name = "load_local_knowledge_dataset"
    description = "Load one CSV or Excel dataset candidate from the mounted Local Knowledge inventory and publish it to AgentShell's general data tools. Use after listing files and pass a folder-relative .csv, .xlsx, .xlsm, or .xls path. Source files remain read-only."
    category = "local_knowledge"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Folder-relative CSV or Excel file path from the Local Knowledge inventory.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, service: LocalKnowledgeToolService | None = None) -> None:
        self.service = service or LocalKnowledgeToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.load_dataset(path=kwargs.get("path"))


class IndexLocalKnowledgeContentTool(BaseTool):
    name = "index_local_knowledge_content"
    description = "Build or extend the lightweight keyword-search index for supported Local Knowledge files. This is explicit so folder refresh stays fast. Use a folder-relative path to index one subtree, or omit path for the mounted root."
    category = "local_knowledge"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": ["string", "null"],
                "description": "Optional folder-relative file or folder path to index. Use null or '.' for the mounted folder root.",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "Maximum unindexed files to process in this call, from 1 to 200. Defaults to 50.",
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": [],
    }

    def __init__(self, service: LocalKnowledgeToolService | None = None) -> None:
        self.service = service or LocalKnowledgeToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.index_content(
            path=kwargs.get("path"),
            limit=kwargs.get("limit"),
        )


class IndexLocalKnowledgeEmbeddingsTool(BaseTool):
    name = "index_local_knowledge_embeddings"
    description = "Build or extend the Local Knowledge embedding index for already-indexed content chunks. This does not perform semantic search yet; it prepares vectors for future semantic retrieval. Use index_local_knowledge_content first when content chunks are missing."
    category = "local_knowledge"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": ["string", "null"],
                "description": "Optional folder-relative file or folder path to embed. Use null or '.' for the mounted folder root.",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "Maximum indexed files to process in this call, from 1 to 50. Defaults to 10.",
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": [],
    }

    def __init__(self, service: LocalKnowledgeToolService | None = None) -> None:
        self.service = service or LocalKnowledgeToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.index_embeddings(
            path=kwargs.get("path"),
            limit=kwargs.get("limit"),
        )


class SearchLocalKnowledgeTool(BaseTool):
    name = "search_local_knowledge"
    description = "Search the Local Knowledge keyword index and return compact snippets with file paths and chunk positions. Use index_local_knowledge_content first if no content has been indexed for the mounted folder or subtree."
    category = "local_knowledge"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword query to search for in indexed Local Knowledge content.",
            },
            "path": {
                "type": ["string", "null"],
                "description": "Optional folder-relative path to restrict search to one subtree.",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "Maximum search results to return, from 1 to 50. Defaults to 10.",
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["query"],
    }

    def __init__(self, service: LocalKnowledgeToolService | None = None) -> None:
        self.service = service or LocalKnowledgeToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.search_content(
            query=kwargs.get("query"),
            path=kwargs.get("path"),
            limit=kwargs.get("limit"),
        )


class SemanticSearchLocalKnowledgeTool(BaseTool):
    name = "semantic_search_local_knowledge"
    description = "Search Local Knowledge indexed embeddings using query embeddings and cosine similarity. Requires content chunks, stored embeddings for the active embedding backend, and an available embedding provider for the query."
    category = "local_knowledge"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language semantic query to compare against embedded Local Knowledge chunks.",
            },
            "path": {
                "type": ["string", "null"],
                "description": "Optional folder-relative path to restrict semantic search to one subtree.",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "Maximum semantic results to return, from 1 to 50. Defaults to 10.",
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["query"],
    }

    def __init__(self, service: LocalKnowledgeToolService | None = None) -> None:
        self.service = service or LocalKnowledgeToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.semantic_search_content(
            query=kwargs.get("query"),
            path=kwargs.get("path"),
            limit=kwargs.get("limit"),
        )
