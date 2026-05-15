from __future__ import annotations

from agents.tools.base import BaseTool
from app.workspace_apps.local_knowledge.services.agent_tool_service import LocalKnowledgeToolService
from app.workspace_apps.local_knowledge.tools.context_tools import (
    GetLocalKnowledgeContextTool,
    GetLocalKnowledgeIndexStatusTool,
    IndexLocalKnowledgeContentTool,
    IndexLocalKnowledgeEmbeddingsTool,
    ListLocalKnowledgeFilesTool,
    LoadLocalKnowledgeDatasetTool,
    ReadLocalKnowledgeFileTool,
    SearchLocalKnowledgeTool,
    SemanticSearchLocalKnowledgeTool,
)


def build_local_knowledge_tools() -> list[BaseTool]:
    """
    Return Local Knowledge app tools.

    Keep this app's tool surface intentionally small and read-only. General
    AgentShell tools still handle shell context and published datasets; these
    tools only expose Local Knowledge folder inventory state.
    """
    service = LocalKnowledgeToolService()
    return [
        GetLocalKnowledgeContextTool(service),
        GetLocalKnowledgeIndexStatusTool(service),
        IndexLocalKnowledgeContentTool(service),
        IndexLocalKnowledgeEmbeddingsTool(service),
        ListLocalKnowledgeFilesTool(service),
        LoadLocalKnowledgeDatasetTool(service),
        ReadLocalKnowledgeFileTool(service),
        SearchLocalKnowledgeTool(service),
        SemanticSearchLocalKnowledgeTool(service),
    ]
