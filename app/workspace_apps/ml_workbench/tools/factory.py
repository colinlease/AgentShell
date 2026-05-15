from __future__ import annotations

from agents.tools.base import BaseTool
from app.workspace_apps.ml_workbench.services.agent_tool_service import MLWorkbenchToolService
from app.workspace_apps.ml_workbench.tools.context_tools import GetMLModelingSetupTool, SetMLModelingSetupTool
from app.workspace_apps.ml_workbench.tools.feature_tools import (
    GetMLFeatureSpecsTool,
    RemoveMLFeatureSpecTool,
    UpsertMLFeatureSpecTool,
)
from app.workspace_apps.ml_workbench.tools.modeling_tools import (
    GetMLCandidateModelsTool,
    GetMLModelComparisonSettingsTool,
    GetMLModelOptionsTool,
    RemoveMLCandidateModelTool,
    TrainMLCandidateModelsTool,
    UpdateMLModelComparisonSettingsTool,
    UpsertMLCandidateModelTool,
)
from app.workspace_apps.ml_workbench.tools.preprocessing_tools import (
    GetMLPreprocessingConfigTool,
    UpdateMLPreprocessingConfigTool,
)
from app.workspace_apps.ml_workbench.tools.results_tools import GetMLCandidateResultDetailsTool, GetMLResultsSummaryTool


def build_ml_workbench_tools() -> list[BaseTool]:
    service = MLWorkbenchToolService()
    return [
        GetMLModelingSetupTool(service),
        SetMLModelingSetupTool(service),
        GetMLPreprocessingConfigTool(service),
        UpdateMLPreprocessingConfigTool(service),
        GetMLFeatureSpecsTool(service),
        UpsertMLFeatureSpecTool(service),
        RemoveMLFeatureSpecTool(service),
        GetMLCandidateModelsTool(service),
        GetMLModelOptionsTool(service),
        GetMLModelComparisonSettingsTool(service),
        UpdateMLModelComparisonSettingsTool(service),
        UpsertMLCandidateModelTool(service),
        RemoveMLCandidateModelTool(service),
        TrainMLCandidateModelsTool(service),
        GetMLResultsSummaryTool(service),
        GetMLCandidateResultDetailsTool(service),
    ]
