from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from app.workspace_apps.ml_workbench.services.agent_tool_service import MLWorkbenchToolService


class GetMLModelingSetupTool(BaseTool):
    name = "get_ml_modeling_setup"
    description = "Return the persisted ML Workbench modeling setup, including problem type, target column, positive class label, identifier columns, ignored columns, selected feature columns, active dataset name, and readiness flags."
    category = "ml_context"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.get_modeling_setup()


class SetMLModelingSetupTool(BaseTool):
    name = "set_ml_modeling_setup"
    description = "Set the core ML Workbench problem definition, including problem type, target column, positive class label, identifier columns, ignored columns, and selected feature columns."
    category = "ml_context"
    scope = "app"
    is_read_only = False
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "problem_type": {
                "type": ["string", "null"],
                "enum": ["classification", "regression", None],
            },
            "target_column": {
                "type": ["string", "null"],
            },
            "positive_class_label": {
                "type": ["string", "number", "integer", "boolean", "null"],
            },
            "id_columns": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "ignored_columns": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "selected_feature_columns": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
        },
        "required": [],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.set_modeling_setup(**kwargs)
