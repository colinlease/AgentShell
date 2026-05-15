from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from app.workspace_apps.ml_workbench.services.agent_tool_service import MLWorkbenchToolService


class GetMLFeatureSpecsTool(BaseTool):
    name = "get_ml_feature_specs"
    description = "Return the current ML Workbench engineered feature definitions, including enabled status, operation summary, source columns, and validation status."
    category = "ml_features"
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
        return self.service.get_feature_specs_summary()


class UpsertMLFeatureSpecTool(BaseTool):
    name = "upsert_ml_feature_spec"
    description = "Create or update one ML Workbench engineered feature specification, including feature name, operation, expression, parameters, enabled state, and optional preview-only validation. Prefer guided operations when possible, use expression mode only when needed, and do not invent operation names or expression_language values. Non-preview saves automatically rebuild Working Data."
    category = "ml_features"
    scope = "app"
    is_read_only = False
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "feature_id": {
                "type": ["string", "null"],
            },
            "feature_name": {
                "type": ["string", "null"],
            },
            "feature_type": {
                "type": ["string", "null"],
            },
            "builder_mode": {
                "type": ["string", "null"],
            },
            "operation_family": {
                "type": ["string", "null"],
            },
            "operation": {
                "type": ["string", "null"],
            },
            "expression": {
                "type": ["string", "null"],
            },
            "expression_language": {
                "type": ["string", "null"],
            },
            "parameters": {
                "type": ["object", "null"],
                "additionalProperties": True,
            },
            "source_columns": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "enabled": {
                "type": ["boolean", "null"],
            },
            "apply_order": {
                "type": ["integer", "null"],
            },
            "description": {
                "type": ["string", "null"],
            },
            "preview_only": {
                "type": ["boolean", "null"],
            },
        },
        "required": [],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.upsert_feature_spec(**kwargs)


class RemoveMLFeatureSpecTool(BaseTool):
    name = "remove_ml_feature_spec"
    description = "Remove one ML Workbench engineered feature specification by feature_id and automatically rebuild Working Data so the dataset reflects the removal."
    category = "ml_features"
    scope = "app"
    is_read_only = False
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "feature_id": {
                "type": "string",
            },
        },
        "required": ["feature_id"],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.remove_feature_spec_summary(str(kwargs.get("feature_id", "")))
