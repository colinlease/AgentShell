from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from app.workspace_apps.ml_workbench.services.agent_tool_service import MLWorkbenchToolService


class GetMLPreprocessingConfigTool(BaseTool):
    name = "get_ml_preprocessing_config"
    description = "Return shared dataset-level preprocessing only: dropped columns, numeric imputation, categorical imputation, datetime handling, and compact working-data status. Does not include scaling, encoding, class rebalancing, or candidate-specific overrides."
    category = "ml_preprocessing"
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
        return self.service.get_preprocessing_config_summary()


class UpdateMLPreprocessingConfigTool(BaseTool):
    name = "update_ml_preprocessing_config"
    description = "Update shared dataset-level preprocessing only: dropped columns, numeric imputation, categorical imputation, datetime handling, and optional working-data rebuild. Do not use this tool for scaling, encoding, class rebalancing, thresholds, tuning, or other candidate-specific settings."
    category = "ml_preprocessing"
    scope = "app"
    is_read_only = False
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "drop_columns": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "numeric_imputation": {
                "type": ["object", "null"],
                "properties": {
                    "strategy": {
                        "type": ["string", "null"],
                        "enum": ["mean", "median", "constant", None],
                    },
                    "fill_value": {
                        "type": ["number", "integer", "null"],
                    },
                    "columns": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                },
                "required": [],
            },
            "categorical_imputation": {
                "type": ["object", "null"],
                "properties": {
                    "strategy": {
                        "type": ["string", "null"],
                        "enum": ["mode", "constant", None],
                    },
                    "fill_value": {
                        "type": ["string", "null"],
                    },
                    "columns": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                },
                "required": [],
            },
            "datetime_handling": {
                "type": ["object", "null"],
                "properties": {
                    "auto_detect": {
                        "type": ["boolean", "null"],
                    },
                    "expanded_columns": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                },
                "required": [],
            },
            "rebuild_working_data": {
                "type": ["boolean", "null"],
            },
        },
        "required": [],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.update_preprocessing_config_summary(**kwargs)
