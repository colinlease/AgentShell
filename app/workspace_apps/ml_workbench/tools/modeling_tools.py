from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from app.workspace_apps.ml_workbench.services.agent_tool_service import MLWorkbenchToolService


class GetMLCandidateModelsTool(BaseTool):
    name = "get_ml_candidate_models"
    description = "Return configured candidate models, including enabled status, candidate-specific preprocessing overrides, tuning and threshold settings, and latest run summary. Candidate-specific preprocessing includes scaling, encoding, class rebalancing, and feature subset overrides."
    category = "ml_modeling"
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
        return self.service.get_candidate_models_summary()


class GetMLModelOptionsTool(BaseTool):
    name = "get_ml_model_options"
    description = "Return available models for the current or requested problem type, including valid model parameters, tuning options, and default metrics. Use this before setting model-specific params."
    category = "ml_modeling"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "problem_type": {
                "type": ["string", "null"],
                "enum": ["classification", "regression", None],
            },
        },
        "required": [],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.get_model_options_summary(problem_type=kwargs.get("problem_type"))


class GetMLModelComparisonSettingsTool(BaseTool):
    name = "get_ml_model_comparison_settings"
    description = "Return shared model comparison settings, including evaluation metric, split strategy, CV folds, test size, random seed, and shared classification threshold settings."
    category = "ml_modeling"
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
        return self.service.get_model_comparison_settings_summary()


class UpdateMLModelComparisonSettingsTool(BaseTool):
    name = "update_ml_model_comparison_settings"
    description = "Update shared model comparison settings, including evaluation metric, split strategy, CV folds, test size, random seed, and shared classification threshold policy and objective."
    category = "ml_modeling"
    scope = "app"
    is_read_only = False
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "evaluation_metric": {
                "type": ["string", "null"],
            },
            "split_strategy": {
                "type": ["string", "null"],
                "enum": ["cross_validation", "train_test_split", None],
            },
            "cv_folds": {
                "type": ["integer", "null"],
                "minimum": 2,
                "maximum": 10,
            },
            "test_size": {
                "type": ["number", "null"],
            },
            "random_seed": {
                "type": ["integer", "null"],
            },
            "classification_threshold_policy": {
                "type": ["string", "null"],
                "enum": ["Use model default", "Set manual threshold", "Optimize threshold", None],
            },
            "classification_threshold_manual_value": {
                "type": ["number", "null"],
            },
            "classification_threshold_objective": {
                "type": ["string", "null"],
                "enum": ["F1", "Precision", "Recall", None],
            },
        },
        "required": [],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.update_model_comparison_settings_summary(**kwargs)


class UpsertMLCandidateModelTool(BaseTool):
    name = "upsert_ml_candidate_model"
    description = "Create or update one candidate model configuration, including model selection, enabled status, custom params, threshold settings, tuning settings, and candidate-specific preprocessing overrides. For preprocessing, prefer canonical keys: scaling_strategy ('none'|'standard'|'minmax'), encoding_strategy ('none'|'one_hot'), class_rebalancing_strategy ('none'|'oversample'|'undersample'), encoding_columns, feature_subset_mode, included_columns, and excluded_columns."
    category = "ml_modeling"
    scope = "app"
    is_read_only = False
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": ["string", "null"],
            },
            "candidate_label": {
                "type": ["string", "null"],
            },
            "model_id": {
                "type": ["string", "null"],
            },
            "enabled": {
                "type": ["boolean", "null"],
            },
            "train_test_split_enabled": {
                "type": ["boolean", "null"],
            },
            "custom_params": {
                "type": ["object", "null"],
                "additionalProperties": True,
            },
            "classification_threshold": {
                "type": ["number", "null"],
            },
            "notes": {
                "type": ["string", "null"],
            },
            "preprocessing": {
                "type": ["object", "null"],
                "properties": {
                    "use_shared_preprocessing": {
                        "type": ["boolean", "null"],
                    },
                    "feature_subset_mode": {
                        "type": ["string", "null"],
                        "enum": [
                            "Use all eligible predictors",
                            "Include only specific predictors",
                            "Exclude specific predictors",
                            None,
                        ],
                    },
                    "included_columns": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "excluded_columns": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "selected_feature_columns": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "excluded_feature_columns": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "encoding_strategy": {
                        "type": ["string", "null"],
                        "enum": ["none", "one_hot", None],
                    },
                    "encoding_columns": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "scaling_strategy": {
                        "type": ["string", "null"],
                        "enum": ["none", "standard", "minmax", None],
                    },
                    "class_rebalancing_strategy": {
                        "type": ["string", "null"],
                        "enum": ["none", "oversample", "undersample", None],
                    },
                },
                "additionalProperties": True,
            },
            "tuning": {
                "type": ["object", "null"],
                "additionalProperties": True,
            },
        },
        "required": [],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.upsert_candidate_model(**kwargs)


class RemoveMLCandidateModelTool(BaseTool):
    name = "remove_ml_candidate_model"
    description = "Remove one ML Workbench candidate model configuration by candidate_id."
    category = "ml_modeling"
    scope = "app"
    is_read_only = False
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
            },
        },
        "required": ["candidate_id"],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.remove_candidate_model_summary(str(kwargs.get("candidate_id", "")))


class TrainMLCandidateModelsTool(BaseTool):
    name = "train_ml_candidate_models"
    description = "Train one or more ML Workbench candidate models using the current persisted configuration and return a compact results package."
    category = "ml_modeling"
    scope = "app"
    is_read_only = False
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "candidate_ids": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "only_enabled": {
                "type": ["boolean", "null"],
            },
            "select_best_candidate": {
                "type": ["boolean", "null"],
            },
            "set_active_stage_to_results": {
                "type": ["boolean", "null"],
            },
        },
        "required": [],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.train_candidate_models_summary(
            candidate_ids=kwargs.get("candidate_ids"),
            only_enabled=kwargs.get("only_enabled"),
            select_best_candidate=kwargs.get("select_best_candidate"),
            set_active_stage_to_results=kwargs.get("set_active_stage_to_results"),
        )
