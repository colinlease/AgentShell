

"""Model registry helpers for the ML Workbench app.

This module turns reusable model specs into a central lookup layer. UI panels,
modeling services, and future AgentShell tools should use these helpers instead
of hardcoding model ids or labels.
"""

from __future__ import annotations

from app.workspace_apps.ml_workbench.models.specs import (
    DEFAULT_MODEL_SPECS,
    PROBLEM_TYPE_CLASSIFICATION,
    PROBLEM_TYPE_REGRESSION,
)
from app.workspace_apps.ml_workbench.schemas import ModelSpec


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec["model_id"]: spec for spec in DEFAULT_MODEL_SPECS
}


DEFAULT_MODEL_IDS_BY_PROBLEM_TYPE: dict[str, list[str]] = {
    PROBLEM_TYPE_CLASSIFICATION: [
        "logistic_regression",
        "random_forest_classifier",
    ],
    PROBLEM_TYPE_REGRESSION: [
        "linear_regression",
        "random_forest_regressor",
    ],
}



def get_all_model_specs() -> list[ModelSpec]:
    """Return all registered model specs in registry order."""
    return list(MODEL_REGISTRY.values())



def get_model_spec(model_id: str) -> ModelSpec | None:
    """Return one registered model spec by id, if present."""
    return MODEL_REGISTRY.get(model_id)



def require_model_spec(model_id: str) -> ModelSpec:
    """Return one registered model spec by id or raise a clear error."""
    spec = get_model_spec(model_id)
    if spec is None:
        raise KeyError(f"Model '{model_id}' is not registered.")
    return spec



def is_supported_model(model_id: str) -> bool:
    """Return True when the model id exists in the registry."""
    return model_id in MODEL_REGISTRY



def get_model_specs_for_problem_type(problem_type: str | None) -> list[ModelSpec]:
    """Return all registered model specs compatible with a problem type."""
    if problem_type is None:
        return []

    normalized_problem_type = str(problem_type).strip().lower()
    if not normalized_problem_type:
        return []

    return [
        spec
        for spec in get_all_model_specs()
        if normalized_problem_type in spec.get("problem_types", [])
    ]



def get_model_ids_for_problem_type(problem_type: str | None) -> list[str]:
    """Return model ids compatible with a problem type."""
    return [spec["model_id"] for spec in get_model_specs_for_problem_type(problem_type)]



def get_default_model_ids_for_problem_type(problem_type: str | None) -> list[str]:
    """Return the default model ids for a problem type.

    Only ids that are currently registered are returned.
    """
    if problem_type is None:
        return []

    normalized_problem_type = str(problem_type).strip().lower()
    if not normalized_problem_type:
        return []

    return [
        model_id
        for model_id in DEFAULT_MODEL_IDS_BY_PROBLEM_TYPE.get(normalized_problem_type, [])
        if is_supported_model(model_id)
    ]



def get_default_model_specs_for_problem_type(problem_type: str | None) -> list[ModelSpec]:
    """Return default model specs for a problem type."""
    return [
        require_model_spec(model_id)
        for model_id in get_default_model_ids_for_problem_type(problem_type)
    ]



def get_model_labels_by_id() -> dict[str, str]:
    """Return a compact mapping of model id to display label."""
    return {
        model_id: spec["label"]
        for model_id, spec in MODEL_REGISTRY.items()
    }