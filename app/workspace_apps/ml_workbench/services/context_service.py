"""Shell-facing context builders for the ML Workbench app."""

from __future__ import annotations

from typing import Any

from app.workspace_apps.ml_workbench.constants import APP_ID, APP_LABEL
from app.workspace_apps.ml_workbench.schemas import (
    AppState,
    ArtifactRecord,
    PublishedDataContext,
    PublishedDatasetContext,
    PublishedModelingContext,
    PublishedUIState,
)
from app.workspace_apps.ml_workbench.services.artifact_service import get_artifact
from app.workspace_apps.ml_workbench.services.dataset_service import (
    get_active_dataset_name,
    get_available_dataset_names,
)
from app.workspace_apps.ml_workbench.state import get_app_state



def _build_published_dataset_context(record: ArtifactRecord) -> PublishedDatasetContext:
    """Build one shell-facing dataset context block from an artifact record."""
    metadata = record["metadata"]
    return {
        "name": record["name"],
        "type": record["object_type"],
        "role": record["role"],
        "rows": int(metadata.get("rows", 0)),
        "columns": int(metadata.get("columns", 0)),
        "column_names": list(metadata.get("column_names", [])),
        "dtype_summary": dict(metadata.get("dtype_summary", {})),
        "missing_summary": dict(metadata.get("missing_summary", {})),
        "target_column": metadata.get("target_column"),
        "problem_type": metadata.get("problem_type"),
        "ready_for_modeling": bool(metadata.get("ready_for_modeling", False)),
        "note": str(metadata.get("note", "")),
    }


def _get_active_model_ids_from_candidates(candidate_models: list[dict[str, Any]]) -> list[str]:
    """Return enabled model ids from the configured candidate models."""
    active_model_ids: list[str] = []
    for candidate_model in candidate_models:
        if not isinstance(candidate_model, dict):
            continue
        if not bool(candidate_model.get("enabled", False)):
            continue
        model_id = candidate_model.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        if model_id not in active_model_ids:
            active_model_ids.append(model_id)
    return active_model_ids


def _build_modeling_context(state: AppState) -> PublishedModelingContext:
    """Build the compact shell-facing modeling context block."""
    model_comparison_config = state.get("model_comparison_config", {})
    candidate_models = list(model_comparison_config.get("candidate_models", []))
    status = state.get("status", {})
    split_strategy = str(model_comparison_config.get("split_strategy", "cross_validation"))
    evaluation_mode = "Train / Test Split" if split_strategy == "train_test_split" else "Cross Validation"
    return {
        "problem_type": state.get("problem_type"),
        "target_column": state.get("target_column"),
        "positive_class_label": state.get("positive_class_label"),
        "active_candidate_id": state.get("active_candidate_id"),
        "active_model_ids": _get_active_model_ids_from_candidates(candidate_models),
        "best_candidate_id": state.get("best_candidate_id"),
        "candidate_models": candidate_models,
        "best_model_run_id": state.get("best_model_run_id"),
        "results_ready": bool(status.get("results_ready", False)),
        "split_strategy": split_strategy,
        "evaluation_mode": evaluation_mode,
        "test_size": float(model_comparison_config.get("test_size", 0.2)),
        "cv_folds": int(model_comparison_config.get("cv_folds", 5)),
        "random_seed": int(model_comparison_config.get("random_seed", model_comparison_config.get("random_state", 42))),
        "classification_threshold_policy": str(
            model_comparison_config.get("classification_threshold_policy", "Use model default")
        ),
        "classification_threshold_manual_value": float(
            model_comparison_config.get("classification_threshold_manual_value", 0.5)
        ),
        "classification_threshold_objective": str(
            model_comparison_config.get("classification_threshold_objective", "F1")
        ),
    }



def build_published_ui_state(state: AppState | None = None) -> PublishedUIState:
    """Return compact shell-facing UI state for the ML Workbench app."""
    resolved_state = state or get_app_state()
    status = resolved_state.get("status", {})
    model_comparison_config = resolved_state.get("model_comparison_config", {})

    return {
        "app_id": APP_ID,
        "app_label": APP_LABEL,
        "app_stage": resolved_state.get("app_stage", ""),
        "dataset_loaded": bool(status.get("dataset_loaded", False)),
        "loaded_file_name": resolved_state.get("loaded_file_name"),
        "problem_type": resolved_state.get("problem_type"),
        "target_column": resolved_state.get("target_column"),
        "active_dataset_name": resolved_state.get("active_dataset_name"),
        "active_candidate_id": resolved_state.get("active_candidate_id"),
        "active_model_ids": _get_active_model_ids_from_candidates(
            list(model_comparison_config.get("candidate_models", []))
        ),
        "models_trained": bool(status.get("models_trained", False)),
        "best_model_run_id": resolved_state.get("best_model_run_id"),
    }



def build_published_data_context(state: AppState | None = None) -> PublishedDataContext:
    """Return shell-facing data context for the ML Workbench app."""
    resolved_state = state or get_app_state()
    dataset_names = get_available_dataset_names()
    dataset_contexts: list[PublishedDatasetContext] = []

    for dataset_name in dataset_names:
        record = get_artifact(dataset_name)
        if record is None:
            continue
        dataset_contexts.append(_build_published_dataset_context(record))

    active_dataset_name = resolved_state.get("active_dataset_name")
    if not active_dataset_name or get_artifact(active_dataset_name) is None:
        active_dataset_name = get_active_dataset_name(default_to_working=True)

    return {
        "has_data": len(dataset_contexts) > 0,
        "active_dataset_name": active_dataset_name,
        "datasets": dataset_contexts,
        "modeling_context": _build_modeling_context(resolved_state),
    }



def build_workspace_snapshot() -> dict[str, Any]:
    """Return a combined snapshot of UI and data context.

    This helper is mainly intended for debugging, inspection, or future app-
    scoped tooling. The shell-facing contract should continue to rely on the
    more specific UI and data context builders above.
    """
    state = get_app_state()
    return {
        "ui_state": build_published_ui_state(state),
        "data_context": build_published_data_context(state),
    }
