"""Manifest helpers for the ML Workbench workspace app.

This module centralizes shell-facing metadata and lightweight descriptive
information so the workspace app can identify itself consistently in both
standalone UI and AgentShell contexts.
"""

from __future__ import annotations

from typing import TypedDict

from app.workspace_apps.ml_workbench.constants import (
    APP_DESCRIPTION,
    APP_ID,
    APP_LABEL,
    APP_TYPE,
    WORKFLOW_STAGE_LABELS,
    WORKFLOW_STAGES,
)


class AppManifest(TypedDict):
    """Typed shell-facing metadata for the ML Workbench app."""

    app_id: str
    app_label: str
    app_type: str
    description: str
    supports_agent_tools: bool
    supports_multiple_artifacts: bool
    supports_export: bool
    workflow_stages: list[str]
    workflow_stage_labels: dict[str, str]



def get_app_manifest() -> AppManifest:
    """Return the canonical manifest for the ML Workbench app."""
    return {
        "app_id": APP_ID,
        "app_label": APP_LABEL,
        "app_type": APP_TYPE,
        "description": APP_DESCRIPTION,
        "supports_agent_tools": True,
        "supports_multiple_artifacts": True,
        "supports_export": True,
        "workflow_stages": list(WORKFLOW_STAGES),
        "workflow_stage_labels": dict(WORKFLOW_STAGE_LABELS),
    }



def get_app_description() -> str:
    """Return a short natural-language description of the app."""
    return APP_DESCRIPTION



def get_app_identity() -> dict[str, str]:
    """Return the minimal identity block used by the workspace contract."""
    return {
        "app_id": APP_ID,
        "app_label": APP_LABEL,
        "app_type": APP_TYPE,
    }
