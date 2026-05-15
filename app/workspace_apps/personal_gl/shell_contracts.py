"""Shell-ready context builders for later AgentShell integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.workspace_apps.personal_gl.runtime import AppRuntime


def get_ui_state(runtime: AppRuntime) -> dict[str, Any]:
    return runtime.state.build_ui_state_snapshot()


def get_data_context(runtime: AppRuntime) -> dict[str, Any]:
    dataset_summaries = runtime.datasets.summaries()
    active_dataset_name = runtime.state.resolve_active_dataset_name(runtime.datasets.names())
    return {
        "has_data": bool(dataset_summaries),
        "dataset_count": len(dataset_summaries),
        "active_dataset_name": active_dataset_name,
        "db_path": str(Path(runtime.db_path).resolve()),
        "datasets": dataset_summaries,
        "dataset_names": [entry["name"] for entry in dataset_summaries],
        "ui_events": runtime.state.get_ui_events(),
    }


def get_dataset_object(runtime: AppRuntime, dataset_name: str | None = None) -> Any:
    dataset = runtime.get_dataset_object(dataset_name)
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy()
    return dataset
