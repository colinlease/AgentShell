"""Shared runtime object passed across tab renderers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

from app.workspace_apps.GLA import DB_PATH

from app.workspace_apps.personal_gl.datasets import DatasetRegistry
from app.workspace_apps.personal_gl.state import AppState


@dataclass
class AppRuntime:
    state: AppState
    datasets: DatasetRegistry
    db_path: Path

    def register_dataset(
        self,
        name: str,
        obj: Any,
        *,
        kind: str = "object",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        set_active: bool = False,
    ) -> None:
        self.datasets.set(
            name,
            obj,
            kind=kind,
            description=description,
            metadata=metadata,
        )
        if set_active:
            self.state.set_active_dataset_name(name)

    def get_dataset_object(self, dataset_name: str | None = None) -> Any:
        if dataset_name not in (None, ""):
            return self.datasets.get(dataset_name)

        active_dataset_name = self.state.resolve_active_dataset_name(self.datasets.names())
        if active_dataset_name:
            return self.datasets.get(active_dataset_name)
        return None


def build_runtime(storage: MutableMapping[str, Any]) -> AppRuntime:
    state = AppState(storage)
    state.ensure_defaults()
    datasets = DatasetRegistry(storage)
    return AppRuntime(state=state, datasets=datasets, db_path=Path(DB_PATH))
