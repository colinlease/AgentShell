"""Dataset registration for shell-ready context access."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, MutableMapping

import pandas as pd

from app.workspace_apps.personal_gl.constants import SESSION_KEY_PREFIX


@dataclass
class DatasetRecord:
    name: str
    obj: Any
    kind: str = "object"
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "type": self.kind,
            "description": self.description,
        }
        if isinstance(self.obj, pd.DataFrame):
            summary["rows"] = int(len(self.obj.index))
            summary["columns"] = int(len(self.obj.columns))
            summary["column_names"] = [str(column) for column in self.obj.columns]
            summary["dtype_summary"] = {
                "numeric": [str(column) for column in self.obj.select_dtypes(include=["number"]).columns],
                "datetime": [str(column) for column in self.obj.select_dtypes(include=["datetime", "datetimetz"]).columns],
                "boolean": [str(column) for column in self.obj.select_dtypes(include=["bool"]).columns],
                "other": [
                    str(column)
                    for column in self.obj.columns
                    if str(column)
                    not in {
                        *[str(col) for col in self.obj.select_dtypes(include=["number"]).columns],
                        *[str(col) for col in self.obj.select_dtypes(include=["datetime", "datetimetz"]).columns],
                        *[str(col) for col in self.obj.select_dtypes(include=["bool"]).columns],
                    }
                ],
            }
            missing_by_column = self.obj.isna().sum()
            summary["missing_summary"] = {
                "columns_with_missing_count": int((missing_by_column > 0).sum()),
                "columns_with_missing": {
                    str(column): int(count)
                    for column, count in missing_by_column.items()
                    if int(count) > 0
                },
            }
        elif hasattr(self.obj, "__len__") and not isinstance(self.obj, (str, bytes, dict)):
            try:
                summary["size"] = len(self.obj)
            except Exception:
                pass
        summary.update(self.metadata)
        return summary


class DatasetRegistry:
    """Stores active dataframe-like objects in session state."""

    def __init__(self, storage: MutableMapping[str, Any]):
        self.storage = storage
        self.storage.setdefault(self._storage_key, {})

    @property
    def _storage_key(self) -> str:
        return f"{SESSION_KEY_PREFIX}_datasets"

    def set(
        self,
        name: str,
        obj: Any,
        *,
        kind: str = "object",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        datasets = dict(self.storage.get(self._storage_key, {}))
        datasets[name] = DatasetRecord(
            name=name,
            obj=obj,
            kind=kind,
            description=description,
            metadata=dict(metadata or {}),
        )
        self.storage[self._storage_key] = datasets

    def get(self, name: str | None = None) -> Any:
        datasets = self.storage.get(self._storage_key, {})
        if name is None:
            return {dataset_name: record.obj for dataset_name, record in datasets.items()}
        record = datasets.get(name)
        return None if record is None else record.obj

    def get_record(self, name: str) -> DatasetRecord | None:
        return self.storage.get(self._storage_key, {}).get(name)

    def names(self) -> list[str]:
        datasets = self.storage.get(self._storage_key, {})
        return sorted(str(name) for name in datasets.keys())

    def clear(self, name: str) -> None:
        datasets = dict(self.storage.get(self._storage_key, {}))
        datasets.pop(name, None)
        self.storage[self._storage_key] = datasets

    def summaries(self) -> list[dict[str, Any]]:
        datasets = self.storage.get(self._storage_key, {})
        return [record.summary() for _, record in sorted(datasets.items())]
