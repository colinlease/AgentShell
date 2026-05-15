from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from domain.services.data_context_service import DataContextService


class GetLoadedDataContextTool(BaseTool):
    """
    Tool wrapper around the deterministic DataContextService.

    This gives the agent a safe, read-only way to inspect what structured
    datasets are currently available in the active app before deciding whether
    deeper dataset-specific profiling or follow-up tool use is needed.
    """

    name = "get_loaded_data_context"
    description = "Return a lightweight structured summary of currently loaded datasets, including dataset names, active dataset, row counts, column counts, column names, dtype groups, and missing-data summaries."
    category = "data"
    scope = "framework"
    is_read_only = True
    is_enabled_by_default = True
    permission_level = "standard"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, data_context_service: DataContextService | None = None) -> None:
        self.data_context_service = data_context_service or DataContextService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Return lightweight structured context for all currently loaded datasets.
        """
        return self.data_context_service.get_loaded_data_context()