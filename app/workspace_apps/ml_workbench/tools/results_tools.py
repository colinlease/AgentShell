from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from app.workspace_apps.ml_workbench.services.agent_tool_service import MLWorkbenchToolService


class GetMLResultsSummaryTool(BaseTool):
    name = "get_ml_results_summary"
    description = "Return the persisted ML Workbench results summary, including best candidate selection, latest run summaries, and compact comparison metrics."
    category = "ml_results"
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
        return self.service.get_results_summary()


class GetMLCandidateResultDetailsTool(BaseTool):
    name = "get_ml_candidate_result_details"
    description = "Return detailed latest-run results for one candidate or run, including full metrics, threshold details, preprocessing summary, params used, and feature columns."
    category = "ml_results"
    scope = "app"
    is_read_only = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": ["string", "null"],
            },
            "run_id": {
                "type": ["string", "null"],
            },
        },
        "required": [],
    }

    def __init__(self, service: MLWorkbenchToolService | None = None) -> None:
        self.service = service or MLWorkbenchToolService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.service.get_candidate_result_details(
            candidate_id=kwargs.get("candidate_id"),
            run_id=kwargs.get("run_id"),
        )
