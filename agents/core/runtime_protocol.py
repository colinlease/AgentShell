from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agents.core.models import AgentRequest
from agents.core.results import AgentResult


class BaseAgentRuntime(ABC):
    """
    Common runtime interface shared by the plain runner and future orchestrators.

    This lets the Streamlit adapter and session-state helpers depend on a small
    stable runtime contract instead of a specific runner implementation.
    """

    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResult:
        raise NotImplementedError

    @abstractmethod
    def create_run_state(
        self,
        request: AgentRequest,
        *,
        run_id: str | None = None,
        target_message_id: str | None = None,
        surface: str | None = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def advance_run(self, run_state: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def finalize_run(self, run_state: Any) -> AgentResult:
        raise NotImplementedError
