from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseWorkspaceApp(ABC):
    """
    Base contract for any application mounted inside the AgentShell Workspace.

    The shell owns high-level routing and host state. Each workspace app owns
    its own internal UI state and can expose that state back to the shell
    through `get_ui_state()`. Apps may also expose structured data/resource
    context separately through `get_data_context()` and may optionally expose
    concrete dataset objects through `get_dataset_object()` for deeper
    read-only dataset tools.
    """

    app_id: str = ""
    app_label: str = ""
    app_type: str = "streamlit"

    def initialize_state(self) -> None:
        """
        Optional hook for app-specific session-state initialization.
        Called by the workspace host before render.
        """
        return None

    @abstractmethod
    def render(self) -> None:
        """
        Render the workspace app inside the shell's Workspace tab.
        """
        raise NotImplementedError

    def get_ui_state(self) -> dict[str, Any]:
        """
        Return a structured snapshot of the app's current UI state.

        This should describe meaningful user-visible app context such as the
        current internal tab, selected object, active filters, current mode,
        open sections, or other state that may help a general shell tool reason
        about what the user is currently looking at.
        """
        return {}

    def get_data_context(self) -> dict[str, Any]:
        """
        Return a structured snapshot of the app's loaded data/resource context.

        This should describe important structured resources currently available
        to the app, such as loaded datasets, tables, schemas, database-backed
        objects, active dataset identifiers, row/column counts, field names,
        or similar metadata that may help general shell tools reason about what
        data is available without overloading `get_ui_state()`.
        """
        return {}

    def get_dataset_object(self, dataset_name: str | None = None) -> Any | None:
        """
        Return the concrete dataset object for a requested dataset name.

        This optional hook is intended for deeper read-only data tools that
        need access to the actual in-memory dataset object rather than only the
        lightweight metadata exposed through `get_data_context()`.

        If `dataset_name` is omitted, apps should return the active dataset
        object when that concept exists.
        """
        return None

    def get_tools(self) -> list[Any]:
        """
        Optional hook for app-specific tools that should be made available when
        this workspace app is active.
        """
        return []
