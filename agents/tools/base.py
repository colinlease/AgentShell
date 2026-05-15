from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Base interface for model-callable tools.

    A tool represents a capability exposed to the agent. Tools should remain
    thin wrappers around deterministic domain services and should not contain
    Streamlit UI logic.

    In addition to execution behavior, each tool carries framework metadata so
    the app can later support grouping, permissions, default enablement, and
    separation between general and app-specific tool sets.
    """

    name: str
    description: str
    schema: dict[str, Any]

    # Framework metadata
    category: str = "general"
    scope: str = "framework"
    is_read_only: bool = True
    is_enabled_by_default: bool = True
    permission_level: str = "standard"

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """
        Execute the tool with validated keyword arguments.
        """
        raise NotImplementedError

    def to_provider_schema(self) -> dict[str, Any]:
        """
        Convert the tool definition into a generic provider-facing schema.

        Only the fields relevant to the model provider are included here.
        Tool-management metadata is intentionally kept separate for framework
        and admin-layer use.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema,
        }

    def to_metadata(self) -> dict[str, Any]:
        """
        Return framework-facing metadata for admin controls, grouping, and
        future permission handling.
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "scope": self.scope,
            "is_read_only": self.is_read_only,
            "is_enabled_by_default": self.is_enabled_by_default,
            "permission_level": self.permission_level,
        }