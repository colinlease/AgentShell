from __future__ import annotations

from typing import Iterable

from agents.tools.base import BaseTool


class ToolRegistry:
    """
    Central registry for model-callable tools.

    This keeps tool registration and lookup separate from UI code and agent
    orchestration, making it easy to swap toolsets by page, workflow, or app.

    The registry also exposes framework-facing metadata so the app can later
    support grouped admin views, permission-aware filtering, and separation
    between general and app-specific tool sets.
    """

    def __init__(self, tools: Iterable[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}

        if tools is not None:
            for tool in tools:
                self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool by its unique name.
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """
        Return a registered tool by name.
        """
        return self._tools[name]

    def has(self, name: str) -> bool:
        """
        Return whether a tool with the given name is registered.
        """
        return name in self._tools

    def list_tools(self) -> list[BaseTool]:
        """
        Return all registered tool instances.
        """
        return list(self._tools.values())

    def list_tool_names(self) -> list[str]:
        """
        Return all registered tool names.
        """
        return list(self._tools.keys())

    def list_tool_schemas(self) -> list[dict]:
        """
        Return provider-facing schema definitions for all registered tools.
        """
        return [tool.to_provider_schema() for tool in self._tools.values()]

    def list_tool_metadata(self) -> list[dict]:
        """
        Return framework-facing metadata for all registered tools.
        """
        return [tool.to_metadata() for tool in self._tools.values()]

    def list_tools_by_category(self, category: str) -> list[BaseTool]:
        """
        Return all registered tools matching a given category.
        """
        return [tool for tool in self._tools.values() if tool.category == category]

    def list_tool_metadata_by_category(self, category: str) -> list[dict]:
        """
        Return tool metadata for all tools matching a given category.
        """
        return [tool.to_metadata() for tool in self.list_tools_by_category(category)]

    def list_tools_by_scope(self, scope: str) -> list[BaseTool]:
        """
        Return all registered tools matching a given scope.
        """
        return [tool for tool in self._tools.values() if tool.scope == scope]

    def list_tool_metadata_by_scope(self, scope: str) -> list[dict]:
        """
        Return tool metadata for all tools matching a given scope.
        """
        return [tool.to_metadata() for tool in self.list_tools_by_scope(scope)]

    def list_read_only_tools(self) -> list[BaseTool]:
        """
        Return all tools marked as read-only.
        """
        return [tool for tool in self._tools.values() if tool.is_read_only]

    def list_write_tools(self) -> list[BaseTool]:
        """
        Return all tools that are not marked as read-only.
        """
        return [tool for tool in self._tools.values() if not tool.is_read_only]