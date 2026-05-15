from __future__ import annotations

from typing import Any

from agents.tools.registry import ToolRegistry


def build_tool_catalog_text(tool_registry: ToolRegistry) -> str:
    """
    Return a compact text catalog of available tools and their parameter names.
    """
    lines: list[str] = []
    for schema in tool_registry.list_tool_schemas():
        name = str(schema.get("name", "")).strip()
        description = str(schema.get("description", "")).strip()
        parameters = schema.get("parameters", {}) if isinstance(schema, dict) else {}
        property_names: list[str] = []
        if isinstance(parameters, dict):
            properties = parameters.get("properties", {})
            if isinstance(properties, dict):
                property_names = [str(key) for key in properties.keys()]

        parameter_text = ", ".join(property_names) if property_names else "none"
        lines.append(
            f"- {name}: {description} | parameters: {parameter_text}"
        )

    return "\n".join(lines)
