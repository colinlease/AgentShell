from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from app.state.ui_state import get_ui_state_snapshot


class GetUIStateTool(BaseTool):
    """
    Return a normalized snapshot of the active embedded app's current UI state.

    This tool is intentionally narrower than full shell/workspace context. It
    is designed to expose what the user is currently interacting with inside
    the active workspace app, such as internal tabs, selected controls,
    filters, view settings, and status indicators.
    """

    name = "get_ui_state"
    description = "Return the current UI interaction state of the active workspace app, including active internal tab, selected controls, filters, open sections, view settings, and status indicators."
    category = "context"
    scope = "framework"
    is_read_only = True
    is_enabled_by_default = True
    permission_level = "standard"

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Return the normalized embedded-app UI-state snapshot for the current session.
        """
        snapshot = get_ui_state_snapshot()
        embedded_app = snapshot.get("embedded_app", {})
        state = embedded_app.get("state", {}) if isinstance(embedded_app, dict) else {}

        if not isinstance(state, dict):
            state = {}

        return {
            "app_loaded": bool(embedded_app.get("app_loaded", False)) if isinstance(embedded_app, dict) else False,
            "app_id": embedded_app.get("app_id") if isinstance(embedded_app, dict) else None,
            "app_label": embedded_app.get("app_label") if isinstance(embedded_app, dict) else None,
            "app_type": embedded_app.get("app_type") if isinstance(embedded_app, dict) else None,
            "state_available": bool(embedded_app.get("state_available", False)) if isinstance(embedded_app, dict) else False,
            "state": state,
        }