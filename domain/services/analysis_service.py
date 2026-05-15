

from __future__ import annotations

from typing import Any

import streamlit as st


class AnalysisService:
    """
    Deterministic analysis service for the current app state.

    This first version is intentionally lightweight. It does not depend on any
    specific domain dataset yet. Instead, it analyzes the current starter app
    shell so the agent can begin using a real domain-service pattern:

    tool -> service -> structured result

    Later, this service can be expanded or replaced with app-specific logic such
    as dataset profiling, report generation, filtering diagnostics, or research
    analysis.
    """

    def run_basic_analysis(self) -> dict[str, Any]:
        """
        Return a simple structured summary of the current application state.
        """
        chat_history = st.session_state.get("chat_history", [])
        theme_name = st.session_state.get("theme_name", "dark")

        findings: list[str] = [
            "The app currently contains three main sections: Workspace, Assistant, and About.",
            f"The current theme is '{theme_name}'.",
            "The custom assistant chat widget is enabled.",
        ]

        warnings: list[str] = []
        next_steps: list[str] = []

        if chat_history:
            findings.append(
                f"The current session contains {len(chat_history)} chat message(s)."
            )
        else:
            findings.append("No chat messages have been recorded in the current session yet.")

        findings.append(
            "No domain-specific dataset or workspace object has been loaded yet."
        )

        warnings.append("The app is still in an early starter-shell stage.")

        next_steps.extend(
            [
                "Add a real domain workflow to the Workspace tab.",
                "Expose a domain-specific analysis tool for that workflow.",
                "Expand the assistant beyond app-context inspection.",
            ]
        )

        return {
            "status": "ok",
            "analysis_type": "basic_summary",
            "subject": "starter_workspace",
            "findings": findings,
            "warnings": warnings,
            "next_steps": next_steps,
        }