from __future__ import annotations

from dataclasses import asdict
from typing import Any

import streamlit as st
from app.state.ui_state import get_active_shell_section, get_shell_sections, get_ui_state_snapshot

from agents.tools.base import BaseTool


class GetAppContextTool(BaseTool):
    """
    Read-only tool that returns a high-level shell/workspace context snapshot.

    This is intended to orient the agent to the current environment without
    dumping detailed embedded-app UI selections or dataset metadata.
    """

    name = "get_app_context"
    description = "Return the current shell and workspace context, including theme, assistant visibility, active shell section, loaded workspace app, and chat/session summary."
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
        Return a normalized shell/workspace context snapshot.
        """
        chat_history = st.session_state.get("chat_history", [])
        snapshot = get_ui_state_snapshot()
        workspace_host = snapshot.get("workspace_host", {}) if isinstance(snapshot, dict) else {}
        shell_sections = get_shell_sections()
        active_shell_section = get_active_shell_section()

        if not isinstance(workspace_host, dict):
            workspace_host = {}

        active_workspace_app_id = workspace_host.get("active_workspace_app_id")
        active_workspace_app_label = workspace_host.get("active_workspace_app_label")
        active_workspace_app_type = workspace_host.get("active_workspace_app_type")
        available_apps = workspace_host.get("available_apps", [])

        if not isinstance(available_apps, list):
            available_apps = []

        normalized_available_apps: list[dict[str, str]] = []
        for app in available_apps:
            if not isinstance(app, dict):
                continue
            normalized_available_apps.append(
                {
                    "app_id": str(app.get("app_id", "")),
                    "app_label": str(app.get("app_label", "")),
                    "app_type": str(app.get("app_type", "")),
                }
            )

        return {
            "theme_name": st.session_state.get("theme_name", "light"),
            "assistant_open": bool(st.session_state.get("assistant_open", False)),
            "shell_active_section": active_shell_section,
            "shell_sections": shell_sections,
            "workspace_loaded": bool(workspace_host.get("workspace_loaded", False)),
            "active_workspace_app_id": str(active_workspace_app_id) if active_workspace_app_id else None,
            "active_workspace_app_label": str(active_workspace_app_label) if active_workspace_app_label else None,
            "active_workspace_app_type": str(active_workspace_app_type) if active_workspace_app_type else None,
            "available_apps": normalized_available_apps,
            "available_app_count": len(normalized_available_apps),
            "has_chat_history": bool(chat_history),
            "chat_message_count": len(chat_history),
            "app_stage": st.session_state.get("app_stage", "app_shell"),
        }


class GetAgentRuntimeCapabilitiesTool(BaseTool):
    """
    Read-only tool that returns current AgentShell runtime capability facts.
    """

    name = "get_agent_runtime_capabilities"
    description = "Return current AgentShell runtime capabilities, feature flags, budgets, and high-level phase behavior."
    category = "context"
    scope = "framework"
    is_read_only = True
    is_enabled_by_default = True
    permission_level = "standard"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Return a concise, user-explainable snapshot of AgentShell runtime behavior.
        """
        from agents.factory import build_runtime_config

        runtime_config = build_runtime_config()
        budgets = asdict(runtime_config.budgets)
        gates = asdict(runtime_config.gates)
        return {
            "capabilities_schema_version": 1,
            "agent_shell": {
                "description": "AgentShell wraps a mounted workspace app with an in-app assistant, framework/app tools, runtime orchestration, and grounded context.",
                "current_runtime": "single assistant runtime with optional hidden orchestration phases",
            },
            "features": asdict(runtime_config.features),
            "budgets": budgets,
            "gates": gates,
            "phases": {
                "triage": "Active only when Planning is enabled. Classifies the turn as skip, light, or deep before execution.",
                "planning": "Hidden pre-execution strategy pass. It prepares compact execution guidance and cannot execute normal app, data, or UI tools.",
                "critique": "Part of deep planning only. Reviews the plan for likely tool-order, context, or schema issues before execution.",
                "execution": "Visible assistant run that can use registered framework, app, and runtime-note tools within provider-turn and execution tool-call budgets.",
                "reflection": "Post-run runtime-note maintenance when Reflection is enabled and the reflection gate fires. It uses note tools only and does not execute app, data, or UI tools.",
                "compaction": "Hidden conversation-history summarization when Context Compaction is enabled and message or character thresholds are met.",
            },
            "phase_logic": {
                "planning": {
                    "enabled_when": "planning_enabled",
                    "flow": "triage chooses skip/light/deep; light plans then executes; deep plans, critiques, then executes.",
                    "critique": "Only part of deep planning; there is no separate critique toggle.",
                },
                "execution": {
                    "limits": "provider turns and total execution tool calls are capped by the execution budgets.",
                    "runtime_note_tools": "available during execution only when Reflection/runtime notes are active.",
                },
                "reflection": {
                    "enabled_when": "reflection_enabled and runtime note store exists",
                    "runs_after": "visible execution ends",
                    "forced_when": [
                        "execution provider-turn budget is exhausted",
                        "provider error occurs",
                        "tool count is strictly greater than reflection_tool_use_threshold",
                    ],
                    "also_runs_when": "one or more tool calls failed",
                    "skips_when": "disabled, no note store, simple zero-tool completed turn, or low-value completed turn",
                    "tool_use_threshold": gates["reflection_tool_use_threshold"],
                },
                "compaction": {
                    "enabled_when": "compaction_enabled",
                    "runs_when": "visible history reaches the message or character trigger while more messages exist than the recent-message keep count.",
                    "keeps_recent_messages": budgets["compaction_keep_recent_messages"],
                },
            },
            "runtime_notes": {
                "enabled_when": "reflection_enabled and runtime note store exists",
                "purpose": "Heuristic reminders for future tool choice, tool order, context gathering, and failure avoidance.",
                "source_of_truth": False,
                "precedence": "Current tool outputs, explicit UI state, app context, and user-provided facts override runtime notes.",
            },
            "usage_guidance": "Use this tool output as the authoritative source when explaining AgentShell runtime behavior, planning, reflection, compaction, runtime notes, or current runtime budgets.",
        }
