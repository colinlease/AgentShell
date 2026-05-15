from __future__ import annotations

import json

from agents.core.models import AgentRequest, ChatMessage
from agents.core.runtime_config import validate_runtime_budget_value
from agents.notes.search import build_targeted_note_query
from agents.tools.registry import ToolRegistry
from agents.prompts.runtime_prompts import build_planning_prompt, format_execution_budget_guidance
from agents.providers.base import BaseLLMProvider
from agents.runtime.json_utils import parse_json_object
from agents.runtime.models import PlanningArtifact
from agents.runtime.tool_dialogue import run_bounded_tool_dialogue


class PlanningService:
    """
    Hidden planning stage that produces compact execution guidance.
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    def plan(
        self,
        *,
        request: AgentRequest,
        app_prompt: str,
        tool_catalog_text: str,
        conversation_context_text: str | None = None,
        notes_tool_registry: ToolRegistry | None = None,
        max_model_calls: int = 1,
        max_note_reads: int = 0,
        execution_provider_turn_budget: int | None = None,
        execution_tool_call_budget: int | None = None,
        execution_note_read_budget: int | None = None,
    ) -> PlanningArtifact:
        validate_runtime_budget_value("planning_model_calls", max_model_calls)
        planning_prompt = build_planning_prompt(
            app_prompt=app_prompt,
            tool_catalog_text=tool_catalog_text,
        )
        planning_user_message = (
            "Create a compact execution plan for the latest user turn.\n"
            "You are not solving the task. You are only preparing the execution strategy.\n"
            "Return JSON only with keys: summary (string), execution_guidance (array of strings), missing_context (array of strings).\n\n"
            f"Hidden prior conversation context:\n{conversation_context_text or '- none'}\n\n"
            f"Latest user turn:\n{request.user_input}\n\n"
            f"Available tools:\n{tool_catalog_text}"
        )
        execution_budget_text = format_execution_budget_guidance(
            max_provider_turns=execution_provider_turn_budget,
            max_tool_calls=execution_tool_call_budget,
            notes_enabled=notes_tool_registry is not None,
            max_note_reads=execution_note_read_budget,
        )
        if execution_budget_text:
            planning_user_message += (
                "\n\n"
                f"{execution_budget_text}\n"
                "Create execution guidance that fits within this budget. Prefer fewer, higher-value tool calls over broad exploration."
            )
        working_messages: list[ChatMessage] = []
        remaining_note_reads = max(0, int(max_note_reads))
        parsed = None
        notes_context: list[dict[str, object]] = []
        notes_tool_activity: list[dict[str, object]] = []
        initial_notes_context_text = ""

        if notes_tool_registry is not None and remaining_note_reads > 0:
            initial_output = self._run_initial_note_lookup(
                request=request,
                notes_tool_registry=notes_tool_registry,
            )
            remaining_note_reads -= 1
            notes_tool_activity.append(
                self._build_notes_tool_activity(
                    tool_name="search_runtime_notes",
                    arguments=initial_output.get("arguments", {}),
                    tool_output=initial_output.get("tool_output", {}),
                )
            )
            notes_context = self._merge_notes_context(
                notes_context,
                self._extract_notes_context(initial_output.get("tool_output", {})),
            )
            initial_notes_context_text = self._format_initial_notes_context(
                initial_output.get("tool_output", {}),
            )

        if initial_notes_context_text:
            planning_user_message += (
                "\n\nInitial heuristic note lookup results:\n"
                "These results are non-authoritative and should be treated as hints only.\n"
                f"{initial_notes_context_text}"
            )
        working_messages.append(ChatMessage(role="user", content=planning_user_message))

        dialogue = run_bounded_tool_dialogue(
            provider=self.provider,
            working_messages=working_messages,
            system_prompt=planning_prompt,
            tool_registry=notes_tool_registry,
            max_model_calls=max_model_calls,
            max_tool_calls=remaining_note_reads,
            execute_tool_call=lambda tool_name, arguments: self._execute_notes_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                notes_tool_registry=notes_tool_registry,
            ),
        )

        for step in dialogue.steps:
            notes_tool_activity.append(
                self._build_notes_tool_activity(
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    tool_output=step.tool_output,
                )
            )
            notes_context = self._merge_notes_context(
                notes_context,
                self._extract_notes_context(step.tool_output),
            )

        parsed = parse_json_object(dialogue.final_text)

        if not parsed:
            return PlanningArtifact(
                summary="Proceed carefully using available tools and verify assumptions before concluding.",
                execution_guidance=[],
                missing_context=[],
                notes_context=notes_context,
                notes_tool_activity=notes_tool_activity,
            )

        summary = str(parsed.get("summary", "")).strip()
        execution_guidance_raw = parsed.get("execution_guidance", [])
        missing_context_raw = parsed.get("missing_context", [])

        execution_guidance = (
            [str(item).strip() for item in execution_guidance_raw if str(item).strip()]
            if isinstance(execution_guidance_raw, list)
            else []
        )
        missing_context = (
            [str(item).strip() for item in missing_context_raw if str(item).strip()]
            if isinstance(missing_context_raw, list)
            else []
        )

        return PlanningArtifact(
            summary=summary or "Proceed carefully using available tools and verify assumptions before concluding.",
            execution_guidance=execution_guidance,
            missing_context=missing_context,
            notes_context=notes_context,
            notes_tool_activity=notes_tool_activity,
        )

    def _run_initial_note_lookup(
        self,
        *,
        request: AgentRequest,
        notes_tool_registry: ToolRegistry,
    ) -> dict[str, object]:
        arguments = {
            "query": build_targeted_note_query(request.user_input, max_terms=2),
            "limit": 3,
        }
        try:
            tool = notes_tool_registry.get("search_runtime_notes")
            tool_output = tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001
            tool_output = {
                "status": "error",
                "message": str(exc),
            }

        return {
            "arguments": arguments,
            "tool_output": tool_output,
        }

    @staticmethod
    def _execute_notes_tool_call(
        *,
        tool_name: str,
        arguments: dict[str, object],
        notes_tool_registry: ToolRegistry | None,
    ) -> tuple[dict[str, object], bool]:
        if notes_tool_registry is None or not notes_tool_registry.has(tool_name):
            return (
                {
                    "status": "error",
                    "message": f"Planning-phase notes tool '{tool_name}' is not available.",
                },
                False,
            )

        tool = notes_tool_registry.get(tool_name)
        try:
            return tool.run(**arguments), True
        except Exception as exc:  # noqa: BLE001
            return (
                {
                    "status": "error",
                    "message": str(exc),
                },
                True,
            )

    @staticmethod
    def _format_initial_notes_context(tool_output: object) -> str:
        if not isinstance(tool_output, dict):
            return "- none"
        return json.dumps(tool_output, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _extract_notes_context(tool_output: object) -> list[dict[str, object]]:
        """
        Normalize note tool outputs into a compact context payload for later critique.
        """
        if not isinstance(tool_output, dict):
            return []

        collected: list[dict[str, object]] = []

        search_results = tool_output.get("results")
        if isinstance(search_results, list):
            for item in search_results:
                if not isinstance(item, dict):
                    continue
                note_payload = item.get("note")
                if not isinstance(note_payload, dict):
                    continue
                collected.append(
                    {
                        "file_name": str(item.get("file_name") or ""),
                        "note_id": str(note_payload.get("note_id") or ""),
                        "title": str(note_payload.get("title") or ""),
                        "statement": str(note_payload.get("statement") or ""),
                        "confidence": note_payload.get("confidence"),
                    }
                )

        single_note = tool_output.get("note")
        if isinstance(single_note, dict):
            nested_note = single_note.get("note")
            note_payload = nested_note if isinstance(nested_note, dict) else single_note
            collected.append(
                {
                    "file_name": str(single_note.get("file_name") or tool_output.get("file_name") or ""),
                    "note_id": str(note_payload.get("note_id") or ""),
                    "title": str(note_payload.get("title") or ""),
                    "statement": str(note_payload.get("statement") or ""),
                    "confidence": note_payload.get("confidence"),
                }
            )

        normalized: list[dict[str, object]] = []
        for item in collected:
            if not item.get("note_id") or not item.get("statement"):
                continue
            normalized.append(item)

        return normalized[:5]

    @staticmethod
    def _merge_notes_context(
        existing: list[dict[str, object]],
        incoming: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """
        Merge note context entries while deduplicating by file and note id.
        """
        merged: list[dict[str, object]] = list(existing)
        seen = {
            (str(item.get("file_name") or ""), str(item.get("note_id") or ""))
            for item in existing
        }
        for item in incoming:
            key = (str(item.get("file_name") or ""), str(item.get("note_id") or ""))
            if key in seen:
                continue
            merged.append(item)
            seen.add(key)
            if len(merged) >= 5:
                break
        return merged

    @staticmethod
    def _build_notes_tool_activity(
        *,
        tool_name: str,
        arguments: dict[str, object],
        tool_output: object,
    ) -> dict[str, object]:
        payload = tool_output if isinstance(tool_output, dict) else {}
        status = str(payload.get("status") or "ok").strip() or "ok"
        result_count = 0
        files_count = 0
        found = False

        results = payload.get("results")
        if isinstance(results, list):
            result_count = len(results)

        files = payload.get("files")
        if isinstance(files, list):
            files_count = len(files)

        note = payload.get("note")
        if isinstance(note, dict):
            found = True

        activity: dict[str, object] = {
            "tool_name": tool_name,
            "status": status,
            "arguments": dict(arguments),
            "message": str(payload.get("message") or "").strip(),
        }
        if tool_name == "search_runtime_notes":
            activity["query"] = str(arguments.get("query") or "").strip()
            activity["file_name"] = str(arguments.get("file_name") or payload.get("file_name") or "").strip()
            activity["result_count"] = result_count
        elif tool_name == "get_runtime_note":
            activity["note_id"] = str(arguments.get("note_id") or payload.get("note_id") or "").strip()
            activity["file_name"] = str(arguments.get("file_name") or payload.get("file_name") or "").strip()
            activity["found"] = found
        elif tool_name == "list_runtime_note_files":
            activity["file_count"] = files_count

        return activity
