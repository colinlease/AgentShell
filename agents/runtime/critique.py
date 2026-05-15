from __future__ import annotations

from agents.core.models import AgentRequest, ChatMessage
from agents.core.runtime_config import validate_runtime_budget_value
from agents.prompts.runtime_prompts import build_critique_prompt
from agents.providers.base import BaseLLMProvider
from agents.runtime.json_utils import parse_json_object
from agents.runtime.models import CritiqueArtifact, PlanningArtifact


class CritiqueService:
    """
    Hidden critique stage that reviews a plan and tightens execution guidance.
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    def critique(
        self,
        *,
        request: AgentRequest,
        plan: PlanningArtifact,
        app_prompt: str,
        tool_catalog_text: str | None = None,
        conversation_context_text: str | None = None,
        max_model_calls: int = 1,
    ) -> CritiqueArtifact:
        validate_runtime_budget_value("critique_model_calls", max_model_calls)
        critique_prompt = build_critique_prompt(
            app_prompt=app_prompt,
            tool_catalog_text=tool_catalog_text,
        )
        notes_context_text = self._format_notes_context(plan.notes_context)
        critique_user_message = (
            "Review the proposed hidden execution plan.\n"
            "Do not solve the task. Do not call tools. Return JSON only with keys: "
            "summary (string), issues (array of strings), revised_execution_guidance (array of strings).\n\n"
            f"Hidden prior conversation context:\n{conversation_context_text or '- none'}\n\n"
            f"Latest user turn:\n{request.user_input}\n\n"
            f"Plan summary:\n{plan.summary}\n\n"
            f"Plan guidance:\n{self._format_lines(plan.execution_guidance)}\n\n"
            f"Plan missing context:\n{self._format_lines(plan.missing_context)}\n\n"
            "Heuristic notes context:\n"
            "These notes are non-authoritative agent reminders. They may be stale or wrong.\n"
            "Use them only to spot likely tool-order mistakes or repeated failure patterns.\n"
            f"{notes_context_text}"
        )

        remaining_model_calls = int(max_model_calls)
        parsed = None
        while remaining_model_calls > 0:
            remaining_model_calls -= 1
            response = self.provider.generate(
                messages=[ChatMessage(role="user", content=critique_user_message)],
                system_prompt=critique_prompt,
                tools=None,
            )
            parsed = parse_json_object(str(response.get("text", "")))
            if parsed is not None:
                break

        if not parsed:
            return CritiqueArtifact(
                summary="No critique changes were applied.",
                issues=[],
                revised_execution_guidance=list(plan.execution_guidance),
            )

        issues_raw = parsed.get("issues", [])
        guidance_raw = parsed.get("revised_execution_guidance", [])
        issues = (
            [str(item).strip() for item in issues_raw if str(item).strip()]
            if isinstance(issues_raw, list)
            else []
        )
        revised_guidance = (
            [str(item).strip() for item in guidance_raw if str(item).strip()]
            if isinstance(guidance_raw, list)
            else []
        )

        return CritiqueArtifact(
            summary=str(parsed.get("summary", "")).strip() or "No critique changes were applied.",
            issues=issues,
            revised_execution_guidance=revised_guidance or list(plan.execution_guidance),
        )

    @staticmethod
    def _format_lines(lines: list[str]) -> str:
        normalized = [str(line).strip() for line in lines if str(line).strip()]
        if not normalized:
            return "- none"
        return "\n".join(f"- {line}" for line in normalized)

    @staticmethod
    def _format_notes_context(notes_context: list[dict[str, object]]) -> str:
        normalized_lines: list[str] = []
        for note in notes_context:
            note_id = str(note.get("note_id") or "").strip()
            file_name = str(note.get("file_name") or "").strip()
            title = str(note.get("title") or "").strip()
            statement = str(note.get("statement") or "").strip()
            if not note_id or not statement:
                continue
            prefix = f"{file_name}/{note_id}" if file_name else note_id
            title_suffix = f" ({title})" if title else ""
            normalized_lines.append(f"- {prefix}{title_suffix}: {statement}")

        if not normalized_lines:
            return "- none"
        return "\n".join(normalized_lines)
