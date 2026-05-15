from __future__ import annotations

from agents.core.models import AgentRequest, ChatMessage
from agents.core.runtime_config import validate_runtime_budget_value
from agents.prompts.runtime_prompts import build_triage_prompt
from agents.providers.base import BaseLLMProvider
from agents.runtime.json_utils import parse_json_object
from agents.runtime.models import TriageDecision


SOCIAL_ACKNOWLEDGEMENTS = {
    "thanks",
    "thank you",
    "thx",
    "ok thanks",
    "okay thanks",
    "got it",
    "sounds good",
    "cool thanks",
    "great thanks",
}


def detect_obvious_low_complexity(user_input: str) -> TriageDecision | None:
    """
    Skip planning for obvious social acknowledgement turns.
    """
    normalized = " ".join(str(user_input or "").strip().lower().split())
    if not normalized:
        return TriageDecision(
            should_plan=False,
            planning_mode="skip",
            reason="empty_turn",
            source="deterministic",
        )

    if normalized in SOCIAL_ACKNOWLEDGEMENTS:
        return TriageDecision(
            should_plan=False,
            planning_mode="skip",
            reason="social_acknowledgement",
            source="deterministic",
        )

    return None


class TriageService:
    """
    Lightweight hidden triage stage that decides whether planning is worth doing.
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    def decide(
        self,
        *,
        request: AgentRequest,
        app_prompt: str,
        tool_catalog_text: str,
        conversation_context_text: str | None = None,
        max_model_calls: int = 1,
    ) -> TriageDecision:
        validate_runtime_budget_value("triage_model_calls", max_model_calls)
        deterministic = detect_obvious_low_complexity(request.user_input)
        if deterministic is not None:
            return deterministic

        triage_prompt = build_triage_prompt(
            app_prompt=app_prompt,
            tool_catalog_text=tool_catalog_text,
        )
        triage_user_message = (
            "Classify whether the latest user turn needs a hidden planning pass.\n"
            "Return JSON only with keys: should_plan (bool), planning_mode ('skip', 'light', or 'deep'), reason (string).\n\n"
            f"Hidden prior conversation context:\n{conversation_context_text or '- none'}\n\n"
            f"Latest user turn:\n{request.user_input}\n\n"
            f"Available tools:\n{tool_catalog_text}"
        )
        remaining_model_calls = int(max_model_calls)
        parsed = None
        while remaining_model_calls > 0:
            remaining_model_calls -= 1
            response = self.provider.generate(
                messages=[ChatMessage(role="user", content=triage_user_message)],
                system_prompt=triage_prompt,
                tools=None,
            )
            parsed = parse_json_object(str(response.get("text", "")))
            if parsed:
                break

        if not parsed:
            return TriageDecision(
                should_plan=True,
                planning_mode="light",
                reason="triage_parse_failed",
                source="fallback",
            )

        should_plan = bool(parsed.get("should_plan", True))
        parsed_mode = str(parsed.get("planning_mode", "light")).strip().lower()
        if parsed_mode not in {"skip", "light", "deep"}:
            parsed_mode = "light"
        planning_mode = parsed_mode
        reason = str(parsed.get("reason", "model_triage"))
        if not should_plan:
            planning_mode = "skip"

        return TriageDecision(
            should_plan=should_plan,
            planning_mode=planning_mode,
            reason=reason,
            source="model",
        )
