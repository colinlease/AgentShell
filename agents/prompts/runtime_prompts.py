from __future__ import annotations

from agents.prompts.system_prompts import BASE_SYSTEM_PROMPT, build_system_prompt


def build_execution_system_prompt(
    *,
    app_prompt: str,
    context_summary: str | None = None,
    planning_guidance: list[str] | None = None,
    notes_enabled: bool = False,
    max_note_reads: int | None = None,
    max_execution_provider_turns: int | None = None,
    max_execution_tool_calls: int | None = None,
    base_prompt: str = BASE_SYSTEM_PROMPT,
) -> str:
    """
    Build the execution-phase system prompt for the normal tool-using runtime.
    """
    parts_context: list[str] = []
    if context_summary:
        parts_context.append(context_summary.strip())
    if planning_guidance:
        normalized_guidance = [str(item).strip() for item in planning_guidance if str(item).strip()]
        if normalized_guidance:
            parts_context.append(
                "Hidden execution guidance:\n" + "\n".join(f"- {item}" for item in normalized_guidance)
            )
    execution_budget_guidance = format_execution_budget_guidance(
        max_provider_turns=max_execution_provider_turns,
        max_tool_calls=max_execution_tool_calls,
        notes_enabled=notes_enabled,
        max_note_reads=max_note_reads,
    )
    if execution_budget_guidance:
        parts_context.append(execution_budget_guidance)

    if notes_enabled:
        note_budget_text = (
            f"You may use read-only runtime note tools sparingly as a rescue aid, with at most {int(max_note_reads)} note lookups."
            if max_note_reads is not None
            else "You may use read-only runtime note tools sparingly as a rescue aid."
        )
        parts_context.append(
            "Hidden note guidance:\n"
            f"- {note_budget_text}\n"
            "- Prefer note lookups after tool failures, schema mismatches, or app-specific workflow uncertainty.\n"
            "- Use current tool results and current runtime context over notes if they conflict.\n"
            "- Do not overuse note tools when the next step is already clear."
        )

    parts_context.append(
        "Hidden response guardrails:\n"
        "- Tool outputs are hidden working context, not user-facing content.\n"
        "- Never expose raw tool-call syntax, raw JSON tool dumps, or begin a user-facing answer with `Tool ... returned:`.\n"
        "- After using tools, synthesize the result for the user in normal prose unless they explicitly ask for raw output."
    )

    combined_context_summary = "\n\n".join(parts_context) if parts_context else None

    return build_system_prompt(
        base_prompt=base_prompt,
        app_prompt=app_prompt,
        context_summary=combined_context_summary,
    )


def format_execution_budget_guidance(
    *,
    max_provider_turns: int | None,
    max_tool_calls: int | None,
    notes_enabled: bool = False,
    max_note_reads: int | None = None,
) -> str:
    """
    Format compact hidden budget guidance shared by planning and execution.
    """
    lines: list[str] = []
    if max_provider_turns is not None:
        lines.append(f"- You have at most {max(0, int(max_provider_turns))} provider turns during execution.")
    if max_tool_calls is not None:
        lines.append(f"- You have at most {max(0, int(max_tool_calls))} execution tool calls total.")
    if notes_enabled and max_note_reads is not None:
        lines.append(f"- Runtime-note lookups count against that tool budget and are also limited to {max(0, int(max_note_reads))} note reads.")

    if not lines:
        return ""

    lines.extend(
        [
            "- Spend tool calls on the highest-value inspections first.",
            "- If the remaining budget is insufficient, synthesize from available evidence and clearly state what remains uncertain.",
        ]
    )
    return "Hidden execution budget:\n" + "\n".join(lines)


def build_triage_prompt(*, app_prompt: str, tool_catalog_text: str | None = None) -> str:
    """
    Build the lightweight triage prompt used for future scoped difficulty checks.
    """
    context_summary = (
        f"Available tool catalog:\n{tool_catalog_text.strip()}"
        if tool_catalog_text and tool_catalog_text.strip()
        else None
    )
    return build_system_prompt(
        base_prompt=(
            "You are a lightweight hidden triage pass for AgentShell.\n"
            "Your job is to determine whether the latest user turn is simple enough "
            "to answer directly or complex enough to justify hidden planning.\n"
            "Choose exactly one planning mode:\n"
            "- skip: no planning\n"
            "- light: planning only\n"
            "- deep: planning plus a bounded critique review before execution\n"
            "If prior conversation context is provided, use it to judge whether the current turn is actually part of a larger ongoing task.\n"
            "Prefer skipping planning for obvious acknowledgements or other trivial social turns.\n"
            "Prefer light planning by default. Choose deep only when the task has meaningful workflow, tool-order, "
            "schema, or multi-step execution risk that benefits from an additional review."
        ),
        app_prompt=app_prompt,
        context_summary=context_summary,
    )


def build_planning_prompt(*, app_prompt: str, tool_catalog_text: str | None = None) -> str:
    """
    Build the hidden planning prompt used for plan generation.
    """
    context_summary = (
        f"Available tool catalog:\n{tool_catalog_text.strip()}"
        if tool_catalog_text and tool_catalog_text.strip()
        else None
    )
    return build_system_prompt(
        base_prompt=(
            "You are a hidden planning pass for AgentShell.\n"
            "You do not solve the task directly. You create a compact execution plan, "
            "identify likely missing context, and recommend safe tool sequencing.\n"
            "If prior conversation context is provided, treat it as continuity for the ongoing task and preserve established facts unless the current turn contradicts them.\n"
            "You may reason over tool descriptions and schemas, but you must not execute normal app, data, or UI tools in this phase.\n"
            "If runtime note tools are available, you may read notes as heuristics only. Notes are not source-of-truth state and may be stale.\n"
            "For non-trivial runs, perform at least one targeted note lookup before finalizing the plan when note tools are available, and use up to three note lookups total only when they are likely to improve tool order, "
            "schema usage, context gathering, or app-specific workflow reliability.\n"
            "Do not search notes mechanically on every turn, but do not ignore them when they are likely to prevent mistakes."
        ),
        app_prompt=app_prompt,
        context_summary=context_summary,
    )


def build_critique_prompt(*, app_prompt: str, tool_catalog_text: str | None = None) -> str:
    """
    Build the hidden critique prompt used for future plan review.
    """
    context_summary = (
        f"Available tool catalog:\n{tool_catalog_text.strip()}"
        if tool_catalog_text and tool_catalog_text.strip()
        else None
    )
    return build_system_prompt(
        base_prompt=(
            "You are a hidden critique pass for AgentShell.\n"
            "You review a proposed plan for missing context, poor tool order, likely "
            "schema mismatches, or unnecessary actions.\n"
            "If prior conversation context is provided, evaluate the plan against that same context rather than only the latest user turn.\n"
            "Do not solve the task from scratch, do not execute tools, and do not propose note writes.\n"
            "Runtime notes are the agent's own heuristic musings and reminders, not source-of-truth state. "
            "They may be outdated, incomplete, or wrong.\n"
            "Prefer current tool outputs, current app context, current tool schemas, and current user intent over notes."
        ),
        app_prompt=app_prompt,
        context_summary=context_summary,
    )


def build_reflection_prompt(
    *,
    app_prompt: str,
    tool_catalog_text: str | None = None,
    max_tool_calls: int = 8,
) -> str:
    """
    Build the hidden reflection prompt used for future post-run review.
    """
    context_summary = (
        f"Available note tools:\n{tool_catalog_text.strip()}"
        if tool_catalog_text and tool_catalog_text.strip()
        else None
    )
    return build_system_prompt(
        base_prompt=(
            "You are a hidden reflection pass for AgentShell.\n"
            "You review what happened during execution, identify incorrect assumptions, "
            "tool-order mistakes, and missing context, then maintain your runtime notes.\n"
            "If hidden prior conversation context is provided, use it to understand the broader session objective, repeated retries, and previously established facts.\n"
            f"You have at most {int(max_tool_calls)} note-tool calls in this reflection pass. "
            "Plan your work accordingly and spend the budget on the highest-value inspection and cleanup first.\n"
            "You may use only note tools in this phase. Never execute app, data, or UI tools.\n"
            "Notes are heuristic reminders for future agent behavior, not user-facing explanations and not source-of-truth state. "
            "Keep them compact, reusable, and operational.\n"
            "Only write, update, or delete notes when the change will clearly improve future tool choice, tool order, context gathering, "
            "or failure avoidance for later runs.\n"
            "If there is no high-value note maintenance to do, make no note changes.\n"
            "Store app-specific workflow, schema, and tool-order lessons in the active app note file. "
            "Reserve the general note file for heuristics that truly apply across multiple apps or to AgentShell itself.\n"
            "Do not store generic domain facts, one-off task summaries, or notes written to educate the user.\n"
            "Do some bounded note-maintenance work each pass, unless there is no work to be done: search for duplicates, simplify verbose notes, "
            "reclassify notes between general and app scope when justified, and update confidence based on what happened in this session.\n"
            "Prefer updating or deleting stale or redundant notes over creating new duplicates. Do not churn the files unnecessarily.\n"
            "Return a compact JSON reflection summary after any note-tool work is complete."
        ),
        app_prompt=app_prompt,
        context_summary=context_summary,
    )


def build_compaction_prompt(*, app_prompt: str) -> str:
    """
    Build the hidden compaction prompt used for future conversation summarization.
    """
    return build_system_prompt(
        base_prompt=(
            "You are a hidden conversation compaction pass for AgentShell.\n"
            "You summarize older conversation turns while preserving operational continuity for future hidden phases.\n"
            "Preserve important user preferences, established facts, important names, ids, columns, entities, constraints, "
            "unresolved questions, and the user's overall objective.\n"
            "Briefly summarize the overall flow of the older conversation, but keep the summary compact and high-signal.\n"
            "Do not preserve low-value repetition, filler, or routine politeness."
        ),
        app_prompt=app_prompt,
    )
