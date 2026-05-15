from __future__ import annotations

from time import time
from uuid import uuid4
from typing import Any

import streamlit as st

from agents.adapters.streamlit_chat import StreamlitChatAdapter


ACTIVE_CHAT_RUN_KEY = "active_chat_run"
ACTIVE_CHAT_RUN_CONTROL_KEY = "active_chat_run_control"
ACTIVE_CHAT_RUN_PROGRESS_KEY = "active_chat_run_progress_by_id"
FINALIZED_CHAT_RUNS_KEY = "finalized_chat_run_ids"
DEFAULT_STOP_REASON = "user_requested"


def _looks_like_run_state(value: Any) -> bool:
    """
    Return whether an object appears to be a resumable runtime state object.
    """
    if value is None:
        return False
    return hasattr(value, "phase") and hasattr(value, "target_message_id")


def get_active_chat_run() -> Any | None:
    """Return the active chat run state, if present."""
    active_run = st.session_state.get(ACTIVE_CHAT_RUN_KEY)
    return active_run if _looks_like_run_state(active_run) else None


def has_active_chat_run() -> bool:
    """Return whether a resumable chat run is currently active."""
    return get_active_chat_run() is not None


def clear_active_chat_run() -> None:
    """Remove the active chat run from session state."""
    active_run = get_active_chat_run()
    run_id = _get_run_id(active_run) if active_run is not None else None
    st.session_state.pop(ACTIVE_CHAT_RUN_KEY, None)
    clear_active_chat_run_control(run_id)


def request_stop_active_chat_run(reason: str = DEFAULT_STOP_REASON) -> bool:
    """
    Request cooperative cancellation of the active chat run.

    The current Streamlit runner can only observe this between provider/tool
    steps. A future backend worker can reuse the same request shape as a durable
    run-control command.
    """
    run_state = get_active_chat_run()
    if run_state is None:
        return False

    run_id = _get_run_id(run_state)
    st.session_state[ACTIVE_CHAT_RUN_CONTROL_KEY] = {
        "run_id": run_id,
        "stop_requested": True,
        "stop_requested_at_ms": _now_ms(),
        "stop_reason": _normalize_stop_reason(reason),
    }
    return True


def has_stop_requested(run_state: Any | None = None) -> bool:
    """
    Return whether the active run has a pending cooperative stop request.
    """
    control = st.session_state.get(ACTIVE_CHAT_RUN_CONTROL_KEY)
    if not isinstance(control, dict) or not bool(control.get("stop_requested", False)):
        return False

    control_run_id = str(control.get("run_id", "") or "").strip()
    if run_state is not None and control_run_id:
        return _get_run_id(run_state) == control_run_id
    return True


def clear_active_chat_run_control(run_id: str | None = None) -> None:
    """
    Clear stop-control state for the active run.
    """
    if run_id:
        control = st.session_state.get(ACTIVE_CHAT_RUN_CONTROL_KEY)
        if isinstance(control, dict):
            control_run_id = str(control.get("run_id", "") or "").strip()
            if control_run_id and control_run_id != run_id:
                return
    st.session_state.pop(ACTIVE_CHAT_RUN_CONTROL_KEY, None)


def stop_active_chat_run(
    chat_adapter: StreamlitChatAdapter,
    run_state: Any | None = None,
    *,
    reason: str | None = None,
) -> bool:
    """
    Finalize the active run as stopped and clear the active-run state.
    """
    active_run = run_state if _looks_like_run_state(run_state) else get_active_chat_run()
    if active_run is None:
        return False

    stop_reason = _normalize_stop_reason(reason or _get_requested_stop_reason())
    _mark_run_state_stopped(active_run, reason=stop_reason)
    st.session_state[ACTIVE_CHAT_RUN_KEY] = active_run
    _finalize_active_chat_run(chat_adapter, active_run)
    return True


def start_chat_run(
    chat_adapter: StreamlitChatAdapter,
    *,
    user_input: str,
    context: dict[str, object] | None = None,
    target_message_id: str | None = None,
    surface: str | None = None,
) -> bool:
    """
    Create a resumable chat run if no other run is active.
    """
    if has_active_chat_run():
        return False

    request = chat_adapter.build_request(user_input, context=context)
    run_state = chat_adapter.agent_runner.create_run_state(
        request,
        run_id=f"run_{uuid4().hex}",
        target_message_id=target_message_id,
        surface=surface,
    )
    st.session_state[ACTIVE_CHAT_RUN_KEY] = run_state
    return True


def process_active_chat_run(chat_adapter: StreamlitChatAdapter) -> bool:
    """
    Advance the active chat run once and rerun when progress was made.
    """
    run_state = get_active_chat_run()
    if run_state is None:
        return False

    run_id = _get_run_id(run_state)

    if getattr(run_state, "phase", None) in {"completed", "failed"}:
        if run_id and _is_run_finalized(run_id):
            clear_active_chat_run()
            return False
        _finalize_active_chat_run(chat_adapter, run_state)
        st.rerun()
        return False

    if has_stop_requested(run_state):
        stop_active_chat_run(chat_adapter, run_state)
        st.rerun()
        return False

    progress_signature = _build_run_progress_signature(run_state)
    if run_id and _was_progress_already_processed(run_id, progress_signature):
        return False

    chat_adapter.agent_runner.advance_run(run_state)
    st.session_state[ACTIVE_CHAT_RUN_KEY] = run_state
    if run_id:
        _mark_progress_processed(run_id, progress_signature)
    _sync_running_status(chat_adapter, run_state)

    if has_stop_requested(run_state):
        stop_active_chat_run(chat_adapter, run_state)
        st.rerun()
        return False

    if getattr(run_state, "phase", None) in {"completed", "failed"}:
        _finalize_active_chat_run(chat_adapter, run_state)
        st.rerun()
        return False

    st.rerun()
    return False


def _sync_running_status(
    chat_adapter: StreamlitChatAdapter,
    run_state: Any,
) -> None:
    target_message_id = run_state.target_message_id
    if not target_message_id:
        return

    status_text = str(getattr(run_state, "status_text", "") or "").strip()
    current_tool_name = str(getattr(run_state, "current_tool_name", "") or "").strip()
    if not status_text and current_tool_name:
        status_text = f"Running {current_tool_name}"

    if not status_text:
        return

    status_state = str(getattr(run_state, "status_state", "running") or "running").strip().lower()
    chat_adapter.set_user_run_status(
        target_message_id,
        text=status_text,
        state=status_state or "running",
        tool_count=len(getattr(run_state, "tool_results", [])),
    )


def _finalize_active_chat_run(
    chat_adapter: StreamlitChatAdapter,
    run_state: Any,
) -> None:
    result = chat_adapter.agent_runner.finalize_run(run_state)
    stopped = str(getattr(run_state, "stop_reason", "") or "") == "stopped"
    if not stopped:
        chat_adapter.append_assistant_response(result)

    if run_state.target_message_id:
        tool_count = len(getattr(run_state, "tool_results", []))
        if stopped:
            status_lines = []
            if tool_count > 0:
                tool_label = "tool" if tool_count == 1 else "tools"
                status_lines.append(f"Ran {tool_count} {tool_label}")
            status_lines.append("Stopped")
            if hasattr(chat_adapter, "set_user_run_status_lines"):
                chat_adapter.set_user_run_status_lines(
                    run_state.target_message_id,
                    lines=status_lines,
                    state="stopped",
                    tool_count=tool_count,
                    display_text="Stopped",
                )
            else:
                chat_adapter.set_user_run_status(
                    run_state.target_message_id,
                    text="Stopped",
                    state="stopped",
                    tool_count=tool_count,
                )
        elif tool_count > 0:
            tool_label = "tool" if tool_count == 1 else "tools"
            chat_adapter.set_user_run_status(
                run_state.target_message_id,
                text=f"Ran {tool_count} {tool_label}",
                state="complete",
                tool_count=tool_count,
            )
        else:
            chat_adapter.clear_user_run_status(run_state.target_message_id)

    run_id = _get_run_id(run_state)
    if run_id:
        _mark_run_finalized(run_id)
        _clear_progress_tracking(run_id)
        clear_active_chat_run_control(run_id)

    clear_active_chat_run()


def _get_run_id(run_state: Any) -> str:
    return str(getattr(run_state, "run_id", "") or "").strip()


def _now_ms() -> int:
    return int(time() * 1000)


def _normalize_stop_reason(reason: str | None) -> str:
    normalized = str(reason or "").strip()
    return normalized or DEFAULT_STOP_REASON


def _get_requested_stop_reason() -> str:
    control = st.session_state.get(ACTIVE_CHAT_RUN_CONTROL_KEY)
    if isinstance(control, dict):
        return _normalize_stop_reason(str(control.get("stop_reason", "") or ""))
    return DEFAULT_STOP_REASON


def _mark_run_state_stopped(run_state: Any, *, reason: str) -> None:
    previous_phase = str(getattr(run_state, "phase", "") or "")
    previous_status = str(getattr(run_state, "status_text", "") or "")
    current_tool_name = str(getattr(run_state, "current_tool_name", "") or "")

    setattr(run_state, "phase", "failed")
    setattr(run_state, "stop_reason", "stopped")
    setattr(run_state, "final_text", "Stopped.")
    setattr(run_state, "error", None)
    setattr(run_state, "status_text", "Stopped")
    setattr(run_state, "status_state", "stopped")
    setattr(run_state, "stopped_reason", reason)

    execution_state = getattr(run_state, "execution_state", None)
    if execution_state is not None:
        setattr(execution_state, "phase", "failed")
        setattr(execution_state, "stop_reason", "stopped")
        setattr(execution_state, "final_text", "Stopped.")
        setattr(execution_state, "error", None)
        setattr(execution_state, "status_text", "Stopped")
        setattr(execution_state, "status_state", "stopped")
        setattr(execution_state, "stopped_reason", reason)
        if getattr(run_state, "execution_terminal_phase", None) is None:
            setattr(run_state, "execution_terminal_phase", "failed")

    trace = getattr(run_state, "trace", None)
    if isinstance(trace, list) and not _trace_has_stopped_event(trace):
        trace.append(
            {
                "stage": "stopped",
                "reason": reason,
                "phase": previous_phase,
                "status_text": previous_status,
                "current_tool_name": current_tool_name or None,
            }
        )


def _trace_has_stopped_event(trace: list[Any]) -> bool:
    for item in trace:
        if isinstance(item, dict) and str(item.get("stage", "") or "") == "stopped":
            return True
    return False


def _get_progress_tracker() -> dict[str, str]:
    tracker = st.session_state.get(ACTIVE_CHAT_RUN_PROGRESS_KEY)
    if isinstance(tracker, dict):
        return tracker

    tracker = {}
    st.session_state[ACTIVE_CHAT_RUN_PROGRESS_KEY] = tracker
    return tracker


def _get_finalized_runs_tracker() -> set[str]:
    tracker = st.session_state.get(FINALIZED_CHAT_RUNS_KEY)
    if isinstance(tracker, set):
        return tracker
    if isinstance(tracker, list):
        normalized = {str(item).strip() for item in tracker if str(item).strip()}
        st.session_state[FINALIZED_CHAT_RUNS_KEY] = normalized
        return normalized

    tracker = set()
    st.session_state[FINALIZED_CHAT_RUNS_KEY] = tracker
    return tracker


def _was_progress_already_processed(run_id: str, signature: str) -> bool:
    return _get_progress_tracker().get(run_id) == signature


def _mark_progress_processed(run_id: str, signature: str) -> None:
    _get_progress_tracker()[run_id] = signature


def _is_run_finalized(run_id: str) -> bool:
    return run_id in _get_finalized_runs_tracker()


def _mark_run_finalized(run_id: str) -> None:
    _get_finalized_runs_tracker().add(run_id)


def _clear_progress_tracking(run_id: str) -> None:
    _get_progress_tracker().pop(run_id, None)


def _build_run_progress_signature(run_state: Any) -> str:
    trace = getattr(run_state, "trace", None)
    tool_results = getattr(run_state, "tool_results", None)

    parts = [
        _get_run_id(run_state),
        str(getattr(run_state, "phase", "") or ""),
        str(getattr(run_state, "stop_reason", "") or ""),
        str(getattr(run_state, "step_index", "") or ""),
        str(len(trace) if isinstance(trace, list) else 0),
        str(len(tool_results) if isinstance(tool_results, list) else 0),
        str(getattr(run_state, "status_text", "") or ""),
        str(getattr(run_state, "status_state", "") or ""),
        _get_pending_tool_call_id(run_state),
        _build_nested_execution_signature(getattr(run_state, "execution_state", None)),
    ]
    return "|".join(parts)


def _build_nested_execution_signature(execution_state: Any) -> str:
    if execution_state is None:
        return ""

    trace = getattr(execution_state, "trace", None)
    tool_results = getattr(execution_state, "tool_results", None)
    parts = [
        str(getattr(execution_state, "phase", "") or ""),
        str(getattr(execution_state, "step_index", "") or ""),
        str(len(trace) if isinstance(trace, list) else 0),
        str(len(tool_results) if isinstance(tool_results, list) else 0),
        _get_pending_tool_call_id(execution_state),
    ]
    return "~".join(parts)


def _get_pending_tool_call_id(run_state: Any) -> str:
    pending_tool_call = getattr(run_state, "pending_tool_call", None)
    tool_call_id = str(getattr(pending_tool_call, "tool_call_id", "") or "").strip()
    if tool_call_id:
        return tool_call_id

    pending_tool_payload = getattr(run_state, "pending_tool_payload", None)
    if isinstance(pending_tool_payload, dict):
        return str(pending_tool_payload.get("tool_call_id", "") or "").strip()

    return ""
