from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from agents.core.runtime_config import validate_runtime_budget_value
from app.state.agent_runtime_metadata import (
    DEFAULT_RUNTIME_BUDGETS,
    DEFAULT_RUNTIME_GATES,
    get_agent_runtime_budget_bounds,
    get_agent_runtime_budget_field_names,
    get_agent_runtime_gate_bounds,
    get_agent_runtime_gate_field_names,
)
from config.settings import PROJECT_ROOT


PLANNING_ENABLED_KEY = "runtime_planning_enabled"
REFLECTION_ENABLED_KEY = "runtime_reflection_enabled"
CRITIQUE_ENABLED_KEY = "runtime_critique_enabled"
COMPACTION_ENABLED_KEY = "runtime_compaction_enabled"
RUNTIME_BUDGET_OVERRIDES_KEY = "runtime_budget_overrides"
RUNTIME_GATE_SETTINGS_KEY = "runtime_gate_settings"
AGENT_RUNTIME_SETTINGS_PATH = PROJECT_ROOT / "config" / "agent_runtime_settings.json"
AGENT_RUNTIME_SETTINGS_VERSION = 1

_PERSISTED_FEATURE_KEYS = {
    "planning_enabled": PLANNING_ENABLED_KEY,
    "reflection_enabled": REFLECTION_ENABLED_KEY,
    "compaction_enabled": COMPACTION_ENABLED_KEY,
}

_FEATURE_DEFAULTS = {
    PLANNING_ENABLED_KEY: False,
    REFLECTION_ENABLED_KEY: False,
    CRITIQUE_ENABLED_KEY: False,
    COMPACTION_ENABLED_KEY: True,
}

_GATE_DEFAULTS = {
    "reflection_tool_use_threshold": DEFAULT_RUNTIME_GATES.reflection_tool_use_threshold,
}


def initialize_agent_runtime_state() -> None:
    """
    Ensure runtime feature toggles and budget overrides exist in session state.

    The Admin Agent subtab settings are initialized from the local repo settings
    file when present, then fall back to framework defaults for missing values.
    """
    persisted_settings = load_persisted_agent_runtime_settings()
    persisted_features = persisted_settings.get("features", {})
    persisted_budget_overrides = persisted_settings.get("budget_overrides", {})
    persisted_gates = persisted_settings.get("gates", {})

    defaults = dict(_FEATURE_DEFAULTS)
    for persisted_key, session_key in _PERSISTED_FEATURE_KEYS.items():
        if persisted_key in persisted_features:
            defaults[session_key] = bool(persisted_features[persisted_key])
    defaults[RUNTIME_BUDGET_OVERRIDES_KEY] = persisted_budget_overrides
    defaults[RUNTIME_GATE_SETTINGS_KEY] = persisted_gates

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_persisted_agent_runtime_settings(path: Path | None = None) -> dict[str, Any]:
    """
    Load and normalize Agent subtab runtime settings from the local JSON file.
    """
    settings_path = path or AGENT_RUNTIME_SETTINGS_PATH
    try:
        raw_payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw_payload = {}

    return _normalize_agent_runtime_settings(raw_payload)


def save_agent_runtime_settings(path: Path | None = None) -> None:
    """
    Persist the current Agent subtab runtime settings to the local JSON file.
    """
    settings_path = path or AGENT_RUNTIME_SETTINGS_PATH
    payload = _build_persisted_agent_runtime_settings()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = settings_path.with_suffix(f"{settings_path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(settings_path)


def _build_persisted_agent_runtime_settings() -> dict[str, Any]:
    features = {
        persisted_key: bool(st.session_state.get(session_key, _FEATURE_DEFAULTS[session_key]))
        for persisted_key, session_key in _PERSISTED_FEATURE_KEYS.items()
    }

    return {
        "version": AGENT_RUNTIME_SETTINGS_VERSION,
        "features": features,
        "gates": _normalize_gate_settings(
            st.session_state.get(RUNTIME_GATE_SETTINGS_KEY, {})
        ),
        "budget_overrides": _normalize_budget_overrides(
            st.session_state.get(RUNTIME_BUDGET_OVERRIDES_KEY, {})
        ),
    }


def _normalize_agent_runtime_settings(raw_payload: Any) -> dict[str, Any]:
    raw_payload = raw_payload if isinstance(raw_payload, dict) else {}
    raw_features = raw_payload.get("features", {})
    raw_features = raw_features if isinstance(raw_features, dict) else {}

    features = {
        persisted_key: bool(raw_features.get(persisted_key, _FEATURE_DEFAULTS[session_key]))
        for persisted_key, session_key in _PERSISTED_FEATURE_KEYS.items()
    }

    return {
        "version": AGENT_RUNTIME_SETTINGS_VERSION,
        "features": features,
        "gates": _normalize_gate_settings(raw_payload.get("gates", {})),
        "budget_overrides": _normalize_budget_overrides(raw_payload.get("budget_overrides", {})),
    }


def _normalize_budget_overrides(raw_overrides: Any) -> dict[str, int]:
    if not isinstance(raw_overrides, dict):
        return {}

    allowed_names = get_agent_runtime_budget_field_names()
    bounds_by_name = get_agent_runtime_budget_bounds()
    normalized: dict[str, int] = {}

    for raw_name, raw_value in raw_overrides.items():
        name = str(raw_name)
        if name not in allowed_names:
            continue

        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue

        validate_runtime_budget_value(name, value)

        min_value, max_value = bounds_by_name.get(name, (None, None))
        if min_value is not None:
            value = max(int(min_value), value)
        if max_value is not None:
            value = min(int(max_value), value)

        default_value = int(getattr(DEFAULT_RUNTIME_BUDGETS, name))
        if value != default_value:
            normalized[name] = value

    return normalized


def _normalize_gate_settings(raw_gates: Any) -> dict[str, int]:
    raw_gates = raw_gates if isinstance(raw_gates, dict) else {}
    allowed_names = get_agent_runtime_gate_field_names()
    bounds_by_name = get_agent_runtime_gate_bounds()
    normalized: dict[str, int] = {}

    for name, default_value in _GATE_DEFAULTS.items():
        if name not in allowed_names:
            continue

        raw_value = raw_gates.get(name, default_value)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = int(default_value)

        min_value, max_value = bounds_by_name.get(name, (None, None))
        if min_value is not None:
            value = max(int(min_value), value)
        if max_value is not None:
            value = min(int(max_value), value)

        normalized[name] = value

    return normalized


def is_planning_enabled() -> bool:
    initialize_agent_runtime_state()
    return bool(st.session_state.get(PLANNING_ENABLED_KEY, False))


def set_planning_enabled(enabled: bool) -> None:
    initialize_agent_runtime_state()
    normalized_enabled = bool(enabled)
    if bool(st.session_state.get(PLANNING_ENABLED_KEY, False)) == normalized_enabled:
        return
    st.session_state[PLANNING_ENABLED_KEY] = normalized_enabled
    save_agent_runtime_settings()


def is_reflection_enabled() -> bool:
    initialize_agent_runtime_state()
    return bool(st.session_state.get(REFLECTION_ENABLED_KEY, False))


def set_reflection_enabled(enabled: bool) -> None:
    initialize_agent_runtime_state()
    normalized_enabled = bool(enabled)
    if bool(st.session_state.get(REFLECTION_ENABLED_KEY, False)) == normalized_enabled:
        return
    st.session_state[REFLECTION_ENABLED_KEY] = normalized_enabled
    save_agent_runtime_settings()


def is_critique_enabled() -> bool:
    initialize_agent_runtime_state()
    return bool(st.session_state.get(CRITIQUE_ENABLED_KEY, False))


def set_critique_enabled(enabled: bool) -> None:
    initialize_agent_runtime_state()
    st.session_state[CRITIQUE_ENABLED_KEY] = bool(enabled)


def is_compaction_enabled() -> bool:
    initialize_agent_runtime_state()
    return bool(st.session_state.get(COMPACTION_ENABLED_KEY, True))


def set_compaction_enabled(enabled: bool) -> None:
    initialize_agent_runtime_state()
    normalized_enabled = bool(enabled)
    if bool(st.session_state.get(COMPACTION_ENABLED_KEY, True)) == normalized_enabled:
        return
    st.session_state[COMPACTION_ENABLED_KEY] = normalized_enabled
    save_agent_runtime_settings()


def get_runtime_budget_overrides() -> dict[str, int]:
    initialize_agent_runtime_state()
    overrides = st.session_state.get(RUNTIME_BUDGET_OVERRIDES_KEY, {})
    return dict(overrides) if isinstance(overrides, dict) else {}


def get_runtime_gate_settings() -> dict[str, int]:
    initialize_agent_runtime_state()
    return _normalize_gate_settings(st.session_state.get(RUNTIME_GATE_SETTINGS_KEY, {}))


def get_runtime_gate_setting(name: str) -> int:
    gates = get_runtime_gate_settings()
    normalized_name = str(name)
    return int(gates.get(normalized_name, _GATE_DEFAULTS.get(normalized_name, 0)))


def set_runtime_gate_setting(name: str, value: int) -> None:
    initialize_agent_runtime_state()
    gates = get_runtime_gate_settings()
    updated_gates = dict(gates)
    updated_gates[str(name)] = int(value)
    updated_gates = _normalize_gate_settings(updated_gates)
    if updated_gates == gates:
        return
    st.session_state[RUNTIME_GATE_SETTINGS_KEY] = updated_gates
    save_agent_runtime_settings()


def set_runtime_budget_override(name: str, value: int) -> None:
    initialize_agent_runtime_state()
    overrides = get_runtime_budget_overrides()
    normalized = _normalize_budget_overrides({str(name): int(value)})
    updated_overrides = dict(overrides)
    if normalized:
        updated_overrides.update(normalized)
    else:
        updated_overrides.pop(str(name), None)
    updated_overrides = _normalize_budget_overrides(updated_overrides)
    if updated_overrides == overrides:
        return
    st.session_state[RUNTIME_BUDGET_OVERRIDES_KEY] = updated_overrides
    save_agent_runtime_settings()


def clear_runtime_budget_override(name: str) -> None:
    initialize_agent_runtime_state()
    overrides = get_runtime_budget_overrides()
    if str(name) not in overrides:
        return
    overrides.pop(str(name), None)
    st.session_state[RUNTIME_BUDGET_OVERRIDES_KEY] = overrides
    save_agent_runtime_settings()


def get_runtime_state_snapshot() -> dict[str, Any]:
    initialize_agent_runtime_state()
    return {
        "planning_enabled": is_planning_enabled(),
        "reflection_enabled": is_reflection_enabled(),
        "critique_enabled": is_critique_enabled(),
        "compaction_enabled": is_compaction_enabled(),
        "budget_overrides": get_runtime_budget_overrides(),
        "gates": get_runtime_gate_settings(),
    }
