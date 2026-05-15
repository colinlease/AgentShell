from __future__ import annotations

from typing import Any

import streamlit as st

from config.settings import get_settings


SELECTED_PROVIDER_KEY = "selected_provider_name"
SELECTED_MODEL_KEY = "selected_model_name"


PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "api_key_attr": "openai_api_key",
        "default_model_attr": "openai_model",
        "models": [
            "gpt-5.2",
            "gpt-5.2-pro",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4.1-mini",
            "gpt-4.1",
        ],
    },
    "gemini": {
        "label": "Gemini",
        "api_key_attr": "gemini_api_key",
        "default_model_attr": "gemini_model",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
    },
    "deepseek": {
        "label": "DeepSeek",
        "api_key_attr": "deepseek_api_key",
        "default_model_attr": "deepseek_model",
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    },
}



def get_provider_specs() -> dict[str, dict[str, Any]]:
    """
    Return the canonical provider specification mapping.

    This mapping is the single source of truth for provider labels, credential
    requirements, default-model fields, and allowed model lists. Future
    providers should be added here so the rest of the app can stay dynamic.
    """
    return PROVIDER_SPECS



def get_configured_provider_names() -> list[str]:
    """
    Return the providers that are currently usable based on configured API keys.
    """
    settings = get_settings()
    configured: list[str] = []

    for provider_name, spec in PROVIDER_SPECS.items():
        api_key_attr = spec["api_key_attr"]
        api_key = getattr(settings, api_key_attr, None)
        if api_key:
            configured.append(provider_name)

    return configured



def get_provider_label(provider_name: str) -> str:
    """
    Return the human-friendly label for a provider.
    """
    spec = PROVIDER_SPECS.get(provider_name, {})
    return str(spec.get("label") or provider_name)



def get_models_for_provider(provider_name: str) -> list[str]:
    """
    Return the allowed model list for a provider.
    """
    spec = PROVIDER_SPECS.get(provider_name, {})
    models = spec.get("models", [])
    return [str(model) for model in models]



def get_default_provider_name() -> str:
    """
    Return the default provider name from settings, if valid and configured.

    Falls back to the first configured provider, then to the first known
    provider if no configured provider is available.
    """
    settings = get_settings()
    configured = get_configured_provider_names()

    candidate = (getattr(settings, "provider_name", "openai") or "openai").strip().lower()
    if candidate in configured:
        return candidate

    if configured:
        return configured[0]

    return next(iter(PROVIDER_SPECS.keys()))



def get_default_model_for_provider(provider_name: str) -> str:
    """
    Return the default model for a provider using settings as the primary
    source of truth when valid and the provider spec model list as fallback.
    """
    settings = get_settings()
    spec = PROVIDER_SPECS.get(provider_name, {})
    model_attr = spec.get("default_model_attr")
    allowed_models = get_models_for_provider(provider_name)

    if model_attr:
        configured_model = getattr(settings, model_attr, None)
        if configured_model and str(configured_model) in allowed_models:
            return str(configured_model)

    if allowed_models:
        return allowed_models[0]

    return ""



def initialize_provider_state() -> None:
    """
    Ensure the current live provider/model selection exists in session state.

    The live selection is initialized from settings defaults, then kept in
    Streamlit session state so provider/model switching can happen without
    restarting the whole app.
    """
    if SELECTED_PROVIDER_KEY not in st.session_state:
        st.session_state[SELECTED_PROVIDER_KEY] = get_default_provider_name()

    configured_provider_names = get_configured_provider_names()
    provider_name = get_selected_provider_name()
    if provider_name not in PROVIDER_SPECS or (
        configured_provider_names and provider_name not in configured_provider_names
    ):
        st.session_state[SELECTED_PROVIDER_KEY] = get_default_provider_name()
        provider_name = get_selected_provider_name()

    valid_models = get_models_for_provider(provider_name)

    if SELECTED_MODEL_KEY not in st.session_state:
        st.session_state[SELECTED_MODEL_KEY] = get_default_model_for_provider(provider_name)

    selected_model = get_selected_model_name()
    if valid_models and selected_model not in valid_models:
        st.session_state[SELECTED_MODEL_KEY] = get_default_model_for_provider(provider_name)



def get_selected_provider_name() -> str:
    """
    Return the current live provider selection from session state.
    """
    provider_name = st.session_state.get(SELECTED_PROVIDER_KEY, get_default_provider_name())
    return str(provider_name).strip().lower()



def get_selected_model_name() -> str:
    """
    Return the current live model selection from session state.
    """
    model_name = st.session_state.get(SELECTED_MODEL_KEY, "")
    return str(model_name)



def set_selected_provider_name(provider_name: str) -> None:
    """
    Persist the current live provider selection and reconcile the selected model
    against that provider's allowed model list.
    """
    normalized_provider = str(provider_name).strip().lower()
    if normalized_provider not in PROVIDER_SPECS:
        normalized_provider = get_default_provider_name()

    st.session_state[SELECTED_PROVIDER_KEY] = normalized_provider

    allowed_models = get_models_for_provider(normalized_provider)
    current_model = get_selected_model_name()
    if allowed_models and current_model not in allowed_models:
        st.session_state[SELECTED_MODEL_KEY] = get_default_model_for_provider(normalized_provider)



def set_selected_model_name(model_name: str) -> None:
    """
    Persist the current live model selection if it is valid for the current
    provider. Invalid values are replaced with the provider default.
    """
    provider_name = get_selected_provider_name()
    allowed_models = get_models_for_provider(provider_name)
    normalized_model = str(model_name)

    if allowed_models and normalized_model not in allowed_models:
        normalized_model = get_default_model_for_provider(provider_name)

    st.session_state[SELECTED_MODEL_KEY] = normalized_model



def get_provider_state_snapshot() -> dict[str, Any]:
    """
    Return a normalized snapshot of provider configuration and the current live
    selection for Admin/debug surfaces.
    """
    configured_names = get_configured_provider_names()
    selected_provider = get_selected_provider_name()
    selected_model = get_selected_model_name()

    return {
        "configured_providers": configured_names,
        "selected_provider": selected_provider,
        "selected_provider_label": get_provider_label(selected_provider),
        "selected_model": selected_model,
        "available_models": get_models_for_provider(selected_provider),
    }


def build_control_rail_model_options() -> list[dict[str, str]]:
    """
    Return compact combined provider/model options for the top control rail.

    Only configured providers are included so the rail presents models that are
    immediately usable in the current session.
    """
    options: list[dict[str, str]] = []

    for provider_name in get_configured_provider_names():
        provider_label = get_provider_label(provider_name)
        for model_name in get_models_for_provider(provider_name):
            value = _build_control_rail_model_value(provider_name, model_name)
            options.append(
                {
                    "provider_name": provider_name,
                    "provider_label": provider_label,
                    "model_name": model_name,
                    "value": value,
                    "label": f"{provider_label} \u00b7 {model_name}",
                }
            )

    return options


def get_active_control_rail_model_option_value() -> str | None:
    """
    Return the active combined provider/model value for the control rail.

    When the current session selection is not present in the configured option
    list, fall back to the first available combined option for display.
    """
    initialize_provider_state()

    options = build_control_rail_model_options()
    if not options:
        return None

    selected_value = _build_control_rail_model_value(
        get_selected_provider_name(),
        get_selected_model_name(),
    )
    option_values = {option["value"] for option in options}
    if selected_value in option_values:
        return selected_value

    return options[0]["value"]


def _build_control_rail_model_value(provider_name: str, model_name: str) -> str:
    """Build the stable combined control-rail value for one provider/model pair."""
    return f"{str(provider_name).strip().lower()}::{str(model_name)}"
