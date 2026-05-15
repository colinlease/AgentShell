from __future__ import annotations

from dataclasses import fields, replace

from agents.adapters.streamlit_chat import StreamlitChatAdapter
from agents.core.agent_runner import AgentRunner
from agents.core.runtime_config import (
    AgentRuntimeConfig,
    RuntimeBudgetConfig,
    RuntimeFeatureFlags,
    RuntimeGateConfig,
    RuntimePaths,
)
from agents.notes.store import RuntimeNoteStore
from agents.prompts.app_prompts import get_app_prompt
from agents.prompts.runtime_prompts import build_execution_system_prompt
from agents.providers.openai_provider import OpenAIProvider
from agents.providers.gemini_provider import GeminiProvider
from agents.providers.deepseek_provider import DeepSeekProvider
from agents.runtime.orchestrator import OrchestratedAgentRuntime
from agents.tools.app_context_tools import GetAgentRuntimeCapabilitiesTool, GetAppContextTool
from agents.tools.data_context_tools import GetLoadedDataContextTool
from agents.tools.data_tools import (
    DeriveDatasetFeaturesTool,
    GetDatasetAggregationTool,
    GetDatasetProfileTool,
    GetDatasetSampleTool,
)
from agents.tools.visualization_tools import GetDatasetChartTool
from agents.tools.ui_state_tools import GetUIStateTool
from agents.tools.registry import ToolRegistry
from app.components.workspace_host import get_active_workspace_app
from app.state.agent_runtime_state import (
    get_runtime_budget_overrides,
    get_runtime_gate_settings,
    initialize_agent_runtime_state,
    is_compaction_enabled,
    is_planning_enabled,
    is_reflection_enabled,
)
from app.state.provider_state import get_selected_model_name, get_selected_provider_name, initialize_provider_state
from config.settings import AppSettings, PROJECT_ROOT, get_settings



SUPPORTED_PROVIDERS = {"openai", "gemini", "deepseek"}


def build_general_tools() -> list:
    """
    Build the general framework-level tools available across apps.

    These tools are intended to remain reusable and app-agnostic. App-specific
    tools can later be added in a separate builder without changing the UI or
    agent bootstrap flow.
    """
    return [
        GetAppContextTool(),
        GetAgentRuntimeCapabilitiesTool(),
        GetLoadedDataContextTool(),
        GetDatasetProfileTool(),
        GetDatasetSampleTool(),
        GetDatasetAggregationTool(),
        DeriveDatasetFeaturesTool(),
        GetDatasetChartTool(),
        GetUIStateTool(),
    ]



def build_app_tools() -> list:
    """
    Build app-specific tools for the current application.

    App-specific tools are exposed by the active workspace app through its
    `get_tools()` hook. Fail closed when app-specific tool construction fails
    so the general shell tool path remains usable.
    """
    active_app = get_active_workspace_app()
    if active_app is None:
        return []

    try:
        app_tools = active_app.get_tools()
    except Exception:
        return []

    return app_tools if isinstance(app_tools, list) else []


def build_streamlit_chat_adapter(settings: AppSettings | None = None) -> StreamlitChatAdapter:
    """
    Build and return the default Streamlit chat adapter for the application.

    This is the central bootstrap point for the agent runtime. It keeps provider
    selection, tool registration, prompt selection, and runner construction out
    of the UI layer so the same agent stack can be reused across apps and later
    extended to support multiple providers.
    """
    settings = settings or get_settings()
    initialize_provider_state()

    provider = build_provider(settings)
    tool_registry = build_tool_registry()
    system_prompt = build_default_system_prompt()
    runtime_config = build_runtime_config(settings)
    execution_runner = AgentRunner(
        provider=provider,
        tool_registry=tool_registry,
        system_prompt=system_prompt,
        max_steps=runtime_config.budgets.execution_provider_turns,
        max_tool_calls=runtime_config.budgets.execution_tool_calls,
    )

    if should_use_orchestrated_runtime(runtime_config):
        runtime = OrchestratedAgentRuntime(
            execution_runner=execution_runner,
            runtime_config=runtime_config,
            triage_provider=provider,
            planning_provider=provider,
            critique_provider=provider,
            reflection_provider=provider,
            note_store=build_note_store(runtime_config),
        )
    else:
        runtime = execution_runner

    return StreamlitChatAdapter(agent_runner=runtime)


def build_provider(settings: AppSettings):
    """
    Build the selected model provider from live app state with settings as the
    fallback source of truth.

    The provider/model selection is read from session state first so the Admin
    panel can switch providers without requiring a full app restart. Settings
    remain the fallback source for defaults and credentials.
    """
    initialize_provider_state()
    provider_name = get_selected_provider_name() or getattr(settings, "provider_name", "openai") or "openai"
    provider_name = provider_name.strip().lower()
    selected_model = get_selected_model_name()

    if provider_name not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider_name}'. Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
        )

    if provider_name == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to your .env file before using the agent."
            )

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=selected_model or settings.openai_model,
        )

    if provider_name == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Add it to your .env file before using the agent."
            )

        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=selected_model or settings.gemini_model,
        )

    if provider_name == "deepseek":
        if not settings.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is missing. Add it to your .env file before using the agent."
            )

        return DeepSeekProvider(
            api_key=settings.deepseek_api_key,
            model=selected_model or settings.deepseek_model,
        )

    raise ValueError(f"Unsupported provider '{provider_name}'.")


def build_tool_registry() -> ToolRegistry:
    """
    Register and return the default tool set for the current app stage.

    Tool construction is intentionally split into general framework tools and
    app-specific tools so the starter shell can grow into multiple application
    types without flattening all tools into a single hard-coded list.
    """
    tools = [
        *build_general_tools(),
        *build_app_tools(),
    ]

    return ToolRegistry(tools=tools)


def build_runtime_feature_flags() -> RuntimeFeatureFlags:
    """
    Build runtime feature flags from live session-state controls.
    """
    initialize_agent_runtime_state()
    return RuntimeFeatureFlags(
        planning_enabled=is_planning_enabled(),
        reflection_enabled=is_reflection_enabled(),
        compaction_enabled=is_compaction_enabled(),
    )


def build_runtime_budget_config() -> RuntimeBudgetConfig:
    """
    Build runtime budget configuration, applying any known session overrides.
    """
    initialize_agent_runtime_state()
    base_config = RuntimeBudgetConfig()
    overrides = get_runtime_budget_overrides()
    known_budget_fields = {field.name for field in fields(RuntimeBudgetConfig)}
    normalized_overrides = {
        key: int(value)
        for key, value in overrides.items()
        if key in known_budget_fields
    }
    return replace(base_config, **normalized_overrides)


def build_runtime_gate_config() -> RuntimeGateConfig:
    """
    Build deterministic runtime gate settings from persisted live state.
    """
    initialize_agent_runtime_state()
    base_config = RuntimeGateConfig()
    gates = get_runtime_gate_settings()
    known_gate_fields = {field.name for field in fields(RuntimeGateConfig)}
    normalized_gates = {
        key: int(value)
        for key, value in gates.items()
        if key in known_gate_fields
    }
    return replace(base_config, **normalized_gates)


def build_runtime_config(settings: AppSettings | None = None) -> AgentRuntimeConfig:
    """
    Build aggregate runtime configuration for the current session.
    """
    _settings = settings or get_settings()
    return AgentRuntimeConfig(
        features=build_runtime_feature_flags(),
        budgets=build_runtime_budget_config(),
        gates=build_runtime_gate_config(),
        paths=RuntimePaths(notes_root=PROJECT_ROOT / "runtime_notes"),
    )


def should_use_orchestrated_runtime(runtime_config: AgentRuntimeConfig) -> bool:
    """
    Return whether the current runtime config requires the orchestration wrapper.
    """
    return any(
        getattr(runtime_config.features, field.name)
        for field in fields(RuntimeFeatureFlags)
    )


def build_note_store(runtime_config: AgentRuntimeConfig) -> RuntimeNoteStore:
    """
    Build the file-backed runtime note store.
    """
    return RuntimeNoteStore(root=runtime_config.paths.notes_root)

# MUST UPDATE WITH APP SPECIFIC DETAILS
def build_default_system_prompt() -> str:
    """
    Build the default system prompt for the starter app shell.
    """
    active_app = get_active_workspace_app()
    active_app_id = getattr(active_app, "app_id", None)
    app_prompt = get_app_prompt(active_app_id)
    return build_execution_system_prompt(app_prompt=app_prompt)
