# Important Codebase Files

This document summarizes the main AgentShell structure. It is intentionally workspace-app agnostic: workspace apps plug into the shell through common contracts, while the shell owns routing, agent runtime setup, and tool registration.

## Application Entry And Workspace Mounting

### `main.py`

The Streamlit entry point for AgentShell.

It initializes shell-level session state, registers workspace apps, renders the active shell section, and starts or resumes the assistant runtime.

This is the first file to inspect when answering: "How does the app start?"

### `app/workspace_apps/base.py`

Defines the core contract that every mounted workspace app follows.

Important methods:

- `initialize_state()`: gives each workspace app a hook to initialize its own session-backed state.
- `render()`: renders the workspace app inside the shell.
- `get_ui_state()`: exposes compact user-interface state to the shell and assistant.
- `get_data_context()`: exposes compact data/resource metadata to the shell and assistant.
- `get_dataset_object()`: optionally returns an actual in-memory dataset object for deeper read-only data tools.
- `get_tools()`: optionally exposes app-specific tools when that app is active.

This is the main file for explaining the boundary between AgentShell and arbitrary workspace apps.

### `app/components/workspace_host.py`

Owns which workspace app is currently active and renders it inside the Workspace section.

Most important flow:

```python
active_app.initialize_state()
active_app.render()
```

The host calls each app's `initialize_state()` hook before rendering the app.

### `app/workspace_apps/bootstrap.py`

Registers the workspace apps that are available to the shell.

It is designed to be safe across Streamlit reruns, so apps are registered once per session instead of repeatedly.

### `app/workspace_apps/registry.py`

Stores registered workspace app instances and tracks the default workspace app.

It provides lookup helpers used by the workspace host to resolve the active app.

## Agent Runtime

### `agents/factory.py`

The central wiring file for the assistant runtime.

It builds:

- the selected model provider
- the tool registry
- the default system prompt
- runtime feature flags and budgets
- the base agent runner
- the optional orchestration wrapper

It also combines general framework tools with app-specific tools exposed by the active workspace app.

### `agents/core/agent_runner.py`

Implements the basic tool-calling agent loo (execution).

Main responsibilities:

- send messages and tool schemas to the model provider
- receive model responses
- detect tool calls
- execute tools through the `ToolRegistry`
- append tool results back into the conversation
- stop when the run completes, fails, or reaches a configured budget

This is the main file for explaining the from-scratch agent loop.

### `agents/runtime/orchestrator.py`

Optional higher-level orchestration wrapper around the base agent runner.

When enabled, it can add phases such as:

- conversation compaction
- triage
- planning
- critique
- execution
- reflection

### `agents/providers/base.py`

Defines the provider-agnostic interface for model providers.

The rest of the runtime talks to this interface instead of depending directly on one model vendor's SDK.

### `agents/providers/openai_provider.py`

OpenAI implementation of the model provider interface.

It converts AgentShell's internal messages and tool schemas into OpenAI Responses API calls, then normalizes the response back into AgentShell's internal format.

## Tool System

### `agents/tools/base.py`

Defines the base interface for all model-callable tools.

Each tool has:

- a name
- a description
- a JSON-schema-like input schema
- metadata such as category, scope, read-only status, default enablement, and permission level
- a `run()` method

This is the root abstraction for framework tools and app-specific tools.

### `agents/tools/registry.py`

Central registry for model-callable tools.

It registers tools, looks them up by name, exposes provider-facing schemas, and exposes metadata for grouping or filtering tools.

### `agents/tools/app_context_tools.py`

Framework-level context tools.

These expose high-level shell and runtime context, such as active workspace app, theme, shell section, chat summary, and runtime capabilities.

### `agents/tools/ui_state_tools.py`

Framework-level UI-state tool.

This exposes the active workspace app's current UI state through the workspace app contract rather than requiring the agent to inspect UI widgets directly.

### `agents/tools/data_context_tools.py`

Framework-level data context tool.

This lets the assistant inspect what structured data or resources are currently available before deciding whether to call deeper data tools.

### `agents/tools/data_tools.py`

Framework-level data analysis tools.

These tools support operations such as dataset profiling, sampling, aggregation, and chart-oriented data retrieval. They are intended to work across workspace apps that expose dataframe-like dataset objects through the app contract.

### `agents/tools/visualization_tools.py`

Framework-level visualization/data chart tool definitions.

These wrap visualization-oriented services so the assistant can request compact chart-ready results instead of directly manipulating UI components.

## Data And State Bridge

### `domain/services/data_context_service.py`

Builds a compact summary of structured data currently available to the shell.

Preferred path:

1. Resolve the active workspace app.
2. Ask it for `get_data_context()`.
3. Normalize that context for the assistant.

Fallback path:

1. Check legacy `st.session_state["loaded_dataframes"]`.
2. Check legacy `st.session_state["active_dataframe_name"]`.

This is important because modern workspace apps should expose data through the app contract, while the old direct-session-state dataframe path remains only for compatibility.

### `domain/services/data_tools_service.py`

Implements the deeper dataframe operations behind the general data tools.

It resolves the active workspace app and asks for the actual dataset object through `get_dataset_object()`. If a dataframe is available, it can profile, sample, aggregate, and filter it in a controlled way.

### `domain/services/visualization_tools_service.py`

Implements chart-ready data generation for visualization tools.

Like the data tools service, it resolves datasets through the active workspace app instead of hard-coding one app's internal state.

### `app/state/ui_state.py`

Maintains shell-level UI state and builds snapshots that include workspace host information and active embedded app state.

This supports tools that need to know what the user is currently looking at without coupling those tools to individual Streamlit widgets.

### `app/state/agent_runtime_state.py`

Stores assistant runtime feature flags and budget settings in session state.

Examples include planning, reflection, compaction, runtime gates, and budget overrides.

### `app/state/provider_state.py`

Stores selected model provider and model choices in session state.

The factory uses this state when building the current provider.

### `app/state/chat_context_state.py`

Stores compacted chat-history context when conversation compaction is enabled.

This helps keep long conversations usable without sending the entire history every time.

### `app/state/chat_run_state.py`

Manages active assistant run state in Streamlit.

This supports resumable/incremental chat execution across Streamlit reruns.

## Prompts And Runtime Guidance

### `agents/prompts/app_prompts.py`

Builds the app-facing system prompt and usage guidance.

This is where the assistant is told how to reason about the shell, workspace apps, data tools, UI state, and app-specific tools.

### `agents/prompts/runtime_prompts.py`

Builds prompts used by runtime phases such as execution and orchestration.

This supports separating the visible assistant behavior from hidden planning or runtime-management behavior.

## Observability

### `agents/observability/run_logger.py`

Persists completed assistant run logs.

This supports later inspection of what happened during agent runs.

### `agents/observability/metrics_aggregator.py`

Aggregates run and tool-use metrics.

Used by admin/performance views to understand reliability, tool usage, and failure patterns.

### `agents/observability/performance_summary.py`

Builds higher-level summaries from collected run metrics.

Useful for explaining how the system can evaluate its own runtime/tool behavior.

## UI Components

### `app/components/chat_panel.py`

Renders the main Agent section chat UI.

### `app/components/assistant_pane.py`

Renders the assistant pane when the assistant is open alongside another shell section.

### `app/components/admin_panel.py`

Renders the Admin section.

This includes runtime controls, provider/model controls, traces, tool metadata, and performance-oriented views.

### `app/components/shell_tabs.py`

Renders the main top-level shell navigation.

### `app/components/control_rail.py`

Renders shell-level controls such as theme/runtime toggles.

## The Short Mental Model

The core flow is:

1. `main.py` starts the Streamlit shell.
2. `bootstrap.py` registers workspace apps.
3. `workspace_host.py` selects the active workspace app and calls `initialize_state()` before rendering.
4. Each workspace app follows the contract in `base.py`.
5. `agents/factory.py` builds the assistant runtime and combines general tools with active app tools.
6. `agent_runner.py` handles the basic model/tool loop.
7. Data and UI tools access app state through app contract methods such as `get_ui_state()`, `get_data_context()`, and `get_dataset_object()`.
8. App-specific tools can be added by implementing `get_tools()` on a workspace app.

