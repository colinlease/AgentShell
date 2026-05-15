# AgentShell

AgentShell is a Streamlit-based harness for hosting domain-specific user apps with an embedded agent runtime. The shell provides the shared structure around an app: workspace mounting, chat/assistant UI, model-provider selection, tool registration, and optional orchestration features such as planning, reflection, and compaction.

At runtime, the shell mounts one active workspace app and lets the agent reason over that app's published UI state, data context, dataset objects, and app-specific tools. In this repo, the registered workspace apps are:

- `Demo App` (default)
- `ML Workbench`
- `Personal GL`
- `Local Knowledge`

## Run It

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file. `config/settings.py` reads local environment values directly.

The required values are:

```env
# Default Provider
PROVIDER_NAME=deepseek

# OpenAI API Key
OPENAI_API_KEY="YOUR KEY HERE"

# OpenAI Default Model
OPENAI_MODEL=gpt-4.1-mini

# Gemini API Key
GEMINI_API_KEY="YOUR KEY HERE"

# Gemini Default Model
GEMINI_MODEL=gemini-2.5-flash

# DeepSeek API Key
DEEPSEEK_API_KEY="YOUR KEY HERE"

# DeepSeek Default Model
DEEPSEEK_MODEL=deepseek-chat

# App name - do not change
APP_NAME=AgentShell
```

Then start the shell:

```bash
streamlit run main.py
```

There is also a standalone runner for the ML Workbench app:

```bash
streamlit run run_ml_workbench.py
```

## How The Code Is Organized

- [`main.py`] is the Streamlit entry point. It initializes session state, bootstraps workspace apps, renders the shell sections, and advances the active chat run.
- [`app/workspace_apps/base.py`] defines the contract for any app mounted inside the shell. This is the main integration point if you want to add a new workspace app.
- [`app/workspace_apps/bootstrap.py`] registers the workspace apps that the shell can host.
- [`app/components/workspace_host.py`] owns which workspace app is active and calls its `initialize_state()` and `render()` methods.
- [`agents/factory.py`] wires the assistant runtime together: provider selection, system prompt construction, tool registry, and the optional orchestrated runtime wrapper.
- [`agents/tools/`] contains framework-level tools. The active workspace app can add its own tools through `get_tools()`.
- [`domain/services/`] contains shared data and visualization services used by the tool layer.

## Integration

If you want to mount a new app in AgentShell:

1. Implement the `BaseWorkspaceApp` contract in [`app/workspace_apps/base.py`].
2. Register the app in [`app/workspace_apps/bootstrap.py`].
3. Expose any app-specific tools through `get_tools()`.
4. Expose compact shell-facing context through `get_ui_state()`, `get_data_context()`, and optionally `get_dataset_object()`.

If you want to change agent behavior, start with [`agents/factory.py`], then inspect [`agents/runtime/`] and [`agents/providers/`].

For more detail, read the files in [`docs/`], especially [`docs/important_codebase_files.md`]. That folder contains the deeper architecture notes in this repo.
