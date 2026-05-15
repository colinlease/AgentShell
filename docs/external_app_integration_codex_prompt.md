# External App Integration Codex Prompt

Use the prompt below as the handoff prompt for another Codex instance that needs to integrate a new external Streamlit app into AgentShell.

```text
You are working inside the AgentShell repository. Your task is to take a first pass at integrating an existing external Streamlit app into AgentShell as a new workspace app.

Target inputs
- External app name: <EXTERNAL_APP_NAME>
- External app source path or module path: <SOURCE_APP_PATH>
- New AgentShell app_id: <TARGET_APP_ID>
- New AgentShell app_label: <TARGET_APP_LABEL>
- Default app status: non-default unless explicitly told otherwise

Your objective
- Mount the external app inside AgentShell as a `BaseWorkspaceApp`.
- Expose compact shell-facing UI state and data context for the app.
- Add app-specific agent tools that let the model inspect and manipulate the app safely.
- Add an app-specific prompt layer so the agent knows when to use general shell tools vs app tools.
- Preserve the external app's existing workflow as much as possible. Do not rewrite the whole app into a new architecture unless required for the integration.

Work from the real codebase. Do not invent contracts. Start by inspecting these files:
- `app/workspace_apps/base.py`
- `app/workspace_apps/registry.py`
- `app/workspace_apps/bootstrap.py`
- `app/components/workspace_host.py`
- `agents/factory.py`
- `agents/tools/base.py`
- `agents/tools/registry.py`
- `agents/prompts/app_prompts.py`
- `agents/prompts/runtime_prompts.py`
- `agents/runtime/catalog.py`
- `app/state/ui_state.py`
- `app/workspace_apps/demo_app/demo_app.py`
- `app/workspace_apps/ml_workbench/ml_app.py`
- `app/workspace_apps/ml_workbench/services/context_service.py`
- `app/workspace_apps/ml_workbench/services/agent_tool_service.py`
- `app/workspace_apps/ml_workbench/tools/factory.py`
- `app/workspace_apps/ml_workbench/tools/`
- `docs/ml_workbench_app_tools.md`
- relevant tests under `tests/`

Current shell architecture
1. AgentShell registers workspace apps in `app/workspace_apps/bootstrap.py`.
2. The active app is mounted through `app/components/workspace_host.py`.
3. Every workspace app must implement `BaseWorkspaceApp` from `app/workspace_apps/base.py`.
4. General tools and app-specific tools are merged in `agents/factory.py` into one `ToolRegistry`.
5. Provider-facing tool schemas are generated from `BaseTool.to_provider_schema()`.
6. App-specific prompt text is selected by `get_app_prompt()` in `agents/prompts/app_prompts.py`.

Existing general tools already available across apps
- `get_app_context`
- `get_loaded_data_context`
- `get_dataset_profile`
- `get_dataset_sample`
- `get_dataset_aggregation`
- `get_dataset_chart`
- `get_ui_state`

Do not recreate those as app-specific tools. Reuse them when the app can expose dataset objects and compact data context cleanly.

Required workspace app contract
Implement a new app adapter that satisfies `BaseWorkspaceApp`:
- `app_id: str`
- `app_label: str`
- `app_type: str = "streamlit"`
- `initialize_state(self) -> None`
- `render(self) -> None`
- `get_ui_state(self) -> dict[str, Any]`
- `get_data_context(self) -> dict[str, Any]`
- `get_dataset_object(self, dataset_name: str | None = None) -> Any | None`
- `get_tools(self) -> list[Any]`

Integration principles
- Keep the app adapter thin.
- Keep tool classes thin.
- Put business logic in deterministic app services, not in Streamlit widget code.
- Do not make tools depend directly on widget layout code.
- Do not dump raw DataFrames or huge payloads into tool outputs.
- Prefer compact, structured, JSON-serializable responses.
- Preserve the external app's current user workflow and terminology where reasonable.

New app file layout
Create a new package under `app/workspace_apps/<TARGET_APP_ID>/`.
Recommended structure:

app/workspace_apps/<TARGET_APP_ID>/
  __init__.py
  constants.py
  manifest.py
  state.py
  <target_module>.py                # thin BaseWorkspaceApp adapter
  services/
    __init__.py
    context_service.py
    agent_tool_service.py
    ... domain helpers extracted as needed
  tools/
    __init__.py
    factory.py
    context_tools.py
    mutation_tools.py
    results_tools.py
    ... additional focused tool modules as needed
  ui/
    ... only if you need to split rendering or preserve imported app pieces
  schemas.py                        # optional but recommended for typed contracts

If the external app already has a clean internal package, adapt around it instead of copying large amounts of code. Only extract helpers when required to make the shell integration stable.

Shell-facing data contracts
1. `get_ui_state()` must return compact user-visible app state.
   Include only things the agent needs to reason about what the user is looking at now:
   - active internal page/tab/step
   - selected objects or records
   - active filters/searches
   - loaded resource ids or names
   - status indicators
   - open panels / mode toggles if important
   Do not dump the entire session state.

2. `get_data_context()` must return compact loaded-resource context.
   Include:
   - `has_data` or similar boolean
   - active resource identifiers
   - list of datasets/tables/documents/entities currently available
   - row/column counts when dataframes exist
   - field names / schema summaries
   - compact modeling/workflow/resource summary blocks when appropriate

3. `get_dataset_object()` should return the active dataframe-like object when the app has one, so general shell data tools can work automatically. If the app has multiple named datasets, support `dataset_name`.

App-specific tool contract
All app tools must subclass `BaseTool` from `agents/tools/base.py`.

Every tool must define:
- `name`
- `description`
- `schema`
- `category`
- `scope = "app"`
- `is_read_only`
- `is_enabled_by_default`
- `permission_level`
- `run(**kwargs)`

Tool naming rules
- Prefix tool names with a stable app-specific namespace.
- Prefer `<verb>_<target_app_id>_<resource>` or a clear domain-specific equivalent.
- Names must be unique across the whole shell.
- Be explicit. Avoid generic names like `update_settings`.

Tool schema rules
- Use JSON-schema-like dicts consistent with existing tools.
- Top-level schema should almost always be:
  - `"type": "object"`
  - `"properties": {...}`
  - `"required": [...]`
- Prefer simple field types: `string`, `number`, `integer`, `boolean`, `array`, `object`.
- Nullable fields may use `["string", "null"]` style, matching existing code.
- Prefer `enum` for constrained values.
- Use `items` for arrays.
- Keep schemas provider-friendly. Avoid `oneOf`, `allOf`, and `anyOf` unless absolutely necessary.
- Avoid complex nested unions that Gemini normalization will strip or degrade.
- Set `"additionalProperties": False` for tightly bounded objects when practical.
- Only accept parameters you actually support.

Tool output contract
Return compact dicts, not prose blobs.

Preferred patterns:
- Read tool:
  - `{"status": "ok", "<payload_key>": ..., "warnings": []}`
- Mutation tool:
  - `{"status": "ok", "message": "...", "<payload_key>": ..., "warnings": []}`
- Validation failure:
  - `{"status": "error", "message": "...", "errors": [{"field": "...", "reason": "..."}]}`

Tool outputs must be:
- deterministic
- JSON-serializable
- compact
- meaningful without extra explanation

Service-layer rules
Create an app-specific agent tool service similar to `MLWorkbenchToolService`.
Responsibilities:
- normalize model inputs
- validate ids, fields, modes, and enum values
- translate tool requests into app state mutations or reads
- return compact normalized response dicts

Do not put these responsibilities in tool classes:
- heavy validation logic
- business rules
- workflow orchestration
- direct widget interactions

General vs app-specific tool boundary
Use existing general tools for:
- shell context
- app context
- loaded data context
- dataframe profiling
- sampling
- aggregation
- charting

Create app-specific tools only for:
- domain state the general tools cannot express
- app workflow mutations
- persisted app configuration
- domain results summaries
- domain objects that are not plain dataframe inspection problems

Expected app-specific tool categories
Adjust names to the app domain, but keep the pattern:
- `*_context`: read persisted app workflow/setup/configuration
- `*_mutation`: update app setup, selections, parameters, or state
- `*_results`: inspect outputs, artifacts, evaluations, statuses
- optional `*_actions`: trigger expensive domain operations intentionally

Minimum app-specific tool set for a first pass
Implement a small but useful tool surface, usually 4-10 tools total:
1. One workspace snapshot or setup inspection tool.
2. One or more focused config/state mutation tools.
3. One or more result/artifact inspection tools.
4. One action tool only if the app has an explicit, meaningful action such as run/train/refresh/execute/export.

Do not expose every UI control as its own tool on the first pass. Group related mutations into coherent domain tools.

App prompt work
Add a dedicated prompt block in `agents/prompts/app_prompts.py` and wire it through `get_app_prompt(<TARGET_APP_ID>)`.

The app prompt must teach the model:
- what the app is for
- the app's main workflow sections
- the difference between shell context and app context
- which app-specific tools map to which workflow areas
- when to use general data tools instead of app tools
- any major constraints or unsupported operations
- never invent ids, columns, resources, or parameter values

Registration work
You must wire the app into the shell:
- register the app in `app/workspace_apps/bootstrap.py`
- ensure it is discoverable through the workspace registry
- ensure `get_tools()` returns the app-specific tool list
- ensure prompt routing recognizes the app_id

Implementation process
1. Inspect the external app and identify:
   - its main workflow stages
   - its state containers
   - its primary data/resources
   - its deterministic business operations
   - which operations are safe for agent exposure
2. Build the thin workspace app adapter.
3. Publish `get_ui_state()` and `get_data_context()`.
4. Expose `get_dataset_object()` if the app has dataframe-like resources.
5. Create an app-specific service layer for tool-safe reads/mutations.
6. Implement a small set of app-specific `BaseTool` wrappers.
7. Add a `tools/factory.py` builder.
8. Register the app in bootstrap.
9. Add the app-specific prompt block.
10. Add or update tests.
11. Add a short docs file if the integration introduces non-obvious tool conventions.

Testing requirements
Add focused tests for the new integration. At minimum cover:
- workspace app registration
- `get_app_prompt(<TARGET_APP_ID>)`
- tool registry includes general + app tools when the app is active
- app-specific tools expose valid provider-facing schemas
- at least one read tool happy path
- at least one mutation tool happy path
- at least one validation failure path

If existing tests provide better patterns, follow them rather than inventing a new style.

Acceptance criteria
- The new app mounts inside AgentShell without breaking the existing apps.
- The app exposes useful `get_ui_state()` and `get_data_context()` snapshots.
- General shell data tools work when the app has datasets.
- App-specific tools are small, coherent, and provider-compatible.
- Tool outputs are compact and structured.
- The app has a dedicated prompt layer.
- The integration is modular enough to extend later.

Implementation constraints
- Do not refactor unrelated apps.
- Do not move general-purpose shell code unless the integration truly requires it.
- Do not revert unrelated changes in the repo.
- If the external app architecture is messy, add a thin adapter/service layer rather than trying to clean up everything.
- If a capability is ambiguous or unsafe to expose, skip it and document the gap briefly.

Expected final deliverable
Make the code changes directly. Then provide a concise summary including:
- what files were added/changed
- what app-specific tools were introduced
- what shell contracts were implemented
- what remains incomplete or approximate in this first pass
- what tests were run

If critical information about `<SOURCE_APP_PATH>` is missing or the app cannot be located, stop early and state exactly what is missing.
```
