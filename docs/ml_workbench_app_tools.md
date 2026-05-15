# ML Workbench App-Specific Tools

This document defines how app-specific agent tools for `ml_workbench` should be designed, where they should live, what contracts they should follow, and how they should interact with the existing AgentShell tool framework.

The goal is to make `ml_workbench` tools:

- modular
- token-efficient
- deterministic
- safe to expose to an agent
- extensible for future workspace apps

This document does not define every concrete tool implementation. It defines the architecture and standards those tools should follow.

## 1. Core Principle

AgentShell already has a good split between:

- shell-wide, always-available tools
- app-specific tools for the active workspace app

ML Workbench tools should use the same tool contract as the existing shell tools, but they should remain owned by the ML Workbench app rather than being added to the general framework tool layer.

The right dependency direction is:

`agent -> app-specific tool wrapper -> ml_workbench service -> app state / artifact registry`

Not:

`agent -> tool -> Streamlit UI widget`

Tools should never depend on UI widget state as their primary implementation boundary.

## 2. Existing Framework Contracts

### 2.1 Base tool contract

All model-callable tools must implement `BaseTool` in:

- [agents/tools/base.py](/Users/colinlease/Desktop/agent_shell/agents/tools/base.py)

Required fields:

- `name`
- `description`
- `schema`
- `run(**kwargs)`

Framework metadata already supported:

- `category`
- `scope`
- `is_read_only`
- `is_enabled_by_default`
- `permission_level`

This is the contract ML Workbench app tools must reuse.

### 2.2 Tool registration

Tools are collected by `ToolRegistry` in:

- [agents/tools/registry.py](/Users/colinlease/Desktop/agent_shell/agents/tools/registry.py)

Important behaviors:

- registers tools by unique `name`
- exposes provider-facing schemas through `list_tool_schemas()`
- exposes framework metadata through `list_tool_metadata()`
- supports filtering by `category` and `scope`

ML Workbench tools should be ordinary `BaseTool` instances that can be inserted into this same registry.

### 2.3 Agent execution loop

Tool execution happens in:

- [agents/core/agent_runner.py](/Users/colinlease/Desktop/agent_shell/agents/core/agent_runner.py)

Important implications:

- the model receives JSON-schema-like tool definitions
- the runner executes one tool call at a time
- tool output is serialized back into the conversation
- tool usage and traces are recorded in session state for Admin

This means ML Workbench tool output must be:

- small
- structured
- JSON-serializable
- meaningful without dumping whole datasets

### 2.4 App-specific tool hook

The workspace app contract already includes:

- `get_tools()` in [app/workspace_apps/base.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/base.py)

This is the correct app-level extension point for app-specific tools.

## 3. Where ML Workbench Tool Code Should Live

ML Workbench app-specific tools should live under the app itself:

- `app/workspace_apps/ml_workbench/tools/`

Recommended file layout:

```text
app/workspace_apps/ml_workbench/tools/
  __init__.py
  context_tools.py
  preprocessing_tools.py
  feature_tools.py
  modeling_tools.py
  results_tools.py
  factory.py
```

### Why this location is correct

These tools are domain-specific.

They should remain close to:

- ML Workbench services in [app/workspace_apps/ml_workbench/services/](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/services)
- ML Workbench state in [app/workspace_apps/ml_workbench/state.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/state.py)
- ML Workbench schemas in [app/workspace_apps/ml_workbench/schemas.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/schemas.py)

They should not live in:

- `agents/tools/`

unless the tool is truly reusable across multiple workspace apps.

## 4. How They Should Be Wired

### 4.1 General tool path

General tools are built in:

- [agents/factory.py](/Users/colinlease/Desktop/agent_shell/agents/factory.py)

Current split:

- `build_general_tools()`
- `build_app_tools()`

ML Workbench tools should be inserted through the app-specific path, not merged into the general tool list.

### 4.2 Recommended future composition flow

The tool assembly flow should become:

1. build shell-wide general tools
2. resolve active workspace app
3. ask active workspace app for app-specific tools via `get_tools()`
4. register all tools in one `ToolRegistry`

That preserves one execution framework while keeping app ownership clear.

### 4.3 Scope metadata

ML Workbench tools should use:

- `scope = "app"`

General framework tools should continue using:

- `scope = "framework"`

This lets Admin and future permissions distinguish:

- always-available shell tools
- active-app-only tools

## 5. What ML Workbench Tools Should Operate On

ML Workbench already has the right internal architecture:

- state model in [state.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/state.py)
- artifact registry in [artifact_service.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/services/artifact_service.py)
- published shell context in [context_service.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/services/context_service.py)
- reusable domain services in:
  - [dataset_service.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/services/dataset_service.py)
  - [preprocessing_service.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/services/preprocessing_service.py)
  - [feature_service.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/services/feature_service.py)
  - [modeling_service.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/services/modeling_service.py)
  - [export_service.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/services/export_service.py)

App-specific tools should call those services directly.

They should not:

- read widget values directly from Streamlit when a service/state helper exists
- manipulate UI presentation objects
- produce HTML as their primary output

## 6. Recommended Tool Categories

### 6.1 Context / inspection tools

Purpose:

- expose app-native ML state in a compact form

Examples:

- `get_ml_workbench_workspace_snapshot`
- `get_ml_workbench_preprocessing_config`
- `get_ml_workbench_feature_specs`
- `get_ml_workbench_candidate_models`
- `get_ml_workbench_results_summary`

Suggested metadata:

- `category = "ml_context"`
- `scope = "app"`
- `is_read_only = True`

### 6.2 Preprocessing mutation tools

Purpose:

- update shared preprocessing rules
- rebuild Working Data

Examples:

- `set_ml_problem_definition`
- `update_ml_drop_columns`
- `update_ml_numeric_imputation`
- `update_ml_categorical_imputation`
- `update_ml_datetime_handling`
- `rebuild_ml_working_data`

Suggested metadata:

- `category = "ml_preprocessing"`
- `scope = "app"`
- `is_read_only = False`

### 6.3 Feature engineering tools

Purpose:

- create, preview, enable, disable, remove engineered features
- automatically rebuild Working Data after non-preview feature mutations so engineered columns stay visible in the dataset

Examples:

- `preview_ml_feature_spec`
- `create_ml_feature_spec`
- `remove_ml_feature_spec`

Suggested metadata:

- `category = "ml_features"`
- `scope = "app"`
- `is_read_only = False`

### 6.4 Modeling tools

Purpose:

- manage candidate models
- train and compare candidates

Examples:

- `create_ml_candidate_model`
- `update_ml_candidate_model`
- `remove_ml_candidate_model`
- `train_ml_candidate_model`
- `train_ml_selected_candidates`
- `select_ml_best_candidate`

Suggested metadata:

- `category = "ml_modeling"`
- `scope = "app"`
- `is_read_only = False`

### 6.5 Results / navigation tools

Purpose:

- summarize persisted modeling outputs
- move the app to the relevant stage

Examples:

- `get_ml_results_summary`
- `get_ml_best_candidate_report`
- `set_ml_active_stage`

Suggested metadata:

- `category = "ml_results"`
- `scope = "app"`

## 7. Output Design Requirements

ML Workbench tools must be token-efficient.

Do not return:

- full dataframes
- large raw row dumps
- full training objects
- large nested JSON blobs copied directly from state

Prefer:

- status
- short message
- key ids / names
- counts
- a compact summary block
- warnings
- next-step hints where useful

### Good output shape

```python
{
    "status": "ok",
    "message": "Updated numeric imputation settings and rebuilt Working Data.",
    "changed": {
        "strategy": "median",
        "columns": ["age", "income"],
    },
    "working_data": {
        "rows": 1240,
        "columns": 18,
        "artifact_name": "working_dataset",
    },
    "warnings": [],
}
```

### Bad output shape

```python
{
    "entire_app_state": {...huge nested object...},
    "working_dataframe_rows": [...hundreds of rows...],
    "all_candidate_run_records": [...full payloads...],
}
```

## 8. Schema Design Standards

All tool schemas should be compact JSON-schema-like dictionaries, similar to the existing tools in:

- [agents/tools/data_tools.py](/Users/colinlease/Desktop/agent_shell/agents/tools/data_tools.py)
- [agents/tools/app_context_tools.py](/Users/colinlease/Desktop/agent_shell/agents/tools/app_context_tools.py)
- [agents/tools/ui_state_tools.py](/Users/colinlease/Desktop/agent_shell/agents/tools/ui_state_tools.py)

### Schema rules

1. Keep argument count low.
2. Prefer explicit enums where possible.
3. Support nullability only where the tool truly needs it.
4. Avoid deep nesting unless the domain requires it.
5. Match ML Workbench state schema names where practical.

### Example: preprocessing tool schema

```python
schema = {
    "type": "object",
    "properties": {
        "strategy": {
            "type": "string",
            "enum": ["mean", "median", "constant"],
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
        },
        "fill_value": {
            "type": ["number", "integer", "null"],
        },
        "rebuild_working_data": {
            "type": ["boolean", "null"],
        },
    },
    "required": ["strategy", "columns"],
}
```

## 9. State and Schema Alignment

ML Workbench tool inputs and outputs should align with the app’s typed schema definitions in:

- [schemas.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/schemas.py)

Important state/schema objects to align with:

- `PreprocessingConfig`
- `FeatureSpec`
- `CandidateModelConfig`
- `ModelComparisonConfig`
- `ModelRunRecord`
- `StatusFlags`
- `AppState`

Tools should not invent alternate field names when a stable schema name already exists in the app.

For example:

- use `target_column`, not `label_column`
- use `candidate_id`, not `model_variant_id`
- use `feature_specs`, not `engineered_features_list`

## 10. Service-First Tool Pattern

Each app tool should be a thin wrapper around one or more app services.

### Preferred pattern

```python
class UpdateNumericImputationTool(BaseTool):
    name = "update_ml_numeric_imputation"
    description = "Update shared numeric imputation settings for ML Workbench."
    category = "ml_preprocessing"
    scope = "app"
    is_read_only = False
    schema = {...}

    def run(self, **kwargs: Any) -> dict[str, Any]:
        service = PreprocessingToolService()
        return service.update_numeric_imputation(...)
```

### Why this is important

It keeps:

- UI logic out of tools
- business logic out of tool wrappers
- future reuse possible from both UI and agent paths

If a needed operation does not yet exist as a reusable service, add or refactor a service first, then wrap it in a tool.

## 11. Recommended New Service Layer for Tooling

ML Workbench has strong domain services already, but some app-specific tool operations will likely benefit from dedicated orchestration services under:

- `app/workspace_apps/ml_workbench/services/agent_tool_service.py`

or split further by domain:

- `preprocessing_tool_service.py`
- `feature_tool_service.py`
- `modeling_tool_service.py`

These service modules should:

- validate tool inputs
- call lower-level ML services
- normalize compact agent-facing outputs

This avoids bloating the existing UI-oriented service calls with agent-format concerns.

## 12. Read vs Write Tool Boundaries

Read-only tools should be the default starting point.

Mutation tools should be added only where:

- the state transition is deterministic
- the effect is easy to summarize
- the UI will reflect the result on rerun

### Read-only tools should:

- inspect current app configuration
- inspect artifacts
- inspect candidate results
- summarize current readiness

### Write tools should:

- perform one clear mutation
- return what changed
- return resulting app status
- avoid chaining too many hidden side effects

## 13. UI Reflection Requirement

For ML Workbench, the UI should visibly reflect tool actions.

That means write tools should usually mutate one of:

- app state
- artifact registry
- persisted candidate run records
- active app stage

So after rerun, the user can open:

- Prepare
- Features
- Models
- Results

and see what the agent changed.

This is already consistent with the current results flow:

- [modeling_panel.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/ui/modeling_panel.py)
- [results_panel.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/ui/results_panel.py)

## 14. First-Pass Tool Inventory

Recommended first implementation wave:

### Read tools

- `get_ml_workbench_snapshot`
- `get_ml_preprocessing_config`
- `get_ml_feature_specs`
- `get_ml_candidate_models`
- `get_ml_results_summary`

### Mutation tools

- `set_ml_problem_definition`
- `update_ml_drop_columns`
- `update_ml_numeric_imputation`
- `update_ml_categorical_imputation`
- `create_ml_feature_spec`
- `remove_ml_feature_spec`
- `create_ml_candidate_model`
- `update_ml_candidate_model`
- `train_ml_candidate_model`
- `train_ml_selected_candidates`
- `set_ml_active_stage`

This is enough to support the intended workflow:

“Look at my loaded data, compare model/spec performance, adjust preprocessing/features/settings, and show me results.”

## 15. Naming Standards

Use names that are:

- explicit
- app-prefixed
- action-oriented

Recommended naming style:

- `get_ml_*`
- `update_ml_*`
- `create_ml_*`
- `remove_ml_*`
- `train_ml_*`
- `set_ml_*`

Examples:

- `get_ml_results_summary`
- `update_ml_numeric_imputation`
- `train_ml_selected_candidates`

Avoid vague names like:

- `run_model`
- `configure_data`
- `change_settings`

## 16. What Not To Do

Do not:

- put ML Workbench tools in `agents/tools/` unless they are truly cross-app
- make tools depend on widget-local session keys when a domain service exists
- return huge state dumps
- return whole datasets
- encode UI styling/presentation logic in tool output
- create one giant “do everything” ML tool

Avoid tools with mixed responsibility like:

- “profile data, mutate preprocessing, train models, export results”

Prefer multiple small tools that compose.

## 17. Summary

The correct architecture for ML Workbench app-specific tools is:

- code lives in `app/workspace_apps/ml_workbench/tools/`
- tools subclass `BaseTool`
- tools are registered through the app-specific tool path
- tools use `scope = "app"`
- tools wrap reusable ML services
- tools return compact structured outputs
- tools align with ML Workbench state/schema names
- tools mutate state/artifacts in ways the UI can reflect clearly

If these rules are followed, ML Workbench can gain powerful agentic control without turning the tool layer into a second ad hoc app implementation.
