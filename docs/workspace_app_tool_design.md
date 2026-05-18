# Workspace App Tool Design

This document is for people adding a new workspace app to AgentShell and deciding what app-specific tools that app should expose.

The goal is not to maximize tool count. The goal is to expose the smallest set of tools that lets the agent understand the app, operate on its real domain objects, and make visible, reliable changes without coupling tool behavior to UI widget code.

## 1. Core Model

AgentShell already separates:

- shell-wide framework tools
- app-specific tools for the active workspace app

That split should remain strict.

The preferred dependency direction is:

`agent -> app-specific tool wrapper -> app service/orchestration layer -> app state / domain objects / persisted artifacts`

Not:

`agent -> tool -> Streamlit widget tree`

If a tool's real implementation boundary is "whatever the current widget values happen to be," the design is probably wrong.

## 2. Existing Contracts You Should Reuse

### 2.1 Workspace app contract

Every mounted app implements [app/workspace_apps/base.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/base.py).

The most relevant hooks are:

- `get_ui_state()`
- `get_data_context()`
- `get_dataset_object()`
- `get_tools()`

Those hooks define what the shell can ask from an app before any app-specific tools exist.

### 2.2 Tool contract

All model-callable tools must implement `BaseTool` from [agents/tools/base.py](/Users/colinlease/Desktop/agent_shell/agents/tools/base.py).

Required interface:

- `name`
- `description`
- `schema`
- `run(**kwargs)`

Important metadata already supported:

- `category`
- `scope`
- `is_read_only`
- `is_enabled_by_default`
- `permission_level`

App-specific tools should reuse this contract directly. They should not invent a parallel tool system.

### 2.3 Tool registration and execution

Tool registration lives in [agents/tools/registry.py](/Users/colinlease/Desktop/agent_shell/agents/tools/registry.py). Tool execution lives in [agents/core/agent_runner.py](/Users/colinlease/Desktop/agent_shell/agents/core/agent_runner.py).

That means app tools must be:

- JSON-serializable
- compact
- deterministic
- understandable from their returned payload alone

### 2.4 App-specific insertion point

General and app-specific tools are composed in [agents/factory.py](/Users/colinlease/Desktop/agent_shell/agents/factory.py). The correct app boundary is still `get_tools()` on the active workspace app, not `agents/tools/` unless the tool is truly reusable across multiple apps.

## 3. When To Add A New App-Specific Tool

Before adding a tool, ask whether the existing framework tools already solve the problem.

Use general tools when the task is:

- shell context
- current UI state
- loaded resource summary
- generic dataframe inspection, sampling, aggregation, or chart prep

Add an app-specific tool only when the agent needs one of these:

- domain state that generic tools cannot describe well
- a workflow mutation specific to the app
- a summary of app-native artifacts or results
- access to domain objects that are not just generic tables

Good app-specific tool reasons:

- "Set the comparison metric for the modeling workflow."
- "Publish the currently selected document chunk labels."
- "Create a backtest run for the selected strategy configuration."
- "Advance the workflow to the review stage."

Bad app-specific tool reasons:

- "Return the current dataframe again, but from this app."
- "Tell the agent what tab the user is on" when `get_ui_state()` already does that.
- "Update a text input widget value" when there is no durable domain mutation behind it.

## 4. Design Around Durable App Boundaries

App tools should operate on durable app concepts, not transient UI implementation details.

Good boundaries:

- persisted app state
- workflow configuration
- domain entities
- artifact registries
- result records
- job/run definitions
- selected stage or review status

Avoid making tools depend directly on:

- Streamlit widget keys
- ad hoc container layout
- presentation-only toggles with no domain effect
- HTML fragments
- full session-state dumps

If a useful action exists only inside a large UI callback today, extract a service first, then wrap that service in a tool.

## 5. Where Tool Code Should Live

App-specific tools should live inside the app package:

- `app/workspace_apps/<app_id>/tools/`

Recommended layout:

```text
app/workspace_apps/<app_id>/
  __init__.py
  <app_module>.py
  state.py
  schemas.py                  # optional but strongly recommended
  services/
    __init__.py
    context_service.py
    agent_tool_service.py
    ... domain services
  tools/
    __init__.py
    factory.py
    context_tools.py
    mutation_tools.py
    results_tools.py
    ... focused domain tool modules
```

This keeps tools close to the app's own:

- state model
- typed schemas
- domain services
- artifact and persistence logic

Use [app/workspace_apps/ml_workbench/tools/](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/tools) and [app/workspace_apps/ml_workbench/services/agent_tool_service.py](/Users/colinlease/Desktop/agent_shell/app/workspace_apps/ml_workbench/services/agent_tool_service.py) as a concrete reference, but not as a required shape for every app.

## 6. Tool Categories That Generalize Across Apps

Most apps do not need many categories. Start with the smallest set that matches the workflow.

### 6.1 Context tools

Purpose:

- expose compact domain-specific state the agent cannot infer from generic shell context

Examples across app types:

- `get_research_review_config`
- `get_annotation_label_set`
- `get_strategy_backtest_setup`
- `get_ml_preprocessing_config`

Typical metadata:

- `scope = "app"`
- `is_read_only = True`

### 6.2 Mutation tools

Purpose:

- update a durable piece of workflow configuration or state

Examples across app types:

- `set_annotation_guidelines`
- `update_report_filters`
- `upsert_strategy_candidate`
- `update_ml_preprocessing_config`

Typical metadata:

- `scope = "app"`
- `is_read_only = False`

### 6.3 Execution tools

Purpose:

- trigger a bounded workflow step that produces app-native outputs

Examples:

- `run_document_chunking`
- `train_candidate_models`
- `execute_backtest`
- `generate_evaluation_report`

These should exist only when the side effect is real, bounded, and easy to summarize.

### 6.4 Results tools

Purpose:

- summarize artifacts, runs, reports, evaluations, or review states

Examples:

- `get_backtest_summary`
- `get_annotation_progress`
- `get_candidate_result_details`
- `get_export_status`

### 6.5 Navigation or stage tools

Purpose:

- move the app to a user-visible workflow stage when that stage matters to collaboration

Examples:

- `set_active_review_stage`
- `open_results_stage`

These should be used sparingly. They are worthwhile only when stage changes are meaningful and visible after rerender.

## 7. Tool Granularity Rules

A tool should do one clear thing.

Prefer:

- `update_model_comparison_settings`
- `remove_candidate_model`
- `get_results_summary`

Avoid:

- `configure_everything_for_training`
- `sync_ui_and_recompute_and_export`
- `do_next_step`

The model performs better when tool names map to explicit domain actions with explicit outputs.

## 8. Read vs Write Boundaries

Start read-only. Add write tools only when the mutation is safe and worth exposing.

Read tools should:

- inspect configuration
- inspect available artifacts
- inspect current readiness or status
- summarize prior results

Write tools should:

- perform one durable mutation
- return what changed
- return the resulting status
- avoid hidden side effects unless those side effects are core to the operation

A useful rule:

- if the user would struggle to see what changed after a rerun, the tool boundary is probably too opaque

## 9. Output Design Requirements

Tool outputs must be token-efficient. Do not dump raw state or raw data unless the payload is genuinely tiny.

Do not return:

- full dataframes
- full session state
- full model objects
- large nested records copied straight from persistence
- prose-heavy explanations when a compact payload would do

Prefer:

- `status`
- `message`
- compact summary blocks
- changed ids, names, counts, and modes
- warnings
- next-step hints when they materially help the agent

Good output shape:

```python
{
    "status": "ok",
    "message": "Updated comparison settings.",
    "changed": {
        "metric": "f1",
        "cross_validation_folds": 5,
    },
    "readiness": {
        "can_train": True,
        "candidate_count": 3,
    },
    "warnings": [],
}
```

Bad output shape:

```python
{
    "entire_app_state": {...},
    "all_intermediate_objects": [...],
    "raw_dataframe_rows": [...hundreds of rows...],
}
```

## 10. Schema Design Standards

Use the same compact JSON-schema-like style as the existing framework tools in:

- [agents/tools/data_tools.py](/Users/colinlease/Desktop/agent_shell/agents/tools/data_tools.py)
- [agents/tools/app_context_tools.py](/Users/colinlease/Desktop/agent_shell/agents/tools/app_context_tools.py)
- [agents/tools/ui_state_tools.py](/Users/colinlease/Desktop/agent_shell/agents/tools/ui_state_tools.py)

Rules:

1. Keep argument count low.
2. Prefer explicit enums for constrained choices.
3. Use nullable fields only where the operation actually supports null.
4. Avoid deep nesting unless the domain requires it.
5. Align field names with the app's real schemas and services.
6. Set `"additionalProperties": False` when you want a tightly bounded contract.

Example:

```python
schema = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},
        "metric": {
            "type": "string",
            "enum": ["accuracy", "f1", "roc_auc"],
        },
        "notes": {"type": ["string", "null"]},
    },
    "required": ["candidate_id", "metric"],
    "additionalProperties": False,
}
```

## 11. Naming And Schema Alignment

Tool names should be explicit and app-namespaced.

Prefer:

- `get_ml_results_summary`
- `update_annotation_guidelines`
- `run_backtest_job`

Avoid:

- `get_results`
- `update_settings`
- `run_process`

Input and output field names should match the app's typed concepts where possible. If the app uses `candidate_id`, do not invent `variant_id` in the tool layer. If the app uses `target_column`, do not rename it to `label_field` just for the agent.

Consistency matters because:

- services stay simpler
- prompts can describe stable terminology
- tool outputs are easier to chain together

## 12. Service-First Tool Pattern

Tool classes should be thin wrappers over services or orchestration helpers.

Preferred pattern:

```python
class UpdateComparisonSettingsTool(BaseTool):
    name = "update_model_comparison_settings"
    description = "Update model comparison settings for the active app."
    category = "modeling"
    scope = "app"
    is_read_only = False
    schema = {...}

    def __init__(self, service: AppToolService) -> None:
        self._service = service

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self._service.update_comparison_settings(**kwargs)
```

The service layer should own:

- validation
- id resolution
- business rules
- orchestration across lower-level services
- normalization of compact agent-facing outputs

The tool wrapper should mainly own:

- metadata
- schema
- forwarding the call

## 13. Prompt And Tool Design Should Match

If a tool exists, the app prompt should tell the model when to use it and what it is for.

Relevant code:

- [agents/prompts/app_prompts.py](/Users/colinlease/Desktop/agent_shell/agents/prompts/app_prompts.py)
- [agents/prompts/runtime_prompts.py](/Users/colinlease/Desktop/agent_shell/agents/prompts/runtime_prompts.py)

The prompt layer should explain:

- what the app's main domain objects are
- when to prefer general shell tools
- when to prefer app-specific tools
- what order to inspect state in before mutating anything

If prompt guidance and tool boundaries disagree, the agent will use the tools poorly.

## 14. A Good First Pass For A New App

For most new apps, the first tool wave should be small:

1. One or two read-only context tools.
2. One mutation tool for the highest-value durable setting.
3. One results tool for the main persisted output.

Only add more when there is a clear repeated need.

That keeps the surface area understandable while you learn:

- which domain concepts the model actually needs
- which mutations are reliable
- which outputs are still too large or ambiguous

## 15. Practical Checklist

Before shipping a new app-specific tool, confirm:

1. The tool acts on a real domain boundary, not a widget boundary.
2. The tool name is explicit and unique.
3. The schema is small and provider-friendly.
4. The output is compact and JSON-serializable.
5. The mutation is visible after rerender, if it is a write tool.
6. The business logic lives in services, not in the tool wrapper.
7. The app prompt tells the model when to use the tool.

If those seven points hold, the tool design is probably sound.

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
