# Using AgentShell

This guide is for people who have already launched AgentShell and want to understand the main UI, the assistant controls, and the runtime/debug surfaces that are available today.

It is intentionally practical rather than exhaustive. Everything here is grounded in the current codebase.

## What You See When The App Starts

AgentShell has four top-level sections:

- `Workspace`: the currently mounted workspace app
- `Agent`: the full-page assistant
- `Admin`: runtime controls, tool visibility, notes, trace, and performance
- `About`: a short shell overview

The workspace apps currently registered in this repo are:

- `Demo App`
- `ML Workbench`
- `Personal GL`
- `Local Knowledge`

The mounted workspace app is controlled by the shell, but each app owns its own internal UI and app-specific tools.

## The Control Rail

The floating control rail at the top is the fastest way to change shell-wide context.

From left to right, it includes:

- `Workspace`: switches the active workspace app
- `Model`: switches the active provider/model pair
- `Theme`: toggles light and dark mode
- `Planning`: toggles hidden planning on or off
- `Reflection`: toggles post-run reflection on or off
- `Assistant`: opens or closes the docked assistant pane

Notes:

- The `Model` selector only shows providers that are actually configured with API keys.
- The model selector is provider-aware. The visible option format is `Provider · model`.
- If the assistant is open while you are in `Workspace`, `Admin`, or `About`, it appears as a docked pane next to the main content.
- If you switch to the `Agent` tab, the assistant takes the full page instead.

## Working With The Assistant

You can talk to the assistant either in:

- the full-page `Agent` tab
- the docked assistant pane opened from the control rail

The same runtime is used in both places. The active workspace, its exposed context, and its currently registered tools are what ground the assistant.

### Asking For Charts

If the active workspace exposes a dataset object, you can ask the assistant for charts directly in chat.

The chart tool currently supports:

- `bar`
- `line`
- `scatter`
- `histogram`

Practical examples:

- "Show a bar chart of revenue by category."
- "Plot monthly sales as a line chart."
- "Make a histogram of age."
- "Show a scatter plot of income vs spend."

A few current constraints matter:

- the assistant builds one chart per chart-tool call
- histogram bins are capped
- scatter plots are capped to keep point counts readable
- you need real column names from the loaded dataset

## Finding The Tools For The Current Workspace

Open `Admin` and then the `Tools` subtab.

That view shows:

- current-session tool call counts
- recent tool events
- all currently registered tools

The available tools are grouped into:

- `General Tools`: shell-wide tools available across workspace apps
- `App-Specific Tools`: tools exposed by the active workspace app
- `Runtime Notes Tools`: note tools used by planning and reflection

App-specific tools change when you switch workspaces.

To see what a tool does, hover over its badge in the `Available Tools` section. The tooltip shows the tool description currently registered for the agent.

## The Admin Tab

The `Admin` tab has five subtabs:

- `Agent`
- `Tools`
- `Notes`
- `Trace`
- `Performance`

### Agent

This is the main runtime control surface.

It includes three selectors:

- `Choose workspace app`
- `Choose provider`
- `Choose model`

The workspace selector mirrors the control rail workspace switch. The provider/model selectors mirror the control rail model selector.

It also includes three runtime toggles:

- `Enable Planning`
- `Enable Reflection`
- `Enable Context Compaction`

Those settings are persisted to `config/agent_runtime_settings.json`.

#### What Planning Does

Planning adds a hidden pre-execution phase before the visible assistant run.

In the current runtime:

- triage decides whether planning should be skipped, light, or deep
- light planning adds a plan before execution
- deep planning adds both planning and critique before execution
- critique is not a separate toggle

Planning does not execute normal app, data, or UI tools during the planning phase.

#### What Reflection Does

Reflection adds a hidden post-run phase that can update runtime notes.

In the current runtime:

- reflection runs after the visible execution ends
- it uses note tools only
- it does not execute normal app, data, or UI tools
- it can be forced when provider-turn budget is exhausted, a provider error happens, or tool usage exceeds the reflection threshold
- it also runs when one or more tool calls fail

#### What Context Compaction Does

Context compaction summarizes older chat history into a hidden summary once thresholds are reached, while keeping a smaller set of recent raw messages.

The `Compacted Context Summary` panel in the `Agent` subtab shows the current hidden summary when compaction has already run in this session.

### Runtime Limits And Gates

The `Agent` subtab also exposes the current runtime budgets.

#### Planning

- `Triage Model Calls`: max hidden model calls for triage
- `Planning Model Calls`: max hidden model calls during planning
- `Planning Note Reads`: max runtime-note reads during planning
- `Critique Model Calls`: max hidden model calls for critique after deep planning

#### Execution

- `Provider Turns`: max visible provider turns during the run
- `Tool Calls`: max execution-phase tool calls
- `Note Reads`: max runtime-note reads during execution

#### Reflection

- `Reflect After Tool Calls`: force reflection only when a completed run uses more than this many tools
- `Reflection Model Calls`: max hidden model calls during reflection
- `Reflection Tool Calls`: max note-tool calls during reflection
- `Reflection Note Writes`: max notes reflection may create or update in one run
- `Reflection Note Deletes`: max notes reflection may delete in one run

#### Context Compaction

- `Trigger After Messages`: compaction message threshold
- `Keep Recent Raw Messages`: how many recent raw messages stay uncompressed
- `Max Summary Chars`: max compacted summary size
- `Compaction Model Calls`: max hidden model calls during compaction
- `Compaction Char Trigger`: optional character-count trigger for compaction

### Tools

Use this subtab when you want to answer:

- "What tools are available right now?"
- "What tools has the assistant used in this session?"
- "Which tools belong to the current workspace?"

This is the easiest place to confirm that switching workspaces also changed the active app-specific tools.

### Notes

This subtab summarizes runtime notes and recent note maintenance activity.

It shows:

- note file count
- total note count
- general note count
- app note count
- note file names
- recent note upserts/deletes recorded in recent runs

Important:

- runtime notes are heuristic reminders, not source-of-truth data
- the Notes subtab is a summary/debug view, not a full note editor

### Trace

Use `Trace` to understand what happened in specific runs.

Each recent run shows:

- provider
- model
- stop reason
- step count
- tool count
- whether planning, reflection, and compaction happened
- the final assistant response

You can expand `Trace` on a run to inspect grouped details for:

- planning
- execution
- reflection

This is the best place to debug tool order, failed tools, or unexpected stopping behavior on a single run.

### Performance

Use `Performance` for aggregated trends across daily summary logs.

It reads from `logs/agent_metrics/` and summarizes:

- run outcomes
- average speed
- tool health
- planning/reflection usage rates
- tool issue rates

The date-range selector currently supports:

- `Today`
- `Last 7 days`
- `This week`
- `This month`
- `All time`

If no daily summary files exist yet, this view will be empty.

## Runtime Notes And Where They Live

Runtime notes are stored under the project root in:

- `runtime_notes/general.json`
- `runtime_notes/apps/<app_id>.json`

That folder is file-backed and is created as note files are written. If reflection has not created any notes yet, you may not see much there.

The runtime note store is not intended to override current app state, current tool outputs, or explicit user instructions. It is a heuristic memory layer for future runs.

## Logs And How To Use Them

AgentShell writes run logs automatically from the Streamlit chat adapter path.

There are two main log outputs:

- `logs/agent_runs/YYYY-MM-DD.jsonl`: one JSON record per completed run
- `logs/agent_metrics/YYYY-MM-DD.summary.json`: daily aggregated metrics built from those run logs

How to use them:

- use `Admin > Trace` when you want to inspect one recent run in detail
- use `Admin > Performance` when you want daily rates and trends
- use `logs/agent_runs/` when you want raw per-run records for offline inspection
- use `logs/agent_metrics/` when you want the pre-aggregated daily summary files that drive the Performance view

The raw run records include compact runtime, outcome, orchestration, trace, and tool summaries. The summary files roll those into counts and rates.

Some workspace apps may also expose their own logs UI. In the current repo, `Personal GL` includes a `Logs` tab inside the workspace with:

- an optional date-range filter
- an event-type filter
- a max-events selector

## Performance, Cost, And Speed Tradeoffs

The shell gives you a few real levers here.

### Model Choice

Changing the active model is usually the biggest tradeoff.

In general:

- smaller/faster models usually reduce latency and spend
- larger/stronger models usually help with harder multi-step tasks

The code does not encode provider pricing, so choose based on your own provider account and tolerance for latency.

### Planning

Turning on planning can improve multi-step behavior, but it adds hidden model work before the visible answer.

Use it when:

- the task has several steps
- tool order matters
- the assistant needs to inspect context before acting

Leave it off when:

- you want the fastest path for simple requests
- you are doing short, obvious turns

### Reflection

Reflection can improve future runs by maintaining runtime notes, but it adds hidden post-run work.

Use it when:

- you want the system to accumulate heuristics over time
- you are debugging repeated tool-order or context mistakes

Lower the reflection threshold or keep reflection enabled if you want more note maintenance. Raise the threshold or disable reflection if you want less hidden work.

### Execution Budgets

Higher execution budgets give the assistant more room to recover, inspect, and continue, but they also allow longer runs.

The most important execution knobs are:

- `Provider Turns`
- `Tool Calls`

Raise them for harder tasks. Lower them if you want shorter, cheaper runs.

### Context Compaction

Compaction can help long conversations by shrinking old history into a summary, but compaction itself is also a hidden runtime phase.

If you keep long chats open, compaction is useful. If you mainly work in short chats, the default behavior is usually enough.

## A Simple Starting Setup

If you just want to get started:

1. Pick a workspace from the control rail.
2. Open the assistant.
3. Ask the assistant to inspect the current app state before making changes.
4. Use `Admin > Tools` to see what the active workspace exposed.
5. Turn on `Planning` for harder tasks.
6. Turn on `Reflection` only when you want runtime-note learning and extra post-run diagnostics.
7. Use `Trace` for one-run debugging and `Performance` for trend monitoring.

That is enough to start using the shell productively without touching every advanced setting on day one.
