from __future__ import annotations

"""
Future home for general read-only data utility tools.

`run_basic_analysis` has been intentionally removed because it overlapped with
more clearly scoped framework tools such as:
- get_app_context
- get_ui_state
- get_loaded_data_context

This module should eventually contain precise dataset-oriented tools such as:
- get_dataset_sample
- get_dataset_profile
- possibly query_loaded_data

Those tools are a better fit for the shell's long-term ML / EDA automation use
cases than a vague catch-all analysis tool.
"""