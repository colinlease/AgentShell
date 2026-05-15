"""Logs tab renderer."""

from __future__ import annotations

import streamlit as st

from app.workspace_apps.personal_gl.constants import DATASET_LOGS
from app.workspace_apps.personal_gl.runtime import AppRuntime
from app.workspace_apps.personal_gl.services.logs import load_event_types, load_logs_dataframe


def render_logs_tab(runtime: AppRuntime) -> None:
    st.subheader("Logs")

    event_types = load_event_types()
    filt_col1, filt_col2, filt_col3 = st.columns([1.4, 1.6, 1.0])
    with filt_col1:
        enable_date_filter = st.checkbox("Filter by date range", key="logs_date_filter", value=False)
        start_date = st.date_input("Start date", key="logs_start_date") if enable_date_filter else None
        end_date = st.date_input("End date", key="logs_end_date") if enable_date_filter else None
    with filt_col2:
        if event_types:
            selected_event_types = st.multiselect(
                "Event types",
                options=event_types,
                default=[],
                key="logs_event_types",
                help="Leave empty to include all event types.",
            )
        else:
            selected_event_types = []
            st.info("No events have been logged yet.")
    with filt_col3:
        max_events = st.selectbox("Max events", options=[50, 100, 250, 500], index=0, key="logs_max_events")

    df_logs = load_logs_dataframe(
        enable_date_filter=enable_date_filter,
        start_date=start_date,
        end_date=end_date,
        selected_event_types=selected_event_types,
        max_events=max_events,
    )
    runtime.register_dataset(
        DATASET_LOGS,
        df_logs,
        kind="dataframe",
        description="Rendered activity log rows",
        metadata={"rows": len(df_logs)},
    )

    if df_logs.empty:
        st.info("No activity has been logged yet for the selected filters.")
        return

    for _, row in df_logs.iterrows():
        st.code(row["display_text"], language="bash")
        st.markdown("&nbsp;")
