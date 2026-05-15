"""Upload panel for the ML Workbench UI."""

from __future__ import annotations

import streamlit as st

from app.workspace_apps.ml_workbench.constants import (
    DEFAULT_PREVIEW_ROW_LIMIT,
    STAGE_PROFILE,
)
from app.workspace_apps.ml_workbench.services.dataset_service import (
    DatasetLoadError,
    load_uploaded_dataset,
)
from app.workspace_apps.ml_workbench.state import set_state_value, update_ui_state
from app.workspace_apps.ml_workbench.ui.layout import render_status_message


def render_upload_panel() -> None:
    """Render the dataset upload panel.

    This panel is intentionally thin. It gathers the uploaded file from the UI
    and delegates the actual loading and artifact creation flow to the dataset
    service.
    """
    with st.container(border=True):
        st.markdown(
            '<div class="mlw-surface-panel-marker" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Upload")
        st.caption(
            "Load a CSV or Excel dataset to begin profiling, preprocessing, "
            "feature engineering, and modeling."
        )

        uploaded_file = st.file_uploader(
            "Upload dataset",
            type=["csv", "xlsx", "xls"],
            key="ml_workbench_file_uploader",
            help="Supported file types: CSV, XLSX, XLS.",
        )

        if uploaded_file is None:
            return

        try:
            load_uploaded_dataset(uploaded_file)
            set_state_value("app_stage", STAGE_PROFILE)
            update_ui_state(preview_row_limit=DEFAULT_PREVIEW_ROW_LIMIT)
            st.rerun()
        except DatasetLoadError as exc:
            render_status_message(str(exc), variant="error")
        except Exception as exc:  # pragma: no cover - defensive branch
            render_status_message(f"Unexpected error while loading dataset: {exc}", variant="error")
