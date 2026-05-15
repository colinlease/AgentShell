"""Notes tab renderer."""

from __future__ import annotations

import streamlit as st

from app.workspace_apps.personal_gl.constants import DATASET_NOTES
from app.workspace_apps.personal_gl.runtime import AppRuntime
from app.workspace_apps.personal_gl.services.notes import archive_note_ids, create_note_record, load_active_notes_df, load_note_accounts


def render_notes_tab(runtime: AppRuntime) -> None:
    st.subheader("Notes")

    _, gl_label_by_number = load_note_accounts()
    gl_multiselect_options = list(gl_label_by_number.keys())

    with st.form("notes_create_form", clear_on_submit=True):
        col_form_left, col_form_right = st.columns([2, 3])
        with col_form_left:
            selected_gl_numbers = st.multiselect(
                "Linked GL(s) - Optional",
                options=gl_multiselect_options,
                format_func=lambda gl: gl_label_by_number.get(gl, str(gl)),
            )
        with col_form_right:
            note_text = st.text_area("Note", value="", height=100)

        if st.form_submit_button("Add Note"):
            text_clean = (note_text or "").strip()
            if not text_clean:
                st.error("Note text cannot be empty.")
            else:
                try:
                    create_note_record(text_clean, selected_gl_numbers)
                except Exception as exc:
                    st.error(f"Could not create note: {exc}")
                else:
                    runtime.state.set_active_tab_hint("notes")
                    runtime.state.append_ui_event("note_create")
                    st.success("Note added.")

    df_notes = load_active_notes_df(gl_label_by_number)
    runtime.register_dataset(
        DATASET_NOTES,
        df_notes,
        kind="dataframe",
        description="Active notes",
        metadata={"rows": len(df_notes)},
    )

    if df_notes.empty:
        st.info("No active notes.")
        return

    edited_df = st.data_editor(
        df_notes,
        use_container_width=True,
        hide_index=True,
        column_order=["Created", "GL(s)", "Note", "Archive"],
        column_config={
            "Archive": st.column_config.CheckboxColumn("Archive", help="Check to archive this note.", default=False),
            "Created": st.column_config.TextColumn("Created", disabled=True),
            "GL(s)": st.column_config.TextColumn("GL(s)", disabled=True),
            "Note": st.column_config.TextColumn("Note", disabled=True),
            "id": st.column_config.NumberColumn("id", disabled=True),
        },
        key="notes_editor",
    )

    if st.button("Save Changes", key="notes_save_changes"):
        if edited_df is None or edited_df.empty:
            st.info("No notes to update.")
            return
        archive_mask = edited_df["Archive"] == True
        if not archive_mask.any():
            st.info("No notes selected to archive.")
            return
        note_ids = edited_df.loc[archive_mask, "id"].astype(int).tolist()
        try:
            archive_note_ids(note_ids)
        except Exception as exc:
            st.error(f"Could not archive notes: {exc}")
        else:
            runtime.state.set_active_tab_hint("notes")
            runtime.state.append_ui_event("note_archive")
            st.success(f"Archived {len(note_ids)} note(s).")
            st.rerun()
