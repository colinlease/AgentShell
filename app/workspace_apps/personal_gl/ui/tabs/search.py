"""Search tab renderer."""

from __future__ import annotations

import streamlit as st

from app.workspace_apps.personal_gl.constants import DATASET_JOURNAL_SEARCH
from app.workspace_apps.personal_gl.runtime import AppRuntime
from app.workspace_apps.personal_gl.services.search import load_journal_search_options, run_journal_entry_search


def render_search_tab(runtime: AppRuntime) -> None:
    st.subheader("Search")
    st.caption("Search for journal entries using any combination of optional filters.")

    search_type = st.selectbox(
        "Search Type",
        options=["Journal Entries"],
        index=0,
        key="search_tab_type",
        disabled=True,
        help="Additional search types will be added later.",
    )
    if search_type != "Journal Entries":
        return

    options = load_journal_search_options()

    col1, col2, col3 = st.columns(3)
    with col1:
        eff_date_from = st.date_input("Effective Date From", value=None, key="search_je_eff_date_from")
        post_date_from = st.date_input("Post Date From", value=None, key="search_je_post_date_from")
        amount = st.number_input(
            "Amount",
            min_value=0.0,
            step=0.01,
            value=None,
            key="search_je_amount",
            placeholder="Optional exact amount",
        )
        je_source = st.selectbox("JE Source", options=options.source_options, index=0, key="search_je_source")
    with col2:
        eff_date_to = st.date_input("Effective Date To", value=None, key="search_je_eff_date_to")
        post_date_to = st.date_input("Post Date To", value=None, key="search_je_post_date_to")
        je_id = st.number_input("JE ID", min_value=1, step=1, value=None, key="search_je_id", placeholder="Optional")
        je_line_id = st.number_input("JE Line ID", min_value=1, step=1, value=None, key="search_je_line_id", placeholder="Optional")
    with col3:
        description = st.text_input("Description", key="search_je_description", placeholder="Optional")
        memo = st.text_input("Memo", key="search_je_memo", placeholder="Optional")
        selected_accounts = st.multiselect("Account(s)", options=options.account_options, key="search_je_accounts", placeholder="Optional")
        dc_filter = st.selectbox("Debit / Credit", options=["Either", "Debit", "Credit"], index=0, key="search_je_dc_filter")

    btn_col1, btn_col2, _ = st.columns([1, 1, 6])
    with btn_col1:
        search_clicked = st.button("Search", key="search_je_run")
    with btn_col2:
        clear_clicked = st.button("Clear", key="search_je_clear")

    if clear_clicked:
        for key, value in {
            "search_je_eff_date_from": None,
            "search_je_post_date_from": None,
            "search_je_amount": None,
            "search_je_source": "Any",
            "search_je_eff_date_to": None,
            "search_je_post_date_to": None,
            "search_je_id": None,
            "search_je_line_id": None,
            "search_je_description": "",
            "search_je_memo": "",
            "search_je_accounts": [],
            "search_je_dc_filter": "Either",
        }.items():
            st.session_state[key] = value
        runtime.state.append_ui_event("search_clear")
        st.rerun()

    if search_clicked:
        runtime.state.set_active_tab_hint("search")
        runtime.state.append_ui_event("search_run")
        results, df_results = run_journal_entry_search(
            effective_date_from=eff_date_from,
            effective_date_to=eff_date_to,
            post_date_from=post_date_from,
            post_date_to=post_date_to,
            amount=amount,
            source=je_source,
            journal_entry_id=je_id,
            journal_line_id=je_line_id,
            description=description,
            memo=memo,
            selected_accounts=selected_accounts,
            account_label_to_id=options.account_label_to_id,
            dc_filter=dc_filter,
        )
        st.markdown("#### Results")
        if df_results.empty:
            st.info("No journal entry lines matched your search criteria.")
            runtime.datasets.clear(DATASET_JOURNAL_SEARCH)
            return

        runtime.register_dataset(
            DATASET_JOURNAL_SEARCH,
            df_results,
            kind="dataframe",
            description="Journal entry search results",
            metadata={"rows": len(df_results)},
        )
        st.caption(f"{len(results):,} journal line(s) found.")
        st.dataframe(df_results, use_container_width=True, hide_index=True)
