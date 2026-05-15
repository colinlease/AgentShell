"""Account history tab renderer."""

from __future__ import annotations

import streamlit as st

from app.workspace_apps.GLA import get_connection, settle_suspense_now
from app.workspace_apps.personal_gl.constants import DATASET_ACCOUNT_HISTORY
from app.workspace_apps.personal_gl.runtime import AppRuntime
from app.workspace_apps.personal_gl.services.history import (
    compute_history_filters,
    load_account_history_dataset,
    load_account_notes_block,
    load_active_accounts,
    load_journal_entry_detail_dataframe,
    load_suspense_context,
)
from app.workspace_apps.personal_gl.ui.formatting import format_money


def render_account_history_tab(runtime: AppRuntime) -> None:
    st.subheader("Account History")
    accounts = load_active_accounts()
    if not accounts:
        st.warning("No accounts available. Define accounts first in Chart of Accounts.")
        return

    account_options = {f"{account.gl_number} – {account.name}": account.id for account in accounts}
    account_labels = list(account_options.keys())
    placeholder = "Select an account"

    col_top = st.columns([3, 2, 2, 2])
    with col_top[0]:
        selected_account_label = st.selectbox("Account", options=[placeholder] + account_labels, index=0)
    if selected_account_label == placeholder:
        st.info("Please select an account to view history.")
        return

    selected_account = next(account for account in accounts if account.id == account_options[selected_account_label])
    is_suspense_selected, _ = load_suspense_context(selected_account.gl_number)

    with col_top[1]:
        start_date = st.date_input("Start date (optional)", value=None)
    with col_top[2]:
        end_date = st.date_input("End date (optional)", value=None)
    with col_top[3]:
        hide_reversed = st.checkbox("Hide Reversed TXNs", value=True, key="acct_hist_hide_reversed")
        current_month_only = st.checkbox("Current Month", value=False, key="acct_hist_current_month")
        previous_month_only = st.checkbox("Previous Month", value=False, key="acct_hist_previous_month")
        suspense_new_only = st.checkbox(
            "Only show new transactions",
            value=False,
            key="acct_hist_suspense_new_only",
            help="When enabled, hides older suspense activity that existed before the last settlement cutoff. Click the Settle Suspense button at the bottom of the page to update the settlement cuttoff.",
        ) if is_suspense_selected else False

    st.session_state["account_history_account_label"] = selected_account_label
    st.session_state["account_history_start"] = start_date.isoformat() if start_date else None
    st.session_state["account_history_end"] = end_date.isoformat() if end_date else None

    notes_block = load_account_notes_block(selected_account.gl_number)
    if notes_block:
        st.markdown("**Notes for this account**")
        st.code(notes_block, language="text")

    start_str, end_str = compute_history_filters(
        start_date=start_date,
        end_date=end_date,
        current_month_only=current_month_only,
        previous_month_only=previous_month_only,
    )
    opening_balance_dollars, ending_balance_dollars, total_debit, total_credit, df_hist, history_rows = load_account_history_dataset(
        account_id=selected_account.id,
        selected_account_normal_balance=selected_account.normal_balance.value,
        start_str=start_str,
        end_str=end_str,
        suspense_new_only=suspense_new_only,
        is_suspense_selected=is_suspense_selected,
        hide_reversed=hide_reversed,
    )
    runtime.register_dataset(
        DATASET_ACCOUNT_HISTORY,
        df_hist,
        kind="dataframe",
        description="Account history rows for the selected account",
        metadata={"account": selected_account_label, "rows": len(df_hist)},
    )

    st.markdown(f"**Opening balance:** {format_money(opening_balance_dollars)}")
    st.markdown(f"**Running balance (ending):** {format_money(ending_balance_dollars)}")

    if df_hist.empty:
        st.info("No transactions found for the selected criteria.")
    else:
        tot_col1, tot_col2 = st.columns(2)
        with tot_col1:
            st.markdown(f"**Total Debits:** {format_money(total_debit)}")
        with tot_col2:
            st.markdown(f"**Total Credits:** {format_money(total_credit)}")
        st.dataframe(df_hist, use_container_width=True)

        je_options = []
        seen_jes = set()
        for row in history_rows:
            if row.journal_entry_id in seen_jes:
                continue
            seen_jes.add(row.journal_entry_id)
            je_options.append(f"{row.journal_entry_id} - {row.effective_date} - {row.description or ''}")

        if je_options:
            selected_je_label = st.selectbox(
                "View JE",
                options=[""] + je_options,
                index=0,
                key="account_history_view_je",
                help="Select a posted journal entry shown in the table above.",
            )
            if selected_je_label:
                try:
                    selected_je_id = int(selected_je_label.split(" - ", 1)[0].strip())
                except Exception:
                    selected_je_id = None
                if selected_je_id is not None:
                    df_je_view = load_journal_entry_detail_dataframe(selected_je_id)
                    if df_je_view.empty:
                        st.info("No lines found for the selected journal entry.")
                    else:
                        def bold_totals(row):
                            return ["font-weight: bold" if row.name == len(df_je_view) - 1 else "" for _ in row]
                        st.table(df_je_view.style.apply(bold_totals, axis=1))

    if is_suspense_selected and st.button("Settle suspense", key="acct_hist_settle_suspense_btn"):
        conn = get_connection()
        try:
            cutoff_id = settle_suspense_now(conn)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            runtime.state.set_active_tab_hint("account_history")
            runtime.state.append_ui_event("suspense_settle")
            st.success(f"Suspense settled through JE ID {cutoff_id}.")
            st.rerun()
        finally:
            conn.close()
