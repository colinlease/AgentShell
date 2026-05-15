"""Warnings tab renderer."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app.workspace_apps.GLA import get_checklist_prefix_metadata, get_connection, set_month_end_checklist_account_completed
from app.workspace_apps.personal_gl.constants import DATASET_WARNINGS_CHECKLIST_DETAIL, DATASET_WARNINGS_CHECKLIST_STATUS
from app.workspace_apps.personal_gl.runtime import AppRuntime
from app.workspace_apps.personal_gl.services.warnings import load_checklist_detail_dataframe, load_checklist_status_dataframe, load_warning_snapshot
from app.workspace_apps.personal_gl.ui.formatting import format_money


def render_warnings_tab(runtime: AppRuntime) -> None:
    st.subheader("Warnings")
    warn_as_of = st.date_input("As of date", value=date.today(), key="warnings_as_of")
    warn_as_of_str = warn_as_of.isoformat()
    snapshot = load_warning_snapshot(warn_as_of_str)

    top_col1, top_col2, top_col3 = st.columns(3)
    with top_col1:
        if snapshot["closed_through"]:
            st.markdown(f"**Closed Through:** `{snapshot['closed_through']}`")
        else:
            st.markdown("**Closed Through:** `No closed period`")
    with top_col2:
        st.markdown(f"**System Post Date:** `{date.today().isoformat()}`")
    with top_col3:
        st.markdown(f"**Database Path:** `{runtime.db_path.name}`")

    accounts = snapshot["accounts"]
    balances_cents = snapshot["balances_cents"]
    if not accounts:
        st.info("No accounts available to evaluate the accounting equation.")
        return

    eqn_diff_c = snapshot["eqn_diff_c"]
    pnl_c = snapshot["pnl_c"]
    tolerance = 1
    if abs(eqn_diff_c) <= tolerance:
        st.success(f"Accounting equation is in balance as of {warn_as_of_str}.")
    elif abs(eqn_diff_c - pnl_c) <= tolerance:
        st.warning(
            "Assets do not match liabilities + equity by "
            f"{format_money(eqn_diff_c / 100.0)}. "
            "This difference is equal to current net income,  which will be closed to retained earnings at period end."
        )
    else:
        st.error(
            "Accounting equation appears to be out of balance by "
            f"{format_money(eqn_diff_c / 100.0)}, and this difference is not explained by current net income. Please investigate."
        )

    suspense_gl = snapshot["suspense_gl"]
    if suspense_gl is None:
        st.warning("Suspense account is not configured. Configure it in Preferences → Operational GL Accounts.")
    elif suspense_gl not in balances_cents:
        st.warning(f"Suspense account is configured as GL {suspense_gl}, but that GL is not an active account.")
    else:
        suspense_balance_c = balances_cents.get(suspense_gl, 0)
        if suspense_balance_c != 0:
            st.warning(f"Suspense account (GL {suspense_gl}) has non-zero balance: {format_money(suspense_balance_c / 100.0)}.")
        else:
            st.success(f"Suspense account (GL {suspense_gl}) has a balance of $0.")

    transfers_gl = snapshot["transfers_gl"]
    if transfers_gl is None:
        st.warning("Transfers Clearing account is not configured. Configure it in Preferences → Operational GL Accounts.")
    elif transfers_gl not in balances_cents:
        st.warning(f"Transfers Clearing account is configured as GL {transfers_gl}, but that GL is not an active account.")
    else:
        transfers_clearing_c = balances_cents.get(transfers_gl, 0)
        if transfers_clearing_c != 0:
            st.warning(f"Transfers Clearing (GL {transfers_gl}) has non-zero balance: {format_money(transfers_clearing_c / 100.0)}.")
        else:
            st.success(f"Transfers Clearing (GL {transfers_gl}) has a balance of $0.")

    negative_expense_accounts = []
    negative_income_accounts = []
    for account in accounts:
        balance_cents = balances_cents.get(account.gl_number, 0)
        if account.type.value == "EXPENSE" and balance_cents < 0:
            negative_expense_accounts.append(f"{account.gl_number} – {account.name}: {format_money(balance_cents / 100.0)}")
        elif account.type.value == "INCOME" and balance_cents < 0:
            negative_income_accounts.append(f"{account.gl_number} – {account.name}: {format_money(balance_cents / 100.0)}")

    if negative_expense_accounts:
        st.warning("Expense account(s) have negative balances (credit balances):\n" + "\n".join(f"- {account}" for account in negative_expense_accounts))
    if negative_income_accounts:
        st.warning("Income account(s) have negative balances (debit balances):\n" + "\n".join(f"- {account}" for account in negative_income_accounts))

    if not snapshot["imbalances"]:
        st.success("All journal entries are balanced.")
    else:
        st.error("Some journal entries are unbalanced. See details below.")
        for je_id, debits_c, credits_c in snapshot["imbalances"]:
            st.write(f"JE #{je_id}: Debits = ${debits_c / 100:.2f}, Credits = ${credits_c / 100:.2f}")

    if not snapshot["month_end_checklists_enabled"]:
        return

    st.markdown("---")
    st.subheader("Month End Closing Checklists")

    status_df = load_checklist_status_dataframe(warn_as_of_str)
    runtime.register_dataset(
        DATASET_WARNINGS_CHECKLIST_STATUS,
        status_df,
        kind="dataframe",
        description="Month-end checklist statuses for the selected as-of date",
        metadata={"rows": len(status_df)},
    )
    if status_df.empty:
        st.info("No active month-end closing checklists found.")
    else:
        st.dataframe(
            status_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Checklist": st.column_config.TextColumn("Checklist"),
                "Progress": st.column_config.ProgressColumn("Progress", help="Percent of checklist accounts marked complete.", min_value=0, max_value=100, format="%d%%"),
                "Completed": st.column_config.TextColumn("Completed"),
                "Completed At": st.column_config.TextColumn("Completed At"),
            },
            key=f"warnings_month_end_checklist_statuses_{warn_as_of_str}",
        )

    active_checklists = snapshot["active_checklists"]
    if not active_checklists:
        st.info("No month-end closing checklists have been created in Preferences.")
        return

    active_checklists_sorted = sorted(
        active_checklists,
        key=lambda checklist: (
            get_checklist_prefix_metadata(checklist.name)[0],
            (checklist.name or "").strip().upper(),
        ),
    )
    checklist_name_to_id = {checklist.name: checklist.id for checklist in active_checklists_sorted}
    selected_checklist_name = st.selectbox(
        "Select Checklist",
        options=[checklist.name for checklist in active_checklists_sorted],
        key="warnings_month_end_checklist_select",
    )
    selected_checklist_id = checklist_name_to_id[selected_checklist_name]
    _, checklist_section_title = get_checklist_prefix_metadata(selected_checklist_name)
    if checklist_section_title:
        st.markdown(f"##### {checklist_section_title}")

    checklist_editor_df, checklist_account_status_map = load_checklist_detail_dataframe(selected_checklist_id, warn_as_of_str)
    runtime.register_dataset(
        DATASET_WARNINGS_CHECKLIST_DETAIL,
        checklist_editor_df,
        kind="dataframe",
        description="Month-end checklist detail rows for the selected checklist",
        metadata={"checklist": selected_checklist_name, "rows": len(checklist_editor_df)},
    )

    if checklist_editor_df.empty:
        st.info("This checklist does not currently contain any active accounts.")
        return

    edited_checklist_df = st.data_editor(
        checklist_editor_df,
        use_container_width=True,
        hide_index=True,
        disabled=["Account", "Most Recent Effective Date", "Post Date", "Current Balance", "_account_id"],
        column_config={
            "Account": st.column_config.TextColumn("Account"),
            "Most Recent Effective Date": st.column_config.TextColumn("Most Recent Effective Date"),
            "Post Date": st.column_config.TextColumn("Post Date"),
            "Current Balance": st.column_config.TextColumn("Current Balance"),
            "Complete": st.column_config.CheckboxColumn("Complete"),
            "_account_id": None,
        },
        key=f"warnings_month_end_checklist_editor_{selected_checklist_id}_{warn_as_of_str}",
    )
    completed_accounts_count = int(edited_checklist_df["Complete"].fillna(False).sum())
    total_accounts_count = len(edited_checklist_df)
    st.caption(f"{completed_accounts_count} / {total_accounts_count} accounts complete")

    if st.button("Save Account Statuses", key=f"warnings_month_end_save_account_statuses_{selected_checklist_id}_{warn_as_of_str}"):
        try:
            conn = get_connection()
            try:
                for _, edited_row in edited_checklist_df.iterrows():
                    account_id = int(edited_row["_account_id"])
                    new_completed = bool(edited_row["Complete"])
                    old_completed = bool(checklist_account_status_map.get(account_id, False))
                    if new_completed != old_completed:
                        set_month_end_checklist_account_completed(
                            conn,
                            checklist_id=selected_checklist_id,
                            account_id=account_id,
                            as_of_date=warn_as_of_str,
                            is_completed=new_completed,
                        )
            finally:
                conn.close()
            runtime.state.set_active_tab_hint("warnings")
            runtime.state.append_ui_event("checklist_status_save")
            st.success("Account statuses saved.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
