"""Chart of accounts tab renderer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.workspace_apps.GLA import create_account, delete_account, get_accounts, get_connection, update_account_active
from app.workspace_apps.personal_gl.runtime import AppRuntime


def render_chart_of_accounts_tab(runtime: AppRuntime) -> None:
    st.subheader("Chart of Accounts")

    with st.expander("Add new account"):
        col_new1, col_new2 = st.columns([1, 3])
        with col_new1:
            new_gl_number = st.number_input("GL Number", min_value=1000, max_value=99999, step=1, value=10000, key="new_gl_number")
        with col_new2:
            new_name = st.text_input("Name", key="new_name")
        col_new3, col_new4 = st.columns(2)
        with col_new3:
            new_type = st.selectbox("Type", options=["ASSET", "LIABILITY", "EQUITY", "INCOME", "EXPENSE"], key="new_type")
        with col_new4:
            new_normal_balance = st.selectbox("Normal balance", options=["DEBIT", "CREDIT"], key="new_normal_balance")

        if st.button("Create account", key="create_account_btn"):
            if not new_name.strip():
                st.error("Account name is required.")
            else:
                conn = get_connection()
                try:
                    create_account(
                        conn=conn,
                        gl_number=int(new_gl_number),
                        name=new_name,
                        type_code=new_type,
                        normal_balance_code=new_normal_balance,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    runtime.state.set_active_tab_hint("chart_of_accounts")
                    runtime.state.append_ui_event("account_create")
                    st.success("Account created successfully.")
                finally:
                    conn.close()

    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("Search (by name or GL number)", value="")
    with col2:
        include_inactive = st.checkbox("Include inactive accounts", value=False)

    conn = get_connection()
    try:
        accounts = get_accounts(conn, include_inactive=include_inactive, search=search or None)
    finally:
        conn.close()

    if accounts:
        df_accounts = pd.DataFrame(
            [
                {
                    "GL Number": account.gl_number,
                    "Name": account.name,
                    "Type": account.type.value,
                    "Normal Balance": account.normal_balance.value,
                    "Active": account.is_active,
                }
                for account in accounts
            ]
        ).sort_values("GL Number")
        st.dataframe(df_accounts, use_container_width=True)
    else:
        st.info("No accounts found. Check your filters or seed data.")

    if not accounts:
        return

    st.markdown("#### Modify existing account")
    label_to_account = {}
    labels = []
    for account in accounts:
        status = "Active" if account.is_active else "Archived"
        label = f"{account.gl_number} – {account.name} ({status})"
        labels.append(label)
        label_to_account[label] = account

    selected_account = label_to_account[
        st.selectbox("Select account to modify", options=labels, key="modify_account_select")
    ]
    st.write("Do not delete accounts with any historical transactions, even if historical transactions have been reversed. Consider Dependancies.")

    col_mod1, col_mod2 = st.columns(2)
    with col_mod1:
        if selected_account.is_active:
            if st.button("Archive account", key="archive_account_btn"):
                conn = get_connection()
                try:
                    update_account_active(conn, selected_account.id, False)
                finally:
                    conn.close()
                runtime.state.set_active_tab_hint("chart_of_accounts")
                runtime.state.append_ui_event("account_archive")
                st.success("Account archived. It will no longer appear as active.")
        else:
            if st.button("Unarchive account", key="unarchive_account_btn"):
                conn = get_connection()
                try:
                    update_account_active(conn, selected_account.id, True)
                finally:
                    conn.close()
                runtime.state.set_active_tab_hint("chart_of_accounts")
                runtime.state.append_ui_event("account_unarchive")
                st.success("Account unarchived and marked as active.")

    with col_mod2:
        if st.button("Delete account", key="delete_account_btn"):
            runtime.state.set_confirm_delete(True)

        if runtime.state.get_confirm_delete():
            st.warning("Are you sure you want to delete this account? This action cannot be undone.")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button("Confirm delete", key="confirm_delete_btn"):
                    conn = get_connection()
                    try:
                        delete_account(conn, selected_account.id)
                    finally:
                        conn.close()
                    runtime.state.set_confirm_delete(False)
                    runtime.state.set_active_tab_hint("chart_of_accounts")
                    runtime.state.append_ui_event("account_delete")
                    st.success("Account and all its journal entry lines have been deleted.")
            with col_c2:
                if st.button("Cancel", key="cancel_delete_btn"):
                    runtime.state.set_confirm_delete(False)
                    st.info("Deletion cancelled.")
