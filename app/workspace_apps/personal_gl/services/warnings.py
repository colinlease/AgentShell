"""Warnings and checklist service helpers."""

from __future__ import annotations

import pandas as pd

from app.workspace_apps.GLA import (
    get_account_balance_as_of,
    get_accounts,
    get_checklist_prefix_metadata,
    get_connection,
    get_last_closed_period_end,
    get_month_end_checklist_account_status_map,
    get_month_end_checklist_display_rows,
    get_month_end_checklist_statuses_for_date,
    get_month_end_checklists,
    get_month_end_closing_checklist_enabled,
    get_suspense_gl_number,
    get_transfers_clearing_gl_number,
    get_unbalanced_journal_entries,
)

from app.workspace_apps.personal_gl.ui.formatting import format_money


def load_warning_snapshot(as_of_date: str) -> dict:
    conn = get_connection()
    try:
        accounts = get_accounts(conn, include_inactive=False)
        closed_through = get_last_closed_period_end(conn)
        balances_cents = {}
        for account in accounts:
            bal_dc = get_account_balance_as_of(conn, account.id, as_of_date)
            balances_cents[account.gl_number] = bal_dc if account.type.value in ("ASSET", "EXPENSE") else -bal_dc

        suspense_gl = get_suspense_gl_number(conn)
        transfers_gl = get_transfers_clearing_gl_number(conn)
        imbalances = get_unbalanced_journal_entries(conn)
        month_end_checklists_enabled = get_month_end_closing_checklist_enabled(conn)
        active_checklists = get_month_end_checklists(conn, include_inactive=False)
    finally:
        conn.close()

    total_assets_c = sum(balances_cents.get(account.gl_number, 0) for account in accounts if account.type.value == "ASSET")
    total_liabilities_c = sum(balances_cents.get(account.gl_number, 0) for account in accounts if account.type.value == "LIABILITY")
    total_equity_c = sum(balances_cents.get(account.gl_number, 0) for account in accounts if account.type.value == "EQUITY")
    total_income_c = sum(balances_cents.get(account.gl_number, 0) for account in accounts if account.type.value == "INCOME")
    total_expense_c = sum(balances_cents.get(account.gl_number, 0) for account in accounts if account.type.value == "EXPENSE")

    return {
        "closed_through": closed_through,
        "accounts": accounts,
        "balances_cents": balances_cents,
        "suspense_gl": suspense_gl,
        "transfers_gl": transfers_gl,
        "imbalances": imbalances,
        "month_end_checklists_enabled": month_end_checklists_enabled,
        "active_checklists": active_checklists,
        "eqn_diff_c": total_assets_c - total_liabilities_c - total_equity_c,
        "pnl_c": total_income_c - total_expense_c,
    }


def load_checklist_status_dataframe(as_of_date: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        checklist_status_rows = get_month_end_checklist_statuses_for_date(conn, as_of_date=as_of_date)
    finally:
        conn.close()

    checklist_status_rows = sorted(
        checklist_status_rows,
        key=lambda row: (
            get_checklist_prefix_metadata(row.checklist_name)[0],
            (row.checklist_name or "").strip().upper(),
        ),
    )

    rows = []
    for row in checklist_status_rows:
        progress_pct = round((row.completed_account_count / row.account_count) * 100) if row.account_count else 0
        if row.completed_account_count == 0:
            completed_status = "Not Started"
        elif row.completed_account_count >= row.account_count:
            completed_status = "Complete"
        else:
            completed_status = "In Progress"
        rows.append(
            {
                "Checklist": row.checklist_name,
                "Progress": progress_pct,
                "Completed": completed_status,
                "Completed At": row.completed_at or "",
            }
        )
    return pd.DataFrame(rows)


def load_checklist_detail_dataframe(checklist_id: int, as_of_date: str) -> tuple[pd.DataFrame, dict[int, bool]]:
    conn = get_connection()
    try:
        checklist_rows = get_month_end_checklist_display_rows(
            conn,
            checklist_id=checklist_id,
            as_of_date=as_of_date,
        )
        checklist_account_status_map = get_month_end_checklist_account_status_map(
            conn,
            checklist_id=checklist_id,
            as_of_date=as_of_date,
        )
    finally:
        conn.close()

    rows = []
    for row in checklist_rows:
        rows.append(
            {
                "Account": f"{row.gl_number} – {row.account_name}",
                "Most Recent Effective Date": row.most_recent_effective_date or "",
                "Post Date": row.most_recent_post_date or "",
                "Current Balance": format_money(row.balance_cents / 100.0),
                "Complete": bool(checklist_account_status_map.get(row.account_id, False)),
                "_account_id": row.account_id,
            }
        )
    return pd.DataFrame(rows), checklist_account_status_map
