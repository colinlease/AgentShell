"""Search-related service helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.workspace_apps.GLA import (
    get_accounts,
    get_connection,
    get_distinct_journal_entry_sources,
    search_journal_entry_lines,
)

from app.workspace_apps.personal_gl.ui.formatting import format_money


@dataclass
class JournalSearchOptions:
    account_label_to_id: dict[str, int]
    account_options: list[str]
    source_options: list[str]


def load_journal_search_options() -> JournalSearchOptions:
    conn = get_connection()
    try:
        accounts = get_accounts(conn, include_inactive=False)
        je_sources = get_distinct_journal_entry_sources(conn)
    finally:
        conn.close()

    account_label_to_id = {
        f"{account.gl_number} – {account.name}": int(account.id)
        for account in accounts
    }
    return JournalSearchOptions(
        account_label_to_id=account_label_to_id,
        account_options=list(account_label_to_id.keys()),
        source_options=["Any"] + list(je_sources),
    )


def run_journal_entry_search(
    *,
    effective_date_from=None,
    effective_date_to=None,
    post_date_from=None,
    post_date_to=None,
    amount: float | None = None,
    source: str | None = None,
    journal_entry_id: int | None = None,
    journal_line_id: int | None = None,
    description: str | None = None,
    memo: str | None = None,
    selected_accounts: list[str] | None = None,
    account_label_to_id: dict[str, int] | None = None,
    dc_filter: str = "Either",
) -> tuple[list[Any], pd.DataFrame]:
    selected_account_ids = [
        account_label_to_id[label]
        for label in (selected_accounts or [])
        if account_label_to_id and label in account_label_to_id
    ]

    amount_cents = None if amount is None else int(round(float(amount) * 100))
    is_debit_filter = None
    if dc_filter == "Debit":
        is_debit_filter = True
    elif dc_filter == "Credit":
        is_debit_filter = False

    conn = get_connection()
    try:
        results = search_journal_entry_lines(
            conn,
            effective_date_from=effective_date_from.isoformat() if effective_date_from else None,
            effective_date_to=effective_date_to.isoformat() if effective_date_to else None,
            post_date_from=post_date_from.isoformat() if post_date_from else None,
            post_date_to=post_date_to.isoformat() if post_date_to else None,
            amount_cents=amount_cents,
            source=None if source == "Any" else source,
            journal_entry_id=journal_entry_id,
            journal_line_id=journal_line_id,
            description_contains=(description or "").strip() or None,
            memo_contains=(memo or "").strip() or None,
            account_ids=selected_account_ids,
            is_debit=is_debit_filter,
        )
    finally:
        conn.close()

    rows = []
    prev_je_id = None
    for result in results:
        same_as_prior_je = prev_je_id == result.journal_entry_id
        rows.append(
            {
                "JE ID": result.journal_entry_id if not same_as_prior_je else "",
                "Line ID": result.journal_line_id,
                "Effective Date": result.effective_date if not same_as_prior_je else "",
                "Post Date": result.post_date if not same_as_prior_je else "",
                "Source": (result.source or "") if not same_as_prior_je else "",
                "Description": (result.description or "") if not same_as_prior_je else "",
                "GL #": result.gl_number,
                "Account": result.account_name,
                "Memo": result.memo or "",
                "Debit": format_money(result.amount_cents / 100.0) if result.is_debit else "",
                "Credit": format_money(result.amount_cents / 100.0) if not result.is_debit else "",
            }
        )
        prev_je_id = result.journal_entry_id

    return results, pd.DataFrame(rows)
