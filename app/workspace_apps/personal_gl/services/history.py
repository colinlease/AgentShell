"""Account history service helpers."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.workspace_apps.GLA import (
    get_account_history,
    get_accounts,
    get_active_notes_for_gl,
    get_connection,
    get_suspense_gl_number,
    get_suspense_last_settled_je_id,
)

from app.workspace_apps.personal_gl.ui.formatting import format_money


def load_active_accounts():
    conn = get_connection()
    try:
        return get_accounts(conn, include_inactive=False)
    finally:
        conn.close()


def compute_history_filters(
    *,
    start_date,
    end_date,
    current_month_only: bool,
    previous_month_only: bool,
) -> tuple[str | None, str | None]:
    start_str = start_date.isoformat() if start_date is not None else None
    end_str = end_date.isoformat() if end_date is not None else None

    today = date.today()
    first_of_this_month = today.replace(day=1)
    first_of_prev_month = (first_of_this_month - timedelta(days=1)).replace(day=1)
    last_of_prev_month = first_of_this_month - timedelta(days=1)

    if current_month_only and previous_month_only:
        return first_of_prev_month.isoformat(), today.isoformat()
    if current_month_only:
        return first_of_this_month.isoformat(), today.isoformat()
    if previous_month_only:
        return first_of_prev_month.isoformat(), last_of_prev_month.isoformat()
    return start_str, end_str


def load_suspense_context(selected_gl_number: int | None) -> tuple[bool, int | None]:
    conn = get_connection()
    try:
        suspense_gl_number = get_suspense_gl_number(conn)
    except Exception:
        suspense_gl_number = None
    finally:
        conn.close()

    is_suspense_selected = (
        suspense_gl_number is not None
        and selected_gl_number is not None
        and int(suspense_gl_number) == int(selected_gl_number)
    )
    return is_suspense_selected, suspense_gl_number


def load_account_notes_block(selected_gl_number: int | None) -> str:
    if selected_gl_number is None:
        return ""
    conn = get_connection()
    try:
        account_notes = get_active_notes_for_gl(conn, selected_gl_number)
    except Exception:
        account_notes = []
    finally:
        conn.close()

    note_lines = []
    for note in account_notes:
        created_str = str(note.created_at) if note.created_at is not None else ""
        if created_str and len(created_str) > 10:
            created_str = created_str[:10]
        header = f"[{created_str}]" if created_str else "[Note]"
        body = (note.text or "").strip() or "(no details provided)"
        note_lines.append(f"{header}\n{body}")
    return "\n\n".join(note_lines)


def load_account_history_dataset(
    *,
    account_id: int,
    selected_account_normal_balance: str,
    start_str: str | None,
    end_str: str | None,
    suspense_new_only: bool,
    is_suspense_selected: bool,
    hide_reversed: bool,
) -> tuple[float, float, float, float, pd.DataFrame, list]:
    min_je_id = None
    if is_suspense_selected and suspense_new_only:
        conn = get_connection()
        try:
            cutoff = get_suspense_last_settled_je_id(conn)
        except Exception:
            cutoff = None
        finally:
            conn.close()
        min_je_id = int(cutoff) if cutoff is not None else 0

    conn = get_connection()
    try:
        opening_balance_cents, history_rows = get_account_history(
            conn,
            account_id=account_id,
            start_date=start_str,
            end_date=end_str,
            min_journal_entry_id=min_je_id,
        )
    finally:
        conn.close()

    if hide_reversed and history_rows:
        je_ids_in_history = {row.journal_entry_id for row in history_rows}
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT id, description
                FROM journal_entries
                WHERE description LIKE 'Reversal of JE %'
                """
            ).fetchall()
        finally:
            conn.close()

        reversed_je_ids = set()
        for row in rows:
            description = row["description"] or ""
            if not description.startswith("Reversal of JE "):
                continue
            remainder = description[len("Reversal of JE ") :]
            original_id_str = remainder.split(":", 1)[0].strip()
            try:
                original_id = int(original_id_str)
            except ValueError:
                continue
            reversal_id = int(row["id"])
            if original_id in je_ids_in_history or reversal_id in je_ids_in_history:
                reversed_je_ids.add(original_id)
                reversed_je_ids.add(reversal_id)

        if reversed_je_ids:
            history_rows = [
                row for row in history_rows if row.journal_entry_id not in reversed_je_ids
            ]

    display_sign = -1.0 if selected_account_normal_balance == "CREDIT" else 1.0
    opening_balance_dollars = (opening_balance_cents / 100.0) * display_sign

    if history_rows:
        ending_balance_cents = opening_balance_cents + sum(
            (row.debit_cents - row.credit_cents) for row in history_rows
        )
        ending_balance_dollars = (ending_balance_cents / 100.0) * display_sign
    else:
        ending_balance_dollars = opening_balance_dollars

    data = []
    total_debit = 0.0
    total_credit = 0.0
    running_balance_cents = opening_balance_cents
    for row in history_rows:
        debit_dollars = row.debit_cents / 100.0
        credit_dollars = row.credit_cents / 100.0
        running_balance_cents += row.debit_cents - row.credit_cents

        total_debit += debit_dollars
        total_credit += credit_dollars

        data.append(
            {
                "Effective Date": row.effective_date,
                "Post Date": row.post_date,
                "JE ID": row.journal_entry_id,
                "Description": row.description,
                "Source": row.source,
                "Memo": row.memo,
                "Debit": format_money(debit_dollars),
                "Credit": format_money(credit_dollars),
                "Running Balance": format_money((running_balance_cents / 100.0) * display_sign),
            }
        )

    return (
        opening_balance_dollars,
        ending_balance_dollars,
        total_debit,
        total_credit,
        pd.DataFrame(data),
        history_rows,
    )


def load_journal_entry_detail_dataframe(journal_entry_id: int) -> pd.DataFrame:
    conn = get_connection()
    try:
        je_rows = conn.execute(
            """
            SELECT
                je.id AS je_id,
                je.effective_date,
                je.post_date,
                jl.is_debit,
                jl.amount_cents,
                a.gl_number,
                a.name AS account_name
            FROM journal_entries je
            JOIN journal_lines jl
              ON jl.journal_entry_id = je.id
            JOIN accounts a
              ON a.id = jl.account_id
            WHERE je.id = ?
            ORDER BY jl.id
            """,
            (journal_entry_id,),
        ).fetchall()
    finally:
        conn.close()

    if not je_rows:
        return pd.DataFrame()

    rows = []
    total_debit = 0.0
    total_credit = 0.0
    eff_date_display = je_rows[0]["effective_date"]
    post_date_display = je_rows[0]["post_date"]
    for row in je_rows:
        amount_dollars = (row["amount_cents"] or 0) / 100.0
        if row["is_debit"]:
            debit = amount_dollars
            credit = 0.0
            total_debit += amount_dollars
        else:
            debit = 0.0
            credit = amount_dollars
            total_credit += amount_dollars
        rows.append(
            {
                "JE ID": row["je_id"],
                "Effective Date": eff_date_display,
                "Post Date": post_date_display,
                "GL Number": row["gl_number"],
                "Account": row["account_name"],
                "Debit": format_money(debit),
                "Credit": format_money(credit),
            }
        )

    rows.append(
        {
            "JE ID": "",
            "Effective Date": "",
            "Post Date": "",
            "GL Number": "",
            "Account": "Totals",
            "Debit": format_money(total_debit),
            "Credit": format_money(total_credit),
        }
    )
    return pd.DataFrame(rows)
