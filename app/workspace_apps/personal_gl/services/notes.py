"""Notes service helpers."""

from __future__ import annotations

import pandas as pd

from app.workspace_apps.GLA import archive_notes, create_note, get_active_notes, get_accounts, get_connection


def load_note_accounts() -> tuple[list, dict[int, str]]:
    conn = get_connection()
    try:
        accounts = get_accounts(conn, include_inactive=False)
    finally:
        conn.close()

    gl_label_by_number = {
        account.gl_number: f"{account.gl_number} – {account.name}"
        for account in accounts
    }
    return accounts, gl_label_by_number


def create_note_record(text: str, gl_numbers: list[int]) -> None:
    conn = get_connection()
    try:
        create_note(conn, text, gl_numbers)
    finally:
        conn.close()


def load_active_notes_df(gl_label_by_number: dict[int, str]) -> pd.DataFrame:
    conn = get_connection()
    try:
        active_notes = get_active_notes(conn)
    finally:
        conn.close()

    rows = []
    for note in active_notes:
        linked_gl_labels = ", ".join(
            gl_label_by_number.get(gl, str(gl)) for gl in note.gl_numbers
        ) if note.gl_numbers else ""
        rows.append(
            {
                "id": note.id,
                "Created": note.created_at,
                "GL(s)": linked_gl_labels,
                "Note": note.text,
                "Archive": False,
            }
        )
    return pd.DataFrame(rows)


def archive_note_ids(note_ids: list[int]) -> None:
    conn = get_connection()
    try:
        archive_notes(conn, note_ids)
    finally:
        conn.close()
