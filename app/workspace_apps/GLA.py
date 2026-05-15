"""
GLA.py

Core data layer for the personal general ledger application.

First step:
- Define basic account types and normal balance enums.
- Provide a SQLite connection helper.
- Implement database initialization for the `accounts` table.
- Seed an initial chart of accounts (lean version you approved).

We will layer other tables (journal entries, imports, etc.) on top of this
in subsequent steps.
"""

from __future__ import annotations

import enum
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import csv
import io
from datetime import datetime, date, timedelta
import pandas as pd
import json
# --- Activity log table and helpers ---


def init_activity_log_table(conn: sqlite3.Connection) -> None:
    """Create the activity_log table if it does not already exist.

    This is a generic audit log capturing high-level activity in the app
    (journal entries, settings changes, account maintenance, notes, etc.).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS activity_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL DEFAULT (datetime('now')),
            event_type   TEXT NOT NULL,
            entity_type  TEXT,
            entity_id    INTEGER,
            summary      TEXT NOT NULL,
            details_json TEXT
        );
        """
    )


def log_activity(
    conn: sqlite3.Connection,
    event_type: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[int] = None,
    summary: Optional[str] = None,
    details: Optional[dict] = None,
) -> None:
    """Best-effort activity logger.

    - Never raises; failures are silently ignored so core flows are not broken.
    - details, if provided, is JSON-encoded into details_json.
    """
    if not event_type:
        return

    summary_text = (summary or "").strip() or event_type

    details_json: Optional[str] = None
    if details is not None:
        try:
            details_json = json.dumps(details, ensure_ascii=False)
        except Exception:
            details_json = None

    try:
        conn.execute(
            """
            INSERT INTO activity_log (event_type, entity_type, entity_id, summary, details_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_type, entity_type, entity_id, summary_text, details_json),
        )
    except Exception:
        # Logging must never interfere with primary operations.
        pass



import os


def _resolve_db_path() -> Path:
    """Resolve the SQLite DB path.

    Default behavior (recommended): store the DB inside the Personal GL app
    package folder so each folder copy uses its own isolated database.

    Overrides:
    - PERSONAL_GLA_DB_PATH: absolute or relative path to the DB file.
    """
    override = os.environ.get("PERSONAL_GLA_DB_PATH")
    if override and str(override).strip():
        p = Path(str(override).strip()).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p)

    # Keep the database inside the Personal GL app folder for hosted copies.
    return Path(__file__).resolve().parent / "personal_gl" / "personal_gl.db"


# Path to the SQLite database file (folder-local by default)
DB_PATH: Path = _resolve_db_path()


class AccountType(str, enum.Enum):
    ASSET = "ASSET"
    LIABILITY = "LIABILITY"
    EQUITY = "EQUITY"
    INCOME = "INCOME"
    EXPENSE = "EXPENSE"


class NormalBalance(str, enum.Enum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"


@dataclass
class Account:
    """
    In-memory representation of a GL account.
    """
    id: Optional[int]
    gl_number: int
    name: str
    type: AccountType
    normal_balance: NormalBalance
    is_active: bool = True


@dataclass
class JournalEntryLine:
    """
    In-memory representation of one line of a journal entry.
    """
    account_id: int
    amount_cents: int
    is_debit: bool
    memo: Optional[str] = None




@dataclass
class AccountHistoryRow:
    """
    One row in an account's history, including running balance.
    Amounts are stored in cents.
    """
    effective_date: str
    post_date: str
    journal_entry_id: int
    description: Optional[str]
    source: Optional[str]
    memo: Optional[str]
    debit_cents: int
    credit_cents: int
    running_balance_cents: int


@dataclass
class JournalEntrySearchRow:
    """
    One JE search result row.

    This is intentionally one row per journal line, with JE-level fields repeated
    so the UI can display naturally grouped results by sorting on JE/date/line order.
    Amounts are stored in cents.
    """
    journal_entry_id: int
    journal_line_id: int
    effective_date: str
    post_date: str
    description: Optional[str]
    source: Optional[str]
    account_id: int
    gl_number: int
    account_name: str
    amount_cents: int
    is_debit: bool
    sort_order: int
    memo: Optional[str]


# --- Notes dataclass ---

@dataclass
class Note:
    """In-memory representation of a note that can optionally be linked to one or more GL numbers."""
    id: int
    text: str
    created_at: str
    archived_at: Optional[str]
    is_archived: bool
    gl_numbers: List[int]


# --- Period closing dataclasses ---

@dataclass
class PeriodCloseLine:
    """
    One line in an automatically generated period-closing journal entry.
    """
    account: Account
    amount_cents: int
    is_debit: bool


@dataclass
class PeriodClosePreview:
    """
    Preview of a period-closing entry (before posting).

    - lines: all lines that would be posted, including the retained earnings line.
    - total_income_cents: total income for the period (credits - debits).
    - total_expense_cents: total expense for the period (debits - credits).
    - net_income_cents: total_income_cents - total_expense_cents.
    """
    lines: List[PeriodCloseLine]
    total_income_cents: int
    total_expense_cents: int
    net_income_cents: int



# --- Transaction Keyword Rule Dataclass ---

from typing import Any

@dataclass
class TxnKeywordRule:
    """Rule for mapping a keyword in an imported transaction description
    to a default offset GL account.

    - keyword: substring to search for (case-insensitive).
    - gl_number: the GL to suggest when the keyword is found.
    - is_active: whether the rule is currently applied.
    - priority: optional integer for tie-breaking if multiple rules match.
    """
    id: Optional[int]
    keyword: str
    gl_number: int
    is_active: bool = True
    priority: int = 0

@dataclass
class ManualTxnTemplate:
    """Template for streamlined manual transactions.

    - name must be unique.
    - base_gl_number is the GL that is always one side of the JE.
    - base_side indicates whether the base GL is debited or credited.
    """
    id: int
    name: str
    base_gl_number: int
    base_side: str  # "DEBIT" or "CREDIT"
    is_active: bool
    created_at: str
    updated_at: str

# --- App-wide settings helpers (for things like retained earnings GL) ---




@dataclass
class UserUploadMapping:
    """User-defined import mapping (Excel-only in the simplest version).

    - name must be unique.
    - base_gl_number is the GL that represents the bank/credit card account.
    - positive_is_debit indicates how to book POSITIVE amounts to the base account:
        True  => positive amounts DEBIT base GL
        False => positive amounts CREDIT base GL
      (negative amounts will book the opposite)
    - column names are the headers in the uploaded Excel file.
    """
    id: int
    name: str
    base_gl_number: int
    positive_is_debit: bool
    effective_date_col: str
    amount_col: str
    description_col: str
    is_active: bool
    created_at: str
    updated_at: str


# --- Month-End Closing Checklist dataclasses ---

@dataclass
class MonthEndChecklist:
    """Saved month-end closing checklist definition."""
    id: int
    name: str
    is_active: bool
    created_at: str
    updated_at: str


@dataclass
class MonthEndChecklistDisplayRow:
    """One account row displayed for a selected month-end closing checklist."""
    account_id: int
    gl_number: int
    account_name: str
    most_recent_effective_date: Optional[str]
    most_recent_post_date: Optional[str]
    balance_cents: int


@dataclass
class MonthEndChecklistStatusRow:
    """Completion status for one checklist as of a selected date."""
    checklist_id: int
    checklist_name: str
    account_count: int
    as_of_date: str
    is_completed: bool
    completed_at: Optional[str]
    completed_account_count: int = 0

@dataclass
class MonthEndChecklistAccountStatusRow:
    """Account-level completion status for one checklist row as of a selected date."""
    checklist_id: int
    account_id: int
    as_of_date: str
    is_completed: bool
    completed_at: Optional[str]


@dataclass
class GenericImportTxn:
    """Normalized representation of one row from a user-defined upload mapping."""
    effective_date: str
    description: str
    amount_cents: int          # absolute amount in cents
    is_debit_for_account: bool # for the base GL
    raw_row: dict


def init_month_end_checklists_table(conn: sqlite3.Connection) -> None:
    """Create the month_end_checklists table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS month_end_checklists (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL UNIQUE,
            is_active  INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )


def init_month_end_checklist_accounts_table(conn: sqlite3.Connection) -> None:
    """Create the month_end_checklist_accounts table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS month_end_checklist_accounts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            checklist_id INTEGER NOT NULL,
            account_id   INTEGER NOT NULL,
            sort_order   INTEGER NOT NULL,
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (checklist_id) REFERENCES month_end_checklists(id) ON DELETE CASCADE,
            FOREIGN KEY (account_id) REFERENCES accounts(id),
            UNIQUE (checklist_id, account_id)
        );
        """
    )


def init_month_end_checklist_status_table(conn: sqlite3.Connection) -> None:
    """Create the month_end_checklist_status table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS month_end_checklist_status (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            checklist_id INTEGER NOT NULL,
            as_of_date   TEXT NOT NULL,
            is_completed INTEGER NOT NULL DEFAULT 0,
            completed_at TEXT,
            updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (checklist_id) REFERENCES month_end_checklists(id) ON DELETE CASCADE,
            UNIQUE (checklist_id, as_of_date)
        );
        """
    )

def init_month_end_checklist_account_status_table(conn: sqlite3.Connection) -> None:
    """Create the month_end_checklist_account_status table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS month_end_checklist_account_status (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            checklist_id INTEGER NOT NULL,
            account_id   INTEGER NOT NULL,
            as_of_date   TEXT NOT NULL,
            is_completed INTEGER NOT NULL DEFAULT 0,
            completed_at TEXT,
            updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (checklist_id) REFERENCES month_end_checklists(id) ON DELETE CASCADE,
            FOREIGN KEY (account_id) REFERENCES accounts(id),
            UNIQUE (checklist_id, account_id, as_of_date)
        );
        """
    )

def get_month_end_closing_checklist_enabled(conn: sqlite3.Connection) -> bool:
    """Return True if the month-end closing checklist feature is enabled."""
    val = get_app_setting(conn, "month_end_closing_checklist_enabled")
    return str(val or "").strip() == "1"

def get_checklist_prefix_metadata(checklist_name: str) -> tuple[int, Optional[str]]:
    """Return (sort_bucket, section_title) for recognized checklist name prefixes.

    Ordering:
      0 = uncategorized
      1 = DLY
      2 = WKLY
      3 = MNTH
      4 = QTRLY
      5 = YRLY
      6 = AUDT
    """
    name = (checklist_name or "").strip().upper()

    if name.startswith("DLY"):
        return 1, "Daily Checklist"
    if name.startswith("WKLY"):
        return 2, "Weekly Checklist"
    if name.startswith("MNTH"):
        return 3, "Monthly Checklist"
    if name.startswith("QTRLY"):
        return 4, "Quarterly Checklist"
    if name.startswith("YRLY"):
        return 5, "Yearly Checklist"
    if name.startswith("AUDT"):
        return 6, "Audit Checklist"
    return 0, None


def is_quarter_end_date(dt: date) -> bool:
    """Return True if the date is a quarter-end month-end."""
    return dt.month in (3, 6, 9, 12)


def is_year_end_date(dt: date) -> bool:
    """Return True if the date is December 31."""
    return dt.month == 12 and dt.day == 31


def checklist_required_for_period_close(checklist_name: str, period_end: date) -> bool:
    """Return True if this checklist should block period close for the given month-end date.

    Prefix rules:
      - Uncategorized: always required
      - DLY: always required
      - WKLY: required only when period_end is a Friday
      - MNTH: always required
      - QTRLY: required only on quarter-end month-ends
      - YRLY: required only on December 31
      - AUDT: never required
    """
    sort_bucket, _ = get_checklist_prefix_metadata(checklist_name)

    # 0 = uncategorized
    if sort_bucket == 0:
        return True

    # 1 = DLY
    if sort_bucket == 1:
        return True

    # 2 = WKLY
    if sort_bucket == 2:
        return period_end.weekday() == 4  # Friday

    # 3 = MNTH
    if sort_bucket == 3:
        return True

    # 4 = QTRLY
    if sort_bucket == 4:
        return is_quarter_end_date(period_end)

    # 5 = YRLY
    if sort_bucket == 5:
        return is_year_end_date(period_end)

    # 6 = AUDT
    if sort_bucket == 6:
        return False

    return True


def get_checklist_period_close_category_label(checklist_name: str) -> str:
    """Human-readable checklist category label for warnings/messages."""
    sort_bucket, _ = get_checklist_prefix_metadata(checklist_name)

    if sort_bucket == 1:
        return "Daily"
    if sort_bucket == 2:
        return "Weekly"
    if sort_bucket == 3:
        return "Monthly"
    if sort_bucket == 4:
        return "Quarterly"
    if sort_bucket == 5:
        return "Yearly"
    if sort_bucket == 6:
        return "Audit"
    return "General"


def evaluate_period_close_checklist_gate(
    conn: sqlite3.Connection,
    period_end: date,
) -> dict:
    """Evaluate whether period close should be allowed for the given period-end date.

    Returns a dict with:
      - allowed: bool
      - required_checklists: list[dict]
      - incomplete_required_checklists: list[dict]

    Notes:
      - Only active checklists are considered
      - Archived checklists do not block close
      - Audit checklists never block close
      - Completion is evaluated for the period_end date itself
    """
    if not get_month_end_closing_checklist_enabled(conn):
        return {
            "allowed": True,
            "required_checklists": [],
            "incomplete_required_checklists": [],
        }

    active_checklists = get_month_end_checklists(conn, include_inactive=False)
    if not active_checklists:
        return {
            "allowed": True,
            "required_checklists": [],
            "incomplete_required_checklists": [],
        }

    period_end_str = period_end.isoformat()
    status_rows = get_month_end_checklist_statuses_for_date(conn, period_end_str)

    status_by_id: dict[int, bool] = {}
    status_by_name: dict[str, bool] = {}

    for row in status_rows or []:
        status_by_id[int(row.checklist_id)] = bool(row.is_completed)
        status_by_name[str(row.checklist_name)] = bool(row.is_completed)

    required_checklists: List[dict] = []
    incomplete_required_checklists: List[dict] = []

    for checklist in active_checklists:
        checklist_name = checklist.name or ""

        if not checklist_required_for_period_close(checklist_name, period_end):
            continue

        is_complete = status_by_id.get(checklist.id)
        if is_complete is None:
            is_complete = status_by_name.get(checklist_name, False)

        checklist_info = {
            "checklist_id": checklist.id,
            "checklist_name": checklist_name,
            "category": get_checklist_period_close_category_label(checklist_name),
            "is_complete": bool(is_complete),
        }

        required_checklists.append(checklist_info)

        if not checklist_info["is_complete"]:
            incomplete_required_checklists.append(checklist_info)

    return {
        "allowed": len(incomplete_required_checklists) == 0,
        "required_checklists": required_checklists,
        "incomplete_required_checklists": incomplete_required_checklists,
    }

def set_month_end_closing_checklist_enabled(conn: sqlite3.Connection, enabled: bool) -> None:
    """Enable/disable the month-end closing checklist feature."""
    set_app_setting(conn, "month_end_closing_checklist_enabled", "1" if enabled else "0")

def _month_end_checklist_from_row(row: sqlite3.Row) -> MonthEndChecklist:
    return MonthEndChecklist(
        id=int(row["id"]),
        name=str(row["name"]),
        is_active=bool(row["is_active"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
def create_month_end_checklist(
    conn: sqlite3.Connection,
    name: str,
    account_ids: List[int],
) -> int:
    """Create a new month-end closing checklist.

    - name must be unique and non-empty.
    - account_ids must contain at least one active account id.
    - checklist membership is stored low-to-high by GL number.
    """
    nm = (name or "").strip()
    if not nm:
        raise ValueError("Checklist name cannot be empty.")

    normalized_ids = [int(x) for x in (account_ids or []) if x is not None]
    normalized_ids = sorted(set(normalized_ids))
    if not normalized_ids:
        raise ValueError("Select at least one active account for the checklist.")

    placeholders = ",".join(["?"] * len(normalized_ids))
    valid_rows = conn.execute(
        f"""
        SELECT id, gl_number
        FROM accounts
        WHERE is_active = 1
          AND id IN ({placeholders})
        ORDER BY gl_number ASC, id ASC
        """,
        normalized_ids,
    ).fetchall()

    ordered_account_ids = [int(r["id"]) for r in valid_rows]
    if len(ordered_account_ids) != len(normalized_ids):
        raise ValueError("All checklist accounts must be active accounts.")

    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO month_end_checklists (name, is_active)
            VALUES (?, 1)
            """,
            (nm,),
        )
        checklist_id = int(cur.lastrowid)

        cur.executemany(
            """
            INSERT INTO month_end_checklist_accounts (checklist_id, account_id, sort_order)
            VALUES (?, ?, ?)
            """,
            [
                (checklist_id, account_id, idx)
                for idx, account_id in enumerate(ordered_account_ids, start=1)
            ],
        )

        log_activity(
            conn,
            event_type="MONTH_END_CHECKLIST_CREATED",
            entity_type="MONTH_END_CHECKLIST",
            entity_id=checklist_id,
            summary=f"Created month-end checklist '{nm}'",
            details={
                "checklist_id": checklist_id,
                "name": nm,
                "account_ids": ordered_account_ids,
            },
        )

        conn.commit()
        return checklist_id
    except sqlite3.IntegrityError:
        conn.rollback()
        raise ValueError(f"A month-end checklist named '{nm}' already exists.")
    except Exception:
        conn.rollback()
        raise



def get_month_end_checklists(
    conn: sqlite3.Connection,
    include_inactive: bool = False,
) -> List[MonthEndChecklist]:
    """Return saved month-end checklists ordered by name."""
    query = "SELECT * FROM month_end_checklists"
    if not include_inactive:
        query += " WHERE is_active = 1"
    query += " ORDER BY LOWER(name) ASC, id ASC"
    cur = conn.execute(query)
    return [_month_end_checklist_from_row(r) for r in cur.fetchall()]



def get_month_end_checklist_by_name(
    conn: sqlite3.Connection,
    name: str,
) -> Optional[MonthEndChecklist]:
    """Return a month-end checklist by exact name, or None."""
    nm = (name or "").strip()
    if not nm:
        return None
    cur = conn.execute(
        "SELECT * FROM month_end_checklists WHERE name = ?",
        (nm,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return _month_end_checklist_from_row(row)



def get_month_end_checklist_account_ids(
    conn: sqlite3.Connection,
    checklist_id: int,
) -> List[int]:
    """Return account ids for a checklist ordered low-to-high by GL number."""
    cur = conn.execute(
        """
        SELECT m.account_id
        FROM month_end_checklist_accounts m
        JOIN accounts a
          ON a.id = m.account_id
        WHERE m.checklist_id = ?
        ORDER BY a.gl_number ASC, a.id ASC, m.id ASC
        """,
        (int(checklist_id),),
    )
    return [int(r["account_id"]) for r in cur.fetchall()]



def update_month_end_checklist_accounts(
    conn: sqlite3.Connection,
    checklist_id: int,
    account_ids: List[int],
) -> None:
    """Replace a checklist's accounts with the provided active accounts.

    Membership is stored low-to-high by GL number.
    """
    normalized_ids = [int(x) for x in (account_ids or []) if x is not None]
    normalized_ids = sorted(set(normalized_ids))
    if not normalized_ids:
        raise ValueError("Select at least one active account for the checklist.")

    placeholders = ",".join(["?"] * len(normalized_ids))
    valid_rows = conn.execute(
        f"""
        SELECT id, gl_number
        FROM accounts
        WHERE is_active = 1
          AND id IN ({placeholders})
        ORDER BY gl_number ASC, id ASC
        """,
        normalized_ids,
    ).fetchall()

    ordered_account_ids = [int(r["id"]) for r in valid_rows]
    if len(ordered_account_ids) != len(normalized_ids):
        raise ValueError("All checklist accounts must be active accounts.")

    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM month_end_checklist_accounts WHERE checklist_id = ?",
            (int(checklist_id),),
        )
        cur.executemany(
            """
            INSERT INTO month_end_checklist_accounts (checklist_id, account_id, sort_order)
            VALUES (?, ?, ?)
            """,
            [
                (int(checklist_id), account_id, idx)
                for idx, account_id in enumerate(ordered_account_ids, start=1)
            ],
        )
        cur.execute(
            """
            UPDATE month_end_checklists
            SET updated_at = datetime('now')
            WHERE id = ?
            """,
            (int(checklist_id),),
        )

        log_activity(
            conn,
            event_type="MONTH_END_CHECKLIST_UPDATED",
            entity_type="MONTH_END_CHECKLIST",
            entity_id=int(checklist_id),
            summary=f"Updated month-end checklist id={int(checklist_id)} accounts",
            details={
                "checklist_id": int(checklist_id),
                "account_ids": ordered_account_ids,
            },
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise



def set_month_end_checklist_active(
    conn: sqlite3.Connection,
    checklist_id: int,
    is_active: bool,
) -> None:
    """Archive/unarchive a month-end checklist by toggling is_active."""
    row = conn.execute(
        "SELECT name FROM month_end_checklists WHERE id = ?",
        (int(checklist_id),),
    ).fetchone()
    checklist_name = str(row["name"]) if row is not None else None

    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE month_end_checklists
            SET is_active = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (1 if is_active else 0, int(checklist_id)),
        )

        log_activity(
            conn,
            event_type="MONTH_END_CHECKLIST_ACTIVATED" if is_active else "MONTH_END_CHECKLIST_ARCHIVED",
            entity_type="MONTH_END_CHECKLIST",
            entity_id=int(checklist_id),
            summary=(
                f"{'Unarchived' if is_active else 'Archived'} month-end checklist"
                + (f" '{checklist_name}'" if checklist_name else "")
            ),
            details={
                "checklist_id": int(checklist_id),
                "is_active": bool(is_active),
                "name": checklist_name,
            },
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise


# --- Month-End Checklist Completion/Status Helpers ---

def _get_month_end_checklist_completion_summary(
    conn: sqlite3.Connection,
    checklist_id: int,
    as_of_date: str,
) -> tuple[int, int, Optional[str]]:
    """Return (account_count, completed_account_count, derived_completed_at) for one checklist/date.

    Overall checklist completion is derived entirely from the account-level rows:
    - complete only when all checklist accounts are completed for the as-of date
    - otherwise incomplete

    derived_completed_at is the most recent account-level completed_at timestamp when the
    checklist is fully complete, else None.
    """
    cur = conn.execute(
        """
        SELECT
            COUNT(DISTINCT m.account_id) AS account_count,
            COALESCE(SUM(CASE WHEN mas.is_completed = 1 THEN 1 ELSE 0 END), 0) AS completed_account_count,
            MAX(CASE WHEN mas.is_completed = 1 THEN mas.completed_at ELSE NULL END) AS derived_completed_at
        FROM month_end_checklist_accounts m
        LEFT JOIN month_end_checklist_account_status mas
          ON mas.checklist_id = m.checklist_id
         AND mas.account_id = m.account_id
         AND mas.as_of_date = ?
        WHERE m.checklist_id = ?
        """,
        (str(as_of_date), int(checklist_id)),
    )
    row = cur.fetchone()

    if row is None:
        return 0, 0, None

    account_count = int(row["account_count"] or 0)
    completed_account_count = int(row["completed_account_count"] or 0)
    derived_completed_at = row["derived_completed_at"]

    if account_count <= 0 or completed_account_count != account_count:
        derived_completed_at = None

    return account_count, completed_account_count, derived_completed_at



def get_month_end_checklist_completed(
    conn: sqlite3.Connection,
    checklist_id: int,
    as_of_date: str,
) -> bool:
    """Return whether a checklist is complete for a given as-of date.

    Completion is derived from account-level checklist rows only.
    """
    account_count, completed_account_count, _ = _get_month_end_checklist_completion_summary(
        conn,
        int(checklist_id),
        str(as_of_date),
    )
    return account_count > 0 and completed_account_count == account_count



def set_month_end_checklist_completed(
    conn: sqlite3.Connection,
    checklist_id: int,
    as_of_date: str,
    is_completed: bool,
) -> None:
    """Persist checklist completion status for a given as-of date."""
    completed_at = datetime.now().isoformat(timespec="seconds") if is_completed else None
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO month_end_checklist_status (checklist_id, as_of_date, is_completed, completed_at, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(checklist_id, as_of_date) DO UPDATE SET
                is_completed = excluded.is_completed,
                completed_at = excluded.completed_at,
                updated_at = datetime('now')
            """,
            (int(checklist_id), str(as_of_date), 1 if is_completed else 0, completed_at),
        )

        log_activity(
            conn,
            event_type="MONTH_END_CHECKLIST_COMPLETED" if is_completed else "MONTH_END_CHECKLIST_UNCHECKED",
            entity_type="MONTH_END_CHECKLIST",
            entity_id=int(checklist_id),
            summary=(
                f"{'Completed' if is_completed else 'Unchecked'} month-end checklist id={int(checklist_id)} "
                f"for {str(as_of_date)}"
            ),
            details={
                "checklist_id": int(checklist_id),
                "as_of_date": str(as_of_date),
                "is_completed": bool(is_completed),
            },
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise

def get_month_end_checklist_account_completed(
    conn: sqlite3.Connection,
    checklist_id: int,
    account_id: int,
    as_of_date: str,
) -> bool:
    """Return whether a checklist account is marked complete for a given as-of date."""
    cur = conn.execute(
        """
        SELECT is_completed
        FROM month_end_checklist_account_status
        WHERE checklist_id = ?
          AND account_id = ?
          AND as_of_date = ?
        """,
        (int(checklist_id), int(account_id), str(as_of_date)),
    )
    row = cur.fetchone()
    if row is None:
        return False
    return bool(row["is_completed"])


def get_month_end_checklist_account_status_map(
    conn: sqlite3.Connection,
    checklist_id: int,
    as_of_date: str,
) -> dict[int, bool]:
    """Return account_id -> completion status for one checklist and as-of date."""
    cur = conn.execute(
        """
        SELECT account_id, is_completed
        FROM month_end_checklist_account_status
        WHERE checklist_id = ?
          AND as_of_date = ?
        """,
        (int(checklist_id), str(as_of_date)),
    )
    rows = cur.fetchall()
    return {int(r["account_id"]): bool(r["is_completed"]) for r in rows}


def set_month_end_checklist_account_completed(
    conn: sqlite3.Connection,
    checklist_id: int,
    account_id: int,
    as_of_date: str,
    is_completed: bool,
) -> None:
    """Persist account-level checklist completion status for a given as-of date."""
    completed_at = datetime.now().isoformat(timespec="seconds") if is_completed else None
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO month_end_checklist_account_status (
                checklist_id, account_id, as_of_date, is_completed, completed_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(checklist_id, account_id, as_of_date) DO UPDATE SET
                is_completed = excluded.is_completed,
                completed_at = excluded.completed_at,
                updated_at = datetime('now')
            """,
            (
                int(checklist_id),
                int(account_id),
                str(as_of_date),
                1 if is_completed else 0,
                completed_at,
            ),
        )

        log_activity(
            conn,
            event_type=(
                "MONTH_END_CHECKLIST_ACCOUNT_COMPLETED"
                if is_completed
                else "MONTH_END_CHECKLIST_ACCOUNT_UNCHECKED"
            ),
            entity_type="MONTH_END_CHECKLIST",
            entity_id=int(checklist_id),
            summary=(
                f"{'Completed' if is_completed else 'Unchecked'} checklist account id={int(account_id)} "
                f"for checklist id={int(checklist_id)} on {str(as_of_date)}"
            ),
            details={
                "checklist_id": int(checklist_id),
                "account_id": int(account_id),
                "as_of_date": str(as_of_date),
                "is_completed": bool(is_completed),
            },
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise

def get_month_end_checklist_display_rows(
    conn: sqlite3.Connection,
    checklist_id: int,
    as_of_date: str,
) -> List[MonthEndChecklistDisplayRow]:
    """Return display rows for a checklist as of a selected date.

    Rows are ordered low-to-high by GL number.
    """
    cur = conn.execute(
        """
        SELECT
            a.id AS account_id,
            a.gl_number,
            a.name AS account_name,
            (
                SELECT je.effective_date
                FROM journal_lines jl
                JOIN journal_entries je
                  ON je.id = jl.journal_entry_id
                WHERE jl.account_id = a.id
                  AND je.effective_date <= ?
                ORDER BY je.effective_date DESC, je.id DESC, jl.sort_order DESC, jl.id DESC
                LIMIT 1
            ) AS most_recent_effective_date,
            (
                SELECT je.post_date
                FROM journal_lines jl
                JOIN journal_entries je
                  ON je.id = jl.journal_entry_id
                WHERE jl.account_id = a.id
                  AND je.effective_date <= ?
                ORDER BY je.effective_date DESC, je.id DESC, jl.sort_order DESC, jl.id DESC
                LIMIT 1
            ) AS most_recent_post_date
        FROM month_end_checklist_accounts m
        JOIN accounts a
          ON a.id = m.account_id
        WHERE m.checklist_id = ?
        ORDER BY a.gl_number ASC, a.id ASC, m.id ASC
        """,
        (str(as_of_date), str(as_of_date), int(checklist_id)),
    )
    rows = cur.fetchall()

    result: List[MonthEndChecklistDisplayRow] = []
    for row in rows:
        account_id = int(row["account_id"])
        result.append(
            MonthEndChecklistDisplayRow(
                account_id=account_id,
                gl_number=int(row["gl_number"]),
                account_name=str(row["account_name"]),
                most_recent_effective_date=row["most_recent_effective_date"],
                most_recent_post_date=row["most_recent_post_date"],
                balance_cents=int(get_account_balance_as_of(conn, account_id, str(as_of_date))),
            )
        )
    return result



def get_month_end_checklist_statuses_for_date(
    conn: sqlite3.Connection,
    as_of_date: str,
) -> List[MonthEndChecklistStatusRow]:
    """Return status rows for all active month-end checklists for one as-of date.

    Overall checklist completion is derived from the account-level completion rows.
    The checklist-level status table is no longer used as the source of truth for display.
    """
    cur = conn.execute(
        """
        SELECT
            c.id AS checklist_id,
            c.name AS checklist_name,
            COUNT(DISTINCT m.account_id) AS account_count,
            COALESCE(SUM(CASE WHEN mas.is_completed = 1 THEN 1 ELSE 0 END), 0) AS completed_account_count,
            MAX(CASE WHEN mas.is_completed = 1 THEN mas.completed_at ELSE NULL END) AS derived_completed_at
        FROM month_end_checklists c
        LEFT JOIN month_end_checklist_accounts m
          ON m.checklist_id = c.id
        LEFT JOIN month_end_checklist_account_status mas
          ON mas.checklist_id = c.id
         AND mas.account_id = m.account_id
         AND mas.as_of_date = ?
        WHERE c.is_active = 1
        GROUP BY c.id, c.name
        ORDER BY LOWER(c.name) ASC, c.id ASC
        """,
        (str(as_of_date),),
    )
    rows = cur.fetchall()

    result: List[MonthEndChecklistStatusRow] = []
    for r in rows:
        account_count = int(r["account_count"] or 0)
        completed_account_count = int(r["completed_account_count"] or 0)
        is_completed = account_count > 0 and completed_account_count == account_count
        completed_at = r["derived_completed_at"] if is_completed else None

        result.append(
            MonthEndChecklistStatusRow(
                checklist_id=int(r["checklist_id"]),
                checklist_name=str(r["checklist_name"]),
                account_count=account_count,
                as_of_date=str(as_of_date),
                is_completed=is_completed,
                completed_at=completed_at,
                completed_account_count=completed_account_count,
            )
        )

    return result


def _parse_date_to_iso(date_str: str) -> str:
    """Parse a date string from bank files into ISO format (YYYY-MM-DD).

    Tries a few common formats, including those with time components such as
    'YYYY-MM-DD HH:MM:SS' which are common when reading from Excel via pandas.
    Raises ValueError if no known format matches.
    """
    if date_str is None:
        raise ValueError("Date string is None")

    s = str(date_str).strip()
    if not s:
        raise ValueError("Empty date string")

    # Try formats without time first, then with time
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue

    raise ValueError(f"Unrecognized date format: {date_str}")



def load_user_defined_excel(
    file_obj,
    mapping: UserUploadMapping,
) -> List[GenericImportTxn]:
    """Load and map a user-defined Excel file using a stored mapping."""
    df = pd.read_excel(file_obj)

    required = {mapping.effective_date_col, mapping.amount_col, mapping.description_col}
    missing = required - set(df.columns.astype(str))
    if missing:
        raise ValueError(
            "Excel file is missing required columns for this mapping: "
            + ", ".join(sorted(missing))
        )

    txns: List[GenericImportTxn] = []
    for row in df.to_dict(orient="records"):
        if not any((str(v).strip() for v in row.values() if v is not None)):
            continue

        eff_raw = row.get(mapping.effective_date_col)
        amt_raw = row.get(mapping.amount_col)
        desc_raw = row.get(mapping.description_col)

        if eff_raw is None or str(eff_raw).strip() == "":
            raise ValueError(f"Missing effective date value in column '{mapping.effective_date_col}'")

        effective_date_iso = _parse_date_to_iso(str(eff_raw))

        if amt_raw is None or str(amt_raw).strip() == "":
            raise ValueError(f"Missing amount value in column '{mapping.amount_col}'")

        try:
            amount_float = float(str(amt_raw).replace(",", "").strip())
        except ValueError as e:
            raise ValueError(f"Could not parse Amount '{amt_raw}' as a number") from e

        if amount_float == 0:
            continue

        # Determine base-side debit/credit based on sign and mapping config
        if amount_float > 0:
            is_debit_for_account = bool(mapping.positive_is_debit)
        else:
            is_debit_for_account = not bool(mapping.positive_is_debit)

        amount_cents = int(round(abs(amount_float) * 100))
        description = (str(desc_raw) if desc_raw is not None else "").strip()

        txns.append(
            GenericImportTxn(
                effective_date=effective_date_iso,
                description=description,
                amount_cents=amount_cents,
                is_debit_for_account=is_debit_for_account,
                raw_row=row,
            )
        )

    return txns


# --- User-defined upload mapping: Excel/CSV loader supporting both formats ---

def _read_uploaded_file_to_dataframe(file_obj) -> pd.DataFrame:
    """Read an uploaded file (Excel or CSV) into a DataFrame.

    This is intentionally format-tolerant and used for user-defined import mappings.

    Rules:
    - Excel extensions: .xlsx, .xls, .xlsm, .xlsb -> pandas.read_excel
    - CSV extension: .csv -> pandas.read_csv (UTF-8/UTF-8-SIG/Latin-1 fallback)

    Notes:
    - Streamlit UploadedFile behaves like a file-like object and may be in binary mode.
    - We do NOT mutate or assume any specific column names here.
    """
    name = getattr(file_obj, "name", "") or ""
    ext = Path(str(name)).suffix.lower()

    # Always rewind if possible (Streamlit UploadedFile is seekable)
    try:
        file_obj.seek(0)
    except Exception:
        pass

    if ext in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        return pd.read_excel(file_obj)

    if ext == ".csv":
        # Read bytes safely, then decode with fallback encodings.
        raw_bytes: bytes
        if hasattr(file_obj, "getvalue"):
            raw_bytes = file_obj.getvalue()
        else:
            raw_bytes = file_obj.read()

        text: str
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = raw_bytes.decode(enc)
                break
            except Exception:
                text = ""
        if not text:
            raise ValueError("Could not decode CSV file. Please save/export the file as UTF-8 and try again.")

        # Use a StringIO wrapper so pandas reads consistently.
        sio = io.StringIO(text)
        return pd.read_csv(sio)

    raise ValueError(
        "Unsupported file type. Please upload an Excel (.xlsx/.xls) or CSV (.csv) file."
    )


def load_user_defined_file(
    file_obj,
    mapping: UserUploadMapping,
) -> List[GenericImportTxn]:
    """Load and map a user-defined upload file (Excel or CSV) using a stored mapping.

    This function is additive and does not change existing Excel-only behavior.
    Use this for user-defined mappings so users can upload either .xlsx/.xls or .csv.
    """
    df = _read_uploaded_file_to_dataframe(file_obj)

    required = {mapping.effective_date_col, mapping.amount_col, mapping.description_col}
    missing = required - set(df.columns.astype(str))
    if missing:
        raise ValueError(
            "Uploaded file is missing required columns for this mapping: "
            + ", ".join(sorted(missing))
        )

    txns: List[GenericImportTxn] = []
    for row in df.to_dict(orient="records"):
        if not any((str(v).strip() for v in row.values() if v is not None)):
            continue

        eff_raw = row.get(mapping.effective_date_col)
        amt_raw = row.get(mapping.amount_col)
        desc_raw = row.get(mapping.description_col)

        # For custom mapping uploads, skip rows that do not have an effective date.
        # This is especially important for CSV files, where blank cells often come
        # through pandas as NaN instead of None/empty string.
        if pd.isna(eff_raw) or str(eff_raw).strip() == "":
            continue

        effective_date_iso = _parse_date_to_iso(str(eff_raw))

        if amt_raw is None or str(amt_raw).strip() == "":
            raise ValueError(f"Missing amount value in column '{mapping.amount_col}'")

        # Parse amount (be tolerant of common CSV formatting like $ and parentheses)
        amt_s = str(amt_raw).strip()
        amt_s = amt_s.replace(",", "").replace("$", "")
        is_paren_negative = amt_s.startswith("(") and amt_s.endswith(")")
        if is_paren_negative:
            amt_s = amt_s[1:-1].strip()

        try:
            amount_float = float(amt_s)
        except ValueError as e:
            raise ValueError(f"Could not parse Amount '{amt_raw}' as a number") from e

        if is_paren_negative:
            amount_float = -abs(amount_float)

        if amount_float == 0:
            continue

        # Determine base-side debit/credit based on sign and mapping config
        if amount_float > 0:
            is_debit_for_account = bool(mapping.positive_is_debit)
        else:
            is_debit_for_account = not bool(mapping.positive_is_debit)

        amount_cents = int(round(abs(amount_float) * 100))
        description = (str(desc_raw) if desc_raw is not None else "").strip()

        txns.append(
            GenericImportTxn(
                effective_date=effective_date_iso,
                description=description,
                amount_cents=amount_cents,
                is_debit_for_account=is_debit_for_account,
                raw_row=row,
            )
        )

    return txns



def get_connection() -> sqlite3.Connection:
    """
    Return a connection to the SQLite database.
    Ensures foreign keys are enforced.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_accounts_table(conn: sqlite3.Connection) -> None:
    """
    Create the `accounts` table if it does not already exist.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            gl_number        INTEGER NOT NULL UNIQUE,
            name             TEXT NOT NULL,
            type             TEXT NOT NULL CHECK (type IN ('ASSET','LIABILITY','EQUITY','INCOME','EXPENSE')),
            normal_balance   TEXT CHECK (normal_balance IN ('DEBIT','CREDIT')),
            is_active        INTEGER NOT NULL DEFAULT 1,
            created_at       TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )


# --- DB initialization: Call all table init helpers here ---


def init_journal_entries_table(conn: sqlite3.Connection) -> None:
    """
    Create the `journal_entries` table if it does not already exist.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS journal_entries (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            effective_date   TEXT NOT NULL,
            post_date        TEXT NOT NULL,
            description      TEXT,
            source           TEXT,
            created_at       TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )


def init_journal_lines_table(conn: sqlite3.Connection) -> None:
    """
    Create the `journal_lines` table if it does not already exist.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS journal_lines (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            journal_entry_id INTEGER NOT NULL,
            account_id       INTEGER NOT NULL,
            amount_cents     INTEGER NOT NULL,
            is_debit         INTEGER NOT NULL,
            sort_order       INTEGER NOT NULL,
            memo             TEXT,
            created_at       TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (journal_entry_id) REFERENCES journal_entries(id) ON DELETE CASCADE,
            FOREIGN KEY (account_id) REFERENCES accounts(id)
        );
        """
    )


def seed_initial_chart_of_accounts(conn: sqlite3.Connection) -> None:
    """
    Seed the initial chart of accounts if the `accounts` table is empty.

    This uses the lean chart of accounts we designed together.
    If you later want to change these, you can either:
    - modify this seed list and recreate the DB, or
    - use the UI we'll build to add/rename/archive accounts.
    """
    cur = conn.execute("SELECT COUNT(*) AS cnt FROM accounts;")
    row = cur.fetchone()
    if row is None or row["cnt"] > 0:
        # Table is not empty; do not re-seed.
        return

    # (gl_number, name, type, normal_balance)
    seed_accounts: List[tuple[int, str, AccountType, NormalBalance]] = [
        # Assets (1xxxx)
        (11000, "Checking – Primary", AccountType.ASSET, NormalBalance.DEBIT),
        (11100, "Savings – Main", AccountType.ASSET, NormalBalance.DEBIT),
        (11200, "Cash On Hand", AccountType.ASSET, NormalBalance.DEBIT),
        (12000, "Certificates of Deposit", AccountType.ASSET, NormalBalance.DEBIT),
        (13000, "Taxable Investments", AccountType.ASSET, NormalBalance.DEBIT),
        (13100, "Retirement Investments", AccountType.ASSET, NormalBalance.DEBIT),
        (14000, "Accrued Income Receivable", AccountType.ASSET, NormalBalance.DEBIT),
        (14100, "Due To/From Friends & Family", AccountType.ASSET, NormalBalance.DEBIT),
        (15000, "Fixed Assets", AccountType.ASSET, NormalBalance.DEBIT),
        (19900, "Suspense - Unapplied Transactions", AccountType.ASSET, NormalBalance.DEBIT),

        # Liabilities (2xxxx)
        (21000, "Credit Card – Primary", AccountType.LIABILITY, NormalBalance.CREDIT),
        (21100, "Other Revolving Debt", AccountType.LIABILITY, NormalBalance.CREDIT),
        (22100, "Auto Loan", AccountType.LIABILITY, NormalBalance.CREDIT),
        (23000, "Other Installment Debt", AccountType.LIABILITY, NormalBalance.CREDIT),
        (25000, "Transfers Clearing", AccountType.LIABILITY, NormalBalance.CREDIT),

        # Equity (3xxxx)
        (30000, "Opening Balance Equity", AccountType.EQUITY, NormalBalance.CREDIT),
        (30010, "Retained Earnings", AccountType.EQUITY, NormalBalance.CREDIT),

        # Income (4xxxx)
        (41000, "Salary & Wages Income", AccountType.INCOME, NormalBalance.CREDIT),
        (42000, "Other Earned Income", AccountType.INCOME, NormalBalance.CREDIT),
        (43000, "Interest & Investment Income", AccountType.INCOME, NormalBalance.CREDIT),
        (45000, "Gifts & Transfers In", AccountType.INCOME, NormalBalance.CREDIT),
        (49000, "Miscellaneous Income", AccountType.INCOME, NormalBalance.CREDIT),

        # Expenses (5xxxx)
        (51000, "Rent & Housing Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (51100, "Phone & Utilities Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (51200, "Grocery Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (51300, "Restaurant & Bar Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (51400, "Gas & Transportation Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (51500, "Car Insurance Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (51600, "Car Maintenance Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (51700, "Travel & Vacation Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (51800, "Subscriptions Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (51900, "Personal & Household Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (52100, "Gifts Given Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
        (56000, "Bank Fees", AccountType.EXPENSE, NormalBalance.DEBIT),
        (59000, "Other Expense", AccountType.EXPENSE, NormalBalance.DEBIT),
    ]

    conn.executemany(
        """
        INSERT INTO accounts (gl_number, name, type, normal_balance, is_active)
        VALUES (?, ?, ?, ?, 1);
        """,
        [(gl, name, acc_type.value, normal.value) for gl, name, acc_type, normal in seed_accounts],
    )
    conn.commit()


def insert_journal_entry(
    conn: sqlite3.Connection,
    effective_date: str,
    post_date: str,
    description: Optional[str],
    source: Optional[str],
    lines: List[JournalEntryLine],
) -> int:
    """
    Insert a journal entry with its lines into the database.

    Validates that total debits equal total credits.

    Returns the id of the inserted journal entry.
    """
    total_debits = sum(line.amount_cents for line in lines if line.is_debit)
    total_credits = sum(line.amount_cents for line in lines if not line.is_debit)

    if total_debits != total_credits:
        raise ValueError(f"Total debits ({total_debits}) do not equal total credits ({total_credits}).")

    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO journal_entries (effective_date, post_date, description, source)
            VALUES (?, ?, ?, ?);
            """,
            (effective_date, post_date, description, source),
        )
        journal_entry_id = cursor.lastrowid

        journal_lines_data = []
        for i, line in enumerate(lines, start=1):
            journal_lines_data.append((
                journal_entry_id,
                line.account_id,
                line.amount_cents,
                1 if line.is_debit else 0,
                i,
                line.memo,
            ))

        cursor.executemany(
            """
            INSERT INTO journal_lines (
                journal_entry_id, account_id, amount_cents, is_debit, sort_order, memo
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            journal_lines_data,
        )

        # Log the posted journal entry (best effort).
        log_activity(
            conn,
            event_type="JOURNAL_ENTRY_POSTED",
            entity_type="JOURNAL_ENTRY",
            entity_id=journal_entry_id,
            summary=(description or "").strip() or f"Journal entry {journal_entry_id} posted",
            details={
                "effective_date": effective_date,
                "post_date": post_date,
                "source": source,
                "total_debits_cents": total_debits,
                "total_credits_cents": total_credits,
            },
        )

        conn.commit()
        return journal_entry_id
    except Exception:
        conn.rollback()
        raise


def post_journal_entry_with_period_lock(
    conn: sqlite3.Connection,
    effective_date: str,
    post_date: str,
    description: Optional[str],
    source: Optional[str],
    lines: List[JournalEntryLine],
    override_pin: Optional[str] = None,
) -> int:
    """
    Wrapper around insert_journal_entry that enforces a closed-period lock.

    If a last_closed_period_end setting exists and the effective_date is on or
    before that date, a valid override PIN is required to post the entry.
    Otherwise, the entry is posted normally.

    This function does NOT change the behavior of insert_journal_entry itself;
    it only adds a guard before calling it.
    """
    # Read the last closed period from settings.
    last_closed = get_last_closed_period_end(conn)

    # If no closed period is configured, allow posting.
    if not last_closed:
        return insert_journal_entry(
            conn=conn,
            effective_date=effective_date,
            post_date=post_date,
            description=description,
            source=source,
            lines=lines,
        )

    # Compare dates as ISO strings (YYYY-MM-DD) which sort lexicographically.
    if effective_date <= last_closed:
        # Posting into a closed period requires a valid override PIN.
        stored_pin = get_override_pin(conn)

        if not stored_pin:
            raise ValueError(
                "Cannot post into a closed period because no override PIN is configured."
            )

        if override_pin is None or override_pin.strip() == "":
            raise ValueError(
                "Cannot post into a closed period without providing an override PIN."
            )

        if override_pin.strip() != stored_pin:
            raise ValueError(
                "Invalid override PIN. Posting into the closed period is not allowed."
            )

        # Valid PIN provided; allow posting to proceed.

    # Either effective_date is after the closed period, or PIN check passed.
    return insert_journal_entry(
        conn=conn,
        effective_date=effective_date,
        post_date=post_date,
        description=description,
        source=source,
        lines=lines,
    )


# --- Query-focused helpers ---

def _account_from_row(row: sqlite3.Row) -> Account:
    """
    Private helper to construct an Account dataclass from a DB row.
    """
    return Account(
        id=row["id"],
        gl_number=row["gl_number"],
        name=row["name"],
        type=AccountType(row["type"]),
        normal_balance=NormalBalance(row["normal_balance"]),
        is_active=bool(row["is_active"]),
    )


def get_accounts(
    conn: sqlite3.Connection,
    include_inactive: bool = False,
    search: Optional[str] = None,
) -> List[Account]:
    """
    Retrieve a list of accounts ordered by gl_number.
    If include_inactive is False, only active accounts are returned.
    If search is provided, filters accounts where the name or gl_number contains the search string (case-insensitive).
    """
    params = []
    query = "SELECT * FROM accounts"
    where_clauses = []
    if not include_inactive:
        where_clauses.append("is_active = 1")
    if search:
        where_clauses.append("(LOWER(name) LIKE ? OR CAST(gl_number AS TEXT) LIKE ?)")
        s = f"%{search.lower()}%"
        params.extend([s, s])
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY gl_number"
    cur = conn.execute(query, params)
    return [_account_from_row(row) for row in cur.fetchall()]


def get_account_by_gl_number(
    conn: sqlite3.Connection,
    gl_number: int
) -> Optional[Account]:
    """
    Retrieve a single account by its gl_number.
    Returns an Account object, or None if not found.
    """
    cur = conn.execute("SELECT * FROM accounts WHERE gl_number = ?", (gl_number,))
    row = cur.fetchone()
    if row is None:
        return None
    return _account_from_row(row)


def get_account_balance_as_of(
    conn: sqlite3.Connection,
    account_id: int,
    as_of_date: str,
) -> int:
    """
    Compute the net balance (in cents, integer) for an account as of a given date (inclusive).
    Joins journal_lines and journal_entries on journal_entry_id.
    Only includes lines where effective_date <= as_of_date and matching account_id.
    Returns debits minus credits (positive means net debit, negative means net credit).
    Returns 0 if no matching rows.
    """
    cur = conn.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN jl.is_debit = 1 THEN jl.amount_cents ELSE 0 END), 0) AS debits,
            COALESCE(SUM(CASE WHEN jl.is_debit = 0 THEN jl.amount_cents ELSE 0 END), 0) AS credits
        FROM journal_lines jl
        JOIN journal_entries je ON jl.journal_entry_id = je.id
        WHERE jl.account_id = ?
          AND je.effective_date <= ?
        """,
        (account_id, as_of_date),
    )
    row = cur.fetchone()
    if row is None:
        return 0
    return row["debits"] - row["credits"]


def get_account_history(
    conn: sqlite3.Connection,
    account_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_journal_entry_id: Optional[int] = None,
) -> tuple[int, List[AccountHistoryRow]]:
    """
    Retrieve the detailed history for a given account, including a running balance.

    Dates are strings 'YYYY-MM-DD'. If start_date is provided, only entries with
    effective_date >= start_date are included. If end_date is provided, only
    entries with effective_date <= end_date are included.

    Returns:
        (opening_balance_cents, rows)

        opening_balance_cents: the net balance (debits - credits, in cents)
        for this account strictly before start_date (or 0 if start_date is None).

        rows: a list of AccountHistoryRow objects ordered by
        effective_date, journal_entry_id, sort_order, each with a running
        balance that starts from opening_balance_cents.
    """
    # Compute opening balance.
    #
    # - If start_date is provided, opening balance is strictly before start_date (existing behavior).
    # - If start_date is not provided but min_journal_entry_id is provided, opening balance is the
    #   balance up through that JE id (so running balances remain correct when you hide older JEs).
    opening_balance_cents = 0

    if start_date is not None:
        cur = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN jl.is_debit = 1 THEN jl.amount_cents ELSE 0 END), 0) AS debits,
                COALESCE(SUM(CASE WHEN jl.is_debit = 0 THEN jl.amount_cents ELSE 0 END), 0) AS credits
            FROM journal_lines jl
            JOIN journal_entries je ON jl.journal_entry_id = je.id
            WHERE jl.account_id = ?
              AND je.effective_date < ?
            """,
            (account_id, start_date),
        )
        row = cur.fetchone()
        if row is not None:
            opening_balance_cents = row["debits"] - row["credits"]

    elif min_journal_entry_id is not None:
        cur = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN jl.is_debit = 1 THEN jl.amount_cents ELSE 0 END), 0) AS debits,
                COALESCE(SUM(CASE WHEN jl.is_debit = 0 THEN jl.amount_cents ELSE 0 END), 0) AS credits
            FROM journal_lines jl
            JOIN journal_entries je ON jl.journal_entry_id = je.id
            WHERE jl.account_id = ?
              AND je.id <= ?
            """,
            (account_id, int(min_journal_entry_id)),
        )
        row = cur.fetchone()
        if row is not None:
            opening_balance_cents = row["debits"] - row["credits"]

    # Build query for the actual history rows in the requested range.
    params: List[object] = [account_id]
    where_clauses = ["jl.account_id = ?"]
    if start_date is not None:
        where_clauses.append("je.effective_date >= ?")
        params.append(start_date)
    if end_date is not None:
        where_clauses.append("je.effective_date <= ?")
        params.append(end_date)
    if min_journal_entry_id is not None:
        where_clauses.append("je.id > ?")
        params.append(int(min_journal_entry_id))

    query = f"""
        SELECT
            je.id AS journal_entry_id,
            je.effective_date,
            je.post_date,
            je.description,
            je.source,
            jl.memo,
            jl.amount_cents,
            jl.is_debit,
            jl.sort_order
        FROM journal_lines jl
        JOIN journal_entries je ON jl.journal_entry_id = je.id
        WHERE {' AND '.join(where_clauses)}
        ORDER BY je.effective_date, je.id, jl.sort_order
    """

    cur = conn.execute(query, params)
    rows = cur.fetchall()

    history: List[AccountHistoryRow] = []
    running = opening_balance_cents

    for r in rows:
        amount = r["amount_cents"]
        is_debit = bool(r["is_debit"])
        delta = amount if is_debit else -amount
        running += delta

        debit_cents = amount if is_debit else 0
        credit_cents = amount if not is_debit else 0

        history.append(
            AccountHistoryRow(
                effective_date=r["effective_date"],
                post_date=r["post_date"],
                journal_entry_id=r["journal_entry_id"],
                description=r["description"],
                source=r["source"],
                memo=r["memo"],
                debit_cents=debit_cents,
                credit_cents=credit_cents,
                running_balance_cents=running,
            )
        )

    return opening_balance_cents, history


# --- Journal entry search helpers ---

def get_distinct_journal_entry_sources(conn: sqlite3.Connection) -> List[str]:
    """Return distinct JE sources sorted alphabetically.

    Blank sources are excluded. The UI uses this to populate the JE Source dropdown.
    """
    cur = conn.execute(
        """
        SELECT DISTINCT TRIM(source) AS source
        FROM journal_entries
        WHERE source IS NOT NULL
          AND TRIM(source) <> ''
        ORDER BY LOWER(TRIM(source)) ASC
        """
    )
    rows = cur.fetchall()
    return [str(row["source"]) for row in rows]


def search_journal_entry_lines(
    conn: sqlite3.Connection,
    effective_date_from: Optional[str] = None,
    effective_date_to: Optional[str] = None,
    post_date_from: Optional[str] = None,
    post_date_to: Optional[str] = None,
    amount_cents: Optional[int] = None,
    source: Optional[str] = None,
    journal_entry_id: Optional[int] = None,
    journal_line_id: Optional[int] = None,
    description_contains: Optional[str] = None,
    memo_contains: Optional[str] = None,
    account_ids: Optional[List[int]] = None,
    is_debit: Optional[bool] = None,
) -> List[JournalEntrySearchRow]:
    """Search journal entry lines using optional filters.

    Returns one row per journal line, with JE-level fields repeated. Text searches
    on description and memo are case-insensitive substring matches.

    Notes:
    - `amount_cents` is matched exactly against `journal_lines.amount_cents`.
    - `account_ids` uses OR semantics: if multiple account ids are provided, lines
      for any of those accounts are returned.
    - Results are sorted so JE lines stay grouped naturally.
    """
    params: List[object] = []
    where_clauses: List[str] = []

    if effective_date_from:
        where_clauses.append("je.effective_date >= ?")
        params.append(str(effective_date_from))

    if effective_date_to:
        where_clauses.append("je.effective_date <= ?")
        params.append(str(effective_date_to))

    if post_date_from:
        where_clauses.append("je.post_date >= ?")
        params.append(str(post_date_from))

    if post_date_to:
        where_clauses.append("je.post_date <= ?")
        params.append(str(post_date_to))

    if amount_cents is not None:
        where_clauses.append("jl.amount_cents = ?")
        params.append(int(amount_cents))

    if source is not None and str(source).strip() != "":
        where_clauses.append("je.source = ?")
        params.append(str(source).strip())

    if journal_entry_id is not None:
        where_clauses.append("je.id = ?")
        params.append(int(journal_entry_id))

    if journal_line_id is not None:
        where_clauses.append("jl.id = ?")
        params.append(int(journal_line_id))

    if description_contains is not None and str(description_contains).strip() != "":
        where_clauses.append("LOWER(COALESCE(je.description, '')) LIKE ?")
        params.append(f"%{str(description_contains).strip().lower()}%")

    if memo_contains is not None and str(memo_contains).strip() != "":
        where_clauses.append("LOWER(COALESCE(jl.memo, '')) LIKE ?")
        params.append(f"%{str(memo_contains).strip().lower()}%")

    normalized_account_ids = [int(x) for x in (account_ids or []) if x is not None]
    if normalized_account_ids:
        placeholders = ",".join(["?"] * len(normalized_account_ids))
        where_clauses.append(f"jl.account_id IN ({placeholders})")
        params.extend(normalized_account_ids)

    if is_debit is not None:
        where_clauses.append("jl.is_debit = ?")
        params.append(1 if bool(is_debit) else 0)

    query = """
        SELECT
            je.id AS journal_entry_id,
            jl.id AS journal_line_id,
            je.effective_date,
            je.post_date,
            je.description,
            je.source,
            a.id AS account_id,
            a.gl_number,
            a.name AS account_name,
            jl.amount_cents,
            jl.is_debit,
            jl.sort_order,
            jl.memo
        FROM journal_lines jl
        JOIN journal_entries je
          ON jl.journal_entry_id = je.id
        JOIN accounts a
          ON jl.account_id = a.id
    """

    if where_clauses:
        query += "\nWHERE " + " AND ".join(where_clauses)

    query += """

        ORDER BY
            je.effective_date ASC,
            je.id ASC,
            jl.sort_order ASC,
            jl.id ASC
    """

    cur = conn.execute(query, params)
    rows = cur.fetchall()

    return [
        JournalEntrySearchRow(
            journal_entry_id=int(r["journal_entry_id"]),
            journal_line_id=int(r["journal_line_id"]),
            effective_date=str(r["effective_date"]),
            post_date=str(r["post_date"]),
            description=r["description"],
            source=r["source"],
            account_id=int(r["account_id"]),
            gl_number=int(r["gl_number"]),
            account_name=str(r["account_name"]),
            amount_cents=int(r["amount_cents"]),
            is_debit=bool(r["is_debit"]),
            sort_order=int(r["sort_order"]),
            memo=r["memo"],
        )
        for r in rows
    ]


def get_unbalanced_journal_entries(
    conn: sqlite3.Connection,
) -> List[tuple[int, int, int]]:
    """
    Return journal entries where total debits and credits do not match.

    Each tuple is (journal_entry_id, total_debits_cents, total_credits_cents).
    """
    cur = conn.execute(
        """
        SELECT
            jl.journal_entry_id,
            SUM(CASE WHEN jl.is_debit = 1 THEN jl.amount_cents ELSE 0 END) AS total_debits,
            SUM(CASE WHEN jl.is_debit = 0 THEN jl.amount_cents ELSE 0 END) AS total_credits
        FROM journal_lines jl
        GROUP BY jl.journal_entry_id
        HAVING total_debits != total_credits;
        """
    )
    rows = cur.fetchall()
    return [
        (
            int(row["journal_entry_id"]),
            int(row["total_debits"]),
            int(row["total_credits"]),
        )
        for row in rows
    ]


# --- Account maintenance helpers ---

def create_account(
    conn: sqlite3.Connection,
    gl_number: int,
    name: str,
    type_code: str,
    normal_balance_code: str,
    is_active: bool = True,
) -> int:
    """
    Create a new account in the chart of accounts.

    type_code should be one of: 'ASSET', 'LIABILITY', 'EQUITY', 'INCOME', 'EXPENSE'.
    normal_balance_code should be 'DEBIT' or 'CREDIT'.

    Returns the new account's id.
    """
    if not name.strip():
        raise ValueError("Account name cannot be empty.")

    # Normalize and validate codes
    type_code = type_code.upper()
    normal_balance_code = normal_balance_code.upper()
    if type_code not in {"ASSET", "LIABILITY", "EQUITY", "INCOME", "EXPENSE"}:
        raise ValueError(f"Invalid account type: {type_code}")
    if normal_balance_code not in {"DEBIT", "CREDIT"}:
        raise ValueError(f"Invalid normal balance: {normal_balance_code}")

    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO accounts (gl_number, name, type, normal_balance, is_active)
            VALUES (?, ?, ?, ?, ?)
            """,
            (gl_number, name.strip(), type_code, normal_balance_code, 1 if is_active else 0),
        )
        new_id = cur.lastrowid

        # Log account creation.
        log_activity(
            conn,
            event_type="ACCOUNT_CREATED",
            entity_type="ACCOUNT",
            entity_id=new_id,
            summary=f"Created account {gl_number} – {name.strip()}",
            details={
                "gl_number": gl_number,
                "name": name.strip(),
                "type_code": type_code,
                "normal_balance_code": normal_balance_code,
                "is_active": bool(is_active),
            },
        )

        conn.commit()
    except sqlite3.IntegrityError as e:
        # Likely duplicate GL number or constraint violation
        raise ValueError(f"Could not create account: {e}") from e

    return new_id


def update_account_active(
    conn: sqlite3.Connection,
    account_id: int,
    is_active: bool,
) -> None:
    """
    Archive or unarchive an account by toggling is_active.
    Transactions remain in the ledger; only visibility changes.
    """
    # Fetch account info for logging.
    cur = conn.cursor()
    cur.execute(
        "SELECT gl_number, name FROM accounts WHERE id = ?",
        (account_id,),
    )
    row = cur.fetchone()
    gl_number = int(row["gl_number"]) if row is not None else None
    name = row["name"] if row is not None else None

    cur.execute(
        "UPDATE accounts SET is_active = ? WHERE id = ?",
        (1 if is_active else 0, account_id),
    )

    # Log account activation/archiving.
    status = "activated" if is_active else "archived"
    summary_parts = [f"Account id={account_id} {status}"]
    if gl_number is not None and name is not None:
        summary_parts.append(f"({gl_number} – {name})")
    summary = " ".join(summary_parts)

    log_activity(
        conn,
        event_type="ACCOUNT_ACTIVATED" if is_active else "ACCOUNT_ARCHIVED",
        entity_type="ACCOUNT",
        entity_id=account_id,
        summary=summary,
        details={
            "account_id": account_id,
            "gl_number": gl_number,
            "name": name,
            "is_active": bool(is_active),
        },
    )

    conn.commit()


def delete_account(
    conn: sqlite3.Connection,
    account_id: int,
) -> None:
    """
    Permanently delete an account and all of its journal lines.
    Journal entries themselves are preserved but may have fewer lines.
    """
    cur = conn.cursor()
    # Fetch account info for logging before deletion.
    cur.execute(
        "SELECT gl_number, name FROM accounts WHERE id = ?",
        (account_id,),
    )
    row = cur.fetchone()
    gl_number = int(row["gl_number"]) if row is not None else None
    name = row["name"] if row is not None else None

    # Delete all journal_lines using this account
    cur.execute(
        "DELETE FROM journal_lines WHERE account_id = ?",
        (account_id,),
    )
    # Delete the account record itself
    cur.execute(
        "DELETE FROM accounts WHERE id = ?",
        (account_id,),
    )

    # Log account deletion.
    summary_parts = [f"Deleted account id={account_id}"]
    if gl_number is not None and name is not None:
        summary_parts.append(f"({gl_number} – {name})")
    summary = " ".join(summary_parts)

    log_activity(
        conn,
        event_type="ACCOUNT_DELETED",
        entity_type="ACCOUNT",
        entity_id=account_id,
        summary=summary,
        details={
            "account_id": account_id,
            "gl_number": gl_number,
            "name": name,
        },
    )

    conn.commit()


# --- Simplified Balance Sheet Category Mapping Helpers ---

def init_bs_category_mappings_table(conn: sqlite3.Connection) -> None:
    """Create the bs_category_mappings table if it does not already exist.

    Each row links a simplified balance sheet category key to a GL number.
    The same GL can appear in multiple categories if desired, but typically
    each GL will belong to one high-level category.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bs_category_mappings (
            category_key TEXT NOT NULL,
            gl_number    INTEGER NOT NULL,
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (category_key, gl_number)
        );
        """
    )


def get_bs_category_gl_numbers(
    conn: sqlite3.Connection,
    category_key: str,
) -> List[int]:
    """Return the list of GL numbers mapped to a simplified BS category.

    The result is always sorted by GL number. If no mappings exist for the
    given category_key, an empty list is returned.
    """
    cur = conn.execute(
        "SELECT gl_number FROM bs_category_mappings WHERE category_key = ? ORDER BY gl_number",
        (category_key,),
    )
    rows = cur.fetchall()
    return [int(row["gl_number"]) for row in rows]


def set_bs_category_gl_numbers(
    conn: sqlite3.Connection,
    category_key: str,
    gl_numbers: List[int],
) -> None:
    """Replace the GL list for a simplified BS category with the provided values.

    This deletes any existing rows for the category_key and inserts the new
    set of GL numbers. Duplicates in gl_numbers are ignored.
    """
    # Normalize and deduplicate GL numbers
    unique_gls = sorted({int(gl) for gl in gl_numbers})

    cur = conn.cursor()
    try:
        # Delete existing mappings for this category
        cur.execute(
            "DELETE FROM bs_category_mappings WHERE category_key = ?",
            (category_key,),
        )

        # Insert new mappings, if any
        if unique_gls:
            rows = [(category_key, gl) for gl in unique_gls]
            cur.executemany(
                """
                INSERT INTO bs_category_mappings (category_key, gl_number)
                VALUES (?, ?)
                """,
                rows,
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise


# --- App-wide settings helpers (for things like retained earnings GL) ---

def init_settings_table(conn: sqlite3.Connection) -> None:
    """Create the app_settings table if it does not already exist.

    This is a simple key/value store for global settings such as the
    retained earnings GL number.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )


# --- Transaction Keyword Rules Table ---

def init_txn_keyword_rules_table(conn: sqlite3.Connection) -> None:
    """Create the txn_keyword_rules table if it does not already exist.

    This stores simple keyword-based auto-mapping rules for imported
    transactions. Each rule says: if the transaction description contains
    `keyword` (case-insensitive), then suggest `gl_number` as the offset GL.

    We assume keywords are unique, so we add a UNIQUE constraint on keyword.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS txn_keyword_rules (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword    TEXT NOT NULL UNIQUE,
            gl_number  INTEGER NOT NULL,
            is_active  INTEGER NOT NULL DEFAULT 1,
            priority   INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (gl_number) REFERENCES accounts(gl_number)
        );
        """
    )

def init_manual_txn_templates_table(conn: sqlite3.Connection) -> None:
    """Create the manual_txn_templates table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_txn_templates (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL UNIQUE,
            base_gl_number   INTEGER NOT NULL,
            base_side        TEXT NOT NULL CHECK (base_side IN ('DEBIT','CREDIT')),
            is_active        INTEGER NOT NULL DEFAULT 1,
            created_at       TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (base_gl_number) REFERENCES accounts(gl_number)
        );
        """
    )

def init_user_upload_mappings_table(conn: sqlite3.Connection) -> None:
    """Create the user_upload_mappings table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_upload_mappings (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            name               TEXT NOT NULL UNIQUE,
            base_gl_number     INTEGER NOT NULL,
            positive_is_debit  INTEGER NOT NULL CHECK (positive_is_debit IN (0,1)),
            effective_date_col TEXT NOT NULL,
            amount_col         TEXT NOT NULL,
            description_col    TEXT NOT NULL,
            is_active          INTEGER NOT NULL DEFAULT 1,
            created_at         TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at         TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (base_gl_number) REFERENCES accounts(gl_number)
        );
        """
    )


def get_app_setting(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Retrieve the value for a given app-wide setting key, or None if not set."""
    cur = conn.execute(
        "SELECT value FROM app_settings WHERE key = ?",
        (key,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return row["value"]


def set_app_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Set or update an app-wide setting key to a given string value."""
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO app_settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                          updated_at = datetime('now')
            """,
            (key, value),
        )

        # Log the setting change (without persisting sensitive values).
        log_activity(
            conn,
            event_type="SETTING_CHANGED",
            entity_type="SETTING",
            entity_id=None,
            summary=f"Setting '{key}' updated",
            details={"key": key},
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise

def get_streamlined_manual_txns_enabled(conn: sqlite3.Connection) -> bool:
    """Return True if streamlined manual transactions are enabled."""
    val = get_app_setting(conn, "streamlined_manual_txns_enabled")
    return str(val or "").strip() == "1"


def set_streamlined_manual_txns_enabled(conn: sqlite3.Connection, enabled: bool) -> None:
    """Enable/disable streamlined manual transactions."""
    set_app_setting(conn, "streamlined_manual_txns_enabled", "1" if enabled else "0")

def _user_upload_mapping_from_row(row: sqlite3.Row) -> UserUploadMapping:
    return UserUploadMapping(
        id=int(row["id"]),
        name=str(row["name"]),
        base_gl_number=int(row["base_gl_number"]),
        positive_is_debit=bool(row["positive_is_debit"]),
        effective_date_col=str(row["effective_date_col"]),
        amount_col=str(row["amount_col"]),
        description_col=str(row["description_col"]),
        is_active=bool(row["is_active"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def create_user_upload_mapping(
    conn: sqlite3.Connection,
    name: str,
    base_gl_number: int,
    positive_is_debit: bool,
    effective_date_col: str,
    amount_col: str,
    description_col: str,
) -> int:
    nm = (name or "").strip()
    if not nm:
        raise ValueError("Mapping name cannot be empty.")

    # Validate base GL exists
    acct = get_account_by_gl_number(conn, int(base_gl_number))
    if acct is None:
        raise ValueError(f"Base GL {base_gl_number} does not exist.")

    edc = (effective_date_col or "").strip()
    amc = (amount_col or "").strip()
    dsc = (description_col or "").strip()
    if not edc or not amc or not dsc:
        raise ValueError("Effective Date, Amount, and Description column names are required.")

    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO user_upload_mappings
                (name, base_gl_number, positive_is_debit, effective_date_col, amount_col, description_col, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (nm, int(base_gl_number), 1 if positive_is_debit else 0, edc, amc, dsc),
        )
        mapping_id = int(cur.lastrowid)

        log_activity(
            conn,
            event_type="USER_UPLOAD_MAPPING_CREATED",
            entity_type="USER_UPLOAD_MAPPING",
            entity_id=mapping_id,
            summary=f"Created upload mapping '{nm}'",
            details={
                "mapping_id": mapping_id,
                "name": nm,
                "base_gl_number": int(base_gl_number),
                "positive_is_debit": bool(positive_is_debit),
                "effective_date_col": edc,
                "amount_col": amc,
                "description_col": dsc,
            },
        )

        conn.commit()
        return mapping_id
    except sqlite3.IntegrityError:
        conn.rollback()
        raise ValueError(f"A mapping named '{nm}' already exists.")
    except Exception:
        conn.rollback()
        raise


def get_user_upload_mappings(
    conn: sqlite3.Connection,
    include_inactive: bool = False,
) -> List[UserUploadMapping]:
    query = "SELECT * FROM user_upload_mappings"
    if not include_inactive:
        query += " WHERE is_active = 1"
    query += " ORDER BY LOWER(name) ASC, id ASC"
    cur = conn.execute(query)
    return [_user_upload_mapping_from_row(r) for r in cur.fetchall()]


def get_user_upload_mapping_by_name(
    conn: sqlite3.Connection,
    name: str,
) -> Optional[UserUploadMapping]:
    nm = (name or "").strip()
    if not nm:
        return None
    cur = conn.execute("SELECT * FROM user_upload_mappings WHERE name = ?", (nm,))
    row = cur.fetchone()
    if row is None:
        return None
    return _user_upload_mapping_from_row(row)


def set_user_upload_mapping_active(
    conn: sqlite3.Connection,
    mapping_id: int,
    is_active: bool,
) -> None:
    cur = conn.cursor()

    row = conn.execute(
        "SELECT name FROM user_upload_mappings WHERE id = ?",
        (int(mapping_id),),
    ).fetchone()
    mapping_name = str(row["name"]) if row is not None else None

    try:
        cur.execute(
            """
            UPDATE user_upload_mappings
            SET is_active = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (1 if is_active else 0, int(mapping_id)),
        )

        log_activity(
            conn,
            event_type="USER_UPLOAD_MAPPING_ACTIVATED" if is_active else "USER_UPLOAD_MAPPING_ARCHIVED",
            entity_type="USER_UPLOAD_MAPPING",
            entity_id=int(mapping_id),
            summary=(
                f"{'Unarchived' if is_active else 'Archived'} upload mapping"
                + (f" '{mapping_name}'" if mapping_name else "")
            ),
            details={"mapping_id": int(mapping_id), "is_active": bool(is_active), "name": mapping_name},
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_retained_earnings_gl_number(conn: sqlite3.Connection) -> Optional[int]:
    """Return the GL number configured as the retained earnings account, if any."""
    val = get_app_setting(conn, "retained_earnings_gl_number")
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def set_retained_earnings_gl_number(conn: sqlite3.Connection, gl_number: int) -> None:
    """Configure which GL number is used as the retained earnings equity account.

    Validates that the GL exists and is of type EQUITY before saving.
    """
    acct = get_account_by_gl_number(conn, gl_number)
    if acct is None:
        raise ValueError(f"Account with GL number {gl_number} does not exist.")
    if acct.type is not AccountType.EQUITY:
        raise ValueError(
            f"Account {gl_number} is type {acct.type.value}, not EQUITY. "
            "Retained earnings must be an EQUITY account."
        )
    set_app_setting(conn, "retained_earnings_gl_number", str(gl_number))


# --- Operational GL account settings helpers ---

def get_suspense_gl_number(conn: sqlite3.Connection) -> Optional[int]:
    """Return the GL number configured as the Suspense account, if any.

    This account is used for temporarily holding unapplied/uncategorized activity.
    """
    val = get_app_setting(conn, "suspense_gl_number")
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def set_suspense_gl_number(conn: sqlite3.Connection, gl_number: Optional[int]) -> None:
    """Configure which GL number is used as the Suspense account.

    Rules:
    - If gl_number is None, the setting is cleared (not configured).
    - Otherwise validates that the GL exists and is of type ASSET or LIABILITY.
    """
    key = "suspense_gl_number"

    if gl_number is None:
        cur = conn.cursor()
        try:
            cur.execute(
                "DELETE FROM app_settings WHERE key = ?",
                (key,),
            )

            log_activity(
                conn,
                event_type="SETTING_CHANGED",
                entity_type="SETTING",
                entity_id=None,
                summary=f"Setting '{key}' cleared",
                details={"key": key, "cleared": True},
            )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return

    acct = get_account_by_gl_number(conn, int(gl_number))
    if acct is None:
        raise ValueError(f"Account with GL number {gl_number} does not exist.")
    if acct.type not in (AccountType.ASSET, AccountType.LIABILITY):
        raise ValueError(
            f"Account {gl_number} is type {acct.type.value}, not ASSET or LIABILITY. "
            "Suspense must be an ASSET or LIABILITY account."
        )

    set_app_setting(conn, key, str(int(gl_number)))


# --- Suspense settlement helpers ---

def get_suspense_last_settled_je_id(conn: sqlite3.Connection) -> Optional[int]:
    """Return the last JE id cutoff used for the Suspense "new since settlement" filter."""
    val = get_app_setting(conn, "suspense_last_settled_je_id")
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def set_suspense_last_settled_je_id(conn: sqlite3.Connection, je_id: int) -> None:
    """Persist the suspense settlement cutoff JE id."""
    set_app_setting(conn, "suspense_last_settled_je_id", str(int(je_id)))


def settle_suspense_now(conn: sqlite3.Connection) -> int:
    """Record a suspense settlement cutoff so the UI can filter to "new" suspense activity.

    Rules:
    - Uses the Suspense GL configured in Preferences (operational accounts).
    - Only allowed when the configured suspense account has a net balance of 0.
    - Stores the current MAX(journal_entries.id) as the cutoff.

    Returns:
        cutoff_je_id (int)
    """
    suspense_gl = get_suspense_gl_number(conn)
    if suspense_gl is None:
        raise ValueError(
            "Suspense account is not configured. Set it in Preferences (Operational Accounts) before settling suspense."
        )

    suspense_acct = get_account_by_gl_number(conn, int(suspense_gl))
    if suspense_acct is None or suspense_acct.id is None:
        raise ValueError(
            "Suspense account is not valid or is missing an internal id. Please re-select a valid suspense account in Preferences."
        )

    # Check that suspense is fully cleared (net balance is zero across all dates).
    far_future = "9999-12-31"
    bal_dc_cents = get_account_balance_as_of(conn, suspense_acct.id, far_future)
    if int(bal_dc_cents) != 0:
        raise ValueError(
            "Cannot settle suspense because the suspense account balance is not $0.00. Clear suspense to zero before settling."
        )

    # Cutoff is the current max JE id (or 0 if none exist yet).
    cur = conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM journal_entries")
    row = cur.fetchone()
    cutoff_id = int(row["max_id"]) if row is not None else 0

    set_suspense_last_settled_je_id(conn, cutoff_id)

    # Additional explicit audit log (settings table already logs a SETTING_CHANGED event).
    log_activity(
        conn,
        event_type="SUSPENSE_SETTLED",
        entity_type="SETTING",
        entity_id=None,
        summary=f"Suspense settled through JE id {cutoff_id}",
        details={"cutoff_je_id": cutoff_id, "suspense_gl_number": int(suspense_gl)},
    )
    conn.commit()
    return cutoff_id


def get_transfers_clearing_gl_number(conn: sqlite3.Connection) -> Optional[int]:
    """Return the GL number configured as the Transfers Clearing account, if any.

    This account is used as a clearing account for internal transfers between
    a user's own accounts.
    """
    val = get_app_setting(conn, "transfers_clearing_gl_number")
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def set_transfers_clearing_gl_number(conn: sqlite3.Connection, gl_number: Optional[int]) -> None:
    """Configure which GL number is used as the Transfers Clearing account.

    Rules:
    - If gl_number is None, the setting is cleared (not configured).
    - Otherwise validates that the GL exists and is of type ASSET or LIABILITY.
    """
    key = "transfers_clearing_gl_number"

    if gl_number is None:
        cur = conn.cursor()
        try:
            cur.execute(
                "DELETE FROM app_settings WHERE key = ?",
                (key,),
            )

            log_activity(
                conn,
                event_type="SETTING_CHANGED",
                entity_type="SETTING",
                entity_id=None,
                summary=f"Setting '{key}' cleared",
                details={"key": key, "cleared": True},
            )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return

    acct = get_account_by_gl_number(conn, int(gl_number))
    if acct is None:
        raise ValueError(f"Account with GL number {gl_number} does not exist.")
    if acct.type not in (AccountType.ASSET, AccountType.LIABILITY):
        raise ValueError(
            f"Account {gl_number} is type {acct.type.value}, not ASSET or LIABILITY. "
            "Transfers Clearing must be an ASSET or LIABILITY account."
        )

    set_app_setting(conn, key, str(int(gl_number)))


def _manual_txn_template_from_row(row: sqlite3.Row) -> ManualTxnTemplate:
    return ManualTxnTemplate(
        id=int(row["id"]),
        name=str(row["name"]),
        base_gl_number=int(row["base_gl_number"]),
        base_side=str(row["base_side"]),
        is_active=bool(row["is_active"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )

def create_manual_txn_template(
    conn: sqlite3.Connection,
    name: str,
    base_gl_number: int,
    base_side: str,
) -> int:
    """Create a new streamlined manual transaction template.

    - Enforces unique template names.
    - Validates base_gl_number exists.
    - Validates base_side is DEBIT/CREDIT.
    Returns the new template id.
    """
    nm = (name or "").strip()
    if not nm:
        raise ValueError("Template name cannot be empty.")

    side = (base_side or "").strip().upper()
    if side not in {"DEBIT", "CREDIT"}:
        raise ValueError("Base side must be 'DEBIT' or 'CREDIT'.")

    acct = get_account_by_gl_number(conn, int(base_gl_number))
    if acct is None:
        raise ValueError(f"Base GL {base_gl_number} does not exist.")

    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO manual_txn_templates (name, base_gl_number, base_side, is_active)
            VALUES (?, ?, ?, 1)
            """,
            (nm, int(base_gl_number), side),
        )
        template_id = int(cur.lastrowid)

        log_activity(
            conn,
            event_type="MANUAL_TXN_TEMPLATE_CREATED",
            entity_type="MANUAL_TXN_TEMPLATE",
            entity_id=template_id,
            summary=f"Created manual txn template '{nm}'",
            details={
                "template_id": template_id,
                "name": nm,
                "base_gl_number": int(base_gl_number),
                "base_side": side,
            },
        )

        conn.commit()
        return template_id
    except sqlite3.IntegrityError:
        conn.rollback()
        # UNIQUE(name) violation is the common case
        raise ValueError(f"A template named '{nm}' already exists.")
    except Exception:
        conn.rollback()
        raise

def get_manual_txn_templates(
    conn: sqlite3.Connection,
    include_inactive: bool = False,
) -> List[ManualTxnTemplate]:
    """Return templates ordered by name."""
    params: list[object] = []
    query = "SELECT * FROM manual_txn_templates"
    if not include_inactive:
        query += " WHERE is_active = 1"
    query += " ORDER BY LOWER(name) ASC, id ASC"
    cur = conn.execute(query, params)
    return [_manual_txn_template_from_row(r) for r in cur.fetchall()]

def set_manual_txn_template_active(
    conn: sqlite3.Connection,
    template_id: int,
    is_active: bool,
) -> None:
    """Archive/unarchive a template by toggling is_active."""
    cur = conn.cursor()

    # For logging
    row = conn.execute(
        "SELECT name FROM manual_txn_templates WHERE id = ?",
        (int(template_id),),
    ).fetchone()
    template_name = str(row["name"]) if row is not None else None

    try:
        cur.execute(
            """
            UPDATE manual_txn_templates
            SET is_active = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (1 if is_active else 0, int(template_id)),
        )

        log_activity(
            conn,
            event_type="MANUAL_TXN_TEMPLATE_ACTIVATED" if is_active else "MANUAL_TXN_TEMPLATE_ARCHIVED",
            entity_type="MANUAL_TXN_TEMPLATE",
            entity_id=int(template_id),
            summary=(
                f"{'Unarchived' if is_active else 'Archived'} manual txn template"
                + (f" '{template_name}'" if template_name else "")
            ),
            details={"template_id": int(template_id), "is_active": bool(is_active), "name": template_name},
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise

def get_manual_txn_template_by_name(
    conn: sqlite3.Connection,
    name: str,
) -> Optional[ManualTxnTemplate]:
    nm = (name or "").strip()
    if not nm:
        return None
    cur = conn.execute(
        "SELECT * FROM manual_txn_templates WHERE name = ?",
        (nm,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return _manual_txn_template_from_row(row)

def get_last_closed_period_end(conn: sqlite3.Connection) -> Optional[str]:
    """
    Return the last closed period end date as an ISO string (YYYY-MM-DD),
    or None if not configured.
    """
    return get_app_setting(conn, "last_closed_period_end")


def set_last_closed_period_end(conn: sqlite3.Connection, period_end: str) -> None:
    """
    Set or update the last closed period end date as an ISO string (YYYY-MM-DD).
    """
    set_app_setting(conn, "last_closed_period_end", period_end)


def get_override_pin(conn: sqlite3.Connection) -> Optional[str]:
    """
    Return the override PIN string used for posting into closed periods,
    or None if not configured.
    """
    return get_app_setting(conn, "override_pin")


def set_override_pin(conn: sqlite3.Connection, pin: str) -> None:
    """
    Set or update the override PIN string used for posting into closed periods.
    """

    set_app_setting(conn, "override_pin", pin)


# --- Financial statements start date setting helpers ---

def _validate_month_end_iso(date_str: str) -> str:
    """Validate that date_str is an ISO date (YYYY-MM-DD) and is a month-end.

    Returns the normalized ISO string if valid.

    Raises ValueError with a user-friendly message if invalid.
    """
    s = (date_str or "").strip()
    try:
        d = datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        raise ValueError(
            "Financials start date must be in YYYY-MM-DD format and must be a month-end date."
        )

    # Compute last day of the month
    if d.month == 12:
        next_month = date(d.year + 1, 1, 1)
    else:
        next_month = date(d.year, d.month + 1, 1)
    last_day = next_month - timedelta(days=1)

    if d != last_day:
        raise ValueError(
            "Financials start date must be a month-end date (e.g., 2025-10-31 or 2025-11-30)."
        )

    return d.isoformat()


def get_financials_start_date(conn: sqlite3.Connection) -> Optional[str]:
    """Return the configured financial statements start date (month-end ISO), or None.

    If the stored value is invalid for any reason, returns None rather than raising,
    so the UI can fall back to a default.
    """
    val = get_app_setting(conn, "financials_start_date")
    if not val:
        return None
    try:
        return _validate_month_end_iso(val)
    except Exception:
        return None


def set_financials_start_date(conn: sqlite3.Connection, start_date: str) -> None:
    """Configure the start month-end date for historical financial statements.

    - Requires ISO format (YYYY-MM-DD)
    - Requires the date to be the last calendar day of its month

    Raises ValueError with a user-friendly message if invalid.
    """

    iso = _validate_month_end_iso(start_date)
    set_app_setting(conn, "financials_start_date", iso)


# --- Month-over-month (month-end) financials helpers ---

def _month_end(d: date) -> date:
    """Return the last calendar day of the month for the given date."""
    if d.month == 12:
        next_month_first = date(d.year + 1, 1, 1)
    else:
        next_month_first = date(d.year, d.month + 1, 1)
    return next_month_first - timedelta(days=1)


def _iter_month_ends(start_month_end_iso: str, end_month_end_iso: str) -> List[str]:
    """Return a list of ISO month-end dates from start through end (inclusive).

    Both inputs must already be validated as ISO month-end dates.
    """
    start_d = datetime.strptime(start_month_end_iso, "%Y-%m-%d").date()
    end_d = datetime.strptime(end_month_end_iso, "%Y-%m-%d").date()

    month_ends: List[str] = []
    cur = start_d
    while cur <= end_d:
        month_ends.append(cur.isoformat())
        # Advance to next month-end
        if cur.month == 12:
            next_month_first = date(cur.year + 1, 1, 1)
        else:
            next_month_first = date(cur.year, cur.month + 1, 1)
        cur = _month_end(next_month_first)

    return month_ends


def _display_balance_cents(acct: Account, balance_debits_minus_credits_cents: int) -> int:
    """Convert debits-minus-credits to a positive-normal display balance.

    - ASSET/EXPENSE: debits - credits (positive is normal)
    - LIABILITY/EQUITY/INCOME: credits - debits (positive is normal)

    This matches the sign convention used throughout the UI financial statements.
    """
    if acct.type in {AccountType.LIABILITY, AccountType.EQUITY, AccountType.INCOME}:
        return -int(balance_debits_minus_credits_cents)
    return int(balance_debits_minus_credits_cents)


# --- Time-series helpers (for charts in Financial Statements) ---

def _parse_iso_ymd(date_iso: str) -> date:
    """Parse an ISO date string (YYYY-MM-DD) into a `date`."""
    return datetime.strptime(str(date_iso).strip(), "%Y-%m-%d").date()


def get_balance_as_of_for_gl_numbers(
    conn: sqlite3.Connection,
    gl_numbers: List[int],
    as_of_date: str,
) -> int:
    """Return aggregated *display* balance (in cents) for the given GL numbers as of as_of_date (inclusive).

    Display balance convention matches the UI:
      - ASSET/EXPENSE: debits - credits
      - LIABILITY/EQUITY/INCOME: credits - debits

    Uses journal_entries.effective_date and excludes inactive (archived) accounts.
    """
    gls = [int(g) for g in (gl_numbers or [])]
    if not gls:
        return 0

    # Validate date (raises if invalid)
    _ = _parse_iso_ymd(as_of_date)

    placeholders = ",".join(["?"] * len(gls))
    params: List[object] = []
    params.extend(gls)
    params.append(as_of_date)

    cur = conn.execute(
        f"""
        SELECT
            COALESCE(
                SUM(
                    CASE
                        WHEN a.type IN ('ASSET','EXPENSE') THEN
                            CASE WHEN jl.is_debit = 1 THEN jl.amount_cents ELSE -jl.amount_cents END
                        ELSE
                            CASE WHEN jl.is_debit = 0 THEN jl.amount_cents ELSE -jl.amount_cents END
                    END
                ),
                0
            ) AS balance_cents
        FROM journal_lines jl
        JOIN journal_entries je ON jl.journal_entry_id = je.id
        JOIN accounts a ON jl.account_id = a.id
        WHERE a.is_active = 1
          AND a.gl_number IN ({placeholders})
          AND je.effective_date <= ?
        """,
        params,
    )
    row = cur.fetchone()
    if row is None:
        return 0
    return int(row["balance_cents"])


def get_daily_net_changes_for_gl_numbers(
    conn: sqlite3.Connection,
    gl_numbers: List[int],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Return daily net changes (display convention, cents) for the given GL numbers.

    Output columns:
      - effective_date (ISO YYYY-MM-DD)
      - net_change_cents (int)

    Uses journal_entries.effective_date and excludes inactive (archived) accounts.
    """
    gls = [int(g) for g in (gl_numbers or [])]
    if not gls:
        return pd.DataFrame(columns=["effective_date", "net_change_cents"])

    start_d = _parse_iso_ymd(start_date)
    end_d = _parse_iso_ymd(end_date)
    if start_d > end_d:
        raise ValueError("start_date must be on or before end_date")

    placeholders = ",".join(["?"] * len(gls))
    params: List[object] = []
    params.extend(gls)
    params.extend([start_date, end_date])

    cur = conn.execute(
        f"""
        SELECT
            je.effective_date AS effective_date,
            COALESCE(
                SUM(
                    CASE
                        WHEN a.type IN ('ASSET','EXPENSE') THEN
                            CASE WHEN jl.is_debit = 1 THEN jl.amount_cents ELSE -jl.amount_cents END
                        ELSE
                            CASE WHEN jl.is_debit = 0 THEN jl.amount_cents ELSE -jl.amount_cents END
                    END
                ),
                0
            ) AS net_change_cents
        FROM journal_lines jl
        JOIN journal_entries je ON jl.journal_entry_id = je.id
        JOIN accounts a ON jl.account_id = a.id
        WHERE a.is_active = 1
          AND a.gl_number IN ({placeholders})
          AND je.effective_date >= ?
          AND je.effective_date <= ?
        GROUP BY je.effective_date
        ORDER BY je.effective_date ASC
        """,
        params,
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["effective_date", "net_change_cents"])

    return pd.DataFrame(
        [{"effective_date": str(r["effective_date"]), "net_change_cents": int(r["net_change_cents"])} for r in rows]
    )


def compute_balance_time_series(
    conn: sqlite3.Connection,
    gl_numbers: List[int],
    start_date: str,
    end_date: str,
    frequency: str = "DAILY",
) -> pd.DataFrame:
    """Compute a balance time series (effective-date basis) for the provided GL numbers.

    Args:
        gl_numbers: list of GL numbers to aggregate.
        start_date/end_date: ISO dates (YYYY-MM-DD), inclusive.
        frequency: "DAILY" or "MONTHLY".

    Returns:
        DataFrame with columns:
          - date (datetime64)
          - balance_cents (int)
          - delta_cents (int)  # period-over-period change

    Notes:
        - DAILY mode builds balances efficiently from an opening balance + daily net changes.
        - MONTHLY mode uses strict calendar month-end dates and computes balances at each month-end.
        - Inactive (archived) accounts are excluded by design.
    """
    freq = (frequency or "").strip().upper()
    if freq not in {"DAILY", "MONTHLY"}:
        raise ValueError("frequency must be 'DAILY' or 'MONTHLY'")

    gls = [int(g) for g in (gl_numbers or [])]
    if not gls:
        return pd.DataFrame(columns=["date", "balance_cents", "delta_cents"])

    start_d = _parse_iso_ymd(start_date)
    end_d = _parse_iso_ymd(end_date)
    if start_d > end_d:
        raise ValueError("start_date must be on or before end_date")

    if freq == "MONTHLY":
        # Strict month-end validation
        start_iso = _validate_month_end_iso(start_date)
        end_iso = _validate_month_end_iso(end_date)

        month_ends = _iter_month_ends(start_iso, end_iso)
        balances: List[int] = []
        for m_end in month_ends:
            balances.append(get_balance_as_of_for_gl_numbers(conn, gls, m_end))

        dfm = pd.DataFrame({
            "date": pd.to_datetime(month_ends),
            "balance_cents": [int(x) for x in balances],
        })
        dfm["delta_cents"] = dfm["balance_cents"].diff().fillna(0).astype(int)
        return dfm

    # DAILY
    opening_as_of = (start_d - timedelta(days=1)).isoformat()
    opening_balance = get_balance_as_of_for_gl_numbers(conn, gls, opening_as_of)

    changes_df = get_daily_net_changes_for_gl_numbers(conn, gls, start_date, end_date)

    full_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    dfd = pd.DataFrame({"date": full_dates})

    if not changes_df.empty:
        tmp = changes_df.copy()
        tmp["date"] = pd.to_datetime(tmp["effective_date"])
        tmp = tmp[["date", "net_change_cents"]]
        dfd = dfd.merge(tmp, on="date", how="left")
    else:
        dfd["net_change_cents"] = 0

    dfd["net_change_cents"] = dfd["net_change_cents"].fillna(0).astype(int)
    dfd["balance_cents"] = (opening_balance + dfd["net_change_cents"].cumsum()).astype(int)
    dfd["delta_cents"] = dfd["balance_cents"].diff().fillna(0).astype(int)

    return dfd[["date", "balance_cents", "delta_cents"]]


def compute_mom_financials_month_end(
    conn: sqlite3.Connection,
    as_of_month_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute a month-over-month financial snapshot report at month-end dates.

    Rules:
    - Requires a configured financials start date (Preferences -> Set Financials Start Date).
    - Requires `as_of_month_end` to be a month-end ISO date (YYYY-MM-DD).

    Returns:
        (bs_df, is_df, ratios_df)

    Each DataFrame has month-end dates as columns.

    Notes:
    - Uses ending balances as-of each month-end date (inclusive).
    - Balance Sheet equity totals include a "Current Earnings (Net Income)" line
      so that Total Assets == Total Liabilities + Total Equity for each column.
    - Includes ALL line items (all active accounts) so columns stay aligned.
    """
    # Validate required settings
    start_iso = get_financials_start_date(conn)
    if not start_iso:
        raise ValueError(
            "Financials start date is not set. Set it in Preferences (Set Financials Start Date) before generating MoM financials."
        )

    # Validate the as-of date is a month-end
    try:
        end_iso = _validate_month_end_iso(as_of_month_end)
    except ValueError:
        raise ValueError(
            "As-of date for MoM financials must be a month-end date (e.g., 2025-11-30)."
        )

    # Build month-end columns
    month_ends = _iter_month_ends(start_iso, end_iso)

    # Pull all active accounts once, grouped for stable ordering
    accounts = sorted(get_accounts(conn, include_inactive=False), key=lambda a: a.gl_number)

    asset_accounts = [a for a in accounts if a.type is AccountType.ASSET]
    liability_accounts = [a for a in accounts if a.type is AccountType.LIABILITY]
    equity_accounts = [a for a in accounts if a.type is AccountType.EQUITY]
    income_accounts = [a for a in accounts if a.type is AccountType.INCOME]
    expense_accounts = [a for a in accounts if a.type is AccountType.EXPENSE]

    def _acct_label(a: Account) -> str:
        return f"{a.gl_number} - {a.name}"

    # Row sets (union across all months by construction)
    bs_row_order: List[str] = []
    bs_row_order.append("ASSETS")
    bs_row_order.extend([_acct_label(a) for a in asset_accounts])
    bs_row_order.append("Total Assets")
    bs_row_order.append("LIABILITIES")
    bs_row_order.extend([_acct_label(a) for a in liability_accounts])
    bs_row_order.append("Total Liabilities")
    bs_row_order.append("EQUITY")
    bs_row_order.extend([_acct_label(a) for a in equity_accounts])
    bs_row_order.append("Current Earnings (Net Income)")
    bs_row_order.append("Total Equity")
    bs_row_order.append("Total Liabilities & Equity")
    bs_row_order.append("Net Worth")

    is_row_order: List[str] = []
    is_row_order.append("INCOME")
    is_row_order.extend([_acct_label(a) for a in income_accounts])
    is_row_order.append("Total Income")
    is_row_order.append("EXPENSES")
    is_row_order.extend([_acct_label(a) for a in expense_accounts])
    is_row_order.append("Total Expenses")
    is_row_order.append("Net Income")

    ratio_row_order: List[str] = [
        "Debt-to-Assets",
        "Debt-to-Equity",
        "Net Margin",
    ]

    # Initialize storage
    bs_data: dict[str, List[object]] = {k: [] for k in bs_row_order}
    is_data: dict[str, List[object]] = {k: [] for k in is_row_order}
    ratios_data: dict[str, List[object]] = {k: [] for k in ratio_row_order}

    for m_end in month_ends:
        # Section headers show as blank rows in numeric columns
        bs_data["ASSETS"].append(None)
        bs_data["LIABILITIES"].append(None)
        bs_data["EQUITY"].append(None)

        is_data["INCOME"].append(None)
        is_data["EXPENSES"].append(None)

        # Compute all balances for this month-end
        total_assets_cents = 0
        total_liabilities_cents = 0
        total_equity_base_cents = 0

        total_income_cents = 0
        total_expenses_cents = 0

        # Balance Sheet line items
        for acct in asset_accounts:
            if acct.id is None:
                bal_disp = 0
            else:
                bal_dc = get_account_balance_as_of(conn, acct.id, m_end)
                bal_disp = _display_balance_cents(acct, bal_dc)
            bs_data[_acct_label(acct)].append(bal_disp)
            total_assets_cents += bal_disp

        for acct in liability_accounts:
            if acct.id is None:
                bal_disp = 0
            else:
                bal_dc = get_account_balance_as_of(conn, acct.id, m_end)
                bal_disp = _display_balance_cents(acct, bal_dc)
            bs_data[_acct_label(acct)].append(bal_disp)
            total_liabilities_cents += bal_disp

        for acct in equity_accounts:
            if acct.id is None:
                bal_disp = 0
            else:
                bal_dc = get_account_balance_as_of(conn, acct.id, m_end)
                bal_disp = _display_balance_cents(acct, bal_dc)
            bs_data[_acct_label(acct)].append(bal_disp)
            total_equity_base_cents += bal_disp

        # Income Statement line items (ending balances as-of month-end)
        for acct in income_accounts:
            if acct.id is None:
                bal_disp = 0
            else:
                bal_dc = get_account_balance_as_of(conn, acct.id, m_end)
                bal_disp = _display_balance_cents(acct, bal_dc)
            is_data[_acct_label(acct)].append(bal_disp)
            total_income_cents += bal_disp

        for acct in expense_accounts:
            if acct.id is None:
                bal_disp = 0
            else:
                bal_dc = get_account_balance_as_of(conn, acct.id, m_end)
                bal_disp = _display_balance_cents(acct, bal_dc)
            is_data[_acct_label(acct)].append(bal_disp)
            total_expenses_cents += bal_disp

        net_income_cents = total_income_cents - total_expenses_cents

        # Balance Sheet totals, with Current Earnings included in Total Equity
        total_equity_inclusive_cents = total_equity_base_cents + net_income_cents
        total_liab_equity_cents = total_liabilities_cents + total_equity_inclusive_cents
        net_worth_cents = total_assets_cents - total_liabilities_cents

        bs_data["Total Assets"].append(total_assets_cents)
        bs_data["Total Liabilities"].append(total_liabilities_cents)
        bs_data["Current Earnings (Net Income)"].append(net_income_cents)
        bs_data["Total Equity"].append(total_equity_inclusive_cents)
        bs_data["Total Liabilities & Equity"].append(total_liab_equity_cents)
        bs_data["Net Worth"].append(net_worth_cents)

        is_data["Total Income"].append(total_income_cents)
        is_data["Total Expenses"].append(total_expenses_cents)
        is_data["Net Income"].append(net_income_cents)

        # Ratios
        if total_assets_cents != 0:
            ratios_data["Debt-to-Assets"].append(total_liabilities_cents / total_assets_cents)
        else:
            ratios_data["Debt-to-Assets"].append(None)

        if total_equity_inclusive_cents != 0:
            ratios_data["Debt-to-Equity"].append(
                total_liabilities_cents / total_equity_inclusive_cents
            )
        else:
            ratios_data["Debt-to-Equity"].append(None)

        if total_income_cents != 0:
            ratios_data["Net Margin"].append(net_income_cents / total_income_cents)
        else:
            ratios_data["Net Margin"].append(None)

    # Build DataFrames (rows are line items; columns are month-end dates)
    bs_df = pd.DataFrame(bs_data, index=month_ends).T
    is_df = pd.DataFrame(is_data, index=month_ends).T
    ratios_df = pd.DataFrame(ratios_data, index=month_ends).T

    return bs_df, is_df, ratios_df

def get_app_access_password(conn: sqlite3.Connection) -> Optional[str]:
    """
    Return the app access password used for gating the UI,
    or None if not configured.

    If this returns None, the app should NOT require a password
    at startup (i.e., anyone can access once the app is running).
    """
    return get_app_setting(conn, "app_access_password")


def set_app_access_password(conn: sqlite3.Connection, password: Optional[str]) -> None:
    """
    Set, update, or clear the app access password.

    - If password is a non-empty string, it is stored as-is.
    - If password is None or empty/whitespace, the setting is removed,
      meaning no password will be required at startup.
    """
    pw = (password or "").strip()

    cur = conn.cursor()
    try:
        if not pw:
            # Clear the password (no row means "no password set")
            cur.execute(
                "DELETE FROM app_settings WHERE key = ?",
                ("app_access_password",),
            )

            log_activity(
                conn,
                event_type="APP_PASSWORD_CLEARED",
                entity_type="SETTING",
                entity_id=None,
                summary="App access password cleared",
                details={"key": "app_access_password"},
            )
        else:
            # Upsert via app_settings helper
            cur.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                              updated_at = datetime('now')
                """,
                ("app_access_password", pw),
            )

            log_activity(
                conn,
                event_type="APP_PASSWORD_SET",
                entity_type="SETTING",
                entity_id=None,
                summary="App access password set or updated",
                details={"key": "app_access_password"},
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise

# --- Transaction Keyword Rule Helpers ---

def _txn_keyword_rule_from_row(row: sqlite3.Row) -> TxnKeywordRule:
    """Private helper to construct a TxnKeywordRule from a DB row."""
    return TxnKeywordRule(
        id=row["id"],
        keyword=row["keyword"],
        gl_number=int(row["gl_number"]),
        is_active=bool(row["is_active"]),
        priority=int(row["priority"]),
    )


def upsert_txn_keyword_rule(
    conn: sqlite3.Connection,
    keyword: str,
    gl_number: int,
    is_active: bool = True,
    priority: int = 0,
) -> int:
    """Create or update a transaction keyword rule.

    - Validates that keyword is non-empty.
    - Validates that gl_number exists as an account.
    - If a rule with the same keyword already exists, it is updated.
    - Returns the rule's id.
    """
    kw = (keyword or "").strip()
    if not kw:
        raise ValueError("Keyword cannot be empty.")

    # Validate GL exists
    acct = get_account_by_gl_number(conn, gl_number)
    if acct is None:
        raise ValueError(f"Account with GL number {gl_number} does not exist.")

    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO txn_keyword_rules (keyword, gl_number, is_active, priority)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(keyword) DO UPDATE SET
                gl_number = excluded.gl_number,
                is_active = excluded.is_active,
                priority = excluded.priority,
                updated_at = datetime('now')
            """,
            (kw, gl_number, 1 if is_active else 0, int(priority)),
        )

        # Fetch the id for the (now upserted) rule
        cur2 = conn.execute(
            "SELECT id FROM txn_keyword_rules WHERE keyword = ?",
            (kw,),
        )
        row = cur2.fetchone()
        rule_id = int(row["id"]) if row is not None else 0

        log_activity(
            conn,
            event_type="TXN_KEYWORD_RULE_UPSERTED",
            entity_type="TXN_KEYWORD_RULE",
            entity_id=rule_id or None,
            summary=f"Keyword rule for '{kw}' upserted",
            details={
                "rule_id": rule_id,
                "keyword": kw,
                "gl_number": gl_number,
                "is_active": bool(is_active),
                "priority": int(priority),
            },
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return rule_id


def get_txn_keyword_rules(
    conn: sqlite3.Connection,
    include_inactive: bool = False,
) -> List[TxnKeywordRule]:
    """Return all transaction keyword rules, ordered by priority desc then id.

    If include_inactive is False, only active rules are returned.
    """
    where_clauses = []
    params: list[Any] = []

    if not include_inactive:
        where_clauses.append("is_active = 1")

    query = "SELECT * FROM txn_keyword_rules"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY priority DESC, id ASC"

    cur = conn.execute(query, params)
    rows = cur.fetchall()
    return [_txn_keyword_rule_from_row(r) for r in rows]


def delete_txn_keyword_rule(conn: sqlite3.Connection, rule_id: int) -> None:
    """Delete a keyword rule by id."""
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM txn_keyword_rules WHERE id = ?",
            (rule_id,),
        )

        log_activity(
            conn,
            event_type="TXN_KEYWORD_RULE_DELETED",
            entity_type="TXN_KEYWORD_RULE",
            entity_id=rule_id,
            summary=f"Deleted transaction keyword rule id={rule_id}",
            details={"rule_id": rule_id},
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise


def find_matching_keyword_rule(
    conn: sqlite3.Connection,
    description: str,
) -> Optional[TxnKeywordRule]:
    """Return the first active keyword rule whose keyword appears in description.

    - Matching is case-insensitive.
    - Rules are evaluated in order of priority (desc), then id (asc).
    - Returns None if there is no match.

    This is intended to be used when staging imported transactions: given a
    transaction description, you can call this to get the suggested offset GL.
    """
    desc = (description or "").lower()
    if not desc:
        return None

    rules = get_txn_keyword_rules(conn, include_inactive=False)
    for rule in rules:
        if rule.keyword.lower() in desc:
            return rule
    return None



# --- Notes tables and helpers ---

def init_notes_table(conn: sqlite3.Connection) -> None:
    """Create the notes table if it does not already exist.

    Notes are never physically deleted; they can be archived to hide them
    from the UI while retaining history.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            text        TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            archived_at TEXT,
            is_archived INTEGER NOT NULL DEFAULT 0
        );
        """
    )


def init_note_gl_links_table(conn: sqlite3.Connection) -> None:
    """Create the note_gl_links table if it does not already exist.

    This provides a many-to-many mapping between notes and GL numbers so a
    note can be linked to zero, one, or many GLs.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS note_gl_links (
            note_id    INTEGER NOT NULL,
            gl_number  INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (note_id, gl_number),
            FOREIGN KEY (note_id)   REFERENCES notes(id)           ON DELETE CASCADE,
            FOREIGN KEY (gl_number) REFERENCES accounts(gl_number)
        );
        """
    )


def create_note(
    conn: sqlite3.Connection,
    text: str,
    gl_numbers: List[int],
) -> int:
    """Create a new note and link it to the provided GL numbers.

    - text: required non-empty string.
    - gl_numbers: list of GL numbers to associate with this note. May be empty.

    Returns the newly created note's id.
    """
    note_text = (text or "").strip()
    if not note_text:
        raise ValueError("Note text cannot be empty.")

    # Normalize GL list: deduplicate and sort, but do not perform additional
    # validation here since GL selection will be driven by the UI.
    unique_gls = sorted({int(gl) for gl in gl_numbers}) if gl_numbers else []

    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO notes (text)
            VALUES (?);
            """,
            (note_text,),
        )
        note_id = int(cur.lastrowid)

        if unique_gls:
            rows = [(note_id, gl) for gl in unique_gls]
            cur.executemany(
                """
                INSERT INTO note_gl_links (note_id, gl_number)
                VALUES (?, ?);
                """,
                rows,
            )

        # Log note creation.
        log_activity(
            conn,
            event_type="NOTE_CREATED",
            entity_type="NOTE",
            entity_id=note_id,
            summary=f"Note {note_id} created",
            details={
                "note_id": note_id,
                "gl_numbers": unique_gls,
                "text_preview": note_text[:200],
            },
        )

        conn.commit()
        return note_id
    except Exception:
        conn.rollback()
        raise


def archive_notes(
    conn: sqlite3.Connection,
    note_ids: List[int],
) -> None:
    """Archive the notes with the given ids.

    Archiving sets is_archived = 1 and archived_at to the current timestamp,
    but does not delete any rows. Archived notes are intended to be hidden
    from the UI.
    """
    if not note_ids:
        return

    # Normalize and deduplicate ids
    unique_ids = sorted({int(nid) for nid in note_ids})
    placeholders = ",".join(["?"] * len(unique_ids))

    now_str = datetime.utcnow().isoformat(timespec="seconds")
    cur = conn.cursor()
    try:
        cur.execute(
            f"UPDATE notes SET is_archived = 1, archived_at = ? WHERE id IN ({placeholders})",
            [now_str, *unique_ids],
        )

        log_activity(
            conn,
            event_type="NOTES_ARCHIVED",
            entity_type="NOTE",
            entity_id=None,
            summary=f"Archived {len(unique_ids)} note(s)",
            details={
                "note_ids": unique_ids,
                "archived_at": now_str,
            },
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_active_notes(conn: sqlite3.Connection) -> List[Note]:
    """Return all non-archived notes, including their linked GL numbers.

    Results are ordered by created_at descending, then id descending.
    """
    cur = conn.execute(
        """
        SELECT
            n.id,
            n.text,
            n.created_at,
            n.archived_at,
            n.is_archived,
            GROUP_CONCAT(l.gl_number, ',') AS gl_numbers_csv
        FROM notes n
        LEFT JOIN note_gl_links l ON n.id = l.note_id
        WHERE n.is_archived = 0
        GROUP BY n.id
        ORDER BY n.created_at DESC, n.id DESC
        """
    )
    rows = cur.fetchall()

    notes: List[Note] = []
    for row in rows:
        csv = row["gl_numbers_csv"]
        if csv:
            gl_numbers = [int(part) for part in str(csv).split(",") if part]
        else:
            gl_numbers = []

        notes.append(
            Note(
                id=int(row["id"]),
                text=row["text"],
                created_at=row["created_at"],
                archived_at=row["archived_at"],
                is_archived=bool(row["is_archived"]),
                gl_numbers=gl_numbers,
            )
        )

    return notes


def get_active_notes_for_gl(
    conn: sqlite3.Connection,
    gl_number: int,
) -> List[Note]:
    """Return all non-archived notes linked to a specific GL number.

    Results are ordered by created_at descending, then id descending.
    """
    cur = conn.execute(
        """
        SELECT
            n.id,
            n.text,
            n.created_at,
            n.archived_at,
            n.is_archived,
            GROUP_CONCAT(l.gl_number, ',') AS gl_numbers_csv
        FROM notes n
        JOIN note_gl_links lg
          ON n.id = lg.note_id AND lg.gl_number = ?
        LEFT JOIN note_gl_links l
          ON n.id = l.note_id
        WHERE n.is_archived = 0
        GROUP BY n.id
        ORDER BY n.created_at DESC, n.id DESC
        """,
        (int(gl_number),),
    )
    rows = cur.fetchall()

    notes: List[Note] = []
    for row in rows:
        csv = row["gl_numbers_csv"]
        if csv:
            gl_numbers = [int(part) for part in str(csv).split(",") if part]
        else:
            gl_numbers = []

        notes.append(
            Note(
                id=int(row["id"]),
                text=row["text"],
                created_at=row["created_at"],
                archived_at=row["archived_at"],
                is_archived=bool(row["is_archived"]),
                gl_numbers=gl_numbers,
            )
        )

    return notes

# --- Period closing (automatic net income -> retained earnings) helpers ---

def _compute_period_movement_cents(
    conn: sqlite3.Connection,
    account: Account,
    period_start: str,
    period_end: str,
) -> int:
    """Compute the movement (in cents) for an account over a given period.

    Returns debits - credits over [period_start, period_end].
    """
    opening_cents, history = get_account_history(
        conn,
        account_id=account.id,
        start_date=period_start,
        end_date=period_end,
    )
    if not history:
        return 0
    final_rb_cents = history[-1].running_balance_cents
    return final_rb_cents - opening_cents


def compute_period_close_preview(
    conn: sqlite3.Connection,
    period_start: str,
    period_end: str,
    retained_earnings_gl_number: int,
) -> PeriodClosePreview:
    """Compute the lines needed to close INCOME and EXPENSE accounts for a period.

    Returns a PeriodClosePreview with:
        - lines: list of PeriodCloseLine, including the retained earnings line.
        - total_income_cents: positive for net credits to income accounts.
        - total_expense_cents: positive for net debits to expense accounts.
        - net_income_cents: total_income_cents - total_expense_cents.

    This function only prepares the JE; it does not insert anything into the DB.
    """
    re_acct = get_account_by_gl_number(conn, retained_earnings_gl_number)
    if re_acct is None:
        raise ValueError(f"Retained earnings GL {retained_earnings_gl_number} does not exist.")
    if re_acct.type is not AccountType.EQUITY:
        raise ValueError(
            f"Retained earnings GL {retained_earnings_gl_number} is type {re_acct.type.value}, not EQUITY."
        )

    # Use all active accounts; if you later archive an income/expense account,
    # it is likely because it's no longer used going forward.
    accounts = get_accounts(conn, include_inactive=False)

    income_accounts = [a for a in accounts if a.type is AccountType.INCOME]
    expense_accounts = [a for a in accounts if a.type is AccountType.EXPENSE]

    lines: List[PeriodCloseLine] = []
    total_income_cents = 0
    total_expense_cents = 0

    # Income accounts: use ending balance as-of period_end (inclusive), not period movement.
    # get_account_balance_as_of returns debits - credits.
    for acct in income_accounts:
        if acct.id is None:
            continue

        balance_dc = get_account_balance_as_of(conn, acct.id, period_end)  # debits - credits
        if balance_dc == 0:
            continue

        # Income amount is credits - debits.
        income_cents = -balance_dc
        if income_cents == 0:
            continue

        # To close, post the opposite side to bring the account to zero.
        # If income_cents > 0 (net credit), DEBIT the income account.
        # If income_cents < 0 (net debit / contra), CREDIT the income account.
        if income_cents > 0:
            closing_is_debit = True
            closing_amount_cents = income_cents
        else:
            closing_is_debit = False
            closing_amount_cents = -income_cents

        lines.append(
            PeriodCloseLine(
                account=acct,
                amount_cents=closing_amount_cents,
                is_debit=closing_is_debit,
            )
        )
        total_income_cents += income_cents

    # Expense accounts: use ending balance as-of period_end (inclusive), not period movement.
    # get_account_balance_as_of returns debits - credits.
    for acct in expense_accounts:
        if acct.id is None:
            continue

        balance_dc = get_account_balance_as_of(conn, acct.id, period_end)  # debits - credits
        if balance_dc == 0:
            continue

        # Expense amount is debits - credits.
        expense_cents = balance_dc
        if expense_cents == 0:
            continue

        # To close, post the opposite side to bring the account to zero.
        # If expense_cents > 0 (net debit), CREDIT the expense account.
        # If expense_cents < 0 (net credit / reversal), DEBIT the expense account.
        if expense_cents > 0:
            closing_is_debit = False
            closing_amount_cents = expense_cents
        else:
            closing_is_debit = True
            closing_amount_cents = -expense_cents

        lines.append(
            PeriodCloseLine(
                account=acct,
                amount_cents=closing_amount_cents,
                is_debit=closing_is_debit,
            )
        )
        total_expense_cents += expense_cents

    net_income_cents = total_income_cents - total_expense_cents

    # If there is no income or expense activity, we may end up with zero net income
    # and no lines; in that case, the caller can decide whether to post anything.
    if not lines and net_income_cents == 0:
        return PeriodClosePreview(
            lines=[],
            total_income_cents=0,
            total_expense_cents=0,
            net_income_cents=0,
        )

    # Add the retained earnings line to balance the JE.
    # net_income_cents > 0 => profit => CREDIT retained earnings
    # net_income_cents < 0 => loss => DEBIT retained earnings
    if net_income_cents > 0:
        re_is_debit = False
        re_amount_cents = net_income_cents
    elif net_income_cents < 0:
        re_is_debit = True
        re_amount_cents = -net_income_cents
    else:
        # No net income but nonzero lines (e.g., pure reclassification between
        # income and expense). In this edge case, we don't need a retained earnings line.
        re_is_debit = False
        re_amount_cents = 0

    if re_amount_cents > 0:
        lines.append(
            PeriodCloseLine(
                account=re_acct,
                amount_cents=re_amount_cents,
                is_debit=re_is_debit,
            )
        )

    return PeriodClosePreview(
        lines=lines,
        total_income_cents=total_income_cents,
        total_expense_cents=total_expense_cents,
        net_income_cents=net_income_cents,
    )


def has_period_close_entry(
    conn: sqlite3.Connection,
    period_end: str,
) -> bool:
    """Return True if an *unreversed* period-closing entry exists for the given period_end.

    A period close for a period ending on `period_end` is posted with:
        - source = 'PERIOD_CLOSE'
        - effective_date = day after period_end

    A reversal of that closing entry is posted with:
        - source = 'PERIOD_CLOSE_REV'
        - effective_date = same effective date as the original closing entry

    Because reversed journal entries remain in history, we treat a period close as
    still "existing" only when the number of original PERIOD_CLOSE entries for that
    close effective date is greater than the number of PERIOD_CLOSE_REV entries for
    that same effective date.
    """
    try:
        period_end_date = datetime.strptime(period_end, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid period_end date '{period_end}': {e}") from e

    close_effective_date = (period_end_date + timedelta(days=1)).isoformat()

    cur = conn.execute(
        """
        SELECT
            SUM(CASE WHEN source = 'PERIOD_CLOSE' THEN 1 ELSE 0 END) AS close_cnt,
            SUM(CASE WHEN source = 'PERIOD_CLOSE_REV' THEN 1 ELSE 0 END) AS rev_cnt
        FROM journal_entries
        WHERE effective_date = ?
          AND source IN ('PERIOD_CLOSE', 'PERIOD_CLOSE_REV')
        """,
        (close_effective_date,),
    )
    row = cur.fetchone()

    close_cnt = int(row["close_cnt"] or 0) if row is not None else 0
    rev_cnt = int(row["rev_cnt"] or 0) if row is not None else 0
    return close_cnt > rev_cnt


def post_period_close_entry(
    conn: sqlite3.Connection,
    period_start: str,
    period_end: str,
    retained_earnings_gl_number: int,
    description: Optional[str] = None,
) -> int:
    """Post a period-closing journal entry that zeroes income/expense into retained earnings.

    The income and expense activity is computed over [period_start, period_end],
    but the effective_date of the closing entry is set to the **day after**
    period_end. This allows month-end financial statements (as of period_end)
    to still show that month's income and expenses, while the balances are
    rolled into retained earnings starting on the first day of the next period.

    Safeguards:
        - Raises ValueError if a closing entry already exists for the given period_end.
        - Raises ValueError if there is no income/expense activity in the period.

    Returns:
        The journal_entry_id of the posted closing entry.
    """
    if has_period_close_entry(conn, period_end):
        raise ValueError(
            f"A period-closing entry already exists for {period_end}. "
            "Reverse it before posting another closing entry for this period."
        )

    try:
        period_end_date = datetime.strptime(period_end, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid period_end date '{period_end}': {e}") from e

    checklist_gate = evaluate_period_close_checklist_gate(conn, period_end_date)
    if not bool(checklist_gate.get("allowed", False)):
        incomplete_required = checklist_gate.get("incomplete_required_checklists", []) or []
        incomplete_labels = []
        for item in incomplete_required:
            category = str(item.get("category") or "General").strip()
            checklist_name = str(item.get("checklist_name") or "Unnamed Checklist").strip()
            incomplete_labels.append(f"{category}: {checklist_name}")

        if incomplete_labels:
            checklist_msg = "; ".join(incomplete_labels)
            raise ValueError(
                f"Cannot post a period-closing entry for {period_end} because required checklists are incomplete: {checklist_msg}."
            )
        raise ValueError(
            f"Cannot post a period-closing entry for {period_end} because required checklists are incomplete."
        )

    preview = compute_period_close_preview(
        conn=conn,
        period_start=period_start,
        period_end=period_end,
        retained_earnings_gl_number=retained_earnings_gl_number,
    )

    if not preview.lines:
        raise ValueError(
            f"No income or expense activity found between {period_start} and {period_end}; "
            "no closing entry is necessary."
        )

    # Build journal entry lines from the preview.
    je_lines: List[JournalEntryLine] = []
    memo_text = f"Period close {period_end}"

    for pl in preview.lines:
        # amount_cents in PeriodCloseLine is always positive.
        je_lines.append(
            JournalEntryLine(
                account_id=pl.account.id,
                amount_cents=pl.amount_cents,
                is_debit=pl.is_debit,
                memo=memo_text,
            )
        )

    # Effective date is the day after period_end (e.g., closing November on 11/30
    # will create a closing entry dated 12/01).
    effective_date_next = (period_end_date + timedelta(days=1)).isoformat()

    post_date_str = date.today().isoformat()

    # Description encodes which period was closed so has_period_close_entry can find it.
    je_description = description or f"Closing entry for {period_end}"

    je_id = insert_journal_entry(
        conn=conn,
        effective_date=effective_date_next,
        post_date=post_date_str,
        description=je_description,
        source="PERIOD_CLOSE",
        lines=je_lines,
    )

    # Log the period close event separately from the generic JE log.
    log_activity(
        conn,
        event_type="PERIOD_CLOSE_POSTED",
        entity_type="JOURNAL_ENTRY",
        entity_id=je_id,
        summary=f"Period close entry posted for {period_end}",
        details={
            "journal_entry_id": je_id,
            "period_start": period_start,
            "period_end": period_end,
            "retained_earnings_gl_number": retained_earnings_gl_number,
        },
    )

    return je_id


def init_db() -> None:
    """
    Initialize the database (currently: accounts table + seed data).

    We will extend this later to create the other core tables:
    - journal_entries
    - journal_lines
    - import_profiles
    - import_uploads
    - import_staging
    - historical_bs_snapshots
    """
    conn = get_connection()
    try:
        init_accounts_table(conn)
        init_journal_entries_table(conn)
        init_journal_lines_table(conn)
        init_bs_category_mappings_table(conn)
        init_settings_table(conn)
        init_txn_keyword_rules_table(conn)
        init_notes_table(conn)
        init_note_gl_links_table(conn)
        init_activity_log_table(conn)
        init_manual_txn_templates_table(conn)
        init_user_upload_mappings_table(conn)
        init_month_end_checklists_table(conn)
        init_month_end_checklist_accounts_table(conn)
        init_month_end_checklist_status_table(conn)
        init_month_end_checklist_account_status_table(conn)
        seed_initial_chart_of_accounts(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    # Running this file directly will initialize the DB with the accounts table
    # and seed the initial chart of accounts if the table is empty.
    init_db()
    print(f"Initialized database at {DB_PATH}")
