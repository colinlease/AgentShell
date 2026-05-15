"""Shared constants for the Personal GL app."""

from __future__ import annotations

from dataclasses import dataclass

APP_ID = "personal_gl"
APP_LABEL = "Personal GL"
APP_TYPE = "streamlit"


@dataclass(frozen=True)
class TabDefinition:
    key: str
    label: str


TAB_DEFINITIONS: tuple[TabDefinition, ...] = (
    TabDefinition("financial_statements", "Financial Statements"),
    TabDefinition("account_history", "Account History"),
    TabDefinition("chart_of_accounts", "Chart of Accounts"),
    TabDefinition("journal_entries", "Journal Entries"),
    TabDefinition("upload_transactions", "Upload TXN"),
    TabDefinition("preferences", "Preferences"),
    TabDefinition("warnings", "Warnings"),
    TabDefinition("notes", "Notes"),
    TabDefinition("search", "Search"),
    TabDefinition("logs", "Logs"),
    TabDefinition("documentation", "Documenation"),
)

TAB_LABELS: list[str] = [tab.label for tab in TAB_DEFINITIONS]
TAB_KEYS: list[str] = [tab.key for tab in TAB_DEFINITIONS]
TAB_LABEL_BY_KEY: dict[str, str] = {tab.key: tab.label for tab in TAB_DEFINITIONS}

DOC_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Using the App", "using_the_app.md"),
    ("Getting Started", "getting_started.md"),
    ("Monthly Adjustments", "common_monthly_adjustments.md"),
    ("Data Stored", "data_stored.md"),
    ("Financial Statements", "financial_statements.md"),
    ("Account History", "account_history.md"),
    ("Chart of Accounts", "chart_of_accounts.md"),
    ("Journal Entries", "journal_entries.md"),
    ("Upload TXN", "upload_txn.md"),
    ("Preferences", "preferences.md"),
    ("Warnings", "warnings.md"),
    ("Notes", "notes.md"),
    ("Logs", "logs.md"),
    ("Transaction Source Codes", "journal_entry_sources.md"),
)

DATASET_SQLITE_DB = "sqlite_db"
DATASET_FINANCIAL_REPORT = "financial_report"
DATASET_ACCOUNT_HISTORY = "account_history"
DATASET_JOURNAL_SEARCH = "journal_search"
DATASET_UPLOAD_STAGE = "upload_stage"
DATASET_BULK_JE_STAGE = "bulk_je_stage"
DATASET_WARNINGS_CHECKLIST_STATUS = "warnings_checklist_status"
DATASET_WARNINGS_CHECKLIST_DETAIL = "warnings_checklist_detail"
DATASET_NOTES = "notes"
DATASET_LOGS = "logs"

DATASET_NAMES: tuple[str, ...] = (
    DATASET_SQLITE_DB,
    DATASET_FINANCIAL_REPORT,
    DATASET_ACCOUNT_HISTORY,
    DATASET_JOURNAL_SEARCH,
    DATASET_UPLOAD_STAGE,
    DATASET_BULK_JE_STAGE,
    DATASET_WARNINGS_CHECKLIST_STATUS,
    DATASET_WARNINGS_CHECKLIST_DETAIL,
    DATASET_NOTES,
    DATASET_LOGS,
)

SESSION_KEY_PREFIX = "personal_gl"

BS_CATEGORIES: dict[str, dict] = {
    "LIQUID_CASH": {"label": "Liquid Cash", "default_gls": [11000, 11100, 11200]},
    "CERTIFICATES": {"label": "Certificates", "default_gls": [12000]},
    "EQUITIES": {"label": "Equities", "default_gls": []},
    "BONDS": {"label": "Bonds", "default_gls": []},
    "CRYPTOCURRENCY": {"label": "Cryptocurrency", "default_gls": []},
    "RETIREMENT_INVEST": {"label": "Retirement Investments", "default_gls": [13100]},
    "LIQUID_INVEST": {"label": "Liquid Investments", "default_gls": [13000]},
    "ILLIQUID_INVEST": {"label": "Illiquid Investments", "default_gls": []},
    "INVENTORY": {"label": "Inventory", "default_gls": []},
    "ACCRUED": {"label": "Accrued Revenue / Receivables", "default_gls": [14000, 14100]},
    "ALLOWANCE_DOUBTFUL": {"label": "Allowance for Doubtful Accounts", "default_gls": []},
    "NET_ACCRUED": {"label": "Net Accrued Revenue / Receivables", "default_gls": []},
    "REAL_ESTATE": {"label": "Real Estate", "default_gls": []},
    "FIXED_ASSETS": {"label": "Fixed Assets", "default_gls": [15000]},
    "ACCUMULATED DEPRECIATION": {"label": "Accumulated Depreciation", "default_gls": []},
    "NET FIXED ASSETS": {"label": "Net Fixed Assets", "default_gls": []},
    "OTHER_ASSETS": {"label": "Other Assets", "default_gls": []},
    "LIQUID_ASSETS": {"label": "Liquid Assets", "default_gls": [11000, 11100, 11200, 13000, 14000, 14100]},
    "ILLIQUID_ASSETS": {"label": "Illiquid Assets", "default_gls": [13100, 15000]},
    "REVOLVING_DEBT": {"label": "Revolving Debt", "default_gls": [21000, 21100]},
    "INSTALLMENT_DEBT": {"label": "Installment Debt", "default_gls": [22100, 23000]},
    "EDUCATION_DEBT": {"label": "Education Debt", "default_gls": []},
    "REAL_ESTATE_DEBT": {"label": "Real Estate Debt", "default_gls": []},
    "PAYABLES": {"label": "Payables / Accrued Expenses", "default_gls": []},
    "OTHER_LIABILITIES": {"label": "Other Liabilities", "default_gls": []},
    "TOTAL_EQUITY": {"label": "Total Equity (Balance Sheet)", "default_gls": [30000, 30100, 30200, 30010]},
}
