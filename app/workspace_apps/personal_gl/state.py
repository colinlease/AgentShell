"""Centralized Streamlit session-state access."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, MutableMapping, Optional

from app.workspace_apps.personal_gl.constants import (
    DATASET_ACCOUNT_HISTORY,
    DATASET_BULK_JE_STAGE,
    DATASET_FINANCIAL_REPORT,
    DATASET_JOURNAL_SEARCH,
    DATASET_LOGS,
    DATASET_NOTES,
    DATASET_SQLITE_DB,
    DATASET_UPLOAD_STAGE,
    DATASET_WARNINGS_CHECKLIST_DETAIL,
    DATASET_WARNINGS_CHECKLIST_STATUS,
    SESSION_KEY_PREFIX,
)


def _session_key(name: str) -> str:
    return f"{SESSION_KEY_PREFIX}_{name}"


@dataclass
class UploadStageMeta:
    bank_gl: int
    source: str
    institution: str


class AppState:
    """Small wrapper around Streamlit session state for cross-rerun state."""

    def __init__(self, storage: MutableMapping[str, Any]):
        self.storage = storage

    def ensure_defaults(self) -> None:
        self.storage.setdefault("app_unlocked", False)
        self.storage.setdefault(_session_key("active_tab_hint"), "financial_statements")
        self.storage.setdefault(_session_key("active_dataset_name"), None)
        self.storage.setdefault("confirm_delete", False)
        self.storage.setdefault("je_reverse_state", {})
        self.storage.setdefault("period_close_ready", False)
        self.storage.setdefault("period_close_ready_date", None)
        self.storage.setdefault(_session_key("ui_events"), [])

    def set_active_tab_hint(self, tab_key: str) -> None:
        self.storage[_session_key("active_tab_hint")] = tab_key

    def get_active_tab_hint(self) -> str:
        return str(self.storage.get(_session_key("active_tab_hint"), "financial_statements"))

    def set_active_dataset_name(self, dataset_name: str | None) -> None:
        self.storage[_session_key("active_dataset_name")] = str(dataset_name) if dataset_name else None

    def get_active_dataset_name(self) -> str | None:
        value = self.storage.get(_session_key("active_dataset_name"))
        return str(value) if value else None

    def resolve_active_dataset_name(self, available_dataset_names: list[str]) -> str | None:
        available = {str(name) for name in available_dataset_names if str(name).strip()}
        if not available:
            return None

        preferred_by_tab = {
            "financial_statements": DATASET_FINANCIAL_REPORT,
            "account_history": DATASET_ACCOUNT_HISTORY,
            "journal_entries": DATASET_BULK_JE_STAGE,
            "upload_transactions": DATASET_UPLOAD_STAGE,
            "warnings": DATASET_WARNINGS_CHECKLIST_DETAIL,
            "notes": DATASET_NOTES,
            "search": DATASET_JOURNAL_SEARCH,
            "logs": DATASET_LOGS,
        }

        current_active = self.get_active_dataset_name()
        if current_active in available:
            return current_active

        preferred = preferred_by_tab.get(self.get_active_tab_hint())
        if preferred in available:
            self.set_active_dataset_name(preferred)
            return preferred

        if DATASET_WARNINGS_CHECKLIST_STATUS in available and self.get_active_tab_hint() == "warnings":
            self.set_active_dataset_name(DATASET_WARNINGS_CHECKLIST_STATUS)
            return DATASET_WARNINGS_CHECKLIST_STATUS

        non_db_datasets = [name for name in sorted(available) if name != DATASET_SQLITE_DB]
        resolved = non_db_datasets[0] if non_db_datasets else DATASET_SQLITE_DB
        self.set_active_dataset_name(resolved)
        return resolved

    def append_ui_event(self, event: str) -> None:
        events = list(self.storage.get(_session_key("ui_events"), []))
        events.append(event)
        self.storage[_session_key("ui_events")] = events[-25:]

    def get_ui_events(self) -> list[str]:
        return list(self.storage.get(_session_key("ui_events"), []))

    def get_app_unlocked(self) -> bool:
        return bool(self.storage.get("app_unlocked", False))

    def set_app_unlocked(self, value: bool) -> None:
        self.storage["app_unlocked"] = bool(value)

    def get_confirm_delete(self) -> bool:
        return bool(self.storage.get("confirm_delete", False))

    def set_confirm_delete(self, value: bool) -> None:
        self.storage["confirm_delete"] = bool(value)

    def get_je_reverse_state(self) -> dict[int, bool]:
        return dict(self.storage.get("je_reverse_state", {}))

    def set_je_reverse_state(self, value: dict[int, bool]) -> None:
        self.storage["je_reverse_state"] = dict(value)

    def get_bulk_je_stage_df(self):
        return self.storage.get("je_bulk_stage_df")

    def set_bulk_je_stage_df(self, value) -> None:
        self.storage["je_bulk_stage_df"] = value

    def clear_bulk_je_stage_df(self) -> None:
        self.storage.pop("je_bulk_stage_df", None)

    def get_upload_stage_df(self):
        return self.storage.get("upload_stage_df")

    def set_upload_stage_df(self, value) -> None:
        self.storage["upload_stage_df"] = value

    def clear_upload_stage(self) -> None:
        self.storage.pop("upload_stage_df", None)
        self.storage.pop("upload_stage_meta", None)

    def get_upload_stage_meta(self) -> Optional[UploadStageMeta]:
        raw = self.storage.get("upload_stage_meta")
        if not raw:
            return None
        if isinstance(raw, UploadStageMeta):
            return raw
        if isinstance(raw, dict):
            return UploadStageMeta(
                bank_gl=int(raw["bank_gl"]),
                source=str(raw["source"]),
                institution=str(raw["institution"]),
            )
        return None

    def set_upload_stage_meta(self, value: UploadStageMeta) -> None:
        self.storage["upload_stage_meta"] = asdict(value)

    def get_period_close_prev_end(self):
        return self.storage.get("period_close_prev_end")

    def set_period_close_prev_end(self, value: date | None) -> None:
        self.storage["period_close_prev_end"] = value

    def get_period_close_ready(self) -> bool:
        return bool(self.storage.get("period_close_ready", False))

    def get_period_close_ready_date(self) -> str | None:
        value = self.storage.get("period_close_ready_date")
        return str(value) if value else None

    def set_period_close_ready(self, ready: bool, period_end_iso: str | None = None) -> None:
        self.storage["period_close_ready"] = bool(ready)
        self.storage["period_close_ready_date"] = period_end_iso if ready else None

    def clear_period_close_ready(self) -> None:
        self.set_period_close_ready(False, None)

    def build_ui_state_snapshot(self) -> dict[str, Any]:
        upload_df = self.get_upload_stage_df()
        bulk_df = self.get_bulk_je_stage_df()
        return {
            "active_tab": self.get_active_tab_hint(),
            "active_dataset_name": self.get_active_dataset_name(),
            "app_unlocked": self.get_app_unlocked(),
            "selected_dates": {
                "financials_as_of": self.storage.get("financials_as_of"),
                "warnings_as_of": self.storage.get("warnings_as_of"),
                "account_history_start": self.storage.get("account_history_start"),
                "account_history_end": self.storage.get("account_history_end"),
                "period_close_end": self.storage.get("close_period_end"),
            },
            "selected_objects": {
                "report_type": self.storage.get("financials_report_type"),
                "account": self.storage.get("account_history_account_label"),
                "checklist": self.storage.get("warnings_month_end_checklist_select"),
                "search_type": self.storage.get("search_tab_type"),
            },
            "search_filters": {
                "search_je_description": self.storage.get("search_je_description"),
                "search_je_memo": self.storage.get("search_je_memo"),
                "search_je_source": self.storage.get("search_je_source"),
                "search_je_accounts": self.storage.get("search_je_accounts"),
                "search_je_dc_filter": self.storage.get("search_je_dc_filter"),
            },
            "flags": {
                "has_upload_stage": upload_df is not None,
                "has_bulk_je_stage": bulk_df is not None,
                "period_close_ready": self.get_period_close_ready(),
                "period_close_ready_date": self.get_period_close_ready_date(),
                "confirm_delete": self.get_confirm_delete(),
            },
        }
