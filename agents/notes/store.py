from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.notes.models import RuntimeNote, RuntimeNoteFile
from agents.notes.search import score_note_match
from agents.notes.validators import build_note_fingerprint, normalize_note_payload, validate_note_count


_NOTE_FILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class RuntimeNoteStore:
    """
    File-backed store for small bounded runtime note collections.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.apps_root = self.root / "apps"

    def list_note_files(self) -> list[dict[str, Any]]:
        """
        Return available note file metadata.
        """
        files: list[dict[str, Any]] = []

        general_path = self._general_path()
        if general_path.exists():
            general_note_file = self._load_note_file(general_path)
            files.append(
                {
                    "file_name": "general",
                    "scope": "general",
                    "app_id": None,
                    "path": str(general_path),
                    "note_count": len(general_note_file.notes),
                }
            )

        if self.apps_root.exists():
            for path in sorted(self.apps_root.glob("*.json")):
                note_file = self._load_note_file(path)
                files.append(
                    {
                        "file_name": path.stem,
                        "scope": "app",
                        "app_id": path.stem,
                        "path": str(path),
                        "note_count": len(note_file.notes),
                    }
                )

        return files

    def search_notes(
        self,
        *,
        query: str,
        file_name: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search notes across all files by default, or within one named file.
        """
        normalized_query = " ".join(str(query or "").strip().split())
        if not normalized_query:
            return []

        candidate_files = (
            [self._resolve_note_path(file_name)] if file_name else self._list_existing_note_paths()
        )

        matches: list[tuple[int, str, RuntimeNote]] = []
        for path in candidate_files:
            if path is None or not path.exists():
                continue
            note_file = self._load_note_file(path)
            for note in note_file.notes:
                score = score_note_match(note, normalized_query)
                if score <= 0:
                    continue
                matches.append((score, path.stem, note))

        matches.sort(key=lambda item: (-item[0], item[1], item[2].note_id))
        limited_matches = matches[: max(1, min(int(limit), 20))]

        results: list[dict[str, Any]] = []
        for score, source_file, note in limited_matches:
            results.append(
                {
                    "file_name": source_file,
                    "score": score,
                    "note": self._note_to_dict(note),
                }
            )

        return results

    def get_note(
        self,
        *,
        note_id: str,
        file_name: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Return one note by id, optionally restricted to a single file.
        """
        normalized_note_id = " ".join(str(note_id or "").strip().split())
        if not normalized_note_id:
            return None

        candidate_files = (
            [self._resolve_note_path(file_name)] if file_name else self._list_existing_note_paths()
        )

        for path in candidate_files:
            if path is None or not path.exists():
                continue
            note_file = self._load_note_file(path)
            for note in note_file.notes:
                if note.note_id != normalized_note_id:
                    continue
                return {
                    "file_name": path.stem,
                    "note": self._note_to_dict(note),
                }

        return None

    def upsert_note(
        self,
        *,
        file_name: str,
        scope: str,
        app_id: str | None,
        note_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create or update one note in a target file.
        """
        payload = dict(note_payload)
        if not str(payload.get("updated_at", "")).strip():
            payload["updated_at"] = self._current_timestamp()

        normalized_note = normalize_note_payload(payload)
        if normalized_note is None:
            raise ValueError("Invalid note payload.")

        path = self._resolve_note_path(file_name)
        self._ensure_parent_dirs(path)
        note_file = self._load_note_file(path, create_if_missing=True, scope=scope, app_id=app_id)

        updated_notes: list[RuntimeNote] = []
        replaced = False
        for note in note_file.notes:
            if note.note_id == normalized_note.note_id:
                updated_notes.append(normalized_note)
                replaced = True
            else:
                updated_notes.append(note)

        duplicate = self._find_duplicate_note(updated_notes, normalized_note, replacing_note_id=normalized_note.note_id)
        if duplicate is not None:
            raise ValueError(
                f"Duplicate note content matches existing note '{duplicate.note_id}'."
            )

        if not replaced:
            validate_note_count(len(updated_notes) + 1)
            updated_notes.append(normalized_note)

        serialized = {
            "scope": scope,
            "app_id": app_id,
            "version": 1,
            "notes": [self._note_to_dict(note) for note in updated_notes],
        }
        path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

        return {
            "status": "ok",
            "file_name": file_name,
            "replaced": replaced,
            "note": self._note_to_dict(normalized_note),
        }

    def delete_note(
        self,
        *,
        note_id: str,
        file_name: str,
    ) -> dict[str, Any]:
        """
        Delete one note from a target file.
        """
        path = self._resolve_note_path(file_name)
        if not path.exists():
            return {"status": "error", "message": "Note file not found."}

        note_file = self._load_note_file(path)
        retained_notes = [note for note in note_file.notes if note.note_id != note_id]
        deleted = len(retained_notes) != len(note_file.notes)

        serialized = {
            "scope": note_file.scope,
            "app_id": note_file.app_id,
            "version": note_file.version,
            "notes": [self._note_to_dict(note) for note in retained_notes],
        }
        path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

        return {
            "status": "ok" if deleted else "error",
            "deleted": deleted,
            "file_name": file_name,
            "note_id": note_id,
        }

    def _list_existing_note_paths(self) -> list[Path]:
        paths: list[Path] = []
        general_path = self._general_path()
        if general_path.exists():
            paths.append(general_path)

        if self.apps_root.exists():
            paths.extend(sorted(self.apps_root.glob("*.json")))

        return paths

    def _general_path(self) -> Path:
        return self.root / "general.json"

    def _resolve_note_path(self, file_name: str | None) -> Path | None:
        normalized = " ".join(str(file_name or "").strip().split())
        if not normalized:
            return None
        if normalized == "general":
            return self._general_path()
        if _NOTE_FILE_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError("Invalid note file name.")
        return self.apps_root / f"{normalized}.json"

    def _ensure_parent_dirs(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load_note_file(
        self,
        path: Path,
        *,
        create_if_missing: bool = False,
        scope: str = "general",
        app_id: str | None = None,
    ) -> RuntimeNoteFile:
        if not path.exists():
            if not create_if_missing:
                return RuntimeNoteFile(scope=scope, app_id=app_id, version=1, notes=[])

            payload = {
                "scope": scope,
                "app_id": app_id,
                "version": 1,
                "notes": [],
            }
            self._ensure_parent_dirs(path)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        notes_raw = raw_payload.get("notes", [])
        notes: list[RuntimeNote] = []
        if isinstance(notes_raw, list):
            for item in notes_raw:
                normalized = normalize_note_payload(item)
                if normalized is not None:
                    notes.append(normalized)

        validate_note_count(len(notes))

        return RuntimeNoteFile(
            scope=str(raw_payload.get("scope", scope)),
            app_id=raw_payload.get("app_id", app_id),
            version=int(raw_payload.get("version", 1)),
            notes=notes,
        )

    @staticmethod
    def _note_to_dict(note: RuntimeNote) -> dict[str, Any]:
        return {
            "note_id": note.note_id,
            "title": note.title,
            "statement": note.statement,
            "tags": list(note.tags),
            "keywords": list(note.keywords),
            "confidence": note.confidence,
            "updated_at": note.updated_at,
        }

    @staticmethod
    def _find_duplicate_note(
        existing_notes: list[RuntimeNote],
        candidate: RuntimeNote,
        *,
        replacing_note_id: str | None = None,
    ) -> RuntimeNote | None:
        candidate_fingerprint = build_note_fingerprint(
            title=candidate.title,
            statement=candidate.statement,
        )
        for note in existing_notes:
            if replacing_note_id and note.note_id == replacing_note_id:
                continue
            note_fingerprint = build_note_fingerprint(
                title=note.title,
                statement=note.statement,
            )
            if note_fingerprint == candidate_fingerprint:
                return note
        return None

    @staticmethod
    def _current_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
