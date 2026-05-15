from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def normalize_folder_path(raw_path: str) -> str:
    """
    Normalize user-entered folder paths without assuming a specific OS style.

    The app runs on the user's machine, so existence validation is still done
    by pathlib on the current platform. This helper accepts common copy/paste
    forms such as quoted macOS paths, tilde paths, and Windows-style paths.
    """
    value = str(raw_path or "").strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()
    if not value:
        return ""
    expanded = os.path.expandvars(os.path.expanduser(value))
    return str(Path(expanded))


def validate_folder_path(raw_path: str) -> dict[str, Any]:
    normalized_path = normalize_folder_path(raw_path)
    if not normalized_path:
        return {
            "status": "error",
            "message": "Enter a folder path before mounting.",
            "normalized_path": "",
        }

    folder_path = Path(normalized_path)
    if not folder_path.exists():
        return {
            "status": "error",
            "message": "Folder path does not exist on this machine.",
            "normalized_path": normalized_path,
        }
    if not folder_path.is_dir():
        return {
            "status": "error",
            "message": "Path exists, but it is not a folder.",
            "normalized_path": normalized_path,
        }

    try:
        resolved_path = folder_path.resolve()
    except OSError:
        resolved_path = folder_path

    return {
        "status": "ok",
        "message": "Folder mounted.",
        "normalized_path": str(resolved_path),
    }

