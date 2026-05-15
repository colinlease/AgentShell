from __future__ import annotations

import json
from typing import Any


def parse_json_object(text: str) -> dict[str, Any] | None:
    """
    Parse a JSON object from raw provider text with light fence stripping.
    """
    if not isinstance(text, str):
        return None

    candidate = text.strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None
