from __future__ import annotations

import re

from agents.notes.models import RuntimeNote


_STOPWORDS = {
    "about",
    "after",
    "before",
    "could",
    "explain",
    "from",
    "have",
    "into",
    "like",
    "many",
    "more",
    "should",
    "than",
    "that",
    "them",
    "then",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def extract_note_search_terms(text: str, *, max_terms: int = 6) -> list[str]:
    """
    Extract compact keyword-like terms from freeform text for note search fallback.
    """
    lowered = " ".join(str(text or "").strip().lower().split())
    if not lowered:
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[a-z0-9_]+", lowered):
        candidate = _normalize_term(raw)
        if not candidate or candidate in seen:
            continue
        terms.append(candidate)
        seen.add(candidate)
        if len(terms) >= max(1, int(max_terms)):
            break
    return terms


def build_targeted_note_query(text: str, *, max_terms: int = 2) -> str:
    """
    Build a short targeted query string for an initial note lookup.
    """
    terms = extract_note_search_terms(text, max_terms=max(2, max_terms * 3))
    if not terms:
        return " ".join(str(text or "").strip().split())[:240]

    ranked = sorted(enumerate(terms), key=lambda item: (-len(item[1]), item[0]))
    selected = [term for _, term in ranked[: max(1, int(max_terms))]]
    return " ".join(selected)[:240]


def score_note_match(note: RuntimeNote, query: str) -> int:
    """
    Score a note for a case-insensitive substring search, with keyword fallback
    when the raw query is a longer user phrase.
    """
    normalized_query = " ".join(str(query or "").strip().lower().split())
    if not normalized_query:
        return 0

    direct_score = _score_fragment(
        note,
        normalized_query,
        title_weight=5,
        statement_weight=4,
        tag_weight=3,
        keyword_weight=4,
    )
    if direct_score > 0:
        return direct_score

    fallback_score = 0
    for term in extract_note_search_terms(normalized_query):
        fallback_score += _score_fragment(
            note,
            term,
            title_weight=2,
            statement_weight=1,
            tag_weight=1,
            keyword_weight=2,
        )
    return fallback_score


def _score_fragment(
    note: RuntimeNote,
    fragment: str,
    *,
    title_weight: int,
    statement_weight: int,
    tag_weight: int,
    keyword_weight: int,
) -> int:
    score = 0
    if fragment in note.title.lower():
        score += title_weight
    if fragment in note.statement.lower():
        score += statement_weight
    if any(fragment in tag.lower() for tag in note.tags):
        score += tag_weight
    if any(fragment in keyword.lower() for keyword in note.keywords):
        score += keyword_weight
    return score


def _normalize_term(raw: str) -> str:
    candidate = raw.strip().lower()
    if len(candidate) < 4 or candidate in _STOPWORDS:
        return ""
    if candidate.endswith("ies") and len(candidate) > 4:
        candidate = f"{candidate[:-3]}y"
    elif candidate.endswith("s") and not candidate.endswith("ss") and len(candidate) > 4:
        candidate = candidate[:-1]
    return candidate if candidate and candidate not in _STOPWORDS else ""
