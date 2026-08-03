"""Strict, framework-independent model actions for Needle episodes.

The accepted grammar is::

    SEARCH[query]
    ANSWER[answer]
    ANSWER[answer] CITATIONS[citation; citation; ...]

Whitespace around the complete action and between citation entries is allowed.
Each citation has the form ``title|sentence_index``. ``CITATIONS[]`` is
accepted and means no citations. Payloads may not contain brackets, newlines,
``|``, or ``;`` where those characters would make the grammar ambiguous.
Corpus and retrieval validation remain responsibilities of SearchEnvironment.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from needle.data import EvidenceRef

_ACTION_PATTERN = re.compile(
    r"^(?P<name>SEARCH|ANSWER)\[(?P<payload>[^\[\]\r\n]*)\]"
    r"(?:[ \t]+CITATIONS\[(?P<citations>[^\[\]\r\n]*)\])?$"
)
_SENTENCE_INDEX_PATTERN = re.compile(r"^[0-9]+$")


class ActionParseError(ValueError):
    """Raised when text does not match the strict Needle action grammar."""


@dataclass(frozen=True, slots=True)
class SearchAction:
    """A request to search the corpus with one non-empty query."""

    query: str


@dataclass(frozen=True, slots=True)
class AnswerAction:
    """A terminal answer and its syntactically parsed evidence citations."""

    answer: str
    citations: tuple[EvidenceRef, ...]


def _normalize_citation_title(title: str) -> str:
    """Normalize only harmless title formatting differences."""
    normalized = unicodedata.normalize("NFKC", title).casefold()
    characters = [
        character
        for character in normalized
        if character.isspace() or unicodedata.category(character)[0] in {"L", "N"}
    ]
    return " ".join("".join(characters).split())


def canonicalize_citation_title(provided_title: str, corpus_titles: Sequence[str]) -> str | None:
    """Return a corpus title for a unique, harmlessly normalized title match.

    Exact titles are preserved. Otherwise, case, Unicode normalization,
    punctuation, and whitespace are ignored. A normalized key that maps to
    more than one corpus title is intentionally rejected as ambiguous.
    """
    if not isinstance(provided_title, str):
        raise TypeError("provided_title must be a string")

    titles = tuple(corpus_titles)
    if any(not isinstance(title, str) for title in titles):
        raise TypeError("corpus_titles must contain strings")

    exact_matches = tuple(title for title in titles if title == provided_title)
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        return None

    normalized_title = _normalize_citation_title(provided_title)
    if not normalized_title:
        return None
    candidates = tuple(
        title for title in titles if _normalize_citation_title(title) == normalized_title
    )
    return candidates[0] if len(candidates) == 1 else None


def _parse_citations(raw: str) -> tuple[EvidenceRef, ...]:
    if not raw.strip():
        return ()

    citations: list[EvidenceRef] = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            raise ActionParseError("citation entries must not be empty")
        parts = entry.split("|")
        if len(parts) != 2:
            raise ActionParseError("each citation must have the form title|sentence_index")
        title, index_text = (part.strip() for part in parts)
        if not title:
            raise ActionParseError("citation title must be non-empty")
        if not _SENTENCE_INDEX_PATTERN.fullmatch(index_text):
            raise ActionParseError("citation sentence_index must be a non-negative integer")
        citations.append(EvidenceRef(title, int(index_text)))

    parsed = tuple(citations)
    if len(set(parsed)) != len(parsed):
        raise ActionParseError("citations must not contain duplicates")
    return parsed


def parse_action(text: str) -> SearchAction | AnswerAction:
    """Parse one strict model action, rejecting malformed or trailing text."""
    if not isinstance(text, str):
        raise TypeError("action text must be a string")
    if "\n" in text or "\r" in text:
        raise ActionParseError("actions must not contain newlines")

    match = _ACTION_PATTERN.fullmatch(text.strip())
    if match is None:
        raise ActionParseError("action does not match the Needle action grammar")

    payload = match.group("payload").strip()
    name = match.group("name")
    if not payload:
        raise ActionParseError(f"{name.lower()} payload must be non-empty")
    if name == "SEARCH":
        if match.group("citations") is not None:
            raise ActionParseError("SEARCH actions cannot include citations")
        return SearchAction(payload)
    return AnswerAction(payload, _parse_citations(match.group("citations") or ""))


__all__ = [
    "ActionParseError",
    "AnswerAction",
    "SearchAction",
    "canonicalize_citation_title",
    "parse_action",
]
