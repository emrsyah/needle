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


__all__ = ["ActionParseError", "AnswerAction", "SearchAction", "parse_action"]
