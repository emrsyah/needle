from dataclasses import FrozenInstanceError

import pytest

from needle.actions import (
    ActionParseError,
    AnswerAction,
    SearchAction,
    canonicalize_citation_title,
    parse_action,
)
from needle.data import EvidenceRef


def test_parses_search_with_outer_and_query_whitespace() -> None:
    assert parse_action("  SEARCH[  Ursula Le Guin  ]  ") == SearchAction("Ursula Le Guin")


def test_parses_answer_without_citations() -> None:
    assert parse_action("ANSWER[United States]") == AnswerAction("United States", ())


def test_parses_answer_and_trimmed_citations() -> None:
    action = parse_action(
        " ANSWER[Canada] CITATIONS[ Ursula K. Le Guin | 1; The Left Hand of Darkness|1 ] "
    )
    assert action == AnswerAction(
        "Canada",
        (EvidenceRef("Ursula K. Le Guin", 1), EvidenceRef("The Left Hand of Darkness", 1)),
    )


def test_empty_citation_block_is_supported() -> None:
    assert parse_action("ANSWER[Canada] CITATIONS[  ]") == AnswerAction("Canada", ())


@pytest.mark.parametrize(
    "text",
    [
        "",
        "SEARCH[]",
        "ANSWER[]",
        "SEARCH[one] trailing",
        "SEARCH[one]\nSEARCH[two]",
        "search[one]",
        "ANSWER[one] CITATIONS[title]",
        "ANSWER[one] CITATIONS[title|-1]",
        "ANSWER[one] CITATIONS[title|nope]",
        "ANSWER[one] CITATIONS[title|0;]",
        "ANSWER[one] CITATIONS[|0]",
        "ANSWER[one] CITATIONS[title|0|extra]",
        "ANSWER[one] CITATIONS[title|0;title|0]",
        "ANSWER[one] CITATIONS[title;other|0]",
        "ANSWER[one] CITATIONS[title|0] unexpected",
    ],
)
def test_rejects_malformed_actions(text: str) -> None:
    with pytest.raises(ActionParseError):
        parse_action(text)


def test_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        parse_action(None)  # type: ignore[arg-type]


def test_parser_does_not_require_citation_retrieval_or_corpus_validation() -> None:
    action = parse_action("ANSWER[Canada] CITATIONS[Not Retrieved|99]")
    assert action.citations == (EvidenceRef("Not Retrieved", 99),)


@pytest.mark.parametrize(
    ("provided", "expected"),
    [
        ("Ursula K. Le Guin", "Ursula K. Le Guin"),
        (" ursula k le guin ", "Ursula K. Le Guin"),
        ("The Left–Hand of ‘Darkness’", "The Left-Hand of 'Darkness'"),
    ],
)
def test_canonicalizes_unique_harmless_title_variations(provided: str, expected: str) -> None:
    assert canonicalize_citation_title(provided, (expected, "Berkeley")) == expected


def test_canonicalization_rejects_ambiguous_or_unknown_titles() -> None:
    titles = ("A/B", "A-B", "Known")

    assert canonicalize_citation_title("A B", titles) is None
    assert canonicalize_citation_title("Unrelated", titles) is None


def test_canonicalization_does_not_create_incorrect_aliases() -> None:
    titles = ("The Office (US)", "The Office (UK)")

    assert canonicalize_citation_title("The Office", titles) is None


def test_actions_are_immutable_and_slotted() -> None:
    action = SearchAction("query")
    assert hasattr(type(action), "__slots__")
    with pytest.raises(FrozenInstanceError):
        action.query = "other"  # type: ignore[misc]
