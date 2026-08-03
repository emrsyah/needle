"""Tests for deterministic BM25 document retrieval."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from needle.data import Document, load_hotpotqa
from needle.retrieval import BM25Retriever, SearchResult, tokenize

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"


def _documents() -> tuple[Document, ...]:
    return (
        Document("Alpha", ("alpha beta",)),
        Document("Bravo", ("bravo charlie",)),
        Document("Charlie", ("charlie delta",)),
    )


def test_tokenize_casefolds_punctuation_and_unicode() -> None:
    assert tokenize("Stra\u00dfe, CAF\u00c9! co-op") == ("strasse", "caf\u00e9", "co", "op")
    assert tokenize("\u6771\u4eac\u3002\u041f\u0440\u0418\u0432\u0415\u0442!") == (
        "\u6771\u4eac",
        "\u043f\u0440\u0438\u0432\u0435\u0442",
    )


def test_tokenize_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text.*string"):
        tokenize(1)  # type: ignore[arg-type]


@pytest.mark.parametrize("documents", [(), [], "not documents"])
def test_rejects_empty_or_wrong_corpus(documents: object) -> None:
    with pytest.raises((TypeError, ValueError), match="documents"):
        BM25Retriever(documents)  # type: ignore[arg-type]


def test_rejects_non_document_and_duplicate_titles() -> None:
    with pytest.raises(TypeError, match="Document"):
        BM25Retriever(("not a document",))  # type: ignore[arg-type]
    duplicate = Document("Same", ("another sentence",))
    with pytest.raises(ValueError, match="duplicate title"):
        BM25Retriever((Document("Same", ("one sentence",)), duplicate))


@pytest.mark.parametrize("query", ["", "...?!", 1, None])
def test_rejects_empty_punctuation_only_or_nonstring_query(query: object) -> None:
    with pytest.raises((TypeError, ValueError), match="query"):
        BM25Retriever(_documents()).search(query)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1, True, False, 1.5, "1"])
def test_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises((TypeError, ValueError), match="top_k"):
        BM25Retriever(_documents()).search("alpha", top_k=top_k)  # type: ignore[arg-type]


def test_results_are_immutable_typed_and_retain_original_documents() -> None:
    documents = _documents()
    retriever = BM25Retriever(documents)

    assert retriever.documents == documents
    assert isinstance(retriever.documents, tuple)
    result = retriever.search("alpha")[0]
    assert isinstance(result, SearchResult)
    assert result.document is documents[0]
    assert isinstance(result.rank, int)
    assert isinstance(result.score, float)
    with pytest.raises(FrozenInstanceError):
        result.rank = 2  # type: ignore[misc]


def test_caps_results_and_breaks_zero_score_ties_by_input_order() -> None:
    documents = _documents()
    results = BM25Retriever(documents).search("unmatched", top_k=99)

    assert [result.document for result in results] == list(documents)
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.score for result in results] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("example_index", "query", "expected_title"),
    [
        (0, "Left Hand author", "The Left Hand of Darkness"),
        (0, "born Berkeley United States", "Ursula K. Le Guin"),
        (1, "Machu Picchu located", "Machu Picchu"),
        (1, "coastline Pacific Ocean", "Peru"),
        (2, "Galapagos part", "Galapagos Islands"),
        (2, "capital city Quito", "Ecuador"),
    ],
)
def test_targeted_fixture_queries_rank_each_intended_support_first(
    example_index: int, query: str, expected_title: str
) -> None:
    example = load_hotpotqa(FIXTURE_PATH)[example_index]

    assert (
        BM25Retriever(example.documents).search(query, top_k=1)[0].document.title == expected_title
    )
