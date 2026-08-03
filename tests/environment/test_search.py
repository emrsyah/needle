"""Tests for the bounded BM25 search environment."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from needle.data import load_hotpotqa
from needle.environment import (
    QuestionObservation,
    SearchEnvironment,
    SearchEnvironmentError,
    SearchObservation,
)
from needle.retrieval import BM25Retriever

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"


def test_reset_starts_a_session_and_returns_the_question() -> None:
    example = load_hotpotqa(FIXTURE_PATH)[0]
    environment = SearchEnvironment(example, max_searches=2)

    observation = environment.reset()

    assert observation == QuestionObservation(
        question_id=example.question_id,
        question=example.question,
        max_searches=2,
    )
    assert environment.started is True
    assert environment.history == ()
    assert environment.remaining_searches == 2


def test_public_api_and_constructor_validation() -> None:
    import needle.environment as environment_module

    example = load_hotpotqa(FIXTURE_PATH)[0]
    environment = SearchEnvironment(example)

    assert environment_module.__all__ == [
        "QuestionObservation",
        "SearchEnvironment",
        "SearchEnvironmentError",
        "SearchObservation",
    ]
    assert environment.example is example
    assert environment.history == ()
    assert environment.remaining_searches == 5
    assert environment.started is False

    with pytest.raises(TypeError, match="example.*QuestionExample"):
        SearchEnvironment("not an example")  # type: ignore[arg-type]
    for value in (True, False, 1.5, "1"):
        with pytest.raises(TypeError, match="top_k"):
            SearchEnvironment(example, top_k=value)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="max_searches"):
            SearchEnvironment(example, max_searches=value)  # type: ignore[arg-type]
    for value in (0, -1):
        with pytest.raises(ValueError, match="top_k"):
            SearchEnvironment(example, top_k=value)
        with pytest.raises(ValueError, match="max_searches"):
            SearchEnvironment(example, max_searches=value)


def test_search_before_reset_does_not_change_initial_state() -> None:
    environment = SearchEnvironment(load_hotpotqa(FIXTURE_PATH)[0], max_searches=2)

    with pytest.raises(SearchEnvironmentError, match="reset"):
        environment.search("Ursula")

    assert environment.history == ()
    assert environment.remaining_searches == 2
    assert environment.started is False


def test_reset_clears_a_previous_history() -> None:
    example = load_hotpotqa(FIXTURE_PATH)[0]
    environment = SearchEnvironment(example, max_searches=2)
    environment.reset()
    environment.search("Ursula")

    repeated_question = environment.reset()

    assert repeated_question.question_id == example.question_id
    assert environment.history == ()
    assert environment.remaining_searches == 2


def test_successful_search_matches_the_direct_bm25_results() -> None:
    example = load_hotpotqa(FIXTURE_PATH)[0]
    environment = SearchEnvironment(example, top_k=2)
    environment.reset()

    observation = environment.search("Ursula Le Guin birthplace")
    expected = BM25Retriever(example.documents).search("Ursula Le Guin birthplace", top_k=2)

    assert [
        (
            result.document is expected_result.document,
            result.document.title,
            result.score,
            result.rank,
        )
        for result, expected_result in zip(observation.results, expected, strict=True)
    ] == [(True, result.document.title, result.score, result.rank) for result in expected]


def test_turn_history_and_remaining_searches_enforce_the_budget() -> None:
    environment = SearchEnvironment(load_hotpotqa(FIXTURE_PATH)[0], max_searches=2)
    environment.reset()

    first = environment.search("Ursula")
    second = environment.search("Berkeley")

    assert (first.turn, first.remaining_searches) == (1, 1)
    assert (second.turn, second.remaining_searches) == (2, 0)
    assert environment.history == (first, second)
    assert environment.remaining_searches == 0
    with pytest.raises(SearchEnvironmentError, match="exhausted"):
        environment.search("California")


def test_searches_strip_whitespace_and_record_repeated_queries() -> None:
    environment = SearchEnvironment(load_hotpotqa(FIXTURE_PATH)[0], max_searches=2)
    environment.reset()

    first = environment.search("  Ursula  ")
    second = environment.search("Ursula")

    assert first.query == "Ursula"
    assert [observation.query for observation in environment.history] == ["Ursula", "Ursula"]
    assert second.turn == 2


@pytest.mark.parametrize("query", ["", "   ", "...?!", None, 1])
def test_invalid_query_does_not_change_history_or_budget(query: object) -> None:
    environment = SearchEnvironment(load_hotpotqa(FIXTURE_PATH)[0], max_searches=2)
    environment.reset()

    with pytest.raises((TypeError, ValueError)):
        environment.search(query)  # type: ignore[arg-type]

    assert environment.history == ()
    assert environment.remaining_searches == 2


def test_observations_and_public_history_are_immutable() -> None:
    environment = SearchEnvironment(load_hotpotqa(FIXTURE_PATH)[0])
    question = environment.reset()
    observation = environment.search("Ursula")

    assert isinstance(question, QuestionObservation)
    assert isinstance(observation, SearchObservation)
    assert isinstance(environment.history, tuple)
    assert isinstance(observation.results, tuple)
    with pytest.raises(FrozenInstanceError):
        question.question = "Changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        observation.turn = 2  # type: ignore[misc]
