"""Tests for terminal answers and completed search trajectories."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from needle.data import EvidenceRef, load_hotpotqa
from needle.environment import (
    AnswerObservation,
    EpisodeTrajectory,
    SearchEnvironment,
    SearchEnvironmentError,
)

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"


def make_environment(max_searches: int = 2, top_k: int = 3) -> SearchEnvironment:
    return SearchEnvironment(load_hotpotqa(FIXTURE_PATH)[0], max_searches=max_searches, top_k=top_k)


def test_answer_requires_reset_without_changing_initial_state() -> None:
    environment = make_environment()

    with pytest.raises(SearchEnvironmentError, match="reset"):
        environment.answer("United States")

    assert environment.terminated is False
    assert environment.answer_observation is None
    assert environment.trajectory is None


def test_zero_search_answer_creates_a_terminal_immutable_trajectory() -> None:
    environment = make_environment()
    question = environment.reset()

    answer = environment.answer("  United States  ")

    assert answer == AnswerObservation("United States", (), searches_used=0)
    assert environment.terminated is True
    assert environment.answer_observation is answer
    assert environment.history == ()
    assert environment.trajectory == EpisodeTrajectory(question, (), answer)
    with pytest.raises(FrozenInstanceError):
        answer.answer = "Canada"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        environment.trajectory.answer = answer  # type: ignore[union-attr,misc]


def test_answer_accepts_retrieved_and_unretrieved_corpus_citations() -> None:
    environment = make_environment(top_k=1)
    question = environment.reset()
    search = environment.search("Ursula")
    retrieved = EvidenceRef("Ursula K. Le Guin", 1)
    unretrieved = EvidenceRef("Berkeley", 1)

    answer = environment.answer("United States", [retrieved, unretrieved])

    assert search.results[0].document.title == retrieved.document_title
    assert all(result.document.title != unretrieved.document_title for result in search.results)
    assert answer.citations == (retrieved, unretrieved)
    assert environment.trajectory == EpisodeTrajectory(question, (search,), answer)


@pytest.mark.parametrize(
    ("answer", "citations"),
    [
        ("", ()),
        ("   ", ()),
        (None, ()),
        ("United States", "not citations"),
        ("United States", ["not an EvidenceRef"]),
        ("United States", [EvidenceRef("Unknown", 0)]),
        ("United States", [EvidenceRef("Ursula K. Le Guin", 2)]),
        ("United States", [EvidenceRef("Ursula K. Le Guin", 1)] * 2),
    ],
)
def test_invalid_answers_do_not_change_active_session(answer: object, citations: object) -> None:
    environment = make_environment()
    environment.reset()
    search = environment.search("Ursula")

    with pytest.raises(SearchEnvironmentError):
        environment.answer(answer, citations)  # type: ignore[arg-type]

    assert environment.history == (search,)
    assert environment.remaining_searches == 1
    assert environment.terminated is False
    assert environment.answer_observation is None
    assert environment.trajectory is None


def test_terminal_answer_rejects_further_searches_or_answers() -> None:
    environment = make_environment()
    environment.reset()
    environment.answer("United States")

    with pytest.raises(SearchEnvironmentError, match="terminated"):
        environment.search("Ursula")
    with pytest.raises(SearchEnvironmentError, match="terminated"):
        environment.answer("Canada")


def test_answer_is_allowed_after_search_budget_is_exhausted() -> None:
    environment = make_environment(max_searches=1)
    environment.reset()
    environment.search("Ursula")

    answer = environment.answer("United States")

    assert answer.searches_used == 1
    assert environment.terminated is True


def test_reset_clears_previous_terminal_state_and_search_history() -> None:
    environment = make_environment()
    environment.reset()
    environment.search("Ursula")
    environment.answer("United States")

    question = environment.reset()

    assert question.question_id == environment.example.question_id
    assert environment.history == ()
    assert environment.terminated is False
    assert environment.answer_observation is None
    assert environment.trajectory is None
