"""Focused tests for the deterministic reward evaluator."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from needle.data import EvidenceRef, load_hotpotqa
from needle.environment import SearchEnvironment
from needle.rewards import (
    RewardConfig,
    answer_exact_match,
    answer_f1,
    citation_precision,
    duplicate_query_penalty,
    evaluate,
    evidence_coverage,
    information_gain,
    normalize_answer,
)

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"


def make_episode(
    *,
    answer: str = "United States",
    citations: tuple[EvidenceRef, ...] = (),
    queries: tuple[str, ...] = (),
):
    example = load_hotpotqa(FIXTURE_PATH)[0]
    environment = SearchEnvironment(example, top_k=1, max_searches=2)
    environment.reset()
    for query in queries:
        environment.search(query)
    environment.answer(answer, citations)
    return example, environment.trajectory


def test_normalization_and_answer_metrics_follow_hotpotqa_conventions() -> None:
    assert normalize_answer(" The United-States! ") == "unitedstates"
    assert answer_exact_match("The United States", "united states") == 1.0
    assert answer_f1("the United States", "United States") == 1.0
    assert answer_f1("United", "United States") == pytest.approx(2 / 3)


def test_evidence_metrics_define_precision_and_partial_coverage() -> None:
    gold = frozenset({EvidenceRef("A", 0), EvidenceRef("B", 1)})
    assert evidence_coverage(gold, (EvidenceRef("A", 0),)) == 0.5
    assert citation_precision(gold, (EvidenceRef("A", 0), EvidenceRef("wrong", 0))) == 0.5
    assert citation_precision(gold, ()) == 0.0


def test_scripted_end_to_end_episode_has_exact_breakdown() -> None:
    example = load_hotpotqa(FIXTURE_PATH)[0]
    facts = tuple(sorted(example.supporting_facts, key=lambda fact: fact.document_title))
    example, trajectory = make_episode(
        citations=facts,
        queries=("Ursula", "The Left Hand of Darkness"),
    )

    breakdown = evaluate(example, trajectory)

    assert breakdown.answer_exact_match == 1.0
    assert breakdown.answer_f1 == 1.0
    assert breakdown.evidence_coverage == 1.0
    assert breakdown.citation_precision == 1.0
    assert breakdown.retrieval_recall == 1.0
    assert breakdown.search_cost == 1.0
    assert breakdown.duplicate_query_penalty == 0.0
    assert breakdown.information_gain == (0.5, 0.5)
    assert breakdown.total == pytest.approx(4.9)


@pytest.mark.parametrize(
    ("answer", "citations", "queries", "expected"),
    [
        ("Canada", (EvidenceRef("Ursula K. Le Guin", 1),), (), (0.0, 0.5, 1.0, 0.0)),
        ("United States", (EvidenceRef("Berkeley", 0),), (), (1.0, 0.0, 0.0, 0.0)),
        (
            "United States",
            (EvidenceRef("Ursula K. Le Guin", 1),),
            (),
            (1.0, 0.5, 1.0, 0.0),
        ),
        ("United States", (), (), (1.0, 0.0, 0.0, 0.0)),
        (
            "United States",
            (EvidenceRef("Ursula K. Le Guin", 1),),
            ("Ursula",),
            (1.0, 0.5, 1.0, 0.5),
        ),
    ],
)
def test_adversarial_answer_evidence_and_partial_retrieval_cases(
    answer: str,
    citations: tuple[EvidenceRef, ...],
    queries: tuple[str, ...],
    expected: tuple[float, float, float, float],
) -> None:
    example, trajectory = make_episode(answer=answer, citations=citations, queries=queries)
    breakdown = evaluate(example, trajectory)

    assert (
        breakdown.answer_exact_match,
        breakdown.evidence_coverage,
        breakdown.citation_precision,
        breakdown.retrieval_recall,
    ) == expected


def test_duplicate_search_is_normalized_and_information_gain_is_diagnostic() -> None:
    example, trajectory = make_episode(
        citations=(EvidenceRef("Ursula K. Le Guin", 1),),
        queries=(" Ursula ", "ursula"),
    )

    breakdown = evaluate(example, trajectory)

    assert duplicate_query_penalty(trajectory) == 0.5
    assert information_gain(example, trajectory) == (0.5, 0.0)
    assert breakdown.information_gain == (0.5, 0.0)
    assert breakdown.total == pytest.approx(3.85)


def test_reward_records_are_immutable() -> None:
    example, trajectory = make_episode()
    breakdown = evaluate(example, trajectory)

    with pytest.raises(FrozenInstanceError):
        breakdown.total = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        RewardConfig().search_cost_weight = 0.0  # type: ignore[misc]


def test_gated_penalties_apply_only_to_correct_answers() -> None:
    gated = RewardConfig(gate_penalties_on_correct=True)
    queries = (" Ursula ", "ursula")
    correct_example, correct = make_episode(queries=queries)
    wrong_example, wrong = make_episode(answer="Canada", queries=queries)

    assert evaluate(correct_example, correct, gated).total == pytest.approx(
        evaluate(correct_example, correct).total
    )
    wrong_breakdown = evaluate(wrong_example, wrong, gated)
    assert wrong_breakdown.answer_exact_match == 0.0
    assert wrong_breakdown.total == pytest.approx(
        wrong_breakdown.answer_f1
        + wrong_breakdown.evidence_coverage
        + wrong_breakdown.citation_precision
        + wrong_breakdown.retrieval_recall
    )
    assert evaluate(wrong_example, wrong).total < wrong_breakdown.total


def test_custom_weights_do_not_include_information_gain() -> None:
    example, trajectory = make_episode(
        citations=(EvidenceRef("Ursula K. Le Guin", 1),), queries=("Ursula",)
    )
    breakdown = evaluate(
        example,
        trajectory,
        RewardConfig(
            answer_exact_match_weight=0.0,
            answer_f1_weight=0.0,
            evidence_coverage_weight=0.0,
            citation_precision_weight=0.0,
            retrieval_recall_weight=0.0,
            search_cost_weight=0.0,
            duplicate_query_penalty_weight=0.0,
        ),
    )
    assert breakdown.information_gain == (0.5,)
    assert breakdown.total == 0.0
