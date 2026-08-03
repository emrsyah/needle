"""Pure, deterministic reward metrics for completed Needle episodes."""

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass

from needle.data import EvidenceRef, QuestionExample
from needle.environment import EpisodeTrajectory

_ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Immutable weights for the documented scalar reward aggregation.

    The total is ``quality - penalties``. Information gain is diagnostic only.
    Defaults are deliberately un-tuned baseline weights.
    """

    answer_exact_match_weight: float = 1.0
    answer_f1_weight: float = 1.0
    evidence_coverage_weight: float = 1.0
    citation_precision_weight: float = 1.0
    retrieval_recall_weight: float = 1.0
    search_cost_weight: float = 0.1
    duplicate_query_penalty_weight: float = 0.1


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """All raw reward components, plus the weighted scalar total."""

    answer_exact_match: float
    answer_f1: float
    evidence_coverage: float
    citation_precision: float
    retrieval_recall: float
    search_cost: float
    duplicate_query_penalty: float
    information_gain: tuple[float, ...]
    total: float


def normalize_answer(text: str) -> str:
    """Normalize an answer using the conventional HotpotQA/SQuAD rules."""
    if not isinstance(text, str):
        raise TypeError("answer must be a string")
    text = text.casefold()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _ARTICLE_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_query(query: str) -> str:
    """Normalize a search query for duplicate detection."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    return _WHITESPACE_PATTERN.sub(" ", query.casefold()).strip()


def answer_exact_match(predicted: str, gold: str) -> float:
    """Return 1.0 when normalized answers match, otherwise 0.0."""
    return float(normalize_answer(predicted) == normalize_answer(gold))


def answer_f1(predicted: str, gold: str) -> float:
    """Return token-level F1 for normalized predicted and gold answers."""
    predicted_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = sum(
        min(predicted_tokens.count(token), gold_tokens.count(token))
        for token in set(predicted_tokens)
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evidence_coverage(
    gold_facts: frozenset[EvidenceRef], citations: Sequence[EvidenceRef]
) -> float:
    """Return the fraction of gold supporting facts cited in the answer."""
    if not gold_facts:
        return 0.0
    return len(gold_facts.intersection(citations)) / len(gold_facts)


def citation_precision(
    gold_facts: frozenset[EvidenceRef], citations: Sequence[EvidenceRef]
) -> float:
    """Return the fraction of final citations that are gold facts.

    With gold facts present, no citations scores 0.0. An empty gold set also
    scores 0.0 because this domain requires supporting facts.
    """
    if not gold_facts or not citations:
        return 0.0
    return len(gold_facts.intersection(citations)) / len(citations)


def retrieval_recall(example: QuestionExample, trajectory: EpisodeTrajectory) -> float:
    """Return gold-fact recall by retrieved document title.

    A retrieved document exposes all of its sentences because the current
    retriever returns whole documents rather than sentence passages.
    """
    if not example.supporting_facts:
        return 0.0
    retrieved_titles = {
        result.document.title for search in trajectory.searches for result in search.results
    }
    exposed = {fact for fact in example.supporting_facts if fact.document_title in retrieved_titles}
    return len(exposed) / len(example.supporting_facts)


def search_cost(trajectory: EpisodeTrajectory) -> float:
    """Return the fraction of the configured search budget consumed."""
    budget = trajectory.question.max_searches
    return trajectory.answer.searches_used / budget


def duplicate_query_penalty(trajectory: EpisodeTrajectory) -> float:
    """Return repeated normalized queries divided by total searches."""
    queries = [normalize_query(search.query) for search in trajectory.searches]
    if not queries:
        return 0.0
    return (len(queries) - len(set(queries))) / len(queries)


def information_gain(example: QuestionExample, trajectory: EpisodeTrajectory) -> tuple[float, ...]:
    """Return newly exposed gold-fact fraction after each search step."""
    seen_titles: set[str] = set()
    gains: list[float] = []
    total = len(example.supporting_facts)
    for search in trajectory.searches:
        before = {fact for fact in example.supporting_facts if fact.document_title in seen_titles}
        seen_titles.update(result.document.title for result in search.results)
        after = {fact for fact in example.supporting_facts if fact.document_title in seen_titles}
        gains.append((len(after) - len(before)) / total)
    return tuple(gains)


def evaluate(
    example: QuestionExample,
    trajectory: EpisodeTrajectory,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Evaluate a completed trajectory with pure deterministic functions."""
    if not isinstance(example, QuestionExample):
        raise TypeError("example must be a QuestionExample")
    if not isinstance(trajectory, EpisodeTrajectory):
        raise TypeError("trajectory must be an EpisodeTrajectory")
    if trajectory.question.question_id != example.question_id:
        raise ValueError("trajectory question does not match example")
    if config is None:
        config = RewardConfig()
    if not isinstance(config, RewardConfig):
        raise TypeError("config must be a RewardConfig")

    metrics = dict(
        answer_exact_match=answer_exact_match(trajectory.answer.answer, example.gold_answer),
        answer_f1=answer_f1(trajectory.answer.answer, example.gold_answer),
        evidence_coverage=evidence_coverage(example.supporting_facts, trajectory.answer.citations),
        citation_precision=citation_precision(
            example.supporting_facts, trajectory.answer.citations
        ),
        retrieval_recall=retrieval_recall(example, trajectory),
        search_cost=search_cost(trajectory),
        duplicate_query_penalty=duplicate_query_penalty(trajectory),
        information_gain=information_gain(example, trajectory),
    )
    total = (
        metrics["answer_exact_match"] * config.answer_exact_match_weight
        + metrics["answer_f1"] * config.answer_f1_weight
        + metrics["evidence_coverage"] * config.evidence_coverage_weight
        + metrics["citation_precision"] * config.citation_precision_weight
        + metrics["retrieval_recall"] * config.retrieval_recall_weight
        - metrics["search_cost"] * config.search_cost_weight
        - metrics["duplicate_query_penalty"] * config.duplicate_query_penalty_weight
    )
    return RewardBreakdown(**metrics, total=total)


__all__ = [
    "RewardBreakdown",
    "RewardConfig",
    "answer_exact_match",
    "answer_f1",
    "citation_precision",
    "duplicate_query_penalty",
    "evaluate",
    "evidence_coverage",
    "information_gain",
    "normalize_answer",
    "normalize_query",
    "retrieval_recall",
    "search_cost",
]
