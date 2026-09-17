"""Immutable, framework-independent records for interactive training rollouts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from needle.environment import EpisodeTrajectory
from needle.rewards import RewardBreakdown

JsonScalar: TypeAlias = str | int | float | bool | None


class RolloutStatus(StrEnum):
    """Terminal status of one policy/environment interaction."""

    VALID = "valid"
    INVALID = "invalid"
    EXHAUSTED = "exhausted"
    TRANSPORT_ERROR = "transport_error"


class FailureKind(StrEnum):
    """Coarse failure categories retained for analysis."""

    PROTOCOL = "protocol"
    BUDGET = "budget"
    TRANSPORT = "transport"
    RETRIEVAL = "retrieval"
    ANSWER = "answer"
    EVIDENCE = "evidence"


PROTOCOL_REWARD = -1.0


class PolicyTransportError(RuntimeError):
    """A policy could not produce an action because its transport failed."""


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Small immutable model/provider metadata attached to one generated action."""

    model: str
    provider: str
    request_id: str | None = None
    extras: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model metadata model must be a non-empty string")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("model metadata provider must be a non-empty string")
        if self.request_id is not None and not isinstance(self.request_id, str):
            raise TypeError("model metadata request_id must be a string or None")
        if not isinstance(self.extras, tuple):
            raise TypeError("model metadata extras must be a tuple")
        keys: set[str] = set()
        for item in self.extras:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("model metadata extras must contain (key, value) tuples")
            key, value = item
            if not isinstance(key, str) or not key:
                raise ValueError("model metadata extra keys must be non-empty strings")
            if key in keys:
                raise ValueError(f"duplicate model metadata key: {key!r}")
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                raise TypeError("model metadata extras must contain JSON scalar values")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("model metadata extras must contain finite floats")
            keys.add(key)

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return {
            "model": self.model,
            "provider": self.provider,
            "request_id": self.request_id,
            "extras": {key: value for key, value in self.extras},
        }


@dataclass(frozen=True, slots=True)
class TokenTrace:
    """Token data for one turn, serialized independently from trainer tensors.

    ``prompt_token_ids`` and ``completion_token_ids`` are intentionally separate.
    Action spans are half-open offsets into ``completion_token_ids`` and must agree
    with ``action_token_mask``. ``old_log_probs`` is already detached and contains
    one value for each ``True`` mask entry, in completion-token order.

    The canonical causal-LM alignment is documented by the stored offsets: with
    ``logits[:, :-1] -> input_ids[:, 1:]``, completion token ``j`` is scored at
    full target index ``prompt_length + j`` and logit index ``prompt_length + j - 1``.
    """

    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    action_token_spans: tuple[tuple[int, int], ...]
    action_token_mask: tuple[bool, ...]
    old_log_probs: tuple[float, ...]
    padding_side: str = "right"

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_text, str):
            raise TypeError("prompt_text must be a string")
        if not isinstance(self.padding_side, str) or self.padding_side != "right":
            raise ValueError("padding_side must be 'right'")
        if not isinstance(self.prompt_token_ids, tuple) or not isinstance(
            self.completion_token_ids, tuple
        ):
            raise TypeError("token IDs must be tuples")
        for name, token_ids in (
            ("prompt_token_ids", self.prompt_token_ids),
            ("completion_token_ids", self.completion_token_ids),
        ):
            if any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in token_ids
            ):
                raise TypeError(f"{name} must contain integer token IDs")
        if not isinstance(self.action_token_mask, tuple) or any(
            not isinstance(value, bool) for value in self.action_token_mask
        ):
            raise TypeError("action_token_mask must be a tuple of booleans")
        if len(self.action_token_mask) != len(self.completion_token_ids):
            raise ValueError("action_token_mask must match completion_token_ids length")
        if not isinstance(self.action_token_spans, tuple):
            raise TypeError("action_token_spans must be a tuple")
        expected_mask = [False] * len(self.completion_token_ids)
        previous_end = 0
        for span in self.action_token_spans:
            if not isinstance(span, tuple) or len(span) != 2:
                raise TypeError("action_token_spans must contain (start, end) tuples")
            start, end = span
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < previous_end
                or start < 0
                or end <= start
                or end > len(self.completion_token_ids)
            ):
                raise ValueError(
                    "action_token_spans must be sorted, non-overlapping, and in bounds"
                )
            for index in range(start, end):
                expected_mask[index] = True
            previous_end = end
        if tuple(expected_mask) != self.action_token_mask:
            raise ValueError("action_token_spans must exactly describe action_token_mask")
        if not isinstance(self.old_log_probs, tuple):
            raise TypeError("old_log_probs must be a tuple")
        if len(self.old_log_probs) != sum(self.action_token_mask):
            raise ValueError("old_log_probs must contain one value per action token")
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in self.old_log_probs
        ):
            raise TypeError("old_log_probs must contain numbers")
        if any(not math.isfinite(float(value)) for value in self.old_log_probs):
            raise ValueError("old_log_probs must contain finite numbers")

    @property
    def has_action_tokens(self) -> bool:
        """Whether this trace can contribute a token-level policy sample."""
        return any(self.action_token_mask)

    @property
    def generated_token_ids(self) -> tuple[int, ...]:
        """Alias emphasizing that completion IDs are the generated tokens."""
        return self.completion_token_ids

    @property
    def old_logp(self) -> tuple[float, ...]:
        """Alias for the detached old log-probabilities used by the trainer."""
        return self.old_log_probs

    def to_json_dict(self) -> dict[str, object]:
        """Return the documented inline JSON tensor-sidecar representation."""
        return {
            "serialization": "inline_json_v1",
            "prompt_text": self.prompt_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "completion_token_ids": list(self.completion_token_ids),
            "action_token_spans": [list(span) for span in self.action_token_spans],
            "action_token_mask": list(self.action_token_mask),
            "old_log_probs": list(self.old_log_probs),
            "padding_side": self.padding_side,
        }


@dataclass(frozen=True, slots=True)
class PolicyResponse:
    """One raw policy action plus optional token data from the local adapter."""

    action_text: str
    model_metadata: ModelMetadata
    tokens: TokenTrace | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_text, str):
            raise TypeError("action_text must be a string")
        if not isinstance(self.model_metadata, ModelMetadata):
            raise TypeError("model_metadata must be ModelMetadata")
        if self.tokens is not None and not isinstance(self.tokens, TokenTrace):
            raise TypeError("tokens must be TokenTrace or None")


@dataclass(frozen=True, slots=True)
class ActionStep:
    """One prompt/observation and its raw model response or validator error."""

    turn: int
    prompt_text: str
    observation_text: str
    raw_action: str | None
    model_metadata: ModelMetadata | None = None
    tokens: TokenTrace | None = None
    validator_error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.turn, bool) or not isinstance(self.turn, int) or self.turn <= 0:
            raise ValueError("turn must be a positive integer")
        if not isinstance(self.prompt_text, str) or not isinstance(self.observation_text, str):
            raise TypeError("prompt_text and observation_text must be strings")
        if self.raw_action is not None and not isinstance(self.raw_action, str):
            raise TypeError("raw_action must be a string or None")
        if self.model_metadata is not None and not isinstance(self.model_metadata, ModelMetadata):
            raise TypeError("model_metadata must be ModelMetadata or None")
        if self.tokens is not None and not isinstance(self.tokens, TokenTrace):
            raise TypeError("tokens must be TokenTrace or None")
        if self.raw_action is None and self.tokens is not None:
            raise ValueError("token data requires a raw action")
        if self.validator_error is not None and (
            not isinstance(self.validator_error, str) or not self.validator_error
        ):
            raise ValueError("validator_error must be a non-empty string or None")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "turn": self.turn,
            "prompt_text": self.prompt_text,
            "observation_text": self.observation_text,
            "raw_action": self.raw_action,
            "model_metadata": (
                self.model_metadata.to_json_dict() if self.model_metadata is not None else None
            ),
            "tokens": self.tokens.to_json_dict() if self.tokens is not None else None,
            "validator_error": self.validator_error,
        }


def _trajectory_to_json_dict(trajectory: EpisodeTrajectory) -> dict[str, object]:
    return {
        "question": {
            "question_id": trajectory.question.question_id,
            "question": trajectory.question.question,
            "max_searches": trajectory.question.max_searches,
        },
        "searches": [
            {
                "turn": search.turn,
                "query": search.query,
                "remaining_searches": search.remaining_searches,
                "results": [
                    {
                        "rank": result.rank,
                        "score": result.score,
                        "document": {
                            "title": result.document.title,
                            "sentences": list(result.document.sentences),
                        },
                    }
                    for result in search.results
                ],
            }
            for search in trajectory.searches
        ],
        "answer": {
            "answer": trajectory.answer.answer,
            "citations": [
                {
                    "document_title": citation.document_title,
                    "sentence_index": citation.sentence_index,
                }
                for citation in trajectory.answer.citations
            ],
            "searches_used": trajectory.answer.searches_used,
        },
    }


def _reward_to_json_dict(reward: RewardBreakdown) -> dict[str, object]:
    return {
        "answer_exact_match": reward.answer_exact_match,
        "answer_f1": reward.answer_f1,
        "evidence_coverage": reward.evidence_coverage,
        "citation_precision": reward.citation_precision,
        "retrieval_recall": reward.retrieval_recall,
        "search_cost": reward.search_cost,
        "duplicate_query_penalty": reward.duplicate_query_penalty,
        "information_gain": list(reward.information_gain),
        "total": reward.total,
    }


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    """Immutable offline-scored rollout and its explicit training outcome."""

    question_id: str
    status: RolloutStatus
    steps: tuple[ActionStep, ...]
    trajectory: EpisodeTrajectory | None
    failure_kind: FailureKind | None
    failure_message: str | None
    training_reward: float | None
    reward_breakdown: RewardBreakdown | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.question_id, str) or not self.question_id:
            raise ValueError("question_id must be a non-empty string")
        if not isinstance(self.status, RolloutStatus):
            raise TypeError("status must be RolloutStatus")
        if not isinstance(self.steps, tuple) or any(
            not isinstance(step, ActionStep) for step in self.steps
        ):
            raise TypeError("steps must be a tuple of ActionStep")
        if self.trajectory is not None and not isinstance(self.trajectory, EpisodeTrajectory):
            raise TypeError("trajectory must be EpisodeTrajectory or None")
        if self.trajectory is not None and self.trajectory.question.question_id != self.question_id:
            raise ValueError("trajectory question does not match question_id")
        if self.reward_breakdown is not None and not isinstance(
            self.reward_breakdown, RewardBreakdown
        ):
            raise TypeError("reward_breakdown must be RewardBreakdown or None")
        if self.failure_kind is not None and not isinstance(self.failure_kind, FailureKind):
            raise TypeError("failure_kind must be FailureKind or None")
        if self.failure_message is not None and (
            not isinstance(self.failure_message, str) or not self.failure_message
        ):
            raise ValueError("failure_message must be a non-empty string or None")
        if self.training_reward is not None:
            if not isinstance(self.training_reward, (int, float)) or isinstance(
                self.training_reward, bool
            ):
                raise TypeError("training_reward must be a number or None")
            if not math.isfinite(float(self.training_reward)):
                raise ValueError("training_reward must be finite")

        if self.status is RolloutStatus.VALID:
            if self.trajectory is None or self.failure_kind is not None:
                raise ValueError("valid rollouts require a trajectory and no failure")
            if self.training_reward is None or self.reward_breakdown is None:
                raise ValueError("valid rollouts require reward and reward_breakdown")
            if self.failure_message is not None:
                raise ValueError("valid rollouts cannot have a failure message")
            if self.training_reward != self.reward_breakdown.total:
                raise ValueError("training_reward must equal reward_breakdown.total")
        elif self.status is RolloutStatus.TRANSPORT_ERROR:
            if self.trajectory is not None or self.training_reward is not None:
                raise ValueError("transport errors cannot have a trajectory or training reward")
            if self.failure_kind is not FailureKind.TRANSPORT:
                raise ValueError("transport errors require transport failure_kind")
            if not self.failure_message:
                raise ValueError("transport errors require a failure message")
            if self.reward_breakdown is not None:
                raise ValueError("transport errors cannot have a reward breakdown")
        else:
            if self.trajectory is not None or self.training_reward != PROTOCOL_REWARD:
                raise ValueError("invalid/exhausted rollouts require the protocol reward")
            if self.failure_kind is None:
                raise ValueError("invalid/exhausted rollouts require failure_kind")
            if not self.failure_message:
                raise ValueError("invalid/exhausted rollouts require a failure message")
            if self.reward_breakdown is not None:
                raise ValueError("invalid/exhausted rollouts cannot have a reward breakdown")
            if (
                self.status is RolloutStatus.EXHAUSTED
                and self.failure_kind is not FailureKind.BUDGET
            ):
                raise ValueError("exhausted rollouts require budget failure_kind")

    @property
    def has_gradient_sample(self) -> bool:
        """Whether this record contains valid action tokens for optimization."""
        return self.status is RolloutStatus.VALID and any(
            step.tokens is not None and step.tokens.has_action_tokens for step in self.steps
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return JSON-safe metadata plus inline token sidecars."""
        return {
            "question_id": self.question_id,
            "status": self.status.value,
            "steps": [step.to_json_dict() for step in self.steps],
            "trajectory": (
                _trajectory_to_json_dict(self.trajectory) if self.trajectory is not None else None
            ),
            "failure_kind": self.failure_kind.value if self.failure_kind is not None else None,
            "failure_message": self.failure_message,
            "training_reward": self.training_reward,
            "reward_breakdown": (
                _reward_to_json_dict(self.reward_breakdown)
                if self.reward_breakdown is not None
                else None
            ),
        }

    def to_json(self) -> str:
        """Serialize one rollout as deterministic JSON for JSONL storage."""
        return json.dumps(self.to_json_dict(), ensure_ascii=False, sort_keys=True)


__all__ = [
    "ActionStep",
    "FailureKind",
    "ModelMetadata",
    "PolicyResponse",
    "PolicyTransportError",
    "PROTOCOL_REWARD",
    "RolloutRecord",
    "RolloutStatus",
    "TokenTrace",
]
