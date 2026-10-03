"""Group-relative advantages and the clipped GRPO policy loss.

Conventions (see docs/plans/2026-09-17-needle-grpo-roadmap.md, Task 8):

- One scalar per rollout: the sequence mean log-probability over every action
  token in every turn of that rollout.
- ``ratio = exp(new_mean - old_mean)``; loss is the negative clipped surrogate.
- Advantages are normalized within each prompt group.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .rollouts import RolloutRecord, RolloutStatus, TokenTrace

ADVANTAGE_EPSILON = 1e-6

LogProbFn = Callable[[TokenTrace], torch.Tensor]


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """Normalize rewards within one group: ``(r - mean) / (std + eps)``.

    Uses the population standard deviation. A zero-variance group yields all zeros.
    """
    if not rewards:
        raise ValueError("rewards must be non-empty")
    values = [float(value) for value in rewards]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if std == 0.0:
        return [0.0] * len(values)
    return [(value - mean) / (std + ADVANTAGE_EPSILON) for value in values]


def masked_sequence_mean(token_log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``token_log_probs`` over positions where ``mask`` is true, per row."""
    if token_log_probs.shape != mask.shape:
        raise ValueError("token_log_probs and mask must have the same shape")
    mask = mask.to(token_log_probs.dtype)
    counts = mask.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("every row needs at least one action token")
    return (token_log_probs * mask).sum(dim=-1) / counts


def clipped_surrogate_loss(
    new_mean: torch.Tensor,
    old_mean: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(loss, clip_fraction)`` for per-rollout sequence means.

    ``loss = -mean(min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A))``.
    """
    ratio = torch.exp(new_mean - old_mean.detach())
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    clip_fraction = ((ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)).to(ratio.dtype).mean()
    return -surrogate.mean(), clip_fraction.detach()


def is_trainable(record: RolloutRecord, *, train_on_invalid: bool = False) -> bool:
    """Whether a rollout contributes a gradient sample.

    Valid rollouts with action tokens always do. Invalid/exhausted rollouts with
    action tokens only do when ``train_on_invalid`` is set. Transport errors never do.
    """
    if record.has_gradient_sample:
        return True
    if not train_on_invalid or record.status not in (
        RolloutStatus.INVALID,
        RolloutStatus.EXHAUSTED,
    ):
        return False
    return any(step.tokens is not None and step.tokens.has_action_tokens for step in record.steps)


def rollout_traces(record: RolloutRecord) -> list[TokenTrace]:
    """Token traces with at least one action token, in turn order."""
    return [
        step.tokens
        for step in record.steps
        if step.tokens is not None and step.tokens.has_action_tokens
    ]


def rollout_mean_log_prob(record: RolloutRecord, log_prob_fn: LogProbFn) -> torch.Tensor:
    """Recompute the rollout's sequence mean over all action tokens in all turns."""
    traces = rollout_traces(record)
    if not traces:
        raise ValueError("rollout has no action tokens")
    total = None
    count = 0
    for trace in traces:
        token_log_probs = log_prob_fn(trace)
        mask = torch.tensor(trace.action_token_mask, device=token_log_probs.device)
        selected = token_log_probs[mask].sum()
        total = selected if total is None else total + selected
        count += int(mask.sum())
    return total / count


def rollout_old_mean(record: RolloutRecord) -> float:
    """Sequence mean of the stored collection-time log-probabilities."""
    values = [value for trace in rollout_traces(record) for value in trace.old_log_probs]
    if not values:
        raise ValueError("rollout has no action tokens")
    return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class GroupStats:
    """Logged statistics for one prompt group."""

    rewards: tuple[float | None, ...]
    advantages: tuple[float, ...]
    reward_mean: float
    reward_std: float
    trainable: int
    skipped_reason: str | None


def score_group(
    records: Sequence[RolloutRecord], *, train_on_invalid: bool = False
) -> tuple[list[float], GroupStats]:
    """Compute advantages for one group.

    Transport errors are excluded from normalization and get advantage 0.
    """
    rewards = [record.training_reward for record in records]
    scored = [(index, reward) for index, reward in enumerate(rewards) if reward is not None]
    advantages = [0.0] * len(records)
    reward_mean = reward_std = 0.0
    if scored:
        values = [float(reward) for _, reward in scored]
        reward_mean = sum(values) / len(values)
        reward_std = math.sqrt(sum((value - reward_mean) ** 2 for value in values) / len(values))
        for (index, _), advantage in zip(scored, group_advantages(values), strict=True):
            advantages[index] = advantage
    trainable = sum(
        is_trainable(record, train_on_invalid=train_on_invalid) and advantages[index] != 0.0
        for index, record in enumerate(records)
    )
    skipped_reason = None
    if not scored:
        skipped_reason = "no_scored_rollouts"
    elif reward_std == 0.0:
        skipped_reason = "zero_reward_variance"
    elif trainable == 0:
        skipped_reason = "no_trainable_rollouts"
    stats = GroupStats(
        rewards=tuple(rewards),
        advantages=tuple(advantages),
        reward_mean=reward_mean,
        reward_std=reward_std,
        trainable=trainable,
        skipped_reason=skipped_reason,
    )
    return advantages, stats


__all__ = [
    "ADVANTAGE_EPSILON",
    "GroupStats",
    "clipped_surrogate_loss",
    "group_advantages",
    "is_trainable",
    "masked_sequence_mean",
    "rollout_mean_log_prob",
    "rollout_old_mean",
    "rollout_traces",
    "score_group",
]
