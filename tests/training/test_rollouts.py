import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from needle.data import load_hotpotqa
from needle.training import (
    FailureKind,
    ModelMetadata,
    PolicyResponse,
    PolicyTransportError,
    RolloutCollector,
    RolloutStatus,
    TokenTrace,
)

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"


def _example():
    return load_hotpotqa(FIXTURE_PATH)[0]


def _token_trace(prompt: str) -> TokenTrace:
    return TokenTrace(
        prompt_text=prompt,
        prompt_token_ids=(10, 11),
        completion_token_ids=(20, 21),
        action_token_spans=((0, 2),),
        action_token_mask=(True, True),
        old_log_probs=(-0.1, -0.2),
    )


class SequencePolicy:
    def __init__(self, actions: tuple[str, ...], *, with_tokens: bool = False) -> None:
        self.actions = iter(actions)
        self.with_tokens = with_tokens
        self.calls: list[tuple[str, str]] = []

    def act(self, prompt_text: str, observation_text: str) -> PolicyResponse:
        self.calls.append((prompt_text, observation_text))
        action = next(self.actions)
        return PolicyResponse(
            action,
            ModelMetadata("fake-policy", "test", request_id=f"request-{len(self.calls)}"),
            _token_trace(prompt_text) if self.with_tokens else None,
        )


def test_token_trace_excludes_prompt_and_padding_from_action_tokens() -> None:
    trace = TokenTrace(
        prompt_text="prompt",
        prompt_token_ids=(1, 2, 3),
        completion_token_ids=(4, 5, 6, 7),
        action_token_spans=((1, 3),),
        action_token_mask=(False, True, True, False),
        old_log_probs=(-1.0, -2.0),
    )

    assert trace.has_action_tokens is True
    assert trace.to_json_dict()["prompt_token_ids"] == [1, 2, 3]
    assert trace.to_json_dict()["completion_token_ids"] == [4, 5, 6, 7]
    with pytest.raises(ValueError, match="exactly describe"):
        TokenTrace(
            prompt_text="prompt",
            prompt_token_ids=(1,),
            completion_token_ids=(2, 3),
            action_token_spans=((0, 1),),
            action_token_mask=(False, True),
            old_log_probs=(-1.0,),
        )


def test_collector_produces_valid_rollout_and_json_safe_record() -> None:
    policy = SequencePolicy(
        ("SEARCH[Ursula]", "ANSWER[United States] CITATIONS[Ursula K. Le Guin|1]"),
        with_tokens=True,
    )
    record = RolloutCollector(policy, top_k=1, max_searches=1).collect(_example())

    assert record.status is RolloutStatus.VALID
    assert record.failure_kind is None
    assert record.training_reward == pytest.approx(3.9)
    assert record.reward_breakdown is not None
    assert record.trajectory is not None
    assert record.has_gradient_sample is True
    assert record.steps[0].observation_text == "(none)"
    assert "Search 1: Ursula" in record.steps[1].observation_text
    assert record.steps[1].validator_error is None

    serialized = json.loads(record.to_json())
    assert serialized["status"] == "valid"
    assert serialized["steps"][0]["tokens"]["prompt_text"] == policy.calls[0][0]
    assert serialized["steps"][0]["tokens"]["completion_token_ids"] == [20, 21]
    with pytest.raises(FrozenInstanceError):
        record.status = RolloutStatus.INVALID  # type: ignore[misc]


def test_collector_records_protocol_failure_with_fixed_reward() -> None:
    policy = SequencePolicy(("not an action",))
    record = RolloutCollector(policy, max_searches=1).collect(_example())

    assert record.status is RolloutStatus.INVALID
    assert record.failure_kind is FailureKind.PROTOCOL
    assert record.training_reward == -1.0
    assert record.trajectory is None
    assert record.steps[0].raw_action == "not an action"
    assert record.steps[0].validator_error is not None
    assert record.has_gradient_sample is False


def test_collector_records_exhausted_search_without_retry() -> None:
    policy = SequencePolicy(("SEARCH[Ursula]", "SEARCH[again]"))
    record = RolloutCollector(policy, top_k=1, max_searches=1).collect(_example())

    assert record.status is RolloutStatus.EXHAUSTED
    assert record.failure_kind is FailureKind.BUDGET
    assert record.training_reward == -1.0
    assert record.steps[-1].validator_error == "search budget is exhausted"
    assert len(policy.calls) == 2


def test_collector_transport_failure_has_no_gradient_sample_or_reward() -> None:
    class BrokenPolicy:
        def act(self, prompt_text: str, observation_text: str) -> PolicyResponse:
            raise PolicyTransportError("gateway unavailable")

    record = RolloutCollector(BrokenPolicy(), max_searches=1).collect(_example())

    assert record.status is RolloutStatus.TRANSPORT_ERROR
    assert record.failure_kind is FailureKind.TRANSPORT
    assert record.training_reward is None
    assert record.steps[0].raw_action is None
    assert record.steps[0].validator_error == "gateway unavailable"
    assert record.has_gradient_sample is False


def test_collector_keeps_citation_validation_in_environment_boundary() -> None:
    policy = SequencePolicy(("SEARCH[Ursula]", "ANSWER[United States] CITATIONS[Unknown|0]"))
    record = RolloutCollector(policy, top_k=1, max_searches=1).collect(_example())

    assert record.status is RolloutStatus.INVALID
    assert record.failure_kind is FailureKind.EVIDENCE
    assert record.training_reward == -1.0
    assert "unknown document" in (record.failure_message or "")
