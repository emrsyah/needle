import math

import pytest

torch = pytest.importorskip("torch")

from needle.training import (  # noqa: E402
    PROTOCOL_REWARD,
    FailureKind,
    ModelMetadata,
    RolloutRecord,
    RolloutStatus,
)
from needle.training.grpo import (  # noqa: E402
    clipped_surrogate_loss,
    group_advantages,
    is_trainable,
    masked_sequence_mean,
    score_group,
)
from needle.training.rollouts import ActionStep, TokenTrace  # noqa: E402

META = ModelMetadata(model="tiny", provider="local")


@pytest.fixture(autouse=True)
def _deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def test_clipped_loss_matches_hand_computation():
    # Batch of 2, right padded to length 6. Row 0 acts at [3, 4]; row 1 at [2, 3].
    logp = torch.tensor(
        [[-9.0, -9.0, -9.0, -0.5, -1.5, -9.0], [-9.0, -9.0, -0.2, -0.4, -9.0, -9.0]],
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [[False, False, False, True, True, False], [False, False, True, True, False, False]]
    )
    new_mean = masked_sequence_mean(logp, mask)
    assert new_mean.tolist() == pytest.approx([-1.0, -0.3], abs=1e-12)

    # Choose old means so the ratios are exactly 1.5 and 0.5 (both clipped at eps=0.2).
    old_mean = torch.tensor([-1.0 - math.log(1.5), -0.3 - math.log(0.5)], dtype=torch.float64)
    advantages = torch.tensor(group_advantages([1.0, 0.0]), dtype=torch.float64)
    a = 0.5 / (0.5 + 1e-6)
    assert advantages.tolist() == pytest.approx([a, -a], abs=1e-12)

    loss, clip_fraction = clipped_surrogate_loss(new_mean, old_mean, advantages, epsilon=0.2)
    # row 0: min(1.5a, 1.2a) = 1.2a; row 1: min(-0.5a, -0.8a) = -0.8a; loss = -(0.4a / 2)
    assert loss.item() == pytest.approx(-0.2 * a, abs=1e-12)
    assert clip_fraction.item() == 1.0


def test_unclipped_ratio_of_one_is_policy_gradient():
    new_mean = torch.tensor([-1.0, -2.0], dtype=torch.float64, requires_grad=True)
    loss, clip_fraction = clipped_surrogate_loss(
        new_mean, new_mean.detach().clone(), torch.tensor([1.0, -1.0], dtype=torch.float64)
    )
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert clip_fraction.item() == 0.0
    assert new_mean.grad.tolist() == pytest.approx([-0.5, 0.5])


def test_masking_excludes_prompt_and_padding_positions():
    logp = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[False, True, True, False]])
    masked_sequence_mean(logp, mask).sum().backward()
    assert logp.grad.tolist() == [[0.0, 0.5, 0.5, 0.0]]


def test_empty_action_mask_is_rejected():
    with pytest.raises(ValueError):
        masked_sequence_mean(torch.zeros((1, 3)), torch.zeros((1, 3), dtype=torch.bool))


def test_zero_variance_group_has_zero_advantages():
    assert group_advantages([0.3, 0.3, 0.3]) == [0.0, 0.0, 0.0]


def _trace():
    return TokenTrace(
        prompt_text="p",
        prompt_token_ids=(1, 2),
        completion_token_ids=(3, 4),
        action_token_spans=((0, 2),),
        action_token_mask=(True, True),
        old_log_probs=(-0.1, -0.2),
    )


def _step(tokens=True, error=None):
    return ActionStep(
        turn=1,
        prompt_text="p",
        observation_text="(none)",
        raw_action="SEARCH[x]" if tokens else None,
        model_metadata=META if tokens else None,
        tokens=_trace() if tokens else None,
        validator_error=error,
    )


def _invalid():
    return RolloutRecord(
        question_id="q",
        status=RolloutStatus.INVALID,
        steps=(_step(error="bad"),),
        trajectory=None,
        failure_kind=FailureKind.PROTOCOL,
        failure_message="bad",
        training_reward=PROTOCOL_REWARD,
    )


def _transport():
    return RolloutRecord(
        question_id="q",
        status=RolloutStatus.TRANSPORT_ERROR,
        steps=(_step(tokens=False, error="down"),),
        trajectory=None,
        failure_kind=FailureKind.TRANSPORT,
        failure_message="down",
        training_reward=None,
    )


def test_transport_errors_never_train_and_are_excluded_from_normalization():
    assert not is_trainable(_transport(), train_on_invalid=True)
    advantages, stats = score_group([_transport(), _invalid(), _invalid()])
    assert advantages == [0.0, 0.0, 0.0]
    assert stats.skipped_reason == "zero_reward_variance"


def test_invalid_rollouts_train_only_when_enabled():
    assert not is_trainable(_invalid())
    assert is_trainable(_invalid(), train_on_invalid=True)
