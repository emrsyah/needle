"""Training boundaries for policy-independent interactive rollouts."""

from .collector import Policy, RolloutCollector
from .rollouts import (
    PROTOCOL_REWARD,
    ActionStep,
    FailureKind,
    ModelMetadata,
    PolicyResponse,
    PolicyTransportError,
    RolloutRecord,
    RolloutStatus,
    TokenTrace,
)

__all__ = [
    "ActionStep",
    "FailureKind",
    "ModelMetadata",
    "Policy",
    "PolicyResponse",
    "PolicyTransportError",
    "PROTOCOL_REWARD",
    "RolloutCollector",
    "RolloutRecord",
    "RolloutStatus",
    "TokenTrace",
]
