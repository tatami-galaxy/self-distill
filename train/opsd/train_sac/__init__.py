"""Online soft actor-critic experiments for privileged self-distillation."""

from .lib import (
    ResidualQHead,
    TopKPolicySupport,
    TopKSoftValueEstimator,
    compute_soft_q_lambda_returns,
    make_soft_value_estimator,
    masked_sequence_mean,
    terminal_token_rewards,
)
from .trainer import SACConfig, SACTrainer

__all__ = [
    "ResidualQHead",
    "SACConfig",
    "SACTrainer",
    "TopKPolicySupport",
    "TopKSoftValueEstimator",
    "compute_soft_q_lambda_returns",
    "make_soft_value_estimator",
    "masked_sequence_mean",
    "terminal_token_rewards",
]
