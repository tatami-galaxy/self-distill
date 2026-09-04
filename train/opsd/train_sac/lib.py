"""Small, testable components for the SDPO-to-SAC experiment."""

from dataclasses import dataclass

import torch
from torch import nn

Q_HEAD_ARCHITECTURE = "linear"


class ResidualQHead(nn.Module):
    """Map a frozen teacher hidden state to a residual hidden state.

    The frozen teacher LM-head row for action ``a`` supplies ``u_a`` outside this
    module, so ``c(s, a) = u_a^T z(h_T(s))``. A zero projection makes the initial
    correction exactly zero without adding an output-scale hyperparameter.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


@dataclass(frozen=True)
class TopKPolicySupport:
    """Student probabilities and full-policy log-probabilities on top-K tokens."""

    token_ids: torch.Tensor
    logps: torch.Tensor
    weights: torch.Tensor
    mass: torch.Tensor


class TopKSoftValueEstimator:
    """Renormalized student-top-K approximation to the soft state value."""

    name = "topk"

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"soft_v_topk must be >= 1; got {k}")
        self.k = k

    def select_token_ids(self, student_logits: torch.Tensor) -> torch.Tensor:
        if self.k > student_logits.size(-1):
            raise ValueError(
                f"soft_v_topk={self.k} exceeds vocabulary size {student_logits.size(-1)}"
            )
        return torch.topk(student_logits.detach(), self.k, dim=-1).indices

    def build_support(
        self, token_ids: torch.Tensor, full_policy_logps: torch.Tensor
    ) -> TopKPolicySupport:
        if token_ids.shape != full_policy_logps.shape:
            raise ValueError("top-K token ids and log-probabilities must have the same shape")
        logps = full_policy_logps.float()
        return TopKPolicySupport(
            token_ids=token_ids,
            logps=logps,
            weights=torch.softmax(logps, dim=-1),
            mass=torch.exp(torch.logsumexp(logps, dim=-1)).clamp(max=1.0),
        )

    def estimate(self, q_values: torch.Tensor, support: TopKPolicySupport) -> torch.Tensor:
        if q_values.shape != support.logps.shape:
            raise ValueError("top-K Q values and policy support must have the same shape")
        return (support.weights * (q_values.float() - support.logps)).sum(dim=-1)


def make_soft_value_estimator(name: str, *, topk: int):
    """Construct a soft-V estimator without coupling it to the trainer
    """

    if name == "topk":
        return TopKSoftValueEstimator(topk)
    if name == "sarsa":
        raise NotImplementedError("sampled SARSA soft-V estimation is not implemented yet")
    raise ValueError(f"unknown soft-V estimator {name!r}; expected 'topk' or 'sarsa'")


def terminal_token_rewards(
    rewards: torch.Tensor, completion_mask: torch.Tensor
) -> torch.Tensor:
    """Place each scalar outcome on the last real token of its completion."""

    if rewards.ndim != 1 or completion_mask.ndim != 2:
        raise ValueError("rewards must be (B,) and completion_mask must be (B, T)")
    if rewards.size(0) != completion_mask.size(0):
        raise ValueError("reward and completion batch sizes differ")

    token_rewards = torch.zeros_like(completion_mask, dtype=rewards.dtype)
    lengths = completion_mask.sum(dim=1).long()
    valid = lengths > 0
    if valid.any():
        rows = torch.arange(rewards.size(0), device=rewards.device)[valid]
        token_rewards[rows, lengths[valid] - 1] = rewards[valid]
    return token_rewards * completion_mask.to(rewards.dtype)


def compute_soft_q_lambda_returns(
    token_rewards: torch.Tensor,
    soft_values: torch.Tensor,
    sampled_policy_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    lam: float,
    gamma: float = 1.0,
) -> torch.Tensor:
    r"""Compute detached forward-view lambda targets for a soft action value.

    For a non-terminal token, the backward recursion is

    ``G_t = r_t + gamma * ((1-lam) V_{t+1}
                           + lam (G_{t+1} - log pi(a_{t+1}|s_{t+1})))``.

    Q excludes the entropy of its current action; that entropy enters when the next
    sampled action is used to extend the return. Right padding marks episode end.
    """

    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1]; got {lam}")
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative; got {gamma}")
    shape = token_rewards.shape
    if any(tensor.shape != shape for tensor in (soft_values, sampled_policy_logps, completion_mask)):
        raise ValueError("all return tensors must have the same (B, T) shape")
    if token_rewards.ndim != 2:
        raise ValueError("return tensors must be rank two")

    rewards = token_rewards.float()
    values = soft_values.float()
    logps = sampled_policy_logps.float()
    mask = completion_mask.bool()
    returns = torch.zeros_like(rewards)
    next_return = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)

    for t in range(rewards.size(1) - 1, -1, -1):
        if t + 1 < rewards.size(1):
            has_next = mask[:, t + 1]
            continuation = (1.0 - lam) * values[:, t + 1] + lam * (
                next_return - logps[:, t + 1]
            )
            continuation = torch.where(has_next, continuation, torch.zeros_like(continuation))
        else:
            continuation = torch.zeros_like(next_return)
        current = rewards[:, t] + gamma * continuation
        current = torch.where(mask[:, t], current, torch.zeros_like(current))
        returns[:, t] = current
        next_return = current
    return returns


def masked_sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over tokens per sequence, then over sequences containing a real token."""

    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("values and mask must have the same (B, T) shape")
    mask = mask.to(values.dtype)
    counts = mask.sum(dim=1)
    valid = counts > 0
    sequence_means = (values * mask).sum(dim=1) / counts.clamp(min=1.0)
    if not valid.any():
        return values.sum() * 0.0
    return sequence_means[valid].mean()
