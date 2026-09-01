"""Memory-conscious token log-probability helpers for causal language models.

The public helpers score one unpadded ``[prompt || completion]`` sequence at a time and return
float32 values.  Keeping this contract explicit avoids shape-dependent bf16 noise when small
teacher/student log-probability differences are the signal, while ``logits_to_keep`` avoids
materializing logits for the whole prompt.
"""

from typing import NamedTuple

import torch


# Rows of the flattened (B*C, V) logit tensor converted to float32 at a time. Bounds the peak
# float32 footprint of the log-softmax to chunk * vocab * 4 bytes (~620MB at 1024 x 151669).
LOGP_CHUNK = 1024
# Entropy materializes more full-vocabulary temporaries than selected-token scoring.
ENTROPY_CHUNK = 256
MODEL_FORWARD_BATCH_SIZE = 1


class PerTokenStats(NamedTuple):
    """Selected-token log-probability and next-token entropy, in nats and float32."""

    logps: torch.Tensor
    entropy: torch.Tensor


def _selective_logps_fp32(
    logits: torch.Tensor, index: torch.Tensor, chunk_size: int = LOGP_CHUNK
) -> torch.Tensor:
    """Return ``log p(index)`` under ``logits``, reduced and returned in float32."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    n_rows, n_tokens, vocab = logits.shape
    flat_logits = logits.reshape(n_rows * n_tokens, vocab)
    flat_index = index.reshape(n_rows * n_tokens, 1)
    chunks = []
    for start in range(0, flat_logits.size(0), chunk_size):
        block = flat_logits[start : start + chunk_size].float()
        selected = block.gather(-1, flat_index[start : start + chunk_size]).squeeze(-1)
        chunks.append(selected - torch.logsumexp(block, dim=-1))
    return torch.cat(chunks).view(n_rows, n_tokens)


def _selective_logps_entropy_fp32(
    logits: torch.Tensor, index: torch.Tensor, chunk_size: int = ENTROPY_CHUNK
) -> PerTokenStats:
    """Return selected-token log-probabilities and categorical entropies in float32."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    n_rows, n_tokens, vocab = logits.shape
    flat_logits = logits.reshape(n_rows * n_tokens, vocab)
    flat_index = index.reshape(n_rows * n_tokens, 1)
    logp_chunks, entropy_chunks = [], []
    for start in range(0, flat_logits.size(0), chunk_size):
        block = flat_logits[start : start + chunk_size].float()
        log_probs = torch.log_softmax(block, dim=-1)
        logp_chunks.append(
            log_probs.gather(-1, flat_index[start : start + chunk_size]).squeeze(-1)
        )
        entropy_chunks.append(-(log_probs.exp() * log_probs).sum(dim=-1))
    return PerTokenStats(
        torch.cat(logp_chunks).view(n_rows, n_tokens),
        torch.cat(entropy_chunks).view(n_rows, n_tokens),
    )


def _completion_logits(
    model,
    input_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    caller: str,
) -> torch.Tensor:
    """Validate the batch-one contract and return logits aligned to completion tokens."""
    if input_ids.ndim != 2 or completion_ids.ndim != 2:
        raise ValueError(f"{caller} expects rank-2 input_ids and completion_ids")
    if (
        input_ids.size(0) != MODEL_FORWARD_BATCH_SIZE
        or completion_ids.size(0) != MODEL_FORWARD_BATCH_SIZE
    ):
        raise ValueError(
            f"{caller} requires physical batch size {MODEL_FORWARD_BATCH_SIZE}; "
            "accumulate gradients instead of padding model inputs"
        )
    n_completion = completion_ids.size(1)
    if input_ids.size(1) <= n_completion:
        raise ValueError("input_ids must contain a non-empty prompt before completion_ids")
    logits = model(
        input_ids=input_ids,
        logits_to_keep=n_completion + 1,
        use_cache=False,
    ).logits
    return logits[:, :-1, :]


def per_token_logps(
    model, input_ids: torch.Tensor, completion_ids: torch.Tensor
) -> torch.Tensor:
    """Log-probabilities of ``completion_ids`` under ``model`` with shape ``(1, C)``.

    Inputs are one unpadded ``[prompt || completion]`` sequence. Position ``i`` predicts token
    ``i+1``, so requesting ``C+1`` trailing logits and dropping the last aligns the remaining
    positions with the completion. Temperature is not applied.
    """
    logits = _completion_logits(model, input_ids, completion_ids, "per_token_logps")
    return _selective_logps_fp32(logits, completion_ids)


def per_token_stats(
    model,
    input_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    entropy_chunk_size: int = ENTROPY_CHUNK,
) -> PerTokenStats:
    """Selected-token log-probabilities and next-token entropies for one completion."""
    logits = _completion_logits(model, input_ids, completion_ids, "per_token_stats")
    return _selective_logps_entropy_fp32(
        logits, completion_ids, chunk_size=entropy_chunk_size
    )
