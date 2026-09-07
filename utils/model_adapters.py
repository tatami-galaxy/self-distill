"""Resolve locally saved PEFT LoRA adapters for vLLM inference.

Transformers/PEFT can infer the base model from ``adapter_config.json``, but
vLLM's offline API needs the base model and adapter request separately. Keep
that translation in one place so evaluation and cache generation load hint
generators identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


@dataclass(frozen=True)
class ModelAdapterSpec:
    """A requested standalone model or a local LoRA adapter over a base model."""

    requested_model: str
    base_model: str
    adapter_path: str | None = None
    rank: int | None = None

    @property
    def is_adapter(self) -> bool:
        return self.adapter_path is not None


def resolve_model_adapter(model: str) -> ModelAdapterSpec:
    """Return the base-model/adapter decomposition for a local PEFT path."""

    adapter_dir = Path(model).expanduser()
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        return ModelAdapterSpec(requested_model=model, base_model=model)

    try:
        config: dict[str, Any] = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read PEFT adapter config {config_path}: {error}"
        ) from error

    peft_type = str(config.get("peft_type", "")).upper()
    if peft_type != "LORA":
        raise ValueError(
            f"{config_path} describes PEFT type {peft_type or '<missing>'!r}; "
            "only LoRA hint-generator adapters are supported by vLLM."
        )
    base_model = str(config.get("base_model_name_or_path", "")).strip()
    if not base_model:
        raise ValueError(f"{config_path} does not record base_model_name_or_path.")

    ranks = [config.get("r")]
    rank_pattern = config.get("rank_pattern") or {}
    if not isinstance(rank_pattern, dict):
        raise TypeError(f"{config_path} has a non-object rank_pattern.")
    ranks.extend(rank_pattern.values())
    try:
        rank = max(int(value) for value in ranks if value is not None)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{config_path} contains an invalid LoRA rank.") from error
    if rank < 1:
        raise ValueError(f"{config_path} contains a non-positive LoRA rank {rank}.")

    return ModelAdapterSpec(
        requested_model=model,
        base_model=base_model,
        adapter_path=str(adapter_dir.resolve()),
        rank=rank,
    )


def _vllm_max_lora_rank(rank: int) -> int:
    for supported_rank in VLLM_LORA_RANKS:
        if rank <= supported_rank:
            return supported_rank
    raise ValueError(
        f"LoRA rank {rank} exceeds vLLM's maximum supported rank {VLLM_LORA_RANKS[-1]}."
    )


def vllm_model_and_adapter(
    model: str,
    *,
    adapter_name: str = "hint-generator",
    adapter_id: int = 1,
) -> tuple[dict[str, Any], Any | None, ModelAdapterSpec]:
    """Return ``LLM`` model kwargs and the matching optional ``LoRARequest``."""

    spec = resolve_model_adapter(model)
    if not spec.is_adapter:
        return {"model": spec.base_model}, None, spec

    from vllm.lora.request import LoRARequest

    model_kwargs = {
        "model": spec.base_model,
        "enable_lora": True,
        "max_lora_rank": _vllm_max_lora_rank(spec.rank),
    }
    request = LoRARequest(
        adapter_name,
        adapter_id,
        spec.adapter_path,
        base_model_name=spec.base_model,
    )
    return model_kwargs, request, spec
