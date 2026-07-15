"""
Train the PPO baseline -- the actor-critic RL contrast to GRPO. Where GRPO scores
a rollout against a *group-relative* mean baseline (no critic), PPO learns a value
function and estimates per-token advantages with GAE. On our sparse terminal
binary reward that learned critic is the whole point: it is the dense per-token
credit-assignment signal, the RL analogue of the OPD/OPSD credit this project
studies.

Uses TRL's classic RLHF `trl.experimental.ppo.PPOTrainer`, adapted to our RLVR
setup (verifiable binary reward, no reward model, no reference model). Three
adaptations, none of which the stock trainer supports directly:

  * Binary verifiable reward instead of a reward MODEL. TRL's PPOTrainer scores
    with an nn.Module (`get_reward` runs a forward + `.score` head over the token
    ids and reads the last-token logit) and never sees the gold answer. We instead
    pass a tiny sentinel `reward_model`, carry the per-example gold answer through a
    custom collator + a dataloader wrapper, and monkeypatch the module-level
    `get_reward` to dispatch on model identity: our sentinel -> `utils.grade()`
    (+1 correct / 0 incorrect); the value model -> the original `get_reward`.
    (GRPOTrainer takes a `reward_funcs` callable; PPOTrainer has no such hook.)

  * Reference-free (`kl_coef=0.0`), matching our GRPO baseline (GRPOConfig.beta
    defaults to 0.0). Justified for verifiable rewards: a ground-truth verifier
    cannot be reward-hacked, and reasoning RL wants the policy to drift from init.
    The constructor still builds a reference copy (it raises if you pass None
    without PEFT), so we free it post-init to reclaim the memory.
    KNOWN RESIDUAL: at kl_coef=0 the rollout loop still runs one policy-as-ref
    forward per micro-batch whose result is multiplied by 0. Harmless but wasted;
    removing it needs a train()-loop override, deferred to the vLLM port.

  * Value model = the policy arch with a scalar head
    (`AutoModelForSequenceClassification`, num_labels=1), initialised from --model.

NO vLLM: PPOTrainer generates with HF `model.generate` (the slow path we dropped
GKD for). So this baseline is deliberately scoped to the 1.7B student at a reduced
completion budget; porting the GOLD-style colocate-vLLM rollout into PPO is a
separate follow-up. NO resume: PPOTrainer.train() takes no resume argument.

GAE: gamma=1.0 (no discounting over a single reasoning episode) and lam=0.95.
lam is the PPO-vs-GRPO dial -- lam->1 makes advantages ~ Monte-Carlo return minus
the critic baseline (the interpretable ablation); lam<1 leans on the critic's
bootstrap for denser per-token credit.

# single GPU, 1.7B, reduced budget (HF generate is slow at long lengths)
CUDA_VISIBLE_DEVICES=0 uv run python -m train.train_ppo \
    --model Qwen/Qwen3-1.7B --dataset deepmath --max-steps 200
"""

import argparse
import gc
import json
import os

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer
import trl.experimental.ppo.ppo_trainer as _ppo_mod

from utils import (
    DATASET_REGISTRY_TRAIN,
    format_prompt_math,
    grade,
    load_train_dataset,
)


# ---------------------------------------------------------------------------
# Binary verifiable reward, wired in where TRL expects a reward MODEL
# ---------------------------------------------------------------------------
#
# TRL's rollout loop calls `get_reward(model, query_responses, pad_token_id,
# context_length)` twice per micro-batch -- once for the value model, once for the
# reward model -- and only ever passes token ids (never the gold answer). We make
# the reward a rule-based verifier by:
#   1. a sentinel `reward_model` (below) whose identity we detect,
#   2. a monkeypatch of module-level `get_reward` that routes the sentinel to
#      binary grading and everything else (the value model) to the original, and
#   3. per-batch gold answers stashed on the sentinel by _GoldStashingLoader.


class BinaryRewardModel(nn.Module):
    """Stand-in for TRL's reward model that returns a verifiable binary reward.

    Never actually forwarded: the monkeypatched `get_reward` intercepts it before
    the `.score`/backbone path runs. It still has to be a real nn.Module because
    the trainer calls `disable_dropout_in_model` on it and moves it to the device;
    the lone dummy parameter makes `.parameters()`/`.to()` well-defined.

    Gold answers for the current rollout batch are stashed on `current_golds` (with
    a `cursor` advanced across the micro-batch slices) by _GoldStashingLoader, so
    `compute` can look up the gold aligned to each decoded completion.
    """

    def __init__(self, tokenizer, reward_correct: float, reward_incorrect: float):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))
        self.tokenizer = tokenizer
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.current_golds: list[str] = []
        self.cursor = 0

    def compute(self, query_responses: torch.Tensor, context_length: int):
        """Grade the completion (tokens past `context_length`) of each row against
        its gold answer. Returns (reward_logits, scores, sequence_lengths) to match
        `get_reward`; only the middle element (the terminal score) is consumed."""
        responses = query_responses[:, context_length:]
        texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        n = len(texts)
        golds = self.current_golds[self.cursor : self.cursor + n]
        self.cursor += n
        if len(golds) != n:
            raise RuntimeError(
                f"reward gold misalignment: needed {n} golds at cursor "
                f"{self.cursor - n}, batch holds {len(self.current_golds)}"
            )
        vals = [
            self.reward_correct if grade(t, g)[1] else self.reward_incorrect
            for t, g in zip(texts, golds)
        ]
        scores = torch.tensor(vals, dtype=torch.float32, device=query_responses.device)
        return None, scores, None


def _install_reward_patch():
    """Monkeypatch module-level `get_reward` once, dispatching on model identity:
    a BinaryRewardModel -> binary grading; anything else (the value model) -> the
    original. Idempotent, and preserves the original for the value-model path."""
    if getattr(_ppo_mod, "_binary_reward_patched", False):
        return
    _orig_get_reward = _ppo_mod.get_reward

    def get_reward(model, query_responses, pad_token_id, context_length):
        if isinstance(model, BinaryRewardModel):
            return model.compute(query_responses, context_length)
        return _orig_get_reward(model, query_responses, pad_token_id, context_length)

    _ppo_mod.get_reward = get_reward
    _ppo_mod._binary_reward_patched = True


_install_reward_patch()


# ---------------------------------------------------------------------------
# Carrying the gold answer to reward time
# ---------------------------------------------------------------------------


class GoldPadCollator:
    """Collate PPO batches while carrying the per-example gold answer through.

    The default `DataCollatorWithPadding` runs `tokenizer.pad`, which rejects a
    string column -- so we pop `gold` (a list of strings, passed through untouched)
    and pad only `input_ids`/`attention_mask`. The trainer's rollout loop reads
    `input_ids`; `gold` rides along for the reward (see _GoldStashingLoader)."""

    def __init__(self, tokenizer):
        self._pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        golds = [f.pop("gold") for f in features]
        batch = self._pad(features)
        batch["gold"] = golds
        return batch


class _GoldStashingLoader:
    """Wrap the (accelerate-prepared) train dataloader so that drawing a batch
    stashes that batch's gold answers onto the reward sentinel and resets its
    per-batch cursor. The rollout loop consumes micro-batch slices of the batch in
    order, so a cursor advanced by BinaryRewardModel.compute stays aligned with the
    queries -- even though the dataloader shuffles (gold is collated in lockstep
    with input_ids, so the stashed order already matches)."""

    def __init__(self, loader, reward_model: BinaryRewardModel):
        self._loader = loader
        self._reward_model = reward_model

    def __iter__(self):
        for batch in self._loader:
            self._reward_model.current_golds = batch["gold"]
            self._reward_model.cursor = 0
            yield batch

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)


# ---------------------------------------------------------------------------
# Trainer subclass: free the ref at kl_coef=0, install the gold-stashing loader
# ---------------------------------------------------------------------------


class BinaryRewardPPOTrainer(PPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reference-free objective: drop the frozen reference copy the constructor
        # built (it raises if you pass None without PEFT). At kl_coef=0 the loop
        # falls back to a policy-as-ref forward whose KL is scaled by 0.
        if self.args.kl_coef == 0.0 and self.ref_model is not None:
            self.ref_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("kl_coef=0: freed the reference model (ref-free objective).")
        # Route per-batch gold answers to the reward sentinel.
        self.dataloader = _GoldStashingLoader(self.dataloader, self.reward_model)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_ppo_dataset(dataset, tokenizer, max_samples=None, max_prompt_length=1024):
    """`input_ids` = the tokenized [system, user] query (same chat format as GRPO /
    eval, so train and inference prompts align), left-padded at collate time for
    generation. `gold` = the final answer string, consumed by the binary reward.

    Queries longer than `max_prompt_length` tokens are dropped (math prompts are
    short; this just bounds the rollout context length / memory)."""
    ds = load_train_dataset(dataset, max_samples=max_samples)

    def _map(row):
        # [msgs] + return_dict so we get the token list, not a BatchEncoding whose
        # len() is the field count (the same apply_chat_template gotcha handled in
        # train_sdft.build_sdft_dataset / gen_hints).
        ids = tokenizer.apply_chat_template(
            [format_prompt_math(row["question"])],
            add_generation_prompt=True, tokenize=True, return_dict=True,
        )["input_ids"][0]
        return {"input_ids": ids, "gold": str(row["final_answer"])}

    ds = ds.map(_map, remove_columns=ds.column_names)
    n_before = len(ds)
    ds = ds.filter(lambda r: len(r["input_ids"]) <= max_prompt_length, num_proc=4)
    if len(ds) < n_before:
        print(f"  dropped {n_before - len(ds)} queries over "
              f"max_prompt_length={max_prompt_length}")
    return ds


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def build_run_meta(args, num_train_examples: int) -> dict:
    """Provenance for run_meta.json (PPOTrainer has no resume, so this is
    descriptive only, not a resume guard like the other trainers)."""
    return {
        "method": "ppo",
        "model": args.model,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        "reward": "binary_verifiable",
        "reward_correct": args.reward_correct,
        "reward_incorrect": args.reward_incorrect,
        "kl_coef": args.kl_coef,
        "gamma": args.gamma,
        "lam": args.lam,
        "num_ppo_epochs": args.num_ppo_epochs,
        "max_completion_length": args.max_completion_length,
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B",
                   help="Policy to train; the value model is this arch + a scalar head.")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()),
                   help="Training dataset (see utils.DATASET_REGISTRY_TRAIN).")
    p.add_argument("--output-root", default="outputs/ppo")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>")
    p.add_argument("--max-samples", type=int, default=None)
    # reward
    p.add_argument("--reward-correct", type=float, default=1.0,
                   help="Terminal reward for a correct answer (matches GRPO's accuracy_reward=1.0).")
    p.add_argument("--reward-incorrect", type=float, default=0.0,
                   help="Terminal reward for a wrong/unparseable answer.")
    # PPO objective
    p.add_argument("--kl-coef", type=float, default=0.0,
                   help="KL-to-reference penalty. 0.0 = reference-free (matches GRPO "
                        "beta=0.0); the reference copy is freed post-init.")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="GAE discount. 1.0 = no discounting over the reasoning episode.")
    p.add_argument("--lam", type=float, default=0.95,
                   help="GAE lambda. ->1.0 = Monte-Carlo return minus critic baseline "
                        "(the interpretable ablation); <1.0 leans on the critic bootstrap.")
    p.add_argument("--num-ppo-epochs", type=int, default=4,
                   help="Gradient passes per rollout batch (PPO clipping reuse).")
    p.add_argument("--num-mini-batches", type=int, default=1)
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--cliprange-value", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.1)
    p.add_argument("--whiten-rewards", action=argparse.BooleanOptionalAction, default=False,
                   help="Whiten the reward tensor. Off by default (sparse terminal reward).")
    p.add_argument("--missing-eos-penalty", type=float, default=1.0,
                   help="Subtracted from the score of completions with no EOS (e.g. "
                        "reasoning truncated at the budget). Pass a negative value to disable.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Rollout sampling temperature (on-policy exploration).")
    # generation / length
    p.add_argument("--max-completion-length", type=int, default=4096,
                   help="Rollout completion budget (PPO's response_length). HF generate "
                        "is slow at long lengths, so this defaults below the SDFT/GOLD "
                        "runs' 8192; raise for parity at the cost of speed.")
    p.add_argument("--max-prompt-length", type=int, default=1024,
                   help="Drop queries longer than this (bounds rollout context / memory).")
    p.add_argument("--local-rollout-forward-batch-size", type=int, default=8,
                   help="Micro-batch for the rollout forwards (generation/value). Keep "
                        "small at long completion lengths to bound memory.")
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup",
                            "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Optimizer updates (PPO 'num_total_batches'). Converted to "
                        "total_episodes = max_steps * per_device_bs * grad_accum "
                        "(single-process); multi-process scales the batch, so the "
                        "update count divides by world size.")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True)
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(args.output_root, model_slug, args.dataset)
    print(f"model: {model_slug}  dataset: {args.dataset}  ->  output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation needs left padding so all queries share context_length.
    tokenizer.padding_side = "left"

    train_dataset = build_ppo_dataset(
        args.dataset, tokenizer,
        max_samples=args.max_samples, max_prompt_length=args.max_prompt_length,
    )
    print(f"Loaded {len(train_dataset)} examples")

    # PPO counts episodes (prompts), not steps: num_total_batches = ceil(total_episodes
    # / batch_size). Single-process batch_size = per_device * grad_accum, so this yields
    # ~max_steps updates. (accelerate multi-process scales batch_size by world_size.)
    total_episodes = (
        args.max_steps
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )
    print(f"  total_episodes={total_episodes} (~{args.max_steps} updates, single-process)")

    training_args = PPOConfig(
        output_dir=output_dir,
        # PPO / GAE
        kl_coef=args.kl_coef,
        gamma=args.gamma,
        lam=args.lam,
        num_ppo_epochs=args.num_ppo_epochs,
        num_mini_batches=args.num_mini_batches,
        cliprange=args.cliprange,
        cliprange_value=args.cliprange_value,
        vf_coef=args.vf_coef,
        whiten_rewards=args.whiten_rewards,
        missing_eos_penalty=None if args.missing_eos_penalty < 0 else args.missing_eos_penalty,
        temperature=args.temperature,
        stop_token="eos",  # truncate each completion at the first EOS before reward
        # rollout / length
        response_length=args.max_completion_length,
        local_rollout_forward_batch_size=args.local_rollout_forward_batch_size,
        num_sample_generations=0,  # no periodic sample logging (no eval set)
        total_episodes=total_episodes,
        # optimization
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=True,
        # bookkeeping
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    # Policy (actor) and value (critic) as nn.Modules -- PPOTrainer takes instances,
    # not model-id strings. The value model is the same arch with a scalar head.
    model_kwargs = {"dtype": torch.bfloat16, "trust_remote_code": True}
    policy = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, **model_kwargs
    )
    reward_model = BinaryRewardModel(tokenizer, args.reward_correct, args.reward_incorrect)

    meta = build_run_meta(args, len(train_dataset))
    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote run metadata -> {os.path.join(output_dir, 'run_meta.json')}")

    trainer = BinaryRewardPPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,  # built then freed post-init at kl_coef=0
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        data_collator=GoldPadCollator(tokenizer),
    )
    trainer.train()
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved model -> {final_dir}")


if __name__ == "__main__":
    main()
