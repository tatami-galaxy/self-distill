"""PPO with a privileged-information-conditioned value function.

This is the asymmetric actor-critic baseline: the policy samples and is optimized under the
ordinary math prompt, while the critic may additionally see privileged information (PI).  The
The experiment supports four matched arms:

  * ``--pi-mode none``   -- the critic reads exactly the policy prompt (the train_ppo.py control)
  * ``--pi-mode answer`` -- the critic also sees the gold final answer
  * ``--pi-mode full``   -- the critic also sees DeepMath's worked solution
  * ``--pi-mode hint``   -- the critic also sees the model's generated DeepMath hint

All non-``none`` modes use the same wording and last-user-turn injection convention as SDFT.
``hint`` loads the model-specific cache under ``data/pi/hint/deepmath`` and validates its
provenance.

Nothing else changes: rewards, GAE, clipped policy/value losses, optimizers, and the randomly
initialized scalar value head all come from train_ppo.py.  In particular, this file does not use
train_ppo_val.py's verifier instruction and does not implement Tether.

The critic-only prompt is constructed once during rollout generation and stored in the rollout
dict.  It therefore stays aligned through TRL's shuffle/split buffering and PPO epoch reuse, so
``old_values`` and fresh value predictions always score the same state representation.

# single GPU, colocate vLLM
CUDA_VISIBLE_DEVICES=0 uv run python -m train.ppo.train_ppo_pi \
    --model Qwen/Qwen3-1.7B --dataset deepmath --pi-mode answer \
    --max-steps 200 --critic-warmup-steps 20

# 4B+: vLLM on its own GPU, policy + critic on another
CUDA_VISIBLE_DEVICES=7 uv run trl vllm-serve \
    --model Qwen/Qwen3-4B --gpu-memory-utilization 0.9 &
CUDA_VISIBLE_DEVICES=6 uv run python -m train.ppo.train_ppo_pi \
    --model Qwen/Qwen3-4B --dataset deepmath --pi-mode answer \
    --vllm-mode server --optim adafactor
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, set_seed
from trl.rewards import accuracy_reward

from train.ppo.train_ppo import PPOConfig, PPOTrainer, is_bitsandbytes_optim
from utils import (
    DATASET_REGISTRY_TRAIN,
    PI_ANSWER,
    PI_FULL,
    PI_HINT,
    compose_pi_messages,
    format_prompt_math,
    hint_path,
    load_hint_cache,
    load_train_dataset,
    validate_resume,
)

PI_MODES = ("none", "answer", "full", "hint")

# These identify what a saved critic's inputs mean.  They are strict resume keys: changing the
# wording or injection rule invalidates both value_model.pt and its optimizer state.
VALUE_PROMPT_VERSIONS = {
    "none": "policy_prompt_v1",
    "answer": "sdft_answer_last_user_v1",
    "full": "sdft_full_last_user_v1",
    "hint": "sdft_hint_last_user_v1",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_ppo_pi_dataset(
    dataset: str = "deepmath",
    max_samples: int | None = None,
    pi_mode: str = "answer",
    model: str | None = None,
    max_value_prompt_length: int | None = 8192,
):
    """Build matched policy/reward rows and keep PI in a critic-only column.

    ``prompt`` is always the ordinary policy prompt.  GRPOTrainer generates from that column;
    it never receives ``privileged_context`` as part of the conversation.  The extra column is
    consumed only by :class:`PPOPITrainer` when constructing the critic prompt.
    """
    if pi_mode not in PI_MODES:
        raise ValueError(f"unknown pi_mode {pi_mode!r}; expected one of {PI_MODES}")
    if pi_mode in ("full", "hint") and dataset != "deepmath":
        raise ValueError(
            f"pi_mode={pi_mode!r} is currently supported only for dataset='deepmath'; "
            f"got {dataset!r}."
        )
    if pi_mode in ("full", "hint") and model is None:
        raise ValueError(f"pi_mode={pi_mode!r} needs --model to build its critic PI.")
    if (
        pi_mode == "full"
        and max_value_prompt_length is not None
        and max_value_prompt_length < 1
    ):
        raise ValueError("max_value_prompt_length must be positive or None")

    if pi_mode == "hint":
        # The cache itself is the source of question order and final answers. Avoid joining it
        # to a separately loaded dataset by question text: duplicates would make that ambiguous,
        # and even a harmless source-order drift could attach a hint to the wrong reward target.
        ds = load_hint_cache(model, dataset, max_samples=max_samples)
    else:
        ds = load_train_dataset(
            dataset,
            max_samples=max_samples,
            require_solution=(pi_mode == "full"),
        )

    def _map(row):
        answer = str(row["final_answer"])
        if pi_mode == "answer":
            privileged_context = PI_ANSWER.format(answer=answer)
        elif pi_mode == "full":
            privileged_context = PI_FULL.format(demo=row["solution"])
        elif pi_mode == "hint":
            privileged_context = PI_HINT.format(hint=row["hint"])
        else:  # none
            privileged_context = ""
        return {
            "prompt": format_prompt_math(row["question"]),
            "solution": "\\boxed{" + answer + "}",
            "privileged_context": privileged_context,
        }

    ds = ds.map(_map, remove_columns=ds.column_names)
    return filter_long_full_value_prompts(
        ds,
        pi_mode=pi_mode,
        model=model,
        max_value_prompt_length=max_value_prompt_length,
    )


def filter_long_full_value_prompts(
    ds,
    pi_mode: str,
    model: str | None,
    max_value_prompt_length: int | None,
):
    """Drop full-PI rows whose exact critic prompt would need truncation.

    This mirrors train_sdft.py's full-PI safeguard. Left-truncating the critic prompt would
    silently remove the beginning of the question or worked solution, so filtering is the safer
    experiment. It runs after ``max_samples``, matching SDFT; the resulting dataset may therefore
    contain fewer than the requested number of examples. Hints are short and are not filtered.
    """
    if pi_mode != "full" or model is None or max_value_prompt_length is None:
        return ds

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def _value_prompt_fits(row):
        messages = compose_pi_messages(row["prompt"], row["privileged_context"])
        ids = tokenizer.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"][0]
        return len(ids) <= max_value_prompt_length

    n_before = len(ds)
    ds = ds.filter(_value_prompt_fits, num_proc=4)
    print(
        f"  pi=full: kept {len(ds)}/{n_before} rows whose critic prompt "
        f"fits max_value_prompt_length={max_value_prompt_length}"
    )
    return ds


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PPOPITrainer(PPOTrainer):
    """PPOTrainer whose critic, and only its critic, can observe PI."""

    def __init__(self, *args, pi_mode: str = "answer", **kwargs):
        if pi_mode not in PI_MODES:
            raise ValueError(f"unknown pi_mode {pi_mode!r}; expected one of {PI_MODES}")
        self.pi_mode = pi_mode
        self._value_rows = None
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        # The control is deliberately the exact base path: no prompt reconstruction, no extra
        # whitespace, and no second tokenization of the critic input.
        if self.pi_mode == "none":
            output = super()._generate_and_score_completions(inputs)
            self._log_value_diagnostics(output)
            return output

        # Stash rows before super(): PPOTrainer computes old_values during this call and reaches
        # _value_inputs.  The constructed tensors are written into the returned rollout dict,
        # after which the raw rows are unnecessary.  Clearing in finally makes a dropped cache
        # fail closed instead of pairing a later rollout with stale PI.
        self._value_rows = inputs
        try:
            output = super()._generate_and_score_completions(inputs)
        finally:
            self._value_rows = None
        self._log_value_diagnostics(output)
        return output

    def _value_inputs(self, batch):
        """Return critic inputs without ever changing what the policy receives."""
        if self.pi_mode == "none":
            return super()._value_inputs(batch)

        if "value_prompt_ids" not in batch:
            if self._value_rows is None:
                raise RuntimeError(
                    "No PI-conditioned value prompt in the batch and no rows stashed to build "
                    "one from. The rollout buffer may have dropped 'value_prompt_ids'."
                )
            n_rows = len(self._value_rows)
            n_completions = batch["completion_ids"].size(0)
            if n_rows != n_completions:
                raise RuntimeError(
                    f"Stashed PI rows ({n_rows}) != completions in the batch "
                    f"({n_completions}). The critic would receive another rollout's PI."
                )
            ids, mask = self._build_value_prompts(
                self._value_rows, batch["completion_ids"].device
            )
            batch["value_prompt_ids"], batch["value_prompt_mask"] = ids, mask

        input_ids = torch.cat(
            [batch["value_prompt_ids"], batch["completion_ids"]], dim=1
        )
        attention_mask = torch.cat(
            [batch["value_prompt_mask"], batch["completion_mask"]], dim=1
        )
        return input_ids, attention_mask, batch["completion_ids"].size(1)

    def _build_value_prompts(self, rows, device):
        """Render PI-conditioned conversations with the aligned tokenizer path."""
        conversations = []
        for i, row in enumerate(rows):
            if "privileged_context" not in row:
                raise RuntimeError(f"row {i} has no 'privileged_context' for the PI critic")
            if not str(row["privileged_context"]).strip():
                raise RuntimeError(f"row {i} has empty privileged information for the PI critic")
            conversations.append(
                compose_pi_messages(row["prompt"], row["privileged_context"])
            )
        ids, mask, _ = self._render_value_prompts(conversations, device)
        return ids, mask

    def _reduce_moments(self, values, targets, mask):
        """Return global count, target/error moments for a masked diagnostic slice."""
        values = values.detach().float()
        targets = targets.detach().float()
        mask = mask.detach().float()
        error = targets - values
        stats = torch.stack(
            [
                mask.sum(),
                (targets * mask).sum(),
                (targets.square() * mask).sum(),
                (error * mask).sum(),
                (error.square() * mask).sum(),
            ]
        )
        return self.accelerator.reduce(stats, reduction="sum")

    def _append_prediction_metrics(self, mode, suffix, values, targets, mask):
        count, target_sum, target_sq_sum, error_sum, error_sq_sum = self._reduce_moments(
            values, targets, mask
        )
        count_value = count.item()
        if count_value <= 0:
            return

        target_mean = target_sum / count
        error_mean = error_sum / count
        target_var = (target_sq_sum / count - target_mean.square()).clamp(min=0.0)
        error_var = (error_sq_sum / count - error_mean.square()).clamp(min=0.0)
        self._metrics[mode][f"ppo/value_outcome_mse{suffix}"].append(
            (error_sq_sum / count).item()
        )
        # EV is undefined when a batch contains only one outcome class.  Omitting it is safer
        # than emitting NaN into TensorBoard; larger/global rollout batches normally contain
        # both outcomes.
        if target_var.item() > 1e-12:
            self._metrics[mode][f"ppo/value_explained_variance{suffix}"].append(
                (1.0 - error_var / target_var).item()
            )

    def _log_value_diagnostics(self, output):
        """Log how much outcome information the critic extracts from its state.

        The outcome target is broadcast over the completion.  Early/middle/late slices make the
        intended PI benefit visible: an answer-conditioned critic can be predictive before the
        sampled reasoning itself reveals whether the trajectory will succeed.
        """
        old_values = output["old_values"]
        completion_mask = output["completion_mask"].float()
        if "scorable" in output:
            completion_mask = completion_mask * output["scorable"]
        outcome = output["terminal_reward"].expand_as(old_values)
        mode = "train" if self.model.training else "eval"

        self._append_prediction_metrics(
            mode, "", old_values, outcome, completion_mask
        )

        positions = torch.arange(old_values.size(1), device=old_values.device).unsqueeze(0)
        lengths = output["completion_mask"].sum(dim=1, keepdim=True).long().clamp(min=1)
        thirds = positions * 3
        buckets = {
            "_early": thirds < lengths,
            "_middle": (thirds >= lengths) & (thirds < 2 * lengths),
            "_late": thirds >= 2 * lengths,
        }
        for suffix, bucket in buckets.items():
            self._append_prediction_metrics(
                mode, suffix, old_values, outcome, completion_mask * bucket
            )

        # This is the unwhitened GAE advantage even when the policy receives a whitened copy.
        raw_advantage = output["returns"] - old_values
        stats = self._reduce_moments(
            torch.zeros_like(raw_advantage), raw_advantage, completion_mask
        )
        count, adv_sum, adv_sq_sum, _, _ = stats
        if count.item() > 0:
            mean = adv_sum / count
            variance = (adv_sq_sum / count - mean.square()).clamp(min=0.0)
            self._metrics[mode]["ppo/raw_advantage_mean"].append(mean.item())
            self._metrics[mode]["ppo/raw_advantage_std"].append(variance.sqrt().item())


# ---------------------------------------------------------------------------
# Resume metadata
# ---------------------------------------------------------------------------


def build_run_meta(args, num_train_examples: int) -> dict:
    return {
        "method": "ppo_pi_vllm",
        "pi_mode": args.pi_mode,
        "value_prompt_version": VALUE_PROMPT_VERSIONS[args.pi_mode],
        "gen_model": args.model if args.pi_mode == "hint" else None,
        "hint_cache_path": (
            hint_path(args.model, args.dataset) if args.pi_mode == "hint" else None
        ),
        "max_value_prompt_length": (
            args.max_value_prompt_length if args.pi_mode == "full" else None
        ),
        "model": args.model,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        "reward": "accuracy_reward",
        "gamma": args.gamma,
        "lam": args.lam,
        "vf_coef": args.vf_coef,
        "cliprange_value": args.cliprange_value,
        "critic_max_grad_norm": args.critic_max_grad_norm,
        # resume-critical: warmup is keyed on the RESTORED global_step, so a resumed run
        # silently trains jointly if it finished warmup -- recording it makes a changed
        # value an error rather than an invisible difference between the two halves.
        "critic_warmup_steps": args.critic_warmup_steps,
        "loss_type": args.loss_type,
        "vllm_mode": args.vllm_mode,
        "optim": args.optim,
        "learning_rate": args.learning_rate,
        "critic_learning_rate": args.critic_learning_rate,
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
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
                   help="Training dataset. 'full' and 'hint' are currently DeepMath-only.")
    p.add_argument("--pi-mode", default="answer", choices=PI_MODES,
                   help="PI exposed only to the critic: none, gold answer, worked solution, "
                        "or a generated self-hint from data/pi/hint.")
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/ppo_pi",
                   help="Separate from the PPO and PPO-verifier experiment roots.")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>_<pi-mode>")
    p.add_argument("--max-samples", type=int, default=None, help="Subset the training set")
    p.add_argument("--max-value-prompt-length", type=int, default=8192,
                   help="Maximum critic prompt length. Full-PI examples exceeding it are "
                        "dropped rather than silently left-truncated; hints are not filtered.")
    # PPO / GAE
    p.add_argument("--gamma", type=float, default=1.0,
                   help="GAE discount. 1.0 = no discounting over the reasoning episode.")
    p.add_argument("--lam", type=float, default=1.0,
                   help="GAE lambda. 1.0 = Monte-Carlo return minus critic baseline.")
    p.add_argument("--vf-coef", type=float, default=0.1, help="Value-loss weight.")
    p.add_argument("--cliprange-value", type=float, default=0.2, help="Value-clipping range.")
    p.add_argument("--critic-max-grad-norm", type=float, default=10.0,
                   help="Clip the critic's gradients to this norm, SEPARATELY from the policy "
                        "(which Trainer clips to max_grad_norm=1.0). Looser than the policy's "
                        "because the critic's raw norms run ~10-40 early on, so a clip of 1.0 "
                        "fires every step and becomes a step-size control instead of a spike "
                        "guard. Pass 0 to disable clipping while still logging "
                        "ppo/critic_grad_norm, so the two can be compared.")
    p.add_argument("--critic-warmup-steps", type=int, default=0,
                   help="Freeze the policy for this many optimizer steps at the start of "
                        "training so the randomly-initialised value head can fit against a "
                        "FIXED policy before its advantages start steering one. Counted "
                        "against --max-steps. 0 = joint training from step 0.")
    p.add_argument("--no-whiten-advantages", dest="whiten_advantages",
                   action="store_false", help="Disable GAE advantage whitening.")
    p.add_argument("--missing-eos-penalty", type=float, default=0.0,
                   help="Subtract from terminal reward if a completion did not end in EOS.")
    p.add_argument("--num-ppo-epochs", type=int, default=1,
                   help=">1 reuses rollouts with PPO's clipped old log probabilities.")
    p.add_argument("--loss-type", default="dapo", choices=["dapo", "bnpo"],
                   help="Token-uniform policy/value loss aggregation.")
    p.add_argument("--epsilon", type=float, default=0.2, help="PPO policy clip range.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Rollout sampling temperature.")
    # generation
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--num-generations", type=int, default=1,
                   help="Rollouts per prompt; PPO uses the critic rather than a group baseline.")
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--critic-learning-rate", type=float, default=1e-5,
                   help="Learning rate for the critic. Defaults to 10x the policy's 1e-6")
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                            "constant", "constant_with_warmup", "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="paged_adamw_8bit",
                   help="Use adafactor in vLLM server mode; bitsandbytes corrupts the policy.")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True)
    # vLLM
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"])
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                   help="COLOCATE ONLY (default 0.25).")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=None,
                   help="COLOCATE ONLY (default 1).")
    p.add_argument("--vllm-server-host", default="0.0.0.0")
    p.add_argument("--vllm-server-port", type=int, default=8000)
    p.add_argument("--vllm-server-timeout", type=float, default=240.0)
    p.add_argument("--vllm-group-port", type=int, default=51216)
    # bookkeeping / resume
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--force-resume", action="store_true")
    args = p.parse_args()

    if args.vllm_mode == "server":
        misplaced = [
            f"{flag} (use `trl vllm-serve {serve_flag}`)"
            for flag, value, serve_flag in (
                ("--vllm-gpu-memory-utilization", args.vllm_gpu_memory_utilization,
                 "--gpu-memory-utilization"),
                ("--vllm-tensor-parallel-size", args.vllm_tensor_parallel_size,
                 "--tensor-parallel-size"),
            )
            if value is not None
        ]
        if misplaced:
            p.error("these only apply to --vllm-mode colocate: " + "; ".join(misplaced))
        if is_bitsandbytes_optim(args.optim):
            p.error(
                f"--optim {args.optim} corrupts the policy under --vllm-mode server; "
                "use --optim adafactor."
            )

    vllm_gpu_mem = (
        0.25 if args.vllm_gpu_memory_utilization is None
        else args.vllm_gpu_memory_utilization
    )
    vllm_tp = 1 if args.vllm_tensor_parallel_size is None else args.vllm_tensor_parallel_size

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(
        args.output_root, model_slug, f"{args.dataset}_{args.pi_mode}"
    )
    print(
        f"model: {model_slug}  dataset: {args.dataset}  pi: {args.pi_mode}  "
        f"->  output: {output_dir}"
    )
    print(
        f"  critic prompt: {VALUE_PROMPT_VERSIONS[args.pi_mode]}; "
        "policy prompt unchanged"
    )
    if args.vllm_mode == "server":
        print(
            f"  vLLM: server at {args.vllm_server_host}:{args.vllm_server_port} "
            f"(weight-sync group port {args.vllm_group_port})"
        )

    train_dataset = build_ppo_pi_dataset(
        args.dataset,
        max_samples=args.max_samples,
        pi_mode=args.pi_mode,
        model=args.model,
        max_value_prompt_length=args.max_value_prompt_length,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training examples remain after selecting and filtering PI.")
    print(f"Loaded {len(train_dataset)} examples")
    print(f"  sample prompt: {train_dataset[0]['prompt'][-1]['content'][:120]!r}")
    print(f"  sample solution: {train_dataset[0]['solution']!r}")
    if args.pi_mode != "none":
        print(f"  sample critic PI: {train_dataset[0]['privileged_context'][:160]!r}")

    training_args = PPOConfig(
        output_dir=output_dir,
        gamma=args.gamma,
        lam=args.lam,
        vf_coef=args.vf_coef,
        cliprange_value=args.cliprange_value,
        critic_max_grad_norm=args.critic_max_grad_norm,
        critic_warmup_steps=args.critic_warmup_steps,
        whiten_advantages=args.whiten_advantages,
        missing_eos_penalty=args.missing_eos_penalty,
        num_iterations=args.num_ppo_epochs,
        loss_type=args.loss_type,
        epsilon=args.epsilon,
        temperature=args.temperature,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=vllm_gpu_mem,
        vllm_tensor_parallel_size=vllm_tp,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_group_port=args.vllm_group_port,
        learning_rate=args.learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    # Seed before constructing the randomly initialized scalar head so matched arms start from
    # identical critic parameters.
    set_seed(args.seed)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=torch.bfloat16, trust_remote_code=True
    )

    meta = build_run_meta(args, len(train_dataset))
    if args.resume_from_checkpoint:
        validate_resume(
            args.resume_from_checkpoint,
            meta,
            args.force_resume,
            strict_keys=("pi_mode", "value_prompt_version"),
        )
    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote run metadata -> {os.path.join(output_dir, 'run_meta.json')}")

    trainer = PPOPITrainer(
        model=args.model,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=train_dataset,
        value_model=value_model,
        pi_mode=args.pi_mode,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved model -> {final_dir}")


if __name__ == "__main__":
    main()
