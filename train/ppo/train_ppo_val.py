"""
PPO with value function modifications:
    * the critic reads the same question the policy does, but under a VERIFIER instruction;
    the critic is asked to JUDGE.

Differences from train_ppo.py
  * `_value_inputs` is overridden to hand the critic `[value_prompt || completion]` instead of
    `[prompt || completion]`, building that prompt once per generation batch and caching it into
    the rollout dict -- so it rides GRPO's shuffle/split buffering and `_compute_loss` reuses the
    same tensors, which is what makes `vpred` and `old_values` provably the same state, as
    `cliprange_value` assumes.
  * `build_run_meta` stamps `value_prompt_version` and `method: ppo_val_vllm`.

RESUME. This arm's critic state representation is not train_ppo.py's, so their checkpoints are
not interchangeable, and neither direction is allowed to happen quietly.

# single GPU, colocate vLLM (<=1.7B)
CUDA_VISIBLE_DEVICES=0 uv run python -m train.ppo.train_ppo_val \
    --model Qwen/Qwen3-1.7B --dataset deepmath --max-steps 400

# 4B+: vLLM on its own GPU, policy + critic on another.
CUDA_VISIBLE_DEVICES=7 uv run trl vllm-serve --model Qwen/Qwen3-4B --gpu-memory-utilization 0.9 &
CUDA_VISIBLE_DEVICES=6 uv run python -m train.ppo.train_ppo_val \
    --model Qwen/Qwen3-4B --dataset deepmath --vllm-mode server --optim adafactor
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, set_seed
from trl.rewards import accuracy_reward

from train.grpo.train_grpo import build_grpo_dataset
from train.ppo.train_ppo import PPOConfig, PPOTrainer, is_bitsandbytes_optim
from utils import DATASET_REGISTRY_TRAIN, validate_resume


# The policy still receives the dataset's ordinary problem-solving prompt. Only the
# critic sees this verifier instruction, followed by the same user problem and then
# the sampled completion whose prefixes it scores.
VALUE_SYSTEM_PROMPT = (
    "You are a careful mathematics verifier. The assistant response below is a "
    "candidate solution produced by another model which may or may not be correct. "
    "Read it incrementally and track how likely this attempt is to reach the correct "
    "final answer. Use the reasoning seen so far as evidence, but judge eventual "
    "final-answer correctness, not writing style."
)
VALUE_PROMPT_VERSION = "verifier_v1"


def compose_value_messages(policy_messages):
    """Replace policy system messages with the critic's verifier instruction.

    Make fresh message dictionaries so constructing the critic input cannot mutate the
    policy conversation retained in the dataset or consumed by GRPOTrainer.
    """
    return [{"role": "system", "content": VALUE_SYSTEM_PROMPT}] + [
        dict(message) for message in policy_messages if message.get("role") != "system"
    ]


# ---------------------------------------------------------------------------
# Trainer
#
# No config subclass: the verifier prompt is a property of this arm, not a hyperparameter,
# so PPOConfig is unchanged. `value_prompt_version` is recorded in run_meta.json -- the file
# validate_resume checks.
# ---------------------------------------------------------------------------


class PPOValTrainer(PPOTrainer):
    """PPOTrainer whose critic reads the question under a verifier instruction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._value_rows = None  # raw rows needed to build the critic-only prompt once

    def _generate_and_score_completions(self, inputs):
        # Stash the raw rows BEFORE super() runs: it computes old_values (and therefore calls
        # _value_inputs) inside, so the verifier prompt must already be buildable by then.
        # GRPO's rollout dict carries only TOKENIZED POLICY prompts, which is why the raw rows
        # are needed at all. `inputs` is the list of dataset row dicts, row-aligned with the
        # tensors super() builds -- GRPO uses an identity collator and RepeatSampler emits each
        # prompt's num_generations repeats consecutively -- so row i belongs to completion i.
        #
        # Cleared in `finally` rather than at the next entry: a later _value_inputs arriving
        # WITHOUT 'value_prompt_ids' (i.e. TRL's buffering dropped the key) must raise, and a
        # stale stash would instead sail past the n_rows == n_completions guard -- the batch
        # sizes match -- and silently pair each verifier prompt with a previous batch's question.
        self._value_rows = inputs
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            self._value_rows = None

    def _value_inputs(self, batch):
        """Override: give the critic the verifier-framed prompt instead of the policy's.

        Built lazily on the first call (inside super()._generate_and_score_completions) and
        WRITTEN BACK into `batch` -- which is the dict that call returns, so the value prompt
        rides GRPO's shuffle/split buffering alongside every other rollout tensor and is already
        present when `_compute_loss` calls this on a micro-batch slice. That is what guarantees
        `vpred` and `old_values` are computed from the same state, as cliprange_value assumes.
        """
        if "value_prompt_ids" not in batch:
            if self._value_rows is None:
                raise RuntimeError(
                    "No value prompt in the batch and no rows stashed to build one from. "
                    "_value_inputs must be reached either inside _generate_and_score_completions "
                    "(which stashes the rows) or on a batch that already carries "
                    "'value_prompt_ids'; a TRL change to the buffering may have dropped the key."
                )
            n_rows = len(self._value_rows)
            n_completions = batch["completion_ids"].size(0)
            if n_rows != n_completions:
                raise RuntimeError(
                    f"Stashed value-prompt rows ({n_rows}) != completions in the batch "
                    f"({n_completions}). The verifier prompt would be paired with the wrong "
                    "rollout."
                )
            ids, mask = self._build_value_prompts(
                self._value_rows, batch["completion_ids"].device
            )
            batch["value_prompt_ids"], batch["value_prompt_mask"] = ids, mask

        input_ids = torch.cat([batch["value_prompt_ids"], batch["completion_ids"]], dim=1)
        attention_mask = torch.cat(
            [batch["value_prompt_mask"], batch["completion_mask"]], dim=1
        )
        return input_ids, attention_mask, batch["completion_ids"].size(1)

    def _build_value_prompts(self, rows, device):
        """Verifier conversations -> (B, P_v) ids + mask, left-padded.

        The composition is the whole of this arm; the rendering (chat template, generation
        boundary, left padding, length logging) is the base class's `_render_value_prompts`,
        shared with every other arm that gives the critic its own prompt.
        """
        conversations = [compose_value_messages(row["prompt"]) for row in rows]
        ids, mask, _ = self._render_value_prompts(conversations, device)
        return ids, mask


# ---------------------------------------------------------------------------
# Resume metadata
# ---------------------------------------------------------------------------


def build_run_meta(args, num_train_examples: int) -> dict:
    return {
        # Distinct from train_ppo.py's "ppo_vllm" so the two arms never share a results
        # grouping. See the module docstring for what this means for the runs written before
        # this file existed.
        "method": "ppo_val_vllm",
        # Critic-state identity. Changing the verifier prompt changes what `value_model.pt`
        # and its optimizer state MEAN, so an older checkpoint is not resumable into a newer
        # prompt. main() passes this key to validate_resume's `strict_keys`, which is what
        # makes a checkpoint that never recorded it a hard error rather than a silently
        # skipped comparison.
        "value_prompt_version": VALUE_PROMPT_VERSION,
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
        # See train_ppo.build_run_meta: optim x vllm_mode decides whether the policy NaNs.
        "optim": args.optim,
        # resume-critical, learning rates: on resume these come from the CHECKPOINT, not the
        # CLI, so recording them turns a silently-ignored change into an error.
        "learning_rate": args.learning_rate,
        "critic_learning_rate": args.critic_learning_rate,
        # resume-critical: dataset order + batch chunking.
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
                   help="Training dataset (see utils.DATASET_REGISTRY_TRAIN).")
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/ppo_val",
                   help="Distinct from train_ppo.py's outputs/ppo: run_meta.json lives at the "
                        "run root, so sharing a root would have the two arms overwrite each "
                        "other's provenance.")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>")
    p.add_argument("--max-samples", type=int, default=None, help="Subset the training set")
    # PPO / GAE
    p.add_argument("--gamma", type=float, default=1.0,
                   help="GAE discount. 1.0 = no discounting over the reasoning episode.")
    p.add_argument("--lam", type=float, default=1.0,
                   help="GAE lambda. ->1.0 = Monte-Carlo return minus critic baseline; "
                        "<1.0 leans on the critic bootstrap.")
    p.add_argument("--vf-coef", type=float, default=0.1, help="Value-loss weight.")
    p.add_argument("--cliprange-value", type=float, default=0.2, help="Value-clipping range.")
    p.add_argument("--critic-max-grad-norm", type=float, default=1.0,
                   help="Clip the critic's gradients to this norm, SEPARATELY from the policy "
                        "(which Trainer clips to max_grad_norm=1.0). Pass 0 to disable clipping "
                        "while still logging ppo/critic_grad_norm, so the two can be compared.")
    p.add_argument("--critic-warmup-steps", type=int, default=0,
                   help="Freeze the policy for this many optimizer steps at the start of "
                        "training so the randomly-initialised value head can fit against a "
                        "FIXED policy before its advantages start steering one. Counted "
                        "against --max-steps. 0 = joint training from step 0.")
    p.add_argument("--no-whiten-advantages", dest="whiten_advantages",
                   action="store_false", help="Disable GAE advantage whitening.")
    p.add_argument("--missing-eos-penalty", type=float, default=0.0,
                   help="Subtract from a completion's terminal reward if it did not end in EOS. "
                        "0.0 disables (matches the GRPO baseline).")
    p.add_argument("--num-ppo-epochs", type=int, default=1,
                   help="Gradient passes reusing each rollout (maps to GRPO num_iterations; "
                        ">1 enables PPO's clip-and-reuse via stored old logprobs).")
    p.add_argument("--loss-type", default="dapo", choices=["dapo", "bnpo"],
                   help="Token-loss aggregation for the clipped policy surrogate (see "
                        "train_ppo.py's module docstring). Both are token-uniform and share "
                        "vf_loss's normalizer; GRPO's 'grpo' and 'dr_grpo' are excluded because "
                        "they would rescale or destabilize the per-token GAE credit.")
    p.add_argument("--epsilon", type=float, default=0.2, help="PPO clip range (policy).")
    p.add_argument("--temperature", type=float, default=1.0, help="Rollout sampling temperature.")
    # generation
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--num-generations", type=int, default=1,
                   help="Rollouts per prompt. PPO uses the critic (not a group) as the "
                        "baseline, so grouping is unused -- each rollout gets its own GAE.")
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--critic-learning-rate", type=float, default=None,
                   help="Learning rate for the critic. Omit to inherit --learning-rate. That "
                        "default (1e-6) suits a pretrained policy but is very slow for the "
                        "critic's RANDOMLY INITIALISED scalar head, which has to learn "
                        "P(correct|prefix) from scratch within --max-steps.")
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="paged_adamw_8bit",
                   help="Optimizer. The 8-bit default keeps the memory footprint of the GRPO "
                        "baseline and is fine in COLOCATE mode. In --vllm-mode server EVERY "
                        "bitsandbytes 8-bit optimizer corrupts the policy; "
                        "pass --optim adafactor there. See train_ppo.py's module docstring.")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    # vLLM
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"],
                   help="'colocate': run the engine in-process on the training GPU (simplest, "
                        "nothing idles, but its KV cache competes with policy+critic -- OOMs from "
                        "~4B). 'server': talk to a standalone `trl vllm-serve` on its own GPU. "
                        "See train_ppo.py's module docstring for the launch recipe.")
    # Colocate-only. Defaults are None so we can tell "user set it" from "left alone" and
    # reject it in server mode, where it is the SERVER's property (see below).
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                   help="COLOCATE ONLY (default 0.25): fraction of the training GPU vLLM may "
                        "reserve. In server mode pass --gpu-memory-utilization to "
                        "`trl vllm-serve` instead.")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=None,
                   help="COLOCATE ONLY (default 1). In server mode pass --tensor-parallel-size to "
                        "`trl vllm-serve` instead.")
    p.add_argument("--vllm-server-host", default="0.0.0.0", help="SERVER MODE: vLLM server host.")
    p.add_argument("--vllm-server-port", type=int, default=8000, help="SERVER MODE: vLLM server port.")
    p.add_argument("--vllm-server-timeout", type=float, default=240.0,
                   help="SERVER MODE: seconds to wait for the server to be reachable.")
    p.add_argument("--vllm-group-port", type=int, default=51216,
                   help="SERVER MODE: port for the NCCL weight-sync group the trainer joins as the "
                        "last rank.")
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    # resume
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume dir ('checkpoint-<step>'). Restores policy, critic (weights + "
                        "optimizer state), scheduler and RNG, and skips already-seen examples. "
                        "Pass the SAME --model, --dataset, --max-samples, --seed and batch config "
                        "as the original run (verified against its run_meta.json). --max-steps is "
                        "the TOTAL budget: training continues up to it.")
    p.add_argument("--force-resume", action="store_true")
    args = p.parse_args()

    # Same guards as train_ppo.py: in server mode these are the SERVER's properties, configured
    # when it is launched, and TRL ignores these config fields entirely -- so accepting them here
    # would silently do nothing, which is exactly the flag you'd reach for after an OOM.
    if args.vllm_mode == "server":
        misplaced = [
            f"{flag} (use `trl vllm-serve {serve_flag}`)"
            for flag, val, serve_flag in (
                ("--vllm-gpu-memory-utilization", args.vllm_gpu_memory_utilization,
                 "--gpu-memory-utilization"),
                ("--vllm-tensor-parallel-size", args.vllm_tensor_parallel_size,
                 "--tensor-parallel-size"),
            )
            if val is not None
        ]
        if misplaced:
            p.error(
                "these only apply to --vllm-mode colocate and would be silently ignored in server "
                "mode, where they are the server's properties: " + "; ".join(misplaced)
            )
    if args.vllm_mode == "server" and is_bitsandbytes_optim(args.optim):
        p.error(
            f"--optim {args.optim} corrupts the policy under --vllm-mode server: finite "
            "gradients, NaN params straight out of optimizer.step(), which surfaces two steps "
            "later as an unrelated-looking dtype error inside TRL. Use --optim adafactor. "
            "See train_ppo.py's module docstring."
        )

    vllm_gpu_mem = 0.25 if args.vllm_gpu_memory_utilization is None else args.vllm_gpu_memory_utilization
    vllm_tp = 1 if args.vllm_tensor_parallel_size is None else args.vllm_tensor_parallel_size

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(args.output_root, model_slug, args.dataset)
    print(f"model: {model_slug}  dataset: {args.dataset}  ->  output: {output_dir}")
    print(f"  critic prompt: {VALUE_PROMPT_VERSION} (verifier framing; the policy's is unchanged)")
    if args.vllm_mode == "server":
        print(f"  vLLM: server at {args.vllm_server_host}:{args.vllm_server_port} "
              f"(weight-sync group port {args.vllm_group_port})")

    train_dataset = build_grpo_dataset(args.dataset, max_samples=args.max_samples)
    print(f"Loaded {len(train_dataset)} examples")
    print(f"  sample prompt: {train_dataset[0]['prompt'][-1]['content'][:120]!r}")
    print(f"  sample solution: {train_dataset[0]['solution']!r}")

    training_args = PPOConfig(
        output_dir=output_dir,
        # PPO / GAE
        gamma=args.gamma,
        lam=args.lam,
        vf_coef=args.vf_coef,
        cliprange_value=args.cliprange_value,
        critic_max_grad_norm=args.critic_max_grad_norm,
        critic_warmup_steps=args.critic_warmup_steps,
        whiten_advantages=args.whiten_advantages,
        missing_eos_penalty=args.missing_eos_penalty,
        # policy surrogate (inherited GRPO machinery)
        num_iterations=args.num_ppo_epochs,
        loss_type=args.loss_type,
        epsilon=args.epsilon,
        temperature=args.temperature,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # generation backend
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        # colocate-only; ignored by TRL in server mode (the server owns these)
        vllm_gpu_memory_utilization=vllm_gpu_mem,
        vllm_tensor_parallel_size=vllm_tp,
        # server-only; ignored in colocate
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_group_port=args.vllm_group_port,
        # optimization
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
        # bookkeeping
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    # Seed FIRST: the critic's `.score` head is randomly initialised right here, before Trainer
    # calls set_seed() -- otherwise --seed would not reproduce a run and any A/B over critic
    # hyperparameters would be confounded by a different value function in each arm.
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
            # Absence is disqualifying: a checkpoint written before the verifier prompt holds
            # a critic trained on a different state representation entirely.
            strict_keys=("value_prompt_version",),
        )
    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote run metadata -> {os.path.join(output_dir, 'run_meta.json')}")

    trainer = PPOValTrainer(
        model=args.model,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=train_dataset,
        value_model=value_model,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)  # saves the policy (eval only needs it)
    print(f"Saved model -> {final_dir}")


if __name__ == "__main__":
    main()
