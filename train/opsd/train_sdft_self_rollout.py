"""Train SDFT with online self-rollout privileged information.

For every on-policy student completion, the teacher prompt includes that exact
completion as an attempted solution and then prefills the same completion as the
sequence to score. The teacher can therefore condition any token on the entire
attempt, including tokens later in the rollout. The attempt is generated online;
no PI cache, verifier outcome, reference answer, or expert solution is used.

This is a single-purpose extension of TRL's experimental `SDFTTrainer`. Generation
must remain student-conditioned because the self-rollout PI does not exist until
after generation.

Resume (`--resume-from-checkpoint`) restores weights/optimizer/scheduler/RNG and
skips already-seen data; pass the same hyperparameters (verified against run_meta.json).
CAVEAT: only sound for `--teacher-model-kind base` (the frozen base teacher is re-loaded
from the base model id, so resume can't corrupt it). For `ema` the EMA teacher state is
held in a callback and is NOT in the checkpoint, so resuming resets it to the base weights
and silently loses the accumulated EMA -- don't resume `ema` runs without accounting for this.
--max-steps is the TOTAL budget and is free to raise, but the LEARNING RATE comes from the
checkpoint rather than the command line, so a changed --learning-rate is refused rather than
silently ignored (full rules in utils.validate_resume).

# single GPU, colocate vLLM. The teacher forward contains approximately two copies
# of the completion (one as PI, one as the scored target), so keep the 8K completion
# and give its teacher prompt 16K. The combined forward remains below a 32K context.
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_sdft_self_rollout \
    --model Qwen/Qwen3-4B --dataset deepmath \
    --max-prompt-length 16384 --max-completion-length 8192

# multiple GPUs: data-parallel via accelerate, one process per GPU. In colocate
# mode each process trains the policy and runs its own vLLM engine on its GPU.
# The global batch = num_processes * per-device-train-batch-size * grad-accum,
# and num_generations must divide it (default 8 divides 4 * 1 * 16 = 64).
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
    -m train.opsd.train_sdft_self_rollout --model Qwen/Qwen3-4B \
    --dataset deepmath --max-samples 8192 \
    --max-prompt-length 16384 --max-completion-length 8192
"""

import argparse
import json
import os
from statistics import mean

import torch
from trl.experimental.sdft import SDFTConfig, SDFTTrainer

from utils import (
    DATASET_REGISTRY_TRAIN,
    TEACHER_PROMPT_TEMPLATE,
    format_prompt_math,
    load_train_dataset,
    validate_resume,
)


# ---------------------------------------------------------------------------
# Privileged context ("c"): a self-contained string appended to the teacher's
# user turn by SDFTTrainer via teacher_prompt_template ("{prompt}\n\n{privileged_context}").
# ---------------------------------------------------------------------------

PI_SELF_ROLLOUT = (
    "Here is an attempted solution to the question above. It may or may not be correct:\n\n"
    "{attempt}\n\n"
    "Now write a complete solution of your own, including the reasoning."
)
SELF_ROLLOUT_PLACEHOLDER = "__SELF_ROLLOUT_PI_IS_CONSTRUCTED_ONLINE__"

class SelfRolloutTeacherContextBuilder:
    """Replace the static PI with the exact completion generated for each row.

    SDFT calls the context builder only after sampling `completion_ids`. Delegating
    the final tensor construction keeps TRL's chat templating, padding, completion
    alignment, callbacks, and loss path unchanged.
    """

    def __init__(self, trainer, delegate):
        self.trainer = trainer
        self.delegate = delegate

    def _decode_completion(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> str:
        ids = completion_ids[completion_mask.bool()].detach().cpu().tolist()
        # Preserve all content tokens (including reasoning markers), removing only
        # terminal generation-control tokens before inserting the text into a user turn.
        terminal_ids = {
            token_id
            for token_id in (
                self.trainer._tokenizer.pad_token_id,
                self.trainer._tokenizer.eos_token_id,
            )
            if token_id is not None
        }
        while ids and ids[-1] in terminal_ids:
            ids.pop()
        return self.trainer._tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _tokenize_untruncated(self, prompts: list) -> list[list[int]]:
        """Tokenize like SDFTTrainer._tokenize_prompts, but do not slice from the left."""
        if not prompts:
            return []
        if isinstance(prompts[0], list):
            tokenized = self.trainer.processing_class.apply_chat_template(
                conversation=prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **self.trainer.chat_template_kwargs,
            )
            return tokenized["input_ids"]
        return self.trainer.processing_class(text=prompts)["input_ids"]

    def _model_context_window(self) -> int | None:
        config = getattr(self.trainer.model, "config", None)
        for name in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def build(
        self,
        prompts,
        privileged_contexts,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ):
        if not (
            len(prompts)
            == len(privileged_contexts)
            == completion_ids.size(0)
            == completion_mask.size(0)
        ):
            raise ValueError(
                "self_rollout PI needs one prompt, placeholder, completion, and mask per row."
            )
        if any(context != SELF_ROLLOUT_PLACEHOLDER for context in privileged_contexts):
            raise ValueError(
                "self_rollout PI expected the online-context placeholder in every dataset row."
            )

        attempts = [
            self._decode_completion(ids, mask)
            for ids, mask in zip(completion_ids, completion_mask, strict=True)
        ]
        online_contexts = [
            PI_SELF_ROLLOUT.format(attempt=attempt) for attempt in attempts
        ]
        teacher_prompts = [
            self.delegate._compose_teacher_prompt(prompt, context)
            for prompt, context in zip(prompts, online_contexts, strict=True)
        ]
        untruncated_ids = self._tokenize_untruncated(teacher_prompts)
        prompt_lengths = [len(ids) for ids in untruncated_ids]
        completion_lengths = completion_mask.sum(dim=1).detach().cpu().tolist()

        max_prompt_length = self.trainer.max_prompt_length
        if max_prompt_length is not None:
            overflowing = [
                (row, prompt_len, int(completion_lengths[row]))
                for row, prompt_len in enumerate(prompt_lengths)
                if prompt_len > max_prompt_length
            ]
            if overflowing:
                preview = ", ".join(
                    f"row {row}: teacher_prompt={prompt_len}, completion={completion_len}"
                    for row, prompt_len, completion_len in overflowing[:5]
                )
                raise ValueError(
                    "Online self_rollout PI would be left-truncated, so the teacher would "
                    "not see the complete question and attempted solution. "
                    f"max_prompt_length={max_prompt_length}; {preview}. Increase "
                    "--max-prompt-length or reduce --max-completion-length."
                )

        # The delegated builder pads to max(prompt length) + max(completion length).
        # Check that actual tensor width before asking the model to forward it.
        context_window = self._model_context_window()
        padded_width = max(prompt_lengths, default=0) + completion_ids.size(1)
        if context_window is not None and padded_width > context_window:
            raise ValueError(
                "Online self_rollout teacher input exceeds the model context window: "
                f"padded teacher width={padded_width}, model context={context_window}. "
                "Reduce --max-completion-length or --max-prompt-length."
            )

        mode = "train" if self.trainer.model.training else "eval"
        if prompt_lengths:
            self.trainer._metrics[mode]["self_rollout/teacher_prompt_mean_length"].append(
                mean(prompt_lengths)
            )
            self.trainer._metrics[mode]["self_rollout/teacher_prompt_max_length"].append(
                max(prompt_lengths)
            )

        return self.delegate.build(
            prompts,
            online_contexts,
            completion_ids,
            completion_mask,
        )

    def select_generation_prompts(self, prompts, privileged_contexts):
        raise ValueError(
            "self_rollout PI cannot use generate_from_teacher: the PI is the student "
            "completion and does not exist until after generation."
        )


class SelfRolloutSDFTTrainer(SDFTTrainer):
    """SDFTTrainer whose teacher PI is constructed from each online completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_context_builder = SelfRolloutTeacherContextBuilder(
            self,
            self.teacher_context_builder,
        )


def build_train_dataset(dataset: str, max_samples: int | None):
    """Build student prompts; the specialized trainer replaces the sentinel online."""
    ds = load_train_dataset(dataset, max_samples=max_samples)

    def _map(row):
        return {
            "prompt": format_prompt_math(row["question"]),
            "privileged_context": SELF_ROLLOUT_PLACEHOLDER,
        }

    return ds.map(_map, remove_columns=ds.column_names)

# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def build_run_meta(args, num_train_examples: int) -> dict:
    """Provenance + resume-critical config for run_meta.json. The second block must
    match on resume for the seeded data-skip to land on the same examples."""
    return {
        "model": args.model,
        "pi_mode": "self_rollout",
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "self_rollout_pi_source": "online_same_completion",
        "self_rollout_pi_template": PI_SELF_ROLLOUT,
        "generate_from_teacher": False,
        "num_train_examples": num_train_examples,
        "distillation_mode": args.distillation_mode,
        "distillation_alpha": args.distillation_alpha,
        "teacher_model_kind": args.teacher_model_kind,
        # resume-critical: on resume the LR comes from the CHECKPOINT, not the CLI (see
        # validate_resume), so recording it turns a silently-ignored change into an error.
        "learning_rate": args.learning_rate,
        # resume-critical: dataset order (seed, length) + batch chunking. If any of
        # these differ from the original run, the shuffle_dataset permutation or the
        # per-step batch boundary shifts and the skip resumes on the wrong data.
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()),
                   help="Training prompts (see utils.DATASET_REGISTRY_TRAIN).")
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/sdft")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>_self_rollout")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Subset the training set")
    # SDFT objective
    p.add_argument("--distillation-mode", default="sampled_token",
                   choices=["topk_logits", "full_logits", "sampled_token"],
                   help="Support the reverse-KL is computed over. 'topk_logits' "
                        "(paper) is the student's top-k; 'sampled_token' is the "
                        "single-sample MC estimator (requires alpha=1.0).")
    p.add_argument("--distillation-alpha", type=float, default=1.0,
                   help="Divergence interpolation: 1.0 = reverse KL (paper), "
                        "0.0 = forward KL, in between = a JS-like mixture.")
    p.add_argument("--distillation-topk", type=int, default=100,
                   help="Top-k support size for distillation-mode=topk_logits.")
    p.add_argument("--distillation-is-clip", type=float, default=2.0,
                   help="Importance-sampling ratio clip; pass a negative value to disable.")
    p.add_argument("--num-loss-tokens-to-skip", type=int, default=0,
                   help="Mask the first N completion tokens from the loss (paper's "
                        "heuristic to suppress copied 'Based on the example...' openings).")
    # teacher
    p.add_argument("--teacher-model-kind", default="base",
                   choices=["base", "live", "ema"],
                   help="'base' = frozen initial student (default); 'ema' = the "
                        "paper's Alg.1 EMA teacher; 'live' = current student.")
    p.add_argument("--teacher-update-rate", type=float, default=0.05,
                   help="EMA rate alpha for teacher-model-kind=ema (phi <- a*theta + (1-a)*phi).")
    p.add_argument("--teacher-sync-steps", type=int, default=1,
                   help="Optimizer steps between EMA teacher updates.")
    # generation
    p.add_argument("--max-prompt-length", type=int, default=16384,
                   help="Teacher prompt budget. self_rollout fails rather than silently "
                        "truncate its online attempted solution.")
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--num-generations", type=int, default=1,
                   help="On-policy rollouts per prompt. Must divide the global batch. "
                        "The paper uses a single rollout per example.")
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="Paper sweeps {5e-6, 1e-5, 5e-5}.")
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup",
                            "inverse_sqrt"],
                   help="LR schedule over --max-steps")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear LR warmup steps before the schedule kicks in.")
    p.add_argument("--optim", default="adamw_bnb_8bit",
                   help="Optimizer. Default 8-bit Adam; use adamw_torch_fused otherwise.")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Total optimizer steps")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    # vLLM
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True,
                   help="Use vLLM for generation. --no-use-vllm falls back to "
                        "transformers generation for debugging.")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3,
                   help="Fraction of GPU memory vLLM may reserve (colocate).")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    # resume
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume from a checkpoint dir (path ending in "
                        "'checkpoint-<step>'). Restores weights/optimizer/scheduler/RNG "
                        "and skips already-seen examples. Pass the SAME --model, --dataset, "
                        "--max-samples, --seed and batch config as the original run (verified "
                        "against its run_meta.json). --max-steps is the TOTAL budget: training "
                        "continues from the checkpoint's step up to --max-steps.")
    p.add_argument("--force-resume", action="store_true",
                   help="Downgrade a run_meta.json hyperparameter mismatch from an error to "
                        "a warning (use only if you understand the data-skip consequences).")
    args = p.parse_args()

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(
        args.output_root, model_slug, f"{args.dataset}_self_rollout"
    )
    print(f"model: {model_slug}  dataset: {args.dataset}  pi: self_rollout  ->  output: {output_dir}")

    train_dataset = build_train_dataset(args.dataset, args.max_samples)
    if len(train_dataset) == 0:
        raise RuntimeError("No training examples were loaded.")
    print(f"Loaded {len(train_dataset)} examples")
    print(f"  sample prompt: {train_dataset[0]['prompt'][-1]['content'][:120]!r}")
    print("  privileged_context: constructed online from each exact student completion")

    training_args = SDFTConfig(
        output_dir=output_dir,
        # SDFT objective
        distillation_mode=args.distillation_mode,
        distillation_alpha=args.distillation_alpha,
        distillation_topk=args.distillation_topk,
        distillation_is_clip=None if args.distillation_is_clip < 0 else args.distillation_is_clip,
        num_loss_tokens_to_skip=args.num_loss_tokens_to_skip,
        # teacher
        teacher_model_kind=args.teacher_model_kind,
        teacher_update_rate=args.teacher_update_rate,
        teacher_sync_steps=args.teacher_sync_steps,
        generate_from_teacher=False,
        teacher_prompt_template=TEACHER_PROMPT_TEMPLATE,
        # generation
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        # generation backend
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        # optimization
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        # SDFTTrainer instantiates the policy from the model string; request bf16
        # explicitly (mirrors train_grpo.py -- dtype defaults to fp32 otherwise).
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        # bookkeeping
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    meta = build_run_meta(args, len(train_dataset))

    # Before resuming, verify the run this checkpoint came from used the same config
    # (its run_meta.json sits beside the checkpoint), so the seeded data-skip lands
    # on the examples that were actually left untrained.
    if args.resume_from_checkpoint:
        validate_resume(args.resume_from_checkpoint, meta, args.force_resume)

    # The self-rollout PI provenance is not represented in SDFTConfig, so keep it
    # beside the checkpoints in a small run_meta.json.
    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(output_dir, "run_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote run metadata -> {meta_path}")

    trainer = SelfRolloutSDFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved model -> {final_dir}")


if __name__ == "__main__":
    main()
