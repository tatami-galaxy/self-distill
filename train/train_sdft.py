"""
Train SDFT -- on-policy self-distillation from a privileged teacher, following
"Self-Distillation Enables Continual Learning" (arXiv 2601.19897), via TRL's
`trl.experimental.sdft.SDFTTrainer`.

Privileged context (`--pi-mode`):
  full   -- the reference r1 solution (reasoning + worked answer) as a worked example
  answer -- only the gold final answer as a hint
  hint   -- a small set of concepts/results the model itself distilled from the
            full solution; precompute with `python -m train.gen_hints` first
            (same --model). Sits between `full` and `answer`.

Resume (`--resume-from-checkpoint`) restores weights/optimizer/scheduler/RNG and
skips already-seen data; pass the same hyperparameters (verified against run_meta.json).
CAVEAT: only sound for `--teacher-model-kind base` (the frozen base teacher is re-loaded
from the base model id, so resume can't corrupt it). For `ema` the EMA teacher state is
held in a callback and is NOT in the checkpoint, so resuming resets it to the base weights
and silently loses the accumulated EMA -- don't resume `ema` runs without accounting for this.

# single GPU, colocate vLLM, check optima and vLLM gpu util for larger models
CUDA_VISIBLE_DEVICES=0 uv run python -m train.train_sdft \
    --model Qwen/Qwen3-4B --pi-mode full --dataset deepmath

# multiple GPUs: data-parallel via accelerate, one process per GPU. In colocate
# mode each process trains the policy and runs its own vLLM engine on its GPU.
# The global batch = num_processes * per-device-train-batch-size * grad-accum,
# and num_generations must divide it (default 8 divides 4 * 1 * 16 = 64).
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
    -m train.train_sdft --model Qwen/Qwen3-4B --pi-mode full --max-samples 8192
"""

import argparse
import json
import os

from datasets import load_from_disk
from trl.experimental.sdft import SDFTConfig, SDFTTrainer

from utils import (
    DATASET_REGISTRY_TRAIN,
    format_prompt_math,
    hint_path,
    load_train_dataset,
    validate_resume,
)


# ---------------------------------------------------------------------------
# Privileged context ("c"): a self-contained string appended to the teacher's
# user turn by SDFTTrainer via teacher_prompt_template ("{prompt}\n\n{privileged_context}").
# ---------------------------------------------------------------------------

PI_FULL = (
    "This is an example of a correct, worked solution to the question above:\n\n"
    "{demo}\n\n"
    "Now write a complete solution of your own, including the reasoning."
)
PI_ANSWER = (
    "Hint: the correct final answer to the question above is \\boxed{{{answer}}}. "
    "Reach it with your own complete reasoning."
)
PI_HINT = (
    "Here are some useful concepts for the question above:\n\n"
    "{hint}\n\n"
    "Use them for your own complete solution if needed."
)

# How SDFTTrainer stitches the student prompt and privileged context into the
# teacher's user turn. Kept here (and passed to SDFTConfig) so the length filter
# below tokenizes the exact teacher prompt the trainer will build.
TEACHER_PROMPT_TEMPLATE = "{prompt}\n\n{privileged_context}"


# ---------------------------------------------------------------------------
# SDFT dataset loader
# ---------------------------------------------------------------------------


def build_sdft_dataset(
    pi_mode: str,
    dataset: str = "deepmath",
    max_samples: int | None = None,
    model: str | None = None,
    max_prompt_length: int | None = None,
):
    """
    `prompt`             -- conversational [system, user] messages, identical to
                            the GRPO / eval format so the student's training and
                            inference prompts align.
    `privileged_context` -- the teacher-only string c (see PI_* above), selected
                            by `pi_mode`. SDFTTrainer requires exactly these two
                            columns; everything else is dropped.

    `pi_mode="hint"` loads precomputed self-hints from disk (see build_hint_dataset);
    `full`/`answer` are built inline from the dataset (question, final_answer,
    solution). `full` requires a solution, so its rows are drawn from the
    solution-bearing subset (see load_train_dataset).

    For `pi_mode="full"` the demo lives entirely in the teacher prompt,
    which SDFTTrainer left-truncates to `max_prompt_length` -- silently lopping
    the start of the demo (and the question). So when `model`/`max_prompt_length`
    are given, drop `full` rows whose teacher prompt doesn't fit, keeping only
    demos the teacher sees intact. Applied after `max_samples`, so the result may
    hold fewer rows.
    """
    if pi_mode == "hint":
        return build_hint_dataset(model, dataset, max_samples)

    ds = load_train_dataset(
        dataset, max_samples=max_samples, require_solution=(pi_mode == "full")
    )

    def _map(row):
        if pi_mode == "answer":
            privileged_context = PI_ANSWER.format(answer=str(row["final_answer"]))
        else:  # full
            privileged_context = PI_FULL.format(demo=row["solution"])
        return {
            "prompt": format_prompt_math(row["question"]),
            "privileged_context": privileged_context,
        }

    ds = ds.map(_map, remove_columns=ds.column_names)

    if pi_mode == "full" and model is not None and max_prompt_length is not None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        def _teacher_prompt_fits(row):
            # Reconstruct SDFTTrainer's teacher prompt: keep the system turn(s),
            # fold the privileged context into the user turn via the template,
            # then tokenize with add_generation_prompt=True (the trainer's path).
            user_text = TEACHER_PROMPT_TEMPLATE.format(
                prompt=row["prompt"][-1]["content"],
                privileged_context=row["privileged_context"],
            )
            messages = row["prompt"][:-1] + [{"role": "user", "content": user_text}]
            # Mirror SDFTTrainer._tokenize_prompts exactly: batch of one, read
            # input_ids. (tokenize=True without return_dict yields a BatchEncoding
            # whose len() is the field count, not the token count.)
            ids = tok.apply_chat_template(
                [messages], add_generation_prompt=True, tokenize=True, return_dict=True
            )["input_ids"][0]
            return len(ids) <= max_prompt_length

        n_before = len(ds)
        ds = ds.filter(_teacher_prompt_fits, num_proc=4)
        print(f"  pi=full: kept {len(ds)}/{n_before} rows whose teacher prompt "
              f"fits max_prompt_length={max_prompt_length}")

    return ds


def build_hint_dataset(model: str | None, dataset: str, max_samples: int | None):
    """Load precomputed self-hints (train/gen_hints.py) and wrap them as PI.

    The hint cache is keyed by dataset + model slug and stamped with `gen_model`
    (and `dataset` for newer caches); we assert they match `model`/`dataset` so
    training only ever uses hints the trained model itself wrote from the same
    dataset (the "hint-self" arm). Hints are short, so no teacher-prompt length
    filter is needed here.
    """
    if model is None:
        raise ValueError("pi_mode='hint' needs --model to locate its hint cache.")
    path = hint_path(model, dataset)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"No hint cache for {model} on {dataset} at {path}. Generate it first:\n"
            f"  python -m train.gen_hints --model {model} --dataset {dataset} --max-samples <N>"
        )

    ds = load_from_disk(path)
    gen_models = set(ds.unique("gen_model"))
    if gen_models != {model}:
        raise ValueError(
            f"Hint cache at {path} was generated by {gen_models}, not {model!r}. "
            f"Regenerate with `python -m train.gen_hints --model {model} --dataset {dataset}`."
        )
    # `dataset` column is present on newer caches (path-keyed for older ones).
    if "dataset" in ds.column_names and set(ds.unique("dataset")) != {dataset}:
        raise ValueError(
            f"Hint cache at {path} was generated for dataset "
            f"{set(ds.unique('dataset'))}, not {dataset!r}."
        )
    if max_samples is not None:
        if max_samples > len(ds):
            raise ValueError(
                f"Requested {max_samples} hint rows but the cache holds only "
                f"{len(ds)}. Regenerate with a larger --max-samples."
            )
        ds = ds.select(range(max_samples))

    def _map(row):
        return {
            "prompt": format_prompt_math(row["question"]),
            "privileged_context": PI_HINT.format(hint=row["hint"]),
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
        "pi_mode": args.pi_mode,
        "gen_model": args.model if args.pi_mode == "hint" else None,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        "distillation_mode": args.distillation_mode,
        "distillation_alpha": args.distillation_alpha,
        "teacher_model_kind": args.teacher_model_kind,
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
                   help="Training dataset (see utils.DATASET_REGISTRY_TRAIN). For "
                        "--pi-mode hint/full the solution-bearing subset is used.")
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/sdft")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>_<pi-mode>")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Subset the training set")
    # privileged context
    p.add_argument("--pi-mode", default="full", choices=["full", "answer", "hint"],
                   help="Teacher-only privileged context: 'full' worked demo (paper "
                        "default), 'answer' boxed value, or 'hint' precomputed "
                        "self-hints (run train.gen_hints first). See PI_* templates.")
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
    p.add_argument("--generate-from-teacher", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="If set, rollouts are drawn from the teacher-conditioned "
                        "prompt instead of the student prompt (leaves on-policy SDFT).")
    # generation
    p.add_argument("--max-prompt-length", type=int, default=8192,
                   help="Prompts longer than this are left-truncated. The teacher "
                        "prompt carries the privileged context, so 'full' demos need room.")
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
        args.output_root, model_slug, f"{args.dataset}_{args.pi_mode}"
    )
    print(f"model: {model_slug}  dataset: {args.dataset}  pi: {args.pi_mode}  ->  output: {output_dir}")

    train_dataset = build_sdft_dataset(
        args.pi_mode,
        dataset=args.dataset,
        max_samples=args.max_samples,
        model=args.model,
        max_prompt_length=args.max_prompt_length,
    )
    print(f"Loaded {len(train_dataset)} examples")
    print(f"  sample prompt: {train_dataset[0]['prompt'][-1]['content'][:120]!r}")
    print(f"  sample privileged_context: {train_dataset[0]['privileged_context'][:120]!r}")

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
        generate_from_teacher=args.generate_from_teacher,
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

    # Provenance: pi_mode (and, for hint, the generating model) are CLI args, not
    # SDFTConfig fields, so they never reach the pickled training_args.bin. Drop a
    # small run_meta.json at the run root -- shared by every checkpoint under
    # output_dir -- so each run is self-describing when sweeping PIs.
    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(output_dir, "run_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote run metadata -> {meta_path}")

    trainer = SDFTTrainer(
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
