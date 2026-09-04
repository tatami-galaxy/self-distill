"""
Train the OPD baseline -- on-policy distillation from a strong *external* teacher,
via TRL's `trl.experimental.gold.GOLDTrainer` with **vLLM** generation.

Why GOLD instead of GKDTrainer: GKD generates on-policy with HF `model.generate`
(no vLLM), which at 8192-token reasoning traces is ~20 tok/s -> ~15 min per micro-
step -> intractable. GOLD inherits GKD's generalized-JSD / on-policy scheduling but
adds the shared GRPO-style vLLM path (colocate engine + per-step weight sync), so
student rollouts are as fast as the SDFT runs. This is exactly the GKD+vLLM ask of
huggingface/trl#4568.

Same-vocab, so NO Universal Logit Distillation: `use_uld_loss=False` routes to the
standard `generalized_jsd_loss` (the GKD loss). Verified: Qwen3-1.7B <-> Qwen3-30B-
A3B-Thinking-2507 share the 151669 vocab, so the teacher-embedding resize is a no-op.

Mapping to the SDFT/OPSD runs (only the teacher should differ -- capability, no PI):
  * use_uld_loss=False -- token-level JSD over the shared vocab (not ULD).
  * lmbda=1.0          -- fully on-policy (student generates every completion).
  * beta=1.0           -- reverse KL KL(student||teacher), == SDFT distillation_alpha=1.0.
                          (GOLD beta: 0=forward KL, 1=reverse KL, 0.5=JSD.)
  * seq_kd=False       -- token-level distillation, not SFT on teacher samples.
  * temperature=1.0, top_p=1.0, top_k=0 -- same unfiltered sampling distribution
                          as SDFT. GOLD otherwise defaults to temperature=0.9 and
                          top_p=0.95; temperature also rescales its distillation logits.
  * learning_rate 1e-5 -- GOLD defaults to 1e-7; we override to match train_sdft.py.

GPU topology (bf16 30B teacher + colocate vLLM don't share one GPU comfortably):
  * teacher on its OWN GPU via --teacher-device cuda:1 (device_map load; accelerate
    IO hooks move inputs->teacher and logits back, and prepare_model won't relocate
    a device_map'd model -- so it never lands on cuda:0 to OOM the vLLM init).
  * student training + colocate student-vLLM engine on cuda:0.
  Launch with 2 visible GPUs, e.g. CUDA_VISIBLE_DEVICES=6,7.

CUDA_VISIBLE_DEVICES=6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python -m train.opd.train_gold \
    --model Qwen/Qwen3-1.7B --dataset deepmath --teacher-model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --teacher-device cuda:1 --max-samples 8192

Validated at 8192 completions: GPU6 (student+vLLM) peak ~60GB, GPU7 (teacher) ~64GB,
~58s / optimizer step at grad-accum 2. expandable_segments avoids fragmentation OOM.

Resume (--resume-from-checkpoint) restores weights/optimizer/scheduler/RNG and skips seen data;
pass the same hyperparameters (verified against run_meta.json). The teacher needs no restoring --
it is frozen and re-loaded from --teacher-model. --max-steps is the TOTAL budget and is free to
raise, but the LEARNING RATE comes from the checkpoint rather than the command line, so a changed
--learning-rate is refused rather than silently ignored (full rules in utils.validate_resume). At
the defaults (num_generations 1 -> 16 prompts per step) one epoch of full DeepMath is ~6,438 steps.
"""

import argparse
import json
import os

import torch
from torch import nn
from trl.experimental.gold import GOLDConfig, GOLDTrainer

from utils import (
    DATASET_REGISTRY_TRAIN,
    format_prompt_math,
    load_train_dataset,
    validate_resume,
)


# GOLD's library defaults differ from SDFT's. Keep these explicit so upgrading TRL
# cannot silently change the OPD arm's rollout or distillation distribution.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
GOLD_RESUME_STRICT_KEYS = ("temperature", "top_p", "top_k")


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def build_run_meta(args, num_train_examples: int) -> dict:
    """Provenance + resume-critical config for run_meta.json. The second block must
    match on resume for the seeded data-skip to land on the same examples."""
    return {
        "method": "gold_opd_vllm",
        "model": args.model,
        "teacher_model": args.teacher_model,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        "use_uld_loss": False,
        "lmbda": args.lmbda,
        "beta": args.beta,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_completion_length": args.max_completion_length,
        # resume-critical: on resume the LR comes from the CHECKPOINT, not the CLI (see
        # validate_resume), so recording it turns a silently-ignored change into an error.
        "learning_rate": args.learning_rate,
        # resume-critical: dataset order (seed, length) + batch chunking. If any of
        # these differ from the original run, the shuffle permutation or the per-step
        # batch boundary shifts and the skip resumes on the wrong data.
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
    }


# ---------------------------------------------------------------------------
# Split-device teacher: keep the frozen 60GB teacher off the student GPU
# ---------------------------------------------------------------------------
#
# GOLD's colocate vLLM engine + the training student + the full-vocab JSD loss
# already fill the student GPU; the 60GB teacher on top OOMs at long sequences.
# device_map alone does NOT work: GOLD's accelerator.prepare_model relocates the
# teacher back onto the student GPU (cuda:0) after loading. So instead we let it
# load there during __init__ (fits alongside vLLM at ~81GB), then relocate it to
# its own GPU *after* init and forward across devices -- transparently to GOLD's
# compute_loss, which just calls self.teacher_model(input_ids=..., attention_mask=...).


def _first_tensor_device(args, kwargs) -> torch.device | None:
    for v in list(args) + list(kwargs.values()):
        if torch.is_tensor(v):
            return v.device
    return None


def _move(v, device):
    return v.to(device) if torch.is_tensor(v) else v


def relocate_teacher_(teacher: nn.Module, device: str) -> nn.Module:
    """Move the frozen teacher to `device` and patch its `forward` so any call --
    positional or keyword, however GOLD reaches it (directly or via unwrap_model) --
    moves tensor inputs onto `device` and returns logits on the caller's device.

    We patch the model's own forward rather than wrapping it in a new module, so the
    object stays a real PreTrainedModel (config/eval/generate intact) for GOLD's
    accelerator.unwrap_model(self.teacher_model) calls.

    First strip any accelerate AlignDevicesHook: prepare_model leaves one whose
    pre_forward force-moves inputs back to its recorded execution_device (cuda:0),
    which would override our relocation and re-trigger the device mismatch.
    """
    from accelerate.hooks import remove_hook_from_module

    remove_hook_from_module(teacher, recurse=True)
    dev = torch.device(device)
    teacher.to(dev).eval()
    orig_forward = teacher.forward

    def forwarding(*args, **kwargs):
        src = _first_tensor_device(args, kwargs) or dev
        args = tuple(_move(a, dev) for a in args)
        kwargs = {k: _move(v, dev) for k, v in kwargs.items()}
        out = orig_forward(*args, **kwargs)
        if getattr(out, "logits", None) is not None:
            out.logits = out.logits.to(src)
        return out

    teacher.forward = forwarding
    return teacher


class SplitDeviceGOLDTrainer(GOLDTrainer):
    """GOLDTrainer that relocates the frozen teacher onto `teacher_device` after the
    base __init__, freeing the student GPU for the colocate vLLM engine + JSD loss."""

    def __init__(self, *args, teacher_device: str = "cuda:1", **kwargs):
        super().__init__(*args, **kwargs)
        teacher = self.accelerator.unwrap_model(self.teacher_model)
        self.teacher_model = relocate_teacher_(teacher, teacher_device)
        torch.cuda.empty_cache()
        print(f"Teacher relocated to {teacher_device}; student on {self.model.device}")


# ---------------------------------------------------------------------------
# GOLD dataset loader (conversational messages; identical to train_gkd.py)
# ---------------------------------------------------------------------------


def build_gold_dataset(dataset: str = "deepmath", max_samples: int | None = None):
    """`messages` = [system, user, assistant] with the system+user turns IDENTICAL
    to the SDFT / eval prompt (format_prompt_math). GOLD's ChatML collator reads
    messages[:-1] as the prompt; at lmbda=1.0 the assistant turn is regenerated
    on-policy, but the collator still needs it for the prompt/completion boundary,
    so we use the gold boxed answer (valid too for any lmbda < 1.0)."""
    ds = load_train_dataset(dataset, max_samples=max_samples)

    def _map(row):
        messages = format_prompt_math(row["question"]) + [
            {"role": "assistant", "content": "\\boxed{" + str(row["final_answer"]) + "}"}
        ]
        return {"messages": messages}

    return ds.map(_map, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B",
                   help="Student to train. Must share a tokenizer with --teacher-model.")
    p.add_argument("--teacher-model", default="Qwen/Qwen3-30B-A3B-Thinking-2507",
                   help="Strong external OPD teacher. Same tokenizer as --model.")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()),
                   help="Training dataset (see utils.DATASET_REGISTRY_TRAIN).")
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/gold")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>_<teacher>")
    p.add_argument("--max-samples", type=int, default=None)
    # GOLD / distillation objective
    p.add_argument("--lmbda", type=float, default=1.0,
                   help="On-policy fraction: 1.0 = student generates every completion.")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Generalized-JSD interpolation: 1.0 = reverse KL (SDFT default), "
                        "0.0 = forward KL, 0.5 = JSD.")
    p.add_argument("--seq-kd", action=argparse.BooleanOptionalAction, default=False,
                   help="Sequence-level KD (SFT on teacher samples) instead of on-policy.")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help="Sampling and distillation-logit temperature. Default 1.0 "
                        "matches SDFT (rather than GOLD's library default 0.9).")
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P,
                   help="Nucleus-sampling threshold. Default 1.0 matches SDFT and "
                        "keeps the full token support.")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help="Top-k sampling cutoff; 0 disables it, matching SDFT.")
    p.add_argument("--max-completion-length", type=int, default=8192,
                   help="On-policy completion budget (GOLD's max_completion_length). "
                        "Matches the SDFT runs.")
    # teacher placement
    p.add_argument("--teacher-device", default="cuda:1",
                   help="device_map target for the bf16 teacher so it loads on its own "
                        "GPU (needs >=2 visible GPUs). Pass '' to colocate it with the "
                        "student (single GPU; only viable at low --vllm-gpu-mem).")
    # vLLM (student generation)
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True,
                   help="Generate student rollouts with vLLM. --no-use-vllm falls back "
                        "to HF generate (the slow GKD path).")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.25,
                   help="Fraction of the student GPU vLLM may reserve (colocate). Keep "
                        "modest; the training student + JSD loss share the GPU. 0.25 "
                        "validated at 8192 (GPU6 peak ~60GB).")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    p.add_argument("--num-generations", type=int, default=1,
                   help="Rollouts per prompt (must divide the global generation batch).")
    # length
    p.add_argument("--max-length", type=int, default=10240,
                   help="Total prompt+completion cap (also the default vLLM context). "
                        "Must exceed prompt + --max-completion-length.")
    # optimization (mirrors train_sdft.py so the OPD baseline is comparable)
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="GOLD defaults to 1e-7; we override to match train_sdft.py.")
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup",
                            "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="With vLLM handling generation, checkpointing no longer slows "
                        "rollouts (only the loss forward), so leaving it on is fine.")
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    # resume
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume from a checkpoint dir (path ending in "
                        "'checkpoint-<step>'). Restores weights/optimizer/scheduler/RNG "
                        "and skips already-seen examples. Pass the SAME --model, "
                        "--teacher-model, --dataset, --max-samples, --seed and batch config "
                        "as the original run (verified against its run_meta.json). --max-steps "
                        "is the TOTAL budget: training continues from the checkpoint's step "
                        "up to --max-steps.")
    p.add_argument("--force-resume", action="store_true",
                   help="Downgrade a run_meta.json hyperparameter mismatch from an error to "
                        "a warning (use only if you understand the data-skip consequences).")
    args = p.parse_args()

    model_slug = args.model.rstrip("/").split("/")[-1]
    teacher_slug = args.teacher_model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(
        args.output_root, model_slug, f"{args.dataset}_{teacher_slug}"
    )
    print(f"model: {model_slug}  teacher: {teacher_slug}  dataset: {args.dataset}  "
          f"->  output: {output_dir}")

    train_dataset = build_gold_dataset(args.dataset, max_samples=args.max_samples)
    print(f"Loaded {len(train_dataset)} examples")

    # GOLD builds its ChatML collator in __init__ before SFTTrainer loads a tokenizer
    # from the model string; pass it explicitly (as in train_gkd.py).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Teacher loads bf16 on the student GPU during init (fits alongside vLLM), then
    # SplitDeviceGOLDTrainer relocates it to --teacher-device before training.
    teacher_init_kwargs = {"dtype": "bfloat16", "trust_remote_code": True}

    training_args = GOLDConfig(
        output_dir=output_dir,
        # distillation objective
        use_uld_loss=False,  # same vocab -> token-level generalized JSD, not ULD
        lmbda=args.lmbda,
        beta=args.beta,
        seq_kd=args.seq_kd,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        # teacher
        teacher_model_name_or_path=args.teacher_model,
        teacher_model_init_kwargs=teacher_init_kwargs,
        # vLLM student generation
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        # data / length
        max_length=args.max_length,
        # optimization
        learning_rate=args.learning_rate,
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

    meta = build_run_meta(args, len(train_dataset))

    # Before resuming, verify the run this checkpoint came from used the same config
    # (its run_meta.json sits beside the checkpoint), so the seeded data-skip lands
    # on the examples that were actually left untrained.
    if args.resume_from_checkpoint:
        validate_resume(
            args.resume_from_checkpoint,
            meta,
            args.force_resume,
            strict_keys=GOLD_RESUME_STRICT_KEYS,
        )

    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    trainer_kwargs = dict(
        model=args.model,
        teacher_model=args.teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    if args.teacher_device:
        if torch.cuda.device_count() < 2:
            raise ValueError(
                f"--teacher-device {args.teacher_device} needs >=2 visible GPUs "
                f"(got {torch.cuda.device_count()}). Launch with e.g. "
                f"CUDA_VISIBLE_DEVICES=6,7, or pass --teacher-device '' to colocate."
            )
        trainer = SplitDeviceGOLDTrainer(teacher_device=args.teacher_device, **trainer_kwargs)
    else:
        trainer = GOLDTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved model -> {final_dir}")


if __name__ == "__main__":
    main()
