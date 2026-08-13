r"""
Stage 3 of the trained-self-teacher arm: ordinary SDFT, but the teacher's weights come from
train/opsd/train_self_teacher/train_logratio_teacher.py instead of being a frozen copy of the student.

Everything about the student's objective is unchanged from train/opsd/train_sdft.py -- same
SDFTTrainer, same dataset builder, same reverse-KL. The single difference is WHICH teacher scores
the student's tokens, which is exactly the intervention under study: with
`distillation_mode="sampled_token"` the student's per-token advantage is the teacher:student log
ratio, so replacing the teacher replaces the credit-assignment signal and nothing else.

WHY A SEPARATE FILE. `SDFTTrainer._setup_teacher_model` builds the teacher from
`get_config_model_id(self.model.config)` -- the STUDENT's own config id -- and `SDFTConfig` has no
teacher-path field anywhere. There is no way to point stock SDFT at a trained teacher, so the
weights have to be loaded in after construction. train/opsd/train_sdft.py is left untouched.

USE --distillation-mode sampled_token (the default). The E-step constrains the ratio only at the
tokens the student actually sampled; the teacher's off-sample probability mass is unconstrained
and free to have drifted. Under `topk_logits` / `full_logits` the student would chase that drift.

PHYSICAL MODEL BATCH SIZE IS FIXED TO 1. TRL buffers generated samples and splits them into
per-device microbatches before the teacher/student forwards, so --num-generations remains an
independent experiment knob. Use gradient accumulation for a larger effective training batch.

# stage 3 against a hint-PI teacher trained with the per-token asymmetric objective
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_self_teacher.sdft_with_teacher \
    --model Qwen/Qwen3-4B --dataset deepmath --pi-mode hint \
    --teacher-path /mnt/data/ujan/self-distill/outputs/self_teacher/Qwen3-4B/deepmath_hint_asymmetric/final

# smoke
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_self_teacher.sdft_with_teacher \
    --model Qwen/Qwen3-1.7B --pi-mode hint --teacher-path <stage-2 final> \
    --max-steps 2 --max-samples 32 --max-completion-length 256 --save-steps 2
"""

import argparse
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM
from trl.experimental.sdft import SDFTConfig, SDFTTrainer

from train.opsd.train_self_teacher.lib import (
    ASYMMETRIC_OBJECTIVES,
    MODEL_FORWARD_BATCH_SIZE,
    PI_MODES,
    TEACHER_VERSION,
    teacher_prompt_template,
)
from train.opsd.train_sdft import build_sdft_dataset
from utils import DATASET_REGISTRY_TRAIN, validate_resume


def load_teacher_meta(teacher_path: str) -> dict:
    """The stage-2 run_meta.json that sits at the teacher run's root.

    A checkpoint records nothing about the objective that produced it, and this arm's whole
    variable is that objective -- so a teacher without provenance cannot be filed, compared, or
    resumed against. Hard error rather than a warning: the alternative is a stage-3 run whose
    results cannot be attributed to anything.
    """
    meta_path = os.path.join(os.path.dirname(teacher_path.rstrip("/")), "run_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"No run_meta.json beside the teacher at {meta_path}. --teacher-path should point at "
            "a 'final' or 'checkpoint-<N>' dir inside a train_self_teacher.py run directory, "
            "whose root carries the metadata describing how that teacher was trained."
        )
    with open(meta_path) as f:
        return json.load(f)


class TrainedTeacherSDFTTrainer(SDFTTrainer):
    """SDFTTrainer whose frozen teacher is overwritten with trained weights after construction.

    Loading in place, after `super().__init__()`, rather than overriding `_setup_teacher_model`:
    the base method also handles deepspeed/FSDP/PEFT preparation and the EMA callback wiring, and
    reimplementing that to change one thing would drift from TRL. Same shape as
    `SplitDeviceGOLDTrainer` in train/opd/train_gold.py, which relocates GOLD's teacher the same way.
    """

    def __init__(self, *args, teacher_path: str, **kwargs):
        super().__init__(*args, **kwargs)

        kind = self.args.teacher_model_kind
        if kind != "base":
            raise ValueError(
                f"--teacher-model-kind must be 'base' to inject a trained teacher, got {kind!r}. "
                "Under 'live' the teacher IS the student, and under 'ema' it is re-synced towards "
                "the student every teacher_sync_steps -- either way the loaded weights would be "
                "silently discarded and the run would report a trained teacher it is not using."
            )
        teacher = self.accelerator.unwrap_model(self.teacher_model)
        if teacher is self.accelerator.unwrap_model(self.model):
            raise RuntimeError(
                "The teacher and the student are the same module, so loading trained teacher "
                "weights would overwrite the student. Expected a separate frozen teacher for "
                "teacher_model_kind='base' on a non-PEFT model."
            )

        # A parameter sampled before and after, so a load that silently does nothing is caught
        # here rather than showing up as "the trained teacher made no difference" after a full run.
        probe_name, probe = next(iter(teacher.named_parameters()))
        before = probe.detach().clone()

        loaded = AutoModelForCausalLM.from_pretrained(
            teacher_path, dtype=torch.bfloat16, trust_remote_code=True
        )
        teacher.load_state_dict(loaded.state_dict())
        del loaded
        gc.collect()

        teacher.requires_grad_(False)  # re-assert: only the student trains in the M-step
        teacher.eval()
        if torch.equal(before, probe.detach()):
            print(f"  WARNING: teacher weights are byte-identical after loading {teacher_path} "
                  f"(checked {probe_name}). Either the teacher was trained for 0 steps, or "
                  f"--teacher-path points at the untrained base model. This run would be plain "
                  f"SDFT while every log said otherwise.")
        else:
            print(f"Loaded trained teacher <- {teacher_path}")


# ---------------------------------------------------------------------------
# Resume metadata
# ---------------------------------------------------------------------------


def build_run_meta(args, teacher_meta: dict, num_train_examples: int) -> dict:
    return {
        "method": "sdft_trained_teacher",
        # Teacher-state identity, carried over from the stage-2 run. Strict key: a checkpoint that
        # never recorded it was trained against a differently-defined teacher, and its student
        # weights are mid-trajectory towards a different target.
        "teacher_version": teacher_meta["teacher_version"],
        "teacher_path": args.teacher_path,
        # Which teacher, trained how. The objective and PI are the variables under study, so they
        # belong in this run's provenance even though they were chosen one stage earlier.
        "teacher_objective": teacher_meta["objective"],
        "teacher_pi_mode": teacher_meta.get("pi_mode"),
        "teacher_sampled_logp_anchor": teacher_meta.get("sampled_logp_anchor"),
        "teacher_asym_margin": teacher_meta.get("asym_margin"),
        "teacher_asym_lift_alpha": teacher_meta.get("asym_lift_alpha"),
        "teacher_asym_anchor_weight": teacher_meta.get("asym_anchor_weight"),
        "teacher_asym_weighting": teacher_meta.get("asym_weighting"),
        "teacher_prompt_template": teacher_prompt_template(args.pi_mode),
        "model": args.model,
        "pi_mode": args.pi_mode,
        "gen_model": args.model if args.pi_mode == "hint" else None,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        "distillation_mode": args.distillation_mode,
        "distillation_alpha": args.distillation_alpha,
        "teacher_model_kind": "base",
        # resume-critical: on resume the LR comes from the CHECKPOINT, not the CLI.
        "learning_rate": args.learning_rate,
        # resume-critical: dataset order + batch chunking.
        "seed": args.seed,
        "per_device_train_batch_size": MODEL_FORWARD_BATCH_SIZE,
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
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="Student to train. MUST be the model the teacher was initialised from.")
    p.add_argument("--teacher-path", required=True,
                   help="A 'final' or 'checkpoint-<N>' dir from train/opsd/train_self_teacher/train_logratio_teacher.py. Its "
                        "run root must carry run_meta.json (the teacher's provenance).")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()))
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/sdft_tt",
                   help="Distinct from outputs/sdft: run_meta.json lives at the run root, so "
                        "sharing a root would have the two arms overwrite each other's provenance.")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>_<pi-mode>_<teacher-objective>")
    p.add_argument("--max-samples", type=int, default=None)
    # privileged context -- must match the teacher's
    p.add_argument("--pi-mode", default="hint", choices=list(PI_MODES),
                   help="MUST match the --pi-mode the teacher was trained under: the teacher was "
                        "calibrated conditioned on that context and is meaningless under another. "
                        "Verified against the teacher's run_meta.json below.")
    # SDFT objective (mirrors train_sdft.py)
    p.add_argument("--distillation-mode", default="sampled_token",
                   choices=["topk_logits", "full_logits", "sampled_token"],
                   help="Keep 'sampled_token'. The E-step constrains the ratio only where the "
                        "student sampled; the teacher's off-sample mass is unconstrained and the "
                        "other two modes would have the student chase it.")
    p.add_argument("--distillation-alpha", type=float, default=1.0,
                   help="1.0 = reverse KL, required by sampled_token.")
    p.add_argument("--distillation-topk", type=int, default=100)
    p.add_argument("--distillation-is-clip", type=float, default=2.0)
    p.add_argument("--num-loss-tokens-to-skip", type=int, default=0)
    # generation
    p.add_argument("--max-prompt-length", type=int, default=8192)
    p.add_argument("--max-completion-length", type=int, default=8192)
    # optimization
    p.add_argument("--num-generations", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="Matches train_sdft.py so the two arms differ only in the teacher.")
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    # vLLM
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3,
                   help="Colocate. Keep modest: the student, the separate trained teacher and the "
                        "engine share the GPU here, one model more than train_grpo.py.")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    # resume
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume dir ('checkpoint-<step>'). The teacher needs no restoring -- it is "
                        "frozen and re-loaded from --teacher-path. Pass the SAME --model, "
                        "--dataset, --pi-mode, --teacher-path, --seed and gradient accumulation "
                        "as the original run (verified against its run_meta.json).")
    p.add_argument("--force-resume", action="store_true")
    args = p.parse_args()

    if args.distillation_mode == "sampled_token" and args.distillation_alpha != 1.0:
        p.error("--distillation-mode sampled_token requires --distillation-alpha 1.0 (reverse KL).")

    teacher_meta = load_teacher_meta(args.teacher_path)
    if teacher_meta.get("teacher_version") != TEACHER_VERSION:
        p.error(
            f"Teacher version {teacher_meta.get('teacher_version')!r} is incompatible with "
            f"the current {TEACHER_VERSION!r} objective semantics. Retrain stage 2."
        )
    if teacher_meta.get("objective") not in ASYMMETRIC_OBJECTIVES:
        p.error(
            f"Teacher objective {teacher_meta.get('objective')!r} is unsupported; expected one "
            f"of {ASYMMETRIC_OBJECTIVES}. Retrain stage 2 with a current asymmetric objective."
        )
    # The teacher was calibrated conditioned on one privileged context and stitched with one
    # template. Feeding it either of the others at M-step time puts it out of distribution in a
    # way nothing downstream would report.
    if teacher_meta.get("pi_mode") != args.pi_mode:
        p.error(
            f"--pi-mode {args.pi_mode!r} does not match the teacher's "
            f"{teacher_meta.get('pi_mode')!r} (from {args.teacher_path}). The teacher's ratio is "
            "only calibrated under the context it was trained on."
        )
    if teacher_meta.get("model") != args.model:
        p.error(
            f"--model {args.model!r} does not match the model the teacher was initialised from "
            f"({teacher_meta.get('model')!r}). The ratio is relative to THAT student."
        )
    expected_template = teacher_prompt_template(args.pi_mode)
    if teacher_meta.get("teacher_prompt_template") != expected_template:
        p.error(
            "The teacher was trained with teacher_prompt_template "
            f"{teacher_meta.get('teacher_prompt_template')!r} but this run would use "
            f"{expected_template!r}. The difference is invisible in the logs and changes the "
            "context the teacher scores under."
        )

    model_slug = args.model.rstrip("/").split("/")[-1]
    teacher_objective = teacher_meta["objective"]
    output_dir = args.output_dir or os.path.join(
        args.output_root, model_slug, f"{args.dataset}_{args.pi_mode}_{teacher_objective}"
    )
    print(f"model: {model_slug}  dataset: {args.dataset}  pi: {args.pi_mode}  ->  {output_dir}")
    print(f"  teacher: {args.teacher_path}")
    print(f"    objective={teacher_meta.get('objective')}  "
          f"sampled_logp_anchor={teacher_meta.get('sampled_logp_anchor')}  "
          f"version={teacher_meta.get('teacher_version')}")

    # `none` has no build path in train_sdft (its ladder is full/answer/hint), so its empty
    # privileged context is attached here. Combined with the concatenating template, the teacher
    # then reads exactly the student's prompt -- which is what makes it the matched control.
    if args.pi_mode == "none":
        from utils import format_prompt_math, load_train_dataset

        train_dataset = load_train_dataset(args.dataset, max_samples=args.max_samples).map(
            lambda row: {
                "prompt": format_prompt_math(row["question"]),
                "privileged_context": "",
            },
            remove_columns=None,
        )
        train_dataset = train_dataset.select_columns(["prompt", "privileged_context"])
    else:
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
        distillation_mode=args.distillation_mode,
        distillation_alpha=args.distillation_alpha,
        distillation_topk=args.distillation_topk,
        distillation_is_clip=None if args.distillation_is_clip < 0 else args.distillation_is_clip,
        num_loss_tokens_to_skip=args.num_loss_tokens_to_skip,
        # The teacher is a separate frozen module, which is the only kind that can carry injected
        # weights (see TrainedTeacherSDFTTrainer).
        teacher_model_kind="base",
        generate_from_teacher=False,
        teacher_prompt_template=expected_template,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        max_steps=args.max_steps,
        per_device_train_batch_size=MODEL_FORWARD_BATCH_SIZE,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    meta = build_run_meta(args, teacher_meta, len(train_dataset))
    if args.resume_from_checkpoint:
        validate_resume(
            args.resume_from_checkpoint, meta, args.force_resume,
            strict_keys=("teacher_version",),
        )
    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote run metadata -> {os.path.join(output_dir, 'run_meta.json')}")

    trainer = TrainedTeacherSDFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        teacher_path=args.teacher_path,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved model -> {final_dir}")


if __name__ == "__main__":
    main()
