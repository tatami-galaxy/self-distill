"""Full-parameter GRPO training for a learned self-hint generator.

The policy sees a math problem plus its worked solution and generates a short hint.
Its frozen initial weights act as the hinted teacher. For each sampled hint the reward is

    alpha * fraction_correct_teacher_rollouts
      - generated_hint_tokens / hint_budget
      - gamma * sampled_reverse_KL_on_student_rollouts.

The first experiment is intentionally single-GPU. Colocated vLLM produces the hints and enters
level-2 sleep before the reward function lazily loads/runs the frozen Hugging Face teacher.

# one-H100 smoke test
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_hint_gen.train_hint_gen \
    --model Qwen/Qwen3-4B --dataset deepmath --max-samples 32 --max-steps 2

# initial run
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_hint_gen.train_hint_gen \
    --model Qwen/Qwen3-4B --dataset deepmath --max-samples 2048 \
    --alpha 1.0 --gamma 1.0 --teacher-rollouts 2 --transfer-rollouts 4
"""

from __future__ import annotations

import argparse
import json
import os

from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from train.opsd.train_hint_gen.lib import (
    HINT_GEN_VERSION,
    HintRewardConfig,
    build_hint_grpo_dataset,
    make_reward_function,
)
from utils import DATASET_REGISTRY_TRAIN, validate_resume


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN))
    p.add_argument("--rollout-root", default="data/rollouts")
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/hint_gen")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--mixed-only", action="store_true")
    # Composite reward.
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--hint-budget", type=int, default=128)
    p.add_argument("--teacher-rollouts", type=int, default=2)
    p.add_argument("--transfer-rollouts", type=int, default=4)
    p.add_argument("--teacher-max-completion-length", type=int, default=4096)
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--teacher-top-p", type=float, default=1.0)
    p.add_argument("--invalid-penalty", type=float, default=1.0)
    p.add_argument(
        "--recompute-student-logps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute unhinted logps with the reward-time frozen model, then cache them "
        "in CPU memory. This avoids mixing clean-process cached numerics with HF forwards "
        "after vLLM initialization.",
    )
    p.add_argument(
        "--clamp-transfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp the rollout-averaged Monte Carlo KL estimate at zero.",
    )
    # GRPO generation and optimization.
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--generator-temperature", type=float, default=1.0)
    p.add_argument("--generator-top-p", type=float, default=1.0)
    p.add_argument("--generator-kl-beta", type=float, default=0.0)
    p.add_argument("--loss-type", default="dapo", choices=["dapo", "grpo", "dr_grpo"])
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--lr-scheduler-type", default="constant")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    # Colocated vLLM. A generation batch is fixed to one complete hint group.
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2)
    p.add_argument("--vllm-max-model-length", type=int, default=32768)
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--vllm-enable-sleep-mode", action=argparse.BooleanOptionalAction, default=True
    )
    # Bookkeeping.
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-completions-to-print", type=int, default=1)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--force-resume", action="store_true")
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.num_generations < 2:
        raise ValueError("num_generations must be >= 2")
    if args.per_device_train_batch_size != 1:
        raise ValueError(
            "The first hint-generator experiment fixes physical batch size to 1; use gradient "
            "accumulation for a larger effective batch."
        )
    if not args.use_vllm:
        raise ValueError(
            "The single-GPU full-parameter experiment requires colocated vLLM sleep mode to "
            "free memory before teacher inference."
        )
    if not args.vllm_enable_sleep_mode:
        raise ValueError("Enable vLLM sleep mode so the frozen teacher fits during reward scoring.")
    if args.generator_temperature <= 0:
        raise ValueError("generator_temperature must be > 0 for GRPO exploration")
    if not 0 < args.generator_top_p <= 1:
        raise ValueError("generator_top_p must be in (0, 1]")
    HintRewardConfig(
        model=args.model,
        dataset=args.dataset,
        hint_budget=args.hint_budget,
        teacher_rollouts=args.teacher_rollouts,
        transfer_rollouts=args.transfer_rollouts,
        teacher_max_completion_length=args.teacher_max_completion_length,
        teacher_temperature=args.teacher_temperature,
        teacher_top_p=args.teacher_top_p,
        alpha=args.alpha,
        gamma=args.gamma,
        invalid_penalty=args.invalid_penalty,
    ).validate()


def build_run_meta(args: argparse.Namespace, num_train_examples: int) -> dict:
    return {
        "method": "hint_gen_grpo",
        "hint_gen_version": HINT_GEN_VERSION,
        "model": args.model,
        "teacher_model": args.model,
        "dataset": args.dataset,
        "rollout_root": args.rollout_root,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        "mixed_only": args.mixed_only,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "hint_budget": args.hint_budget,
        "teacher_rollouts": args.teacher_rollouts,
        "transfer_rollouts": args.transfer_rollouts,
        "teacher_max_completion_length": args.teacher_max_completion_length,
        "teacher_temperature": args.teacher_temperature,
        "teacher_top_p": args.teacher_top_p,
        "invalid_penalty": args.invalid_penalty,
        "recompute_student_logps": args.recompute_student_logps,
        "clamp_transfer": args.clamp_transfer,
        "num_generations": args.num_generations,
        "generator_temperature": args.generator_temperature,
        "generator_top_p": args.generator_top_p,
        "generator_kl_beta": args.generator_kl_beta,
        "loss_type": args.loss_type,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    model_slug = args.model.rstrip("/").split("/")[-1]
    default_name = f"{args.dataset}_a{args.alpha:g}_g{args.gamma:g}"
    output_dir = args.output_dir or os.path.join(args.output_root, model_slug, default_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # prompt, question, final_answer
    train_dataset = build_hint_grpo_dataset(
        model=args.model,
        dataset=args.dataset,
        tokenizer=tokenizer,
        rollout_root=args.rollout_root,
        transfer_rollouts=args.transfer_rollouts,
        max_samples=args.max_samples,
        max_model_length=args.vllm_max_model_length,
        hint_budget=args.hint_budget,
        mixed_only=args.mixed_only,
    )
    print(f"model: {args.model}  dataset: {args.dataset}  output: {output_dir}")
    print(f"  examples: {len(train_dataset)}")
    print(
        f"  reward: {args.alpha:g} * sufficiency - length/{args.hint_budget} "
        f"- {args.gamma:g} * transfer"
    )

    reward_config = HintRewardConfig(
        model=args.model,
        dataset=args.dataset,
        rollout_root=args.rollout_root,
        hint_budget=args.hint_budget,
        teacher_rollouts=args.teacher_rollouts,
        transfer_rollouts=args.transfer_rollouts,
        teacher_max_completion_length=args.teacher_max_completion_length,
        teacher_temperature=args.teacher_temperature,
        teacher_top_p=args.teacher_top_p,
        alpha=args.alpha,
        gamma=args.gamma,
        invalid_penalty=args.invalid_penalty,
        recompute_student_logps=args.recompute_student_logps,
        clamp_transfer=args.clamp_transfer,
    )
    reward_func = make_reward_function(reward_config, tokenizer)

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.hint_budget,
        temperature=args.generator_temperature,
        top_p=args.generator_top_p,
        chat_template_kwargs={"enable_thinking": False},
        loss_type=args.loss_type,
        beta=args.generator_kl_beta,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_enable_sleep_mode=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,
    )

    meta = build_run_meta(args, len(train_dataset))
    if args.resume_from_checkpoint:
        validate_resume(
            args.resume_from_checkpoint,
            meta,
            args.force_resume,
            strict_keys=("hint_gen_version",),
        )
    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run_meta.json"), "w") as handle:
            json.dump(meta, handle, indent=2)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved hint generator -> {final_dir}")


if __name__ == "__main__":
    main()
