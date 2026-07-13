"""
# single GPU, colocate vLLM
CUDA_VISIBLE_DEVICES=0 uv run python -m train.train_grpo \
    --model Qwen/Qwen3-4B --max-samples 8192

# multiple GPUs: data-parallel via accelerate, one process per GPU. In colocate
# mode each process trains the policy and runs its own vLLM engine on its GPU.
# The global batch = num_processes * per-device-train-batch-size * grad-accum,
# and num_generations must divide it (default 8 divides 4 * 8 * 4 = 128).
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
    -m train.train_grpo --model Qwen/Qwen3-4B --max-samples 8192
"""

import argparse
import os

from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

from utils import DATASET_REGISTRY_TRAIN, format_prompt_math, load_train_dataset


# ---------------------------------------------------------------------------
# GRPO dataset loader
# ---------------------------------------------------------------------------


def build_grpo_dataset(dataset: str = "deepmath", max_samples: int | None = None):
    """
    `prompt`   -- conversational [system, user] messages (same format as eval);
                  GRPOTrainer applies the chat template and generates from it.
    `solution` -- the gold answer wrapped in \\boxed{}. accuracy_reward parses
                  this column directly with math_verify, and bare values (e.g.
                  "204", "\\frac{1}{2}") don't reliably parse without the anchor.

    Loaders yield (question, final_answer, solution); the worked solution is unused
    here and dropped.
    """
    ds = load_train_dataset(dataset, max_samples=max_samples)

    def _map(row):
        return {
            "prompt": format_prompt_math(row["question"]),
            "solution": "\\boxed{" + str(row["final_answer"]) + "}",
        }

    return ds.map(_map, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()),
                   help="Training dataset (see utils.DATASET_REGISTRY_TRAIN).")
    p.add_argument("--output-root", default="outputs/grpo")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Subset the training set")
    # GRPO
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--num-generations", type=int, default=8,
                   help="Rollouts per prompt (the GRPO group size).")
    p.add_argument("--loss-type", default="dapo",
                   choices=["dapo", "grpo", "dr_grpo"],
                   help="Token-loss aggregation")
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
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
    p.add_argument("--per-device-train-batch-size", type=int, default=1,
                   help="should be a multiple of --num-generations.")
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
    p.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=False,
                   help="Log sample completions to the run's report backend.")
    p.add_argument("--num-completions-to-print", type=int, default=1,
                   help="How many completions to print/log when --log-completions is set.")
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(args.output_root, model_slug, args.dataset)
    print(f"model: {model_slug}  dataset: {args.dataset}  ->  output: {output_dir}")

    train_dataset = build_grpo_dataset(args.dataset, max_samples=args.max_samples)
    print(f"Loaded {len(train_dataset)} examples")
    print(f"  sample prompt: {train_dataset[0]['prompt'][-1]['content'][:120]!r}")
    print(f"  sample solution: {train_dataset[0]['solution']!r}")

    training_args = GRPOConfig(
        output_dir=output_dir,
        # GRPO objective
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        loss_type=args.loss_type,
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
        # model load: GRPOTrainer instantiates the policy from the model string;
        # dtype defaults to fp32 there, so bf16 must be requested explicitly.
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        # bookkeeping
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        report_to=args.report_to,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved model -> {final_dir}")


if __name__ == "__main__":
    main()
