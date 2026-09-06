"""Train online SDPO-initialized soft actor-critic with privileged information.

The student samples an untruncated temperature-1 rollout. A separate frozen copy
of the same model reads privileged information and supplies both teacher
log-probabilities and hidden states. The critic is

    Q(s, a) = log pi_T(a | s, PI) + W_lm[a]^T A h_T(s),

where only the zero-initialized square matrix A is trained by the critic loss.
Actor and critic each update once per generated effective batch, synchronously,
with beta=gamma=1 and no critic warmup or target network.

Single GPU:
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_sac.train_sac \
    --model Qwen/Qwen3-1.7B --dataset deepmath --pi-mode full \
    --soft-v-estimator topk --soft-v-topk 100 --lam 0

Four GPUs, data parallel:
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
    -m train.opsd.train_sac.train_sac --model Qwen/Qwen3-4B \
    --dataset deepmath --pi-mode full --max-samples 8192


Current problems : 

The concern is that the current parameterization :
c_phi(s,a)=w_a^T A_\phi h_T(s) must fit the return offset through an action-dependent projection. 
It has no explicit state-only scalar component. 
Fitting that offset can introduce unwanted differences between actions.
The damaging route is the learned critic changing subsequent actor coefficients.

There is an initialization-versus-objective mismatch : 
Initializing Q_0=log pi_T gives the desired self-distillation actor gradient. 
It does not make Q_0 a Bellman-consistent soft value function for binary correctness rewards.
Consequently, CRITIC LEARNING HAS NO REQUIREMENT TO PRESERVE THE TEACHER'S PREFERENCES. 
The teacher term remains frozen in the formula, but the residual can cancel or overwhelm it. 
Even outcome-only regression would eventually move Q away from log-probabilities.

For λ=0, bootstraps from the initialization instead of exposing the full return immediately. 
That explains the much smaller early regression error. 
It does not eliminate the underlying mismatch: the λ=0 run also worsens.

"""

import argparse
import json
import os

from trl.rewards import accuracy_reward

from train.opsd.train_sdft import (
    build_sdft_dataset,
    hint_generator_run_slug,
    resolve_hint_source,
)
from utils import DATASET_REGISTRY_TRAIN, TEACHER_PROMPT_TEMPLATE, validate_resume

from .lib import Q_HEAD_ARCHITECTURE
from .trainer import GAMMA, SOFT_BETA, SACConfig, SACTrainer


def sac_run_name(dataset: str, pi_slug: str, soft_v_estimator: str) -> str:
    """Name an SAC run by every architectural choice currently under study."""
    return f"{dataset}_{pi_slug}_{soft_v_estimator}_{Q_HEAD_ARCHITECTURE}"


def build_run_meta(args, num_train_examples: int) -> dict:
    hint_generator_model = None
    hint_cache_path = None
    hint_source = None
    if args.pi_mode == "hint":
        hint_generator_model, hint_cache_path = resolve_hint_source(
            args.model,
            args.dataset,
            getattr(args, "hint_generator_model", None),
            getattr(args, "hint_cache", None),
        )
        hint_cache_path = os.path.abspath(hint_cache_path)
        hint_source = "self" if hint_generator_model == args.model else "trained_generator"
    elif args.pi_mode == "rollout":
        hint_generator_model = args.model

    return {
        "method": "online_sac_pi",
        "model": args.model,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        "pi_mode": args.pi_mode,
        "gen_model": hint_generator_model,
        "hint_generator_model": hint_generator_model if args.pi_mode == "hint" else None,
        "hint_cache": hint_cache_path,
        "hint_source": hint_source,
        "rollout_pi_root": args.rollout_pi_root if args.pi_mode == "rollout" else None,
        "rollout_pi_sample_idx": args.rollout_pi_sample_idx if args.pi_mode == "rollout" else None,
        "reward": "accuracy_reward",
        "teacher_model_kind": "base",
        "q_head_architecture": Q_HEAD_ARCHITECTURE,
        "q_parameterization": "frozen_lm_head_zero_linear_residual",
        "soft_v_estimator": args.soft_v_estimator,
        "soft_v_topk": args.soft_v_topk,
        "beta": SOFT_BETA,
        "gamma": GAMMA,
        "lam": args.lam,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys())
    )
    parser.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/sac")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument(
        "--pi-mode", default="hint", choices=["full", "answer", "hint", "rollout"]
    )
    parser.add_argument("--hint-generator-model", default=None)
    parser.add_argument("--hint-cache", default=None)
    parser.add_argument("--rollout-pi-root", default="data/pi/attempted_solution_8k")
    parser.add_argument("--rollout-pi-sample-idx", type=int, default=0)

    parser.add_argument(
        "--soft-v-estimator",
        default="topk",
        choices=["topk", "sarsa"],
        help="Soft-V estimator. 'sarsa' is not implemented yet",
    )
    parser.add_argument("--soft-v-topk", type=int, default=100)
    parser.add_argument("--lam", type=float, default=0.0)

    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-completion-length", type=int, default=8192)
    parser.add_argument("--num-generations", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--lr-scheduler-type",
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
        ],
    )
    parser.add_argument("--optim", default="adamw_bnb_8bit")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument(
        "--use-vllm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--force-resume", action="store_true")
    args = parser.parse_args()

    if args.soft_v_estimator == "sarsa":
        parser.error("--soft-v-estimator sarsa is not implemented yet")
    if args.soft_v_topk < 1:
        parser.error("--soft-v-topk must be >= 1")
    if not 0.0 <= args.lam <= 1.0:
        parser.error("--lam must be in [0, 1]")
    if args.rollout_pi_sample_idx < 0:
        parser.error("--rollout-pi-sample-idx must be >= 0")
    if args.pi_mode != "hint" and (
        args.hint_generator_model is not None or args.hint_cache is not None
    ):
        parser.error("--hint-generator-model and --hint-cache require --pi-mode hint")

    pi_slug = args.pi_mode
    if args.pi_mode == "hint":
        generator, _ = resolve_hint_source(
            args.model, args.dataset, args.hint_generator_model, args.hint_cache
        )
        if generator != args.model:
            pi_slug = f"hint_{hint_generator_run_slug(generator)}"

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(
        args.output_root,
        model_slug,
        sac_run_name(args.dataset, pi_slug, args.soft_v_estimator),
    )
    print(
        f"model: {model_slug}  dataset: {args.dataset}  pi: {args.pi_mode}  "
        f"soft-V: {args.soft_v_estimator}  Q-head: {Q_HEAD_ARCHITECTURE}  "
        f"->  output: {output_dir}"
    )

    train_dataset = build_sdft_dataset(
        args.pi_mode,
        dataset=args.dataset,
        max_samples=args.max_samples,
        model=args.model,
        max_prompt_length=args.max_prompt_length,
        rollout_pi_root=args.rollout_pi_root,
        rollout_pi_sample_idx=args.rollout_pi_sample_idx,
        hint_generator_model=args.hint_generator_model,
        hint_cache=args.hint_cache,
        include_reward_solution=True,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training examples remain after PI construction and filtering")
    print(f"Loaded {len(train_dataset)} examples")
    print(f"  sample prompt: {train_dataset[0]['prompt'][-1]['content'][:120]!r}")
    print(f"  sample PI: {train_dataset[0]['privileged_context'][:120]!r}")
    print(f"  sample reward solution: {train_dataset[0]['solution']!r}")

    training_args = SACConfig(
        output_dir=output_dir,
        soft_v_estimator=args.soft_v_estimator,
        soft_v_topk=args.soft_v_topk,
        lam=args.lam,
        teacher_model_kind="base",
        teacher_prompt_template=TEACHER_PROMPT_TEMPLATE,
        generate_from_teacher=False,
        distillation_is_clip=None,
        num_loss_tokens_to_skip=0,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        num_iterations=1,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=None,
        repetition_penalty=1.0,
        generation_kwargs=None,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        use_teacher_server=False,
        use_liger_kernel=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=0,
        optim=args.optim,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        steps_per_generation=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=True,
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    meta = build_run_meta(args, len(train_dataset))
    if args.resume_from_checkpoint:
        validate_resume(args.resume_from_checkpoint, meta, args.force_resume)
    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(output_dir, "run_meta.json")
        with open(meta_path, "w") as handle:
            json.dump(meta, handle, indent=2)
        print(f"Wrote run metadata -> {meta_path}")

    trainer = SACTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        reward_func=accuracy_reward,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved policy and Q head -> {final_dir}")


if __name__ == "__main__":
    main()
