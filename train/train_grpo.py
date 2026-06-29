"""GRPO baseline: on-policy RL with a verifiable correctness reward.

This is the RL-from-scratch comparison point for the SDFT PI study. Where SDFT
distills a privileged-information teacher, GRPO instead samples a group of
rollouts per prompt and reinforces the ones that get the right answer (group
-relative advantage). It uses *no* privileged context -- only the student's own
exploration and a binary correctness reward.

To keep the comparison fair it trains on the SAME questions as the SDFT arms:
point `--data` at any built PI arm dir (the PI is ignored; only `question` and
`answer` are used) and pass the SAME `--keep-indices` shared keep-set. The system
prompt and chat format are shared with eval (`eval.run_eval`) and with SDFT.

Reward: +1 if the student's `\boxed{}` answer matches the gold `answer`, else 0
(`eval.utils.grade_answer`, same grader SDFT logs). GRPO logs mean reward, which
is the on-policy answer accuracy -- directly comparable to SDFT's
`reward/answer_accuracy`.

Output dir defaults to `outputs/grpo/<model>/<dataset>` (one baseline per
model+dataset; the PI arm is irrelevant since GRPO ignores it).

Generation uses a vLLM server by default (launch it on its own GPU first), same
as train_sdft; colocate and transformers fallbacks are available.

Examples
--------
# server on a dedicated GPU, then GRPO on another
CUDA_VISIBLE_DEVICES=7 trl vllm-serve --model Qwen/Qwen3-4B --port 8000 \
    --gpu_memory_utilization 0.9 --max_model_len 16384
CUDA_VISIBLE_DEVICES=4 python -m train.train_grpo --data data/pi_numina/answer \
    --keep-indices data/pi_numina/keep_8192.json --vllm-server-port 8000

python -m train.train_grpo --data data/pi_deepmath/answer --vllm-mode colocate
"""

import argparse
import json
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import GRPOConfig, GRPOTrainer

from eval.run_eval import SYSTEM_PROMPT
from eval.utils import extract_boxed_answer, grade_answer


def build_prompt(question: str) -> list[dict]:
    """Conversational student prompt -- identical to SDFT and eval. GRPO applies
    the chat template itself (with enable_thinking), so return raw messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def to_grpo_columns(ds):
    """Map a built dataset (question, answer, privileged_context) to what GRPO
    consumes: `prompt` (drives generation) and `answer` (passed to the reward).
    `privileged_context` is dropped -- GRPO is a no-PI baseline."""
    return ds.map(
        lambda row: {"prompt": build_prompt(row["question"]), "answer": row["answer"]},
        remove_columns=ds.column_names,
    )


def correctness_reward(completions, answer, **kwargs):
    """+1 if the rollout's boxed answer matches the gold answer, else 0.

    `completions` is a list of conversational completions ([{role, content}, ...]);
    `answer` is the gold column, expanded by num_generations to align 1:1."""
    rewards = []
    for completion, gold in zip(completions, answer):
        text = completion[-1]["content"] if isinstance(completion, list) else completion
        rewards.append(1.0 if grade_answer(extract_boxed_answer(text), gold) else 0.0)
    return rewards


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True,
                   help="A built dataset dir (any PI arm; PI ignored), e.g. data/pi_numina/answer")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--output-root", default="outputs/grpo")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Subset the training set")
    p.add_argument("--keep-indices", default=None,
                   help="Shared keep-set JSON (use the SAME one as the SDFT arms so GRPO "
                        "trains on identical questions).")
    p.add_argument("--max-completion-length", type=int, default=8192)
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--optim", default="adamw_bnb_8bit",
                   help="Default 8-bit Adam keeps optimizer states small; use adamw_torch for fp32.")
    p.add_argument("--max-steps", type=int, default=500,
                   help="Total optimizer steps to train (overrides epochs).")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--num-generations", type=int, default=8,
                   help="Group size for group-relative advantage; effective batch must be divisible by this.")
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL coefficient to the reference model. 0 (default) = no ref model "
                        "loaded (lighter); >0 adds a frozen ref copy + KL penalty.")
    # generation backend (mirrors train_sdft)
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True,
                   help="Use vLLM for generation. --no-use-vllm falls back to transformers.")
    p.add_argument("--vllm-mode", default="colocate", choices=["server", "colocate"])
    p.add_argument("--vllm-server-host", default="0.0.0.0")
    p.add_argument("--vllm-server-port", type=int, default=8000)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2,
                   help="Colocate only; in server mode set it on `trl vllm-serve`.")
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # One baseline per model+dataset (GRPO ignores the PI arm), mirroring the
    # SDFT layout: <output_root>/<model>/<dataset>.
    dataset_tag = os.path.basename(os.path.dirname(args.data.rstrip("/")))  # e.g. pi_numina
    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(args.output_root, model_slug, dataset_tag)
    print(f"model: {model_slug}  dataset: {dataset_tag}  ->  output: {output_dir}")

    ds = load_from_disk(args.data)
    if args.keep_indices:
        with open(args.keep_indices) as f:
            keep = json.load(f)
        if len(ds) != keep["n_total"]:
            raise ValueError(
                f"keep-set was built on {keep['n_total']} rows but this dataset has {len(ds)}; "
                "rebuild them / the keep-set with the same seed."
            )
        ds = ds.select(keep["indices"])
        print(f"  shared keep-set: {len(ds)}/{keep['n_total']} rows from {args.keep_indices}")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    train_dataset = to_grpo_columns(ds)
    print(f"Loaded {len(train_dataset)} examples")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = GRPOConfig(
        output_dir=output_dir,
        # sampling / objective
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        chat_template_kwargs={"enable_thinking": True},
        # generation backend
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        # optimization
        learning_rate=args.learning_rate,
        optim=args.optim,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        # keep `answer` so the reward function receives it
        remove_unused_columns=False,
        # bookkeeping
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16
    )
    model.config.use_cache = False  # gradient checkpointing

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=correctness_reward,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved student -> {output_dir}")


if __name__ == "__main__":
    main()
