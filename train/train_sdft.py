"""Self-Distillation Fine-Tuning (SDFT) on a privileged-information (PI) dataset.

Trains one student (default Qwen3-4B, thinking on) with TRL's experimental
`SDFTTrainer`. The student generates on-policy from `system + question`; a
frozen `base` teacher re-scores those tokens after also seeing the PI (injected
into the user turn as `"{question}\n\n{privileged_context}"`), and the teacher's
demonstration-conditioned distribution is distilled back into the student via
reverse KL over the student's top-k support.

Run once per PI variant (one of the dirs produced by
`hint_gen.build_pi_datasets`): full, answer, tail, hint_<model>_<think>,
alpha_<a>. The output dir defaults to `outputs/sdft/<variant>`.

Locked-in design (see memory):
  - reverse KL                 -> distillation_alpha = 1.0
  - vLLM top-k logit objective -> distillation_mode = "topk_logits", use_vllm
  - frozen self-teacher        -> teacher_model_kind = "base"
The student / teacher are one network (same tokenizer) differing only in the PI.

The system prompt and chat format are shared with eval (`eval.run_eval`) so the
trained student is prompted at eval time exactly as it was trained.

Examples
--------
python -m train.train_sdft --pi-data data/pi/full
python -m train.train_sdft --pi-data data/pi/hint_Qwen3-4B_nothink --max-samples 256
python -m train.train_sdft --pi-data data/pi/alpha_0.5 --no-use-vllm   # transformers gen (debug)
"""

import argparse
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.sdft import SDFTConfig, SDFTTrainer

from eval.run_eval import SYSTEM_PROMPT 


def build_prompt(question: str) -> list[dict]:
    """Conversational student prompt. SDFT applies the chat template itself
    (with enable_thinking), so we return raw messages, not a rendered string."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def to_sdft_columns(ds):
    """Map the PI dataset (question, answer, privileged_context) to the columns
    SDFTTrainer consumes (prompt, privileged_context). `answer` is dropped: SDFT
    has no reward, it only distills the teacher distribution."""
    return ds.map(
        lambda row: {
            "prompt": build_prompt(row["question"]),
            "privileged_context": row["privileged_context"],
        },
        remove_columns=ds.column_names,
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pi-data", required=True,
                   help="Path to a built PI dataset dir, e.g. data/pi/full")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--output-root", default="outputs/sdft")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<basename of --pi-data>")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Subset the training set (quick smoke tests)")
    # lengths -- the tiny SDFT defaults (512/256) truncate math reasoning + PI
    p.add_argument("--max-prompt-length", type=int, default=4096,
                   help="Teacher prompt holds the PI and is left-truncated; must fit "
                        "system + question + privileged_context for the longest PI.")
    p.add_argument("--max-completion-length", type=int, default=4096)
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8,
                   help="Rollouts per prompt; effective batch must be divisible by this.")
    p.add_argument("--temperature", type=float, default=1.0)
    # distillation objective (locked-in defaults)
    p.add_argument("--distillation-alpha", type=float, default=1.0,
                   help="1.0 = reverse KL (locked in). 0.0 = forward KL, in-between = JSD.")
    p.add_argument("--distillation-topk", type=int, default=100)
    p.add_argument("--teacher-model-kind", default="base", choices=["base", "ema", "live"])
    # generation backend
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True,
                   help="vLLM colocate generation (locked in). --no-use-vllm falls back "
                        "to transformers generation for debugging.")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--report-to", default="none")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    variant = os.path.basename(args.pi_data.rstrip("/"))
    output_dir = args.output_dir or os.path.join(args.output_root, variant)
    print(f"PI variant: {variant}  ->  output: {output_dir}")

    ds = load_from_disk(args.pi_data)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    train_dataset = to_sdft_columns(ds)
    print(f"Loaded {len(train_dataset)} examples")
    print(f"  sample PI: {ds[0]['privileged_context'][:160]!r}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16
    )
    model.config.use_cache = False  # gradient checkpointing below

    training_args = SDFTConfig(
        output_dir=output_dir,
        # objective
        distillation_mode="topk_logits",
        distillation_alpha=args.distillation_alpha,
        distillation_topk=args.distillation_topk,
        teacher_model_kind=args.teacher_model_kind,
        # student keeps the plain prompt for on-policy generation (default);
        # the teacher sees prompt + PI via the default teacher_prompt_template.
        chat_template_kwargs={"enable_thinking": True},
        # lengths / sampling
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        # generation backend
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        # optimization
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        # bookkeeping
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    trainer = SDFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved student -> {output_dir}")


if __name__ == "__main__":
    main()
