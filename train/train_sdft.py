"""Self-Distillation Fine-Tuning (SDFT) on a privileged-information (PI) dataset.

Trains one student (default Qwen3-4B, thinking on) with TRL's experimental
`SDFTTrainer`. The student generates on-policy from `system + question`; a
frozen `base` teacher re-scores those tokens after also seeing the PI (injected
into the user turn as `"{question}\n\n{privileged_context}"`), and the teacher's
demonstration-conditioned distribution is distilled back into the student via
reverse KL over the student's top-k support.

Run once per PI variant (one of the dirs produced by
`hint_gen.build_pi_datasets`): full, answer, tail, hint_<model>_<think>,
alpha_<a>. The output dir defaults to `outputs/sdft/<model>/<dataset>/<variant>`.

Locked-in design (see memory):
  - reverse KL                 -> distillation_alpha = 1.0
  - vLLM top-k logit objective -> distillation_mode = "topk_logits", use_vllm
  - frozen self-teacher        -> teacher_model_kind = "base"
The student / teacher are one network (same tokenizer) differing only in the PI.

The system prompt and chat format are shared with eval (`eval.run_eval`) so the
trained student is prompted at eval time exactly as it was trained.

Generation uses a vLLM *server* by default: launch it on its own GPU first, then
point training at it. This frees the training GPU of vLLM's weights+KV cache, so
the teacher forward and a larger batch fit. The teacher (frozen `base`) is scored
locally; only the student's on-policy generation goes to the server.

Examples
--------
# 1. start the vLLM server on a dedicated GPU (e.g. GPU 7)
CUDA_VISIBLE_DEVICES=7 trl vllm-serve --model Qwen/Qwen3-4B --port 8000 \
    --gpu_memory_utilization 0.9 --max_model_len 16384

# 2. train an arm on another GPU, pointing at that server
CUDA_VISIBLE_DEVICES=4 python -m train.train_sdft --pi-data data/pi_deepmath/full \
    --keep-indices data/pi_deepmath/keep_8192.json --vllm-server-port 8000

# colocate (no server) or transformers (debug) still available:
python -m train.train_sdft --pi-data data/pi_numina/answer --vllm-mode colocate
python -m train.train_sdft --pi-data data/pi_deepmath/alpha_0.5 --no-use-vllm
"""

import argparse
import json
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.sdft import SDFTConfig, SDFTTrainer

from eval.run_eval import SYSTEM_PROMPT
from eval.utils import grade_answer


def build_prompt(question: str) -> list[dict]:
    """Conversational student prompt. SDFT applies the chat template itself
    (with enable_thinking), so we return raw messages, not a rendered string."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def to_sdft_columns(ds):
    """Map the PI dataset (question, answer, privileged_context) to the columns
    the trainer uses: `prompt` and `privileged_context` drive SDFT; `answer` is
    the gold final answer, carried through (SDFTConfig keeps unused columns) so
    RewardLoggingSDFTTrainer can grade the student's rollouts. It does not affect
    the gradient -- SDFT only distills the teacher."""
    return ds.map(
        lambda row: {
            "prompt": build_prompt(row["question"]),
            "answer": row["answer"],
            "privileged_context": row["privileged_context"],
        },
        remove_columns=ds.column_names,
    )


def teacher_messages(prompt: list[dict], privileged_context: str, template: str) -> list[dict]:
    """Reconstruct the teacher prompt exactly as SDFTTrainer does: the PI is
    folded into the last user turn via `template`, system turns are kept as-is.
    Mirrors `SDFTTrainer._compose_teacher_prompt` so length filtering matches
    what the trainer will actually tokenize."""
    system_messages = prompt[:-1]
    user_text = prompt[-1]["content"]
    teacher_text = template.format(prompt=user_text, privileged_context=privileged_context)
    return system_messages + [{"role": "user", "content": teacher_text}]


def filter_overlong(ds, tokenizer, max_prompt_length, template, chat_template_kwargs):
    """Drop rows whose teacher prompt exceeds `max_prompt_length` tokens.

    The trainer left-truncates over-length prompts, which for a long PI (e.g.
    `full`) would silently chop the system prompt / question and keep only the
    PI tail. Filtering instead keeps every retained example intact. The teacher
    prompt is the longer of the two (it carries the PI), so it bounds both."""
    def keep(row):
        msgs = teacher_messages(row["prompt"], row["privileged_context"], template)
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, **chat_template_kwargs,
        )
        return len(enc["input_ids"]) <= max_prompt_length

    n0 = len(ds)
    ds = ds.filter(keep)
    print(f"  teacher prompt > {max_prompt_length} tok: dropped {n0 - len(ds)}, kept {len(ds)}/{n0}")
    return ds


class RewardLoggingSDFTTrainer(SDFTTrainer):
    """SDFT trainer that also logs the student's answer accuracy as `reward/answer_accuracy`.

    SDFT has no reward -- the gradient comes only from distilling the teacher.
    But grading the student's on-policy rollouts against the gold `answer` each
    step is a key diagnostic: it tracks whether the student is actually getting
    better at the task, and whether PI-induced exploration collapse trades search
    (and accuracy) for shorter, teacher-imitating traces. This is logging only;
    it never enters the loss.
    """

    def _prepare_training_batch(self, inputs):
        batch = super()._prepare_training_batch(inputs)
        self._log_answer_accuracy(inputs, batch)
        return batch

    def _log_answer_accuracy(self, inputs, batch):
        if not inputs or inputs[0].get("answer") is None:
            return
        mode = "train" if self.model.training else "eval"
        # _get_completion_ids_list trims each rollout to its real length; decoded
        # text keeps <think>...</think> and the \boxed{} answer (only specials are
        # stripped). inputs is already expanded by num_generations, so it aligns
        # 1:1 with the completions.
        completions = self.processing_class.batch_decode(
            self._get_completion_ids_list(batch), skip_special_tokens=True
        )
        correct = []
        for ex, text in zip(inputs, completions, strict=True):
            correct.append(float(grade_answer(text, ex["answer"])))
        acc = self.accelerator.gather(torch.tensor(correct, device=self.accelerator.device))
        self._metrics[mode]["reward/answer_accuracy"].append(acc.float().mean().item())


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pi-data", required=True,
                   help="Path to a built PI dataset dir, e.g. data/pi_deepmath/full or data/pi_numina/answer")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--output-root", default="outputs/sdft")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>/<variant>")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Subset the training set")
    p.add_argument("--keep-indices", default=None,
                   help="JSON from train.build_keep_set: a shared keep-set of row indices so "
                        "every PI arm trains on identical questions. When set, rows are selected "
                        "from it and per-arm length filtering is skipped.")
    p.add_argument("--max-prompt-length", type=int, default=8192,
                   help="Teacher prompt holds the PI and is left-truncated; must fit "
                        "system + question + privileged_context for the longest PI.")
    p.add_argument("--max-completion-length", type=int, default=8192)
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--optim", default="adamw_bnb_8bit",
                   help="Optimizer. Default 8-bit Adam ; use adamw_torch for fp32.")
    p.add_argument("--max-steps", type=int, default=500,
                   help="Total optimizer steps to train (overrides epochs).")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--num-generations", type=int, default=1,
                   help="Rollouts per prompt; effective batch must be divisible by this.")
    # distillation objective
    p.add_argument("--distillation-alpha", type=float, default=1.0,
                   help="1.0 = reverse KL, 0.0 = forward KL, in-between = JSD.")
    p.add_argument("--distillation-topk", type=int, default=100)
    p.add_argument("--teacher-model-kind", default="base", choices=["base", "ema", "live"])
    # generation backend (vLLM server by default; launch it separately --
    #   CUDA_VISIBLE_DEVICES=<gpu> trl vllm-serve --model <model> --port <port>)
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True,
                   help="Use vLLM for generation. --no-use-vllm falls back to "
                        "transformers generation for debugging.")
    p.add_argument("--vllm-mode", default="colocate", choices=["server", "colocate"],
                   help="server: connect to a separate `trl vllm-serve` process on its own GPU "
                        "(frees training-GPU memory, allows higher batch). colocate: run vLLM "
                        "in-process, sharing the training GPU.")
    p.add_argument("--vllm-server-host", default="0.0.0.0")
    p.add_argument("--vllm-server-port", type=int, default=8000)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2,
                   help="Colocate only; in server mode set --gpu_memory_utilization on `trl vllm-serve`.")
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Default output dir disambiguates by model and dataset, since we sweep PIs
    # across multiple students and datasets: <output_root>/<model>/<dataset>/<variant>
    # e.g. outputs/sdft/Qwen3-4B/pi_numina/answer
    variant = os.path.basename(args.pi_data.rstrip("/"))
    dataset_tag = os.path.basename(os.path.dirname(args.pi_data.rstrip("/")))  # e.g. pi_numina
    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(args.output_root, model_slug, dataset_tag, variant)
    print(f"model: {model_slug}  dataset: {dataset_tag}  PI: {variant}  ->  output: {output_dir}")

    ds = load_from_disk(args.pi_data)
    if args.keep_indices:
        with open(args.keep_indices) as f:
            keep = json.load(f)
        if len(ds) != keep["n_total"]:
            raise ValueError(
                f"keep-set was built on {keep['n_total']} rows but this arm has {len(ds)}; "
                "the arms are misaligned -- rebuild them / the keep-set with the same seed."
            )
        if keep["max_prompt_length"] != args.max_prompt_length:
            raise ValueError(
                f"keep-set built at max_prompt_length={keep['max_prompt_length']} "
                f"but tried to train at {args.max_prompt_length}"
            )
        ds = ds.select(keep["indices"])
        print(f"  shared keep-set: {len(ds)}/{keep['n_total']} rows from {args.keep_indices}")
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
        # bookkeeping
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    # Without a shared keep-set, fall back to per-arm length filtering (debug /
    # single-arm runs). Drops rows whose teacher prompt would be left-truncated.
    # Done before the model load so it fails fast / is cheap to iterate.
    if not args.keep_indices:
        train_dataset = filter_overlong(
            train_dataset, tokenizer,
            training_args.max_prompt_length,
            training_args.teacher_prompt_template,
            training_args.chat_template_kwargs,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16
    )
    model.config.use_cache = False  # gradient checkpointing below

    trainer = RewardLoggingSDFTTrainer(
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
