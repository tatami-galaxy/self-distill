r"""
Train the SFT baseline via TRL's `SFTTrainer`.

The traces are R1's rather than the student's, so this is off-policy distillation from an
external teacher: the off-policy counterpart to train_gold.py's OPD. Two other corpora would
slot into `build_sft_dataset` unchanged if they are ever wanted -- teacher-sampled completions
(the exactly-matched off-policy control for GOLD) and rejection-sampled self-completions (the
counterpart to the SDFT/OPSD self-distillation arms). Both cost a generation pass first; the
R1 traces are already on disk. If a second corpus is added, `sft` should join run_eval.py's
VARIANT_REQUIRED so the two never share a results folder.

DATA (`build_sft_dataset`), in TRL's prompt-completion conversational format:
  prompt     -- format_prompt_math(question), IDENTICAL to eval / train_grpo / train_ppo, so
                the tokens the student is trained to continue are the tokens it is asked to
                continue at eval.
  completion -- [{"role": "assistant", "content": "<think>\n" + trace}]
TRL infers `completion_only_loss=True` from exactly those two columns (sft_trainer.py's
`"prompt" in dataset_sample and "completion" in dataset_sample`), so the prompt is masked to
-100 and only the assistant turn trains. It is inferred rather than passed so that a future
change to the column names fails loudly instead of silently training on the prompt too.

THE THINK BLOCK. DeepMath's r1_solution_1 follows R1's convention: exactly one `</think>`, no
opening tag (measured: 200/200 rows). Qwen3's chat template SPLITS assistant content on
`</think>` -- content that contains one is emitted verbatim behind an opener the template
supplies itself, so for these traces the explicit `<think>\n` prepend in
`format_think_completion` is a verified NO-OP (the two renderings are byte-identical). It is
kept anyway because it is only a no-op for templates that split that way: a template that does
not would render the reasoning outside any think block, and this arm is meant to be readable by
whichever model it is pointed at. The student's own rollouts open with `<think>\n` (its
generation prompt ends at `<|im_start|>assistant\n`, so the model emits the opener itself),
which is the format this reproduces either way.

LENGTH: CAP THE COMPLETION, AND DROP RATHER THAN TRUNCATE. The cap counts the RENDERED assistant turn, so it includes the `<|im_end|>` terminator: about
two tokens stricter than the RL arms' pure-generation budget.

Resume (--resume-from-checkpoint) restores weights/optimizer/scheduler/RNG and skips seen data;
pass the same hyperparameters (verified against run_meta.json). --max-steps is the TOTAL budget
and is free to raise, but the LEARNING RATE comes from the checkpoint rather than the command
line, so a changed --learning-rate is refused rather than silently ignored. Full rules in
utils.validate_resume. Both length caps are recorded too, because changing either changes which
rows survive the filter and therefore what the seeded data-skip lands on.

# single GPU, no vLLM
CUDA_VISIBLE_DEVICES=0 uv run python -m train.train_sft \
    --model Qwen/Qwen3-1.7B --dataset deepmath

# smoke
CUDA_VISIBLE_DEVICES=0 uv run python -m train.train_sft \
    --model Qwen/Qwen3-1.7B --max-samples 64 --max-steps 2 --save-steps 2
"""

import argparse
import json
import os

from trl import SFTConfig, SFTTrainer

from utils import (
    DATASET_REGISTRY_TRAIN,
    format_prompt_math,
    load_train_dataset,
    validate_resume,
)


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Stamped into run_meta.json. The completion format is what the checkpoint was taught to
# produce, so a change to it means a resumed run is continuing a differently-shaped target --
# recorded here rather than left implicit in the code. Not a `strict_key`: no checkpoint
# predates it, so absence carries no information yet.
COMPLETION_FORMAT_VERSION = "think_wrapped_v1"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def format_think_completion(solution: str) -> str | None:
    """The reference trace as an assistant turn the chat template will render VERBATIM,
    or None if the trace cannot be made into one (the caller drops those rows).

    R1-convention traces -- exactly one `</think>`, no opener -- get `<think>\\n` prepended.
    Under Qwen3's template that is a no-op (it splits on `</think>` and supplies the opener);
    it makes the target well-formed under templates that do not. See the module docstring.

    Rejected, all for the same reason (the rendered target would teach a broken format):
      * empty                        -- nothing to imitate.
      * no `</think>`                -- a non-reasoning worked answer; prepending an opener
                                        would train a think block that is never closed.
      * more than one `</think>`     -- malformed; which one ends the reasoning is a guess.
      * `<think>` not at the start   -- same, from the other end.
    """
    text = (solution or "").strip()
    if not text:
        return None
    if text.count(THINK_CLOSE) != 1:
        return None
    # `</think>` does NOT contain `<think>` (its `<` is followed by `/`), so this counts
    # opening tags only and needs no correction for the closing one.
    n_open = text.count(THINK_OPEN)
    if n_open == 0:
        return f"{THINK_OPEN}\n{text}"
    if n_open == 1 and text.startswith(THINK_OPEN):
        return text
    return None


def to_sft_example(question: str, solution: str) -> dict | None:
    """One row in TRL's prompt-completion conversational format, or None to drop it."""
    completion = format_think_completion(solution)
    if completion is None:
        return None
    return {
        "prompt": format_prompt_math(question),
        "completion": [{"role": "assistant", "content": completion}],
    }


def build_sft_dataset(
    dataset: str = "deepmath",
    model: str | None = None,
    max_samples: int | None = None,
    max_completion_length: int | None = None,
    max_prompt_length: int | None = None,
    num_proc: int = 8,
):
    """Columns `prompt` + `completion` (see module docstring); everything else is dropped.

    `require_solution=True` filters solution-less rows BEFORE the `max_samples` prefix, so a
    subset request yields that many rows that actually carry a trace (a no-op for DeepMath,
    essential for DeepScaleR, where ~82% have none).

    When `model` is given, rows are dropped whose rendered completion exceeds
    `max_completion_length` or whose rendered prompt exceeds `max_prompt_length`, so the trainer
    never truncates one. The two halves are capped SEPARATELY (rather than by their sum) because
    the completion's budget is the one that has to match the RL arms' --max-completion-length;
    see the module docstring. The measurement uses TRL's own `apply_chat_template` and tokenizes
    each half with `add_special_tokens=False` -- exactly what SFTTrainer does downstream, so the
    filter and the trainer agree on what "too long" means rather than approximately agreeing.
    """
    ds = load_train_dataset(dataset, max_samples=max_samples, require_solution=True)

    n_raw = len(ds)
    ds = ds.filter(lambda row: format_think_completion(row["solution"]) is not None, num_proc=4)
    if len(ds) < n_raw:
        print(f"  dropped {n_raw - len(ds)}/{n_raw} rows whose trace is not a single "
              f"well-formed reasoning block (see format_think_completion)")

    ds = ds.map(
        lambda row: to_sft_example(row["question"], row["solution"]),
        remove_columns=ds.column_names,
    )

    if model is not None and (max_completion_length is not None or max_prompt_length is not None):
        from transformers import AutoTokenizer
        from trl.data_utils import apply_chat_template

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        completion_cap = max_completion_length or float("inf")
        prompt_cap = max_prompt_length or float("inf")

        def _measure(batch):
            rendered = [
                apply_chat_template({"prompt": prompt, "completion": completion}, tok)
                for prompt, completion in zip(batch["prompt"], batch["completion"])
            ]
            prompt_ids = tok([r["prompt"] for r in rendered], add_special_tokens=False)["input_ids"]
            completion_ids = tok(
                [r["completion"] for r in rendered], add_special_tokens=False
            )["input_ids"]
            return {
                "n_prompt_tokens": [len(p) for p in prompt_ids],
                "n_completion_tokens": [len(c) for c in completion_ids],
            }

        # Measure once into columns, so the two caps can be REPORTED separately: a non-zero
        # prompt-drop count means the prompt guard is doing real work, which on this data it
        # should not be, and is worth seeing rather than having folded into one total.
        ds = ds.map(_measure, batched=True, batch_size=256, num_proc=num_proc)
        n_before = len(ds)
        over_completion = sum(n > completion_cap for n in ds["n_completion_tokens"])
        over_prompt = sum(n > prompt_cap for n in ds["n_prompt_tokens"])
        ds = ds.filter(
            lambda batch: [
                c <= completion_cap and p <= prompt_cap
                for c, p in zip(batch["n_completion_tokens"], batch["n_prompt_tokens"])
            ],
            batched=True,
            batch_size=256,
            num_proc=num_proc,
        )
        ds = ds.remove_columns(["n_prompt_tokens", "n_completion_tokens"])
        if len(ds) < n_before:
            print(f"  kept {len(ds)}/{n_before} rows: dropped {over_completion} over "
                  f"--max-completion-length {max_completion_length}, {over_prompt} over "
                  f"--max-prompt-length {max_prompt_length} (counts overlap if a row exceeds "
                  f"both). The dropped traces are the longest -- see the module docstring.")

    return ds


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def build_run_meta(args, num_train_examples: int) -> dict:
    """Provenance + resume-critical config for run_meta.json. The second block must match on
    resume for the seeded data-skip to land on the same examples."""
    return {
        "method": "sft",
        "model": args.model,
        "dataset": args.dataset,
        # Which traces were imitated. A bare "sft" would not distinguish this arm from one
        # trained on teacher-sampled or rejection-sampled completions later.
        "source": "reference_trace",
        "completion_format": COMPLETION_FORMAT_VERSION,
        "max_samples": args.max_samples,
        "num_train_examples": num_train_examples,
        # resume-critical: both caps decide which rows survive build_sft_dataset's filter, so
        # changing either changes the dataset the seeded skip is indexing into.
        "max_completion_length": args.max_completion_length,
        "max_prompt_length": args.max_prompt_length,
        "completion_only_loss": True,
        # resume-critical: on resume the LR comes from the CHECKPOINT, not the CLI (see
        # validate_resume), so recording it turns a silently-ignored change into an error.
        "learning_rate": args.learning_rate,
        # resume-critical: dataset order (seed, length) + batch chunking.
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()),
                   help="Training dataset (see utils.DATASET_REGISTRY_TRAIN). The "
                        "solution-bearing subset is used -- the trace IS the target here.")
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/sft")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>")
    p.add_argument("--max-samples", type=int, default=None, help="Subset the training set")
    # data
    p.add_argument("--max-completion-length", type=int, default=8192,
                   help="Token cap on the completion alone.Rows above "
                        "it are DROPPED by the dataset builder, not truncated. "
                        "Keeps ~85% of DeepMath.")
    p.add_argument("--max-prompt-length", type=int, default=1024,
                   help="Token cap on the PROMPT half; rows above it are dropped too. Bounds "
                        "the total so SFTConfig.max_length (their sum) is a backstop that never "
                        "fires. Drops nothing on DeepMath (prompts max ~350 tokens).")
    p.add_argument("--dataset-num-proc", type=int, default=8,
                   help="Workers for the length filter and TRL's tokenization map.")
    # optimization (mirrors train_sdft.py / train_gold.py, the other imitation arms)
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="1e-5 to match train_sdft.py / train_gold.py, NOT the RL arms' 1e-6.")
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="adamw_bnb_8bit",
                   help="Optimizer. Default 8-bit Adam; use adamw_torch_fused otherwise. "
                        "Nothing here uses vLLM, so train_ppo.py's server-mode 8-bit hazard "
                        "does not apply.")
    p.add_argument("--max-steps", type=int, default=400, help="Total optimizer steps")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    # resume
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume from a checkpoint dir (path ending in 'checkpoint-<step>'). "
                        "Restores weights/optimizer/scheduler/RNG and skips already-seen "
                        "examples. Pass the SAME --model, --dataset, --max-samples, length "
                        "caps, --seed and batch config as the original run (verified against "
                        "its run_meta.json). --max-steps is the TOTAL budget.")
    p.add_argument("--force-resume", action="store_true",
                   help="Downgrade a run_meta.json hyperparameter mismatch from an error to a "
                        "warning (use only if you understand the data-skip consequences).")
    args = p.parse_args()

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(args.output_root, model_slug, args.dataset)
    print(f"model: {model_slug}  dataset: {args.dataset}  ->  output: {output_dir}")

    train_dataset = build_sft_dataset(
        args.dataset,
        model=args.model,
        max_samples=args.max_samples,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        num_proc=args.dataset_num_proc,
    )
    print(f"Loaded {len(train_dataset)} examples")
    print(f"  sample prompt: {train_dataset[0]['prompt'][-1]['content'][:120]!r}")
    print(f"  sample completion head: "
          f"{train_dataset[0]['completion'][0]['content'][:80]!r}")
    print(f"  sample completion tail: "
          f"{train_dataset[0]['completion'][0]['content'][-80:]!r}")

    training_args = SFTConfig(
        output_dir=output_dir,
        # data
        # Backstop only. build_sft_dataset has already dropped every row whose prompt exceeds
        # --max-prompt-length or whose completion exceeds --max-completion-length, so no
        # surviving sequence can reach this sum and TRL's keep_start truncation never fires.
        max_length=args.max_prompt_length + args.max_completion_length,
        # Packing would concatenate examples into fixed blocks, which both crosses example
        # boundaries and is unsupported alongside the prompt-masking this arm depends on.
        packing=False,
        dataset_num_proc=args.dataset_num_proc,
        # completion_only_loss is left unset: TRL infers True from the prompt+completion
        # columns, so a future column rename fails loudly instead of quietly training the
        # prompt tokens too.
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
        # SFTTrainer instantiates the model from the model string; request bf16 explicitly
        # (mirrors train_grpo.py -- dtype defaults to fp32 otherwise).
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        # bookkeeping
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    meta = build_run_meta(args, len(train_dataset))

    # Before resuming, verify the run this checkpoint came from used the same config (its
    # run_meta.json sits beside the checkpoint), so the seeded data-skip lands on the examples
    # that were actually left untrained.
    if args.resume_from_checkpoint:
        validate_resume(args.resume_from_checkpoint, meta, args.force_resume)

    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(output_dir, "run_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote run metadata -> {meta_path}")

    trainer = SFTTrainer(
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
