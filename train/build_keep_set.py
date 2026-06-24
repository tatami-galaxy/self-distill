"""Compute a shared keep-set of training rows across all PI arms.

Filtering each arm by its own teacher-prompt length drops ~16% of `full` (its
longest, hardest problems) but ~0% of the others, confounding the PI comparison
with a difference in training-question difficulty. Instead we keep a row only if
its teacher prompt fits within `--max-prompt-length` for EVERY arm, and write the
surviving row indices to one JSON that every training run consumes via
`train.train_sdft --keep-indices`. So all arms train on identical questions.

All arms are built from the same `hint_gen.build_pi_datasets` subset (same seed),
so they are row-aligned; this script asserts that before intersecting. The
teacher prompt is reconstructed and tokenized exactly as the trainer does, with
the same model tokenizer and `enable_thinking`, so the budget matches training.

Example
-------
python -m train.build_keep_set --pi-root data/pi --max-prompt-length 8192
"""

import argparse
import json
import os

from datasets import load_from_disk
from transformers import AutoTokenizer

from train.train_sdft import to_sdft_columns, teacher_messages

TEACHER_TEMPLATE = "{prompt}\n\n{privileged_context}"  # SDFTConfig default


def is_dataset_dir(path: str) -> bool:
    """A saved HF dataset dir carries a state.json / dataset_info.json."""
    return os.path.isdir(path) and any(
        os.path.isfile(os.path.join(path, f)) for f in ("state.json", "dataset_info.json")
    )


def discover_arms(pi_root: str) -> list[str]:
    """PI arm dirs under pi_root (skips `hints` and other non-dataset dirs)."""
    return [
        name for name in sorted(os.listdir(pi_root))
        if is_dataset_dir(os.path.join(pi_root, name))
    ]


def teacher_lengths(ds, tokenizer, kw, num_proc) -> list[int]:
    """Tokenized teacher-prompt length per row (matches the trainer)."""
    def tlen(row):
        enc = tokenizer.apply_chat_template(
            teacher_messages(row["prompt"], row["privileged_context"], TEACHER_TEMPLATE),
            add_generation_prompt=True, tokenize=True, return_dict=True, **kw,
        )
        return {"tlen": len(enc["input_ids"])}
    return ds.map(tlen, num_proc=num_proc)["tlen"]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pi-root", default="data/pi")
    p.add_argument("--arms", nargs="*", default=None,
                   help="Arm dir names to intersect (default: all dataset dirs under --pi-root)")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--max-prompt-length", type=int, default=8192)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True,
                   help="Must match training (Qwen3-4B thinking on).")
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--output", default=None,
                   help="Default: <pi-root>/keep_<max_prompt_length>.json")
    args = p.parse_args()

    arms = args.arms or discover_arms(args.pi_root)
    if not arms:
        p.error(f"no PI dataset dirs found under {args.pi_root}")
    output = args.output or os.path.join(args.pi_root, f"keep_{args.max_prompt_length}.json")
    kw = {"enable_thinking": args.enable_thinking}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Intersecting {len(arms)} arms at <= {args.max_prompt_length} tok: {arms}")

    n_total = None
    ref_questions = None
    fits_all = None  # fits_all[i] True iff row i fits in every arm so far
    per_arm_over = {}
    for arm in arms:
        ds = load_from_disk(os.path.join(args.pi_root, arm))
        if n_total is None:
            n_total = len(ds)
            ref_questions = ds["question"]
            fits_all = [True] * n_total
        elif len(ds) != n_total or ds["question"] != ref_questions:
            raise ValueError(
                f"arm '{arm}' is not row-aligned with the others "
                f"(len {len(ds)} vs {n_total}); rebuild all arms with the same seed/--max-samples."
            )
        lens = teacher_lengths(to_sdft_columns(ds), tokenizer, kw, args.num_proc)
        over = sum(1 for n in lens if n > args.max_prompt_length)
        per_arm_over[arm] = over
        for i, n in enumerate(lens):
            if n > args.max_prompt_length:
                fits_all[i] = False
        print(f"  {arm:28s} over-length: {over}/{n_total}")

    keep = [i for i, ok in enumerate(fits_all) if ok]
    print(f"Shared keep-set: {len(keep)}/{n_total} rows ({n_total - len(keep)} dropped)")

    meta = {
        "max_prompt_length": args.max_prompt_length,
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "template": TEACHER_TEMPLATE,
        "arms": arms,
        "per_arm_over_length": per_arm_over,
        "n_total": n_total,
        "n_kept": len(keep),
        "indices": keep,
    }
    with open(output, "w") as f:
        json.dump(meta, f)
    print(f"Wrote keep-set -> {output}")


if __name__ == "__main__":
    main()
