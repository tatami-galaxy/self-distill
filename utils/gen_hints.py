"""
Generate hint caches for SDFT's `--pi-mode hint`.

The model reads each DeepMath problem together with its reference solution and
compresses it into a SMALL set of useful concepts and intermediate results --
NOT the final answer. This is summarization of a solution the model can see, so
the resulting context carries no answer and sits between the `full` (whole
solution) and `answer` (boxed value) privileged contexts. The original
"hint-self" arm uses the student itself as `--model`; a separately trained hint
generator can be used by passing its model/checkpoint here instead.

Every cache is stamped with the exact `--model` string in `gen_model`.
train_sdft.py validates that against `--hint-generator-model`, so use the same
string in both commands.

Anti-leak: DeepMath solutions end in \\boxed{answer}; a hint that restates the
answer would collapse this PI into the `answer` PI and confound the experiment.
Rows whose hint contains \\boxed or the gold final answer are dropped.

Output: an on-disk HF dataset with columns question, final_answer, hint,
gen_model, dataset. The default path is data/pi/hint/<dataset>/<model-slug>/;
use `--output-dir` for trained checkpoints so different runs cannot share the
same generic checkpoint slug.

# Original self-hint cache
CUDA_VISIBLE_DEVICES=0 uv run python -m utils.gen_hints \
    --model Qwen/Qwen3-1.7B --dataset deepmath --max-samples 20000

# Cache from a trained hint generator
CUDA_VISIBLE_DEVICES=0 uv run python -m utils.gen_hints \
    --model /mnt/data/ujan/self-distill/outputs/hint_gen/Qwen3-1.7B/deepmath_a1_g1/checkpoint-100 \
    --dataset deepmath --max-samples 20000 \
    --output-dir data/pi/hint/deepmath/Qwen3-1.7B-a1g1-checkpoint-100
"""

import argparse
import os
import re

from datasets import Dataset, load_from_disk
from vllm import LLM, SamplingParams

from utils import DATASET_REGISTRY_TRAIN, hint_path, load_train_dataset

HINT_SYSTEM = (
    "You are given a math problem and a full worked solution. Extract a "
    "SHORT list of the key concepts that are most useful for solving the problem."
)

HINT_USER = (
    "Problem:\n{problem}\n\n"
    "Worked solution (for your reference only):\n{solution}\n\n"
    "In a few lines mention key ideas and concepts "
    "that might be relevant for solving the problem. "
    "Output only the key ideas. "
    "Be as brief as possible. "
    "Do NOT state or compute the final answer under any circumstances. "
)


def build_messages(problem: str, solution: str) -> list[dict]:
    return [
        {"role": "system", "content": HINT_SYSTEM},
        {"role": "user", "content": HINT_USER.format(problem=problem, solution=solution)},
    ]


def leaks_answer(hint: str, gold: str) -> bool:
    """True if the hint reveals the final answer.

    `\\boxed` is always a leak. For the gold value we require a word-boundary
    match and skip golds shorter than 2 chars: single-digit answers (0-9) appear
    incidentally in almost every hint, so substring-matching them would drop
    nearly everything -- there we rely on the no-answer instruction instead.
    Dropping is the safe error (lost data, no confound); missing a leak is not,
    so the check errs toward dropping for distinctive answers.
    """
    if "\\boxed" in hint:
        return True
    g = str(gold).strip()
    return len(g) >= 2 and re.search(
        rf"(?<![\w.]){re.escape(g)}(?![\w.])", hint
    ) is not None


def strip_thinking(text: str) -> str | None:
    """Drop a model's ``<think>...</think>`` reasoning trace, keeping what follows.

    Some models (e.g. OLMo) think by default and ignore ``enable_thinking=False``,
    emitting the trace inline; the actual hint is the text after ``</think>``. We
    strip it so the trace (which we don't want, and which leaks answers) never
    reaches the cache or the leak check. Returns ``None`` when ``<think>`` was
    opened but never closed -- the trace was truncated by ``--max-tokens`` and
    there is no usable hint, which also signals the budget is too small for a
    thinking model.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        return None
    return text.strip()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B",
                   help="Hint generator model or local checkpoint. For self-hints this is "
                        "the student model; otherwise pass it to train_sdft.py as well.")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()),
                   help="Source dataset (see utils.DATASET_REGISTRY_TRAIN); only its "
                        "solution-bearing rows are used. MUST match the training --dataset.")
    p.add_argument("--max-samples", type=int, default=20000,
                   help="How many rows to generate hints for. Generate "
                        "for the largest N you will train on; training takes a prefix.")
    p.add_argument("--output-root", default="data/pi/hint")
    p.add_argument("--output-dir", default=None,
                   help="Exact cache directory; overrides --output-root path construction.")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if a large-enough cache already exists.")
    # generation
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Max hint length. Hints are short lists; 512 is ample.")
    p.add_argument("--seed", type=int, default=42)
    # vLLM
    p.add_argument("--max-model-len", type=int, default=32768,
                   help="vLLM context. Rows whose (problem+solution) prompt plus "
                        "--max-tokens exceeds this are skipped.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    args = p.parse_args()

    out_dir = args.output_dir or hint_path(args.model, args.dataset, args.output_root)

    # Reuse guard: skip if a compatible cache (same model, >= requested rows) exists.
    if not args.force and os.path.isdir(out_dir):
        cached = load_from_disk(out_dir)
        same_model = set(cached.unique("gen_model")) == {args.model}
        if same_model and len(cached) >= args.max_samples:
            print(f"Reusing {len(cached)} cached hints at {out_dir} "
                  f"(>= {args.max_samples}, model matches). Use --force to regenerate.")
            return

    ds = load_train_dataset(args.dataset, max_samples=args.max_samples, require_solution=True)
    print(f"Loaded {len(ds)} {args.dataset} rows (with solutions) for hint generation "
          f"with {args.model}")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts and drop any whose input won't leave room for the hint.
    budget = args.max_model_len - args.max_tokens
    rows, conversations = [], []
    n_too_long = 0
    for row in ds:
        messages = build_messages(row["question"], row["solution"])
        # return_dict + input_ids gives the real token count (a bare tokenize=True
        # returns a BatchEncoding whose len() is the field count). enable_thinking
        # is passed straight through to the Qwen chat template, matching llm.chat below.
        enc = tokenizer.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=True,
            return_dict=True, enable_thinking=False,
        )
        n_prompt = len(enc["input_ids"][0])
        if n_prompt > budget:
            n_too_long += 1
            continue
        rows.append(row)
        conversations.append(messages)
    if n_too_long:
        print(f"  skipped {n_too_long} rows whose prompt exceeded the context budget")

    sampling = SamplingParams(
        max_tokens=args.max_tokens, seed=args.seed,
    )
    outputs = llm.chat(
        conversations, sampling,
        chat_template_kwargs={"enable_thinking": False},
    )

    kept, n_leaked, n_empty, n_unclosed = [], 0, 0, 0
    for row, out in zip(rows, outputs, strict=True):
        # Strip any inline reasoning trace first, so only the concise hint is
        # cached and leak-checked (thinking models ignore enable_thinking=False).
        hint = strip_thinking(out.outputs[0].text)
        if hint is None:
            n_unclosed += 1
            continue
        if not hint:
            n_empty += 1
            continue
        if leaks_answer(hint, row["final_answer"]):
            n_leaked += 1
            continue
        kept.append({
            "question": row["question"],
            "final_answer": row["final_answer"],
            "hint": hint,
            "gen_model": args.model,
            "dataset": args.dataset,
        })
    print(f"Generated {len(outputs)} hints -> kept {len(kept)} "
          f"(dropped {n_leaked} answer leaks, {n_empty} empty, {n_unclosed} unclosed-thinking)")
    if n_unclosed:
        print(f"  note: {n_unclosed} outputs were an unclosed <think> trace -- "
              f"raise --max-tokens (currently {args.max_tokens}) for this thinking model")

    Dataset.from_list(kept).save_to_disk(out_dir)
    print(f"Saved hint dataset -> {out_dir}")
    print(f"  sample hint: {kept[0]['hint'][:200]!r}" if kept else "  (no rows kept!)")


if __name__ == "__main__":
    main()
