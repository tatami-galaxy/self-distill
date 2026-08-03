"""
Characterize the GENERATION BEHAVIOR of an SDFT teacher under each privileged
context (PI), and contrast the self-teacher (OPSD) with a strong external teacher
(OPD).

In SDFT the teacher never generates during training -- it *scores* the student's
on-policy tokens. So "how long does the teacher reason" and "how much uncertainty
does it verbalize" are properties of the teacher's generation distribution
pi(.|x,c), which we surface by sampling free completions from each teacher prompt
(built exactly as SDFTTrainer builds it, via passk_pi.build_teacher_messages).

Two behaviors, per completion:
  * length          -- number of generated tokens (how much the teacher reasons)
  * verbalized      -- epistemic-marker count E(y) over the fixed 10-token set from
    uncertainty        "..." (arXiv 2603.24472): {wait, hmm, perhaps, maybe, actually,
                       alternatively, seems, might, likely, check}. Reported raw AND
                       per-1k-tokens (arms differ sharply in length), split into the
                       <think> trace vs the post-</think> answer.

Two teacher kinds share this one script; they differ only in which model generates:
  * self / OPSD  -- teacher-model == problem-model (the student), PI in {none, answer,
                    hint, full}. The collapse prediction: full PI -> short, low-E(y).
  * strong / OPD -- teacher-model = a bigger model (e.g. Qwen3-30B-A3B-Thinking-2507),
                    PI = none (its edge is capability, not information).

The problem set is FIXED across both runs: --problem-model's hint cache, restricted
to the full-PI-feasible subset under that model's tokenizer -- computed the same way
regardless of which arms an invocation runs, so separate model-load runs line up
problem-for-problem and every mean shares a denominator.

# self-teacher (OPSD), all PIs
CUDA_VISIBLE_DEVICES=7 uv run python -m eval.teacher_behavior \
    --teacher-model Qwen/Qwen3-1.7B --pi-modes none answer hint full

# strong teacher (OPD), query-only, same problems
CUDA_VISIBLE_DEVICES=7 uv run python -m eval.teacher_behavior \
    --teacher-model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --problem-model Qwen/Qwen3-1.7B --pi-modes none
"""

import argparse
import json
import os
import re
import time

from vllm import LLM, SamplingParams

from eval.passk_pi import (
    build_teacher_messages,
    load_eval_problems,
    restrict_to_full_feasible,
)
from utils import DATASET_REGISTRY_TRAIN, grade


# ---------------------------------------------------------------------------
# Uncertainty verbalization -- epistemic markers (arXiv 2603.24472, set T)
# ---------------------------------------------------------------------------

EPISTEMIC_MARKERS = [
    "wait", "hmm", "perhaps", "maybe", "actually",
    "alternatively", "seems", "might", "likely", "check",
]
# Whole-word, case-insensitive: "check" must not fire inside "backcheck". We match
# the paper's tokens verbatim (no stemming) so E(y) stays comparable to theirs.
_MARKER_RE = {m: re.compile(rf"\b{m}\b", re.IGNORECASE) for m in EPISTEMIC_MARKERS}


def count_epistemic(text: str) -> dict[str, int]:
    """Per-marker occurrence counts in `text` (the paper's E(y) = sum of these)."""
    return {m: len(rx.findall(text)) for m, rx in _MARKER_RE.items()}


def split_think(text: str) -> tuple[str, str, bool]:
    """Split a completion into (think_trace, post_answer, closed).

    Qwen3 thinking completions are `<think>...</think> answer`. We split on the
    first `</think>`: everything before is the reasoning trace, everything after
    is the polished answer. `closed` is False when the trace never closed (the
    cap truncated it) -- then the whole thing is trace and there is no answer.
    """
    if "</think>" in text:
        pre, post = text.split("</think>", 1)
        return pre, post, True
    return text, "", False


def measure_completion(text: str, n_tokens: int, finish_reason: str | None,
                       gold: str) -> dict:
    """All per-completion behavior metrics for one teacher generation."""
    think, post, closed = split_think(text)
    e_think = count_epistemic(think)
    e_post = count_epistemic(post)
    e_total = {m: e_think[m] + e_post[m] for m in EPISTEMIC_MARKERS}
    _, correct = grade(text, gold)
    return {
        "text": text,
        "n_tokens": n_tokens,
        "correct": correct,
        # Two distinct flags, kept separate:
        #  * truncated -- hit the token cap (finish_reason "length"); its length is
        #    right-censored and biases mean_tokens downward.
        #  * unclosed  -- no </think> in the completion. A truncated completion is
        #    always unclosed, but a completion can also finish (reason "stop") while
        #    still inside the think block -- e.g. under full PI it reaches \boxed
        #    before emitting </think>. That length is genuine, not censored.
        "truncated": finish_reason == "length",
        "unclosed": not closed,
        "e_think": sum(e_think.values()),
        "e_post": sum(e_post.values()),
        "e_total": sum(e_total.values()),
        "e_by_marker": e_total,
    }


# ---------------------------------------------------------------------------
# Per-arm aggregation
# ---------------------------------------------------------------------------


def summarize(records: list[dict]) -> dict:
    """Mean behavior over all completions in one arm.

    E(y) per-1k-tokens uses pooled totals (sum E / sum tokens), not a mean of
    per-completion rates, so short truncated completions don't dominate the rate.
    """
    n = len(records)
    tot_tokens = sum(r["n_tokens"] for r in records)
    e_total = sum(r["e_total"] for r in records)
    e_think = sum(r["e_think"] for r in records)
    by_marker = {m: sum(r["e_by_marker"][m] for r in records) for m in EPISTEMIC_MARKERS}
    return {
        "n_completions": n,
        "mean_tokens": tot_tokens / n,
        "pass@1": sum(r["correct"] for r in records) / n,
        "trunc_rate": sum(r["truncated"] for r in records) / n,
        "unclosed_rate": sum(r["unclosed"] for r in records) / n,
        "mean_e_total": e_total / n,
        "mean_e_think": e_think / n,
        "mean_e_post": sum(r["e_post"] for r in records) / n,
        # length-normalized: the comparable rate across arms of differing length
        "e_per_1k_tokens": 1000 * e_total / tot_tokens if tot_tokens else 0.0,
        "e_by_marker_per_1k": {
            m: (1000 * by_marker[m] / tot_tokens if tot_tokens else 0.0)
            for m in EPISTEMIC_MARKERS
        },
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_arm(llm, tokenizer, problems, pi_mode, sampling_params) -> list[dict]:
    """Sample completions for one PI arm and measure each; one record per (problem,
    sample). Carries problem_idx so self- and strong-teacher runs align by problem."""
    prompts = [
        tokenizer.apply_chat_template(
            build_teacher_messages(p, pi_mode), tokenize=False, add_generation_prompt=True
        )
        for p in problems
    ]
    outputs = llm.generate(prompts, sampling_params)
    records = []
    for pi, (p, out) in enumerate(zip(problems, outputs, strict=True)):
        for si, comp in enumerate(out.outputs):
            rec = measure_completion(
                comp.text, len(comp.token_ids), comp.finish_reason, p["answer"]
            )
            rec["problem_idx"] = pi
            rec["sample_idx"] = si
            records.append(rec)
    return records


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--teacher-model", default="Qwen/Qwen3-1.7B",
                   help="The model that GENERATES. Self-teacher (== --problem-model) "
                        "or a strong external teacher (OPD).")
    p.add_argument("--problem-model", default=None,
                   help="Whose hint cache defines the (fixed) eval set. Defaults to "
                        "--teacher-model (the self-teacher case). Set to the student "
                        "when the teacher is a strong external model, so both runs "
                        "score the SAME problems.")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()),
                   help="Dataset whose hint cache (under --problem-model) defines the "
                        "fixed eval set. Must match the hint cache's dataset.")
    p.add_argument("--pi-modes", nargs="+", default=["none", "answer", "hint", "full"],
                   choices=["none", "answer", "hint", "full"],
                   help="PI arms to generate. Strong-teacher OPD uses just 'none'.")
    p.add_argument("--num-problems", type=int, default=128,
                   help="Random subset of the problem-model's hint cache.")
    p.add_argument("--n", type=int, default=4, help="Samples per problem.")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Completion budget. Thinking teachers can overrun this; "
                        "truncated completions are flagged (trunc_rate) since their "
                        "censored length biases mean_tokens downward.")
    p.add_argument("--output-dir", default="results/teacher_behavior")
    p.add_argument("--save-completions", action=argparse.BooleanOptionalAction, default=True,
                   help="Dump raw completion text + per-completion metrics to JSONL "
                        "(so future metrics are a re-parse, not a re-generation).")
    # vLLM
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.n < 1:
        p.error("--n must be >= 1")
    problem_model = args.problem_model or args.teacher_model

    # Fixed eval set: problem-model's hint cache, restricted to full-PI-feasible
    # under that model's tokenizer -- computed regardless of which arms we run, so
    # self- and strong-teacher invocations line up problem-for-problem.
    problems = load_eval_problems(
        problem_model, args.num_problems, args.seed, need_full=True, dataset=args.dataset
    )
    from transformers import AutoTokenizer
    problem_tok = AutoTokenizer.from_pretrained(problem_model, trust_remote_code=True)
    problems = restrict_to_full_feasible(
        problems, problem_tok, args.max_model_len - args.max_tokens
    )
    print(f"teacher={args.teacher_model}  problems from={problem_model}  "
          f"|eval set|={len(problems)}  pi_modes={args.pi_modes}")

    llm = LLM(
        model=args.teacher_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=args.n, max_tokens=args.max_tokens,
        seed=args.seed,
    )

    out_dir = os.path.join(args.output_dir, args.teacher_model.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    t0 = time.time()
    for mode in args.pi_modes:
        records = generate_arm(llm, tokenizer, problems, mode, sampling_params)
        summary[mode] = summarize(records)
        s = summary[mode]
        print(f"  {mode:7s}  tokens={s['mean_tokens']:7.1f}  "
              f"E(y)={s['mean_e_total']:5.2f}  E/1k={s['e_per_1k_tokens']:5.2f}  "
              f"pass@1={s['pass@1']*100:5.1f}%  trunc={s['trunc_rate']*100:4.1f}%  "
              f"unclosed={s['unclosed_rate']*100:4.1f}%")
        if args.save_completions:
            jl = os.path.join(out_dir, f"completions_{mode}.jsonl")
            with open(jl, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
    elapsed = time.time() - t0

    out = {
        "teacher_model": args.teacher_model,
        "problem_model": problem_model,
        "n_problems": len(problems),
        "n_samples": args.n,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "epistemic_markers": EPISTEMIC_MARKERS,
        "elapsed_s": elapsed,
        "behavior": summary,
    }
    summary_path = os.path.join(out_dir, "teacher_behavior_summary.json")
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
    main()
