"""Token-level first-error labels for credit rollouts (one judge config per run).

For each (incorrect) rollout we ask a judge model to locate the FIRST mathematical
/ reasoning error that is NOT corrected later, as a *verbatim quote*. The quote is
matched against the canonical token reconstruction of the completion
(``common.reconstruct_trace``) and mapped to a contiguous token span — giving a
per-token error mask aligned with the per-token credit signals (A_t / reweight_t).

This produces ONE label set per invocation; the credit study uses four (see
design.md / the comparison script):
  - matched regime (judge = the model whose credit we compare, same info):
      opd_blind   : judge = 30B teacher,  --pi none
      opsd_final  : judge = student,      --pi answer
      opsd_gold   : judge = student,      --pi solution
  - external anchor (fixed strong judge, best-informed; separate analysis):
      anchor      : judge = Qwen/Qwen3.6-27B,  --pi solution|answer|both

The judge's tokenizer is irrelevant — it only emits text; the span mapping uses the
student's stored ``token_str``. Privileged-info ``solution`` is DeepMath's
r1_solution_1, rejoined by problem_id (= deepmath_<idx>).

Run from the repo root, e.g. (OPSD-final matched judge, student self):
    PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=7 \
    python -m credit_assignment.label_errors \
        --rollouts data/credit_assignment/rollouts_Qwen_Qwen3-1.7B_deepmath.jsonl \
        --judge-model Qwen/Qwen3-1.7B --pi answer --label opsd_final \
        --max-traces 200 --max-model-len 40000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

import re

from credit_assignment.common import (
    faithful_trace_and_offsets,
    quote_to_token_span,
)


SYSTEM_PROMPT = (
    "You are a careful math grader. You are given a problem and a step-by-step "
    "solution attempt. The attempt may or may not contain an error. Find the FIRST "
    "place that is mathematically or logically wrong and is NOT corrected by a later "
    "step. If every step is correct, report no error."
)

USER_TEMPLATE = """Problem:
{question}
{pi_block}
Solution attempt to grade (verbatim):
<<<
{trace}
>>>

The "quote" and "pivotal" fields MUST be copied EXACTLY, character-for-character,
from the text between <<< and >>> (same words, same LaTeX, same symbols). Do NOT
paraphrase, translate, summarize, shorten, or fix them — if a string does not appear
verbatim in the attempt it is wrong. After any reasoning, end your reply with ONLY a
JSON object of this exact form and nothing after it:
{{"has_error": true or false, "quote": "the verbatim sentence or equation from the attempt containing the first uncorrected error, or null", "pivotal": "the shortest verbatim substring of that quote that is the actual mistake, or null", "reason": "one short sentence"}}"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Token-level first-error labels for credit rollouts.")
    p.add_argument("--rollouts", required=True, help="JSONL with per-token token_str (rollouts or advantages).")
    p.add_argument("--judge-model", required=True)
    p.add_argument("--pi", choices=["none", "answer", "solution", "both"], default="none",
                   help="Privileged info the judge sees about the correct result.")
    p.add_argument("--label", required=True, help="Label key for outputs (e.g. opd_blind, opsd_final, anchor).")
    p.add_argument("--only-incorrect", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-truncated", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip completions with no closing </think> (uncorrected-error is ill-defined).")
    p.add_argument("--max-traces", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=16000)
    p.add_argument("--fuzzy-cutoff", type=float, default=60.0,
                   help="rapidfuzz partial_ratio (0-100) floor for a fuzzy quote match.")
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True,
                   help="Judge's own thinking mode (Qwen3); the critique JSON is parsed from the tail.")
    p.add_argument("--output", default=None,
                   help="JSONL path (default data/credit_assignment/errors_<label>_<rolloutstem>.jsonl)")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    return p.parse_args()


def load_rollouts(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    kept = []
    for r in records:
        if args.only_incorrect and r.get("correct"):
            continue
        if args.skip_truncated and "</think>" not in r.get("completion_text", ""):
            continue
        kept.append(r)
    if args.max_traces is not None and args.max_traces < len(kept):
        import random
        random.seed(args.seed)
        kept = random.sample(kept, args.max_traces)
    return kept


def build_solution_map(records: list[dict]) -> dict[str, str]:
    """Rejoin DeepMath r1_solution_1 by problem_id (= deepmath_<idx>)."""
    from datasets import load_dataset

    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    sol: dict[str, str] = {}
    for r in records:
        pid = str(r["problem_id"])
        if pid.startswith("deepmath_"):
            sol[pid] = ds[int(pid.split("_", 1)[1])]["r1_solution_1"] or ""
    return sol


def pi_block(pi: str, rec: dict, solution_map: dict[str, str]) -> str:
    parts = []
    if pi in ("answer", "both"):
        parts.append(f"The correct final answer is: {rec['gold_answer']}")
    if pi in ("solution", "both"):
        sol = solution_map.get(str(rec["problem_id"]), "")
        if sol.strip():
            parts.append(f"A correct reference solution:\n{sol}")
    return ("\n" + "\n\n".join(parts) + "\n") if parts else "\n"


def _extract_str_field(text: str, field: str) -> str | None:
    """Raw value of a string field, taking backslashes LITERALLY (not JSON escapes).

    The quote/pivotal must match the trace, which contains literal LaTeX backslashes
    (``\\frac``, ``\\infty``). JSON unescaping would turn ``\\frac`` into a form-feed
    (``\\f``) and never match, so we capture the verbatim substring between the field's
    quotes instead — ending at a ``"`` that is followed by ``,`` or ``}`` (so interior
    quotes don't terminate early). Returns the last match (the final answer object).
    """
    pat = re.compile(r'"' + re.escape(field) + r'"\s*:\s*"(.*?)"\s*(?=[,}])', re.DOTALL)
    matches = pat.findall(text)
    return matches[-1] if matches else None


def parse_quote_critique(text: str) -> dict:
    """Tolerant critique parser (regex, not json.loads — LaTeX-safe).

    Reads the LAST ``has_error`` and the raw quote/pivotal/reason field values. We
    avoid ``json.loads`` entirely because math quotes carry ``{}`` and lone
    backslashes that break both brace-matching and JSON-escape rules.
    """
    flags = re.findall(r'"has_error"\s*:\s*(true|false)', text)
    if not flags:
        return {"has_error": None, "quote": None, "pivotal": None, "reason": None, "parse_ok": False}
    reason = _extract_str_field(text, "reason")
    if flags[-1] == "false":
        return {"has_error": False, "quote": None, "pivotal": None, "reason": reason, "parse_ok": True}
    return {
        "has_error": True,
        "quote": _extract_str_field(text, "quote"),
        "pivotal": _extract_str_field(text, "pivotal"),
        "reason": reason,
        "parse_ok": True,
    }


def main() -> None:
    args = parse_args()
    rollouts_path = Path(args.rollouts)
    stem = rollouts_path.stem.replace("rollouts_", "").replace("advantages_", "")
    output_path = (
        Path(args.output) if args.output
        else rollouts_path.parent / f"errors_{args.label}_{stem}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = select_records(load_rollouts(rollouts_path), args)
    print(f"Labeling {len(records)} rollouts (label={args.label}, pi={args.pi})", file=sys.stderr)
    if not records:
        raise SystemExit("No rollouts selected.")

    solution_map = build_solution_map(records) if args.pi in ("solution", "both") else {}

    llm_kwargs = dict(
        model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Canonical faithful traces + per-token offsets via the *student* tokenizer
    # (judge-agnostic; keeps math unicode intact so quotes map exactly).
    from transformers import AutoTokenizer
    student_tok = AutoTokenizer.from_pretrained(
        records[0]["student_model"], trust_remote_code=True)
    traces, offsets = [], []
    for r in records:
        full, offs = faithful_trace_and_offsets(
            student_tok, [t["token_id"] for t in r["tokens"]])
        traces.append(full)
        offsets.append(offs)

    prompts = []
    for r, trace in zip(records, traces):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                question=r["problem"], pi_block=pi_block(args.pi, r, solution_map), trace=trace)},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0, seed=args.seed)
    outputs = llm.generate(prompts, sampling)

    status_counts: dict[str, int] = {}
    n_error = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for r, trace, offs, output in zip(records, traces, offsets, outputs, strict=True):
            crit = parse_quote_critique(output.outputs[0].text)
            span = {"status": "no_error", "score": None, "char_span": None, "token_span": None}
            pivotal_span = None
            if crit["parse_ok"] and crit["has_error"] and crit["quote"]:
                span = quote_to_token_span(trace, offs, crit["quote"], fuzzy_cutoff=args.fuzzy_cutoff)
                n_error += int(span["status"] != "not_found")
                if crit["pivotal"] and span["char_span"]:
                    pivotal_span = quote_to_token_span(trace, offs, crit["pivotal"], fuzzy_cutoff=args.fuzzy_cutoff)
            elif not crit["parse_ok"]:
                span = {"status": "parse_failed", "score": None, "char_span": None, "token_span": None}
            status_counts[span["status"]] = status_counts.get(span["status"], 0) + 1

            handle.write(json.dumps({
                "problem_id": r.get("problem_id"),
                "row_index": r.get("row_index"),
                "label": args.label,
                "judge_model": args.judge_model,
                "pi": args.pi,
                "correct": r.get("correct"),
                "num_completion_tokens": len(r["tokens"]),
                "has_error": crit["has_error"],
                "quote": crit["quote"],
                "pivotal": crit["pivotal"],
                "reason": crit["reason"],
                "parse_ok": crit["parse_ok"],
                "match_status": span["status"],
                "match_score": span["score"],
                "char_span": span["char_span"],
                "token_span": span["token_span"],
                "pivotal_token_span": pivotal_span["token_span"] if pivotal_span else None,
                "pivotal_match_score": pivotal_span["score"] if pivotal_span else None,
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} labels -> {output_path}", file=sys.stderr)
    print(f"  located errors: {n_error} | match-status: {json.dumps(status_counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
