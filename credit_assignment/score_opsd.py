"""Score student rollouts under a *privileged self-teacher* — per-token OPSD credit.

On-policy self-distillation (OPSD): the teacher is the *same* model that produced
the rollout, re-run with privileged information ``f`` prepended to the prompt. The
per-token credit is

    A_t = log pi_theta(y_t | x, f, y_<t) - log pi_theta(y_t | x, y_<t)

i.e. the pointwise information gain from f (a pointwise mutual information, in
expectation -KL(pi_unpriv || pi_priv)_t under on-policy sampling). Semantically
distinct from OPD's capability gap, but the same syntactic log-ratio — see design.md.

The unprivileged baseline ``log pi_theta(y_t | x, y_<t)`` is exactly the
``student_lp`` already stored in the rollouts (read off the T=1 generation), so
OPSD needs only ONE new forward pass: the privileged (f-augmented) teacher-forcing.
The completion token ids are reused verbatim; only the prompt prefix gains f.

Two ``f`` choices are supported, scored in a single model-loaded session:
  - ``final_answer``  : the gold boxed answer (mild hint, verifier-like).
  - ``gold_solution`` : DeepMath's r1_solution_1 reference trace (strong, leak-prone),
                        rejoined to the raw dataset by problem_id (= deepmath_<idx>).

Run from the repo root, e.g.:
    PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 \
    python -m credit_assignment.score_opsd \
        --rollouts data/credit_assignment/rollouts_Qwen_Qwen3-1.7B_deepmath.jsonl \
        --f final_answer gold_solution --max-model-len 40000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from vllm import LLM

from credit_assignment.common import (
    build_prompt_ids,
    expected_advantage,
    score_prompt_logprobs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-teacher score rollouts -> per-token A_t (OPSD).")
    p.add_argument("--rollouts", required=True, help="JSONL from generate_rollouts.py")
    p.add_argument("--student", default=None,
                   help="Model to load (default: the student_model recorded in the rollouts).")
    p.add_argument("--f", nargs="+", default=["final_answer", "gold_solution"],
                   choices=["final_answer", "gold_solution"],
                   help="Privileged-info variant(s) to score; one output file each.")
    p.add_argument("--topk", type=int, default=20, help="top-k privileged logprobs to store")
    p.add_argument("--score-batch-size", type=int, default=8)
    p.add_argument("--output-dir", default=None,
                   help="Dir for outputs (default: alongside the rollouts file).")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Sequences whose f-augmented prompt+completion exceed this are skipped.")
    return p.parse_args()


def load_rollouts(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_solution_map(records: list[dict]) -> dict[str, str]:
    """Rejoin DeepMath r1_solution_1 by problem_id (= ``deepmath_<idx>``).

    The eval loader drops the reference solution, so gold_solution f must come
    from the raw dataset, indexed by the original row position.
    """
    from datasets import load_dataset

    needed = {r["problem_id"] for r in records if str(r["problem_id"]).startswith("deepmath_")}
    if not needed:
        return {}
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    sol: dict[str, str] = {}
    for r in records:
        pid = str(r["problem_id"])
        if not pid.startswith("deepmath_"):
            continue
        idx = int(pid.split("_", 1)[1])
        sol[pid] = ds[idx]["r1_solution_1"] or ""
    missing = sum(1 for pid in needed if not sol.get(pid))
    if missing:
        print(f"WARNING: {missing}/{len(needed)} rollouts have no r1_solution_1", file=sys.stderr)
    return sol


def f_text(variant: str, rec: dict, solution_map: dict[str, str]) -> str | None:
    """Privileged-info string appended to the user turn for one rollout."""
    if variant == "final_answer":
        return f"Hint: the correct final answer is \\boxed{{{rec['gold_answer']}}}."
    if variant == "gold_solution":
        sol = solution_map.get(str(rec["problem_id"]), "")
        if not sol.strip():
            return None
        return f"Here is a worked reference solution you may rely on:\n\n{sol}"
    raise ValueError(variant)


def score_variant(
    llm: LLM,
    tokenizer,
    records: list[dict],
    variant: str,
    solution_map: dict[str, str],
    topk: int,
    batch_size: int,
    max_model_len: int | None,
    output_path: Path,
) -> None:
    """Score one f variant and write an advantages JSONL (OPD-compatible schema)."""
    n_written = 0
    n_skipped = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start : batch_start + batch_size]
            seqs: list[tuple[list[int], int]] = []
            scored_idx: list[int] = []
            for k, rec in enumerate(batch):
                comp_ids = [t["token_id"] for t in rec["tokens"]]
                if not comp_ids:
                    continue
                f = f_text(variant, rec, solution_map)
                if f is None:  # no reference solution for this rollout
                    n_skipped += 1
                    continue
                enable_thinking = rec.get("sampling", {}).get("enable_thinking", True)
                prompt_ids = build_prompt_ids(
                    tokenizer, rec["problem"], enable_thinking=enable_thinking,
                    privileged_info=f,
                )
                full_ids = prompt_ids + comp_ids
                if max_model_len is not None and len(full_ids) > max_model_len:
                    n_skipped += 1
                    continue
                seqs.append((full_ids, len(prompt_ids)))
                scored_idx.append(k)
            if not seqs:
                continue

            scored = score_prompt_logprobs(llm, seqs, topk=topk)

            for k, per_token in zip(scored_idx, scored):
                rec = batch[k]
                for tok, tscore in zip(rec["tokens"], per_token, strict=True):
                    priv_lp = tscore["chosen_lp"]
                    # Stored as teacher_* so visualize.py / the OPD schema apply unchanged;
                    # here the "teacher" is the privileged self-pass pi_theta(.|x,f,.).
                    tok["teacher_lp"] = priv_lp
                    tok["teacher_topk"] = tscore["topk"]
                    a_t = priv_lp - tok["student_lp"]
                    tok["A_t"] = a_t
                    abar = expected_advantage(tok.get("student_topk"), tscore["topk"])
                    tok["Abar_t"] = abar
                    tok["reweight_t"] = (
                        math.exp(tok["student_lp"]) * (a_t - abar)
                        if abar is not None else None
                    )
                rec["teacher_model"] = f"opsd_self:{rec['student_model']}"
                rec["credit"] = {
                    "method": "opsd",
                    "divergence": "reverse_kl",
                    "f": variant,
                    "signals": {
                        "A_t": "log pi(y_t|x,f) - log pi(y_t|x): pointwise info gain from f",
                        "Abar_t": "sum_v pi_unpriv(v)[log pi_priv(v) - log pi_unpriv(v)] = -KL_t (top-k approx)",
                        "reweight_t": "pi(y_t)*(A_t - Abar_t): dense-training reweight of y_t",
                    },
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(
        f"[{variant}] wrote {n_written} scored rollouts ({n_skipped} skipped) -> {output_path}",
        file=sys.stderr,
    )


def main() -> None:
    args = parse_args()
    rollouts_path = Path(args.rollouts)
    out_dir = Path(args.output_dir) if args.output_dir else rollouts_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_rollouts(rollouts_path)
    print(f"Loaded {len(records)} rollouts from {rollouts_path}", file=sys.stderr)

    student = args.student or records[0]["student_model"]
    print(f"Self-teacher model: {student}", file=sys.stderr)

    solution_map = build_solution_map(records) if "gold_solution" in args.f else {}

    llm_kwargs = dict(
        model=student,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_logprobs=args.topk,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    stem = rollouts_path.stem.replace("rollouts_", "")
    for variant in args.f:
        output_path = out_dir / f"advantages_opsd_{variant}_revkl_{stem}.jsonl"
        score_variant(
            llm, tokenizer, records, variant, solution_map,
            topk=args.topk, batch_size=args.score_batch_size,
            max_model_len=args.max_model_len, output_path=output_path,
        )


if __name__ == "__main__":
    main()
