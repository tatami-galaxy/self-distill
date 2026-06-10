"""Score student rollouts under a teacher and compute per-token OPD credit.

Reads a rollouts JSONL (from generate_rollouts.py), teacher-forces each rollout's
exact token ids through the teacher at T=1 (vLLM prompt_logprobs), and computes
the sampled-token reverse-KL credit per completion token:

    A_t = teacher_lp(y_t) - student_lp(y_t)

A_t > 0: teacher endorses the sampled token more than the student (reinforce).
A_t < 0: teacher dislikes it (blame). See design.md.

The student rollout is reused as-is; only the teacher changes. Full-distribution
KL is deferred (top-k log-probs for both models are stored for that later).

Run from the repo root, e.g.:
    PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=6,7 \
    python -m credit_assignment.score_teacher \
        --rollouts data/credit_assignment/rollouts_Qwen_Qwen3-1.7B_deepmath.jsonl \
        --teacher Qwen/Qwen3-30B-A3B-Thinking-2507 --tensor-parallel-size 2 \
        --max-model-len 33000

Notes for the 30B-A3B MoE teacher:
  - PATH must include the venv bin so FlashInfer's JIT can find `ninja`.
  - Needs tensor-parallel-size >= 2 (weights ~60 GB) and leave headroom for the
    FlashInfer workspace + CUDA-graph capture, hence the 0.80 memory default.
  - --max-model-len must exceed the longest prompt+completion in the rollouts
    (rollouts go up to 32k generated tokens).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from vllm import LLM

from credit_assignment.common import expected_advantage, score_prompt_logprobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher-score rollouts -> per-token A_t (OPD).")
    p.add_argument("--rollouts", required=True, help="JSONL from generate_rollouts.py")
    p.add_argument("--teacher", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    p.add_argument("--topk", type=int, default=20, help="top-k teacher logprobs to store")
    p.add_argument("--score-batch-size", type=int, default=8)
    p.add_argument("--output", default=None,
                   help="JSONL path (default advantages_opd_<teacher>_revkl_<rolloutstem>.jsonl)")
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80,
                   help="Kept below ~0.85 to leave room for the FlashInfer MoE "
                        "workspace + CUDA-graph capture (0.9 OOMs the 30B-A3B teacher).")
    p.add_argument("--max-model-len", type=int, default=None)
    return p.parse_args()


def default_output(args: argparse.Namespace, rollouts_path: Path) -> Path:
    teacher_slug = args.teacher.replace("/", "_")
    stem = rollouts_path.stem.replace("rollouts_", "")
    return rollouts_path.parent / f"advantages_opd_{teacher_slug}_revkl_{stem}.jsonl"


def load_rollouts(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    rollouts_path = Path(args.rollouts)
    output_path = Path(args.output) if args.output else default_output(args, rollouts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_rollouts(rollouts_path)
    print(f"Loaded {len(records)} rollouts from {rollouts_path}", file=sys.stderr)

    llm_kwargs = dict(
        model=args.teacher,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    n_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for batch_start in range(0, len(records), args.score_batch_size):
            batch = records[batch_start : batch_start + args.score_batch_size]
            seqs: list[tuple[list[int], int]] = []
            scored_idx: list[int] = []
            for k, rec in enumerate(batch):
                comp_ids = [t["token_id"] for t in rec["tokens"]]
                if not comp_ids:
                    continue
                full_ids = list(rec["prompt_token_ids"]) + comp_ids
                seqs.append((full_ids, len(rec["prompt_token_ids"])))
                scored_idx.append(k)
            if not seqs:
                continue

            scored = score_prompt_logprobs(llm, seqs, topk=args.topk)

            for k, per_token in zip(scored_idx, scored):
                rec = batch[k]
                for tok, tscore in zip(rec["tokens"], per_token, strict=True):
                    teacher_lp = tscore["chosen_lp"]
                    tok["teacher_lp"] = teacher_lp
                    tok["teacher_topk"] = tscore["topk"]
                    # Sampled-token advantage (1-sample REINFORCE coefficient).
                    a_t = teacher_lp - tok["student_lp"]
                    tok["A_t"] = a_t
                    # Student-weighted expected advantage = -KL_t: the per-position
                    # pull the *dense* OPD/OPSD gradient responds to (top-k approx).
                    abar = expected_advantage(tok.get("student_topk"), tscore["topk"])
                    tok["Abar_t"] = abar
                    # Dense-training logit change of the sampled token (proportional;
                    # learning rate dropped): pi_student(y_t) * (A_t - Abar_t). Sign =
                    # whether this token is up/down-weighted under training.
                    tok["reweight_t"] = (
                        math.exp(tok["student_lp"]) * (a_t - abar)
                        if abar is not None else None
                    )
                rec["teacher_model"] = args.teacher
                rec["credit"] = {
                    "method": "opd",
                    "divergence": "reverse_kl",
                    "signals": {
                        "A_t": "sampled-token log-ratio  log pi_T(y_t) - log pi_S(y_t)",
                        "Abar_t": "sum_v pi_S(v)[log pi_T(v) - log pi_S(v)] = -KL_t (top-k approx)",
                        "reweight_t": "pi_S(y_t) * (A_t - Abar_t): dense-training reweight of y_t",
                    },
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"Wrote {n_written} scored rollouts -> {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
