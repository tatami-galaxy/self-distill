"""Generate student rollouts on a DeepMath subset and self-score log-probs at T=1.

For each sampled problem:
  1. Build the chat prompt (eval SYSTEM_PROMPT + chat template), keep exact prompt ids.
  2. Sample one student rollout at T=1, top_p=1, requesting top-k logprobs.
  3. Read the student's per-token log-probs from the generation output.
  4. Grade the boxed answer against gold.

Saves one JSONL record per rollout (correct and incorrect) under
data/credit_assignment/. The rollouts file is teacher-agnostic.

Run from the repo root, e.g.:
    python -m credit_assignment.generate_rollouts \
        --student Qwen/Qwen3-1.7B --num-samples 8 --max-tokens 32000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from error_detect.common import (
    build_prompt_ids,
    read_generation_logprobs,
    token_strings,
)
from eval.utils import DATASET_REGISTRY_EVAL, extract_boxed_answer, grade_answer


@dataclass
class RolloutRecord:
    problem_id: str
    row_index: int
    problem: str
    gold_answer: str
    level: int
    subject: str
    student_model: str
    prompt_token_ids: list[int]
    completion_text: str
    pred_answer: str | None
    correct: bool
    num_completion_tokens: int
    tokens: list[dict]  # {token_id, token_str, student_lp, student_topk}
    sampling: dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate + self-score student rollouts.")
    p.add_argument("--student", default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_EVAL))
    p.add_argument("--levels", nargs="*", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)  # trl grpoconfig default
    p.add_argument("--top-p", type=float, default=1.0) # trl grpoconfig default
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--topk", type=int, default=20, help="top-k logprobs to store per token")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True,
        help="Toggle the student's thinking mode in the chat template (default on).",
    )
    p.add_argument("--output", default=None,
                   help="JSONL path (default data/credit_assignment/rollouts_<student>_<dataset>.jsonl)")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    return p.parse_args()


def default_output(args: argparse.Namespace) -> Path:
    slug = args.student.replace("/", "_")
    return Path(f"data/credit_assignment/rollouts_{slug}_{args.dataset}.jsonl")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else default_output(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.temperature != 1.0 or args.top_p != 1.0:
        print(
            f"WARNING: temperature={args.temperature}, top_p={args.top_p} != (1.0, 1.0). "
            "Reading student log-probs from the generation output assumes the rollout is "
            "drawn from the policy's true (T=1, top_p=1) distribution; with truncated/"
            "scaled sampling these log-probs no longer match pi_theta and A_t will be biased.",
            file=sys.stderr,
        )

    problems = DATASET_REGISTRY_EVAL[args.dataset](levels=args.levels)
    print(f"Loaded {len(problems)} problems from {args.dataset}", file=sys.stderr)
    if args.num_samples is not None and args.num_samples < len(problems):
        import random

        random.seed(args.seed)
        problems = random.sample(problems, args.num_samples)
    print(f"Generating rollouts for {len(problems)} problems", file=sys.stderr)

    llm = LLM(
        model=args.student,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        trust_remote_code=True,
        max_logprobs=args.topk,
    )
    tokenizer = llm.get_tokenizer()

    prompt_ids_list = [
        build_prompt_ids(tokenizer, p["problem"], enable_thinking=args.enable_thinking)
        for p in problems
    ]

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        logprobs=args.topk,
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_list], sampling
    )

    sampling_meta = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "topk": args.topk,
    }

    n_written = 0
    n_correct = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for i, (prob, output) in enumerate(zip(problems, outputs, strict=True)):
            comp = output.outputs[0]
            comp_ids = list(comp.token_ids)
            if not comp_ids:  # empty generation
                continue
            # sampled (temp 1.0) token lp, topk lp
            per_token = read_generation_logprobs(output, topk=args.topk)
            strs = token_strings(tokenizer, comp_ids)
            tokens = [
                {
                    "token_id": int(tid),
                    "token_str": strs[j],
                    "student_lp": per_token[j]["chosen_lp"],
                    "student_topk": per_token[j]["topk"],
                }
                for j, tid in enumerate(comp_ids)
            ]
            pred_answer = extract_boxed_answer(comp.text)
            correct = grade_answer(pred_answer, prob["answer"], prob.get("problem", ""))
            record = RolloutRecord(
                problem_id=prob.get("unique_id") or f"{args.dataset}:{i}",
                row_index=i,
                problem=prob["problem"],
                gold_answer=str(prob["answer"]),
                level=int(prob.get("level", 0)),
                subject=prob.get("subject", ""),
                student_model=args.student,
                prompt_token_ids=prompt_ids_list[i],
                completion_text=comp.text,
                pred_answer=pred_answer,
                correct=correct,
                num_completion_tokens=len(comp_ids),
                tokens=tokens,
                sampling=sampling_meta,
            )
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            n_written += 1
            n_correct += int(correct)

    print(
        f"Wrote {n_written} rollouts ({n_correct} correct, "
        f"{n_written - n_correct} incorrect) -> {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
