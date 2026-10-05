"""Generate complete self-hints on the frozen teacher cohort, with soft length targets.

Example: CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m utils.gen_teacher_hint_variants
  --teacher-study-dir results/teacher_uncertainty/default_hint/Qwen_Qwen3-1.7B
  --num-problems 20 --output-dir results/teacher_hint_variants/pilot/Qwen3-1.7B/hints
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from eval.hint_compare_cache import ConditionCache, digest
from eval.teacher_cohort import load_teacher_cohort
from eval.teacher_hint_common import LEVELS, freeze, request_seed, save_rows, write_json
from utils.gen_hint_variants import LEVELS as DEFINITIONS
from utils.gen_hint_variants import classify_hint
from utils.gen_hints import HINT_VALIDATION_VERSION

PROMPT_VERSION = "teacher_hint_soft_targets_v2"


def build_messages(row, level, target):
    forms = {
        "hint_short": "Use exactly ONE short sentence naming the pivotal idea. No equations, numbered steps, or calculations.",
        "hint_medium": "Use two or three concise sentences. Name the strategy and one or two intermediate insights. Do not carry out the calculations.",
        "hint_detailed": "Use five to eight coaching sentences describing the strategy, useful intermediate relationships, and why the steps connect. You may use symbolic equations, but leave their evaluation to the solver.",
    }
    return [
        {
            "role": "system",
            "content": "You are a mathematics tutor writing a HINT for a learner who must still solve the problem. The reference is private teaching material, not a response to imitate. Give instructions for what the learner should try; never perform the solution for them. Never reveal the requested value, final expression, or equivalent answer. Do not use a thinking block or boxed expressions.",
        },
        {
            "role": "user",
            "content": f"<problem>\n{row['question']}\n</problem>\n\n<private_reference>\n{row['solution']}\n</private_reference>\n\nNow write ONLY a hint addressed to the learner. {DEFINITIONS[level][1]} {forms[level]} Aim for about {target} tokens, finishing naturally. Use coaching verbs such as 'consider', 'rewrite', or 'look for'. Do not repeat the question. Do not substitute numerical values, evaluate the last expression, or state what the answer is. Stop once the learner has guidance for their next steps.",
        },
    ]


def diagnostics(rows):
    result = {}
    for level in LEVELS:
        selected = [r for r in rows if r["condition"] == level]
        lengths = sorted(r["n_tokens"] for r in selected)
        result[level] = {
            "n_hints": len(selected),
            "mean_tokens": sum(lengths) / len(lengths) if lengths else None,
            "median_tokens": lengths[len(lengths) // 2] if lengths else None,
            "min_tokens": min(lengths, default=None),
            "max_tokens": max(lengths, default=None),
            "invalid_counts": dict(
                Counter(r["invalid_reason"] for r in selected if r["invalid_reason"])
            ),
            "n_truncated": sum(r["truncated"] for r in selected),
        }
    grouped = {}
    for row in rows:
        grouped.setdefault(row["question_id"], {})[row["condition"]] = row
    result["n_valid_triplets"] = sum(
        all(
            level in group
            and not group[level]["invalid_reason"]
            and not group[level]["truncated"]
            for level in LEVELS
        )
        for group in grouped.values()
    )
    result["n_questions"] = len(grouped)
    return result


def generate(args):
    from transformers import AutoTokenizer

    from eval.demo_gain import tokenizer_hash

    problems, source = load_teacher_cohort(args.teacher_study_dir, args.cohort_dir)
    if args.num_problems:
        problems = problems[: args.num_problems]
    model, revision = source["teacher_model"], source["generation"]["revision"]
    if not revision:
        raise ValueError("Source study must pin a model revision")
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, trust_remote_code=True
    )
    if tokenizer_hash(tokenizer) != source["generation"]["tokenizer_hash"]:
        raise ValueError("Source tokenizer mismatch")
    config = {
        "version": 1,
        "prompt_version": PROMPT_VERSION,
        "definitions_hash": digest(DEFINITIONS),
        "prompts_hash": digest(
            [
                [
                    build_messages(
                        row,
                        level,
                        getattr(args, level.removeprefix("hint_") + "_target"),
                    )
                    for level in LEVELS
                ]
                for row in problems
            ]
        ),
        "validation_version": HINT_VALIDATION_VERSION,
        "model": model,
        "revision": revision,
        "tokenizer_hash": tokenizer_hash(tokenizer),
        "source_meta_hash": digest(source),
        "problems": problems,
        "selection": "first_N_in_frozen_teacher_order"
        if args.num_problems
        else "all_frozen_teacher_questions",
        "targets": {
            level: getattr(args, level.removeprefix("hint_") + "_target")
            for level in LEVELS
        },
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": -1,
        "seed": args.seed,
        "dtype": "bfloat16",
        "enable_thinking": False,
        "vllm_version": __import__("vllm").__version__,
    }
    root = Path(args.output_dir)
    if args.dry_run:
        print(
            f"{model}: {len(problems)} questions, {3 * len(problems)} hint requests; targets={config['targets']}"
        )
        return
    freeze(root / "experiment.json", config)
    cache = ConditionCache(root, "hints", config)
    rows, pending = [], []
    for problem in problems:
        for level in LEVELS:
            messages = build_messages(problem, level, config["targets"][level])
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            seed = request_seed(args.seed, problem["question_id"], level)
            request = {"prompt": prompt, "seed": seed, "gold": problem["final_answer"]}
            identity = {
                "question_id": problem["question_id"],
                "question_idx": problem["question_idx"],
                "demo_hash": problem["demo_hash"],
                "condition": level,
                "sample_idx": 0,
                "target_tokens": config["targets"][level],
                "seed": seed,
            }
            key = cache.key(request)
            saved = cache.load(key)
            if (
                saved is None
                and len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                + args.max_tokens
                > args.max_model_len
            ):
                saved = {
                    "hint": "",
                    "token_ids": [],
                    "n_tokens": 0,
                    "truncated": False,
                    "invalid_reason": "over_generator_context",
                    "finish_reason": None,
                }
                cache.save(key, saved)
            if saved is None:
                pending.append((identity, request, key))
            else:
                rows.append({**identity, **saved})
    if pending:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model,
            revision=revision,
            tokenizer_revision=revision,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            generation_config="vllm",
            seed=args.seed,
            trust_remote_code=True,
        )
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            outputs = llm.generate(
                [r[1]["prompt"] for r in batch],
                [
                    SamplingParams(
                        n=1,
                        seed=r[1]["seed"],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=1.0,
                        top_k=-1,
                    )
                    for r in batch
                ],
            )
            for (identity, request, key), output in zip(batch, outputs, strict=True):
                if len(output.outputs) != 1:
                    raise ValueError("Expected one hint per request")
                completion = output.outputs[0]
                hint = completion.text.strip()
                saved = {
                    "hint": hint,
                    "token_ids": list(completion.token_ids),
                    "n_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "truncated": completion.finish_reason == "length",
                    "invalid_reason": classify_hint(hint, request["gold"]),
                }
                cache.save(key, saved)
                rows.append({**identity, **saved})
            print(
                f"Hints: {min(start + args.batch_size, len(pending))}/{len(pending)} uncached requests",
                flush=True,
            )
    rows.sort(key=lambda r: (r["question_idx"], r["condition"]))
    save_rows(root, "hints", rows, config)
    by_id = {p["question_id"]: p for p in problems}
    write_json(
        root / "review_template.json",
        {
            "version": 1,
            "hints_hash": digest(rows),
            "reviews": [
                {
                    "question_id": r["question_id"],
                    "condition": r["condition"],
                    "hint_hash": digest(r["hint"]),
                    "hint": r["hint"],
                    "question": by_id[r["question_id"]]["question"],
                    "gold_answer": by_id[r["question_id"]]["final_answer"],
                    "automatic_flag": r["invalid_reason"],
                    "decision": "pending",
                    "reason": "",
                }
                for r in rows
            ],
        },
    )
    report = diagnostics(rows)
    write_json(root / "diagnostics.json", report)
    print(report, flush=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-study-dir", required=True)
    p.add_argument("--cohort-dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-problems", type=int, default=0)
    for level in LEVELS:
        p.add_argument(
            "--" + level.removeprefix("hint_") + "-target",
            type=int,
            default=DEFINITIONS[level][0],
        )
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=40000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    if (
        args.num_problems < 0
        or min(
            args.batch_size, args.short_target, args.medium_target, args.detailed_target
        )
        < 1
    ):
        p.error("Counts and targets must be positive; num-problems may be zero")
    if (
        not 0 < args.max_tokens < args.max_model_len
        or not 0 < args.gpu_memory_utilization <= 1
        or args.temperature <= 0
    ):
        p.error("Invalid context, memory, or sampling setting")
    generate(args)


if __name__ == "__main__":
    main()
