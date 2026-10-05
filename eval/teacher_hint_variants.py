"""Matched self-teacher accuracy and uncertainty under three generated hint levels.

Prepare -> generate -> summarize; sweep also runs cognitive classification and plots
in separate processes. The same completions supply every measurement.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

from eval.hint_compare_cache import ConditionCache, digest
from eval.teacher_hint_common import (
    ARMS,
    LEVELS,
    contrasts,
    freeze,
    load_rows,
    paired_metrics,
    read_json,
    request_seed,
    save_rows,
    validate_completions,
    write_json,
)

DEFAULTS = {
    "n": 8,
    "max_tokens": 8192,
    "max_model_len": 40000,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "seed": 42,
}


def messages_for(problem, arm):
    from utils import PI_HINT, compose_pi_messages, format_prompt

    messages = format_prompt(problem["question"], problem["dataset"])
    return (
        messages
        if arm == "none"
        else compose_pi_messages(messages, PI_HINT.format(hint=problem["hints"][arm]))
    )


def read_hint_reviews(path, hints, hints_hash):
    """Explicit, complete human review, bound to exact hint text and artifacts.

    A reviewer can accept a lexical false positive or reject a false negative.
    Empty/truncated hints remain ineligible regardless of a review decision.
    """
    review = read_json(path)
    if review.get("version") != 1 or review.get("hints_hash") != hints_hash:
        raise ValueError("Hint review refers to a different hint artifact")
    expected = {(r["question_id"], r["condition"]): digest(r["hint"]) for r in hints}
    decisions = {}
    for row in review["reviews"]:
        key = (row["question_id"], row["condition"])
        if (
            key in decisions
            or key not in expected
            or row.get("hint_hash") != expected[key]
        ):
            raise ValueError("Duplicate, unknown, or changed hint in review")
        if (
            row.get("decision") not in ("accept", "reject")
            or not str(row.get("reason", "")).strip()
        ):
            raise ValueError("Each reviewed hint needs accept/reject and a reason")
        decisions[key] = row
    if set(decisions) != set(expected):
        raise ValueError(
            "Review must cover every hint; partial review cannot certify a cohort"
        )
    return decisions, {"path": str(Path(path).resolve()), "hash": digest(review)}


def select_problems(problems, hints, reviews=None):
    """A single, ordered intersection; no solver/judge outcomes enter selection."""
    source = {p["question_id"]: p for p in problems}
    if len(source) != len(problems) or not problems:
        raise ValueError("Empty or duplicated source cohort")
    indexed = {}
    for hint in hints:
        key = (hint["question_id"], hint["condition"])
        if key in indexed or hint["sample_idx"] != 0 or hint["condition"] not in LEVELS:
            raise ValueError("Duplicate or unexpected hint identity")
        problem = source.get(hint["question_id"])
        if (
            problem is None
            or hint["demo_hash"] != problem["demo_hash"]
            or hint["question_idx"] != problem["question_idx"]
        ):
            raise ValueError("Hint source/demonstration mismatch")
        indexed[key] = hint
    selected, excluded = [], []
    for problem in problems:
        values, reasons = {}, []
        for level in LEVELS:
            hint = indexed.get((problem["question_id"], level))
            if hint is None:
                raise ValueError(f"Missing {level} for {problem['question_id']}")
            reviewed = (
                reviews.get((problem["question_id"], level))
                if reviews is not None
                else None
            )
            if reviewed is not None and reviewed["decision"] == "reject":
                reasons.append(f"{level}:review_rejected:{reviewed['reason']}")
            if hint["invalid_reason"] and (
                reviewed is None or hint["invalid_reason"] != "answer_leak"
            ):
                reasons.append(f"{level}:{hint['invalid_reason']}")
            if hint["truncated"]:
                reasons.append(f"{level}:truncated")
            if not hint["hint"].strip():
                reasons.append(f"{level}:empty")
            values[level] = hint["hint"]
        if reasons:
            excluded.append(
                {"question_id": problem["question_id"], "reasons": sorted(set(reasons))}
            )
        else:
            selected.append({**problem, "hints": values})
    return selected, excluded


def load_experiment(root):
    experiment = read_json(Path(root) / "experiment.json")
    hints_config = read_json(Path(experiment["hints_dir"]) / "experiment.json")
    _, meta = load_rows(experiment["hints_dir"], "hints", hints_config)
    if (
        digest(hints_config) != experiment["hints_config_hash"]
        or meta["hash"] != experiment["hints_hash"]
    ):
        raise ValueError("Hint source changed after study preparation")
    review = experiment.get("hint_review")
    if review and digest(read_json(review["path"])) != review["hash"]:
        raise ValueError("Hint review changed after study preparation")
    current_prompts = digest(
        [
            [messages_for(problem, arm) for arm in ARMS]
            for problem in experiment["problems"]
        ]
    )
    if current_prompts != experiment["prompt_hash"]:
        raise ValueError("Teacher prompt construction changed after preparation")
    return experiment


def prepare(args):
    from transformers import AutoTokenizer

    from eval.demo_gain import tokenizer_hash

    root = Path(args.output_dir)
    if (root / "experiment.json").exists() and not args.hints_dir:
        experiment = load_experiment(root)
        if args.hint_review and str(Path(args.hint_review).resolve()) != (
            experiment.get("hint_review") or {}
        ).get("path"):
            raise ValueError("Changed hint review; prepare a new output directory")
        for key in DEFAULTS:
            if (
                getattr(args, key) is not None
                and getattr(args, key) != experiment["generation"][key]
            ):
                raise ValueError(f"Changed {key}; use a new output directory")
        return experiment
    if not args.hints_dir:
        raise ValueError("Initial preparation requires --hints-dir")
    hint_root = Path(args.hints_dir).resolve()
    if hint_root == root.resolve():
        raise ValueError("Hint and evaluation output directories must differ")
    hints_config = read_json(hint_root / "experiment.json")
    hints, hint_meta = load_rows(hint_root, "hints", hints_config)
    reviews, review_meta = (
        read_hint_reviews(args.hint_review, hints, hint_meta["hash"])
        if args.hint_review
        else (None, None)
    )
    problems, excluded = select_problems(hints_config["problems"], hints, reviews)
    generation = {
        key: getattr(args, key) if getattr(args, key) is not None else value
        for key, value in DEFAULTS.items()
    }
    if (
        generation["n"] < 1
        or not 0 < generation["max_tokens"] < generation["max_model_len"]
        or generation["temperature"] <= 0
        or not 0 < generation["top_p"] <= 1
        or generation["top_k"] not in [-1]
        and generation["top_k"] < 1
    ):
        raise ValueError("Invalid generation settings")
    model, revision = hints_config["model"], hints_config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, trust_remote_code=True
    )
    if tokenizer_hash(tokenizer) != hints_config["tokenizer_hash"]:
        raise ValueError("Hint/teacher tokenizer mismatch")
    feasible = []
    for problem in problems:
        over = []
        for arm in ARMS:
            ids = tokenizer.apply_chat_template(
                messages_for(problem, arm),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if len(ids) + generation["max_tokens"] > generation["max_model_len"]:
                over.append(arm + ":context_overflow")
        if over:
            excluded.append({"question_id": problem["question_id"], "reasons": over})
        else:
            feasible.append(problem)
    if not feasible:
        raise ValueError("No questions have three usable, context-feasible hints")
    generation.update(
        model=model,
        revision=revision,
        tokenizer_hash=tokenizer_hash(tokenizer),
        dtype="bfloat16",
        enable_thinking=True,
        vllm_version=__import__("vllm").__version__,
    )
    experiment = {
        "version": 1,
        "hints_dir": str(hint_root),
        "hints_config_hash": digest(hints_config),
        "hints_hash": hint_meta["hash"],
        "hint_review": review_meta,
        "hint_validation": "reviewed"
        if review_meta
        else "lexical_only_not_semantically_validated",
        "source_meta_hash": hints_config["source_meta_hash"],
        "arms": list(ARMS),
        "problems": feasible,
        "n_source_questions": len(hints_config["problems"]),
        "excluded": excluded,
        "generation": generation,
        "prompt_hash": digest(
            [[messages_for(p, arm) for arm in ARMS] for p in feasible]
        ),
    }
    print(
        f"Common cohort: {len(feasible)}/{experiment['n_source_questions']}; {len(feasible) * 4 * generation['n']} responses"
    )
    print(
        "Exclusions:",
        dict(Counter(reason for row in excluded for reason in row["reasons"])),
    )
    if not args.dry_run:
        freeze(root / "experiment.json", experiment)
    return experiment


def generate(args, experiment):
    from transformers import AutoTokenizer

    from eval.demo_gain import tokenizer_hash
    from eval.teacher_uncertainty import measure_completion

    config = experiment["generation"]
    root = Path(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], trust_remote_code=True
    )
    if tokenizer_hash(tokenizer) != config["tokenizer_hash"]:
        raise ValueError("Teacher tokenizer changed after preparation")
    if __import__("vllm").__version__ != config["vllm_version"]:
        raise ValueError("Inference version changed after preparation")
    cache = ConditionCache(
        root,
        "completions",
        {"generation": config, "experiment_hash": digest(experiment)},
    )
    pending, by_arm = [], {arm: [] for arm in ARMS}
    for arm in ARMS:
        for problem in experiment["problems"]:
            prompt = tokenizer.apply_chat_template(
                messages_for(problem, arm),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            request = {
                "prompt": prompt,
                "gold": problem["final_answer"],
                "dataset": problem["dataset"],
                "seed": request_seed(config["seed"], problem["question_id"], arm),
            }
            key = cache.key(request)
            saved = cache.load(key)
            if saved is None:
                pending.append((arm, problem, request, key))
            else:
                by_arm[arm].extend(saved)
    if pending:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=config["model"],
            revision=config["revision"],
            tokenizer_revision=config["revision"],
            dtype=config["dtype"],
            max_model_len=config["max_model_len"],
            generation_config="vllm",
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=config["seed"],
            trust_remote_code=True,
        )
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            params = [
                SamplingParams(
                    n=config["n"],
                    max_tokens=config["max_tokens"],
                    seed=request["seed"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                )
                for _, _, request, _ in batch
            ]
            outputs = llm.generate([item[2]["prompt"] for item in batch], params)
            for (arm, problem, request, key), output in zip(
                batch, outputs, strict=True
            ):
                if len(output.outputs) != config["n"]:
                    raise ValueError("Incomplete response group")
                rows = []
                for i, completion in enumerate(output.outputs):
                    rows.append(
                        {
                            **measure_completion(
                                completion.text,
                                len(completion.token_ids),
                                completion.finish_reason,
                                problem["final_answer"],
                                problem["dataset"],
                            ),
                            "question_idx": problem["question_idx"],
                            "question_id": problem["question_id"],
                            "sample_idx": i,
                            "condition": arm,
                            "token_ids": list(completion.token_ids),
                            "finish_reason": completion.finish_reason,
                            "request_seed": request["seed"],
                        }
                    )
                cache.save(key, rows)
                by_arm[arm].extend(rows)
            print(
                f"Teacher generation: {min(start + args.batch_size, len(pending))}/{len(pending)} uncached questions/arms",
                flush=True,
            )
    order = {p["question_idx"]: i for i, p in enumerate(experiment["problems"])}
    for arm, rows in by_arm.items():
        rows.sort(key=lambda row: (order[row["question_idx"]], row["sample_idx"]))
        validate_completions(rows, experiment["problems"], config["n"])
        save_rows(
            root / arm,
            "completions",
            rows,
            {"experiment_hash": digest(experiment), "arm": arm},
        )


def load_completions(root, experiment, arm):
    rows, meta = load_rows(
        Path(root) / arm,
        "completions",
        {"experiment_hash": digest(experiment), "arm": arm},
    )
    validate_completions(rows, experiment["problems"], experiment["generation"]["n"])
    return rows, meta


def uncertainty_metrics():
    metrics = {
        key: (lambda r, k=key: r[k], lambda r: 1)
        for key in ("n_tokens", "e_total", "e_think", "e_post", "truncated", "unclosed")
    }
    metrics["e_per_1k_tokens"] = (
        lambda r: 1000 * r["e_total"],
        lambda r: r["n_tokens"],
    )
    return metrics


def accuracy_rows(rows, ks):
    from eval.run_eval import pass_at_k

    grouped = {}
    for row in rows:
        grouped.setdefault(row["question_idx"], []).append(row)
    result = []
    for qid, group in grouped.items():
        if any(k > len(group) or k < 1 for k in ks):
            raise ValueError("pass@k requires 1 <= k <= sample count")
        result.append(
            {
                "question_idx": qid,
                **{
                    f"pass@{k}": pass_at_k(
                        len(group), sum(r["correct"] for r in group), k
                    )
                    for k in ks
                },
            }
        )
    return result


def summarize(args, experiment):
    from eval.teacher_uncertainty import summarize as summarize_uncertainty

    n = experiment["generation"]["n"]
    ks = args.k or [k for k in (1, 2, 4, 8) if k <= n]
    if not ks or min(ks) < 1 or max(ks) > n or len(set(ks)) != len(ks):
        raise ValueError("Require unique k values in [1, n]")
    rows = {arm: load_completions(args.output_dir, experiment, arm)[0] for arm in ARMS}
    accuracy = {arm: accuracy_rows(records, ks) for arm, records in rows.items()}
    metrics = {
        f"pass@{k}": (lambda r, key=f"pass@{k}": r[key], lambda r: 1) for k in ks
    }
    result = {
        "experiment_hash": digest(experiment),
        "n_questions": len(experiment["problems"]),
        "n_samples": n,
        "hint_validation": experiment.get(
            "hint_validation", "lexical_only_not_semantically_validated"
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "uncertainty_unit": "question",
        "arms": {},
        "contrasts": {},
    }
    for arm in ARMS:
        result["arms"][arm] = {
            "uncertainty": summarize_uncertainty(rows[arm]),
            "accuracy": paired_metrics(
                accuracy[arm], metrics, args.bootstrap_samples, args.bootstrap_seed
            ),
            "uncertainty_intervals": paired_metrics(
                rows[arm],
                uncertainty_metrics(),
                args.bootstrap_samples,
                args.bootstrap_seed,
            ),
        }
    for left, right in contrasts():
        result["contrasts"][f"{left}-minus-{right}"] = {
            "accuracy": paired_metrics(
                accuracy[left],
                metrics,
                args.bootstrap_samples,
                args.bootstrap_seed,
                baseline=accuracy[right],
            ),
            "uncertainty": paired_metrics(
                rows[left],
                uncertainty_metrics(),
                args.bootstrap_samples,
                args.bootstrap_seed,
                baseline=rows[right],
            ),
        }
    write_json(Path(args.output_dir) / "summary.json", result)
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=["prepare", "generate", "summarize", "sweep"],
        default="prepare",
    )
    p.add_argument("--hints-dir")
    p.add_argument(
        "--hint-review",
        help="Complete text-hashed review JSON; optional lexical false-positive/negative adjudication",
    )
    p.add_argument("--output-dir", required=True)
    for key, value in DEFAULTS.items():
        p.add_argument("--" + key.replace("_", "-"), type=type(value))
    p.add_argument("--k", nargs="+", type=int)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if (
        args.batch_size < 1
        or args.bootstrap_samples < 1
        or not 0 < args.gpu_memory_utilization <= 1
    ):
        raise ValueError("Invalid batch size, bootstrap count, or GPU memory setting")
    experiment = prepare(args)
    if args.dry_run or args.phase == "prepare":
        return
    if args.phase == "generate":
        generate(args, experiment)
    elif args.phase == "summarize":
        summarize(args, experiment)
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.teacher_hint_variants",
                "--phase",
                "generate",
                "--output-dir",
                args.output_dir,
                "--batch-size",
                str(args.batch_size),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ],
            check=True,
        )
        summarize(args, experiment)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.teacher_hint_variant_behaviors",
                "--phase",
                "sweep",
                "--output-dir",
                args.output_dir,
                "--bootstrap-samples",
                str(args.bootstrap_samples),
                "--bootstrap-seed",
                str(args.bootstrap_seed),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.viz.teacher_hint_variants",
                "--output-dir",
                args.output_dir,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
