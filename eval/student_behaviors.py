r"""Measure cognitive behaviors in cached, unhinted SDFT checkpoint rollouts.

No solver is loaded or sampled. Only the judge generates text. Reuses the exact
rubric, segmentation, judge configuration and counting rules of teacher_behaviors.
Defaults: both Qwen3 sizes, original answer/hint/full/rollout arms, every cached
step (including 0), and sample_idx < 4. Learned-hint runs are intentionally excluded.
The question intersection is fixed across all selected arms and checkpoints WITHIN
each model. Models have separate cohorts and are not paired with each other.

# Plan all checkpoints; tokenize segments and estimate judge work, without loading a GPU model.
uv run python -m eval.student_behaviors --dry-run

# Classify both models, reusing completed checkpoint classifications on rerun.
CUDA_VISIBLE_DEVICES=0 uv run python -m eval.student_behaviors

# A smaller pilot (step 0 is always included for paired comparisons).
CUDA_VISIBLE_DEVICES=0 uv run python -m eval.student_behaviors \
    --models Qwen3-1.7B --steps 40 --max-questions 8 \
    --output-root results/student_behaviors_8k_pilot

# Recompute summaries/curves from classifications, without loading a judge or tokenizer.
uv run python -m eval.student_behaviors --phase summarize

# Matched-budget 8k teacher references: classify EXISTING teacher completions.
# Run this command once per model (also use --teacher-model Qwen/Qwen3-4B).
CUDA_VISIBLE_DEVICES=0 uv run python -m eval.teacher_behaviors \
    --teacher-model Qwen/Qwen3-1.7B --samples-per-problem 4 \
    --completions-root results/teacher_uncertainty_8k \
    --output-root results/teacher_behaviors_8k

The teacher and student cohorts do not overlap in the current archive. Teacher
results are separate UNPAIRED descriptive references, not paired student controls.
The default student caches use an 8192-token budget. Truncation is retained and
reported; per-token normalization does not remove censoring or length effects.

Outputs: <output-root>/<model>/<run>/step-XXXXXX/{behaviors.jsonl,meta.json,
trajectories.jsonl,summary.json}; <output-root>/<model>/summary.json and PNG curves.
Each checkpoint is committed atomically and can be resumed independently. An
interrupted checkpoint is reclassified. Provenance includes source content,
rollout configuration, question identities and rubric/judge settings. --force
replaces classifications with incompatible provenance. Changing bootstrap count
only recomputes summaries. Failed chunks drop their whole trajectory, as in the
teacher analysis. Paired deltas retain questions with all requested samples usable
in BOTH conditions, and bootstrap questions jointly (never pair sample indices).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from eval import teacher_behaviors as tb
from eval.teacher_uncertainty import count_epistemic, split_think

SCHEMA_VERSION = 1
DEFAULT_MODELS = ("Qwen3-1.7B", "Qwen3-4B")
DEFAULT_ARMS = ("answer", "hint", "full", "rollout")
GENERATION_KEYS = (
    "base_model",
    "dataset",
    "max_completion_length",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "seed",
)
JUDGE_KEYS = (
    "classifier_model",
    "chunk_tokens",
    "context_paragraphs",
    "evidence",
    "temperature",
    "top_p",
    "max_output_tokens",
    "seed",
    "max_model_len",
)


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_model(args, model: str) -> tuple[list[dict], dict]:
    """Intersect stored cohorts, validate generation settings, and inventory caches."""
    runs, identities = {}, {}
    for arm in args.arms:
        root = Path(args.rollout_root) / model / f"deepmath_{arm}"
        meta = tb.read_json(root / "cohort_meta.json")
        if meta.get("status") != "complete":
            raise ValueError(f"Incomplete cohort: {root}")
        cohort = load_from_disk(str(root / "cohort"))
        questions = {}
        for row in cohort:
            qid = row["question_id"]
            identity = (
                int(row["question_idx"]),
                row["question"],
                str(row["final_answer"]),
            )
            if qid in questions or (qid in identities and identities[qid] != identity):
                raise ValueError(
                    f"Duplicate or inconsistent question identity in {root}: {qid}"
                )
            questions[qid] = identity
            identities[qid] = identity
        if list(cohort["question_id"]) != meta["question_ids"]:
            raise ValueError(f"Cohort IDs differ from metadata: {root}")
        if tb.fingerprint_ids(cohort["question_id"]) != meta["cohort_fingerprint"]:
            raise ValueError(f"Cohort fingerprint mismatch: {root}")
        runs[arm] = (root, meta, questions)
    common = set.intersection(*(set(value[2]) for value in runs.values()))
    common = sorted(common, key=lambda q: (identities[q][0], q))
    if args.max_questions is not None:
        common = common[: args.max_questions]
    if not common:
        raise ValueError(f"No common questions for {model}")
    selected = {qid: identities[qid] for qid in common}
    if len({value[0] for value in selected.values()}) != len(selected):
        raise ValueError("Question indices must uniquely identify the common cohort")

    jobs, generation = [], None
    for arm, (root, cohort_meta, _) in runs.items():
        steps = {
            int(p.name[5:]): p
            for p in root.glob("step-*")
            if p.is_dir() and p.name[5:].isdigit()
        }
        wanted = sorted(set(args.steps or steps) | {0})
        missing = set(wanted) - set(steps)
        if missing:
            raise ValueError(f"Missing cached steps {sorted(missing)} in {root}")
        for step in wanted:
            source = steps[step]
            meta = tb.read_json(source / "rollout_meta.json")
            config = meta["config"]
            if meta.get("status") != "complete" or not (source / "rollouts").is_dir():
                raise ValueError(f"Incomplete rollout cache: {source}")
            if config["step"] != step or config["base_model"].split("/")[-1] != model:
                raise ValueError(f"Wrong checkpoint/model identity: {source}")
            if config["cohort_fingerprint"] != cohort_meta["cohort_fingerprint"]:
                raise ValueError(f"Rollout cohort mismatch: {source}")
            if config["n"] < args.samples_per_problem:
                raise ValueError(f"Insufficient cached samples: {source}")
            signature = {key: config[key] for key in GENERATION_KEYS}
            if generation is not None and signature != generation:
                raise ValueError(
                    f"Generation settings differ across checkpoints/arms: {source}"
                )
            generation = signature
            jobs.append(
                {
                    "model": model,
                    "arm": arm,
                    "step": step,
                    "source": source,
                    "generation": config,
                    "questions": selected,
                    "output": Path(args.output_root) / model / root.name / source.name,
                }
            )
    manifest = {
        "model": model,
        "question_ids": common,
        "n_questions": len(common),
        "question_fingerprint": fingerprint(selected),
        "generation": generation,
        "arms": list(args.arms),
        "samples_per_problem": args.samples_per_problem,
    }
    return jobs, manifest


def adapt_rollouts(rows, questions: dict, samples_per_problem: int) -> list[dict]:
    """Convert cached student records to the classifier's source schema without grading."""
    selected, seen = [], set()
    for row in rows:
        qid, sample = row["question_id"], int(row["sample_idx"])
        if qid not in questions or not 0 <= sample < samples_per_problem:
            continue
        identity = (int(row["question_idx"]), row["question"], str(row["final_answer"]))
        if identity != questions[qid]:
            raise ValueError(f"Rollout question identity mismatch: {qid}")
        key = (qid, sample)
        if key in seen:
            raise ValueError(f"Duplicate rollout: {key}")
        seen.add(key)
        n_tokens = int(row["n_tokens"])
        if n_tokens < 1 or len(row["completion_ids"]) != n_tokens:
            raise ValueError(f"Invalid completion token count: {key}")
        text = row["completion_text"]
        think, _, closed = split_think(text)
        if float(row["reward"]) not in (0.0, 1.0):
            raise ValueError(f"Expected binary cached correctness: {key}")
        selected.append(
            {
                "question_id": qid,
                "question_idx": identity[0],
                "sample_idx": sample,
                "rollout_id": row["rollout_id"],
                "text": text,
                "n_tokens": n_tokens,
                "token_fingerprint": fingerprint(row["completion_ids"]),
                "correct": bool(row["reward"]),
                "truncated": bool(row["truncated"]),
                "finish_reason": row["finish_reason"],
                "unclosed": not closed,
                "e_think": sum(count_epistemic(think).values()),
                "e_total": sum(count_epistemic(text).values()),
            }
        )
    expected = {
        (qid, sample) for qid in questions for sample in range(samples_per_problem)
    }
    if seen != expected:
        raise ValueError(
            f"Missing {len(expected - seen)} requested question/sample pairs"
        )
    return sorted(selected, key=lambda row: (row["question_idx"], row["sample_idx"]))


def load_source_rows(job: dict, samples_per_problem: int) -> list[dict]:
    """Select via small metadata columns before decoding long token/text columns."""
    dataset = load_from_disk(str(job["source"] / "rollouts"))
    indices = [
        i
        for i, (qid, sample) in enumerate(
            zip(dataset["question_id"], dataset["sample_idx"], strict=True)
        )
        if qid in job["questions"] and 0 <= sample < samples_per_problem
    ]
    return adapt_rollouts(
        dataset.select(indices), job["questions"], samples_per_problem
    )


def classification_config(args, job: dict, source_rows: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "student_cognitive_behaviors",
        "model": job["model"],
        "arm": job["arm"],
        "step": job["step"],
        "source": str(job["source"].resolve()),
        "generation": job["generation"],
        "questions": fingerprint(job["questions"]),
        "source_fingerprint": fingerprint(source_rows),
        "samples_per_problem": args.samples_per_problem,
        "rubric_version": tb.BEHAVIOR_RUBRIC_VERSION,
        "rubric_fingerprint": tb.rubric_fingerprint(),
        "judge": {key: getattr(args, key) for key in JUDGE_KEYS},
    }


def cached_classification(output: Path, config: dict, force: bool) -> list[dict] | None:
    if force:
        return None
    meta_path, rows_path = output / "meta.json", output / "behaviors.jsonl"
    if not meta_path.is_file():
        return None  # Interrupted checkpoints have no committed metadata.
    meta = tb.read_json(meta_path)
    if meta.get("config") != config:
        raise ValueError(
            f"Incompatible classification cache: {output}. Use a new output root or --force."
        )
    if meta.get("status") != "complete" or not rows_path.is_file():
        raise ValueError(f"Incomplete classification cache: {output}; use --force")
    rows = read_jsonl(rows_path)
    if fingerprint(rows) != meta.get("chunks_fingerprint"):
        raise ValueError(f"Classification content mismatch: {output}; use --force")
    return rows


def paired_differences(
    candidate: list[dict],
    baseline: list[dict],
    samples: int,
    seed: int,
    samples_per_problem: int,
) -> dict:
    """Joint question bootstrap of pooled-metric differences, with complete sample groups."""

    def group(rows):
        out = defaultdict(list)
        for row in rows:
            out[row["question_idx"]].append(row)
        expected = set(range(samples_per_problem))
        return {
            q: rs
            for q, rs in out.items()
            if len(rs) == samples_per_problem
            and {r["sample_idx"] for r in rs} == expected
        }

    a, b = group(candidate), group(baseline)
    common = sorted(a.keys() & b.keys())
    result = {
        "sign": "checkpoint_minus_step_zero",
        "n_questions": len(common),
        "question_indices": common,
        "uncertainty_unit": "paired_question_bootstrap",
        "selection": "all_requested_samples_classified_in_both_conditions",
        "metrics": {},
    }
    if not common:
        return result
    metrics = {
        "pass@1": (lambda r: r["correct"], lambda r: 1),
        "mean_tokens": (lambda r: r["n_tokens"], lambda r: 1),
        "truncation_rate": (lambda r: r["truncated"], lambda r: 1),
    }
    for behavior in tb.BEHAVIORS:
        metrics[f"{behavior}/rate_per_1k"] = (
            lambda r, name=behavior: 1000 * r[name],
            lambda r: r["n_tokens"],
        )
        metrics[f"{behavior}/mean_per_trajectory"] = (
            lambda r, name=behavior: r[name],
            lambda r: 1,
        )
        metrics[f"{behavior}/prevalence"] = (
            lambda r, name=behavior: r[name] > 0,
            lambda r: 1,
        )
    rng = np.random.default_rng(seed)
    # Same bootstrap question draws for both conditions and every metric.
    draws = rng.integers(len(common), size=(samples, len(common)))
    for name, (numerator, denominator) in metrics.items():
        totals = []
        for groups in (a, b):
            totals.append(
                np.array(
                    [
                        [
                            sum(numerator(r) for r in groups[q]),
                            sum(denominator(r) for r in groups[q]),
                        ]
                        for q in common
                    ],
                    dtype=float,
                )
            )
        ta, tb_ = totals
        value = ta[:, 0].sum() / ta[:, 1].sum() - tb_[:, 0].sum() / tb_[:, 1].sum()
        sa, sb = ta[draws].sum(axis=1), tb_[draws].sum(axis=1)
        deltas = sa[:, 0] / sa[:, 1] - sb[:, 0] / sb[:, 1]
        result["metrics"][name] = {
            "delta": float(value),
            "ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        }
    return result


def summarize_checkpoint(source_rows, chunks, args):
    trajectories = tb.collapse_to_trajectories(chunks, source_rows)
    source_by_key = {tb.trajectory_key(row): row for row in source_rows}
    for row in trajectories:
        original = source_by_key[tb.trajectory_key(row)]
        row.update(
            question_id=original["question_id"], rollout_id=original["rollout_id"]
        )
    summary = tb.summarize_arm(trajectories, chunks, args.bootstrap_samples, args.seed)
    summary["n_source_trajectories"] = len(source_rows)
    summary["n_trajectories_dropped"] = len(source_rows) - len(trajectories)
    # Keep generation diagnostics over ALL selected completions, even if judging failed.
    summary["source_diagnostics"] = {
        "pass@1": sum(r["correct"] for r in source_rows) / len(source_rows),
        "mean_tokens": sum(r["n_tokens"] for r in source_rows) / len(source_rows),
        "truncation_rate": sum(r["truncated"] for r in source_rows) / len(source_rows),
        "unclosed_rate": sum(r["unclosed"] for r in source_rows) / len(source_rows),
    }
    return trajectories, summary


def plot_curves(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ("rate_per_1k", "mean_per_trajectory", "prevalence"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, behavior in zip(axes.flat, tb.BEHAVIORS, strict=True):
            for arm, steps in summary["arms"].items():
                points = sorted(
                    (int(step), item["summary"][behavior])
                    for step, item in steps.items()
                    if behavior in item["summary"]
                )
                if not points:
                    continue
                x, records = zip(*points)
                (line,) = ax.plot(
                    x, [r[metric] for r in records], marker=".", label=arm
                )
                ci = f"{metric}_ci95"
                if ci in records[0]:
                    ax.fill_between(
                        x,
                        [r[ci][0] for r in records],
                        [r[ci][1] for r in records],
                        color=line.get_color(),
                        alpha=0.12,
                    )
            ax.set(
                title=behavior.replace("_", " "), xlabel="Training step", ylabel=metric
            )
            ax.grid(alpha=0.2)
        axes.flat[0].legend()
        fig.suptitle(
            f"{summary['model']} · unhinted student · {summary['n_questions']} questions"
        )
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric in zip(
        axes.flat,
        ("pass@1", "mean_tokens", "truncation_rate", "unclosed_rate"),
        strict=True,
    ):
        for arm, steps in summary["arms"].items():
            points = sorted(
                (int(step), item["summary"]["source_diagnostics"][metric])
                for step, item in steps.items()
            )
            ax.plot(
                [p[0] for p in points], [p[1] for p in points], marker=".", label=arm
            )
        ax.set(title=metric, xlabel="Training step")
        ax.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.suptitle(f"{summary['model']} · all selected completions")
    fig.savefig(output / "generation_diagnostics.png", dpi=160)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rollout-root", default="results/advantage_dynamics")
    parser.add_argument("--output-root", default="results/student_behaviors_8k")
    parser.add_argument(
        "--models", nargs="+", choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS)
    )
    parser.add_argument(
        "--arms", nargs="+", choices=DEFAULT_ARMS, default=list(DEFAULT_ARMS)
    )
    parser.add_argument("--steps", nargs="+", type=int, default=None)
    parser.add_argument("--samples-per-problem", type=int, default=4)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit the common cohort for a pilot.",
    )
    parser.add_argument("--phase", choices=("sweep", "summarize"), default="sweep")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    tb.add_classifier_args(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.samples_per_problem < 1
        or args.chunk_tokens < 1
        or args.bootstrap_samples < 1
    ):
        parser.error("Sample, chunk-token, and bootstrap counts must be positive")
    if args.max_questions is not None and args.max_questions < 1:
        parser.error("--max-questions must be positive")
    if args.steps is not None and any(step < 0 for step in args.steps):
        parser.error("--steps must be nonnegative")
    if args.context_paragraphs < 0 or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("Invalid context or judge sampling settings")
    if args.phase == "summarize" and args.force:
        parser.error("--force requires --phase sweep")
    args.models, args.arms = (
        list(dict.fromkeys(args.models)),
        list(dict.fromkeys(args.arms)),
    )
    if args.max_output_tokens is None:
        args.max_output_tokens = 2048 if args.evidence else 1024
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    source_root, output_root = (
        Path(args.rollout_root).resolve(),
        Path(args.output_root).resolve(),
    )
    if output_root.is_relative_to(source_root) or source_root.is_relative_to(
        output_root
    ):
        parser.error("Input and output roots must be separate directory trees")
    if not tb.examples_are_filled() and not args.allow_placeholder_examples:
        parser.error("The classifier rubric still contains placeholder examples")

    # Check all cohort and rollout manifests before any judge allocation.
    prepared = [prepare_model(args, model) for model in args.models]
    tokenizer = None
    llm = sampling_params = None
    system_prompt = tb.render_system_prompt()
    total_chunks = total_tokens = 0
    for jobs, manifest in prepared:
        print(
            f"{manifest['model']}: {manifest['n_questions']} common questions, {len(jobs)} checkpoints"
        )
        results, baselines = defaultdict(dict), {}
        for job in jobs:
            source_rows = load_source_rows(job, args.samples_per_problem)
            config = classification_config(args, job, source_rows)
            chunks = cached_classification(job["output"], config, args.force)
            label = f"{job['model']}/{job['arm']}/step-{job['step']:06d}"
            if args.phase == "summarize" and chunks is None:
                raise FileNotFoundError(
                    f"No classification cache for {label}; run --phase sweep first"
                )
            if chunks is None:
                if tokenizer is None:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        args.classifier_model, trust_remote_code=True
                    )
                plan = tb.build_chunk_plan(
                    source_rows, tokenizer, args.chunk_tokens, args.context_paragraphs
                )
                n_tokens = sum(row["n_classifier_tokens"] for row in plan)
                total_chunks += len(plan)
                total_tokens += n_tokens
                print(
                    f"{label}: {len(source_rows)} trajectories, {len(plan)} judge calls, {n_tokens:,} segment tokens",
                    flush=True,
                )
                if args.dry_run:
                    continue
                if llm is None:
                    llm, sampling_params = tb.create_classifier(args)
                chunks = tb.classify_chunks(
                    llm, sampling_params, plan, system_prompt, args.evidence
                )
                write_jsonl(job["output"] / "behaviors.jsonl", chunks)
                tb.write_json_atomic(
                    job["output"] / "meta.json",
                    {
                        "status": "complete",
                        "config": config,
                        "chunks_fingerprint": fingerprint(chunks),
                    },
                )
                del plan
            else:
                print(f"{label}: reusing {len(chunks)} classified chunks", flush=True)
            if args.dry_run:
                continue
            trajectories, summary = summarize_checkpoint(source_rows, chunks, args)
            if job["step"] == 0:
                baselines[job["arm"]] = trajectories
            paired = paired_differences(
                trajectories,
                baselines[job["arm"]],
                args.bootstrap_samples,
                args.seed,
                args.samples_per_problem,
            )
            item = {"summary": summary, "paired_vs_step_zero": paired}
            write_jsonl(job["output"] / "trajectories.jsonl", trajectories)
            tb.write_json_atomic(
                job["output"] / "summary.json", {"config": config, **item}
            )
            results[job["arm"]][str(job["step"])] = item
        if not args.dry_run:
            summary = {
                "schema_version": SCHEMA_VERSION,
                "method": "student_cognitive_behaviors",
                **manifest,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
                "judge": {key: getattr(args, key) for key in JUDGE_KEYS},
                "rubric_fingerprint": tb.rubric_fingerprint(),
                "arms": dict(results),
                "teacher_comparison": "unpaired; teacher references use a separate cohort",
            }
            output = Path(args.output_root) / manifest["model"]
            tb.write_json_atomic(output / "summary.json", summary)
            if args.plots:
                plot_curves(summary, output)
    if args.dry_run:
        print(
            f"Pending judge work: {total_chunks:,} calls, {total_tokens:,} segment tokens "
            f"(excluding cached rubric/context), at most {total_chunks * args.max_output_tokens:,} output tokens"
        )


if __name__ == "__main__":
    main()
