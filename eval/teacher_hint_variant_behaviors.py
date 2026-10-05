"""Judge the hint-level study's saved responses using the original cognitive rubric."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from eval.hint_compare_cache import ConditionCache, digest
from eval.teacher_hint_common import (
    ARMS,
    contrasts,
    freeze,
    load_rows,
    paired_metrics,
    read_json,
    save_rows,
    write_json,
)
from eval.teacher_hint_variants import load_completions, load_experiment


def cognitive_metrics():
    from eval.teacher_behaviors import BEHAVIORS

    metrics = {}
    for behavior in BEHAVIORS:
        metrics[f"{behavior}/rate_per_1k"] = (
            lambda r, b=behavior: 1000 * r[b],
            lambda r: r["n_tokens"],
        )
        metrics[f"{behavior}/mean_per_trajectory"] = (
            lambda r, b=behavior: r[b],
            lambda r: 1,
        )
        metrics[f"{behavior}/prevalence"] = (
            lambda r, b=behavior: r[b] > 0,
            lambda r: 1,
        )
    return metrics


def validate_chunks(chunks, expected_counts):
    expected = {
        (key, i) for key, count in expected_counts.items() for i in range(count)
    }
    actual = [
        (f"{r['question_idx']}:{r['sample_idx']}", r["chunk_idx"]) for r in chunks
    ]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing, duplicated, or unexpected judge chunks")


def classify(args, experiment):
    from transformers import AutoConfig, AutoTokenizer

    from eval import teacher_behaviors as tb
    from eval.demo_gain import tokenizer_hash

    root = Path(args.output_dir)
    config_path = root / "judge_config.json"
    previous = read_json(config_path) if config_path.exists() else None
    model = args.judge_model or (previous["model"] if previous else "Qwen/Qwen3.8-27B")
    requested_revision = args.judge_revision or (
        previous["revision"] if previous else None
    )
    revision = (
        getattr(
            AutoConfig.from_pretrained(
                model, revision=requested_revision, trust_remote_code=True
            ),
            "_commit_hash",
            None,
        )
        or requested_revision
    )
    if not revision:
        raise ValueError("Judge revision must be pinned")
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, trust_remote_code=True
    )
    n = args.samples_per_problem
    if n is None:
        n = (
            previous["samples_per_problem"]
            if previous
            else min(4, experiment["generation"]["n"])
        )
    if not 1 <= n <= experiment["generation"]["n"]:
        raise ValueError("Judge sample count must be between 1 and generated n")
    config = {
        "model": model,
        "revision": revision,
        "tokenizer_hash": tokenizer_hash(tokenizer),
        "rubric_fingerprint": tb.rubric_fingerprint(),
        "samples_per_problem": n,
        "chunk_tokens": 1000,
        "context_paragraphs": 0,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_output_tokens": 1024,
        "max_model_len": 16384,
        "seed": 42,
        "evidence": False,
        "vllm_version": __import__("vllm").__version__,
        "experiment_hash": digest(experiment),
    }
    if previous is not None and previous != config:
        raise ValueError(
            "Judge configuration changed; use a new study output directory"
        )
    system = tb.render_system_prompt()
    plans = {}
    for arm in ARMS:
        source, _ = load_completions(root, experiment, arm)
        source = [r for r in source if r["sample_idx"] < n]
        plan = tb.build_chunk_plan(
            source, tokenizer, config["chunk_tokens"], config["context_paragraphs"]
        )
        # Never silently truncate a rubric/segment prompt.
        for chunk in plan:
            ids = tokenizer.apply_chat_template(
                tb.build_messages(system, chunk["text"], chunk.get("context")),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(ids) + config["max_output_tokens"] > config["max_model_len"]:
                raise ValueError("Judge prompt exceeds context budget")
        plans[arm] = (source, plan)
    print(
        f"Judge: {sum(len(plan) for _, plan in plans.values())} chunks across {len(ARMS)} arms",
        flush=True,
    )
    if args.dry_run:
        return
    freeze(config_path, config)
    cache = ConditionCache(root, "behaviors", config)
    llm, params = None, None
    for arm, (source, plan) in plans.items():
        source_hash = digest(source)
        rows, pending = [], []
        for chunk in plan:
            key = cache.key({"arm": arm, "source_hash": source_hash, "chunk": chunk})
            saved = cache.load(key)
            if saved is None:
                pending.append((chunk, key))
            else:
                rows.append(saved)
        if pending and llm is None:
            judge_args = SimpleNamespace(
                classifier_model=model,
                revision=revision,
                max_model_len=config["max_model_len"],
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=1,
                enable_prefix_caching=True,
                max_num_seqs=256,
                seed=config["seed"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_output_tokens=config["max_output_tokens"],
                evidence=False,
            )
            llm, params = tb.create_classifier(judge_args)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            scored = tb.classify_chunks(
                llm, params, [item[0] for item in batch], system, False
            )
            for (_, key), row in zip(batch, scored, strict=True):
                cache.save(key, row)
                rows.append(row)
            print(
                f"{arm}: {min(start + args.batch_size, len(pending))}/{len(pending)} uncached chunks",
                flush=True,
            )
        rows.sort(
            key=lambda row: (row["question_idx"], row["sample_idx"], row["chunk_idx"])
        )
        counts = dict(
            Counter(f"{row['question_idx']}:{row['sample_idx']}" for row in plan)
        )
        validate_chunks(rows, counts)
        artifact_config = {
            "judge": config,
            "source_hash": source_hash,
            "chunk_counts": counts,
        }
        save_rows(root / arm, "behaviors", rows, artifact_config)


def summarize(args, experiment):
    from eval import teacher_behaviors as tb

    root = Path(args.output_dir)
    judge = read_json(root / "judge_config.json")
    if (
        judge["experiment_hash"] != digest(experiment)
        or judge["rubric_fingerprint"] != tb.rubric_fingerprint()
    ):
        raise ValueError("Judge experiment or rubric changed")
    n = judge["samples_per_problem"]
    if args.samples_per_problem is not None and args.samples_per_problem != n:
        raise ValueError("Summary sample count differs from saved judgments")
    trajectories = {}
    result = {
        "experiment_hash": digest(experiment),
        "hint_validation": experiment.get(
            "hint_validation", "lexical_only_not_semantically_validated"
        ),
        "judge": judge,
        "arms": {},
        "contrasts": {},
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "uncertainty_unit": "question; contrasts require complete sample groups in both arms",
    }
    for arm in ARMS:
        source, _ = load_completions(root, experiment, arm)
        source = [r for r in source if r["sample_idx"] < n]
        chunks, meta = load_rows(root / arm, "behaviors")
        if meta["config"]["judge"] != judge or meta["config"]["source_hash"] != digest(
            source
        ):
            raise ValueError("Judgment source or configuration changed")
        if set(meta["config"]["chunk_counts"]) != {
            tb.trajectory_key(r) for r in source
        }:
            raise ValueError("Missing trajectory from chunk plan")
        validate_chunks(chunks, meta["config"]["chunk_counts"])
        collapsed = tb.collapse_to_trajectories(chunks, source)
        trajectories[arm] = collapsed
        summary = tb.summarize_arm(
            collapsed, chunks, args.bootstrap_samples, args.bootstrap_seed
        )
        summary["n_source_trajectories"] = len(source)
        summary["n_trajectories_dropped"] = len(source) - len(collapsed)
        result["arms"][arm] = {
            "cognitive": summary,
            "intervals": paired_metrics(
                collapsed,
                cognitive_metrics(),
                args.bootstrap_samples,
                args.bootstrap_seed,
            ),
        }
        save_rows(
            root / arm,
            "trajectories",
            collapsed,
            {"behaviors_hash": meta["hash"], "judge": judge},
        )
    for left, right in contrasts():
        result["contrasts"][f"{left}-minus-{right}"] = paired_metrics(
            trajectories[left],
            cognitive_metrics(),
            args.bootstrap_samples,
            args.bootstrap_seed,
            baseline=trajectories[right],
            complete_n=n,
        )
    write_json(root / "behaviors_summary.json", result)
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--phase", choices=["classify", "summarize", "sweep"], default="sweep"
    )
    p.add_argument("--samples-per-problem", type=int)
    p.add_argument("--judge-model")
    p.add_argument("--judge-revision")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
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
    experiment = load_experiment(args.output_dir)
    if args.phase != "summarize":
        classify(args, experiment)
    if not args.dry_run and args.phase != "classify":
        summarize(args, experiment)


if __name__ == "__main__":
    main()
