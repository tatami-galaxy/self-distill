"""Create an immutable DeepMath selection set outside declared training pools.

Exclusion is conservative: it covers every eligible training question, not just
the unknown subset consumed before early stopping. Historical datasets/caches
must still reflect the data used by the runs; old metadata does not hash them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def question_key(question: str) -> str:
    # Exclude repeated questions even when reference answers differ.
    return fingerprint(" ".join(question.split()))


def training_questions(
    run_dir: Path, source=None, *, allow_missing_cache=False
) -> tuple[set[str], dict]:
    """Reconstruct a conservative training pool from this repository's metadata."""
    from datasets import load_from_disk

    from utils import hint_path, load_train_dataset

    run_dir = Path(run_dir).resolve()
    meta = read_json(run_dir / "run_meta.json")
    required = {"model", "dataset", "max_samples", "num_train_examples"}
    if not required.issubset(meta):
        raise ValueError(f"Incomplete training provenance in {run_dir}/run_meta.json")
    method = meta.get("method")
    supported = {
        "grpo",
        "ppo_vllm",
        "ppo_pi_vllm",
        "ppo_val_vllm",
        "gold_opd_vllm",
        "sft",
        "online_sac_pi",
    }
    if method not in supported and not (method is None and "pi_mode" in meta):
        raise ValueError(f"Unsupported training method {method!r} in {run_dir}")
    if meta["dataset"] != "deepmath":
        raise ValueError(
            "This exclusion audit currently supports DeepMath training runs only"
        )
    limit = meta["max_samples"]
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("Invalid max_samples in training metadata")
    mode = meta.get("pi_mode")
    cache_path = None
    if mode in {"hint", "rollout"}:
        generator = (
            meta.get("hint_generator_model") or meta.get("gen_model") or meta["model"]
        )
        if mode == "rollout":
            generator = meta["model"]
        cache_path = meta.get("hint_cache") or hint_path(generator, "deepmath")
        try:
            cache = load_from_disk(cache_path)
        except FileNotFoundError:
            if not allow_missing_cache:
                raise
            print(f"Unverified training overlap: missing hint cache {cache_path}")
            return set(), {
                "run_dir": str(run_dir),
                "run_meta": meta,
                "hint_cache": cache_path,
                "excluded_questions": 0,
                "question_fingerprint": None,
                "policy": "missing_hint_cache_overlap_unverified",
            }
        if set(cache.unique("gen_model")) != {generator}:
            raise ValueError(f"Hint generator provenance mismatch in {cache_path}")
        if "dataset" in cache.column_names and set(cache.unique("dataset")) != {
            "deepmath"
        }:
            raise ValueError(f"Dataset mismatch in {cache_path}")
        rows = cache.select_columns(["question"])
    else:
        rows = source if source is not None else load_train_dataset("deepmath")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    if len(rows) < meta["num_train_examples"]:
        raise ValueError(
            f"Current training source is shorter than recorded in {run_dir}"
        )
    keys = {question_key(row["question"]) for row in rows}
    return keys, {
        "run_dir": str(run_dir),
        "run_meta": meta,
        "hint_cache": cache_path,
        "excluded_questions": len(keys),
        "question_fingerprint": fingerprint(sorted(keys)),
        "policy": "all_eligible_questions_before_length_filtering",
    }


def sample_problems(rows, excluded: set[str], count: int, seed: int) -> list[dict]:
    if count < 1:
        raise ValueError("num_problems must be positive")
    eligible = {}
    for row in rows:
        key = question_key(row["question"])
        if key not in excluded:
            eligible.setdefault(
                key,
                {
                    "question_id": key,
                    "problem": row["question"],
                    "answer": str(row["final_answer"]),
                },
            )
    if len(eligible) < count:
        raise ValueError(
            f"Only {len(eligible)} questions remain after training exclusions; need {count}. "
            "A run configured on all DeepMath cannot be retrospectively held out from "
            "metadata alone. Use a separately reserved pool or an audited record of consumed data."
        )
    # Sort by stable identity: source row order does not affect the random split.
    chosen = random.Random(seed).sample(sorted(eligible), count)
    return [eligible[key] for key in chosen]


def load_validation(path) -> dict:
    split = read_json(path)
    if split.get("schema_version") != 1 or split.get("dataset") != "deepmath":
        raise ValueError("Unsupported validation manifest")
    problems = split["problems"]
    if not problems or split["fingerprint"] != fingerprint(problems):
        raise ValueError("Validation manifest fingerprint mismatch or empty split")
    keys = [question_key(row["problem"]) for row in problems]
    if len(set(keys)) != len(keys) or keys != [row["question_id"] for row in problems]:
        raise ValueError("Validation question identities are invalid or duplicated")
    return split


def audit_run(split, run_dir, *, allow_training_overlap=False) -> dict:
    run_dir = Path(run_dir).resolve()
    allowed = any(
        row["run_dir"] == str(run_dir)
        and row["policy"] == "missing_hint_cache_overlap_unverified"
        for row in split.get("exclusions", [])
    )
    keys, audit = training_questions(run_dir, allow_missing_cache=allowed)
    overlap = keys.intersection(row["question_id"] for row in split["problems"])
    if overlap and not allow_training_overlap:
        raise ValueError(
            f"Validation overlaps {len(overlap)} eligible training questions in {run_dir}. "
            "Use --allow-training-overlap to select on this fixed set while recording "
            "the overlap, or use a held-out split. Eligible does not mean actually consumed."
        )
    if overlap:
        audit["validation_overlap"] = {
            "eligible_question_count": len(overlap),
            "validation_question_count": len(split["problems"]),
            "question_ids": sorted(overlap),
            "actual_training_exposure": "unknown",
            "allowed": True,
        }
        print(
            f"Using fixed selection set with {len(overlap)}/{len(split['problems'])} "
            "questions in the eligible training pool; actual training exposure is unknown."
        )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/validation/deepmath_128.json")
    parser.add_argument("--num-problems", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-cache",
        action="store_true",
        help="Record missing historical hint caches as unverified overlap and continue",
    )
    parser.add_argument(
        "--exclude-run",
        nargs="+",
        required=True,
        help="Exact run directories whose training pools must be excluded",
    )
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Split already exists; reuse it or choose a new output path")
    from utils import load_train_dataset

    source = load_train_dataset("deepmath")
    excluded, audits = set(), []
    for directory in sorted(set(args.exclude_run)):
        keys, audit = training_questions(
            Path(directory), source, allow_missing_cache=args.allow_missing_cache
        )
        excluded.update(keys)
        audits.append(audit)
    problems = sample_problems(source, excluded, args.num_problems, args.seed)
    write_json(
        args.output,
        {
            "schema_version": 1,
            "dataset": "deepmath",
            "seed": args.seed,
            "num_problems": len(problems),
            "fingerprint": fingerprint(problems),
            "source_fingerprint": getattr(source, "_fingerprint", None),
            "exclusions": audits,
            "exclusion_status": "partial"
            if any(
                a["policy"] == "missing_hint_cache_overlap_unverified" for a in audits
            )
            else "complete_for_declared_runs",
            "problems": problems,
            "caveat": "Historical exclusions assume current source/cache contents match training. "
            "This manifest does not modify training loaders or exclude pretraining exposure.",
        },
    )
    print(f"Saved {len(problems)} fixed validation questions -> {args.output}")


if __name__ == "__main__":
    main()
