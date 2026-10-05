"""Artifacts and question-paired statistics for the separate hint-level study."""

from __future__ import annotations

import json
import os
from pathlib import Path

from eval.hint_compare_cache import digest

LEVELS = ("hint_short", "hint_medium", "hint_detailed")
ARMS = ("none", *LEVELS)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def freeze(path, config):
    if Path(path).exists() and read_json(path) != config:
        raise ValueError(
            f"Configuration or sources changed: use a new output directory ({path})"
        )
    write_json(path, config)


def save_rows(root, name, rows, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.jsonl"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    os.replace(temporary, path)
    write_json(
        root / f"{name}_meta.json",
        {"config": config, "hash": digest(rows), "count": len(rows)},
    )


def load_rows(root, name, config=None):
    root = Path(root)
    meta = read_json(root / f"{name}_meta.json")
    rows = [
        json.loads(s)
        for s in (root / f"{name}.jsonl").read_text().splitlines()
        if s.strip()
    ]
    if meta["hash"] != digest(rows) or meta["count"] != len(rows):
        raise ValueError(f"Corrupt {name} artifact in {root}")
    if config is not None and meta["config"] != config:
        raise ValueError(f"Changed {name} provenance in {root}")
    return rows, meta


def request_seed(seed, *identity):
    return int(digest([seed, *identity])[:8], 16) % (2**31)


def validate_completions(rows, problems, n):
    expected = {
        (p["question_idx"], i): p["question_id"] for p in problems for i in range(n)
    }
    actual = {}
    for row in rows:
        key = (row["question_idx"], row["sample_idx"])
        if key in actual:
            raise ValueError("Duplicate completion")
        actual[key] = row["question_id"]
    if actual != expected:
        raise ValueError("Missing, extra, or misaligned completions")


def paired_metrics(
    rows, metrics, samples=10000, seed=42, baseline=None, complete_n=None
):
    """Pooled ratios; joint question resampling for contrasts, never sample pairing.

    metrics maps names to (per-row numerator, per-row denominator) callables.
    For cognitive contrasts require complete sample groups in both arms.
    """
    from collections import defaultdict

    import numpy as np

    def groups(records):
        grouped = defaultdict(list)
        for row in records:
            grouped[row["question_idx"]].append(row)
        if complete_n is not None:
            grouped = {
                q: rs
                for q, rs in grouped.items()
                if len(rs) == complete_n
                and {r["sample_idx"] for r in rs} == set(range(complete_n))
            }
        return grouped

    if samples < 1:
        raise ValueError("Bootstrap count must be positive")
    left = groups(rows)
    right = groups(baseline) if baseline is not None else None
    questions = sorted(set(left) & set(right) if right is not None else left)
    result = {
        "n_questions": len(questions),
        "question_indices": questions,
        "metrics": {},
    }
    if not questions:
        return result
    draws = np.random.default_rng(seed).integers(
        len(questions), size=(samples, len(questions))
    )
    for name, (numerator, denominator) in metrics.items():
        estimates, distributions = [], []
        for group in [left, right] if right is not None else [left]:
            totals = np.array(
                [
                    [
                        sum(numerator(r) for r in group[q]),
                        sum(denominator(r) for r in group[q]),
                    ]
                    for q in questions
                ],
                dtype=float,
            )
            sums = totals[draws].sum(axis=1)
            if totals[:, 1].sum() <= 0 or (sums[:, 1] <= 0).any():
                raise ValueError(f"Zero denominator for {name}")
            estimates.append(totals[:, 0].sum() / totals[:, 1].sum())
            distributions.append(sums[:, 0] / sums[:, 1])
        value = estimates[0] - estimates[1] if right is not None else estimates[0]
        distribution = (
            distributions[0] - distributions[1]
            if right is not None
            else distributions[0]
        )
        result["metrics"][name] = {
            "delta" if right is not None else "mean": float(value),
            "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        }
    return result


def contrasts():
    return [(arm, "none") for arm in LEVELS] + [
        ("hint_detailed", "hint_medium"),
        ("hint_medium", "hint_short"),
    ]
