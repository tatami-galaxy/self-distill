"""Read the exact retained self-teacher cohort, preserving its original PI."""

import json
from pathlib import Path

from eval.hint_compare_cache import digest


def resolve_cohort_dir(recorded_dir, cohort_dir=None):
    """Resolve recorded results paths from the current project root.

    Older manifests store absolute paths from the machine that prepared them.
    Reuse their results/... suffix in this checkout, even if the old path still
    exists. Explicit overrides and absolute paths outside results are left intact.
    """
    if cohort_dir is not None:
        return Path(cohort_dir)
    root = Path(recorded_dir)
    if root.is_absolute() and "results" in root.parts:
        root = Path(*root.parts[root.parts.index("results"):])
    return root


def load_teacher_cohort(study_dir, cohort_dir=None):
    """Validate frozen sources and recover retained rows in teacher-study order.

    Rows include the cached hint and full reference solution. Consumers generating
    unprivileged responses must explicitly select only the question and answer.
    """
    study_dir = Path(study_dir).resolve()
    meta = json.loads((study_dir / "teacher_uncertainty_run_meta.json").read_text())
    source = meta["source"]
    recorded = source.get("cohort")
    if not recorded:
        raise ValueError("Teacher study needs recorded cohort provenance and question IDs")
    root = resolve_cohort_dir(recorded["cohort_dir"], cohort_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    with (root / "cohort.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if (
        digest(rows) != manifest["cohort_hash"]
        or manifest["cohort_hash"] != recorded["cohort_hash"]
    ):
        raise ValueError("Teacher source cohort checksum mismatch")
    if (
        manifest["model"] != source["problem_model"]
        or meta["teacher_model"] != source["problem_model"]
    ):
        raise ValueError("Use a self-teacher study for the student's base model")
    if manifest["tokenizer_hash"] != recorded["tokenizer_hash"]:
        raise ValueError("Teacher source tokenizer mismatch")
    ids, indices = source["question_ids"], source["question_indices"]
    if (
        not ids or len(ids) != len(set(ids))
        or len(indices) != len(set(indices)) or len(ids) != len(indices)
    ):
        raise ValueError("Teacher question identities must be nonempty and unique")
    by_id = {r["question_id"]: r for r in rows}
    if len(by_id) != len(rows) or len(ids) != meta["n_problems"]:
        raise ValueError("Invalid teacher cohort size or duplicate source identities")
    selected = []
    for qid, idx in zip(ids, indices, strict=True):
        row = by_id.get(qid)
        if row is None or row["question_idx"] != idx:
            raise ValueError(f"Teacher question missing or misaligned: {qid}")
        selected.append({**row, "dataset": manifest["dataset"]})
    return selected, meta
