"""Match initial within-question length/transfer reward variation (CPU only)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev


def calibrate(hints, transfer, hint_budget, transfer_rollouts):
    """Equal question weight; valid hints only; clamp after averaging rollouts."""
    if hint_budget <= 0 or transfer_rollouts < 1:
        raise ValueError("Hint budget and transfer rollout count must be positive")
    scores = defaultdict(dict)
    for row in transfer:
        if row["generator_id"] != "fresh_base":
            continue
        values = scores[row["hint_id"]]
        position = row["rollout_position"]
        if position in values:
            raise ValueError("Duplicate transfer rollout")
        value = float(row["raw_transfer"])
        if not math.isfinite(value):
            raise ValueError("Non-finite transfer score")
        values[position] = value
    groups = defaultdict(list)
    records = []
    invalid = 0
    seen = set()
    for hint in hints:
        if hint["generator_id"] != "fresh_base":
            raise ValueError("Calibration requires fresh_base hints only")
        hid = hint["hint_id"]
        if hid in seen:
            raise ValueError("Duplicate hint ID")
        seen.add(hid)
        if not hint["validity_observable"]:
            raise ValueError("Hint validity must be observable")
        if hint["invalid_reason"]:
            invalid += 1
            continue
        values = scores[hid]
        if set(values) != set(range(transfer_rollouts)):
            raise ValueError(f"Missing or unexpected transfer rollouts for {hid}")
        cost = min(hint["num_hint_tokens"] / hint_budget, 1.0)
        raw = fmean(values.values())
        groups[hint["question_id"]].append((cost, max(0.0, raw)))
        records.append((hint["question_id"], hid, cost, sorted(values.items())))
    eligible = [values for values in groups.values() if len(values) >= 2]
    if not eligible:
        raise ValueError("Need at least two valid hints per calibration question")
    length_std = fmean(pstdev(c for c, _ in values) for values in eligible)
    transfer_std = fmean(pstdev(t for _, t in values) for values in eligible)
    if min(length_std, transfer_std) <= 0:
        raise ValueError("Both penalties must have positive within-question variation")
    return {
        "method": "mean_within_question_population_std_valid_fresh_base",
        "length_cost_coef": 1.0,
        "gamma": length_std / transfer_std,
        "mean_within_question_std_cost": length_std,
        "mean_within_question_std_transfer": transfer_std,
        "eligible_questions": len(eligible),
        "excluded_questions": len({h["question_id"] for h in hints}) - len(eligible),
        "valid_hints": len(records),
        "invalid_hints": invalid,
        "input_sha256": hashlib.sha256(json.dumps(sorted(records)).encode()).hexdigest(),
    }


def from_results(root):
    from datasets import load_from_disk

    def read(relative):
        meta = json.loads((root / relative).read_text())
        if meta["status"] != "complete":
            raise ValueError(f"Incomplete {relative}")
        return meta

    cohort = read("cohort_meta.json")
    hints = read("hints/fresh_base_meta.json")
    transfer = read("transfer_meta.json")
    hc, tc = hints["config"], transfer["config"]
    model = cohort["config"]["model"]
    if not (model == hc["generator_model"] == hc["base_model"]
            == tc["student_and_teacher_model"]):
        raise ValueError("Calibration models disagree")
    if hc.get("generator_adapter_path") is not None:
        raise ValueError("Calibration must use the untrained base generator")
    if not (cohort["cohort_fingerprint"] == hc["cohort_fingerprint"] == tc["cohort_fingerprint"]):
        raise ValueError("Calibration cohorts disagree")
    if tc["transfer_max_completion_tokens"] != 0:
        raise ValueError("Use full student trajectories, matching training")
    result = calibrate(
        list(load_from_disk(str(root / "hints/fresh_base"))),
        load_from_disk(str(root / "transfer")),
        hc["hint_max_tokens"], tc["transfer_rollouts"],
    )
    result.update({
        "model": model, "dataset": hc["dataset"], "hint_budget": hc["hint_max_tokens"],
        "transfer_rollouts": tc["transfer_rollouts"], "clamp_transfer": True,
        "source": str(root), "cohort_fingerprint": cohort["cohort_fingerprint"],
        "hint_fingerprint": hints["hint_fingerprint"],
        "selected_rollout_fingerprint": tc["selected_rollout_fingerprint"],
        "hints_per_problem": hc["hints_per_problem"], "generator_temperature": hc["temperature"],
        "generator_top_p": hc["top_p"], "calibration_seed": hc["seed"],
    })
    return result


def load_gamma(path, model):
    """Validate a frozen calibration against the matched training recipe."""
    value = json.loads(path.read_text())
    expected = {
        "model": model, "dataset": "deepmath", "hint_budget": 128,
        "transfer_rollouts": 4, "clamp_transfer": True, "length_cost_coef": 1.0,
        "generator_temperature": 1.0, "generator_top_p": 1.0,
        "method": "mean_within_question_population_std_valid_fresh_base",
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f"Calibration {key} must be {wanted!r}")
    gamma = float(value["gamma"])
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Calibration gamma must be finite and positive")
    return gamma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-dir", type=Path)
    source.add_argument("--calibration", type=Path, help="Validate a frozen file and print gamma")
    parser.add_argument("--model", help="Required when reading a frozen calibration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.calibration:
        if not args.model or args.output:
            parser.error("--calibration requires --model and does not take --output")
        print(load_gamma(args.calibration, args.model))
    else:
        result = from_results(args.results_dir)
        if args.model and result["model"] != args.model:
            parser.error("Results model differs from --model")
        encoded = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")


if __name__ == "__main__":
    main()
