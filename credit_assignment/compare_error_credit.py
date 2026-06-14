"""Compare OPD vs OPSD per-token credit against the judge's first-error step.

For each incorrect rollout we have, on the *same* completion token indexing:
  - per-token credit (A_t, Abar_t, reweight_t) under OPD and OPSD (one file each),
  - a judge label: the first uncorrected-error step + every step's token span.

The question: does credit (blame) concentrate on the judge's error step, and does
OPSD localize it more sharply than OPD? Raw segment means are not comparable across
methods/rollouts (different teachers => different magnitudes), so the headline is a
**within-rollout gap**: the error step's mean signal minus the mean over the other
think steps of the same rollout/method. We also report:
  - min within the segment (the sharpest "forking" token the mean dilutes),
  - the error step's percentile rank among think steps (scale-free),
  - a random-step placebo gap (control: should be ~0),
  - the preceding step's gap (control for credit firing *before* the visible error).

Streams the 11 GB advantages files once each, reusing the judge's stored step spans
(no re-segmentation), and writes one small row per (rollout, method). Aggregation
(means +/- SE, paired OPD-vs-OPSD bootstrap CIs and Wilcoxon tests) runs off rows.

Run from the repo root, e.g.:
    python -m credit_assignment.compare_error_credit \
        --errors data/credit_assignment/errors_Qwen_Qwen3.6-27B_Qwen_Qwen3-1.7B_deepmath.jsonl \
        --opd        data/credit_assignment/advantages_opd_..._Qwen_Qwen3-1.7B_deepmath.jsonl \
        --opsd-gold  data/credit_assignment/advantages_opsd_gold_solution_revkl_Qwen_Qwen3-1.7B_deepmath.jsonl \
        --opsd-final data/credit_assignment/advantages_opsd_final_answer_revkl_Qwen_Qwen3-1.7B_deepmath.jsonl \
        --rows-out data/credit_assignment/error_credit_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SIGNALS = ("A_t", "Abar_t", "reweight_t")
MIN_THINK_STEPS = 4  # need an error step + preceding + a baseline pool to compare


# --------------------------------------------------------------------- loading


def load_judge_labels(path: Path) -> dict[str, dict]:
    """problem_id -> {err_idx, steps}, keeping only valid in-range first errors."""
    labels: dict[str, dict] = {}
    n_total = n_no_error = n_bad = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_total += 1
            fe = rec.get("first_error_step")
            steps = rec.get("steps") or []
            if fe is None or fe == -1:
                n_no_error += 1
                continue
            if not (0 <= fe < len(steps)):
                n_bad += 1
                continue
            labels[str(rec["problem_id"])] = {"err_idx": int(fe), "steps": steps}
    print(f"[judge] {len(labels)} usable labels "
          f"({n_no_error} no-error/-1, {n_bad} out-of-range, of {n_total})", file=sys.stderr)
    return labels


def stream_matched(path: Path, pids: set[str], max_rollouts: int | None):
    """Yield advantage records whose problem_id is in ``pids`` (streamed)."""
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec["problem_id"]) not in pids:
                continue
            yield rec
            n += 1
            if max_rollouts and n >= max_rollouts:
                return


# ------------------------------------------------------------------ per-rollout


def _step_stats(vals: list, s: int, e: int) -> tuple[float | None, float | None]:
    """(mean, min) over ``vals[s:e]`` ignoring None; (None, None) if empty."""
    xs = [v for v in vals[s:e] if isinstance(v, (int, float))]
    if not xs:
        return None, None
    return float(np.mean(xs)), float(min(xs))


def compute_row(rec: dict, label: dict, rng: np.random.Generator) -> dict | None:
    """One (rollout, method) row: per-signal gap/min/rank + placebo & precedence.

    Baseline for every gap is the mean over *think* steps excluding the error step
    and its predecessor, so the error gap and the preceding-step gap share a clean
    reference. Returns None if there are too few think steps to form a baseline.
    """
    tokens = rec["tokens"]
    steps = label["steps"]
    err = label["err_idx"]
    think = [i for i, st in enumerate(steps) if st.get("region") == "think"]
    if err not in think or len(think) < MIN_THINK_STEPS:
        return None
    prev = err - 1 if (err - 1) in think else None
    pool = [i for i in think if i not in (err, prev)]  # baseline / placebo steps
    if len(pool) < 2:
        return None
    placebo = int(rng.choice(pool))

    out: dict = {
        "problem_id": str(rec["problem_id"]),
        "error_step_idx": err,
        "n_steps": len(steps),
        "n_think_steps": len(think),
        "has_prev": prev is not None,
        "placebo_step_idx": placebo,
    }
    for sig in SIGNALS:
        vals = [t.get(sig) for t in tokens]
        step_mean: dict[int, float | None] = {}
        step_min: dict[int, float | None] = {}
        for i in think:
            m, mn = _step_stats(vals, steps[i]["tok_start"], steps[i]["tok_end"])
            step_mean[i], step_min[i] = m, mn
        err_mean, err_min = step_mean[err], step_min[err]

        pool_means = [step_mean[i] for i in pool if step_mean[i] is not None]
        pool_mins = [step_min[i] for i in pool if step_min[i] is not None]
        base_mean = float(np.mean(pool_means)) if pool_means else None
        base_min = float(np.mean(pool_mins)) if pool_mins else None

        # percentile rank of the error step among all think steps (0 = most negative)
        valid = [step_mean[i] for i in think if step_mean[i] is not None]
        rank = is_most = None
        if err_mean is not None and len(valid) > 1:
            rank = float(sum(v < err_mean for v in valid) / (len(valid) - 1))
            is_most = bool(err_mean == min(valid))

        def gap(a, b):
            return (a - b) if (a is not None and b is not None) else None

        # placebo baseline excludes the placebo step itself
        pl_pool = [step_mean[i] for i in pool if i != placebo and step_mean[i] is not None]
        pl_base = float(np.mean(pl_pool)) if pl_pool else base_mean

        out[sig] = {
            "err_mean": err_mean, "err_min": err_min,
            "base_mean": base_mean, "base_min": base_min,
            "gap": gap(err_mean, base_mean),
            "gap_min": gap(err_min, base_min),
            "err_rank": rank, "is_most_blamed": is_most,
            "prev_gap": gap(step_mean.get(prev), base_mean) if prev is not None else None,
            "placebo_gap": gap(step_mean[placebo], pl_base),
        }
    return out


# -------------------------------------------------------------------- aggregate


def _mean_se(xs: list) -> tuple[float, float, int]:
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), se, a.size


def _bootstrap_paired(diffs: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Mean paired difference + 95% bootstrap CI (resampling rollouts)."""
    if diffs.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = diffs[rng.integers(0, diffs.size, size=(n_boot, diffs.size))].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def aggregate(rows: list[dict], methods: list[str], n_boot: int, seed: int) -> None:
    by_method = {m: [r for r in rows if r["method"] == m] for m in methods}

    print("\n" + "=" * 98)
    print("Per-method localization (mean +/- SE over rollouts). gap<0 => error step "
          "more blamed than baseline.")
    print("=" * 98)
    for m in methods:
        mr = by_method[m]
        print(f"\n### {m}   (n={len(mr)} rollouts)")
        print(f"  {'signal':<11}{'gap':>16}{'gap_min':>16}{'err_rank':>10}"
              f"{'%most':>8}{'placebo':>12}{'prev_gap':>12}")
        for sig in SIGNALS:
            g_m, g_se, _ = _mean_se([r[sig]["gap"] for r in mr])
            gm_m, gm_se, _ = _mean_se([r[sig]["gap_min"] for r in mr])
            rk, _, _ = _mean_se([r[sig]["err_rank"] for r in mr])
            most_vals = [1.0 if r[sig]["is_most_blamed"] else 0.0
                         for r in mr if r[sig]["is_most_blamed"] is not None]
            most = float(np.mean(most_vals)) if most_vals else float("nan")
            pl, _, _ = _mean_se([r[sig]["placebo_gap"] for r in mr])
            pv, _, _ = _mean_se([r[sig]["prev_gap"] for r in mr])
            print(f"  {sig:<11}{g_m:>8.3f}±{g_se:<6.3f}{gm_m:>8.3f}±{gm_se:<6.3f}"
                  f"{rk:>10.3f}{most:>8.2f}{pl:>12.3f}{pv:>12.3f}")

    if len(methods) > 1:
        print("\n" + "=" * 98)
        print("Paired difference in error-step gap vs OPD (negative => that method "
              "blames the error MORE).")
        print("95% bootstrap CI; Wilcoxon signed-rank p (paired, same rollouts).")
        print("=" * 98)
        base = "opd" if "opd" in methods else methods[0]
        base_idx = {r["problem_id"]: r for r in by_method[base]}
        for m in methods:
            if m == base:
                continue
            print(f"\n### {m} - {base}")
            for sig in SIGNALS:
                pairs = [(r[sig]["gap"], base_idx[r["problem_id"]][sig]["gap"])
                         for r in by_method[m]
                         if r["problem_id"] in base_idx
                         and r[sig]["gap"] is not None
                         and base_idx[r["problem_id"]][sig]["gap"] is not None]
                diffs = np.asarray([a - b for a, b in pairs], dtype=float)
                d, lo, hi = _bootstrap_paired(diffs, n_boot, seed)
                try:
                    p = float(wilcoxon(diffs, zero_method="wilcox").pvalue) if diffs.size else float("nan")
                except ValueError:  # all-zero differences
                    p = float("nan")
                flag = "*" if (p < 0.05) else " "
                print(f"  {sig:<11} Δgap={d:>8.3f}  95%CI[{lo:>7.3f},{hi:>7.3f}]  "
                      f"p={p:>8.4g} {flag}  (n={diffs.size})")


# -------------------------------------------------------------------------- cli


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare OPD/OPSD credit vs judge first-error step.")
    p.add_argument("--errors", required=True, help="judge errors JSONL from label_errors.py")
    p.add_argument("--opd", default=None)
    p.add_argument("--opsd-gold", default=None)
    p.add_argument("--opsd-final", default=None)
    p.add_argument("--rows-out", default=None, help="per-(rollout,method) rows JSONL")
    p.add_argument("--rows-in", default=None, help="skip streaming; aggregate these rows instead")
    p.add_argument("--max-rollouts", type=int, default=None, help="cap per method (quick test)")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.rows_in:
        rows = [json.loads(l) for l in Path(args.rows_in).open() if l.strip()]
        methods = list(dict.fromkeys(r["method"] for r in rows))
        aggregate(rows, methods, args.n_boot, args.seed)
        return

    labels = load_judge_labels(Path(args.errors))
    pids = set(labels)
    method_paths = [(m, p) for m, p in
                    (("opd", args.opd), ("opsd_gold", args.opsd_gold),
                     ("opsd_final", args.opsd_final)) if p]
    if not method_paths:
        raise SystemExit("Provide at least one of --opd/--opsd-gold/--opsd-final.")

    rows: list[dict] = []
    for method, path in method_paths:
        n_in = n_skip = 0
        for rec in stream_matched(Path(path), pids, args.max_rollouts):
            # deterministic per-rollout placebo, independent of method/order
            r_rng = np.random.default_rng(abs(hash(str(rec["problem_id"]))) % (2**32) + args.seed)
            row = compute_row(rec, labels[str(rec["problem_id"])], r_rng)
            if row is None:
                n_skip += 1
                continue
            row["method"] = method
            rows.append(row)
            n_in += 1
        print(f"[{method}] {n_in} rows ({n_skip} skipped: too few think steps)", file=sys.stderr)

    if args.rows_out:
        with Path(args.rows_out).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows -> {args.rows_out}", file=sys.stderr)

    aggregate(rows, [m for m, _ in method_paths], args.n_boot, args.seed)


if __name__ == "__main__":
    main()
