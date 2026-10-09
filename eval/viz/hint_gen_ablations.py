"""Plot a joint hint_gen_compare summary: attributes, validity, duals, and contrasts.

CPU only. See docs/hint_gen_ablations.md for training and evaluation commands.
Intervals resample questions, not training seeds. Run one figure per training seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PANELS = (
    ("training_aligned_sufficiency", "Sufficiency (invalid hints = 0)"),
    ("normalized_cost", "Length cost (tokens / budget)"),
    ("raw_transfer_nats_per_token", "Transfer cost (nats/token)"),
    ("invalid_fraction", "Invalid hint fraction"),
    ("truncated_fraction", "Truncated hint fraction"),
)


def default_arm_specs(summary):
    specs = ["full=full", "no_transfer=no_transfer", "no_length=no_length"]
    if any(label.startswith("sufficiency_only-step") for label in summary["generator_order"]):
        specs.append("sufficiency_only=sufficiency_only")
    return specs


def collect_series(summary, arm_specs):
    if "pass@1" not in summary["no_hint_self_teacher"]:
        raise ValueError("Ablation plots require an evaluation with --k 1 (optionally other k values)")
    arms = {}
    for spec in arm_specs:
        name, sep, prefix = spec.partition("=")
        if not sep or not name or not prefix or name in arms:
            raise ValueError("Use unique --arm NAME=PREFIX entries")
        pattern = re.compile(re.escape(prefix) + r"-step([1-9]\d*)")
        points = sorted(
            (int(match[1]), label)
            for label in summary["generator_order"]
            if (match := pattern.fullmatch(label))
        )
        if not points:
            raise ValueError(f"No checkpoints labeled {prefix}-stepN")
        arms[name] = [(0, "fresh_base"), *points]
    steps = [tuple(step for step, _ in points) for points in arms.values()]
    if len(arms) < 2:
        raise ValueError("Specify at least two ablation arms")
    if any(values != steps[0] for values in steps):
        raise ValueError("All arms must contain the same evaluated steps")
    used = [label for points in arms.values() for step, label in points if step]
    if len(set(used)) != len(used):
        raise ValueError("Each arm must use distinct checkpoint labels")
    metas = [summary["generator_runs"][label]["run_meta"] for label in used]
    for key in ("model", "dataset", "tau", "hint_budget", "seed"):
        if any(meta.get(key) != metas[0].get(key) for meta in metas):
            raise ValueError(f"Matched ablation arms disagree on {key}")
    if any(meta.get("method") != "constrained_hint_gen_grpo" for meta in metas):
        raise ValueError("Ablation plots require constrained hint-generator runs")
    if summary["hint_budget"] != metas[0]["hint_budget"]:
        raise ValueError("Evaluation hint budget must match training for these ablation plots")
    return arms


def contrast_series(summary, arms, reference):
    if reference not in arms:
        raise ValueError(f"Unknown reference arm {reference!r}")
    reference_points = dict(arms[reference])
    result = {}
    for name, points in arms.items():
        if name == reference:
            continue
        rows = []
        for step, label in points:
            if step == 0:
                continue
            key = f"{label}_minus_{reference_points[step]}"
            if key not in summary["paired_comparisons"]:
                raise ValueError(
                    f"Missing {key}; summarize with --compare {label}={reference_points[step]}"
                )
            rows.append((step, summary["paired_comparisons"][key]["metrics"]))
        result[name] = rows
    return result


def plot(summary, arms, reference, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contrasts = contrast_series(summary, arms, reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(arms, plt.get_cmap("tab10").colors, strict=False))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    exports = []
    for ax, (metric, title) in zip(axes.flat, PANELS):
        for name, points in arms.items():
            estimates = [summary["generators"][label]["metric_estimates"][metric]
                         for _, label in points]
            xs = [step for step, _ in points]
            ax.plot(xs, [row["mean"] for row in estimates], "o-", color=colors[name], label=name)
            ax.fill_between(xs, [row["ci95"][0] for row in estimates],
                            [row["ci95"][1] for row in estimates], color=colors[name], alpha=0.12)
            if metric == "raw_transfer_nats_per_token":
                ax.plot(xs, [summary["generators"][label]["metric_estimates"]
                             ["clamped_transfer_nats_per_token"]["mean"] for _, label in points],
                        "--", color=colors[name], alpha=0.7)
            for (step, label), row in zip(points, estimates, strict=True):
                exports.append({"arm": name, "step": step, "generator": label,
                                "metric": metric, "mean": row["mean"],
                                "ci_low": row["ci95"][0], "ci_high": row["ci95"][1]})
        ax.set(title=title, xlabel="Optimizer step")
        ax.grid(alpha=0.2)
    metas = summary["generator_runs"]
    first_label = arms[reference][1][1]
    tau = metas[first_label]["run_meta"]["tau"]
    axes[0, 0].axhline(tau, color="black", linestyle="--", label=f"Target τ={tau:g}")
    baseline = summary["no_hint_self_teacher"]["pass@1"]
    axes[0, 0].axhline(baseline, color="gray", linestyle=":", label="No hint")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].secondary_yaxis(
        "right", functions=(lambda cost: cost * summary["hint_budget"],
                            lambda tokens: tokens / summary["hint_budget"])
    ).set_ylabel("Hint tokens")
    axes[0, 2].text(0.03, 0.97, "Dashed: clamped estimate", transform=axes[0, 2].transAxes,
                    va="top", fontsize=8)

    dual_ax = axes[1, 2]
    missing_duals = []
    for name, points in arms.items():
        meta = metas[points[1][1]]["run_meta"]
        xs, ys = [0], [meta["dual_init"]]
        for step, label in points[1:]:
            state_path = Path(metas[label]["checkpoint"]) / "constrained_reward_state.json"
            xs.append(step)
            if state_path.is_file():
                ys.append(json.loads(state_path.read_text())["dual_lambda"])
            else:
                ys.append(float("nan"))
                missing_duals.append(label)
        dual_ax.plot(xs, ys, "o-", color=colors[name], label=name)
    dual_ax.set(title="Sufficiency multiplier λ", xlabel="Optimizer step")
    dual_ax.grid(alpha=0.2)
    if missing_duals:
        dual_ax.text(0.03, 0.97, "Some checkpoint states unavailable", transform=dual_ax.transAxes,
                     va="top", fontsize=8)
        print("Missing dual states:", ", ".join(missing_duals))
    fig.suptitle(f"{summary['model']} — hint objective ablations\n"
                 "Pointwise 95% question bootstrap intervals; one training seed")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"attributes.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (metric, title) in zip(axes, PANELS[:3], strict=True):
        for name, points in contrasts.items():
            xs = [0, *[step for step, _ in points]]
            rows = [values[metric] for _, values in points]
            ax.plot(xs, [0, *[row["mean_delta_candidate_minus_reference"] for row in rows]],
                    "o-", color=colors[name], label=f"{name} − {reference}")
            ax.fill_between(xs, [0, *[row["delta_ci95"][0] for row in rows]],
                            [0, *[row["delta_ci95"][1] for row in rows]],
                            color=colors[name], alpha=0.12)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=f"Δ {title}", xlabel="Optimizer step")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"differences.{extension}", dpi=180)
    plt.close(fig)
    with (output_dir / "attributes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exports[0]))
        writer.writeheader()
        writer.writerows(exports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--arm", action="append", metavar="NAME=PREFIX",
                        help="Labels in the summary must be PREFIX-stepN. Repeat per arm.")
    parser.add_argument("--reference", default="full")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    specs = args.arm or default_arm_specs(summary)
    arms = collect_series(summary, specs)
    plot(summary, arms, args.reference, args.output_dir or args.summary.parent / "ablations")


if __name__ == "__main__":
    main()
