"""Plot the separate hint-level study; reads summaries without loading models."""

import argparse
import csv
from pathlib import Path

from eval.teacher_hint_common import ARMS, read_json


def plot(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    summary = read_json(root / "summary.json")
    behaviors = (
        read_json(root / "behaviors_summary.json")
        if (root / "behaviors_summary.json").exists()
        else None
    )
    if behaviors and behaviors["experiment_hash"] != summary["experiment_hash"]:
        raise ValueError(
            "Generation and cognitive summaries describe different experiments"
        )
    accuracy_metric = next(iter(summary["arms"]["none"]["accuracy"]["metrics"]))
    panels = [
        ("accuracy", accuracy_metric, f"Accuracy ({accuracy_metric})"),
        ("uncertainty_intervals", "n_tokens", "Mean response tokens"),
        (
            "uncertainty_intervals",
            "e_per_1k_tokens",
            "Uncertainty markers / 1,000 tokens",
        ),
        ("uncertainty_intervals", "truncated", "Truncated fraction"),
    ]
    if behaviors:
        panels += [
            (
                "cognitive",
                f"{name}/rate_per_1k",
                name.replace("_", " ").capitalize() + " / 1,000 tokens",
            )
            for name in (
                "verification",
                "backtracking",
                "subgoal_setting",
                "backward_chaining",
            )
        ]
    fig, axes = plt.subplots(
        2 if behaviors else 1, 4, figsize=(18, 8 if behaviors else 4), squeeze=False
    )
    table = []
    for axis, (section, metric, title) in zip(axes.flat, panels, strict=True):
        for i, arm in enumerate(ARMS):
            estimates = (
                behaviors["arms"][arm]["intervals"]
                if section == "cognitive"
                else summary["arms"][arm][section]
            )["metrics"]
            if metric not in estimates:
                axis.text(i, 0, "No usable data", rotation=90, fontsize=8)
                continue
            item = estimates[metric]
            low, high = item["ci95"]
            axis.bar(
                i, item["mean"], color=["#777777", "#67a9cf", "#2166ac", "#b2182b"][i]
            )
            axis.vlines(i, low, high, color="black", linewidth=1.2)
            axis.hlines([low, high], i - 0.06, i + 0.06, color="black", linewidth=1.2)
            table.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "mean": item["mean"],
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        axis.set_xticks(range(4), ["None", "Short", "Medium", "Detailed"], rotation=25)
        axis.set_title(title, fontsize=10)
    fig.suptitle(
        f"Self-teacher hint levels: {summary['n_questions']} questions, {summary['n_samples']} responses/arm/question"
        + (
            "\nLexical hint filter only; answer secrecy not validated"
            if summary.get("hint_validation") != "reviewed"
            else "\nReviewed hint selection"
        )
    )
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    with (root / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["arm", "metric", "mean", "ci95_low", "ci95_high"]
        )
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    plot(parser.parse_args().output_dir)
