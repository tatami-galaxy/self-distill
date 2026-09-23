"""Plot a completed demonstration-gain run (CPU only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.demo_gain import read_json, read_rows
from eval.hint_compare_cache import digest


def plot(output_dir):
    root = Path(output_dir)
    summary = read_json(root / "summary.json")
    if summary["score_index_hash"] != digest(read_json(root / "score_index.json")):
        raise ValueError("Summary is stale; rerun aggregate")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    arms = list(summary["conditions"])
    colors = dict(zip(arms, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for ax, metric, label in zip(
        axes,
        ("total_gain", "normalized_gain"),
        ("Total gain (nats/demo)", "Normalized gain (nats/token)"),
        strict=True,
    ):
        for i, arm in enumerate(arms):
            value = summary["conditions"][arm][metric]
            ax.bar(i, value["mean"], color=colors[arm])
            ax.vlines(i, *value["ci95"], color="black")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(range(len(arms)), arms, rotation=30, ha="right")
        ax.set_ylabel(label)
    fig.suptitle(f"{summary['model']} — {summary['n_questions']} paired questions")
    fig.savefig(figures / "gain.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    bins = summary["position_bins"]
    for ax, field, x in (
        (axes[0], "position_gain", (np.arange(bins) + 0.5) / bins * 100),
        (axes[1], "cumulative_gain", np.linspace(0, 100, bins + 1)),
    ):
        for arm in arms:
            values = summary["conditions"][arm][field]
            ci = np.asarray(values["ci95"], dtype=float)
            ax.plot(x, values["mean"], label=arm, color=colors[arm])
            ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=0.12, color=colors[arm])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("Demonstration position (%)")
    axes[0].set_ylabel("Mean token gain (nats/token)")
    axes[1].set_ylabel("Cumulative gain (nats/demo)")
    axes[1].legend(fontsize=8)
    fig.savefig(figures / "position_gain.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for arm in arms:
        values = summary["conditions"][arm]["early_gain"]
        x = np.arange(1, len(values["mean"]) + 1)
        ci = np.asarray(values["ci95"], dtype=float)
        axes[0].plot(
            x,
            np.asarray(values["mean"], dtype=float),
            label=arm,
            color=colors[arm],
            alpha=0.8,
        )
        axes[0].fill_between(x, ci[:, 0], ci[:, 1], alpha=0.08, color=colors[arm])
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_ylabel("Token gain (nats)")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, values["n_questions"], color="black")
    axes[1].set_ylabel("Contributing questions")
    axes[1].set_xlabel("Demonstration token position")
    fig.savefig(figures / "early_gain.png", dpi=160)
    plt.close(fig)
    rows = [
        r
        for r in read_rows(root / "per_question.jsonl")
        if r["hint_tokens"] is not None
    ]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for arm in arms:
            selected = [r for r in rows if r["condition"] == arm]
            if selected:
                ax.scatter(
                    [r["hint_tokens"] for r in selected],
                    [r["normalized_gain"] for r in selected],
                    label=arm,
                    color=colors[arm],
                    alpha=0.5,
                    s=15,
                )
        ax.set_xlabel("Generated hint tokens")
        ax.set_ylabel("Demonstration gain (nats/token)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.legend()
        fig.savefig(figures / "hint_length_gain.png", dpi=160)
        plt.close(fig)
    print(f"Saved plots -> {figures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    plot(parser.parse_args().output_dir)
