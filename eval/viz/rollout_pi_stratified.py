"""Rollout PI stratified by whether the cached rollout was itself correct.

Reads `paired_rollout_minus_none` from results/passk_pi_{budget}/*/passk_pi_summary.json.
That block pairs each problem's `none` and `rollout` results and splits them by
`attempt_correct`, the reward on the cached rollout. Reward is used only as a post-hoc
label -- rollout selection is `fixed_sample_idx_without_reward` -- so the split does not
leak into the prompts.

Each row is one model, drawn as an arrow from its `none` accuracy to its `rollout`
accuracy. Levels matter as much as deltas here: the gains in the correct stratum are
small because `none` is already at ceiling, which only reads if the ceiling is visible.

Usage:
    .venv/bin/python results/viz/rollout_pi_stratified.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# -------------------- paths --------------------


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).parent).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "results").is_dir() and (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repository root containing results/ and train/")


ROOT = find_repo_root()
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"  # generated output; not tracked by git
OUT = FIGURES / "rollout_pi_stratified.png"

# -------------------- style --------------------

CANVAS = "#FBF9F4"
CARD = "#FFFFFF"
INK = "#191917"
MUTED = "#6E6C66"
GRID = "#E8E5DE"

SANS = FontProperties(family=["Carlito", "DejaVu Sans"])

MODEL_STYLE = {
    "Qwen3-1.7B": {"color": "#3372DA", "label": "Qwen3-1.7B"},
    "Qwen3-4B": {"color": "#E1603C", "label": "Qwen3-4B"},
}
MODEL_ORDER = ["Qwen3-1.7B", "Qwen3-4B"]

BUDGETS = ["8k", "16k"]
KS = [1, 8]
STRATA = [
    ("attempt_correct", "Rollout correct"),
    ("attempt_incorrect", "Rollout incorrect"),
]

# Row positions, top group then bottom group, with a gap between strata.
ROW_Y = {
    ("attempt_correct", "Qwen3-1.7B"): 3.55,
    ("attempt_correct", "Qwen3-4B"): 2.75,
    ("attempt_incorrect", "Qwen3-1.7B"): 1.20,
    ("attempt_incorrect", "Qwen3-4B"): 0.40,
}
HEADER_Y = {"attempt_correct": 4.22, "attempt_incorrect": 1.87}

XLIM = (0, 152)
DELTA_X = 106  # left edge of the annotation column


# -------------------- data --------------------


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_stratified() -> dict:
    """rows[budget][k][stratum][model] -> none, rollout, delta, ci, n."""
    rows: dict = {}
    for budget in BUDGETS:
        paths = sorted(RESULTS.glob(f"passk_pi_{budget}/*/passk_pi_summary.json"))
        if not paths:
            raise FileNotFoundError(f"No summaries under results/passk_pi_{budget}/")
        rows[budget] = {k: {stratum: {} for stratum, _ in STRATA} for k in KS}
        for path in paths:
            summary = read_json(path)
            model = summary["model"].split("/")[-1]
            paired = summary.get("paired_rollout_minus_none")
            if paired is None:
                raise KeyError(f"{path} has no paired_rollout_minus_none block")
            for stratum, _ in STRATA:
                group = paired[stratum]
                for k in KS:
                    metrics = group["pass_at_k"][f"pass@{k}"]
                    rows[budget][k][stratum][model] = {
                        "none": metrics["none"] * 100,
                        "rollout": metrics["rollout"] * 100,
                        "delta": metrics["delta"] * 100,
                        "ci": [value * 100 for value in metrics["delta_ci95"]],
                        "n": group["n_problems"],
                    }
    return rows


# -------------------- plot --------------------


def draw_panel(ax, panel: dict, show_delta_header: bool) -> None:
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.35, 4.75)

    for x in (0, 25, 50, 75, 100):
        ax.axvline(x, color=GRID, linewidth=1.1, zorder=0)

    for stratum, stratum_label in STRATA:
        n_values = [panel[stratum][model]["n"] for model in MODEL_ORDER if model in panel[stratum]]
        n_text = " / ".join(str(value) for value in n_values)
        ax.text(
            1.5, HEADER_Y[stratum], f"{stratum_label}   n = {n_text}",
            ha="left", va="center", fontproperties=SANS, fontsize=11.5,
            fontweight="bold", color=MUTED,
        )

        for model in MODEL_ORDER:
            if model not in panel[stratum]:
                continue
            entry = panel[stratum][model]
            colour = MODEL_STYLE[model]["color"]
            y = ROW_Y[(stratum, model)]

            ax.annotate(
                "",
                xy=(entry["rollout"], y), xytext=(entry["none"], y),
                arrowprops={
                    "arrowstyle": "-|>,head_length=0.75,head_width=0.32",
                    "color": colour, "linewidth": 2.6, "shrinkA": 4, "shrinkB": 0,
                },
            )
            ax.plot(
                entry["none"], y, "o", color=CARD, markersize=10.5,
                markeredgecolor=colour, markeredgewidth=2.4, zorder=4,
            )

            significant = entry["ci"][0] > 0 or entry["ci"][1] < 0
            ax.text(
                DELTA_X, y, f"{entry['delta']:+.1f}",
                ha="left", va="center", fontproperties=SANS, fontsize=12.5,
                fontweight="bold" if significant else "normal",
                color=colour if significant else MUTED,
            )
            ax.text(
                DELTA_X + 15, y, f"[{entry['ci'][0]:+.1f}, {entry['ci'][1]:+.1f}]",
                ha="left", va="center", fontproperties=SANS, fontsize=10.5, color=MUTED,
            )

    if show_delta_header:
        ax.text(
            DELTA_X, 4.62, "Δ (95% CI)",
            ha="left", va="center", fontproperties=SANS, fontsize=11,
            fontweight="bold", color=INK,
        )

    ax.set_yticks([ROW_Y[(stratum, model)] for stratum, _ in STRATA for model in MODEL_ORDER])
    ax.set_yticklabels(
        [MODEL_STYLE[model]["label"] for _ in STRATA for model in MODEL_ORDER]
    )
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", length=0, pad=7, labelsize=12.5, colors=INK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(SANS)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.set_axisbelow(True)


def build_figure(rows: dict):
    fig = plt.figure(figsize=(16.0, 9.6), dpi=200, facecolor=CANVAS)
    card = FancyBboxPatch(
        (0.010, 0.015), 0.980, 0.970,
        boxstyle="round,pad=0,rounding_size=0.016",
        transform=fig.transFigure, facecolor=CARD, edgecolor="none", zorder=-10,
    )
    fig.patches.append(card)

    fig.text(
        0.5, 0.955, "Rollout PI correct vs incorrect",
        ha="center", va="center", fontproperties=SANS, fontsize=25,
        fontweight="bold", color=INK,
    )

    left, width, gap = 0.088, 0.415, 0.055
    bottom, height, vgap = 0.155, 0.310, 0.095
    for row_index, budget in enumerate(BUDGETS):
        for col_index, k in enumerate(KS):
            ax = fig.add_axes([
                left + col_index * (width + gap),
                bottom + (len(BUDGETS) - 1 - row_index) * (height + vgap),
                width, height,
            ], zorder=5)
            ax.set_facecolor("none")
            draw_panel(ax, rows[budget][k], show_delta_header=(row_index == 0))
            ax.set_title(
                f"pass@{k}   ·   {budget} budget",
                fontproperties=SANS, fontsize=16, fontweight="bold",
                color=INK, pad=26, loc="left", x=0.0,
            )

    fig.text(
        0.5, 0.100, "Accuracy (%) — open circle: no PI,  arrowhead: rollout PI",
        ha="center", va="center", fontproperties=SANS, fontsize=14,
        fontweight="bold", color=INK,
    )

    handles = [
        Line2D([], [], color=MODEL_STYLE[model]["color"], linewidth=2.8,
               marker="o", markerfacecolor=CARD, markeredgewidth=2.2,
               markeredgecolor=MODEL_STYLE[model]["color"], markersize=9,
               label=MODEL_STYLE[model]["label"])
        for model in MODEL_ORDER
    ]
    legend = fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.018),
        frameon=True, prop=FontProperties(family=["Carlito", "DejaVu Sans"], size=13),
        handletextpad=0.7, borderpad=0.7, labelspacing=0.5, ncol=2, columnspacing=1.6,
    )
    frame = legend.get_frame()
    frame.set_facecolor("#FDFCF9")
    frame.set_edgecolor("#DCD8CE")
    frame.set_linewidth(1.1)
    frame.set_boxstyle("round,pad=0.4,rounding_size=0.12")
    for text in legend.get_texts():
        text.set_color(INK)

    return fig


if __name__ == "__main__":
    mpl.rcParams["savefig.facecolor"] = CANVAS
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_stratified()
    for budget in BUDGETS:
        print(f"=== {budget}")
        for k in KS:
            for stratum, label in STRATA:
                for model, entry in rows[budget][k][stratum].items():
                    print(
                        f"  pass@{k} {label:18s} {model:11s} n={entry['n']:3d} "
                        f"{entry['none']:5.1f} -> {entry['rollout']:5.1f}  "
                        f"delta={entry['delta']:+6.2f} "
                        f"CI=[{entry['ci'][0]:+6.2f},{entry['ci'][1]:+6.2f}]"
                    )
    figure = build_figure(rows)
    figure.savefig(OUT, dpi=200, facecolor=CANVAS)
    print(f"\nWrote {OUT.relative_to(ROOT)}")
