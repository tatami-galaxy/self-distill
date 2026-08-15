"""Cognitive-behavior rates across the PI ladder.

Reads results/teacher_behaviors_16k/<teacher>/behaviors_summary.json, which counts the
four Gandhi et al. (arXiv:2503.01307) behaviors in self-teacher completions under each
privileged context.

Plots `rate_per_1k` — the length-normalized figure, the only one that separates
"the teacher got shorter" from "the teacher reasons differently". Error bars are the
recorded 95% question-clustered bootstrap CIs.

Usage:
    .venv/bin/python eval/viz/teacher_behaviors_bars.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
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
BEHAVIORS_DIR = RESULTS / "teacher_behaviors_16k"
FIGURES = RESULTS / "figures"  # generated output; not tracked by git
OUT = FIGURES / "teacher_behaviors_rate_per_1k.png"

# -------------------- style --------------------

CANVAS = "#FBF9F4"
CARD = "#FFFFFF"
INK = "#191917"
LABEL = "#33312E"
MUTED = "#6E6C66"
GRID = "#E8E5DE"

SANS = FontProperties(family=["Carlito", "DejaVu Sans"])
SERIF = FontProperties(family=["Caladea", "DejaVu Serif"])

MODELS = [("Qwen_Qwen3-1.7B", "Qwen3-1.7B"), ("Qwen_Qwen3-4B", "Qwen3-4B")]

# Ordered along the PI ladder. Colours match the other figures; `none` takes the neutral
# grey that the untrained baseline uses in sdft_passk_bars.py.
PI_MODES = [
    ("none", "none", "#A3A099"),
    ("hint", "hint", "#3D74D0"),
    ("answer", "answer", "#E9B02E"),
    ("rollout", "rollout", "#E0673C"),
    ("full", "full", "#49B083"),
]

BEHAVIORS = [
    ("verification", "verification"),
    ("backtracking", "backtracking"),
    ("subgoal_setting", "subgoal setting"),
    ("backward_chaining", "backward chaining"),
]


# -------------------- data --------------------


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_rates() -> dict:
    """rates[model][behavior][pi] -> (value, ci_low, ci_high)."""
    rates: dict = {}
    for slug, label in MODELS:
        path = BEHAVIORS_DIR / slug / "behaviors_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"No behavior summary at {path}")
        behavior = read_json(path)["behavior"]
        rates[label] = {}
        for key, _ in BEHAVIORS:
            rates[label][key] = {}
            for pi_mode, _, _ in PI_MODES:
                entry = behavior[pi_mode][key]
                low, high = entry["rate_per_1k_ci95"]
                rates[label][key][pi_mode] = (entry["rate_per_1k"], low, high)
    return rates


# -------------------- plot --------------------


def draw_panel(ax, label: str, panel: dict) -> None:
    x = np.arange(len(BEHAVIORS), dtype=float)
    width = 0.8 / len(PI_MODES)

    for index, (pi_mode, _, colour) in enumerate(PI_MODES):
        offset = (index - (len(PI_MODES) - 1) / 2) * width
        values = [panel[key][pi_mode][0] for key, _ in BEHAVIORS]
        lower = [value - panel[key][pi_mode][1] for value, (key, _) in zip(values, BEHAVIORS)]
        upper = [panel[key][pi_mode][2] - value for value, (key, _) in zip(values, BEHAVIORS)]
        bars = ax.bar(
            x + offset, values, width=width * 0.86, color=colour,
            edgecolor=CARD, linewidth=0.7, zorder=3,
            yerr=[lower, upper], ecolor="#57544E", capsize=2.0,
            error_kw={"linewidth": 1.0, "zorder": 5},
        )
        for bar, value, error in zip(bars, values, upper, strict=True):
            text = ax.text(
                bar.get_x() + bar.get_width() / 2, value + error + 0.035,
                f"{value:.2f}", ha="center", va="bottom",
                fontproperties=SANS, fontsize=7.6, color=LABEL, zorder=6,
            )
            text.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground=CARD)])

    ax.set_xticks(x, [name for _, name in BEHAVIORS])
    ax.set_ylim(0, 2.35)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.grid(axis="y", color=GRID, linewidth=1.1, zorder=0)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.8)

    ax.tick_params(axis="both", length=0, pad=8, labelsize=13, colors=INK)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(SANS)
    ax.set_title(label, fontproperties=SANS, fontsize=17, fontweight="bold", color=INK, pad=12)
    ax.set_ylabel(
        "Occurrences per 1k tokens", fontproperties=SANS,
        fontsize=13, fontweight="bold", color=INK, labelpad=10,
    )


def build_figure(rates: dict):
    fig = plt.figure(figsize=(13.0, 11.0), dpi=200, facecolor=CANVAS)
    fig.patches.append(FancyBboxPatch(
        (0.012, 0.016), 0.976, 0.968,
        boxstyle="round,pad=0,rounding_size=0.016",
        transform=fig.transFigure, facecolor=CARD, edgecolor="none", zorder=-10,
    ))
    fig.text(
        0.5, 0.960, "Cognitive behaviors of the self-teacher under each PI",
        ha="center", va="center", fontproperties=SANS, fontsize=24,
        fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.921, "Length-normalized counts of four cognitive behaviors",
        ha="center", va="center", fontproperties=SERIF, fontsize=15, color="#4A4844",
    )

    left, width = 0.092, 0.870
    bottom, height, vgap = 0.140, 0.305, 0.100
    for index, (_, label) in enumerate(MODELS):
        ax = fig.add_axes([
            left, bottom + (len(MODELS) - 1 - index) * (height + vgap), width, height,
        ], zorder=5)
        ax.set_facecolor("none")
        draw_panel(ax, label, rates[label])

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor=CARD, linewidth=0.7)
        for _, _, colour in PI_MODES
    ]
    legend = fig.legend(
        handles, [name for _, name, _ in PI_MODES],
        loc="lower center", bbox_to_anchor=(0.5, 0.020), frameon=True, ncol=len(PI_MODES),
        prop=FontProperties(family=["Carlito", "DejaVu Sans"], size=14),
        handletextpad=0.7, borderpad=0.8, columnspacing=1.9, handlelength=1.5,
    )
    frame = legend.get_frame()
    frame.set_facecolor("#FDFCF9")
    frame.set_edgecolor("#DCD8CE")
    frame.set_linewidth(1.1)
    frame.set_boxstyle("round,pad=0.45,rounding_size=0.12")
    for text in legend.get_texts():
        text.set_color(INK)

    return fig


if __name__ == "__main__":
    mpl.rcParams["savefig.facecolor"] = CANVAS
    FIGURES.mkdir(parents=True, exist_ok=True)
    rates = load_rates()
    for _, label in MODELS:
        print(f"=== {label}   (rate per 1k tokens)")
        header = "  ".join(f"{name:>10s}" for name, _, _ in PI_MODES)
        print(f"  {'behavior':18s} {header}")
        for key, name in BEHAVIORS:
            row = "  ".join(f"{rates[label][key][pi][0]:>10.3f}" for pi, _, _ in PI_MODES)
            print(f"  {name:18s} {row}")
    figure = build_figure(rates)
    figure.savefig(OUT, dpi=200, facecolor=CANVAS)
    print(f"\nWrote {OUT.relative_to(ROOT)}")
