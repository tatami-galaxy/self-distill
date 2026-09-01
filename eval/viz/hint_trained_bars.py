"""SDFT with generator-written hints, against base and the original hint cache.

Three arms on Qwen3-1.7B:

    Base                the untrained model
    SDFT hint           distilled on the legacy hint cache
    SDFT hint-trained   distilled on hints from the reward-trained generator
                        (alpha=1, gamma=1, generator checkpoint-100)

For each (arm, k) the best value across every checkpoint is taken, chosen independently
per k -- so pass@1 and pass@16 for one arm may come from different checkpoints, and the
stdout dump records which. There is no held-out split behind that choice, so these are
optimistic and equally so across arms.

Only Qwen3-1.7B has an alpha=1, gamma=1 generator; the 4B runs are alpha 2.5 and 3.

Writes results/figures/hint_trained_bars.png.

Usage:
    .venv/bin/python eval/viz/hint_trained_bars.py
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
from matplotlib.ticker import PercentFormatter

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
OUT = FIGURES / "hint_trained_bars.png"

MODEL = "Qwen3-1.7B"
TRAIN_DATASET = "deepmath"
EVAL_DATASETS = ["aime24", "aime25"]
GENERATOR = "a1_g1_checkpoint-100"
KS = [1, 8, 16]

# -------------------- style --------------------

CANVAS = "#FBF9F4"
CARD = "#FFFFFF"
INK = "#191917"
LABEL = "#33312E"
GRID = "#E8E5DE"

SANS = FontProperties(family=["Carlito", "DejaVu Sans"])
SERIF = FontProperties(family=["Caladea", "DejaVu Serif"])

# Base keeps the neutral grey used elsewhere; `hint` keeps its blue. The generator-written
# arm takes a violet -- a free slot, so it cannot be confused with rollout/answer/full.
ARMS = [
    ("base", "Base", "#BFBCB4"),
    ("hint", "SDFT hint", "#3D74D0"),
    ("hint_trained", "SDFT hint-trained", "#7B5EA7"),
]


# -------------------- data --------------------


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def arm_dir(arm: str, eval_dataset: str) -> Path:
    if arm == "base":
        return RESULTS / eval_dataset / "base" / MODEL
    root = RESULTS / eval_dataset / TRAIN_DATASET / MODEL / "sdft"
    if arm == "hint_trained":
        return root / "hint_trained" / GENERATOR
    return root / arm


def best_over_checkpoints(paths: list[Path]) -> dict:
    """Max pass@k over every summary, recording which checkpoint supplied each max."""
    best: dict = {}
    for path in paths:
        pass_at_k = read_json(path).get("pass_at_k") or {}
        for k in KS:
            value = pass_at_k.get(f"pass@{k}")
            if value is None:
                continue
            if k not in best or value > best[k]["value"]:
                best[k] = {"value": value, "source": str(path.parent.relative_to(RESULTS))}
    return best


def load_table() -> dict:
    """table[eval_dataset][arm][k] -> {value, source}."""
    table: dict = {}
    for eval_dataset in EVAL_DATASETS:
        table[eval_dataset] = {}
        for arm, _, _ in ARMS:
            directory = arm_dir(arm, eval_dataset)
            paths = sorted(directory.rglob("summary.json"))
            if not paths:
                raise FileNotFoundError(f"No summaries for {arm} under {directory}")
            table[eval_dataset][arm] = best_over_checkpoints(paths)
    return table


# -------------------- plot --------------------


def draw_panel(ax, eval_dataset: str, panel: dict) -> None:
    x = np.arange(len(KS), dtype=float)
    width = 0.8 / len(ARMS)

    for index, (arm, _, colour) in enumerate(ARMS):
        offset = (index - (len(ARMS) - 1) / 2) * width
        values = [panel[arm].get(k, {}).get("value", np.nan) for k in KS]
        bars = ax.bar(
            x + offset, values, width=width * 0.86, color=colour,
            edgecolor=CARD, linewidth=0.8, zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            if np.isnan(value):
                continue
            text = ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.016, f"{value * 100:.1f}",
                ha="center", va="bottom", fontproperties=SANS, fontsize=10.5,
                color=LABEL, zorder=6,
            )
            text.set_path_effects([path_effects.withStroke(linewidth=2.6, foreground=CARD)])

    # Baseline reference across each group, so a regression is visible at a glance.
    for position, k in zip(x, KS, strict=True):
        base = panel["base"].get(k, {}).get("value")
        if base is None:
            continue
        ax.plot(
            [position - 0.44, position + 0.44], [base, base],
            linestyle=(0, (3, 3)), color=INK, linewidth=1.3, zorder=4,
        )

    ax.set_xticks(x, [f"pass@{k}" for k in KS])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=1.1, zorder=0)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(2.0)

    ax.tick_params(axis="both", length=0, pad=8, labelsize=13.5, colors=INK)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(SANS)
    ax.set_title(
        eval_dataset.upper(), fontproperties=SANS, fontsize=17,
        fontweight="bold", color=INK, pad=14,
    )


def build_figure(table: dict):
    fig = plt.figure(figsize=(11.5, 10.6), dpi=200, facecolor=CANVAS)
    fig.patches.append(FancyBboxPatch(
        (0.012, 0.018), 0.976, 0.964,
        boxstyle="round,pad=0,rounding_size=0.018",
        transform=fig.transFigure, facecolor=CARD, edgecolor="none", zorder=-10,
    ))
    fig.text(
        0.5, 0.960, "Distilling on generator-written hints",
        ha="center", va="center", fontproperties=SANS, fontsize=25,
        fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.922,
        f"{MODEL} · trained on DeepMath · dashed line marks the base model",
        ha="center", va="center", fontproperties=SERIF, fontsize=16, color="#4A4844",
    )

    left, width = 0.115, 0.845
    bottom, height, vgap = 0.155, 0.300, 0.105
    for index, eval_dataset in enumerate(EVAL_DATASETS):
        ax = fig.add_axes([
            left,
            bottom + (len(EVAL_DATASETS) - 1 - index) * (height + vgap),
            width, height,
        ], zorder=5)
        ax.set_facecolor("none")
        draw_panel(ax, eval_dataset, table[eval_dataset])

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor=CARD, linewidth=0.8)
        for _, _, colour in ARMS
    ]
    legend = fig.legend(
        handles, [name for _, name, _ in ARMS],
        loc="lower center", bbox_to_anchor=(0.5, 0.022), frameon=True, ncol=len(ARMS),
        prop=FontProperties(family=["Carlito", "DejaVu Sans"], size=14),
        handletextpad=0.7, borderpad=0.8, columnspacing=2.0, handlelength=1.5,
    )
    frame = legend.get_frame()
    frame.set_facecolor("#FDFCF9")
    frame.set_edgecolor("#DCD8CE")
    frame.set_linewidth(1.2)
    frame.set_boxstyle("round,pad=0.45,rounding_size=0.12")
    for text in legend.get_texts():
        text.set_color(INK)

    return fig


if __name__ == "__main__":
    mpl.rcParams["savefig.facecolor"] = CANVAS
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = load_table()
    for eval_dataset in EVAL_DATASETS:
        print(f"=== {eval_dataset}")
        for arm, label, _ in ARMS:
            for k in KS:
                entry = table[eval_dataset][arm].get(k)
                if entry is None:
                    print(f"  {label:20s} pass@{k:<2d}    --")
                    continue
                print(
                    f"  {label:20s} pass@{k:<2d} {entry['value'] * 100:5.1f}   "
                    f"{entry['source']}"
                )
    figure = build_figure(table)
    figure.savefig(OUT, dpi=200, facecolor=CANVAS)
    print(f"\nWrote {OUT.relative_to(ROOT)}")
