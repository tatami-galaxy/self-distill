"""Self-distillation pass@k on AIME24, best checkpoint per model and PI variant.

For each (model, arm, k) the best value across every checkpoint and run is taken.
Note that the maximum is chosen independently per k, so pass@1 and pass@16 for the
same arm may come from different checkpoints; the stdout dump records which.
This is an optimistic, selection-biased estimate -- there is no held-out split
behind the checkpoint choice.

Writes results/figures/sdft_passk_bars.png.

Usage:
    .venv/bin/python eval/viz/sdft_passk_bars.py
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
OUT = FIGURES / "sdft_passk_bars.png"

EVAL_DATASET = "aime24"
TRAIN_DATASET = "deepmath"

# -------------------- style --------------------

CANVAS = "#FBF9F4"
CARD = "#FFFFFF"
INK = "#191917"
LABEL = "#33312E"
GRID = "#E8E5DE"

SANS = FontProperties(family=["Carlito", "DejaVu Sans"])
SERIF = FontProperties(family=["Caladea", "DejaVu Serif"])

MODELS = ["Qwen3-1.7B", "Qwen3-4B"]
KS = [1, 8, 16]

# Bar order and colours. The PI-to-hue assignment matches results_overview.py
# (hint blue, answer amber, full green) in a more saturated palette; the untrained
# baseline stays a neutral warm grey.
ARMS = [
    ("base", "Base", "#BFBCB4"),
    ("rollout", "SDFT rollout", "#E0673C"),
    ("hint", "SDFT hint", "#3D74D0"),
    ("answer", "SDFT answer", "#E9B02E"),
    ("full", "SDFT full", "#49B083"),
]
SKIP_VARIANTS = {"hint-ema-logit"}  # only trained for one model so far


# -------------------- data --------------------


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


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


def load_arms() -> dict:
    """table[model][arm][k] -> {value, source}."""
    table: dict = {}
    for model in MODELS:
        table[model] = {}

        base_dir = RESULTS / EVAL_DATASET / "base" / model
        base_paths = sorted(base_dir.rglob("summary.json"))
        if not base_paths:
            raise FileNotFoundError(f"No base summaries for {model} under {base_dir}")
        table[model]["base"] = best_over_checkpoints(base_paths)

        for arm, _, _ in ARMS[1:]:
            variant_dir = RESULTS / EVAL_DATASET / TRAIN_DATASET / model / "sdft" / arm
            if not variant_dir.is_dir():
                print(f"  ! no sdft/{arm} for {model}; leaving it blank")
                table[model][arm] = {}
                continue
            paths = [
                path for path in sorted(variant_dir.rglob("summary.json"))
                if not SKIP_VARIANTS.intersection(path.parts)
            ]
            table[model][arm] = best_over_checkpoints(paths)
    return table


# -------------------- plot --------------------


def draw_panel(ax, model: str, arms: dict) -> None:
    x = np.arange(len(KS), dtype=float)
    width = 0.8 / len(ARMS)

    for index, (arm, _, colour) in enumerate(ARMS):
        offset = (index - (len(ARMS) - 1) / 2) * width
        values = [arms.get(arm, {}).get(k, {}).get("value", np.nan) for k in KS]
        bars = ax.bar(
            x + offset, values, width=width * 0.88, color=colour,
            edgecolor=CARD, linewidth=0.8, zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            if np.isnan(value):
                continue
            label = ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.016, f"{value * 100:.1f}",
                ha="center", va="bottom", fontproperties=SANS, fontsize=10.5,
                color=LABEL, zorder=6,
            )
            # Keep labels legible where they land on the dashed baseline.
            label.set_path_effects([
                path_effects.withStroke(linewidth=2.6, foreground=CARD),
            ])

    # Baseline reference across each group makes regressions readable at a glance.
    for position, k in zip(x, KS, strict=True):
        base = arms.get("base", {}).get(k, {}).get("value")
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
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(SANS)
    ax.set_title(model, fontproperties=SANS, fontsize=17, fontweight="bold", color=INK, pad=14)


def build_figure(table: dict):
    fig = plt.figure(figsize=(12.0, 11.0), dpi=200, facecolor=CANVAS)
    card = FancyBboxPatch(
        (0.012, 0.018), 0.976, 0.964,
        boxstyle="round,pad=0,rounding_size=0.018",
        transform=fig.transFigure, facecolor=CARD, edgecolor="none", zorder=-10,
    )
    fig.patches.append(card)

    fig.text(
        0.5, 0.960, "Self-distillation on AIME24",
        ha="center", va="center", fontproperties=SANS, fontsize=25,
        fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.922, "Trained on DeepMath · dashed line marks the base model",
        ha="center", va="center", fontproperties=SERIF, fontsize=16, color="#4A4844",
    )

    left, width = 0.090, 0.870
    bottom, height, vgap = 0.135, 0.300, 0.105
    for index, model in enumerate(MODELS):
        ax = fig.add_axes([
            left,
            bottom + (len(MODELS) - 1 - index) * (height + vgap),
            width, height,
        ], zorder=5)
        ax.set_facecolor("none")
        draw_panel(ax, model, table[model])

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor=CARD, linewidth=0.8)
        for _, _, colour in ARMS
    ]
    legend = fig.legend(
        handles, [label for _, label, _ in ARMS],
        loc="lower center", bbox_to_anchor=(0.5, 0.022), frameon=True, ncol=len(ARMS),
        prop=FontProperties(family=["Carlito", "DejaVu Sans"], size=14),
        handletextpad=0.7, borderpad=0.8, columnspacing=1.9, handlelength=1.5,
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
    table = load_arms()
    for model in MODELS:
        print(f"=== {model}")
        for arm, label, _ in ARMS:
            for k in KS:
                entry = table[model].get(arm, {}).get(k)
                if entry is None:
                    print(f"  {label:14s} pass@{k:<2d}    --")
                    continue
                print(
                    f"  {label:14s} pass@{k:<2d} {entry['value'] * 100:5.1f}   "
                    f"{entry['source']}"
                )
    figure = build_figure(table)
    figure.savefig(OUT, dpi=200, facecolor=CANVAS)
    print(f"\nWrote {OUT.relative_to(ROOT)}")
