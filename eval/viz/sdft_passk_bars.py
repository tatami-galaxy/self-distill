"""Self-distillation avg@16 on AIME24 and AIME25, best checkpoint per model and arm.

The summaries record `pass@1`, but every eval ran with n_samples=16 and the unbiased
estimator reduces to c/n at k=1 (eval/run_eval.py). So `pass@1` here is the mean
per-sample accuracy over 16 samples, and is labelled avg@16 throughout.

For each (model, arm) the best value across every checkpoint is taken. This is an
optimistic, selection-biased estimate -- there is no held-out split behind the
checkpoint choice -- and `hint_trained` additionally maximizes over generator settings,
so it draws from three times as many candidates as the other arms. The stdout dump
records which checkpoint supplied each maximum.

Writes results/figures/sdft_avg16_bars.png.

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
OUT = FIGURES / "sdft_avg16_bars.png"
TABLE_OUT = FIGURES / "sdft_avg16_table.tex"

EVAL_DATASETS = ["aime24", "aime25"]
TRAIN_DATASET = "deepmath"
MODELS = ["Qwen3-1.7B", "Qwen3-4B"]

# The key stored in summary.json, and how it is labelled here. See the module docstring.
METRIC_KEY = "pass@1"
METRIC_LABEL = "avg@16"

# -------------------- style --------------------

CANVAS = "#FBF9F4"
CARD = "#FFFFFF"
INK = "#191917"
LABEL = "#33312E"
GRID = "#E8E5DE"

SANS = FontProperties(family=["Carlito", "DejaVu Sans"])
SERIF = FontProperties(family=["Caladea", "DejaVu Serif"])

# The two reference methods lead, as light and dark neutrals; the PI arms follow in
# colour, with the generator-written hints last.
ARMS = [
    ("base", "Base", "#BFBCB4"),
    ("grpo", "GRPO", "#57544E"),
    ("hint", "SDFT hint", "#3D74D0"),
    ("answer", "SDFT answer", "#E9B02E"),
    ("full", "SDFT full", "#49B083"),
    ("hint_trained", "SDFT hint-trained", "#7B5EA7"),
]
SKIP_VARIANTS = {"hint-ema-logit"}  # only trained for one model so far

# `hint_trained` takes the best over generator settings, restricted to those present for
# both models on both eval sets. t0.7_g6_* exists only for Qwen3-1.7B on AIME24; including
# it would give that one cell an advantage the other three cannot draw on.
HINT_TRAINED_GENERATORS = ["a1_g1", "a2.5_g1", "a3_g1"]


# -------------------- data --------------------


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def best_over_checkpoints(paths: list[Path]) -> dict | None:
    """Max of the metric over every summary, recording which checkpoint supplied it."""
    best = None
    for path in paths:
        value = (read_json(path).get("pass_at_k") or {}).get(METRIC_KEY)
        if value is None:
            continue
        if best is None or value > best["value"]:
            best = {"value": value, "source": str(path.parent.relative_to(RESULTS))}
    return best


def arm_directories(arm: str, eval_dataset: str, model: str) -> list[Path]:
    """Where each arm's checkpoint summaries live; layouts differ between arms."""
    if arm == "base":
        return [RESULTS / eval_dataset / "base" / model]
    trained = RESULTS / eval_dataset / TRAIN_DATASET / model
    if arm == "grpo":
        return [trained / "grpo"]
    if arm == "hint_trained":
        return [
            trained / "sdft" / "hint_trained" / f"{generator}_checkpoint-100"
            for generator in HINT_TRAINED_GENERATORS
        ]
    return [trained / "sdft" / arm]


def load_table() -> dict:
    """table[eval_dataset][model][arm] -> {value, source} or None."""
    table: dict = {}
    for eval_dataset in EVAL_DATASETS:
        table[eval_dataset] = {}
        for model in MODELS:
            table[eval_dataset][model] = {}
            for arm, _, _ in ARMS:
                directories = [
                    directory for directory in arm_directories(arm, eval_dataset, model)
                    if directory.is_dir()
                ]
                if not directories:
                    print(f"  ! no {arm} for {model} on {eval_dataset}; leaving it blank")
                    table[eval_dataset][model][arm] = None
                    continue
                paths = [
                    path
                    for directory in directories
                    for path in sorted(directory.rglob("summary.json"))
                    if not SKIP_VARIANTS.intersection(path.parts)
                ]
                table[eval_dataset][model][arm] = best_over_checkpoints(paths)
    return table


# -------------------- plot --------------------


def draw_panel(ax, eval_dataset: str, panel: dict) -> None:
    x = np.arange(len(MODELS), dtype=float)
    width = 0.8 / len(ARMS)

    for index, (arm, _, colour) in enumerate(ARMS):
        offset = (index - (len(ARMS) - 1) / 2) * width
        values = [
            panel[model][arm]["value"] if panel[model][arm] else np.nan
            for model in MODELS
        ]
        bars = ax.bar(
            x + offset, values, width=width * 0.86, color=colour,
            edgecolor=CARD, linewidth=0.8, zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            if np.isnan(value):
                continue
            text = ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.014, f"{value * 100:.1f}",
                ha="center", va="bottom", fontproperties=SANS, fontsize=9.6,
                color=LABEL, zorder=6,
            )
            text.set_path_effects([path_effects.withStroke(linewidth=2.6, foreground=CARD)])

    # Baseline reference across each model group, so a regression reads at a glance.
    for position, model in zip(x, MODELS, strict=True):
        base = panel[model]["base"]
        if base is None:
            continue
        ax.plot(
            [position - 0.44, position + 0.44], [base["value"], base["value"]],
            linestyle=(0, (3, 3)), color=INK, linewidth=1.3, zorder=4,
        )

    ax.set_xticks(x, MODELS)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=1.1, zorder=0)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(2.0)

    ax.tick_params(axis="both", length=0, pad=8, labelsize=14, colors=INK)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(SANS)
    ax.set_title(
        eval_dataset.upper(), fontproperties=SANS, fontsize=18,
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
        0.5, 0.960, f"Self-distillation results · {METRIC_LABEL}",
        ha="center", va="center", fontproperties=SANS, fontsize=25,
        fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.922, "Trained on DeepMath · dashed line marks the base model",
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
        ax.set_ylabel(
            f"{METRIC_LABEL} (%)", fontproperties=SANS,
            fontsize=14, fontweight="bold", color=INK, labelpad=10,
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor=CARD, linewidth=0.8)
        for _, _, colour in ARMS
    ]
    legend = fig.legend(
        handles, [name for _, name, _ in ARMS],
        loc="lower center", bbox_to_anchor=(0.5, 0.022), frameon=True, ncol=len(ARMS),
        prop=FontProperties(family=["Carlito", "DejaVu Sans"], size=14),
        handletextpad=0.7, borderpad=0.8, columnspacing=1.7, handlelength=1.5,
    )
    frame = legend.get_frame()
    frame.set_facecolor("#FDFCF9")
    frame.set_edgecolor("#DCD8CE")
    frame.set_linewidth(1.2)
    frame.set_boxstyle("round,pad=0.45,rounding_size=0.12")
    for text in legend.get_texts():
        text.set_color(INK)

    return fig


# -------------------- LaTeX table --------------------


def latex_table(table: dict) -> str:
    """A booktabs table of the same numbers, with the best trained arm bold per column.

    `base` is a reference row rather than a competitor, so it is excluded from the bold
    comparison; bolding it would leave the reader working out which rows are methods.
    """
    columns = [(dataset, model) for dataset in EVAL_DATASETS for model in MODELS]

    def value_of(dataset: str, model: str, arm: str) -> float | None:
        entry = table[dataset][model][arm]
        return entry["value"] * 100 if entry else None

    winners = {}
    for dataset, model in columns:
        scored = [
            (value_of(dataset, model, arm), arm)
            for arm, _, _ in ARMS
            if arm != "base" and value_of(dataset, model, arm) is not None
        ]
        winners[(dataset, model)] = max(scored)[1] if scored else None

    lines = [
        r"\begin{tabular}{l" + "cc" * len(EVAL_DATASETS) + "}",
        r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{dataset.upper()}}}" for dataset in EVAL_DATASETS
        ) + r" \\",
    ]
    lines.append(" ".join(
        rf"\cmidrule(lr){{{2 + 2 * index}-{3 + 2 * index}}}"
        for index in range(len(EVAL_DATASETS))
    ))
    lines.append(
        "Method & " + " & ".join(model for _, model in columns) + r" \\"
    )
    lines.append(r"\midrule")

    for arm, label, _ in ARMS:
        cells = []
        for dataset, model in columns:
            value = value_of(dataset, model, arm)
            if value is None:
                cells.append("--")
                continue
            if arm == "base":
                cells.append(f"{value:.1f}")
                continue
            base = value_of(dataset, model, "base")
            delta = "" if base is None else f" ({value - base:+.1f})"
            cell = f"{value:.1f}{delta}"
            cells.append(rf"\textbf{{{cell}}}" if winners[(dataset, model)] == arm else cell)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        if arm == "base":
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    mpl.rcParams["savefig.facecolor"] = CANVAS
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = load_table()
    for eval_dataset in EVAL_DATASETS:
        print(f"########## {eval_dataset}  ({METRIC_LABEL})")
        for model in MODELS:
            print(f"=== {model}")
            for arm, label, _ in ARMS:
                entry = table[eval_dataset][model][arm]
                if entry is None:
                    print(f"  {label:20s}    --")
                    continue
                print(f"  {label:20s} {entry['value'] * 100:5.1f}   {entry['source']}")
    figure = build_figure(table)
    figure.savefig(OUT, dpi=200, facecolor=CANVAS)
    print(f"\nWrote {OUT.relative_to(ROOT)}")

    latex = latex_table(table)
    TABLE_OUT.write_text(latex + "\n")
    print(f"Wrote {TABLE_OUT.relative_to(ROOT)}\n")
    print(latex)
