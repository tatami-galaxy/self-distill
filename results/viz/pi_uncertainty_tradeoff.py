"""Privileged information: verbalized uncertainty vs. pass@8.

Combines two result families that were produced by separate runs:

  - x: `mean_e_think` from results/teacher_uncertainty/*/teacher_uncertainty_summary.json
  - y: `pass@8`      from results/passk_pi/*/passk_pi_summary.json

One curve per model; one marker per PI mode. The x-axis is inverted so the PI
ladder reads left to right, from no privileged information to the full solution.

Usage:
    .venv/bin/python results/viz/pi_uncertainty_tradeoff.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter

# -------------------- paths --------------------


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).parent).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "results").is_dir() and (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repository root containing results/ and train/")


ROOT = find_repo_root()
RESULTS = ROOT / "results"
OUT = Path(__file__).parent / "pi_uncertainty_tradeoff_linear_inverted.png"

# -------------------- style --------------------

CANVAS = "#FBF9F4"
CARD = "#FFFFFF"
INK = "#191917"
GRID = "#E8E5DE"
ARROW = "#7A7872"

SANS = FontProperties(family=["Carlito", "DejaVu Sans"])
SERIF = FontProperties(family=["Caladea", "DejaVu Serif"])

MODEL_STYLE = {
    "Qwen3-4B": {"color": "#E1603C", "label": "Qwen3-4B", "z": 3},
    "Qwen3-1.7B": {"color": "#3372DA", "label": "Qwen3-1.7B", "z": 2},
}

PI_ORDER = ["full", "rollout", "answer", "hint", "none"]

# -------------------- axis layout --------------------

XLIM = (100, -4)  # inverted: markers increase to the left
YLIM = (80, 101.5)
XTICKS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
YTICKS = [80, 85, 90, 95, 100]
LEGEND_ANCHOR = (0.48, 0.03)

# Hand-tuned label offsets in points: (dx, dy, ha, va)
LABEL_OFFSETS = {
    ("Qwen3-4B", "full"): (-2, 15, "center", "bottom"),
    ("Qwen3-4B", "rollout"): (-4, -16, "center", "top"),
    ("Qwen3-4B", "answer"): (4, 16, "center", "bottom"),
    ("Qwen3-4B", "hint"): (-14, 2, "right", "center"),
    ("Qwen3-4B", "none"): (-14, 1, "right", "center"),
    ("Qwen3-1.7B", "full"): (2, -16, "center", "top"),
    ("Qwen3-1.7B", "rollout"): (2, -16, "center", "top"),
    ("Qwen3-1.7B", "answer"): (15, 6, "left", "bottom"),
    ("Qwen3-1.7B", "hint"): (14, -4, "left", "center"),
    ("Qwen3-1.7B", "none"): (2, -16, "center", "top"),
}

MAX_PROBLEM_COUNT_SPREAD = 5


# -------------------- data --------------------


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_points() -> dict[str, list[dict]]:
    """Join teacher-uncertainty x-values onto passk_pi y-values, keyed by model + PI mode."""
    uncertainty: dict[tuple[str, str], dict] = {}
    for path in sorted(RESULTS.glob("teacher_uncertainty/*/teacher_uncertainty_summary.json")):
        summary = read_json(path)
        model = summary["teacher_model"].split("/")[-1]
        conditions = summary.get("uncertainty", summary.get("behavior", {}))
        for pi_mode, metrics in conditions.items():
            uncertainty[(model, pi_mode)] = metrics

    series: dict[str, list[dict]] = {}
    for path in sorted(RESULTS.glob("passk_pi/*/passk_pi_summary.json")):
        summary = read_json(path)
        model = summary["model"].split("/")[-1]
        points = []
        for pi_mode, metrics in summary["pass_at_k"].items():
            key = (model, pi_mode)
            if key not in uncertainty:
                print(f"  ! no teacher_uncertainty row for {model}/{pi_mode}; skipping")
                continue
            points.append(
                {
                    "pi_mode": pi_mode,
                    "x": uncertainty[key]["mean_e_think"],
                    "y": metrics["pass@8"] * 100,
                    "pass@1": metrics["pass@1"] * 100,
                    "n_problems": summary["n_problems"],
                    "max_tokens": summary["max_tokens"],
                }
            )
        points.sort(key=lambda point: PI_ORDER.index(point["pi_mode"]))
        series[model] = points
    return series


# -------------------- plot --------------------


def subtitle_text(series: dict[str, list[dict]]) -> str:
    """Read run parameters off the loaded summaries so the caption can't drift.

    Problem counts that differ only slightly across runs are reported as a single
    approximate number. A larger spread would make that approximation misleading,
    so it is refused rather than rounded away.
    """
    problems = sorted({point["n_problems"] for points in series.values() for point in points})
    tokens = sorted({point["max_tokens"] for points in series.values() for point in points})

    spread = problems[-1] - problems[0]
    if spread > MAX_PROBLEM_COUNT_SPREAD:
        per_model = ", ".join(
            f"{model}={points[0]['n_problems']}"
            for model, points in sorted(series.items())
            if points
        )
        raise ValueError(
            f"Problem counts differ by {spread} across runs ({per_model}), which exceeds the "
            f"tolerance of {MAX_PROBLEM_COUNT_SPREAD}. Reporting a single approximate count "
            "would misrepresent the comparison; re-run the evals on a common problem set."
        )

    problem_text = str(problems[0]) if spread == 0 else f"~{problems[-1]}"
    token_text = str(tokens[0]) if len(tokens) == 1 else f"{tokens[0]}–{tokens[-1]}"
    return f"DeepMath · {problem_text} problems · {token_text} max tokens"


def add_direction_arrows(fig, ax, xlabel, ylabel) -> None:
    """Draw an arrow outside each spine pointing the way that axis increases.

    Each arrow is measured to span exactly its axis label, so the two stay visually
    tied together. Direction is read off the axis limits, so the inverted x-axis
    gets an arrow pointing left without any extra bookkeeping.
    """
    fig.canvas.draw()  # label extents are only measurable once laid out
    renderer = fig.canvas.get_renderer()
    to_axes = ax.transAxes.inverted()

    style = {
        "xycoords": "axes fraction",
        "textcoords": "axes fraction",
        "annotation_clip": False,
        "arrowprops": {
            "arrowstyle": "-|>,head_length=0.7,head_width=0.25",
            "color": ARROW,
            "linewidth": 1.7,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    }

    box = xlabel.get_window_extent(renderer)
    left = to_axes.transform((box.x0, box.y0))[0]
    right = to_axes.transform((box.x1, box.y1))[0]
    x_start, x_end = (left, right) if XLIM[1] > XLIM[0] else (right, left)
    ax.annotate("", xy=(x_end, -0.105), xytext=(x_start, -0.105), **style)

    box = ylabel.get_window_extent(renderer)
    bottom = to_axes.transform((box.x0, box.y0))[1]
    top = to_axes.transform((box.x1, box.y1))[1]
    y_start, y_end = (bottom, top) if YLIM[1] > YLIM[0] else (top, bottom)
    ax.annotate("", xy=(-0.052, y_end), xytext=(-0.052, y_start), **style)


def build_figure(series: dict[str, list[dict]]):
    fig = plt.figure(figsize=(15.2, 8.3), dpi=200, facecolor=CANVAS)

    card = FancyBboxPatch(
        (0.012, 0.018),
        0.976,
        0.964,
        boxstyle="round,pad=0,rounding_size=0.018",
        transform=fig.transFigure,
        facecolor=CARD,
        edgecolor="none",
        zorder=-10,
    )
    fig.patches.append(card)

    ax = fig.add_axes([0.125, 0.185, 0.835, 0.625], zorder=5)
    ax.set_facecolor("none")

    # ---- titles ----
    fig.text(
        0.5, 0.935, "Verbalized uncertainty vs. accuracy under privileged information",
        ha="center", va="center", fontproperties=SANS, fontsize=25, fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.876, subtitle_text(series),
        ha="center", va="center", fontproperties=SERIF, fontsize=16.5, color="#4A4844",
    )

    # ---- series ----
    for model, points in series.items():
        style = MODEL_STYLE.get(model, {"color": "#888888", "label": model, "z": 1})
        xs = [point["x"] for point in points]
        ys = [point["y"] for point in points]
        ax.plot(
            xs, ys, "-", color=style["color"], linewidth=3.0,
            solid_capstyle="round", zorder=style["z"],
        )
        ax.plot(
            xs, ys, "o", color=style["color"], markersize=13.5,
            markeredgecolor=CARD, markeredgewidth=2.0, zorder=style["z"] + 4,
            label=style["label"],
        )
        for point in points:
            dx, dy, ha, va = LABEL_OFFSETS.get(
                (model, point["pi_mode"]), (0, 13, "center", "bottom")
            )
            ax.annotate(
                point["pi_mode"],
                xy=(point["x"], point["y"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha, va=va,
                fontproperties=SANS, fontsize=12.5, fontweight="bold",
                color=style["color"], zorder=style["z"] + 5,
            )

    # ---- axes ----
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(YTICKS)

    ax.grid(axis="x", color=GRID, linewidth=1.1, zorder=0)
    ax.grid(axis="y", color=GRID, linewidth=1.1, linestyle=(0, (2, 4)), zorder=0)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(2.0)

    ax.tick_params(axis="both", length=0, pad=9, labelsize=14.5, colors=INK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(SANS)

    ylabel = ax.set_ylabel(
        "pass@8 (%)", fontproperties=SANS, fontsize=17,
        fontweight="bold", color=INK, labelpad=36,
    )
    xlabel = fig.text(
        0.543, 0.066, "Mean CoT epistemic markers",
        ha="center", va="center", fontproperties=SANS, fontsize=17.5, fontweight="bold", color=INK,
    )

    # ---- legend ----
    legend = ax.legend(
        loc="lower left", bbox_to_anchor=LEGEND_ANCHOR, frameon=True,
        prop=FontProperties(family=["Carlito", "DejaVu Sans"], size=15),
        handletextpad=0.7, borderpad=0.85, labelspacing=0.62, markerscale=1.05,
    )
    frame = legend.get_frame()
    frame.set_facecolor("#FDFCF9")
    frame.set_edgecolor("#DCD8CE")
    frame.set_linewidth(1.2)
    frame.set_boxstyle("round,pad=0.45,rounding_size=0.12")
    for text in legend.get_texts():
        text.set_color(INK)

    add_direction_arrows(fig, ax, xlabel, ylabel)

    return fig


if __name__ == "__main__":
    mpl.rcParams["savefig.facecolor"] = CANVAS
    series = load_points()
    for model, points in series.items():
        print(f"{model}:")
        for point in points:
            print(
                f"  {point['pi_mode']:8s} e_think={point['x']:6.2f}  "
                f"pass@8={point['y']:5.1f}  pass@1={point['pass@1']:5.1f}"
            )

    figure = build_figure(series)
    figure.savefig(OUT, dpi=200, facecolor=CANVAS)
    print(f"\nWrote {OUT}")
