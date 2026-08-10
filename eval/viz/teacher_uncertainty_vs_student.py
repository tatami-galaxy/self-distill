"""Teacher uncertainty verbalization vs. the accuracy of the student it trains.

  - x: `mean_e_think` from results/teacher_uncertainty_{TEACHER_BUDGET}/, i.e. how much
       the teacher verbalizes uncertainty while generating under each PI condition.
  - y: AIME24 pass@k of the SDFT student trained on that teacher's output, taking the
       best checkpoint per k (same selection as sdft_passk_bars.py, and equally
       selection-biased -- no held-out split behind the checkpoint choice).

One panel per model, one curve per k, one marker per PI condition. There is no `none`
arm because no student was distilled without privileged information.

Writes results/figures/teacher_uncertainty_vs_student.png.

Usage:
    .venv/bin/python eval/viz/teacher_uncertainty_vs_student.py
"""

from __future__ import annotations

import json
import math
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
FIGURES = RESULTS / "figures"  # generated output; not tracked by git
def output_path(budget: str) -> Path:
    return FIGURES / f"teacher_uncertainty_vs_student_{budget}.png"

EVAL_DATASET = "aime24"
TRAIN_DATASET = "deepmath"
# Which teacher_uncertainty_<budget> runs supply the x-axis. The 16k teachers push the
# `hint`/`answer` cluster right and collapse `rollout` onto `full`, which is harder to
# read without changing the conclusion; add "16k" back here to regenerate that version.
TEACHER_BUDGETS = ["8k"]

# -------------------- style --------------------

CANVAS = "#FBF9F4"
CARD = "#FFFFFF"
INK = "#191917"
MUTED = "#6E6C66"
GRID = "#E8E5DE"
ARROW = "#7A7872"

SANS = FontProperties(family=["Carlito", "DejaVu Sans"])
SERIF = FontProperties(family=["Caladea", "DejaVu Serif"])

MODELS = ["Qwen3-1.7B", "Qwen3-4B"]
PI_MODES = ["full", "rollout", "answer", "hint"]  # ordered along x at plot time

# Curve colours reuse the sdft_passk_bars palette, here keyed by k rather than by PI.
K_STYLE = {
    1: {"color": "#E0673C", "label": "pass@1"},
    8: {"color": "#E9B02E", "label": "pass@8"},
    16: {"color": "#3D74D0", "label": "pass@16"},
}
KS = [1, 8, 16]

YLIM = (15, 100)
YTICKS = [20, 40, 60, 80, 100]

# PI names sit just above each panel, outside the axes because the 4B curves run close
# to the top of the y-range. Conditions whose teachers land closer together than this
# many markers get their labels staggered either side of the shared guide.
STAGGER_THRESHOLD = 6.0


def axis_bounds(series: dict[str, list[dict]]) -> tuple[tuple[float, float], list[int]]:
    """Fit the x-range to the data so a different budget doesn't clip or pad the panels."""
    largest = max(point["x"] for points in series.values() for point in points)
    top = int(math.ceil(largest / 10.0)) * 10
    return (-2, top + 8), list(range(0, top + 1, 10))


def label_placements(points: list[dict]) -> dict[int, tuple[int, str]]:
    """Index -> (dx in points, horizontal alignment), staggering crowded neighbours."""
    runs: list[list[int]] = [[0]]
    for index in range(1, len(points)):
        if points[index]["x"] - points[index - 1]["x"] < STAGGER_THRESHOLD:
            runs[-1].append(index)
        else:
            runs.append([index])

    placement: dict[int, tuple[int, str]] = {}
    for run in runs:
        if len(run) == 1:
            placement[run[0]] = (0, "center")
            continue
        for position, index in enumerate(run):
            step = (position // 2 + 1) * 9
            placement[index] = (-step, "right") if position % 2 == 0 else (step, "left")
    return placement


# -------------------- data --------------------


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_teacher_uncertainty(budget: str) -> dict:
    """(model, pi_mode) -> mean_e_think."""
    values: dict[tuple[str, str], float] = {}
    pattern = f"teacher_uncertainty_{budget}/*/teacher_uncertainty_summary.json"
    paths = sorted(RESULTS.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No teacher-uncertainty summaries matched {pattern}")
    for path in paths:
        summary = read_json(path)
        model = summary["teacher_model"].split("/")[-1]
        conditions = summary.get("uncertainty", summary.get("behavior", {}))
        for pi_mode, metrics in conditions.items():
            values[(model, pi_mode)] = metrics["mean_e_think"]
    return values


def best_student_pass_at_k(model: str, pi_mode: str) -> dict[int, float]:
    """Max AIME24 pass@k over every checkpoint of the SDFT run for this PI arm."""
    variant_dir = RESULTS / EVAL_DATASET / TRAIN_DATASET / model / "sdft" / pi_mode
    if not variant_dir.is_dir():
        raise FileNotFoundError(f"No SDFT results for {model}/{pi_mode} at {variant_dir}")
    best: dict[int, float] = {}
    for path in sorted(variant_dir.rglob("summary.json")):
        pass_at_k = read_json(path).get("pass_at_k") or {}
        for k in KS:
            value = pass_at_k.get(f"pass@{k}")
            if value is not None and (k not in best or value > best[k]):
                best[k] = value
    missing = [k for k in KS if k not in best]
    if missing:
        raise ValueError(f"{model}/{pi_mode} is missing pass@{missing}")
    return best


def load_points(budget: str) -> dict[str, list[dict]]:
    """series[model] -> points sorted along the uncertainty axis."""
    uncertainty = load_teacher_uncertainty(budget)
    series: dict[str, list[dict]] = {}
    for model in MODELS:
        points = []
        for pi_mode in PI_MODES:
            key = (model, pi_mode)
            if key not in uncertainty:
                raise KeyError(f"No teacher_uncertainty row for {model}/{pi_mode}")
            student = best_student_pass_at_k(model, pi_mode)
            points.append({
                "pi_mode": pi_mode,
                "x": uncertainty[key],
                **{k: student[k] * 100 for k in KS},
            })
        points.sort(key=lambda point: point["x"])
        series[model] = points
    return series


# -------------------- plot --------------------


def draw_panel(ax, model: str, points: list[dict], xlim, xticks) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)

    # A guide per PI condition ties the three curves to a shared x position.
    placement = label_placements(points)
    for index, point in enumerate(points):
        ax.axvline(
            point["x"], color=GRID, linewidth=1.2, linestyle=(0, (2, 4)), zorder=0,
        )
        dx, ha = placement[index]
        ax.annotate(
            point["pi_mode"],
            xy=(point["x"], 1.0), xycoords=("data", "axes fraction"),
            xytext=(dx, 9), textcoords="offset points",
            ha=ha, va="bottom", fontproperties=SANS, fontsize=12.5,
            fontweight="bold", color=MUTED, annotation_clip=False,
        )

    xs = [point["x"] for point in points]
    for k in KS:
        style = K_STYLE[k]
        ys = [point[k] for point in points]
        ax.plot(
            xs, ys, "-", color=style["color"], linewidth=3.0,
            solid_capstyle="round", zorder=3,
        )
        ax.plot(
            xs, ys, "o", color=style["color"], markersize=12.5,
            markeredgecolor=CARD, markeredgewidth=2.0, zorder=4,
            label=style["label"],
        )

    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(YTICKS)
    ax.grid(axis="y", color=GRID, linewidth=1.1, linestyle=(0, (2, 4)), zorder=0)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(2.0)

    ax.tick_params(axis="both", length=0, pad=8, labelsize=13.5, colors=INK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(SANS)
    ax.set_title(model, fontproperties=SANS, fontsize=18, fontweight="bold", color=INK, pad=34)

    return ax.set_ylabel(
        "Student pass@k (%)", fontproperties=SANS, fontsize=15,
        fontweight="bold", color=INK, labelpad=30,
    )


def arrow_style() -> dict:
    return {
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


def build_figure(series: dict[str, list[dict]]):
    fig = plt.figure(figsize=(12.0, 13.0), dpi=200, facecolor=CANVAS)
    card = FancyBboxPatch(
        (0.012, 0.014), 0.976, 0.972,
        boxstyle="round,pad=0,rounding_size=0.015",
        transform=fig.transFigure, facecolor=CARD, edgecolor="none", zorder=-10,
    )
    fig.patches.append(card)

    fig.text(
        0.5, 0.970, "Self-teacher uncertainty vs student accuracy",
        ha="center", va="center", fontproperties=SANS, fontsize=25,
        fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.938, "Students distilled on DeepMath, evaluated on AIME24",
        ha="center", va="center", fontproperties=SERIF, fontsize=15, color="#4A4844",
    )

    xlim, xticks = axis_bounds(series)
    left, width = 0.115, 0.845
    bottom, height, vgap = 0.165, 0.270, 0.150
    axes = []
    for index, model in enumerate(MODELS):
        ax = fig.add_axes([
            left, bottom + (len(MODELS) - 1 - index) * (height + vgap), width, height,
        ], zorder=5)
        ax.set_facecolor("none")
        ylabel = draw_panel(ax, model, series[model], xlim, xticks)
        axes.append((ax, ylabel))

    xlabel_y = 0.078
    xlabel = fig.text(
        0.5, xlabel_y, "Teacher uncertainty — mean CoT epistemic markers",
        ha="center", va="center", fontproperties=SANS, fontsize=16,
        fontweight="bold", color=INK,
    )

    # Direction arrows: one beside each y-label, one above the shared x-label. Both are
    # measured against their label so the arrow spans exactly the text.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, ylabel in axes:
        box = ylabel.get_window_extent(renderer)
        to_axes = ax.transAxes.inverted()
        low = to_axes.transform((box.x0, box.y0))[1]
        high = to_axes.transform((box.x1, box.y1))[1]
        ax.annotate("", xy=(-0.058, high), xytext=(-0.058, low), **arrow_style())

    box = xlabel.get_window_extent(renderer)
    to_figure = fig.transFigure.inverted()
    left_edge = to_figure.transform((box.x0, box.y0))[0]
    right_edge = to_figure.transform((box.x1, box.y1))[0]
    style = arrow_style()
    style["xycoords"] = style["textcoords"] = "figure fraction"
    axes[-1][0].annotate(
        "", xy=(right_edge, xlabel_y + 0.030),
        xytext=(left_edge, xlabel_y + 0.030), **style,
    )

    handles = [
        plt.Line2D([], [], color=K_STYLE[k]["color"], linewidth=3.0, marker="o",
                   markersize=10, markeredgecolor=CARD, markeredgewidth=1.8,
                   label=K_STYLE[k]["label"])
        for k in KS
    ]
    legend = fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.010),
        frameon=True, ncol=len(KS),
        prop=FontProperties(family=["Carlito", "DejaVu Sans"], size=14),
        handletextpad=0.7, borderpad=0.8, columnspacing=2.2, handlelength=1.9,
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
    for budget in TEACHER_BUDGETS:
        print(f"########## teacher measured at {budget}")
        series = load_points(budget)
        for model, points in series.items():
            print(f"=== {model}")
            for point in points:
                scores = "  ".join(f"pass@{k}={point[k]:5.1f}" for k in KS)
                print(f"  {point['pi_mode']:8s} e_think={point['x']:6.2f}  {scores}")
        out = output_path(budget)
        figure = build_figure(series)
        figure.savefig(out, dpi=200, facecolor=CANVAS)
        plt.close(figure)
        print(f"  wrote {out.relative_to(ROOT)}\n")
