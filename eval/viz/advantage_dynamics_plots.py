"""Plots for the SDFT advantage-dynamics sweep.

Reads results/advantage_dynamics/<model>/<dataset>_<pi>/dynamics.json and writes:

  advantage_signal_vs_drift.png     A[training PI] and A[none] per checkpoint. A[none] is
                                    the drift control: -KL(pi_k || pi_0), zero at step 0.
  advantage_response_length.png     Mean tokens per unprivileged student rollout.

Each run keeps its own 128-problem cohort as recorded. `full` requires a reference
solution so its cohort overlaps the others by only 114 (1.7B) / 111 (4B) problems;
absolute levels are therefore not strictly comparable across PI at a given step.

Usage:
    .venv/bin/python eval/viz/advantage_dynamics_plots.py
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
DYNAMICS = RESULTS / "advantage_dynamics"
FIGURES = RESULTS / "figures"  # generated output; not tracked by git

MODELS = ["Qwen3-1.7B", "Qwen3-4B"]
DATASET = "deepmath"
PI_MODES = ["hint", "answer", "rollout", "full"]

# -------------------- style --------------------

CANVAS = "#FBF9F4"
CARD = "#FFFFFF"
INK = "#191917"
MUTED = "#6E6C66"
GRID = "#E8E5DE"
DRIFT = "#8C8880"

SANS = FontProperties(family=["Carlito", "DejaVu Sans"])
SERIF = FontProperties(family=["Caladea", "DejaVu Serif"])

# Same PI palette as sdft_passk_bars.py.
PI_COLOR = {
    "hint": "#3D74D0",
    "answer": "#E9B02E",
    "rollout": "#E0673C",
    "full": "#49B083",
}


def new_figure(width: float, height: float, title: str, subtitle: str):
    fig = plt.figure(figsize=(width, height), dpi=200, facecolor=CANVAS)
    fig.patches.append(FancyBboxPatch(
        (0.012, 0.016), 0.976, 0.968,
        boxstyle="round,pad=0,rounding_size=0.016",
        transform=fig.transFigure, facecolor=CARD, edgecolor="none", zorder=-10,
    ))
    fig.text(
        0.5, 0.955, title, ha="center", va="center",
        fontproperties=SANS, fontsize=24, fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.912, subtitle, ha="center", va="center",
        fontproperties=SERIF, fontsize=15, color="#4A4844",
    )
    return fig


def style_axes(ax, title: str | None = None) -> None:
    ax.set_facecolor("none")
    ax.grid(axis="y", color=GRID, linewidth=1.1, linestyle=(0, (2, 4)), zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.8)
    ax.tick_params(axis="both", length=0, pad=7, labelsize=12, colors=INK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(SANS)
    if title:
        ax.set_title(
            title, fontproperties=SANS, fontsize=15,
            fontweight="bold", color=INK, pad=12,
        )


def add_legend(fig, handles, labels, anchor=(0.5, 0.012), ncol=None):
    legend = fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=anchor,
        frameon=True, ncol=ncol or len(labels),
        prop=FontProperties(family=["Carlito", "DejaVu Sans"], size=13),
        handletextpad=0.7, borderpad=0.75, columnspacing=2.0, handlelength=1.8,
    )
    frame = legend.get_frame()
    frame.set_facecolor("#FDFCF9")
    frame.set_edgecolor("#DCD8CE")
    frame.set_linewidth(1.1)
    frame.set_boxstyle("round,pad=0.42,rounding_size=0.12")
    for text in legend.get_texts():
        text.set_color(INK)
    return legend


# -------------------- data --------------------


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def run_dir(model: str, pi_mode: str) -> Path:
    path = DYNAMICS / model / f"{DATASET}_{pi_mode}"
    if not path.is_dir():
        raise FileNotFoundError(f"No advantage-dynamics run at {path}")
    return path


def load_advantages(model: str, pi_mode: str) -> dict:
    """Per-step advantage statistics straight from dynamics.json."""
    dynamics = read_json(run_dir(model, pi_mode) / "dynamics.json")
    if dynamics["training_pi_mode"] != pi_mode:
        raise ValueError(
            f"{model}/{pi_mode}: dynamics.json records training PI "
            f"{dynamics['training_pi_mode']!r}"
        )
    rows = []
    for entry in dynamics["steps"]:
        signal = entry["advantages"][pi_mode]
        drift = entry["advantages"]["none"]
        outcome = signal["by_outcome"]
        rows.append({
            "step": int(entry["step"]),
            "signal": signal["mean_advantage_per_token"],
            "signal_ci": signal["question_cluster_ci95"],
            "drift": drift["mean_advantage_per_token"],
            "drift_ci": drift["question_cluster_ci95"],
            "mean_tokens": signal["num_tokens"] / signal["num_rollouts"],
            # Kept for a future correct-vs-incorrect panel; not plotted today.
            "correct": outcome["correct"]["mean_advantage_per_token"],
            "incorrect": outcome["incorrect"]["mean_advantage_per_token"],
        })
    rows.sort(key=lambda row: row["step"])
    return rows


def load_all() -> dict:
    return {
        model: {pi_mode: load_advantages(model, pi_mode) for pi_mode in PI_MODES}
        for model in MODELS
    }


# -------------------- figure 1: signal vs drift --------------------


PANELS = [
    ("drift", "Drift from base"),
    ("signal", "Self-teacher signal"),
]


def figure_signal_vs_drift(data: dict):
    fig = new_figure(
        14.4, 9.0,
        "SDFT policy drift and self-teacher signal",
        "Token-weighted advantage per checkpoint",
    )
    left, width, hgap = 0.098, 0.395, 0.070
    bottom, height, vgap = 0.150, 0.285, 0.115

    for row_index, model in enumerate(MODELS):
        for col_index, (key, label) in enumerate(PANELS):
            ax = fig.add_axes([
                left + col_index * (width + hgap),
                bottom + (len(MODELS) - 1 - row_index) * (height + vgap),
                width, height,
            ], zorder=5)
            for pi_mode in PI_MODES:
                rows = data[model][pi_mode]
                steps = [row["step"] for row in rows]
                ax.fill_between(
                    steps,
                    [row[f"{key}_ci"][0] for row in rows],
                    [row[f"{key}_ci"][1] for row in rows],
                    color=PI_COLOR[pi_mode], alpha=0.16, linewidth=0,
                )
                ax.plot(
                    steps, [row[key] for row in rows], "-o", color=PI_COLOR[pi_mode],
                    linewidth=2.5, markersize=4.5, label=pi_mode, zorder=4,
                )
            ax.axhline(0, color=INK, linewidth=1.0, alpha=0.35, zorder=1)
            ax.set_xlim(-8, 208)
            ax.set_xticks([0, 50, 100, 150, 200])
            style_axes(ax, f"{model} · {label}")
            if col_index == 0:
                ax.set_ylabel(
                    "Mean advantage / token", fontproperties=SANS,
                    fontsize=13, fontweight="bold", color=INK, labelpad=8,
                )

    fig.text(
        0.5, 0.092, "Training step", ha="center", va="center",
        fontproperties=SANS, fontsize=15, fontweight="bold", color=INK,
    )
    handles = [
        Line2D([], [], color=PI_COLOR[pi], linewidth=2.6, marker="o", markersize=6, label=pi)
        for pi in PI_MODES
    ]
    add_legend(fig, handles, PI_MODES, anchor=(0.5, 0.012))
    return fig


# -------------------- figure 2: response length --------------------


def figure_response_length(data: dict):
    fig = new_figure(
        11.5, 10.6,
        "Student response length under self-distillation",
        "Mean tokens per unprivileged student rollout on DeepMath",
    )
    left, width = 0.115, 0.845
    bottom, height, vgap = 0.155, 0.300, 0.105
    for index, model in enumerate(MODELS):
        ax = fig.add_axes([
            left, bottom + (len(MODELS) - 1 - index) * (height + vgap), width, height,
        ], zorder=5)
        for pi_mode in PI_MODES:
            rows = data[model][pi_mode]
            ax.plot(
                [row["step"] for row in rows], [row["mean_tokens"] for row in rows],
                "-o", color=PI_COLOR[pi_mode], linewidth=2.6, markersize=5,
                label=pi_mode, zorder=4,
            )
        ax.set_xlim(-8, 208)
        ax.set_xticks([0, 50, 100, 150, 200])
        # From zero, so the ~3x collapse reads at its true size.
        ax.set_ylim(0, 7400)
        ax.set_yticks([0, 2000, 4000, 6000])
        style_axes(ax, model)
        ax.set_ylabel(
            "Mean response length (tokens)", fontproperties=SANS,
            fontsize=13, fontweight="bold", color=INK, labelpad=8,
        )

    fig.text(
        0.5, 0.095, "Training step", ha="center", va="center",
        fontproperties=SANS, fontsize=15, fontweight="bold", color=INK,
    )
    handles = [
        Line2D([], [], color=PI_COLOR[pi], linewidth=2.6, marker="o", markersize=6, label=pi)
        for pi in PI_MODES
    ]
    add_legend(fig, handles, PI_MODES, anchor=(0.5, 0.012))
    return fig


if __name__ == "__main__":
    mpl.rcParams["savefig.facecolor"] = CANVAS
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_all()

    for model in MODELS:
        print(f"=== {model}")
        for pi_mode in PI_MODES:
            first, last = data[model][pi_mode][0], data[model][pi_mode][-1]
            print(
                f"  {pi_mode:8s} A[pi] {first['signal']:+.4f} -> {last['signal']:+.4f}   "
                f"A[none] {first['drift']:+.4f} -> {last['drift']:+.4f}   "
                f"len {first['mean_tokens']:.0f} -> {last['mean_tokens']:.0f}"
            )

    for name, builder in (
        ("advantage_signal_vs_drift", figure_signal_vs_drift),
        ("advantage_response_length", figure_response_length),
    ):
        figure = builder(data)
        out = FIGURES / f"{name}.png"
        figure.savefig(out, dpi=200, facecolor=CANVAS)
        plt.close(figure)
        print(f"wrote {out.relative_to(ROOT)}")
