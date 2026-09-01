"""Reward terms of the hint generator through training.

The generator is trained against R(h) = a*S(h) - b*|h| - g*T(h), where

    S(h)  self-teacher accuracy under the hint  (sufficiency pass@1)
    |h|   hint length in tokens
    T(h)  transfer cost to the student, in nats per token of KL

This plots each term per generator checkpoint, read from
results/hint_gen_compare/<model>/<dataset>_a<alpha>_g<gamma>/summary.json.

`fresh_base` is drawn at step 0 -- it is the untrained generator and the summary's
designated primary control. `legacy_base` is excluded: it is a filtered historical
cache whose invalid-output rate is not observable, so it is not on the same footing.

Two views are drawn per panel. `admissible_only` (solid) drops hints that leak the
answer; `all_outputs` (dashed) keeps them. The gap between the two is the leak, which
is large at step 0 (38% of outputs) and shrinks to 14% by step 100 -- so reading only
the dashed line would credit leaked answers to S(h).

Usage:
    .venv/bin/python eval/viz/hint_gen_reward_terms.py
"""

from __future__ import annotations

import json
import re
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
COMPARE = RESULTS / "hint_gen_compare"
FIGURES = RESULTS / "figures"  # generated output; not tracked by git
OUT = FIGURES / "hint_gen_reward_terms.png"

# alpha=1, gamma=1 exists only for Qwen3-1.7B; the 4B runs are a2.5 and a3.
MODEL = "Qwen3-1.7B"
RUN = "deepmath_a1_g1"

EXCLUDED_GENERATORS = {"legacy_base"}  # historical cache, validity not observable

# -------------------- style --------------------

CANVAS = "#FBF9F4"
CARD = "#FFFFFF"
INK = "#191917"
MUTED = "#6E6C66"
GRID = "#E8E5DE"

SANS = FontProperties(family=["Carlito", "DejaVu Sans"])
SERIF = FontProperties(family=["Caladea", "DejaVu Serif"])

VIEWS = [
    ("admissible_only", "-", "admissible only"),
    ("all_outputs", "--", "all outputs"),
]

PANELS = [
    ("sufficiency", "S(h) — self-teacher pass@1", "#3D74D0"),
    ("hint_tokens", "|h| — hint length (tokens)", "#E9B02E"),
    ("transfer", "T(h) — transfer cost (nats/token)", "#E0673C"),
]


# -------------------- data --------------------


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def generator_step(name: str) -> int:
    """Map a generator name onto a training step; the untrained base sits at 0."""
    if name == "fresh_base":
        return 0
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if not match:
        raise ValueError(f"Unrecognized generator name: {name!r}")
    return int(match.group(1))


def load_run() -> tuple[list[dict], float, dict]:
    summary = read_json(COMPARE / MODEL / RUN / "summary.json")
    baseline = summary["no_hint_self_teacher"]["pass@1"]

    rows = []
    for name in summary["generator_order"]:
        if name in EXCLUDED_GENERATORS:
            continue
        entry = {"generator": name, "step": generator_step(name)}
        for view, _, _ in VIEWS:
            block = summary["generators"][name][view]
            entry[view] = {
                "sufficiency": block["sufficiency"]["pass@1"]["value"],
                "hint_tokens": block["hint_tokens"]["mean"],
                "transfer": block["transfer"]["mean_clamped_nats_per_token"],
            }
        rows.append(entry)
    rows.sort(key=lambda row: row["step"])

    match = re.fullmatch(r".*_a([\d.]+)_g([\d.]+)", RUN)
    weights = {"alpha": match.group(1), "gamma": match.group(2)} if match else {}
    return rows, baseline, weights


# -------------------- plot --------------------


def build_figure(rows: list[dict], baseline: float, weights: dict):
    fig = plt.figure(figsize=(15.0, 5.9), dpi=200, facecolor=CANVAS)
    fig.patches.append(FancyBboxPatch(
        (0.012, 0.020), 0.976, 0.960,
        boxstyle="round,pad=0,rounding_size=0.018",
        transform=fig.transFigure, facecolor=CARD, edgecolor="none", zorder=-10,
    ))
    fig.text(
        0.5, 0.940, "Hint-generator reward terms through training",
        ha="center", va="center", fontproperties=SANS, fontsize=24,
        fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.878,
        f"{MODEL} · α={weights.get('alpha', '?')}, γ={weights.get('gamma', '?')} · "
        "R(h) = αS(h) − β|h| − γT(h)",
        ha="center", va="center", fontproperties=SERIF, fontsize=15, color="#4A4844",
    )

    steps = [row["step"] for row in rows]
    left, width, hgap = 0.062, 0.270, 0.055
    for index, (key, label, colour) in enumerate(PANELS):
        ax = fig.add_axes([left + index * (width + hgap), 0.245, width, 0.500], zorder=5)
        ax.set_facecolor("none")

        if key == "sufficiency":
            ax.axhline(
                baseline, color=INK, linewidth=1.4, linestyle=(0, (3, 3)), zorder=2,
            )
            ax.annotate(
                f"no hint  {baseline:.3f}", xy=(0.98, baseline), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points", ha="right", va="bottom",
                fontproperties=SANS, fontsize=10.5, color=MUTED,
            )

        for view, style, _ in VIEWS:
            ax.plot(
                steps, [row[view][key] for row in rows], style, color=colour,
                linewidth=2.5, marker="o", markersize=5,
                markeredgecolor=CARD, markeredgewidth=1.2, zorder=4,
            )

        ax.set_xlim(-6, 106)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", color=GRID, linewidth=1.1, linestyle=(0, (2, 4)), zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK)
            ax.spines[side].set_linewidth(1.8)
        ax.tick_params(axis="both", length=0, pad=7, labelsize=12, colors=INK)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(SANS)
        ax.set_title(
            label, fontproperties=SANS, fontsize=14.5,
            fontweight="bold", color=INK, pad=12,
        )

    fig.text(
        0.5, 0.138, "Generator training step  (0 = untrained base)",
        ha="center", va="center", fontproperties=SANS, fontsize=14,
        fontweight="bold", color=INK,
    )
    handles = [
        Line2D([], [], color=MUTED, linewidth=2.4, linestyle=style, marker="o",
               markersize=6, markeredgecolor=CARD, markeredgewidth=1.2, label=name)
        for _, style, name in VIEWS
    ]
    legend = fig.legend(
        handles, [name for _, _, name in VIEWS], loc="lower center",
        bbox_to_anchor=(0.5, 0.022), frameon=True, ncol=len(VIEWS),
        prop=FontProperties(family=["Carlito", "DejaVu Sans"], size=13),
        handletextpad=0.7, borderpad=0.75, columnspacing=2.2, handlelength=2.2,
    )
    frame = legend.get_frame()
    frame.set_facecolor("#FDFCF9")
    frame.set_edgecolor("#DCD8CE")
    frame.set_linewidth(1.1)
    frame.set_boxstyle("round,pad=0.42,rounding_size=0.12")
    for text in legend.get_texts():
        text.set_color(INK)

    return fig


if __name__ == "__main__":
    mpl.rcParams["savefig.facecolor"] = CANVAS
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows, baseline, weights = load_run()
    print(f"{MODEL} / {RUN}   no-hint self-teacher pass@1 = {baseline:.4f}")
    for view, _, name in VIEWS:
        print(f"=== {name}")
        print(f"  {'step':>5}  {'S(h)':>7}  {'|h|':>7}  {'T(h)':>9}")
        for row in rows:
            values = row[view]
            print(
                f"  {row['step']:>5}  {values['sufficiency']:>7.4f}  "
                f"{values['hint_tokens']:>7.1f}  {values['transfer']:>9.5f}"
            )
    figure = build_figure(rows, baseline, weights)
    figure.savefig(OUT, dpi=200, facecolor=CANVAS)
    print(f"\nWrote {OUT.relative_to(ROOT)}")
