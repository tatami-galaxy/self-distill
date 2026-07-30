# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (.venv)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training and evaluation results
#
# This notebook is the first, deliberately descriptive view of the repository's results. It scans the JSON summaries each time it runs, so new models, checkpoints, variants, and runs appear without manually copying numbers into the notebook.
#
# Data conventions:
#
# - Data source : JSON files under `results/`
# - Tables below are tidy views derived from those files; no result is re-entered by hand.
# - AIME24 baselines are matched by exact model name. Training curves retain algorithm, variant, run, and checkpoint.
# - The summary table uses the **latest available checkpoint per arm**, not the best checkpoint, to avoid implicit checkpoint selection.
# - Older AIME24 summaries do not contain `arm` or `eval_config`; their identities are recovered from the directory layout and are marked as schema 0.
# - Scores are shown on a 0–1 scale in tables and as percentages in plots.
#

# %%
"""Interactive results overview for Zed's Python REPL.

After editing this file, synchronize it with the shareable notebook:

    .venv/bin/jupytext --sync results/viz/results_overview.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.ticker import PercentFormatter

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 120)
plt.style.use("seaborn-v0_8-whitegrid")


def find_repo_root(start: Path | None = None) -> Path:
    """Work when launched from the repo root, results/viz, or a parent directory."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "results").is_dir() and (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError("Could not find a repository root containing results/ and train/")


ROOT = find_repo_root()
RESULTS = ROOT / "results"
print(f"Repository: {ROOT}")
print(f"Results:    {RESULTS}")


# %% [markdown]
# ## AIME24
#
# For the initial overview, checkpoint curves are more informative than a single bar per method: they show training dynamics and make clear which checkpoint is being compared. Each panel uses the exact base-model evaluation as a horizontal reference. Variants and repeated runs remain separate series.
#

# %%
AIME_DIR = RESULTS / "aime24"


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def checkpoint_number(step: str | None) -> int | None:
    if not step:
        return None
    match = re.fullmatch(r"checkpoint-(\d+)", step)
    return int(match.group(1)) if match else None


def fallback_aime_arm(path: Path) -> dict:
    """Recover arm identity from the path used by schema-0 summaries."""
    parts = path.relative_to(AIME_DIR).parts[:-1]  # drop summary.json
    if len(parts) == 2 and parts[0] == "base":
        return {
            "algo": "base",
            "model": parts[1],
            "train_dataset": None,
            "variant": None,
            "run": None,
            "step": "base",
        }

    if len(parts) < 4:
        raise ValueError(f"Unrecognized AIME24 result path: {path}")

    train_dataset, model, algo = parts[:3]
    step = parts[-1]
    extras = list(parts[3:-1])
    run_parts = [part for part in extras if re.fullmatch(r"run-\d+", part)]
    variant_parts = [part for part in extras if part not in run_parts]
    return {
        "algo": algo,
        "model": model,
        "train_dataset": train_dataset,
        "variant": "/".join(variant_parts) or None,
        "run": run_parts[0] if run_parts else None,
        "step": step,
    }


def load_aime24() -> pd.DataFrame:
    rows = []
    for path in sorted(AIME_DIR.rglob("summary.json")):
        summary = read_json(path)
        fallback = fallback_aime_arm(path)
        arm = {**fallback, **(summary.get("arm") or {})}
        eval_config = summary.get("eval_config") or {}
        sampling = eval_config.get("sampling") or {}
        row = {
            **arm,
            "checkpoint": checkpoint_number(arm.get("step")),
            "is_base": arm.get("algo") == "base",
            "schema_version": summary.get("schema_version", 0),
            "dataset_size": summary.get("dataset_size"),
            "n_samples": summary.get("n_samples"),
            "max_tokens": summary.get("max_tokens"),
            "temperature": sampling.get("temperature"),
            "top_p": sampling.get("top_p"),
            "top_k": sampling.get("top_k"),
            "generation_seed": sampling.get("seed"),
            "extraction_failures": summary.get("extraction_failures"),
            "total_samples": summary.get("total_samples"),
            "elapsed_s": summary.get("elapsed_s"),
            "source": str(path.relative_to(ROOT)),
        }
        row.update(summary.get("pass_at_k") or {})
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ("variant", "run", "train_dataset"):
        frame[column] = frame[column].where(frame[column].notna(), None)
    return frame.sort_values(
        ["model", "is_base", "algo", "variant", "run", "checkpoint"],
        na_position="first",
    ).reset_index(drop=True)


aime = load_aime24()
print(f"Loaded {len(aime)} AIME24 summaries across {aime['model'].nunique()} models.")
display(
    aime.groupby(
        ["schema_version", "dataset_size", "n_samples", "max_tokens"],
        dropna=False,
    ).size().rename("summary_files").reset_index()
)


# %%
def method_label(row: pd.Series, include_run: bool = True) -> str:
    if row["algo"] == "base":
        return "Base"
    label = str(row["algo"]).upper()
    if row.get("variant"):
        label += f" [{row['variant']}]"
    if include_run and row.get("run"):
        label += f" · {row['run']}"
    return label


def latest_aime_arms(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[frame["is_base"]].copy()
    trained = frame[~frame["is_base"]].copy()
    trained["variant_key"] = trained["variant"].fillna("")
    trained["run_key"] = trained["run"].fillna("")
    group_columns = ["model", "train_dataset", "algo", "variant_key", "run_key"]
    trained = (
        trained.sort_values("checkpoint")
        .groupby(group_columns, as_index=False, dropna=False)
        .tail(1)
    )
    latest = pd.concat([base, trained], ignore_index=True)
    latest["arm"] = latest.apply(method_label, axis=1)
    return latest.sort_values(["model", "is_base", "algo", "variant", "run"])


aime_latest = latest_aime_arms(aime)
latest_columns = [
    "model", "arm", "train_dataset", "step", "pass@1", "pass@8", "pass@16",
    "dataset_size", "n_samples", "max_tokens", "schema_version",
]
display(
    aime_latest[latest_columns]
    .style.format({"pass@1": "{:.3f}", "pass@8": "{:.3f}", "pass@16": "{:.3f}"})
    .background_gradient(subset=["pass@1", "pass@8", "pass@16"], cmap="Blues", vmin=0, vmax=1)
)


# %%
ALGO_COLORS = {
    "sft": "#4c78a8",
    "grpo": "#f58518",
    "gold": "#54a24b",
    "sdft": "#e45756",
    "ppo": "#7a5195",
    "ppo_val": "#9c755f",
    "ppo_pi": "#e377c2",
    "sdft_tt": "#79706e",
}
LINESTYLES = ["-", "--", "-.", ":"]


def plot_aime_curves(frame: pd.DataFrame, metric: str = "pass@1", columns: int = 2):
    models = sorted(frame["model"].unique())
    rows = math.ceil(len(models) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 4.0 * rows), squeeze=False)

    for ax, model in zip(axes.flat, models):
        model_frame = frame[frame["model"] == model]
        base = model_frame[model_frame["is_base"]]
        if not base.empty:
            value = base.iloc[-1][metric]
            ax.axhline(value, color="black", linewidth=1.6, linestyle="--", label=f"Base ({value:.3f})")

        trained = model_frame[~model_frame["is_base"]].copy()
        trained["series"] = trained.apply(method_label, axis=1)
        for index, (series, group) in enumerate(trained.groupby("series", sort=True)):
            algo = group.iloc[0]["algo"]
            same_algo_series = sorted(trained.loc[trained["algo"] == algo, "series"].unique())
            style_index = same_algo_series.index(series) % len(LINESTYLES)
            group = group.sort_values("checkpoint")
            ax.plot(
                group["checkpoint"], group[metric], marker="o", markersize=4,
                linewidth=1.8, linestyle=LINESTYLES[style_index],
                color=ALGO_COLORS.get(algo), label=series,
            )

        ax.set_title(model)
        ax.set_xlabel("Training checkpoint")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="best")

    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    fig.suptitle(f"AIME24 {metric} over training", fontsize=16, y=1.005)
    fig.tight_layout()
    return fig


plot_aime_curves(aime, "pass@1");


# %% [markdown]
# The table below is a compact model × arm view of the same latest-checkpoint data. Blank cells mean that arm has not yet been evaluated for that model. Runs remain separate columns; no averaging across seeds or runs is done at this stage.
#

# %%
latest_matrix = aime_latest.pivot(index="model", columns="arm", values="pass@1")
display(
    latest_matrix.style
    .format("{:.3f}", na_rep="—")
    .background_gradient(cmap="Blues", vmin=0, vmax=1)
)


# %% [markdown]
# ## Privileged-information pass@k
#
# These results measure how the same problem-solving model behaves when generation itself is conditioned on no PI, a hint, the answer, or the full solution. The raw pass@k values are shown directly; improvements relative to `none` can be added when deeper comparative analysis begins.
#

# %%
PASSK_PI_DIR = RESULTS / "passk_pi"
PI_ORDER = ["none", "hint", "answer", "full"]
PI_COLORS = {
    "none": "#9d9d9d",
    "hint": "#4c78a8",
    "answer": "#f58518",
    "full": "#54a24b",
}


def load_passk_pi() -> pd.DataFrame:
    rows = []
    for path in sorted(PASSK_PI_DIR.glob("*/passk_pi_summary.json")):
        summary = read_json(path)
        model = summary["model"].split("/")[-1]
        for pi_mode, metrics in summary["pass_at_k"].items():
            rows.append({
                "model": model,
                "model_id": summary["model"],
                "pi_mode": pi_mode,
                "n_problems": summary["n_problems"],
                "n_samples": summary["n_samples"],
                "max_tokens": summary["max_tokens"],
                "seed": summary["seed"],
                "source": str(path.relative_to(ROOT)),
                **metrics,
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["pi_mode"] = pd.Categorical(frame["pi_mode"], PI_ORDER, ordered=True)
        frame = frame.sort_values(["model", "pi_mode"]).reset_index(drop=True)
    return frame


passk_pi = load_passk_pi()
print(f"Loaded {len(passk_pi)} PI conditions across {passk_pi['model'].nunique()} models.")
display(
    passk_pi[["model", "pi_mode", "pass@1", "pass@8", "n_problems", "n_samples", "max_tokens", "seed"]]
    .style.format({"pass@1": "{:.3f}", "pass@8": "{:.3f}"})
    .background_gradient(subset=["pass@1", "pass@8"], cmap="Greens", vmin=0, vmax=1)
)


# %%
def grouped_bar(ax, frame: pd.DataFrame, value: str, title: str, percent: bool = False):
    models = sorted(frame["model"].unique())
    modes = [mode for mode in PI_ORDER if mode in set(frame["pi_mode"].astype(str))]
    x = np.arange(len(models), dtype=float)
    width = 0.8 / max(len(modes), 1)

    for index, mode in enumerate(modes):
        values = (
            frame[frame["pi_mode"].astype(str) == mode]
            .set_index("model")[value]
            .reindex(models)
        )
        offset = (index - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, values, width=width * 0.92, label=mode, color=PI_COLORS[mode])

    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(value)
    if percent:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title="PI", fontsize=9)


fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
grouped_bar(axes[0], passk_pi, "pass@1", "PI-conditioned generation: pass@1", percent=True)
grouped_bar(axes[1], passk_pi, "pass@8", "PI-conditioned generation: pass@8", percent=True)
fig.tight_layout()


# %% [markdown]
# ## Teacher behavior
#
# Teacher-behavior summaries describe more than answer accuracy. The first dashboard keeps differently scaled quantities on separate axes: correctness, response length, truncation, and epistemic-marker density. This makes the behavioral shift under richer PI visible without combining incompatible units.
#

# %%
TEACHER_DIR = RESULTS / "teacher_behavior"


def load_teacher_behavior() -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior_rows = []
    marker_rows = []
    for path in sorted(TEACHER_DIR.glob("*/teacher_behavior_summary.json")):
        summary = read_json(path)
        teacher = summary["teacher_model"].split("/")[-1]
        problem_model = summary["problem_model"].split("/")[-1]
        for pi_mode, metrics in summary["behavior"].items():
            common = {
                "teacher_model": teacher,
                "teacher_model_id": summary["teacher_model"],
                "problem_model": problem_model,
                "pi_mode": pi_mode,
                "n_problems": summary["n_problems"],
                "n_samples": summary["n_samples"],
                "max_tokens": summary["max_tokens"],
                "seed": summary["seed"],
                "source": str(path.relative_to(ROOT)),
            }
            behavior_rows.append({
                **common,
                **{key: value for key, value in metrics.items() if key != "e_by_marker_per_1k"},
            })
            for marker, value in metrics.get("e_by_marker_per_1k", {}).items():
                marker_rows.append({**common, "marker": marker, "per_1k_tokens": value})

    behavior = pd.DataFrame(behavior_rows)
    markers = pd.DataFrame(marker_rows)
    for frame in (behavior, markers):
        if not frame.empty:
            frame["pi_mode"] = pd.Categorical(frame["pi_mode"], PI_ORDER, ordered=True)
            frame.sort_values(["teacher_model", "pi_mode"], inplace=True)
            frame.reset_index(drop=True, inplace=True)
    return behavior, markers


teacher_behavior, teacher_markers = load_teacher_behavior()
teacher_columns = [
    "teacher_model", "problem_model", "pi_mode", "pass@1", "mean_tokens",
    "trunc_rate", "unclosed_rate", "e_per_1k_tokens", "n_completions",
]
display(
    teacher_behavior[teacher_columns]
    .style.format({
        "pass@1": "{:.3f}", "mean_tokens": "{:,.0f}", "trunc_rate": "{:.3f}",
        "unclosed_rate": "{:.3f}", "e_per_1k_tokens": "{:.2f}",
    })
)


# %%
teacher_plot = teacher_behavior.rename(columns={"teacher_model": "model"})
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
grouped_bar(axes[0, 0], teacher_plot, "pass@1", "Teacher correctness", percent=True)
grouped_bar(axes[0, 1], teacher_plot, "mean_tokens", "Mean completion length")
grouped_bar(axes[1, 0], teacher_plot, "trunc_rate", "Truncation rate", percent=True)
grouped_bar(axes[1, 1], teacher_plot, "e_per_1k_tokens", "Epistemic markers per 1k tokens")
fig.suptitle("Teacher behavior by privileged context", fontsize=16, y=1.01)
fig.tight_layout()


# %% [markdown]
# ## Result inventory and next additions
#
# The inventory below makes missing result families explicit and provides a quick check after adding files. Natural next analysis layers are:
#
# 1. paired, per-problem uncertainty estimates from `results.json` rather than treating completions as independent;
# 2. aggregation across repeated training seeds, with mean and confidence intervals;
# 3. latest-versus-best checkpoint reporting and checkpoint-selection rules;
# 4. deltas from each exact base model and learning-curve summaries;
# 5. response-length, extraction-failure, and per-problem difficulty analysis; and
# 6. additional benchmark sections using the same tidy-table pattern.
#

# %%
inventory = pd.DataFrame([
    {
        "section": "aime24",
        "summary_files": len(aime),
        "models": aime["model"].nunique(),
        "conditions_or_arms": aime[["model", "algo", "variant", "run"]].drop_duplicates().shape[0],
    },
    {
        "section": "passk_pi",
        "summary_files": passk_pi["source"].nunique(),
        "models": passk_pi["model"].nunique(),
        "conditions_or_arms": len(passk_pi),
    },
    {
        "section": "teacher_behavior",
        "summary_files": teacher_behavior["source"].nunique(),
        "models": teacher_behavior["teacher_model"].nunique(),
        "conditions_or_arms": len(teacher_behavior),
    },
])
display(inventory)
