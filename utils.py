import json
import logging
import os
import threading

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# ---------------------------------------------------------------------------
# Prompts and config
# ---------------------------------------------------------------------------


MATH_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)

def format_prompt_math(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def validate_resume(resume_path: str, this_meta: dict, force: bool = False) -> None:
    """Guard a `--resume-from-checkpoint`: check the checkpoint against the original
    run's run_meta.json (written at the run root = the checkpoint's parent dir).

    Restoring weights/optimizer/scheduler/RNG works from any checkpoint, but resuming
    on the right *examples* also needs the dataset and batch config to be identical
    (shuffle_dataset uses a seeded permutation; the per-step batch boundary depends on
    the batch config). Every key in `this_meta` present in the prior meta is compared;
    a mismatch is a hard error (data would silently misalign) unless `force` downgrades
    it to a warning. A missing run_meta.json warns and skips the check.

    Shared by train_sdft / train_grpo / train_gold; each passes its own build_run_meta.
    """
    if not os.path.isdir(resume_path):
        raise FileNotFoundError(f"--resume-from-checkpoint {resume_path} is not a directory")
    if not os.path.basename(resume_path.rstrip("/")).startswith("checkpoint-"):
        print(f"  warning: {resume_path} does not look like a 'checkpoint-<step>' dir")

    meta_path = os.path.join(os.path.dirname(resume_path.rstrip("/")), "run_meta.json")
    if not os.path.isfile(meta_path):
        print(f"  warning: no run_meta.json next to the checkpoint ({meta_path}); "
              f"cannot verify hyperparameters match -- data-skip alignment is on you.")
        return

    with open(meta_path) as f:
        prior = json.load(f)
    mismatches = [
        f"    {k}: checkpoint={prior[k]!r} vs now={this_meta[k]!r}"
        for k in this_meta
        if k in prior and prior[k] != this_meta[k]
    ]
    if mismatches:
        msg = ("Resume config differs from the checkpoint's run_meta.json "
               f"({meta_path}):\n" + "\n".join(mismatches))
        if force:
            print(f"  warning (--force-resume): {msg}")
        else:
            raise ValueError(
                msg + "\n  These affect which examples the data-skip lands on. "
                "Pass the original run's args, or --force-resume to override."
            )
    else:
        print(f"  resume config matches {meta_path}")


def hint_path(model: str, dataset: str, root: str = "data/pi/hint") -> str:
    """On-disk cache for `--pi-mode hint`, keyed by dataset then model slug so hints
    are never crossed between datasets or between models (self-hint purity). Written
    by train/gen_hints.py; read by train/train_sdft.py and eval/passk_pi.py."""
    return os.path.join(root, dataset, model.rstrip("/").split("/")[-1])

# math_verify normalizes units, prioritizes a \boxed{} match, and (with
# try_extract_without_anchor=False) only extracts a properly formatted answer.
_PRED_EXTRACTION_CONFIG = [
    LatexExtractionConfig(
        normalization_config=NormalizationConfig(units=True),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _timeouts() -> tuple[int | None, int | None]:
    """(parsing_timeout, verify_timeout) for math_verify.

    math_verify's timeouts use ``signal.alarm``, which only works on the main
    thread; off the main thread they must be disabled (and the noisy parser /
    grader loggers silenced) to avoid a ValueError.
    """
    is_main = threading.current_thread() is threading.main_thread()
    if not is_main:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)
        return None, None
    return 10, 5


def _parse_pred(text: str, parsing_timeout: int | None):
    return parse(
        text or "",
        extraction_config=_PRED_EXTRACTION_CONFIG,
        extraction_mode="first_match",
        parsing_timeout=parsing_timeout,
    )


def extract_answer(text: str) -> str | None:
    """Extract the final answer from a solution/completion via math_verify.

    Prioritizes a ``\\boxed{}`` match. Returns a readable string form of the
    parsed answer, or ``None`` if nothing could be extracted.
    """
    parsing_timeout, _ = _timeouts()
    parsed = _parse_pred(text, parsing_timeout)
    # parse() returns [sympy_value, latex_string]; the trailing string form is
    # the most human-readable for reporting.
    return str(parsed[-1]) if parsed else None


def grade(response: str, gold: str) -> tuple[str | None, bool]:
    """Extract the answer from ``response`` and grade it against ``gold``.

    Uses the TRL accuracy-reward logic (math_verify ``parse`` + ``verify``).
    Returns ``(pred_str, correct)``; ``pred_str`` is ``None`` on an extraction
    failure and ``correct`` is ``False`` if the gold answer can't be parsed.
    """
    parsing_timeout, verify_timeout = _timeouts()
    # Our gold answers are bare values (e.g. "\frac{1}{2}", "(3,\pi/2)", "204").
    # Wrapping in \boxed{} lets math_verify parse every form (fractions, roots,
    # tuples, intervals, text) that a bare parse() would otherwise miss.
    gold_parsed = parse("\\boxed{" + str(gold) + "}", parsing_timeout=parsing_timeout)
    answer_parsed = _parse_pred(response, parsing_timeout)
    pred_str = str(answer_parsed[-1]) if answer_parsed else None
    if not gold_parsed:
        return pred_str, False
    correct = bool(verify(gold_parsed, answer_parsed, timeout_seconds=verify_timeout))
    return pred_str, correct


# used in train_sdft.py
def grade_answer(response: str, gold: str) -> bool:
    """Boolean equivalence check between a completion and the gold answer."""
    return grade(response, gold)[1]


# ---------------------------------------------------------------------------
# Dataset loaders – each returns list[dict] with keys:
# problem, answer, unique_id
# ---------------------------------------------------------------------------

DATASET_REGISTRY_EVAL: dict[str, callable] = {}
DATASET_REGISTRY_TRAIN: dict[str, callable] = {}


def register_dataset_eval(name):
    def wrapper(fn):
        DATASET_REGISTRY_EVAL[name] = fn
        return fn
    return wrapper

def register_dataset_train(name):
    def wrapper(fn):
        DATASET_REGISTRY_TRAIN[name] = fn
        return fn
    return wrapper


# ---------------------------------------------------------------------------
# Common training schema
#
# Every DATASET_REGISTRY_TRAIN loader returns exactly three columns:
#   question      -- the problem statement (str)
#   final_answer  -- the gold final answer (str; always present -- rows without
#                    one are dropped by the loader)
#   solution      -- a worked-solution demo (str; "" when the dataset has none
#                    for that row, e.g. ~82% of DeepScaleR)
# so the train/eval scripts are dataset-agnostic and adding a dataset is one loader.
# ---------------------------------------------------------------------------


def _nonempty(x) -> bool:
    return x is not None and len(str(x).strip()) > 0


def has_solution(row: dict) -> bool:
    """True if the row carries a non-empty worked-solution demo.

    The paths that consume `solution` -- SDFT `--pi-mode full`, train/gen_hints.py,
    and passk_pi's full arm -- filter on this. A no-op for DeepMath (its loader
    already requires a solution) but essential for DeepScaleR, where most rows have
    an answer but no solution."""
    return _nonempty(row.get("solution"))


def load_train_dataset(
    dataset: str,
    max_samples: int | None = None,
    require_solution: bool = False,
) -> "Dataset":
    """Load a registered training dataset in the common schema.

    `require_solution=True` (SDFT full / hint gen) drops rows without a `solution`
    *before* taking the `max_samples` prefix, so the caller gets `max_samples` rows
    that actually have a demo -- otherwise on DeepScaleR the prefix would be ~82%
    solution-less. For DeepMath the drop is a no-op, so the prefix is unchanged."""
    loader = DATASET_REGISTRY_TRAIN[dataset]
    if not require_solution:
        return loader(max_samples=max_samples)
    ds = loader()
    ds = ds.filter(has_solution, num_proc=4)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds

# ---------------------------------------------------------------------------
# Eval loaders
# ---------------------------------------------------------------------------

@register_dataset_eval("aime24")
def load_aime24() -> list[dict]:
    ds = load_dataset("math-ai/aime24", split="test")
    out = []
    for row in ds:
        answer = extract_answer(row["solution"]) or ""
        out.append({
            "problem": row["problem"],
            "answer": answer,
            "unique_id": f"aime24_{row['id']}",
        })
    return out


@register_dataset_eval("aime25")
def load_aime25() -> list[dict]:
    ds = load_dataset("math-ai/aime25", split="test")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "unique_id": f"aime25_{row['id']}",
        })
    return out


@register_dataset_eval("aime26")
def load_aime26() -> list[dict]:
    ds = load_dataset("MathArena/aime_2026", split="test")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "unique_id": f"aime26_{row['problem_idx']}",
        })
    return out


@register_dataset_eval("beyond_aime")
def load_beyond_aime() -> list[dict]:
    ds = load_dataset("ByteDance-Seed/BeyondAIME", split="test")
    out = []
    for idx, row in enumerate(ds):
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "unique_id": f"beyondaime_{idx}",
        })
    return out


@register_dataset_eval("math500")
def load_math500(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for row in ds:
        level = int(str(row["level"]).removeprefix("Level "))
        if levels and level not in levels:
            continue
        out.append({
            "problem": row["problem"],
            "answer": row["answer"],
            "solution": row["solution"],
            "level": level,
            "subject": row["subject"],
            "unique_id": row.get("unique_id", ""),
        })
    return out


# ---------------------------------------------------------------------------
# Train loaders
# ---------------------------------------------------------------------------


@register_dataset_train("deepmath")
def load_deepmath(
    max_samples: int | None = None,
) -> "Dataset":
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")

    # Drop rows without a final answer or where any of the three solutions is empty.
    ds = ds.filter(
        lambda x: _nonempty(x["final_answer"]) and all(
            _nonempty(x[f"r1_solution_{i}"]) for i in (1, 2, 3)
        ),
        num_proc=4,
    )

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Normalize to the common schema (solution = the first R1 trace).
    return ds.map(
        lambda row: {
            "question": row["question"],
            "final_answer": str(row["final_answer"]),
            "solution": row["r1_solution_1"],
        },
        remove_columns=ds.column_names,
    )


@register_dataset_train("deepscaler")
def load_deepscaler(
    max_samples: int | None = None,
) -> "Dataset":
    """agentica-org/DeepScaleR-Preview-Dataset: (problem, answer, solution), 40,315
    rows. Every row has an answer, but ~82% have an empty `solution` (official
    solution unavailable); those are usable for GRPO/GOLD/`--pi-mode answer` and
    dropped by `require_solution` for the solution-consuming paths (~7.4k remain)."""
    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")

    ds = ds.filter(lambda x: _nonempty(x["answer"]), num_proc=4)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds.map(
        lambda row: {
            "question": row["problem"],
            "final_answer": str(row["answer"]),
            "solution": row["solution"] or "",
        },
        remove_columns=ds.column_names,
    )


@register_dataset_train("openthoughts")
def load_openthoughts(
    max_samples: int | None = None,
    seed: int = 42,
) -> "Dataset":
    """Load OpenThoughts-114k (metadata subset), filtered to math domain.

    Maps deepseek_reasoning -> solution, extracts boxed answer from
    deepseek_solution (falls back to ground_truth_solution).
    """
    ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")

    # Filter to math domain
    ds = ds.filter(lambda x: x["domain"] == "math", num_proc=4)

    # Map to the common schema (question, final_answer, solution)
    def _map_columns(example):
        # Try extracting boxed answer from deepseek_solution first
        answer = extract_answer(example["deepseek_solution"] or "")
        if not answer and example.get("ground_truth_solution"):
            answer = extract_answer(example["ground_truth_solution"])
        return {
            "question": example["problem"],
            "solution": example["deepseek_reasoning"],
            "final_answer": answer or "",
        }

    ds = ds.map(_map_columns, remove_columns=ds.column_names, num_proc=4)

    # Drop rows with empty solution or answer
    ds = ds.filter(
        lambda x: _nonempty(x["solution"]) and _nonempty(x["final_answer"]),
        num_proc=4,
    )

    ds = ds.shuffle(seed=seed)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds
