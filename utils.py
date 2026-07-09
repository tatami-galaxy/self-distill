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


def hint_path(model: str, root: str = "data/pi/hint") -> str:
    """On-disk cache for `--pi-mode hint`, keyed by model slug so hints generated
    by one model are never loaded for another (self-hint purity). Written by
    train/gen_hints.py, read by train/train_sdft.py."""
    return os.path.join(root, model.rstrip("/").split("/")[-1])


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

    # Drop rows where any of the three solutions is empty
    ds = ds.filter(
        lambda x: all(
            x[f"r1_solution_{i}"] is not None and len(x[f"r1_solution_{i}"].strip()) > 0
            for i in (1, 2, 3)
        ),
        num_proc=4,
    )

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds


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

    # Map to standard columns
    def _map_columns(example):
        # Try extracting boxed answer from deepseek_solution first
        answer = extract_answer(example["deepseek_solution"] or "")
        if not answer and example.get("ground_truth_solution"):
            answer = extract_answer(example["ground_truth_solution"])
        return {
            "problem": example["problem"],
            "solution": example["deepseek_reasoning"],
            "answer": answer or "",
        }

    ds = ds.map(_map_columns, remove_columns=ds.column_names, num_proc=4)

    # Drop rows with empty solution or answer
    ds = ds.filter(
        lambda x: (
            x["solution"] is not None
            and len(x["solution"].strip()) > 0
            and x["answer"] is not None
            and len(x["answer"].strip()) > 0
        ),
        num_proc=4,
    )

    ds = ds.shuffle(seed=seed)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds
