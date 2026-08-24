r"""Compare learned hint generators with matched base-model hints.

The only policy that changes across arms is the hint generator.  Every generated
hint is evaluated with the original frozen base model as the self-teacher and on
the same cached base-student rollouts.  The evaluator reports three deliberately
separate properties:

* compression/validity: generated token length, truncation, and invalid outputs;
* sufficiency: hinted frozen-teacher pass@k and its paired lift over no hint;
* transfer: the raw sampled reverse-KL estimate ``log p - log q_hint`` averaged
  over the same unhinted-student rollouts used by hint-generator training.

``fresh_base`` is regenerated with exactly the same budget and sampling settings
as the checkpoints and is the sole base-model control.  The existing
``data/pi/hint`` cache is used only to define the paired question cohort.

Generation, vLLM teacher sampling, and Hugging Face log-probability scoring run in
separate spawned processes.  This is a correctness boundary: initializing a vLLM
engine changes process-global Torch state, while transfer is a small difference of
two log probabilities and must be computed in a clean process.

Example (run separately for each base-model size):

    CUDA_VISIBLE_DEVICES=0 uv run python -m eval.hint_gen_compare \
      --run-dir /mnt/data/ujan/self-distill/outputs/hint_gen/Qwen3-1.7B/deepmath_a1_g1 \
      --num-problems 64 --hints-per-problem 4 \
      --teacher-rollouts 4 --k 1 4

The default ``--phase sweep`` prepares the fixed cohort, generates hints, scores
sufficiency and transfer, and writes ``summary.json``.  Every expensive stage is
provenance-checked and reusable; pass ``--force`` to replace incompatible caches.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import random
import re
import shutil
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from eval.run_eval import pass_at_k
from train.opsd.train_hint_gen.lib import invalid_hint_reason
from train.opsd.train_sdft import PI_HINT
from train.opsd.train_self_teacher.lib import per_token_logps, rollout_path
from utils import (
    DATASET_REGISTRY_TRAIN,
    compose_pi_messages,
    format_prompt_math,
    grade,
    load_hint_cache,
    load_train_dataset,
)
from utils.gen_hints import build_messages as build_hint_generator_messages

SCHEMA_VERSION = 1
METHOD = "hint_generator_comparison"
GENERATOR_IDS_RESERVED = {"fresh_base", "no_hint"}
PHASES = ("sweep", "prepare", "generate", "sufficiency", "transfer", "summarize")


# ---------------------------------------------------------------------------
# Stable identities and atomic caches
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_question_id(question_idx: int, question: str, final_answer: str) -> str:
    """Identify a source-cache row even if question text is duplicated."""
    return _sha256_text(f"{question_idx}\0{question}\0{final_answer}")[:24]


def stable_hint_id(generator_id: str, question_id: str, sample_idx: int) -> str:
    return _sha256_text(f"{generator_id}\0{question_id}\0{sample_idx}")[:24]


def fingerprint(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def model_slug(model: str) -> str:
    return model.rstrip("/").split("/")[-1].replace(os.sep, "_")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def save_dataset_atomic(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    dataset.save_to_disk(str(temporary))
    if path.exists():
        shutil.rmtree(path)
    os.rename(temporary, path)


def cache_matches(
    dataset_path: Path, meta_path: Path, config: dict, force: bool
) -> bool:
    """Return True for a reusable cache, otherwise reject or clear it."""
    exists = dataset_path.exists() or meta_path.exists()
    if dataset_path.is_dir() and meta_path.is_file():
        meta = read_json(meta_path)
        if (
            meta.get("status") == "complete"
            and meta.get("config") == config
            and not force
        ):
            print(f"Reusing {dataset_path}")
            return True
    if exists and not force:
        raise ValueError(
            f"Existing cache at {dataset_path} has different or incomplete provenance. "
            "Use --force to replace it."
        )
    if force:
        if dataset_path.is_dir():
            shutil.rmtree(dataset_path)
        if meta_path.exists():
            meta_path.unlink()
    return False


def output_root(args: argparse.Namespace) -> Path:
    return Path(
        args.output_dir or f"results/hint_gen_compare/{model_slug(args.model)}"
    ).resolve()


def cohort_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = output_root(args)
    return root / "cohort", root / "cohort_meta.json"


def hint_paths(args: argparse.Namespace, generator_id: str) -> tuple[Path, Path]:
    root = output_root(args) / "hints"
    return root / generator_id, root / f"{generator_id}_meta.json"


def score_paths(args: argparse.Namespace, kind: str) -> tuple[Path, Path]:
    root = output_root(args)
    return root / kind, root / f"{kind}_meta.json"


def cohort_fingerprint(cohort: Dataset) -> str:
    return fingerprint(
        (row["question_id"], row["question"], row["final_answer"]) for row in cohort
    )


def hint_fingerprint(hints: Iterable[dict]) -> str:
    return fingerprint(
        (
            row["hint_id"],
            row["generator_id"],
            row["question_id"],
            int(row["hint_sample_idx"]),
            row["hint"],
            list(row["hint_token_ids"]),
        )
        for row in hints
    )


# ---------------------------------------------------------------------------
# Generator/checkpoint discovery
# ---------------------------------------------------------------------------


def parse_checkpoint_spec(spec: str) -> tuple[str, str]:
    """Parse ``[LABEL=]PATH`` and return a safe generator ID plus path."""
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        path = spec
        label = Path(path.rstrip("/")).name
    if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(
            f"Invalid checkpoint label {label!r}; use letters, digits, '.', '_' or '-'."
        )
    if label in GENERATOR_IDS_RESERVED:
        raise ValueError(f"Checkpoint label {label!r} is reserved.")
    if not path:
        raise ValueError(f"Checkpoint spec {spec!r} has an empty path.")
    return label, path


def discover_checkpoints(
    run_dir: str | Path, steps: Iterable[int] | None = None
) -> dict[int, str]:
    """Discover numeric ``checkpoint-<step>`` directories in ascending order.

    This mirrors ``eval.advantage_dynamics.discover_checkpoints``. Non-numeric
    siblings such as ``checkpoint-final`` and ordinary files are ignored.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Hint-generator run directory does not exist: {root}")
    checkpoints = {}
    for path in root.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if path.is_dir() and suffix.isdigit():
            checkpoints[int(suffix)] = str(path.resolve())
    checkpoints = dict(sorted(checkpoints.items()))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-<step> directories found under {root}")
    if steps is None:
        return checkpoints
    requested = list(dict.fromkeys(int(step) for step in steps))
    missing = sorted(set(requested).difference(checkpoints))
    if missing:
        raise ValueError(
            f"Requested checkpoint steps {missing} are absent from {root}; "
            f"available steps: {list(checkpoints)}"
        )
    return {step: checkpoints[step] for step in sorted(requested)}


def load_hint_run_meta(run_dir: str | Path) -> dict:
    """Load and validate the identity fields shared by all run checkpoints."""
    root = Path(run_dir)
    meta_path = root / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No run_meta.json at {meta_path}")
    meta = read_json(meta_path)
    required = {"method", "model", "dataset"}
    missing = required.difference(meta)
    if missing:
        raise ValueError(f"{meta_path} is missing required fields {sorted(missing)}")
    if meta["method"] != "hint_gen_grpo":
        raise ValueError(
            f"{root} is method={meta['method']!r}, not a hint_gen_grpo run."
        )
    return meta


def resolve_run_configuration(args: argparse.Namespace) -> None:
    """Infer model/dataset from ``--run-dir`` and reject conflicting overrides."""
    if getattr(args, "run_dir", None):
        meta = load_hint_run_meta(args.run_dir)
        for name in ("model", "dataset"):
            explicit = getattr(args, name, None)
            if explicit is not None and explicit != meta[name]:
                raise ValueError(
                    f"--{name.replace('_', '-')}={explicit!r} conflicts with "
                    f"run_meta.json {name}={meta[name]!r}."
                )
            setattr(args, name, meta[name])
    else:
        if args.model is None:
            args.model = "Qwen/Qwen3-1.7B"
        if args.dataset is None:
            args.dataset = "deepmath"


def validate_hint_checkpoint(path: str, base_model: str, dataset: str) -> None:
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    meta_path = checkpoint.parent / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No hint-generator run_meta.json next to {checkpoint}")
    meta = read_json(meta_path)
    expected = {"method": "hint_gen_grpo", "model": base_model, "dataset": dataset}
    mismatches = {
        key: (meta.get(key), value)
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Checkpoint {checkpoint} is not from the requested hint run: {mismatches}"
        )


def generator_variants(args: argparse.Namespace) -> list[tuple[str, str]]:
    variants = [("fresh_base", args.model)]
    seen = {"fresh_base"}
    resolved = getattr(args, "resolved_checkpoints", None)
    if resolved is not None:
        checkpoint_variants = list(resolved)
    elif getattr(args, "run_dir", None):
        checkpoint_variants = [
            (f"checkpoint-{step}", path)
            for step, path in discover_checkpoints(
                args.run_dir, getattr(args, "steps", None)
            ).items()
        ]
    else:
        checkpoint_variants = [
            parse_checkpoint_spec(spec) for spec in getattr(args, "checkpoint", [])
        ]
    for label, path in checkpoint_variants:
        if label in seen:
            raise ValueError(f"Duplicate generator label {label!r}.")
        validate_hint_checkpoint(path, args.model, args.dataset)
        variants.append((label, str(Path(path).resolve())))
        seen.add(label)
    return variants


def expected_generator_ids(args: argparse.Namespace) -> list[str]:
    return [label for label, _ in generator_variants(args)]


# ---------------------------------------------------------------------------
# Fixed paired cohort
# ---------------------------------------------------------------------------


def _difficulty(successes: int, total: int) -> str:
    if successes == 0:
        return "hard"
    if successes == total:
        return "easy"
    return "intermediate"


def prepare_config(args: argparse.Namespace) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "dataset": args.dataset,
        "num_problems": args.num_problems,
        "seed": args.seed,
        "hint_root": str(Path(args.hint_root).resolve()),
        "rollout_root": str(Path(args.rollout_root).resolve()),
        "transfer_rollouts": args.transfer_rollouts,
    }


def prepare_phase(args: argparse.Namespace) -> None:
    cohort_path, meta_path = cohort_paths(args)
    config = prepare_config(args)
    if cache_matches(cohort_path, meta_path, config, args.force):
        return

    hints = load_hint_cache(args.model, args.dataset, root=args.hint_root)
    rollout_cache_path = rollout_path(args.model, args.dataset, args.rollout_root)
    if not os.path.isdir(rollout_cache_path):
        raise FileNotFoundError(f"No student rollout cache at {rollout_cache_path}")
    rollouts = load_from_disk(rollout_cache_path)
    required = {"question", "completion_ids", "gen_model", "dataset"}
    missing = required.difference(rollouts.column_names)
    if missing:
        raise ValueError(f"Student rollout cache is missing columns {sorted(missing)}")
    if set(rollouts.unique("gen_model")) != {args.model}:
        raise ValueError("Student rollout cache was generated by a different model.")
    if set(rollouts.unique("dataset")) != {args.dataset}:
        raise ValueError("Student rollout cache was generated for a different dataset.")

    rollout_indices: dict[str, list[int]] = defaultdict(list)
    sample_indices = (
        rollouts["sample_idx"]
        if "sample_idx" in rollouts.column_names
        else [0] * len(rollouts)
    )
    rollout_ids = (
        rollouts["rollout_id"]
        if "rollout_id" in rollouts.column_names
        else [""] * len(rollouts)
    )
    for row_idx, question in enumerate(rollouts["question"]):
        rollout_indices[str(question)].append(row_idx)
    for indices in rollout_indices.values():
        indices.sort(key=lambda idx: (int(sample_indices[idx]), str(rollout_ids[idx])))

    solutions = {
        str(row["question"]): str(row["solution"])
        for row in load_train_dataset(args.dataset, require_solution=True)
    }
    eligible = []
    for question_idx, row in enumerate(hints):
        question = str(row["question"])
        indices = rollout_indices.get(question, [])
        if question not in solutions or len(indices) < args.transfer_rollouts:
            continue
        selected = indices[: args.transfer_rollouts]
        rewards = (
            [float(rollouts[idx]["reward"]) for idx in selected]
            if "reward" in rollouts.column_names
            else []
        )
        successes = sum(value > 0.5 for value in rewards) if rewards else -1
        eligible.append(
            {
                "question_idx": question_idx,
                "question_id": stable_question_id(
                    question_idx, question, str(row["final_answer"])
                ),
                "question": question,
                "final_answer": str(row["final_answer"]),
                "solution": solutions[question],
                "student_rollout_successes": successes,
                "student_rollout_count": len(selected),
                "difficulty": (
                    _difficulty(successes, len(selected)) if rewards else "unknown"
                ),
            }
        )
    if not eligible:
        raise RuntimeError(
            "No hint-cache questions have both a solution and enough rollouts."
        )
    n = min(args.num_problems, len(eligible))
    selected_positions = sorted(
        random.Random(args.seed).sample(range(len(eligible)), n)
    )
    cohort_rows = [eligible[index] for index in selected_positions]
    cohort = Dataset.from_list(cohort_rows)

    save_dataset_atomic(cohort, cohort_path)
    cohort_fp = cohort_fingerprint(cohort)
    write_json_atomic(
        meta_path,
        {
            "status": "complete",
            "method": METHOD,
            "config": config,
            "cohort_fingerprint": cohort_fp,
            "num_problems": len(cohort),
            "selection": "random_subset_of_hint_cache_intersected_with_rollouts",
            "difficulty_definition": "hard=0/L, easy=L/L, intermediate=otherwise",
        },
    )
    print(f"Prepared {len(cohort)} paired problems -> {cohort_path}")


def load_cohort(args: argparse.Namespace) -> tuple[Dataset, dict]:
    path, meta_path = cohort_paths(args)
    if not path.is_dir() or not meta_path.is_file():
        raise FileNotFoundError("Run --phase prepare (or --phase sweep) first.")
    meta = read_json(meta_path)
    if meta.get("config") != prepare_config(args):
        raise ValueError("Cohort provenance differs from the requested configuration.")
    cohort = load_from_disk(str(path))
    if meta.get("cohort_fingerprint") != cohort_fingerprint(cohort):
        raise ValueError("Cohort fingerprint mismatch.")
    return cohort, meta


# ---------------------------------------------------------------------------
# Matched hint generation
# ---------------------------------------------------------------------------


def generation_config(
    args: argparse.Namespace, generator_id: str, generator_model: str, cohort_fp: str
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": generator_id,
        "generator_model": generator_model,
        "base_model": args.model,
        "dataset": args.dataset,
        "cohort_fingerprint": cohort_fp,
        "hints_per_problem": args.hints_per_problem,
        "hint_max_tokens": args.hint_max_tokens,
        "temperature": args.generator_temperature,
        "top_p": args.generator_top_p,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
    }


def generate_phase(args: argparse.Namespace) -> None:
    if not args.generator_id or not args.generator_model:
        raise ValueError(
            "Internal generate phase requires --generator-id and --generator-model."
        )
    from vllm import LLM, SamplingParams

    cohort, cohort_meta = load_cohort(args)
    out, meta_path = hint_paths(args, args.generator_id)
    config = generation_config(
        args, args.generator_id, args.generator_model, cohort_meta["cohort_fingerprint"]
    )
    if cache_matches(out, meta_path, config, args.force):
        return

    llm = LLM(
        model=args.generator_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    conversations = [
        build_hint_generator_messages(row["question"], row["solution"])
        for row in cohort
    ]
    prompt_budget = args.max_model_len - args.hint_max_tokens
    for problem, messages in zip(cohort, conversations, strict=True):
        ids = tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            enable_thinking=False,
        )["input_ids"][0]
        if len(ids) > prompt_budget:
            raise ValueError(
                f"Generator prompt for question_id={problem['question_id']} has {len(ids)} "
                f"tokens but budget is {prompt_budget}; the common cohort cannot be changed "
                "for only one generator. Increase --max-model-len."
            )
    sampling = SamplingParams(
        n=args.hints_per_problem,
        max_tokens=args.hint_max_tokens,
        temperature=args.generator_temperature,
        top_p=args.generator_top_p,
        seed=args.seed,
    )
    outputs = llm.chat(
        conversations,
        sampling,
        chat_template_kwargs={"enable_thinking": False},
    )
    rows = []
    for problem, output in zip(cohort, outputs, strict=True):
        if len(output.outputs) != args.hints_per_problem:
            raise RuntimeError("vLLM returned the wrong number of hint samples.")
        for sample_idx, candidate in enumerate(output.outputs):
            hint = candidate.text.strip()
            reason = invalid_hint_reason(hint, problem["final_answer"])
            token_ids = list(candidate.token_ids)
            rows.append(
                {
                    "hint_id": stable_hint_id(
                        args.generator_id, problem["question_id"], sample_idx
                    ),
                    "question_id": problem["question_id"],
                    "question_idx": int(problem["question_idx"]),
                    "generator_id": args.generator_id,
                    "generator_model": args.generator_model,
                    "generator_kind": "matched_generation",
                    "hint_sample_idx": sample_idx,
                    "generation_seed": args.seed,
                    "hint": hint,
                    "hint_token_ids": token_ids,
                    "num_hint_tokens": len(token_ids),
                    "length_source": "generation_token_ids",
                    "truncated": candidate.finish_reason == "length",
                    "invalid_reason": reason or "",
                    "validity_observable": True,
                }
            )
    hints = Dataset.from_list(rows)
    save_dataset_atomic(hints, out)
    write_json_atomic(
        meta_path,
        {
            "status": "complete",
            "method": METHOD,
            "config": config,
            "num_hints": len(hints),
            "hint_fingerprint": hint_fingerprint(hints),
        },
    )
    print(f"Generated {len(hints)} hints with {args.generator_id} -> {out}")
    del llm
    gc.collect()


def load_all_hints(args: argparse.Namespace) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    metadata = {}
    for generator_id in expected_generator_ids(args):
        path, meta_path = hint_paths(args, generator_id)
        if not path.is_dir() or not meta_path.is_file():
            raise FileNotFoundError(
                f"Missing hints for generator {generator_id!r}: {path}"
            )
        dataset = load_from_disk(str(path))
        meta = read_json(meta_path)
        if meta.get("hint_fingerprint") != hint_fingerprint(dataset):
            raise ValueError(f"Hint fingerprint mismatch for {generator_id!r}.")
        rows.extend(dict(row) for row in dataset)
        metadata[generator_id] = meta
    hint_ids = [row["hint_id"] for row in rows]
    if len(hint_ids) != len(set(hint_ids)):
        raise ValueError("Hint IDs are not unique across generator arms.")
    return rows, metadata


# ---------------------------------------------------------------------------
# Frozen-teacher sufficiency
# ---------------------------------------------------------------------------


def hinted_teacher_messages(question: str, hint: str) -> list[dict]:
    return compose_pi_messages(format_prompt_math(question), PI_HINT.format(hint=hint))


def sufficiency_config(args: argparse.Namespace, cohort_fp: str, hints_fp: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "base_teacher": args.model,
        "dataset": args.dataset,
        "cohort_fingerprint": cohort_fp,
        "hint_fingerprint": hints_fp,
        "teacher_rollouts": args.teacher_rollouts,
        "teacher_max_tokens": args.teacher_max_tokens,
        "temperature": args.teacher_temperature,
        "top_p": args.teacher_top_p,
        "seed": args.teacher_seed,
        "max_model_len": args.max_model_len,
        "save_teacher_samples": args.save_teacher_samples,
    }


def sufficiency_phase(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    cohort, cohort_meta = load_cohort(args)
    hints, _ = load_all_hints(args)
    hints_fp = hint_fingerprint(hints)
    out, meta_path = score_paths(args, "sufficiency")
    config = sufficiency_config(args, cohort_meta["cohort_fingerprint"], hints_fp)
    if cache_matches(out, meta_path, config, args.force):
        return

    problems = {row["question_id"]: dict(row) for row in cohort}
    conditions = []
    for problem in cohort:
        conditions.append(
            {
                "condition_id": f"no_hint:{problem['question_id']}",
                "hint_id": "",
                "question_id": problem["question_id"],
                "generator_id": "no_hint",
                "hint_sample_idx": -1,
                "messages": format_prompt_math(problem["question"]),
            }
        )
    for hint in hints:
        problem = problems[hint["question_id"]]
        conditions.append(
            {
                "condition_id": hint["hint_id"],
                "hint_id": hint["hint_id"],
                "question_id": hint["question_id"],
                "generator_id": hint["generator_id"],
                "hint_sample_idx": int(hint["hint_sample_idx"]),
                "messages": hinted_teacher_messages(problem["question"], hint["hint"]),
            }
        )

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.teacher_seed,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    prompt_budget = args.max_model_len - args.teacher_max_tokens
    prompts = []
    for condition in conditions:
        ids = tokenizer.apply_chat_template(
            condition["messages"], add_generation_prompt=True, tokenize=True
        )
        if len(ids) > prompt_budget:
            raise ValueError(
                f"Teacher prompt for {condition['condition_id']} has {len(ids)} tokens but "
                f"budget is {prompt_budget}. Increase --max-model-len or lower "
                "--teacher-max-tokens."
            )
        prompts.append(
            tokenizer.apply_chat_template(
                condition["messages"], add_generation_prompt=True, tokenize=False
            )
        )
    sampling = SamplingParams(
        n=args.teacher_rollouts,
        max_tokens=args.teacher_max_tokens,
        temperature=args.teacher_temperature,
        top_p=args.teacher_top_p,
        seed=args.teacher_seed,
    )
    outputs = llm.generate(prompts, sampling)
    rows = []
    for condition, output in zip(conditions, outputs, strict=True):
        problem = problems[condition["question_id"]]
        completions = list(output.outputs)
        correct = [
            bool(grade(candidate.text, problem["final_answer"])[1])
            for candidate in completions
        ]
        row = {
            "condition_id": condition["condition_id"],
            "hint_id": condition["hint_id"],
            "question_id": condition["question_id"],
            "generator_id": condition["generator_id"],
            "hint_sample_idx": condition["hint_sample_idx"],
            "n_samples": len(completions),
            "n_correct": sum(correct),
            "n_truncated": sum(
                candidate.finish_reason == "length" for candidate in completions
            ),
        }
        if args.save_teacher_samples:
            row["completion_texts"] = [candidate.text for candidate in completions]
            row["completion_correct"] = correct
        rows.append(row)
    scores = Dataset.from_list(rows)
    save_dataset_atomic(scores, out)
    write_json_atomic(
        meta_path,
        {
            "status": "complete",
            "method": METHOD,
            "config": config,
            "num_conditions": len(scores),
            "score_fingerprint": fingerprint(
                (row["condition_id"], row["n_samples"], row["n_correct"])
                for row in scores
            ),
        },
    )
    print(f"Scored {len(scores)} teacher conditions -> {out}")
    del llm
    gc.collect()


# ---------------------------------------------------------------------------
# Clean-process sampled reverse-KL scoring
# ---------------------------------------------------------------------------


def transfer_config(args: argparse.Namespace, cohort_fp: str, hints_fp: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "student_and_teacher_model": args.model,
        "dataset": args.dataset,
        "rollout_root": str(Path(args.rollout_root).resolve()),
        "cohort_fingerprint": cohort_fp,
        "hint_fingerprint": hints_fp,
        "transfer_rollouts": args.transfer_rollouts,
        "transfer_max_completion_tokens": args.transfer_max_completion_tokens,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "sign": "student_logp_minus_hinted_teacher_logp",
        "clamped": False,
        "rollout_aggregation": "equal_weight_after_within_rollout_token_mean",
    }


def _input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def _render_prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"][0]
    )


def _score_completion(model, prompt_ids: list[int], completion_ids: list[int]):
    import torch

    device = _input_device(model)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    completion = torch.tensor(
        completion_ids, dtype=torch.long, device=device
    ).unsqueeze(0)
    with torch.inference_mode():
        values = per_token_logps(
            model, torch.cat([prompt, completion], dim=1), completion
        )
    return values.squeeze(0).float().cpu()


def raw_sampled_transfer(student_logps, teacher_logps) -> tuple[float, float, int]:
    """Return rollout mean, log-ratio sum, and token count without KL clamping."""
    if student_logps.shape != teacher_logps.shape:
        raise ValueError("Student and hinted-teacher log-probability shapes differ.")
    if student_logps.numel() == 0:
        raise ValueError("Cannot score an empty student completion.")
    difference = student_logps.float() - teacher_logps.float()
    return (
        float(difference.mean().item()),
        float(difference.sum().item()),
        difference.numel(),
    )


def transfer_phase(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cohort, cohort_meta = load_cohort(args)
    hints, _ = load_all_hints(args)
    hints_fp = hint_fingerprint(hints)
    out, meta_path = score_paths(args, "transfer")
    config = transfer_config(args, cohort_meta["cohort_fingerprint"], hints_fp)
    if cache_matches(out, meta_path, config, args.force):
        return

    rollout_cache_path = rollout_path(args.model, args.dataset, args.rollout_root)
    rollouts = load_from_disk(rollout_cache_path)
    indices_by_question: dict[str, list[int]] = defaultdict(list)
    sample_indices = (
        rollouts["sample_idx"]
        if "sample_idx" in rollouts.column_names
        else [0] * len(rollouts)
    )
    rollout_ids = (
        rollouts["rollout_id"]
        if "rollout_id" in rollouts.column_names
        else [""] * len(rollouts)
    )
    for index, question in enumerate(rollouts["question"]):
        indices_by_question[str(question)].append(index)
    for indices in indices_by_question.values():
        indices.sort(key=lambda idx: (int(sample_indices[idx]), str(rollout_ids[idx])))

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True
        )
        .eval()
        .to("cuda")
    )
    problems = {row["question_id"]: dict(row) for row in cohort}
    selected_rollouts: dict[str, list[tuple[int, list[int], Any]]] = {}
    print(f"Scoring unhinted student log probabilities for {len(cohort)} questions")
    for problem_number, problem in enumerate(cohort, start=1):
        indices = indices_by_question.get(problem["question"], [])[
            : args.transfer_rollouts
        ]
        if len(indices) != args.transfer_rollouts:
            raise ValueError(
                f"Not enough rollouts for question_id={problem['question_id']}"
            )
        prompt_ids = _render_prompt_ids(
            tokenizer, format_prompt_math(problem["question"])
        )
        scored = []
        for rollout_position, index in enumerate(indices):
            completion_ids = list(rollouts[index]["completion_ids"])
            if args.transfer_max_completion_tokens:
                completion_ids = completion_ids[: args.transfer_max_completion_tokens]
            if len(prompt_ids) + len(completion_ids) > args.max_model_len:
                raise ValueError(
                    f"Unhinted transfer sequence exceeds --max-model-len for "
                    f"question_id={problem['question_id']}. Set "
                    "--transfer-max-completion-tokens."
                )
            student_logps = _score_completion(model, prompt_ids, completion_ids)
            scored.append((rollout_position, completion_ids, student_logps))
        selected_rollouts[problem["question_id"]] = scored
        if problem_number % 10 == 0:
            print(f"  student {problem_number}/{len(cohort)}")

    rows = []
    print(f"Scoring {len(hints)} hinted-teacher conditions")
    for hint_number, hint in enumerate(hints, start=1):
        problem = problems[hint["question_id"]]
        prompt_ids = _render_prompt_ids(
            tokenizer, hinted_teacher_messages(problem["question"], hint["hint"])
        )
        for rollout_position, completion_ids, student_logps in selected_rollouts[
            hint["question_id"]
        ]:
            if len(prompt_ids) + len(completion_ids) > args.max_model_len:
                raise ValueError(
                    f"Hinted transfer sequence exceeds --max-model-len for hint_id="
                    f"{hint['hint_id']}. Set --transfer-max-completion-tokens or increase "
                    "--max-model-len."
                )
            teacher_logps = _score_completion(model, prompt_ids, completion_ids)
            raw_mean, log_ratio_sum, n_tokens = raw_sampled_transfer(
                student_logps, teacher_logps
            )
            rows.append(
                {
                    "hint_id": hint["hint_id"],
                    "question_id": hint["question_id"],
                    "generator_id": hint["generator_id"],
                    "hint_sample_idx": int(hint["hint_sample_idx"]),
                    "rollout_position": rollout_position,
                    "raw_transfer": raw_mean,
                    "log_ratio_sum": log_ratio_sum,
                    "num_tokens": n_tokens,
                }
            )
        if hint_number % 10 == 0:
            print(f"  hinted teacher {hint_number}/{len(hints)}")
    scores = Dataset.from_list(rows)
    save_dataset_atomic(scores, out)
    write_json_atomic(
        meta_path,
        {
            "status": "complete",
            "method": METHOD,
            "config": config,
            "num_rows": len(scores),
            "score_fingerprint": fingerprint(
                (row["hint_id"], row["rollout_position"], row["raw_transfer"])
                for row in scores
            ),
        },
    )
    print(f"Saved {len(scores)} raw transfer estimates -> {out}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Question-balanced summaries and paired comparisons
# ---------------------------------------------------------------------------


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a percentile of an empty list.")
    position = probability * (len(ordered) - 1)
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def question_means(rows: Iterable[dict], value) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(float(value(row)))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def mean_question_balanced(rows: Iterable[dict], value) -> float | None:
    means = question_means(rows, value)
    return sum(means.values()) / len(means) if means else None


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> list[float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty group.")
    rng = random.Random(seed)
    estimates = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def paired_question_difference(
    candidate: dict[str, float], reference: dict[str, float], samples: int, seed: int
) -> dict:
    common = sorted(set(candidate).intersection(reference))
    if not common:
        raise ValueError("Paired comparison has no common questions.")
    deltas = [candidate[key] - reference[key] for key in common]
    return {
        "n_questions": len(common),
        "mean_delta_candidate_minus_reference": sum(deltas) / len(deltas),
        "delta_ci95": bootstrap_mean_ci(deltas, samples, seed),
    }


def _load_score_dataset(args: argparse.Namespace, kind: str) -> tuple[list[dict], dict]:
    path, meta_path = score_paths(args, kind)
    if not path.is_dir() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing {kind} scores; run --phase {kind} first.")
    return [dict(row) for row in load_from_disk(str(path))], read_json(meta_path)


def summarize_generator_subset(
    hints: list[dict],
    sufficiency_by_hint: dict[str, dict],
    no_hint_by_question: dict[str, dict],
    transfer_by_hint: dict[str, list[dict]],
    ks: list[int],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    if not hints:
        return {"n_hints": 0, "n_questions": 0}
    lengths = [float(row["num_hint_tokens"]) for row in hints]
    validity_observable = all(bool(row["validity_observable"]) for row in hints)
    invalid_fraction = (
        sum(bool(row["invalid_reason"]) for row in hints) / len(hints)
        if validity_observable
        else None
    )
    summary = {
        "n_hints": len(hints),
        "n_questions": len({row["question_id"] for row in hints}),
        "hint_tokens": describe(lengths),
        "truncated_fraction": sum(bool(row["truncated"]) for row in hints) / len(hints),
        "validity_observable": validity_observable,
        "invalid_fraction": invalid_fraction,
        "invalid_counts": dict(
            sorted(
                (reason, sum(row["invalid_reason"] == reason for row in hints))
                for reason in {
                    row["invalid_reason"] for row in hints if row["invalid_reason"]
                }
            )
        ),
    }

    pass_summary = {}
    for k in ks:
        hint_values = {}
        delta_values = {}
        for hint in hints:
            score = sufficiency_by_hint[hint["hint_id"]]
            value = pass_at_k(int(score["n_samples"]), int(score["n_correct"]), k)
            hint_values[hint["hint_id"]] = value
            none = no_hint_by_question[hint["question_id"]]
            none_value = pass_at_k(int(none["n_samples"]), int(none["n_correct"]), k)
            delta_values[hint["hint_id"]] = value - none_value
        values_by_question = question_means(
            hints,
            lambda row: hint_values[row["hint_id"]],  # noqa: B023
        )
        deltas_by_question = question_means(
            hints,
            lambda row: delta_values[row["hint_id"]],  # noqa: B023
        )
        pass_summary[f"pass@{k}"] = {
            "value": sum(values_by_question.values()) / len(values_by_question),
            "delta_over_no_hint": sum(deltas_by_question.values())
            / len(deltas_by_question),
            "delta_ci95": bootstrap_mean_ci(
                list(deltas_by_question.values()), bootstrap_samples, seed + k
            ),
        }
    summary["sufficiency"] = pass_summary
    summary["teacher_truncated_fraction"] = mean_question_balanced(
        hints,
        lambda row: (
            sufficiency_by_hint[row["hint_id"]]["n_truncated"]
            / sufficiency_by_hint[row["hint_id"]]["n_samples"]
        ),
    )

    subset_hint_ids = {row["hint_id"] for row in hints}
    per_hint_transfer = {
        hint_id: sum(float(row["raw_transfer"]) for row in transfer_by_hint[hint_id])
        / len(transfer_by_hint[hint_id])
        for hint_id in subset_hint_ids
    }
    transfer_question_means = question_means(
        hints, lambda row: per_hint_transfer[row["hint_id"]]
    )
    summary["transfer"] = {
        "mean_raw_nats_per_token": (
            sum(transfer_question_means.values()) / len(transfer_question_means)
        ),
        "mean_clamped_nats_per_token": mean_question_balanced(
            hints, lambda row: max(0.0, per_hint_transfer[row["hint_id"]])
        ),
        "negative_hint_fraction": sum(value < 0 for value in per_hint_transfer.values())
        / len(per_hint_transfer),
        "aggregation": "tokens_within_rollout_then_rollouts_within_hint_then_hints_within_question",
    }
    return summary


def metric_maps(
    hints: list[dict],
    sufficiency_by_hint: dict[str, dict],
    transfer_by_hint: dict[str, list[dict]],
    ks: list[int],
) -> dict[str, dict[str, float]]:
    per_hint_transfer = {
        hint_id: sum(float(row["raw_transfer"]) for row in rows) / len(rows)
        for hint_id, rows in transfer_by_hint.items()
    }
    maps = {
        "mean_hint_tokens": question_means(hints, lambda row: row["num_hint_tokens"]),
        "invalid_fraction": question_means(
            hints, lambda row: bool(row["invalid_reason"])
        ),
        "raw_transfer_nats_per_token": question_means(
            hints, lambda row: per_hint_transfer[row["hint_id"]]
        ),
    }
    for k in ks:
        maps[f"pass@{k}"] = question_means(
            hints,
            lambda row, k=k: pass_at_k(
                int(sufficiency_by_hint[row["hint_id"]]["n_samples"]),
                int(sufficiency_by_hint[row["hint_id"]]["n_correct"]),
                k,
            ),
        )
    return maps


def summarize_phase(args: argparse.Namespace) -> dict:
    cohort, cohort_meta = load_cohort(args)
    hints, hint_meta = load_all_hints(args)
    sufficiency, suff_meta = _load_score_dataset(args, "sufficiency")
    transfer, transfer_meta = _load_score_dataset(args, "transfer")

    sufficiency_by_hint = {
        row["hint_id"]: row for row in sufficiency if row["generator_id"] != "no_hint"
    }
    no_hint_by_question = {
        row["question_id"]: row
        for row in sufficiency
        if row["generator_id"] == "no_hint"
    }
    transfer_by_hint: dict[str, list[dict]] = defaultdict(list)
    for row in transfer:
        transfer_by_hint[row["hint_id"]].append(row)
    expected_hint_ids = {row["hint_id"] for row in hints}
    if set(sufficiency_by_hint) != expected_hint_ids:
        raise ValueError("Sufficiency scores do not cover exactly the generated hints.")
    if set(transfer_by_hint) != expected_hint_ids:
        raise ValueError("Transfer scores do not cover exactly the generated hints.")
    if set(no_hint_by_question) != set(cohort["question_id"]):
        raise ValueError("No-hint sufficiency scores do not cover exactly the cohort.")

    difficulty_by_question = {row["question_id"]: row["difficulty"] for row in cohort}
    by_generator: dict[str, list[dict]] = defaultdict(list)
    for row in hints:
        by_generator[row["generator_id"]].append(row)

    no_hint_pass = {}
    for k in args.k:
        values = [
            pass_at_k(int(row["n_samples"]), int(row["n_correct"]), k)
            for row in no_hint_by_question.values()
        ]
        no_hint_pass[f"pass@{k}"] = sum(values) / len(values)

    generator_summaries = {}
    metric_maps_by_generator = {}
    for generator_index, generator_id in enumerate(expected_generator_ids(args)):
        generator_hints = by_generator[generator_id]
        all_outputs = summarize_generator_subset(
            generator_hints,
            sufficiency_by_hint,
            no_hint_by_question,
            transfer_by_hint,
            args.k,
            args.bootstrap_samples,
            args.seed + generator_index * 1000,
        )
        admissible_hints = [row for row in generator_hints if not row["invalid_reason"]]
        admissible = summarize_generator_subset(
            admissible_hints,
            sufficiency_by_hint,
            no_hint_by_question,
            transfer_by_hint,
            args.k,
            args.bootstrap_samples,
            args.seed + generator_index * 1000 + 100,
        )
        difficulty_summaries = {}
        for difficulty in ("hard", "intermediate", "easy", "unknown"):
            subset = [
                row
                for row in generator_hints
                if difficulty_by_question[row["question_id"]] == difficulty
            ]
            if subset:
                difficulty_summaries[difficulty] = summarize_generator_subset(
                    subset,
                    sufficiency_by_hint,
                    no_hint_by_question,
                    transfer_by_hint,
                    args.k,
                    args.bootstrap_samples,
                    args.seed + generator_index * 1000 + 200,
                )
        generator_summaries[generator_id] = {
            "all_outputs": all_outputs,
            "admissible_only": admissible,
            "by_student_difficulty": difficulty_summaries,
        }
        metric_maps_by_generator[generator_id] = metric_maps(
            generator_hints, sufficiency_by_hint, transfer_by_hint, args.k
        )

    paired = {}
    reference = "fresh_base"
    for candidate_index, candidate in enumerate(expected_generator_ids(args)):
        if candidate == reference:
            continue
        label = f"{candidate}_minus_{reference}"
        paired[label] = {
            "candidate": candidate,
            "reference": reference,
            "sign": "candidate_minus_reference",
            "metrics": {
                metric: paired_question_difference(
                    metric_maps_by_generator[candidate][metric],
                    metric_maps_by_generator[reference][metric],
                    args.bootstrap_samples,
                    args.seed + candidate_index * 10_000 + metric_index,
                )
                for metric_index, metric in enumerate(
                    metric_maps_by_generator[candidate]
                )
            },
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "model": args.model,
        "dataset": args.dataset,
        "n_problems": len(cohort),
        "generator_order": expected_generator_ids(args),
        "no_hint_self_teacher": no_hint_pass,
        "generators": generator_summaries,
        "paired_comparisons": paired,
        "provenance": {
            "cohort": cohort_meta,
            "hints": hint_meta,
            "sufficiency": suff_meta,
            "transfer": transfer_meta,
        },
        "notes": {
            "primary_control": "fresh_base",
            "transfer_sign": "positive means hinted teacher is farther from the student",
            "uncertainty_unit": "paired question-level bootstrap",
        },
    }
    path = output_root(args) / "summary.json"
    write_json_atomic(path, summary)
    print(f"Saved comparison summary -> {path}")
    for generator_id in expected_generator_ids(args):
        metrics = generator_summaries[generator_id]["all_outputs"]
        suff = "  ".join(
            f"pass@{k}={metrics['sufficiency'][f'pass@{k}']['value']:.3f}"
            for k in args.k
        )
        print(
            f"  {generator_id:20s} len={metrics['hint_tokens']['mean']:.1f}  "
            f"T={metrics['transfer']['mean_raw_nats_per_token']:+.4f}  {suff}"
        )
    return summary


# ---------------------------------------------------------------------------
# Process orchestration and CLI
# ---------------------------------------------------------------------------


def _phase_worker(payload: dict, phase: str, extras: dict | None = None) -> None:
    values = dict(payload)
    values["phase"] = phase
    if extras:
        values.update(extras)
    dispatch(argparse.Namespace(**values))


def _spawn_phase(
    args: argparse.Namespace, phase: str, extras: dict | None = None
) -> None:
    print(
        f"\n=== {phase}" + (f" {extras.get('generator_id')}" if extras else "") + " ==="
    )
    process = multiprocessing.get_context("spawn").Process(
        target=_phase_worker, args=(vars(args), phase, extras)
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"Phase {phase!r} failed with exit code {process.exitcode}.")


def sweep_phase(args: argparse.Namespace) -> None:
    prepare_phase(args)
    for generator_id, generator_model in generator_variants(args):
        _spawn_phase(
            args,
            "generate",
            {"generator_id": generator_id, "generator_model": generator_model},
        )
    _spawn_phase(args, "sufficiency")
    _spawn_phase(args, "transfer")
    summarize_phase(args)


def dispatch(args: argparse.Namespace) -> None:
    if args.phase == "sweep":
        sweep_phase(args)
    elif args.phase == "prepare":
        prepare_phase(args)
    elif args.phase == "generate":
        generate_phase(args)
    elif args.phase == "sufficiency":
        sufficiency_phase(args)
    elif args.phase == "transfer":
        transfer_phase(args)
    elif args.phase == "summarize":
        summarize_phase(args)
    else:
        raise ValueError(f"Unknown phase {args.phase!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", choices=PHASES, default="sweep")
    parser.add_argument(
        "--model",
        default=None,
        help="Frozen base student/teacher. Inferred from --run-dir when omitted.",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY_TRAIN),
        default=None,
        help="Training dataset. Inferred from --run-dir when omitted.",
    )
    checkpoint_source = parser.add_mutually_exclusive_group()
    checkpoint_source.add_argument(
        "--run-dir",
        default=None,
        help="Hint-generator run directory; all numeric checkpoint-* children are swept.",
    )
    checkpoint_source.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="[LABEL=]PATH",
        help="Explicit hint-generator checkpoint override. Repeat as needed.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Optional numeric checkpoint steps selected from --run-dir.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hint-root", default="data/pi/hint")
    parser.add_argument("--rollout-root", default="data/rollouts")
    parser.add_argument("--num-problems", type=int, default=64)
    parser.add_argument("--hints-per-problem", type=int, default=4)
    parser.add_argument("--hint-max-tokens", type=int, default=128)
    parser.add_argument("--generator-temperature", type=float, default=1.0)
    parser.add_argument("--generator-top-p", type=float, default=1.0)
    parser.add_argument("--teacher-rollouts", type=int, default=4)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--teacher-max-tokens", type=int, default=8192)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-top-p", type=float, default=1.0)
    parser.add_argument("--teacher-seed", type=int, default=314159)
    parser.add_argument("--save-teacher-samples", action="store_true")
    parser.add_argument("--transfer-rollouts", type=int, default=4)
    parser.add_argument(
        "--transfer-max-completion-tokens",
        type=int,
        default=0,
        help="Fixed prefix length for every transfer rollout; 0 scores the full cache.",
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    # Internal arguments populated by the sweep's spawned generation processes.
    parser.add_argument("--generator-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--generator-model", default=None, help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        resolve_run_configuration(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if args.steps is not None and args.run_dir is None:
        parser.error("--steps requires --run-dir")
    if args.steps is not None and any(step < 0 for step in args.steps):
        parser.error("--steps values must be >= 0")
    if args.num_problems < 1:
        parser.error("--num-problems must be >= 1")
    if args.hints_per_problem < 1 or args.hint_max_tokens < 1:
        parser.error("--hints-per-problem and --hint-max-tokens must be >= 1")
    if args.teacher_rollouts < 1 or args.transfer_rollouts < 1:
        parser.error("--teacher-rollouts and --transfer-rollouts must be >= 1")
    if any(k < 1 or k > args.teacher_rollouts for k in args.k):
        parser.error("Every --k must satisfy 1 <= k <= --teacher-rollouts")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be >= 1")
    if args.transfer_max_completion_tokens < 0:
        parser.error("--transfer-max-completion-tokens must be >= 0")
    if args.hint_max_tokens + 1 >= args.max_model_len:
        parser.error("--max-model-len must leave room for the generator prompt")
    if args.teacher_max_tokens + 1 >= args.max_model_len:
        parser.error("--max-model-len must leave room for the teacher prompt")
    if args.generator_temperature <= 0 or args.teacher_temperature <= 0:
        parser.error("Sampling temperatures must be > 0")
    for value, name in (
        (args.generator_top_p, "--generator-top-p"),
        (args.teacher_top_p, "--teacher-top-p"),
        (args.gpu_memory_utilization, "--gpu-memory-utilization"),
    ):
        if not 0 < value <= 1:
            parser.error(f"{name} must be in (0, 1]")
    # Discover once before the sweep starts. Training may continue writing later
    # checkpoints, but every phase in this invocation must see the same frozen arm list.
    try:
        args.resolved_checkpoints = generator_variants(args)[1:]
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    dispatch(args)


if __name__ == "__main__":
    main()
