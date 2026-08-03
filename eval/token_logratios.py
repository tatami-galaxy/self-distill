r"""Paired token log-ratio and entropy analysis on fixed frozen-student rollouts.

Unlike eval.teacher_behavior, which samples each teacher, this module scores the exact same
student completion tokens under one explicitly selected teacher model and PI condition.

Score conditions independently:

    python -m eval.token_logratios score \
      --student-model Qwen/Qwen3-1.7B \
      --teacher-model Qwen/Qwen3-1.7B --pi hint

    python -m eval.token_logratios score \
      --student-model Qwen/Qwen3-1.7B \
      --teacher-model Qwen/Qwen3-8B --pi none

Then combine any compatible set:

    python -m eval.token_logratios summarize \
      --score-dirs <self-none> <self-hint> <self-answer> <self-solution> <strong-none> \
      --output-dir results/token_logratios/combined

The CLI accepts both 'solution' and the repo's canonical name 'full'. Raw teacher
log-probabilities and entropy are stored so comparisons can be recombined without rescoring.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk

from train.opsd.train_self_teacher.lib import (
    ENTROPY_CHUNK,
    build_teacher_dataset,
    cross_fitted_logistic,
    macro_group_rank_auc,
    per_token_stats,
    rollout_path,
)
from utils import DATASET_REGISTRY_TRAIN


PI_ALIASES = {
    "none": "none",
    "hint": "hint",
    "answer": "answer",
    "solution": "full",
    "full": "full",
}
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
REQUIRED_ROLLOUT_COLUMNS = {
    "question",
    "final_answer",
    "completion_ids",
    "completion_text",
    "reward",
    "student_logps",
}
PREFIX_QUANTILES = (25, 50, 75, 100)


# ---------------------------------------------------------------------------
# Stable identities and provenance
# ---------------------------------------------------------------------------


def canonical_pi_mode(pi: str) -> str:
    try:
        return PI_ALIASES[pi]
    except KeyError as exc:
        raise ValueError(f"unknown PI {pi!r}; expected one of {tuple(PI_ALIASES)}") from exc


def model_slug(model: str) -> str:
    return model.rstrip("/").split("/")[-1].replace(os.sep, "_")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_question_id(question: str) -> str:
    return _sha256_text(question)[:20]


def stable_rollout_id(row: dict, row_index: int) -> str:
    """Stable across independently scored conditions, including legacy rollout caches."""
    if row.get("rollout_id"):
        return str(row["rollout_id"])
    payload = json.dumps(
        {
            "question": row["question"],
            "completion_ids": list(row["completion_ids"]),
            "sample_idx": row.get("sample_idx", row_index),
            "row_index": row_index,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _sha256_text(payload)[:24]


def fingerprint_ids(ids: list[str]) -> str:
    h = hashlib.sha256()
    for value in ids:
        h.update(value.encode())
        h.update(b"\0")
    return h.hexdigest()


def tokenizer_vocab_hash(tokenizer) -> str:
    h = hashlib.sha256()
    for token, token_id in sorted(tokenizer.get_vocab().items()):
        h.update(token.encode("utf-8", errors="surrogatepass"))
        h.update(b"\0")
        h.update(str(token_id).encode())
        h.update(b"\n")
    return h.hexdigest()


def tokenizer_metadata(tokenizer) -> dict:
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_hash": tokenizer_vocab_hash(tokenizer),
        "chat_template_hash": _sha256_text(str(getattr(tokenizer, "chat_template", ""))),
        "special_ids": sorted(int(value) for value in tokenizer.all_special_ids),
    }


def verify_tokenizer_compatibility(student_tokenizer, teacher_tokenizer) -> tuple[dict, dict]:
    """Cached student logps align by token ID, so mismatched vocabularies are invalid."""
    student_meta = tokenizer_metadata(student_tokenizer)
    teacher_meta = tokenizer_metadata(teacher_tokenizer)
    if student_meta["vocab_hash"] != teacher_meta["vocab_hash"]:
        raise ValueError(
            "Student and teacher token vocabularies do not have identical token-to-ID mappings. "
            "Retokenizing text cannot preserve cached token log-probability alignment."
        )
    if student_meta["special_ids"] != teacher_meta["special_ids"]:
        raise ValueError("Student and teacher special-token IDs differ; refusing misaligned score.")
    return student_meta, teacher_meta


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(tmp, path)


def add_rollout_identities(rollouts):
    def _identity(row, index):
        return {
            "rollout_id": stable_rollout_id(row, index),
            "question_id": row.get("question_id") or stable_question_id(row["question"]),
            "rollout_index": index,
        }

    return rollouts.map(_identity, with_indices=True, num_proc=1)


def validate_rollout_cache(rollouts) -> None:
    missing = REQUIRED_ROLLOUT_COLUMNS - set(rollouts.column_names)
    if missing:
        raise ValueError(f"rollout cache is missing required columns: {sorted(missing)}")
    for index, row in enumerate(rollouts):
        n_tokens = len(row["completion_ids"])
        if not n_tokens:
            raise ValueError(f"rollout row {index} has an empty completion")
        if len(row["student_logps"]) != n_tokens:
            raise ValueError(
                f"rollout row {index} has {n_tokens} tokens but "
                f"{len(row['student_logps'])} student logps"
            )


def _find_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(sequence):
        return None
    for start in range(len(sequence) - len(pattern) + 1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def _truncation_fields(row: dict, rollout_max_tokens: int | None) -> tuple[str, bool | None]:
    reason = str(row.get("finish_reason") or "unknown")
    if row.get("truncated") is not None:
        return reason, bool(row["truncated"])
    if reason != "unknown":
        return reason, reason == "length"
    if rollout_max_tokens is not None:
        return reason, len(row["completion_ids"]) >= rollout_max_tokens
    return reason, None


# ---------------------------------------------------------------------------
# Per-condition scoring
# ---------------------------------------------------------------------------


def _default_rollouts(args) -> str:
    return rollout_path(args.student_model, args.dataset, args.rollout_root)


def _default_score_dir(args, pi_mode: str) -> Path:
    return (
        Path(args.output_root)
        / model_slug(args.student_model)
        / model_slug(args.teacher_model)
        / pi_mode
    )


def _safe_remove_run_dir(path: Path) -> None:
    resolved = path.resolve()
    if resolved in (Path("/"), Path.cwd().resolve()):
        raise ValueError(f"refusing to remove unsafe output path {resolved}")
    shutil.rmtree(resolved)


def _is_student_reference(args) -> bool:
    return (
        args.teacher_model == args.student_model
        and args.teacher_revision == args.student_revision
        and canonical_pi_mode(args.pi) == "none"
    )


def _score_config(
    args,
    pi_mode: str,
    source_fingerprint: str,
    eligible_ids: list[str],
    source_dataset_fingerprint: str | None,
) -> dict:
    return {
        "student_model": args.student_model,
        "student_revision": args.student_revision,
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "pi_mode": pi_mode,
        "dataset": args.dataset,
        "rollouts": str(Path(args.rollouts or _default_rollouts(args)).resolve()),
        "source_rollout_fingerprint": source_fingerprint,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "eligible_rollout_fingerprint": fingerprint_ids(eligible_ids),
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "prompt_template_source": args.prompt_template_source,
        "rollout_max_tokens": args.rollout_max_tokens,
        "entropy_chunk_size": args.entropy_chunk_size,
        "reference_atol": args.reference_atol,
        "device": args.device,
        "device_map": args.device_map,
        "label": args.label,
    }


def _condition_label(args, pi_mode: str) -> str:
    if args.label:
        return args.label
    label = f"{model_slug(args.teacher_model)}__{pi_mode}"
    if args.prompt_template_source == "student" and args.teacher_model != args.student_model:
        label += "__student_template"
    return label


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def _save_score_part(shards_dir: Path, part_index: int, rows: list[dict]) -> Path:
    final = shards_dir / f"part-{part_index:06d}"
    tmp = shards_dir / f".part-{part_index:06d}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    Dataset.from_list(rows).save_to_disk(str(tmp))
    os.rename(tmp, final)
    return final


def _load_existing_parts(shards_dir: Path) -> tuple[list[Path], set[str]]:
    parts = sorted(path for path in shards_dir.glob("part-*") if path.is_dir())
    scored: set[str] = set()
    for part in parts:
        dataset = load_from_disk(str(part))
        overlap = scored.intersection(dataset["rollout_id"])
        if overlap:
            raise ValueError(f"duplicate rollout IDs across score shards: {sorted(overlap)[:3]}")
        scored.update(dataset["rollout_id"])
    return parts, scored


def _load_tokenizers(args):
    from transformers import AutoTokenizer

    student_kwargs = {"trust_remote_code": True}
    teacher_kwargs = {"trust_remote_code": True}
    if args.student_revision:
        student_kwargs["revision"] = args.student_revision
    if args.teacher_revision:
        teacher_kwargs["revision"] = args.teacher_revision
    student = AutoTokenizer.from_pretrained(args.student_model, **student_kwargs)
    teacher = AutoTokenizer.from_pretrained(args.teacher_model, **teacher_kwargs)
    student_meta, teacher_meta = verify_tokenizer_compatibility(student, teacher)
    return student, teacher, student_meta, teacher_meta


def _prepare_teacher_rows(args, student_tokenizer, teacher_tokenizer, pi_mode: str):
    rollout_dir = args.rollouts or _default_rollouts(args)
    rollouts = load_from_disk(rollout_dir)
    source_dataset_fingerprint = getattr(rollouts, "_fingerprint", None)
    validate_rollout_cache(rollouts)
    rollouts = add_rollout_identities(rollouts)
    if args.max_rollouts is not None:
        rollouts = rollouts.select(range(min(args.max_rollouts, len(rollouts))))
    source_fingerprint = fingerprint_ids(list(rollouts["rollout_id"]))

    prompt_tokenizer = (
        student_tokenizer if args.prompt_template_source == "student" else teacher_tokenizer
    )
    teacher_rows = build_teacher_dataset(
        rollouts,
        pi_mode,
        prompt_tokenizer,
        dataset=args.dataset,
        max_teacher_prompt_length=None,
    )
    n_after_pi_join = len(teacher_rows)
    eligible_indices = [
        index
        for index, row in enumerate(teacher_rows)
        if len(row["teacher_prompt_ids"]) + len(row["completion_ids"]) <= args.max_model_len
    ]
    teacher_rows = teacher_rows.select(eligible_indices)
    if not len(teacher_rows):
        raise RuntimeError("No rollouts fit teacher prompt + completion inside --max-model-len.")
    return {
        "rows": teacher_rows,
        "source_fingerprint": source_fingerprint,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "n_source": len(rollouts),
        "n_missing_pi": len(rollouts) - n_after_pi_join,
        "n_context_dropped": n_after_pi_join - len(teacher_rows),
    }


def _load_teacher_model(args):
    from transformers import AutoModelForCausalLM

    kwargs = {"dtype": DTYPES[args.dtype], "trust_remote_code": True}
    if args.teacher_revision:
        kwargs["revision"] = args.teacher_revision
    if args.device_map:
        kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **kwargs).eval()
    if not args.device_map:
        model.to(args.device)
    return model


def _score_row(
    row: dict,
    model,
    input_device: torch.device,
    student_tokenizer,
    args,
) -> tuple[dict, float | None]:
    completion_id_list = list(row["completion_ids"])
    prompt_ids = torch.tensor(
        row["teacher_prompt_ids"], dtype=torch.long, device=input_device
    ).unsqueeze(0)
    completion_ids = torch.tensor(
        completion_id_list, dtype=torch.long, device=input_device
    ).unsqueeze(0)
    with torch.inference_mode():
        stats = per_token_stats(
            model,
            torch.cat([prompt_ids, completion_ids], dim=1),
            completion_ids,
            entropy_chunk_size=args.entropy_chunk_size,
        )
    teacher_logps = stats.logps.squeeze(0).cpu()
    teacher_entropy = stats.entropy.squeeze(0).cpu()
    student_logps = torch.tensor(row["student_logps"], dtype=torch.float32)
    if teacher_logps.numel() != student_logps.numel():
        raise RuntimeError("teacher/student token scores lost alignment")

    reference_diff = None
    if _is_student_reference(args):
        reference_diff = (teacher_logps - student_logps).abs().max().item()
        if reference_diff > args.reference_atol:
            raise RuntimeError(
                "self/no-PI score does not reproduce cached student logps: "
                f"max |delta|={reference_diff:.6g} > --reference-atol={args.reference_atol}. "
                "Verify model revision and numerical environment."
            )

    close_ids = student_tokenizer.encode("</think>", add_special_tokens=False)
    boxed_ids = student_tokenizer.encode("\\boxed", add_special_tokens=False)
    close_start = _find_subsequence(completion_id_list, close_ids)
    box_start = _find_subsequence(completion_id_list, boxed_ids)
    finish_reason, truncated = _truncation_fields(row, args.rollout_max_tokens)
    result = {
        "rollout_id": row["rollout_id"],
        "question_id": row["question_id"],
        "rollout_index": int(row["rollout_index"]),
        "reward": float(row["reward"]),
        "n_tokens": len(completion_id_list),
        "finish_reason": finish_reason,
        "truncated": truncated,
        "think_close_start": -1 if close_start is None else close_start,
        "think_close_end": -1 if close_start is None else close_start + len(close_ids),
        "boxed_start": -1 if box_start is None else box_start,
        "prompt_tokens": len(row["teacher_prompt_ids"]),
        "has_pi": bool(row["has_pi"]),
        "teacher_logps": teacher_logps.tolist(),
        "teacher_entropy": teacher_entropy.tolist(),
    }
    if reference_diff is not None:
        result["reference_abs_logp_delta"] = reference_diff
    return result, reference_diff


def score_condition(args) -> Path:
    pi_mode = canonical_pi_mode(args.pi)
    student_tokenizer, teacher_tokenizer, student_meta, teacher_meta = _load_tokenizers(args)
    if (
        args.teacher_model != args.student_model
        and student_meta["chat_template_hash"] != teacher_meta["chat_template_hash"]
    ):
        print(
            "warning: student and teacher chat templates differ; "
            f"using the {args.prompt_template_source} template"
        )

    prepared = _prepare_teacher_rows(args, student_tokenizer, teacher_tokenizer, pi_mode)
    teacher_rows = prepared["rows"]
    eligible_ids = list(teacher_rows["rollout_id"])
    out_dir = (
        Path(args.output_dir) if args.output_dir else _default_score_dir(args, pi_mode)
    ).resolve()
    config = _score_config(
        args,
        pi_mode,
        prepared["source_fingerprint"],
        eligible_ids,
        prepared["source_dataset_fingerprint"],
    )
    config_hash = _sha256_text(json.dumps(config, sort_keys=True))
    meta_path = out_dir / "run_meta.json"

    if out_dir.exists() and args.force:
        _safe_remove_run_dir(out_dir)
    if (out_dir / "scores").is_dir():
        existing = json.loads(meta_path.read_text())
        if existing.get("config_hash") == config_hash and existing.get("status") == "complete":
            print(f"Complete compatible score already exists at {out_dir}")
            return out_dir
        raise FileExistsError(
            f"{out_dir} contains incompatible scores; use --force or another --output-dir"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(exist_ok=True)
    existing_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if existing_meta and existing_meta.get("config_hash") != config_hash:
        raise ValueError(f"incomplete run at {out_dir} has another config; use --force")
    label = _condition_label(args, pi_mode)
    meta = {
        **existing_meta,
        **config,
        "config_hash": config_hash,
        "label": label,
        "pi_cli": args.pi,
        "is_student_reference": _is_student_reference(args),
        "status": "scoring",
        "created_at": existing_meta.get(
            "created_at", dt.datetime.now(dt.timezone.utc).isoformat()
        ),
        "git_commit": git_commit(),
        "student_tokenizer": student_meta,
        "teacher_tokenizer": teacher_meta,
        "n_source_rollouts": prepared["n_source"],
        "n_eligible_rollouts": len(teacher_rows),
        "n_dropped_missing_pi": prepared["n_missing_pi"],
        "n_dropped_context": prepared["n_context_dropped"],
    }
    write_json_atomic(meta_path, meta)

    parts, scored_ids = _load_existing_parts(shards_dir)
    if scored_ids - set(eligible_ids):
        raise ValueError("resume shards contain rows outside the current eligible cohort")
    print(
        f"condition={label} PI={pi_mode} eligible={len(eligible_ids)} "
        f"already_scored={len(scored_ids)}"
    )
    if len(scored_ids) < len(eligible_ids):
        model = _load_teacher_model(args)
        device = _input_device(model)
        part_index = len(parts)
        buffer: list[dict] = []
        reference_diffs = []
        for row in teacher_rows:
            if row["rollout_id"] in scored_ids:
                continue
            scored_row, reference_diff = _score_row(
                row, model, device, student_tokenizer, args
            )
            buffer.append(scored_row)
            if reference_diff is not None:
                reference_diffs.append(reference_diff)
            if len(buffer) >= args.save_every:
                part = _save_score_part(shards_dir, part_index, buffer)
                scored_ids.update(item["rollout_id"] for item in buffer)
                print(f"  saved {part.name}: {len(scored_ids)}/{len(eligible_ids)}")
                buffer = []
                part_index += 1
        if buffer:
            part = _save_score_part(shards_dir, part_index, buffer)
            scored_ids.update(item["rollout_id"] for item in buffer)
            print(f"  saved {part.name}: {len(scored_ids)}/{len(eligible_ids)}")
        if reference_diffs:
            meta["reference_max_abs_logp_delta"] = max(
                float(meta.get("reference_max_abs_logp_delta", 0.0)),
                max(reference_diffs),
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    parts, scored_ids = _load_existing_parts(shards_dir)
    if scored_ids != set(eligible_ids):
        raise RuntimeError(f"scoring finished with {len(eligible_ids) - len(scored_ids)} missing rows")
    scores = concatenate_datasets([load_from_disk(str(path)) for path in parts])
    scores = scores.sort("rollout_index")
    if "reference_abs_logp_delta" in scores.column_names:
        meta["reference_max_abs_logp_delta"] = max(
            float(value) for value in scores["reference_abs_logp_delta"]
        )
    temporary = out_dir / f".scores.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    scores.save_to_disk(str(temporary))
    os.rename(temporary, out_dir / "scores")
    meta.update(
        {
            "status": "complete",
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "n_scored_rollouts": len(scores),
        }
    )
    write_json_atomic(meta_path, meta)
    if not args.keep_shards:
        shutil.rmtree(shards_dir)
    print(f"Saved {label} token scores -> {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if not left_ss or not right_ss:
        return None
    covariance = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    return covariance / math.sqrt(left_ss * right_ss)


def _prefix_mean(values: list[float], percentile: int) -> float:
    end = max(1, math.ceil(len(values) * percentile / 100))
    return sum(values[:end]) / end


def _top_mass_share(values: list[float], fraction: float) -> float:
    magnitudes = sorted((abs(value) for value in values), reverse=True)
    total = sum(magnitudes)
    if not total:
        return 0.0
    keep = max(1, math.ceil(len(magnitudes) * fraction))
    return sum(magnitudes[:keep]) / total


def trace_metrics(
    row: dict,
    reference_entropy: list[float] | None,
    ratio_threshold: float,
) -> dict:
    teacher_logps = [float(value) for value in row["teacher_logps"]]
    student_logps = [float(value) for value in row["student_logps"]]
    entropy = [float(value) for value in row["teacher_entropy"]]
    if not (len(teacher_logps) == len(student_logps) == len(entropy) == row["n_tokens"]):
        raise ValueError(f"unaligned arrays for rollout {row['rollout_id']}")
    ratios = [
        teacher - student
        for teacher, student in zip(teacher_logps, student_logps, strict=True)
    ]
    n_tokens = len(ratios)
    ratio_mean = sum(ratios) / n_tokens
    absolute_mass = sum(abs(value) for value in ratios)
    last_quartile = math.floor(0.75 * n_tokens)
    result = {
        "rollout_id": row["rollout_id"],
        "question_id": row["question_id"],
        "reward": float(row["reward"]),
        "truncated": row.get("truncated"),
        "n_tokens": n_tokens,
        "ratio_mean": ratio_mean,
        "ratio_sum": sum(ratios),
        "ratio_sqrt_normalized": sum(ratios) / math.sqrt(n_tokens),
        "ratio_std": (
            sum((value - ratio_mean) ** 2 for value in ratios) / n_tokens
        ) ** 0.5,
        "positive_ratio_mass_per_token": sum(max(value, 0.0) for value in ratios) / n_tokens,
        "negative_ratio_mass_per_token": sum(max(-value, 0.0) for value in ratios) / n_tokens,
        "positive_token_fraction": sum(value > ratio_threshold for value in ratios) / n_tokens,
        "negative_token_fraction": sum(value < -ratio_threshold for value in ratios) / n_tokens,
        "credit_mass_last_quartile": (
            sum(abs(value) for value in ratios[last_quartile:]) / absolute_mass
            if absolute_mass
            else 0.0
        ),
        "credit_mass_top_1pct": _top_mass_share(ratios, 0.01),
        "credit_mass_top_5pct": _top_mass_share(ratios, 0.05),
        "credit_mass_top_10pct": _top_mass_share(ratios, 0.10),
        "entropy_mean": sum(entropy) / n_tokens,
    }
    for percentile in PREFIX_QUANTILES:
        result[f"ratio_prefix_q{percentile}"] = _prefix_mean(ratios, percentile)
        result[f"entropy_prefix_q{percentile}"] = _prefix_mean(entropy, percentile)

    think_end = int(row.get("think_close_end", -1))
    if think_end > 0:
        result["ratio_mean_think"] = mean(ratios[:think_end])
        result["ratio_mean_post"] = mean(ratios[think_end:])
        result["entropy_mean_think"] = mean(entropy[:think_end])
        result["entropy_mean_post"] = mean(entropy[think_end:])
    else:
        result["ratio_mean_think"] = mean(ratios)
        result["ratio_mean_post"] = None
        result["entropy_mean_think"] = mean(entropy)
        result["entropy_mean_post"] = None

    boxed_start = int(row.get("boxed_start", -1))
    if boxed_start >= 0:
        result["ratio_mean_boxed_suffix"] = mean(ratios[boxed_start:])
        result["entropy_mean_boxed_suffix"] = mean(entropy[boxed_start:])
    else:
        result["ratio_mean_boxed_suffix"] = None
        result["entropy_mean_boxed_suffix"] = None

    if reference_entropy is not None:
        if len(reference_entropy) != n_tokens:
            raise ValueError(f"reference entropy misaligned for {row['rollout_id']}")
        entropy_delta = [
            value - float(reference)
            for value, reference in zip(entropy, reference_entropy, strict=True)
        ]
        result.update(
            {
                "entropy_delta_mean": sum(entropy_delta) / n_tokens,
                "entropy_collapse_mass_per_token": (
                    sum(max(-value, 0.0) for value in entropy_delta) / n_tokens
                ),
                "entropy_expansion_mass_per_token": (
                    sum(max(value, 0.0) for value in entropy_delta) / n_tokens
                ),
                "entropy_collapse_token_fraction": (
                    sum(value < 0 for value in entropy_delta) / n_tokens
                ),
                "entropy_expansion_token_fraction": (
                    sum(value > 0 for value in entropy_delta) / n_tokens
                ),
                "ratio_entropy_delta_correlation": pearson(ratios, entropy_delta),
            }
        )
        for percentile in PREFIX_QUANTILES:
            result[f"entropy_delta_prefix_q{percentile}"] = _prefix_mean(
                entropy_delta, percentile
            )
    return result


TRACE_MEAN_KEYS = (
    "n_tokens",
    "ratio_mean",
    "ratio_sum",
    "ratio_sqrt_normalized",
    "ratio_std",
    "positive_ratio_mass_per_token",
    "negative_ratio_mass_per_token",
    "positive_token_fraction",
    "negative_token_fraction",
    "credit_mass_last_quartile",
    "credit_mass_top_1pct",
    "credit_mass_top_5pct",
    "credit_mass_top_10pct",
    "entropy_mean",
    "ratio_mean_think",
    "ratio_mean_post",
    "entropy_mean_think",
    "entropy_mean_post",
    "ratio_mean_boxed_suffix",
    "entropy_mean_boxed_suffix",
    "entropy_delta_mean",
    "entropy_collapse_mass_per_token",
    "entropy_expansion_mass_per_token",
    "entropy_collapse_token_fraction",
    "entropy_expansion_token_fraction",
    "ratio_entropy_delta_correlation",
)


def aggregate_trace_metrics(rows: list[dict]) -> dict:
    result = {"n_rollouts": len(rows)}
    for key in TRACE_MEAN_KEYS:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            result[f"mean_{key}"] = mean(values)
    return result


def paired_question_effect(
    trace_rows: list[dict],
    key: str,
    correct_minus_wrong: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict | None:
    grouped: dict[str, dict[bool, list[float]]] = defaultdict(
        lambda: {False: [], True: []}
    )
    for row in trace_rows:
        if row.get(key) is not None:
            grouped[row["question_id"]][row["reward"] > 0.5].append(float(row[key]))
    differences = []
    for outcomes in grouped.values():
        if outcomes[True] and outcomes[False]:
            correct, wrong = mean(outcomes[True]), mean(outcomes[False])
            differences.append(correct - wrong if correct_minus_wrong else wrong - correct)
    if not differences:
        return None
    rng = random.Random(seed)
    bootstrap = [
        mean([rng.choice(differences) for _ in differences])
        for _ in range(bootstrap_samples)
    ]
    return {
        "n_mixed_questions": len(differences),
        "mean": mean(differences),
        "ci95": [quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)],
    }


def _update_position_accumulator(
    accum: dict[tuple[int, str, str], list[float]],
    row: dict,
    reference_entropy: list[float] | None,
    bins: int,
) -> None:
    ratios = [
        float(teacher) - float(student)
        for teacher, student in zip(
            row["teacher_logps"], row["student_logps"], strict=True
        )
    ]
    entropy = [float(value) for value in row["teacher_entropy"]]
    reference = (
        [float(value) for value in reference_entropy]
        if reference_entropy is not None
        else None
    )
    strata = ("all", "correct" if row["reward"] > 0.5 else "wrong")
    for bin_index in range(bins):
        start = math.floor(len(ratios) * bin_index / bins)
        end = math.floor(len(ratios) * (bin_index + 1) / bins)
        if end <= start:
            continue
        values = {
            "ratio": mean(ratios[start:end]),
            "entropy": mean(entropy[start:end]),
        }
        if reference is not None:
            values["entropy_delta"] = mean(
                [
                    value - ref
                    for value, ref in zip(
                        entropy[start:end], reference[start:end], strict=True
                    )
                ]
            )
        for stratum in strata:
            for metric, value in values.items():
                total_count = accum[(bin_index, stratum, metric)]
                total_count[0] += value
                total_count[1] += 1


def _finish_position_curves(
    accum: dict[tuple[int, str, str], list[float]], bins: int
) -> list[dict]:
    curves = []
    for bin_index in range(bins):
        row = {
            "bin": bin_index,
            "position_start": bin_index / bins,
            "position_end": (bin_index + 1) / bins,
        }
        for stratum in ("all", "correct", "wrong"):
            for metric in ("ratio", "entropy", "entropy_delta"):
                total, count = accum.get((bin_index, stratum, metric), (0.0, 0))
                if count:
                    row[f"{stratum}_{metric}"] = total / count
        curves.append(row)
    return curves


def prefix_outcome_metrics(
    traces: list[dict],
    prefix_stem: str,
    higher_is_better: bool,
) -> dict:
    """Question-grouped prefix AUC and cross-fitted Brier from per-trace scores."""
    rewards = torch.tensor([float(row["reward"]) for row in traces])
    question_ids = [row["question_id"] for row in traces]
    result = {}
    both_classes = rewards.numel() and rewards.min() < 0.5 < rewards.max()
    for percentile in PREFIX_QUANTILES:
        scores = torch.tensor(
            [float(row[f"{prefix_stem}_q{percentile}"]) for row in traces]
        )
        if not higher_is_better:
            scores = -scores
        auc, count = macro_group_rank_auc(scores, rewards, question_ids)
        if auc is not None:
            result[f"within_question_auc_q{percentile}"] = auc
            result[f"mixed_question_count_q{percentile}"] = count
        if both_classes:
            fitted = cross_fitted_logistic(
                scores,
                rewards,
                question_ids,
                n_folds=5,
                nonnegative_slope=True,
            )
            if fitted is not None:
                predictions, constant, _ = fitted
                result[f"brier_q{percentile}_fitted"] = (
                    (predictions - rewards) ** 2
                ).mean().item()
                if "brier_floor_crossfit" not in result:
                    result["brier_floor_crossfit"] = (
                        (constant - rewards) ** 2
                    ).mean().item()
    return result


def condition_summary(
    rows,
    ratio_threshold: float,
    bins: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    """Consume (score_row_with_student_logps, reference_entropy_or_none) pairs once."""
    traces = []
    position_accum: dict[tuple[int, str, str], list[float]] = defaultdict(
        lambda: [0.0, 0]
    )
    for row, reference_entropy in rows:
        traces.append(trace_metrics(row, reference_entropy, ratio_threshold))
        _update_position_accumulator(position_accum, row, reference_entropy, bins)
    if not traces:
        raise ValueError("condition has no common rollout rows")

    summary = {
        "overall": aggregate_trace_metrics(traces),
        "correct": aggregate_trace_metrics([row for row in traces if row["reward"] > 0.5]),
        "wrong": aggregate_trace_metrics([row for row in traces if row["reward"] <= 0.5]),
        "complete": aggregate_trace_metrics(
            [row for row in traces if row.get("truncated") is False]
        ),
        "truncated": aggregate_trace_metrics(
            [row for row in traces if row.get("truncated") is True]
        ),
        "correct_complete": aggregate_trace_metrics(
            [
                row for row in traces
                if row["reward"] > 0.5 and row.get("truncated") is False
            ]
        ),
        "wrong_complete": aggregate_trace_metrics(
            [
                row for row in traces
                if row["reward"] <= 0.5 and row.get("truncated") is False
            ]
        ),
        "within_question_ratio_correct_minus_wrong": paired_question_effect(
            traces, "ratio_mean", True, bootstrap_samples, seed
        ),
        "within_question_entropy_wrong_minus_correct": paired_question_effect(
            traces, "entropy_mean", False, bootstrap_samples, seed + 1
        ),
        "position_curves": _finish_position_curves(position_accum, bins),
        "ratio_outcome_metrics": prefix_outcome_metrics(
            traces, "ratio_prefix", higher_is_better=True
        ),
        "negative_entropy_outcome_metrics": prefix_outcome_metrics(
            traces, "entropy_prefix", higher_is_better=False
        ),
    }
    if any("entropy_delta_mean" in row for row in traces):
        summary["within_question_entropy_delta_wrong_minus_correct"] = paired_question_effect(
            traces, "entropy_delta_mean", False, bootstrap_samples, seed + 2
        )
    return summary, traces


def pairwise_agreement(left, right, ratio_threshold: float) -> dict:
    """Stream two aligned conditions and compare their token-ratio allocations."""
    correlations, agreements = [], []
    for left_row, right_row in zip(left, right, strict=True):
        left_ratios = [
            float(teacher) - float(student)
            for teacher, student in zip(
                left_row["teacher_logps"], left_row["student_logps"], strict=True
            )
        ]
        right_ratios = [
            float(teacher) - float(student)
            for teacher, student in zip(
                right_row["teacher_logps"], right_row["student_logps"], strict=True
            )
        ]
        correlation = pearson(left_ratios, right_ratios)
        if correlation is not None:
            correlations.append(correlation)
        salient = [
            index
            for index, (left_value, right_value) in enumerate(
                zip(left_ratios, right_ratios, strict=True)
            )
            if abs(left_value) > ratio_threshold or abs(right_value) > ratio_threshold
        ]
        if salient:
            agreements.append(
                sum(
                    (left_ratios[index] > 0) == (right_ratios[index] > 0)
                    for index in salient
                )
                / len(salient)
            )
    return {
        "mean_within_trace_ratio_correlation": mean(correlations),
        "n_correlation_traces": len(correlations),
        "mean_salient_sign_agreement": mean(agreements),
        "n_sign_agreement_traces": len(agreements),
    }


def _row_index(dataset) -> dict[str, int]:
    return {
        rollout_id: index
        for index, rollout_id in enumerate(dataset["rollout_id"])
    }


def _source_rollout_index(runs, common_ids: set[str]):
    """Load the immutable rollout cache only when score artifacts omit duplicated student arrays."""
    if all("student_logps" in dataset.column_names for _, _, dataset in runs):
        return None, {}
    rollout_paths = {meta.get("rollouts") for _, meta, _ in runs}
    if None in rollout_paths or len(rollout_paths) != 1:
        raise ValueError("compact score runs must point to the same source rollout cache")
    source = load_from_disk(next(iter(rollout_paths)))
    expected_dataset_fingerprints = {
        meta.get("source_dataset_fingerprint")
        for _, meta, _ in runs
        if meta.get("source_dataset_fingerprint") is not None
    }
    if len(expected_dataset_fingerprints) == 1 and getattr(
        source, "_fingerprint", None
    ) != next(iter(expected_dataset_fingerprints)):
        raise ValueError("source rollout dataset contents changed after scoring")
    n_source_values = {meta.get("n_source_rollouts") for _, meta, _ in runs}
    if None not in n_source_values and len(n_source_values) == 1:
        source = source.select(range(min(next(iter(n_source_values)), len(source))))

    index_by_id = {}
    source_ids = []
    if "rollout_id" in source.column_names:
        source_ids = list(source["rollout_id"])
        index_by_id = {
            rollout_id: index for index, rollout_id in enumerate(source_ids)
        }
    else:
        for index, row in enumerate(source):
            rollout_id = stable_rollout_id(row, index)
            source_ids.append(rollout_id)
            if rollout_id in common_ids:
                index_by_id[rollout_id] = index
    missing = common_ids - set(index_by_id)
    if missing:
        raise ValueError(f"source rollout cache is missing {len(missing)} scored rollout IDs")
    expected = {meta["source_rollout_fingerprint"] for _, meta, _ in runs}
    if len(expected) == 1 and fingerprint_ids(source_ids) != next(iter(expected)):
        raise ValueError("source rollout cache fingerprint changed after scoring")
    return source, index_by_id


def _condition_row_stream(
    dataset,
    score_index: dict[str, int],
    ordered_ids: list[str],
    source,
    source_index: dict[str, int],
):
    """Yield score rows enriched with cached student logps, one rollout at a time."""
    for rollout_id in ordered_ids:
        row = dict(dataset[score_index[rollout_id]])
        if "student_logps" not in row:
            row["student_logps"] = source[source_index[rollout_id]]["student_logps"]
        yield row


def _condition_reference_stream(
    dataset,
    score_index: dict[str, int],
    ordered_ids: list[str],
    source,
    source_index: dict[str, int],
    reference_dataset,
    reference_index: dict[str, int] | None,
):
    rows = _condition_row_stream(
        dataset, score_index, ordered_ids, source, source_index
    )
    for row in rows:
        reference_entropy = (
            reference_dataset[reference_index[row["rollout_id"]]]["teacher_entropy"]
            if reference_dataset is not None
            else None
        )
        yield row, reference_entropy


def _ratio_noise_sample(rows, max_tokens: int, seed: int) -> list[float]:
    """Reservoir sample self/no-PI absolute deltas without retaining every token."""
    rng = random.Random(seed)
    sample: list[float] = []
    seen = 0
    for row in rows:
        for teacher, student in zip(
            row["teacher_logps"], row["student_logps"], strict=True
        ):
            value = abs(float(teacher) - float(student))
            seen += 1
            if len(sample) < max_tokens:
                sample.append(value)
            else:
                replace = rng.randrange(seen)
                if replace < max_tokens:
                    sample[replace] = value
    return sample


def summarize_runs(args) -> Path:
    runs = []
    for directory in args.score_dirs:
        path = Path(directory).resolve()
        meta_path, scores_path = path / "run_meta.json", path / "scores"
        if not meta_path.is_file() or not scores_path.is_dir():
            raise FileNotFoundError(f"{path} is not a complete score directory")
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "complete":
            raise ValueError(f"{path} is not marked complete")
        runs.append((path, meta, load_from_disk(str(scores_path))))

    labels = [meta["label"] for _, meta, _ in runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"condition labels must be unique; got {labels}")
    student_models = {meta["student_model"] for _, meta, _ in runs}
    student_revisions = {meta.get("student_revision") for _, meta, _ in runs}
    source_fingerprints = {meta["source_rollout_fingerprint"] for _, meta, _ in runs}
    source_dataset_fingerprints = {
        meta.get("source_dataset_fingerprint")
        for _, meta, _ in runs
        if meta.get("source_dataset_fingerprint") is not None
    }
    if (
        len(student_models) != 1
        or len(student_revisions) != 1
        or len(source_fingerprints) != 1
        or len(source_dataset_fingerprints) > 1
    ):
        raise ValueError("score runs must use the same student and source rollout cohort")

    score_indices = {
        meta["label"]: _row_index(dataset)
        for _, meta, dataset in runs
    }
    common_ids = set.intersection(
        *(set(index) for index in score_indices.values())
    )
    if not common_ids:
        raise ValueError("score runs have no common rollout IDs")
    first_label = labels[0]
    first_dataset = runs[0][2]
    ordered_ids = [
        rollout_id
        for rollout_id in first_dataset.sort("rollout_index")["rollout_id"]
        if rollout_id in common_ids
    ]
    run_by_label = {
        meta["label"]: (path, meta, dataset)
        for path, meta, dataset in runs
    }
    source, source_index = _source_rollout_index(runs, common_ids)

    reference_label = next(
        (
            meta["label"]
            for _, meta, _ in runs
            if meta.get("is_student_reference", False)
        ),
        None,
    )
    reference_dataset = (
        run_by_label[reference_label][2] if reference_label is not None else None
    )
    reference_index = (
        score_indices[reference_label] if reference_label is not None else None
    )

    def rows_for(label):
        dataset = run_by_label[label][2]
        return _condition_row_stream(
            dataset, score_indices[label], ordered_ids, source, source_index
        )

    ratio_threshold = 0.0
    if reference_label is not None:
        noise = _ratio_noise_sample(
            rows_for(reference_label), args.noise_sample_tokens, args.seed
        )
        ratio_threshold = quantile(noise, args.noise_quantile) or 0.0

    summaries, per_trace = {}, []
    for index, label in enumerate(labels):
        dataset = run_by_label[label][2]
        rows = _condition_reference_stream(
            dataset,
            score_indices[label],
            ordered_ids,
            source,
            source_index,
            reference_dataset,
            reference_index,
        )
        summary, traces = condition_summary(
            rows,
            ratio_threshold,
            args.position_bins,
            args.bootstrap_samples,
            args.seed + index * 17,
        )
        summaries[label] = summary
        per_trace.extend({"condition": label, **row} for row in traces)

    agreement = {}
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            agreement[f"{left_label}__vs__{right_label}"] = pairwise_agreement(
                rows_for(left_label), rows_for(right_label), ratio_threshold
            )

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} exists; use --force to replace it")
        _safe_remove_run_dir(output_dir)
    output_dir.mkdir(parents=True)
    Dataset.from_list(per_trace).save_to_disk(str(output_dir / "per_trace"))
    first_question_ids = {
        first_dataset[score_indices[first_label][rollout_id]]["question_id"]
        for rollout_id in ordered_ids
    }
    write_json_atomic(
        output_dir / "summary.json",
        {
            "student_model": next(iter(student_models)),
            "student_revision": next(iter(student_revisions)),
            "conditions": labels,
            "reference_condition": reference_label,
            "n_common_rollouts": len(common_ids),
            "n_common_questions": len(first_question_ids),
            "ratio_noise_threshold": ratio_threshold,
            "ratio_noise_quantile": args.noise_quantile,
            "ratio_noise_sample_tokens": args.noise_sample_tokens,
            "condition_summaries": summaries,
            "pairwise_agreement": agreement,
            "score_dirs": {meta["label"]: str(path) for path, meta, _ in runs},
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    print(
        f"Combined {len(labels)} conditions on {len(common_ids)} common rollouts "
        f"-> {output_dir}"
    )
    if reference_label is None:
        print("warning: no student/no-PI condition; entropy deltas are unavailable")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score", help="score one explicit teacher/PI condition")
    score.add_argument("--student-model", required=True, help="Model that generated the rollouts.")
    score.add_argument("--teacher-model", required=True, help="Model scoring the fixed rollouts.")
    score.add_argument(
        "--pi",
        required=True,
        choices=tuple(PI_ALIASES),
        help="Teacher PI; solution aliases the repo's full mode.",
    )
    score.add_argument("--student-revision")
    score.add_argument("--teacher-revision")
    score.add_argument("--dataset", default="deepmath", choices=tuple(DATASET_REGISTRY_TRAIN))
    score.add_argument(
        "--rollouts",
        help="HF rollout dataset; defaults to data/rollouts/<dataset>/<student slug>.",
    )
    score.add_argument("--rollout-root", default="data/rollouts")
    score.add_argument("--output-root", default="results/token_logratios")
    score.add_argument("--output-dir", help="Exact condition output directory.")
    score.add_argument("--label", help="Condition label used by summarize.")
    score.add_argument("--max-rollouts", type=int)
    score.add_argument(
        "--rollout-max-tokens",
        type=int,
        help="For legacy caches, infer truncation when n_tokens reaches this cap.",
    )
    score.add_argument("--max-model-len", type=int, default=32768)
    score.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    score.add_argument("--device", default="cuda")
    score.add_argument(
        "--device-map",
        help="Transformers device_map such as auto; when set, --device is ignored.",
    )
    score.add_argument(
        "--prompt-template-source",
        choices=("teacher", "student"),
        default="teacher",
        help="Use the teacher's native template or the student's controlled boundary.",
    )
    score.add_argument("--entropy-chunk-size", type=int, default=ENTROPY_CHUNK)
    score.add_argument("--reference-atol", type=float, default=1e-4)
    score.add_argument("--save-every", type=int, default=8)
    score.add_argument("--keep-shards", action="store_true")
    score.add_argument("--force", action="store_true")

    summarize = commands.add_parser(
        "summarize", help="combine arbitrary compatible score directories"
    )
    summarize.add_argument("--score-dirs", nargs="+", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--position-bins", type=int, default=20)
    summarize.add_argument("--noise-quantile", type=float, default=0.99)
    summarize.add_argument("--noise-sample-tokens", type=int, default=1_000_000)
    summarize.add_argument("--bootstrap-samples", type=int, default=2000)
    summarize.add_argument("--seed", type=int, default=42)
    summarize.add_argument("--force", action="store_true")
    return parser


def validate_args(args, parser: argparse.ArgumentParser) -> None:
    if args.command == "score":
        if args.max_rollouts is not None and args.max_rollouts < 1:
            parser.error("--max-rollouts must be >= 1")
        if args.max_model_len < 2:
            parser.error("--max-model-len must be >= 2")
        if args.entropy_chunk_size < 1:
            parser.error("--entropy-chunk-size must be >= 1")
        if args.save_every < 1:
            parser.error("--save-every must be >= 1")
        if args.reference_atol < 0:
            parser.error("--reference-atol must be non-negative")
    else:
        if args.position_bins < 1:
            parser.error("--position-bins must be >= 1")
        if not 0 <= args.noise_quantile <= 1:
            parser.error("--noise-quantile must be between 0 and 1")
        if args.noise_sample_tokens < 1:
            parser.error("--noise-sample-tokens must be >= 1")
        if args.bootstrap_samples < 1:
            parser.error("--bootstrap-samples must be >= 1")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    if args.command == "score":
        score_condition(args)
    else:
        summarize_runs(args)


if __name__ == "__main__":
    main()
