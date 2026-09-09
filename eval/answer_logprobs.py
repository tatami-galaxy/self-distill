r"""Compare PI-conditioned teacher likelihood of students' final boxed answers to correctness.

Use ALL cached student rollouts; no question subsampling or new generation.
The teacher is the frozen generating model, under none/answer/hint/full PI.
Pooled correctness discrimination includes every usable question. Within-question
ranking is a separate analysis restricted to questions containing both outcomes.

CUDA_VISIBLE_DEVICES=0 python -m eval.answer_logprobs --model Qwen/Qwen3-1.7B
CUDA_VISIBLE_DEVICES=1 python -m eval.answer_logprobs --model Qwen/Qwen3-4B

Defaults read data/rollouts/deepmath/<model>/ and write
results/answer_logprobs/<model>/. Phases: prepare, score, aggregate, all (default).
Preparation and aggregation are CPU-only; scoring loads one frozen HF model.
No vLLM is imported or initialized. Completed rollout scores are reusable.

Primary scores are the sum and mean of log-probabilities over tokens overlapping
the final boxed answer's contents; whole-box scores are secondary. Original
completion IDs are preserved, including noncanonical tokenizations. Boundary
overlap with formatting is recorded because sub-token likelihood is undefined.
Only the causal prefix and preceding answer tokens condition each prediction.

Labels verify the selected box against the gold answer. Missing/malformed boxes,
unparseable labels, missing PI, and inputs exceeding the common context limit
are reported, never assigned artificial low likelihoods. No prompt truncation.
Likelihood is NOT a calibrated probability of correctness. Bootstrap intervals
resample questions, preserving dependence among rollouts and PI conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import numpy as np

from utils import (
    PI_ANSWER,
    PI_FULL,
    PI_HINT,
    compose_pi_messages,
    format_prompt_math,
    grade,
    load_hint_cache,
    load_train_dataset,
    rollout_path,
)

PI_MODES = ("none", "answer", "hint", "full")
SCORE_NAMES = ("answer_sum", "answer_mean", "box_sum", "box_mean")
SCHEMA_VERSION = 1


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def jsonl_rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_stamp(path):
    root = Path(path).resolve()
    return [
        (str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def experiment_config(args):
    return {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "revision": args.revision,
        "dataset": args.dataset,
        "rollout_cache": str(Path(args.rollout_cache).resolve()),
        "pi_modes": list(args.pi_modes),
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "teacher_prompts": {
            "answer": PI_ANSWER,
            "hint": PI_HINT,
            "full": PI_FULL,
            "plain": format_prompt_math("{question}"),
            "pi": compose_pi_messages(format_prompt_math("{question}"), "{pi}"),
        },
    }


def load_manifest(args):
    manifest = read_json(Path(args.output_dir) / "manifest.json")
    if manifest["config"] != experiment_config(args):
        raise ValueError(
            "Experiment settings changed; use the original settings or a new output directory"
        )
    return manifest


def final_box(text):
    """Last boxed command, with nested/escaped braces; reject an unfinished final box."""
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None, "no_box"
    match = matches[-1]
    depth = 1
    escaped = False
    for pos in range(match.end(), len(text)):
        char = text[pos]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0:
            start, end = match.end(), pos
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start == end:
                return None, "empty_box"
            return {
                "box_chars": [match.start(), pos + 1],
                "answer_chars": [start, end],
                "answer": text[start:end],
                "box_count": len(matches),
            }, None
    return None, "malformed_final_box"


def token_spans(tokenizer, ids, text, spans):
    """Map character spans to ORIGINAL tokens, expanding any boundary overlap.

    Fast-tokenizer offsets are usable only if re-encoding reproduces the exact IDs.
    Otherwise search decoded original prefixes and validate both boundary strings.
    """
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"] if list(encoded["input_ids"]) == ids else None

    @cache
    def prefix(index):
        return tokenizer.decode(
            ids[:index], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def first_length_at_least(length):
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(prefix(mid)) < length:
                lo = mid + 1
            else:
                hi = mid
        return lo

    results = []
    for char_start, char_end in spans:
        if offsets is not None:
            indices = [
                i for i, (a, b) in enumerate(offsets) if a < char_end and b > char_start
            ]
            if not indices:
                raise ValueError("empty_token_span")
            start, end = indices[0], indices[-1] + 1
        else:
            start = max(0, first_length_at_least(char_start + 1) - 1)
            end = first_length_at_least(char_end)
        # Byte tokens can end inside a Unicode character; expand to valid boundaries.
        while start > 0 and (
            not text.startswith(prefix(start)) or len(prefix(start)) > char_start
        ):
            start -= 1
        while end < len(ids) and (
            not text.startswith(prefix(end)) or len(prefix(end)) < char_end
        ):
            end += 1
        before, after = prefix(start), prefix(end)
        if (
            not text.startswith(before)
            or not text.startswith(after)
            or not len(before) <= char_start < char_end <= len(after)
            or start >= end
        ):
            raise ValueError("unaligned_token_span")
        results.append(
            {
                "tokens": [start, end],
                "left_overlap_chars": char_start - len(before),
                "right_overlap_chars": len(after) - char_end,
            }
        )
    return results


def teacher_messages(problem, mode):
    messages = format_prompt_math(problem["question"])
    if mode == "none":
        return messages
    pi = {
        "answer": lambda: PI_ANSWER.format(answer=problem["final_answer"]),
        "hint": lambda: PI_HINT.format(hint=problem["hint"]),
        "full": lambda: PI_FULL.format(demo=problem["solution"]),
    }[mode]()
    return compose_pi_messages(messages, pi)


def context_index(rows, column, wanted):
    """Exact question+gold joins; conflicting context duplicates are unusable."""
    index = {}
    for row in rows:
        key = (str(row["question"]), str(row["final_answer"]))
        if key not in wanted:
            continue
        value = str(row.get(column) or "").strip()
        if key not in index:
            index[key] = value
        elif index[key] != value:
            index[key] = ""
    return index


def prepare(args):
    from datasets import load_from_disk
    from transformers import AutoConfig, AutoTokenizer

    out = Path(args.output_dir)
    if (out / "manifest.json").exists():
        manifest = load_manifest(args)
        if manifest["source_stamp"] != source_stamp(args.rollout_cache):
            raise ValueError("Rollout cache changed; use a new output directory")
        if file_hash(out / "prepared.jsonl") != manifest["prepared_sha256"]:
            raise ValueError("Prepared rollout records changed")
        print("Prepared cohort already exists; reusing all rows.")
        return
    ds = load_from_disk(args.rollout_cache)
    required = {
        "question",
        "final_answer",
        "completion_ids",
        "reward",
        "gen_model",
        "dataset",
    }
    if not required <= set(ds.column_names):
        raise ValueError(
            f"Rollout cache lacks columns: {sorted(required - set(ds.column_names))}"
        )
    if set(ds.unique("gen_model")) != {args.model} or set(ds.unique("dataset")) != {
        args.dataset
    }:
        raise ValueError(
            "Rollout cache model/dataset does not match the requested experiment"
        )
    if "mixed_only" in ds.column_names and any(ds.unique("mixed_only")):
        raise ValueError(
            "This experiment requires the full cache, not a mixed-outcome-only cache"
        )
    model_config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    revision = getattr(model_config, "_commit_hash", None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=revision,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    native_limit = getattr(model_config, "max_position_embeddings", None)
    context_limit = args.max_model_len or native_limit
    if not context_limit:
        raise ValueError(
            "Set --max-model-len; model config does not declare a context limit"
        )
    if native_limit and context_limit > native_limit:
        raise ValueError("--max-model-len exceeds the model's declared context length")
    keys = {
        (str(q), str(a))
        for q, a in zip(ds["question"], ds["final_answer"], strict=True)
    }
    hints, solutions = {}, {}
    if "hint" in args.pi_modes:
        hints = context_index(load_hint_cache(args.model, args.dataset), "hint", keys)
    if "full" in args.pi_modes:
        solutions = context_index(
            load_train_dataset(args.dataset, require_solution=True), "solution", keys
        )

    problems, missing_context = {}, set()
    for question, answer in sorted(keys):
        key = (question, answer)
        problem = {
            "question": question,
            "final_answer": answer,
            "hint": hints.get(key, ""),
            "solution": solutions.get(key, ""),
        }
        if (
            "hint" in args.pi_modes
            and not problem["hint"]
            or "full" in args.pi_modes
            and not problem["solution"]
        ):
            missing_context.add(key)
            continue
        qid = fingerprint(key)[:24]
        prompts = {
            mode: list(
                tokenizer.apply_chat_template(
                    [teacher_messages(problem, mode)],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                )["input_ids"][0]
            )
            for mode in args.pi_modes
        }
        problems[qid] = {**problem, "prompt_ids": prompts}

    out.mkdir(parents=True, exist_ok=True)
    exclusions = Counter()
    diagnostics = Counter()
    source_outcomes = defaultdict(list)
    kept_outcomes = defaultdict(list)
    gold_validity = {}
    seen_ids = set()
    temporary = out / "prepared.jsonl.tmp"
    with temporary.open("w") as handle:
        for row_index, row in enumerate(ds):
            key = (str(row["question"]), str(row["final_answer"]))
            qid = fingerprint(key)[:24]
            reward = float(row["reward"])
            if reward not in (0.0, 1.0):
                raise ValueError(f"Non-binary cached reward at row {row_index}")
            source_outcomes[qid].append(int(reward))
            if key in missing_context:
                exclusions["missing_or_ambiguous_pi"] += 1
                continue
            ids = list(row["completion_ids"])
            text = tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            box, reason = final_box(text)
            if reason:
                exclusions[reason] += 1
                continue
            if key not in gold_validity:
                gold_validity[key] = grade("\\boxed{" + key[1] + "}", key[1])[1]
            if not gold_validity[key]:
                exclusions["unparseable_gold"] += 1
                continue
            box_text = text[slice(*box["box_chars"])]
            prediction, correct = grade(box_text, key[1])
            if prediction is None:
                exclusions["unparseable_box"] += 1
                continue
            try:
                box_span, answer_span = token_spans(
                    tokenizer, ids, text, [box["box_chars"], box["answer_chars"]]
                )
            except ValueError as error:
                exclusions[str(error)] += 1
                continue
            box_start, box_end = box_span["tokens"]
            answer_start, answer_end = answer_span["tokens"]
            if not box_start <= answer_start < answer_end <= box_end:
                raise ValueError("Answer token span must be contained in the box span")
            if any(
                len(prompt) + box_end > context_limit
                for prompt in problems[qid]["prompt_ids"].values()
            ):
                exclusions["context_too_long"] += 1
                continue
            rid = fingerprint([qid, row_index, ids])[:24]
            if rid in seen_ids:
                raise ValueError("Duplicate prepared rollout identity")
            seen_ids.add(rid)
            record = {
                "rollout_id": rid,
                "question_id": qid,
                "source_row": row_index,
                "source_rollout_id": row.get("rollout_id"),
                "source_question_idx": row.get("question_idx"),
                "sample_idx": row.get("sample_idx"),
                "completion_ids": ids[:box_end],
                "original_completion_length": len(ids),
                "box": box,
                "box_span": box_span,
                "answer_span": answer_span,
                "parsed_answer": str(prediction),
                "correct": int(correct),
                "cached_reward": reward,
                "label_disagreement": bool(correct != bool(reward)),
                "truncated": row.get("truncated"),
                "finish_reason": row.get("finish_reason"),
                "answer_repeated_before_box": box["answer"]
                in text[: box["box_chars"][0]],
                "prefix_tail": text[
                    max(0, box["box_chars"][0] - 1200) : box["box_chars"][0]
                ],
                "box_inside_think": text.rfind("<think>", 0, box["box_chars"][0])
                > text.rfind("</think>", 0, box["box_chars"][0]),
            }
            diagnostics["label_disagreements"] += record["label_disagreement"]
            diagnostics["answer_boundary_overlap"] += bool(
                answer_span["left_overlap_chars"] or answer_span["right_overlap_chars"]
            )
            diagnostics["truncated_with_usable_box"] += record["truncated"] is True
            kept_outcomes[qid].append(int(correct))
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            if (row_index + 1) % 500 == 0:
                print(f"Prepared {row_index + 1}/{len(ds)} source rows", flush=True)
    temporary.replace(out / "prepared.jsonl")
    questions = {
        qid: problem for qid, problem in problems.items() if qid in kept_outcomes
    }
    write_json(out / "questions.json", questions)
    metadata_columns = (
        "generation_seed",
        "max_completion_length",
        "mixed_only",
        "question_source",
    )
    manifest = {
        "config": experiment_config(args),
        "model_revision": revision,
        "local_model_stamp": source_stamp(args.model)
        if Path(args.model).is_dir()
        else None,
        "source_stamp": source_stamp(args.rollout_cache),
        "generation_metadata": {
            key: ds.unique(key) if key in ds.column_names else None
            for key in metadata_columns
        },
        "legacy_missing_columns": [
            key
            for key in (
                "rollout_id",
                "question_idx",
                "sample_idx",
                "truncated",
                *metadata_columns,
            )
            if key not in ds.column_names
        ],
        "tokenizer_identity": fingerprint(
            [tokenizer.get_vocab(), tokenizer.chat_template, tokenizer.all_special_ids]
        ),
        "context_limit": context_limit,
        "source_rollouts": len(ds),
        "source_questions": len(source_outcomes),
        "source_mixed_questions": sum(
            0 < sum(v) < len(v) for v in source_outcomes.values()
        ),
        "scorable_rollouts": sum(map(len, kept_outcomes.values())),
        "scorable_questions": len(kept_outcomes),
        "scorable_mixed_questions": sum(
            0 < sum(v) < len(v) for v in kept_outcomes.values()
        ),
        "exclusions": dict(exclusions),
        "diagnostics": dict(diagnostics),
        "prepared_sha256": file_hash(out / "prepared.jsonl"),
        "questions_sha256": file_hash(out / "questions.json"),
    }
    if not kept_outcomes:
        raise ValueError(f"No usable boxed answers; exclusions: {dict(exclusions)}")
    write_json(out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "source_rollouts",
                    "source_questions",
                    "scorable_rollouts",
                    "scorable_questions",
                    "scorable_mixed_questions",
                    "exclusions",
                )
            },
            indent=2,
        )
    )


def score_box(model, prompt_ids, record, device):
    """Return original-box token logps; only the causal prefix through the box is supplied."""
    import torch

    from utils.model_scoring import per_token_logps

    start, end = record["box_span"]["tokens"]
    ids = record["completion_ids"]
    inputs = torch.tensor([prompt_ids + ids[:end]], dtype=torch.long, device=device)
    targets = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logps = (
            per_token_logps(model, inputs, targets).squeeze(0).float().cpu().tolist()
        )
    if len(logps) != end - start or not all(math.isfinite(v) for v in logps):
        raise ValueError("Invalid teacher answer log-probabilities")
    return logps


def reduce_box_logps(logps, record):
    start, _ = record["box_span"]["tokens"]
    a, b = record["answer_span"]["tokens"]
    answer_logps = logps[a - start : b - start]
    if len(answer_logps) != b - a or not answer_logps:
        raise ValueError("Answer log-probabilities are not aligned to the box")
    return {
        "answer_sum": sum(answer_logps),
        "answer_mean": float(np.mean(answer_logps)),
        "box_sum": sum(logps),
        "box_mean": float(np.mean(logps)),
        "answer_token_count": len(answer_logps),
        "box_token_count": len(logps),
        "box_token_logps": logps,
    }


def verify_prepared(out, manifest):
    for name, key in (
        ("prepared.jsonl", "prepared_sha256"),
        ("questions.json", "questions_sha256"),
    ):
        if file_hash(out / name) != manifest[key]:
            raise ValueError(f"{name} changed since preparation")


def score(args):
    import torch
    from transformers import AutoModelForCausalLM

    out = Path(args.output_dir)
    manifest = load_manifest(args)
    verify_prepared(out, manifest)
    signature = fingerprint(manifest)
    if (
        manifest["local_model_stamp"] is not None
        and source_stamp(args.model) != manifest["local_model_stamp"]
    ):
        raise ValueError("Local teacher checkpoint changed")
    questions = read_json(out / "questions.json")
    model = None
    completed = 0
    for record in jsonl_rows(out / "prepared.jsonl"):
        path = out / "scores" / (record["rollout_id"] + ".json")
        if path.exists():
            cached = read_json(path)
            if (
                cached["signature"] != signature
                or cached["rollout_id"] != record["rollout_id"]
                or set(cached["scores"]) != set(args.pi_modes)
            ):
                raise ValueError(f"Incompatible cached score: {path}")
        else:
            if model is None:
                model = (
                    AutoModelForCausalLM.from_pretrained(
                        args.model,
                        revision=manifest["model_revision"],
                        dtype=getattr(torch, args.dtype),
                        trust_remote_code=True,
                        local_files_only=args.local_files_only,
                    )
                    .to(args.device)
                    .eval()
                )
                model.requires_grad_(False)
            scores = {
                mode: reduce_box_logps(
                    score_box(
                        model,
                        questions[record["question_id"]]["prompt_ids"][mode],
                        record,
                        args.device,
                    ),
                    record,
                )
                for mode in args.pi_modes
            }
            write_json(
                path,
                {
                    "signature": signature,
                    "rollout_id": record["rollout_id"],
                    "scores": scores,
                },
            )
        completed += 1
        if completed % 50 == 0:
            print(
                f"Scored {completed}/{manifest['scorable_rollouts']} rollouts",
                flush=True,
            )
    print(f"All {completed} rollouts scored under {', '.join(args.pi_modes)}.")


def roc_auc(labels, scores):
    from scipy.stats import rankdata

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "std": None, "quantiles": None}
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": dict(
            zip(
                ("min", "p05", "p25", "median", "p75", "p95", "max"),
                np.quantile(values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
                strict=True,
            )
        ),
    }


def question_groups(question_ids):
    groups = defaultdict(list)
    for index, qid in enumerate(question_ids):
        groups[qid].append(index)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def comparison_metrics(labels, scores, groups):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    both = bool(labels.any() and (~labels).any())
    gap = float(scores[labels].mean() - scores[~labels].mean()) if both else None
    correlation = (
        float(np.corrcoef(labels.astype(float), scores)[0, 1])
        if both and np.ptp(scores)
        else None
    )
    within = []
    within_gaps = []
    for indices in groups:
        y, s = labels[indices], scores[indices]
        if y.any() and (~y).any():
            differences = s[y, None] - s[~y][None, :]
            within.append(float(((differences > 0) + 0.5 * (differences == 0)).mean()))
            within_gaps.append(float(s[y].mean() - s[~y].mean()))
    return {
        "roc_auc": roc_auc(labels, scores),
        "correct_minus_incorrect": gap,
        "point_biserial": correlation,
        "within_question_pair_accuracy": float(np.mean(within)) if within else None,
        "within_question_score_gap": float(np.mean(within_gaps))
        if within_gaps
        else None,
        "mixed_questions": len(within),
    }


def bootstrap_comparison(labels, scores, groups, *, samples, seed, baseline=None):
    """Resample whole questions; same draws are used for paired PI-vs-none AUC."""
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    # Compute each question's pair ranking once, not once per bootstrap draw.
    within = np.full(len(groups), np.nan)
    for index, group in enumerate(groups):
        y, s = labels[group], scores[group]
        if y.any() and (~y).any():
            differences = s[y, None] - s[~y][None, :]
            within[index] = ((differences > 0) + 0.5 * (differences == 0)).mean()
    for _ in range(samples):
        chosen = rng.integers(0, len(groups), len(groups))
        selected = [groups[i] for i in chosen]
        indices = np.concatenate(selected)
        y, s = labels[indices], scores[indices]
        auc = roc_auc(y, s)
        if auc is not None:
            draws["roc_auc"].append(auc)
            draws["correct_minus_incorrect"].append(float(s[y].mean() - s[~y].mean()))
            if baseline is not None:
                draws["auc_difference_vs_none"].append(
                    auc - roc_auc(y, baseline[indices])
                )
        valid = within[chosen]
        valid = valid[np.isfinite(valid)]
        if len(valid):
            draws["within_question_pair_accuracy"].append(float(np.mean(valid)))
    return {
        key: {
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "valid_draws": len(values),
        }
        for key, values in draws.items()
    }


def summarize(rows, modes, *, bootstrap_samples, seed):
    labels = np.asarray([r["correct"] for r in rows], dtype=bool)
    groups = question_groups([r["question_id"] for r in rows])
    result = {
        "rollouts": len(rows),
        "questions": len(groups),
        "correct": int(labels.sum()),
        "incorrect": int((~labels).sum()),
        "correct_fraction": float(labels.mean()),
        "conditions": {},
    }
    for mode in modes:
        condition = {}
        for name in SCORE_NAMES:
            values = np.asarray([r["scores"][mode][name] for r in rows])
            baseline = (
                np.asarray([r["scores"]["none"][name] for r in rows])
                if "none" in modes
                else None
            )
            metrics = comparison_metrics(labels, values, groups)
            metrics["correct_scores"] = describe(values[labels])
            metrics["incorrect_scores"] = describe(values[~labels])
            # Bootstrap primary content scores; box scores are secondary diagnostics.
            if name.startswith("answer_"):
                metrics["bootstrap"] = bootstrap_comparison(
                    labels,
                    values,
                    groups,
                    samples=bootstrap_samples,
                    seed=seed,
                    baseline=baseline if mode != "none" else None,
                )
            if baseline is not None:
                metrics["auc_difference_vs_none"] = (
                    metrics["roc_auc"] - roc_auc(labels, baseline)
                    if metrics["roc_auc"] is not None
                    else None
                )
                delta = values - baseline
                metrics["pi_minus_none"] = {
                    "correct": describe(delta[labels]),
                    "incorrect": describe(delta[~labels]),
                    "roc_auc": roc_auc(labels, delta),
                }
            strata = {}
            for label, mask in (
                (
                    "one_answer_token",
                    np.asarray([r["answer_token_count"] == 1 for r in rows]),
                ),
                (
                    "multiple_answer_tokens",
                    np.asarray([r["answer_token_count"] > 1 for r in rows]),
                ),
                (
                    "answer_repeated",
                    np.asarray([r["answer_repeated_before_box"] for r in rows]),
                ),
                (
                    "answer_not_repeated",
                    np.asarray([not r["answer_repeated_before_box"] for r in rows]),
                ),
            ):
                strata[label] = {
                    "n": int(mask.sum()),
                    "roc_auc": roc_auc(labels[mask], values[mask]),
                }
            metrics["strata"] = strata
            condition[name] = metrics
        result["conditions"][mode] = condition
    return result


def plot_distributions(rows, modes, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes, strict=True):
        scores = np.asarray([r["scores"][mode]["answer_mean"] for r in rows])
        bins = np.histogram_bin_edges(scores, bins=40)
        for correct, label, color in (
            (True, "Correct", "tab:blue"),
            (False, "Incorrect", "tab:orange"),
        ):
            selected = [
                s
                for s, row in zip(scores, rows, strict=True)
                if bool(row["correct"]) == correct
            ]
            if selected:
                ax.hist(
                    selected,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    label=label,
                    color=color,
                )
        ax.set(
            title=mode,
            xlabel="Mean answer log-probability (nats/token)",
            ylabel="Density",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "answer_score_distributions.png", dpi=160)
    plt.close(fig)


def aggregate(args):
    out = Path(args.output_dir)
    manifest = load_manifest(args)
    verify_prepared(out, manifest)
    signature = fingerprint(manifest)
    questions = read_json(out / "questions.json")
    rows = []
    temporary = out / "answers.jsonl.tmp"
    with temporary.open("w") as handle:
        for record in jsonl_rows(out / "prepared.jsonl"):
            path = out / "scores" / (record["rollout_id"] + ".json")
            if not path.exists():
                raise ValueError(
                    f"Missing score {path}; complete the score phase before aggregation"
                )
            cached = read_json(path)
            if (
                cached["signature"] != signature
                or cached["rollout_id"] != record["rollout_id"]
            ):
                raise ValueError(f"Score provenance mismatch: {path}")
            if set(cached["scores"]) != set(args.pi_modes):
                raise ValueError(f"Missing PI condition: {path}")
            row = {
                key: value for key, value in record.items() if key != "completion_ids"
            }
            row["question"] = questions[record["question_id"]]["question"]
            row["gold_answer"] = questions[record["question_id"]]["final_answer"]
            row["scores"] = cached["scores"]
            row["answer_token_count"] = (
                record["answer_span"]["tokens"][1] - record["answer_span"]["tokens"][0]
            )
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
    temporary.replace(out / "answers.jsonl")
    if len(rows) != manifest["scorable_rollouts"]:
        raise ValueError("Scored cohort size does not match preparation")
    summary = summarize(
        rows, args.pi_modes, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    summary["coverage"] = {
        key: manifest[key]
        for key in (
            "source_rollouts",
            "source_questions",
            "source_mixed_questions",
            "scorable_rollouts",
            "scorable_questions",
            "scorable_mixed_questions",
            "exclusions",
            "diagnostics",
            "legacy_missing_columns",
        )
    }
    summary["bootstrap_samples"] = args.bootstrap_samples
    summary["bootstrap_seed"] = args.seed
    write_json(out / "summary.json", summary)
    for mode in args.pi_modes:
        metric = summary["conditions"][mode]["answer_mean"]
        print(
            f"{mode}: pooled AUC={metric['roc_auc']}, "
            f"within-question pair accuracy={metric['within_question_pair_accuracy']}"
        )
    plot_distributions(rows, args.pi_modes, out)
    disagreements = {}
    for mode in args.pi_modes:
        for correct, label in (
            (False, "high_likelihood_incorrect"),
            (True, "low_likelihood_correct"),
        ):
            selected = [r for r in rows if bool(r["correct"]) == correct]
            disagreements[f"{mode}/{label}"] = sorted(
                selected,
                key=lambda r: r["scores"][mode]["answer_mean"],
                reverse=not correct,
            )[:20]
    write_json(out / "disagreements.json", disagreements)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision")
    parser.add_argument("--dataset", default="deepmath")
    parser.add_argument(
        "--rollout-cache", help="Defaults to data/rollouts/<dataset>/<model-slug>."
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--pi-modes", nargs="+", choices=PI_MODES, default=list(PI_MODES)
    )
    parser.add_argument(
        "--phase", choices=("prepare", "score", "aggregate", "all"), default="all"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="Common total context cap; defaults to model configuration.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/config/tokenizer from the local cache only.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Question-bootstrap seed; no question subsampling.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if len(set(args.pi_modes)) != len(args.pi_modes):
        parser.error("--pi-modes must not contain duplicates")
    if args.max_model_len is not None and args.max_model_len < 1:
        parser.error("--max-model-len must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be nonnegative")
    args.rollout_cache = args.rollout_cache or rollout_path(args.model, args.dataset)
    args.output_dir = args.output_dir or str(
        Path("results/answer_logprobs") / args.model.rstrip("/").split("/")[-1]
    )
    for phase, function in (
        ("prepare", prepare),
        ("score", score),
        ("aggregate", aggregate),
    ):
        if args.phase in (phase, "all"):
            function(args)


if __name__ == "__main__":
    main()
