"""Frozen-model PI gain on complete reasoning demonstrations. See docs/demo_gain.md.

Prepare a common cohort, generate hint variants in a separate process, score,
then aggregate. No training, sliding windows, or silent prompt/target truncation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from eval.hint_compare_cache import ConditionCache, digest, model_identity
from utils import (
    PI_FULL,
    PI_HINT,
    PI_ROLLOUT,
    answer_context,
    compose_pi_messages,
    format_prompt,
    load_hint_cache,
    load_train_dataset,
    rollout_path,
)

VERSION = 1
CORE = ("answer", "full", "rollout", "hint")
VARIANTS = ("hint_detailed", "hint_medium", "hint_short")
CONDITIONS = CORE + VARIANTS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def question_id(question, answer):
    return digest([str(question), str(answer)])[:24]


def messages_for(row, condition, hint=None):
    messages = row.get("base_messages") or format_prompt(row["question"], "deepmath")
    if condition == "none":
        return messages
    if condition == "answer":
        context = answer_context(row.get("answer_pi", row["final_answer"]), "deepmath")
    elif condition == "full":
        context = PI_FULL.format(demo=row["solution"])
    elif condition == "rollout":
        context = PI_ROLLOUT.format(attempt=row["rollout"])
    elif condition == "hint" or condition in VARIANTS:
        context = PI_HINT.format(hint=row["hint"] if hint is None else hint)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return compose_pi_messages(messages, context)


def render_target(tokenizer, messages, solution):
    """Get the actual assistant suffix, including terminator, without double think tags."""
    from train.sft.train_sft import format_think_completion

    completion = format_think_completion(solution)
    if completion is None:
        raise ValueError("malformed_thinking_trace")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": completion}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full.startswith(prompt):
        raise ValueError("assistant_template_prefix_mismatch")
    ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    all_ids = list(tokenizer(full, add_special_tokens=False)["input_ids"])
    if all_ids[: len(ids)] != ids:
        raise ValueError("assistant_token_boundary_mismatch")
    target = all_ids[len(ids) :]
    if not ids or not target:
        raise ValueError("empty_prompt_or_target")
    return ids, target


def load_rollouts(model, root, sample_idx, hints):
    """Select solely by sample index, never by verifier outcome."""
    from datasets import load_from_disk

    cache = load_from_disk(rollout_path(model, "deepmath", root))
    required = {
        "question",
        "question_idx",
        "completion_text",
        "sample_idx",
        "gen_model",
        "dataset",
        "question_source",
        "mixed_only",
    }
    if not required <= set(cache.column_names):
        raise ValueError(
            f"Rollout PI cache lacks {sorted(required - set(cache.column_names))}"
        )
    for field, expected in (
        ("gen_model", model),
        ("dataset", "deepmath"),
        ("question_source", "hints"),
        ("mixed_only", False),
    ):
        if set(cache.unique(field)) != {expected}:
            raise ValueError(f"Incompatible rollout PI cache: {field}")
    attempts = {}
    for row in cache.select_columns(sorted(required)):
        if int(row["sample_idx"]) != sample_idx:
            continue
        idx = int(row["question_idx"])
        if idx in attempts:
            raise ValueError(f"Duplicate rollout sample for index {idx}")
        if not 0 <= idx < len(hints) or hints[idx]["question"] != row["question"]:
            raise ValueError("Rollout and hint cache question identities differ")
        attempts[idx] = str(row["completion_text"])
    return attempts


def tokenizer_hash(tokenizer):
    return digest({"vocab": tokenizer.get_vocab(), "template": tokenizer.chat_template})


def prepare(args):
    from transformers import AutoConfig, AutoTokenizer

    if not args.model:
        raise ValueError("prepare requires --model")
    out = Path(args.output_dir)
    if (out / "manifest.json").exists():
        raise ValueError(
            "Cohort already prepared. Use score/aggregate or a new output directory."
        )
    conditions = args.conditions or list(CONDITIONS)
    config = AutoConfig.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    revision = getattr(config, "_commit_hash", None) or args.revision
    tok = AutoTokenizer.from_pretrained(
        args.model, revision=revision, trust_remote_code=True
    )
    native = getattr(config, "max_position_embeddings", None)
    limit = args.max_model_len or native
    if not limit or limit < 1 or (native and limit > native):
        raise ValueError(
            "Context limit must be positive and within the native model limit"
        )
    hints = load_hint_cache(args.model, "deepmath")
    attempts = (
        load_rollouts(
            args.model, args.rollout_pi_root, args.rollout_pi_sample_idx, hints
        )
        if "rollout" in conditions
        else {}
    )
    solutions, ambiguous = {}, set()
    for row in load_train_dataset("deepmath", require_solution=True):
        key = (row["question"], str(row["final_answer"]))
        if key in solutions and solutions[key] != row["solution"]:
            ambiguous.add(key)
        solutions[key] = row["solution"]
    order = list(range(len(hints)))
    random.Random(args.seed).shuffle(order)
    rows, excluded, seen = [], Counter(), set()
    for idx in order:
        source = hints[idx]
        q, answer = str(source["question"]), str(source["final_answer"])
        qid = question_id(q, answer)
        if qid in seen:
            excluded["duplicate_question"] += 1
            continue
        seen.add(qid)
        if (q, answer) in ambiguous:
            excluded["ambiguous_demo"] += 1
            continue
        solution = solutions.get((q, answer))
        if not solution:
            excluded["missing_demo"] += 1
            continue
        if "rollout" in conditions and not attempts.get(idx):
            excluded["missing_rollout"] += 1
            continue
        row = {
            "question_id": qid,
            "question_idx": idx,
            "question": q,
            "final_answer": answer,
            "solution": solution,
            "hint": source["hint"],
            "rollout": attempts.get(idx),
            "demo_hash": digest(solution),
        }
        prompts, target = {}, None
        try:
            for condition in dict.fromkeys(
                ["none", "full"] + [c for c in conditions if c in CORE]
            ):
                ids, current = render_target(
                    tok, messages_for(row, condition), solution
                )
                if target is not None and current != target:
                    raise ValueError("target_tokens_differ_across_conditions")
                target = current
                if len(ids) + len(target) > limit:
                    raise ValueError(f"over_context_{condition}")
                prompts[condition] = ids
        except ValueError as error:
            excluded[str(error)] += 1
            continue
        row.update(prompt_ids=prompts, target_ids=target, target_tokens=len(target))
        rows.append(row)
        if args.num_problems and len(rows) >= args.num_problems:
            break
    if not rows:
        raise ValueError(f"No eligible demonstrations: {dict(excluded)}")
    manifest = {
        "version": VERSION,
        "model": args.model,
        "revision": revision,
        "model_identity": model_identity(args.model),
        "dataset": "deepmath",
        "demo_source": "r1_solution_1",
        "ambiguous_source_question_answers": len(ambiguous),
        "target_format": "complete_assistant_think_v1_including_terminator",
        "tokenizer_hash": tokenizer_hash(tok),
        "max_model_len": limit,
        "seed": args.seed,
        "n_questions": len(rows),
        "requested_questions": args.num_problems,
        "conditions": conditions,
        "rollout_pi_root": args.rollout_pi_root,
        "rollout_pi_sample_idx": args.rollout_pi_sample_idx,
        "rollout_selection": "fixed_sample_idx_without_reward",
        "exclusions_in_scanned_candidates": dict(excluded),
        "cohort_hash": digest(rows),
    }
    write_rows(out / "cohort.jsonl", rows)
    write_json(out / "manifest.json", manifest)
    print(f"Prepared {len(rows)} questions; exclusions: {dict(excluded)}", flush=True)


def load_cohort(output_dir):
    root = Path(output_dir)
    manifest, rows = read_json(root / "manifest.json"), read_rows(root / "cohort.jsonl")
    if manifest["version"] != VERSION or digest(rows) != manifest["cohort_hash"]:
        raise ValueError("Prepared cohort identity or contents changed")
    return manifest, rows


def load_variants(directory, cohort):
    root = Path(directory)
    meta, rows = read_json(root / "manifest.json"), read_rows(root / "hints.jsonl")
    if digest(rows) != meta["hints_hash"]:
        raise ValueError("Hint variants changed since generation")
    identities = {row["question_id"]: row for row in cohort}
    grouped = {}
    for row in rows:
        qid = row["question_id"]
        if qid not in identities:
            continue
        if row["demo_hash"] != identities[qid]["demo_hash"]:
            raise ValueError("Hint variant refers to a different demonstration")
        group = grouped.setdefault((qid, row["condition"]), [])
        if any(r["sample_idx"] == row["sample_idx"] for r in group):
            raise ValueError("Duplicate hint sample")
        group.append(row)
    for group in grouped.values():
        group.sort(key=lambda r: r["sample_idx"])
    return grouped, meta


def check_logps(values, length):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError("Invalid or misaligned demonstration log probabilities")
    return array


def score_tokens(model, prompt_ids, target_ids, device, block_size):
    """Bound logit memory by recomputing full prefixes; never reset or slide context."""
    import torch

    from utils.model_scoring import per_token_logps

    result = []
    for start in range(0, len(target_ids), block_size):
        end = min(start + block_size, len(target_ids))
        inputs = torch.tensor([prompt_ids + target_ids[:end]], device=device)
        targets = torch.tensor([target_ids[start:end]], device=device)
        with torch.inference_mode():
            result.extend(
                per_token_logps(model, inputs, targets).squeeze(0).cpu().tolist()
            )
    check_logps(result, len(target_ids))
    return result


def score(args):
    from transformers import AutoTokenizer

    out = Path(args.output_dir)
    manifest, cohort = load_cohort(out)
    conditions = args.conditions or manifest["conditions"]
    if args.model and args.model != manifest["model"]:
        raise ValueError("Scorer model differs from prepared model")
    if digest(model_identity(manifest["model"])) != digest(manifest["model_identity"]):
        raise ValueError("Local scoring model changed")
    pi_artifacts = []
    if manifest.get("dataset") == "am_qwen3_math":
        from eval.am_demo_gain import preflight_am

        jobs, exclusions, pi_artifacts, diagnostics = preflight_am(
            args, manifest, cohort, conditions
        )
        variant_meta = None
    else:
        variant_conditions = [c for c in conditions if c in VARIANTS]
        variants, variant_meta, tok = {}, None, None
        if variant_conditions:
            variants, variant_meta = load_variants(
                args.hint_variants_dir or out / "hints", cohort
            )
            tok = AutoTokenizer.from_pretrained(
                manifest["model"], revision=manifest["revision"], trust_remote_code=True
            )
            if tokenizer_hash(tok) != manifest["tokenizer_hash"]:
                raise ValueError("Scoring tokenizer changed")
        # Complete preflight before GPU loading. All conditions use one common set.
        jobs, exclusions = [], Counter()
        for row in cohort:
            arms = {
                "none": [{"prompt_ids": row["prompt_ids"]["none"], "sample_idx": 0}]
            }
            reason = None
            for condition in conditions:
                if condition in CORE:
                    if condition not in row["prompt_ids"]:
                        raise ValueError(
                            f"{condition} was not prepared; use a new cohort directory"
                        )
                    arms[condition] = [
                        {"prompt_ids": row["prompt_ids"][condition], "sample_idx": 0}
                    ]
                else:
                    hints = variants.get((row["question_id"], condition), [])
                    expected = variant_meta["config"]["samples_per_level"]
                    if {h["sample_idx"] for h in hints} != set(range(expected)):
                        raise ValueError(
                            f"Missing {condition} samples for {row['question_id']}; generate them first"
                        )
                    arms[condition] = []
                    for hint in hints:
                        if hint["invalid_reason"] or hint["truncated"]:
                            reason = f"invalid_or_truncated_{condition}"
                            break
                        ids, target = render_target(
                            tok,
                            messages_for(row, condition, hint["hint"]),
                            row["solution"],
                        )
                        if target != row["target_ids"]:
                            raise ValueError(
                                "Variant target tokens differ from baseline"
                            )
                        if len(ids) + len(target) > manifest["max_model_len"]:
                            reason = f"over_context_{condition}"
                            break
                        arms[condition].append(
                            {
                                "prompt_ids": ids,
                                "sample_idx": hint["sample_idx"],
                                "hint_tokens": hint["n_tokens"],
                            }
                        )
                if reason:
                    break
            if reason:
                exclusions[reason] += 1
            else:
                jobs.append((row, arms))
    if not jobs:
        raise ValueError(f"No common valid questions: {dict(exclusions)}")
    config = {
        "version": VERSION,
        "model_identity": manifest["model_identity"],
        "revision": manifest["revision"],
        "tokenizer_hash": manifest["tokenizer_hash"],
        "dtype": args.dtype,
        "block_size": args.block_size,
        "attention": "sdpa",
        "reduction": "fp32_untempered_batch_one",
    }
    cache = ConditionCache(out, "demo_logps", config, force=args.force)
    model, records = None, []
    for index, (row, arms) in enumerate(jobs):
        references = {}
        for condition, samples in arms.items():
            references[condition] = []
            for sample in samples:
                key = cache.key(
                    {
                        "prompt_ids": sample["prompt_ids"],
                        "target_ids": row["target_ids"],
                    }
                )
                saved = cache.load(key)
                if saved is None:
                    if model is None:
                        import torch
                        from transformers import AutoModelForCausalLM

                        model = (
                            AutoModelForCausalLM.from_pretrained(
                                manifest["model"],
                                revision=manifest["revision"],
                                dtype=getattr(torch, args.dtype),
                                attn_implementation="sdpa",
                                trust_remote_code=True,
                            )
                            .to(args.device)
                            .eval()
                        )
                        model.requires_grad_(False)
                    saved = {
                        "logps": score_tokens(
                            model,
                            sample["prompt_ids"],
                            row["target_ids"],
                            args.device,
                            args.block_size,
                        )
                    }
                    cache.save(key, saved)
                check_logps(saved["logps"], row["target_tokens"])
                references[condition].append(
                    {
                        "key": key,
                        "sample_idx": sample["sample_idx"],
                        "hint_tokens": sample.get("hint_tokens"),
                    }
                )
        records.append(
            {
                "question_id": row["question_id"],
                "n_tokens": row["target_tokens"],
                "thinking_end": row.get("thinking_end"),
                "scores": references,
            }
        )
        print(f"Scored/reused {index + 1}/{len(jobs)} questions", flush=True)
    write_json(
        out / "score_index.json",
        {
            "version": VERSION,
            "dataset": manifest.get("dataset", "deepmath"),
            "pi_artifacts": pi_artifacts,
            "cohort_hash": manifest["cohort_hash"],
            "conditions": conditions,
            "cache_config": config,
            "records": records,
            "cache_stats": cache.stats(),
            "n_prepared": len(cohort),
            "exclusions": dict(exclusions),
            "hint_variants_hash": variant_meta["hints_hash"] if variant_meta else None,
            "hint_variants_dir": str(
                Path(args.hint_variants_dir or out / "hints").resolve()
            )
            if variant_meta
            else None,
            "hint_generation_diagnostics": (
                diagnostics
                if pi_artifacts
                else variant_meta.get("diagnostics")
                if variant_meta
                else None
            ),
        },
    )


def token_metrics(gains, bins, early_tokens):
    gains = np.asarray(gains, dtype=np.float64)
    if not len(gains) or not np.isfinite(gains).all():
        raise ValueError("Empty or nonfinite gains")
    # Fractional boundaries distribute a boundary token proportionally. No empty
    # bins for short traces, and the final cumulative value equals total gain.
    edges = np.linspace(0, len(gains), bins + 1)
    cumulative = np.interp(
        edges, np.arange(len(gains) + 1), np.r_[0.0, np.cumsum(gains)]
    )
    early = [float(x) for x in gains[:early_tokens]]
    early.extend([None] * (early_tokens - len(early)))
    return {
        "total_gain": float(gains.sum()),
        "normalized_gain": float(gains.mean()),
        "position_gain": (np.diff(cumulative) / np.diff(edges)).tolist(),
        "cumulative_gain": cumulative.tolist(),
        "early_gain": early,
    }


def estimate(values, draws):
    """Question-balanced means and pointwise question-bootstrap intervals."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        boot = values[draws].mean(axis=1)
        return {
            "mean": float(values.mean()),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    support = np.isfinite(values).sum(axis=0)
    mean = np.divide(
        np.nansum(values, axis=0),
        support,
        out=np.full(values.shape[1], np.nan),
        where=support > 0,
    )
    boot = []
    # Avoid a bootstrap_samples x questions x early_tokens allocation.
    for indices in draws:
        selected = values[indices]
        counts = np.isfinite(selected).sum(axis=0)
        boot.append(
            np.divide(
                np.nansum(selected, axis=0),
                counts,
                out=np.full(values.shape[1], np.nan),
                where=counts > 0,
            )
        )
    boot = np.asarray(boot)
    intervals = [
        np.quantile(boot[np.isfinite(boot[:, j]), j], [0.025, 0.975]).tolist()
        if np.isfinite(boot[:, j]).any()
        else [None, None]
        for j in range(values.shape[1])
    ]
    return {
        "mean": [float(x) if np.isfinite(x) else None for x in mean],
        "ci95": intervals,
        "n_questions": support.tolist(),
    }


def aggregate(args):
    out = Path(args.output_dir)
    manifest, cohort = load_cohort(out)
    report_dir = out
    index = read_json(report_dir / "score_index.json")
    if index.get("pi_artifacts"):
        from eval.am_demo_gain import verify_artifacts

        verify_artifacts(index["pi_artifacts"], cohort)
    if index["cohort_hash"] != manifest["cohort_hash"]:
        raise ValueError("Score index belongs to another cohort")
    if args.conditions and args.conditions != index["conditions"]:
        raise ValueError("Requested conditions differ from score index; rerun score")
    if index.get("hint_variants_dir"):
        _, hint_meta = load_variants(index["hint_variants_dir"], cohort)
        if hint_meta["hints_hash"] != index["hint_variants_hash"]:
            raise ValueError("Hint artifacts changed; rerun score before aggregate")
    cache = ConditionCache(out, "demo_logps", index["cache_config"])
    records, by_arm = [], {condition: [] for condition in index["conditions"]}
    lengths = []
    for row in index["records"]:
        length = row["n_tokens"]
        lengths.append(length)
        baseline = cache.load(row["scores"]["none"][0]["key"])
        if baseline is None:
            raise ValueError("Incomplete baseline scores; rerun score")
        base = check_logps(baseline["logps"], length)
        for condition in index["conditions"]:
            samples = []
            for reference in row["scores"][condition]:
                saved = cache.load(reference["key"])
                if saved is None:
                    raise ValueError("Incomplete condition scores; rerun score")
                logps = check_logps(saved["logps"], length)
                gains = logps - base
                samples.append(gains)
                metrics = token_metrics(gains, args.position_bins, args.early_tokens)
                if row.get("thinking_end") is not None:
                    from eval.am_demo_gain import region_metrics

                    metrics.update(region_metrics(gains, row["thinking_end"]))
                records.append(
                    {
                        "question_id": row["question_id"],
                        "condition": condition,
                        "sample_idx": reference["sample_idx"],
                        "n_tokens": length,
                        "hint_tokens": reference["hint_tokens"],
                        "token_gains": gains.tolist(),
                        "baseline_logp": float(base.sum()),
                        "conditional_logp": float(logps.sum()),
                        **metrics,
                    }
                )
            averaged = np.mean(samples, axis=0)
            metrics = token_metrics(averaged, args.position_bins, args.early_tokens)
            if row.get("thinking_end") is not None:
                from eval.am_demo_gain import region_metrics

                metrics.update(region_metrics(averaged, row["thinking_end"]))
            by_arm[condition].append(metrics)
    if not lengths:
        raise ValueError("No scored questions")
    draws = np.random.default_rng(args.seed).integers(
        len(lengths), size=(args.bootstrap_samples, len(lengths))
    )
    summaries = {}
    for condition, rows in by_arm.items():
        summaries[condition] = {
            field: estimate([r[field] for r in rows], draws)
            for field in (
                "total_gain",
                "normalized_gain",
                "position_gain",
                "cumulative_gain",
                "early_gain",
            )
        }
        for field in rows[0]:
            if field not in summaries[condition]:
                summaries[condition][field] = estimate([r[field] for r in rows], draws)
        summaries[condition]["pooled_token_gain"] = sum(
            r["total_gain"] for r in rows
        ) / sum(lengths)
    comparisons = {}
    for a, b in itertools.combinations(index["conditions"], 2):
        comparisons[f"{a}_minus_{b}"] = {
            field: estimate(
                [
                    x[field] - y[field]
                    for x, y in zip(by_arm[a], by_arm[b], strict=True)
                ],
                draws,
            )
            for field in ("total_gain", "normalized_gain")
        }
    summary = {
        "dataset": manifest.get("dataset", "deepmath"),
        "source": manifest.get("source"),
        "region_definition": "thinking includes closing think tag; final includes answer wrapper and assistant terminator; early fraction is first 5%",
        "method": "frozen_demonstration_log_likelihood_gain",
        "version": VERSION,
        "model": manifest["model"],
        "cohort_hash": manifest["cohort_hash"],
        "score_index_hash": digest(index),
        "n_questions": len(lengths),
        "n_prepared": index["n_prepared"],
        "exclusions": index["exclusions"],
        "hint_generation_diagnostics": index["hint_generation_diagnostics"],
        "conditions": summaries,
        "paired_differences": comparisons,
        "position_bins": args.position_bins,
        "early_tokens": args.early_tokens,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "uncertainty_unit": "question",
        "curve_intervals": "pointwise_not_simultaneous",
        "aggregation": "mean_over_hint_samples_then_equal_weight_questions",
        "target_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": float(np.mean(lengths)),
        },
    }
    write_rows(report_dir / "per_question.jsonl", records)
    write_json(report_dir / "summary.json", summary)
    print(
        f"Aggregated {len(lengths)} paired questions -> {report_dir / 'summary.json'}"
    )
    return summary


def build_parser(*, deepmath=True):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase", choices=["prepare", "score", "aggregate", "all"], default="all"
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument(
        "--num-problems", type=int, default=128, help="0 selects all eligible questions"
    )
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=None)
    if deepmath:
        p.add_argument("--rollout-pi-root", default="data/pi/attempted_solution_8k")
        p.add_argument("--rollout-pi-sample-idx", type=int, default=0)
        p.add_argument("--hint-variants-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--position-bins", type=int, default=20)
    p.add_argument("--early-tokens", type=int, default=1024)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute requested scores; never replaces the cohort",
    )
    return p


def validate_args(args):
    if (
        min(
            args.block_size,
            args.position_bins,
            args.early_tokens,
            args.bootstrap_samples,
        )
        < 1
    ):
        raise ValueError(
            "Block size, bin count, early tokens, and bootstrap count must be positive"
        )
    if args.num_problems < 0 or getattr(args, "rollout_pi_sample_idx", 0) < 0:
        raise ValueError("Question count and rollout sample index must be nonnegative")
    if args.conditions and len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Duplicate conditions")


def main():
    args = build_parser().parse_args()
    validate_args(args)
    if args.phase in ("prepare", "all"):
        prepare(args)
    if args.phase in ("score", "all"):
        score(args)
    if args.phase in ("aggregate", "all"):
        aggregate(args)


if __name__ == "__main__":
    main()
