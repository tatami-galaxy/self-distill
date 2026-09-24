"""Frozen-model demonstration gain on AM-Qwen3 math.

Run prepare, utils.gen_am_demo_pi, score, aggregate, then eval.viz.demo_gain.
Each run has one common cohort and writes its report directly to --output-dir.
See docs/am_demo_gain.md. Token scoring and statistics are shared with demo_gain.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from eval import demo_gain as dg
from eval.hint_compare_cache import digest, model_identity

DATASET = "am_qwen3_math"
REPO = "a-m-team/AM-Qwen3-Distilled"
FALLBACK_SYSTEM = (
    "You are a helpful assistant. First think inside <think> </think> tags, "
    "then provide the final response inside <answer> </answer> tags."
)
GENERATED = ("rollout", "hint", *dg.VARIANTS)
HINTS = ("hint", *dg.VARIANTS)


def choice_answers(question, gold):
    """Resolve explicit A-E choices without guessing at missing option text."""
    letter = str(gold).strip().strip("() ").upper()
    if letter not in tuple("ABCDE"):
        return []
    text = re.sub(r"\\(?:textbf|mathrm|text|mathbf)\s*\{([^{}]*)\}", r"\1", question)
    matches = list(re.finditer(r"\(([A-E])\)|(?:^|\n)\s*([A-E])[.)]\s+", text))
    if len({m.group(1) or m.group(2) for m in matches}) < 2:
        return []
    for i, match in enumerate(matches):
        if (match.group(1) or match.group(2)) == letter:
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            answer = text[match.end() : end]
            answer = re.sub(r"\\(?:qquad|quad)|\\[ ,;!]", " ", answer).strip(" $\n\t.;")
            return [answer] if answer else []
    return []


def normalize_record(raw, row_number):
    turns = raw.get("conversations")
    if not isinstance(turns, list) or len(turns) != 2:
        raise ValueError("not_single_turn")
    user, assistant = turns
    if user.get("from") != "human" or assistant.get("from") != "assistant":
        raise ValueError("unexpected_roles")
    ui, ai = user.get("info") or {}, assistant.get("info") or {}
    if ui.get("category") != "math":
        raise ValueError("not_math")
    if ai.get("verify_score") != 1.0:
        raise ValueError("not_verified")
    q, gold, solution = (
        user.get("value"),
        ui.get("ground_truth"),
        assistant.get("value"),
    )
    if not all(isinstance(x, str) and x.strip() for x in (q, gold, solution)):
        raise ValueError("missing_question_answer_or_demo")
    thinking, answer = ai.get("think_content"), ai.get("answer_content")
    if not all(isinstance(x, str) and x.strip() for x in (thinking, answer)):
        raise ValueError("missing_trace_segments")
    match = re.fullmatch(
        r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*", solution, re.DOTALL
    )
    if not match or solution.count("</think>") != 1 or solution.count("<think>") != 1:
        raise ValueError("malformed_demo")
    if match[1].strip() != thinking.strip() or match[2].strip() != answer.strip():
        raise ValueError("inconsistent_trace_segments")
    system = raw.get("system")
    if system is not None and not isinstance(system, str):
        raise ValueError("invalid_system_prompt")
    options = choice_answers(q, gold)
    if gold.strip().strip("() ").upper() in tuple("ABCDE") and not options:
        raise ValueError("unresolved_multiple_choice")
    return {
        "question_id": digest([REPO, q.strip(), gold.strip()])[:24],
        "question_idx": row_number,
        "question": q,
        "final_answer": gold.strip(),
        "answer_aliases": options,
        "answer_pi": f"{gold.strip()} ({options[0]})" if options else gold.strip(),
        "solution": solution,
        "demo_hash": digest(solution),
        "source_row": row_number,
        "source_dataset": ui.get("source"),
        "verification_score": ai["verify_score"],
        "used_fallback_system": not bool(system and system.strip()),
        "base_messages": [
            {
                "role": "system",
                "content": system if system and system.strip() else FALLBACK_SYSTEM,
            },
            {"role": "user", "content": q},
        ],
    }


def scan_candidates(path, seed):
    """One sequential pass; retain offsets, not the 14+ GB of trace text.

    First verified occurrence wins for repeated questions with the same gold.
    Conflicting gold answers exclude the question entirely, independent of order.
    """
    seen, conflicts, excluded = {}, set(), Counter()
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        row_number = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            checksum.update(line)
            try:
                row = normalize_record(json.loads(line), row_number)
            except (ValueError, TypeError, AttributeError) as error:
                excluded[
                    str(error)
                    if isinstance(error, ValueError)
                    and not isinstance(error, json.JSONDecodeError)
                    else "malformed_record"
                ] += 1
                row_number += 1
                continue
            key = digest(row["question"].strip())
            if key in seen:
                excluded["duplicate_question"] += 1
                if seen[key][0] != row["final_answer"]:
                    conflicts.add(key)
            else:
                seen[key] = (
                    row["final_answer"],
                    offset,
                    row_number,
                    digest([seed, row["question_id"]]),
                )
            row_number += 1
    excluded["conflicting_gold_questions"] = len(conflicts)
    candidates = sorted(
        (v[3], v[1], v[2]) for k, v in seen.items() if k not in conflicts
    )
    return candidates, excluded, checksum.hexdigest(), row_number


def prepare_am(args):
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    if not args.model:
        raise ValueError("prepare requires --model")
    out = Path(args.output_dir)
    if (out / "manifest.json").exists():
        raise ValueError("Cohort already prepared; use a new output directory")
    path = (
        Path(args.data_file)
        if args.data_file
        else Path(
            hf_hub_download(
                REPO, "math.jsonl", repo_type="dataset", revision=args.dataset_revision
            )
        )
    )
    config = AutoConfig.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    revision = getattr(config, "_commit_hash", None) or args.revision
    tok = AutoTokenizer.from_pretrained(
        args.model, revision=revision, trust_remote_code=True
    )
    native = config.max_position_embeddings
    limit = args.max_model_len or native
    if not 0 < limit <= native:
        raise ValueError("Context limit must be positive and within native model limit")
    print(f"Scanning {path} for verified math records (CPU only)", flush=True)
    candidates, source_exclusions, checksum, total = scan_candidates(path, args.seed)
    print(
        f"Indexed {len(candidates)} unique candidates from {total} records", flush=True
    )
    rows, excluded = [], Counter()
    end_ids = tok.encode("</think>", add_special_tokens=False)
    with path.open("rb") as handle:
        for _, offset, number in candidates:
            handle.seek(offset)
            row = normalize_record(json.loads(handle.readline()), number)
            prompts, target = {}, None
            try:
                for condition in ("none", "answer", "full"):
                    ids, current = dg.render_target(
                        tok, dg.messages_for(row, condition), row["solution"]
                    )
                    if target is not None and current != target:
                        raise ValueError("target_tokens_differ_across_conditions")
                    target = current
                    if len(ids) + len(target) > limit:
                        raise ValueError(f"over_context_{condition}")
                    prompts[condition] = ids
                boundaries = [
                    i + len(end_ids)
                    for i in range(len(target))
                    if target[i : i + len(end_ids)] == end_ids
                ]
                if len(boundaries) != 1 or not 0 < boundaries[0] < len(target):
                    raise ValueError("invalid_thinking_boundary")
            except ValueError as error:
                excluded[str(error)] += 1
                continue
            row.update(
                prompt_ids=prompts,
                target_ids=target,
                target_tokens=len(target),
                thinking_end=boundaries[0],
            )
            rows.append(row)
            if args.num_problems and len(rows) >= args.num_problems:
                break
    if not rows:
        raise ValueError(f"No eligible AM demonstrations: {dict(excluded)}")
    manifest = {
        "version": dg.VERSION,
        "dataset": DATASET,
        "model": args.model,
        "revision": revision,
        "model_identity": model_identity(args.model),
        "tokenizer_hash": dg.tokenizer_hash(tok),
        "source": {
            "repo": REPO,
            "revision": args.dataset_revision if not args.data_file else None,
            "file": "math.jsonl",
            "local_path": str(path.absolute()),
            "sha256": checksum,
            "total_records": total,
            "eligible_unique_candidates": len(candidates),
        },
        "demo_source": "assistant.value; checked against think_content and answer_content",
        "target_format": "am_think_answer_v1_including_terminator",
        "selection": "seeded_hash_order_after_verification; first_verified_duplicate; conflicting_gold_excluded",
        "max_model_len": limit,
        "seed": args.seed,
        "requested_questions": args.num_problems,
        "n_questions": len(rows),
        "conditions": args.conditions or list(dg.CONDITIONS),
        "source_exclusions": dict(source_exclusions),
        "exclusions_in_scanned_candidates": dict(excluded),
        "source_distribution": dict(Counter(r["source_dataset"] for r in rows)),
        "fallback_system_count": sum(r["used_fallback_system"] for r in rows),
        "cohort_hash": digest(rows),
    }
    dg.write_rows(out / "cohort.jsonl", rows)
    dg.write_json(out / "manifest.json", manifest)
    print(
        f"Prepared {len(rows)} AM questions -> {out}; exclusions {dict(excluded)}",
        flush=True,
    )
    return manifest


def load_artifact(directory, cohort):
    directory = Path(directory)
    meta = dg.read_json(directory / "manifest.json")
    rows = dg.read_rows(directory / "samples.jsonl")
    if meta.get("dataset") != DATASET or meta["cohort_hash"] != digest(cohort):
        raise ValueError("Generated PI belongs to another dataset or cohort")
    if digest(rows) != meta["samples_hash"]:
        raise ValueError("Generated PI contents changed")
    identities = {r["question_id"]: r["demo_hash"] for r in cohort}
    grouped = {}
    for row in rows:
        if identities.get(row["question_id"]) != row["demo_hash"]:
            raise ValueError("Generated PI refers to another demonstration")
        key = (row["question_id"], row["condition"])
        group = grouped.setdefault(key, [])
        if any(x["sample_idx"] == row["sample_idx"] for x in group):
            raise ValueError("Duplicate PI sample")
        group.append(row)
    reference = {
        "directory": str(directory.resolve()),
        "samples_hash": meta["samples_hash"],
        "config_hash": digest(meta["config"]),
    }
    return grouped, meta, reference


def verify_artifacts(references, cohort):
    for reference in references:
        _, _, current = load_artifact(reference["directory"], cohort)
        if current != reference:
            raise ValueError("PI artifacts changed; rerun score before aggregate")


def preflight_am(args, manifest, cohort, conditions):
    from transformers import AutoTokenizer

    root = Path(getattr(args, "pi_dir", None) or args.output_dir)
    required = [c for c in conditions if c in GENERATED]
    generated, artifacts, diagnostics, metas = {}, [], {}, {}
    for kind in ("hints", "rollouts"):
        wanted = [c for c in required if (c == "rollout") == (kind == "rollouts")]
        if not wanted:
            continue
        groups, meta, reference = load_artifact(root / kind, cohort)
        if kind == "rollouts" and meta["config"]["model"] != manifest["model"]:
            raise ValueError("Rollout PI must come from the frozen student model")
        artifacts.append(reference)
        generated.update(groups)
        diagnostics[kind] = meta["diagnostics"]
        for c in wanted:
            if c not in meta["conditions"]:
                raise ValueError(f"Missing {c}; generate PI first")
            metas[c] = meta
    tok = (
        AutoTokenizer.from_pretrained(
            manifest["model"], revision=manifest["revision"], trust_remote_code=True
        )
        if required
        else None
    )
    if tok is not None and dg.tokenizer_hash(tok) != manifest["tokenizer_hash"]:
        raise ValueError("Scoring tokenizer changed")
    jobs, exclusions = [], Counter()
    for row in cohort:
        arms = {"none": [{"prompt_ids": row["prompt_ids"]["none"], "sample_idx": 0}]}
        failures = []
        for condition in conditions:
            if condition in ("answer", "full"):
                arms[condition] = [
                    {"prompt_ids": row["prompt_ids"][condition], "sample_idx": 0}
                ]
                continue
            samples = generated.get((row["question_id"], condition), [])
            expected = (
                1
                if condition == "rollout"
                else metas[condition]["config"]["samples_per_level"]
            )
            if {h["sample_idx"] for h in samples} != set(range(expected)):
                raise ValueError(
                    f"Missing {condition} samples for {row['question_id']}"
                )
            arms[condition] = []
            for sample in sorted(samples, key=lambda x: x["sample_idx"]):
                if sample["invalid_reason"] or sample["truncated"]:
                    failures.append(f"invalid_or_truncated_{condition}")
                    continue
                changed = (
                    {**row, "rollout": sample["text"]}
                    if condition == "rollout"
                    else row
                )
                ids, target = dg.render_target(
                    tok,
                    dg.messages_for(changed, condition, sample["text"]),
                    row["solution"],
                )
                if target != row["target_ids"]:
                    raise ValueError("PI target tokens differ from baseline")
                if len(ids) + len(target) > manifest["max_model_len"]:
                    failures.append(f"over_context_{condition}")
                    continue
                arms[condition].append(
                    {
                        "prompt_ids": ids,
                        "sample_idx": sample["sample_idx"],
                        "hint_tokens": sample["n_tokens"]
                        if condition != "rollout"
                        else None,
                    }
                )
        if failures:
            # Unlike first-failure counts, these expose every failing condition.
            exclusions.update(set(failures))
        else:
            jobs.append((row, arms))
    return jobs, exclusions, artifacts, diagnostics


def region_metrics(gains, thinking_end):
    gains = np.asarray(gains, dtype=np.float64)
    if not 0 < thinking_end < len(gains):
        raise ValueError("Invalid thinking boundary")
    cut = len(gains) * 0.05
    early = float(
        np.interp(cut, np.arange(len(gains) + 1), np.r_[0.0, np.cumsum(gains)])
    )
    regions = {
        "thinking": (float(gains[:thinking_end].sum()), thinking_end),
        "final": (float(gains[thinking_end:].sum()), len(gains) - thinking_end),
        "first_5pct": (early, cut),
        "remaining_95pct": (float(gains.sum()) - early, len(gains) - cut),
    }
    return {
        f"{name}_{metric}": value if metric == "total_gain" else value / length
        for name, (value, length) in regions.items()
        for metric in ("total_gain", "normalized_gain")
    }


def build_parser():
    p = dg.build_parser(deepmath=False)
    p.description = __doc__
    p.add_argument("--data-file", help="Optional local AM math.jsonl")
    p.add_argument(
        "--dataset-revision", default="498448170567e330435019c5321faa0a15e19118"
    )
    p.add_argument("--pi-dir", help="Generated PI root; defaults to --output-dir")
    return p


def main():
    args = build_parser().parse_args()
    dg.validate_args(args)
    if args.phase in ("prepare", "all"):
        prepare_am(args)
    # Fail before scoring if this entry point receives a different experiment.
    manifest = dg.read_json(Path(args.output_dir) / "manifest.json")
    if manifest.get("dataset") != DATASET:
        raise ValueError("eval.am_demo_gain requires an AM-Qwen3 cohort")
    if args.phase in ("score", "all"):
        dg.score(args)
    if args.phase in ("aggregate", "all"):
        dg.aggregate(args)


if __name__ == "__main__":
    main()
