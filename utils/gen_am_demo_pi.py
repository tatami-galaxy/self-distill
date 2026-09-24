"""Generate AM demonstration-gain PI in one stage.

By default, generate the cohort's requested hints and rollout PI in sequential,
separate processes. --kind hints or --kind rollouts runs just that component.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from eval import demo_gain as dg
from eval.am_demo_gain import DATASET, HINTS
from eval.hint_compare_cache import ConditionCache, digest, model_identity
from utils.am_hint_validation import VERSION as VALIDATION_VERSION
from utils.am_hint_validation import validation_flags
from utils.gen_hint_variants import LEVELS
from utils.gen_hints import build_messages as original_hint_messages
from utils.model_adapters import vllm_model_and_adapter


def hint_messages(row, condition, target):
    if condition == "hint":
        return original_hint_messages(row["question"], row["solution"])
    return [
        {
            "role": "system",
            "content": (
                "Extract a mathematical hint from the worked solution. Output only the hint, "
                "without thinking or answer blocks. Never state the final answer, an equivalent "
                "expression, or the correct multiple-choice option or its conclusion."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{row['question']}\n\nWorked solution (reference only):\n{row['solution']}\n\n"
                f"{LEVELS[condition][1]} Aim for at most {target} tokens. Finish the hint in a complete sentence."
            ),
        },
    ]


def generate(args):
    from transformers import AutoConfig, AutoTokenizer

    manifest, cohort = dg.load_cohort(args.cohort_dir)
    if manifest.get("dataset") != DATASET:
        raise ValueError("This generator requires an AM-Qwen3 cohort")
    model = args.model or manifest["model"]
    if args.kind == "rollouts" and model != manifest["model"]:
        raise ValueError("Rollouts must use the frozen student model")
    root = Path(args.output_dir or Path(args.cohort_dir) / args.kind)
    if root.resolve() == Path(args.cohort_dir).resolve():
        raise ValueError("PI output must be separate from cohort files")
    conditions = list(args.levels or HINTS) if args.kind == "hints" else ["rollout"]
    kwargs, adapter, spec = vllm_model_and_adapter(model)
    requested = args.revision or (
        manifest["revision"] if model == manifest["model"] else None
    )
    mc = AutoConfig.from_pretrained(
        spec.base_model, revision=requested, trust_remote_code=True
    )
    revision = getattr(mc, "_commit_hash", None) or requested
    tok = AutoTokenizer.from_pretrained(
        spec.base_model, revision=revision, trust_remote_code=True
    )
    limit = args.max_model_len or min(
        manifest["max_model_len"], mc.max_position_embeddings
    )
    if not 0 < limit <= mc.max_position_embeddings:
        raise ValueError("Generator context limit exceeds native limit")
    cap = args.max_new_tokens or (1024 if args.kind == "hints" else 8192)
    targets = {
        "hint": args.hint_target,
        "hint_detailed": args.detailed_target,
        "hint_medium": args.medium_target,
        "hint_short": args.short_target,
    }
    if args.kind == "hints" and any(targets[c] > cap for c in conditions):
        raise ValueError(
            "Generation ceiling must be at least the requested hint length"
        )
    config = {
        "version": 1,
        "dataset": DATASET,
        "kind": args.kind,
        "model": model,
        "model_identity": model_identity(model),
        "base_identity": model_identity(spec.base_model),
        "revision": revision,
        "tokenizer_hash": dg.tokenizer_hash(tok),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_model_len": limit,
        "max_new_tokens": cap,
        "samples_per_level": args.samples_per_level if args.kind == "hints" else 1,
        "enable_thinking": args.kind == "rollouts",
        "validation_version": VALIDATION_VERSION,
        "selection": "fixed_sample_index_no_correctness_filter_no_retries",
    }
    if args.kind == "rollouts" and (
        digest(model_identity(model)) != digest(manifest["model_identity"])
        or revision != manifest["revision"]
        or dg.tokenizer_hash(tok) != manifest["tokenizer_hash"]
    ):
        raise ValueError("Rollout model/tokenizer differs from the frozen scorer")
    cache = ConditionCache(
        root,
        "generation",
        {k: v for k, v in config.items() if k != "samples_per_level"},
        force=args.force,
    )
    results, pending = [], []
    for row in cohort:
        for condition in conditions:
            messages = (
                row["base_messages"]
                if condition == "rollout"
                else hint_messages(row, condition, targets[condition])
            )
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=condition == "rollout",
            )
            prompt_len = len(tok.encode(prompt, add_special_tokens=False))
            for sample_idx in range(config["samples_per_level"]):
                seed = int(
                    digest([args.seed, row["question_id"], condition, sample_idx])[:8],
                    16,
                ) % (2**31)
                request = {
                    "messages": messages,
                    "seed": seed,
                    "max_tokens": cap,
                    "final_answer": row["final_answer"],
                    "answer_aliases": row.get("answer_aliases", []),
                    "requested_tokens": targets.get(condition),
                }
                key = cache.key(request)
                identity = {
                    "question_id": row["question_id"],
                    "demo_hash": row["demo_hash"],
                    "condition": condition,
                    "sample_idx": sample_idx,
                    "seed": seed,
                    "request_key": key,
                    "requested_tokens": targets.get(condition),
                }
                saved = cache.load(key)
                if saved is not None:
                    results.append({**identity, **saved})
                elif prompt_len + cap > limit:
                    saved = {
                        "text": "",
                        "token_ids": [],
                        "n_tokens": 0,
                        "truncated": False,
                        "finish_reason": None,
                        "invalid_reason": "over_generator_context",
                        "validation_flags": ["over_generator_context"],
                        "length_noncompliant": False,
                    }
                    cache.save(key, saved)
                    results.append({**identity, **saved})
                else:
                    pending.append((row, identity, request, key))
    if pending:
        from vllm import LLM, SamplingParams

        llm = LLM(
            **kwargs,
            revision=revision,
            tokenizer_revision=revision,
            max_model_len=limit,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
            trust_remote_code=True,
            enforce_eager=args.enforce_eager,
        )
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            params = [
                SamplingParams(
                    max_tokens=req["max_tokens"],
                    seed=req["seed"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                for _, _, req, _ in batch
            ]
            outputs = llm.chat(
                [req["messages"] for _, _, req, _ in batch],
                params,
                lora_request=adapter,
                chat_template_kwargs={"enable_thinking": args.kind == "rollouts"},
            )
            for (row, identity, _, key), output in zip(batch, outputs, strict=True):
                if len(output.outputs) != 1:
                    raise ValueError("Expected one completion per request")
                completion = output.outputs[0]
                text = completion.text.strip()
                flags = (
                    validation_flags(text, row)
                    if args.kind == "hints"
                    else ([] if text else ["empty"])
                )
                if completion.finish_reason not in ("stop", "length"):
                    flags.append("unexpected_finish_reason")
                tokens = list(completion.token_ids)
                saved = {
                    "text": text,
                    "token_ids": tokens,
                    "n_tokens": len(tokens),
                    "finish_reason": completion.finish_reason,
                    "truncated": completion.finish_reason == "length",
                    "invalid_reason": flags[0] if flags else "",
                    "validation_flags": flags,
                    "length_noncompliant": args.kind == "hints"
                    and len(tokens) > identity["requested_tokens"],
                }
                cache.save(key, saved)
                results.append({**identity, **saved})
            print(
                f"Generated {min(start + len(batch), len(pending))}/{len(pending)} uncached {args.kind}",
                flush=True,
            )
    results.sort(key=lambda r: (r["question_id"], r["condition"], r["sample_idx"]))
    diagnostics = {}
    for condition in conditions:
        rows = [r for r in results if r["condition"] == condition]
        diagnostics[condition] = {
            "n_samples": len(rows),
            "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows),
            "n_valid_complete": sum(
                not r["invalid_reason"] and not r["truncated"] for r in rows
            ),
            "n_truncated": sum(r["truncated"] for r in rows),
            "n_length_noncompliant": sum(r["length_noncompliant"] for r in rows),
            "validation_flags": dict(
                Counter(flag for r in rows for flag in r["validation_flags"])
            ),
        }
    dg.write_rows(root / "samples.jsonl", results)
    dg.write_json(
        root / "manifest.json",
        {
            "dataset": DATASET,
            "source": manifest["source"],
            "config": config,
            "cohort_hash": manifest["cohort_hash"],
            "conditions": conditions,
            "samples_hash": digest(results),
            "diagnostics": diagnostics,
            "cache_stats": cache.stats(),
        },
    )
    # Small audit file, with unmodified text and labels, for inspecting hint quality.
    question_map = {r["question_id"]: r for r in cohort}
    audit = [
        {
            **r,
            "question": question_map[r["question_id"]]["question"],
            "final_answer": question_map[r["question_id"]]["final_answer"],
            "answer_aliases": question_map[r["question_id"]].get("answer_aliases", []),
        }
        for r in results
    ]
    dg.write_rows(root / "audit.jsonl", audit)
    print(f"Saved {len(results)} samples -> {root}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-dir", required=True)
    p.add_argument(
        "--kind",
        choices=["all", "hints", "rollouts"],
        default="all",
        help="Default: generate all PI requested by the prepared cohort",
    )
    p.add_argument(
        "--output-dir",
        help="PI root in all mode; component directory in hints/rollouts mode",
    )
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--levels", nargs="+", choices=HINTS)
    p.add_argument(
        "--hint-target",
        type=int,
        default=128,
        help="Length diagnostic only; retains original hint prompt",
    )
    p.add_argument("--detailed-target", type=int, default=512)
    p.add_argument("--medium-target", type=int, default=128)
    p.add_argument("--short-target", type=int, default=32)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        help="Ceiling for hints in all/hints mode (1024), or rollouts in rollouts mode (8192)",
    )
    p.add_argument(
        "--rollout-max-new-tokens",
        type=int,
        default=8192,
        help="Rollout ceiling when --kind all",
    )
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--samples-per-level", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--enforce-eager", action="store_true", help="Skip compilation for small pilots"
    )
    p.add_argument("--force", action="store_true")
    return p


def run_generation(args):
    if args.kind != "all":
        return generate(args)
    import multiprocessing

    manifest = dg.read_json(Path(args.cohort_dir) / "manifest.json")
    if manifest.get("dataset") != DATASET:
        raise ValueError("This generator requires an AM-Qwen3 cohort")
    root = Path(args.output_dir or args.cohort_dir)
    hints = args.levels or [c for c in HINTS if c in manifest["conditions"]]
    context = multiprocessing.get_context("spawn")
    for kind in ("hints", "rollouts"):
        if (kind == "hints" and not hints) or (
            kind == "rollouts" and "rollout" not in manifest["conditions"]
        ):
            continue
        child = argparse.Namespace(**vars(args))
        child.kind = kind
        child.output_dir = str(root / kind)
        child.levels = hints if kind == "hints" else None
        if kind == "rollouts":
            # A custom hint generator must never become the student rollout model.
            child.model = child.revision = None
            child.samples_per_level = 1
            child.max_new_tokens = args.rollout_max_new_tokens
        process = context.Process(target=generate, args=(child,))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"AM {kind} generation failed (exit {process.exitcode})")


def main():
    args = build_parser().parse_args()
    if (
        min(
            args.hint_target,
            args.detailed_target,
            args.medium_target,
            args.short_target,
            args.samples_per_level,
            args.batch_size,
            args.tensor_parallel_size,
            args.rollout_max_new_tokens,
        )
        < 1
    ):
        raise ValueError("Counts and requested lengths must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError("Generation ceiling must be positive")
    if args.max_model_len is not None and args.max_model_len < 1:
        raise ValueError("Context limit must be positive")
    if (
        not 0 <= args.temperature
        or not 0 < args.top_p <= 1
        or not 0 < args.gpu_memory_utilization < 1
    ):
        raise ValueError("Invalid sampling or memory settings")
    if args.levels and len(set(args.levels)) != len(args.levels):
        raise ValueError("Duplicate hint levels")
    if args.kind == "rollouts" and (args.levels or args.samples_per_level != 1):
        raise ValueError("Rollout PI uses exactly one fixed sample per question")
    run_generation(args)


if __name__ == "__main__":
    main()
