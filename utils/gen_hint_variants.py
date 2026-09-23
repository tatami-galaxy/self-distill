"""Generate detailed/medium/short hints for a prepared demo-gain cohort.

Independent prompts see the same question and first R1 trace. Outputs, including
invalid/truncated hints, are cached per request. No reward-based selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from eval.demo_gain import load_cohort, write_json, write_rows
from eval.hint_compare_cache import ConditionCache, digest, model_identity
from utils.gen_hints import HINT_VALIDATION_VERSION, leaks_answer
from utils.model_adapters import vllm_model_and_adapter

LEVELS = {
    "hint_detailed": (
        512,
        "Give a detailed hint explaining the main strategy, important intermediate relationships, and how the steps connect. Omit the final calculation and final answer.",
    ),
    "hint_medium": (
        128,
        "Give a moderately concise hint containing the main strategy and one or two essential intermediate insights. Omit routine derivations and the final answer.",
    ),
    "hint_short": (
        32,
        "Give a very short hint containing only the pivotal idea, substitution, or theorem needed to get started. Omit derivations and the final answer.",
    ),
}


def build_messages(row, condition, budget):
    return [
        {
            "role": "system",
            "content": "You extract useful mathematical hints from a worked solution. Output only the hint, without a thinking block. Never state or compute the final answer.",
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{row['question']}\n\nWorked solution (for your reference only):\n{row['solution']}\n\n"
                f"{LEVELS[condition][1]} Keep the hint complete and comfortably within {budget} tokens."
            ),
        },
    ]


def classify_hint(text, answer):
    if not text.strip():
        return "empty"
    if "<think>" in text or "</think>" in text:
        return "thinking"
    if leaks_answer(text, answer):
        return "answer_leak"
    return ""


def generate(args):
    from transformers import AutoConfig, AutoTokenizer

    manifest, cohort = load_cohort(args.cohort_dir)
    root = Path(args.output_dir or Path(args.cohort_dir) / "hints")
    if root.resolve() == Path(args.cohort_dir).resolve():
        raise ValueError(
            "Hint output must be separate from the prepared cohort directory"
        )
    model = args.model or manifest["model"]
    kwargs, adapter, spec = vllm_model_and_adapter(model)
    requested_revision = args.revision or (
        manifest["revision"] if model == manifest["model"] else None
    )
    model_config = AutoConfig.from_pretrained(
        spec.base_model, revision=requested_revision, trust_remote_code=True
    )
    revision = getattr(model_config, "_commit_hash", None) or requested_revision
    tok = AutoTokenizer.from_pretrained(
        spec.base_model, revision=revision, trust_remote_code=True
    )
    config = {
        "version": 1,
        "model": model,
        "model_identity": model_identity(model),
        "base_identity": model_identity(spec.base_model),
        "revision": revision,
        "template_hash": digest(tok.chat_template),
        "vocab_hash": digest(tok.get_vocab()),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "samples_per_level": args.samples_per_level,
        "validation_version": HINT_VALIDATION_VERSION,
        "enable_thinking": False,
        "max_model_len": args.max_model_len,
    }
    cache_config = {k: v for k, v in config.items() if k != "samples_per_level"}
    cache = ConditionCache(root, "generation", cache_config, force=args.force)
    results, pending = [], []
    for row in cohort:
        for condition in args.levels:
            budget = getattr(args, condition.removeprefix("hint_") + "_budget")
            messages = build_messages(row, condition, budget)
            prompt_ids = tok.apply_chat_template(
                [messages],
                return_dict=True,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )["input_ids"][0]
            for sample_idx in range(args.samples_per_level):
                seed = int(
                    digest([args.seed, row["question_id"], condition, sample_idx])[:8],
                    16,
                ) % (2**31)
                request = {
                    "messages": messages,
                    "max_tokens": budget,
                    "seed": seed,
                    "answer": row["final_answer"],
                }
                key = cache.key(request)
                saved = cache.load(key)
                identity = {
                    "question_id": row["question_id"],
                    "demo_hash": row["demo_hash"],
                    "condition": condition,
                    "sample_idx": sample_idx,
                    "budget": budget,
                    "request_key": key,
                    "seed": seed,
                }
                if saved is not None:
                    results.append({**identity, **saved})
                elif len(prompt_ids) + budget > args.max_model_len:
                    saved = {
                        "hint": "",
                        "token_ids": [],
                        "n_tokens": 0,
                        "truncated": False,
                        "invalid_reason": "over_generator_context",
                        "finish_reason": None,
                    }
                    cache.save(key, saved)
                    results.append({**identity, **saved})
                else:
                    pending.append((identity, request, key))
    if pending:
        from vllm import LLM, SamplingParams

        llm = LLM(
            **kwargs,
            revision=revision,
            tokenizer_revision=revision,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
            trust_remote_code=True,
        )
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            params = [
                SamplingParams(
                    max_tokens=request["max_tokens"],
                    seed=request["seed"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                for _, request, _ in batch
            ]
            outputs = llm.chat(
                [request["messages"] for _, request, _ in batch],
                params,
                lora_request=adapter,
                chat_template_kwargs={"enable_thinking": False},
            )
            for (identity, request, key), output in zip(batch, outputs, strict=True):
                if len(output.outputs) != 1:
                    raise ValueError("Expected one completion per hint request")
                completion = output.outputs[0]
                hint = completion.text.strip()
                saved = {
                    "hint": hint,
                    "token_ids": list(completion.token_ids),
                    "n_tokens": len(completion.token_ids),
                    "truncated": completion.finish_reason == "length",
                    "invalid_reason": classify_hint(hint, request["answer"]),
                    "finish_reason": completion.finish_reason,
                }
                cache.save(key, saved)
                results.append({**identity, **saved})
            print(
                f"Generated {min(start + len(batch), len(pending))}/{len(pending)} uncached hints",
                flush=True,
            )
    results.sort(key=lambda r: (r["question_id"], r["condition"], r["sample_idx"]))
    diagnostics = {}
    for condition in args.levels:
        subset = [r for r in results if r["condition"] == condition]
        diagnostics[condition] = {
            "n_hints": len(subset),
            "mean_tokens": sum(r["n_tokens"] for r in subset) / len(subset),
            "invalid_counts": dict(
                Counter(r["invalid_reason"] for r in subset if r["invalid_reason"])
            ),
            "n_truncated": sum(r["truncated"] for r in subset),
        }
    write_rows(root / "hints.jsonl", results)
    write_json(
        root / "manifest.json",
        {
            "config": config,
            "cohort_hash": manifest["cohort_hash"],
            "hints_hash": digest(results),
            "levels": args.levels,
            "diagnostics": diagnostics,
            "cache_stats": cache.stats(),
        },
    )
    print(f"Saved {len(results)} hints -> {root}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--model",
        default=None,
        help="Defaults to the cohort's frozen model; override for a shared generator",
    )
    p.add_argument("--revision", default=None)
    p.add_argument("--levels", nargs="+", choices=list(LEVELS), default=list(LEVELS))
    for name, (budget, _) in LEVELS.items():
        p.add_argument(
            "--" + name.removeprefix("hint_") + "-budget", type=int, default=budget
        )
    p.add_argument("--samples-per-level", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--force", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if (
        min(
            args.samples_per_level,
            args.batch_size,
            args.max_model_len,
            args.tensor_parallel_size,
            args.short_budget,
            args.medium_budget,
            args.detailed_budget,
        )
        < 1
    ):
        raise ValueError("Counts and token budgets must be positive")
    if (
        not 0 <= args.temperature
        or not 0 < args.top_p <= 1
        or not 0 < args.gpu_memory_utilization < 1
    ):
        raise ValueError("Invalid sampling or memory settings")
    if len(set(args.levels)) != len(args.levels):
        raise ValueError("Duplicate hint levels")
    generate(args)


if __name__ == "__main__":
    main()
