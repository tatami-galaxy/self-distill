"""
Evaluate language models on MATH and other math benchmarks.

Usage:

    CUDA_VISIBLE_DEVICES=0 uv run python -m src.eval.run_eval \
    --model Qwen/Qwen2.5-Math-7B \
    --dataset math500 \
    --num_samples 10 \

"""

import argparse
import json
import os
import time
from collections import defaultdict
from vllm import LLM, SamplingParams
from utils import (
    extract_boxed_answer,
    is_equiv,
    DATASET_REGISTRY_EVAL,
)

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)

RAW_PROMPT = "Can you solve the following math problem? "
RAW_COT = " Please reason step by step, and put your final answer within \\boxed{}."


def format_prompt_chat(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def format_prompt_raw(problem: str) -> str:
    return RAW_PROMPT + problem + RAW_COT


def build_prompt(problem: str, prompt_mode: str, tokenizer, template_tok=None,
                 enable_thinking=None) -> str:
    """Build a prompt string from a problem, respecting prompt_mode."""
    if prompt_mode == "raw":
        return format_prompt_raw(problem)
    messages = format_prompt_chat(problem)
    tok = template_tok or tokenizer
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tok.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------    

def evaluate_model(
    model_name: str,
    problems: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 4096,
    chat_template_tokenizer=None,
    enable_thinking: bool | None = None,
    dtype: str = "bfloat16",
    prompt_mode: str = "chat",
) -> dict:
    """Run evaluation and return results dict."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Problems:   {len(problems)}")
    print(f"{'='*60}")

    # TODO : fix_mistral_regex=True?
    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype=dtype,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Build prompts
    template_tok = chat_template_tokenizer or tokenizer
    prompts = []
    for p in problems:
        prompts.append(build_prompt(
            p["problem"], prompt_mode, tokenizer, template_tok, enable_thinking,
        ))

    # Generate
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Generation took {elapsed:.1f}s ({len(problems)/elapsed:.1f} problems/s)")

    # Score
    results = []
    for prob, output in zip(problems, outputs):
        completion = output.outputs[0]
        response = completion.text
        pred_answer = extract_boxed_answer(response)
        correct = is_equiv(pred_answer, prob["answer"]) if pred_answer else False
        results.append({
            **prob,
            "response": response,
            "pred_answer": pred_answer,
            "correct": correct,
            "num_tokens_generated": len(completion.token_ids),
        })

    return {
        "model": model_name,
        "results": results,
        "elapsed_s": elapsed,
        "max_tokens": max_tokens,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(eval_output: dict):
    """Print accuracy breakdown by level and subject."""
    model = eval_output["model"]
    results = eval_output["results"]
    total = len(results)
    correct = sum(r["correct"] for r in results)

    print(f"\n{'='*60}")
    print(f"Results: {model}")
    print(f"Overall: {correct}/{total} = {correct/total*100:.1f}%")

    lens = [r["num_tokens_generated"] for r in results if "num_tokens_generated" in r]
    if lens:
        correct_lens = [
            r["num_tokens_generated"] for r in results
            if "num_tokens_generated" in r and r["correct"]
        ]
        incorrect_lens = [
            r["num_tokens_generated"] for r in results
            if "num_tokens_generated" in r and not r["correct"]
        ]
        print(f"Avg tokens generated: {sum(lens)/len(lens):.1f} (n={len(lens)})")
        if correct_lens:
            print(f"  correct:   {sum(correct_lens)/len(correct_lens):.1f} (n={len(correct_lens)})")
        if incorrect_lens:
            print(f"  incorrect: {sum(incorrect_lens)/len(incorrect_lens):.1f} (n={len(incorrect_lens)})")
    print(f"{'='*60}")

    # By level
    by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_level[r["level"]]["total"] += 1
        by_level[r["level"]]["correct"] += int(r["correct"])

    print(f"\n{'Level':<10} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 35)
    for level in sorted(by_level):
        d = by_level[level]
        acc = d["correct"] / d["total"] * 100 if d["total"] else 0
        print(f"Level {level:<4} {d['correct']:>8} {d['total']:>6} {acc:>7.1f}%")

    # By subject
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_subject[r["subject"]]["total"] += 1
        by_subject[r["subject"]]["correct"] += int(r["correct"])

    print(f"\n{'Subject':<25} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 50)
    for subj in sorted(by_subject):
        d = by_subject[subj]
        acc = d["correct"] / d["total"] * 100 if d["total"] else 0
        print(f"{subj:<25} {d['correct']:>8} {d['total']:>6} {acc:>7.1f}%")

    # Extraction failures
    no_answer = sum(1 for r in results if r["pred_answer"] is None)
    if no_answer:
        print(f"\nExtraction failures (no \\boxed{{}}): {no_answer}/{total}")


def save_results(eval_output: dict, output_dir: str):
    """Save full results and summary to disk."""
    os.makedirs(output_dir, exist_ok=True)
    model_slug = eval_output["model"].replace("/", "_")

    # Full per-problem results
    results_path = os.path.join(output_dir, f"{model_slug}_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_output["results"], f, indent=2)

    # Summary
    results = eval_output["results"]
    total = len(results)
    correct = sum(r["correct"] for r in results)

    by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_level[r["level"]]["total"] += 1
        by_level[r["level"]]["correct"] += int(r["correct"])

    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_subject[r["subject"]]["total"] += 1
        by_subject[r["subject"]]["correct"] += int(r["correct"])

    lens = [r["num_tokens_generated"] for r in results if "num_tokens_generated" in r]
    correct_lens = [
        r["num_tokens_generated"] for r in results
        if "num_tokens_generated" in r and r["correct"]
    ]
    incorrect_lens = [
        r["num_tokens_generated"] for r in results
        if "num_tokens_generated" in r and not r["correct"]
    ]

    summary = {
        "model": eval_output["model"],
        "method": eval_output.get("method", "greedy"),
        "dataset_size": total,
        "overall_accuracy": correct / total if total else 0,
        "elapsed_s": eval_output["elapsed_s"],
        "max_tokens": eval_output.get("max_tokens"),
        "avg_tokens_generated": sum(lens) / len(lens) if lens else None,
        "avg_tokens_correct": (
            sum(correct_lens) / len(correct_lens) if correct_lens else None
        ),
        "avg_tokens_incorrect": (
            sum(incorrect_lens) / len(incorrect_lens) if incorrect_lens else None
        ),
        "by_level": {
            str(k): {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for k, v in sorted(by_level.items())
        },
        "by_subject": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for k, v in sorted(by_subject.items())
        },
        "extraction_failures": sum(
            1 for r in results if r["pred_answer"] is None
        ),
    }

    summary_path = os.path.join(output_dir, f"{model_slug}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on math benchmarks")
    parser.add_argument(
        "--model", nargs="+", required=True,
        help="HuggingFace model name(s) or local checkpoint path(s)",
    )
    parser.add_argument(
        "--dataset", default="math500", choices=list(DATASET_REGISTRY_EVAL.keys()),
        help="Benchmark dataset to evaluate on",
    )
    parser.add_argument(
        "--levels", nargs="*", type=int, default=None,
        help="Filter to specific MATH difficulty levels (1-5)",
    )
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--prompt_mode", type=str, default="chat",
                        choices=["chat", "raw"],
                        help="Prompt format: 'chat' uses system+user chat template, "
                             "'raw' uses plain CoT string (matches Power-SMC reference for base models)")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Max context length for vLLM KV cache. Use 0 for model default.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate on a random subset of N samples (useful for quick tests)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset selection")
    parser.add_argument("--chat_template_model", type=str, default=None,
                        help="Load chat template from this model (e.g. the instruct variant) "
                             "for base models that lack one")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="For models with a toggleable thinking mode (e.g. Qwen3): "
                            "pass --enable-thinking or --no-enable-thinking to override "
                            "the template default. Leave unset to use the model default.") 
    

    args = parser.parse_args()

    # Load dataset
    loader = DATASET_REGISTRY_EVAL[args.dataset]
    problems = loader(levels=args.levels)
    print(f"Loaded {len(problems)} problems from {args.dataset}")
    if args.levels:
        print(f"  Filtered to levels: {args.levels}")

    # Subset selection
    if args.num_samples is not None and args.num_samples < len(problems):
        import random
        random.seed(args.seed)
        problems = random.sample(problems, args.num_samples)
        print(f"  Subsampled to {len(problems)} problems (seed={args.seed})")

    # Load chat template tokenizer if specified
    chat_template_tokenizer = None
    if args.chat_template_model:
        from transformers import AutoTokenizer
        chat_template_tokenizer = AutoTokenizer.from_pretrained(
            args.chat_template_model, trust_remote_code=True
        )
        print(f"Using chat template from: {args.chat_template_model}")

    # Evaluate model
    model_slug = args.model.replace("/", "_")
    output_dir = args.output_dir+'/'+args.dataset+'/'+model_slug
    eval_output = evaluate_model(
        model_name=args.model,
        problems=problems,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len or None,
        chat_template_tokenizer=chat_template_tokenizer,
        enable_thinking=args.enable_thinking,
        dtype=args.dtype,
        prompt_mode=args.prompt_mode,
    )
    print_report(eval_output)
    save_results(eval_output, output_dir)


if __name__ == "__main__":
    main()
