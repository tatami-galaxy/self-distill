"""
Evaluate language models on MATH and other math benchmarks.

Usage:

    CUDA_VISIBLE_DEVICES=0 uv run python -m eval.run_eval \
    --model Qwen/Qwen2.5-Math-7B \
    --dataset math500 \
    --num_samples 10 \

"""

import argparse
import json
import os
import time
from vllm import LLM, SamplingParams
from .utils import (
    grade,
    DATASET_REGISTRY_EVAL,
)

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


def format_prompt(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def build_prompt(problem: str, tokenizer, template_tok=None) -> str:
    """Build a prompt string from a problem"""
    messages = format_prompt(problem)
    tok = template_tok or tokenizer
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    return tok.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------    

def evaluate_model(
    model_name: str,
    problems: list[dict],
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    chat_template_tokenizer=None,
    prompt_mode: str = "chat",
) -> dict:
    """Run evaluation and return results dict."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Problems:   {len(problems)}")
    print(f"{'='*60}")

    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
    )

    # Build prompts
    template_tok = chat_template_tokenizer or tokenizer
    prompts = []
    for p in problems:
        prompts.append(build_prompt(
            p["problem"], prompt_mode, tokenizer, template_tok)
        )

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
        pred_answer, correct = grade(response, prob["answer"])
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

    summary = {
        "model": eval_output["model"],
        "method": eval_output.get("method", "greedy"),
        "dataset_size": total,
        "overall_accuracy": correct / total if total else 0,
        "elapsed_s": eval_output["elapsed_s"],
        "max_tokens": eval_output.get("max_tokens"),
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
        "--model", type=str, required=True,
        help="HuggingFace model name or local checkpoint path",
    )
    parser.add_argument(
        "--dataset", default="aime24", choices=list(DATASET_REGISTRY_EVAL.keys()),
        help="Benchmark dataset to evaluate on",
    )
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_tokens", type=int, default=32000)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate on a random subset of N samples (useful for quick tests)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset selection")
    parser.add_argument("--chat_template_model", type=str, default=None,
                        help="Load chat template from this model (e.g. the instruct variant) "
                             "for base models that lack one")
    

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
        tensor_parallel_size=args.tensor_parallel_size,
        chat_template_tokenizer=chat_template_tokenizer,
        prompt_mode=args.prompt_mode,
    )
    print_report(eval_output)
    save_results(eval_output, output_dir)


if __name__ == "__main__":
    main()
