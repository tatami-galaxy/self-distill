"""
Evaluate language models on math benchmarks.

Usage:

    CUDA_VISIBLE_DEVICES=0 uv run python -m eval.run_eval \
    --model Qwen/Qwen3-4B \
    --dataset aime24 \

"""

import argparse
import json
import os
import time
from vllm import LLM, SamplingParams
from utils import (
    grade,
    DATASET_REGISTRY_EVAL,
    format_prompt_math,
)


# ---------------------------------------------------------------------------
# pass@k estimator
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    Given ``n`` samples of which ``c`` are correct, returns the probability that
    at least one of ``k`` samples drawn without replacement is correct:
    ``1 - C(n-c, k) / C(n, k)``. Averaged over problems this is a lower-variance
    estimate than sampling exactly ``k`` and checking for any hit, and lets a
    single run of ``n`` samples yield pass@k for every ``k <= n``.
    """
    if k > n:
        raise ValueError(f"pass@{k} needs n>={k} samples, got n={n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def build_prompt(problem: str, tokenizer, template_tok=None) -> str:
    """Build a prompt string from a problem"""
    messages = format_prompt_math(problem)
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
    n_samples: int = 1,
) -> dict:
    """Run evaluation and return results dict.

    Samples ``n_samples`` completions per problem (one vLLM request with
    ``n=n_samples``, which shares the prompt prefix across rollouts) and grades
    each, so the caller can compute pass@k for any ``k <= n_samples``. Sampling
    (temperature, top_p, ...) is left to vLLM's model defaults.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Problems:   {len(problems)}")
    print(f"Samples/problem: {n_samples}")
    print(f"{'='*60}")

    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=n_samples,
        max_tokens=max_tokens,
    )

    # Build prompts
    template_tok = chat_template_tokenizer or tokenizer
    prompts = []
    for p in problems:
        prompts.append(build_prompt(
            p["problem"], tokenizer, template_tok)
        )

    # Generate
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    n_gen = len(problems) * n_samples
    print(f"Generation took {elapsed:.1f}s ({n_gen/elapsed:.1f} samples/s)")

    # Score: grade each of the n_samples completions per problem.
    results = []
    for prob, output in zip(problems, outputs):
        samples = []
        for completion in output.outputs:
            response = completion.text
            pred_answer, correct = grade(response, prob["answer"])
            samples.append({
                "response": response,
                "pred_answer": pred_answer,
                "correct": correct,
                "num_tokens_generated": len(completion.token_ids),
            })
        results.append({
            **prob,
            "samples": samples,
            "n_samples": len(samples),
            "n_correct": sum(s["correct"] for s in samples),
        })

    return {
        "model": model_name,
        "results": results,
        "elapsed_s": elapsed,
        "max_tokens": max_tokens,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_pass_at_k(results: list[dict], ks: list[int]) -> dict[int, float]:
    """Average the unbiased pass@k over all problems, for each k in ``ks``."""
    total = len(results)
    return {
        k: sum(pass_at_k(r["n_samples"], r["n_correct"], k) for r in results) / total
        for k in ks
    }


def print_report(eval_output: dict, ks: list[int]):
    model = eval_output["model"]
    results = eval_output["results"]
    total = len(results)
    n_samples = eval_output["n_samples"]

    print(f"\n{'='*60}")
    print(f"Results: {model}")
    print(f"Problems: {total}  |  Samples/problem: {n_samples}")
    for k, acc in compute_pass_at_k(results, ks).items():
        print(f"pass@{k}: {acc*100:.1f}%")

    # Extraction failures (across all samples)
    total_samples = sum(r["n_samples"] for r in results)
    no_answer = sum(
        1 for r in results for s in r["samples"] if s["pred_answer"] is None
    )
    if no_answer:
        print(f"\nExtraction failures (no \\boxed{{}}): {no_answer}/{total_samples} samples")


def save_results(eval_output: dict, output_dir: str, ks: list[int]):
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
    pass_at_ks = compute_pass_at_k(results, ks)
    total_samples = sum(r["n_samples"] for r in results)

    summary = {
        "model": eval_output["model"],
        "dataset_size": total,
        "n_samples": eval_output["n_samples"],
        "pass_at_k": {f"pass@{k}": acc for k, acc in pass_at_ks.items()},
        "elapsed_s": eval_output["elapsed_s"],
        "max_tokens": eval_output.get("max_tokens"),
        "extraction_failures": sum(
            1 for r in results for s in r["samples"] if s["pred_answer"] is None
        ),
        "total_samples": total_samples,
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
    parser.add_argument("--n", type=int, default=16,
                        help="Number of samples to draw per problem (n>=max(k) for pass@k)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 8, 16],
                        help="pass@k value(s) to report, e.g. --k 1 8 16")
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

    if max(args.k) > args.n:
        parser.error(f"--k values must be <= --n ({args.n}); got --k {args.k}")
    if args.n > 1:
        print("Note: pass@k needs stochastic sampling; relying on vLLM's model "
              "default sampling (ensure temperature > 0 in the model's generation config)")

    # Load dataset
    loader = DATASET_REGISTRY_EVAL[args.dataset]
    problems = loader()
    print(f"Loaded {len(problems)} problems from {args.dataset}")

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
        n_samples=args.n,
    )
    print_report(eval_output, args.k)
    save_results(eval_output, output_dir, args.k)


if __name__ == "__main__":
    main()
