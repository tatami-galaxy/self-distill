"""
Evaluate language models on math benchmarks.

Results are filed by ARM (see arm_path below), so every checkpoint of every training
method lands in a predictable place and a whole sweep can be read by globbing.

Usage:

    # the untrained model -- results/aime24/base/Qwen3-4B/
    CUDA_VISIBLE_DEVICES=0 uv run python -m eval.run_eval \
    --model Qwen/Qwen3-4B --dataset aime24 --algo base

    # a trained checkpoint -- results/aime24/deepmath/Qwen3-4B/sdft/hint/checkpoint-120/
    CUDA_VISIBLE_DEVICES=0 uv run python -m eval.run_eval \
    --model /mnt/data/ujan/self-distill/outputs/sdft/Qwen3-4B/deepmath_hint/checkpoint-120 \
    --dataset aime24 --algo sdft --model_name Qwen3-4B --train_dataset deepmath --variant hint \
    --run run-2 --step checkpoint-120

"""

import argparse
import json
import os
import re
import time
from vllm import LLM, SamplingParams
from utils import (
    grade,
    DATASET_REGISTRY_EVAL,
    DATASET_REGISTRY_TRAIN,
    format_prompt_math,
)


# ---------------------------------------------------------------------------
# Result layout
# ---------------------------------------------------------------------------
#
#   <output_dir>/<eval dataset>/base/<model>/
#   <output_dir>/<eval dataset>/<train dataset>/<model>/<algo>[/<variant>][/<run>]/<step>/
#
# each holding results.json + summary.json. The untrained model sits at `base/`, a sibling
# of the train-dataset dirs, because it belongs to no training run -- one evaluation serves
# every train dataset.
#
# Every component is passed EXPLICITLY rather than parsed out of --model. A checkpoint path
# encodes its arm only by convention, and the convention has already been broken once: the
# two PPO runs live in output dirs that were renamed to `Qwen3-1.7B-run-{1,2}` after the
# fact, so their checkpoints' own paths (and the `model` field of the summaries written from
# them) still say plain `Qwen3-1.7B`. Inferring from a path would have silently merged the
# two runs into one folder -- a misfiling that looks entirely plausible afterwards. The cost
# is a longer command; the benefit is that an hours-long eval cannot land in the wrong arm.

# `ppo_val` is its own algorithm rather than a --variant of `ppo`: the two trainers are
# separate files writing separate `method` stamps, and filing them together would leave the
# verifier arm and its own control distinguished only by a --run label -- the same merge the
# VARIANT_REQUIRED note below exists to prevent. Adding a name here does not move any results
# already on disk.
#
# `sft` takes no --variant today because there is exactly one corpus (the dataset's reference
# traces). If train_sft.py ever grows a second one -- teacher-sampled or rejection-sampled
# completions -- the corpus becomes the variable under study and `sft` should join
# VARIANT_REQUIRED below, exactly as `sdft` did for its privileged context.
ALGOS = (
    "base", "grpo", "ppo", "ppo_pi", "ppo_val", "gold", "sdft", "sft"
)

# Bumped whenever summary.json gains or changes a field. Summaries written before this
# existed have no `schema_version` at all: v1 summaries carry `arm`, and the earliest ones
# (e.g. results/aime24/base/) do not even carry that. A reader comparing two runs needs to
# distinguish "this field is absent because it was never recorded" from "absent because it
# was null", and only a version stamp answers that.
SUMMARY_SCHEMA_VERSION = 2

# Algorithms the algorithm name alone does not identify: for SDFT the privileged context and
# for GOLD the teacher ARE the independent variable under study, so filing two of them under
# a bare `sdft/` or `gold/` would merge runs that the experiment exists to tell apart.
VARIANT_REQUIRED = {
    "ppo_pi": "the critic's privileged context, e.g. --variant answer or --variant none",
    "sdft": "the privileged context, e.g. --variant hint or --variant self_rollout",
    "gold": "the teacher, e.g. --variant Qwen3-30B-A3B-Thinking-2507",
}

# What a checkpoint directory is called: `checkpoint-<N>`, TRL's end-of-training `final/`, or
# a prefixed variant like `base-token-checkpoint-20`.
_STEP_RE = re.compile(r"(?:^final$|checkpoint-\d+$)")


def arm_path(args, error) -> tuple[list[str], dict]:
    """Resolve this evaluation's arm to (path components, provenance dict).

    Components are appended below `<output_dir>/<eval dataset>`. `error` is the
    ArgumentParser's error hook, so a bad flag combination fails immediately rather than
    after the model has loaded and generated.
    """
    leaf = os.path.basename(args.model.rstrip("/"))
    is_checkpoint = bool(_STEP_RE.search(leaf))

    model = args.model_name
    if model is None:
        if is_checkpoint:
            error(f"--model's last path component ({leaf!r}) names a checkpoint, not the model "
                  "being evaluated, so the results folder cannot be named from it. Pass "
                  "--model_name (e.g. --model_name Qwen3-4B).")
        model = leaf  # a hub id like Qwen/Qwen3-4B -- the basename IS the model

    if args.algo == "base":
        misplaced = [
            flag for flag, val in (("--train_dataset", args.train_dataset),
                                   ("--variant", args.variant), ("--run", args.run),
                                   ("--step", args.step)) if val is not None
        ]
        if misplaced:
            error(f"{', '.join(misplaced)} do not apply to --algo base: the untrained model "
                  "belongs to no training run, and its results are shared across train datasets "
                  "at base/<model>/. Accepting them would silently file the same numbers twice.")
        return ["base", model], {"algo": "base", "model": model}

    if args.train_dataset is None:
        error(f"--train_dataset is required for --algo {args.algo}: it names the dataset this "
              "checkpoint was trained on, which is the top level of the results tree.")
    if args.algo in VARIANT_REQUIRED and args.variant is None:
        error(f"--variant is required for --algo {args.algo}: it names {VARIANT_REQUIRED[args.algo]}. "
              "Without it, runs differing in exactly the variable under study share one folder.")

    step = args.step or (leaf if is_checkpoint else None)
    if step is None:
        error(f"--step is required: --model's last path component ({leaf!r}) is not a "
              "'checkpoint-<N>' or 'final' dir, so the step cannot be taken from it.")

    # `run` only earns a directory level when there is more than one run to separate; pass it
    # for every run of a repeated arm, so the sibling folders are symmetric.
    run = args.run
    if run is not None and run.isdigit():
        run = f"run-{run}"  # --run 2 and --run run-2 name the same folder

    parts = [args.train_dataset, model, args.algo]
    parts += [p for p in (args.variant, run) if p]
    parts.append(step)
    return parts, {
        "algo": args.algo, "model": model, "train_dataset": args.train_dataset,
        "variant": args.variant, "run": run, "step": step,
    }


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
    gpu_memory_utilization: float = 0.9,
) -> dict:
    """Run evaluation and return results dict.

    Samples ``n_samples`` completions per problem (one vLLM request with
    ``n=n_samples``, which shares the prompt prefix across rollouts) and grades
    each, so the caller can compute pass@k for any ``k <= n_samples``.

    Sampling is left at vLLM's ``SamplingParams`` defaults, which already are the
    distribution the trainers roll out at (temperature 1.0, top_p 1.0, top_k 0 =
    all tokens; GRPOConfig uses the same three). Note this is NOT the checkpoint's
    generation_config -- vLLM consults that only when ``sampling_params is None``
    (``vllm/entrypoints/llm.py``), and we always build one, so Qwen3's shipped
    0.6/0.95/20 never applies here.

    The resolved values are read back off the constructed object and returned under
    ``sampling``, so the summary records what actually ran rather than a literal
    that could drift from vLLM's defaults.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Problems:   {len(problems)}")
    print(f"Samples/problem: {n_samples}")
    print(f"{'='*60}")

    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=n_samples,
        max_tokens=max_tokens,
    )
    # Read back rather than hardcode: these are vLLM's defaults, so the record stays
    # truthful if a future vLLM changes one (top_k's disabled sentinel already moved
    # from -1 to 0 between 0.6.x and 0.25.x).
    sampling = {
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "repetition_penalty": sampling_params.repetition_penalty,
        "seed": sampling_params.seed,
    }
    print(f"Sampling (vLLM defaults): {sampling}")

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
        "sampling": sampling,
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


def save_results(
    eval_output: dict, output_dir: str, ks: list[int], arm: dict, eval_config: dict
):
    """Save full results and summary to disk.

    Fixed filenames: the arm is carried by the directory path (see arm_path), so a summary
    is found the same way in every folder and a sweep is one glob. `arm` is also embedded in
    the summary, so a file that gets moved still says which run produced it.

    `eval_config` extends that self-description to HOW the numbers were produced -- the
    sampling distribution, the completion budget, the prompt template, the subset, and the
    library versions. pass@k is a property of the sampling distribution, so a summary that
    omits it cannot be compared against another summary except on trust. Older summaries
    predate this block (and some predate `arm`), hence `schema_version`: a reader can tell
    "recorded and equal" from "never recorded" instead of guessing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Full per-problem results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(eval_output["results"], f, indent=2)

    # Summary
    results = eval_output["results"]
    total = len(results)
    pass_at_ks = compute_pass_at_k(results, ks)
    total_samples = sum(r["n_samples"] for r in results)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "model": eval_output["model"],
        "arm": arm,
        "eval_config": eval_config,
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

    summary_path = os.path.join(output_dir, "summary.json")
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
    # Arm identification -- where the results are filed (see arm_path).
    parser.add_argument(
        "--algo", required=True, choices=ALGOS,
        help="Training method that produced --model, or 'base' for the untrained model. "
             "Required: it is the folder the results are compared in, and nothing in the "
             "checkpoint reliably reports it.",
    )
    parser.add_argument(
        "--model_name", default=None,
        help="Model folder name, e.g. Qwen3-4B. Defaults to --model's basename, which is "
             "right for a hub id but not for a checkpoint path (there the basename is the "
             "step), so it is required whenever --model points at a checkpoint dir.",
    )
    parser.add_argument(
        "--train_dataset", default=None, choices=list(DATASET_REGISTRY_TRAIN.keys()),
        help="Dataset --model was TRAINED on (top level of the results tree). Required for "
             "every --algo except base; see utils.DATASET_REGISTRY_TRAIN.",
    )
    parser.add_argument(
        "--variant", default=None,
        help="What distinguishes this arm within its algorithm: the privileged context for "
             "sdft (full/answer/hint/rollout/self_rollout) or the teacher slug for gold. "
             "Required where the algorithm has multiple experiment arms; optional elsewhere.",
    )
    parser.add_argument(
        "--run", default=None,
        help="Repeat label when an arm was trained more than once, e.g. --run 2 or "
             "--run run-2 (both file under run-2/). Omit for a single run.",
    )
    parser.add_argument(
        "--step", default=None,
        help="Checkpoint folder name, e.g. checkpoint-120. Defaults to --model's basename "
             "when that is a 'checkpoint-<N>' or 'final' dir.",
    )
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--n", type=int, default=16,
                        help="Number of samples to draw per problem (n>=max(k) for pass@k)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 8, 16],
                        help="pass@k value(s) to report, e.g. --k 1 8 16")
    parser.add_argument("--max_tokens", type=int, default=32000)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of the GPU vLLM may reserve. This is measured against "
                             "TOTAL device memory, not free memory, so the 0.9 default requires "
                             "an essentially idle GPU -- lower it to share one with another job. "
                             "Matches the flag on eval/passk_pi.py and utils/gen_hints.py.")
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

    # Resolve the destination BEFORE loading anything: an eval is hours of generation, and a
    # flag combination that cannot be filed should fail in the first second, not the last.
    parts, arm = arm_path(args, parser.error)
    output_dir = os.path.join(args.output_dir, args.dataset, *parts)
    print(f"Results -> {output_dir}")

    if args.n > 1:
        # pass@k needs stochastic sampling, and it gets it: vLLM's SamplingParams defaults
        # are temperature 1.0 / top_p 1.0 / top_k 0 (= all tokens), the same distribution
        # the trainers roll out at. The checkpoint's own generation_config is NOT consulted
        # -- vLLM reads that only when sampling_params is None, and evaluate_model always
        # constructs one. The resolved values are printed there and recorded in summary.json.
        print("Note: sampling uses vLLM's SamplingParams defaults (temperature 1.0, "
              "top_p 1.0, top_k 0 = all tokens), which match the training rollout config; "
              "the checkpoint's generation_config does not apply.")

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
    eval_output = evaluate_model(
        model_name=args.model,
        problems=problems,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        chat_template_tokenizer=chat_template_tokenizer,
        n_samples=args.n,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print_report(eval_output, args.k)

    # Everything that shaped the numbers but is not derivable from them. `sampling` comes
    # back resolved from the SamplingParams object rather than restated here. `num_samples`
    # and `seed` together identify the subset: dataset_size alone cannot distinguish a full
    # 30-problem AIME24 from a 30-problem draw out of a larger set.
    import transformers
    import vllm

    eval_config = {
        "eval_dataset": args.dataset,
        "n": args.n,
        "k": args.k,
        "max_tokens": args.max_tokens,
        "sampling": eval_output["sampling"],
        "num_samples": args.num_samples,
        "seed": args.seed,
        "chat_template_model": args.chat_template_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "vllm_version": vllm.__version__,
        "transformers_version": transformers.__version__,
    }
    save_results(eval_output, output_dir, args.k, arm, eval_config)


if __name__ == "__main__":
    main()
