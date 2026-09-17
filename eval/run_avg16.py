"""Average accuracy over 16 samples per problem, reusing run_eval unchanged.

Results default to results/selected_ood, separate from the historical sweeps.
This reports only avg@16, with no best-of-k metrics.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from eval.deepmath_validation import fingerprint, read_json, write_json


def model_stamp(model: str) -> dict:
    path = Path(model)
    if not path.is_dir():
        return {"model": model}
    if (path / "adapter_config.json").exists():
        raise ValueError(
            "run_eval evaluates full model checkpoints; merge adapters first"
        )
    weights = sorted([*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")])
    if not (path / "config.json").is_file() or not weights:
        raise ValueError(f"No complete model checkpoint at {path}")
    files = sorted(
        set(weights + list(path.glob("*.json")) + list(path.glob("*.jinja")))
    )
    return {
        "model": str(path.resolve()),
        "files": [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files],
    }


def evaluation_config(
    model,
    problems,
    *,
    n,
    max_tokens,
    max_model_len,
    tensor_parallel_size,
    gpu_memory_utilization,
    chat_template_model=None,
):
    # run_eval owns decoding defaults. Record its exact source and dependency
    # versions so an upgrade cannot silently reuse an earlier cached score.
    root = Path(__file__).resolve().parents[1]
    return {
        "model": model_stamp(model),
        "problems": fingerprint(problems),
        "n": n,
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "chat_template_model": chat_template_model,
        "versions": {
            name: version(name) for name in ("vllm", "transformers", "math-verify")
        },
        "code": {
            name: fingerprint((root / name).read_text())
            for name in ("eval/run_eval.py", "eval/run_avg16.py", "utils/utils.py")
        },
        "sampling_policy": "unchanged run_eval.evaluate_model defaults",
    }


def accuracy_summary(output, n):
    rows = output["results"]
    if not rows or any(
        row["n_samples"] != n or len(row["samples"]) != n for row in rows
    ):
        raise ValueError(f"Expected exactly {n} samples for every question")
    from eval.run_eval import compute_pass_at_k

    accuracy = compute_pass_at_k(rows, [1])[1]
    return {
        "metric": f"avg@{n}",
        "accuracy": accuracy,
        "dataset_size": len(rows),
        "n_samples": n,
        "total_correct": sum(row["n_correct"] for row in rows),
        "total_samples": len(rows) * n,
        "extraction_failures": sum(
            s["pred_answer"] is None for row in rows for s in row["samples"]
        ),
        # run_eval does not retain finish_reason; this is a budget-hit diagnostic,
        # not an exact truncation label (EOS can coincide with the final token).
        "completion_budget_hit_fraction": sum(
            s["num_tokens_generated"] >= output["max_tokens"]
            for row in rows
            for s in row["samples"]
        )
        / (len(rows) * n),
        "sampling": output["sampling"],
        "elapsed_s": output["elapsed_s"],
    }


def cached_evaluation(model, problems, output_dir, config):
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if (
            summary.get("config") != config
            or not (output_dir / "results.json").is_file()
        ):
            raise ValueError(
                f"Incompatible or incomplete cache at {output_dir}; use a new output directory"
            )
        print(f"Reusing {summary_path}")
        return summary
    from eval.run_eval import evaluate_model

    tokenizer = None
    if config["chat_template_model"]:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["chat_template_model"], trust_remote_code=True
        )
    output = evaluate_model(
        model,
        problems,
        n_samples=config["n"],
        max_tokens=config["max_tokens"],
        max_model_len=config["max_model_len"],
        tensor_parallel_size=config["tensor_parallel_size"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        chat_template_tokenizer=tokenizer,
    )
    if len(output["results"]) != len(problems):
        raise ValueError("Evaluator returned an incomplete question set")
    if model_stamp(model) != config["model"]:
        raise ValueError(
            "Checkpoint changed during evaluation; refusing to cache the score"
        )
    summary = {
        "schema_version": 1,
        "config": config,
        **accuracy_summary(output, config["n"]),
    }
    write_json(output_dir / "results.json", output["results"])
    write_json(summary_path, summary)
    return summary


def add_generation_args(parser):
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--chat-template-model", default=None)


def config_from_args(args, model, problems, n):
    if args.max_tokens < 1 or args.tensor_parallel_size < 1:
        raise ValueError("Token budget and tensor parallel size must be positive")
    if args.max_model_len is not None and args.max_model_len <= args.max_tokens:
        raise ValueError(
            "max-model-len must leave room for the prompt above max-tokens"
        )
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    return evaluation_config(
        model,
        problems,
        n=n,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        chat_template_model=args.chat_template_model,
    )


def main():
    from eval.run_eval import ALGOS, arm_path
    from utils import DATASET_REGISTRY_EVAL, DATASET_REGISTRY_TRAIN

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model")
    group.add_argument(
        "--selection", help="selection.json written by eval.select_checkpoint"
    )
    parser.add_argument(
        "--dataset", default="aime24", choices=list(DATASET_REGISTRY_EVAL)
    )
    parser.add_argument("--algo", required=True, choices=ALGOS)
    parser.add_argument("--model-name", "--model_name", dest="model_name")
    parser.add_argument(
        "--train-dataset",
        "--train_dataset",
        dest="train_dataset",
        choices=list(DATASET_REGISTRY_TRAIN),
    )
    parser.add_argument("--variant")
    parser.add_argument("--run")
    parser.add_argument("--step")
    parser.add_argument("--output-dir", default="results/selected_ood")
    add_generation_args(parser)
    args = parser.parse_args()
    selection = None
    if args.selection:
        selection = read_json(args.selection)
        args.model = selection["best_checkpoint"]
        if model_stamp(args.model) != selection["best_model_stamp"]:
            parser.error(
                "Selected checkpoint no longer matches its validation evaluation"
            )
        meta = selection["training_audit"]["run_meta"]
        args.model_name = args.model_name or meta["model"].rstrip("/").split("/")[-1]
        args.train_dataset = args.train_dataset or meta["dataset"]
        args.run = args.run or Path(selection["run_dir"]).name
    parts, arm = arm_path(args, parser.error)
    problems = DATASET_REGISTRY_EVAL[args.dataset]()
    config = config_from_args(args, args.model, problems, n=16)
    config.update({"arm": arm, "dataset": args.dataset, "selection": selection})
    summary = cached_evaluation(
        args.model,
        problems,
        Path(args.output_dir) / args.dataset / Path(*parts),
        config,
    )
    print(f"{args.dataset}: avg@16 = {summary['accuracy']:.4f}")


if __name__ == "__main__":
    main()
