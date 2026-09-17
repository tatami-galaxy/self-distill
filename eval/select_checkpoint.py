"""Select one checkpoint within one training run using fixed DeepMath accuracy.

Each checkpoint runs in a fresh process, releasing its vLLM engine before the
next one starts. All checkpoints see the same question order, one sample per
question, and the unchanged run_eval decoding defaults. Exact ties favor the
earliest step. Only numeric checkpoint-N directories are considered; final is
excluded because it commonly duplicates the last numbered checkpoint.
"""

from __future__ import annotations

import argparse
import multiprocessing
import re
from pathlib import Path

from eval.deepmath_validation import (
    audit_run,
    fingerprint,
    load_validation,
    read_json,
    write_json,
)
from eval.run_avg16 import (
    add_generation_args,
    cached_evaluation,
    config_from_args,
    model_stamp,
)


def discover_checkpoints(run_dir):
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "run_meta.json").is_file():
        raise ValueError(
            "Pass one exact run directory containing run_meta.json and checkpoint-N directories"
        )
    checkpoints = []
    for path in run_dir.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            model_stamp(
                str(path)
            )  # fail before spending GPU time on an incomplete sweep
            checkpoints.append((int(match[1]), str(path)))
    if not checkpoints:
        raise ValueError(f"No numeric checkpoint directories under {run_dir}")
    return sorted(checkpoints)


def choose_best(scores):
    if not scores:
        raise ValueError("No checkpoint scores")
    return min(scores, key=lambda score: (-score["accuracy"], score["step"]))


def _evaluate_worker(model, problems, output_dir, config):
    cached_evaluation(model, problems, output_dir, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--validation-file", default="data/validation/deepmath_128.json"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--allow-training-overlap",
        action="store_true",
        help="Allow the fixed selection set to overlap eligible training data; record overlap in results",
    )
    parser.add_argument(
        "--phase",
        choices=["sweep", "summarize"],
        default="sweep",
        help="summarize selects from existing scores without loading a model",
    )
    add_generation_args(parser)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoints = discover_checkpoints(run_dir)
    split = load_validation(args.validation_file)
    audit = audit_run(
        split, run_dir, allow_training_overlap=args.allow_training_overlap
    )
    problems = split["problems"]
    output_dir = Path(
        args.output_dir
        or (
            Path("results/validation")
            / run_dir.parent.name
            / f"{run_dir.name}-{fingerprint(str(run_dir))[:8]}"
        )
    )
    # Freeze the candidate list for this invocation; later checkpoints can be
    # added on a subsequent sweep without changing any cached earlier score.
    planned = []
    for step, model in checkpoints:
        config = config_from_args(args, model, problems, n=1)
        config.update(
            {"validation_fingerprint": split["fingerprint"], "training_audit": audit}
        )
        directory = output_dir / f"checkpoint-{step}"
        summary_path = directory / "summary.json"
        if summary_path.exists():
            old = read_json(summary_path)
            if (
                old.get("config") != config
                or not (directory / "results.json").is_file()
            ):
                raise ValueError(
                    f"Incompatible or incomplete score at {directory}; use a new output directory"
                )
        elif args.phase == "summarize":
            raise ValueError(f"Missing validation score: {summary_path}")
        planned.append((step, model, directory, config))

    scores = []
    for step, model, directory, config in planned:
        if not (directory / "summary.json").exists():
            process = multiprocessing.get_context("spawn").Process(
                target=_evaluate_worker,
                args=(model, problems, directory, config),
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Evaluation failed for {model} (exit {process.exitcode}); rerun to resume"
                )
        summary = read_json(directory / "summary.json")
        if summary["config"] != config or model_stamp(model) != config["model"]:
            raise ValueError(f"Checkpoint or score changed during sweep: {model}")
        scores.append(
            {
                "step": step,
                "checkpoint": model,
                "accuracy": summary["accuracy"],
                "correct": summary["total_correct"],
                "questions": summary["dataset_size"],
            }
        )
        print(
            f"checkpoint-{step}: {summary['total_correct']}/{len(problems)} = {summary['accuracy']:.4f}"
        )
    best = choose_best(scores)
    write_json(
        output_dir / "selection.json",
        {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "validation_file": str(Path(args.validation_file).resolve()),
            "validation_fingerprint": split["fingerprint"],
            "training_audit": audit,
            "metric": "avg@1",
            "tie_break": "earliest_step",
            "scores": scores,
            "best_checkpoint": best["checkpoint"],
            "best_step": best["step"],
            "best_accuracy": best["accuracy"],
            "best_model_stamp": model_stamp(best["checkpoint"]),
            "evaluation_config": next(
                config for step, _, _, config in planned if step == best["step"]
            ),
        },
    )
    print(f"Selected {best['checkpoint']} -> {output_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
