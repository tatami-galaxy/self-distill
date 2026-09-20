"""Run eval.run_eval sequentially for every checkpoint in a training directory.

Only immediate directories named checkpoint-<number> are evaluated, in numeric
step order. Each evaluation runs in a fresh process and receives the directory
name as --step. A failed evaluation stops the sweep.

Additional options are forwarded to eval.run_eval, which keeps its usual defaults
and result layout. --variant and --run have the same meaning as in that module.
Use --dry-run to print commands without loading models or writing results.

    CUDA_VISIBLE_DEVICES=0 uv run python -m eval.run_eval_checkpoints \
        --model-dir /mnt/data/ujan/self-distill/outputs/sdft/Qwen3-4B/deepmath_hint \
        --algo sdft --model_name Qwen3-4B --train_dataset deepmath \
        --variant hint --run run-2 --dataset aime24 --n 16 --k 1 8 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys


def discover_checkpoints(model_dir: Path) -> list[Path]:
    """Find checkpoint directories without including final/ or nested runs."""
    return sorted(
        (
            path for path in model_dir.iterdir()
            if re.fullmatch(r"checkpoint-[0-9]+", path.name) and path.is_dir()
        ),
        key=lambda path: (int(path.name.removeprefix("checkpoint-")), path.name),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--model-dir", "--model_dir", required=True, type=Path)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--run")
    parser.add_argument("--dry-run", action="store_true")
    args, eval_args = parser.parse_known_args(argv)

    if args.algo == "base":
        parser.error("--algo base is not a checkpoint evaluation; use eval.run_eval directly")
    if any(arg.split("=", 1)[0] in {"--model", "--step"} for arg in eval_args):
        parser.error("--model and --step are set automatically for each checkpoint")
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        parser.error(f"Not a model directory: {model_dir}")
    checkpoints = discover_checkpoints(model_dir)
    if not checkpoints:
        parser.error(f"No checkpoint-<number> directories found in {model_dir}")

    shared_args = [
        "--algo", args.algo,
        "--model_name", args.model_name,
        "--train_dataset", args.train_dataset,
    ]
    for flag, value in (("--variant", args.variant), ("--run", args.run)):
        if value is not None:
            shared_args.extend([flag, value])

    for index, checkpoint in enumerate(checkpoints, start=1):
        command = [
            sys.executable, "-m", "eval.run_eval",
            *shared_args, *eval_args,
            "--model", str(checkpoint), "--step", checkpoint.name,
        ]
        print(f"[{index}/{len(checkpoints)}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            result = subprocess.run(command, check=False)
            if result.returncode:
                print(f"Evaluation failed for {checkpoint.name}; stopping.", file=sys.stderr)
                return result.returncode if result.returncode > 0 else 128 - result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
