"""Checkpoint sweeps accept arbitrary training variants without exposing PI at eval."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eval import run_eval_checkpoints


class SolutionCheckpointTest(unittest.TestCase):
    def test_solution_variant_is_forwarded_in_step_order_without_running_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("checkpoint-20", "checkpoint-3", "final"):
                (root / name).mkdir()
            output = io.StringIO()
            with (
                mock.patch.object(run_eval_checkpoints.subprocess, "run") as run,
                redirect_stdout(output),
            ):
                status = run_eval_checkpoints.main(
                    [
                        "--model-dir",
                        directory,
                        "--algo",
                        "sdft",
                        "--model_name",
                        "Qwen3-1.7B",
                        "--train_dataset",
                        "deepmath",
                        "--variant",
                        "solution",
                        "--dataset",
                        "aime24",
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 0)
            run.assert_not_called()
            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("--step checkpoint-3", lines[0])
            self.assertIn("--step checkpoint-20", lines[1])
            for line in lines:
                self.assertIn("--variant solution", line)
                self.assertNotIn("--pi-mode", line)

    def test_downstream_evaluator_routes_solution_variant(self):
        from eval.run_eval import arm_path

        args = SimpleNamespace(
            model="/outputs/deepmath_solution/checkpoint-20",
            model_name="Qwen3-1.7B",
            algo="sdft",
            train_dataset="deepmath",
            variant="solution",
            run=None,
            step="checkpoint-20",
        )

        def error(message):
            raise AssertionError(message)

        parts, meta = arm_path(args, error)
        self.assertEqual(
            parts, ["deepmath", "Qwen3-1.7B", "sdft", "solution", "checkpoint-20"]
        )
        self.assertEqual(meta["variant"], "solution")


if __name__ == "__main__":
    unittest.main()
