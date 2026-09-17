import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from eval.deepmath_validation import (
    audit_run,
    fingerprint,
    load_validation,
    question_key,
    sample_problems,
    training_questions,
    write_json,
)
from eval.run_avg16 import accuracy_summary, cached_evaluation, model_stamp
from eval.select_checkpoint import choose_best, discover_checkpoints


def source():
    return Dataset.from_list(
        [
            {"question": f"question {i}", "final_answer": str(i), "solution": "demo"}
            for i in range(10)
        ]
    )


def make_run(root, **overrides):
    meta = {
        "method": "grpo",
        "model": "base",
        "dataset": "deepmath",
        "max_samples": 3,
        "num_train_examples": 3,
        **overrides,
    }
    write_json(root / "run_meta.json", meta)
    return meta


class ValidationSplitTest(unittest.TestCase):
    def test_fixed_sampling_is_order_independent_and_excludes_duplicate_questions(self):
        rows = list(source()) + [{"question": " question 0 ", "final_answer": "other"}]
        excluded = {question_key("question 0"), question_key("question 1")}
        a = sample_problems(rows, excluded, 5, 42)
        b = sample_problems(list(reversed(rows)), excluded, 5, 42)
        self.assertEqual(a, b)
        self.assertTrue(excluded.isdisjoint(row["question_id"] for row in a))
        self.assertEqual(len({row["question_id"] for row in a}), 5)

    def test_excluding_full_training_pool_fails_instead_of_leaking(self):
        rows = source()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root, max_samples=None, num_train_examples=len(rows))
            keys, _ = training_questions(root, rows)
        with self.assertRaisesRegex(ValueError, "retrospectively held out"):
            sample_problems(rows, keys, 1, 42)

    def test_training_prefix_and_split_overlap_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root)
            keys, _ = training_questions(root, source())
            self.assertEqual(keys, {question_key(f"question {i}") for i in range(3)})
            split = {"problems": sample_problems(source(), set(), 10, 42)}
            with (
                patch("utils.load_train_dataset", return_value=source()),
                self.assertRaisesRegex(ValueError, "overlaps 3"),
            ):
                audit_run(split, root)

    def test_hint_cache_identity_and_prefix(self):
        rows = source().add_column("gen_model", ["generator"] * 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(
                root,
                method=None,
                pi_mode="hint",
                hint_cache="cache",
                hint_generator_model="generator",
            )
            with patch("datasets.load_from_disk", return_value=rows):
                keys, audit = training_questions(root)
            self.assertEqual(len(keys), 3)
            self.assertEqual(audit["hint_cache"], "cache")
            with (
                patch(
                    "datasets.load_from_disk",
                    return_value=rows.remove_columns("gen_model").add_column(
                        "gen_model", ["wrong"] * 10
                    ),
                ),
                self.assertRaisesRegex(ValueError, "provenance mismatch"),
            ):
                training_questions(root)

    def test_missing_cache_exception_is_explicit_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root, method=None, pi_mode="hint", hint_cache="missing")
            with patch(
                "datasets.load_from_disk", side_effect=FileNotFoundError("missing")
            ):
                with self.assertRaises(FileNotFoundError):
                    training_questions(root)
                keys, audit = training_questions(root, allow_missing_cache=True)
                self.assertEqual(keys, set())
                self.assertEqual(
                    audit["policy"], "missing_hint_cache_overlap_unverified"
                )
                split = {
                    "problems": sample_problems(source(), set(), 3, 42),
                    "exclusions": [audit],
                }
                self.assertEqual(audit_run(split, root), audit)
                split["exclusions"] = []
                with self.assertRaises(FileNotFoundError):
                    audit_run(split, root)

    def test_manifest_tampering_is_rejected(self):
        problems = sample_problems(source(), set(), 3, 42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            split = {
                "schema_version": 1,
                "dataset": "deepmath",
                "problems": problems,
                "fingerprint": fingerprint(problems),
            }
            write_json(path, split)
            self.assertEqual(load_validation(path), split)
            split["problems"][0]["answer"] = "tampered"
            write_json(path, split)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_validation(path)


class CheckpointSelectionTest(unittest.TestCase):
    def test_numeric_discovery_ignores_final_and_ties_choose_earlier_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root)
            for name in (
                "checkpoint-100",
                "checkpoint-20",
                "final",
                "checkpoint-final",
            ):
                child = root / name
                child.mkdir()
                (child / "config.json").write_text("{}")
                (child / "model.safetensors").write_bytes(b"fake weights")
            checkpoints = discover_checkpoints(root)
            self.assertEqual([step for step, _ in checkpoints], [20, 100])
            self.assertEqual(
                choose_best(
                    [
                        {"step": 100, "accuracy": 0.75},
                        {"step": 20, "accuracy": 0.75},
                    ]
                )["step"],
                20,
            )

    def test_cached_sweep_writes_selection_and_rejects_partial_scores(self):
        from eval.deepmath_validation import read_json
        from eval.select_checkpoint import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, out = root / "run", root / "scores"
            make_run(run)
            problems = sample_problems(
                source(), {question_key(f"question {i}") for i in range(3)}, 4, 42
            )
            split = {
                "schema_version": 1,
                "dataset": "deepmath",
                "problems": problems,
                "fingerprint": fingerprint(problems),
            }
            manifest = root / "split.json"
            write_json(manifest, split)
            _, audit = training_questions(run, source())

            def config(args, model, problems, n):
                return {"model": model_stamp(model), "n": n}

            for step, correct in [(20, 3), (100, 2)]:
                checkpoint = run / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "config.json").write_text("{}")
                (checkpoint / "model.safetensors").write_bytes(b"weights")
                expected = config(None, str(checkpoint), problems, 1)
                expected.update(
                    {
                        "validation_fingerprint": split["fingerprint"],
                        "training_audit": audit,
                    }
                )
                write_json(
                    out / checkpoint.name / "summary.json",
                    {
                        "config": expected,
                        "accuracy": correct / 4,
                        "total_correct": correct,
                        "dataset_size": 4,
                    },
                )
                write_json(out / checkpoint.name / "results.json", [])
            argv = [
                "select",
                "--run-dir",
                str(run),
                "--validation-file",
                str(manifest),
                "--output-dir",
                str(out),
                "--phase",
                "summarize",
            ]
            with (
                patch("sys.argv", argv),
                patch("utils.load_train_dataset", return_value=source()),
                patch("eval.select_checkpoint.config_from_args", side_effect=config),
                patch(
                    "eval.select_checkpoint.multiprocessing.get_context"
                ) as gpu_process,
            ):
                main()
                selection = read_json(out / "selection.json")
                self.assertEqual(selection["best_step"], 20)
                self.assertEqual(len(selection["scores"]), 2)
                gpu_process.assert_not_called()
                (out / "checkpoint-100" / "summary.json").unlink()
                with self.assertRaisesRegex(ValueError, "Missing validation score"):
                    main()

    def test_incomplete_checkpoint_is_rejected_before_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root)
            (root / "checkpoint-20").mkdir()
            with self.assertRaisesRegex(ValueError, "complete model checkpoint"):
                discover_checkpoints(root)

    def test_changed_checkpoint_invalidates_stamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            weights = root / "model.safetensors"
            weights.write_bytes(b"weights")
            stamp = model_stamp(str(root))
            weights.write_bytes(b"different weights")
            self.assertNotEqual(model_stamp(str(root)), stamp)

    def test_cache_reuse_and_mismatch_never_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {"config": {"n": 1}, "accuracy": 0.5}
            write_json(root / "summary.json", summary)
            write_json(root / "results.json", [])
            self.assertEqual(cached_evaluation("unused", [], root, {"n": 1}), summary)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                cached_evaluation("unused", [], root, {"n": 16})

    def test_avg16_is_mean_correctness_not_best_of_16(self):
        rows = []
        for correct in (4, 12):
            rows.append(
                {
                    "n_samples": 16,
                    "n_correct": correct,
                    "samples": [{"pred_answer": "1", "num_tokens_generated": 10}] * 16,
                }
            )
        output = {"results": rows, "max_tokens": 20, "sampling": {}, "elapsed_s": 1}
        summary = accuracy_summary(output, 16)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["metric"], "avg@16")
        self.assertNotIn("pass_at_k", summary)
        rows[0]["n_samples"] = 1
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            accuracy_summary(output, 16)


if __name__ == "__main__":
    unittest.main()
