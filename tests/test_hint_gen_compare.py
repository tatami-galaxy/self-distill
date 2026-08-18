import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from eval import hint_gen_compare


class IdentityAndDefinitionTest(unittest.TestCase):
    def test_question_and_hint_ids_are_stable_and_sample_specific(self):
        question_id = hint_gen_compare.stable_question_id(7, "question", "answer")
        self.assertEqual(
            question_id,
            hint_gen_compare.stable_question_id(7, "question", "answer"),
        )
        self.assertNotEqual(
            question_id,
            hint_gen_compare.stable_question_id(8, "question", "answer"),
        )
        self.assertNotEqual(
            hint_gen_compare.stable_hint_id("checkpoint-10", question_id, 0),
            hint_gen_compare.stable_hint_id("checkpoint-10", question_id, 1),
        )

    def test_transfer_is_student_minus_hinted_teacher_without_clamping(self):
        student = torch.tensor([-2.0, -1.0, -3.0])
        teacher = torch.tensor([-1.0, -2.0, -2.0])
        mean, total, count = hint_gen_compare.raw_sampled_transfer(student, teacher)
        self.assertAlmostEqual(mean, -1.0 / 3.0)
        self.assertAlmostEqual(total, -1.0)
        self.assertEqual(count, 3)

    def test_transfer_rejects_misaligned_token_scores(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            hint_gen_compare.raw_sampled_transfer(torch.zeros(2), torch.zeros(3))


class CheckpointSpecTest(unittest.TestCase):
    def test_default_label_is_checkpoint_directory_name(self):
        self.assertEqual(
            hint_gen_compare.parse_checkpoint_spec("/runs/checkpoint-40"),
            ("checkpoint-40", "/runs/checkpoint-40"),
        )

    def test_explicit_label_is_supported(self):
        self.assertEqual(
            hint_gen_compare.parse_checkpoint_spec("early=/runs/checkpoint-10"),
            ("early", "/runs/checkpoint-10"),
        )

    def test_reserved_and_path_like_labels_are_rejected(self):
        for value in ("fresh_base=/run/checkpoint-1", "bad/label=/run/checkpoint-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                hint_gen_compare.parse_checkpoint_spec(value)

    def test_run_directory_discovers_numeric_checkpoints_in_step_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint-20").mkdir()
            (root / "checkpoint-3").mkdir()
            (root / "checkpoint-final").mkdir()
            (root / "checkpoint-4").write_text("not a directory")

            checkpoints = hint_gen_compare.discover_checkpoints(root)
            self.assertEqual(list(checkpoints), [3, 20])
            self.assertEqual(
                list(hint_gen_compare.discover_checkpoints(root, steps=[20])), [20]
            )
            with self.assertRaisesRegex(ValueError, "available steps"):
                hint_gen_compare.discover_checkpoints(root, steps=[10])

    def test_run_metadata_infers_model_and_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_meta.json").write_text(
                json.dumps(
                    {
                        "method": "hint_gen_grpo",
                        "model": "Qwen/Qwen3-4B",
                        "dataset": "deepmath",
                    }
                )
            )
            args = SimpleNamespace(run_dir=str(root), model=None, dataset=None)
            hint_gen_compare.resolve_run_configuration(args)
            self.assertEqual(args.model, "Qwen/Qwen3-4B")
            self.assertEqual(args.dataset, "deepmath")

            conflicting = SimpleNamespace(
                run_dir=str(root), model="different", dataset=None
            )
            with self.assertRaisesRegex(ValueError, "conflicts"):
                hint_gen_compare.resolve_run_configuration(conflicting)

    def test_generator_variants_include_every_discovered_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_meta.json").write_text(
                json.dumps(
                    {
                        "method": "hint_gen_grpo",
                        "model": "base",
                        "dataset": "deepmath",
                    }
                )
            )
            for step in (20, 3):
                (root / f"checkpoint-{step}").mkdir()
            args = SimpleNamespace(
                model="base",
                dataset="deepmath",
                run_dir=str(root),
                steps=None,
                checkpoint=[],
                resolved_checkpoints=None,
            )
            variants = hint_gen_compare.generator_variants(args)
            self.assertEqual(
                [label for label, _ in variants],
                ["fresh_base", "checkpoint-3", "checkpoint-20"],
            )


class SummaryTest(unittest.TestCase):
    @staticmethod
    def hint(hint_id, question_id, length, invalid=""):
        return {
            "hint_id": hint_id,
            "question_id": question_id,
            "num_hint_tokens": length,
            "truncated": False,
            "invalid_reason": invalid,
            "validity_observable": True,
        }

    def test_summary_averages_hints_within_question_before_questions(self):
        # q1 deliberately has two hints and q2 has one. A flat hint average would
        # produce pass@1=2/3; the intended question-balanced result is 3/4.
        hints = [
            self.hint("h1", "q1", 10, invalid="answer_leak"),
            self.hint("h2", "q1", 20),
            self.hint("h3", "q2", 30),
        ]
        sufficiency = {
            "h1": {"n_samples": 2, "n_correct": 0, "n_truncated": 0},
            "h2": {"n_samples": 2, "n_correct": 2, "n_truncated": 0},
            "h3": {"n_samples": 2, "n_correct": 2, "n_truncated": 1},
        }
        no_hint = {
            "q1": {"n_samples": 2, "n_correct": 0},
            "q2": {"n_samples": 2, "n_correct": 1},
        }
        transfer = {
            "h1": [{"raw_transfer": 0.2}, {"raw_transfer": 0.4}],
            "h2": [{"raw_transfer": 0.1}, {"raw_transfer": 0.1}],
            "h3": [{"raw_transfer": 0.6}, {"raw_transfer": 0.6}],
        }

        summary = hint_gen_compare.summarize_generator_subset(
            hints,
            sufficiency,
            no_hint,
            transfer,
            ks=[1, 2],
            bootstrap_samples=50,
            seed=7,
        )

        self.assertAlmostEqual(summary["sufficiency"]["pass@1"]["value"], 0.75)
        self.assertAlmostEqual(
            summary["sufficiency"]["pass@1"]["delta_over_no_hint"], 0.5
        )
        # q1: mean(mean(.2,.4), mean(.1,.1))=.2; q2=.6; across q=.4.
        self.assertAlmostEqual(summary["transfer"]["mean_raw_nats_per_token"], 0.4)
        self.assertEqual(summary["invalid_fraction"], 1 / 3)
        self.assertEqual(summary["invalid_counts"], {"answer_leak": 1})

    def test_legacy_validity_rate_is_reported_as_unobservable(self):
        hints = [self.hint("h1", "q1", 10)]
        hints[0]["validity_observable"] = False
        summary = hint_gen_compare.summarize_generator_subset(
            hints,
            {"h1": {"n_samples": 1, "n_correct": 1, "n_truncated": 0}},
            {"q1": {"n_samples": 1, "n_correct": 0}},
            {"h1": [{"raw_transfer": -0.25}]},
            ks=[1],
            bootstrap_samples=10,
            seed=1,
        )
        self.assertFalse(summary["validity_observable"])
        self.assertIsNone(summary["invalid_fraction"])
        self.assertEqual(summary["transfer"]["mean_clamped_nats_per_token"], 0.0)

    def test_paired_difference_bootstraps_questions(self):
        comparison = hint_gen_compare.paired_question_difference(
            {"q1": 2.0, "q2": 4.0},
            {"q1": 1.0, "q2": 1.0},
            samples=50,
            seed=9,
        )
        self.assertEqual(comparison["n_questions"], 2)
        self.assertEqual(comparison["mean_delta_candidate_minus_reference"], 2.0)


if __name__ == "__main__":
    unittest.main()
