"""Token-logratio CLI, alignment, and aggregate-analysis tests."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset, load_from_disk

from eval import token_logratios


class FakeTokenizer:
    def __init__(self, vocab, special_ids=(0, 1), template="template"):
        self._vocab = vocab
        self.all_special_ids = list(special_ids)
        self.chat_template = template
        self.name_or_path = "fake"

    def get_vocab(self):
        return self._vocab


class TokenLogratioCliTest(unittest.TestCase):
    def test_models_and_pi_are_independent_cli_options(self):
        args = token_logratios.build_parser().parse_args(
            [
                "score",
                "--student-model", "Qwen/Qwen3-1.7B",
                "--teacher-model", "Qwen/Qwen3-8B",
                "--pi", "solution",
            ]
        )
        self.assertEqual(args.student_model, "Qwen/Qwen3-1.7B")
        self.assertEqual(args.teacher_model, "Qwen/Qwen3-8B")
        self.assertEqual(token_logratios.canonical_pi_mode(args.pi), "full")

    def test_all_requested_pi_spellings_are_accepted(self):
        parser = token_logratios.build_parser()
        for pi in ("none", "hint", "answer", "solution", "full"):
            with self.subTest(pi=pi):
                args = parser.parse_args(
                    ["score", "--student-model", "s", "--teacher-model", "t", "--pi", pi]
                )
                self.assertEqual(args.pi, pi)


class IdentityAndTokenizerTest(unittest.TestCase):
    def test_legacy_rollout_identity_is_stable_and_row_disambiguated(self):
        row = {"question": "q", "completion_ids": [1, 2]}
        self.assertEqual(
            token_logratios.stable_rollout_id(row, 3),
            token_logratios.stable_rollout_id(row, 3),
        )
        self.assertNotEqual(
            token_logratios.stable_rollout_id(row, 3),
            token_logratios.stable_rollout_id(row, 4),
        )

    def test_tokenizer_mapping_mismatch_is_rejected(self):
        student = FakeTokenizer({"a": 1, "b": 2})
        compatible = FakeTokenizer({"b": 2, "a": 1}, template="other")
        student_meta, teacher_meta = token_logratios.verify_tokenizer_compatibility(
            student, compatible
        )
        self.assertEqual(student_meta["vocab_hash"], teacher_meta["vocab_hash"])
        with self.assertRaisesRegex(ValueError, "token-to-ID"):
            token_logratios.verify_tokenizer_compatibility(
                student, FakeTokenizer({"a": 2, "b": 1})
            )


class TraceMeasurementTest(unittest.TestCase):
    def test_ratio_entropy_and_concentration_measurements(self):
        row = {
            "rollout_id": "r",
            "question_id": "q",
            "reward": 1.0,
            "truncated": False,
            "n_tokens": 4,
            "teacher_logps": [1.0, -1.0, 0.5, -0.5],
            "student_logps": [0.0, 0.0, 0.0, 0.0],
            "teacher_entropy": [1.0, 2.0, 3.0, 4.0],
            "think_close_end": 2,
        }
        metrics = token_logratios.trace_metrics(row, [2.0, 2.0, 2.0, 2.0], 0.1)
        self.assertAlmostEqual(metrics["ratio_mean"], 0.0)
        self.assertAlmostEqual(metrics["positive_ratio_mass_per_token"], 0.375)
        self.assertAlmostEqual(metrics["negative_ratio_mass_per_token"], 0.375)
        self.assertAlmostEqual(metrics["positive_token_fraction"], 0.5)
        self.assertAlmostEqual(metrics["credit_mass_last_quartile"], 1 / 6)
        self.assertAlmostEqual(metrics["entropy_delta_mean"], 0.5)
        self.assertAlmostEqual(metrics["entropy_collapse_mass_per_token"], 0.25)
        self.assertAlmostEqual(metrics["entropy_expansion_mass_per_token"], 0.75)
        self.assertAlmostEqual(metrics["ratio_mean_think"], 0.0)
        self.assertAlmostEqual(metrics["entropy_mean_post"], 3.5)

    def test_question_effect_uses_only_mixed_questions(self):
        rows = [
            {"question_id": "mixed", "reward": 1.0, "ratio_mean": 0.4},
            {"question_id": "mixed", "reward": 0.0, "ratio_mean": -0.2},
            {"question_id": "easy", "reward": 1.0, "ratio_mean": 9.0},
        ]
        result = token_logratios.paired_question_effect(
            rows, "ratio_mean", True, bootstrap_samples=20, seed=1
        )
        self.assertEqual(result["n_mixed_questions"], 1)
        self.assertAlmostEqual(result["mean"], 0.6)


class SummarizeIntegrationTest(unittest.TestCase):
    @staticmethod
    def score_row(rollout_id, reward, teacher_logps, entropy):
        return {
            "rollout_id": rollout_id,
            "question_id": "question",
            "rollout_index": int(rollout_id[-1]),
            "reward": reward,
            "truncated": False,
            "n_tokens": 2,
            "teacher_logps": teacher_logps,
            "teacher_entropy": entropy,
            "think_close_end": -1,
        }

    @staticmethod
    def write_run(root, source, name, teacher_model, pi_mode, rows):
        path = root / name
        Dataset.from_list(rows).save_to_disk(str(path / "scores"))
        (path / "run_meta.json").write_text(
            json.dumps(
                {
                    "label": name,
                    "status": "complete",
                    "student_model": "student",
                    "teacher_model": teacher_model,
                    "pi_mode": pi_mode,
                    "is_student_reference": teacher_model == "student" and pi_mode == "none",
                    "source_rollout_fingerprint": token_logratios.fingerprint_ids(["r0", "r1"]),
                    "source_dataset_fingerprint": load_from_disk(str(source))._fingerprint,
                    "rollouts": str(source),
                    "n_source_rollouts": 2,
                }
            )
        )
        return path

    def test_manual_condition_directories_are_intersected_and_combined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "rollouts"
            Dataset.from_list(
                [
                    {"rollout_id": "r0", "student_logps": [-1.0, -1.0]},
                    {"rollout_id": "r1", "student_logps": [-1.0, -1.0]},
                ]
            ).save_to_disk(str(source))
            baseline = self.write_run(
                root,
                source,
                "self_none",
                "student",
                "none",
                [
                    self.score_row("r0", 0.0, [-1.0, -1.0], [2.0, 2.0]),
                    self.score_row("r1", 1.0, [-1.0, -1.0], [2.0, 2.0]),
                ],
            )
            hint = self.write_run(
                root,
                source,
                "self_hint",
                "student",
                "hint",
                [
                    self.score_row("r0", 0.0, [-1.2, -1.2], [2.5, 2.5]),
                    self.score_row("r1", 1.0, [-0.8, -0.8], [1.5, 1.5]),
                ],
            )
            output = root / "combined"
            args = argparse.Namespace(
                score_dirs=[str(baseline), str(hint)],
                output_dir=str(output),
                force=False,
                noise_quantile=0.99,
                noise_sample_tokens=100,
                position_bins=2,
                bootstrap_samples=20,
                seed=42,
            )
            token_logratios.summarize_runs(args)

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["reference_condition"], "self_none")
            self.assertEqual(summary["n_common_rollouts"], 2)
            hint_summary = summary["condition_summaries"]["self_hint"]
            self.assertAlmostEqual(
                hint_summary["within_question_ratio_correct_minus_wrong"]["mean"], 0.4
            )
            self.assertAlmostEqual(
                hint_summary["within_question_entropy_wrong_minus_correct"]["mean"], 1.0
            )
            traces = load_from_disk(str(output / "per_trace"))
            self.assertEqual(len(traces), 4)


if __name__ == "__main__":
    unittest.main()
