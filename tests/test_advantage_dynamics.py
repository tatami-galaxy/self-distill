"""Unit tests for the checkpoint-level SDFT advantage evaluator."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eval import advantage_dynamics_sdft
from utils import format_prompt_math


class FakeTokenizer:
    def __init__(self, vocab, special_ids=(0, 1), template="template"):
        self._vocab = vocab
        self.all_special_ids = list(special_ids)
        self.chat_template = template
        self.name_or_path = "fake"

    def get_vocab(self):
        return self._vocab


def compatible_training_args(**overrides):
    values = {
        "distillation_mode": "sampled_token",
        "distillation_alpha": 1.0,
        "teacher_model_kind": "base",
        "generate_from_teacher": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": None,
        "repetition_penalty": 1.0,
        "gradient_accumulation_steps": 16,
        "steps_per_generation": 16,
        "num_iterations": 1,
        "distillation_is_clip": 2.0,
        "num_loss_tokens_to_skip": 0,
        "max_completion_length": 8192,
        "max_prompt_length": 8192,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class AdvantageDefinitionTest(unittest.TestCase):
    def test_advantage_is_teacher_minus_student_log_probability(self):
        teacher = [-1.0, -3.0, -2.0]
        student = [-2.0, -2.0, -2.5]
        advantages = advantage_dynamics_sdft.compute_advantages(teacher, student)
        self.assertEqual(advantages, [1.0, -1.0, 0.5])
        sampled_reverse_kl = sum(s - t for s, t in zip(student, teacher)) / 3
        self.assertAlmostEqual(sum(advantages) / 3, -sampled_reverse_kl)

    def test_misaligned_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            advantage_dynamics_sdft.compute_advantages([-1.0], [-1.0, -2.0])

    def test_summary_is_token_weighted_and_honors_skipped_tokens(self):
        rows = [
            {
                "eval_pi_mode": "hint",
                "question_id": "q1",
                "reward": 1.0,
                "truncated": False,
                "loss_token_start": 1,
                "advantages": [99.0, 3.0],
            },
            {
                "eval_pi_mode": "hint",
                "question_id": "q2",
                "reward": 0.0,
                "truncated": True,
                "loss_token_start": 0,
                "advantages": [1.0, 1.0, 1.0],
            },
        ]
        summary = advantage_dynamics_sdft.summarize_score_rows(rows, bootstrap_samples=20, seed=7)
        hint = summary["hint"]
        self.assertAlmostEqual(hint["mean_advantage_per_token"], 1.5)
        self.assertAlmostEqual(hint["mean_rollout_advantage"], 2.0)
        self.assertEqual(hint["num_tokens"], 4)
        self.assertEqual(hint["by_outcome"]["correct"]["num_tokens"], 1)
        self.assertEqual(hint["by_outcome"]["truncated"]["num_tokens"], 3)

    def test_question_bootstrap_is_deterministic(self):
        totals = {"a": (2.0, 2), "b": (-1.0, 1)}
        first = advantage_dynamics_sdft.question_cluster_bootstrap_ci(totals, 50, 3)
        second = advantage_dynamics_sdft.question_cluster_bootstrap_ci(totals, 50, 3)
        self.assertEqual(first, second)

    def test_dynamics_reuses_already_computed_step_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = {
                "output_dir": temporary,
                "run_dir": "run",
                "base_model": "base",
                "dataset": "deepmath",
                "training_pi_mode": "hint",
            }
            summaries = [{"step": 20, "advantages": {}}]
            with mock.patch.object(
                advantage_dynamics_sdft,
                "aggregate_step",
                side_effect=AssertionError("aggregate_step should not be called"),
            ):
                dynamics = advantage_dynamics_sdft.aggregate_dynamics(
                    SimpleNamespace(), run, [20], summaries
                )
            self.assertEqual(dynamics["steps"], summaries)
            self.assertTrue((Path(temporary) / "dynamics.json").is_file())


class ProvenanceTest(unittest.TestCase):
    def test_tokenizer_mapping_mismatch_is_rejected(self):
        student = FakeTokenizer({"a": 1, "b": 2})
        compatible = FakeTokenizer({"b": 2, "a": 1}, template="different")
        student_meta, teacher_meta = advantage_dynamics_sdft.verify_tokenizer_compatibility(
            student, compatible
        )
        self.assertEqual(student_meta["vocab_hash"], teacher_meta["vocab_hash"])
        with self.assertRaisesRegex(ValueError, "token-to-ID"):
            advantage_dynamics_sdft.verify_tokenizer_compatibility(
                student, FakeTokenizer({"a": 2, "b": 1})
            )

    def test_checkpoint_discovery_is_numeric_and_includes_base_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint-20").mkdir()
            (root / "checkpoint-3").mkdir()
            (root / "checkpoint-final").mkdir()
            (root / "checkpoint-4").write_text("not a directory")
            checkpoints = advantage_dynamics_sdft.discover_checkpoints(root)
            self.assertEqual(list(checkpoints), [0, 3, 20])
            self.assertEqual(checkpoints[0], "")

    def test_training_configuration_accepts_current_runs(self):
        config = advantage_dynamics_sdft.validate_training_configuration(
            compatible_training_args()
        )
        self.assertFalse(config["importance_correction_active"])
        self.assertEqual(config["distillation_is_clip"], 2.0)

    def test_training_configuration_rejects_non_base_teacher(self):
        with self.assertRaisesRegex(ValueError, "teacher_model_kind"):
            advantage_dynamics_sdft.validate_training_configuration(
                compatible_training_args(teacher_model_kind="ema")
            )

    def test_training_configuration_rejects_active_importance_correction(self):
        with self.assertRaisesRegex(ValueError, "importance correction"):
            advantage_dynamics_sdft.validate_training_configuration(
                compatible_training_args(gradient_accumulation_steps=15)
            )

    def test_ids_are_stable_and_distinguish_samples(self):
        question_id = advantage_dynamics_sdft.stable_question_id("q", "a")
        first = advantage_dynamics_sdft.stable_rollout_id("checkpoint", question_id, 0, [1, 2], 42)
        self.assertEqual(
            first,
            advantage_dynamics_sdft.stable_rollout_id("checkpoint", question_id, 0, [1, 2], 42),
        )
        self.assertNotEqual(
            first,
            advantage_dynamics_sdft.stable_rollout_id("checkpoint", question_id, 1, [1, 2], 42),
        )


class PromptAndCliTest(unittest.TestCase):
    def test_none_is_student_prompt_and_pi_is_inserted(self):
        problem = {
            "question": "What is 1+1?",
            "final_answer": "2",
            "hint": "Add the units.",
            "solution": "One plus one is two.",
            "rollout": "I tried adding.",
        }
        self.assertEqual(
            advantage_dynamics_sdft.build_teacher_messages(problem, "none"),
            format_prompt_math(problem["question"]),
        )
        for pi_mode, text in (
            ("answer", "2"),
            ("hint", "Add the units."),
            ("full", "One plus one is two."),
            ("rollout", "I tried adding."),
        ):
            with self.subTest(pi_mode=pi_mode):
                messages = advantage_dynamics_sdft.build_teacher_messages(problem, pi_mode)
                self.assertIn(text, messages[-1]["content"])

    def test_parser_requires_training_pi(self):
        with self.assertRaises(SystemExit):
            advantage_dynamics_sdft.build_parser().parse_args(["--run-dir", "run"])

    def test_parser_accepts_one_training_pi(self):
        args = advantage_dynamics_sdft.build_parser().parse_args(
            ["--run-dir", "run", "--pi-mode", "hint"]
        )
        self.assertEqual(args.phase, "sweep")
        self.assertEqual(args.pi_mode, "hint")
        self.assertIsNone(args.steps)

    def test_run_pi_mismatch_is_rejected_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "run_meta.json").write_text(
                '{"model":"base","dataset":"deepmath","pi_mode":"hint"}'
            )
            args = advantage_dynamics_sdft.build_parser().parse_args(
                ["--run-dir", temporary, "--pi-mode", "full"]
            )
            with self.assertRaisesRegex(ValueError, "does not match training PI"):
                advantage_dynamics_sdft.prepare_run(args)


if __name__ == "__main__":
    unittest.main()
