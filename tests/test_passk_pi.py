import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from datasets import Dataset

from eval import passk_pi
from eval.hint_compare_cache import ConditionCache


def rollout_cache(*, mixed_only=False):
    return Dataset.from_dict(
        {
            "question": ["q1", "q1", "q2", "q2"],
            "completion_text": [
                "q1 attempt zero",
                "q1 attempt one",
                "q2 attempt zero",
                "q2 attempt one",
            ],
            "question_idx": [0, 0, 1, 1],
            "sample_idx": [0, 1, 0, 1],
            # Opposite rewards ensure selecting sample_idx=1 cannot accidentally mean
            # selecting correct or incorrect attempts.
            "reward": [1.0, 0.0, 0.0, 1.0],
            "gen_model": ["student"] * 4,
            "dataset": ["deepmath"] * 4,
            "question_source": ["hints"] * 4,
            "mixed_only": [mixed_only] * 4,
            "generation_seed": [42] * 4,
            "max_completion_length": [128] * 4,
        }
    )


class RolloutPiPromptTest(unittest.TestCase):
    def test_rollout_is_presented_as_an_unverified_attempt(self):
        messages = passk_pi.build_teacher_messages(
            {"question": "Solve q", "rollout": "attempt text"}, "rollout"
        )
        user_text = messages[-1]["content"]
        self.assertIn("attempt text", user_text)
        self.assertIn("may or may not be correct", user_text)
        self.assertIn("Solve q", user_text)


class RolloutPiCacheTest(unittest.TestCase):
    def test_fixed_sample_index_is_selected_without_reward_filtering(self):
        with (
            mock.patch.object(passk_pi, "rollout_path", return_value="pi/cache"),
            mock.patch.object(passk_pi.os.path, "isdir", return_value=True),
            mock.patch.object(passk_pi, "load_from_disk", return_value=rollout_cache()),
        ):
            attempts, metadata = passk_pi.load_rollout_pi(
                "student", "deepmath", "pi", sample_idx=1
            )

        self.assertEqual(
            attempts,
            {
                0: passk_pi.RolloutAttempt("q1", "q1 attempt one", False),
                1: passk_pi.RolloutAttempt("q2", "q2 attempt one", True),
            },
        )
        self.assertEqual(metadata["sample_idx"], 1)
        self.assertEqual(
            metadata["selection_policy"], "fixed_sample_idx_without_reward"
        )
        self.assertEqual(
            metadata["reward_usage"], "posthoc_attempt_correctness_stratification_only"
        )
        self.assertEqual(metadata["n_available_questions"], 2)

    def test_duplicate_question_text_at_distinct_source_indices_is_allowed(self):
        cache = (
            rollout_cache().select([0, 2]).map(lambda _: {"question": "duplicate q"})
        )
        with (
            mock.patch.object(passk_pi, "rollout_path", return_value="pi/cache"),
            mock.patch.object(passk_pi.os.path, "isdir", return_value=True),
            mock.patch.object(passk_pi, "load_from_disk", return_value=cache),
        ):
            attempts, _ = passk_pi.load_rollout_pi(
                "student", "deepmath", "pi", sample_idx=0
            )

        self.assertEqual(set(attempts), {0, 1})
        self.assertEqual(attempts[0].question, attempts[1].question)

        problems = [
            {"question_idx": 0, "question": "duplicate q"},
            {"question_idx": 1, "question": "duplicate q"},
        ]
        passk_pi.attach_rollout_pi(problems, attempts)
        self.assertEqual(
            [problem["rollout"] for problem in problems],
            ["q1 attempt zero", "q2 attempt zero"],
        )
        self.assertEqual(
            [problem["attempt_correct"] for problem in problems],
            [True, False],
        )

    def test_attachment_rejects_source_index_text_mismatch(self):
        with self.assertRaisesRegex(ValueError, "caches disagree"):
            passk_pi.attach_rollout_pi(
                [{"question_idx": 0, "question": "hint question"}],
                {0: passk_pi.RolloutAttempt("rollout question", "attempt", True)},
            )

    def test_mixed_only_cache_is_rejected_as_verifier_selected(self):
        with (
            mock.patch.object(passk_pi, "rollout_path", return_value="pi/cache"),
            mock.patch.object(passk_pi.os.path, "isdir", return_value=True),
            mock.patch.object(
                passk_pi, "load_from_disk", return_value=rollout_cache(mixed_only=True)
            ),
            self.assertRaisesRegex(ValueError, "verifier outcomes"),
        ):
            passk_pi.load_rollout_pi("student", "deepmath", "pi", sample_idx=0)


class PairedRolloutDifferenceTest(unittest.TestCase):
    def test_paired_differences_are_stratified_by_attempt_correctness(self):
        problems = [
            {"question_idx": 0, "attempt_correct": True},
            {"question_idx": 1, "attempt_correct": True},
            {"question_idx": 2, "attempt_correct": False},
            {"question_idx": 3, "attempt_correct": False},
        ]

        def results(correct_counts):
            return [
                {"question_idx": idx, "n_samples": 2, "n_correct": count}
                for idx, count in enumerate(correct_counts)
            ]

        paired, per_problem = passk_pi.paired_rollout_minus_none(
            problems,
            {
                "none": results([0, 2, 1, 0]),
                "rollout": results([1, 2, 0, 1]),
            },
            ks=[1, 2],
            bootstrap_samples=200,
            seed=42,
        )

        correct = paired["attempt_correct"]
        self.assertEqual(correct["n_problems"], 2)
        self.assertAlmostEqual(correct["pass_at_k"]["pass@1"]["none"], 0.5)
        self.assertAlmostEqual(correct["pass_at_k"]["pass@1"]["rollout"], 0.75)
        self.assertAlmostEqual(correct["pass_at_k"]["pass@1"]["delta"], 0.25)
        self.assertAlmostEqual(correct["pass_at_k"]["pass@2"]["delta"], 0.5)

        incorrect = paired["attempt_incorrect"]
        self.assertEqual(incorrect["n_problems"], 2)
        self.assertAlmostEqual(incorrect["pass_at_k"]["pass@1"]["delta"], 0.0)
        self.assertAlmostEqual(incorrect["pass_at_k"]["pass@2"]["delta"], 0.0)
        self.assertEqual(len(per_problem), 4)
        self.assertEqual(per_problem[0]["pass_at_k"]["pass@2"]["delta"], 1.0)

    def test_bootstrap_interval_is_deterministic(self):
        first = passk_pi.paired_bootstrap_ci([-1.0, 0.0, 1.0], 200, seed=7)
        second = passk_pi.paired_bootstrap_ci([-1.0, 0.0, 1.0], 200, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 0.0)
        self.assertGreaterEqual(first[1], 0.0)

    def test_pairing_rejects_misaligned_question_indices(self):
        with self.assertRaisesRegex(ValueError, "Cannot pair rollout results"):
            passk_pi.paired_rollout_minus_none(
                [{"question_idx": 0, "attempt_correct": True}],
                {
                    "none": [{"question_idx": 0, "n_samples": 1, "n_correct": 0}],
                    "rollout": [{"question_idx": 1, "n_samples": 1, "n_correct": 1}],
                },
                ks=[1],
                bootstrap_samples=10,
                seed=42,
            )


class CommonProblemSetTest(unittest.TestCase):
    def test_eval_problem_sampling_is_restricted_to_rollout_cache_intersection(self):
        hints = Dataset.from_dict(
            {
                "question": ["q0", "q1", "q2"],
                "final_answer": ["0", "1", "2"],
                "hint": ["h0", "h1", "h2"],
                "gen_model": ["student"] * 3,
                "dataset": ["deepmath"] * 3,
            }
        )
        with (
            mock.patch.object(passk_pi, "hint_path", return_value="hints/cache"),
            mock.patch.object(passk_pi.os.path, "isdir", return_value=True),
            mock.patch.object(passk_pi, "load_from_disk", return_value=hints),
        ):
            problems = passk_pi.load_eval_problems(
                "student",
                num_problems=2,
                seed=42,
                need_full=False,
                required_question_indices={1, 2},
            )

        self.assertEqual({problem["question"] for problem in problems}, {"q1", "q2"})
        self.assertEqual({problem["question_idx"] for problem in problems}, {1, 2})

    def test_long_rollout_pi_can_bind_common_prompt_set(self):
        class Tokenizer:
            @staticmethod
            def apply_chat_template(conversations, **_kwargs):
                text = conversations[0][-1]["content"]
                length = 20 if "LONG ATTEMPT" in text else 5
                return {"input_ids": [list(range(length))]}

        problems = [
            {"question": "q1", "rollout": "short"},
            {"question": "q2", "rollout": "LONG ATTEMPT"},
        ]
        feasible = passk_pi.restrict_to_pi_feasible(
            problems, Tokenizer(), budget=10, pi_modes=["none", "rollout"]
        )
        self.assertEqual(feasible, [problems[0]])


class DemoCohortTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "model": "student",
            "model_identity": {"model": "student"},
            "revision": "rev",
            "cohort_hash": "cohort",
            "tokenizer_hash": "tok",
            "rollout_pi_root": "rollouts",
            "rollout_pi_sample_idx": 0,
        }
        self.cohort = [
            {
                "question_id": f"q{i}",
                "question_idx": i,
                "question": f"Question {i}",
                "final_answer": "42",
                "solution": "Trace</think>42",
                "rollout": "attempt",
                "hint": "old hint",
                "prompt_ids": {},
                "target_ids": [1],
            }
            for i in range(3)
        ]
        self.variants = {
            (r["question_id"], mode): [
                {
                    "sample_idx": 0,
                    "hint": f"{mode} advice",
                    "invalid_reason": "",
                    "truncated": False,
                },
                {
                    "sample_idx": 1,
                    "hint": "other advice",
                    "invalid_reason": "",
                    "truncated": False,
                },
            ]
            for r in self.cohort
            for mode in passk_pi.HINT_VARIANTS
        }
        self.meta = {
            "hints_hash": "hints",
            "config": {
                "model": "student",
                "model_identity": {"model": "student"},
                "revision": "rev",
                "samples_per_level": 2,
            },
        }

    def load(self, **kwargs):
        with (
            mock.patch.object(
                passk_pi, "load_cohort", return_value=(self.manifest, self.cohort)
            ),
            mock.patch.object(
                passk_pi, "load_variants", return_value=(self.variants, self.meta)
            ),
            mock.patch.object(Path, "is_file", return_value=True),
        ):
            return passk_pi.load_demo_problems(
                "cohort", "student", passk_pi.DEFAULT_PI_MODES, 0, **kwargs
            )

    def test_common_valid_set_uses_fixed_hint_sample_not_best_available(self):
        self.variants[("q0", "hint_short")][0]["truncated"] = True
        self.variants[("q1", "hint_detailed")][0]["invalid_reason"] = "answer_leak"
        rows, metadata = self.load()
        self.assertEqual([r["question_id"] for r in rows], ["q2"])
        self.assertEqual(metadata["n_prepared"], 3)
        self.assertEqual(metadata["n_valid_pi"], 1)
        self.assertEqual(rows[0]["hint_short"], "hint_short advice")
        self.assertNotIn("target_ids", rows[0])
        self.assertEqual(len(self.load(hint_sample_idx=1)[0]), 3)

    def test_missing_artifact_is_error_even_if_another_hint_is_invalid(self):
        self.variants[("q0", "hint_short")][0]["truncated"] = True
        del self.variants[("q0", "hint_detailed")]
        with self.assertRaisesRegex(ValueError, "Missing fixed"):
            self.load()

    def test_foreign_teacher_or_revision_is_rejected(self):
        for key, value in [
            ("model", "another-teacher"),
            ("revision", "another-revision"),
        ]:
            with (
                self.subTest(key=key),
                mock.patch.dict(self.meta["config"], {key: value}),
                self.assertRaisesRegex(ValueError, "Hint generator"),
            ):
                self.load()

    def test_variant_prompts_use_the_requested_hint_only(self):
        row = self.load()[0][0]
        for mode in passk_pi.HINT_VARIANTS:
            messages = passk_pi.build_teacher_messages(row, mode)
            self.assertIn(f"{mode} advice", messages[-1]["content"])
            self.assertNotIn("Trace</think>42", messages[-1]["content"])
            for other in set(passk_pi.HINT_VARIANTS) - {mode}:
                self.assertNotIn(f"{other} advice", messages[-1]["content"])
        self.assertNotIn(
            "advice", passk_pi.build_teacher_messages(row, "none")[-1]["content"]
        )
        self.assertIn(
            "Trace</think>42",
            passk_pi.build_teacher_messages(row, "full")[-1]["content"],
        )


class AllArmPairingTest(unittest.TestCase):
    def test_pairing_uses_question_identity_and_unbiased_pass_at_k(self):
        baseline = [
            {"question_idx": 1, "n_samples": 4, "n_correct": 1},
            {"question_idx": 2, "n_samples": 4, "n_correct": 0},
        ]
        arm = [
            {"question_idx": 2, "n_samples": 4, "n_correct": 4},
            {"question_idx": 1, "n_samples": 4, "n_correct": 2},
        ]
        report = passk_pi.paired_against_none(
            {"none": baseline, "hint_short": arm}, [1, 2, 4], 100, 42
        )
        self.assertEqual(report["none"]["pass@2"]["mean"], 0.25)
        self.assertAlmostEqual(report["hint_short"]["pass@2"]["mean"], 11 / 12)
        self.assertAlmostEqual(report["hint_short"]["pass@2"]["delta_vs_none"], 2 / 3)
        self.assertEqual(report["none"]["pass@4"]["delta_ci95"], [0, 0])
        with self.assertRaisesRegex(ValueError, "Cannot pair"):
            passk_pi.paired_against_none(
                {"none": baseline, "full": arm[:1]}, [1], 100, 42
            )


class GenerationCacheTest(unittest.TestCase):
    def test_resume_preserves_samples_and_reports_truncation(self):
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kw: (
            str(messages) + str(kw["enable_thinking"])
        )
        llm = mock.Mock()
        llm.generate.return_value = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="correct", finish_reason="stop", token_ids=[1, 2]
                    ),
                    SimpleNamespace(
                        text="unfinished", finish_reason="length", token_ids=[3, 4]
                    ),
                ]
            )
        ]
        problems = [
            {"question_idx": 7, "question_id": "qid", "question": "q", "answer": "42"}
        ]
        sampling = SimpleNamespace(n=2)
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                passk_pi,
                "grade",
                side_effect=lambda text, *_: (None, text == "correct"),
            ),
        ):
            cache = ConditionCache(Path(directory), "generation", {"n": 2})
            first = passk_pi.eval_pi_mode(
                llm, tokenizer, problems, "none", sampling, cache=cache
            )
            resumed = passk_pi.eval_pi_mode(
                llm, tokenizer, problems, "none", sampling, cache=cache
            )
            self.assertEqual(first, resumed)
            self.assertEqual(first[0]["n_correct"], 1)
            self.assertEqual(first[0]["n_truncated"], 1)
            self.assertEqual(first[0]["samples"][1]["text"], "unfinished")
            self.assertEqual(llm.generate.call_count, 1)
            passk_pi.eval_pi_mode(
                llm,
                tokenizer,
                problems,
                "none",
                sampling,
                cache=cache,
                enable_thinking=False,
            )
            self.assertEqual(llm.generate.call_count, 2)

    def test_all_hint_prompts_participate_in_context_filter(self):
        class Tokenizer:
            @staticmethod
            def apply_chat_template(conversations, **kwargs):
                assert kwargs["enable_thinking"] is False
                text = conversations[0][-1]["content"]
                return {"input_ids": [list(range(50 if "long hint" in text else 5))]}

        problems = [
            {"question": "q1", "hint_short": "short"},
            {"question": "q2", "hint_short": "long hint"},
        ]
        actual = passk_pi.restrict_to_pi_feasible(
            problems, Tokenizer(), 10, ["none", "hint_short"], enable_thinking=False
        )
        self.assertEqual(actual, problems[:1])


class PasskCliTest(unittest.TestCase):
    def test_preflight_then_all_arms_resume_and_settings_guard_without_model(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, **kwargs):
                if tokenize:
                    return {"input_ids": [[1, 2, 3]]}
                return str(messages)

        problems = [
            {
                "question_idx": i,
                "question_id": f"q{i}",
                "question": f"Question {i}",
                "answer": "42",
                "solution": "trace</think>42",
                "rollout": "attempt",
                **{mode: mode + " advice" for mode in passk_pi.HINT_VARIANTS},
            }
            for i in range(2)
        ]
        generated = SimpleNamespace(
            outputs=[
                SimpleNamespace(text="correct", finish_reason="stop", token_ids=[1]),
                SimpleNamespace(text="wrong", finish_reason="length", token_ids=[2]),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                passk_pi,
                "load_demo_problems",
                return_value=(problems, {"revision": "rev", "tokenizer_hash": "tok"}),
            ),
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(_commit_hash="rev"),
            ),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
            ),
            mock.patch.object(passk_pi, "tokenizer_hash", return_value="tok"),
            mock.patch.object(passk_pi, "LLM") as llm,
            mock.patch.object(
                passk_pi,
                "grade",
                side_effect=lambda text, *_: (None, text == "correct"),
            ),
        ):
            llm.return_value.generate.side_effect = lambda prompts, _: [
                generated for p in prompts
            ]
            argv = [
                "passk_pi",
                "--model",
                "student",
                "--cohort-dir",
                "cohort",
                "--num-problems",
                "0",
                "--n",
                "2",
                "--k",
                "1",
                "2",
                "--paired-bootstrap-samples",
                "20",
                "--output-dir",
                directory,
            ]
            with mock.patch.object(sys, "argv", argv + ["--prepare-only"]):
                passk_pi.main()
            llm.assert_not_called()
            with mock.patch.object(sys, "argv", argv):
                passk_pi.main()
                passk_pi.main()
            self.assertEqual(llm.return_value.generate.call_count, 7)
            summary = json.loads(
                (Path(directory) / "student/passk_pi_summary.json").read_text()
            )
            self.assertEqual(set(summary["pass_at_k"]), set(passk_pi.DEFAULT_PI_MODES))
            self.assertEqual(
                summary["pass_at_k"]["none"], {"pass@1": 0.5, "pass@2": 1.0}
            )
            self.assertEqual(summary["truncation_rate"]["hint_short"], 0.5)
            self.assertEqual(summary["cache_stats"], {"hits": 14, "misses": 0})
            self.assertEqual(
                summary["paired_against_none"]["full"]["pass@1"]["delta_vs_none"], 0
            )
            with (
                mock.patch.object(sys, "argv", argv + ["--seed", "7"]),
                self.assertRaisesRegex(ValueError, "use a new --output-dir"),
            ):
                passk_pi.main()


if __name__ == "__main__":
    unittest.main()
