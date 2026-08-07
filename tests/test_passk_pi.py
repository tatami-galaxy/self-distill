import unittest
from unittest import mock

from datasets import Dataset

from eval import passk_pi


def rollout_cache(*, mixed_only=False):
    return Dataset.from_dict({
        "question": ["q1", "q1", "q2", "q2"],
        "completion_text": ["q1 attempt zero", "q1 attempt one",
                            "q2 attempt zero", "q2 attempt one"],
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
    })


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
        self.assertEqual(metadata["selection_policy"], "fixed_sample_idx_without_reward")
        self.assertEqual(
            metadata["reward_usage"], "posthoc_attempt_correctness_stratification_only"
        )
        self.assertEqual(metadata["n_available_questions"], 2)

    def test_duplicate_question_text_at_distinct_source_indices_is_allowed(self):
        cache = rollout_cache().select([0, 2]).map(lambda _: {"question": "duplicate q"})
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
        ):
            with self.assertRaisesRegex(ValueError, "verifier outcomes"):
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
        hints = Dataset.from_dict({
            "question": ["q0", "q1", "q2"],
            "final_answer": ["0", "1", "2"],
            "hint": ["h0", "h1", "h2"],
            "gen_model": ["student"] * 3,
            "dataset": ["deepmath"] * 3,
        })
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


if __name__ == "__main__":
    unittest.main()
