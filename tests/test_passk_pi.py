import unittest
from unittest import mock

from datasets import Dataset

from eval import passk_pi


def rollout_cache(*, mixed_only=False):
    return Dataset.from_dict({
        "question": ["q1", "q1", "q2", "q2"],
        "completion_text": ["q1 attempt zero", "q1 attempt one",
                            "q2 attempt zero", "q2 attempt one"],
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

        self.assertEqual(attempts, {"q1": "q1 attempt one", "q2": "q2 attempt one"})
        self.assertEqual(metadata["sample_idx"], 1)
        self.assertEqual(metadata["selection_policy"], "fixed_sample_idx_without_reward")
        self.assertEqual(metadata["n_available_questions"], 2)

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
                required_questions={"q1", "q2"},
            )

        self.assertEqual({problem["question"] for problem in problems}, {"q1", "q2"})

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
