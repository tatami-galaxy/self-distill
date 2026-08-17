import unittest

import torch
from datasets import Dataset

from train.opsd.train_hint_gen.lib import (
    CompositeHintReward,
    HintRewardConfig,
    completion_text,
    composite_reward,
    group_student_rollouts,
    index_student_rollouts,
    invalid_hint_reason,
    normalized_hint_cost,
    sampled_reverse_kl,
)


class FakeBackend:
    def __init__(self, sufficiency=0.75, transfer=0.2):
        self.sufficiency = sufficiency
        self.transfer = transfer
        self.calls = []

    def score_sufficiency(self, question, final_answer, hint):
        self.calls.append(("s", question, final_answer, hint))
        return self.sufficiency

    def score_transfer(self, question, hint):
        self.calls.append(("t", question, hint))
        return self.transfer


class HintGenerationHelpersTest(unittest.TestCase):
    def test_conversational_completion_text(self):
        completion = [{"role": "assistant", "content": "  use AM-GM  "}]
        self.assertEqual(completion_text(completion), "use AM-GM")

    def test_invalid_reasons_reuse_existing_leak_policy(self):
        self.assertEqual(invalid_hint_reason("", "17"), "empty")
        self.assertEqual(invalid_hint_reason("<think>x</think>idea", "17"), "thinking")
        self.assertEqual(invalid_hint_reason("The answer is 17", "17"), "answer_leak")
        self.assertIsNone(invalid_hint_reason("Use modular arithmetic", "17"))

    def test_cost_and_composite_reward(self):
        self.assertEqual(normalized_hint_cost(list(range(32)), 128), 0.25)
        self.assertEqual(normalized_hint_cost(list(range(200)), 128), 1.0)
        self.assertAlmostEqual(composite_reward(0.75, 0.25, 0.2, 2.0, 0.5), 1.15)

    def test_sampled_reverse_kl_and_clamp(self):
        student = torch.tensor([-1.0, -2.0])
        teacher = torch.tensor([-1.5, -2.5])
        self.assertAlmostEqual(sampled_reverse_kl(student, teacher), 0.5)
        self.assertEqual(sampled_reverse_kl(teacher, student), 0.0)
        self.assertAlmostEqual(
            sampled_reverse_kl(teacher, student, clamp_nonnegative=False), -0.5
        )

    def test_rollout_group_validation_and_order(self):
        rows = [
            {
                "question": "q",
                "completion_ids": [2],
                "sample_idx": 1,
                "gen_model": "m",
                "dataset": "d",
            },
            {
                "question": "q",
                "completion_ids": [1],
                "sample_idx": 0,
                "gen_model": "m",
                "dataset": "d",
            },
        ]
        grouped = group_student_rollouts(rows, "m", "d")
        self.assertEqual([r["sample_idx"] for r in grouped["q"]], [0, 1])
        with self.assertRaisesRegex(ValueError, "not 'other'"):
            group_student_rollouts(rows, "other", "d")

    def test_arrow_rollout_index_does_not_materialize_token_columns(self):
        dataset = Dataset.from_list(
            [
                {
                    "question": "q",
                    "completion_ids": [2],
                    "sample_idx": 1,
                    "rollout_id": "b",
                    "gen_model": "m",
                    "dataset": "d",
                },
                {
                    "question": "q",
                    "completion_ids": [1],
                    "sample_idx": 0,
                    "rollout_id": "a",
                    "gen_model": "m",
                    "dataset": "d",
                },
            ]
        )
        self.assertEqual(index_student_rollouts(dataset, "m", "d"), {"q": [1, 0]})


class CompositeHintRewardTest(unittest.TestCase):
    def test_computes_formula_and_logs_components(self):
        config = HintRewardConfig(
            model="m",
            dataset="d",
            hint_budget=8,
            alpha=2.0,
            gamma=0.5,
        )
        backend = FakeBackend(sufficiency=0.75, transfer=0.2)
        reward = CompositeHintReward(config, backend)
        extras, metrics = {}, {}

        values = reward(
            prompts=[[{"role": "user", "content": "generator prompt"}]],
            completions=[[{"role": "assistant", "content": "Use parity."}]],
            completion_ids=[[10, 11]],
            question=["q"],
            final_answer=["17"],
            log_extra=lambda name, value: extras.__setitem__(name, value),
            log_metric=lambda name, value: metrics.__setitem__(name, value),
        )

        self.assertAlmostEqual(values[0], 1.15)
        self.assertEqual(extras["hint_sufficiency"], [0.75])
        self.assertEqual(extras["hint_cost"], [0.25])
        self.assertEqual(extras["hint_transfer"], [0.2])
        self.assertEqual(metrics["hint/invalid_fraction"], 0.0)
        self.assertEqual(len(backend.calls), 2)

    def test_invalid_hint_gets_penalty_without_teacher_call(self):
        config = HintRewardConfig(
            model="m", dataset="d", hint_budget=8, invalid_penalty=1.5
        )
        backend = FakeBackend()
        reward = CompositeHintReward(config, backend)

        values = reward(
            prompts=[[]],
            completions=[[{"role": "assistant", "content": "The answer is 17"}]],
            completion_ids=[[1, 2, 3, 4]],
            question=["q"],
            final_answer=["17"],
        )

        self.assertEqual(values, [-2.0])
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
