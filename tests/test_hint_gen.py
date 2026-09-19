import unittest

import torch
from datasets import Dataset

from train.opsd.train_hint_gen.lib import (
    CompositeHintReward,
    ConstrainedHintReward,
    ConstrainedHintRewardConfig,
    HintRewardConfig,
    completion_text,
    composite_reward,
    constrained_reward,
    dual_ascent_step,
    group_student_rollouts,
    index_student_rollouts,
    invalid_hint_reason,
    normalized_hint_cost,
    rank_invalid_hints,
    sampled_reverse_kl,
)


class FakeBackend:
    def __init__(self, sufficiency=0.75, transfer=0.2):
        self.sufficiency = sufficiency
        self.transfer = transfer
        self.calls = []
        self.last_metrics = {}

    def score_hints(self, questions, answers, hints):
        return [(self.score_sufficiency(q, a, h), self.score_transfer(q, h))
                for q, a, h in zip(questions, answers, hints, strict=True)]

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

    def test_constrained_reward_and_projected_dual_step(self):
        self.assertAlmostEqual(
            constrained_reward(0.75, 0.25, 0.2, 0.7, 0.5, 2.0),
            -0.25,
        )
        self.assertEqual(dual_ascent_step(0.1, 0.0, 1.0, 1.0, 5.0), 0.0)
        self.assertEqual(dual_ascent_step(4.9, 1.0, 0.0, 1.0, 5.0), 5.0)

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

    def test_all_invalid_group_has_zero_reward_without_teacher_call(self):
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

        self.assertEqual(values, [0.0])
        self.assertEqual(backend.calls, [])


class ConstrainedHintRewardTest(unittest.TestCase):
    def test_computes_reward_updates_dual_and_logs_state(self):
        config = ConstrainedHintRewardConfig(
            model="m",
            dataset="d",
            hint_budget=8,
            tau=0.7,
            gamma=0.5,
            dual_lr=0.1,
            dual_init=2.0,
            dual_max=3.0,
        )
        backend = FakeBackend(sufficiency=0.75, transfer=0.2)
        reward = ConstrainedHintReward(config, backend)
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

        self.assertAlmostEqual(values[0], -0.25)
        self.assertAlmostEqual(reward.dual_lambda, 1.995)
        self.assertEqual(reward.dual_updates, 1)
        self.assertAlmostEqual(extras["hint_constraint_margin"][0], 0.05)
        self.assertEqual(extras["hint_dual_lambda"], [2.0])
        self.assertEqual(metrics["hint/dual_lambda"], 2.0)
        self.assertAlmostEqual(metrics["hint/dual_lambda_next"], 1.995)
        self.assertEqual(len(backend.calls), 2)

    def test_all_invalid_group_still_updates_constraint_multiplier(self):
        config = ConstrainedHintRewardConfig(
            model="m",
            dataset="d",
            hint_budget=8,
            tau=0.75,
            invalid_penalty=1.5,
            dual_lr=0.1,
            dual_init=2.0,
        )
        backend = FakeBackend()
        reward = ConstrainedHintReward(config, backend)

        values = reward(
            prompts=[[]],
            completions=[[{"role": "assistant", "content": "The answer is 17"}]],
            completion_ids=[[1, 2, 3, 4]],
            question=["q"],
            final_answer=["17"],
        )

        self.assertEqual(values, [0.0])
        self.assertAlmostEqual(reward.dual_lambda, 2.075)
        self.assertEqual(backend.calls, [])

    def test_dual_state_round_trip_and_config_guard(self):
        config = ConstrainedHintRewardConfig(
            model="m", dataset="d", tau=0.7, gamma=4.0, dual_lr=0.1
        )
        reward = ConstrainedHintReward(config, FakeBackend())
        reward.dual_lambda = 1.25
        reward.dual_updates = 9

        restored = ConstrainedHintReward(config, FakeBackend())
        restored.load_state_dict(reward.state_dict())
        self.assertEqual(restored.dual_lambda, 1.25)
        self.assertEqual(restored.dual_updates, 9)

        incompatible = reward.state_dict()
        incompatible["gamma"] = 1.0
        with self.assertRaisesRegex(ValueError, "does not match"):
            restored.load_state_dict(incompatible)

    def test_config_validates_tau_and_dual_range(self):
        with self.assertRaisesRegex(ValueError, "tau"):
            ConstrainedHintRewardConfig(model="m", dataset="d", tau=1.1).validate()
        with self.assertRaisesRegex(ValueError, "dual_init"):
            ConstrainedHintRewardConfig(
                model="m", dataset="d", dual_init=2.0, dual_max=1.0
            ).validate()


class AnswerLeakRegressionTest(unittest.TestCase):
    def test_rejects_sentence_punctuation_and_equivalent_numeric_answers(self):
        for text in [
            "The answer is 17.", "The answer is 17!", "The answer is 17,",
            "The answer is 017.", "The answer is 17.0.",
            "The answer is seventeen.", "17.",
        ]:
            with self.subTest(text=text):
                self.assertEqual(invalid_hint_reason(text, "17"), "answer_leak")

    def test_rejects_explicit_single_digit_answers_but_allows_intermediate_numbers(self):
        for text in [
            "The final answer is 7.", "Answer: 7.", r"The answer is $7$.",
            r"The answer is \(7\).", "The answer is **7**.",
            "The final value equals seven.", "7", "seven.",
        ]:
            with self.subTest(text=text):
                self.assertEqual(invalid_hint_reason(text, "7"), "answer_leak")
        for text in ["Work modulo 7.", "Consider 7 cases.", "Use 7 as an intermediate value."]:
            with self.subTest(text=text):
                self.assertIsNone(invalid_hint_reason(text, "7"))

    def test_does_not_match_substrings_or_parts_of_decimal_numbers(self):
        for text in ["Use 117 cases.", "Consider 17.5.", "Use 0.17.", "Use .17.", "Use x17.", "Consider -17."]:
            with self.subTest(text=text):
                self.assertIsNone(invalid_hint_reason(text, "17"))
        self.assertIsNone(invalid_hint_reason("The answer is 7.5.", "7"))
        self.assertIsNone(invalid_hint_reason("The answer is found using parity.", "7"))

    def test_non_numeric_answers_and_boxed_answers(self):
        self.assertEqual(invalid_hint_reason("The answer is x+y.", "x+y"), "answer_leak")
        self.assertEqual(invalid_hint_reason(r"Thus \boxed{7}.", "7"), "answer_leak")
        self.assertEqual(invalid_hint_reason("The answer is forty-two.", "42"), "answer_leak")


class InvalidRewardRankingRegressionTest(unittest.TestCase):
    def test_invalid_hints_have_negative_advantages_even_with_large_transfer_costs(self):
        for reward_type, config_type in [
            (CompositeHintReward, HintRewardConfig),
            (ConstrainedHintReward, ConstrainedHintRewardConfig),
        ]:
            for gamma in [7.0, 1000.0]:
                with self.subTest(reward=reward_type.__name__, gamma=gamma):
                    config = config_type(model="m", dataset="deepmath", gamma=gamma)
                    backend = FakeBackend(sufficiency=0.75, transfer=0.3)
                    reward = reward_type(config, backend)
                    extras = {}
                    values = reward(
                        prompts=[[]] * 4,
                        completions=["Use parity.", "The answer is 17.", "", "Use symmetry."],
                        completion_ids=[[1] * 32, [2] * 8, [3], [4] * 64],
                        question=["q"] * 4, final_answer=["17"] * 4,
                        log_extra=lambda name, value: extras.__setitem__(name, value),
                    )
                    self.assertEqual(values[1], values[2])
                    self.assertLess(values[1], min(values[0], values[3]))
                    rewards = torch.tensor(values, dtype=torch.float32)
                    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
                    self.assertTrue((advantages[[1, 2]] < 0).all())
                    self.assertEqual(len(backend.calls), 4)  # Only the two valid hints scored.
                    self.assertEqual(extras['hint_raw_reward'][1:3], [None, None])
                    self.assertEqual(extras['hint_reward'], values)
                    self.assertEqual(extras['hint_cost'][1:3], [8 / 128, 1 / 128])
                    if isinstance(reward, ConstrainedHintReward):
                        self.assertAlmostEqual(reward.dual_lambda, 1 + 0.05 * (0.7 - 0.375))
                        self.assertEqual(reward.dual_updates, 1)

    def test_all_invalid_hints_have_zero_advantage_regardless_of_length(self):
        for reward_type, config_type in [
            (CompositeHintReward, HintRewardConfig),
            (ConstrainedHintReward, ConstrainedHintRewardConfig),
        ]:
            with self.subTest(reward=reward_type.__name__):
                backend = FakeBackend()
                reward = reward_type(config_type(model="m", dataset="deepmath"), backend)
                values = reward(
                    prompts=[[]] * 3, completions=["", "The answer is 7.", "<think>hidden</think>"],
                    completion_ids=[[1], [2] * 20, [3] * 128],
                    question=["q"] * 3, final_answer=["7"] * 3,
                )
                self.assertEqual(values, [0.0, 0.0, 0.0])
                self.assertEqual(backend.calls, [])
                if isinstance(reward, ConstrainedHintReward):
                    self.assertAlmostEqual(reward.dual_lambda, 1.035)
                    self.assertEqual(reward.dual_updates, 1)

    def test_separate_questions_do_not_share_reward_floors(self):
        values = rank_invalid_hints(
            [-10, 0, 5, 0, 0], [0, 1, 0, 1, 1], ['a', 'a', 'b', 'b', 'c'], 1.0,
        )
        self.assertEqual(values, [-10, -11, 5, 4, 0])

    def test_floor_remains_strict_after_trl_float32_conversion(self):
        values = rank_invalid_hints([-1e10, 0], [0, 1], ['q', 'q'], 1.0)
        values = torch.tensor(values, dtype=torch.float32)
        self.assertLess(values[1], values[0])

    def test_rejects_nonfinite_rewards_and_nonpositive_margins(self):
        for value in [float('nan'), float('inf')]:
            with self.assertRaisesRegex(ValueError, 'finite'):
                rank_invalid_hints([value, 0], [0, 1], ['q', 'q'], 1.0)
        for config_type in [HintRewardConfig, ConstrainedHintRewardConfig]:
            for margin in [0, -1, float('nan'), float('inf')]:
                with self.subTest(config=config_type.__name__, margin=margin):
                    with self.assertRaisesRegex(ValueError, 'invalid_penalty'):
                        config_type(model='m', dataset='deepmath', invalid_penalty=margin).validate()


if __name__ == "__main__":
    unittest.main()
