"""Asymmetric PPO critic: PI isolation, rollout caching, and prompt alignment."""

import types
import unittest
from collections import defaultdict
from unittest.mock import Mock, patch

import torch
from datasets import Dataset

from tests.helpers import FakeChatTokenizer, make_prompt_stub
from train.ppo.train_ppo import PPOTrainer
from train.ppo.train_ppo_pi import (
    VALUE_PROMPT_VERSIONS,
    PPOPITrainer,
    build_ppo_pi_dataset,
    build_run_meta,
)
from utils import PI_ANSWER, format_prompt_math


class PPIPIDatasetTest(unittest.TestCase):
    def source_dataset(self):
        return Dataset.from_dict(
            {
                "question": ["What is 40 + 33?", "What is 6 times 9?"],
                "final_answer": ["73", "54"],
                "solution": ["unused demo 1", "unused demo 2"],
            }
        )

    def test_answer_and_none_are_matched_except_for_critic_only_pi(self):
        with patch(
            "train.ppo.train_ppo_pi.load_train_dataset",
            side_effect=lambda *args, **kwargs: self.source_dataset(),
        ):
            control = build_ppo_pi_dataset("deepmath", pi_mode="none")
            answer = build_ppo_pi_dataset("deepmath", pi_mode="answer")

        self.assertEqual(control["prompt"], answer["prompt"])
        self.assertEqual(control["solution"], answer["solution"])
        self.assertEqual(control["privileged_context"], ["", ""])
        self.assertEqual(
            answer[0]["privileged_context"], PI_ANSWER.format(answer="73")
        )
        # The policy-facing user turn remains the question, with no answer injection.
        self.assertEqual(answer[0]["prompt"][-1]["content"], "What is 40 + 33?")
        self.assertNotIn("73", answer[0]["prompt"][-1]["content"])

    def test_rejects_unknown_pi_mode(self):
        with self.assertRaisesRegex(ValueError, "unknown pi_mode"):
            build_ppo_pi_dataset("deepmath", pi_mode="full")


class PPOPIValueInputTest(unittest.TestCase):
    @staticmethod
    def batch():
        return {
            "prompt_ids": torch.tensor([[0, 1, 2], [3, 4, 5]]),
            "prompt_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "completion_ids": torch.tensor([[11, 12], [21, 22]]),
            "completion_mask": torch.ones((2, 2), dtype=torch.long),
        }

    def trainer_stub(self, pi_mode):
        trainer = object.__new__(PPOPITrainer)
        trainer.pi_mode = pi_mode
        trainer._value_rows = None
        return trainer

    def test_none_is_exact_base_value_input_path(self):
        trainer = self.trainer_stub("none")
        batch = self.batch()

        actual = PPOPITrainer._value_inputs(trainer, batch)
        expected = PPOTrainer._value_inputs(trainer, batch)

        for actual_item, expected_item in zip(actual, expected, strict=True):
            if isinstance(actual_item, torch.Tensor):
                self.assertTrue(torch.equal(actual_item, expected_item))
            else:
                self.assertEqual(actual_item, expected_item)
        self.assertNotIn("value_prompt_ids", batch)

    def test_answer_prompt_is_cached_once_and_survives_without_raw_rows(self):
        trainer = self.trainer_stub("answer")
        rows = [
            {"prompt": format_prompt_math("q0"), "privileged_context": "answer pi 0"},
            {"prompt": format_prompt_math("q1"), "privileged_context": "answer pi 1"},
        ]
        trainer._value_rows = rows
        value_ids = torch.tensor([[0, 31, 32], [41, 42, 43]])
        value_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        trainer._build_value_prompts = Mock(return_value=(value_ids, value_mask))
        batch = self.batch()

        first = PPOPITrainer._value_inputs(trainer, batch)
        trainer._value_rows = None
        second = PPOPITrainer._value_inputs(trainer, batch)

        trainer._build_value_prompts.assert_called_once_with(
            rows, batch["completion_ids"].device
        )
        self.assertIs(batch["value_prompt_ids"], value_ids)
        for first_item, second_item in zip(first, second, strict=True):
            if isinstance(first_item, torch.Tensor):
                self.assertTrue(torch.equal(first_item, second_item))
            else:
                self.assertEqual(first_item, second_item)

    def test_answer_mode_fails_closed_if_cache_or_alignment_is_missing(self):
        trainer = self.trainer_stub("answer")
        with self.assertRaisesRegex(RuntimeError, "value_prompt_ids"):
            PPOPITrainer._value_inputs(trainer, self.batch())

        trainer._value_rows = [
            {"prompt": format_prompt_math("only one"), "privileged_context": "pi"}
        ]
        trainer._build_value_prompts = Mock()
        with self.assertRaisesRegex(RuntimeError, "1.*2"):
            PPOPITrainer._value_inputs(trainer, self.batch())
        trainer._build_value_prompts.assert_not_called()

    def test_answer_mode_rejects_missing_or_empty_pi(self):
        tokenizer = FakeChatTokenizer()
        stub = make_prompt_stub(tokenizer)
        for row in (
            {"prompt": format_prompt_math("q")},
            {"prompt": format_prompt_math("q"), "privileged_context": "  "},
        ):
            with self.subTest(row=row), self.assertRaisesRegex(
                RuntimeError, "privileged"
            ):
                PPOPITrainer._build_value_prompts(stub, [row], torch.device("cpu"))


class PPOPIAlignmentTest(unittest.TestCase):
    def test_only_critic_gets_answer_pi_and_keeps_generation_boundary(self):
        from trl import GRPOTrainer

        tokenizer = FakeChatTokenizer()
        stub = make_prompt_stub(
            tokenizer,
            tools=[],
            chat_template="test-template",
            chat_template_kwargs={"test_option": "sentinel"},
        )
        prompts = [
            format_prompt_math("short question"),
            format_prompt_math("a substantially longer question with several words"),
        ]
        contexts = [
            PI_ANSWER.format(answer="17"),
            PI_ANSWER.format(answer="29"),
        ]

        policy_ids, _, _ = GRPOTrainer._tokenize_prompts(stub, prompts)
        value_ids, value_mask = PPOPITrainer._build_value_prompts(
            stub,
            [
                {"prompt": prompt, "privileged_context": context}
                for prompt, context in zip(prompts, contexts, strict=True)
            ],
            torch.device("cpu"),
        )

        for row, unpadded_policy_ids in enumerate(policy_ids):
            unpadded_value_ids = value_ids[row][value_mask[row].bool()].tolist()
            self.assertEqual(
                list(unpadded_policy_ids[-len(tokenizer.generation_header):]),
                tokenizer.generation_header,
            )
            self.assertEqual(
                unpadded_value_ids[-len(tokenizer.generation_header):],
                tokenizer.generation_header,
            )
            self.assertEqual(value_mask[row].tolist(), sorted(value_mask[row].tolist()))

        policy_call, critic_call = tokenizer.calls
        self.assertEqual(policy_call["conversation"], prompts)
        self.assertNotIn(contexts[0], policy_call["conversation"][0][-1]["content"])
        self.assertEqual(
            critic_call["conversation"][0][-1]["content"],
            prompts[0][-1]["content"] + "\n\n" + contexts[0],
        )
        # Unlike train_ppo_val, the system instruction is left untouched.
        self.assertEqual(
            critic_call["conversation"][0][0], prompts[0][0]
        )
        for call in (policy_call, critic_call):
            self.assertEqual(call["chat_template"], "test-template")
            self.assertTrue(call["add_generation_prompt"])
            self.assertEqual(call["kwargs"]["test_option"], "sentinel")


class PPOPIDiagnosticsTest(unittest.TestCase):
    def test_perfect_predictions_have_unit_explained_variance(self):
        class IdentityAccelerator:
            @staticmethod
            def reduce(tensor, reduction):
                self.assertEqual(reduction, "sum")
                return tensor

        trainer = object.__new__(PPOPITrainer)
        trainer.accelerator = IdentityAccelerator()
        trainer.model = types.SimpleNamespace(training=True)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        outcomes = torch.tensor([[0.0], [1.0]])
        old_values = outcomes.expand(2, 3).clone()
        output = {
            "old_values": old_values,
            "returns": old_values.clone(),
            "terminal_reward": outcomes,
            "completion_mask": torch.ones((2, 3)),
            "scorable": torch.ones((2, 1)),
        }

        PPOPITrainer._log_value_diagnostics(trainer, output)

        metrics = trainer._metrics["train"]
        self.assertEqual(metrics["ppo/value_outcome_mse"], [0.0])
        self.assertEqual(metrics["ppo/value_explained_variance"], [1.0])
        self.assertEqual(metrics["ppo/value_explained_variance_early"], [1.0])
        self.assertEqual(metrics["ppo/value_explained_variance_middle"], [1.0])
        self.assertEqual(metrics["ppo/value_explained_variance_late"], [1.0])
        self.assertEqual(metrics["ppo/raw_advantage_mean"], [0.0])
        self.assertEqual(metrics["ppo/raw_advantage_std"], [0.0])


class PPOPIMetadataTest(unittest.TestCase):
    @staticmethod
    def args(pi_mode):
        return types.SimpleNamespace(
            pi_mode=pi_mode,
            model="Qwen/Qwen3-1.7B",
            dataset="deepmath",
            max_samples=None,
            gamma=1.0,
            lam=1.0,
            vf_coef=0.1,
            cliprange_value=0.2,
            critic_max_grad_norm=1.0,
            loss_type="dapo",
            vllm_mode="colocate",
            optim="paged_adamw_8bit",
            learning_rate=1e-6,
            critic_learning_rate=None,
            seed=42,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_generations=1,
        )

    def test_pi_arms_have_distinct_strict_critic_identity(self):
        control = build_run_meta(self.args("none"), 100)
        answer = build_run_meta(self.args("answer"), 100)

        self.assertEqual(control["method"], answer["method"])
        self.assertNotEqual(control["pi_mode"], answer["pi_mode"])
        self.assertEqual(
            answer["value_prompt_version"], VALUE_PROMPT_VERSIONS["answer"]
        )
        self.assertNotEqual(
            control["value_prompt_version"], answer["value_prompt_version"]
        )


if __name__ == "__main__":
    unittest.main()
