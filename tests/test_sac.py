import tempfile
import types
import unittest
from collections import defaultdict
from unittest import mock

import torch
from datasets import Dataset

from train.opsd import train_sdft
from train.opsd.train_sac.lib import (
    ResidualQHead,
    TopKSoftValueEstimator,
    compute_soft_q_lambda_returns,
    make_soft_value_estimator,
    masked_sequence_mean,
    terminal_token_rewards,
)
from train.opsd.train_sac.trainer import SACConfig, SACTrainer
from train.opsd.train_sac.train_sac import sac_run_name


class ResidualQHeadTest(unittest.TestCase):
    def test_zero_initialization_preserves_teacher_q_and_can_learn(self):
        head = ResidualQHead(hidden_size=3)
        hidden = torch.tensor([[[1.0, 2.0, 3.0]]])
        action = torch.tensor([[[2.0, -1.0, 0.5]]])

        correction = (head(hidden) * action).sum()
        self.assertEqual(correction.item(), 0.0)

        (correction - 1.0).square().backward()
        self.assertGreater(head.projection.weight.grad.abs().sum().item(), 0.0)


class TopKSoftValueTest(unittest.TestCase):
    def test_renormalizes_topk_but_keeps_full_policy_logps_in_soft_value(self):
        estimator = TopKSoftValueEstimator(k=2)
        logits = torch.tensor([[[2.0, 1.0, 0.0]]])
        ids = estimator.select_token_ids(logits)
        full_logps = torch.log_softmax(logits, dim=-1).gather(-1, ids)
        support = estimator.build_support(ids, full_logps)
        q = torch.tensor([[[0.5, -0.25]]])

        value = estimator.estimate(q, support)
        expected_weights = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
        expected = (
            expected_weights
            * (q[0, 0] - torch.log_softmax(logits[0, 0], dim=0)[:2])
        ).sum()

        self.assertTrue(torch.allclose(value[0, 0], expected))
        self.assertTrue(torch.allclose(support.weights[0, 0], expected_weights))
        self.assertLess(support.mass.item(), 1.0)

    def test_sarsa_dispatch_is_reserved_but_not_silently_substituted(self):
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            make_soft_value_estimator("sarsa", topk=100)


class SoftLambdaReturnTest(unittest.TestCase):
    def setUp(self):
        self.mask = torch.ones((1, 3))
        self.rewards = torch.tensor([[0.0, 0.0, 1.0]])
        self.values = torch.tensor([[10.0, 20.0, 30.0]])
        self.logps = torch.tensor([[-0.1, -0.2, -0.3]])

    def test_lambda_zero_is_one_step_expected_sarsa(self):
        returns = compute_soft_q_lambda_returns(
            self.rewards, self.values, self.logps, self.mask, lam=0.0
        )
        self.assertTrue(torch.allclose(returns, torch.tensor([[20.0, 30.0, 1.0]])))

    def test_lambda_one_is_sampled_soft_monte_carlo_return(self):
        returns = compute_soft_q_lambda_returns(
            self.rewards, self.values, self.logps, self.mask, lam=1.0
        )
        self.assertTrue(torch.allclose(returns, torch.tensor([[1.5, 1.3, 1.0]])))

    def test_padding_terminates_bootstrap(self):
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        rewards = terminal_token_rewards(torch.tensor([1.0]), mask)
        returns = compute_soft_q_lambda_returns(
            rewards, self.values, self.logps, mask, lam=0.0
        )
        self.assertTrue(torch.equal(rewards, torch.tensor([[0.0, 1.0, 0.0]])))
        self.assertTrue(torch.equal(returns, torch.tensor([[20.0, 1.0, 0.0]])))


class ReductionTest(unittest.TestCase):
    def test_sequence_uniform_mean_excludes_unscorable_rows(self):
        values = torch.tensor([[1.0, 3.0, 0.0], [100.0, 100.0, 100.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual(masked_sequence_mean(values, mask).item(), 2.0)


class SACConfigTest(unittest.TestCase):
    @staticmethod
    def make_config(output_dir, **overrides):
        kwargs = {
            "output_dir": output_dir,
            "teacher_model_kind": "base",
            "num_generations": 1,
            "num_iterations": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "steps_per_generation": 2,
            "report_to": "none",
            "use_cpu": True,
            "use_vllm": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": None,
            "repetition_penalty": 1.0,
            "generation_kwargs": None,
        }
        kwargs.update(overrides)
        return SACConfig(**kwargs)

    def test_accepts_minimal_online_topk_configuration(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = self.make_config(output_dir)
        self.assertEqual(config.soft_v_estimator, "topk")
        self.assertEqual(config.soft_v_topk, 100)
        self.assertEqual(config.lam, 0.0)
        self.assertEqual(config.num_iterations, 1)

    def test_run_name_records_linear_q_head(self):
        self.assertEqual(
            sac_run_name("deepmath", "full", "topk"),
            "deepmath_full_topk_linear",
        )

    def test_rejects_rollout_distribution_mismatch(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "untruncated temperature-1"):
                self.make_config(output_dir, top_p=0.95)

    def test_rejects_reused_rollout_updates(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "num_iterations=1"):
                self.make_config(output_dir, num_iterations=2)


class SACDatasetTest(unittest.TestCase):
    def test_optional_reward_solution_preserves_sdft_default(self):
        source = Dataset.from_list([
            {"question": "q", "final_answer": "7", "solution": "worked"}
        ])
        with mock.patch.object(train_sdft, "load_train_dataset", return_value=source):
            ordinary = train_sdft.build_sdft_dataset("answer")
            sac = train_sdft.build_sdft_dataset("answer", include_reward_solution=True)

        self.assertEqual(ordinary.column_names, ["prompt", "privileged_context"])
        self.assertEqual(
            set(sac.column_names), {"prompt", "privileged_context", "solution"}
        )
        self.assertEqual(sac[0]["solution"], r"\boxed{7}")


class JointSACLossTest(unittest.TestCase):
    def test_one_loss_backpropagates_to_actor_and_residual_q(self):
        trainer = object.__new__(SACTrainer)
        trainer.args = types.SimpleNamespace(lam=0.0)
        trainer.current_gradient_accumulation_steps = 1
        trainer.soft_value_estimator = TopKSoftValueEstimator(k=2)
        trainer.q_head = ResidualQHead(hidden_size=2)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer.accelerator = types.SimpleNamespace(gather=lambda tensor: tensor)

        student_logits = torch.tensor(
            [[[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], requires_grad=True
        )
        teacher_logps = torch.log_softmax(
            torch.tensor([[[1.0, 2.0, 0.0], [1.0, 0.0, 2.0]]]), dim=-1
        )
        action_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        completion_ids = torch.tensor([[0, 1]])

        trainer._forward_logits = lambda *args, **kwargs: student_logits

        def teacher_scores(_inputs, topk_ids):
            return {
                "hidden": torch.tensor([[[1.0, 2.0], [2.0, 1.0]]]),
                "sampled_action_vectors": action_vectors[completion_ids],
                "sampled_teacher_logps": teacher_logps.gather(
                    -1, completion_ids.unsqueeze(-1)
                ).squeeze(-1),
                "topk_teacher_logps": teacher_logps.gather(-1, topk_ids),
                "topk_corrections": torch.zeros_like(topk_ids, dtype=torch.float32),
            }

        trainer._teacher_scores_and_q_inputs = teacher_scores
        inputs = {
            "prompt_ids": torch.tensor([[9]]),
            "prompt_mask": torch.ones((1, 1), dtype=torch.long),
            "completion_ids": completion_ids,
            "completion_mask": torch.ones((1, 2), dtype=torch.long),
            "teacher_input_ids": torch.tensor([[8, 0, 1]]),
            "teacher_attention_mask": torch.ones((1, 3), dtype=torch.long),
            "terminal_rewards": torch.tensor([1.0]),
            "scorable": torch.tensor([1.0]),
        }
        model = types.SimpleNamespace(training=True)

        loss = SACTrainer.compute_loss(trainer, model, inputs)
        loss.backward()

        self.assertGreater(student_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            trainer.q_head.projection.weight.grad.abs().sum().item(), 0.0
        )
        self.assertEqual(trainer._metrics["train"]["sac/actor_loss"].__len__(), 1)
        self.assertEqual(trainer._metrics["train"]["sac/q_loss"].__len__(), 1)


if __name__ == "__main__":
    unittest.main()
