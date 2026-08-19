import tempfile
import types
import unittest
from collections import defaultdict
from unittest.mock import patch

import torch
from trl import GRPOTrainer

from train.ppo.train_ppo import PPOConfig, PPOTrainer


class PPOConfigTest(unittest.TestCase):
    def make_config(self, output_dir, num_generations):
        return PPOConfig(
            output_dir=output_dir,
            num_generations=num_generations,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            report_to="none",
            use_cpu=True,
        )

    def test_rejects_zero_generations(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(
                ValueError, "PPO requires num_generations to be at least 1"
            ):
                self.make_config(output_dir, num_generations=0)

    def test_rejects_negative_generations(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(
                ValueError, "PPO requires num_generations to be at least 1"
            ):
                self.make_config(output_dir, num_generations=-1)

    def test_forces_reference_free_objective(self):
        # __post_init__ does work before super() now (the num_generations dance), so guard
        # that it still forces the reference-free / no-group-normalization objective that
        # makes PPO-vs-GRPO isolate the critic. Checked at num_generations=1, the new path.
        with tempfile.TemporaryDirectory() as output_dir:
            config = self.make_config(output_dir, num_generations=1)

        self.assertEqual(config.beta, 0.0)
        self.assertEqual(config.scale_rewards, "none")

    def test_accepts_one_generation(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = self.make_config(output_dir, num_generations=1)

        self.assertEqual(config.num_generations, 1)
        self.assertEqual(config.generation_batch_size, 16)

    def test_preserves_two_generations(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = self.make_config(output_dir, num_generations=2)

        self.assertEqual(config.num_generations, 2)
        self.assertEqual(config.generation_batch_size, 16)

    def test_rejects_negative_critic_warmup(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "critic_warmup_steps must be >= 0"):
                PPOConfig(
                    output_dir=output_dir,
                    critic_warmup_steps=-1,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=16,
                    report_to="none",
                    use_cpu=True,
                )


class CriticWarmupTest(unittest.TestCase):
    """The warmup window, and the guarantee that it freezes the policy but not the critic."""

    @staticmethod
    def stub(critic_warmup_steps, global_step, training=True):
        trainer = object.__new__(PPOTrainer)
        trainer.args = types.SimpleNamespace(
            critic_warmup_steps=critic_warmup_steps,
            cliprange_value=0.2,
            vf_coef=0.5,
        )
        trainer.state = types.SimpleNamespace(global_step=global_step)
        trainer.model = types.SimpleNamespace(training=training)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer.accelerator = types.SimpleNamespace(
            gather=lambda tensor: tensor, num_processes=1
        )
        trainer.loss_type = "bnpo"
        trainer.current_gradient_accumulation_steps = 1
        return trainer

    def test_window_covers_exactly_the_requested_optimizer_steps(self):
        # global_step counts COMPLETED steps, so step k is in warmup iff k < N.
        for global_step, expected in ((0, True), (19, True), (20, False), (21, False)):
            with self.subTest(global_step=global_step):
                trainer = self.stub(20, global_step)
                self.assertEqual(trainer.in_critic_warmup, expected)

    def test_disabled_by_default_and_never_active_in_eval(self):
        self.assertFalse(self.stub(0, 0).in_critic_warmup)
        self.assertFalse(self.stub(20, 0, training=False).in_critic_warmup)

    def run_loss(self, trainer):
        """Drive _compute_loss with a stubbed policy loss and a stubbed critic forward.

        Both halves hang off their own leaf tensor, so the gradients afterwards say exactly
        which of the two the loss actually trained.
        """
        policy_leaf = torch.ones(1, requires_grad=True)
        vpred = torch.zeros((2, 3), requires_grad=True)
        trainer._value_inputs = lambda batch: (vpred,)
        trainer._get_per_token_values = lambda values: values
        inputs = {
            "completion_mask": torch.ones((2, 3)),
            "old_values": torch.zeros((2, 3)),
            "returns": torch.ones((2, 3)),
        }
        with patch.object(GRPOTrainer, "_compute_loss", lambda *_: policy_leaf.sum()):
            PPOTrainer._compute_loss(trainer, None, inputs).backward()
        return policy_leaf, vpred

    def test_warmup_zeroes_the_policy_gradient_but_still_trains_the_critic(self):
        policy_leaf, vpred = self.run_loss(self.stub(20, global_step=0))

        self.assertEqual(policy_leaf.grad.abs().sum().item(), 0.0)
        self.assertGreater(vpred.grad.abs().sum().item(), 0.0)

    def test_after_warmup_both_receive_gradient(self):
        policy_leaf, vpred = self.run_loss(self.stub(20, global_step=20))

        self.assertGreater(policy_leaf.grad.abs().sum().item(), 0.0)
        self.assertGreater(vpred.grad.abs().sum().item(), 0.0)

    def test_warmup_state_is_logged_every_step(self):
        for global_step, expected in ((0, 1.0), (20, 0.0)):
            with self.subTest(global_step=global_step):
                trainer = self.stub(20, global_step)
                self.run_loss(trainer)
                self.assertEqual(
                    trainer._metrics["train"]["ppo/critic_warmup"], [expected]
                )


class PerTokenValueAlignmentTest(unittest.TestCase):
    def test_value_slice_scores_the_state_before_each_completion_token(self):
        class TokenEchoBackbone(torch.nn.Module):
            def forward(self, input_ids, attention_mask, use_cache):
                return types.SimpleNamespace(
                    last_hidden_state=input_ids.float().unsqueeze(-1)
                )

        class IdentityScore(torch.nn.Module):
            def forward(self, hidden):
                return hidden

        value_model = types.SimpleNamespace(
            base_model_prefix="backbone",
            backbone=TokenEchoBackbone(),
            score=IdentityScore(),
        )
        stub = types.SimpleNamespace(value_model=value_model)

        # Prompt width P=2, completion width C=3. Since the fake backbone echoes each
        # input token, values aligned to completion actions must be the preceding tokens
        # at absolute positions P-1 through P+C-2.
        input_ids = torch.tensor(
            [
                [91, 92, 11, 12, 13],
                [0, 93, 21, 22, 0],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ]
        )

        values = PPOTrainer._get_per_token_values(
            stub, input_ids, attention_mask, logits_to_keep=3
        )

        expected = torch.tensor(
            [
                [92.0, 11.0, 12.0],
                [93.0, 21.0, 22.0],
            ]
        )
        self.assertTrue(torch.equal(values, expected))


if __name__ == "__main__":
    unittest.main()
