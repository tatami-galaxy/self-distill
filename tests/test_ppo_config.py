import tempfile
import types
import unittest

import torch

from train.train_ppo import PPOConfig, PPOTrainer


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
