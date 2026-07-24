import tempfile
import unittest

from train.train_ppo import PPOConfig


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


if __name__ == "__main__":
    unittest.main()
