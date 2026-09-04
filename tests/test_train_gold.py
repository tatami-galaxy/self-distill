import types
import unittest

from trl.experimental.sdft import SDFTConfig

from train.opd import train_gold


class GoldDefaultsTest(unittest.TestCase):
    def test_generation_defaults_match_sdft(self):
        fields = SDFTConfig.__dataclass_fields__

        self.assertEqual(train_gold.DEFAULT_TEMPERATURE, fields["temperature"].default)
        self.assertEqual(train_gold.DEFAULT_TOP_P, fields["top_p"].default)
        self.assertEqual(train_gold.DEFAULT_TOP_K, fields["top_k"].default)

    def test_generation_distribution_is_recorded_for_resume(self):
        args = types.SimpleNamespace(
            model="student",
            teacher_model="teacher",
            dataset="deepmath",
            max_samples=128,
            lmbda=1.0,
            beta=1.0,
            temperature=train_gold.DEFAULT_TEMPERATURE,
            top_p=train_gold.DEFAULT_TOP_P,
            top_k=train_gold.DEFAULT_TOP_K,
            max_completion_length=8192,
            learning_rate=1e-5,
            seed=42,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_generations=1,
        )

        meta = train_gold.build_run_meta(args, num_train_examples=100)

        self.assertEqual(meta["temperature"], 1.0)
        self.assertEqual(meta["top_p"], 1.0)
        self.assertEqual(meta["top_k"], 0)
        self.assertTrue(
            {"temperature", "top_p", "top_k"}.issubset(
                train_gold.GOLD_RESUME_STRICT_KEYS
            )
        )


if __name__ == "__main__":
    unittest.main()
