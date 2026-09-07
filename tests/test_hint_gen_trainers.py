import tempfile
import unittest
from pathlib import Path

from peft import get_peft_model
from transformers import AutoModelForCausalLM, LlamaConfig

from train.opsd.train_hint_gen import train_constrained_hint_gen, train_hint_gen
from train.opsd.train_hint_gen.lib import (
    lora_config_from_args,
    lora_target_modules,
)


class HintTrainerLoRAConfigTest(unittest.TestCase):
    def test_both_trainers_expose_matching_lora_configuration(self):
        for trainer in (train_hint_gen, train_constrained_hint_gen):
            with self.subTest(trainer=trainer.__name__):
                args = trainer.build_parser().parse_args(
                    [
                        "--use-lora",
                        "--lora-r",
                        "32",
                        "--lora-alpha",
                        "64",
                        "--lora-dropout",
                        "0.05",
                        "--lora-target-modules",
                        "q_proj,v_proj",
                    ]
                )
                trainer.validate_args(args)
                config = lora_config_from_args(args)
                meta = trainer.build_run_meta(args, num_train_examples=10)

                self.assertEqual(config.r, 32)
                self.assertEqual(config.lora_alpha, 64)
                self.assertEqual(config.lora_dropout, 0.05)
                self.assertEqual(config.target_modules, {"q_proj", "v_proj"})
                self.assertEqual(meta["training_mode"], "lora")
                self.assertTrue(meta["use_lora"])
                self.assertEqual(meta["lora_r"], 32)
                self.assertEqual(meta["lora_target_modules"], "q_proj,v_proj")

    def test_full_training_remains_the_default(self):
        for trainer in (train_hint_gen, train_constrained_hint_gen):
            with self.subTest(trainer=trainer.__name__):
                args = trainer.build_parser().parse_args([])
                self.assertIsNone(lora_config_from_args(args))
                meta = trainer.build_run_meta(args, num_train_examples=10)
                self.assertEqual(meta["training_mode"], "full")
                self.assertFalse(meta["use_lora"])
                self.assertIsNone(meta["lora_r"])

    def test_lora_config_saves_adapter_without_full_model_weights(self):
        args = train_hint_gen.build_parser().parse_args(["--use-lora"])
        base_model = AutoModelForCausalLM.from_config(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        )
        model = get_peft_model(base_model, lora_config_from_args(args))

        with tempfile.TemporaryDirectory() as temporary:
            model.save_pretrained(temporary)
            saved = {path.name for path in Path(temporary).iterdir()}

        self.assertIn("adapter_config.json", saved)
        self.assertIn("adapter_model.safetensors", saved)
        self.assertNotIn("model.safetensors", saved)
        self.assertNotIn("pytorch_model.bin", saved)

    def test_target_module_parser_and_invalid_rank(self):
        self.assertEqual(lora_target_modules("all-linear"), "all-linear")
        self.assertEqual(lora_target_modules(" q_proj, v_proj "), ["q_proj", "v_proj"])
        args = train_hint_gen.build_parser().parse_args(["--use-lora", "--lora-r", "0"])
        with self.assertRaisesRegex(ValueError, "lora_r"):
            train_hint_gen.validate_args(args)


if __name__ == "__main__":
    unittest.main()
