import json
import tempfile
import unittest
from pathlib import Path

from utils.model_adapters import resolve_model_adapter, vllm_model_and_adapter


class ModelAdapterTest(unittest.TestCase):
    def test_full_model_passes_through_without_a_lora_request(self):
        kwargs, request, spec = vllm_model_and_adapter("Qwen/Qwen3-1.7B")

        self.assertEqual(kwargs, {"model": "Qwen/Qwen3-1.7B"})
        self.assertIsNone(request)
        self.assertFalse(spec.is_adapter)

    def test_local_lora_resolves_base_rank_and_vllm_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "checkpoint-10"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                        "r": 24,
                        "rank_pattern": {"model.layers.0.self_attn.q_proj": 32},
                    }
                )
            )

            kwargs, request, spec = vllm_model_and_adapter(
                str(adapter), adapter_name="checkpoint-10", adapter_id=7
            )

        self.assertTrue(spec.is_adapter)
        self.assertEqual(spec.base_model, "Qwen/Qwen3-1.7B")
        self.assertEqual(spec.rank, 32)
        self.assertEqual(kwargs["model"], "Qwen/Qwen3-1.7B")
        self.assertTrue(kwargs["enable_lora"])
        self.assertEqual(kwargs["max_lora_rank"], 32)
        self.assertEqual(request.lora_name, "checkpoint-10")
        self.assertEqual(request.lora_int_id, 7)
        self.assertEqual(request.lora_path, spec.adapter_path)
        self.assertEqual(request.base_model_name, "Qwen/Qwen3-1.7B")

    def test_non_lora_peft_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "IA3",
                        "base_model_name_or_path": "base",
                        "r": 8,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "only LoRA"):
                resolve_model_adapter(str(adapter))

    def test_missing_base_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": 8})
            )
            with self.assertRaisesRegex(ValueError, "base_model_name_or_path"):
                resolve_model_adapter(str(adapter))


if __name__ == "__main__":
    unittest.main()
