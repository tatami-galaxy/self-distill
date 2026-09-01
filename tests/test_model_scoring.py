"""Tests for the shared causal-LM token-scoring helpers."""

import math
import types
import unittest

import torch

from utils.model_scoring import (
    _selective_logps_entropy_fp32,
    _selective_logps_fp32,
    per_token_logps,
    per_token_stats,
)


class ModelScoringTest(unittest.TestCase):
    class RecordingModel:
        def __init__(self, vocab=16):
            self.vocab = vocab
            self.calls = []

        def __call__(self, input_ids, logits_to_keep, use_cache):
            self.calls.append({"input_ids": input_ids, "logits_to_keep": logits_to_keep})
            logits = torch.zeros(
                input_ids.size(0), logits_to_keep, self.vocab, dtype=torch.bfloat16
            )
            return types.SimpleNamespace(logits=logits)

    def test_physical_batches_larger_than_one_are_rejected(self):
        model = self.RecordingModel()
        with self.assertRaisesRegex(ValueError, "physical batch size 1"):
            per_token_logps(
                model,
                torch.tensor([[5, 6, 7], [3, 4, 5]]),
                torch.tensor([[6, 7], [4, 5]]),
            )
        self.assertEqual(model.calls, [])

    def test_logps_are_float32_and_only_completion_logits_are_requested(self):
        model = self.RecordingModel()
        logps = per_token_logps(
            model, torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3]])
        )
        self.assertEqual(logps.dtype, torch.float32)
        self.assertEqual(model.calls[0]["logits_to_keep"], 3)

    def test_fp32_logps_match_reference_log_softmax(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 11)
        index = torch.randint(0, 11, (2, 5))
        expected = torch.log_softmax(logits, dim=-1).gather(
            -1, index.unsqueeze(-1)
        ).squeeze(-1)
        actual = _selective_logps_fp32(logits, index, chunk_size=3)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_fp32_logps_and_entropy_match_reference(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 5, 11, dtype=torch.bfloat16)
        index = torch.randint(0, 11, (2, 5))
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        expected_logps = log_probs.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        expected_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

        actual = _selective_logps_entropy_fp32(logits, index, chunk_size=3)
        self.assertEqual(actual.logps.dtype, torch.float32)
        self.assertEqual(actual.entropy.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual.logps, expected_logps, atol=1e-6))
        self.assertTrue(torch.allclose(actual.entropy, expected_entropy, atol=1e-6))

    def test_entropy_is_invariant_to_constant_logit_shift(self):
        logits = torch.randn(1, 4, 9)
        index = torch.tensor([[0, 1, 2, 3]])
        original = _selective_logps_entropy_fp32(logits, index, chunk_size=2)
        shifted = _selective_logps_entropy_fp32(logits + 123.0, index, chunk_size=2)
        self.assertTrue(torch.allclose(original.logps, shifted.logps, atol=1e-5))
        self.assertTrue(torch.allclose(original.entropy, shifted.entropy, atol=1e-5))

    def test_per_token_stats_reuses_batch_one_alignment(self):
        model = self.RecordingModel(vocab=8)
        stats = per_token_stats(
            model,
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[2, 3]]),
            entropy_chunk_size=1,
        )
        self.assertEqual(model.calls[0]["logits_to_keep"], 3)
        self.assertEqual(stats.logps.shape, (1, 2))
        self.assertEqual(stats.entropy.shape, (1, 2))
        self.assertTrue(torch.allclose(stats.entropy, torch.full((1, 2), math.log(8))))


if __name__ == "__main__":
    unittest.main()
