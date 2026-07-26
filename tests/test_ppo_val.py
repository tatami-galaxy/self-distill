"""Verifier-prompt critic: composition, caching, and the cross-library alignment invariants.

`PPOValTrainer` differs from `PPOTrainer` in exactly one thing -- the critic reads the question
under a verifier instruction instead of the policy's -- so these are the tests that hold that
difference to being the ONLY one. The alignment cases assert against TRL's real
`_tokenize_prompts`, not a reimplementation of it: a TRL change to prompt rendering would break
the critic's token alignment without raising anything.
"""

import types
import unittest
from unittest.mock import Mock

import torch

from tests.helpers import TOKENIZER_ID, FakeChatTokenizer, make_prompt_stub
from train.train_ppo_val import (
    PPOValTrainer,
    VALUE_SYSTEM_PROMPT,
    compose_value_messages,
)
from utils import format_prompt_math


class ValuePromptTest(unittest.TestCase):
    def test_replaces_policy_system_prompt_and_preserves_question(self):
        policy_messages = [
            {"role": "system", "content": "Solve the problem carefully."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
        original_messages = [dict(message) for message in policy_messages]

        value_messages = compose_value_messages(policy_messages)

        self.assertEqual(
            value_messages[0],
            {"role": "system", "content": VALUE_SYSTEM_PROMPT},
        )
        self.assertEqual(
            value_messages[1],
            {"role": "user", "content": "What is 2 + 2?"},
        )
        self.assertEqual(len(value_messages), 2)
        self.assertEqual(policy_messages, original_messages)


class ValuePromptCacheTest(unittest.TestCase):
    def make_batch(self):
        return {
            "completion_ids": torch.tensor([[11, 12], [21, 22]]),
            "completion_mask": torch.ones((2, 2), dtype=torch.long),
        }

    def test_builds_cache_once_then_reuses_it_after_rows_are_cleared(self):
        rows = [
            {"prompt": format_prompt_math("What is 2 + 2?")},
            {"prompt": format_prompt_math("What is 3 + 3?")},
        ]
        value_prompt_ids = torch.tensor([[0, 31, 32], [41, 42, 43]])
        value_prompt_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        builder = Mock(return_value=(value_prompt_ids, value_prompt_mask))
        stub = types.SimpleNamespace(_value_rows=rows, _build_value_prompts=builder)
        batch = self.make_batch()

        first = PPOValTrainer._value_inputs(stub, batch)

        builder.assert_called_once_with(rows, batch["completion_ids"].device)
        self.assertIs(batch["value_prompt_ids"], value_prompt_ids)
        self.assertIs(batch["value_prompt_mask"], value_prompt_mask)

        # Buffered micro-batches no longer have access to raw rows. The tensors written
        # into the rollout must therefore be sufficient, with no attempt to rebuild.
        stub._value_rows = None
        second = PPOValTrainer._value_inputs(stub, batch)

        builder.assert_called_once()
        for first_item, second_item in zip(first, second, strict=True):
            if isinstance(first_item, torch.Tensor):
                self.assertTrue(torch.equal(first_item, second_item))
            else:
                self.assertEqual(first_item, second_item)

    def test_uncached_batch_without_rows_fails_closed(self):
        stub = types.SimpleNamespace(_value_rows=None, _build_value_prompts=Mock())

        with self.assertRaisesRegex(RuntimeError, "value_prompt_ids"):
            PPOValTrainer._value_inputs(stub, self.make_batch())

        stub._build_value_prompts.assert_not_called()

    def test_row_completion_count_mismatch_fails_before_building(self):
        stub = types.SimpleNamespace(
            _value_rows=[{"prompt": format_prompt_math("Only one row")}],
            _build_value_prompts=Mock(),
        )

        with self.assertRaisesRegex(RuntimeError, "1.*2"):
            PPOValTrainer._value_inputs(stub, self.make_batch())

        stub._build_value_prompts.assert_not_called()


class HermeticValuePromptAlignmentTest(unittest.TestCase):
    def test_policy_and_value_prompts_share_boundary_and_left_padding(self):
        from trl import GRPOTrainer

        tokenizer = FakeChatTokenizer()
        # Sentinel values for the three rendering arguments the critic's prompt must carry over
        # from TRL's path, so the assertions below distinguish "forwarded" from "defaulted".
        stub = make_prompt_stub(
            tokenizer,
            tools=[],
            chat_template="test-template",
            chat_template_kwargs={"test_option": "sentinel"},
        )
        prompts = [
            format_prompt_math("What is 2 + 2?"),
            format_prompt_math("Explain why the sum of two even integers is even."),
        ]

        policy_ids, _, _ = GRPOTrainer._tokenize_prompts(stub, prompts)
        value_ids, value_mask = PPOValTrainer._build_value_prompts(
            stub,
            [{"prompt": prompt} for prompt in prompts],
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

        policy_call, value_call = tokenizer.calls
        for call in (policy_call, value_call):
            self.assertIsNone(call["tools"])
            self.assertEqual(call["chat_template"], "test-template")
            self.assertTrue(call["add_generation_prompt"])
            self.assertTrue(call["tokenize"])
            self.assertTrue(call["return_dict"])
            self.assertEqual(call["kwargs"]["test_option"], "sentinel")
        self.assertEqual(
            value_call["conversation"][0][0],
            {"role": "system", "content": VALUE_SYSTEM_PROMPT},
        )


class ValuePromptAlignmentTest(unittest.TestCase):
    """Pin the two invariants the per-token critic silently depends on.

    Both are CROSS-LIBRARY: they hold only while the base class's `_render_value_prompts`
    renders prompts the way TRL's GRPOTrainer._tokenize_prompts does. A TRL change to prompt
    rendering would break the critic's token alignment without raising anything, so these
    assert against TRL's real method rather than a reimplementation of it.

    Tokenizer only -- no model weights are loaded.
    """

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_ID, trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:  # no local cache / no network
            raise unittest.SkipTest(f"tokenizer {TOKENIZER_ID} unavailable: {exc}")

    def make_stub(self):
        return make_prompt_stub(self.tokenizer)

    def test_value_prompt_is_contiguous_with_the_completion(self):
        # NB: `cat([prompt, completion])[:, -C:] == completion` is a tautology of the concat and
        # cannot fail, so it is not the invariant worth asserting. What can actually break is the
        # PADDING SIDE: right-padding the value prompt leaves pad tokens BETWEEN the prompt and
        # the completion. The completion would still be the last C columns, but `values[:, 0]`
        # (which `_get_per_token_values` reads from the position just before the first completion
        # token) would be taken off a PAD position instead of the generation header, and every
        # row's value curve would be offset by its own pad count.
        rows = [
            {"prompt": format_prompt_math("What is 2 + 2?")},
            {"prompt": format_prompt_math("Let x satisfy " + "x^2 - 5x + 6 = 0. " * 40)},
        ]
        stub = self.make_stub()
        ids, mask = PPOValTrainer._build_value_prompts(stub, rows, torch.device("cpu"))

        self.assertEqual(ids.shape, mask.shape)
        self.assertGreater(int((mask[0] == 0).sum()), 0, "ragged rows should force padding")

        # Left-padded ⇒ within each row every 0 precedes every 1 (mask is non-decreasing).
        for row in range(mask.size(0)):
            row_mask = mask[row].tolist()
            self.assertEqual(
                row_mask, sorted(row_mask),
                "value prompt must be LEFT-padded so its real tokens end flush against the "
                "completion",
            )

        completion_ids = torch.tensor([[11, 12, 13, 14], [21, 22, 0, 0]])
        completion_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        batch = {
            "value_prompt_ids": ids,
            "value_prompt_mask": mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        }
        input_ids, attention_mask, logits_to_keep = PPOValTrainer._value_inputs(stub, batch)

        n_completion = completion_ids.size(1)
        self.assertEqual(logits_to_keep, n_completion)
        self.assertTrue(torch.equal(input_ids[:, -n_completion:], completion_ids))
        self.assertTrue(torch.equal(attention_mask[:, -n_completion:], completion_mask))

        # The load-bearing assertion: the position feeding values[:, 0] is a REAL prompt token
        # (the generation header) for every row, not padding.
        self.assertTrue(
            torch.equal(
                attention_mask[:, -n_completion - 1],
                torch.ones(attention_mask.size(0), dtype=attention_mask.dtype),
            ),
            "padding must not sit between the value prompt and the completion",
        )

    def test_value_prompt_ends_on_the_policy_generation_header(self):
        question = "What is the sum of the roots of x^2 - 5x + 6 = 0?"
        policy_messages = format_prompt_math(question)
        stub = self.make_stub()

        # The generation header is whatever add_generation_prompt appends; derive it rather
        # than hard-coding tokens, so this stays tokenizer-agnostic.
        with_header = self.tokenizer.apply_chat_template(
            [policy_messages], add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"][0]
        without_header = self.tokenizer.apply_chat_template(
            [policy_messages], add_generation_prompt=False, tokenize=True, return_dict=True
        )["input_ids"][0]
        header = with_header[len(without_header):]
        self.assertGreater(len(header), 0, "chat template appends no generation header")

        # TRL's real tokenizer path for the policy prompt -- the thing we must not drift from.
        from trl import GRPOTrainer

        policy_ids, _, _ = GRPOTrainer._tokenize_prompts(stub, [policy_messages])
        self.assertEqual(list(policy_ids[0][-len(header):]), list(header))

        ids, mask = PPOValTrainer._build_value_prompts(
            stub, [{"prompt": policy_messages}], torch.device("cpu")
        )
        value_ids = ids[0][mask[0].bool()].tolist()  # strip left padding
        self.assertEqual(value_ids[-len(header):], list(header))

        # Same boundary, different framing: the verifier system turn made it longer.
        self.assertGreater(len(value_ids), len(policy_ids[0]))
        self.assertIn(
            VALUE_SYSTEM_PROMPT[:40],
            self.tokenizer.decode(value_ids),
        )


if __name__ == "__main__":
    unittest.main()
