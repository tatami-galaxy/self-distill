import tempfile
import types
import unittest
from collections import defaultdict
from unittest.mock import Mock

import torch

from train.train_ppo import (
    PPOConfig,
    PPOTrainer,
    VALUE_SYSTEM_PROMPT,
    compose_value_messages,
)
from utils import format_prompt_math

TOKENIZER_ID = "Qwen/Qwen3-1.7B"


class FakeChatTokenizer:
    """Minimal chat tokenizer for always-on prompt-boundary tests."""

    pad_token_id = 0
    generation_header = [901, 902]

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        chat_template=None,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        **kwargs,
    ):
        self.calls.append(
            {
                "conversation": conversation,
                "tools": tools,
                "chat_template": chat_template,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "return_dict": return_dict,
                "kwargs": kwargs,
            }
        )
        role_tokens = {"system": 101, "user": 102, "assistant": 103}
        input_ids = []
        for messages in conversation:
            ids = []
            for message in messages:
                ids.append(role_tokens[message["role"]])
                ids.extend(200 + len(word) for word in message["content"].split())
            if add_generation_prompt:
                ids.extend(self.generation_header)
            input_ids.append(ids)
        return {"input_ids": input_ids}


def make_prompt_stub(tokenizer):
    """Attributes read by our prompt builder and TRL's `_tokenize_prompts`.

    Listed EXPLICITLY rather than defaulted via a catch-all `__getattr__`: an
    AttributeError here means TRL's prompt rendering grew a branch, which is exactly the
    drift these tests exist to catch, so it should stop the suite and be read. A stub that
    answered None to anything would swallow it and let the alignment assertions pass
    against a path TRL no longer takes.
    """
    return types.SimpleNamespace(
        processing_class=tokenizer,
        _tokenizer=tokenizer,
        tools=[],
        chat_template="test-template",
        chat_template_kwargs={"test_option": "sentinel"},
        _is_vlm=False,
        # TRL 1.9.0: `_tokenize_prompts` renders each prompt with its own tool schema when
        # environments are configured. None selects the single batched apply_chat_template
        # call -- the path `_build_value_prompts` mirrors, and the one under test.
        environment_factories=None,
        _metrics={"train": defaultdict(list)},
        model=types.SimpleNamespace(training=True),
    )


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

        first = PPOTrainer._value_inputs(stub, batch)

        builder.assert_called_once_with(rows, batch["completion_ids"].device)
        self.assertIs(batch["value_prompt_ids"], value_prompt_ids)
        self.assertIs(batch["value_prompt_mask"], value_prompt_mask)

        # Buffered micro-batches no longer have access to raw rows. The tensors written
        # into the rollout must therefore be sufficient, with no attempt to rebuild.
        stub._value_rows = None
        second = PPOTrainer._value_inputs(stub, batch)

        builder.assert_called_once()
        for first_item, second_item in zip(first, second, strict=True):
            if isinstance(first_item, torch.Tensor):
                self.assertTrue(torch.equal(first_item, second_item))
            else:
                self.assertEqual(first_item, second_item)

    def test_uncached_batch_without_rows_fails_closed(self):
        stub = types.SimpleNamespace(_value_rows=None, _build_value_prompts=Mock())

        with self.assertRaisesRegex(RuntimeError, "value_prompt_ids"):
            PPOTrainer._value_inputs(stub, self.make_batch())

        stub._build_value_prompts.assert_not_called()

    def test_row_completion_count_mismatch_fails_before_building(self):
        stub = types.SimpleNamespace(
            _value_rows=[{"prompt": format_prompt_math("Only one row")}],
            _build_value_prompts=Mock(),
        )

        with self.assertRaisesRegex(RuntimeError, "1.*2"):
            PPOTrainer._value_inputs(stub, self.make_batch())

        stub._build_value_prompts.assert_not_called()


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


class HermeticValuePromptAlignmentTest(unittest.TestCase):
    def test_policy_and_value_prompts_share_boundary_and_left_padding(self):
        from trl import GRPOTrainer

        tokenizer = FakeChatTokenizer()
        stub = make_prompt_stub(tokenizer)
        prompts = [
            format_prompt_math("What is 2 + 2?"),
            format_prompt_math("Explain why the sum of two even integers is even."),
        ]

        policy_ids, _, _ = GRPOTrainer._tokenize_prompts(stub, prompts)
        value_ids, value_mask = PPOTrainer._build_value_prompts(
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

    Both are CROSS-LIBRARY: they hold only while `_build_value_prompts` renders prompts the
    way TRL's GRPOTrainer._tokenize_prompts does. A TRL change to prompt rendering would
    break the critic's token alignment without raising anything, so these assert against
    TRL's real method rather than a reimplementation of it.

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
        """The attributes `_build_value_prompts` and TRL's `_tokenize_prompts` read."""
        return types.SimpleNamespace(
            processing_class=self.tokenizer,
            _tokenizer=self.tokenizer,
            tools=None,
            chat_template=None,
            chat_template_kwargs={},
            _is_vlm=False,
            environment_factories=None,  # see make_prompt_stub
            _metrics={"train": defaultdict(list)},
            model=types.SimpleNamespace(training=True),
        )

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
        ids, mask = PPOTrainer._build_value_prompts(stub, rows, torch.device("cpu"))

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
        input_ids, attention_mask, logits_to_keep = PPOTrainer._value_inputs(stub, batch)

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

        ids, mask = PPOTrainer._build_value_prompts(
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
