import types
import unittest
from collections import defaultdict
from unittest import mock

import torch
from datasets import Dataset

from train.opsd import train_sdft_self_rollout as self_rollout
from utils import compose_pi_messages


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    pieces = {
        11: "alpha",
        12: " beta",
        21: "gamma",
        22: " delta",
    }

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[token_id] for token_id in ids)


class FakeProcessing:
    def __init__(self, fixed_length=None):
        self.fixed_length = fixed_length
        self.teacher_prompts = None

    def apply_chat_template(self, conversation, **_kwargs):
        self.teacher_prompts = conversation
        lengths = [
            self.fixed_length
            if self.fixed_length is not None
            else len(messages[-1]["content"].split())
            for messages in conversation
        ]
        return {"input_ids": [list(range(length)) for length in lengths]}


class CapturingDelegate:
    def __init__(self):
        self.contexts = None
        self.completion_ids = None
        self.completion_mask = None

    @staticmethod
    def _compose_teacher_prompt(prompt, context):
        return compose_pi_messages(prompt, context)

    def build(self, prompts, contexts, completion_ids, completion_mask):
        self.contexts = contexts
        self.completion_ids = completion_ids
        self.completion_mask = completion_mask
        return {"built": True}


def make_builder(*, max_prompt_length=1000, context_window=4096, fixed_length=None):
    delegate = CapturingDelegate()
    trainer = types.SimpleNamespace(
        _tokenizer=FakeTokenizer(),
        processing_class=FakeProcessing(fixed_length=fixed_length),
        chat_template_kwargs={},
        max_prompt_length=max_prompt_length,
        model=types.SimpleNamespace(
            training=True,
            config=types.SimpleNamespace(max_position_embeddings=context_window),
        ),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )
    return self_rollout.SelfRolloutTeacherContextBuilder(trainer, delegate), delegate


class SelfRolloutPromptTest(unittest.TestCase):

    def test_each_completion_is_used_as_its_own_pi_and_prefilled_target(self):
        builder, delegate = make_builder()
        prompts = [
            [{"role": "user", "content": "question zero"}],
            [{"role": "user", "content": "question one"}],
        ]
        completion_ids = torch.tensor([
            [11, 12, 2, 0],
            [21, 22, 0, 0],
        ])
        completion_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ])

        result = builder.build(
            prompts,
            [self_rollout.SELF_ROLLOUT_PLACEHOLDER] * 2,
            completion_ids,
            completion_mask,
        )

        self.assertEqual(result, {"built": True})
        self.assertIn("alpha beta", delegate.contexts[0])
        self.assertNotIn("gamma delta", delegate.contexts[0])
        self.assertIn("gamma delta", delegate.contexts[1])
        self.assertNotIn("alpha beta", delegate.contexts[1])
        self.assertIn("may or may not be correct", delegate.contexts[0])
        # The original generated IDs remain the target appended by TRL's delegate.
        self.assertIs(delegate.completion_ids, completion_ids)
        self.assertIs(delegate.completion_mask, completion_mask)

    def test_terminal_eos_and_padding_are_not_inserted_into_pi(self):
        builder, delegate = make_builder()
        builder.build(
            [[{"role": "user", "content": "q"}]],
            [self_rollout.SELF_ROLLOUT_PLACEHOLDER],
            torch.tensor([[11, 12, 2, 0]]),
            torch.tensor([[1, 1, 1, 0]]),
        )
        self.assertIn("alpha beta", delegate.contexts[0])

    def test_online_prompt_overflow_raises_instead_of_left_truncating(self):
        builder, _ = make_builder(max_prompt_length=10, fixed_length=20)
        with self.assertRaisesRegex(ValueError, "would be left-truncated"):
            builder.build(
                [[{"role": "user", "content": "q"}]],
                [self_rollout.SELF_ROLLOUT_PLACEHOLDER],
                torch.tensor([[11, 12]]),
                torch.tensor([[1, 1]]),
            )

    def test_combined_teacher_input_must_fit_model_context(self):
        builder, _ = make_builder(
            max_prompt_length=20,
            context_window=10,
            fixed_length=8,
        )
        with self.assertRaisesRegex(ValueError, "exceeds the model context window"):
            builder.build(
                [[{"role": "user", "content": "q"}]],
                [self_rollout.SELF_ROLLOUT_PLACEHOLDER],
                torch.tensor([[11, 12, 0, 0]]),
                torch.tensor([[1, 1, 0, 0]]),
            )

    def test_teacher_conditioned_generation_is_circular(self):
        builder, _ = make_builder()
        with self.assertRaisesRegex(ValueError, "does not exist until after generation"):
            builder.select_generation_prompts([], [])


class SelfRolloutDatasetTest(unittest.TestCase):
    def test_dataset_contains_only_prompt_and_online_placeholder(self):
        source = Dataset.from_dict({"question": ["q0", "q1"]})
        with mock.patch.object(
            self_rollout, "load_train_dataset", return_value=source
        ) as load:
            ds = self_rollout.build_train_dataset("deepmath", max_samples=2)

        load.assert_called_once_with("deepmath", max_samples=2)
        self.assertEqual(ds.column_names, ["prompt", "privileged_context"])
        self.assertEqual(
            ds["privileged_context"],
            [self_rollout.SELF_ROLLOUT_PLACEHOLDER] * 2,
        )
        self.assertEqual(
            [row["prompt"][-1]["content"] for row in ds],
            ["q0", "q1"],
        )


class SelfRolloutRunMetadataTest(unittest.TestCase):
    def test_online_same_completion_provenance_is_recorded(self):
        args = types.SimpleNamespace(
            model="student",
            dataset="deepmath",
            max_samples=128,
            distillation_mode="sampled_token",
            distillation_alpha=1.0,
            teacher_model_kind="base",
            learning_rate=1e-5,
            seed=42,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_generations=1,
            max_prompt_length=16384,
        )

        meta = self_rollout.build_run_meta(args, num_train_examples=100)

        self.assertEqual(meta["pi_mode"], "self_rollout")
        self.assertEqual(meta["self_rollout_pi_source"], "online_same_completion")
        self.assertEqual(meta["self_rollout_pi_template"], self_rollout.PI_SELF_ROLLOUT)
        self.assertNotIn("gen_model", meta)
        self.assertNotIn("rollout_pi_root", meta)
        self.assertNotIn("rollout_pi_sample_idx", meta)
        self.assertFalse(meta["generate_from_teacher"])


if __name__ == "__main__":
    unittest.main()
