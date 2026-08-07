import types
import unittest
from unittest import mock

from datasets import Dataset

from train.opsd import train_sdft


def rollout_cache(*, mixed_only=False):
    # Deliberately interleave sample indices and give sample_idx=1 opposite rewards.
    # The selected PI text must depend only on sample_idx, never on reward.
    return Dataset.from_dict({
        "question": ["q0", "q0", "q1", "q1"],
        "question_idx": [0, 0, 1, 1],
        "completion_text": ["q0 attempt zero", "q0 attempt one",
                            "q1 attempt zero", "q1 attempt one"],
        "sample_idx": [0, 1, 0, 1],
        "reward": [1.0, 0.0, 0.0, 1.0],
        "gen_model": ["student"] * 4,
        "dataset": ["deepmath"] * 4,
        "question_source": ["hints"] * 4,
        "mixed_only": [mixed_only] * 4,
        "generation_seed": [42] * 4,
        "max_completion_length": [8192] * 4,
    })


def hint_cache():
    return Dataset.from_dict({
        "question": ["q0", "q1"],
        "final_answer": ["0", "1"],
        "hint": ["h0", "h1"],
    })


class RolloutSdftDatasetTest(unittest.TestCase):
    def build(self, cache=None, sample_idx=1):
        with (
            mock.patch.object(train_sdft, "rollout_pi_path", return_value="pi/cache"),
            mock.patch.object(train_sdft.os.path, "isdir", return_value=True),
            mock.patch.object(
                train_sdft, "load_from_disk",
                return_value=cache if cache is not None else rollout_cache(),
            ),
            mock.patch.object(train_sdft, "load_hint_cache", return_value=hint_cache()),
        ):
            return train_sdft.build_sdft_dataset(
                "rollout",
                dataset="deepmath",
                max_samples=2,
                model="student",
                max_prompt_length=None,
                rollout_pi_root="pi",
                rollout_pi_sample_idx=sample_idx,
            )

    def test_fixed_sample_is_joined_in_hint_source_order_without_reward_selection(self):
        ds = self.build(sample_idx=1)

        self.assertEqual(ds.column_names, ["prompt", "privileged_context"])
        self.assertEqual(
            [row["prompt"][-1]["content"] for row in ds],
            ["q0", "q1"],
        )
        self.assertIn("q0 attempt one", ds[0]["privileged_context"])
        self.assertIn("q1 attempt one", ds[1]["privileged_context"])
        self.assertIn("may or may not be correct", ds[0]["privileged_context"])

    def test_verifier_selected_mixed_only_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "verifier outcomes"):
            self.build(cache=rollout_cache(mixed_only=True))

    def test_question_text_mismatch_at_source_index_is_rejected(self):
        cache = rollout_cache().map(
            lambda row: {
                "question": "wrong q0" if row["question_idx"] == 0 else row["question"]
            }
        )
        with self.assertRaisesRegex(ValueError, "caches disagree"):
            self.build(cache=cache)

    def test_missing_fixed_attempt_is_rejected(self):
        cache = rollout_cache().filter(lambda row: row["question_idx"] == 0)
        with self.assertRaisesRegex(ValueError, "lacks sample_idx=1"):
            self.build(cache=cache)


class LongPiPromptFilterTest(unittest.TestCase):
    def test_rollout_prompt_is_filtered_using_exact_teacher_messages(self):
        class Tokenizer:
            @staticmethod
            def apply_chat_template(conversations, **_kwargs):
                user_text = conversations[0][-1]["content"]
                length = 20 if "LONG ATTEMPT" in user_text else 5
                return {"input_ids": [list(range(length))]}

        ds = Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": "q0"}],
                "privileged_context": "short attempt",
            },
            {
                "prompt": [{"role": "user", "content": "q1"}],
                "privileged_context": "LONG ATTEMPT",
            },
        ])
        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()
        ):
            filtered = train_sdft.filter_long_pi_prompts(
                ds, "rollout", model="student", max_prompt_length=10
            )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["prompt"][-1]["content"], "q0")


class RolloutRunMetadataTest(unittest.TestCase):
    def test_rollout_cache_selection_is_recorded(self):
        args = types.SimpleNamespace(
            model="student",
            pi_mode="rollout",
            dataset="deepmath",
            max_samples=128,
            rollout_pi_root="data/pi/attempted_solution_8k",
            rollout_pi_sample_idx=0,
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

        meta = train_sdft.build_run_meta(args, num_train_examples=100)

        self.assertEqual(meta["gen_model"], "student")
        self.assertEqual(meta["rollout_pi_sample_idx"], 0)
        self.assertEqual(
            meta["rollout_pi_selection_policy"], "fixed_sample_idx_without_reward"
        )


if __name__ == "__main__":
    unittest.main()
