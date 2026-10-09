import json
import tempfile
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset
from trl import GRPOTrainer

from tests.helpers import FakeChatTokenizer
from train.rlsd.train_rlsd import (
    TEACHER_FILE,
    RLSDTrainer,
    build_dataset,
    build_parser,
    dataset_digest,
    mixing_coefficient,
    reweight_advantages,
    validate_args,
)


def stub(step=0):
    trainer = object.__new__(RLSDTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.teacher_model = torch.nn.Linear(2, 2).requires_grad_(False)
    trainer.state = types.SimpleNamespace(global_step=step)
    trainer.rlsd_lambda, trainer.rlsd_decay_steps = 0.5, 50
    trainer.rlsd_clip_range, trainer.teacher_sync_steps = 0.2, 10
    trainer.last_teacher_sync_step = 0
    trainer.accelerator = types.SimpleNamespace(
        unwrap_model=lambda m: m, reduce=lambda x, **kw: x
    )
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.args = types.SimpleNamespace(should_save=True)
    return trainer


class EvidenceTests(unittest.TestCase):
    def test_direction_bounds_and_stop_gradient(self):
        student = torch.zeros(3, 3, requires_grad=True)
        teacher = torch.tensor([[1000.0, -1000.0, 0.0]] * 3, requires_grad=True)
        a = torch.tensor([2.0, -2.0, 0.0])
        result, _ = reweight_advantages(
            a, teacher, student, torch.ones_like(student), 0.5, 0.2
        )
        torch.testing.assert_close(
            result, torch.tensor([[2.2, 1.8, 2.0], [-1.8, -2.2, -2.0], [0.0, 0.0, 0.0]])
        )
        self.assertFalse(result.requires_grad)
        self.assertIsNone(teacher.grad)

    def test_padding_is_ignored_and_nonfinite_real_scores_fail(self):
        mask = torch.tensor([[1.0, 0.0]])
        teacher = torch.tensor([[0.0, float("nan")]])
        result, _ = reweight_advantages(
            torch.ones(1), teacher, torch.zeros_like(mask), mask, 0.5, 0.2
        )
        torch.testing.assert_close(result, mask)
        teacher[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            reweight_advantages(torch.ones(1), teacher, mask, mask, 0.5, 0.2)

    def test_schedule(self):
        self.assertEqual(
            [mixing_coefficient(s, 0.5, 50) for s in [0, 25, 50, 100]],
            [0.5, 0.25, 0.0, 0.0],
        )
        self.assertEqual(mixing_coefficient(100, 0.5, 0), 0.5)

    def test_zero_mixing_is_exact_parent_loss_and_gradient_without_scoring(self):
        trainer = stub(50)
        x = torch.tensor([0.2, -0.3], requires_grad=True)
        inputs = {"advantages": torch.tensor([1.0, -1.0])}

        def parent(self, model, batch):
            self.assert_batch = batch
            return -(x.exp() * batch["advantages"]).sum()

        with (
            patch.object(GRPOTrainer, "_compute_loss", parent),
            patch.object(trainer, "_evidence_logps") as score,
        ):
            loss = trainer._compute_loss(None, inputs)
            loss.backward()
            score.assert_not_called()
        self.assertIs(trainer.assert_batch, inputs)
        torch.testing.assert_close(x.grad, -x.detach().exp() * inputs["advantages"])

    def test_weighted_gradient_and_buffer_not_mutated(self):
        trainer = stub()
        x = torch.zeros(2, 2, requires_grad=True)
        inputs = {
            "advantages": torch.tensor([1.0, -1.0]),
            "completion_mask": torch.ones(2, 2),
        }
        teacher = torch.tensor([[0.1, -0.1], [0.1, -0.1]], requires_grad=True)
        trainer._evidence_logps = lambda *_: (teacher, x)

        def parent(self, model, batch):
            return -(x.exp() * batch["advantages"]).sum()

        expected, _ = reweight_advantages(
            inputs["advantages"], teacher, x, inputs["completion_mask"], 0.5, 0.2
        )
        with patch.object(GRPOTrainer, "_compute_loss", parent):
            trainer._compute_loss(None, inputs).backward()
        torch.testing.assert_close(x.grad, -expected)
        self.assertIsNone(teacher.grad)
        self.assertEqual(inputs["advantages"].shape, (2,))

    def test_evidence_uses_current_model_exact_tokens_and_unpadding(self):
        trainer = stub()
        current = object()
        batch = {
            "prompt_ids": torch.tensor([[0, 1, 2]]),
            "prompt_mask": torch.tensor([[0, 1, 1]]),
            "teacher_prompt_ids": torch.tensor([[3, 4, 5]]),
            "teacher_prompt_mask": torch.ones(1, 3),
            "completion_ids": torch.tensor([[6, 7, 0]]),
            "completion_mask": torch.tensor([[1, 1, 0]]),
            "old_per_token_logps": torch.full((1, 3), -999.0),
        }
        calls = []

        def score(model, ids, completion):
            calls.append(
                (model, ids.tolist(), completion.tolist(), torch.is_grad_enabled())
            )
            return torch.full((1, 2), -1.0 if model is current else -2.0)

        with patch("train.rlsd.train_rlsd.per_token_logps", score):
            teacher, student = trainer._evidence_logps(current, batch)
        self.assertEqual(calls[0], (current, [[1, 2, 6, 7]], [[6, 7]], False))
        self.assertEqual(calls[1][1:], ([[3, 4, 5, 6, 7]], [[6, 7]], False))
        torch.testing.assert_close(student, torch.tensor([[-1.0, -1.0, 0.0]]))
        torch.testing.assert_close(teacher, torch.tensor([[-2.0, -2.0, 0.0]]))

    def test_empty_sign_subsets_do_not_dilute_diagnostics(self):
        trainer = stub()
        scores = torch.zeros(1, 2)
        trainer._evidence_logps = lambda *_: (scores, scores)
        with patch.object(GRPOTrainer, "_compute_loss", return_value=torch.tensor(0.0)):
            for advantage in (1.0, -1.0, 0.0):
                trainer._compute_loss(
                    None,
                    {
                        "advantages": torch.tensor([advantage]),
                        "completion_mask": torch.ones_like(scores),
                    },
                )
        metrics = trainer._metrics["train"]
        self.assertEqual(metrics["rlsd/all/multiplier"], [1.0, 1.0, 1.0])
        self.assertEqual(metrics["rlsd/positive/multiplier"], [1.0])
        self.assertEqual(metrics["rlsd/negative/multiplier"], [1.0])


class LifecycleTests(unittest.TestCase):
    def test_sync_once_after_update_before_checkpoint(self):
        trainer = stub(10)
        seen = []

        def parent(self, *args, **kwargs):
            seen.append(self.last_teacher_sync_step)

        with patch.object(GRPOTrainer, "_maybe_log_save_evaluate", parent):
            trainer._maybe_log_save_evaluate()
            for p in trainer.model.parameters():
                p.data.add_(1.0)
            trainer._maybe_log_save_evaluate()
        self.assertEqual(seen, [10, 10])
        self.assertFalse(
            torch.equal(trainer.model.weight, trainer.teacher_model.weight)
        )
        trainer.state.global_step = 20
        with patch.object(GRPOTrainer, "_maybe_log_save_evaluate", parent):
            trainer._maybe_log_save_evaluate()
        torch.testing.assert_close(trainer.model.weight, trainer.teacher_model.weight)

    def test_checkpoint_restores_stale_teacher_not_current_policy(self):
        trainer = stub(13)
        trainer.last_teacher_sync_step = 10
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-13"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": 13})
            )
            trainer._get_output_dir = lambda _: directory
            with patch.object(GRPOTrainer, "_save_checkpoint"):
                trainer._save_checkpoint(None, None)
            restored = stub(13)
            with patch.object(GRPOTrainer, "_load_from_checkpoint"):
                restored._load_from_checkpoint(str(checkpoint))
            torch.testing.assert_close(
                restored.teacher_model.weight, trainer.teacher_model.weight
            )
            self.assertEqual(restored.last_teacher_sync_step, 10)
            self.assertEqual(restored.mixing, trainer.mixing)
            saved = torch.load(checkpoint / TEACHER_FILE, weights_only=True)
            saved["global_step"] = 12
            torch.save(saved, checkpoint / TEACHER_FILE)
            with self.assertRaisesRegex(ValueError, "disagree"):
                restored._load_from_checkpoint(str(checkpoint))

    def test_missing_teacher_rejected(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(FileNotFoundError),
        ):
            stub()._load_from_checkpoint(directory)


class DatasetTests(unittest.TestCase):
    def test_three_sources_use_shared_sdft_loader_and_reward_targets(self):
        source = Dataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": "q"}],
                    "privileged_context": "hint",
                    "solution": r"\boxed{1}",
                }
            ]
        )
        for mode in ["answer", "hint", "hint_trained"]:
            args = build_parser().parse_args(["--pi-mode", mode])
            if mode == "hint_trained":
                args.hint_generator_model, args.hint_cache = "/trained", "/cache"
            validate_args(args)
            with patch(
                "train.rlsd.train_rlsd.build_sdft_dataset", return_value=source
            ) as loader:
                ds = build_dataset(args, FakeChatTokenizer())
            self.assertEqual(ds[0], source[0])
            self.assertTrue(loader.call_args.kwargs["include_reward_solution"])
            self.assertEqual(
                loader.call_args.args[0], "answer" if mode == "answer" else "hint"
            )
            self.assertEqual(
                loader.call_args.kwargs["hint_generator_model"],
                args.hint_generator_model,
            )

    def test_oversized_hint_is_filtered_without_truncation(self):
        source = Dataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": "q"}],
                    "privileged_context": "too long",
                    "solution": "1",
                }
            ]
        )
        args = build_parser().parse_args(["--max-prompt-length", "1"])
        with (
            patch("train.rlsd.train_rlsd.build_sdft_dataset", return_value=source),
            self.assertRaisesRegex(ValueError, "No RLSD examples"),
        ):
            build_dataset(args, FakeChatTokenizer())

    def test_fingerprint_detects_content_and_order_changes(self):
        a = [{"prompt": "a", "hint": "x"}, {"prompt": "b", "hint": "y"}]
        self.assertNotEqual(dataset_digest(a), dataset_digest(list(reversed(a))))
        self.assertNotEqual(
            dataset_digest(a), dataset_digest([a[0], {**a[1], "hint": "z"}])
        )

    def test_generator_requires_trained_mode(self):
        for argv in [
            ["--pi-mode", "hint_trained"],
            ["--hint-generator-model", "/trained"],
        ]:
            with self.assertRaises(ValueError):
                validate_args(build_parser().parse_args(argv))

    def test_eval_routes_all_three_variants(self):
        from eval.run_eval import ALGOS, arm_path

        self.assertIn("rlsd", ALGOS)
        for variant in ["answer", "hint", "hint_trained"]:
            args = types.SimpleNamespace(
                model="/runs/checkpoint-20",
                model_name="Qwen3-1.7B",
                algo="rlsd",
                train_dataset="deepmath",
                variant=variant,
                run=None,
                step=None,
            )
            parts, _ = arm_path(args, lambda msg: self.fail(msg))
            self.assertEqual(
                parts, ["deepmath", "Qwen3-1.7B", "rlsd", variant, "checkpoint-20"]
            )


if __name__ == "__main__":
    unittest.main()
