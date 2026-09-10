"""CPU checks for random Q, actor freezing, and warmup checkpoint boundaries."""

import copy
import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from transformers import TrainerState
from trl.experimental.sdft import SDFTTrainer

from tests import test_sac, test_sac_state_scale
from train.opsd.train_sac.lib import ResidualQHead
from train.opsd.train_sac.train_sac import build_parser, build_run_meta, sac_run_name
from train.opsd.train_sac.trainer import Q_INIT_FILE
from utils import validate_resume


class _Actor(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits.detach().clone())
        self.grad_modes = []

    def forward(self):
        self.grad_modes.append(torch.is_grad_enabled())
        return self.logits * 1.0


def make_experiment(warmup=2, accumulation=2):
    trainer, logits, inputs, metrics = test_sac_state_scale.make_trainer(scaled=False)
    trainer.args.q_init = "random"
    trainer.args.lam = 1.0
    trainer.args.critic_warmup_steps = warmup
    trainer.state = TrainerState()
    trainer.current_gradient_accumulation_steps = accumulation
    trainer.q_head = ResidualQHead(2, q_init="random")
    actor = _Actor(logits)
    trainer._forward_logits = lambda model, *args: model()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=0.01, weight_decay=0.1)
    with mock.patch.object(SDFTTrainer, "create_optimizer", return_value=optimizer):
        trainer.create_optimizer()
    return trainer, actor, optimizer, inputs, metrics


def optimizer_step(trainer, actor, optimizer, inputs):
    optimizer.zero_grad(set_to_none=True)
    for _ in range(trainer.current_gradient_accumulation_steps):
        trainer.compute_loss(actor, inputs).backward()
    optimizer.step()
    trainer.state.global_step += 1


class RandomQTest(unittest.TestCase):
    def test_small_random_initialization_is_seeded_and_teacher_stays_zero(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            head = ResidualQHead(128, q_init="random", q_init_scale=0.02)
            torch.manual_seed(7)
            same = ResidualQHead(128, q_init="random", q_init_scale=0.02)
        self.assertTrue(torch.equal(head.projection.weight, same.projection.weight))
        self.assertAlmostEqual(
            head.projection.weight.std().item(), 0.02 / math.sqrt(128), delta=0.0001
        )
        self.assertEqual(ResidualQHead(2).projection.weight.count_nonzero().item(), 0)
        with self.assertRaisesRegex(ValueError, "linear"):
            ResidualQHead(2, q_init="random", learn_state_scale=True)

    def test_random_q_scoring_omits_teacher_logps_on_both_paths(self):
        trainer, actor, _, inputs, metrics = make_experiment(warmup=0)
        with torch.no_grad():
            # Teacher logits would be NaN if the old log-probability offset survived.
            trainer.teacher_model.head.bias.fill_(float("nan"))
            hidden = trainer.teacher_model.decoder.hidden[:, :-1]
            all_q = (
                trainer.q_head(hidden).residual_hidden
                @ trainer.teacher_model.head.weight.T
            )
            logps = actor.logits.log_softmax(-1)
            top_ids = logps.topk(2, dim=-1).indices
            top_logps = logps.gather(-1, top_ids)
            expected_v = (
                top_logps.softmax(-1) * (all_q.gather(-1, top_ids) - top_logps)
            ).sum(-1)
            expected_q = all_q.gather(
                -1, inputs["completion_ids"].unsqueeze(-1)
            ).squeeze(-1)
            expected_targets = torch.stack([1.0 - logps[:, 1, 1], torch.ones(1)], dim=1)
        with mock.patch(
            "train.opsd.train_sac.trainer.selective_log_softmax",
            side_effect=AssertionError(
                "random Q must not score teacher log-probabilities"
            ),
        ):
            scored = trainer._teacher_scores_and_q_inputs(inputs, top_ids)
        self.assertNotIn("sampled_teacher_logps", scored)
        self.assertNotIn("topk_teacher_logps", scored)
        with mock.patch("train.opsd.train_sac.trainer._TEACHER_SCORE_CHUNK_SIZE", 1):
            loss = trainer.compute_loss(actor, inputs)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(metrics["sampled_q"], expected_q)
        torch.testing.assert_close(metrics["soft_values"], expected_v)
        torch.testing.assert_close(metrics["q_targets"], expected_targets)
        loss.backward()
        self.assertGreater(trainer.q_head.projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(actor.logits.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in trainer.teacher_model.parameters()))
        with torch.no_grad():
            trainer.q_head.projection.weight.add_(10)
        trainer.compute_loss(actor, inputs)
        torch.testing.assert_close(metrics["q_targets"], expected_targets)


class CriticWarmupTest(unittest.TestCase):
    def test_actor_and_optimizer_state_stay_frozen_until_optimizer_step_boundary(self):
        trainer, actor, optimizer, inputs, metrics = make_experiment()
        original_actor = actor.logits.detach().clone()
        original_q = trainer.q_head.projection.weight.detach().clone()
        for _ in range(2):
            optimizer_step(trainer, actor, optimizer, inputs)
            self.assertEqual(metrics["critic_warmup"].item(), 1)
            self.assertEqual(metrics["actor_loss"].item(), 0)
            self.assertIsNone(actor.logits.grad)
            self.assertTrue(torch.equal(actor.logits, original_actor))
            self.assertNotIn(actor.logits, optimizer.state)
        self.assertEqual(actor.grad_modes, [False] * 4)
        self.assertFalse(torch.equal(trainer.q_head.projection.weight, original_q))
        optimizer_step(trainer, actor, optimizer, inputs)
        self.assertEqual(metrics["critic_warmup"].item(), 0)
        self.assertEqual(actor.grad_modes[-2:], [True, True])
        self.assertFalse(torch.equal(actor.logits, original_actor))
        self.assertEqual(optimizer.state[actor.logits]["step"].item(), 1)

    def test_accumulated_warmup_matches_one_effective_batch(self):
        trainer, actor, optimizer, inputs, _ = make_experiment(accumulation=1)
        accumulated, actor2, optimizer2, inputs2, _ = make_experiment(accumulation=2)
        accumulated.q_head.load_state_dict(trainer.q_head.state_dict())
        optimizer_step(trainer, actor, optimizer, inputs)
        optimizer_step(accumulated, actor2, optimizer2, inputs2)
        torch.testing.assert_close(
            trainer.q_head.projection.weight.grad,
            accumulated.q_head.projection.weight.grad,
        )
        torch.testing.assert_close(
            trainer.q_head.projection.weight, accumulated.q_head.projection.weight
        )

    def test_eval_does_not_report_warmup_or_suppress_actor_loss(self):
        trainer, actor, _, inputs, metrics = make_experiment()
        actor.eval()
        with torch.no_grad():
            trainer.compute_loss(actor, inputs)
        self.assertEqual(metrics["critic_warmup"].item(), 0)
        self.assertNotEqual(metrics["actor_loss"].item(), 0)

    def test_checkpoint_resume_matches_uninterrupted_at_both_warmup_boundaries(self):
        # The parent Trainer restores policy, optimizer, and global_step; exercise
        # the actual Q save/load hooks and loss phase with those restored states.
        for saved_step in (1, 2):
            with (
                self.subTest(saved_step=saved_step),
                tempfile.TemporaryDirectory() as tmp,
            ):
                trainer, actor, optimizer, inputs, _ = make_experiment()
                for _ in range(saved_step):
                    optimizer_step(trainer, actor, optimizer, inputs)
                trainer._save_q_head(tmp)
                trainer.state.save_to_json(str(Path(tmp) / "trainer_state.json"))
                actor_state = copy.deepcopy(actor.state_dict())
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                optimizer_step(trainer, actor, optimizer, inputs)

                resumed, actor2, optimizer2, inputs2, metrics = make_experiment()
                with mock.patch.object(
                    SDFTTrainer,
                    "_load_from_checkpoint",
                    side_effect=lambda *args, target=actor2, state=actor_state: (
                        target.load_state_dict(state)
                    ),
                ):
                    resumed._load_from_checkpoint(tmp)
                resumed.state = TrainerState.load_from_json(
                    str(Path(tmp) / "trainer_state.json")
                )
                optimizer2.load_state_dict(optimizer_state)
                optimizer_step(resumed, actor2, optimizer2, inputs2)
                self.assertEqual(metrics["critic_warmup"].item(), float(saved_step < 2))
                torch.testing.assert_close(actor.logits, actor2.logits, rtol=0, atol=0)
                torch.testing.assert_close(
                    trainer.q_head.projection.weight,
                    resumed.q_head.projection.weight,
                    rtol=0,
                    atol=0,
                )

    def test_checkpoint_rejects_same_shaped_head_with_different_forward_semantics(self):
        trainer, _, _, _, _ = make_experiment()
        teacher, _, _, _ = test_sac_state_scale.make_trainer(scaled=False)
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(SDFTTrainer, "_load_from_checkpoint"),
        ):
            trainer._save_q_head(tmp)
            with self.assertRaisesRegex(ValueError, "q_init"):
                teacher._load_from_checkpoint(tmp)
            teacher._save_q_head(tmp)
            with self.assertRaisesRegex(ValueError, "q_init"):
                trainer._load_from_checkpoint(tmp)
            # Legacy teacher checkpoints had no initialization sidecar.
            (Path(tmp) / Q_INIT_FILE).unlink()
            teacher._load_from_checkpoint(tmp)
            with self.assertRaisesRegex(ValueError, "q_init"):
                trainer._load_from_checkpoint(tmp)


class RandomConfigTest(unittest.TestCase):
    def test_lambda_below_one_warns_but_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertWarnsRegex(UserWarning, "initially untrained critic"):
                config = test_sac.SACConfigTest.make_config(
                    tmp, q_init="random", lam=0.9
                )
            self.assertEqual(config.lam, 0.9)
            for kwargs in (
                {"q_init": "random", "lam": 1.0},
                {"q_init": "teacher", "lam": 0.0},
            ):
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    test_sac.SACConfigTest.make_config(tmp, **kwargs)
                self.assertFalse(any("Random Q" in str(w.message) for w in seen))

    def test_invalid_random_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kwargs, message in (
                ({"q_head_architecture": "state_scaled_linear"}, "linear"),
                ({"q_init_scale": 0}, "positive"),
                ({"q_init_scale": float("nan")}, "finite"),
                ({"critic_warmup_steps": -1}, "warmup"),
                ({"lam": 1.1}, "lam"),
            ):
                with (
                    self.subTest(kwargs=kwargs),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    test_sac.SACConfigTest.make_config(
                        tmp, **({"q_init": "random", "lam": 1.0} | kwargs)
                    )

    def test_cli_defaults_experiment_identity_and_resume_metadata(self):
        parser = build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(
            (defaults.q_init, defaults.critic_warmup_steps, defaults.lam),
            ("teacher", 0, 0.0),
        )
        args = parser.parse_args(
            [
                "--pi-mode",
                "answer",
                "--q-init",
                "random",
                "--q-init-scale",
                "0.01",
                "--critic-warmup-steps",
                "20",
                "--max-steps",
                "220",
                "--lam",
                "1",
            ]
        )
        meta = build_run_meta(args, 2)
        self.assertEqual(meta["q_parameterization"], "frozen_lm_head_random_linear")
        self.assertEqual(
            (meta["q_init_scale"], meta["critic_warmup_steps"], meta["lam"]),
            (0.01, 20, 1.0),
        )
        names = {
            sac_run_name(
                "deepmath",
                "answer",
                "topk",
                q_init="random",
                critic_warmup_steps=20,
                lam=lam,
            )
            for lam in (0.9, 1.0)
        }
        self.assertEqual(len(names), 2)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint-10"
            checkpoint.mkdir()
            (Path(tmp) / "run_meta.json").write_text(json.dumps(meta))
            validate_resume(str(checkpoint), meta)
            for key, value in (
                ("q_init", "teacher"),
                ("critic_warmup_steps", 0),
                ("q_init_scale", 0.1),
            ):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    validate_resume(str(checkpoint), meta | {key: value})


if __name__ == "__main__":
    unittest.main()
