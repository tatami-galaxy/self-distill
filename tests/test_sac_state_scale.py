"""CPU checks for the shared positive state scale, including real Q scoring paths."""

import json
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from trl.experimental.sdft import SDFTTrainer

from tests import test_sac
from train.opsd.train_sac.lib import ResidualQHead, TopKSoftValueEstimator
from train.opsd.train_sac.train_sac import build_parser, build_run_meta, sac_run_name
from train.opsd.train_sac.trainer import SACTrainer
from utils import validate_resume


class StateScaleHeadTest(unittest.TestCase):
    def test_identity_initialization_and_both_projections_learn(self):
        head = ResidualQHead(2, learn_state_scale=True)
        hidden = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])
        action = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        teacher_logps = torch.tensor([[-0.5, -2.0]])
        residual, scale = head(hidden)
        q = scale * teacher_logps + (residual * action).sum(-1)
        self.assertTrue(torch.equal(q, teacher_logps))
        self.assertTrue(torch.equal(scale, torch.ones((1, 2))))
        q.square().mean().backward()
        for parameter in head.parameters():
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_positive_state_scale_preserves_multiplicative_rankings(self):
        head = ResidualQHead(2, learn_state_scale=True)
        with torch.no_grad():
            head.scale_projection.weight.copy_(torch.tensor([[0.5, -0.5]]))
        _, scale = head(torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]))
        self.assertGreater(scale[0, 0].item(), 1.0)
        self.assertGreater(scale[0, 1].item(), 0.0)
        self.assertLess(scale[0, 1].item(), 1.0)
        logps = torch.tensor([[[-1.0, -3.0, -2.0], [-3.0, -2.0, -1.0]]])
        self.assertTrue(
            torch.equal((scale.unsqueeze(-1) * logps).argsort(-1), logps.argsort(-1))
        )

    def test_scale_keeps_small_updates_in_fp32_under_autocast(self):
        head = ResidualQHead(2, learn_state_scale=True)
        with torch.no_grad():
            head.scale_projection.weight.fill_(0.0001)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, scale = head(torch.ones((1, 2, 2), dtype=torch.bfloat16))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(torch.all(scale > 1.0))


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Parameter(torch.tensor([[[1.0, 2.0], [2.0, 1.0], [9.0, 9.0]]]))

    def forward(self, **kwargs):
        return types.SimpleNamespace(last_hidden_state=self.hidden)


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()
        self.head = nn.Linear(2, 3)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
            self.head.bias.copy_(torch.tensor([0.1, -0.2, 0.0]))

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.head


def make_trainer(scaled=True):
    trainer = object.__new__(SACTrainer)
    trainer.args = types.SimpleNamespace(
        lam=0.0, should_save=True, q_init="teacher", critic_warmup_steps=0
    )
    trainer.current_gradient_accumulation_steps = 1
    trainer.temperature = 1.0
    trainer.soft_value_estimator = TopKSoftValueEstimator(2)
    trainer.q_head = ResidualQHead(2, learn_state_scale=scaled)
    trainer.teacher_model = _Teacher()
    trainer.accelerator = types.SimpleNamespace(unwrap_model=lambda model: model)
    trainer._get_teacher_context_for_self_distillation = nullcontext
    trainer._sac_forward_redirection = lambda wrapped, unwrapped, fn, *args: fn(*args)
    logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], requires_grad=True)
    trainer._forward_logits = lambda *args: logits
    captured = {}
    trainer._record_sac_metrics = lambda mode, **kwargs: captured.update(kwargs)
    inputs = {
        "prompt_ids": torch.tensor([[9]]),
        "prompt_mask": torch.ones((1, 1)),
        "completion_ids": torch.tensor([[0, 1]]),
        "completion_mask": torch.ones((1, 2)),
        "teacher_input_ids": torch.tensor([[8, 0, 1]]),
        "teacher_attention_mask": torch.ones((1, 3)),
        "terminal_rewards": torch.tensor([1.0]),
        "scorable": torch.tensor([1.0]),
    }
    return trainer, logits, inputs, captured


class StateScaleTrainerTest(unittest.TestCase):
    def test_initial_loss_and_actor_gradient_match_original_head(self):
        results = []
        for scaled in (False, True):
            trainer, logits, inputs, metrics = make_trainer(scaled)
            loss = trainer.compute_loss(types.SimpleNamespace(training=True), inputs)
            loss.backward()
            results.append((loss.detach(), logits.grad, metrics["soft_advantages"]))
        for original, scaled in zip(*results):
            torch.testing.assert_close(original, scaled, rtol=0, atol=0)

    def test_sampled_and_topk_q_use_same_scale_and_targets_are_detached(self):
        trainer, logits, inputs, metrics = make_trainer()
        with torch.no_grad():
            trainer.q_head.scale_projection.weight.copy_(torch.tensor([[0.15, -0.2]]))
            trainer.q_head.projection.weight.copy_(
                torch.tensor([[0.1, 0.2], [-0.1, 0.3]])
            )
        # Exercise multiple teacher scoring chunks rather than mocking their results.
        with mock.patch("train.opsd.train_sac.trainer._TEACHER_SCORE_CHUNK_SIZE", 1):
            loss = trainer.compute_loss(types.SimpleNamespace(training=True), inputs)
        with torch.no_grad():
            hidden = trainer.teacher_model.decoder.hidden[:, :-1]
            vectors = trainer.teacher_model.head.weight
            teacher_logps = trainer.teacher_model.head(hidden).log_softmax(-1)
            scale = (
                (hidden @ trainer.q_head.scale_projection.weight.T).squeeze(-1).exp()
            )
            correction = hidden @ trainer.q_head.projection.weight.T @ vectors.T
            all_q = scale.unsqueeze(-1) * teacher_logps + correction
            student_logps = logits.log_softmax(-1)
            top_ids = logits.topk(2, dim=-1).indices
            top_logps = student_logps.gather(-1, top_ids)
            soft_v = (
                top_logps.softmax(-1) * (all_q.gather(-1, top_ids) - top_logps)
            ).sum(-1)
            ids = inputs["completion_ids"]
            sampled_q = all_q.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            sampled_logps = student_logps.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            targets = torch.stack([soft_v[:, 1], torch.ones(1)], dim=1)
            advantage = sampled_q - sampled_logps - soft_v
            actor_grad = (
                -advantage.unsqueeze(-1)
                * (nn.functional.one_hot(ids, 3) - student_logps.exp())
                / ids.numel()
            )
            dq = 2 * (sampled_q - targets) / ids.numel()
            sampled_teacher = teacher_logps.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            scale_grad = ((dq * sampled_teacher * scale).unsqueeze(-1) * hidden).sum(
                (0, 1)
            )
            additive_grad = torch.einsum("bt,bti,btj->ij", dq, vectors[ids], hidden)
        torch.testing.assert_close(metrics["sampled_q"], sampled_q)
        torch.testing.assert_close(metrics["soft_values"], soft_v)
        torch.testing.assert_close(metrics["q_targets"], targets)
        loss.backward()
        torch.testing.assert_close(logits.grad, actor_grad)
        torch.testing.assert_close(
            trainer.q_head.scale_projection.weight.grad[0], scale_grad
        )
        torch.testing.assert_close(trainer.q_head.projection.weight.grad, additive_grad)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in trainer.teacher_model.parameters()
            )
        )

    def test_optimizer_owns_both_projections_without_duplicates(self):
        trainer, _, _, _ = make_trainer()
        policy = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([policy], lr=0.1)
        with mock.patch.object(SDFTTrainer, "create_optimizer", return_value=optimizer):
            trainer.create_optimizer()
            trainer.create_optimizer()
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertTrue(
            {id(p) for p in trainer.q_head.parameters()} <= {id(p) for p in parameters}
        )

    def test_checkpoint_restores_scale_and_rejects_cross_architecture(self):
        trainer, _, _, _ = make_trainer()
        with torch.no_grad():
            trainer.q_head.scale_projection.weight.fill_(0.2)
            trainer.q_head.projection.weight.fill_(0.3)
        original = {k: v.clone() for k, v in trainer.q_head.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            trainer._save_q_head(tmp)
            with torch.no_grad():
                for parameter in trainer.q_head.parameters():
                    parameter.zero_()
            with mock.patch.object(SDFTTrainer, "_load_from_checkpoint"):
                trainer._load_from_checkpoint(tmp)
                for key, value in trainer.q_head.state_dict().items():
                    torch.testing.assert_close(value, original[key])
                legacy, _, _, _ = make_trainer(scaled=False)
                with self.assertRaisesRegex(RuntimeError, "scale_projection"):
                    legacy._load_from_checkpoint(tmp)
                legacy._save_q_head(tmp)
                with self.assertRaisesRegex(RuntimeError, "scale_projection"):
                    trainer._load_from_checkpoint(tmp)
                # Original linear checkpoints still load without new parameter keys.
                legacy._load_from_checkpoint(tmp)


class StateScaleConfigTest(unittest.TestCase):
    def test_cli_config_and_run_identity(self):
        parser = build_parser()
        original = parser.parse_args(["--pi-mode", "answer"])
        scaled = parser.parse_args(
            ["--pi-mode", "answer", "--q-head-architecture", "state_scaled_linear"]
        )
        self.assertEqual(original.q_head_architecture, "linear")
        self.assertEqual(scaled.q_head_architecture, "state_scaled_linear")
        self.assertEqual(
            sac_run_name("deepmath", "answer", "topk", scaled.q_head_architecture),
            "deepmath_answer_topk_state_scaled_linear",
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = test_sac.SACConfigTest.make_config(
                tmp, q_head_architecture=scaled.q_head_architecture
            )
            self.assertEqual(config.q_head_architecture, "state_scaled_linear")
            with self.assertRaisesRegex(ValueError, "q_head_architecture"):
                test_sac.SACConfigTest.make_config(tmp, q_head_architecture="invalid")
            checkpoint = Path(tmp) / "checkpoint-20"
            checkpoint.mkdir()
            original_meta = build_run_meta(original, 2)
            (Path(tmp) / "run_meta.json").write_text(json.dumps(original_meta))
            validate_resume(str(checkpoint), original_meta)
            with self.assertRaisesRegex(ValueError, "q_head_architecture"):
                validate_resume(str(checkpoint), build_run_meta(scaled, 2))


if __name__ == "__main__":
    unittest.main()
