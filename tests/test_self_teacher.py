"""Trained self-teacher: prompt alignment, the two objectives, and the diagnostics.

The objective tests are the important ones. `objective_pointwise` exists to concentrate the
teacher's corrective pull on the tokens it currently penalizes MOST -- the off-PI exploration that
nonetheless worked -- and that property lives entirely in the choice of link function. A
well-meaning change to a sigmoid link would silently invert it (sigma' vanishes in the tail), so
the gradient ordering is pinned here rather than left to the docstring.

The alignment tests assert against TRL's real `_tokenize_prompts`, not a reimplementation: the
teacher's logprobs are only comparable to the student's cached ones while both prompts end on the
identical generation header.
"""

import json
import math
import os
import tempfile
import types
import unittest
from unittest import mock

import torch

from tests.helpers import TOKENIZER_ID, FakeChatTokenizer, make_prompt_stub
from train.opsd.train_self_teacher.lib import (
    calibration_metrics,
    cohort_mean_ratio,
    collate_teacher_batch,
    compose_teacher_messages,
    fit_logistic,
    log_ratio,
    objective_endpoint,
    objective_pointwise,
    penalized_correct_indices,
    privileged_context,
    rank_auc,
    render_teacher_prompt_ids,
    sequence_logit,
    teacher_inputs,
    teacher_prompt_template,
    worst_correct_mean,
)
from utils import TEACHER_PROMPT_TEMPLATE, format_prompt_math, validate_resume


class TeacherPromptCompositionTest(unittest.TestCase):
    def test_none_mode_leaves_the_student_prompt_byte_identical(self):
        # The load-bearing property of the matched control: with an identical context the teacher
        # is the student, so rho_t == 0 everywhere at init and every bit of signal provably comes
        # from the E-step. TRL applies the template unconditionally, so the DEFAULT template would
        # append "\n\n" here and quietly destroy that.
        student = format_prompt_math("What is 2 + 2?")
        teacher = compose_teacher_messages(student, "", teacher_prompt_template("none"))
        self.assertEqual(teacher, student)

    def test_default_template_would_not_be_a_no_op(self):
        # Documents exactly what teacher_prompt_template('none') is avoiding.
        student = format_prompt_math("What is 2 + 2?")
        teacher = compose_teacher_messages(student, "", TEACHER_PROMPT_TEMPLATE)
        self.assertNotEqual(teacher, student)
        self.assertTrue(teacher[-1]["content"].endswith("\n\n"))

    def test_pi_is_appended_to_the_user_turn_and_system_survives(self):
        student = format_prompt_math("What is 2 + 2?")
        teacher = compose_teacher_messages(student, "Hint: think about parity.")
        self.assertEqual(teacher[0], student[0])  # system turn untouched
        self.assertEqual(teacher[-1]["role"], "user")  # still ends on a user turn
        self.assertIn("What is 2 + 2?", teacher[-1]["content"])
        self.assertIn("Hint: think about parity.", teacher[-1]["content"])
        self.assertEqual(student, format_prompt_math("What is 2 + 2?"))  # not mutated

    def test_pi_ladder_wording_comes_from_train_sdft(self):
        row = {"hint": "H", "final_answer": "7", "solution": "S"}
        self.assertIn("H", privileged_context(row, "hint"))
        self.assertIn("7", privileged_context(row, "answer"))
        self.assertIn("S", privileged_context(row, "full"))
        self.assertEqual(privileged_context(row, "none"), "")
        with self.assertRaises(ValueError):
            privileged_context(row, "nonsense")


class HermeticPromptBoundaryTest(unittest.TestCase):
    def test_teacher_prompt_ends_on_the_same_generation_header_as_the_policy(self):
        from trl import GRPOTrainer

        tokenizer = FakeChatTokenizer()
        stub = make_prompt_stub(tokenizer)
        prompts = [
            format_prompt_math("What is 2 + 2?"),
            format_prompt_math("Explain why the sum of two even integers is even."),
        ]

        policy_ids, _, _ = GRPOTrainer._tokenize_prompts(stub, prompts)
        teacher_ids = render_teacher_prompt_ids(
            tokenizer, [compose_teacher_messages(p, "Hint: parity.") for p in prompts]
        )

        header = tokenizer.generation_header
        for policy_row, teacher_row in zip(policy_ids, teacher_ids, strict=True):
            self.assertEqual(list(policy_row[-len(header):]), header)
            self.assertEqual(list(teacher_row[-len(header):]), header)
            # Same boundary, longer prompt: the PI made it so, and nothing else did.
            self.assertGreater(len(teacher_row), len(policy_row))


class CollationAlignmentTest(unittest.TestCase):
    def make_rows(self):
        return [
            {"teacher_prompt_ids": [91, 92, 93], "completion_ids": [11, 12, 13],
             "student_logps": [-1.0, -2.0, -3.0], "reward": 1.0},
            {"teacher_prompt_ids": [94], "completion_ids": [21],
             "student_logps": [-0.5], "reward": 0.0},
        ]

    def test_prompt_is_left_padded_and_completion_right_padded(self):
        batch = collate_teacher_batch(self.make_rows(), pad_token_id=0)

        for row in range(2):
            prompt_mask = batch["teacher_prompt_mask"][row].tolist()
            self.assertEqual(prompt_mask, sorted(prompt_mask),
                             "teacher prompt must be LEFT-padded so its real tokens end flush "
                             "against the completion")
            completion_mask = batch["completion_mask"][row].tolist()
            self.assertEqual(completion_mask, sorted(completion_mask, reverse=True),
                             "completion must be RIGHT-padded so pads fall after every real token")

    def test_completion_occupies_the_last_columns_behind_a_real_token(self):
        batch = collate_teacher_batch(self.make_rows(), pad_token_id=0)
        input_ids, attention_mask, logits_to_keep = teacher_inputs(batch)

        n_completion = batch["completion_ids"].size(1)
        self.assertEqual(logits_to_keep, n_completion)
        self.assertTrue(torch.equal(input_ids[:, -n_completion:], batch["completion_ids"]))
        # The position that scores the first completion token must be a real prompt token (the
        # generation header) for every row, not padding.
        self.assertTrue(
            torch.equal(
                attention_mask[:, -n_completion - 1],
                torch.ones(attention_mask.size(0), dtype=attention_mask.dtype),
            )
        )

    def test_init_logps_ride_along_only_when_present(self):
        # --kl-anchor adds `teacher_logps_init` in a pre-pass. The collator has to carry it, and
        # must not require it: every run without the anchor has no such column.
        rows = self.make_rows()
        self.assertNotIn("teacher_logps_init", collate_teacher_batch(rows, pad_token_id=0))

        rows[0]["teacher_logps_init"] = [-1.0, -1.0, -1.0]
        rows[1]["teacher_logps_init"] = [-2.0]
        batch = collate_teacher_batch(rows, pad_token_id=0)
        self.assertEqual(
            batch["teacher_logps_init"].shape, batch["student_logps"].shape,
            "the anchor target must be padded exactly like student_logps or the proximal term "
            "would compare different tokens",
        )
        self.assertEqual(batch["teacher_logps_init"][1].tolist(), [-2.0, 0.0, 0.0])

    def test_student_logps_pad_to_zero_and_the_ratio_masks_them(self):
        batch = collate_teacher_batch(self.make_rows(), pad_token_id=0)
        teacher_logps = torch.full_like(batch["student_logps"], -7.0)
        ratios = log_ratio(teacher_logps, batch)
        # Row 1 has one real token; its two padded positions must contribute nothing.
        self.assertEqual(ratios[1, 1:].abs().sum().item(), 0.0)
        self.assertAlmostEqual(ratios[1, 0].item(), -7.0 - (-0.5), places=5)


class NumericalPrecisionTest(unittest.TestCase):
    """rho is a difference of two nearly-equal logprobs, so precision here IS the signal.

    Measured on Qwen3-1.7B with the `--pi-mode none` control, where rho is analytically zero:
    bf16 logprobs + padded inputs read a spurious `ratio_dispersion` of 0.041; float32 logprobs
    halved it to 0.023; unpadded inputs took it to exactly 0. These pin the two code-level
    properties that got it there.
    """

    class RecordingModel:
        """Minimal stand-in that reports the kwargs it was called with."""

        def __init__(self, vocab=16):
            self.vocab = vocab
            self.calls = []

        def __call__(self, input_ids, attention_mask, position_ids, logits_to_keep, use_cache):
            self.calls.append({"position_ids": position_ids, "logits_to_keep": logits_to_keep})
            logits = torch.zeros(
                input_ids.size(0), logits_to_keep, self.vocab, dtype=torch.bfloat16
            )
            return types.SimpleNamespace(logits=logits)

    def test_position_ids_come_from_the_mask_not_a_bare_arange(self):
        # A bare arange would give the left-padded row [0,1,2,3,4]; deriving from the mask
        # restarts its real tokens at 0, so the forward means the same thing at any padding.
        from train.opsd.train_self_teacher.lib import per_token_logps

        model = self.RecordingModel()
        input_ids = torch.tensor([[0, 0, 5, 6, 7], [1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        per_token_logps(model, input_ids, attention_mask, torch.tensor([[6, 7], [4, 5]]))

        self.assertTrue(torch.equal(
            model.calls[0]["position_ids"],
            torch.tensor([[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]),
        ))
        # Only the C+1 trailing logit positions are requested; a full-length logit tensor is
        # several GB at these sequence lengths.
        self.assertEqual(model.calls[0]["logits_to_keep"], 3)

    def test_logps_are_float32_even_from_bfloat16_logits(self):
        from train.opsd.train_self_teacher.lib import per_token_logps

        model = self.RecordingModel()
        logps = per_token_logps(
            model, torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]), torch.tensor([[2, 3]])
        )
        self.assertEqual(logps.dtype, torch.float32)

    def test_fp32_logps_match_a_reference_log_softmax(self):
        from train.opsd.train_self_teacher.lib import _selective_logps_fp32

        torch.manual_seed(0)
        logits = torch.randn(2, 5, 11)
        index = torch.randint(0, 11, (2, 5))
        expected = torch.log_softmax(logits, dim=-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)
        # Chunk smaller than the flattened row count, to exercise the chunk seam.
        actual = _selective_logps_fp32(logits, index, chunk_size=3)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_concat_padded_assembles_ragged_chunks(self):
        from train.opsd.train_self_teacher.lib import concat_padded

        out = concat_padded([torch.ones(1, 2), torch.full((2, 4), 3.0)])
        self.assertEqual(out.shape, (3, 4))
        self.assertTrue(torch.equal(out[0], torch.tensor([1.0, 1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(out[1], torch.full((4,), 3.0)))


class PointwiseObjectiveTest(unittest.TestCase):
    """(c). The gradient must push hardest on the tokens the teacher penalizes most."""

    def setUp(self):
        # One correct trace whose per-token ratios span from strongly penalized to mildly liked.
        self.ratios = torch.tensor([[-3.0, -1.0, 0.0, 0.5]], requires_grad=True)
        self.mask = torch.ones(1, 4)
        self.reward = torch.tensor([1.0])

    def _grad(self, loss):
        (grad,) = torch.autograd.grad(loss, self.ratios)
        return grad[0]

    def test_squared_pushes_hardest_on_the_most_penalized_token(self):
        grad = self._grad(objective_pointwise(self.ratios, self.mask, self.reward, tau=0.1))
        # Descent moves rho by -grad, so a MORE NEGATIVE grad is a LARGER upward push. The
        # ordering must be strictly increasing from the most penalized token to the least.
        self.assertLess(grad[0].item(), grad[1].item())
        self.assertLess(grad[1].item(), grad[2].item())
        self.assertLess(grad[2].item(), grad[3].item())
        # And the most penalized token is pushed UP, not down.
        self.assertLess(grad[0].item(), 0.0)

    def test_sigmoid_link_would_invert_the_ordering(self):
        # Not an assertion about our code -- a guard on the reasoning behind it. Squared error on
        # a SIGMOID carries a sigma' factor that vanishes in the tail, so the token at rho=-3
        # would receive LESS pull than the one at rho=-1: the opposite of the mechanism this
        # objective exists to provide. Pinned so a future "let's bound it with a sigmoid" change
        # has to confront it.
        values = torch.sigmoid(self.ratios)
        loss = ((values - self.reward.unsqueeze(1)) ** 2).mean()
        grad = self._grad(loss)
        self.assertGreater(abs(grad[0].item()), 0.0)
        self.assertLess(abs(grad[0].item()), abs(grad[1].item()))

    def test_logistic_variant_keeps_pulling_in_the_tail(self):
        grad = self._grad(
            objective_pointwise(self.ratios, self.mask, self.reward, loss="logistic", beta=1.0)
        )
        # BCE-with-logits saturates at a CONSTANT rather than at zero, so the tail token still
        # gets the largest push.
        self.assertLess(grad[0].item(), grad[3].item())
        self.assertLess(grad[0].item(), 0.0)

    def test_wrong_traces_are_pushed_down(self):
        grad = self._grad(
            objective_pointwise(self.ratios, self.mask, torch.tensor([0.0]), tau=0.1)
        )
        self.assertGreater(grad[3].item(), 0.0)  # the token the teacher likes most, pushed down

    def test_padding_is_excluded(self):
        ratios = torch.tensor([[-3.0, -1.0, 99.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        with_pad = objective_pointwise(ratios, mask, torch.tensor([1.0]), tau=0.1)
        without = objective_pointwise(ratios[:, :2], mask[:, :2], torch.tensor([1.0]), tau=0.1)
        self.assertAlmostEqual(with_pad.item(), without.item(), places=5)

    def test_rejects_unknown_loss(self):
        with self.assertRaisesRegex(ValueError, "unknown pointwise loss"):
            objective_pointwise(self.ratios, self.mask, self.reward, loss="hinge")


class EndpointObjectiveTest(unittest.TestCase):
    """(a). One constraint per trace, so per-token allocation stays free."""

    def test_sequence_logit_is_the_masked_mean_ratio(self):
        ratios = torch.tensor([[0.2, 0.4, 99.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        bias = torch.tensor(0.5)
        logit = sequence_logit(ratios, mask, beta=2.0, bias=bias, length_norm="mean")
        self.assertAlmostEqual(logit.item(), 2.0 * (0.6 / 2) + 0.5, places=5)

    def test_length_norms_agree_at_length_one(self):
        ratios = torch.tensor([[0.3]])
        mask = torch.ones(1, 1)
        bias = torch.tensor(0.0)
        values = [
            sequence_logit(ratios, mask, 1.0, bias, norm).item()
            for norm in ("mean", "sqrt", "none")
        ]
        self.assertTrue(all(abs(v - 0.3) < 1e-6 for v in values))

    def test_sparse_and_uniform_allocations_are_equally_optimal(self):
        # The property that preserves credit assignment: only the TOTAL is constrained, so ten
        # tokens at 0.4 and four thousand at 0 score exactly the same as a uniform spread of the
        # same total. Objective (c) cannot say this.
        mask = torch.ones(1, 8)
        reward = torch.tensor([1.0])
        bias = torch.tensor(0.0)
        sparse = torch.zeros(1, 8)
        sparse[0, 2] = 1.6
        uniform = torch.full((1, 8), 0.2)
        self.assertAlmostEqual(
            objective_endpoint(sparse, mask, reward, 1.0, bias, "mean").item(),
            objective_endpoint(uniform, mask, reward, 1.0, bias, "mean").item(),
            places=5,
        )

    def test_bias_absorbs_the_base_rate(self):
        # With a zero ratio the prediction is entirely the bias, which is what lets the ratio stop
        # encoding "this policy solves ~60% of problems".
        ratios = torch.zeros(1, 4)
        mask = torch.ones(1, 4)
        logit = sequence_logit(ratios, mask, 1.0, torch.tensor(1.5), "mean")
        self.assertAlmostEqual(logit.item(), 1.5, places=6)

    def test_rejects_unknown_length_norm(self):
        with self.assertRaisesRegex(ValueError, "unknown length_norm"):
            sequence_logit(torch.zeros(1, 2), torch.ones(1, 2), 1.0, torch.tensor(0.0), "log")


class CalibrationBiasOptimizerTest(unittest.TestCase):
    def test_bias_grad_accumulates_within_step_and_clears_after_optimizer_step(self):
        """The external bias is not reached by Trainer's `model.zero_grad()`."""
        from transformers import Trainer

        from train.opsd.train_self_teacher.train import SelfTeacherTrainer

        policy_parameter = torch.nn.Parameter(torch.tensor(0.0))
        base_optimizer = torch.optim.SGD([policy_parameter], lr=0.1)

        # Exercise SelfTeacherTrainer.create_optimizer directly without constructing a full
        # Accelerator/model stack. Patching only the parent implementation keeps this test focused
        # on the extra parameter group and its post-step lifecycle.
        trainer = object.__new__(SelfTeacherTrainer)
        trainer.calibration_bias = torch.nn.Parameter(torch.tensor(0.0))
        trainer.bias_learning_rate = 0.1
        with mock.patch.object(Trainer, "create_optimizer", return_value=base_optimizer):
            optimizer = SelfTeacherTrainer.create_optimizer(trainer)

        (2.0 * trainer.calibration_bias).backward()
        (3.0 * trainer.calibration_bias).backward()
        self.assertAlmostEqual(trainer.calibration_bias.grad.item(), 5.0)

        optimizer.step()
        self.assertIsNone(trainer.calibration_bias.grad)

        # A new optimizer step starts from a clean gradient rather than inheriting the previous 5.
        (4.0 * trainer.calibration_bias).backward()
        self.assertAlmostEqual(trainer.calibration_bias.grad.item(), 4.0)


class DiagnosticsTest(unittest.TestCase):
    def test_flat_ratio_has_zero_dispersion(self):
        # A flat critic assigns identical credit everywhere -- no credit assignment at all. This
        # is the reading that tells you objective (c) has been run too long.
        ratios = torch.full((2, 6), 0.3)
        mask = torch.ones(2, 6)
        metrics = calibration_metrics(ratios, mask, torch.tensor([1.0, 0.0]))
        self.assertAlmostEqual(metrics["ratio_dispersion"], 0.0, places=6)

    def test_late_firing_ratio_concentrates_credit_in_the_last_quartile(self):
        # What an answer-string-matching teacher looks like: it only moves at the end.
        ratios = torch.zeros(1, 8)
        ratios[0, 6:] = 1.0
        mask = torch.ones(1, 8)
        metrics = calibration_metrics(ratios, mask, torch.tensor([1.0]))
        self.assertAlmostEqual(metrics["credit_mass_last_quartile"], 1.0, places=6)

    def test_evenly_spread_credit_scores_a_quarter(self):
        ratios = torch.full((1, 8), 0.25)
        mask = torch.ones(1, 8)
        metrics = calibration_metrics(ratios, mask, torch.tensor([1.0]))
        self.assertAlmostEqual(metrics["credit_mass_last_quartile"], 0.25, places=6)

    def test_padding_does_not_shift_the_quantiles(self):
        # A row's quantiles are taken over its REAL length, so appending padding must not move
        # them -- otherwise short rows in a long batch would be scored at the wrong prefix.
        ratios = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        short = calibration_metrics(ratios[:, :2], torch.ones(1, 2), torch.tensor([1.0]))
        padded = calibration_metrics(
            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            torch.tensor([1.0]),
        )
        for key in ("brier_q25", "brier_q50", "brier_q100", "value_at_start"):
            self.assertAlmostEqual(short[key], padded[key], places=6)

    def test_separates_correct_from_wrong(self):
        ratios = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        mask = torch.ones(2, 2)
        metrics = calibration_metrics(ratios, mask, torch.tensor([1.0, 0.0]))
        self.assertAlmostEqual(metrics["mean_ratio_correct"], 0.5, places=6)
        self.assertAlmostEqual(metrics["mean_ratio_wrong"], -0.5, places=6)
        self.assertAlmostEqual(metrics["outcome_mean"], 0.5, places=6)

    def test_empty_batch_returns_nothing_rather_than_dividing_by_zero(self):
        self.assertEqual(
            calibration_metrics(torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([1.0])), {}
        )


class FittedCalibrationTest(unittest.TestCase):
    """The raw Brier carries `--beta` and the bias; these pin what the companions remove.

    Numbers quoted in the docstrings come from a 400-trace simulation: two scores with identical
    information but a 10x scale difference read 0.2262 vs 0.2312 raw, and 0.1917 vs 0.1871 fitted.
    """

    def make_batch(self, n=64, seed=0):
        torch.manual_seed(seed)
        reward = (torch.rand(n) < 0.4).float()
        return reward

    def test_flat_ratio_fits_to_the_floor(self):
        # A constant score carries no information, so the fitted link can do no better than the
        # best constant predictor -- and must do no worse.
        reward = self.make_batch()
        ratios = torch.full((reward.numel(), 8), 0.3)
        mask = torch.ones_like(ratios)
        metrics = calibration_metrics(ratios, mask, reward)

        self.assertAlmostEqual(
            metrics["brier_q100_fitted"], metrics["brier_floor"], places=4,
            msg="a flat ratio must fit exactly to the base-rate floor",
        )
        self.assertAlmostEqual(metrics["auc_q100"], 0.5, places=6)

    def test_brier_floor_is_the_best_constant_predictor(self):
        reward = self.make_batch()
        ratios = torch.zeros(reward.numel(), 4)
        mask = torch.ones_like(ratios)
        metrics = calibration_metrics(ratios, mask, reward)

        p = reward.mean().item()
        self.assertAlmostEqual(metrics["brier_floor"], p * (1 - p), places=6)
        # And it really is the minimum over constants.
        for candidate in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertLessEqual(
                metrics["brier_floor"], ((candidate - reward) ** 2).mean().item() + 1e-9
            )

    def test_fitted_brier_and_auc_are_scale_invariant(self):
        # THE POINT OF THE FITTED SERIES. Rescaling every ratio changes nothing about how the
        # traces are ordered, so a discrimination metric must not move -- while the raw Brier,
        # which reads through a fixed --beta, does.
        reward = self.make_batch()
        torch.manual_seed(1)
        base = 0.2 * (reward.unsqueeze(1) - 0.4) + 0.2 * torch.randn(reward.numel(), 6)
        mask = torch.ones_like(base)

        small = calibration_metrics(base, mask, reward)
        large = calibration_metrics(10.0 * base, mask, reward)

        self.assertAlmostEqual(small["brier_q100_fitted"], large["brier_q100_fitted"], places=4)
        self.assertAlmostEqual(small["auc_q100"], large["auc_q100"], places=6)
        # The slope absorbs the rescaling reciprocally, but only to within the ridge's influence:
        # an L2 penalty on (a, b) is not scale-equivariant, so the 10x-larger score is fitted with
        # a slope slightly more than 10x smaller. Measured at ~0.3% here; assert relatively rather
        # than pretending a regularized estimator is exactly homogeneous.
        self.assertAlmostEqual(
            small["platt_slope_q100"] / (10.0 * large["platt_slope_q100"]), 1.0, places=2,
            msg="the fitted slope must absorb the rescaling",
        )
        self.assertNotAlmostEqual(
            small["brier_q100"], large["brier_q100"], places=3,
            msg="the RAW brier is expected to move -- that is what motivates the fitted one",
        )

    def test_informative_ratio_beats_the_floor(self):
        reward = self.make_batch(n=128)
        torch.manual_seed(2)
        ratios = (0.6 * (reward.unsqueeze(1) - 0.4)).expand(-1, 6) + 0.05 * torch.randn(
            reward.numel(), 6
        )
        metrics = calibration_metrics(ratios, torch.ones_like(ratios), reward)

        self.assertLess(metrics["brier_q100_fitted"], metrics["brier_floor"])
        self.assertGreater(metrics["auc_q100"], 0.9)

    def test_single_class_batch_omits_the_fitted_metrics(self):
        # With one outcome class the fit is unidentifiable and AUC undefined; the keys must be
        # absent rather than reporting a degenerate 0.0.
        reward = torch.ones(8)
        ratios = torch.randn(8, 5)
        metrics = calibration_metrics(ratios, torch.ones_like(ratios), reward)

        self.assertIn("brier_q100", metrics)
        self.assertAlmostEqual(metrics["brier_floor"], 0.0, places=9)
        for key in ("brier_q100_fitted", "auc_q100", "platt_slope_q100"):
            self.assertNotIn(key, metrics)

    def test_perfect_separation_stays_finite(self):
        # A 64-row batch can separate perfectly by chance. The unpenalised MLE would diverge and
        # report a meaningless Brier of 0; the ridge keeps it finite.
        reward = torch.cat([torch.zeros(16), torch.ones(16)])
        ratios = torch.cat([-torch.ones(16, 4), torch.ones(16, 4)])
        metrics = calibration_metrics(ratios, torch.ones_like(ratios), reward)

        self.assertTrue(math.isfinite(metrics["platt_slope_q100"]))
        self.assertTrue(math.isfinite(metrics["brier_q100_fitted"]))
        self.assertAlmostEqual(metrics["auc_q100"], 1.0, places=6)

    def test_rank_auc_counts_ties_as_half(self):
        scores = torch.tensor([1.0, 1.0, 1.0, 1.0])
        reward = torch.tensor([1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(rank_auc(scores, reward), 0.5, places=6)
        self.assertIsNone(rank_auc(scores, torch.ones(4)))

    def test_fit_logistic_recovers_known_parameters(self):
        torch.manual_seed(3)
        scores = torch.randn(4000)
        true_a, true_b = 2.0, -0.7
        reward = (torch.rand(4000) < torch.sigmoid(true_a * scores + true_b)).float()
        a, b = fit_logistic(scores, reward)
        self.assertAlmostEqual(a, true_a, delta=0.25)
        self.assertAlmostEqual(b, true_b, delta=0.25)


class PenalizedCorrectTest(unittest.TestCase):
    def test_selects_the_worst_scored_correct_traces_only(self):
        # Four correct traces and one wrong one the teacher likes even less. The metric must
        # ignore the wrong trace entirely -- the population of interest is exploration that WORKED.
        ratios = torch.tensor([
            [-2.0, -2.0],  # correct, most penalized
            [0.0, 0.0],    # correct
            [1.0, 1.0],    # correct
            [2.0, 2.0],    # correct
            [-9.0, -9.0],  # wrong, and penalized hardest of all
        ])
        mask = torch.ones(5, 2)
        reward = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(worst_correct_mean(ratios, mask, reward), -2.0, places=6)

    def test_fixed_cohort_follows_initial_rows_when_the_ranking_changes(self):
        reward = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        mask = torch.ones(5, 2)
        initial = torch.tensor([
            [-2.0, -2.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [-9.0, -9.0],
        ])
        cohort = penalized_correct_indices(initial, mask, reward, decile=0.25)
        self.assertEqual(cohort, [0])

        # Row 0 recovers, while row 1 becomes the new worst correct trace. The fixed-cohort metric
        # must follow row 0; the moving-tail metric intentionally follows row 1.
        final = torch.tensor([
            [3.0, 3.0], [-4.0, -4.0], [1.0, 1.0], [2.0, 2.0], [-9.0, -9.0],
        ])
        self.assertAlmostEqual(cohort_mean_ratio(final, mask, cohort), 3.0, places=6)
        self.assertAlmostEqual(
            worst_correct_mean(final, mask, reward, decile=0.25), -4.0, places=6
        )

    def test_returns_none_without_any_correct_trace(self):
        self.assertIsNone(
            worst_correct_mean(torch.zeros(2, 2), torch.ones(2, 2), torch.tensor([0.0, 0.0]))
        )


class StageThreeResumeTest(unittest.TestCase):
    """A student cannot be resumed into a run whose teacher is a different object."""

    def make_args(self, pi_mode="hint"):
        return types.SimpleNamespace(
            model="Qwen/Qwen3-4B", dataset="deepmath", max_samples=None, pi_mode=pi_mode,
            teacher_path="/tmp/teacher/final", distillation_mode="sampled_token",
            distillation_alpha=1.0, learning_rate=1e-5, seed=42,
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            num_generations=1, max_prompt_length=8192,
        )

    def make_checkpoint(self, tmp, prior_meta):
        with open(os.path.join(tmp, "run_meta.json"), "w") as f:
            json.dump(prior_meta, f)
        ckpt = os.path.join(tmp, "checkpoint-20")
        os.makedirs(ckpt)
        return ckpt

    def test_teacher_version_absence_is_disqualifying(self):
        from train.opsd.train_self_teacher.sdft_with_teacher import build_run_meta

        args = self.make_args()
        meta = build_run_meta(args, {"teacher_version": "logratio_v1", "objective": "pointwise"}, 100)
        legacy = {k: v for k, v in meta.items() if k != "teacher_version"}
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, legacy)
            with self.assertRaisesRegex(ValueError, "teacher_version"):
                validate_resume(ckpt, meta, strict_keys=("teacher_version",))

    def test_a_different_teacher_objective_is_a_mismatch(self):
        from train.opsd.train_self_teacher.sdft_with_teacher import build_run_meta

        args = self.make_args()
        pointwise = build_run_meta(args, {"teacher_version": "logratio_v1", "objective": "pointwise"}, 100)
        endpoint = build_run_meta(args, {"teacher_version": "logratio_v1", "objective": "endpoint"}, 100)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, pointwise)
            with self.assertRaisesRegex(ValueError, "teacher_objective"):
                validate_resume(ckpt, endpoint, strict_keys=("teacher_version",))

    def test_matching_run_resumes(self):
        from train.opsd.train_self_teacher.sdft_with_teacher import build_run_meta

        args = self.make_args()
        meta = build_run_meta(args, {"teacher_version": "logratio_v1", "objective": "pointwise"}, 100)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, dict(meta))
            validate_resume(ckpt, meta, strict_keys=("teacher_version",))

    def test_none_mode_records_the_concatenating_template(self):
        from train.opsd.train_self_teacher.sdft_with_teacher import build_run_meta

        meta = build_run_meta(self.make_args("none"), {"teacher_version": "logratio_v1"}, 100)
        self.assertEqual(meta["teacher_prompt_template"], "{prompt}{privileged_context}")


class RealTokenizerAlignmentTest(unittest.TestCase):
    """Cross-library: holds only while TRL renders prompts the way we mirror. Tokenizer only."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_ID, trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:  # no local cache / no network
            raise unittest.SkipTest(f"tokenizer {TOKENIZER_ID} unavailable: {exc}")

    def test_teacher_prompt_ends_on_the_policy_generation_header(self):
        from trl import GRPOTrainer

        messages = format_prompt_math("What is the sum of the roots of x^2 - 5x + 6 = 0?")
        stub = make_prompt_stub(self.tokenizer)

        with_header = self.tokenizer.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"][0]
        without_header = self.tokenizer.apply_chat_template(
            [messages], add_generation_prompt=False, tokenize=True, return_dict=True
        )["input_ids"][0]
        header = with_header[len(without_header):]
        self.assertGreater(len(header), 0, "chat template appends no generation header")

        policy_ids, _, _ = GRPOTrainer._tokenize_prompts(stub, [messages])
        self.assertEqual(list(policy_ids[0][-len(header):]), list(header))

        teacher_ids = render_teacher_prompt_ids(
            self.tokenizer,
            [compose_teacher_messages(messages, "Hint: Vieta's formulas.")],
        )[0]
        self.assertEqual(list(teacher_ids[-len(header):]), list(header))
        self.assertIn("Vieta", self.tokenizer.decode(teacher_ids))

    def test_none_mode_teacher_prompt_is_token_identical_to_the_policy_prompt(self):
        # The exact null, at the token level: identical ids means rho_t == 0 at every position for
        # an untrained teacher, so `none` measures the E-step and nothing else.
        from trl import GRPOTrainer

        messages = format_prompt_math("What is 2 + 2?")
        stub = make_prompt_stub(self.tokenizer)

        policy_ids, _, _ = GRPOTrainer._tokenize_prompts(stub, [messages])
        teacher_ids = render_teacher_prompt_ids(
            self.tokenizer,
            [compose_teacher_messages(messages, "", teacher_prompt_template("none"))],
        )[0]
        self.assertEqual(list(teacher_ids), list(policy_ids[0]))


if __name__ == "__main__":
    unittest.main()
