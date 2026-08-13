"""Trained self-teacher: prompt alignment, asymmetric objectives, and diagnostics.

The objective tests pin the disjoint fixed lift/complement masks, the distinction between
per-token and aggregate lift constraints, and the optional sampled-logp anchor.

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
    ASYMMETRIC_OBJECTIVES,
    TEACHER_VERSION,
    asymmetric_lift_mask,
    calibration_metrics,
    collate_teacher_batch,
    compose_teacher_messages,
    fit_logistic,
    log_ratio,
    macro_group_rank_auc,
    objective_asymmetric,
    privileged_context,
    rank_auc,
    render_teacher_prompt_ids,
    sampled_logp_anchor,
    teacher_inputs,
    teacher_prompt_template,
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
        # Initialization-anchored losses add this in a pre-pass. The collator has to carry it, and
        # must not require it: objectives without an initialization reference have no such column.
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

    def test_asymmetric_row_weight_rides_along_when_present(self):
        rows = self.make_rows()
        rows[0]["asym_lift_weight"] = 1.25
        rows[1]["asym_lift_weight"] = 2.5
        batch = collate_teacher_batch(rows, pad_token_id=0)
        self.assertTrue(torch.equal(batch["asym_lift_weight"], torch.tensor([1.25, 2.5])))

    def test_student_logps_pad_to_zero_and_the_ratio_masks_them(self):
        batch = collate_teacher_batch(self.make_rows(), pad_token_id=0)
        teacher_logps = torch.full_like(batch["student_logps"], -7.0)
        ratios = log_ratio(teacher_logps, batch)
        # Row 1 has one real token; its two padded positions must contribute nothing.
        self.assertEqual(ratios[1, 1:].abs().sum().item(), 0.0)
        self.assertAlmostEqual(ratios[1, 0].item(), -7.0 - (-0.5), places=5)


class NumericalPrecisionTest(unittest.TestCase):
    """rho is a difference of two nearly-equal logprobs, so precision here IS the signal.

    Locally measured on Qwen3-1.7B with the `--pi-mode none` control, where rho is analytically
    zero: bf16 logprobs + padded inputs read a spurious `ratio_dispersion` of 0.041; float32
    logprobs halved it to 0.023; unpadded inputs took it to exactly 0. The padded and unpadded
    forwards are mathematically equivalent with correct masks/positions, but shape-dependent
    finite-precision kernels need not be bitwise identical. These tests pin the two code-level
    properties that removed the measured noise; they do not identify attention-mask leakage as
    its cause.
    """

    class RecordingModel:
        """Minimal stand-in that reports the kwargs it was called with."""

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
        from train.opsd.train_self_teacher.lib import per_token_logps

        model = self.RecordingModel()
        with self.assertRaisesRegex(ValueError, "physical batch size 1"):
            per_token_logps(
                model,
                torch.tensor([[5, 6, 7], [3, 4, 5]]),
                torch.tensor([[6, 7], [4, 5]]),
            )
        self.assertEqual(model.calls, [])

    def test_logps_are_float32_even_from_bfloat16_logits(self):
        from train.opsd.train_self_teacher.lib import per_token_logps

        model = self.RecordingModel()
        logps = per_token_logps(
            model, torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3]])
        )
        self.assertEqual(logps.dtype, torch.float32)
        # Only the C+1 trailing logit positions are requested; a full-length logit tensor is
        # several GB at these sequence lengths.
        self.assertEqual(model.calls[0]["logits_to_keep"], 3)

    def test_fp32_logps_match_a_reference_log_softmax(self):
        from train.opsd.train_self_teacher.lib import _selective_logps_fp32

        torch.manual_seed(0)
        logits = torch.randn(2, 5, 11)
        index = torch.randint(0, 11, (2, 5))
        expected = torch.log_softmax(logits, dim=-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)
        # Chunk smaller than the flattened row count, to exercise the chunk seam.
        actual = _selective_logps_fp32(logits, index, chunk_size=3)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_fp32_logps_and_entropy_match_reference(self):
        from train.opsd.train_self_teacher.lib import _selective_logps_entropy_fp32

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

    def test_entropy_is_invariant_to_a_constant_logit_shift(self):
        from train.opsd.train_self_teacher.lib import _selective_logps_entropy_fp32

        logits = torch.randn(1, 4, 9)
        index = torch.tensor([[0, 1, 2, 3]])
        original = _selective_logps_entropy_fp32(logits, index, chunk_size=2)
        shifted = _selective_logps_entropy_fp32(logits + 123.0, index, chunk_size=2)
        self.assertTrue(torch.allclose(original.logps, shifted.logps, atol=1e-5))
        self.assertTrue(torch.allclose(original.entropy, shifted.entropy, atol=1e-5))

    def test_per_token_stats_reuses_the_batch_one_alignment(self):
        from train.opsd.train_self_teacher.lib import per_token_stats

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

    def test_concat_padded_assembles_ragged_chunks(self):
        from train.opsd.train_self_teacher.lib import concat_padded

        out = concat_padded([torch.ones(1, 2), torch.full((2, 4), 3.0)])
        self.assertEqual(out.shape, (3, 4))
        self.assertTrue(torch.equal(out[0], torch.tensor([1.0, 1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(out[1], torch.full((4,), 3.0)))


class AsymmetricObjectiveTest(unittest.TestCase):
    """Only successful, initially penalized tokens move; the complement stays anchored."""

    def test_only_current_objective_names_are_public(self):
        self.assertEqual(ASYMMETRIC_OBJECTIVES, ("asymmetric", "asymmetric_aggregate"))

    def test_lift_mask_is_fixed_and_target_is_finite(self):
        initial = torch.tensor([[-0.4, -0.1, 0.0, 0.2]])
        mask = torch.ones(1, 4)
        reward = torch.tensor([1.0])

        lift_mask = asymmetric_lift_mask(initial, mask, reward, margin=0.0)
        self.assertEqual(lift_mask.tolist(), [[True, True, False, False]])

        ratios = initial.clone().requires_grad_(True)
        loss = objective_asymmetric(
            ratios,
            initial,
            mask,
            reward,
            torch.ones(1),
            margin=0.0,
            lift_alpha=0.5,
        )
        (grad,) = torch.autograd.grad(loss, ratios)
        self.assertLess(grad[0, 0].item(), 0.0)
        self.assertLess(grad[0, 1].item(), 0.0)
        self.assertEqual(grad[0, 2].item(), 0.0)
        self.assertEqual(grad[0, 3].item(), 0.0)

        target = torch.tensor([[-0.2, -0.05, 0.0, 0.2]])
        zero_loss = objective_asymmetric(
            target,
            initial,
            mask,
            reward,
            torch.ones(1),
            margin=0.0,
            lift_alpha=0.5,
        )
        self.assertAlmostEqual(zero_loss.item(), 0.0, places=7)

    def test_failed_rollout_is_anchor_only(self):
        initial = torch.tensor([[-0.4, 0.2]])
        ratios = torch.tensor([[-0.1, 0.5]], requires_grad=True)
        loss = objective_asymmetric(
            ratios, initial, torch.ones(1, 2), torch.tensor([0.0]), torch.zeros(1)
        )
        (grad,) = torch.autograd.grad(loss, ratios)
        self.assertAlmostEqual(loss.item(), 0.09, places=6)
        self.assertGreater(grad[0, 0].item(), 0.0)
        self.assertGreater(grad[0, 1].item(), 0.0)

    def test_successful_token_above_margin_is_anchored(self):
        initial = torch.tensor([[0.2]])
        ratios = torch.tensor([[0.4]])
        loss = objective_asymmetric(
            ratios, initial, torch.ones(1, 1), torch.ones(1), torch.zeros(1)
        )
        self.assertAlmostEqual(loss.item(), 0.04, places=6)

    def test_padding_is_excluded_from_both_disjoint_masks(self):
        full_initial = torch.tensor([[-0.4, 100.0]])
        full_ratios = torch.tensor([[-0.2, -999.0]])
        full_mask = torch.tensor([[1.0, 0.0]])
        full_loss = objective_asymmetric(
            full_ratios, full_initial, full_mask, torch.ones(1), torch.ones(1)
        )
        short_loss = objective_asymmetric(
            full_ratios[:, :1], full_initial[:, :1], torch.ones(1, 1),
            torch.ones(1), torch.ones(1),
        )
        self.assertAlmostEqual(full_loss.item(), short_loss.item(), places=7)

    def test_aggregate_lift_allows_token_residuals_to_compensate(self):
        initial = torch.tensor([[-0.4, -0.2]])
        ratios = torch.tensor([[-0.1, 0.1]])
        args = (initial, torch.ones(1, 2), torch.ones(1), torch.ones(1))

        per_token = objective_asymmetric(ratios, *args)
        aggregate = objective_asymmetric(ratios, *args, aggregate=True)

        self.assertGreater(per_token.item(), 0.0)
        self.assertAlmostEqual(aggregate.item(), 0.0, places=7)

    def test_aggregate_lift_has_equal_gradient_on_every_lift_token(self):
        initial = torch.tensor([[-0.4, -0.2]])
        ratios = initial.clone().requires_grad_(True)
        loss = objective_asymmetric(
            ratios,
            initial,
            torch.ones(1, 2),
            torch.ones(1),
            torch.ones(1),
            aggregate=True,
        )
        (grad,) = torch.autograd.grad(loss, ratios)

        self.assertLess(grad[0, 0].item(), 0.0)
        self.assertAlmostEqual(grad[0, 0].item(), grad[0, 1].item(), places=7)

    def test_aggregate_matches_per_token_for_one_lift_token(self):
        initial = torch.tensor([[-0.4]])
        ratios = torch.tensor([[-0.1]])
        args = (initial, torch.ones(1, 1), torch.ones(1), torch.ones(1))

        per_token = objective_asymmetric(ratios, *args)
        aggregate = objective_asymmetric(ratios, *args, aggregate=True)

        self.assertAlmostEqual(per_token.item(), aggregate.item(), places=7)


class SampledLogpAnchorTest(unittest.TestCase):
    def test_masks_padding_and_averages_sampled_token_drift(self):
        current = torch.tensor([[-0.2, 0.4, 99.0]], requires_grad=True)
        initial = torch.tensor([[-0.4, 0.1, -10.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])

        loss = sampled_logp_anchor(current, initial, mask)
        self.assertAlmostEqual(loss.item(), (0.2 ** 2 + 0.3 ** 2) / 2, places=7)
        (grad,) = torch.autograd.grad(loss, current)
        self.assertEqual(grad[0, 2].item(), 0.0)

    def test_rejects_misaligned_inputs(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            sampled_logp_anchor(torch.zeros(1, 2), torch.zeros(1, 1), torch.ones(1, 2))


class TrainerLoggingTest(unittest.TestCase):
    def make_trainer(self):
        from train.opsd.train_self_teacher.train_logratio_teacher import LogRatioTeacherTrainer

        trainer = object.__new__(LogRatioTeacherTrainer)
        trainer._train_ratio_sum = None
        trainer._train_ratio_count = None
        trainer.accelerator = types.SimpleNamespace(reduce=lambda value, reduction: value)
        return trainer

    def test_ratio_mean_accumulates_by_trajectory_and_resets_at_log(self):
        from train.opsd.train_self_teacher.train_logratio_teacher import LogRatioTeacherTrainer

        trainer = self.make_trainer()
        LogRatioTeacherTrainer._accumulate_train_ratio(
            trainer,
            torch.tensor([[1.0, 1.0, 0.0], [3.0, 3.0, 3.0]]),
            torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
        )
        LogRatioTeacherTrainer._accumulate_train_ratio(
            trainer, torch.tensor([[5.0]]), torch.tensor([[1.0]])
        )

        metrics = LogRatioTeacherTrainer._pop_train_ratio_metrics(trainer)
        self.assertAlmostEqual(metrics["st/ratio_mean"], 3.0, places=6)
        self.assertIsNone(trainer._train_ratio_sum)
        self.assertIsNone(trainer._train_ratio_count)
        self.assertEqual(LogRatioTeacherTrainer._pop_train_ratio_metrics(trainer), {})

    def test_diagnostic_log_does_not_consume_parent_logging_trigger(self):
        from transformers import Trainer

        from train.opsd.train_self_teacher.train_logratio_teacher import LogRatioTeacherTrainer

        trainer = self.make_trainer()
        trainer.control = types.SimpleNamespace(should_log=True)
        trainer.diagnostics = lambda: {"dashboard": 1.0}
        trainer._pop_train_ratio_metrics = lambda: {"st/ratio_mean": 2.0}
        logged = []

        def consume_trigger(logs):
            logged.append(logs)
            trainer.control.should_log = False

        trainer.log = consume_trigger
        parent_saw_should_log = []

        def parent(*args, **kwargs):
            parent_saw_should_log.append(trainer.control.should_log)
            return "parent-result"

        with mock.patch.object(Trainer, "_maybe_log_save_evaluate", side_effect=parent):
            result = LogRatioTeacherTrainer._maybe_log_save_evaluate(trainer)

        self.assertEqual(result, "parent-result")
        self.assertEqual(parent_saw_should_log, [True])
        self.assertEqual(logged, [{"st/dashboard": 1.0, "st/ratio_mean": 2.0}])


class DiagnosticsTest(unittest.TestCase):
    def test_flattening_reduces_retained_dispersion_to_zero(self):
        initial = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
        current = torch.full_like(initial, 0.3)
        metrics = calibration_metrics(
            current, torch.ones_like(current), torch.tensor([1.0, 0.0]),
            initial_ratios=initial,
        )
        self.assertLess(metrics["ratio_dispersion_retained"], 1e-6)

    def test_unchanged_teacher_starts_at_unit_retention_and_zero_drift(self):
        ratios = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
        metrics = calibration_metrics(
            ratios, torch.ones_like(ratios), torch.tensor([1.0, 0.0]),
            initial_ratios=ratios,
        )
        self.assertAlmostEqual(metrics["ratio_dispersion_retained"], 1.0, places=6)
        self.assertAlmostEqual(metrics["wrong_ratio_delta"], 0.0, places=6)
        self.assertAlmostEqual(metrics["wrong_ratio_rms_drift"], 0.0, places=6)

    def test_late_firing_ratio_concentrates_credit_in_the_last_quartile(self):
        ratios = torch.zeros(1, 8)
        ratios[0, 6:] = 1.0
        metrics = calibration_metrics(ratios, torch.ones_like(ratios), torch.tensor([1.0]))
        self.assertAlmostEqual(metrics["credit_mass_last_quartile"], 1.0, places=6)

    def test_evenly_spread_credit_scores_a_quarter(self):
        ratios = torch.full((1, 8), 0.25)
        metrics = calibration_metrics(ratios, torch.ones_like(ratios), torch.tensor([1.0]))
        self.assertAlmostEqual(metrics["credit_mass_last_quartile"], 0.25, places=6)

    def test_failure_dashboard_reports_signed_and_rms_drift(self):
        initial = torch.zeros(3, 2)
        current = torch.tensor([
            [9.0, 9.0],  # correct: excluded
            [1.0, -1.0],  # wrong: mean delta 0, RMS 1
            [2.0, 2.0],  # wrong: mean delta 2, RMS 2
        ])
        metrics = calibration_metrics(
            current, torch.ones_like(current), torch.tensor([1.0, 0.0, 0.0]),
            initial_ratios=initial,
        )
        self.assertAlmostEqual(metrics["wrong_ratio_delta"], 1.0, places=6)
        self.assertAlmostEqual(metrics["wrong_ratio_rms_drift"], 1.5, places=6)

    def test_correct_relief_uses_all_initially_penalized_successes(self):
        initial = torch.tensor([
            [-2.0, -2.0], [0.0, 0.0],  # q1: correct centered at -1
            [-4.0, -4.0], [0.0, 0.0],  # q2: correct centered at -2
        ])
        current = torch.tensor([
            [-1.0, -1.0], [0.0, 0.0],  # centered relief +0.5
            [-2.0, -2.0], [0.0, 0.0],  # centered relief +1.0
        ])
        metrics = calibration_metrics(
            current,
            torch.ones_like(current),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            question_ids=["q1", "q1", "q2", "q2"],
            initial_ratios=initial,
        )
        self.assertEqual(metrics["correct_penalized_count"], 2.0)
        self.assertAlmostEqual(metrics["correct_penalty_relief"], 5.0 / 6.0, places=6)
        self.assertAlmostEqual(metrics["wrong_ratio_delta"], 0.0, places=6)
        self.assertAlmostEqual(metrics["wrong_ratio_rms_drift"], 0.0, places=6)

    def test_empty_batch_returns_nothing_rather_than_dividing_by_zero(self):
        self.assertEqual(
            calibration_metrics(torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([1.0])), {}
        )


class FittedCalibrationTest(unittest.TestCase):
    """Fitted Brier and within-question AUC are the compact outcome dashboard."""

    def make_batch(self, n=64, seed=0):
        torch.manual_seed(seed)
        return (torch.rand(n) < 0.4).float()

    def question_ids(self, n):
        return [f"q{i // 4}" for i in range(n)]

    def test_flat_ratio_fits_to_the_floor(self):
        reward = self.make_batch()
        ratios = torch.full((reward.numel(), 8), 0.3)
        metrics = calibration_metrics(ratios, torch.ones_like(ratios), reward)

        self.assertAlmostEqual(
            metrics["brier_q100_fitted"], metrics["brier_floor_crossfit"], places=6,
            msg="a flat ratio must fit exactly to the base-rate floor",
        )

    def test_fitted_brier_and_within_auc_are_scale_invariant(self):
        reward = self.make_batch()
        torch.manual_seed(1)
        base = 0.2 * (reward.unsqueeze(1) - 0.4) + 0.2 * torch.randn(reward.numel(), 6)
        mask = torch.ones_like(base)
        questions = self.question_ids(reward.numel())

        small = calibration_metrics(base, mask, reward, question_ids=questions)
        large = calibration_metrics(10.0 * base, mask, reward, question_ids=questions)

        self.assertAlmostEqual(small["brier_q100_fitted"], large["brier_q100_fitted"], places=4)
        self.assertAlmostEqual(
            small["within_question_auc_q100"], large["within_question_auc_q100"], places=6
        )

    def test_informative_ratio_beats_the_floor(self):
        reward = self.make_batch(n=128)
        torch.manual_seed(2)
        ratios = (0.6 * (reward.unsqueeze(1) - 0.4)).expand(-1, 6) + 0.05 * torch.randn(
            reward.numel(), 6
        )
        metrics = calibration_metrics(
            ratios, torch.ones_like(ratios), reward,
            question_ids=self.question_ids(reward.numel()),
        )

        self.assertLess(metrics["brier_q100_fitted"], metrics["brier_floor_crossfit"])
        self.assertGreater(metrics["within_question_auc_q100"], 0.9)

    def test_single_class_batch_omits_outcome_metrics(self):
        reward = torch.ones(8)
        ratios = torch.randn(8, 5)
        metrics = calibration_metrics(
            ratios, torch.ones_like(ratios), reward,
            question_ids=self.question_ids(reward.numel()),
        )

        self.assertEqual(metrics["mixed_question_count"], 0.0)
        for key in (
            "brier_floor_crossfit", "brier_q100_fitted", "within_question_auc_q100"
        ):
            self.assertNotIn(key, metrics)

    def test_perfect_separation_stays_finite(self):
        reward = torch.cat([torch.zeros(16), torch.ones(16)])
        ratios = torch.cat([-torch.ones(16, 4), torch.ones(16, 4)])
        questions = [f"q{i % 16}" for i in range(reward.numel())]
        metrics = calibration_metrics(
            ratios, torch.ones_like(ratios), reward, question_ids=questions
        )

        self.assertTrue(math.isfinite(metrics["brier_q100_fitted"]))
        self.assertAlmostEqual(metrics["within_question_auc_q100"], 1.0, places=6)

    def test_question_difficulty_is_neutral_within_question(self):
        ratios, reward, questions = [], [], []
        examples = [
            (2.0, [1, 1, 1, 0]),
            (-2.0, [1, 0, 0, 0]),
            (1.0, [1, 1, 0, 0]),
            (-1.0, [1, 0, 0, 0]),
        ]
        for question_idx, (difficulty, outcomes) in enumerate(examples):
            for outcome in outcomes:
                ratios.append([difficulty] * 4)
                reward.append(outcome)
                questions.append(f"q{question_idx}")
        ratios = torch.tensor(ratios)
        reward = torch.tensor(reward, dtype=torch.float32)
        metrics = calibration_metrics(
            ratios, torch.ones_like(ratios), reward, question_ids=questions
        )

        self.assertAlmostEqual(metrics["within_question_auc_q100"], 0.5, places=6)
        self.assertEqual(metrics["mixed_question_count"], 4.0)
        for q in (25, 50, 75, 100):
            self.assertIn(f"within_question_auc_q{q}", metrics)

    def test_reversed_ratio_cannot_be_rescued_by_platt_scaling(self):
        reward = torch.cat([torch.zeros(16), torch.ones(16)])
        ratios = torch.cat([torch.ones(16, 4), -torch.ones(16, 4)])
        questions = [f"q{i % 16}" for i in range(reward.numel())]
        metrics = calibration_metrics(
            ratios, torch.ones_like(ratios), reward, question_ids=questions
        )

        self.assertAlmostEqual(metrics["within_question_auc_q100"], 0.0, places=6)
        self.assertAlmostEqual(
            metrics["brier_q100_fitted"], metrics["brier_floor_crossfit"], places=6
        )

    def test_padding_does_not_shift_prefix_quantiles(self):
        reward = torch.tensor([float(i % 2) for i in range(32)])
        signal = 2.0 * reward - 1.0
        short = signal.unsqueeze(1).expand(-1, 2).clone()
        padded = torch.cat([short, torch.zeros(32, 2)], dim=1)
        short_mask = torch.ones_like(short)
        padded_mask = torch.cat([short_mask, torch.zeros(32, 2)], dim=1)
        questions = [f"q{i // 2}" for i in range(32)]

        short_metrics = calibration_metrics(
            short, short_mask, reward, question_ids=questions
        )
        padded_metrics = calibration_metrics(
            padded, padded_mask, reward, question_ids=questions
        )
        for q in (25, 50, 75, 100):
            self.assertAlmostEqual(
                short_metrics[f"brier_q{q}_fitted"],
                padded_metrics[f"brier_q{q}_fitted"],
                places=6,
            )
            self.assertAlmostEqual(
                short_metrics[f"within_question_auc_q{q}"],
                padded_metrics[f"within_question_auc_q{q}"],
                places=6,
            )

    def test_group_auc_macro_averages_only_mixed_questions(self):
        scores = torch.tensor([1.0, 1.0, -1.0, -1.0, 3.0, 3.0])
        reward = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        groups = ["a", "a", "b", "b", "all-correct", "all-correct"]
        auc, count = macro_group_rank_auc(scores, reward, groups)
        self.assertAlmostEqual(auc, 0.5, places=6)
        self.assertEqual(count, 2)

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


class QuestionSplitTest(unittest.TestCase):
    def make_dataset(self):
        from datasets import Dataset

        return Dataset.from_list([
            {"question": f"q{question}", "reward": float(rollout % 2)}
            for question in range(20)
            for rollout in range(4)
        ])

    def test_split_is_deterministic_disjoint_and_keeps_questions_whole(self):
        from train.opsd.train_self_teacher.train_logratio_teacher import split_question_groups

        dataset = self.make_dataset()
        train, validation, held_out = split_question_groups(dataset, 0.2, seed=17)
        train_questions = set(train["question"])
        validation_questions = set(validation["question"])
        self.assertFalse(train_questions & validation_questions)
        self.assertEqual(validation_questions, held_out)
        self.assertEqual(len(train) + len(validation), len(dataset))
        self.assertTrue(all(validation["question"].count(q) == 4 for q in held_out))

        train_again, validation_again, held_out_again = split_question_groups(dataset, 0.2, seed=17)
        self.assertEqual(held_out, held_out_again)
        self.assertEqual(train["question"], train_again["question"])
        self.assertEqual(validation["question"], validation_again["question"])

    def test_diagnostic_cap_never_splits_a_question(self):
        from train.opsd.train_self_teacher.train_logratio_teacher import select_complete_diagnostic_questions

        dataset = self.make_dataset()
        diagnostic, questions = select_complete_diagnostic_questions(dataset, max_rows=10, seed=3)
        self.assertLessEqual(len(diagnostic), 10)
        self.assertEqual(set(diagnostic["question"]), set(questions))
        self.assertTrue(all(diagnostic["question"].count(q) == 4 for q in questions))


class AsymmetricQuestionWeightTest(unittest.TestCase):
    """Static row weights must recover a macro-average over eligible questions."""

    @staticmethod
    def make_dataset(rows):
        from datasets import Dataset

        return Dataset.from_list(rows)

    def test_each_eligible_question_gets_equal_total_weight(self):
        from train.opsd.train_self_teacher.train_logratio_teacher import (
            add_asymmetric_lift_weights,
        )

        dataset = self.make_dataset(
            [
                {"question": "q1", "reward": 1.0, "student_logps": [0.0], "teacher_logps_init": [-0.4]},
                {"question": "q1", "reward": 1.0, "student_logps": [0.0], "teacher_logps_init": [-0.1]},
                {"question": "q2", "reward": 1.0, "student_logps": [0.0], "teacher_logps_init": [-0.2]},
                {"question": "q3", "reward": 0.0, "student_logps": [0.0], "teacher_logps_init": [-0.5]},
            ]
        )
        weighted = add_asymmetric_lift_weights(dataset, margin=0.0)
        weights = weighted["asym_lift_weight"]
        self.assertEqual(weights, [1.0, 1.0, 2.0, 0.0])
        self.assertAlmostEqual(sum(weights[:2]), weights[2])
        self.assertAlmostEqual(sum(weights), len(dataset))

    def test_no_eligible_question_produces_zero_lift_weights(self):
        from train.opsd.train_self_teacher.train_logratio_teacher import add_asymmetric_lift_weights

        dataset = self.make_dataset(
            [
                {"question": "q1", "reward": 1.0, "student_logps": [0.0], "teacher_logps_init": [0.0]},
                {"question": "q2", "reward": 0.0, "student_logps": [0.0], "teacher_logps_init": [-0.5]},
            ]
        )
        weighted = add_asymmetric_lift_weights(dataset, margin=0.0)
        self.assertEqual(weighted["asym_lift_weight"], [0.0, 0.0])


class StageThreeResumeTest(unittest.TestCase):
    """A student cannot be resumed into a run whose teacher is a different object."""

    def make_args(self, pi_mode="hint"):
        return types.SimpleNamespace(
            model="Qwen/Qwen3-4B", dataset="deepmath", max_samples=None, pi_mode=pi_mode,
            teacher_path="/tmp/teacher/final", distillation_mode="sampled_token",
            distillation_alpha=1.0, learning_rate=1e-5, seed=42,
            gradient_accumulation_steps=16, num_generations=1, max_prompt_length=8192,
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
        meta = build_run_meta(args, {"teacher_version": TEACHER_VERSION, "objective": "asymmetric"}, 100)
        legacy = {k: v for k, v in meta.items() if k != "teacher_version"}
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, legacy)
            with self.assertRaisesRegex(ValueError, "teacher_version"):
                validate_resume(ckpt, meta, strict_keys=("teacher_version",))

    def test_a_different_teacher_objective_is_a_mismatch(self):
        from train.opsd.train_self_teacher.sdft_with_teacher import build_run_meta

        args = self.make_args()
        per_token = build_run_meta(args, {"teacher_version": TEACHER_VERSION, "objective": "asymmetric"}, 100)
        aggregate = build_run_meta(args, {"teacher_version": TEACHER_VERSION, "objective": "asymmetric_aggregate"}, 100)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, per_token)
            with self.assertRaisesRegex(ValueError, "teacher_objective"):
                validate_resume(ckpt, aggregate, strict_keys=("teacher_version",))

    def test_matching_run_resumes(self):
        from train.opsd.train_self_teacher.sdft_with_teacher import build_run_meta

        args = self.make_args()
        meta = build_run_meta(args, {"teacher_version": TEACHER_VERSION, "objective": "asymmetric"}, 100)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, dict(meta))
            validate_resume(ckpt, meta, strict_keys=("teacher_version",))

    def test_physical_forward_batch_is_fixed_to_one(self):
        from train.opsd.train_self_teacher.sdft_with_teacher import build_run_meta

        meta = build_run_meta(
            self.make_args(), {"teacher_version": TEACHER_VERSION, "objective": "asymmetric"}, 100
        )
        self.assertEqual(meta["per_device_train_batch_size"], 1)

    def test_none_mode_records_the_concatenating_template(self):
        from train.opsd.train_self_teacher.sdft_with_teacher import build_run_meta

        meta = build_run_meta(
            self.make_args("none"),
            {"teacher_version": TEACHER_VERSION, "objective": "asymmetric"}, 100,
        )
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
