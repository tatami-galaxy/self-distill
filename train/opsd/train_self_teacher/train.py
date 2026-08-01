r"""
Stage 2 of the trained-self-teacher arm: regress the teacher:student log ratio onto the verified
outcome.

It reads the frozen-student rollouts cached by
train/opsd/train_self_teacher/gen_rollouts.py, attaches a privileged context, and trains a copy of the student -- the
teacher -- so that

    rho_t = log pi_phi(y_t | x, c, y_<t) - log pi_theta(y_t | x, y_<t)

predicts whether the rollout it came from was correct. Two objectives (see train/opsd/train_self_teacher/lib.py
for the full rationale):

  --objective pointwise : regress each rho_t towards +/- tau.
                          Its fixed point is a flat advantage, so it is meant to be run as a
                          bounded nudge, not to convergence. Watch `ratio_dispersion_retained`.
  --objective endpoint  : regress only the trace total sigmoid(beta*S_T/N + b).
                          One constraint per trace instead of one per token, so per-token allocation
                          stays free and credit can be sparse. Gentler, cannot flatten.

Stage 3 is only worth if this stage shows the ratio became outcome-predictive :
INIT vs TRAINED, printed at startup and written to diagnostics_init.json.
Diagnostics use COMPLETE HELD-OUT QUESTIONS: no rollout from a diagnostic question reaches the optimizer.

  brier_q*_fitted              question-grouped out-of-fold outcome Brier; compare with
                               brier_floor_crossfit.
  within_question_auc_q*       shortcut-resistant ranking within mixed-outcome questions.
  correct_penalty_relief       weighted question-centered relief on every initially penalized
                               successful rollout; correct_penalized_count is its support.
  wrong_ratio_delta / wrong_ratio_rms_drift
                               signed and magnitude drift from initialization on failures.
  ratio_dispersion_retained    fraction of the initial within-trace credit structure retained.
  credit_mass_last_quartile    late-firing detector; `--pi-mode answer` is expected to trip it.

# one E-step, hint PI, pointwise objective
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_self_teacher.train \
    --model Qwen/Qwen3-4B --dataset deepmath --pi-mode hint --objective pointwise

# the full ladder off the one rollout cache
for pi in hint answer full none; do for obj in pointwise endpoint; do
  CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_self_teacher.train \
      --model Qwen/Qwen3-4B --pi-mode $pi --objective $obj; done; done

# smoke
CUDA_VISIBLE_DEVICES=0 uv run python -m train.opsd.train_self_teacher.train \
    --model Qwen/Qwen3-1.7B --pi-mode none --max-steps 2 --save-steps 2 --diag-rows 4
"""

import argparse
import hashlib
import json
import os
from functools import partial

import torch
from datasets import load_from_disk
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from train.opsd.train_self_teacher.lib import (
    LENGTH_NORMS,
    MODEL_FORWARD_BATCH_SIZE,
    PI_MODES,
    TEACHER_VERSION,
    build_teacher_dataset,
    calibration_metrics,
    collate_teacher_batch,
    concat_padded,
    log_ratio,
    objective_endpoint,
    objective_pointwise,
    rollout_path,
    teacher_prompt_template,
    teacher_token_logps,
)
from utils import DATASET_REGISTRY_TRAIN, validate_resume


# The learnable calibration bias, checkpointed beside the teacher. A bare state dict rather than
# part of the model, for the same reason train_ppo.py keeps its critic out of `self.model`: the
# teacher must stay a plain PreTrainedModel so `save_pretrained` produces something
# eval/teacher_behavior.py and stage 3 can load directly.
CALIBRATION_FILE = "calibration.pt"

# Resume identity for the data boundary: older checkpoints trained on rows now held out.
QUESTION_SPLIT_VERSION = "question_hash_v1"


class SelfTeacherTrainer(Trainer):
    """HF Trainer whose loss is the outcome regression on the teacher:student log ratio."""

    def __init__(self, *args, objective, pointwise_loss, tau, beta, length_norm,
                 bias_learning_rate, kl_anchor, diag_crossfit_folds=5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.objective = objective
        self.pointwise_loss = pointwise_loss
        self.tau = tau
        self.beta = beta
        self.length_norm = length_norm
        self.bias_learning_rate = bias_learning_rate
        self.kl_anchor = kl_anchor
        self.diag_crossfit_folds = diag_crossfit_folds
        # VPD's delta: one scalar absorbing the outcome base rate so the ratio does not have to
        # encode it, and the "shift every rho by a constant" degenerate solution is unavailable.
        self.calibration_bias = nn.Parameter(
            torch.zeros((), device=self.accelerator.device, dtype=torch.float32)
        )
        self._diag_rows = None
        self._diag_collate = None
        self._diag_question_ids = None
        self._diag_initial_ratios = None
        self._train_ratio_sum = None
        self._train_ratio_count = None

    # -- the bias needs its own, much larger learning rate ---------------------

    def create_optimizer(self, *args, **kwargs):
        """Append the calibration bias as its own param group.

        Set HERE rather than patched onto param_groups afterwards, so `create_scheduler` (which
        runs next and records each group's `initial_lr`) schedules it from its own base rate --
        the same reason train_ppo.py adds its critic group inside create_optimizer.

        The separate rate is not a nicety: at the teacher's 1e-6 a single scalar would take tens
        of thousands of steps to travel the couple of nats that a 60%-vs-40% base rate implies,
        so the ratio would spend the whole run compensating for it.
        """
        optimizer = super().create_optimizer(*args, **kwargs)
        optimizer.add_param_group({"params": [self.calibration_bias], "lr": self.bias_learning_rate})

        # Transformers clears gradients with `model.zero_grad()` after each optimizer update. The
        # calibration scalar deliberately lives outside `model`, so without an explicit clear its
        # gradient would leak across optimizer steps forever. An optimizer post-step hook preserves
        # accumulation across all microbatches in THIS update and clears exactly at its boundary.
        bias = self.calibration_bias

        def clear_bias_grad(_optimizer, _args, _kwargs):
            bias.grad = None

        self._bias_grad_clear_hook = optimizer.register_step_post_hook(clear_bias_grad)
        return optimizer

    # -- loss ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("SelfTeacherTrainer does not support returning outputs")

        logps = teacher_token_logps(model, inputs)
        ratios = log_ratio(logps, inputs)
        mask = inputs["completion_mask"].float()
        reward = inputs["reward"]

        if self.objective == "pointwise":
            loss = objective_pointwise(
                ratios, mask, reward, tau=self.tau, loss=self.pointwise_loss, beta=self.beta
            )
        else:
            loss = objective_endpoint(
                ratios, mask, reward, self.beta, self.calibration_bias, self.length_norm
            )

        if self.kl_anchor > 0.0:
            # Proximal term against the teacher AT INIT, keeping it usable as an imitation target
            # in stage 3. Needed because the R=0 branch of either objective is unlikelihood
            # training on PI-conformant tokens, which can improve the ratio while degrading the
            # text the teacher would generate.
            drift = (logps - inputs["teacher_logps_init"]) ** 2
            loss = loss + self.kl_anchor * (drift * mask).sum() / mask.sum().clamp(min=1.0)

        if model.training:
            self._accumulate_train_ratio(ratios, mask)
        return loss

    @torch.no_grad()
    def _accumulate_train_ratio(self, ratios: torch.Tensor, mask: torch.Tensor) -> None:
        """Accumulate `st/ratio_mean` over the logging interval: the mean over TRAJECTORIES of
        each trace's own token mean.

        Equal per trajectory, not per token. Completions here span ~2.5k-8.2k tokens, so a
        token-weighted mean would report mostly what the longest traces are doing -- and length
        is not incidental in this cache: failures are overwhelmingly budget truncations, so
        token-weighting would tilt the reading towards the R=0 rows. This weighting is also the
        one `calibration_metrics` uses for its `row_mean`, so the training-interval reading and
        the held-out dashboard are the SAME quantity and their drift is comparable.

        Accumulated here rather than logged inline because `self.control.should_log` is always
        False by the time compute_loss runs: the flag is set at on_step_end and cleared by the
        first `self.log()` of the step that follows it, so an `if should_log` guard in this
        method never fires (it silently logged nothing for the whole of the first 4B sweep).
        Summing into a tensor also keeps the per-microbatch cost to one small reduction and
        defers the cross-process sync to `_pop_train_ratio_metrics`, once per log."""
        lengths = mask.sum(dim=1)
        valid = lengths > 0
        if not valid.any():
            return
        row_means = (ratios * mask).sum(dim=1) / lengths.clamp(min=1.0)
        ratio_sum = row_means[valid].sum().detach()
        ratio_count = valid.sum().to(dtype=ratio_sum.dtype).detach()
        if self._train_ratio_sum is None:
            self._train_ratio_sum = ratio_sum.clone()
            self._train_ratio_count = ratio_count.clone()
        else:
            self._train_ratio_sum.add_(ratio_sum)
            self._train_ratio_count.add_(ratio_count)

    @torch.no_grad()
    def _pop_train_ratio_metrics(self) -> dict[str, float]:
        """Reduce and reset interval training-ratio accumulators."""
        if self._train_ratio_sum is None:
            return {}
        ratio_sum = self.accelerator.reduce(self._train_ratio_sum, reduction="sum")
        ratio_count = self.accelerator.reduce(self._train_ratio_count, reduction="sum")
        self._train_ratio_sum = None
        self._train_ratio_count = None
        return {
            "st/ratio_mean": (ratio_sum / ratio_count.clamp(min=1.0)).item(),
            "st/calibration_bias": self.calibration_bias.item(),
        }

    # -- diagnostics -----------------------------------------------------------

    def set_diagnostic_rows(self, rows: list[dict], collate) -> None:
        """Fix the traces the diagnostics are read on. Stored as ROWS, not as one collated
        batch: `diagnostics` collates each chunk at its own width so the forwards stay
        unpadded (see self_teacher.per_token_logps)."""
        self._diag_rows = rows
        self._diag_collate = collate
        self._diag_question_ids = [row["question"] for row in rows]
        # The first reading on these fixed rows becomes every later drift metric's reference.
        self._diag_initial_ratios = None

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        """No-grad forward over the fixed diagnostic rows -> the metrics block.

        FIXED rows, so every reading is on the same traces and a change in a metric is a change
        in the teacher rather than a change in the sample.

        Each row is collated and forwarded separately :
          * unpadded. In our Qwen3-1.7B measurements, changing the padded input shape shifted rho
            by ~0.03 nats in bf16 -- a third of --tau. (see self_teacher.per_token_logps).
          * bounded. One forward over all --diag-rows would materialize a huge (rows, C, vocab)
            logit tensor; no_grad does not help, since it must exist to be reduced.
        Per-row results are padded and assembled AFTERWARDS, which is lossless because
        every metric below is mask-weighted.
        """
        if not self._diag_rows:
            return {}
        device = self.accelerator.device
        was_training = self.model.training
        self.model.eval()
        ratio_chunks, mask_chunks, reward_chunks = [], [], []
        try:
            for start in range(0, len(self._diag_rows), MODEL_FORWARD_BATCH_SIZE):
                chunk = self._diag_rows[start : start + MODEL_FORWARD_BATCH_SIZE]
                batch = {k: v.to(device) for k, v in self._diag_collate(chunk).items()}
                logps = teacher_token_logps(self.model, batch)
                # teacher-student log ratio
                ratio_chunks.append(log_ratio(logps, batch))
                mask_chunks.append(batch["completion_mask"].float())
                reward_chunks.append(batch["reward"])
        finally:
            self.model.train(was_training)
        ratios = concat_padded(ratio_chunks)
        mask = concat_padded(mask_chunks)
        reward = torch.cat(reward_chunks)

        if self._diag_initial_ratios is None:
            # CPU storage avoids permanently consuming accelerator memory between log steps.
            self._diag_initial_ratios = ratios.detach().cpu().clone()
        return calibration_metrics(
            ratios, mask, reward,
            length_norm=self.length_norm,
            crossfit_folds=self.diag_crossfit_folds,
            question_ids=self._diag_question_ids,
            initial_ratios=self._diag_initial_ratios,
        )

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        should_log = self.control.should_log
        if should_log:
            logs = {f"st/{k}": v for k, v in self.diagnostics().items()}
            logs.update(self._pop_train_ratio_metrics())
            self.log(logs)
            # CallbackHandler.on_log consumes this trigger. Restore it so Trainer logs and resets
            # its interval loss, grad norm, learning rate, FLOPs, and global-step bookkeeping.
            self.control.should_log = True
        return super()._maybe_log_save_evaluate(*args, **kwargs)

    # -- checkpoint the bias alongside the teacher -----------------------------

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            ckpt_dir = os.path.join(
                self._get_output_dir(trial), f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            )
            torch.save({"calibration_bias": self.calibration_bias.detach().cpu()},
                       os.path.join(ckpt_dir, CALIBRATION_FILE))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        path = os.path.join(resume_from_checkpoint, CALIBRATION_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No {CALIBRATION_FILE} in {resume_from_checkpoint}. The calibration bias absorbs "
                "the outcome base rate, so resuming without it would restart from 0 while the "
                "teacher's ratio still assumes the learned offset -- every diagnostic would look "
                "healthy and every value would be shifted."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        # In place: the optimizer holds a reference to this exact tensor.
        with torch.no_grad():
            self.calibration_bias.copy_(state["calibration_bias"].to(self.calibration_bias.device))
        print(f"Restored calibration bias <- {path}")


# ---------------------------------------------------------------------------
# Init-teacher pre-pass for --kl-anchor
# ---------------------------------------------------------------------------


def add_init_teacher_logps(dataset, model, collate, device) -> "Dataset":
    """Score every row under the UNTRAINED teacher, so the anchor has something to pull towards.

    One no-grad pass at startup. Only run when --kl-anchor > 0; otherwise the column is absent
    and the collator omits it.
    """
    print(f"  --kl-anchor: scoring {len(dataset)} rows under the initial teacher")
    was_training = model.training
    model.eval()
    all_logps: list[list[float]] = []
    try:
        for start in range(0, len(dataset), MODEL_FORWARD_BATCH_SIZE):
            rows = [dataset[start]]
            batch = {k: v.to(device) for k, v in collate(rows).items()}
            with torch.no_grad():
                logps = teacher_token_logps(model, batch).float().cpu()
            for row_idx, row in enumerate(rows):
                all_logps.append(logps[row_idx, : len(row["completion_ids"])].tolist())
    finally:
        model.train(was_training)
    return dataset.add_column("teacher_logps_init", all_logps)


# ---------------------------------------------------------------------------
# Question-grouped held-out diagnostics
# ---------------------------------------------------------------------------


def _question_hash(question: str, seed: int, salt: str = "split") -> bytes:
    return hashlib.sha256(f"{salt}:{seed}:{question}".encode()).digest()


def split_question_groups(dataset, validation_fraction: float, seed: int):
    """Split an HF dataset without allowing any question to cross the boundary.

    Membership is a hash threshold rather than a shuffle position, so an overlapping question is
    assigned identically across PI arms even when `full` drops additional overlength prompts.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    questions = sorted(set(dataset["question"]), key=lambda q: _question_hash(q, seed))
    if len(questions) < 2:
        raise ValueError("question-group validation requires at least two distinct questions")

    threshold = int(validation_fraction * (1 << 256))
    validation_questions = {
        question
        for question in questions
        if int.from_bytes(_question_hash(question, seed), "big") < threshold
    }
    # Small smoke datasets can miss a probabilistic threshold entirely. Preserve determinism while
    # guaranteeing that both the optimizer and the diagnostics receive at least one question.
    if not validation_questions:
        validation_questions = {questions[0]}
    elif len(validation_questions) == len(questions):
        validation_questions.remove(questions[-1])

    train_indices, validation_indices = [], []
    for idx, question in enumerate(dataset["question"]):
        target = validation_indices if question in validation_questions else train_indices
        target.append(idx)
    return dataset.select(train_indices), dataset.select(validation_indices), validation_questions


def select_complete_diagnostic_questions(dataset, max_rows: int, seed: int):
    """Cap diagnostic cost while retaining every rollout of each selected question."""
    if max_rows <= 0:
        raise ValueError("diag_rows must be positive")
    rows_by_question: dict[str, list[int]] = {}
    for idx, question in enumerate(dataset["question"]):
        rows_by_question.setdefault(question, []).append(idx)
    ordered = sorted(
        rows_by_question, key=lambda q: _question_hash(q, seed, salt="diagnostic")
    )
    selected: list[int] = []
    selected_questions: list[str] = []
    for question in ordered:
        rows = rows_by_question[question]
        if selected and len(selected) + len(rows) > max_rows:
            continue
        selected.extend(rows)
        selected_questions.append(question)
        if len(selected) >= max_rows:
            break
    return dataset.select(sorted(selected)), selected_questions


def count_mixed_questions(dataset) -> int:
    outcomes: dict[str, set[float]] = {}
    for question, reward in zip(dataset["question"], dataset["reward"], strict=True):
        outcomes.setdefault(question, set()).add(float(reward))
    return sum(len(values) > 1 for values in outcomes.values())


# ---------------------------------------------------------------------------
# Resume metadata
# ---------------------------------------------------------------------------


def build_run_meta(
    args,
    num_train_examples: int,
    num_validation_examples: int = 0,
    num_diag_examples: int = 0,
    num_train_questions: int = 0,
    num_validation_questions: int = 0,
    num_diag_questions: int = 0,
) -> dict:
    return {
        "method": "self_teacher_estep",
        # Teacher-state identity. Changing the objective family or the ratio definition changes
        # what these weights MEAN, so an older checkpoint is not resumable into a newer one.
        # Passed to validate_resume's strict_keys, which is what makes a checkpoint that never
        # recorded it a hard error rather than a silently skipped comparison -- the same
        # mechanism and rationale as train_ppo_val.py's value_prompt_version.
        "teacher_version": TEACHER_VERSION,
        "model": args.model,
        "dataset": args.dataset,
        "pi_mode": args.pi_mode,
        # Hint provenance: hints are only "self" if the trained model wrote them.
        "gen_model": args.model if args.pi_mode == "hint" else None,
        "rollout_cache": rollout_path(args.model, args.dataset, args.rollout_root),
        # The literal the teacher's context was stitched with. Stage 3 reads this back and
        # refuses to run against a different one: the teacher would be scoring a context it was
        # never trained on, differing by nothing more visible than two newlines.
        "teacher_prompt_template": teacher_prompt_template(args.pi_mode),
        "num_train_examples": num_train_examples,
        # The objective and its shape parameters: all of them change what the ratio converges to.
        "num_validation_examples": num_validation_examples,
        "num_diag_examples": num_diag_examples,
        "num_train_questions": num_train_questions,
        "num_validation_questions": num_validation_questions,
        "num_diag_questions": num_diag_questions,
        "question_split_version": QUESTION_SPLIT_VERSION,
        # Resume-critical: these keys determine exactly which rows reached the optimizer.
        "validation_fraction": getattr(args, "validation_fraction", None),
        "validation_seed": getattr(args, "validation_seed", None),
        "diag_crossfit_folds": getattr(args, "diag_crossfit_folds", None),
        "objective": args.objective,
        "diag_rows_limit": getattr(args, "diag_rows", None),
        "pointwise_loss": args.pointwise_loss,
        "tau": args.tau,
        "beta": args.beta,
        "length_norm": args.length_norm,
        "kl_anchor": args.kl_anchor,
        "max_teacher_prompt_length": args.max_teacher_prompt_length,
        # resume-critical, learning rates: on resume these come from the CHECKPOINT, not the CLI
        # (optimizer.load_state_dict restores each group's lr AFTER create_scheduler), so
        # recording them turns a silently-ignored change into an error.
        "learning_rate": args.learning_rate,
        "bias_learning_rate": args.bias_learning_rate,
        # resume-critical: dataset order + batch chunking.
        "seed": args.seed,
        "per_device_train_batch_size": MODEL_FORWARD_BATCH_SIZE,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="The student. The teacher is initialised from these same weights, and the "
                        "rollout cache must have been generated by them.")
    p.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN.keys()))
    p.add_argument("--rollout-root", default="data/rollouts",
                   help="Where train/opsd/train_self_teacher/gen_rollouts.py wrote its cache.")
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/self_teacher")
    p.add_argument("--output-dir", default=None,
                   help="Override; defaults to <output-root>/<model>/<dataset>_<pi-mode>_<objective>")
    p.add_argument("--max-samples", type=int, default=None, help="Subset the rollout cache")
    # privileged context
    p.add_argument("--pi-mode", default="hint", choices=list(PI_MODES),
                   help="What the TEACHER sees on top of the question (the student never sees it, "
                        "and the rollouts were sampled without it): 'hint' precomputed self-hints, "
                        "'answer' the boxed gold, 'full' the worked solution, 'none' the matched "
                        "control where rho starts at exactly 0 everywhere.")
    p.add_argument("--max-teacher-prompt-length", type=int, default=8192,
                   help="Rows whose teacher prompt exceeds this are DROPPED, not truncated: the "
                        "user turn is '{question}\\n\\n{PI}', so left-truncation would remove the "
                        "question and the chat template header and keep the demo. Binding for "
                        "--pi-mode full only.")
    # objective
    p.add_argument("--objective", default="pointwise", choices=["pointwise", "endpoint"],
                   help="pointwise regresses every rho_t towards +/-tau; "
                        "endpoint regresses only the trace total.")
    p.add_argument("--pointwise-loss", default="squared", choices=["squared", "logistic"],
                   help="For --objective pointwise. 'squared' targets a bounded +/-tau on the RAW "
                        "ratio (finite fixed point). 'logistic' is BCE on beta*rho, whose gradient "
                        "saturates at a constant rather than vanishing -- use it if the bounded "
                        "target is reached before the ratio becomes informative.")
    p.add_argument("--tau", type=float, default=0.1,
                   help="Target magnitude in nats per token for --pointwise-loss squared.")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Scale on the ratio inside the sigmoid (endpoint / logistic).")
    p.add_argument("--length-norm", default="mean", choices=list(LENGTH_NORMS),
                   help="Divisor on S_T for --objective endpoint. 'mean' (per token) removes the "
                        "length dependence; with 'none' a fixed beta saturates the sigmoid on long "
                        "traces and silently trains on short ones only.")
    p.add_argument("--kl-anchor", type=float, default=0.0,
                   help="Proximal penalty pulling the teacher's logprobs back towards their values "
                        "at init, keeping it usable as stage 3's imitation target. Costs one "
                        "no-grad pass over the dataset at startup. 0 disables.")
    # optimization
    p.add_argument("--learning-rate", type=float, default=1e-6,
                   help="Teacher LR. Low by design: this is a corrective nudge to a pretrained "
                        "model, not SFT. Raise towards 1e-5 if the diagnostics do not move.")
    p.add_argument("--bias-learning-rate", type=float, default=1e-2,
                   help="LR for the single calibration-bias scalar, SEPARATE from the teacher's. "
                        "It has to travel a couple of nats to absorb the outcome base rate, which "
                        "at the teacher's rate would take the entire run.")
    p.add_argument("--lr-scheduler-type", default="constant",
                   choices=["linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optim", default="adamw_bnb_8bit",
                   help="Nothing here uses vLLM, so train_ppo.py's server-mode 8-bit hazard does "
                        "not apply (same reasoning as train_sft.py).")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    # diagnostics
    p.add_argument("--validation-fraction", type=float, default=0.1,
                   help="Fraction of QUESTIONS held out from teacher updates. Assignment is a "
                        "stable hash, so no rollout of a validation question can leak into train.")
    p.add_argument("--validation-seed", type=int, default=42,
                   help="Independent seed for the stable question split. Keep fixed across PI arms.")
    p.add_argument("--diag-rows", type=int, default=256,
                   help="Approximate maximum held-out rows scored at every log step. Complete "
                        "questions are retained, so the actual count may differ slightly. These "
                        "rows never reach the optimizer.")
    p.add_argument("--diag-crossfit-folds", type=int, default=5,
                   help="Question-grouped folds for out-of-fold Platt Brier diagnostics.")
    # bookkeeping
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    # resume
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume dir ('checkpoint-<step>'). Restores the teacher, the calibration "
                        "bias, the optimizer/scheduler/RNG, and skips seen rows. Pass the SAME "
                        "--model, --dataset, --pi-mode, --objective, --seed and gradient "
                        "accumulation as the original run (verified against its run_meta.json).")
    p.add_argument("--force-resume", action="store_true")
    args = p.parse_args()

    if not 0 < args.validation_fraction < 1:
        p.error("--validation-fraction must be strictly between 0 and 1")
    if args.diag_rows <= 0:
        p.error("--diag-rows must be positive")
    if args.diag_crossfit_folds < 2:
        p.error("--diag-crossfit-folds must be at least 2")

    if args.objective == "endpoint" and args.length_norm == "none":
        print("  warning: --objective endpoint with --length-norm none. S_T accumulates over "
              "thousands of tokens, so the sigmoid will saturate on long traces and their "
              "gradient will vanish. Use 'mean' unless you have rescaled --beta to match.")

    model_slug = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(
        args.output_root, model_slug, f"{args.dataset}_{args.pi_mode}_{args.objective}"
    )
    print(f"model: {model_slug}  dataset: {args.dataset}  pi: {args.pi_mode}  "
          f"objective: {args.objective}  ->  output: {output_dir}")

    # -- data ---------------------------------------------------------------
    cache_dir = rollout_path(args.model, args.dataset, args.rollout_root)
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"No rollout cache for {args.model} on {args.dataset} at {cache_dir}. Generate it "
            f"first:\n  python -m train.opsd.train_self_teacher.gen_rollouts --model {args.model} "
            f"--dataset {args.dataset}"
        )
    # question, final_answer, completion_ids, completion_text, reward, n_tokens,
    # gen_model, dataset, question_source, student_logps
    rollouts = load_from_disk(cache_dir)
    gen_models = set(rollouts.unique("gen_model"))
    if gen_models != {args.model}:
        raise ValueError(
            f"Rollout cache at {cache_dir} was generated by {gen_models}, not {args.model!r}. The "
            "rollouts would not be on-policy for this student, so the ratio would be measured "
            "against a distribution nothing produced."
        )
    if "student_logps" not in rollouts.column_names:
        raise ValueError(
            f"Rollout cache at {cache_dir} has no `student_logps` column. Run "
            f"`python -m train.opsd.train_self_teacher.gen_rollouts --model {args.model} "
            f"--dataset {args.dataset}` to complete the cache."
        )
    if args.max_samples is not None:
        rollouts = rollouts.select(range(min(args.max_samples, len(rollouts))))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_teacher_dataset(
        rollouts, args.pi_mode, tokenizer,
        dataset=args.dataset, max_teacher_prompt_length=args.max_teacher_prompt_length,
    )
    print(f"Loaded {len(train_dataset)} rollouts  "
          f"(pass rate {sum(train_dataset['reward']) / max(len(train_dataset), 1):.3f})")
    if args.pi_mode != "none" and not all(train_dataset["has_pi"]):
        missing = sum(1 for x in train_dataset["has_pi"] if not x)
        raise ValueError(
            f"{missing}/{len(train_dataset)} rows reached the teacher with an EMPTY privileged "
            f"context under --pi-mode {args.pi_mode}. The run would silently be its own control."
        )

    # Collect rollouts per question, hash questions
    # question A → hash → train      → all A rollouts train
    # question B → hash → validation → all B rollouts held out
    # question C → hash → train      → all C rollouts train
    train_dataset, validation_dataset, validation_questions = split_question_groups(
        train_dataset, args.validation_fraction, args.validation_seed
    )
    # Dont want to use full validation set for diagnostics
    # question A → rows [0, 1, 2, 3]
    # question B → rows [4, 5, 6, 7]
    diagnostic_dataset, diagnostic_questions = select_complete_diagnostic_questions(
        validation_dataset, args.diag_rows, args.validation_seed
    )
    train_questions = set(train_dataset["question"])
    n_mixed_diag = count_mixed_questions(diagnostic_dataset)
    print(
        f"Question split: train {len(train_questions)} questions/{len(train_dataset)} rollouts  |  "
        f"held out {len(validation_questions)} questions/{len(validation_dataset)} rollouts"
    )
    print(
        f"  monitoring {len(diagnostic_questions)} complete held-out questions/"
        f"{len(diagnostic_dataset)} rollouts ({n_mixed_diag} mixed-outcome questions)"
    )
    if n_mixed_diag == 0:
        print("  warning: the diagnostic subset has no mixed-outcome question; "
              "within-question AUC and correct-penalty relief are undefined. Raise --diag-rows "
              "or generate more rollouts per question.")

    # -- model --------------------------------------------------------------
    set_seed(args.seed)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    )
    if args.gradient_checkpointing:
        # Inputs are token ids (no grad); without this, checkpointing can drop the teacher's
        # gradients entirely -- the same fix GRPO applies to its policy and train_ppo.py to its
        # critic.
        teacher.enable_input_require_grads()

    collate = partial(collate_teacher_batch, pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=output_dir,
        # Our columns are not `forward` arguments, so Trainer's default column pruning would
        # strip every one of them.
        remove_unused_columns=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        max_steps=args.max_steps,
        per_device_train_batch_size=MODEL_FORWARD_BATCH_SIZE,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )

    meta = build_run_meta(
        args,
        len(train_dataset),
        len(validation_dataset),
        len(diagnostic_dataset),
        len(train_questions),
        len(validation_questions),
        len(diagnostic_questions),
    )
    if args.resume_from_checkpoint:
        validate_resume(
            args.resume_from_checkpoint, meta, args.force_resume,
            # Absence is disqualifying: a checkpoint written before this key holds a teacher
            # trained under a different notion of what the ratio means.
            strict_keys=("teacher_version", "question_split_version"),
        )
    if training_args.process_index == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote run metadata -> {os.path.join(output_dir, 'run_meta.json')}")

    trainer = SelfTeacherTrainer(
        model=teacher,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate,
        processing_class=tokenizer,
        objective=args.objective,
        pointwise_loss=args.pointwise_loss,
        tau=args.tau,
        beta=args.beta,
        length_norm=args.length_norm,
        bias_learning_rate=args.bias_learning_rate,
        kl_anchor=args.kl_anchor,
        diag_crossfit_folds=args.diag_crossfit_folds,
    )

    if args.kl_anchor > 0.0:
        train_dataset = add_init_teacher_logps(
            train_dataset, trainer.model, collate,
            trainer.accelerator.device,
        )
        trainer.train_dataset = train_dataset

    # -- init diagnostics: the baseline every later reading is compared against --
    n_diag = len(diagnostic_dataset)
    trainer.set_diagnostic_rows([diagnostic_dataset[i] for i in range(n_diag)], collate)
    init_metrics = trainer.diagnostics()
    print(f"\nInit diagnostics (untrained teacher, {n_diag} rows) -- the go/no-go baseline:")
    for key, value in init_metrics.items():
        print(f"  {key:28s} {value: .4f}")
    if training_args.process_index == 0:
        with open(os.path.join(output_dir, "diagnostics_init.json"), "w") as f:
            json.dump(init_metrics, f, indent=2)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_metrics = trainer.diagnostics()
    print("\nFinal diagnostics:")
    for key, value in final_metrics.items():
        delta = value - init_metrics[key] if key in init_metrics else None
        suffix = f"   (init {init_metrics[key]: .4f}, delta {delta:+.4f})" if delta is not None else ""
        print(f"  {key:28s} {value: .4f}{suffix}")

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)  # save_pretrained: loadable by stage 3 and teacher_behavior.py
    tokenizer.save_pretrained(final_dir)
    torch.save({"calibration_bias": trainer.calibration_bias.detach().cpu()},
               os.path.join(final_dir, CALIBRATION_FILE))
    if training_args.process_index == 0:
        with open(os.path.join(output_dir, "diagnostics_final.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)
    print(f"Saved teacher -> {final_dir}")


if __name__ == "__main__":
    main()
