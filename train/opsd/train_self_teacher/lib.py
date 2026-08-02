r"""
Shared machinery for the TRAINED SELF-TEACHER arm: the objectives, the prompt/tensor
plumbing, and the diagnostics. No CLI -- see train/opsd/train_self_teacher/gen_rollouts.py (stage 1),
train/opsd/train_self_teacher/train.py (stage 2) and train/opsd/train_self_teacher/sdft_with_teacher.py (stage 3).

SDFT's default objective in this repo (`distillation_mode="sampled_token"`, alpha=1.0) is
mechanically a policy gradient whose per-token advantage is the teacher:student log ratio.
TRL's `compute_sampled_token_self_distillation_loss` computes

    loss_t = sg[log pi_s(y_t) - log pi_t(y_t)] * log pi_s(y_t)

so gradient descent raises log pi_s exactly where the teacher prefers a token more than the
student does. That coefficient is currently COMPUTED, not learned: with a frozen teacher it
measures PI-CONFORMITY. A student that explores a correct solution the PI does not suggest is
penalized for the deviation.

This arm trains the teacher so the ratio tracks the OUTCOME instead. Same construction as
VPD's E-step (arXiv 2605.15113) with the reference prior set to the current student, and the
same parameterization as the implicit-PRM literature. What is simpler here: our labels are
VERIFIABLE (math_verify via accuracy_reward gives an exact 0/1), so the unpaired-preference
machinery VPD needs for textual feedback is unnecessary and the E-step is plain supervised
calibration.

    rho_t = log pi_phi(y_t | x, c, y_<t) - log pi_theta(y_t | x, y_<t)
    S_t   = sum_{s<=t} rho_s = log [ pi_phi(y_<=t | x, c) / pi_theta(y_<=t | x) ]

`S_t` is the sequential log-likelihood ratio of the prefix under the PI-informed teacher vs the
un-informed student, so `sigmoid(S_t)` reads as "how teacher-like is this prefix". Regressing it
onto the outcome is the direct statement of the goal: TEACHER-LIKENESS SHOULD MEAN
SUCCESS-LIKENESS, NOT PI-CONFORMITY.

Three objectives, differing in how hard they constrain the teacher (see `objective_pointwise`,
`objective_endpoint`, and `objective_asymmetric`). A fourth -- regressing every prefix `S_t`, the full process-reward variant
-- is deliberately not implemented yet; it is one extra target in `objective_endpoint`.

THE STUDENT IS FROZEN throughout stages 1-2, which is what lets stage 1 cache `student_logps`
and stage 2 hold only the teacher in memory. That shortcut encodes the probe's premise and goes
away if the E- and M-steps are ever alternated.
"""

import hashlib
import os

import torch
import torch.nn.functional as F
from trl.trainer.utils import pad

from train.opsd.train_sdft import PI_ANSWER, PI_FULL, PI_HINT
from utils import (
    TEACHER_PROMPT_TEMPLATE,
    compose_pi_messages,
    format_prompt_math,
    load_hint_cache,
    load_train_dataset,
)


# Same ladder as train_ppo_pi.PI_MODES, and the wording of each rung comes from train_sdft's
# PI_* templates, so "hint" means the same string in every arm of the study.
PI_MODES = ("hint", "answer", "full", "none")

# Every local bf16 scoring/training forward over ragged sequences is one sequence wide. Changing
# the padded tensor shape can change finite-precision kernel selection/operation order; the
# resulting logit shifts were locally large enough to contaminate the teacher/student ratio.
MODEL_FORWARD_BATCH_SIZE = 1

# Stamped into stage-2 run_meta.json and carried into stage 3 as a strict resume key. Bump it
# when a change alters what the trained teacher's weights MEAN (the objective family, the ratio
# definition, the prompt convention) -- not for ordinary hyperparameters, which are recorded
# individually. See utils.validate_resume's `strict_keys` for why absence must be disqualifying.
TEACHER_VERSION = "logratio_v1"

LENGTH_NORMS = ("mean", "sqrt", "none")


# ---------------------------------------------------------------------------
# Rollout cache location
# ---------------------------------------------------------------------------


def rollout_path(model: str, dataset: str, root: str = "data/rollouts") -> str:
    """On-disk cache of frozen-student rollouts, keyed by dataset then model slug.
    """
    return os.path.join(root, dataset, model.rstrip("/").split("/")[-1])


# ---------------------------------------------------------------------------
# The PI ladder
# ---------------------------------------------------------------------------


def privileged_context(row: dict, pi_mode: str) -> str:
    """The teacher-only string for one row, worded exactly as train_sdft.py words it.

    Empty for `none`, which is the matched control: with no PI the teacher is an identical copy
    of the student reading an identical context, so rho_t == 0 at every position and the whole
    signal has to come from the E-step. That makes `none` the cleanest test of the log-ratio
    parameterization itself, against train_ppo_pi.py's randomly-initialised `.score` head.
    """
    if pi_mode == "hint":
        return PI_HINT.format(hint=row["hint"])
    if pi_mode == "answer":
        return PI_ANSWER.format(answer=str(row["final_answer"]))
    if pi_mode == "full":
        return PI_FULL.format(demo=row["solution"])
    if pi_mode == "none":
        return ""
    raise ValueError(f"unknown pi_mode {pi_mode!r}; expected one of {PI_MODES}")


def teacher_prompt_template(pi_mode: str) -> str:
    """The `{prompt}`/`{privileged_context}` template this PI mode stitches with.
    Everything except `none` uses SDFTTrainer's default, `"{prompt}\\n\\n{privileged_context}"`.
    """
    return "{prompt}{privileged_context}" if pi_mode == "none" else TEACHER_PROMPT_TEMPLATE


def compose_teacher_messages(
    prompt_messages: list[dict], context: str, template: str = TEACHER_PROMPT_TEMPLATE
) -> list[dict]:
    """Fold the privileged context into the prompt the way SDFTTrainer does.

    Delegates to `utils.compose_pi_messages`, which implements
    SDFTTrainer._compose_teacher_prompt's convention: the PI is appended to the LAST USER TURN
    via the template rather than added as a new message, so the rendered prompt still ends on the
    same generation header as the student's and a completion attaches at the same boundary. That
    is what makes the teacher's logprobs comparable to the student's token for token.

    Applied unconditionally, empty context included -- see `teacher_prompt_template`.
    """
    return compose_pi_messages(prompt_messages, context, template)


def render_teacher_prompt_ids(tokenizer, conversations: list[list[dict]]) -> list[list[int]]:
    """Tokenize teacher conversations the way TRL tokenizes prompts: batched
    `apply_chat_template` with `add_generation_prompt=True`, no padding (the collator pads).

    Deliberately NOT truncating here. The user turn is "{question}\\n\\n{privileged_context}",
    so left-truncation removes the QUESTION and the template's opening header first and keeps
    the demo -- the teacher would score a completion against a headerless fragment. Rows that do
    not fit are DROPPED by `build_teacher_dataset` instead, the same choice
    `build_ppo_pi_dataset` and `build_sdft_dataset` make for the same reason.
    """
    return tokenizer.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]


def build_teacher_dataset(
    rollouts,
    pi_mode: str,
    tokenizer,
    dataset: str = "deepmath",
    max_teacher_prompt_length: int | None = None,
):
    """Attach the PI to a rollout cache and pre-tokenize the teacher prompt.

    `rollouts` is the stage-1 cache (see train/opsd/train_self_teacher/gen_rollouts.py): one row per ROLLOUT, carrying
    `question`, `final_answer`, `completion_ids`, `reward` and the frozen student's
    `student_logps`. This adds:

      `teacher_prompt_ids` -- the question under the PI, rendered with add_generation_prompt.

    For `hint` and `full` the PI is not in the rollout cache (it is per-question, not per
    rollout), so it is joined back on by question text -- from the hint cache for `hint`, from
    the dataset's worked solutions for `full`. Rows whose PI cannot be found are dropped and
    counted; a large drop means the rollout cache and the PI source disagree about the question
    set, which would otherwise silently shrink the arm.

    Rows whose teacher prompt exceeds `max_teacher_prompt_length` are dropped rather than
    truncated (see `render_teacher_prompt_ids`). A no-op for hint/answer/none; `full` is the arm
    that needs it.
    """
    if pi_mode not in PI_MODES:
        raise ValueError(f"unknown pi_mode {pi_mode!r}; expected one of {PI_MODES}")
    template = teacher_prompt_template(pi_mode)

    # Join the per-question PI source onto the per-rollout cache.
    pi_by_question: dict[str, dict] = {}
    if pi_mode == "hint":
        # load_hint_cache asserts gen_model/dataset, so a hint written by another model or for
        # another dataset cannot reach the teacher (the "self-hint purity" invariant).
        hints = load_hint_cache(rollouts.unique("gen_model")[0], dataset)
        pi_by_question = {r["question"]: r for r in hints}

    elif pi_mode == "full":
        pi_by_question = {r["question"]: r for r in load_train_dataset(dataset)}

    n_before = len(rollouts)
    if pi_mode in ("hint", "full"):
        rollouts = rollouts.filter(lambda r: r["question"] in pi_by_question, num_proc=4)
        if len(rollouts) < n_before:
            print(f"  pi={pi_mode}: dropped {n_before - len(rollouts)}/{n_before} rollouts whose "
                  f"question has no {pi_mode} to join")

    def _map(batch):
        conversations, contexts = [], []
        for question, answer in zip(batch["question"], batch["final_answer"]):
            source = dict(pi_by_question.get(question, {}))
            source["final_answer"] = answer
            context = privileged_context(source, pi_mode)
            contexts.append(context)
            conversations.append(
                compose_teacher_messages(format_prompt_math(question), context, template)
            )
        return {
            "teacher_prompt_ids": render_teacher_prompt_ids(tokenizer, conversations),
            "has_pi": [bool(c) for c in contexts],
        }

    ds = rollouts.map(_map, batched=True, batch_size=64, num_proc=1)

    if max_teacher_prompt_length is not None:
        n_before = len(ds)
        ds = ds.filter(
            lambda r: len(r["teacher_prompt_ids"]) <= max_teacher_prompt_length, num_proc=4
        )
        if len(ds) < n_before:
            print(f"  pi={pi_mode}: kept {len(ds)}/{n_before} rollouts whose teacher prompt fits "
                  f"max_teacher_prompt_length={max_teacher_prompt_length}")
    return ds


# ---------------------------------------------------------------------------
# Tensor plumbing
# ---------------------------------------------------------------------------


def collate_teacher_batch(rows: list[dict], pad_token_id: int) -> dict:
    """Pad a list of cache rows into the GRPO tensor layout.

    Teacher prompts are LEFT-padded and completions RIGHT-padded, giving
    `[PAD.. prompt][completion ..PAD]`. Left padding on the prompt is load-bearing: it keeps the
    prompt's real tokens flush against the completion, so the logits that score the first
    completion token come off the generation header rather than a pad. (The same invariant
    train_ppo_val.py's tests pin for the critic; here it decides whether rho_1 is meaningful.)
    Production forwards contain exactly one row, so `pad` adds no tokens. The layout rules remain
    explicit because this helper is also tested with multiple rows. Trailing completion pads are
    excluded from downstream objectives by `completion_mask`; they are not assumed to be
    numerically inert inside a multi-row model forward.
    """
    prompts = [torch.tensor(r["teacher_prompt_ids"], dtype=torch.long) for r in rows]
    completions = [torch.tensor(r["completion_ids"], dtype=torch.long) for r in rows]
    student_logps = [torch.tensor(r["student_logps"], dtype=torch.float32) for r in rows]

    batch = {
        "teacher_prompt_ids": pad(prompts, padding_value=pad_token_id, padding_side="left"),
        "teacher_prompt_mask": pad(
            [torch.ones_like(p) for p in prompts], padding_value=0, padding_side="left"
        ),
        "completion_ids": pad(completions, padding_value=pad_token_id, padding_side="right"),
        "completion_mask": pad(
            [torch.ones_like(c) for c in completions], padding_value=0, padding_side="right"
        ),
        "student_logps": pad(student_logps, padding_value=0.0, padding_side="right"),
        "reward": torch.tensor([float(r["reward"]) for r in rows], dtype=torch.float32),
    }
    # Present when an initialization-anchored loss needs it; carried the same way as student_logps
    # so the reference ratio and anchor term line up token for token.
    if "teacher_logps_init" in rows[0]:
        batch["teacher_logps_init"] = pad(
            [torch.tensor(r["teacher_logps_init"], dtype=torch.float32) for r in rows],
            padding_value=0.0, padding_side="right",
        )
    if "asym_lift_weight" in rows[0]:
        batch["asym_lift_weight"] = torch.tensor(
            [float(r["asym_lift_weight"]) for r in rows], dtype=torch.float32
        )
    return batch


def teacher_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """`[teacher_prompt || completion]` -> (input_ids, attention_mask, logits_to_keep).

    Same shape of contract as `PPOTrainer._value_inputs`: a longer teacher prompt shifts nothing,
    because the completion's logprobs are sliced from the END of the sequence. The mask remains
    useful for checking/carrying the collator layout; the strict unpadded batch-one logprob
    forward does not need to pass it to the model.
    """
    input_ids = torch.cat([batch["teacher_prompt_ids"], batch["completion_ids"]], dim=1)
    attention_mask = torch.cat([batch["teacher_prompt_mask"], batch["completion_mask"]], dim=1)
    return input_ids, attention_mask, batch["completion_ids"].size(1)


# Rows of the flattened (B*C, V) logit tensor converted to float32 at a time. Bounds the peak
# float32 footprint of the log-softmax to chunk * vocab * 4 bytes (~620MB at 1024 x 151669).
LOGP_CHUNK = 1024


def _selective_logps_fp32(
    logits: torch.Tensor, index: torch.Tensor, chunk_size: int = LOGP_CHUNK
) -> torch.Tensor:
    """log p(index) under `logits`, computed and RETURNED in float32.

    Deliberately not `trl.trainer.utils.selective_log_softmax`, which returns the logits' own
    dtype. That is fine wherever a logprob is used on its own, and wrong here: this arm's entire
    signal is

        rho_t = log pi_teacher(y_t) - log pi_student(y_t)

    a difference of two nearly-equal quantities, so it is exactly the catastrophic-cancellation
    case. bfloat16 carries an 8-bit mantissa, so a logprob near -8 has an ulp of 0.0625; two
    independent roundings put a noise floor of a few hundredths of a nat on rho. MEASURED on
    Qwen3-1.7B: the `--pi-mode none` control, where teacher and student are the same weights on
    the same tokens and rho is therefore ZERO analytically, had within-trace dispersion of 0.041
    through the bf16 path -- a noise floor half the size of the default --tau of 0.1. Returning
    float32 halved it to 0.023; the remainder disappeared when the two inputs were evaluated as
    unpadded batch-one tensors (see `per_token_logps`). These numbers are local measurements, not
    general error bounds for Qwen or bfloat16.

    The gather-minus-logsumexp form (rather than a full log_softmax) avoids materializing a
    second (B, C, V) tensor, and is numerically stable once the inputs are float32 -- which is
    why TRL avoids it in the bf16 branch and we can use it here.
    """
    n_rows, n_tokens, vocab = logits.shape
    flat_logits = logits.reshape(n_rows * n_tokens, vocab)
    flat_index = index.reshape(n_rows * n_tokens, 1)
    chunks = []
    for start in range(0, flat_logits.size(0), chunk_size):
        block = flat_logits[start : start + chunk_size].float()
        selected = block.gather(-1, flat_index[start : start + chunk_size]).squeeze(-1)
        chunks.append(selected - torch.logsumexp(block, dim=-1))
    return torch.cat(chunks).view(n_rows, n_tokens)


def per_token_logps(
    model, input_ids: torch.Tensor, completion_ids: torch.Tensor
) -> torch.Tensor:
    """Log-probabilities of `completion_ids` under `model` -> (1, C).

    Temperature is NOT applied. The ratio is between two raw next-token distributions; stage 1
    scores the student the same way, so the two are directly subtractable.

    `input_ids` is one unpadded `[prompt || completion]` sequence. Position i predicts token i+1,
    so asking for C+1 trailing logits and dropping the last aligns the remaining C positions with
    `completion_ids`. Avoiding a full (1, L, V) logit tensor also saves several GB at L~8k.

    The physical batch-one contract is enforced here rather than left to every caller. With a
    correct mask and position IDs, padded and unpadded forwards are mathematically equivalent.
    They need not be numerically identical: changing tensor shape can change GEMM/SDPA kernels,
    tiling, and reduction order, and bfloat16 makes the resulting roundoff easier to see. PyTorch
    documents this class of behavior under "Numerical accuracy"; this is not evidence that pads
    semantically leak through the attention mask.

    Locally measured on Qwen3-1.7B:

        17 left pads   std 0.031, max |d| 0.12
        64 left pads   std 0.035, max |d| 0.30
        batched B=2 at EQUAL width          std 0.0      (bit-identical)
        17 left pads, fp32 WEIGHTS          std 0.00001

    No attention mask or explicit position IDs are needed for a single unpadded sequence: the
    model's ordinary causal mask and arange positions are exactly the intended inputs.
    """
    if input_ids.ndim != 2 or completion_ids.ndim != 2:
        raise ValueError("per_token_logps expects rank-2 input_ids and completion_ids")
    if (
        input_ids.size(0) != MODEL_FORWARD_BATCH_SIZE
        or completion_ids.size(0) != MODEL_FORWARD_BATCH_SIZE
    ):
        raise ValueError(
            f"per_token_logps requires physical batch size {MODEL_FORWARD_BATCH_SIZE}; "
            "accumulate gradients instead of padding model inputs"
        )
    n_completion = completion_ids.size(1)
    if input_ids.size(1) <= n_completion:
        raise ValueError("input_ids must contain a non-empty prompt before completion_ids")
    logits = model(
        input_ids=input_ids,
        logits_to_keep=n_completion + 1,
        use_cache=False,
    ).logits
    logits = logits[:, :-1, :]  # drop the position predicting past the sequence end
    return _selective_logps_fp32(logits, completion_ids)


def concat_padded(tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Right-pad (B_i, C_i) blocks to a common width and concatenate along the batch axis.

    Lets a diagnostic pass collate and forward each chunk at its OWN width -- unpadded at chunk
    size 1 -- and only then assemble the results. Padding the resulting rho/mask tensors is free
    and lossless, because every metric here is mask-weighted. Padding the model input instead
    changes its tensor shape and can change finite-precision numerics (see `per_token_logps`).
    """
    width = max(tensor.size(1) for tensor in tensors)
    return torch.cat(
        [F.pad(tensor, (0, width - tensor.size(1)), value=pad_value) for tensor in tensors], dim=0
    )


def teacher_token_logps(model, batch: dict) -> torch.Tensor:
    """The teacher's per-token logprobs for `batch`'s completion -> (B, C).

    `teacher_inputs` also reports `logits_to_keep` -- which `per_token_logps` derives for itself
    -- so composing the two here keeps the redundant third element from being splatted into the
    wrong argument at each of the three call sites.
    """
    input_ids, _, _ = teacher_inputs(batch)
    return per_token_logps(model, input_ids, batch["completion_ids"])


def log_ratio(teacher_logps: torch.Tensor, batch: dict) -> torch.Tensor:
    """rho_t = log pi_teacher(y_t | x, c) - log pi_student(y_t | x), masked to real tokens.

    The student half is read from the stage-1 cache rather than recomputed: the student is frozen
    for the whole E-step, so its logprobs are constants of the problem.
    """
    return (teacher_logps - batch["student_logps"]) * batch["completion_mask"]


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def objective_pointwise(
    ratios: torch.Tensor,
    mask: torch.Tensor,
    reward: torch.Tensor,
    tau: float = 0.1,
    loss: str = "squared",
    beta: float = 1.0,
) -> torch.Tensor:
    """(c) Regress each token's rho_t towards the outcome.

    With the squared form its gradient w.r.t. the teacher is

        sum_t 2*(rho_t - target) * grad log pi_phi(y_t | x, c, y_<t)

    i.e. WEIGHTED SFT of the teacher on the student's own trace, with per-token weight
    (target - rho_t).

    `squared` regresses the RAW ratio onto a bounded target +/- tau (nats per token). Two
    deliberate choices:

      * NOT squared-error-on-sigmoid. A sigmoid link contributes a sigma' factor that VANISHES in
        the tail, so the most-penalized tokens -- the ones being targeted -- would receive the
        LEAST pull. That inverts the mechanism. `tests/test_self_teacher.py` pins the ordering.
      * A FINITE target. rho_t -> +/- tau is a reachable fixed point, so the degenerate limit of
        over-training is "REINFORCE with reward tau*(2R-1)" rather than a blow-up. The flatness
        is still real, which is what `ratio_dispersion_retained` detects -- but it is bounded.

    `logistic` is the unbounded alternative: BCE(sigmoid(beta*rho_t), R), whose gradient saturates
    at a constant beta instead of at zero, so it keeps pulling the tail tokens indefinitely. Use
    it when the bounded target's fixed point is reached before the ratio becomes informative.
    """
    reward = reward.unsqueeze(1)  # (B, 1) -> broadcast over tokens
    if loss == "squared":
        # 2R - 1 makes the target symmetric around the meaningful baseline \(\rho=0\),
        # while \(\tau\) limits how far the teacher moves away from the student
        target = tau * (2.0 * reward - 1.0)
        return _masked_mean((ratios - target) ** 2, mask)
    if loss == "logistic":
        per_token = F.binary_cross_entropy_with_logits(
            beta * ratios, reward.expand_as(ratios), reduction="none"
        )
        return _masked_mean(per_token, mask)
    raise ValueError(f"unknown pointwise loss {loss!r}; expected 'squared' or 'logistic'")


def sequence_logit(
    ratios: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
    bias: torch.Tensor,
    length_norm: str = "mean",
) -> torch.Tensor:
    """beta * S_T / N + b -- the log-odds that a trace succeeds, from its total log-ratio.

    Length normalization is not cosmetic. S_T accumulates over completions spanning ~2.5k-8.2k
    tokens here, so with N=1 a fixed beta saturates the sigmoid on long traces and their gradient
    vanishes -- the objective would silently train on short traces only. `mean` (N=L) makes beta
    interpretable as "nats per token" and removes the length dependence entirely; `sqrt` is the
    middle ground if per-token averaging washes out a genuinely accumulating signal.

    `bias` is VPD's delta: a single learnable scalar that absorbs the outcome BASE RATE, so the
    ratio does not have to encode "this policy solves ~60% of problems" and the degenerate
    solution of shifting every rho_t by a constant is unavailable.
    """
    totals = (ratios * mask).sum(dim=1)  # (B,)
    lengths = mask.sum(dim=1).clamp(min=1.0)
    if length_norm == "mean":
        totals = totals / lengths
    elif length_norm == "sqrt":
        totals = totals / lengths.sqrt()
    elif length_norm != "none":
        raise ValueError(f"unknown length_norm {length_norm!r}; expected one of {LENGTH_NORMS}")
    return beta * totals + bias


def objective_endpoint(
    ratios: torch.Tensor,
    mask: torch.Tensor,
    reward: torch.Tensor,
    beta: float,
    bias: torch.Tensor,
    length_norm: str = "mean",
) -> torch.Tensor:
    """(a) Regress only the TRACE TOTAL onto the outcome.

    The minimal intervention: one constraint per trace instead of one per token. It says "the
    teacher's overall preference for this trace should track whether it succeeded" and leaves the
    per-token allocation completely free -- so a 5000-token correct trace can be explained by ten
    tokens at +0.3 and 4990 at 0. That freedom to be SPARSE is what preserves credit assignment,
    and it is why this variant cannot degenerate into a flat advantage the way (c) can.

    Its gradient weights every token in a trace equally, so it is a gentler and less targeted
    corrective than (c) -- the two are complements, not rivals, which is why the probe runs both.
    """
    return F.binary_cross_entropy_with_logits(
        sequence_logit(ratios, mask, beta, bias, length_norm), reward
    )


def asymmetric_lift_mask(
    initial_ratios: torch.Tensor,
    mask: torch.Tensor,
    reward: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """The FIXED set of successful, initially penalized completion tokens to lift.

    The decision is made from `rho^0`, not the current ratio, so a token remains in the lift set
    after it crosses the margin and trains towards a finite target instead of switching losses.
    """
    # This is intentionally the simplest lift policy. A later version may replace it with a
    # semantically targeted mask -- for example, lifting only penalized uncertainty-verbalization
    # tokens -- without changing the loss or the question-balancing machinery below.
    return (
        mask.bool()
        & (reward.view(-1, 1) > 0.5)
        & (initial_ratios.detach() < margin)
    )


def _masked_row_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean within each row; empty rows contribute zero."""
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def objective_asymmetric(
    ratios: torch.Tensor,
    initial_ratios: torch.Tensor,
    mask: torch.Tensor,
    reward: torch.Tensor,
    lift_weight: torch.Tensor,
    margin: float = 0.0,
    lift_alpha: float = 1.0,
    anchor_weight: float = 1.0,
) -> torch.Tensor:
    """Lift only successful tokens the initial PI teacher penalized; preserve everything else.

    For a targeted token, the finite target is

        rho^target = rho^0 + alpha * (margin - rho^0).

    `lift_weight` is computed once for the whole training set so the expected lift contribution is
    equal across eligible questions. It must NOT be normalized inside the physical batch: model
    forwards are one row wide, where such normalization would cancel the balancing entirely.

    The anchor is the exact complement of the fixed lift mask over real completion tokens. Thus a
    token is never simultaneously pulled toward the lift target and back toward initialization.
    """
    if not 0.0 <= lift_alpha <= 1.0:
        raise ValueError("lift_alpha must be between 0 and 1")
    if anchor_weight < 0.0:
        raise ValueError("anchor_weight must be non-negative")
    if ratios.shape != initial_ratios.shape or ratios.shape != mask.shape:
        raise ValueError("ratios, initial_ratios, and mask must have identical shapes")
    if lift_weight.numel() != ratios.size(0):
        raise ValueError("lift_weight must contain one scalar per rollout")

    initial_ratios = initial_ratios.detach()
    completion_mask = mask.bool()
    lift_mask = asymmetric_lift_mask(initial_ratios, completion_mask, reward, margin)
    anchor_mask = completion_mask & ~lift_mask

    lift_target = initial_ratios + lift_alpha * (margin - initial_ratios)
    lift_per_row = _masked_row_mean((ratios - lift_target) ** 2, lift_mask)
    lift_loss = (lift_weight.view(-1).to(lift_per_row) * lift_per_row).mean()

    anchor_per_row = _masked_row_mean((ratios - initial_ratios) ** 2, anchor_mask)
    anchor_loss = anchor_per_row.mean()
    return lift_loss + anchor_weight * anchor_loss


# ---------------------------------------------------------------------------
# Diagnostics -- the actual deliverable of the probe
# ---------------------------------------------------------------------------


def calibration_metrics(
    ratios: torch.Tensor,
    mask: torch.Tensor,
    reward: torch.Tensor,
    length_norm: str = "mean",
    question_ids: list[str] | None = None,
    crossfit_folds: int = 5,
    initial_ratios: torch.Tensor | None = None,
) -> dict[str, float]:
    """Decision dashboard for a trained self-teacher.

    At prefix t the score is `S_t / N_t`, where S_t is the running sum of rho and N_t is the
    configured prefix normalizer. Outcome Briers use QUESTION-GROUPED OUT-OF-FOLD Platt scaling,
    which absorbs the raw score's scale and offset. The slope is constrained non-negative because
    a sign flip would conceal a ratio that trains SDFT in the wrong direction.

    When `initial_ratios` is supplied, the same frozen diagnostic traces at initialization become
    the reference for all mechanism and drift metrics. Stage 2 always supplies it.

    Returned keys:
      brier_q*_fitted            At this point in the trajectory, does the accumulated
                                 teacher–student ratio contain a positively oriented,
                                 out-of-question signal about eventual success?

      brier_floor_crossfit       How well could we predict using only the rollout success
                                 base rate, without looking at the ratio?

                                 want : brier_q*_fitted < brier_floor_crossfit

      within_question_auc_q*     Difficulty shortcut : For the same question,
                                 does the ratio assign a higher prefix score to
                                 correct rollouts than to incorrect rollouts?

                                 1.0 -> every correct rollout outranks every incorrect rollout
                                 0.5 -> no within-question discrimination
                                 0.0 -> every incorrect rollout outranks every correct rollout

      mixed_question_count       number of questions supporting the within-question metrics.

      correct_penalty_relief     question-centered improvement on every initially penalized
                                 successful rollout, weighted by its initial penalty magnitude.
                                 Delta > 0 → the correct rollout was relieved
                                 Delta = 0 → its relative treatment did not change
                                 Delta < 0 → it became even more penalized
      correct_penalized_count    number of successful rollouts supporting that statistic.

      wrong_ratio_delta          How training changes the teacher on incorrect held-out
                                 trajectories relative to the initial PI-conditioned
                                 teacher : signed mean trace-ratio drift on incorrect rollouts.
                                 negative → failed rollouts became less preferred by the teacher
                                 zero     → no average signed change
                                 positive → failed rollouts became more preferred on average
      wrong_ratio_rms_drift      mean per-trace token-level RMS drift on incorrect rollouts.

                                 Signed delta | RMS drift | Interpretation |
                                 |---:|---:|---|
                                 | Near 0 | Near 0 | Failures were preserved |
                                 | Negative | Similar magnitude | Mostly uniform suppression |
                                 | Positive | Similar magnitude | Mostly uniform lifting |
                                 | Near 0 | Large | Strong redistribution that cancels in the mean |
                                 | Small negative | Much larger | Large mixed changes with a slight downward net effect

                                 Desirable for asymmetric objective :
                                 correct_penalty_relief > 0
                                 wrong_ratio_delta     ≈ 0
                                 wrong_ratio_rms_drift ≈ 0

      ratio_dispersion_retained  Measures how much token-to-token variation remains
                                 inside each trajectory relative to the initial
                                 PI-conditioned teacher.
                                 1.0       → same average amount of within-trajectory variation
                                 0.5       → half the initial variation remains
                                 near 0    → ratios have become nearly constant within trajectories
                                 greater 1 → ratios have become more variable or spiky

      credit_mass_last_quartile  Measures where the magnitude of the token-level
                                 teacher–student ratio is concentrated.
                                 It is designed to detect a teacher that only becomes
                                 informative near the final answer.
                                 `--pi-mode answer` is expected to trip this,
                                 which doubles as a positive control that the metric works.

                                 approximately 0.25 → ratio magnitude distributed uniformly
                                 greater than 0.25  → disproportionately late-firing
                                 near 1.0           → almost all signal occurs near the end
                                 less than 0.25     → more signal occurs in earlier portions
    """
    mask = mask.float()
    lengths = mask.sum(dim=1)
    valid = lengths > 0
    if not valid.any():
        return {}
    reward = reward.view(-1)
    out: dict[str, float] = {}

    cumulative = (ratios * mask).cumsum(dim=1)
    positions = torch.arange(
        1, mask.size(1) + 1, device=mask.device, dtype=torch.float32
    ).unsqueeze(0)
    if length_norm == "mean":
        normalizer = positions
    elif length_norm == "sqrt":
        normalizer = positions.sqrt()
    else:
        normalizer = torch.ones_like(positions)
    scores = cumulative / normalizer

    if question_ids is not None and len(question_ids) != ratios.size(0):
        raise ValueError(
            f"question_ids has {len(question_ids)} rows but ratios has {ratios.size(0)}"
        )
    valid_rows = torch.nonzero(valid, as_tuple=False).flatten().cpu().tolist()
    fit_groups = (
        [question_ids[i] for i in valid_rows]
        if question_ids is not None
        else list(range(len(valid_rows)))
    )

    r_valid = reward[valid]
    both_classes = r_valid.min() < 0.5 < r_valid.max()
    mixed_rows = None
    if question_ids is not None:
        mixed_valid, n_mixed = _mixed_group_mask(r_valid, fit_groups)
        out["mixed_question_count"] = float(n_mixed)
        mixed_rows = torch.zeros_like(valid)
        mixed_rows[valid] = mixed_valid

    for q in (25, 50, 75, 100):
        if not both_classes:
            continue
        idx = (lengths.float() * q / 100).ceil().long().clamp(min=1) - 1
        s_q = scores.gather(1, idx.unsqueeze(1))[valid].flatten()

        # A negative slope would hide a ratio that trains SDFT in the wrong direction.
        crossfit = cross_fitted_logistic(
            s_q, r_valid, fit_groups, n_folds=crossfit_folds, nonnegative_slope=True
        )
        if crossfit is not None:
            fitted, constant, _ = crossfit
            out[f"brier_q{q}_fitted"] = ((fitted - r_valid) ** 2).mean().item()
            if "brier_floor_crossfit" not in out:
                out["brier_floor_crossfit"] = ((constant - r_valid) ** 2).mean().item()

        if question_ids is not None:
            grouped_auc, _ = macro_group_rank_auc(s_q, r_valid, fit_groups)
            if grouped_auc is not None:
                out[f"within_question_auc_q{q}"] = grouped_auc

    # Per-trace summaries keep short and long completions equally weighted.
    row_mean = (ratios * mask).sum(dim=1) / lengths.clamp(min=1.0)
    variance = ((ratios - row_mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / lengths.clamp(min=1.0)
    dispersion = variance[valid].sqrt().mean()

    if initial_ratios is not None:
        if initial_ratios.shape != ratios.shape:
            raise ValueError("initial_ratios must have the same shape as ratios")
        initial_ratios = initial_ratios.to(device=ratios.device, dtype=ratios.dtype)
        initial_row_mean = (initial_ratios * mask).sum(dim=1) / lengths.clamp(min=1.0)
        initial_variance = (
            (initial_ratios - initial_row_mean.unsqueeze(1)) ** 2 * mask
        ).sum(dim=1) / lengths.clamp(min=1.0)
        initial_dispersion = initial_variance[valid].sqrt().mean()
        # The epsilon makes the matched `pi_mode=none` control read 1.0 at initialization while
        # still loudly exposing any subsequently created dispersion.
        eps = torch.finfo(torch.float32).eps
        out["ratio_dispersion_retained"] = (
            (dispersion + eps) / (initial_dispersion + eps)
        ).item()

        delta = ratios - initial_ratios
        row_delta = row_mean - initial_row_mean
        wrong = valid & (reward <= 0.5)
        if wrong.any():
            row_rms = ((delta.square() * mask).sum(dim=1) / lengths.clamp(min=1.0)).sqrt()
            out["wrong_ratio_delta"] = row_delta[wrong].mean().item()
            out["wrong_ratio_rms_drift"] = row_rms[wrong].mean().item()

        if question_ids is not None and mixed_rows is not None:
            initial_centered = group_centered_scores(initial_row_mean, question_ids, valid)
            current_centered = group_centered_scores(row_mean, question_ids, valid)
            correct_penalized = (
                valid & mixed_rows & (reward > 0.5) & (initial_centered < 0)
            )
            out["correct_penalized_count"] = float(correct_penalized.sum())
            weights = (-initial_centered).clamp_min(0) * correct_penalized
            if weights.sum() > 0:
                out["correct_penalty_relief"] = (
                    weights * (current_centered - initial_centered)
                ).sum().div(weights.sum()).item()

    pos = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
    last_quarter = (pos.float() >= 0.75 * lengths.unsqueeze(1).float()) & mask.bool()
    magnitude = (ratios * mask).abs()
    total = magnitude.sum(dim=1)
    share = (magnitude * last_quarter).sum(dim=1) / total.clamp(min=1e-8)
    keep = valid & (total > 0)
    if keep.any():
        out["credit_mass_last_quartile"] = share[keep].mean().item()
    return out


def fit_logistic(
    scores: torch.Tensor,
    reward: torch.Tensor,
    iters: int = 25,
    ridge: float = 1e-4,
    nonnegative_slope: bool = False,
) -> tuple[float, float]:
    """Maximum-likelihood fit of `sigmoid(a*s + b)` to binary `reward` -- Platt scaling.

    Scores are standardized before the fit, making the slope penalty invariant to PI arms whose
    ratios differ only in scale. Ridge applies to the slope only: the intercept must remain free to
    represent the outcome base rate exactly. The returned parameters are converted back to the raw
    score scale. If `nonnegative_slope`, an anti-predictive fit is projected to the best constant
    predictor because a sign flip would hide a ratio that trains SDFT in the wrong direction.
    """
    scores = scores.detach().double().flatten()
    reward = reward.detach().double().flatten()
    base_rate = reward.mean().clamp(1e-8, 1 - 1e-8)
    base_intercept = torch.logit(base_rate)
    score_mean = scores.mean()
    score_scale = scores.std(unbiased=False)
    if score_scale < 1e-12:
        return 0.0, base_intercept.item()

    standardized = (scores - score_mean) / score_scale
    design = torch.stack([standardized, torch.ones_like(standardized)], dim=1)
    weights = torch.stack([torch.zeros_like(base_intercept), base_intercept])
    penalty = torch.diag(torch.tensor([ridge, 0.0], dtype=torch.float64, device=scores.device))
    for _ in range(iters):
        probs = torch.sigmoid(design @ weights)
        gradient = design.T @ (probs - reward) + penalty @ weights
        curvature = (probs * (1 - probs)).clamp(min=1e-9)
        hessian = (design * curvature.unsqueeze(1)).T @ design + penalty
        step = torch.linalg.solve(hessian, gradient)
        weights = weights - step
        if step.abs().max() < 1e-10:
            break

    raw_slope = weights[0] / score_scale
    raw_intercept = weights[1] - weights[0] * score_mean / score_scale
    if nonnegative_slope and raw_slope < 0:
        return 0.0, base_intercept.item()
    return raw_slope.item(), raw_intercept.item()


def rank_auc(scores: torch.Tensor, reward: torch.Tensor) -> float | None:
    """Probability that a correct trace outranks an incorrect one. None if only one class.

    TIES COUNT AS HALF. Without that a flat score -- every trace identical, the degenerate case
    this metric most needs to identify -- reads 0.0 rather than 0.5, i.e. looks perfectly
    ANTI-predictive instead of uninformative.

    Scale-free and link-free by construction, so unlike the Brier scores it is unaffected by
    `--beta`; it is the cheapest honest answer to "does the ratio order traces at all".
    """
    scores = scores.detach().flatten()
    reward = reward.detach().flatten()
    positive = scores[reward > 0.5]
    negative = scores[reward <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    delta = positive.unsqueeze(1) - negative.unsqueeze(0)
    return ((delta > 0).float() + 0.5 * (delta == 0).float()).mean().item()


def _group_rows(group_ids: list[object]) -> dict[object, list[int]]:
    grouped: dict[object, list[int]] = {}
    for idx, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(idx)
    return grouped


def _mixed_group_mask(
    reward: torch.Tensor, group_ids: list[object]
) -> tuple[torch.Tensor, int]:
    """Rows belonging to groups with both outcomes, plus the number of such groups."""
    reward = reward.flatten()
    if len(group_ids) != reward.numel():
        raise ValueError("group_ids must align one-to-one with reward")
    mixed = torch.zeros_like(reward, dtype=torch.bool)
    count = 0
    for rows in _group_rows(group_ids).values():
        index = torch.tensor(rows, device=reward.device, dtype=torch.long)
        group_reward = reward.index_select(0, index)
        if group_reward.numel() and group_reward.min() < 0.5 < group_reward.max():
            mixed[index] = True
            count += 1
    return mixed, count


def macro_group_rank_auc(
    scores: torch.Tensor, reward: torch.Tensor, group_ids: list[object]
) -> tuple[float | None, int]:
    """Macro AUC over groups containing both outcomes.

    A question-only score is constant inside a question and therefore reads exactly 0.5 even if
    its global AUC is high. Questions contribute equally regardless of rollout count.
    """
    if len(group_ids) != scores.numel():
        raise ValueError("group_ids must align one-to-one with scores")
    scores, reward = scores.flatten(), reward.flatten()
    aucs = []
    for rows in _group_rows(group_ids).values():
        index = torch.tensor(rows, device=scores.device, dtype=torch.long)
        auc = rank_auc(scores.index_select(0, index), reward.index_select(0, index))
        if auc is not None:
            aucs.append(auc)
    return (sum(aucs) / len(aucs), len(aucs)) if aucs else (None, 0)




def cross_fitted_logistic(
    scores: torch.Tensor,
    reward: torch.Tensor,
    group_ids: list[object],
    n_folds: int = 5,
    nonnegative_slope: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[float]] | None:
    """Out-of-fold Platt predictions, keeping every group wholly inside one fold.

    Returns `(fitted, constant, slopes)`. `constant` is the matching out-of-fold intercept-only
    baseline, so comparing its Brier with `fitted` cannot reward two parameters merely for being
    fit on the rows they score. None when fewer than two groups or only one outcome exists.
    """
    scores = scores.detach().double().flatten()
    reward = reward.detach().double().flatten()
    if len(group_ids) != scores.numel():
        raise ValueError("group_ids must align one-to-one with scores")
    if reward.numel() == 0 or reward.min() == reward.max():
        return None

    groups = list(_group_rows(group_ids))
    if len(groups) < 2:
        return None
    n_folds = max(2, min(n_folds, len(groups)))
    groups.sort(key=lambda group: hashlib.sha256(repr(group).encode()).digest())
    fold_for_group = {group: idx % n_folds for idx, group in enumerate(groups)}
    row_folds = torch.tensor(
        [fold_for_group[group] for group in group_ids], device=scores.device, dtype=torch.long
    )

    fitted = torch.empty_like(scores)
    constant = torch.empty_like(scores)
    slopes: list[float] = []
    for fold in range(n_folds):
        test = row_folds == fold
        train = ~test
        if not test.any() or not train.any():
            continue
        train_reward = reward[train]
        base_rate = train_reward.mean().clamp(1e-8, 1 - 1e-8)
        constant[test] = base_rate
        if train_reward.min() == train_reward.max():
            slope, intercept = 0.0, torch.logit(base_rate).item()
        else:
            slope, intercept = fit_logistic(
                scores[train], train_reward, nonnegative_slope=nonnegative_slope
            )
        fitted[test] = torch.sigmoid(slope * scores[test] + intercept)
        slopes.append(slope)
    if not slopes:
        return None
    return fitted.float(), constant.float(), slopes


def group_centered_scores(
    scores: torch.Tensor, group_ids: list[object], valid: torch.Tensor | None = None
) -> torch.Tensor:
    """Subtract each group's mean score, using only valid rows in that group."""
    scores = scores.flatten()
    if len(group_ids) != scores.numel():
        raise ValueError("group_ids must align one-to-one with scores")
    valid = torch.ones_like(scores, dtype=torch.bool) if valid is None else valid.bool().flatten()
    centered = torch.zeros_like(scores)
    for rows in _group_rows(group_ids).values():
        index = torch.tensor(rows, device=scores.device, dtype=torch.long)
        keep = index[valid.index_select(0, index)]
        if keep.numel():
            centered[keep] = scores.index_select(0, keep) - scores.index_select(0, keep).mean()
    return centered
