r"""
Shared machinery for the TRAINED SELF-TEACHER arm: the objectives, the prompt/tensor
plumbing, and the diagnostics. No CLI -- see train/opsd/train_self_teacher/gen_rollouts.py (stage 1),
train/opsd/train_self_teacher/train.py (stage 2) and train/opsd/train_self_teacher/sdft_with_teacher.py (stage 3).

WHY THIS ARM EXISTS
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

Two objectives, differing in how hard they constrain the teacher (see `objective_pointwise` and
`objective_endpoint`). A third -- regressing every prefix `S_t`, the full process-reward variant
-- is deliberately not implemented yet; it is one extra target in `objective_endpoint`.

THE STUDENT IS FROZEN throughout stages 1-2, which is what lets stage 1 cache `student_logps`
and stage 2 hold only the teacher in memory. That shortcut encodes the probe's premise and goes
away if the E- and M-steps are ever alternated.
"""

import math
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

    Mirrors `utils.hint_path` exactly, and for the same reason: a rollout is only "on-policy"
    for the model that produced it, so crossing caches between models would silently break the
    self-distillation premise. Written by train/opsd/train_self_teacher/gen_rollouts.py; read by
    train/opsd/train_self_teacher/train.py.

    Note this cache is PI-INDEPENDENT -- rollouts are sampled from the student's un-privileged
    prompt, and the PI only enters at teacher-training time -- so ONE cache serves every
    --pi-mode. Generation is paid once for the whole ladder.
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

    `none` uses a bare concatenation instead, and the reason is not cosmetic. TRL applies the
    template UNCONDITIONALLY, so under the default an empty context still appends "\\n\\n" and the
    teacher reads a context the student never saw. The whole point of the `none` arm is that at
    init the teacher is the student under an IDENTICAL context, giving rho_t == 0 at every
    position -- a hard, checkable null in which every bit of signal provably comes from the
    E-step, and under which an untrained teacher makes stage 3 an exact no-op. Two trailing
    newlines would quietly destroy that.

    Stage 2 uses this to build its prompts and stage 3 passes it to SDFTConfig, so the teacher
    reads the same context when it is trained and when it scores. Both stamp it into
    run_meta.json and stage 3 verifies the two agree.
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
    Trailing pads on the completion are harmless -- they sit after every real token and are
    removed by `completion_mask`.
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
    # Present only under --kl-anchor, which adds it in a pre-pass; carried the same way as
    # student_logps so the anchor term lines up token for token.
    if "teacher_logps_init" in rows[0]:
        batch["teacher_logps_init"] = pad(
            [torch.tensor(r["teacher_logps_init"], dtype=torch.float32) for r in rows],
            padding_value=0.0, padding_side="right",
        )
    return batch


def teacher_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """`[teacher_prompt || completion]` -> (input_ids, attention_mask, logits_to_keep).

    Same shape of contract as `PPOTrainer._value_inputs`: a longer teacher prompt shifts nothing,
    because the completion's logprobs are sliced from the END of the sequence.
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
    the same tokens and rho is therefore ZERO analytically, read a `ratio_dispersion` of 0.041
    through the bf16 path -- a noise floor half the size of the default --tau of 0.1. Returning
    float32 halved it to 0.023. (The rest was input padding, not the log-softmax; see
    `per_token_logps`.)

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
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, completion_ids: torch.Tensor
) -> torch.Tensor:
    """Log-probabilities of `completion_ids` under `model`, aligned to the completion -> (B, C).

    Position i's logits predict token i+1, so the C completion tokens are scored by the C logit
    positions ending one before the last. `logits_to_keep=C+1` asks the model for exactly those
    (plus the trailing one we drop), which matters a lot here: a full (B, L, 151669) logit tensor
    at L~8k is several GB, and this is the tensor the backward pass has to hold.

    Temperature is NOT applied. The ratio is between two raw next-token distributions; stage 1
    scores the student the same way, so the two are directly subtractable.

    POSITION IDS ARE PASSED EXPLICITLY. Left unset, transformers derives them as a bare `arange`
    over the padded sequence (`modeling_qwen3.py:399`: `torch.arange(inputs_embeds.shape[1]) +
    past_seen_tokens`) with no reference to the attention mask, so a left-padded row's real
    tokens are evaluated at RoPE positions shifted by that row's own pad count. Inside one GRPO
    step that cancels, because every forward shares a padding layout; it does NOT cancel across
    our two stages, which pad differently. `cumsum(-1) - 1` restarts each row at position 0 on
    its first real token, so the forward means the same thing at any padding.

    IT IS NOT SUFFICIENT, THOUGH -- DO NOT PAD THESE FORWARDS. Fixing the positions left the
    `--pi-mode none` control's spurious dispersion essentially unchanged (0.0228 -> 0.0225),
    because the remaining error is bfloat16 arithmetic, not geometry: attention over pad
    positions perturbs the real tokens' logits even though the mask removes their contribution
    exactly. Measured on Qwen3-1.7B, same row, mask-derived positions throughout:

        17 left pads   std 0.031, max |d| 0.12
        64 left pads   std 0.035, max |d| 0.30
        batched B=2 at EQUAL width          std 0.0      (bit-identical)
        17 left pads, fp32 WEIGHTS          std 0.00001

    So the cost is padding, not batching, and it is comparable to the entire scale of rho. Every
    caller therefore scores UNPADDED by default -- gen_rollouts' --score-batch-size 1, stage 2's
    --diag-batch-size 1, and per_device_train_batch_size 1 -- which also makes all three agree
    with the batch-size-1 forward TRL performs in stage 3.
    """
    n_completion = completion_ids.size(1)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.clamp(min=0)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logits_to_keep=n_completion + 1,
        use_cache=False,
    ).logits
    logits = logits[:, :-1, :]  # drop the position predicting past the sequence end
    return _selective_logps_fp32(logits, completion_ids)


def concat_padded(tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Right-pad (B_i, C_i) blocks to a common width and concatenate along the batch axis.

    Lets a diagnostic pass collate and forward each chunk at its OWN width -- unpadded at chunk
    size 1 -- and only then assemble the results. Padding the resulting rho/mask tensors is free
    and lossless, because every metric here is mask-weighted; padding the model's INPUT is not
    (see `per_token_logps`).
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
    input_ids, attention_mask, _ = teacher_inputs(batch)
    return per_token_logps(model, input_ids, attention_mask, batch["completion_ids"])


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

    The sharpest instrument for "stop penalizing off-PI exploration". With the squared form its
    gradient w.r.t. the teacher is

        sum_t 2*(rho_t - target) * grad log pi_phi(y_t | x, c, y_<t)

    i.e. WEIGHTED SFT of the teacher on the student's own trace, with per-token weight
    (target - rho_t). On a correct trace the tokens the teacher currently likes LEAST -- large
    negative rho_t, which is exactly the off-PI deviation -- carry the largest upward push. That
    concentration is the whole point of this variant.

    `squared` regresses the RAW ratio onto a bounded target +/- tau (nats per token). Two
    deliberate choices:

      * NOT squared-error-on-sigmoid. A sigmoid link contributes a sigma' factor that VANISHES in
        the tail, so the most-penalized tokens -- the ones being targeted -- would receive the
        LEAST pull. That inverts the mechanism. `tests/test_self_teacher.py` pins the ordering.
      * A FINITE target. rho_t -> +/- tau is a reachable fixed point, so the degenerate limit of
        over-training is "REINFORCE with reward tau*(2R-1)" rather than a blow-up. The flatness
        is still real, which is what `ratio_dispersion` is for -- but it is bounded flatness.

    `logistic` is the unbounded alternative: BCE(sigmoid(beta*rho_t), R), whose gradient saturates
    at a constant beta instead of at zero, so it keeps pulling the tail tokens indefinitely. Use
    it when the bounded target's fixed point is reached before the ratio becomes informative.
    """
    reward = reward.unsqueeze(1)  # (B, 1) -> broadcast over tokens
    if loss == "squared":
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


# ---------------------------------------------------------------------------
# Diagnostics -- the actual deliverable of the probe
# ---------------------------------------------------------------------------


def calibration_metrics(
    ratios: torch.Tensor,
    mask: torch.Tensor,
    reward: torch.Tensor,
    beta: float = 1.0,
    bias: float = 0.0,
    length_norm: str = "mean",
) -> dict[str, float]:
    """Is the ratio predicting the outcome, and where does its credit land?

    Ported from `PPOPITrainer._log_calibration_metrics`, which asks the identical question of a
    scalar-head critic -- so the two parameterizations are read on the same instruments.

    The value at prefix t is `sigmoid(beta * S_t / N_t + b)` with S_t the running sum of rho and
    N_t the PREFIX length (not the trace length), so V_t reads as a running probability rather
    than shrinking towards the base rate early in long traces.

    Returned keys:
      brier_q{25,50,75,100}      squared error vs the realized outcome at prefix quantiles.
                                 Should FALL across quantiles for a working critic.
      value_at_start             V at the first completion token. It can only encode "how often
                                 does this policy solve this problem", so if brier_q100 is not
                                 much better than this, the teacher is reading QUESTION DIFFICULTY
                                 rather than the trace. The shortcut guard.
      ratio_dispersion           within-trace std of rho_t, averaged over traces. THE STOPPING
                                 SIGNAL for objective (c): a flat critic assigns identical credit
                                 everywhere, i.e. no credit assignment at all.
      credit_mass_last_quartile  share of sum|rho_t| falling in the last 25% of the trace. A
                                 teacher that has learned to string-match the final answer only
                                 moves at the end. `--pi-mode answer` is expected to trip this,
                                 which doubles as a positive control that the metric works.
      mean_ratio_correct/_wrong  the raw separation the objective is trying to create.
    """
    mask = mask.float()
    lengths = mask.sum(dim=1)  # (B,)
    valid = lengths > 0
    if not valid.any():
        return {}
    reward = reward.view(-1, 1)  # (B, 1)
    out: dict[str, float] = {}

    cumulative = (ratios * mask).cumsum(dim=1)  # (B, C)
    positions = torch.arange(
        1, mask.size(1) + 1, device=mask.device, dtype=torch.float32
    ).unsqueeze(0)
    if length_norm == "mean":
        normalizer = positions
    elif length_norm == "sqrt":
        normalizer = positions.sqrt()
    else:
        normalizer = torch.ones_like(positions)
    values = torch.sigmoid(beta * cumulative / normalizer + bias)  # (B, C)

    for q in (25, 50, 75, 100):
        idx = (lengths.float() * q / 100).ceil().long().clamp(min=1) - 1  # (B,)
        v_q = values.gather(1, idx.unsqueeze(1))  # (B, 1)
        out[f"brier_q{q}"] = ((v_q - reward) ** 2)[valid].mean().item()
    out["value_at_start"] = values[valid, 0].mean().item()
    out["outcome_mean"] = reward[valid].mean().item()

    # Within-trace dispersion of the raw ratio. Computed per row over its real tokens only, so a
    # short trace and a long one contribute equally.
    row_mean = (ratios * mask).sum(dim=1) / lengths.clamp(min=1.0)
    variance = ((ratios - row_mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / lengths.clamp(min=1.0)
    out["ratio_dispersion"] = variance[valid].sqrt().mean().item()

    # Where the credit lands.
    pos = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)  # (1, C)
    last_quarter = (pos.float() >= 0.75 * lengths.unsqueeze(1).float()) & mask.bool()
    magnitude = (ratios * mask).abs()
    total = magnitude.sum(dim=1)
    share = (magnitude * last_quarter).sum(dim=1) / total.clamp(min=1e-8)
    keep = valid & (total > 0)
    if keep.any():
        out["credit_mass_last_quartile"] = share[keep].mean().item()

    flat_reward = reward.view(-1)
    for name, rows in (("correct", flat_reward > 0.5), ("wrong", flat_reward <= 0.5)):
        rows = rows & valid
        if rows.any():
            out[f"mean_ratio_{name}"] = row_mean[rows].mean().item()
    return out


def penalized_correct_mean(
    ratios: torch.Tensor, mask: torch.Tensor, reward: torch.Tensor, decile: float = 0.1
) -> float | None:
    """Mean rho over the WORST-SCORED CORRECT traces -- the headline metric for this arm.

    A correct trace the teacher assigns a very negative mean rho to is, by construction, a
    solution the student found that the PI-conditioned teacher dislikes: off-PI exploration that
    worked. The bottom `decile` of correct traces by initial mean rho is the population this arm
    exists to stop punishing, so the probe reports how their mean rho MOVES from init.

    Separate from calibration on purpose: a teacher can become better calibrated overall while
    still penalizing exactly these traces, and that outcome would look like success on the Brier
    scores while failing the actual goal.

    Returns None when the batch holds no correct traces.
    """
    mask = mask.float()
    lengths = mask.sum(dim=1)
    correct = (reward.view(-1) > 0.5) & (lengths > 0)
    if not correct.any():
        return None
    row_mean = (ratios * mask).sum(dim=1) / lengths.clamp(min=1.0)
    scores = row_mean[correct]
    k = max(1, math.ceil(decile * scores.numel()))
    return scores.topk(k, largest=False).values.mean().item()
