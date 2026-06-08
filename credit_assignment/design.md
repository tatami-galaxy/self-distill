# Token-level credit assignment in OPD / OPSD — design note

Goal: **measure, save, and visualize** per-token credit (advantage) that a teacher
assigns to a student's own generations, for on-policy distillation (OPD) and
on-policy self-distillation (OPSD). No training in this phase — forward passes only.

Reference material:
- Thinking Machines, *On-Policy Distillation* (per-token reverse-KL, Fig 6).
- HuggingFaceH4 GOLD (TRL): teacher-logit + tokenizer-alignment machinery (the
  *trainer* is not what we need; the per-token signal lives inside its loss, before
  the mask-and-mean reduction).
- OPSD/SDPO family: arXiv 2601.18734v3 (privileged-teacher KL distillation; forward
  KL best), 2604.03128 (SDPO, signed per-token log-ratio advantage, Fig 6
  green/red credit), 2601.20802 (RLSD; warns privileged-teacher matching leaks PI).

## Unified credit definition

Per-token credit is the same formula for both methods; only the teacher changes:

    A_t = log π_teacher(y_t | ·) − log π_student(y_t | x, y_<t)

- **OPD:**  π_teacher = π_T, a separate frozen model.
- **OPSD:** π_teacher = π_θ(· | x, f, y_<t), same weights, privileged context f.

Policy gradient is identical in both cases (only A_t's teacher swaps):

    ∇J = E_{y~π_θ}[ Σ_t A_t · ∇ log π_θ(y_t | x, y_<t) ]

This unification is **syntactic, not semantic**:
- OPD's A_t = teacher–student capability gap.
- OPSD's A_t = information gain from PI = pointwise MI log[π(y_t|x,f)/π(y_t|x)].
  That semantic gap is the axis we are measuring, so the shared scaffold makes the
  OPD-vs-OPSD comparison meaningful rather than trivial.

## Why reverse KL is the default

Because we sample on-policy (from the student), reverse KL falls out for free:

    E_{y_t ~ π_θ}[A_t] = −KL( π_θ(·|·) ‖ π_T(·|·) )   (per position)

So the sampled-token A_t is an **unbiased single-sample estimate of the negative
per-position reverse KL** — the cheap signed scalar we sample IS the Monte-Carlo
estimate of the full-distribution quantity TM color in Fig 6.

Consequences:
- A_t is a per-token **reward**, not a centered advantage. E[A_t|state] = −KL_t ≠ 0;
  with γ=0 (TM's and SDPO's choice) there is no return-to-go and no value baseline.
- **Forward KL does not reuse this estimator.** 2601.18734 found forward KL best for
  OPSD; forward KL on student samples needs importance weights π_T/π_θ (higher
  variance) or the full distribution. → strong reason to save top-k (below) so other
  divergences are recomputations, not reruns.

## Top-k + vLLM: sufficient for the headline signal, approximate for full KL

Same tokenizer for teacher and student (start with same-family Qwen, Jaccard ≈ 0.9999),
so "per-token" is well-defined and we skip all GOLD ULD/merge machinery.

- **Sampled-token A_t (primary credit):** only needs the logprob of the *actually
  sampled* token from both models. One vLLM instance does it all:
  1. generate student rollout (returns sampled-token logprobs + top-k),
  2. teacher-force the same token ids through the teacher via `prompt_logprobs`,
  3. for OPSD, teacher-force the same ids through the same model with f prepended.
  No full vocab needed for the headline number.

- **Full-distribution KL (faithful Fig-6 + divergence sweep):** independent top-k
  lists from two models cover *different* token sets, so you cannot compute an exact
  KL from them (you lack π_T's prob on tokens outside its own top-k). From top-k you
  get only a **truncated/approximate KL** (union of supports + tail correction);
  biased, usually close for k≈20–50 on peaked LLM dists — measure sensitivity to k.
  **Exact full KL needs full logits → an HF teacher-forcing forward pass**, run only
  on the handful of traces we actually render.

Split: vLLM + top-k for everything at scale; HF full-logit pass on the visualized
subset to validate the top-k approximation and reproduce Fig 6 exactly. Save top-k
logprobs (both models, aligned on the union of supports) regardless — cheap insurance.

## Scoring conventions

- **Score at T=1, sample at T>0.** Rollouts are sampled with temperature; KL/ratio is
  computed on the models' true (T=1) distributions, consistently for both models.
- **Same prompt rendering.** Same-family ⇒ reuse the student's rendered prompt
  verbatim for the teacher. Only completion tokens get a signal (prompt is context).
- **Sign:** A_t > 0 ⇒ teacher endorses the sampled token more than student
  (reinforce); A_t < 0 ⇒ teacher dislikes it (suppress / "blame").

## Forward-looking caveats (training phase, not now)

- In OPSD the teacher shares weights with the student ⇒ A_t's teacher is
  **non-stationary** (co-moves with θ), unlike OPD's frozen teacher. This is the crux
  of the RLSD leakage warning (privileged pass drifts toward leaking f). Irrelevant
  for a single-snapshot measurement.

## Teacher / student (phase 1)

- **Teacher:** `Qwen/Qwen3-30B-A3B-Thinking-2507` (MoE 30B/3B-active, strong reasoning,
  high vLLM throughput for scoring). **Student:** Qwen3 <10B (e.g. Qwen3-1.7B / 4B).
- **Tokenizer verified identical** (vocab 151643, identical token→id map and encodings)
  across the whole original Qwen3 family: 1.7B / 4B / 8B / 14B / 32B / 30B-A3B /
  235B-A22B and the 2507 instruct+thinking variants. So per-token A_t is well-defined
  and we skip all GOLD ULD/merge machinery.
- **Qwen3.6-27B is NOT usable as a same-tokenizer teacher:** vocab 248044, ~116k
  teacher-only tokens, and even the ~131k shared token *strings* are almost all
  re-indexed (string Jaccard ≈ 0.49, id-level alignment ≈ 0). It would force the
  cross-tokenizer GOLD path — deferred.
- Note: `f` (privileged-info conditioning) is an **OPSD-only** knob. OPD has no `f`;
  its only choice is the divergence (= reverse KL for phase 1).

## Scope sweep (later)

Measure credit assignment across: choice of PI f (gold solution vs final answer vs
exec feedback), divergence (reverse / forward / JS KL), teacher choice (OPD external
vs OPSD self), correct vs incorrect traces. Start: **OPD, reverse KL, sampled-token
A_t, vLLM top-k, same-tokenizer Qwen pair.**

## Implementation plan (phase 1: OPD, reverse KL)

Three scripts; **rollouts are generated once, scoring is a separate pass per
(teacher, f, divergence)** — makes the future sweep cheap and loads one model per script.

    credit_assignment/generate_rollouts.py  # student: sample + self-score logprobs @T=1
    credit_assignment/score_teacher.py      # teacher: score same ids, compute A_t (OPD)
    credit_assignment/visualize.py          # render selected rollouts as HTML heatmaps
    data/credit_assignment/rollouts_<student>_deepmath.jsonl      # reusable
    data/credit_assignment/advantages_opd_<teacher>_revkl.jsonl   # per scoring config

**generate_rollouts.py** — subset via `DATASET_REGISTRY_EVAL["deepmath"]` + random.sample
(seed), prompt via `format_prompt_chat` + `apply_chat_template(add_generation_prompt=True)`;
tokenize prompt ourselves to keep exact `prompt_token_ids`. Generate (T>0), then **self-score
at T=1 via `prompt_logprobs`** (re-feed prompt_ids+completion_ids). Save all rollouts (correct
and incorrect) + grade via extract_boxed_answer/grade_answer.

**score_teacher.py** — load Qwen3-30B-A3B-Thinking-2507; feed the *same* ids with
`prompt_logprobs=k, temperature=0`; per token `A_t = teacher_lp[y_t] − student_lp[y_t]`
(exact, sampled-token reverse-KL form). (Full-distribution KL deferred — compute later.)

**visualize.py** — standalone HTML; per example: problem, gold/pred, correctness, completion
as colored `<span>` tokens, colored by signed `A_t` (diverging colormap: green +ve = endorse,
red −ve = blame; 2604.03128 style). Per-seq symmetric p95 clip; `white-space: pre-wrap`;
newline→`<br>`; hover tooltip.

Record schema (rollouts JSONL): problem_id, row_index, problem, gold_answer, level, subject,
student_model, prompt_token_ids, completion_text, pred_answer, correct, num_completion_tokens,
`tokens:[{token_id, token_str, student_lp, student_topk}]`, sampling{...}. `token_str` is the
incremental-decode surface form (`decode(ids[:i+1]) − decode(ids[:i])`) to handle byte-level BPE.

Gotchas: (1) use `prompt_logprobs` (T=1, no sampling-temp leak), not generation logprobs, for
*both* models. (2) top-k exact for `A_t`, approximate for KL. (3) JSONL fine at subset scale;
parquet is the scale-up path.

### Locked phase-1 config
- **Student:** `Qwen/Qwen3-1.7B` (large gap to 30B teacher ⇒ richest credit signal).
- **Thinking:** enabled on student (match the Thinking teacher's regime).
- Teacher `Qwen/Qwen3-30B-A3B-Thinking-2507`; dataset DeepMath-103K (`deepmath` loader).
- Defaults: num_samples 64, temperature 0.6, top_p 0.95, max_tokens 16384, topk 20, seed 42.
- Build order: N=8 smoke test on each script → scale to 64.

## Deferred analysis

Compare credit assignment against the independently-detected first-error step
(`error_detect/identify_errors.py`, with char offsets via `error_detect/segment.py`):
does the credit spike localize to the flagged error? (Discuss later.)
