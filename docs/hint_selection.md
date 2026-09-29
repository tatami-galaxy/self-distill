# Hint selection without generator training

For each question and demonstration, sample **8 hints at temperature 1.4** using
the existing hint prompt. Reject empty hints, thinking blocks, detected answer
leaks, and hints cut off by the 128-token limit. Generate **8 frozen-teacher
solutions per valid hint** to estimate sufficiency. The teacher receives the
question and hint, never the demonstration.

For each question, define `best = max(S)` over valid candidates and retain those
with `S >= max(0, best - epsilon)`. Choose the candidate minimizing

```text
generated_hint_tokens / hint_budget + gamma * transfer_cost
```

Defaults are **epsilon=0.125**, **gamma=6**, and four fixed cached student
trajectories per hint. Transfer cost matches constrained training: compute raw
`log p_student - log p_hinted_teacher`, average tokens within each trajectory,
then average trajectories equally, then clamp the result at zero. Raw scores
are retained. `--no-clamp-transfer` disables the final clamp. Ties prefer higher
sufficiency, fewer tokens, then lower candidate index. This is a per-question
empirical constraint, not the trained generator's population-average constraint.

There is no unhinted-teacher baseline constraint. If every valid candidate has
zero successes, the literal rule still selects the cheapest and labels it
`selected_zero_success`. If all candidates are invalid, the question is labelled
`no_valid_hints` and omitted from the training export; no replacement question or
fallback hint is silently substituted. Both outcomes are counted in the summary.

## Run a small pilot

Use an existing HF cohort to avoid reloading DeepMath. This samples ten questions
with matching cached student trajectories; the selected cohort is frozen on disk.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m eval.hint_selection \
  --model Qwen/Qwen3-1.7B \
  --cohort-dir results/hint_gen_compare/Qwen3-1.7B/deepmath_t0.7_g6_lora_r16/cohort \
  --num-questions 10 \
  --output-dir data/pi/hint_selection/deepmath/Qwen3-1.7B/pilot_n8_k8_t1.4
```

Without `--cohort-dir`, load solution-bearing rows from the training dataset.
The default count is 128; `--num-questions 0` uses all eligible questions.
Eligibility requires at least `--transfer-rollouts` cached unhinted trajectories
under `data/rollouts/<dataset>/<model>/`. Selection uses the lowest stored sample
indices without inspecting rewards. Legacy rollout caches without sample indices
use stored per-question row order, recorded explicitly in the cohort. Model/dataset identity and available answer
identities are checked. Prompts and student tokens are never silently truncated.

The hint sampler uses top-p 1, no top-k filtering, and thinking disabled. Teacher
sampling matches the constrained runs: temperature 1, top-p 1, top-k 20, up to
8192 tokens. A teacher solution reaching the cap is graded normally and remains
in the denominator. Hint/teacher sample counts, temperatures, budgets, gamma, and
epsilon are configurable. Pin a remote model using `--revision` for reproducibility.

All stages run sequentially on the visible GPU(s). Generator and teacher vLLM
engines run in separate spawned processes from HF transfer scoring. The frozen
HF scorer uses one GPU. `--teacher-batch-size` controls simultaneous hinted
conditions (default 8, each with 8 teacher samples).

## Reuse and epsilon comparisons

Preparation and the three inference stages are independently resumable:
`--phase prepare`, `generate`, `sufficiency`, `transfer`, and `select`.
After preparation, pass `--output-dir` to identify the experiment; inference
settings come from its saved manifest. `--phase select` loads no model weights.

```sh
.venv/bin/python -m eval.hint_selection --phase select \
  --output-dir data/pi/hint_selection/deepmath/Qwen3-1.7B/pilot_n8_k8_t1.4 \
  --epsilon 0
```

Changing epsilon, gamma, or clamping reuses all candidates and scores and writes
a separate selection directory. Changing N, K, sampling, source, or model settings
requires a new experiment directory. `--force` reruns the requested inference
stages under the same configuration. Completed per-question generations and
per-condition scores survive interruptions. The frozen source snapshot includes
the exact demonstrations, generator prompt IDs, and student completion IDs.

Artifacts are isolated from existing caches:

```text
data/pi/hint_selection/<dataset>/<model>/<experiment>/
  manifest.json, cohort.json
  candidates.json, sufficiency.json, transfer.json
  score_cache/{candidates,sufficiency,student_logps,transfer}/
  selections/epsilon_0.125_gamma_6_<fingerprint>/
    selection.json     # winner, threshold, eligible IDs and status per question
    summary.json       # coverage, exclusions, settings, selected empirical S
    hints/             # HF dataset, directly usable by SDFT
```

The existing `data/pi/hint` caches are never updated. Input/output overlap with
the established cache directories is rejected. Explicitly pass the printed new
`hints/` path to SDFT:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.opsd.train_sdft \
  --model Qwen/Qwen3-1.7B --dataset deepmath --pi-mode hint \
  --hint-cache <printed-selection-directory>/hints \
  --output-root /mnt/data/ujan/self-distill/outputs/sdft_selected_hint
```

Use distinct SDFT output roots for epsilon/gamma variants too. The generator
identity stays the frozen base model; selection is recorded in additional cache
columns and selection artifacts. No trained hint-generator checkpoint is needed.

## Interpretation

Selected sufficiency is **used for selection**, not an independent evaluation.
Use fresh teacher rollouts for that subsequent comparison. The existing lexical
leak detector is retained for comparability; it has known false positives and
false negatives (including equivalent mathematical spellings). Passing it is
not a guarantee against answer leakage. Inspect the selected hints before
interpreting their sufficiency as evidence of useful guidance.
