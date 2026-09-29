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

## Generate a subset for a 200–1,000-step SDFT run

Start with **2,048 questions**. The current Qwen3-1.7B and Qwen3-4B student
rollout caches each cover 2,048 questions with four trajectories per question.
Load solution-bearing rows from DeepMath and select this bounded cohort with
matching cached trajectories; preparation freezes the cohort on disk. Repeated
questions with the same reference answer may have different worked solutions:
preparation keeps the first demonstration that fits the context limit in the
seeded shuffle order, with one selected row per question. Conflicting reference
answers among rollout-eligible source rows still raise an error; duplicates
outside the rollout-backed cohort do not block preparation.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m eval.hint_selection \
  --model Qwen/Qwen3-1.7B \
  --dataset deepmath \
  --num-questions 2048 \
  --output-dir data/pi/hint_selection/deepmath/Qwen3-1.7B/q2048_n8_k8_t1.4
```

The count is explicit; the script default remains 128. Omit `--cohort-dir` to
avoid restricting the run to the small evaluation cohort. Questions with no valid
hint are omitted from the export, so use `summary.json`'s `n_selected` when
calculating the actual number of passes. Requesting more than 2,048 questions
requires extending the cached student rollouts first.

For the single-GPU training command below, the effective batch is
`1 GPU × 1 example × 16 accumulation steps = 16`, with one on-policy rollout per
prompt. A 2,048-row export therefore gives approximately **128 optimizer steps
per pass**:

| Optimizer steps | Prompt presentations | Approximate passes over 2,048 rows |
| --- | --- | --- |
| 200 | 3,200 | 1.56 |
| 500 | 8,000 | 3.91 |
| 1,000 | 16,000 | 7.81 |

Start with **500 steps** and change `--max-steps` to 200 or 1,000 as needed.
SDFT cycles through the selected questions and generates fresh on-policy
completions on subsequent passes; the selected hints stay fixed. The number of
unique questions does not need to equal the number of prompt presentations.
With multiple training GPUs, multiply the effective batch by the number of
processes: four GPUs at the same per-device settings give 64 examples per step,
or about 32 steps per pass. To limit repeated question exposure at that batch
size, generate student rollouts and selected hints for a larger subset.

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
  --output-dir data/pi/hint_selection/deepmath/Qwen3-1.7B/q2048_n8_k8_t1.4 \
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
  --max-steps 500 --num-generations 1 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
  --output-root /mnt/data/ujan/self-distill/outputs/sdft_selected_hint_q2048_eps0125
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
