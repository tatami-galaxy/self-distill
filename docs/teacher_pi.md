# Teacher uncertainty and cognitive behaviors across PI types

Use the same seven conditions as the pass@k comparison:
`none answer rollout full hint_short hint_medium hint_detailed`.
The original `hint` condition is also supported if explicitly requested.

There are two stages:

1. `eval.teacher_uncertainty` samples fresh teacher completions and reports length,
   accuracy, epistemic-marker counts/rates, truncation, and unclosed thinking traces.
2. `eval.teacher_behaviors` classifies those saved completions into verification,
   backtracking, subgoal setting, and backward chaining. It loads a judge model;
   it does not regenerate teacher responses.

These commands do not train a model. Run them from the repository root and set
`CUDA_VISIBLE_DEVICES` to the GPU chosen at launch. Examples use GPU 7 sequentially.

## Source cohort and hints

`--cohort-dir` loads the prepared demo-gain cohort and completed `hints/manifest.json`
and `hints/hints.jsonl`. It reuses the same full demonstration, fixed unverified
rollout, and generated hints as [passk_pi.md](passk_pi.md). No hints are regenerated.

The cohort model must match `--problem-model`, which defaults to `--teacher-model`.
Hint model/revision and artifact checksums are checked. Select one fixed
`--hint-sample-idx` (default 0) before checking validity. A question is excluded from
every arm if any required hint is invalid or truncated. Missing artifacts are an
error; invalid hints are never replaced by a different sample.

`--num-problems 0` uses all valid questions. Positive values cap the valid list in
prepared order before context filtering. Every requested arm and the full-PI
prompt must fit within `max_model_len - max_tokens` under the problem model's
tokenizer. The retained questions are identical across arms. Metadata records
question IDs, hashes, filter conditions, exclusions, and sampling settings.

The hint variants require `--cohort-dir`. Without it, the existing hint-cache
workflow remains available for the original PI types. With a cohort, the default
PI list is the seven conditions above; the commands spell them out explicitly.

## 1. Generate uncertainty completions

For both model sizes, use eight responses per question, thinking enabled, and an
8,192-token completion budget. Thinking consumes part of that budget. Sampling is
explicit: temperature 0.6, top-p 0.95, top-k 20, seed 42, BF16 weights. These match
the documented pass@k settings, but this script generates its own responses.

```sh
for model in Qwen3-1.7B Qwen3-4B; do
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.teacher_uncertainty \
    --teacher-model "Qwen/$model" \
    --cohort-dir "results/demo_gain/solution/$model" \
    --pi-modes none answer rollout full hint_short hint_medium hint_detailed \
    --num-problems 0 --hint-sample-idx 0 --n 8 \
    --max-tokens 8192 --max-model-len 40000 \
    --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 \
    --gpu-memory-utilization 0.9 \
    --output-dir results/teacher_uncertainty/demo_gain
done
```

For a CPU preflight, append `--prepare-only` and set `CUDA_VISIBLE_DEVICES=''`.
This checks source artifacts, tokenizer identity, and prompt lengths and writes
metadata without loading model weights or generating samples. A pilot can use
`--num-problems 4 --n 2` with a separate output directory.

Keep completion saving enabled (the default), since stage 2 reads those files.
Each model writes under its slug, e.g.
`results/teacher_uncertainty/demo_gain/Qwen_Qwen3-1.7B/`:

- `teacher_uncertainty_run_meta.json`: frozen cohort and generation configuration.
- `completions_<condition>.jsonl`: response text, question ID/source index, sample
  index, correctness, token count, marker counts, and truncation/closure flags.
- `teacher_uncertainty_summary.json`: per-condition aggregate metrics and metadata.

Completion files are written atomically after each arm. Rerunning regenerates
responses; this evaluator does not reuse the pass@k generation cache. Use a new
output directory when changing generation settings, selected PI conditions, or
cohort. Identical metadata permits a rerun, but it still regenerates all arms.

## 2. Classify cognitive behaviors

After generation finishes, classify the saved responses with the existing judge
and rubric. The example keeps the first four responses per question in every arm;
use `--samples-per-problem 0` to classify all eight.

```sh
for model in Qwen3-1.7B Qwen3-4B; do
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.teacher_behaviors \
    --teacher-model "Qwen/$model" \
    --completions-root results/teacher_uncertainty/demo_gain \
    --output-root results/teacher_behaviors/demo_gain \
    --pi-modes none answer rollout full hint_short hint_medium hint_detailed \
    --classifier-model Qwen/Qwen3.8-27B \
    --samples-per-problem 4 --chunk-tokens 1000
done
```

Append `--dry-run` to plan segmentation and print token costs without loading
judge weights. The classifier tokenizer is still needed. Source arms must contain
the same selected `(question_idx, sample_idx)` pairs with consistent question IDs;
duplicate or misaligned records are rejected before judge inference.

Outputs under `results/teacher_behaviors/demo_gain/<model-slug>/` include:

- `behaviors_<condition>.jsonl`: per-segment classifier counts and character spans.
- `behaviors_meta_<condition>.json`: rubric, configuration, and source fingerprint.
- `behaviors_summary.json`: per-condition rates per 1,000 teacher tokens, mean counts
  per trajectory, prevalence, and question-clustered bootstrap intervals.

Completed classifier arms can be reused when provenance matches. Fingerprints now
include source text and metrics as well as IDs. Older ID-only caches will fail the
provenance check; use a separate output root or `--force` to reclassify them.

A classifier parse failure still drops the affected trajectory from that arm's
aggregate, as in the existing experiment. Inspect `n_trajectories_dropped`: identical
source cohorts do not guarantee identical usable cohorts after judge failures.

## Optional strong-teacher baseline

A strong teacher can generate only `none` while using the same student-cohort hint
validity and context filters. Specify the same problem model, cohort, question cap,
hint index, context/completion budgets, and all seven alignment modes:

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.teacher_uncertainty \
  --teacher-model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --problem-model Qwen/Qwen3-1.7B \
  --cohort-dir results/demo_gain/solution/Qwen3-1.7B \
  --pi-modes none \
  --align-pi-modes none answer rollout full hint_short hint_medium hint_detailed \
  --num-problems 0 --hint-sample-idx 0 --n 8 \
  --max-tokens 8192 --max-model-len 40000 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 \
  --output-dir results/teacher_uncertainty/demo_gain_strong_for_1.7B
```

The strong teacher's actual prompts must also fit its tokenizer. If they do not,
the evaluator fails instead of silently shrinking the frozen question set. Keep
strong baselines aligned to different student cohorts in separate output roots.

Epistemic markers are a lexical proxy; classifier counts are rubric-based estimates.
Report length, marker rates, truncation, and cognitive-behavior rates together.
The full and answer conditions expose the answer, so their accuracy can reflect
copying. Hint validation selects a subset of the training-cache questions, and the
retained cohorts can differ across model sizes.
