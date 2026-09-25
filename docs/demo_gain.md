# Final-solution likelihood gain

This experiment measures how privileged information (PI) changes a frozen
model's likelihood of the **final solution after `</think>`** in DeepMath's
`r1_solution_1`. The reference thinking trace is removed from the assistant
history. The target is the complete post-thinking response, including its worked
explanation, final answer, and assistant terminator—not only the boxed answer.

Let `s` be the final solution, `x` the question, and `c` the PI. We measure

```text
G(c) = sum_t [log p(s_t | x, c, s_<t) - log p(s_t | x, s_<t)]
```

Only earlier solution tokens appear in `s_<t`. Qwen's chat template is rendered
with `enable_thinking=False`, so an empty `<think>\n\n</think>\n\n` block is part
of the fixed assistant header. Those delimiters are not scored, and no reference
thinking tokens are inserted there. Templates must yield an exact prompt/target
boundary and a target without thinking tags.

The original complete demonstration remains available to construct PI: `full`
contains it, and hints are generated from it. Thus full PI still exposes the
reference reasoning through the privileged user prompt. Every arm, including
the no-PI baseline, uses the same empty-thinking assistant header and identical
solution target IDs.

Total gain sums token gains; normalized gain divides by solution length. Negative
values are retained. `none` is scored once as the baseline, not as a treatment.
This is a model-specific reference-solution likelihood gain, not an autonomous
solving score, semantic equivalence score, or fitted V-information estimate.

## Run

Run from the repository root. Preparation, hint generation, scoring, aggregation,
and visualization are separate stages. vLLM generation must not initialize in
the HF scoring process.

```sh
# 1. CPU preparation: no model weights.
.venv/bin/python -m eval.demo_gain --phase prepare \
  --model Qwen/Qwen3-1.7B --num-problems 128 \
  --output-dir results/demo_gain/solution/Qwen3-1.7B

# 2. GPU hint generation from the original demonstrations.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m utils.gen_hint_variants \
  --cohort-dir results/demo_gain/solution/Qwen3-1.7B

# 3. GPU likelihood scoring of final solutions.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m eval.demo_gain --phase score \
  --output-dir results/demo_gain/solution/Qwen3-1.7B

# 4. CPU statistics.
.venv/bin/python -m eval.demo_gain --phase aggregate \
  --output-dir results/demo_gain/solution/Qwen3-1.7B

# 5. CPU figures.
.venv/bin/python -m eval.viz.demo_gain \
  --output-dir results/demo_gain/solution/Qwen3-1.7B
```

Repeat with `Qwen/Qwen3-4B` and a separate output directory. A pilot can use
`--num-problems 4`. To evaluate only existing PI, omit hint generation and pass
`--conditions answer full rollout hint` during preparation.

Solution-only scoring is the sole target. Prepare a new cohort directory;
whole-trace manifests and scores are rejected. Slicing existing log-probability
arrays after `</think>` cannot produce this measurement because those predictions
were conditioned on the reference thinking trace. New model forwards are needed.

The `all` phase combines prepare/score/aggregate for a new directory with available
PI; it does not generate hints. Resume existing cohorts using explicit
score/aggregate phases.

## PI and cohort selection

Conditions are `answer`, `full`, `rollout`, `hint`, `hint_detailed`, `hint_medium`,
and `hint_short`. The existing self-hint cache supplies `hint`. Additional hint
variants independently request detailed, medium, and short guidance from the
same question and complete reference demonstration, with generation caps of
512, 128, and 32 tokens. All hints use the same PI wrapper during scoring.
Requested lengths do not guarantee an ordering of usable information.

Generator controls include `--detailed-budget`, `--medium-budget`,
`--short-budget`, `--samples-per-level`, sampling parameters, and `--seed`.
`--model` overrides the generator, including local LoRA checkpoints.
`--output-dir` and scorer `--hint-variants-dir` allow reuse of a compatible hint
artifact covering every requested question and demonstration.

Empty hints, thinking blocks, detected answer leaks, and truncated hints are
labelled and retained in generation artifacts. Scoring uses the common set of
questions for which all requested variants are valid, complete, and context
feasible. Missing samples are an error. No retries or likelihood-based selection
are performed. The answer-leak detector is lexical and can make mistakes; report
validation exclusions alongside the gain estimates.

Questions come from the model's self-hint cache and are joined to DeepMath's first
R1 trace by question plus gold answer. Seeded ordering, deduplication, and
feasibility filtering precede the requested count; `--num-problems 0` uses all
eligible questions. Conflicting source demonstrations are excluded. This is a
training-cache diagnostic cohort, not a held-out benchmark.

Rollout PI defaults to `data/pi/attempted_solution_8k`, sample 0. Selection never
reads correctness rewards. `--rollout-pi-root` and `--rollout-pi-sample-idx` select
another fixed cache.

The source must have a well-formed, unique `</think>` boundary and a nonempty
final solution. Text after that boundary is rendered as the assistant response.
Chat-template whitespace handling applies; no mathematical content is rewritten.
All prepared questions must fit the full-PI prompt plus the **solution-only**
target, and every other selected condition must fit too. No prompts or targets
are truncated. `--max-model-len` defaults to the model's native limit and may
reduce, but never extend it. Newly generated hints receive the same check.

## Scoring and reuse

HF scoring uses one unpadded sequence, frozen eval-mode weights, SDPA, bf16 by
default, and FP32 selected-token log-probability reductions. Raw probabilities
are scored without temperature or top-k filtering. `--dtype float32` is available.
`--block-size 1024` bounds logit memory by recomputing the question/PI/header plus
all preceding **solution** tokens for each block. It never restores the removed
thinking trace, resets the prefix, or slides the context.

Atomic, checksummed cache entries are keyed by exact prompt and target IDs,
model revision, tokenizer identity, and numerical settings. The manifest and
report record the solution-only target format. Completed entries survive
interruption; only a completed scoring pass publishes a new score index.
A fully cached pass does not load weights. `--force` recomputes requested scores
without replacing the prepared cohort.

Add hint conditions by generating them and rescoring with an expanded
`--conditions` list. Include any core conditions you intend to compare during
preparation. Aggregation uses the latest complete score index and its common
question set.

## Statistics and artifacts

Outputs under the experiment directory:

- `manifest.json`, `cohort.jsonl`: frozen sources, solution target IDs, and provenance.
- `hints/hints.jsonl`, `hints/manifest.json`: generated hints and validation labels.
- `score_cache/demo_logps/`: per-token solution log probabilities.
- `score_index.json`: common scored cohort and cache references.
- `per_question.jsonl`: per-hint solution gains and position curves.
- `summary.json`: question-balanced estimates and paired condition differences.
- `figures/`: total/normalized gains, relative-position/cumulative curves,
  early-token curves with support, and hint-length versus gain scatter.

Multiple hint samples are averaged within question before aggregation. Primary
normalized gain equally weights questions; the pooled-token statistic instead
weights long solutions more heavily. Confidence intervals jointly bootstrap
questions across conditions, without generator retraining or between-seed
uncertainty. Curve intervals are pointwise, not simultaneous.

Relative-position curves use 20 bins over the final solution. Fractional boundary
tokens are allocated proportionally, conserving total gain even for short targets.
Early curves cover up to 1,024 solution tokens and report decreasing support as
solutions end. These positions are not aligned semantic stages. Adjust
`--position-bins`, `--early-tokens`, and `--bootstrap-samples` without rescoring.

Full PI can benefit from literal copying. Every target prediction is
teacher-forced on earlier solution tokens, so gain measures incremental
predictability of that supplied final solution rather than independent discovery.
