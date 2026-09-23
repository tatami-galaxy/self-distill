# Demonstration likelihood gain

This experiment scores DeepMath's `r1_solution_1` with a frozen model. It measures
how PI changes the likelihood of the complete reference trace, including its
reasoning, final answer, and assistant terminator. It is a model-specific
log-likelihood gain, not a fitted V-entropy estimator or semantic equivalence score.

For each target token, `gain = conditional_logp - no_pi_logp`. Total gain sums
these values; normalized gain divides by target length. Negative values are kept.
`none` is scored once and cached as the reference; it is not a treatment row.

## Run

Run each command from the repository root. These are separate processes on
purpose: vLLM generation must not initialize in the HF scoring process.

```sh
# CPU preparation (tokenizer/config and dataset access, no model weights).
.venv/bin/python -m eval.demo_gain --phase prepare \
  --model Qwen/Qwen3-1.7B --num-problems 128 \
  --output-dir results/demo_gain/Qwen3-1.7B

# GPU hint generation; defaults to the cohort's own model.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m utils.gen_hint_variants \
  --cohort-dir results/demo_gain/Qwen3-1.7B

# GPU likelihood scoring. No generation or training.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m eval.demo_gain --phase score \
  --output-dir results/demo_gain/Qwen3-1.7B

# CPU statistics and figures.
.venv/bin/python -m eval.demo_gain --phase aggregate \
  --output-dir results/demo_gain/Qwen3-1.7B
.venv/bin/python -m eval.viz.demo_gain \
  --output-dir results/demo_gain/Qwen3-1.7B
```

Repeat with `Qwen/Qwen3-4B` and a separate output directory. Start with
`--num-problems 4` in a pilot directory to inspect formatting, hint validity,
and memory use before a full experiment. No experiment is launched by the tests.

To evaluate only existing PI, omit the generation step and pass
`--conditions answer full rollout hint` during prepare. The `all` phase combines
prepare/score/aggregate for a new directory with available PI. An existing cohort
is immutable: use explicit score/aggregate phases to resume.

## Conditions and targets

The existing `hint` cache is preserved. Additional conditions are `hint_detailed`,
`hint_medium`, and `hint_short`, with generation caps of 512, 128, and 32 tokens.
They request progressively fewer solution details, independently from the same
question and demo. All receive the same PI_HINT wrapper during scoring. Their
lengths are not assumed to establish an ordering of usable information.

Generator controls include `--detailed-budget`, `--medium-budget`, `--short-budget`,
`--samples-per-level`, `--temperature`, `--top-p`, `--top-k`, and `--seed`.
`--model` overrides the default generator (including local LoRA checkpoints).
`--output-dir` and scorer `--hint-variants-dir` allow a shared hint artifact.
Model-specific cohorts can differ: shared generator weights alone do not make
cross-model comparisons paired. Shared hints must cover each requested cohort.

Empty outputs, thinking blocks, and detected answer leaks are labelled. Truncated
hints are also retained and flagged. No retries or likelihood-based selection are
performed. The existing answer-leak detector is lexical, not a semantic guarantee.
The main scorer uses only questions with all requested new hint samples valid,
complete, and context-feasible. Missing samples are an error. The manifest reports
all generation failures, and the score index reports excluded-question counts.
This is a conditional-on-validity comparison, not an all-generated-output estimate.

Questions originate in the model's existing self-hint cache. They are joined to
the first R1 trace by question plus gold answer, sampled in a seeded order, and
filtered before taking the requested count (`--num-problems 0` uses all eligible
questions). Duplicate question identities are removed. Question/answer pairs with
conflicting source demonstrations are excluded and counted. This is a training-cache
diagnostic cohort, not a holdout benchmark. Rollout PI defaults to
`data/pi/attempted_solution_8k` sample 0. It is selected without reading rewards;
`--rollout-pi-root` and `--rollout-pi-sample-idx` select another fixed cache.

The SFT thinking formatter and full chat rendering determine the completion,
including its terminator. The rendered prompt must be a token-exact prefix of the
full conversation. Target token IDs must match in every condition. Bad traces or
unsupported template boundaries are reported, never silently rewritten further.
The source demonstration itself is retained verbatim for the full-PI prompt.

All prepared questions must fit the full-demo condition, which includes the demo
in both prompt and target. Other selected conditions must fit too. There is no
prompt or target truncation. `--max-model-len` defaults to the model's declared
native context limit and may reduce, but never extend it. Newly generated hints
receive the same feasibility check. Report the exclusions and target-length
statistics when interpreting results; long demonstrations can be underrepresented.

## Scoring and reuse

HF scoring uses one unpadded sequence, SDPA, frozen eval-mode weights, bf16 by
default, and FP32 selected-token log-softmax reductions. Raw model probabilities
are scored without temperature or top-k filtering. `--dtype float32` is available.
`--block-size 1024` limits vocabulary-logit memory: each target block is evaluated
with the entire causal prefix recomputed. It does not reset or slide the context.
Increasing this size trades more memory for fewer repeated prefix forwards.
Block size is included in cache provenance because numerics can change slightly.

Each atomic, checksummed score entry is keyed by exact prompt and target IDs,
model identity/revision, tokenizer identity, and numerical settings. Completed
entries survive interruption. Only a completed scoring pass publishes a new
score index. A fully cached pass does not load model weights. `--force` recomputes
requested scores; it does not replace the prepared cohort or remove other entries.

Add hint conditions by generating the levels and rerunning score with an expanded
`--conditions` list. Baseline and existing conditions reuse their scores for the
retained questions. Preparation should include any core PI conditions you intend
to compare. Aggregation always follows the latest complete score index, with all
selected conditions using its common question set.

## Statistics and artifacts

Outputs under the experiment directory:

- `manifest.json`, `cohort.jsonl`: frozen sources, rendered token IDs, selection metadata.
- `hints/hints.jsonl`, `hints/manifest.json`: raw hints, labels, generation provenance.
- `score_cache/demo_logps/`: per-token conditional and baseline log probabilities.
- `score_index.json`: common scored cohort and references to exact cache entries.
- `per_question.jsonl`: per-hint total/normalized gain, token gains, and curves.
- `summary.json`: question-balanced statistics and all paired condition differences.
- `figures/`: total/normalized bars, relative-position/cumulative curves, early-token
  curves with question support, and hint-length versus gain scatter.

Multiple hint samples are averaged within question before aggregation. Primary
normalized gain is the mean of per-question means; the distinct pooled-token
statistic weights long traces more heavily. Confidence intervals bootstrap
questions jointly across conditions. They do not include generator retraining or
between-seed uncertainty. Curve intervals are pointwise, not simultaneous bands.

Relative-position curves use 20 bins by default. Fractional boundary tokens are
allocated proportionally using cumulative interpolation, preserving total gain
and avoiding empty bins in short traces. Cumulative curves end at mean total gain.
Early curves default to 1,024 token positions and report support as traces end.
These are token positions, not aligned semantic stages. Use `--position-bins`,
`--early-tokens`, and `--bootstrap-samples` to recompute statistics without rescoring.

The full-demo condition can benefit from literal copying. Every position is
teacher-forced on the preceding demonstration tokens. Gains measure incremental
predictability along that supplied path, not autonomous discovery or semantic
reconstruction. The target includes the final answer; no reasoning-only claim is
made by the aggregate metric.
