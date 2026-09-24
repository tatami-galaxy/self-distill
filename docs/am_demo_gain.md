# AM-Qwen3 math demonstration gain

This independent experiment scores verified math demonstrations from
`a-m-team/AM-Qwen3-Distilled` with a frozen model. Run it in the same five stages
as DeepMath: **prepare → generate PI → score → aggregate → visualize**.
Use `eval.am_demo_gain` for preparation, scoring, and aggregation. Each run writes
one report directly to its output directory.
The cohort size and generation settings need not match the DeepMath experiment.

## Run

Run from the repository root. This example prepares 128 questions; choose any
`--num-problems` value independently of DeepMath. For a pilot, use 4 questions and
a separate output directory. Preparation needs tokenizer/config and dataset
access but does not load model weights.

```sh

# 1. Prepare the AM cohort (CPU).
.venv/bin/python -m eval.am_demo_gain --phase prepare \
  --model Qwen/Qwen3-1.7B --num-problems 128 --seed 42 \
  --output-dir results/demo_gain/am_qwen3_math/Qwen3-1.7B

# 2. Generate fresh hints and rollout PI (GPU).
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m utils.gen_am_demo_pi \
  --cohort-dir results/demo_gain/am_qwen3_math/Qwen3-1.7B

# 3. Score reference demonstrations under each PI condition (GPU).
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.am_demo_gain --phase score \
  --output-dir results/demo_gain/am_qwen3_math/Qwen3-1.7B

# 4. Aggregate paired question statistics (CPU).
.venv/bin/python -m eval.am_demo_gain --phase aggregate \
  --output-dir results/demo_gain/am_qwen3_math/Qwen3-1.7B

# 5. Plot the report (CPU).
.venv/bin/python -m eval.viz.demo_gain \
  --output-dir results/demo_gain/am_qwen3_math/Qwen3-1.7B
```

Generation and scoring run in separate processes: vLLM must not initialize in
the HF scoring process. The generation command also runs hints and rollouts in
sequential child processes, releasing GPU memory between them. It reads the
requested conditions from the prepared manifest and skips unrequested PI kinds.
For the default seven conditions, each question requests four hints and one
unprivileged student rollout. Neither training nor correctness-based selection
is performed.

## Conditions and generation settings

The default conditions are `answer`, `full`, `rollout`, `hint`, `hint_detailed`,
`hint_medium`, and `hint_short`. A no-PI baseline is always scored and cached.
All selected conditions share one common eligible question set, just as in
DeepMath. Every requested generated sample must be valid, complete, and fit the
scorer's context. Failure counts are per-condition and may overlap.

To omit rollout PI, add this to the **prepare** command:

```sh
--conditions answer full hint hint_detailed hint_medium hint_short
```

Generation then produces only hints, and scoring inherits those conditions from
the manifest. `--conditions answer full` needs no generation at all. Scoring can
also override `--conditions`; aggregation follows the latest complete score
index. Changing conditions replaces the report in the run directory while reusing
compatible raw scores. If adding rollout PI after preparing without it, generate
it explicitly with `--kind rollouts` before scoring the expanded condition set.
Likewise, `--kind hints --levels ...` can add hint conditions.

Hint generation defaults to one sample per condition, temperature 0.7, top-p 1,
and no top-k filtering. Detailed/medium/short hints request 512/128/32 tokens,
with an independent **1,024-token generation ceiling**. This avoids making the
requested hint length a hard cutoff. Hints longer than their requested length
are labelled `length_noncompliant` but retained if valid and complete. The
original-style `hint` keeps its original prompt; its 128-token target is only a
length diagnostic. Requested lengths do not guarantee an information ordering.

For example, customize the generation stage with:

```sh
.venv/bin/python -m utils.gen_am_demo_pi \
  --cohort-dir "$AM_DEMO_GAIN_DIR" \
  --samples-per-level 1 \
  --detailed-target 512 --medium-target 128 --short-target 32 \
  --max-new-tokens 1024 --rollout-max-new-tokens 8192 \
  --temperature 0.7 --top-p 1.0 --top-k -1 --seed 42 \
  --batch-size 16 --gpu-memory-utilization 0.8
```

`--kind hints` or `--kind rollouts` reruns only that component. In either single
component mode, `--max-new-tokens` sets that component's ceiling; in the default
`--kind all` mode it sets the hint ceiling and `--rollout-max-new-tokens` sets the
rollout ceiling. Rollouts always use one sample from the scorer's exact model and
revision, with thinking enabled and only the original question/system prompt.
Their correctness is not evaluated for filtering.

`--model` and `--revision` can override the hint generator; in all mode these
apply only to hints. `--levels` restricts hint generation, so ensure it covers
every hint condition requested at scoring time. `--max-model-len`,
`--tensor-parallel-size`, and `--enforce-eager` control generation resources.
The context limit defaults to the prepared limit, bounded by the generator's
native context. `--output-dir` in all mode sets an alternate PI root containing
`hints/` and `rollouts/`; pass it to the scorer as `--pi-dir`. In single component
mode, `--output-dir` is that component's directory itself.

## Cohort and validation

Preparation reads `math.jsonl` from the HF cache, downloading only that dataset
file if needed. The default dataset revision is
`498448170567e330435019c5321faa0a15e19118`; change it with `--dataset-revision`.
`--data-file /path/to/math.jsonl` uses a local source and records its SHA256.
`--revision` pins the scoring model; otherwise its resolved revision is saved.
The 14.6 GB source is scanned sequentially, retaining compact offsets and
question identities rather than loading all traces into memory.

Rows must have a human/assistant pair, math category, verification score exactly
1, nonempty ground truth and thinking/final segments, and consistent
`<think>...</think><answer>...</answer>` content. These are the publisher's
answer-verification labels, not process verification. For repeated questions,
the first verified occurrence wins; conflicting gold strings exclude the
question. Seeded hash ordering selects eligible unique questions until the
requested number fits. `--num-problems 0` selects all eligible questions.
Some AM questions have DeepMath upstream provenance; they remain in this AM
corpus, but preparation does not use the local DeepMath caches.

The original demonstration and system prompt are preserved, with a documented
format-compatible fallback for missing system prompts. Explicit A–E choices
are resolved for answer PI and hint validation; unresolved letter answers are
excluded. Every condition has identical target IDs, including answer tags and
the assistant terminator. The full-demo condition must fit both the PI copy and
the target. `--max-model-len` may reduce the native context limit; nothing is
silently truncated.

Hint validation checks answer leaks using lexical rules, LaTeX normalization,
mathematical equivalence, and resolved multiple-choice content. These heuristics
can reject incidental intermediate values or miss semantic paraphrases. Inspect
`hints/audit.jsonl` and exclusion counts when interpreting the retained cohort.
Invalid and truncated generations are saved, never retried or selected by gain.

## Scoring, outputs, and resuming

Scoring and aggregation share the DeepMath implementation. The metric is the
conditional minus no-PI log likelihood of the full reference trace. HF scoring
uses frozen eval-mode weights, bf16 by default, and FP32 log-probability
reductions. `--dtype float32` and `--block-size 1024` control numerics and logit
memory; each block retains the complete causal prefix.

The run directory contains:

- `manifest.json`, `cohort.jsonl`: immutable cohort and source provenance.
- `hints/`, `rollouts/`: generated samples, manifests, caches, and audit files.
- `score_cache/demo_logps/`: reusable raw likelihood scores.
- `score_index.json`, `per_question.jsonl`, `summary.json`: the current report.
- `figures/`: gain, position, early-token, hint-length, and region plots.

AM also reports thinking/final and first-5%/remaining-95% gains. Thinking includes
its closing tag; the final region includes answer tags and the assistant
terminator. Fractional allocation at the 5% boundary conserves total gain.
Multiple hints are averaged within question, and questions are bootstrapped
jointly across conditions. Full PI can benefit from copying: every prediction
is teacher-forced on the reference prefix.

To resume, rerun the interrupted generation or scoring stage, then aggregate
and visualize. Completed requests are checksummed and cached. Use a new run
directory to change the cohort, scoring model revision, or prepared context
limit. `--force` recomputes requested scores without replacing the cohort.
The `all` phase combines prepare/score/aggregate only; it does not generate PI,
so use the explicit five-stage workflow above for a fresh default run.
