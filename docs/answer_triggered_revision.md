# Answer-triggered revision among backtracking-marked answer-PI responses

This study estimates the fraction of **answer-PI trajectories already marked with
at least one backtracking event by the original cognitive judge** that contain an
observed answer-triggered revision. Full, solution, hint, and unmarked answer-PI
responses are outside this study. The default sources select 469 trajectories for
Qwen3-1.7B and 434 for Qwen3-4B (from 800 originally judged responses per model).

## Event and measurement

A positive requires an ordered sequence in the saved solver response:

1. The solver derives a candidate final answer or conclusion that differs from the
   supplied answer. Intermediate results and equivalent mathematical forms do not qualify.
2. The solver explicitly recognizes the disagreement with the supplied answer.
3. It acts on that disagreement by revising a calculation, substantive assumption,
   or reasoning approach. Promising to check, copying the supplied answer, or
   ordinary verification without an explicit disagreement does not qualify.

The judge sees the question, supplied answer, and **whole response**, without the
original behavior labels or counts. It returns `yes`, `no`, or `uncertain`. A positive
includes three exact, ordered, non-overlapping evidence quotes and a revision type:
`local_correction`, `assumption_revision`, or `approach_change`. Local corrections
qualify for this event even though the original cognitive rubric reserves
backtracking for abandoning and replacing an approach.

Only the earliest clearly supported complete event is extracted: this is a binary
trajectory-incidence study, not a count of all episodes. The type breakdown describes
that extracted event; it does not measure whether any later event changes approach.
Character offsets are validated against the original response. This validates quote
fidelity and order, not the semantic correctness of the judge or mathematical
equivalence. Review positive evidence and a sample of negatives before treating
judge estimates as validated measurements.

The primary quantity is `fraction_of_selected` = positive trajectories / all selected
trajectories, reported only when every selected trajectory has a resolved label.
If classification leaves uncertain, invalid, overflowing, or pending rows, the report
instead includes:

- `fraction_among_resolved`: positives / (`yes` + `no`).
- `selected_fraction_bounds`: [positives / selected, (positives + unresolved) / selected].
- Separate counts for every unresolved status.

The bounds describe missing-label uncertainty; they are not confidence intervals
and do not account for semantic judge errors. The 95% bootstrap interval for the
resolved fraction resamples questions, retaining their selected trajectories as
clusters. This is a trajectory-weighted proportion; questions with more marked
trajectories contribute more observations. The sampling unit is the question.

Truncated solver responses stay in the denominator. The event concerns the observed
text only. A mismatch with no revision before the cutoff is not a complete event;
the truncated-response count is reported. Final-answer correctness does not filter
selection or classification, and the event does not establish causal benefit.

## Commands

Validate sources and print selection counts without writes or loading the judge:

```sh
.venv/bin/python -m eval.answer_triggered_revision --dry-run
```

Freeze the study inputs without loading model weights:

```sh
.venv/bin/python -m eval.answer_triggered_revision --phase prepare
```

On a free GPU, run a small pilot and review the extracted evidence:

```sh
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m eval.answer_triggered_revision \
  --limit 8 --output-dir results/answer_triggered_revision/pilot
```

The pilot takes the first eight selected trajectories per model; it is a pipeline
check, not a representative estimate. The full study has no limit:

```sh
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m eval.answer_triggered_revision
```

Use `--teacher-models Qwen/Qwen3-1.7B` for one model. Otherwise both model cohorts are
classified sequentially with one judge load. The default judge is Qwen/Qwen3.8-27B
with BF16 weights, temperature zero, JSON-constrained output, a 32,768-token context,
1,536 output tokens, and batches of eight complete trajectories. Inputs that do not
fit are recorded as unresolved; response text is never silently truncated.

Reruns reuse per-trajectory caches after interruption. `--phase summarize` regenerates
summaries without loading a model. `--reclassify` recomputes labels under the same
configuration. A changed rubric, judge configuration, source, or selection requires
a new output directory. Original teacher completions and judgments are preserved.

## Artifacts

Under `results/answer_triggered_revision/default_hint/<model-slug>/`:

- `selection.json`, `selected.jsonl`: original source hashes, original judge provenance,
  selected trajectory identities, questions, supplied answers, and response text.
- `score_cache/revision/`: checksummed per-trajectory judgments committed after each batch.
- `judgments.jsonl`, `judgments_meta.json`: ordered labels, evidence spans, raw judge
  outputs, rubric hash, judge revision, tokenizer identity, and runtime configuration.
- `summary.json`: conditional fraction, uncertainty bounds/interval, failure counts,
  truncated-response count, and earliest-event revision-type breakdown.

An exact positive fraction should be reported as: “Among answer-PI responses marked
as backtracking by the original judge, X% contain a judged explicit candidate-answer
mismatch followed by revision.” It is not the prevalence among all answer-PI responses,
nor the fraction of individual backtracking events attributable to answer feedback.
