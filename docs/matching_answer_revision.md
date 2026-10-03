# Matching-answer conclusions followed by reasoning revision

`eval/matching_answer_revision.py` studies saved answer-PI trajectories already
marked as backtracking by the original teacher-behavior judge. It uses the same
selection as `answer_triggered_revision.py`: 469 trajectories for Qwen3-1.7B and
434 for Qwen3-4B. It does not regenerate responses or filter on final correctness.

For **each model**, the script completes both passes before moving to the next:

1. **Discovery:** locate all distinct episodes where the solver claims to have
   derived the supplied answer, criticizes that derivation, and appears to revisit
   it. This pass does not assess mathematical validity. Repeated discussion of the
   same conclusion and criticism is grouped as one episode.
2. **Validation:** independently assess each candidate using the full response:
   matching conclusion, a specific identified flaw, the flaw's mathematical
   validity, an executed revision, and its connection to that criticism. An actual
   revision can be incorrect or unfinished; merely promising to check is insufficient.

Incomplete and uncertain candidates also go to validation, allowing the second
pass to reassess discovery's interpretation. Each episode has its own request;
there is no fixed episode-count cap. If a discovery output exceeds the token
budget or fails quote validation, the trajectory remains unresolved, not negative.
Both passes see the question, supplied answer, and full saved response. Validation
also sees the proposed episode. Neither sees original backtracking labels/counts.
One judge instance is reused across both passes and models.

## Quantities reported

Both metrics use all selected backtracking-marked trajectories as their denominator:

- **Behavioral:** a matching conclusion, specific self-criticism of its derivation,
  and an executed revision linked to that criticism.
- **Strict:** the behavioral sequence plus a judge-confirmed error or necessary
  missing justification in the criticized reasoning.

Each trajectory counts once if any candidate qualifies. A rejected first episode
does not hide a later positive. A known positive resolves trajectory incidence even
when another episode is unresolved. Otherwise invalid, overflowing, or uncertain
episodes prevent a negative trajectory label. A valid empty discovery is negative.
Episode-level status and flaw counts are also recorded; they are not trajectory
prevalences. The revision-type breakdown uses the first validated positive in
discovery order, not all revisions in the response.

As in the existing study, summaries include resolved fractions, selected-cohort
bounds for unresolved labels, and trajectory-weighted question-cluster bootstrap
95% intervals. Bounds and intervals do not capture semantic judge errors. Solver
responses truncated during original generation stay in the denominator; the event
concerns observed text only. Inputs that exceed the judge context are unresolved;
the script never silently truncates them.

Review positives, unresolved judgments, and a sample of discovery negatives before
interpreting prevalence. Validation cannot recover episodes absent from discovery.
This study measures an observable sequence; it does not establish that answer PI
caused premature conclusions or incorrect reasoning. Both this sequence and an
answer-mismatch-triggered revision may occur within one trajectory.

## Running

Run from the project root. Validate selection without writes or GPU inference:

```sh
.venv/bin/python -m eval.matching_answer_revision --dry-run
```

Prepare inputs without loading the judge:

```sh
.venv/bin/python -m eval.matching_answer_revision --phase prepare
```

For a pipeline pilot on an available GPU (set its index accordingly):

```sh
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m eval.matching_answer_revision \
  --limit 8 --output-dir results/matching_answer_revision/pilot
```

The pilot uses the first eight selected trajectories per model, not a representative
sample. To classify the full cohort, omit the limit and use the default directory:

```sh
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m eval.matching_answer_revision
```

Default execution order is 1.7B discovery → 1.7B validation → 1.7B summary →
4B discovery → 4B validation → 4B summary. `--teacher-models Qwen/Qwen3-4B`
runs just that model. Defaults: Qwen/Qwen3.8-27B, BF16, temperature zero,
32,768-token context, 4,096 output tokens per request, batch size eight. The larger
output budget accommodates multiple discovery episodes; exhausted outputs remain
unresolved. Both passes use JSON-constrained generation with thinking disabled.

Rerunning resumes cached requests. `--phase summarize` validates and summarizes
artifacts without loading the judge. `--reclassify` reruns both passes under the
same settings, including downstream validation if discovery changes. Changed
selection, rubric, or judge settings require a new output directory. Validation
caches depend on the complete discovery artifact hash as well as episode content.

## Artifacts

Under `results/matching_answer_revision/default_hint/<model-slug>/`:

- `selection.json`, `selected.jsonl`: frozen cohort and source provenance.
- `discovery.jsonl`, `discovery_meta.json`: candidate episodes, raw outputs,
  exact evidence spans, and judge configuration.
- `validation.jsonl`, `validation_meta.json`: per-episode assessments, raw outputs,
  evidence spans, and discovery provenance.
- `score_cache/discovery/`, `score_cache/validation/`: resumable per-request caches.
- `trajectory_judgments.jsonl`: separate behavioral and strict trajectory labels.
- `summary.json`: both incidence estimates and stage/episode diagnostics.

Quote checks verify fidelity and chronology, not mathematical correctness. The
two studies have separate rubrics and artifact directories; existing results are
not changed.
