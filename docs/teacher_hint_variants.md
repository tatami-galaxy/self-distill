# Self-teacher behavior under detailed, medium, and short hints

This separate pipeline imports the existing accuracy estimators, uncertainty
measurements, and cognitive rubric. It does not modify the original evaluators.
One response set supplies accuracy, uncertainty, and cognitive measurements.
These are training-source diagnostic questions, not a held-out benchmark.

## Design

Run each model on its original frozen teacher-study question IDs, in their saved
order. A positive `--num-problems` selects the first N for a pipeline pilot; it
is not a random or representative evaluation sample. Zero uses every question.

The same frozen model generates three independent hints from each question and
full reference demonstration. Requested lengths are 512/128/32 tokens for
detailed/medium/short; all have a separate 1,024-token generation cutoff. The
prompt asks for coaching rather than a completed solution. Actual lengths are
reported and may exceed the targets. Hint styles are not a verified ordering of
information content. One fixed hint is generated per question per level, with
no outcome-based selection or retries.

Teacher arms are `none`, `hint_short`, `hint_medium`, and `hint_detailed`. Hinted
arms share the original `PI_HINT` wrapper. The original demonstration is never
included in teacher solving prompts. Hint generation disables thinking; teacher
responses enable it. Model revisions and tokenizer identities are pinned.

## Generate and inspect hints

Generate hints for the full frozen cohort on GPU 7:

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m utils.gen_teacher_hint_variants \
  --teacher-study-dir results/teacher_uncertainty/default_hint/Qwen_Qwen3-4B \
  --output-dir results/teacher_hint_variants/Qwen3-4B/hints
```

Use Qwen3-1.7B's corresponding study and output directory for that model.
`--dry-run` reads the source and tokenizer without writing artifacts or loading
model weights.

The generator writes `experiment.json`, `hints.jsonl`, `hints_meta.json`, and
`diagnostics.json`. Invalid and truncated hints remain in these artifacts.
Inspect examples alongside the question and answer in `experiment.json`.

**A lexical pass is not proof of answer secrecy.** The existing detector misses
some spelled-out answers, equivalent fractions, and conclusions such as an
impossibility proof when the supplied answer is "No". It can also flag incidental
intermediate values. Teacher accuracy may reflect this answer disclosure.

Optional complete review JSON can adjudicate these cases before evaluation:
Copy the generated `review_template.json` to a separate review file and replace
each pending decision and empty reason. The template includes the question,
gold answer, hint text, and automatic flag for inspection. Generator reruns
replace the template, so keep completed reviews in a different file.

```json
{
  "version": 1,
  "hints_hash": "<hash from hints_meta.json>",
  "reviews": [
    {
      "question_id": "<saved ID>",
      "condition": "hint_short",
      "hint_hash": "<eval.hint_compare_cache.digest(hint_text)>",
      "decision": "accept",
      "reason": "Names a method without revealing the requested result."
    }
  ]
}
```

Include exactly one entry for every hint, using `accept` or `reject` plus a
reason. Review can overturn a lexical leakage flag or reject an unflagged hint;
it cannot override empty, truncated, or otherwise malformed hints. The review
is hashed and frozen into the experiment. Partial reviews are rejected.
Without review, outputs are explicitly labelled
`lexical_only_not_semantically_validated`.

## Run the full sweep

The default workflow prepares the cohort, generates teacher responses, computes
accuracy and uncertainty, classifies cognitive behaviors, and writes plots:

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.teacher_hint_variants --phase sweep \
  --hints-dir results/teacher_hint_variants/Qwen3-4B/hints \
  --output-dir results/teacher_hint_variants/Qwen3-4B
```

Add `--hint-review PATH` for reviewed selection. Preparation freezes the common
subset with three usable hints and context-feasible prompts in every arm.
It records all excluded IDs and reasons, without replacement. Controls use the
same retained questions. A context overflow never silently truncates a prompt.
Adding `--dry-run` makes preparation read-only.

Main-study defaults: eight responses/question/arm, 8,192 completion tokens,
temperature 0.6, top-p 0.95, top-k 20, 40,000-token context, seed 42, BF16. Seeds
are derived per question and condition. Identical seeds do not promise bitwise
reproducibility across different inference batches or software versions.

Resume the prepared run and execute all remaining stages sequentially:

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.teacher_hint_variants \
  --phase sweep --output-dir results/teacher_hint_variants/Qwen3-4B
```

Solver and judge run in separate processes so their GPU allocations do not
overlap. Sweep uses the saved generation settings; omit the original `--hints-dir`
when resuming. Changed source artifacts or settings require a new directory.

For independent stages or custom judge options:

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.teacher_hint_variants \
  --phase generate --output-dir results/teacher_hint_variants/Qwen3-4B
.venv/bin/python -m eval.teacher_hint_variants \
  --phase summarize --output-dir results/teacher_hint_variants/Qwen3-4B
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.teacher_hint_variant_behaviors \
  --phase sweep --output-dir results/teacher_hint_variants/Qwen3-4B
.venv/bin/python -m eval.viz.teacher_hint_variants \
  --output-dir results/teacher_hint_variants/Qwen3-4B
```

The judge defaults to Qwen/Qwen3.8-27B, with a resolved pinned revision, the exact
original rubric, temperature 0.7, top-p 1, 1,000-token paragraph-aligned chunks,
no preceding-context paragraphs, and 1,024 output tokens. Classification defaults
to the first `min(4, n)` responses per question. Use `--samples-per-problem` to
change this before the first classification. `--judge-model` and
`--judge-revision` can select another judge. `--dry-run` plans chunks without
loading judge weights. `--phase summarize` loads no tokenizer or model weights.

Generation caches complete question/arm groups immediately; classification
caches each completed batch's chunks. Interruptions resume without repeating
committed requests. Failed judge outputs are cached as failures; they are not
silently retried or converted into zero counts. One failed chunk drops the
entire trajectory from cognitive statistics only.

## Measurements and artifacts

Accuracy uses all responses: pass@1/2/4/8 where `k <= n`. Uncertainty includes
the same ten lexical markers, raw counts, pooled counts per 1,000 generated
tokens, thinking/post-thinking counts, length, truncation, and unclosed traces.
All truncated teacher responses remain in these statistics.

Cognitive metrics are verification, backtracking, subgoal setting, and backward
chaining: counts/trajectory, prevalence, and pooled rates per 1,000 teacher
tokens. Both the count numerator and token denominator exclude failed
trajectories. These are judged textual behaviors, not validated internal states.

95% bootstrap intervals resample questions, retaining responses within each
question. Contrasts are each hint versus `none`, detailed minus medium, and
medium minus short. Cognitive contrasts require complete judged sample groups
in both arms and report their exact retained question IDs. Independent response
indices are never treated as paired outcomes. Hint randomness is not separately
estimated: results are conditional on the fixed generated hints.

Each arm directory contains `completions.jsonl`, `behaviors.jsonl`, and
`trajectories.jsonl`, plus checksummed metadata. Raw completions preserve token
IDs and finish reasons. Root files include:

- `experiment.json`: inputs, exclusions, generation settings, and provenance.
- `judge_config.json`: pinned judge and rubric identity.
- `summary.json`: accuracy, uncertainty, intervals, and contrasts.
- `behaviors_summary.json`: cognitive metrics, exclusions, and contrasts.
- `comparison.png`, `comparison.pdf`, `comparison.csv`: plots and estimates.

## Initial pilot observations

The first 1.7B prompt avoided cutoff truncation but often reproduced solutions.
The revised coaching prompt (`teacher_hint_soft_targets_v2`) produced mean hint
lengths of about 41/124/431 tokens for 1.7B and 32/96/323 for 4B, with no truncated
hints in either 20-question pilot. The automatic common subsets were 12/20 and
20/20 respectively. Manual spot checks found answer disclosures missed by the
lexical detector in both models. These are **pipeline diagnostics**, not clean
estimates of the effect of hint information. Review hints before the full study.

Original pilot artifacts are preserved under `results/teacher_hint_variants/pilot/`;
revised ones are under `results/teacher_hint_variants/pilot_v2/`.

The revised 4B pipeline pilot completed all four arms on 20 questions with two
responses per question/arm: 160 completions and 1,039 cognitive segments.
All judge outputs parsed successfully; no trajectories were dropped. Its figure
is `results/teacher_hint_variants/pilot_v2/Qwen3-4B/comparison.png`, with PDF and
CSV counterparts. This pilot uses lexical selection and includes missed answer
disclosures; its accuracy changes should not be attributed solely to hint quality.
