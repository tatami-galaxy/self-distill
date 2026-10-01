# Student cognitive behaviors and uncertainty verbalization

`eval.student_behaviors` generates fresh **question-only** responses from SDFT
checkpoints. It uses the exact question IDs and order recorded in the teacher
study in [teacher_pi.md](teacher_pi.md), without reapplying PI filters.
Supported training conditions are
`answer`, `hint`, `full`, and `solution`; rollout-trained runs are excluded.
Run one model size per output directory. These are training-source diagnostic
questions, not a held-out benchmark.

The pipeline reuses the teacher scripts' uncertainty measurements, cognitive
rubric, judge, segmentation, and aggregation. Generation defaults come from the
teacher metadata (currently eight samples, 8192 tokens, temperature 0.6, top-p
0.95, top-k 20). It classifies the first four samples by default. Truncated
responses remain in the generation statistics. A failed judge chunk excludes
its whole trajectory only from cognitive statistics. Paired differences against
the base resample questions; they never pair independent sample indices.

## Select checkpoints

Pass explicit names and training directories with repeated `--run NAME=PATH`.
`--checkpoint-selection all` selects every immediate `checkpoint-N` directory
in numeric order. `final/` is excluded to avoid duplicating the final saved step.
A single shared base-model generation supplies step zero for every run.

Alternatively use `--checkpoint-selection best --selection-benchmark aime24`.
Each `--benchmark-results NAME=PATH` names that run's benchmark directory,
containing `checkpoint-N/{summary.json,results.json}`. Selection maximizes
`summary["pass_at_k"]["pass@1"]` with 16 samples per question: **avg@16**.
It checks the score against the saved samples and compares question identities
and recorded generation settings. Ties choose the earlier step. Missing results
are listed, and a winning checkpoint with missing weights is an error.

Run the full 1.7B study, selecting the best checkpoint for each PI using AIME24 avg@16:

```sh
student_training_root=/mnt/data/ujan/self-distill/outputs/sdft/Qwen3-1.7B
student_benchmark_root=results/aime24/deepmath/Qwen3-1.7B/sdft
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.student_behaviors --phase sweep \
  --teacher-study-dir results/teacher_uncertainty/default_hint/Qwen_Qwen3-1.7B \
  --run answer="$student_training_root/deepmath_answer" \
  --run hint="$student_training_root/deepmath_hint" \
  --run full="$student_training_root/deepmath_full" \
  --run solution="$student_training_root/deepmath_solution" \
  --benchmark-results answer="$student_benchmark_root/answer/run-1" \
  --benchmark-results hint="$student_benchmark_root/hint/run-1" \
  --benchmark-results full="$student_benchmark_root/full/run-1" \
  --benchmark-results solution="$student_benchmark_root/solution" \
  --checkpoint-selection best --selection-benchmark aime24 \
  --output-dir results/student_behaviors/Qwen3-1.7B
```

The sweep selects checkpoints, generates responses, classifies behaviors, and
writes summaries and plots. Append `--dry-run` to print the selection and response
counts without writing or loading models. Use `--phase prepare` to save only the
experiment and selection manifests.

For 4B, change the model paths and use `hint` rather than `hint/run-1` for the
benchmark directory. Its older hint summaries lack `eval_config`; add
`--allow-legacy-benchmark-results` to use them. Their sample counts, actual
avg@16, and question sets are still verified, but missing benchmark identity
and generation metadata are explicitly recorded as provenance gaps.

The initial preparation stage writes `experiment.json` and
`selection-best-aime24.json`. Selections remain frozen on subsequent calls.
To incorporate newly evaluated checkpoints,
repeat preparation with the original input arguments and `--refresh-selection`.
Adding another named training run also requires refreshing the selection.

## Resume or run individual phases

To resume a prepared or interrupted sweep, the saved plan supplies the source
and checkpoint paths:

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.student_behaviors --phase sweep \
  --checkpoint-selection best --selection-benchmark aime24 \
  --output-dir results/student_behaviors/Qwen3-1.7B
```

Use `--phase generate`, `--phase classify`, or `--phase summarize` separately
when placing solver and judge work on different GPUs. Each solver checkpoint
runs in a fresh process; classification keeps one judge loaded across pending
checkpoints. `summarize` loads neither a model nor a tokenizer. Judge generation
options have a `--judge-` prefix, separate from student sampling options.

Generation commits each completed question's samples; classification commits
each completed batch's segments. Reruns reuse these caches after interruptions.
Completed artifacts carry content checksums and configuration provenance.
`--reclassify` replaces judgments without regenerating student responses.
Changing the bootstrap count only changes summaries. To expand from best to
all checkpoints, prepare an `all` selection in the same output directory with
the same `--run` arguments; existing checkpoint caches are reused.

Each `NAME/checkpoint-N/` (and shared `base/`) contains `completions.jsonl`,
`completions_meta.json`, `behaviors.jsonl`, `behaviors_meta.json`,
`trajectories.jsonl`, and `summary.json`. Raw completions include generated
token IDs and finish reasons. Root `summary-best-aime24.json/.png` or
`summary-all.json/.png` contains the selected study and plots. Accuracy and
uncertainty use all generated samples; cognitive statistics use successfully
judged samples, with exclusions reported separately.

Matched teacher references are read from the source study and its corresponding
`teacher_behaviors/default_hint` directory; `--teacher-behaviors-dir` overrides
the latter. Their source fingerprints and judge configuration must match before
cognitive references are plotted. Legacy teacher judge revisions were not
recorded, which remains explicit in the report. Best-mode plots label the
selected step and benchmark avg@16; all-mode plots show trajectories over steps.
The selection benchmark chooses weights; behavior measurements always use the
frozen teacher-study questions and their own generation budget.
