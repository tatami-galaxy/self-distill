# SDFT advantage dynamics on the teacher-behavior cohort

`eval.advantage_dynamics_sdft` measures the raw token-level training signal
`log p_base(y_t | question, PI, prefix) - log p_checkpoint(y_t | question, prefix)`.
Each checkpoint generates fresh question-only responses. Those same tokens are
scored under the frozen base teacher with the run's training PI and with `none`
as a policy-drift control. Supported training arms are `answer`, `hint`, `full`,
and `solution`; `--pi-mode` must match the training run's metadata.

## Cohort and PI

Pass the model's self-teacher uncertainty study with `--teacher-study-dir`.
Its retained IDs and order define the exact question cohort used by
`eval.teacher_behaviors` and `eval.student_behaviors`. The evaluator validates the
original cohort checksum, model, dataset, and tokenizer, and reuses its original
hints and reference solutions. It does not resample the hint cache or filter on
correctness or judge success. The current studies contain 200 questions per model.

The default `--num-problems 0` uses all retained questions. A positive value uses
the first N in teacher-study order, before length filtering. `--cohort-dir` can
override the recorded source location if artifacts have moved; checksums must
still match. IDs are preserved verbatim for joins with behavior measurements.

Prompts must fit the smaller of the training prompt cap and
`max_model_len - max_completion_length`. An oversized **full-PI** prompt drops that
question from both the PI and `none` conditions for the run, without replacement.
Retained order stays fixed across checkpoints. Other prompt overflows, missing PI,
and an empty retained cohort are errors. Excluded IDs, reasons, lengths, and counts
are recorded in `cohort_meta.json`, the run manifest, and checkpoint summaries.

`solution` uses the complete worked response after the reference's `</think>`
boundary, wrapped exactly like `full`. It removes the reference thinking trace;
the student still generates its own reasoning. Rollout PI and its cache options
have been removed from this evaluator.

## Run

Sweep all four PI arms sequentially for one model size:

```sh
CUDA_VISIBLE_DEVICES=7 bash eval/run_advantage_dynamics.sh Qwen3-1.7B
CUDA_VISIBLE_DEVICES=7 bash eval/run_advantage_dynamics.sh Qwen3-4B
```

The model defaults to `Qwen3-1.7B` if omitted. Additional evaluator options follow
the model, for example `Qwen3-1.7B --n 8 --steps 0 20`. The script stops on the first
failure; rerunning reuses compatible completed generation and scoring caches.

To run a single PI arm:

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.advantage_dynamics_sdft \
  --phase sweep \
  --run-dir /mnt/data/ujan/self-distill/outputs/sdft/Qwen3-1.7B/deepmath_solution \
  --pi-mode solution \
  --teacher-study-dir results/teacher_uncertainty/default_hint/Qwen_Qwen3-1.7B
```

Repeat for `answer`, `hint`, and `full` and for the 4B model with its corresponding
teacher study and training directories. By default, the sweep includes the base
model (step zero) and every saved checkpoint. Use `--steps 0 20 40` to restrict it,
or `--step N` for one checkpoint. `--phase generate` and `--phase score` require
`--step`; `--phase aggregate` reads saved scores and verifies their provenance.

Generation defaults to two responses per question (`--n 2`) and the saved training
completion budget. It retains raw policy sampling: temperature 1, top-p 1, and
disabled top-k/min-p filtering. These settings preserve the negative reverse-KL
interpretation of expected token advantage. They intentionally differ from the
behavior studies' temperature 0.6 sampling. Shared questions do not imply shared
trajectories or paired sample indices. Increase `--n` explicitly if needed.

## Outputs and reuse

New outputs default to
`results/advantage_dynamics/teacher_cohort/<model>/<training-run>/`.
Earlier independent-cohort results under `results/advantage_dynamics/<model>/`
are preserved. Do not use an old output directory for a new study.

The cohort cache records source metadata and content hashes. Generation and score
cache identities include this provenance; changed sources or settings are rejected
unless explicitly recomputed with `--force`. Each `step-NNNNNN/` contains rollouts,
scores, their metadata, and `summary.json`; `dynamics.json` combines selected steps.

`eval/viz/advantage_dynamics_plots.py` reads the new output root and plots the four
current PI arms. Full-PI filtering can change its question set, so consult the
recorded exclusions before interpreting differences between arms. These are
training-source diagnostic questions, not a held-out benchmark.
