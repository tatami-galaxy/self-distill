# CodeIO output prediction

Use `--dataset codeio` with the training entrypoints, hint generation, rollout
caching, hint comparison, PI pass@k, teacher uncertainty, and advantage analyses.
Input prediction is not included: a correct input need not match a single reference
and would require execution-based verification.

## Dataset and targets

The source is the official
[CodeIO-PyEdu-Reasoning release](https://huggingface.co/datasets/hkust-nlp/CodeIO-PyEdu-Reasoning),
pinned to revision `6f45f6f4091ecd089054574ab46936994f420b24`.
The [official repository](https://github.com/hkust-nlp/CodeIO) releases only the
PythonEdu reasoning subset, not the full corpus from the
[paper](https://arxiv.org/html/2502.07316v4).

The public schema contains `prompt`, `turn_1`, `feedback_1`, `turn_2`, and
`feedback_2`; it does not contain a separate execution-ground-truth output.
This integration keeps output-prediction rows whose **first turn** received the
exact upstream feedback `Correct output!` and contains a strict JSON answer.
That response supplies both the worked demonstration and the reference output.
Failed responses, input-prediction rows, and second-turn-only successes are
excluded. This verified subset differs from the paper's unfiltered SFT setup.
For floating outputs, the reference is an upstream-accepted approximation, not
necessarily the exact execution value; tolerance-boundary decisions can therefore
differ from checking against the unavailable execution target.

Prompts retain the original problem, I/O specification, concrete input, and
reference code. The system instruction requests natural-language reasoning ending
in `{"output": ...}`. SFT preserves the original rationale without requiring
DeepMath/R1 `</think>` tags. Full/answer/hint/rollout PI remains teacher-only.
No dataset code or generated code is executed.

The loader streams the source and applies `--max-samples` after filtering and split
selection. Hugging Face caches the normalized Arrow dataset. Limits select a stable
prefix, not a random sample. Run metadata records the source, revision, target rule,
split rule, and grader version.

## Split and evaluation

Reference-code text (trailing line whitespace removed) is SHA-256 hashed; hashes
with `int(hash, 16) % 100 < 5` form the holdout. All other functions form training.
Different inputs for identical reference code stay in the same split; duplicate
prompts are removed. This is a repository-defined split, not an official CodeIO
benchmark. It does not identify semantically equivalent or differently formatted
implementations of the same function.

`eval.run_eval --dataset codeio` evaluates the first 256 eligible holdout examples.
The same evaluation supports checkpoints and reports the existing accuracy/pass@k
metrics. `eval.passk_pi` and `eval.hint_gen_compare` instead use training hint-cache
cohorts, as they do for math; their diagnostics are not holdout accuracy.

The grader reads the final JSON object and requires exactly the `output` key.
Dictionary order is ignored; list order, strings, nulls, and booleans retain their
JSON semantics. Numbers follow the upstream relative tolerance of 0.001 plus equal
integer parts. Booleans are distinct from numbers. Malformed JSON, duplicate keys,
non-finite numbers, Python literals, and extra answer keys are rejected. No `eval`
or execution-based fallback is used. These parsing/type rules are stricter than
some upstream recovery heuristics.

Hint leakage checks reject explicit output objects and distinctive literal final
values; this is still a heuristic. Use the evaluator's invalid-output rate alongside
its admissible-only sufficiency and transfer statistics.

## Example workflow

```sh
# Generate the base hint cache first; rollouts use its surviving question cohort.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m utils.gen_hints \
  --model Qwen/Qwen3-1.7B --dataset codeio --max-samples 2048 --max-tokens 128

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m utils.gen_rollouts \
  --model Qwen/Qwen3-1.7B --dataset codeio --n 4

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.opsd.train_hint_gen.train_constrained_hint_gen \
  --model Qwen/Qwen3-1.7B --dataset codeio --max-samples 1024 \
  --tau 0.7 --gamma 4 --use-lora --learning-rate 1e-5

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m eval.hint_gen_compare \
  --run-dir /path/to/codeio_hint_generator_run \
  --output-dir results/hint_gen_compare/Qwen3-1.7B/codeio_run \
  --num-problems 64 --hints-per-problem 4

# Original self-hint SDFT; learned generators use --hint-generator-model/--hint-cache.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.opsd.train_sdft \
  --model Qwen/Qwen3-1.7B --dataset codeio --pi-mode hint

# GRPO, PPO, GOLD OPD, and SFT also accept --dataset codeio.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m eval.run_eval \
  --model Qwen/Qwen3-1.7B --dataset codeio --algo base
```

These are starting commands, not tuned CodeIO hyperparameters. Reference code can
make prompts long; keep context limits and generation budgets matched across arms.
The hint cache can have fewer rows than requested because invalid hints are dropped.

`eval.answer_logprobs` remains a math boxed-answer-span analysis and rejects CodeIO.
The DeepMath validation selector is also task-specific. The teacher-behavior judge's
rubric/examples are math-oriented and have not been validated for code reasoning.
