# SDFT with solution-only PI

`--pi-mode solution` supplies the teacher with the complete reference response
**after `</think>`**: worked explanation and final answer, without the reference
thinking trace. It uses the same prompt wrapper as `full` and the same extraction
rule as the solution PI in the analysis experiments.

The student's prompt remains question-only. This changes the teacher's privileged
context, not the on-policy rollout or distillation loss: the teacher still scores
the student's sampled tokens, including its thinking tokens.

## Training

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m train.opsd.train_sdft \
  --model Qwen/Qwen3-1.7B --dataset deepmath --pi-mode solution \
  --teacher-model-kind base \
  --max-prompt-length 8192 --max-completion-length 8192 \
  --max-steps 200 --save-steps 20
```

Repeat with `--model Qwen/Qwen3-4B` for 4B. Other optimization and generation
arguments work as before. The default output directory is:

```text
/mnt/data/ujan/self-distill/outputs/sdft/Qwen3-1.7B/deepmath_solution/
```

The dataset loader requires reference solutions. It accepts traces with an
explicit `<think>` opener and R1 traces with only the closing tag. Rows with a
missing/ambiguous boundary or an empty final response are dropped; it never
falls back to exposing the whole trace. The exact teacher prompt is then checked
against `--max-prompt-length`, and overlong rows are dropped instead of truncated.
These filters run after `--max-samples`, so the resulting dataset can be smaller.
References without a `</think>` boundary are unsuitable for this PI mode.

`run_meta.json` records `pi_mode: solution` and the retained example count.
Use the same PI mode and hyperparameters when resuming a solution run; a `full`
checkpoint is not a matching `solution` resume. Existing resume checks apply.

## Evaluate checkpoints

`eval.run_eval_checkpoints` already supports `--variant solution`; no change to
its routing is required. It evaluates the trained student on ordinary benchmark
questions. `--variant` labels the training condition and result directories;
it does not supply PI to the evaluated model.

First inspect the planned sweep without loading a model:

```sh
.venv/bin/python -m eval.run_eval_checkpoints \
  --model-dir /mnt/data/ujan/self-distill/outputs/sdft/Qwen3-1.7B/deepmath_solution \
  --algo sdft --model_name Qwen3-1.7B --train_dataset deepmath \
  --variant solution --dataset aime24 --n 16 --k 1 8 16 --dry-run
```

The training directory must exist and contain `checkpoint-<number>` directories.
To evaluate, remove `--dry-run`:

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.run_eval_checkpoints \
  --model-dir /mnt/data/ujan/self-distill/outputs/sdft/Qwen3-1.7B/deepmath_solution \
  --algo sdft --model_name Qwen3-1.7B --train_dataset deepmath \
  --variant solution --dataset aime24 --n 16 --k 1 8 16
```

Repeat with the 4B training directory and `--model_name Qwen3-4B`.
Results use the usual layout, for example:

```text
results/aime24/deepmath/Qwen3-1.7B/sdft/solution/checkpoint-20/
```

The sweep visits immediate `checkpoint-<number>` directories in numeric order;
it does not include `final/`. Add `--run <name>` if multiple training runs need
separate result directories.
