# Hint objective ablations

The constrained trainer uses

```text
R(h) = lambda * (S(h) - tau) - length_cost_coef * C(h) - gamma * T(h)
C(h) = generated_hint_tokens / hint_budget
lambda <- clip(lambda + dual_lr * (tau - mean(S)), 0, dual_max)
```

Sufficiency remains active in every arm. The four arms complete a 2×2 ablation
of the explicit length and transfer penalties:

| Arm | `--length-cost-coef` | `--gamma` |
| --- | --- | --- |
| Full objective | 1 | 6 |
| No transfer penalty | 1 | 0 |
| No length penalty | 0 | 6 |
| Sufficiency only | 0 | 0 |

All components are still measured during training, including transfer at gamma=0.
The no-length arm retains the 128-token hard cap and the original brief-hint prompt:
it removes the explicit length penalty, not all influences on length. Keep the cap,
prompt, training cohort/order, rollout cache, backend, optimizer, sampling, LoRA,
invalid-output handling, and dual settings matched. Each arm starts from the base
model; do not resume a full-objective checkpoint to create an ablation.

## Train the two ablations

These commands match the existing `deepmath_t0.7_g6_lora_r16` configuration.
Choose free GPUs; the parent sees only the training GPU, and `--teacher-gpu` is
the separate physical GPU used by the frozen teacher. Commands run sequentially.
Change `ablation_model` to `Qwen3-4B` to repeat the experiment at 4B.

```sh
ablation_model=Qwen3-1.7B

CUDA_VISIBLE_DEVICES=4 VLLM_USE_V2_MODEL_RUNNER=0 \
.venv/bin/python -m train.opsd.train_hint_gen.train_constrained_hint_gen \
  --model "Qwen/$ablation_model" --dataset deepmath \
  --tau 0.7 --gamma 0 --length-cost-coef 1 --hint-budget 128 \
  --use-lora --lora-r 16 --lora-alpha 32 --lora-dropout 0 \
  --learning-rate 1e-5 --max-steps 100 --save-steps 10 --seed 42 \
  --num-generations 8 --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --dual-init 1 --dual-lr 0.05 --dual-max 20 \
  --teacher-backend vllm --teacher-gpu 5 \
  --teacher-rollouts 4 --transfer-rollouts 4 \
  --teacher-max-completion-length 8192 --teacher-temperature 1 --teacher-top-k 20 \
  --teacher-gpu-memory-utilization 0.8 --teacher-max-num-seqs 32

CUDA_VISIBLE_DEVICES=4 VLLM_USE_V2_MODEL_RUNNER=0 \
.venv/bin/python -m train.opsd.train_hint_gen.train_constrained_hint_gen \
  --model "Qwen/$ablation_model" --dataset deepmath \
  --tau 0.7 --gamma 6 --length-cost-coef 0 --hint-budget 128 \
  --use-lora --lora-r 16 --lora-alpha 32 --lora-dropout 0 \
  --learning-rate 1e-5 --max-steps 100 --save-steps 10 --seed 42 \
  --num-generations 8 --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --dual-init 1 --dual-lr 0.05 --dual-max 20 \
  --teacher-backend vllm --teacher-gpu 5 \
  --teacher-rollouts 4 --transfer-rollouts 4 \
  --teacher-max-completion-length 8192 --teacher-temperature 1 --teacher-top-k 20 \
  --teacher-gpu-memory-utilization 0.8 --teacher-max-num-seqs 32
```

## Train sufficiency only

Set both coefficients to zero, preserving tau, the adaptive multiplier, validity
rules, and the other training settings. This starts a fresh LoRA run from the base
model. Repeat with `ablation_model=Qwen3-4B` for the larger model.

```sh
ablation_model=Qwen3-1.7B
CUDA_VISIBLE_DEVICES=4 VLLM_USE_V2_MODEL_RUNNER=0 \
.venv/bin/python -m train.opsd.train_hint_gen.train_constrained_hint_gen \
  --model "Qwen/$ablation_model" --dataset deepmath \
  --tau 0.7 --gamma 0 --length-cost-coef 0 --hint-budget 128 \
  --use-lora --lora-r 16 --lora-alpha 32 --lora-dropout 0 \
  --learning-rate 1e-5 --max-steps 100 --save-steps 10 --seed 42 \
  --num-generations 8 --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --dual-init 1 --dual-lr 0.05 --dual-max 20 \
  --teacher-backend vllm --teacher-gpu 5 \
  --teacher-rollouts 4 --transfer-rollouts 4 \
  --teacher-max-completion-length 8192 --teacher-temperature 1 --teacher-top-k 20 \
  --teacher-gpu-memory-utilization 0.8 --teacher-max-num-seqs 32
```

For valid hints the reward is now `lambda * (S - tau)`. Length and transfer are
still measured but do not affect this reward. Invalid hints keep the existing
relative penalty and count as zero sufficiency in the dual update. The hard
128-token cap and brief-hint prompt remain. This is an ablation of the constrained
objective, not a change to fixed-weight accuracy maximization: lambda can reach
zero, at which point valid hints have equal rewards until the dual rises again.

Default run directories under `/mnt/data/ujan/self-distill/outputs/hint_gen/<model>/`:

- Full: `deepmath_t0.7_g6_lora_r16` (existing baseline).
- No transfer: `deepmath_t0.7_g0_lora_r16`.
- No length: `deepmath_t0.7_g6_c0_lora_r16`.
- Sufficiency only: `deepmath_t0.7_g0_c0_lora_r16`.

The default length coefficient is one, preserving the old objective and output
names. Run metadata and dual state record the coefficient; incompatible resumes
are rejected. Old dual states without the field are interpreted as coefficient
one. Resume with the original arguments plus `--resume-from-checkpoint PATH`.
Use explicit new `--output-dir` paths for additional seeds or repeated runs.
Reuse the old baseline only when its data and training settings match; otherwise
repeat the full arm with gamma=6 and length-cost-coef=1 in a new directory.

## Evaluate matched checkpoints

Use one output directory for all arms, giving them a common frozen question cohort,
one freshly sampled base-generator control, and one no-hint solver control.
`--compare CANDIDATE=REFERENCE` adds question-paired contrasts against the full arm;
the existing contrasts against `fresh_base` are also retained.

Run the Bash wrapper for each model after the checkpoints exist. It evaluates
steps 20, 40, 60, 80, and 100, then generates the plots:

```sh
CUDA_VISIBLE_DEVICES=4,5 bash eval/run_hint_gen_ablations.sh Qwen3-1.7B
CUDA_VISIBLE_DEVICES=4,5 bash eval/run_hint_gen_ablations.sh Qwen3-4B
```

To add the sufficiency-only checkpoints, use the same output directory and
evaluation settings with `--include-sufficiency-only`:

```sh
CUDA_VISIBLE_DEVICES=4,5 bash eval/run_hint_gen_ablations.sh Qwen3-1.7B \
  --include-sufficiency-only
CUDA_VISIBLE_DEVICES=4,5 bash eval/run_hint_gen_ablations.sh Qwen3-4B \
  --include-sufficiency-only
```

This reuses the frozen cohort, old hints and condition scores, and shared controls;
only the new generator arm needs inference. The joint summary and figures are
rebuilt for all four arms. Keep the flag on subsequent invocations to retain that
four-arm view; omitting it rebuilds the original three-arm view without deleting
the cached sufficiency-only data. All selected checkpoints must exist.

Additional paired contrasts compare sufficiency-only against full, and each of
the original arms against sufficiency-only. In particular, `no_length` versus
`sufficiency_only` isolates adding transfer pressure when length cost is absent;
`no_transfer` versus `sufficiency_only` isolates adding length pressure when
transfer cost is absent. To plot those contrasts instead of differences from full:

```sh
.venv/bin/python -m eval.viz.hint_gen_ablations \
  --summary results/hint_gen_ablations/Qwen3-1.7B/seed42/summary.json \
  --reference sufficiency_only \
  --output-dir results/hint_gen_ablations/Qwen3-1.7B/seed42/versus_sufficiency_only
```

Outputs default to `results/hint_gen_ablations/<model>/seed42/`. The wrapper uses
the same 64-question, four-hint, four-teacher-rollout settings described above,
and defaults `VLLM_USE_V2_MODEL_RUNNER` to 0 unless already set. GPU visibility is
inherited from the caller. It stops on evaluation failure before plotting.

Use `--steps` to change the matched checkpoints, `--run-root` to change the
directory containing the training runs, and `--output-dir` to choose another result
directory. Other evaluator options are forwarded. For example:

```sh
bash eval/run_hint_gen_ablations.sh Qwen3-1.7B --steps 10 20 40 --dry-run
CUDA_VISIBLE_DEVICES=4,5 bash eval/run_hint_gen_ablations.sh Qwen3-4B \
  --steps 20 40 --num-problems 128 \
  --output-dir results/hint_gen_ablations/Qwen3-4B/q128_seed42
```

`--dry-run` prints the evaluation and plotting commands without running them.
Include step 10 only if available in every selected run. Evaluate one model size and
training seed per summary. The plotter expects `PREFIX-stepN` labels;
The default plotter includes sufficiency-only automatically when present;
custom prefixes can be supplied with repeated `--arm NAME=PREFIX` and
`--reference NAME`. It checks matched steps, training seeds, targets, and budgets.

Evaluation caches are reusable when adding checkpoints. To add contrasts or
rebuild the new statistics from completed scores, rerun the **same evaluation
script command**, retaining all sampling/cohort arguments and checkpoints, with
`--phase summarize`. No models are loaded in that phase. The plotter also uses
only local JSON and optionally reads checkpoint dual states; unavailable dual
states are displayed as gaps. Existing older summaries need resummarization.

## Read the results

`summary.json` preserves all-output, admissible-only, and difficulty-stratified
statistics. New `metric_estimates` provide question-bootstrap intervals for length,
normalized cost, raw/clamped transfer, invalid/truncated fractions, and sufficiency.
`training_aligned_sufficiency` is the question-balanced mean success fraction with
invalid hints assigned zero. It retains questions with no valid hints, unlike an
admissible-only view. Its lift over the no-hint baseline is reported separately.
Truncation alone remains distinct from invalidity, matching training behavior.

The plotter writes `ablations/attributes.png/.pdf/.csv` and
`ablations/differences.png/.pdf`. The main figure contains the three attributes,
invalid/truncated hint rates, and the checkpointed dual multiplier. The sufficiency
panel includes tau and unhinted accuracy; the transfer panel shows raw and clamped
estimates. Difference panels show each ablation minus the full objective at the
same step. Lower transfer cost means greater compatibility under the fixed-rollout
proxy, not demonstrated downstream student learning.

Keeping the sufficiency constraint does not guarantee finite-step attainment.
Read target violations and baseline lift alongside lambda, especially when it
approaches `dual_max`. Inspect all-output versus admissible-only sufficiency in
the summary before attributing gains to useful hints: leaked answers can inflate
all-output scores, and lexical validation does not establish semantic secrecy.

Intervals are pointwise and resample questions; independent hint/sample indices
are not paired outcomes. They do not estimate variation from retraining. Replicate
all selected arms across seeds before making a robustness claim. The default cohort
is a training-source diagnostic subset; a held-out study needs an explicitly
excluded question split. A no-length arm that saturates the token cap motivates a
separate matched larger-budget experiment, rather than changing just that arm.
