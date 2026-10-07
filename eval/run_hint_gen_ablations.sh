#!/usr/bin/env bash
# Evaluate all three hint objective arms on matched checkpoints, then plot.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash eval/run_hint_gen_ablations.sh [MODEL] [OPTIONS] [EVALUATOR_OPTIONS...]

MODEL: Qwen3-1.7B (default) or Qwen3-4B; the Qwen/ prefix is also accepted.
Set CUDA_VISIBLE_DEVICES externally (e.g. 4,5 for parallel scoring).

Options:
  --steps N [N ...]    Matched positive checkpoint steps (default: 20 40 60 80 100).
  --run-root PATH      Directory containing the three training runs for this model.
  --output-dir PATH    Evaluation output (default: results/hint_gen_ablations/MODEL/seed42).
  --phase PHASE        sweep (default) or summarize; both are followed by plotting.
  --dry-run           Print commands without executing them.
  -h, --help          Show this help.

Other options (e.g. --num-problems 128) are forwarded to eval.hint_gen_compare.
The script supplies checkpoint labels and matched contrasts for all three arms.
VLLM_USE_V2_MODEL_RUNNER defaults to 0 unless already set.
USAGE
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

model=Qwen3-1.7B
if (( $# > 0 )) && [[ "$1" != -* ]]; then
  model="${1#Qwen/}"
  shift
fi
case "$model" in
  Qwen3-1.7B|Qwen3-4B) ;;
  *) printf 'Unsupported model: %s\n' "$model" >&2; exit 2 ;;
esac

run_root="/mnt/data/ujan/self-distill/outputs/hint_gen/$model"
output_dir="$repo_root/results/hint_gen_ablations/$model/seed42"
steps=(20 40 60 80 100)
phase=sweep
dry_run=false
extra_options=()
while (( $# > 0 )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
    --steps)
      shift
      steps=()
      while (( $# > 0 )) && [[ "$1" =~ ^[1-9][0-9]*$ ]]; do
        steps+=("$1")
        shift
      done
      if (( ${#steps[@]} == 0 )); then
        printf '%s\n' '--steps requires positive checkpoint numbers.' >&2
        exit 2
      fi
      ;;
    --run-root|--output-dir|--phase)
      if (( $# < 2 )) || [[ "$2" == -* ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --run-root) run_root="$2" ;;
        --output-dir) output_dir="$2" ;;
        --phase) phase="$2" ;;
      esac
      shift 2
      ;;
    --model|--dataset|--run-dir|--checkpoint|--compare|--steps=*|--run-root=*|--output-dir=*|--phase=*|--model=*|--dataset=*|--run-dir=*|--checkpoint=*|--compare=*)
      printf 'Unsupported script option: %s (see --help).\n' "$1" >&2
      exit 2
      ;;
    *) extra_options+=("$1"); shift ;;
  esac
done
if [[ "$phase" != sweep && "$phase" != summarize ]]; then
  printf '%s\n' '--phase must be sweep or summarize.' >&2
  exit 2
fi

checkpoint_options=()
for step in "${steps[@]}"; do
  checkpoint_options+=(
    --checkpoint "full-step$step=$run_root/deepmath_t0.7_g6_lora_r16/checkpoint-$step"
    --checkpoint "no_transfer-step$step=$run_root/deepmath_t0.7_g0_lora_r16/checkpoint-$step"
    --checkpoint "no_length-step$step=$run_root/deepmath_t0.7_g6_c0_lora_r16/checkpoint-$step"
    --compare "no_transfer-step$step=full-step$step"
    --compare "no_length-step$step=full-step$step"
  )
done

export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
evaluate=(
  "$repo_root/.venv/bin/python" -m eval.hint_gen_compare
  --model "Qwen/$model" --dataset deepmath --phase "$phase"
  --output-dir "$output_dir" --num-problems 64
  --hints-per-problem 4 --hint-max-tokens 128
  --teacher-rollouts 4 --transfer-rollouts 4 --k 1 4
  --teacher-max-tokens 8192 --teacher-temperature 1 --teacher-top-k 20
  --generator-temperature 1 --seed 42 --teacher-seed 314159
  "${checkpoint_options[@]}" "${extra_options[@]}"
)
plot=("$repo_root/.venv/bin/python" -m eval.viz.hint_gen_ablations
      --summary "$output_dir/summary.json")

if "$dry_run"; then
  printf 'VLLM_USE_V2_MODEL_RUNNER=%q\n' "$VLLM_USE_V2_MODEL_RUNNER"
  printf '%q ' "${evaluate[@]}"
  printf '\n'
  printf '%q ' "${plot[@]}"
  printf '\n'
else
  printf 'Evaluating hint ablations: %s (steps: %s)\n' "$model" "${steps[*]}"
  "${evaluate[@]}"
  "${plot[@]}"
fi
