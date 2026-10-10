#!/usr/bin/env bash
# Fresh 2x2 objective ablation with fixed, calibrated penalty coefficients.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash train/opsd/train_hint_gen/run_signal_matched.sh [MODEL] [OPTIONS]
MODEL: Qwen3-1.7B (default) or Qwen3-4B (Qwen/ prefix accepted).
  --train-gpu ID       Physical training GPU (default: 4).
  --teacher-gpu ID     Physical frozen teacher GPU (default: 5).
  --calibration FILE   Frozen calibration JSON (default: bundled per-model file).
  --run-root PATH      Parent of full/no_transfer/no_length/sufficiency_only runs.
  --arms ARM [ARM ...] Selected arms (default: all four); runs sequentially.
  --dry-run           Print commands without starting training or writing runs.
Runs use seed 42 and 100 steps. Existing nonempty run directories are rejected.
USAGE
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$repo_root"
model=Qwen3-1.7B
if (( $# > 0 )) && [[ "$1" != -* ]]; then model="${1#Qwen/}"; shift; fi
case "$model" in Qwen3-1.7B|Qwen3-4B) ;; *) usage >&2; exit 2 ;; esac
train_gpu=4
teacher_gpu=5
calibration="$repo_root/train/opsd/train_hint_gen/calibrations/$model.json"
run_root="/mnt/data/ujan/self-distill/outputs/hint_gen/$model/signal_matched_seed42"
arms=(full no_transfer no_length sufficiency_only)
dry_run=false
while (( $# > 0 )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
    --arms)
      shift; arms=()
      while (( $# > 0 )) && [[ "$1" != -* ]]; do
        case "$1" in full|no_transfer|no_length|sufficiency_only) arms+=("$1") ;;
          *) printf 'Unknown arm: %s\n' "$1" >&2; exit 2 ;; esac
        shift
      done
      (( ${#arms[@]} > 0 )) || { usage >&2; exit 2; }
      ;;
    --train-gpu|--teacher-gpu|--calibration|--run-root)
      (( $# >= 2 )) && [[ "$2" != -* ]] || { usage >&2; exit 2; }
      case "$1" in
        --train-gpu) train_gpu="$2" ;; --teacher-gpu) teacher_gpu="$2" ;;
        --calibration) calibration="$2" ;; --run-root) run_root="$2" ;;
      esac
      shift 2 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$train_gpu" =~ ^[0-9]+$ && "$teacher_gpu" =~ ^[0-9]+$ && "$train_gpu" != "$teacher_gpu" ]] || {
  printf '%s\n' 'Choose two distinct physical GPU numbers.' >&2; exit 2;
}
python="$repo_root/.venv/bin/python"
gamma="$("$python" -m eval.calibrate_hint_gen_penalties --calibration "$calibration" --model "Qwen/$model")"
# Check every destination before launching the first arm.
declare -A seen=()
for arm in "${arms[@]}"; do
  [[ -z "${seen[$arm]:-}" ]] || { printf 'Duplicate arm: %s\n' "$arm" >&2; exit 2; }
  seen[$arm]=1
  if ! "$dry_run" && [[ -e "$run_root/$arm" ]] && [[ -n "$(ls -A -- "$run_root/$arm")" ]]; then
    printf 'Run already exists: %s (use another --run-root or select remaining --arms)\n' "$run_root/$arm" >&2
    exit 2
  fi
done
for arm in "${arms[@]}"; do
  cost=1; transfer="$gamma"
  case "$arm" in no_transfer) transfer=0 ;; no_length) cost=0 ;; sufficiency_only) cost=0; transfer=0 ;; esac
  command=("$python" -m train.opsd.train_hint_gen.train_constrained_hint_gen
    --model "Qwen/$model" --dataset deepmath --output-dir "$run_root/$arm"
    --tau 0.7 --gamma "$transfer" --length-cost-coef "$cost" --hint-budget 128
    --use-lora --lora-r 16 --lora-alpha 32 --lora-dropout 0
    --learning-rate 1e-5 --max-steps 100 --save-steps 10 --seed 42
    --num-generations 8 --per-device-train-batch-size 1 --gradient-accumulation-steps 4
    --dual-init 1 --dual-lr 0.05 --dual-max 20
    --teacher-backend vllm --teacher-gpu "$teacher_gpu"
    --teacher-rollouts 4 --transfer-rollouts 4
    --teacher-max-completion-length 8192 --teacher-temperature 1 --teacher-top-k 20
    --teacher-gpu-memory-utilization 0.8 --teacher-max-num-seqs 32)
  if "$dry_run"; then
    printf 'CUDA_VISIBLE_DEVICES=%q VLLM_USE_V2_MODEL_RUNNER=%q ' "$train_gpu" "${VLLM_USE_V2_MODEL_RUNNER:-0}"
    printf '%q ' "${command[@]}"; printf '\n'
  else
    mkdir -p -- "$run_root/$arm"
    cp -- "$calibration" "$run_root/$arm/penalty_calibration.json"
    CUDA_VISIBLE_DEVICES="$train_gpu" VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}" "${command[@]}"
  fi
done
