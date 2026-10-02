#!/usr/bin/env bash
# Sweep answer, hint, full, and solution PI sequentially for one model size.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage: bash eval/run_advantage_dynamics.sh [MODEL] [EVALUATOR_OPTIONS...]

MODEL defaults to Qwen3-1.7B; use Qwen3-4B for the larger model.
Set CUDA_VISIBLE_DEVICES externally to select the GPU.
Extra options are passed to every evaluator invocation, e.g. --n 8 --steps 0 20.
USAGE
  exit 0
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

model="${1:-Qwen3-1.7B}"
if (( $# > 0 )); then
  shift
fi

for pi in answer hint full solution; do
  printf '\nRunning advantage dynamics: %s / %s\n' "$model" "$pi"
  "$repo_root/.venv/bin/python" -m eval.advantage_dynamics_sdft \
    --run-dir "/mnt/data/ujan/self-distill/outputs/sdft/$model/deepmath_$pi" \
    --pi-mode "$pi" \
    --teacher-study-dir "$repo_root/results/teacher_uncertainty/default_hint/Qwen_$model" \
    "$@"
done
