#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z ${1:-} || $1 == -* ]]; then
  echo "Usage: bash eval/run_advantage_comparison.sh STUDENT_MODEL" >&2
  exit 2
fi

student=${1%/}
model_name=${student##*/}
if [[ -d $student ]]; then
  student=$(cd -- "$student" && pwd -P)
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$repo_root"

args=(
  --student "$student"
  --opd-teacher Qwen/Qwen3-8B
  --output-dir "results/advantage_comparison/$model_name"
  --tensor-parallel-size 1
  --mc-samples 16
  --mc-batch-size 128
  --pi-modes answer full hint
)

for phase in prepare generate plan; do
  CUDA_VISIBLE_DEVICES=4 uv run python -m eval.advantage_comparison "${args[@]}" --phase "$phase"
done

pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((4 + shard)) uv run python -m eval.advantage_comparison "${args[@]}" --phase mc --num-shards 4 --shard "$shard" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ $status -ne 0 ]]; then
  echo "MC failed; rerun this command to resume the cached work." >&2
  exit "$status"
fi

CUDA_VISIBLE_DEVICES=4,5 uv run python -m eval.advantage_comparison "${args[@]}" --phase score
uv run python -m eval.advantage_comparison "${args[@]}" --phase aggregate
