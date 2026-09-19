# Hint-generator comparison: two GPUs and incremental scores

Use two independent GPUs for a sweep:

```sh
CUDA_VISIBLE_DEVICES=4,5 python -m eval.hint_gen_compare \
  --run-dir /mnt/data/ujan/self-distill/outputs/hint_gen/Qwen3-1.7B/deepmath_t0.7_g6_lora_r16 \
  --output-dir results/hint_gen_compare/Qwen3-1.7B/deepmath_t0.7_g6_lora_r16 \
  --steps 20 40 60 80 100
```

Exactly two entries in `CUDA_VISIBLE_DEVICES` automatically enable phase
parallelism when `--tensor-parallel-size=1`. Alternatively, pass `--gpus 4 5`.
These are **physical GPU indices**, or two GPU UUIDs, rather than indices remapped
inside `CUDA_VISIBLE_DEVICES`. If visibility is set, both selectors must appear
in it. Numeric selectors and UUIDs cannot be mixed.

The first GPU generates hints with one persistent base-model vLLM engine for
`fresh_base` and the requested LoRA checkpoints. Each adapter gets a distinct
request ID; the engine supports the largest requested adapter rank. Full-model
checkpoints still use separate generation processes. Once hints are ready, the
first GPU runs frozen-teacher sufficiency while the second runs Hugging Face
transfer scoring. Workers inherit their GPU visibility before importing CUDA
libraries. Each single-GPU vLLM engine runs inside its phase worker, and failure
of either scoring phase stops its peer.

A single visible GPU retains sequential scoring. Tensor parallelism remains
available for sequential vLLM phases; it cannot be combined with the two-GPU
phase-parallel mode. `--phase sufficiency` and `--phase transfer` also honor
`--gpus`, selecting the first and second GPU respectively.

## Incremental evaluation and recovery

Rerun with the same output directory and a longer checkpoint list to score only
new conditions. For example, extend `--steps 20 40` to `--steps 20 40 60` without
`--force`. Completed scores live under:

- `score_cache/sufficiency/`: one entry for each no-hint or hinted condition.
- `score_cache/transfer/`: one entry per hint and selected student rollout.
- `score_cache/student_logps/`: unhinted student token log probabilities, reused
  across generator arms and invocations.

Keys include the model identity, relevant scoring settings, actual messages,
condition identity, and (for transfer) the exact completion token IDs. Sufficiency
also keys on the gold grading target. Local model files are identified by path,
size and modification time. Remote models use their model identifier; keep the
resolved model revision fixed when sharing an output directory.

Changing hint text under the same ID, changing sampling/scoring settings, or
changing selected completion tokens produces new score entries. GPU placement
and teacher condition batch size do not invalidate scores. No-hint controls are
retained when adding checkpoints. Fully cached phases skip model initialization.

`--teacher-condition-batch-size` defaults to 64 conditions; each condition
requests `--teacher-rollouts` samples. Every completed batch saves its results;
transfer saves each rollout immediately. Interrupted phases resume from these
entries. Each teacher request retains the explicit `--teacher-seed`, so cache
hits do not consume a shared sampling RNG stream. Different GPU kernels or
request scheduling can still introduce numerical variation.

The usual `sufficiency/`, `transfer/` and `summary.json` outputs are rebuilt for
the currently selected arms, preserving the existing metrics and aggregation.
Phase metadata records cache hits/misses and elapsed time. Incomplete or stale
aggregate scores cannot be summarized. `--force` regenerates requested hints and
scores, ignoring matching condition entries; it does not purge unrelated cache
entries. Old aggregate-only scoring outputs need one scoring pass to populate
the new condition caches; existing compatible generated hints remain reusable.

Invalid hints are still evaluated, including their sufficiency and transfer.
The teacher token budget, top-k, pass@k definitions, and batch-one FP32
log-probability reduction are unchanged.

## GPU smoke

Use three existing LoRA checkpoints and a fresh artifact directory:

```sh
CUDA_VISIBLE_DEVICES=4,5 HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0 \
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m tests.smoke_hint_gen_compare \
  --run-dir /mnt/data/ujan/self-distill/outputs/hint_gen/Qwen3-1.7B/deepmath_t0.7_g6_lora_r16 \
  --steps 10 20 30 --gpus 4 5 \
  --output-dir results/debug/hint_compare_smoke
```

This evaluates two synthetic arithmetic questions, extends a two-checkpoint sweep
with a third checkpoint, and repeats with all caches populated. It checks adapter
engine reuse, GPU assignment, phase overlap, finite transfer values, cache hit
counts, and exact retention of earlier scores. It is a correctness smoke, not a
full-length throughput benchmark. All artifacts go into the requested directory.
