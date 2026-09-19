# Two-GPU frozen-teacher scoring for hint training

Both hint trainers support an optional `--teacher-backend vllm`. One GPU runs
the policy, its colocated rollout engine, and the frozen HF transfer scorer. A
persistent process on a second GPU runs the original base model in vLLM solely
for sufficiency generation. No policy weights or LoRA adapters are sent to it.

Expose only the training GPU to the parent process. `--teacher-gpu` is the
second GPU's index or UUID in host CUDA order, not an index into the parent's
`CUDA_VISIBLE_DEVICES` list. Use a single Python training process, not a
multi-process `torchrun`/Accelerate launch.

```sh
CUDA_VISIBLE_DEVICES=4 VLLM_USE_V2_MODEL_RUNNER=0 \
uv run python -m train.opsd.train_hint_gen.train_constrained_hint_gen \
    --model Qwen/Qwen3-4B --dataset deepmath --use-lora \
    --tau 0.7 --gamma 7 --learning-rate 1e-5 \
    --teacher-backend vllm --teacher-gpu 5 \
    --teacher-rollouts 4 --teacher-top-k 20 \
    --teacher-max-completion-length 8192 \
    --teacher-gpu-memory-utilization 0.8 --teacher-max-num-seqs 32
```

The same teacher flags work with `train.opsd.train_hint_gen.train_hint_gen`.
The existing single-GPU HF teacher remains the default (`--teacher-backend hf`).
The legacy vLLM runner environment setting above matches the tested local stack.
`--teacher-enforce-eager` can disable teacher CUDA graphs for debugging.

All valid hints in a reward call are submitted together with the configured
number of independent teacher rollouts per hint. Invalid hints are omitted.
The parent computes the existing transfer scores while the teacher generates,
then grades the returned solutions and restores the original hint ordering.
Top-k, top-p, temperature, thinking template, token budget, success-fraction
reward, and constrained dual update retain their configured meanings. Switching
HF to vLLM need not reproduce identical samples or numerical probabilities.

Request seeds come from the trainer's checkpointed CPU torch RNG. The worker
has an independent CUDA environment and distributed rendezvous; it cannot join
the policy's process group. Timeouts and errors propagate to training, and the
trainer closes the worker on exit. The default request timeout is 1,800 seconds
(`--teacher-timeout`). Backend settings are included in run metadata; resuming
an older run without matching backend metadata requires the existing explicit
`--force-resume` override.

Logged metrics include teacher generation and grading seconds, generated tokens,
mean completion length, truncation fraction, transfer-scoring seconds, and total
scoring seconds. Generation and transfer overlap, so their times should not be
added to estimate total wall time. The first scoring call includes lazy startup.

## GPU smoke

Run from the repository root using two free GPUs and a fresh output directory:

```sh
CUDA_VISIBLE_DEVICES=4 VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m tests.smoke_hint_teacher_vllm \
    --teacher-gpu 5 --output-dir /tmp/hint-teacher-smoke
```

This uses the cached Qwen3-1.7B model, two synthetic math questions, two teacher
rollouts per hint, a 1,024-token teacher budget, and two real LoRA optimizer
steps. It verifies batched output counts, completed correct teacher answers,
finite transfer scores, policy updates, dual checkpoints, and identical full
teacher-weight hashes before/after policy training. The policy engine uses eager
execution; the teacher uses CUDA graphs unless `--teacher-enforce-eager` is given.
This is an integration smoke, not an 8K DeepMath throughput benchmark.
