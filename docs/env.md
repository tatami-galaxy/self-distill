# Environment

Python 3.13 venv at `.venv`, created with uv:

```bash
uv venv --python 3.13
VIRTUAL_ENV=.venv uv pip install 'datasets>=4' 'trl>=0.20' \
    torch transformers accelerate vllm math-verify matplotlib tensorboard pytest
```

Pinned as of 2026-07-25: torch 2.11.0+cu130, transformers 5.14.1, trl 1.9.0,
vllm 0.25.1, datasets 5.0.0, math-verify 0.9.0.

**Keep the `datasets>=4` / `trl>=0.20` floors.** Without them uv prefers a newer
`fsspec` than `datasets` allows and backtracks `datasets` to **1.1.1** (a 2020
release), dragging `trl` down to 0.11.4 with it. There is no real conflict — the
floors just stop the resolver taking that branch.

## CUDA: FlashInfer's JIT needs a toolkit pointed out to it

vLLM compiles its sampler on demand via FlashInfer, which finds a toolkit through
`flashinfer/jit/cpp_ext.py:get_cuda_path()`: `CUDA_HOME`/`CUDA_PATH` first, else
`which nvcc`. This host's only system toolkit is Ubuntu's CUDA **11.5**, which
predates Hopper, so the default search yields:

```
nvcc fatal : Unsupported gpu architecture 'compute_90a'
```

and vLLM dies in engine init — every script here, not just eval.

Two things are needed, and both are easy to get half-right:

**1. A consistent CUDA 13.0 package set.** The nvidia wheels unpack into a shared
`site-packages/nvidia/cu13/` prefix and arrive at *mismatched* versions by default.
Each mismatch fails differently and later than the last:

| skew | symptom |
|---|---|
| nvcc 13.2 vs runtime headers 13.0 | `"CUDA compiler and CUDA toolkit headers are incompatible"` (CCCL) |
| frontend 13.2/13.3 vs ptxas 13.0 | `ptxas fatal: Unsupported .version 9.2; current version is '9.0'` |
| cudafe++ 13.0 vs cuda-crt 13.3 | `macro "__cudaLaunch" passed 2 arguments, but takes just 1` |

Install them together so uv resolves one version:

```bash
VIRTUAL_ENV=.venv uv pip install \
    "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" \
    "nvidia-nvvm==13.0.*" "nvidia-cuda-runtime==13.0.96"
```

13.0 rather than latest because torch is built against cu130. Note `uv pip install
--reinstall` on any one of these pulls the others *up* transitively and reopens the
skew — pin the whole set in a single command.

**2. `CUDA_HOME` set for every interpreter.** Entry points run as `python -m train.X`
/ `python -m eval.X`, which never sources `.venv/bin/activate`, so a shell export in
a profile does not reliably cover them. `.venv/lib/python3.13/site-packages/sitecustomize.py`
does — Python imports it automatically, including in the EngineCore subprocess vLLM
spawns:

```python
import os, os.path
_cuda_home = os.path.join(os.path.dirname(__file__), "nvidia", "cu13")
if os.path.isfile(os.path.join(_cuda_home, "bin", "nvcc")):
    os.environ.setdefault("CUDA_HOME", _cuda_home)
```

**site-packages is gitignored, so this file does not survive `rm -rf .venv`.**
Recreate it along with the pins above.

Verify with a cold FlashInfer cache and no env var set:

```bash
rm -rf ~/.cache/flashinfer/*/90a/cached_ops/sampling
env -u CUDA_HOME CUDA_VISIBLE_DEVICES=<free gpu> .venv/bin/python -m eval.run_eval \
    --model Qwen/Qwen3-1.7B --dataset aime24 --algo base \
    --num_samples 2 --n 2 --k 1 --max_tokens 64 \
    --gpu_memory_utilization 0.35 --output_dir /tmp/eval_smoke
```

It should exit 0 and leave a built `sampling.so` in the cache. The escape hatch, if
the toolchain is ever broken again, is `VLLM_USE_FLASHINFER_SAMPLER=0` — vLLM falls
back to its native PyTorch sampler and skips the JIT entirely.

## Sharing GPUs

vLLM's `gpu_memory_utilization` is a fraction of **total** device memory, not free
memory, so the 0.9 default demands an essentially idle GPU. Every vLLM entry point
here exposes a flag for it (`run_eval.py --gpu_memory_utilization`, `passk_pi.py` and
`gen_hints.py --gpu-memory-utilization`); lower it to share a card with another job.
