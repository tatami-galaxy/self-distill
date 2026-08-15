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

**3. `ninja` on `PATH`.** The JIT shells out to `ninja`, which pip installs to
`.venv/bin/ninja`. Running `.venv/bin/python` directly does **not** put `.venv/bin`
on `PATH` — only `activate` does — so the build dies with:

```
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

`which ninja` returning nothing while `.venv/bin/ninja` exists is the tell. Fix it in
the same `sitecustomize.py`, for the same reason as `CUDA_HOME`.

**4. A `lib64` the linker can use.** `flashinfer/jit/cpp_ext.py` links with
`-L$cuda_home/lib64 -L$cuda_home/lib64/stubs -lcudart -lcuda`, but the nvidia wheels
install to `lib/` (no `lib64`), ship `libcudart.so.13` with no `.so` symlink for
`-lcudart` to resolve, and ship no driver stub at all:

```
/usr/bin/ld: cannot find -lcudart
```

```bash
cd .venv/lib/python3.13/site-packages/nvidia/cu13
mkdir -p lib64/stubs
ln -sf ../lib/libcudart.so.13 lib64/libcudart.so
ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 lib64/stubs/libcuda.so
```

Linking `-lcuda` against the real driver rather than a stub is fine here: the result
is dlopened into a process where `libcuda` is already resident.

**site-packages is gitignored, so neither the file nor these symlinks survive
`rm -rf .venv`.** Recreate all of it along with the pins above.

### When this actually bites

Items 3 and 4 stayed hidden for a long time because the dense Qwen3 models never
reach a JIT path in normal use, and `VLLM_USE_FLASHINFER_SAMPLER=0` sidesteps the
sampler. **The Qwen3.x-27B judges (eval/teacher_behaviors.py) remove that escape hatch**: they
are hybrid models whose Gated DeltaNet layers call FlashInfer's `gdn_prefill` kernel on
the core forward path, so the JIT toolchain has to work end to end. Two further
symptoms specific to those models:

* `max_num_seqs (1024) exceeds available Mamba cache blocks (572)` at engine init —
  one Mamba block per running sequence. Pass `--max-num-seqs 256`.
* ~54GB of bf16 weights, so it wants a GPU to itself at `--gpu-memory-utilization 0.9`.

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
