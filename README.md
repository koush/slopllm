# glm.js

GPU-accelerated inference for the GLM-5.1 model (1.5TB BF16 MoE with MLA+DSA attention), implemented in CUDA with Node.js bindings and Python-verified correctness.

## Goals

- Layer-by-layer inference on a single GPU for GLM-5.1 (`zai-org/GLM-5.1`)
- Shared CUDA library (`libglm_ops.so`) callable from both Node.js addon and Python ctypes
- Op-by-op and module-by-module correctness verification against PyTorch reference
- 29 CUDA kernels covering: normalization, activation, linear ops, attention, MoE routing, and rotary embeddings
- All kernels operate on BF16 with FP32 accumulation, matching PyTorch behavior

## Requirements

- NVIDIA GPU with compute capability 12.0 (Blackwell, sm_120)
- CUDA Toolkit (nvcc)
- Node.js + npm
- Python 3 with PyTorch (nightly for Blackwell), pytest, safetensors
- GLM-5.1 model cached in Python HF cache (`~/.cache/huggingface/hub/`)

## Building

```bash
npm install                    # install node dependencies
npm run build:all              # build CUDA lib + Node addon + TypeScript
```

This produces:
- `build/Release/libglm_ops.so` — shared CUDA library
- `build/Release/glm.node` — Node.js addon

Individual build steps:
```bash
npm run build:cuda             # build libglm_ops.so only
npm run build:addon            # build CUDA lib + Node addon
npm run build                  # build TypeScript only
```

## Testing

Python tests verify each CUDA kernel against PyTorch reference via ctypes:

```bash
npm run test:python
```

Or directly:

```bash
cd tests/python && pytest -v .
```

Install Python dependencies:

```bash
pip install pytest torch safetensors
```

## Fused MoE down/reduction

GLM-5.1 enables fused NVFP4 down projection and weighted expert reduction for
local hidden size 6144, MoE intermediate size 256, eight experts per token,
and up to 32 token rows. The tested eight-GPU model has global MoE intermediate
size 2048. This
covers request batches 1–8 with the default three-token MTP draft. Other shapes
and BF16 experts use the existing path.

Set `GLM_FUSED_MOE_DOWN_REDUCE=0` to run the original model path for comparison;
unset it or use `1` to enable fusion. The setting is read at process startup.
With the persistent model loader, set it in the executor environment through
`/fork` or `/restart`; the weights do not need to be reloaded.

The kernel preserves BF16 rounding of each expert result. Shared-expert addition
and the standalone P2P barriers remain separate. The implementation is in
`csrc/glm_gemv.cu`. The model calls `Tensor.swiGluMlpMoeReduce()` with the
routing-normalization stream result. The parallel backend supplies each shard's
result and a device-local wait function to `GlmTensor`, which chooses fused or
unfused execution. Each local implementation waits before
fused down/combine, or after unfused down projection. Routing tensors remain
caller-owned.

`swiGluMlpMoeReduce()` is the Tensor-level entry point for both paths. The
unfused implementation calls the native `scatterAddRows` binding directly;
neither `scatterAddRows` nor `mulMatIdReduce` is exposed as a Tensor operation.

`withStream()` exposes `streamId`, allowing each shard to enqueue a local
CUDA wait. The producer wrapper retains ownership of stream disposal and resource
recycling; its existing `streamWaitEvent()` still waits on every device.

## Profiling

`run_glm51_multiple_mtp.ts --batch-size 1 --warmup-runs 2 --profile` runs two
complete warmup generations followed by one measured generation in the same
executor. It reuses the model, workspace, and CUDA graph cache, resetting sequence
state for each run. `--warmup-runs` defaults to 0; `--profile` calls
`cudaProfilerStart/Stop` around only the final run (including its prefill).
Generation still ends at EOS or the script's token budget (default 2000).

Launch with `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`
to record only that final run. Add `--trace=cuda,nvtx --cuda-graph-trace=node
--sample=none --cpuctxsw=none` for per-kernel graph tracing. With a resident model,
send the complete Nsight/Node command to the loader's `/spawn?follow` endpoint,
placing shared model arguments from `/model-args` after the script path. See
[loader profiling instructions](AGENTS.md#spawn-arbitrary-executables-and-profile-resident-weights).

Capture an Nsight Systems trace of a Qwen3-32B 8-GPU decode run (skips
prefill via `--delay`, captures 4s of steady-state decode):

```bash
HF_HOME=/mnt/storage/.cache/huggingface nsys profile \
    --trace=cuda \
    --output=/tmp/qwen3_p2p \
    --force-overwrite=true \
    --duration=10 \
  npx tsx src/run_qwen3_unified.ts \
    --gpus 0,1,2,3,4,5,6,7 \
    --prompt "tell me a 1000 word story" \
    --greedy \
    --max-new-tokens 1500 \
    --no-cuda-graph
```

Get the per-kernel breakdown:

```bash
nsys stats --report cuda_gpu_kern_sum --force-export=true --format=csv \
  /tmp/qwen3_p2p.nsys-rep | head -20
```

To compare with the NCCL-only baseline, prefix the run with
`GLM_DISABLE_P2P_ALLREDUCE=1`.

Or directly:

```bash
cd tests/python && pytest -v .
```

Node tests:

```bash
npm run test:node
```

## Project Structure

```
csrc/
  glm_ops.h          # C API header (29 kernel declarations)
  glm_ops.cu         # CUDA kernel implementations + C API
  glm_ops.cpp        # N-API wrapper (Node addon)
src/                  # TypeScript source
tests/python/         # PyTorch verification tests (158 tests)
  helpers.py          # ctypes bindings, model path helpers
  conftest.py         # pytest fixtures
  test_*.py           # per-op and per-module tests
vendor/
  streaming_nvfp4_quantize.py  # reference Python model architecture
binding.gyp          # node-gyp config
Makefile             # CUDA shared library build
```
# slopllm
