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

## Profiling

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
