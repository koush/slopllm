# slopllm

GPU-accelerated inference for the GLM-5.1 model (1.5TB BF16 MoE with MLA+DSA attention), implemented in CUDA with Node.js bindings.

## Goals

- Fast inference of GLM-5.x (glm_moe_dsa) on 8x RTX 6000 Pro GPUs.
- Persistent model loader process for instant server updates (developments or restarts).
- OpenAI-compatible serving (`src/openai-server.ts`) with:
  - Continuous batching and chunked prefill via a generation scheduler
  - Paged KV cache with prefix sharing across requests
  - CUDA graph capture for decode, MTP speculative decoding, and phased prefill
  - Streaming (SSE) and non-streaming chat completions, tool calls, and reasoning content
  - Sampling controls (temperature, top-p, top-k, repetition/presence penalty, stop sequences)
  - Multi-GPU serving via the persistent model loader (tensor + context parallelism)
  - Prometheus metrics (`/metrics`), health check, and tokenization endpoints

## Requirements

- NVIDIA GPU with compute capability 12.0 (Blackwell, sm_120)
- CUDA Toolkit (nvcc)
- Node.js + npm
- local-inference-lab/GLM-5.3-NVFP4 model cached in HF cache (`~/.cache/huggingface/hub/`)

## Building

```bash
git submodule init
git submodule update
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

## Running

### Standalone server

The following command runs the server standalone, but for instant restarts and debugging (useful during development) the [persistent model loader](#persistent-model-loader) is highly recommended.

```bash
NCCL_P2P_LEVEL=SYS \
npx tsx src/openai-server.ts --glm51 --gpus 0,1,2,3,4,5,6,7 --arena 92 --cp --mtp \
  --host 0.0.0.0 --port 8000 --phased-prefill
```

### Persistent model loader

`src/run_model_loader.ts` decouples the model lifetime from the server lifetime. The loader process loads the weights and GPU arena once, then starts executor processes (e.g. the OpenAI server) against that resident model. The arena is exported with CUDA IPC and mapped at a process-local address by each executor, which replays the model allocation layout against the imported base. Stopping, restarting, or replacing an executor — for profiling, benchmarking, or config changes — does not reload the 1.5TB of weights; only tearing down the loader releases the model runtime.

```bash
NCCL_P2P_LEVEL=SYS \
npx tsx src/run_model_loader.ts \
  --arena 92 --gpus 0,1,2,3,4,5,6,7 --cp --glm51 --mtp \
  src/openai-server.ts --host 0.0.0.0 --port 8000 --max-pages 2048 --phased-prefill
```

Arguments before the executor path are shared model arguments passed to every executor; arguments after it apply only to that executor. A control server (default `127.0.0.1:8099`) manages the executor without touching the resident weights:

```bash
curl http://127.0.0.1:8099/status      # inspect loader and executor state
curl -N http://127.0.0.1:8099/follow   # stream loader + executor console output
curl http://127.0.0.1:8000/health      # server is ready
curl -X POST http://127.0.0.1:8099/stop     # stop the executor, model stays on GPU
curl -X POST http://127.0.0.1:8099/fork     # start it (or a new command) again
curl -X POST http://127.0.0.1:8099/restart   # stop + start the configured executor
```

### Testing host prefix caching

Start the server with `--max-host-pages <pages>` to enable pinned-host KV caching.
After a completion, flush the GPU KV cache with:

```bash
curl -X POST http://127.0.0.1:8000/admin/flush_gpu_cache \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

The server must be started with `--admin-api-key <key>`; without it the
endpoint returns 401. With `--api-key <key>`, all endpoints additionally
require `Authorization: Bearer <key>`.

The call waits for earlier generation requests to finish, offloads retained GPU
prefixes to the host cache where capacity permits, and empties the GPU cache.
Later requests wait behind the flush. The response reports `gpuPagesFreed` and
`hostPagesUsed`; model weights remain resident. Without a host tier, retained
prefixes are discarded.

Repeat the completion to exercise host restoration. Server output includes
`Host cache hit` with matched token counts and `Host cache restore complete`
with the restored prefix length and remaining prefill tokens.

To discard all offloaded host prefixes, flush the pinned-host cache with:

```bash
curl -X POST http://127.0.0.1:8000/admin/flush_host_cache \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

Like the GPU flush, it waits for earlier generation requests to finish. The
response reports `hostPagesFreed`. It requires a host tier
(`--max-host-pages <pages>`).

For prompt/output diagnostics, set `SLOPLLM_CHAT_DEBUG=1` in the server environment.
`Chat debug` log records include the exact prompt token IDs and decoded prompt,
plus generated token IDs and raw decoded output, preserving special tokens.
Call `POST /tokenize` with `messages` and `chat_template_kwargs` to log the
rendered prompt and return without generation. With the model loader, inspect
these records using `curl -N http://127.0.0.1:8099/follow`.
