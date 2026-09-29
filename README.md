# slopllm

GPU-accelerated inference for local-inference-lab/GLM-5.3-NVFP4, implemented in CUDA with Node.js bindings.

## Goals

- Fast inference of GLM-5.x (glm_moe_dsa) on 8x RTX 6000 Pro GPUs.
- Persistent model loader process for instant server updates (developments or restarts).
- OpenAI-compatible serving (`src/openai-server.ts`) with:
  - Continuous batching and chunked prefill via a generation scheduler
  - Paged KV cache with prefix sharing across requests
  - CUDA graph capture for decode, MTP speculative decoding, and phased prefill
  - Streaming (SSE) and non-streaming chat completions, tool calls, and reasoning content
  - Sampling controls (temperature, top-p, top-k, repetition/presence penalty, stop sequences)
  - Prometheus metrics (`/metrics`), health check, and tokenization endpoints
- Multi-GPU serving with tensor parallelism and context parallelism via the persistent model loader.

## Requirements

- NVIDIA GPU with compute capability 12.0 (Blackwell, sm_120)
- CUDA Toolkit (nvcc)
- Node.js + npm

Fetch the model into the HF cache:

```bash
hf download local-inference-lab/GLM-5.3-NVFP4
```

## Docker

The prebuilt image is published on Docker Hub (`koush/slopllm`). Clone the repository for its `docker-compose.yml`, then start the server:

```bash
git clone https://github.com/koush/slopllm.git
cd slopllm
docker compose up -d --wait    # pulls koush/slopllm and starts the server (all GPUs, host networking + IPC)
docker compose logs -f         # follow server output
docker compose down            # stop
```

Requirements: the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) and the model already downloaded into the host Hugging Face cache (see [Requirements](#requirements)).

The compose file runs the OpenAI server with `--arena 92 --mtp --cp` across all GPUs, mounts the host Hugging Face cache read-only at its default path, and sets `NCCL_P2P_LEVEL=SYS` by default — tuned with the best defaults for an 8x RTX Pro 6000 host. Edit `docker-compose.yml` (or its environment variables) to customize. Model load takes several minutes (~90GB of weights across 8 GPUs); the health check has a long start period to match.

## Running

### Standalone server

The following command runs the server standalone, but for instant restarts and debugging (useful during development) the [persistent model loader](#persistent-model-loader) is highly recommended.

```bash
NCCL_P2P_LEVEL=SYS \
npx tsx src/openai-server.ts --gpus 0,1,2,3,4,5,6,7 --cp --mtp \
  --host 127.0.0.1 --port 8000
```

`NCCL_P2P_LEVEL=SYS` works around NCCL routing host-staged collectives on multi-GPU hosts.

Smoke test once `/health` responds:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

### Server arguments

`npx tsx src/openai-server.ts --help` prints the full list. Common arguments:

| Argument | Description | Default |
|---|---|---|
| `--host`, `--port` | Listen address | `127.0.0.1:8000` |
| `--gpus <list>` | GPU device IDs for tensor + context parallelism | `0` |
| `--arena <int>` | Per-GPU arena (GiB): one large CUDA allocation shared by weights and KV | `92` with the GLM model |
| `--cp` | Context parallelism: interleave KV tokens across GPUs | off |
| `--mtp [int]` | MTP speculative decoding draft tokens | off; `3` when passed without a value |
| `--batch-size <int>` | Maximum concurrent requests | `8` |
| `--chunk-size <int>` | Prefill token budget | `8192` |
| `--max-pages <int>` | GPU KV cache pages | `batch-size × ceil(chunk-size / 64)` |
| `--max-host-pages <int>` | Pinned-host KV pages for prefix offload/restore | `0` (disabled) |
| `--api-key <key>` | Bearer token required on all endpoints | unset |
| `--admin-api-key <key>` | Bearer token enabling `/admin` endpoints (cache flush) | unset |

> **Note:** the server does not perform an exhaustive warmup across all context length and batch size combinations. When a new combination is first encountered, CUDA graphs must be captured for it, causing a momentary stall at these graph compile points. Subsequent requests with the same shape replay the captured graph without stalling.

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

## Development and the Model Loader

`src/run_model_loader.ts` decouples the model lifetime from the server lifetime. The loader process loads the weights and GPU arena once, then starts executor processes (e.g. the OpenAI server) against that resident model. The arena is exported with CUDA IPC and mapped at a process-local address by each executor, which replays the model allocation layout against the imported base. Stopping, restarting, or replacing an executor — for profiling, benchmarking, or config changes — does not reload the weights; only tearing down the loader releases the model runtime.

```bash
NCCL_P2P_LEVEL=SYS \
npx tsx src/run_model_loader.ts \
  --arena 92 --gpus 0,1,2,3,4,5,6,7 --cp --mtp \
  src/openai-server.ts --host 127.0.0.1 --port 8000 --max-pages 2048
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
