# GLM.js Architecture Notes

## Overview

TypeScript inference engine for the GLM-5.1 model on NVIDIA GPUs. Ships a native C++/CUDA addon (`glm.node`) wrapped by TypeScript operator classes. Supports multi-GPU via tensor parallelism and context parallelism, CUDA graph capture for decode and prefill, and paged KV caching with prefix sharing. Qwen3 and Qwen3.5 are also implemented but serve primarily as broader correctness test targets, not production targets.

## Core Abstractions

### Tensor (`src/tensor.ts`)
- Abstract base class. Concrete implementations: `GlmTensor` (single GPU) and `ParallelTensor` (multi-GPU).
- Each tensor owns a GPU memory pointer (`data`), shape, dtype, and a reference to its `WorkspaceBase`.
- Operations (`.linear()`, `.rmsnorm()`, `.bmm()`, etc.) allocate output tensors from the workspace and call into the native addon.
- `SamplingWorkspace` extends `WorkspaceBase` for GPU-side top-k/top-p sampling with repetition penalty.

### Workspace (`src/workspace.ts`)
- `WorkspaceBase` manages GPU memory lifecycle via **dispose-recycle pooling**:
  - `alloc()` checks the device or pinned-host disposed pool for best-fit reuse and only calls `newTensor()` when no reusable allocation exists.
  - Unnamed tensors are tracked temporaries. Named tensors are persistent; duplicate named allocations throw, while `ensureAlloc()` returns a compatible existing tensor.
  - Disposed pinned tensors remain in `synchronizingHost` until device synchronization completes, preventing reuse while an asynchronous copy may still reference them.
  - **`freeze()`** prevents further allocations after model loading so weight addresses remain stable.
  - **`startTracking()`** returns a scope that disposes unnamed temporaries on exit. `clearTracking()` provides explicit phase-boundary cleanup and can preserve tensors from a `TensorTree`.
- This pattern is what makes the inference loop graph-capturable: the same GPU addresses are reused each step deterministically.

### Tensor Lifetime and `using` Pattern

Tensors implement `Disposable` via `[Symbol.dispose]()`. Model code can use `using` to release intermediates at the end of a lexical scope:

```typescript
using gate = hiddenStates.linear(weights.gate);
using up = hiddenStates.linear(weights.up);
using activated = gate.siluAndMul(up);
return activated.linear(weights.down);
```

Tracking is normally owned by the caller around a complete model operation. A returned tensor must be removed from that tracking scope:

```typescript
function forward(state: ExecutionState): Tensor {
  using _tracker = state.ws.startTracking();
  return model.forwardModel(state).removeTracking();
}
```

Key rules:
- Named tensors cannot be disposed and live until the workspace is freed.
- `using` on an unnamed tensor auto-disposes at block scope exit.
- `startTracking()` cleans all tracked tensors at once; use individual `using` declarations when an intermediate should be released earlier.
- `removeTracking()` stages the tensor and its backing view chain. When the tracking scope exits, the tensor returns to `tracked` under caller ownership.
- Allocations made on an alternate stream are not recycled until its `withStream()` wrapper is disposed. Call `streamWaitEvent()` before consuming the result on the current stream.

### Allocator (`src/allocator.ts`)
- `ArenaAllocator`: bump allocator with 256-byte alignment. `free()` is a no-op. Used for the weight arena when `--arena` flag is set — a single large `cudaMalloc` carved up linearly.
- For non-arena mode, `WorkspaceBase` uses `GlmOps.alloc/free` (real `cudaMalloc`/`cudaFree`) but the dispose-recycle pool means actual allocations rarely happen after warmup.

### GlmOps (`src/glm_ops.ts`)
- Single-GPU device backend. Wraps the native addon (`glm.node`).
- Owns CUDA context, stream management (7 alternate streams via `withStream()`), and the allocator.
- CUDA graph API: `graphBeginCapture/EndCapture/Instantiate/Launch/Destroy`.
- FlashInfer integration: plan/run split for batch prefill/decode and MLA attention.
- NCCL and P2P primitives exposed for `ParallelOps`.

### ParallelOps (`src/parallel_ops.ts`)
- Multi-GPU backend implementing `DeviceOps`. Wraps N `GlmOps` instances.
- `ParallelTensor`: has N shards (one per GPU), delegates ops to each shard, inserts collective communication (AllReduce, AllGather) as needed based on `TensorParallelism` annotations.
- `TensorParallelism` enum: `Replicated`, `Column` (output-dim sharded), `Row` (input-dim sharded), `PartialSum` (needs AllReduce), `PartialSoftmax` (for CP merge).
- Communication: NCCL AllReduce/AllGather for large tensors; custom P2P kernels for small AllReduce/AllGather (≤8192 elements) and fused RMSNorm+AllReduce.
- Per-device shard workspaces created lazily via `getShardWorkspaces()`.

## Tensor Parallelism

Weight parallelism is assigned per-tensor during loading.

Linear op parallelism rules (weight × input → output):
- Column × Replicated → Row (no comm)
- Row × Row → PartialSum (AllReduce needed)
- Replicated × Replicated → Replicated (no comm)
- Row weight → AllGather weight to Replicated, then proceed (K-dim mismatch)
- Row input → AllGather input to Replicated, then proceed (K-dim mismatch)

## Context Parallelism (Interleaved Token-per-GPU)

When `--cp` flag is set (GLM-5.1 only), GPUs operate as context-parallel shards for attention while retaining tensor-parallel sharding for linear layers:

- **KV cache is Row-sharded**: each GPU stores every Nth token's KV (tokens interleaved across GPUs). `PagedKVCache` uses `TensorParallelism.Row` for ckv/kpe tensors.
- **Position IDs**: decode/prefill kernels receive `cpWorldSize=N` and `cpRank=i`. The CUDA kernel assigns position `i, i+N, i+2N, ...` to GPU `i`.
- **Prefill**: Each GPU runs MLA prefill over its token subset with `cpWorldSize`/`cpRank` params. The FlashInfer plan computes `effectivePageSize = pageSize / worldSize` and `effectiveNumHeads = numHeads` (not sharded). Output is `PartialSoftmax` — each shard has partial attention output + log-sum-exp.
- **CP Merge**: Partial attention outputs are combined with an online-softmax merge. Small decode workloads use a custom P2P path; larger or prefill workloads use AllGather plus ReduceScatter.
- **Page allocation**: `PagedKVCache` distributes pages round-robin across GPUs. Page `p` is stored on GPU `p % worldSize`. The effective page size per GPU is `pageSize / worldSize`.
- **KV and Indexer K Prefetch**: During prefetch, the the next layer's kv is prefetched to avoid exposed q gather and CP merge on the critical path. Sparse CKV prefetc is also implemented for decode, but may be disabled since the performance gain was within run variance.

## Paged KV Cache (`src/paged_kv.ts`)

- Fixed `PAGE_SIZE=64` tokens per page. Pages are ref-counted for prefix sharing (only full pages are shared; partial last pages are copied).
- `Sequence`: ordered list of pages tracking `allocLen` and `tokenIds`.
- `PagedKVCache` extends `WorkspaceBase` — KV cache tensors (`kData[]`/`vData[]` or `ckvData[]`/`kpeData[]`) are pre-allocated GPU buffers indexed by layer and page ID.
- Dirty flags (`pagesDirtyHost`, `pagesDirtyDevice`, `positionIdsDirty`) control conditional updates — plan calls and host→device copies are skipped if nothing changed.
- MLA path: uses `ckvData[layer]` `[maxPages, pageSize, kvLoraRank]` and `kpeData[layer]` `[maxPages, pageSize, qkRopeDim]` instead of separate K/V.
- Standard path: uses `kData[layer]` `[maxPages, nKv*pageSize*hd]` and `vData[layer]`.

## Execution Workspace (`src/execution-workspace.ts`)

- `ExecutionWorkspace` extends `WorkspaceBase`, pre-allocating all GPU and pinned buffers needed for prefill/decode at construction time. No GPU allocations happen in the hot path.
- `ExecutionState`: holds per-step context (batchSize, totalTokens, seqLens, isDecode, cache reference).
- **Plan/Run split**:
  - `planPrefill()`: allocates pages, fills indptr/positionIds/slotMapping on host, calls FlashInfer plan kernel, copies indices to device.
  - `planDecode()`: allocates decode token pages, updates position IDs (only if dirty), calls FlashInfer decode plan (only if pages changed), copies indices.
  - `forwardPrefill/forwardDecode()`: runs the model forward pass using the planned state.
- FlashInfer plan writes workspace buffers (floatWs, intWs, planInfo). Run reads them. This split is essential for CUDA graph capture — plan runs outside the graph, run is captured.

## CUDA Graph Capture (`src/capture-manager.ts`)

- `CaptureManager`: key → `{warmupSteps, graphExec}`. First 3 calls run eagerly (warmup). On the 3rd warmup call, `graphBeginCapture()` is called, the lambda runs, then `graphEndCapture()` + `graphInstantiate()`. Subsequent calls with the same key launch the cached graph directly via `graphLaunch()`.
- Works because all GPU pointers are stable (workspace recycling). The plan step (which writes to plan buffers and may change kernel selection) runs outside the graph.
- Decode path: `captureManager.run(() => { inputIdsBuf.memcpy, decodeStep, model.forward, computeLogits, doSample }, ['decode'])`.

## Model Loading (`src/chat_model.ts`, `src/glm51_model.ts`)

- `ChatModel` extends `WorkspaceBase` — the model IS its weight workspace.
- `fromPretrained(modelDir)`: scans for safetensors files, memory-maps each shard, calls `loadTensor()` for each weight. After loading, calls `freeze()` to lock the workspace.
- `loadTensor()` is model-specific:
  - GLM-5.1: splits MLA fused weights (q_b_proj → q_nope_proj + q_pe_proj, kv_b_proj → k_nope_proj + v_proj), computes absorbed weight (k_nope_proj @ q_nope_proj), handles NVFP4 scale renaming.
  - Weights are loaded with mmap+DMA (`mmapLoadAsync`) for zero-copy GPU upload.
  - F32 weights (norms, A_log) are converted to BF16 on load.

## Inference Flow

1. **Prefill**: `planPrefill()` → `prepareInput()` → `forwardInput()` → `model.forward(state)` → `computeLogits()`
2. **Decode loop**: `planDecode()` → `captureManager.run({ inputIdsBuf.memcpy, decodeStep, model.forward, computeLogits, doSample })` → copy token to host → `reportTokens()` → yield

## Scratchpad

The `scratchpad/` directory is gitignored and intended for ad-hoc scripts, diagnostics, and experimentation. It is not checked in — use it for anything temporary that shouldn't pollute the repo (e.g., comparing hidden states across layers, debugging KV cache issues, testing new kernels in isolation).

## Testing

```bash
npm run build:all          # build CUDA addon + TypeScript (required before any tests)
npm run test:python        # Python tests: pytest tests/python/ - validates CUDA kernels against PyTorch
npm run test:node          # TypeScript tests: end-to-end model inference via tsx --test
npm test                   # runs both
# individual tests:
cd tests/python && pytest -v test_linear.py   # single Python test
pytest -v test_linear.py            # specific GPU
npx tsx --test tests/test_glm51.ts            # single TS test
npx tsx --test tests/test_parallel.ts  # multi-GPU
```

# Persistent Model Loader

`src/run_model_loader.ts` loads the model and GPU arena once, then starts executor processes against that resident model. Each arena is exported with CUDA IPC and opened at a process-local address by the executor, which replays the model allocation layout against that imported base. Stopping or restarting an executor does not reload the weights. Stopping the loader process releases the model runtime.

Executor CUDA contexts, streams, cuBLAS handles, NCCL communicators, graphs, and workspaces are process-local and are released by process teardown. The loader remains the sole owner of the exported arena allocations. For custom direct-P2P collectives, each executor GPU context opens every owner's arena handle and translates tensor pointers through the resulting per-reader/per-owner base table; process-local P2P metadata pointers do not require translation. NCCL collectives remain available as fallback.

The loader currently supports Qwen3 and GLM-5.1. It requires `--arena <GiB>` and does not support Qwen3.5 or FP8.

## Start the Loader

Start the loader and its initial executor in one command:

```bash
NCCL_P2P_LEVEL=SYS NCCL_TOPO_FILE=/root/chat/vllm/topo_fixed.xml npx tsx src/run_model_loader.ts \
  --arena 92 --gpus 0,1,2,3,4,5,6,7 --cp --glm51 --mtp \
  src/openai-server.ts --host 0.0.0.0 --port 8000 --max-pages 2048 --phased-prefill
```

Arguments before the executor path are shared model arguments and are passed to every executor process. Arguments after the path apply only to that executor. Loader options default to `--control-host 127.0.0.1 --control-port 8099` and must appear before the executor path.

The model is ready when the control endpoint responds and the executor reports its own service as ready:

```bash
curl http://127.0.0.1:8099/status
curl http://127.0.0.1:8000/health
```

## Control the Executor

```bash
# Inspect the current command and executor state.
curl http://127.0.0.1:8099/status

# Restart the configured executor without reloading the model.
curl -X POST http://127.0.0.1:8099/restart

# Stop only the executor. The model remains resident on the GPUs.
curl -X POST http://127.0.0.1:8099/stop

# Start the last configured executor again.
curl -X POST http://127.0.0.1:8099/fork
```

To replace the executor or its process-specific arguments, stop the current executor and provide a JSON command array:

```bash
curl -X POST http://127.0.0.1:8099/stop
curl -X POST http://127.0.0.1:8099/fork \
  -H 'content-type: application/json' \
  -d '["src/openai-server.ts", "--host", "0.0.0.0", "--port", "8000"]'
```

Add `?follow` to `/fork`, `/spawn`, or `/restart` to stream executor output through process exit and output closure:

```bash
curl -N -X POST 'http://127.0.0.1:8099/restart?follow'
```

Changing model/shared arguments requires restarting the loader itself. The control server has no authentication, so keep it bound to `127.0.0.1` unless it is protected by other means.

Executor environment overrides can be supplied to `/fork` or `/spawn` with an object instead of a command array:

```bash
curl -X POST http://127.0.0.1:8099/fork \
  -H 'content-type: application/json' \
  -d '{"command":["src/openai-server.ts","--port","8000"],"env":{"GLM_GRAPH_DIAGNOSTICS":"0"}}'
```

To keep the configured command but replace its environment overrides, use `/restart`:

```bash
curl -X POST http://127.0.0.1:8099/restart \
  -H 'content-type: application/json' \
  -d '{"env":{"GLM_GRAPH_DIAGNOSTICS":"0"}}'
```

Values are strings; `null` unsets an inherited variable. An `env` object replaces the complete override map, and `{}` restores inheritance. Empty-body restarts retain the configured overrides. Overrides affect only the executor, appear in `/status`, and cannot replace loader-managed CUDA IPC/layout variables. A new command array starts with no overrides.

### Spawn arbitrary executables and profile resident weights

`/fork` (formerly `/run`, which is no longer available) launches a JS/TS entry point with `tsx/cjs` and prepends shared model arguments. `/spawn` launches an exact executable/argument array without a shell or argument injection. Both inherit the loader's CUDA IPC/layout environment. `/status` includes `mode`, and `/restart` preserves the configured mode. Empty-body `/fork` or `/spawn` requires a previously configured command of that same mode.

For `/spawn`, obtain shared model arguments from `GET /model-args` and place them at the appropriate position in the command. The loader cannot validate model flags inside arbitrary commands; executor runtime layout validation still applies. For example, with the GLM loader configuration above:

```bash
curl -X POST http://127.0.0.1:8099/stop
curl -N -X POST 'http://127.0.0.1:8099/spawn?follow' \
  -H 'content-type: application/json' \
  -d '["nsys","profile","--trace=cuda,nvtx","--cuda-graph-trace=node","--sample=none","--cpuctxsw=none","--output=/tmp/glm-mtp","node","--require","tsx/cjs","src/run_glm51_multiple_mtp.ts","--arena","92","--gpus","0,1,2,3,4,5,6,7","--cp","--glm51","--mtp","--batch-size","1","--max-new-tokens","128","--max-pages","512"]'
```

Let the benchmark exit naturally to finish the Nsight report. `/stop` sends SIGTERM to a spawned command's process group and allows 30 seconds for shutdown/output draining before SIGKILL. Forked executors retain their Node IPC shutdown request and five-second fallback. Stop the current executor before launching another; the loader and resident weights remain alive throughout.

## NCCL Topology

The following environment variables hsould be used to override the default topology which prevents host staged all gather and all reduce when using NCCL.

```
NCCL_P2P_LEVEL=SYS
NCCL_TOPO_FILE=/root/chat/vllm/topo_fixed.xml
```

NCCL erroneously (the 8 GPUs are connected via 2 switches) sees device interconnect as NODE and the environment variables use optimal p2p routing.

## Vendor

`vendor/` contains external and forked reference implementations, including FlashInfer, llama.cpp, SGLang, vLLM variants, Transformers variants, and b12x. Use them for comparison when implementing CUDA kernels or inference algorithms, but treat `src/` and `csrc/` as authoritative for this project. The FlashInfer fork contains the project's context-parallel changes.

# Workflow

You must NEVER "git commit" unless the user explicitly asks you to commit. If you think the user intends to commit, you must ask for permission per commit. There may be multiple changes that may need to be in multiple commits. You must not proactively commit a change just because you made a prior commit. You must wait for explicit permission to commit every change to ensure the user has fully reviewed it. Instructions to make a change is not implicit permission to commit. Permission to commit is only for a single commit.

You must NEVER use "git stash pop" to reapply stashed changes. You MUST use "git stash apply" instead. "git stash pop" is potentially destructive and may cause data loss when used in conjunction with an coding agent harness rollback. The user will clean up any entries left behind from usage of "git stash apply".

# Production GLM-5.1 Model Config (zai-org/GLM-5.1)

Model Path:
`/mnt/storage/.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-NVFP4`

| Key | Value |
|---|---|
| architectures | GlmMoeDsaForCausalLM |
| model_type | glm_moe_dsa |
| dtype | bfloat16 |
| hidden_size | 6144 |
| intermediate_size | 12288 |
| moe_intermediate_size | 2048 |
| num_hidden_layers | 78 |
| num_attention_heads | 64 |
| num_key_value_heads | 64 |
| kv_lora_rank | 512 (attn output dim, input to v_expand) |
| q_lora_rank | 2048 |
| qk_nope_head_dim | 192 |
| qk_rope_head_dim | 64 |
| qk_head_dim | 256 (= qk_nope + qk_rope) |
| v_head_dim | 256 (output of v_expand; note: ≠ qk_nope_head_dim) |
| head_dim | 192 (= qk_nope_head_dim) |
| n_routed_experts | 256 |
| n_shared_experts | 1 |
| num_experts_per_tok | 8 |
| routed_scaling_factor | 2.5 |
| scoring_func | sigmoid |
| topk_method | noaux_tc |
| index_topk | 2048 |
| index_head_dim | 128 |
| index_n_heads | 32 |
| first_k_dense_replace | 3 |
| num_nextn_predict_layers | 1 |
| max_position_embeddings | 202752 |
| vocab_size | 154880 |
| rms_norm_eps | 1e-05 |
| rope_interleave | true |
