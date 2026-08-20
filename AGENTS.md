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

Weight parallelism is assigned per-tensor during loading:
- **Column**: gate_proj, up_proj, embed_tokens, q_nope_proj, k_nope_proj, absorbed weight (output dim sharded across GPUs; linear output is Row-parallel)
- **Row**: o_proj, down_proj, lm_head (input dim sharded; linear output is PartialSum, needs AllReduce)
- **Replicated**: norms, bias, RoPE freqs, position IDs, q_pe_proj, v_proj, ckv_proj, k_pe_proj

Linear op parallelism rules (weight × input → output):
- Column × Replicated → Row (no comm)
- Row × Row → PartialSum (AllReduce needed)
- Replicated × Replicated → Replicated (no comm)
- Row weight → AllGather weight to Replicated, then proceed (K-dim mismatch)
- Row input → AllGather input to Replicated, then proceed (K-dim mismatch)

## Context Parallelism (Interleaved Token-per-GPU)

When `--cp` flag is set (GLM-5.1 only), GPUs operate as context-parallel shards for attention while retaining tensor-parallel sharding for linear layers:

- **KV cache is Row-sharded**: each GPU stores every Nth token's KV (tokens interleaved across GPUs). `PagedKVCache` uses `TensorParallelism.Row` for ckv/kpe tensors.
- **MLA weights split by role**:
  - **Replicated in CP** (were Column in TP-only): q_pe_proj, v_proj — so each GPU can compute attention independently over its KV shard.
  - **Still Column-parallel in CP**: q_nope_proj, k_nope_proj, absorbed weight — the absorbed weight is computed from k_nope_proj × q_nope_proj BMM (Column × Column → Column). This causes Q to be Row-parallel after the absorbed projection, requiring an AllGather to Replicated before attention.
  - **Still Row/Column-parallel in CP**: o_proj (Row → AllReduce), down_proj (Row → AllReduce), gate_proj/up_proj (Column) — same as TP-only mode.
- **Position IDs**: decode/prefill kernels receive `cpWorldSize=N` and `cpRank=i`. The CUDA kernel assigns position `i, i+N, i+2N, ...` to GPU `i`.
- **Prefill**: Each GPU runs MLA prefill over its token subset with `cpWorldSize`/`cpRank` params. The FlashInfer plan computes `effectivePageSize = pageSize / worldSize` and `effectiveNumHeads = numHeads` (not sharded). Output is `PartialSoftmax` — each shard has partial attention output + log-sum-exp.
- **CP Merge**: Partial attention outputs are combined with an online-softmax merge. Small decode workloads use a custom P2P path; larger or prefill workloads use AllGather plus ReduceScatter.
- **Page allocation**: `PagedKVCache` distributes pages round-robin across GPUs. Page `p` is stored on GPU `p % worldSize`. The effective page size per GPU is `pageSize / worldSize`.

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

`src/run_model_loader.ts` loads the model and GPU arena once, then starts executor workers against that resident model. Stopping or restarting a worker does not reload the weights. Stopping the loader process releases the model runtime.

The loader currently supports Qwen3 and GLM-5.1. It requires `--arena <GiB>` and does not support Qwen3.5 or FP8.

## Start the Loader

Start the loader and its initial executor in one command:

```bash
npx tsx src/run_model_loader.ts \
  --arena 92 --gpus 0,1,2,3,4,5,6,7 --cp --glm51 --mtp \
  src/openai-server.ts --host 0.0.0.0 --port 8010
```

Arguments before the executor path are shared model arguments and are passed to every worker. Arguments after the path apply only to that executor. Loader options default to `--control-host 127.0.0.1 --control-port 8099` and must appear before the executor path.

The model is ready when the control endpoint responds and the worker reports its own service as ready:

```bash
curl http://127.0.0.1:8099/status
curl http://127.0.0.1:8010/health
```

## Control the Worker

```bash
# Inspect the current command and worker state.
curl http://127.0.0.1:8099/status

# Restart the configured worker without reloading the model.
curl -X POST http://127.0.0.1:8099/restart

# Stop only the worker. The model remains resident on the GPUs.
curl -X POST http://127.0.0.1:8099/stop

# Start the last configured worker again.
curl -X POST http://127.0.0.1:8099/run
```

To replace the executor or its worker-specific arguments, stop the current worker and provide a JSON command array:

```bash
curl -X POST http://127.0.0.1:8099/stop
curl -X POST http://127.0.0.1:8099/run \
  -H 'content-type: application/json' \
  -d '["src/openai-server.ts", "--host", "0.0.0.0", "--port", "8010"]'
```

Add `?follow` to `/run` or `/restart` to stream worker output until that worker exits:

```bash
curl -N -X POST 'http://127.0.0.1:8099/restart?follow'
```

Changing model/shared arguments requires restarting the loader itself. The control server has no authentication, so keep it bound to `127.0.0.1` unless it is protected by other means.

## Vendor

`vendor/` contains external and forked reference implementations, including FlashInfer, llama.cpp, SGLang, vLLM variants, Transformers variants, and b12x. Use them for comparison when implementing CUDA kernels or inference algorithms, but treat `src/` and `csrc/` as authoritative for this project. The FlashInfer fork contains the project's context-parallel changes.

# Workflow

You must NEVER "git commit" unless the user explicitly asks you to commit. If you think the user intends to commit, you must ask for permission per commit. There may be multiple changes that may need to be in multiple commits. You must not proactively commit a change just because you made a prior commit. You must wait for explicit permission to commit every change to ensure the user has fully reviewed it. Instructions to make a change is not implicit permission to commit. Permission to commit is only for a single commit.

You must NEVER use "git stash pop" to reapply stashed changes. You MUST use "git stash apply" instead. "git stash pop" is potentially destructive and may cause data loss when used in conjunction with an coding agent harness rollback. The user will clean up any entries left behind from usage of "git stash apply".

# Production GLM-5.1 Model Config (zai-org/GLM-5.1)

Model Path:
`/mnt/storage/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/`

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
