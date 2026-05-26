# GLM.js Architecture Notes

## Overview

TypeScript inference engine for the GLM-5.1 model on NVIDIA GPUs. Ships a native C++/CUDA addon (`glm.node`) wrapped by TypeScript operator classes. Supports multi-GPU via tensor parallelism and context parallelism, CUDA graph capture for decode, and paged KV caching with prefix sharing. Qwen3 and Qwen3.5 are also implemented but serve primarily as broader correctness test targets, not production targets.

## Core Abstractions

### Tensor (`src/tensor.ts`)
- Abstract base class. Concrete implementations: `GlmTensor` (single GPU) and `ParallelTensor` (multi-GPU).
- Each tensor owns a GPU memory pointer (`data`), shape, dtype, and a reference to its `WorkspaceBase`.
- Operations (`.linear()`, `.rmsnorm()`, `.bmm()`, etc.) allocate output tensors from the workspace and call into the native addon.
- `SamplingWorkspace` extends `WorkspaceBase` for GPU-side top-k/top-p sampling with repetition penalty.

### Workspace (`src/workspace.ts`)
- `WorkspaceBase` manages GPU memory lifecycle via **dispose-recycle pooling**:
  - `alloc()` → checks `disposed` set for best-fit reuse; only calls `newTensor()` (real GPU alloc) if no match found.
  - `disposed` set: tensors whose `[Symbol.dispose]()` was called but whose memory hasn't been freed. Next alloc reuses the pointer.
  - Named tensors: re-allocating with the same name disposes the old one, recycling its memory.
  - **`freeze()`**: after model loading, prevents further allocations — all weight addresses are stable for graph capture.
  - **`startTracking()`**: returns a scope that disposes all unnamed temporaries when exited, returning their memory to the pool for the next forward pass.
- This pattern is what makes the inference loop graph-capturable: the same GPU addresses are reused each step deterministically.

### Tensor Lifetime and `using` Pattern

Tensors implement `Disposable` via `[Symbol.dispose]()`. TypeScript 5.2+ explicit resource management (`using`) provides scoped cleanup:

```typescript
// Forward pass: startTracking creates a scope where all unnamed allocs are tracked.
// When the scope exits, every tracked tensor is disposed → its GPU memory returns
// to the workspace's "disposed" pool for reuse next step. Same addresses, deterministic.
using _tracker = ws.startTracking();

// Unnamed temporaries auto-dispose at scope exit via "using"
using gateBuf = normed.linear(weights.gate_proj, BS);  // disposed when scope ends
using upBuf = normed.linear(weights.up_proj, BS);      // disposed when scope ends
using siluBuf = gateBuf.siluAndMul(upBuf, intermediate, BS);

// Named tensors (model weights) throw on dispose — they persist for the model's lifetime
const w = ws.alloc([N, K], "BF16", "model.layers.0.mlp.gate_proj.weight");

// removeTracking() opts out: tensor survives the tracking scope (used for forward outputs)
const output = hiddenStates.linear(lmHead, BS);
return output.removeTracking();  // caller is responsible for disposing this
```

Key rules:
- Named tensors (`setName()` or allocated with a name) cannot be disposed — they live until replaced or the workspace is freed.
- `using` on an unnamed tensor auto-disposes at block scope exit.
- `startTracking()` / its `[Symbol.dispose]()` cleans up all tracked tensors at once — cheaper than individual `using` per tensor in hot loops.
- `removeTracking()` moves a tensor from `tracked` to `exported`, exempting it from the bulk dispose.

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
- **Column**: gate_proj, up_proj, embed_tokens (output dim sharded across GPUs; output is Row-parallel)
- **Row**: o_proj, down_proj, lm_head (input dim sharded; output is PartialSum, needs AllReduce)
- **Replicated**: norms, bias, RoPE freqs, position IDs

Linear op parallelism rules:
- Column × Replicated → Row (no comm)
- Row × Row → PartialSum (AllReduce needed)
- Replicated × Replicated → Replicated (no comm)
- Row × Replicated needs AllGather of input first

## Context Parallelism (Interleaved Token-per-GPU)

When `--cp` flag is set (GLM-5.1 only), all GPUs operate as context-parallel shards instead of tensor-parallel shards:

- **KV cache is Row-sharded**: each GPU stores every Nth token's KV (tokens interleaved across GPUs). `PagedKVCache` uses `TensorParallelism.Row` for ckv/kpe tensors.
- **MLA weights are Replicated** instead of Column: q_pe_proj, v_proj, absorbed weight loaded as Replicated so each GPU can compute attention independently over its shard of the sequence.
- **Position IDs**: `decodeStep`/`mlaDecodeStep` receive `cpWorldSize=N` and `cpRank=i`. The CUDA kernel assigns position `i, i+N, i+2N, ...` to GPU `i`.
- **Prefill**: Each GPU runs MLA prefill over its token subset with `cpWorldSize`/`cpRank` params. The FlashInfer plan computes `effectivePageSize = pageSize / worldSize` and `effectiveNumHeads = numHeads` (not sharded). Output is `PartialSoftmax` — each shard has partial attention output + log-sum-exp.
- **CP Merge**: After prefill/decode, `contextParallelMerge()` or `p2pCpMerge()` combines partial softmax outputs across GPUs using the online softmax trick: `merged_v = Σ(exp(lse_i - lse_max) * v_i) / Σ(exp(lse_i - lse_max))`. This uses either NCCL AllGather + local merge, or fused P2P kernel.
- **Page allocation**: `PagedKVCache` distributes pages round-robin across GPUs. Page `p` is stored on GPU `p % worldSize`. The effective page size per GPU is `pageSize / worldSize`.

## Paged KV Cache (`src/paged_kv.ts`)

- Fixed `PAGE_SIZE=16` tokens per page. Pages are ref-counted for prefix sharing (only full pages are shared; partial last pages are copied).
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

## Testing

```bash
npm run build:all          # build CUDA addon + TypeScript (required before any tests)
npm run test:python        # Python tests: pytest tests/python/ - validates CUDA kernels against PyTorch
npm run test:node          # TypeScript tests: end-to-end model inference via tsx --test
npm test                   # runs both
# individual tests:
cd tests/python && pytest -v test_linear.py   # single Python test
GLM_GPU=1 pytest -v test_linear.py            # specific GPU
npx tsx --test tests/test_glm51.ts            # single TS test
GLM_GPUS=0,1 npx tsx --test tests/test_parallel.ts  # multi-GPU
```

## Key Design Decisions

- **Dispose-recycle workspace**: enables CUDA graph capture without explicit memory pool management. Tensors are allocated once, disposed into a pool, and reused at the same addresses.
- **Pre-allocated buffers**: `ExecutionWorkspace` constructor allocates all plan/work/index buffers. No dynamic GPU allocation during inference.
- **Plan/Run split**: FlashInfer plan is not graph-capturable (writes host buffers, may compile kernels), but run is. Plan only executes when dirty flags indicate changes.
- **P2P AllReduce**: custom kernel for small AllReduce/AllGather avoids NCCL overhead for partial sums in tensor-parallel decode.
- **Multi-stream**: `GlmOps.withStream()` provides alternate CUDA streams with event-based synchronization. Used in GLM-5.1 MoE to overlap shared expert computation with routed expert computation.
- **Strided mmap loading**: weight tensors loaded via `memcpy2dHostToDeviceAsync` with pitch/width/height for row-parallel or column-parallel sharding, avoiding host-side copies.
