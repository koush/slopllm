# GLM.js Architecture Notes

## Overview

TypeScript inference engine for GLM models on NVIDIA GPUs. The production default is `local-inference-lab/GLM-5.3-NVFP4`; the implementation and CLI retain the names `Glm51Model` and `--glm51`. Ships a native C++/CUDA addon (`glm.node`) wrapped by TypeScript operator classes. Supports tensor and context parallelism, CUDA graph capture for decode, eager chunked/phased prefill, paged KV caching with prefix sharing, and MTP speculative decoding. Qwen3 and Qwen3.5 are also implemented but serve primarily as broader correctness test targets, not production targets.

## Core Abstractions

### Tensor (`src/tensor.ts`)
- Abstract base class. Concrete implementations: `GlmTensor` (single GPU) and `ParallelTensor` (multi-GPU).
- Tensors carry shape, dtype, and a `WorkspaceBase` reference. `GlmTensor` wraps device or pinned-host memory; views share backing allocations, while `ParallelTensor` owns per-device shards.
- Operations (`.linear()`, `.rmsnorm()`, `.bmm()`, etc.) allocate output tensors from the workspace and call into the native addon.
- `SamplingWorkspace` (`src/sampling.ts`) extends `WorkspaceBase` for GPU-side top-k/top-p sampling, repetition/presence penalties, and speculative sampling support.

### Workspace (`src/workspace.ts`)
- `WorkspaceBase` manages GPU memory lifecycle via **stream-aware heaps**:
  - Unnamed device allocations search `heapByKey` along the active stream lineage, then the synchronized heap. `Heap` splits/coalesces free address ranges; `newTensor()` is called when no reusable range exists. Pinned-host temporaries use a separate best-fit disposed-tensor pool.
  - Unnamed tensors are tracked temporaries. Named tensors are persistent; duplicate named allocations throw, while `ensureAlloc()` returns a compatible existing tensor.
  - Disposed pinned tensors remain in `synchronizingHost` until device synchronization completes, preventing reuse while an asynchronous copy may still reference them.
  - **`freeze()`** prevents further allocations after model loading so weight addresses remain stable.
  - **`startTracking()`** returns a scope that disposes unnamed temporaries on exit. `clearTracking()` provides explicit phase-boundary cleanup and can preserve tensors from a `TensorTree`.
- This pattern is what makes the inference loop graph-capturable: the same GPU addresses are reused each step deterministically.

### Lifetime and `using` Pattern

Tensors implement `Disposable` via `[Symbol.dispose]()`. Model code can use `using` to release intermediates at the end of a lexical scope:

```typescript
using gate = hiddenStates.linear(weights.gate);
using up = hiddenStates.linear(weights.up);
using activated = gate.siluAndMul(up);
return activated.linear(weights.down);
```

Allocation tracking is owned by the caller around a complete model operation. A returned tensor must be removed from that tracking scope:

```typescript
function forward(state: ExecutionState): Tensor {
  // Scope exit disposes remaining tracked tensors. clearTracking/startTracking
  // warn when cleaning up leftovers from the preceding operation.
  using _tracker = state.ws.startTracking();
  // removeTracking allows the tensor to escape workspace cleanup.
  return model.forwardModel(state).removeTracking();
}
```

Key rules:
- Named tensors cannot be disposed and live until the workspace is freed.
- `using` on an unnamed tensor auto-disposes at block scope exit.
- `startTracking()` cleans all tracked tensors at once; use individual `using` declarations when an intermediate should be released earlier.
- `removeTracking()` stages the tensor and its backing view chain. When the tracking scope exits, the tensor returns to `tracked` under caller ownership.
- Disposed device ranges can be reused within an ordered stream lineage. A completion wait promotes stream-local heaps and deferred view references to the waiting stream; the stream handle remains reserved until disposal. Call `streamWaitEvent()` before consuming the result on the current stream.

### Allocator (`src/heap.ts`, `src/glm_ops.ts`)
- `Heap` manages free address ranges with 256-byte alignment, splitting, coalescing, and exact-range `claim()` for captured tensors. Ordinary allocations use the low end; long-term named allocations use the high end.
- `GlmOps.heap` owns the device allocation pool. `--arena <GiB>` supplies one large CUDA allocation per GPU (or imports it through CUDA IPC); weights and runtime allocations share that arena.
- Without an arena, `GlmOps` obtains additional native allocations when its heap cannot satisfy a request and releases those native allocations when the backend is freed. Workspace heaps recycle temporary ranges between operations.

### GlmOps (`src/glm_ops.ts`)
- Single-GPU device backend. Wraps the native addon (`glm.node`).
- Owns CUDA context, device heap, and alternate streams. `withStream(fn)` uses normal priority; `withStream(true, fn)` uses high priority.
- CUDA graph API: `graphBeginCapture/EndCapture/Instantiate/Launch/Destroy`.
- FlashInfer integration: plan/run split for batch prefill/decode and MLA attention.
- NCCL and P2P primitives exposed for `ParallelOps`.

### ParallelOps (`src/parallel_ops.ts`)
- Multi-GPU backend implementing `DeviceOps`. Wraps N `GlmOps` instances.
- `ParallelTensor`: has N shards (one per GPU), delegates ops to each shard, inserts collective communication (AllReduce, AllGather) as needed based on `TensorParallelism` annotations.
- `TensorParallelism` enum: `Replicated`, `Column` (first-dimension sharding for matrices), `Row` (last-dimension sharding for matrices), `PartialSum` (needs reduction), `PartialSoftmax` (for CP merge). Weight matrices use `[output, input]`; activations use `[tokens, features]`.
- Communication: custom P2P collectives with NCCL fallbacks, including AllReduce/AllGather, reduce-scatter, CP merge, and fused RMSNorm+AllReduce. Dispatch depends on tensor size, dtype, layout, and P2P availability. `P2PGroup` isolates staging allocations under group-specific heap keys until barriers make them reusable.
- Per-device shard workspaces created lazily via `getShardWorkspaces()`.

### CUDA Streams and withStream

`withStream(fn)` runs `fn` right away on the host — host execution never becomes
asynchronous. The only difference is that GPU kernels launched inside `fn` are queued to a
separate CUDA stream instead of the current one. The `withStream` wrapper method is an intentional departure from CUDA streams to enforce lexical scoping of streams and their encapsulated resources. Two synchronization points are set up
automatically:

- On entry, the new stream waits on an event recorded on the calling stream, so the new
  stream's first kernel cannot start until everything the calling stream queued *before*
  the `withStream` call has finished.
- When the callback returns, an event is recorded on the new stream and the calling
  stream becomes current again. The handle still reserves the alternate stream.
  `streamWaitEvent()` orders the current stream behind its work and promotes its
  reusable resources; `synchronize()` waits on the host. Handle disposal returns the
  stream and establishes ordering when pending resources need transferring.
  Consume the returned `.result` only after the explicit completion wait.

While the scope is open, the two streams run in parallel. Host code is single-threaded, so
a scope is always created, run, and joined in order — code on the calling stream cannot
queue kernels while a scope is open. Kernels are only ever launched inside a scope's
lexical body; the returned result object can add wait edges later (`streamWaitEvent`) but
never launches new kernels.
Streams are created using withStream and are waited using `streamWaitEvent`.

```typescript
function someOp(normed: Tensor) {
  using tensorA = normed.linear(someWeight);

  // this stream is synchronized with calling stream up to this point.
  using stream1 = ops.withStream(() => {
    return tensorA.linear(otherWeight);
  });

  // this stream is also synchronized with calling stream up to this point.
  // but it runs parallel with the prior stream.
  using stream2 = ops.withStream(() => {
    return tensorA.linear(anotherWeight);
  });

  // this runs after tensorA, but in parallel with tensorC and tensorD. Those streams have not been waited.
  using tensorB = normed.linear(yetAnotherWeight);

  stream1.streamWaitEvent();
  stream2.streamWaitEvent();

  // the return values must become owned/disposed.
  // notably, the return values are immediately available (their destination allocations are known),
  // but they are only *ready* after the streamWaitEvent.
  using tensorC = stream1.result;
  using tensorD = stream2.result;

  // all tensors and streams are disposed
}
```

Streams that outlive their lexical scope should ensure the closure properly captures inputs.

```ts
function someOpThatReturnsAStream(normed: Tensor) {
  using tensorA = normed.linear(someWeight);

  // The caller owns this handle; do not declare it with `using` here.
  return ops.withStream(() => {
    // Retain the input in the alternate stream's deferred view references.
    using tensorAClone = tensorA.viewClone();
    return tensorAClone.linear(anotherWeight);
  });
}

using stream = someOpThatReturnsAStream(normed);
stream.streamWaitEvent();
using tensorB = stream.result;
```


## Tensor Parallelism

Weight parallelism is assigned per-tensor during loading.

Linear op parallelism rules (weight × input → output):
- Column × Replicated → Row (no comm)
- Row × Row → PartialSum (AllReduce needed)
- Replicated × Replicated → Replicated (no comm)
- Replicated × Column → Column (token/batch sharding)
- Replicated × PartialSum → PartialSum; PartialSum × Replicated → PartialSum
- When no direct path applies: reduce a PartialSum input, gather a Row weight/input, or gather a Column input for Column × Column, then retry.
- Selected large replicated attention/indexer projections are narrowed to column-parallel weights for small decode batches.

## Context Parallelism (Interleaved Token-per-GPU)

When `--cp` is set on the GLM backend, GPUs operate as context-parallel shards for attention while retaining tensor-parallel sharding for linear layers:

- **KV cache is Row-sharded**: each GPU stores every Nth token's KV (tokens interleaved across GPUs). `PagedKVCache` uses `TensorParallelism.Row` for ckv/kpe tensors.
- **Position IDs**: planning supplies global token positions. CP-aware cache/attention kernels use rank and world size to map tokens `i, i+N, i+2N, ...` to GPU `i`.
- **Attention**: shard-local attention produces partial output plus log-sum-exp for CP merging. GLM also gathers CKV/indexer caches for selected workloads to run against flat gathered buffers. Production GLM uses sparse MLA; dense MLA uses the FlashInfer plan/run path.
- **CP Merge**: Partial attention outputs are combined with an online-softmax merge using a custom P2P push or tree-reduce path, or AllGather plus ReduceScatter, depending on output size and P2P availability.
- **Page allocation**: every logical page spans all CP GPUs. `pageSize = physicalPageSize * worldSize`, so the default physical page holds 64 tokens per GPU and a logical page holds `64 * worldSize` interleaved tokens.
- **KV and Indexer K Prefetch**: `prefetchLayerResources()` gathers future-layer CKV and indexer K/scales on alternate streams when the selected attention path uses gathered KV. Phased prefill shares gathered buffers between paired chunks.

## Paged KV Cache (`src/paged_kv.ts`)

- Default physical `PAGE_SIZE=64`, defined in `src/paged_sequence.ts`; logical page size grows by world size under CP. GLM requires physical page size 64. Pages are ref-counted for prefix sharing; full pages are shared and partial-page copying is supported.
- `Sequence` (`src/paged_sequence.ts`): ordered pages with allocation length, committed token history, and a pending `targetToken` for the next decode input.
- `PagedKVCache` extends `WorkspaceBase` — KV cache tensors (`kData[]`/`vData[]` or `ckvData[]`/`kpeData[]`) are pre-allocated GPU buffers indexed by layer and page ID.
- Planning metadata belongs to `ExecutionState` and is rebuilt/uploaded per plan; the old page/position dirty flags are gone.
- Dense MLA uses BF16 `ckvData[layer]` `[maxPages, pageSize, kvLoraRank]` and `kpeData[layer]` `[maxPages, pageSize, qkRopeDim]`.
- Sparse MLA packs FP8 CKV, FP32 block scales, and BF16 rotary keys into U8 `ckvData[layer]` `[maxPages, pageSize, bytesPerToken]`. Indexer keys use U8 `kData` with F32 `kScaleData`; shared-indexer layers omit their own indexer cache and reuse prior top-k selections.
- Standard path: uses `kData[layer]` `[maxPages, nKv*pageSize*hd]` and `vData[layer]`.

## Execution Workspace (`src/execution-workspace.ts`)

- `ExecutionWorkspace` extends `WorkspaceBase`. Persistent device/pinned planning buffers are allocated lazily per plan slot using `ensureAlloc()`. Forward temporaries use workspace heaps; warmup establishes reusable allocations before capture.
- `ExecutionState`: holds per-step context (batchSize, totalTokens, seqLens, isDecode, cache reference).
- Each plan gets a distinct slot, allowing plan+plan+run+run without overwriting metadata. Tracking/clear boundaries reset the slot counter. Each slot packs metadata into one device/host buffer pair with stable views; page indices come last so changing page count does not shift other pointers.
- **Plan/Run split**:
  - `planPrefill()` / `planDecode()`: reserve pages, fill position/page/sequence metadata, invoke the appropriate dense FlashInfer or sparse-MLA planning path, and upload metadata. Sparse prefill has no attention-kernel plan but still prepares CP page lengths.
  - `state.setInput()`: uploads host token IDs or copies a GPU token tensor into the slot's input buffer.
  - `forwardPrefill/forwardDecode()`: runs the model forward pass using the planned state.
- FlashInfer plan writes workspace buffers (floatWs, intWs, planInfo). Run reads them. This split is essential for CUDA graph capture — plan runs outside the graph, run is captured.

## CUDA Graph Capture (`src/capture-manager.ts`)

- `CaptureManager` implements `ExecutionManager.execute({ states, inputs, key }, fn)`. For a stable key, the first call runs eagerly; the second records, instantiates, and launches the graph; subsequent calls replay it. An empty key executes eagerly.
- Keys include caller parameters, each state's batch size/total tokens, backend capture keys, and explicit input tensor signatures including memory ranges. Padded KV-length buckets are added when execution calls `getGraphVariantPaddedKvLen()`.
- Captures retain input/output tensors and the set of participating workspaces. Replay validates inputs and clear workspace ownership, launches the graph, reconciles host heap bookkeeping, and returns uncaptured output views.
- Planning and input upload run outside the normal decode graph; model forward, logits, and token selection run inside it. `ExecutionState.captureAll()` supports graphs spanning multiple planned states. `GLM_GRAPH_DIAGNOSTICS=1` enables binding/replay diagnostics.
- `ChatModel.executePrefill()` always uses an empty capture key, so production chunked/phased prefill executes eagerly.

## Model Loading (`src/chat_model.ts`, `src/glm51_model.ts`)

- `ChatModel` extends `WorkspaceBase` — the model IS its weight workspace.
- `fromPretrained(modelDir)`: scans for safetensors files, memory-maps each shard, calls `loadTensor()` for each weight. After loading, calls `freeze()` to lock the workspace.
- `loadTensor()` is model-specific:
  - GLM: splits MLA fused weights (`q_b_proj` → `q_nope_proj` + `q_pe_proj`, `kv_b_proj` → `k_nope_proj` + transposed `v_proj`, `kv_a_proj_with_mqa` → `ckv_proj` + `k_pe_proj`) and handles NVFP4 scale renaming. The query path uses factorized projections rather than a load-time precomputed absorbed weight.
  - Weights are uploaded from memory-mapped safetensors with `mmapLoadAsync`, avoiding a JavaScript-side weight-buffer copy.
  - F32 weights (norms, A_log) are converted to BF16 on load.

## Scratchpad

The `scratchpad/` directory is gitignored and intended for ad-hoc scripts, diagnostics, and experimentation. It is not checked in — use it for anything temporary that shouldn't pollute the repo (e.g., comparing hidden states across layers, debugging KV cache issues, testing new kernels in isolation).

## Testing

```bash
npm run build:all          # build CUDA addon + TypeScript (required before any tests)
npm run test:python        # Python tests: pytest tests/python/ - validates CUDA kernels against PyTorch
npm run test:node          # TypeScript suite entry point: tsx --test tests/test_all.ts
npm test                   # runs both
# individual tests:
pytest -v tests/python/test_linear.py        # single Python test
npx tsx --test tests/test_glm51.ts            # single TS test
npx tsx --test tests/test_parallel.ts  # multi-GPU
```

# Persistent Model Loader

`src/run_model_loader.ts` loads the model and GPU arena once, then starts executor processes against that resident model. Each arena is exported with CUDA IPC and opened at a process-local address by the executor, which replays the model allocation layout against that imported base. Stopping or restarting an executor does not reload the weights. Stopping the loader process releases the model runtime.

Executor CUDA contexts, streams, cuBLAS handles, NCCL communicators, graphs, and workspaces are process-local and are released by process teardown. The loader remains the sole owner of the exported arena allocations. For custom direct-P2P collectives, each executor GPU context opens every owner's arena handle and translates tensor pointers through the resulting per-reader/per-owner base table; process-local P2P metadata pointers do not require translation. NCCL collectives remain available as fallback.

The loader supports Qwen3 and the GLM backend (including the default GLM-5.3 checkpoint via `--glm51`). It requires `--arena <GiB>` and rejects `--qwen35` and `--fp8`.

## Start the Loader

Start the loader and its initial executor in one command:

```bash
NCCL_P2P_LEVEL=SYS NCCL_TOPO_FILE=/root/chat/vllm/topo_fixed.xml npx tsx src/run_model_loader.ts \
  --arena 92 --gpus 0,4,5,7,1,2,3,6 --cp --glm51 --mtp \
  src/openai-server.ts --host 0.0.0.0 --port 8000 --max-pages 8192 --phased-prefill
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

`GET /follow` streams the loader's console output — the most recent 128KB of loader and executor messages, followed by live output — and ends when the current executor's output closes:

```bash
curl -N http://127.0.0.1:8099/follow
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
  -d '["nsys","profile","--trace=cuda,nvtx","--cuda-graph-trace=node","--sample=none","--cpuctxsw=none","--output=/tmp/glm-mtp","node","--require","tsx/cjs","src/run_glm51_multiple_mtp.ts","--arena","92","--gpus","0,4,5,7,1,2,3,6","--cp","--glm51","--mtp","--batch-size","1","--max-new-tokens","128","--max-pages","512"]'
```

Let the benchmark exit naturally to finish the Nsight report. `/stop` sends SIGTERM to a spawned command's process group and allows 30 seconds for shutdown/output draining before SIGKILL. Forked executors retain their Node IPC shutdown request and five-second fallback. Stop the current executor before launching another; the loader and resident weights remain alive throughout.

## NCCL Topology

On the production eight-GPU/two-switch host, use the following topology override to avoid host-staged NCCL AllGather and AllReduce:

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

# Production GLM-5.3 Model Config (local-inference-lab/GLM-5.3-NVFP4)

Model Path:
`$HF_HOME/hub/models--local-inference-lab--GLM-5.3-NVFP4/snapshots/<revision>`

`src/model_cli.ts` selects this repository by default for `--glm51`. The cache's `refs/main` identifies the snapshot revision. Values below are from the checkpoint's `config.json`.

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
| index_topk_freq | 4 |
| index_skip_topk_offset | 3 |
| index_share_for_mtp_iteration | true |
| indexer_types | full at layers 0, 1, 2, then 6, 10, …, 74; shared otherwise |
| first_k_dense_replace | 3 |
| num_nextn_predict_layers | 1 |
| max_position_embeddings | 1048576 |
| vocab_size | 154880 |
| rms_norm_eps | 1e-05 |
| rope_interleave | true |
| indexer_rope_interleave | true |
| rope_parameters | default RoPE, rope_theta = 8000000 |
