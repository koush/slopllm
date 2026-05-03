# Context Parallelism for MLA (Sequence-Sharded KV)

## Problem

Current tensor parallelism for GLM-5.1 replicates the KV cache on all GPUs. For long contexts, KV memory becomes the bottleneck:
- Full model: `(kvLoraRank + qkRopeDim) × 2 bytes × seqLen × numLayers = 576 × S × 78` per GPU
- At S=128K: **~11 GB/GPU** just for KV cache

Context parallelism shards the KV sequence across GPUs, cutting KV memory by ~1/N.

## Design

### Two Parallelism Modes

| | Head Parallelism (current) | Context Parallelism (new) |
|---|---|---|
| KV cache | Replicated (full copy per GPU) | Sharded by sequence position |
| Q heads | Sharded across GPUs | Replicated (all heads per GPU) |
| MLA attention | Each GPU: partial heads × full KV | Each GPU: all heads × partial KV |
| Post-attention merge | AllReduce o_proj output | Softmax merge + AllReduce v_expand |
| KV memory | Full per GPU | 1/N per GPU |
| Communication | ~32 KB/token/layer (o_proj) | ~176 KB/token/layer (Q gather + v_expand merge + o_proj) |

Both modes can coexist — the user selects based on context length vs throughput tradeoff. Head parallelism is lower latency; context parallelism enables longer contexts.

### Data Flow: Context Parallelism (Decode, 2 GPUs)

```
Per GPU (shard i):
  1. Input layernorm (Replicated — same input on all GPUs)
  2. Q projections (Column parallel → AllGather to get all heads)
     absorbed: [BS, shardNHeads*kvLoraRank] → AllGather → [BS, nHeads*kvLoraRank]
     q_pe_proj: [BS, shardNHeads*qkRopeDim] → AllGather → [BS, nHeads*qkRopeDim]
  3. KV projections (Replicated — ckv_normed, k_pe on all GPUs)
  4. KV cache append: append new token to THIS GPU's KV shard only
     (determine which GPU owns position S using S % worldSize)
  5. RoPE on Q (Replicated — all heads available)
  6. MLA attention: all Q heads × THIS GPU's KV shard
     → partial_attn_out [BS, nHeads, 1, kvLoraRank]  (decode S=1)
     → partial_lse [BS, nHeads]  (NEW — log-sum-exp per head per query, base-2)
  7. v_expand: partial_v_out [BS, nHeads*vHeadDim]  (v_expand BEFORE merge)

Cross-GPU communication:
  8. AllReduce(partial_lse, MAX) → global_lse  (tiny: nHeads × BS × 4 bytes)
  9. Scale: scaled_v_out = exp2(partial_lse - global_lse) * partial_v_out
     scaled_sum = exp2(partial_lse - global_lse)  (per-head scalar)
 10. AllReduce(scaled_v_out + scaled_sum, SUM)
     → combined_v_out, combined_sum on each GPU

Per GPU (shard i):
 11. Normalize: merged_v_out = combined_v_out / combined_sum  [BS, nHeads*vHeadDim]
 12. o_proj (Row parallel → PartialSum → AllReduce)  [BS, hiddenSize]
 13. Residual + post-attention norm (Replicated)
 14. MLP (existing MoE/dense parallelism)
```

**Communication per decode token per layer:**
- AllGather Q: `nHeads × (kvLoraRank + qkRopeDim) × 2 = 128 × 576 × 2 = 144 KB`
- AllReduce MAX lse: `nHeads × 4 = 512 B` (negligible)
- AllReduce SUM v_out: `nHeads × vHeadDim × 2 = 128 × 128 × 2 = 32 KB`
- AllReduce SUM lse sum: `nHeads × 4 = 512 B` (negligible)
- AllReduce o_proj: `hiddenSize × 2 = 32 KB` (same as current)
- **Total: ~208 KB/token/layer** vs current ~32 KB/token/layer

**KV memory savings:** 50% per GPU (2 GPUs), 75% per GPU (4 GPUs)

### Why v_expand BEFORE the AllReduce

v_expand is a per-head linear operation: `v_expand(scale * attn) = scale * v_expand(attn)`. This means we can merge after v_expand:

```
merged = (scale_0 * attn_0 + scale_1 * attn_1) / (scale_0 + scale_1)
v_expand(merged) = (scale_0 * v_expand(attn_0) + scale_1 * v_expand(attn_1)) / (scale_0 + scale_1)
```

This reduces AllReduce volume by **4×** (vHeadDim=128 vs kvLoraRank=512 per head per position).

### Prefill Strategy

Context parallelism applies to both prefill and decode — each GPU builds and maintains only its KV shard from the start.

- Each GPU appends only positions it owns to its KV cache (position `p` is owned by GPU `p % worldSize`)
- Each GPU runs MLA prefill/decode against its KV shard (all Q heads × partial KV)
- Merge partial results after each layer using the softmax merge kernel
- KV memory savings apply from the start, not just during decode

Prefill is compute-bound (attention FLOPS dominate). The extra merge communication (~32 KB/layer) adds <0.004% overhead per layer and is negligible compared to attention compute (~135 ms/layer at S=128K). There is no performance reason to use head parallelism for prefill.

Using context parallelism throughout avoids the complexity of a mode switch (no KV cache redistribution, no dual parallelism modes, no page table compaction after prefill).

## Implementation Plan

### Phase 1: Expose LSE from FlashInfer (no FlashInfer modifications)

FlashInfer already computes `final_lse` internally. We just need to allocate a buffer and pass it through our C wrapper.

**Files to modify:**

1. **`csrc/glm_flash.cu`**
   - `glm_mla_prefill_run`: Add `float* lse` output parameter. Set `params.final_lse = lse` instead of `nullptr`.
   - `glm_mla_decode_run`: Add `float* lse` output parameter. Pass it through to the decode params.

2. **`csrc/glm_ops.cu`** (native addon bindings)
   - Add `lse` pointer parameter to `mlaPrefillRun` and `mlaDecodeRun` N-API functions.

3. **`src/glm_ops.ts`** (`NativeAddon` interface + `GlmOps` class)
   - Add `lse` number parameter to `mlaPrefillRun` and `mlaDecodeRun`.
   - Update method signatures.

4. **`src/device_ops.ts`** (abstract interface)
   - Add `lse: Tensor` parameter to `mlaPrefillRun` and `mlaDecodeRun`.

5. **`src/paged_kv.ts`**
   - `mlaPrefillPaged`: Allocate F32 LSE buffer `[nHeads * totalTokens]`, pass to `mlaPrefillRun`. Return `{attnOut, lse}` instead of just `attnOut`.
   - `mlaDecodePaged`: Allocate F32 LSE buffer `[nHeads * batchSize]`, pass to `mlaDecodeRun`. Return `{attnOut, lse}`.

6. **`src/parallel_ops.ts`**
   - `mlaPrefillRun`: Thread `lse` parameter through, allocate per-shard LSE buffers.
   - `mlaDecodeRun`: Same.

7. **Tests**: `tests/python/test_mla_kernels.py` — Add LSE validation tests.

**LSE output shapes:**
- Prefill: `[totalTokens, numHeads]` (F32, one value per head per query position; FlashInfer writes `final_lse[q * num_heads + r]`)
- Decode: `[batchSize, numHeads]` (F32, one value per head per batch element; FlashInfer writes `lse[batch * num_qo_heads + head]`)

### Phase 2: Custom Softmax Merge + Scale/Divide Kernels

New CUDA kernels for the cross-GPU softmax correction.

**New file: `csrc/glm_context_parallel.cu`**

1. **`mla_scale_lse_kernel`**: Scale v_out by LSE correction factor and compute correction sum
    ```cuda
    // Input: partial_v_out [BS, nHeads * vHeadDim], partial_lse [BS, nHeads], global_lse [BS, nHeads]
    // Output: scaled_v_out [BS, nHeads * vHeadDim], scaled_sum [BS, nHeads]
    // Per element: scaled_v_out[b,h,j] = exp2(partial_lse[b,h] - global_lse[b,h]) * partial_v_out[b,h,j]
    // Per element: scaled_sum[b,h] = exp2(partial_lse[b,h] - global_lse[b,h])
   ```
   This fuses the per-head scale factor broadcast with the element-wise multiply. The `partial_lse[h,b] - global_lse[h,b]` is a per-head-per-position scalar broadcast across `vHeadDim` dimensions.

2. **`mla_softmax_divide_kernel`**: Normalize combined v_out by combined sum
    ```cuda
    // Input: combined_v_out [BS, nHeads * vHeadDim], combined_sum [BS, nHeads]
    // Output: merged_v_out [BS, nHeads * vHeadDim]
    // Per element: result[b,h,j] = combined_v_out[b,h,j] / combined_sum[b,h]
   ```

**New TypeScript APIs:**

3. **`src/glm_ops.ts`**: Add `mlaScaleLse(scaledVOut, scaledSum, partialVOut, partialLse, globalLse, nHeads, vHeadDim, seqLen, batch)` and `mlaSoftmaxDivide(output, combinedVOut, combinedSum, nHeads, vHeadDim, seqLen, batch)`.

4. **`src/parallel_ops.ts`**: Parallel versions of the above.

### Phase 3: AllReduce MAX Support

Currently only `NCCL_SUM` is used. Need `NCCL_MAX` for the LSE AllReduce.

**Files to modify:**

1. **`src/parallel_ops.ts`**
   - Add `allReduceMax(): ParallelTensor` method on `ParallelTensor` (similar to `allReduce()` but using `NCCL_MAX`).
   - Modify `doAllReduce` to accept a reduction op parameter.
   - Support F32 AllReduce MAX (LSE is float32).

2. **`csrc/glm_nccl.cpp`**: No changes needed — `ncclAllReduce` already accepts an `op` parameter. We just need to pass `NCCL_MAX` (value 2) instead of `NCCL_SUM` (value 0) from TypeScript.

### Phase 4: Sequence-Sharded KV Cache

**Page ownership:** Pages are assigned to GPUs round-robin: GPU `r` owns page `p` where `p % worldSize == r`. Each page is a contiguous block of `pageSize` tokens (currently 16). This means:
- GPU 0 (4 GPUs): pages 0, 4, 8, 12, ... → positions 0-15, 64-79, 128-143, 192-207, ...
- Each GPU's page table is compact — just its own pages, no gaps
- Position IDs passed to FlashInfer are the real (non-contiguous) positions, which is fine — FlashInfer uses them for RoPE, not for indexing
- Memory: each GPU stores `ceil(totalPages / worldSize)` pages ≈ 1/N of total KV

During decode, token at position `S` falls in page `S // pageSize`, owned by GPU `(S // pageSize) % worldSize`. Only that GPU appends to its KV cache.

**Files to modify:**

1. **`src/paged_kv.ts`** — `PagedKVCache`
   - Add `sequenceShardIndex: number` and `worldSize: number` properties (0 for single-GPU, >0 for context parallelism).
   - Modify `mlaKvCacheAppend`: Only append to this GPU's shard. If the new position falls in a page owned by another GPU, skip.
   - Modify page allocation: Each GPU allocates `ceil(maxPages / worldSize)` pages (saves memory).
   - Modify `seqKvLens` tracking: Each GPU tracks its shard of positions.

2. **`src/parallel_ops.ts`** — KV cache operations
   - Modify `mlaKvCacheAppend` for context parallelism: each GPU appends only tokens in pages it owns.

### Phase 5: Parallel MLA Attention with Context Parallelism

**Files to modify:**

1. **`src/parallel_ops.ts`** — New `mlaContextParallelDecode()` method
   ```typescript
    mlaContextParallelAttention(
      qAbsorbedR: ParallelTensor,  // AllGathered to Replicated (all heads)
      qPeR: ParallelTensor,        // AllGathered to Replicated (all heads)
      pagedKV: PagedKVCache,       // Sharded KV cache
      layerIdx: number,
      batchSize: number,
      nHeads: number,
      kvLoraRank: number,
      qkRopeDim: number,
      vHeadDim: number,
      vProj: ParallelTensor,       // v_proj weight (Replicated)
      smScale: number,
    ): ParallelTensor {
      // 1. MLA attention (prefill or decode) against this GPU's KV shard
     //    → partial_attn_out, partial_lse
     // 2. v_expand partial_attn_out → partial_v_out
     // 3. AllReduce MAX partial_lse → global_lse
     // 4. Scale: scaled_v_out = exp2(partial_lse - global_lse) * partial_v_out
     //           scaled_sum = exp2(partial_lse - global_lse)
     // 5. AllReduce SUM scaled_v_out + scaled_sum
     // 6. Divide: merged_v_out = combined_v_out / combined_sum
     // 7. Return merged_v_out (Replicated, [BS, nHeads * vHeadDim])
   }
   ```

 2. **`src/glm51_model.ts`** — Modify `mlaLayer`
    - Add context parallelism mode flag
    - In context parallelism mode:
      a. AllGather Q projections (absorbed, q_pe) from Column → Replicated
      b. Call `mlaContextParallelAttention()` instead of the standard attention path
      c. v_expand is done inside `mlaContextParallelAttention()` (before merge)
    - In head parallelism mode: existing flow unchanged
    - Same flow for both prefill and decode — the merge is on the critical path between layers but adds <0.004% overhead per layer

 3. **Q projection AllGather**: The `absorbed` and `q_pe_proj` weights are Column parallel. Their output needs to be AllGathered before MLA attention in context parallelism mode.
    - `absorbed`: `[BS, shardNHeads * kvLoraRank]` → AllGather → `[BS, nHeads * kvLoraRank]`
    - `qPeLin`: `[BS, shardNHeads * qkRopeDim]` → AllGather → `[BS, nHeads * qkRopeDim]`
    - These become Replicated tensors after AllGather.

### Phase 6: Prefill with Context Parallelism

Prefill uses the same context-parallel flow as decode. Each GPU builds only its KV shard and computes partial attention. The merge is identical.

**Causal masking with sequence-sharded KV:** Each GPU computes partial attention against its KV shard. The causal mask is handled by FlashInfer internally — tokens only attend to positions ≤ their own. The softmax merge combines partial results correctly regardless of which GPU holds which positions.

**Prefill-specific changes:**
- Each GPU only appends its shard of tokens to the KV cache during prefill (position `p` owned by GPU `p % worldSize`)
- `mlaPrefillPaged` is called with each GPU's shard of the page table
- Merge flow is the same as decode (LSE output + v_expand + scale + AllReduce + divide)

### Phase 7: Testing & Validation

1. **Single-GPU unit test**: Verify LSE output matches expected values
2. **Two-GPU context parallelism test**: Compare output with head parallelism (should match within BF16 tolerance)
3. **Python reference test**: Compare context-parallel decode output with single-GPU reference
4. **End-to-end generation test**: Verify greedy decoding matches single-GPU output
5. **Memory test**: Verify KV memory usage is halved with 2 GPUs

## Open Questions

1. **Hybrid parallelism**: Should we support head parallelism + context parallelism simultaneously (e.g., 4 GPUs = 2 head shards × 2 sequence shards)?
   - **Recommendation**: Defer to future work. Start with pure context parallelism.

2. **AllGather Q overhead**: For decode, AllGathering the Q projections adds ~144 KB/GPU/layer. Is this acceptable?
   - **Recommendation**: Yes — NVLink bandwidth (300 GB/s) makes this ~0.5 μs, negligible vs attention compute time

3. **P2P for LSE AllReduce**: The LSE is tiny (`nHeads × 4 = 512 B` for full model decode). P2P AllReduce (max 8192 elements) could handle this without NCCL.
   - **Recommendation**: Use P2P for LSE AllReduce MAX if `nHeads × batchSize ≤ 8192`; otherwise fall back to NCCL.

## Estimated Effort

| Phase | Description | Effort |
|---|---|---|
| 1 | Expose LSE from FlashInfer | 1-2 days |
| 2 | Custom softmax merge kernels | 2-3 days |
| 3 | AllReduce MAX support | 0.5 days |
| 4 | Sequence-sharded KV cache | 2-3 days |
| 5 | Parallel MLA with context parallelism (decode) | 3-4 days |
| 6 | Prefill with context parallelism | 2-3 days |
| 7 | Testing & validation | 2-3 days |
| **Total** | | **13-19 days** |

Phases 1-3 are foundational and can be done first. Phases 4-5 are the core implementation. Phases 6-7 build on top.

## Key Reference: FlashInfer LSE Format

FlashInfer's `final_lse` is in **log-base-2**: `lse = log2(sum(exp(sm_scale * logits)))`.

The softmax merge formula uses this directly:
```
m = max(lse_0, lse_1)             // in log2 space
scale_0 = exp2(lse_0 - m)         // correction factor for GPU 0
scale_1 = exp2(lse_1 - m)         // correction factor for GPU 1
merged = (scale_0 * partial_0 + scale_1 * partial_1) / (scale_0 + scale_1)
```

This matches FlashInfer's own internal merge logic (`state_t::merge` in `state.cuh`).
