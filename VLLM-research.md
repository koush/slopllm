# vLLM Indexer Research

## Scope

This document compares the GLM.js sparse-attention indexer with the indexer
paths in `vendor/vllm-fork`, and records possible follow-up work for production
GLM-5.1 dimensions:

- 32 indexer query heads
- 128-dimensional indexer heads
- top-k 2048
- page size 64
- SM120 RTX PRO 6000 GPUs
- 8-way context parallelism

The B12X kernel implementation itself is external to the vLLM tree. The vendor
tree contains its integration, cache contract, dispatch logic, and tests, but
imports the actual kernels from `b12x.attention.nsa_indexer`.

## Current GLM.js Design

### Decode and Small-Q Scoring

`src/glm_ops.ts` dispatches `totalQ <= GLM_INDEXER_DIRECT_DISPATCH_MAX` (64 by
default) to `indexerScoreTopkV2`. The CUDA scorer in
`csrc/glm_indexer.cu` assigns a warp to each KV position and loops over all
indexer heads. Each head performs a 128-dimensional BF16 dot product with FP32
accumulation, followed by ReLU and the weighted head reduction. The final score
is rounded to BF16 before selection.

The scorer parallelizes long contexts across multiple CTAs, which is important
when decode has only a few query rows. Its main weakness is scalar CUDA-core
work over all 32 heads rather than tensor-core MQA scoring.

### Prefill Scoring

Large-Q prefill uses `idx_prefill_score_mma_kernel`, a BF16 tensor-core kernel.
It:

- Tiles query rows and KV positions.
- Loads each K slab into shared memory once and reuses it across query rows and
  all 32 indexer heads.
- Uses `mma.sync.m16n8k16` with FP32 accumulation.
- Uses `cp.async` for query staging.
- Prunes KV tiles beyond the causal limit.
- Supports flat replicated K and paged K layouts.
- Uses query sharding across GPUs when K has been gathered and replicated.

This is already the stronger part of the implementation and should remain the
baseline for any replacement.

### Top-K

Scores are BF16, so GLM.js exploits the finite 16-bit key space to perform exact
histogram selection.

Decode/small-Q uses:

1. Multi-block score generation into `[totalQ, maxKv]` BF16 scratch.
2. A full 65,536-bucket histogram per row.
3. Threshold selection.
4. Deterministic count and gather passes.

Prefill uses a smaller exact two-level hierarchy:

1. Score into `[totalQ, maxKv]` BF16 scratch.
2. Build a 1,024-bucket coarse histogram.
3. Find the winning coarse bucket.
4. Build a 64-bucket fine histogram within that bucket.
5. Find the exact BF16 threshold.
6. Gather selected positions and scores.

Tie handling is deterministic. Entries above the threshold and the first
threshold ties are emitted in ascending source-position order. This matters
because the downstream sparse-attention path currently behaves differently
when selected indices are not ascending.

### Context Parallelism

Sparse gather is currently disabled independently in `shouldGatherKv`. Full KV
gather remains available for eligible prefill workloads. In that path, the
indexer sees flat replicated K and query rows are partitioned across GPUs, then
the selected rows are all-gathered.

Decode keeps indexer K context-sharded. Each rank scores its local K shard,
local top-k candidates are exchanged, and a global top-k is selected from the
union. This is exact: any global top-k element must be present in the top-k of
the shard that owns it.

The current CP merge gathers candidate values and indices separately and uses a
generic top-k merge. Its rank-major result is sorted by global index before
sparse attention. Disabling that sort has caused long-generation degeneration;
the sort currently costs about 9% of decode throughput.

## Precision and Cache Layouts

### Main Sparse MLA KV

GLM.js already uses the production 656-byte mixed-precision representation:

```text
512 FP8 latent values      = 512 bytes
4 FP32 scales              =  16 bytes
64 BF16 RoPE values        = 128 bytes
Total                      = 656 bytes/token/layer
```

This is represented by `bytesPerToken` in `src/paged_kv.ts`.

### Indexer KV

GLM.js currently stores indexer K as BF16:

```text
128 BF16 values            = 256 bytes/token/layer
```

The vLLM DeepSeek-V3.2/GLM-style indexer and B12X path use:

```text
128 FP8 E4M3 values        = 128 bytes
1 FP32 scale               =   4 bytes
Total                      = 132 bytes/token/layer
```

For the B12X sparse indexer this FP8 layout is mandatory, not an optional cache
mode. B12X rejects the FP4 indexer-cache option. Selecting B12X is optional via
`VLLM_USE_B12X_SPARSE_INDEXER=1` or the `B12X_MLA_SPARSE` attention backend,
but the B12X kernel contract itself requires FP8 indexer K.

Moving only GLM.js indexer K from BF16 to this FP8 representation would reduce
indexer-cache storage and scan traffic by approximately 48.4%. It is not
bit-compatible with current BF16 scoring and must be evaluated as a numerical
change.

Sparse selection does not remove historical KV storage. All past indexer keys
must remain available because the selected set changes for every query. Sparse
attention saves bandwidth primarily by reading the 656-byte main MLA KV only for
selected tokens, while the smaller indexer KV is scanned across the context.

## vLLM Approaches

### DeepGEMM Scoring

The standard CUDA path quantizes indexer Q and K to FP8, folds Q quantization and
attention scales into FP32 per-head weights, and computes FP32 logits with
DeepGEMM MQA kernels. Decode reads the paged FP8 K cache directly; prefill can
gather quantized values and scales into contiguous buffers.

This path still materializes a full FP32 logits matrix before top-k, making its
score scratch twice the size of GLM.js BF16 scratch.

### Persistent Top-K

vLLM's persistent top-k supports k values 512, 1024, and 2048. It dispatches by
row length:

- `length <= topk`: identity output.
- Short rows: a 2,048-bin shared-memory histogram.
- Medium rows: a single-CTA histogram/radix path.
- Long rows: coordinated multi-CTA radix selection.

The long-row path requires occupancy-aware launch sizing and global barrier
state. Its workspace state must be reset before every invocation, including
CUDA graph replay.

This implementation consumes FP32 logits and uses atomic candidate collection,
so it cannot be copied directly without addressing GLM.js ordering and
determinism requirements.

### Cooperative Top-K

vLLM also has a CUDA-cluster top-k path using TMA, distributed shared memory,
and cluster synchronization. vLLM explicitly excludes this path on SM120, so it
is not a practical target for the RTX PRO 6000 deployment.

### B12X

B12X provides an SM120-specific paged FP8 scorer/top-k interface with:

- Fixed page size 64.
- Fixed indexer head dimension 128.
- FP8 K rows plus one FP32 scale.
- FP8 query input and FP32 head weights.
- Caller-owned, graph-stable scratch.
- Optional FP32 top-k scores for context-parallel merging.
- An active-width device value to bound useful decode work under a fixed
  captured launch.
- Streamed prefill supertiles, nominally 32K K rows.

The external kernel source is not available in this repository, so its exact
fusion strategy cannot be copied from `vendor/vllm-fork` directly.

## Ranked Work Items

### 1. Add Representative Decode Benchmarks

Priority: highest prerequisite.

There is a production-shaped prefill benchmark, but no equivalent decode
benchmark. Add CUDA-event benchmarks that separate:

- Score generation.
- Top-k selection.
- CP candidate communication.
- CP candidate merge.
- Index sorting.
- Slot conversion.
- Sparse-attention consumption.

Measure batch sizes 1, 4, 16, 32, and 64 at contexts 2K, 8K, 32K, 64K, 128K,
and 200K. Include MTP widths 2 and 4, mixed sequence lengths, eager execution,
and CUDA graph replay.

The prefill benchmark should continue to model the intended flat replicated-K,
query-sharded path. Add a separate paged/context-sharded benchmark rather than
replacing it.

### 2. Specialize the Existing BF16 Decode Scorer

Priority: high; lowest numerical risk.

Add a production specialization for `(heads=32, headDim=128)`. Candidate
changes include:

- Move invariant K loads outside the per-head loop where register pressure
  permits.
- Use aligned BF16x2 or wider vector loads.
- Fully unroll the dimension and head loops.
- Specialize page-size and flat/paged addressing.
- Test a small tile of K positions per CTA instead of independent warp work.
- Tune blocks per query and warps per block for SM120.

This preserves the existing FP32 accumulation and BF16 score rounding, making
it the safest optimization baseline.

### 3. Prototype a Tensor-Core BF16 Decode Scorer

Priority: high.

Use the 32 indexer heads as the MMA M dimension and a tile of KV positions as N:

```text
Q_heads [32, 128] x K_tile^T [128, N] -> per-head scores [32, N]
```

Then apply ReLU and the 32 head weights to reduce each KV position to one score.
This makes tensor-core decode possible even when there is only one query row and
loads each K tile once. It avoids the poor query-M utilization that would result
from applying the current prefill query tiling directly to decode.

Compare this against the specialized scalar path before changing cache
precision.

### 4. Prototype an SM120 FP8 Paged Decode Scorer

Priority: high potential; medium/high numerical risk.

Implement the B12X-style data contract:

- Quantize indexer K to FP8 E4M3 with one FP32/UE8M0-derived scale per
  128-element row during cache insertion.
- Quantize indexer Q to FP8.
- Fold Q scale, attention scale, and per-head indexer weights into FP32 weights.
- Use SM120 tensor-core instructions for paged MQA scoring.
- Produce globally comparable FP32 scores when CP merging is required.
- Use precompiled production specializations rather than runtime JIT.

Do not target FP4 first. The reference B12X SM120 path itself requires FP8.

### 5. Avoid Full Decode Score Materialization

Priority: high potential after scorer baselines exist.

The current decode pipeline writes a capacity-sized BF16 score row and rereads
it during histogram and gather passes. Evaluate bounded streaming alternatives:

- Score K supertiles and retain a local top-k per tile, then merge candidates.
- Accumulate a coarse BF16 histogram while scoring, determine the threshold,
  then rescore only the winning bucket for exact fine selection.
- Use a persistent scorer/top-k scheduler with fixed graph launch shape and a
  device-side active width.

The local-candidate approach is likely fastest but changes ordering and may be
more difficult to make exactly equivalent. Histogram plus selective rescoring
best preserves BF16 semantics but repeats some scoring work.

### 6. Improve Decode Top-K Without Porting vLLM Blindly

Priority: medium.

Because GLM.js scores are BF16, a two-round radix-256 selector can be much
smaller than vLLM's four-round FP32 radix selector. Possible work:

- Replace the 65,536-entry global histogram with two 256-bin radix passes.
- Use a single CTA for short and medium rows.
- Use coordinated multi-CTA selection only for long rows and very small batch.
- Add an identity fast path when valid row length is at most top-k.
- Keep deterministic tie selection by global position.

Any persistent multi-CTA design must account for occupancy, launch headroom,
workspace initialization, and CUDA graph replay exactly as vLLM does.

### 7. Pack CP Candidate Communication

Priority: medium; relatively contained.

The current CP decode merge all-gathers top-k BF16 scores and I32 indices as
separate tensors. Pack each candidate into one record containing score bits and
global position, perform one collective, then run the merge over the packed
buffer.

Keep logical global token positions through the global merge. Physical cache
slots are rank-specific and should only be derived after global selection.

### 8. Specialize the CP Candidate Merge

Priority: medium.

The merge input width is fixed at `worldSize * topk`, or 16,384 candidates for
8-way CP and top-k 2048. Implement a dedicated stable selector for this shape
rather than invoking a generic tensor top-k and then gathering indices.

The selector should define a total order such as `(score descending, global
position ascending)` and emit indices in the order required by sparse
attention.

### 9. Find and Remove the Downstream Index-Order Dependency

Priority: high correctness issue; performance benefit after resolution.

The current CP result must be sorted by global index. Without sorting, long
generation degenerates even though the selected set is unchanged. Investigate:

- `topkToSlots` compaction.
- `gatherTopkCkv` ordering and deduplication.
- Sparse MLA tile scheduling.
- Candidate-length handling.
- Whether accumulation order alone explains the divergence.
- Whether any physical slots are omitted, duplicated, or left unwritten.

Once sparse attention is genuinely permutation-invariant, the explicit sort
and its approximately 9% decode-throughput cost can be removed. Until then,
every new top-k implementation must preserve ascending global-index order.

### 10. Add Active-Width Scheduling for Captured Decode

Priority: medium.

Decode uses capacity-sized scratch and a graph-stable launch. Introduce a
stable-address device scalar containing the live maximum context width, filled
outside capture. A persistent or tiled kernel can use it to avoid claiming
inactive K tiles while retaining fixed graph topology and buffer addresses.

The existing row lengths already prevent invalid scoring, so the value here is
reduced launch/scheduler overhead and the ability to use bounded scratch, not a
basic correctness fix.

### 11. Stream Prefill Supertiles

Priority: medium/high for very long contexts; high implementation risk.

The current prefill MMA scorer is efficient, but it materializes and scans the
entire BF16 score matrix. Evaluate 16K-32K K supertiles with bounded scratch.

Retain the current kernel for shorter prefills until benchmarks establish a
crossover. Query sharding and flat K gathering already remove substantial
redundant prefill work.

### 12. Use Finer Graph KV Buckets

Priority: low/medium.

Power-of-two padded KV lengths can approach 2x score scratch and launch work near
bucket boundaries. Evaluate finer long-context graph buckets, such as fixed
16K or 32K increments, while preserving stable allocation and capture behavior.

### 13. Avoid Unneeded Top-K Score Outputs

Priority: low.

Top-k scores are required for CP candidate merging but may be unnecessary for
non-CP execution and query-sharded prefill after row gathering. Make score
output optional where doing so does not perturb workspace allocation order or
CUDA graph pointer stability.

## Numerical and Correctness Requirements

### BF16 Compatibility

The existing direct and prefill scorers already use different accumulation
orders, but both round final scores to BF16 before exact selection. New BF16
paths should preserve:

- FP32 dot-product accumulation.
- Per-head ReLU.
- Weighted head reduction.
- Final BF16 rounding before top-k.
- Stable tie selection by token position.

### FP8 Acceptance Criteria

FP8 is a model-level numerical change. Validate more than kernel reconstruction
error:

- Top-k recall against the BF16 path across real score distributions.
- Sparse-attention output error.
- Per-layer hidden-state error.
- Greedy generation agreement length.
- Long-context quality evaluations.
- MTP acceptance rate and accepted-token distribution.
- CP and non-CP agreement.

### Masks and Coordinates

Preserve the current handling of:

- Global causal bounds under interleaved CP.
- Local-to-global position remapping.
- Custom tree masks in global coordinates.
- Query-sharded `qGlobalStart`.
- Flat versus paged K addressing.
- `-1` padding semantics.

## CUDA Graph Requirements

Any replacement must preserve the inference engine's stable-address model:

- Allocate scratch before capture or through deterministic workspace recycling.
- Keep captured grid topology stable unless separate graph variants are used.
- Put changing context metadata in stable-address device buffers.
- Resolve kernel specialization, launch attributes, and plans before capture.
- Reset persistent barrier/counter state on every invocation.
- Warm all decode/MTP row-width policies before capture.
- Do not introduce host synchronization into decode.

## Suggested Implementation Sequence

1. Add decode and component-level CP benchmarks.
2. Implement the specialized scalar BF16 decode scorer.
3. Implement and compare a BF16 tensor-core decode scorer.
4. Pack CP score/index candidates and specialize the 16,384-to-2,048 merge.
5. Diagnose the sparse-attention ordering dependency and remove the sort if
   possible.
6. Prototype FP8 indexer K insertion and an SM120 paged decode scorer behind an
   experimental flag.
7. Run model-level numerical evaluation before making FP8 the default.
8. Prototype bounded score/top-k streaming for decode.
9. Evaluate streamed prefill only after decode work and profiling show it is
   worthwhile.

## Key Source Locations

- `csrc/glm_indexer.cu`: GLM.js scoring, histogram top-k, sorting, and slot
  conversion kernels.
- `src/glm_ops.ts`: single-device dispatch and scratch allocation.
- `src/parallel_ops.ts`: query sharding, CP candidate merge, and index sorting.
- `src/paged_kv.ts`: main and indexer KV cache layouts.
- `tests/python/bench_indexer_prefill.py`: production-shaped prefill benchmark.
- `vendor/vllm-fork/csrc/libtorch_stable/cache_kernels.cu`: FP8 indexer K cache
  insertion and gather.
- `vendor/vllm-fork/csrc/libtorch_stable/persistent_topk.cuh`: persistent top-k
  algorithms.
- `vendor/vllm-fork/csrc/libtorch_stable/cooperative_topk.cuh`: cluster top-k,
  not selected on SM120.
- `vendor/vllm-fork/vllm/model_executor/layers/sparse_attn_indexer.py`: DeepGEMM
  and B12X dispatch, CP merge, and B12X contracts.
- `vendor/vllm-fork/vllm/utils/deep_gemm.py`: paged and contiguous MQA-logits
  interfaces.
