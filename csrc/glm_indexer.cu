#include "glm_ops.h"
#include "glm_indexer_cache.cuh"
#include "glm_nvfp4.cuh"
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Remap a local KV position to its global position for context-parallel
// interleaved sharding. cpWorldSize=0 (non-CP) is a no-op identity.
static __device__ __forceinline__ int cp_remap(int pos, int cpWorldSize, int cpRank) {
    return cpWorldSize > 0 ? pos * cpWorldSize + cpRank : pos;
}

// ---------------------------------------------------------------------------
// Fused indexer score kernel
// Computes: out[qi, ki] = sum_h weights[qi,h] * ReLU(sum_d q[qi,h,d] * k[ki,d]) * scale)
// Reads K from paged cache. One block per query token. One warp per head.
// Invalid positions (beyond kvLen or causal limit) remain -inf.
// ---------------------------------------------------------------------------

template <bool FLAT>
__global__ void indexer_score_kernel(
    __nv_bfloat16* __restrict__ out,          // [totalQ, maxKvLen]
    const __nv_bfloat16* __restrict__ q,      // [totalQ, idxNHeads, idxHeadDim]
    const uint8_t* __restrict__ kData,        // [maxPages, pageSize, idxHeadDim]
    const float* __restrict__ kScaleData,     // [maxPages, pageSize]
    const __nv_bfloat16* __restrict__ weights,// [totalQ, idxNHeads]
    const int32_t* __restrict__ pageIndices,  // [numPages]
    const int32_t* __restrict__ pageIndptr,   // [B+1]
    const int32_t* __restrict__ lastPageLen,  // [B]
    const int32_t* __restrict__ qoIndptr,     // [B+1]
    float scale,
    int idxNHeads, int idxHeadDim, int pageSize, int maxKvLen,
    int causal,
    const int32_t* __restrict__ kvTokenIndptr
) {
    const int qIdx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpIdx = tid / 32;
    const int lane = tid % 32;

    // Find sequence for this query (linear scan, B is small)
    int seq = 0;
    while (qoIndptr[seq + 1] <= qIdx) seq++;

    const int qLocalPos = qIdx - qoIndptr[seq];
    const int numQueries = qoIndptr[seq + 1] - qoIndptr[seq];

    const int pageStart = FLAT ? 0 : pageIndptr[seq];
    const int pageEnd = FLAT ? 0 : pageIndptr[seq + 1];
    const int numPages = pageEnd - pageStart;
    const int flatStart = FLAT ? kvTokenIndptr[seq] : 0;
    const int kvLen = FLAT
        ? kvTokenIndptr[seq + 1] - flatStart
        : (numPages > 0 ? (numPages - 1) * pageSize + lastPageLen[seq] : 0);
    const int prefixLen = max(0, kvLen - numQueries);
    const int causalLimit = causal ? (prefixLen + qLocalPos) : (kvLen - 1);

    // Shared memory: q [idxNHeads*idxHeadDim] + weights [idxNHeads] + scores [idxNHeads]
    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* w_s = q_s + idxNHeads * idxHeadDim;
    float* score_s = reinterpret_cast<float*>(w_s + idxNHeads);

    // Load q into shared memory
    for (int i = tid; i < idxNHeads * idxHeadDim; i += blockDim.x) {
        q_s[i] = q[(size_t)qIdx * idxNHeads * idxHeadDim + i];
    }
    // Load weights
    if (tid < idxNHeads) {
        w_s[tid] = weights[qIdx * idxNHeads + tid];
    }
    // Initialize output row to -inf
    for (int i = tid; i < maxKvLen; i += blockDim.x) {
        out[(size_t)qIdx * maxKvLen + i] = __float2bfloat16(-INFINITY);
    }
    __syncthreads();

    if constexpr (FLAT) {
        for (int localK = 0; localK < kvLen && localK <= causalLimit; localK++) {
            if (warpIdx < idxNHeads) {
                const size_t rowId = (size_t)flatStart + localK;
                const uint8_t* k_ptr = kData + rowId * idxHeadDim;
                const __nv_bfloat16* q_ptr = q_s + warpIdx * idxHeadDim;
                const float k_scale = kScaleData[rowId];
                float partial = 0.0f;
                for (int d = lane; d < idxHeadDim; d += 32)
                    partial += __bfloat162float(q_ptr[d]) * fp8_e4m3_to_float(k_ptr[d]);
                for (int offset = 16; offset > 0; offset >>= 1)
                    partial += __shfl_xor_sync(0xffffffff, partial, offset);
                if (lane == 0) {
                    partial *= k_scale * scale;
                    score_s[warpIdx] = fmaxf(partial, 0.0f);
                }
            }
            __syncthreads();
            if (tid == 0) {
                float indexScore = 0.0f;
                for (int h = 0; h < idxNHeads; h++)
                    indexScore += __bfloat162float(w_s[h]) * score_s[h];
                out[(size_t)qIdx * maxKvLen + localK] = __float2bfloat16(indexScore);
            }
            __syncthreads();
        }
    } else {
        // Iterate over pages
        for (int pageIdx = pageStart; pageIdx < pageEnd; pageIdx++) {
            const int localPageIdx = pageIdx - pageStart;
            const int firstLocalK = localPageIdx * pageSize;
            if (firstLocalK > causalLimit) break;

            const int32_t pageId = pageIndices[pageIdx];
            const bool isLastPage = (pageIdx == pageEnd - 1);
            const int tokensInPage = isLastPage ? lastPageLen[seq] : pageSize;

            for (int t = 0; t < tokensInPage; t++) {
                const int localK = firstLocalK + t;
                if (localK > causalLimit) break;

                // Each warp computes one head's dot product
                if (warpIdx < idxNHeads) {
                    const size_t rowId = (size_t)pageId * pageSize + t;
                    const uint8_t* k_ptr = kData + rowId * idxHeadDim;
                    const __nv_bfloat16* q_ptr = q_s + warpIdx * idxHeadDim;
                    const float k_scale = kScaleData[rowId];

                    float partial = 0.0f;
                    for (int d = lane; d < idxHeadDim; d += 32) {
                        partial += __bfloat162float(q_ptr[d]) * fp8_e4m3_to_float(k_ptr[d]);
                    }
                    // Warp reduce
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        partial += __shfl_xor_sync(0xffffffff, partial, offset);
                    }
                    if (lane == 0) {
                        partial *= k_scale * scale;
                        partial = fmaxf(partial, 0.0f);
                        score_s[warpIdx] = partial;
                    }
                }

                __syncthreads();

                // Thread 0 computes weighted sum and writes output
                if (tid == 0) {
                    float indexScore = 0.0f;
                    for (int h = 0; h < idxNHeads; h++) {
                        indexScore += __bfloat162float(w_s[h]) * score_s[h];
                    }
                    out[(size_t)qIdx * maxKvLen + localK] = __float2bfloat16(indexScore);
                }

                __syncthreads();
            }
        }
    }
}

void glm_indexer_score(GlmCtx* ctx, void* out, const void* q, const void* kData,
                       const float* kScaleData, const void* weights, const int32_t* pageIndices,
                       const int32_t* pageIndptr, const int32_t* lastPageLen,
                       const int32_t* qoIndptr, float scale,
                       int totalQ, int idxNHeads, int idxHeadDim,
                       int pageSize, int maxKvLen, int causal,
                       const int32_t* kvTokenIndptr) {
    cudaSetDevice(ctx->device_id);
    int block_size = idxNHeads * 32;
    if (block_size > 1024) block_size = 1024;
    int smem_size = idxNHeads * idxHeadDim * sizeof(__nv_bfloat16)  // q_s
                  + idxNHeads * sizeof(__nv_bfloat16)                // w_s
                  + idxNHeads * sizeof(float);                       // score_s
    if (kvTokenIndptr) {
        indexer_score_kernel<true><<<totalQ, block_size, smem_size, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)q, (const uint8_t*)kData,
            kScaleData,
            (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr,
            scale, idxNHeads, idxHeadDim, pageSize, maxKvLen, causal, kvTokenIndptr);
    } else {
        indexer_score_kernel<false><<<totalQ, block_size, smem_size, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)q, (const uint8_t*)kData,
            kScaleData,
            (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr,
            scale, idxNHeads, idxHeadDim, pageSize, maxKvLen, causal, nullptr);
    }
}

template <int HEAD_DIM>
__global__ void indexer_kv_cache_append_flat_kernel(
    uint8_t* __restrict__ kData,
    float* __restrict__ kScaleData,
    const __nv_bfloat16* __restrict__ appendK,
    const int32_t* __restrict__ kvTokenIndptr,
    const int32_t* __restrict__ batchIndices,
    const int32_t* __restrict__ positions,
    uint32_t nnz, size_t appendStrideN) {
    for (uint32_t i = blockIdx.x; i < nnz; i += gridDim.x) {
        const size_t dstRow = (size_t)kvTokenIndptr[batchIndices[i]] + positions[i];
        uint8_t* dst = kData + dstRow * HEAD_DIM;
        const __nv_bfloat16* src = appendK + (size_t)i * appendStrideN;
        __shared__ float scratch[4];
        pack_indexer_k_row<HEAD_DIM>(dst, kScaleData + dstRow, src, scratch);
    }
}

void glm_indexer_kv_cache_append_flat(GlmCtx* ctx, void* kData, float* kScaleData,
    const void* appendK, const int32_t* kvTokenIndptr,
    const int32_t* batchIndices, const int32_t* positions,
    uint32_t nnz, uint32_t headDim, size_t appendStrideN) {
    cudaSetDevice(ctx->device_id);
    const int blocks = min((uint32_t)65535, max((uint32_t)1, nnz));
    if (headDim == 128) {
        indexer_kv_cache_append_flat_kernel<128><<<blocks, 64, 0, GLM_STREAM(ctx)>>>(
            (uint8_t*)kData, kScaleData, (const __nv_bfloat16*)appendK,
            kvTokenIndptr, batchIndices, positions, nnz, appendStrideN);
    } else if (headDim == 64) {
        indexer_kv_cache_append_flat_kernel<64><<<blocks, 32, 0, GLM_STREAM(ctx)>>>(
            (uint8_t*)kData, kScaleData, (const __nv_bfloat16*)appendK,
            kvTokenIndptr, batchIndices, positions, nnz, appendStrideN);
    } else {
        fprintf(stderr, "glm_indexer_kv_cache_append_flat: unsupported headDim=%u\n", headDim);
    }
}

// ---------------------------------------------------------------------------
// Topk-to-slots kernel: convert token positions to physical KV cache slot IDs
//
// topk_idx: [num_tokens, topk] — token positions within sequence (0..kvLen-1)
// page_indices: flat page IDs (from PagedKVCache.indices)
// page_indptr: [B+1] — per-sequence page range start/end
// last_page_len: [B] — tokens in last partial page per sequence
// batch_indices: [num_tokens] — which sequence each query belongs to
// Output: slots [num_tokens, topk] — physical slot = page_id * eff_page_size + offset, or -1
//
// CP mode (cp_world_size > 1): tokens interleaved across GPUs. GPU cp_rank
// stores tokens at global positions cp_rank, cp_rank+N, cp_rank+2N, ...
// Filter: token_pos % cp_world_size != cp_rank → -1
// Local pos = (token_pos - cp_rank) / cp_world_size
// Effective page size = page_size / cp_world_size
// ---------------------------------------------------------------------------

#define TTS_THREADS 256
#define TTS_WARPS (TTS_THREADS / 32)

// Order-preserving block-wide stream compaction of one chunk of candidates.
// Every thread in the block must call this with the same `base`; `slot` is the
// calling thread's candidate (negative = dropped). Valid slots are appended to
// out[] in ascending threadIdx.x order via a ballot prefix sum, so the result
// depends only on the input. Returns the new base, uniform across the block.
static __device__ __forceinline__ int tts_compact_append(
    int32_t* __restrict__ out, int slot, int base, int* __restrict__ s_warp
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const unsigned vote = __ballot_sync(0xffffffffu, slot >= 0);
    if (lane == 0) s_warp[warp] = __popc(vote);
    __syncthreads();

    // Every thread scans the 8 warp counts: prefix -> this warp's base, sum ->
    // the block total. Cheaper than a real scan at this width.
    int warp_off = 0, total = 0;
#pragma unroll
    for (int w = 0; w < TTS_WARPS; w++) {
        const int c = s_warp[w];
        if (w < warp) warp_off += c;
        total += c;
    }

    if (slot >= 0)
        out[base + warp_off + __popc(vote & ((1u << lane) - 1))] = slot;

    __syncthreads();  // s_warp is reused by the next chunk
    return base + total;
}

// One block per query token. Valid slots are compacted to a contiguous prefix
// [0, count) and the count is written to topk_length[token]; the sparse
// attention kernel then only walks ceil(count/BI) candidate tiles instead of the
// full TOPK. This matters most under CP, where topk_to_slots keeps only the
// ~topk/world positions that live on this rank (the rest were scattered -1).
//
// The slot is the linear physical index into this rank's KV cache:
//   slot = abs_page * eff_page_size + offset,   offset in [0, eff_page_size)
// This must match exactly how concat_and_cache_ds_mla (glm_flash.cu) writes each
// token: slot = page_id * eff_page_size + (local_pos % eff_page_size). The sparse
// MLA kernel decodes slot / page_block_size and slot % page_block_size, but with
// strideKvBlock = page_block_size * bytes_per_token that collapses back to linear
// addressing (kv_cache + slot * BPT), so the linear slot is what it needs.
// In non-CP mode eff_page_size == page_size, so this reduces to the dense layout.
// (Do NOT use page_size as the stride under CP: the physical pages hold only
//  eff_page_size = page_size / cp_world_size tokens each.)
//
// Compaction preserves input order: valid entries appear in the same relative
// order as in topk_idx, so slots[t, 0..count) correspond to the valid subset of
// topk_idx[t, *] in order. This is what makes the output deterministic — a
// plain atomicAdd bump would compact the same set in whatever order the warps
// happened to retire, which changes the order the sparse MLA kernel accumulates
// its candidate tiles in and therefore perturbs the logits run to run.
__global__ void topk_to_slots_kernel(
    int32_t* __restrict__ slots,          // [num_tokens, topk]
    int32_t* __restrict__ topk_length,    // [num_tokens] (nullable)
    const int32_t* __restrict__ topk_idx, // [num_tokens, topk]
    const int32_t* __restrict__ page_indices,
    const int32_t* __restrict__ page_indptr,
    const int32_t* __restrict__ last_page_len,
    const int32_t* __restrict__ batch_indices,
    int topk, int page_size, int num_tokens,
    uint32_t cp_world_size, uint32_t cp_rank,
    const int32_t* __restrict__ kv_token_indptr  // [batch+1] (used when cp_world_size == 1)
) {
    const int token = blockIdx.x;
    if (token >= num_tokens) return;
    const int seq = batch_indices[token];

    int32_t* out = slots + (size_t)token * topk;
    const int32_t* in = topk_idx + (size_t)token * topk;

    __shared__ int s_warp[TTS_WARPS];
    int count = 0;

    // The chunk loop is block-uniform (threads past topk carry slot = -1) so the
    // ballot and __syncthreads() inside tts_compact_append see the whole block.
    if (cp_world_size == 1) {
        // Flat-index mode: slot = kvTokenIndptr[seq] + token_pos.
        // No CP filtering or page mapping — used after CKV gather when each
        // GPU has the full de-interleaved KV in a flat buffer.
        const int flat_base = kv_token_indptr[seq];

        for (int i0 = 0; i0 < topk; i0 += blockDim.x) {
            const int i = i0 + threadIdx.x;
            int slot = -1;
            if (i < topk) {
                const int token_pos = in[i];
                if (token_pos >= 0) slot = flat_base + token_pos;
            }
            count = tts_compact_append(out, slot, count, s_warp);
        }
    } else {
        // Paged-slot mode: slot = abs_page * eff_page_size + offset
        // cp_world_size == 0 (non-CP) or > 1 (CP paged)
        const int num_pages = page_indptr[seq + 1] - page_indptr[seq];
        const int page_base = page_indptr[seq];
        const int eff_page_size = (cp_world_size > 1) ? (page_size / (int)cp_world_size) : page_size;
        const int local_kv_len = (num_pages - 1) * eff_page_size + last_page_len[seq];

        for (int i0 = 0; i0 < topk; i0 += blockDim.x) {
            const int i = i0 + threadIdx.x;
            int slot = -1;
            if (i < topk) {
                const int token_pos = in[i];
                int local_pos = -1;
                if (token_pos >= 0) {
                    if (cp_world_size > 1) {
                        if ((uint32_t)token_pos % cp_world_size == cp_rank)
                            local_pos = (token_pos - (int)cp_rank) / (int)cp_world_size;
                    } else {
                        local_pos = token_pos;
                    }
                }
                if (local_pos >= 0 && local_pos < local_kv_len) {
                    const int abs_page = page_indices[page_base + local_pos / eff_page_size];
                    slot = abs_page * eff_page_size + (local_pos % eff_page_size);
                }
            }
            count = tts_compact_append(out, slot, count, s_warp);
        }
    }

    for (int i = count + threadIdx.x; i < topk; i += blockDim.x) out[i] = -1;
    if (threadIdx.x == 0 && topk_length) topk_length[token] = count;
}

void glm_topk_to_slots(GlmCtx* ctx, int32_t* slots, int32_t* topk_length, const int32_t* topk_idx,
                       const int32_t* page_indices, const int32_t* page_indptr,
                       const int32_t* last_page_len, const int32_t* batch_indices,
                       int num_tokens, int topk, int page_size,
                       uint32_t cp_world_size, uint32_t cp_rank,
                       const int32_t* kv_token_indptr) {
    cudaSetDevice(ctx->device_id);
    topk_to_slots_kernel<<<num_tokens, TTS_THREADS, 0, GLM_STREAM(ctx)>>>(
        slots, topk_length, topk_idx, page_indices, page_indptr, last_page_len, batch_indices,
        topk, page_size, num_tokens, cp_world_size, cp_rank, kv_token_indptr);
}

// ---------------------------------------------------------------------------
// Exact large-K top-K over bf16 scores, by two-pass radix selection.
//
// Indexer scores are bf16-rounded, so each score is one of only 65536 possible
// bit patterns. Selecting the top-K (K up to a few thousand)
// out of N (up to ~200k) is therefore an exact radix problem: histogram the high
// key byte, then histogram the low byte only in the winning high-byte bucket.
// Short rows use one block with warp-private shared bins. Long rows use the
// existing split grid and a compact 256-bin global histogram per row.
// All passes have a fixed launch shape (grid sized for max context), so this is
// CUDA-graph capturable; per-row length is read on-device from row_len.
//
//   scratch: [batch, 1056] i32          meta: [batch, 4] i32 scratch
//   meta layout per row: [tau_key, tie_take, out_count, tie_count]
// Output out_idx[row, 0..K) = selected position indices, -1 padded.
//
// The selection is a pure function of the input: entries with key > tau land in
// out[0, numAbove) in ascending position order, then the FIRST tie_take entries
// with key == tau land in out[numAbove, K). See idx_gather_count_kernel.
// ---------------------------------------------------------------------------

#define IDX_RADIX_BUCKETS 256

// Per-row scratch is shared by three non-overlapping phases. FP8 decode needs
// 4096 bytes of quantized Q followed by 32 fp32 effective weights (4224 bytes),
// while deterministic gather needs at most 2*256 i32 block counts.
#define IDX_SCRATCH_I32 1056

// meta[0] sentinel meaning "the threshold pass already wrote the whole row"
// (len <= topk). No bf16 key can reach it, so the gather is a no-op either way.
#define IDX_TAU_IDENTITY 0x7FFFFFFF

#define IDX_GATHER_THREADS 256
#define IDX_GATHER_WARPS   (IDX_GATHER_THREADS / 32)

// Block-wide ordered slot assignment for one tile of candidates, fused across
// the two selection classes (key > tau and key == tau, which are mutually
// exclusive). Every thread in the block must call this with the same `base*`;
// `takeA`/`takeT` mark the calling thread's candidate for each class, where a
// thread takes at most one. Selected threads receive consecutive slots in
// ascending threadIdx.x order via a ballot prefix sum, so the assignment depends
// only on the input -- never on the order warps happen to retire. `*total*`
// receives the block-wide count (uniform), which the caller adds to its base
// for the next tile. One barrier pair and one warp-count scan serve both
// classes, halving the per-tile barrier traffic versus two idx_ordered_slot
// calls. Requires blockDim.x == IDX_GATHER_THREADS.
static __device__ __forceinline__ void idx_ordered_slot_pair(
    bool takeA, int baseA, bool takeT, int baseT,
    int* __restrict__ s_warp,   // [2 * IDX_GATHER_WARPS]
    int* __restrict__ slotA, int* __restrict__ slotT,
    int* __restrict__ totalA, int* __restrict__ totalT)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const unsigned voteA = __ballot_sync(0xffffffffu, takeA);
    const unsigned voteT = __ballot_sync(0xffffffffu, takeT);
    if (lane == 0) {
        s_warp[warp] = __popc(voteA);
        s_warp[IDX_GATHER_WARPS + warp] = __popc(voteT);
    }
    __syncthreads();

    // Every thread scans the warp counts once for both classes: prefix -> this
    // warp's base, sum -> the block total. Cheaper than a real scan at this
    // width.
    int warpOffA = 0, warpOffT = 0, sumA = 0, sumT = 0;
#pragma unroll
    for (int w = 0; w < IDX_GATHER_WARPS; w++) {
        const int cA = s_warp[w];
        const int cT = s_warp[IDX_GATHER_WARPS + w];
        if (w < warp) { warpOffA += cA; warpOffT += cT; }
        sumA += cA; sumT += cT;
    }

    *slotA = takeA ? (baseA + warpOffA + __popc(voteA & ((1u << lane) - 1))) : -1;
    *slotT = takeT ? (baseT + warpOffT + __popc(voteT & ((1u << lane) - 1))) : -1;
    __syncthreads();   // s_warp is reused by the next tile
    *totalA = sumA;
    *totalT = sumT;
}

static __device__ __forceinline__ int bf16_key_bits(unsigned short u) {
    // NaNs are invalid candidates, like masked -inf, not larger than finite logits.
    if ((u & 0x7FFF) > 0x7F80) return 127;
    // Order-preserving float->uint key: negatives -> [0,0x7FFF] (reversed),
    // non-negatives -> [0x8000,0xFFFF]. Monotonic in value across the full range,
    // so histogram bucket order == score order. Scores are signed (final weighted
    // sum is not ReLU'd), so this must handle the sign bit.
    unsigned short k = (u & 0x8000) ? (unsigned short)(~u) : (unsigned short)(u | 0x8000);
    return (int)k;
}

static __device__ __forceinline__ int bf16_key(const __nv_bfloat16* p) {
    return bf16_key_bits(*reinterpret_cast<const unsigned short*>(p));
}

// One block per row performs both radix scans and initializes the output. A
// private histogram per warp reduces shared-atomic contention; threads then
// reduce the eight copies while locating the winning byte. The len<=topk branch
// returns before either score scan.
__global__ void idx_radix_threshold_kernel(
    int32_t* __restrict__ meta,
    int32_t* __restrict__ out_idx, __nv_bfloat16* __restrict__ out_scores,
    const int32_t* __restrict__ row_len,
    const __nv_bfloat16* __restrict__ scores, int stride, int topk,
    int cpWorldSize, int cpRank
) {
    const int row = blockIdx.x;
    const int len = row_len ? row_len[row] : stride;
    const __nv_bfloat16 neg_inf = __float2bfloat16(-INFINITY);
    // Identity case: all positions fit in topk. Write position-preserving
    // values directly (deterministic), skipping any position whose score is
    // -inf (masked out by a custom tree mask). tau=INT_MAX disables the gather
    // pass entirely (no bf16 key can exceed it).
    if (len <= topk) {
        const __nv_bfloat16* s = scores + (size_t)row * stride;
        for (int i = threadIdx.x; i < topk; i += blockDim.x) {
            bool valid = (i < len && bf16_key(&s[i]) > 127);
            out_idx[(size_t)row * topk + i] = valid ? cp_remap(i, cpWorldSize, cpRank) : -1;
            out_scores[(size_t)row * topk + i] = valid ? s[i] : neg_inf;
        }
        if (threadIdx.x == 0) {
            int32_t* m = meta + (size_t)row * 4;
            m[0] = IDX_TAU_IDENTITY; m[1] = 0; m[2] = topk; m[3] = 0;
        }
        return;
    }

    for (int i = threadIdx.x; i < topk; i += blockDim.x) {
        out_idx[(size_t)row * topk + i] = -1;
        out_scores[(size_t)row * topk + i] = neg_inf;
    }

    __shared__ int sh[IDX_GATHER_WARPS][IDX_RADIX_BUCKETS];
    __shared__ int s_winner;
    __shared__ int s_above;
    int* bins = &sh[0][0];
    const int warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < IDX_GATHER_WARPS * IDX_RADIX_BUCKETS;
         i += blockDim.x)
        bins[i] = 0;
    __syncthreads();

    // Pass 1: high byte of the monotonic bf16 key.
    for (int i = threadIdx.x; i < len; i += blockDim.x)
        atomicAdd(&sh[warp][bf16_key(&scores[(size_t)row * stride + i]) >> 8], 1);
    __syncthreads();

    if (threadIdx.x == 0) {
        int cum = 0;
        for (int bucket = IDX_RADIX_BUCKETS - 1; bucket >= 0; bucket--) {
            int count = 0;
#pragma unroll
            for (int w = 0; w < IDX_GATHER_WARPS; w++) count += sh[w][bucket];
            if (cum + count >= topk) {
                s_winner = bucket;
                s_above = cum;
                break;
            }
            cum += count;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < IDX_GATHER_WARPS * IDX_RADIX_BUCKETS;
         i += blockDim.x)
        bins[i] = 0;
    __syncthreads();

    // Pass 2: low byte among entries in the winning high-byte bucket.
    const int winningHigh = s_winner;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        const int key = bf16_key(&scores[(size_t)row * stride + i]);
        if ((key >> 8) == winningHigh) atomicAdd(&sh[warp][key & 0xff], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int numAbove = s_above;
        int tau = winningHigh << 8;
        for (int bucket = IDX_RADIX_BUCKETS - 1; bucket >= 0; bucket--) {
            int count = 0;
#pragma unroll
            for (int w = 0; w < IDX_GATHER_WARPS; w++) count += sh[w][bucket];
            if (numAbove + count >= topk) {
                tau |= bucket;
                break;
            }
            numAbove += count;
        }
        int32_t* m = meta + (size_t)row * 4;
        m[0] = tau;                   // threshold bucket
        m[1] = topk - (int)numAbove;  // ties to take from the tau bucket
        m[2] = 0;                     // out_count
        m[3] = 0;                     // tie_count
    }
}

// Long rows need more than one CTA to saturate the GPU. Each split first builds
// a shared 256-bin histogram, then contributes its nonzero bins to the compact
// per-row global histogram. LOW=false selects the high key byte; LOW=true only
// counts low bytes in the winning high-byte bucket recorded in meta.
template <bool LOW>
__global__ void idx_radix_hist_split_kernel(
    const __nv_bfloat16* __restrict__ scores, const int32_t* __restrict__ row_len,
    int32_t* __restrict__ scratch, const int32_t* __restrict__ meta,
    int stride, int topk
) {
    const int row = blockIdx.y;
    const int len = row_len ? row_len[row] : stride;
    if (len <= topk) return;

    // Surplus split: this block's first grid-stride position already exceeds
    // the row, so the scan loop below cannot run; skip the shared-memory
    // clear and barriers.
    if ((int)blockIdx.x * blockDim.x >= len) return;

    __shared__ int bins[IDX_RADIX_BUCKETS];
    bins[threadIdx.x] = 0;
    __syncthreads();

    const int winningHigh = LOW ? meta[(size_t)row * 4] : 0;
    const __nv_bfloat16* s = scores + (size_t)row * stride;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len;
         i += gridDim.x * blockDim.x) {
        const int key = bf16_key(&s[i]);
        if (!LOW || (key >> 8) == winningHigh)
            atomicAdd(&bins[LOW ? (key & 0xff) : (key >> 8)], 1);
    }
    __syncthreads();

    const int count = bins[threadIdx.x];
    if (count)
        atomicAdd(scratch + (size_t)row * IDX_SCRATCH_I32 + threadIdx.x, count);
}

// Exclusive descending prefix count for one histogram bucket per thread.
// All 256 threads participate; integer sums preserve exact cutoff/tie semantics.
static __device__ __forceinline__ int idx_radix_count_above(int count) {
    __shared__ int warpTotals[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int inclusive = count;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int preceding = __shfl_up_sync(0xffffffff, inclusive, offset);
        if (lane >= offset) inclusive += preceding;
    }
    if (lane == 31) warpTotals[warp] = inclusive;
    __syncthreads();
    int above = inclusive - count;
#pragma unroll
    for (int w = 0; w < 8; w++) {
        if (w < warp) above += warpTotals[w];
    }
    return above;
}

// Resolve the high-byte bucket, initialize outputs, and clear the compact
// histogram for reuse by the low-byte pass.
__global__ void idx_radix_high_threshold_kernel(
    int32_t* __restrict__ scratch, int32_t* __restrict__ meta,
    int32_t* __restrict__ out_idx, __nv_bfloat16* __restrict__ out_scores,
    const int32_t* __restrict__ row_len,
    const __nv_bfloat16* __restrict__ scores, int stride, int topk,
    int cpWorldSize, int cpRank
) {
    const int row = blockIdx.x;
    const int len = row_len ? row_len[row] : stride;
    const __nv_bfloat16 neg_inf = __float2bfloat16(-INFINITY);
    int32_t* m = meta + (size_t)row * 4;

    if (len <= topk) {
        const __nv_bfloat16* s = scores + (size_t)row * stride;
        for (int i = threadIdx.x; i < topk; i += blockDim.x) {
            const bool valid = i < len && bf16_key(&s[i]) > 127;
            out_idx[(size_t)row * topk + i] =
                valid ? cp_remap(i, cpWorldSize, cpRank) : -1;
            out_scores[(size_t)row * topk + i] = valid ? s[i] : neg_inf;
        }
        if (threadIdx.x == 0) {
            m[0] = IDX_TAU_IDENTITY; m[1] = 0; m[2] = topk; m[3] = 0;
        }
        return;
    }

    for (int i = threadIdx.x; i < topk; i += blockDim.x) {
        out_idx[(size_t)row * topk + i] = -1;
        out_scores[(size_t)row * topk + i] = neg_inf;
    }
    int32_t* hist = scratch + (size_t)row * IDX_SCRATCH_I32;
    const int bucket = IDX_RADIX_BUCKETS - 1 - threadIdx.x;
    const int count = hist[bucket];
    const int above = idx_radix_count_above(count);
    if (above < topk && above + count >= topk) {
        m[0] = bucket;
        m[1] = above;
    }
    if (threadIdx.x == 0) { m[2] = 0; m[3] = 0; }
    __syncthreads();
    hist[threadIdx.x] = 0;
}

__global__ void idx_radix_low_threshold_kernel(
    const int32_t* __restrict__ scratch, int32_t* __restrict__ meta,
    const int32_t* __restrict__ row_len, int stride, int topk
) {
    const int row = blockIdx.x;
    const int len = row_len ? row_len[row] : stride;
    if (len <= topk) return;

    int32_t* m = meta + (size_t)row * 4;
    __shared__ int highState[2];
    if (threadIdx.x == 0) {
        highState[0] = m[0];
        highState[1] = m[1];
    }
    __syncthreads();
    const int high = highState[0];
    const int highAbove = highState[1];
    const int32_t* hist = scratch + (size_t)row * IDX_SCRATCH_I32;
    const int bucket = IDX_RADIX_BUCKETS - 1 - threadIdx.x;
    const int count = hist[bucket];
    const int numAbove = highAbove + idx_radix_count_above(count);
    if (numAbove < topk && numAbove + count >= topk) {
        m[0] = (high << 8) | bucket;
        m[1] = topk - numAbove;
    }
}

// Gather pass A: per-block candidate counts.
//
// Block bx owns the CONTIGUOUS chunk [lo, hi) of the row rather than a strided
// slice, so an exclusive prefix sum over the blocks' counts gives each block the
// output offset it needs to emit its candidates in ascending position order.
// That is what makes the selection reproducible: a plain atomicAdd cursor
// compacts the same set in whatever order the warps happen to retire, which
// permutes topk_idx, which permutes the slots topk_to_slots emits, which changes
// the order sparse MLA accumulates its candidate tiles in -- so the logits move
// run to run. Under context parallelism it is worse than cosmetic: every rank
// runs this selection independently over identical data, and once tau lands on a
// real score bucket the ranks disagree about WHICH tied entries to keep. The
// per-rank KV fan-out in gatherTopkCkv assumes they agree, so a disagreement
// leaves a flat slot that no rank ever wrote.
//
// Integer counts are order-independent, so the shared-memory atomics here are
// safe. `counts` is [batch, IDX_SCRATCH_I32] using 2 ints per block;
// num_splits is <= 256 at every call site.
__global__ void idx_gather_count_kernel(
    const __nv_bfloat16* __restrict__ scores, const int32_t* __restrict__ row_len,
    const int32_t* __restrict__ meta, int32_t* __restrict__ counts, int stride
) {
    const int row = blockIdx.y;
    const int tau = meta[(size_t)row * 4];
    int32_t* c = counts + (size_t)row * IDX_SCRATCH_I32 + (size_t)blockIdx.x * 2;
    if (tau == IDX_TAU_IDENTITY) {           // block-uniform
        if (threadIdx.x == 0) { c[0] = 0; c[1] = 0; }
        return;
    }

    const int len = row_len ? row_len[row] : stride;
    const __nv_bfloat16* s = scores + (size_t)row * stride;
    const int chunk = (len + (int)gridDim.x - 1) / (int)gridDim.x;
    const int lo = min((int)blockIdx.x * chunk, len);
    const int hi = min(lo + chunk, len);

    // Surplus split: contiguous window empty. Pass B's exclusive prefix sums
    // every preceding block's counts slot unconditionally, so the zero counts
    // must still be written; the scan, atomics, and barriers can be skipped.
    if (lo >= hi) {
        if (threadIdx.x == 0) { c[0] = 0; c[1] = 0; }
        return;
    }

    int nAbove = 0, nTie = 0;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const int key = bf16_key(&s[i]);
        nAbove += (key > tau);
        nTie   += (key == tau);
    }

    __shared__ int sAbove, sTie;
    if (threadIdx.x == 0) { sAbove = 0; sTie = 0; }
    __syncthreads();
    if (nAbove) atomicAdd(&sAbove, nAbove);
    if (nTie)   atomicAdd(&sTie, nTie);
    __syncthreads();
    if (threadIdx.x == 0) { c[0] = sAbove; c[1] = sTie; }
}

// Gather pass B: emit candidates at deterministic offsets.
//
// Layout is exact, never sparse: the threshold pass derived numAbove from a
// histogram of this same buffer with this same row length, so out[0, numAbove)
// receives exactly the key > tau entries and out[numAbove, topk) exactly the
// first tie_take entries at key == tau. (The -1/-inf prefill the threshold pass
// wrote covers the row regardless.)
__global__ void idx_gather_write_kernel(
    const __nv_bfloat16* __restrict__ scores, const int32_t* __restrict__ row_len,
    const int32_t* __restrict__ meta, const int32_t* __restrict__ counts,
    int32_t* __restrict__ out_idx, __nv_bfloat16* __restrict__ out_scores,
    int stride, int topk, int cpWorldSize, int cpRank
) {
    const int row = blockIdx.y;
    const int32_t* m = meta + (size_t)row * 4;
    const int tau = m[0];
    if (tau == IDX_TAU_IDENTITY) return;     // block-uniform; row already written
    const int tieTake = m[1];
    const int aboveTotal = topk - tieTake;

    const int len = row_len ? row_len[row] : stride;
    const __nv_bfloat16* s = scores + (size_t)row * stride;
    int32_t* out = out_idx + (size_t)row * topk;
    __nv_bfloat16* out_s = out_scores + (size_t)row * topk;

    const int chunk = (len + (int)gridDim.x - 1) / (int)gridDim.x;
    const int lo = min((int)blockIdx.x * chunk, len);
    const int hi = min(lo + chunk, len);

    // Surplus split: contiguous window empty and this pass emits nothing, so
    // skip the prefix scan over the preceding blocks' counts.
    if (lo >= hi) return;

    // Exclusive prefix over the preceding blocks' counts -> this block's bases.
    const int32_t* c = counts + (size_t)row * IDX_SCRATCH_I32;
    __shared__ int sAboveBase, sTieBase;
    if (threadIdx.x == 0) {
        int a = 0, t = 0;
        for (int b = 0; b < (int)blockIdx.x; b++) { a += c[2 * b]; t += c[2 * b + 1]; }
        sAboveBase = a; sTieBase = t;
    }
    __syncthreads();
    int aboveOff = sAboveBase, tieOff = sTieBase;

    __shared__ int s_warp[2 * IDX_GATHER_WARPS];
    // The tile loop is block-uniform (threads past hi carry take = false) so the
    // ballots and __syncthreads() inside idx_ordered_slot_pair see the whole
    // block.
    for (int base = lo; base < hi; base += blockDim.x) {
        const int i = base + threadIdx.x;
        const bool inRange = i < hi;
        const int key = inRange ? bf16_key(&s[i]) : -1;   // -1 matches no tau
        const bool isAbove = inRange && key > tau;
        const bool isTie = inRange && key == tau;
        int aSlot, tRank, aTotal, tTotal;

        idx_ordered_slot_pair(isAbove, aboveOff, isTie, tieOff,
                              s_warp, &aSlot, &tRank, &aTotal, &tTotal);
        if (isAbove) {
            if (aSlot < aboveTotal) {
                out[aSlot] = cp_remap(i, cpWorldSize, cpRank);
                out_s[aSlot] = s[i];
            }
        } else if (isTie) {
            if (tRank < tieTake) {
                out[aboveTotal + tRank] = cp_remap(i, cpWorldSize, cpRank);
                out_s[aboveTotal + tRank] = s[i];
            }
        }
        aboveOff += aTotal;
        tieOff += tTotal;
    }
}

// ---------------------------------------------------------------------------
// Sort one top-k row ascending by index, -1 padding last.
//
// The CP merge concatenates the per-shard top-k lists rank-major, so its output
// is ordered by (P mod W, P div W) rather than by global position P. Whenever
// the merge and the replicated path select the same set -- which is every
// context up to topk, where the selection keeps everything -- the ONLY
// difference is that order. It is not cosmetic: topk_to_slots compacts
// preserving input order, and sparse MLA accumulates its candidate tiles in slot
// order, so the same tokens summed in a different order move the logits in the
// low bits and greedy decoding diverges within a few tokens.
//
// One block per row, bitonic sort in shared memory. The row is padded up to a
// power of two with INT_MAX (which also parks the -1 padding at the end), so a
// non-power-of-two topk works too.
// ---------------------------------------------------------------------------
#define IDX_SORT_SENTINEL 0x7FFFFFFF

__global__ void idx_sort_by_index_kernel(
    int32_t* __restrict__ out_idx, __nv_bfloat16* __restrict__ out_scores,
    int topk, int n2)
{
    extern __shared__ char smem_sort[];
    int32_t* sk = reinterpret_cast<int32_t*>(smem_sort);
    __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(sk + n2);

    const int row = blockIdx.x;
    int32_t* gi = out_idx + (size_t)row * topk;
    __nv_bfloat16* gv = out_scores + (size_t)row * topk;
    const __nv_bfloat16 neg_inf = __float2bfloat16(-INFINITY);

    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
        const bool inRow = i < topk;
        const int32_t v = inRow ? gi[i] : -1;
        sk[i] = (v < 0) ? IDX_SORT_SENTINEL : v;
        sv[i] = inRow ? gv[i] : neg_inf;
    }
    __syncthreads();

    // Each pair is touched by exactly one thread (the one holding the lower
    // index), so the strided loop needs no extra guarding inside a pass.
    for (int k = 2; k <= n2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n2; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool asc = ((i & k) == 0);
                    if ((sk[i] > sk[ixj]) == asc) {
                        int32_t tk = sk[i]; sk[i] = sk[ixj]; sk[ixj] = tk;
                        __nv_bfloat16 tv = sv[i]; sv[i] = sv[ixj]; sv[ixj] = tv;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < topk; i += blockDim.x) {
        gi[i] = (sk[i] == IDX_SORT_SENTINEL) ? -1 : sk[i];
        gv[i] = sv[i];
    }
}

void glm_sort_topk_by_index(GlmCtx* ctx, int32_t* out_idx, __nv_bfloat16* out_scores,
                            int batch, int topk) {
    cudaSetDevice(ctx->device_id);
    int n2 = 1;
    while (n2 < topk) n2 <<= 1;
    size_t smem = (size_t)n2 * (sizeof(int32_t) + sizeof(__nv_bfloat16));
    idx_sort_by_index_kernel<<<batch, 256, smem, GLM_STREAM(ctx)>>>(
        out_idx, out_scores, topk, n2);
}

// ---------------------------------------------------------------------------
// Automatic split budget (num_splits == 0 sentinel). Sizes the launch to one
// resident wave of split blocks spread across all query rows: SM count x
// per-SM occupancy, divided by the row count, clamped to the 256 split cap
// shared by every call site. Every input is a shape or config constant — no KV length — so
// CUDA-graph callers can pass the sentinel and keep a stable launch geometry
// across replays. Surplus splits are correct: the score kernels grid-stride
// over device-side numValid and exit block-uniformly before their Q smem
// load, and the merge kernels clip their scan windows to device-side
// row_len, so blocks beyond the useful range find no work early and exit.
//
// Occupancy is specific to the compiled kernel: template instantiations can
// differ in register usage even at identical block size and shared memory.
// Callers must therefore query idx_query_occupancy on the exact
// specialization(s) the launch will select.
// ---------------------------------------------------------------------------

// Per-SM resident-block count for one compiled kernel instantiation.
// Returns 1 on any query failure so callers fall back to a conservative
// single-block budget.
template <typename Kernel>
static int idx_query_occupancy(Kernel kernel, int blockThreads, size_t dynamicSmem) {
    int resident = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, kernel, blockThreads, dynamicSmem) != cudaSuccess
        || resident < 1) {
        return 1;
    }
    return resident;
}

static int idx_auto_num_splits(int resident, int rows) {
    if (rows < 1) rows = 1;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 1;
    }
    int smCount = 0;
    if (cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device) != cudaSuccess
        || smCount <= 0) {
        return 1;
    }
    long long budget = ((long long)smCount * resident + rows - 1) / rows;
    if (budget < 1) {
        return 1;
    }
    return budget > 256 ? 256 : (int)budget;
}

void glm_topk_from_scores(GlmCtx* ctx, int32_t* out_idx,
    __nv_bfloat16* out_scores,
    const void* scores, const int32_t* row_len,
    int32_t* hist, int32_t* meta,
    int batch, int stride, int topk, int num_splits,
    int cpWorldSize, int cpRank) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    // 0 = automatic budget (see idx_auto_num_splits); negative clamps to 1.
    // The split grid is shared by both radix passes and the gather kernels, so
    // size it by the least resident of those instantiations.
    if (num_splits == 0) {
        const int resident = std::min(std::min(
            idx_query_occupancy(idx_radix_hist_split_kernel<false>, IDX_GATHER_THREADS, 0),
            idx_query_occupancy(idx_radix_hist_split_kernel<true>, IDX_GATHER_THREADS, 0)),
            std::min(idx_query_occupancy(idx_gather_count_kernel, IDX_GATHER_THREADS, 0),
                     idx_query_occupancy(idx_gather_write_kernel, IDX_GATHER_THREADS, 0)));
        num_splits = idx_auto_num_splits(resident, batch);
    } else if (num_splits < 0) {
        num_splits = 1;
    }
    dim3 grid(num_splits, batch);
    // Full rows (including padded CP candidate merges) amortize the extra
    // histogram launches once splitting removes about 6K scores from each
    // CTA's scan. Query count influences this through the caller's split budget:
    // e.g. 8K/4 splits and 12K/2 splits qualify, but a single split never does.
    // With device-side lengths, stride is only capacity: retain the conservative
    // cutoff so short/identity rows in large buffers don't pay extra launches.
    const int splitScan = (stride + num_splits - 1) / num_splits;
    const bool parallelRadix = stride > topk && (row_len
        ? stride > 32768
        : num_splits > 1 && stride - splitScan >= 6144);
    if (!parallelRadix) {
        idx_radix_threshold_kernel<<<batch, 256, 0, stream>>>(
            meta, out_idx, out_scores, row_len, (const __nv_bfloat16*)scores,
            stride, topk, cpWorldSize, cpRank);
    } else {
        // Single-row small memsets have expensive graph-launch overhead. BF16
        // +0 clears the same bits (two elements per I32). Multi-row pitched
        // memset has no such overhead in measurements and avoids clearing the
        // unused scratch columns, which matters for large batches.
        if (batch == 1) {
            glm_fill(ctx, hist, 0.0, IDX_RADIX_BUCKETS, "I32");
        } else {
            cudaMemset2DAsync(hist, (size_t)IDX_SCRATCH_I32 * sizeof(int32_t), 0,
                              (size_t)IDX_RADIX_BUCKETS * sizeof(int32_t), batch, stream);
        }
        idx_radix_hist_split_kernel<false><<<grid, 256, 0, stream>>>(
            (const __nv_bfloat16*)scores, row_len, hist, meta, stride, topk);
        idx_radix_high_threshold_kernel<<<batch, 256, 0, stream>>>(
            hist, meta, out_idx, out_scores, row_len,
            (const __nv_bfloat16*)scores, stride, topk, cpWorldSize, cpRank);
        idx_radix_hist_split_kernel<true><<<grid, 256, 0, stream>>>(
            (const __nv_bfloat16*)scores, row_len, hist, meta, stride, topk);
        idx_radix_low_threshold_kernel<<<batch, 256, 0, stream>>>(
            hist, meta, row_len, stride, topk);
    }
    idx_gather_count_kernel<<<grid, IDX_GATHER_THREADS, 0, stream>>>(
        (const __nv_bfloat16*)scores, row_len, meta, hist, stride);
    idx_gather_write_kernel<<<grid, IDX_GATHER_THREADS, 0, stream>>>(
        (const __nv_bfloat16*)scores, row_len, meta, hist, out_idx, out_scores,
        stride, topk, cpWorldSize, cpRank);
}

// ---------------------------------------------------------------------------
// Multi-block indexer scoring: writes per-position scores into a buffer instead
// of feeding a serial heap. grid = (num_splits, totalQ); one warp per KV
// position so decode (small totalQ) spreads across many SMs. The per-head math
// matches indexer_score_topk_kernel exactly (per-head ReLU, weighted sum, final
// bf16 round) so the downstream selection is identical.
// ---------------------------------------------------------------------------

// Shared device helper: check if a KV position is masked out. Defined here so
// both the v2 score kernel below and the two-level prefill kernels can use it.
//
// `pos` must be a GLOBAL sequence position and `mask_prefix_len` must be derived
// from the GLOBAL kv length. The mask bitmap is built over global KV columns --
// row q, column (P - prefix) for global position P (see buildChunkedMTPMask in
// mtp.ts, which lays out bit q*maskKvLen + c). Under context parallelism a shard
// holds every cpWorldSize-th token, so passing a shard-local position here reads
// an unrelated column: the tree mask then gates the wrong tokens entirely,
// leaving the draft positions it exists to separate completely unmasked.
static __device__ __forceinline__ bool idx_is_masked(
    int pos, int qLocalPos,
    const uint8_t* mask_ptr, int mask_kv_len_val, int mask_prefix_len)
{
    if (!mask_ptr) return false;
    if (pos < mask_prefix_len) return false;
    int mask_offset = qLocalPos * mask_kv_len_val + (pos - mask_prefix_len);
    return !((mask_ptr[mask_offset >> 3] >> (mask_offset & 7)) & 1);
}

// Inclusive causal limit for one query row, in THIS SHARD's local KV positions.
//
// The causal bound is a statement about global sequence positions -- query at
// global position P may see global KV [0, P] -- so under context parallelism it
// has to be evaluated in global coordinates and only then mapped back. Local
// position p holds global position p*cpWorldSize + cpRank, so the last local
// position at or below a global limit G is floor((G - cpRank) / cpWorldSize).
//
// Every consumer of the causal bound must come through here. Deriving any one of
// them from the LOCAL kvLen instead (prefixLen = kvLen - numQueries) silently
// disagrees with the others: rowLen advertises positions as valid that the score
// pass never writes, and the top-k then selects whatever stale bf16 the recycled
// `scores` buffer happened to hold at those slots.
//
// qSeqPos is the query's position within its sequence, already shifted by
// qGlobalStart for the query-sharded path. cpWorldSize <= 1 means non-CP, where
// kvLen == globalKvLen and this reduces to the original arithmetic exactly.
static __device__ __forceinline__ int idx_local_causal_limit(
    int qSeqPos, int numQueries, int causal,
    int kvLen, int globalKvLen, int cpWorldSize, int cpRank)
{
    if (cpWorldSize <= 1) {
        const int prefixLen = max(0, kvLen - numQueries);
        return causal ? (prefixLen + qSeqPos) : (kvLen - 1);
    }
    const int globalPrefix = max(0, globalKvLen - numQueries);
    const int globalLimit  = causal ? (globalPrefix + qSeqPos) : (globalKvLen - 1);
    // FLOOR division, not C's truncation-toward-zero. globalLimit < cpRank means
    // this rank owns nothing at or below the limit (its first token is global
    // position cpRank), so the answer is -1 -> numValid 0. Truncation returns 0
    // instead, which hands every rank its local position 0 and makes query q see
    // global positions 0..W-1 rather than just 0. That hits the first W-1 query
    // rows of every sequence, so a short prompt diverges from the replicated
    // path on its very first prefill.
    if (globalLimit < cpRank) return -1;
    return min((globalLimit - cpRank) / cpWorldSize, kvLen - 1);
}

// HAS_MASK is a compile-time switch: the whole custom-mask path (extra args,
// per-position bit test) is elided when false, so the common no-mask case pays
// nothing. The masked variant writes -inf for masked positions so they land in
// the lowest histogram bucket and are never selected — identical semantics to
// the two-level prefill score kernel.
template <bool HAS_MASK, bool FLAT>
__global__ void idx_score_kernel(
    __nv_bfloat16* __restrict__ scores,       // [totalQ, maxKv]
    int32_t* __restrict__ row_len,            // [totalQ]  (numValid per query)
    const __nv_bfloat16* __restrict__ q,      // [totalQ, idxNHeads, idxHeadDim]
    const uint8_t* __restrict__ kData,        // [maxPages, pageSize, idxHeadDim]
    const float* __restrict__ kScaleData,     // [maxPages, pageSize]
    const __nv_bfloat16* __restrict__ weights,// [totalQ, idxNHeads]
    const int32_t* __restrict__ pageIndices, const int32_t* __restrict__ pageIndptr,
    const int32_t* __restrict__ lastPageLen, const int32_t* __restrict__ qoIndptr,
    float scale, int idxNHeads, int idxHeadDim, int pageSize, int maxKv, int causal,
    int qGlobalStart,
    const uint8_t* __restrict__ custom_mask,
    const int32_t* __restrict__ mask_indptr, const int32_t* __restrict__ mask_kv_len,
    int cpWorldSize, int cpRank,
    const int32_t* __restrict__ globalLastPageLen,
    const int32_t* __restrict__ kvTokenIndptr
) {
    const int qIdx = blockIdx.y;
    const int globalQuery = qIdx + qGlobalStart;
    int seq = 0;
    while (qoIndptr[seq + 1] <= globalQuery) {
        seq++;
    }
    const int qSeqPos = globalQuery - qoIndptr[seq];
    const int numQueries = qoIndptr[seq + 1] - qoIndptr[seq];
    const int pageStart = FLAT ? 0 : pageIndptr[seq];
    const int numPages = FLAT ? 0 : pageIndptr[seq + 1] - pageStart;
    const int flatStart = FLAT ? kvTokenIndptr[seq] : 0;
    const int kvLen = FLAT
        ? kvTokenIndptr[seq + 1] - flatStart
        : (numPages > 0 ? (numPages - 1) * pageSize + lastPageLen[seq] : 0);
    // Under CP this shard holds every cpWorldSize-th token, so the causal bound
    // must be taken in global coordinates and mapped back (see the helper).
    const int cpW = (cpWorldSize > 1 && globalLastPageLen) ? cpWorldSize : 1;
    const int globalKvLen = (cpW > 1)
        ? (numPages > 0 ? (numPages - 1) * (pageSize * cpW) + globalLastPageLen[seq] : 0)
        : kvLen;
    const int numValid = idx_local_causal_limit(
        qSeqPos, numQueries, causal, kvLen, globalKvLen, cpW, cpRank) + 1;

    if (blockIdx.x == 0 && threadIdx.x == 0) row_len[qIdx] = numValid;

    // Block-uniform early exit: this block's first candidate position is
    // blockIdx.x * warpsPerBlock; if it already exceeds the causal limit the
    // grid-stride loop below cannot run, and the Q smem load + barrier would
    // be wasted. Placed after the row_len write so block 0 still records it.
    if (blockIdx.x * (blockDim.x >> 5) >= numValid) return;

    // Custom-mask setup — compiled out entirely when !HAS_MASK.
    const uint8_t* mask_ptr = nullptr;
    int mask_kv_len_val = 0, mask_prefix_len = 0;
    if constexpr (HAS_MASK) {
        mask_ptr = custom_mask + mask_indptr[seq];
        mask_kv_len_val = mask_kv_len ? mask_kv_len[seq] : numQueries;
        // Global coordinates: the mask window is the last mask_kv_len_val tokens
        // of the FULL sequence, not of this shard. Identical to kvLen off CP.
        mask_prefix_len = max(0, globalKvLen - mask_kv_len_val);
    }
    // Local pos -> global pos for the mask lookup (identity when cpMaskW == 0).
    const int cpMaskW = (cpW > 1) ? cpW : 0;
    const int maskRow = qSeqPos;

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* w_s = q_s + idxNHeads * idxHeadDim;
    for (int i = threadIdx.x; i < idxNHeads * idxHeadDim; i += blockDim.x)
        q_s[i] = q[(size_t)qIdx * idxNHeads * idxHeadDim + i];
    for (int i = threadIdx.x; i < idxNHeads; i += blockDim.x)
        w_s[i] = weights[qIdx * idxNHeads + i];
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpsPerBlock = blockDim.x >> 5;
    for (int pos = blockIdx.x * warpsPerBlock + warp; pos < numValid;
         pos += gridDim.x * warpsPerBlock) {
        if constexpr (HAS_MASK) {
            if (idx_is_masked(cp_remap(pos, cpMaskW, cpRank), maskRow,
                              mask_ptr, mask_kv_len_val, mask_prefix_len)) {
                if (lane == 0)
                    scores[(size_t)qIdx * maxKv + pos] = __float2bfloat16(-INFINITY);
                continue;
            }
        }
        size_t rowId;
        if constexpr (FLAT) {
            rowId = (size_t)flatStart + pos;
        } else {
            const int pageId = pageIndices[pageStart + pos / pageSize];
            rowId = (size_t)pageId * pageSize + pos % pageSize;
        }
        const uint8_t* kbase = kData + rowId * idxHeadDim;
        const float kScale = kScaleData[rowId];
        float acc = 0.f;
        for (int h = 0; h < idxNHeads; h++) {
            const __nv_bfloat16* qh = q_s + h * idxHeadDim;
            float partial = 0.f;
            for (int d = lane; d < idxHeadDim; d += 32)
                partial += __bfloat162float(qh[d]) * fp8_e4m3_to_float(kbase[d]);
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor_sync(0xffffffff, partial, off);
            if (lane == 0) acc += __bfloat162float(w_s[h]) * fmaxf(partial * kScale * scale, 0.f);
        }
        if (lane == 0)
            scores[(size_t)qIdx * maxKv + pos] = __float2bfloat16(acc);
    }
}

// SM120 decode scorer for the production indexer shape. The selector scratch is
// idle during scoring, so each query uses its row for an FP8 Q matrix followed by
// effective FP32 head weights.
namespace idxfp8 {
constexpr int NHEADS = 32;
constexpr int HD = 128;
constexpr int Q_BYTES = NHEADS * HD;
constexpr int WARPS = 8;
constexpr int K_ROWS = 8;

template <bool SWIZZLED>
__device__ __forceinline__ const uint8_t* smem_row_ptr(
    const uint8_t* base, int row, int col) {
    if constexpr (SWIZZLED) {
        const int swizzledVec = (col >> 4) ^ (row & 7);
        return base + row * HD + swizzledVec * 16 + (col & 15);
    }
    return base + row * HD + col;
}

__device__ __forceinline__ void ldm_x4(uint32_t& a0, uint32_t& a1,
                                       uint32_t& a2, uint32_t& a3,
                                       const void* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}

__device__ __forceinline__ void ldm_x2(uint32_t& b0, uint32_t& b1,
                                       const void* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b0), "=r"(b1) : "r"(addr));
}

template <bool SWIZZLED = false>
__device__ __forceinline__ void load_a(uint32_t& a0, uint32_t& a1,
                                       uint32_t& a2, uint32_t& a3,
                                       const uint8_t* base, int lane,
                                       int kOffset = 0) {
    const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int col = kOffset + (lane >> 4) * 16;
    ldm_x4(a0, a1, a2, a3, smem_row_ptr<SWIZZLED>(base, row, col));
}

template <bool SWIZZLED = false>
__device__ __forceinline__ void load_b(uint32_t& b0, uint32_t& b1,
                                       const uint8_t* base, int lane,
                                       int kOffset = 0) {
    const int row = lane & 7;
    const int col = kOffset + ((lane >> 3) & 1) * 16;
    ldm_x2(b0, b1, smem_row_ptr<SWIZZLED>(base, row, col));
}

__device__ __forceinline__ void tma_load_2d(
    void* dst, const CUtensorMap& map, int x, int y, uint64_t* barrier) {
    const uint32_t dstAddr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    const uint32_t barrierAddr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.tile "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(dstAddr), "l"(reinterpret_cast<uint64_t>(&map)),
           "r"(x), "r"(y), "r"(barrierAddr) : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* barrier, uint32_t bytes) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "{ .reg .b64 state;\n"
        "mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n }\n"
        :: "r"(addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{ .reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n"
            "selp.u32 %0, 1, 0, p;\n }\n"
            : "=r"(done) : "r"(addr) : "memory");
    }
}

struct Acc { float x0, x1, x2, x3; };

__device__ __forceinline__ Acc mma(uint32_t a0, uint32_t a1, uint32_t a2,
                                   uint32_t a3, uint32_t b0, uint32_t b1,
                                   Acc c) {
    Acc d;
    // UE8M0 0x7f is unity.  Explicit instruction scales keep this on the SM120
    // block-scaled FP8 path while row scales remain software-visible FP32.
    constexpr uint32_t one = 0x7f7f7f7f;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32"
        ".row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
        "{%14}, {%15,%16}, {%17}, {%18,%19};"
        : "=f"(d.x0), "=f"(d.x1), "=f"(d.x2), "=f"(d.x3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c.x0), "f"(c.x1), "f"(c.x2), "f"(c.x3),
          "r"(one), "n"((uint16_t)0), "n"((uint16_t)0), "r"(one),
          "n"((uint16_t)0), "n"((uint16_t)0));
    return d;
}

// NH heads per call: 32 for the replicated/standalone path, 32/worldSize for
// Row-sharded quantization. 8 lanes per head, 4 heads per warp. weightsOffset
// shifts into a Replicated [rows, globalNHeads] weight row so a Row-sharded
// caller reads its contiguous head slice in place (no gather).
template <int NH>
__global__ void quantize_q_kernel(const __nv_bfloat16* __restrict__ q,
                                  const __nv_bfloat16* __restrict__ weights,
                                  int weightsStride, int weightsOffset,
                                  uint8_t* __restrict__ q8Data, int q8Stride,
                                  float* __restrict__ effectiveWeights, int ewStride,
                                  float scale) {
    static_assert(NH % 4 == 0, "4 head groups per warp");
    const int qi = blockIdx.x;
    const int tid = threadIdx.x;
    const int head = tid >> 3;
    const int lane = tid & 7;
    uint8_t* q8 = q8Data + (size_t)qi * q8Stride;
    float* ew = effectiveWeights + (size_t)qi * ewStride;
    const __nv_bfloat16* src = q + (size_t)qi * (NH * HD) + head * HD;

    float vmax = 0.f;
#pragma unroll
    for (int d = lane; d < HD; d += 8)
        vmax = fmaxf(vmax, fabsf(__bfloat162float(src[d])));
#pragma unroll
    for (int off = 4; off; off >>= 1)
        vmax = fmaxf(vmax, __shfl_down_sync(0xffffffffu, vmax, off, 8));
    if (lane == 0) {
        const float qs = indexer_ue8m0_scale(vmax);
        ew[head] = __bfloat162float(weights[(size_t)qi * weightsStride + weightsOffset + head]) * qs * scale;
    }
    __syncwarp();
    const float inv = 1.f / __shfl_sync(0xffffffffu, lane == 0 ? indexer_ue8m0_scale(vmax) : 0.f,
                                        (head & 3) * 8, 32);
#pragma unroll
    for (int d = lane; d < HD; d += 8) {
        q8[head * HD + d] = __nv_cvt_float_to_fp8(
            __bfloat162float(src[d]) * inv, __NV_SATFINITE, __NV_E4M3);
    }
}

template <bool HAS_MASK, bool FLAT>
__global__ void score_kernel(
    __nv_bfloat16* __restrict__ scores, int32_t* __restrict__ row_len,
    const uint8_t* __restrict__ kData, const int32_t* __restrict__ pageIndices,
    const int32_t* __restrict__ pageIndptr, const int32_t* __restrict__ lastPageLen,
    const int32_t* __restrict__ qoIndptr, int pageSize, int maxKv, int causal, int qGlobalStart,
    const uint8_t* __restrict__ custom_mask, const int32_t* __restrict__ mask_indptr,
    const int32_t* __restrict__ mask_kv_len, int cpWorldSize, int cpRank,
    const int32_t* __restrict__ globalLastPageLen,
    const int32_t* __restrict__ kvTokenIndptr,
    const uint8_t* __restrict__ q8Data, int q8Stride,
    const float* __restrict__ effectiveWeights, int weightStride,
    const float* __restrict__ kScaleData) {
    const int qi = blockIdx.y;
    const int globalQuery = qi + qGlobalStart;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    int seq = 0;
    while (qoIndptr[seq + 1] <= globalQuery) {
        seq++;
    }
    const int qSeqPos = globalQuery - qoIndptr[seq];
    const int nQuery = qoIndptr[seq + 1] - qoIndptr[seq];
    const int pageStart = FLAT ? 0 : pageIndptr[seq];
    const int nPages = FLAT ? 0 : pageIndptr[seq + 1] - pageStart;
    const int flatStart = FLAT ? kvTokenIndptr[seq] : 0;
    const int kvLen = FLAT ? kvTokenIndptr[seq + 1] - flatStart
        : (nPages ? (nPages - 1) * pageSize + lastPageLen[seq] : 0);
    const int cpW = (cpWorldSize > 1 && globalLastPageLen) ? cpWorldSize : 1;
    const int globalKvLen = cpW > 1
        ? (nPages ? (nPages - 1) * pageSize * cpW + globalLastPageLen[seq] : 0) : kvLen;
    const int numValid = idx_local_causal_limit(
        qSeqPos, nQuery, causal, kvLen, globalKvLen, cpW, cpRank) + 1;
    if (blockIdx.x == 0 && threadIdx.x == 0) row_len[qi] = numValid;

    // Block-uniform early exit: this block's first tile is blockIdx.x * WARPS;
    // when its base position is already past the causal limit the tile loop
    // cannot run, and the Q smem load + barrier would be wasted. Placed after
    // the row_len write so block 0 still records it.
    if (blockIdx.x * WARPS * K_ROWS >= numValid) return;

    const uint8_t* tmp = q8Data + (size_t)qi * q8Stride;
    const float* ew = effectiveWeights + (size_t)qi * weightStride;
    extern __shared__ uint8_t smem[];
    uint8_t* sq = smem;
    uint8_t* sk = sq + Q_BYTES + warp * K_ROWS * HD;
    float* skScale = reinterpret_cast<float*>(sq + Q_BYTES + WARPS * K_ROWS * HD) + warp * K_ROWS;
    for (int i = threadIdx.x; i < Q_BYTES; i += blockDim.x) sq[i] = tmp[i];
    __syncthreads();

    const uint8_t* mask = nullptr;
    int maskKv = 0, maskPrefix = 0;
    if constexpr (HAS_MASK) {
        mask = custom_mask + mask_indptr[seq];
        maskKv = mask_kv_len ? mask_kv_len[seq] : nQuery;
        maskPrefix = max(0, globalKvLen - maskKv);
    }
    const int cpMaskW = cpW > 1 ? cpW : 0;
    const int tilesPerGrid = gridDim.x * WARPS;
    for (int tile = blockIdx.x * WARPS + warp; tile * K_ROWS < numValid;
         tile += tilesPerGrid) {
        const int base = tile * K_ROWS;
        for (int i = lane * 4; i < K_ROWS * HD; i += 32 * 4) {
            const int n = i / HD, col = i % HD, pos = base + n;
            uint32_t v = 0;
            if (pos < numValid) {
                const uint8_t* row;
                if constexpr (FLAT) {
                    const size_t rowId = (size_t)flatStart + (size_t)pos;
                    row = kData + rowId * HD;
                    if (col == 0) skScale[n] = kScaleData[rowId];
                } else {
                    const int pid = pageIndices[pageStart + pos / pageSize];
                    const size_t rowId = (size_t)pid * (size_t)pageSize + (size_t)(pos % pageSize);
                    row = kData + rowId * HD;
                    if (col == 0) skScale[n] = kScaleData[rowId];
                }
                v = *reinterpret_cast<const uint32_t*>(row + col);
            } else if (col == 0) {
                skScale[n] = 0.f;
            }
            *reinterpret_cast<uint32_t*>(sk + i) = v;
        }
        __syncwarp();

        Acc lo{0.f, 0.f, 0.f, 0.f}, hi{0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kk = 0; kk < HD; kk += 32) {
            uint32_t a0, a1, a2, a3, b0, b1;
            load_b(b0, b1, sk + kk, lane);
            load_a(a0, a1, a2, a3, sq + kk, lane);
            lo = mma(a0, a1, a2, a3, b0, b1, lo);
            load_a(a0, a1, a2, a3, sq + 16 * HD + kk, lane);
            hi = mma(a0, a1, a2, a3, b0, b1, hi);
        }
        const int gid = lane >> 2;
        const int pair = lane & 3;
        float s0 = ew[gid] * fmaxf(lo.x0, 0.f) + ew[gid + 8] * fmaxf(lo.x2, 0.f)
                 + ew[gid + 16] * fmaxf(hi.x0, 0.f) + ew[gid + 24] * fmaxf(hi.x2, 0.f);
        float s1 = ew[gid] * fmaxf(lo.x1, 0.f) + ew[gid + 8] * fmaxf(lo.x3, 0.f)
                 + ew[gid + 16] * fmaxf(hi.x1, 0.f) + ew[gid + 24] * fmaxf(hi.x3, 0.f);
        s0 *= skScale[pair * 2];
        s1 *= skScale[pair * 2 + 1];
#pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, off);
            s1 += __shfl_xor_sync(0xffffffffu, s1, off);
        }
        if (gid == 0) {
            const int p0 = base + pair * 2, p1 = p0 + 1;
            bool ok0 = p0 < numValid, ok1 = p1 < numValid;
            if constexpr (HAS_MASK) {
                if (ok0) ok0 = !idx_is_masked(cp_remap(p0, cpMaskW, cpRank), qSeqPos,
                                                mask, maskKv, maskPrefix);
                if (ok1) ok1 = !idx_is_masked(cp_remap(p1, cpMaskW, cpRank), qSeqPos,
                                                mask, maskKv, maskPrefix);
            }
            if (p0 < numValid) scores[(size_t)qi * maxKv + p0] =
                ok0 ? __float2bfloat16(s0) : __float2bfloat16(-INFINITY);
            if (p1 < numValid) scores[(size_t)qi * maxKv + p1] =
                ok1 ? __float2bfloat16(s1) : __float2bfloat16(-INFINITY);
        }
        __syncwarp();
    }
}
} // namespace idxfp8

static bool indexer_use_fp8_mma(int nHeads, int headDim) {
    const char* env = std::getenv("GLM_INDEXER_DECODE_FP8_MMA");
    return nHeads == idxfp8::NHEADS && headDim == idxfp8::HD
        && (!env || std::strcmp(env, "0") != 0);
}

// Full v2 indexer top-k: multi-block score -> radix select. Drop-in
// replacement for glm_indexer_score_topk with the same out_idx semantics.
// scratch: scores [totalQ, maxKv] bf16, rowLen [totalQ] i32, hist [totalQ,1056]
// i32, meta [totalQ, 4] i32.
void glm_indexer_score_topk_v2(GlmCtx* ctx, int32_t* out_idx,
    __nv_bfloat16* out_scores,
    const void* q, const void* kData, const float* kScaleData, const void* weights,
    const int32_t* pageIndices, const int32_t* pageIndptr,
    const int32_t* lastPageLen, const int32_t* qoIndptr,
    float scale, int totalQ, int idxNHeads, int idxHeadDim,
    int pageSize, int topk, int causal, int qGlobalStart,
    const uint8_t* custom_mask, const int32_t* mask_indptr, const int32_t* mask_kv_len,
    void* scores, int32_t* rowLen, int32_t* hist, int32_t* meta,
    int maxKv, int num_splits, int cpWorldSize, int cpRank,
    const int32_t* globalLastPageLen, const int32_t* kvTokenIndptr,
    const float* precomputed_ew) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    int block = 256;
    size_t smem = (size_t)idxNHeads * idxHeadDim * sizeof(__nv_bfloat16)
                + idxNHeads * sizeof(__nv_bfloat16);
    const size_t fp8Smem = (size_t)idxfp8::Q_BYTES
        + idxfp8::WARPS * idxfp8::K_ROWS * idxfp8::HD
        + idxfp8::WARPS * idxfp8::K_ROWS * sizeof(float);
    const int effectiveCpWorldSize = kvTokenIndptr ? 0 : cpWorldSize;
    const int effectiveCpRank = kvTokenIndptr ? 0 : cpRank;
    // Dispatch the mask-free variant when there is no custom mask so its bit-test
    // path is compiled out (zero cost for the common case).
    const bool hasMask = custom_mask && mask_indptr;
    // precomputed_ew: q is already-quantized FP8 [totalQ, idxNHeads, idxHeadDim]
    // with a matching [totalQ, idxNHeads] F32 effective-weights tensor; the
    // internal quantize launch is skipped. Only valid for the FP8 shape.
    const bool precomputed = precomputed_ew != nullptr;
    if (precomputed && !indexer_use_fp8_mma(idxNHeads, idxHeadDim)) {
        fprintf(stderr, "glm_indexer_score_topk_v2: precomputed_ew requires "
                        "idxNHeads=%d idxHeadDim=%d and FP8 MMA enabled (got %d/%d)\n",
                idxfp8::NHEADS, idxfp8::HD, idxNHeads, idxHeadDim);
        return;
    }
    const bool useFp8Mma = indexer_use_fp8_mma(idxNHeads, idxHeadDim) || precomputed;
    // 0 = automatic budget (see idx_auto_num_splits), resolved after the
    // dispatch variant is known so the occupancy query targets the exact
    // specialization the LAUNCH macros select below (HAS_MASK, FLAT);
    // negative clamps to 1. The resolved budget is deliberately forwarded to
    // glm_topk_from_scores, sizing both stages from one wave calculation:
    // the stages' grids are independent, and their 256-thread blocks are
    // thread-bound at the same resident count on current targets. Direct
    // callers can still pass 0 to glm_topk_from_scores for per-stage sizing
    // if the merge kernels' occupancy ever diverges.
    if (num_splits == 0) {
        const bool flat = kvTokenIndptr != nullptr;
        const int resident = useFp8Mma
            ? (flat
                ? (hasMask
                    ? idx_query_occupancy(idxfp8::score_kernel<true, true>, block, fp8Smem)
                    : idx_query_occupancy(idxfp8::score_kernel<false, true>, block, fp8Smem))
                : (hasMask
                    ? idx_query_occupancy(idxfp8::score_kernel<true, false>, block, fp8Smem)
                    : idx_query_occupancy(idxfp8::score_kernel<false, false>, block, fp8Smem)))
            : (flat
                ? (hasMask
                    ? idx_query_occupancy(idx_score_kernel<true, true>, block, smem)
                    : idx_query_occupancy(idx_score_kernel<false, true>, block, smem))
                : (hasMask
                    ? idx_query_occupancy(idx_score_kernel<true, false>, block, smem)
                    : idx_query_occupancy(idx_score_kernel<false, false>, block, smem)));
        num_splits = idx_auto_num_splits(resident, totalQ);
    } else if (num_splits < 0) {
        num_splits = 1;
    }
    dim3 grid(num_splits, totalQ);
    if (useFp8Mma) {
        const uint8_t* q8Data;
        const float* effectiveWeights;
        int q8StrideBytes, ewStride;
        if (precomputed) {
            q8Data = reinterpret_cast<const uint8_t*>(q);
            effectiveWeights = precomputed_ew;
            q8StrideBytes = idxNHeads * idxHeadDim;
            ewStride = idxNHeads;
        } else {
            uint8_t* q8Scratch = reinterpret_cast<uint8_t*>(hist);
            q8Data = q8Scratch;
            effectiveWeights = reinterpret_cast<float*>(q8Scratch + idxfp8::Q_BYTES);
            q8StrideBytes = IDX_SCRATCH_I32 * sizeof(int32_t);
            ewStride = IDX_SCRATCH_I32;
            idxfp8::quantize_q_kernel<idxfp8::NHEADS><<<totalQ, idxfp8::NHEADS * 8, 0, stream>>>(
                (const __nv_bfloat16*)q, (const __nv_bfloat16*)weights,
                idxfp8::NHEADS, 0,
                q8Scratch, IDX_SCRATCH_I32 * sizeof(int32_t),
                reinterpret_cast<float*>(q8Scratch + idxfp8::Q_BYTES), IDX_SCRATCH_I32, scale);
        }
        #define LAUNCH_IDX_FP8(HAS_MASK, FLAT) \
        idxfp8::score_kernel<(HAS_MASK), (FLAT)><<<grid, block, fp8Smem, stream>>>( \
            (__nv_bfloat16*)scores, rowLen, (const uint8_t*)kData, pageIndices, pageIndptr, \
            lastPageLen, qoIndptr, pageSize, maxKv, causal, qGlobalStart, custom_mask, mask_indptr, \
            mask_kv_len, effectiveCpWorldSize, effectiveCpRank, globalLastPageLen, \
            kvTokenIndptr, q8Data, q8StrideBytes, \
            effectiveWeights, ewStride, kScaleData)
        if (kvTokenIndptr) {
            if (hasMask) LAUNCH_IDX_FP8(true, true); else LAUNCH_IDX_FP8(false, true);
        } else {
            if (hasMask) LAUNCH_IDX_FP8(true, false); else LAUNCH_IDX_FP8(false, false);
        }
#undef LAUNCH_IDX_FP8
        glm_topk_from_scores(ctx, out_idx, out_scores, scores, rowLen, hist, meta,
                             totalQ, maxKv, topk, num_splits,
                             effectiveCpWorldSize, effectiveCpRank);
        return;
    }
#define LAUNCH_IDX_SCORE(HAS_MASK, FLAT) \
    idx_score_kernel<(HAS_MASK), (FLAT)><<<grid, block, smem, stream>>>( \
        (__nv_bfloat16*)scores, rowLen, (const __nv_bfloat16*)q, (const uint8_t*)kData, \
        kScaleData, (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr, \
        scale, idxNHeads, idxHeadDim, pageSize, maxKv, causal, \
        qGlobalStart, custom_mask, mask_indptr, mask_kv_len, \
        effectiveCpWorldSize, effectiveCpRank, globalLastPageLen, kvTokenIndptr)
    if (kvTokenIndptr) {
        if (hasMask) LAUNCH_IDX_SCORE(true, true); else LAUNCH_IDX_SCORE(false, true);
    } else {
        if (hasMask) LAUNCH_IDX_SCORE(true, false); else LAUNCH_IDX_SCORE(false, false);
    }
#undef LAUNCH_IDX_SCORE
    glm_topk_from_scores(ctx, out_idx, out_scores, scores, rowLen, hist, meta,
                         totalQ, maxKv, topk, num_splits, effectiveCpWorldSize, effectiveCpRank);
}

// Standalone indexer Q quantization for Row-sharded callers. Outputs are
// dedicated contiguous tensors: q8 [rows, nHeads*HD] U8 and effectiveWeights
// [rows, nHeads] F32 — the same per-row layout the fused paths carve out of
// their histogram scratch, so results can feed those kernels unchanged.
void glm_indexer_quantize_q(GlmCtx* ctx, uint8_t* out_q8, float* out_ew,
    const void* q, const void* weights, int nHeads, int weightsStride,
    int weightsOffset, int totalQ, float scale) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(weights);
    switch (nHeads) {
#define IDX_QUANT(NH) idxfp8::quantize_q_kernel<NH><<<totalQ, NH * 8, 0, stream>>>( \
        qb, wb, weightsStride, weightsOffset, out_q8, NH * idxfp8::HD, out_ew, NH, scale)
      case 32: IDX_QUANT(32); break;
      case 16: IDX_QUANT(16); break;
      case 8:  IDX_QUANT(8);  break;
      case 4:  IDX_QUANT(4);  break;
      default:
        fprintf(stderr, "glm_indexer_quantize_q: unsupported nHeads=%d\n", nHeads);
        break;
#undef IDX_QUANT
    }
}

// ---------------------------------------------------------------------------
// Two-level prefill indexer: score once → coarse/fine histogram top-K.
//
// Replaces the v1 serial heap kernel for prefill. Scores are computed once
// into a [totalQ, maxKv] BF16 buffer (same as v2), then a 2-level histogram
// (1024 coarse + 64 fine = 65536 total buckets) selects the exact top-K.
// This remains the high-throughput path for large query counts while keeping a
// single scoring pass and bounded histogram scratch.
//
// Pipeline (6 kernel launches, same stream):
//   1. score into buffer     — compute score per position, write to [totalQ, maxKv]
//   2. coarse hist from buf  — read scores, bin into 1024 coarse buckets
//   3. coarse threshold      — find winning coarse bucket (1 block/query)
//   4. fine hist from buf    — read scores in winning bucket, bin into 64 fine buckets
//   5. fine threshold        — find exact threshold (1 block/query)
//   6. gather from buf       — read scores, write positions at/above threshold
//
// Scratch: scores [totalQ, maxKv] BF16 + coarseHist [totalQ, 1024] i32
//   + fineHist [totalQ, 64] i32 + meta [totalQ, 4] i32 + rowLen [totalQ] i32
//   = ~1.12 GB for totalQ=4096, maxKv=135K (vs 2.1 GB for v2)
//
// The per-head scoring math matches idx_score_kernel exactly: per-head ReLU,
// weighted sum, final bf16 round. Custom mask support: masked positions get
// -inf scores so they fall into the lowest histogram bucket and are never
// selected.
// ---------------------------------------------------------------------------

#define IDX_COARSE_BUCKETS 1024
#define IDX_FINE_BUCKETS 64   // 65536 bf16 keys / IDX_COARSE_BUCKETS

// ---------------------------------------------------------------------------
// Tensor-core (mma.sync m16n8k16 bf16) primitives for the score kernel.
// Same instruction/fragment layout the MoE GEMM uses (glm_mma_moe.cu); kept in
// a local namespace so the anonymous-namespace helpers there don't collide.
// ---------------------------------------------------------------------------
namespace idxmma {
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
// Pad smem row strides so the 16 ldmatrix rows don't all land in the same bank
// set (row byte-stride was a multiple of 128 -> up to 16-way conflict).
constexpr int PAD_A = 8;   // Q row stride = idxHeadDim + PAD_A
constexpr int PAD_B = 8;   // K row stride = TN + PAD_B

struct FragA { uint32_t reg[4]; };
struct FragB { uint32_t reg[2]; };
struct FragC { float reg[4]; };

// 16-byte cp.async (8 bf16). pred=false issues a 0-byte copy (leaves smem intact).
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    unsigned s = __cvta_generic_to_shared(smem);
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
        :: "r"(s), "l"(gmem), "n"(16), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mma_m16n8k16(FragC& d, const FragA& a, const FragB& b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d.reg[0]), "+f"(d.reg[1]), "+f"(d.reg[2]), "+f"(d.reg[3])
        : "r"(a.reg[0]), "r"(a.reg[1]), "r"(a.reg[2]), "r"(a.reg[3]),
          "r"(b.reg[0]), "r"(b.reg[1]));
}

// A operand: 16x16 bf16 tile of Q_h from row-major smem [rows, stride].
__device__ __forceinline__ void ldm_a(FragA& a, const __nv_bfloat16* s, int stride, int row_off) {
    int lane = threadIdx.x & 31;
    int row = lane & 7, mat = (lane >> 3) & 1, col = (lane >> 4) << 3;
    uint32_t addr = __cvta_generic_to_shared(s + (size_t)(row_off + mat * 8 + row) * stride + col);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a.reg[0]), "=r"(a.reg[1]), "=r"(a.reg[2]), "=r"(a.reg[3]) : "r"(addr));
}

// B operand: 16x8 (k,n) tile of K from k-major smem [k, stride]; .trans loads it
// as the col operand.
__device__ __forceinline__ void ldm_b(FragB& b, const __nv_bfloat16* s, int stride, int col_off) {
    int lane = threadIdx.x & 31;
    int row = lane & 15;
    uint32_t addr = __cvta_generic_to_shared(s + (size_t)row * stride + col_off);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(b.reg[0]), "=r"(b.reg[1]) : "r"(addr));
}
} // namespace idxmma


// ---------------------------------------------------------------------------
// Pass 1: Score into buffer via tensor cores (mma.sync m16n8k16 bf16).
//
//   score[q,pos] = Σ_h w[q,h] · ReLU(scale · Σ_d Q[q,h,d]·K[pos,d])
//
// K is shared across the idxNHeads heads (a single idxHeadDim vector per KV
// position), so each output tile loads its K slab once and streams the
// per-head Q tiles through it, folding scale·ReLU·weight into a persistent
// fp32 score accumulator (one MMA pass, no per-head materialization).
//
// Grid: (ceil(maxKv/TN), numQueryTiles). The block geometry is templated.
// blockIdx.y maps to (seq, query tile) so tiles never cross a sequence
// boundary; blockIdx.x is a 64-wide KV tile of global positions. Masked
// positions are written -inf (lowest histogram bucket); positions beyond the
// causal limit / kvLen are left untouched (never read by the histogram passes).
// ---------------------------------------------------------------------------
template <int TM, int TN, int WARPS, int Q_BUFFERS, bool FLAT>
__global__ void __launch_bounds__(WARPS * 32)
idx_prefill_score_mma_kernel(
    __nv_bfloat16* __restrict__ scores,       // [totalQ, maxKv]
    int32_t* __restrict__ rowLen,             // [totalQ]
    const __nv_bfloat16* __restrict__ q,      // [totalQ, idxNHeads, idxHeadDim]
    const uint8_t* __restrict__ kData,        // [maxPages, pageSize, idxHeadDim]
    const float* __restrict__ kScaleData,     // [maxPages, pageSize]
    const __nv_bfloat16* __restrict__ weights,// [totalQ, idxNHeads]
    const int32_t* __restrict__ pageIndices, const int32_t* __restrict__ pageIndptr,
    const int32_t* __restrict__ lastPageLen, const int32_t* __restrict__ qoIndptr,
    int totalQ, float scale, int idxNHeads, int idxHeadDim, int pageSize,
    int maxKv, int causal, int qGlobalStart,
    const uint8_t* __restrict__ custom_mask,
    const int32_t* __restrict__ mask_indptr, const int32_t* __restrict__ mask_kv_len,
    int cpWorldSize, int cpRank,
    const int32_t* __restrict__ globalLastPageLen,
    const int32_t* __restrict__ kvTokenIndptr)
{
    using namespace idxmma;
    constexpr int CTA = WARPS * 32;
    constexpr int NUM_M = TM / MMA_M;
    constexpr int NPW = TN / (MMA_N * WARPS);
    static_assert(TM % MMA_M == 0);
    static_assert(TN % (MMA_N * WARPS) == 0);
    static_assert(Q_BUFFERS == 1 || Q_BUFFERS == 2);

    // Tile each sequence's intersection with this shard. qStart remains local
    // to q/weights/output; sequence metadata uses the full batch coordinates.
    const int gy = blockIdx.y;
    int seq = -1, qStart = 0;
    {
        int acc = 0;
        for (int s = 0; ; s++) {
            const int qs = max(qoIndptr[s], qGlobalStart);
            const int qe = min(qoIndptr[s + 1], qGlobalStart + totalQ);
            const int nt = (max(0, qe - qs) + TM - 1) / TM;
            if (gy < acc + nt) {
                seq = s;
                qStart = qs - qGlobalStart + (gy - acc) * TM;
                break;
            }
            acc += nt;
            if (qoIndptr[s + 1] >= qGlobalStart + totalQ) {
                break;
            }
        }
    }
    if (seq < 0 || qStart >= totalQ) {
        return;
    }

    const int qoStart   = qoIndptr[seq];
    const int numQueries = qoIndptr[seq + 1] - qoStart;
    // Clamp to both the sequence end (multi-seq) and the local row count (shard).
    const int m_valid   = min(TM, min(qoIndptr[seq + 1] - qGlobalStart, totalQ) - qStart);
    const int pageStart = FLAT ? 0 : pageIndptr[seq];
    const int numPages  = FLAT ? 0 : pageIndptr[seq + 1] - pageStart;
    const int flatStart = FLAT ? kvTokenIndptr[seq] : 0;
    const int kvLen = FLAT
        ? kvTokenIndptr[seq + 1] - flatStart
        : (numPages > 0 ? (numPages - 1) * pageSize + lastPageLen[seq] : 0);
    // Under CP this shard holds every cpWorldSize-th token, so the causal bound
    // must be taken in global coordinates and mapped back (see the helper).
    const int cpW = (cpWorldSize > 1 && globalLastPageLen) ? cpWorldSize : 1;
    const int globalKvLen = (cpW > 1)
        ? (numPages > 0 ? (numPages - 1) * (pageSize * cpW) + globalLastPageLen[seq] : 0)
        : kvLen;
    const int tileStart = blockIdx.x * TN;
    if (tileStart >= kvLen) return;

    // Per-query causal limits, computed once and shared by all three consumers:
    // the tile prune below, rowLen, and the per-element epilogue check. Keeping
    // them on one value is the point -- when the epilogue used its own
    // local-kvLen formula it stopped short of what rowLen advertised, so the
    // top-k scanned score slots the MMA pass never wrote.
    // qGlobalStart shifts a shard's local query row to its true sequence position
    // for causal limits (0 in the non-sharded path → identical to before).
    __shared__ int s_causalLimit[TM];
    for (int ql = threadIdx.x; ql < m_valid; ql += CTA) {
        s_causalLimit[ql] = idx_local_causal_limit(
            (qStart + ql) - qoStart + qGlobalStart, numQueries, causal,
            kvLen, globalKvLen, cpW, cpRank);
    }
    __syncthreads();

    // Causal prune: skip the whole tile if it sits past the last query's limit.
    // The limit is nondecreasing in query position, so the last row bounds them all.
    if (tileStart > s_causalLimit[m_valid - 1]) return;   // block-uniform

    const uint8_t* mask_ptr = nullptr;
    int mask_kv_len_val = 0, mask_prefix_len = 0;
    if (custom_mask && mask_indptr) {
        mask_ptr = custom_mask + mask_indptr[seq];
        mask_kv_len_val = mask_kv_len ? mask_kv_len[seq] : numQueries;
        // Global coordinates: the mask window is the last mask_kv_len_val tokens
        // of the FULL sequence, not of this shard. Identical to kvLen off CP.
        mask_prefix_len = max(0, globalKvLen - mask_kv_len_val);
    }
    // Local pos -> global pos for the mask lookup (identity when cpMaskW == 0).
    const int cpMaskW = (cpW > 1) ? cpW : 0;

    // rowLen (= numValid per query) written once, by the first KV tile.
    if (blockIdx.x == 0) {
        for (int ql = threadIdx.x; ql < m_valid; ql += CTA)
            rowLen[qStart + ql] = s_causalLimit[ql] + 1;
    }

    const int strideA = idxHeadDim + PAD_A;   // padded Q row stride
    const int strideB = TN + PAD_B;            // padded K row stride

    extern __shared__ char smem[];
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem);  // [idxHeadDim, strideB] (d-major)
    __nv_bfloat16* qbuf0 = smem_b + (size_t)idxHeadDim * strideB;
    __nv_bfloat16* w_s = qbuf0 + (size_t)Q_BUFFERS * TM * strideA;
    __nv_bfloat16* qbuf[2] = {qbuf0, Q_BUFFERS == 2 ? qbuf0 + (size_t)TM * strideA : qbuf0};

    // Zero both Q buffers once: padded rows (ql >= m_valid) are never cp.async'd,
    // so they stay 0 and fold to nothing (avoids NaN from 0*inf on stale smem).
    for (int i = threadIdx.x; i < Q_BUFFERS * TM * strideA; i += CTA)
        qbuf0[i] = __float2bfloat16(0.f);

    // Load the K slab once, transposed into d-major smem: smem_b[d*strideB + pos].
    for (int i = threadIdx.x; i < TN * idxHeadDim; i += CTA) {
        int pos = i / idxHeadDim, d = i % idxHeadDim;
        int gpos = tileStart + pos;
        __nv_bfloat16 v = __float2bfloat16(0.f);
        if (gpos < kvLen) {
            size_t rowId;
            if constexpr (FLAT) {
                rowId = (size_t)flatStart + gpos;
            } else {
                int pageId = pageIndices[pageStart + gpos / pageSize];
                rowId = (size_t)pageId * pageSize + gpos % pageSize;
            }
            const uint8_t* row = kData + rowId * idxHeadDim;
            const float rowScale = kScaleData[rowId];
            v = __float2bfloat16(fp8_e4m3_to_float(row[d]) * rowScale);
        }
        smem_b[(size_t)d * strideB + pos] = v;
    }
    // Load weights for this query tile (padded rows -> 0 so they fold to nothing).
    for (int i = threadIdx.x; i < TM * idxNHeads; i += CTA) {
        int ql = i / idxNHeads, h = i % idxNHeads;
        w_s[i] = (ql < m_valid) ? weights[(size_t)(qStart + ql) * idxNHeads + h]
                                : __float2bfloat16(0.f);
    }

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_col = warp * (MMA_N * NPW);   // this warp's base column in the tile
    const int dsteps = idxHeadDim / MMA_K;

    FragC acc[NUM_M][NPW];
    #pragma unroll
    for (int mi = 0; mi < NUM_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < NPW; ni++)
            #pragma unroll
            for (int r = 0; r < 4; r++) acc[mi][ni].reg[r] = 0.f;

    // Stage a head's Q_h [TM, idxHeadDim] into dst via cp.async (16B = 8 bf16).
    auto issue_q = [&](int h, __nv_bfloat16* dst) {
        for (int e = threadIdx.x * 8; e < TM * idxHeadDim; e += CTA * 8) {
            int ql = e / idxHeadDim, d = e % idxHeadDim;
            const __nv_bfloat16* src = q + (size_t)(qStart + ql) * idxNHeads * idxHeadDim
                                       + (size_t)h * idxHeadDim + d;
            cp_async16(dst + (size_t)ql * strideA + d, src, ql < m_valid);
        }
    };

    auto compute_head = [&](int h, const __nv_bfloat16* qa) {
        FragC c[NUM_M][NPW];
        #pragma unroll
        for (int mi = 0; mi < NUM_M; mi++)
            #pragma unroll
            for (int ni = 0; ni < NPW; ni++)
                #pragma unroll
                for (int r = 0; r < 4; r++) c[mi][ni].reg[r] = 0.f;

        for (int ds = 0; ds < dsteps; ds++) {
            FragA a[NUM_M];
            #pragma unroll
            for (int mi = 0; mi < NUM_M; mi++)
                ldm_a(a[mi], qa + ds * MMA_K, strideA, mi * MMA_M);
            #pragma unroll
            for (int ni = 0; ni < NPW; ni++) {
                FragB b;
                ldm_b(b, smem_b + (size_t)ds * MMA_K * strideB, strideB, warp_col + ni * MMA_N);
                #pragma unroll
                for (int mi = 0; mi < NUM_M; mi++)
                    mma_m16n8k16(c[mi][ni], a[mi], b);
            }
        }

        // Fold this head: acc += w[q,h] * ReLU(scale * S_h). reg{0,1} -> row
        // group, reg{2,3} -> row group+8 (m16n8 C layout).
        const int group = lane >> 2;
        #pragma unroll
        for (int mi = 0; mi < NUM_M; mi++) {
            float w0 = __bfloat162float(w_s[(mi * MMA_M + group) * idxNHeads + h]);
            float w1 = __bfloat162float(w_s[(mi * MMA_M + group + 8) * idxNHeads + h]);
            #pragma unroll
            for (int ni = 0; ni < NPW; ni++) {
                acc[mi][ni].reg[0] += w0 * fmaxf(scale * c[mi][ni].reg[0], 0.f);
                acc[mi][ni].reg[1] += w0 * fmaxf(scale * c[mi][ni].reg[1], 0.f);
                acc[mi][ni].reg[2] += w1 * fmaxf(scale * c[mi][ni].reg[2], 0.f);
                acc[mi][ni].reg[3] += w1 * fmaxf(scale * c[mi][ni].reg[3], 0.f);
            }
        }
    };

    if constexpr (Q_BUFFERS == 2) {
        // Prefetch head h+1 while head h computes.
        issue_q(0, qbuf[0]); cp_commit();
        for (int h = 0; h < idxNHeads; h++) {
            if (h + 1 < idxNHeads) {
                issue_q(h + 1, qbuf[(h + 1) & 1]);
                cp_commit();
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncthreads();
            compute_head(h, qbuf[h & 1]);
            __syncthreads();
        }
    } else {
        // A single Q buffer lowers shared-memory use at the cost of serializing
        // each head's Q load with its MMA work.
        for (int h = 0; h < idxNHeads; h++) {
            issue_q(h, qbuf[0]);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
            compute_head(h, qbuf[0]);
            __syncthreads();
        }
    }

    // Epilogue: write scores with per-element causal / mask / bounds checks.
    const int group = lane >> 2;
    const int colb  = (lane & 3) * 2;
    auto write_one = [&](int ql, int pos_in_tile, float val) {
        if (ql >= m_valid) return;
        int gpos = tileStart + pos_in_tile;
        if (gpos >= kvLen) return;
        int qg = qStart + ql;
        // Same limit rowLen was written from -- see s_causalLimit above.
        if (gpos > s_causalLimit[ql]) return;
        __nv_bfloat16 out =
            (mask_ptr && idx_is_masked(cp_remap(gpos, cpMaskW, cpRank),
                                       qg - qoStart + qGlobalStart,
                                       mask_ptr, mask_kv_len_val, mask_prefix_len))
                ? __float2bfloat16(-INFINITY)
                : __float2bfloat16(val);
        scores[(size_t)qg * maxKv + gpos] = out;
    };
    #pragma unroll
    for (int mi = 0; mi < NUM_M; mi++) {
        int ql0 = mi * MMA_M + group;
        int ql1 = ql0 + 8;
        #pragma unroll
        for (int ni = 0; ni < NPW; ni++) {
            int pos_base = warp_col + ni * MMA_N + colb;
            write_one(ql0, pos_base,     acc[mi][ni].reg[0]);
            write_one(ql0, pos_base + 1, acc[mi][ni].reg[1]);
            write_one(ql1, pos_base,     acc[mi][ni].reg[2]);
            write_one(ql1, pos_base + 1, acc[mi][ni].reg[3]);
        }
    }
}

// Production-shape prefill scorer using SM120 block-scaled FP8 MMA. Q is
// quantized once into caller scratch; each CTA then reuses one packed K tile
// across a query tile and all 32 heads. Q_BATCH=2 preserves the pipelined
// single-head staging path; larger batches keep several heads resident and
// amortize block-wide synchronization across them.
template <int TM, int TN, int WARPS, int Q_BATCH, bool FLAT, bool TMA_SWIZZLE = false>
__global__ void __launch_bounds__(WARPS * 32)
idx_prefill_score_fp8_mma_kernel(
    __grid_constant__ CUtensorMap const kTensorMap,
    __nv_bfloat16* __restrict__ scores, int32_t* __restrict__ rowLen,
    const uint8_t* __restrict__ q8Data, int q8Stride,
    const float* __restrict__ effectiveWeights, int weightStride,
    const uint8_t* __restrict__ kData,
    const float* __restrict__ kScaleData,
    const int32_t* __restrict__ pageIndices, const int32_t* __restrict__ pageIndptr,
    const int32_t* __restrict__ lastPageLen, const int32_t* __restrict__ qoIndptr,
    int totalQ, int pageSize, int maxKv, int causal, int qGlobalStart,
    const uint8_t* __restrict__ custom_mask,
    const int32_t* __restrict__ mask_indptr, const int32_t* __restrict__ mask_kv_len,
    int cpWorldSize, int cpRank, const int32_t* __restrict__ globalLastPageLen,
    const int32_t* __restrict__ kvTokenIndptr)
{
    constexpr int CTA = WARPS * 32;
    constexpr int NUM_M = TM / 16;
    constexpr int NPW = TN / (idxfp8::K_ROWS * WARPS);
    static_assert(TM % 16 == 0);
    static_assert(TN % (idxfp8::K_ROWS * WARPS) == 0);
    static_assert(Q_BATCH >= 2 && idxfp8::NHEADS % Q_BATCH == 0);
    static_assert(!TMA_SWIZZLE || (FLAT && TN == 256));

    const int gy = blockIdx.y;
    int seq = -1, qStart = 0;
    {
        int acc = 0;
        for (int s = 0; ; s++) {
            const int qs = max(qoIndptr[s], qGlobalStart);
            const int qe = min(qoIndptr[s + 1], qGlobalStart + totalQ);
            const int nt = (max(0, qe - qs) + TM - 1) / TM;
            if (gy < acc + nt) {
                seq = s;
                qStart = qs - qGlobalStart + (gy - acc) * TM;
                break;
            }
            acc += nt;
            if (qoIndptr[s + 1] >= qGlobalStart + totalQ) {
                break;
            }
        }
    }
    if (seq < 0 || qStart >= totalQ) {
        return;
    }

    const int qoStart = qoIndptr[seq];
    const int numQueries = qoIndptr[seq + 1] - qoStart;
    const int mValid = min(TM, min(qoIndptr[seq + 1] - qGlobalStart, totalQ) - qStart);
    const int pageStart = FLAT ? 0 : pageIndptr[seq];
    const int numPages = FLAT ? 0 : pageIndptr[seq + 1] - pageStart;
    const int flatStart = FLAT ? kvTokenIndptr[seq] : 0;
    const int kvLen = FLAT
        ? kvTokenIndptr[seq + 1] - flatStart
        : (numPages ? (numPages - 1) * pageSize + lastPageLen[seq] : 0);
    const int cpW = (cpWorldSize > 1 && globalLastPageLen) ? cpWorldSize : 1;
    const int globalKvLen = cpW > 1
        ? (numPages ? (numPages - 1) * pageSize * cpW + globalLastPageLen[seq] : 0)
        : kvLen;
    const int tileStart = blockIdx.x * TN;
    if (tileStart >= kvLen) return;

    __shared__ int causalLimit[TM];
    for (int ql = threadIdx.x; ql < mValid; ql += CTA) {
        causalLimit[ql] = idx_local_causal_limit(
            qStart + ql - qoStart + qGlobalStart, numQueries, causal,
            kvLen, globalKvLen, cpW, cpRank);
    }
    __syncthreads();
    if (tileStart > causalLimit[mValid - 1]) return;

    const uint8_t* mask = nullptr;
    int maskKv = 0, maskPrefix = 0;
    if (custom_mask && mask_indptr) {
        mask = custom_mask + mask_indptr[seq];
        maskKv = mask_kv_len ? mask_kv_len[seq] : numQueries;
        maskPrefix = max(0, globalKvLen - maskKv);
    }
    const int cpMaskW = cpW > 1 ? cpW : 0;
    if (blockIdx.x == 0) {
        for (int ql = threadIdx.x; ql < mValid; ql += CTA)
            rowLen[qStart + ql] = causalLimit[ql] + 1;
    }

    extern __shared__ __align__(1024) char smem[];
    uint8_t* sk = reinterpret_cast<uint8_t*>(smem);
    float* skScale = reinterpret_cast<float*>(sk + TN * idxfp8::HD);
    uint8_t* sq0 = reinterpret_cast<uint8_t*>(skScale + TN);
    float* sharedWeights = reinterpret_cast<float*>(
        sq0 + Q_BATCH * TM * idxfp8::HD);
    uint64_t* kBarrier = reinterpret_cast<uint64_t*>(
        sharedWeights + TM * idxfp8::NHEADS);

    if constexpr (TMA_SWIZZLE) {
        if (threadIdx.x == 0) {
            idxfp8::mbarrier_init(kBarrier);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            idxfp8::mbarrier_arrive_expect_tx(kBarrier, TN * idxfp8::HD);
            idxfp8::tma_load_2d(sk, kTensorMap, 0, flatStart + tileStart, kBarrier);
        }
        for (int n = threadIdx.x; n < TN; n += CTA) {
            const int pos = tileStart + n;
            skScale[n] = pos < kvLen ? kScaleData[flatStart + pos] : 0.f;
        }
    } else {
        for (int i = threadIdx.x * 4; i < TN * idxfp8::HD; i += CTA * 4) {
            const int n = i / idxfp8::HD, col = i % idxfp8::HD;
            const int pos = tileStart + n;
            uint32_t value = 0;
            float rowScale = 0.f;
            if (pos < kvLen) {
                size_t rowId;
                if constexpr (FLAT) {
                    rowId = (size_t)flatStart + pos;
                } else {
                    const int pageId = pageIndices[pageStart + pos / pageSize];
                    rowId = (size_t)pageId * pageSize + pos % pageSize;
                }
                const uint8_t* row = kData + rowId * idxfp8::HD;
                value = *reinterpret_cast<const uint32_t*>(row + col);
                if (col == 0) rowScale = kScaleData[rowId];
            }
            *reinterpret_cast<uint32_t*>(sk + i) = value;
            if (col == 0) skScale[n] = rowScale;
        }
    }
    for (int i = threadIdx.x; i < TM * idxfp8::NHEADS; i += CTA) {
        const int ql = i / idxfp8::NHEADS, h = i % idxfp8::NHEADS;
        sharedWeights[i] = ql < mValid
            ? effectiveWeights[(size_t)(qStart + ql) * weightStride + h] : 0.f;
    }
    if constexpr (TMA_SWIZZLE) {
        if (threadIdx.x == 0) idxfp8::mbarrier_wait(kBarrier);
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpCol = warp * idxfp8::K_ROWS * NPW;

    for (int i = threadIdx.x; i < Q_BATCH * TM * idxfp8::HD; i += CTA) sq0[i] = 0;
    __syncthreads();

    idxfp8::Acc acc[NUM_M][NPW];
#pragma unroll
    for (int mi = 0; mi < NUM_M; mi++)
#pragma unroll
        for (int ni = 0; ni < NPW; ni++)
            acc[mi][ni] = {0.f, 0.f, 0.f, 0.f};

    auto issueQ = [&](int h, uint8_t* dst) {
        for (int i = threadIdx.x * 16; i < TM * idxfp8::HD; i += CTA * 16) {
            const int ql = i / idxfp8::HD, col = i % idxfp8::HD;
            const uint8_t* src = q8Data + (size_t)(qStart + ql) * q8Stride
                + h * idxfp8::HD + col;
            uint8_t* qDst = dst + i;
            if constexpr (TMA_SWIZZLE) {
                qDst = const_cast<uint8_t*>(idxfp8::smem_row_ptr<true>(dst, ql, col));
            }
            idxmma::cp_async16(qDst, src, ql < mValid);
        }
    };

    if constexpr (Q_BATCH == 2) {
        issueQ(0, sq0);
        idxmma::cp_commit();
        for (int h = 0; h < idxfp8::NHEADS; h++) {
            if (h + 1 < idxfp8::NHEADS) {
                issueQ(h + 1, sq0 + ((h + 1) & 1) * TM * idxfp8::HD);
                idxmma::cp_commit();
                idxmma::cp_wait<1>();
            } else {
                idxmma::cp_wait<0>();
            }
            __syncthreads();
            const uint8_t* qHead = sq0 + (h & 1) * TM * idxfp8::HD;

            idxfp8::Acc c[NUM_M][NPW];
#pragma unroll
            for (int mi = 0; mi < NUM_M; mi++)
#pragma unroll
                for (int ni = 0; ni < NPW; ni++)
                    c[mi][ni] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll
            for (int kk = 0; kk < idxfp8::HD; kk += 32) {
                uint32_t a0[NUM_M], a1[NUM_M], a2[NUM_M], a3[NUM_M];
#pragma unroll
                for (int mi = 0; mi < NUM_M; mi++)
                    idxfp8::load_a<TMA_SWIZZLE>(a0[mi], a1[mi], a2[mi], a3[mi],
                                   qHead + mi * 16 * idxfp8::HD, lane, kk);
#pragma unroll
                for (int ni = 0; ni < NPW; ni++) {
                    uint32_t b0, b1;
                    idxfp8::load_b<TMA_SWIZZLE>(b0, b1,
                        sk + (warpCol + ni * idxfp8::K_ROWS) * idxfp8::HD, lane, kk);
#pragma unroll
                    for (int mi = 0; mi < NUM_M; mi++)
                        c[mi][ni] = idxfp8::mma(a0[mi], a1[mi], a2[mi], a3[mi],
                                                b0, b1, c[mi][ni]);
                }
            }

            const int group = lane >> 2;
#pragma unroll
            for (int mi = 0; mi < NUM_M; mi++) {
                const int q0 = mi * 16 + group;
                const int q1 = q0 + 8;
                const float w0 = q0 < mValid
                    ? sharedWeights[q0 * idxfp8::NHEADS + h] : 0.f;
                const float w1 = q1 < mValid
                    ? sharedWeights[q1 * idxfp8::NHEADS + h] : 0.f;
#pragma unroll
                for (int ni = 0; ni < NPW; ni++) {
                    acc[mi][ni].x0 += w0 * fmaxf(c[mi][ni].x0, 0.f);
                    acc[mi][ni].x1 += w0 * fmaxf(c[mi][ni].x1, 0.f);
                    acc[mi][ni].x2 += w1 * fmaxf(c[mi][ni].x2, 0.f);
                    acc[mi][ni].x3 += w1 * fmaxf(c[mi][ni].x3, 0.f);
                }
            }
            __syncthreads();
        }
    } else {
        for (int hBase = 0; hBase < idxfp8::NHEADS; hBase += Q_BATCH) {
#pragma unroll
            for (int hi = 0; hi < Q_BATCH; hi++)
                issueQ(hBase + hi, sq0 + hi * TM * idxfp8::HD);
            idxmma::cp_commit();
            idxmma::cp_wait<0>();
            __syncthreads();

#pragma unroll
            for (int hi = 0; hi < Q_BATCH; hi++) {
                const int h = hBase + hi;
                const uint8_t* qHead = sq0 + hi * TM * idxfp8::HD;
                idxfp8::Acc c[NUM_M][NPW];
#pragma unroll
                for (int mi = 0; mi < NUM_M; mi++)
#pragma unroll
                    for (int ni = 0; ni < NPW; ni++)
                        c[mi][ni] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll
                for (int kk = 0; kk < idxfp8::HD; kk += 32) {
                    uint32_t a0[NUM_M], a1[NUM_M], a2[NUM_M], a3[NUM_M];
#pragma unroll
                    for (int mi = 0; mi < NUM_M; mi++)
                        idxfp8::load_a<TMA_SWIZZLE>(a0[mi], a1[mi], a2[mi], a3[mi],
                                       qHead + mi * 16 * idxfp8::HD, lane, kk);
#pragma unroll
                    for (int ni = 0; ni < NPW; ni++) {
                        uint32_t b0, b1;
                        idxfp8::load_b<TMA_SWIZZLE>(b0, b1,
                            sk + (warpCol + ni * idxfp8::K_ROWS) * idxfp8::HD,
                            lane, kk);
#pragma unroll
                        for (int mi = 0; mi < NUM_M; mi++)
                            c[mi][ni] = idxfp8::mma(a0[mi], a1[mi], a2[mi], a3[mi],
                                                    b0, b1, c[mi][ni]);
                    }
                }

                const int group = lane >> 2;
#pragma unroll
                for (int mi = 0; mi < NUM_M; mi++) {
                    const int q0 = mi * 16 + group;
                    const int q1 = q0 + 8;
                    const float w0 = q0 < mValid
                        ? sharedWeights[q0 * idxfp8::NHEADS + h] : 0.f;
                    const float w1 = q1 < mValid
                        ? sharedWeights[q1 * idxfp8::NHEADS + h] : 0.f;
#pragma unroll
                    for (int ni = 0; ni < NPW; ni++) {
                        acc[mi][ni].x0 += w0 * fmaxf(c[mi][ni].x0, 0.f);
                        acc[mi][ni].x1 += w0 * fmaxf(c[mi][ni].x1, 0.f);
                        acc[mi][ni].x2 += w1 * fmaxf(c[mi][ni].x2, 0.f);
                        acc[mi][ni].x3 += w1 * fmaxf(c[mi][ni].x3, 0.f);
                    }
                }
            }
            __syncthreads();
        }
    }

    const int group = lane >> 2;
    const int colBase = (lane & 3) * 2;
    auto writeOne = [&](int ql, int posInTile, float value) {
        if (ql >= mValid) return;
        const int pos = tileStart + posInTile;
        if (pos >= kvLen || pos > causalLimit[ql]) return;
        const int qg = qStart + ql;
        const bool masked = mask && idx_is_masked(
            cp_remap(pos, cpMaskW, cpRank), qg - qoStart + qGlobalStart,
            mask, maskKv, maskPrefix);
        scores[(size_t)qg * maxKv + pos] = masked
            ? __float2bfloat16(-INFINITY)
            : __float2bfloat16(value * skScale[posInTile]);
    };
#pragma unroll
    for (int mi = 0; mi < NUM_M; mi++) {
        const int q0 = mi * 16 + group;
        const int q1 = q0 + 8;
#pragma unroll
        for (int ni = 0; ni < NPW; ni++) {
            const int pos = warpCol + ni * idxfp8::K_ROWS + colBase;
            writeOne(q0, pos, acc[mi][ni].x0);
            writeOne(q0, pos + 1, acc[mi][ni].x1);
            writeOne(q1, pos, acc[mi][ni].x2);
            writeOne(q1, pos + 1, acc[mi][ni].x3);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Coarse histogram from score buffer
// Grid: totalQ  Block: 256.  One block owns a whole row: it accumulates into a
// shared 1024-bucket histogram (fast shared atomics, no cross-block contention)
// and writes it out directly — the row's sole writer, so no global atomics.
// ---------------------------------------------------------------------------
__global__ void idx_prefill_coarse_hist_buf_kernel(
    int32_t* __restrict__ coarseHist,         // [totalQ, IDX_COARSE_BUCKETS]
    const __nv_bfloat16* __restrict__ scores, // [totalQ, maxKv]
    const int32_t* __restrict__ rowLen,       // [totalQ]
    int maxKv)
{
    const int row = blockIdx.x;
    const int len = rowLen ? rowLen[row] : maxKv;
    const __nv_bfloat16* s = scores + (size_t)row * maxKv;

    __shared__ int sh[IDX_COARSE_BUCKETS];
    for (int b = threadIdx.x; b < IDX_COARSE_BUCKETS; b += blockDim.x) sh[b] = 0;
    __syncthreads();
    for (int i = threadIdx.x * 8; i < len; i += blockDim.x * 8) {
        if (i + 7 < len && ((uintptr_t)(s + i) & 15) == 0) {
            const ulonglong2 packed = *reinterpret_cast<const ulonglong2*>(s + i);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned long long word = j < 4 ? packed.x : packed.y;
                atomicAdd(&sh[bf16_key_bits((unsigned short)(word >> ((j & 3) * 16))) / IDX_FINE_BUCKETS], 1);
            }
        } else {
            for (int j = 0; j < 8 && i + j < len; j++)
                atomicAdd(&sh[bf16_key(&s[i + j]) / IDX_FINE_BUCKETS], 1);
        }
    }
    __syncthreads();
    int32_t* h = coarseHist + (size_t)row * IDX_COARSE_BUCKETS;
    for (int b = threadIdx.x; b < IDX_COARSE_BUCKETS; b += blockDim.x) h[b] = sh[b];
}

// ---------------------------------------------------------------------------
// Pass 3: Coarse threshold
// Grid: totalQ  Block: 256
// Scans 1024 coarse buckets from highest to lowest, finds the winning bucket
// where cumulative count >= topk. Also initializes out_idx to -1 and handles
// the identity case (total count <= topk).
// ---------------------------------------------------------------------------
__global__ void idx_prefill_coarse_threshold_kernel(
    const int32_t* __restrict__ coarseHist, int32_t* __restrict__ meta,
    int32_t* __restrict__ out_idx, __nv_bfloat16* __restrict__ out_scores, int topk,
    const __nv_bfloat16* __restrict__ scores, const int32_t* __restrict__ rowLen, int maxKv,
    int cpWorldSize, int cpRank)
{
    const int row = blockIdx.x;
    const int32_t* h = coarseHist + (size_t)row * IDX_COARSE_BUCKETS;
    const __nv_bfloat16 neg_inf = __float2bfloat16(-INFINITY);

    // Initialize output to -1 / -inf
    for (int i = threadIdx.x; i < topk; i += blockDim.x) {
        out_idx[(size_t)row * topk + i] = -1;
        out_scores[(size_t)row * topk + i] = neg_inf;
    }

    // Sum all buckets to get total count
    const int t = threadIdx.x;
    const int PER = IDX_COARSE_BUCKETS / 256;  // 4 buckets per thread
    long localSum = 0;
    for (int k = t * PER; k < (t + 1) * PER; k++) localSum += h[k];

    __shared__ long partial[256];
    __shared__ long s_total;
    partial[t] = localSum;
    __syncthreads();

    if (t == 0) {
        long total = 0;
        for (int j = 0; j < 256; j++) total += partial[j];
        s_total = total;
    }
    __syncthreads();

    long total = s_total;

    // Identity case: all positions fit in topk. Write position-preserving
    // values directly (deterministic), skipping any position whose score is
    // -inf (masked out by a custom tree mask). tau=-1 signals the gather to
    // skip (identity already handled).
    if (total <= topk) {
        const __nv_bfloat16* s = scores + (size_t)row * maxKv;
        const int len = rowLen ? rowLen[row] : maxKv;
        for (int i = threadIdx.x; i < topk; i += blockDim.x) {
            bool valid = (i < len && *reinterpret_cast<const unsigned short*>(&s[i]) != 0xFF80);
            out_idx[(size_t)row * topk + i] = valid ? cp_remap(i, cpWorldSize, cpRank) : -1;
            out_scores[(size_t)row * topk + i] = valid ? s[i] : neg_inf;
        }
        if (t == 0) {
            int32_t* m = meta + (size_t)row * 4;
            m[0] = -1;  // identity sentinel — gather is a no-op
            m[1] = 0;
            m[2] = 0;
            m[3] = 0;
        }
        return;
    }

    // Find winning coarse bucket: scan from highest (bucket 1023) to lowest (0)
    __shared__ int s_winner;
    __shared__ long s_above;
    if (t == 0) {
        long cum = 0;
        int winner = -1;
        for (int j = 255; j >= 0; j--) {
            if (cum + partial[j] >= topk) { winner = j; break; }
            cum += partial[j];
        }
        s_winner = winner;
        s_above = cum;
    }
    __syncthreads();

    int winner = s_winner;
    long above = s_above;

    // Refine within winning thread's band
    if (t == winner) {
        long cum = above;
        long numAbove = above;
        int tau = t * PER;
        for (int k = (t + 1) * PER - 1; k >= t * PER; k--) {
            long c = h[k];
            if (cum + c >= topk) { tau = k; numAbove = cum; break; }
            cum += c;
        }
        int32_t* m = meta + (size_t)row * 4;
        m[0] = tau;                    // winning coarse bucket
        m[1] = (int)numAbove;          // count in coarse buckets above winner
        m[2] = 0;
        m[3] = 0;
    }
}

// ---------------------------------------------------------------------------
// Pass 4: Fine histogram from score buffer
// Grid: totalQ  Block: 256.  One block per row (see coarse pass): shared 64-bucket
// histogram over positions in the winning coarse bucket, direct global write.
// ---------------------------------------------------------------------------
__global__ void idx_prefill_fine_hist_buf_kernel(
    int32_t* __restrict__ fineHist,           // [totalQ, IDX_FINE_BUCKETS]
    const int32_t* __restrict__ meta,         // [totalQ, 4] — reads coarse_tau
    const __nv_bfloat16* __restrict__ scores, // [totalQ, maxKv]
    const int32_t* __restrict__ rowLen,       // [totalQ]
    int maxKv)
{
    const int row = blockIdx.x;
    int32_t coarseTau = meta[(size_t)row * 4];
    if (coarseTau < 0) return;  // identity case — skip

    const int len = rowLen ? rowLen[row] : maxKv;
    const __nv_bfloat16* s = scores + (size_t)row * maxKv;

    __shared__ int sh[IDX_FINE_BUCKETS];
    for (int b = threadIdx.x; b < IDX_FINE_BUCKETS; b += blockDim.x) sh[b] = 0;
    __syncthreads();
    for (int i = threadIdx.x * 8; i < len; i += blockDim.x * 8) {
        if (i + 7 < len && ((uintptr_t)(s + i) & 15) == 0) {
            const ulonglong2 packed = *reinterpret_cast<const ulonglong2*>(s + i);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned long long word = j < 4 ? packed.x : packed.y;
                int key = bf16_key_bits((unsigned short)(word >> ((j & 3) * 16)));
                if (key / IDX_FINE_BUCKETS == coarseTau)
                    atomicAdd(&sh[key % IDX_FINE_BUCKETS], 1);
            }
        } else {
            for (int j = 0; j < 8 && i + j < len; j++) {
                int key = bf16_key(&s[i + j]);
                if (key / IDX_FINE_BUCKETS == coarseTau)
                    atomicAdd(&sh[key % IDX_FINE_BUCKETS], 1);
            }
        }
    }
    __syncthreads();
    int32_t* h = fineHist + (size_t)row * IDX_FINE_BUCKETS;
    for (int b = threadIdx.x; b < IDX_FINE_BUCKETS; b += blockDim.x) h[b] = sh[b];
}

// ---------------------------------------------------------------------------
// Pass 5: Fine threshold
// Grid: totalQ  Block: 256
// Scans 64 fine buckets within the winning coarse bucket, combines with
// num_above_coarse to find the exact threshold (full 16-bit bf16_key).
// ---------------------------------------------------------------------------
__global__ void idx_prefill_fine_threshold_kernel(
    const int32_t* __restrict__ fineHist, int32_t* __restrict__ meta, int topk)
{
    const int row = blockIdx.x;
    int32_t* m = meta + (size_t)row * 4;
    int coarseTau = m[0];
    if (coarseTau < 0) return;  // identity — pass through

    long numAboveCoarse = m[1];
    const int32_t* fh = fineHist + (size_t)row * IDX_FINE_BUCKETS;

    if (threadIdx.x == 0) {
        long cum = numAboveCoarse;
        long numAbove = numAboveCoarse;
        int fineTau = 0;
        for (int k = IDX_FINE_BUCKETS - 1; k >= 0; k--) {
            long c = fh[k];
            if (cum + c >= topk) { fineTau = k; numAbove = cum; break; }
            cum += c;
        }
        m[0] = coarseTau * IDX_FINE_BUCKETS + fineTau;
        m[1] = (int)(topk - numAbove);  // tie_take
        m[2] = 0;                        // out_count (for gather)
        m[3] = 0;                        // tie_count (for gather)
    }
}

// ---------------------------------------------------------------------------
// Pass 6: Gather from score buffer
// Grid: totalQ  Block: 256.  One block per row, so the ordered compaction needs
// no cross-block prefix: running offsets live in registers across tiles. Writes
// positions with key > tau into out[0, numAbove) and the FIRST tie_take entries
// at key == tau into out[numAbove, topk), both in ascending position order, so
// the result is a pure function of the scores (see idx_gather_count_kernel for
// why that matters). Identity (tau<0) was already written by the threshold pass.
//
// Slots the selection does not reach keep the -1 / -inf prefill that
// idx_prefill_coarse_threshold_kernel writes over the whole row before the fine
// passes run. The sparse-MLA kernel clamps negative indices to page 0 at load
// and masks them out in QK, so -1 is the required sentinel; without the prefill
// the tail would hold stale indices and the KV gather would read out of bounds.
// ---------------------------------------------------------------------------
__global__ void idx_prefill_gather_buf_kernel(
    int32_t* __restrict__ out_idx,           // [totalQ, topk]
    __nv_bfloat16* __restrict__ out_scores,  // [totalQ, topk]
    const int32_t* __restrict__ meta,        // [totalQ, 4]
    const __nv_bfloat16* __restrict__ scores,// [totalQ, maxKv]
    const int32_t* __restrict__ rowLen,      // [totalQ]
    int maxKv, int topk, int cpWorldSize, int cpRank)
{
    const int row = blockIdx.x;
    const int32_t* m = meta + (size_t)row * 4;
    const int tau = m[0];
    if (tau < 0) return;   // block-uniform; threshold kernel already wrote the row
    const int tieTake = m[1];
    const int aboveTotal = topk - tieTake;

    const int len = rowLen ? rowLen[row] : maxKv;
    const __nv_bfloat16* s = scores + (size_t)row * maxKv;
    int32_t* out = out_idx + (size_t)row * topk;
    __nv_bfloat16* out_s = out_scores + (size_t)row * topk;

    __shared__ int s_warp[2 * IDX_GATHER_WARPS];
    int aboveOff = 0, tieOff = 0;
    // The tile loop is block-uniform (threads past len carry take = false) so
    // the ballots and __syncthreads() inside idx_ordered_slot_pair see the
    // whole block.
    for (int base = 0; base < len; base += blockDim.x) {
        const int i = base + threadIdx.x;
        const bool inRange = i < len;
        const int key = inRange ? bf16_key(&s[i]) : -1;   // -1 matches no tau
        const bool isAbove = inRange && key > tau;
        const bool isTie = inRange && key == tau;
        int aSlot, tRank, aTotal, tTotal;

        idx_ordered_slot_pair(isAbove, aboveOff, isTie, tieOff,
                              s_warp, &aSlot, &tRank, &aTotal, &tTotal);
        if (isAbove) {
            if (aSlot < aboveTotal) {
                out[aSlot] = cp_remap(i, cpWorldSize, cpRank);
                out_s[aSlot] = s[i];
            }
        } else if (isTie) {
            if (tRank < tieTake) {
                out[aboveTotal + tRank] = cp_remap(i, cpWorldSize, cpRank);
                out_s[aboveTotal + tRank] = s[i];
            }
        }
        aboveOff += aTotal;
        tieOff += tTotal;
    }
}

// Faster prefill compaction when stable ordering and first-position tie
// selection are not required. CTAs own contiguous row slices and reserve output
// ranges a warp at a time, avoiding the block-wide barriers and serial full-row
// walk in idx_prefill_gather_buf_kernel. The selected score multiset is exact,
// but output order and the chosen subset of threshold ties are unspecified.
__global__ void idx_prefill_gather_atomic_buf_kernel(
    int32_t* __restrict__ out_idx,           // [totalQ, topk]
    __nv_bfloat16* __restrict__ out_scores,  // [totalQ, topk]
    int32_t* __restrict__ meta,              // [totalQ, 4]
    const __nv_bfloat16* __restrict__ scores,// [totalQ, maxKv]
    const int32_t* __restrict__ rowLen,      // [totalQ]
    int maxKv, int topk, int cpWorldSize, int cpRank)
{
    const int row = blockIdx.y;
    int32_t* m = meta + (size_t)row * 4;
    const int tau = m[0];
    if (tau < 0) return;  // threshold kernel already wrote the identity row

    const int tieTake = m[1];
    const int aboveTotal = topk - tieTake;
    const int len = rowLen ? rowLen[row] : maxKv;
    const int chunk = (len + (int)gridDim.x - 1) / (int)gridDim.x;
    const int lo = min((int)blockIdx.x * chunk, len);
    const int hi = min(lo + chunk, len);
    const __nv_bfloat16* s = scores + (size_t)row * maxKv;
    int32_t* out = out_idx + (size_t)row * topk;
    __nv_bfloat16* out_s = out_scores + (size_t)row * topk;
    const int lane = threadIdx.x & 31;

    for (int base = lo; base < hi; base += blockDim.x) {
        const int i = base + threadIdx.x;
        const bool inRange = i < hi;
        const int key = inRange ? bf16_key(&s[i]) : -1;

        const unsigned aboveVote = __ballot_sync(0xffffffffu, inRange && key > tau);
        const int aboveCount = __popc(aboveVote);
        int aboveBase = 0;
        if (lane == 0 && aboveCount)
            aboveBase = atomicAdd(m + 2, aboveCount);
        aboveBase = __shfl_sync(0xffffffffu, aboveBase, 0);
        if (aboveVote & (1u << lane)) {
            const int slot = aboveBase + __popc(aboveVote & ((1u << lane) - 1));
            if (slot < aboveTotal) {
                out[slot] = cp_remap(i, cpWorldSize, cpRank);
                out_s[slot] = s[i];
            }
        }

        const unsigned tieVote = __ballot_sync(0xffffffffu, inRange && key == tau);
        const int tieCount = __popc(tieVote);
        int tieBase = 0;
        if (lane == 0 && tieCount)
            tieBase = atomicAdd(m + 3, tieCount);
        tieBase = __shfl_sync(0xffffffffu, tieBase, 0);
        if (tieVote & (1u << lane)) {
            const int slot = tieBase + __popc(tieVote & ((1u << lane) - 1));
            if (slot < tieTake) {
                out[aboveTotal + slot] = cp_remap(i, cpWorldSize, cpRank);
                out_s[aboveTotal + slot] = s[i];
            }
        }
    }
}

static bool make_indexer_k_tma_map(
    CUtensorMap* map, const void* kData, uint64_t rows) {
    static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
        void* entry = nullptr;
        cudaDriverEntryPointQueryResult queryResult;
        cudaError_t status = cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled", &entry, 12000,
            cudaEnableDefault, &queryResult);
        if (status != cudaSuccess || queryResult != cudaDriverEntryPointSuccess) {
            return static_cast<PFN_cuTensorMapEncodeTiled_v12000>(nullptr);
        }
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(entry);
    }();
    if (!encode || rows == 0) return false;

    constexpr uint32_t rank = 2;
    const uint64_t globalDims[rank] = {idxfp8::HD, rows};
    const uint64_t globalStrides[rank - 1] = {idxfp8::HD};
    const uint32_t boxDims[rank] = {idxfp8::HD, 256};
    const uint32_t elementStrides[rank] = {1, 1};
    return encode(
        map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank,
        const_cast<void*>(kData), globalDims, globalStrides,
        boxDims, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// Host function: score once into buffer, then 2-level histogram top-K.
void glm_indexer_score_topk_prefill(GlmCtx* ctx, int32_t* out_idx,
    __nv_bfloat16* out_scores,
    const void* q, const void* kData, const float* kScaleData, const void* weights,
    const int32_t* pageIndices, const int32_t* pageIndptr,
    const int32_t* lastPageLen, const int32_t* qoIndptr,
    float scale, int totalQ, int idxNHeads, int idxHeadDim,
    int pageSize, int topk, int causal, int qGlobalStart,
    const uint8_t* custom_mask, const int32_t* mask_indptr, const int32_t* mask_kv_len,
    void* scores, int32_t* rowLen, int maxKv,
    int32_t* coarseHist, int32_t* fineHist, int32_t* meta,
    int queryTiles, int cpWorldSize, int cpRank,
    const int32_t* globalLastPageLen, const int32_t* kvTokenIndptr,
    const float* precomputed_ew) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    const int effectiveCpWorldSize = kvTokenIndptr ? 0 : cpWorldSize;
    const int effectiveCpRank = kvTokenIndptr ? 0 : cpRank;

    // Pass 1: tensor-core score into buffer. queryTiles is a graph-stable upper
    // bound for sum(ceil(sequenceQ/TM)); out-of-range tiles return immediately.
    // precomputed_ew: q is already-quantized FP8 [totalQ, idxNHeads, idxHeadDim]
    // with a matching [totalQ, idxNHeads] F32 effective-weights tensor; the
    // internal quantize launch is skipped and the histogram scratch stays
    // untouched by the scoring pass.
    const bool precomputed = precomputed_ew != nullptr;
    if (precomputed && !indexer_use_fp8_mma(idxNHeads, idxHeadDim)) {
        fprintf(stderr, "glm_indexer_score_topk_prefill: precomputed_ew requires "
                        "idxNHeads=%d idxHeadDim=%d and FP8 MMA enabled (got %d/%d)\n",
                idxfp8::NHEADS, idxfp8::HD, idxNHeads, idxHeadDim);
        return;
    }
    if (indexer_use_fp8_mma(idxNHeads, idxHeadDim) || precomputed) {
        const uint8_t* q8Data;
        const float* effectiveWeights;
        int q8StrideBytes, ewStride;
        if (precomputed) {
            q8Data = reinterpret_cast<const uint8_t*>(q);
            effectiveWeights = precomputed_ew;
            q8StrideBytes = idxNHeads * idxHeadDim;
            ewStride = idxNHeads;
        } else {
            uint8_t* q8Scratch = reinterpret_cast<uint8_t*>(coarseHist);
            q8Data = q8Scratch;
            effectiveWeights = reinterpret_cast<float*>(fineHist);
            q8StrideBytes = IDX_COARSE_BUCKETS * sizeof(int32_t);
            ewStride = IDX_FINE_BUCKETS;
            idxfp8::quantize_q_kernel<idxfp8::NHEADS><<<totalQ, idxfp8::NHEADS * 8, 0, stream>>>(
                (const __nv_bfloat16*)q, (const __nv_bfloat16*)weights,
                idxfp8::NHEADS, 0,
                q8Scratch, IDX_COARSE_BUCKETS * sizeof(int32_t),
                reinterpret_cast<float*>(fineHist), IDX_FINE_BUCKETS, scale);
        }
#define LAUNCH_INDEXER_PREFILL_FP8(TM, TN, WARPS, Q_BATCH, FLAT, TMA_SWIZZLE) do { \
        const size_t fp8Smem = (TN) * idxfp8::HD + (TN) * sizeof(float) \
            + (Q_BATCH) * (TM) * idxfp8::HD \
            + (TM) * idxfp8::NHEADS * sizeof(float) + sizeof(uint64_t); \
        cudaFuncSetAttribute( \
            (void*)idx_prefill_score_fp8_mma_kernel<(TM), (TN), (WARPS), (Q_BATCH), (FLAT), (TMA_SWIZZLE)>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fp8Smem); \
        dim3 grid((maxKv + (TN) - 1) / (TN), queryTiles * (64 / (TM))); \
        idx_prefill_score_fp8_mma_kernel<(TM), (TN), (WARPS), (Q_BATCH), (FLAT), (TMA_SWIZZLE)> \
            <<<grid, (WARPS) * 32, fp8Smem, stream>>>( \
                kTensorMap, (__nv_bfloat16*)scores, rowLen, q8Data, \
                q8StrideBytes, effectiveWeights, \
                ewStride, (const uint8_t*)kData, kScaleData, pageIndices, pageIndptr, \
                lastPageLen, qoIndptr, totalQ, pageSize, maxKv, causal, \
                qGlobalStart, custom_mask, mask_indptr, mask_kv_len, \
                effectiveCpWorldSize, effectiveCpRank, globalLastPageLen, kvTokenIndptr); \
    } while (0)
        const char* fp8Config = std::getenv("GLM_INDEXER_PREFILL_FP8_CONFIG");
        const bool tmaRequested = (fp8Config
            && std::strcmp(fp8Config, "q32_k256_w8_h8_tma") == 0)
            || (!fp8Config && kvTokenIndptr);
        const bool batchedQ = tmaRequested || (fp8Config
            && std::strcmp(fp8Config, "q32_k256_w8_h8") == 0);
        CUtensorMap kTensorMap{};
        const bool useTma = tmaRequested && kvTokenIndptr
            && make_indexer_k_tma_map(&kTensorMap, kData, maxKv);
        if (useTma) {
            LAUNCH_INDEXER_PREFILL_FP8(32, 256, 8, 8, true, true);
        } else if (batchedQ) {
            if (kvTokenIndptr) LAUNCH_INDEXER_PREFILL_FP8(32, 256, 8, 8, true, false);
            else LAUNCH_INDEXER_PREFILL_FP8(32, 256, 8, 8, false, false);
        } else {
            if (kvTokenIndptr) LAUNCH_INDEXER_PREFILL_FP8(64, 288, 12, 2, true, false);
            else LAUNCH_INDEXER_PREFILL_FP8(64, 288, 12, 2, false, false);
        }
#undef LAUNCH_INDEXER_PREFILL_FP8
    } else {
#define LAUNCH_INDEXER_PREFILL(TM, TN, WARPS, Q_BUFFERS, FLAT) do { \
        constexpr int cta = (WARPS) * 32; \
        const int strideA = idxHeadDim + idxmma::PAD_A; \
        const int strideB = (TN) + idxmma::PAD_B; \
        size_t mma_smem = ((size_t)idxHeadDim * strideB \
                           + (size_t)(Q_BUFFERS) * (TM) * strideA \
                           + (size_t)(TM) * idxNHeads) * sizeof(__nv_bfloat16); \
        cudaFuncSetAttribute( \
            (void*)idx_prefill_score_mma_kernel<(TM), (TN), (WARPS), (Q_BUFFERS), (FLAT)>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)mma_smem); \
        dim3 grid((maxKv + (TN) - 1) / (TN), queryTiles); \
        idx_prefill_score_mma_kernel<(TM), (TN), (WARPS), (Q_BUFFERS), (FLAT)> \
            <<<grid, cta, mma_smem, stream>>>( \
                (__nv_bfloat16*)scores, rowLen, \
                 (const __nv_bfloat16*)q, (const uint8_t*)kData, kScaleData, \
                (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr, \
                totalQ, scale, idxNHeads, idxHeadDim, pageSize, maxKv, causal, \
                qGlobalStart, custom_mask, mask_indptr, mask_kv_len, \
                effectiveCpWorldSize, effectiveCpRank, globalLastPageLen, kvTokenIndptr); \
    } while (0)

        const char* config = std::getenv("GLM_INDEXER_PREFILL_CONFIG");
        if (!config) {
            if (kvTokenIndptr && maxKv >= 16384)
                LAUNCH_INDEXER_PREFILL(64, 288, 12, 1, true);
            else if (kvTokenIndptr)
                LAUNCH_INDEXER_PREFILL(64, 256, 16, 1, true);
            else
                LAUNCH_INDEXER_PREFILL(64, 256, 16, 1, false);
        } else if (std::strcmp(config, "q64_k256_w16_q1") == 0) {
            if (kvTokenIndptr) LAUNCH_INDEXER_PREFILL(64, 256, 16, 1, true);
            else LAUNCH_INDEXER_PREFILL(64, 256, 16, 1, false);
        } else if (std::strcmp(config, "q64_k288_w12_q1") == 0) {
            if (kvTokenIndptr) LAUNCH_INDEXER_PREFILL(64, 288, 12, 1, true);
            else LAUNCH_INDEXER_PREFILL(64, 288, 12, 1, false);
        } else if (std::strcmp(config, "q64_k192_w8_q2") == 0) {
            if (kvTokenIndptr) LAUNCH_INDEXER_PREFILL(64, 192, 8, 2, true);
            else LAUNCH_INDEXER_PREFILL(64, 192, 8, 2, false);
        }
        else {
            fprintf(stderr, "Unknown GLM_INDEXER_PREFILL_CONFIG=%s\n", config);
            if (kvTokenIndptr) LAUNCH_INDEXER_PREFILL(64, 256, 16, 1, true);
            else LAUNCH_INDEXER_PREFILL(64, 256, 16, 1, false);
        }
#undef LAUNCH_INDEXER_PREFILL
    }

    // FP8 scoring borrows the histogram buffers, but no reset is needed:
    // coarse_hist overwrites every bucket, coarse_threshold writes all four
    // metadata fields, and fine_hist overwrites every bucket of non-identity
    // rows. Identity rows skip every fine-histogram reader. All producers follow
    // scoring on this stream, so the borrowed storage is no longer in use.

    // Passes 2-6: histogram + gather from buffer.
    {
        // Pass 2: coarse histogram — one block per row
        idx_prefill_coarse_hist_buf_kernel<<<totalQ, 256, 0, stream>>>(
            coarseHist, (const __nv_bfloat16*)scores, rowLen, maxKv);

        // Pass 3: coarse threshold (1 block per query)
        idx_prefill_coarse_threshold_kernel<<<totalQ, 256, 0, stream>>>(
            coarseHist, meta, out_idx, out_scores, topk, (const __nv_bfloat16*)scores, rowLen, maxKv, effectiveCpWorldSize, effectiveCpRank);

        // Pass 4: fine histogram — one block per row
        idx_prefill_fine_hist_buf_kernel<<<totalQ, 256, 0, stream>>>(
            fineHist, meta, (const __nv_bfloat16*)scores, rowLen, maxKv);

        // Pass 5: fine threshold (1 block per query)
        idx_prefill_fine_threshold_kernel<<<totalQ, 256, 0, stream>>>(
            fineHist, meta, topk);

        // Pass 6: parallel unordered compaction by default. The ordered kernel
        // remains available for diagnostics that require repeatable tie choice.
        const char* deterministicGather = std::getenv("GLM_INDEXER_PREFILL_DETERMINISTIC");
        // A single row scan has less scheduling overhead at short contexts.
        if (maxKv < 32768 || (deterministicGather && std::strcmp(deterministicGather, "1") == 0)) {
            idx_prefill_gather_buf_kernel<<<totalQ, 256, 0, stream>>>(
                out_idx, out_scores, meta, (const __nv_bfloat16*)scores, rowLen,
                maxKv, topk, effectiveCpWorldSize, effectiveCpRank);
        } else {
            const int gatherSplits = min(32, max(1, (maxKv + 4095) / 4096));
            idx_prefill_gather_atomic_buf_kernel<<<dim3(gatherSplits, totalQ), 256, 0, stream>>>(
                out_idx, out_scores, meta, (const __nv_bfloat16*)scores, rowLen,
                maxKv, topk, effectiveCpWorldSize, effectiveCpRank);
        }
    }
}
