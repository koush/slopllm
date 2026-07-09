#include "glm_ops.h"
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Fused indexer score kernel
// Computes: out[qi, ki] = sum_h weights[qi,h] * ReLU(sum_d q[qi,h,d] * k[ki,d]) * scale)
// Reads K from paged cache. One block per query token. One warp per head.
// Invalid positions (beyond kvLen or causal limit) remain -inf.
// ---------------------------------------------------------------------------

__global__ void indexer_score_kernel(
    __nv_bfloat16* __restrict__ out,          // [totalQ, maxKvLen]
    const __nv_bfloat16* __restrict__ q,      // [totalQ, idxNHeads, idxHeadDim]
    const __nv_bfloat16* __restrict__ kData,  // [maxPages, pageSize, idxHeadDim]
    const __nv_bfloat16* __restrict__ weights,// [totalQ, idxNHeads]
    const int32_t* __restrict__ pageIndices,  // [numPages]
    const int32_t* __restrict__ pageIndptr,   // [B+1]
    const int32_t* __restrict__ lastPageLen,  // [B]
    const int32_t* __restrict__ qoIndptr,     // [B+1]
    float scale,
    int idxNHeads, int idxHeadDim, int pageSize, int maxKvLen,
    int causal
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

    const int pageStart = pageIndptr[seq];
    const int pageEnd = pageIndptr[seq + 1];
    const int numPages = pageEnd - pageStart;
    const int kvLen = numPages > 0 ? (numPages - 1) * pageSize + lastPageLen[seq] : 0;
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
                const __nv_bfloat16* k_ptr = kData + (size_t)pageId * pageSize * idxHeadDim + t * idxHeadDim;
                const __nv_bfloat16* q_ptr = q_s + warpIdx * idxHeadDim;

                float partial = 0.0f;
                for (int d = lane; d < idxHeadDim; d += 32) {
                    partial += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
                }
                // Warp reduce
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial += __shfl_xor_sync(0xffffffff, partial, offset);
                }
                if (lane == 0) {
                    partial *= scale;
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

void glm_indexer_score(GlmCtx* ctx, void* out, const void* q, const void* kData,
                       const void* weights, const int32_t* pageIndices,
                       const int32_t* pageIndptr, const int32_t* lastPageLen,
                       const int32_t* qoIndptr, float scale,
                       int totalQ, int idxNHeads, int idxHeadDim,
                       int pageSize, int maxKvLen, int causal) {
    cudaSetDevice(ctx->device_id);
    int block_size = idxNHeads * 32;
    if (block_size > 1024) block_size = 1024;
    int smem_size = idxNHeads * idxHeadDim * sizeof(__nv_bfloat16)  // q_s
                  + idxNHeads * sizeof(__nv_bfloat16)                // w_s
                  + idxNHeads * sizeof(float);                       // score_s
    indexer_score_kernel<<<totalQ, block_size, smem_size, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)q, (const __nv_bfloat16*)kData,
        (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr,
        scale, idxNHeads, idxHeadDim, pageSize, maxKvLen, causal);
}

// ---------------------------------------------------------------------------
// Fused indexer score + topk kernel
// Computes index scores for all KV positions and maintains a running top-k
// min-heap in shared memory. Outputs [totalQ, topk] int32 indices directly.
// When kvLen < topk, remaining entries are -1.
// ---------------------------------------------------------------------------

__global__ void indexer_score_topk_kernel(
    int32_t* __restrict__ out_idx,       // [totalQ, topk]
    const __nv_bfloat16* __restrict__ q,      // [totalQ, idxNHeads, idxHeadDim]
    const __nv_bfloat16* __restrict__ kData,  // [maxPages, pageSize, idxHeadDim]
    const __nv_bfloat16* __restrict__ weights,// [totalQ, idxNHeads]
    const int32_t* __restrict__ pageIndices,  // [numPages]
    const int32_t* __restrict__ pageIndptr,   // [B+1]
    const int32_t* __restrict__ lastPageLen,  // [B]
    const int32_t* __restrict__ qoIndptr,     // [B+1]
    float scale,
    int idxNHeads, int idxHeadDim, int pageSize, int topk,
    int causal,
    const uint8_t* __restrict__ custom_mask,  // bit-packed [qo_len, mask_kv_len], null = no mask
    const int32_t* __restrict__ mask_indptr,  // [B+1] offset into custom_mask per seq
    const int32_t* __restrict__ mask_kv_len   // [B] mask width per seq, null = numQueries
) {
    const int qIdx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpIdx = tid / 32;
    const int lane = tid % 32;

    int seq = 0;
    while (qoIndptr[seq + 1] <= qIdx) seq++;

    const int qLocalPos = qIdx - qoIndptr[seq];
    const int numQueries = qoIndptr[seq + 1] - qoIndptr[seq];

    const int pageStart = pageIndptr[seq];
    const int pageEnd = pageIndptr[seq + 1];
    const int numPages = pageEnd - pageStart;
    const int kvLen = numPages > 0 ? (numPages - 1) * pageSize + lastPageLen[seq] : 0;
    const int prefixLen = max(0, kvLen - numQueries);
    const int causalLimit = causal ? (prefixLen + qLocalPos) : (kvLen - 1);

    // CausalCustom mask setup: positions < mask_prefix_len are always attended
    // (causal is trivially satisfied). Positions >= mask_prefix_len are filtered
    // by the bit-packed custom mask at [qLocalPos * mask_kv_len + (pos - mask_prefix_len)].
    const uint8_t* mask_ptr = nullptr;
    int mask_kv_len_val = 0;
    int mask_prefix_len = 0;
    if (custom_mask && mask_indptr) {
        mask_ptr = custom_mask + mask_indptr[seq];
        mask_kv_len_val = mask_kv_len ? mask_kv_len[seq] : numQueries;
        mask_prefix_len = max(0, kvLen - mask_kv_len_val);
    }

    // Fast path: when the number of candidate positions fits within topk, the
    // selection is deterministic (every valid position is kept), so the score +
    // top-k heap is pure waste. Emit the identity selection directly. This is a
    // device-side branch on kvLen (from indptr/lastPageLen), so it stays
    // CUDA-graph capturable — the launch shape is unchanged across replays.
    const int numValid = causalLimit + 1;  // positions 0..causalLimit
    if (numValid <= topk) {
        for (int i = tid; i < topk; i += blockDim.x) {
            if (i < numValid) {
                bool valid = true;
                if (mask_ptr && i >= mask_prefix_len) {
                    int mask_offset = qLocalPos * mask_kv_len_val + (i - mask_prefix_len);
                    valid = ((mask_ptr[mask_offset >> 3] >> (mask_offset & 7)) & 1);
                }
                out_idx[(size_t)qIdx * topk + i] = valid ? i : -1;
            } else {
                out_idx[(size_t)qIdx * topk + i] = -1;
            }
        }
        return;
    }

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* w_s = q_s + idxNHeads * idxHeadDim;
    float* score_s = reinterpret_cast<float*>(w_s + idxNHeads);
    float* heap_vals = score_s + idxNHeads;
    int32_t* heap_idx = reinterpret_cast<int32_t*>(heap_vals + topk);

    for (int i = tid; i < idxNHeads * idxHeadDim; i += blockDim.x) {
        q_s[i] = q[(size_t)qIdx * idxNHeads * idxHeadDim + i];
    }
    if (tid < idxNHeads) {
        w_s[tid] = weights[qIdx * idxNHeads + tid];
    }
    if (tid == 0) {
        for (int i = 0; i < topk; i++) {
            heap_vals[i] = -INFINITY;
            heap_idx[i] = -1;
        }
    }
    __syncthreads();

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

            // CausalCustom mask: skip masked-out positions (they never enter the heap)
            if (mask_ptr && localK >= mask_prefix_len) {
                int mask_offset = qLocalPos * mask_kv_len_val + (localK - mask_prefix_len);
                if (!((mask_ptr[mask_offset >> 3] >> (mask_offset & 7)) & 1)) continue;
            }

            if (warpIdx < idxNHeads) {
                const __nv_bfloat16* k_ptr = kData + (size_t)pageId * pageSize * idxHeadDim + t * idxHeadDim;
                const __nv_bfloat16* q_ptr = q_s + warpIdx * idxHeadDim;

                float partial = 0.0f;
                for (int d = lane; d < idxHeadDim; d += 32) {
                    partial += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
                }
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial += __shfl_xor_sync(0xffffffff, partial, offset);
                }
                if (lane == 0) {
                    partial *= scale;
                    partial = fmaxf(partial, 0.0f);
                    score_s[warpIdx] = partial;
                }
            }

            __syncthreads();

            if (tid == 0) {
                float indexScore = 0.0f;
                for (int h = 0; h < idxNHeads; h++) {
                    indexScore += __bfloat162float(w_s[h]) * score_s[h];
                }
                indexScore = __bfloat162float(__float2bfloat16(indexScore));

                if (indexScore > heap_vals[0]) {
                    heap_vals[0] = indexScore;
                    heap_idx[0] = localK;
                    int pos = 0;
                    while (true) {
                        int left = 2 * pos + 1;
                        int right = 2 * pos + 2;
                        int smallest = pos;
                        if (left < topk && heap_vals[left] < heap_vals[smallest]) smallest = left;
                        if (right < topk && heap_vals[right] < heap_vals[smallest]) smallest = right;
                        if (smallest == pos) break;
                        float tv = heap_vals[pos]; heap_vals[pos] = heap_vals[smallest]; heap_vals[smallest] = tv;
                        int32_t ti = heap_idx[pos]; heap_idx[pos] = heap_idx[smallest]; heap_idx[smallest] = ti;
                        pos = smallest;
                    }
                }
            }

            __syncthreads();
        }
    }

    for (int i = tid; i < topk; i += blockDim.x) {
        out_idx[(size_t)qIdx * topk + i] = heap_idx[i];
    }
}

void glm_indexer_score_topk(GlmCtx* ctx, int32_t* out_idx,
    const void* q, const void* kData, const void* weights,
    const int32_t* pageIndices, const int32_t* pageIndptr,
    const int32_t* lastPageLen, const int32_t* qoIndptr,
    float scale, int totalQ, int idxNHeads, int idxHeadDim,
    int pageSize, int topk, int causal,
    const uint8_t* custom_mask, const int32_t* mask_indptr, const int32_t* mask_kv_len) {
    cudaSetDevice(ctx->device_id);
    int block_size = idxNHeads * 32;
    if (block_size > 1024) block_size = 1024;
    int smem_size = idxNHeads * idxHeadDim * sizeof(__nv_bfloat16)
                  + idxNHeads * sizeof(__nv_bfloat16)
                  + idxNHeads * sizeof(float)
                  + topk * sizeof(float)
                  + topk * sizeof(int32_t);
    indexer_score_topk_kernel<<<totalQ, block_size, smem_size, GLM_STREAM(ctx)>>>(
        out_idx, (const __nv_bfloat16*)q, (const __nv_bfloat16*)kData,
        (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr,
        scale, idxNHeads, idxHeadDim, pageSize, topk, causal,
        custom_mask, mask_indptr, mask_kv_len);
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
// topk_idx[t, *] in order.
__global__ void topk_to_slots_kernel(
    int32_t* __restrict__ slots,          // [num_tokens, topk]
    int32_t* __restrict__ topk_length,    // [num_tokens] (nullable)
    const int32_t* __restrict__ topk_idx, // [num_tokens, topk]
    const int32_t* __restrict__ page_indices,
    const int32_t* __restrict__ page_indptr,
    const int32_t* __restrict__ last_page_len,
    const int32_t* __restrict__ batch_indices,
    int topk, int page_size, int num_tokens,
    uint32_t cp_world_size, uint32_t cp_rank
) {
    const int token = blockIdx.x;
    if (token >= num_tokens) return;
    const int seq = batch_indices[token];
    const int num_pages = page_indptr[seq + 1] - page_indptr[seq];
    const int page_base = page_indptr[seq];
    const int eff_page_size = (cp_world_size > 1) ? (page_size / (int)cp_world_size) : page_size;
    const int local_kv_len = (num_pages - 1) * eff_page_size + last_page_len[seq];

    int32_t* out = slots + (size_t)token * topk;
    const int32_t* in = topk_idx + (size_t)token * topk;

    extern __shared__ int32_t s_slots[];  // [topk]

    // Pass 1: compute slot for each topk entry in parallel, store in smem
    // (or -1 if the entry is invalid: negative, wrong CP rank, or beyond kvLen).
    for (int i = threadIdx.x; i < topk; i += blockDim.x) {
        const int token_pos = in[i];
        int slot = -1;
        if (token_pos >= 0) {
            int local_pos;
            if (cp_world_size > 1) {
                if ((uint32_t)token_pos % cp_world_size == cp_rank) {
                    local_pos = (token_pos - (int)cp_rank) / (int)cp_world_size;
                } else {
                    local_pos = -1;
                }
            } else {
                local_pos = token_pos;
            }
            if (local_pos >= 0 && local_pos < local_kv_len) {
                const int abs_page = page_indices[page_base + local_pos / eff_page_size];
                slot = abs_page * eff_page_size + (local_pos % eff_page_size);
            }
        }
        s_slots[i] = slot;
    }
    __syncthreads();

    // Pass 2: serial compaction in thread 0 (preserves input order).
    if (threadIdx.x == 0) {
        int count = 0;
        for (int i = 0; i < topk; i++) {
            if (s_slots[i] >= 0)
                out[count++] = s_slots[i];
        }
        for (int i = count; i < topk; i++) out[i] = -1;
        if (topk_length) topk_length[token] = count;
    }
}

void glm_topk_to_slots(GlmCtx* ctx, int32_t* slots, int32_t* topk_length, const int32_t* topk_idx,
                       const int32_t* page_indices, const int32_t* page_indptr,
                       const int32_t* last_page_len, const int32_t* batch_indices,
                       int num_tokens, int topk, int page_size,
                       uint32_t cp_world_size, uint32_t cp_rank) {
    cudaSetDevice(ctx->device_id);
    int smem = topk * sizeof(int32_t);
    topk_to_slots_kernel<<<num_tokens, 256, smem, GLM_STREAM(ctx)>>>(
        slots, topk_length, topk_idx, page_indices, page_indptr, last_page_len, batch_indices,
        topk, page_size, num_tokens, cp_world_size, cp_rank);
}

// ---------------------------------------------------------------------------
// Exact large-K top-K over non-negative bf16 scores, by histogram.
//
// Indexer scores are ReLU'd and bf16-rounded, so each score is one of only
// 32768 non-negative bf16 values. Selecting the top-K (K up to a few thousand)
// out of N (up to ~200k) is therefore an exact histogram problem, done in three
// fully-parallel multi-block passes — no per-thread K-arrays, no serial heap.
// All passes have a fixed launch shape (grid sized for max context), so this is
// CUDA-graph capturable; per-row length is read on-device from row_len.
//
//   hist:  [batch, 32768] i32 scratch   meta: [batch, 4] i32 scratch
//   meta layout per row: [tau_key, tie_take, out_count, tie_count]
// Output out_idx[row, 0..K) = selected position indices (unordered), -1 padded.
// ---------------------------------------------------------------------------

#define IDX_NBUCKET 65536   // full signed bf16 range, mapped to a monotonic key

static __device__ __forceinline__ int bf16_key(const __nv_bfloat16* p) {
    // Order-preserving float->uint key: negatives -> [0,0x7FFF] (reversed),
    // non-negatives -> [0x8000,0xFFFF]. Monotonic in value across the full range,
    // so histogram bucket order == score order. Scores are signed (final weighted
    // sum is not ReLU'd), so this must handle the sign bit.
    unsigned short u = *reinterpret_cast<const unsigned short*>(p);
    unsigned short k = (u & 0x8000) ? (unsigned short)(~u) : (unsigned short)(u | 0x8000);
    return (int)k;
}

__global__ void idx_hist_kernel(
    const __nv_bfloat16* __restrict__ scores, const int32_t* __restrict__ row_len,
    int32_t* __restrict__ hist, int stride
) {
    const int row = blockIdx.y;
    const int len = row_len ? row_len[row] : stride;
    const __nv_bfloat16* s = scores + (size_t)row * stride;
    int32_t* h = hist + (size_t)row * IDX_NBUCKET;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) {
        atomicAdd(&h[bf16_key(&s[i])], 1);
    }
}

// One block (256 threads) per row: init out_idx to -1, then find the threshold
// bucket tau (largest set of top buckets whose counts sum to >= topk) with a
// two-level parallel scan over the 32768 buckets — 128 buckets/thread, a 256-way
// block scan, then the winning thread refines within its range.
__global__ void idx_threshold_kernel(
    const int32_t* __restrict__ hist, int32_t* __restrict__ meta,
    int32_t* __restrict__ out_idx, const int32_t* __restrict__ row_len,
    int stride, int topk
) {
    const int row = blockIdx.x;
    const int len = row_len ? row_len[row] : stride;
    for (int i = threadIdx.x; i < topk; i += blockDim.x)
        out_idx[(size_t)row * topk + i] = (i < len && len <= topk) ? i : -1;
    // Identity case: every valid position is selected — write it directly and
    // disable the gather pass (tau = INT_MAX so no position matches).
    if (len <= topk) {
        if (threadIdx.x == 0) {
            int32_t* m = meta + (size_t)row * 4;
            m[0] = 0x7FFFFFFF; m[1] = 0; m[2] = topk; m[3] = 0;
        }
        return;
    }

    const int t = threadIdx.x;                 // 0..255
    const int PER = IDX_NBUCKET / 256;         // 128 buckets per thread
    const int32_t* h = hist + (size_t)row * IDX_NBUCKET;
    // Thread t owns the t-th band from the top: buckets [hi-PER, hi).
    const int hi = IDX_NBUCKET - t * PER;
    long localSum = 0;
    for (int k = hi - PER; k < hi; k++) localSum += h[k];

    __shared__ long partial[256];
    __shared__ int s_winner;
    __shared__ long s_above;   // count in bands strictly above the winning band
    partial[t] = localSum;
    __syncthreads();

    if (t == 0) {
        long cum = 0; int winner = 255;
        for (int j = 0; j < 256; j++) {
            if (cum + partial[j] >= topk) { winner = j; break; }
            cum += partial[j];
        }
        s_winner = winner;
        s_above = cum;         // total in bands 0..winner-1
    }
    __syncthreads();

    if (t == s_winner) {
        const int whi = IDX_NBUCKET - t * PER;
        long cum = s_above; long numAbove = s_above; int tau = whi - PER;
        for (int k = whi - 1; k >= whi - PER; k--) {
            long c = h[k];
            if (cum + c >= topk) { tau = k; numAbove = cum; break; }
            cum += c;
        }
        int32_t* m = meta + (size_t)row * 4;
        m[0] = tau;                   // threshold bucket
        m[1] = topk - (int)numAbove;  // ties to take from the tau bucket
        m[2] = 0;                     // out_count
        m[3] = 0;                     // tie_count
    }
}

__global__ void idx_gather_kernel(
    const __nv_bfloat16* __restrict__ scores, const int32_t* __restrict__ row_len,
    int32_t* __restrict__ meta, int32_t* __restrict__ out_idx, int stride, int topk
) {
    const int row = blockIdx.y;
    const int len = row_len ? row_len[row] : stride;
    const __nv_bfloat16* s = scores + (size_t)row * stride;
    int32_t* m = meta + (size_t)row * 4;
    const int tau = m[0];
    const int tieTake = m[1];
    int32_t* out = out_idx + (size_t)row * topk;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) {
        const int key = bf16_key(&s[i]);
        if (key > tau) {
            int p = atomicAdd(&m[2], 1);
            if (p < topk) out[p] = i;
        } else if (key == tau) {
            int t = atomicAdd(&m[3], 1);
            if (t < tieTake) {
                int p = atomicAdd(&m[2], 1);
                if (p < topk) out[p] = i;
            }
        }
    }
}

void glm_topk_from_scores(GlmCtx* ctx, int32_t* out_idx,
    const void* scores, const int32_t* row_len,
    int32_t* hist, int32_t* meta,
    int batch, int stride, int topk, int num_splits) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    cudaMemsetAsync(hist, 0, (size_t)batch * IDX_NBUCKET * sizeof(int32_t), stream);
    cudaMemsetAsync(meta, 0, (size_t)batch * 4 * sizeof(int32_t), stream);
    dim3 grid(num_splits, batch);
    idx_hist_kernel<<<grid, 256, 0, stream>>>(
        (const __nv_bfloat16*)scores, row_len, hist, stride);
    idx_threshold_kernel<<<batch, 256, 0, stream>>>(hist, meta, out_idx, row_len, stride, topk);
    idx_gather_kernel<<<grid, 256, 0, stream>>>(
        (const __nv_bfloat16*)scores, row_len, meta, out_idx, stride, topk);
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
static __device__ __forceinline__ bool idx_is_masked(
    int pos, int qLocalPos,
    const uint8_t* mask_ptr, int mask_kv_len_val, int mask_prefix_len)
{
    if (!mask_ptr) return false;
    if (pos < mask_prefix_len) return false;
    int mask_offset = qLocalPos * mask_kv_len_val + (pos - mask_prefix_len);
    return !((mask_ptr[mask_offset >> 3] >> (mask_offset & 7)) & 1);
}

// HAS_MASK is a compile-time switch: the whole custom-mask path (extra args,
// per-position bit test) is elided when false, so the common no-mask case pays
// nothing. The masked variant writes -inf for masked positions so they land in
// the lowest histogram bucket and are never selected — identical semantics to
// the two-level prefill score kernel.
template <bool HAS_MASK>
__global__ void idx_score_kernel(
    __nv_bfloat16* __restrict__ scores,       // [totalQ, maxKv]
    int32_t* __restrict__ row_len,            // [totalQ]  (numValid per query)
    const __nv_bfloat16* __restrict__ q,      // [totalQ, idxNHeads, idxHeadDim]
    const __nv_bfloat16* __restrict__ kData,  // [maxPages, pageSize, idxHeadDim]
    const __nv_bfloat16* __restrict__ weights,// [totalQ, idxNHeads]
    const int32_t* __restrict__ pageIndices, const int32_t* __restrict__ pageIndptr,
    const int32_t* __restrict__ lastPageLen, const int32_t* __restrict__ qoIndptr,
    float scale, int idxNHeads, int idxHeadDim, int pageSize, int maxKv, int causal,
    const uint8_t* __restrict__ custom_mask,
    const int32_t* __restrict__ mask_indptr, const int32_t* __restrict__ mask_kv_len,
    int qGlobalStart
) {
    const int qIdx = blockIdx.y;
    int seq = 0;
    while (qoIndptr[seq + 1] <= qIdx) seq++;
    const int qLocalPos = qIdx - qoIndptr[seq];
    const int numQueries = qoIndptr[seq + 1] - qoIndptr[seq];
    const int pageStart = pageIndptr[seq];
    const int numPages = pageIndptr[seq + 1] - pageStart;
    const int kvLen = numPages > 0 ? (numPages - 1) * pageSize + lastPageLen[seq] : 0;
    // qGlobalStart shifts a shard's local query row to its true sequence position
    // (0 outside query-sharding) so the causal limit and mask row stay correct.
    const int prefixLen = max(0, kvLen - numQueries) + qGlobalStart;
    const int causalLimit = causal ? (prefixLen + qLocalPos) : (kvLen - 1);
    const int numValid = causalLimit + 1;

    if (blockIdx.x == 0 && threadIdx.x == 0) row_len[qIdx] = numValid;

    // Custom-mask setup — compiled out entirely when !HAS_MASK.
    const uint8_t* mask_ptr = nullptr;
    int mask_kv_len_val = 0, mask_prefix_len = 0;
    if constexpr (HAS_MASK) {
        mask_ptr = custom_mask + mask_indptr[seq];
        mask_kv_len_val = mask_kv_len ? mask_kv_len[seq] : numQueries;
        mask_prefix_len = max(0, kvLen - mask_kv_len_val);
    }
    const int maskRow = qLocalPos + qGlobalStart;

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
            if (idx_is_masked(pos, maskRow, mask_ptr, mask_kv_len_val, mask_prefix_len)) {
                if (lane == 0)
                    scores[(size_t)qIdx * maxKv + pos] = __float2bfloat16(-INFINITY);
                continue;
            }
        }
        const int pageId = pageIndices[pageStart + pos / pageSize];
        const __nv_bfloat16* kbase = kData + (size_t)pageId * pageSize * idxHeadDim
                                     + (pos % pageSize) * idxHeadDim;
        float acc = 0.f;
        for (int h = 0; h < idxNHeads; h++) {
            const __nv_bfloat16* qh = q_s + h * idxHeadDim;
            float partial = 0.f;
            for (int d = lane; d < idxHeadDim; d += 32)
                partial += __bfloat162float(qh[d]) * __bfloat162float(kbase[d]);
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor_sync(0xffffffff, partial, off);
            if (lane == 0) acc += __bfloat162float(w_s[h]) * fmaxf(partial * scale, 0.f);
        }
        if (lane == 0)
            scores[(size_t)qIdx * maxKv + pos] = __float2bfloat16(acc);
    }
}

// Full v2 indexer top-k: multi-block score -> histogram select. Drop-in
// replacement for glm_indexer_score_topk with the same out_idx semantics.
// scratch: scores [totalQ, maxKv] bf16, rowLen [totalQ] i32, hist [totalQ,32768]
// i32, meta [totalQ, 4] i32.
void glm_indexer_score_topk_v2(GlmCtx* ctx, int32_t* out_idx,
    const void* q, const void* kData, const void* weights,
    const int32_t* pageIndices, const int32_t* pageIndptr,
    const int32_t* lastPageLen, const int32_t* qoIndptr,
    float scale, int totalQ, int idxNHeads, int idxHeadDim,
    int pageSize, int topk, int causal,
    const uint8_t* custom_mask, const int32_t* mask_indptr, const int32_t* mask_kv_len,
    void* scores, int32_t* rowLen, int32_t* hist, int32_t* meta,
    int maxKv, int num_splits, int qGlobalStart) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    dim3 grid(num_splits, totalQ);
    int block = 256;
    size_t smem = (size_t)idxNHeads * idxHeadDim * sizeof(__nv_bfloat16)
                + idxNHeads * sizeof(__nv_bfloat16);
    // Dispatch the mask-free variant when there is no custom mask so its bit-test
    // path is compiled out (zero cost for the common case).
    auto kern = (custom_mask && mask_indptr) ? idx_score_kernel<true> : idx_score_kernel<false>;
    kern<<<grid, block, smem, stream>>>(
        (__nv_bfloat16*)scores, rowLen, (const __nv_bfloat16*)q, (const __nv_bfloat16*)kData,
        (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr,
        scale, idxNHeads, idxHeadDim, pageSize, maxKv, causal,
        custom_mask, mask_indptr, mask_kv_len, qGlobalStart);
    glm_topk_from_scores(ctx, out_idx, scores, rowLen, hist, meta,
                         totalQ, maxKv, topk, num_splits);
}

// ---------------------------------------------------------------------------
// Two-level prefill indexer: score once → coarse/fine histogram top-K.
//
// Replaces the v1 serial heap kernel for prefill. Scores are computed once
// into a [totalQ, maxKv] BF16 buffer (same as v2), then a 2-level histogram
// (1024 coarse + 64 fine = 65536 total buckets) selects the exact top-K.
// This avoids v2's 65536-bucket histogram (1.07 GB) while keeping a single
// scoring pass.
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
#define IDX_FINE_BUCKETS 64   // IDX_NBUCKET / IDX_COARSE_BUCKETS

// ---------------------------------------------------------------------------
// Tensor-core (mma.sync m16n8k16 bf16) primitives for the score kernel.
// Same instruction/fragment layout the MoE GEMM uses (glm_mma_moe.cu); kept in
// a local namespace so the anonymous-namespace helpers there don't collide.
// ---------------------------------------------------------------------------
namespace idxmma {
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int WARPS = 8, CTA = WARPS * 32;   // 256 threads
constexpr int TM = 64;            // queries per tile      (NUM_M = TM/16 = 4)
constexpr int TN = 192;           // KV positions per tile  (NPW = 3)
constexpr int NUM_M = TM / MMA_M;                       // 4
constexpr int NPW = TN / (MMA_N * WARPS);               // 3
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
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n"
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
// position), so each output tile loads its 64-wide K slab once and streams the
// per-head Q tiles through it, folding scale·ReLU·weight into a persistent
// fp32 score accumulator (one MMA pass, no per-head materialization).
//
// Grid: (ceil(maxKv/TN), numQueryTiles).  Block: 128 threads (4 warps).
// blockIdx.y maps to (seq, query tile) so tiles never cross a sequence
// boundary; blockIdx.x is a 64-wide KV tile of global positions. Masked
// positions are written -inf (lowest histogram bucket); positions beyond the
// causal limit / kvLen are left untouched (never read by the histogram passes).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(idxmma::CTA)
idx_prefill_score_mma_kernel(
    __nv_bfloat16* __restrict__ scores,       // [totalQ, maxKv]
    int32_t* __restrict__ rowLen,             // [totalQ]
    const __nv_bfloat16* __restrict__ q,      // [totalQ, idxNHeads, idxHeadDim]
    const __nv_bfloat16* __restrict__ kData,  // [maxPages, pageSize, idxHeadDim]
    const __nv_bfloat16* __restrict__ weights,// [totalQ, idxNHeads]
    const int32_t* __restrict__ pageIndices, const int32_t* __restrict__ pageIndptr,
    const int32_t* __restrict__ lastPageLen, const int32_t* __restrict__ qoIndptr,
    int totalQ, float scale, int idxNHeads, int idxHeadDim, int pageSize,
    int maxKv, int causal,
    const uint8_t* __restrict__ custom_mask,
    const int32_t* __restrict__ mask_indptr, const int32_t* __restrict__ mask_kv_len,
    int qGlobalStart)
{
    using namespace idxmma;

    // Map blockIdx.y -> (seq, qStart). Tiles are laid out per-sequence so a tile
    // never straddles two sequences. The scan reads qoIndptr[0..B]; it stops once
    // a sequence reaches totalQ, so no batch count is needed.
    const int gy = blockIdx.y;
    int seq = -1, qStart = 0;
    {
        int acc = 0;
        for (int s = 0; ; s++) {
            int qs = qoIndptr[s], qe = qoIndptr[s + 1];
            int nt = (qe - qs + TM - 1) / TM;
            if (gy < acc + nt) { seq = s; qStart = qs + (gy - acc) * TM; break; }
            acc += nt;
            if (qe >= totalQ) break;
        }
    }
    if (seq < 0) return;
    // Under query-sharding q/weights/scores/out are the shard's local [totalQ,...]
    // buffers while qoIndptr still describes the full sequence, so stop once a tile
    // lands past this shard's local row count (also covers the launch slop blocks).
    if (qStart >= totalQ) return;

    const int qoStart   = qoIndptr[seq];
    const int numQueries = qoIndptr[seq + 1] - qoStart;
    // Clamp to both the sequence end (multi-seq) and the local row count (shard).
    const int m_valid   = min(TM, min(qoIndptr[seq + 1], totalQ) - qStart);
    const int pageStart = pageIndptr[seq];
    const int numPages  = pageIndptr[seq + 1] - pageStart;
    const int kvLen     = numPages > 0 ? (numPages - 1) * pageSize + lastPageLen[seq] : 0;
    // qGlobalStart shifts a shard's local query row to its true sequence position
    // for causal limits (0 in the non-sharded path → identical to before).
    const int prefixLen = max(0, kvLen - numQueries) + qGlobalStart;
    const int tileStart = blockIdx.x * TN;
    if (tileStart >= kvLen) return;
    // Causal prune: skip the whole tile if it sits past the last query's limit.
    const int maxQGlobal = qStart + m_valid - 1;
    const int maxCausal  = causal ? (prefixLen + (maxQGlobal - qoStart)) : (kvLen - 1);
    if (tileStart > maxCausal) return;

    const uint8_t* mask_ptr = nullptr;
    int mask_kv_len_val = 0, mask_prefix_len = 0;
    if (custom_mask && mask_indptr) {
        mask_ptr = custom_mask + mask_indptr[seq];
        mask_kv_len_val = mask_kv_len ? mask_kv_len[seq] : numQueries;
        mask_prefix_len = max(0, kvLen - mask_kv_len_val);
    }

    // rowLen (= numValid per query) written once, by the first KV tile.
    if (blockIdx.x == 0) {
        for (int ql = threadIdx.x; ql < m_valid; ql += CTA) {
            int qg = qStart + ql;
            rowLen[qg] = (causal ? (prefixLen + (qg - qoStart)) : (kvLen - 1)) + 1;
        }
    }

    const int strideA = idxHeadDim + PAD_A;   // padded Q row stride
    const int strideB = TN + PAD_B;            // padded K row stride

    extern __shared__ char smem[];
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem);   // [idxHeadDim, strideB] (d-major)
    __nv_bfloat16* qbuf0  = smem_b + (size_t)idxHeadDim * strideB;    // [TM, strideA] (buffer 0)
    __nv_bfloat16* qbuf1  = qbuf0 + (size_t)TM * strideA;             // [TM, strideA] (buffer 1)
    __nv_bfloat16* w_s     = qbuf1 + (size_t)TM * strideA;            // [TM, idxNHeads]
    __nv_bfloat16* qbuf[2] = { qbuf0, qbuf1 };

    // Zero both Q buffers once: padded rows (ql >= m_valid) are never cp.async'd,
    // so they stay 0 and fold to nothing (avoids NaN from 0*inf on stale smem).
    for (int i = threadIdx.x; i < 2 * TM * strideA; i += CTA)
        qbuf0[i] = __float2bfloat16(0.f);

    // Load the K slab once, transposed into d-major smem: smem_b[d*strideB + pos].
    for (int i = threadIdx.x; i < TN * idxHeadDim; i += CTA) {
        int pos = i / idxHeadDim, d = i % idxHeadDim;
        int gpos = tileStart + pos;
        __nv_bfloat16 v = __float2bfloat16(0.f);
        if (gpos < kvLen) {
            int pageId = pageIndices[pageStart + gpos / pageSize];
            v = kData[(size_t)pageId * pageSize * idxHeadDim + (gpos % pageSize) * idxHeadDim + d];
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

    // 2-stage software pipeline: prefetch head h+1's Q while head h computes.
    issue_q(0, qbuf[0]); cp_commit();

    for (int h = 0; h < idxNHeads; h++) {
        if (h + 1 < idxNHeads) { issue_q(h + 1, qbuf[(h + 1) & 1]); cp_commit(); cp_wait<1>(); }
        else                   { cp_wait<0>(); }
        __syncthreads();
        const __nv_bfloat16* qa = qbuf[h & 1];

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
        __syncthreads();   // all reads of qbuf[h&1] done before it is refilled at h+2
    }

    // Epilogue: write scores with per-element causal / mask / bounds checks.
    const int group = lane >> 2;
    const int colb  = (lane & 3) * 2;
    auto write_one = [&](int ql, int pos_in_tile, float val) {
        if (ql >= m_valid) return;
        int gpos = tileStart + pos_in_tile;
        if (gpos >= kvLen) return;
        int qg = qStart + ql;
        int cl = causal ? (prefixLen + (qg - qoStart)) : (kvLen - 1);
        if (gpos > cl) return;
        __nv_bfloat16 out =
            (mask_ptr && idx_is_masked(gpos, qg - qoStart + qGlobalStart, mask_ptr, mask_kv_len_val, mask_prefix_len))
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
    for (int i = threadIdx.x; i < len; i += blockDim.x)
        atomicAdd(&sh[bf16_key(&s[i]) / IDX_FINE_BUCKETS], 1);
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
    int32_t* __restrict__ out_idx, int topk)
{
    const int row = blockIdx.x;
    const int32_t* h = coarseHist + (size_t)row * IDX_COARSE_BUCKETS;

    // Initialize output to -1
    for (int i = threadIdx.x; i < topk; i += blockDim.x)
        out_idx[(size_t)row * topk + i] = -1;

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

    // Identity case: all positions fit in topk
    if (total <= topk) {
        if (t == 0) {
            int32_t* m = meta + (size_t)row * 4;
            m[0] = -1;  // identity sentinel
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
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        int key = bf16_key(&s[i]);
        if (key / IDX_FINE_BUCKETS == coarseTau)
            atomicAdd(&sh[key % IDX_FINE_BUCKETS], 1);
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
// Grid: totalQ  Block: 256.  One block per row: compaction counters live in
// shared memory (no cross-block global-atomic contention). Writes positions with
// key > tau; for key == tau, takes up to tie_take. Identity (tau<0) takes all.
// ---------------------------------------------------------------------------
__global__ void idx_prefill_gather_buf_kernel(
    int32_t* __restrict__ out_idx,           // [totalQ, topk]
    const int32_t* __restrict__ meta,        // [totalQ, 4]
    const __nv_bfloat16* __restrict__ scores,// [totalQ, maxKv]
    const int32_t* __restrict__ rowLen,      // [totalQ]
    int maxKv, int topk)
{
    const int row = blockIdx.x;
    const int32_t* m = meta + (size_t)row * 4;
    const int tau = m[0];
    const int tieTake = m[1];
    const int len = rowLen ? rowLen[row] : maxKv;
    const __nv_bfloat16* s = scores + (size_t)row * maxKv;
    int32_t* out = out_idx + (size_t)row * topk;

    __shared__ int s_count, s_tie;
    if (threadIdx.x == 0) { s_count = 0; s_tie = 0; }
    __syncthreads();

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        if (tau < 0) {
            int p = atomicAdd(&s_count, 1);
            if (p < topk) out[p] = i;
        } else {
            int key = bf16_key(&s[i]);
            if (key > tau) {
                int p = atomicAdd(&s_count, 1);
                if (p < topk) out[p] = i;
            } else if (key == tau) {
                int t = atomicAdd(&s_tie, 1);
                if (t < tieTake) {
                    int p = atomicAdd(&s_count, 1);
                    if (p < topk) out[p] = i;
                }
            }
        }
    }
    // Fill unused slots with -1. Rows shorter than topk (causal prefix) and any
    // threshold undershoot from bf16 ties leave a tail; the sparse-MLA kernel
    // clamps negative indices to page 0 at load and masks them out in QK, so -1
    // is the required sentinel. Without this the tail holds stale indices from a
    // prior chunk and the sparse-MLA KV gather reads out of bounds.
    __syncthreads();
    for (int i = min(s_count, topk) + threadIdx.x; i < topk; i += blockDim.x)
        out[i] = -1;
}

// Host function: score once into buffer, then 2-level histogram top-K.
void glm_indexer_score_topk_prefill(GlmCtx* ctx, int32_t* out_idx,
    const void* q, const void* kData, const void* weights,
    const int32_t* pageIndices, const int32_t* pageIndptr,
    const int32_t* lastPageLen, const int32_t* qoIndptr,
    float scale, int totalQ, int idxNHeads, int idxHeadDim,
    int pageSize, int topk, int causal,
    const uint8_t* custom_mask, const int32_t* mask_indptr, const int32_t* mask_kv_len,
    void* scores, int32_t* rowLen, int maxKv,
    int32_t* coarseHist, int32_t* fineHist, int32_t* meta,
    int numSplits, int qGlobalStart) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);

    // Zero histograms and meta.
    cudaMemsetAsync(coarseHist, 0, (size_t)totalQ * IDX_COARSE_BUCKETS * sizeof(int32_t), stream);
    cudaMemsetAsync(fineHist, 0, (size_t)totalQ * IDX_FINE_BUCKETS * sizeof(int32_t), stream);
    cudaMemsetAsync(meta, 0, (size_t)totalQ * 4 * sizeof(int32_t), stream);

    // Pass 1: tensor-core score into buffer. Grid (ceil(maxKv/TN), queryTiles);
    // the +slop covers per-sequence tile rounding without needing a batch count
    // (out-of-range tiles map to seq<0 and return immediately).
    {
        using idxmma::TM; using idxmma::TN;
        const int strideA = idxHeadDim + idxmma::PAD_A;
        const int strideB = TN + idxmma::PAD_B;
        // K slab + double-buffered Q (2 heads) + weights (all padded to match kernel).
        size_t mma_smem = ((size_t)idxHeadDim * strideB + 2 * (size_t)TM * strideA
                           + (size_t)TM * idxNHeads) * sizeof(__nv_bfloat16);
        // Per-device opt-in for >48KB dynamic smem. cudaFuncSetAttribute is
        // per-device, so this must run on every device (cheap, idempotent).
        cudaFuncSetAttribute(idx_prefill_score_mma_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)mma_smem);
        dim3 grid((maxKv + TN - 1) / TN, (totalQ + TM - 1) / TM + 256);
        idx_prefill_score_mma_kernel<<<grid, idxmma::CTA, mma_smem, stream>>>(
            (__nv_bfloat16*)scores, rowLen,
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)kData,
            (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr,
            totalQ, scale, idxNHeads, idxHeadDim, pageSize, maxKv, causal,
            custom_mask, mask_indptr, mask_kv_len, qGlobalStart);
    }

    // Passes 2-6: histogram + gather from buffer.
    {
        // Pass 2: coarse histogram — one block per row
        idx_prefill_coarse_hist_buf_kernel<<<totalQ, 256, 0, stream>>>(
            coarseHist, (const __nv_bfloat16*)scores, rowLen, maxKv);

        // Pass 3: coarse threshold (1 block per query)
        idx_prefill_coarse_threshold_kernel<<<totalQ, 256, 0, stream>>>(
            coarseHist, meta, out_idx, topk);

        // Pass 4: fine histogram — one block per row
        idx_prefill_fine_hist_buf_kernel<<<totalQ, 256, 0, stream>>>(
            fineHist, meta, (const __nv_bfloat16*)scores, rowLen, maxKv);

        // Pass 5: fine threshold (1 block per query)
        idx_prefill_fine_threshold_kernel<<<totalQ, 256, 0, stream>>>(
            fineHist, meta, topk);

        // Pass 6: gather from buffer — one block per row
        idx_prefill_gather_buf_kernel<<<totalQ, 256, 0, stream>>>(
            out_idx, meta, (const __nv_bfloat16*)scores, rowLen, maxKv, topk);
    }
}
