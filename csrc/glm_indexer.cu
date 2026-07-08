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

    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < topk; i += blockDim.x) {
        const int token_pos = in[i];
        if (token_pos < 0) continue;
        int local_pos;
        if (cp_world_size > 1) {
            if ((uint32_t)token_pos % cp_world_size != cp_rank) continue;
            local_pos = (token_pos - (int)cp_rank) / (int)cp_world_size;
        } else {
            local_pos = token_pos;
        }
        if (local_pos >= local_kv_len) continue;
        const int abs_page = page_indices[page_base + local_pos / eff_page_size];
        const int slot = abs_page * eff_page_size + (local_pos % eff_page_size);
        out[atomicAdd(&s_count, 1)] = slot;   // compact to front
    }
    __syncthreads();

    const int count = s_count;
    for (int i = count + threadIdx.x; i < topk; i += blockDim.x) out[i] = -1;
    if (threadIdx.x == 0 && topk_length) topk_length[token] = count;
}

void glm_topk_to_slots(GlmCtx* ctx, int32_t* slots, int32_t* topk_length, const int32_t* topk_idx,
                       const int32_t* page_indices, const int32_t* page_indptr,
                       const int32_t* last_page_len, const int32_t* batch_indices,
                       int num_tokens, int topk, int page_size,
                       uint32_t cp_world_size, uint32_t cp_rank) {
    cudaSetDevice(ctx->device_id);
    topk_to_slots_kernel<<<num_tokens, 256, 0, GLM_STREAM(ctx)>>>(
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
__global__ void idx_score_kernel(
    __nv_bfloat16* __restrict__ scores,       // [totalQ, maxKv]
    int32_t* __restrict__ row_len,            // [totalQ]  (numValid per query)
    const __nv_bfloat16* __restrict__ q,      // [totalQ, idxNHeads, idxHeadDim]
    const __nv_bfloat16* __restrict__ kData,  // [maxPages, pageSize, idxHeadDim]
    const __nv_bfloat16* __restrict__ weights,// [totalQ, idxNHeads]
    const int32_t* __restrict__ pageIndices, const int32_t* __restrict__ pageIndptr,
    const int32_t* __restrict__ lastPageLen, const int32_t* __restrict__ qoIndptr,
    float scale, int idxNHeads, int idxHeadDim, int pageSize, int maxKv, int causal
) {
    const int qIdx = blockIdx.y;
    int seq = 0;
    while (qoIndptr[seq + 1] <= qIdx) seq++;
    const int qLocalPos = qIdx - qoIndptr[seq];
    const int numQueries = qoIndptr[seq + 1] - qoIndptr[seq];
    const int pageStart = pageIndptr[seq];
    const int numPages = pageIndptr[seq + 1] - pageStart;
    const int kvLen = numPages > 0 ? (numPages - 1) * pageSize + lastPageLen[seq] : 0;
    const int prefixLen = max(0, kvLen - numQueries);
    const int causalLimit = causal ? (prefixLen + qLocalPos) : (kvLen - 1);
    const int numValid = causalLimit + 1;

    if (blockIdx.x == 0 && threadIdx.x == 0) row_len[qIdx] = numValid;

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
    void* scores, int32_t* rowLen, int32_t* hist, int32_t* meta,
    int maxKv, int num_splits) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    dim3 grid(num_splits, totalQ);
    int block = 256;
    size_t smem = (size_t)idxNHeads * idxHeadDim * sizeof(__nv_bfloat16)
                + idxNHeads * sizeof(__nv_bfloat16);
    idx_score_kernel<<<grid, block, smem, stream>>>(
        (__nv_bfloat16*)scores, rowLen, (const __nv_bfloat16*)q, (const __nv_bfloat16*)kData,
        (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr,
        scale, idxNHeads, idxHeadDim, pageSize, maxKv, causal);
    glm_topk_from_scores(ctx, out_idx, scores, rowLen, hist, meta,
                         totalQ, maxKv, topk, num_splits);
}

// ---------------------------------------------------------------------------
// Fused two-level prefill indexer: score + coarse/fine histogram top-K.
//
// Replaces the v1 serial heap kernel for prefill. Instead of one CTA per query
// serially scanning all KV tokens, this uses a multi-block grid (numSplits ×
// totalQ) that parallelizes across both the KV and query dimensions. Scores are
// recomputed in each pass (cheap — just dot products) to avoid materializing a
// [totalQ, maxKv] score buffer.
//
// Pipeline (5 kernel launches, all fully parallel, same stream):
//   1. score + coarse hist  — compute score, bin into 1024 coarse buckets
//   2. coarse threshold     — find winning coarse bucket (1 block/query)
//   3. score + fine hist    — recompute score, bin into 64 fine buckets within winner
//   4. fine threshold       — find exact threshold (1 block/query)
//   5. score + gather       — recompute score, write positions above threshold
//
// Scratch: coarseHist [totalQ, 1024] i32 + fineHist [totalQ, 64] i32 + meta [totalQ, 4] i32
//   = ~17 MB for totalQ=4096 (vs 2.1 GB for v2 with scores buffer)
//
// The per-head scoring math matches idx_score_kernel exactly: per-head ReLU,
// weighted sum, final bf16 round. Custom mask support: masked positions are
// skipped entirely (not scored, not histogrammed, not gathered).
// ---------------------------------------------------------------------------

#define IDX_COARSE_BUCKETS 1024
#define IDX_FINE_BUCKETS 64   // IDX_NBUCKET / IDX_COARSE_BUCKETS

// Shared device helper: compute indexer score for one KV position.
// q_s / w_s are in shared memory (loaded by the calling block).
// Returns bf16-rounded score. Only lane 0 produces a valid result.
static __device__ __forceinline__ __nv_bfloat16 idx_compute_score(
    const __nv_bfloat16* __restrict__ q_s,   // [idxNHeads, idxHeadDim] in smem
    const __nv_bfloat16* __restrict__ w_s,   // [idxNHeads] in smem
    const __nv_bfloat16* __restrict__ kData,
    const int32_t* __restrict__ pageIndices,
    int pageStart, int pos, int pageSize,
    int idxNHeads, int idxHeadDim, float scale,
    int warp, int lane)
{
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
    return __float2bfloat16(acc);
}

// Shared device helper: check if a KV position is masked out.
static __device__ __forceinline__ bool idx_is_masked(
    int pos, int qLocalPos,
    const uint8_t* mask_ptr, int mask_kv_len_val, int mask_prefix_len)
{
    if (!mask_ptr) return false;
    if (pos < mask_prefix_len) return false;
    int mask_offset = qLocalPos * mask_kv_len_val + (pos - mask_prefix_len);
    return !((mask_ptr[mask_offset >> 3] >> (mask_offset & 7)) & 1);
}

// Shared device helper: load query info (seq, causalLimit, mask) for a given qIdx.
struct IdxQueryInfo {
    int seq, qLocalPos, numQueries;
    int pageStart, numPages, kvLen;
    int causalLimit, numValid;
    const uint8_t* mask_ptr;
    int mask_kv_len_val, mask_prefix_len;
};

static __device__ IdxQueryInfo idx_load_query(
    int qIdx,
    const int32_t* __restrict__ qoIndptr,
    const int32_t* __restrict__ pageIndptr,
    const int32_t* __restrict__ lastPageLen,
    int pageSize, int causal,
    const uint8_t* custom_mask, const int32_t* mask_indptr, const int32_t* mask_kv_len)
{
    IdxQueryInfo info;
    int seq = 0;
    while (qoIndptr[seq + 1] <= qIdx) seq++;
    info.seq = seq;
    info.qLocalPos = qIdx - qoIndptr[seq];
    info.numQueries = qoIndptr[seq + 1] - qoIndptr[seq];
    info.pageStart = pageIndptr[seq];
    info.numPages = pageIndptr[seq + 1] - info.pageStart;
    info.kvLen = info.numPages > 0 ? (info.numPages - 1) * pageSize + lastPageLen[seq] : 0;
    int prefixLen = max(0, info.kvLen - info.numQueries);
    info.causalLimit = causal ? (prefixLen + info.qLocalPos) : (info.kvLen - 1);
    info.numValid = info.causalLimit + 1;

    info.mask_ptr = nullptr;
    info.mask_kv_len_val = 0;
    info.mask_prefix_len = 0;
    if (custom_mask && mask_indptr) {
        info.mask_ptr = custom_mask + mask_indptr[seq];
        info.mask_kv_len_val = mask_kv_len ? mask_kv_len[seq] : info.numQueries;
        info.mask_prefix_len = max(0, info.kvLen - info.mask_kv_len_val);
    }
    return info;
}

// Shared device helper: load q and weights into shared memory.
static __device__ void idx_load_qw(
    __nv_bfloat16* q_s, __nv_bfloat16* w_s,
    const __nv_bfloat16* q, const __nv_bfloat16* weights,
    int qIdx, int idxNHeads, int idxHeadDim)
{
    for (int i = threadIdx.x; i < idxNHeads * idxHeadDim; i += blockDim.x)
        q_s[i] = q[(size_t)qIdx * idxNHeads * idxHeadDim + i];
    for (int i = threadIdx.x; i < idxNHeads; i += blockDim.x)
        w_s[i] = weights[qIdx * idxNHeads + i];
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Pass 1: Fused score + coarse histogram
// Grid: (numSplits, totalQ)  Block: 256
// Each block processes a slice of KV positions for one query, computes scores,
// and atomicAdds into a 1024-bucket coarse histogram.
// ---------------------------------------------------------------------------
__global__ void idx_prefill_coarse_hist_kernel(
    int32_t* __restrict__ coarseHist,    // [totalQ, IDX_COARSE_BUCKETS]
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kData,
    const __nv_bfloat16* __restrict__ weights,
    const int32_t* __restrict__ pageIndices, const int32_t* __restrict__ pageIndptr,
    const int32_t* __restrict__ lastPageLen, const int32_t* __restrict__ qoIndptr,
    float scale, int idxNHeads, int idxHeadDim, int pageSize, int causal,
    const uint8_t* __restrict__ custom_mask,
    const int32_t* __restrict__ mask_indptr, const int32_t* __restrict__ mask_kv_len)
{
    const int qIdx = blockIdx.y;
    IdxQueryInfo info = idx_load_query(qIdx, qoIndptr, pageIndptr, lastPageLen,
                                       pageSize, causal, custom_mask, mask_indptr, mask_kv_len);
    int32_t* hist = coarseHist + (size_t)qIdx * IDX_COARSE_BUCKETS;

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* w_s = q_s + idxNHeads * idxHeadDim;
    idx_load_qw(q_s, w_s, q, weights, qIdx, idxNHeads, idxHeadDim);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpsPerBlock = blockDim.x >> 5;
    for (int pos = blockIdx.x * warpsPerBlock + warp; pos < info.numValid;
         pos += gridDim.x * warpsPerBlock) {
        if (idx_is_masked(pos, info.qLocalPos, info.mask_ptr,
                          info.mask_kv_len_val, info.mask_prefix_len))
            continue;
        __nv_bfloat16 score = idx_compute_score(q_s, w_s, kData, pageIndices,
                                                info.pageStart, pos, pageSize,
                                                idxNHeads, idxHeadDim, scale, warp, lane);
        if (lane == 0) {
            int key = bf16_key(&score);
            int coarseKey = key / IDX_FINE_BUCKETS;
            atomicAdd(&hist[coarseKey], 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Coarse threshold
// Grid: totalQ  Block: 256
// Scans 1024 coarse buckets from highest to lowest, finds the winning bucket
// where cumulative count >= topk. Also initializes out_idx to -1 and handles
// the identity case (total non-masked count <= topk).
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

    // Sum all buckets to get total non-masked count
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

    // Identity case: all non-masked positions fit in topk
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
    // Each thread owns 4 consecutive buckets; thread 255 owns the highest band.
    // Thread t owns buckets [t*PER, (t+1)*PER). We scan from t=255 down to t=0.
    __shared__ long s_cum;
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
        s_above = cum;  // count in bands strictly above winner
    }
    __syncthreads();

    int winner = s_winner;
    long above = s_above;

    // Refine within winning thread's band
    if (t == winner) {
        long cum = above;
        long numAbove = above;
        int tau = t * PER;  // lowest bucket in band (fallback)
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
// Pass 3: Fused score + fine histogram
// Grid: (numSplits, totalQ)  Block: 256
// Recomputes scores. Only bins positions whose coarse bucket == winning bucket
// into 64 fine buckets. Positions in higher coarse buckets are already counted.
// ---------------------------------------------------------------------------
__global__ void idx_prefill_fine_hist_kernel(
    int32_t* __restrict__ fineHist,     // [totalQ, IDX_FINE_BUCKETS]
    const int32_t* __restrict__ meta,   // [totalQ, 4] — reads coarse_tau from pass 2
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kData,
    const __nv_bfloat16* __restrict__ weights,
    const int32_t* __restrict__ pageIndices, const int32_t* __restrict__ pageIndptr,
    const int32_t* __restrict__ lastPageLen, const int32_t* __restrict__ qoIndptr,
    float scale, int idxNHeads, int idxHeadDim, int pageSize, int causal,
    const uint8_t* __restrict__ custom_mask,
    const int32_t* __restrict__ mask_indptr, const int32_t* __restrict__ mask_kv_len)
{
    const int qIdx = blockIdx.y;
    int32_t coarseTau = meta[(size_t)qIdx * 4];
    if (coarseTau < 0) return;  // identity case — skip

    IdxQueryInfo info = idx_load_query(qIdx, qoIndptr, pageIndptr, lastPageLen,
                                       pageSize, causal, custom_mask, mask_indptr, mask_kv_len);
    int32_t* hist = fineHist + (size_t)qIdx * IDX_FINE_BUCKETS;

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* w_s = q_s + idxNHeads * idxHeadDim;
    idx_load_qw(q_s, w_s, q, weights, qIdx, idxNHeads, idxHeadDim);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpsPerBlock = blockDim.x >> 5;
    for (int pos = blockIdx.x * warpsPerBlock + warp; pos < info.numValid;
         pos += gridDim.x * warpsPerBlock) {
        if (idx_is_masked(pos, info.qLocalPos, info.mask_ptr,
                          info.mask_kv_len_val, info.mask_prefix_len))
            continue;
        __nv_bfloat16 score = idx_compute_score(q_s, w_s, kData, pageIndices,
                                                info.pageStart, pos, pageSize,
                                                idxNHeads, idxHeadDim, scale, warp, lane);
        if (lane == 0) {
            int key = bf16_key(&score);
            int coarseKey = key / IDX_FINE_BUCKETS;
            if (coarseKey == coarseTau) {
                int fineKey = key % IDX_FINE_BUCKETS;
                atomicAdd(&hist[fineKey], 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 4: Fine threshold
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

    // 64 buckets, 256 threads — first 64 threads each handle 1 bucket
    const int t = threadIdx.x;
    __shared__ long s_cum;
    __shared__ int s_fineTau;
    __shared__ long s_numAboveFine;

    if (t == 0) {
        long cum = numAboveCoarse;
        long numAbove = numAboveCoarse;
        int fineTau = 0;
        // Scan from highest fine bucket (63) to lowest (0)
        for (int k = IDX_FINE_BUCKETS - 1; k >= 0; k--) {
            long c = fh[k];
            if (cum + c >= topk) { fineTau = k; numAbove = cum; break; }
            cum += c;
        }
        // Exact threshold key = coarseTau * 64 + fineTau
        m[0] = coarseTau * IDX_FINE_BUCKETS + fineTau;
        m[1] = (int)(topk - numAbove);  // tie_take
        m[2] = 0;                        // out_count (for gather)
        m[3] = 0;                        // tie_count (for gather)
    }
}

// ---------------------------------------------------------------------------
// Pass 5: Fused score + gather
// Grid: (numSplits, totalQ)  Block: 256
// Recomputes scores. Writes positions with key > tau to output (atomicAdd
// compaction). For key == tau, takes up to tie_take positions. In identity
// mode (tau == -1), takes all non-masked positions.
// ---------------------------------------------------------------------------
__global__ void idx_prefill_gather_kernel(
    int32_t* __restrict__ out_idx,      // [totalQ, topk]
    int32_t* __restrict__ meta,         // [totalQ, 4]
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kData,
    const __nv_bfloat16* __restrict__ weights,
    const int32_t* __restrict__ pageIndices, const int32_t* __restrict__ pageIndptr,
    const int32_t* __restrict__ lastPageLen, const int32_t* __restrict__ qoIndptr,
    float scale, int idxNHeads, int idxHeadDim, int pageSize, int topk, int causal,
    const uint8_t* __restrict__ custom_mask,
    const int32_t* __restrict__ mask_indptr, const int32_t* __restrict__ mask_kv_len)
{
    const int qIdx = blockIdx.y;
    int32_t* m = meta + (size_t)qIdx * 4;
    int tau = m[0];
    int tieTake = m[1];

    IdxQueryInfo info = idx_load_query(qIdx, qoIndptr, pageIndptr, lastPageLen,
                                       pageSize, causal, custom_mask, mask_indptr, mask_kv_len);
    int32_t* out = out_idx + (size_t)qIdx * topk;

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* w_s = q_s + idxNHeads * idxHeadDim;
    idx_load_qw(q_s, w_s, q, weights, qIdx, idxNHeads, idxHeadDim);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpsPerBlock = blockDim.x >> 5;
    for (int pos = blockIdx.x * warpsPerBlock + warp; pos < info.numValid;
         pos += gridDim.x * warpsPerBlock) {
        if (idx_is_masked(pos, info.qLocalPos, info.mask_ptr,
                          info.mask_kv_len_val, info.mask_prefix_len))
            continue;

        if (tau < 0) {
            // Identity mode: take all non-masked positions
            if (lane == 0) {
                int p = atomicAdd(&m[2], 1);
                if (p < topk) out[p] = pos;
            }
        } else {
            __nv_bfloat16 score = idx_compute_score(q_s, w_s, kData, pageIndices,
                                                    info.pageStart, pos, pageSize,
                                                    idxNHeads, idxHeadDim, scale, warp, lane);
            if (lane == 0) {
                int key = bf16_key(&score);
                if (key > tau) {
                    int p = atomicAdd(&m[2], 1);
                    if (p < topk) out[p] = pos;
                } else if (key == tau) {
                    int t = atomicAdd(&m[3], 1);
                    if (t < tieTake) {
                        int p = atomicAdd(&m[2], 1);
                        if (p < topk) out[p] = pos;
                    }
                }
            }
        }
    }
}

// Host function: orchestrates the 5-pass fused two-level prefill indexer.
void glm_indexer_score_topk_prefill(GlmCtx* ctx, int32_t* out_idx,
    const void* q, const void* kData, const void* weights,
    const int32_t* pageIndices, const int32_t* pageIndptr,
    const int32_t* lastPageLen, const int32_t* qoIndptr,
    float scale, int totalQ, int idxNHeads, int idxHeadDim,
    int pageSize, int topk, int causal,
    const uint8_t* custom_mask, const int32_t* mask_indptr, const int32_t* mask_kv_len,
    int32_t* coarseHist, int32_t* fineHist, int32_t* meta,
    int numSplits) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);

    size_t smem = (size_t)idxNHeads * idxHeadDim * sizeof(__nv_bfloat16)
                + idxNHeads * sizeof(__nv_bfloat16);
    dim3 grid(numSplits, totalQ);
    int block = 256;

    // Zero histograms and meta
    cudaMemsetAsync(coarseHist, 0, (size_t)totalQ * IDX_COARSE_BUCKETS * sizeof(int32_t), stream);
    cudaMemsetAsync(fineHist, 0, (size_t)totalQ * IDX_FINE_BUCKETS * sizeof(int32_t), stream);
    cudaMemsetAsync(meta, 0, (size_t)totalQ * 4 * sizeof(int32_t), stream);

    // Pass 1: score + coarse hist
    idx_prefill_coarse_hist_kernel<<<grid, block, smem, stream>>>(
        coarseHist, (const __nv_bfloat16*)q, (const __nv_bfloat16*)kData,
        (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr,
        scale, idxNHeads, idxHeadDim, pageSize, causal,
        custom_mask, mask_indptr, mask_kv_len);

    // Pass 2: coarse threshold (1 block per query)
    idx_prefill_coarse_threshold_kernel<<<totalQ, 256, 0, stream>>>(
        coarseHist, meta, out_idx, topk);

    // Pass 3: score + fine hist (only within winning coarse bucket)
    idx_prefill_fine_hist_kernel<<<grid, block, smem, stream>>>(
        fineHist, meta, (const __nv_bfloat16*)q, (const __nv_bfloat16*)kData,
        (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr,
        scale, idxNHeads, idxHeadDim, pageSize, causal,
        custom_mask, mask_indptr, mask_kv_len);

    // Pass 4: fine threshold (1 block per query)
    idx_prefill_fine_threshold_kernel<<<totalQ, 256, 0, stream>>>(
        fineHist, meta, topk);

    // Pass 5: score + gather
    idx_prefill_gather_kernel<<<grid, block, smem, stream>>>(
        out_idx, meta, (const __nv_bfloat16*)q, (const __nv_bfloat16*)kData,
        (const __nv_bfloat16*)weights, pageIndices, pageIndptr, lastPageLen, qoIndptr,
        scale, idxNHeads, idxHeadDim, pageSize, topk, causal,
        custom_mask, mask_indptr, mask_kv_len);
}
