// ---------------------------------------------------------------------------
// gather_topk_ckv: sparse topk-driven gather of packed CKV tokens into a
// flat-format buffer. Reads only the tokens referenced by any query's topk
// (instead of gathering the ENTIRE KV cache like the prefill path's
// gatherPages) and writes them at the flat_slot given by topk_idx[entry].
// Positions not referenced retain whatever stale data was there — the sparse
// MLA kernel reads only the topk_length-bounded prefix, so stale data is
// never touched.
//
// INPUT CONTRACT (topk_idx): [num_tokens, topk] int32 holding OUTPUT flat
// slots — exactly the format that topk_to_slots produces in its flat mode
// (cp_world_size == 1: slot = kv_token_indptr[seq] + token_pos). In other
// words this is the SAME indexing the downstream sparse MLA kernel uses to
// read the gathered buffer; the same sharedSlots tensor can be fed in
// directly. Invalid entries are encoded as -1 (compaction sentinel from
// topk_to_slots) and are silently skipped.
//
// Two call shapes share the same kernel body:
//   * N=1, cp_world_size=0 — single-GPU / non-CP. No fan-out, no P2P, no CP
//     filter; the warp reads BPT bytes from its local paged cache and writes
//     them once to the single output flat buffer. peers[1..7] are unused.
//     This is the degenerate case — multi-GPU / P2P is NOT required.
//   * N>=1, cp_world_size>0 — context-parallel. Each GPU holds every Nth KV
//     token in its local paged cache. For sparse MLA decode each GPU needs a
//     complete flat-view of the topk tokens selected across all ranks, so
//     each warp reads BPT bytes ONCE and fan-out writes them to all N peer
//     flat buffers (cross-GPU via P2P). After all N ranks run every peer's
//     flat buffer holds the topk union at the correct flat slots.
//
// Each warp processes exactly one (query, k) topk entry:
//   1. seq        = batch_indices[entry / topk]
//      flat_slot  = topk_idx[entry]  (output flat slot, -1 = skip)
//      token_pos  = flat_slot - kv_token_indptr[seq]
//      (reverse of topk_to_slots' flat-mode transform; needed for the CP
//      filter and the per-rank paged slot lookup below.)
//   2. CP filter: if cp_world_size > 0, drop tokens whose global pos doesn't
//      live on this rank (pos % cp_world_size != cp_rank). cp_world_size == 0
//      means "no CP at all" — the filter is skipped and local_pos == token_pos.
//      cp_world_size == 1 is degenerate CP — the filter is taken (and is a
//      no-op since pos % 1 == 0 for all), preserving the same code path as
//      real CP for callers that always set it > 0.
//   3. Map (seq, local_pos) → local paged slot:
//        abs_page  = page_indices[page_indptr[seq] + local_pos / eff_page_size]
//        src_slot  = abs_page * eff_page_size + (local_pos % eff_page_size)
//   4. (flat_slot is taken directly from topk_idx — no offsetting here.)
//   5. Read ONCE — BPT bytes of the token from local_kv_cache[src_slot * BPT]
//      into registers (16-byte int4 vector + byte tail for non-16-aligned BPT)
//   6. Write MANY — fan those same bytes to all N peer flat buffers at
//      flat_pX[flat_slot * BPT]. With N=1 this is just a single write.
//
// The 8-pointer explicit passing convention follows sum_pointers_smem /
// p2p_allgather: N (1..8) is the active count; entries >= N are ignored. For
// the local-mirror test path all 8 pointers may target the same GPU's memory.
//
// DEDUPLICATION (num_tokens > 1)
//
// With num_tokens > 1 the same flat_slot is typically selected by several
// queries — MTP's verification rows are adjacent positions in one sequence, so
// their top-2048 sets are nearly identical. The one-warp-per-entry form above
// then writes identical bytes to identical addresses on all N peers once per
// duplicate, multiplying the P2P traffic by roughly num_tokens.
//
// So for num_tokens > 1 the work is split into three launches:
//
//   1. reset_scratch_kernel  — zero the bitmap and the compaction counter (a
//      kernel rather than cudaMemsetAsync, which measures poorly here).
//   2. mark_unique_kernel    — one THREAD per entry. Does all the scalar
//      preamble (seq / token_pos / CP filter / paged src_slot), then claims the
//      slot with atomicOr on a bitmap. The thread that flips the bit from 0
//      appends {src_slot, flat_slot} to a compacted list via atomicAdd. The CP
//      filter runs BEFORE the claim, so the list holds only slots this rank
//      owns and is already ~1/cp_world_size of the entry count.
//   3. fanout_topk_ckv_kernel — one WARP per UNIQUE entry, grid-striding over
//      the device-side count. Pure copy: src/dst come straight out of the list,
//      no index lookups at all.
//
// Scratch is caller-owned (allocated from the TS workspace with alloc + using,
// which is stream-safe: disposal is deferred until the owning stream is
// released). Nothing here calls cudaMalloc — allocation on a capturing stream
// is illegal and a silent failure would latch bad pointers for the process
// lifetime. Because the workspace recycles blocks, the scratch contents are
// arbitrary on entry and step 1 clears them every call.
//
// num_tokens == 1 keeps the original single-kernel path: a single query's topk
// positions are already distinct, so there is nothing to dedup and the hot
// decode path takes no extra launches.
//
// Graph-capture safe: launch shapes are host-constant and the unique count is
// consumed device-side by a grid-stride loop, never read back.
// ---------------------------------------------------------------------------

#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

__global__ void gather_topk_ckv_kernel(
    uint8_t* __restrict__ flat_p0, uint8_t* __restrict__ flat_p1,
    uint8_t* __restrict__ flat_p2, uint8_t* __restrict__ flat_p3,
    uint8_t* __restrict__ flat_p4, uint8_t* __restrict__ flat_p5,
    uint8_t* __restrict__ flat_p6, uint8_t* __restrict__ flat_p7,
    const uint8_t* __restrict__ local_kv_cache,
    const int32_t* __restrict__ topk_idx,
    const int32_t* __restrict__ batch_indices,
    const int32_t* __restrict__ page_indices,
    const int32_t* __restrict__ page_indptr,
    const int32_t* __restrict__ kv_token_indptr,
    int N,
    int cp_world_size, int cp_rank,
    int eff_page_size, int bpt_bytes,
    int total_entries, int topk)
{
    constexpr int WARP = 32;
    constexpr int WARPS_PER_BLOCK = 4;
    int warp_id = threadIdx.x / WARP;
    int lane    = threadIdx.x % WARP;
    int entry   = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (entry >= total_entries) return;

    int token = entry / topk;
    int seq   = batch_indices[token];

    int flat_slot = topk_idx[entry];
    if (flat_slot < 0) return;
    int token_pos = flat_slot - kv_token_indptr[seq];

    int local_pos;
    if (cp_world_size > 0) {
        if ((uint32_t)token_pos % (uint32_t)cp_world_size != (uint32_t)cp_rank) return;
        local_pos = (token_pos - cp_rank) / cp_world_size;
    } else {
        local_pos = token_pos;
    }

    int page_base   = page_indptr[seq];
    int page_in_seq = local_pos / eff_page_size;
    int offset_in   = local_pos - page_in_seq * eff_page_size;
    int abs_page    = page_indices[page_base + page_in_seq];
    int64_t src_slot  = (int64_t)abs_page * eff_page_size + offset_in;

    uint8_t* peers[8] = {
        flat_p0, flat_p1, flat_p2, flat_p3, flat_p4, flat_p5, flat_p6, flat_p7
    };

    const uint8_t* src = local_kv_cache + src_slot * (int64_t)bpt_bytes;
    const int64_t dst_base = flat_slot * (int64_t)bpt_bytes;

    constexpr int VEC = 16;
    const int vec_count = bpt_bytes / VEC;
    const int tail_start = vec_count * VEC;

    for (int v = lane; v < vec_count; v += WARP) {
        int4 r = *reinterpret_cast<const int4*>(src + v * VEC);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j >= N) break;
            uint8_t* dst = peers[j] + dst_base + v * VEC;
            *reinterpret_cast<int4*>(dst) = r;
        }
    }

    for (int b = tail_start + lane; b < bpt_bytes; b += WARP) {
        uint8_t byte = src[b];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j >= N) break;
            peers[j][dst_base + b] = byte;
        }
    }
}

// Zero the bitmap and the compaction counter. The scratch comes from the
// workspace recycler, so its contents are arbitrary on entry and must be
// cleared every call — the buffer is not persistent across calls.
__global__ void reset_scratch_kernel(uint32_t* __restrict__ bitmap, int words,
                                     int32_t* __restrict__ counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) *counter = 0;
    for (; i < words; i += gridDim.x * blockDim.x) bitmap[i] = 0u;
}

// One thread per (query, k) entry. Survivors of the CP filter race to claim
// their flat_slot; the winner appends the resolved {src_slot, flat_slot}.
__global__ void mark_unique_kernel(
    const int32_t* __restrict__ topk_idx,
    const int32_t* __restrict__ batch_indices,
    const int32_t* __restrict__ page_indices,
    const int32_t* __restrict__ page_indptr,
    const int32_t* __restrict__ kv_token_indptr,
    uint32_t* __restrict__ bitmap,
    int2* __restrict__ unique,
    int32_t* __restrict__ counter,
    int cp_world_size, int cp_rank,
    int eff_page_size,
    int total_entries, int topk)
{
    int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= total_entries) return;

    int flat_slot = topk_idx[entry];
    if (flat_slot < 0) return;

    int seq = batch_indices[entry / topk];
    int token_pos = flat_slot - kv_token_indptr[seq];

    int local_pos;
    if (cp_world_size > 0) {
        if ((uint32_t)token_pos % (uint32_t)cp_world_size != (uint32_t)cp_rank) return;
        local_pos = (token_pos - cp_rank) / cp_world_size;
    } else {
        local_pos = token_pos;
    }

    // Claim before doing the page lookups so losers pay nothing extra.
    uint32_t word = (uint32_t)flat_slot >> 5;
    uint32_t bit  = 1u << ((uint32_t)flat_slot & 31);
    if (atomicOr(&bitmap[word], bit) & bit) return;

    int page_base   = page_indptr[seq];
    int page_in_seq = local_pos / eff_page_size;
    int offset_in   = local_pos - page_in_seq * eff_page_size;
    int abs_page    = page_indices[page_base + page_in_seq];
    int src_slot    = abs_page * eff_page_size + offset_in;

    unique[atomicAdd(counter, 1)] = make_int2(src_slot, flat_slot);
}

// One warp per unique entry, grid-striding over the device-side count. Copy
// body matches gather_topk_ckv_kernel; the index math is already resolved.
__global__ void fanout_topk_ckv_kernel(
    uint8_t* __restrict__ flat_p0, uint8_t* __restrict__ flat_p1,
    uint8_t* __restrict__ flat_p2, uint8_t* __restrict__ flat_p3,
    uint8_t* __restrict__ flat_p4, uint8_t* __restrict__ flat_p5,
    uint8_t* __restrict__ flat_p6, uint8_t* __restrict__ flat_p7,
    const uint8_t* __restrict__ local_kv_cache,
    const int2* __restrict__ unique,
    const int32_t* __restrict__ counter,
    int N, int bpt_bytes)
{
    constexpr int WARP = 32;
    constexpr int WARPS_PER_BLOCK = 4;
    int warp_id = threadIdx.x / WARP;
    int lane    = threadIdx.x % WARP;

    const int count = *counter;
    const int stride = gridDim.x * WARPS_PER_BLOCK;

    uint8_t* peers[8] = {
        flat_p0, flat_p1, flat_p2, flat_p3, flat_p4, flat_p5, flat_p6, flat_p7
    };

    constexpr int VEC = 16;
    const int vec_count = bpt_bytes / VEC;
    const int tail_start = vec_count * VEC;

    for (int i = blockIdx.x * WARPS_PER_BLOCK + warp_id; i < count; i += stride) {
        int2 u = unique[i];
        const uint8_t* src = local_kv_cache + (int64_t)u.x * (int64_t)bpt_bytes;
        const int64_t dst_base = (int64_t)u.y * (int64_t)bpt_bytes;

        for (int v = lane; v < vec_count; v += WARP) {
            int4 r = *reinterpret_cast<const int4*>(src + v * VEC);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (j >= N) break;
                *reinterpret_cast<int4*>(peers[j] + dst_base + v * VEC) = r;
            }
        }

        for (int b = tail_start + lane; b < bpt_bytes; b += WARP) {
            uint8_t byte = src[b];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (j >= N) break;
                peers[j][dst_base + b] = byte;
            }
        }
    }
}

extern "C" {

void glm_gather_topk_ckv(
    GlmCtx* ctx,
    void* flat_p0, void* flat_p1, void* flat_p2, void* flat_p3,
    void* flat_p4, void* flat_p5, void* flat_p6, void* flat_p7,
    void* local_kv_cache,
    int32_t* topk_idx, int32_t* batch_indices,
    int32_t* page_indices, int32_t* page_indptr, int32_t* kv_token_indptr,
    int N, int cp_world_size, int cp_rank,
    int eff_page_size, int bpt_bytes,
    int num_tokens, int topk, int padded_kv_len,
    void* scratch_bitmap, void* scratch_unique, void* scratch_counter)
{
    cudaSetDevice(ctx->device_id);

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int total_entries = num_tokens * topk;
    cudaStream_t stream = GLM_STREAM(ctx);

    // Escape hatch: GLM_GATHER_NO_DEDUP=1 forces the original one-warp-per-entry
    // kernel for every num_tokens, so the dedup pass can be A/B'd against it
    // without a rebuild. Duplicates then write the same bytes to the same
    // addresses again, which is redundant but correct.
    const char* nd = getenv("GLM_GATHER_NO_DEDUP");
    const bool no_dedup = nd && nd[0] == '1';

    if (num_tokens <= 1 || no_dedup) {
        // Single query: topk positions are already distinct, nothing to dedup.
        int grid = (total_entries + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (grid < 1) grid = 1;

        gather_topk_ckv_kernel<<<grid, THREADS, 0, stream>>>(
            (uint8_t*)flat_p0, (uint8_t*)flat_p1, (uint8_t*)flat_p2, (uint8_t*)flat_p3,
            (uint8_t*)flat_p4, (uint8_t*)flat_p5, (uint8_t*)flat_p6, (uint8_t*)flat_p7,
            (const uint8_t*)local_kv_cache,
            topk_idx, batch_indices, page_indices, page_indptr, kv_token_indptr,
            N, cp_world_size, cp_rank, eff_page_size, bpt_bytes,
            total_entries, topk);
        return;
    }

    // Scratch is caller-owned (allocated through the TS workspace) so nothing
    // here ever calls cudaMalloc — allocation on a capturing stream is illegal
    // and used to fail silently, latching null pointers for the process life.
    uint32_t* bitmap  = (uint32_t*)scratch_bitmap;
    int2*     unique  = (int2*)scratch_unique;
    int32_t*  counter = (int32_t*)scratch_counter;

    int bitmap_words = (padded_kv_len + 31) / 32;

    int zg = (bitmap_words + 255) / 256;
    if (zg < 1) zg = 1;
    if (zg > 1024) zg = 1024;
    reset_scratch_kernel<<<zg, 256, 0, stream>>>(bitmap, bitmap_words, counter);

    constexpr int MARK_THREADS = 256;
    int mark_grid = (total_entries + MARK_THREADS - 1) / MARK_THREADS;
    if (mark_grid < 1) mark_grid = 1;
    mark_unique_kernel<<<mark_grid, MARK_THREADS, 0, stream>>>(
        topk_idx, batch_indices, page_indices, page_indptr, kv_token_indptr,
        bitmap, unique, counter,
        cp_world_size, cp_rank, eff_page_size,
        total_entries, topk);

    // Host-constant grid; the kernel grid-strides over the device-side count.
    int fanout_grid = (total_entries + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (fanout_grid > 4096) fanout_grid = 4096;
    if (fanout_grid < 1) fanout_grid = 1;
    fanout_topk_ckv_kernel<<<fanout_grid, THREADS, 0, stream>>>(
        (uint8_t*)flat_p0, (uint8_t*)flat_p1, (uint8_t*)flat_p2, (uint8_t*)flat_p3,
        (uint8_t*)flat_p4, (uint8_t*)flat_p5, (uint8_t*)flat_p6, (uint8_t*)flat_p7,
        (const uint8_t*)local_kv_cache,
        unique, counter,
        N, bpt_bytes);
}

} // extern "C"
