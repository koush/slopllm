// ---------------------------------------------------------------------------
// Context Parallelism: Online Softmax Merge Tree Kernel
//
// Merges partial attention outputs from multiple KV sequence shards using
// FlashInfer's state_t::merge (online softmax correction).
//
// Each shard produces:
//   - partial_v_out: [B, num_heads, head_dim] BF16 (after v_expand)
//   - partial_lse:   [B, num_heads] F32 (base-2 log-sum-exp)
//
// The merge computes:
//   For each (batch, head):
//     state_t accumulates across shards: merge(v_i, lse_i, d=1.0)
//     Then normalize: v /= d
//
// Uses FlashInfer's state_t<vec_size>::merge() for the FP32 merge math.
// BF16 input -> FP32 cast_load, merge in FP32, cast_store -> BF16 output.
// ---------------------------------------------------------------------------

#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

#include "flashinfer/attention/state.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "flashinfer/math.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

#include <cuda/barrier>
#include <cuda/ptx>

constexpr int CP_TREE_MAX_SHARDS = 8;

template <int NUM_SHARDS, int VEC_SIZE, int BDX>
__global__ void __launch_bounds__(BDX, 2)
cp_merge_tree_kernel(
    const __nv_bfloat16* v0,  const __nv_bfloat16* v1,  const __nv_bfloat16* v2,  const __nv_bfloat16* v3,
    const __nv_bfloat16* v4,  const __nv_bfloat16* v5,  const __nv_bfloat16* v6,  const __nv_bfloat16* v7,
    const float* lse0,  const float* lse1,  const float* lse2,  const float* lse3,
    const float* lse4,  const float* lse5,  const float* lse6,  const float* lse7,
    __nv_bfloat16* output_v,
    float* output_lse,
    int64_t numel,
    int batch_size,
    int num_heads,
    int v_head_dim,
    int heads_per_block,
    int v_peer_stride,
    int lse_peer_stride,
    int shard_n_heads,
    int head_offset,
    int input_n_heads)
{
    (void)numel; (void)heads_per_block; (void)v_peer_stride; (void)lse_peer_stride; (void)v_head_dim;
    constexpr int head_dim = VEC_SIZE * BDX;
    constexpr int v_shard_bytes = head_dim * sizeof(__nv_bfloat16);

    auto block = cg::this_thread_block();
    int tid = threadIdx.x;

    int64_t bh = blockIdx.x;
    int b = (int)(bh / shard_n_heads);
    int local_h = (int)(bh % shard_n_heads);
    int h = local_h + head_offset;

    if (b >= batch_size) return;

    const __nv_bfloat16* v_ptrs_orig[CP_TREE_MAX_SHARDS] = {
        v0, v1, v2, v3, v4, v5, v6, v7
    };
    const float* lse_ptrs_orig[CP_TREE_MAX_SHARDS] = {
        lse0, lse1, lse2, lse3, lse4, lse5, lse6, lse7
    };

    // Rotate pointer order by blockIdx.x so different blocks hit different peers first,
    // distributing P2P load across the fabric.
    const __nv_bfloat16* v_ptrs[CP_TREE_MAX_SHARDS];
    const float* lse_ptrs[CP_TREE_MAX_SHARDS];
    int rot = (int)(blockIdx.x % NUM_SHARDS);
    #pragma unroll
    for (int s = 0; s < NUM_SHARDS; s++) {
        v_ptrs[s] = v_ptrs_orig[(s + rot) % NUM_SHARDS];
        lse_ptrs[s] = lse_ptrs_orig[(s + rot) % NUM_SHARDS];
    }

    extern __shared__ char smem[];
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem);
    float* smem_lse = reinterpret_cast<float*>(smem + NUM_SHARDS * v_shard_bytes);

    int64_t v_in_offset = (int64_t)(b * input_n_heads + h) * head_dim;
    int64_t lse_in_offset = (int64_t)b * num_heads + h;
    int64_t v_out_offset = (int64_t)(b * shard_n_heads + local_h) * head_dim;
    int64_t lse_out_offset = (int64_t)b * shard_n_heads + local_h;

    // Scalar load all lse values directly (no async copy needed for 4 bytes)
    for (int s = tid; s < NUM_SHARDS; s += BDX) {
        smem_lse[s] = lse_ptrs[s][lse_in_offset];
    }
    __syncthreads();

    // Issue all v memcpy_async up front
    #pragma unroll
    for (int s = 0; s < NUM_SHARDS; s++) {
        cg::memcpy_async(block,
            smem_v + s * head_dim,
            v_ptrs[s] + v_in_offset,
            v_shard_bytes);
    }

    // Interleave wait_prior + compute: wait on copies 0..s, then merge shard s.
    // wait_prior<NUM_SHARDS - 1 - s> waits for the first s+1 copies to complete.
    flashinfer::state_t<VEC_SIZE> st;
    st.init();

    auto wait_and_merge = [&] <int s>() {
        if constexpr (s < NUM_SHARDS - 1) {
            cg::wait_prior<NUM_SHARDS - 1 - s>(block);
        } else {
            cg::wait(block);
        }
        flashinfer::vec_t<float, VEC_SIZE> v;
        v.cast_load(smem_v + s * head_dim + tid * VEC_SIZE);
        st.merge(v, smem_lse[s], 1.0f);
    };

    [&] <int... Is>(std::integer_sequence<int, Is...>) {
        (wait_and_merge.template operator()<Is>(), ...);
    }(std::make_integer_sequence<int, NUM_SHARDS>{});

    st.normalize();
    st.o.cast_store(output_v + v_out_offset + tid * VEC_SIZE);

    if (output_lse != nullptr && tid == 0)
        output_lse[lse_out_offset] = st.get_lse();
}

static void launch_cp_merge_tree(
    const __nv_bfloat16* v_ptrs[8],
    const float* lse_ptrs[8],
    int num_shards,
    __nv_bfloat16* output_v,
    float* output_lse,
    int64_t numel,
    int batch_size,
    int num_heads,
    int v_head_dim,
    int shard_n_heads,
    int head_offset,
    int input_n_heads,
    cudaStream_t stream)
{
    int64_t total_heads = (int64_t)batch_size * shard_n_heads;
    int grid = (int)total_heads;

    #define LAUNCH_CP_TREE(NS, VEC_SIZE, BDX) \
        cp_merge_tree_kernel<NS, VEC_SIZE, BDX><<<grid, BDX, NS * (BDX * VEC_SIZE) * sizeof(__nv_bfloat16) + NS * sizeof(float), stream>>>( \
            v_ptrs[0], v_ptrs[1], v_ptrs[2], v_ptrs[3], \
            v_ptrs[4], v_ptrs[5], v_ptrs[6], v_ptrs[7], \
            lse_ptrs[0], lse_ptrs[1], lse_ptrs[2], lse_ptrs[3], \
            lse_ptrs[4], lse_ptrs[5], lse_ptrs[6], lse_ptrs[7], \
            output_v, output_lse, numel, batch_size, num_heads, v_head_dim, \
            0, 0, 0, shard_n_heads, head_offset, input_n_heads)

    #define DISPATCH_VHEAD_DIM(NS) \
        switch (v_head_dim) { \
            case 32:  LAUNCH_CP_TREE(NS, 4, 8); break; \
            case 64:  LAUNCH_CP_TREE(NS, 4, 16); break; \
            case 128: LAUNCH_CP_TREE(NS, 4, 32); break; \
            case 256: LAUNCH_CP_TREE(NS, 4, 64); break; \
            case 512: LAUNCH_CP_TREE(NS, 4, 128); break; \
            default: \
                fprintf(stderr, "launch_cp_merge_tree: unsupported v_head_dim=%d\n", v_head_dim); \
                break; \
        }

    switch (num_shards) {
        case 1:  DISPATCH_VHEAD_DIM(1); break;
        case 2:  DISPATCH_VHEAD_DIM(2); break;
        case 3:  DISPATCH_VHEAD_DIM(3); break;
        case 4:  DISPATCH_VHEAD_DIM(4); break;
        case 8:  DISPATCH_VHEAD_DIM(8); break;
        default:
            fprintf(stderr, "launch_cp_merge_tree: unsupported num_shards=%d (must be 1, 2, 3, 4, or 8)\n", num_shards);
            break;
    }

    #undef LAUNCH_CP_TREE
    #undef DISPATCH_VHEAD_DIM
}

extern "C" {

void glm_cp_merge_tree(
    GlmCtx* ctx,
    const void* v0,  const void* v1,  const void* v2,  const void* v3,
    const void* v4,  const void* v5,  const void* v6,  const void* v7,
    const float* lse0,  const float* lse1,  const float* lse2,  const float* lse3,
    const float* lse4,  const float* lse5,  const float* lse6,  const float* lse7,
    int num_shards,
    void* output_v,
    float* output_lse,
    int64_t numel,
    int batch_size,
    int num_heads,
    int v_head_dim,
    int shard_n_heads,
    int head_offset,
    int input_n_heads)
{
    cudaSetDevice(ctx->device_id);

    if (num_shards < 1 || num_shards > CP_TREE_MAX_SHARDS) {
        fprintf(stderr, "glm_cp_merge_tree: num_shards=%d out of range [1, %d]\n",
                num_shards, CP_TREE_MAX_SHARDS);
        return;
    }

    const __nv_bfloat16* v_ptrs[8] = {
        reinterpret_cast<const __nv_bfloat16*>(v0),
        reinterpret_cast<const __nv_bfloat16*>(v1),
        reinterpret_cast<const __nv_bfloat16*>(v2),
        reinterpret_cast<const __nv_bfloat16*>(v3),
        reinterpret_cast<const __nv_bfloat16*>(v4),
        reinterpret_cast<const __nv_bfloat16*>(v5),
        reinterpret_cast<const __nv_bfloat16*>(v6),
        reinterpret_cast<const __nv_bfloat16*>(v7),
    };
    const float* lse_ptr_arr[8] = {
        lse0, lse1, lse2, lse3, lse4, lse5, lse6, lse7
    };

    launch_cp_merge_tree(
        v_ptrs, lse_ptr_arr, num_shards,
        reinterpret_cast<__nv_bfloat16*>(output_v), output_lse,
        numel, batch_size, num_heads, v_head_dim,
        shard_n_heads, head_offset, input_n_heads,
        GLM_STREAM(ctx));
}

} // extern "C"

// ---------------------------------------------------------------------------
// Push-based CP merge, phase 1: scatter heads to the rank that owns them.
//
// Every rank holds a full-width partial (all num_heads) for its own KV shard,
// but rank r only needs to produce the merged output for its head slice
// [r*shard_n_heads, (r+1)*shard_n_heads). This kernel *writes* each peer's head
// slice into that peer's staging buffer at slot `rank`, so no GPU issues a P2P
// read -- all cross-device traffic is posted writes, same as the other
// write-based collectives in glm_p2p.cu.
//
// Staging layout per rank: [world, batch, shard_n_heads, head_dim] for v and
// [world, batch, shard_n_heads] for lse; slot j receives rank j's contribution.
//
// The grid is peer-major, which is what keeps the posted writes large: a peer's
// destination region (slot `rank`, every batch, every local head, every dim) is
// fully contiguous because `slot` is the outermost staging dimension, and at
// batch=1 the matching source range is contiguous too. So one block streams a
// whole batch*shard_n_heads*head_dim run to one peer with every thread active,
// instead of shattering it into head_dim-sized dribbles. `blocks_per_peer`
// splits that run further so there is enough in flight to fill the links.
//
// Write-hammering control is two-level, mirroring p2p_allgather_row_write:
// destination pointers arrive already rotated by rank (dsts[k] is peer
// (rank + k) % N, done host-side), and the slot is picked from blockIdx.x, so
// concurrent blocks target different peers and rank r's blocks start on peer r.
// At every instant each receiver's inbound link has one sender.
// ---------------------------------------------------------------------------

constexpr int CP_SCATTER_THREADS = 128;

template <int N>
__global__ void __launch_bounds__(CP_SCATTER_THREADS, 4)
cp_merge_scatter_kernel(
    const __nv_bfloat16* __restrict__ local_v,   // [batch, input_n_heads, head_dim]
    const float* __restrict__ local_lse,         // [batch, num_heads]
    __nv_bfloat16* dv0, __nv_bfloat16* dv1, __nv_bfloat16* dv2, __nv_bfloat16* dv3,
    __nv_bfloat16* dv4, __nv_bfloat16* dv5, __nv_bfloat16* dv6, __nv_bfloat16* dv7,
    float* dl0, float* dl1, float* dl2, float* dl3,
    float* dl4, float* dl5, float* dl6, float* dl7,
    int batch_size,
    int shard_n_heads,
    int head_dim,
    int input_n_heads,
    int num_heads,
    int rank,
    int blocks_per_peer)
{
    __nv_bfloat16* dv[CP_TREE_MAX_SHARDS] = { dv0, dv1, dv2, dv3, dv4, dv5, dv6, dv7 };
    float*         dl[CP_TREE_MAX_SHARDS] = { dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7 };

    const int tid   = threadIdx.x;
    const int slot  = (int)(blockIdx.x % N);          // index into the rotated dst arrays
    const int chunk = (int)(blockIdx.x / N);          // which slice of this peer's run
    const int peer  = (rank + slot) % N;              // logical rank owning this head slice

    const int head_stride = shard_n_heads * head_dim;         // per batch row, per peer
    const int64_t n_elem  = (int64_t)batch_size * head_stride; // contiguous at the destination

    // Destination base for this rank's slot in the peer's staging buffer.
    __nv_bfloat16* dst_v = dv[slot] + (int64_t)rank * n_elem;

    // int4 (8 bf16) stores need head_dim % 8 == 0; every offset into src and dst
    // is a whole number of head_dim-sized rows off a 256B-aligned base, so that
    // one check covers alignment on both sides, and a vector never straddles a
    // head boundary. Odd head dims fall back to element stores.
    constexpr int VEC_ELEMS = 8;
    const bool vectorized = (head_dim % VEC_ELEMS) == 0;

    if (vectorized) {
        const int64_t n_vec  = n_elem / VEC_ELEMS;
        const int     hd_vec = head_dim / VEC_ELEMS;
        const int     hs_vec = head_stride / VEC_ELEMS;
        for (int64_t v = (int64_t)chunk * CP_SCATTER_THREADS + tid;
             v < n_vec;
             v += (int64_t)blocks_per_peer * CP_SCATTER_THREADS)
        {
            const int b   = (int)(v / hs_vec);
            const int rem = (int)(v % hs_vec);
            const int lh  = rem / hd_vec;
            const int e   = rem % hd_vec;
            const int64_t src_vec =
                ((int64_t)(b * input_n_heads + peer * shard_n_heads + lh) * head_dim) / VEC_ELEMS + e;
            reinterpret_cast<int4*>(dst_v)[v] = reinterpret_cast<const int4*>(local_v)[src_vec];
        }
    } else {
        for (int64_t i = (int64_t)chunk * CP_SCATTER_THREADS + tid;
             i < n_elem;
             i += (int64_t)blocks_per_peer * CP_SCATTER_THREADS)
        {
            const int b   = (int)(i / head_stride);
            const int rem = (int)(i % head_stride);
            const int lh  = rem / head_dim;
            const int e   = rem % head_dim;
            dst_v[i] = local_v[(int64_t)(b * input_n_heads + peer * shard_n_heads + lh) * head_dim + e];
        }
    }

    // LSE is batch*shard_n_heads floats per peer -- tiny, so the peer's first
    // chunk block carries all of it.
    if (chunk == 0) {
        const int64_t n_lse = (int64_t)batch_size * shard_n_heads;
        float* dst_lse = dl[slot] + (int64_t)rank * n_lse;
        for (int64_t i = tid; i < n_lse; i += CP_SCATTER_THREADS) {
            const int b  = (int)(i / shard_n_heads);
            const int lh = (int)(i % shard_n_heads);
            dst_lse[i] = local_lse[(int64_t)b * num_heads + peer * shard_n_heads + lh];
        }
    }
}

// ---------------------------------------------------------------------------
// Push-based CP merge, phase 2: merge the local staging buffer.
//
// Purely local HBM reads (the barrier after phase 1 guarantees every peer's
// slot has landed), so there is no P2P latency to hide and no smem staging --
// just a straight online-softmax merge across the N slots.
//
// Slot order is fixed 0..N-1 on every rank. state_t::merge is an online
// softmax, so accumulation order changes the rounding; keeping it in rank order
// makes the merged result identical on every GPU and across runs.
// ---------------------------------------------------------------------------

template <int NUM_SHARDS, int VEC_SIZE, int BDX>
__global__ void __launch_bounds__(BDX, 2)
cp_merge_local_kernel(
    const __nv_bfloat16* __restrict__ stage_v,   // [NUM_SHARDS, batch, shard_n_heads, head_dim]
    const float* __restrict__ stage_lse,         // [NUM_SHARDS, batch, shard_n_heads]
    __nv_bfloat16* __restrict__ output_v,        // [batch, shard_n_heads, head_dim]
    float* __restrict__ output_lse,              // [batch, shard_n_heads] (nullable)
    int batch_size,
    int shard_n_heads)
{
    constexpr int head_dim = VEC_SIZE * BDX;
    const int tid = threadIdx.x;
    const int64_t bh = blockIdx.x;
    if ((int)(bh / shard_n_heads) >= batch_size) return;

    const int64_t v_slot_stride   = (int64_t)batch_size * shard_n_heads * head_dim;
    const int64_t lse_slot_stride = (int64_t)batch_size * shard_n_heads;

    flashinfer::state_t<VEC_SIZE> st;
    st.init();

    #pragma unroll
    for (int s = 0; s < NUM_SHARDS; s++) {
        flashinfer::vec_t<float, VEC_SIZE> v;
        v.cast_load(stage_v + s * v_slot_stride + bh * head_dim + tid * VEC_SIZE);
        st.merge(v, stage_lse[s * lse_slot_stride + bh], 1.0f);
    }

    st.normalize();
    st.o.cast_store(output_v + bh * head_dim + tid * VEC_SIZE);

    if (output_lse != nullptr && tid == 0)
        output_lse[bh] = st.get_lse();
}

extern "C" {

void glm_cp_merge_scatter(
    GlmCtx* ctx,
    const void* local_v,
    const float* local_lse,
    void* dv0, void* dv1, void* dv2, void* dv3,
    void* dv4, void* dv5, void* dv6, void* dv7,
    float* dl0, float* dl1, float* dl2, float* dl3,
    float* dl4, float* dl5, float* dl6, float* dl7,
    int world_size,
    int batch_size,
    int shard_n_heads,
    int v_head_dim,
    int input_n_heads,
    int num_heads,
    int rank)
{
    cudaSetDevice(ctx->device_id);

    if (world_size <= 0 || batch_size <= 0 || shard_n_heads <= 0) return;

    // One block per peer, split further so the whole payload can be in flight:
    // size blocks_per_peer to cover a peer's run in a single pass per thread.
    const int64_t units_per_peer = (v_head_dim % 8 == 0)
        ? ((int64_t)batch_size * shard_n_heads * v_head_dim) / 8
        :  (int64_t)batch_size * shard_n_heads * v_head_dim;
    int blocks_per_peer = (int)((units_per_peer + CP_SCATTER_THREADS - 1) / CP_SCATTER_THREADS);
    if (blocks_per_peer < 1)  blocks_per_peer = 1;
    if (blocks_per_peer > 64) blocks_per_peer = 64;
    const int grid = world_size * blocks_per_peer;

    #define LAUNCH_CP_SCATTER(N) \
        cp_merge_scatter_kernel<N><<<grid, CP_SCATTER_THREADS, 0, GLM_STREAM(ctx)>>>( \
            reinterpret_cast<const __nv_bfloat16*>(local_v), local_lse, \
            reinterpret_cast<__nv_bfloat16*>(dv0), reinterpret_cast<__nv_bfloat16*>(dv1), \
            reinterpret_cast<__nv_bfloat16*>(dv2), reinterpret_cast<__nv_bfloat16*>(dv3), \
            reinterpret_cast<__nv_bfloat16*>(dv4), reinterpret_cast<__nv_bfloat16*>(dv5), \
            reinterpret_cast<__nv_bfloat16*>(dv6), reinterpret_cast<__nv_bfloat16*>(dv7), \
            dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, \
            batch_size, shard_n_heads, v_head_dim, input_n_heads, num_heads, rank, \
            blocks_per_peer)

    switch (world_size) {
        case 1: LAUNCH_CP_SCATTER(1); break;
        case 2: LAUNCH_CP_SCATTER(2); break;
        case 4: LAUNCH_CP_SCATTER(4); break;
        case 8: LAUNCH_CP_SCATTER(8); break;
        default:
            fprintf(stderr, "glm_cp_merge_scatter: unsupported world_size=%d (must be 1, 2, 4 or 8)\n",
                    world_size);
            break;
    }

    #undef LAUNCH_CP_SCATTER
}

void glm_cp_merge_local(
    GlmCtx* ctx,
    const void* stage_v,
    const float* stage_lse,
    void* output_v,
    float* output_lse,
    int world_size,
    int batch_size,
    int shard_n_heads,
    int v_head_dim)
{
    cudaSetDevice(ctx->device_id);

    const int grid = batch_size * shard_n_heads;
    if (grid <= 0) return;

    #define LAUNCH_CP_LOCAL(NS, VEC_SIZE, BDX) \
        cp_merge_local_kernel<NS, VEC_SIZE, BDX><<<grid, BDX, 0, GLM_STREAM(ctx)>>>( \
            reinterpret_cast<const __nv_bfloat16*>(stage_v), stage_lse, \
            reinterpret_cast<__nv_bfloat16*>(output_v), output_lse, \
            batch_size, shard_n_heads)

    #define DISPATCH_CP_LOCAL_DIM(NS) \
        switch (v_head_dim) { \
            case 32:  LAUNCH_CP_LOCAL(NS, 4, 8);   break; \
            case 64:  LAUNCH_CP_LOCAL(NS, 4, 16);  break; \
            case 128: LAUNCH_CP_LOCAL(NS, 4, 32);  break; \
            case 256: LAUNCH_CP_LOCAL(NS, 4, 64);  break; \
            case 512: LAUNCH_CP_LOCAL(NS, 4, 128); break; \
            default: \
                fprintf(stderr, "glm_cp_merge_local: unsupported v_head_dim=%d\n", v_head_dim); \
                break; \
        }

    switch (world_size) {
        case 1: DISPATCH_CP_LOCAL_DIM(1); break;
        case 2: DISPATCH_CP_LOCAL_DIM(2); break;
        case 4: DISPATCH_CP_LOCAL_DIM(4); break;
        case 8: DISPATCH_CP_LOCAL_DIM(8); break;
        default:
            fprintf(stderr, "glm_cp_merge_local: unsupported world_size=%d (must be 1, 2, 4 or 8)\n",
                    world_size);
            break;
    }

    #undef LAUNCH_CP_LOCAL
    #undef DISPATCH_CP_LOCAL_DIM
}

} // extern "C"

// ---------------------------------------------------------------------------
// CP Correction Kernel: rescale local v_out by exp2(lse_local - global_lse)
//
// After AllGathering all N ranks' LSEs, each GPU computes the global LSE
// (online softmax merge) and rescales its own v_out so that a simple sum
// across ranks produces the exact merged attention output.
//
//   lses:        [N, B, H] F32 — gathered LSEs from all ranks (base-2)
//   v_out:       [B, H, D] BF16 — local partial attention output (in-place)
//   global_lse:  [B, H] F32 — output, the merged LSE (optional, pass nullptr)
//   rank:        which slice of lses belongs to this GPU
// ---------------------------------------------------------------------------

template <int VEC_SIZE, int BDX>
__global__ void __launch_bounds__(BDX, 4)
cp_correct_attn_out_kernel(
    __nv_bfloat16* __restrict__ v_out,
    const float* __restrict__ lses,
    float* __restrict__ global_lse,
    int batch_size,
    int num_heads,
    int world_size,
    int rank)
{
    constexpr int head_dim = VEC_SIZE * BDX;
    int tid = threadIdx.x;
    int64_t bh = blockIdx.x;
    int b = (int)(bh / num_heads);
    int h = (int)(bh % num_heads);
    if (b >= batch_size) return;

    int64_t lse_base = (int64_t)b * num_heads + h;

    // Load all N LSE values into registers
    float local_lses[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < world_size)
            local_lses[i] = lses[(int64_t)i * batch_size * num_heads + lse_base];
        else
            local_lses[i] = -INFINITY;
    }

    // Compute global LSE: lse_max = max(lse_i), global = log2(sum(2^(lse_i - lse_max))) + lse_max
    float lse_max = -INFINITY;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < world_size && local_lses[i] > lse_max)
            lse_max = local_lses[i];
    }
    if (lse_max == -INFINITY) lse_max = 0.0f;

    float sum_exp2 = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < world_size) {
            float v = local_lses[i] - lse_max;
            if (v > -20.0f)  // exp2(-20) ~ 1e-6, skip very small
                sum_exp2 += exp2f(v);
        }
    }
    float glse = log2f(sum_exp2) + lse_max;

    if (global_lse != nullptr && tid == 0)
        global_lse[lse_base] = glse;

    // Compute rescale factor for this rank
    float factor = exp2f(local_lses[rank] - glse);

    // Rescale v_out in-place
    int64_t v_offset = (int64_t)(b * num_heads + h) * head_dim;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        int idx = tid * VEC_SIZE + i;
        if (idx < head_dim) {
            __nv_bfloat16 val = v_out[v_offset + idx];
            float fv = __bfloat162float(val) * factor;
            v_out[v_offset + idx] = __float2bfloat16(fv);
        }
    }
}

static void launch_cp_correct_attn_out(
    __nv_bfloat16* v_out,
    const float* lses,
    float* global_lse,
    int batch_size,
    int num_heads,
    int v_head_dim,
    int world_size,
    int rank,
    cudaStream_t stream)
{
    int64_t total_heads = (int64_t)batch_size * num_heads;
    int grid = (int)total_heads;

    #define LAUNCH_CP_CORRECT(VEC_SIZE, BDX) \
        cp_correct_attn_out_kernel<VEC_SIZE, BDX><<<grid, BDX, 0, stream>>>( \
            v_out, lses, global_lse, batch_size, num_heads, world_size, rank)

    switch (v_head_dim) {
        case 32:  LAUNCH_CP_CORRECT(4, 8); break;
        case 64:  LAUNCH_CP_CORRECT(4, 16); break;
        case 128: LAUNCH_CP_CORRECT(4, 32); break;
        case 256: LAUNCH_CP_CORRECT(4, 64); break;
        case 512: LAUNCH_CP_CORRECT(4, 128); break;
        default:
            fprintf(stderr, "launch_cp_correct_attn_out: unsupported v_head_dim=%d\n", v_head_dim);
            break;
    }

    #undef LAUNCH_CP_CORRECT
}

extern "C" {

void glm_cp_correct_attn_out(
    GlmCtx* ctx,
    void* v_out,
    const float* lses,
    float* global_lse,
    int batch_size,
    int num_heads,
    int v_head_dim,
    int world_size,
    int rank)
{
    cudaSetDevice(ctx->device_id);
    launch_cp_correct_attn_out(
        reinterpret_cast<__nv_bfloat16*>(v_out),
        lses,
        global_lse,
        batch_size,
        num_heads,
        v_head_dim,
        world_size,
        rank,
        GLM_STREAM(ctx));
}

} // extern "C"
