// ---------------------------------------------------------------------------
// Context Parallelism: Online Softmax Merge Kernel
//
// Merges partial attention outputs from multiple KV sequence shards using
// FlashInfer's state_t::merge (online softmax correction).
//
// Each shard produces:
//   - partial_v_out: [B, input_n_heads, head_dim] BF16 (after v_expand)
//   - partial_lse:   [B, num_heads] F32 (base-2 log-sum-exp, all heads)
//
// The merge computes:
//   For each (batch, head):
//     state_t accumulates across shards: merge(v_i, lse_i, d=1.0)
//     Then normalize: v /= d
//
// Uses FlashInfer's state_t<vec_size>::merge() for the FP32 merge math.
// BF16 input -> FP32 cast_load, merge in FP32, cast_store -> BF16 output.
//
// Head-grouped merge: when shard_n_heads < num_heads, only processes and
// outputs heads [head_offset, head_offset + shard_n_heads). Input v_out uses
// input_n_heads stride (specifies how many heads per input shard; equals
// shard_n_heads for column-parallel v_proj, or num_heads for replicated v_proj
// producing Row-parallel output). Input lse uses num_heads stride (full head
// layout). Output uses shard_n_heads stride, producing contiguous
// [B * shard_n_heads * head_dim] (Row-parallel layout).
// When shard_n_heads == num_heads, head_offset == 0, and input_n_heads == num_heads,
// equivalent to full merge.
// ---------------------------------------------------------------------------

#include "glm_ops.h"
#include "glm_p2p_common.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

#include "flashinfer/attention/state.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "flashinfer/math.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

namespace {

constexpr int CP_MAX_SHARDS = 16;

template <int VEC_SIZE, int BDX, int NUM_SHARDS, typename LseLoader, typename VLoader>
__device__ __forceinline__ void cp_merge_one_pair(
    int tid,
    int b, int h, int num_heads,
    int local_h, int shard_n_heads,
    float* s_lse,
    LseLoader load_lse,
    VLoader load_v,
    __nv_bfloat16* merged_v_out,
    float* merged_lse,
    int rank_offset = 0)
{
    constexpr int head_dim = VEC_SIZE * BDX;

    load_lse(s_lse, b, h, num_heads);
    __syncthreads();

    if (tid < BDX) {
        flashinfer::state_t<VEC_SIZE> st;
        st.init();

        // Stagger shard read order by rank so all GPUs don't hammer the same
        // peer's P2P buffer simultaneously (see p2p_allreduce_oneshot_kernel).
        #pragma unroll
        for (int ss = 0; ss < NUM_SHARDS; ++ss) {
            int s = ss + rank_offset;
            if (s >= NUM_SHARDS) s -= NUM_SHARDS;
            flashinfer::vec_t<float, VEC_SIZE> v;
            load_v(v, s, b, h, head_dim, tid);
            st.merge(v, s_lse[s], 1.0f);
        }

        st.normalize();
        st.o.cast_store(merged_v_out + (b * shard_n_heads + local_h) * head_dim + tid * VEC_SIZE);

        if (merged_lse != nullptr && tid == 0) {
            merged_lse[b * shard_n_heads + local_h] = st.get_lse();
        }
    }
}

struct CPMergeParams {
    const void*  v_ptrs[CP_MAX_SHARDS];
    const float* lse_ptrs[CP_MAX_SHARDS];
};

template <int VEC_SIZE, int BDX, int NUM_SHARDS>
__global__ void __launch_bounds__(BDX, 1)
cp_merge_kernel(
    CPMergeParams params,
    __nv_bfloat16* __restrict__ merged_v_out,
    float* __restrict__ merged_lse,
    int batch_size,
    int num_heads,
    int shard_n_heads,
    int head_offset,
    int input_n_heads)
{
    int tid = threadIdx.x;
    int bh = blockIdx.x;
    int b = bh / shard_n_heads;
    int local_h = bh % shard_n_heads;
    int h = local_h + head_offset;

    if (b >= batch_size) return;

    __shared__ float s_lse[CP_MAX_SHARDS];

    auto load_lse = [&](float* slse, int b_, int h_, int nh) {
        if (tid < NUM_SHARDS) {
            slse[tid] = params.lse_ptrs[tid][b_ * nh + h_];
        }
    };

    auto load_v = [&](flashinfer::vec_t<float, VEC_SIZE>& v, int s, int b_, int h_, int hd, int tid_) {
        const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(params.v_ptrs[s]);
        v.cast_load(v_ptr + (b_ * input_n_heads + h_) * hd + tid_ * VEC_SIZE);
    };

    cp_merge_one_pair<VEC_SIZE, BDX, NUM_SHARDS>(
        tid, b, h, num_heads, local_h, shard_n_heads, s_lse,
        load_lse, load_v,
        merged_v_out, merged_lse);
}

template <int VEC_SIZE, int BDX>
void launch_cp_merge(
    CPMergeParams& params,
    __nv_bfloat16* merged_v_out,
    float* merged_lse,
    int num_shards,
    int batch_size,
    int num_heads,
    int shard_n_heads,
    int head_offset,
    int input_n_heads,
    cudaStream_t stream)
{
    int grid = batch_size * shard_n_heads;
    switch (num_shards) {
        case 1:
            cp_merge_kernel<VEC_SIZE, BDX, 1><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        case 2:
            cp_merge_kernel<VEC_SIZE, BDX, 2><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        case 3:
            cp_merge_kernel<VEC_SIZE, BDX, 3><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        case 4:
            cp_merge_kernel<VEC_SIZE, BDX, 4><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        case 8:
            cp_merge_kernel<VEC_SIZE, BDX, 8><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        case 16:
            cp_merge_kernel<VEC_SIZE, BDX, 16><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        default:
            fprintf(stderr, "launch_cp_merge: unsupported num_shards=%d (must be 1-4, 8, or 16)\n", num_shards);
            break;
    }
}

} // namespace

extern "C" {

void glm_context_parallel_merge(
    GlmCtx* ctx,
    const void* const* partial_v_outs,
    const float* const* partial_lses,
    int num_shards,
    void* merged_v_out,
    float* merged_lse,
    int batch_size,
    int num_heads,
    int v_head_dim)
{
    glm_context_parallel_merge_heads(ctx, partial_v_outs, partial_lses,
        num_shards, merged_v_out, merged_lse,
        batch_size, num_heads, num_heads, 0, num_heads, v_head_dim);
}

void glm_context_parallel_merge_heads(
    GlmCtx* ctx,
    const void* const* partial_v_outs,
    const float* const* partial_lses,
    int num_shards,
    void* merged_v_out,
    float* merged_lse,
    int batch_size,
    int num_heads,
    int shard_n_heads,
    int head_offset,
    int input_n_heads,
    int v_head_dim)
{
    cudaSetDevice(ctx->device_id);

    if (num_shards < 1 || num_shards > CP_MAX_SHARDS) {
        fprintf(stderr, "glm_context_parallel_merge_heads: num_shards=%d out of range [1, %d]\n",
                num_shards, CP_MAX_SHARDS);
        return;
    }
    if (head_offset < 0 || shard_n_heads < 1 || head_offset + shard_n_heads > num_heads) {
        fprintf(stderr, "glm_context_parallel_merge_heads: head_offset=%d + shard_n_heads=%d > num_heads=%d\n",
                head_offset, shard_n_heads, num_heads);
        return;
    }
    if (input_n_heads < shard_n_heads || input_n_heads > num_heads) {
        fprintf(stderr, "glm_context_parallel_merge_heads: input_n_heads=%d out of range [shard_n_heads=%d, num_heads=%d]\n",
                input_n_heads, shard_n_heads, num_heads);
        return;
    }

    CPMergeParams params;
    memset(&params, 0, sizeof(params));
    for (int i = 0; i < num_shards; ++i) {
        params.v_ptrs[i] = partial_v_outs[i];
        params.lse_ptrs[i] = partial_lses[i];
    }

    // Dispatch based on v_head_dim: VEC_SIZE=4, BDX = v_head_dim / VEC_SIZE
    // Supported: 32 (8 threads), 64 (16 threads), 128 (32 threads),
    //            256 (64 threads), 512 (128 threads)
    switch (v_head_dim) {
        case 32:
            launch_cp_merge<4, 8>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, GLM_STREAM(ctx));
            break;
        case 64:
            launch_cp_merge<4, 16>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, GLM_STREAM(ctx));
            break;
        case 128:
            launch_cp_merge<4, 32>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, GLM_STREAM(ctx));
            break;
        case 256:
            launch_cp_merge<4, 64>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, GLM_STREAM(ctx));
            break;
        case 512:
            launch_cp_merge<4, 128>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, GLM_STREAM(ctx));
            break;
        default:
            fprintf(stderr, "glm_context_parallel_merge_heads: unsupported v_head_dim=%d "
                    "(must be 32, 64, 128, 256, or 512)\n", v_head_dim);
            break;
    }
}

} // extern "C"

// ---------------------------------------------------------------------------
// CP Merge Tree: smem-staged online softmax merge using cp.async.bulk
//
// Merges up to 16 partial attention outputs using the online softmax trick,
// staged through shared memory via cp.async.bulk (like sum_pointers_smem_kernel).
// Supports in-place operation (output may alias one of the inputs) since all
// reads go through smem before any writes.
//
// Each block processes HEADS_PER_BLOCK heads for one batch element.
// Thread 0 issues cp.async.bulk for all N peers' v_out + lse tiles.
// After barrier, each warp processes one head: loads lse from smem, computes
// max_lse and d, then merges v_head_dim elements across all shards.
// ---------------------------------------------------------------------------

#include <cuda/barrier>
#include <cuda/ptx>

constexpr int CP_TREE_MAX_SHARDS = 16;

template <int VEC_SIZE, int BDX>
__global__ void __launch_bounds__(BDX, 2)
cp_merge_tree_kernel(
    const __nv_bfloat16* v0,  const __nv_bfloat16* v1,  const __nv_bfloat16* v2,  const __nv_bfloat16* v3,
    const __nv_bfloat16* v4,  const __nv_bfloat16* v5,  const __nv_bfloat16* v6,  const __nv_bfloat16* v7,
    const __nv_bfloat16* v8,  const __nv_bfloat16* v9,  const __nv_bfloat16* v10, const __nv_bfloat16* v11,
    const __nv_bfloat16* v12, const __nv_bfloat16* v13, const __nv_bfloat16* v14, const __nv_bfloat16* v15,
    const float* lse0,  const float* lse1,  const float* lse2,  const float* lse3,
    const float* lse4,  const float* lse5,  const float* lse6,  const float* lse7,
    const float* lse8,  const float* lse9,  const float* lse10, const float* lse11,
    const float* lse12, const float* lse13, const float* lse14, const float* lse15,
    __nv_bfloat16* output_v,
    float* output_lse,
    int N,
    int64_t numel,
    int batch_size,
    int num_heads,
    int v_head_dim,
    int heads_per_block,
    int v_peer_stride,
    int lse_peer_stride)
{
    (void)numel; (void)heads_per_block; (void)v_peer_stride; (void)lse_peer_stride; (void)v_head_dim;
    constexpr int head_dim = VEC_SIZE * BDX;
    constexpr int v_shard_bytes = head_dim * sizeof(__nv_bfloat16);

    constexpr int NUM_SHARDS = CP_TREE_MAX_SHARDS;
    auto block = cg::this_thread_block();
    auto thread = cg::this_thread();
    int tid = threadIdx.x;

    int64_t bh = blockIdx.x;
    int b = (int)(bh / num_heads);
    int h = (int)(bh % num_heads);

    if (b >= batch_size) return;

    const __nv_bfloat16* v_ptrs[CP_TREE_MAX_SHARDS] = {
        v0, v1, v2, v3, v4, v5, v6, v7,
        v8, v9, v10, v11, v12, v13, v14, v15
    };
    const float* lse_ptrs[CP_TREE_MAX_SHARDS] = {
        lse0, lse1, lse2, lse3, lse4, lse5, lse6, lse7,
        lse8, lse9, lse10, lse11, lse12, lse13, lse14, lse15
    };

    extern __shared__ char smem[];
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem);
    float* smem_lse = reinterpret_cast<float*>(smem + N * v_shard_bytes);

    int64_t v_offset = (int64_t)(b * num_heads + h) * head_dim;
    int64_t lse_offset = (int64_t)b * num_heads + h;

    #pragma unroll
    for (int s = 0; s < NUM_SHARDS; s++) {
        if (s >= N) break;
        cg::memcpy_async(block,
            smem_v + s * head_dim,
            v_ptrs[s] + v_offset,
            v_shard_bytes);
    }

    if (tid < N) {
        cg::memcpy_async(thread, &smem_lse[tid], lse_ptrs[tid] + lse_offset, sizeof(float));
    }

    cg::wait(block);

    if (tid < BDX) {
        flashinfer::state_t<VEC_SIZE> st;
        st.init();

        #pragma unroll
        for (int s = 0; s < NUM_SHARDS; s++) {
            if (s >= N) break;
            flashinfer::vec_t<float, VEC_SIZE> v;
            v.cast_load(smem_v + s * head_dim + tid * VEC_SIZE);
            st.merge(v, smem_lse[s], 1.0f);
        }

        st.normalize();
        st.o.cast_store(output_v + v_offset + tid * VEC_SIZE);

        if (output_lse != nullptr && tid == 0)
            output_lse[lse_offset] = st.get_lse();
    }
}

static void launch_cp_merge_tree(
    const __nv_bfloat16* v_ptrs[16],
    const float* lse_ptrs[16],
    int num_shards,
    __nv_bfloat16* output_v,
    float* output_lse,
    int64_t numel,
    int batch_size,
    int num_heads,
    int v_head_dim,
    cudaStream_t stream)
{
    int64_t total_heads = (int64_t)batch_size * num_heads;
    int grid = (int)total_heads;
    if (grid > 65535) grid = 65535;

    #define LAUNCH_CP_TREE(VEC_SIZE, BDX) \
        cp_merge_tree_kernel<VEC_SIZE, BDX><<<grid, BDX, num_shards * (BDX * VEC_SIZE) * sizeof(__nv_bfloat16) + num_shards * sizeof(float), stream>>>( \
            v_ptrs[0], v_ptrs[1], v_ptrs[2], v_ptrs[3], \
            v_ptrs[4], v_ptrs[5], v_ptrs[6], v_ptrs[7], \
            v_ptrs[8], v_ptrs[9], v_ptrs[10], v_ptrs[11], \
            v_ptrs[12], v_ptrs[13], v_ptrs[14], v_ptrs[15], \
            lse_ptrs[0], lse_ptrs[1], lse_ptrs[2], lse_ptrs[3], \
            lse_ptrs[4], lse_ptrs[5], lse_ptrs[6], lse_ptrs[7], \
            lse_ptrs[8], lse_ptrs[9], lse_ptrs[10], lse_ptrs[11], \
            lse_ptrs[12], lse_ptrs[13], lse_ptrs[14], lse_ptrs[15], \
            output_v, output_lse, num_shards, numel, batch_size, num_heads, v_head_dim, \
            0, 0, 0)

    switch (v_head_dim) {
        case 32:  LAUNCH_CP_TREE(4, 8); break;
        case 64:  LAUNCH_CP_TREE(4, 16); break;
        case 128: LAUNCH_CP_TREE(4, 32); break;
        case 256: LAUNCH_CP_TREE(4, 64); break;
        case 512: LAUNCH_CP_TREE(4, 128); break;
        default:
            fprintf(stderr, "launch_cp_merge_tree: unsupported v_head_dim=%d (must be 32, 64, 128, 256, or 512)\n", v_head_dim);
            break;
    }

    #undef LAUNCH_CP_TREE
}

extern "C" {

void glm_cp_merge_tree(
    GlmCtx* ctx,
    const void* v0,  const void* v1,  const void* v2,  const void* v3,
    const void* v4,  const void* v5,  const void* v6,  const void* v7,
    const void* v8,  const void* v9,  const void* v10, const void* v11,
    const void* v12, const void* v13, const void* v14, const void* v15,
    const float* lse0,  const float* lse1,  const float* lse2,  const float* lse3,
    const float* lse4,  const float* lse5,  const float* lse6,  const float* lse7,
    const float* lse8,  const float* lse9,  const float* lse10, const float* lse11,
    const float* lse12, const float* lse13, const float* lse14, const float* lse15,
    int num_shards,
    void* output_v,
    float* output_lse,
    int64_t numel,
    int batch_size,
    int num_heads,
    int v_head_dim)
{
    cudaSetDevice(ctx->device_id);

    if (num_shards < 1 || num_shards > CP_TREE_MAX_SHARDS) {
        fprintf(stderr, "glm_cp_merge_tree: num_shards=%d out of range [1, %d]\n",
                num_shards, CP_TREE_MAX_SHARDS);
        return;
    }

    const __nv_bfloat16* v_ptrs[16] = {
        reinterpret_cast<const __nv_bfloat16*>(v0),
        reinterpret_cast<const __nv_bfloat16*>(v1),
        reinterpret_cast<const __nv_bfloat16*>(v2),
        reinterpret_cast<const __nv_bfloat16*>(v3),
        reinterpret_cast<const __nv_bfloat16*>(v4),
        reinterpret_cast<const __nv_bfloat16*>(v5),
        reinterpret_cast<const __nv_bfloat16*>(v6),
        reinterpret_cast<const __nv_bfloat16*>(v7),
        reinterpret_cast<const __nv_bfloat16*>(v8),
        reinterpret_cast<const __nv_bfloat16*>(v9),
        reinterpret_cast<const __nv_bfloat16*>(v10),
        reinterpret_cast<const __nv_bfloat16*>(v11),
        reinterpret_cast<const __nv_bfloat16*>(v12),
        reinterpret_cast<const __nv_bfloat16*>(v13),
        reinterpret_cast<const __nv_bfloat16*>(v14),
        reinterpret_cast<const __nv_bfloat16*>(v15),
    };
    const float* lse_ptr_arr[16] = {
        lse0, lse1, lse2, lse3, lse4, lse5, lse6, lse7,
        lse8, lse9, lse10, lse11, lse12, lse13, lse14, lse15
    };

    launch_cp_merge_tree(
        v_ptrs, lse_ptr_arr, num_shards,
        reinterpret_cast<__nv_bfloat16*>(output_v), output_lse,
        numel, batch_size, num_heads, v_head_dim,
        GLM_STREAM(ctx));
}

} // extern "C"
