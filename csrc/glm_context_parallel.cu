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

#include <cuda/barrier>
#include <cuda/ptx>

constexpr int CP_TREE_MAX_SHARDS = 16;

template <int NUM_SHARDS, int VEC_SIZE, int BDX>
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
        v0, v1, v2, v3, v4, v5, v6, v7,
        v8, v9, v10, v11, v12, v13, v14, v15
    };
    const float* lse_ptrs_orig[CP_TREE_MAX_SHARDS] = {
        lse0, lse1, lse2, lse3, lse4, lse5, lse6, lse7,
        lse8, lse9, lse10, lse11, lse12, lse13, lse14, lse15
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
    const __nv_bfloat16* v_ptrs[16],
    const float* lse_ptrs[16],
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
            v_ptrs[8], v_ptrs[9], v_ptrs[10], v_ptrs[11], \
            v_ptrs[12], v_ptrs[13], v_ptrs[14], v_ptrs[15], \
            lse_ptrs[0], lse_ptrs[1], lse_ptrs[2], lse_ptrs[3], \
            lse_ptrs[4], lse_ptrs[5], lse_ptrs[6], lse_ptrs[7], \
            lse_ptrs[8], lse_ptrs[9], lse_ptrs[10], lse_ptrs[11], \
            lse_ptrs[12], lse_ptrs[13], lse_ptrs[14], lse_ptrs[15], \
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
        case 16: DISPATCH_VHEAD_DIM(16); break;
        default:
            fprintf(stderr, "launch_cp_merge_tree: unsupported num_shards=%d (must be 1, 2, 3, 4, 8, or 16)\n", num_shards);
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
        shard_n_heads, head_offset, input_n_heads,
        GLM_STREAM(ctx));
}

} // extern "C"
