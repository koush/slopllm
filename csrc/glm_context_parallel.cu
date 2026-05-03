// ---------------------------------------------------------------------------
// Context Parallelism: Online Softmax Merge Kernel
//
// Merges partial attention outputs from multiple KV sequence shards using
// FlashInfer's state_t::merge (online softmax correction).
//
// Each shard produces:
//   - partial_v_out: [B, H, head_dim] BF16 (after v_expand)
//   - partial_lse:   [B, H] F32 (base-2 log-sum-exp)
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

namespace {

constexpr int CP_MAX_SHARDS = 16;

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
    int num_heads)
{
    constexpr int head_dim = VEC_SIZE * BDX;
    int tid = threadIdx.x;
    int bh = blockIdx.x;
    int b = bh / num_heads;
    int h = bh % num_heads;

    if (b >= batch_size) return;

    __shared__ float s_lse[CP_MAX_SHARDS];

    if (tid < NUM_SHARDS) {
        s_lse[tid] = params.lse_ptrs[tid][b * num_heads + h];
    }
    __syncthreads();

    flashinfer::state_t<VEC_SIZE> st;
    st.init();

    #pragma unroll
    for (int s = 0; s < NUM_SHARDS; ++s) {
        const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(params.v_ptrs[s]);
        flashinfer::vec_t<float, VEC_SIZE> v;
        v.cast_load(v_ptr + (b * num_heads + h) * head_dim + tid * VEC_SIZE);
        st.merge(v, s_lse[s], 1.0f);
    }

    st.normalize();

    st.o.cast_store(merged_v_out + (b * num_heads + h) * head_dim + tid * VEC_SIZE);

    if (merged_lse != nullptr && tid == 0) {
        merged_lse[b * num_heads + h] = st.get_lse();
    }
}

template <int VEC_SIZE, int BDX>
void launch_cp_merge(
    CPMergeParams& params,
    __nv_bfloat16* merged_v_out,
    float* merged_lse,
    int num_shards,
    int batch_size,
    int num_heads,
    cudaStream_t stream)
{
    int grid = batch_size * num_heads;
    switch (num_shards) {
        case 1:
            cp_merge_kernel<VEC_SIZE, BDX, 1><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
            break;
        case 2:
            cp_merge_kernel<VEC_SIZE, BDX, 2><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
            break;
        case 3:
            cp_merge_kernel<VEC_SIZE, BDX, 3><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
            break;
        case 4:
            cp_merge_kernel<VEC_SIZE, BDX, 4><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
            break;
        case 8:
            cp_merge_kernel<VEC_SIZE, BDX, 8><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
            break;
        case 16:
            cp_merge_kernel<VEC_SIZE, BDX, 16><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
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
    cudaSetDevice(ctx->device_id);

    if (num_shards < 1 || num_shards > CP_MAX_SHARDS) {
        fprintf(stderr, "glm_context_parallel_merge: num_shards=%d out of range [1, %d]\n",
                num_shards, CP_MAX_SHARDS);
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
                num_shards, batch_size, num_heads, GLM_STREAM(ctx));
            break;
        case 64:
            launch_cp_merge<4, 16>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, GLM_STREAM(ctx));
            break;
        case 128:
            launch_cp_merge<4, 32>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, GLM_STREAM(ctx));
            break;
        case 256:
            launch_cp_merge<4, 64>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, GLM_STREAM(ctx));
            break;
        case 512:
            launch_cp_merge<4, 128>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, GLM_STREAM(ctx));
            break;
        default:
            fprintf(stderr, "glm_context_parallel_merge: unsupported v_head_dim=%d "
                    "(must be 32, 64, 128, 256, or 512)\n", v_head_dim);
            break;
    }
}

} // extern "C"
