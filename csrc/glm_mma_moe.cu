#include "glm_ops.h"
#include "glm_nvfp4.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

namespace {

using namespace nvcuda::wmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARPS_PER_CTA = 4;
constexpr int TM = WMMA_M;
constexpr int TN = WMMA_N * WARPS_PER_CTA;
constexpr int TK = WMMA_K;
constexpr int CTA_SIZE = WARPS_PER_CTA * 32;
constexpr int QUANT_GROUP = 16;

__device__ __forceinline__ void uint4_to_bf16x8(
    const uint4& v, __nv_bfloat16 out[8]) {
    auto* h = reinterpret_cast<const __nv_bfloat16*>(&v);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = h[i];
}

__global__ void histogram_kernel(
    const int* __restrict__ expert_ids,
    int count,
    int* __restrict__ expert_counts,
    int num_experts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        int eid = expert_ids[i];
        if (eid >= 0 && eid < num_experts) {
            atomicAdd(&expert_counts[eid], 1);
        }
    }
}

__global__ void prefix_sum_kernel(
    const int* __restrict__ expert_counts,
    int* __restrict__ expert_offsets,
    int num_experts)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        expert_offsets[0] = 0;
        for (int i = 0; i < num_experts; i++) {
            expert_offsets[i + 1] = expert_offsets[i] + expert_counts[i];
        }
    }
}

__global__ void scatter_input_kernel(
    const int* __restrict__ expert_ids,
    int count,
    int top_k,
    const __nv_bfloat16* __restrict__ input,
    int K,
    __nv_bfloat16* __restrict__ sorted_input,
    int* __restrict__ sorted_to_original,
    int* __restrict__ expert_offsets)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int eid = expert_ids[i];
    int pos = atomicAdd(&expert_offsets[eid], 1);

    int bid = i / top_k;
    const __nv_bfloat16* src = input + (size_t)bid * K;
    __nv_bfloat16* dst = sorted_input + (size_t)pos * K;

    int num_uint4 = K / 8;
    const uint4* src_v4 = reinterpret_cast<const uint4*>(src);
    uint4* dst_v4 = reinterpret_cast<uint4*>(dst);
    for (int k = 0; k < num_uint4; k++) {
        dst_v4[k] = src_v4[k];
    }
    for (int k = num_uint4 * 8; k < K; k++) {
        dst[k] = src[k];
    }

    sorted_to_original[pos] = i;
}

__global__ void restore_offsets_kernel(
    int* __restrict__ expert_offsets,
    const int* __restrict__ expert_counts,
    int num_experts)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_experts) {
        expert_offsets[e] -= expert_counts[e];
    }
}

__global__ void unscatter_output_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ sorted_output,
    int N,
    const int* __restrict__ sorted_to_original,
    int count)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;

    int orig = sorted_to_original[pos];
    const __nv_bfloat16* src = sorted_output + (size_t)pos * N;
    __nv_bfloat16* dst = output + (size_t)orig * N;

    int num_uint4 = N / 8;
    const uint4* src_v4 = reinterpret_cast<const uint4*>(src);
    uint4* dst_v4 = reinterpret_cast<uint4*>(dst);
    for (int k = 0; k < num_uint4; k++) {
        dst_v4[k] = src_v4[k];
    }
    for (int k = num_uint4 * 8; k < N; k++) {
        dst[k] = src[k];
    }
}

__global__ void __launch_bounds__(CTA_SIZE, 4)
grouped_nvfp4_wmma_kernel(
    __nv_bfloat16* __restrict__ sorted_output,
    const __nv_bfloat16* __restrict__ sorted_input,
    int K,
    const uint8_t* const* __restrict__ weight_ptrs,
    const __nv_fp8_e4m3* const* __restrict__ scale_ptrs,
    const float* const* __restrict__ scale2_ptrs,
    const int* __restrict__ expert_offsets,
    int num_experts,
    int N)
{
    int num_n_tiles = (N + TN - 1) / TN;
    int expert_id = blockIdx.x / num_n_tiles;
    int n_tile = blockIdx.x % num_n_tiles;

    if (expert_id >= num_experts) return;

    int M_e = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
    if (M_e == 0) return;

    int n_start = n_tile * TN;
    int n_valid = min(TN, N - n_start);
    if (n_valid <= 0) return;

    int warp_id = threadIdx.x / 32;
    int n_warp_start = n_start + warp_id * WMMA_N;
    int n_warp_valid = min(WMMA_N, N - n_warp_start);
    (void)n_warp_valid;

    const __nv_bfloat16* expert_input = sorted_input + (size_t)expert_offsets[expert_id] * K;
    __nv_bfloat16* expert_output = sorted_output + (size_t)expert_offsets[expert_id] * N;

    float scale_2_val = *scale2_ptrs[expert_id];
    const uint8_t* weight_base = weight_ptrs[expert_id];
    const __nv_fp8_e4m3* scale_base = scale_ptrs[expert_id];
    int num_k_groups = K / QUANT_GROUP;

    extern __shared__ uint8_t smem_buf[];
    __nv_bfloat16* smem_a     = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* smem_b     = smem_a + TM * TK;
    uint8_t*        smem_fp4   = reinterpret_cast<uint8_t*>(smem_b + TK * TN);
    __nv_fp8_e4m3* smem_scale = reinterpret_cast<__nv_fp8_e4m3*>(smem_fp4 + TN * (TK / 2));
    float*          smem_c    = reinterpret_cast<float*>(smem_scale + TN);

    int num_k_steps = K / TK;

    for (int m_start = 0; m_start < M_e; m_start += TM) {
        int m_valid = min(TM, M_e - m_start);

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        for (int k_step = 0; k_step < num_k_steps; k_step++) {
            int k_start = k_step * TK;

            for (int i = threadIdx.x; i < TM * TK; i += CTA_SIZE) {
                int m = i / TK;
                int k = i % TK;
                int row = m_start + m;
                __nv_bfloat16 val = __float2bfloat16(0.0f);
                if (row < M_e) {
                    val = expert_input[(size_t)row * K + k_start + k];
                }
                smem_a[m * TK + k] = val;
            }

            int fp4_bytes = n_valid * (TK / 2);
            for (int i = threadIdx.x; i < fp4_bytes; i += CTA_SIZE) {
                int n = i / (TK / 2);
                int k_packed = i % (TK / 2);
                int global_n = n_start + n;
                if (global_n < N) {
                    smem_fp4[i] = weight_base[(size_t)global_n * (K / 2) + k_start / 2 + k_packed];
                } else {
                    smem_fp4[i] = 0;
                }
            }

            for (int i = threadIdx.x; i < n_valid; i += CTA_SIZE) {
                int global_n = n_start + i;
                if (global_n < N) {
                    smem_scale[i] = scale_base[(size_t)global_n * num_k_groups + k_start / QUANT_GROUP];
                } else {
                    smem_scale[i] = static_cast<__nv_fp8_e4m3>(0);
                }
            }

            __syncthreads();

            int total_bf16 = n_valid * TK;
            for (int i = threadIdx.x; i < total_bf16; i += CTA_SIZE) {
                int n = i / TK;
                int k = i % TK;
                int k_packed = k / 2;
                uint8_t packed_byte = smem_fp4[n * (TK / 2) + k_packed];

                float2 f2 = fp4x2_to_float2(packed_byte);
                float fval = (k % 2 == 0) ? f2.x : f2.y;

                float block_scale = static_cast<float>(smem_scale[n]);
                float scaled_val = fval * block_scale * scale_2_val;

                smem_b[k + n * TK] = __float2bfloat16(scaled_val);
            }

            __syncthreads();

            if (n_warp_valid > 0) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;

                load_matrix_sync(a_frag, smem_a, TK);
                load_matrix_sync(b_frag, smem_b + warp_id * WMMA_N * TK, TK);

                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }

        if (n_warp_valid > 0) {
            store_matrix_sync(smem_c + warp_id * WMMA_N, c_frag, TN, mem_row_major);
        }

        __syncthreads();

        for (int i = threadIdx.x; i < m_valid * n_valid; i += CTA_SIZE) {
            int m = i / n_valid;
            int n = i % n_valid;
            float val = smem_c[m * TN + n];
            int row = m_start + m;
            int col = n_start + n;
            if (row < M_e && col < N) {
                expert_output[(size_t)row * N + col] = __float2bfloat16(val);
            }
        }

        __syncthreads();
    }
}

} // namespace

extern "C" {

size_t glm_mma_moe_workspace_size(int count, int N, int K, int num_experts) {
    size_t sorted_input = (size_t)count * K * 2;
    size_t sorted_output = (size_t)count * N * 2;
    size_t expert_counts = (size_t)num_experts * 4;
    size_t expert_offsets = (size_t)(num_experts + 1) * 4;
    size_t sorted_to_original = (size_t)count * 4;
    return sorted_input + sorted_output + expert_counts + expert_offsets + sorted_to_original;
}

void glm_nvfp4_mul_mat_id_grouped_mma(GlmCtx* ctx, void* output, const void* input,
                                        const void* const* weight_ptrs,
                                        const void* const* scale_ptrs,
                                        const void* const* scale2_ptrs,
                                        const int* expert_ids, int top_k,
                                        int count, int N, int K,
                                        int num_experts, void* workspace) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    if (count == 0 || N == 0 || K == 0) return;

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    size_t offset = 0;

    __nv_bfloat16* sorted_input = reinterpret_cast<__nv_bfloat16*>(ws + offset);
    offset += (size_t)count * K * 2;

    __nv_bfloat16* sorted_output = reinterpret_cast<__nv_bfloat16*>(ws + offset);
    offset += (size_t)count * N * 2;

    int* expert_counts = reinterpret_cast<int*>(ws + offset);
    offset += (size_t)num_experts * 4;

    int* expert_offsets = reinterpret_cast<int*>(ws + offset);
    offset += (size_t)(num_experts + 1) * 4;

    int* sorted_to_original = reinterpret_cast<int*>(ws + offset);

    int block_size = 256;

    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);

    int grid_size = (count + block_size - 1) / block_size;
    histogram_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_ids, count, expert_counts, num_experts);

    prefix_sum_kernel<<<1, 1, 0, stream>>>(
        expert_counts, expert_offsets, num_experts);

    grid_size = (count + block_size - 1) / block_size;
    scatter_input_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_ids, count, top_k,
        reinterpret_cast<const __nv_bfloat16*>(input), K,
        sorted_input, sorted_to_original, expert_offsets);

    grid_size = (num_experts + block_size - 1) / block_size;
    restore_offsets_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_offsets, expert_counts, num_experts);

    int num_n_tiles = (N + TN - 1) / TN;
    int total_ctas = num_experts * num_n_tiles;

    size_t smem_size = TM * TK * 2 +
                       TK * TN * 2 +
                       TN * (TK / 2) +
                       TN +
                       TM * TN * 4;

    grouped_nvfp4_wmma_kernel<<<total_ctas, CTA_SIZE, smem_size, stream>>>(
        sorted_output, sorted_input, K,
        reinterpret_cast<const uint8_t* const*>(weight_ptrs),
        reinterpret_cast<const __nv_fp8_e4m3* const*>(scale_ptrs),
        reinterpret_cast<const float* const*>(scale2_ptrs),
        expert_offsets, num_experts, N);

    grid_size = (count + block_size - 1) / block_size;
    unscatter_output_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output),
        sorted_output, N, sorted_to_original, count);
}

} // extern "C"
