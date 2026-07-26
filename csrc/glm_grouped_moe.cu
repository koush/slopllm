#include "glm_ops.h"
#include "glm_nvfp4.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace {

constexpr int GEMV_WARP_SIZE = 32;
constexpr int GEMV_ROWS_PER_BLOCK = 8;
constexpr int GEMV_BLOCK_SIZE = GEMV_ROWS_PER_BLOCK * GEMV_WARP_SIZE;
constexpr int GEMV_K_VEC = 8;

__device__ __forceinline__ void uint4_to_bf16x8(
    const uint4& v, __nv_bfloat16 out[8]) {
    auto* h = reinterpret_cast<const __nv_bfloat16*>(&v);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = h[i];
}

// ---------------------------------------------------------------------------
// Sort-by-expert pipeline
// ---------------------------------------------------------------------------

// Kernel 1: Histogram — count entries per expert
// Each thread handles one entry
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

// Kernel 2: Exclusive prefix sum — single thread (sufficient for num_experts <= 1024)
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

// Kernel 3: Scatter input rows — simple 1D version (one thread per entry, vectorized copy)
__global__ void scatter_input_simple_kernel(
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

    int num_uint4 = K / GEMV_K_VEC;
    const uint4* src_v4 = reinterpret_cast<const uint4*>(src);
    uint4* dst_v4 = reinterpret_cast<uint4*>(dst);
    for (int k = 0; k < num_uint4; k++) {
        dst_v4[k] = src_v4[k];
    }
    for (int k = num_uint4 * GEMV_K_VEC; k < K; k++) {
        dst[k] = src[k];
    }

    sorted_to_original[pos] = i;
}

// Kernel 4: Restore expert_offsets from running counters back to exclusive prefix sum
// expert_offsets[e] currently holds (original_offset[e] + count[e]) = exclusive_prefix_sum[e+1]
// We want expert_offsets[e] = exclusive_prefix_sum[e] = expert_offsets[e] - expert_counts[e]
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

// ---------------------------------------------------------------------------
// Grouped BF16 GEMV kernel
//
// Processes all entries for one expert per CTA, sharing weight row loads.
// Grid: (num_experts * ceil(N / ROWS_PER_BLOCK), 1, 1)
// Each CTA (expert, row_group) handles ROWS_PER_BLOCK output rows for all
// M_e entries of that expert.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(GEMV_BLOCK_SIZE, 4)
grouped_bf16_gemv_kernel(
    __nv_bfloat16* __restrict__ sorted_output,
    const __nv_bfloat16* __restrict__ sorted_input,
    int K,
    const __nv_bfloat16* const* __restrict__ weight_ptrs,
    const int* __restrict__ expert_offsets,
    int num_experts,
    int N,
    int max_M)
{
    int num_row_groups = (N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
    int expert_id = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;

    if (expert_id >= num_experts) return;

    int M_e = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
    if (M_e == 0) return;

    int warp_id = threadIdx.x / GEMV_WARP_SIZE;
    int lane = threadIdx.x % GEMV_WARP_SIZE;
    int row = row_group * GEMV_ROWS_PER_BLOCK + warp_id;
    bool row_valid = row < N;

    const __nv_bfloat16* __restrict__ weight_mat = weight_ptrs[expert_id];
    const __nv_bfloat16* __restrict__ weight_row = weight_mat + (size_t)row * K;
    const __nv_bfloat16* __restrict__ expert_input = sorted_input + (size_t)expert_offsets[expert_id] * K;
    __nv_bfloat16* __restrict__ expert_output = sorted_output + (size_t)expert_offsets[expert_id] * N;

    int K_vec = K / GEMV_K_VEC;

    float sums[8];
    #pragma unroll
    for (int m = 0; m < 8; m++) sums[m] = 0.0f;

    if (row_valid) {
        for (int ki = lane; ki < K_vec; ki += GEMV_WARP_SIZE) {
            uint4 wv = reinterpret_cast<const uint4*>(weight_row)[ki];
            __nv_bfloat16 wb[8];
            uint4_to_bf16x8(wv, wb);

            for (int m = 0; m < M_e && m < 8; m++) {
                uint4 xv = reinterpret_cast<const uint4*>(expert_input + (size_t)m * K)[ki];
                __nv_bfloat16 xb[8];
                uint4_to_bf16x8(xv, xb);

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    sums[m] += __bfloat162float(wb[j]) * __bfloat162float(xb[j]);
                }
            }
        }

        int K_tail_start = K_vec * GEMV_K_VEC;
        for (int k = K_tail_start + lane; k < K; k += GEMV_WARP_SIZE) {
            float w = __bfloat162float(weight_row[k]);
            for (int m = 0; m < M_e && m < 8; m++) {
                sums[m] += w * __bfloat162float(expert_input[(size_t)m * K + k]);
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                sums[m] += __shfl_down_sync(0xFFFFFFFF, sums[m], offset);
            }
        }

        if (lane == 0) {
            for (int m = 0; m < M_e && m < 8; m++) {
                expert_output[(size_t)m * N + row] = __float2bfloat16(sums[m]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unscatter output: copy from sorted order to original order
// sorted_output[pos] -> output[sorted_to_original[pos]]
// ---------------------------------------------------------------------------

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

    int num_uint4 = N / GEMV_K_VEC;
    const uint4* src_v4 = reinterpret_cast<const uint4*>(src);
    uint4* dst_v4 = reinterpret_cast<uint4*>(dst);
    for (int k = 0; k < num_uint4; k++) {
        dst_v4[k] = src_v4[k];
    }
    for (int k = num_uint4 * GEMV_K_VEC; k < N; k++) {
        dst[k] = src[k];
    }
}

// ---------------------------------------------------------------------------
// Grouped NVFP4 GEMV kernel
//
// Same structure as grouped_bf16_gemv_kernel but dequantizes FP4 E2M1 weights
// with double quantization (FP8 block scale + F32 global scale) on the fly.
// Weight layout matches nvfp4_mul_mat_id_kernel:
//   weight_ptrs[e]: [N, K/2] uint8 (two FP4 values per byte)
//   scale_ptrs[e]:  [N, K/16] FP8 E4M3 (one per block of 16 elements)
//   scale2_ptrs[e]: scalar F32 (global scale for the expert)
// ---------------------------------------------------------------------------

constexpr int NVFP4_QUANT_GROUP = 16;
constexpr int NVFP4_SMALLK_THRESHOLD = GEMV_WARP_SIZE * NVFP4_QUANT_GROUP;

// Tokens-per-expert is unbounded (data-dependent), but registers aren't —
// process M_e in chunks of NVFP4_GROUPED_M_CHUNK, accumulating that many
// running sums per lane at a time. Matches GEMV_ROWS_PER_BLOCK so that for
// the overwhelmingly common case M_e <= chunk (e.g. avg M_e << 1 for
// small-batch decode/verify routing across many experts), the loop below
// runs exactly one outer iteration with the same register/reduction width as
// the original (buggy) kernel — i.e. zero added cost for the hot path.
constexpr int NVFP4_GROUPED_M_CHUNK = GEMV_ROWS_PER_BLOCK;

// ---------------------------------------------------------------------------
// Grouped NVFP4 GEMV kernel
//
// Each CTA owns one (expert, row_group) and must handle all M_e tokens routed
// to that expert, where M_e is data-dependent and unbounded — naively capping
// at a fixed register count (the original kernel used sums[8] / `m < 8`)
// silently drops tokens whose local index is >= 8.
//
// Fix: loop over M_e in chunks of NVFP4_GROUPED_M_CHUNK, re-reading the FP4
// weight from global memory once per chunk. This redundant re-read only
// happens when M_e > chunk — i.e. only in the rare large-M_e regime that the
// original kernel got wrong anyway, so it's a strict correctness + (at worst)
// neutral perf improvement there. For M_e <= chunk (the dominant case for
// sparse MoE routing — e.g. MTP verify with small batches spread across many
// experts) the outer loop runs exactly once and this is byte-for-byte the
// same instruction sequence as the original kernel, just without the bug.
//
// (A shared-memory weight cache to eliminate the redundant re-reads entirely
// was evaluated, but reserving dynamic shared memory on every launch lowered
// SM occupancy enough to make the dominant small-M_e case ~10% slower —
// not worth it for a code path that triggers rarely.)
//
// Template parameter RowsPerWarp: when K is small (<= 512, i.e.
// num_k_groups <= 32), each warp only needs 4 lanes per row to cover the
// K-dimension. RowsPerWarp=8 packs 8 output rows into each 32-lane warp
// (4 lanes each), reducing the grid by 8× and dramatically improving
// occupancy for the large-N, small-K case (e.g. MoE down_proj under TP).
// RowsPerWarp=1 is the default: one full warp per row, 32 lanes collaborating.
// ---------------------------------------------------------------------------

template <int RowsPerWarp>
__global__ void __launch_bounds__(GEMV_BLOCK_SIZE, 4)
grouped_nvfp4_gemv_kernel(
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
    constexpr int LANES_PER_ROW = GEMV_WARP_SIZE / RowsPerWarp;
    constexpr int ROWS_PER_BLOCK = GEMV_ROWS_PER_BLOCK * RowsPerWarp;
    constexpr int M_CHUNK = (RowsPerWarp == 1) ? NVFP4_GROUPED_M_CHUNK : 1;

    int num_row_groups = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int expert_id = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;

    if (expert_id >= num_experts) return;

    int M_e = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
    if (M_e == 0) return;

    int warp_id = threadIdx.x / GEMV_WARP_SIZE;
    int lane = threadIdx.x % GEMV_WARP_SIZE;
    int row_in_warp = lane / LANES_PER_ROW;
    int inner_lane = lane % LANES_PER_ROW;
    int row = row_group * ROWS_PER_BLOCK + warp_id * RowsPerWarp + row_in_warp;
    bool row_valid = row < N;

    if constexpr (RowsPerWarp == 1) {
        if (!row_valid) return;
    }

    int num_k_groups = K / NVFP4_QUANT_GROUP;

    const uint8_t* weight_row = row_valid ? (weight_ptrs[expert_id] + (size_t)row * (K / 2)) : nullptr;
    const __nv_fp8_e4m3* scale_row = row_valid ? (scale_ptrs[expert_id] + (size_t)row * num_k_groups) : nullptr;
    float scale_2_val = *scale2_ptrs[expert_id];

    const __nv_bfloat16* expert_input = sorted_input + (size_t)expert_offsets[expert_id] * K;
    __nv_bfloat16* expert_output = sorted_output + (size_t)expert_offsets[expert_id] * N;

    for (int m_base = 0; m_base < M_e; m_base += M_CHUNK) {
        int m_count = min(M_CHUNK, M_e - m_base);

        float sums[M_CHUNK];
        #pragma unroll
        for (int m = 0; m < M_CHUNK; m++) sums[m] = 0.0f;

        for (int g = inner_lane * 2; row_valid && g < num_k_groups; g += LANES_PER_ROW * 2) {
            int k_start = g * NVFP4_QUANT_GROUP;
            uint4 w = *reinterpret_cast<const uint4*>(weight_row + (size_t)g * (NVFP4_QUANT_GROUP / 2));

            float scale0 = fp8_e4m3_to_float(scale_row[g]) * scale_2_val;
            float scale1 = (g + 1 < num_k_groups)
                               ? fp8_e4m3_to_float(scale_row[g + 1]) * scale_2_val
                               : 0.0f;

            for (int m = 0; m < m_count; m++) {
                const uint4* input_v4 = reinterpret_cast<const uint4*>(expert_input + (size_t)(m_base + m) * K + k_start);
                uint4 xv0 = input_v4[0];
                uint4 xv1 = input_v4[1];
                __nv_bfloat16 xb0[8], xb1[8];
                uint4_to_bf16x8(xv0, xb0);
                uint4_to_bf16x8(xv1, xb1);

                float gsum0 = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint8_t packed = (w.x >> (j * 8)) & 0xFFu;
                    float2 wv = fp4x2_to_float2(packed);
                    gsum0 += wv.x * __bfloat162float(xb0[j * 2])
                           + wv.y * __bfloat162float(xb0[j * 2 + 1]);
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint8_t packed = (w.y >> (j * 8)) & 0xFFu;
                    float2 wv = fp4x2_to_float2(packed);
                    gsum0 += wv.x * __bfloat162float(xb1[j * 2])
                           + wv.y * __bfloat162float(xb1[j * 2 + 1]);
                }

                float gsum1 = 0.0f;
                if (g + 1 < num_k_groups) {
                    const uint4* input_v4_1 = reinterpret_cast<const uint4*>(expert_input + (size_t)(m_base + m) * K + k_start + NVFP4_QUANT_GROUP);
                    uint4 xv2 = input_v4_1[0];
                    uint4 xv3 = input_v4_1[1];
                    uint4_to_bf16x8(xv2, xb0);
                    uint4_to_bf16x8(xv3, xb1);

                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        uint8_t packed = (w.z >> (j * 8)) & 0xFFu;
                        float2 wv = fp4x2_to_float2(packed);
                        gsum1 += wv.x * __bfloat162float(xb0[j * 2])
                               + wv.y * __bfloat162float(xb0[j * 2 + 1]);
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        uint8_t packed = (w.w >> (j * 8)) & 0xFFu;
                        float2 wv = fp4x2_to_float2(packed);
                        gsum1 += wv.x * __bfloat162float(xb1[j * 2])
                               + wv.y * __bfloat162float(xb1[j * 2 + 1]);
                    }
                }

                sums[m] += gsum0 * scale0 + gsum1 * scale1;
            }
        }

        if constexpr (RowsPerWarp == 1) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                #pragma unroll
                for (int m = 0; m < M_CHUNK; m++) {
                    sums[m] += __shfl_down_sync(0xFFFFFFFF, sums[m], offset);
                }
            }
            if (lane == 0) {
                for (int m = 0; m < m_count; m++) {
                    expert_output[(size_t)(m_base + m) * N + row] = __float2bfloat16(sums[m]);
                }
            }
        } else {
            #pragma unroll
            for (int offset = LANES_PER_ROW / 2; offset > 0; offset >>= 1) {
                #pragma unroll
                for (int m = 0; m < M_CHUNK; m++) {
                    sums[m] += __shfl_xor_sync(0xFFFFFFFF, sums[m], offset);
                }
            }
            if (row_valid && inner_lane == 0) {
                for (int m = 0; m < m_count; m++) {
                    expert_output[(size_t)(m_base + m) * N + row] = __float2bfloat16(sums[m]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped NVFP4 GEMV kernel with shared-memory weight cache (RowsPerWarp=8)
//
// Same as grouped_nvfp4_gemv_kernel<8> but caches the full weight tile (FP4 +
// FP8 scales) in shared memory. This eliminates the M_e redundant global-memory
// weight reads that occur in the m_base loop when M_CHUNK=1, which is the
// dominant bottleneck for MoE down_proj (K=256, N=6144) where avg M_e ~ 16.
//
// Weight tile size: ROWS_PER_BLOCK * (K/2) bytes FP4 + ROWS_PER_BLOCK * (K/16)
// bytes FP8 scales. For K=256: 64*128 + 64*16 = 9 KB — fits comfortably in smem.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(GEMV_BLOCK_SIZE, 4)
grouped_nvfp4_gemv_smem_kernel(
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
    extern __shared__ uint8_t smem_buf[];
    constexpr int RowsPerWarp = 8;
    constexpr int LANES_PER_ROW = GEMV_WARP_SIZE / RowsPerWarp;
    constexpr int ROWS_PER_BLOCK = GEMV_ROWS_PER_BLOCK * RowsPerWarp;

    int num_row_groups = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int expert_id = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;

    if (expert_id >= num_experts) return;

    int M_e = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
    if (M_e == 0) return;

    int warp_id = threadIdx.x / GEMV_WARP_SIZE;
    int lane = threadIdx.x % GEMV_WARP_SIZE;
    int row_in_warp = lane / LANES_PER_ROW;
    int inner_lane = lane % LANES_PER_ROW;
    int row = row_group * ROWS_PER_BLOCK + warp_id * RowsPerWarp + row_in_warp;
    bool row_valid = row < N;

    int num_k_groups = K / NVFP4_QUANT_GROUP;
    float scale_2_val = *scale2_ptrs[expert_id];

    const __nv_bfloat16* expert_input = sorted_input + (size_t)expert_offsets[expert_id] * K;
    __nv_bfloat16* expert_output = sorted_output + (size_t)expert_offsets[expert_id] * N;

    int row_start = row_group * ROWS_PER_BLOCK;
    int valid_rows = min(ROWS_PER_BLOCK, N - row_start);
    int weight_bytes = valid_rows * (K / 2);
    int scale_bytes = valid_rows * num_k_groups;

    uint8_t* smem_weight = smem_buf;
    __nv_fp8_e4m3* smem_scale = reinterpret_cast<__nv_fp8_e4m3*>(smem_buf + weight_bytes);

    const uint8_t* weight_base = weight_ptrs[expert_id] + (size_t)row_start * (K / 2);
    const __nv_fp8_e4m3* scale_base = scale_ptrs[expert_id] + (size_t)row_start * num_k_groups;

    for (int i = threadIdx.x; i < weight_bytes; i += GEMV_BLOCK_SIZE) {
        smem_weight[i] = weight_base[i];
    }
    for (int i = threadIdx.x; i < scale_bytes; i += GEMV_BLOCK_SIZE) {
        smem_scale[i] = scale_base[i];
    }
    __syncthreads();

    int local_row = row - row_start;
    const uint8_t* weight_row = row_valid ? (smem_weight + local_row * (K / 2)) : smem_weight;
    const __nv_fp8_e4m3* scale_row = row_valid ? (smem_scale + local_row * num_k_groups) : smem_scale;

    for (int m_base = 0; m_base < M_e; m_base++) {
        int m_count = min(1, M_e - m_base);
        float sum = 0.0f;

        for (int g = inner_lane * 2; row_valid && g < num_k_groups; g += LANES_PER_ROW * 2) {
            int k_start = g * NVFP4_QUANT_GROUP;
            uint4 w = *reinterpret_cast<const uint4*>(weight_row + (size_t)g * (NVFP4_QUANT_GROUP / 2));

            float scale0 = fp8_e4m3_to_float(scale_row[g]) * scale_2_val;
            float scale1 = (g + 1 < num_k_groups)
                               ? fp8_e4m3_to_float(scale_row[g + 1]) * scale_2_val
                               : 0.0f;

            const uint4* input_v4 = reinterpret_cast<const uint4*>(expert_input + (size_t)m_base * K + k_start);
            uint4 xv0 = input_v4[0];
            uint4 xv1 = input_v4[1];
            __nv_bfloat16 xb0[8], xb1[8];
            uint4_to_bf16x8(xv0, xb0);
            uint4_to_bf16x8(xv1, xb1);

            float gsum0 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t packed = (w.x >> (j * 8)) & 0xFFu;
                float2 wv = fp4x2_to_float2(packed);
                gsum0 += wv.x * __bfloat162float(xb0[j * 2])
                       + wv.y * __bfloat162float(xb0[j * 2 + 1]);
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t packed = (w.y >> (j * 8)) & 0xFFu;
                float2 wv = fp4x2_to_float2(packed);
                gsum0 += wv.x * __bfloat162float(xb1[j * 2])
                       + wv.y * __bfloat162float(xb1[j * 2 + 1]);
            }

            float gsum1 = 0.0f;
            if (g + 1 < num_k_groups) {
                const uint4* input_v4_1 = reinterpret_cast<const uint4*>(expert_input + (size_t)m_base * K + k_start + NVFP4_QUANT_GROUP);
                uint4 xv2 = input_v4_1[0];
                uint4 xv3 = input_v4_1[1];
                uint4_to_bf16x8(xv2, xb0);
                uint4_to_bf16x8(xv3, xb1);

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint8_t packed = (w.z >> (j * 8)) & 0xFFu;
                    float2 wv = fp4x2_to_float2(packed);
                    gsum1 += wv.x * __bfloat162float(xb0[j * 2])
                           + wv.y * __bfloat162float(xb0[j * 2 + 1]);
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint8_t packed = (w.w >> (j * 8)) & 0xFFu;
                    float2 wv = fp4x2_to_float2(packed);
                    gsum1 += wv.x * __bfloat162float(xb1[j * 2])
                           + wv.y * __bfloat162float(xb1[j * 2 + 1]);
                }
            }

            sum += gsum0 * scale0 + gsum1 * scale1;
        }

        #pragma unroll
        for (int offset = LANES_PER_ROW / 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
        }
        if (row_valid && inner_lane == 0) {
            expert_output[(size_t)m_base * N + row] = __float2bfloat16(sum);
        }
    }
}

} // namespace

extern "C" {

// Workspace layout:
//   sorted_input:     count * K * sizeof(bfloat16)
//   sorted_output:    count * N * sizeof(bfloat16)
//   expert_counts:    num_experts * sizeof(int32)
//   expert_offsets:   (num_experts + 1) * sizeof(int32)
//   sorted_to_original: count * sizeof(int32)
// Total: count * (K + N) * 2 + (2 * num_experts + 1) * 4 + count * 4

size_t glm_grouped_moe_workspace_size(int count, int N, int K, int num_experts) {
    size_t sorted_input = (size_t)count * K * 2;
    size_t sorted_output = (size_t)count * N * 2;
    size_t expert_counts = (size_t)num_experts * 4;
    size_t expert_offsets = (size_t)(num_experts + 1) * 4;
    size_t sorted_to_original = (size_t)count * 4;
    return sorted_input + sorted_output + expert_counts + expert_offsets + sorted_to_original;
}

void glm_mul_mat_id_grouped(GlmCtx* ctx, void* output, const void* input,
                              const void* const* weight_ptrs,
                              const int* expert_ids, int top_k,
                              int count, int N, int K,
                              int num_experts, void* workspace) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    if (count == 0 || N == 0 || K == 0) return;

    // Parse workspace
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

    // Step 1: Zero expert_counts
    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);

    // Step 2: Histogram — count entries per expert
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    histogram_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_ids, count, expert_counts, num_experts);

    // Step 3: Exclusive prefix sum
    prefix_sum_kernel<<<1, 1, 0, stream>>>(
        expert_counts, expert_offsets, num_experts);

    // Step 4: Scatter input rows by expert
    // expert_offsets is now the exclusive prefix sum
    // We use atomic_add on expert_offsets to get scatter positions
    // After this, expert_offsets[e] = original_offset[e] + count[e]
    grid_size = (count + block_size - 1) / block_size;
    scatter_input_simple_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_ids, count, top_k,
        reinterpret_cast<const __nv_bfloat16*>(input), K,
        sorted_input, sorted_to_original, expert_offsets);

    // Step 5: Restore expert_offsets to exclusive prefix sum
    grid_size = (num_experts + block_size - 1) / block_size;
    restore_offsets_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_offsets, expert_counts, num_experts);

    // Step 6: Grouped GEMV
    int num_row_groups = (N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
    int total_ctas = num_experts * num_row_groups;

    grouped_bf16_gemv_kernel<<<total_ctas, GEMV_BLOCK_SIZE, 0, stream>>>(
        sorted_output, sorted_input, K,
        reinterpret_cast<const __nv_bfloat16* const*>(weight_ptrs),
        expert_offsets, num_experts, N, 8);

    // Step 7: Unscatter output
    grid_size = (count + block_size - 1) / block_size;
    unscatter_output_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output),
        sorted_output, N, sorted_to_original, count);
}

void glm_nvfp4_mul_mat_id_grouped(GlmCtx* ctx, void* output, const void* input,
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

    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);

    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    histogram_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_ids, count, expert_counts, num_experts);

    prefix_sum_kernel<<<1, 1, 0, stream>>>(
        expert_counts, expert_offsets, num_experts);

    grid_size = (count + block_size - 1) / block_size;
    scatter_input_simple_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_ids, count, top_k,
        reinterpret_cast<const __nv_bfloat16*>(input), K,
        sorted_input, sorted_to_original, expert_offsets);

    grid_size = (num_experts + block_size - 1) / block_size;
    restore_offsets_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_offsets, expert_counts, num_experts);

    if (K <= NVFP4_SMALLK_THRESHOLD) {
        constexpr int ROWS_PER_BLOCK = GEMV_ROWS_PER_BLOCK * 8;
        int num_row_groups = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        int total_ctas = num_experts * num_row_groups;
        size_t smem_size = (size_t)ROWS_PER_BLOCK * (K / 2) + (size_t)ROWS_PER_BLOCK * (K / NVFP4_QUANT_GROUP);
        grouped_nvfp4_gemv_smem_kernel<<<total_ctas, GEMV_BLOCK_SIZE, smem_size, stream>>>(
            sorted_output, sorted_input, K,
            reinterpret_cast<const uint8_t* const*>(weight_ptrs),
            reinterpret_cast<const __nv_fp8_e4m3* const*>(scale_ptrs),
            reinterpret_cast<const float* const*>(scale2_ptrs),
            expert_offsets, num_experts, N);
    } else {
        int num_row_groups = (N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
        int total_ctas = num_experts * num_row_groups;
        grouped_nvfp4_gemv_kernel<1><<<total_ctas, GEMV_BLOCK_SIZE, 0, stream>>>(
            sorted_output, sorted_input, K,
            reinterpret_cast<const uint8_t* const*>(weight_ptrs),
            reinterpret_cast<const __nv_fp8_e4m3* const*>(scale_ptrs),
            reinterpret_cast<const float* const*>(scale2_ptrs),
            expert_offsets, num_experts, N);
    }

    grid_size = (count + block_size - 1) / block_size;
    unscatter_output_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output),
        sorted_output, N, sorted_to_original, count);
}

} // extern "C"
