#include "glm_ops.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cutlass/float8.h>

#include <flashinfer/allocator.h>
#include <flashinfer/gemm/gemm_groupwise_sm120.cuh>

#include <cstdio>

namespace {

using Fp8E4M3 = cutlass::float_e4m3_t;
using Bf16 = cutlass::bfloat16_t;

constexpr size_t FP8_GEMM_WORKSPACE_SIZE = 32 * 1024 * 1024;
constexpr int FP8_QUANT_GROUP_SIZE = 128;

__global__ void fp8_quantize_kernel(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output,
    float* scales,
    int m, int k, int num_groups_k) {

    int row = blockIdx.y;
    int group = blockIdx.x;

    if (row >= m || group >= num_groups_k) return;

    const __nv_bfloat16* row_in = input + row * k;
    __nv_fp8_e4m3* row_out = output + row * k;
    int k_start = group * FP8_QUANT_GROUP_SIZE;
    int k_end = min(k_start + FP8_QUANT_GROUP_SIZE, k);

    float amax = 0.0f;
    for (int i = k_start; i < k_end; ++i) {
        float val = __bfloat162float(row_in[i]);
        amax = fmaxf(amax, fabsf(val));
    }

    float scale = amax / 448.0f;
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    // Row-major output: scales[row * num_groups_k + group]
    scales[row * num_groups_k + group] = scale;

    for (int i = k_start + threadIdx.x; i < k_end; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]) * inv_scale;
        val = fmaxf(-448.0f, fminf(448.0f, val));
        row_out[i] = __nv_fp8_e4m3(val);
    }
}

// Transpose a 2D float array from row-major to column-major (in-place not possible)
// src: [rows, cols] row-major -> dst: [rows, cols] column-major
// dst[i + j * rows] = src[i * cols + j]
__global__ void transpose_scales_kernel(
    const float* src, float* dst, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int i = idx / cols;
    int j = idx % cols;
    // Row-major src: src[i * cols + j]
    // Column-major dst: dst[i + j * rows]
    dst[i + j * rows] = src[i * cols + j];
}

// ---------------------------------------------------------------------------
// Fused FP8 dequantize + GEMV kernel for M=1 decode
// Computes: output[j] = sum_k(input[k] * fp8_weight[j,k] * scale_inv[j/128, k/128])
// No activation quantization needed — BF16 input used directly.
// Each warp computes one output row; each block has ROWS_PER_BLOCK warps.
// ---------------------------------------------------------------------------

constexpr int FP8_GEMV_WARP_SIZE = 32;
constexpr int FP8_GEMV_ROWS_PER_BLOCK = 4;
constexpr int FP8_GEMV_BLOCK_SIZE = FP8_GEMV_ROWS_PER_BLOCK * FP8_GEMV_WARP_SIZE;

__global__ void fp8_dequantize_gemv_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_fp8_e4m3* __restrict__ weight,
    const float* __restrict__ scale_inv,
    int N, int K) {

    int row = blockIdx.x * FP8_GEMV_ROWS_PER_BLOCK + threadIdx.x / FP8_GEMV_WARP_SIZE;
    int lane = threadIdx.x % FP8_GEMV_WARP_SIZE;

    if (row >= N) return;

    int num_k_blocks = K / 128;
    int n_block = row / 128;
    const __nv_fp8_e4m3* weight_row = weight + (size_t)row * K;

    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        float scale = scale_inv[n_block * num_k_blocks + kb];
        int k_start = kb << 7;

        #pragma unroll
        for (int ki = lane; ki < 128; ki += FP8_GEMV_WARP_SIZE) {
            int k = k_start + ki;
            float w_val = static_cast<float>(weight_row[k]) * scale;
            float x_val = __bfloat162float(input[k]);
            sum += w_val * x_val;
        }
    }

    int remaining_start = num_k_blocks * 128;
    for (int k = remaining_start + lane; k < K; k += FP8_GEMV_WARP_SIZE) {
        int kb = k / 128;
        float scale = scale_inv[n_block * num_k_blocks + kb];
        float w_val = static_cast<float>(weight_row[k]) * scale;
        float x_val = __bfloat162float(input[k]);
        sum += w_val * x_val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        output[row] = __float2bfloat16(sum);
    }
}

} // namespace

extern "C" {

void glm_fp8_linear_decode(GlmCtx* ctx, void* bf16_out, const void* bf16_input,
                            const void* fp8_weight, const float* weight_scale,
                            int n, int k) {
    int grid_size = (n + FP8_GEMV_ROWS_PER_BLOCK - 1) / FP8_GEMV_ROWS_PER_BLOCK;
    fp8_dequantize_gemv_kernel<<<grid_size, FP8_GEMV_BLOCK_SIZE, 0, ctx->stream>>>(
        reinterpret_cast<__nv_bfloat16*>(bf16_out),
        reinterpret_cast<const __nv_bfloat16*>(bf16_input),
        reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
        weight_scale,
        n, k);
}

void glm_fp8_linear(GlmCtx* ctx, void* bf16_out, const void* fp8_input,
                    const float* act_scale, const void* fp8_weight,
                    const float* weight_scale, void* workspace,
                    size_t workspace_size, int m, int n, int k) {
    auto* A = reinterpret_cast<Fp8E4M3*>(const_cast<void*>(fp8_input));
    auto* B = reinterpret_cast<Fp8E4M3*>(const_cast<void*>(fp8_weight));
    auto* D = reinterpret_cast<Bf16*>(bf16_out);

    int num_groups_k = (k + FP8_QUANT_GROUP_SIZE - 1) / FP8_QUANT_GROUP_SIZE;
    int num_groups_n = (n + FP8_QUANT_GROUP_SIZE - 1) / FP8_QUANT_GROUP_SIZE;

    // CUTLASS expects column-major scale layout:
    //   SFA: [M, K/128] with stride (1, M) = column-major
    //   SFB: [N/128, K/128] with stride (1, N/128) = column-major
    // Our API provides row-major scales, so we need to transpose.
    // Use the workspace buffer for transposed scales (they're small).
    size_t act_scale_bytes = (size_t)m * num_groups_k * sizeof(float);
    size_t weight_scale_bytes = (size_t)num_groups_n * num_groups_k * sizeof(float);
    float* act_scale_t = reinterpret_cast<float*>(workspace);
    float* weight_scale_t = act_scale_t + act_scale_bytes / sizeof(float);

    int act_total = m * num_groups_k;
    int block_act = (act_total + 255) / 256;
    transpose_scales_kernel<<<block_act, 256, 0, ctx->stream>>>(
        act_scale, act_scale_t, m, num_groups_k);

    int wt_total = num_groups_n * num_groups_k;
    int block_wt = (wt_total + 255) / 256;
    transpose_scales_kernel<<<block_wt, 256, 0, ctx->stream>>>(
        weight_scale, weight_scale_t, num_groups_n, num_groups_k);

    // Adjust workspace pointer and size past the transposed scales
    char* ws_ptr = reinterpret_cast<char*>(workspace) +
                   act_scale_bytes + weight_scale_bytes;
    size_t ws_remaining = workspace_size - act_scale_bytes - weight_scale_bytes;

    cudaError_t err = flashinfer::gemm::CutlassGroupwiseScaledGEMMSM120<
        1, 128, 128, false, Fp8E4M3, Bf16>(
        ws_ptr, ws_remaining, A, B,
        act_scale_t, weight_scale_t,
        D, m, n, k, 1, ctx->stream);

    if (err != cudaSuccess) {
        fprintf(stderr, "glm_fp8_linear: FP8 GEMM failed: %s\n", cudaGetErrorString(err));
    }
}

size_t glm_fp8_gemm_workspace_size(int m, int n, int k) {
    return FP8_GEMM_WORKSPACE_SIZE;
}

void glm_fp8_quantize(GlmCtx* ctx, void* fp8_out, float* scales,
                      const void* bf16_input, int m, int k) {
    int num_groups_k = (k + FP8_QUANT_GROUP_SIZE - 1) / FP8_QUANT_GROUP_SIZE;

    dim3 grid(num_groups_k, m);
    dim3 block(256);

    fp8_quantize_kernel<<<grid, block, 0, ctx->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(bf16_input),
        reinterpret_cast<__nv_fp8_e4m3*>(fp8_out),
        scales, m, k, num_groups_k);
}

} // extern "C"
