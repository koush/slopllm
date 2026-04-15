#include "glm_ops.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

constexpr int FP8_GEMV_WARP_SIZE = 32;
constexpr int FP8_GEMV_ROWS_PER_BLOCK = 4;
constexpr int FP8_GEMV_BLOCK_SIZE = FP8_GEMV_ROWS_PER_BLOCK * FP8_GEMV_WARP_SIZE;

__global__ void fp8_dequantize_gemv_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_fp8_e4m3* __restrict__ weight,
    const float* __restrict__ scale_inv,
    int M, int N, int K) {

    int num_row_groups = (N + FP8_GEMV_ROWS_PER_BLOCK - 1) / FP8_GEMV_ROWS_PER_BLOCK;
    int m = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;
    int row = row_group * FP8_GEMV_ROWS_PER_BLOCK + threadIdx.x / FP8_GEMV_WARP_SIZE;
    int lane = threadIdx.x % FP8_GEMV_WARP_SIZE;

    if (m >= M || row >= N) return;

    int num_k_blocks = K / 128;
    int n_block = row / 128;
    const __nv_fp8_e4m3* weight_row = weight + (size_t)row * K;
    const __nv_bfloat16* input_row = input + (size_t)m * K;

    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        float scale = scale_inv[n_block * num_k_blocks + kb];
        int k_start = kb << 7;

        #pragma unroll
        for (int ki = lane; ki < 128; ki += FP8_GEMV_WARP_SIZE) {
            int k = k_start + ki;
            float w_val = static_cast<float>(weight_row[k]) * scale;
            float x_val = __bfloat162float(input_row[k]);
            sum += w_val * x_val;
        }
    }

    int remaining_start = num_k_blocks * 128;
    for (int k = remaining_start + lane; k < K; k += FP8_GEMV_WARP_SIZE) {
        int kb = k / 128;
        float scale = scale_inv[n_block * num_k_blocks + kb];
        float w_val = static_cast<float>(weight_row[k]) * scale;
        float x_val = __bfloat162float(input_row[k]);
        sum += w_val * x_val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        output[(size_t)m * N + row] = __float2bfloat16(sum);
    }
}

} // namespace

extern "C" {

void glm_fp8_linear_decode(GlmCtx* ctx, void* bf16_out, const void* bf16_input,
                            const void* fp8_weight, const float* weight_scale,
                            int m, int n, int k) {
    int num_row_groups = (n + FP8_GEMV_ROWS_PER_BLOCK - 1) / FP8_GEMV_ROWS_PER_BLOCK;
    int grid_size = m * num_row_groups;
    fp8_dequantize_gemv_kernel<<<grid_size, FP8_GEMV_BLOCK_SIZE, 0, ctx->stream>>>(
        reinterpret_cast<__nv_bfloat16*>(bf16_out),
        reinterpret_cast<const __nv_bfloat16*>(bf16_input),
        reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
        weight_scale,
        m, n, k);
}

} // extern "C"
