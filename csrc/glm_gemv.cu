#include "glm_ops.h"

#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define CUBLAS(ctx) (*reinterpret_cast<cublasHandle_t*>(&(ctx)->cublas_handle))

namespace {

// ---------------------------------------------------------------------------
// GEMV kernel: optimized for M=1 decode
// Each warp computes one output element. Input tiles are cooperatively loaded
// into shared memory to avoid redundant global reads across warps in a block.
// ---------------------------------------------------------------------------

constexpr int FP8_GEMV_WARP_SIZE = 32;
constexpr int FP8_GEMV_ROWS_PER_BLOCK = 4;
constexpr int FP8_GEMV_BLOCK_SIZE = FP8_GEMV_ROWS_PER_BLOCK * FP8_GEMV_WARP_SIZE;
constexpr int FP8_GEMV_K_TILE = 128;

__global__ void __launch_bounds__(FP8_GEMV_BLOCK_SIZE, 8)
fp8_dequantize_gemv_kernel(
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

    bool valid = m < M && row < N;

    __shared__ __nv_bfloat16 smem_input[FP8_GEMV_K_TILE];

    int num_k_blocks = K / 128;
    const __nv_bfloat16* input_row = input + (size_t)m * K;
    const __nv_fp8_e4m3* weight_row = weight + (size_t)row * K;
    int n_block = row / 128;

    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        int k_start = kb << 7;

        for (int i = threadIdx.x; i < FP8_GEMV_K_TILE; i += FP8_GEMV_BLOCK_SIZE) {
            smem_input[i] = input_row[k_start + i];
        }
        __syncthreads();

        if (valid) {
            float scale = scale_inv[n_block * num_k_blocks + kb];

            #pragma unroll
            for (int ki = lane; ki < FP8_GEMV_K_TILE; ki += FP8_GEMV_WARP_SIZE) {
                float w_val = static_cast<float>(weight_row[k_start + ki]) * scale;
                float x_val = __bfloat162float(smem_input[ki]);
                sum += w_val * x_val;
            }
        }

        __syncthreads();
    }

    int remaining_start = num_k_blocks * 128;
    int remaining = K - remaining_start;
    if (remaining > 0) {
        for (int i = threadIdx.x; i < remaining; i += FP8_GEMV_BLOCK_SIZE) {
            smem_input[i] = input_row[remaining_start + i];
        }
        __syncthreads();

        if (valid) {
            int kb = remaining_start / 128;
            float scale = scale_inv[n_block * num_k_blocks + kb];
            for (int k = remaining_start + lane; k < K; k += FP8_GEMV_WARP_SIZE) {
                int ki = k - remaining_start;
                float w_val = static_cast<float>(weight_row[k]) * scale;
                float x_val = __bfloat162float(smem_input[ki]);
                sum += w_val * x_val;
            }
        }

        __syncthreads();
    }

    if (valid) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane == 0) {
            output[(size_t)m * N + row] = __float2bfloat16(sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Smem-staged GEMM kernel: optimized for M>1 (batch decode / prefill)
// Each block computes an M_TILE x N_TILE output tile.
// Weight and input tiles are staged to shared memory for reuse across M rows.
// ---------------------------------------------------------------------------

constexpr int FP8_GEMM_M_TILE = 16;
constexpr int FP8_GEMM_N_TILE = 32;
constexpr int FP8_GEMM_K_TILE = 128;
constexpr int FP8_GEMM_BLOCK_DIM = FP8_GEMM_M_TILE * FP8_GEMM_N_TILE;
constexpr int FP8_GEMM_WEIGHT_PAD = 4;

__global__ void __launch_bounds__(FP8_GEMM_BLOCK_DIM, 4)
fp8_dequantize_gemm_smem_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_fp8_e4m3* __restrict__ weight,
    const float* __restrict__ scale_inv,
    int M, int N, int K) {

    int m_start = blockIdx.y * FP8_GEMM_M_TILE;
    int n_start = blockIdx.x * FP8_GEMM_N_TILE;
    int local_m = threadIdx.x / FP8_GEMM_N_TILE;
    int local_n = threadIdx.x % FP8_GEMM_N_TILE;
    int m = m_start + local_m;
    int n = n_start + local_n;

    __shared__ __nv_bfloat16 smem_input[FP8_GEMM_M_TILE][FP8_GEMM_K_TILE];
    __shared__ __nv_fp8_e4m3 smem_weight[FP8_GEMM_N_TILE][FP8_GEMM_K_TILE + FP8_GEMM_WEIGHT_PAD];

    float sum = 0.0f;
    int num_k_tiles = (K + FP8_GEMM_K_TILE - 1) / FP8_GEMM_K_TILE;
    int num_k_blocks = K / 128;
    int valid_m = min(FP8_GEMM_M_TILE, M - m_start);
    int valid_n = min(FP8_GEMM_N_TILE, N - n_start);

    for (int kb = 0; kb < num_k_tiles; kb++) {
        int k_start = kb * FP8_GEMM_K_TILE;
        int k_tile = min(FP8_GEMM_K_TILE, K - k_start);

        for (int i = threadIdx.x; i < valid_m * k_tile; i += FP8_GEMM_BLOCK_DIM) {
            int lm = i / k_tile;
            int lk = i % k_tile;
            smem_input[lm][lk] = input[(m_start + lm) * K + k_start + lk];
        }

        for (int i = threadIdx.x; i < valid_n * k_tile; i += FP8_GEMM_BLOCK_DIM) {
            int ln = i / k_tile;
            int lk = i % k_tile;
            smem_weight[ln][lk] = weight[(n_start + ln) * K + k_start + lk];
        }

        __syncthreads();

        if (m < M && n < N) {
            int n_block = n / 128;
            float scale = scale_inv[n_block * num_k_blocks + kb];

            #pragma unroll
            for (int k = 0; k < FP8_GEMM_K_TILE; k++) {
                if (k < k_tile) {
                    float w_val = static_cast<float>(smem_weight[local_n][k]) * scale;
                    float x_val = __bfloat162float(smem_input[local_m][k]);
                    sum += w_val * x_val;
                }
            }
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        output[(size_t)m * N + n] = __float2bfloat16(sum);
    }
}

} // namespace

extern "C" {

void glm_fp8_linear_decode(GlmCtx* ctx, void* bf16_out, const void* bf16_input,
                            const void* fp8_weight, const float* weight_scale,
                            int m, int n, int k) {
    cudaSetDevice(ctx->device_id);
    if (m == 1) {
        int num_row_groups = (n + FP8_GEMV_ROWS_PER_BLOCK - 1) / FP8_GEMV_ROWS_PER_BLOCK;
        int grid_size = m * num_row_groups;
        fp8_dequantize_gemv_kernel<<<grid_size, FP8_GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
            reinterpret_cast<__nv_bfloat16*>(bf16_out),
            reinterpret_cast<const __nv_bfloat16*>(bf16_input),
            reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
            weight_scale,
            m, n, k);
    } else {
        dim3 grid((n + FP8_GEMM_N_TILE - 1) / FP8_GEMM_N_TILE,
                  (m + FP8_GEMM_M_TILE - 1) / FP8_GEMM_M_TILE);
        fp8_dequantize_gemm_smem_kernel<<<grid, FP8_GEMM_BLOCK_DIM, 0, GLM_STREAM(ctx)>>>(
            reinterpret_cast<__nv_bfloat16*>(bf16_out),
            reinterpret_cast<const __nv_bfloat16*>(bf16_input),
            reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
            weight_scale,
            m, n, k);
    }
}

// ---------------------------------------------------------------------------
// Linear (BF16 GEMM via cuBLAS)
//   out = input @ weight.T
//   input:  [batch, k]  row-major BF16
//   weight: [n, k]      row-major BF16
//   out:    [batch, n]  row-major BF16
// ---------------------------------------------------------------------------

void glm_linear(GlmCtx* ctx, void* out, const void* input,
                const void* weight, int batch, int n, int k) {
    cudaSetDevice(ctx->device_id);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasGemmEx(CUBLAS(ctx),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n, batch, k,
        &alpha,
        weight, CUDA_R_16BF, k,
        input,  CUDA_R_16BF, k,
        &beta,
        out,    CUDA_R_16BF, n,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

} // extern "C"
