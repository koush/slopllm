#include "glm_ops.h"

#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define CUBLAS(ctx) (*reinterpret_cast<cublasHandle_t*>(&(ctx)->cublas_handle))

namespace {

// ---------------------------------------------------------------------------
// GEMV kernel: optimized for M=1 decode
// Each warp computes one output element. Input tiles are cooperatively loaded
// into shared memory as bfloat162 vectors to avoid redundant global reads and
// eliminate bank conflicts (each 4-byte bfloat162 maps to one bank, so 32
// consecutive elements span 32 banks conflict-free when accessed by a warp).
// ROWS_PER_BLOCK=8 amortizes the shared input tile across more warps and
// improves occupancy for small N.
// ---------------------------------------------------------------------------

constexpr int FP8_GEMV_WARP_SIZE = 32;
constexpr int FP8_GEMV_ROWS_PER_BLOCK = 8;
constexpr int FP8_GEMV_BLOCK_SIZE = FP8_GEMV_ROWS_PER_BLOCK * FP8_GEMV_WARP_SIZE;
constexpr int FP8_GEMV_K_TILE = 128;
constexpr int FP8_GEMV_K_TILE_VEC = FP8_GEMV_K_TILE / 2;

__global__ void __launch_bounds__(FP8_GEMV_BLOCK_SIZE, 4)
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

    __shared__ __nv_bfloat162 smem_vec[FP8_GEMV_K_TILE_VEC];

    int num_k_blocks = K / 128;
    const __nv_bfloat16* input_row = input + (size_t)m * K;
    const __nv_fp8_e4m3* weight_row = weight + (size_t)row * K;
    int n_block = row / 128;

    float sum = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        int k_start = kb << 7;

        const __nv_bfloat162* input_vec = reinterpret_cast<const __nv_bfloat162*>(input_row + k_start);
        for (int i = threadIdx.x; i < FP8_GEMV_K_TILE_VEC; i += FP8_GEMV_BLOCK_SIZE) {
            smem_vec[i] = input_vec[i];
        }
        __syncthreads();

        if (valid) {
            float scale = scale_inv[n_block * num_k_blocks + kb];

            #pragma unroll
            for (int ki = lane; ki < FP8_GEMV_K_TILE_VEC; ki += FP8_GEMV_WARP_SIZE) {
                __nv_bfloat162 x2 = smem_vec[ki];
                float x0 = __bfloat162float(x2.x);
                float x1 = __bfloat162float(x2.y);
                float w0 = static_cast<float>(weight_row[k_start + ki * 2]) * scale;
                float w1 = static_cast<float>(weight_row[k_start + ki * 2 + 1]) * scale;
                sum += w0 * x0 + w1 * x1;
            }
        }

        __syncthreads();
    }

    int remaining_start = num_k_blocks * 128;
    int remaining = K - remaining_start;
    if (remaining > 0) {
        __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_vec);
        for (int i = threadIdx.x; i < remaining; i += FP8_GEMV_BLOCK_SIZE) {
            smem[i] = input_row[remaining_start + i];
        }
        __syncthreads();

        if (valid) {
            int kb = remaining_start / 128;
            float scale = scale_inv[n_block * num_k_blocks + kb];
            for (int k = remaining_start + lane; k < K; k += FP8_GEMV_WARP_SIZE) {
                int ki = k - remaining_start;
                float w_val = static_cast<float>(weight_row[k]) * scale;
                float x_val = __bfloat162float(smem[ki]);
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

// ---------------------------------------------------------------------------
// BF16 GEMV kernel: optimized for M=1 (single-token decode)
// Each warp computes one output element. Input tiles are cooperatively loaded
// into shared memory as bfloat162 vectors to avoid redundant global reads and
// eliminate bank conflicts (each 4-byte bfloat162 maps to one bank, so 32
// consecutive elements span 32 banks conflict-free when accessed by a warp).
// Weight rows are also read via bfloat162 vector loads for 2x load throughput.
// ROWS_PER_BLOCK=8 amortizes the shared input tile across more warps and
// improves occupancy for small N.
// ---------------------------------------------------------------------------

constexpr int BF16_GEMV_WARP_SIZE = 32;
constexpr int BF16_GEMV_ROWS_PER_BLOCK = 8;
constexpr int BF16_GEMV_BLOCK_SIZE = BF16_GEMV_ROWS_PER_BLOCK * BF16_GEMV_WARP_SIZE;
constexpr int BF16_GEMV_K_TILE = 128;
constexpr int BF16_GEMV_K_TILE_VEC = BF16_GEMV_K_TILE / 2;

__global__ void __launch_bounds__(BF16_GEMV_BLOCK_SIZE, 4)
bf16_gemv_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int M, int N, int K) {

    if (N == 0 || M == 0 || K == 0) return;

    int num_row_groups = (N + BF16_GEMV_ROWS_PER_BLOCK - 1) / BF16_GEMV_ROWS_PER_BLOCK;
    int m = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;
    int row = row_group * BF16_GEMV_ROWS_PER_BLOCK + threadIdx.x / BF16_GEMV_WARP_SIZE;
    int lane = threadIdx.x % BF16_GEMV_WARP_SIZE;

    bool valid = m < M && row < N;

    __shared__ __nv_bfloat162 smem_vec[BF16_GEMV_K_TILE_VEC];

    if (m < M) {
        const __nv_bfloat16* input_row = input + (size_t)m * K;
        const __nv_bfloat16* weight_row = weight + (size_t)row * K;

        float sum = 0.0f;

        int num_k_tiles = K / BF16_GEMV_K_TILE;
        int remaining_start = num_k_tiles * BF16_GEMV_K_TILE;

        for (int kb = 0; kb < num_k_tiles; kb++) {
            int k_start = kb * BF16_GEMV_K_TILE;

            const __nv_bfloat162* input_vec = reinterpret_cast<const __nv_bfloat162*>(input_row + k_start);
            for (int i = threadIdx.x; i < BF16_GEMV_K_TILE_VEC; i += BF16_GEMV_BLOCK_SIZE) {
                smem_vec[i] = input_vec[i];
            }
            __syncthreads();

            if (valid) {
                const __nv_bfloat162* weight_vec = reinterpret_cast<const __nv_bfloat162*>(weight_row + k_start);
                #pragma unroll
                for (int ki = lane; ki < BF16_GEMV_K_TILE_VEC; ki += BF16_GEMV_WARP_SIZE) {
                    __nv_bfloat162 w2 = weight_vec[ki];
                    __nv_bfloat162 x2 = smem_vec[ki];
                    float w0 = __bfloat162float(w2.x);
                    float w1 = __bfloat162float(w2.y);
                    float x0 = __bfloat162float(x2.x);
                    float x1 = __bfloat162float(x2.y);
                    sum += w0 * x0 + w1 * x1;
                }
            }

            __syncthreads();
        }

        int remaining = K - remaining_start;
        if (remaining > 0) {
            __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_vec);
            for (int i = threadIdx.x; i < remaining; i += BF16_GEMV_BLOCK_SIZE) {
                smem[i] = input_row[remaining_start + i];
            }
            __syncthreads();

            if (valid) {
                for (int k = remaining_start + lane; k < K; k += BF16_GEMV_WARP_SIZE) {
                    int ki = k - remaining_start;
                    float w_val = __bfloat162float(weight_row[k]);
                    float x_val = __bfloat162float(smem[ki]);
                    sum += w_val * x_val;
                }
            }

            __syncthreads();
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (valid && lane == 0) {
            output[(size_t)m * N + row] = __float2bfloat16(sum);
        }
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

    if (batch == 1) {
        int num_row_groups = (n + BF16_GEMV_ROWS_PER_BLOCK - 1) / BF16_GEMV_ROWS_PER_BLOCK;
        int grid_size = batch * num_row_groups;
        bf16_gemv_kernel<<<grid_size, BF16_GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
            reinterpret_cast<__nv_bfloat16*>(out),
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            batch, n, k);
        return;
    }

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
