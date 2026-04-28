#include "glm_ops.h"

#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define CUBLAS(ctx) (*reinterpret_cast<cublasHandle_t*>(&(ctx)->cublas_handle))

namespace {

constexpr int GEMV_WARP_SIZE = 32;
constexpr int GEMV_ROWS_PER_BLOCK = 8;
constexpr int GEMV_BLOCK_SIZE = GEMV_ROWS_PER_BLOCK * GEMV_WARP_SIZE;
constexpr int FP8_QUANT_BLOCK = 128;

// ---------------------------------------------------------------------------
// BF16 GEMV kernel: optimized for M=1 (single-token decode)
// Each warp computes one output element. Input tiles are cooperatively loaded
// into shared memory as bfloat162 vectors to avoid redundant global reads and
// eliminate bank conflicts. Weight rows are also read via bfloat162 vector
// loads for 2x load throughput. ROWS_PER_BLOCK=8 amortizes the shared input
// tile across more warps and improves occupancy for small N.
// K_TILE is a template parameter: larger values reduce __syncthreads barriers
// at the cost of more shared memory and longer unrolled inner loops.
// ---------------------------------------------------------------------------

template<int K_TILE>
__global__ void __launch_bounds__(GEMV_BLOCK_SIZE, 4)
bf16_gemv_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int M, int N, int K) {

    constexpr int K_TILE_VEC = K_TILE / 2;

    if (N == 0 || M == 0 || K == 0) return;

    int num_row_groups = (N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
    int m = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;
    int row = row_group * GEMV_ROWS_PER_BLOCK + threadIdx.x / GEMV_WARP_SIZE;
    int lane = threadIdx.x % GEMV_WARP_SIZE;

    bool valid = m < M && row < N;

    __shared__ __nv_bfloat162 smem_vec[K_TILE_VEC];

    if (m < M) {
        const __nv_bfloat16* input_row = input + (size_t)m * K;
        const __nv_bfloat16* weight_row = weight + (size_t)row * K;

        float sum = 0.0f;

        int num_k_tiles = K / K_TILE;
        int remaining_start = num_k_tiles * K_TILE;

        for (int kb = 0; kb < num_k_tiles; kb++) {
            int k_start = kb * K_TILE;

            const __nv_bfloat162* input_vec = reinterpret_cast<const __nv_bfloat162*>(input_row + k_start);
            for (int i = threadIdx.x; i < K_TILE_VEC; i += GEMV_BLOCK_SIZE) {
                smem_vec[i] = input_vec[i];
            }
            __syncthreads();

            if (valid) {
                const __nv_bfloat162* weight_vec = reinterpret_cast<const __nv_bfloat162*>(weight_row + k_start);
                #pragma unroll
                for (int ki = lane; ki < K_TILE_VEC; ki += GEMV_WARP_SIZE) {
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
            for (int i = threadIdx.x; i < remaining; i += GEMV_BLOCK_SIZE) {
                smem[i] = input_row[remaining_start + i];
            }
            __syncthreads();

            if (valid) {
                for (int k = remaining_start + lane; k < K; k += GEMV_WARP_SIZE) {
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

// ---------------------------------------------------------------------------
// FP8 GEMV kernel: optimized for M=1 (single-token decode)
// Same structure as BF16 GEMV but dequantizes FP8 weights with per-block
// scale factors (128x128 block quantization). K_TILE must be a multiple of
// FP8_QUANT_BLOCK (128) so each K-tile contains an integer number of
// quantization blocks, reducing __syncthreads barriers proportionally.
// ---------------------------------------------------------------------------

template<int K_TILE>
__global__ void __launch_bounds__(GEMV_BLOCK_SIZE, 4)
fp8_dequantize_gemv_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_fp8_e4m3* __restrict__ weight,
    const float* __restrict__ scale_inv,
    int M, int N, int K) {

    constexpr int K_TILE_VEC = K_TILE / 2;
    constexpr int BLOCKS_PER_TILE = K_TILE / FP8_QUANT_BLOCK;

    int num_row_groups = (N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
    int m = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;
    int row = row_group * GEMV_ROWS_PER_BLOCK + threadIdx.x / GEMV_WARP_SIZE;
    int lane = threadIdx.x % GEMV_WARP_SIZE;

    bool valid = m < M && row < N;

    __shared__ __nv_bfloat162 smem_vec[K_TILE_VEC];

    int num_k_tiles = K / K_TILE;
    int num_k_blocks = K / FP8_QUANT_BLOCK;
    const __nv_bfloat16* input_row = input + (size_t)m * K;
    const __nv_fp8_e4m3* weight_row = weight + (size_t)row * K;
    int n_block = row / FP8_QUANT_BLOCK;

    float sum = 0.0f;

    for (int tile = 0; tile < num_k_tiles; tile++) {
        int k_start = tile * K_TILE;

        const __nv_bfloat162* input_vec = reinterpret_cast<const __nv_bfloat162*>(input_row + k_start);
        for (int i = threadIdx.x; i < K_TILE_VEC; i += GEMV_BLOCK_SIZE) {
            smem_vec[i] = input_vec[i];
        }
        __syncthreads();

        if (valid) {
            #pragma unroll
            for (int b = 0; b < BLOCKS_PER_TILE; b++) {
                int kb = tile * BLOCKS_PER_TILE + b;
                float scale = scale_inv[n_block * num_k_blocks + kb];
                #pragma unroll
                for (int ki = lane; ki < FP8_QUANT_BLOCK / 2; ki += GEMV_WARP_SIZE) {
                    __nv_bfloat162 x2 = smem_vec[b * (FP8_QUANT_BLOCK / 2) + ki];
                    float x0 = __bfloat162float(x2.x);
                    float x1 = __bfloat162float(x2.y);
                    float w0 = static_cast<float>(weight_row[k_start + b * FP8_QUANT_BLOCK + ki * 2]) * scale;
                    float w1 = static_cast<float>(weight_row[k_start + b * FP8_QUANT_BLOCK + ki * 2 + 1]) * scale;
                    sum += w0 * x0 + w1 * x1;
                }
            }
        }

        __syncthreads();
    }

    int remaining_start = num_k_tiles * K_TILE;
    int remaining = K - remaining_start;
    if (remaining > 0) {
        __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_vec);
        for (int i = threadIdx.x; i < remaining; i += GEMV_BLOCK_SIZE) {
            smem[i] = input_row[remaining_start + i];
        }
        __syncthreads();

        if (valid) {
            int remaining_k_blocks_start = num_k_tiles * BLOCKS_PER_TILE;
            int remaining_k_blocks = remaining / FP8_QUANT_BLOCK;
            for (int b = 0; b < remaining_k_blocks; b++) {
                int kb = remaining_k_blocks_start + b;
                float scale = scale_inv[n_block * num_k_blocks + kb];
                int b_start = b * FP8_QUANT_BLOCK;
                for (int k = b_start + lane; k < b_start + FP8_QUANT_BLOCK; k += GEMV_WARP_SIZE) {
                    float w_val = static_cast<float>(weight_row[remaining_start + k]) * scale;
                    float x_val = __bfloat162float(smem[k]);
                    sum += w_val * x_val;
                }
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
        int num_row_groups = (n + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
        int grid_size = m * num_row_groups;
        if (k >= 512) {
            fp8_dequantize_gemv_kernel<512><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
                weight_scale, m, n, k);
        } else if (k >= 256) {
            fp8_dequantize_gemv_kernel<256><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
                weight_scale, m, n, k);
        } else {
            fp8_dequantize_gemv_kernel<128><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
                weight_scale, m, n, k);
        }
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
        int num_row_groups = (n + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
        int grid_size = batch * num_row_groups;
        if (k >= 512) {
            bf16_gemv_kernel<512><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(out),
                reinterpret_cast<const __nv_bfloat16*>(input),
                reinterpret_cast<const __nv_bfloat16*>(weight),
                batch, n, k);
        } else if (k >= 256) {
            bf16_gemv_kernel<256><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(out),
                reinterpret_cast<const __nv_bfloat16*>(input),
                reinterpret_cast<const __nv_bfloat16*>(weight),
                batch, n, k);
        } else {
            bf16_gemv_kernel<128><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(out),
                reinterpret_cast<const __nv_bfloat16*>(input),
                reinterpret_cast<const __nv_bfloat16*>(weight),
                batch, n, k);
        }
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
