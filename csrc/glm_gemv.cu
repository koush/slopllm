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

__device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xFFFFFFFF, x, offset));
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xFFFFFFFF, x, offset);
    }
    return x;
}

// ---------------------------------------------------------------------------
// BF16 GEMV kernel: optimized for M=1 (single-token decode)
// Each warp computes one output element. ROWS_PER_BLOCK=8 amortizes launch
// overhead. We use 16-byte (uint4 == 8 BF16) vector loads for both input and
// weight rows so peak memory throughput is reached with as few outstanding
// loads as possible. The input row is small (a few KB) and lives in L1 across
// the 8 warps of a block, so we don't bother staging it into shared memory.
//
// Throughput analysis (per warp, per K-iteration):
//   warp loads 32 lanes * uint4 = 32 * 16 B = 512 B of weights and 512 B of
//   input. With K=5120, each warp does K/256 = 20 iterations.
// ---------------------------------------------------------------------------

constexpr int GEMV_K_VEC = 8;  // BF16 elements per uint4

__device__ __forceinline__ void uint4_to_bf16x8(
    const uint4& v, __nv_bfloat16 out[8]) {
    auto* h = reinterpret_cast<const __nv_bfloat16*>(&v);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = h[i];
}

__global__ void __launch_bounds__(GEMV_BLOCK_SIZE, 4)
bf16_gemv_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int M, int N, int K) {

    if (N == 0 || M == 0 || K == 0) return;

    int num_row_groups = (N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
    int m = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;
    int warp_id = threadIdx.x / GEMV_WARP_SIZE;
    int row = row_group * GEMV_ROWS_PER_BLOCK + warp_id;
    int lane = threadIdx.x % GEMV_WARP_SIZE;

    if (m >= M) return;

    const __nv_bfloat16* input_row  = input  + (size_t)m   * K;
    const __nv_bfloat16* weight_row = weight + (size_t)row * K;

    // K_VEC = 8 BF16 / uint4 load.
    // Each warp strides by (32 lanes * 8 elem) = 256 BF16 elements per iter.
    int K_vec = K / GEMV_K_VEC;       // # full uint4 chunks
    int K_tail_start = K_vec * GEMV_K_VEC;

    float sum = 0.0f;
    bool row_valid = row < N;

    if (row_valid) {
        const uint4* input_v4  = reinterpret_cast<const uint4*>(input_row);
        const uint4* weight_v4 = reinterpret_cast<const uint4*>(weight_row);

        // 32-lane warp strides through K_vec chunks.
        for (int ki = lane; ki < K_vec; ki += GEMV_WARP_SIZE) {
            uint4 wv = weight_v4[ki];
            uint4 xv = input_v4[ki];
            __nv_bfloat16 wb[8], xb[8];
            uint4_to_bf16x8(wv, wb);
            uint4_to_bf16x8(xv, xb);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sum += __bfloat162float(wb[j]) * __bfloat162float(xb[j]);
            }
        }

        // Scalar tail (rare: K is typically a multiple of 256/8=32).
        for (int k = K_tail_start + lane; k < K; k += GEMV_WARP_SIZE) {
            sum += __bfloat162float(weight_row[k]) * __bfloat162float(input_row[k]);
        }
    }

    // Warp reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (row_valid && lane == 0) {
        output[(size_t)m * N + row] = __float2bfloat16(sum);
    }
}

// ---------------------------------------------------------------------------
// Split-K BF16 GEMV: for small-N decode where the regular kernel can't fill
// the GPU. Each block handles one output row split across K_SPLIT segments.
// We use 4 warps each owning a contiguous K segment of size ~K/4, sum
// across warps via shared memory, and a single warp writes the final result.
// Block count = N * K_SPLIT_BLOCKS. Output is fully computed in one launch
// (no atomics, no separate reduction).
// ---------------------------------------------------------------------------

constexpr int GEMV_SPLITK_WARPS = 4;
constexpr int GEMV_SPLITK_BLOCK_SIZE = GEMV_SPLITK_WARPS * GEMV_WARP_SIZE;
constexpr int GEMV_SPLITK_PARTITIONS = 4;  // K is split into this many segments

template <int NumPartitions>
__global__ void __launch_bounds__(GEMV_SPLITK_BLOCK_SIZE * NumPartitions, 2)
bf16_gemv_splitk_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int M, int N, int K) {

    if (N == 0 || M == 0 || K == 0) return;

    int m = blockIdx.x / N;
    int row = blockIdx.x % N;

    constexpr int TOTAL_WARPS = NumPartitions * GEMV_SPLITK_WARPS;
    int tid = threadIdx.x;
    int warp_id = tid / GEMV_WARP_SIZE;
    int lane = tid % GEMV_WARP_SIZE;

    const __nv_bfloat16* input_row  = input  + (size_t)m   * K;
    const __nv_bfloat16* weight_row = weight + (size_t)row * K;

    int K_vec = K / GEMV_K_VEC;
    int K_tail_start = K_vec * GEMV_K_VEC;

    const uint4* input_v4  = reinterpret_cast<const uint4*>(input_row);
    const uint4* weight_v4 = reinterpret_cast<const uint4*>(weight_row);

    float sum = 0.0f;
    int g_thread = warp_id * GEMV_WARP_SIZE + lane;
    int g_threads = TOTAL_WARPS * GEMV_WARP_SIZE;
    for (int ki = g_thread; ki < K_vec; ki += g_threads) {
        uint4 wv = weight_v4[ki];
        uint4 xv = input_v4[ki];
        __nv_bfloat16 wb[8], xb[8];
        uint4_to_bf16x8(wv, wb);
        uint4_to_bf16x8(xv, xb);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            sum += __bfloat162float(wb[j]) * __bfloat162float(xb[j]);
        }
    }
    for (int k = K_tail_start + g_thread; k < K; k += g_threads) {
        sum += __bfloat162float(weight_row[k]) * __bfloat162float(input_row[k]);
    }

    // Warp reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // First lane of each warp writes its partial to shared memory.
    __shared__ float warp_sums[TOTAL_WARPS];
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    // First warp reduces across warps.
    if (warp_id == 0) {
        float s = (lane < TOTAL_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = TOTAL_WARPS / 2; offset > 0; offset >>= 1) {
            s += __shfl_down_sync(0xFFFFFFFF, s, offset);
        }
        if (lane == 0) {
            output[(size_t)m * N + row] = __float2bfloat16(s);
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
    const __nv_bfloat16* __restrict__ scale_inv,
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
                float scale = __bfloat162float(scale_inv[n_block * num_k_blocks + kb]);
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
                float scale = __bfloat162float(scale_inv[n_block * num_k_blocks + kb]);
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
    const __nv_bfloat16* __restrict__ scale_inv,
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
            float scale = __bfloat162float(scale_inv[n_block * num_k_blocks + kb]);

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
// NVFP4 (W4A16) dequantize + GEMV for decode
// FP4 E2M1 weights packed as uint8 (2 per byte), double quantization:
//   weight_scale  (FP8 E4M3, per-block) + weight_scale_2 (F32, global)
//   dequant: LUT[nibble] * float(weight_scale[block]) * weight_scale_2
// GROUP_SIZE = 16
// ---------------------------------------------------------------------------

constexpr int NVFP4_QUANT_GROUP = 16;

// Decode 4-bit E2M1 float to float32 using register arithmetic only.
// Constant memory LUT with divergent warp access serializes to 32 sequential
// fetches; this replaces it with pure register ops (no memory traffic).
// Bit layout: [sign][exp1][exp0][mantissa], exponent bias = 1.
// Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
__device__ __forceinline__ float fp4_e2m1_decode(uint8_t nibble) {
    uint32_t n = (uint32_t)nibble & 0x7u;
    // n=0 → 0.0, n=1 → 0.5 (subnormal), n≥2 → normal: 1.m * 2^(e-1)
    uint32_t fp = (n < 2u) ? (n * 0x3F000000u)
                            : (((126u + (n >> 1u)) << 23u) | ((n & 1u) << 22u));
    fp |= (uint32_t)(nibble >> 3u) << 31u;
    return __uint_as_float(fp);
}

template<int K_TILE>
__global__ void __launch_bounds__(GEMV_BLOCK_SIZE, 4)
nvfp4_dequantize_gemv_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const __nv_fp8_e4m3* __restrict__ weight_scale,
    const float* __restrict__ weight_scale_2,
    int M, int N, int K) {

    constexpr int K_TILE_VEC = K_TILE / 2;
    constexpr int GROUPS_PER_TILE = K_TILE / NVFP4_QUANT_GROUP;

    int num_row_groups = (N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
    int m = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;
    int row = row_group * GEMV_ROWS_PER_BLOCK + threadIdx.x / GEMV_WARP_SIZE;
    int lane = threadIdx.x % GEMV_WARP_SIZE;

    bool valid = m < M && row < N;

    __shared__ __nv_bfloat162 smem_vec[K_TILE_VEC];

    int num_k_tiles = K / K_TILE;
    int num_k_groups = K / NVFP4_QUANT_GROUP;
    const __nv_bfloat16* input_row = input + (size_t)m * K;
    const uint8_t* weight_row = weight + (size_t)row * (K / 2);
    float scale_2_val = *weight_scale_2;

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
            for (int g = 0; g < GROUPS_PER_TILE; g++) {
                int kg = tile * GROUPS_PER_TILE + g;
                float scale = static_cast<float>(weight_scale[row * num_k_groups + kg]) * scale_2_val;
                int g_start = g * NVFP4_QUANT_GROUP;
                #pragma unroll
                for (int ki = lane; ki < NVFP4_QUANT_GROUP / 2; ki += GEMV_WARP_SIZE) {
                    __nv_bfloat162 x2 = smem_vec[(g_start + ki * 2) / 2];
                    float x0 = __bfloat162float(x2.x);
                    float x1 = __bfloat162float(x2.y);
                    uint8_t packed = weight_row[(k_start + g_start) / 2 + ki];
                    float w0 = fp4_e2m1_decode(packed & 0x0Fu) * scale;
                    float w1 = fp4_e2m1_decode(packed >> 4u) * scale;
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
            int remaining_groups_start = num_k_tiles * GROUPS_PER_TILE;
            int remaining_groups = remaining / NVFP4_QUANT_GROUP;
            for (int g = 0; g < remaining_groups; g++) {
                int kg = remaining_groups_start + g;
                float scale = static_cast<float>(weight_scale[row * num_k_groups + kg]) * scale_2_val;
                int g_start = g * NVFP4_QUANT_GROUP;
                for (int k = g_start + lane; k < g_start + NVFP4_QUANT_GROUP; k += GEMV_WARP_SIZE) {
                    uint8_t packed = weight_row[(remaining_start + k) / 2];
                    float w0 = fp4_e2m1_decode(packed & 0x0Fu) * scale;
                    float w1 = fp4_e2m1_decode(packed >> 4u) * scale;
                    float x_val = __bfloat162float(smem[k]);
                    sum += (k % 2 == 0 ? w0 : w1) * x_val;
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

constexpr int NVFP4_GEMM_M_TILE = 16;
constexpr int NVFP4_GEMM_N_TILE = 32;
constexpr int NVFP4_GEMM_K_TILE = 128;
constexpr int NVFP4_GEMM_BLOCK_DIM = NVFP4_GEMM_M_TILE * NVFP4_GEMM_N_TILE;
// Narrow M-tile used when M <= NVFP4_GEMM_SMALL_M_THRESHOLD (e.g. MTP
// verification batches): quadruples grid.y vs. NVFP4_GEMM_M_TILE=16 while still
// reusing each decoded weight tile across 4 rows. See the kernel comment for
// why this trades less reuse for more occupancy than the M_TILE=16 variant.
constexpr int NVFP4_GEMM_SMALL_M_TILE = 4;
constexpr int NVFP4_GEMM_SMALL_M_THRESHOLD = 32;
// When M > 1 and N is small, the smem GEMM kernel launches too few CTAs to
// fill the GPU (e.g. N=256 → grid.x=8 → only 32 CTAs at M=15).  cuBLAS with
// split-K gives much higher occupancy.  Below this N threshold (per shard),
// dequantize FP4→BF16 and call cublasGemmEx instead.
constexpr int NVFP4_CUBLAS_N_THRESHOLD = 512;
// Pad each smem_weight row so its stride in 4-byte words (65) is coprime with
// the 32 shared-memory banks — every lane of a warp (which spans all 32 `n`
// values for a fixed `k`) then lands in a distinct bank, avoiding conflicts.
constexpr int NVFP4_GEMM_SMEM_PAD = 2;

// ---------------------------------------------------------------------------
// NVFP4 dequantize-only kernel: converts FP4 weights to BF16 in a workspace
// buffer so that a subsequent cublasGemmEx call can use Tensor Cores + split-K.
// Each thread decodes one output element.  The grid is (N * K) elements total.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 8)
nvfp4_dequantize_to_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const uint8_t* __restrict__ weight,
    const __nv_fp8_e4m3* __restrict__ weight_scale,
    const float* __restrict__ weight_scale_2,
    int N, int K) {

    int num_k_groups = K / NVFP4_QUANT_GROUP;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

    int n = idx / K;
    int k = idx % K;
    int kg = k / NVFP4_QUANT_GROUP;
    float scale = static_cast<float>(weight_scale[n * num_k_groups + kg]) * (*weight_scale_2);
    uint8_t packed = weight[n * (K / 2) + k / 2];
    float val = (k & 1) ? fp4_e2m1_decode(packed >> 4u) * scale
                         : fp4_e2m1_decode(packed & 0x0Fu) * scale;
    dst[idx] = __float2bfloat16(val);
}

// ---------------------------------------------------------------------------
// NVFP4 dequantize+GEMM (M > 1): the FP4 weight nibble and its scale depend
// only on (n, k), not on the output row m. The naive approach — every thread
// decoding its own (m, n) element — redoes the same decode + scale multiply
// once per row in the M tile (16x redundant ALU work, plus 16x redundant
// `weight_scale` loads). Instead we cooperatively decode+scale each weight
// tile into shared memory exactly once per K-tile (using all 512 threads of
// the block), then every row in the M tile just does a plain BF16x2 FMA
// against the pre-decoded values. This keeps the same general tiling (works
// for any M, amortizing the FP4 weight read across the whole M tile) while
// removing the redundant decode work — beneficial for both small-M (e.g.
// speculative-decode verification) and large-M (prefill) shapes.
// ---------------------------------------------------------------------------
// Templated on M_TILE so small-M callers (e.g. speculative-decode verification,
// where M ~ 10s of tokens) can use a narrower tile. A narrower M_TILE means:
//   - more CTAs along the M dimension (grid.y = ceil(M / M_TILE)) -> better SM
//     occupancy when M doesn't fill a single M_TILE=16 tile (the M=15 case
//     launches only ceil(N/N_TILE) CTAs total at M_TILE=16, badly under-filling
//     a 188-SM GPU), and
//   - less per-block input-staging work (the smem_input load loop is the part
//     of this kernel whose cost actually scales with valid_m = min(M_TILE, M),
//     not the weight decode/compute, which runs the same number of iterations
//     regardless of how many M-rows are valid).
// Weight tiles are still decoded once per N_TILE and reused across the (now
// smaller) M_TILE rows, so this keeps the bandwidth-efficient reuse that makes
// the GEMV fallback a bad idea at M > 1 (it re-reads the whole weight matrix
// once per row), while trading less of it for occupancy than M_TILE=16 does.
template <int M_TILE>
__global__ void __launch_bounds__(M_TILE * NVFP4_GEMM_N_TILE, 4)
nvfp4_dequantize_gemm_smem_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const __nv_fp8_e4m3* __restrict__ weight_scale,
    const float* __restrict__ weight_scale_2,
    int M, int N, int K) {

    constexpr int BLOCK_DIM = M_TILE * NVFP4_GEMM_N_TILE;

    int m_start = blockIdx.y * M_TILE;
    int n_start = blockIdx.x * NVFP4_GEMM_N_TILE;
    int local_m = threadIdx.x / NVFP4_GEMM_N_TILE;
    int local_n = threadIdx.x % NVFP4_GEMM_N_TILE;
    int m = m_start + local_m;
    int n = n_start + local_n;

    __shared__ __nv_bfloat16 smem_input[M_TILE][NVFP4_GEMM_K_TILE];
    // Pre-decoded, pre-scaled BF16 weight tile (shared across all M_TILE rows).
    __shared__ __nv_bfloat16 smem_weight[NVFP4_GEMM_N_TILE][NVFP4_GEMM_K_TILE + NVFP4_GEMM_SMEM_PAD];

    float sum = 0.0f;
    int num_k_tiles = (K + NVFP4_GEMM_K_TILE - 1) / NVFP4_GEMM_K_TILE;
    int num_k_groups = K / NVFP4_QUANT_GROUP;
    int valid_m = min(M_TILE, M - m_start);
    int valid_n = min(NVFP4_GEMM_N_TILE, N - n_start);
    float scale_2_val = *weight_scale_2;

    for (int kb = 0; kb < num_k_tiles; kb++) {
        int k_start = kb * NVFP4_GEMM_K_TILE;
        int k_tile = min(NVFP4_GEMM_K_TILE, K - k_start);

        for (int i = threadIdx.x; i < valid_m * k_tile; i += BLOCK_DIM) {
            int lm = i / k_tile;
            int lk = i % k_tile;
            smem_input[lm][lk] = input[(m_start + lm) * K + k_start + lk];
        }

        // K_TILE is a multiple of NVFP4_QUANT_GROUP (16) and weights are packed
        // 2-per-byte, so k_tile is always even — k_tile/2 byte-pairs to decode.
        int k_tile_pairs = k_tile / 2;
        for (int i = threadIdx.x; i < valid_n * k_tile_pairs; i += BLOCK_DIM) {
            int ln = i / k_tile_pairs;
            int lk_pair = i % k_tile_pairs;
            int k = lk_pair * 2;
            int kg = kb * (NVFP4_GEMM_K_TILE / NVFP4_QUANT_GROUP) + k / NVFP4_QUANT_GROUP;
            float scale = static_cast<float>(weight_scale[(n_start + ln) * num_k_groups + kg]) * scale_2_val;
            uint8_t packed = weight[(n_start + ln) * (K / 2) + k_start / 2 + lk_pair];
            smem_weight[ln][k]     = __float2bfloat16(fp4_e2m1_decode(packed & 0x0Fu) * scale);
            smem_weight[ln][k + 1] = __float2bfloat16(fp4_e2m1_decode(packed >> 4u) * scale);
        }

        __syncthreads();

        if (m < M && n < N) {
            #pragma unroll 4
            for (int k = 0; k < k_tile; k += 2) {
                __nv_bfloat162 x2 = *reinterpret_cast<__nv_bfloat162*>(&smem_input[local_m][k]);
                __nv_bfloat162 w2 = *reinterpret_cast<__nv_bfloat162*>(&smem_weight[local_n][k]);
                sum += __bfloat162float(x2.x) * __bfloat162float(w2.x)
                     + __bfloat162float(x2.y) * __bfloat162float(w2.y);
            }
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        output[(size_t)m * N + n] = __float2bfloat16(sum);
    }
}

// ---------------------------------------------------------------------------
// nvfp4_mul_mat_id kernels (must be outside extern "C" due to templates)
// No shared memory — each warp reads input directly from global memory (L1-cached).
// This eliminates __syncthreads() barriers present in the tiled version.
//
// Template parameter RowsPerWarp: 1 for normal K (> 512), 2 for small K (<= 512).
// When RowsPerWarp=2, lanes 0-15 compute row0 and lanes 16-31 compute row1,
// doubling throughput when num_k_groups <= 32 (half the warp would otherwise
// sit idle).  Each block covers GEMV_ROWS_PER_BLOCK * RowsPerWarp rows.
// ---------------------------------------------------------------------------

template <int RowsPerWarp>
__global__ void __launch_bounds__(GEMV_BLOCK_SIZE, 4)
nvfp4_mul_mat_id_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* const* __restrict__ weight_ptrs,
    const __nv_fp8_e4m3* const* __restrict__ scale_ptrs,
    const float* const* __restrict__ scale2_ptrs,
    const int* __restrict__ expert_ids,
    int top_k, int count, int N, int K) {

    constexpr int LANES_PER_ROW = GEMV_WARP_SIZE / RowsPerWarp;
    constexpr int ROWS_PER_BLOCK = GEMV_ROWS_PER_BLOCK * RowsPerWarp;

    int num_row_groups = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int entry = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;
    int warp_id = threadIdx.x / GEMV_WARP_SIZE;
    int lane = threadIdx.x % GEMV_WARP_SIZE;

    if (entry >= count) return;

    int row_in_warp = lane / LANES_PER_ROW;
    int inner_lane = lane % LANES_PER_ROW;
    int row = row_group * ROWS_PER_BLOCK + warp_id * RowsPerWarp + row_in_warp;

    int bid = entry / top_k;
    int eid = expert_ids[entry];
    const __nv_bfloat16* input_row  = input + (size_t)bid * K;
    const uint8_t* weight_row = weight_ptrs[eid] + (size_t)row * (K / 2);
    const __nv_fp8_e4m3* scale_row = scale_ptrs[eid] + (size_t)row * (K / NVFP4_QUANT_GROUP);
    float scale_2_val = *scale2_ptrs[eid];

    int num_k_groups = K / NVFP4_QUANT_GROUP;
    bool row_valid = row < N;

    float sum = 0.0f;

    if (row_valid) {
        for (int g = inner_lane; g < num_k_groups; g += LANES_PER_ROW) {
            float scale = static_cast<float>(scale_row[g]) * scale_2_val;
            int k_start = g * NVFP4_QUANT_GROUP;

            const uint4* input_v4 = reinterpret_cast<const uint4*>(input_row + k_start);
            uint4 xv0 = input_v4[0];
            uint4 xv1 = input_v4[1];
            __nv_bfloat16 xb0[8], xb1[8];
            uint4_to_bf16x8(xv0, xb0);
            uint4_to_bf16x8(xv1, xb1);

            uint32_t w_lo = *reinterpret_cast<const uint32_t*>(weight_row + g * (NVFP4_QUANT_GROUP / 2));
            uint32_t w_hi = *reinterpret_cast<const uint32_t*>(weight_row + g * (NVFP4_QUANT_GROUP / 2) + 4);

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t packed = (w_lo >> (j * 8)) & 0xFFu;
                sum += fp4_e2m1_decode(packed & 0x0Fu) * scale * __bfloat162float(xb0[j * 2])
                     + fp4_e2m1_decode(packed >> 4u) * scale * __bfloat162float(xb0[j * 2 + 1]);
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint8_t packed = (w_hi >> (j * 8)) & 0xFFu;
                sum += fp4_e2m1_decode(packed & 0x0Fu) * scale * __bfloat162float(xb1[j * 2])
                     + fp4_e2m1_decode(packed >> 4u) * scale * __bfloat162float(xb1[j * 2 + 1]);
            }
        }
    }

    if constexpr (RowsPerWarp == 1) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (row_valid && lane == 0) {
            output[(size_t)entry * N + row] = __float2bfloat16(sum);
        }
    } else {
        // Half-warp reduction: XOR strides 1,2,4,8 within each 16-lane half.
        // Skip stride 16 which would cross the row boundary.
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
        if (row_valid && inner_lane == 0) {
            output[(size_t)entry * N + row] = __float2bfloat16(sum);
        }
    }
}

template <int NumPartitions>
__global__ void __launch_bounds__(GEMV_SPLITK_BLOCK_SIZE * NumPartitions, 2)
nvfp4_mul_mat_id_splitk_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* const* __restrict__ weight_ptrs,
    const __nv_fp8_e4m3* const* __restrict__ scale_ptrs,
    const float* const* __restrict__ scale2_ptrs,
    const int* __restrict__ expert_ids,
    int top_k, int count, int N, int K) {

    int entry = blockIdx.x / N;
    int row = blockIdx.x % N;

    constexpr int TOTAL_WARPS = NumPartitions * GEMV_SPLITK_WARPS;
    int tid = threadIdx.x;
    int warp_id = tid / GEMV_WARP_SIZE;
    int lane = tid % GEMV_WARP_SIZE;

    int bid = entry / top_k;
    int eid = expert_ids[entry];
    const __nv_bfloat16* input_row  = input + (size_t)bid * K;
    const uint8_t* weight_row = weight_ptrs[eid] + (size_t)row * (K / 2);
    const __nv_fp8_e4m3* scale_row = scale_ptrs[eid] + (size_t)row * (K / NVFP4_QUANT_GROUP);
    float scale_2_val = *scale2_ptrs[eid];

    int num_k_groups = K / NVFP4_QUANT_GROUP;
    int g_thread = warp_id * GEMV_WARP_SIZE + lane;
    int g_threads = TOTAL_WARPS * GEMV_WARP_SIZE;

    float sum = 0.0f;

    for (int g = g_thread; g < num_k_groups; g += g_threads) {
        float scale = static_cast<float>(scale_row[g]) * scale_2_val;
        int k_start = g * NVFP4_QUANT_GROUP;

        const uint4* input_v4 = reinterpret_cast<const uint4*>(input_row + k_start);
        uint4 xv0 = input_v4[0];
        uint4 xv1 = input_v4[1];
        __nv_bfloat16 xb0[8], xb1[8];
        uint4_to_bf16x8(xv0, xb0);
        uint4_to_bf16x8(xv1, xb1);

        uint32_t w_lo = *reinterpret_cast<const uint32_t*>(weight_row + g * (NVFP4_QUANT_GROUP / 2));
        uint32_t w_hi = *reinterpret_cast<const uint32_t*>(weight_row + g * (NVFP4_QUANT_GROUP / 2) + 4);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t packed = (w_lo >> (j * 8)) & 0xFFu;
            sum += fp4_e2m1_decode(packed & 0x0Fu) * scale * __bfloat162float(xb0[j * 2])
                 + fp4_e2m1_decode(packed >> 4u) * scale * __bfloat162float(xb0[j * 2 + 1]);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t packed = (w_hi >> (j * 8)) & 0xFFu;
            sum += fp4_e2m1_decode(packed & 0x0Fu) * scale * __bfloat162float(xb1[j * 2])
                 + fp4_e2m1_decode(packed >> 4u) * scale * __bfloat162float(xb1[j * 2 + 1]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    __shared__ float warp_sums[TOTAL_WARPS];
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        float s = (lane < TOTAL_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = TOTAL_WARPS / 2; offset > 0; offset >>= 1) {
            s += __shfl_down_sync(0xFFFFFFFF, s, offset);
        }
        if (lane == 0) {
            output[(size_t)entry * N + row] = __float2bfloat16(s);
        }
    }
}

}

// ---------------------------------------------------------------------------
// mul_mat_id: Indexed matrix-vector multiplication for MoE expert dispatch.
//
// For each entry i in [0, count), computes:
//   output[i, :] = input[i / top_k, :] @ weights[expert_ids[i], :, :].T
//
// input:       [batch, K]          BF16
// weight_ptrs: [num_experts]       array of device pointers, each [N, K] BF16
// expert_ids:  [count]             int32
// ---------------------------------------------------------------------------

template <int RowsPerWarp>
__global__ void __launch_bounds__(GEMV_BLOCK_SIZE, 4)
bf16_mul_mat_id_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* const* __restrict__ weight_ptrs,
    const int* __restrict__ expert_ids,
    int top_k, int count, int N, int K) {

    if (N == 0 || count == 0 || K == 0) return;

    constexpr int LANES_PER_ROW = GEMV_WARP_SIZE / RowsPerWarp;
    constexpr int ROWS_PER_BLOCK = GEMV_ROWS_PER_BLOCK * RowsPerWarp;

    int num_row_groups = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int entry = blockIdx.x / num_row_groups;
    int row_group = blockIdx.x % num_row_groups;
    int warp_id = threadIdx.x / GEMV_WARP_SIZE;
    int lane = threadIdx.x % GEMV_WARP_SIZE;

    if (entry >= count) return;

    int row_in_warp = lane / LANES_PER_ROW;
    int inner_lane = lane % LANES_PER_ROW;
    int row = row_group * ROWS_PER_BLOCK + warp_id * RowsPerWarp + row_in_warp;

    int bid = entry / top_k;
    int eid = expert_ids[entry];
    const __nv_bfloat16* input_row  = input  + (size_t)bid * K;
    const __nv_bfloat16* weight_mat = weight_ptrs[eid];
    const __nv_bfloat16* weight_row = weight_mat + (size_t)row * K;

    int K_vec = K / GEMV_K_VEC;
    int K_tail_start = K_vec * GEMV_K_VEC;

    float sum = 0.0f;
    bool row_valid = row < N;

    if (row_valid) {
        const uint4* input_v4  = reinterpret_cast<const uint4*>(input_row);
        const uint4* weight_v4 = reinterpret_cast<const uint4*>(weight_row);

        for (int ki = inner_lane; ki < K_vec; ki += LANES_PER_ROW) {
            uint4 wv = weight_v4[ki];
            uint4 xv = input_v4[ki];
            __nv_bfloat16 wb[8], xb[8];
            uint4_to_bf16x8(wv, wb);
            uint4_to_bf16x8(xv, xb);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sum += __bfloat162float(wb[j]) * __bfloat162float(xb[j]);
            }
        }

        for (int k = K_tail_start + inner_lane; k < K; k += LANES_PER_ROW) {
            sum += __bfloat162float(weight_row[k]) * __bfloat162float(input_row[k]);
        }
    }

    if constexpr (RowsPerWarp == 1) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (row_valid && lane == 0) {
            output[(size_t)entry * N + row] = __float2bfloat16(sum);
        }
    } else {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
        if (row_valid && inner_lane == 0) {
            output[(size_t)entry * N + row] = __float2bfloat16(sum);
        }
    }
}

template <int NumPartitions>
__global__ void __launch_bounds__(GEMV_SPLITK_BLOCK_SIZE * NumPartitions, 2)
bf16_mul_mat_id_splitk_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* const* __restrict__ weight_ptrs,
    const int* __restrict__ expert_ids,
    int top_k, int count, int N, int K) {

    if (N == 0 || count == 0 || K == 0) return;

    int entry = blockIdx.x / N;
    int row = blockIdx.x % N;

    constexpr int TOTAL_WARPS = NumPartitions * GEMV_SPLITK_WARPS;
    int tid = threadIdx.x;
    int warp_id = tid / GEMV_WARP_SIZE;
    int lane = tid % GEMV_WARP_SIZE;

    int bid = entry / top_k;
    int eid = expert_ids[entry];
    const __nv_bfloat16* input_row  = input  + (size_t)bid * K;
    const __nv_bfloat16* weight_mat = weight_ptrs[eid];
    const __nv_bfloat16* weight_row = weight_mat + (size_t)row * K;

    int K_vec = K / GEMV_K_VEC;
    int K_tail_start = K_vec * GEMV_K_VEC;

    const uint4* input_v4  = reinterpret_cast<const uint4*>(input_row);
    const uint4* weight_v4 = reinterpret_cast<const uint4*>(weight_row);

    float sum = 0.0f;
    int g_thread = warp_id * GEMV_WARP_SIZE + lane;
    int g_threads = TOTAL_WARPS * GEMV_WARP_SIZE;
    for (int ki = g_thread; ki < K_vec; ki += g_threads) {
        uint4 wv = weight_v4[ki];
        uint4 xv = input_v4[ki];
        __nv_bfloat16 wb[8], xb[8];
        uint4_to_bf16x8(wv, wb);
        uint4_to_bf16x8(xv, xb);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            sum += __bfloat162float(wb[j]) * __bfloat162float(xb[j]);
        }
    }
    for (int k = K_tail_start + g_thread; k < K; k += g_threads) {
        sum += __bfloat162float(weight_row[k]) * __bfloat162float(input_row[k]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    __shared__ float warp_sums[TOTAL_WARPS];
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        float s = (lane < TOTAL_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = TOTAL_WARPS / 2; offset > 0; offset >>= 1) {
            s += __shfl_down_sync(0xFFFFFFFF, s, offset);
        }
        if (lane == 0) {
            output[(size_t)entry * N + row] = __float2bfloat16(s);
        }
    }
}

// ---------------------------------------------------------------------------
// NVFP4 linear split-K kernel (M>1, N<=512).
//
// Fused dequantize + GEMM for small-N prefill/verification batches.
// Replaces the cuBLAS fallback path (dequantize_to_bf16 + cublasGemmEx)
// with a single kernel launch, avoiding multiple split-K kernel overhead.
//
// Each CTA computes one output element: output[input_row, row].
// NumPartitions warps split the K dimension, then reduce via shared memory.
// No expert ID indirection (unlike nvfp4_mul_mat_id_splitk_kernel).
// ---------------------------------------------------------------------------

template <int NumPartitions>
__global__ void __launch_bounds__(GEMV_SPLITK_BLOCK_SIZE * NumPartitions, 2)
nvfp4_linear_splitk_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ fp4_weight,
    const __nv_fp8_e4m3* __restrict__ weight_scale,
    const float* __restrict__ weight_scale_2,
    int m, int n, int k) {

    int input_row = blockIdx.x / n;
    int row = blockIdx.x % n;

    constexpr int TOTAL_WARPS = NumPartitions * GEMV_SPLITK_WARPS;
    int tid = threadIdx.x;
    int warp_id = tid / GEMV_WARP_SIZE;
    int lane = tid % GEMV_WARP_SIZE;

    const __nv_bfloat16* input_row_ptr = input + (size_t)input_row * k;
    const uint8_t* weight_row = fp4_weight + (size_t)row * (k / 2);
    const __nv_fp8_e4m3* scale_row = weight_scale + (size_t)row * (k / NVFP4_QUANT_GROUP);
    float scale_2_val = *weight_scale_2;

    int num_k_groups = k / NVFP4_QUANT_GROUP;
    int g_thread = warp_id * GEMV_WARP_SIZE + lane;
    int g_threads = TOTAL_WARPS * GEMV_WARP_SIZE;

    float sum = 0.0f;

    for (int g = g_thread; g < num_k_groups; g += g_threads) {
        float scale = static_cast<float>(scale_row[g]) * scale_2_val;
        int k_start = g * NVFP4_QUANT_GROUP;

        const uint4* input_v4 = reinterpret_cast<const uint4*>(input_row_ptr + k_start);
        uint4 xv0 = input_v4[0];
        uint4 xv1 = input_v4[1];
        __nv_bfloat16 xb0[8], xb1[8];
        uint4_to_bf16x8(xv0, xb0);
        uint4_to_bf16x8(xv1, xb1);

        uint32_t w_lo = *reinterpret_cast<const uint32_t*>(weight_row + g * (NVFP4_QUANT_GROUP / 2));
        uint32_t w_hi = *reinterpret_cast<const uint32_t*>(weight_row + g * (NVFP4_QUANT_GROUP / 2) + 4);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t packed = (w_lo >> (j * 8)) & 0xFFu;
            sum += fp4_e2m1_decode(packed & 0x0Fu) * scale * __bfloat162float(xb0[j * 2])
                 + fp4_e2m1_decode(packed >> 4u) * scale * __bfloat162float(xb0[j * 2 + 1]);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t packed = (w_hi >> (j * 8)) & 0xFFu;
            sum += fp4_e2m1_decode(packed & 0x0Fu) * scale * __bfloat162float(xb1[j * 2])
                 + fp4_e2m1_decode(packed >> 4u) * scale * __bfloat162float(xb1[j * 2 + 1]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    __shared__ float warp_sums[TOTAL_WARPS];
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        float s = (lane < TOTAL_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = TOTAL_WARPS / 2; offset > 0; offset >>= 1) {
            s += __shfl_down_sync(0xFFFFFFFF, s, offset);
        }
        if (lane == 0) {
            output[(size_t)input_row * n + row] = __float2bfloat16(s);
        }
    }
}

extern "C" {

void glm_fp8_linear_decode(GlmCtx* ctx, void* bf16_out, const void* bf16_input,
                            const void* fp8_weight, const void* weight_scale,
                            int m, int n, int k) {
    cudaSetDevice(ctx->device_id);
    const __nv_bfloat16* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(weight_scale);
    if (m == 1) {
        int num_row_groups = (n + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
        int grid_size = m * num_row_groups;
        if (k >= 512) {
            fp8_dequantize_gemv_kernel<512><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
                scale_ptr, m, n, k);
        } else if (k >= 256) {
            fp8_dequantize_gemv_kernel<256><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
                scale_ptr, m, n, k);
        } else {
            fp8_dequantize_gemv_kernel<128><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
                scale_ptr, m, n, k);
        }
    } else {
        dim3 grid((n + FP8_GEMM_N_TILE - 1) / FP8_GEMM_N_TILE,
                  (m + FP8_GEMM_M_TILE - 1) / FP8_GEMM_M_TILE);
        fp8_dequantize_gemm_smem_kernel<<<grid, FP8_GEMM_BLOCK_DIM, 0, GLM_STREAM(ctx)>>>(
            reinterpret_cast<__nv_bfloat16*>(bf16_out),
            reinterpret_cast<const __nv_bfloat16*>(bf16_input),
            reinterpret_cast<const __nv_fp8_e4m3*>(fp8_weight),
            scale_ptr,
            m, n, k);
    }
}

void glm_nvfp4_linear_decode(GlmCtx* ctx, void* bf16_out, const void* bf16_input,
                               const void* fp4_weight, const void* weight_scale,
                               const float* weight_scale_2, int m, int n, int k,
                               void* bf16_workspace) {
    cudaSetDevice(ctx->device_id);
    const __nv_fp8_e4m3* scale_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(weight_scale);
    if (m == 1) {
        int num_row_groups = (n + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
        int grid_size = m * num_row_groups;
        if (k >= 512) {
            nvfp4_dequantize_gemv_kernel<512><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const uint8_t*>(fp4_weight),
                scale_ptr, weight_scale_2, m, n, k);
        } else if (k >= 256) {
            nvfp4_dequantize_gemv_kernel<256><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const uint8_t*>(fp4_weight),
                scale_ptr, weight_scale_2, m, n, k);
        } else {
            nvfp4_dequantize_gemv_kernel<128><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const uint8_t*>(fp4_weight),
                scale_ptr, weight_scale_2, m, n, k);
        }
    } else if (m > 1 && n <= NVFP4_CUBLAS_N_THRESHOLD) {
        // Small-N, M>1 path: fused NVFP4 dequantize + split-K GEMM.
        // One CTA per (row, input_row) pair, NumPartitions warps split K.
        int grid_size = m * n;
        int num_k_groups = k / NVFP4_QUANT_GROUP;
        constexpr int SPLITK_FULL_THREADS = GEMV_SPLITK_PARTITIONS * GEMV_SPLITK_WARPS * GEMV_WARP_SIZE;
        if (num_k_groups < SPLITK_FULL_THREADS) {
            constexpr int block_size = GEMV_SPLITK_BLOCK_SIZE * 2;
            nvfp4_linear_splitk_kernel<2><<<grid_size, block_size, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const uint8_t*>(fp4_weight),
                scale_ptr, weight_scale_2,
                m, n, k);
        } else {
            constexpr int block_size = GEMV_SPLITK_BLOCK_SIZE * GEMV_SPLITK_PARTITIONS;
            nvfp4_linear_splitk_kernel<4><<<grid_size, block_size, 0, GLM_STREAM(ctx)>>>(
                reinterpret_cast<__nv_bfloat16*>(bf16_out),
                reinterpret_cast<const __nv_bfloat16*>(bf16_input),
                reinterpret_cast<const uint8_t*>(fp4_weight),
                scale_ptr, weight_scale_2,
                m, n, k);
        }
    } else if (m <= NVFP4_GEMM_SMALL_M_THRESHOLD) {
        // Narrow-tile variant: more CTAs along M (better occupancy at small M)
        // and less per-block input-staging work, while still reusing decoded
        // weight tiles across NVFP4_GEMM_SMALL_M_TILE rows (unlike the GEVM
        // fallback, which would re-read the whole weight matrix per row).
        dim3 grid((n + NVFP4_GEMM_N_TILE - 1) / NVFP4_GEMM_N_TILE,
                  (m + NVFP4_GEMM_SMALL_M_TILE - 1) / NVFP4_GEMM_SMALL_M_TILE);
        nvfp4_dequantize_gemm_smem_kernel<NVFP4_GEMM_SMALL_M_TILE><<<grid, NVFP4_GEMM_SMALL_M_TILE * NVFP4_GEMM_N_TILE, 0, GLM_STREAM(ctx)>>>(
            reinterpret_cast<__nv_bfloat16*>(bf16_out),
            reinterpret_cast<const __nv_bfloat16*>(bf16_input),
            reinterpret_cast<const uint8_t*>(fp4_weight),
            scale_ptr, weight_scale_2, m, n, k);
    } else {
        dim3 grid((n + NVFP4_GEMM_N_TILE - 1) / NVFP4_GEMM_N_TILE,
                  (m + NVFP4_GEMM_M_TILE - 1) / NVFP4_GEMM_M_TILE);
        nvfp4_dequantize_gemm_smem_kernel<NVFP4_GEMM_M_TILE><<<grid, NVFP4_GEMM_BLOCK_DIM, 0, GLM_STREAM(ctx)>>>(
            reinterpret_cast<__nv_bfloat16*>(bf16_out),
            reinterpret_cast<const __nv_bfloat16*>(bf16_input),
            reinterpret_cast<const uint8_t*>(fp4_weight),
            scale_ptr, weight_scale_2, m, n, k);
    }
}

void glm_nvfp4_mul_mat_id(GlmCtx* ctx, void* output, const void* input,
                            const void* const* weight_ptrs,
                            const void* const* scale_ptrs,
                            const void* const* scale2_ptrs,
                            const int* expert_ids, int top_k,
                            int count, int N, int K) {
    cudaSetDevice(ctx->device_id);
    if (count == 0 || N == 0 || K == 0) return;

    constexpr int SPLITK_THRESHOLD = 1024;
    constexpr int SPLITK_FULL_THREADS = GEMV_SPLITK_PARTITIONS * GEMV_SPLITK_WARPS * GEMV_WARP_SIZE;
    constexpr int SMALLK_THRESHOLD = GEMV_WARP_SIZE * NVFP4_QUANT_GROUP; // K <= 512: num_k_groups <= 32
    constexpr int MIN_ROWSPERWARP_BLOCKS = 256;
    int num_k_groups = K / NVFP4_QUANT_GROUP;
    if (N < SPLITK_THRESHOLD) {
        bool splitk_underutilized = num_k_groups < SPLITK_FULL_THREADS;
        if (splitk_underutilized) {
            constexpr int ROWS_PER_BLOCK_SMALLK = GEMV_ROWS_PER_BLOCK * 2;
            int grid_rpw_smallk = count * ((N + ROWS_PER_BLOCK_SMALLK - 1) / ROWS_PER_BLOCK_SMALLK);
            if (grid_rpw_smallk >= MIN_ROWSPERWARP_BLOCKS) {
                nvfp4_mul_mat_id_kernel<2><<<grid_rpw_smallk, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                    (__nv_bfloat16*)output,
                    (const __nv_bfloat16*)input,
                    (const uint8_t* const*)weight_ptrs,
                    (const __nv_fp8_e4m3* const*)scale_ptrs,
                    (const float* const*)scale2_ptrs,
                    expert_ids, top_k,
                    count, N, K);
                return;
            }
            int grid_rpw_normalk = count * ((N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
            if (grid_rpw_normalk >= MIN_ROWSPERWARP_BLOCKS) {
                nvfp4_mul_mat_id_kernel<1><<<grid_rpw_normalk, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                    (__nv_bfloat16*)output,
                    (const __nv_bfloat16*)input,
                    (const uint8_t* const*)weight_ptrs,
                    (const __nv_fp8_e4m3* const*)scale_ptrs,
                    (const float* const*)scale2_ptrs,
                    expert_ids, top_k,
                    count, N, K);
                return;
            }
        }
        int grid_size = count * N;
        if (splitk_underutilized) {
            constexpr int block_size = GEMV_SPLITK_BLOCK_SIZE * 2;
            nvfp4_mul_mat_id_splitk_kernel<2><<<grid_size, block_size, 0, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)output,
                (const __nv_bfloat16*)input,
                (const uint8_t* const*)weight_ptrs,
                (const __nv_fp8_e4m3* const*)scale_ptrs,
                (const float* const*)scale2_ptrs,
                expert_ids, top_k,
                count, N, K);
        } else {
            constexpr int block_size = GEMV_SPLITK_BLOCK_SIZE * GEMV_SPLITK_PARTITIONS;
            nvfp4_mul_mat_id_splitk_kernel<4><<<grid_size, block_size, 0, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)output,
                (const __nv_bfloat16*)input,
                (const uint8_t* const*)weight_ptrs,
                (const __nv_fp8_e4m3* const*)scale_ptrs,
                (const float* const*)scale2_ptrs,
                expert_ids, top_k,
                count, N, K);
        }
    } else if (K <= SMALLK_THRESHOLD) {
        constexpr int ROWS_PER_BLOCK = GEMV_ROWS_PER_BLOCK * 2;
        int num_row_groups = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        int grid_size = count * num_row_groups;
        nvfp4_mul_mat_id_kernel<2><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)output,
            (const __nv_bfloat16*)input,
            (const uint8_t* const*)weight_ptrs,
            (const __nv_fp8_e4m3* const*)scale_ptrs,
            (const float* const*)scale2_ptrs,
            expert_ids, top_k,
            count, N, K);
    } else {
        int num_row_groups = (N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
        int grid_size = count * num_row_groups;
        nvfp4_mul_mat_id_kernel<1><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)output,
            (const __nv_bfloat16*)input,
            (const uint8_t* const*)weight_ptrs,
            (const __nv_fp8_e4m3* const*)scale_ptrs,
            (const float* const*)scale2_ptrs,
            expert_ids, top_k,
            count, N, K);
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
        // Split-K variant when N is small enough that the row-major kernel
        // would launch too few blocks to fill the GPU. Threshold tuned for
        // RTX PRO 6000 / sm_120 (~140 SMs). Each row-major block has
        // ROWS_PER_BLOCK=8 rows, so block count = ceil(N/8). To hit ~140
        // blocks we need N >= ~1100. Below that, use split-K (one block per
        // row, multiple warps splitting K).
        constexpr int SPLITK_THRESHOLD = 1024;
        constexpr int SPLITK_FULL_THREADS = GEMV_SPLITK_PARTITIONS * GEMV_SPLITK_WARPS * GEMV_WARP_SIZE;
        if (n < SPLITK_THRESHOLD) {
            int grid_size = batch * n;
            int K_vec = k / GEMV_K_VEC;
            if (K_vec < SPLITK_FULL_THREADS) {
                constexpr int SPLITK_BLOCK = GEMV_SPLITK_WARPS * 2 * GEMV_WARP_SIZE;
                bf16_gemv_splitk_kernel<2><<<grid_size, SPLITK_BLOCK, 0, GLM_STREAM(ctx)>>>(
                    reinterpret_cast<__nv_bfloat16*>(out),
                    reinterpret_cast<const __nv_bfloat16*>(input),
                    reinterpret_cast<const __nv_bfloat16*>(weight),
                    batch, n, k);
            } else {
                constexpr int SPLITK_BLOCK = GEMV_SPLITK_WARPS * GEMV_SPLITK_PARTITIONS * GEMV_WARP_SIZE;
                bf16_gemv_splitk_kernel<4><<<grid_size, SPLITK_BLOCK, 0, GLM_STREAM(ctx)>>>(
                    reinterpret_cast<__nv_bfloat16*>(out),
                    reinterpret_cast<const __nv_bfloat16*>(input),
                    reinterpret_cast<const __nv_bfloat16*>(weight),
                    batch, n, k);
            }
        } else {
            int num_row_groups = (n + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
            int grid_size = batch * num_row_groups;
            bf16_gemv_kernel<<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
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

void glm_mul_mat_id(GlmCtx* ctx, void* output, const void* input,
                      const void* const* weight_ptrs,
                      const int* expert_ids, int top_k,
                      int count, int N, int K) {
    cudaSetDevice(ctx->device_id);
    if (count == 0 || N == 0 || K == 0) return;

    constexpr int SPLITK_THRESHOLD = 1024;
    constexpr int SPLITK_FULL_THREADS = GEMV_SPLITK_PARTITIONS * GEMV_SPLITK_WARPS * GEMV_WARP_SIZE;
    constexpr int SMALLK_THRESHOLD = GEMV_WARP_SIZE * GEMV_K_VEC; // K <= 256: K_vec <= 32
    constexpr int MIN_ROWSPERWARP_BLOCKS = 256;
    int K_vec = K / GEMV_K_VEC;
    if (N < SPLITK_THRESHOLD) {
        bool splitk_underutilized = K_vec < SPLITK_FULL_THREADS;
        if (splitk_underutilized) {
            constexpr int ROWS_PER_BLOCK_SMALLK = GEMV_ROWS_PER_BLOCK * 2;
            int grid_rpw_smallk = count * ((N + ROWS_PER_BLOCK_SMALLK - 1) / ROWS_PER_BLOCK_SMALLK);
            if (grid_rpw_smallk >= MIN_ROWSPERWARP_BLOCKS) {
                bf16_mul_mat_id_kernel<2><<<grid_rpw_smallk, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                    (__nv_bfloat16*)output,
                    (const __nv_bfloat16*)input,
                    (const __nv_bfloat16* const*)weight_ptrs,
                    expert_ids, top_k,
                    count, N, K);
                return;
            }
            int grid_rpw_normalk = count * ((N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
            if (grid_rpw_normalk >= MIN_ROWSPERWARP_BLOCKS) {
                bf16_mul_mat_id_kernel<1><<<grid_rpw_normalk, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                    (__nv_bfloat16*)output,
                    (const __nv_bfloat16*)input,
                    (const __nv_bfloat16* const*)weight_ptrs,
                    expert_ids, top_k,
                    count, N, K);
                return;
            }
        }
        int grid_size = count * N;
        if (splitk_underutilized) {
            constexpr int block_size = GEMV_SPLITK_BLOCK_SIZE * 2;
            bf16_mul_mat_id_splitk_kernel<2><<<grid_size, block_size, 0, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)output,
                (const __nv_bfloat16*)input,
                (const __nv_bfloat16* const*)weight_ptrs,
                expert_ids, top_k,
                count, N, K);
        } else {
            constexpr int block_size = GEMV_SPLITK_BLOCK_SIZE * GEMV_SPLITK_PARTITIONS;
            bf16_mul_mat_id_splitk_kernel<4><<<grid_size, block_size, 0, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)output,
                (const __nv_bfloat16*)input,
                (const __nv_bfloat16* const*)weight_ptrs,
                expert_ids, top_k,
                count, N, K);
        }
    } else if (K <= SMALLK_THRESHOLD) {
        constexpr int ROWS_PER_BLOCK = GEMV_ROWS_PER_BLOCK * 2;
        int num_row_groups = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        int grid_size = count * num_row_groups;
        bf16_mul_mat_id_kernel<2><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)output,
            (const __nv_bfloat16*)input,
            (const __nv_bfloat16* const*)weight_ptrs,
            expert_ids, top_k,
            count, N, K);
    } else {
        int num_row_groups = (N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
        int grid_size = count * num_row_groups;
        bf16_mul_mat_id_kernel<1><<<grid_size, GEMV_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)output,
            (const __nv_bfloat16*)input,
            (const __nv_bfloat16* const*)weight_ptrs,
            expert_ids, top_k,
            count, N, K);
    }
}

} // extern "C"
