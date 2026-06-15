#include "glm_ops.h"
#include "glm_nvfp4.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int WARPS_PER_CTA = 4;
constexpr int MMA_PER_WARP_N = 2;
constexpr int TM = MMA_M;
constexpr int TN = MMA_N * WARPS_PER_CTA * MMA_PER_WARP_N;
constexpr int TK = MMA_K;
constexpr int CTA_SIZE = WARPS_PER_CTA * 32;
constexpr int QUANT_GROUP = 16;

struct FragA { uint32_t reg[4]; };
struct FragB { uint32_t reg[2]; };
struct FragC { float reg[4]; };

__device__ __forceinline__ void mma_sync_bf16_f32(
    FragC& d, const FragA& a, const FragB& b, const FragC& c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d.reg[0]), "+f"(d.reg[1]), "+f"(d.reg[2]), "+f"(d.reg[3])
        : "r"(a.reg[0]), "r"(a.reg[1]), "r"(a.reg[2]), "r"(a.reg[3]),
          "r"(b.reg[0]), "r"(b.reg[1])
    );
}

__device__ __forceinline__ void load_frag_a(FragA& a, const __nv_bfloat16* smem_a, int stride) {
    int lane = threadIdx.x % 32;
    int row = lane & 7;
    int mat = (lane >> 3) & 1;
    int col_offset = (lane >> 4) << 3;
    uint32_t addr = __cvta_generic_to_shared(smem_a + (mat * 8 + row) * stride + col_offset);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a.reg[0]), "=r"(a.reg[1]), "=r"(a.reg[2]), "=r"(a.reg[3])
        : "r"(addr));
}

__device__ __forceinline__ void load_frag_b(FragB& b, const __nv_bfloat16* smem_b, int b_stride, int col_offset) {
    int lane = threadIdx.x % 32;
    int row = lane % 16;
    uint32_t addr = __cvta_generic_to_shared(smem_b + row * b_stride + col_offset);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(b.reg[0]), "=r"(b.reg[1])
        : "r"(addr));
}

__device__ __forceinline__ void store_frag_c(float* smem_c, const FragC& c, int warp_id, int mma_idx) {
    int lane = threadIdx.x % 32;
    int group_id = lane >> 2;
    int tid_in_group = lane & 3;
    int col_base = tid_in_group * 2;
    int col = warp_id * (MMA_N * MMA_PER_WARP_N) + mma_idx * MMA_N + col_base;
    smem_c[group_id * TN + col + 0] = c.reg[0];
    smem_c[group_id * TN + col + 1] = c.reg[1];
    smem_c[(group_id + 8) * TN + col + 0] = c.reg[2];
    smem_c[(group_id + 8) * TN + col + 1] = c.reg[3];
}

__device__ __forceinline__ void nvfp4_load_weights(
    uint8_t* smem_fp4_base,
    __nv_bfloat16* smem_b,
    int n_start, int n_valid, int N, int K, int k_start,
    const uint8_t* weight_base,
    const __nv_fp8_e4m3* scale_base,
    float scale_2_val,
    int num_k_groups)
{
    uint8_t* smem_fp4 = smem_fp4_base;
    __nv_fp8_e4m3* smem_scale = reinterpret_cast<__nv_fp8_e4m3*>(smem_fp4 + n_valid * (TK / 2));

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

    for (int i = threadIdx.x; i < n_valid * TK; i += CTA_SIZE) {
        int k = i / n_valid;
        int n = i % n_valid;
        int global_n = n_start + n;
        __nv_bfloat16 val = __float2bfloat16(0.0f);
        if (global_n < N) {
            int k_packed = k / 2;
            uint8_t packed_byte = smem_fp4[n * (TK / 2) + k_packed];
            float2 f2 = fp4x2_to_float2(packed_byte);
            float fval = (k % 2 == 0) ? f2.x : f2.y;
            float block_scale = static_cast<float>(smem_scale[n]);
            float scaled_val = fval * block_scale * scale_2_val;
            val = __float2bfloat16(scaled_val);
        }
        smem_b[k * TN + n] = val;
    }

    __syncthreads();
}

__device__ __forceinline__ void bf16_load_weights(
    uint8_t*,
    __nv_bfloat16* smem_b,
    int n_start, int n_valid, int N, int K, int k_start,
    const __nv_bfloat16* weight_base)
{
    for (int i = threadIdx.x; i < n_valid * TK; i += CTA_SIZE) {
        int k = i / n_valid;
        int n = i % n_valid;
        int global_n = n_start + n;
        __nv_bfloat16 val = __float2bfloat16(0.0f);
        if (global_n < N) {
            val = weight_base[(size_t)global_n * K + k_start + k];
        }
        smem_b[k * TN + n] = val;
    }

    __syncthreads();
}

template <bool IsNvFP4>
__global__ void __launch_bounds__(CTA_SIZE, 8)
grouped_mma_kernel(
    __nv_bfloat16* __restrict__ sorted_output,
    const __nv_bfloat16* __restrict__ sorted_input,
    int K,
    const void* const* __restrict__ weight_ptrs,
    const void* const* __restrict__ scale_ptrs,
    const void* const* __restrict__ scale2_ptrs,
    const int* __restrict__ expert_offsets,
    int num_experts,
    int N,
    int* __restrict__ tile_counter)
{
    int num_n_tiles = (N + TN - 1) / TN;
    int warp_id = threadIdx.x / 32;
    int warp_col_offset = warp_id * (MMA_N * MMA_PER_WARP_N);

    extern __shared__ uint8_t smem_buf[];
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* smem_b = smem_a + TM * TK;
    uint8_t* smem_fp4_base = reinterpret_cast<uint8_t*>(smem_b + TK * TN);
    constexpr size_t smem_extra = IsNvFP4 ? (TN * (TK / 2) + TN) : 0;
    float* smem_c = reinterpret_cast<float*>(smem_fp4_base + smem_extra);
    int* tile_prefix = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(smem_c) + TM * TN * sizeof(float));

    if (threadIdx.x == 0) {
        int cumulative = 0;
        tile_prefix[0] = 0;
        for (int e = 0; e < num_experts; e++) {
            int M_e = expert_offsets[e + 1] - expert_offsets[e];
            cumulative += ((M_e + TM - 1) / TM) * num_n_tiles;
            tile_prefix[e + 1] = cumulative;
        }
    }
    __syncthreads();

    int total_tiles = tile_prefix[num_experts];

    __shared__ int s_work_idx;

    for (;;) {
        if (threadIdx.x == 0) {
            s_work_idx = atomicAdd(tile_counter, 1);
        }
        __syncthreads();

        int work_idx = s_work_idx;
        if (work_idx >= total_tiles) break;

        int lo = 0, hi = num_experts;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (tile_prefix[mid + 1] <= work_idx) lo = mid + 1;
            else hi = mid;
        }
        int expert_id = lo;
        int local_idx = work_idx - tile_prefix[expert_id];
        int m_tile = local_idx / num_n_tiles;
        int n_tile = local_idx % num_n_tiles;

        int M_e = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
        int m_start = m_tile * TM;
        int m_valid = min(TM, M_e - m_start);
        if (m_valid <= 0) continue;

        int n_start = n_tile * TN;
        int n_valid = min(TN, N - n_start);
        if (n_valid <= 0) continue;

        int n_warp_start = n_start + warp_col_offset;
        int n_warp_valid = 0;
        if (n_warp_start < N) {
            n_warp_valid = min(MMA_N * MMA_PER_WARP_N, N - n_warp_start);
        }

        const __nv_bfloat16* expert_input = sorted_input + (size_t)expert_offsets[expert_id] * K;
        __nv_bfloat16* expert_output = sorted_output + (size_t)expert_offsets[expert_id] * N;

        float scale_2_val = 0.0f;
        const uint8_t* weight_base_fp4 = nullptr;
        const __nv_fp8_e4m3* scale_base = nullptr;
        int num_k_groups = K / QUANT_GROUP;
        const __nv_bfloat16* weight_base_bf16 = nullptr;

        if constexpr (IsNvFP4) {
            scale_2_val = *reinterpret_cast<const float* const*>(scale2_ptrs)[expert_id];
            weight_base_fp4 = reinterpret_cast<const uint8_t* const*>(weight_ptrs)[expert_id];
            scale_base = reinterpret_cast<const __nv_fp8_e4m3* const*>(scale_ptrs)[expert_id];
        } else {
            weight_base_bf16 = reinterpret_cast<const __nv_bfloat16* const*>(weight_ptrs)[expert_id];
            (void)scale_ptrs; (void)scale2_ptrs;
        }

        int num_k_steps = K / TK;

        FragC c0, c1;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            c0.reg[i] = 0.0f;
            c1.reg[i] = 0.0f;
        }

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

            __syncthreads();

            if constexpr (IsNvFP4) {
                nvfp4_load_weights(smem_fp4_base, smem_b, n_start, n_valid, N, K, k_start,
                                   weight_base_fp4, scale_base, scale_2_val, num_k_groups);
            } else {
                bf16_load_weights(nullptr, smem_b, n_start, n_valid, N, K, k_start,
                                  weight_base_bf16);
            }

            if (n_warp_valid > 0) {
                FragA a_frag;
                load_frag_a(a_frag, smem_a, TK);

                FragB b_frag0;
                load_frag_b(b_frag0, smem_b, TN, warp_col_offset);
                mma_sync_bf16_f32(c0, a_frag, b_frag0, c0);

                if (n_warp_valid > MMA_N) {
                    FragB b_frag1;
                    load_frag_b(b_frag1, smem_b, TN, warp_col_offset + MMA_N);
                    mma_sync_bf16_f32(c1, a_frag, b_frag1, c1);
                }
            }

            __syncthreads();
        }

        if (n_warp_valid > 0) {
            store_frag_c(smem_c, c0, warp_id, 0);
            if (n_warp_valid > MMA_N) {
                store_frag_c(smem_c, c1, warp_id, 1);
            }
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

__global__ void __launch_bounds__(CTA_SIZE, 4)
debug_mma_bf16_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weights,
    int M, int K, int N)
{
    int num_n_tiles = (N + TN - 1) / TN;
    int n_tile = blockIdx.x % num_n_tiles;
    int m_tile = blockIdx.x / num_n_tiles;

    int n_start = n_tile * TN;
    int n_valid = min(TN, N - n_start);
    if (n_valid <= 0) return;

    int m_start = m_tile * TM;
    int m_valid = min(TM, M - m_start);
    if (m_valid <= 0) return;

    int warp_id = threadIdx.x / 32;
    int warp_col_offset = warp_id * (MMA_N * MMA_PER_WARP_N);
    int n_warp_start = n_start + warp_col_offset;
    int n_warp_valid = 0;
    if (n_warp_start < N) {
        n_warp_valid = min(MMA_N * MMA_PER_WARP_N, N - n_warp_start);
    }

    extern __shared__ uint8_t smem_buf[];
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* smem_b = smem_a + TM * TK;
    float* smem_c = reinterpret_cast<float*>(smem_b + TK * TN);

    int num_k_steps = K / TK;

    FragC c0, c1;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c0.reg[i] = 0.0f;
        c1.reg[i] = 0.0f;
    }

    for (int k_step = 0; k_step < num_k_steps; k_step++) {
        int k_start = k_step * TK;

        for (int i = threadIdx.x; i < TM * TK; i += CTA_SIZE) {
            int m = i / TK;
            int k = i % TK;
            int row = m_start + m;
            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (row < M) {
                val = input[(size_t)row * K + k_start + k];
            }
            smem_a[m * TK + k] = val;
        }

        for (int i = threadIdx.x; i < n_valid * TK; i += CTA_SIZE) {
            int k = i / n_valid;
            int n = i % n_valid;
            int global_n = n_start + n;
            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_n < N) {
                val = weights[(size_t)global_n * K + k_start + k];
            }
            smem_b[k * TN + n] = val;
        }

        __syncthreads();

        if (n_warp_valid > 0) {
            FragA a_frag;
            load_frag_a(a_frag, smem_a, TK);

            FragB b_frag0;
            load_frag_b(b_frag0, smem_b, TN, warp_col_offset);
            mma_sync_bf16_f32(c0, a_frag, b_frag0, c0);

            if (n_warp_valid > MMA_N) {
                FragB b_frag1;
                load_frag_b(b_frag1, smem_b, TN, warp_col_offset + MMA_N);
                mma_sync_bf16_f32(c1, a_frag, b_frag1, c1);
            }
        }

        __syncthreads();
    }

    if (n_warp_valid > 0) {
        store_frag_c(smem_c, c0, warp_id, 0);
        if (n_warp_valid > MMA_N) {
            store_frag_c(smem_c, c1, warp_id, 1);
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < m_valid * n_valid; i += CTA_SIZE) {
        int m = i / n_valid;
        int n = i % n_valid;
        float val = smem_c[m * TN + n];
        int row = m_start + m;
        int col = n_start + n;
        if (row < M && col < N) {
            output[(size_t)row * N + col] = val;
        }
    }
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

constexpr size_t nvfp4_smem_extra() { return TN * (TK / 2) + TN; }
constexpr size_t bf16_smem_extra() { return 0; }

} // namespace

extern "C" {

size_t glm_mma_moe_workspace_size(int count, int N, int K, int num_experts) {
    size_t sorted_input = (size_t)count * K * 2;
    size_t sorted_output = (size_t)count * N * 2;
    size_t expert_counts = (size_t)num_experts * 4;
    size_t expert_offsets = (size_t)(num_experts + 1) * 4;
    size_t sorted_to_original = (size_t)count * 4;
    size_t tile_counter = 4;
    return sorted_input + sorted_output + expert_counts + expert_offsets + sorted_to_original + tile_counter;
}

static void dispatch_sort_scatter(GlmCtx* ctx, const void* input, int K,
                                   const int* expert_ids, int top_k, int count,
                                   int num_experts,
                                   __nv_bfloat16* sorted_input, int* sorted_to_original,
                                   int* expert_counts, int* expert_offsets,
                                   cudaStream_t stream) {
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
    offset += (size_t)count * 4;

    int* tile_counter = reinterpret_cast<int*>(ws + offset);

    dispatch_sort_scatter(ctx, input, K, expert_ids, top_k, count, num_experts,
                          sorted_input, sorted_to_original, expert_counts, expert_offsets, stream);

    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, ctx->device_id);
    int grid_size = num_SMs * 8;

    cudaMemsetAsync(tile_counter, 0, sizeof(int), stream);

    size_t smem_size = TM * TK * 2 + TK * TN * 2 + nvfp4_smem_extra() + TM * TN * 4
                       + (num_experts + 1) * sizeof(int);

    grouped_mma_kernel<true><<<grid_size, CTA_SIZE, smem_size, stream>>>(
        sorted_output, sorted_input, K,
        weight_ptrs, scale_ptrs, scale2_ptrs,
        expert_offsets, num_experts, N,
        tile_counter);

    int block_size = 256;
    int unscatter_grid = (count + block_size - 1) / block_size;
    unscatter_output_kernel<<<unscatter_grid, block_size, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output),
        sorted_output, N, sorted_to_original, count);
}

void glm_bf16_mul_mat_id_grouped_mma(GlmCtx* ctx, void* output, const void* input,
                                       const void* const* weight_ptrs,
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
    offset += (size_t)count * 4;

    int* tile_counter = reinterpret_cast<int*>(ws + offset);

    dispatch_sort_scatter(ctx, input, K, expert_ids, top_k, count, num_experts,
                          sorted_input, sorted_to_original, expert_counts, expert_offsets, stream);

    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, ctx->device_id);
    int grid_size = num_SMs * 8;

    cudaMemsetAsync(tile_counter, 0, sizeof(int), stream);

    size_t smem_size = TM * TK * 2 + TK * TN * 2 + bf16_smem_extra() + TM * TN * 4
                       + (num_experts + 1) * sizeof(int);

    grouped_mma_kernel<false><<<grid_size, CTA_SIZE, smem_size, stream>>>(
        sorted_output, sorted_input, K,
        weight_ptrs, nullptr, nullptr,
        expert_offsets, num_experts, N,
        tile_counter);

    int block_size = 256;
    int unscatter_grid = (count + block_size - 1) / block_size;
    unscatter_output_kernel<<<unscatter_grid, block_size, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output),
        sorted_output, N, sorted_to_original, count);
}

} // extern "C"

extern "C" {

void glm_mma_moe_debug2(GlmCtx* ctx, float* output, const void* input,
                         const void* weights_bf16, int M, int K, int N) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);

    int num_n_tiles = (N + TN - 1) / TN;
    int num_m_tiles = (M + TM - 1) / TM;
    int total_ctas = num_m_tiles * num_n_tiles;

    size_t smem_size = TM * TK * 2 + TK * TN * 2 + TM * TN * 4;

    debug_mma_bf16_kernel<<<total_ctas, CTA_SIZE, smem_size, stream>>>(
        output,
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<const __nv_bfloat16*>(weights_bf16),
        M, K, N);
}

void glm_mma_moe_debug(GlmCtx* ctx, float* output, const void* input,
                        const void* weights, int M, int K, int N) {
    glm_mma_moe_debug2(ctx, output, input, weights, M, K, N);
}

} // extern "C"
