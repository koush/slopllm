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

__device__ __forceinline__ void load_frag_a_row(FragA& a, const __nv_bfloat16* smem_a, int stride, int row_offset) {
    int lane = threadIdx.x % 32;
    int row = lane & 7;
    int mat = (lane >> 3) & 1;
    int col_offset = (lane >> 4) << 3;
    uint32_t addr = __cvta_generic_to_shared(smem_a + (row_offset + mat * 8 + row) * stride + col_offset);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a.reg[0]), "=r"(a.reg[1]), "=r"(a.reg[2]), "=r"(a.reg[3])
        : "r"(addr));
}

template <int TN>
__device__ __forceinline__ void load_frag_b(FragB& b, const __nv_bfloat16* smem_b, int b_stride, int col_offset) {
    int lane = threadIdx.x % 32;
    int row = lane % 16;
    uint32_t addr = __cvta_generic_to_shared(smem_b + row * b_stride + col_offset);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(b.reg[0]), "=r"(b.reg[1])
        : "r"(addr));
}

template <int TN, int MMA_PER_WARP_N>
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

__device__ __forceinline__ void cp_async_ca_16(void* smem_ptr, const void* gmem_ptr, int pred) {
    uint32_t smem_int = __cvta_generic_to_shared(smem_ptr);
    int src_size = pred ? 16 : 0;
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
        :: "r"(smem_int), "l"(gmem_ptr), "n"(16), "r"(src_size));
}

__device__ __forceinline__ void cp_async_ca_8(void* smem_ptr, const void* gmem_ptr, int pred) {
    uint32_t smem_int = __cvta_generic_to_shared(smem_ptr);
    int src_size = pred ? 8 : 0;
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
        :: "r"(smem_int), "l"(gmem_ptr), "n"(8), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

template <int TM, int TN, bool IsNvFP4>
__global__ void __launch_bounds__(CTA_SIZE, (TN >= 128 ? 4 : (TM >= 64 ? 4 : 8)))
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
    constexpr int NUM_M_TILES = TM / MMA_M;
    constexpr int MMA_PER_WARP_N = TN / (MMA_N * WARPS_PER_CTA);
    int num_n_tiles = (N + TN - 1) / TN;
    int warp_id = threadIdx.x / 32;
    int warp_col_offset = warp_id * (MMA_N * MMA_PER_WARP_N);

    extern __shared__ uint8_t smem_buf[];
    __nv_bfloat16* smem_a_buf0 = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* smem_b = smem_a_buf0 + TM * TK;
    uint8_t* smem_fp4_buf0 = reinterpret_cast<uint8_t*>(smem_b + TK * TN);
    constexpr size_t smem_extra = IsNvFP4 ? (TN * (TK / 2) + TN) : 0;
    __nv_bfloat16* smem_a_buf1 = reinterpret_cast<__nv_bfloat16*>(smem_fp4_buf0 + smem_extra);
    uint8_t* smem_fp4_buf1 = reinterpret_cast<uint8_t*>(smem_a_buf1 + TM * TK);
    float* smem_c = reinterpret_cast<float*>(smem_a_buf1);
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

        FragC c[NUM_M_TILES * MMA_PER_WARP_N];
        #pragma unroll
        for (int i = 0; i < NUM_M_TILES * MMA_PER_WARP_N; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                c[i].reg[j] = 0.0f;
            }
        }

        if constexpr (IsNvFP4) {
            auto load_fp4_async = [&](int k_start, __nv_bfloat16* sa, uint8_t* sfp4) {
                for (int m = threadIdx.x; m < TM; m += CTA_SIZE) {
                    int row = m_start + m;
                    int pred = (row < M_e) ? 1 : 0;
                    const __nv_bfloat16* gmem_ptr = expert_input + (size_t)row * K + k_start;
                    cp_async_ca_16(sa + m * TK, gmem_ptr, pred);
                    cp_async_ca_16(sa + m * TK + 8, gmem_ptr + 8, pred);
                }
                for (int n = threadIdx.x; n < n_valid; n += CTA_SIZE) {
                    int global_n = n_start + n;
                    int pred = (global_n < N) ? 1 : 0;
                    const void* gmem_ptr = weight_base_fp4 + (size_t)global_n * (K / 2) + k_start / 2;
                    cp_async_ca_8(sfp4 + n * (TK / 2), gmem_ptr, pred);
                }
                for (int i = threadIdx.x; i < n_valid; i += CTA_SIZE) {
                    int global_n = n_start + i;
                    __nv_fp8_e4m3* smem_scale = reinterpret_cast<__nv_fp8_e4m3*>(sfp4 + n_valid * (TK / 2));
                    if (global_n < N) {
                        smem_scale[i] = scale_base[(size_t)global_n * num_k_groups + k_start / QUANT_GROUP];
                    } else {
                        smem_scale[i] = static_cast<__nv_fp8_e4m3>(0);
                    }
                }
            };

            auto dequant_fp4_warp = [&](uint8_t* sfp4) {
                __nv_fp8_e4m3* smem_scale = reinterpret_cast<__nv_fp8_e4m3*>(sfp4 + n_valid * (TK / 2));
                int lane_id = threadIdx.x % 32;
                for (int i = lane_id; i < n_warp_valid * TK; i += 32) {
                    int k = i / n_warp_valid;
                    int n_local = i % n_warp_valid;
                    int n = warp_col_offset + n_local;
                    int k_packed = k / 2;
                    uint8_t packed_byte = sfp4[n * (TK / 2) + k_packed];
                    float2 f2 = fp4x2_to_float2(packed_byte);
                    float fval = (k % 2 == 0) ? f2.x : f2.y;
                    float block_scale = static_cast<float>(smem_scale[n]);
                    float scaled_val = fval * block_scale * scale_2_val;
                    smem_b[k * TN + n] = __float2bfloat16(scaled_val);
                }
            };

            load_fp4_async(0, smem_a_buf0, smem_fp4_buf0);
            cp_async_commit();
            cp_async_wait_all();
            __syncthreads();
            dequant_fp4_warp(smem_fp4_buf0);
            __syncwarp();

            for (int k_step = 0; k_step < num_k_steps; k_step++) {
                int buf = k_step % 2;
                __nv_bfloat16* smem_a_cur = buf ? smem_a_buf1 : smem_a_buf0;

                if (k_step < num_k_steps - 1) {
                    int next_buf = 1 - buf;
                    __nv_bfloat16* smem_a_next = next_buf ? smem_a_buf1 : smem_a_buf0;
                    uint8_t* smem_fp4_next = next_buf ? smem_fp4_buf1 : smem_fp4_buf0;
                    load_fp4_async((k_step + 1) * TK, smem_a_next, smem_fp4_next);
                    cp_async_commit();
                }

                if (n_warp_valid > 0) {
                    FragA a[NUM_M_TILES];
                    #pragma unroll
                    for (int mi = 0; mi < NUM_M_TILES; mi++) {
                        load_frag_a_row(a[mi], smem_a_cur, TK, mi * MMA_M);
                    }

                    #pragma unroll
                    for (int ni = 0; ni < MMA_PER_WARP_N; ni++) {
                        if (n_warp_valid > ni * MMA_N) {
                            FragB b_frag;
                            load_frag_b<TN>(b_frag, smem_b, TN, warp_col_offset + ni * MMA_N);
                            #pragma unroll
                            for (int mi = 0; mi < NUM_M_TILES; mi++) {
                                mma_sync_bf16_f32(c[mi * MMA_PER_WARP_N + ni], a[mi], b_frag, c[mi * MMA_PER_WARP_N + ni]);
                            }
                        }
                    }
                }

                __syncthreads();

                if (k_step < num_k_steps - 1) {
                    cp_async_wait_all();
                    int next_buf = 1 - buf;
                    uint8_t* smem_fp4_next = next_buf ? smem_fp4_buf1 : smem_fp4_buf0;
                    dequant_fp4_warp(smem_fp4_next);
                    __syncwarp();
                }
            }
        } else {
            for (int k_step = 0; k_step < num_k_steps; k_step++) {
                int k_start = k_step * TK;

                for (int m = threadIdx.x; m < TM; m += CTA_SIZE) {
                    int row = m_start + m;
                    int pred = (row < M_e) ? 1 : 0;
                    const __nv_bfloat16* gmem_ptr = expert_input + (size_t)row * K + k_start;
                    cp_async_ca_16(smem_a_buf0 + m * TK, gmem_ptr, pred);
                    cp_async_ca_16(smem_a_buf0 + m * TK + 8, gmem_ptr + 8, pred);
                }

                for (int i = threadIdx.x; i < n_valid * TK; i += CTA_SIZE) {
                    int k = i / n_valid;
                    int n = i % n_valid;
                    int global_n = n_start + n;
                    __nv_bfloat16 val = __float2bfloat16(0.0f);
                    if (global_n < N) {
                        val = weight_base_bf16[(size_t)global_n * K + k_start + k];
                    }
                    smem_b[k * TN + n] = val;
                }

                cp_async_commit();
                cp_async_wait_all();
                __syncthreads();

                if (n_warp_valid > 0) {
                    FragA a[NUM_M_TILES];
                    #pragma unroll
                    for (int mi = 0; mi < NUM_M_TILES; mi++) {
                        load_frag_a_row(a[mi], smem_a_buf0, TK, mi * MMA_M);
                    }

                    #pragma unroll
                    for (int ni = 0; ni < MMA_PER_WARP_N; ni++) {
                        if (n_warp_valid > ni * MMA_N) {
                            FragB b_frag;
                            load_frag_b<TN>(b_frag, smem_b, TN, warp_col_offset + ni * MMA_N);
                            #pragma unroll
                            for (int mi = 0; mi < NUM_M_TILES; mi++) {
                                mma_sync_bf16_f32(c[mi * MMA_PER_WARP_N + ni], a[mi], b_frag, c[mi * MMA_PER_WARP_N + ni]);
                            }
                        }
                    }
                }

                __syncthreads();
            }
        }

        if (n_warp_valid > 0) {
            #pragma unroll
            for (int mi = 0; mi < NUM_M_TILES; mi++) {
                #pragma unroll
                for (int ni = 0; ni < MMA_PER_WARP_N; ni++) {
                    if (n_warp_valid > ni * MMA_N) {
                        store_frag_c<TN, MMA_PER_WARP_N>(smem_c + mi * MMA_M * TN, c[mi * MMA_PER_WARP_N + ni], warp_id, ni);
                    }
                }
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

template <int TM, int TN>
constexpr size_t nvfp4_smem_extra() { return TN * (TK / 2) + TN; }

template <int TM, int TN>
constexpr size_t mma_smem_size(int num_experts) {
    constexpr size_t smem_extra = nvfp4_smem_extra<TM, TN>();
    size_t peak_kloop = (size_t)TM * TK * 2 + (size_t)TK * TN * 2 + smem_extra + (size_t)TM * TK * 2 + smem_extra;
    size_t peak_output = (size_t)TM * TK * 2 + (size_t)TK * TN * 2 + smem_extra + (size_t)TM * TN * 4;
    size_t peak = peak_kloop > peak_output ? peak_kloop : peak_output;
    return peak + (num_experts + 1) * sizeof(int);
}

template <int TM, int TN>
static void launch_mma_kernel_nvfp4(GlmCtx* ctx, int num_experts, int N,
                                      __nv_bfloat16* sorted_output,
                                      const __nv_bfloat16* sorted_input, int K,
                                      const void* const* weight_ptrs,
                                      const void* const* scale_ptrs,
                                      const void* const* scale2_ptrs,
                                      const int* expert_offsets,
                                      int* tile_counter, cudaStream_t stream) {
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, ctx->device_id);
    int min_blocks = (TN >= 128) ? 4 : (TM >= 64 ? 4 : 8);
    int grid_size = num_SMs * min_blocks;

    size_t smem_size = mma_smem_size<TM, TN>(num_experts);

    grouped_mma_kernel<TM, TN, true><<<grid_size, CTA_SIZE, smem_size, stream>>>(
        sorted_output, sorted_input, K,
        weight_ptrs, scale_ptrs, scale2_ptrs,
        expert_offsets, num_experts, N,
        tile_counter);
}

template <int TM, int TN>
static void launch_mma_kernel_bf16(GlmCtx* ctx, int num_experts, int N,
                                      __nv_bfloat16* sorted_output,
                                      const __nv_bfloat16* sorted_input, int K,
                                      const void* const* weight_ptrs,
                                      const int* expert_offsets,
                                      int* tile_counter, cudaStream_t stream) {
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, ctx->device_id);
    int min_blocks = (TN >= 128) ? 4 : (TM >= 64 ? 4 : 8);
    int grid_size = num_SMs * min_blocks;

    size_t smem_size = (size_t)TM * TK * 2 + (size_t)TK * TN * 2 + (size_t)TM * TN * 4
                       + (num_experts + 1) * sizeof(int);

    grouped_mma_kernel<TM, TN, false><<<grid_size, CTA_SIZE, smem_size, stream>>>(
        sorted_output, sorted_input, K,
        weight_ptrs, nullptr, nullptr,
        expert_offsets, num_experts, N,
        tile_counter);
}

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

#define DEFINE_MMA_FUNC(NAME, TM_VAL, TN_VAL, QUANT) \
void NAME(GlmCtx* ctx, void* output, const void* input, \
          const void* const* weight_ptrs, \
          QUANT \
          const int* expert_ids, int top_k, \
          int count, int N, int K, \
          int num_experts, void* workspace) { \
    cudaSetDevice(ctx->device_id); \
    cudaStream_t stream = GLM_STREAM(ctx); \
    if (count == 0 || N == 0 || K == 0) return; \
    uint8_t* ws = static_cast<uint8_t*>(workspace); \
    size_t offset = 0; \
    __nv_bfloat16* sorted_input = reinterpret_cast<__nv_bfloat16*>(ws + offset); \
    offset += (size_t)count * K * 2; \
    __nv_bfloat16* sorted_output = reinterpret_cast<__nv_bfloat16*>(ws + offset); \
    offset += (size_t)count * N * 2; \
    int* expert_counts = reinterpret_cast<int*>(ws + offset); \
    offset += (size_t)num_experts * 4; \
    int* expert_offsets = reinterpret_cast<int*>(ws + offset); \
    offset += (size_t)(num_experts + 1) * 4; \
    int* sorted_to_original = reinterpret_cast<int*>(ws + offset); \
    offset += (size_t)count * 4; \
    int* tile_counter = reinterpret_cast<int*>(ws + offset); \
    dispatch_sort_scatter(ctx, input, K, expert_ids, top_k, count, num_experts, \
                          sorted_input, sorted_to_original, expert_counts, expert_offsets, stream); \
    cudaMemsetAsync(tile_counter, 0, sizeof(int), stream); \
    LAUNCH_CALL; \
    int block_size = 256; \
    int unscatter_grid = (count + block_size - 1) / block_size; \
    unscatter_output_kernel<<<unscatter_grid, block_size, 0, stream>>>( \
        reinterpret_cast<__nv_bfloat16*>(output), \
        sorted_output, N, sorted_to_original, count); \
}

#define DEFINE_NVFP4_MMA_FUNC(NAME, TM_VAL, TN_VAL) \
void NAME(GlmCtx* ctx, void* output, const void* input, \
          const void* const* weight_ptrs, \
          const void* const* scale_ptrs, \
          const void* const* scale2_ptrs, \
          const int* expert_ids, int top_k, \
          int count, int N, int K, \
          int num_experts, void* workspace) { \
    cudaSetDevice(ctx->device_id); \
    cudaStream_t stream = GLM_STREAM(ctx); \
    if (count == 0 || N == 0 || K == 0) return; \
    uint8_t* ws = static_cast<uint8_t*>(workspace); \
    size_t offset = 0; \
    __nv_bfloat16* sorted_input = reinterpret_cast<__nv_bfloat16*>(ws + offset); \
    offset += (size_t)count * K * 2; \
    __nv_bfloat16* sorted_output = reinterpret_cast<__nv_bfloat16*>(ws + offset); \
    offset += (size_t)count * N * 2; \
    int* expert_counts = reinterpret_cast<int*>(ws + offset); \
    offset += (size_t)num_experts * 4; \
    int* expert_offsets = reinterpret_cast<int*>(ws + offset); \
    offset += (size_t)(num_experts + 1) * 4; \
    int* sorted_to_original = reinterpret_cast<int*>(ws + offset); \
    offset += (size_t)count * 4; \
    int* tile_counter = reinterpret_cast<int*>(ws + offset); \
    dispatch_sort_scatter(ctx, input, K, expert_ids, top_k, count, num_experts, \
                          sorted_input, sorted_to_original, expert_counts, expert_offsets, stream); \
    cudaMemsetAsync(tile_counter, 0, sizeof(int), stream); \
    launch_mma_kernel_nvfp4<TM_VAL, TN_VAL>(ctx, num_experts, N, sorted_output, sorted_input, K, \
                                              weight_ptrs, scale_ptrs, scale2_ptrs, \
                                              expert_offsets, tile_counter, stream); \
    int block_size = 256; \
    int unscatter_grid = (count + block_size - 1) / block_size; \
    unscatter_output_kernel<<<unscatter_grid, block_size, 0, stream>>>( \
        reinterpret_cast<__nv_bfloat16*>(output), \
        sorted_output, N, sorted_to_original, count); \
}

#define DEFINE_BF16_MMA_FUNC(NAME, TM_VAL, TN_VAL) \
void NAME(GlmCtx* ctx, void* output, const void* input, \
          const void* const* weight_ptrs, \
          const int* expert_ids, int top_k, \
          int count, int N, int K, \
          int num_experts, void* workspace) { \
    cudaSetDevice(ctx->device_id); \
    cudaStream_t stream = GLM_STREAM(ctx); \
    if (count == 0 || N == 0 || K == 0) return; \
    uint8_t* ws = static_cast<uint8_t*>(workspace); \
    size_t offset = 0; \
    __nv_bfloat16* sorted_input = reinterpret_cast<__nv_bfloat16*>(ws + offset); \
    offset += (size_t)count * K * 2; \
    __nv_bfloat16* sorted_output = reinterpret_cast<__nv_bfloat16*>(ws + offset); \
    offset += (size_t)count * N * 2; \
    int* expert_counts = reinterpret_cast<int*>(ws + offset); \
    offset += (size_t)num_experts * 4; \
    int* expert_offsets = reinterpret_cast<int*>(ws + offset); \
    offset += (size_t)(num_experts + 1) * 4; \
    int* sorted_to_original = reinterpret_cast<int*>(ws + offset); \
    offset += (size_t)count * 4; \
    int* tile_counter = reinterpret_cast<int*>(ws + offset); \
    dispatch_sort_scatter(ctx, input, K, expert_ids, top_k, count, num_experts, \
                          sorted_input, sorted_to_original, expert_counts, expert_offsets, stream); \
    cudaMemsetAsync(tile_counter, 0, sizeof(int), stream); \
    launch_mma_kernel_bf16<TM_VAL, TN_VAL>(ctx, num_experts, N, sorted_output, sorted_input, K, \
                                             weight_ptrs, expert_offsets, tile_counter, stream); \
    int block_size = 256; \
    int unscatter_grid = (count + block_size - 1) / block_size; \
    unscatter_output_kernel<<<unscatter_grid, block_size, 0, stream>>>( \
        reinterpret_cast<__nv_bfloat16*>(output), \
        sorted_output, N, sorted_to_original, count); \
}

DEFINE_NVFP4_MMA_FUNC(glm_nvfp4_mul_mat_id_grouped_mma, 32, 64)
DEFINE_NVFP4_MMA_FUNC(glm_nvfp4_mul_mat_id_grouped_mma_tm64, 64, 64)
DEFINE_NVFP4_MMA_FUNC(glm_nvfp4_mul_mat_id_grouped_mma_tm32_tn128, 32, 128)
DEFINE_NVFP4_MMA_FUNC(glm_nvfp4_mul_mat_id_grouped_mma_tm16_tn128, 16, 128)

DEFINE_BF16_MMA_FUNC(glm_bf16_mul_mat_id_grouped_mma, 32, 64)
DEFINE_BF16_MMA_FUNC(glm_bf16_mul_mat_id_grouped_mma_tm64, 64, 64)
DEFINE_BF16_MMA_FUNC(glm_bf16_mul_mat_id_grouped_mma_tm32_tn128, 32, 128)
DEFINE_BF16_MMA_FUNC(glm_bf16_mul_mat_id_grouped_mma_tm16_tn128, 16, 128)

} // extern "C"
