#include "glm_ops.h"
#include "glm_nvfp4.cuh"

#include <cuda/ptx>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace ptx = cuda::ptx;

namespace {

constexpr int TK = 32;
constexpr int QUANT_GROUP = 16;
constexpr int K_GROUPS_PER_STEP = TK / QUANT_GROUP;
constexpr int SCALE_BATCH = 32;

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

constexpr int CONSUMER_WARPS = 2;
constexpr int WARPS_PER_CTA = CONSUMER_WARPS + 1;
constexpr int CTA_SIZE = WARPS_PER_CTA * 32;

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

__device__ __forceinline__
void mbar_wait(uint64_t* bar, uint32_t phase) {
    while (!ptx::mbarrier_try_wait_parity(bar, phase))
        ;
}

template <int TM, int TN, int MaxExperts, int NumBuffers = 2>
struct PcSmem {
    uint64_t full[NumBuffers];
    uint64_t empty[CONSUMER_WARPS][NumBuffers];
    alignas(16) __nv_bfloat16 a_buf[NumBuffers][TM * TK];
    alignas(8) uint8_t fp4_buf[NumBuffers][TN * (TK / 2)];
    alignas(16) __nv_fp8_e4m3 scale_batch[NumBuffers][TN * SCALE_BATCH];
    int n_valid_buf[NumBuffers];
    int ks_buf[NumBuffers];
    float scale_2_buf[NumBuffers];
    int finished;
    int tile_prefix[MaxExperts + 1];
    int tile_expert_id[NumBuffers];
    int tile_m_start[NumBuffers];
    int tile_n_start[NumBuffers];
    int tile_m_valid[NumBuffers];
    int tile_n_valid[NumBuffers];
    int num_k_groups;
};

template <int TM, int TN, int MaxExperts, int NumBuffers = 2>
__device__ void producer(
    const __nv_bfloat16* __restrict__ sorted_input,
    int K,
    const void* const* __restrict__ weight_ptrs,
    const void* const* __restrict__ scale_ptrs,
    const void* const* __restrict__ scale2_ptrs,
    const int* __restrict__ expert_offsets,
    int num_experts,
    int N,
    int* __restrict__ tile_counter,
    PcSmem<TM, TN, MaxExperts, NumBuffers>* smem)
{
    int lane_id = threadIdx.x % 32;

    int num_n_tiles = (N + TN - 1) / TN;
    int num_k_groups = K / QUANT_GROUP;

    if (lane_id == 0) {
        int cumulative = 0;
        smem->tile_prefix[0] = 0;
        for (int e = 0; e < num_experts; e++) {
            int M_e = expert_offsets[e + 1] - expert_offsets[e];
            cumulative += ((M_e + TM - 1) / TM) * num_n_tiles;
            smem->tile_prefix[e + 1] = cumulative;
        }
    }
    __syncwarp();

    int total_tiles = smem->tile_prefix[num_experts];
    uint32_t empty_phase[CONSUMER_WARPS][NumBuffers] = {};
    int stage = 0;

    for (;;) {
        int work_idx;
        if (lane_id == 0) work_idx = atomicAdd(tile_counter, 1);
        work_idx = __shfl_sync(0xffffffff, work_idx, 0);
        if (work_idx >= total_tiles) break;

        int lo = 0, hi = num_experts;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (smem->tile_prefix[mid + 1] <= work_idx) lo = mid + 1;
            else hi = mid;
        }
        int expert_id = lo;
        int local_idx = work_idx - smem->tile_prefix[expert_id];
        int m_tile = local_idx / num_n_tiles;
        int n_tile = local_idx % num_n_tiles;

        int M_e = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
        int m_start = m_tile * TM;
        int m_valid = min(TM, M_e - m_start);
        if (m_valid <= 0) continue;

        int n_start = n_tile * TN;
        int n_valid = min(TN, N - n_start);
        if (n_valid <= 0) continue;

        const __nv_bfloat16* expert_input = sorted_input + (size_t)expert_offsets[expert_id] * K;
        const uint8_t* weight_base_fp4 = reinterpret_cast<const uint8_t* const*>(weight_ptrs)[expert_id];
        const __nv_fp8_e4m3* scale_base = reinterpret_cast<const __nv_fp8_e4m3* const*>(scale_ptrs)[expert_id];
        float scale_2_val = *reinterpret_cast<const float* const*>(scale2_ptrs)[expert_id];

        for (int k_group_batch = 0; k_group_batch < num_k_groups; k_group_batch += SCALE_BATCH) {
            int batch_size = min(SCALE_BATCH, num_k_groups - k_group_batch);
            int k_steps_in_batch = batch_size / K_GROUPS_PER_STEP;

            for (int ks = 0; ks < k_steps_in_batch; ks++) {
                int buf = stage % NumBuffers;

#ifndef PRODUCER_ONLY
                if (stage >= NumBuffers) {
                    for (int cw = 0; cw < CONSUMER_WARPS; cw++) {
                        mbar_wait(&smem->empty[cw][buf], empty_phase[cw][buf]);
                        empty_phase[cw][buf] ^= 1;
                    }
                }
#endif

                int k_group_idx = k_group_batch + ks * K_GROUPS_PER_STEP;
                int k_start = k_group_idx * QUANT_GROUP;
                __nv_bfloat16* sa = smem->a_buf[buf];
                uint8_t* sfp4 = smem->fp4_buf[buf];

#ifndef SKIP_ACT_LOAD
                for (int m = lane_id; m < TM; m += 32) {
                    int row = m_start + m;
                    if (row < M_e) {
                        const __nv_bfloat16* gmem_ptr = expert_input + (size_t)row * K + k_start;
                        int block_row = m / 16;
                        int tile_row = (m % 16) / 8;
                        int local_row = m % 8;
                        int row_base = local_row * 8;
                        for (int c = 0; c < TK / 8; c++) {
                            int block_col = c / 2;
                            int tile_col = c % 2;
                            int offset = (block_row * (TK / 16) + block_col) * 256
                                       + (tile_col * 2 + tile_row) * 64 + row_base;
                            cp_async_ca_16(reinterpret_cast<uint4*>(sa + offset),
                                            reinterpret_cast<const uint4*>(gmem_ptr + c * 8),
                                            1);
                        }
                    }
                }
#endif

                constexpr int TOTAL_N_GROUPS = TN / 8;
                constexpr int BASE_N_GROUPS = TOTAL_N_GROUPS / CONSUMER_WARPS;
                constexpr int REM_N_GROUPS = TOTAL_N_GROUPS % CONSUMER_WARPS;
                for (int cw = 0; cw < CONSUMER_WARPS; cw++) {
                    int cw_n_start = cw * BASE_N_GROUPS * 8 + (cw < REM_N_GROUPS ? cw * 8 : REM_N_GROUPS * 8);
                    int cw_n_count = (BASE_N_GROUPS + (cw < REM_N_GROUPS ? 1 : 0)) * 8;
                    int cw_n_valid = (cw_n_start < n_valid) ? min(cw_n_start + cw_n_count, n_valid) - cw_n_start : 0;

#ifndef SKIP_FP4_LOAD
                    for (int n = cw_n_start + lane_id; n < cw_n_start + cw_n_valid; n += 32) {
                        int global_n = n_start + n;
                        if (global_n < N) {
                            const void* gmem_ptr = weight_base_fp4 + (size_t)global_n * (K / 2) + k_start / 2;
                            cp_async_ca_16(reinterpret_cast<uint4*>(sfp4 + n * (TK / 2)),
                                           reinterpret_cast<const uint4*>(gmem_ptr),
                                           1);
                        }
                    }
#endif

#ifndef SKIP_SCALE_LOAD
                    if (ks < NumBuffers) {
                        int batch_16b = batch_size / 16;
                        int total_16b = cw_n_valid * batch_16b;
                        for (int i = lane_id; i < total_16b; i += 32) {
                            int n = cw_n_start + i / batch_16b;
                            int b = (i % batch_16b) * 16;
                            int global_n = n_start + n;
                            if (global_n < N) {
                                const __nv_fp8_e4m3* gmem_scale = scale_base + (size_t)global_n * num_k_groups + k_group_batch;
                                cp_async_ca_16(reinterpret_cast<uint4*>(smem->scale_batch[buf] + n * SCALE_BATCH + b),
                                                reinterpret_cast<const uint4*>(gmem_scale + b),
                                                1);
                            }
                        }
                        int remainder = batch_size % 16;
                        if (remainder > 0) {
                            int b = batch_16b * 16;
                            for (int n = cw_n_start + lane_id; n < cw_n_start + cw_n_valid; n += 32) {
                                int global_n = n_start + n;
                                if (global_n < N && remainder >= 8) {
                                    const __nv_fp8_e4m3* gmem_scale = scale_base + (size_t)global_n * num_k_groups + k_group_batch;
                                    cp_async_ca_8(reinterpret_cast<uint2*>(smem->scale_batch[buf] + n * SCALE_BATCH + b),
                                                    reinterpret_cast<const uint2*>(gmem_scale + b),
                                                    1);
                                }
                            }
                        }
                    }
#endif
                }

                cp_async_commit();
                cp_async_wait_all();

#ifndef PRODUCER_ONLY
                if (lane_id == 0) {
                    smem->n_valid_buf[buf] = n_valid;
                    smem->ks_buf[buf] = k_group_idx;
                    smem->scale_2_buf[buf] = scale_2_val;
                    smem->tile_expert_id[buf] = expert_id;
                    smem->tile_m_start[buf] = m_start;
                    smem->tile_n_start[buf] = n_start;
                    smem->tile_m_valid[buf] = m_valid;
                    smem->tile_n_valid[buf] = n_valid;
                }

                __syncwarp();
                if (lane_id == 0) {
                    ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cta,
                                         ptx::space_shared, &smem->full[buf], 1);
                }
#endif

                stage++;
            }
        }
    }

    if (lane_id == 0) smem->finished = 1;
    __syncwarp();
#ifndef PRODUCER_ONLY
    if (stage == 0) {
        for (int b = 0; b < NumBuffers; b++) {
            if (lane_id == 0) {
                smem->n_valid_buf[b] = 0;
                ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cta,
                                     ptx::space_shared, &smem->full[b], 1);
            }
        }
    } else {
        for (int b = 0; b < NumBuffers; b++) {
            for (int cw = 0; cw < CONSUMER_WARPS; cw++) {
                mbar_wait(&smem->empty[cw][b], empty_phase[cw][b]);
                empty_phase[cw][b] ^= 1;
            }
        }
        __syncwarp();
        for (int b = 0; b < NumBuffers; b++) {
            if (lane_id == 0) {
                smem->n_valid_buf[b] = 0;
                ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cta,
                                     ptx::space_shared, &smem->full[b], 1);
            }
        }
    }
#endif
}

template <int TM, int TN, int MaxExperts, int NumBuffers = 2, int ConsumerWarps = CONSUMER_WARPS>
__device__ void consumer(
    PcSmem<TM, TN, MaxExperts, NumBuffers>* smem,
    __nv_bfloat16* __restrict__ sorted_output,
    const int* __restrict__ expert_offsets,
    int N)
{
    constexpr int TOTAL_N_GROUPS = TN / 8;
    constexpr int BASE_N_GROUPS = TOTAL_N_GROUPS / ConsumerWarps;
    constexpr int REM_N_GROUPS = TOTAL_N_GROUPS % ConsumerWarps;
    constexpr int N_GROUPS_PER_WARP = BASE_N_GROUPS + 1;
    constexpr int ACC_STRIDE = 4 * N_GROUPS_PER_WARP;

    int lane_id = threadIdx.x % 32;
    int consumer_warp_id = (threadIdx.x / 32) - 1;
    int my_n_groups = BASE_N_GROUPS + (consumer_warp_id < REM_N_GROUPS ? 1 : 0);
    int my_n_start = consumer_warp_id * BASE_N_GROUPS * 8 + (consumer_warp_id < REM_N_GROUPS ? consumer_warp_id * 8 : REM_N_GROUPS * 8);
    uint32_t full_phase[NumBuffers] = {};
    int stage = 0;
    int t0 = lane_id % 4, t1 = lane_id / 4;

    float frag_c_accum[2 * ACC_STRIDE] = {0};

    for (;;) {
        int buf = stage % NumBuffers;
        mbar_wait(&smem->full[buf], full_phase[buf]);
        full_phase[buf] ^= 1;

        bool is_finished = smem->finished != 0 && smem->n_valid_buf[buf] == 0;
        if (is_finished) {
            __syncwarp();
            if (lane_id == 0) {
                ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cta,
                                     ptx::space_shared, &smem->empty[consumer_warp_id][buf], 1);
            }
            break;
        }

#ifndef CONSUMER_NOOP
        int n_valid = smem->n_valid_buf[buf];
        int k_group_idx = smem->ks_buf[buf];
        float scale_2_val = smem->scale_2_buf[buf];
        uint8_t* sfp4 = smem->fp4_buf[buf];
        __nv_bfloat16* sa = smem->a_buf[buf];

        int expert_id = smem->tile_expert_id[buf];
        int m_start = smem->tile_m_start[buf];
        int n_start_tile = smem->tile_n_start[buf];
        int m_valid = smem->tile_m_valid[buf];
        bool is_tile_done = (k_group_idx + K_GROUPS_PER_STEP >= smem->num_k_groups);

        for (int n_group = 0; n_group < my_n_groups; n_group++) {
            int n_start_local = my_n_start + n_group * 8;
            if (n_start_local >= n_valid) break;

        for (int sub_ks = 0; sub_ks < K_GROUPS_PER_STEP; sub_ks++) {
            int fp4_offset = sub_ks * (QUANT_GROUP / 2);
            int a_offset = sub_ks * QUANT_GROUP;
            int scale_idx = k_group_idx + sub_ks;

        uint32_t frag_a_id[4] = {0};
        {
            for (int v2 = 0; v2 < 2; v2++)
                for (int v1 = 0; v1 < 2; v1++)
                    for (int v0 = 0; v0 < 4; v0++) {
                        int m = t1 + 8 * v1;
                        int k = 4 * t0 + v0 + 16 * v2;
                        uint8_t val = (k == m) ? 0x02 : 0x00;
                        frag_a_id[v1 + 2 * v2] |= ((uint32_t)val << (v0 * 8));
                    }
            for (int i = 0; i < 4; i++) frag_a_id[i] <<= 2;
        }

        uint32_t frag_b_fp4[2] = {0};
        {
            for (int v1 = 0; v1 < 2; v1++)
                for (int v0 = 0; v0 < 4; v0++) {
                    int n = n_start_local + t1;
                    int k = 4 * t0 + v0 + 16 * v1;
                    uint8_t val = 0;
                    if (n < n_valid && k < QUANT_GROUP) {
                        int k_packed = k / 2;
                        uint8_t byte_val = sfp4[n * (TK / 2) + fp4_offset + k_packed];
                        val = (k % 2 == 0) ? (byte_val & 0xF) : (byte_val >> 4);
                    }
                    frag_b_fp4[v1] |= ((uint32_t)val << (v0 * 8));
                }
            frag_b_fp4[0] <<= 2;
            frag_b_fp4[1] <<= 2;
        }

        float frag_d[4];
        asm volatile(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(frag_d[0]), "=f"(frag_d[1]), "=f"(frag_d[2]), "=f"(frag_d[3])
            : "r"(frag_a_id[0]), "r"(frag_a_id[1]), "r"(frag_a_id[2]), "r"(frag_a_id[3]),
              "r"(frag_b_fp4[0]), "r"(frag_b_fp4[1]),
              "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
        );

        float bs0 = (n_start_local + 2 * t0 < n_valid) ?
            static_cast<float>(smem->scale_batch[buf][(n_start_local + 2 * t0) * SCALE_BATCH + (scale_idx % SCALE_BATCH)]) : 0.0f;
        float bs1 = (n_start_local + 2 * t0 + 1 < n_valid) ?
            static_cast<float>(smem->scale_batch[buf][(n_start_local + 2 * t0 + 1) * SCALE_BATCH + (scale_idx % SCALE_BATCH)]) : 0.0f;
        frag_d[0] *= bs0 * scale_2_val;
        frag_d[1] *= bs1 * scale_2_val;
        frag_d[2] *= bs0 * scale_2_val;
        frag_d[3] *= bs1 * scale_2_val;

            uint32_t pack_01, pack_23;
            asm volatile("{ .reg .b16 lo, hi; "
                "cvt.rn.bf16.f32 lo, %2; cvt.rn.bf16.f32 hi, %3; mov.b32 %0, {lo, hi}; }"
                : "=r"(pack_01) : "r"(0), "f"(frag_d[0]), "f"(frag_d[1]));
            asm volatile("{ .reg .b16 lo, hi; "
                "cvt.rn.bf16.f32 lo, %2; cvt.rn.bf16.f32 hi, %3; mov.b32 %0, {lo, hi}; }"
                : "=r"(pack_23) : "r"(0), "f"(frag_d[2]), "f"(frag_d[3]));

        int src_lo_lane = 4 * (2 * t0)     + t1 / 2;
        int src_hi_lane = 4 * (2 * t0 + 1) + t1 / 2;

        uint32_t p01_lo = __shfl_sync(0xFFFFFFFF, pack_01, src_lo_lane);
        uint32_t p23_lo = __shfl_sync(0xFFFFFFFF, pack_23, src_lo_lane);
        uint32_t p01_hi = __shfl_sync(0xFFFFFFFF, pack_01, src_hi_lane);
        uint32_t p23_hi = __shfl_sync(0xFFFFFFFF, pack_23, src_hi_lane);

        int sel = (t1 & 1) * 16;
        uint32_t b_reg_0 = (((p01_hi >> sel) & 0xFFFF) << 16) | ((p01_lo >> sel) & 0xFFFF);
        uint32_t b_reg_1 = (((p23_hi >> sel) & 0xFFFF) << 16) | ((p23_lo >> sel) & 0xFFFF);

        for (int m_tile = 0; m_tile < TM; m_tile += 16) {
            uint32_t a_reg[4];
            {
                int block_row = m_tile / 16;
                int block_col = sub_ks;
                int row = lane_id & 7;
                int mat = (lane_id >> 3) & 1;
                int col_offset = (lane_id >> 4) << 3;
                int tile_row = mat;
                int tile_col = col_offset >> 3;
                int addr_offset = (block_row * (TK / 16) + block_col) * 256
                                + (tile_col * 2 + tile_row) * 64 + row * 8;
                uint32_t smem_ptr = __cvta_generic_to_shared(sa + addr_offset);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
                    : "r"(smem_ptr));
            }

            int acc_base = (m_tile / 16) * ACC_STRIDE + n_group * 4;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10, %11, %12, %13};\n"
                : "=f"(frag_c_accum[acc_base+0]), "=f"(frag_c_accum[acc_base+1]),
                  "=f"(frag_c_accum[acc_base+2]), "=f"(frag_c_accum[acc_base+3])
                : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                  "r"(b_reg_0), "r"(b_reg_1),
                  "f"(frag_c_accum[acc_base+0]), "f"(frag_c_accum[acc_base+1]),
                  "f"(frag_c_accum[acc_base+2]), "f"(frag_c_accum[acc_base+3])
            );
        }

        } // end sub_ks loop

        } // end n_group loop
#endif // CONSUMER_NOOP

        __syncwarp();
        if (lane_id == 0) {
            ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cta,
                                 ptx::space_shared, &smem->empty[consumer_warp_id][buf], 1);
        }

#ifndef CONSUMER_NOOP
        if (is_tile_done) {
            __nv_bfloat16* expert_output = sorted_output + (size_t)expert_offsets[expert_id] * N;

            for (int n_group = 0; n_group < my_n_groups; n_group++) {
                int n_start_local = my_n_start + n_group * 8;

                for (int m_tile = 0; m_tile < TM; m_tile += 16) {
                    int acc_base = (m_tile / 16) * ACC_STRIDE + n_group * 4;
                    int row0 = m_tile + t1;
                    int row1 = m_tile + t1 + 8;
                    int col0 = n_start_tile + n_start_local + 2 * t0;
                    int col1 = n_start_tile + n_start_local + 2 * t0 + 1;

                    if (row0 < m_valid && col0 < N)
                        expert_output[(size_t)(m_start + row0) * N + col0] =
                            __float2bfloat16(frag_c_accum[acc_base + 0]);
                    if (row0 < m_valid && col1 < N)
                        expert_output[(size_t)(m_start + row0) * N + col1] =
                            __float2bfloat16(frag_c_accum[acc_base + 1]);
                    if (row1 < m_valid && col0 < N)
                        expert_output[(size_t)(m_start + row1) * N + col0] =
                            __float2bfloat16(frag_c_accum[acc_base + 2]);
                    if (row1 < m_valid && col1 < N)
                        expert_output[(size_t)(m_start + row1) * N + col1] =
                            __float2bfloat16(frag_c_accum[acc_base + 3]);
                }
            }

            for (int i = 0; i < 2 * ACC_STRIDE; i++) frag_c_accum[i] = 0.0f;
        }
#endif

        stage++;
    }
}

template <int TM, int TN, int MaxExperts, int NumBuffers = 2>
constexpr size_t pc_smem_size() {
    constexpr size_t alignment = alignof(PcSmem<TM, TN, MaxExperts, NumBuffers>);
    return sizeof(PcSmem<TM, TN, MaxExperts, NumBuffers>) + alignment - 1;
}

template <int TM, int TN, int MaxExperts, int NumBuffers = 2>
__global__ void __launch_bounds__(CTA_SIZE, 8)
grouped_mma_pc_kernel(
    const __nv_bfloat16* __restrict__ sorted_input,
    __nv_bfloat16* __restrict__ sorted_output,
    int K,
    const void* const* __restrict__ weight_ptrs,
    const void* const* __restrict__ scale_ptrs,
    const void* const* __restrict__ scale2_ptrs,
    const int* __restrict__ expert_offsets,
    int num_experts,
    int N,
    int* __restrict__ tile_counter)
{
    extern __shared__ __align__(alignof(PcSmem<TM, TN, MaxExperts, NumBuffers>)) uint8_t smem_buf[];
    auto* smem = reinterpret_cast<PcSmem<TM, TN, MaxExperts, NumBuffers>*>(smem_buf);

    if (threadIdx.x == 0) {
        for (int s = 0; s < NumBuffers; s++) {
            ptx::mbarrier_init(&smem->full[s], 1);
            for (int cw = 0; cw < CONSUMER_WARPS; cw++) {
                ptx::mbarrier_init(&smem->empty[cw][s], 1);
            }
        }
        smem->finished = 0;
        smem->num_k_groups = K / QUANT_GROUP;
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;

    if (warp_id == 0) {
        producer<TM, TN, MaxExperts, NumBuffers>(
            sorted_input, K, weight_ptrs, scale_ptrs, scale2_ptrs,
            expert_offsets, num_experts, N, tile_counter,
            smem);
    } else if (warp_id >= 1 && warp_id <= CONSUMER_WARPS) {
#ifndef PRODUCER_ONLY
        consumer<TM, TN, MaxExperts, NumBuffers>(smem, sorted_output, expert_offsets, N);
#endif
    }
}

template <int TM, int TN, int MaxExperts, int NumBuffers = 2>
static void launch_pc_kernel(GlmCtx* ctx, int num_experts, int N,
                              const __nv_bfloat16* sorted_input,
                              __nv_bfloat16* sorted_output, int K,
                              const void* const* weight_ptrs,
                              const void* const* scale_ptrs,
                              const void* const* scale2_ptrs,
                              const int* expert_offsets,
                              int* tile_counter, cudaStream_t stream) {
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, ctx->device_id);
    int grid_size = num_SMs * 8;

    size_t smem_size = pc_smem_size<TM, TN, MaxExperts, NumBuffers>();

    cudaFuncSetAttribute(
        (void*)grouped_mma_pc_kernel<TM, TN, MaxExperts, NumBuffers>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    grouped_mma_pc_kernel<TM, TN, MaxExperts, NumBuffers><<<grid_size, CTA_SIZE, smem_size, stream>>>(
        sorted_input, sorted_output, K, weight_ptrs, scale_ptrs, scale2_ptrs,
        expert_offsets, num_experts, N, tile_counter);
}

} // namespace

extern "C" {

size_t glm_mma_moe_pc_workspace_size(int count, int N, int K, int num_experts) {
    size_t sorted_input = (size_t)count * K * 2;
    size_t sorted_output = (size_t)count * N * 2;
    size_t expert_counts = (size_t)num_experts * 4;
    size_t expert_offsets = (size_t)(num_experts + 1) * 4;
    size_t sorted_to_original = (size_t)count * 4;
    size_t tile_counter = 4;
    return sorted_input + sorted_output + expert_counts + expert_offsets + sorted_to_original + tile_counter;
}

void glm_nvfp4_mul_mat_id_grouped_mma_pc(GlmCtx* ctx, void* output,
                                          const void* input,
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

    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;

    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);

    histogram_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_ids, count, expert_counts, num_experts);

    prefix_sum_kernel<<<1, 1, 0, stream>>>(
        expert_counts, expert_offsets, num_experts);

    scatter_input_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_ids, count, top_k,
        reinterpret_cast<const __nv_bfloat16*>(input), K,
        sorted_input, sorted_to_original, expert_offsets);

    grid_size = (num_experts + block_size - 1) / block_size;
    restore_offsets_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_offsets, expert_counts, num_experts);

    cudaMemsetAsync(tile_counter, 0, sizeof(int), stream);
    cudaMemsetAsync(sorted_output, 0, (size_t)count * N * 2, stream);

    constexpr int MaxExperts = 256;
    launch_pc_kernel<32, 64, MaxExperts>(ctx, num_experts, N, sorted_input, sorted_output, K,
                              weight_ptrs, scale_ptrs, scale2_ptrs,
                              expert_offsets, tile_counter, stream);

    grid_size = (count + block_size - 1) / block_size;
    unscatter_output_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output),
        sorted_output, N, sorted_to_original, count);
}

} // extern "C"
