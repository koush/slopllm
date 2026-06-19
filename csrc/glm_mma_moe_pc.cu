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

constexpr int QUANT_GROUP = 16;
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

__device__ __forceinline__
bool mbar_test(uint64_t* bar, uint32_t phase) {
    return !ptx::mbarrier_try_wait_parity(bar, phase);
}

template <int TM, int TN, int MaxExperts, int NumBuffers = 2, int TK = 32>
struct PcSmem {
    static constexpr int K_GROUPS_PER_STEP = TK / QUANT_GROUP;
    uint64_t full[CONSUMER_WARPS][NumBuffers];
    uint64_t empty[CONSUMER_WARPS][NumBuffers];
    static constexpr int NumABuffers = NumBuffers + 1;
    alignas(16) __nv_bfloat16 a_buf[NumABuffers][TM * TK];
    alignas(16) uint8_t fp4_buf[NumBuffers][TN * (TK / 2)];
    alignas(16) __nv_fp8_e4m3 scale_batch[NumBuffers][TN * SCALE_BATCH];
    int n_valid_buf[NumABuffers];
    int ks_buf[NumABuffers];
    float scale_2_buf[NumABuffers];
    int finished;
    int tile_prefix[MaxExperts + 1];
    int tile_expert_id[NumABuffers];
    int tile_m_start[NumABuffers];
    int tile_n_start[NumABuffers];
    int tile_m_valid[NumABuffers];
    int tile_n_valid[NumABuffers];
    int num_k_groups;
};

template <int TM, int TN, int MaxExperts, int NumBuffers = 2, int TK = 32>
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
    PcSmem<TM, TN, MaxExperts, NumBuffers, TK>* smem)
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
            int k_steps_in_batch = batch_size / PcSmem<TM, TN, MaxExperts, NumBuffers, TK>::K_GROUPS_PER_STEP;

            for (int ks = 0; ks < k_steps_in_batch; ks++) {
                int buf = stage % NumBuffers;
                int abuf = stage % smem->NumABuffers;

                int k_group_idx = k_group_batch + ks * PcSmem<TM, TN, MaxExperts, NumBuffers, TK>::K_GROUPS_PER_STEP;
                int k_start = k_group_idx * QUANT_GROUP;
                __nv_bfloat16* sa = smem->a_buf[abuf];
                uint8_t* sfp4 = smem->fp4_buf[buf];

#ifndef PRODUCER_ONLY
                if (lane_id == 0) {
                    smem->n_valid_buf[abuf] = n_valid;
                    smem->ks_buf[abuf] = k_group_idx;
                    smem->scale_2_buf[abuf] = scale_2_val;
                    smem->tile_expert_id[abuf] = expert_id;
                    smem->tile_m_start[abuf] = m_start;
                    smem->tile_n_start[abuf] = n_start;
                    smem->tile_m_valid[abuf] = m_valid;
                    smem->tile_n_valid[abuf] = n_valid;
                }
#endif

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

                int last_cw_arrived = 0;

                constexpr int TOTAL_N_GROUPS = TN / 8;
                constexpr int BASE_N_GROUPS = TOTAL_N_GROUPS / CONSUMER_WARPS;
                constexpr int REM_N_GROUPS = TOTAL_N_GROUPS % CONSUMER_WARPS;
                for (int cw = 0; cw < CONSUMER_WARPS; cw++) {
#ifndef PRODUCER_ONLY
                    if (stage >= NumBuffers) {
                        mbar_wait(&smem->empty[cw][buf], empty_phase[cw][buf]);
                        empty_phase[cw][buf] ^= 1;
                    } 
#endif
                    int cw_n_start = cw * BASE_N_GROUPS * 8 + (cw < REM_N_GROUPS ? cw * 8 : REM_N_GROUPS * 8);
                    int cw_n_count = (BASE_N_GROUPS + (cw < REM_N_GROUPS ? 1 : 0)) * 8;
                    int cw_n_valid = (cw_n_start < n_valid) ? min(cw_n_start + cw_n_count, n_valid) - cw_n_start : 0;

#ifndef SKIP_FP4_LOAD
                    for (int n = cw_n_start + lane_id; n < cw_n_start + cw_n_valid; n += 32) {
                        int global_n = n_start + n;
                        if (global_n < N) {
                            const void* gmem_ptr = weight_base_fp4 + (size_t)global_n * (K / 2) + k_start / 2;
                            if constexpr (TK == 64) {
                                // Same swizzle as the consumer's regB_0/regB_1 read: swap
                                // which 16-byte half of the row each chunk lands in.
                                int swiz16 = ((n >> 2) & 1) * 16;
                                cp_async_ca_16(reinterpret_cast<uint4*>(sfp4 + n * (TK / 2) + (0 ^ swiz16)),
                                               reinterpret_cast<const uint4*>(gmem_ptr),
                                               1);
                                cp_async_ca_16(reinterpret_cast<uint4*>(sfp4 + n * (TK / 2) + (16 ^ swiz16)),
                                               reinterpret_cast<const uint4*>(gmem_ptr + 16),
                                               1);
                            } else {
                                cp_async_ca_16(reinterpret_cast<uint4*>(sfp4 + n * (TK / 2)),
                                               reinterpret_cast<const uint4*>(gmem_ptr),
                                               1);
                            }
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
                                // Same swizzle as the consumer's sfb_packed read.
                                int swiz16 = ((n >> 2) & 1) * 16;
                                cp_async_ca_16(reinterpret_cast<uint4*>(smem->scale_batch[buf] + n * SCALE_BATCH + (b ^ swiz16)),
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
                                    int swiz16 = ((n >> 2) & 1) * 16;
                                    cp_async_ca_8(reinterpret_cast<uint2*>(smem->scale_batch[buf] + n * SCALE_BATCH + (b ^ swiz16)),
                                                    reinterpret_cast<const uint2*>(gmem_scale + b),
                                                    1);
                                }
                            }
                        }
                    }
#endif

#ifndef PRODUCER_ONLY
                    // if (stage >= NumBuffers) {
                        // on last consumer or if the next consumer is not ready, wait for all pending cp.async and signal all pending consumers
                        if (cw == CONSUMER_WARPS - 1 || !mbar_test(&smem->empty[cw + 1][buf], empty_phase[cw + 1][buf])) {
                            cp_async_commit();
                            cp_async_wait_all();
                            __syncwarp();
                            if (lane_id == 0) {
                                for (int cw_start = last_cw_arrived; cw_start <= cw; cw_start++) {
                                    ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cta,
                                                        ptx::space_shared, &smem->full[cw_start][buf], 1);
                                }
                                last_cw_arrived = cw + 1;
                            }
                        }
                    // }
#else
                    cp_async_commit();
                    cp_async_wait_all();
#endif
                }

                stage++;
            }
        }
    }

    if (lane_id == 0) smem->finished = 1;
    __syncwarp();
#ifndef PRODUCER_ONLY
    if (stage == 0) {
        for (int b = 0; b < smem->NumABuffers; b++)
            smem->n_valid_buf[b] = 0;
        for (int b = 0; b < NumBuffers; b++) {
            if (lane_id == 0) {
                for (int cw = 0; cw < CONSUMER_WARPS; cw++) {
                    ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cta,
                                         ptx::space_shared, &smem->full[cw][b], 1);
                }
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
        for (int b = 0; b < smem->NumABuffers; b++)
            smem->n_valid_buf[b] = 0;
        for (int b = 0; b < NumBuffers; b++) {
            if (lane_id == 0) {
                for (int cw = 0; cw < CONSUMER_WARPS; cw++) {
                    ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cta,
                                         ptx::space_shared, &smem->full[cw][b], 1);
                }
            }
        }
    }
#endif
}

template <int TM, int TN, int MaxExperts, int NumBuffers = 2, int TK = 32, int ConsumerWarps = CONSUMER_WARPS>
__device__ void consumer(
    PcSmem<TM, TN, MaxExperts, NumBuffers, TK>* smem,
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
        int abuf = stage % smem->NumABuffers;
        mbar_wait(&smem->full[consumer_warp_id][buf], full_phase[buf]);
        full_phase[buf] ^= 1;

        bool is_finished = smem->finished != 0 && smem->n_valid_buf[abuf] == 0;
        if (is_finished) {
            __syncwarp();
            if (lane_id == 0) {
                ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cta,
                                     ptx::space_shared, &smem->empty[consumer_warp_id][buf], 1);
            }
            break;
        }

#ifndef CONSUMER_NOOP
        int n_valid = smem->n_valid_buf[abuf];
        int k_group_idx = smem->ks_buf[abuf];
        float scale_2_val = smem->scale_2_buf[abuf];
        uint16_t s2_u16;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(s2_u16) : "f"(scale_2_val));
        uint32_t s2_pair = (uint32_t)s2_u16 | ((uint32_t)s2_u16 << 16);
        uint8_t* sfp4 = smem->fp4_buf[buf];
        __nv_bfloat16* sa = smem->a_buf[abuf];

        int expert_id = smem->tile_expert_id[abuf];
        int m_start = smem->tile_m_start[abuf];
        int n_start_tile = smem->tile_n_start[abuf];
        int m_valid = smem->tile_m_valid[abuf];
        constexpr int KGPS = PcSmem<TM, TN, MaxExperts, NumBuffers, TK>::K_GROUPS_PER_STEP;
        bool is_tile_done = (k_group_idx + KGPS >= smem->num_k_groups);

        constexpr uint8_t UE4M3_ONE = 0x38;
        uint32_t sfa_packed = (uint32_t)UE4M3_ONE | ((uint32_t)UE4M3_ONE << 8)
                            | ((uint32_t)UE4M3_ONE << 16) | ((uint32_t)UE4M3_ONE << 24);

        int sfb_shift = (k_group_idx & 3) * 8;
        int sfb_base_off = (k_group_idx % SCALE_BATCH) & ~3;

        int ldm_row = lane_id & 7;
        int ldm_tile_row = (lane_id >> 3) & 1;
        int ldm_tile_col = ((lane_id >> 4) << 3) >> 3;
        int ldm_base = (ldm_tile_col * 2 + ldm_tile_row) * 64 + ldm_row * 8;

        for (int n_group = 0; n_group < my_n_groups; n_group++) {
            int n_start_local = my_n_start + n_group * 8;
            if (n_start_local >= n_valid) break;

            int n_col = n_start_local + t1;
            // Bank-conflict swizzle: each row of fp4_buf (when TK==64) and of
            // scale_batch is 32 bytes (8 words), which divides 32 banks with
            // period 4 -- n_col and n_col+4 alias the same banks at different
            // addresses. XORing the within-row offset's bit 4 (value 16) by
            // bit 2 of n_col relocates alternate n_col's data into the other
            // half of the row, which spreads all 8 n_col's across all 32
            // banks with no aliasing. Producer writes (below) apply the exact
            // same XOR to the destination offset, so the data lands wherever
            // this read expects it.
            int swiz16 = ((n_col >> 2) & 1) * 16;

            uint32_t sfb_packed = 0;
            uint32_t regB_0 = 0, regB_1 = 0;
            if (n_col < n_valid) {
                int base = n_col * SCALE_BATCH + (sfb_base_off ^ swiz16);
                uint32_t raw = *reinterpret_cast<const uint32_t*>(
                    reinterpret_cast<const uint8_t*>(&smem->scale_batch[buf][0]) + base);
                sfb_packed = raw >> sfb_shift;

                const uint8_t* fp4_row = sfp4 + n_col * (TK / 2);
                if constexpr (TK > 32) {
                    regB_0 = *reinterpret_cast<const uint32_t*>(fp4_row + (4 * t0 ^ swiz16));
                    regB_1 = *reinterpret_cast<const uint32_t*>(fp4_row + ((4 * t0 + 16) ^ swiz16));
                } else {
                    regB_0 = *reinterpret_cast<const uint32_t*>(fp4_row + 4 * t0);
                }
            }

            for (int sub_ks = 0; sub_ks < KGPS; sub_ks++) {
                int shift = sub_ks * QUANT_GROUP;

                uint32_t regA[4] = {0, 0, 0, 0};
                for (int reg = 0; reg < 4; reg++) {
                    int v1 = reg % 2, v2 = reg / 2;
                    int m = t1 + 8 * v1;
                    for (int v0 = 0; v0 < 8; v0++) {
                        int k = 8 * t0 + v0 + 32 * v2;
                        if (k == m + shift) regA[reg] |= (uint32_t)0x2 << (4 * v0);
                    }
                }

                float frag_d[4];
                asm(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
                    ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,  %1,  %2,  %3},"
                    "{%4,  %5,  %6,  %7},"
                    "{%8,  %9},"
                    "{%10, %11, %12, %13},"
                    "{%14},"
                    "{%15, %16},"
                    "{%17},"
                    "{%18, %19};\n"
                    : "=f"(frag_d[0]), "=f"(frag_d[1]), "=f"(frag_d[2]), "=f"(frag_d[3])
                    : "r"(regA[0]), "r"(regA[1]), "r"(regA[2]), "r"(regA[3]),
                    "r"(regB_0), "r"(regB_1),
                    "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                    "r"(sfa_packed), "h"((uint16_t)0), "h"((uint16_t)0),
                    "r"(sfb_packed), "h"((uint16_t)0), "h"((uint16_t)0)
                );

                uint32_t pack_01, pack_23;
                asm("{ .reg .b16 lo, hi; "
                    "cvt.rn.bf16.f32 lo, %2; cvt.rn.bf16.f32 hi, %3; mov.b32 %0, {lo, hi}; }"
                    : "=r"(pack_01) : "r"(0), "f"(frag_d[0]), "f"(frag_d[1]));
                asm("{ .reg .b16 lo, hi; "
                    "cvt.rn.bf16.f32 lo, %2; cvt.rn.bf16.f32 hi, %3; mov.b32 %0, {lo, hi}; }"
                    : "=r"(pack_23) : "r"(0), "f"(frag_d[2]), "f"(frag_d[3]));

                // pack_01/pack_23 are in mma1's CLayout (SM80_16x8_Row: thread
                // (t0,t1) holds row=t1, cols={2t0,2t0+1}). mma2's B operand
                // wants the *transpose* of that (row<->col swapped: thread
                // (t0,t1) needs n=t1, k={2t0,2t0+1}) -- exactly what
                // movmatrix.trans does in one warp-synchronous instruction,
                // replacing the previous 4 __shfl_sync calls + shift/mask/OR
                // recombination (verified algebraically equivalent: source
                // lanes (t1/2, 2t0) and (t1/2, 2t0+1) with parity-based half
                // selection, same as the old src_lo_lane/src_hi_lane/sel).
                uint32_t b_reg_0, b_reg_1;
                asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                    : "=r"(b_reg_0) : "r"(pack_01));
                asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                    : "=r"(b_reg_1) : "r"(pack_23));

                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(b_reg_0) : "r"(b_reg_0), "r"(s2_pair));
                asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(b_reg_1) : "r"(b_reg_1), "r"(s2_pair));

                #pragma unroll
                for (int m_tile = 0; m_tile < TM; m_tile += 16) {
                    uint32_t a_reg[4];
                    {
                        int addr_offset = (m_tile / 16 * (TK / 16) + sub_ks) * 256 + ldm_base;
                        uint32_t smem_ptr = __cvta_generic_to_shared(sa + addr_offset);
                        asm (
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                            "{%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
                            : "r"(smem_ptr));
                    }

                    int acc_base = (m_tile / 16) * ACC_STRIDE + n_group * 4;
                    asm (
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

template <int TM, int TN, int MaxExperts, int NumBuffers = 2, int TK = 32>
constexpr size_t pc_smem_size() {
    constexpr size_t alignment = alignof(PcSmem<TM, TN, MaxExperts, NumBuffers, TK>);
    return sizeof(PcSmem<TM, TN, MaxExperts, NumBuffers, TK>) + alignment - 1;
}

template <int TM, int TN, int MaxExperts, int NumBuffers = 2, int TK = 32>
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
    extern __shared__ __align__(alignof(PcSmem<TM, TN, MaxExperts, NumBuffers, TK>)) uint8_t smem_buf[];
    auto* smem = reinterpret_cast<PcSmem<TM, TN, MaxExperts, NumBuffers, TK>*>(smem_buf);

    if (threadIdx.x == 0) {
        for (int s = 0; s < NumBuffers; s++) {
            for (int cw = 0; cw < CONSUMER_WARPS; cw++) {
                ptx::mbarrier_init(&smem->full[cw][s], 1);
                ptx::mbarrier_init(&smem->empty[cw][s], 1);
            }
        }
        smem->finished = 0;
        smem->num_k_groups = K / QUANT_GROUP;
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;

    if (warp_id == 0) {
        producer<TM, TN, MaxExperts, NumBuffers, TK>(
            sorted_input, K, weight_ptrs, scale_ptrs, scale2_ptrs,
            expert_offsets, num_experts, N, tile_counter,
            smem);
    } else if (warp_id >= 1 && warp_id <= CONSUMER_WARPS) {
#ifndef PRODUCER_ONLY
        consumer<TM, TN, MaxExperts, NumBuffers, TK>(smem, sorted_output, expert_offsets, N);
#endif
    }
}

template <int TM, int TN, int MaxExperts, int NumBuffers = 2, int TK = 32>
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

    size_t smem_size = pc_smem_size<TM, TN, MaxExperts, NumBuffers, TK>();

    cudaFuncSetAttribute(
        (void*)grouped_mma_pc_kernel<TM, TN, MaxExperts, NumBuffers, TK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    grouped_mma_pc_kernel<TM, TN, MaxExperts, NumBuffers, TK><<<grid_size, CTA_SIZE, smem_size, stream>>>(
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
    const char* tk_env = getenv("GLM_MOE_TK");
    int tk = tk_env ? atoi(tk_env) : 32;
    if (tk == 64) {
        launch_pc_kernel<32, 64, MaxExperts, 2, 64>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                  weight_ptrs, scale_ptrs, scale2_ptrs,
                                  expert_offsets, tile_counter, stream);
    } else {
        launch_pc_kernel<32, 64, MaxExperts>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                  weight_ptrs, scale_ptrs, scale2_ptrs,
                                  expert_offsets, tile_counter, stream);
    }

    grid_size = (count + block_size - 1) / block_size;
    unscatter_output_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output),
        sorted_output, N, sorted_to_original, count);
}

} // extern "C"
