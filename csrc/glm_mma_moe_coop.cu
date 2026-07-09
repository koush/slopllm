// Cooperative MoE NVFP4 grouped GEMM kernel.
//
// Replaces the producer/consumer mbarrier split of glm_mma_moe_pc.cu with a
// single-role cooperative design (B12X-style): ALL warps stage cp.async loads
// into a depth-4 smem pipeline, __syncthreads, then ALL warps run the W4A16
// NVFP4 dequant + BF16 MMA on the arrived tile. No mbarriers, no dedicated
// producer warp. This puts every warp on both the load-issue and MMA axes.
//
// Numerics are identical to the PC kernel: mxf4nvf4 block-scaled MMA dequants
// fp4->bf16 (A=ue4m3 selector), movmatrix.trans, scale_2 mul, then
// m16n8k16 bf16 MMA against the ldmatrix'd activation.
//
// Template parameters:
//   TM          - M tile size (rows per CTA tile)
//   TN          - N tile size (cols per CTA tile)
//   DEPTH       - cp.async pipeline depth
//   NWARPS      - number of warps per CTA
//   MaxExperts  - max number of experts
//   SPLIT_M     - if true, warps split M dimension; if false, split N dimension
//   TK          - K tile size (BF16 elements per k-step)

#include "glm_ops.h"
#include "glm_nvfp4.cuh"

#include <cuda/ptx>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace ptx = cuda::ptx;

namespace {

constexpr int QUANT_GROUP = 16;
constexpr int SCALE_BATCH = 32;

__global__ void histogram_kernel(const int* __restrict__ expert_ids, int count,
                                  int* __restrict__ expert_counts, int num_experts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        int eid = expert_ids[i];
        if (eid >= 0 && eid < num_experts) atomicAdd(&expert_counts[eid], 1);
    }
}

__global__ void prefix_sum_kernel(const int* __restrict__ expert_counts,
                                   int* __restrict__ expert_offsets, int num_experts) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        expert_offsets[0] = 0;
        for (int i = 0; i < num_experts; i++)
            expert_offsets[i + 1] = expert_offsets[i] + expert_counts[i];
    }
}

__global__ void scatter_input_kernel(const int* __restrict__ expert_ids, int count, int top_k,
                                     const __nv_bfloat16* __restrict__ input, int K,
                                     __nv_bfloat16* __restrict__ sorted_input,
                                     int* __restrict__ sorted_to_original,
                                     int* __restrict__ expert_offsets) {
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
    for (int k = 0; k < num_uint4; k++) dst_v4[k] = src_v4[k];
    for (int k = num_uint4 * 8; k < K; k++) dst[k] = src[k];
    sorted_to_original[pos] = i;
}

__global__ void restore_offsets_kernel(int* __restrict__ expert_offsets,
                                       const int* __restrict__ expert_counts, int num_experts) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_experts) expert_offsets[e] -= expert_counts[e];
}

__global__ void unscatter_output_kernel(__nv_bfloat16* __restrict__ output,
                                         const __nv_bfloat16* __restrict__ sorted_output, int N,
                                         const int* __restrict__ sorted_to_original, int count) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    int orig = sorted_to_original[pos];
    const __nv_bfloat16* src = sorted_output + (size_t)pos * N;
    __nv_bfloat16* dst = output + (size_t)orig * N;
    int num_uint4 = N / 8;
    const uint4* src_v4 = reinterpret_cast<const uint4*>(src);
    uint4* dst_v4 = reinterpret_cast<uint4*>(dst);
    for (int k = 0; k < num_uint4; k++) dst_v4[k] = src_v4[k];
    for (int k = num_uint4 * 8; k < N; k++) dst[k] = src[k];
}

__device__ __forceinline__ void cp_async_ca_16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t s = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_ca_8(void* smem_ptr, const void* gmem_ptr) {
    uint32_t s = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 8;\n" :: "r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }

template <int TM, int TN, int DEPTH, int NWARPS, int MaxExperts, bool SPLIT_M, int TK>
struct CoopSmem {
    static constexpr int K_GROUPS_PER_STEP = TK / QUANT_GROUP;
    alignas(16) __nv_bfloat16 a_buf[DEPTH][TM * TK];
    alignas(16) uint8_t fp4_buf[DEPTH][TN * (TK / 2)];
    alignas(16) __nv_fp8_e4m3 scale_batch[DEPTH][TN * SCALE_BATCH];
    int tile_prefix[MaxExperts + 1];
    int work_idx;
};

template <int TM, int TN, int DEPTH, int NWARPS, int MaxExperts, bool SPLIT_M, int TK = 32>
__global__ void __launch_bounds__(NWARPS * 32, 4)
coop_moe_kernel(const __nv_bfloat16* __restrict__ sorted_input,
                __nv_bfloat16* __restrict__ sorted_output, int K,
                const void* const* __restrict__ weight_ptrs,
                const void* const* __restrict__ scale_ptrs,
                const void* const* __restrict__ scale2_ptrs,
                const int* __restrict__ expert_offsets, int num_experts, int N,
                int* __restrict__ tile_counter) {
    constexpr int CTA_THREADS = NWARPS * 32;
    constexpr int KGPS = TK / QUANT_GROUP;
    constexpr int TOTAL_N_GROUPS = TN / 8;
    constexpr int TOTAL_M_TILES = TM / 16;

    // Warp work assignment
    constexpr int N_GROUPS_FOR_WARP = SPLIT_M ? TOTAL_N_GROUPS : (TOTAL_N_GROUPS / NWARPS);
    constexpr int M_TILES_FOR_WARP = SPLIT_M ? (TOTAL_M_TILES / NWARPS) : TOTAL_M_TILES;
    constexpr int ACC_STRIDE = 4 * N_GROUPS_FOR_WARP;
    constexpr int ACC_SIZE = M_TILES_FOR_WARP * ACC_STRIDE;

    static_assert(SPLIT_M || (TOTAL_N_GROUPS % NWARPS == 0), "N_GROUPS must divide NWARPS for N-split");
    static_assert(!SPLIT_M || (TOTAL_M_TILES % NWARPS == 0), "M_TILES must divide NWARPS for M-split");

    extern __shared__ __align__(16) uint8_t smem_buf[];
    auto* smem = reinterpret_cast<CoopSmem<TM, TN, DEPTH, NWARPS, MaxExperts, SPLIT_M, TK>*>(smem_buf);

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    int t0 = lane % 4, t1 = lane / 4;
    int num_n_tiles = (N + TN - 1) / TN;
    int num_k_groups = K / QUANT_GROUP;
    int num_k_steps = num_k_groups / KGPS;

    int my_n_start_local;
    int my_m_start_tile;
    if constexpr (SPLIT_M) {
        my_n_start_local = 0;
        my_m_start_tile = warp * M_TILES_FOR_WARP;
    } else {
        my_n_start_local = warp * N_GROUPS_FOR_WARP * 8;
        my_m_start_tile = 0;
    }

    // Compute tile_prefix once (warp 0), then sync.
    if (warp == 0 && lane == 0) {
        int cumulative = 0;
        smem->tile_prefix[0] = 0;
        for (int e = 0; e < num_experts; e++) {
            int Me = expert_offsets[e + 1] - expert_offsets[e];
            cumulative += ((Me + TM - 1) / TM) * num_n_tiles;
            smem->tile_prefix[e + 1] = cumulative;
        }
    }
    __syncthreads();
    int total_tiles = smem->tile_prefix[num_experts];

    // Per-warp MMA accumulators (persist across the K-loop of a tile).
    float frag_c_accum[ACC_SIZE] = {0};

    int ldm_row = lane & 7;
    int ldm_tile_row = (lane >> 3) & 1;
    int ldm_tile_col = lane >> 4;
    int ldm_base = (ldm_tile_col * 2 + ldm_tile_row) * 64 + ldm_row * 8;

    for (;;) {
        // Grab a tile cooperatively.
        if (tid == 0) smem->work_idx = atomicAdd(tile_counter, 1);
        __syncthreads();
        int work_idx = smem->work_idx;
        if (work_idx >= total_tiles) break;

        // Decode (all threads identical).
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
        int Me = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
        int m_start = m_tile * TM;
        int m_valid = min(TM, Me - m_start);
        int n_start = n_tile * TN;
        int n_valid = min(TN, N - n_start);
        if (m_valid <= 0 || n_valid <= 0) continue;

        const __nv_bfloat16* expert_input = sorted_input + (size_t)expert_offsets[expert_id] * K;
        const uint8_t* wbase = reinterpret_cast<const uint8_t* const*>(weight_ptrs)[expert_id];
        const __nv_fp8_e4m3* scale_base = reinterpret_cast<const __nv_fp8_e4m3* const*>(scale_ptrs)[expert_id];
        float scale_2_val = *reinterpret_cast<const float* const*>(scale2_ptrs)[expert_id];
        uint16_t s2_u16;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(s2_u16) : "f"(scale_2_val));
        uint32_t s2_pair = (uint32_t)s2_u16 | ((uint32_t)s2_u16 << 16);

        constexpr uint8_t UE4M3_ONE = 0x38;
        uint32_t sfa_packed = (uint32_t)UE4M3_ONE | ((uint32_t)UE4M3_ONE << 8)
                            | ((uint32_t)UE4M3_ONE << 16) | ((uint32_t)UE4M3_ONE << 24);

        // ---- cooperative cp.async depth-DEPTH pipeline ----
        int stage = 0;
        for (int k_group_batch = 0; k_group_batch < num_k_groups; k_group_batch += SCALE_BATCH) {
            int batch_size = min(SCALE_BATCH, num_k_groups - k_group_batch);
            int k_steps_in_batch = batch_size / KGPS;
            for (int ks = 0; ks < k_steps_in_batch; ks++) {
                int buf = stage % DEPTH;
                int k_group_idx = k_group_batch + ks * KGPS;
                int k_start = k_group_idx * QUANT_GROUP;
                __nv_bfloat16* sa = smem->a_buf[buf];
                uint8_t* sfp4 = smem->fp4_buf[buf];

                // --- stage A (swizzled for ldmatrix.x4) ---
                constexpr int A_OPS = TM * (TK / 8);
                for (int idx = tid; idx < A_OPS; idx += CTA_THREADS) {
                    int m = idx / (TK / 8);
                    int c = idx % (TK / 8);
                    int row = m_start + m;
                    if (row < Me) {
                        const __nv_bfloat16* gmem = expert_input + (size_t)row * K + k_start;
                        int block_row = m / 16, tile_row = (m % 16) / 8, local_row = m % 8, row_base = local_row * 8;
                        int block_col = c / 2, tile_col = c % 2;
                        int offset = (block_row * (TK / 16) + block_col) * 256 + (tile_col * 2 + tile_row) * 64 + row_base;
                        cp_async_ca_16(sa + offset, gmem + c * 8);
                    }
                }
                // --- stage fp4 weights ---
                for (int n = tid; n < TN; n += CTA_THREADS) {
                    int global_n = n_start + n;
                    if (global_n < N) {
                        const void* gmem = wbase + (size_t)global_n * (K / 2) + k_start / 2;
                        if constexpr (TK == 64) {
                            int swiz16 = ((n >> 2) & 1) * 16;
                            cp_async_ca_16(sfp4 + n * (TK / 2) + (0 ^ swiz16), gmem);
                            cp_async_ca_16(sfp4 + n * (TK / 2) + (16 ^ swiz16), (const char*)gmem + 16);
                        } else {
                            cp_async_ca_16(sfp4 + n * (TK / 2), gmem);
                        }
                    }
                }
                // --- stage scales (first DEPTH k-steps of each batch) ---
                if (ks < DEPTH) {
                    int batch_16b = batch_size / 16;
                    int total_16b = TN * batch_16b;
                    for (int i = tid; i < total_16b; i += CTA_THREADS) {
                        int n = i / batch_16b;
                        int b = (i % batch_16b) * 16;
                        int global_n = n_start + n;
                        if (global_n < N) {
                            const __nv_fp8_e4m3* gmem = scale_base + (size_t)global_n * num_k_groups + k_group_batch;
                            int swiz16 = ((n >> 2) & 1) * 16;
                            cp_async_ca_16(reinterpret_cast<uint4*>(smem->scale_batch[buf] + n * SCALE_BATCH + (b ^ swiz16)),
                                           reinterpret_cast<const uint4*>(gmem + b));
                        }
                    }
                    int remainder = batch_size % 16;
                    if (remainder > 0) {
                        int b = batch_16b * 16;
                        for (int n = tid; n < TN; n += CTA_THREADS) {
                            int global_n = n_start + n;
                            if (global_n < N && remainder >= 8) {
                                const __nv_fp8_e4m3* gmem = scale_base + (size_t)global_n * num_k_groups + k_group_batch;
                                int swiz16 = ((n >> 2) & 1) * 16;
                                cp_async_ca_8(reinterpret_cast<uint2*>(smem->scale_batch[buf] + n * SCALE_BATCH + (b ^ swiz16)),
                                              reinterpret_cast<const uint2*>(gmem + b));
                            }
                        }
                    }
                }

                cp_async_commit();

                // --- compute on the oldest completed buffer ---
                if (stage >= DEPTH - 1) {
                    cp_async_wait_group<DEPTH - 1>();
                    __syncthreads();
                    int done = (stage - DEPTH + 1) % DEPTH;
                    int done_kgi = (stage - DEPTH + 1) * KGPS;
                    __nv_bfloat16* csa = smem->a_buf[done];
                    uint8_t* csfp4 = smem->fp4_buf[done];
                    int sfb_shift = (done_kgi & 3) * 8;
                    int sfb_base_off = (done_kgi % SCALE_BATCH) & ~3;

                    for (int n_group = 0; n_group < N_GROUPS_FOR_WARP; n_group++) {
                        int n_start_local = my_n_start_local + n_group * 8;
                        if (n_start_local >= n_valid) break;
                        int n_col = n_start_local + t1;
                        int swiz16 = ((n_col >> 2) & 1) * 16;
                        uint32_t sfb_packed = 0, regB_0 = 0, regB_1 = 0;
                        if (n_col < n_valid) {
                            int base = n_col * SCALE_BATCH + (sfb_base_off ^ swiz16);
                            uint32_t raw = *reinterpret_cast<const uint32_t*>(
                                reinterpret_cast<const uint8_t*>(&smem->scale_batch[done][0]) + base);
                            sfb_packed = raw >> sfb_shift;
                            const uint8_t* fp4_row = csfp4 + n_col * (TK / 2);
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
                            asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
                                ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                                "{%14},{%15,%16},{%17},{%18,%19};\n"
                                : "=f"(frag_d[0]), "=f"(frag_d[1]), "=f"(frag_d[2]), "=f"(frag_d[3])
                                : "r"(regA[0]), "r"(regA[1]), "r"(regA[2]), "r"(regA[3]),
                                "r"(regB_0), "r"(regB_1),
                                "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                                "r"(sfa_packed), "h"((uint16_t)0), "h"((uint16_t)0),
                                "r"(sfb_packed), "h"((uint16_t)0), "h"((uint16_t)0));
                            uint32_t pack_01, pack_23;
                            asm("{ .reg .b16 lo,hi; cvt.rn.bf16.f32 lo,%2; cvt.rn.bf16.f32 hi,%3; mov.b32 %0,{lo,hi}; }"
                                : "=r"(pack_01) : "r"(0), "f"(frag_d[0]), "f"(frag_d[1]));
                            asm("{ .reg .b16 lo,hi; cvt.rn.bf16.f32 lo,%2; cvt.rn.bf16.f32 hi,%3; mov.b32 %0,{lo,hi}; }"
                                : "=r"(pack_23) : "r"(0), "f"(frag_d[2]), "f"(frag_d[3]));
                            uint32_t b_reg_0, b_reg_1;
                            asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(b_reg_0) : "r"(pack_01));
                            asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(b_reg_1) : "r"(pack_23));
                            asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(b_reg_0) : "r"(b_reg_0), "r"(s2_pair));
                            asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(b_reg_1) : "r"(b_reg_1), "r"(s2_pair));
                            #pragma unroll
                            for (int mt = 0; mt < M_TILES_FOR_WARP; mt++) {
                                int m_tile2 = (my_m_start_tile + mt) * 16;
                                int acc_base = mt * ACC_STRIDE + n_group * 4;
                                uint32_t a_reg[4];
                                int addr = (m_tile2 / 16 * (TK / 16) + sub_ks) * 256 + ldm_base;
                                uint32_t sp = __cvta_generic_to_shared(csa + addr);
                                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                    : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
                                    : "r"(sp));
                                asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                                    : "=f"(frag_c_accum[acc_base+0]), "=f"(frag_c_accum[acc_base+1]),
                                      "=f"(frag_c_accum[acc_base+2]), "=f"(frag_c_accum[acc_base+3])
                                    : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                                      "r"(b_reg_0), "r"(b_reg_1),
                                      "f"(frag_c_accum[acc_base+0]), "f"(frag_c_accum[acc_base+1]),
                                      "f"(frag_c_accum[acc_base+2]), "f"(frag_c_accum[acc_base+3]));
                            }
                        }
                    }
                    __syncthreads();
                }
                stage++;
            }
        }

        // ---- epilogue: drain remaining DEPTH-1 buffers ----
        cp_async_wait_all();
        __syncthreads();
        for (int j = 0; j < DEPTH - 1; j++) {
            int done_stage = stage - DEPTH + 1 + j;
            if (done_stage >= num_k_steps) break;
            int done = done_stage % DEPTH;
            int done_kgi = done_stage * KGPS;
            int sfb_shift = (done_kgi & 3) * 8;
            int sfb_base_off = (done_kgi % SCALE_BATCH) & ~3;
            __nv_bfloat16* csa = smem->a_buf[done];
            uint8_t* csfp4 = smem->fp4_buf[done];

            for (int n_group = 0; n_group < N_GROUPS_FOR_WARP; n_group++) {
                int n_start_local = my_n_start_local + n_group * 8;
                if (n_start_local >= n_valid) break;
                int n_col = n_start_local + t1;
                int swiz16 = ((n_col >> 2) & 1) * 16;
                uint32_t sfb_packed = 0, regB_0 = 0, regB_1 = 0;
                if (n_col < n_valid) {
                    int base = n_col * SCALE_BATCH + (sfb_base_off ^ swiz16);
                    uint32_t raw = *reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const uint8_t*>(&smem->scale_batch[done][0]) + base);
                    sfb_packed = raw >> sfb_shift;
                    const uint8_t* fp4_row = csfp4 + n_col * (TK / 2);
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
                    asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
                        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                        "{%14},{%15,%16},{%17},{%18,%19};\n"
                        : "=f"(frag_d[0]), "=f"(frag_d[1]), "=f"(frag_d[2]), "=f"(frag_d[3])
                        : "r"(regA[0]), "r"(regA[1]), "r"(regA[2]), "r"(regA[3]),
                        "r"(regB_0), "r"(regB_1),
                        "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                        "r"(sfa_packed), "h"((uint16_t)0), "h"((uint16_t)0),
                        "r"(sfb_packed), "h"((uint16_t)0), "h"((uint16_t)0));
                    uint32_t pack_01, pack_23;
                    asm("{ .reg .b16 lo,hi; cvt.rn.bf16.f32 lo,%2; cvt.rn.bf16.f32 hi,%3; mov.b32 %0,{lo,hi}; }"
                        : "=r"(pack_01) : "r"(0), "f"(frag_d[0]), "f"(frag_d[1]));
                    asm("{ .reg .b16 lo,hi; cvt.rn.bf16.f32 lo,%2; cvt.rn.bf16.f32 hi,%3; mov.b32 %0,{lo,hi}; }"
                        : "=r"(pack_23) : "r"(0), "f"(frag_d[2]), "f"(frag_d[3]));
                    uint32_t b_reg_0, b_reg_1;
                    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(b_reg_0) : "r"(pack_01));
                    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(b_reg_1) : "r"(pack_23));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(b_reg_0) : "r"(b_reg_0), "r"(s2_pair));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(b_reg_1) : "r"(b_reg_1), "r"(s2_pair));
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES_FOR_WARP; mt++) {
                        int m_tile2 = (my_m_start_tile + mt) * 16;
                        int acc_base = mt * ACC_STRIDE + n_group * 4;
                        uint32_t a_reg[4];
                        int addr = (m_tile2 / 16 * (TK / 16) + sub_ks) * 256 + ldm_base;
                        uint32_t sp = __cvta_generic_to_shared(csa + addr);
                        asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                            : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
                            : "r"(sp));
                        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=f"(frag_c_accum[acc_base+0]), "=f"(frag_c_accum[acc_base+1]),
                              "=f"(frag_c_accum[acc_base+2]), "=f"(frag_c_accum[acc_base+3])
                            : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                              "r"(b_reg_0), "r"(b_reg_1),
                              "f"(frag_c_accum[acc_base+0]), "f"(frag_c_accum[acc_base+1]),
                              "f"(frag_c_accum[acc_base+2]), "f"(frag_c_accum[acc_base+3]));
                    }
                }
            }
            __syncthreads();
        }

        // ---- writeback ----
        __nv_bfloat16* expert_output = sorted_output + (size_t)expert_offsets[expert_id] * N;
        for (int n_group = 0; n_group < N_GROUPS_FOR_WARP; n_group++) {
            int n_start_local = my_n_start_local + n_group * 8;
            for (int mt = 0; mt < M_TILES_FOR_WARP; mt++) {
                int m_tile2 = (my_m_start_tile + mt) * 16;
                int acc_base = mt * ACC_STRIDE + n_group * 4;
                int row0 = m_tile2 + t1, row1 = m_tile2 + t1 + 8;
                int col0 = n_start + n_start_local + 2 * t0, col1 = n_start + n_start_local + 2 * t0 + 1;
                if (row0 < m_valid && col0 < N)
                    expert_output[(size_t)(m_start + row0) * N + col0] = __float2bfloat16(frag_c_accum[acc_base + 0]);
                if (row0 < m_valid && col1 < N)
                    expert_output[(size_t)(m_start + row0) * N + col1] = __float2bfloat16(frag_c_accum[acc_base + 1]);
                if (row1 < m_valid && col0 < N)
                    expert_output[(size_t)(m_start + row1) * N + col0] = __float2bfloat16(frag_c_accum[acc_base + 2]);
                if (row1 < m_valid && col1 < N)
                    expert_output[(size_t)(m_start + row1) * N + col1] = __float2bfloat16(frag_c_accum[acc_base + 3]);
            }
        }
        for (int i = 0; i < ACC_SIZE; i++) frag_c_accum[i] = 0.0f;
        __syncthreads();
    }
}

template <int TM, int TN, int DEPTH, int NWARPS, int MaxExperts, bool SPLIT_M, int TK = 32>
static void launch_coop(GlmCtx* ctx, int num_experts, int N,
                        const __nv_bfloat16* sorted_input, __nv_bfloat16* sorted_output, int K,
                        const void* const* weight_ptrs, const void* const* scale_ptrs,
                        const void* const* scale2_ptrs, const int* expert_offsets,
                        int* tile_counter, cudaStream_t stream) {
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, ctx->device_id);
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, ctx->device_id);
    size_t smem = sizeof(CoopSmem<TM, TN, DEPTH, NWARPS, MaxExperts, SPLIT_M, TK>) + 16;
    int cta_from_smem = max_smem / (smem + 256);
    int cta_from_threads = 1536 / (NWARPS * 32);
    int cta_per_sm = cta_from_smem < cta_from_threads ? cta_from_smem : cta_from_threads;
    if (cta_per_sm < 1) cta_per_sm = 1;
    int grid = num_SMs * cta_per_sm;
    cudaFuncSetAttribute((void*)coop_moe_kernel<TM, TN, DEPTH, NWARPS, MaxExperts, SPLIT_M, TK>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    coop_moe_kernel<TM, TN, DEPTH, NWARPS, MaxExperts, SPLIT_M, TK><<<grid, NWARPS * 32, smem, stream>>>(
        sorted_input, sorted_output, K, weight_ptrs, scale_ptrs, scale2_ptrs,
        expert_offsets, num_experts, N, tile_counter);
}

static void launch_coop_configured(GlmCtx* ctx, int num_experts, int N,
                                   const __nv_bfloat16* sorted_input, __nv_bfloat16* sorted_output, int K,
                                   const void* const* weight_ptrs, const void* const* scale_ptrs,
                                   const void* const* scale2_ptrs, const int* expert_offsets,
                                   int* tile_counter, cudaStream_t stream) {
    constexpr int MaxExperts = 256;
    const char* cfg_env = getenv("GLM_COOP_CONFIG");
    std::string cfg(cfg_env ? cfg_env : "tm64_tn128_d2_nw2");

    const char* nw_env = getenv("GLM_COOP_NWARPS");
    if (nw_env && !cfg_env) {
        int nw = atoi(nw_env);
        if (nw == 2) cfg = "tm32_nw2";
        else if (nw == 8) cfg = "tm32_nw8";
        else cfg = "tm32_nw4";
    }

    if (cfg == "tm32_nw2")
        launch_coop<32, 64, 4, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm32_nw8")
        launch_coop<32, 64, 4, 8, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_nw2")
        launch_coop<64, 64, 4, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_nw4")
        launch_coop<64, 64, 4, 4, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm128_nw2")
        launch_coop<128, 64, 4, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm128_nw4")
        launch_coop<128, 64, 4, 4, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_nw2")
        launch_coop<64, 128, 4, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_nw4")
        launch_coop<64, 128, 4, 4, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_mw2")
        launch_coop<64, 64, 4, 2, MaxExperts, true>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                     weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_mw4")
        launch_coop<64, 64, 4, 4, MaxExperts, true>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                     weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm128_mw4")
        launch_coop<128, 64, 4, 4, MaxExperts, true>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_d3_nw2")
        launch_coop<64, 64, 3, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_d6_nw2")
        launch_coop<64, 64, 6, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_d2_nw2")
        launch_coop<64, 64, 2, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm128_d3_nw2")
        launch_coop<128, 64, 3, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_d3_nw2")
        launch_coop<64, 128, 3, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_d3_nw4")
        launch_coop<64, 64, 3, 4, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_d2_nw4")
        launch_coop<64, 64, 2, 4, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm128_d3_nw4")
        launch_coop<128, 64, 3, 4, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_d2_nw1")
        launch_coop<64, 64, 2, 1, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_d2_nw2")
        launch_coop<64, 128, 2, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_d2_nw1")
        launch_coop<64, 128, 2, 1, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_d3_nw1")
        launch_coop<64, 128, 3, 1, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm128_tn128_d3_nw2")
        launch_coop<128, 128, 3, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                        weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm128_tn128_d2_nw2")
        launch_coop<128, 128, 2, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                        weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn256_d3_nw2")
        launch_coop<64, 256, 3, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn256_d2_nw2")
        launch_coop<64, 256, 2, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_d2k64_nw2")
        launch_coop<64, 128, 2, 2, MaxExperts, false, 64>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                            weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_d3k64_nw2")
        launch_coop<64, 128, 3, 2, MaxExperts, false, 64>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                            weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_d2k64_nw4")
        launch_coop<64, 128, 2, 4, MaxExperts, false, 64>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                            weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_d2k64_nw2")
        launch_coop<64, 64, 2, 2, MaxExperts, false, 64>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                          weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm32_tn128_d2_nw2")
        launch_coop<32, 128, 2, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                        weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm32_tn128_d3_nw2")
        launch_coop<32, 128, 3, 2, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                        weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_d2_nw4")
        launch_coop<64, 128, 2, 4, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else if (cfg == "tm64_tn128_d3_nw4")
        launch_coop<64, 128, 3, 4, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                       weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
    else
        launch_coop<32, 64, 4, 4, MaxExperts, false>(ctx, num_experts, N, sorted_input, sorted_output, K,
                                                      weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
}

} // namespace

extern "C" {

size_t glm_mma_moe_coop_workspace_size(int count, int N, int K, int num_experts) {
    size_t sorted_input = (size_t)count * K * 2;
    size_t sorted_output = (size_t)count * N * 2;
    size_t expert_counts = (size_t)num_experts * 4;
    size_t expert_offsets = (size_t)(num_experts + 1) * 4;
    size_t sorted_to_original = (size_t)count * 4;
    size_t tile_counter = 4;
    return sorted_input + sorted_output + expert_counts + expert_offsets + sorted_to_original + tile_counter;
}

void glm_nvfp4_mul_mat_id_grouped_mma_coop(GlmCtx* ctx, void* output, const void* input,
                                           const void* const* weight_ptrs, const void* const* scale_ptrs,
                                           const void* const* scale2_ptrs, const int* expert_ids,
                                           int top_k, int count, int N, int K, int num_experts,
                                           void* workspace) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    if (count == 0 || N == 0 || K == 0) return;

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    size_t offset = 0;
    __nv_bfloat16* sorted_input = reinterpret_cast<__nv_bfloat16*>(ws + offset); offset += (size_t)count * K * 2;
    __nv_bfloat16* sorted_output = reinterpret_cast<__nv_bfloat16*>(ws + offset); offset += (size_t)count * N * 2;
    int* expert_counts = reinterpret_cast<int*>(ws + offset); offset += (size_t)num_experts * 4;
    int* expert_offsets = reinterpret_cast<int*>(ws + offset); offset += (size_t)(num_experts + 1) * 4;
    int* sorted_to_original = reinterpret_cast<int*>(ws + offset); offset += (size_t)count * 4;
    int* tile_counter = reinterpret_cast<int*>(ws + offset);

    int block = 256, grid = (count + block - 1) / block;
    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);
    histogram_kernel<<<grid, block, 0, stream>>>(expert_ids, count, expert_counts, num_experts);
    prefix_sum_kernel<<<1, 1, 0, stream>>>(expert_counts, expert_offsets, num_experts);
    scatter_input_kernel<<<grid, block, 0, stream>>>(expert_ids, count, top_k,
        reinterpret_cast<const __nv_bfloat16*>(input), K, sorted_input, sorted_to_original, expert_offsets);
    grid = (num_experts + block - 1) / block;
    restore_offsets_kernel<<<grid, block, 0, stream>>>(expert_offsets, expert_counts, num_experts);
    cudaMemsetAsync(tile_counter, 0, sizeof(int), stream);
    cudaMemsetAsync(sorted_output, 0, (size_t)count * N * 2, stream);

    launch_coop_configured(ctx, num_experts, N, sorted_input, sorted_output, K,
                           weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);

    grid = (count + block - 1) / block;
    unscatter_output_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(output),
                                                        sorted_output, N, sorted_to_original, count);
}

// ---------------------------------------------------------------------------
// Split MoE coop: scatter once, run GEMM multiple times with different weights
// (e.g. gate + up share the same scatter), unscatter each output separately.
//
// Scatter workspace layout:
//   [0, count*K*2):                        sorted_input
//   [count*K*2, +num_experts*4):           expert_counts
//   [count*K*2 + num_experts*4, +(num_experts+1)*4): expert_offsets
//   [count*K*2 + num_experts*4 + (num_experts+1)*4, +count*4): sorted_to_original
//
// GEMM workspace layout:
//   [0, count*N*2):   sorted_output
//   [count*N*2, +4):  tile_counter
// ---------------------------------------------------------------------------

size_t glm_mma_moe_coop_scatter_workspace_size(int count, int K, int num_experts) {
    return (size_t)count * K * 2
         + (size_t)num_experts * 4
         + (size_t)(num_experts + 1) * 4
         + (size_t)count * 4;
}

size_t glm_mma_moe_coop_gemm_workspace_size(int count, int N) {
    return (size_t)count * N * 2 + 4;
}

void glm_mma_moe_coop_scatter(GlmCtx* ctx, const void* input, const int* expert_ids,
                              int top_k, int count, int K, int num_experts,
                              void* scatter_workspace) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    if (count == 0 || K == 0) return;

    uint8_t* ws = static_cast<uint8_t*>(scatter_workspace);
    size_t offset = 0;
    __nv_bfloat16* sorted_input = reinterpret_cast<__nv_bfloat16*>(ws + offset); offset += (size_t)count * K * 2;
    int* expert_counts = reinterpret_cast<int*>(ws + offset); offset += (size_t)num_experts * 4;
    int* expert_offsets = reinterpret_cast<int*>(ws + offset); offset += (size_t)(num_experts + 1) * 4;
    int* sorted_to_original = reinterpret_cast<int*>(ws + offset);

    int block = 256, grid = (count + block - 1) / block;
    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);
    histogram_kernel<<<grid, block, 0, stream>>>(expert_ids, count, expert_counts, num_experts);
    prefix_sum_kernel<<<1, 1, 0, stream>>>(expert_counts, expert_offsets, num_experts);
    scatter_input_kernel<<<grid, block, 0, stream>>>(expert_ids, count, top_k,
        reinterpret_cast<const __nv_bfloat16*>(input), K, sorted_input, sorted_to_original, expert_offsets);
    grid = (num_experts + block - 1) / block;
    restore_offsets_kernel<<<grid, block, 0, stream>>>(expert_offsets, expert_counts, num_experts);
}

void glm_mma_moe_coop_gemm(GlmCtx* ctx,
                           const void* const* weight_ptrs, const void* const* scale_ptrs,
                           const void* const* scale2_ptrs,
                           int num_experts, int N, int K, int count,
                           const void* scatter_workspace, void* gemm_workspace) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    if (count == 0 || N == 0 || K == 0) return;

    const uint8_t* sws = static_cast<const uint8_t*>(scatter_workspace);
    const __nv_bfloat16* sorted_input = reinterpret_cast<const __nv_bfloat16*>(sws);
    const int* expert_offsets = reinterpret_cast<const int*>(sws + (size_t)count * K * 2 + (size_t)num_experts * 4);

    uint8_t* gws = static_cast<uint8_t*>(gemm_workspace);
    __nv_bfloat16* sorted_output = reinterpret_cast<__nv_bfloat16*>(gws);
    int* tile_counter = reinterpret_cast<int*>(gws + (size_t)count * N * 2);

    cudaMemsetAsync(tile_counter, 0, sizeof(int), stream);
    cudaMemsetAsync(sorted_output, 0, (size_t)count * N * 2, stream);

    launch_coop_configured(ctx, num_experts, N, sorted_input, sorted_output, K,
                           weight_ptrs, scale_ptrs, scale2_ptrs, expert_offsets, tile_counter, stream);
}

void glm_mma_moe_coop_unscatter(GlmCtx* ctx, void* output,
                                int count, int N, int K, int num_experts,
                                const void* scatter_workspace, const void* gemm_workspace) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    if (count == 0 || N == 0) return;

    const uint8_t* sws = static_cast<const uint8_t*>(scatter_workspace);
    const int* sorted_to_original = reinterpret_cast<const int*>(
        sws + (size_t)count * K * 2 + (size_t)num_experts * 4 + (size_t)(num_experts + 1) * 4);

    const uint8_t* gws = static_cast<const uint8_t*>(gemm_workspace);
    const __nv_bfloat16* sorted_output = reinterpret_cast<const __nv_bfloat16*>(gws);

    int block = 256, grid = (count + block - 1) / block;
    unscatter_output_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(output),
                                                        sorted_output, N, sorted_to_original, count);
}

} // extern "C"
