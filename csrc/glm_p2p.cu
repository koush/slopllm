// ---------------------------------------------------------------------------
// Custom P2P "one-shot" AllReduce for small messages (single-block design).
//
// Targets PCIe-only multi-GPU systems where NCCL ring AllReduce is ~30-50 us
// per call due to multi-hop launch latency. This implementation completes a
// 10 KB AllReduce in ~5-10 us by:
//   - Mapping every peer's data buffer directly via cudaDeviceEnablePeerAccess
//     (single-process, all-GPUs-in-same-cuCtx topology).
//   - Each rank scatters its local input into a double-buffered slot
//     (selected by the call counter) before waiting for peers.
//   - Each rank publishes a data-ready flag (odd seq value) and waits for
//     all peers' data-ready flags before reading their data.
//   - Double buffering + the data-ready wait prevents any rank from getting
//     2+ calls ahead: a rank cannot complete call N+1 until all peers
//     publish data-ready for N+1, which requires them to have finished
//     call N, so the next call's slot is safe to reuse.
//   - Each rank reads from every peer's data buffer (at the current slot)
//     in parallel and sums.
//
// We use a *single block* per AllReduce — fine for hidden sizes up to
// block_size * VEC = 1024 * 8 = 8192 BF16 elements (16 KB). For Qwen3-32B
// (hidden=5120 BF16 = 10 KB) this covers the AllReduces emitted after o_proj
// and down_proj.
//
// CUDA Graph compatibility:
//   The seq counter lives in device memory. Each kernel call atomicAdds it to
//   obtain a fresh seq, so the captured graph node has no encoded seq and
//   replays produce the correct fresh seq each time.
//
// Thread-safety:
//   This is a single-stream API. Concurrent AllReduces from different streams
//   on the same instance would race on the seq counter and flag. Callers
//   should serialize via stream events.
// ---------------------------------------------------------------------------
#include "glm_ops.h"
#include "glm_p2p_common.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>

namespace cg = cooperative_groups;

namespace {

constexpr int P2P_AR_BLOCK_SIZE = 1024;
constexpr int P2P_AR_VEC_BF16 = 8;   // uint4 = 8 bf16
constexpr int P2P_AR_VEC_F32 = 4;    // uint4 = 4 fp32

// ---------------------------------------------------------------------------
// P2P data sync: scatter local data to P2P buffer, publish flag, wait for peers.
// Writes slot_offset for the subsequent multi-block compute kernel.
// Single block, shared by allreduce, allgather, etc.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(P2P_AR_BLOCK_SIZE, 1)
p2p_data_sync_kernel(
    void* const* peer_data,
    int* const* peer_flags,
    unsigned long long* my_seq_counter,
    int* slot_offset_out,
    int my_rank,
    int world_size,
    int max_slot_bytes,
    const void* __restrict__ my_input,
    int input_bytes)
{
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    __shared__ unsigned int s_seq;
    __shared__ int          s_slot_offset;
    __shared__ void*        s_peer_data[P2P_AR_MAX_WORLD];
    __shared__ int*         s_peer_flags[P2P_AR_MAX_WORLD];

    if (tid == 0) {
        unsigned long long s = atomicAdd(my_seq_counter, 2ULL) + 2ULL;
        s_seq = (unsigned int)(s & 0x7FFFFFFEu);
        if (s_seq == 0) s_seq = 2;
        s_slot_offset = ((s_seq >> 1) & 1) * max_slot_bytes;
    }
    if (tid < world_size) {
        s_peer_data[tid]  = peer_data[tid];
        s_peer_flags[tid] = peer_flags[tid];
    }
    __syncthreads();

    int seq = (int)s_seq;
    int slot_offset = s_slot_offset;

    // Scatter local input to P2P buffer
    // s_peer_data[my_rank] via smem: dynamic index, would spill registers
    char* my_data = static_cast<char*>(s_peer_data[my_rank]) + slot_offset;
    {
        const char* in_b = static_cast<const char*>(my_input);
        if (in_b != my_data) {
            int num_vec = input_bytes / 16;
            if (((size_t)in_b | (size_t)my_data) % 16 == 0 && num_vec > 0) {
                const uint4* in_v4 = reinterpret_cast<const uint4*>(in_b);
                uint4*       my_v4 = reinterpret_cast<uint4*>(my_data);
                for (int i = tid; i < num_vec; i += bs) {
                    my_v4[i] = in_v4[i];
                }
                int tail_start = num_vec * 16;
                for (int i = tail_start + tid; i < input_bytes; i += bs) {
                    my_data[i] = in_b[i];
                }
            } else {
                for (int i = tid; i < input_bytes; i += bs) {
                    my_data[i] = in_b[i];
                }
            }
        }
    }

    // Publish data-ready flag and wait for all peers
    p2p_publish_and_wait(tid, my_rank, world_size, seq,
                         s_peer_flags, s_peer_flags[my_rank]);

    // Write slot_offset for Phase 2
    if (tid == 0) {
        *slot_offset_out = slot_offset;
    }
}

// ---------------------------------------------------------------------------
// P2P barrier: increment seq, publish flag, wait for peers. No data scatter.
// Writes slot_offset so callers know which double-buffer slot was selected.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(P2P_AR_BLOCK_SIZE, 1)
p2p_barrier_kernel(
    int* const* peer_flags,
    unsigned long long* my_seq_counter,
    int* slot_offset_out,
    int my_rank,
    int world_size,
    int max_slot_bytes)
{
    int tid = threadIdx.x;

    __shared__ unsigned int s_seq;
    __shared__ int          s_slot_offset;
    __shared__ int*         s_peer_flags[P2P_AR_MAX_WORLD];

    if (tid == 0) {
        unsigned long long s = atomicAdd(my_seq_counter, 2ULL) + 2ULL;
        s_seq = (unsigned int)(s & 0x7FFFFFFEu);
        if (s_seq == 0) s_seq = 2;
        s_slot_offset = ((s_seq >> 1) & 1) * max_slot_bytes;
    }
    if (tid < world_size) {
        s_peer_flags[tid] = peer_flags[tid];
    }
    __syncthreads();

    int seq = (int)s_seq;

    p2p_publish_and_wait(tid, my_rank, world_size, seq,
                         s_peer_flags, s_peer_flags[my_rank]);

    if (tid == 0) {
        *slot_offset_out = s_slot_offset;
    }
}

// ---------------------------------------------------------------------------
// P2P AllReduce – fused single-block (scatter + sync + reduce in one kernel).
//
// Used for allreduce where the payload is small enough for a single block
// (up to block_size * VEC elements). Avoids the extra kernel launch overhead
// of the two-phase (sync + multi-block) approach.
// ---------------------------------------------------------------------------

template <typename T, int VEC>
__global__ void __launch_bounds__(P2P_AR_BLOCK_SIZE, 1)
p2p_allreduce_oneshot_kernel(
    void* const* peer_data,
    int* const* peer_flags,
    unsigned long long* my_seq_counter,
    int my_rank, int world_size, int max_slot_bytes,
    const T* __restrict__ in,
    T* __restrict__ out,
    int count)
{
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    __shared__ unsigned int s_seq;
    __shared__ int          s_slot_offset;
    __shared__ void*        s_peer_data[P2P_AR_MAX_WORLD];
    __shared__ int*         s_peer_flags[P2P_AR_MAX_WORLD];

    if (tid == 0) {
        unsigned long long s = atomicAdd(my_seq_counter, 2ULL) + 2ULL;
        s_seq = (unsigned int)(s & 0x7FFFFFFEu);
        if (s_seq == 0) s_seq = 2;
        s_slot_offset = ((s_seq >> 1) & 1) * max_slot_bytes;
    }
    if (tid < world_size) {
        s_peer_data[tid]  = peer_data[tid];
        s_peer_flags[tid] = peer_flags[tid];
    }
    __syncthreads();

    int seq = (int)s_seq;
    int slot_offset = s_slot_offset;

    // Rank-rotated peer data pointers in registers. rr is a compile-time
    // constant in the unrolled loop below, so peer_pv[rr] avoids register spill.
    // Contrast with s_peer_flags[tid] where tid is dynamic — that must stay in smem.
    const char* peer_pv[P2P_AR_MAX_WORLD];
    #pragma unroll
    for (int rr = 0; rr < P2P_AR_MAX_WORLD; ++rr) {
        int r = rr + my_rank; if (r >= world_size) r -= world_size;
        peer_pv[rr] = static_cast<const char*>(s_peer_data[r]) + slot_offset;
    }

    T* my_data = reinterpret_cast<T*>(
        const_cast<char*>(peer_pv[0]));

    // Step 1: scatter local input into our peer-visible data buffer.
    if (in != my_data) {
        const uint4* in_v4 = reinterpret_cast<const uint4*>(in);
        uint4*       out_v4 = reinterpret_cast<uint4*>(my_data);
        int count_v = count / VEC;
        for (int i = tid; i < count_v; i += bs) {
            out_v4[i] = in_v4[i];
        }
        int tail = count_v * VEC;
        for (int i = tail + tid; i < count; i += bs) {
            my_data[i] = in[i];
        }
    }

    // Step 2: publish data-ready, wait for peers.
    p2p_publish_and_wait(tid, my_rank, world_size, seq,
                         s_peer_flags, s_peer_flags[my_rank]);

    // Step 3: read all peers at current slot, sum, write to output.
    if constexpr (VEC == 8) {
        int count_v = count / 8;
        for (int i = tid; i < count_v; i += bs) {
            float2 a01 = {0,0}, a23 = {0,0}, a45 = {0,0}, a67 = {0,0};
            #pragma unroll
            for (int rr = 0; rr < P2P_AR_MAX_WORLD; ++rr) {
                if (rr >= world_size) break;
                const uint4* pv = reinterpret_cast<const uint4*>(peer_pv[rr]);
                uint4 raw = pv[i];
                auto* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
                float2 c0 = __bfloat1622float2(h2[0]);
                float2 c1 = __bfloat1622float2(h2[1]);
                float2 c2 = __bfloat1622float2(h2[2]);
                float2 c3 = __bfloat1622float2(h2[3]);
                a01.x += c0.x; a01.y += c0.y;
                a23.x += c1.x; a23.y += c1.y;
                a45.x += c2.x; a45.y += c2.y;
                a67.x += c3.x; a67.y += c3.y;
            }
            __nv_bfloat162 packed[4] = {
                __float22bfloat162_rn(a01), __float22bfloat162_rn(a23),
                __float22bfloat162_rn(a45), __float22bfloat162_rn(a67),
            };
            uint4 out_raw;
            __builtin_memcpy(&out_raw, packed, 16);
            reinterpret_cast<uint4*>(out)[i] = out_raw;
        }
        int tail = count_v * 8;
        for (int i = tail + tid; i < count; i += bs) {
            float s = 0.0f;
            #pragma unroll
            for (int rr = 0; rr < P2P_AR_MAX_WORLD; ++rr) {
                if (rr >= world_size) break;
                const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(peer_pv[rr]);
                s += __bfloat162float(p[i]);
            }
            out[i] = __float2bfloat16(s);
        }
    } else {
        int count_v = count / 4;
        for (int i = tid; i < count_v; i += bs) {
            float a0=0,a1=0,a2=0,a3=0;
            #pragma unroll
            for (int rr = 0; rr < P2P_AR_MAX_WORLD; ++rr) {
                if (rr >= world_size) break;
                const uint4* pv = reinterpret_cast<const uint4*>(peer_pv[rr]);
                uint4 raw = pv[i];
                auto* f = reinterpret_cast<const float*>(&raw);
                a0 += f[0]; a1 += f[1]; a2 += f[2]; a3 += f[3];
            }
            float packed[4] = {a0, a1, a2, a3};
            uint4 out_raw;
            __builtin_memcpy(&out_raw, packed, 16);
            reinterpret_cast<uint4*>(out)[i] = out_raw;
        }
        int tail = count_v * 4;
        for (int i = tail + tid; i < count; i += bs) {
            float s = 0.0f;
            #pragma unroll
            for (int rr = 0; rr < P2P_AR_MAX_WORLD; ++rr) {
                if (rr >= world_size) break;
                const float* p = reinterpret_cast<const float*>(peer_pv[rr]);
                s += p[i];
            }
            out[i] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// P2P AllReduce – multi-block smem-staged with progressive push.
//
// Uses p2p_publish_and_wait_any (push-based) to overlap NVLink data
// transfer with P2P spin-wait latency. Block 0 pushes its flag into all
// peers' local flag arrays and polls its own array for readiness, writing
// the cumulative ready mask to ready_mask_d; ALL blocks (including block 0)
// spin-read ready_mask_d and issue cg::memcpy_async for newly-ready peers.
//
// Peer data pointers passed as kernel args (p0..p7) for CUDA graph safety —
// no mutation of persistent instance state.
// ---------------------------------------------------------------------------

template <typename scalar_t, int ElemsPerWarp>
__global__ void __launch_bounds__(1024, 1)
p2p_allreduce_smem_kernel(
    const scalar_t* p0,  const scalar_t* p1,  const scalar_t* p2,  const scalar_t* p3,
    const scalar_t* p4,  const scalar_t* p5,  const scalar_t* p6,  const scalar_t* p7,
    int* const* peer_flags,
    unsigned long long* my_seq_counter,
    int* ready_mask_d,
    int my_rank, int world_size,
    scalar_t* __restrict__ out,
    int64_t numel,
    int smem_stride)
{
    constexpr int WarpSize = 32;
    constexpr int MaxN = P2P_AR_MAX_WORLD;
    int tid = threadIdx.x;
    int warp_id = tid / WarpSize;
    int lane    = tid % WarpSize;
    int warps_per_block = blockDim.x / WarpSize;
    int64_t my_warp_id = (int64_t)blockIdx.x * warps_per_block + warp_id;

    __shared__ int* s_peer_flags[MaxN];
    __shared__ int* s_my_flags;
    if (tid < world_size) s_peer_flags[tid] = peer_flags[tid];
    if (tid == 0) s_my_flags = peer_flags[my_rank];
    __syncthreads();
    int64_t total_warps = (int64_t)gridDim.x * warps_per_block;

    const scalar_t* ptrs[MaxN];
    ptrs[0] = p0; ptrs[1] = p1; ptrs[2] = p2; ptrs[3] = p3;
    ptrs[4] = p4; ptrs[5] = p5; ptrs[6] = p6; ptrs[7] = p7;

    auto warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());
    extern __shared__ char smem_raw[];
    char* warp_smem = smem_raw + warp_id * smem_stride;

    int64_t warp_start = my_warp_id * ElemsPerWarp;
    bool warp_active = (warp_start < numel);
    int64_t warp_elems = warp_active ? min((int64_t)ElemsPerWarp, numel - warp_start) : 0;
    size_t warp_copy_bytes = warp_elems * sizeof(scalar_t);

    // Copy self to smem immediately — local data, overlaps with P2P spin
    if (warp_active) {
        scalar_t* self_dst = reinterpret_cast<scalar_t*>(
            warp_smem + my_rank * ElemsPerWarp * sizeof(scalar_t));
        cg::memcpy_async(warp, self_dst, ptrs[my_rank] + warp_start, warp_copy_bytes);
    }

    int64_t total_warp_stride = total_warps * ElemsPerWarp;

    for (int64_t seg_start = warp_start; seg_start < numel; seg_start += total_warp_stride) {
        int64_t seg_elems = min((int64_t)ElemsPerWarp, numel - seg_start);
        size_t seg_copy_bytes = seg_elems * sizeof(scalar_t);

        if (seg_start == warp_start) {
            int prev_ready_mask = 0;
            while (true) {
                int ready_mask = p2p_publish_and_wait_any(
                    tid, my_rank, world_size,
                    my_seq_counter, prev_ready_mask, ready_mask_d,
                    s_peer_flags, s_my_flags);
                if (ready_mask == -1) break;

                int new_gpu_mask = ready_mask ^ prev_ready_mask;

                for (int gpu = 0; gpu < world_size; gpu++) {
                    if (!(new_gpu_mask & (1 << gpu))) continue;

                    if (warp_active) {
                        scalar_t* dst = reinterpret_cast<scalar_t*>(
                            warp_smem + gpu * ElemsPerWarp * sizeof(scalar_t));
                        cg::memcpy_async(warp, dst, ptrs[gpu] + seg_start, seg_copy_bytes);
                    }
                }

                prev_ready_mask = ready_mask;
            }
        } else {
            // All peers ready — memcpy_async for all peers at once
            for (int gpu = 0; gpu < world_size; gpu++) {
                if (warp_active) {
                    scalar_t* dst = reinterpret_cast<scalar_t*>(
                        warp_smem + gpu * ElemsPerWarp * sizeof(scalar_t));
                    cg::memcpy_async(warp, dst, ptrs[gpu] + seg_start, seg_copy_bytes);
                }
            }
        }

        if (warp_active) {
            cg::wait(warp);

            if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
                for (int i = lane; i < seg_elems; i += WarpSize) {
                    float acc = 0.0f;
                    for (int gpu = 0; gpu < world_size; gpu++) {
                        acc += __bfloat162float(
                            reinterpret_cast<const __nv_bfloat16*>(
                                warp_smem + gpu * ElemsPerWarp * sizeof(__nv_bfloat16))[i]);
                    }
                    out[seg_start + i] = __float2bfloat16(acc);
                }
            } else {
                for (int i = lane; i < seg_elems; i += WarpSize) {
                    float acc = 0.0f;
                    for (int gpu = 0; gpu < world_size; gpu++) {
                        acc += reinterpret_cast<const float*>(
                            warp_smem + gpu * ElemsPerWarp * sizeof(float))[i];
                    }
                    out[seg_start + i] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2 kernels: multi-block compute after P2P sync.
// Each reads slot_offset from device memory (written by p2p_data_sync_kernel)
// and peer_data pointers passed as kernel arguments.
// ---------------------------------------------------------------------------

template <typename T, int VEC>
__global__ void p2p_allreduce_multi_kernel(
    void* const* peer_data,
    const int* __restrict__ slot_offset_ptr,
    T* __restrict__ out,
    int count,
    int world_size,
    int my_rank)
{
    int slot_offset = *slot_offset_ptr;
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    // Rank-rotated peer data pointers in registers. rr is a compile-time
    // constant in the unrolled loop below, so peer_pv[rr] avoids register spill.
    const char* peer_pv[P2P_AR_MAX_WORLD];
    #pragma unroll
    for (int rr = 0; rr < P2P_AR_MAX_WORLD; ++rr) {
        int r = rr + my_rank; if (r >= world_size) r -= world_size;
        peer_pv[rr] = static_cast<const char*>(peer_data[r]) + slot_offset;
    }

    if constexpr (VEC == 8) {
        int count_v = count / 8;
        for (int i = blockIdx.x * bs + tid; i < count_v; i += gridDim.x * bs) {
            float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
            #pragma unroll
            for (int r = 0; r < P2P_AR_MAX_WORLD; ++r) {
                if (r >= world_size) break;
                const uint4* pv = reinterpret_cast<const uint4*>(peer_pv[r]);
                uint4 raw = pv[i];
                auto* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
                a0 += __bfloat162float(h[0]);
                a1 += __bfloat162float(h[1]);
                a2 += __bfloat162float(h[2]);
                a3 += __bfloat162float(h[3]);
                a4 += __bfloat162float(h[4]);
                a5 += __bfloat162float(h[5]);
                a6 += __bfloat162float(h[6]);
                a7 += __bfloat162float(h[7]);
            }
            __nv_bfloat16 packed[8] = {
                __float2bfloat16(a0), __float2bfloat16(a1),
                __float2bfloat16(a2), __float2bfloat16(a3),
                __float2bfloat16(a4), __float2bfloat16(a5),
                __float2bfloat16(a6), __float2bfloat16(a7),
            };
            uint4 out_raw;
            __builtin_memcpy(&out_raw, packed, 16);
            reinterpret_cast<uint4*>(out)[i] = out_raw;
        }
        int tail = count_v * 8;
        for (int i = tail + blockIdx.x * bs + tid; i < count; i += gridDim.x * bs) {
            float s = 0.0f;
            #pragma unroll
            for (int r = 0; r < P2P_AR_MAX_WORLD; ++r) {
                if (r >= world_size) break;
                const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(peer_pv[r]);
                s += __bfloat162float(p[i]);
            }
            out[i] = __float2bfloat16(s);
        }
    } else {
        int count_v = count / 4;
        for (int i = blockIdx.x * bs + tid; i < count_v; i += gridDim.x * bs) {
            float a0=0,a1=0,a2=0,a3=0;
            #pragma unroll
            for (int r = 0; r < P2P_AR_MAX_WORLD; ++r) {
                if (r >= world_size) break;
                const uint4* pv = reinterpret_cast<const uint4*>(peer_pv[r]);
                uint4 raw = pv[i];
                auto* f = reinterpret_cast<const float*>(&raw);
                a0 += f[0]; a1 += f[1]; a2 += f[2]; a3 += f[3];
            }
            float packed[4] = {a0, a1, a2, a3};
            uint4 out_raw;
            __builtin_memcpy(&out_raw, packed, 16);
            reinterpret_cast<uint4*>(out)[i] = out_raw;
        }
        int tail = count_v * 4;
        for (int i = tail + blockIdx.x * bs + tid; i < count; i += gridDim.x * bs) {
            float s = 0.0f;
            #pragma unroll
            for (int r = 0; r < P2P_AR_MAX_WORLD; ++r) {
                if (r >= world_size) break;
                const float* p = reinterpret_cast<const float*>(peer_pv[r]);
                s += p[i];
            }
            out[i] = s;
        }
    }
}

__global__ void __launch_bounds__(P2P_AR_BLOCK_SIZE, 4)
p2p_allgather_column_multi_kernel(
    void* const* peer_data,
    const int* __restrict__ slot_offset_ptr,
    void* __restrict__ out,
    int shard_bytes,
    int world_size,
    int my_rank)
{
    int slot_offset = *slot_offset_ptr;
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    char* out_b = static_cast<char*>(out);
    // Stagger the read order per-rank so all GPUs don't converge on the same
    // peer's memory at the same time (see p2p_allreduce_oneshot_kernel).
    for (int rr = 0; rr < world_size; ++rr) {
        int r = rr + my_rank; if (r >= world_size) r -= world_size;
        const char* src = static_cast<const char*>(peer_data[r]) + slot_offset;
        char* dst = out_b + (size_t)r * shard_bytes;

        bool aligned16 = (((size_t)src | (size_t)dst) & 15) == 0;
        if (aligned16 && shard_bytes >= 16) {
            const uint4* src_v4 = reinterpret_cast<const uint4*>(src);
            uint4*       dst_v4 = reinterpret_cast<uint4*>(dst);
            int num_vec = shard_bytes / 16;
            for (int i = blockIdx.x * bs + tid; i < num_vec; i += gridDim.x * bs) {
                dst_v4[i] = src_v4[i];
            }
            int tail_start = num_vec * 16;
            for (int i = tail_start + blockIdx.x * bs + tid; i < shard_bytes; i += gridDim.x * bs) {
                dst[i] = src[i];
            }
        } else {
            for (int i = blockIdx.x * bs + tid; i < shard_bytes; i += gridDim.x * bs) {
                dst[i] = src[i];
            }
        }
    }
}

__global__ void __launch_bounds__(P2P_AR_BLOCK_SIZE, 4)
p2p_allgather_row_multi_kernel(
    void* const* peer_data,
    const int* __restrict__ slot_offset_ptr,
    void* __restrict__ out,
    int shard_dim1_bytes,
    int full_dim1_bytes,
    int outer,
    int world_size,
    int my_rank)
{
    int slot_offset = *slot_offset_ptr;
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    char* out_b = static_cast<char*>(out);

    for (int row = blockIdx.x; row < outer; row += gridDim.x) {
        // Stagger the read order per-rank (see p2p_allreduce_oneshot_kernel).
        for (int rr = 0; rr < world_size; ++rr) {
            int r = rr + my_rank; if (r >= world_size) r -= world_size;
            const char* src = static_cast<const char*>(peer_data[r]) + slot_offset
                              + (size_t)row * shard_dim1_bytes;
            char* dst = out_b + (size_t)row * full_dim1_bytes + (size_t)r * shard_dim1_bytes;

            bool src_aligned16 = ((size_t)src & 15) == 0;
            bool dst_aligned16 = ((size_t)dst & 15) == 0;
            if (src_aligned16 && dst_aligned16 && shard_dim1_bytes >= 16) {
                const uint4* src_v4 = reinterpret_cast<const uint4*>(src);
                uint4*       dst_v4 = reinterpret_cast<uint4*>(dst);
                int num_vec = shard_dim1_bytes / 16;
                for (int i = tid; i < num_vec; i += bs) {
                    dst_v4[i] = src_v4[i];
                }
                int tail_start = num_vec * 16;
                for (int i = tail_start + tid; i < shard_dim1_bytes; i += bs) {
                    dst[i] = src[i];
                }
            } else if (shard_dim1_bytes >= 4 && (((size_t)src | (size_t)dst) & 3) == 0) {
                const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(src);
                uint32_t*       dst_u32 = reinterpret_cast<uint32_t*>(dst);
                int num_u32 = shard_dim1_bytes / 4;
                for (int i = tid; i < num_u32; i += bs) {
                    dst_u32[i] = src_u32[i];
                }
                int tail_start = num_u32 * 4;
                for (int i = tail_start + tid; i < shard_dim1_bytes; i += bs) {
                    dst[i] = src[i];
                }
            } else {
                for (int i = tid; i < shard_dim1_bytes; i += bs) {
                    dst[i] = src[i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// P2P AllGather – Column layout (contiguous per rank).
//
// Two-phase: p2p_data_sync_kernel (scatter + sync) + multi-block gather.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// P2P AllGather – Row layout (interleaved).
//
// Two-phase: p2p_data_sync_kernel (scatter + sync) + multi-block gather.
// The multi-block kernel uses one block per row, parallelizing across rows.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// P2P Row-parallel RMSNorm
//
// Computes RMSNorm on a row-parallel tensor (each rank holds shard_dim columns
// of the full hidden dimension). Instead of allGathering the full tensor, this
// kernel only communicates a scalar (sum of squares) per row across ranks:
//
//   1. Each rank computes local sum(x_shard^2) per row
//   2. P2P sync: each rank writes its partial sums to its P2P buffer,
//      waits for all peers, then reads all peers' partial sums and sums them
//   3. Each rank computes inv_rms = rsqrt(total_sum / full_dim + eps)
//   4. Each rank normalizes its local shard: out = weight * x * inv_rms
//
// Communication cost: batch * sizeof(float) per rank (vs batch * full_dim * sizeof(bf16)
// for allGather). Output remains row-parallel (shard_dim columns per rank).
//
// Layout:
//   input:  [batch, shard_dim] BF16  (row-parallel shard)
//   weight: [shard_dim] BF16        (row-parallel shard of the weight vector)
//   output: [batch, shard_dim] BF16  (row-parallel shard)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(P2P_AR_BLOCK_SIZE, 1)
p2p_rmsnorm_kernel(
    void* const* peer_data,
    int* const* peer_flags,
    unsigned long long* my_seq_counter,
    int my_rank,
    int world_size,
    int max_slot_bytes,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float eps,
    int shard_dim,
    int full_dim,
    int batch)
{
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    __shared__ unsigned int s_seq;
    __shared__ int          s_slot_offset;
    __shared__ void*        s_peer_data[P2P_AR_MAX_WORLD];
    __shared__ int*         s_peer_flags[P2P_AR_MAX_WORLD];
    __shared__ float        s_inv_rms[2048];

    if (tid == 0) {
        unsigned long long s = atomicAdd(my_seq_counter, 2ULL) + 2ULL;
        s_seq = (unsigned int)(s & 0x7FFFFFFEu);
        if (s_seq == 0) s_seq = 2;
        s_slot_offset = ((s_seq >> 1) & 1) * max_slot_bytes;
    }
    if (tid < world_size) {
        s_peer_data[tid]  = peer_data[tid];
        s_peer_flags[tid] = peer_flags[tid];
    }
    __syncthreads();

    int seq = (int)s_seq;
    int slot_offset = s_slot_offset;

    // Rank-rotated peer data pointers in registers. rr is a compile-time
    // constant in the unrolled loop below, so peer_pv[rr] avoids register spill.
    // Contrast with s_peer_flags[tid] where tid is dynamic — that must stay in smem.
    const char* peer_pv[P2P_AR_MAX_WORLD];
    #pragma unroll
    for (int rr = 0; rr < P2P_AR_MAX_WORLD; ++rr) {
        int r = rr + my_rank; if (r >= world_size) r -= world_size;
        peer_pv[rr] = static_cast<const char*>(s_peer_data[r]) + slot_offset;
    }

    float* my_data = reinterpret_cast<float*>(
        const_cast<char*>(peer_pv[0]));

    // ---- Step 1: Compute local sum of squares per row, write to P2P buffer.
    for (int row = tid; row < batch; row += bs) {
        float local_sum = 0.0f;
        const __nv_bfloat16* x = input + (size_t)row * shard_dim;
        for (int i = 0; i < shard_dim; ++i) {
            float f = __bfloat162float(x[i]);
            local_sum += f * f;
        }
        my_data[row] = local_sum;
    }

    // ---- Step 2: Publish data-ready, wait for peers.
    p2p_publish_and_wait(tid, my_rank, world_size, seq,
                         s_peer_flags, s_peer_flags[my_rank]);

    // ---- Step 3: Read all peers' partial sums, compute inv_rms per row.
    for (int row = tid; row < batch; row += bs) {
        float total_sum = 0.0f;
        #pragma unroll
        for (int rr = 0; rr < P2P_AR_MAX_WORLD; ++rr) {
            if (rr >= world_size) break;
            const float* peer_buf = reinterpret_cast<const float*>(peer_pv[rr]);
            total_sum += peer_buf[row];
        }
        s_inv_rms[row] = rsqrtf(total_sum / (float)full_dim + eps);
    }
    __syncthreads();

    // ---- Step 4: Normalize and apply weight.
    for (int row = 0; row < batch; ++row) {
        const __nv_bfloat16* x = input + (size_t)row * shard_dim;
        __nv_bfloat16* o = output + (size_t)row * shard_dim;
        float inv_rms = s_inv_rms[row];

        for (int i = tid; i < shard_dim; i += bs) {
            float f = __bfloat162float(x[i]);
            float w = __bfloat162float(weight[i]);
            o[i] = __float2bfloat16(w * f * inv_rms);
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int glm_p2p_enable_peer_access(GlmCtx* ctx, int peer_device) {
    cudaSetDevice(ctx->device_id);
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, ctx->device_id, peer_device);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_p2p_enable_peer_access: cudaDeviceCanAccessPeer(%d, %d) failed: %s\n",
                ctx->device_id, peer_device, cudaGetErrorString(err));
        return -1;
    }
    if (!can_access) return -1;
    err = cudaDeviceEnablePeerAccess(peer_device, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return 0;
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_p2p_enable_peer_access: cudaDeviceEnablePeerAccess(%d) on dev %d failed: %s\n",
                peer_device, ctx->device_id, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

GlmP2PInstance* glm_p2p_create_instance(GlmCtx* ctx, int my_rank, int world_size) {
    if (world_size > P2P_AR_MAX_WORLD || world_size <= 0) {
        fprintf(stderr, "glm_p2p_create_instance: invalid world_size %d\n", world_size);
        return nullptr;
    }
    cudaSetDevice(ctx->device_id);
    auto* inst = new GlmP2PInstance();
    inst->world_size = world_size;
    inst->my_rank = my_rank;
    inst->max_bytes = 0;
    inst->device_id = ctx->device_id;

    // Metadata-only allocation: peer_data[N] | peer_flags[N] | seq_counter | flags[N] | slot_offset | ready_mask
    // Data buffer is provided externally via p2pSetPeers.
    size_t header = sizeof(void*) * world_size
                   + sizeof(int*)  * world_size
                   + sizeof(unsigned long long)
                   + sizeof(int)  * world_size
                   + sizeof(int)
                   + sizeof(int);
    size_t header_aligned = (header + 255) & ~size_t(255);  // 256B align

    void* base = nullptr;
    cudaError_t err = cudaMalloc(&base, header_aligned);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_p2p_create_instance: cudaMalloc(%zu) failed: %s\n", header_aligned, cudaGetErrorString(err));
        delete inst;
        return nullptr;
    }
    cudaMemset(base, 0, header_aligned);  // zero so peer_data ptrs are null, seq=0, flag=0

    char* p = static_cast<char*>(base);
    inst->metadata_alloc_d = base;
    inst->peer_data_arr_d  = reinterpret_cast<void**>(p); p += sizeof(void*) * world_size;
    inst->peer_flags_arr_d = reinterpret_cast<int**>(p);  p += sizeof(int*)  * world_size;
    inst->seq_counter_d    = reinterpret_cast<unsigned long long*>(p); p += sizeof(unsigned long long);
    inst->my_flags_d       = reinterpret_cast<int*>(p); p += sizeof(int) * world_size;
    inst->slot_offset_d    = reinterpret_cast<int*>(p); p += sizeof(int);
    inst->ready_mask_d     = reinterpret_cast<int*>(p);
    return inst;
}

void glm_p2p_destroy_instance(GlmP2PInstance* inst) {
    if (!inst) return;
    cudaSetDevice(inst->device_id);
    cudaFree(inst->metadata_alloc_d);
    delete inst;
}

int* glm_p2p_get_flag_ptr(GlmP2PInstance* inst) {
    return inst ? inst->my_flags_d : nullptr;
}

void glm_p2p_set_max_bytes(GlmP2PInstance* inst, size_t max_bytes) {
    if (inst) inst->max_bytes = max_bytes;
}

void glm_p2p_set_peers(GlmCtx* ctx, GlmP2PInstance* inst,
                       const void* const* peer_data_ptrs,
                       int* const* peer_flag_ptrs) {
    cudaSetDevice(ctx->device_id);
    int N = inst->world_size;
    cudaMemcpy(inst->peer_data_arr_d, peer_data_ptrs, sizeof(void*) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(inst->peer_flags_arr_d, peer_flag_ptrs, sizeof(int*) * N, cudaMemcpyHostToDevice);
}

void glm_p2p_allreduce(GlmCtx* ctx, GlmP2PInstance* inst,
                       const void* in, void* out, int count, int dtype) {
    cudaSetDevice(ctx->device_id);

    if (dtype == 9) {
        if ((size_t)count * 2 > inst->max_bytes) {
            fprintf(stderr, "glm_p2p_allreduce: count %d * 2 > max_bytes %zu\n", count, inst->max_bytes);
            return;
        }
        p2p_allreduce_oneshot_kernel<__nv_bfloat16, P2P_AR_VEC_BF16>
            <<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(in),
                static_cast<__nv_bfloat16*>(out),
                count);
    } else if (dtype == 7) {
        if ((size_t)count * 4 > inst->max_bytes) {
            fprintf(stderr, "glm_p2p_allreduce: count %d * 4 > max_bytes %zu\n", count, inst->max_bytes);
            return;
        }
        p2p_allreduce_oneshot_kernel<float, P2P_AR_VEC_F32>
            <<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const float*>(in),
                static_cast<float*>(out),
                count);
    } else {
        fprintf(stderr, "glm_p2p_allreduce: unsupported dtype %d\n", dtype);
    }
}

void glm_p2p_allreduce_smem(GlmCtx* ctx, GlmP2PInstance* inst,
                             const void* p0,  const void* p1,
                             const void* p2,  const void* p3,
                             const void* p4,  const void* p5,
                             const void* p6,  const void* p7,
                             void* output, int N, int64_t numel, int dtype) {
    cudaSetDevice(ctx->device_id);

    // Zero ready_mask_d before launch
    cudaMemsetAsync(inst->ready_mask_d, 0, sizeof(int), GLM_STREAM(ctx));

    constexpr int WarpSize = 32;
    constexpr int ElemsPerWarp = 512;
    constexpr int SmemBudget = 32 * 1024;

    int elem_size = (dtype == 9) ? 2 : 4;
    int smem_per_warp = N * ElemsPerWarp * elem_size;
    if (smem_per_warp < 1) smem_per_warp = 1;

    int warps_per_block = SmemBudget / smem_per_warp;
    if (warps_per_block > 32) warps_per_block = 32;
    if (warps_per_block < 1) warps_per_block = 1;

    int block_size = warps_per_block * WarpSize;
    int64_t total_warps = (numel + ElemsPerWarp - 1) / ElemsPerWarp;
    if (total_warps < 1) total_warps = 1;
    int grid = (int)((total_warps + warps_per_block - 1) / warps_per_block);
    if (grid > 65535) grid = 65535;
    int smem_bytes = warps_per_block * smem_per_warp;

    if (dtype == 9) {
        p2p_allreduce_smem_kernel<__nv_bfloat16, ElemsPerWarp>
            <<<grid, block_size, smem_bytes, GLM_STREAM(ctx)>>>(
                static_cast<const __nv_bfloat16*>(p0),
                static_cast<const __nv_bfloat16*>(p1),
                static_cast<const __nv_bfloat16*>(p2),
                static_cast<const __nv_bfloat16*>(p3),
                static_cast<const __nv_bfloat16*>(p4),
                static_cast<const __nv_bfloat16*>(p5),
                static_cast<const __nv_bfloat16*>(p6),
                static_cast<const __nv_bfloat16*>(p7),
                inst->peer_flags_arr_d, inst->seq_counter_d, inst->ready_mask_d,
                inst->my_rank, N,
                static_cast<__nv_bfloat16*>(output),
                numel, smem_per_warp);
    } else if (dtype == 7) {
        p2p_allreduce_smem_kernel<float, ElemsPerWarp>
            <<<grid, block_size, smem_bytes, GLM_STREAM(ctx)>>>(
                static_cast<const float*>(p0),
                static_cast<const float*>(p1),
                static_cast<const float*>(p2),
                static_cast<const float*>(p3),
                static_cast<const float*>(p4),
                static_cast<const float*>(p5),
                static_cast<const float*>(p6),
                static_cast<const float*>(p7),
                inst->peer_flags_arr_d, inst->seq_counter_d, inst->ready_mask_d,
                inst->my_rank, N,
                static_cast<float*>(output),
                numel, smem_per_warp);
    } else {
        fprintf(stderr, "glm_p2p_allreduce_smem: unsupported dtype %d\n", dtype);
    }
}

void glm_p2p_allgather(GlmCtx* ctx, GlmP2PInstance* inst,
                        const void* sendbuf, void* recvbuf,
                        int num_bytes) {
    cudaSetDevice(ctx->device_id);

    if ((size_t)num_bytes > inst->max_bytes) {
        fprintf(stderr, "glm_p2p_allgather: num_bytes %d > max_bytes %zu\n", num_bytes, inst->max_bytes);
        return;
    }

    // Phase 1: P2P sync
    p2p_data_sync_kernel<<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
        inst->slot_offset_d,
        inst->my_rank, inst->world_size, (int)inst->max_bytes,
        sendbuf, num_bytes);

    // Phase 2: Multi-block column gather
    int grid = (num_bytes / 16 + P2P_AR_BLOCK_SIZE - 1) / P2P_AR_BLOCK_SIZE;
    if (grid < 1) grid = 1;
    p2p_allgather_column_multi_kernel<<<grid, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_data_arr_d, inst->slot_offset_d,
        recvbuf, num_bytes, inst->world_size, inst->my_rank);
}

void glm_p2p_allgather_row(GlmCtx* ctx, GlmP2PInstance* inst,
                             const void* sendbuf, void* recvbuf,
                             int shard_bytes, int shard_dim1_bytes,
                             int full_dim1_bytes, int outer) {
    cudaSetDevice(ctx->device_id);

    if ((size_t)shard_bytes > inst->max_bytes) {
        fprintf(stderr, "glm_p2p_allgather_row: shard_bytes %d > max_bytes %zu\n", shard_bytes, inst->max_bytes);
        return;
    }

    // Phase 1: P2P sync
    p2p_data_sync_kernel<<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
        inst->slot_offset_d,
        inst->my_rank, inst->world_size, (int)inst->max_bytes,
        sendbuf, shard_bytes);

    // Phase 2: Multi-block row gather (one block per row)
    int grid = outer;
    if (grid < 1) grid = 1;
    p2p_allgather_row_multi_kernel<<<grid, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_data_arr_d, inst->slot_offset_d,
        recvbuf, shard_dim1_bytes, full_dim1_bytes, outer, inst->world_size, inst->my_rank);
}

void glm_p2p_rmsnorm(GlmCtx* ctx, GlmP2PInstance* inst,
                      const void* input, const void* weight, void* output,
                      float eps, int shard_dim, int full_dim, int batch) {
    cudaSetDevice(ctx->device_id);

    size_t required_bytes = (size_t)batch * sizeof(float);
    if (required_bytes > inst->max_bytes) {
        fprintf(stderr, "glm_p2p_rmsnorm: batch %d requires %zu bytes > max_bytes %zu\n",
                batch, required_bytes, inst->max_bytes);
        return;
    }

    p2p_rmsnorm_kernel<<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
        inst->my_rank, inst->world_size, (int)inst->max_bytes,
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output),
        eps, shard_dim, full_dim, batch);
}

void glm_p2p_barrier(GlmCtx* ctx, GlmP2PInstance* inst) {
    cudaSetDevice(ctx->device_id);
    p2p_barrier_kernel<<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_flags_arr_d, inst->seq_counter_d,
        inst->slot_offset_d,
        inst->my_rank, inst->world_size, (int)inst->max_bytes);
}

} // extern "C"
