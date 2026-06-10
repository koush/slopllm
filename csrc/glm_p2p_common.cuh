#ifndef GLM_P2P_COMMON_CUH
#define GLM_P2P_COMMON_CUH

#include "glm_ops.h"

__device__ __forceinline__ void p2p_publish_and_wait(
    int tid,
    int my_rank,
    int world_size,
    int seq,
    int* const* peer_flag_arrays,
    int* my_flags)
{
    __syncthreads();

    if (tid < world_size) {
        int val = seq + 1;
        if (tid == my_rank) {
            my_flags[my_rank] = val;
        } else {
            asm volatile("st.global.release.sys.s32 [%0], %1;"
                         :: "l"(peer_flag_arrays[tid] + my_rank), "r"(val));
        }
    }

    if (tid < world_size) {
        int target = seq + 1;
        int v;
        do {
            asm volatile("ld.volatile.global.s32 %0, [%1];"
                         : "=r"(v) : "l"(my_flags + tid));
            if (v < target) __nanosleep(32);
        } while (v < target);
        asm volatile("fence.acquire.sys;");
    }
    __syncthreads();
}

// Progressive P2P sync callable from any block.
//
// Block 0: pushes own flag into all peers' flag arrays (push), then polls
// own local my_flags array until at least one new peer is ready, computes
// cumulative mask, writes it to *ready_mask_d.
//
// Other blocks: spin-read *ready_mask_d until it advances past
// prev_ready_mask.
//
// Returns cumulative ready_mask (bit N set = peer N ready, self excluded).
// Returns full_mask when all peers are ready.
// Returns -1 if called again after all peers were already reported.
//
// Call in a loop:  int mask = 0; while ((mask = fn(...,mask)) != -1) { ... }
//
// ready_mask_d – device int, zeroed before kernel launch
__device__ __forceinline__ int p2p_publish_and_wait_any(
    int tid,
    int my_rank,
    int world_size,
    unsigned long long* my_seq_counter,
    int prev_ready_mask,
    int* ready_mask_d,
    int* const* peer_flag_arrays,
    int* my_flags)
{
    constexpr int MaxN = P2P_AR_MAX_WORLD;
    int full_mask = ((1 << world_size) - 1) ^ (1 << my_rank);

    if (prev_ready_mask == full_mask) return -1;

    __shared__ unsigned int s_seq;
    __shared__ int*         s_peer_flags[MaxN];
    __shared__ volatile int smem_flag_ready;
    __shared__ int          s_ready_mask;
    __shared__ int          s_observed;

    if (blockIdx.x == 0) {
        if (prev_ready_mask == 0) {
            if (tid == 0) {
                unsigned long long s = atomicAdd(my_seq_counter, 2ULL) + 2ULL;
                s_seq = (unsigned int)(s & 0x7FFFFFFEu);
                if (s_seq == 0) s_seq = 2;
            }
            if (tid < world_size)
                s_peer_flags[tid] = peer_flag_arrays[tid];
            __syncthreads();

            // Push: write our flag into every peer's local flag array
            if (tid < world_size) {
                int val = (int)s_seq + 1;
                if (tid == my_rank) {
                    my_flags[my_rank] = val;
                } else {
                    asm volatile("st.global.release.sys.s32 [%0], %1;"
                                 :: "l"(s_peer_flags[tid] + my_rank), "r"(val));
                }
            }
        }

        if (tid == 0) smem_flag_ready = 0;
        __syncthreads();

        // Poll own local my_flags for peer readiness
        while (true) {
            if (tid < world_size && tid != my_rank
                && !(prev_ready_mask & (1 << tid))) {
                int v;
                asm volatile("ld.volatile.global.s32 %0, [%1];"
                             : "=r"(v) : "l"(my_flags + tid));
                if (v >= (int)s_seq + 1) {
                    smem_flag_ready = 1;
                    break;
                }
                __nanosleep(32);
            }
            if (smem_flag_ready) break;
        }

        if (tid < 32) {
            __syncwarp();
            asm volatile("fence.acquire.sys;");
            uint32_t my_bit = 0;
            if (tid < world_size && tid != my_rank) {
                int v;
                asm volatile("ld.volatile.global.s32 %0, [%1];"
                             : "=r"(v) : "l"(my_flags + tid));
                if (v >= (int)s_seq + 1) my_bit = (1u << tid);
            }
            uint32_t ballot = __ballot_sync(0xFFFFFFFFu, my_bit != 0);
            if (tid == 0) s_ready_mask = (int)ballot;
        }
        __syncthreads();

        int cumulative = prev_ready_mask | s_ready_mask;

        if (tid == 0) {
            *(volatile int*)ready_mask_d = cumulative;
            __threadfence();
        }
        __syncthreads();

        return cumulative;
    } else {
        if (tid == 0) {
            int cur = prev_ready_mask;
            while (cur == prev_ready_mask) {
                cur = *(volatile int*)ready_mask_d;
            }
            s_observed = cur;
        }
        __syncthreads();

        return s_observed;
    }
}

#endif
