#ifndef GLM_P2P_COMMON_CUH
#define GLM_P2P_COMMON_CUH

#include "glm_ops.h"

__device__ __forceinline__ void p2p_spin_until(volatile int* flag, int target) {
    while (*flag < target) { /* spin */ }
}

__device__ __forceinline__ void p2p_publish_and_wait(
    int tid,
    int* const* s_peer_flags,
    int my_rank,
    int world_size,
    int seq)
{
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        volatile int* mf = s_peer_flags[my_rank];
        *mf = seq + 1;
    }
    if (tid < world_size) {
        volatile int* pf = s_peer_flags[tid];
        p2p_spin_until(pf, seq + 1);
    }
    __syncthreads();
    __threadfence_system();
}

// Progressive P2P sync callable from any block.
//
// Block 0: publishes own flag (once, on first call), spins on P2P flags
// until at least one new peer is ready, computes cumulative mask, writes
// it to *ready_mask_d.
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
    int* const* peer_flags,
    int my_rank,
    int world_size,
    unsigned long long* my_seq_counter,
    int prev_ready_mask,
    int* ready_mask_d)
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
                s_peer_flags[tid] = peer_flags[tid];
            __syncthreads();
            if (tid == 0) {
                __threadfence_system();
                *(volatile int*)s_peer_flags[my_rank] = s_seq + 1;
            }
        }

        if (tid == 0) smem_flag_ready = 0;
        __syncthreads();

        while (true) {
            if (tid < world_size && tid != my_rank
                && !(prev_ready_mask & (1 << tid))) {
                volatile int* pf = s_peer_flags[tid];
                if (*pf >= s_seq + 1) {
                    smem_flag_ready = 1;
                    break;
                }
            }
            if (smem_flag_ready) break;
        }

        if (tid < 32) {
            __syncwarp();
            uint32_t my_bit = 0;
            if (tid < world_size && tid != my_rank) {
                volatile int* pf = s_peer_flags[tid];
                if (*pf >= s_seq + 1) my_bit = (1u << tid);
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
        __threadfence_system();

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
