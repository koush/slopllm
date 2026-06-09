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
        // s_peer_flags via smem: my_rank is dynamic across kernels, smem avoids spill
        volatile int* mf = s_peer_flags[my_rank];
        *mf = seq + 1;
    }
    // s_peer_flags[tid] via smem: dynamic index would spill register array
    if (tid < world_size) {
        volatile int* pf = s_peer_flags[tid];
        p2p_spin_until(pf, seq + 1);
    }
    __syncthreads();
    __threadfence_system();
}

__device__ __forceinline__ int p2p_publish_and_wait_any(
    int tid,
    int* const* s_peer_flags,
    int my_rank,
    int world_size,
    int seq,
    int prev_ready_mask)
{
    int full_mask = ((1 << world_size) - 1) ^ (1 << my_rank);

    if (prev_ready_mask == full_mask) return -1;

    __shared__ volatile int smem_flag_ready;
    if (tid == 0) smem_flag_ready = 0;

    if (prev_ready_mask == 0) {
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            volatile int* mf = s_peer_flags[my_rank];
            *mf = seq + 1;
        }
    }
    __syncthreads();

    while (true) {
        if (tid < world_size && tid != my_rank
            && !(prev_ready_mask & (1 << tid))) {
            volatile int* pf = s_peer_flags[tid];
            if (*pf >= seq + 1) {
                smem_flag_ready = 1;
                break;
            }
        }
        if (smem_flag_ready) break;
    }

    __shared__ int s_ready_mask;
    if (tid < 32) {
        __syncwarp();
        uint32_t my_bit = 0;
        if (tid < world_size && tid != my_rank) {
            volatile int* pf = s_peer_flags[tid];
            if (*pf >= seq + 1) my_bit = (1u << tid);
        }
        uint32_t ballot = __ballot_sync(0xFFFFFFFFu, my_bit != 0);
        if (tid == 0) s_ready_mask = (int)ballot;
    }
    __syncthreads();
    __threadfence_system();

    return s_ready_mask;
}

#endif
