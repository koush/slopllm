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

#endif
