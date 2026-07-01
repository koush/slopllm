#ifndef GLM_P2P_COMMON_CUH
#define GLM_P2P_COMMON_CUH

#include "glm_ops.h"

__device__ __forceinline__ void p2p_publish_and_wait(
    int tid,
    int my_rank,
    int world_size,
    int seq,
    int* const* peer_flag_arrays,
    int* my_flags,
    int peer_rank = -1)
{
    __syncthreads();

    bool active = (peer_rank < 0 && tid < world_size) ||
                  (peer_rank >= 0 && tid == peer_rank);

    if (active) {
        int val = seq + 1;
        if (tid == my_rank) {
            my_flags[my_rank] = val;
        } else {
            asm volatile("st.global.release.sys.s32 [%0], %1;"
                         :: "l"(peer_flag_arrays[tid] + my_rank), "r"(val));
        }
    }

    if (active) {
        int target = seq + 1;
        int v;
        do {
            asm volatile("ld.volatile.global.s32 %0, [%1];"
                         : "=r"(v) : "l"(my_flags + tid));
            if ((int)((unsigned)v - (unsigned)target) < 0) __nanosleep(32);
        } while ((int)((unsigned)v - (unsigned)target) < 0);
        asm volatile("fence.acquire.sys;");
    }
    __syncthreads();
}

#endif
