#pragma once

#include <cuda_bf16.h>
#include "flashinfer/attention/state.cuh"
#include "flashinfer/vec_dtypes.cuh"

template <int VEC_SIZE, int BDX, int NUM_SHARDS, typename LseLoader, typename VLoader>
__device__ __forceinline__ void cp_merge_one_pair(
    int tid,
    int b, int h, int num_heads,
    float* s_lse,
    LseLoader load_lse,
    VLoader load_v,
    __nv_bfloat16* merged_v_out,
    float* merged_lse)
{
    constexpr int head_dim = VEC_SIZE * BDX;

    load_lse(s_lse, b, h, num_heads);
    __syncthreads();

    if (tid < BDX) {
        flashinfer::state_t<VEC_SIZE> st;
        st.init();

        #pragma unroll
        for (int s = 0; s < NUM_SHARDS; ++s) {
            flashinfer::vec_t<float, VEC_SIZE> v;
            load_v(v, s, b, h, num_heads, head_dim, tid);
            st.merge(v, s_lse[s], 1.0f);
        }

        st.normalize();
        st.o.cast_store(merged_v_out + (b * num_heads + h) * head_dim + tid * VEC_SIZE);

        if (merged_lse != nullptr && tid == 0) {
            merged_lse[b * num_heads + h] = st.get_lse();
        }
    }
}
