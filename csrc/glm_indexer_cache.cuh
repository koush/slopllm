#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

__device__ __forceinline__ float indexer_ue8m0_scale(float amax) {
    float raw = fmaxf(amax, 1e-4f) / 448.f;
    uint32_t bits = __float_as_uint(raw);
    if (bits & 0x007fffffu)
        bits = (bits + 0x00800000u) & 0x7f800000u;
    return __uint_as_float(bits);
}

template <int HEAD_DIM>
__device__ __forceinline__ void pack_indexer_k_row(
    uint8_t* dst, float* dst_scale,
    const __nv_bfloat16* src, float* scratch) {
    static_assert(HEAD_DIM % 2 == 0);
    const int pair = threadIdx.x;
    float local_max = 0.f;
    float2 values = make_float2(0.f, 0.f);
    if (pair < HEAD_DIM / 2) {
        values = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(src)[pair]);
        local_max = fmaxf(fabsf(values.x), fabsf(values.y));
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = local_max;
    __syncthreads();

    if (threadIdx.x == 0) {
        float amax = 0.f;
        for (int warp = 0; warp < (blockDim.x + 31) / 32; warp++)
            amax = fmaxf(amax, scratch[warp]);
        scratch[0] = indexer_ue8m0_scale(amax);
        *dst_scale = scratch[0];
    }
    __syncthreads();

    if (pair < HEAD_DIM / 2) {
        values.x /= scratch[0];
        values.y /= scratch[0];
        reinterpret_cast<uint16_t*>(dst)[pair] =
            __nv_cvt_float2_to_fp8x2(values, __NV_SATFINITE, __NV_E4M3);
    }
}
