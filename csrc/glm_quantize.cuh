#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace glm_quantize {
// cuBLASLt VEC32_UE8M0: 128 rows x four K32 scales per tile.
__device__ inline size_t mx_scale_offset(int row, int group, int k) {
    return size_t(row / 128) * (k / 128) * 512 + (group / 4) * 512
        + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
}

// Eight values per lane in the vectorized paths. MX uses four-lane K32
// subgroups and tiled E8M0 scales; ordinary quantization uses K128/FP32.
template <bool VECTORIZED, bool MX = false>
__global__ void quantize_fp8_kernel(const __nv_bfloat16* input, uint8_t* values,
    void* scales, size_t num_blocks, uint32_t block_size,
    int rows = 0, int k = 0, int stride = 0) {
    static_assert(!MX || VECTORIZED);
    constexpr int SUBGROUP = MX ? 4 : 16;
    const size_t block = (size_t)blockIdx.x * (256 / SUBGROUP) + threadIdx.x / SUBGROUP;
    const int lane = threadIdx.x % SUBGROUP;
    if (block >= num_blocks) return;
    size_t offset = block * (VECTORIZED ? SUBGROUP * 8 : block_size);
    size_t inputOffset = offset;
    bool valid = true;
    int row = 0, group = 0, padded_k = 0;
    if constexpr (MX) {
        padded_k = (k + 127) / 128 * 128;
        row = block / (padded_k / 32);
        group = block % (padded_k / 32);
        valid = row < rows && group * 32 < k;
        inputOffset = size_t(row) * stride + group * 32;
    }
    const unsigned mask = __activemask();
    int4 loaded;
    float amax = 0.f;
    if constexpr (VECTORIZED) {
        loaded = valid ? *reinterpret_cast<const int4*>(input + inputOffset + lane * 8)
            : make_int4(0, 0, 0, 0);
        const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&loaded);
        __nv_bfloat162 m = __float2bfloat162_rn(0.f);
        #pragma unroll
        for (int j = 0; j < 4; ++j) m = __hmax2(m, __habs2(pairs[j]));
        #pragma unroll
        for (int delta = SUBGROUP / 2; delta > 0; delta /= 2)
            m = __hmax2(m, __shfl_xor_sync(mask, m, delta, SUBGROUP));
        amax = __bfloat162float(__hmax(m.x, m.y));
    } else {
        for (uint32_t d = lane; d < block_size; d += SUBGROUP)
            amax = fmaxf(amax, fabsf(__bfloat162float(input[offset + d])));
        for (int delta = SUBGROUP / 2; delta > 0; delta /= 2)
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, delta, SUBGROUP));
    }
    uint32_t bits;
    if constexpr (MX) {
        // 448 = 1.75 * 2^8. Compare the mantissa directly to avoid
        // division rounding a scale boundary just above an exact power of two.
        const uint32_t a = __float_as_uint(fmaxf(amax, 1e-4f));
        const int exponent = int(a >> 23) - 127 - 8 + ((a & 0x7fffff) > 0x600000);
        bits = uint32_t(max(-127, min(127, exponent)) + 127) << 23;
    } else {
        const float raw = fmaxf(amax, 1e-4f) / 448.f;
        bits = __float_as_uint(raw);
        if (bits & 0x007FFFFF) bits = (bits + 0x00800000) & 0x7F800000;
    }
    const float scale = __uint_as_float(bits);
    if (lane == 0) {
        if constexpr (MX)
            static_cast<uint8_t*>(scales)[mx_scale_offset(row, group, padded_k)] = valid ? bits >> 23 : 0;
        else static_cast<float*>(scales)[block] = scale;
    }
    const float inv_scale = 1.f / scale;
    if constexpr (VECTORIZED) {
        const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&loaded);
        uint32_t packed[2] = {};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 v = __bfloat1622float2(pairs[j]);
            v.x = fmaxf(-448.f, fminf(448.f, v.x * inv_scale));
            v.y = fmaxf(-448.f, fminf(448.f, v.y * inv_scale));
            const uint16_t p = __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
            packed[j >> 1] |= (uint32_t)p << ((j & 1) * 16);
        }
        if (!MX || row < rows)
            *reinterpret_cast<uint2*>(values + offset + lane * 8) = make_uint2(packed[0], packed[1]);
    } else {
        for (uint32_t d = lane; d < block_size; d += SUBGROUP) {
            const float v = fmaxf(-448.f, fminf(448.f, __bfloat162float(input[offset + d]) * inv_scale));
            values[offset + d] = __nv_fp8_e4m3(v).__x;
        }
    }
}
} // namespace glm_quantize
