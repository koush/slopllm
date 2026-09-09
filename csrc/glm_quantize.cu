#include "glm_ops.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// One 16-thread subgroup per last-dimension block. Match sparse MLA's
// power-of-two scale rounding and E4M3 conversion exactly.
template <bool VECTORIZED>
__global__ void quantize_fp8_kernel(const __nv_bfloat16* input, uint8_t* values,
                                    float* scales, size_t num_blocks, uint32_t block_size) {
    const size_t block = (size_t)blockIdx.x * 16 + threadIdx.x / 16;
    const int lane = threadIdx.x % 16;
    if (block >= num_blocks) return;
    const size_t offset = block * (VECTORIZED ? 128 : block_size);
    const unsigned mask = __activemask();
    int4 loaded;
    float amax = 0.f;
    if constexpr (VECTORIZED) {
        // Eight consecutive BF16 values per lane: one coalesced 16-byte load,
        // retained in registers for the quantization pass.
        loaded = *reinterpret_cast<const int4*>(input + offset + lane * 8);
        const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&loaded);
        __nv_bfloat162 m = __float2bfloat162_rn(0.f);
        #pragma unroll
        for (int j = 0; j < 4; ++j) m = __hmax2(m, __habs2(pairs[j]));
        #pragma unroll
        for (int delta = 8; delta > 0; delta /= 2) {
            m = __hmax2(m, __shfl_xor_sync(mask, m, delta, 16));
        }
        amax = __bfloat162float(__hmax(m.x, m.y));
    } else {
        for (uint32_t d = lane; d < block_size; d += 16) {
            amax = fmaxf(amax, fabsf(__bfloat162float(input[offset + d])));
        }
        for (int delta = 8; delta > 0; delta /= 2) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, delta, 16));
        }
    }
    float raw = fmaxf(amax, 1e-4f) / 448.f;
    uint32_t bits = __float_as_uint(raw);
    if (bits & 0x007FFFFF) bits = (bits + 0x00800000) & 0x7F800000;
    const float scale = __uint_as_float(bits);
    if (lane == 0) scales[block] = scale;
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
        *reinterpret_cast<uint2*>(values + offset + lane * 8) = make_uint2(packed[0], packed[1]);
    } else {
        for (uint32_t d = lane; d < block_size; d += 16) {
            const float v = fmaxf(-448.f, fminf(448.f, __bfloat162float(input[offset + d]) * inv_scale));
            values[offset + d] = __nv_fp8_e4m3(v).__x;
        }
    }
}

extern "C" void glm_quantize_fp8(GlmCtx* ctx, const void* input, uint8_t* values,
                                 float* scales, size_t num_blocks, uint32_t block_size) {
    cudaSetDevice(ctx->device_id);
    if (num_blocks == 0) return;
    const bool vectorized = block_size == 128 &&
        reinterpret_cast<uintptr_t>(input) % 16 == 0 && reinterpret_cast<uintptr_t>(values) % 8 == 0;
    auto kernel = vectorized ? quantize_fp8_kernel<true> : quantize_fp8_kernel<false>;
    kernel<<<(num_blocks + 15) / 16, 256, 0, GLM_STREAM(ctx)>>>(
        static_cast<const __nv_bfloat16*>(input), values, scales, num_blocks, block_size);
}
