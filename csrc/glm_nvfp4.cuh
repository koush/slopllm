#pragma once

#include <cuda_fp16.h>
#include <string.h>

// SM120a native FP4→F16x2 conversion instruction.
// Takes a packed byte (two E2M1 FP4 values: lo nibble → .x, hi nibble → .y)
// and returns a 32-bit register containing two F16 values packed as __half2.
__device__ __forceinline__ uint32_t fp4x2_to_half2(uint8_t packed) {
    uint32_t result;
    uint32_t b = (uint32_t)packed;
    asm volatile(
        "{\n"
        ".reg .b8 fp4_byte;\n"
        "mov.b32 {fp4_byte, _, _, _}, %1;\n"
        "cvt.rn.f16x2.e2m1x2 %0, fp4_byte;\n"
        "}\n"
        : "=r"(result)
        : "r"(b));
    return result;
}

// Decode a single E2M1 nibble to float (software, always available).
__device__ __forceinline__ float fp4_e2m1_decode(uint8_t nibble) {
    uint32_t n = (uint32_t)nibble & 0x7u;
    uint32_t fp = (n < 2u) ? (n * 0x3F000000u)
                            : (((126u + (n >> 1u)) << 23u) | ((n & 1u) << 22u));
    fp |= (uint32_t)(nibble >> 3u) << 31u;
    return __uint_as_float(fp);
}

// Decode both nibbles of a packed byte to float2.
// On SM120a+, uses a single hardware cvt.rn.f16x2.e2m1x2 instruction,
// then extracts two floats via memcpy bit-reinterpret + __half22float2.
// On older architectures, falls back to software decode.
__device__ __forceinline__ float2 fp4x2_to_float2(uint8_t packed) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    uint32_t h2 = fp4x2_to_half2(packed);
    __half2 h;
    memcpy(&h, &h2, sizeof(h));
    return __half22float2(h);
#else
    float2 f;
    f.x = fp4_e2m1_decode(packed & 0x0Fu);
    f.y = fp4_e2m1_decode(packed >> 4u);
    return f;
#endif
}
