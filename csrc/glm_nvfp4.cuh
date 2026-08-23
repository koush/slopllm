#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
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

// FP8 E4M3 → FP16 conversion via inline PTX, reading the HIGH output lane.
//
// Workaround for a hardware defect on certain SM120a parts (e.g. GPU 3 on this
// cluster): the LOW output lane of `cvt.rn.f16x2.e4m3x2` intermittently faults
// (returns garbage), while the HIGH lane is always clean. The compiler's
// `static_cast<float>` / `__half` conversions always read the low lane, so they
// always hit the defect.
//
// This routine duplicates the input byte into both lanes of the f16x2
// conversion, then extracts the (clean) HIGH f16 result.
__device__ __forceinline__ __half fp8_e4m3_to_half(uint8_t b) {
    uint16_t in = (uint16_t)b | ((uint16_t)b << 8);
    uint32_t h2;
    asm volatile(
        "cvt.rn.f16x2.e4m3x2 %0, %1;\n"
        : "=r"(h2)
        : "h"(in));
    uint16_t h16 = (uint16_t)(h2 >> 16);
    __half h;
    memcpy(&h, &h16, sizeof(h));
    return h;
}

__device__ __forceinline__ __half fp8_e4m3_to_half(__nv_fp8_e4m3 v) {
    return fp8_e4m3_to_half(reinterpret_cast<const uint8_t&>(v));
}

// FP8 E4M3 → FP32 convenience wrapper. Equivalent to the former
// `static_cast<float>(__nv_fp8_e4m3)` but routed through the high-lane PTX path
// to avoid the GPU 3 low-lane defect. The intermediate FP16 is exact (every
// E4M3 value is representable in FP16), so the result is bit-identical to a
// direct FP8→FP32 convert on healthy hardware.
__device__ __forceinline__ float fp8_e4m3_to_float(__nv_fp8_e4m3 v) {
    return __half2float(fp8_e4m3_to_half(v));
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t v) {
    return __half2float(fp8_e4m3_to_half(v));
}
