#include "glm_ops.h"
#include "glm_quantize.cuh"

extern "C" void glm_quantize_fp8(GlmCtx* ctx, const void* input, uint8_t* values,
                                 float* scales, size_t num_blocks, uint32_t block_size) {
    cudaSetDevice(ctx->device_id);
    if (num_blocks == 0) return;
    const bool vectorized = block_size == 128 &&
        reinterpret_cast<uintptr_t>(input) % 16 == 0 && reinterpret_cast<uintptr_t>(values) % 8 == 0;
    auto kernel = vectorized ? glm_quantize::quantize_fp8_kernel<true> : glm_quantize::quantize_fp8_kernel<false>;
    kernel<<<(num_blocks + 15) / 16, 256, 0, GLM_STREAM(ctx)>>>(
        static_cast<const __nv_bfloat16*>(input), values, scales, num_blocks, block_size, 0, 0, 0);
}
