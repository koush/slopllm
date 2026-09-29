#include "glm_ops.h"
#include <cuda_bf16.h>
#include <math_constants.h>

// Coefficients are [rows, side=2, tap=2, channels/group_size].
// Base kernels are [side=2, tap=2, channels]. Each block is independent.
__global__ void dflash2_conv_kernel(__nv_bfloat16* out, const __nv_bfloat16* x,
    const __nv_bfloat16* delta, const __nv_bfloat16* base,
    int rows, int channels, int block_size, int group_size, int side) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (c >= channels || row >= rows) return;
    int groups = channels / group_size;
    int di = row * 4 * groups + side * 2 * groups + c / group_size;
    int bi = side * 2 * channels + c;
    float a = __bfloat162float(base[bi]) + __bfloat162float(delta[di]);
    float value = a * __bfloat162float(x[row * channels + c]);
    if (row % block_size) {
        float b = __bfloat162float(base[bi + channels]) + __bfloat162float(delta[di + groups]);
        value += b * __bfloat162float(x[(row - 1) * channels + c]);
    }
    out[row * channels + c] = __float2bfloat16(value);
}

void glm_dflash2_conv(GlmCtx* ctx, void* out, const void* input,
    const void* coefficients, const void* base, int rows, int channels,
    int block_size, int group_size, int side) {
    cudaSetDevice(ctx->device_id);
    if (!rows) return;
    dflash2_conv_kernel<<<dim3((channels + 255) / 256, rows), 256, 0, GLM_STREAM(ctx)>>>(
        static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(coefficients), static_cast<const __nv_bfloat16*>(base),
        rows, channels, block_size, group_size, side);
}

__global__ void dflash2_scores(float* scores, const __nv_bfloat16* gates,
    const int* ids, const float* logits, const __nv_bfloat16* predecessor,
    const __nv_bfloat16* successor, const int* anchors, int depth, int k, int rank, int vocab) {
    int edge = blockIdx.x;
    int candidate = edge % k, previous = (edge / k) % k, row = edge / (k * k);
    int a = row % depth ? ids[(row - 1) * k + previous] : anchors[row / depth];
    int b = ids[row * k + candidate];
    if (a < 0 || a >= vocab || b < 0 || b >= vocab) {
        if (!threadIdx.x) scores[edge] = -CUDART_INF_F;
        return;
    }
    float sum = 0;
    for (int r = threadIdx.x; r < rank; r += blockDim.x) {
        // Match the BF16 gated codebook and BF16 bilinear output in the reference.
        float gated = __bfloat162float(__float2bfloat16(
            __bfloat162float(predecessor[a * rank + r]) * __bfloat162float(gates[row * rank + r])));
        sum += gated * __bfloat162float(successor[b * rank + r]);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) scores[edge] = logits[row * k + candidate] + __bfloat162float(__float2bfloat16(partial[0]));
}

__global__ void dflash2_walk(int* tokens, const float* scores, const int* ids, int depth, int k) {
    int batch = blockIdx.x;
    if (threadIdx.x) return;
    int previous = 0;
    for (int d = 0; d < depth; d++) {
        int row = batch * depth + d;
        const float* values = scores + (row * k + previous) * k;
        int best = 0;
        for (int j = 1; j < k; j++) if (values[j] > values[best]) best = j;
        tokens[row] = ids[row * k + best];
        previous = best;
    }
}

void glm_dflash2_select(GlmCtx* ctx, float* scores, int* tokens, const void* gates,
    const int* candidates, const float* logits, const void* predecessor,
    const void* successor, const int* anchors, int batch, int depth, int top_k, int rank, int vocab) {
    cudaSetDevice(ctx->device_id);
    dflash2_scores<<<batch * depth * top_k * top_k, 256, 0, GLM_STREAM(ctx)>>>(scores,
        static_cast<const __nv_bfloat16*>(gates), candidates, logits,
        static_cast<const __nv_bfloat16*>(predecessor), static_cast<const __nv_bfloat16*>(successor),
        anchors, depth, top_k, rank, vocab);
    dflash2_walk<<<batch, 32, 0, GLM_STREAM(ctx)>>>(tokens, scores, candidates, depth, top_k);
}
