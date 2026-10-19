#include "glm_ops.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {
using bf16 = __nv_bfloat16;

__device__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// All callers use complete warps. The returned sum is broadcast to the block.
__device__ float block_sum(float x, float* scratch) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int d = 16; d; d /= 2) x += __shfl_down_sync(0xffffffff, x, d);
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        x = lane < blockDim.x / 32 ? scratch[lane] : 0.f;
        for (int d = 16; d; d /= 2) x += __shfl_down_sync(0xffffffff, x, d);
        if (lane == 0) scratch[0] = x;
    }
    __syncthreads();
    const float result = scratch[0];
    __syncthreads();
    return result;
}

// Correctness-first packed recurrent scan. One block owns a sequence/head and
// keeps its FP32 state live across tokens. Empty sequences leave state untouched.
// Inputs are token-major [tokens, heads, 128]; beta is [tokens, heads].
__global__ void kda_scan(bf16* out, float* state, const bf16* q, const bf16* k,
                         const bf16* v, const bf16* forget, const bf16* beta,
                         const float* a_log, const float* dt_bias,
                         const int* cu, int heads, float lower_bound) {
    const int h = blockIdx.x % heads, b = blockIdx.x / heads, j = threadIdx.x;
    const int begin = cu[b], end = cu[b + 1];
    if (begin == end) return;
    float* sh = state + (static_cast<size_t>(b) * heads + h) * 128 * 128;
    float s[128];
    for (int i = 0; i < 128; ++i) s[i] = sh[i * 128 + j];
    __shared__ float sq[128], sk[128], decay[128], scratch[4];
    const float rate = expf(a_log[h]);
    for (int t = begin; t < end; ++t) {
        const size_t offset = (static_cast<size_t>(t) * heads + h) * 128;
        const float qv = __bfloat162float(q[offset + j]);
        const float kv = __bfloat162float(k[offset + j]);
        const float qnorm = sqrtf(block_sum(qv * qv, scratch) + 1e-6f);
        const float knorm = sqrtf(block_sum(kv * kv, scratch) + 1e-6f);
        sq[j] = qv / qnorm * rsqrtf(128.f);
        sk[j] = kv / knorm;
        decay[j] = expf(lower_bound * sigmoid(rate *
            (__bfloat162float(forget[offset + j]) + dt_bias[h * 128 + j])));
        // PyTorch's sigmoid of BF16 beta logits produces BF16 before the FP32 scan.
        const float bet = __bfloat162float(__float2bfloat16(sigmoid(__bfloat162float(beta[t * heads + h]))));
        __syncthreads();
        float memory = 0.f;
        for (int i = 0; i < 128; ++i) {
            s[i] *= decay[i];
            memory += sk[i] * s[i];
        }
        const float delta = bet * (__bfloat162float(v[offset + j]) - memory);
        float y = 0.f;
        for (int i = 0; i < 128; ++i) {
            s[i] += sk[i] * delta;
            y += sq[i] * s[i];
        }
        out[offset + j] = __float2bfloat16(y);
        __syncthreads();
    }
    for (int i = 0; i < 128; ++i) sh[i * 128 + j] = s[i];
}

__global__ void kda_norm(bf16* out, const bf16* x, const bf16* gate,
                         const bf16* weight, float eps) {
    const size_t offset = static_cast<size_t>(blockIdx.x) * 128 + threadIdx.x;
    const float value = __bfloat162float(x[offset]);
    __shared__ float scratch[4];
    const float inv = rsqrtf(block_sum(value * value, scratch) / 128.f + eps);
    out[offset] = __float2bfloat16(value * inv * __bfloat162float(weight[threadIdx.x]) *
                                  sigmoid(__bfloat162float(gate[offset])));
}

// mHC coefficient projection is FP32 (including normalization), not a BF16 GEMM.
constexpr int MHC_PARTS = 8;

// Parallelize both the 24 coefficients and K. Scratch is [rows,25,8]:
// 24 dot-product partials plus squared-norm partials, all FP32.
__global__ void mhc_projection_partials(float* partials, const bf16* x, const float* fn, int width) {
    const int c = blockIdx.x, row = blockIdx.y, part = blockIdx.z, tid = threadIdx.x;
    const bf16* xr = x + size_t(row) * width;
    float dot = 0.f, norm = 0.f;
    for (int i = part * blockDim.x + tid; i < width; i += MHC_PARTS * blockDim.x) {
        const float value = float(xr[i]);
        dot += value * fn[size_t(c) * width + i];
        if (c == 0) norm += value * value;
    }
    __shared__ float scratch[8];
    const float result = block_sum(dot, scratch);
    if (tid == 0) partials[(size_t(row) * 25 + c) * MHC_PARTS + part] = result;
    if (c == 0) {
        const float squared = block_sum(norm, scratch);
        if (tid == 0) partials[(size_t(row) * 25 + 24) * MHC_PARTS + part] = squared;
    }
}

template <typename Weight, bool DELAYED = false>
__global__ void mhc_pre(bf16* collapsed, float* post, float* comb,
                        const bf16* x, const Weight* fn, const float* base,
                        const float* scale, int dim, float norm_eps, float hc_eps, int iters,
                        const float* previous_pre, float* next_pre, const float* partials) {
    const int row = blockIdx.x, tid = threadIdx.x, width = 4 * dim;
    const bf16* xr = x + static_cast<size_t>(row) * width;
    __shared__ float scratch[8], logits[24], pre[4], p[4], mix[16];
    if constexpr (DELAYED) {
        if (tid < 24) {
            float dot = 0.f, squared = 0.f;
            for (int p = 0; p < MHC_PARTS; p++) {
                dot += partials[(size_t(row) * 25 + tid) * MHC_PARTS + p];
                squared += partials[(size_t(row) * 25 + 24) * MHC_PARTS + p];
            }
            logits[tid] = dot * rsqrtf(squared / width + norm_eps);
        }
        __syncthreads();
    } else {
        float sum = 0.f;
        for (int i = tid; i < width; i += blockDim.x) {
            const float value = __bfloat162float(xr[i]);
            sum += value * value;
        }
        const float inv = rsqrtf(block_sum(sum, scratch) / width + norm_eps);
        for (int c = 0; c < 24; ++c) {
            float dot = 0.f;
            for (int i = tid; i < width; i += blockDim.x)
                dot += (__bfloat162float(xr[i]) * inv) * float(fn[c * width + i]);
            const float result = block_sum(dot, scratch);
            if (tid == 0) logits[c] = result;
        }
    }
    if constexpr (DELAYED) {
        if (tid < 4) {
            next_pre[row * 4 + tid] = sigmoid(logits[tid] * scale[0] + base[tid]) + hc_eps;
            post[row * 4 + tid] = 2.f * sigmoid(logits[4 + tid] * scale[1] + base[4 + tid]);
        }
        // One lane per 4x4 mixing coefficient. Shuffle broadcasts retain the
        // original left-to-right denominator sums without serial matrix loops.
        if (tid < 16) {
            constexpr unsigned mask = 0xffff;
            const int first = (tid / 4) * 4, col = tid % 4;
            float value = logits[8 + tid] * scale[2] + base[8 + tid];
            float maxval = -INFINITY;
            for (int j = 0; j < 4; j++) maxval = fmaxf(maxval, __shfl_sync(mask, value, first + j));
            value = expf(value - maxval);
            float denom = 0.f;
            for (int j = 0; j < 4; j++) denom += __shfl_sync(mask, value, first + j);
            value = value / denom + hc_eps;
            for (int iter = 0; iter < iters; iter++) {
                if (iter > 0) {
                    denom = hc_eps;
                    for (int j = 0; j < 4; j++) denom += __shfl_sync(mask, value, first + j);
                    value /= denom;
                }
                denom = hc_eps;
                for (int i = 0; i < 4; i++) denom += __shfl_sync(mask, value, i * 4 + col);
                value /= denom;
            }
            comb[row * 16 + tid] = value;
        }
    } else if (tid == 0) {
        for (int i = 0; i < 4; ++i) {
            pre[i] = sigmoid(logits[i] * scale[0] + base[i]) + hc_eps;
            if constexpr (DELAYED) next_pre[row * 4 + i] = pre[i];
            p[i] = 2.f * sigmoid(logits[4 + i] * scale[1] + base[4 + i]);
            post[row * 4 + i] = p[i];
        }
        for (int i = 0; i < 4; ++i) {
            float maxval = -INFINITY, denom = 0.f;
            for (int j = 0; j < 4; ++j) {
                const int idx = i * 4 + j;
                mix[idx] = logits[8 + idx] * scale[2] + base[8 + idx];
                maxval = fmaxf(maxval, mix[idx]);
            }
            for (int j = 0; j < 4; ++j) { mix[i * 4 + j] = expf(mix[i * 4 + j] - maxval); denom += mix[i * 4 + j]; }
            for (int j = 0; j < 4; ++j) mix[i * 4 + j] = mix[i * 4 + j] / denom + hc_eps;
        }
        for (int iter = 0; iter < iters; ++iter) {
            if (iter > 0) {
                for (int i = 0; i < 4; ++i) {
                    float denom = hc_eps;
                    for (int j = 0; j < 4; ++j) denom += mix[i * 4 + j];
                    for (int j = 0; j < 4; ++j) mix[i * 4 + j] /= denom;
                }
            }
            for (int j = 0; j < 4; ++j) {
                float denom = hc_eps;
                for (int i = 0; i < 4; ++i) denom += mix[i * 4 + j];
                for (int i = 0; i < 4; ++i) mix[i * 4 + j] /= denom;
            }
        }
        for (int i = 0; i < 16; ++i) comb[row * 16 + i] = mix[i];
    }
    __syncthreads();
    for (int i = tid; i < dim; i += blockDim.x) {
        float value = 0.f;
        for (int s = 0; s < 4; ++s) {
            const float coefficient = DELAYED ? (previous_pre ? previous_pre[row * 4 + s] : float(s == 0)) : pre[s];
            value += coefficient * __bfloat162float(xr[s * dim + i]);
        }
        collapsed[static_cast<size_t>(row) * dim + i] = __float2bfloat16(value);
    }
}

__global__ void mhc_post(bf16* out, const bf16* residual, const bf16* block,
                         const float* post, const float* comb, int dim) {
    const int row = blockIdx.x, stream = blockIdx.y;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float value = post[row * 4 + stream] * __bfloat162float(block[static_cast<size_t>(row) * dim + d]);
        for (int i = 0; i < 4; ++i)
            value += comb[row * 16 + i * 4 + stream] *
                __bfloat162float(residual[(static_cast<size_t>(row) * 4 + i) * dim + d]);
        out[(static_cast<size_t>(row) * 4 + stream) * dim + d] = __float2bfloat16(value);
    }
}

__global__ void mhc_streams(bf16* out, const bf16* x, int dim, bool expand) {
    const int row = blockIdx.x;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        if (expand) {
            const bf16 value = x[static_cast<size_t>(row) * dim + d];
            for (int s = 0; s < 4; ++s) out[(static_cast<size_t>(row) * 4 + s) * dim + d] = value;
        } else {
            float value = 0.f;
            for (int s = 0; s < 4; ++s) value += __bfloat162float(x[(static_cast<size_t>(row) * 4 + s) * dim + d]);
            out[static_cast<size_t>(row) * dim + d] = __float2bfloat16(value * .25f);
        }
    }
}
} // namespace

extern "C" {
void glm_kda_scan(GlmCtx* ctx, void* out, float* state, const void* q, const void* k,
                  const void* v, const void* forget, const void* beta, const float* a_log,
                  const float* dt_bias, const int* cu, int heads, int batch, float lower_bound) {
    cudaSetDevice(ctx->device_id);
    if (batch > 0) kda_scan<<<batch * heads, 128, 0, GLM_STREAM(ctx)>>>(
        (bf16*)out, state, (const bf16*)q, (const bf16*)k, (const bf16*)v,
        (const bf16*)forget, (const bf16*)beta, a_log, dt_bias, cu, heads, lower_bound);
}
void glm_kda_norm(GlmCtx* ctx, void* out, const void* x, const void* gate, const void* weight, int rows, float eps) {
    cudaSetDevice(ctx->device_id);
    if (rows > 0) kda_norm<<<rows, 128, 0, GLM_STREAM(ctx)>>>((bf16*)out, (const bf16*)x, (const bf16*)gate, (const bf16*)weight, eps);
}
void glm_mhc_pre(GlmCtx* ctx, void* out, float* post, float* comb, const void* x,
                 const void* fn, const float* base, const float* scale,
                 int rows, int dim, float norm_eps, float hc_eps, int iters) {
    cudaSetDevice(ctx->device_id);
    if (rows > 0) mhc_pre<bf16><<<rows, 256, 0, GLM_STREAM(ctx)>>>((bf16*)out, post, comb, (const bf16*)x, (const bf16*)fn, base, scale, dim, norm_eps, hc_eps, iters, nullptr, nullptr, nullptr);
}
void glm_dsv41_mhc_pre(GlmCtx* ctx, void* out, float* pre, float* post, float* comb,
                       const void* x, const float* fn, const float* base, const float* scale,
                       const float* previous_pre, int rows, int dim, float norm_eps, float hc_eps, int iters, float* partials) {
    cudaSetDevice(ctx->device_id);
    if (rows > 0) {
        mhc_projection_partials<<<dim3(24, rows, MHC_PARTS), 256, 0, GLM_STREAM(ctx)>>>(partials, (const bf16*)x, fn, 4 * dim);
        mhc_pre<float, true><<<rows, 256, 0, GLM_STREAM(ctx)>>>(
            (bf16*)out, post, comb, (const bf16*)x, fn, base, scale, dim, norm_eps, hc_eps, iters, previous_pre, pre, partials);
    }
}
void glm_mhc_post(GlmCtx* ctx, void* out, const void* residual, const void* block,
                  const float* post, const float* comb, int rows, int dim) {
    cudaSetDevice(ctx->device_id);
    if (rows > 0) mhc_post<<<dim3(rows, 4), 256, 0, GLM_STREAM(ctx)>>>((bf16*)out, (const bf16*)residual, (const bf16*)block, post, comb, dim);
}
void glm_mhc_streams(GlmCtx* ctx, void* out, const void* x, int rows, int dim, int expand) {
    cudaSetDevice(ctx->device_id);
    if (rows > 0) mhc_streams<<<rows, 256, 0, GLM_STREAM(ctx)>>>((bf16*)out, (const bf16*)x, dim, expand != 0);
}
}
