#include "glm_ops.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>

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

// mHC weights and accumulators are FP32; BF16 residuals must not cause the
// projection weights or the normalization intermediates to be rounded to BF16.
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

__device__ __forceinline__ unsigned mhc_tf32(float value) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void mhc_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mhc_copy_async(void* shared, const void* global, bool valid) {
    const unsigned address = __cvta_generic_to_shared(shared);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
        :: "r"(address), "l"(global), "r"(valid ? 16 : 0) : "memory");
}

template<int TileM>
__device__ __forceinline__ void mhc_load_async(bf16* sx, float* sw, const bf16* x,
                                               const float* fn, int first_row, int kbase, int width, int rows) {
    // The async path requires width divisible by eight: all source rows and
    // 16-byte vectors are aligned, including the final partially filled tile.
    for (int i = threadIdx.x; i < TileM * 16; i += blockDim.x) {
        const int r = i / 16, k = (i % 16) * 8;
        const bool valid = first_row + r < rows && kbase + k < width;
        mhc_copy_async(sx + r * 136 + k, valid ? x + size_t(first_row + r) * width + kbase + k : x, valid);
    }
    for (int i = threadIdx.x; i < 24 * 32; i += blockDim.x) {
        const int c = i / 32, k = (i % 32) * 4;
        const bool valid = kbase + k < width;
        mhc_copy_async(sw + c * 132 + k, valid ? fn + size_t(c) * width + kbase + k : fn, valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Like b12x's corrected TF32 mHC projection, BF16 activations are represented
// exactly and FP32 weights are split into TF32 high + residual. Flush separate
// high/low accumulators every 32 K elements to bound Tensor Core accumulation
// error. The scratch contract remains [rows,25,8], including FP32 squared norms.
template<int TileM, int KWarps, bool Pipeline = false>
__global__ void mhc_projection_tf32(float* partials, const bf16* x, const float* fn, int width, int rows) {
    constexpr int TileK = 128;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int first_row = blockIdx.x * TileM, part = blockIdx.y;
    // Lane groups address different rows at the same K offsets. Padding makes
    // those rows hit distinct shared-memory banks instead of serializing them.
    __shared__ __align__(16) bf16 sx[Pipeline ? 2 : 1][TileM][TileK + 8];
    __shared__ __align__(16) float sw[Pipeline ? 2 : 1][24][TileK + 4];
    __shared__ float reductions[KWarps][TileM][24];
    constexpr int NormWarps = KWarps == 1 ? 4 : 3 * KWarps;
    float total[TileM / 16][4] = {};
    float norm[(TileM + NormWarps - 1) / NormWarps] = {};
    const int tiles = (width + TileK - 1) / TileK;
    if constexpr (Pipeline) {
        if (part < tiles) mhc_load_async<TileM>(&sx[0][0][0], &sw[0][0][0], x, fn, first_row, part * TileK, width, rows);
    }
    for (int tile = part; tile < tiles; tile += MHC_PARTS) {
        const int stage = Pipeline ? ((tile - part) / MHC_PARTS) % 2 : 0;
        const int kbase = tile * TileK;
        if constexpr (Pipeline) {
            if (tile + MHC_PARTS < tiles) {
                mhc_load_async<TileM>(&sx[stage ^ 1][0][0], &sw[stage ^ 1][0][0], x, fn,
                    first_row, (tile + MHC_PARTS) * TileK, width, rows);
                // Current tile is ready; the next tile can transfer during MMA.
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else asm volatile("cp.async.wait_group 0;" ::: "memory");
        } else {
            for (int i = tid; i < TileM * TileK; i += blockDim.x) {
                const int r = i / TileK, k = i % TileK;
                sx[stage][r][k] = first_row + r < rows && kbase + k < width
                    ? x[size_t(first_row + r) * width + kbase + k] : __float2bfloat16(0.f);
            }
            for (int i = tid; i < 24 * TileK; i += blockDim.x) {
                const int c = i / TileK, k = i % TileK;
                sw[stage][c][k] = kbase + k < width ? fn[size_t(c) * width + kbase + k] : 0.f;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < (TileM + NormWarps - 1) / NormWarps; r++) {
            #pragma unroll
            for (int k = lane; k < TileK; k += 32) {
                if (r * NormWarps + warp < TileM) {
                    const float value = float(sx[stage][r * NormWarps + warp][k]);
                    norm[r] += value * value;
                }
            }
        }
        if (warp < 3 * KWarps) {
            #pragma unroll
            for (int m = 0; m < TileM / 16; m++) {
                for (int chunk = (warp / 3) * 32; chunk < TileK; chunk += 32 * KWarps) {
                    float high[4] = {}, low[4] = {};
                    #pragma unroll
                    for (int kk = 0; kk < 32; kk += 8) {
                        const int r = m * 16 + lane / 4, k = chunk + kk + lane % 4;
                        const unsigned a[4] = {
                            __float_as_uint(float(sx[stage][r][k])), __float_as_uint(float(sx[stage][r + 8][k])),
                            __float_as_uint(float(sx[stage][r][k + 4])), __float_as_uint(float(sx[stage][r + 8][k + 4])) };
                        const float b0 = sw[stage][(warp % 3) * 8 + lane / 4][k], b1 = sw[stage][(warp % 3) * 8 + lane / 4][k + 4];
                        const unsigned h0 = mhc_tf32(b0), h1 = mhc_tf32(b1);
                        mhc_mma(low, a, mhc_tf32(b0 - __uint_as_float(h0)), mhc_tf32(b1 - __uint_as_float(h1)));
                        mhc_mma(high, a, h0, h1);
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; i++) total[m][i] += high[i] + low[i];
                }
            }
        }
        // All readers must finish before this buffer becomes a future load target.
        __syncthreads();
    }
    if (warp < 3 * KWarps) {
        #pragma unroll
        for (int m = 0; m < TileM / 16; m++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int row = first_row + m * 16 + lane / 4 + (i / 2) * 8;
                const int c = (warp % 3) * 8 + (lane % 4) * 2 + i % 2;
                if constexpr (KWarps > 1) reductions[warp / 3][row - first_row][c] = total[m][i];
                else if (row < rows) partials[(size_t(row) * 25 + c) * MHC_PARTS + part] = total[m][i];
            }
        }
    }
    if constexpr (KWarps > 1) {
        __syncthreads();
        for (int i = tid; i < TileM * 24; i += blockDim.x) {
            float sum = 0.f;
            #pragma unroll
            for (int group = 0; group < KWarps; group++) sum += reductions[group][i / 24][i % 24];
            if (first_row + i / 24 < rows) partials[(size_t(first_row + i / 24) * 25 + i % 24) * MHC_PARTS + part] = sum;
        }
    }
    #pragma unroll
    for (int r = 0; r < (TileM + NormWarps - 1) / NormWarps; r++) {
        float value = norm[r];
        for (int delta = 16; delta > 0; delta /= 2) value += __shfl_down_sync(0xffffffff, value, delta);
        const int row = first_row + r * NormWarps + warp;
        if (lane == 0 && r * NormWarps + warp < TileM && row < rows) partials[(size_t(row) * 25 + 24) * MHC_PARTS + part] = value;
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
        const char* mode = getenv("SLOPLLM_DSV41_MHC_PROJECTION");
        if (mode && strcmp(mode, "fp32") && strcmp(mode, "tf32")) {
            fprintf(stderr, "SLOPLLM_DSV41_MHC_PROJECTION must be fp32 or tf32\n");
            abort();
        }
        const bool tensor_core = mode ? !strcmp(mode, "tf32") : rows >= 128;
        const char* pipeline_mode = getenv("SLOPLLM_DSV41_MHC_PIPELINE");
        if (pipeline_mode && strcmp(pipeline_mode, "0") && strcmp(pipeline_mode, "1")) {
            fprintf(stderr, "SLOPLLM_DSV41_MHC_PIPELINE must be 0 or 1\n"); abort();
        }
        const bool pipeline = (4 * dim) % 8 == 0
            && ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(fn)) & 15) == 0
            && (!pipeline_mode || !strcmp(pipeline_mode, "1"));
        // Decode favors the FP32 split-K reduction. Medium batches parallelize
        // K within the CTA; large batches favor more rows and weight reuse.
        if (!tensor_core) mhc_projection_partials<<<dim3(24, rows, MHC_PARTS), 256, 0, GLM_STREAM(ctx)>>>(partials, (const bf16*)x, fn, 4 * dim);
        else if (rows < 2048) {
            if (pipeline) mhc_projection_tf32<16,4,true><<<dim3((rows+15)/16, MHC_PARTS), 384, 0, GLM_STREAM(ctx)>>>(partials, (const bf16*)x, fn, 4 * dim, rows);
            else mhc_projection_tf32<16,4><<<dim3((rows+15)/16, MHC_PARTS), 384, 0, GLM_STREAM(ctx)>>>(partials, (const bf16*)x, fn, 4 * dim, rows);
        } else {
            if (pipeline) mhc_projection_tf32<32,1,true><<<dim3((rows+31)/32, MHC_PARTS), 128, 0, GLM_STREAM(ctx)>>>(partials, (const bf16*)x, fn, 4 * dim, rows);
            else mhc_projection_tf32<32,1><<<dim3((rows+31)/32, MHC_PARTS), 128, 0, GLM_STREAM(ctx)>>>(partials, (const bf16*)x, fn, 4 * dim, rows);
        }
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
