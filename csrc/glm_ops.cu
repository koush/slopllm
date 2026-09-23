#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace cg = cooperative_groups;

// Passed by value so captured launches need no temporary pointer/size buffers.
struct PrefetchL2Inputs {
    const char* data[8];
    size_t bytes[8];
};

// Bulk range prefetch: each thread covers one contiguous 16B-aligned slice of
// each tensor with a single cp.async.bulk.prefetch (one instruction per
// (thread, tensor) instead of one per 32B offset). The fractional evict_last
// policy marks only 25% of prefetched lines evict_last; the rest insert at
// evict-normal, so the full range still warms L2 while stale sticky lines are
// capped at ~6MB and cannot pin the L2 against concurrent kernels.
__global__ void prefetch_l2_kernel(PrefetchL2Inputs inputs, int count) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 0.25;" : "=l"(policy));
    const int total_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int t = 0; t < count; t++) {
        const size_t bytes = inputs.bytes[t];
        size_t chunk = ((bytes + total_threads - 1) / total_threads + 15) & ~size_t(15);
        const size_t start = (size_t)tid * chunk;
        if (start >= bytes) continue;
        const char* base = inputs.data[t] + start;
        size_t len = chunk < bytes - start ? chunk : bytes - start;
        len &= ~size_t(15);  // bulk prefetch requires 16B-multiple size
        if (len) {
            asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                         :: "l"(base), "r"((uint32_t)len), "l"(policy) : "memory");
        } else {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(base) : "memory");
        }
    }
}

void glm_prefetch_l2(GlmCtx* ctx, const void* const* data, const size_t* bytes, int count) {
    PrefetchL2Inputs inputs{};
    int used = 0;
    size_t blocks = 0;
    for (int i = 0; i < count; i++) {
        if (!bytes[i]) continue;
        inputs.data[used] = static_cast<const char*>(data[i]);
        inputs.bytes[used++] = bytes[i];
        blocks = std::min(size_t(32), blocks + (bytes[i] - 1) / (256 * 32) + 1);
    }
    if (!used) return;
    cudaSetDevice(ctx->device_id);
    prefetch_l2_kernel<<<blocks, 256, 0, GLM_STREAM(ctx)>>>(inputs, used);
}

#define CUBLAS(ctx) (*reinterpret_cast<cublasHandle_t*>(&(ctx)->cublas_handle))

// ---------------------------------------------------------------------------
// Host helpers: block size computation
// ---------------------------------------------------------------------------

inline int compute_block_size(int dim, bool power_of_two = false, int max_block = 256, int min_block = 32) {
    int bs = max_block;
    if (bs > dim) bs = ((dim + 31) / 32) * 32;
    if (power_of_two) {
        int p = 1;
        while (p < bs) p <<= 1;
        bs = p;
    }
    if (bs > max_block) bs = max_block;
    if (bs < min_block) bs = min_block;
    return bs;
}

// ---------------------------------------------------------------------------
// Device helpers: tree reduction
// ---------------------------------------------------------------------------

__device__ __forceinline__ float rope_rotate_neox(float val, float paired_val, float cos_val, float sin_val, int d, int half) {
    float rotated = (d < half) ? -paired_val : paired_val;
    return val * cos_val + rotated * sin_val;
}

__device__ __forceinline__ float rope_rotate_interleaved(float val, float paired_val, float cos_val, float sin_val, int d) {
    float rotated = (d % 2 == 0) ? -paired_val : paired_val;
    return val * cos_val + rotated * sin_val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* s_partial, int tid) {
    constexpr int WarpSize = 32;
    int lane = tid % WarpSize;
    int warp = tid / WarpSize;
    int warps_per_block = blockDim.x / WarpSize;

    #pragma unroll
    for (int offset = WarpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (lane == 0) s_partial[warp] = val;
    __syncthreads();

    if (warp == 0) {
        val = (tid < warps_per_block) ? s_partial[tid] : 0.0f;
        #pragma unroll
        for (int offset = WarpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) s_partial[0] = val;
    }
    __syncthreads();
    return s_partial[0];
}

__device__ void block_reduce_max_idx(float* s_vals, int* s_idxs, int tid) {
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] > s_vals[tid] ||
                (s_vals[tid + s] == s_vals[tid] && s_idxs[tid + s] < s_idxs[tid])) {
                s_vals[tid] = s_vals[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }
}

__device__ void block_reduce_max(float* sdata, int tid) {
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
}

__device__ __forceinline__ float fast_tanh(float x) {
    float result;
    asm("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 0.5f * (fast_tanh(0.5f * x) + 1.0f);
}

__device__ __forceinline__ float silu_f(float x) {
    return x * 0.5f * (fast_tanh(0.5f * x) + 1.0f);
}

__device__ __forceinline__ float relu_f(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ---------------------------------------------------------------------------
// BF16 vector I/O helpers now live in glm_ops.h (shared across .cu TUs).
// ---------------------------------------------------------------------------

// Compute sum of squares of bf16 vector, reduce across block, return inv_rms.
// Caller must provide extern __shared__ float sdata[].
template <bool EvenDim = true>
__device__ float compute_inv_rms(const __nv_bfloat16* x, int dim, float eps, float* sdata) {
    float sum = 0.0f;
    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float2 v = load_bf16x2(x + i);
        sum += v.x * v.x + v.y * v.y;
    }
    if constexpr (!EvenDim) {
        if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
            float val = __bfloat162float(x[dim - 1]);
            sum += val * val;
        }
    }
    float total = block_reduce_sum(sum, sdata, threadIdx.x);
    return rsqrtf(total / dim + eps);
}

// ---------------------------------------------------------------------------
// RMSNorm kernel (register-cached: reads input once)
// ---------------------------------------------------------------------------

template <int MaxPairs, bool EvenDim = true>
__global__ void __launch_bounds__(1024) rmsnorm_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    float eps,
    int dim
) {
    int row = blockIdx.x;
    const __nv_bfloat16* x = input + row * dim;
    __nv_bfloat16* o = out + row * dim;

    extern __shared__ float sdata[];

    float2 xv[MaxPairs];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < MaxPairs; k++) {
        int i = threadIdx.x * 2 + k * blockDim.x * 2;
        if (i + 1 < dim) {
            xv[k] = load_bf16x2(x + i);
            sum += xv[k].x * xv[k].x + xv[k].y * xv[k].y;
        }
    }
    if constexpr (!EvenDim) {
        if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
            float val = __bfloat162float(x[dim - 1]);
            sum += val * val;
        }
    }

    float total = block_reduce_sum(sum, sdata, threadIdx.x);
    float inv_rms = rsqrtf(total / dim + eps);

    #pragma unroll
    for (int k = 0; k < MaxPairs; k++) {
        int i = threadIdx.x * 2 + k * blockDim.x * 2;
        if (i + 1 < dim) {
            float2 wv = load_bf16x2(weight + i);
            store_bf16x2(o + i, wv.x * xv[k].x * inv_rms, wv.y * xv[k].y * inv_rms);
        }
    }
    if constexpr (!EvenDim) {
        if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
            float xi = __bfloat162float(x[dim - 1]);
            float wi = __bfloat162float(weight[dim - 1]);
            o[dim - 1] = __float2bfloat16(wi * xi * inv_rms);
        }
    }
}

// Grid-stride fallback for large pairs (double-read, no caching).
template <bool EvenDim = true>
__global__ void __launch_bounds__(1024, 1) rmsnorm_kernel_stride(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    float eps,
    int dim
) {
    int row = blockIdx.x;
    const __nv_bfloat16* x = input + row * dim;
    __nv_bfloat16* o = out + row * dim;

    extern __shared__ float sdata[];

    float inv_rms = compute_inv_rms<EvenDim>(x, dim, eps, sdata);

    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float2 xv = load_bf16x2(x + i);
        float2 wv = load_bf16x2(weight + i);
        store_bf16x2(o + i, wv.x * xv.x * inv_rms, wv.y * xv.y * inv_rms);
    }
    if constexpr (!EvenDim) {
        if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
            float xi = __bfloat162float(x[dim - 1]);
            float wi = __bfloat162float(weight[dim - 1]);
            o[dim - 1] = __float2bfloat16(wi * xi * inv_rms);
        }
    }
}

void glm_rmsnorm(GlmCtx* ctx, void* out, const void* input,
                 const void* weight, float eps, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int max_block = (dim >= 4096 && batch < 64) ? 1024 : 256;
    int block_size = compute_block_size(dim, false, max_block);
    size_t shared_mem = (block_size / 32) * sizeof(float);
    int pairs = (dim + 2 * block_size - 1) / (2 * block_size);
    bool even = (dim & 1) == 0;

#define DISPATCH_RMS(P, EV) \
    rmsnorm_kernel<P, EV><<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>( \
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input, \
        (const __nv_bfloat16*)weight, eps, dim)
#define DISPATCH_RMS_PAIRS(P) do { \
    if (even) { DISPATCH_RMS(P, true); } else { DISPATCH_RMS(P, false); } \
} while(0)

    if (pairs <= 24) {
        switch (pairs) {
            case 1: DISPATCH_RMS_PAIRS(1); break;
            case 2: DISPATCH_RMS_PAIRS(2); break;
            case 3: DISPATCH_RMS_PAIRS(3); break;
            case 4: DISPATCH_RMS_PAIRS(4); break;
            case 5: case 6: DISPATCH_RMS_PAIRS(6); break;
            case 7: case 8: DISPATCH_RMS_PAIRS(8); break;
            case 9: case 10: case 11: case 12: DISPATCH_RMS_PAIRS(12); break;
            case 13: case 14: case 15: case 16: DISPATCH_RMS_PAIRS(16); break;
            default: DISPATCH_RMS_PAIRS(24); break;
        }
    } else {
        if (even) {
            rmsnorm_kernel_stride<true><<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
                (const __nv_bfloat16*)weight, eps, dim);
        } else {
            rmsnorm_kernel_stride<false><<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
                (const __nv_bfloat16*)weight, eps, dim);
        }
    }
#undef DISPATCH_RMS
#undef DISPATCH_RMS_PAIRS
}

// ---------------------------------------------------------------------------
// Fused Add + RMSNorm kernel (register-cached: reads a,b once)
// out[i] = weight[i] * (input_a[i] + input_b[i]) * inv_rms
// residual[i] = input_a[i] + input_b[i]
// ---------------------------------------------------------------------------

template <int MaxPairs, bool EvenDim = true>
__global__ void __launch_bounds__(1024) fused_add_rmsnorm_kernel(
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ input_a,
    const __nv_bfloat16* __restrict__ input_b,
    const __nv_bfloat16* __restrict__ weight,
    float eps, int dim
) {
    int row = blockIdx.x;
    const __nv_bfloat16* a = input_a + row * dim;
    const __nv_bfloat16* b = input_b + row * dim;
    __nv_bfloat16* o = out + row * dim;
    __nv_bfloat16* r = residual + row * dim;

    extern __shared__ float sdata[];

    float2 sv[MaxPairs];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < MaxPairs; k++) {
        int i = threadIdx.x * 2 + k * blockDim.x * 2;
        if (i + 1 < dim) {
            float2 av = load_bf16x2(a + i);
            float2 bv = load_bf16x2(b + i);
            sv[k] = make_float2(av.x + bv.x, av.y + bv.y);
            sum += sv[k].x * sv[k].x + sv[k].y * sv[k].y;
        }
    }
    if constexpr (!EvenDim) {
        if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
            float ai = __bfloat162float(a[dim - 1]);
            float bi = __bfloat162float(b[dim - 1]);
            float si = ai + bi;
            sum += si * si;
        }
    }

    float total = block_reduce_sum(sum, sdata, threadIdx.x);

    float inv_rms = rsqrtf(total / dim + eps);

    #pragma unroll
    for (int k = 0; k < MaxPairs; k++) {
        int i = threadIdx.x * 2 + k * blockDim.x * 2;
        if (i + 1 < dim) {
            float2 wv = load_bf16x2(weight + i);
            store_bf16x2(r + i, sv[k].x, sv[k].y);
            store_bf16x2(o + i, wv.x * sv[k].x * inv_rms, wv.y * sv[k].y * inv_rms);
        }
    }
    if constexpr (!EvenDim) {
        if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
            float ai = __bfloat162float(a[dim - 1]);
            float bi = __bfloat162float(b[dim - 1]);
            float wi = __bfloat162float(weight[dim - 1]);
            float si = ai + bi;
            r[dim - 1] = __float2bfloat16(si);
            o[dim - 1] = __float2bfloat16(wi * si * inv_rms);
        }
    }
}

// Grid-stride fallback for large pairs (double-read, no caching).
template <bool EvenDim = true>
__global__ void __launch_bounds__(1024, 1) fused_add_rmsnorm_kernel_stride(
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ input_a,
    const __nv_bfloat16* __restrict__ input_b,
    const __nv_bfloat16* __restrict__ weight,
    float eps, int dim
) {
    int row = blockIdx.x;
    const __nv_bfloat16* a = input_a + row * dim;
    const __nv_bfloat16* b = input_b + row * dim;
    __nv_bfloat16* o = out + row * dim;
    __nv_bfloat16* r = residual + row * dim;

    extern __shared__ float sdata[];

    float sum = 0.0f;
    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float2 av = load_bf16x2(a + i);
        float2 bv = load_bf16x2(b + i);
        float s0 = av.x + bv.x;
        float s1 = av.y + bv.y;
        sum += s0 * s0 + s1 * s1;
    }
    if constexpr (!EvenDim) {
        if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
            float ai = __bfloat162float(a[dim - 1]);
            float bi = __bfloat162float(b[dim - 1]);
            float si = ai + bi;
            sum += si * si;
        }
    }

    float total = block_reduce_sum(sum, sdata, threadIdx.x);

    float inv_rms = rsqrtf(total / dim + eps);

    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float2 av = load_bf16x2(a + i);
        float2 bv = load_bf16x2(b + i);
        float2 wv = load_bf16x2(weight + i);
        float s0 = av.x + bv.x;
        float s1 = av.y + bv.y;
        store_bf16x2(r + i, s0, s1);
        store_bf16x2(o + i, wv.x * s0 * inv_rms, wv.y * s1 * inv_rms);
    }
    if constexpr (!EvenDim) {
        if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
            float ai = __bfloat162float(a[dim - 1]);
            float bi = __bfloat162float(b[dim - 1]);
            float wi = __bfloat162float(weight[dim - 1]);
            float si = ai + bi;
            r[dim - 1] = __float2bfloat16(si);
            o[dim - 1] = __float2bfloat16(wi * si * inv_rms);
        }
    }
}

void glm_fused_add_rmsnorm(GlmCtx* ctx, void* out, void* residual,
                            const void* input_a, const void* input_b,
                            const void* weight, float eps, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int max_block = (dim >= 4096 && batch < 64) ? 1024 : 256;
    int block_size = compute_block_size(dim, false, max_block);
    size_t shared_mem = (block_size / 32) * sizeof(float);
    int pairs = (dim + 2 * block_size - 1) / (2 * block_size);
    bool even = (dim & 1) == 0;

#define DISPATCH_FUSED(P, EV) \
    fused_add_rmsnorm_kernel<P, EV><<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>( \
        (__nv_bfloat16*)out, (__nv_bfloat16*)residual, \
        (const __nv_bfloat16*)input_a, (const __nv_bfloat16*)input_b, \
        (const __nv_bfloat16*)weight, eps, dim)
#define DISPATCH_FUSED_PAIRS(P) do { \
    if (even) { DISPATCH_FUSED(P, true); } else { DISPATCH_FUSED(P, false); } \
} while(0)

    if (pairs <= 24) {
        switch (pairs) {
            case 1: DISPATCH_FUSED_PAIRS(1); break;
            case 2: DISPATCH_FUSED_PAIRS(2); break;
            case 3: DISPATCH_FUSED_PAIRS(3); break;
            case 4: DISPATCH_FUSED_PAIRS(4); break;
            case 5: case 6: DISPATCH_FUSED_PAIRS(6); break;
            case 7: case 8: DISPATCH_FUSED_PAIRS(8); break;
            case 9: case 10: case 11: case 12: DISPATCH_FUSED_PAIRS(12); break;
            case 13: case 14: case 15: case 16: DISPATCH_FUSED_PAIRS(16); break;
            default: DISPATCH_FUSED_PAIRS(24); break;
        }
    } else {
        if (even) {
            fused_add_rmsnorm_kernel_stride<true><<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out, (__nv_bfloat16*)residual,
                (const __nv_bfloat16*)input_a, (const __nv_bfloat16*)input_b,
                (const __nv_bfloat16*)weight, eps, dim);
        } else {
            fused_add_rmsnorm_kernel_stride<false><<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out, (__nv_bfloat16*)residual,
                (const __nv_bfloat16*)input_a, (const __nv_bfloat16*)input_b,
                (const __nv_bfloat16*)weight, eps, dim);
        }
    }
#undef DISPATCH_FUSED
#undef DISPATCH_FUSED_PAIRS
}

// ---------------------------------------------------------------------------
// Fused per-head RMSNorm + RoPE kernel
// Input: [batch * seq_len, n_heads * head_dim] (projection output, row-major)
// Output: [batch, n_heads, seq_len, head_dim] (HND layout for attention)
// Applies per-head RMSNorm then RoPE, with layout transpose.
// ---------------------------------------------------------------------------

template <bool kInterleaved>
__global__ void __launch_bounds__(256, 4) fused_norm_rope_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ cos_emb,
    const __nv_bfloat16* __restrict__ sin_emb,
    float eps, int rope_dim, int head_dim,
    int n_heads, int seq_len, int batch, int in_stride
) {
    int bhs = blockIdx.x;
    int s = bhs % seq_len;
    int h = (bhs / seq_len) % n_heads;
    int b = bhs / (seq_len * n_heads);

    const __nv_bfloat16* x = in + (b * seq_len + s) * n_heads * in_stride + h * in_stride;
    __nv_bfloat16* o = out + ((b * n_heads + h) * seq_len + s) * head_dim;

    extern __shared__ float sdata[];

    float inv_rms = compute_inv_rms(x, head_dim, eps, sdata);
    int half = rope_dim / 2;
    int cos_base = (b * seq_len + s) * rope_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float xi = __bfloat162float(x[i]);
        float wi = __bfloat162float(weight[i]);
        float ni = wi * xi * inv_rms;

        if (i < rope_dim) {
            if constexpr (kInterleaved) {
                int cos_idx = cos_base + (i >> 1);
                float ci = __bfloat162float(cos_emb[cos_idx]);
                float si = __bfloat162float(sin_emb[cos_idx]);
                int paired_i = (i % 2 == 0) ? i + 1 : i - 1;
                float paired_norm = __bfloat162float(weight[paired_i]) * __bfloat162float(x[paired_i]) * inv_rms;
                ni = rope_rotate_interleaved(ni, paired_norm, ci, si, i);
            } else {
                float ci = __bfloat162float(cos_emb[cos_base + i]);
                float si = __bfloat162float(sin_emb[cos_base + i]);
                float paired_norm = (i < half)
                    ? __bfloat162float(weight[i + half]) * __bfloat162float(x[i + half]) * inv_rms
                    : __bfloat162float(weight[i - half]) * __bfloat162float(x[i - half]) * inv_rms;
                ni = rope_rotate_neox(ni, paired_norm, ci, si, i, half);
            }
        }
        o[i] = __float2bfloat16(ni);
    }
}

void glm_fused_norm_rope(GlmCtx* ctx, void* out, const void* in,
                          const void* weight, const void* cos_emb, const void* sin_emb,
                          float eps, int rope_dim, int head_dim,
                          int n_heads, int seq_len, int batch, int in_stride, bool interleaved) {
    cudaSetDevice(ctx->device_id);
    int total_rows = batch * n_heads * seq_len;
    int block_size = compute_block_size(head_dim, true);
    size_t shared_mem = (block_size / 32) * sizeof(float);
    if (interleaved) {
        fused_norm_rope_kernel<true><<<total_rows, block_size, shared_mem, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)in,
            (const __nv_bfloat16*)weight,
            (const __nv_bfloat16*)cos_emb, (const __nv_bfloat16*)sin_emb,
            eps, rope_dim, head_dim, n_heads, seq_len, batch, in_stride);
    } else {
        fused_norm_rope_kernel<false><<<total_rows, block_size, shared_mem, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)in,
            (const __nv_bfloat16*)weight,
            (const __nv_bfloat16*)cos_emb, (const __nv_bfloat16*)sin_emb,
            eps, rope_dim, head_dim, n_heads, seq_len, batch, in_stride);
    }
}

// ---------------------------------------------------------------------------
// MLA V-Expand kernel
// attn_out: [batch, attn_n_heads, seq_len, kv_lora_rank] (HND layout from FlashInfer)
// v_proj: [n_heads, kv_lora_rank, v_head_dim] (transposed layout for coalesced access)
// result: [batch, seq_len, n_heads * v_head_dim]
// Computes result[b,s,h,j] = sum_k(attn_out[b,h,s,k] * v_proj[h,k,j])
//
// Two key optimizations vs the naive layout [n_heads, v_head_dim, kv_lora_rank]:
//  1. attn_row is shared by all threads in a block -> loaded once into shared memory
//  2. v_proj[h,k,j]: for fixed k, consecutive threads (j) hit consecutive addresses
// ---------------------------------------------------------------------------

// KV_LR > 0: kv_lora_rank is a compile-time constant (enables smem-load unroll).
// KV_LR == 0: fall back to the runtime kv_lora_rank parameter.
template <int KV_LR, int BDX, int RPB>
__global__ void __launch_bounds__(BDX, 4) mla_v_expand_kernel(
    __nv_bfloat16* __restrict__ result,
    const __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ v_proj,
    int kv_lora_rank, int v_head_dim, int n_heads,
    int seq_len, int batch,
    int attn_n_heads, int head_offset,
    int v_proj_head_offset,
    int total_rows
) {
    extern __shared__ float s_attn[];  // [RPB * kv_lora_rank]

    constexpr int TPR = BDX / RPB;
    constexpr bool CT = (KV_LR > 0);
    const int kv_lr = CT ? KV_LR : kv_lora_rank;

    int row_in_block = threadIdx.x / TPR;
    int lane = threadIdx.x % TPR;

    int bhs = blockIdx.x * RPB + row_in_block;
    bool valid = (bhs < total_rows);

    int s = bhs % seq_len;
    int h = (bhs / seq_len) % n_heads;
    int b = bhs / (seq_len * n_heads);

    float* my_s_attn = s_attn + row_in_block * kv_lr;

    if (valid) {
        const __nv_bfloat16* attn_row = attn_out + ((b * attn_n_heads + h + head_offset) * seq_len + s) * kv_lr;
        if constexpr (CT) {
            #pragma unroll
            for (int k = lane; k < KV_LR; k += TPR)
                my_s_attn[k] = __bfloat162float(attn_row[k]);
        } else {
            for (int k = lane; k < kv_lr; k += TPR)
                my_s_attn[k] = __bfloat162float(attn_row[k]);
        }
    }
    __syncthreads();

    if (!valid) return;

    const __nv_bfloat16* w_base = v_proj + (h + v_proj_head_offset) * kv_lr * v_head_dim;
    __nv_bfloat16* result_base = result + (b * seq_len + s) * (n_heads * v_head_dim) + h * v_head_dim;

    for (int j = lane * 2; j < v_head_dim; j += TPR * 2) {
        float sum0 = 0.0f, sum1 = 0.0f;
        if constexpr (CT) {
            #pragma unroll 8
            for (int k = 0; k < KV_LR; k++) {
                float2 wv = load_bf16x2(w_base + k * v_head_dim + j);
                float a = my_s_attn[k];
                sum0 += a * wv.x;
                sum1 += a * wv.y;
            }
        } else {
            for (int k = 0; k < kv_lr; k++) {
                float2 wv = load_bf16x2(w_base + k * v_head_dim + j);
                float a = my_s_attn[k];
                sum0 += a * wv.x;
                sum1 += a * wv.y;
            }
        }
        store_bf16x2(result_base + j, sum0, sum1);
    }
}

// ---------------------------------------------------------------------------
// MLA V-Expand kernel v2: v_proj tiled in shared memory via cg::memcpy_async
//
// Reverses the smem strategy of v1: v_proj tiles are loaded into shared memory
// via cg::memcpy_async (cooperative async copy), and attn values are kept in
// registers. This avoids L2 reads in the inner loop (smem ~30 cycles vs
// L2 ~200 cycles).
//
// Double-buffered with cg::wait_prior: while computing on tile t, tile t+1 is
// in flight. Tile t+2 is prefetched into the current buffer after compute
// finishes (the P2P kernel pattern).
//
// RPB is always 1: cg::memcpy_async is a block-cooperative copy requiring all
// threads to share the same source address, so all threads in a block must
// work on the same head. BDX=128 gives full thread utilization for VHD=256
// (128 threads × 2 elements = 256).
//
// Template parameters:
//   KV_LR  - compile-time kv_lora_rank (e.g. 512)
//   VHD    - compile-time v_head_dim (e.g. 256)
//   BDX    - block size (128)
//   TILE_K - k-dimension tile size (16)
// ---------------------------------------------------------------------------

template <int KV_LR, int VHD, int BDX, int TILE_K>
__global__ void __launch_bounds__(BDX, 8) mla_v_expand_kernel_v2(
    __nv_bfloat16* __restrict__ result,
    const __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ v_proj,
    int n_heads, int seq_len, int batch,
    int attn_n_heads, int head_offset,
    int v_proj_head_offset,
    int total_rows
) {
    static_assert(VHD % 2 == 0, "v_head_dim must be even for bf16x2 loads");
    static_assert(KV_LR % TILE_K == 0, "kv_lora_rank must be divisible by TILE_K");

    constexpr int TPR = BDX;  // RPB=1
    constexpr int NUM_TILES = KV_LR / TILE_K;
    constexpr int VEC = 2;
    constexpr int J_ITERS = (VHD + TPR * VEC - 1) / (TPR * VEC);

    // Smem layout:
    //   s_attn:  [KV_LR] float, 16-byte aligned
    //   s_vproj: [2][TILE_K][VHD] bf16 (double-buffered)
    extern __shared__ char smem_raw[];
    constexpr size_t attn_bytes = KV_LR * sizeof(float);
    constexpr size_t attn_bytes_aligned = (attn_bytes + 15) & ~size_t(15);
    float* s_attn = reinterpret_cast<float*>(smem_raw);
    __nv_bfloat16* s_vproj = reinterpret_cast<__nv_bfloat16*>(
        smem_raw + attn_bytes_aligned);

    auto block = cg::this_thread_block();

    int lane = threadIdx.x;

    int bhs = blockIdx.x;  // RPB=1
    bool valid = (bhs < total_rows);

    int s_pos = bhs % seq_len;
    int h = (bhs / seq_len) % n_heads;
    int b = bhs / (seq_len * n_heads);

    // Use a valid pointer for invalid threads (cg::memcpy_async requires all
    // threads to participate; the data is meaningless but the address must be
    // in-bounds).
    const __nv_bfloat16* w_base = valid
        ? v_proj + (h + v_proj_head_offset) * KV_LR * VHD
        : v_proj;

    constexpr size_t tile_bytes = (size_t)(TILE_K * VHD * sizeof(__nv_bfloat16));

    // Prologue: issue tile 0 into buf[0], tile 1 into buf[1] (if exists).
    // Issued before attn load so async copies are in flight during the
    // bf16→float conversion + smem writes below.
    cg::memcpy_async(block, s_vproj, w_base, tile_bytes);
    if (NUM_TILES > 1) {
        cg::memcpy_async(block, s_vproj + TILE_K * VHD,
                         w_base + TILE_K * VHD, tile_bytes);
    }

    // Phase 1: Load attn into smem — overlaps with async v_proj copies
    if (valid) {
        const __nv_bfloat16* attn_row = attn_out +
            ((b * attn_n_heads + h + head_offset) * seq_len + s_pos) * KV_LR;
        #pragma unroll
        for (int k = lane; k < KV_LR; k += BDX)
            s_attn[k] = __bfloat162float(attn_row[k]);
    }

    __syncthreads();

    // Phase 2: Pipelined tile computation
    float sum[J_ITERS][2] = {};
    float attn_reg[TILE_K];

    for (int t = 0; t < NUM_TILES; t++) {
        int buf = t % 2;

        // Load attn tile into registers — smem read overlaps with wait below
        if (valid) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk++)
                attn_reg[kk] = s_attn[t * TILE_K + kk];
        }

        // Wait for current tile (keep next in flight if any)
        if (t < NUM_TILES - 1) {
            cg::wait_prior<1>(block);
        } else {
            cg::wait(block);
        }

        if (valid) {
            // Compute against v_proj tile in smem
            const __nv_bfloat16* vproj_tile = s_vproj + buf * TILE_K * VHD;

            #pragma unroll
            for (int ji = 0; ji < J_ITERS; ji++) {
                int j = lane * VEC + ji * TPR * VEC;
                if (j < VHD) {
                    #pragma unroll
                    for (int kk = 0; kk < TILE_K; kk++) {
                        float2 wv = load_bf16x2(vproj_tile + kk * VHD + j);
                        float a = attn_reg[kk];
                        sum[ji][0] += a * wv.x;
                        sum[ji][1] += a * wv.y;
                    }
                }
            }
        }

        __syncthreads();  // ensure all threads done reading smem before prefetch

        // Prefetch tile t+2 into current buffer (reuse, not next buffer)
        if (t + 2 < NUM_TILES) {
            cg::memcpy_async(block,
                s_vproj + buf * TILE_K * VHD,
                w_base + (t + 2) * TILE_K * VHD,
                tile_bytes);
        }
    }

    // Store results
    if (valid) {
        __nv_bfloat16* result_base =
            result + (b * seq_len + s_pos) * (n_heads * VHD) + h * VHD;
        #pragma unroll
        for (int ji = 0; ji < J_ITERS; ji++) {
            int j = lane * VEC + ji * TPR * VEC;
            if (j < VHD)
                store_bf16x2(result_base + j, sum[ji][0], sum[ji][1]);
        }
    }
}

void glm_mla_v_expand(GlmCtx* ctx, void* result, const void* attn_out,
                       const void* v_proj,
                       int kv_lora_rank, int v_head_dim, int n_heads,
                       int seq_len, int batch,
                       int attn_n_heads, int head_offset,
                       int v_proj_head_offset) {
    cudaSetDevice(ctx->device_id);

    // cuBLAS strided batched GEMM: for each head h,
    //   result[b*s, h*V : (h+1)*V] = attn_out[h, b, s, :] @ v_proj[h, :, :]
    //
    // Requires batch==1 or seq_len==1 (strided batched can't express both > 1).

    if (batch == 1 || seq_len == 1) {
        // Column-major (cuBLAS convention): C_cm = A_cm @ B_cm
        //   A_cm = v_proj[h]  : [V, Lkv]     lda=V,       strideA = Lkv*V
        //   B_cm = attn_out[h]: [Lkv, B*S]   ldb varies,  strideB varies
        //   C_cm = result[h]  : [V, B*S]     ldc = N*V,   strideC = V
        //
        // B_cm strides depend on prefill vs decode:
        //   Prefill (B=1, S>1): per-head data is contiguous [S, Lkv]
        //     ldb = Lkv, strideB = S*Lkv
        //   Decode (S=1, B>=1): per-head data is strided across attn_n_heads
        //     ldb = attn_n_heads*Lkv, strideB = Lkv

        int BS = batch * seq_len;
        long long Lkv = kv_lora_rank;
        long long V = v_head_dim;

        long long lda = V;
        long long strideA = Lkv * V;

        long long ldb, strideB;
        if (seq_len > 1) {
            ldb = Lkv;
            strideB = (long long)seq_len * Lkv;
        } else {
            ldb = (long long)attn_n_heads * Lkv;
            strideB = Lkv;
        }
        const void* B_base = (const char*)attn_out + (long long)head_offset * strideB * sizeof(__nv_bfloat16);

        long long ldc = (long long)n_heads * V;
        long long strideC = V;

        const void* A_base = (const char*)v_proj + (long long)v_proj_head_offset * strideA * sizeof(__nv_bfloat16);

        float alpha = 1.0f, beta = 0.0f;

        if (true) {
            // this works for dense MLA but not for sparse MLA path?
            cublasGemmStridedBatchedEx(CUBLAS(ctx),
                CUBLAS_OP_N, CUBLAS_OP_N,
                V,          // m
                BS,         // n
                Lkv,        // k
                &alpha,
                A_base, CUDA_R_16BF, lda, strideA,
                B_base, CUDA_R_16BF, ldb, strideB,
                &beta,
                result, CUDA_R_16BF, ldc, strideC,
                n_heads,    // batchCount
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        else {
            // NOTE: this no longer seems true and the issue may have been another bug in the code being reflected here?

            // Use cublasGemmEx per-head loop instead of cublasGemmStridedBatchedEx.
            // The strided batched API causes illegal memory access on SM120 with
            // the decode layout (seq_len=1, ldb=attn_n_heads*Lkv).
            for (int h = 0; h < n_heads; h++) {
                const void* A_h = (const char*)A_base + (long long)h * strideA * sizeof(__nv_bfloat16);
                const void* B_h = (const char*)B_base + (long long)h * strideB * sizeof(__nv_bfloat16);
                void* C_h = (char*)result + (long long)h * strideC * sizeof(__nv_bfloat16);
                cublasGemmEx(CUBLAS(ctx),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    V, BS, Lkv,
                    &alpha,
                    A_h, CUDA_R_16BF, lda,
                    B_h, CUDA_R_16BF, ldb,
                    &beta,
                    C_h, CUDA_R_16BF, ldc,
                    CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
        }
        return;
    }

    // Fallback: batch > 1 && seq_len > 1 (unit tests only).
    // v2 kernel for production dims, v1 for everything else.
    {
        int total_rows = batch * n_heads * seq_len;

        if (kv_lora_rank == 512 && v_head_dim == 256) {
            constexpr int KV_LR = 512;
            constexpr int VHD = 256;
            constexpr int TILE_K = 16;
            constexpr int BDX_V2 = 128;
            constexpr size_t attn_bytes = KV_LR * sizeof(float);
            constexpr size_t attn_bytes_aligned = (attn_bytes + 15) & ~size_t(15);
            constexpr size_t vproj_bytes = 2 * TILE_K * VHD * sizeof(__nv_bfloat16);
            size_t shmem_size = attn_bytes_aligned + vproj_bytes;
            mla_v_expand_kernel_v2<KV_LR, VHD, BDX_V2, TILE_K>
                <<<total_rows, BDX_V2, shmem_size, GLM_STREAM(ctx)>>>(
                    (__nv_bfloat16*)result, (const __nv_bfloat16*)attn_out,
                    (const __nv_bfloat16*)v_proj,
                    n_heads, seq_len, batch,
                    attn_n_heads, head_offset, v_proj_head_offset, total_rows);
            return;
        }

        int threads_per_row = compute_block_size(v_head_dim / 2, true);
        int rows_per_block = max(1, 256 / threads_per_row);
        int block_size = rows_per_block * threads_per_row;
        int grid = (total_rows + rows_per_block - 1) / rows_per_block;
        size_t shmem_size = (size_t)rows_per_block * kv_lora_rank * sizeof(float);
        auto launch_v1 = [&]<int KV_LR, int RPB>() {
            mla_v_expand_kernel<KV_LR, 256, RPB><<<grid, block_size, shmem_size, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)result, (const __nv_bfloat16*)attn_out,
                (const __nv_bfloat16*)v_proj,
                kv_lora_rank, v_head_dim, n_heads, seq_len, batch,
                attn_n_heads, head_offset, v_proj_head_offset, total_rows);
        };
        if (kv_lora_rank == 512) {
            if (rows_per_block == 8) launch_v1.template operator()<512, 8>();
            else if (rows_per_block == 4) launch_v1.template operator()<512, 4>();
            else if (rows_per_block == 2) launch_v1.template operator()<512, 2>();
            else launch_v1.template operator()<512, 1>();
        } else if (kv_lora_rank == 128) {
            if (rows_per_block == 8) launch_v1.template operator()<128, 8>();
            else if (rows_per_block == 4) launch_v1.template operator()<128, 4>();
            else if (rows_per_block == 2) launch_v1.template operator()<128, 2>();
            else launch_v1.template operator()<128, 1>();
        } else {
            if (rows_per_block >= 8) launch_v1.template operator()<0, 8>();
            else if (rows_per_block >= 4) launch_v1.template operator()<0, 4>();
            else if (rows_per_block == 2) launch_v1.template operator()<0, 2>();
            else launch_v1.template operator()<0, 1>();
        }
    }
}

// ---------------------------------------------------------------------------
// SiLU + Mul kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) silu_and_mul_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int total
) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    if (idx + 1 < total) {
        float2 gv = load_bf16x2(gate + idx);
        float2 uv = load_bf16x2(up + idx);
        store_bf16x2(out + idx, silu_f(gv.x) * uv.x, silu_f(gv.y) * uv.y);
    } else if (idx < total) {
        float g = __bfloat162float(gate[idx]);
        float u = __bfloat162float(up[idx]);
        out[idx] = __float2bfloat16(silu_f(g) * u);
    }
}

void glm_silu_and_mul(GlmCtx* ctx, void* out, const void* gate,
                      const void* up, int intermediate, int batch) {
    cudaSetDevice(ctx->device_id);
    int total = batch * intermediate;
    int block_size = 256;
    int grid = (total + 2 * block_size - 1) / (2 * block_size);
    silu_and_mul_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up, total);
}



// ---------------------------------------------------------------------------
// LayerNorm kernel (with bias)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) layernorm_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float eps,
    int dim
) {
    int row = blockIdx.x;
    const __nv_bfloat16* x = input + row * dim;
    __nv_bfloat16* o = out + row * dim;

    extern __shared__ float sdata[];

    float mean = 0.0f;
    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float2 v = load_bf16x2(x + i);
        mean += v.x + v.y;
    }
    if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
        mean += __bfloat162float(x[dim - 1]);
    }
    mean = block_reduce_sum(mean, sdata, threadIdx.x) / dim;

    float var = 0.0f;
    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float2 v = load_bf16x2(x + i);
        float d0 = v.x - mean, d1 = v.y - mean;
        var += d0 * d0 + d1 * d1;
    }
    if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
        float d = __bfloat162float(x[dim - 1]) - mean;
        var += d * d;
    }
    float inv_std = rsqrtf(block_reduce_sum(var, sdata, threadIdx.x) / dim + eps);

    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float2 xv = load_bf16x2(x + i);
        float2 wv = load_bf16x2(weight + i);
        float b0 = bias ? __bfloat162float(bias[i]) : 0.0f;
        float b1 = bias ? __bfloat162float(bias[i + 1]) : 0.0f;
        store_bf16x2(o + i, wv.x * (xv.x - mean) * inv_std + b0,
                              wv.y * (xv.y - mean) * inv_std + b1);
    }
    if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
        float xi = __bfloat162float(x[dim - 1]);
        float wi = __bfloat162float(weight[dim - 1]);
        float bi = bias ? __bfloat162float(bias[dim - 1]) : 0.0f;
        o[dim - 1] = __float2bfloat16(wi * (xi - mean) * inv_std + bi);
    }
}

void glm_layernorm(GlmCtx* ctx, void* out, const void* input,
                   const void* weight, const void* bias, float eps, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int block_size = compute_block_size(dim);
    size_t shared_mem = (block_size / 32) * sizeof(float);
    layernorm_kernel<<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)weight, (const __nv_bfloat16*)bias,
        eps, dim);
}

// ---------------------------------------------------------------------------
// Element-wise unary kernel (templated)
// F: float -> float
// ---------------------------------------------------------------------------

template<auto F>
__global__ void __launch_bounds__(256, 4) ew_unary_kernel(__nv_bfloat16* out, const __nv_bfloat16* input, int n) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    if (idx + 1 < n) {
        float2 v = load_bf16x2(input + idx);
        store_bf16x2(out + idx, F(v.x), F(v.y));
    } else if (idx < n) {
        out[idx] = __float2bfloat16(F(__bfloat162float(input[idx])));
    }
}

// ---------------------------------------------------------------------------
// Element-wise binary 2D kernel (templated)
// F: (float, float) -> float
// out[r, c] = F(a[r * a_stride + c], b[r * b_stride + c])
// out: [rows * dim] contiguous
// a:   row stride = a_stride (a_stride == dim for contiguous, 0 for broadcast)
// b:   row stride = b_stride (b_stride == dim for contiguous, 0 for broadcast)
// ---------------------------------------------------------------------------

template<typename F>
__global__ void __launch_bounds__(256, 4) ew_binary_2d_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    int dim, int rows, int a_stride, int b_stride
) {
    int total = rows * dim;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    if (idx + 1 < total) {
        int r = idx / dim;
        int c = idx % dim;
        __nv_bfloat162 av = *reinterpret_cast<const __nv_bfloat162*>(a + r * a_stride + c);
        __nv_bfloat162 bv = *reinterpret_cast<const __nv_bfloat162*>(b + r * b_stride + c);
        *reinterpret_cast<__nv_bfloat162*>(out + idx) = F{}(av, bv);
    } else if (idx < total) {
        int r = idx / dim;
        int c = idx % dim;
        out[idx] = F{}(a[r * a_stride + c], b[r * b_stride + c]);
    }
}

// ---------------------------------------------------------------------------
// Element-wise unary 2D kernel (templated) — pitched input
// Reads from a 2D pitched view of input: element(r, c) = in[r * pitch + col_offset + c]
// F: (float, float) -> float, receives (out_val, in_val)
// out: [rows * cols] contiguous
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float sigmoid_mul_f(float val, float gate) {
    return val * 0.5f * (fast_tanh(0.5f * gate) + 1.0f);
}

template<auto F>
__global__ void __launch_bounds__(256, 4) ew_unary_2d_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int rows, int cols, int pitch, int col_offset
) {
    int total = rows * cols;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    if (idx + 1 < total) {
        int r0 = idx / cols, c0 = idx % cols;
        int r1 = (idx + 1) / cols, c1 = (idx + 1) % cols;
        float v0 = __bfloat162float(out[idx]);
        float v1 = __bfloat162float(out[idx + 1]);
        float g0 = __bfloat162float(in[r0 * pitch + col_offset + c0]);
        float g1 = __bfloat162float(in[r1 * pitch + col_offset + c1]);
        store_bf16x2(out + idx, F(v0, g0), F(v1, g1));
    } else if (idx < total) {
        int r = idx / cols, c = idx % cols;
        float v = __bfloat162float(out[idx]);
        float g = __bfloat162float(in[r * pitch + col_offset + c]);
        out[idx] = __float2bfloat16(F(v, g));
    }
}

void glm_gate_sigmoid_mul(
    GlmCtx* ctx, void* attn_out, const void* gate_interleaved,
    int batch_seq, int num_heads, int head_dim
) {
    cudaSetDevice(ctx->device_id);
    int rows = batch_seq * num_heads;
    int cols = head_dim;
    int pitch = head_dim * 2;
    int total = rows * cols;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    ew_unary_2d_kernel<sigmoid_mul_f><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)attn_out,
        (const __nv_bfloat16*)gate_interleaved,
        rows, cols, pitch, head_dim);
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

void glm_relu(GlmCtx* ctx, void* out, const void* input, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    ew_unary_kernel<relu_f><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input, n);
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

void glm_sigmoid(GlmCtx* ctx, void* out, const void* input, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    ew_unary_kernel<sigmoid_f><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input, n);
}

// ---------------------------------------------------------------------------
// Softmax kernel (row-wise, with optional mask)
// input: [batch, dim], mask: [batch, dim] or NULL, out: [batch, dim]
// mask values: 0.0 = keep, -inf = masked out
// UseCache: if true, cache input values in shared memory (faster for dim <= 48KB/block)
// ---------------------------------------------------------------------------

template<bool UseCache>
__global__ void __launch_bounds__(256, 4) softmax_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* mask,
    int dim
) {
    int row = blockIdx.x;
    const __nv_bfloat16* x = input + row * dim;
    const __nv_bfloat16* m = mask ? mask + row * dim : nullptr;
    __nv_bfloat16* o = out + row * dim;

    extern __shared__ float sdata[];
    float* s_cache = UseCache ? sdata + blockDim.x : nullptr;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        if (m) val += __bfloat162float(m[i]);
        if constexpr (UseCache) s_cache[i] = val;
        if (val > max_val) max_val = val;
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    block_reduce_max(sdata, threadIdx.x);
    max_val = sdata[0];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val;
        if constexpr (UseCache) {
            val = s_cache[i] - max_val;
        } else {
            val = __bfloat162float(x[i]);
            if (m) val += __bfloat162float(m[i]);
            val -= max_val;
        }
        sum += __expf(val);
    }
    float inv_sum = 1.0f / block_reduce_sum(sum, sdata, threadIdx.x);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val;
        if constexpr (UseCache) {
            val = expf(s_cache[i] - max_val) * inv_sum;
        } else {
            val = __bfloat162float(x[i]);
            if (m) val += __bfloat162float(m[i]);
            val = expf(val - max_val) * inv_sum;
        }
        o[i] = __float2bfloat16(val);
    }
}

void glm_softmax(GlmCtx* ctx, void* out, const void* input,
                 const void* mask, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int block_size = compute_block_size(dim);
    size_t shared_mem = block_size * sizeof(float);
    const __nv_bfloat16* mask_ptr = mask ? (const __nv_bfloat16*)mask : nullptr;
    if ((dim + block_size) * sizeof(float) <= 48 * 1024) {
        shared_mem = (dim + block_size) * sizeof(float);
        softmax_kernel<true><<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input, mask_ptr, dim);
    } else {
        softmax_kernel<false><<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input, mask_ptr, dim);
    }
}

// ---------------------------------------------------------------------------
// Causal mask kernel
// Fills upper triangle with -inf: out[i][j] = (j > offset + i) ? -inf : 0
// out: [rows, cols] BF16
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) causal_mask_kernel(
    __nv_bfloat16* out,
    int rows,
    int cols,
    int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        float val = (col > offset + row) ? -INFINITY : 0.0f;
        out[idx] = __float2bfloat16(val);
    }
}

void glm_causal_mask(GlmCtx* ctx, void* out, int seq_len) {
    cudaSetDevice(ctx->device_id);
    int total = seq_len * seq_len;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    causal_mask_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, seq_len, seq_len, 0);
}

// ---------------------------------------------------------------------------
// Fill kernel
// ---------------------------------------------------------------------------

template <typename T>
__global__ void __launch_bounds__(256, 4) fill_kernel(T* out, T value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

template <typename T>
static cudaError_t launch_fill(GlmCtx* ctx, void* out, T value, int n) {
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    fill_kernel<T><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        static_cast<T*>(out), value, n);
    return cudaGetLastError();
}

cudaError_t glm_fill(GlmCtx* ctx, void* out, double value, int n, const char* dtype) {
    if (n < 0 || !dtype) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    cudaError_t err = cudaSetDevice(ctx->device_id);
    if (err != cudaSuccess) return err;
#define FILL_TYPE(NAME, TYPE) \
    if (strcmp(dtype, NAME) == 0) return launch_fill(ctx, out, static_cast<TYPE>(value), n)
    FILL_TYPE("BF16", __nv_bfloat16);
    FILL_TYPE("F16", __half);
    FILL_TYPE("F32", float);
    FILL_TYPE("F64", double);
    FILL_TYPE("I8", int8_t);
    FILL_TYPE("U8", uint8_t);
    FILL_TYPE("I16", int16_t);
    FILL_TYPE("U16", uint16_t);
    FILL_TYPE("I32", int32_t);
    FILL_TYPE("U32", uint32_t);
    FILL_TYPE("I64", int64_t);
    FILL_TYPE("U64", uint64_t);
    FILL_TYPE("BOOL", bool);
    FILL_TYPE("F8_E4M3", __nv_fp8_e4m3);
    FILL_TYPE("F8_E5M2", __nv_fp8_e5m2);
#undef FILL_TYPE
    if (strcmp(dtype, "C64") == 0) return launch_fill(ctx, out, make_float2(static_cast<float>(value), 0.0f), n);
    return cudaErrorInvalidValue;
}

// ---------------------------------------------------------------------------
// Gather kernel (along last dim, type-agnostic via ELEM_SIZE template)
// out[b, i] = input[b, indices[b, i]]
// ---------------------------------------------------------------------------

template<int ELEM_SIZE>
__global__ void __launch_bounds__(256, 4) gather_kernel(
    char* __restrict__ out,
    const char* __restrict__ input,
    const int* indices,
    int k,
    int in_dim,
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * k;
    if (idx < total) {
        int b = idx / k;
        int i = idx % k;
        int col = indices[b * k + i];
        const char* src = input + (size_t)(b * in_dim + col) * ELEM_SIZE;
        char* dst = out + (size_t)idx * ELEM_SIZE;
        if constexpr (ELEM_SIZE == 4) {
            *reinterpret_cast<int*>(dst) = *reinterpret_cast<const int*>(src);
        } else if constexpr (ELEM_SIZE == 2) {
            *reinterpret_cast<uint16_t*>(dst) = *reinterpret_cast<const uint16_t*>(src);
        } else {
            *reinterpret_cast<uint8_t*>(dst) = *reinterpret_cast<const uint8_t*>(src);
        }
    }
}

void glm_gather(GlmCtx* ctx, void* out, const void* input, const int* indices,
                int k, int in_dim, int batch, int elem_size) {
    cudaSetDevice(ctx->device_id);
    int total = batch * k;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    switch (elem_size) {
        case 1:
            gather_kernel<1><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
                (char*)out, (const char*)input, indices, k, in_dim, batch);
            break;
        case 2:
            gather_kernel<2><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
                (char*)out, (const char*)input, indices, k, in_dim, batch);
            break;
        case 4:
            gather_kernel<4><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
                (char*)out, (const char*)input, indices, k, in_dim, batch);
            break;
        default:
            fprintf(stderr, "glm_gather: unsupported elem_size %d\n", elem_size);
            break;
    }
}

// ---------------------------------------------------------------------------
// Scatter scalar kernel (along last dim)
// out[b, indices[b, i]] = value
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) scatter_scalar_kernel(
    __nv_bfloat16* out,
    const int* indices,
    float value,
    int k,
    int out_dim,
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * k;
    if (idx < total) {
        int b = idx / k;
        int i = idx % k;
        int col = indices[b * k + i];
        out[b * out_dim + col] = __float2bfloat16(value);
    }
}

void glm_scatter_scalar(GlmCtx* ctx, void* out, const int* indices, float value,
                        int k, int out_dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int total = batch * k;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    scatter_scalar_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, indices, value, k, out_dim, batch);
}

// ---------------------------------------------------------------------------
// Deinterleave kernel: rearranges all-gathered interleaved KV shards into
// sequential token order.
//
// After NCCL all-gather of interleaved compact KV, the buffer layout is:
//   [shard0: tok 0, ws, 2*ws, ...] [shard1: tok 1, 1+ws, ...] ... [shard_{ws-1}: ...]
//
// For output token i:
//   rank = i % ws
//   local_idx = i / ws
//   src_row = shard_offsets[rank] + local_idx
//
// Each block processes BLOCK_ROWS output rows. Threads cooperatively copy D
// elements per row using vectorized loads. When D is a multiple of 8, int4
// (16-byte) loads are used; otherwise falls back to scalar bf16 copies to
// avoid misalignment (row stride = D * 2 bytes must be 16-byte aligned for
// int4).
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256) deinterleave_kernel(
    char* __restrict__ out,
    const char* __restrict__ in,
    int world_size,
    int padded_total_len,
    const int32_t* __restrict__ page_indptr,
    const int32_t* __restrict__ kv_token_indptr,
    int batch_size,
    int page_size,
    int D
) {
    constexpr int BLOCK_ROWS = 4;
    const int total_len = kv_token_indptr[batch_size];
    const int chunk_len = padded_total_len / world_size;
    const int row_start = blockIdx.x * BLOCK_ROWS;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    #pragma unroll
    for (int r = 0; r < BLOCK_ROWS; r++) {
        const int out_row = row_start + r;
        if (out_row >= total_len) break;

        // Find which sequence this row belongs to
        int seq = 0;
        while (seq < batch_size - 1 && out_row >= kv_token_indptr[seq + 1]) seq++;

        const int seq_start = kv_token_indptr[seq];
        const int local_in_seq = out_row - seq_start;
        const int rank = local_in_seq % world_size;
        const int local_idx = local_in_seq / world_size;

        // Compute per-rank source offset: sum of count(rank, s') for s' < seq
        // count(rank, s') = ceil((seq_len[s'] - rank) / ws) if seq_len[s'] > rank else 0
        int src_offset = 0;
        for (int s = 0; s < seq; s++) {
            const int slen = kv_token_indptr[s + 1] - kv_token_indptr[s];
            if (slen > rank) {
                src_offset += (slen - rank + world_size - 1) / world_size;
            }
        }

        const int src_row = rank * chunk_len + src_offset + local_idx;

        const char* src = in + (size_t)src_row * D;
        char* dst = out + (size_t)out_row * D;

        if (D % 16 == 0) {
            constexpr int VEC = 16;
            const int vec_count = D / VEC;
            for (int j = tid; j < vec_count; j += block_size) {
                *reinterpret_cast<int4*>(dst + j * VEC) =
                    *reinterpret_cast<const int4*>(src + j * VEC);
            }
        } else {
            for (int j = tid; j < D; j += block_size) {
                dst[j] = src[j];
            }
        }
    }
}

void glm_deinterleave(GlmCtx* ctx, void* out, const void* in,
                      int world_size, int max_total_len,
                      const int32_t* page_indptr,
                      const int32_t* kv_token_indptr,
                      int batch_size, int page_size, int D) {
    cudaSetDevice(ctx->device_id);
    constexpr int BLOCK_ROWS = 4;
    constexpr int BLOCK_SIZE = 256;
    int grid = (max_total_len + BLOCK_ROWS - 1) / BLOCK_ROWS;
    deinterleave_kernel<<<grid, BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        (char*)out, (const char*)in,
        world_size, max_total_len,
        page_indptr, kv_token_indptr,
        batch_size, page_size, D);
}

// ---------------------------------------------------------------------------
// Gather pages kernel: copies KV data from scattered pages into a contiguous
// buffer. Handles ALL sequences in one launch using device-resident index
// tensors — no host work, graph-capturable.
//
// in: [maxPages, pageSize, D] — paged KV data (ckvData or kpeData)
// out: [totalKvLen, D] — contiguous tokens across all sequences (packed)
//
// page_indices[0..numPages-1]: flat page IDs for all sequences
// page_indptr[0..B]: page-level cumulative offsets (page_indptr[B] = numPages)
// last_page_len[0..B-1]: valid tokens in last page per sequence
//
// Output offsets computed on-device via prefix sum of per-sequence lengths:
//   seqLen[seq] = (pages_in_seq - 1) * page_size + last_page_len[seq]
//
// Each block copies one page. Finds its sequence via linear scan of page_indptr.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256) gather_pages_kernel(
    char* __restrict__ out,
    const char* __restrict__ in,
    const int32_t* __restrict__ page_indices,
    const int32_t* __restrict__ page_indptr,
    const int32_t* __restrict__ last_page_len,
    int batch_size,
    int page_size,
    int D
) {
    const int num_pages = page_indptr[batch_size];
    const int gp = blockIdx.x;
    if (gp >= num_pages) return;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Find which sequence this page belongs to (linear scan, B is small)
    int seq = 0;
    while (seq < batch_size - 1 && gp >= page_indptr[seq + 1]) seq++;

    const int local_page = gp - page_indptr[seq];
    const int pages_in_seq = page_indptr[seq + 1] - page_indptr[seq];
    const bool is_last = (local_page == pages_in_seq - 1);
    const int copy_len = is_last ? min(last_page_len[seq], page_size) : page_size;

    // Compute output offset: prefix sum of preceding sequences' lengths + local offset
    int out_offset = local_page * page_size;
    for (int s = 0; s < seq; s++) {
        const int ps = page_indptr[s + 1] - page_indptr[s];
        out_offset += (ps - 1) * page_size + min(last_page_len[s], page_size);
    }

    const int32_t src_page = page_indices[gp];

    const int total_bytes = copy_len * D;
    const char* src = in + (size_t)src_page * page_size * D;
    char* dst = out + (size_t)out_offset * D;

    if (D % 16 == 0) {
        constexpr int VEC = 16;
        const int vec_count = total_bytes / VEC;
        for (int j = tid; j < vec_count; j += block_size) {
            *reinterpret_cast<int4*>(dst + j * VEC) =
                *reinterpret_cast<const int4*>(src + j * VEC);
        }
    } else {
        for (int j = tid; j < total_bytes; j += block_size) {
            dst[j] = src[j];
        }
    }
}

void glm_gather_pages(GlmCtx* ctx, void* out, const void* in,
                      const int32_t* page_indices,
                      const int32_t* page_indptr,
                      const int32_t* last_page_len,
                      int max_pages, int batch_size,
                      int page_size, int D) {
    cudaSetDevice(ctx->device_id);
    constexpr int BLOCK_SIZE = 256;
    int grid = max_pages;
    gather_pages_kernel<<<grid, BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        (char*)out, (const char*)in,
        page_indices, page_indptr, last_page_len,
        batch_size, page_size, D);
}

// ---------------------------------------------------------------------------
// Cat last dim kernel
// out[i, :a_dim] = a[i, :], out[i, a_dim:] = b[i, :]
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) cat_last_dim_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    int a_last_dim,
    int b_last_dim,
    int outer
) {
    int out_dim = a_last_dim + b_last_dim;
    int total = outer * out_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int row = idx / out_dim;
        int col = idx % out_dim;
        if (col < a_last_dim) {
            out[idx] = a[row * a_last_dim + col];
        } else {
            out[idx] = b[row * b_last_dim + (col - a_last_dim)];
        }
    }
}

void glm_cat_last_dim(GlmCtx* ctx, void* out, const void* a, const void* b,
                      int a_last_dim, int b_last_dim, int outer) {
    cudaSetDevice(ctx->device_id);
    int total = outer * (a_last_dim + b_last_dim);
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    cat_last_dim_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)a,
        (const __nv_bfloat16*)b, a_last_dim, b_last_dim, outer);
}

// ---------------------------------------------------------------------------
// Masked fill kernel
// out[i] = (mask[i] != 0) ? value : input[i]
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) masked_fill_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* mask,
    float value,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float m = __bfloat162float(mask[idx]);
        if (m != 0.0f) {
            out[idx] = __float2bfloat16(value);
        } else {
            out[idx] = input[idx];
        }
    }
}

void glm_masked_fill(GlmCtx* ctx, void* out, const void* input, const void* mask,
                     float value, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    masked_fill_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)mask, value, n);
}

// ---------------------------------------------------------------------------
// Index add kernel (along first dim)
// out[indices[i], d] += values[i, d]  for each i, d
// NOTE: indices must be unique within a call for correctness.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) index_add_kernel(
    __nv_bfloat16* out,
    const int* indices,
    const __nv_bfloat16* values,
    int n_indices,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_indices * dim;
    if (idx < total) {
        int i = idx / dim;
        int d = idx % dim;
        int row = indices[i];
        float out_val = __bfloat162float(out[row * dim + d]);
        float add_val = __bfloat162float(values[idx]);
        out[row * dim + d] = __float2bfloat16(out_val + add_val);
    }
}

void glm_index_add(GlmCtx* ctx, void* out, const int* indices, const void* values,
                   int n_indices, int dim) {
    cudaSetDevice(ctx->device_id);
    int total = n_indices * dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    index_add_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, indices, (const __nv_bfloat16*)values,
        n_indices, dim);
}

// ---------------------------------------------------------------------------
// Rotary embedding kernel
// Computes cos/sin from inv_freq and position_ids
// cos_out, sin_out: [batch, seq_len, dim]  where dim = dim_half * 2
// inv_freq: [dim_half], position_ids: [batch, seq_len] (int32)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) rotary_embedding_kernel(
    __nv_bfloat16* cos_out,
    __nv_bfloat16* sin_out,
    const __nv_bfloat16* inv_freq,
    const int* position_ids,
    int dim_half,
    int seq_len,
    int batch
) {
    int dim = dim_half * 2;
    int total = batch * seq_len * dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int d = idx % dim;
        int s = (idx / dim) % seq_len;
        int b = idx / (seq_len * dim);
        float freq = __bfloat162float(inv_freq[d % dim_half]) *
                     (float)position_ids[b * seq_len + s];
        float cos_val, sin_val;
        sincosf(freq, &sin_val, &cos_val);
        cos_out[idx] = __float2bfloat16(cos_val);
        sin_out[idx] = __float2bfloat16(sin_val);
    }
}

void glm_rotary_embedding(GlmCtx* ctx, void* cos_out, void* sin_out,
                          const void* inv_freq, const int* position_ids,
                          int dim_half, int batch, int seq_len) {
    cudaSetDevice(ctx->device_id);
    int dim = dim_half * 2;
    int total = batch * seq_len * dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    rotary_embedding_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)cos_out, (__nv_bfloat16*)sin_out,
        (const __nv_bfloat16*)inv_freq, position_ids,
        dim_half, seq_len, batch);
}

// ---------------------------------------------------------------------------
// Apply rotary position embedding kernel
// out = x * cos + rotate_half(x) * sin
// rotate_half(x)[..., d] = (d < half) ? -x[..., d+half] : x[..., d-half]
//
// unsqueeze_dim=1: x is [batch, n_heads, seq_len, in_stride] (head-major)
// unsqueeze_dim=2: x is [batch, seq_len, n_heads, in_stride] (token-major)
// cos, sin are [batch, seq_len, rope_dim] (broadcast over heads)
// rope_dim <= head_dim: only first rope_dim dims get RoPE, rest pass through
// in_stride >= head_dim: per-head window stride in x. When in_stride >
// head_dim, the first head_dim dims of each head window are gathered into a
// packed [..., n_heads, head_dim] output (stride compaction).
// When head_dim == rope_dim and in_stride == head_dim, this reduces to the
// original behavior.
// ---------------------------------------------------------------------------

template <bool kInterleaved, int kHeadDim = 0, int kRopeDim = 0, int kUnsqueezeDim = 0>
__global__ void __launch_bounds__(256, 4) apply_rotary_pos_emb_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* cos_emb,
    const __nv_bfloat16* sin_emb,
    int rope_dim,
    int head_dim,
    int seq_len,
    int n_heads,
    int batch,
    int unsqueeze_dim,
    int in_stride
) {
    if constexpr (kHeadDim > 0) head_dim = kHeadDim;
    if constexpr (kHeadDim > 0) rope_dim = kRopeDim;
    if constexpr (kUnsqueezeDim > 0) unsqueeze_dim = kUnsqueezeDim;

    int d, b, h, s;
    if constexpr (kHeadDim > 0) {
        // Common GLM shapes: preserve ropeTranspose's one token/head per
        // block, including its sequence-first block order. Small decode
        // workloads otherwise collapse into too few 256-thread blocks.
        d = threadIdx.x;
        s = blockIdx.x % seq_len;
        h = (blockIdx.x / seq_len) % n_heads;
        b = blockIdx.x / (seq_len * n_heads);
    } else {
        int total = batch * n_heads * seq_len * head_dim;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) return;

        d = idx % head_dim;
        if (unsqueeze_dim == 1) {
            int rest = idx / head_dim;
            s = rest % seq_len;
            h = (rest / seq_len) % n_heads;
            b = rest / (n_heads * seq_len);
        } else {
            int rest = idx / head_dim;
            h = rest % n_heads;
            s = (rest / n_heads) % seq_len;
            b = rest / (n_heads * seq_len);
        }
    }

    int x_idx, out_idx;
    if (unsqueeze_dim == 1) {
        x_idx = ((b * n_heads + h) * seq_len + s) * in_stride + d;
        out_idx = ((b * n_heads + h) * seq_len + s) * head_dim + d;
    } else {
        x_idx = ((b * seq_len + s) * n_heads + h) * in_stride + d;
        out_idx = ((b * seq_len + s) * n_heads + h) * head_dim + d;
    }

    if (d < rope_dim) {
        int cos_base = (b * seq_len + s) * rope_dim;
        float x_val = __bfloat162float(x[x_idx]);

        if constexpr (kInterleaved) {
            int cos_idx = cos_base + (d >> 1);
            float cos_val = __bfloat162float(cos_emb[cos_idx]);
            float sin_val = __bfloat162float(sin_emb[cos_idx]);
            int paired_idx = (d % 2 == 0) ? x_idx + 1 : x_idx - 1;
            float paired_val = __bfloat162float(x[paired_idx]);
            out[out_idx] = __float2bfloat16(rope_rotate_interleaved(x_val, paired_val, cos_val, sin_val, d));
        } else {
            int half = rope_dim / 2;
            int cos_idx = cos_base + d;
            float cos_val = __bfloat162float(cos_emb[cos_idx]);
            float sin_val = __bfloat162float(sin_emb[cos_idx]);
            int paired_idx = (d < half) ? x_idx + half : x_idx - half;
            float paired_val = __bfloat162float(x[paired_idx]);
            out[out_idx] = __float2bfloat16(rope_rotate_neox(x_val, paired_val, cos_val, sin_val, d, half));
        }
    } else {
        out[out_idx] = x[x_idx];
    }
}

template <bool kInterleaved>
static void launch_apply_rotary_pos_emb(GlmCtx* ctx, void* out, const void* x,
                                      const void* cos, const void* sin,
                                      int rope_dim, int head_dim, int n_heads, int seq_len,
                                      int batch, int unsqueeze_dim, int in_stride) {
    const int rows = batch * n_heads * seq_len;
    if (unsqueeze_dim == 2 && rope_dim == 64) {
        if (head_dim == 64) {
            apply_rotary_pos_emb_kernel<kInterleaved, 64, 64, 2><<<rows, 64, 0, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
                (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin,
                rope_dim, head_dim, seq_len, n_heads, batch, unsqueeze_dim, in_stride);
            return;
        }
        if (head_dim == 128) {
            apply_rotary_pos_emb_kernel<kInterleaved, 128, 64, 2><<<rows, 128, 0, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
                (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin,
                rope_dim, head_dim, seq_len, n_heads, batch, unsqueeze_dim, in_stride);
            return;
        }
    }
    const int block_size = 256;
    const int grid = (rows * head_dim + block_size - 1) / block_size;
    apply_rotary_pos_emb_kernel<kInterleaved><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin,
        rope_dim, head_dim, seq_len, n_heads, batch, unsqueeze_dim, in_stride);
}

void glm_apply_rotary_pos_emb(GlmCtx* ctx, void* out, const void* x,
                               const void* cos, const void* sin,
                               int rope_dim, int head_dim, int n_heads, int seq_len,
                               int batch, int unsqueeze_dim, int in_stride, bool interleaved) {
    cudaSetDevice(ctx->device_id);
    if (interleaved) {
        launch_apply_rotary_pos_emb<true>(ctx, out, x, cos, sin,
            rope_dim, head_dim, n_heads, seq_len, batch, unsqueeze_dim, in_stride);
    } else {
        launch_apply_rotary_pos_emb<false>(ctx, out, x, cos, sin,
            rope_dim, head_dim, n_heads, seq_len, batch, unsqueeze_dim, in_stride);
    }
}

// ---------------------------------------------------------------------------
// TopK kernel - two-phase register + warp shuffle approach
// Each thread streams its elements, maintaining top-K in registers.
// Then warp-level bitonic merge via shuffles reduces to warp top-K.
// Finally cross-warp reduction via shared memory.
// Works for any dim (no smem proportional to dim).
// Template specialized for K=1,2,4,8.
// ---------------------------------------------------------------------------

template<int K>
__device__ inline void insert_topk(float top_vals[K], int top_idxs[K], float val, int idx) {
    // Find insertion point: top_vals is sorted descending
    // If val <= top_vals[K-1], reject
    if (val <= top_vals[K - 1]) return;
    // Find where to insert
    int pos = K - 1;
    while (pos > 0 && val > top_vals[pos - 1]) {
        top_vals[pos] = top_vals[pos - 1];
        top_idxs[pos] = top_idxs[pos - 1];
        pos--;
    }
    top_vals[pos] = val;
    top_idxs[pos] = idx;
}

template<int K>
__device__ inline void merge_topk(float top_vals[K], int top_idxs[K],
                                   const float partner_vals[K], const int partner_idxs[K]) {
    // Merge two sorted descending lists of size K, keep top K
    // Use a temporary buffer
    float tmp_vals[K];
    int tmp_idxs[K];
    int i = 0, j = 0, m = 0;
    while (m < K && (i < K || j < K)) {
        float vi = (i < K) ? top_vals[i] : -INFINITY;
        float vj = (j < K) ? partner_vals[j] : -INFINITY;
        if (vi > vj || (vi == vj && (i < K) && ((j >= K) || top_idxs[i] < partner_idxs[j]))) {
            tmp_vals[m] = vi;
            tmp_idxs[m] = top_idxs[i];
            i++;
        } else {
            tmp_vals[m] = vj;
            tmp_idxs[m] = partner_idxs[j];
            j++;
        }
        m++;
    }
    for (int n = 0; n < K; n++) {
        top_vals[n] = tmp_vals[n];
        top_idxs[n] = tmp_idxs[n];
    }
}

template<int K>
__global__ void __launch_bounds__(256, 4) topk_kernel_v2(
    __nv_bfloat16* out_values,
    int* out_indices,
    const __nv_bfloat16* input,
    int dim,
    int batch,
    int offset
) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const __nv_bfloat16* row_in = input + row * dim;
    __nv_bfloat16* row_vals = out_values + row * K;
    int* row_idxs = out_indices + row * K;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    // Phase 1: Each thread streams its elements, maintaining top-K in registers
    float top_vals[K];
    int top_idxs[K];
    for (int j = 0; j < K; j++) {
        top_vals[j] = -INFINITY;
        top_idxs[j] = -1;
    }

    for (int i = tid * 2; i + 1 < dim; i += blockDim.x * 2) {
        float2 v = load_bf16x2(row_in + i);
        insert_topk<K>(top_vals, top_idxs, v.x, i + offset);
        insert_topk<K>(top_vals, top_idxs, v.y, i + 1 + offset);
    }
    if ((dim & 1) && tid == (dim / 2) % blockDim.x) {
        float val = __bfloat162float(row_in[dim - 1]);
        insert_topk<K>(top_vals, top_idxs, val, (dim - 1) + offset);
    }

    // Phase 2: Warp-level bitonic merge via shuffles
    // After log2(32)=5 stages, all lanes in a warp have the warp's top-K
    for (int stage = 0; stage < 5; stage++) {
        int partner_lane = lane ^ (1 << stage);
        float partner_vals[K];
        int partner_idxs[K];
        for (int j = 0; j < K; j++) {
            partner_vals[j] = __shfl_sync(0xffffffff, top_vals[j], partner_lane);
            partner_idxs[j] = __shfl_sync(0xffffffff, top_idxs[j], partner_lane);
        }
        merge_topk<K>(top_vals, top_idxs, partner_vals, partner_idxs);
    }

    // Phase 3: Cross-warp reduction via shared memory
    // Lane 0 of each warp writes its top-K to smem
    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int* s_idxs = reinterpret_cast<int*>(s_vals + num_warps * K);

    if (lane == 0) {
        for (int j = 0; j < K; j++) {
            s_vals[warp_id * K + j] = top_vals[j];
            s_idxs[warp_id * K + j] = top_idxs[j];
        }
    }
    __syncthreads();

    // Thread 0 merges all warps' top-K into final result
    if (tid == 0) {
        // Initialize with warp 0's candidates
        for (int j = 0; j < K; j++) {
            top_vals[j] = s_vals[j];
            top_idxs[j] = s_idxs[j];
        }
        // Merge remaining warps
        for (int w = 1; w < num_warps; w++) {
            float w_vals[K];
            int w_idxs[K];
            for (int j = 0; j < K; j++) {
                w_vals[j] = s_vals[w * K + j];
                w_idxs[j] = s_idxs[w * K + j];
            }
            merge_topk<K>(top_vals, top_idxs, w_vals, w_idxs);
        }
        // Write output
        for (int j = 0; j < K; j++) {
            row_vals[j] = __float2bfloat16(top_vals[j]);
            row_idxs[j] = top_idxs[j];
        }
    }
}

// ---------------------------------------------------------------------------
// TopK kernel - shared-memory parallel argmax for small dim (≤ 1024).
// Loads all elements into shared memory, then K passes of warp-shuffle argmax
// with cross-warp reduction. Each pass needs only 2 __syncthreads.
// Shared memory: dim * sizeof(float) + num_warps * (sizeof(float) + sizeof(int))
// ---------------------------------------------------------------------------

template<int K>
__global__ void __launch_bounds__(256, 4) topk_kernel_smem(
    __nv_bfloat16* out_values,
    int* out_indices,
    const __nv_bfloat16* input,
    int dim,
    int batch,
    int offset
) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const __nv_bfloat16* row_in = input + row * dim;
    __nv_bfloat16* row_vals = out_values + row * K;
    int* row_idxs = out_indices + row * K;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    float* w_vals = s_vals + dim;
    int* w_idxs = reinterpret_cast<int*>(w_vals + num_warps);

    for (int i = tid * 2; i + 1 < dim; i += blockDim.x * 2) {
        float2 v = load_bf16x2(row_in + i);
        s_vals[i] = v.x;
        s_vals[i + 1] = v.y;
    }
    if ((dim & 1) && tid == (dim / 2) % blockDim.x) {
        s_vals[dim - 1] = __bfloat162float(row_in[dim - 1]);
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        float my_val = -INFINITY;
        int my_idx = -1;
        for (int i = tid; i < dim; i += blockDim.x) {
            float v = s_vals[i];
            if (v > my_val || (v == my_val && i < my_idx)) {
                my_val = v;
                my_idx = i;
            }
        }

        for (int off = 16; off > 0; off >>= 1) {
            float pv = __shfl_down_sync(0xffffffff, my_val, off);
            int pi = __shfl_down_sync(0xffffffff, my_idx, off);
            if (pv > my_val || (pv == my_val && pi < my_idx)) {
                my_val = pv;
                my_idx = pi;
            }
        }

        if (lane == 0) {
            w_vals[warp_id] = my_val;
            w_idxs[warp_id] = my_idx;
        }
        __syncthreads();

        if (tid == 0) {
            float best_val = w_vals[0];
            int best_idx = w_idxs[0];
            for (int w = 1; w < num_warps; w++) {
                if (w_vals[w] > best_val || (w_vals[w] == best_val && w_idxs[w] < best_idx)) {
                    best_val = w_vals[w];
                    best_idx = w_idxs[w];
                }
            }
            row_vals[k] = __float2bfloat16(best_val);
            row_idxs[k] = best_idx + offset;
            s_vals[best_idx] = -INFINITY;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// TopK kernel (along last dim, unsorted) - legacy smem-based approach
// One block per row. Loads row into shared memory, finds top-k by
// k passes of parallel argmax.
// Shared memory: dim * (sizeof(float) + sizeof(int)) + blockDim * (sizeof(float) + sizeof(int))
// Only suitable for small dim (e.g., MoE routing with 256 experts)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) topk_kernel(
    __nv_bfloat16* out_values,
    int* out_indices,
    const __nv_bfloat16* input,
    int k,
    int dim,
    int batch,
    int offset
) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const __nv_bfloat16* row_in = input + row * dim;
    __nv_bfloat16* row_vals = out_values + row * k;
    int* row_idxs = out_indices + row * k;

    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int* s_idxs = reinterpret_cast<int*>(s_vals + dim);
    float* s_reduce = reinterpret_cast<float*>(s_idxs + dim);
    int* s_reduce_idx = reinterpret_cast<int*>(s_reduce + blockDim.x);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        s_vals[i] = __bfloat162float(row_in[i]);
        s_idxs[i] = i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    for (int j = 0; j < k; j++) {
        float my_max = -INFINITY;
        int my_idx = -1;
        for (int i = tid; i < dim; i += blockDim.x) {
            if (s_vals[i] > my_max || (s_vals[i] == my_max && i < my_idx)) {
                my_max = s_vals[i];
                my_idx = i;
            }
        }
        s_reduce[tid] = my_max;
        s_reduce_idx[tid] = my_idx;
        __syncthreads();

        block_reduce_max_idx(s_reduce, s_reduce_idx, tid);

        if (tid == 0) {
            int best = s_reduce_idx[0];
            if (best >= 0) {
                row_vals[j] = __float2bfloat16(s_vals[best]);
                row_idxs[j] = s_idxs[best] + offset;
                s_vals[best] = __int_as_float(0x7fc00000);
            } else {
                row_vals[j] = __float2bfloat16(-INFINITY);
                row_idxs[j] = 0;
            }
        }
        __syncthreads();
    }
}

// Router specialization: one warp owns all 256 scores in registers. Retain
// the generic selector's descending value / ascending index ordering.
__global__ void topk8_router_kernel(
    __nv_bfloat16* out_values, int* out_indices,
    const __nv_bfloat16* input, int offset
) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    float values[8];
#pragma unroll
    for (int j = 0; j < 8; j++)
        values[j] = __bfloat162float(input[(size_t)row * 256 + lane + j * 32]);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        float best = -INFINITY;
        int index = -1;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int candidate = lane + j * 32;
            if (values[j] > best || (values[j] == best && candidate < index)) {
                best = values[j];
                index = candidate;
            }
        }
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, best, delta);
            const int otherIndex = __shfl_down_sync(0xffffffff, index, delta);
            if (other > best || (other == best && otherIndex < index)) {
                best = other;
                index = otherIndex;
            }
        }
        const int winner = __shfl_sync(0xffffffff, index, 0);
        if (lane == 0) {
            out_values[(size_t)row * 8 + k] = __float2bfloat16(best);
            out_indices[(size_t)row * 8 + k] = winner + offset;
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
            if (lane + j * 32 == winner) values[j] = -INFINITY;
    }
}

// Fuse the GLM router's BF16 sigmoid -> biased selection -> unbiased weights.
// Keep both BF16 materialization points: rounding them away changes expert IDs.
__global__ void route_top8_kernel(
    __nv_bfloat16* out_weights, int* out_indices,
    const __nv_bfloat16* logits, const __nv_bfloat16* bias,
    float scale, bool normalize
) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    float scores[8], original[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int expert = lane + j * 32;
        original[j] = __bfloat162float(__float2bfloat16(
            sigmoid_f(__bfloat162float(logits[(size_t)row * 256 + expert]))));
        scores[j] = __bfloat162float(__float2bfloat16(
            original[j] + __bfloat162float(bias[expert])));
    }
    float selected = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        float best = -INFINITY;
        int index = -1;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int candidate = lane + j * 32;
            if (scores[j] > best || (scores[j] == best && candidate < index)) {
                best = scores[j];
                index = candidate;
            }
        }
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, best, delta);
            const int other_index = __shfl_down_sync(0xffffffff, index, delta);
            if (other > best || (other == best && other_index < index)) {
                best = other;
                index = other_index;
            }
        }
        const int winner = __shfl_sync(0xffffffff, index, 0);
        float winner_score = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if (lane + j * 32 == winner) {
                winner_score = original[j];
                scores[j] = -INFINITY;
            }
        }
        winner_score = __shfl_sync(0xffffffff, winner_score, winner & 31);
        if (lane == k) {
            out_indices[(size_t)row * 8 + k] = winner;
            selected = winner_score;
        }
    }
    // Same descending-stride addition tree as row_normalize_kernel for cols=8.
    float sum = fabsf(selected);
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    }
    sum = __shfl_sync(0xffffffff, sum, 0);
    const float multiplier = normalize ? (sum > 0.0f ? scale / sum : 0.0f) : scale;
    if (lane < 8) {
        out_weights[(size_t)row * 8 + lane] = __float2bfloat16(selected * multiplier);
    }
}

void glm_route_top8(GlmCtx* ctx, void* out_weights, int* out_indices,
                    const void* logits, const void* bias, int rows, float scale, bool normalize) {
    cudaSetDevice(ctx->device_id);
    route_top8_kernel<<<rows, 32, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out_weights, out_indices,
        (const __nv_bfloat16*)logits, (const __nv_bfloat16*)bias, scale, normalize);
}

void glm_topk(GlmCtx* ctx, void* out_values, int* out_indices,
              const void* input, int k, int dim, int batch, int offset) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = batch;

    if (k == 8 && dim == 256) {
        topk8_router_kernel<<<batch, 32, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, offset);
        return;
    }

    if (k <= 8 && dim <= 1024) {
        // Shared-memory parallel argmax: K passes of warp-shuffle reduction.
        // O(dim) shared memory, only 2 __syncthreads per pass.
        int num_warps = block_size / 32;
        size_t shared_mem = dim * sizeof(float) + num_warps * sizeof(float) + num_warps * sizeof(int);

        switch (k) {
            case 1: topk_kernel_smem<1><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 2: topk_kernel_smem<2><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 3: topk_kernel_smem<3><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 4: topk_kernel_smem<4><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 5: topk_kernel_smem<5><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 6: topk_kernel_smem<6><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 7: topk_kernel_smem<7><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 8: topk_kernel_smem<8><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
        }
    } else if (k <= 8) {
        // Two-phase register + warp shuffle kernel: O(1) shared memory, works for any dim
        int num_warps = block_size / 32;
        size_t shared_mem = num_warps * k * sizeof(float) + num_warps * k * sizeof(int);

        switch (k) {
            case 1: topk_kernel_v2<1><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 2: topk_kernel_v2<2><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 3: topk_kernel_v2<3><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 4: topk_kernel_v2<4><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 5: topk_kernel_v2<5><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 6: topk_kernel_v2<6><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 7: topk_kernel_v2<7><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
            case 8: topk_kernel_v2<8><<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
                (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset); break;
        }
    } else {
        // Legacy smem kernel: O(dim) shared memory, only for small dim
        size_t shared_mem = dim * sizeof(float) + dim * sizeof(int) +
                            block_size * sizeof(float) + block_size * sizeof(int);
        topk_kernel<<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out_values, out_indices,
            (const __nv_bfloat16*)input, k, dim, batch, offset);
    }
}

// ---------------------------------------------------------------------------
// Batched matrix multiply (contiguous row-major tensors)
// transA=0, transB=0: C[b] = alpha * A[b] @ B[b] + beta * C[b]
//   A: [batch, M, K],  B: [batch, K, N],  C: [batch, M, N]
// transA=0, transB=1: C[b] = alpha * A[b] @ B[b]^T + beta * C[b]
//   A: [batch, M, K],  B: [batch, N, K],  C: [batch, M, N]
// transA=1, transB=0: C[b] = alpha * A[b]^T @ B[b] + beta * C[b]
//   A: [batch, K, M],  B: [batch, K, N],  C: [batch, M, N]
// transA=1, transB=1: C[b] = alpha * A[b]^T @ B[b]^T + beta * C[b]
//   A: [batch, K, M],  B: [batch, N, K],  C: [batch, M, N]
// ---------------------------------------------------------------------------

void glm_bmm(GlmCtx* ctx, void* C, const void* A, const void* B,
             float alpha, float beta,
             int batch, int M, int N, int K, int transA, int transB, int tokenMajor) {
    cudaSetDevice(ctx->device_id);

    // cuBLAS computes C_cm = op(A_gemm) @ op(B_gemm) in column-major.
    // We derive the cuBLAS parameters from the row-major intent.
    //
    // Row-major result: C_rm = op(A_rm) @ op(B_rm)
    // Column-major result: C_cm = C_rm^T = op(B_rm)^T @ op(A_rm)^T
    //   = op(B_cm) @ op(A_cm)^T
    //
    // So: A_gemm = B (col-major), B_gemm = A (col-major)
    //     transa_gemm depends on transB, transb_gemm depends on transA

    cublasOperation_t transa_gemm, transb_gemm;
    int m_gemm, n_gemm, k_gemm;
    int lda_gemm, ldb_gemm, ldc_gemm;
    long long strideA_gemm, strideB_gemm, strideC;

    // Row-major shapes:
    //   A_rm: [K, M] if transA, else [M, K]  → A_cm: [M, K] or [K, M]
    //   B_rm: [N, K] if transB, else [K, N]  → B_cm: [K, N] or [N, K]
    //   C_rm: [M, N]                          → C_cm: [N, M]

    // A_gemm = B (col-major):
    if (transB) {
        // B_rm = [N, K], B_cm = [K, N]
        transa_gemm = CUBLAS_OP_T;  // op(B_cm) = B_cm^T = [N, K]
        lda_gemm = K;               // rows of B_cm
        strideA_gemm = (long long)N * K;
    } else {
        // B_rm = [K, N], B_cm = [N, K]
        transa_gemm = CUBLAS_OP_N;  // op(B_cm) = B_cm = [N, K]
        lda_gemm = N;               // rows of B_cm
        strideA_gemm = (long long)K * N;
    }

    // B_gemm = A (col-major):
    if (transA) {
        // A_rm = [K, M], A_cm = [M, K]
        transb_gemm = CUBLAS_OP_T;  // op(A_cm) = A_cm^T = [K, M]
        ldb_gemm = M;               // rows of A_cm
        strideB_gemm = (long long)K * M;
    } else {
        // A_rm = [M, K], A_cm = [K, M]
        transb_gemm = CUBLAS_OP_N;  // op(A_cm) = A_cm = [K, M]
        ldb_gemm = K;               // rows of A_cm
        strideB_gemm = (long long)M * K;
    }

    m_gemm = N;  // rows of op(A_gemm)
    n_gemm = M;  // cols of op(B_gemm)
    k_gemm = K;  // cols of op(A_gemm) = rows of op(B_gemm)
    ldc_gemm = N;
    strideC = (long long)M * N;

    if (tokenMajor) {
        // A=[M,batch,K], B=[batch,K,N], C=[M,batch,N]. Each head's
        // rows are separated by the other heads, while head bases are adjacent.
        ldb_gemm = batch * K;
        strideB_gemm = K;
        ldc_gemm = batch * N;
        strideC = N;
    }

    cublasGemmStridedBatchedEx(CUBLAS(ctx),
        transa_gemm, transb_gemm,
        m_gemm, n_gemm, k_gemm,
        &alpha,
        B, CUDA_R_16BF, lda_gemm, strideA_gemm,
        A, CUDA_R_16BF, ldb_gemm, strideB_gemm,
        &beta,
        C, CUDA_R_16BF, ldc_gemm, strideC,
        batch,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ---------------------------------------------------------------------------
// Scale kernel: out = input * scale
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) scale_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    if (idx + 1 < n) {
        float2 v = load_bf16x2(input + idx);
        store_bf16x2(out + idx, v.x * scale, v.y * scale);
    } else if (idx < n) {
        out[idx] = __float2bfloat16(__bfloat162float(input[idx]) * scale);
    }
}

void glm_scale(GlmCtx* ctx, void* out, const void* input, float scale, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    scale_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input, scale, n);
}

// ---------------------------------------------------------------------------
// Add: out = a + b
// ---------------------------------------------------------------------------

struct add_f {
    __device__ __forceinline__ __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const { return __hadd2(a, b); }
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const { return __hadd(a, b); }
};

void glm_add(GlmCtx* ctx, void* out, const void* a, const void* b, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + 2 * block_size - 1) / (2 * block_size);
    ew_binary_2d_kernel<add_f><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
        n, 1, n, n);
}

void glm_add_broadcast(GlmCtx* ctx, void* out, const void* a, const void* b, int dim, int rows) {
    cudaSetDevice(ctx->device_id);
    int total = rows * dim;
    int block_size = 256;
    int grid = (total + 2 * block_size - 1) / (2 * block_size);
    ew_binary_2d_kernel<add_f><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
        dim, rows, dim, 0);
}

// ---------------------------------------------------------------------------
// Row-scale-add: out[i, d] += scale[i] * input[i, d]
// out:     [rows, dim] BF16 (in-place accumulation)
// input:   [rows, dim] BF16
// scales:  [rows] BF16 (per-row scaling factor)
// ---------------------------------------------------------------------------
// Expand/repeat along dim 1 of a 4D tensor
// input:  [batch, dim1_in, seq_len, head_dim]
// output: [batch, dim1_out, seq_len, head_dim]
// out[b, h_out, s, d] = input[b, h_out * dim1_in / dim1_out, s, d]
// Requires dim1_out % dim1_in == 0
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) expand_dim1_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int dim1_out,
    int dim1_in,
    int seq_len,
    int head_dim,
    int batch,
    int head_stride,
    int expand_ratio
) {
    int total = batch * dim1_out * seq_len * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int d = idx % head_dim;
        int rest = idx / head_dim;
        int s = rest % seq_len;
        rest /= seq_len;
        int h_out = rest % dim1_out;
        int b = rest / dim1_out;

        int h_in = h_out / expand_ratio;
        int in_idx = (b * dim1_in + h_in) * head_stride + s * head_dim + d;
        out[idx] = input[in_idx];
    }
}

void glm_expand_dim1(GlmCtx* ctx, void* out, const void* input,
                     int dim1_out, int dim1_in, int seq_len, int head_dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int total = batch * dim1_out * seq_len * head_dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    expand_dim1_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        dim1_out, dim1_in, seq_len, head_dim, batch, seq_len * head_dim,
        dim1_out / dim1_in);
}

void glm_expand_dim1_strided(GlmCtx* ctx, void* out, const void* input,
                             int dim1_out, int dim1_in, int seq_len, int head_dim,
                             int batch, int head_stride) {
    cudaSetDevice(ctx->device_id);
    int total = batch * dim1_out * seq_len * head_dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    expand_dim1_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        dim1_out, dim1_in, seq_len, head_dim, batch, head_stride,
        dim1_out / dim1_in);
}

// ---------------------------------------------------------------------------
// Specialized transpose for {0,2,1,3}: swaps dims 1 and 2
// input:  [dim0, dim1, dim2, dim3]  output: [dim0, dim2, dim1, dim3]
// out[b, s, h, d] = in[b, h, s, d]
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) transpose_0213_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int dim0, int dim1, int dim2, int dim3
) {
    int total = dim0 * dim1 * dim2 * dim3;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int i3 = idx % dim3;
    int rest = idx / dim3;
    int i2 = rest % dim2;
    rest /= dim2;
    int i1 = rest % dim1;
    int i0 = rest / dim1;

    int out_idx = ((i0 * dim2 + i2) * dim1 + i1) * dim3 + i3;
    out[out_idx] = input[idx];
}

// ---------------------------------------------------------------------------
// Specialized transpose for {0,1,3,2}: swaps the last two dims (true 2D
// transpose per (i0,i1) slice). Unlike the naive scatter kernels above, a
// permutation that moves the fastest-varying input dim (dim3) to the
// second-to-last output position makes BOTH the naive read and write strides
// non-unit for one side -- e.g. coalesced reads but scattered writes with
// stride dim2, or vice versa. A tiled shared-memory transpose lets every
// thread do coalesced global reads AND coalesced global writes, paying for
// the transpose with a fast on-chip shared-memory shuffle instead.
// input:  [dim0, dim1, dim2, dim3]  output: [dim0, dim1, dim3, dim2]
// out[b, h, j, i] = in[b, h, i, j]
// ---------------------------------------------------------------------------

constexpr int TRANSPOSE_TILE_DIM = 32;
constexpr int TRANSPOSE_BLOCK_ROWS = 8;

__global__ void __launch_bounds__(TRANSPOSE_TILE_DIM * TRANSPOSE_BLOCK_ROWS)
transpose_swap_last2_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    int dim01, int dim2, int dim3
) {
    __shared__ __nv_bfloat16 tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int batch = blockIdx.z;
    const __nv_bfloat16* in_base = input + (size_t)batch * dim2 * dim3;
    __nv_bfloat16* out_base = out + (size_t)batch * dim3 * dim2;

    int col = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;  // index along dim3 (input row-contiguous)
    int row0 = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        int row = row0 + j;
        if (row < dim2 && col < dim3)
            tile[threadIdx.y + j][threadIdx.x] = in_base[(size_t)row * dim3 + col];
    }
    __syncthreads();

    // Transposed coordinates: output row runs along dim3, output col along dim2
    int out_col = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    int out_row0 = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        int out_row = out_row0 + j;
        if (out_row < dim3 && out_col < dim2)
            out_base[(size_t)out_row * dim2 + out_col] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ---------------------------------------------------------------------------
// General 4D transpose kernel
// input:  [dim0, dim1, dim2, dim3] with row-major layout
// output: [dim_perm0, dim_perm1, dim_perm2, dim_perm3]
// perm[0..3] specifies the output dimension order
// e.g., perm={0,2,1,3} swaps dims 1 and 2
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) transpose_4d_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int dim0, int dim1, int dim2, int dim3,
    int perm0, int perm1, int perm2, int perm3
) {
    int total = dim0 * dim1 * dim2 * dim3;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode input index to (i0, i1, i2, i3)
    int i3 = idx % dim3;
    int rest = idx / dim3;
    int i2 = rest % dim2;
    rest /= dim2;
    int i1 = rest % dim1;
    int i0 = rest / dim1;

    // Compute output dimensions
    int out_dims[4];
    out_dims[0] = (perm0 == 0) ? dim0 : (perm0 == 1) ? dim1 : (perm0 == 2) ? dim2 : dim3;
    out_dims[1] = (perm1 == 0) ? dim0 : (perm1 == 1) ? dim1 : (perm1 == 2) ? dim2 : dim3;
    out_dims[2] = (perm2 == 0) ? dim0 : (perm2 == 1) ? dim1 : (perm2 == 2) ? dim2 : dim3;
    out_dims[3] = (perm3 == 0) ? dim0 : (perm3 == 1) ? dim1 : (perm3 == 2) ? dim2 : dim3;

    // Map input indices to output indices via permutation
    int in_idx[4] = {i0, i1, i2, i3};
    int o0 = in_idx[perm0];
    int o1 = in_idx[perm1];
    int o2 = in_idx[perm2];
    int o3 = in_idx[perm3];

    int out_idx = ((o0 * out_dims[1] + o1) * out_dims[2] + o2) * out_dims[3] + o3;
    out[out_idx] = input[idx];
}

// Swap dimensions 0 and 1 while preserving each contiguous [dim2, dim3]
// row. This is the owner-merge layout conversion [source, query, K] ->
// [query, source, K], and supports any element size.
__global__ void __launch_bounds__(256, 4) transpose_swap_first2_bytes_kernel(
    uint8_t* __restrict__ out, const uint8_t* __restrict__ input,
    int dim0, int dim1, int row_bytes) {
    int row = blockIdx.x;
    int i0 = row / dim1;
    int i1 = row - i0 * dim1;
    const uint8_t* src = input + (int64_t)row * row_bytes;
    uint8_t* dst = out + (int64_t)(i1 * dim0 + i0) * row_bytes;

    constexpr int VEC = 16;
    if ((row_bytes & (VEC - 1)) == 0) {
        int vecs = row_bytes / VEC;
        for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
            *reinterpret_cast<int4*>(dst + (int64_t)i * VEC) =
                *reinterpret_cast<const int4*>(src + (int64_t)i * VEC);
        }
    } else {
        for (int i = threadIdx.x; i < row_bytes; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

static void transpose_4d_impl(GlmCtx* ctx, void* out, const void* input,
                              int dim0, int dim1, int dim2, int dim3,
                              int perm0, int perm1, int perm2, int perm3,
                              int elem_bytes) {
    cudaSetDevice(ctx->device_id);
    int total = dim0 * dim1 * dim2 * dim3;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    if (perm0 == 1 && perm1 == 0 && perm2 == 2 && perm3 == 3) {
        transpose_swap_first2_bytes_kernel<<<dim0 * dim1, block_size, 0, GLM_STREAM(ctx)>>>(
            (uint8_t*)out, (const uint8_t*)input,
            dim0, dim1, dim2 * dim3 * elem_bytes);
    } else if (elem_bytes != 2) {
        fprintf(stderr, "glm_transpose_4d_typed: only permutation (1,0,2,3) supports elem_bytes=%d\n", elem_bytes);
    } else if (perm0 == 0 && perm1 == 2 && perm2 == 1 && perm3 == 3) {
        transpose_0213_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
            dim0, dim1, dim2, dim3);
    } else if (perm0 == 0 && perm1 == 1 && perm2 == 3 && perm3 == 2) {
        ::dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
        ::dim3 grid3((dim3 + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
                     (dim2 + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
                     dim0 * dim1);
        transpose_swap_last2_kernel<<<grid3, block, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
            dim0 * dim1, dim2, dim3);
    } else {
        transpose_4d_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
            dim0, dim1, dim2, dim3, perm0, perm1, perm2, perm3);
    }
}

void glm_transpose_4d(GlmCtx* ctx, void* out, const void* input,
                       int dim0, int dim1, int dim2, int dim3,
                       int perm0, int perm1, int perm2, int perm3) {
    transpose_4d_impl(ctx, out, input, dim0, dim1, dim2, dim3,
                      perm0, perm1, perm2, perm3, sizeof(__nv_bfloat16));
}

void glm_transpose_4d_typed(GlmCtx* ctx, void* out, const void* input,
                            int dim0, int dim1, int dim2, int dim3,
                            int perm0, int perm1, int perm2, int perm3,
                            int elem_bytes) {
    transpose_4d_impl(ctx, out, input, dim0, dim1, dim2, dim3,
                      perm0, perm1, perm2, perm3, elem_bytes);
}

// ---------------------------------------------------------------------------
// Mul: out = a * b
// ---------------------------------------------------------------------------

struct mul_f {
    __device__ __forceinline__ __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const { return __hmul2(a, b); }
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const { return __hmul(a, b); }
};

void glm_mul(GlmCtx* ctx, void* out, const void* a, const void* b, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + 2 * block_size - 1) / (2 * block_size);
    ew_binary_2d_kernel<mul_f><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
        n, 1, n, n);
}

void glm_mul_broadcast(GlmCtx* ctx, void* out, const void* a, const void* b, int dim, int rows) {
    cudaSetDevice(ctx->device_id);
    int total = rows * dim;
    int block_size = 256;
    int grid = (total + 2 * block_size - 1) / (2 * block_size);
    ew_binary_2d_kernel<mul_f><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
        dim, rows, dim, 0);
}

// ---------------------------------------------------------------------------
// Reduce-sum along last dimension kernel
//   input:  [rows, cols]  BF16
//   output: [rows]        BF16
//   out[i] = sum(input[i, :])
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) reduce_sum_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int cols
) {
    int row = blockIdx.x;
    const __nv_bfloat16* row_ptr = input + row * cols;
    float sum = 0.0f;
    for (int i = threadIdx.x * 2; i + 1 < cols; i += blockDim.x * 2) {
        float2 v = load_bf16x2(row_ptr + i);
        sum += v.x + v.y;
    }
    if ((cols & 1) && threadIdx.x == (cols / 2) % blockDim.x) {
        sum += __bfloat162float(row_ptr[cols - 1]);
    }
    extern __shared__ float sdata[];
    float total = block_reduce_sum(sum, sdata, threadIdx.x);
    if (threadIdx.x == 0) {
        out[row] = __float2bfloat16(total);
    }
}

void glm_reduce_sum(GlmCtx* ctx, void* out, const void* input, int rows, int cols) {
    cudaSetDevice(ctx->device_id);
    int block_size = compute_block_size(cols);
    size_t shared_mem = (block_size / 32) * sizeof(float);
    reduce_sum_kernel<<<rows, block_size, shared_mem, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input, cols);
}

// ---------------------------------------------------------------------------
// Row-normalize kernel (L1-normalize rows and scale)
//   input:  [rows, cols]  BF16
//   output: [rows, cols]  BF16
//   if normalize: output[i,:] = input[i,:] / sum(|input[i,:]|) * scale
//   else:         output[i,:] = input[i,:] * scale
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) row_normalize_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    float scale,
    int rows, int cols, bool normalize
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* row_in = input + row * cols;
    __nv_bfloat16* row_out = out + row * cols;

    if (!normalize) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            row_out[c] = __float2bfloat16(__bfloat162float(row_in[c]) * scale);
        }
        return;
    }

    extern __shared__ float sdata[];
    float thread_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(row_in[c]);
        thread_sum += fabsf(v);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float row_sum = sdata[0];
    float inv_sum = (row_sum > 0.0f) ? (scale / row_sum) : 0.0f;

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        row_out[c] = __float2bfloat16(__bfloat162float(row_in[c]) * inv_sum);
    }
}

void glm_row_normalize(GlmCtx* ctx, void* out, const void* input,
                        float scale, int rows, int cols, bool normalize) {
    cudaSetDevice(ctx->device_id);
    int block_size = compute_block_size(cols);
    size_t shared_mem = block_size * sizeof(float);
    row_normalize_kernel<<<rows, block_size, shared_mem, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        scale, rows, cols, normalize);
}

// ---------------------------------------------------------------------------
// Group-mask multiply kernel
//   scores:     [batch, num_experts]       BF16 (in-place)
//   group_mask: [batch, n_group]           BF16
//   For each batch b and expert e:
//     group = e / experts_per_group
//     scores[b, e] *= group_mask[b, group]
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) group_mask_mul_kernel(
    __nv_bfloat16* scores,
    const __nv_bfloat16* group_mask,
    int num_experts, int experts_per_group, int n_group, int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_experts;
    if (idx >= total) return;
    int b = idx / num_experts;
    int e = idx % num_experts;
    int g = e / experts_per_group;
    float mask_val = __bfloat162float(group_mask[b * n_group + g]);
    float score_val = __bfloat162float(scores[idx]);
    scores[idx] = __float2bfloat16(score_val * mask_val);
}

void glm_group_mask_mul(GlmCtx* ctx, void* scores, const void* group_mask,
                         int num_experts, int experts_per_group, int n_group, int batch) {
    cudaSetDevice(ctx->device_id);
    int total = batch * num_experts;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    group_mask_mul_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)scores, (const __nv_bfloat16*)group_mask,
        num_experts, experts_per_group, n_group, batch);
}

// ---------------------------------------------------------------------------
// Index-select kernel (gather rows by index)
//   src:      [src_rows, dim]  BF16
//   indices:  [k]              int32
//   out:      [k, dim]         BF16
//   out[i, :] = src[indices[i], :]
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) index_select_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* src,
    const int* indices,
    int dim,
    int k,
    int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * dim) return;
    int i = idx / dim;
    int d = idx % dim;
    int src_row = indices[i] + offset;
    out[idx] = src[src_row * dim + d];
}

void glm_index_select(GlmCtx* ctx, void* out, const void* src,
                       const void* indices, int dim, int k, int offset) {
    cudaSetDevice(ctx->device_id);
    int total = k * dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    index_select_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)src,
        (const int*)indices, dim, k, offset);
}

// ---------------------------------------------------------------------------
// Max kernel (find max value and index per row in BF16 input)
//   out_values[row] = max(input[row * dim : (row+1) * dim])
//   out_indices[row] = argmax(input[row * dim : (row+1) * dim])
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) max_kernel(__nv_bfloat16* out_values, int* out_indices,
                              const __nv_bfloat16* input, int dim, int batch, int offset) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const __nv_bfloat16* row_in = input + row * dim;

    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int* s_idxs = reinterpret_cast<int*>(s_vals + blockDim.x);

    float my_max = -INFINITY;
    int my_idx = -1;
    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float2 v = load_bf16x2(row_in + i);
        if (v.x > my_max || (v.x == my_max && i < my_idx)) { my_max = v.x; my_idx = i; }
        if (v.y > my_max || (v.y == my_max && (i + 1) < my_idx)) { my_max = v.y; my_idx = i + 1; }
    }
    if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
        float val = __bfloat162float(row_in[dim - 1]);
        if (val > my_max || (val == my_max && (dim - 1) < my_idx)) { my_max = val; my_idx = dim - 1; }
    }
    s_vals[threadIdx.x] = my_max;
    s_idxs[threadIdx.x] = my_idx;
    __syncthreads();

    block_reduce_max_idx(s_vals, s_idxs, threadIdx.x);

    if (threadIdx.x == 0) {
        out_values[row] = __float2bfloat16(s_vals[0]);
        out_indices[row] = s_idxs[0] + offset;
    }
}

void glm_max(GlmCtx* ctx, void* out_values, int* out_indices, const void* input, int dim, int batch, int offset) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = batch;
    size_t shared_mem = block_size * sizeof(float) + block_size * sizeof(int);
    max_kernel<<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out_values, out_indices, (const __nv_bfloat16*)input, dim, batch, offset);
}

// ---------------------------------------------------------------------------
// Arange kernel (fill int32 buffer with sequential values)
//   out[i] = start + i * step
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) arange_kernel(int* out, int start, int step, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = start + idx * step;
    }
}

void glm_arange(GlmCtx* ctx, int* out, int start, int step, int count) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (count + block_size - 1) / block_size;
    arange_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(out, start, step, count);
}

// ---------------------------------------------------------------------------
// KV cache write kernel (vLLM-style slot_mapping scatter)
// src_k: [batch, n_kv, hd] or [n_kv, batch, hd] BF16 — layout described by strides
// src_v: [batch, n_kv, hd] or [n_kv, batch, hd] BF16 — layout described by strides
// dst_k, dst_v: [max_pages, n_kv, page_size, hd] BF16
// slot_mapping: [batch] int32 — slot = page * page_size + slot_in_page, -1 = skip
// K and V can have different layouts (different strides).
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) kv_cache_write_kernel(
    __nv_bfloat16* dst_k,
    __nv_bfloat16* dst_v,
    const __nv_bfloat16* src_k,
    const __nv_bfloat16* src_v,
    const int32_t* slot_mapping,
    uint32_t n_kv,
    uint32_t page_size,
    uint32_t hd,
    uint32_t src_k_token_stride,
    uint32_t src_k_head_stride,
    uint32_t src_v_token_stride,
    uint32_t src_v_head_stride
) {
    int token = blockIdx.x;
    int h = blockIdx.y;
    if (token >= gridDim.x || h >= gridDim.y) return;

    int32_t slot = slot_mapping[token];
    if (slot < 0) return;

    int32_t page = slot / (int32_t)page_size;
    int32_t slot_in_page = slot % (int32_t)page_size;

    int64_t dst_off = (int64_t)page * n_kv * page_size * hd
                     + (int64_t)h * page_size * hd
                     + (int64_t)slot_in_page * hd;
    int64_t src_k_off = (int64_t)token * src_k_token_stride + (int64_t)h * src_k_head_stride;
    int64_t src_v_off = (int64_t)token * src_v_token_stride + (int64_t)h * src_v_head_stride;

    for (int d = threadIdx.x; d < (int)hd; d += blockDim.x) {
        dst_k[dst_off + d] = src_k[src_k_off + d];
        dst_v[dst_off + d] = src_v[src_v_off + d];
    }
}

void glm_kv_cache_write(GlmCtx* ctx,
                          void* src_k, void* src_v,
                          void* dst_k, void* dst_v,
                          int32_t* slot_mapping,
                          uint32_t batch_size, uint32_t n_kv,
                          uint32_t hd, uint32_t page_size,
                          uint32_t src_k_token_stride, uint32_t src_k_head_stride,
                          uint32_t src_v_token_stride, uint32_t src_v_head_stride) {
    cudaSetDevice(ctx->device_id);
    dim3 grid(batch_size, n_kv);
    dim3 block(hd);
    kv_cache_write_kernel<<<grid, block, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)dst_k, (__nv_bfloat16*)dst_v,
        (const __nv_bfloat16*)src_k, (const __nv_bfloat16*)src_v,
        slot_mapping, n_kv, page_size, hd,
        src_k_token_stride, src_k_head_stride,
        src_v_token_stride, src_v_head_stride);
}

// ---------------------------------------------------------------------------
// Scatter-add with row-wise scaling
//   out:      [num_rows, dim]   BF16
//   input:    [count, dim]      BF16   (count = num_rows * top_k)
//   scales:   [count]           BF16
//   top_k:    number of expert entries per row
//   For each row r in [0, num_rows), for each j in [0, top_k):
//     out[r, d] += scales[r * top_k + j] * input[r * top_k + j, d]
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256) scatter_add_rows_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* scales,
    int top_k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    float accum = 0.0f;
    for (int j = 0; j < top_k; j++) {
        accum += __bfloat162float(scales[j]) * __bfloat162float(input[(size_t)j * dim + idx]);
    }
    out[idx] = __float2bfloat16(accum);
}

__global__ void __launch_bounds__(256, 4) scatter_add_rows_batched_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* scales,
    int top_k, int dim, int num_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * dim;
    if (idx >= total) return;
    int row = idx / dim;
    int d = idx % dim;
    float accum = 0.0f;
    int base = row * top_k;
    for (int j = 0; j < top_k; j++) {
        accum += __bfloat162float(scales[base + j]) * __bfloat162float(input[(size_t)(base + j) * dim + d]);
    }
    out[idx] = __float2bfloat16(accum);
}

// ---------------------------------------------------------------------------
// Rotate input IDs for MTP prefill
// ---------------------------------------------------------------------------

// One block per sequence. Each block shifts its sequence left by 1
// and appends a new token at the last position.
// input_ids:  [totalTokens] I32  (read-only)
// output_ids: [totalTokens] I32  (write-only)
// qo_indptr:  [batchSize+1] I32  (cumulative offsets)
// new_tokens: [batchSize] I32    (new token per sequence)
__global__ void __launch_bounds__(256, 4) rotate_input_ids_kernel(
    const int* __restrict__ input_ids,
    int* __restrict__ output_ids,
    const int* __restrict__ qo_indptr,
    const int* __restrict__ new_tokens,
    int batch_size
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    int start = qo_indptr[b];
    int len = qo_indptr[b + 1] - qo_indptr[b];
    int new_token = new_tokens[b];

    // Shift left by 1: output[i] = input[i + 1]
    int shift_len = len - 1;
    for (int off = threadIdx.x; off < shift_len; off += blockDim.x) {
        output_ids[start + off] = input_ids[start + off + 1];
    }

    // Append new token at last position
    if (len > 0 && threadIdx.x == 0) {
        output_ids[start + len - 1] = new_token;
    }
}

extern "C" {

void glm_scatter_add_rows(GlmCtx* ctx, void* out, const void* input,
                            const void* scales, int top_k,
                            int dim, int num_rows, void* workspace) {
    cudaSetDevice(ctx->device_id);
    (void)workspace;
    int block_size = 256;
    if (num_rows == 1) {
        int grid = (dim + block_size - 1) / block_size;
        scatter_add_rows_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
            (const __nv_bfloat16*)scales, top_k, dim);
    } else {
        int total = num_rows * dim;
        int grid = (total + block_size - 1) / block_size;
        scatter_add_rows_batched_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
            (const __nv_bfloat16*)scales, top_k, dim, num_rows);
    }
}

void glm_rotate_input_ids(GlmCtx* ctx, int* output_ids, const int* input_ids,
                           const int* qo_indptr, const int* new_tokens,
                           int batch_size) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    rotate_input_ids_kernel<<<batch_size, block_size, 0, GLM_STREAM(ctx)>>>(
        input_ids, output_ids, qo_indptr, new_tokens, batch_size);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Smem-staged sum of N tensors (element-wise, max 8 inputs).
// Peers are streamed through D double-buffered smem slots while threads
// accumulate in FP32 registers, so the per-peer read size is decoupled from N
// (always the full block_stride, 16 KB for BF16). D = min(N, smem_budget /
// per_peer_bytes) is chosen on the host: small tensors issue all N peers in
// flight (one drain); large tensors pipeline 2 peers. wait_prior<D-1> requires
// a compile-time D, hence the template parameter and the host switch.
// ---------------------------------------------------------------------------

template <typename scalar_t, int ElemsPerWarp, int D_VAL, bool WRITEBACK>
__global__ void __launch_bounds__(512, 2)
sum_pointers_smem_kernel(
    const scalar_t* p0,  const scalar_t* p1,  const scalar_t* p2,  const scalar_t* p3,
    const scalar_t* p4,  const scalar_t* p5,  const scalar_t* p6,  const scalar_t* p7,
    scalar_t* __restrict__ output,
    int N, int64_t numel, int64_t peer_stride_elems)
{
    constexpr int WarpSize = 32;
    auto block = cg::this_thread_block();
    extern __shared__ char smem_raw[];

    int warp_id = threadIdx.x / WarpSize;
    int lane    = threadIdx.x % WarpSize;
    int warps_per_block = blockDim.x / WarpSize;
    int64_t block_stride = (int64_t)warps_per_block * ElemsPerWarp;
    int64_t total_blocks = gridDim.x;
    int64_t my_start = (int64_t)blockIdx.x * block_stride;

    const char* peers[8] = {
        reinterpret_cast<const char*>(p0), reinterpret_cast<const char*>(p1),
        reinterpret_cast<const char*>(p2), reinterpret_cast<const char*>(p3),
        reinterpret_cast<const char*>(p4), reinterpret_cast<const char*>(p5),
        reinterpret_cast<const char*>(p6), reinterpret_cast<const char*>(p7),
    };

    constexpr int VEC = (sizeof(scalar_t) == 2) ? 2 : 1;
    constexpr int PAIRS = ElemsPerWarp / (WarpSize * VEC);
    int64_t warp_start = (int64_t)warp_id * ElemsPerWarp;
    const size_t elem_sz = sizeof(scalar_t);

    for (int64_t blk = my_start; blk < numel; blk += total_blocks * block_stride) {
        int64_t elems = min(block_stride, numel - blk);
        size_t copy_bytes = (size_t)elems * elem_sz;

        float2 acc[PAIRS];
        #pragma unroll
        for (int k = 0; k < PAIRS; k++) acc[k] = {0.0f, 0.0f};

        #pragma unroll
        for (int k = 0; k < D_VAL; k++) {
            cg::memcpy_async(block,
                smem_raw + (size_t)k * peer_stride_elems * elem_sz,
                peers[k] + (size_t)blk * elem_sz,
                copy_bytes);
        }

        int steady = max(0, N - D_VAL);
        for (int j = 0; j < steady; j++) {
            cg::wait_prior<D_VAL - 1>(block);
            char* buf = smem_raw + (size_t)(j % D_VAL) * peer_stride_elems * elem_sz;
            #pragma unroll
            for (int k = 0; k < PAIRS; k++) {
                int64_t off = warp_start + lane * VEC + (int64_t)k * WarpSize * VEC;
                if (off + VEC <= elems) {
                    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
                        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(buf + off * elem_sz);
                        float2 f = __bfloat1622float2(v);
                        acc[k].x += f.x; acc[k].y += f.y;
                    } else {
                        float v = *reinterpret_cast<const float*>(buf + off * elem_sz);
                        acc[k].x += v;
                    }
                }
            }
            // Every thread must be done reading this slot before the next peer
            // is streamed into it: cg::wait_prior only orders the fill side, so
            // without this a fast warp's refill overwrites a slow warp's reads.
            __syncthreads();
            cg::memcpy_async(block,
                smem_raw + (size_t)(j % D_VAL) * peer_stride_elems * elem_sz,
                peers[j + D_VAL] + (size_t)blk * elem_sz,
                copy_bytes);
        }

        cg::wait(block);
        for (int j = steady; j < N; j++) {
            char* buf = smem_raw + (size_t)(j % D_VAL) * peer_stride_elems * elem_sz;
            #pragma unroll
            for (int k = 0; k < PAIRS; k++) {
                int64_t off = warp_start + lane * VEC + (int64_t)k * WarpSize * VEC;
                if (off + VEC <= elems) {
                    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
                        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(buf + off * elem_sz);
                        float2 f = __bfloat1622float2(v);
                        acc[k].x += f.x; acc[k].y += f.y;
                    } else {
                        float v = *reinterpret_cast<const float*>(buf + off * elem_sz);
                        acc[k].x += v;
                    }
                }
            }
        }

        if constexpr (WRITEBACK) {
            for (int j = 0; j < N; j++) {
                #pragma unroll
                for (int k = 0; k < PAIRS; k++) {
                    int64_t off = warp_start + lane * VEC + (int64_t)k * WarpSize * VEC;
                    if (off + VEC <= elems) {
                        if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
                            *reinterpret_cast<__nv_bfloat162*>(
                                const_cast<char*>(peers[j]) + (size_t)(blk + off) * elem_sz) =
                                __float22bfloat162_rn(acc[k]);
                        } else {
                            *reinterpret_cast<float*>(
                                const_cast<char*>(peers[j]) + (size_t)(blk + off) * elem_sz) = acc[k].x;
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k = 0; k < PAIRS; k++) {
                int64_t off = warp_start + lane * VEC + (int64_t)k * WarpSize * VEC;
                if (off + VEC <= elems) {
                    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
                        *reinterpret_cast<__nv_bfloat162*>(
                            reinterpret_cast<char*>(output) + (size_t)(blk + off) * elem_sz) =
                            __float22bfloat162_rn(acc[k]);
                    } else {
                        *reinterpret_cast<float*>(
                            reinterpret_cast<char*>(output) + (size_t)(blk + off) * elem_sz) = acc[k].x;
                    }
                }
            }
        }

        __syncthreads();   // ensure drain's ld.shared done before next blk's cp.async
    }
}

extern "C" {

void glm_sum_pointers(GlmCtx* ctx,
    void* p0,  void* p1,  void* p2,  void* p3,
    void* p4,  void* p5,  void* p6,  void* p7,
    void* output, int N, int64_t numel, int dtype, bool writeback) {
    cudaSetDevice(ctx->device_id);

    constexpr int ElemsPerWarp = 512;
    constexpr int WarpsPerBlock = 16;
    constexpr int64_t BlockStride = (int64_t)WarpsPerBlock * ElemsPerWarp;
    constexpr int64_t SmemBudget = 32 * 1024;

    int elem_size = (dtype == 9) ? 2 : 4;
    int64_t peer_stride_elems = (numel < BlockStride) ? numel : BlockStride;
    int64_t peer_stride_bytes = peer_stride_elems * elem_size;
    if (peer_stride_bytes < 1) peer_stride_bytes = 1;
    int64_t budget = SmemBudget / peer_stride_bytes;
    int D = (int)((budget < N) ? budget : N);
    if (D < 1) D = 1;
    if (D > 8) D = 8;

    int64_t total_warps = (numel + ElemsPerWarp - 1) / ElemsPerWarp;
    if (total_warps == 0) total_warps = 1;
    int grid = (int)((total_warps + WarpsPerBlock - 1) / WarpsPerBlock);
    int block_size = WarpsPerBlock * 32;
    int64_t smem_bytes = (int64_t)D * peer_stride_bytes;

#define DISPATCH_SUM(SCT, DVAL, WB) do { \
    cudaFuncSetAttribute( \
        (void*)sum_pointers_smem_kernel<SCT, ElemsPerWarp, DVAL, WB>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 32768); \
    sum_pointers_smem_kernel<SCT, ElemsPerWarp, DVAL, WB><<<grid, block_size, smem_bytes, GLM_STREAM(ctx)>>>( \
        (const SCT*)p0, (const SCT*)p1, (const SCT*)p2, (const SCT*)p3, \
        (const SCT*)p4, (const SCT*)p5, (const SCT*)p6, (const SCT*)p7, \
        (SCT*)output, N, numel, peer_stride_elems); \
} while (0)

#define DISPATCH_DTYPE(SCT, WB) do { \
    switch (D) { \
        case 8: DISPATCH_SUM(SCT, 8, WB); break; \
        case 7: DISPATCH_SUM(SCT, 7, WB); break; \
        case 6: DISPATCH_SUM(SCT, 6, WB); break; \
        case 5: DISPATCH_SUM(SCT, 5, WB); break; \
        case 4: DISPATCH_SUM(SCT, 4, WB); break; \
        case 3: DISPATCH_SUM(SCT, 3, WB); break; \
        case 2: DISPATCH_SUM(SCT, 2, WB); break; \
        default: DISPATCH_SUM(SCT, 1, WB); break; \
    } \
} while (0)

    if (dtype == 9) {
        if (writeback) DISPATCH_DTYPE(__nv_bfloat16, true);
        else           DISPATCH_DTYPE(__nv_bfloat16, false);
    } else {
        if (writeback) DISPATCH_DTYPE(float, true);
        else           DISPATCH_DTYPE(float, false);
    }
#undef DISPATCH_SUM
#undef DISPATCH_DTYPE
}

// ---------------------------------------------------------------------------
// write_pointers: graph-capturable pointer array writer
// ---------------------------------------------------------------------------

__global__ void write_pointers_kernel(
    void** dst,
    void* p0, void* p1, void* p2, void* p3,
    void* p4, void* p5, void* p6, void* p7,
    int n)
{
    if (threadIdx.x < n) {
        switch (threadIdx.x) {
            case 0: dst[0] = p0; break;
            case 1: dst[1] = p1; break;
            case 2: dst[2] = p2; break;
            case 3: dst[3] = p3; break;
            case 4: dst[4] = p4; break;
            case 5: dst[5] = p5; break;
            case 6: dst[6] = p6; break;
            case 7: dst[7] = p7; break;
        }
    }
}

void glm_write_pointers(GlmCtx* ctx, void* dst,
                        void* p0, void* p1, void* p2, void* p3,
                        void* p4, void* p5, void* p6, void* p7,
                        int n) {
    cudaSetDevice(ctx->device_id);
    write_pointers_kernel<<<1, 8, 0, GLM_STREAM(ctx)>>>(
        (void**)dst, p0, p1, p2, p3, p4, p5, p6, p7, n);
}

} // extern "C"
