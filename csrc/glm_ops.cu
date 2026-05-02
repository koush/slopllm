#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

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

__device__ __forceinline__ float rope_rotate(float val, float paired_val, float cos_val, float sin_val, int d, int half) {
    float rotated = (d < half) ? -paired_val : paired_val;
    return val * cos_val + rotated * sin_val;
}

__device__ void block_reduce_sum(float* sdata, int tid) {
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
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

__device__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// ---------------------------------------------------------------------------
// BF16 vector I/O helpers
// ---------------------------------------------------------------------------

__device__ inline void load_bf16x2(const __nv_bfloat16* ptr, float& v0, float& v1) {
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(ptr);
    v0 = __bfloat162float(v.x);
    v1 = __bfloat162float(v.y);
}

__device__ inline void store_bf16x2(__nv_bfloat16* ptr, float v0, float v1) {
    __nv_bfloat162 v;
    v.x = __float2bfloat16(v0);
    v.y = __float2bfloat16(v1);
    *reinterpret_cast<__nv_bfloat162*>(ptr) = v;
}

// Compute sum of squares of bf16 vector, reduce across block, return inv_rms.
// Caller must provide extern __shared__ float sdata[].
__device__ float compute_inv_rms(const __nv_bfloat16* x, int dim, float eps, float* sdata) {
    float sum = 0.0f;
    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float v0, v1;
        load_bf16x2(x + i, v0, v1);
        sum += v0 * v0 + v1 * v1;
    }
    if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
        float val = __bfloat162float(x[dim - 1]);
        sum += val * val;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    block_reduce_sum(sdata, threadIdx.x);
    return rsqrtf(sdata[0] / dim + eps);
}

// ---------------------------------------------------------------------------
// RMSNorm kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) rmsnorm_kernel(
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

    float inv_rms = compute_inv_rms(x, dim, eps, sdata);

    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float x0, x1, w0, w1;
        load_bf16x2(x + i, x0, x1);
        load_bf16x2(weight + i, w0, w1);
        store_bf16x2(o + i, w0 * x0 * inv_rms, w1 * x1 * inv_rms);
    }
    if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
        float xi = __bfloat162float(x[dim - 1]);
        float wi = __bfloat162float(weight[dim - 1]);
        o[dim - 1] = __float2bfloat16(wi * xi * inv_rms);
    }
}

void glm_rmsnorm(GlmCtx* ctx, void* out, const void* input,
                 const void* weight, float eps, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int block_size = compute_block_size(dim);
    size_t shared_mem = block_size * sizeof(float);
    rmsnorm_kernel<<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)weight, eps, dim);
}

// ---------------------------------------------------------------------------
// Fused Add + RMSNorm kernel
// out[i] = weight[i] * (input_a[i] + input_b[i]) * inv_rms
// residual[i] = input_a[i] + input_b[i]
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) fused_add_rmsnorm_kernel(
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
        float a0, a1, b0, b1;
        load_bf16x2(a + i, a0, a1);
        load_bf16x2(b + i, b0, b1);
        float s0 = a0 + b0;
        float s1 = a1 + b1;
        sum += s0 * s0 + s1 * s1;
    }
    if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
        float ai = __bfloat162float(a[dim - 1]);
        float bi = __bfloat162float(b[dim - 1]);
        float si = ai + bi;
        sum += si * si;
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    block_reduce_sum(sdata, threadIdx.x);

    float inv_rms = rsqrtf(sdata[0] / dim + eps);

    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float a0, a1, b0, b1, w0, w1;
        load_bf16x2(a + i, a0, a1);
        load_bf16x2(b + i, b0, b1);
        load_bf16x2(weight + i, w0, w1);
        float s0 = a0 + b0;
        float s1 = a1 + b1;
        store_bf16x2(r + i, s0, s1);
        store_bf16x2(o + i, w0 * s0 * inv_rms, w1 * s1 * inv_rms);
    }
    if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
        float ai = __bfloat162float(a[dim - 1]);
        float bi = __bfloat162float(b[dim - 1]);
        float wi = __bfloat162float(weight[dim - 1]);
        float si = ai + bi;
        r[dim - 1] = __float2bfloat16(si);
        o[dim - 1] = __float2bfloat16(wi * si * inv_rms);
    }
}

void glm_fused_add_rmsnorm(GlmCtx* ctx, void* out, void* residual,
                            const void* input_a, const void* input_b,
                            const void* weight, float eps, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int block_size = compute_block_size(dim);
    size_t shared_mem = block_size * sizeof(float);
    fused_add_rmsnorm_kernel<<<batch, block_size, shared_mem, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (__nv_bfloat16*)residual,
        (const __nv_bfloat16*)input_a, (const __nv_bfloat16*)input_b,
        (const __nv_bfloat16*)weight, eps, dim);
}

// ---------------------------------------------------------------------------
// Fused per-head RMSNorm + RoPE kernel
// Input: [batch * seq_len, n_heads * head_dim] (projection output, row-major)
// Output: [batch, n_heads, seq_len, head_dim] (HND layout for attention)
// Applies per-head RMSNorm then RoPE, with layout transpose.
// ---------------------------------------------------------------------------

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
            float ci = __bfloat162float(cos_emb[cos_base + i]);
            float si = __bfloat162float(sin_emb[cos_base + i]);
            float paired_norm = (i < half)
                ? __bfloat162float(weight[i + half]) * __bfloat162float(x[i + half]) * inv_rms
                : __bfloat162float(weight[i - half]) * __bfloat162float(x[i - half]) * inv_rms;
            ni = rope_rotate(ni, paired_norm, ci, si, i, half);
        }
        o[i] = __float2bfloat16(ni);
    }
}

void glm_fused_norm_rope(GlmCtx* ctx, void* out, const void* in,
                          const void* weight, const void* cos_emb, const void* sin_emb,
                          float eps, int rope_dim, int head_dim,
                          int n_heads, int seq_len, int batch, int in_stride) {
    cudaSetDevice(ctx->device_id);
    int total_rows = batch * n_heads * seq_len;
    int block_size = compute_block_size(head_dim, true);
    size_t shared_mem = block_size * sizeof(float);
    fused_norm_rope_kernel<<<total_rows, block_size, shared_mem, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)in,
        (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)cos_emb, (const __nv_bfloat16*)sin_emb,
        eps, rope_dim, head_dim, n_heads, seq_len, batch, in_stride);
}

// ---------------------------------------------------------------------------
// RoPE + Head Transpose kernel
// Input: [batch * seq_len, n_heads * in_stride] (projection output, interleaved heads)
// Output: [batch * n_heads, seq_len, head_dim] (per-head contiguous for attention)
// Applies RoPE to first rope_dim dimensions if rope_dim > 0; otherwise just transposes.
// Cos/sin embeddings: [batch, seq_len, rope_dim] (only used if rope_dim > 0)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) rope_transpose_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ cos_emb,
    const __nv_bfloat16* __restrict__ sin_emb,
    int rope_dim, int head_dim, int n_heads,
    int seq_len, int batch, int in_stride
) {
    int bhs = blockIdx.x;
    int s = bhs % seq_len;
    int h = (bhs / seq_len) % n_heads;
    int b = bhs / (seq_len * n_heads);

    const __nv_bfloat16* x = in + (b * seq_len + s) * n_heads * in_stride + h * in_stride;
    __nv_bfloat16* o = out + ((b * seq_len + s) * n_heads + h) * head_dim;

    int half = rope_dim / 2;
    int cos_base = (b * seq_len + s) * rope_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float xi = __bfloat162float(x[i]);
        if (i < rope_dim && rope_dim > 0) {
            float ci = __bfloat162float(cos_emb[cos_base + i]);
            float si = __bfloat162float(sin_emb[cos_base + i]);
            float paired = (i < half)
                ? __bfloat162float(x[i + half])
                : __bfloat162float(x[i - half]);
            o[i] = __float2bfloat16(rope_rotate(xi, paired, ci, si, i, half));
        } else {
            o[i] = __float2bfloat16(xi);
        }
    }
}

void glm_rope_transpose(GlmCtx* ctx, void* out, const void* in,
                         const void* cos_emb, const void* sin_emb,
                         int rope_dim, int head_dim, int n_heads,
                         int seq_len, int batch, int in_stride) {
    cudaSetDevice(ctx->device_id);
    int total_rows = batch * n_heads * seq_len;
    int block_size = compute_block_size(head_dim, true);
    rope_transpose_kernel<<<total_rows, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)in,
        (const __nv_bfloat16*)cos_emb, (const __nv_bfloat16*)sin_emb,
        rope_dim, head_dim, n_heads, seq_len, batch, in_stride);
}

// ---------------------------------------------------------------------------
// MLA V-Expand kernel
// attn_out: [batch, n_heads, seq_len, kv_lora_rank] (HND layout from FlashInfer)
// v_proj: [n_heads * v_head_dim, kv_lora_rank] (row-major, per-head weights)
// result: [batch, n_heads, seq_len, v_head_dim] (HND layout)
// Computes result[b,h,s,j] = sum_k(attn_out[b,h,s,k] * v_proj[h*v_head_dim+j, k])
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) mla_v_expand_kernel(
    __nv_bfloat16* __restrict__ result,
    const __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ v_proj,
    int kv_lora_rank, int v_head_dim, int n_heads,
    int seq_len, int batch
) {
    int bhs = blockIdx.x;
    int s = bhs % seq_len;
    int h = (bhs / seq_len) % n_heads;
    int b = bhs / (seq_len * n_heads);

    const __nv_bfloat16* attn_row = attn_out + ((b * n_heads + h) * seq_len + s) * kv_lora_rank;
    const __nv_bfloat16* w_base = v_proj + h * v_head_dim * kv_lora_rank;

    for (int j = threadIdx.x; j < v_head_dim; j += blockDim.x) {
        float sum = 0.0f;
        const __nv_bfloat16* w_row = w_base + j * kv_lora_rank;
        for (int k = 0; k < kv_lora_rank; k++) {
            sum += __bfloat162float(attn_row[k]) * __bfloat162float(w_row[k]);
        }
        result[(b * seq_len + s) * (n_heads * v_head_dim) + h * v_head_dim + j] = __float2bfloat16(sum);
    }
}

void glm_mla_v_expand(GlmCtx* ctx, void* result, const void* attn_out,
                       const void* v_proj,
                       int kv_lora_rank, int v_head_dim, int n_heads,
                       int seq_len, int batch) {
    cudaSetDevice(ctx->device_id);
    int total_rows = batch * n_heads * seq_len;
    int block_size = compute_block_size(v_head_dim, true);
    mla_v_expand_kernel<<<total_rows, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)result, (const __nv_bfloat16*)attn_out,
        (const __nv_bfloat16*)v_proj,
        kv_lora_rank, v_head_dim, n_heads, seq_len, batch);
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
        float g0, g1, u0, u1;
        load_bf16x2(gate + idx, g0, g1);
        load_bf16x2(up + idx, u0, u1);
        store_bf16x2(out + idx, g0 * sigmoid_f(g0) * u0, g1 * sigmoid_f(g1) * u1);
    } else if (idx < total) {
        float g = __bfloat162float(gate[idx]);
        float u = __bfloat162float(up[idx]);
        out[idx] = __float2bfloat16(g * sigmoid_f(g) * u);
    }
}

void glm_silu_and_mul(GlmCtx* ctx, void* out, const void* gate,
                      const void* up, int intermediate, int batch) {
    cudaSetDevice(ctx->device_id);
    int total = batch * intermediate;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    silu_and_mul_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up, total);
}



// ---------------------------------------------------------------------------
// Embedding lookup kernel
// ---------------------------------------------------------------------------

void glm_embedding(GlmCtx* ctx, void* out, const void* table,
                   const int* ids, int hidden, int seq_len) {
    glm_index_select(ctx, out, table, ids, hidden, seq_len);
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
        float v0, v1;
        load_bf16x2(x + i, v0, v1);
        mean += v0 + v1;
    }
    if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
        mean += __bfloat162float(x[dim - 1]);
    }
    sdata[threadIdx.x] = mean;
    __syncthreads();
    block_reduce_sum(sdata, threadIdx.x);
    mean = sdata[0] / dim;

    float var = 0.0f;
    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float v0, v1;
        load_bf16x2(x + i, v0, v1);
        float d0 = v0 - mean, d1 = v1 - mean;
        var += d0 * d0 + d1 * d1;
    }
    if ((dim & 1) && threadIdx.x == (dim / 2) % blockDim.x) {
        float d = __bfloat162float(x[dim - 1]) - mean;
        var += d * d;
    }
    sdata[threadIdx.x] = var;
    __syncthreads();
    block_reduce_sum(sdata, threadIdx.x);
    float inv_std = rsqrtf(sdata[0] / dim + eps);

    for (int i = threadIdx.x * 2; i + 1 < dim; i += blockDim.x * 2) {
        float x0, x1, w0, w1;
        load_bf16x2(x + i, x0, x1);
        load_bf16x2(weight + i, w0, w1);
        float b0 = bias ? __bfloat162float(bias[i]) : 0.0f;
        float b1 = bias ? __bfloat162float(bias[i + 1]) : 0.0f;
        store_bf16x2(o + i, w0 * (x0 - mean) * inv_std + b0,
                              w1 * (x1 - mean) * inv_std + b1);
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
    size_t shared_mem = block_size * sizeof(float);
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
        float v0, v1;
        load_bf16x2(input + idx, v0, v1);
        store_bf16x2(out + idx, F(v0), F(v1));
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

template<auto F>
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
        float a0, a1, b0, b1;
        load_bf16x2(a + r * a_stride + c, a0, a1);
        load_bf16x2(b + r * b_stride + c, b0, b1);
        store_bf16x2(out + idx, F(a0, b0), F(a1, b1));
    } else if (idx < total) {
        int r = idx / dim;
        int c = idx % dim;
        out[idx] = __float2bfloat16(F(__bfloat162float(a[r * a_stride + c]), __bfloat162float(b[r * b_stride + c])));
    }
}

// ---------------------------------------------------------------------------
// Element-wise unary 2D kernel (templated) — pitched input
// Reads from a 2D pitched view of input: element(r, c) = in[r * pitch + col_offset + c]
// F: (float, float) -> float, receives (out_val, in_val)
// out: [rows * cols] contiguous
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float sigmoid_mul_f(float val, float gate) {
    float sig = 1.0f / (1.0f + __expf(-gate));
    return val * sig;
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

static __device__ __forceinline__ float relu_f(float v) { return fmaxf(v, 0.0f); }

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
    sdata[threadIdx.x] = sum;
    __syncthreads();
    block_reduce_sum(sdata, threadIdx.x);
    float inv_sum = 1.0f / sdata[0];

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
// Fills upper triangle with -inf: out[i][j] = (j > i) ? -inf : 0
// out: [seq_len, seq_len] BF16
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) causal_mask_kernel(
    __nv_bfloat16* out,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;
    if (idx < total) {
        int row = idx / seq_len;
        int col = idx % seq_len;
        float val = (col > row) ? -INFINITY : 0.0f;
        out[idx] = __float2bfloat16(val);
    }
}

void glm_causal_mask(GlmCtx* ctx, void* out, int seq_len) {
    cudaSetDevice(ctx->device_id);
    int total = seq_len * seq_len;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    causal_mask_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, seq_len);
}

// ---------------------------------------------------------------------------
// Fill kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) fill_kernel(__nv_bfloat16* out, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(value);
    }
}

void glm_fill(GlmCtx* ctx, void* out, float value, int n) {
    cudaSetDevice(ctx->device_id);
    if (value == 0.0f) {
        cudaMemsetAsync(out, 0, n * sizeof(__nv_bfloat16), GLM_STREAM(ctx));
    } else {
        int block_size = 256;
        int grid = (n + block_size - 1) / block_size;
        fill_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, value, n);
    }
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
// unsqueeze_dim=1: x is [batch, n_heads, seq_len, head_dim]
// unsqueeze_dim=2: x is [batch, seq_len, n_heads, head_dim]
// cos, sin are [batch, seq_len, rope_dim] (broadcast over heads)
// rope_dim <= head_dim: only first rope_dim dims get RoPE, rest pass through
// When head_dim == rope_dim, this reduces to the original behavior.
// ---------------------------------------------------------------------------

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
    int unsqueeze_dim
) {
    int total = batch * n_heads * seq_len * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % head_dim;
    int b, h, s;
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

    int x_idx;
    if (unsqueeze_dim == 1) {
        x_idx = ((b * n_heads + h) * seq_len + s) * head_dim + d;
    } else {
        x_idx = ((b * seq_len + s) * n_heads + h) * head_dim + d;
    }

    if (d < rope_dim) {
        int cos_idx = (b * seq_len + s) * rope_dim + d;
        int half = rope_dim / 2;

        float x_val = __bfloat162float(x[x_idx]);
        float cos_val = __bfloat162float(cos_emb[cos_idx]);
        float sin_val = __bfloat162float(sin_emb[cos_idx]);

        int paired_idx = (d < half) ? x_idx + half : x_idx - half;
        float paired_val = __bfloat162float(x[paired_idx]);

        out[x_idx] = __float2bfloat16(rope_rotate(x_val, paired_val, cos_val, sin_val, d, half));
    } else {
        out[x_idx] = x[x_idx];
    }
}

void glm_apply_rotary_pos_emb(GlmCtx* ctx, void* out, const void* x,
                               const void* cos, const void* sin,
                               int rope_dim, int n_heads, int seq_len,
                               int batch, int unsqueeze_dim) {
    glm_apply_rotary_pos_emb_partial(ctx, out, x, cos, sin,
                                      rope_dim, rope_dim, n_heads, seq_len,
                                      batch, unsqueeze_dim);
}

void glm_apply_rotary_pos_emb_partial(GlmCtx* ctx, void* out, const void* x,
                                        const void* cos, const void* sin,
                                        int rope_dim, int head_dim, int n_heads, int seq_len,
                                        int batch, int unsqueeze_dim) {
    cudaSetDevice(ctx->device_id);
    int total = batch * n_heads * seq_len * head_dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    apply_rotary_pos_emb_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin,
        rope_dim, head_dim, seq_len, n_heads, batch, unsqueeze_dim);
}

// ---------------------------------------------------------------------------
// TopK kernel (along last dim, unsorted)
// One block per row. Loads row into shared memory, finds top-k by
// k passes of parallel argmax.
// Shared memory: dim * (sizeof(float) + sizeof(int)) + blockDim * (sizeof(float) + sizeof(int))
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) topk_kernel(
    __nv_bfloat16* out_values,
    int* out_indices,
    const __nv_bfloat16* input,
    int k,
    int dim,
    int batch
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
                row_idxs[j] = s_idxs[best];
                s_vals[best] = __int_as_float(0x7fc00000);
            } else {
                row_vals[j] = __float2bfloat16(-INFINITY);
                row_idxs[j] = 0;
            }
        }
        __syncthreads();
    }
}

void glm_topk(GlmCtx* ctx, void* out_values, int* out_indices,
              const void* input, int k, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = batch;
    size_t shared_mem = dim * sizeof(float) + dim * sizeof(int) +
                        block_size * sizeof(float) + block_size * sizeof(int);
    topk_kernel<<<grid, block_size, shared_mem, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out_values, out_indices,
        (const __nv_bfloat16*)input, k, dim, batch);
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
             int batch, int M, int N, int K, int transA, int transB) {
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
        float v0, v1;
        load_bf16x2(input + idx, v0, v1);
        store_bf16x2(out + idx, v0 * scale, v1 * scale);
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

static __device__ __forceinline__ float add_f(float a, float b) { return a + b; }

void glm_add(GlmCtx* ctx, void* out, const void* a, const void* b, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    ew_binary_2d_kernel<add_f><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
        n, 1, n, n);
}

void glm_add_broadcast(GlmCtx* ctx, void* out, const void* a, const void* b, int dim, int rows) {
    cudaSetDevice(ctx->device_id);
    int total = rows * dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
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

__global__ void __launch_bounds__(256, 4) row_scale_add_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* scales,
    int rows,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (idx < total) {
        int row = idx / dim;
        float s = __bfloat162float(scales[row]);
        float v = __bfloat162float(input[idx]);
        float o = __bfloat162float(out[idx]);
        out[idx] = __float2bfloat16(o + s * v);
    }
}

void glm_row_scale_add(GlmCtx* ctx, void* out, const void* input,
                        const void* scales, int rows, int dim) {
    cudaSetDevice(ctx->device_id);
    int total = rows * dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    row_scale_add_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)scales, rows, dim);
}

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

void glm_transpose_4d(GlmCtx* ctx, void* out, const void* input,
                      int dim0, int dim1, int dim2, int dim3,
                      int perm0, int perm1, int perm2, int perm3) {
    cudaSetDevice(ctx->device_id);
    int total = dim0 * dim1 * dim2 * dim3;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    if (perm0 == 0 && perm1 == 2 && perm2 == 1 && perm3 == 3) {
        transpose_0213_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
            dim0, dim1, dim2, dim3);
    } else {
        transpose_4d_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
            dim0, dim1, dim2, dim3, perm0, perm1, perm2, perm3);
    }
}

// ---------------------------------------------------------------------------
// Mul: out = a * b
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float mul_f(float a, float b) { return a * b; }

void glm_mul(GlmCtx* ctx, void* out, const void* a, const void* b, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    ew_binary_2d_kernel<mul_f><<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
        n, 1, n, n);
}

void glm_mul_broadcast(GlmCtx* ctx, void* out, const void* a, const void* b, int dim, int rows) {
    cudaSetDevice(ctx->device_id);
    int total = rows * dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
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
        float v0, v1;
        load_bf16x2(row_ptr + i, v0, v1);
        sum += v0 + v1;
    }
    if ((cols & 1) && threadIdx.x == (cols / 2) % blockDim.x) {
        sum += __bfloat162float(row_ptr[cols - 1]);
    }
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    block_reduce_sum(sdata, threadIdx.x);
    if (threadIdx.x == 0) {
        out[row] = __float2bfloat16(sdata[0]);
    }
}

void glm_reduce_sum(GlmCtx* ctx, void* out, const void* input, int rows, int cols) {
    cudaSetDevice(ctx->device_id);
    int block_size = compute_block_size(cols);
    size_t shared_mem = block_size * sizeof(float);
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
// Expert-scale kernel
//   out:      [batch]       BF16
//   weights:  [batch, topK] BF16
//   indices:  [batch, topK] int32
//   For each batch b, find the first k where indices[b, k] == expert_id,
//   then out[b] = weights[b, k]. If not found, out[b] = 0.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 4) expert_scale_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* weights,
    const int* indices,
    int expert_id, int topK, int batch
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    float scale = 0.0f;
    for (int k = 0; k < topK; k++) {
        if (indices[b * topK + k] == expert_id) {
            scale = __bfloat162float(weights[b * topK + k]);
            break;
        }
    }
    out[b] = __float2bfloat16(scale);
}

void glm_expert_scale(GlmCtx* ctx, void* out, const void* weights,
                       const int* indices, int expert_id, int topK, int batch) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (batch + block_size - 1) / block_size;
    expert_scale_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)weights,
        indices, expert_id, topK, batch);
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
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * dim) return;
    int i = idx / dim;
    int d = idx % dim;
    int src_row = indices[i];
    out[idx] = src[src_row * dim + d];
}

void glm_index_select(GlmCtx* ctx, void* out, const void* src,
                       const void* indices, int dim, int k) {
    cudaSetDevice(ctx->device_id);
    int total = k * dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    index_select_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)src,
        (const int*)indices, dim, k);
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
        float v0, v1;
        load_bf16x2(row_in + i, v0, v1);
        if (v0 > my_max || (v0 == my_max && i < my_idx)) { my_max = v0; my_idx = i; }
        if (v1 > my_max || (v1 == my_max && (i + 1) < my_idx)) { my_max = v1; my_idx = i + 1; }
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
// Decode step bookkeeping kernel
// Increments position_ids, computes slot_mapping and last_page_len
// for batch decode. All tensors are device-side.
// position_ids: [batch_size] int32 — incremented by 1 in-place
// last_page_len: [batch_size] int32 — computed from position + page_size
// slot_mapping: [batch_size] int32 — computed from position + indices + indptr
// indptr: [batch_size + 1] int32 — page table indptr (read-only)
// indices: [max_pages] int32 — page table indices (read-only)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(128) decode_step_kernel(
    int32_t* position_ids,
    int32_t* last_page_len,
    int32_t* slot_mapping,
    const int32_t* indptr,
    const int32_t* indices,
    uint32_t page_size,
    uint32_t batch_size
) {
    uint32_t seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch_size) return;

    int32_t pos = position_ids[seq] + 1;
    position_ids[seq] = pos;

    int32_t kv_len = pos + 1;
    int32_t remainder = kv_len % (int32_t)page_size;
    last_page_len[seq] = (remainder != 0) ? remainder : (int32_t)page_size;

    int32_t page_idx = pos / (int32_t)page_size;
    int32_t page_offset = pos % (int32_t)page_size;
    int32_t seq_page_start = indptr[seq];
    int32_t abs_page = indices[seq_page_start + page_idx];
    slot_mapping[seq] = abs_page * (int32_t)page_size + page_offset;
}

void glm_decode_step(GlmCtx* ctx,
                      int32_t* position_ids,
                      int32_t* last_page_len,
                      int32_t* slot_mapping,
                      const int32_t* indptr,
                      const int32_t* indices,
                      uint32_t page_size,
                      uint32_t batch_size) {
    cudaSetDevice(ctx->device_id);
    dim3 grid((batch_size + 127) / 128);
    dim3 block(128);
    decode_step_kernel<<<grid, block, 0, GLM_STREAM(ctx)>>>(
        position_ids, last_page_len, slot_mapping,
        indptr, indices, page_size, batch_size);
}

// ---------------------------------------------------------------------------
// Scatter-add with row-wise scaling (atomic BF16 addition)
//   out:      [rows_out, dim]   BF16 (accumulated in-place)
//   input:    [count, dim]      BF16
//   scales:   [count]           BF16
//   batch_ids:[count]           int32
//   For each i in [0, count):
//     out[batch_ids[i], d] += scales[i] * input[i, d]  for all d
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256) scatter_add_rows_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* scales,
    const int* batch_ids,
    int dim, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    float accum = 0.0f;
    for (int i = 0; i < count; i++) {
        accum += __bfloat162float(scales[i]) * __bfloat162float(input[(size_t)i * dim + idx]);
    }
    out[idx] = __float2bfloat16(accum);
}

__global__ void __launch_bounds__(256, 4) scatter_add_rows_batched_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* scales,
    const int* batch_ids,
    int dim, int count, int num_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * dim;
    if (idx >= total) return;
    int row = idx / dim;
    int d = idx % dim;
    float accum = 0.0f;
    for (int i = 0; i < count; i++) {
        if (batch_ids[i] == row) {
            accum += __bfloat162float(scales[i]) * __bfloat162float(input[(size_t)i * dim + d]);
        }
    }
    out[idx] = __float2bfloat16(accum);
}

void glm_scatter_add_rows(GlmCtx* ctx, void* out, const void* input,
                            const void* scales, const int* batch_ids,
                            int dim, int count, int num_rows, void* workspace) {
    cudaSetDevice(ctx->device_id);
    (void)workspace;
    int block_size = 256;
    if (num_rows == 1) {
        int grid = (dim + block_size - 1) / block_size;
        scatter_add_rows_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
            (const __nv_bfloat16*)scales, batch_ids, dim, count);
    } else {
        int total = num_rows * dim;
        int grid = (total + block_size - 1) / block_size;
        scatter_add_rows_batched_kernel<<<grid, block_size, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
            (const __nv_bfloat16*)scales, batch_ids, dim, count, num_rows);
    }
}


