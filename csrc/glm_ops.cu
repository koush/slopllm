#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define CUBLAS(ctx) (*reinterpret_cast<cublasHandle_t*>(&(ctx)->cublas_handle))

// ---------------------------------------------------------------------------
// Context management
// ---------------------------------------------------------------------------

GlmCtx* glm_init(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_init: cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(err));
        return nullptr;
    }
    GlmCtx* ctx = new GlmCtx();
    ctx->device_id = device_id;
    cudaStreamCreate(&ctx->stream);
    cublasCreate(&CUBLAS(ctx));
    cublasSetStream(CUBLAS(ctx), ctx->stream);
    return ctx;
}

void glm_free(GlmCtx* ctx) {
    if (!ctx) return;
    cublasDestroy(CUBLAS(ctx));
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

// ---------------------------------------------------------------------------
// GPU memory management
// ---------------------------------------------------------------------------

void* glm_alloc(GlmCtx* ctx, size_t bytes) {
    void* ptr = nullptr;
    cudaSetDevice(ctx->device_id);
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_alloc: cudaMalloc(%zu) failed: %s\n", bytes, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void glm_free_buf(GlmCtx* ctx, void* ptr) {
    if (ptr) cudaFree(ptr);
}

void glm_h2d(GlmCtx* ctx, void* dst, const void* src, size_t bytes) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, ctx->stream);
}

void glm_d2h(GlmCtx* ctx, void* dst, const void* src, size_t bytes) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
}

// ---------------------------------------------------------------------------
// RMSNorm kernel
// ---------------------------------------------------------------------------

__global__ void rmsnorm_kernel(
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

    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        sum += val * val;
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xi = __bfloat162float(x[i]);
        float wi = __bfloat162float(weight[i]);
        o[i] = __float2bfloat16(wi * xi * inv_rms);
    }
}

void glm_rmsnorm(GlmCtx* ctx, void* out, const void* input,
                 const void* weight, float eps, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    if (block_size > dim) block_size = (dim + 31) / 32 * 32;
    size_t shared_mem = block_size * sizeof(float);
    rmsnorm_kernel<<<batch, block_size, shared_mem, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)weight, eps, dim);
}

// ---------------------------------------------------------------------------
// SiLU + Mul kernel
// ---------------------------------------------------------------------------

__global__ void silu_and_mul_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float g = __bfloat162float(gate[idx]);
        float u = __bfloat162float(up[idx]);
        float sig = 1.0f / (1.0f + expf(-g));
        out[idx] = __float2bfloat16(g * sig * u);
    }
}

void glm_silu_and_mul(GlmCtx* ctx, void* out, const void* gate,
                      const void* up, int intermediate, int batch) {
    cudaSetDevice(ctx->device_id);
    int total = batch * intermediate;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    silu_and_mul_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up, total);
}

// ---------------------------------------------------------------------------
// Linear (BF16 GEMM via cuBLAS)
//   out = input @ weight.T
//   input:  [batch, k]  row-major BF16
//   weight: [n, k]      row-major BF16
//   out:    [batch, n]  row-major BF16
// ---------------------------------------------------------------------------

void glm_linear(GlmCtx* ctx, void* out, const void* input,
                const void* weight, int batch, int n, int k) {
    cudaSetDevice(ctx->device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major GEMM: C[b,n] = A[b,k] @ B[n,k]^T
    // cuBLAS col-major: C^T[n,b] = B[n,k] @ A^T[k,b]
    // cublasGemmEx: op(A)=B, op(B)=A^T
    //   op(A): [n, k], transa=N, lda=k
    //   op(B): [k, b], transb=T, ldb=k
    //   C:    [n, b], ldc=n
    cublasGemmEx(CUBLAS(ctx),
        CUBLAS_OP_T,      // transa: A^T from row-major input
        CUBLAS_OP_N,       // transb: weight stays as-is
        n,                 // m of output
        batch,             // n of output
        k,                 // k (inner dim)
        &alpha,
        weight, CUDA_R_16BF, k,    // A = weight [n,k] row-maj -> col-maj [k,n], op(A)=T -> [n,k]
        input,  CUDA_R_16BF, k,    // B = input [b,k] row-maj -> col-maj [k,b], op(B)=N -> [k,b]
        &beta,
        out,    CUDA_R_16BF, n,    // C = out [b,n] row-maj -> col-maj [n,b]
        CUDA_R_32F,                  // compute type: FP32 accumulation
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ---------------------------------------------------------------------------
// Embedding lookup kernel
// ---------------------------------------------------------------------------

__global__ void embedding_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* table,
    const int* ids,
    int hidden,
    int seq_len
) {
    int token_idx = blockIdx.x;
    if (token_idx >= seq_len) return;
    int id = ids[token_idx];
    const __nv_bfloat16* src = table + id * hidden;
    __nv_bfloat16* dst = out + token_idx * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void glm_embedding(GlmCtx* ctx, void* out, const void* table,
                   const int* ids, int hidden, int seq_len) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    embedding_kernel<<<seq_len, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)table,
        ids, hidden, seq_len);
}

// ---------------------------------------------------------------------------
// LayerNorm kernel (with bias)
// ---------------------------------------------------------------------------

__global__ void layernorm_kernel(
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
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        mean += __bfloat162float(x[i]);
    }
    sdata[threadIdx.x] = mean;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    mean = sdata[0] / dim;

    float var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = __bfloat162float(x[i]) - mean;
        var += d * d;
    }
    sdata[threadIdx.x] = var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float xi = __bfloat162float(x[i]);
        float wi = __bfloat162float(weight[i]);
        float bi = bias ? __bfloat162float(bias[i]) : 0.0f;
        o[i] = __float2bfloat16(wi * (xi - mean) * inv_std + bi);
    }
}

void glm_layernorm(GlmCtx* ctx, void* out, const void* input,
                   const void* weight, const void* bias, float eps, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    if (block_size > dim) block_size = (dim + 31) / 32 * 32;
    size_t shared_mem = block_size * sizeof(float);
    layernorm_kernel<<<batch, block_size, shared_mem, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)weight, (const __nv_bfloat16*)bias,
        eps, dim);
}

// ---------------------------------------------------------------------------
// ReLU kernel
// ---------------------------------------------------------------------------

__global__ void relu_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(input[idx]);
        out[idx] = __float2bfloat16(fmaxf(v, 0.0f));
    }
}

void glm_relu(GlmCtx* ctx, void* out, const void* input, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    relu_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input, n);
}

// ---------------------------------------------------------------------------
// Sigmoid kernel
// ---------------------------------------------------------------------------

__global__ void sigmoid_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(input[idx]);
        out[idx] = __float2bfloat16(1.0f / (1.0f + expf(-v)));
    }
}

void glm_sigmoid(GlmCtx* ctx, void* out, const void* input, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    sigmoid_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input, n);
}

// ---------------------------------------------------------------------------
// Softmax kernel (row-wise, with optional mask)
// input: [batch, dim], mask: [batch, dim] or NULL, out: [batch, dim]
// mask values: 0.0 = keep, -inf = masked out
// ---------------------------------------------------------------------------

__global__ void softmax_kernel(
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

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        if (m) val += __bfloat162float(m[i]);
        if (val > max_val) max_val = val;
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        if (m) val += __bfloat162float(m[i]);
        sum += expf(val - max_val);
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        if (m) val += __bfloat162float(m[i]);
        o[i] = __float2bfloat16(expf(val - max_val) * inv_sum);
    }
}

void glm_softmax(GlmCtx* ctx, void* out, const void* input,
                 const void* mask, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    if (block_size > dim) block_size = (dim + 31) / 32 * 32;
    size_t shared_mem = block_size * sizeof(float);
    softmax_kernel<<<batch, block_size, shared_mem, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        mask ? (const __nv_bfloat16*)mask : nullptr,
        dim);
}

// ---------------------------------------------------------------------------
// Causal mask kernel
// Fills upper triangle with -inf: out[i][j] = (j > i) ? -inf : 0
// out: [seq_len, seq_len] BF16
// ---------------------------------------------------------------------------

__global__ void causal_mask_kernel(
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
    causal_mask_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, seq_len);
}

// ---------------------------------------------------------------------------
// Fill kernel
// ---------------------------------------------------------------------------

__global__ void fill_kernel(__nv_bfloat16* out, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(value);
    }
}

void glm_fill(GlmCtx* ctx, void* out, float value, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    fill_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, value, n);
}

// ---------------------------------------------------------------------------
// Gather kernel (along last dim)
// out[b, i] = input[b, indices[b, i]]
// ---------------------------------------------------------------------------

__global__ void gather_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
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
        out[idx] = input[b * in_dim + col];
    }
}

void glm_gather(GlmCtx* ctx, void* out, const void* input, const int* indices,
                int k, int in_dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int total = batch * k;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    gather_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        indices, k, in_dim, batch);
}

// ---------------------------------------------------------------------------
// Scatter scalar kernel (along last dim)
// out[b, indices[b, i]] = value
// ---------------------------------------------------------------------------

__global__ void scatter_scalar_kernel(
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
    scatter_scalar_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, indices, value, k, out_dim, batch);
}

// ---------------------------------------------------------------------------
// Cat last dim kernel
// out[i, :a_dim] = a[i, :], out[i, a_dim:] = b[i, :]
// ---------------------------------------------------------------------------

__global__ void cat_last_dim_kernel(
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
    cat_last_dim_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)a,
        (const __nv_bfloat16*)b, a_last_dim, b_last_dim, outer);
}

// ---------------------------------------------------------------------------
// Masked fill kernel
// out[i] = (mask[i] != 0) ? value : input[i]
// ---------------------------------------------------------------------------

__global__ void masked_fill_kernel(
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
    masked_fill_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)mask, value, n);
}

// ---------------------------------------------------------------------------
// Index add kernel (along first dim)
// out[indices[i], d] += values[i, d]  for each i, d
// NOTE: indices must be unique within a call for correctness.
// ---------------------------------------------------------------------------

__global__ void index_add_kernel(
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
    index_add_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, indices, (const __nv_bfloat16*)values,
        n_indices, dim);
}

// ---------------------------------------------------------------------------
// Rotary embedding kernel
// Computes cos/sin from inv_freq and position_ids
// cos_out, sin_out: [batch, seq_len, dim]  where dim = dim_half * 2
// inv_freq: [dim_half], position_ids: [batch, seq_len] (int32)
// ---------------------------------------------------------------------------

__global__ void rotary_embedding_kernel(
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
        cos_out[idx] = __float2bfloat16(cosf(freq));
        sin_out[idx] = __float2bfloat16(sinf(freq));
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
    rotary_embedding_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)cos_out, (__nv_bfloat16*)sin_out,
        (const __nv_bfloat16*)inv_freq, position_ids,
        dim_half, seq_len, batch);
}

// ---------------------------------------------------------------------------
// Apply rotary position embedding kernel
// out = x * cos + rotate_half(x) * sin
// rotate_half(x)[..., d] = (d < half) ? -x[..., d+half] : x[..., d-half]
//
// unsqueeze_dim=1: x is [batch, n_heads, seq_len, rope_dim]
// unsqueeze_dim=2: x is [batch, seq_len, n_heads, rope_dim]
// cos, sin are [batch, seq_len, rope_dim] (broadcast over heads)
// ---------------------------------------------------------------------------

__global__ void apply_rotary_pos_emb_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* cos_emb,
    const __nv_bfloat16* sin_emb,
    int rope_dim,
    int seq_len,
    int n_heads,
    int batch,
    int unsqueeze_dim
) {
    int total = batch * n_heads * seq_len * rope_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % rope_dim;
    int b, s;
    if (unsqueeze_dim == 1) {
        int rest = idx / rope_dim;
        s = rest % seq_len;
        b = rest / (n_heads * seq_len);
    } else {
        int rest = idx / rope_dim;
        s = (rest / n_heads) % seq_len;
        b = rest / (n_heads * seq_len);
    }

    int cos_idx = (b * seq_len + s) * rope_dim + d;
    int half = rope_dim / 2;

    float x_val = __bfloat162float(x[idx]);
    float cos_val = __bfloat162float(cos_emb[cos_idx]);
    float sin_val = __bfloat162float(sin_emb[cos_idx]);

    float x_rot;
    if (d < half) {
        x_rot = -__bfloat162float(x[idx + half]);
    } else {
        x_rot = __bfloat162float(x[idx - half]);
    }

    out[idx] = __float2bfloat16(x_val * cos_val + x_rot * sin_val);
}

void glm_apply_rotary_pos_emb(GlmCtx* ctx, void* out, const void* x,
                              const void* cos, const void* sin,
                              int rope_dim, int n_heads, int seq_len,
                              int batch, int unsqueeze_dim) {
    cudaSetDevice(ctx->device_id);
    int total = batch * n_heads * seq_len * rope_dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    apply_rotary_pos_emb_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin,
        rope_dim, seq_len, n_heads, batch, unsqueeze_dim);
}

// ---------------------------------------------------------------------------
// TopK kernel (along last dim, unsorted)
// One block per row. Loads row into shared memory, finds top-k by
// k passes of parallel argmax.
// Shared memory: dim * (sizeof(float) + sizeof(int)) + blockDim * (sizeof(float) + sizeof(int))
// ---------------------------------------------------------------------------

__global__ void topk_kernel(
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
            if (s_vals[i] >= my_max) {
                my_max = s_vals[i];
                my_idx = i;
            }
        }
        s_reduce[tid] = my_max;
        s_reduce_idx[tid] = my_idx;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_reduce[tid + s] > s_reduce[tid]) {
                    s_reduce[tid] = s_reduce[tid + s];
                    s_reduce_idx[tid] = s_reduce_idx[tid + s];
                }
            }
            __syncthreads();
        }

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
    topk_kernel<<<grid, block_size, shared_mem, ctx->stream>>>(
        (__nv_bfloat16*)out_values, out_indices,
        (const __nv_bfloat16*)input, k, dim, batch);
}

// ---------------------------------------------------------------------------
// Batched matrix multiply (contiguous row-major tensors)
// transB=1: C[b] = alpha * A[b] @ B[b]^T + beta * C[b]
//   A: [batch, M, K],  B: [batch, N, K],  C: [batch, M, N]
// transB=0: C[b] = alpha * A[b] @ B[b] + beta * C[b]
//   A: [batch, M, K],  B: [batch, K, N],  C: [batch, M, N]
// ---------------------------------------------------------------------------

void glm_bmm(GlmCtx* ctx, void* C, const void* A, const void* B,
             float alpha, float beta,
             int batch, int M, int N, int K, int transB) {
    cudaSetDevice(ctx->device_id);
    if (transB) {
        // A @ B^T:  A=[M,K], B=[N,K], C=[M,N]
        // cuBLAS col-major: op(A_gemm)=B^T[N,K], op(B_gemm)=A[K,M]
        // transa=T, transb=N, m=N, n=M, k=K
        long long strideA = (long long)M * K;
        long long strideB = (long long)N * K;
        long long strideC = (long long)M * N;
        cublasGemmStridedBatchedEx(CUBLAS(ctx),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, CUDA_R_16BF, K, strideB,
            A, CUDA_R_16BF, K, strideA,
            &beta,
            C, CUDA_R_16BF, N, strideC,
            batch,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        // A @ B:  A=[M,K], B=[K,N], C=[M,N]
        // cuBLAS col-major: op(A_gemm)=B[N,K], op(B_gemm)=A[K,M]
        // transa=N, transb=N, m=N, n=M, k=K
        long long strideA = (long long)M * K;
        long long strideB = (long long)K * N;
        long long strideC = (long long)M * N;
        cublasGemmStridedBatchedEx(CUBLAS(ctx),
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, CUDA_R_16BF, N, strideB,
            A, CUDA_R_16BF, K, strideA,
            &beta,
            C, CUDA_R_16BF, N, strideC,
            batch,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

// ---------------------------------------------------------------------------
// Scale kernel: out = input * scale
// ---------------------------------------------------------------------------

__global__ void scale_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(__bfloat162float(input[idx]) * scale);
    }
}

void glm_scale(GlmCtx* ctx, void* out, const void* input, float scale, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    scale_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input, scale, n);
}

// ---------------------------------------------------------------------------
// Add kernel: out = a + b
// ---------------------------------------------------------------------------

__global__ void add_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

void glm_add(GlmCtx* ctx, void* out, const void* a, const void* b, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    add_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, n);
}

// ---------------------------------------------------------------------------
// Expand/repeat along dim 1 of a 4D tensor
// input:  [batch, dim1_in, seq_len, head_dim]
// output: [batch, dim1_out, seq_len, head_dim]
// out[b, h_out, s, d] = input[b, h_out * dim1_in / dim1_out, s, d]
// Requires dim1_out % dim1_in == 0
// ---------------------------------------------------------------------------

__global__ void expand_dim1_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int dim1_out,
    int dim1_in,
    int seq_len,
    int head_dim,
    int batch,
    int head_stride
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

        int h_in = h_out * dim1_in / dim1_out;
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
    expand_dim1_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        dim1_out, dim1_in, seq_len, head_dim, batch, seq_len * head_dim);
}

void glm_expand_dim1_strided(GlmCtx* ctx, void* out, const void* input,
                             int dim1_out, int dim1_in, int seq_len, int head_dim,
                             int batch, int head_stride) {
    cudaSetDevice(ctx->device_id);
    int total = batch * dim1_out * seq_len * head_dim;
    int block_size = 256;
    int grid = (total + block_size - 1) / block_size;
    expand_dim1_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        dim1_out, dim1_in, seq_len, head_dim, batch, head_stride);
}

// ---------------------------------------------------------------------------
// General 4D transpose kernel
// input:  [dim0, dim1, dim2, dim3] with row-major layout
// output: [dim_perm0, dim_perm1, dim_perm2, dim_perm3]
// perm[0..3] specifies the output dimension order
// e.g., perm={0,2,1,3} swaps dims 1 and 2
// ---------------------------------------------------------------------------

__global__ void transpose_4d_kernel(
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
    transpose_4d_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input,
        dim0, dim1, dim2, dim3, perm0, perm1, perm2, perm3);
}

// ---------------------------------------------------------------------------
// Element-wise multiply kernel
//   out[i] = a[i] * b[i]  (BF16, FP32 accumulation)
// ---------------------------------------------------------------------------

__global__ void mul_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(va * vb);
    }
}

void glm_mul(GlmCtx* ctx, void* out, const void* a, const void* b, int n) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    mul_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)a,
        (const __nv_bfloat16*)b, n);
}

// ---------------------------------------------------------------------------
// Reduce-sum along last dimension kernel
//   input:  [rows, cols]  BF16
//   output: [rows]        BF16
//   out[i] = sum(input[i, :])
// ---------------------------------------------------------------------------

__global__ void reduce_sum_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int cols
) {
    int row = blockIdx.x;
    const __nv_bfloat16* row_ptr = input + row * cols;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        sum += __bfloat162float(row_ptr[c]);
    }
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = __float2bfloat16(sdata[0]);
    }
}

void glm_reduce_sum(GlmCtx* ctx, void* out, const void* input, int rows, int cols) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    if (block_size > cols) block_size = (cols + 31) / 32 * 32;
    if (block_size < 32) block_size = 32;
    size_t shared_mem = block_size * sizeof(float);
    reduce_sum_kernel<<<rows, block_size, shared_mem, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)input, cols);
}

// ---------------------------------------------------------------------------
// Index-select kernel (gather rows by index)
//   src:      [src_rows, dim]  BF16
//   indices:  [k]              int32
//   out:      [k, dim]         BF16
//   out[i, :] = src[indices[i], :]
// ---------------------------------------------------------------------------

__global__ void index_select_kernel(
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
    index_select_kernel<<<grid, block_size, 0, ctx->stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)src,
        (const int*)indices, dim, k);
}

// ---------------------------------------------------------------------------
// Argmax kernel (find index of max value per row in BF16 input)
//   out_indices[row] = argmax(input[row * dim : (row+1) * dim])
// ---------------------------------------------------------------------------

__global__ void argmax_kernel(int* out_indices, const __nv_bfloat16* input,
                              int dim, int batch) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const __nv_bfloat16* row_in = input + row * dim;

    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int* s_idxs = reinterpret_cast<int*>(s_vals + blockDim.x);

    float my_max = -INFINITY;
    int my_idx = -1;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        if (val > my_max) {
            my_max = val;
            my_idx = i;
        }
    }
    s_vals[threadIdx.x] = my_max;
    s_idxs[threadIdx.x] = my_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_vals[threadIdx.x + s] > s_vals[threadIdx.x]) {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + s];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_indices[row] = s_idxs[0];
    }
}

void glm_argmax(GlmCtx* ctx, int* out_index, const void* input, int dim, int batch) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = batch;
    size_t shared_mem = block_size * sizeof(float) + block_size * sizeof(int);
    argmax_kernel<<<grid, block_size, shared_mem, ctx->stream>>>(
        out_index, (const __nv_bfloat16*)input, dim, batch);
}

// ---------------------------------------------------------------------------
// Arange kernel (fill int32 buffer with sequential values)
//   out[i] = start + i * step
// ---------------------------------------------------------------------------

__global__ void arange_kernel(int* out, int start, int step, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = start + idx * step;
    }
}

void glm_arange(GlmCtx* ctx, int* out, int start, int step, int count) {
    cudaSetDevice(ctx->device_id);
    int block_size = 256;
    int grid = (count + block_size - 1) / block_size;
    arange_kernel<<<grid, block_size, 0, ctx->stream>>>(out, start, step, count);
}

// ---------------------------------------------------------------------------
// Device-to-device memory copy
// ---------------------------------------------------------------------------

void glm_memcpy(GlmCtx* ctx, void* dst, const void* src, size_t bytes) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, ctx->stream);
}

// ---------------------------------------------------------------------------
// Stream synchronization
// ---------------------------------------------------------------------------

void glm_synchronize(GlmCtx* ctx) {
    cudaStreamSynchronize(ctx->stream);
}
