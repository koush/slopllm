#include "glm_ops.h"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

__device__ __forceinline__ float bf162float(const nv_bfloat16& v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __bfloat162float(v);
#else
    return __half2float(__ushort_as_half(*(const uint16_t*)&v));
#endif
}

__device__ __forceinline__ nv_bfloat16 float2bf16(float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __float2bfloat16(v);
#else
    nv_bfloat16 r;
    *(uint16_t*)&r = __half_as_ushort(__float2half(v));
    return r;
#endif
}

__device__ __forceinline__ float softplus_f(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return logf(1.0f + expf(x));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_f(float x) {
    return x * sigmoid_f(x);
}

__device__ float block_reduce_sum(float val, float* s_partial, int tid, int blockDimX) {
    s_partial[tid] = val;
    __syncthreads();
    for (int stride = blockDimX / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }
    return s_partial[0];
}

__global__ void gdn_recurrent_step_kernel(
    nv_bfloat16* __restrict__ output,
    float* __restrict__ state,
    const nv_bfloat16* __restrict__ q,
    const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v,
    const nv_bfloat16* __restrict__ a_raw,
    const nv_bfloat16* __restrict__ b_raw,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    int num_heads, int d_k, int d_v
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int blockDimX = blockDim.x;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_k = s_q + d_k;
    float* s_v = s_k + d_k;
    float* s_v_old = s_v + d_v;
    float* s_delta = s_v_old + d_v;
    float* s_partial = s_delta + d_v;

    for (int i = tid; i < d_k; i += blockDimX) {
        s_q[i] = bf162float(q[h * d_k + i]);
        s_k[i] = bf162float(k[h * d_k + i]);
    }
    for (int i = tid; i < d_v; i += blockDimX) {
        s_v[i] = bf162float(v[h * d_v + i]);
    }
    __syncthreads();

    float q_norm_sq = 0.0f;
    for (int i = tid; i < d_k; i += blockDimX) {
        q_norm_sq += s_q[i] * s_q[i];
    }
    q_norm_sq = block_reduce_sum(q_norm_sq, s_partial, tid, blockDimX);
    float q_scale = 1.0f / sqrtf((float)d_k);
    float q_inv_norm = rsqrtf(q_norm_sq + 1e-8f);
    for (int i = tid; i < d_k; i += blockDimX) {
        s_q[i] *= q_inv_norm * q_scale;
    }

    float k_norm_sq = 0.0f;
    for (int i = tid; i < d_k; i += blockDimX) {
        k_norm_sq += s_k[i] * s_k[i];
    }
    k_norm_sq = block_reduce_sum(k_norm_sq, s_partial, tid, blockDimX);
    float k_inv_norm = rsqrtf(k_norm_sq + 1e-8f);
    for (int i = tid; i < d_k; i += blockDimX) {
        s_k[i] *= k_inv_norm;
    }
    __syncthreads();

    float beta, decay;
    if (tid == 0) {
        float a = bf162float(a_raw[h]);
        float b = bf162float(b_raw[h]);
        float al = A_log[h];
        float dtb = dt_bias[h];
        s_partial[0] = sigmoid_f(b);
        s_partial[1] = expf(-expf(al) * softplus_f(a + dtb));
    }
    __syncthreads();
    beta = s_partial[0];
    decay = s_partial[1];

    float* state_h = state + h * d_k * d_v;

    for (int idx = tid; idx < d_k * d_v; idx += blockDimX) {
        state_h[idx] *= decay;
    }
    __syncthreads();

    for (int i = tid; i < d_v; i += blockDimX) {
        float sum = 0.0f;
        for (int j = 0; j < d_k; j++) {
            sum += state_h[j * d_v + i] * s_k[j];
        }
        s_v_old[i] = sum;
    }
    __syncthreads();

    for (int i = tid; i < d_v; i += blockDimX) {
        s_delta[i] = beta * (s_v[i] - s_v_old[i]);
    }
    __syncthreads();

    for (int j = tid; j < d_k; j += blockDimX) {
        float kj = s_k[j];
        for (int i = 0; i < d_v; i++) {
            state_h[j * d_v + i] += kj * s_delta[i];
        }
    }
    __syncthreads();

    for (int i = tid; i < d_v; i += blockDimX) {
        float sum = 0.0f;
        for (int j = 0; j < d_k; j++) {
            sum += state_h[j * d_v + i] * s_q[j];
        }
        output[h * d_v + i] = float2bf16(sum);
    }
}

void glm_gdn_recurrent_step(
    GlmCtx* ctx,
    void* output,
    void* state,
    const void* q,
    const void* k,
    const void* v,
    const void* a_raw,
    const void* b_raw,
    const float* A_log,
    const float* dt_bias,
    int num_heads, int d_k, int d_v
) {
    int smem_size = (2 * d_k + 3 * d_v + 128) * sizeof(float);
    gdn_recurrent_step_kernel<<<num_heads, 128, smem_size, ctx->stream>>>(
        (nv_bfloat16*)output,
        (float*)state,
        (const nv_bfloat16*)q,
        (const nv_bfloat16*)k,
        (const nv_bfloat16*)v,
        (const nv_bfloat16*)a_raw,
        (const nv_bfloat16*)b_raw,
        A_log,
        dt_bias,
        num_heads, d_k, d_v
    );
}

__global__ void gdn_prefill_kernel(
    nv_bfloat16* __restrict__ output,
    float* __restrict__ state,
    const nv_bfloat16* __restrict__ q,
    const nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ v,
    const nv_bfloat16* __restrict__ a_raw,
    const nv_bfloat16* __restrict__ b_raw,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    int seq_len, int num_heads, int d_k, int d_v
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int blockDimX = blockDim.x;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_k = s_q + d_k;
    float* s_v = s_k + d_k;
    float* s_v_old = s_v + d_v;
    float* s_delta = s_v_old + d_v;
    float* s_partial = s_delta + d_v;

    float* state_h = state + h * d_k * d_v;
    float al = A_log[h];
    float dtb = dt_bias[h];
    float neg_exp_al = -expf(al);

    for (int t = 0; t < seq_len; t++) {
        for (int i = tid; i < d_k; i += blockDimX) {
            s_q[i] = bf162float(q[t * num_heads * d_k + h * d_k + i]);
            s_k[i] = bf162float(k[t * num_heads * d_k + h * d_k + i]);
        }
        for (int i = tid; i < d_v; i += blockDimX) {
            s_v[i] = bf162float(v[t * num_heads * d_v + h * d_v + i]);
        }
        __syncthreads();

        float q_norm_sq = 0.0f;
        for (int i = tid; i < d_k; i += blockDimX) {
            q_norm_sq += s_q[i] * s_q[i];
        }
        q_norm_sq = block_reduce_sum(q_norm_sq, s_partial, tid, blockDimX);
        float q_scale = 1.0f / sqrtf((float)d_k);
        float q_inv_norm = rsqrtf(q_norm_sq + 1e-8f);
        for (int i = tid; i < d_k; i += blockDimX) {
            s_q[i] *= q_inv_norm * q_scale;
        }

        float k_norm_sq = 0.0f;
        for (int i = tid; i < d_k; i += blockDimX) {
            k_norm_sq += s_k[i] * s_k[i];
        }
        k_norm_sq = block_reduce_sum(k_norm_sq, s_partial, tid, blockDimX);
        float k_inv_norm = rsqrtf(k_norm_sq + 1e-8f);
        for (int i = tid; i < d_k; i += blockDimX) {
            s_k[i] *= k_inv_norm;
        }
        __syncthreads();

        float beta, decay;
        if (tid == 0) {
            float a = bf162float(a_raw[t * num_heads + h]);
            float b = bf162float(b_raw[t * num_heads + h]);
            s_partial[0] = sigmoid_f(b);
            s_partial[1] = expf(neg_exp_al * softplus_f(a + dtb));
        }
        __syncthreads();
        beta = s_partial[0];
        decay = s_partial[1];

        for (int idx = tid; idx < d_k * d_v; idx += blockDimX) {
            state_h[idx] *= decay;
        }
        __syncthreads();

        for (int i = tid; i < d_v; i += blockDimX) {
            float sum = 0.0f;
            for (int j = 0; j < d_k; j++) {
                sum += state_h[j * d_v + i] * s_k[j];
            }
            s_v_old[i] = sum;
        }
        __syncthreads();

        for (int i = tid; i < d_v; i += blockDimX) {
            s_delta[i] = beta * (s_v[i] - s_v_old[i]);
        }
        __syncthreads();

        for (int j = tid; j < d_k; j += blockDimX) {
            float kj = s_k[j];
            for (int i = 0; i < d_v; i++) {
                state_h[j * d_v + i] += kj * s_delta[i];
            }
        }
        __syncthreads();

        for (int i = tid; i < d_v; i += blockDimX) {
            float sum = 0.0f;
            for (int j = 0; j < d_k; j++) {
                sum += state_h[j * d_v + i] * s_q[j];
            }
            output[t * num_heads * d_v + h * d_v + i] = float2bf16(sum);
        }
        __syncthreads();
    }
}

void glm_gdn_prefill(
    GlmCtx* ctx,
    void* output,
    void* state,
    const void* q,
    const void* k,
    const void* v,
    const void* a_raw,
    const void* b_raw,
    const float* A_log,
    const float* dt_bias,
    int seq_len, int num_heads, int d_k, int d_v
) {
    int smem_size = (2 * d_k + 3 * d_v + 128) * sizeof(float);
    gdn_prefill_kernel<<<num_heads, 128, smem_size, ctx->stream>>>(
        (nv_bfloat16*)output,
        (float*)state,
        (const nv_bfloat16*)q,
        (const nv_bfloat16*)k,
        (const nv_bfloat16*)v,
        (const nv_bfloat16*)a_raw,
        (const nv_bfloat16*)b_raw,
        A_log,
        dt_bias,
        seq_len, num_heads, d_k, d_v
    );
}

__global__ void causal_conv1d_kernel(
    nv_bfloat16* __restrict__ output,
    nv_bfloat16* __restrict__ conv_state,
    const nv_bfloat16* __restrict__ input,
    const nv_bfloat16* __restrict__ weight,
    int conv_dim, int seq_len, int kernel_size
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;

    const nv_bfloat16* w = weight + c * kernel_size;
    const nv_bfloat16* x = input + c * seq_len;
    nv_bfloat16* out = output + c * seq_len;

    for (int t = 0; t < seq_len; t++) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int xt = t - (kernel_size - 1) + k;
            float xv = (xt >= 0) ? bf162float(x[xt]) : 0.0f;
            sum += bf162float(w[k]) * xv;
        }
        out[t] = float2bf16(silu_f(sum));
    }

    if (conv_state != nullptr) {
        int state_len = kernel_size - 1;
        nv_bfloat16* cs = conv_state + c * state_len;
        for (int i = 0; i < state_len; i++) {
            int src_t = seq_len - state_len + i;
            cs[i] = (src_t >= 0) ? x[src_t] : float2bf16(0.0f);
        }
    }
}

void glm_causal_conv1d(
    GlmCtx* ctx,
    void* output,
    void* conv_state,
    const void* input,
    const void* weight,
    int conv_dim, int seq_len, int kernel_size
) {
    int threads = 256;
    int blocks = (conv_dim + threads - 1) / threads;
    causal_conv1d_kernel<<<blocks, threads, 0, ctx->stream>>>(
        (nv_bfloat16*)output,
        (nv_bfloat16*)conv_state,
        (const nv_bfloat16*)input,
        (const nv_bfloat16*)weight,
        conv_dim, seq_len, kernel_size
    );
}

__global__ void causal_conv1d_update_kernel(
    nv_bfloat16* __restrict__ output,
    nv_bfloat16* __restrict__ conv_state,
    const nv_bfloat16* __restrict__ input,
    const nv_bfloat16* __restrict__ weight,
    int conv_dim, int kernel_size
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;

    int state_len = kernel_size - 1;
    nv_bfloat16* cs = conv_state + c * state_len;
    const nv_bfloat16* w = weight + c * kernel_size;
    float x_new = bf162float(input[c]);

    float sum = bf162float(w[kernel_size - 1]) * x_new;
    for (int k = 0; k < state_len; k++) {
        sum += bf162float(w[k]) * bf162float(cs[k]);
    }

    output[c] = float2bf16(silu_f(sum));

    for (int k = 0; k < state_len - 1; k++) {
        cs[k] = cs[k + 1];
    }
    if (state_len > 0) {
        cs[state_len - 1] = float2bf16(x_new);
    }
}

void glm_causal_conv1d_update(
    GlmCtx* ctx,
    void* output,
    void* conv_state,
    const void* input,
    const void* weight,
    int conv_dim, int kernel_size
) {
    int threads = 256;
    int blocks = (conv_dim + threads - 1) / threads;
    causal_conv1d_update_kernel<<<blocks, threads, 0, ctx->stream>>>(
        (nv_bfloat16*)output,
        (nv_bfloat16*)conv_state,
        (const nv_bfloat16*)input,
        (const nv_bfloat16*)weight,
        conv_dim, kernel_size
    );
}

__global__ void rmsnorm_gated_kernel(
    nv_bfloat16* __restrict__ output,
    const nv_bfloat16* __restrict__ input,
    const nv_bfloat16* __restrict__ gate,
    const nv_bfloat16* __restrict__ weight,
    float eps, int dim, int batch
) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float s_partial[];

    const nv_bfloat16* x = input + idx * dim;
    const nv_bfloat16* g = gate + idx * dim;
    nv_bfloat16* out = output + idx * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float xi = bf162float(x[i]);
        sum_sq += xi * xi;
    }
    sum_sq = block_reduce_sum(sum_sq, s_partial, tid, blockDim.x);

    float inv_rms = rsqrtf(sum_sq / dim + eps);

    for (int i = tid; i < dim; i += blockDim.x) {
        float xi = bf162float(x[i]);
        float gi = bf162float(g[i]);
        float wi = bf162float(weight[i]);
        float normed = xi * inv_rms * wi;
        float silu_gi = silu_f(gi);
        out[i] = float2bf16(normed * silu_gi);
    }
}

void glm_rmsnorm_gated(
    GlmCtx* ctx,
    void* output,
    const void* input,
    const void* gate,
    const void* weight,
    float eps, int dim, int batch
) {
    int smem_size = 128 * sizeof(float);
    rmsnorm_gated_kernel<<<batch, 128, smem_size, ctx->stream>>>(
        (nv_bfloat16*)output,
        (const nv_bfloat16*)input,
        (const nv_bfloat16*)gate,
        (const nv_bfloat16*)weight,
        eps, dim, batch
    );
}

__global__ void qkv_split_kernel(
    nv_bfloat16* __restrict__ q_out,
    nv_bfloat16* __restrict__ k_out,
    nv_bfloat16* __restrict__ v_out,
    const nv_bfloat16* __restrict__ qkv_in,
    int seq_len, int num_heads, int d_k, int d_v
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k_total = num_heads * d_k;
    int v_total = num_heads * d_v;
    int q_size = seq_len * k_total;
    int k_size = seq_len * k_total;
    int v_size = seq_len * v_total;
    int total = q_size + k_size + v_size;

    if (tid >= total) return;

    if (tid < q_size) {
        int d = tid % d_k;
        int rest = tid / d_k;
        int h = rest % num_heads;
        int t = rest / num_heads;
        int src_channel = h * d_k + d;
        q_out[tid] = qkv_in[src_channel * seq_len + t];
    } else if (tid < q_size + k_size) {
        int kid = tid - q_size;
        int d = kid % d_k;
        int rest = kid / d_k;
        int h = rest % num_heads;
        int t = rest / num_heads;
        int src_channel = k_total + h * d_k + d;
        k_out[kid] = qkv_in[src_channel * seq_len + t];
    } else {
        int vid = tid - q_size - k_size;
        int d = vid % d_v;
        int rest = vid / d_v;
        int h = rest % num_heads;
        int t = rest / num_heads;
        int src_channel = 2 * k_total + h * d_v + d;
        v_out[vid] = qkv_in[src_channel * seq_len + t];
    }
}

void glm_qkv_split(
    GlmCtx* ctx,
    void* q_out, void* k_out, void* v_out,
    const void* qkv_in,
    int seq_len, int num_heads, int d_k, int d_v
) {
    int k_total = num_heads * d_k;
    int v_total = num_heads * d_v;
    int q_size = seq_len * k_total;
    int k_size = seq_len * k_total;
    int v_size = seq_len * v_total;
    int total = q_size + k_size + v_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_kernel<<<blocks, threads, 0, ctx->stream>>>(
        (nv_bfloat16*)q_out,
        (nv_bfloat16*)k_out,
        (nv_bfloat16*)v_out,
        (nv_bfloat16*)qkv_in,
        seq_len, num_heads, d_k, d_v
    );
}

__global__ void interleaved_split_kernel(
    nv_bfloat16* __restrict__ q_out,
    nv_bfloat16* __restrict__ gate_out,
    const nv_bfloat16* __restrict__ qg_in,
    int batch_seq, int num_heads, int head_dim
) {
    int total = batch_seq * num_heads * head_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int hd2 = head_dim * 2;
    int s = tid / (num_heads * head_dim);
    int rest = tid % (num_heads * head_dim);
    int h = rest / head_dim;
    int d = rest % head_dim;

    int src_offset = s * num_heads * hd2 + h * hd2 + d;
    int gate_src_offset = src_offset + head_dim;
    int dst_offset = s * num_heads * head_dim + h * head_dim + d;

    q_out[dst_offset] = qg_in[src_offset];
    gate_out[dst_offset] = qg_in[gate_src_offset];
}

void glm_interleaved_split(
    GlmCtx* ctx,
    void* q_out,
    void* gate_out,
    const void* qg_in,
    int batch_seq, int num_heads, int head_dim
) {
    int total = batch_seq * num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    interleaved_split_kernel<<<blocks, threads, 0, ctx->stream>>>(
        (nv_bfloat16*)q_out,
        (nv_bfloat16*)gate_out,
        (nv_bfloat16*)qg_in,
        batch_seq, num_heads, head_dim
    );
}
