#ifndef GLM_OPS_H
#define GLM_OPS_H

#include <stdint.h>
#include <stddef.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void* ptr;
    uint64_t size;
} GlmMmapEntry;

struct GlmCtx {
    int device_id;
    cudaStream_t stream;
    void* cublas_handle;

    void** gpu_allocs;
    int gpu_alloc_count;
    int gpu_alloc_capacity;

    GlmMmapEntry* mmaps;
    int mmap_count;
    int mmap_capacity;
};

typedef struct GlmCtx GlmCtx;

// ---------------------------------------------------------------------------
// GPU allocation handles
// ---------------------------------------------------------------------------

int glm_alloc_h(GlmCtx* ctx, size_t bytes);
void glm_free_h(GlmCtx* ctx, int handle);
void* glm_deref(GlmCtx* ctx, int handle);

// ---------------------------------------------------------------------------
// Mmap handles
// ---------------------------------------------------------------------------

int glm_mmap_open(GlmCtx* ctx, const char* path);
void glm_mmap_load(GlmCtx* ctx, int gpu_handle, int mmap_handle,
                   uint64_t offset, uint64_t nbytes);
void glm_mmap_close(GlmCtx* ctx, int mmap_handle);

GlmCtx* glm_init(int device_id);
void glm_free(GlmCtx* ctx);

void* glm_alloc(GlmCtx* ctx, size_t bytes);
void glm_free_buf(GlmCtx* ctx, void* ptr);
void glm_h2d(GlmCtx* ctx, void* dst, const void* src, size_t bytes);
void glm_d2h(GlmCtx* ctx, void* dst, const void* src, size_t bytes);

void glm_rmsnorm(GlmCtx* ctx, void* out, const void* input,
                 const void* weight, float eps, int dim, int batch);

void glm_silu_and_mul(GlmCtx* ctx, void* out, const void* gate,
                      const void* up, int intermediate, int batch);

void glm_linear(GlmCtx* ctx, void* out, const void* input,
                const void* weight, int batch, int n, int k);

void glm_embedding(GlmCtx* ctx, void* out, const void* table,
                   const int* ids, int hidden, int seq_len);

void glm_layernorm(GlmCtx* ctx, void* out, const void* input,
                   const void* weight, const void* bias, float eps, int dim, int batch);

void glm_relu(GlmCtx* ctx, void* out, const void* input, int n);

void glm_sigmoid(GlmCtx* ctx, void* out, const void* input, int n);

void glm_softmax(GlmCtx* ctx, void* out, const void* input,
                 const void* mask, int dim, int batch);

void glm_causal_mask(GlmCtx* ctx, void* out, int seq_len);

void glm_fill(GlmCtx* ctx, void* out, float value, int n);

void glm_gather(GlmCtx* ctx, void* out, const void* input, const int* indices,
                int k, int in_dim, int batch);

void glm_scatter_scalar(GlmCtx* ctx, void* out, const int* indices, float value,
                        int k, int out_dim, int batch);

void glm_cat_last_dim(GlmCtx* ctx, void* out, const void* a, const void* b,
                      int a_last_dim, int b_last_dim, int outer);

void glm_masked_fill(GlmCtx* ctx, void* out, const void* input, const void* mask,
                     float value, int n);

void glm_index_add(GlmCtx* ctx, void* out, const int* indices, const void* values,
                   int n_indices, int dim);

void glm_rotary_embedding(GlmCtx* ctx, void* cos_out, void* sin_out,
                          const void* inv_freq, const int* position_ids,
                          int dim_half, int batch, int seq_len);

void glm_apply_rotary_pos_emb(GlmCtx* ctx, void* out, const void* x,
                              const void* cos, const void* sin,
                              int rope_dim, int n_heads, int seq_len,
                              int batch, int unsqueeze_dim);

void glm_topk(GlmCtx* ctx, void* out_values, int* out_indices,
              const void* input, int k, int dim, int batch);

void glm_bmm(GlmCtx* ctx, void* C, const void* A, const void* B,
             float alpha, float beta,
             int batch, int M, int N, int K, int transB);

void glm_scale(GlmCtx* ctx, void* out, const void* input, float scale, int n);

void glm_add(GlmCtx* ctx, void* out, const void* a, const void* b, int n);

void glm_expand_dim1(GlmCtx* ctx, void* out, const void* input,
                     int dim1_out, int dim1_in, int seq_len, int head_dim, int batch);

void glm_expand_dim1_strided(GlmCtx* ctx, void* out, const void* input,
                             int dim1_out, int dim1_in, int seq_len, int head_dim,
                             int batch, int head_stride);

void glm_transpose_4d(GlmCtx* ctx, void* out, const void* input,
                      int dim0, int dim1, int dim2, int dim3,
                      int perm0, int perm1, int perm2, int perm3);

void glm_mul(GlmCtx* ctx, void* out, const void* a, const void* b, int n);

void glm_reduce_sum(GlmCtx* ctx, void* out, const void* input, int rows, int cols);

void glm_index_select(GlmCtx* ctx, void* out, const void* src,
                       const void* indices, int dim, int k);

void glm_arange(GlmCtx* ctx, int* out, int start, int step, int count);

void glm_argmax(GlmCtx* ctx, int* out_index, const void* input, int dim, int batch);

void glm_memcpy(GlmCtx* ctx, void* dst, const void* src, size_t bytes);

void glm_synchronize(GlmCtx* ctx);

void glm_flash_prefill(
    GlmCtx* ctx,
    void* q, void* k, void* v, void* o, void* tmp,
    int qo_len, int kv_len,
    int num_qo_heads, int num_kv_heads, int head_dim,
    int q_stride_n, int q_stride_h,
    int kv_stride_n, int kv_stride_h,
    int v_stride_n, int v_stride_h,
    int mask_mode, int kv_layout, float sm_scale);

void glm_flash_decode(
    GlmCtx* ctx,
    void* q, void* k, void* v, void* o, void* tmp,
    int kv_len,
    int num_qo_heads, int num_kv_heads, int head_dim,
    int q_stride_n, int q_stride_h,
    int kv_stride_n, int kv_stride_h,
    float sm_scale);

#ifdef __cplusplus
}
#endif

#endif
