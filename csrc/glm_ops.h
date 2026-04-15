#ifndef GLM_OPS_H
#define GLM_OPS_H

#include <stdint.h>
#include <stddef.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

struct GlmCtx {
    int device_id;
    cudaStream_t stream;
    void* cublas_handle;
};

typedef struct GlmCtx GlmCtx;

GlmCtx* glm_init(int device_id);
void glm_free(GlmCtx* ctx);

void* glm_alloc(GlmCtx* ctx, size_t bytes);
void glm_free_buf(GlmCtx* ctx, void* ptr);

void glm_h2d(GlmCtx* ctx, void* dst, const void* src, size_t bytes);
void glm_d2h(GlmCtx* ctx, void* dst, const void* src, size_t bytes);

void* glm_mmap_open(const char* path);
void glm_mmap_close(void* ptr, uint64_t size);
void glm_mmap_load(GlmCtx* ctx, void* gpu_dst, const void* mmap_ptr,
                   uint64_t offset, uint64_t nbytes);

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

void glm_kv_cache_write(GlmCtx* ctx,
                         void* src_k, void* src_v,
                         void* dst_k, void* dst_v,
                         int32_t* slot_mapping,
                         uint32_t batch_size, uint32_t n_kv,
                         uint32_t hd, uint32_t page_size,
                         uint32_t src_token_stride, uint32_t src_head_stride);

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

void* glm_alloc_pinned(size_t bytes);
void glm_free_pinned(void* ptr);
void glm_write_pinned(void* dst, const void* src, size_t size);

void glm_batch_decode_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* indptr_h,
    uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t page_size,
    bool enable_cuda_graph);

void glm_batch_decode_run(
    GlmCtx* ctx,
    void* q, void* o,
    void* k_data, void* v_data,
    int32_t* indices, int32_t* indptr_d, int32_t* last_page_len,
    void* float_ws, void* int_ws,
    int64_t* plan_info,
    uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t page_size, float sm_scale);

void glm_batch_prefill_paged_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* qo_indptr_h, int32_t* paged_kv_indptr_h,
    uint32_t total_qo_rows, uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t page_size, int mask_mode);

void glm_batch_prefill_paged_run(
    GlmCtx* ctx,
    void* q, void* o,
    void* k_data, void* v_data,
    int32_t* indices, int32_t* indptr_d, int32_t* last_page_len,
    void* float_ws, void* int_ws,
    int32_t* q_indptr_d,
    int64_t* plan_info,
    uint32_t total_qo_rows, uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t page_size,
    int32_t q_stride_n, int32_t q_stride_h,
    int mask_mode, float sm_scale);

// CUDA Graph operations
void glm_graph_begin_capture(GlmCtx* ctx);
void* glm_graph_end_capture(GlmCtx* ctx);
void* glm_graph_instantiate(void* graph);
void glm_graph_launch(void* graph_exec, GlmCtx* ctx);
int glm_graph_exec_update(void* graph_exec, void* graph);
void glm_graph_destroy(void* graph);
void glm_graph_exec_destroy(void* graph_exec);

// FP8 block-wise GEMM (SM120 CUTLASS)
// Performs: output[m,n] = (act_scale * fp8_input[m,k]) @ (weight_scale * fp8_weight[n,k]).T
// fp8_input: row-major [m, k] FP8 E4M3
// act_scale: row-major [m, k/128] float32 (per-group-of-128 along K, ScaleGranularityM=1)
// fp8_weight: row-major [n, k] FP8 E4M3 (= column-major [k, n] for CUTLASS)
// weight_scale: row-major [n/128, k/128] float32 (block-wise 128x128)
// bf16_out: row-major [m, n] BF16
void glm_fp8_linear(GlmCtx* ctx, void* bf16_out, const void* fp8_input,
                    const float* act_scale, const void* fp8_weight,
                    const float* weight_scale, void* workspace,
                    size_t workspace_size, int m, int n, int k);

// Query workspace size for FP8 GEMM
size_t glm_fp8_gemm_workspace_size(int m, int n, int k);

// Quantize BF16 input to FP8 E4M3 with per-group-of-128 float32 scales
// fp8_out: row-major [m, k] FP8 E4M3
// scales: row-major [m, ceil(k/128)] float32
// bf16_input: row-major [m, k] BF16
void glm_fp8_quantize(GlmCtx* ctx, void* fp8_out, float* scales,
                      const void* bf16_input, int m, int k);

// Fused FP8 dequantize + GEMV for decode (any M)
// Computes: output[m, j] = sum_k(bf16_input[m, k] * fp8_weight[j, k] * weight_scale[j/128, k/128])
// No activation quantization — BF16 input used directly.
// bf16_out: row-major [M, N] BF16
// bf16_input: row-major [M, K] BF16
// fp8_weight: row-major [N, K] FP8 E4M3
// weight_scale: row-major [N/128, K/128] float32 (block-wise scale_inv)
void glm_fp8_linear_decode(GlmCtx* ctx, void* bf16_out, const void* bf16_input,
                            const void* fp8_weight, const float* weight_scale,
                            int m, int n, int k);

#ifdef __cplusplus
}
#endif

#endif
