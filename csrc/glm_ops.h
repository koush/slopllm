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

void glm_apply_rotary_pos_emb_partial(GlmCtx* ctx, void* out, const void* x,
                                       const void* cos, const void* sin,
                                       int rope_dim, int head_dim, int n_heads, int seq_len,
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
    uint32_t head_dim, uint32_t page_size,
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

// Gated DeltaNet recurrent step (decode, T=1, batched)
// Fused: L2 norm q,k + gate computation + delta rule update
// output: [batch_size, num_heads, d_v] BF16
// state: [batch_size, state_stride] FP32 (updated in-place)
// q: [batch_size, num_heads, d_k] BF16
// k: [batch_size, num_heads, d_k] BF16
// v: [batch_size, num_heads, d_v] BF16
// a_raw: [batch_size, num_heads] BF16
// b_raw: [batch_size, num_heads] BF16
// A_log: [num_heads] FP32 (shared across batch)
// dt_bias: [num_heads] FP32 (shared across batch)
// state_stride: stride (in float elements) between batch elements in state
void glm_gdn_recurrent_step(GlmCtx* ctx, void* output, void* state,
                              const void* q, const void* k, const void* v,
                              const void* a_raw, const void* b_raw,
                              const float* A_log, const float* dt_bias,
                              int num_heads, int d_k, int d_v,
                              int batch_size, int state_stride);

// Gated DeltaNet prefill (sequential over tokens, batched with cu_seqlens)
// output: [total_seq_len, num_heads, d_v] BF16 (packed)
// state: [batch_size, state_stride] FP32 (updated in-place, should be zero-initialized)
// q: [total_seq_len, num_heads, d_k] BF16 (packed)
// k: [total_seq_len, num_heads, d_k] BF16 (packed)
// v: [total_seq_len, num_heads, d_v] BF16 (packed)
// a_raw: [total_seq_len, num_heads] BF16 (packed)
// b_raw: [total_seq_len, num_heads] BF16 (packed)
// cu_seqlens: [batch_size + 1] int32 (cumulative sequence lengths)
// A_log: [num_heads] FP32 (shared across batch)
// dt_bias: [num_heads] FP32 (shared across batch)
// state_stride: stride (in float elements) between batch elements in state
void glm_gdn_prefill(GlmCtx* ctx, void* output, void* state,
                      const void* q, const void* k, const void* v,
                      const void* a_raw, const void* b_raw,
                      const float* A_log, const float* dt_bias,
                      const int* cu_seqlens,
                      int total_seq_len, int num_heads, int d_k, int d_v,
                      int batch_size, int state_stride);

// Causal conv1d with SiLU activation (batched prefill with cu_seqlens)
// input/output layout: [conv_dim, total_seq_len] (channel-first, packed sequences, BF16)
// weight layout: [conv_dim, kernel_size] (BF16)
// conv_state: [batch_size, conv_state_stride] BF16 (per-batch conv state, updated in-place)
// cu_seqlens: [batch_size + 1] int32 (cumulative sequence lengths)
// conv_state_stride: stride (in bf16 elements) between batch elements in conv_state
void glm_causal_conv1d(GlmCtx* ctx, void* output, void* conv_state,
                        const void* input, const void* weight,
                        const int* cu_seqlens,
                        int conv_dim, int total_seq_len, int kernel_size,
                        int batch_size, int conv_state_stride);

// Causal conv1d update with SiLU activation (batched decode)
// input/output layout: [batch_size, conv_dim] BF16
// conv_state: [batch_size, conv_state_stride] BF16 (per-batch conv state, updated in-place)
// weight layout: [conv_dim, kernel_size] (BF16)
// conv_state_stride: stride (in bf16 elements) between batch elements in conv_state
void glm_causal_conv1d_update(GlmCtx* ctx, void* output, void* conv_state,
                               const void* input, const void* weight,
                               int conv_dim, int kernel_size,
                               int batch_size, int conv_state_stride);

// RMSNorm gated: output = RMSNorm(input) * weight * SiLU(gate)
// output: [batch, dim] BF16
// input: [batch, dim] BF16
// gate: [batch, dim] BF16
// weight: [dim] BF16
void glm_rmsnorm_gated(GlmCtx* ctx, void* output, const void* input,
                        const void* gate, const void* weight,
                        float eps, int dim, int batch);

// QKV split from conv1d output [convDim, S] to GDN prefill layout
// qkv_in: [num_heads * qkv_stride, seq_len] BF16 (channel-first, output of causal_conv1d)
// q_out: [seq_len, num_heads, d_k] BF16
// k_out: [seq_len, num_heads, d_k] BF16
// v_out: [seq_len, num_heads, d_v] BF16
// qkv_stride = 2 * d_k + d_v (per-head QKV channel stride)
void glm_qkv_split(GlmCtx* ctx, void* q_out, void* k_out, void* v_out,
                    const void* qkv_in,
                    int seq_len, int num_heads, int d_k, int d_v);

// Split interleaved [query|gate] per head:
// qg_in: [batch_seq, num_heads, head_dim * 2] BF16 (row-major)
// q_out: [batch_seq, num_heads * head_dim] BF16
// gate_out: [batch_seq, num_heads * head_dim] BF16
void glm_interleaved_split(GlmCtx* ctx, void* q_out, void* gate_out,
                            const void* qg_in,
                            int batch_seq, int num_heads, int head_dim);

// GPU sampling: temperature, repetition/presence penalty, top-K, softmax, top-P, multinomial
// out_token: [1] int32 - sampled token ID
// topk_vals: [SAMPLING_MAX_TOPK] float32 - workspace for top-K values
// topk_idxs: [SAMPLING_MAX_TOPK] int32 - workspace for top-K indices
// workspace: [vocab_size] float32 - workspace for F32 logits
// logits: [vocab_size] bfloat16 - input logits
// penalty_tokens: [num_penalty_tokens] int32 - token IDs for penalty
void glm_sample(GlmCtx* ctx, int* out_token, float* topk_vals, int* topk_idxs,
                float* workspace, const void* logits, const int* penalty_tokens,
                int vocab_size, int num_penalty_tokens,
                float temperature, float repetition_penalty, float presence_penalty,
                int top_k, float top_p, float random_val);

#ifdef __cplusplus
}
#endif

#endif
