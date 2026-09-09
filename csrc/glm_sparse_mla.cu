#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

namespace flashinfer::sparse_mla_sm120 {
    bool sparse_mla_prefill_dispatch(
        ModelType mt, int num_heads, int topk, int page_block_size,
        int topk_extra, int extra_page_block_size,
        const __nv_bfloat16* Q, const uint8_t* KV_cache, const int32_t* indices,
        const uint8_t* extra_KV_cache, const int32_t* extra_indices,
        __nv_bfloat16* output, float* out_lse, float sm_scale, int num_tokens,
        size_t stride_kv_block, size_t stride_kv_block_extra,
        const float* attn_sink, const int* topk_length,
        const int* extra_topk_length, cudaStream_t stream,
         const __nv_bfloat16* Q_rope_split, const float* Q_scales);

    bool launch_sparse_mla_decode_dsv3_2(
        ModelType mt, int num_heads, int topk, int num_tokens,
        int num_splits, const __nv_bfloat16* Q, const uint8_t* KV_cache,
        const int32_t* indices, __nv_bfloat16* mid_out, float* mid_lse,
        __nv_bfloat16* output, float* out_lse, const int* topk_length,
        const float* attn_sink, int chunks_per_block_override,
        float sm_scale, size_t stride_kv_block, cudaStream_t stream,
         const __nv_bfloat16* Q_rope_split, const float* Q_scales);
}

#include "glm_ops.h"

extern "C" {

void glm_sparse_mla_prefill(
    GlmCtx* ctx,
    void* q,                  // contiguous Q or split Q_nope
    void* q_rope,             // split [num_tokens, num_heads, 64] or null
    void* kv_cache,           // [num_pages, page_size, bpt] U8
    int32_t* indices,         // [num_tokens, topk] — flat slot IDs, -1 = invalid
    void* output,             // [num_tokens, num_heads, d_v] BF16
    float* out_lse,           // [num_tokens, num_heads] FP32
    uint32_t num_tokens,
    uint32_t num_heads,
    uint32_t topk,
    uint32_t page_block_size,
    float sm_scale,
    size_t stride_kv_block,   // page_block_size * bytes_per_token
    int32_t* topk_length,     // [num_tokens] or null
    const float* q_scales     // [num_tokens, num_heads, 4] or null for BF16 Q
) {
    cudaSetDevice(ctx->device_id);
    using flashinfer::sparse_mla_sm120::sparse_mla_prefill_dispatch;

    bool ok = sparse_mla_prefill_dispatch(
        ModelType::GLM_NSA,
        num_heads, topk, page_block_size,
        0, 0,  // no dual-cache
        (const __nv_bfloat16*)q,
        (const uint8_t*)kv_cache,
        indices,
        nullptr, nullptr,  // no extra cache
        (__nv_bfloat16*)output,
        out_lse,
        sm_scale, num_tokens,
        stride_kv_block, 0,
        nullptr,  // no attn_sink
        topk_length,
        nullptr,
        GLM_STREAM(ctx),
        (const __nv_bfloat16*)q_rope, q_scales);

    if (!ok) {
        fprintf(stderr, "glm_sparse_mla_prefill: dispatch failed for num_heads=%u topk=%u\n",
                num_heads, topk);
    }
}

void glm_sparse_mla_decode(
    GlmCtx* ctx,
    void* q,                  // contiguous Q or split Q_nope
    void* q_rope,             // split [num_tokens, num_heads, 64] or null
    void* kv_cache,           // [num_pages, page_size, bpt] U8
    int32_t* indices,         // [num_tokens, topk] — flat slot IDs, -1 = invalid
    void* mid_out,            // [num_tokens, num_heads, num_splits, d_v] BF16
    float* mid_lse,           // [num_tokens, num_heads, num_splits] FP32
    void* output,             // [num_tokens, num_heads, d_v] BF16
    float* out_lse,           // [num_tokens, num_heads] FP32
    uint32_t num_tokens,
    uint32_t num_heads,
    uint32_t topk,
    uint32_t num_splits,
    float sm_scale,
    size_t stride_kv_block,
    int32_t* topk_length,     // [num_tokens] or null
    int chunks_per_block_override,  // 0 = use default heuristic
    const float* q_scales           // [num_tokens, num_heads, 4] or null for BF16 Q
) {
    cudaSetDevice(ctx->device_id);
    using flashinfer::sparse_mla_sm120::launch_sparse_mla_decode_dsv3_2;

    bool ok = launch_sparse_mla_decode_dsv3_2(
        ModelType::GLM_NSA,
        num_heads, topk, num_tokens,
        num_splits,
        (const __nv_bfloat16*)q,
        (const uint8_t*)kv_cache,
        indices,
        (__nv_bfloat16*)mid_out,
        mid_lse,
        (__nv_bfloat16*)output,
        out_lse,
        topk_length,
        nullptr,  // no attn_sink
        chunks_per_block_override,
        sm_scale,
        stride_kv_block,
        GLM_STREAM(ctx),
        (const __nv_bfloat16*)q_rope, q_scales);

    if (!ok) {
        fprintf(stderr, "glm_sparse_mla_decode: dispatch failed for num_heads=%u topk=%u\n",
                num_heads, topk);
    }
}

} // extern "C"
