#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

namespace flashinfer::sparse_mla_sm120 {
    bool launch_sparse_mla_decode_dsv4(
        ModelType mt, int num_heads, int topk, int page_block_size, int num_tokens, int num_splits,
        const __nv_bfloat16* Q, const uint8_t* KV_cache, const int32_t* indices,
        __nv_bfloat16* mid_out, float* mid_lse, __nv_bfloat16* output, float* out_lse,
        const int* topk_length, const float* attn_sink, const uint8_t* extra_KV_cache,
        const int32_t* extra_indices, const int* extra_topk_length, int extra_topk,
        int pbs_extra, size_t stride_extra_kv_block, int chunks_per_block_override,
        float sm_scale, size_t stride_kv_block, size_t stride_indices_token,
        size_t stride_extra_indices_token, size_t stride_out_lse, cudaStream_t stream);
    bool sparse_mla_prefill_dispatch(
        ModelType mt, PrefillVariant variant, int num_heads, int topk, int page_block_size,
        int topk_extra, int extra_page_block_size,
        const __nv_bfloat16* Q, const uint8_t* KV_cache, const int32_t* indices,
        const uint8_t* extra_KV_cache, const int32_t* extra_indices,
        __nv_bfloat16* output, float* out_lse, float sm_scale, int num_tokens,
        size_t stride_kv_block, size_t stride_kv_block_extra, size_t stride_out_lse,
        const float* attn_sink, const int* topk_length,
        const int* extra_topk_length, cudaStream_t stream,
         const __nv_bfloat16* Q_rope_split, const float* Q_scales);

    bool launch_sparse_mla_decode_dsv3_2(
        ModelType mt, int num_heads, int topk, int num_tokens,
        int num_splits, const __nv_bfloat16* Q, const uint8_t* KV_cache,
        const int32_t* indices, __nv_bfloat16* mid_out, float* mid_lse,
        __nv_bfloat16* output, float* out_lse, const int* topk_length,
        const float* attn_sink, int chunks_per_block_override,
        float sm_scale, size_t stride_kv_block, size_t stride_indices_token,
        int stride_kv_row, size_t stride_out_lse, cudaStream_t stream,
         const __nv_bfloat16* Q_rope_split, const float* Q_scales);
}

#include "glm_ops.h"

extern "C" {

bool glm_dsv4_sparse_attention(
    GlmCtx* ctx, const void* q, const uint8_t* kv, const int32_t* indices,
    const int32_t* lengths, const uint8_t* extra_kv, const int32_t* extra_indices,
    const int32_t* extra_lengths, const float* sinks, void* output, float* lse,
    void* mid_out, float* mid_lse, int tokens, int heads, int topk, int extra_topk,
    int extra_page_size, int splits, float scale) {
    cudaSetDevice(ctx->device_id);
    using namespace flashinfer::sparse_mla_sm120;
    if (tokens <= 0 || heads <= 0 || heads > 128 || topk <= 0 || topk % 64
        || extra_topk < 0 || extra_topk % 64 || (extra_kv != nullptr) != (extra_topk > 0)
        || (extra_kv && (!extra_indices || extra_page_size != 64))) return false;
    if (splits > 0) {
        if (!mid_out || !mid_lse || tokens > 64) return false;
        return launch_sparse_mla_decode_dsv4(ModelType::DSV4, heads, topk, 64, tokens, splits,
            (const __nv_bfloat16*)q, kv, indices, (__nv_bfloat16*)mid_out, mid_lse,
            (__nv_bfloat16*)output, lse, lengths, sinks, extra_kv, extra_indices, extra_lengths,
            extra_topk, extra_page_size, size_t(extra_page_size) * 584, 0, scale,
            64 * 584, topk, extra_topk, heads, GLM_STREAM(ctx));
    }
    return sparse_mla_prefill_dispatch(ModelType::DSV4,
        extra_kv ? PrefillVariant::MG_DUAL : PrefillVariant::MG,
        heads, topk, 64, extra_topk, extra_page_size,
        (const __nv_bfloat16*)q, kv, indices, extra_kv, extra_indices,
        (__nv_bfloat16*)output, lse, scale, tokens, 64 * 584,
        size_t(extra_page_size) * 584, heads, sinks, lengths, extra_lengths,
        GLM_STREAM(ctx), nullptr, nullptr);
}

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
        num_heads >= 64 ? PrefillVariant::SWAPAB : num_heads >= 32 ? PrefillVariant::MG : PrefillVariant::SG,
        num_heads, topk, page_block_size,
        0, 0,  // no dual-cache
        (const __nv_bfloat16*)q,
        (const uint8_t*)kv_cache,
        indices,
        nullptr, nullptr,  // no extra cache
        (__nv_bfloat16*)output,
        out_lse,
        sm_scale, num_tokens,
        stride_kv_block, 0, num_heads,
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
        topk, 656, num_heads,
        GLM_STREAM(ctx),
        (const __nv_bfloat16*)q_rope, q_scales);

    if (!ok) {
        fprintf(stderr, "glm_sparse_mla_decode: dispatch failed for num_heads=%u topk=%u\n",
                num_heads, topk);
    }
}

} // extern "C"
