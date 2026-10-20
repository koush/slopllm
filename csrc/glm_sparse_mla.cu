#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <flashinfer/attention/sparse_mla_sm120/execution/attention_params.cuh>
#include <flashinfer/attention/sparse_mla_sm120/execution/attention_plan.h>
#include "../vendor/flashinfer/csrc/sparse_mla_sm120/attention_dispatch.h"
#include "glm_ops.h"

namespace {
using namespace flashinfer::sparse_mla_sm120;
using namespace flashinfer::sparse_mla_sm120::execution;

bool run_sparse(GlmCtx* ctx, ModelType model, AttentionParams& p, int variant) {
    cudaSetDevice(ctx->device_id);
    // Device capabilities are stable and querying them here does not synchronize
    // the GPU. The resolver uses host-only geometry, including during capture.
    int sms = 0, shared = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx->device_id) != cudaSuccess
        || cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, ctx->device_id) != cudaSuccess)
        return false;
    AttentionMetadata m{};
    m.model = int(model);
    m.tokens = p.num_tokens;
    m.heads = p.num_heads;
    m.topk = p.topk;
    m.extra_topk = p.extra_topk;
    m.page_size = p.page_size;
    m.extra_page_size = p.extra_page_size;
    m.page_stride_bytes = p.page_stride_bytes;
    m.extra_page_stride_bytes = p.extra_page_stride_bytes;
    m.row_stride_bytes = model == ModelType::GLM_NSA ? 656 : 584;
    m.extra_row_stride_bytes = p.extra_topk ? 584 : 0;
    m.indices_stride = p.indices_stride_elems;
    m.extra_indices_stride = p.extra_indices_stride_elems;
    m.lse_stride = p.out_lse_stride_elems;
    m.has_lengths = p.topk_length != nullptr;
    m.has_extra_lengths = p.extra_topk_length != nullptr;
    m.has_sink = p.attn_sink != nullptr;
    m.variant = variant;
    try {
        auto plan = resolve_attention(m, NumericRoute::QkBF16PvFP8,
                                      p.chunks_per_block, {sms, size_t(shared)});
        if (variant) {
            auto result = dispatch_prefill(p, plan, GLM_STREAM(ctx));
            if (result.error != cudaSuccess)
                fprintf(stderr, "sparse MLA prefill: %s\n", cudaGetErrorString(result.error));
            return result.supported && result.error == cudaSuccess;
        }
        if (p.allocated_splits < plan.active_splits || !p.mid_out || !p.mid_lse) return false;
        auto error = model == ModelType::GLM_NSA
            ? dispatch_dsv32(p, plan, GLM_STREAM(ctx))
            : dispatch_decode(p, plan, GLM_STREAM(ctx));
        if (error != cudaSuccess) fprintf(stderr, "sparse MLA decode: %s\n", cudaGetErrorString(error));
        return error == cudaSuccess;
    } catch (const std::exception& e) {
        fprintf(stderr, "sparse MLA plan: %s\n", e.what());
        return false;
    }
}

AttentionParams glm_params(void* q, void* q_rope, void* kv, int32_t* indices,
                           void* output, float* lse, int tokens, int heads, int topk,
                           int page_size, float scale, size_t page_stride,
                           int32_t* lengths, const float* q_scales) {
    AttentionParams p{};
    p.q = (const __nv_bfloat16*)q;
    p.q_rope_split = (const __nv_bfloat16*)q_rope;
    p.q_scales = q_scales;
    p.kv = (const uint8_t*)kv;
    p.indices = indices;
    p.output = (__nv_bfloat16*)output;
    p.out_lse = lse;
    p.num_tokens = tokens;
    p.num_heads = heads;
    p.topk = topk;
    p.page_size = page_size;
    p.page_stride_bytes = page_stride;
    p.indices_stride_elems = topk;
    p.out_lse_stride_elems = heads;
    p.sm_scale = scale;
    p.topk_length = lengths;
    return p;
}
} // namespace

extern "C" {

bool glm_dsv4_sparse_attention(
    GlmCtx* ctx, const void* q, const uint8_t* kv, const int32_t* indices,
    const int32_t* lengths, const uint8_t* extra_kv, const int32_t* extra_indices,
    const int32_t* extra_lengths, const float* sinks, void* output, float* lse,
    void* mid_out, float* mid_lse, int tokens, int heads, int topk, int extra_topk,
    int extra_page_size, int splits, float scale) {
    if (tokens <= 0 || heads <= 0 || heads > 128 || topk <= 0 || topk % 64
        || extra_topk < 0 || extra_topk % 64 || (extra_kv != nullptr) != (extra_topk > 0)
        || (extra_kv && (!extra_indices || extra_page_size != 64))
        || (splits > 0 && tokens > 64)) return false;
    AttentionParams p{};
    p.q = (const __nv_bfloat16*)q;
    p.kv = kv;
    p.indices = indices;
    p.topk_length = lengths;
    p.extra_kv = extra_kv;
    p.extra_indices = extra_indices;
    p.extra_topk_length = extra_lengths;
    p.attn_sink = sinks;
    p.output = (__nv_bfloat16*)output;
    p.out_lse = lse;
    p.mid_out = (__nv_bfloat16*)mid_out;
    p.mid_lse = mid_lse;
    p.num_tokens = tokens;
    p.num_heads = heads;
    p.topk = topk;
    p.extra_topk = extra_topk;
    p.page_size = 64;
    p.extra_page_size = extra_page_size;
    p.page_stride_bytes = 64 * 584;
    p.extra_page_stride_bytes = size_t(extra_page_size) * 584;
    p.indices_stride_elems = topk;
    p.extra_indices_stride_elems = extra_topk;
    p.out_lse_stride_elems = heads;
    p.allocated_splits = splits;
    p.sm_scale = scale;
    return run_sparse(ctx, ModelType::DSV4, p, splits > 0 ? 0 : extra_kv ? 3 : 2);
}

void glm_sparse_mla_prefill(
    GlmCtx* ctx, void* q, void* q_rope, void* kv_cache, int32_t* indices,
    void* output, float* out_lse, uint32_t num_tokens, uint32_t num_heads,
    uint32_t topk, uint32_t page_block_size, float sm_scale, size_t stride_kv_block,
    int32_t* topk_length, const float* q_scales) {
    auto p = glm_params(q, q_rope, kv_cache, indices, output, out_lse, num_tokens,
                        num_heads, topk, page_block_size, sm_scale, stride_kv_block,
                        topk_length, q_scales);
    if (!run_sparse(ctx, ModelType::GLM_NSA, p, num_heads >= 64 ? 4 : num_heads >= 32 ? 2 : 1))
        fprintf(stderr, "glm_sparse_mla_prefill: dispatch failed for num_heads=%u topk=%u\n", num_heads, topk);
}

void glm_sparse_mla_decode(
    GlmCtx* ctx, void* q, void* q_rope, void* kv_cache, int32_t* indices,
    void* mid_out, float* mid_lse, void* output, float* out_lse,
    uint32_t num_tokens, uint32_t num_heads, uint32_t topk, uint32_t num_splits,
    float sm_scale, size_t stride_kv_block, int32_t* topk_length,
    int chunks_per_block_override, const float* q_scales) {
    auto p = glm_params(q, q_rope, kv_cache, indices, output, out_lse, num_tokens,
                        num_heads, topk, 64, sm_scale, stride_kv_block, topk_length, q_scales);
    p.mid_out = (__nv_bfloat16*)mid_out;
    p.mid_lse = mid_lse;
    p.allocated_splits = num_splits;
    p.chunks_per_block = chunks_per_block_override;
    if (!run_sparse(ctx, ModelType::GLM_NSA, p, 0))
        fprintf(stderr, "glm_sparse_mla_decode: dispatch failed for num_heads=%u topk=%u\n", num_heads, topk);
}
} // extern "C"
