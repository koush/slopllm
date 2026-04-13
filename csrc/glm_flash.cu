#include "glm_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/pos_enc.cuh>

using DType = __nv_bfloat16;
using DTypeO = __nv_bfloat16;

extern "C" {

void glm_flash_prefill(
    GlmCtx* ctx,
    void* q, void* k, void* v, void* o, void* tmp,
    int qo_len, int kv_len,
    int num_qo_heads, int num_kv_heads, int head_dim,
    int q_stride_n, int q_stride_h,
    int kv_stride_n, int kv_stride_h,
    int v_stride_n, int v_stride_h,
    int mask_mode, int kv_layout, float sm_scale) {

  using AttentionVariant = flashinfer::DefaultAttention<false, false, false, false>;
  using Params = flashinfer::SinglePrefillParams<DType, DType, DTypeO>;

  Params params;
  params.q = static_cast<DType*>(q);
  params.k = static_cast<DType*>(k);
  params.v = static_cast<DType*>(v);
  params.o = static_cast<DTypeO*>(o);
  params.lse = nullptr;
  params.maybe_custom_mask = nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.qo_len = qo_len;
  params.kv_len = kv_len;
  params.q_stride_n = q_stride_n;
  params.q_stride_h = q_stride_h;
  params.k_stride_n = kv_stride_n;
  params.k_stride_h = kv_stride_h;
  params.v_stride_n = v_stride_n;
  params.v_stride_h = v_stride_h;
  params.head_dim = head_dim;
  params.window_left = -1;
  params.logits_soft_cap = 0.0f;
  params.sm_scale = sm_scale;
  params.rope_rcp_scale = 1.0f;
  params.rope_rcp_theta = 1.0f;
  params.partition_kv = false;
  params.group_size = flashinfer::uint_fastdiv(num_qo_heads / num_kv_heads);

  flashinfer::MaskMode flash_mask = static_cast<flashinfer::MaskMode>(mask_mode);

  cudaError_t status;
  if (flash_mask == flashinfer::MaskMode::kCausal) {
    status = flashinfer::SinglePrefillWithKVCacheDispatched<
        128, 128,
        flashinfer::PosEncodingMode::kNone,
        false,
        flashinfer::MaskMode::kCausal,
        AttentionVariant, Params>(
        params, static_cast<DTypeO*>(tmp), ctx->stream);
  } else {
    status = flashinfer::SinglePrefillWithKVCacheDispatched<
        128, 128,
        flashinfer::PosEncodingMode::kNone,
        false,
        flashinfer::MaskMode::kNone,
        AttentionVariant, Params>(
        params, static_cast<DTypeO*>(tmp), ctx->stream);
  }

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_flash_prefill failed: %s\n", cudaGetErrorString(status));
  }
}

void glm_flash_decode(
    GlmCtx* ctx,
    void* q, void* k, void* v, void* o, void* tmp,
    int kv_len,
    int num_qo_heads, int num_kv_heads, int head_dim,
    int q_stride_n, int q_stride_h,
    int kv_stride_n, int kv_stride_h,
    float sm_scale) {

  using AttentionVariant = flashinfer::DefaultAttention<false, false, false, false>;
  using Params = flashinfer::SingleDecodeParams<DType, DType, DTypeO>;

  Params params;
  params.q = static_cast<DType*>(q);
  params.k = static_cast<DType*>(k);
  params.v = static_cast<DType*>(v);
  params.o = static_cast<DTypeO*>(o);
  params.lse = nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.kv_len = kv_len;
  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.q_stride_n = q_stride_n;
  params.q_stride_h = q_stride_h;
  params.kv_stride_n = kv_stride_n;
  params.kv_stride_h = kv_stride_h;
  params.window_left = -1;
  params.logits_soft_cap = 0.0f;
  params.sm_scale = sm_scale;
  params.rope_rcp_scale = 1.0f;
  params.rope_rcp_theta = 1.0f;
  params.kv_chunk_size = 0;

  cudaError_t status =
      flashinfer::SingleDecodeWithKVCacheDispatched<
          128,
          flashinfer::PosEncodingMode::kNone,
          AttentionVariant, Params>(
          params, static_cast<DTypeO*>(tmp), ctx->stream);

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_flash_decode failed: %s\n", cudaGetErrorString(status));
  }
}

} // extern "C"
