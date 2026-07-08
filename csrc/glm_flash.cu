#include "glm_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/mla_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

using DType = __nv_bfloat16;
using DTypeO = __nv_bfloat16;
using IdType = int32_t;
using AttentionVariant = flashinfer::DefaultAttention<false, false, false, false>;
constexpr auto POS_ENC = flashinfer::PosEncodingMode::kNone;

#define DISPATCH_HEAD_DIM(HEAD_DIM_VAL, ...) \
  do { \
    if ((HEAD_DIM_VAL) == 256) { \
      constexpr uint32_t HEAD_DIM = 256; \
      __VA_ARGS__; \
    } else if ((HEAD_DIM_VAL) == 128) { \
      constexpr uint32_t HEAD_DIM = 128; \
      __VA_ARGS__; \
    } else { \
      fprintf(stderr, "Unsupported head_dim: %u\n", HEAD_DIM_VAL); \
    } \
  } while (0)

namespace {

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM>
cudaError_t dispatch_decode_work_est(
    bool& split_kv, uint32_t& max_grid_size,
    uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy,
    uint32_t batch_size, IdType* kv_indptr_h,
    uint32_t num_qo_heads, uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream) {
  using DecodeParams = flashinfer::BatchDecodeParams<DType, DType, DTypeO, IdType>;
  return flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
      GROUP_SIZE, HEAD_DIM, POS_ENC, AttentionVariant, DecodeParams>(
      split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy,
      batch_size, kv_indptr_h, num_qo_heads, page_size, enable_cuda_graph, stream);
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE>
cudaError_t dispatch_batch_prefill_paged_run_inner(
    flashinfer::BatchPrefillPagedParams<DType, DType, DTypeO, IdType>& params,
    DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream) {
  return flashinfer::BatchPrefillWithPagedKVCacheDispatched<
      CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENC, false, MASK_MODE,
      AttentionVariant,
      flashinfer::BatchPrefillPagedParams<DType, DType, DTypeO, IdType>>(
      params, tmp_v, tmp_s, enable_pdl, stream);
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE>
cudaError_t dispatch_batch_prefill_ragged_run_inner(
    flashinfer::BatchPrefillRaggedParams<DType, DType, DTypeO, IdType>& params,
    DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream) {
  return flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
      CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENC, false, MASK_MODE,
      AttentionVariant,
      flashinfer::BatchPrefillRaggedParams<DType, DType, DTypeO, IdType>>(
      params, tmp_v, tmp_s, enable_pdl, stream);
}

template <uint32_t HEAD_DIM>
void glm_batch_decode_plan_impl(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* indptr_h,
    uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t page_size,
    bool enable_cuda_graph) {

  using DecodeParams = flashinfer::BatchDecodeParams<DType, DType, DTypeO, IdType>;

  flashinfer::DecodePlanInfo info;
  uint32_t group_size = num_qo_heads / num_kv_heads;

  auto work_est = [&](bool& split_kv, uint32_t& max_grid_size,
                      uint32_t& max_num_pages_per_batch,
                      uint32_t& new_batch_size, uint32_t& gdy,
                      uint32_t bs, IdType* kv_indptr,
                      uint32_t nqh, uint32_t ps,
                      bool ecg, cudaStream_t s) -> cudaError_t {
    switch (group_size) {
      case 1: return dispatch_decode_work_est<1, HEAD_DIM>(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s);
      case 2: return dispatch_decode_work_est<2, HEAD_DIM>(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s);
      case 4: return dispatch_decode_work_est<4, HEAD_DIM>(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s);
      case 8: return dispatch_decode_work_est<8, HEAD_DIM>(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s);
      case 16: return dispatch_decode_work_est<16, HEAD_DIM>(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s);
      default:
        fprintf(stderr, "Unsupported group_size %u\n", group_size);
        return cudaErrorInvalidValue;
    }
  };

  cudaError_t status = flashinfer::DecodePlan<HEAD_DIM, POS_ENC, AttentionVariant, DecodeParams>(
      float_ws, float_ws_size,
      int_ws, pinned_int_ws, int_ws_size,
      info,
      indptr_h,
      batch_size,
      num_qo_heads,
      page_size,
      enable_cuda_graph,
      GLM_STREAM(ctx),
      work_est);

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_batch_decode_plan failed: %s\n", cudaGetErrorString(status));
    return;
  }

  auto vec = info.ToVector();
  memcpy(plan_info, vec.data(), sizeof(int64_t) * vec.size());
}

} // anonymous namespace

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

  cudaSetDevice(ctx->device_id);

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
  DISPATCH_HEAD_DIM(head_dim, {
    if (flash_mask == flashinfer::MaskMode::kCausal) {
      status = flashinfer::SinglePrefillWithKVCacheDispatched<
          HEAD_DIM, HEAD_DIM,
          flashinfer::PosEncodingMode::kNone,
          false,
          flashinfer::MaskMode::kCausal,
          AttentionVariant, Params>(
          params, static_cast<DTypeO*>(tmp), GLM_STREAM(ctx));
    } else {
      status = flashinfer::SinglePrefillWithKVCacheDispatched<
          HEAD_DIM, HEAD_DIM,
          flashinfer::PosEncodingMode::kNone,
          false,
          flashinfer::MaskMode::kNone,
          AttentionVariant, Params>(
          params, static_cast<DTypeO*>(tmp), GLM_STREAM(ctx));
    }
  });

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

  cudaSetDevice(ctx->device_id);

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

  cudaError_t status;
  DISPATCH_HEAD_DIM(head_dim, {
    status = flashinfer::SingleDecodeWithKVCacheDispatched<
        HEAD_DIM,
        flashinfer::PosEncodingMode::kNone,
        AttentionVariant, Params>(
        params, static_cast<DTypeO*>(tmp), GLM_STREAM(ctx));
  });

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_flash_decode failed: %s\n", cudaGetErrorString(status));
  }
}

void glm_batch_decode_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* indptr_h,
    uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t page_size,
    bool enable_cuda_graph) {

  cudaSetDevice(ctx->device_id);

  DISPATCH_HEAD_DIM(head_dim, {
    glm_batch_decode_plan_impl<HEAD_DIM>(ctx, float_ws, float_ws_size,
        int_ws, pinned_int_ws, int_ws_size,
        plan_info, indptr_h, batch_size,
        num_qo_heads, num_kv_heads, page_size, enable_cuda_graph);
  });
}

void glm_batch_decode_run(
    GlmCtx* ctx,
    void* q, void* o,
    void* k_data, void* v_data,
    int32_t* indices, int32_t* indptr_d, int32_t* last_page_len,
    void* float_ws, void* int_ws,
    int64_t* plan_info,
    uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t page_size, float sm_scale) {

  cudaSetDevice(ctx->device_id);

  using DecodeParams = flashinfer::BatchDecodeParams<DType, DType, DTypeO, IdType>;

  flashinfer::DecodePlanInfo info;
  info.FromVector(std::vector<int64_t>(plan_info, plan_info + 10));

  flashinfer::paged_kv_t<DType, IdType> paged_kv(
      num_kv_heads, page_size, head_dim,
      batch_size,
      flashinfer::QKVLayout::kHND,
      static_cast<DType*>(k_data),
      static_cast<DType*>(v_data),
      indices, indptr_d, last_page_len, nullptr);

  DecodeParams params(
      static_cast<DType*>(q),
      nullptr, // q_rope_offset
      paged_kv,
      static_cast<DTypeO*>(o),
      nullptr, // lse
      nullptr, // maybe_alibi_slopes
      num_qo_heads,
      IdType(num_qo_heads * head_dim), // q_stride_n
      IdType(head_dim), // q_stride_h
      -1, // window_left
      0.0f, // logits_soft_cap
      sm_scale,
      1.0f, // rope_scale
      1.0f // rope_theta
  );

  params.padded_batch_size = info.padded_batch_size;
  params.request_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.request_indices_offset);
  params.kv_tile_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_tile_indices_offset);
  params.o_indptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.o_indptr_offset);
  params.kv_chunk_size_ptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_chunk_size_ptr_offset);
  params.block_valid_mask = info.split_kv
      ? reinterpret_cast<bool*>(static_cast<char*>(int_ws) + info.block_valid_mask_offset)
      : nullptr;
  params.partition_kv = info.split_kv;

  DTypeO* tmp_v = info.split_kv
      ? reinterpret_cast<DTypeO*>(static_cast<char*>(float_ws) + info.v_offset)
      : nullptr;
  float* tmp_s = info.split_kv
      ? reinterpret_cast<float*>(static_cast<char*>(float_ws) + info.s_offset)
      : nullptr;

  cudaError_t status;
  DISPATCH_HEAD_DIM(head_dim, {
      status =
      flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, POS_ENC, AttentionVariant, DecodeParams>(
          params, tmp_v, tmp_s, false, GLM_STREAM(ctx));
  });

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_batch_decode_run failed: %s\n", cudaGetErrorString(status));
  }
}

void glm_batch_prefill_paged_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* qo_indptr_h, int32_t* paged_kv_indptr_h,
    uint32_t total_qo_rows, uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t page_size, int mask_mode) {

  cudaSetDevice(ctx->device_id);

  flashinfer::PrefillPlanInfo info;

  cudaError_t status = flashinfer::PrefillPlan<IdType>(
      float_ws, float_ws_size,
      int_ws, pinned_int_ws, int_ws_size,
      info,
      qo_indptr_h,
      paged_kv_indptr_h,
      total_qo_rows,
      batch_size,
      num_qo_heads,
      num_kv_heads,
      head_dim, // head_dim_qk
      head_dim, // head_dim_vo
      page_size,
      false, // enable_cuda_graph
      2, // sizeof_dtype_o (bf16 = 2 bytes)
      -1, // window_left
      -1, // fixed_split_size
      false, // disable_split_kv
      0, // num_colocated_ctas
      GLM_STREAM(ctx));

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_batch_prefill_paged_plan failed: %s\n", cudaGetErrorString(status));
    return;
  }

  auto vec = info.ToVector();
  memcpy(plan_info, vec.data(), sizeof(int64_t) * vec.size());
}

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
    int mask_mode, float sm_scale) {

  cudaSetDevice(ctx->device_id);

  using PrefillParams = flashinfer::BatchPrefillPagedParams<DType, DType, DTypeO, IdType>;

  flashinfer::PrefillPlanInfo info;
  info.FromVector(std::vector<int64_t>(plan_info, plan_info + 15));
  flashinfer::MaskMode flash_mask = static_cast<flashinfer::MaskMode>(mask_mode);

  flashinfer::paged_kv_t<DType, IdType> paged_kv(
      num_kv_heads, page_size, head_dim,
      batch_size,
      flashinfer::QKVLayout::kHND,
      static_cast<DType*>(k_data),
      static_cast<DType*>(v_data),
      indices, indptr_d, last_page_len, nullptr);

  PrefillParams params(
      static_cast<DType*>(q),
      paged_kv,
      nullptr, // maybe_custom_mask
      q_indptr_d,
      nullptr, // maybe_mask_indptr
      nullptr, // maybe_q_rope_offset
      static_cast<DTypeO*>(o),
      nullptr, // lse
      nullptr, // maybe_alibi_slopes
      num_qo_heads,
      q_stride_n,
      q_stride_h,
      -1, // window_left
      0.0f, // logits_soft_cap
      sm_scale,
      1.0f, // rope_scale
      1.0f // rope_theta
  );

  params.request_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.request_indices_offset);
  params.qo_tile_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.qo_tile_indices_offset);
  params.kv_tile_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_tile_indices_offset);
  params.merge_indptr = info.split_kv
      ? reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.merge_indptr_offset)
      : nullptr;
  params.o_indptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.o_indptr_offset);
  params.kv_chunk_size_ptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_chunk_size_ptr_offset);
  params.block_valid_mask = info.split_kv
      ? reinterpret_cast<bool*>(static_cast<char*>(int_ws) + info.block_valid_mask_offset)
      : nullptr;
  params.max_total_num_rows = total_qo_rows;
  if (info.enable_cuda_graph) {
    params.total_num_rows = reinterpret_cast<uint32_t*>(static_cast<char*>(int_ws) + info.total_num_rows_offset);
  } else {
    params.total_num_rows = nullptr;
  }
  params.padded_batch_size = info.padded_batch_size;
  params.partition_kv = info.split_kv;
  params.maybe_prefix_len_ptr = nullptr;
  params.maybe_token_pos_in_items_ptr = nullptr;
  params.token_pos_in_items_len = 0;
  params.maybe_max_item_len_ptr = nullptr;

  DTypeO* tmp_v = info.split_kv
      ? reinterpret_cast<DTypeO*>(static_cast<char*>(float_ws) + info.v_offset)
      : nullptr;
  float* tmp_s = info.split_kv
      ? reinterpret_cast<float*>(static_cast<char*>(float_ws) + info.s_offset)
      : nullptr;

  uint32_t cta_tile_q = static_cast<uint32_t>(info.cta_tile_q);

  cudaError_t status = cudaSuccess;

  DISPATCH_HEAD_DIM(head_dim, {
    if (flash_mask == flashinfer::MaskMode::kCausal) {
      switch (cta_tile_q) {
        case 128: status = dispatch_batch_prefill_paged_run_inner<128, HEAD_DIM, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 64: status = dispatch_batch_prefill_paged_run_inner<64, HEAD_DIM, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 16: status = dispatch_batch_prefill_paged_run_inner<16, HEAD_DIM, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 1: status = dispatch_batch_prefill_paged_run_inner<1, HEAD_DIM, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        default: fprintf(stderr, "Unsupported cta_tile_q: %u\n", cta_tile_q); status = cudaErrorInvalidValue;
      }
    } else {
      switch (cta_tile_q) {
        case 128: status = dispatch_batch_prefill_paged_run_inner<128, HEAD_DIM, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 64: status = dispatch_batch_prefill_paged_run_inner<64, HEAD_DIM, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 16: status = dispatch_batch_prefill_paged_run_inner<16, HEAD_DIM, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 1: status = dispatch_batch_prefill_paged_run_inner<1, HEAD_DIM, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        default: fprintf(stderr, "Unsupported cta_tile_q: %u\n", cta_tile_q); status = cudaErrorInvalidValue;
      }
    }
  });

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_batch_prefill_paged_run failed: %s\n", cudaGetErrorString(status));
  }
}

void glm_batch_prefill_ragged_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* qo_indptr_h, int32_t* kv_indptr_h,
    uint32_t total_qo_rows, uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim, int mask_mode) {

  cudaSetDevice(ctx->device_id);

  flashinfer::PrefillPlanInfo info;

  cudaError_t status = flashinfer::PrefillPlan<IdType>(
      float_ws, float_ws_size,
      int_ws, pinned_int_ws, int_ws_size,
      info,
      qo_indptr_h,
      kv_indptr_h,
      total_qo_rows,
      batch_size,
      num_qo_heads,
      num_kv_heads,
      head_dim, // head_dim_qk
      head_dim, // head_dim_vo
      1, // page_size=1 for ragged (kv_indptr is token-level)
      false, // enable_cuda_graph
      2, // sizeof_dtype_o (bf16 = 2 bytes)
      -1, // window_left
      -1, // fixed_split_size
      false, // disable_split_kv
      0, // num_colocated_ctas
      GLM_STREAM(ctx));

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_batch_prefill_ragged_plan failed: %s\n", cudaGetErrorString(status));
    return;
  }

  auto vec = info.ToVector();
  memcpy(plan_info, vec.data(), sizeof(int64_t) * vec.size());
}

void glm_batch_prefill_ragged_run(
    GlmCtx* ctx,
    void* q, void* k, void* v, void* o,
    void* float_ws, void* int_ws,
    int32_t* q_indptr_d, int32_t* kv_indptr_d,
    int64_t* plan_info,
    uint32_t total_qo_rows, uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
    int32_t q_stride_n, int32_t q_stride_h,
    int32_t kv_stride_n, int32_t kv_stride_h,
    int32_t v_stride_n, int32_t v_stride_h,
    int mask_mode, float sm_scale) {

  cudaSetDevice(ctx->device_id);

  using RaggedParams = flashinfer::BatchPrefillRaggedParams<DType, DType, DTypeO, IdType>;

  flashinfer::PrefillPlanInfo info;
  info.FromVector(std::vector<int64_t>(plan_info, plan_info + 15));
  flashinfer::MaskMode flash_mask = static_cast<flashinfer::MaskMode>(mask_mode);

  RaggedParams params(
      static_cast<DType*>(q),
      static_cast<DType*>(k),
      static_cast<DType*>(v),
      nullptr, // maybe_custom_mask
      q_indptr_d,
      kv_indptr_d,
      nullptr, // maybe_mask_indptr
      nullptr, // maybe_q_rope_offset
      nullptr, // maybe_k_rope_offset
      static_cast<DTypeO*>(o),
      nullptr, // lse
      nullptr, // maybe_alibi_slopes
      num_qo_heads, num_kv_heads,
      q_stride_n, q_stride_h,
      kv_stride_n, kv_stride_h,
      -1, // window_left
      0.0f, // logits_soft_cap
      sm_scale,
      1.0f, // rope_scale
      1.0f // rope_theta
  );

  params.request_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.request_indices_offset);
  params.qo_tile_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.qo_tile_indices_offset);
  params.kv_tile_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_tile_indices_offset);
  params.merge_indptr = info.split_kv
      ? reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.merge_indptr_offset)
      : nullptr;
  params.o_indptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.o_indptr_offset);
  params.kv_chunk_size_ptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_chunk_size_ptr_offset);
  params.block_valid_mask = info.split_kv
      ? reinterpret_cast<bool*>(static_cast<char*>(int_ws) + info.block_valid_mask_offset)
      : nullptr;
  params.max_total_num_rows = total_qo_rows;
  if (info.enable_cuda_graph) {
    params.total_num_rows = reinterpret_cast<uint32_t*>(static_cast<char*>(int_ws) + info.total_num_rows_offset);
  } else {
    params.total_num_rows = nullptr;
  }
  params.padded_batch_size = info.padded_batch_size;
  params.partition_kv = info.split_kv;
  params.maybe_prefix_len_ptr = nullptr;
  params.maybe_token_pos_in_items_ptr = nullptr;
  params.token_pos_in_items_len = 0;
  params.maybe_max_item_len_ptr = nullptr;

  DTypeO* tmp_v = info.split_kv
      ? reinterpret_cast<DTypeO*>(static_cast<char*>(float_ws) + info.v_offset)
      : nullptr;
  float* tmp_s = info.split_kv
      ? reinterpret_cast<float*>(static_cast<char*>(float_ws) + info.s_offset)
      : nullptr;

  uint32_t cta_tile_q = static_cast<uint32_t>(info.cta_tile_q);

  cudaError_t status = cudaSuccess;

  DISPATCH_HEAD_DIM(head_dim, {
    if (flash_mask == flashinfer::MaskMode::kCausal) {
      switch (cta_tile_q) {
        case 128: status = dispatch_batch_prefill_ragged_run_inner<128, HEAD_DIM, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 64: status = dispatch_batch_prefill_ragged_run_inner<64, HEAD_DIM, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 16: status = dispatch_batch_prefill_ragged_run_inner<16, HEAD_DIM, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 1: status = dispatch_batch_prefill_ragged_run_inner<1, HEAD_DIM, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        default: fprintf(stderr, "Unsupported cta_tile_q: %u\n", cta_tile_q); status = cudaErrorInvalidValue;
      }
    } else {
      switch (cta_tile_q) {
        case 128: status = dispatch_batch_prefill_ragged_run_inner<128, HEAD_DIM, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 64: status = dispatch_batch_prefill_ragged_run_inner<64, HEAD_DIM, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 16: status = dispatch_batch_prefill_ragged_run_inner<16, HEAD_DIM, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        case 1: status = dispatch_batch_prefill_ragged_run_inner<1, HEAD_DIM, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, GLM_STREAM(ctx)); break;
        default: fprintf(stderr, "Unsupported cta_tile_q: %u\n", cta_tile_q); status = cudaErrorInvalidValue;
      }
    }
  });

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_batch_prefill_ragged_run failed: %s\n", cudaGetErrorString(status));
  }
}

// ---------------------------------------------------------------------------
// MLA Prefill
// ---------------------------------------------------------------------------

constexpr uint32_t MLA_HEAD_DIM_CKV = 512;
constexpr uint32_t MLA_HEAD_DIM_KPE = 64;
constexpr uint32_t MLA_HEAD_DIM_CKV_SMALL = 128;
constexpr uint32_t MLA_HEAD_DIM_KPE_SMALL = 64;

#define DISPATCH_MLA_HEAD_DIMS(HEAD_DIM_CKV_VAL, HEAD_DIM_KPE_VAL, ...) \
  do { \
    if ((HEAD_DIM_CKV_VAL) == MLA_HEAD_DIM_CKV && (HEAD_DIM_KPE_VAL) == MLA_HEAD_DIM_KPE) { \
      constexpr uint32_t HEAD_DIM_CKV = MLA_HEAD_DIM_CKV; \
      constexpr uint32_t HEAD_DIM_KPE = MLA_HEAD_DIM_KPE; \
      __VA_ARGS__; \
    } else if ((HEAD_DIM_CKV_VAL) == MLA_HEAD_DIM_CKV_SMALL && (HEAD_DIM_KPE_VAL) == MLA_HEAD_DIM_KPE_SMALL) { \
      constexpr uint32_t HEAD_DIM_CKV = MLA_HEAD_DIM_CKV_SMALL; \
      constexpr uint32_t HEAD_DIM_KPE = MLA_HEAD_DIM_KPE_SMALL; \
      __VA_ARGS__; \
    } else { \
      fprintf(stderr, "Unsupported MLA head dims: ckv=%u kpe=%u\n", HEAD_DIM_CKV_VAL, HEAD_DIM_KPE_VAL); \
    } \
  } while (0)

void glm_mla_prefill_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* qo_indptr_h, int32_t* kv_indptr_h, int32_t* kv_len_h,
    uint32_t batch_size, uint32_t num_heads, uint32_t head_dim_o,
    bool causal, uint32_t cp_world_size, uint32_t cp_rank) {

  cudaSetDevice(ctx->device_id);

  flashinfer::MLAPlanInfo info;
  cudaError_t status = flashinfer::MLAPlan<IdType>(
      float_ws, float_ws_size,
      int_ws, pinned_int_ws, int_ws_size,
      info,
      qo_indptr_h, kv_indptr_h, kv_len_h,
      batch_size, num_heads, head_dim_o,
      causal, GLM_STREAM(ctx), cp_world_size, cp_rank);

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_mla_prefill_plan failed: %s\n", cudaGetErrorString(status));
    return;
  }

  auto vec = info.ToVector();
  memcpy(plan_info, vec.data(), sizeof(int64_t) * vec.size());
}

void glm_mla_prefill_run(
    GlmCtx* ctx,
    void* q_nope, void* q_pe,
    void* ckv_data, void* kpe_data,
    int32_t* kv_indices,
    void* o,
    void* float_ws, void* int_ws,
    int64_t* plan_info,
    uint32_t num_heads, uint32_t page_size,
    int mask_mode, float sm_scale,
    uint32_t q_nope_stride_n, uint32_t q_nope_stride_h,
    uint32_t q_pe_stride_n, uint32_t q_pe_stride_h,
    uint32_t ckv_stride_page, uint32_t ckv_stride_n,
    uint32_t kpe_stride_page, uint32_t kpe_stride_n,
    uint32_t o_stride_n, uint32_t o_stride_h,
    uint32_t head_dim_ckv, uint32_t head_dim_kpe,
    float* lse,
    uint32_t cp_world_size, uint32_t cp_rank,
    void* custom_mask, int32_t* mask_indptr, int32_t* mask_kv_len) {

  cudaSetDevice(ctx->device_id);

  using MLAParams = flashinfer::MLAParams<DType, DType, DTypeO, IdType>;

  flashinfer::MLAPlanInfo info;
  info.FromVector(std::vector<int64_t>(plan_info, plan_info + 19));

  MLAParams params = {};
  params.q_nope = static_cast<DType*>(q_nope);
  params.q_pe = static_cast<DType*>(q_pe);
  params.ckv = static_cast<DType*>(ckv_data);
  params.kpe = static_cast<DType*>(kpe_data);

  params.q_indptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.q_indptr_offset);
  params.kv_indptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_indptr_offset);
  params.partial_indptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.partial_indptr_offset);
  params.kv_indices = static_cast<IdType*>(kv_indices);
  params.q_len = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.q_len_offset);
  params.kv_len = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_len_offset);
  params.q_start = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.q_start_offset);
  params.kv_start = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_start_offset);
  params.kv_end = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_end_offset);
  params.work_indptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.work_indptr_offset);
  params.merge_packed_offset_start = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.merge_packed_offset_start_offset);
  params.merge_packed_offset_end = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.merge_packed_offset_end_offset);
  params.merge_partial_packed_offset_start = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.merge_partial_packed_offset_start_offset);
  params.merge_partial_packed_offset_end = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.merge_partial_packed_offset_end_offset);
  params.merge_partial_stride = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.merge_partial_stride_offset);

  params.final_o = static_cast<DTypeO*>(o);
  params.final_lse = lse;
  params.partial_o = reinterpret_cast<DTypeO*>(static_cast<char*>(float_ws) + info.partial_o_offset);
  params.partial_lse = reinterpret_cast<float*>(static_cast<char*>(float_ws) + info.partial_lse_offset);

  params.num_heads = flashinfer::uint_fastdiv(num_heads);
  params.block_size = flashinfer::uint_fastdiv(page_size);

  params.q_nope_stride_n = q_nope_stride_n;
  params.q_nope_stride_h = q_nope_stride_h;
  params.q_pe_stride_n = q_pe_stride_n;
  params.q_pe_stride_h = q_pe_stride_h;
  params.ckv_stride_page = ckv_stride_page;
  params.ckv_stride_n = ckv_stride_n;
  params.kpe_stride_page = kpe_stride_page;
  params.kpe_stride_n = kpe_stride_n;
  params.o_stride_n = o_stride_n;
  params.o_stride_h = o_stride_h;

  params.sm_scale = sm_scale;
  params.return_lse_base_on_e = false;
  params.cp_world_size = cp_world_size;
  params.cp_rank = cp_rank;

  params.maybe_custom_mask = static_cast<uint8_t*>(custom_mask);
  params.maybe_mask_indptr = mask_indptr;
  params.maybe_mask_kv_len = mask_kv_len;
  params.batch_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.batch_indices_offset);

  flashinfer::MaskMode flash_mask = static_cast<flashinfer::MaskMode>(mask_mode);

  cudaError_t status;
  DISPATCH_MLA_HEAD_DIMS(head_dim_ckv, head_dim_kpe, {
    if (flash_mask == flashinfer::MaskMode::kCausal) {
      status = flashinfer::mla::BatchMLAPagedAttention<
          flashinfer::MaskMode::kCausal, HEAD_DIM_CKV, HEAD_DIM_KPE, MLAParams>(
          params, info.num_blks_x, info.num_blks_y, GLM_STREAM(ctx));
    } else if (flash_mask == flashinfer::MaskMode::kCustom) {
      status = flashinfer::mla::BatchMLAPagedAttention<
          flashinfer::MaskMode::kCustom, HEAD_DIM_CKV, HEAD_DIM_KPE, MLAParams>(
          params, info.num_blks_x, info.num_blks_y, GLM_STREAM(ctx));
    } else if (flash_mask == flashinfer::MaskMode::kCausalCustom) {
      status = flashinfer::mla::BatchMLAPagedAttention<
          flashinfer::MaskMode::kCausalCustom, HEAD_DIM_CKV, HEAD_DIM_KPE, MLAParams>(
          params, info.num_blks_x, info.num_blks_y, GLM_STREAM(ctx));
    } else {
      status = flashinfer::mla::BatchMLAPagedAttention<
          flashinfer::MaskMode::kNone, HEAD_DIM_CKV, HEAD_DIM_KPE, MLAParams>(
          params, info.num_blks_x, info.num_blks_y, GLM_STREAM(ctx));
    }
  });

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_mla_prefill_run failed: %s\n", cudaGetErrorString(status));
  }
}

// ---------------------------------------------------------------------------
// MLA Decode
// ---------------------------------------------------------------------------

namespace {

using MLADecodeParams = flashinfer::BatchDecodeParamsMLA<DType, DType, DTypeO, IdType>;
using MLAAttentionVariant = flashinfer::DefaultAttention<false, false, false, false>;

cudaError_t mla_decode_work_est(
    bool& split_kv, uint32_t& max_grid_size,
    uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy,
    uint32_t batch_size, IdType* kv_indptr_h,
    uint32_t num_qo_heads, uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream,
    uint32_t head_dim_ckv, uint32_t head_dim_kpe) {
  cudaError_t status = cudaErrorNotSupported;
  DISPATCH_MLA_HEAD_DIMS(head_dim_ckv, head_dim_kpe, {
    status = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA<
        HEAD_DIM_CKV, HEAD_DIM_KPE, MLAAttentionVariant, MLADecodeParams>(
        split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy,
        batch_size, kv_indptr_h, num_qo_heads, page_size, enable_cuda_graph, stream);
  });
  return status;
}

} // anonymous namespace

void glm_mla_decode_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* indptr_h,
    uint32_t batch_size, uint32_t num_qo_heads,
    uint32_t page_size, bool enable_cuda_graph,
    uint32_t head_dim_ckv, uint32_t head_dim_kpe) {

  cudaSetDevice(ctx->device_id);

  flashinfer::DecodePlanInfo info;

  auto work_est = [&](bool& split_kv, uint32_t& max_grid_size,
                      uint32_t& max_num_pages_per_batch,
                      uint32_t& new_batch_size, uint32_t& gdy,
                      uint32_t bs, IdType* kv_indptr,
                      uint32_t nqh, uint32_t ps,
                      bool ecg, cudaStream_t s) -> cudaError_t {
    return mla_decode_work_est(split_kv, max_grid_size, max_num_pages_per_batch,
                               new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s,
                               head_dim_ckv, head_dim_kpe);
  };

  cudaError_t status;
  DISPATCH_MLA_HEAD_DIMS(head_dim_ckv, head_dim_kpe, {
    status = flashinfer::DecodePlan<
        HEAD_DIM_CKV, flashinfer::PosEncodingMode::kNone,
        MLAAttentionVariant, MLADecodeParams>(
        float_ws, float_ws_size,
        int_ws, pinned_int_ws, int_ws_size,
        info,
        indptr_h,
        batch_size,
        num_qo_heads,
        page_size,
        enable_cuda_graph,
        GLM_STREAM(ctx),
        work_est);
  });

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_mla_decode_plan failed: %s\n", cudaGetErrorString(status));
    return;
  }

  auto vec = info.ToVector();
  memcpy(plan_info, vec.data(), sizeof(int64_t) * vec.size());
}

void glm_mla_decode_run(
    GlmCtx* ctx,
    void* q_nope, void* q_pe,
    void* ckv_data, void* kpe_data,
    int32_t* indices, int32_t* indptr_d, int32_t* last_page_len,
    void* o,
    void* float_ws, void* int_ws,
    int64_t* plan_info,
    uint32_t batch_size, uint32_t num_qo_heads,
    uint32_t page_size, float sm_scale,
    uint32_t head_dim_ckv, uint32_t head_dim_kpe,
    float* lse) {

  cudaSetDevice(ctx->device_id);

  flashinfer::DecodePlanInfo info;
  info.FromVector(std::vector<int64_t>(plan_info, plan_info + 10));

  flashinfer::paged_kv_mla_t<DType, IdType> paged_kv(
      page_size, head_dim_ckv, head_dim_kpe, batch_size,
      static_cast<DType*>(ckv_data), static_cast<DType*>(kpe_data),
      indices, indptr_d, last_page_len, nullptr);

  MLADecodeParams params(
      static_cast<DType*>(q_nope),
      static_cast<DType*>(q_pe),
      nullptr, // q_rope_offset (use seq_len - 1 as default)
      paged_kv,
      static_cast<DTypeO*>(o),
      lse,     // lse (optional, nullptr to skip)
      num_qo_heads,
      -1, // window_left
      0.0f, // logits_soft_cap
      sm_scale,
      1.0f, // rope_scale
      1000000.0f); // rope_theta (GLM-5.1 uses 1M)

  params.padded_batch_size = info.padded_batch_size;
  params.request_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.request_indices_offset);
  params.kv_tile_indices = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_tile_indices_offset);
  params.o_indptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.o_indptr_offset);
  params.kv_chunk_size_ptr = reinterpret_cast<IdType*>(static_cast<char*>(int_ws) + info.kv_chunk_size_ptr_offset);
  params.block_valid_mask = info.split_kv
      ? reinterpret_cast<bool*>(static_cast<char*>(int_ws) + info.block_valid_mask_offset)
      : nullptr;
  params.partition_kv = info.split_kv;

  DTypeO* tmp_v = info.split_kv
      ? reinterpret_cast<DTypeO*>(static_cast<char*>(float_ws) + info.v_offset)
      : nullptr;
  float* tmp_s = info.split_kv
      ? reinterpret_cast<float*>(static_cast<char*>(float_ws) + info.s_offset)
      : nullptr;

  cudaError_t status;
  DISPATCH_MLA_HEAD_DIMS(head_dim_ckv, head_dim_kpe, {
    status = flashinfer::BatchDecodeWithPagedKVCacheDispatchedMLA<
        HEAD_DIM_CKV, HEAD_DIM_KPE, MLAAttentionVariant, MLADecodeParams>(
        params, tmp_v, tmp_s, false, GLM_STREAM(ctx));
  });

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_mla_decode_run failed: %s\n", cudaGetErrorString(status));
  }
}

// ---------------------------------------------------------------------------
// MLA KV Cache Append
// ---------------------------------------------------------------------------

void glm_mla_kv_cache_append(
    GlmCtx* ctx,
    void* ckv_data, void* kpe_data,
    int32_t* indices, int32_t* indptr, int32_t* last_page_len,
    void* append_ckv, void* append_kpe,
    int32_t* batch_indices, int32_t* positions,
    uint32_t nnz, uint32_t page_size,
    uint32_t head_dim_ckv, uint32_t head_dim_kpe,
    size_t append_ckv_stride_n, size_t append_kpe_stride_n,
    uint32_t cp_world_size, uint32_t cp_rank) {

  cudaSetDevice(ctx->device_id);

  uint32_t vPS = (cp_world_size > 1) ? (page_size / cp_world_size) : page_size;

  flashinfer::paged_kv_mla_t<DType, IdType> paged_kv(
      vPS, head_dim_ckv, head_dim_kpe, 0,
      static_cast<DType*>(ckv_data), static_cast<DType*>(kpe_data),
      indices, indptr, last_page_len, nullptr);

  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, ctx->device_id);

  constexpr uint32_t vec_size = 2;
  cudaError_t status = cudaSuccess;

  // Indexer K-only append: head_dim_kpe == 0 skips kpe writes (kernel guard: tx*vec_size < 0)
  if (head_dim_kpe == 0) {
    if (head_dim_ckv == 128) {
      constexpr uint32_t HC = 128;
      uint32_t bdx = HC / vec_size;
      auto kernel = flashinfer::AppendPagedKVMlaCacheKernel<HC, 0, vec_size, DType, IdType>;
      int num_blocks_per_sm = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, bdx, 0);
      num_blocks_per_sm = std::min(num_blocks_per_sm, (int)((nnz + num_sms - 1) / num_sms));
      dim3 nblks(num_blocks_per_sm * num_sms);
      dim3 nthrs(bdx);
      void* args[] = {(void*)&paged_kv, (void*)&append_ckv, (void*)&append_kpe,
                      (void*)&batch_indices, (void*)&positions, (void*)&nnz,
                      (void*)&append_ckv_stride_n, (void*)&append_kpe_stride_n,
                      (void*)&cp_rank, (void*)&cp_world_size};
      cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, GLM_STREAM(ctx));
      status = cudaGetLastError();
    } else if (head_dim_ckv == 64) {
      constexpr uint32_t HC = 64;
      uint32_t bdx = HC / vec_size;
      auto kernel = flashinfer::AppendPagedKVMlaCacheKernel<HC, 0, vec_size, DType, IdType>;
      int num_blocks_per_sm = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, bdx, 0);
      num_blocks_per_sm = std::min(num_blocks_per_sm, (int)((nnz + num_sms - 1) / num_sms));
      dim3 nblks(num_blocks_per_sm * num_sms);
      dim3 nthrs(bdx);
      void* args[] = {(void*)&paged_kv, (void*)&append_ckv, (void*)&append_kpe,
                      (void*)&batch_indices, (void*)&positions, (void*)&nnz,
                      (void*)&append_ckv_stride_n, (void*)&append_kpe_stride_n,
                      (void*)&cp_rank, (void*)&cp_world_size};
      cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, GLM_STREAM(ctx));
      status = cudaGetLastError();
    } else {
      fprintf(stderr, "glm_mla_kv_cache_append: unsupported indexer head_dim_ckv=%u\n", head_dim_ckv);
    }
    if (status != cudaSuccess) {
      fprintf(stderr, "glm_mla_kv_cache_append failed: %s\n", cudaGetErrorString(status));
    }
    return;
  }

  DISPATCH_MLA_HEAD_DIMS(head_dim_ckv, head_dim_kpe, {
    uint32_t bdx = HEAD_DIM_CKV / vec_size;
    auto kernel = flashinfer::AppendPagedKVMlaCacheKernel<HEAD_DIM_CKV, HEAD_DIM_KPE, vec_size, DType, IdType>;
    int num_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, bdx, 0);
    num_blocks_per_sm = std::min(num_blocks_per_sm, (int)((nnz + num_sms - 1) / num_sms));
    dim3 nblks(num_blocks_per_sm * num_sms);
    dim3 nthrs(bdx);
    void* args[] = {(void*)&paged_kv, (void*)&append_ckv, (void*)&append_kpe,
                    (void*)&batch_indices, (void*)&positions, (void*)&nnz,
                    (void*)&append_ckv_stride_n, (void*)&append_kpe_stride_n,
                    (void*)&cp_rank, (void*)&cp_world_size};
    cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, GLM_STREAM(ctx));
    status = cudaGetLastError();
  });

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_mla_kv_cache_append failed: %s\n", cudaGetErrorString(status));
  }
}

} // extern "C"

// ---------------------------------------------------------------------------
// Concat and Cache DS-MLA: BF16 ckv + kpe → packed FP8 paged cache
//
// Per-token layout (656 bytes for kv_lora_rank=512, pe_dim=64):
//   [0, kv_lora_rank)              FP8 e4m3 quantized ckv
//   [kv_lora_rank, +num_tiles*4)   num_tiles × FP32 per-128-block scales
//   [kv_lora_rank+num_tiles*4, BPT)  BF16 kpe (raw copy, not quantized)
// ---------------------------------------------------------------------------

template <int KV_LORA_RANK, int PE_DIM>
__global__ void concat_and_cache_ds_mla_kernel(
    uint8_t* __restrict__ kv_cache,
    const __nv_bfloat16* __restrict__ append_ckv,
    const __nv_bfloat16* __restrict__ append_kpe,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ indptr,
    const int32_t* __restrict__ batch_indices,
    const int32_t* __restrict__ positions,
    int page_size,
    size_t ckv_stride_n, size_t kpe_stride_n,
    uint32_t cp_world_size, uint32_t cp_rank
) {
    constexpr int SCALE_BLOCK = 128;
    constexpr int NUM_TILES = KV_LORA_RANK / SCALE_BLOCK;
    constexpr int THREADS_PER_TILE = SCALE_BLOCK / 8;
    constexpr int NOPE_THREADS = NUM_TILES * THREADS_PER_TILE;
    constexpr unsigned NOPE_MASK = (NOPE_THREADS >= 32) ? 0xFFFFFFFFu : ((1u << NOPE_THREADS) - 1u);
    constexpr int SCALE_BYTES = NUM_TILES * 4;
    constexpr int BPT = KV_LORA_RANK + SCALE_BYTES + PE_DIM * 2;
    constexpr float kFp8ScaleDivisor = 448.f;

    const int token_idx = blockIdx.x;

    const int batch = batch_indices[token_idx];
    int pos = positions[token_idx];

    // Context parallelism: each GPU stores every Nth token (interleaved).
    // Filter out tokens not belonging to this rank, and remap to local position.
    int eff_page_size = page_size;
    if (cp_world_size > 0) {
        if ((uint32_t)pos % cp_world_size != cp_rank) return;
        pos = ((uint32_t)pos - cp_rank) / cp_world_size;
        eff_page_size = page_size / (int)cp_world_size;
    }

    const int page_in_seq = pos / eff_page_size;
    const int offset_in_page = pos % eff_page_size;
    const int page_id = indices[indptr[batch] + page_in_seq];
    const size_t slot = (size_t)page_id * page_size + offset_in_page;

    uint8_t* dst = kv_cache + slot * BPT;
    const __nv_bfloat16* src_ckv = append_ckv + (size_t)token_idx * ckv_stride_n;
    const __nv_bfloat16* src_kpe = append_kpe + (size_t)token_idx * kpe_stride_n;

    if (threadIdx.x >= NOPE_THREADS) {
        const int pe_idx = (threadIdx.x - NOPE_THREADS) * 2;
        if (pe_idx < PE_DIM) {
            int32_t vals = *reinterpret_cast<const int32_t*>(&src_kpe[pe_idx]);
            *reinterpret_cast<int32_t*>(&dst[KV_LORA_RANK + SCALE_BYTES + pe_idx * 2]) = vals;
        }
        return;
    }

    const int tile_idx = threadIdx.x / THREADS_PER_TILE;
    const int lane_in_tile = threadIdx.x % THREADS_PER_TILE;

    const int src_offset = threadIdx.x * 8;
    int4 vals_i4 = *reinterpret_cast<const int4*>(&src_ckv[src_offset]);
    const __nv_bfloat162* vals2 = reinterpret_cast<const __nv_bfloat162*>(&vals_i4);

    // Max |val| over the 8 elements. bf16 magnitude order is preserved by the
    // conversion to float, so reduce in bf16x2 (abs+max, 2 lanes/op, tree-shaped)
    // instead of 8 scalar float converts on a serial fmaxf chain.
    __nv_bfloat162 m2 = __hmax2(__hmax2(__habs2(vals2[0]), __habs2(vals2[1])),
                                __hmax2(__habs2(vals2[2]), __habs2(vals2[3])));

    // Reduce across the 16-thread tile. A shuffle always moves a 32-bit lane, so
    // carry the packed bf16x2 (one __hmax2 per step) and collapse to a float only
    // once, after the reduction.
    #pragma unroll
    for (int mask = 8; mask > 0; mask /= 2) {
        m2 = __hmax2(m2, __shfl_xor_sync(NOPE_MASK, m2, mask, 16));
    }
    float max_abs = __bfloat162float(__hmax(m2.x, m2.y));

    float tile_scale = fmaxf(max_abs / kFp8ScaleDivisor, FLT_MIN);

    if (lane_in_tile == 0) {
        float* scale_dst = reinterpret_cast<float*>(&dst[KV_LORA_RANK]);
        scale_dst[tile_idx] = tile_scale;
    }

    // Quantize 2-at-a-time: bf16x2 -> float2 -> fp8x2, packed into two u32 stores.
    uint32_t packed[2] = {0u, 0u};
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float2 f = __bfloat1622float2(vals2[j]);
        f.x /= tile_scale;
        f.y /= tile_scale;
        uint16_t p = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
        packed[j >> 1] |= (uint32_t)p << ((j & 1) * 16);
    }
    *reinterpret_cast<uint32_t*>(&dst[src_offset]) = packed[0];
    *reinterpret_cast<uint32_t*>(&dst[src_offset + 4]) = packed[1];
}

extern "C" {

void glm_concat_and_cache_ds_mla(
    GlmCtx* ctx,
    void* kv_cache,
    void* append_ckv, void* append_kpe,
    int32_t* indices, int32_t* indptr,
    int32_t* batch_indices, int32_t* positions,
    uint32_t nnz, uint32_t page_size,
    uint32_t kv_lora_rank, uint32_t pe_dim,
    size_t append_ckv_stride_n, size_t append_kpe_stride_n,
    uint32_t cp_world_size, uint32_t cp_rank
) {
    cudaSetDevice(ctx->device_id);

    if (kv_lora_rank == 512 && pe_dim == 64) {
        constexpr int BLOCK_SIZE = 96;
        concat_and_cache_ds_mla_kernel<512, 64><<<nnz, BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
            (uint8_t*)kv_cache,
            (const __nv_bfloat16*)append_ckv,
            (const __nv_bfloat16*)append_kpe,
            indices, indptr, batch_indices, positions,
            page_size, append_ckv_stride_n, append_kpe_stride_n,
            cp_world_size, cp_rank);
    } else if (kv_lora_rank == 128 && pe_dim == 64) {
        constexpr int BLOCK_SIZE = 64;
        concat_and_cache_ds_mla_kernel<128, 64><<<nnz, BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
            (uint8_t*)kv_cache,
            (const __nv_bfloat16*)append_ckv,
            (const __nv_bfloat16*)append_kpe,
            indices, indptr, batch_indices, positions,
            page_size, append_ckv_stride_n, append_kpe_stride_n,
            cp_world_size, cp_rank);
    } else {
        fprintf(stderr, "glm_concat_and_cache_ds_mla: unsupported kv_lora_rank=%u pe_dim=%u\n",
                kv_lora_rank, pe_dim);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_concat_and_cache_ds_mla failed: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"
