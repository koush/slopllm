#include "glm_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/mask.cuh>
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
constexpr uint32_t HEAD_DIM = 128;
constexpr auto POS_ENC = flashinfer::PosEncodingMode::kNone;

namespace {

template <uint32_t GROUP_SIZE>
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

template <uint32_t CTA_TILE_Q, flashinfer::MaskMode MASK_MODE>
cudaError_t dispatch_batch_prefill_ragged_run_inner(
    flashinfer::BatchPrefillRaggedParams<DType, DType, DTypeO, IdType>& params,
    DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream) {
  return flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
      CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENC, false, MASK_MODE,
      AttentionVariant,
      flashinfer::BatchPrefillRaggedParams<DType, DType, DTypeO, IdType>>(
      params, tmp_v, tmp_s, enable_pdl, stream);
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

void* glm_alloc_pinned(size_t bytes) {
  void* ptr = nullptr;
  cudaError_t status = cudaMallocHost(&ptr, bytes);
  if (status != cudaSuccess) {
    fprintf(stderr, "glm_alloc_pinned failed: %s\n", cudaGetErrorString(status));
    return nullptr;
  }
  return ptr;
}

void glm_free_pinned(void* ptr) {
  if (ptr) cudaFreeHost(ptr);
}

void glm_write_pinned(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void glm_batch_decode_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* indptr_h,
    uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t page_size) {

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
      case 1: return dispatch_decode_work_est<1>(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s);
      case 2: return dispatch_decode_work_est<2>(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s);
      case 4: return dispatch_decode_work_est<4>(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s);
      case 8: return dispatch_decode_work_est<8>(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s);
      case 16: return dispatch_decode_work_est<16>(split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, bs, kv_indptr, nqh, ps, ecg, s);
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
      false, // enable_cuda_graph
      ctx->stream,
      work_est);

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_batch_decode_plan failed: %s\n", cudaGetErrorString(status));
    return;
  }

  auto vec = info.ToVector();
  memcpy(plan_info, vec.data(), sizeof(int64_t) * vec.size());
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

  cudaError_t status =
      flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, POS_ENC, AttentionVariant, DecodeParams>(
          params, tmp_v, tmp_s, false, ctx->stream);

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_batch_decode_run failed: %s\n", cudaGetErrorString(status));
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
      1, // page_size (ragged)
      false, // enable_cuda_graph
      2, // sizeof_dtype_o (bf16 = 2 bytes)
      -1, // window_left
      -1, // fixed_split_size
      false, // disable_split_kv
      0, // num_colocated_ctas
      ctx->stream);

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
    uint32_t q_stride_n, uint32_t q_stride_h,
    uint32_t kv_stride_n, uint32_t kv_stride_h,
    int mask_mode, float sm_scale) {

  using PrefillParams = flashinfer::BatchPrefillRaggedParams<DType, DType, DTypeO, IdType>;

  flashinfer::PrefillPlanInfo info;
  info.FromVector(std::vector<int64_t>(plan_info, plan_info + 15));
  flashinfer::MaskMode flash_mask = static_cast<flashinfer::MaskMode>(mask_mode);

  PrefillParams params(
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
      num_qo_heads,
      num_kv_heads,
      q_stride_n,
      q_stride_h,
      kv_stride_n,
      kv_stride_h,
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

  if (flash_mask == flashinfer::MaskMode::kCausal) {
    switch (cta_tile_q) {
      case 128: status = dispatch_batch_prefill_ragged_run_inner<128, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, ctx->stream); break;
      case 64: status = dispatch_batch_prefill_ragged_run_inner<64, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, ctx->stream); break;
      case 16: status = dispatch_batch_prefill_ragged_run_inner<16, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, ctx->stream); break;
      case 1: status = dispatch_batch_prefill_ragged_run_inner<1, flashinfer::MaskMode::kCausal>(params, tmp_v, tmp_s, false, ctx->stream); break;
      default: fprintf(stderr, "Unsupported cta_tile_q: %u\n", cta_tile_q); status = cudaErrorInvalidValue;
    }
  } else {
    switch (cta_tile_q) {
      case 128: status = dispatch_batch_prefill_ragged_run_inner<128, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, ctx->stream); break;
      case 64: status = dispatch_batch_prefill_ragged_run_inner<64, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, ctx->stream); break;
      case 16: status = dispatch_batch_prefill_ragged_run_inner<16, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, ctx->stream); break;
      case 1: status = dispatch_batch_prefill_ragged_run_inner<1, flashinfer::MaskMode::kNone>(params, tmp_v, tmp_s, false, ctx->stream); break;
      default: fprintf(stderr, "Unsupported cta_tile_q: %u\n", cta_tile_q); status = cudaErrorInvalidValue;
    }
  }

  if (status != cudaSuccess) {
    fprintf(stderr, "glm_batch_prefill_ragged_run failed: %s\n", cudaGetErrorString(status));
  }
}

} // extern "C"
