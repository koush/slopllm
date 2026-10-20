#include "glm_ops.h"
#include "glm_indexer_cache.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
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
constexpr auto POS_ENC = flashinfer::PosEncodingMode::kNone;

#undef DISPATCH_HEAD_DIM
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
      true, CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENC, false, MASK_MODE, false,
      AttentionVariant,
      flashinfer::BatchPrefillPagedParams<DType, DType, DTypeO, IdType>>(
      params, tmp_v, tmp_s, enable_pdl, stream);
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, flashinfer::MaskMode MASK_MODE>
cudaError_t dispatch_batch_prefill_ragged_run_inner(
    flashinfer::BatchPrefillRaggedParams<DType, DType, DTypeO, IdType>& params,
    DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream) {
  return flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
      CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENC, false, MASK_MODE, false,
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

template <int HEAD_DIM>
__global__ void indexer_kv_cache_append_kernel(
    uint8_t* __restrict__ k_data,
    float* __restrict__ k_scale_data,
    const __nv_bfloat16* __restrict__ append_k,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ indptr,
    const int32_t* __restrict__ batch_indices,
    const int32_t* __restrict__ positions,
    uint32_t nnz, uint32_t page_size, size_t append_stride_n,
    uint32_t cp_world_size, uint32_t cp_rank) {
    const uint32_t token_idx = blockIdx.x;
    if (token_idx >= nnz) return;

    int pos = positions[token_idx];
    int effective_page_size = page_size;
    if (cp_world_size > 1) {
        if ((uint32_t)pos % cp_world_size != cp_rank) return;
        pos = (pos - (int)cp_rank) / (int)cp_world_size;
        effective_page_size = page_size / (int)cp_world_size;
    }
    const int batch = batch_indices[token_idx];
    const int page_id = indices[indptr[batch] + pos / effective_page_size];
    const size_t slot = (size_t)page_id * effective_page_size + pos % effective_page_size;
    uint8_t* dst = k_data + slot * HEAD_DIM;
    float* dst_scale = k_scale_data + slot;
    const __nv_bfloat16* src = append_k + (size_t)token_idx * append_stride_n;
    __shared__ float scratch[4];
    pack_indexer_k_row<HEAD_DIM>(dst, dst_scale, src, scratch);
}

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
          false,
          AttentionVariant, Params>(
          params, static_cast<DTypeO*>(tmp), GLM_STREAM(ctx));
    } else {
      status = flashinfer::SinglePrefillWithKVCacheDispatched<
          HEAD_DIM, HEAD_DIM,
          flashinfer::PosEncodingMode::kNone,
          false,
          flashinfer::MaskMode::kNone,
          false,
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
      0, // uniform_q_len
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
      0, // uniform_q_len
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

  cudaError_t status = cudaSuccess;

  // Indexer K-only append: BF16 source -> packed FP8 E4M3 row + FP32 scale.
  if (head_dim_kpe == 0) {
    if (head_dim_ckv == 128) {
      indexer_kv_cache_append_kernel<128><<<nnz, 64, 0, GLM_STREAM(ctx)>>>(
          (uint8_t*)ckv_data, (float*)kpe_data, (const __nv_bfloat16*)append_ckv,
          indices, indptr, batch_indices, positions, nnz, page_size,
          append_ckv_stride_n, cp_world_size, cp_rank);
    } else if (head_dim_ckv == 64) {
      indexer_kv_cache_append_kernel<64><<<nnz, 32, 0, GLM_STREAM(ctx)>>>(
          (uint8_t*)ckv_data, (float*)kpe_data, (const __nv_bfloat16*)append_ckv,
          indices, indptr, batch_indices, positions, nnz, page_size,
          append_ckv_stride_n, cp_world_size, cp_rank);
    } else {
      fprintf(stderr, "glm_mla_kv_cache_append: unsupported indexer head_dim_ckv=%u\n", head_dim_ckv);
    }
    if (status != cudaSuccess) {
      fprintf(stderr, "glm_mla_kv_cache_append failed: %s\n", cudaGetErrorString(status));
    }
    return;
  }

  fprintf(stderr, "glm_mla_kv_cache_append: only indexer K cache append is supported\n");
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
__device__ __forceinline__ void pack_ds_mla_row(
    uint8_t* dst,
    const __nv_bfloat16* src_ckv,
    const __nv_bfloat16* src_kpe
) {
    constexpr int SCALE_BLOCK = 128;
    constexpr int NUM_TILES = KV_LORA_RANK / SCALE_BLOCK;
    constexpr int THREADS_PER_TILE = SCALE_BLOCK / 8;
    constexpr int NOPE_THREADS = NUM_TILES * THREADS_PER_TILE;
    constexpr unsigned NOPE_MASK = (NOPE_THREADS >= 32) ? 0xFFFFFFFFu : ((1u << NOPE_THREADS) - 1u);
    constexpr int SCALE_BYTES = NUM_TILES * 4;
    constexpr float kFp8ScaleDivisor = 448.f;

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

    __nv_bfloat162 m2 = __hmax2(__hmax2(__habs2(vals2[0]), __habs2(vals2[1])),
                                __hmax2(__habs2(vals2[2]), __habs2(vals2[3])));
    #pragma unroll
    for (int mask = 8; mask > 0; mask /= 2) {
        m2 = __hmax2(m2, __shfl_xor_sync(NOPE_MASK, m2, mask, 16));
    }
    float max_abs = __bfloat162float(__hmax(m2.x, m2.y));
    float tile_scale = fmaxf(max_abs / kFp8ScaleDivisor, FLT_MIN);

    if (lane_in_tile == 0) {
        reinterpret_cast<float*>(&dst[KV_LORA_RANK])[tile_idx] = tile_scale;
    }

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
    constexpr int SCALE_BYTES = (KV_LORA_RANK / 128) * 4;
    constexpr int BPT = KV_LORA_RANK + SCALE_BYTES + PE_DIM * 2;

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

    size_t slot;
    if (indices == nullptr) {
        // Flat (gathered) destination: the buffer is indexed by the SAME flat
        // slot the gather wrote and sparse MLA reads, i.e. a de-interleaved
        // TOKEN prefix sum. `indptr` is therefore kv_token_indptr in this mode,
        // not the page indptr — scaling by eff_page_size here would only agree
        // for batch 0, and silently corrupt gathered history for every forked
        // sequence (MTP's draft passes).
        slot = (size_t)indptr[batch] + pos;
    } else {
        const int page_in_seq = pos / eff_page_size;
        const int offset_in_page = pos % eff_page_size;
        const int page_id = indices[indptr[batch] + page_in_seq];
        slot = (size_t)page_id * eff_page_size + offset_in_page;
    }

    uint8_t* dst = kv_cache + slot * BPT;
    const __nv_bfloat16* src_ckv = append_ckv + (size_t)token_idx * ckv_stride_n;
    const __nv_bfloat16* src_kpe = append_kpe + (size_t)token_idx * kpe_stride_n;
    pack_ds_mla_row<KV_LORA_RANK, PE_DIM>(dst, src_ckv, src_kpe);
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
