#ifndef GLM_OPS_H
#define GLM_OPS_H

#include <stdint.h>
#include <stddef.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM_MAX_STREAMS 64

struct GlmCtx {
    int device_id;
    cudaStream_t streams[GLM_MAX_STREAMS];
    int active_stream;
    cudaEvent_t events[GLM_MAX_STREAMS];
    void* cublas_handle;
    void* cublaslt_handle;
};

typedef struct GlmCtx GlmCtx;

#define GLM_STREAM(ctx) ((ctx)->streams[(ctx)->active_stream])

GlmCtx* glm_init(int device_id);
void glm_free(GlmCtx* ctx);

void* glm_alloc(GlmCtx* ctx, size_t bytes);
void glm_free_buf(GlmCtx* ctx, void* ptr);
cudaError_t glm_cuda_ipc_get_mem_handle(GlmCtx* ctx, cudaIpcMemHandle_t* handle, void* ptr);
cudaError_t glm_cuda_ipc_open_mem_handle(GlmCtx* ctx, void** ptr, const cudaIpcMemHandle_t* handle);
cudaError_t glm_cuda_ipc_close_mem_handle(GlmCtx* ctx, void* ptr);

void glm_h2d(GlmCtx* ctx, void* dst, const void* src, size_t bytes);
void glm_d2h(GlmCtx* ctx, void* dst, const void* src, size_t bytes);
void glm_write_pointers(GlmCtx* ctx, void* dst,
                        void* p0, void* p1, void* p2, void* p3,
                        void* p4, void* p5, void* p6, void* p7,
                        int n);

void* glm_mmap_open(const char* path);
void glm_mmap_close(void* ptr, uint64_t size);
void glm_mmap_load(GlmCtx* ctx, void* gpu_dst, const void* mmap_ptr,
                   uint64_t offset, uint64_t nbytes);

void glm_rmsnorm(GlmCtx* ctx, void* out, const void* input,
                 const void* weight, float eps, int dim, int batch);

void glm_fused_add_rmsnorm(GlmCtx* ctx, void* out, void* residual,
                            const void* input_a, const void* input_b,
                            const void* weight, float eps, int dim, int batch);

// Fused per-head RMSNorm + RoPE with layout transpose
// Input: [batch * seq_len, n_heads * head_dim] (projection output)
// Output: [batch, n_heads, seq_len, head_dim] (HND layout for attention)
// Applies per-head RMSNorm then RoPE, with [NSHD -> HNSD] transpose.
// in_stride: per-head stride in input (head_dim for contiguous, head_dim*2 for interleaved [Q|gate])
void glm_fused_norm_rope(GlmCtx* ctx, void* out, const void* in,
                          const void* weight, const void* cos_emb, const void* sin_emb,
                          float eps, int rope_dim, int head_dim,
                          int n_heads, int seq_len, int batch, int in_stride, bool interleaved);

void glm_silu_and_mul(GlmCtx* ctx, void* out, const void* gate,
                      const void* up, int intermediate, int batch);

void glm_linear(GlmCtx* ctx, void* out, const void* input,
                const void* weight, int batch, int n, int k,
                void* workspace, size_t workspace_size);

void glm_layernorm(GlmCtx* ctx, void* out, const void* input,
                   const void* weight, const void* bias, float eps, int dim, int batch);

void glm_relu(GlmCtx* ctx, void* out, const void* input, int n);

void glm_sigmoid(GlmCtx* ctx, void* out, const void* input, int n);

void glm_softmax(GlmCtx* ctx, void* out, const void* input,
                 const void* mask, int dim, int batch);

void glm_indexer_score(GlmCtx* ctx, void* out, const void* q, const void* kData,
                       const float* kScaleData, const void* weights, const int32_t* pageIndices,
                       const int32_t* pageIndptr, const int32_t* lastPageLen,
                       const int32_t* qoIndptr, float scale,
                       int totalQ, int idxNHeads, int idxHeadDim,
                       int pageSize, int maxKvLen, int causal,
                       const int32_t* kvTokenIndptr);

void glm_indexer_kv_cache_append_flat(GlmCtx* ctx, void* kData, float* kScaleData,
    const void* appendK, const int32_t* kvTokenIndptr,
    const int32_t* batchIndices, const int32_t* positions,
    uint32_t nnz, uint32_t headDim, size_t appendStrideN);



void glm_indexer_score_topk_prefill(GlmCtx* ctx, int32_t* out_idx,
    __nv_bfloat16* out_scores,
    const void* q, const void* kData, const float* kScaleData, const void* weights,
    const int32_t* pageIndices, const int32_t* pageIndptr,
    const int32_t* lastPageLen, const int32_t* qoIndptr,
    float scale, int totalQ, int idxNHeads, int idxHeadDim,
    int pageSize, int topk, int causal, int qGlobalStart,
    const uint8_t* custom_mask, const int32_t* mask_indptr, const int32_t* mask_kv_len,
    void* scores, int32_t* rowLen, int maxKv,
    int32_t* coarseHist, int32_t* fineHist, int32_t* meta,
    int queryTiles, int cpWorldSize, int cpRank,
    const int32_t* globalLastPageLen, const int32_t* kvTokenIndptr);

void glm_indexer_score_topk_v2(GlmCtx* ctx, int32_t* out_idx,
    __nv_bfloat16* out_scores,
    const void* q, const void* kData, const float* kScaleData, const void* weights,
    const int32_t* pageIndices, const int32_t* pageIndptr,
    const int32_t* lastPageLen, const int32_t* qoIndptr,
    float scale, int totalQ, int idxNHeads, int idxHeadDim,
    int pageSize, int topk, int causal, int qGlobalStart,
    const uint8_t* custom_mask, const int32_t* mask_indptr, const int32_t* mask_kv_len,
    void* scores, int32_t* rowLen, int32_t* hist, int32_t* meta,
    int maxKv, int num_splits, int cpWorldSize, int cpRank,
    const int32_t* globalLastPageLen, const int32_t* kvTokenIndptr);

void glm_sort_topk_by_index(GlmCtx* ctx, int32_t* out_idx,
    __nv_bfloat16* out_scores, int batch, int topk);

void glm_topk_from_scores(GlmCtx* ctx, int32_t* out_idx,
    __nv_bfloat16* out_scores,
    const void* scores, const int32_t* row_len,
    int32_t* hist, int32_t* meta,
    int batch, int stride, int topk, int num_splits,
    int cpWorldSize, int cpRank);

void glm_topk_to_slots(GlmCtx* ctx, int32_t* slots, int32_t* topk_length, const int32_t* topk_idx,
                       const int32_t* page_indices, const int32_t* page_indptr,
                       const int32_t* last_page_len, const int32_t* batch_indices,
                       int num_tokens, int topk, int page_size,
                       uint32_t cp_world_size, uint32_t cp_rank,
                       const int32_t* kv_token_indptr);

cudaError_t glm_fill(GlmCtx* ctx, void* out, double value, int n, const char* dtype);
// Up to eight device ranges, issued concurrently within one bounded grid.
void glm_prefetch_l2(GlmCtx* ctx, const void* const* data, const size_t* bytes, int count);

void glm_gather(GlmCtx* ctx, void* out, const void* input, const int* indices,
                int k, int in_dim, int batch, int elem_size);

void glm_scatter_scalar(GlmCtx* ctx, void* out, const int* indices, float value,
                        int k, int out_dim, int batch);

void glm_deinterleave(GlmCtx* ctx, void* out, const void* in,
                      int world_size, int max_total_len,
                      const int32_t* page_indptr,
                      const int32_t* kv_token_indptr,
                      int batch_size, int page_size, int D);

void glm_gather_pages(GlmCtx* ctx, void* out, const void* in,
                      const int32_t* page_indices,
                      const int32_t* page_indptr,
                      const int32_t* last_page_len,
                      int max_pages, int batch_size,
                      int page_size, int D);

void glm_causal_mask(GlmCtx* ctx, void* out, int seq_len);

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
                              int rope_dim, int head_dim, int n_heads, int seq_len,
                              int batch, int unsqueeze_dim, int in_stride, bool interleaved);

// MLA V-Expand: per-head matmul attn_out @ v_proj^T
// attn_out: [B, attn_n_heads, S, kv_lora_rank] (HND)
// v_proj: [n_heads, kv_lora_rank, v_head_dim] (transposed layout for coalesced access)
// result: [B, S, n_heads * v_head_dim]
// head_offset: offset into attn_n_heads dimension of attn_out for this shard
void glm_mla_v_expand(GlmCtx* ctx, void* result, const void* attn_out,
                       const void* v_proj,
                       int kv_lora_rank, int v_head_dim, int n_heads,
                       int seq_len, int batch,
                       int attn_n_heads, int head_offset,
                       int v_proj_head_offset);

void glm_route_top8(GlmCtx* ctx, void* out_weights, int* out_indices,
                    const void* logits, const void* bias, int rows, float scale, bool normalize);

void glm_topk(GlmCtx* ctx, void* out_values, int* out_indices,
              const void* input, int k, int dim, int batch, int offset);

void glm_bmm(GlmCtx* ctx, void* C, const void* A, const void* B,
             float alpha, float beta,
             int batch, int M, int N, int K, int transA, int transB, int tokenMajor = 0);

void glm_scale(GlmCtx* ctx, void* out, const void* input, float scale, int n);

void glm_add(GlmCtx* ctx, void* out, const void* a, const void* b, int n);

void glm_add_broadcast(GlmCtx* ctx, void* out, const void* a, const void* b, int dim, int rows);

void glm_expand_dim1(GlmCtx* ctx, void* out, const void* input,
                     int dim1_out, int dim1_in, int seq_len, int head_dim, int batch);

void glm_expand_dim1_strided(GlmCtx* ctx, void* out, const void* input,
                             int dim1_out, int dim1_in, int seq_len, int head_dim,
                             int batch, int head_stride);

void glm_transpose_4d(GlmCtx* ctx, void* out, const void* input,
                       int dim0, int dim1, int dim2, int dim3,
                       int perm0, int perm1, int perm2, int perm3);
void glm_transpose_4d_typed(GlmCtx* ctx, void* out, const void* input,
                            int dim0, int dim1, int dim2, int dim3,
                            int perm0, int perm1, int perm2, int perm3,
                            int elem_bytes);

void glm_mul(GlmCtx* ctx, void* out, const void* a, const void* b, int n);

void glm_mul_broadcast(GlmCtx* ctx, void* out, const void* a, const void* b, int dim, int rows);

void glm_reduce_sum(GlmCtx* ctx, void* out, const void* input, int rows, int cols);

void glm_row_normalize(GlmCtx* ctx, void* out, const void* input,
                        float scale, int rows, int cols, bool normalize);

void glm_group_mask_mul(GlmCtx* ctx, void* scores, const void* group_mask,
                         int num_experts, int experts_per_group, int n_group, int batch);

void glm_mul_mat_id(GlmCtx* ctx, void* output, const void* input,
                     const void* const* weight_ptrs,
                     const int* expert_ids, int top_k,
                     int count, int N, int K);

void glm_nvfp4_mul_mat_id(GlmCtx* ctx, void* output, const void* input,
                             const void* const* weight_ptrs,
                             const void* const* scale_ptrs,
                             const void* const* scale2_ptrs,
                             const int* expert_ids, int top_k,
                             int count, int N, int K);

// Fused NVFP4 down + weighted reduction: N=6144, K=256, top_k=8, 1..32 rows.
// Preserves BF16 rounding of each down result before the weighted sum.
void glm_nvfp4_mul_mat_id_reduce(GlmCtx* ctx, void* output, const void* input,
                                const void* const* weight_ptrs, const void* const* scale_ptrs,
                                const void* const* scale2_ptrs, const int* expert_ids,
                                const void* routing_weights, int num_rows);

void glm_scatter_add_rows(GlmCtx* ctx, void* out, const void* input,
                             const void* scales, int top_k,
                             int dim, int num_rows, void* workspace);

// Rotate input IDs for MTP prefill: shifts each sequence left by 1,
// appends new_token at the last position.
// output_ids/input_ids: [totalTokens] I32
// qo_indptr: [batchSize+1] I32 cumulative offsets
// new_tokens: [batchSize] I32 new token per sequence
void glm_rotate_input_ids(GlmCtx* ctx, int* output_ids, const int* input_ids,
                           const int* qo_indptr, const int* new_tokens,
                           int batch_size);

// Element-wise sum of N tensors (max 8). Pointers passed as kernel args
// for CUDA graph compatibility. dtype: 9=BF16, 7=F32.
void glm_sum_pointers(GlmCtx* ctx,
    void* p0,  void* p1,  void* p2,  void* p3,
    void* p4,  void* p5,  void* p6,  void* p7,
    void* output, int N, int64_t numel, int dtype, bool writeback = false);

// Fused P2P AllReduce + Add + RMSNorm. Reads N peer partial-sum pointers
// (smem-staged, same read pattern as glm_sum_pointers), accumulates them in
// registers, adds inputA (residual), then RMSNorms each row of `dim` elements:
//   s        = inputA + sum_{peer<N} p_peer
//   residual = s
//   out      = weight * s * rsqrt(mean(s^2) + eps)
// Constraints: dim % 512 == 0 and 8192 % dim == 0 (tiles stay row-aligned).
// dtype: 9=BF16, 7=F32.
void glm_rmsnorm_pointers_smem(GlmCtx* ctx,
    const void* p0,  const void* p1,  const void* p2,  const void* p3,
    const void* p4,  const void* p5,  const void* p6,  const void* p7,
    const void* inputA, const void* weight,
    void* out, void* residual,
    int N, int64_t numel, int dim, float eps, int dtype);

void glm_index_select(GlmCtx* ctx, void* out, const void* src,
                       const void* indices, int dim, int k, int offset);

void glm_arange(GlmCtx* ctx, int* out, int start, int step, int count);

void glm_max(GlmCtx* ctx, void* out_values, int* out_indices, const void* input, int dim, int batch, int offset);

void glm_memcpy(GlmCtx* ctx, void* dst, const void* src, size_t bytes, int kind);

// 2D memory copy (async on stream)
// kind: MemcpyKind values (0=H2H, 1=H2D, 2=D2H, 3=D2D, 4=Default)
void glm_memcpy2d(GlmCtx* ctx, void* dst, size_t dpitch,
                  const void* src, size_t spitch,
                  size_t width, size_t height, int kind);

// Peer-to-peer memory copy (async on stream, explicit src/dst devices)
void glm_memcpy_peer(GlmCtx* ctx, void* dst, int dstDevice,
                     const void* src, int srcDevice, size_t bytes);

// 3D peer-to-peer memory copy (async on stream)
void glm_memcpy3d_peer(GlmCtx* ctx,
    void* dstPtr, size_t dstPitch, size_t dstXSize, size_t dstYSize, int dstDevice,
    size_t dstPosX, size_t dstPosY, size_t dstPosZ,
    const void* srcPtr, size_t srcPitch, size_t srcXSize, size_t srcYSize, int srcDevice,
    size_t srcPosX, size_t srcPosY, size_t srcPosZ,
    size_t width, size_t height, size_t depth);

// ---------------------------------------------------------------------------
// NCCL operations
// ---------------------------------------------------------------------------

#define GLM_NCCL_UNIQUE_ID_BYTES 128

// Write NCCL unique ID (128 bytes) to out_id (host memory)
void glm_nccl_unique_id(void* out_id);

// Group start/end: fuse multiple NCCL operations across devices
int glm_nccl_group_start();
int glm_nccl_group_end();

// Initialize NCCL communicator for a given rank
// Returns opaque ncclComm_t pointer (0 on failure)
void* glm_nccl_comm_init_rank(int device_id, int rank, int world_size, const void* unique_id);

// Initialize NCCL communicators for all devices in a single process
// comms: output array of ndev void* pointers (caller-allocated)
// devlist: array of ndev device IDs
// Returns 0 on success, -1 on failure
int glm_nccl_comm_init_all(void** comms, int ndev, const int* devlist);

// Destroy NCCL communicator
void glm_nccl_comm_destroy(void* comm);

// All-reduce: sendbuff and recvbuff are device pointers
// datatype: ncclDataType_t values (7=float32, 9=bfloat16)
// op: ncclRedOp_t values (0=sum, 1=prod, 2=max, 3=min)
void glm_nccl_all_reduce(void* comm, GlmCtx* ctx,
                          const void* sendbuff, void* recvbuff,
                          size_t count, int datatype, int op);

// All-gather: sendbuff and recvbuff are device pointers
// recvbuff must be world_size * sendcount elements
void glm_nccl_all_gather(void* comm, GlmCtx* ctx,
                          const void* sendbuff, void* recvbuff,
                          size_t count, int datatype);

// Send/Recv: point-to-point communication between ranks
// sendbuff/recvbuff are device pointers, count is number of elements
void glm_nccl_send(void* comm, GlmCtx* ctx,
                    const void* sendbuff, size_t count, int datatype, int peer);
void glm_nccl_recv(void* comm, GlmCtx* ctx,
                    void* recvbuff, size_t count, int datatype, int peer);

// Reduce-scatter: sendbuff and recvbuff are device pointers
// recvcount is the number of elements each rank receives (total / world_size)
// datatype: ncclDataType_t values (7=float32, 9=bfloat16)
// op: ncclRedOp_t values (0=sum, 1=prod, 2=max, 3=min)
void glm_nccl_reduce_scatter(void* comm, GlmCtx* ctx,
                              const void* sendbuff, void* recvbuff,
                              size_t recvcount, int datatype, int op);

// ---------------------------------------------------------------------------
// Custom P2P AllReduce (small messages, single-process multi-GPU).
//
// On systems where NCCL ring AllReduce is latency-bound (PCIe-only, no
// NVLink), this primitive completes ~3-5x faster for small payloads by
// using direct peer-mapped reads.
//
// Usage:
//   1) For each (rank, peer) pair where rank != peer:
//        glm_p2p_enable_peer_access(ctx[rank], peer)
//   2) For each rank:
//        inst[rank] = glm_p2p_create_instance(ctx[rank], rank, world)
//   3) Allocate data buffers (2 * slot_bytes per rank, double-buffered):
//        data_bufs[rank] = cudaMalloc(2 * slot_bytes)
//   4) Set max_bytes and build peer pointer arrays:
//        glm_p2p_set_max_bytes(inst[rank], slot_bytes)
//        glm_p2p_set_peers(ctx[rank], inst[rank], data_ptrs, flag_ptrs)
//   5) For each AllReduce, every rank calls (in lock-step program order):
//        glm_p2p_allreduce(ctx[rank], inst[rank], in, out, count, dtype)
//
// dtype follows NCCL convention: 9 = bfloat16, 7 = float32.
// ---------------------------------------------------------------------------

struct GlmP2PInstance {
    int**               peer_flags_arr_d;
    unsigned long long* seq_counter_d;
    int*                my_flags_d;  // int[world_size]: rank's flag array, peers write into my_flags_d[their_rank]
    void*               metadata_alloc_d;
    int                 world_size;
    int                 my_rank;
    int                 device_id;
    int                 nanosleep_ns;
};

static constexpr int P2P_AR_MAX_WORLD = 8;

// Enable peer access from this ctx's device to peer_device. Idempotent.
// Returns 0 on success, -1 if peer access cannot be enabled.
int glm_p2p_enable_peer_access(GlmCtx* ctx, int peer_device);

// Create a P2P instance on this device (metadata only, no data buffer).
// Call glm_p2p_set_peers and glm_p2p_set_max_bytes before use.
GlmP2PInstance* glm_p2p_create_instance(GlmCtx* ctx, int my_rank,
                                         int world_size, const int* device_ids);

// Free instance state.
void glm_p2p_destroy_instance(GlmP2PInstance* inst);

// Get this rank's peer-visible flag array pointer (int[world_size]).
int* glm_p2p_get_flag_ptr(GlmP2PInstance* inst);

// Configure this rank's view of all peers' flag pointers.
// peer_flag_ptrs[r] = device pointer (on rank r) to rank r's flag array (int[world_size]).
void glm_p2p_set_peers(GlmCtx* ctx, GlmP2PInstance* inst,
                       int* const* peer_flag_ptrs);

// Write-based AllGather (Row layout). Caller issues arrive before and wait
// after. Each GPU reads its local shard and writes to all N peers' outputs.
//   peer_j_output + row * full_dim1_bytes + rank * shard_dim1_bytes
void glm_p2p_allgather_row_write(GlmCtx* ctx,
    const void* local_shard,
    const void* p0,  const void* p1,  const void* p2,  const void* p3,
    const void* p4,  const void* p5,  const void* p6,  const void* p7,
    void* output, int N, int shard_dim1_bytes, int full_dim1_bytes, int outer, int rank,
    int src_peer_stride_bytes);

// Push-based reduce-scatter AllReduce (write+write), phase 1: scatter. Each GPU
// writes its N chunks to peers' staging buffers [N, chunkLen] at slot `rank`.
// Peer order rotated by rank (permutation schedule, single-block-safe).
void glm_p2p_reduce_scatter_write(GlmCtx* ctx,
    const void* local_shard,
    void* p0, void* p1, void* p2, void* p3,
    void* p4, void* p5, void* p6, void* p7,
    int N, int chunk_bytes, int rank);

// Push-based reduce-scatter AllReduce, phase 2: reduce local staging [N, chunkLen]
// and write reduced chunk `rank` to every peer's output at offset rank*chunk_len.
void glm_p2p_reduce_gather_write(GlmCtx* ctx,
    const void* staging,
    void* p0, void* p1, void* p2, void* p3,
    void* p4, void* p5, void* p6, void* p7,
    int N, int chunk_len, int rank, int dtype);

// P2P barrier: single fused kernel that increments the seq counter, publishes
// the flag, and spins on peers' flags. Same seq-counter progression as the
// split pair below, one launch instead of two.
void glm_p2p_barrier(GlmCtx* ctx, GlmP2PInstance* inst, int peer_rank = -1);

// P2P barrier split into two phases. Between arrive and wait the caller may
// launch other work on the same stream; arrive's release makes prior writes
// visible, wait's acquire sees peers' writes before subsequent reads.
void glm_p2p_arrive(GlmCtx* ctx, GlmP2PInstance* inst, int peer_rank = -1);
void glm_p2p_wait(GlmCtx* ctx, GlmP2PInstance* inst, int peer_rank = -1);

void glm_kv_cache_write(GlmCtx* ctx,
                         void* src_k, void* src_v,
                         void* dst_k, void* dst_v,
                         int32_t* slot_mapping,
                         uint32_t batch_size, uint32_t n_kv,
                         uint32_t hd, uint32_t page_size,
                         uint32_t src_k_token_stride, uint32_t src_k_head_stride,
                         uint32_t src_v_token_stride, uint32_t src_v_head_stride);

cudaError_t glm_synchronize(GlmCtx* ctx);

cudaError_t glm_synchronize_stream(GlmCtx* ctx, int stream_idx);

void glm_set_stream(GlmCtx* ctx, int stream_idx);

void glm_event_record(GlmCtx* ctx, int event_idx, int stream_idx);

void glm_stream_wait_event(GlmCtx* ctx, int stream_idx, int event_idx);

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

void glm_batch_prefill_ragged_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* qo_indptr_h, int32_t* kv_indptr_h,
    uint32_t total_qo_rows, uint32_t batch_size,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim, int mask_mode);

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
    int mask_mode, float sm_scale);

// ---------------------------------------------------------------------------
// MLA (Multi-head Latent Attention) operations
// ---------------------------------------------------------------------------

// MLA Prefill: Plan phase
// Allocates workspace and computes scheduling metadata for MLA prefill.
// plan_info: output array of at least 18 int64_t elements (MLAPlanInfo)
void glm_mla_prefill_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* qo_indptr_h, int32_t* kv_indptr_h, int32_t* kv_len_h,
    uint32_t batch_size, uint32_t num_heads, uint32_t head_dim_o,
    bool causal, uint32_t cp_world_size = 0, uint32_t cp_rank = 0);

// MLA Prefill: Run phase
// Executes MLA paged attention (prefill/incremental-prefill).
// q_nope: [packed_qo_len, num_heads, head_dim_ckv] BF16
// q_pe:   [packed_qo_len, num_heads, head_dim_kpe] BF16
// ckv_data: [num_pages, page_size, head_dim_ckv] BF16
// kpe_data: [num_pages, page_size, head_dim_kpe] BF16
// o: [packed_qo_len, num_heads, head_dim_ckv] BF16
// lse: [packed_qo_len, num_heads] F32 (optional, pass nullptr to skip)
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
    void* custom_mask = nullptr, int32_t* mask_indptr = nullptr,
    int32_t* mask_kv_len = nullptr);

// MLA Decode: Plan phase
// plan_info: output array of at least 10 int64_t elements (DecodePlanInfo)
void glm_mla_decode_plan(
    GlmCtx* ctx,
    void* float_ws, size_t float_ws_size,
    void* int_ws, void* pinned_int_ws, size_t int_ws_size,
    int64_t* plan_info,
    int32_t* indptr_h,
    uint32_t batch_size, uint32_t num_qo_heads,
    uint32_t page_size, bool enable_cuda_graph,
    uint32_t head_dim_ckv, uint32_t head_dim_kpe);

// MLA Decode: Run phase
// q_nope: [batch_size, num_heads, head_dim_ckv] BF16
// q_pe:   [batch_size, num_heads, head_dim_kpe] BF16
// ckv_data: [num_pages, page_size, head_dim_ckv] BF16
// kpe_data: [num_pages, page_size, head_dim_kpe] BF16
// o: [batch_size, num_heads, head_dim_ckv] BF16
// lse: [batch_size, num_heads] F32 (optional, pass nullptr to skip)
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
    float* lse);

// MLA: Append entries to paged KV cache
// append_ckv: [nnz, head_dim_ckv] BF16
// append_kpe: [nnz, head_dim_kpe] BF16
void glm_mla_kv_cache_append(
    GlmCtx* ctx,
    void* ckv_data, void* kpe_data,
    int32_t* indices, int32_t* indptr, int32_t* last_page_len,
    void* append_ckv, void* append_kpe,
    int32_t* batch_indices, int32_t* positions,
    uint32_t nnz, uint32_t page_size,
    uint32_t head_dim_ckv, uint32_t head_dim_kpe,
    size_t append_ckv_stride_n, size_t append_kpe_stride_n,
    uint32_t cp_world_size = 1, uint32_t cp_rank = 0);

// Sparse MLA: Quantize BF16 ckv+kpe → packed FP8 paged cache
// kv_cache: [num_blocks, page_size, BPT] uint8 (BPT = kv_lora_rank + num_tiles*4 + pe_dim*2)
// append_ckv: [nnz, kv_lora_rank] BF16
// append_kpe: [nnz, pe_dim] BF16
void glm_concat_and_cache_ds_mla(
    GlmCtx* ctx,
    void* kv_cache,
    void* append_ckv, void* append_kpe,
    int32_t* indices, int32_t* indptr,
    int32_t* batch_indices, int32_t* positions,
    uint32_t nnz, uint32_t page_size,
    uint32_t kv_lora_rank, uint32_t pe_dim,
    size_t append_ckv_stride_n, size_t append_kpe_stride_n,
    uint32_t cp_world_size = 0, uint32_t cp_rank = 0);

// Sparse MLA SM120: prefill attention over topk-selected KV slots
// q: [num_tokens, num_heads, d_qk] BF16 (d_qk = 576 = 512 nope + 64 rope)
// kv_cache: [num_pages, page_size, BPT] U8 (packed FP8)
// indices: [num_tokens, topk] int32 — flat slot IDs (page_id * page_size + offset), -1 = skip
// output: [num_tokens, num_heads, d_v] BF16 (d_v = 512)
// out_lse: [num_tokens, num_heads] FP32
// stride_kv_block: page_block_size * bytes_per_token
// topk_length: [num_tokens] int32 or null (effective top-k per token)
void glm_quantize_fp8(GlmCtx* ctx, const void* input, uint8_t* values,
                      float* scales, size_t num_blocks, uint32_t block_size);

// q_scales != null selects split E4M3 Q with four FP32 scales per head.
void glm_sparse_mla_prefill(
    GlmCtx* ctx,
    void* q, void* q_rope, void* kv_cache,
    int32_t* indices,
    void* output, float* out_lse,
    uint32_t num_tokens, uint32_t num_heads, uint32_t topk,
    uint32_t page_block_size,
    float sm_scale, size_t stride_kv_block,
    int32_t* topk_length, const float* q_scales = nullptr);

// Sparse MLA SM120: decode attention (split-K + merge)
// mid_out: [num_tokens, num_heads, num_splits, d_v] BF16 (scratch)
// mid_lse: [num_tokens, num_heads, num_splits] FP32 (scratch)
// num_splits = ceil(topk / 64)
// chunks_per_block_override: 0 = auto-tune
void glm_sparse_mla_decode(
    GlmCtx* ctx,
    void* q, void* q_rope, void* kv_cache,
    int32_t* indices,
    void* mid_out, float* mid_lse,
    void* output, float* out_lse,
    uint32_t num_tokens, uint32_t num_heads, uint32_t topk,
    uint32_t num_splits,
    float sm_scale, size_t stride_kv_block,
    int32_t* topk_length,
    int chunks_per_block_override, const float* q_scales = nullptr);
// Sparse topk-driven gather of BPT-byte CKV tokens into a flat-format output
// buffer. Same kernel body for two call shapes:
//   * N=1, cp_world_size=0 — single-GPU / non-CP. The warp reads BPT bytes
//     from this rank's local paged KV cache ONCE and writes them once to the
//     single output flat buffer at flat_slot = topk_idx[entry]. Degenerate
//     case: no fan-out, no P2P, no peer semantics. multi-GPU / P2P is NOT
//     required.
//   * N>=1, cp_world_size>0 — context-parallel. Each warp reads BPT bytes
//     ONCE from this rank's local paged KV cache and fan-out writes them to
//     all N peer flat buffers at flat_slot = topk_idx[entry] (cross-GPU via
//     P2P). Tokens not on this rank (pos % cp_world_size != cp_rank, where
//     pos = flat_slot - kv_token_indptr[seq] is reverse-derived) are skipped.
//     Writes are idempotent — duplicate topk entries across queries produce
//     the same bytes at the same slot. Non-topk flat slots are NOT touched
//     (consumer reads only topk slots governed by topk_length).
//
// INPUT CONTRACT (topk_idx): [num_tokens, topk] int32 holding OUTPUT flat
// slots — exactly the format topk_to_slots produces in its flat mode
// (cp_world_size == 1: slot = kv_token_indptr[seq] + token_pos). Invalid
// entries are encoded as -1 and silently skipped. Same indexing the
// downstream sparse MLA kernel uses to read the gathered buffer.
//
// flat_p0..p7: peer flat output buffers, size total_flat_slots * bpt_bytes
//              each (only the first N are written; entries >= N ignored).
// local_kv_cache: this rank's paged KV cache, accessed linearly as
//                 kv_cache[src_slot * bpt_bytes] where
//                 src_slot = abs_page * eff_page_size + offset_in_page.
// topk_idx:        [num_tokens, topk] int32 — OUTPUT flat slots (-1 skip),
//                  as produced by topk_to_slots in flat (cp_world_size == 1)
//                  mode.
// batch_indices:   [num_tokens] int32 — sequence index per query.
// page_indices:    per-rank page-id table (same layout as concat_and_cache_ds_mla).
// page_indptr:     [B+1] int32 — page range per seq.
// kv_token_indptr: [B+1] int32 — global de-interleaved token prefix sum per
//                  seq; used to recover token_pos = flat_slot - kv_token_indptr[seq]
//                  for the CP-rank filter and the paged src_slot lookup.
// N:               number of active peers (1..8). N=1 selects the degenerate
//                  single-buffer path (no fan-out).
// cp_world_size:   0 (no CP at all — filter skipped, local_pos == token_pos) or
//                  > 0 (CP filter applied: pos % ws == cp_rank kept). cp_world_size
//                  == 1 is degenerate CP — filter is a no-op (pos % 1 == 0).
// cp_rank:         this rank's CP id.
// eff_page_size:   page_size / cp_world_size (or == page_size non-CP).
// bpt_bytes:       bytes per token (prod: 656). Multiples of 16 use vectorized
//                  int4 copies; non-multiples use scalar byte tail.
void glm_gather_topk_ckv(
    GlmCtx* ctx,
    void* flat_p0, void* flat_p1, void* flat_p2, void* flat_p3,
    void* flat_p4, void* flat_p5, void* flat_p6, void* flat_p7,
    void* local_kv_cache,
    int32_t* topk_idx, int32_t* batch_indices,
    int32_t* page_indices, int32_t* page_indptr, int32_t* kv_token_indptr,
    int N, int cp_world_size, int cp_rank,
    int eff_page_size, int bpt_bytes,
    int num_tokens, int topk, int padded_kv_len,
    void* scratch_bitmap, void* scratch_unique, void* scratch_counter);

// CUDA Graph operations
void glm_graph_begin_capture(GlmCtx* ctx);
void* glm_graph_end_capture(GlmCtx* ctx);
void* glm_graph_instantiate(GlmCtx* ctx, void* graph);
void glm_graph_launch(GlmCtx* ctx, void* graph_exec);
void glm_graph_destroy(GlmCtx* ctx, void* graph);
void glm_graph_exec_destroy(GlmCtx* ctx, void* graph_exec);

// Fused FP8 dequantize + GEMV for decode (any M)
// Computes: output[m, j] = sum_k(bf16_input[m, k] * fp8_weight[j, k] * bf16_scale_inv[j/128, k/128])
// No activation quantization — BF16 input used directly.
// bf16_out: row-major [M, N] BF16
// bf16_input: row-major [M, K] BF16
// fp8_weight: row-major [N, K] FP8 E4M3
// weight_scale: row-major [N/128, K/128] BF16 (block-wise scale_inv, dequantized in kernel)
void glm_fp8_linear_decode(GlmCtx* ctx, void* bf16_out, const void* bf16_input,
                            const void* fp8_weight, const void* weight_scale,
                            int m, int n, int k);

// Fused NVFP4 dequantize + GEMV/GEMM for decode (any M)
// W4A16: FP4 E2M1 weights (uint8 packed), BF16 input, double quantization scales
// Computes: output[m, j] = sum_k(bf16_input[m, k] * fp4_lut[weight[j,k/2]] * float(weight_scale[j, k/16]) * weight_scale_2)
// bf16_out: row-major [M, N] BF16
// bf16_input: row-major [M, K] BF16
// fp4_weight: row-major [N, K/2] uint8 (two FP4 E2M1 values per byte)
// weight_scale: row-major [N, K/16] FP8 E4M3 (per-block scale, dequantized in kernel)
// weight_scale_2: scalar F32 (global scale = amax / (6*448))
void glm_nvfp4_linear_decode(GlmCtx* ctx, void* bf16_out, const void* bf16_input,
                              const void* fp4_weight, const void* weight_scale,
                              const float* weight_scale_2, int m, int n, int k,
                              void* bf16_workspace = nullptr);

// Gated DeltaNet recurrent step (decode, T=1, batched)
// Fused: L2 norm q,k + gate computation + delta rule update
// output: [batch_size, num_heads, d_v] BF16
// state: [batch_size, state_stride] FP32 (updated in-place)
// qkv: fused QKV tensor — layout described by qkv_ch_stride and qkv_seq_stride
//   For [convDim, batch] layout: qkv_ch_stride=batch, qkv_seq_stride=1
//   For [batch, convDim] layout: qkv_ch_stride=1, qkv_seq_stride=convDim
//   convDim = num_heads * (2*d_k + d_v), Q channels first, then K, then V
// a_raw: [batch_size, num_heads] BF16
// b_raw: [batch_size, num_heads] BF16
// A_log: [num_heads] FP32 (shared across batch)
// dt_bias: [num_heads] FP32 (shared across batch)
// state_stride: stride (in float elements) between batch elements in state
void glm_gdn_recurrent_step(GlmCtx* ctx, void* output, void* state,
                              const void* qkv,
                              const void* a_raw, const void* b_raw,
                              const float* A_log, const float* dt_bias,
                              int num_heads, int d_k, int d_v,
                              int batch_size, int state_stride, int qkv_ch_stride, int qkv_seq_stride);

// Gated DeltaNet prefill (sequential over tokens, batched with cu_seqlens)
// output: [total_seq_len, num_heads, d_v] BF16 (packed)
// state: [batch_size, state_stride] FP32 (updated in-place, should be zero-initialized)
// qkv: fused QKV tensor — layout described by qkv_ch_stride and qkv_seq_stride
//   For [convDim, total_seq_len] layout: qkv_ch_stride=total_seq_len, qkv_seq_stride=1
//   For [total_seq_len, convDim] layout: qkv_ch_stride=1, qkv_seq_stride=convDim
//   convDim = num_heads * (2*d_k + d_v), Q channels first, then K, then V
// a_raw: [total_seq_len, num_heads] BF16 (packed)
// b_raw: [total_seq_len, num_heads] BF16 (packed)
// cu_seqlens: [batch_size + 1] int32 (cumulative sequence lengths)
// A_log: [num_heads] FP32 (shared across batch)
// dt_bias: [num_heads] FP32 (shared across batch)
// state_stride: stride (in float elements) between batch elements in state
void glm_gdn_prefill(GlmCtx* ctx, void* output, void* state,
                      const void* qkv,
                      const void* a_raw, const void* b_raw,
                      const float* A_log, const float* dt_bias,
                      const int* cu_seqlens,
                      int total_seq_len, int num_heads, int d_k, int d_v,
                      int batch_size, int state_stride, int qkv_ch_stride, int qkv_seq_stride);

// Causal conv1d with SiLU activation (batched prefill with cu_seqlens)
// input/output layout described by ch_stride and seq_stride:
//   For [conv_dim, total_seq_len] layout: ch_stride=total_seq_len, seq_stride=1
//   For [total_seq_len, conv_dim] layout: ch_stride=1, seq_stride=conv_dim
// weight layout: [conv_dim, kernel_size] (BF16)
// conv_state: [batch_size, conv_state_stride] BF16 (per-batch conv state, updated in-place)
// cu_seqlens: [batch_size + 1] int32 (cumulative sequence lengths)
// conv_state_stride: stride (in bf16 elements) between batch elements in conv_state
void glm_causal_conv1d(GlmCtx* ctx, void* output, void* conv_state,
                        const void* input, const void* weight,
                        const int* cu_seqlens,
                        int conv_dim, int total_seq_len, int kernel_size,
                        int batch_size, int conv_state_stride,
                        int ch_stride, int seq_stride);

// Causal conv1d update with SiLU activation (batched decode)
// input/output layout: [batch_size, conv_dim] BF16
// conv_state: [batch_size, conv_state_stride] BF16 (per-batch conv state, updated in-place)
// weight layout: [conv_dim, kernel_size] (BF16)
// conv_state_stride: stride (in bf16 elements) between batch elements in conv_state
void glm_causal_conv1d_update(GlmCtx* ctx, void* output, void* conv_state,
                                const void* input, const void* weight,
                                int conv_dim, int kernel_size,
                                int batch_size, int conv_state_stride);


// CP Merge Tree: smem-staged online softmax merge using cp.async.bulk.
// Merges up to 8 partial (v_out, lse) pairs using the online softmax trick.
// Supports in-place operation (output_v may alias one of the v inputs, output_lse may alias one of the lse inputs).
// Individual pointer arguments (not arrays) for CUDA graph capture compatibility.
// v_out inputs: [batch_size * input_n_heads * v_head_dim] BF16 each
// lse inputs: [batch_size * num_heads] F32 each
// output_v: [batch_size * shard_n_heads * v_head_dim] BF16
// output_lse: [batch_size * shard_n_heads] F32 (optional - pass nullptr to skip)
// numel: batch_size * num_heads * v_head_dim (total BF16 elements in v_out)
// shard_n_heads: number of heads to process/output (default num_heads = all)
// head_offset: first head to process (default 0)
// input_n_heads: heads per input shard stride (default num_heads = all heads per shard)
void glm_cp_merge_tree(
    GlmCtx* ctx,
    const void* v0,  const void* v1,  const void* v2,  const void* v3,
    const void* v4,  const void* v5,  const void* v6,  const void* v7,
    const float* lse0,  const float* lse1,  const float* lse2,  const float* lse3,
    const float* lse4,  const float* lse5,  const float* lse6,  const float* lse7,
    int num_shards,
    void* output_v,
    float* output_lse,
    int64_t numel,
    int batch_size,
    int num_heads,
    int v_head_dim,
    int shard_n_heads,
    int head_offset,
    int input_n_heads);

// Push-based CP merge, phase 1: scatter each peer's head slice into that peer's
// staging buffer at slot `rank`. Destination pointers must be pre-rotated by
// rank host-side (dv[k]/dl[k] belong to peer (rank + k) % world_size).
// local_v:   [batch_size, input_n_heads, v_head_dim] BF16
// local_lse: [batch_size, num_heads] F32
// staging:   [world_size, batch_size, shard_n_heads, v_head_dim] BF16
//            [world_size, batch_size, shard_n_heads] F32
void glm_cp_merge_scatter(
    GlmCtx* ctx,
    const void* local_v,
    const float* local_lse,
    void* dv0, void* dv1, void* dv2, void* dv3,
    void* dv4, void* dv5, void* dv6, void* dv7,
    float* dl0, float* dl1, float* dl2, float* dl3,
    float* dl4, float* dl5, float* dl6, float* dl7,
    int world_size,
    int batch_size,
    int shard_n_heads,
    int v_head_dim,
    int input_n_heads,
    int num_heads,
    int rank);

// Push-based CP merge, phase 2: online-softmax merge of the local staging
// buffer's world_size slots. Must run after a barrier that proves every peer's
// phase-1 writes have landed.
// output_v:   [batch_size, shard_n_heads, v_head_dim] BF16
// output_lse: [batch_size, shard_n_heads] F32 (nullable)
void glm_cp_merge_local(
    GlmCtx* ctx,
    const void* stage_v,
    const float* stage_lse,
    void* output_v,
    float* output_lse,
    int world_size,
    int batch_size,
    int shard_n_heads,
    int v_head_dim);

// CP correction: rescale local v_out by exp2(lse_local - global_lse) in-place.
// v_out: [batch_size, num_heads, v_head_dim] BF16 (in-place)
// lses: [world_size, batch_size, num_heads] F32 — all-gathered LSEs (base-2)
// global_lse: [batch_size, num_heads] F32 — output merged LSE (optional, nullptr to skip)
// world_size: number of DCP ranks
// rank: this rank's index in the DCP group
void glm_cp_correct_attn_out(
    GlmCtx* ctx,
    void* v_out,
    const float* lses,
    float* global_lse,
    int batch_size,
    int num_heads,
    int v_head_dim,
    int world_size,
    int rank);

// RMSNorm gated: output = RMSNorm(input) * weight * SiLU(gate)
// output: [batch, dim] BF16
// input: [batch, dim] BF16
// gate: [batch, dim] BF16
// weight: [dim] BF16
void glm_rmsnorm_gated(GlmCtx* ctx, void* output, const void* input,
                        const void* gate, const void* weight,
                        float eps, int dim, int batch);

// Gate sigmoid multiply: attn_out[i] *= sigmoid(gate_interleaved[r * pitch + c])
// attn_out: [batch_seq * num_heads * head_dim] BF16 (in-place, contiguous)
// gate_interleaved: [batch_seq, num_heads, head_dim * 2] BF16 (pitched, reads gate portion at offset head_dim per head)
// pitch = head_dim * 2 (row stride in elements for the pitched 2D view)
void glm_gate_sigmoid_mul(GlmCtx* ctx, void* attn_out, const void* gate_interleaved,
                           int batch_seq, int num_heads, int head_dim);

// GPU batch sampling: each block handles one sequence
// out_tokens: [batch_size] int32 - sampled token IDs
// topk_vals: [batch_size * SAMPLING_MAX_TOPK * SAMPLING_BLOCK_SIZE] float32 - workspace
// topk_idxs: [batch_size * SAMPLING_MAX_TOPK * SAMPLING_BLOCK_SIZE] int32 - workspace
// workspace: [batch_size * vocab_size] float32 - F32 logits workspace
// logits: [batch_size * vocab_size] bfloat16 - input logits
// penalty_tokens: [batch_size * max_window] int32 - circular buffer of penalty token IDs (writable)
// penalty_count: [batch_size] int32 - number of entries per sequence (writable, incremented by kernel)
// max_window: maximum penalty window size (circular buffer stride per sequence)
// temperatures, repetition_penalties, presence_penalties, top_ks, top_ps: [batch_size]
// step_counter: [1] uint32 - stream-ordered RNG counter; no concurrent stream use
// Optional out_probs/out_ids: [batch_size, support_capacity], padded with 0/-1.
// Caller must ensure effective support <= support_capacity <= 256 when exporting.
void glm_sample_batch(GlmCtx* ctx, int* out_tokens, float* topk_vals, int* topk_idxs,
                      float* workspace, const void* logits,
                      int* penalty_tokens, int* penalty_count,
                      int max_window, int vocab_size, int batch_size,
                      const float* temperatures, const float* repetition_penalties,
                      const float* presence_penalties, const int* top_ks,
                      const float* top_ps, unsigned int* step_counter,
                      int max_effective_k, float* out_probs = nullptr,
                      int* out_ids = nullptr, int support_capacity = 0);

// Penalty-free sampling from unsorted BF16 values and unique global I32 IDs [B,K].
// Requires B >= 0 and 1 <= K <= support_capacity <= 256; invalid raw C calls are no-ops.
// out_tokens: [B]; out_probs/out_ids: [B,support_capacity], padded with 0/-1.
// Parameters are [B]; step_counter is [1], advances by B, and is stream-exclusive.
void glm_sample_candidates(GlmCtx* ctx, int* out_tokens, float* out_probs, int* out_ids,
                           const void* candidate_values, const int* candidate_ids,
                           const float* temperatures, const int* top_ks, const float* top_ps,
                           unsigned int* step_counter, int batch_size,
                           int candidate_count, int support_capacity);

// All pointers are device pointers. Sparse supports have unique IDs, trailing -1
// padding, and finite nonnegative probabilities normalized to one.
// q: [B,D,C], p: [B,D+1,C], draft_tokens: [B,D]. 1 <= C <= 256.
// out_tokens: [B,D+1]; out_accepted: [B], accepted prefix length (not bonus).
// RNG advances by B*(2*D+2); the counter must not be shared across streams.
void glm_spec_reject_linear(GlmCtx* ctx, int* out_tokens, int* out_accepted,
                           const int* draft_tokens, const float* q_probs, const int* q_ids,
                           const float* p_probs, const int* p_ids, unsigned int* step_counter,
                           int batch_size, int depth, int capacity);

// Grouped MoE using Tensor Core MMA (SM120+)
// Uses Tensor Core MMA for prefill (M > threshold) instead of scalar GEMV.
size_t glm_mma_moe_workspace_size(int count, int N, int K, int num_experts);

// Grouped BF16 MoE using Tensor Core MMA (SM120+)
// Same interface as glm_mul_mat_id but uses Tensor Core MMA
// for prefill (M > threshold) instead of scalar GEMV.
void glm_bf16_mul_mat_id_grouped_mma(GlmCtx* ctx, void* output, const void* input,
                                       const void* const* weight_ptrs,
                                       const int* expert_ids, int top_k,
                                       int count, int N, int K,
                                       int num_experts, void* workspace);

// Cooperative all-warps kernel (B12X-style): all warps stage cp.async loads
// then all warps run MMA. No producer/consumer split, no mbarriers.
// Uses TM=64, TN=128, DEPTH=2, NWARPS=2 by default (GLM_COOP_CONFIG env override).
size_t glm_mma_moe_coop_workspace_size(int count, int N, int K, int num_experts);

void glm_nvfp4_mul_mat_id_grouped_mma_coop(GlmCtx* ctx, void* output,
                                            const void* input,
                                            const void* const* weight_ptrs,
                                            const void* const* scale_ptrs,
                                            const void* const* scale2_ptrs,
                                            const int* expert_ids, int top_k,
                                            int count, int N, int K,
                                            int num_experts, void* workspace);

// Split MoE coop: scatter once, then optionally keep GEMM outputs expert-sorted.
size_t glm_mma_moe_coop_scatter_workspace_size(int count, int K, int num_experts);
size_t glm_mma_moe_coop_gemm_workspace_size(int count, int N);
void glm_mma_moe_coop_scatter(GlmCtx* ctx, const void* input, const int* expert_ids,
                              int top_k, int count, int K, int num_experts,
                              void* scatter_workspace);
void glm_mma_moe_coop_gemm(GlmCtx* ctx,
                           const void* const* weight_ptrs, const void* const* scale_ptrs,
                           const void* const* scale2_ptrs,
                           int num_experts, int N, int K, int count,
                           int scatter_k, const void* scatter_workspace,
                           const void* sorted_input_override, bool output_sorted,
                           void* gemm_workspace, void* output);

#ifdef __cplusplus
}
#endif

// ---------------------------------------------------------------------------
// Device-only BF16 vector I/O helpers (available to all .cu TUs that include
// this header). Kept here so kernels defined in translation units other than
// glm_ops.cu (e.g. glm_p2p.cu) can share the same helpers.
// ---------------------------------------------------------------------------
#if defined(__CUDACC__)
#include <cuda_bf16.h>

__device__ __forceinline__ float2 load_bf16x2(const __nv_bfloat16* ptr) {
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(ptr);
    return __bfloat1622float2(v);
}

__device__ __forceinline__ void store_bf16x2(__nv_bfloat16* ptr, float v0, float v1) {
    float2 f = {v0, v1};
    *reinterpret_cast<__nv_bfloat162*>(ptr) = __float22bfloat162_rn(f);
}
#endif

#endif
