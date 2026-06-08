// ---------------------------------------------------------------------------
// Context Parallelism: Online Softmax Merge Kernel
//
// Merges partial attention outputs from multiple KV sequence shards using
// FlashInfer's state_t::merge (online softmax correction).
//
// Each shard produces:
//   - partial_v_out: [B, input_n_heads, head_dim] BF16 (after v_expand)
//   - partial_lse:   [B, num_heads] F32 (base-2 log-sum-exp, all heads)
//
// The merge computes:
//   For each (batch, head):
//     state_t accumulates across shards: merge(v_i, lse_i, d=1.0)
//     Then normalize: v /= d
//
// Uses FlashInfer's state_t<vec_size>::merge() for the FP32 merge math.
// BF16 input -> FP32 cast_load, merge in FP32, cast_store -> BF16 output.
//
// Head-grouped merge: when shard_n_heads < num_heads, only processes and
// outputs heads [head_offset, head_offset + shard_n_heads). Input v_out uses
// input_n_heads stride (specifies how many heads per input shard; equals
// shard_n_heads for column-parallel v_proj, or num_heads for replicated v_proj
// producing Row-parallel output). Input lse uses num_heads stride (full head
// layout). Output uses shard_n_heads stride, producing contiguous
// [B * shard_n_heads * head_dim] (Row-parallel layout).
// When shard_n_heads == num_heads, head_offset == 0, and input_n_heads == num_heads,
// equivalent to full merge.
// ---------------------------------------------------------------------------

#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

#include "flashinfer/attention/state.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "flashinfer/math.cuh"

namespace {

constexpr int CP_MAX_SHARDS = 16;

template <int VEC_SIZE, int BDX, int NUM_SHARDS, typename LseLoader, typename VLoader>
__device__ __forceinline__ void cp_merge_one_pair(
    int tid,
    int b, int h, int num_heads,
    int local_h, int shard_n_heads,
    float* s_lse,
    LseLoader load_lse,
    VLoader load_v,
    __nv_bfloat16* merged_v_out,
    float* merged_lse,
    int rank_offset = 0)
{
    constexpr int head_dim = VEC_SIZE * BDX;

    load_lse(s_lse, b, h, num_heads);
    __syncthreads();

    if (tid < BDX) {
        flashinfer::state_t<VEC_SIZE> st;
        st.init();

        // Stagger shard read order by rank so all GPUs don't hammer the same
        // peer's P2P buffer simultaneously (see p2p_allreduce_oneshot_kernel).
        #pragma unroll
        for (int ss = 0; ss < NUM_SHARDS; ++ss) {
            int s = ss + rank_offset;
            if (s >= NUM_SHARDS) s -= NUM_SHARDS;
            flashinfer::vec_t<float, VEC_SIZE> v;
            load_v(v, s, b, h, head_dim, tid);
            st.merge(v, s_lse[s], 1.0f);
        }

        st.normalize();
        st.o.cast_store(merged_v_out + (b * shard_n_heads + local_h) * head_dim + tid * VEC_SIZE);

        if (merged_lse != nullptr && tid == 0) {
            merged_lse[b * shard_n_heads + local_h] = st.get_lse();
        }
    }
}

struct CPMergeParams {
    const void*  v_ptrs[CP_MAX_SHARDS];
    const float* lse_ptrs[CP_MAX_SHARDS];
};

template <int VEC_SIZE, int BDX, int NUM_SHARDS>
__global__ void __launch_bounds__(BDX, 1)
cp_merge_kernel(
    CPMergeParams params,
    __nv_bfloat16* __restrict__ merged_v_out,
    float* __restrict__ merged_lse,
    int batch_size,
    int num_heads,
    int shard_n_heads,
    int head_offset,
    int input_n_heads)
{
    int tid = threadIdx.x;
    int bh = blockIdx.x;
    int b = bh / shard_n_heads;
    int local_h = bh % shard_n_heads;
    int h = local_h + head_offset;

    if (b >= batch_size) return;

    __shared__ float s_lse[CP_MAX_SHARDS];

    auto load_lse = [&](float* slse, int b_, int h_, int nh) {
        if (tid < NUM_SHARDS) {
            slse[tid] = params.lse_ptrs[tid][b_ * nh + h_];
        }
    };

    auto load_v = [&](flashinfer::vec_t<float, VEC_SIZE>& v, int s, int b_, int h_, int hd, int tid_) {
        const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(params.v_ptrs[s]);
        v.cast_load(v_ptr + (b_ * input_n_heads + h_) * hd + tid_ * VEC_SIZE);
    };

    cp_merge_one_pair<VEC_SIZE, BDX, NUM_SHARDS>(
        tid, b, h, num_heads, local_h, shard_n_heads, s_lse,
        load_lse, load_v,
        merged_v_out, merged_lse);
}

template <int VEC_SIZE, int BDX>
void launch_cp_merge(
    CPMergeParams& params,
    __nv_bfloat16* merged_v_out,
    float* merged_lse,
    int num_shards,
    int batch_size,
    int num_heads,
    int shard_n_heads,
    int head_offset,
    int input_n_heads,
    cudaStream_t stream)
{
    int grid = batch_size * shard_n_heads;
    switch (num_shards) {
        case 1:
            cp_merge_kernel<VEC_SIZE, BDX, 1><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        case 2:
            cp_merge_kernel<VEC_SIZE, BDX, 2><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        case 3:
            cp_merge_kernel<VEC_SIZE, BDX, 3><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        case 4:
            cp_merge_kernel<VEC_SIZE, BDX, 4><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        case 8:
            cp_merge_kernel<VEC_SIZE, BDX, 8><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        case 16:
            cp_merge_kernel<VEC_SIZE, BDX, 16><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads);
            break;
        default:
            fprintf(stderr, "launch_cp_merge: unsupported num_shards=%d (must be 1-4, 8, or 16)\n", num_shards);
            break;
    }
}

} // namespace

extern "C" {

void glm_context_parallel_merge(
    GlmCtx* ctx,
    const void* const* partial_v_outs,
    const float* const* partial_lses,
    int num_shards,
    void* merged_v_out,
    float* merged_lse,
    int batch_size,
    int num_heads,
    int v_head_dim)
{
    glm_context_parallel_merge_heads(ctx, partial_v_outs, partial_lses,
        num_shards, merged_v_out, merged_lse,
        batch_size, num_heads, num_heads, 0, num_heads, v_head_dim);
}

void glm_context_parallel_merge_heads(
    GlmCtx* ctx,
    const void* const* partial_v_outs,
    const float* const* partial_lses,
    int num_shards,
    void* merged_v_out,
    float* merged_lse,
    int batch_size,
    int num_heads,
    int shard_n_heads,
    int head_offset,
    int input_n_heads,
    int v_head_dim)
{
    cudaSetDevice(ctx->device_id);

    if (num_shards < 1 || num_shards > CP_MAX_SHARDS) {
        fprintf(stderr, "glm_context_parallel_merge_heads: num_shards=%d out of range [1, %d]\n",
                num_shards, CP_MAX_SHARDS);
        return;
    }
    if (head_offset < 0 || shard_n_heads < 1 || head_offset + shard_n_heads > num_heads) {
        fprintf(stderr, "glm_context_parallel_merge_heads: head_offset=%d + shard_n_heads=%d > num_heads=%d\n",
                head_offset, shard_n_heads, num_heads);
        return;
    }
    if (input_n_heads < shard_n_heads || input_n_heads > num_heads) {
        fprintf(stderr, "glm_context_parallel_merge_heads: input_n_heads=%d out of range [shard_n_heads=%d, num_heads=%d]\n",
                input_n_heads, shard_n_heads, num_heads);
        return;
    }

    CPMergeParams params;
    memset(&params, 0, sizeof(params));
    for (int i = 0; i < num_shards; ++i) {
        params.v_ptrs[i] = partial_v_outs[i];
        params.lse_ptrs[i] = partial_lses[i];
    }

    // Dispatch based on v_head_dim: VEC_SIZE=4, BDX = v_head_dim / VEC_SIZE
    // Supported: 32 (8 threads), 64 (16 threads), 128 (32 threads),
    //            256 (64 threads), 512 (128 threads)
    switch (v_head_dim) {
        case 32:
            launch_cp_merge<4, 8>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, GLM_STREAM(ctx));
            break;
        case 64:
            launch_cp_merge<4, 16>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, GLM_STREAM(ctx));
            break;
        case 128:
            launch_cp_merge<4, 32>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, GLM_STREAM(ctx));
            break;
        case 256:
            launch_cp_merge<4, 64>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, GLM_STREAM(ctx));
            break;
        case 512:
            launch_cp_merge<4, 128>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, GLM_STREAM(ctx));
            break;
        default:
            fprintf(stderr, "glm_context_parallel_merge_heads: unsupported v_head_dim=%d "
                    "(must be 32, 64, 128, 256, or 512)\n", v_head_dim);
            break;
    }
}

} // extern "C"

// ---------------------------------------------------------------------------
// P2P Context Parallel Merge: two-phase P2P sync + multi-block merge.
//
// Phase 1 (p2p_cp_sync_kernel): single-block kernel that:
//   1. Scatters local partial_v_out + partial_lse to P2P data buffer
//   2. Publishes data-ready flag and waits for all peers
//   3. Writes slot_offset for Phase 2
//
// Phase 2 (p2p_cp_merge_multi_kernel): multi-block kernel that:
//   1. Reads all peers' data from peer-mapped P2P buffers
//   2. Merges using state_t::merge across all shards (one block per head)
//   3. Writes merged output
//
// P2P buffer layout per slot:
//   [v_out: B*H*vHeadDim*2 bytes] [lse: B*H*4 bytes]
//
// Reuses GlmP2PInstance for P2P infrastructure (double-buffered data,
// flag-based sync, seq counter).
// ---------------------------------------------------------------------------

constexpr int P2P_CP_BLOCK_SIZE = 1024;

__device__ __forceinline__ void p2p_spin_until(volatile int* flag, int target) {
    while (*flag < target) { /* spin */ }
}

__global__ void __launch_bounds__(P2P_CP_BLOCK_SIZE, 1)
p2p_cp_sync_kernel(
    void* const* peer_data,
    int* const* peer_flags,
    unsigned long long* my_seq_counter,
    int* slot_offset_out,
    int my_rank,
    int world_size,
    int max_slot_bytes,
    const __nv_bfloat16* __restrict__ my_v_out,
    const float* __restrict__ my_lse,
    int batch_size,
    int num_heads,
    int v_out_bytes)
{
    int tid = threadIdx.x;
    int bs = blockDim.x;

    __shared__ unsigned int s_seq;
    __shared__ int          s_slot_offset;
    __shared__ void*        s_peer_data[P2P_AR_MAX_WORLD];
    __shared__ int*         s_peer_flags[P2P_AR_MAX_WORLD];

    if (tid == 0) {
        unsigned long long s = atomicAdd(my_seq_counter, 2ULL) + 2ULL;
        s_seq = (unsigned int)(s & 0x7FFFFFFFu);
        if (s_seq == 0) s_seq = 2;
        s_slot_offset = ((s_seq >> 1) & 1) * max_slot_bytes;
    }
    if (tid < world_size) {
        s_peer_data[tid]  = peer_data[tid];
        s_peer_flags[tid] = peer_flags[tid];
    }
    __syncthreads();

    int seq = (int)s_seq;
    int slot_offset = s_slot_offset;

    // Scatter my_v_out to P2P buffer
    // s_peer_data[my_rank] via smem: dynamic index, would spill registers
    char* my_data = static_cast<char*>(s_peer_data[my_rank]) + slot_offset;
    {
        const uint4* src_v4 = reinterpret_cast<const uint4*>(my_v_out);
        uint4*       dst_v4 = reinterpret_cast<uint4*>(my_data);
        int count_v4 = v_out_bytes / 16;
        for (int i = tid; i < count_v4; i += bs) {
            dst_v4[i] = src_v4[i];
        }
        int tail = count_v4 * 16;
        const __nv_bfloat16* src_tail = my_v_out + (tail / 2);
        __nv_bfloat16* dst_tail = reinterpret_cast<__nv_bfloat16*>(my_data) + (tail / 2);
        for (int i = tid; i < (v_out_bytes - tail) / 2; i += bs) {
            dst_tail[i] = src_tail[i];
        }
    }

    // Scatter my_lse to P2P buffer (after v_out)
    {
        float* dst_lse = reinterpret_cast<float*>(my_data + v_out_bytes);
        int lse_count = batch_size * num_heads;
        for (int i = tid; i < lse_count; i += bs) {
            dst_lse[i] = my_lse[i];
        }
    }

    // Publish data-ready flag
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        // s_peer_flags via smem: my_rank is dynamic across kernels, smem avoids spill
        volatile int* mf = s_peer_flags[my_rank];
        *mf = seq + 1;
    }

    // Wait for all peers
    // s_peer_flags[tid] via smem: dynamic index would spill register array
    if (tid < world_size) {
        volatile int* pf = s_peer_flags[tid];
        p2p_spin_until(pf, seq + 1);
    }
    __syncthreads();
    __threadfence_system();

    // Write slot_offset for Phase 2
    if (tid == 0) {
        *slot_offset_out = slot_offset;
    }
}

template <int VEC_SIZE, int BDX, int NUM_SHARDS>
__global__ void __launch_bounds__(BDX, 1)
p2p_cp_merge_multi_kernel(
    void* const* peer_data,
    const int* __restrict__ slot_offset_ptr,
    __nv_bfloat16* __restrict__ merged_v_out,
    float* __restrict__ merged_lse,
    int batch_size,
    int num_heads,
    int shard_n_heads,
    int head_offset,
    int input_n_heads,
    int v_out_bytes,
    int my_rank)
{
    constexpr int head_dim = VEC_SIZE * BDX;
    int tid = threadIdx.x;
    int bh = blockIdx.x;
    int b = bh / shard_n_heads;
    int local_h = bh % shard_n_heads;
    int h = local_h + head_offset;

    if (b >= batch_size) return;

    int slot_offset = *slot_offset_ptr;

    // Rank-rotated peer data pointers in registers. rr is a compile-time
    // constant in the unrolled merge loop, so peer_pv[s] avoids register spill.
    // peer_pv[tid] in load_lse is dynamic but called once (negligible spill cost).
    const char* peer_pv[CP_MAX_SHARDS];
    #pragma unroll
    for (int rr = 0; rr < CP_MAX_SHARDS; ++rr) {
        if (rr >= NUM_SHARDS) break;
        int r = rr + my_rank; if (r >= NUM_SHARDS) r -= NUM_SHARDS;
        peer_pv[rr] = static_cast<const char*>(peer_data[r]) + slot_offset;
    }

    __shared__ float s_lse[CP_MAX_SHARDS];

    auto load_lse = [&](float* slse, int b_, int h_, int nh) {
        // tid is dynamic but only NUM_SHARDS threads execute this once,
        // so register spill cost is negligible.
        if (tid < NUM_SHARDS) {
            const float* peer_lse = reinterpret_cast<const float*>(
                peer_pv[tid] + v_out_bytes);
            slse[tid] = peer_lse[b_ * nh + h_];
        }
    };

    auto load_v = [&](flashinfer::vec_t<float, VEC_SIZE>& v, int s, int b_, int h_, int hd, int tid_) {
        const __nv_bfloat16* peer_v = reinterpret_cast<const __nv_bfloat16*>(
            peer_pv[s]);
        v.cast_load(peer_v + (b_ * input_n_heads + h_) * hd + tid_ * VEC_SIZE);
    };

    cp_merge_one_pair<VEC_SIZE, BDX, NUM_SHARDS>(
        tid, b, h, num_heads, local_h, shard_n_heads, s_lse,
        load_lse, load_v,
        merged_v_out, merged_lse, my_rank);
}

template <int VEC_SIZE, int BDX>
void launch_p2p_cp_merge(
    void** peer_data, int** peer_flags, unsigned long long* seq_counter,
    int* slot_offset_out,
    int my_rank, int world_size, int max_slot_bytes,
    const __nv_bfloat16* my_v_out, const float* my_lse,
    __nv_bfloat16* merged_v_out, float* merged_lse,
    int num_shards, int batch_size, int num_heads, int shard_n_heads, int head_offset,
    int input_n_heads, int v_out_bytes,
    cudaStream_t stream)
{
    // Phase 1: P2P sync (single block)
    p2p_cp_sync_kernel<<<1, P2P_CP_BLOCK_SIZE, 0, stream>>>(
        peer_data, peer_flags, seq_counter, slot_offset_out,
        my_rank, world_size, max_slot_bytes,
        my_v_out, my_lse, batch_size, num_heads, v_out_bytes);

    // Phase 2: Multi-block merge
    int grid = batch_size * shard_n_heads;
    switch (num_shards) {
        case 2:
            p2p_cp_merge_multi_kernel<VEC_SIZE, BDX, 2><<<grid, BDX, 0, stream>>>(
                peer_data, slot_offset_out,
                merged_v_out, merged_lse,
                batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_out_bytes, my_rank);
            break;
        case 4:
            p2p_cp_merge_multi_kernel<VEC_SIZE, BDX, 4><<<grid, BDX, 0, stream>>>(
                peer_data, slot_offset_out,
                merged_v_out, merged_lse,
                batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_out_bytes, my_rank);
            break;
        case 8:
            p2p_cp_merge_multi_kernel<VEC_SIZE, BDX, 8><<<grid, BDX, 0, stream>>>(
                peer_data, slot_offset_out,
                merged_v_out, merged_lse,
                batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_out_bytes, my_rank);
            break;
        case 16:
            p2p_cp_merge_multi_kernel<VEC_SIZE, BDX, 16><<<grid, BDX, 0, stream>>>(
                peer_data, slot_offset_out,
                merged_v_out, merged_lse,
                batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_out_bytes, my_rank);
            break;
        default:
            fprintf(stderr, "launch_p2p_cp_merge: unsupported num_shards=%d (must be 2, 4, 8, or 16)\n", num_shards);
            break;
    }
}

extern "C" {

void glm_p2p_cp_merge(
    GlmCtx* ctx,
    GlmP2PInstance* inst,
    const void* my_v_out,
    const float* my_lse,
    void* merged_v_out,
    float* merged_lse,
    int num_shards,
    int batch_size,
    int num_heads,
    int v_head_dim)
{
    glm_p2p_cp_merge_heads(ctx, inst, my_v_out, my_lse,
        merged_v_out, merged_lse,
        num_shards, batch_size, num_heads, num_heads, 0, num_heads, v_head_dim);
}

void glm_p2p_cp_merge_heads(
    GlmCtx* ctx,
    GlmP2PInstance* inst,
    const void* my_v_out,
    const float* my_lse,
    void* merged_v_out,
    float* merged_lse,
    int num_shards,
    int batch_size,
    int num_heads,
    int shard_n_heads,
    int head_offset,
    int input_n_heads,
    int v_head_dim)
{
    cudaSetDevice(ctx->device_id);

    if (num_shards < 2 || num_shards > P2P_AR_MAX_WORLD) {
        fprintf(stderr, "glm_p2p_cp_merge_heads: num_shards=%d out of range [2, %d]\n",
                num_shards, P2P_AR_MAX_WORLD);
        return;
    }
    if (head_offset < 0 || shard_n_heads < 1 || head_offset + shard_n_heads > num_heads) {
        fprintf(stderr, "glm_p2p_cp_merge_heads: head_offset=%d + shard_n_heads=%d > num_heads=%d\n",
                head_offset, shard_n_heads, num_heads);
        return;
    }
    if (input_n_heads < shard_n_heads || input_n_heads > num_heads) {
        fprintf(stderr, "glm_p2p_cp_merge_heads: input_n_heads=%d out of range [shard_n_heads=%d, num_heads=%d]\n",
                input_n_heads, shard_n_heads, num_heads);
        return;
    }

    // P2P buffer holds per-shard data: v_out is [B, input_n_heads, D],
    // lse is [B, num_heads] (full heads). Only the merge phase processes
    // shard_n_heads starting at head_offset.
    int v_out_bytes = batch_size * input_n_heads * v_head_dim * 2;
    int lse_bytes = batch_size * num_heads * 4;
    int slot_bytes = v_out_bytes + lse_bytes;

    if ((size_t)slot_bytes > inst->max_bytes) {
        fprintf(stderr, "glm_p2p_cp_merge_heads: slot_bytes=%d exceeds max_bytes=%zu\n",
                slot_bytes, inst->max_bytes);
        return;
    }

    switch (v_head_dim) {
        case 32:
            launch_p2p_cp_merge<4, 8>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->slot_offset_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(my_v_out), my_lse,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_out_bytes,
                GLM_STREAM(ctx));
            break;
        case 64:
            launch_p2p_cp_merge<4, 16>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->slot_offset_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(my_v_out), my_lse,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_out_bytes,
                GLM_STREAM(ctx));
            break;
        case 128:
            launch_p2p_cp_merge<4, 32>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->slot_offset_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(my_v_out), my_lse,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_out_bytes,
                GLM_STREAM(ctx));
            break;
        case 256:
            launch_p2p_cp_merge<4, 64>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->slot_offset_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(my_v_out), my_lse,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_out_bytes,
                GLM_STREAM(ctx));
            break;
        case 512:
            launch_p2p_cp_merge<4, 128>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->slot_offset_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(my_v_out), my_lse,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, shard_n_heads, head_offset, input_n_heads, v_out_bytes,
                GLM_STREAM(ctx));
            break;
        default:
            fprintf(stderr, "glm_p2p_cp_merge_heads: unsupported v_head_dim=%d "
                    "(must be 32, 64, 128, 256, or 512)\n", v_head_dim);
            break;
    }
}

} // extern "C"
