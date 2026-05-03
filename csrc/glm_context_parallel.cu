// ---------------------------------------------------------------------------
// Context Parallelism: Online Softmax Merge Kernel
//
// Merges partial attention outputs from multiple KV sequence shards using
// FlashInfer's state_t::merge (online softmax correction).
//
// Each shard produces:
//   - partial_v_out: [B, H, head_dim] BF16 (after v_expand)
//   - partial_lse:   [B, H] F32 (base-2 log-sum-exp)
//
// The merge computes:
//   For each (batch, head):
//     state_t accumulates across shards: merge(v_i, lse_i, d=1.0)
//     Then normalize: v /= d
//
// Uses FlashInfer's state_t<vec_size>::merge() for the FP32 merge math.
// BF16 input -> FP32 cast_load, merge in FP32, cast_store -> BF16 output.
// ---------------------------------------------------------------------------

#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

#include "flashinfer/attention/state.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "flashinfer/math.cuh"
#include "cp_merge_impl.cuh"

namespace {

constexpr int CP_MAX_SHARDS = 16;

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
    int num_heads)
{
    int tid = threadIdx.x;
    int bh = blockIdx.x;
    int b = bh / num_heads;
    int h = bh % num_heads;

    if (b >= batch_size) return;

    __shared__ float s_lse[CP_MAX_SHARDS];

    auto load_lse = [&](float* slse, int b_, int h_, int nh) {
        if (tid < NUM_SHARDS) {
            slse[tid] = params.lse_ptrs[tid][b_ * nh + h_];
        }
    };

    auto load_v = [&](flashinfer::vec_t<float, VEC_SIZE>& v, int s, int b_, int h_, int nh, int hd, int tid_) {
        const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(params.v_ptrs[s]);
        v.cast_load(v_ptr + (b_ * nh + h_) * hd + tid_ * VEC_SIZE);
    };

    cp_merge_one_pair<VEC_SIZE, BDX, NUM_SHARDS>(
        tid, b, h, num_heads, s_lse,
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
    cudaStream_t stream)
{
    int grid = batch_size * num_heads;
    switch (num_shards) {
        case 1:
            cp_merge_kernel<VEC_SIZE, BDX, 1><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
            break;
        case 2:
            cp_merge_kernel<VEC_SIZE, BDX, 2><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
            break;
        case 3:
            cp_merge_kernel<VEC_SIZE, BDX, 3><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
            break;
        case 4:
            cp_merge_kernel<VEC_SIZE, BDX, 4><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
            break;
        case 8:
            cp_merge_kernel<VEC_SIZE, BDX, 8><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
            break;
        case 16:
            cp_merge_kernel<VEC_SIZE, BDX, 16><<<grid, BDX, 0, stream>>>(
                params, merged_v_out, merged_lse, batch_size, num_heads);
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
    cudaSetDevice(ctx->device_id);

    if (num_shards < 1 || num_shards > CP_MAX_SHARDS) {
        fprintf(stderr, "glm_context_parallel_merge: num_shards=%d out of range [1, %d]\n",
                num_shards, CP_MAX_SHARDS);
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
                num_shards, batch_size, num_heads, GLM_STREAM(ctx));
            break;
        case 64:
            launch_cp_merge<4, 16>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, GLM_STREAM(ctx));
            break;
        case 128:
            launch_cp_merge<4, 32>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, GLM_STREAM(ctx));
            break;
        case 256:
            launch_cp_merge<4, 64>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, GLM_STREAM(ctx));
            break;
        case 512:
            launch_cp_merge<4, 128>(params,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, GLM_STREAM(ctx));
            break;
        default:
            fprintf(stderr, "glm_context_parallel_merge: unsupported v_head_dim=%d "
                    "(must be 32, 64, 128, 256, or 512)\n", v_head_dim);
            break;
    }
}

} // extern "C"

// ---------------------------------------------------------------------------
// P2P Context Parallel Merge: fused P2P sync + online softmax merge.
//
// Single-block kernel that:
//   1. Scatters local partial_v_out + partial_lse to P2P data buffer
//   2. Publishes data-ready flag and waits for all peers
//   3. Reads all peers' data from peer-mapped P2P buffers
//   4. Merges using state_t::merge across all shards
//   5. Writes merged output
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

template <int VEC_SIZE, int BDX, int NUM_SHARDS>
__global__ void __launch_bounds__(P2P_CP_BLOCK_SIZE, 1)
p2p_cp_merge_kernel(
    void* const* peer_data,            // [world_size] device ptrs (peer-mapped)
    int* const* peer_flags,            // [world_size] device ptrs (peer-mapped)
    unsigned long long* my_seq_counter,
    int my_rank,
    int world_size,
    int max_slot_bytes,
    const __nv_bfloat16* __restrict__ my_v_out,
    const float* __restrict__ my_lse,
    __nv_bfloat16* __restrict__ merged_v_out,
    float* __restrict__ merged_lse,
    int batch_size,
    int num_heads,
    int v_out_bytes)              // B * H * vHeadDim * sizeof(bf16)
{
    constexpr int head_dim = VEC_SIZE * BDX;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    // ---- P2P sync phase ----
    __shared__ unsigned int s_seq;
    __shared__ int          s_slot_offset;
    __shared__ void*        s_peer_data[P2P_AR_MAX_WORLD];
    __shared__ int*         s_peer_flags[P2P_AR_MAX_WORLD];

    if (tid == 0) {
        unsigned long long s = atomicAdd(my_seq_counter, 2ULL) + 2ULL;
        s_seq = (unsigned int)(s & 0x7FFFFFFEu);
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
        volatile int* mf = s_peer_flags[my_rank];
        *mf = seq + 1;
    }

    // Wait for all peers
    if (tid < world_size) {
        volatile int* pf = s_peer_flags[tid];
        p2p_spin_until(pf, seq + 1);
    }
    __syncthreads();
    __threadfence_system();

    // ---- Merge phase ----
    __shared__ float s_lse[P2P_AR_MAX_WORLD];

    int num_pairs = batch_size * num_heads;

    for (int bh = 0; bh < num_pairs; ++bh) {
        int b = bh / num_heads;
        int h = bh % num_heads;

        auto load_lse = [&](float* slse, int b_, int h_, int nh) {
            if (tid < NUM_SHARDS) {
                const float* peer_lse = reinterpret_cast<const float*>(
                    static_cast<char*>(s_peer_data[tid]) + slot_offset + v_out_bytes);
                slse[tid] = peer_lse[b_ * nh + h_];
            }
        };

        auto load_v = [&](flashinfer::vec_t<float, VEC_SIZE>& v, int s, int b_, int h_, int nh, int hd, int tid_) {
            const __nv_bfloat16* peer_v = reinterpret_cast<const __nv_bfloat16*>(
                static_cast<char*>(s_peer_data[s]) + slot_offset);
            v.cast_load(peer_v + (b_ * nh + h_) * hd + tid_ * VEC_SIZE);
        };

        cp_merge_one_pair<VEC_SIZE, BDX, NUM_SHARDS>(
            tid, b, h, num_heads, s_lse,
            load_lse, load_v,
            merged_v_out, merged_lse);

        __syncthreads();
    }
}

template <int VEC_SIZE, int BDX>
void launch_p2p_cp_merge(
    void** peer_data, int** peer_flags, unsigned long long* seq_counter,
    int my_rank, int world_size, int max_slot_bytes,
    const __nv_bfloat16* my_v_out, const float* my_lse,
    __nv_bfloat16* merged_v_out, float* merged_lse,
    int num_shards, int batch_size, int num_heads, int v_out_bytes,
    cudaStream_t stream)
{
    switch (num_shards) {
        case 2:
            p2p_cp_merge_kernel<VEC_SIZE, BDX, 2><<<1, P2P_CP_BLOCK_SIZE, 0, stream>>>(
                peer_data, peer_flags, seq_counter,
                my_rank, world_size, max_slot_bytes,
                my_v_out, my_lse, merged_v_out, merged_lse,
                batch_size, num_heads, v_out_bytes);
            break;
        case 4:
            p2p_cp_merge_kernel<VEC_SIZE, BDX, 4><<<1, P2P_CP_BLOCK_SIZE, 0, stream>>>(
                peer_data, peer_flags, seq_counter,
                my_rank, world_size, max_slot_bytes,
                my_v_out, my_lse, merged_v_out, merged_lse,
                batch_size, num_heads, v_out_bytes);
            break;
        case 8:
            p2p_cp_merge_kernel<VEC_SIZE, BDX, 8><<<1, P2P_CP_BLOCK_SIZE, 0, stream>>>(
                peer_data, peer_flags, seq_counter,
                my_rank, world_size, max_slot_bytes,
                my_v_out, my_lse, merged_v_out, merged_lse,
                batch_size, num_heads, v_out_bytes);
            break;
        case 16:
            p2p_cp_merge_kernel<VEC_SIZE, BDX, 16><<<1, P2P_CP_BLOCK_SIZE, 0, stream>>>(
                peer_data, peer_flags, seq_counter,
                my_rank, world_size, max_slot_bytes,
                my_v_out, my_lse, merged_v_out, merged_lse,
                batch_size, num_heads, v_out_bytes);
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
    cudaSetDevice(ctx->device_id);

    if (num_shards < 2 || num_shards > P2P_AR_MAX_WORLD) {
        fprintf(stderr, "glm_p2p_cp_merge: num_shards=%d out of range [2, %d]\n",
                num_shards, P2P_AR_MAX_WORLD);
        return;
    }

    int v_out_bytes = batch_size * num_heads * v_head_dim * 2;
    int lse_bytes = batch_size * num_heads * 4;
    int slot_bytes = v_out_bytes + lse_bytes;

    if ((size_t)slot_bytes > inst->max_bytes) {
        fprintf(stderr, "glm_p2p_cp_merge: slot_bytes=%d exceeds max_bytes=%zu\n",
                slot_bytes, inst->max_bytes);
        return;
    }

    switch (v_head_dim) {
        case 32:
            launch_p2p_cp_merge<4, 8>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(my_v_out), my_lse,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, v_out_bytes,
                GLM_STREAM(ctx));
            break;
        case 64:
            launch_p2p_cp_merge<4, 16>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(my_v_out), my_lse,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, v_out_bytes,
                GLM_STREAM(ctx));
            break;
        case 128:
            launch_p2p_cp_merge<4, 32>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(my_v_out), my_lse,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, v_out_bytes,
                GLM_STREAM(ctx));
            break;
        case 256:
            launch_p2p_cp_merge<4, 64>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(my_v_out), my_lse,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, v_out_bytes,
                GLM_STREAM(ctx));
            break;
        case 512:
            launch_p2p_cp_merge<4, 128>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(my_v_out), my_lse,
                static_cast<__nv_bfloat16*>(merged_v_out), merged_lse,
                num_shards, batch_size, num_heads, v_out_bytes,
                GLM_STREAM(ctx));
            break;
        default:
            fprintf(stderr, "glm_p2p_cp_merge: unsupported v_head_dim=%d "
                    "(must be 32, 64, 128, 256, or 512)\n", v_head_dim);
            break;
    }
}

} // extern "C"
