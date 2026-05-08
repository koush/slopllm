// ---------------------------------------------------------------------------
// Custom P2P "one-shot" AllReduce for small messages (single-block design).
//
// Targets PCIe-only multi-GPU systems where NCCL ring AllReduce is ~30-50 us
// per call due to multi-hop launch latency. This implementation completes a
// 10 KB AllReduce in ~5-10 us by:
//   - Mapping every peer's data buffer directly via cudaDeviceEnablePeerAccess
//     (single-process, all-GPUs-in-same-cuCtx topology).
//   - Each rank scatters its local input into a double-buffered slot
//     (selected by the call counter) before waiting for peers.
//   - Each rank publishes a data-ready flag (odd seq value) and waits for
//     all peers' data-ready flags before reading their data.
//   - Double buffering + the data-ready wait prevents any rank from getting
//     2+ calls ahead: a rank cannot complete call N+1 until all peers
//     publish data-ready for N+1, which requires them to have finished
//     call N, so the next call's slot is safe to reuse.
//   - Each rank reads from every peer's data buffer (at the current slot)
//     in parallel and sums.
//
// We use a *single block* per AllReduce — fine for hidden sizes up to
// block_size * VEC = 1024 * 8 = 8192 BF16 elements (16 KB). For Qwen3-32B
// (hidden=5120 BF16 = 10 KB) this covers the AllReduces emitted after o_proj
// and down_proj.
//
// CUDA Graph compatibility:
//   The seq counter lives in device memory. Each kernel call atomicAdds it to
//   obtain a fresh seq, so the captured graph node has no encoded seq and
//   replays produce the correct fresh seq each time.
//
// Thread-safety:
//   This is a single-stream API. Concurrent AllReduces from different streams
//   on the same instance would race on the seq counter and flag. Callers
//   should serialize via stream events.
// ---------------------------------------------------------------------------
#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace {

constexpr int P2P_AR_BLOCK_SIZE = 1024;
constexpr int P2P_AR_VEC_BF16 = 8;   // uint4 = 8 bf16
constexpr int P2P_AR_VEC_F32 = 4;    // uint4 = 4 fp32

// Spin-wait for `*flag >= target`. Volatile load forces fresh read from L2.
__device__ __forceinline__ void spin_until(volatile int* flag, int target) {
    while (*flag < target) { /* spin */ }
}

template <typename T, int VEC>
__global__ void __launch_bounds__(P2P_AR_BLOCK_SIZE, 1)
p2p_allreduce_oneshot_kernel(
    void* const* peer_data,            // [world_size] device-side ptrs (peer-mapped)
    int* const* peer_flags,            // [world_size] device-side ptrs (peer-mapped)
    unsigned long long* my_seq_counter,
    int my_rank,
    int world_size,
    int max_slot_bytes,               // bytes per double-buffer slot
    const T* __restrict__ in,
    T* __restrict__ out,
    int count) {

    // Single-block kernel.
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    __shared__ unsigned int s_seq;
    __shared__ int          s_slot_offset;
    __shared__ void*        s_peer_data[P2P_AR_MAX_WORLD];
    __shared__ int*         s_peer_flags[P2P_AR_MAX_WORLD];

    // Cache peer pointers + grab fresh seq (two phases per call: arrival + data-ready).
    if (tid == 0) {
        unsigned long long s = atomicAdd(my_seq_counter, 2ULL) + 2ULL;
        s_seq = (unsigned int)(s & 0x7FFFFFFEu);  // keep even, positive int range
        if (s_seq == 0) s_seq = 2;  // seq 0 collides with reset state; must be even
        s_slot_offset = ((s_seq >> 1) & 1) * max_slot_bytes;
    }
    if (tid < world_size) {
        s_peer_data[tid]  = peer_data[tid];
        s_peer_flags[tid] = peer_flags[tid];
    }
    __syncthreads();

    int seq = (int)s_seq;
    int slot_offset = s_slot_offset;
    T* my_data = reinterpret_cast<T*>(
        static_cast<char*>(s_peer_data[my_rank]) + slot_offset);

    // ---- Step 1: scatter local input into our peer-visible data buffer.
    // Double buffering ensures we write to a different slot than the one
    // peers are reading from the prior call, so this can happen before
    // waiting for peers.
    if (in != my_data) {
        const uint4* in_v4 = reinterpret_cast<const uint4*>(in);
        uint4*       out_v4 = reinterpret_cast<uint4*>(my_data);
        int count_v = count / VEC;
        for (int i = tid; i < count_v; i += bs) {
            out_v4[i] = in_v4[i];
        }
        int tail = count_v * VEC;
        for (int i = tail + tid; i < count; i += bs) {
            my_data[i] = in[i];
        }
    }

    // ---- Step 2: ensure all writes visible system-wide, then publish data-ready flag.
    // The seq counter uses even values for the per-call base; data-ready is seq + 1 (odd).
    // Double buffering + the data-ready wait prevents any rank from getting 2+ calls
    // ahead: a rank cannot complete call N+1 (and thus start N+2) until all peers
    // publish data-ready for N+1, which requires them to have finished call N.
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        volatile int* mf = s_peer_flags[my_rank];
        *mf = seq + 1;
    }

    // ---- Step 3: each thread waits on one peer's data-ready flag.
    if (tid < world_size) {
        volatile int* pf = s_peer_flags[tid];
        spin_until(pf, seq + 1);
    }
    __syncthreads();
    // Acquire-fence: ensure subsequent peer data loads are not reordered
    // above the flag observation. Combined with the producer-side
    // threadfence_system, this gives release/acquire ordering across the
    // PCIe domain.
    __threadfence_system();

    // ---- Step 4: read all peers at current slot, sum, write to local output.
    if constexpr (VEC == 8) {
        int count_v = count / 8;
        for (int i = tid; i < count_v; i += bs) {
            float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
            #pragma unroll
            for (int r = 0; r < P2P_AR_MAX_WORLD; ++r) {
                if (r >= world_size) break;
                const uint4* pv = reinterpret_cast<const uint4*>(
                    static_cast<const char*>(s_peer_data[r]) + slot_offset);
                uint4 raw = pv[i];
                auto* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
                a0 += __bfloat162float(h[0]);
                a1 += __bfloat162float(h[1]);
                a2 += __bfloat162float(h[2]);
                a3 += __bfloat162float(h[3]);
                a4 += __bfloat162float(h[4]);
                a5 += __bfloat162float(h[5]);
                a6 += __bfloat162float(h[6]);
                a7 += __bfloat162float(h[7]);
            }
            __nv_bfloat16 packed[8] = {
                __float2bfloat16(a0), __float2bfloat16(a1),
                __float2bfloat16(a2), __float2bfloat16(a3),
                __float2bfloat16(a4), __float2bfloat16(a5),
                __float2bfloat16(a6), __float2bfloat16(a7),
            };
            uint4 out_raw;
            __builtin_memcpy(&out_raw, packed, 16);
            reinterpret_cast<uint4*>(out)[i] = out_raw;
        }
        int tail = count_v * 8;
        for (int i = tail + tid; i < count; i += bs) {
            float s = 0.0f;
            #pragma unroll
            for (int r = 0; r < P2P_AR_MAX_WORLD; ++r) {
                if (r >= world_size) break;
                const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(
                    static_cast<const char*>(s_peer_data[r]) + slot_offset);
                s += __bfloat162float(p[i]);
            }
            out[i] = __float2bfloat16(s);
        }
    } else {
        int count_v = count / 4;
        for (int i = tid; i < count_v; i += bs) {
            float a0=0,a1=0,a2=0,a3=0;
            #pragma unroll
            for (int r = 0; r < P2P_AR_MAX_WORLD; ++r) {
                if (r >= world_size) break;
                const uint4* pv = reinterpret_cast<const uint4*>(
                    static_cast<const char*>(s_peer_data[r]) + slot_offset);
                uint4 raw = pv[i];
                auto* f = reinterpret_cast<const float*>(&raw);
                a0 += f[0]; a1 += f[1]; a2 += f[2]; a3 += f[3];
            }
            float packed[4] = {a0, a1, a2, a3};
            uint4 out_raw;
            __builtin_memcpy(&out_raw, packed, 16);
            reinterpret_cast<uint4*>(out)[i] = out_raw;
        }
        int tail = count_v * 4;
        for (int i = tail + tid; i < count; i += bs) {
            float s = 0.0f;
            #pragma unroll
            for (int r = 0; r < P2P_AR_MAX_WORLD; ++r) {
                if (r >= world_size) break;
                const float* p = reinterpret_cast<const float*>(
                    static_cast<const char*>(s_peer_data[r]) + slot_offset);
                s += p[i];
            }
            out[i] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// P2P AllGather – Column layout (contiguous per rank).
//
// Each rank scatters its shard to its P2P data buffer, then reads all peers'
// shards and writes them contiguously to the output:
//   output[rank * shard_bytes .. (rank+1) * shard_bytes] = peer_shard[rank]
//
// Dtype-agnostic: copies raw bytes using uint4 (16-byte) vectorisation.
// Uses the same double-buffered synchronisation as p2p_allreduce_oneshot_kernel.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(P2P_AR_BLOCK_SIZE, 1)
p2p_allgather_column_kernel(
    void* const* peer_data,
    int* const* peer_flags,
    unsigned long long* my_seq_counter,
    int my_rank, int world_size, int max_slot_bytes,
    const void* __restrict__ in,
    void* __restrict__ out,
    int shard_bytes)
{
    int tid = threadIdx.x;
    int bs  = blockDim.x;

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
    char* my_data = static_cast<char*>(s_peer_data[my_rank]) + slot_offset;

    // ---- Step 1: scatter local shard into our peer-visible data buffer.
    {
        const uint4* in_v4  = reinterpret_cast<const uint4*>(in);
        uint4*       my_v4  = reinterpret_cast<uint4*>(my_data);
        int num_vec = shard_bytes / 16;
        for (int i = tid; i < num_vec; i += bs) {
            my_v4[i] = in_v4[i];
        }
        int tail_start = num_vec * 16;
        const char* in_b = static_cast<const char*>(in);
        for (int i = tail_start + tid; i < shard_bytes; i += bs) {
            my_data[i] = in_b[i];
        }
    }

    // ---- Step 2: publish data-ready, wait for peers.
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        volatile int* mf = s_peer_flags[my_rank];
        *mf = seq + 1;
    }
    if (tid < world_size) {
        volatile int* pf = s_peer_flags[tid];
        spin_until(pf, seq + 1);
    }
    __syncthreads();
    __threadfence_system();

    // ---- Step 3: copy all peers' shards into output (contiguous per rank).
    char* out_b = static_cast<char*>(out);
    for (int r = 0; r < world_size; ++r) {
        const char* src = static_cast<const char*>(s_peer_data[r]) + slot_offset;
        char* dst = out_b + (size_t)r * shard_bytes;

        const uint4* src_v4 = reinterpret_cast<const uint4*>(src);
        uint4*       dst_v4 = reinterpret_cast<uint4*>(dst);
        int num_vec = shard_bytes / 16;
        for (int i = tid; i < num_vec; i += bs) {
            dst_v4[i] = src_v4[i];
        }
        int tail_start = num_vec * 16;
        for (int i = tail_start + tid; i < shard_bytes; i += bs) {
            dst[i] = src[i];
        }
    }
}

// ---------------------------------------------------------------------------
// P2P AllGather – Row layout (interleaved).
//
// Each rank scatters its shard to its P2P data buffer, then reads all peers'
// shards and writes them in interleaved layout:
//   dst = out + row * full_dim1_bytes + rank * shard_dim1_bytes
//   src = peer_shard[rank] + row * shard_dim1_bytes
//
// Dtype-agnostic: copies raw bytes using uint4 (16-byte) vectorisation.
// This eliminates the temp-buffer + memcpy2d that the NCCL Row AllGather
// path requires.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(P2P_AR_BLOCK_SIZE, 1)
p2p_allgather_row_kernel(
    void* const* peer_data,
    int* const* peer_flags,
    unsigned long long* my_seq_counter,
    int my_rank, int world_size, int max_slot_bytes,
    const void* __restrict__ in,
    void* __restrict__ out,
    int shard_bytes,
    int shard_dim1_bytes,
    int full_dim1_bytes,
    int outer)
{
    int tid = threadIdx.x;
    int bs  = blockDim.x;

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
    char* my_data = static_cast<char*>(s_peer_data[my_rank]) + slot_offset;

    // ---- Step 1: scatter local shard into our peer-visible data buffer.
    {
        const uint4* in_v4  = reinterpret_cast<const uint4*>(in);
        uint4*       my_v4  = reinterpret_cast<uint4*>(my_data);
        int num_vec = shard_bytes / 16;
        for (int i = tid; i < num_vec; i += bs) {
            my_v4[i] = in_v4[i];
        }
        int tail_start = num_vec * 16;
        const char* in_b = static_cast<const char*>(in);
        for (int i = tail_start + tid; i < shard_bytes; i += bs) {
            my_data[i] = in_b[i];
        }
    }

    // ---- Step 2: publish data-ready, wait for peers.
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        volatile int* mf = s_peer_flags[my_rank];
        *mf = seq + 1;
    }
    if (tid < world_size) {
        volatile int* pf = s_peer_flags[tid];
        spin_until(pf, seq + 1);
    }
    __syncthreads();
    __threadfence_system();

    // ---- Step 3: copy all peers' shards into output in interleaved layout.
    char* out_b = static_cast<char*>(out);
    for (int r = 0; r < world_size; ++r) {
        const char* src_base = static_cast<const char*>(s_peer_data[r]) + slot_offset;

        for (int row = 0; row < outer; ++row) {
            const char* src = src_base + (size_t)row * shard_dim1_bytes;
            char* dst = out_b + (size_t)row * full_dim1_bytes + (size_t)r * shard_dim1_bytes;

            const uint4* src_v4 = reinterpret_cast<const uint4*>(src);
            uint4*       dst_v4 = reinterpret_cast<uint4*>(dst);
            int num_vec = shard_dim1_bytes / 16;
            for (int i = tid; i < num_vec; i += bs) {
                dst_v4[i] = src_v4[i];
            }
            int tail_start = num_vec * 16;
            for (int i = tail_start + tid; i < shard_dim1_bytes; i += bs) {
                dst[i] = src[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// P2P Row-parallel RMSNorm
//
// Computes RMSNorm on a row-parallel tensor (each rank holds shard_dim columns
// of the full hidden dimension). Instead of allGathering the full tensor, this
// kernel only communicates a scalar (sum of squares) per row across ranks:
//
//   1. Each rank computes local sum(x_shard^2) per row
//   2. P2P sync: each rank writes its partial sums to its P2P buffer,
//      waits for all peers, then reads all peers' partial sums and sums them
//   3. Each rank computes inv_rms = rsqrt(total_sum / full_dim + eps)
//   4. Each rank normalizes its local shard: out = weight * x * inv_rms
//
// Communication cost: batch * sizeof(float) per rank (vs batch * full_dim * sizeof(bf16)
// for allGather). Output remains row-parallel (shard_dim columns per rank).
//
// Layout:
//   input:  [batch, shard_dim] BF16  (row-parallel shard)
//   weight: [shard_dim] BF16        (row-parallel shard of the weight vector)
//   output: [batch, shard_dim] BF16  (row-parallel shard)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(P2P_AR_BLOCK_SIZE, 1)
p2p_rmsnorm_kernel(
    void* const* peer_data,
    int* const* peer_flags,
    unsigned long long* my_seq_counter,
    int my_rank,
    int world_size,
    int max_slot_bytes,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float eps,
    int shard_dim,
    int full_dim,
    int batch)
{
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    __shared__ unsigned int s_seq;
    __shared__ int          s_slot_offset;
    __shared__ void*        s_peer_data[P2P_AR_MAX_WORLD];
    __shared__ int*         s_peer_flags[P2P_AR_MAX_WORLD];
    __shared__ float        s_inv_rms[2048];

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
    float* my_data = reinterpret_cast<float*>(
        static_cast<char*>(s_peer_data[my_rank]) + slot_offset);

    // ---- Step 1: Compute local sum of squares per row, write to P2P buffer.
    for (int row = tid; row < batch; row += bs) {
        float local_sum = 0.0f;
        const __nv_bfloat16* x = input + (size_t)row * shard_dim;
        for (int i = 0; i < shard_dim; ++i) {
            float f = __bfloat162float(x[i]);
            local_sum += f * f;
        }
        my_data[row] = local_sum;
    }

    // ---- Step 2: Publish data-ready, wait for peers.
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        volatile int* mf = s_peer_flags[my_rank];
        *mf = seq + 1;
    }
    if (tid < world_size) {
        volatile int* pf = s_peer_flags[tid];
        spin_until(pf, seq + 1);
    }
    __syncthreads();
    __threadfence_system();

    // ---- Step 3: Read all peers' partial sums, compute inv_rms per row.
    for (int row = tid; row < batch; row += bs) {
        float total_sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < P2P_AR_MAX_WORLD; ++r) {
            if (r >= world_size) break;
            const float* peer_buf = reinterpret_cast<const float*>(
                static_cast<const char*>(s_peer_data[r]) + slot_offset);
            total_sum += peer_buf[row];
        }
        s_inv_rms[row] = rsqrtf(total_sum / (float)full_dim + eps);
    }
    __syncthreads();

    // ---- Step 4: Normalize and apply weight.
    for (int row = 0; row < batch; ++row) {
        const __nv_bfloat16* x = input + (size_t)row * shard_dim;
        __nv_bfloat16* o = output + (size_t)row * shard_dim;
        float inv_rms = s_inv_rms[row];

        for (int i = tid; i < shard_dim; i += bs) {
            float f = __bfloat162float(x[i]);
            float w = __bfloat162float(weight[i]);
            o[i] = __float2bfloat16(w * f * inv_rms);
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int glm_p2p_enable_peer_access(GlmCtx* ctx, int peer_device) {
    cudaSetDevice(ctx->device_id);
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, ctx->device_id, peer_device);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_p2p_enable_peer_access: cudaDeviceCanAccessPeer(%d, %d) failed: %s\n",
                ctx->device_id, peer_device, cudaGetErrorString(err));
        return -1;
    }
    if (!can_access) return -1;
    err = cudaDeviceEnablePeerAccess(peer_device, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return 0;
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_p2p_enable_peer_access: cudaDeviceEnablePeerAccess(%d) on dev %d failed: %s\n",
                peer_device, ctx->device_id, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

GlmP2PInstance* glm_p2p_create_instance(GlmCtx* ctx, int my_rank, int world_size, size_t max_bytes) {
    if (world_size > P2P_AR_MAX_WORLD || world_size <= 0) {
        fprintf(stderr, "glm_p2p_create_instance: invalid world_size %d\n", world_size);
        return nullptr;
    }
    cudaSetDevice(ctx->device_id);
    auto* inst = new GlmP2PInstance();
    inst->world_size = world_size;
    inst->my_rank = my_rank;
    inst->max_bytes = max_bytes;
    inst->device_id = ctx->device_id;

    // Combined allocation: peer_data[N] | peer_flags[N] | seq_counter | flag | data_buffer[2]
    // Data buffer is doubled for ping-pong double buffering across calls.
    size_t header = sizeof(void*) * world_size
                   + sizeof(int*)  * world_size
                   + sizeof(unsigned long long)
                   + sizeof(int);
    size_t header_aligned = (header + 255) & ~size_t(255);  // 256B align data
    size_t total = header_aligned + max_bytes * 2;

    void* base = nullptr;
    cudaError_t err = cudaMalloc(&base, total);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_p2p_create_instance: cudaMalloc(%zu) failed: %s\n", total, cudaGetErrorString(err));
        delete inst;
        return nullptr;
    }
    cudaMemset(base, 0, total);  // synchronous zero so seq=0, flag=0 are safe initial state.

    char* p = static_cast<char*>(base);
    inst->base_alloc_d = base;
    inst->peer_data_arr_d  = reinterpret_cast<void**>(p); p += sizeof(void*) * world_size;
    inst->peer_flags_arr_d = reinterpret_cast<int**>(p);  p += sizeof(int*)  * world_size;
    inst->seq_counter_d    = reinterpret_cast<unsigned long long*>(p); p += sizeof(unsigned long long);
    inst->my_flag_d        = reinterpret_cast<int*>(p);   p += sizeof(int);
    inst->my_data_d        = static_cast<char*>(base) + header_aligned;
    return inst;
}

void glm_p2p_destroy_instance(GlmP2PInstance* inst) {
    if (!inst) return;
    cudaSetDevice(inst->device_id);
    cudaFree(inst->base_alloc_d);
    delete inst;
}

void* glm_p2p_get_data_ptr(GlmP2PInstance* inst) {
    return inst ? inst->my_data_d : nullptr;
}

int* glm_p2p_get_flag_ptr(GlmP2PInstance* inst) {
    return inst ? inst->my_flag_d : nullptr;
}

void glm_p2p_set_peers(GlmCtx* ctx, GlmP2PInstance* inst,
                       const void* const* peer_data_ptrs,
                       int* const* peer_flag_ptrs) {
    cudaSetDevice(ctx->device_id);
    int N = inst->world_size;
    cudaMemcpy(inst->peer_data_arr_d, peer_data_ptrs, sizeof(void*) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(inst->peer_flags_arr_d, peer_flag_ptrs, sizeof(int*) * N, cudaMemcpyHostToDevice);
}

void glm_p2p_allreduce(GlmCtx* ctx, GlmP2PInstance* inst,
                       const void* in, void* out, int count, int dtype) {
    cudaSetDevice(ctx->device_id);

    if (dtype == 9) {
        // Validate count fits in single block.
        // (block_size * VEC) = 1024 * 8 = 8192 elements max.
        if ((size_t)count * 2 > inst->max_bytes) {
            fprintf(stderr, "glm_p2p_allreduce: count %d * 2 > max_bytes %zu\n", count, inst->max_bytes);
            return;
        }
        p2p_allreduce_oneshot_kernel<__nv_bfloat16, P2P_AR_VEC_BF16>
            <<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const __nv_bfloat16*>(in),
                static_cast<__nv_bfloat16*>(out),
                count);
    } else if (dtype == 7) {
        if ((size_t)count * 4 > inst->max_bytes) {
            fprintf(stderr, "glm_p2p_allreduce: count %d * 4 > max_bytes %zu\n", count, inst->max_bytes);
            return;
        }
        p2p_allreduce_oneshot_kernel<float, P2P_AR_VEC_F32>
            <<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
                inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
                inst->my_rank, inst->world_size, (int)inst->max_bytes,
                static_cast<const float*>(in),
                static_cast<float*>(out),
                count);
    } else {
        fprintf(stderr, "glm_p2p_allreduce: unsupported dtype %d\n", dtype);
    }
}

size_t glm_p2p_max_bytes(GlmP2PInstance* inst) {
    return inst ? inst->max_bytes : 0;
}

void glm_p2p_allgather(GlmCtx* ctx, GlmP2PInstance* inst,
                        const void* sendbuf, void* recvbuf,
                        int num_bytes) {
    cudaSetDevice(ctx->device_id);

    if ((size_t)num_bytes > inst->max_bytes) {
        fprintf(stderr, "glm_p2p_allgather: num_bytes %d > max_bytes %zu\n", num_bytes, inst->max_bytes);
        return;
    }
    p2p_allgather_column_kernel<<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
        inst->my_rank, inst->world_size, (int)inst->max_bytes,
        sendbuf, recvbuf, num_bytes);
}

void glm_p2p_allgather_row(GlmCtx* ctx, GlmP2PInstance* inst,
                             const void* sendbuf, void* recvbuf,
                             int shard_bytes, int shard_dim1_bytes,
                             int full_dim1_bytes, int outer) {
    cudaSetDevice(ctx->device_id);

    if ((size_t)shard_bytes > inst->max_bytes) {
        fprintf(stderr, "glm_p2p_allgather_row: shard_bytes %d > max_bytes %zu\n", shard_bytes, inst->max_bytes);
        return;
    }
    p2p_allgather_row_kernel<<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
        inst->my_rank, inst->world_size, (int)inst->max_bytes,
        sendbuf, recvbuf, shard_bytes, shard_dim1_bytes, full_dim1_bytes, outer);
}

void glm_p2p_rmsnorm(GlmCtx* ctx, GlmP2PInstance* inst,
                      const void* input, const void* weight, void* output,
                      float eps, int shard_dim, int full_dim, int batch) {
    cudaSetDevice(ctx->device_id);

    size_t required_bytes = (size_t)batch * sizeof(float);
    if (required_bytes > inst->max_bytes) {
        fprintf(stderr, "glm_p2p_rmsnorm: batch %d requires %zu bytes > max_bytes %zu\n",
                batch, required_bytes, inst->max_bytes);
        return;
    }

    p2p_rmsnorm_kernel<<<1, P2P_AR_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_data_arr_d, inst->peer_flags_arr_d, inst->seq_counter_d,
        inst->my_rank, inst->world_size, (int)inst->max_bytes,
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output),
        eps, shard_dim, full_dim, batch);
}

} // extern "C"
