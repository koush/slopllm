// ---------------------------------------------------------------------------
// Custom P2P "one-shot" AllReduce for small messages (single-block design).
//
// Targets PCIe-only multi-GPU systems where NCCL ring AllReduce is ~30-50 us
// per call due to multi-hop launch latency. This implementation completes a
// 10 KB AllReduce in ~5-10 us by:
//   - Mapping every peer's data buffer directly via cudaDeviceEnablePeerAccess
//     (single-process, all-GPUs-in-same-cuCtx topology).
//   - Each rank scatters its local input into its peer-visible buffer.
//   - Each rank uses a two-phase flag protocol per call (even = arrival,
//     odd = data-ready) with a single flag word visible to all peers.
//   - Arrival phase: each rank publishes its flag and waits for all peers,
//     preventing a fast rank from overwriting its buffer before slow peers
//     finish reading from the prior call.
//   - Data-ready phase: after writing data, each rank publishes an odd flag
//     value and waits for all peers before reading their data.
//   - Each rank reads from every peer's data buffer in parallel and sums.
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

constexpr int P2P_AR_MAX_WORLD = 8;
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
    const T* __restrict__ in,
    T* __restrict__ out,
    int count) {

    // Single-block kernel.
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    __shared__ unsigned int s_seq;       // truncated 32-bit seq for flag sig
    __shared__ void*       s_peer_data[P2P_AR_MAX_WORLD];
    __shared__ int*        s_peer_flags[P2P_AR_MAX_WORLD];

    // Cache peer pointers + grab fresh seq (two phases per call: arrival + data-ready).
    if (tid == 0) {
        unsigned long long s = atomicAdd(my_seq_counter, 2ULL) + 2ULL;
        s_seq = (unsigned int)(s & 0x7FFFFFFEu);  // keep even, positive int range
        if (s_seq == 0) s_seq = 2;  // seq 0 collides with reset state; must be even
    }
    if (tid < world_size) {
        s_peer_data[tid]  = peer_data[tid];
        s_peer_flags[tid] = peer_flags[tid];
    }
    __syncthreads();

    int seq = (int)s_seq;
    T* my_data = static_cast<T*>(s_peer_data[my_rank]);

    // ---- Step 0: arrival barrier.
    // Each rank publishes its arrival (even seq) and waits for all peers.
    // This prevents a fast rank from overwriting its data buffer before
    // slow peers finish reading from the prior call.
    if (tid == 0) {
        volatile int* mf = s_peer_flags[my_rank];
        *mf = seq;
    }
    if (tid < world_size) {
        volatile int* pf = s_peer_flags[tid];
        spin_until(pf, seq);
    }
    __syncthreads();

    // ---- Step 1: scatter local input into our peer-visible data buffer.
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
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        volatile int* mf = s_peer_flags[my_rank];
        *mf = seq + 1;  // data-ready phase: odd value
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

    // ---- Step 4: read all peers, sum, write to local output.
    if constexpr (VEC == 8) {
        int count_v = count / 8;
        for (int i = tid; i < count_v; i += bs) {
            float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
            #pragma unroll
            for (int r = 0; r < P2P_AR_MAX_WORLD; ++r) {
                if (r >= world_size) break;
                const uint4* pv = reinterpret_cast<const uint4*>(s_peer_data[r]);
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
                const __nv_bfloat16* p = static_cast<const __nv_bfloat16*>(s_peer_data[r]);
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
                const uint4* pv = reinterpret_cast<const uint4*>(s_peer_data[r]);
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
                const float* p = static_cast<const float*>(s_peer_data[r]);
                s += p[i];
            }
            out[i] = s;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

struct GlmP2PInstance {
    void**              peer_data_arr_d;   // device array of N peer data ptrs
    int**               peer_flags_arr_d;  // device array of N peer flag ptrs
    unsigned long long* seq_counter_d;     // device-resident seq counter
    int*                my_flag_d;         // device pointer to this rank's flag
    void*               my_data_d;         // device pointer to this rank's data buf
    void*               base_alloc_d;      // base alloc to free
    size_t              max_bytes;
    int                 world_size;
    int                 my_rank;
    int                 device_id;
};

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

    // Combined allocation: peer_data[N] | peer_flags[N] | seq_counter | flag | data_buffer
    size_t header = sizeof(void*) * world_size
                  + sizeof(int*)  * world_size
                  + sizeof(unsigned long long)
                  + sizeof(int);
    size_t header_aligned = (header + 255) & ~size_t(255);  // 256B align data
    size_t total = header_aligned + max_bytes;

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
                inst->my_rank, inst->world_size,
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
                inst->my_rank, inst->world_size,
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

} // extern "C"
