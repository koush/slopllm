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
#include "glm_p2p_common.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>

namespace cg = cooperative_groups;

namespace {

constexpr int P2P_BARRIER_BLOCK_SIZE = 32;
constexpr int P2P_AR_VEC_BF16 = 8;   // uint4 = 8 bf16
constexpr int P2P_AR_VEC_F32 = 4;    // uint4 = 4 fp32

// ---------------------------------------------------------------------------
// P2P barrier: increment seq, publish flag, wait for peers. No data scatter.
// Writes slot_offset so callers know which double-buffer slot was selected.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(P2P_BARRIER_BLOCK_SIZE, 1)
p2p_barrier_kernel(
    int* const* peer_flags,
    unsigned long long* my_seq_counter,
    int my_rank,
    int world_size,
    int nanosleep_ns,
    int peer_rank = -1)
{
    int tid = threadIdx.x;

    __shared__ unsigned int s_seq;
    __shared__ int*         s_peer_flags[P2P_AR_MAX_WORLD];

    if (tid == 0) {
        unsigned long long s = atomicAdd(my_seq_counter, 2ULL) + 2ULL;
        s_seq = (unsigned int)s;
        if (s_seq == 0) s_seq = 2;
    }
    if (tid < world_size) {
        s_peer_flags[tid] = peer_flags[tid];
    }
    __syncwarp();

    int seq = (int)s_seq;

    p2p_publish_and_wait(tid, my_rank, world_size, seq,
                         s_peer_flags, s_peer_flags[my_rank],
                         nanosleep_ns, peer_rank);
}


// ---------------------------------------------------------------------------
// Smem-staged AllGather kernels (barrier + async copy + double buffer).
//
// Reads directly from peer GPU memory into smem via cg::memcpy_async in
// fixed-size tiles (1024 bytes), then scalar-writes to global.
// Double-buffered: while writing from buf[0], buf[1] reads are in flight.
// cg::wait_prior<N> lets us wait for buf[0] while buf[1] is still pending.
//
// Requires a p2p_barrier before launch to ensure peer data is visible.
// Peer pointers are passed as 8 separate args (like sum_pointers_smem_kernel).
// Block rotation by blockIdx.x % N distributes P2P read load across peers.
// ---------------------------------------------------------------------------

constexpr int AG_SMEM_MAX_N = 8;

template <int N, int D_VAL>
__global__ void __launch_bounds__(512, 2)
p2p_allgather_smem_kernel(
    const void* __restrict__ p0,  const void* __restrict__ p1,
    const void* __restrict__ p2,  const void* __restrict__ p3,
    const void* __restrict__ p4,  const void* __restrict__ p5,
    const void* __restrict__ p6,  const void* __restrict__ p7,
    void* __restrict__ output,
    int shard_bytes,
    int rank)
{
    constexpr int THREADS = 512;
    constexpr int VEC = 16;
    constexpr int BLOCK_BYTES = THREADS * VEC;     // 8192

    auto block = cg::this_thread_block();
    extern __shared__ char smem_raw[];

    const char* ptrs[AG_SMEM_MAX_N] = {
        static_cast<const char*>(p0), static_cast<const char*>(p1),
        static_cast<const char*>(p2), static_cast<const char*>(p3),
        static_cast<const char*>(p4), static_cast<const char*>(p5),
        static_cast<const char*>(p6), static_cast<const char*>(p7),
    };

    int tile_bytes = (BLOCK_BYTES < shard_bytes) ? BLOCK_BYTES : shard_bytes;
    int peer_stride = (tile_bytes + VEC - 1) & ~(VEC - 1);
    int tid = threadIdx.x;
    int P = (D_VAL < N) ? D_VAL : N;

    for (int64_t blk_base = (int64_t)blockIdx.x * tile_bytes;
         blk_base < (int64_t)shard_bytes;
         blk_base += (int64_t)gridDim.x * tile_bytes)
    {
        int elems = (tile_bytes < shard_bytes - blk_base) ? tile_bytes : shard_bytes - blk_base;
        int n_vec = elems / VEC;
        int tail  = elems - n_vec * VEC;

        #pragma unroll
        for (int k = 0; k < D_VAL; k++) {
            if (k < P) {
                int peer = (k + rank) % N;
                cg::memcpy_async(block,
                    smem_raw + (size_t)k * peer_stride,
                    ptrs[peer] + blk_base,
                    elems);
            }
        }

        int steady = (N > D_VAL) ? N - D_VAL : 0;
        for (int j = 0; j < steady; j++) {
            cg::wait_prior<D_VAL - 1>(block);
            int slot = j % D_VAL;
            int actual_peer = (j + rank) % N;
            const char* src = smem_raw + (size_t)slot * peer_stride;
            char* dst = static_cast<char*>(output)
                        + (int64_t)actual_peer * shard_bytes + blk_base;
            for (int i = tid; i < n_vec; i += THREADS)
                *reinterpret_cast<int4*>(dst + i * VEC) =
                    *reinterpret_cast<const int4*>(src + i * VEC);
            if (tail > 0) {
                int ti = n_vec * VEC + tid;
                if (ti < elems) dst[ti] = src[ti];
            }
            int next_peer = (j + D_VAL + rank) % N;
            cg::memcpy_async(block,
                smem_raw + (size_t)slot * peer_stride,
                ptrs[next_peer] + blk_base,
                elems);
        }

        cg::wait(block);
        for (int j = steady; j < N; j++) {
            int slot = j % D_VAL;
            int actual_peer = (j + rank) % N;
            const char* src = smem_raw + (size_t)slot * peer_stride;
            char* dst = static_cast<char*>(output)
                        + (int64_t)actual_peer * shard_bytes + blk_base;
            for (int i = tid; i < n_vec; i += THREADS)
                *reinterpret_cast<int4*>(dst + i * VEC) =
                    *reinterpret_cast<const int4*>(src + i * VEC);
            if (tail > 0) {
                int ti = n_vec * VEC + tid;
                if (ti < elems) dst[ti] = src[ti];
            }
        }
        __syncthreads();
    }
}

// Redesigned allgather: 1 block per row (no no-op blocks), 512 threads,
// D_VAL-deep smem pipeline streaming N peers, vectorized int4 smem→global
// copy.  Rotation by `rank` spreads peer-read order across GPUs so they
// don't all hammer peer 0 first.  D_VAL adapts to data size (host picks
// D = min(N, 32KB / peer_stride)): small data → D=N (all in flight, single
// drain); large data → D=2..4 (double/triple-buffer pipeline).
template <int N, int D_VAL>
__global__ void __launch_bounds__(512, 2)
p2p_allgather_row_smem_kernel(
    const void* __restrict__ p0,  const void* __restrict__ p1,
    const void* __restrict__ p2,  const void* __restrict__ p3,
    const void* __restrict__ p4,  const void* __restrict__ p5,
    const void* __restrict__ p6,  const void* __restrict__ p7,
    void* __restrict__ output,
    int shard_dim1_bytes,
    int full_dim1_bytes,
    int outer,
    int rank)
{
    constexpr int THREADS = 512;
    constexpr int VEC = 16;                       // int4
    constexpr int BLOCK_BYTES = THREADS * VEC;     // 8192

    auto block = cg::this_thread_block();
    extern __shared__ char smem_raw[];

    const char* ptrs[AG_SMEM_MAX_N] = {
        static_cast<const char*>(p0), static_cast<const char*>(p1),
        static_cast<const char*>(p2), static_cast<const char*>(p3),
        static_cast<const char*>(p4), static_cast<const char*>(p5),
        static_cast<const char*>(p6), static_cast<const char*>(p7),
    };

    int tile_bytes = (BLOCK_BYTES < shard_dim1_bytes) ? BLOCK_BYTES : shard_dim1_bytes;
    int peer_stride = (tile_bytes + VEC - 1) & ~(VEC - 1);
    int num_tiles = (shard_dim1_bytes + tile_bytes - 1) / tile_bytes;

    int tid = threadIdx.x;
    int P = (D_VAL < N) ? D_VAL : N;

    for (int row = blockIdx.x; row < outer; row += gridDim.x)
    {
        char* row_out = static_cast<char*>(output) + (int64_t)row * full_dim1_bytes;

        for (int tile = 0; tile < num_tiles; tile++)
        {
            int64_t blk = (int64_t)tile * tile_bytes;
            int elems = (tile_bytes < shard_dim1_bytes - blk) ? tile_bytes : shard_dim1_bytes - blk;
            int n_vec = elems / VEC;
            int tail  = elems - n_vec * VEC;

            // Issue first D_VAL peer reads (rotated by rank)
            #pragma unroll
            for (int k = 0; k < D_VAL; k++) {
                if (k < P) {
                    int peer = (k + rank) % N;
                    cg::memcpy_async(block,
                        smem_raw + (size_t)k * peer_stride,
                        ptrs[peer] + (int64_t)row * shard_dim1_bytes + blk,
                        elems);
                }
            }

            // Steady: stream remaining N-D_VAL peers through D_VAL smem slots
            int steady = (N > D_VAL) ? N - D_VAL : 0;
            for (int j = 0; j < steady; j++) {
                cg::wait_prior<D_VAL - 1>(block);
                int slot = j % D_VAL;
                int actual_peer = (j + rank) % N;
                const char* src = smem_raw + (size_t)slot * peer_stride;
                char* dst = row_out + (int64_t)actual_peer * shard_dim1_bytes + blk;
                for (int i = tid; i < n_vec; i += THREADS)
                    *reinterpret_cast<int4*>(dst + i * VEC) =
                        *reinterpret_cast<const int4*>(src + i * VEC);
                if (tail > 0) {
                    int ti = n_vec * VEC + tid;
                    if (ti < elems) dst[ti] = src[ti];
                }
                int next_peer = (j + D_VAL + rank) % N;
                cg::memcpy_async(block,
                    smem_raw + (size_t)slot * peer_stride,
                    ptrs[next_peer] + (int64_t)row * shard_dim1_bytes + blk,
                    elems);
            }

            // Drain remaining D_VAL peers
            cg::wait(block);
            for (int j = steady; j < N; j++) {
                int slot = j % D_VAL;
                int actual_peer = (j + rank) % N;
                const char* src = smem_raw + (size_t)slot * peer_stride;
                char* dst = row_out + (int64_t)actual_peer * shard_dim1_bytes + blk;
                for (int i = tid; i < n_vec; i += THREADS)
                    *reinterpret_cast<int4*>(dst + i * VEC) =
                        *reinterpret_cast<const int4*>(src + i * VEC);
                if (tail > 0) {
                    int ti = n_vec * VEC + tid;
                    if (ti < elems) dst[ti] = src[ti];
                }
            }
            __syncthreads();   // ensure drain's ld.shared done before next tile's cp.async
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

GlmP2PInstance* glm_p2p_create_instance(GlmCtx* ctx, int my_rank, int world_size) {
    if (world_size > P2P_AR_MAX_WORLD || world_size <= 0) {
        fprintf(stderr, "glm_p2p_create_instance: invalid world_size %d\n", world_size);
        return nullptr;
    }
    cudaSetDevice(ctx->device_id);
    auto* inst = new GlmP2PInstance();
    inst->world_size = world_size;
    inst->my_rank = my_rank;
    inst->device_id = ctx->device_id;

    // Detect NVLink vs PCIe: use higher performance rank = NVLink (fast poll),
    // lower = PCIe (back off more to reduce interconnect contention).
    int min_perf_rank = 0x7fffffff;
    for (int p = 0; p < world_size; p++) {
        if (p == my_rank) continue;
        int rank = 0;
        cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank,
                                  ctx->device_id, p);
        if (rank < min_perf_rank) min_perf_rank = rank;
    }
    inst->nanosleep_ns = (min_perf_rank > 0) ? 32 : 200;

    // Metadata-only allocation: peer_flags[N] | seq_counter | flags[N]
    size_t header = sizeof(int*)  * world_size
                   + sizeof(unsigned long long)
                   + sizeof(int)  * world_size;
    size_t header_aligned = (header + 255) & ~size_t(255);  // 256B align

    void* base = nullptr;
    cudaError_t err = cudaMalloc(&base, header_aligned);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_p2p_create_instance: cudaMalloc(%zu) failed: %s\n", header_aligned, cudaGetErrorString(err));
        delete inst;
        return nullptr;
    }
    cudaMemset(base, 0, header_aligned);

    char* p = static_cast<char*>(base);
    inst->metadata_alloc_d = base;
    inst->peer_flags_arr_d = reinterpret_cast<int**>(p);  p += sizeof(int*)  * world_size;
    inst->seq_counter_d    = reinterpret_cast<unsigned long long*>(p); p += sizeof(unsigned long long);
    inst->my_flags_d       = reinterpret_cast<int*>(p);
    return inst;
}

void glm_p2p_destroy_instance(GlmP2PInstance* inst) {
    if (!inst) return;
    cudaSetDevice(inst->device_id);
    cudaFree(inst->metadata_alloc_d);
    delete inst;
}

int* glm_p2p_get_flag_ptr(GlmP2PInstance* inst) {
    return inst ? inst->my_flags_d : nullptr;
}

void glm_p2p_set_peers(GlmCtx* ctx, GlmP2PInstance* inst,
                       int* const* peer_flag_ptrs) {
    cudaSetDevice(ctx->device_id);
    int N = inst->world_size;
    cudaMemcpy(inst->peer_flags_arr_d, peer_flag_ptrs, sizeof(int*) * N, cudaMemcpyHostToDevice);
}

void glm_p2p_barrier(GlmCtx* ctx, GlmP2PInstance* inst, int peer_rank) {
    cudaSetDevice(ctx->device_id);
    p2p_barrier_kernel<<<1, P2P_BARRIER_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_flags_arr_d, inst->seq_counter_d,
        inst->my_rank, inst->world_size,
        inst->nanosleep_ns, peer_rank);
}

// ---------------------------------------------------------------------------
// Launchers for smem-staged AllGather kernels.
// Caller must invoke p2p_barrier before calling these.
// Peer pointers are passed directly (no P2P buffer/slot mechanism needed).
// ---------------------------------------------------------------------------

#define LAUNCH_AG_SMEM(N, D) \
    do { \
        p2p_allgather_smem_kernel<N, D><<<grid, 512, smem, stream>>>( \
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], \
            output, shard_bytes, rank); \
    } while(0)

void glm_p2p_allgather_smem(GlmCtx* ctx,
    const void* p0,  const void* p1,  const void* p2,  const void* p3,
    const void* p4,  const void* p5,  const void* p6,  const void* p7,
    void* output, int N, int shard_bytes, int rank) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    const void* p[8] = {p0, p1, p2, p3, p4, p5, p6, p7};

    constexpr int BLOCK_BYTES = 8192;
    constexpr int SMEM_BUDGET  = 32768;
    int tile_bytes = (BLOCK_BYTES < shard_bytes) ? BLOCK_BYTES : shard_bytes;
    int peer_stride = (tile_bytes + 15) & ~15;
    int D = N;
    if (peer_stride > 0) {
        int cap = SMEM_BUDGET / peer_stride;
        D = (N < cap) ? N : cap;
    }
    if (D < 1) D = 1;
    size_t smem = (size_t)D * peer_stride;
    int num_tiles = (shard_bytes + tile_bytes - 1) / tile_bytes;
    int grid = num_tiles;
    if (grid > 512) grid = 512;
    if (grid < 1) grid = 1;

    switch (N) {
        case 2:
            switch (D) {
                case 1: LAUNCH_AG_SMEM(2, 1); break;
                default: LAUNCH_AG_SMEM(2, 2); break;
            }
            break;
        case 4:
            switch (D) {
                case 1: LAUNCH_AG_SMEM(4, 1); break;
                case 2: LAUNCH_AG_SMEM(4, 2); break;
                case 3: LAUNCH_AG_SMEM(4, 3); break;
                default: LAUNCH_AG_SMEM(4, 4); break;
            }
            break;
        case 8:
            switch (D) {
                case 1: LAUNCH_AG_SMEM(8, 1); break;
                case 2: LAUNCH_AG_SMEM(8, 2); break;
                case 3: LAUNCH_AG_SMEM(8, 3); break;
                case 4: LAUNCH_AG_SMEM(8, 4); break;
                case 5: LAUNCH_AG_SMEM(8, 5); break;
                case 6: LAUNCH_AG_SMEM(8, 6); break;
                case 7: LAUNCH_AG_SMEM(8, 7); break;
                default: LAUNCH_AG_SMEM(8, 8); break;
            }
            break;
        default:
            fprintf(stderr, "glm_p2p_allgather_smem: unsupported N=%d\n", N);
            break;
    }
    #undef LAUNCH_AG_SMEM
}

#define LAUNCH_AG_ROW_SMEM(N, D) \
    do { \
        p2p_allgather_row_smem_kernel<N, D><<<grid, 512, smem, stream>>>( \
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], \
            output, shard_dim1_bytes, full_dim1_bytes, outer, rank); \
    } while(0)

void glm_p2p_allgather_row_smem(GlmCtx* ctx,
    const void* p0,  const void* p1,  const void* p2,  const void* p3,
    const void* p4,  const void* p5,  const void* p6,  const void* p7,
    void* output, int N, int shard_dim1_bytes, int full_dim1_bytes, int outer,
    int rank) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    const void* p[8] = {p0, p1, p2, p3, p4, p5, p6, p7};

    constexpr int BLOCK_BYTES = 8192;
    constexpr int SMEM_BUDGET  = 32768;
    int tile_bytes = (BLOCK_BYTES < shard_dim1_bytes) ? BLOCK_BYTES : shard_dim1_bytes;
    int peer_stride = (tile_bytes + 15) & ~15;
    int D = N;
    if (peer_stride > 0) {
        int cap = SMEM_BUDGET / peer_stride;
        D = (N < cap) ? N : cap;
    }
    if (D < 1) D = 1;
    size_t smem = (size_t)D * peer_stride;
    int grid = outer;
    if (grid > 512) grid = 512;
    if (grid < 1) grid = 1;

    switch (N) {
        case 2:
            switch (D) {
                case 1: LAUNCH_AG_ROW_SMEM(2, 1); break;
                default: LAUNCH_AG_ROW_SMEM(2, 2); break;
            }
            break;
        case 4:
            switch (D) {
                case 1: LAUNCH_AG_ROW_SMEM(4, 1); break;
                case 2: LAUNCH_AG_ROW_SMEM(4, 2); break;
                case 3: LAUNCH_AG_ROW_SMEM(4, 3); break;
                default: LAUNCH_AG_ROW_SMEM(4, 4); break;
            }
            break;
        case 8:
            switch (D) {
                case 1: LAUNCH_AG_ROW_SMEM(8, 1); break;
                case 2: LAUNCH_AG_ROW_SMEM(8, 2); break;
                case 3: LAUNCH_AG_ROW_SMEM(8, 3); break;
                case 4: LAUNCH_AG_ROW_SMEM(8, 4); break;
                case 5: LAUNCH_AG_ROW_SMEM(8, 5); break;
                case 6: LAUNCH_AG_ROW_SMEM(8, 6); break;
                case 7: LAUNCH_AG_ROW_SMEM(8, 7); break;
                default: LAUNCH_AG_ROW_SMEM(8, 8); break;
            }
            break;
        default:
            fprintf(stderr, "glm_p2p_allgather_row_smem: unsupported N=%d\n", N);
            break;
    }
    #undef LAUNCH_AG_ROW_SMEM
}

} // extern "C"
