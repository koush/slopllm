// ---------------------------------------------------------------------------
// Custom P2P barrier and smem-staged AllGather for multi-GPU decode.
//
// P2P barrier: single-warp kernel that publishes a seq-counted flag to all
// peers via st.global.release.sys and spin-waits for their flags via
// ld.acquire.sys. Used before AllGather and reduce-scatter to ensure peer
// data is visible.
//
// Smem-staged AllGather: reads peer GPU memory into shared memory via
// cg::memcpy_async, then vector-copies to global output. D-adaptive pipeline
// depth, rank-rotated peer read order.
//
// CUDA Graph compatibility:
//   The seq counter lives in device memory. Each kernel call atomicAdds it to
//   obtain a fresh seq, so the captured graph node has no encoded seq and
//   replays produce the correct fresh seq each time.
//
// Thread-safety:
//   Single-stream API. Concurrent calls from different streams on the same
//   instance would race on the seq counter and flag.
// ---------------------------------------------------------------------------
#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <type_traits>
#include <cstdio>

namespace cg = cooperative_groups;

namespace {

constexpr int P2P_BARRIER_BLOCK_SIZE = 32;
constexpr int P2P_AR_VEC_BF16 = 8;   // uint4 = 8 bf16
constexpr int P2P_AR_VEC_F32 = 4;    // uint4 = 4 fp32

// ---------------------------------------------------------------------------
// P2P barrier, split into arrive + wait so callers can overlap work between
// publishing their flag and spinning on peers' flags.
//
// arrive: increment my_seq_counter, publish flag to peers (release.sys).
// wait:   read my_seq_counter back to recover the target, spin on peers' flags
//         (acquire.sys). Safe because the API is single-stream per instance:
//         no other arrive touches my_seq_counter between the two launches, so
//         *my_seq_counter == s (the value arrive computed) when wait reads it.
//
// Graph-capturable: my_seq_counter lives in device memory and is atomicAdd'd
// fresh each replay; wait reads it device-side, so no host-encoded seq.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(P2P_BARRIER_BLOCK_SIZE, 1)
p2p_arrive_kernel(
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

    bool active = (peer_rank < 0 && tid < world_size) ||
                  (peer_rank >= 0 && tid == peer_rank);

    if (active) {
        int val = seq + 1;
        if (tid == my_rank) {
            s_peer_flags[my_rank][my_rank] = val;
        } else {
            // per Claude
            // Kernel completion on GPU *i* is a system-scope synchronizing event:
            // it drains the device's caches and write buffers to the point of coherence.
            // This is exactly why `cudaMemcpyPeerAsync` after a kernel on the same stream works,
            // and why the host can read results after `cudaStreamSynchronize`. So yes — all of
            // kernel 1's P2P stores are pushed out before kernel 2 begins issuing. That edge is
            // real, and the `.release` *ordering* on the flag store is redundant with respect to it.

            asm volatile("st.global.relaxed.sys.s32 [%0], %1;"
                         :: "l"(s_peer_flags[tid] + my_rank), "r"(val));

            // so this is not needed because the kernel boundary gaurantees it.
            // however if the arrive/barrier/op is FUSED then it would be needed.
            // asm volatile("st.global.release.sys.s32 [%0], %1;"
            //              :: "l"(s_peer_flags[tid] + my_rank), "r"(val));
        }
    }
    __syncwarp();
    (void)nanosleep_ns;  // unused on the publish side
}

__global__ void __launch_bounds__(P2P_BARRIER_BLOCK_SIZE, 1)
p2p_wait_kernel(
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
        // arrive already ran on this stream, so *my_seq_counter == s. Recover
        // the same s_seq arrive published (with the same wrap guard) and derive
        // the target from it.
        unsigned long long s = *my_seq_counter;
        s_seq = (unsigned int)s;
        if (s_seq == 0) s_seq = 2;
    }
    if (tid < world_size) {
        s_peer_flags[tid] = peer_flags[tid];
    }
    __syncwarp();

    int seq = (int)s_seq;

    bool active = (peer_rank < 0 && tid < world_size) ||
                  (peer_rank >= 0 && tid == peer_rank);

    if (active) {
        int target = seq + 1;
        int* my_flags = s_peer_flags[my_rank];
        int v;
        do {
            asm volatile("ld.acquire.sys.b32 %0, [%1];"
                         : "=r"(v) : "l"(my_flags + tid));
            if ((int)((unsigned)v - (unsigned)target) < 0) __nanosleep(nanosleep_ns);
        } while ((int)((unsigned)v - (unsigned)target) < 0);
    }
    __syncwarp();
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
            // All threads must finish draining this slot before the next peer
            // is streamed into it (cg::wait_prior only orders the fill side).
            __syncthreads();
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
                // All threads must finish draining this slot before the next
                // peer is streamed into it (wait_prior only orders the fill).
                __syncthreads();
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

// ---------------------------------------------------------------------------
// Fused P2P AllReduce + Add + RMSNorm (smem-staged).
//
// Same peer read pattern as sum_pointers_smem_kernel (glm_ops.cu): N peer
// partial-sum pointers stream through a D-deep smem pipeline while threads
// accumulate in FP32 registers. After accumulation, inputA (residual) is added,
// then each row of `dim` elements is RMSNormed:
//   s        = inputA + sum_{peer<N} p_peer
//   residual = s
//   out      = weight * s * rsqrt(mean(s^2) + eps)
//
// Row geometry: ElemsPerWarp (512) divides dim, so each warp's 512 elements
// lie within a single row; one row spans warps_per_row = dim / ElemsPerWarp
// warps. After accumulation every warp shuffle-reduces its sum-of-squares,
// writes one float to a small reduction region at the end of smem, one
// __syncthreads, then sums its row's warps_per_row slots to get inv_rms.
// Host enforces dim % 512 == 0 and 8192 % dim == 0 (row-aligned tiles).
// ---------------------------------------------------------------------------

template <typename scalar_t, int ElemsPerWarp, int D_VAL>
__global__ void __launch_bounds__(512, 2)
rmsnorm_pointers_smem_kernel(
    const scalar_t* p0,  const scalar_t* p1,
    const scalar_t* p2,  const scalar_t* p3,
    const scalar_t* p4,  const scalar_t* p5,
    const scalar_t* p6,  const scalar_t* p7,
    const scalar_t* __restrict__ inputA,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ residual,
    int N, int64_t numel, int64_t peer_stride_elems, int dim, float eps)
{
    constexpr int WarpSize = 32;
    auto block = cg::this_thread_block();
    extern __shared__ char smem_raw[];

    int warp_id = threadIdx.x / WarpSize;
    int lane    = threadIdx.x % WarpSize;
    int warps_per_block = blockDim.x / WarpSize;
    int64_t block_stride = (int64_t)warps_per_block * ElemsPerWarp;
    int64_t total_blocks = gridDim.x;
    int64_t my_start = (int64_t)blockIdx.x * block_stride;

    const char* peers[8] = {
        reinterpret_cast<const char*>(p0), reinterpret_cast<const char*>(p1),
        reinterpret_cast<const char*>(p2), reinterpret_cast<const char*>(p3),
        reinterpret_cast<const char*>(p4), reinterpret_cast<const char*>(p5),
        reinterpret_cast<const char*>(p6), reinterpret_cast<const char*>(p7),
    };

    constexpr int VEC = (sizeof(scalar_t) == 2) ? 2 : 1;
    constexpr int PAIRS = ElemsPerWarp / (WarpSize * VEC);
    int64_t warp_start = (int64_t)warp_id * ElemsPerWarp;
    const size_t elem_sz = sizeof(scalar_t);

    int warps_per_row = dim / ElemsPerWarp;                       // >= 1
    int row_warp_base = (warp_id / warps_per_row) * warps_per_row;
    int weight_base   = (warp_id % warps_per_row) * ElemsPerWarp; // weight col for this warp

    // Reduction slots at the end of smem, past the peer pipeline region.
    float* reduce_smem = reinterpret_cast<float*>(
        smem_raw + (size_t)D_VAL * peer_stride_elems * elem_sz);

    for (int64_t blk = my_start; blk < numel; blk += total_blocks * block_stride) {
        int64_t elems = min(block_stride, numel - blk);
        size_t copy_bytes = (size_t)elems * elem_sz;

        float2 acc[PAIRS];
        #pragma unroll
        for (int k = 0; k < PAIRS; k++) acc[k] = {0.0f, 0.0f};

        // --- Peer pipeline (identical read pattern to sum_pointers_smem) ---
        int P = min(D_VAL, N);
        #pragma unroll
        for (int k = 0; k < D_VAL; k++) {
            if (k < P)
                cg::memcpy_async(block,
                    smem_raw + (size_t)k * peer_stride_elems * elem_sz,
                    peers[k] + (size_t)blk * elem_sz,
                    copy_bytes);
        }

        int steady = max(0, N - D_VAL);
        for (int j = 0; j < steady; j++) {
            cg::wait_prior<D_VAL - 1>(block);
            char* buf = smem_raw + (size_t)(j % D_VAL) * peer_stride_elems * elem_sz;
            #pragma unroll
            for (int k = 0; k < PAIRS; k++) {
                int64_t off = warp_start + lane * VEC + (int64_t)k * WarpSize * VEC;
                if (off + VEC <= elems) {
                    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
                        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(buf + off * elem_sz);
                        float2 f = __bfloat1622float2(v);
                        acc[k].x += f.x; acc[k].y += f.y;
                    } else {
                        float v = *reinterpret_cast<const float*>(buf + off * elem_sz);
                        acc[k].x += v;
                    }
                }
            }
            // All threads must finish reading this slot before the next peer is
            // streamed into it (cg::wait_prior only orders the fill side).
            __syncthreads();
            cg::memcpy_async(block,
                smem_raw + (size_t)(j % D_VAL) * peer_stride_elems * elem_sz,
                peers[j + D_VAL] + (size_t)blk * elem_sz,
                copy_bytes);
        }

        cg::wait(block);
        for (int j = steady; j < N; j++) {
            char* buf = smem_raw + (size_t)(j % D_VAL) * peer_stride_elems * elem_sz;
            #pragma unroll
            for (int k = 0; k < PAIRS; k++) {
                int64_t off = warp_start + lane * VEC + (int64_t)k * WarpSize * VEC;
                if (off + VEC <= elems) {
                    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
                        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(buf + off * elem_sz);
                        float2 f = __bfloat1622float2(v);
                        acc[k].x += f.x; acc[k].y += f.y;
                    } else {
                        float v = *reinterpret_cast<const float*>(buf + off * elem_sz);
                        acc[k].x += v;
                    }
                }
            }
        }

        // --- Add inputA (residual), accumulate sum-of-squares ---
        float sum_sq = 0.0f;
        #pragma unroll
        for (int k = 0; k < PAIRS; k++) {
            int64_t off = warp_start + lane * VEC + (int64_t)k * WarpSize * VEC;
            if (off + VEC <= elems) {
                if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
                    float2 av = load_bf16x2(inputA + (blk + off));
                    acc[k].x += av.x; acc[k].y += av.y;
                    sum_sq += acc[k].x * acc[k].x + acc[k].y * acc[k].y;
                } else {
                    float a = inputA[blk + off];
                    acc[k].x += a;
                    sum_sq += acc[k].x * acc[k].x;
                }
            }
        }

        // --- Intra-warp shuffle reduce ---
        #pragma unroll
        for (int offset = WarpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

        // --- Cross-warp exchange: 1 slot/warp, 1 sync, sum row partners ---
        if (lane == 0) reduce_smem[warp_id] = sum_sq;
        __syncthreads();
        float row_sq = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            row_sq += reduce_smem[row_warp_base + w];
        float inv_rms = rsqrtf(row_sq / (float)dim + eps);

        // --- Write residual and normed output ---
        #pragma unroll
        for (int k = 0; k < PAIRS; k++) {
            int64_t off = warp_start + lane * VEC + (int64_t)k * WarpSize * VEC;
            if (off + VEC <= elems) {
                int64_t gidx = blk + off;
                int wcol = weight_base + lane * VEC + (int64_t)k * WarpSize * VEC;
                if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
                    store_bf16x2(residual + gidx, acc[k].x, acc[k].y);
                    float2 wv = load_bf16x2(weight + wcol);
                    store_bf16x2(out + gidx, wv.x * acc[k].x * inv_rms,
                                            wv.y * acc[k].y * inv_rms);
                } else {
                    residual[gidx] = acc[k].x;
                    float w = weight[wcol];
                    out[gidx] = w * acc[k].x * inv_rms;
                }
            }
        }

        __syncthreads();   // smem reads done before next tile's cp.async
    }
}

// ---------------------------------------------------------------------------
// Write-based Row AllGather: each GPU reads its local shard from HBM and
// writes it to all N peers' output buffers via NVLink. Fire-and-forget writes
// avoid the round-trip latency of P2P reads. The caller issues arrive before
// this kernel and wait after, so the barrier absorbs write latency.
// ---------------------------------------------------------------------------

constexpr int AG_WRITE_THREADS = 128;

template <int N>
__global__ void __launch_bounds__(AG_WRITE_THREADS, 4)
p2p_allgather_row_write_kernel(
    const void* __restrict__ local_shard,
    void* __restrict__ out0,  void* __restrict__ out1,
    void* __restrict__ out2,  void* __restrict__ out3,
    void* __restrict__ out4,  void* __restrict__ out5,
    void* __restrict__ out6,  void* __restrict__ out7,
    int shard_dim1_bytes,
    int full_dim1_bytes,
    int outer,
    int rank)
{
    constexpr int VEC = 16;  // int4
    const int tid = threadIdx.x;
    const int n_vec = shard_dim1_bytes / VEC;
    const int tail = shard_dim1_bytes - n_vec * VEC;

    void* dsts[AG_SMEM_MAX_N] = {
        static_cast<char*>(out0), static_cast<char*>(out1),
        static_cast<char*>(out2), static_cast<char*>(out3),
        static_cast<char*>(out4), static_cast<char*>(out5),
        static_cast<char*>(out6), static_cast<char*>(out7),
    };

    for (int row = blockIdx.x; row < outer; row += gridDim.x)
    {
        const char* src = static_cast<const char*>(local_shard)
                          + (int64_t)row * shard_dim1_bytes;

        #pragma unroll
        for (int j = 0; j < N; j++) {
            int peer = (j + blockIdx.x) % N;
            char* dst = static_cast<char*>(dsts[peer])
                        + (int64_t)row * full_dim1_bytes
                        + (int64_t)rank * shard_dim1_bytes;
            for (int i = tid; i < n_vec; i += AG_WRITE_THREADS)
                *reinterpret_cast<int4*>(dst + i * VEC) =
                    *reinterpret_cast<const int4*>(src + i * VEC);
            if (tail > 0) {
                int ti = n_vec * VEC + tid;
                if (ti < shard_dim1_bytes) dst[ti] = src[ti];
            }
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

GlmP2PInstance* glm_p2p_create_instance(GlmCtx* ctx, int my_rank, int world_size,
                                        const int* device_ids) {
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
                                  ctx->device_id, device_ids[p]);
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

void glm_p2p_arrive(GlmCtx* ctx, GlmP2PInstance* inst, int peer_rank) {
    cudaSetDevice(ctx->device_id);
    p2p_arrive_kernel<<<1, P2P_BARRIER_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_flags_arr_d, inst->seq_counter_d,
        inst->my_rank, inst->world_size,
        inst->nanosleep_ns, peer_rank);
}

void glm_p2p_wait(GlmCtx* ctx, GlmP2PInstance* inst, int peer_rank) {
    cudaSetDevice(ctx->device_id);
    p2p_wait_kernel<<<1, P2P_BARRIER_BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
        inst->peer_flags_arr_d, inst->seq_counter_d,
        inst->my_rank, inst->world_size,
        inst->nanosleep_ns, peer_rank);
}

void glm_p2p_barrier(GlmCtx* ctx, GlmP2PInstance* inst, int peer_rank) {
    glm_p2p_arrive(ctx, inst, peer_rank);
    glm_p2p_wait(ctx, inst, peer_rank);
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

#define LAUNCH_AG_ROW_WRITE(N) \
    do { \
        p2p_allgather_row_write_kernel<N><<<grid, AG_WRITE_THREADS, 0, stream>>>( \
            local_shard, \
            const_cast<void*>(p[0]), const_cast<void*>(p[1]), \
            const_cast<void*>(p[2]), const_cast<void*>(p[3]), \
            const_cast<void*>(p[4]), const_cast<void*>(p[5]), \
            const_cast<void*>(p[6]), const_cast<void*>(p[7]), \
            shard_dim1_bytes, full_dim1_bytes, outer, rank); \
    } while(0)

void glm_p2p_allgather_row_write(GlmCtx* ctx,
    const void* local_shard,
    const void* p0,  const void* p1,  const void* p2,  const void* p3,
    const void* p4,  const void* p5,  const void* p6,  const void* p7,
    void* /*output*/, int N, int shard_dim1_bytes, int full_dim1_bytes, int outer,
    int rank) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);
    const void* p[8] = {p0, p1, p2, p3, p4, p5, p6, p7};

    int grid = outer;
    if (grid > 512) grid = 512;
    if (grid < 1) grid = 1;

    switch (N) {
        case 2:  LAUNCH_AG_ROW_WRITE(2);  break;
        case 4:  LAUNCH_AG_ROW_WRITE(4);  break;
        case 8:  LAUNCH_AG_ROW_WRITE(8);  break;
        default:
            fprintf(stderr, "glm_p2p_allgather_row_write: unsupported N=%d\n", N);
            break;
    }
    #undef LAUNCH_AG_ROW_WRITE
}

// ---------------------------------------------------------------------------
// Fused P2P AllReduce + Add + RMSNorm launcher.
// ---------------------------------------------------------------------------

#define LAUNCH_RMSNORM_PTRS(SCT, DVAL) do { \
    cudaFuncSetAttribute( \
        (void*)rmsnorm_pointers_smem_kernel<SCT, ElemsPerWarp, DVAL>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 65536); \
    rmsnorm_pointers_smem_kernel<SCT, ElemsPerWarp, DVAL><<<grid, block_size, smem_bytes, stream>>>( \
        (const SCT*)p0, (const SCT*)p1, (const SCT*)p2, (const SCT*)p3, \
        (const SCT*)p4, (const SCT*)p5, (const SCT*)p6, (const SCT*)p7, \
        (const SCT*)inputA, (const SCT*)weight, \
        (SCT*)out, (SCT*)residual, \
        N, numel, peer_stride_elems, dim, eps); \
} while (0)

void glm_rmsnorm_pointers_smem(GlmCtx* ctx,
    const void* p0,  const void* p1,  const void* p2,  const void* p3,
    const void* p4,  const void* p5,  const void* p6,  const void* p7,
    const void* inputA, const void* weight,
    void* out, void* residual,
    int N, int64_t numel, int dim, float eps, int dtype) {
    cudaSetDevice(ctx->device_id);
    cudaStream_t stream = GLM_STREAM(ctx);

    constexpr int ElemsPerWarp = 512;
    constexpr int WarpsPerBlock = 16;
    constexpr int64_t BlockStride = (int64_t)WarpsPerBlock * ElemsPerWarp;  // 8192
    constexpr int64_t SmemBudget = 32 * 1024;

    // Row-alignment requirements (see kernel comment).
    if (dim <= 0 || ElemsPerWarp <= 0 || dim % ElemsPerWarp != 0 ||
        BlockStride % dim != 0 || N <= 0 || N > 8) {
        fprintf(stderr, "glm_rmsnorm_pointers_smem: invalid args dim=%d N=%d "
                        "(need dim%%512==0 and 8192%%dim==0, 1<=N<=8)\n", dim, N);
        return;
    }

    int elem_size = (dtype == 9) ? 2 : 4;
    int64_t peer_stride_elems = (numel < BlockStride) ? numel : BlockStride;
    int64_t peer_stride_bytes = peer_stride_elems * elem_size;
    if (peer_stride_bytes < 1) peer_stride_bytes = 1;
    int64_t budget = SmemBudget / peer_stride_bytes;
    int D = (int)((budget < N) ? budget : N);
    if (D < 1) D = 1;
    if (D > 8) D = 8;

    int64_t total_warps = (numel + ElemsPerWarp - 1) / ElemsPerWarp;
    if (total_warps == 0) total_warps = 1;
    int grid = (int)((total_warps + WarpsPerBlock - 1) / WarpsPerBlock);
    int block_size = WarpsPerBlock * 32;
    int64_t smem_bytes = (int64_t)D * peer_stride_bytes
                       + (int64_t)WarpsPerBlock * sizeof(float);

    if (dtype == 9) {
        switch (D) {
            case 8: LAUNCH_RMSNORM_PTRS(__nv_bfloat16, 8); break;
            case 7: LAUNCH_RMSNORM_PTRS(__nv_bfloat16, 7); break;
            case 6: LAUNCH_RMSNORM_PTRS(__nv_bfloat16, 6); break;
            case 5: LAUNCH_RMSNORM_PTRS(__nv_bfloat16, 5); break;
            case 4: LAUNCH_RMSNORM_PTRS(__nv_bfloat16, 4); break;
            case 3: LAUNCH_RMSNORM_PTRS(__nv_bfloat16, 3); break;
            case 2: LAUNCH_RMSNORM_PTRS(__nv_bfloat16, 2); break;
            default: LAUNCH_RMSNORM_PTRS(__nv_bfloat16, 1); break;
        }
    } else {
        switch (D) {
            case 8: LAUNCH_RMSNORM_PTRS(float, 8); break;
            case 7: LAUNCH_RMSNORM_PTRS(float, 7); break;
            case 6: LAUNCH_RMSNORM_PTRS(float, 6); break;
            case 5: LAUNCH_RMSNORM_PTRS(float, 5); break;
            case 4: LAUNCH_RMSNORM_PTRS(float, 4); break;
            case 3: LAUNCH_RMSNORM_PTRS(float, 3); break;
            case 2: LAUNCH_RMSNORM_PTRS(float, 2); break;
            default: LAUNCH_RMSNORM_PTRS(float, 1); break;
        }
    }
    #undef LAUNCH_RMSNORM_PTRS
}

} // extern "C"
