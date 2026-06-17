// FP4 Identity Dequant via MXF8F6F4 Block-Scaled MMA
//
// Dequantizes MXFP4 weights: dequant = I_fp4 @ W_fp4 * scale_I * scale_W
//
// MMA: mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X
//      .m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0
//
// Register loading: this MMA op (SM120_16x8x32_TN_VS) inherits its
// ALayout/BLayout from the 8-bit SM80_16x8x32_S32S8S8S32_TN traits, and
// ldmatrix.sync.aligned.m8n16.x4/x2.shared.b8x16.b4x16_p64 (the dedicated
// Blackwell FP4 ldmatrix variant, Copy_Atom<SM100_SU4_DU8x16_x4/x2_LDSM_N>)
// does NOT produce data matching that inherited ALayout for this particular
// op — confirmed empirically by probing every register slot's source (m,k)
// with a uniquely-valued smem tile: roughly 3/4 of each thread's fragment
// reads back zero regardless of the per-lane address scheme used (uniform
// pointer, the address ldmatrix.x4's own SrcLayout implies, or CuTe's
// make_tiled_copy_A/partition_S with or without an explicit Tile<> mma
// permutation). So this kernel does not use ldmatrix for either operand:
//
//   - A is the identity matrix, which is synthesizable in closed form
//     (1.0 iff m==k), so it needs no smem/load at all — just compute
//     each thread's 16 register bytes directly from ALayout's per-thread
//     (M,K) mapping.
//   - B's weight data is loaded with a plain indexed smem read (sB(n,k))
//     using BLayout's per-thread (N,K) mapping, instead of ldmatrix.
//
// ALayout/BLayout per-thread mapping (mma_traits_sm80.hpp), with
// T = lane = t0 + 4*t1  (t0 = lane%4, t1 = lane/4):
//   A: m = t1 + 8*v1,  k = 4*t0 + v0 + 16*v2,  reg = v1 + 2*v2,  byte = v0
//   B: n = t1,         k = 4*t0 + v0 + 16*v1,  reg = v1,         byte = v0
//
// STATUS: verified correct against the reference dequant (0 errors, PASS).
//
// Build:
//   nvcc -O2 -std=c++20 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --extended-lambda \
//     -Ivendor/flashinfer/3rdparty/cutlass/include \
//     -Ivendor/flashinfer/3rdparty/cccl/libcudacxx/include \
//     -Icsrc \
//     scratchpad/fp4_identity_dequant.cu -o scratchpad/fp4_identity_dequant

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cutlass/float_subbyte.h>
#include <cutlass/float8.h>

#include <cfloat>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace cute;

// ─── Constants ───────────────────────────────────────────────────────────────

constexpr int MMA_M  = 16;
constexpr int MMA_N  = 8;
constexpr int MMA_K  = 32;
constexpr int SF_VS  = 32;
constexpr int WARPSZ = 32;
constexpr uint8_t E8M0_ONE = 127;

// ─── E2M1 / E8M0 helpers ────────────────────────────────────────────────────

__host__ __device__ __forceinline__ float e2m1_nibble_to_float(uint32_t nibble) {
    uint32_t n = nibble & 0x7u;
    uint32_t fp = (n < 2u) ? (n * 0x3F000000u)
                            : (((126u + (n >> 1u)) << 23u) | ((n & 1u) << 22u));
    fp |= (uint32_t)(nibble >> 3u) << 31u;
    float result;
    memcpy(&result, &fp, sizeof(result));
    return result;
}

__host__ __device__ __forceinline__ float e8m0_to_float(uint8_t raw) {
    return exp2f((float)((int)raw - 127));
}

// ─── CuTe type aliases ──────────────────────────────────────────────────────

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementC = float;
using ElementSF = float_ue8m0_t;

// B's smem tile: one FP4 value per byte (sparse — not the 2-per-byte format
// used in HBM), K-major, no swizzle. GMMA::Layout_K_INTER_Atom<uint8_t> =
// upcast<8>(Layout<Shape<_8,_128>,Stride<_128,_1>>) = Layout<Shape<_8,_16>,
// Stride<_16,_1>> in uint8_t units, tiled here to the full (N,K) element shape.
using SmemLayoutAtomB = decltype(GMMA::Layout_K_INTER_Atom<uint8_t>{});
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{}, make_shape(Int<MMA_N>{}, Int<MMA_K>{}), Step<_1, _2>{}));

constexpr int SMEM_B_SIZE = static_cast<int>(cosize(SmemLayoutB{}));
constexpr int SMEM_SFB_SIZE = MMA_N;  // one E8M0 per N column
constexpr int SMEM_TOTAL = SMEM_B_SIZE + SMEM_SFB_SIZE;

// ─── Kernel ──────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(WARPSZ)
fp4_identity_dequant_kernel(
    const uint8_t* __restrict__ fp4_weights,
    const uint8_t* __restrict__ e8m0_scales,
    float* __restrict__ output,
    int K, int N)
{
    int lane_id = threadIdx.x;
    int m_start = blockIdx.x * MMA_M;
    int n_start = blockIdx.y * MMA_N;
    int k_group = m_start / SF_VS;
    int k_start = k_group * SF_VS;
    int pattern = (m_start % MMA_K) / MMA_M;  // which 16 of the 32 K columns this M-block's diagonal hits

    if (m_start >= K || n_start >= N) return;

    extern __shared__ uint8_t smem_buf[];
    uint8_t* b_smem   = smem_buf;
    uint8_t* sfb_smem = b_smem + SMEM_B_SIZE;

    for (int i = lane_id; i < SMEM_TOTAL; i += WARPSZ) smem_buf[i] = 0;

    // Load weight tile B from global (2-per-byte compressed) into smem
    // (1-per-byte sparse), matching the MMA's expected register format.
    {
        auto sB = make_tensor(make_smem_ptr(b_smem), SmemLayoutB{});
        int N_packed = (N + 1) / 2;
        int n_valid = min(MMA_N, N - n_start);
        int k_valid = min(MMA_K, K - k_start);
        for (int idx = lane_id; idx < n_valid * MMA_K; idx += WARPSZ) {
            int n_local = idx / MMA_K;
            int k_local = idx % MMA_K;
            int n_global = n_start + n_local;
            int k_global = k_start + k_local;
            if (k_global >= K) continue;
            uint8_t gbyte = fp4_weights[k_global * N_packed + n_global / 2];
            sB(n_local, k_local) = (gbyte >> ((n_global % 2) * 4)) & 0xF;
        }
    }

    // Load E8M0 scales
    for (int n = lane_id; n < min(MMA_N, N - n_start); n += WARPSZ)
        sfb_smem[n] = e8m0_scales[k_group * N + n_start + n];

    __syncthreads();

    // Block-scaled MMA atom: 16x8x32 with VS=32. Used here only to derive
    // the correctly-shaped register fragments (partition_fragment_A/B) and
    // to apply the FP4 left-shift (ldmatrix-style low-nibble -> MMA's
    // expected bit position) via the library's fp4_shift_A/B helpers.
    using MMA_Op = SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
        ElementA, ElementB, ElementC, ElementSF, SF_VS>;
    using MMA_Atom = MMA_Atom<MMA_Traits<MMA_Op>>;
    using TiledMMA = decltype(make_tiled_mma(MMA_Atom{}));
    auto thr_mma = TiledMMA{}.get_thread_slice(lane_id);

    // A's fragment needs no real smem backing (it's synthesized directly
    // below) — this tensor exists only so partition_fragment_A can derive
    // the right per-thread register shape from SmemLayoutA's (M,K) extent.
    using SmemLayoutAtomA = decltype(GMMA::Layout_K_INTER_Atom<uint8_t>{});
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{}, make_shape(Int<MMA_M>{}, Int<MMA_K>{}), Step<_1, _2>{}));
    auto sA_shape_only = make_tensor(make_smem_ptr(b_smem), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(b_smem), SmemLayoutB{});

    auto tCrA = thr_mma.partition_fragment_A(sA_shape_only);
    auto tCrB = thr_mma.partition_fragment_B(sB);

    // A: identity matrix, synthesized directly in registers (no load at
    // all — see ALayout mapping in the file header comment).
    // B: gathered directly from smem via BLayout's per-thread (N,K) mapping.
    {
        int t0 = lane_id % 4, t1 = lane_id / 4;
        int col_offset = pattern * MMA_M;
        auto rA8 = recast<uint8_t>(tCrA);
        for (int v2 = 0; v2 < 2; v2++)
            for (int v1 = 0; v1 < 2; v1++)
                for (int v0 = 0; v0 < 4; v0++) {
                    int m = t1 + 8 * v1;
                    int k = 4 * t0 + v0 + 16 * v2;
                    rA8((v1 + 2 * v2) * 4 + v0) = (k == m + col_offset) ? 0x02 : 0x00;
                }

        auto rB8 = recast<uint8_t>(tCrB);
        for (int v1 = 0; v1 < 2; v1++)
            for (int v0 = 0; v0 < 4; v0++) {
                int n = t1;
                int k = 4 * t0 + v0 + 16 * v1;
                rB8(v1 * 4 + v0) = sB(n, k);
            }
    }

    // FP4 left-shift: registers hold the raw nibble in the low bits;
    // MMA F8F6F4 expects it in bits [5:2].
    fp4_shift_A(MMA_Op{}, tCrA);
    fp4_shift_B(MMA_Op{}, tCrB);

    // SFB: one E8M0 per N column. Per MMA_Traits<SM120_16x8x32_TN_VS<...>>'s
    // SFBLayout (mma_traits_sm120.hpp): Layout<Shape<Shape<_4,_8>,_32>,
    // Stride<Stride<_0,_1>,_8>> -> N = T/4 (matches BLayout's n=t1, the same
    // per-thread N index used to load B itself).
    uint8_t sfb = sfb_smem[lane_id / 4];
    uint8_t sfa = E8M0_ONE;  // identity scale = 2^0 = 1.0

    // ─── Block-scaled MMA ─────────────────────────────────────────────────────

    auto rA = recast<uint32_t>(tCrA);
    auto rB = recast<uint32_t>(tCrB);

    float frag_c[4] = {0,0,0,0};
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(frag_c[0]), "=f"(frag_c[1]), "=f"(frag_c[2]), "=f"(frag_c[3])
        : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]),
          "r"(rB[0]), "r"(rB[1]),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
          "r"((uint32_t)sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"((uint32_t)sfb), "h"((uint16_t)0), "h"((uint16_t)0)
    );

    // Write output: CLayout SM80_16x8_Row
    int m0 = lane_id / 4, m1 = m0 + 8;
    int col0 = (lane_id % 4) * 2, col1 = col0 + 1;
    auto write = [&](int m, int n, float v) {
        int mg = m_start + m, ng = n_start + n;
        if (mg < K && ng < N) output[mg * N + ng] = v;
    };
    write(m0, col0, frag_c[0]); write(m0, col1, frag_c[1]);
    write(m1, col0, frag_c[2]); write(m1, col1, frag_c[3]);
}

// ─── Reference ────────────────────────────────────────────────────────────────

void fp4_dequant_reference(
    const uint8_t* fp4_weights, const uint8_t* e8m0_scales,
    float* output, int K, int N)
{
    int N_packed = (N + 1) / 2;
    for (int k = 0; k < K; k++) {
        int kg = k / SF_VS;
        for (int n = 0; n < N; n++) {
            uint8_t packed = fp4_weights[k * N_packed + n / 2];
            uint32_t nib = (n % 2 == 0) ? (packed & 0xF) : (packed >> 4);
            output[k * N + n] = e2m1_nibble_to_float(nib) * e8m0_to_float(e8m0_scales[kg * N + n]);
        }
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    int K = 64, N = 32;
    int N_packed = (N + 1) / 2;
    int n_kgroups = K / SF_VS;

    std::vector<uint8_t> h_fp4(K * N_packed, 0);
    std::vector<uint8_t> h_scales(n_kgroups * N, 0);
    std::vector<float> h_output(K * N, 0);
    std::vector<float> h_ref(K * N, 0);

    srand(42);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n += 2) {
            uint8_t lo = rand() % 16, hi = (n+1<N) ? (rand()%16) : 0;
            h_fp4[k * N_packed + n/2] = (lo & 0xF) | ((hi & 0xF) << 4);
        }
    for (int kg = 0; kg < n_kgroups; kg++)
        for (int n = 0; n < N; n++)
            h_scales[kg * N + n] = 120 + (rand() % 16);

    fp4_dequant_reference(h_fp4.data(), h_scales.data(), h_ref.data(), K, N);

    uint8_t *d_fp4, *d_scales; float *d_output;
    cudaMalloc(&d_fp4, K * N_packed);
    cudaMalloc(&d_scales, n_kgroups * N);
    cudaMalloc(&d_output, K * N * sizeof(float));
    cudaMemcpy(d_fp4, h_fp4.data(), K * N_packed, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), n_kgroups * N, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, K * N * sizeof(float));

    dim3 grid(K / MMA_M, N / MMA_N);
    dim3 block(WARPSZ);
    fp4_identity_dequant_kernel<<<grid, block, SMEM_TOTAL>>>(
        d_fp4, d_scales, d_output, K, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(h_output.data(), d_output, K * N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0; int n_err = 0;
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) {
            float d = fabsf(h_output[k*N+n] - h_ref[k*N+n]);
            float r = fabsf(h_ref[k*N+n]) > 1e-6f ? d / fabsf(h_ref[k*N+n]) : d;
            if (r > 0.15f || d > 1.0f) { n_err++; if (d > max_err) max_err = d; }
        }

    printf("K=%d N=%d: %d errors, max_abs_err=%.4f\n", K, N, n_err, max_err);
    printf("%s\n", n_err == 0 ? "PASS" : "FAIL");

    cudaFree(d_fp4); cudaFree(d_scales); cudaFree(d_output);
    return n_err == 0 ? 0 : 1;
}
