#include "glm_ops.h"
#include "glm_nvfp4.cuh"
#include <cublasLt.h>

namespace {
// cuBLASLt VEC32_UE8M0: 128 rows x four K32 scales per tile.
__device__ size_t mx_scale_offset(int row, int group, int k) {
    return size_t(row / 128) * (k / 128) * 512 + (group / 4) * 512
        + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
}

__global__ void fp8_pack_scales(uint8_t* packed, const uint8_t* scales, int n, int k) {
    int padded_k = (k + 127) / 128 * 128;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int row = i / (padded_k / 32), g = i % (padded_k / 32);
    if (row < ((n + 127) / 128) * 128)
        packed[mx_scale_offset(row, g, padded_k)] = row < n && g < k / 32 ? scales[(row / 32) * (k / 32) + g] : 0;
}

__global__ void fp8_pad_weight(uint8_t* out, const uint8_t* weight, int n, int k, int padded_k) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(n) * padded_k) out[i] = i % padded_k < k ? weight[(i / padded_k) * k + i % padded_k] : 0;
}

__global__ void fp8_quantize_rows(uint8_t* out, uint8_t* scales, const __nv_bfloat16* input,
                                int m, int k, int stride) {
    int lane = threadIdx.x % 32;
    int padded_k = (k + 127) / 128 * 128;
    int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int row = group / (padded_k / 32), g = group % (padded_k / 32);
    if (row >= ((m + 127) / 128) * 128) return;
    float value = row < m && g * 32 + lane < k ? __bfloat162float(input[size_t(row) * stride + g * 32 + lane]) : 0.f;
    float amax = fabsf(value);
    for (int d = 16; d; d /= 2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d));
    int exponent = max(-127, min(127, int(ceilf(log2f(fmaxf(amax, 1e-4f) / 448.f)))));
    if (lane == 0) scales[mx_scale_offset(row, g, padded_k)] = row < m && g * 32 < k ? exponent + 127 : 0;
    if (row < m) out[size_t(row) * padded_k + g * 32 + lane] =
        __nv_cvt_float_to_fp8(ldexpf(value, -exponent), __NV_SATFINITE, __NV_E4M3);
}
}

// Scratch belongs to the caller; no device allocation or weight transformation
// persists beyond this projection. Groups execute in order on the active stream.
extern "C" int glm_fp8_linear_lt(GlmCtx* ctx, void* output, const void* input,
    const void* weight, const void* scales, void* quantized, void* input_scales,
    void* weight_scales, void* scratch, int m, int n, int k, int groups) {
    if (m <= 0 || n <= 0 || k <= 0 || groups <= 0 || n % groups ||
        (n / groups) % 32 || k % 32) return CUBLAS_STATUS_INVALID_VALUE;
    cudaSetDevice(ctx->device_id);
    const int rows = n / groups;
    const int padded_k = (k + 127) / 128 * 128;
    constexpr size_t scratch_bytes = 32 * 1024 * 1024;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    auto run = [&]() -> int {
#define FP8_LT_CHECK(call) do { auto status = (call); if (status != CUBLAS_STATUS_SUCCESS) return int(status); } while (0)
        FP8_LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t trans = CUBLAS_OP_T;
        auto mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        FP8_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
        FP8_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
        FP8_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
        FP8_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scales, sizeof(weight_scales)));
        FP8_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &input_scales, sizeof(input_scales)));
        FP8_LT_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, padded_k, rows, padded_k));
        FP8_LT_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, padded_k, m, padded_k));
        FP8_LT_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, rows, m, n));
        FP8_LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        FP8_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &scratch_bytes, sizeof(scratch_bytes)));
        cublasLtMatmulHeuristicResult_t heuristic;
        int count = 0;
        auto handle = reinterpret_cast<cublasLtHandle_t>(ctx->cublaslt_handle);
        FP8_LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, op, a, b, c, c, pref, 1, &heuristic, &count));
        if (!count) return CUBLAS_STATUS_NOT_SUPPORTED;
        float alpha = 1.f, beta = 0.f;
        for (int g = 0; g < groups; ++g) {
            fp8_pack_scales<<<(((rows + 127) / 128) * 128 * (padded_k / 32) + 255) / 256, 256, 0, GLM_STREAM(ctx)>>>(
                (uint8_t*)weight_scales, (const uint8_t*)scales + size_t(g) * (rows / 32) * (k / 32), rows, k);
            fp8_quantize_rows<<<(((m + 127) / 128) * 128 * padded_k + 255) / 256, 256, 0, GLM_STREAM(ctx)>>>(
                (uint8_t*)quantized, (uint8_t*)input_scales, (const __nv_bfloat16*)input + g * k, m, k, groups * k);
            const uint8_t* w = (const uint8_t*)weight + size_t(g) * rows * k;
            if (k != padded_k) {
                auto* padded = (uint8_t*)scratch + scratch_bytes;
                fp8_pad_weight<<<(size_t(rows) * padded_k + 255) / 256, 256, 0, GLM_STREAM(ctx)>>>(padded, w, rows, k, padded_k);
                w = padded;
            }
            auto error = cudaGetLastError();
            if (error != cudaSuccess) return 1000 + int(error);
            auto* out = (__nv_bfloat16*)output + g * rows;
            FP8_LT_CHECK(cublasLtMatmul(handle, op, &alpha, w, a,
                quantized, b, &beta, out, c, out, c, &heuristic.algo, scratch, scratch_bytes, GLM_STREAM(ctx)));
        }
        return 0;
#undef FP8_LT_CHECK
    };
    int status = run();
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
    return status;
}

namespace {
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int WORK_VALUES = 16;

struct Bf16Scale {
    using Storage = __nv_bfloat16;
    __device__ static float decode(Storage value) { return __bfloat162float(value); }
};

struct E8m0Scale {
    using Storage = uint8_t;
    __device__ static float decode(Storage value) {
        const uint32_t e = value;
        return __uint_as_float(e == 255 ? 0x7fc00000u : e == 0 ? 0x00400000u : e << 23);
    }
};

// Based on the tuned routed FP4 CUDA-core path: two uint4 activation loads,
// register-local dot products and a subwarp reduction. FP8 needs one uint4
// weight load for the same 16 logical values. Always use the high-lane FP8
// conversion helper (the low half of packed E4M3 conversion faults on GPU7).
template<bool Aligned>
__device__ __forceinline__ float dot16(const uint8_t* weight, const __nv_bfloat16* input, int valid) {
    float2 sum = {0.f, 0.f};
    if constexpr (Aligned) {
        const uint4 packed = *reinterpret_cast<const uint4*>(weight);
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        const uint4 xv0 = reinterpret_cast<const uint4*>(input)[0];
        const uint4 xv1 = reinterpret_cast<const uint4*>(input)[1];
        const auto* x0 = reinterpret_cast<const __nv_bfloat162*>(&xv0);
        const auto* x1 = reinterpret_cast<const __nv_bfloat162*>(&xv1);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t word = words[j / 2] >> ((j % 2) * 16);
            const auto w0 = __float2bfloat16(fp8_e4m3_to_float(uint8_t(word)));
            const auto w1 = __float2bfloat16(fp8_e4m3_to_float(uint8_t(word >> 8)));
            const auto x = j < 4 ? x0[j] : x1[j - 4];
            sum.x = bf16_fma_f32(w0, x.x, sum.x);
            sum.y = bf16_fma_f32(w1, x.y, sum.y);
        }
    } else {
        // Non-vector-aligned row strides and the final partial work unit.
        #pragma unroll
        for (int j = 0; j < 16; j += 2) {
            if (j < valid) sum.x = fmaf(fp8_e4m3_to_float(weight[j]), float(input[j]), sum.x);
            if (j + 1 < valid) sum.y = fmaf(fp8_e4m3_to_float(weight[j + 1]), float(input[j + 1]), sum.y);
        }
    }
    return sum.x + sum.y;
}

template<int RowsPerWarp, int ScaleRows, int ScaleCols, class Scale, bool Aligned>
__global__ void __launch_bounds__(BLOCK_SIZE, 4) fp8_linear_kernel(
    __nv_bfloat16* __restrict__ output, const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ weight, const typename Scale::Storage* __restrict__ scales,
    int N, int K, int groups) {
    static_assert(ScaleCols % WORK_VALUES == 0);
    constexpr int LANES_PER_ROW = WARP_SIZE / RowsPerWarp;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK * RowsPerWarp + warp * RowsPerWarp + lane / LANES_PER_ROW;
    const int inner_lane = lane % LANES_PER_ROW;
    const int token = blockIdx.y;
    const int work_units = (K + WORK_VALUES - 1) / WORK_VALUES;
    const int scale_stride = (K + ScaleCols - 1) / ScaleCols;
    float sum = 0.f;
    if (row < N) {
        const auto* x = input + (size_t(token) * groups + row / (N / groups)) * K;
        const auto* w = weight + size_t(row) * K;
        const auto* s = scales + size_t(row / ScaleRows) * scale_stride;
        if constexpr (RowsPerWarp == 8) {
            // Same small-K specialization as FP4: at most 32 work units.
            #pragma unroll
            for (int g0 = 0; g0 < 32; g0 += LANES_PER_ROW) {
                const int g = inner_lane + g0;
                if (g >= work_units) continue;
                const float scale = Scale::decode(s[g * WORK_VALUES / ScaleCols]);
                sum = fmaf(dot16<Aligned>(w + g * WORK_VALUES, x + g * WORK_VALUES,
                    min(WORK_VALUES, K - g * WORK_VALUES)), scale, sum);
            }
        } else {
            #pragma unroll 2
            for (int g = inner_lane; g < work_units; g += LANES_PER_ROW) {
                const float scale = Scale::decode(s[g * WORK_VALUES / ScaleCols]);
                sum = fmaf(dot16<Aligned>(w + g * WORK_VALUES, x + g * WORK_VALUES,
                    min(WORK_VALUES, K - g * WORK_VALUES)), scale, sum);
            }
        }
    }
    #pragma unroll
    for (int offset = LANES_PER_ROW / 2; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (row < N && inner_lane == 0) output[size_t(token) * N + row] = __float2bfloat16(sum);
}

template<int RowsPerWarp, int ScaleRows, int ScaleCols, class Scale, bool Aligned>
void launch(GlmCtx* ctx, void* out, const void* input, const void* weight, const void* scales,
            int M, int N, int K, int groups) {
    constexpr int rows = WARPS_PER_BLOCK * RowsPerWarp;
    fp8_linear_kernel<RowsPerWarp, ScaleRows, ScaleCols, Scale, Aligned>
        <<<dim3((N + rows - 1) / rows, M), BLOCK_SIZE, 0, GLM_STREAM(ctx)>>>(
            (__nv_bfloat16*)out, (const __nv_bfloat16*)input, (const uint8_t*)weight,
            (const typename Scale::Storage*)scales, N, K, groups);
}

// FP4's small-N split-K strategy: one CTA per output row, distributing the
// reduction over four warps instead of leaving almost all SMs idle.
template<int ScaleRows, int ScaleCols, class Scale, bool Aligned>
__global__ void __launch_bounds__(128, 4) fp8_linear_splitk_kernel(
    __nv_bfloat16* __restrict__ output, const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ weight, const typename Scale::Storage* __restrict__ scales,
    int N, int K, int groups) {
    const int row = blockIdx.x, token = blockIdx.y;
    const auto* x = input + (size_t(token) * groups + row / (N / groups)) * K;
    const auto* w = weight + size_t(row) * K;
    const auto* s = scales + size_t(row / ScaleRows) * ((K + ScaleCols - 1) / ScaleCols);
    float sum = 0.f;
    for (int g = threadIdx.x; g < (K + 15) / 16; g += 128) {
        sum = fmaf(dot16<Aligned>(w + g * 16, x + g * 16, min(16, K - g * 16)),
            Scale::decode(s[g * 16 / ScaleCols]), sum);
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float partial[4];
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 4 ? partial[lane] : 0.f;
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
        if (lane == 0) output[size_t(token) * N + row] = __float2bfloat16(sum);
    }
}

template<int ScaleRows, int ScaleCols, class Scale>
void dispatch(GlmCtx* ctx, void* out, const void* input, const void* weight, const void* scales,
              int M, int N, int K, int groups) {
    if (M * N <= 4096 && N <= 1024 && K >= 1024) {
        if (K % 16 == 0) fp8_linear_splitk_kernel<ScaleRows, ScaleCols, Scale, true>
            <<<dim3(N, M), 128, 0, GLM_STREAM(ctx)>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)input,
                (const uint8_t*)weight, (const typename Scale::Storage*)scales, N, K, groups);
        else fp8_linear_splitk_kernel<ScaleRows, ScaleCols, Scale, false>
            <<<dim3(N, M), 128, 0, GLM_STREAM(ctx)>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)input,
                (const uint8_t*)weight, (const typename Scale::Storage*)scales, N, K, groups);
    } else if (K <= 512) {
        if (K % 16 == 0) launch<8, ScaleRows, ScaleCols, Scale, true>(ctx, out, input, weight, scales, M, N, K, groups);
        else launch<8, ScaleRows, ScaleCols, Scale, false>(ctx, out, input, weight, scales, M, N, K, groups);
    } else {
        if (K % 16 == 0) launch<2, ScaleRows, ScaleCols, Scale, true>(ctx, out, input, weight, scales, M, N, K, groups);
        else launch<2, ScaleRows, ScaleCols, Scale, false>(ctx, out, input, weight, scales, M, N, K, groups);
    }
}
} // namespace

// scale_kind: 0 = BF16 (legacy 128x128), 1 = E8M0 (DeepSeek 32x32).
// Grouped projections concatenate groups of output rows and input features;
// weights remain [N, K], input is [M, groups*K], output is [M, N].
extern "C" int glm_fp8_block_linear(GlmCtx* ctx, void* out, const void* input,
    const void* weight, const void* scales, int M, int N, int K, int block_size, int scale_kind, int groups) {
    if (M <= 0 || M > 65535 || N <= 0 || K <= 0 || groups <= 0 || N % groups ||
        !((block_size == 128 && scale_kind == 0) || (block_size == 32 && scale_kind == 1)))
        return cudaErrorInvalidValue;
    cudaSetDevice(ctx->device_id);
    if (scale_kind == 0) dispatch<128, 128, Bf16Scale>(ctx, out, input, weight, scales, M, N, K, groups);
    else dispatch<32, 32, E8m0Scale>(ctx, out, input, weight, scales, M, N, K, groups);
    return cudaGetLastError();
}

extern "C" void glm_fp8_linear_decode(GlmCtx* ctx, void* out, const void* input,
    const void* weight, const void* scales, int M, int N, int K) {
    glm_fp8_block_linear(ctx, out, input, weight, scales, M, N, K, 128, 0, 1);
}
