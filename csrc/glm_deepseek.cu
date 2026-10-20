#include "glm_ops.h"
#include "glm_nvfp4.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
// [uncompressed, compressed, compressed ratio-2] x tokens x 32 (cos, sin) pairs.
// Positions remain device inputs, so captured forwards regenerate the tables.
__global__ void rotary_embedding_device(float* out, const int32_t* position, int tokens) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 3 * tokens * 32) return;
    const int variant = index / (tokens * 32), token = (index / 32) % tokens;
    const int rotary_pair = index % 32;
    const bool compressed = variant != 0;
    const double theta = compressed ? 160000. : 10000.;
    const double pi = 3.14159265358979323846;
    const int low = max(0, int(floor(64. * log(65536. / (32. * 2. * pi)) / (2. * log(theta)))));
    const int high = min(63, int(ceil(64. * log(65536. / (2. * pi)) / (2. * log(theta)))));
    const double ramp = fmin(1., fmax(0., (rotary_pair - low) / (high == low ? .001 : double(high - low))));
    const float freq = float(pow(theta, -rotary_pair / 32.) * (compressed ? 1. - ramp + ramp / 16. : 1.));
    const int token_position = position[token];
    const int p = variant == 2 ? ((token_position + 1) / 2 - 1) * 2 : token_position;
    const float angle = p * freq;
    out[index * 2] = float(cos(double(angle)));
    out[index * 2 + 1] = float(sin(double(angle)));
}

__global__ void rope_device(__nv_bfloat16* out, const __nv_bfloat16* in,
                            const float* rotary, int heads, int dim,
                            bool inverse, int heads_per_token) {
    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_index >= heads * dim / 2) return;
    const int head = pair_index / (dim / 2), pair = pair_index % (dim / 2);
    const int i = head * dim + pair * 2;
    if (pair < (dim - 64) / 2) { out[i] = in[i]; out[i + 1] = in[i + 1]; return; }
    const int offset = (head / heads_per_token) * 64 + (pair - (dim - 64) / 2) * 2;
    const double c = rotary[offset], s = double(rotary[offset + 1]) * (inverse ? -1. : 1.);
    const double a = float(in[i]), b = float(in[i + 1]);
    out[i] = __float2bfloat16(float(a * c - b * s));
    out[i + 1] = __float2bfloat16(float(a * s + b * c));
}

__global__ void swiglu_device(__nv_bfloat16* out, const __nv_bfloat16* gate,
                               const __nv_bfloat16* up, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double g = fmin(double(float(gate[i])), 10.);
    out[i] = __float2bfloat16(float(g / (1. + __expf(float(-g))) * fmin(10., fmax(-10., double(float(up[i]))))));
}

// GLM-style register-resident scores and warp-local top-k. Each warp owns
// one token in the throughput path. For small batches, ComputeThreads spreads
// sqrt-softplus across a block before handing selection to its first warp.
// Scores, biased comparisons, normalization and output weights remain FP32.
template<int Experts, int TopK, int Warps, int ComputeThreads = 0>
__global__ void route_warp_device(int32_t* ids, float* weights, const float* logits,
                                  const float* bias, int tokens) {
    const int row = ComputeThreads ? blockIdx.x : blockIdx.x * Warps + threadIdx.x / 32;
    if (row >= tokens) return;
    __shared__ float transformed[ComputeThreads ? Experts : 1];
    if constexpr (ComputeThreads > 0) {
        for (int e = threadIdx.x; e < Experts; e += ComputeThreads) {
            const float x = logits[size_t(row) * Experts + e];
            transformed[e] = sqrtf(fmaxf(x, 0.f) + log1pf(__expf(-fabsf(x))));
        }
        __syncthreads();
        if (threadIdx.x >= 32) return;
    }
    const int lane = threadIdx.x % 32;
    float scores[Experts / 32], original[Experts / 32];
    #pragma unroll
    for (int j = 0; j < Experts / 32; j++) {
        const int expert = lane + j * 32;
        if constexpr (ComputeThreads > 0) original[j] = transformed[expert];
        else {
            const float x = logits[size_t(row) * Experts + expert];
            original[j] = sqrtf(fmaxf(x, 0.f) + log1pf(__expf(-fabsf(x))));
        }
        scores[j] = original[j] + bias[expert];
    }
    float selected = 0.f;
    #pragma unroll
    for (int k = 0; k < TopK; k++) {
        float best = -INFINITY;
        int index = Experts;
        #pragma unroll
        for (int j = 0; j < Experts / 32; j++) {
            const int candidate = lane + j * 32;
            if (scores[j] > best || (scores[j] == best && candidate < index)) {
                best = scores[j]; index = candidate;
            }
        }
        #pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) {
            const float other = __shfl_down_sync(0xffffffff, best, delta);
            const int other_id = __shfl_down_sync(0xffffffff, index, delta);
            if (other > best || (other == best && other_id < index)) {
                best = other; index = other_id;
            }
        }
        const int winner = __shfl_sync(0xffffffff, index, 0);
        float value = 0.f;
        #pragma unroll
        for (int j = 0; j < Experts / 32; j++) {
            if (lane + j * 32 == winner) { value = original[j]; scores[j] = -INFINITY; }
        }
        value = __shfl_sync(0xffffffff, value, winner % 32);
        if (lane == k) { ids[size_t(row) * TopK + k] = winner; selected = value; }
    }
    float sum = selected;
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    sum = __shfl_sync(0xffffffff, sum, 0);
    if (lane < TopK) weights[size_t(row) * TopK + lane] = selected * (1.5f / (sum > 0.f ? sum : 1.f));
}

__global__ void collapse_device(__nv_bfloat16* out, const __nv_bfloat16* hidden,
                                const float* pre, int dim) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    out += blockIdx.y * dim;
    hidden += blockIdx.y * 4 * dim;
    pre += blockIdx.y * 4;
    float sum = 0.f;
    for (int s = 0; s < 4; s++) sum += float(hidden[s * dim + d]) * pre[s];
    out[d] = __float2bfloat16(sum);
}

__global__ void cache_device(uint8_t* cache, const __nv_bfloat16* input, const int32_t* position, int ratio, const int32_t* slots) {
    const int row = blockIdx.y;
    input += row * 512;
    position += row;
    if (ratio && (*position + 1) % ratio) return;
    const int slot = slots[row];
    if (slot < 0) return;
    uint8_t* page = cache + size_t(slot / 64) * 64 * 584;
    uint8_t* data = page + (slot % 64) * 576;
    uint8_t* scales = page + 64 * 576 + (slot % 64) * 8;
    const int block = blockIdx.x, d = threadIdx.x;
    if (block == 7) {
        reinterpret_cast<__nv_bfloat16*>(data + 448)[d] = input[448 + d];
        if (!d) scales[7] = 0;
        return;
    }
    __shared__ float maxima[64];
    maxima[d] = fmaxf(1e-4f, fabsf(float(input[block * 64 + d])));
    __syncthreads();
    for (int s = 32; s; s >>= 1) { if (d < s) maxima[d] = fmaxf(maxima[d], maxima[d + s]); __syncthreads(); }
    const int exponent = int(ceil(log2(double(maxima[0]) / 448.)));
    if (!d) scales[block] = exponent + 127;
    data[block * 64 + d] = __nv_fp8_e4m3(ldexpf(float(input[block * 64 + d]), -exponent)).__x;
}

__global__ void embedding_device(__nv_bfloat16* out, const __nv_bfloat16* table,
                                  const int32_t* token, int start, int rows, int tokens, int dim, int streams) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * streams * dim) return;
    const int row = token[i / (streams * dim)] - start;
    out[i] = row >= 0 && row < rows ? table[size_t(row) * dim + i % dim] : __float2bfloat16(0.f);
}

__global__ void hash_device(int32_t* out, const int32_t* lookback, const int32_t* token,
    const int32_t* map, const int32_t* primes, const int32_t* offsets, int tokens,
    const int32_t* qo_indptr, int batch_size) {
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= tokens) return;
    int batch = 0, start = 0;
    if (qo_indptr) {
        // Upper-bound search skips empty sequences, including repeated offsets.
        int end = batch_size;
        while (batch < end) {
            const int mid = (batch + end) / 2;
            if (qo_indptr[mid + 1] <= token_idx) batch = mid + 1;
            else end = mid;
        }
        start = qo_indptr[batch];
        lookback += batch * 3;
    }
    constexpr uint64_t multipliers[2][4] = {{76632096046245ULL, 4839876093313ULL, 35959672319349ULL, 73987337458391ULL},
        {67716810739261ULL, 51510806800915ULL, 30921347202721ULL, 82619226485591ULL}};
    const int p = token_idx - start;
    for (int layer = 0; layer < 2; layer++) {
        uint64_t rolling = 0;
        for (int shift = 0; shift < 4; shift++) {
            const int id = p >= shift ? token[token_idx - shift] : lookback[3 + p - shift];
            rolling ^= uint64_t(map[id]) * multipliers[layer][shift];
            if (shift) for (int h = 0; h < 8; h++) {
                const int i = layer * 24 + (shift - 1) * 8 + h;
                out[(layer * tokens + token_idx) * 24 + (shift - 1) * 8 + h] = rolling % primes[i] + offsets[i];
            }
        }
    }
}

__global__ void index_quant_device(uint8_t* out, const __nv_bfloat16* input, const int32_t* position,
                                    int heads, int ratio, const int32_t* slots = nullptr) {
    input += blockIdx.y * heads * 128;
    position += blockIdx.y;
    if (!ratio) out += size_t(blockIdx.y) * heads * 132;
    if (ratio && (*position + 1) % ratio) return;
    const int h = blockIdx.x, d = threadIdx.x;
    __shared__ float maxima[128];
    const float value = float(input[h * 128 + d]);
    maxima[d] = fmaxf(1e-4f, fabsf(value));
    __syncthreads();
    for (int s = 64; s; s >>= 1) { if (d < s) maxima[d] = fmaxf(maxima[d], maxima[d+s]); __syncthreads(); }
    const int exponent = int(ceil(log2(double(maxima[0]) / 448.)));
    const uint8_t quantized = __nv_fp8_e4m3(ldexpf(value, -exponent)).__x;
    const int slot = slots ? slots[blockIdx.y] : 0;
    if (slot < 0) return;
    uint8_t* record = out + (size_t(slot) + h) * 132;
    record[d] = quantized;
    if (d == 0) *reinterpret_cast<float*>(record + 128) = ldexpf(1.f, exponent);
}

// Logical pages always span 64 original tokens. Ratio-2 uses the first 32
// records of each physical page, preserving ordinary page sharing and COW.
__global__ void paged_slots_device(int32_t* out, const int32_t* position, const int32_t* pages,
                                    const int32_t* selected, int width, int ratio, int mode, const int32_t* batch_indices, int page_stride) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (col >= width) return;
    if (batch_indices) pages += size_t(batch_indices[row]) * page_stride;
    const int p = position[row];
    int ordinal;
    if (mode == 0) ordinal = (p + 1) % ratio ? -1 : (p + 1) / ratio - 1;
    else if (mode == 1) { const int key = max(0, p - 127) + col; ordinal = key <= p ? key : -1; }
    else { ordinal = selected[row * width + col]; if (ordinal >= (p + 1) / ratio) ordinal = -1; }
    out[row * width + col] = ordinal < 0 ? -1 : pages[ordinal * ratio / 64] * 64 + ordinal % (64 / ratio);
}

// Masks address logical compressed positions, independently of future CP storage.
// Prepare every slot on replay, including padding, without modifying shared scores.
template <bool BLOCK_MAX>
__global__ void dsv41_selection_prepare(__nv_bfloat16* prepared, int32_t* lengths,
    const __nv_bfloat16* scores, const int32_t* candidates, const int32_t* positions,
    int capacity, int ratio) {
    const int row = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = min((positions[row] + 1) / ratio, capacity);
    const int blocks = (capacity + 7) / 8;
    const int stride = ((BLOCK_MAX ? blocks : capacity) + 7) / 8 * 8;
    if (i == 0) {
        // Reserve the latest block; radix selection chooses up to 2047 others.
        lengths[row] = BLOCK_MAX ? max(0, (count + 7) / 8 - 1) : count;
    }
    if (i >= stride) {
        return;
    }
    float value = -INFINITY;
    if constexpr (BLOCK_MAX) {
        if (i < (count + 7) / 8 - 1) {
            for (int j = i * 8; j < min(count, i * 8 + 8); ++j) {
                value = fmaxf(value, float(scores[size_t(row) * capacity + j]));
            }
        }
    } else if (i < count && (!candidates || candidates[size_t(row) * blocks + i / 8])) {
        value = float(scores[size_t(row) * capacity + i]);
    }
    // Equal signed zeros must have the same radix key.
    prepared[size_t(row) * stride + i] = __float2bfloat16(value == 0.f ? 0.f : value);
}

__global__ void dsv41_candidate_mask(int32_t* mask, const int32_t* selected,
    const int32_t* positions, int capacity) {
    const int row = blockIdx.x;
    const int blocks = (capacity + 7) / 8;
    for (int i = threadIdx.x; i < 2047; i += blockDim.x) {
        const int index = selected[size_t(row) * 2047 + i];
        if (index >= 0) {
            mask[size_t(row) * blocks + index] = 1;
        }
    }
    const int count = min(positions[row] + 1, capacity);
    if (threadIdx.x == 0 && count > 0) {
        mask[size_t(row) * blocks + (count - 1) / 8] = 1;
    }
}

// Caller-owned scratch: hist[tokens,1088], meta[tokens,4], lengths[tokens],
// indices[tokens,k], selected_scores[tokens,k], then 16-byte-aligned prepared
// scores[tokens,stride]. Prefill vector loads require aligned rows and base.
// Keep the byte calculation paired with GlmOps.dsv41SelectionScratchBytes.
struct Dsv41SelectionScratch {
    int32_t *hist, *meta, *lengths, *indices;
    __nv_bfloat16 *selected_scores, *prepared;
    Dsv41SelectionScratch(void* buffer, int tokens, int k) {
        hist = static_cast<int32_t*>(buffer);
        meta = hist + size_t(tokens) * 1088;
        lengths = meta + size_t(tokens) * 4;
        indices = lengths + tokens;
        selected_scores = reinterpret_cast<__nv_bfloat16*>(indices + size_t(tokens) * k);
        const uintptr_t prepared_base = reinterpret_cast<uintptr_t>(selected_scores + size_t(tokens) * k);
        prepared = reinterpret_cast<__nv_bfloat16*>((prepared_base + 15) & ~uintptr_t(15));
    }
    void select(GlmCtx* ctx, int32_t* out, int tokens, int stride, int k) {
        if (tokens <= 64) {
            glm_topk_from_scores(ctx, out, selected_scores, prepared, lengths,
                hist, meta, tokens, stride, k, 0, 1, 0);
        } else {
            glm_topk_from_scores_prefill(ctx, out, selected_scores, prepared, lengths,
                hist, hist + size_t(tokens) * 1024, meta, tokens, stride, k, 1, 0, true);
        }
    }
};

__global__ void to_bf16_device(__nv_bfloat16* out, const float* input, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2bfloat16(input[i]);
}

__device__ float exponent_scale(uint8_t exponent) {
    return exponent == 255 ? nanf("") : ldexpf(1.f, int(exponent) - 127);
}

__global__ void engram_lookup(__nv_bfloat16* out, const int32_t* ids, const uint8_t* table,
                              const uint8_t* scales, int count, int row_start, int row_end) {
    const int item = blockIdx.x, d = threadIdx.x;
    if (item >= count) return;
    const int id = ids[item];
    float value = 0.f;
    if (id >= row_start && id < row_end) {
        const size_t row = id - row_start;
        value = fp8_e4m3_to_float(table[row * 256 + d]) * exponent_scale(scales[row * 8 + d / 32]);
    }
    out[size_t(item) * 256 + d] = __float2bfloat16(value);
}

__global__ void engram_gate(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* kv,
                            const __nv_bfloat16* qw, const __nv_bfloat16* kw, int dim, float eps) {
    const int token = blockIdx.x / 4, stream = blockIdx.x % 4, tid = threadIdx.x;
    const size_t offset = size_t(blockIdx.x) * dim;
    const size_t key_offset = (size_t(token) * 5 + stream) * dim;
    float xx = 0.f, kk = 0.f, dot = 0.f;
    for (int d = tid; d < dim; d += 256) {
        const float h = float(x[offset + d]), k = float(kv[key_offset + d]);
        xx += h * h;
        kk += k * k;
        dot += h * float(qw[stream * dim + d]) * float(kw[stream * dim + d]) * k;
    }
    __shared__ float sums[3][256];
    sums[0][tid] = xx; sums[1][tid] = kk; sums[2][tid] = dot;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (tid < step) for (int i = 0; i < 3; i++) sums[i][tid] += sums[i][tid + step];
        __syncthreads();
    }
    const float score = sums[2][0] * rsqrtf(sums[0][0] / dim + eps)
        * rsqrtf(sums[1][0] / dim + eps) * rsqrtf(float(dim));
    const float gate_input = sqrtf(fmaxf(fabsf(score), 1e-6f)) * (score < 0.f ? -1.f : 1.f);
    const float gate = 1.f / (1.f + __expf(-gate_input));
    for (int d = tid; d < dim; d += 256) {
        out[offset + d] = __float2bfloat16(float(x[offset + d]) + gate * float(kv[(size_t(token) * 5 + 4) * dim + d]));
    }
}

__global__ void compress_latent(__nv_bfloat16* out, int32_t* valid, const float* state,
    const int32_t* state_position, float* next_state, int32_t* next_position, const float* raw, const int32_t* positions,
    const int32_t* cu, const __nv_bfloat16* weight, int ratio, float eps) {
    const int sequence = blockIdx.x, d = threadIdx.x;
    __shared__ float squares[512];
    int saved_position = state_position ? state_position[sequence] : -1;
    float saved_value = saved_position >= 0 ? state[size_t(sequence) * 1024 + d] : 0.f;
    float saved_gate = saved_position >= 0 ? state[size_t(sequence) * 1024 + 512 + d] : 0.f;
    for (int token = cu[sequence]; token < cu[sequence + 1]; token++) {
        const int position = positions[token];
        const float value = raw[size_t(token) * (512 * ratio) + d];
        float pooled = value;
        bool emit = ratio == 1;
        if (ratio == 2) {
            const float gate = raw[size_t(token) * 1024 + 512 + d];
            if ((position & 1) == 0) {
                saved_value = value;
                saved_gate = gate;
                saved_position = position;
            } else {
                emit = saved_position == position - 1;
                const float peak = fmaxf(saved_gate, gate);
                const float a = __expf(saved_gate - peak), b = __expf(gate - peak);
                pooled = saved_value * (a / (a + b)) + value * (b / (a + b));
            }
        }
        if (d == 0) valid[token] = emit ? 1 : (ratio == 2 && (position & 1) ? -1 : 0);
        squares[d] = emit ? pooled * pooled : 0.f;
        __syncthreads();
        for (int step = 256; step; step >>= 1) {
            if (d < step) squares[d] += squares[d + step];
            __syncthreads();
        }
        out[size_t(token) * 512 + d] = emit
            ? __float2bfloat16(pooled * rsqrtf(squares[0] / 512 + eps) * float(weight[d]))
            : __float2bfloat16(0.f);
        __syncthreads();
    }
    if (ratio == 2) {
        next_state[size_t(sequence) * 1024 + d] = saved_value;
        next_state[size_t(sequence) * 1024 + 512 + d] = saved_gate;
        if (d == 0) next_position[sequence] = saved_position;
    }
}
}

void glm_dsv41_rotary_embedding(GlmCtx* ctx, float* out, const int32_t* positions, int tokens) {
    cudaSetDevice(ctx->device_id);
    rotary_embedding_device<<<(3*tokens*32+255)/256,256,0,GLM_STREAM(ctx)>>>(out,positions,tokens);
}
void glm_dsv41_rope(GlmCtx* ctx, void* out, const void* input, const float* rotary,
    int rows, int dim, int compressed, int inverse, int ratio, int heads) {
    cudaSetDevice(ctx->device_id);
    rope_device<<<(rows*dim/2+255)/256,256,0,GLM_STREAM(ctx)>>>((__nv_bfloat16*)out,
        (const __nv_bfloat16*)input,rotary + (compressed ? (ratio == 2 ? 2 : 1) : 0) * (rows / heads) * 64,
        rows,dim,inverse,heads);
}
void glm_dsv41_swiglu(GlmCtx* ctx, void* out, const void* gate, const void* up, int count) {
    cudaSetDevice(ctx->device_id);
    swiglu_device<<<(count+255)/256,256,0,GLM_STREAM(ctx)>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)gate,(const __nv_bfloat16*)up,count);
}
void glm_dsv41_route(GlmCtx* ctx, int32_t* ids, float* weights, const float* logits, const float* bias, int tokens, int experts) {
    cudaSetDevice(ctx->device_id);
    // Small batches favor parallel score evaluation; large batches favor four
    // register-resident rows per CTA. Both paths use the same warp reduction.
    if (experts == 128) {
        if (tokens <= 2048) route_warp_device<128, 3, 1, 128><<<tokens,128,0,GLM_STREAM(ctx)>>>(ids,weights,logits,bias,tokens);
        else route_warp_device<128, 3, 4><<<(tokens+3)/4,128,0,GLM_STREAM(ctx)>>>(ids,weights,logits,bias,tokens);
    } else {
        if (tokens <= 512) route_warp_device<384, 6, 1, 384><<<tokens,384,0,GLM_STREAM(ctx)>>>(ids,weights,logits,bias,tokens);
        else if (tokens <= 1024) route_warp_device<384, 6, 1, 128><<<tokens,128,0,GLM_STREAM(ctx)>>>(ids,weights,logits,bias,tokens);
        else route_warp_device<384, 6, 4><<<(tokens+3)/4,128,0,GLM_STREAM(ctx)>>>(ids,weights,logits,bias,tokens);
    }
}
void glm_dsv41_collapse(GlmCtx* ctx, void* out, const void* hidden, const float* pre, int dim, int tokens) {
    cudaSetDevice(ctx->device_id);
    collapse_device<<<dim3((dim+255)/256,tokens),256,0,GLM_STREAM(ctx)>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)hidden,pre,dim);
}
void glm_dsv41_cache_insert(GlmCtx* ctx, uint8_t* cache, const void* input, const int32_t* positions,
    int ratio, int tokens, const int32_t* slots) {
    cudaSetDevice(ctx->device_id);
    cache_device<<<dim3(8,tokens),64,0,GLM_STREAM(ctx)>>>(cache,(const __nv_bfloat16*)input,positions,ratio,slots);
}
void glm_dsv41_embedding(GlmCtx* ctx, void* out, const void* weight, const int32_t* tokens, int start, int rows, int count, int dim, int streams) {
    cudaSetDevice(ctx->device_id);
    embedding_device<<<(count*dim*streams+255)/256,256,0,GLM_STREAM(ctx)>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)weight,tokens,start,rows,count,dim,streams);
}
void glm_dsv41_hash(GlmCtx* ctx, int32_t* out, const int32_t* lookback, const int32_t* tokens,
    const int32_t* map, const int32_t* primes, const int32_t* offsets, int count,
    const int32_t* qo_indptr, int batch_size) {
    cudaSetDevice(ctx->device_id);
    if (count) hash_device<<<(count+127)/128,128,0,GLM_STREAM(ctx)>>>(out,lookback,tokens,map,primes,offsets,count,qo_indptr,batch_size);
}
void glm_dsv41_index_quantize(GlmCtx* ctx, uint8_t* out, const void* input, const int32_t* positions,
    int heads, int ratio, int tokens, const int32_t* slots) {
    cudaSetDevice(ctx->device_id);
    index_quant_device<<<dim3(heads,tokens),128,0,GLM_STREAM(ctx)>>>(out,(const __nv_bfloat16*)input,positions,heads,ratio,slots);
}
void glm_dsv41_candidates(GlmCtx* ctx, int32_t* out, const void* scores, const int32_t* positions, int capacity, int tokens, void* scratch) {
    cudaSetDevice(ctx->device_id);
    const int blocks = (capacity + 7) / 8;
    Dsv41SelectionScratch s(scratch, tokens, 2047);
    dsv41_selection_prepare<true><<<dim3((blocks + 255) / 256, tokens),256,0,GLM_STREAM(ctx)>>>(
        s.prepared, s.lengths, (const __nv_bfloat16*)scores, nullptr, positions, capacity, 1);
    s.select(ctx, s.indices, tokens, (blocks + 7) / 8 * 8, 2047);
    cudaMemsetAsync(out, 0, size_t(tokens) * blocks * sizeof(int32_t), GLM_STREAM(ctx));
    dsv41_candidate_mask<<<tokens,256,0,GLM_STREAM(ctx)>>>(out, s.indices, positions, capacity);
}
void glm_dsv41_select(GlmCtx* ctx, int32_t* out, const void* scores, const int32_t* candidates,
    const int32_t* positions, int capacity, int ratio, int tokens, void* scratch) {
    cudaSetDevice(ctx->device_id);
    Dsv41SelectionScratch s(scratch, tokens, 512);
    dsv41_selection_prepare<false><<<dim3((capacity + 255) / 256, tokens),256,0,GLM_STREAM(ctx)>>>(
        s.prepared, s.lengths, (const __nv_bfloat16*)scores, candidates, positions, capacity, ratio);
    s.select(ctx, out, tokens, (capacity + 7) / 8 * 8, 512);
}
void glm_dsv41_to_bf16(GlmCtx* ctx, void* out, const float* input, int count) {
    cudaSetDevice(ctx->device_id);
    to_bf16_device<<<(count+255)/256,256,0,GLM_STREAM(ctx)>>>((__nv_bfloat16*)out,input,count);
}
void glm_dsv4_paged_slots(GlmCtx* ctx, int32_t* out, const int32_t* positions, const int32_t* pages,
    const int32_t* selected, int width, int ratio, int tokens, int mode, const int32_t* batch_indices, int page_stride) {
    cudaSetDevice(ctx->device_id);
    paged_slots_device<<<dim3((width+127)/128,tokens),128,0,GLM_STREAM(ctx)>>>(out,positions,pages,selected,width,ratio,mode,batch_indices,page_stride);
}

extern "C" void glm_dsv41_engram_lookup(GlmCtx* ctx, void* out, const int32_t* ids,
    const uint8_t* table, const uint8_t* scales, int count, int row_start, int row_end) {
    cudaSetDevice(ctx->device_id);
    if (count > 0) engram_lookup<<<count, 256, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, ids, table, scales, count, row_start, row_end);
}

extern "C" void glm_dsv41_engram_gate(GlmCtx* ctx, void* out, const void* x, const void* kv,
    const void* qw, const void* kw, int tokens, int dim, float eps) {
    cudaSetDevice(ctx->device_id);
    if (tokens > 0) engram_gate<<<tokens * 4, 256, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x, (const __nv_bfloat16*)kv,
        (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw, dim, eps);
}

extern "C" void glm_dsv41_compress(GlmCtx* ctx, void* out, int32_t* valid, const float* state,
    const int32_t* state_position, float* next_state, int32_t* next_position, const float* raw, const int32_t* positions, const int32_t* cu,
    const void* weight, int batch, int ratio, float eps) {
    cudaSetDevice(ctx->device_id);
    if (batch > 0) compress_latent<<<batch, 512, 0, GLM_STREAM(ctx)>>>(
        (__nv_bfloat16*)out, valid, state, state_position, next_state, next_position, raw, positions, cu,
        (const __nv_bfloat16*)weight, ratio, eps);
}
