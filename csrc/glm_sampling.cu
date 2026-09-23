#include "glm_ops.h"
#include <cuda_bf16.h>
#include <float.h>
#include <cmath>

#define SAMPLING_BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

struct HeapEntry {
    float val;
    int idx;
};

__device__ void heap_sift_down(HeapEntry* heap, int pos, int size) {
    while (true) {
        int left = 2 * pos + 1;
        int right = 2 * pos + 2;
        int smallest = pos;
        if (left < size && heap[left].val < heap[smallest].val) smallest = left;
        if (right < size && heap[right].val < heap[smallest].val) smallest = right;
        if (smallest != pos) {
            HeapEntry tmp = heap[pos];
            heap[pos] = heap[smallest];
            heap[smallest] = tmp;
            pos = smallest;
        } else break;
    }
}

__device__ void heap_insert(HeapEntry* heap, int& size, int capacity, float val, int idx) {
    if (!(val > -INFINITY)) return;
    if (size < capacity) {
        heap[size] = {val, idx};
        int pos = size;
        size++;
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            if (heap[pos].val < heap[parent].val) {
                HeapEntry tmp = heap[pos];
                heap[pos] = heap[parent];
                heap[parent] = tmp;
                pos = parent;
            } else break;
        }
    } else if (val > heap[0].val) {
        heap[0] = {val, idx};
        heap_sift_down(heap, 0, size);
    }
}

__device__ float hash_to_random(unsigned int seed) {
    // lowbias32 mixes consecutive counters; a single xorshift leaves draws correlated.
    seed ^= seed >> 16;
    seed *= 0x7feb352du;
    seed ^= seed >> 15;
    seed *= 0x846ca68bu;
    seed ^= seed >> 16;
    // Use 24 bits so float conversion cannot round the result up to 1.
    return (float)(seed >> 8) * 0x1p-24f;
}

// Stream ordering keeps the base stable for every row, including graph replay.
__global__ void sampling_advance_counter(unsigned int* counter, unsigned int count) {
    *counter += count;
}

__device__ void insertion_sort_descending(HeapEntry* arr, int size) {
    for (int i = 1; i < size; i++) {
        HeapEntry key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j].val < key.val) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

struct SamplingParams {
    float temperature;
    float repetition_penalty;
    float presence_penalty;
    int top_k;
    float top_p;
    int num_penalty_tokens;
    int pen_base;
    int pen_start;
    int effective_k;
};

__device__ SamplingParams sampling_load_params(
    int seq_idx,
    const float* temperatures,
    const float* repetition_penalties,
    const float* presence_penalties,
    const int* top_ks,
    const float* top_ps,
    int* penalty_count,
    int max_window,
    int vocab_size
) {
    SamplingParams p;
    p.temperature = temperatures[seq_idx];
    p.repetition_penalty = repetition_penalties[seq_idx];
    p.presence_penalty = presence_penalties[seq_idx];
    p.top_k = top_ks[seq_idx];
    p.top_p = top_ps[seq_idx];
    p.pen_base = seq_idx * max_window;
    p.num_penalty_tokens = 0;
    p.pen_start = 0;
    if (max_window > 0) {
        p.num_penalty_tokens = min(penalty_count[seq_idx], max_window);
        p.pen_start = (penalty_count[seq_idx] - p.num_penalty_tokens) % max_window;
    }
    if (p.temperature <= 0.0f) {
        p.effective_k = 1;
    } else if (p.top_k > 0) {
        p.effective_k = (p.top_k < vocab_size) ? p.top_k : vocab_size;
    } else {
        p.effective_k = 32;
    }
    return p;
}

__device__ void sampling_scale_logits(
    int tid, int block_size,
    const __nv_bfloat16* seq_logits,
    float* seq_workspace,
    int vocab_size,
    float inv_temp
) {
    // Odd vocabulary strides can leave later rows unaligned for paired accesses.
    if ((reinterpret_cast<uintptr_t>(seq_logits) % alignof(__nv_bfloat162)) == 0 &&
        (reinterpret_cast<uintptr_t>(seq_workspace) % alignof(float2)) == 0) {
        const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(seq_logits);
        auto* output = reinterpret_cast<float2*>(seq_workspace);
        for (int i = tid; i < vocab_size / 2; i += block_size) {
            float2 val = __bfloat1622float2(pairs[i]);
            output[i] = make_float2(val.x * inv_temp, val.y * inv_temp);
        }
        if ((vocab_size & 1) && tid == 0) {
            seq_workspace[vocab_size - 1] = __bfloat162float(seq_logits[vocab_size - 1]) * inv_temp;
        }
        return;
    }
    for (int i = tid; i < vocab_size; i += block_size) {
        float val = __bfloat162float(seq_logits[i]);
        seq_workspace[i] = val * inv_temp;
    }
}

__device__ void sampling_apply_penalties(
    int tid,
    const SamplingParams& p,
    int* penalty_tokens,
    float* seq_workspace,
    int vocab_size,
    int max_window
) {
    if (tid == 0 && p.num_penalty_tokens > 0) {
        for (int i = 0; i < p.num_penalty_tokens; i++) {
            int idx = penalty_tokens[p.pen_base + (p.pen_start + i) % max_window];
            if (idx < 0 || idx >= vocab_size) continue;
            bool seen = false;
            for (int j = 0; j < i; j++) {
                if (penalty_tokens[p.pen_base + (p.pen_start + j) % max_window] == idx) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;
            float val = seq_workspace[idx];
            if (p.repetition_penalty != 1.0f) {
                seq_workspace[idx] = (val > 0.0f) ? (val / p.repetition_penalty) : (val * p.repetition_penalty);
                val = seq_workspace[idx];
            }
            if (p.presence_penalty != 0.0f) {
                seq_workspace[idx] = val - p.presence_penalty;
            }
        }
    }
}

template<typename T>
__device__ int sampling_sparse_draw(const T* probs, const int* ids,
                                    int capacity, float uniform) {
    double total = 0.0;
    for (int i = 0; i < capacity && ids[i] >= 0; i++) total += probs[i];
    double threshold = (double)uniform * total;
    double cumulative = 0.0;
    int sampled = ids[0];
    for (int i = 0; i < capacity && ids[i] >= 0; i++) {
        if (probs[i] <= 0.0) continue;
        sampled = ids[i];
        cumulative += probs[i];
        if (threshold < cumulative) break;
    }
    return sampled;
}

__device__ int sampling_softmax_topp_sample_and_append(
    int seq_idx,
    const SamplingParams& p,
    float* seq_topk_vals,
    int* seq_topk_idxs,
    int* seq_out,
    int* penalty_tokens,
    int* penalty_count,
    int num_topk,
    unsigned int* step_counter,
    int max_window,
    float* out_probs, int* out_ids, int support_capacity
) {
    // Empty support (all NaN/-inf) has a deterministic, valid fallback.
    if (num_topk == 0) {
        num_topk = 1;
        seq_topk_vals[0] = 0.0f;
        seq_topk_idxs[0] = 0;
    }
    float max_val = seq_topk_vals[0];
    float sum = 0.0f;
    for (int i = 0; i < num_topk; i++) {
        float v = max_val == INFINITY ? (seq_topk_vals[i] == INFINITY ? 1.0f : 0.0f)
                                      : __expf(seq_topk_vals[i] - max_val);
        seq_topk_vals[i] = v;
        sum += v;
    }
    for (int i = 0; i < num_topk; i++) {
        seq_topk_vals[i] /= sum;
    }

    if (p.top_p < 1.0f && p.top_k <= 0) {
        float cumsum = 0.0f;
        int cutoff = num_topk;
        for (int i = 0; i < num_topk; i++) {
            cumsum += seq_topk_vals[i];
            if (cumsum > p.top_p) { cutoff = i + 1; break; }
        }
        float renorm = 0.0f;
        for (int i = 0; i < cutoff; i++) renorm += seq_topk_vals[i];
        for (int i = cutoff; i < num_topk; i++) seq_topk_vals[i] = 0.0f;
        if (renorm > 0.0f) {
            for (int i = 0; i < cutoff; i++) seq_topk_vals[i] /= renorm;
        }
    }

    for (int i = 0; i < support_capacity; i++) {
        size_t offset = (size_t)seq_idx * support_capacity + i;
        if (out_probs) out_probs[offset] = i < num_topk ? seq_topk_vals[i] : 0.0f;
        if (out_ids) out_ids[offset] = i < num_topk ? seq_topk_idxs[i] : -1;
    }

    unsigned int step = *step_counter + (unsigned int)seq_idx;
    float random_val = hash_to_random(step);

    int sampled = sampling_sparse_draw(seq_topk_vals, seq_topk_idxs, num_topk, random_val);
    seq_out[0] = sampled;

    if (max_window > 0) {
        int write_pos = penalty_count[seq_idx] % max_window;
        penalty_tokens[seq_idx * max_window + write_pos] = sampled;
        penalty_count[seq_idx]++;
    }

    return sampled;
}

// ---------------------------------------------------------------------------
// Batch sampling kernel: each block handles one sequence
// ---------------------------------------------------------------------------

template<int MAX_K>
__global__ void __launch_bounds__(SAMPLING_BLOCK_SIZE, 4) sampling_kernel_batch(
    int* __restrict__ out_tokens,
    float* __restrict__ topk_vals,
    int* __restrict__ topk_idxs,
    float* __restrict__ workspace,
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ penalty_tokens,
    int* __restrict__ penalty_count,
    int max_window,
    int vocab_size,
    const float* __restrict__ temperatures,
    const float* __restrict__ repetition_penalties,
    const float* __restrict__ presence_penalties,
    const int* __restrict__ top_ks,
    const float* __restrict__ top_ps,
    unsigned int* __restrict__ step_counter,
    float* out_probs, int* out_ids, int support_capacity
) {
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    SamplingParams p = sampling_load_params(seq_idx, temperatures, repetition_penalties,
        presence_penalties, top_ks, top_ps, penalty_count, max_window, vocab_size);

    const __nv_bfloat16* seq_logits = logits + (size_t)seq_idx * vocab_size;
    float* seq_workspace = workspace + (size_t)seq_idx * vocab_size;
    int* seq_out = out_tokens + seq_idx;
    float* seq_topk_vals = topk_vals + (size_t)seq_idx * MAX_K * block_size;
    int* seq_topk_idxs = topk_idxs + (size_t)seq_idx * MAX_K * block_size;

    HeapEntry local_heap[MAX_K];
    int local_size = 0;

    float inv_temp = (p.temperature > 0.0f) ? fminf(1.0f / p.temperature, FLT_MAX) : 1.0f;

    if (p.num_penalty_tokens > 0 && (p.repetition_penalty != 1.0f || p.presence_penalty != 0.0f)) {
        sampling_scale_logits(tid, block_size, seq_logits, seq_workspace, vocab_size, inv_temp);
        __syncthreads();
        sampling_apply_penalties(tid, p, penalty_tokens, seq_workspace, vocab_size, max_window);
        __syncthreads();
        for (int i = tid; i < vocab_size; i += block_size) {
            heap_insert(local_heap, local_size, p.effective_k, seq_workspace[i], i);
        }
    } else if ((reinterpret_cast<uintptr_t>(seq_logits) % alignof(__nv_bfloat162)) == 0) {
        const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(seq_logits);
        for (int i = tid; i < vocab_size / 2; i += block_size) {
            float2 val = __bfloat1622float2(pairs[i]);
            heap_insert(local_heap, local_size, p.effective_k, val.x * inv_temp, 2 * i);
            heap_insert(local_heap, local_size, p.effective_k, val.y * inv_temp, 2 * i + 1);
        }
        if ((vocab_size & 1) && tid == 0) {
            float val = __bfloat162float(seq_logits[vocab_size - 1]) * inv_temp;
            heap_insert(local_heap, local_size, p.effective_k, val, vocab_size - 1);
        }
    } else {
        for (int i = tid; i < vocab_size; i += block_size) {
            float val = __bfloat162float(seq_logits[i]) * inv_temp;
            heap_insert(local_heap, local_size, p.effective_k, val, i);
        }
    }

    for (int i = local_size; i < p.effective_k; i++) {
        local_heap[i].val = -INFINITY;
        local_heap[i].idx = -1;
    }
    local_size = p.effective_k;
    for (int i = (p.effective_k / 2) - 1; i >= 0; i--) {
        heap_sift_down(local_heap, i, p.effective_k);
    }

    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = block_size >> 5;

    for (int stride = 16; stride >= 1; stride >>= 1) {
        for (int i = 0; i < p.effective_k; i++) {
            float partner_val = __shfl_xor_sync(0xFFFFFFFF, local_heap[i].val, stride);
            int partner_idx = __shfl_xor_sync(0xFFFFFFFF, local_heap[i].idx, stride);
            if ((lane_id & stride) == 0 && partner_idx >= 0) {
                heap_insert(local_heap, local_size, p.effective_k, partner_val, partner_idx);
            }
        }
    }

    extern __shared__ char smem[];
    float* smem_warp_vals = reinterpret_cast<float*>(smem);
    int* smem_warp_idxs = reinterpret_cast<int*>(smem + num_warps * MAX_K * sizeof(float));

    if (lane_id == 0) {
        for (int i = 0; i < p.effective_k; i++) {
            smem_warp_vals[warp_id * MAX_K + i] = local_heap[i].val;
            smem_warp_idxs[warp_id * MAX_K + i] = local_heap[i].idx;
        }
    }
    __syncthreads();

    if (tid == 0) {
        HeapEntry merge_heap[MAX_K];
        int merge_size = 0;

        for (int w = 0; w < num_warps; w++) {
            for (int i = 0; i < p.effective_k; i++) {
                float val = smem_warp_vals[w * MAX_K + i];
                int idx = smem_warp_idxs[w * MAX_K + i];
                if (idx >= 0) {
                    heap_insert(merge_heap, merge_size, p.effective_k, val, idx);
                }
            }
        }

        insertion_sort_descending(merge_heap, merge_size);
        for (int i = 0; i < merge_size; i++) {
            seq_topk_vals[i] = merge_heap[i].val;
            seq_topk_idxs[i] = merge_heap[i].idx;
        }

        sampling_softmax_topp_sample_and_append(seq_idx, p, seq_topk_vals, seq_topk_idxs,
            seq_out, penalty_tokens, penalty_count, merge_size, step_counter, max_window,
            out_probs, out_ids, support_capacity);
    }
}

// ---------------------------------------------------------------------------
// Batch argmax kernel (fallback for large K)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(SAMPLING_BLOCK_SIZE, 4) sampling_kernel_argmax_batch(
    int* __restrict__ out_tokens,
    float* __restrict__ topk_vals,
    int* __restrict__ topk_idxs,
    float* __restrict__ workspace,
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ penalty_tokens,
    int* __restrict__ penalty_count,
    int max_window,
    int vocab_size,
    const float* __restrict__ temperatures,
    const float* __restrict__ repetition_penalties,
    const float* __restrict__ presence_penalties,
    const int* __restrict__ top_ks,
    const float* __restrict__ top_ps,
    unsigned int* __restrict__ step_counter,
    int max_effective_k,
    float* out_probs, int* out_ids, int support_capacity
) {
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;

    SamplingParams p = sampling_load_params(seq_idx, temperatures, repetition_penalties,
        presence_penalties, top_ks, top_ps, penalty_count, max_window, vocab_size);

    const __nv_bfloat16* seq_logits = logits + (size_t)seq_idx * vocab_size;
    float* seq_workspace = workspace + (size_t)seq_idx * vocab_size;
    int* seq_out = out_tokens + seq_idx;
    float* seq_topk_vals = topk_vals + (size_t)seq_idx * max_effective_k;
    int* seq_topk_idxs = topk_idxs + (size_t)seq_idx * max_effective_k;

    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int* s_idxs = reinterpret_cast<int*>(s_vals + blockDim.x);

    float inv_temp = (p.temperature > 0.0f) ? fminf(1.0f / p.temperature, FLT_MAX) : 1.0f;

    bool has_penalties = (p.num_penalty_tokens > 0) && (p.repetition_penalty != 1.0f || p.presence_penalty != 0.0f);
    sampling_scale_logits(tid, blockDim.x, seq_logits, seq_workspace, vocab_size, inv_temp);
    __syncthreads();
    if (has_penalties) {
        sampling_apply_penalties(tid, p, penalty_tokens, seq_workspace, vocab_size, max_window);
        __syncthreads();
    }

    int num_topk = 0;
    for (int k = 0; k < p.effective_k; k++) {
        float my_max = -INFINITY;
        int my_idx = -1;
        for (int i = tid; i < vocab_size; i += blockDim.x) {
            float val = seq_workspace[i];
            if (val > my_max) {
                my_max = val;
                my_idx = i;
            }
        }
        s_vals[tid] = my_max;
        s_idxs[tid] = my_idx;
        __syncthreads();

        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_vals[tid + s] > s_vals[tid]) {
                    s_vals[tid] = s_vals[tid + s];
                    s_idxs[tid] = s_idxs[tid + s];
                }
            }
            __syncthreads();
        }

        // All threads see the same reduced index, so the exit is block-uniform.
        if (s_idxs[0] < 0) break;
        num_topk++;
        if (tid == 0) {
            seq_topk_vals[k] = s_vals[0];
            seq_topk_idxs[k] = s_idxs[0];
            seq_workspace[s_idxs[0]] = -INFINITY;
        }
        __syncthreads();
    }

    if (tid == 0) {
        sampling_softmax_topp_sample_and_append(seq_idx, p, seq_topk_vals, seq_topk_idxs,
            seq_out, penalty_tokens, penalty_count, num_topk, step_counter, max_window,
            out_probs, out_ids, support_capacity);
    }
}

// ---------------------------------------------------------------------------
// Host function: dispatch to appropriate kernel based on max_effective_k
// ---------------------------------------------------------------------------

void glm_sample_batch(GlmCtx* ctx, int* out_tokens, float* topk_vals, int* topk_idxs,
                      float* workspace, const void* logits,
                      int* penalty_tokens, int* penalty_count,
                      int max_window, int vocab_size, int batch_size,
                      const float* temperatures, const float* repetition_penalties,
                      const float* presence_penalties, const int* top_ks,
                      const float* top_ps, unsigned int* step_counter,
                      int max_effective_k, float* out_probs, int* out_ids,
                      int support_capacity) {
    cudaSetDevice(ctx->device_id);
    if (batch_size <= 0) return;

    int block_size = SAMPLING_BLOCK_SIZE;

    int num_warps = block_size / 32;

    if (max_effective_k <= 1) {
        size_t smem = num_warps * 2 * (sizeof(float) + sizeof(int));
        sampling_kernel_batch<2><<<batch_size, block_size, smem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter, out_probs, out_ids, support_capacity);
    } else if (max_effective_k <= 8) {
        size_t smem = num_warps * 8 * (sizeof(float) + sizeof(int));
        sampling_kernel_batch<8><<<batch_size, block_size, smem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter, out_probs, out_ids, support_capacity);
    } else if (max_effective_k <= 16) {
        size_t smem = num_warps * 16 * (sizeof(float) + sizeof(int));
        sampling_kernel_batch<16><<<batch_size, block_size, smem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter, out_probs, out_ids, support_capacity);
    } else if (max_effective_k <= 32) {
        size_t smem = num_warps * 32 * (sizeof(float) + sizeof(int));
        sampling_kernel_batch<32><<<batch_size, block_size, smem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter, out_probs, out_ids, support_capacity);
    } else if (max_effective_k <= 64) {
        size_t smem = num_warps * 64 * (sizeof(float) + sizeof(int));
        sampling_kernel_batch<64><<<batch_size, block_size, smem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter, out_probs, out_ids, support_capacity);
    } else {
        size_t shared_mem = block_size * (sizeof(float) + sizeof(int));
        sampling_kernel_argmax_batch<<<batch_size, block_size, shared_mem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter, max_effective_k, out_probs, out_ids, support_capacity);
    }
    sampling_advance_counter<<<1, 1, 0, GLM_STREAM(ctx)>>>(step_counter, (unsigned int)batch_size);
}

__global__ void sampling_kernel_candidates(
    int* out_tokens, float* out_probs, int* out_ids,
    const __nv_bfloat16* candidate_values, const int* candidate_ids,
    const float* temperatures, const int* top_ks, const float* top_ps,
    unsigned int* step_counter, int candidate_count, int support_capacity) {
    __shared__ float vals[256];
    __shared__ int ids[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    SamplingParams p = {};
    p.temperature = temperatures[row];
    p.top_k = top_ks[row];
    p.top_p = top_ps[row];
    p.effective_k = p.temperature <= 0.0f ? 1
        : min(p.top_k > 0 ? p.top_k : 32, candidate_count);
    float inv_temp = p.temperature > 0.0f ? fminf(1.0f / p.temperature, FLT_MAX) : 1.0f;
    float val = -INFINITY;
    int id = -1;
    if (tid < candidate_count) {
        size_t offset = (size_t)row * candidate_count + tid;
        id = candidate_ids[offset];
        float raw = __bfloat162float(candidate_values[offset]);
        if (id >= 0 && raw > -INFINITY) val = raw * inv_temp;
        if (!(val > -INFINITY)) id = -1;
    }
    vals[tid] = id >= 0 ? val : -INFINITY;
    ids[tid] = id;
    __syncthreads();

    // Bitonic paired sort: scaled logits descending, then global token ID ascending.
    // Sorting after FP32 scaling also makes overflow-induced ties deterministic.
    for (int size = 2; size <= blockDim.x; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int other = tid ^ stride;
            if (other > tid) {
                bool before = vals[tid] > vals[other] ||
                    (vals[tid] == vals[other] && ids[tid] < ids[other]);
                bool after = vals[tid] < vals[other] ||
                    (vals[tid] == vals[other] && ids[tid] > ids[other]);
                if ((tid & size) == 0 ? after : before) {
                    float tmp_val = vals[tid];
                    int tmp_id = ids[tid];
                    vals[tid] = vals[other];
                    ids[tid] = ids[other];
                    vals[other] = tmp_val;
                    ids[other] = tmp_id;
                }
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        int count = 0;
        while (count < p.effective_k && ids[count] >= 0) count++;
        sampling_softmax_topp_sample_and_append(row, p, vals, ids,
            out_tokens + row, nullptr, nullptr, count, step_counter, 0,
            out_probs, out_ids, support_capacity);
    }
}

void glm_sample_candidates(GlmCtx* ctx, int* out_tokens, float* out_probs, int* out_ids,
                           const void* candidate_values, const int* candidate_ids,
                           const float* temperatures, const int* top_ks, const float* top_ps,
                           unsigned int* step_counter, int batch_size,
                           int candidate_count, int support_capacity) {
    if (!ctx || batch_size <= 0 || candidate_count < 1 || candidate_count > 256 ||
        support_capacity < candidate_count || support_capacity > 256 ||
        !out_tokens || !out_probs || !out_ids || !candidate_values || !candidate_ids ||
        !temperatures || !top_ks || !top_ps || !step_counter) return;
    cudaSetDevice(ctx->device_id);
    int block_size = 32;
    while (block_size < candidate_count) block_size <<= 1;
    sampling_kernel_candidates<<<batch_size, block_size, 0, GLM_STREAM(ctx)>>>(
        out_tokens, out_probs, out_ids, (const __nv_bfloat16*)candidate_values,
        candidate_ids, temperatures, top_ks, top_ps, step_counter,
        candidate_count, support_capacity);
    sampling_advance_counter<<<1, 1, 0, GLM_STREAM(ctx)>>>(step_counter, (unsigned int)batch_size);
}

__device__ float sampling_sparse_lookup(const float* probs, const int* ids,
                                        int capacity, int token) {
    for (int i = 0; i < capacity && ids[i] >= 0; i++) {
        if (ids[i] == token) return probs[i];
    }
    return 0.0f;
}

__global__ void spec_reject_linear_kernel(
    int* out_tokens, int* out_accepted, const int* draft_tokens,
    const float* q_probs, const int* q_ids, const float* p_probs, const int* p_ids,
    const unsigned int* step_counter, int depth, int capacity) {
    __shared__ double residual[256];
    if (threadIdx.x != 0) return;
    size_t seq = blockIdx.x;
    size_t p_base = seq * ((size_t)depth + 1) * capacity;
    size_t q_base = seq * (size_t)depth * capacity;
    int* out = out_tokens + seq * ((size_t)depth + 1);
    // D+1 target draws, D acceptance draws, one distinct correction draw.
    unsigned int stride = 2u * (unsigned int)depth + 2u;
    unsigned int base = *step_counter + (unsigned int)seq * stride;
    for (int d = 0; d <= depth; d++) {
        size_t offset = p_base + (size_t)d * capacity;
        out[d] = sampling_sparse_draw(p_probs + offset, p_ids + offset,
                                     capacity, hash_to_random(base + (unsigned int)d));
    }
    out_accepted[seq] = depth;
    for (int d = 0; d < depth; d++) {
        const float* p = p_probs + p_base + (size_t)d * capacity;
        const int* pi = p_ids + p_base + (size_t)d * capacity;
        const float* q = q_probs + q_base + (size_t)d * capacity;
        const int* qi = q_ids + q_base + (size_t)d * capacity;
        int token = draft_tokens[seq * (size_t)depth + d];
        double p_total = 0.0, q_total = 0.0;
        for (int i = 0; i < capacity && pi[i] >= 0; i++) p_total += p[i];
        for (int i = 0; i < capacity && qi[i] >= 0; i++) q_total += q[i];
        double px = sampling_sparse_lookup(p, pi, capacity, token);
        double qx = sampling_sparse_lookup(q, qi, capacity, token);
        float uniform = hash_to_random(base + (unsigned int)depth + 1u + (unsigned int)d);
        if (qx > 0.0 && p_total > 0.0 &&
            uniform < fmin(1.0, (px * q_total) / (qx * p_total))) {
            out[d] = token;
            continue;
        }
        out_accepted[seq] = d;
        double total = 0.0;
        for (int i = 0; i < capacity; i++) {
            if (pi[i] < 0) break;
            double pn = p_total > 0.0 ? (double)p[i] / p_total : 0.0;
            double qn = q_total > 0.0 ? (double)sampling_sparse_lookup(q, qi, capacity, pi[i]) / q_total : 0.0;
            residual[i] = fmax(0.0, pn - qn);
            total += residual[i];
        }
        float correction = hash_to_random(base + stride - 1u);
        // A zero residual is only a numerical/invalid-proposal corner case.
        out[d] = total > 0.0 ? sampling_sparse_draw(residual, pi, capacity, correction)
                             : sampling_sparse_draw(p, pi, capacity, correction);
        break;
    }
}

void glm_spec_reject_linear(GlmCtx* ctx, int* out_tokens, int* out_accepted,
                           const int* draft_tokens, const float* q_probs, const int* q_ids,
                           const float* p_probs, const int* p_ids, unsigned int* step_counter,
                           int batch_size, int depth, int capacity) {
    cudaSetDevice(ctx->device_id);
    if (batch_size <= 0) return;
    spec_reject_linear_kernel<<<batch_size, 32, 0, GLM_STREAM(ctx)>>>(
        out_tokens, out_accepted, draft_tokens, q_probs, q_ids, p_probs, p_ids,
        step_counter, depth, capacity);
    sampling_advance_counter<<<1, 1, 0, GLM_STREAM(ctx)>>>(
        step_counter, (unsigned int)batch_size * (2u * (unsigned int)depth + 2u));
}
