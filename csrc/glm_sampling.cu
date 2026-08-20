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
            heap[pos] = tmp;
            pos = smallest;
        } else break;
    }
}

__device__ void heap_insert(HeapEntry* heap, int& size, int capacity, float val, int idx) {
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
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return (float)seed / (float)0xFFFFFFFFu;
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

__device__ int sampling_softmax_topp_sample_and_append(
    int seq_idx,
    const SamplingParams& p,
    float* seq_topk_vals,
    int* seq_topk_vals_int,
    int* seq_topk_idxs,
    int* seq_out,
    int* penalty_tokens,
    int* penalty_count,
    int num_topk,
    unsigned int* step_counter,
    int max_window
) {
    float max_val = seq_topk_vals[0];
    float sum = 0.0f;
    for (int i = 0; i < num_topk; i++) {
        float v = __expf(seq_topk_vals[i] - max_val);
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

    unsigned int step = atomicAdd(step_counter, 1);
    float random_val = hash_to_random(step);

    float cumsum = 0.0f;
    int sampled = seq_topk_idxs[num_topk - 1];
    for (int i = 0; i < num_topk; i++) {
        cumsum += seq_topk_vals[i];
        if (random_val <= cumsum) { sampled = seq_topk_idxs[i]; break; }
    }
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
    unsigned int* __restrict__ step_counter
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

    float inv_temp = (p.temperature > 0.0f) ? (1.0f / p.temperature) : 1.0f;

    if (p.num_penalty_tokens > 0 && (p.repetition_penalty != 1.0f || p.presence_penalty != 0.0f)) {
        sampling_scale_logits(tid, block_size, seq_logits, seq_workspace, vocab_size, inv_temp);
        __syncthreads();
        sampling_apply_penalties(tid, p, penalty_tokens, seq_workspace, vocab_size, max_window);
        __syncthreads();
        for (int i = tid; i < vocab_size; i += block_size) {
            heap_insert(local_heap, local_size, p.effective_k, seq_workspace[i], i);
        }
    } else {
        for (int i = tid; i < vocab_size; i += block_size) {
            float val = __bfloat162float(seq_logits[i]) * inv_temp;
            heap_insert(local_heap, local_size, p.effective_k, val, i);
        }
    }

    for (int i = local_size; i < p.effective_k; i++) {
        local_heap[i].val = -FLT_MAX;
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

        sampling_softmax_topp_sample_and_append(seq_idx, p, seq_topk_vals, nullptr, seq_topk_idxs,
            seq_out, penalty_tokens, penalty_count, merge_size, step_counter, max_window);
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
    int max_effective_k
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

    float inv_temp = (p.temperature > 0.0f) ? (1.0f / p.temperature) : 1.0f;

    bool has_penalties = (p.num_penalty_tokens > 0) && (p.repetition_penalty != 1.0f || p.presence_penalty != 0.0f);
    if (has_penalties) {
        sampling_scale_logits(tid, blockDim.x, seq_logits, seq_workspace, vocab_size, inv_temp);
        __syncthreads();
        sampling_apply_penalties(tid, p, penalty_tokens, seq_workspace, vocab_size, max_window);
        __syncthreads();
    }

    for (int k = 0; k < p.effective_k; k++) {
        float my_max = -FLT_MAX;
        int my_idx = -1;
        for (int i = tid; i < vocab_size; i += blockDim.x) {
            float val = has_penalties ? seq_workspace[i] : (__bfloat162float(seq_logits[i]) * inv_temp);
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

        if (tid == 0) {
            seq_topk_vals[k] = s_vals[0];
            seq_topk_idxs[k] = s_idxs[0];
            seq_workspace[s_idxs[0]] = -FLT_MAX;
        }
        __syncthreads();
    }

    if (tid == 0) {
        sampling_softmax_topp_sample_and_append(seq_idx, p, seq_topk_vals, nullptr, seq_topk_idxs,
            seq_out, penalty_tokens, penalty_count, p.effective_k, step_counter, max_window);
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
                      int max_effective_k) {
    cudaSetDevice(ctx->device_id);

    int block_size = SAMPLING_BLOCK_SIZE;

    int num_warps = block_size / 32;

    if (max_effective_k <= 1) {
        size_t smem = num_warps * 2 * (sizeof(float) + sizeof(int));
        sampling_kernel_batch<2><<<batch_size, block_size, smem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter);
    } else if (max_effective_k <= 8) {
        size_t smem = num_warps * 8 * (sizeof(float) + sizeof(int));
        sampling_kernel_batch<8><<<batch_size, block_size, smem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter);
    } else if (max_effective_k <= 16) {
        size_t smem = num_warps * 16 * (sizeof(float) + sizeof(int));
        sampling_kernel_batch<16><<<batch_size, block_size, smem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter);
    } else if (max_effective_k <= 32) {
        size_t smem = num_warps * 32 * (sizeof(float) + sizeof(int));
        sampling_kernel_batch<32><<<batch_size, block_size, smem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter);
    } else if (max_effective_k <= 64) {
        size_t smem = num_warps * 64 * (sizeof(float) + sizeof(int));
        sampling_kernel_batch<64><<<batch_size, block_size, smem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter);
    } else {
        size_t shared_mem = block_size * (sizeof(float) + sizeof(int));
        sampling_kernel_argmax_batch<<<batch_size, block_size, shared_mem, GLM_STREAM(ctx)>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_count,
            max_window, vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, step_counter, max_effective_k);
    }
}
