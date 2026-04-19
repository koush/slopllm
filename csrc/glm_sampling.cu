#include "glm_ops.h"
#include <cuda_bf16.h>
#include <float.h>
#include <cmath>

#define SAMPLING_BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Device helpers: min-heap for top-K selection
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
    const int* __restrict__ penalty_tokens,
    const int* __restrict__ penalty_offsets,
    int vocab_size,
    const float* __restrict__ temperatures,
    const float* __restrict__ repetition_penalties,
    const float* __restrict__ presence_penalties,
    const int* __restrict__ top_ks,
    const float* __restrict__ top_ps,
    const float* __restrict__ random_vals
) {
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float temperature = temperatures[seq_idx];
    float repetition_penalty = repetition_penalties[seq_idx];
    float presence_penalty = presence_penalties[seq_idx];
    int top_k = top_ks[seq_idx];
    float top_p = top_ps[seq_idx];
    float random_val = random_vals[seq_idx];

    int pen_start = penalty_offsets[seq_idx];
    int pen_end = penalty_offsets[seq_idx + 1];
    int num_penalty_tokens = pen_end - pen_start;

    int effective_k;
    if (temperature <= 0.0f && top_k <= 0) {
        effective_k = 1;
    } else if (top_k > 0) {
        effective_k = (top_k < vocab_size) ? top_k : vocab_size;
    } else {
        effective_k = 64;
    }

    const __nv_bfloat16* seq_logits = logits + (size_t)seq_idx * vocab_size;
    float* seq_workspace = workspace + (size_t)seq_idx * vocab_size;
    int* seq_out = out_tokens + seq_idx;
    float* seq_topk_vals = topk_vals + (size_t)seq_idx * MAX_K * block_size;
    int* seq_topk_idxs = topk_idxs + (size_t)seq_idx * MAX_K * block_size;
    const int* seq_penalty_tokens = penalty_tokens + pen_start;

    HeapEntry local_heap[MAX_K];
    int local_size = 0;

    float inv_temp = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    for (int i = tid; i < vocab_size; i += block_size) {
        float val = __bfloat162float(seq_logits[i]);
        seq_workspace[i] = val * inv_temp;
    }
    __syncthreads();

    for (int i = tid; i < num_penalty_tokens; i += block_size) {
        int idx = seq_penalty_tokens[i];
        if (idx >= 0 && idx < vocab_size) {
            float val = seq_workspace[idx];
            if (repetition_penalty != 1.0f) {
                seq_workspace[idx] = (val > 0.0f) ? (val / repetition_penalty) : (val * repetition_penalty);
                val = seq_workspace[idx];
            }
            if (presence_penalty != 0.0f) {
                seq_workspace[idx] = val - presence_penalty;
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < vocab_size; i += block_size) {
        heap_insert(local_heap, local_size, effective_k, seq_workspace[i], i);
    }

    for (int i = 0; i < local_size; i++) {
        seq_topk_vals[tid * MAX_K + i] = local_heap[i].val;
        seq_topk_idxs[tid * MAX_K + i] = local_heap[i].idx;
    }
    for (int i = local_size; i < effective_k; i++) {
        seq_topk_vals[tid * MAX_K + i] = -FLT_MAX;
        seq_topk_idxs[tid * MAX_K + i] = -1;
    }
    __syncthreads();

    if (tid == 0) {
        HeapEntry merge_heap[MAX_K];
        int merge_size = 0;

        for (int t = 0; t < block_size; t++) {
            for (int i = 0; i < effective_k; i++) {
                float val = seq_topk_vals[t * MAX_K + i];
                int idx = seq_topk_idxs[t * MAX_K + i];
                if (idx >= 0) {
                    heap_insert(merge_heap, merge_size, effective_k, val, idx);
                }
            }
        }

        for (int i = 0; i < merge_size; i++) {
            seq_topk_vals[i] = merge_heap[i].val;
            seq_topk_idxs[i] = merge_heap[i].idx;
        }

        for (int i = 1; i < merge_size; i++) {
            float key_val = seq_topk_vals[i];
            int key_idx = seq_topk_idxs[i];
            int j = i - 1;
            while (j >= 0 && seq_topk_vals[j] < key_val) {
                seq_topk_vals[j + 1] = seq_topk_vals[j];
                seq_topk_idxs[j + 1] = seq_topk_idxs[j];
                j--;
            }
            seq_topk_vals[j + 1] = key_val;
            seq_topk_idxs[j + 1] = key_idx;
        }

        float max_val = seq_topk_vals[0];
        float sum = 0.0f;
        for (int i = 0; i < merge_size; i++) {
            float v = expf(seq_topk_vals[i] - max_val);
            seq_topk_vals[i] = v;
            sum += v;
        }
        for (int i = 0; i < merge_size; i++) {
            seq_topk_vals[i] /= sum;
        }

        if (top_p < 1.0f && top_k <= 0) {
            float cumsum = 0.0f;
            int cutoff = merge_size;
            for (int i = 0; i < merge_size; i++) {
                cumsum += seq_topk_vals[i];
                if (cumsum > top_p) {
                    cutoff = i + 1;
                    break;
                }
            }
            float renorm = 0.0f;
            for (int i = 0; i < cutoff; i++) renorm += seq_topk_vals[i];
            for (int i = cutoff; i < merge_size; i++) seq_topk_vals[i] = 0.0f;
            if (renorm > 0.0f) {
                for (int i = 0; i < cutoff; i++) seq_topk_vals[i] /= renorm;
            }
        }

        float cumsum = 0.0f;
        int sampled = seq_topk_idxs[merge_size - 1];
        for (int i = 0; i < merge_size; i++) {
            cumsum += seq_topk_vals[i];
            if (random_val <= cumsum) {
                sampled = seq_topk_idxs[i];
                break;
            }
        }
        seq_out[0] = sampled;
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
    const int* __restrict__ penalty_tokens,
    const int* __restrict__ penalty_offsets,
    int vocab_size,
    const float* __restrict__ temperatures,
    const float* __restrict__ repetition_penalties,
    const float* __restrict__ presence_penalties,
    const int* __restrict__ top_ks,
    const float* __restrict__ top_ps,
    const float* __restrict__ random_vals,
    int max_effective_k
) {
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;

    float temperature = temperatures[seq_idx];
    float repetition_penalty = repetition_penalties[seq_idx];
    float presence_penalty = presence_penalties[seq_idx];
    int top_k = top_ks[seq_idx];
    float top_p = top_ps[seq_idx];
    float random_val = random_vals[seq_idx];

    int pen_start = penalty_offsets[seq_idx];
    int pen_end = penalty_offsets[seq_idx + 1];
    int num_penalty_tokens = pen_end - pen_start;

    int effective_k;
    if (temperature <= 0.0f && top_k <= 0) {
        effective_k = 1;
    } else if (top_k > 0) {
        effective_k = (top_k < vocab_size) ? top_k : vocab_size;
    } else {
        effective_k = 64;
    }

    const __nv_bfloat16* seq_logits = logits + (size_t)seq_idx * vocab_size;
    float* seq_workspace = workspace + (size_t)seq_idx * vocab_size;
    int* seq_out = out_tokens + seq_idx;
    float* seq_topk_vals = topk_vals + (size_t)seq_idx * max_effective_k;
    int* seq_topk_idxs = topk_idxs + (size_t)seq_idx * max_effective_k;
    const int* seq_penalty_tokens = penalty_tokens + pen_start;

    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int* s_idxs = reinterpret_cast<int*>(s_vals + blockDim.x);

    float inv_temp = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(seq_logits[i]);
        seq_workspace[i] = val * inv_temp;
    }
    __syncthreads();

    for (int i = tid; i < num_penalty_tokens; i += blockDim.x) {
        int idx = seq_penalty_tokens[i];
        if (idx >= 0 && idx < vocab_size) {
            float val = seq_workspace[idx];
            if (repetition_penalty != 1.0f) {
                seq_workspace[idx] = (val > 0.0f) ? (val / repetition_penalty) : (val * repetition_penalty);
                val = seq_workspace[idx];
            }
            if (presence_penalty != 0.0f) {
                seq_workspace[idx] = val - presence_penalty;
            }
        }
    }
    __syncthreads();

    for (int k = 0; k < effective_k; k++) {
        float my_max = -FLT_MAX;
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

        if (tid == 0) {
            seq_topk_vals[k] = s_vals[0];
            seq_topk_idxs[k] = s_idxs[0];
            seq_workspace[s_idxs[0]] = -FLT_MAX;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_val = seq_topk_vals[0];
        float sum = 0.0f;
        for (int i = 0; i < effective_k; i++) {
            float v = expf(seq_topk_vals[i] - max_val);
            seq_topk_vals[i] = v;
            sum += v;
        }
        for (int i = 0; i < effective_k; i++) {
            seq_topk_vals[i] /= sum;
        }
        if (top_p < 1.0f && top_k <= 0) {
            float cumsum = 0.0f;
            int cutoff = effective_k;
            for (int i = 0; i < effective_k; i++) {
                cumsum += seq_topk_vals[i];
                if (cumsum > top_p) { cutoff = i + 1; break; }
            }
            float renorm = 0.0f;
            for (int i = 0; i < cutoff; i++) renorm += seq_topk_vals[i];
            for (int i = cutoff; i < effective_k; i++) seq_topk_vals[i] = 0.0f;
            if (renorm > 0.0f) {
                for (int i = 0; i < cutoff; i++) seq_topk_vals[i] /= renorm;
            }
        }
        float cumsum = 0.0f;
        int sampled = seq_topk_idxs[effective_k - 1];
        for (int i = 0; i < effective_k; i++) {
            cumsum += seq_topk_vals[i];
            if (random_val <= cumsum) { sampled = seq_topk_idxs[i]; break; }
        }
        seq_out[0] = sampled;
    }
}

// ---------------------------------------------------------------------------
// Host function: dispatch to appropriate kernel based on max_effective_k
// ---------------------------------------------------------------------------

void glm_sample_batch(GlmCtx* ctx, int* out_tokens, float* topk_vals, int* topk_idxs,
                      float* workspace, const void* logits,
                      const int* penalty_tokens, const int* penalty_offsets,
                      int vocab_size, int batch_size,
                      const float* temperatures, const float* repetition_penalties,
                      const float* presence_penalties, const int* top_ks,
                      const float* top_ps, const float* random_vals,
                      int max_effective_k) {
    cudaSetDevice(ctx->device_id);

    int block_size = SAMPLING_BLOCK_SIZE;

    if (max_effective_k <= 1) {
        sampling_kernel_batch<2><<<batch_size, block_size, 0, ctx->stream>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_offsets,
            vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, random_vals);
    } else if (max_effective_k <= 8) {
        sampling_kernel_batch<8><<<batch_size, block_size, 0, ctx->stream>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_offsets,
            vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, random_vals);
    } else if (max_effective_k <= 16) {
        sampling_kernel_batch<16><<<batch_size, block_size, 0, ctx->stream>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_offsets,
            vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, random_vals);
    } else if (max_effective_k <= 32) {
        sampling_kernel_batch<32><<<batch_size, block_size, 0, ctx->stream>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_offsets,
            vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, random_vals);
    } else if (max_effective_k <= 64) {
        sampling_kernel_batch<64><<<batch_size, block_size, 0, ctx->stream>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_offsets,
            vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, random_vals);
    } else {
        size_t shared_mem = block_size * (sizeof(float) + sizeof(int));
        sampling_kernel_argmax_batch<<<batch_size, block_size, shared_mem, ctx->stream>>>(
            out_tokens, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens, penalty_offsets,
            vocab_size, temperatures, repetition_penalties, presence_penalties,
            top_ks, top_ps, random_vals, max_effective_k);
    }
}
