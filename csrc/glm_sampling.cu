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

// Insert into min-heap with runtime capacity.
// The heap array must have at least `capacity` entries allocated.
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
// Templated sampling kernel: heap-based top-K with compile-time MAX_K
// MAX_K determines the local array size; effective_k (runtime) determines
// the actual heap capacity used.
// ---------------------------------------------------------------------------

template<int MAX_K>
__global__ void sampling_kernel_templated(
    int* __restrict__ out_token,
    float* __restrict__ topk_vals,
    int* __restrict__ topk_idxs,
    float* __restrict__ workspace,
    const __nv_bfloat16* __restrict__ logits,
    const int* __restrict__ penalty_tokens,
    int vocab_size,
    int num_penalty_tokens,
    float temperature,
    float repetition_penalty,
    float presence_penalty,
    int top_k,
    float top_p,
    float random_val,
    int effective_k
) {
    // Each thread maintains a local min-heap of effective_k entries
    // Array is sized MAX_K (compile-time) but only effective_k entries are used
    HeapEntry local_heap[MAX_K];
    int local_size = 0;

    // Phase 1: Convert BF16 -> F32 + temperature scaling
    float inv_temp = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(logits[i]);
        workspace[i] = val * inv_temp;
    }
    __syncthreads();

    // Phase 2: Apply repetition and presence penalties
    for (int i = threadIdx.x; i < num_penalty_tokens; i += blockDim.x) {
        int idx = penalty_tokens[i];
        if (idx >= 0 && idx < vocab_size) {
            float val = workspace[idx];
            if (repetition_penalty != 1.0f) {
                workspace[idx] = (val > 0.0f) ? (val / repetition_penalty) : (val * repetition_penalty);
                val = workspace[idx];
            }
            if (presence_penalty != 0.0f) {
                workspace[idx] = val - presence_penalty;
            }
        }
    }
    __syncthreads();

    // Phase 3: Per-thread top-K using min-heap with effective_k capacity
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        heap_insert(local_heap, local_size, effective_k, workspace[i], i);
    }

    // Phase 4: Write per-thread heaps to global memory for merging
    // Each thread writes local_size entries (should equal effective_k)
    int tid = threadIdx.x;
    for (int i = 0; i < local_size; i++) {
        topk_vals[tid * MAX_K + i] = local_heap[i].val;
        topk_idxs[tid * MAX_K + i] = local_heap[i].idx;
    }
    // Zero-fill remaining entries (in case local_size < effective_k for small vocabs)
    for (int i = local_size; i < effective_k; i++) {
        topk_vals[tid * MAX_K + i] = -FLT_MAX;
        topk_idxs[tid * MAX_K + i] = -1;
    }
    __syncthreads();

    // Phase 5: Thread 0 merges all per-thread heaps into final top-K
    if (tid == 0) {
        HeapEntry merge_heap[MAX_K];
        int merge_size = 0;

        for (int t = 0; t < blockDim.x; t++) {
            for (int i = 0; i < effective_k; i++) {
                float val = topk_vals[t * MAX_K + i];
                int idx = topk_idxs[t * MAX_K + i];
                if (idx >= 0) {
                    heap_insert(merge_heap, merge_size, effective_k, val, idx);
                }
            }
        }

        // Extract top-K from merge heap into output arrays
        // The min-heap contains effective_k elements but not in sorted order
        // Copy to output and sort descending
        for (int i = 0; i < merge_size; i++) {
            topk_vals[i] = merge_heap[i].val;
            topk_idxs[i] = merge_heap[i].idx;
        }

        // Insertion sort descending (K is small)
        for (int i = 1; i < merge_size; i++) {
            float key_val = topk_vals[i];
            int key_idx = topk_idxs[i];
            int j = i - 1;
            while (j >= 0 && topk_vals[j] < key_val) {
                topk_vals[j + 1] = topk_vals[j];
                topk_idxs[j + 1] = topk_idxs[j];
                j--;
            }
            topk_vals[j + 1] = key_val;
            topk_idxs[j + 1] = key_idx;
        }

        // Phase 6: Softmax + top-P + multinomial sample
        float max_val = topk_vals[0];

        float sum = 0.0f;
        for (int i = 0; i < merge_size; i++) {
            float v = expf(topk_vals[i] - max_val);
            topk_vals[i] = v;
            sum += v;
        }

        for (int i = 0; i < merge_size; i++) {
            topk_vals[i] /= sum;
        }

        if (top_p < 1.0f && top_k <= 0) {
            float cumsum = 0.0f;
            int cutoff = merge_size;
            for (int i = 0; i < merge_size; i++) {
                cumsum += topk_vals[i];
                if (cumsum > top_p) {
                    cutoff = i + 1;
                    break;
                }
            }
            float renorm = 0.0f;
            for (int i = 0; i < cutoff; i++) renorm += topk_vals[i];
            for (int i = cutoff; i < merge_size; i++) topk_vals[i] = 0.0f;
            if (renorm > 0.0f) {
                for (int i = 0; i < cutoff; i++) topk_vals[i] /= renorm;
            }
        }

        float cumsum = 0.0f;
        int sampled = topk_idxs[merge_size - 1];
        for (int i = 0; i < merge_size; i++) {
            cumsum += topk_vals[i];
            if (random_val <= cumsum) {
                sampled = topk_idxs[i];
                break;
            }
        }
        out_token[0] = sampled;
    }
}

// ---------------------------------------------------------------------------
// K-passes argmax kernel (fallback for large K)
// ---------------------------------------------------------------------------

__global__ void sampling_kernel_argmax(
    int* __restrict__ out_token,
    float* __restrict__ topk_vals,
    int* __restrict__ topk_idxs,
    float* __restrict__ workspace,
    const __nv_bfloat16* __restrict__ logits,
    const int* __restrict__ penalty_tokens,
    int vocab_size,
    int num_penalty_tokens,
    float temperature,
    float repetition_penalty,
    float presence_penalty,
    int top_k,
    float top_p,
    float random_val,
    int effective_k
) {
    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int* s_idxs = reinterpret_cast<int*>(s_vals + blockDim.x);

    // Phase 1: Convert BF16 -> F32 + temperature scaling
    float inv_temp = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(logits[i]);
        workspace[i] = val * inv_temp;
    }
    __syncthreads();

    // Phase 2: Apply penalties
    for (int i = threadIdx.x; i < num_penalty_tokens; i += blockDim.x) {
        int idx = penalty_tokens[i];
        if (idx >= 0 && idx < vocab_size) {
            float val = workspace[idx];
            if (repetition_penalty != 1.0f) {
                workspace[idx] = (val > 0.0f) ? (val / repetition_penalty) : (val * repetition_penalty);
                val = workspace[idx];
            }
            if (presence_penalty != 0.0f) {
                workspace[idx] = val - presence_penalty;
            }
        }
    }
    __syncthreads();

    // Phase 3: Top-K via K passes of parallel argmax
    for (int k = 0; k < effective_k; k++) {
        float my_max = -FLT_MAX;
        int my_idx = -1;
        for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
            float val = workspace[i];
            if (val > my_max) {
                my_max = val;
                my_idx = i;
            }
        }
        s_vals[threadIdx.x] = my_max;
        s_idxs[threadIdx.x] = my_idx;
        __syncthreads();

        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (s_vals[threadIdx.x + s] > s_vals[threadIdx.x]) {
                    s_vals[threadIdx.x] = s_vals[threadIdx.x + s];
                    s_idxs[threadIdx.x] = s_idxs[threadIdx.x + s];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            topk_vals[k] = s_vals[0];
            topk_idxs[k] = s_idxs[0];
            workspace[s_idxs[0]] = -FLT_MAX;
        }
        __syncthreads();
    }

    // Phase 4: Softmax + top-P + sample
    if (threadIdx.x == 0) {
        float max_val = topk_vals[0];
        float sum = 0.0f;
        for (int i = 0; i < effective_k; i++) {
            float v = expf(topk_vals[i] - max_val);
            topk_vals[i] = v;
            sum += v;
        }
        for (int i = 0; i < effective_k; i++) {
            topk_vals[i] /= sum;
        }
        if (top_p < 1.0f && top_k <= 0) {
            float cumsum = 0.0f;
            int cutoff = effective_k;
            for (int i = 0; i < effective_k; i++) {
                cumsum += topk_vals[i];
                if (cumsum > top_p) { cutoff = i + 1; break; }
            }
            float renorm = 0.0f;
            for (int i = 0; i < cutoff; i++) renorm += topk_vals[i];
            for (int i = cutoff; i < effective_k; i++) topk_vals[i] = 0.0f;
            if (renorm > 0.0f) {
                for (int i = 0; i < cutoff; i++) topk_vals[i] /= renorm;
            }
        }
        float cumsum = 0.0f;
        int sampled = topk_idxs[effective_k - 1];
        for (int i = 0; i < effective_k; i++) {
            cumsum += topk_vals[i];
            if (random_val <= cumsum) { sampled = topk_idxs[i]; break; }
        }
        out_token[0] = sampled;
    }
}

// ---------------------------------------------------------------------------
// Host function: dispatch to appropriate kernel based on effective_k
// ---------------------------------------------------------------------------

void glm_sample(GlmCtx* ctx, int* out_token, float* topk_vals, int* topk_idxs,
                float* workspace, const void* logits, const int* penalty_tokens,
                int vocab_size, int num_penalty_tokens,
                float temperature, float repetition_penalty, float presence_penalty,
                int top_k, float top_p, float random_val) {
    cudaSetDevice(ctx->device_id);

    int effective_k;
    if (temperature <= 0.0f && top_k <= 0) {
        effective_k = 1;
    } else if (top_k > 0) {
        effective_k = (top_k < vocab_size) ? top_k : vocab_size;
    } else {
        effective_k = 64; // internal top-K for top-P-only case
    }

    int block_size = SAMPLING_BLOCK_SIZE;

    if (effective_k <= 1) {
        sampling_kernel_templated<2><<<1, block_size, 0, ctx->stream>>>(
            out_token, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens,
            vocab_size, num_penalty_tokens,
            temperature, repetition_penalty, presence_penalty,
            top_k, top_p, random_val, effective_k);
    } else if (effective_k <= 8) {
        sampling_kernel_templated<8><<<1, block_size, 0, ctx->stream>>>(
            out_token, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens,
            vocab_size, num_penalty_tokens,
            temperature, repetition_penalty, presence_penalty,
            top_k, top_p, random_val, effective_k);
    } else if (effective_k <= 16) {
        sampling_kernel_templated<16><<<1, block_size, 0, ctx->stream>>>(
            out_token, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens,
            vocab_size, num_penalty_tokens,
            temperature, repetition_penalty, presence_penalty,
            top_k, top_p, random_val, effective_k);
    } else if (effective_k <= 32) {
        sampling_kernel_templated<32><<<1, block_size, 0, ctx->stream>>>(
            out_token, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens,
            vocab_size, num_penalty_tokens,
            temperature, repetition_penalty, presence_penalty,
            top_k, top_p, random_val, effective_k);
    } else if (effective_k <= 64) {
        sampling_kernel_templated<64><<<1, block_size, 0, ctx->stream>>>(
            out_token, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens,
            vocab_size, num_penalty_tokens,
            temperature, repetition_penalty, presence_penalty,
            top_k, top_p, random_val, effective_k);
    } else {
        // Fallback: K-passes argmax for large K
        size_t shared_mem = block_size * (sizeof(float) + sizeof(int));
        sampling_kernel_argmax<<<1, block_size, shared_mem, ctx->stream>>>(
            out_token, topk_vals, topk_idxs, workspace,
            (const __nv_bfloat16*)logits, penalty_tokens,
            vocab_size, num_penalty_tokens,
            temperature, repetition_penalty, presence_penalty,
            top_k, top_p, random_val, effective_k);
    }
}
