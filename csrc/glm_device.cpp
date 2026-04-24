#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define CUBLAS(ctx) (*reinterpret_cast<cublasHandle_t*>(&(ctx)->cublas_handle))

// ---------------------------------------------------------------------------
// Context management
// ---------------------------------------------------------------------------

GlmCtx* glm_init(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_init: cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(err));
        return nullptr;
    }
    GlmCtx* ctx = new GlmCtx();
    ctx->device_id = device_id;
    cudaStreamCreate(&ctx->stream);
    cublasCreate(&CUBLAS(ctx));
    cublasSetStream(CUBLAS(ctx), ctx->stream);

    return ctx;
}

void glm_free(GlmCtx* ctx) {
    if (!ctx) return;

    cublasDestroy(CUBLAS(ctx));
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

// ---------------------------------------------------------------------------
// GPU memory management
// ---------------------------------------------------------------------------

void* glm_alloc(GlmCtx* ctx, size_t bytes) {
    void* ptr = nullptr;
    cudaSetDevice(ctx->device_id);
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_alloc: cudaMalloc(%zu) failed: %s\n", bytes, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void glm_free_buf(GlmCtx* ctx, void* ptr) {
    (void)ctx;
    if (ptr) cudaFree(ptr);
}

// ---------------------------------------------------------------------------
// Pinned host memory
// ---------------------------------------------------------------------------

void* glm_alloc_pinned(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t status = cudaMallocHost(&ptr, bytes);
    if (status != cudaSuccess) {
        fprintf(stderr, "glm_alloc_pinned failed: %s\n", cudaGetErrorString(status));
        return nullptr;
    }
    return ptr;
}

void glm_free_pinned(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

void glm_write_pinned(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
}

// ---------------------------------------------------------------------------
// Mmap
// ---------------------------------------------------------------------------

void* glm_mmap_open(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "glm_mmap_open: cannot open %s\n", path);
        return nullptr;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "glm_mmap_open: fstat failed for %s\n", path);
        close(fd);
        return nullptr;
    }

    size_t size = (size_t)st.st_size;
    void* ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED) {
        fprintf(stderr, "glm_mmap_open: mmap failed for %s (size=%llu)\n", path, (unsigned long long)size);
        return nullptr;
    }

    return ptr;
}

void glm_mmap_close(void* ptr, uint64_t size) {
    if (ptr) {
        munmap(ptr, (size_t)size);
    }
}

void glm_mmap_load(GlmCtx* ctx, void* gpu_dst, const void* mmap_ptr,
                   uint64_t offset, uint64_t nbytes) {
    const void* src = (const char*)mmap_ptr + offset;
    cudaMemcpyAsync(gpu_dst, src, nbytes, cudaMemcpyHostToDevice, ctx->stream);
}

// ---------------------------------------------------------------------------
// Host <-> Device memory copy
// ---------------------------------------------------------------------------

void glm_h2d(GlmCtx* ctx, void* dst, const void* src, size_t bytes) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, ctx->stream);
}

void glm_d2h(GlmCtx* ctx, void* dst, const void* src, size_t bytes) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
}

// ---------------------------------------------------------------------------
// Device <-> Device memory copy
// ---------------------------------------------------------------------------

void glm_memcpy(GlmCtx* ctx, void* dst, const void* src, size_t bytes, int kind) {
    cudaMemcpyAsync(dst, src, bytes, static_cast<cudaMemcpyKind>(kind), ctx->stream);
}

void glm_memcpy2d(GlmCtx* ctx, void* dst, size_t dpitch,
                  const void* src, size_t spitch,
                  size_t width, size_t height, int kind) {
    cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                      static_cast<cudaMemcpyKind>(kind), ctx->stream);
}

// ---------------------------------------------------------------------------
// Stream synchronization
// ---------------------------------------------------------------------------

void glm_synchronize(GlmCtx* ctx) {
    cudaError_t err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_synchronize failed: %s\n", cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// CUDA Graph operations
// ---------------------------------------------------------------------------

void glm_graph_begin_capture(GlmCtx* ctx) {
    cudaSetDevice(ctx->device_id);
    cudaError_t err = cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_graph_begin_capture failed: %s\n", cudaGetErrorString(err));
    }
}

void* glm_graph_end_capture(GlmCtx* ctx) {
    cudaSetDevice(ctx->device_id);
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(ctx->stream, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_graph_end_capture failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    if (!graph) {
        fprintf(stderr, "glm_graph_end_capture returned null graph\n");
    }
    return reinterpret_cast<void*>(graph);
}

void* glm_graph_instantiate(void* graph) {
    cudaGraphExec_t graph_exec = nullptr;
    cudaError_t err = cudaGraphInstantiate(&graph_exec, reinterpret_cast<cudaGraph_t>(graph), nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_graph_instantiate failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return reinterpret_cast<void*>(graph_exec);
}

void glm_graph_launch(void* graph_exec, GlmCtx* ctx) {
    cudaSetDevice(ctx->device_id);
    cudaError_t err = cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph_exec), ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_graph_launch failed: %s\n", cudaGetErrorString(err));
    }
}

int glm_graph_exec_update(void* graph_exec, void* graph) {
    cudaGraphExecUpdateResultInfo result_info = {};
    cudaGraphExecUpdate(reinterpret_cast<cudaGraphExec_t>(graph_exec),
                        reinterpret_cast<cudaGraph_t>(graph),
                        &result_info);
    return (result_info.result == cudaGraphExecUpdateSuccess) ? 0 : 1;
}

void glm_graph_destroy(void* graph) {
    if (graph) {
        cudaError_t err = cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph));
        if (err != cudaSuccess) {
            fprintf(stderr, "glm_graph_destroy failed: %s\n", cudaGetErrorString(err));
        }
    }
}

void glm_graph_exec_destroy(void* graph_exec) {
    if (graph_exec) {
        cudaError_t err = cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(graph_exec));
        if (err != cudaSuccess) {
            fprintf(stderr, "glm_graph_exec_destroy failed: %s\n", cudaGetErrorString(err));
        }
    }
}
