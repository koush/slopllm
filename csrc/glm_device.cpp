#include "glm_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
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
    ctx->active_stream = 0;
    for (int i = 0; i < GLM_MAX_STREAMS; i++) {
        cudaStreamCreate(&ctx->streams[i]);
        cudaEventCreate(&ctx->events[i]);
    }
    cublasCreate(&CUBLAS(ctx));
    cublasSetStream(CUBLAS(ctx), ctx->streams[0]);
    cublasSetMathMode(CUBLAS(ctx), CUBLAS_TENSOR_OP_MATH);
    cublasLtCreate(reinterpret_cast<cublasLtHandle_t*>(&ctx->cublaslt_handle));
    cudaMalloc(&ctx->cublaslt_workspace, 32 * 1024 * 1024);
    // this suppresses most non deterministic output
    // can be used for sanity checking in case of deviation
    // cublasSetMathMode(CUBLAS(ctx), CUBLAS_PEDANTIC_MATH);

    return ctx;
}

void glm_free(GlmCtx* ctx) {
    if (!ctx) return;
    cudaSetDevice(ctx->device_id);
    cublasDestroy(CUBLAS(ctx));
    cublasLtDestroy(*reinterpret_cast<cublasLtHandle_t*>(&ctx->cublaslt_handle));
    cudaFree(ctx->cublaslt_workspace);
    for (int i = 0; i < GLM_MAX_STREAMS; i++) {
        cudaStreamDestroy(ctx->streams[i]);
        cudaEventDestroy(ctx->events[i]);
    }
    delete ctx;
}

// ---------------------------------------------------------------------------
// GPU memory management
// ---------------------------------------------------------------------------

void* glm_alloc(GlmCtx* ctx, size_t bytes) {
    void* ptr = nullptr;
    cudaSetDevice(ctx->device_id);
    cudaError_t prev = cudaGetLastError();
    if (prev != cudaSuccess) {
        fprintf(stderr, "glm_alloc: pending CUDA error on device %d before alloc: %s\n", ctx->device_id, cudaGetErrorString(prev));
    }
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_alloc: cudaMalloc(%zu) failed on device %d: %s\n", bytes, ctx->device_id, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void glm_free_buf(GlmCtx* ctx, void* ptr) {
    cudaSetDevice(ctx->device_id);
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
    cudaSetDevice(ctx->device_id);
    const void* src = (const char*)mmap_ptr + offset;
    cudaMemcpyAsync(gpu_dst, src, nbytes, cudaMemcpyHostToDevice, GLM_STREAM(ctx));
}

// ---------------------------------------------------------------------------
// Host <-> Device memory copy
// ---------------------------------------------------------------------------

void glm_h2d(GlmCtx* ctx, void* dst, const void* src, size_t bytes) {
    cudaSetDevice(ctx->device_id);
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, GLM_STREAM(ctx));
}

void glm_d2h(GlmCtx* ctx, void* dst, const void* src, size_t bytes) {
    cudaSetDevice(ctx->device_id);
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, GLM_STREAM(ctx));
    cudaStreamSynchronize(GLM_STREAM(ctx));
}

// ---------------------------------------------------------------------------
// Device <-> Device memory copy
// ---------------------------------------------------------------------------

void glm_memcpy(GlmCtx* ctx, void* dst, const void* src, size_t bytes, int kind) {
    cudaSetDevice(ctx->device_id);
    cudaMemcpyAsync(dst, src, bytes, static_cast<cudaMemcpyKind>(kind), GLM_STREAM(ctx));
}

void glm_memcpy2d(GlmCtx* ctx, void* dst, size_t dpitch,
                   const void* src, size_t spitch,
                   size_t width, size_t height, int kind) {
    cudaSetDevice(ctx->device_id);
    cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                       static_cast<cudaMemcpyKind>(kind), GLM_STREAM(ctx));
}

void glm_memcpy_peer(GlmCtx* ctx, void* dst, int dstDevice,
                      const void* src, int srcDevice, size_t bytes) {
    cudaSetDevice(ctx->device_id);
    cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, bytes, GLM_STREAM(ctx));
}

void glm_memcpy3d_peer(GlmCtx* ctx,
    void* dstPtr, size_t dstPitch, size_t dstXSize, size_t dstYSize, int dstDevice,
    size_t dstPosX, size_t dstPosY, size_t dstPosZ,
    const void* srcPtr, size_t srcPitch, size_t srcXSize, size_t srcYSize, int srcDevice,
    size_t srcPosX, size_t srcPosY, size_t srcPosZ,
    size_t width, size_t height, size_t depth) {
    cudaSetDevice(ctx->device_id);
    cudaMemcpy3DPeerParms p{};
    p.dstPtr.ptr = dstPtr;
    p.dstPtr.pitch = dstPitch;
    p.dstPtr.xsize = dstXSize;
    p.dstPtr.ysize = dstYSize;
    p.dstDevice = dstDevice;
    p.dstPos.x = dstPosX;
    p.dstPos.y = dstPosY;
    p.dstPos.z = dstPosZ;
    p.srcPtr.ptr = const_cast<void*>(srcPtr);
    p.srcPtr.pitch = srcPitch;
    p.srcPtr.xsize = srcXSize;
    p.srcPtr.ysize = srcYSize;
    p.srcDevice = srcDevice;
    p.srcPos.x = srcPosX;
    p.srcPos.y = srcPosY;
    p.srcPos.z = srcPosZ;
    p.extent.width = width;
    p.extent.height = height;
    p.extent.depth = depth;
    cudaMemcpy3DPeerAsync(&p, GLM_STREAM(ctx));
}

// ---------------------------------------------------------------------------
// Stream synchronization
// ---------------------------------------------------------------------------

void glm_synchronize(GlmCtx* ctx) {
    cudaSetDevice(ctx->device_id);
    cudaError_t err = cudaStreamSynchronize(ctx->streams[ctx->active_stream]);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_synchronize failed: %s\n", cudaGetErrorString(err));
    }
}

void glm_synchronize_stream(GlmCtx* ctx, int stream_idx) {
    if (stream_idx < 0 || stream_idx >= GLM_MAX_STREAMS) {
        fprintf(stderr, "glm_synchronize_stream: invalid stream index %d\n", stream_idx);
        return;
    }
    cudaSetDevice(ctx->device_id);
    cudaError_t err = cudaStreamSynchronize(ctx->streams[stream_idx]);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_synchronize_stream failed: %s\n", cudaGetErrorString(err));
    }
}

void glm_set_stream(GlmCtx* ctx, int stream_idx) {
    if (stream_idx < 0 || stream_idx >= GLM_MAX_STREAMS) {
        fprintf(stderr, "glm_set_stream: invalid stream index %d (max %d)\n", stream_idx, GLM_MAX_STREAMS - 1);
        return;
    }
    cudaSetDevice(ctx->device_id);
    ctx->active_stream = stream_idx;
    cublasSetStream(CUBLAS(ctx), ctx->streams[stream_idx]);
}

void glm_event_record(GlmCtx* ctx, int event_idx, int stream_idx) {
    if (event_idx < 0 || event_idx >= GLM_MAX_STREAMS || stream_idx < 0 || stream_idx >= GLM_MAX_STREAMS) {
        fprintf(stderr, "glm_event_record: invalid index event=%d stream=%d\n", event_idx, stream_idx);
        return;
    }
    cudaSetDevice(ctx->device_id);
    cudaEventRecord(ctx->events[event_idx], ctx->streams[stream_idx]);
}

void glm_stream_wait_event(GlmCtx* ctx, int stream_idx, int event_idx) {
    if (stream_idx < 0 || stream_idx >= GLM_MAX_STREAMS || event_idx < 0 || event_idx >= GLM_MAX_STREAMS) {
        fprintf(stderr, "glm_stream_wait_event: invalid index stream=%d event=%d\n", stream_idx, event_idx);
        return;
    }
    cudaSetDevice(ctx->device_id);
    cudaStreamWaitEvent(ctx->streams[stream_idx], ctx->events[event_idx], 0);
}

// ---------------------------------------------------------------------------
// CUDA Graph operations
// ---------------------------------------------------------------------------

void glm_graph_begin_capture(GlmCtx* ctx) {
    cudaSetDevice(ctx->device_id);
    cudaError_t err = cudaStreamBeginCapture(GLM_STREAM(ctx), cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_graph_begin_capture failed: %s\n", cudaGetErrorString(err));
    }
}

void* glm_graph_end_capture(GlmCtx* ctx) {
    cudaSetDevice(ctx->device_id);
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(GLM_STREAM(ctx), &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_graph_end_capture failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    if (!graph) {
        fprintf(stderr, "glm_graph_end_capture returned null graph\n");
    }
    return reinterpret_cast<void*>(graph);
}

void* glm_graph_instantiate(GlmCtx* ctx, void* graph) {
    cudaSetDevice(ctx->device_id);
    cudaGraphExec_t graph_exec = nullptr;
    cudaError_t err = cudaGraphInstantiate(&graph_exec, reinterpret_cast<cudaGraph_t>(graph), nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_graph_instantiate failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return reinterpret_cast<void*>(graph_exec);
}

void glm_graph_launch(GlmCtx* ctx, void* graph_exec) {
    cudaSetDevice(ctx->device_id);
    cudaError_t err = cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph_exec), GLM_STREAM(ctx));
    if (err != cudaSuccess) {
        fprintf(stderr, "glm_graph_launch failed: %s\n", cudaGetErrorString(err));
    }
}

int glm_graph_exec_update(GlmCtx* ctx, void* graph_exec, void* graph) {
    cudaSetDevice(ctx->device_id);
    cudaGraphExecUpdateResultInfo result_info = {};
    cudaGraphExecUpdate(reinterpret_cast<cudaGraphExec_t>(graph_exec),
                        reinterpret_cast<cudaGraph_t>(graph),
                        &result_info);
    return (result_info.result == cudaGraphExecUpdateSuccess) ? 0 : 1;
}

void glm_graph_destroy(GlmCtx* ctx, void* graph) {
    if (graph) {
        cudaSetDevice(ctx->device_id);
        cudaError_t err = cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph));
        if (err != cudaSuccess) {
            fprintf(stderr, "glm_graph_destroy failed: %s\n", cudaGetErrorString(err));
        }
    }
}

void glm_graph_exec_destroy(GlmCtx* ctx, void* graph_exec) {
    if (graph_exec) {
        cudaSetDevice(ctx->device_id);
        cudaError_t err = cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(graph_exec));
        if (err != cudaSuccess) {
            fprintf(stderr, "glm_graph_exec_destroy failed: %s\n", cudaGetErrorString(err));
        }
    }
}
