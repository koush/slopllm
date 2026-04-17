#include "glm_ops.h"
#include <nccl.h>
#include <cstdio>

void glm_nccl_unique_id(void* out_id) {
    ncclResult_t result = ncclGetUniqueId(static_cast<ncclUniqueId*>(out_id));
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_unique_id failed: %s\n", ncclGetErrorString(result));
    }
}

void* glm_nccl_comm_init_rank(int rank, int world_size, const void* unique_id) {
    ncclComm_t comm = nullptr;
    cudaSetDevice(0); // caller should have set device via GlmCtx
    ncclResult_t result = ncclCommInitRank(&comm, world_size,
                         *static_cast<const ncclUniqueId*>(unique_id), rank);
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_comm_init_rank failed: %s\n", ncclGetErrorString(result));
        return nullptr;
    }
    return static_cast<void*>(comm);
}

void glm_nccl_comm_destroy(void* comm) {
    if (comm) {
        ncclCommDestroy(static_cast<ncclComm_t>(comm));
    }
}

void glm_nccl_all_reduce(void* comm, GlmCtx* ctx,
                          const void* sendbuff, void* recvbuff,
                          size_t count, int datatype, int op) {
    ncclResult_t result = ncclAllReduce(sendbuff, recvbuff, count,
                          static_cast<ncclDataType_t>(datatype),
                          static_cast<ncclRedOp_t>(op),
                          static_cast<ncclComm_t>(comm), ctx->stream);
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_all_reduce failed: %s\n", ncclGetErrorString(result));
    }
}

void glm_nccl_all_gather(void* comm, GlmCtx* ctx,
                          const void* sendbuff, void* recvbuff,
                          size_t count, int datatype) {
    ncclResult_t result = ncclAllGather(sendbuff, recvbuff, count,
                          static_cast<ncclDataType_t>(datatype),
                          static_cast<ncclComm_t>(comm), ctx->stream);
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_all_gather failed: %s\n", ncclGetErrorString(result));
    }
}
