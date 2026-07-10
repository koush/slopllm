#include "glm_ops.h"
#include <nccl.h>
#include <cstdio>

void glm_nccl_unique_id(void* out_id) {
    ncclResult_t result = ncclGetUniqueId(static_cast<ncclUniqueId*>(out_id));
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_unique_id failed: %s\n", ncclGetErrorString(result));
    }
}

int glm_nccl_group_start() {
    ncclResult_t result = ncclGroupStart();
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_group_start failed: %s\n", ncclGetErrorString(result));
        return -1;
    }
    return 0;
}

int glm_nccl_group_end() {
    ncclResult_t result = ncclGroupEnd();
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_group_end failed: %s\n", ncclGetErrorString(result));
        return -1;
    }
    return 0;
}

void* glm_nccl_comm_init_rank(int device_id, int rank, int world_size, const void* unique_id) {
    ncclComm_t comm = nullptr;
    cudaSetDevice(device_id);
    ncclResult_t result = ncclCommInitRank(&comm, world_size,
                         *static_cast<const ncclUniqueId*>(unique_id), rank);
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_comm_init_rank failed: %s\n", ncclGetErrorString(result));
        return nullptr;
    }
    return static_cast<void*>(comm);
}

int glm_nccl_comm_init_all(void** comms, int ndev, const int* devlist) {
    ncclResult_t result = ncclCommInitAll(reinterpret_cast<ncclComm_t*>(comms), ndev, devlist);
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_comm_init_all failed: %s\n", ncclGetErrorString(result));
        return -1;
    }
    return 0;
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
                          static_cast<ncclComm_t>(comm), GLM_STREAM(ctx));
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_all_reduce failed: %s\n", ncclGetErrorString(result));
    }
}

void glm_nccl_all_gather(void* comm, GlmCtx* ctx,
                          const void* sendbuff, void* recvbuff,
                          size_t count, int datatype) {
    ncclResult_t result = ncclAllGather(sendbuff, recvbuff, count,
                          static_cast<ncclDataType_t>(datatype),
                          static_cast<ncclComm_t>(comm), GLM_STREAM(ctx));
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_all_gather failed: %s\n", ncclGetErrorString(result));
    }
}

void glm_nccl_send(void* comm, GlmCtx* ctx,
                    const void* sendbuff, size_t count, int datatype, int peer) {
    ncclResult_t result = ncclSend(sendbuff, count,
                          static_cast<ncclDataType_t>(datatype),
                          peer,
                          static_cast<ncclComm_t>(comm), GLM_STREAM(ctx));
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_send failed: %s\n", ncclGetErrorString(result));
    }
}

void glm_nccl_recv(void* comm, GlmCtx* ctx,
                    void* recvbuff, size_t count, int datatype, int peer) {
    ncclResult_t result = ncclRecv(recvbuff, count,
                          static_cast<ncclDataType_t>(datatype),
                          peer,
                          static_cast<ncclComm_t>(comm), GLM_STREAM(ctx));
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_recv failed: %s\n", ncclGetErrorString(result));
    }
}

void glm_nccl_reduce_scatter(void* comm, GlmCtx* ctx,
                              const void* sendbuff, void* recvbuff,
                              size_t recvcount, int datatype, int op) {
    ncclResult_t result = ncclReduceScatter(sendbuff, recvbuff, recvcount,
                          static_cast<ncclDataType_t>(datatype),
                          static_cast<ncclRedOp_t>(op),
                          static_cast<ncclComm_t>(comm), GLM_STREAM(ctx));
    if (result != ncclSuccess) {
        fprintf(stderr, "glm_nccl_reduce_scatter failed: %s\n", ncclGetErrorString(result));
    }
}
