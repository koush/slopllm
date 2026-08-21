#include <napi.h>
#include "glm_ops.h"
#include <cstdint>
#include <cuda_bf16.h>

static Napi::Value Init(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "Expected device_id (number)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    int device_id = info[0].As<Napi::Number>().Int32Value();
    GlmCtx* ctx = glm_init(device_id);
    return Napi::Number::New(env, reinterpret_cast<uintptr_t>(ctx));
}

static Napi::Value Free(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "Expected ctx pointer").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ptr = info[0].As<Napi::Number>().Int64Value();
    glm_free(reinterpret_cast<GlmCtx*>(ptr));
    return env.Undefined();
}

static Napi::Value Alloc(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, size)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    size_t bytes = info[1].As<Napi::Number>().Int64Value();
    void* ptr = glm_alloc(reinterpret_cast<GlmCtx*>(ctx_ptr), bytes);
    return Napi::Number::New(env, reinterpret_cast<uintptr_t>(ptr));
}

static Napi::Value FreeBuf(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, ptr)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t buf_ptr = info[1].As<Napi::Number>().Int64Value();
    glm_free_buf(reinterpret_cast<GlmCtx*>(ctx_ptr), reinterpret_cast<void*>(buf_ptr));
    return env.Undefined();
}

static Napi::Value H2D(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (ctx, gpu_ptr, cpu_buf)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t gpu_ptr = info[1].As<Napi::Number>().Int64Value();
    Napi::Buffer<char> buf = info[2].As<Napi::Buffer<char>>();
    glm_h2d(reinterpret_cast<GlmCtx*>(ctx_ptr),
            reinterpret_cast<void*>(gpu_ptr), buf.Data(), buf.Length());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("h2D failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value WritePointers(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 11) {
        Napi::TypeError::New(env, "Expected (ctx, dst, p0, p1, p2, p3, p4, p5, p6, p7, n)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t dst_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t ptrs[8];
    for (int i = 0; i < 8; i++) {
        ptrs[i] = info[2 + i].As<Napi::Number>().Int64Value();
    }
    int n = info[10].As<Napi::Number>().Int32Value();
    glm_write_pointers(reinterpret_cast<GlmCtx*>(ctx_ptr),
                       reinterpret_cast<void*>(dst_ptr),
                       reinterpret_cast<void*>(ptrs[0]),
                       reinterpret_cast<void*>(ptrs[1]),
                       reinterpret_cast<void*>(ptrs[2]),
                       reinterpret_cast<void*>(ptrs[3]),
                       reinterpret_cast<void*>(ptrs[4]),
                       reinterpret_cast<void*>(ptrs[5]),
                       reinterpret_cast<void*>(ptrs[6]),
                       reinterpret_cast<void*>(ptrs[7]),
                       n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("writePointers failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value D2H(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (ctx, cpu_buf, gpu_ptr)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    Napi::Buffer<char> buf = info[1].As<Napi::Buffer<char>>();
    uintptr_t gpu_ptr = info[2].As<Napi::Number>().Int64Value();
    glm_d2h(reinterpret_cast<GlmCtx*>(ctx_ptr),
            buf.Data(), reinterpret_cast<void*>(gpu_ptr), buf.Length());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("d2H failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Rmsnorm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, weight, eps, dim, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t wt_ptr = info[3].As<Napi::Number>().Int64Value();
    float eps = info[4].As<Napi::Number>().FloatValue();
    int dim = info[5].As<Napi::Number>().Int32Value();
    int batch = info[6].As<Napi::Number>().Int32Value();
    glm_rmsnorm(reinterpret_cast<GlmCtx*>(ctx_ptr),
                reinterpret_cast<void*>(out_ptr),
                reinterpret_cast<const void*>(in_ptr),
                reinterpret_cast<const void*>(wt_ptr),
                eps, dim, batch);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("rmsnorm failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value FusedAddRmsnorm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 9) {
        Napi::TypeError::New(env, "Expected (ctx, out, residual, input_a, input_b, weight, eps, dim, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t res_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t a_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t b_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t wt_ptr = info[5].As<Napi::Number>().Int64Value();
    float eps = info[6].As<Napi::Number>().FloatValue();
    int dim = info[7].As<Napi::Number>().Int32Value();
    int batch = info[8].As<Napi::Number>().Int32Value();
    glm_fused_add_rmsnorm(reinterpret_cast<GlmCtx*>(ctx_ptr),
                           reinterpret_cast<void*>(out_ptr),
                           reinterpret_cast<void*>(res_ptr),
                           reinterpret_cast<const void*>(a_ptr),
                           reinterpret_cast<const void*>(b_ptr),
                           reinterpret_cast<const void*>(wt_ptr),
                           eps, dim, batch);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("fusedAddRmsnorm failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value FusedNormRope(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 13) {
        Napi::TypeError::New(env, "Expected (ctx, out, in, weight, cos, sin, eps, rope_dim, head_dim, n_heads, seq_len, batch, in_stride[, interleaved])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t wt_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t cos_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t sin_ptr = info[5].As<Napi::Number>().Int64Value();
    float eps = info[6].As<Napi::Number>().FloatValue();
    int rope_dim = info[7].As<Napi::Number>().Int32Value();
    int head_dim = info[8].As<Napi::Number>().Int32Value();
    int n_heads = info[9].As<Napi::Number>().Int32Value();
    int seq_len = info[10].As<Napi::Number>().Int32Value();
    int batch = info[11].As<Napi::Number>().Int32Value();
    int in_stride = info[12].As<Napi::Number>().Int32Value();
    bool interleaved = info.Length() > 13 ? info[13].As<Napi::Boolean>().Value() : false;
    glm_fused_norm_rope(reinterpret_cast<GlmCtx*>(ctx_ptr),
                         reinterpret_cast<void*>(out_ptr),
                         reinterpret_cast<const void*>(in_ptr),
                         reinterpret_cast<const void*>(wt_ptr),
                         reinterpret_cast<const void*>(cos_ptr),
                         reinterpret_cast<const void*>(sin_ptr),
                         eps, rope_dim, head_dim, n_heads, seq_len, batch, in_stride, interleaved);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("fusedNormRope failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value SiluAndMul(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, gate, up, intermediate, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t gate_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t up_ptr = info[3].As<Napi::Number>().Int64Value();
    int intermediate = info[4].As<Napi::Number>().Int32Value();
    int batch = info[5].As<Napi::Number>().Int32Value();
    glm_silu_and_mul(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(out_ptr),
                     reinterpret_cast<const void*>(gate_ptr),
                     reinterpret_cast<const void*>(up_ptr),
                     intermediate, batch);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("siluAndMul failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Linear(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 9) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, weight, batch, n, k, workspace, workspace_size)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t wt_ptr = info[3].As<Napi::Number>().Int64Value();
    int batch = info[4].As<Napi::Number>().Int32Value();
    int n = info[5].As<Napi::Number>().Int32Value();
    int k = info[6].As<Napi::Number>().Int32Value();
    uintptr_t workspace_ptr = info[7].As<Napi::Number>().Int64Value();
    size_t workspace_size = info[8].As<Napi::Number>().Int64Value();
    glm_linear(reinterpret_cast<GlmCtx*>(ctx_ptr),
               reinterpret_cast<void*>(out_ptr),
               reinterpret_cast<const void*>(in_ptr),
               reinterpret_cast<const void*>(wt_ptr),
               batch, n, k,
               reinterpret_cast<void*>(workspace_ptr), workspace_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("linear failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Layernorm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 8) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, weight, bias, eps, dim, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t wt_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t bias_ptr = info[4].As<Napi::Number>().Int64Value();
    float eps = info[5].As<Napi::Number>().FloatValue();
    int dim = info[6].As<Napi::Number>().Int32Value();
    int batch = info[7].As<Napi::Number>().Int32Value();
    glm_layernorm(reinterpret_cast<GlmCtx*>(ctx_ptr),
                  reinterpret_cast<void*>(out_ptr),
                  reinterpret_cast<const void*>(in_ptr),
                  reinterpret_cast<const void*>(wt_ptr),
                  reinterpret_cast<const void*>(bias_ptr),
                  eps, dim, batch);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("layernorm failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Relu(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, n)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    int n = info[3].As<Napi::Number>().Int32Value();
    glm_relu(reinterpret_cast<GlmCtx*>(ctx_ptr),
             reinterpret_cast<void*>(out_ptr),
             reinterpret_cast<const void*>(in_ptr), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("relu failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Sigmoid(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, n)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    int n = info[3].As<Napi::Number>().Int32Value();
    glm_sigmoid(reinterpret_cast<GlmCtx*>(ctx_ptr),
                reinterpret_cast<void*>(out_ptr),
                reinterpret_cast<const void*>(in_ptr), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("sigmoid failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Softmax(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, mask, dim, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t mask_ptr = info[3].IsNull() ? 0 : info[3].As<Napi::Number>().Int64Value();
    int dim = info[4].As<Napi::Number>().Int32Value();
    int batch = info[5].As<Napi::Number>().Int32Value();
    glm_softmax(reinterpret_cast<GlmCtx*>(ctx_ptr),
                reinterpret_cast<void*>(out_ptr),
                reinterpret_cast<const void*>(in_ptr),
                mask_ptr ? reinterpret_cast<const void*>(mask_ptr) : nullptr,
                dim, batch);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("softmax failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value IndexerScore(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 16) {
        Napi::TypeError::New(env, "Expected (ctx, out, q, kData, weights, pageIndices, pageIndptr, lastPageLen, qoIndptr, scale, totalQ, idxNHeads, idxHeadDim, pageSize, maxKvLen, causal)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t q_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t kData_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t weights_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t pageIndices_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t pageIndptr_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t lastPageLen_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t qoIndptr_ptr = info[8].As<Napi::Number>().Int64Value();
    float scale = info[9].As<Napi::Number>().FloatValue();
    int totalQ = info[10].As<Napi::Number>().Int32Value();
    int idxNHeads = info[11].As<Napi::Number>().Int32Value();
    int idxHeadDim = info[12].As<Napi::Number>().Int32Value();
    int pageSize = info[13].As<Napi::Number>().Int32Value();
    int maxKvLen = info[14].As<Napi::Number>().Int32Value();
    int causal = info[15].As<Napi::Number>().Int32Value();
    const int32_t* kv_token_indptr = info.Length() > 16
        ? reinterpret_cast<const int32_t*>(info[16].As<Napi::Number>().Int64Value())
        : nullptr;
    glm_indexer_score(reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(out_ptr),
        reinterpret_cast<const void*>(q_ptr),
        reinterpret_cast<const void*>(kData_ptr),
        reinterpret_cast<const void*>(weights_ptr),
        reinterpret_cast<const int32_t*>(pageIndices_ptr),
        reinterpret_cast<const int32_t*>(pageIndptr_ptr),
        reinterpret_cast<const int32_t*>(lastPageLen_ptr),
        reinterpret_cast<const int32_t*>(qoIndptr_ptr),
        scale, totalQ, idxNHeads, idxHeadDim, pageSize, maxKvLen, causal,
        kv_token_indptr);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("indexerScore failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value IndexerScoreTopkPrefill(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 31) {
        Napi::TypeError::New(env, "Expected 31 args").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_idx_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t out_scores_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t q_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t kData_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t weights_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t pageIndices_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t pageIndptr_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t lastPageLen_ptr = info[8].As<Napi::Number>().Int64Value();
    uintptr_t qoIndptr_ptr = info[9].As<Napi::Number>().Int64Value();
    float scale = info[10].As<Napi::Number>().FloatValue();
    int totalQ = info[11].As<Napi::Number>().Int32Value();
    int idxNHeads = info[12].As<Napi::Number>().Int32Value();
    int idxHeadDim = info[13].As<Napi::Number>().Int32Value();
    int pageSize = info[14].As<Napi::Number>().Int32Value();
    int topk = info[15].As<Napi::Number>().Int32Value();
    int causal = info[16].As<Napi::Number>().Int32Value();
    int qGlobalStart = info[17].As<Napi::Number>().Int32Value();
    const uint8_t* custom_mask = nullptr;
    const int32_t* mask_indptr = nullptr;
    const int32_t* mask_kv_len = nullptr;
    if (info.Length() >= 19 && info[18].IsNumber()) custom_mask = reinterpret_cast<const uint8_t*>(info[18].As<Napi::Number>().Int64Value());
    if (info.Length() >= 20 && info[19].IsNumber()) mask_indptr = reinterpret_cast<const int32_t*>(info[19].As<Napi::Number>().Int64Value());
    if (info.Length() >= 21 && info[20].IsNumber()) mask_kv_len = reinterpret_cast<const int32_t*>(info[20].As<Napi::Number>().Int64Value());
    uintptr_t scores_ptr = info[21].As<Napi::Number>().Int64Value();
    uintptr_t rowLen_ptr = info[22].As<Napi::Number>().Int64Value();
    int maxKv = info[23].As<Napi::Number>().Int32Value();
    uintptr_t coarseHist_ptr = info[24].As<Napi::Number>().Int64Value();
    uintptr_t fineHist_ptr = info[25].As<Napi::Number>().Int64Value();
    uintptr_t meta_ptr = info[26].As<Napi::Number>().Int64Value();
    int numSplits = info[27].As<Napi::Number>().Int32Value();
    int cpWorldSize = info[28].As<Napi::Number>().Int32Value();
    int cpRank = info[29].As<Napi::Number>().Int32Value();
    const int32_t* global_last_page_len = reinterpret_cast<const int32_t*>((uintptr_t)info[30].As<Napi::Number>().Int64Value());
    const int32_t* kv_token_indptr = info.Length() > 31
        ? reinterpret_cast<const int32_t*>((uintptr_t)info[31].As<Napi::Number>().Int64Value())
        : nullptr;
    glm_indexer_score_topk_prefill(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<int32_t*>(out_idx_ptr),
        reinterpret_cast<__nv_bfloat16*>(out_scores_ptr),
        reinterpret_cast<const void*>(q_ptr),
        reinterpret_cast<const void*>(kData_ptr),
        reinterpret_cast<const void*>(weights_ptr),
        reinterpret_cast<const int32_t*>(pageIndices_ptr),
        reinterpret_cast<const int32_t*>(pageIndptr_ptr),
        reinterpret_cast<const int32_t*>(lastPageLen_ptr),
        reinterpret_cast<const int32_t*>(qoIndptr_ptr),
        scale, totalQ, idxNHeads, idxHeadDim, pageSize, topk, causal,
        qGlobalStart, custom_mask, mask_indptr, mask_kv_len,
        reinterpret_cast<void*>(scores_ptr),
        reinterpret_cast<int32_t*>(rowLen_ptr),
        maxKv,
        reinterpret_cast<int32_t*>(coarseHist_ptr),
        reinterpret_cast<int32_t*>(fineHist_ptr),
        reinterpret_cast<int32_t*>(meta_ptr),
        numSplits, cpWorldSize, cpRank, global_last_page_len, kv_token_indptr);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("indexerScoreTopkPrefill failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value IndexerScoreTopkV2(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 30) {
        Napi::TypeError::New(env, "Expected 30 args (…, outScores, causal, qGlobalStart, customMask, maskIndptr, maskKvLen, scores, rowLen, hist, meta, maxKv, numSplits, cpWorldSize, cpRank, globalLastPageLen)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    glm_indexer_score_topk_v2(
        reinterpret_cast<GlmCtx*>((uintptr_t)info[0].As<Napi::Number>().Int64Value()),
        reinterpret_cast<int32_t*>((uintptr_t)info[1].As<Napi::Number>().Int64Value()),
        reinterpret_cast<__nv_bfloat16*>((uintptr_t)info[2].As<Napi::Number>().Int64Value()),
        reinterpret_cast<const void*>((uintptr_t)info[3].As<Napi::Number>().Int64Value()),
        reinterpret_cast<const void*>((uintptr_t)info[4].As<Napi::Number>().Int64Value()),
        reinterpret_cast<const void*>((uintptr_t)info[5].As<Napi::Number>().Int64Value()),
        reinterpret_cast<const int32_t*>((uintptr_t)info[6].As<Napi::Number>().Int64Value()),
        reinterpret_cast<const int32_t*>((uintptr_t)info[7].As<Napi::Number>().Int64Value()),
        reinterpret_cast<const int32_t*>((uintptr_t)info[8].As<Napi::Number>().Int64Value()),
        reinterpret_cast<const int32_t*>((uintptr_t)info[9].As<Napi::Number>().Int64Value()),
        info[10].As<Napi::Number>().FloatValue(),
        info[11].As<Napi::Number>().Int32Value(),
        info[12].As<Napi::Number>().Int32Value(),
        info[13].As<Napi::Number>().Int32Value(),
        info[14].As<Napi::Number>().Int32Value(),
        info[15].As<Napi::Number>().Int32Value(),
        info[16].As<Napi::Number>().Int32Value(),
        info[17].As<Napi::Number>().Int32Value(),
        reinterpret_cast<const uint8_t*>((uintptr_t)info[18].As<Napi::Number>().Int64Value()),
        reinterpret_cast<const int32_t*>((uintptr_t)info[19].As<Napi::Number>().Int64Value()),
        reinterpret_cast<const int32_t*>((uintptr_t)info[20].As<Napi::Number>().Int64Value()),
        reinterpret_cast<void*>((uintptr_t)info[21].As<Napi::Number>().Int64Value()),
        reinterpret_cast<int32_t*>((uintptr_t)info[22].As<Napi::Number>().Int64Value()),
        reinterpret_cast<int32_t*>((uintptr_t)info[23].As<Napi::Number>().Int64Value()),
        reinterpret_cast<int32_t*>((uintptr_t)info[24].As<Napi::Number>().Int64Value()),
        info[25].As<Napi::Number>().Int32Value(),
        info[26].As<Napi::Number>().Int32Value(),
        info[27].As<Napi::Number>().Int32Value(),
        info[28].As<Napi::Number>().Int32Value(),
        reinterpret_cast<const int32_t*>((uintptr_t)info[29].As<Napi::Number>().Int64Value()),
        info.Length() > 30
            ? reinterpret_cast<const int32_t*>((uintptr_t)info[30].As<Napi::Number>().Int64Value())
            : nullptr);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("indexerScoreTopkV2 failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value TopkToSlots(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 13) {
        Napi::TypeError::New(env, "Expected (ctx, slots, topkLength, topkIdx, pageIndices, pageIndptr, lastPageLen, batchIndices, numTokens, topk, pageSize, cpWorldSize, cpRank, kvTokenIndptr?)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t slots_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t topk_length_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t topk_idx_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t page_indices_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t page_indptr_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t last_page_len_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t batch_indices_ptr = info[7].As<Napi::Number>().Int64Value();
    int num_tokens = info[8].As<Napi::Number>().Int32Value();
    int topk = info[9].As<Napi::Number>().Int32Value();
    int page_size = info[10].As<Napi::Number>().Int32Value();
    uint32_t cp_world_size = info[11].As<Napi::Number>().Uint32Value();
    uint32_t cp_rank = info[12].As<Napi::Number>().Uint32Value();
    uintptr_t kv_token_indptr_ptr = info.Length() > 13 ? info[13].As<Napi::Number>().Int64Value() : 0;
    glm_topk_to_slots(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<int32_t*>(slots_ptr),
        reinterpret_cast<int32_t*>(topk_length_ptr),
        reinterpret_cast<const int32_t*>(topk_idx_ptr),
        reinterpret_cast<const int32_t*>(page_indices_ptr),
        reinterpret_cast<const int32_t*>(page_indptr_ptr),
        reinterpret_cast<const int32_t*>(last_page_len_ptr),
        reinterpret_cast<const int32_t*>(batch_indices_ptr),
        num_tokens, topk, page_size, cp_world_size, cp_rank,
        reinterpret_cast<const int32_t*>(kv_token_indptr_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("topkToSlots failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Fill(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4) {
        Napi::TypeError::New(env, "Expected (ctx, out, value, n)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    float value = info[2].As<Napi::Number>().FloatValue();
    int n = info[3].As<Napi::Number>().Int32Value();
    glm_fill(reinterpret_cast<GlmCtx*>(ctx_ptr),
             reinterpret_cast<void*>(out_ptr), value, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("fill failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Gather(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 8) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, indices, k, in_dim, batch, elem_size)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t idx_ptr = info[3].As<Napi::Number>().Int64Value();
    int k = info[4].As<Napi::Number>().Int32Value();
    int in_dim = info[5].As<Napi::Number>().Int32Value();
    int batch = info[6].As<Napi::Number>().Int32Value();
    int elem_size = info[7].As<Napi::Number>().Int32Value();
    glm_gather(reinterpret_cast<GlmCtx*>(ctx_ptr),
               reinterpret_cast<void*>(out_ptr),
               reinterpret_cast<const void*>(in_ptr),
               reinterpret_cast<const int*>(idx_ptr),
               k, in_dim, batch, elem_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("gather failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value ScatterScalar(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
        Napi::TypeError::New(env, "Expected (ctx, out, indices, value, k, out_dim, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t idx_ptr = info[2].As<Napi::Number>().Int64Value();
    float value = info[3].As<Napi::Number>().FloatValue();
    int k = info[4].As<Napi::Number>().Int32Value();
    int out_dim = info[5].As<Napi::Number>().Int32Value();
    int batch = info[6].As<Napi::Number>().Int32Value();
    glm_scatter_scalar(reinterpret_cast<GlmCtx*>(ctx_ptr),
                       reinterpret_cast<void*>(out_ptr),
                       reinterpret_cast<const int*>(idx_ptr),
                       value, k, out_dim, batch);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("scatterScalar failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Deinterleave(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 10) {
        Napi::TypeError::New(env, "Expected (ctx, out, in, world_size, max_total_len, page_indptr, kv_token_indptr, batch_size, page_size, D)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    int world_size = info[3].As<Napi::Number>().Int32Value();
    int max_total_len = info[4].As<Napi::Number>().Int32Value();
    uintptr_t page_indptr_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t kv_token_indptr_ptr = info[6].As<Napi::Number>().Int64Value();
    int batch_size = info[7].As<Napi::Number>().Int32Value();
    int page_size = info[8].As<Napi::Number>().Int32Value();
    int D = info[9].As<Napi::Number>().Int32Value();
    glm_deinterleave(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(out_ptr),
                     reinterpret_cast<const void*>(in_ptr),
                     world_size, max_total_len,
                     reinterpret_cast<const int32_t*>(page_indptr_ptr),
                     reinterpret_cast<const int32_t*>(kv_token_indptr_ptr),
                     batch_size, page_size, D);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("deinterleave failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value GatherPages(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 10) {
        Napi::TypeError::New(env, "Expected (ctx, out, in, page_indices, page_indptr, last_page_len, max_pages, batch_size, page_size, D)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t indices_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t page_indptr_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t last_page_len_ptr = info[5].As<Napi::Number>().Int64Value();
    int max_pages = info[6].As<Napi::Number>().Int32Value();
    int batch_size = info[7].As<Napi::Number>().Int32Value();
    int page_size = info[8].As<Napi::Number>().Int32Value();
    int D = info[9].As<Napi::Number>().Int32Value();
    glm_gather_pages(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(out_ptr),
                     reinterpret_cast<const void*>(in_ptr),
                     reinterpret_cast<const int32_t*>(indices_ptr),
                     reinterpret_cast<const int32_t*>(page_indptr_ptr),
                     reinterpret_cast<const int32_t*>(last_page_len_ptr),
                     max_pages, batch_size, page_size, D);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("gatherPages failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MaskedFill(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, mask, value, n)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t mask_ptr = info[3].As<Napi::Number>().Int64Value();
    float value = info[4].As<Napi::Number>().FloatValue();
    int n = info[5].As<Napi::Number>().Int32Value();
    glm_masked_fill(reinterpret_cast<GlmCtx*>(ctx_ptr),
                    reinterpret_cast<void*>(out_ptr),
                    reinterpret_cast<const void*>(in_ptr),
                    reinterpret_cast<const void*>(mask_ptr),
                    value, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("maskedFill failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value IndexAdd(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, indices, values, n_indices, dim)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t idx_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t val_ptr = info[3].As<Napi::Number>().Int64Value();
    int n_indices = info[4].As<Napi::Number>().Int32Value();
    int dim = info[5].As<Napi::Number>().Int32Value();
    glm_index_add(reinterpret_cast<GlmCtx*>(ctx_ptr),
                  reinterpret_cast<void*>(out_ptr),
                  reinterpret_cast<const int*>(idx_ptr),
                  reinterpret_cast<const void*>(val_ptr),
                  n_indices, dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("indexAdd failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value RotaryEmbedding(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 8) {
        Napi::TypeError::New(env, "Expected (ctx, cos_out, sin_out, inv_freq, position_ids, dim_half, batch, seq_len)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t cos_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t sin_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t inv_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t pos_ptr = info[4].As<Napi::Number>().Int64Value();
    int dim_half = info[5].As<Napi::Number>().Int32Value();
    int batch = info[6].As<Napi::Number>().Int32Value();
    int seq_len = info[7].As<Napi::Number>().Int32Value();
    glm_rotary_embedding(reinterpret_cast<GlmCtx*>(ctx_ptr),
                         reinterpret_cast<void*>(cos_ptr),
                         reinterpret_cast<void*>(sin_ptr),
                         reinterpret_cast<const void*>(inv_ptr),
                         reinterpret_cast<const int*>(pos_ptr),
                         dim_half, batch, seq_len);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("rotaryEmbedding failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value ApplyRotaryPosEmb(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 10) {
        Napi::TypeError::New(env, "Expected (ctx, out, x, cos, sin, rope_dim, n_heads, seq_len, batch, unsqueeze_dim[, interleaved])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t x_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t cos_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t sin_ptr = info[4].As<Napi::Number>().Int64Value();
    int rope_dim = info[5].As<Napi::Number>().Int32Value();
    int n_heads = info[6].As<Napi::Number>().Int32Value();
    int seq_len = info[7].As<Napi::Number>().Int32Value();
    int batch = info[8].As<Napi::Number>().Int32Value();
    int unsqueeze_dim = info[9].As<Napi::Number>().Int32Value();
    bool interleaved = info.Length() > 10 ? info[10].As<Napi::Boolean>().Value() : false;
    glm_apply_rotary_pos_emb(reinterpret_cast<GlmCtx*>(ctx_ptr),
                             reinterpret_cast<void*>(out_ptr),
                             reinterpret_cast<const void*>(x_ptr),
                             reinterpret_cast<const void*>(cos_ptr),
                             reinterpret_cast<const void*>(sin_ptr),
                             rope_dim, n_heads, seq_len, batch, unsqueeze_dim, interleaved);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("applyRotaryPosEmb failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value ApplyRotaryPosEmbPartial(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 11) {
        Napi::TypeError::New(env, "Expected (ctx, out, x, cos, sin, rope_dim, head_dim, n_heads, seq_len, batch, unsqueeze_dim[, interleaved])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t x_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t cos_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t sin_ptr = info[4].As<Napi::Number>().Int64Value();
    int rope_dim = info[5].As<Napi::Number>().Int32Value();
    int head_dim = info[6].As<Napi::Number>().Int32Value();
    int n_heads = info[7].As<Napi::Number>().Int32Value();
    int seq_len = info[8].As<Napi::Number>().Int32Value();
    int batch = info[9].As<Napi::Number>().Int32Value();
    int unsqueeze_dim = info[10].As<Napi::Number>().Int32Value();
    bool interleaved = info.Length() > 11 ? info[11].As<Napi::Boolean>().Value() : false;
    glm_apply_rotary_pos_emb_partial(reinterpret_cast<GlmCtx*>(ctx_ptr),
                                      reinterpret_cast<void*>(out_ptr),
                                      reinterpret_cast<const void*>(x_ptr),
                                      reinterpret_cast<const void*>(cos_ptr),
                                      reinterpret_cast<const void*>(sin_ptr),
                                      rope_dim, head_dim, n_heads, seq_len, batch, unsqueeze_dim, interleaved);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("applyRotaryPosEmbPartial failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value RopeTranspose(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 11) {
        Napi::TypeError::New(env, "Expected (ctx, out, in, cos, sin, rope_dim, head_dim, n_heads, seq_len, batch, in_stride[, interleaved])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t cos_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t sin_ptr = info[4].As<Napi::Number>().Int64Value();
    int rope_dim = info[5].As<Napi::Number>().Int32Value();
    int head_dim = info[6].As<Napi::Number>().Int32Value();
    int n_heads = info[7].As<Napi::Number>().Int32Value();
    int seq_len = info[8].As<Napi::Number>().Int32Value();
    int batch = info[9].As<Napi::Number>().Int32Value();
    int in_stride = info[10].As<Napi::Number>().Int32Value();
    bool interleaved = info.Length() > 11 ? info[11].As<Napi::Boolean>().Value() : false;
    glm_rope_transpose(reinterpret_cast<GlmCtx*>(ctx_ptr),
                        reinterpret_cast<void*>(out_ptr),
                        reinterpret_cast<const void*>(in_ptr),
                        reinterpret_cast<const void*>(cos_ptr),
                        reinterpret_cast<const void*>(sin_ptr),
                        rope_dim, head_dim, n_heads, seq_len, batch, in_stride, interleaved);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("ropeTranspose failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MlaVExpand(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 12) {
        Napi::TypeError::New(env, "Expected (ctx, result, attn_out, v_proj, kv_lora_rank, v_head_dim, n_heads, seq_len, batch, attn_n_heads, head_offset, v_proj_head_offset)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t result_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t attn_out_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t v_proj_ptr = info[3].As<Napi::Number>().Int64Value();
    int kv_lora_rank = info[4].As<Napi::Number>().Int32Value();
    int v_head_dim = info[5].As<Napi::Number>().Int32Value();
    int n_heads = info[6].As<Napi::Number>().Int32Value();
    int seq_len = info[7].As<Napi::Number>().Int32Value();
    int batch = info[8].As<Napi::Number>().Int32Value();
    int attn_n_heads = info[9].As<Napi::Number>().Int32Value();
    int head_offset = info[10].As<Napi::Number>().Int32Value();
    int v_proj_head_offset = info[11].As<Napi::Number>().Int32Value();
    glm_mla_v_expand(reinterpret_cast<GlmCtx*>(ctx_ptr),
                      reinterpret_cast<void*>(result_ptr),
                      reinterpret_cast<const void*>(attn_out_ptr),
                      reinterpret_cast<const void*>(v_proj_ptr),
                      kv_lora_rank, v_head_dim, n_heads, seq_len, batch,
                      attn_n_heads, head_offset, v_proj_head_offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("mlaVExpand failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Topk(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
        Napi::TypeError::New(env, "Expected (ctx, out_values, out_indices, input, k, dim, batch[, offset])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_vals_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t out_idxs_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[3].As<Napi::Number>().Int64Value();
    int k = info[4].As<Napi::Number>().Int32Value();
    int dim = info[5].As<Napi::Number>().Int32Value();
    int batch = info[6].As<Napi::Number>().Int32Value();
    int offset = info.Length() > 7 ? info[7].As<Napi::Number>().Int32Value() : 0;
    glm_topk(reinterpret_cast<GlmCtx*>(ctx_ptr),
             reinterpret_cast<void*>(out_vals_ptr),
             reinterpret_cast<int*>(out_idxs_ptr),
             reinterpret_cast<const void*>(in_ptr),
             k, dim, batch, offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("topk failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value TopkFromScores(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 11) {
        Napi::TypeError::New(env, "Expected (ctx, out_values, out_indices, scores, row_len, hist, meta, batch, stride, topk, num_splits[, cpWorldSize, cpRank])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_vals_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t out_idxs_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t scores_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t row_len_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t hist_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t meta_ptr = info[6].As<Napi::Number>().Int64Value();
    int batch = info[7].As<Napi::Number>().Int32Value();
    int stride = info[8].As<Napi::Number>().Int32Value();
    int topk = info[9].As<Napi::Number>().Int32Value();
    int num_splits = info[10].As<Napi::Number>().Int32Value();
    int cpWorldSize = info.Length() > 11 ? info[11].As<Napi::Number>().Int32Value() : 0;
    int cpRank = info.Length() > 12 ? info[12].As<Napi::Number>().Int32Value() : 0;
    glm_topk_from_scores(reinterpret_cast<GlmCtx*>(ctx_ptr),
              reinterpret_cast<int32_t*>(out_idxs_ptr),
              reinterpret_cast<__nv_bfloat16*>(out_vals_ptr),
              reinterpret_cast<const void*>(scores_ptr),
              reinterpret_cast<const int32_t*>(row_len_ptr),
              reinterpret_cast<int32_t*>(hist_ptr),
              reinterpret_cast<int32_t*>(meta_ptr),
              batch, stride, topk, num_splits,
              cpWorldSize, cpRank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("topkFromScores failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value SortTopkByIndex(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
        Napi::TypeError::New(env, "Expected (ctx, out_idx, out_scores, batch, topk)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    glm_sort_topk_by_index(
        reinterpret_cast<GlmCtx*>((uintptr_t)info[0].As<Napi::Number>().Int64Value()),
        reinterpret_cast<int32_t*>((uintptr_t)info[1].As<Napi::Number>().Int64Value()),
        reinterpret_cast<__nv_bfloat16*>((uintptr_t)info[2].As<Napi::Number>().Int64Value()),
        info[3].As<Napi::Number>().Int32Value(),
        info[4].As<Napi::Number>().Int32Value());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("sortTopkByIndex failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Bmm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 12) {
        Napi::TypeError::New(env, "Expected (ctx, C, A, B, alpha, beta, batch, M, N, K, transA, transB)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t c_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t a_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t b_ptr = info[3].As<Napi::Number>().Int64Value();
    float alpha = info[4].As<Napi::Number>().FloatValue();
    float beta = info[5].As<Napi::Number>().FloatValue();
    int batch = info[6].As<Napi::Number>().Int32Value();
    int M = info[7].As<Napi::Number>().Int32Value();
    int N = info[8].As<Napi::Number>().Int32Value();
    int K = info[9].As<Napi::Number>().Int32Value();
    int transA = info[10].As<Napi::Number>().Int32Value();
    int transB = info[11].As<Napi::Number>().Int32Value();
    glm_bmm(reinterpret_cast<GlmCtx*>(ctx_ptr),
            reinterpret_cast<void*>(c_ptr),
            reinterpret_cast<const void*>(a_ptr),
            reinterpret_cast<const void*>(b_ptr),
            alpha, beta, batch, M, N, K, transA, transB);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("bmm failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Scale(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, scale, n)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    float scale = info[3].As<Napi::Number>().FloatValue();
    int n = info[4].As<Napi::Number>().Int32Value();
    glm_scale(reinterpret_cast<GlmCtx*>(ctx_ptr),
              reinterpret_cast<void*>(out_ptr),
              reinterpret_cast<const void*>(in_ptr), scale, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("scale failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value SumPointers(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 13) {
        Napi::TypeError::New(env, "Expected (ctx, p0..p7, out, N, numel, dtype[, writeback])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t p[8];
    for (int i = 0; i < 8; i++) {
        p[i] = info[1 + i].As<Napi::Number>().Int64Value();
    }
    uintptr_t out_ptr = info[9].As<Napi::Number>().Int64Value();
    int N = info[10].As<Napi::Number>().Int32Value();
    int64_t numel = info[11].As<Napi::Number>().Int64Value();
    int dtype = info[12].As<Napi::Number>().Int32Value();
    bool writeback = false;
    if (info.Length() >= 14 && info[13].IsBoolean()) {
        writeback = info[13].As<Napi::Boolean>().Value();
    }
    glm_sum_pointers(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(p[0]),  reinterpret_cast<void*>(p[1]),
                     reinterpret_cast<void*>(p[2]),  reinterpret_cast<void*>(p[3]),
                     reinterpret_cast<void*>(p[4]),  reinterpret_cast<void*>(p[5]),
                     reinterpret_cast<void*>(p[6]),  reinterpret_cast<void*>(p[7]),
                     reinterpret_cast<void*>(out_ptr), N, numel, dtype, writeback);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("sumPointers failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value RmsnormPointersSmem(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 18) {
        Napi::TypeError::New(env, "Expected (ctx, p0..p7, inputA, weight, out, residual, N, numel, dim, eps, dtype)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t p[8];
    for (int i = 0; i < 8; i++) {
        p[i] = info[1 + i].As<Napi::Number>().Int64Value();
    }
    uintptr_t inputA_ptr = info[9].As<Napi::Number>().Int64Value();
    uintptr_t weight_ptr = info[10].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[11].As<Napi::Number>().Int64Value();
    uintptr_t res_ptr = info[12].As<Napi::Number>().Int64Value();
    int N = info[13].As<Napi::Number>().Int32Value();
    int64_t numel = info[14].As<Napi::Number>().Int64Value();
    int dim = info[15].As<Napi::Number>().Int32Value();
    float eps = info[16].As<Napi::Number>().FloatValue();
    int dtype = info[17].As<Napi::Number>().Int32Value();
    glm_rmsnorm_pointers_smem(reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<const void*>(p[0]),  reinterpret_cast<const void*>(p[1]),
        reinterpret_cast<const void*>(p[2]),  reinterpret_cast<const void*>(p[3]),
        reinterpret_cast<const void*>(p[4]),  reinterpret_cast<const void*>(p[5]),
        reinterpret_cast<const void*>(p[6]),  reinterpret_cast<const void*>(p[7]),
        reinterpret_cast<const void*>(inputA_ptr),
        reinterpret_cast<const void*>(weight_ptr),
        reinterpret_cast<void*>(out_ptr),
        reinterpret_cast<void*>(res_ptr),
        N, numel, dim, eps, dtype);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("rmsnormPointersSmem failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Add(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
        Napi::TypeError::New(env, "Expected (ctx, out, a, b, n)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t a_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t b_ptr = info[3].As<Napi::Number>().Int64Value();
    int n = info[4].As<Napi::Number>().Int32Value();
    glm_add(reinterpret_cast<GlmCtx*>(ctx_ptr),
            reinterpret_cast<void*>(out_ptr),
            reinterpret_cast<const void*>(a_ptr),
            reinterpret_cast<const void*>(b_ptr), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("add failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value AddBroadcast(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, a, b, dim, rows)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t a_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t b_ptr = info[3].As<Napi::Number>().Int64Value();
    int dim = info[4].As<Napi::Number>().Int32Value();
    int rows = info[5].As<Napi::Number>().Int32Value();
    glm_add_broadcast(reinterpret_cast<GlmCtx*>(ctx_ptr),
                       reinterpret_cast<void*>(out_ptr),
                       reinterpret_cast<const void*>(a_ptr),
                       reinterpret_cast<const void*>(b_ptr), dim, rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("addBroadcast failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Transpose4d(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 11) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, d0, d1, d2, d3, p0, p1, p2, p3)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    int d0 = info[3].As<Napi::Number>().Int32Value();
    int d1 = info[4].As<Napi::Number>().Int32Value();
    int d2 = info[5].As<Napi::Number>().Int32Value();
    int d3 = info[6].As<Napi::Number>().Int32Value();
    int p0 = info[7].As<Napi::Number>().Int32Value();
    int p1 = info[8].As<Napi::Number>().Int32Value();
    int p2 = info[9].As<Napi::Number>().Int32Value();
    int p3 = info[10].As<Napi::Number>().Int32Value();
    glm_transpose_4d(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(out_ptr),
                     reinterpret_cast<const void*>(in_ptr),
                     d0, d1, d2, d3, p0, p1, p2, p3);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("transpose4d failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Mul(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
        Napi::TypeError::New(env, "Expected (ctx, out, a, b, n)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t a_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t b_ptr = info[3].As<Napi::Number>().Int64Value();
    int n = info[4].As<Napi::Number>().Int32Value();
    glm_mul(reinterpret_cast<GlmCtx*>(ctx_ptr),
            reinterpret_cast<void*>(out_ptr),
            reinterpret_cast<const void*>(a_ptr),
            reinterpret_cast<const void*>(b_ptr),
            n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("mul failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MulBroadcast(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, a, b, dim, rows)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t a_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t b_ptr = info[3].As<Napi::Number>().Int64Value();
    int dim = info[4].As<Napi::Number>().Int32Value();
    int rows = info[5].As<Napi::Number>().Int32Value();
    glm_mul_broadcast(reinterpret_cast<GlmCtx*>(ctx_ptr),
                       reinterpret_cast<void*>(out_ptr),
                       reinterpret_cast<const void*>(a_ptr),
                       reinterpret_cast<const void*>(b_ptr), dim, rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("mulBroadcast failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value ReduceSum(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, rows, cols)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    int rows = info[3].As<Napi::Number>().Int32Value();
    int cols = info[4].As<Napi::Number>().Int32Value();
    glm_reduce_sum(reinterpret_cast<GlmCtx*>(ctx_ptr),
                   reinterpret_cast<void*>(out_ptr),
                   reinterpret_cast<const void*>(in_ptr),
                   rows, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("reduceSum failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value RowNormalize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, scale, rows, cols, normalize)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    float scale = info[3].As<Napi::Number>().FloatValue();
    int rows = info[4].As<Napi::Number>().Int32Value();
    int cols = info[5].As<Napi::Number>().Int32Value();
    bool normalize = info[6].As<Napi::Boolean>().Value();
    glm_row_normalize(reinterpret_cast<GlmCtx*>(ctx_ptr),
                       reinterpret_cast<void*>(out_ptr),
                       reinterpret_cast<const void*>(in_ptr),
                       scale, rows, cols, normalize);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("rowNormalize failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value GroupMaskMul(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
        Napi::TypeError::New(env, "Expected (ctx, scores, group_mask, num_experts, experts_per_group, n_group, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t scores_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t mask_ptr = info[2].As<Napi::Number>().Int64Value();
    int num_experts = info[3].As<Napi::Number>().Int32Value();
    int experts_per_group = info[4].As<Napi::Number>().Int32Value();
    int n_group = info[5].As<Napi::Number>().Int32Value();
    int batch = info[6].As<Napi::Number>().Int32Value();
    glm_group_mask_mul(reinterpret_cast<GlmCtx*>(ctx_ptr),
                        reinterpret_cast<void*>(scores_ptr),
                        reinterpret_cast<const void*>(mask_ptr),
                        num_experts, experts_per_group, n_group, batch);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("groupMaskMul failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MulMatId(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 8) {
        Napi::TypeError::New(env, "Expected (ctx, output, input, weight_ptrs, expert_ids, top_k, count, N, K)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t wptrs_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t eids_ptr = info[4].As<Napi::Number>().Int64Value();
    int top_k = info[5].As<Napi::Number>().Int32Value();
    int count = info[6].As<Napi::Number>().Int32Value();
    int N = info[7].As<Napi::Number>().Int32Value();
    int K = info[8].As<Napi::Number>().Int32Value();
    glm_mul_mat_id(reinterpret_cast<GlmCtx*>(ctx_ptr),
                    reinterpret_cast<void*>(out_ptr),
                    reinterpret_cast<const void*>(in_ptr),
                    reinterpret_cast<const void* const*>(wptrs_ptr),
                    reinterpret_cast<const int*>(eids_ptr),
                    top_k, count, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("mulMatId failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Nvfp4MulMatId(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 10) {
        Napi::TypeError::New(env, "Expected (ctx, output, input, weight_ptrs, scale_ptrs, scale2_ptrs, expert_ids, top_k, count, N, K)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t wptrs_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t sptrs_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t s2ptrs_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t eids_ptr = info[6].As<Napi::Number>().Int64Value();
    int top_k = info[7].As<Napi::Number>().Int32Value();
    int count = info[8].As<Napi::Number>().Int32Value();
    int N = info[9].As<Napi::Number>().Int32Value();
    int K = info[10].As<Napi::Number>().Int32Value();
    glm_nvfp4_mul_mat_id(reinterpret_cast<GlmCtx*>(ctx_ptr),
                          reinterpret_cast<void*>(out_ptr),
                          reinterpret_cast<const void*>(in_ptr),
                          reinterpret_cast<const void* const*>(wptrs_ptr),
                          reinterpret_cast<const void* const*>(sptrs_ptr),
                          reinterpret_cast<const void* const*>(s2ptrs_ptr),
                          reinterpret_cast<const int*>(eids_ptr),
                          top_k, count, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("nvfp4MulMatId failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value ScatterAddRows(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 8) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, scales, top_k, dim, num_rows, workspace)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t scales_ptr = info[3].As<Napi::Number>().Int64Value();
    int top_k = info[4].As<Napi::Number>().Int32Value();
    int dim = info[5].As<Napi::Number>().Int32Value();
    int num_rows = info[6].As<Napi::Number>().Int32Value();
    uintptr_t ws_ptr = info[7].As<Napi::Number>().Int64Value();
    glm_scatter_add_rows(reinterpret_cast<GlmCtx*>(ctx_ptr),
                          reinterpret_cast<void*>(out_ptr),
                          reinterpret_cast<const void*>(in_ptr),
                          reinterpret_cast<const void*>(scales_ptr),
                          top_k, dim, num_rows,
                          reinterpret_cast<void*>(ws_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("scatterAddRows failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MmaMoeWorkspaceSize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4) {
        Napi::TypeError::New(env, "Expected (count, N, K, num_experts)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    int count = info[0].As<Napi::Number>().Int32Value();
    int N = info[1].As<Napi::Number>().Int32Value();
    int K = info[2].As<Napi::Number>().Int32Value();
    int num_experts = info[3].As<Napi::Number>().Int32Value();
    size_t ws_size = glm_mma_moe_workspace_size(count, N, K, num_experts);
    return Napi::Number::New(env, static_cast<double>(ws_size));
}

static Napi::Value MmaMoeCoopWorkspaceSize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4) {
        Napi::TypeError::New(env, "Expected (count, N, K, num_experts)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    int count = info[0].As<Napi::Number>().Int32Value();
    int N = info[1].As<Napi::Number>().Int32Value();
    int K = info[2].As<Napi::Number>().Int32Value();
    int num_experts = info[3].As<Napi::Number>().Int32Value();
    size_t ws_size = glm_mma_moe_coop_workspace_size(count, N, K, num_experts);
    return Napi::Number::New(env, static_cast<double>(ws_size));
}

static Napi::Value Nvfp4MulMatIdGroupedMmaCoop(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 13) {
        Napi::TypeError::New(env, "Expected (ctx, output, input, weight_ptrs, scale_ptrs, scale2_ptrs, expert_ids, top_k, count, N, K, num_experts, workspace)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t wptrs_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t sptrs_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t s2ptrs_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t eids_ptr = info[6].As<Napi::Number>().Int64Value();
    int top_k = info[7].As<Napi::Number>().Int32Value();
    int count = info[8].As<Napi::Number>().Int32Value();
    int N = info[9].As<Napi::Number>().Int32Value();
    int K = info[10].As<Napi::Number>().Int32Value();
    int num_experts = info[11].As<Napi::Number>().Int32Value();
    uintptr_t ws_ptr = info[12].As<Napi::Number>().Int64Value();
    glm_nvfp4_mul_mat_id_grouped_mma_coop(reinterpret_cast<GlmCtx*>(ctx_ptr),
                                           reinterpret_cast<void*>(out_ptr),
                                           reinterpret_cast<const void*>(in_ptr),
                                           reinterpret_cast<const void* const*>(wptrs_ptr),
                                           reinterpret_cast<const void* const*>(sptrs_ptr),
                                           reinterpret_cast<const void* const*>(s2ptrs_ptr),
                                           reinterpret_cast<const int*>(eids_ptr),
                                           top_k, count, N, K, num_experts,
                                           reinterpret_cast<void*>(ws_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("nvfp4MulMatIdGroupedMmaCoop failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MmaMoeCoopScatterWorkspaceSize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (count, K, num_experts)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    int count = info[0].As<Napi::Number>().Int32Value();
    int K = info[1].As<Napi::Number>().Int32Value();
    int num_experts = info[2].As<Napi::Number>().Int32Value();
    size_t ws_size = glm_mma_moe_coop_scatter_workspace_size(count, K, num_experts);
    return Napi::Number::New(env, static_cast<double>(ws_size));
}

static Napi::Value MmaMoeCoopGemmWorkspaceSize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (count, N)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    int count = info[0].As<Napi::Number>().Int32Value();
    int N = info[1].As<Napi::Number>().Int32Value();
    size_t ws_size = glm_mma_moe_coop_gemm_workspace_size(count, N);
    return Napi::Number::New(env, static_cast<double>(ws_size));
}

static Napi::Value MmaMoeCoopScatter(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 8) {
        Napi::TypeError::New(env, "Expected (ctx, input, expert_ids, top_k, count, K, num_experts, workspace)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t eids_ptr = info[2].As<Napi::Number>().Int64Value();
    int top_k = info[3].As<Napi::Number>().Int32Value();
    int count = info[4].As<Napi::Number>().Int32Value();
    int K = info[5].As<Napi::Number>().Int32Value();
    int num_experts = info[6].As<Napi::Number>().Int32Value();
    uintptr_t ws_ptr = info[7].As<Napi::Number>().Int64Value();
    glm_mma_moe_coop_scatter(reinterpret_cast<GlmCtx*>(ctx_ptr),
                             reinterpret_cast<const void*>(in_ptr),
                             reinterpret_cast<const int*>(eids_ptr),
                             top_k, count, K, num_experts,
                             reinterpret_cast<void*>(ws_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("mmaMoeCoopScatter failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MmaMoeCoopGemm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 11) {
        Napi::TypeError::New(env, "Expected (ctx, weight_ptrs, scale_ptrs, scale2_ptrs, num_experts, N, K, count, scatter_workspace, gemm_workspace, output)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t wptrs_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t sptrs_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t s2ptrs_ptr = info[3].As<Napi::Number>().Int64Value();
    int num_experts = info[4].As<Napi::Number>().Int32Value();
    int N = info[5].As<Napi::Number>().Int32Value();
    int K = info[6].As<Napi::Number>().Int32Value();
    int count = info[7].As<Napi::Number>().Int32Value();
    uintptr_t scatter_ws_ptr = info[8].As<Napi::Number>().Int64Value();
    uintptr_t gemm_ws_ptr = info[9].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[10].As<Napi::Number>().Int64Value();
    glm_mma_moe_coop_gemm(reinterpret_cast<GlmCtx*>(ctx_ptr),
                          reinterpret_cast<const void* const*>(wptrs_ptr),
                          reinterpret_cast<const void* const*>(sptrs_ptr),
                          reinterpret_cast<const void* const*>(s2ptrs_ptr),
                          num_experts, N, K, count,
                          reinterpret_cast<const void*>(scatter_ws_ptr),
                          reinterpret_cast<void*>(gemm_ws_ptr),
                          reinterpret_cast<void*>(out_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("mmaMoeCoopGemm failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Bf16MulMatIdGroupedMma(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 11) {
        Napi::TypeError::New(env, "Expected (ctx, output, input, weight_ptrs, expert_ids, top_k, count, N, K, num_experts, workspace)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t wptrs_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t eids_ptr = info[4].As<Napi::Number>().Int64Value();
    int top_k = info[5].As<Napi::Number>().Int32Value();
    int count = info[6].As<Napi::Number>().Int32Value();
    int N = info[7].As<Napi::Number>().Int32Value();
    int K = info[8].As<Napi::Number>().Int32Value();
    int num_experts = info[9].As<Napi::Number>().Int32Value();
    uintptr_t ws_ptr = info[10].As<Napi::Number>().Int64Value();
    glm_bf16_mul_mat_id_grouped_mma(reinterpret_cast<GlmCtx*>(ctx_ptr),
                                       reinterpret_cast<void*>(out_ptr),
                                       reinterpret_cast<const void*>(in_ptr),
                                       reinterpret_cast<const void* const*>(wptrs_ptr),
                                       reinterpret_cast<const int*>(eids_ptr),
                                       top_k, count, N, K, num_experts,
                                       reinterpret_cast<void*>(ws_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("bf16MulMatIdGroupedMma failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value IndexSelect(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, src, indices, dim, k[, offset])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t src_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t idx_ptr = info[3].As<Napi::Number>().Int64Value();
    int dim = info[4].As<Napi::Number>().Int32Value();
    int k = info[5].As<Napi::Number>().Int32Value();
    int offset = (info.Length() >= 7) ? info[6].As<Napi::Number>().Int32Value() : 0;
    glm_index_select(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(out_ptr),
                     reinterpret_cast<const void*>(src_ptr),
                     reinterpret_cast<const void*>(idx_ptr),
                     dim, k, offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("indexSelect failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Arange(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
        Napi::TypeError::New(env, "Expected (ctx, out, start, step, count)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    int start = info[2].As<Napi::Number>().Int32Value();
    int step = info[3].As<Napi::Number>().Int32Value();
    int count = info[4].As<Napi::Number>().Int32Value();
    glm_arange(reinterpret_cast<GlmCtx*>(ctx_ptr),
               reinterpret_cast<int*>(out_ptr), start, step, count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("arange failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Max(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out_values, out_indices, input, dim, batch[, offset])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_vals_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t out_idxs_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[3].As<Napi::Number>().Int64Value();
    int dim = info[4].As<Napi::Number>().Int32Value();
    int batch = info[5].As<Napi::Number>().Int32Value();
    int offset = info.Length() > 6 ? info[6].As<Napi::Number>().Int32Value() : 0;
    glm_max(reinterpret_cast<GlmCtx*>(ctx_ptr),
            reinterpret_cast<void*>(out_vals_ptr),
            reinterpret_cast<int*>(out_idxs_ptr),
            reinterpret_cast<const void*>(in_ptr), dim, batch, offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("max failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value KvCacheWrite(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 14) {
        Napi::TypeError::New(env, "Expected (ctx, src_k, src_v, dst_k, dst_v, slot_mapping, batch_size, n_kv, hd, page_size, src_k_token_stride, src_k_head_stride, src_v_token_stride, src_v_head_stride)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t src_k_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t src_v_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t dst_k_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t dst_v_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t slot_ptr = info[5].As<Napi::Number>().Int64Value();
    uint32_t batch_size = info[6].As<Napi::Number>().Uint32Value();
    uint32_t n_kv = info[7].As<Napi::Number>().Uint32Value();
    uint32_t hd = info[8].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[9].As<Napi::Number>().Uint32Value();
    uint32_t src_k_token_stride = info[10].As<Napi::Number>().Uint32Value();
    uint32_t src_k_head_stride = info[11].As<Napi::Number>().Uint32Value();
    uint32_t src_v_token_stride = info[12].As<Napi::Number>().Uint32Value();
    uint32_t src_v_head_stride = info[13].As<Napi::Number>().Uint32Value();
    glm_kv_cache_write(reinterpret_cast<GlmCtx*>(ctx_ptr),
                        reinterpret_cast<void*>(src_k_ptr),
                        reinterpret_cast<void*>(src_v_ptr),
                        reinterpret_cast<void*>(dst_k_ptr),
                        reinterpret_cast<void*>(dst_v_ptr),
                        reinterpret_cast<int32_t*>(slot_ptr),
                        batch_size, n_kv, hd, page_size,
                        src_k_token_stride, src_k_head_stride,
                        src_v_token_stride, src_v_head_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("kvCacheWrite failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Memcpy(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
        Napi::TypeError::New(env, "Expected (ctx, dst, src, bytes, kind)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t dst_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t src_ptr = info[2].As<Napi::Number>().Int64Value();
    size_t bytes = info[3].As<Napi::Number>().Int64Value();
    int kind = info[4].As<Napi::Number>().Int32Value();
    glm_memcpy(reinterpret_cast<GlmCtx*>(ctx_ptr),
               reinterpret_cast<void*>(dst_ptr),
               reinterpret_cast<const void*>(src_ptr), bytes, kind);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("memcpy failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Synchronize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (ctx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    glm_synchronize(reinterpret_cast<GlmCtx*>(ctx_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("synchronize failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value SynchronizeStream(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[1].IsNumber()) {
        Napi::TypeError::New(env, "Expected (ctx, stream_idx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    int stream_idx = info[1].As<Napi::Number>().Int32Value();
    glm_synchronize_stream(reinterpret_cast<GlmCtx*>(ctx_ptr), stream_idx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("synchronizeStream failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

class SynchronizeWorker : public Napi::AsyncWorker {
public:
    SynchronizeWorker(Napi::Promise::Deferred deferred, GlmCtx* ctx, int stream_idx)
        : Napi::AsyncWorker(deferred.Env()), deferred_(deferred), ctx_(ctx), stream_idx_(stream_idx) {}

    void Execute() override {
        glm_synchronize_stream(ctx_, stream_idx_);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            SetError(std::string("synchronizeStreamAsync failed: ") + cudaGetErrorString(err));
        }
    }

    void OnOK() override {
        deferred_.Resolve(Env().Undefined());
    }

    void OnError(const Napi::Error& error) override {
        deferred_.Reject(error.Value());
    }

private:
    Napi::Promise::Deferred deferred_;
    GlmCtx* ctx_;
    int stream_idx_;
};

static Napi::Value SynchronizeAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "Expected (ctx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    GlmCtx* ctx = reinterpret_cast<GlmCtx*>(ctx_ptr);
    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
    (new SynchronizeWorker(deferred, ctx, ctx->active_stream))->Queue();
    return deferred.Promise();
}

static Napi::Value SynchronizeStreamAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
        Napi::TypeError::New(env, "Expected (ctx, stream_idx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    int stream_idx = info[1].As<Napi::Number>().Int32Value();
    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
    (new SynchronizeWorker(deferred, reinterpret_cast<GlmCtx*>(ctx_ptr), stream_idx))->Queue();
    return deferred.Promise();
}

static Napi::Value SetStream(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[1].IsNumber()) {
        Napi::TypeError::New(env, "Expected (ctx, stream_idx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    int stream_idx = info[1].As<Napi::Number>().Int32Value();
    glm_set_stream(reinterpret_cast<GlmCtx*>(ctx_ptr), stream_idx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("setStream failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value EventRecord(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[1].IsNumber() || !info[2].IsNumber()) {
        Napi::TypeError::New(env, "Expected (ctx, event_idx, stream_idx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    int event_idx = info[1].As<Napi::Number>().Int32Value();
    int stream_idx = info[2].As<Napi::Number>().Int32Value();
    glm_event_record(reinterpret_cast<GlmCtx*>(ctx_ptr), event_idx, stream_idx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("eventRecord failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value StreamWaitEvent(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[1].IsNumber() || !info[2].IsNumber()) {
        Napi::TypeError::New(env, "Expected (ctx, stream_idx, event_idx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    int stream_idx = info[1].As<Napi::Number>().Int32Value();
    int event_idx = info[2].As<Napi::Number>().Int32Value();
    glm_stream_wait_event(reinterpret_cast<GlmCtx*>(ctx_ptr), stream_idx, event_idx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("streamWaitEvent failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MmapOpen(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected (path)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string path = info[0].As<Napi::String>().Utf8Value();
    void* ptr = glm_mmap_open(path.c_str());
    if (!ptr) {
        Napi::Error::New(env, "glm_mmap_open failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    return Napi::Number::New(env, reinterpret_cast<uintptr_t>(ptr));
}

// ---------------------------------------------------------------------------
// Async H2D workers (mmapLoad, memcpyHostToDevice, memcpy2dHostToDevice)
// ---------------------------------------------------------------------------

class MmapLoadWorker : public Napi::AsyncWorker {
public:
    MmapLoadWorker(Napi::Promise::Deferred deferred, GlmCtx* ctx,
                   void* gpu_dst, const void* mmap_ptr, uint64_t offset, uint64_t nbytes)
        : Napi::AsyncWorker(deferred.Env()),
          deferred_(deferred), ctx_(ctx),
          gpu_dst_(gpu_dst), mmap_ptr_(mmap_ptr), offset_(offset), nbytes_(nbytes) {}

    void Execute() override {
        glm_mmap_load(ctx_, gpu_dst_, mmap_ptr_, offset_, nbytes_);
    }

    void OnOK() override {
        deferred_.Resolve(Env().Undefined());
    }

    void OnError(const Napi::Error& error) override {
        deferred_.Reject(error.Value());
    }

private:
    Napi::Promise::Deferred deferred_;
    GlmCtx* ctx_;
    void* gpu_dst_;
    const void* mmap_ptr_;
    uint64_t offset_;
    uint64_t nbytes_;
};

class Memcpy2dHostToDeviceWorker : public Napi::AsyncWorker {
public:
    Memcpy2dHostToDeviceWorker(Napi::Promise::Deferred deferred, GlmCtx* ctx,
                               void* dst, size_t dpitch, const void* src, size_t spitch,
                               size_t width, size_t height)
        : Napi::AsyncWorker(deferred.Env()),
          deferred_(deferred), ctx_(ctx),
          dst_(dst), dpitch_(dpitch), src_(src), spitch_(spitch),
          width_(width), height_(height) {}

    void Execute() override {
        glm_memcpy2d(ctx_, dst_, dpitch_, src_, spitch_, width_, height_,
                     cudaMemcpyHostToDevice);
    }

    void OnOK() override {
        deferred_.Resolve(Env().Undefined());
    }

    void OnError(const Napi::Error& error) override {
        deferred_.Reject(error.Value());
    }

private:
    Napi::Promise::Deferred deferred_;
    GlmCtx* ctx_;
    void* dst_;
    size_t dpitch_;
    const void* src_;
    size_t spitch_;
    size_t width_;
    size_t height_;
};

static Napi::Value MmapLoadAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
        Napi::TypeError::New(env, "Expected (ctx, gpu_dst, mmap_ptr, offset, nbytes)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t gpu_dst = info[1].As<Napi::Number>().Int64Value();
    uintptr_t mmap_ptr = info[2].As<Napi::Number>().Int64Value();
    uint64_t offset = info[3].As<Napi::Number>().Int64Value();
    uint64_t nbytes = info[4].As<Napi::Number>().Int64Value();

    GlmCtx* ctx = reinterpret_cast<GlmCtx*>(ctx_ptr);

    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
    auto worker = new MmapLoadWorker(deferred, ctx,
        reinterpret_cast<void*>(gpu_dst), reinterpret_cast<const void*>(mmap_ptr),
        offset, nbytes);
    worker->Queue();
    return deferred.Promise();
}

static Napi::Value Memcpy2dHostToDeviceAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
        Napi::TypeError::New(env, "Expected (ctx, dst, dpitch, src, spitch, width, height)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t dst_ptr = info[1].As<Napi::Number>().Int64Value();
    size_t dpitch = info[2].As<Napi::Number>().Int64Value();
    uintptr_t src_ptr = info[3].As<Napi::Number>().Int64Value();
    size_t spitch = info[4].As<Napi::Number>().Int64Value();
    size_t width = info[5].As<Napi::Number>().Int64Value();
    size_t height = info[6].As<Napi::Number>().Int64Value();

    GlmCtx* ctx = reinterpret_cast<GlmCtx*>(ctx_ptr);

    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
    auto worker = new Memcpy2dHostToDeviceWorker(deferred, ctx,
        reinterpret_cast<void*>(dst_ptr), dpitch,
        reinterpret_cast<const void*>(src_ptr), spitch,
        width, height);
    worker->Queue();
    return deferred.Promise();
}

static Napi::Value MmapClose(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (mmap_ptr, size)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t mmap_ptr = info[0].As<Napi::Number>().Int64Value();
    uint64_t size = info[1].As<Napi::Number>().Int64Value();
    glm_mmap_close(reinterpret_cast<void*>(mmap_ptr), size);
    return env.Undefined();
}

static Napi::Value AllocPinned(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (bytes)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    size_t bytes = info[0].As<Napi::Number>().Int64Value();
    void* ptr = glm_alloc_pinned(bytes);
    return Napi::Number::New(env, reinterpret_cast<uintptr_t>(ptr));
}

static Napi::Value FreePinned(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (ptr)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ptr = info[0].As<Napi::Number>().Int64Value();
    glm_free_pinned(reinterpret_cast<void*>(ptr));
    return env.Undefined();
}

static Napi::Value HostPointerToBuffer(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ptr, size)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ptr = info[0].As<Napi::Number>().Int64Value();
    size_t size = info[1].As<Napi::Number>().Int64Value();
    return Napi::Buffer<uint8_t>::New(env, reinterpret_cast<uint8_t*>(ptr), size);
}

static Napi::Value BatchDecodePlan(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 14) {
        Napi::TypeError::New(env, "Expected (ctx, float_ws, float_ws_size, int_ws, pinned_int_ws, int_ws_size, plan_info, indptr_h, batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, enable_cuda_graph)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t float_ws = info[1].As<Napi::Number>().Int64Value();
    size_t float_ws_size = info[2].As<Napi::Number>().Int64Value();
    uintptr_t int_ws = info[3].As<Napi::Number>().Int64Value();
    uintptr_t pinned_int_ws = info[4].As<Napi::Number>().Int64Value();
    size_t int_ws_size = info[5].As<Napi::Number>().Int64Value();
    uintptr_t plan_info_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t indptr_h_ptr = info[7].As<Napi::Number>().Int64Value();
    uint32_t batch_size = info[8].As<Napi::Number>().Uint32Value();
    uint32_t num_qo_heads = info[9].As<Napi::Number>().Uint32Value();
    uint32_t num_kv_heads = info[10].As<Napi::Number>().Uint32Value();
    uint32_t head_dim = info[11].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[12].As<Napi::Number>().Uint32Value();
    bool enable_cuda_graph = info[13].As<Napi::Boolean>().Value();
    glm_batch_decode_plan(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(float_ws), float_ws_size,
        reinterpret_cast<void*>(int_ws), reinterpret_cast<void*>(pinned_int_ws), int_ws_size,
        reinterpret_cast<int64_t*>(plan_info_ptr),
        reinterpret_cast<int32_t*>(indptr_h_ptr),
        batch_size, num_qo_heads, num_kv_heads, head_dim, page_size,
        enable_cuda_graph);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("batchDecodePlan failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value BatchDecodeRun(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 17) {
        Napi::TypeError::New(env, "Expected (ctx, q, o, k_data, v_data, indices, indptr_d, last_page_len, float_ws, int_ws, plan_info, batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, sm_scale)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t q_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t o_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t k_data_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t v_data_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t indices_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t indptr_d_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t last_page_len_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t float_ws_ptr = info[8].As<Napi::Number>().Int64Value();
    uintptr_t int_ws_ptr = info[9].As<Napi::Number>().Int64Value();
    uintptr_t plan_info_ptr = info[10].As<Napi::Number>().Int64Value();
    uint32_t batch_size = info[11].As<Napi::Number>().Uint32Value();
    uint32_t num_qo_heads = info[12].As<Napi::Number>().Uint32Value();
    uint32_t num_kv_heads = info[13].As<Napi::Number>().Uint32Value();
    uint32_t head_dim = info[14].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[15].As<Napi::Number>().Uint32Value();
    float sm_scale = info[16].As<Napi::Number>().FloatValue();
    glm_batch_decode_run(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(q_ptr), reinterpret_cast<void*>(o_ptr),
        reinterpret_cast<void*>(k_data_ptr), reinterpret_cast<void*>(v_data_ptr),
        reinterpret_cast<int32_t*>(indices_ptr),
        reinterpret_cast<int32_t*>(indptr_d_ptr),
        reinterpret_cast<int32_t*>(last_page_len_ptr),
        reinterpret_cast<void*>(float_ws_ptr), reinterpret_cast<void*>(int_ws_ptr),
        reinterpret_cast<int64_t*>(plan_info_ptr),
        batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, sm_scale);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Napi::Error::New(env, std::string("batchDecodeRun failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
        }
    }
    return env.Undefined();
}

static Napi::Value BatchPrefillPagedPlan(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 16) {
        Napi::TypeError::New(env, "Expected (ctx, float_ws, float_ws_size, int_ws, pinned_int_ws, int_ws_size, plan_info, qo_indptr_h, paged_kv_indptr_h, total_qo_rows, batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, mask_mode)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t float_ws = info[1].As<Napi::Number>().Int64Value();
    size_t float_ws_size = info[2].As<Napi::Number>().Int64Value();
    uintptr_t int_ws = info[3].As<Napi::Number>().Int64Value();
    uintptr_t pinned_int_ws = info[4].As<Napi::Number>().Int64Value();
    size_t int_ws_size = info[5].As<Napi::Number>().Int64Value();
    uintptr_t plan_info_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t qo_indptr_h_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t paged_kv_indptr_h_ptr = info[8].As<Napi::Number>().Int64Value();
    uint32_t total_qo_rows = info[9].As<Napi::Number>().Uint32Value();
    uint32_t batch_size = info[10].As<Napi::Number>().Uint32Value();
    uint32_t num_qo_heads = info[11].As<Napi::Number>().Uint32Value();
    uint32_t num_kv_heads = info[12].As<Napi::Number>().Uint32Value();
    uint32_t head_dim = info[13].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[14].As<Napi::Number>().Uint32Value();
    int mask_mode = info[15].As<Napi::Number>().Int32Value();
    glm_batch_prefill_paged_plan(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(float_ws), float_ws_size,
        reinterpret_cast<void*>(int_ws), reinterpret_cast<void*>(pinned_int_ws), int_ws_size,
        reinterpret_cast<int64_t*>(plan_info_ptr),
        reinterpret_cast<int32_t*>(qo_indptr_h_ptr),
        reinterpret_cast<int32_t*>(paged_kv_indptr_h_ptr),
        total_qo_rows, batch_size,
        num_qo_heads, num_kv_heads, head_dim, page_size, mask_mode);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Napi::Error::New(env, std::string("batchPrefillPagedPlan failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
        }
    }
    return env.Undefined();
}

static Napi::Value BatchPrefillPagedRun(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 21) {
        Napi::TypeError::New(env, "Expected (ctx, q, o, k_data, v_data, indices, indptr_d, last_page_len, float_ws, int_ws, q_indptr_d, plan_info, total_qo_rows, batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, q_stride_n, q_stride_h, mask_mode, sm_scale)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t q_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t o_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t k_data_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t v_data_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t indices_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t indptr_d_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t last_page_len_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t float_ws_ptr = info[8].As<Napi::Number>().Int64Value();
    uintptr_t int_ws_ptr = info[9].As<Napi::Number>().Int64Value();
    uintptr_t q_indptr_d_ptr = info[10].As<Napi::Number>().Int64Value();
    uintptr_t plan_info_ptr = info[11].As<Napi::Number>().Int64Value();
    uint32_t total_qo_rows = info[12].As<Napi::Number>().Uint32Value();
    uint32_t batch_size = info[13].As<Napi::Number>().Uint32Value();
    uint32_t num_qo_heads = info[14].As<Napi::Number>().Uint32Value();
    uint32_t num_kv_heads = info[15].As<Napi::Number>().Uint32Value();
    uint32_t head_dim = info[16].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[17].As<Napi::Number>().Uint32Value();
    int32_t q_stride_n = info[18].As<Napi::Number>().Int32Value();
    int32_t q_stride_h = info[19].As<Napi::Number>().Int32Value();
    int mask_mode = info[20].As<Napi::Number>().Int32Value();
    float sm_scale = info[21].As<Napi::Number>().FloatValue();
    glm_batch_prefill_paged_run(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(q_ptr), reinterpret_cast<void*>(o_ptr),
        reinterpret_cast<void*>(k_data_ptr), reinterpret_cast<void*>(v_data_ptr),
        reinterpret_cast<int32_t*>(indices_ptr),
        reinterpret_cast<int32_t*>(indptr_d_ptr),
        reinterpret_cast<int32_t*>(last_page_len_ptr),
        reinterpret_cast<void*>(float_ws_ptr), reinterpret_cast<void*>(int_ws_ptr),
        reinterpret_cast<int32_t*>(q_indptr_d_ptr),
        reinterpret_cast<int64_t*>(plan_info_ptr),
        total_qo_rows, batch_size,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        q_stride_n, q_stride_h,
        mask_mode, sm_scale);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Napi::Error::New(env, std::string("batchPrefillPagedRun failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
        }
    }
    return env.Undefined();
}

static Napi::Value BatchPrefillRaggedPlan(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 15) {
        Napi::TypeError::New(env, "Expected (ctx, float_ws, float_ws_size, int_ws, pinned_int_ws, int_ws_size, plan_info, qo_indptr_h, kv_indptr_h, total_qo_rows, batch_size, num_qo_heads, num_kv_heads, head_dim, mask_mode)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t float_ws = info[1].As<Napi::Number>().Int64Value();
    size_t float_ws_size = info[2].As<Napi::Number>().Int64Value();
    uintptr_t int_ws = info[3].As<Napi::Number>().Int64Value();
    uintptr_t pinned_int_ws = info[4].As<Napi::Number>().Int64Value();
    size_t int_ws_size = info[5].As<Napi::Number>().Int64Value();
    uintptr_t plan_info_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t qo_indptr_h_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t kv_indptr_h_ptr = info[8].As<Napi::Number>().Int64Value();
    uint32_t total_qo_rows = info[9].As<Napi::Number>().Uint32Value();
    uint32_t batch_size = info[10].As<Napi::Number>().Uint32Value();
    uint32_t num_qo_heads = info[11].As<Napi::Number>().Uint32Value();
    uint32_t num_kv_heads = info[12].As<Napi::Number>().Uint32Value();
    uint32_t head_dim = info[13].As<Napi::Number>().Uint32Value();
    int mask_mode = info[14].As<Napi::Number>().Int32Value();
    glm_batch_prefill_ragged_plan(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(float_ws), float_ws_size,
        reinterpret_cast<void*>(int_ws), reinterpret_cast<void*>(pinned_int_ws), int_ws_size,
        reinterpret_cast<int64_t*>(plan_info_ptr),
        reinterpret_cast<int32_t*>(qo_indptr_h_ptr),
        reinterpret_cast<int32_t*>(kv_indptr_h_ptr),
        total_qo_rows, batch_size,
        num_qo_heads, num_kv_heads, head_dim, mask_mode);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Napi::Error::New(env, std::string("batchPrefillRaggedPlan failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
        }
    }
    return env.Undefined();
}

static Napi::Value BatchPrefillRaggedRun(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 23) {
        Napi::TypeError::New(env, "Expected (ctx, q, k, v, o, float_ws, int_ws, q_indptr_d, kv_indptr_d, plan_info, total_qo_rows, batch_size, num_qo_heads, num_kv_heads, head_dim, q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, v_stride_n, v_stride_h, mask_mode, sm_scale)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t q_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t k_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t v_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t o_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t float_ws_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t int_ws_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t q_indptr_d_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t kv_indptr_d_ptr = info[8].As<Napi::Number>().Int64Value();
    uintptr_t plan_info_ptr = info[9].As<Napi::Number>().Int64Value();
    uint32_t total_qo_rows = info[10].As<Napi::Number>().Uint32Value();
    uint32_t batch_size = info[11].As<Napi::Number>().Uint32Value();
    uint32_t num_qo_heads = info[12].As<Napi::Number>().Uint32Value();
    uint32_t num_kv_heads = info[13].As<Napi::Number>().Uint32Value();
    uint32_t head_dim = info[14].As<Napi::Number>().Uint32Value();
    int32_t q_stride_n = info[15].As<Napi::Number>().Int32Value();
    int32_t q_stride_h = info[16].As<Napi::Number>().Int32Value();
    int32_t kv_stride_n = info[17].As<Napi::Number>().Int32Value();
    int32_t kv_stride_h = info[18].As<Napi::Number>().Int32Value();
    int32_t v_stride_n = info[19].As<Napi::Number>().Int32Value();
    int32_t v_stride_h = info[20].As<Napi::Number>().Int32Value();
    int mask_mode = info[21].As<Napi::Number>().Int32Value();
    float sm_scale = info[22].As<Napi::Number>().FloatValue();
    glm_batch_prefill_ragged_run(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(q_ptr), reinterpret_cast<void*>(k_ptr),
        reinterpret_cast<void*>(v_ptr), reinterpret_cast<void*>(o_ptr),
        reinterpret_cast<void*>(float_ws_ptr), reinterpret_cast<void*>(int_ws_ptr),
        reinterpret_cast<int32_t*>(q_indptr_d_ptr),
        reinterpret_cast<int32_t*>(kv_indptr_d_ptr),
        reinterpret_cast<int64_t*>(plan_info_ptr),
        total_qo_rows, batch_size,
        num_qo_heads, num_kv_heads, head_dim,
        q_stride_n, q_stride_h,
        kv_stride_n, kv_stride_h,
        v_stride_n, v_stride_h,
        mask_mode, sm_scale);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Napi::Error::New(env, std::string("batchPrefillRaggedRun failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
        }
    }
    return env.Undefined();
}

static Napi::Value MlaPrefillPlan(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 14) {
        Napi::TypeError::New(env, "Expected (ctx, float_ws, float_ws_size, int_ws, pinned_int_ws, int_ws_size, plan_info, qo_indptr_h, kv_indptr_h, kv_len_h, batch_size, num_heads, head_dim_o, causal, [cp_world_size, cp_rank])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t float_ws = info[1].As<Napi::Number>().Int64Value();
    size_t float_ws_size = info[2].As<Napi::Number>().Int64Value();
    uintptr_t int_ws = info[3].As<Napi::Number>().Int64Value();
    uintptr_t pinned_int_ws = info[4].As<Napi::Number>().Int64Value();
    size_t int_ws_size = info[5].As<Napi::Number>().Int64Value();
    uintptr_t plan_info_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t qo_indptr_h_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t kv_indptr_h_ptr = info[8].As<Napi::Number>().Int64Value();
    uintptr_t kv_len_h_ptr = info[9].As<Napi::Number>().Int64Value();
    uint32_t batch_size = info[10].As<Napi::Number>().Uint32Value();
    uint32_t num_heads = info[11].As<Napi::Number>().Uint32Value();
    uint32_t head_dim_o = info[12].As<Napi::Number>().Uint32Value();
    bool causal = info[13].As<Napi::Boolean>().Value();
    uint32_t cp_world_size = info.Length() > 14 ? info[14].As<Napi::Number>().Uint32Value() : 0;
    uint32_t cp_rank = info.Length() > 15 ? info[15].As<Napi::Number>().Uint32Value() : 0;
    glm_mla_prefill_plan(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(float_ws), float_ws_size,
        reinterpret_cast<void*>(int_ws), reinterpret_cast<void*>(pinned_int_ws), int_ws_size,
        reinterpret_cast<int64_t*>(plan_info_ptr),
        reinterpret_cast<int32_t*>(qo_indptr_h_ptr),
        reinterpret_cast<int32_t*>(kv_indptr_h_ptr),
        reinterpret_cast<int32_t*>(kv_len_h_ptr),
        batch_size, num_heads, head_dim_o, causal, cp_world_size, cp_rank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("mlaPrefillPlan failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MlaPrefillRun(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 29) {
        Napi::TypeError::New(env, "Expected (ctx, q_nope, q_pe, ckv_data, kpe_data, kv_indices, o, float_ws, int_ws, plan_info, num_heads, page_size, mask_mode, sm_scale, q_nope_stride_n, q_nope_stride_h, q_pe_stride_n, q_pe_stride_h, ckv_stride_page, ckv_stride_n, kpe_stride_page, kpe_stride_n, o_stride_n, o_stride_h, head_dim_ckv, head_dim_kpe, lse, cp_world_size, cp_rank[, custom_mask, mask_indptr])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t q_nope_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t q_pe_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t ckv_data_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t kpe_data_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t kv_indices_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t o_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t float_ws_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t int_ws_ptr = info[8].As<Napi::Number>().Int64Value();
    uintptr_t plan_info_ptr = info[9].As<Napi::Number>().Int64Value();
    uint32_t num_heads = info[10].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[11].As<Napi::Number>().Uint32Value();
    int mask_mode = info[12].As<Napi::Number>().Int32Value();
    float sm_scale = info[13].As<Napi::Number>().FloatValue();
    uint32_t q_nope_stride_n = info[14].As<Napi::Number>().Uint32Value();
    uint32_t q_nope_stride_h = info[15].As<Napi::Number>().Uint32Value();
    uint32_t q_pe_stride_n = info[16].As<Napi::Number>().Uint32Value();
    uint32_t q_pe_stride_h = info[17].As<Napi::Number>().Uint32Value();
    uint32_t ckv_stride_page = info[18].As<Napi::Number>().Uint32Value();
    uint32_t ckv_stride_n = info[19].As<Napi::Number>().Uint32Value();
    uint32_t kpe_stride_page = info[20].As<Napi::Number>().Uint32Value();
    uint32_t kpe_stride_n = info[21].As<Napi::Number>().Uint32Value();
    uint32_t o_stride_n = info[22].As<Napi::Number>().Uint32Value();
    uint32_t o_stride_h = info[23].As<Napi::Number>().Uint32Value();
    uint32_t head_dim_ckv = info[24].As<Napi::Number>().Uint32Value();
    uint32_t head_dim_kpe = info[25].As<Napi::Number>().Uint32Value();
    uintptr_t lse_ptr = info[26].As<Napi::Number>().Int64Value();
    uint32_t cp_world_size = info[27].As<Napi::Number>().Uint32Value();
    uint32_t cp_rank = info[28].As<Napi::Number>().Uint32Value();
    void* custom_mask_ptr = nullptr;
    int32_t* mask_indptr_ptr = nullptr;
    int32_t* mask_kv_len_ptr = nullptr;
    if (info.Length() >= 31) {
        custom_mask_ptr = reinterpret_cast<void*>(info[29].As<Napi::Number>().Int64Value());
        mask_indptr_ptr = reinterpret_cast<int32_t*>(info[30].As<Napi::Number>().Int64Value());
    }
    if (info.Length() >= 32) {
        mask_kv_len_ptr = reinterpret_cast<int32_t*>(info[31].As<Napi::Number>().Int64Value());
    }
    glm_mla_prefill_run(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(q_nope_ptr), reinterpret_cast<void*>(q_pe_ptr),
        reinterpret_cast<void*>(ckv_data_ptr), reinterpret_cast<void*>(kpe_data_ptr),
        reinterpret_cast<int32_t*>(kv_indices_ptr), reinterpret_cast<void*>(o_ptr),
        reinterpret_cast<void*>(float_ws_ptr), reinterpret_cast<void*>(int_ws_ptr),
        reinterpret_cast<int64_t*>(plan_info_ptr),
        num_heads, page_size, mask_mode, sm_scale,
        q_nope_stride_n, q_nope_stride_h,
        q_pe_stride_n, q_pe_stride_h,
        ckv_stride_page, ckv_stride_n,
        kpe_stride_page, kpe_stride_n,
        o_stride_n, o_stride_h,
        head_dim_ckv, head_dim_kpe,
        reinterpret_cast<float*>(lse_ptr),
        cp_world_size, cp_rank,
        custom_mask_ptr, mask_indptr_ptr, mask_kv_len_ptr);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Napi::Error::New(env, std::string("mlaPrefillRun failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
        }
    }
    return env.Undefined();
}

static Napi::Value MlaDecodePlan(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 14) {
        Napi::TypeError::New(env, "Expected (ctx, float_ws, float_ws_size, int_ws, pinned_int_ws, int_ws_size, plan_info, indptr_h, batch_size, num_qo_heads, page_size, enable_cuda_graph, head_dim_ckv, head_dim_kpe)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t float_ws = info[1].As<Napi::Number>().Int64Value();
    size_t float_ws_size = info[2].As<Napi::Number>().Int64Value();
    uintptr_t int_ws = info[3].As<Napi::Number>().Int64Value();
    uintptr_t pinned_int_ws = info[4].As<Napi::Number>().Int64Value();
    size_t int_ws_size = info[5].As<Napi::Number>().Int64Value();
    uintptr_t plan_info_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t indptr_h_ptr = info[7].As<Napi::Number>().Int64Value();
    uint32_t batch_size = info[8].As<Napi::Number>().Uint32Value();
    uint32_t num_qo_heads = info[9].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[10].As<Napi::Number>().Uint32Value();
    bool enable_cuda_graph = info[11].As<Napi::Boolean>().Value();
    uint32_t head_dim_ckv = info[12].As<Napi::Number>().Uint32Value();
    uint32_t head_dim_kpe = info[13].As<Napi::Number>().Uint32Value();
    glm_mla_decode_plan(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(float_ws), float_ws_size,
        reinterpret_cast<void*>(int_ws), reinterpret_cast<void*>(pinned_int_ws), int_ws_size,
        reinterpret_cast<int64_t*>(plan_info_ptr),
        reinterpret_cast<int32_t*>(indptr_h_ptr),
        batch_size, num_qo_heads, page_size, enable_cuda_graph,
        head_dim_ckv, head_dim_kpe);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("mlaDecodePlan failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MlaDecodeRun(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 19) {
        Napi::TypeError::New(env, "Expected (ctx, q_nope, q_pe, ckv_data, kpe_data, indices, indptr_d, last_page_len, o, float_ws, int_ws, plan_info, batch_size, num_qo_heads, page_size, sm_scale, head_dim_ckv, head_dim_kpe, lse) — note 19+ params").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t q_nope_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t q_pe_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t ckv_data_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t kpe_data_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t indices_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t indptr_d_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t last_page_len_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t o_ptr = info[8].As<Napi::Number>().Int64Value();
    uintptr_t float_ws_ptr = info[9].As<Napi::Number>().Int64Value();
    uintptr_t int_ws_ptr = info[10].As<Napi::Number>().Int64Value();
    uintptr_t plan_info_ptr = info[11].As<Napi::Number>().Int64Value();
    uint32_t batch_size = info[12].As<Napi::Number>().Uint32Value();
    uint32_t num_qo_heads = info[13].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[14].As<Napi::Number>().Uint32Value();
    float sm_scale = info[15].As<Napi::Number>().FloatValue();
    uint32_t head_dim_ckv = info[16].As<Napi::Number>().Uint32Value();
    uint32_t head_dim_kpe = info[17].As<Napi::Number>().Uint32Value();
    uintptr_t lse_ptr = info[18].As<Napi::Number>().Int64Value();
    glm_mla_decode_run(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(q_nope_ptr), reinterpret_cast<void*>(q_pe_ptr),
        reinterpret_cast<void*>(ckv_data_ptr), reinterpret_cast<void*>(kpe_data_ptr),
        reinterpret_cast<int32_t*>(indices_ptr),
        reinterpret_cast<int32_t*>(indptr_d_ptr),
        reinterpret_cast<int32_t*>(last_page_len_ptr),
        reinterpret_cast<void*>(o_ptr),
        reinterpret_cast<void*>(float_ws_ptr), reinterpret_cast<void*>(int_ws_ptr),
        reinterpret_cast<int64_t*>(plan_info_ptr),
        batch_size, num_qo_heads, page_size, sm_scale,
        head_dim_ckv, head_dim_kpe,
        reinterpret_cast<float*>(lse_ptr));
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Napi::Error::New(env, std::string("mlaDecodeRun failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
        }
    }
    return env.Undefined();
}

static Napi::Value MlaKvCacheAppend(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 16) {
        Napi::TypeError::New(env, "Expected at least 16 args (ctx, ckv_data, kpe_data, indices, indptr, last_page_len, append_ckv, append_kpe, batch_indices, positions, nnz, page_size, head_dim_ckv, head_dim_kpe, append_ckv_stride_n, append_kpe_stride_n[, cp_world_size, cp_rank])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t ckv_data_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t kpe_data_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t indices_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t indptr_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t last_page_len_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t append_ckv_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t append_kpe_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t batch_indices_ptr = info[8].As<Napi::Number>().Int64Value();
    uintptr_t positions_ptr = info[9].As<Napi::Number>().Int64Value();
    uint32_t nnz = info[10].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[11].As<Napi::Number>().Uint32Value();
    uint32_t head_dim_ckv = info[12].As<Napi::Number>().Uint32Value();
    uint32_t head_dim_kpe = info[13].As<Napi::Number>().Uint32Value();
    size_t append_ckv_stride_n = info[14].As<Napi::Number>().Int64Value();
    size_t append_kpe_stride_n = info[15].As<Napi::Number>().Int64Value();
    uint32_t cp_world_size = (info.Length() >= 17) ? info[16].As<Napi::Number>().Uint32Value() : 1;
    uint32_t cp_rank = (info.Length() >= 18) ? info[17].As<Napi::Number>().Uint32Value() : 0;
    glm_mla_kv_cache_append(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(ckv_data_ptr), reinterpret_cast<void*>(kpe_data_ptr),
        reinterpret_cast<int32_t*>(indices_ptr),
        reinterpret_cast<int32_t*>(indptr_ptr),
        reinterpret_cast<int32_t*>(last_page_len_ptr),
        reinterpret_cast<void*>(append_ckv_ptr), reinterpret_cast<void*>(append_kpe_ptr),
        reinterpret_cast<int32_t*>(batch_indices_ptr),
        reinterpret_cast<int32_t*>(positions_ptr),
        nnz, page_size, head_dim_ckv, head_dim_kpe,
        append_ckv_stride_n, append_kpe_stride_n,
        cp_world_size, cp_rank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("mlaKvCacheAppend failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value ConcatAndCacheDsMla(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 14) {
        Napi::TypeError::New(env, "Expected at least 14 args (ctx, kv_cache, append_ckv, append_kpe, indices, indptr, batch_indices, positions, nnz, page_size, kv_lora_rank, pe_dim, append_ckv_stride_n, append_kpe_stride_n[, cp_world_size, cp_rank])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t kv_cache_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t append_ckv_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t append_kpe_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t indices_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t indptr_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t batch_indices_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t positions_ptr = info[7].As<Napi::Number>().Int64Value();
    uint32_t nnz = info[8].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[9].As<Napi::Number>().Uint32Value();
    uint32_t kv_lora_rank = info[10].As<Napi::Number>().Uint32Value();
    uint32_t pe_dim = info[11].As<Napi::Number>().Uint32Value();
    size_t append_ckv_stride_n = info[12].As<Napi::Number>().Int64Value();
    size_t append_kpe_stride_n = info[13].As<Napi::Number>().Int64Value();
    uint32_t cp_world_size = (info.Length() >= 15) ? info[14].As<Napi::Number>().Uint32Value() : 0;
    uint32_t cp_rank = (info.Length() >= 16) ? info[15].As<Napi::Number>().Uint32Value() : 0;
    glm_concat_and_cache_ds_mla(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(kv_cache_ptr),
        reinterpret_cast<void*>(append_ckv_ptr), reinterpret_cast<void*>(append_kpe_ptr),
        reinterpret_cast<int32_t*>(indices_ptr),
        reinterpret_cast<int32_t*>(indptr_ptr),
        reinterpret_cast<int32_t*>(batch_indices_ptr),
        reinterpret_cast<int32_t*>(positions_ptr),
        nnz, page_size, kv_lora_rank, pe_dim,
        append_ckv_stride_n, append_kpe_stride_n,
        cp_world_size, cp_rank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("concatAndCacheDsMla failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value SparseMlaPrefill(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 12) {
        Napi::TypeError::New(env, "Expected 12 args (ctx, q, kv_cache, indices, output, out_lse, num_tokens, num_heads, topk, page_block_size, sm_scale, stride_kv_block)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t q_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t kv_cache_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t indices_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t output_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t out_lse_ptr = info[5].As<Napi::Number>().Int64Value();
    uint32_t num_tokens = info[6].As<Napi::Number>().Uint32Value();
    uint32_t num_heads = info[7].As<Napi::Number>().Uint32Value();
    uint32_t topk = info[8].As<Napi::Number>().Uint32Value();
    uint32_t page_block_size = info[9].As<Napi::Number>().Uint32Value();
    float sm_scale = info[10].As<Napi::Number>().FloatValue();
    size_t stride_kv_block = info[11].As<Napi::Number>().Int64Value();
    int32_t* topk_length = nullptr;
    if (info.Length() >= 13 && info[12].IsNumber()) {
        topk_length = reinterpret_cast<int32_t*>(info[12].As<Napi::Number>().Int64Value());
    }
    glm_sparse_mla_prefill(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(q_ptr),
        reinterpret_cast<void*>(kv_cache_ptr),
        reinterpret_cast<int32_t*>(indices_ptr),
        reinterpret_cast<void*>(output_ptr),
        reinterpret_cast<float*>(out_lse_ptr),
        num_tokens, num_heads, topk, page_block_size,
        sm_scale, stride_kv_block, topk_length);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("sparseMlaPrefill failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value SparseMlaDecode(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 15) {
        Napi::TypeError::New(env, "Expected 15 args (ctx, q, kv_cache, indices, mid_out, mid_lse, output, out_lse, num_tokens, num_heads, topk, num_splits, sm_scale, stride_kv_block, chunks_per_block)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t q_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t kv_cache_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t indices_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t mid_out_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t mid_lse_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t output_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t out_lse_ptr = info[7].As<Napi::Number>().Int64Value();
    uint32_t num_tokens = info[8].As<Napi::Number>().Uint32Value();
    uint32_t num_heads = info[9].As<Napi::Number>().Uint32Value();
    uint32_t topk = info[10].As<Napi::Number>().Uint32Value();
    uint32_t num_splits = info[11].As<Napi::Number>().Uint32Value();
    float sm_scale = info[12].As<Napi::Number>().FloatValue();
    size_t stride_kv_block = info[13].As<Napi::Number>().Int64Value();
    int chunks_per_block = info[14].As<Napi::Number>().Int32Value();
    int32_t* topk_length = nullptr;
    if (info.Length() >= 16 && info[15].IsNumber()) {
        topk_length = reinterpret_cast<int32_t*>(info[15].As<Napi::Number>().Int64Value());
    }
    glm_sparse_mla_decode(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(q_ptr),
        reinterpret_cast<void*>(kv_cache_ptr),
        reinterpret_cast<int32_t*>(indices_ptr),
        reinterpret_cast<void*>(mid_out_ptr),
        reinterpret_cast<float*>(mid_lse_ptr),
        reinterpret_cast<void*>(output_ptr),
        reinterpret_cast<float*>(out_lse_ptr),
        num_tokens, num_heads, topk, num_splits,
        sm_scale, stride_kv_block, topk_length, chunks_per_block);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("sparseMlaDecode failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value GatherTopkCkv(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 26) {
        Napi::TypeError::New(env, "Expected 26 args (ctx, flat_p0..p7, local_kv_cache, topk_idx, batch_indices, page_indices, page_indptr, kv_token_indptr, N, cp_world_size, cp_rank, eff_page_size, bpt_bytes, num_tokens, topk, padded_kv_len, scratch_bitmap, scratch_unique, scratch_counter)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t flat_p[8];
    for (int i = 0; i < 8; i++) {
        flat_p[i] = info[1 + i].As<Napi::Number>().Int64Value();
    }
    uintptr_t local_kv_cache_ptr = info[9].As<Napi::Number>().Int64Value();
    uintptr_t topk_idx_ptr = info[10].As<Napi::Number>().Int64Value();
    uintptr_t batch_indices_ptr = info[11].As<Napi::Number>().Int64Value();
    uintptr_t page_indices_ptr = info[12].As<Napi::Number>().Int64Value();
    uintptr_t page_indptr_ptr = info[13].As<Napi::Number>().Int64Value();
    uintptr_t kv_token_indptr_ptr = info[14].As<Napi::Number>().Int64Value();
    int N = info[15].As<Napi::Number>().Int32Value();
    int cp_world_size = info[16].As<Napi::Number>().Int32Value();
    int cp_rank = info[17].As<Napi::Number>().Int32Value();
    int eff_page_size = info[18].As<Napi::Number>().Int32Value();
    int bpt_bytes = info[19].As<Napi::Number>().Int32Value();
    int num_tokens = info[20].As<Napi::Number>().Int32Value();
    int topk = info[21].As<Napi::Number>().Int32Value();
    int padded_kv_len = info[22].As<Napi::Number>().Int32Value();
    uintptr_t scratch_bitmap_ptr = info[23].As<Napi::Number>().Int64Value();
    uintptr_t scratch_unique_ptr = info[24].As<Napi::Number>().Int64Value();
    uintptr_t scratch_counter_ptr = info[25].As<Napi::Number>().Int64Value();
    glm_gather_topk_ckv(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(flat_p[0]), reinterpret_cast<void*>(flat_p[1]),
        reinterpret_cast<void*>(flat_p[2]), reinterpret_cast<void*>(flat_p[3]),
        reinterpret_cast<void*>(flat_p[4]), reinterpret_cast<void*>(flat_p[5]),
        reinterpret_cast<void*>(flat_p[6]), reinterpret_cast<void*>(flat_p[7]),
        reinterpret_cast<void*>(local_kv_cache_ptr),
        reinterpret_cast<int32_t*>(topk_idx_ptr),
        reinterpret_cast<int32_t*>(batch_indices_ptr),
        reinterpret_cast<int32_t*>(page_indices_ptr),
        reinterpret_cast<int32_t*>(page_indptr_ptr),
        reinterpret_cast<int32_t*>(kv_token_indptr_ptr),
        N, cp_world_size, cp_rank,
        eff_page_size, bpt_bytes,
        num_tokens, topk, padded_kv_len,
        reinterpret_cast<void*>(scratch_bitmap_ptr),
        reinterpret_cast<void*>(scratch_unique_ptr),
        reinterpret_cast<void*>(scratch_counter_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("gatherTopkCkv failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value GraphBeginCapture(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (ctx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    glm_graph_begin_capture(reinterpret_cast<GlmCtx*>(ctx_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("graphBeginCapture failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value GraphEndCapture(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (ctx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    void* graph = glm_graph_end_capture(reinterpret_cast<GlmCtx*>(ctx_ptr));
    if (!graph) {
        cudaError_t err = cudaGetLastError();
        std::string msg = "graphEndCapture failed: null graph returned";
        if (err != cudaSuccess) msg += std::string(" (") + cudaGetErrorString(err) + ")";
        Napi::Error::New(env, msg).ThrowAsJavaScriptException();
        return Napi::Number::New(env, 0);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("graphEndCapture failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return Napi::Number::New(env, reinterpret_cast<uintptr_t>(graph));
}

static Napi::Value GraphInstantiate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, graph)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t graph_ptr = info[1].As<Napi::Number>().Int64Value();
    void* graph_exec = glm_graph_instantiate(reinterpret_cast<GlmCtx*>(ctx_ptr),
                                              reinterpret_cast<void*>(graph_ptr));
    if (!graph_exec) {
        cudaError_t err = cudaGetLastError();
        std::string msg = "graphInstantiate failed: null graph_exec returned";
        if (err != cudaSuccess) msg += std::string(" (") + cudaGetErrorString(err) + ")";
        Napi::Error::New(env, msg).ThrowAsJavaScriptException();
        return Napi::Number::New(env, 0);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("graphInstantiate failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return Napi::Number::New(env, reinterpret_cast<uintptr_t>(graph_exec));
}

static Napi::Value GraphLaunch(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, graph_exec)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t graph_exec_ptr = info[1].As<Napi::Number>().Int64Value();
    glm_graph_launch(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(graph_exec_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("graphLaunch failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value GraphDestroy(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, graph)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t graph_ptr = info[1].As<Napi::Number>().Int64Value();
    glm_graph_destroy(reinterpret_cast<GlmCtx*>(ctx_ptr),
                      reinterpret_cast<void*>(graph_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("graphDestroy failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value GraphExecDestroy(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, graph_exec)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t graph_exec_ptr = info[1].As<Napi::Number>().Int64Value();
    glm_graph_exec_destroy(reinterpret_cast<GlmCtx*>(ctx_ptr),
                            reinterpret_cast<void*>(graph_exec_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("graphExecDestroy failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Fp8LinearDecode(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
        Napi::TypeError::New(env, "Expected (ctx, bf16_out, bf16_input, fp8_weight, weight_scale, m, n, k)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t input_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t weight_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t scale_ptr = info[4].As<Napi::Number>().Int64Value();
    int m = info[5].As<Napi::Number>().Int32Value();
    int n = info[6].As<Napi::Number>().Int32Value();
    int k = info[7].As<Napi::Number>().Int32Value();
    glm_fp8_linear_decode(reinterpret_cast<GlmCtx*>(ctx_ptr),
                           reinterpret_cast<void*>(out_ptr),
                           reinterpret_cast<const void*>(input_ptr),
                           reinterpret_cast<const void*>(weight_ptr),
                           reinterpret_cast<const float*>(scale_ptr),
                           m, n, k);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("fp8LinearDecode failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Nvfp4LinearDecode(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 8) {
        Napi::TypeError::New(env, "Expected (ctx, bf16_out, bf16_input, fp4_weight, weight_scale, weight_scale_2, m, n, k[, bf16_workspace])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t input_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t weight_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t scale_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t scale2_ptr = info[5].As<Napi::Number>().Int64Value();
    int m = info[6].As<Napi::Number>().Int32Value();
    int n = info[7].As<Napi::Number>().Int32Value();
    int k = info[8].As<Napi::Number>().Int32Value();
    uintptr_t ws_ptr = 0;
    if (info.Length() >= 10) {
        ws_ptr = info[9].As<Napi::Number>().Int64Value();
    }
    glm_nvfp4_linear_decode(reinterpret_cast<GlmCtx*>(ctx_ptr),
                             reinterpret_cast<void*>(out_ptr),
                             reinterpret_cast<const void*>(input_ptr),
                             reinterpret_cast<const void*>(weight_ptr),
                             reinterpret_cast<const void*>(scale_ptr),
                             reinterpret_cast<const float*>(scale2_ptr),
                             m, n, k,
                             reinterpret_cast<void*>(ws_ptr));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("nvfp4LinearDecode failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value GdnRecurrentStep(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 15) {
        Napi::TypeError::New(env, "Expected (ctx, output, state, qkv, a_raw, b_raw, A_log, dt_bias, num_heads, d_k, d_v, batch_size, state_stride, qkv_ch_stride, qkv_seq_stride)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t state_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t qkv_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t a_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t b_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t alog_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t dtb_ptr = info[7].As<Napi::Number>().Int64Value();
    int num_heads = info[8].As<Napi::Number>().Int32Value();
    int d_k = info[9].As<Napi::Number>().Int32Value();
    int d_v = info[10].As<Napi::Number>().Int32Value();
    int batch_size = info[11].As<Napi::Number>().Int32Value();
    int state_stride = info[12].As<Napi::Number>().Int32Value();
    int qkv_ch_stride = info[13].As<Napi::Number>().Int32Value();
    int qkv_seq_stride = info[14].As<Napi::Number>().Int32Value();
    glm_gdn_recurrent_step(reinterpret_cast<GlmCtx*>(ctx_ptr),
                            reinterpret_cast<void*>(out_ptr),
                            reinterpret_cast<float*>(state_ptr),
                            reinterpret_cast<const void*>(qkv_ptr),
                            reinterpret_cast<const void*>(a_ptr),
                            reinterpret_cast<const void*>(b_ptr),
                            reinterpret_cast<const float*>(alog_ptr),
                            reinterpret_cast<const float*>(dtb_ptr),
                            num_heads, d_k, d_v,
                            batch_size, state_stride, qkv_ch_stride, qkv_seq_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("gdnRecurrentStep failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value GdnPrefill(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 17) {
        Napi::TypeError::New(env, "Expected (ctx, output, state, qkv, a_raw, b_raw, A_log, dt_bias, cu_seqlens, total_seq_len, num_heads, d_k, d_v, batch_size, state_stride, qkv_ch_stride, qkv_seq_stride)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t state_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t qkv_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t a_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t b_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t alog_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t dtb_ptr = info[7].As<Napi::Number>().Int64Value();
    uintptr_t cu_seqlens_ptr = info[8].As<Napi::Number>().Int64Value();
    int total_seq_len = info[9].As<Napi::Number>().Int32Value();
    int num_heads = info[10].As<Napi::Number>().Int32Value();
    int d_k = info[11].As<Napi::Number>().Int32Value();
    int d_v = info[12].As<Napi::Number>().Int32Value();
    int batch_size = info[13].As<Napi::Number>().Int32Value();
    int state_stride = info[14].As<Napi::Number>().Int32Value();
    int qkv_ch_stride = info[15].As<Napi::Number>().Int32Value();
    int qkv_seq_stride = info[16].As<Napi::Number>().Int32Value();
    glm_gdn_prefill(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(out_ptr),
                     reinterpret_cast<float*>(state_ptr),
                     reinterpret_cast<const void*>(qkv_ptr),
                     reinterpret_cast<const void*>(a_ptr),
                     reinterpret_cast<const void*>(b_ptr),
                     reinterpret_cast<const float*>(alog_ptr),
                     reinterpret_cast<const float*>(dtb_ptr),
                     reinterpret_cast<const int*>(cu_seqlens_ptr),
                     total_seq_len, num_heads, d_k, d_v,
                     batch_size, state_stride, qkv_ch_stride, qkv_seq_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("gdnPrefill failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value CausalConv1d(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 13) {
        Napi::TypeError::New(env, "Expected (ctx, output, conv_state, input, weight, cu_seqlens, conv_dim, total_seq_len, kernel_size, batch_size, conv_state_stride, ch_stride, seq_stride)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t cs_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t w_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t cu_seqlens_ptr = info[5].As<Napi::Number>().Int64Value();
    int conv_dim = info[6].As<Napi::Number>().Int32Value();
    int total_seq_len = info[7].As<Napi::Number>().Int32Value();
    int kernel_size = info[8].As<Napi::Number>().Int32Value();
    int batch_size = info[9].As<Napi::Number>().Int32Value();
    int conv_state_stride = info[10].As<Napi::Number>().Int32Value();
    int ch_stride = info[11].As<Napi::Number>().Int32Value();
    int seq_stride = info[12].As<Napi::Number>().Int32Value();
    glm_causal_conv1d(reinterpret_cast<GlmCtx*>(ctx_ptr),
                       reinterpret_cast<void*>(out_ptr),
                       reinterpret_cast<void*>(cs_ptr),
                       reinterpret_cast<const void*>(in_ptr),
                       reinterpret_cast<const void*>(w_ptr),
                       reinterpret_cast<const int*>(cu_seqlens_ptr),
                       conv_dim, total_seq_len, kernel_size,
                       batch_size, conv_state_stride, ch_stride, seq_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("causalConv1d failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value CausalConv1dUpdate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 8) {
        Napi::TypeError::New(env, "Expected (ctx, output, conv_state, input, weight, conv_dim, kernel_size, batch_size, conv_state_stride)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t cs_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t w_ptr = info[4].As<Napi::Number>().Int64Value();
    int conv_dim = info[5].As<Napi::Number>().Int32Value();
    int kernel_size = info[6].As<Napi::Number>().Int32Value();
    int batch_size = info[7].As<Napi::Number>().Int32Value();
    int conv_state_stride = info[8].As<Napi::Number>().Int32Value();
    glm_causal_conv1d_update(reinterpret_cast<GlmCtx*>(ctx_ptr),
                               reinterpret_cast<void*>(out_ptr),
                               reinterpret_cast<void*>(cs_ptr),
                               reinterpret_cast<const void*>(in_ptr),
                               reinterpret_cast<const void*>(w_ptr),
                               conv_dim, kernel_size,
                               batch_size, conv_state_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("causalConv1dUpdate failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value RmsnormGated(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 8) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, gate, weight, eps, dim, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t gate_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t w_ptr = info[4].As<Napi::Number>().Int64Value();
    float eps = info[5].As<Napi::Number>().FloatValue();
    int dim = info[6].As<Napi::Number>().Int32Value();
    int batch = info[7].As<Napi::Number>().Int32Value();
    glm_rmsnorm_gated(reinterpret_cast<GlmCtx*>(ctx_ptr),
                       reinterpret_cast<void*>(out_ptr),
                       reinterpret_cast<const void*>(in_ptr),
                       reinterpret_cast<const void*>(gate_ptr),
                       reinterpret_cast<const void*>(w_ptr),
                       eps, dim, batch);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("rmsnormGated failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value SampleBatch(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 18) {
        Napi::TypeError::New(env, "Expected (ctx, out_tokens, topk_vals, topk_idxs, workspace, logits, penalty_tokens, penalty_count, max_window, vocab_size, batch_size, temperatures, repetition_penalties, presence_penalties, top_ks, top_ps, step_counter, max_effective_k)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_tokens_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t topk_vals_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t topk_idxs_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t workspace_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t logits_ptr = info[5].As<Napi::Number>().Int64Value();
    uintptr_t penalty_tokens_ptr = info[6].As<Napi::Number>().Int64Value();
    uintptr_t penalty_count_ptr = info[7].As<Napi::Number>().Int64Value();
    int max_window = info[8].As<Napi::Number>().Int32Value();
    int vocab_size = info[9].As<Napi::Number>().Int32Value();
    int batch_size = info[10].As<Napi::Number>().Int32Value();
    uintptr_t temperatures_ptr = info[11].As<Napi::Number>().Int64Value();
    uintptr_t rep_penalties_ptr = info[12].As<Napi::Number>().Int64Value();
    uintptr_t pres_penalties_ptr = info[13].As<Napi::Number>().Int64Value();
    uintptr_t top_ks_ptr = info[14].As<Napi::Number>().Int64Value();
    uintptr_t top_ps_ptr = info[15].As<Napi::Number>().Int64Value();
    uintptr_t step_counter_ptr = info[16].As<Napi::Number>().Int64Value();
    int max_effective_k = info[17].As<Napi::Number>().Int32Value();
    glm_sample_batch(reinterpret_cast<GlmCtx*>(ctx_ptr),
               reinterpret_cast<int*>(out_tokens_ptr),
               reinterpret_cast<float*>(topk_vals_ptr),
               reinterpret_cast<int*>(topk_idxs_ptr),
               reinterpret_cast<float*>(workspace_ptr),
               reinterpret_cast<const void*>(logits_ptr),
               reinterpret_cast<int*>(penalty_tokens_ptr),
               reinterpret_cast<int*>(penalty_count_ptr),
               max_window, vocab_size, batch_size,
               reinterpret_cast<const float*>(temperatures_ptr),
               reinterpret_cast<const float*>(rep_penalties_ptr),
               reinterpret_cast<const float*>(pres_penalties_ptr),
               reinterpret_cast<const int*>(top_ks_ptr),
               reinterpret_cast<const float*>(top_ps_ptr),
               reinterpret_cast<unsigned int*>(step_counter_ptr),
               max_effective_k);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("sampleBatch failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value CpMergeTree(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 27) {
        Napi::TypeError::New(env, "Expected (ctx, v0..v7, lse0..lse7, num_shards, output_v, output_lse, numel, batch_size, num_heads, v_head_dim, shard_n_heads, head_offset, input_n_heads)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t vp[8], lp[8];
    for (int i = 0; i < 8; i++) {
        vp[i] = info[1 + i].As<Napi::Number>().Int64Value();
        lp[i] = info[9 + i].As<Napi::Number>().Int64Value();
    }
    int num_shards = info[17].As<Napi::Number>().Int32Value();
    uintptr_t output_v_ptr = info[18].As<Napi::Number>().Int64Value();
    uintptr_t output_lse_ptr = info[19].As<Napi::Number>().Int64Value();
    int64_t numel = info[20].As<Napi::Number>().Int64Value();
    int batch_size = info[21].As<Napi::Number>().Int32Value();
    int num_heads = info[22].As<Napi::Number>().Int32Value();
    int v_head_dim = info[23].As<Napi::Number>().Int32Value();
    int shard_n_heads = info[24].As<Napi::Number>().Int32Value();
    int head_offset = info[25].As<Napi::Number>().Int32Value();
    int input_n_heads = info[26].As<Napi::Number>().Int32Value();

    glm_cp_merge_tree(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<const void*>(vp[0]),  reinterpret_cast<const void*>(vp[1]),
        reinterpret_cast<const void*>(vp[2]),  reinterpret_cast<const void*>(vp[3]),
        reinterpret_cast<const void*>(vp[4]),  reinterpret_cast<const void*>(vp[5]),
        reinterpret_cast<const void*>(vp[6]),  reinterpret_cast<const void*>(vp[7]),
        reinterpret_cast<const float*>(lp[0]),  reinterpret_cast<const float*>(lp[1]),
        reinterpret_cast<const float*>(lp[2]),  reinterpret_cast<const float*>(lp[3]),
        reinterpret_cast<const float*>(lp[4]),  reinterpret_cast<const float*>(lp[5]),
        reinterpret_cast<const float*>(lp[6]),  reinterpret_cast<const float*>(lp[7]),
        num_shards,
        reinterpret_cast<void*>(output_v_ptr),
        reinterpret_cast<float*>(output_lse_ptr),
        numel, batch_size, num_heads, v_head_dim,
        shard_n_heads, head_offset, input_n_heads);

    cudaError_t cp_err = cudaGetLastError();
    if (cp_err != cudaSuccess) {
        Napi::Error::New(env, std::string("cpMergeTree failed: ") + cudaGetErrorString(cp_err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value CpMergeScatter(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 26) {
        Napi::TypeError::New(env, "Expected (ctx, local_v, local_lse, dv0..dv7, dl0..dl7, world_size, batch_size, shard_n_heads, v_head_dim, input_n_heads, num_heads, rank)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t local_v_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t local_lse_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t dv[8], dl[8];
    for (int i = 0; i < 8; i++) {
        dv[i] = info[3 + i].As<Napi::Number>().Int64Value();
        dl[i] = info[11 + i].As<Napi::Number>().Int64Value();
    }
    int world_size = info[19].As<Napi::Number>().Int32Value();
    int batch_size = info[20].As<Napi::Number>().Int32Value();
    int shard_n_heads = info[21].As<Napi::Number>().Int32Value();
    int v_head_dim = info[22].As<Napi::Number>().Int32Value();
    int input_n_heads = info[23].As<Napi::Number>().Int32Value();
    int num_heads = info[24].As<Napi::Number>().Int32Value();
    int rank = info[25].As<Napi::Number>().Int32Value();

    glm_cp_merge_scatter(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<const void*>(local_v_ptr),
        reinterpret_cast<const float*>(local_lse_ptr),
        reinterpret_cast<void*>(dv[0]), reinterpret_cast<void*>(dv[1]),
        reinterpret_cast<void*>(dv[2]), reinterpret_cast<void*>(dv[3]),
        reinterpret_cast<void*>(dv[4]), reinterpret_cast<void*>(dv[5]),
        reinterpret_cast<void*>(dv[6]), reinterpret_cast<void*>(dv[7]),
        reinterpret_cast<float*>(dl[0]), reinterpret_cast<float*>(dl[1]),
        reinterpret_cast<float*>(dl[2]), reinterpret_cast<float*>(dl[3]),
        reinterpret_cast<float*>(dl[4]), reinterpret_cast<float*>(dl[5]),
        reinterpret_cast<float*>(dl[6]), reinterpret_cast<float*>(dl[7]),
        world_size, batch_size, shard_n_heads, v_head_dim,
        input_n_heads, num_heads, rank);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("cpMergeScatter failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value CpMergeLocal(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 9) {
        Napi::TypeError::New(env, "Expected (ctx, stage_v, stage_lse, output_v, output_lse, world_size, batch_size, shard_n_heads, v_head_dim)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t stage_v_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t stage_lse_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t output_v_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t output_lse_ptr = info[4].As<Napi::Number>().Int64Value();
    int world_size = info[5].As<Napi::Number>().Int32Value();
    int batch_size = info[6].As<Napi::Number>().Int32Value();
    int shard_n_heads = info[7].As<Napi::Number>().Int32Value();
    int v_head_dim = info[8].As<Napi::Number>().Int32Value();

    glm_cp_merge_local(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<const void*>(stage_v_ptr),
        reinterpret_cast<const float*>(stage_lse_ptr),
        reinterpret_cast<void*>(output_v_ptr),
        reinterpret_cast<float*>(output_lse_ptr),
        world_size, batch_size, shard_n_heads, v_head_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("cpMergeLocal failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value CpCorrectAttnOut(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 9) {
        Napi::TypeError::New(env, "Expected (ctx, v_out, lses, global_lse, batch_size, num_heads, v_head_dim, world_size, rank)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t v_out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t lses_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t global_lse_ptr = info[3].As<Napi::Number>().Int64Value();
    int batch_size = info[4].As<Napi::Number>().Int32Value();
    int num_heads = info[5].As<Napi::Number>().Int32Value();
    int v_head_dim = info[6].As<Napi::Number>().Int32Value();
    int world_size = info[7].As<Napi::Number>().Int32Value();
    int rank = info[8].As<Napi::Number>().Int32Value();

    glm_cp_correct_attn_out(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(v_out_ptr),
        reinterpret_cast<const float*>(lses_ptr),
        reinterpret_cast<float*>(global_lse_ptr),
        batch_size, num_heads, v_head_dim, world_size, rank);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("cpCorrectAttnOut failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value GateSigmoidMul(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, attn_out, gate_interleaved, batch_seq, num_heads, head_dim)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t gate_ptr = info[2].As<Napi::Number>().Int64Value();
    int batch_seq = info[3].As<Napi::Number>().Int32Value();
    int num_heads = info[4].As<Napi::Number>().Int32Value();
    int head_dim = info[5].As<Napi::Number>().Int32Value();
    glm_gate_sigmoid_mul(reinterpret_cast<GlmCtx*>(ctx_ptr),
                          reinterpret_cast<void*>(out_ptr),
                          reinterpret_cast<const void*>(gate_ptr),
                          batch_seq, num_heads, head_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("gateSigmoidMul failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Memcpy2d(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
        Napi::TypeError::New(env, "Expected (ctx, dst, dpitch, src, spitch, width, height, kind)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t dst_ptr = info[1].As<Napi::Number>().Int64Value();
    size_t dpitch = info[2].As<Napi::Number>().Int64Value();
    uintptr_t src_ptr = info[3].As<Napi::Number>().Int64Value();
    size_t spitch = info[4].As<Napi::Number>().Int64Value();
    size_t width = info[5].As<Napi::Number>().Int64Value();
    size_t height = info[6].As<Napi::Number>().Int64Value();
    int kind = info[7].As<Napi::Number>().Int32Value();
    glm_memcpy2d(reinterpret_cast<GlmCtx*>(ctx_ptr),
                 reinterpret_cast<void*>(dst_ptr), dpitch,
                 reinterpret_cast<const void*>(src_ptr), spitch,
                 width, height, kind);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("memcpy2d failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value MemcpyPeer(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, dst, dstDevice, src, srcDevice, bytes)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t dst_ptr = info[1].As<Napi::Number>().Int64Value();
    int dstDevice = info[2].As<Napi::Number>().Int32Value();
    uintptr_t src_ptr = info[3].As<Napi::Number>().Int64Value();
    int srcDevice = info[4].As<Napi::Number>().Int32Value();
    size_t bytes = info[5].As<Napi::Number>().Int64Value();
    glm_memcpy_peer(reinterpret_cast<GlmCtx*>(ctx_ptr),
                    reinterpret_cast<void*>(dst_ptr), dstDevice,
                    reinterpret_cast<const void*>(src_ptr), srcDevice, bytes);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("memcpyPeer failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value Memcpy3dPeer(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 20) {
        Napi::TypeError::New(env, "Expected (ctx, dstPtr, dstPitch, dstXSize, dstYSize, dstDevice, dstPosX, dstPosY, dstPosZ, srcPtr, srcPitch, srcXSize, srcYSize, srcDevice, srcPosX, srcPosY, srcPosZ, width, height, depth)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t dstPtr = info[1].As<Napi::Number>().Int64Value();
    size_t dstPitch = info[2].As<Napi::Number>().Int64Value();
    size_t dstXSize = info[3].As<Napi::Number>().Int64Value();
    size_t dstYSize = info[4].As<Napi::Number>().Int64Value();
    int dstDevice = info[5].As<Napi::Number>().Int32Value();
    size_t dstPosX = info[6].As<Napi::Number>().Int64Value();
    size_t dstPosY = info[7].As<Napi::Number>().Int64Value();
    size_t dstPosZ = info[8].As<Napi::Number>().Int64Value();
    uintptr_t srcPtr = info[9].As<Napi::Number>().Int64Value();
    size_t srcPitch = info[10].As<Napi::Number>().Int64Value();
    size_t srcXSize = info[11].As<Napi::Number>().Int64Value();
    size_t srcYSize = info[12].As<Napi::Number>().Int64Value();
    int srcDevice = info[13].As<Napi::Number>().Int32Value();
    size_t srcPosX = info[14].As<Napi::Number>().Int64Value();
    size_t srcPosY = info[15].As<Napi::Number>().Int64Value();
    size_t srcPosZ = info[16].As<Napi::Number>().Int64Value();
    size_t width = info[17].As<Napi::Number>().Int64Value();
    size_t height = info[18].As<Napi::Number>().Int64Value();
    size_t depth = info[19].As<Napi::Number>().Int64Value();
    glm_memcpy3d_peer(reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(dstPtr), dstPitch, dstXSize, dstYSize, dstDevice,
        dstPosX, dstPosY, dstPosZ,
        reinterpret_cast<const void*>(srcPtr), srcPitch, srcXSize, srcYSize, srcDevice,
        srcPosX, srcPosY, srcPosZ,
        width, height, depth);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("memcpy3dPeer failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value NcclUniqueId(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsBuffer()) {
        Napi::TypeError::New(env, "Expected (outId: Buffer of 128 bytes)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::Buffer<char> buf = info[0].As<Napi::Buffer<char>>();
    if (buf.Length() < GLM_NCCL_UNIQUE_ID_BYTES) {
        Napi::TypeError::New(env, "outId buffer must be at least 128 bytes").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    glm_nccl_unique_id(buf.Data());
    return env.Undefined();
}

static Napi::Value NcclGroupStart(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int result = glm_nccl_group_start();
    if (result != 0) {
        Napi::Error::New(env, "ncclGroupStart failed").ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value NcclGroupEnd(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int result = glm_nccl_group_end();
    if (result != 0) {
        Napi::Error::New(env, "ncclGroupEnd failed").ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value NcclCommInitRank(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4) {
        Napi::TypeError::New(env, "Expected (deviceId, rank, worldSize, uniqueId)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    int device_id = info[0].As<Napi::Number>().Int32Value();
    int rank = info[1].As<Napi::Number>().Int32Value();
    int world_size = info[2].As<Napi::Number>().Int32Value();
    uintptr_t unique_id_ptr = info[3].As<Napi::Number>().Int64Value();
    void* comm = glm_nccl_comm_init_rank(device_id, rank, world_size,
                   reinterpret_cast<const void*>(unique_id_ptr));
    return Napi::Number::New(env, reinterpret_cast<uintptr_t>(comm));
}

static Napi::Value NcclCommDestroy(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (comm)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t comm_ptr = info[0].As<Napi::Number>().Int64Value();
    glm_nccl_comm_destroy(reinterpret_cast<void*>(comm_ptr));
    return env.Undefined();
}

static Napi::Value NcclCommInitAll(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "Expected (deviceIds: number[])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::Array devArr = info[0].As<Napi::Array>();
    int ndev = devArr.Length();
    std::vector<int> devlist(ndev);
    for (int i = 0; i < ndev; i++) {
        devlist[i] = devArr.Get(i).As<Napi::Number>().Int32Value();
    }
    std::vector<void*> comms(ndev, nullptr);
    int result = glm_nccl_comm_init_all(comms.data(), ndev, devlist.data());
    if (result != 0) {
        Napi::Error::New(env, "ncclCommInitAll failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::Array resultArr = Napi::Array::New(env, ndev);
    for (int i = 0; i < ndev; i++) {
        resultArr.Set(i, Napi::Number::New(env, reinterpret_cast<uintptr_t>(comms[i])));
    }
    return resultArr;
}

static Napi::Value NcclAllReduce(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
        Napi::TypeError::New(env, "Expected (comm, ctx, sendbuff, recvbuff, count, datatype, op)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t comm_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t ctx_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t send_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t recv_ptr = info[3].As<Napi::Number>().Int64Value();
    size_t count = info[4].As<Napi::Number>().Int64Value();
    int datatype = info[5].As<Napi::Number>().Int32Value();
    int op = info[6].As<Napi::Number>().Int32Value();
    glm_nccl_all_reduce(reinterpret_cast<void*>(comm_ptr),
                         reinterpret_cast<GlmCtx*>(ctx_ptr),
                         reinterpret_cast<const void*>(send_ptr),
                         reinterpret_cast<void*>(recv_ptr),
                         count, datatype, op);
    return env.Undefined();
}

static Napi::Value NcclAllGather(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (comm, ctx, sendbuff, recvbuff, count, datatype)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t comm_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t ctx_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t send_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t recv_ptr = info[3].As<Napi::Number>().Int64Value();
    size_t count = info[4].As<Napi::Number>().Int64Value();
    int datatype = info[5].As<Napi::Number>().Int32Value();
    glm_nccl_all_gather(reinterpret_cast<void*>(comm_ptr),
                         reinterpret_cast<GlmCtx*>(ctx_ptr),
                         reinterpret_cast<const void*>(send_ptr),
                         reinterpret_cast<void*>(recv_ptr),
                         count, datatype);
    return env.Undefined();
}

static Napi::Value NcclSend(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (comm, ctx, sendbuff, count, datatype, peer)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t comm_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t ctx_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t send_ptr = info[2].As<Napi::Number>().Int64Value();
    size_t count = info[3].As<Napi::Number>().Int64Value();
    int datatype = info[4].As<Napi::Number>().Int32Value();
    int peer = info[5].As<Napi::Number>().Int32Value();
    glm_nccl_send(reinterpret_cast<void*>(comm_ptr),
                   reinterpret_cast<GlmCtx*>(ctx_ptr),
                   reinterpret_cast<const void*>(send_ptr),
                   count, datatype, peer);
    return env.Undefined();
}

static Napi::Value NcclRecv(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (comm, ctx, recvbuff, count, datatype, peer)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t comm_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t ctx_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t recv_ptr = info[2].As<Napi::Number>().Int64Value();
    size_t count = info[3].As<Napi::Number>().Int64Value();
    int datatype = info[4].As<Napi::Number>().Int32Value();
    int peer = info[5].As<Napi::Number>().Int32Value();
    glm_nccl_recv(reinterpret_cast<void*>(comm_ptr),
                   reinterpret_cast<GlmCtx*>(ctx_ptr),
                   reinterpret_cast<void*>(recv_ptr),
                   count, datatype, peer);
    return env.Undefined();
}

static Napi::Value NcclReduceScatter(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
        Napi::TypeError::New(env, "Expected (comm, ctx, sendbuff, recvbuff, recvcount, datatype, op)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t comm_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t ctx_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t send_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t recv_ptr = info[3].As<Napi::Number>().Int64Value();
    size_t recvcount = info[4].As<Napi::Number>().Int64Value();
    int datatype = info[5].As<Napi::Number>().Int32Value();
    int op = info[6].As<Napi::Number>().Int32Value();
    glm_nccl_reduce_scatter(reinterpret_cast<void*>(comm_ptr),
                             reinterpret_cast<GlmCtx*>(ctx_ptr),
                             reinterpret_cast<const void*>(send_ptr),
                             reinterpret_cast<void*>(recv_ptr),
                             recvcount, datatype, op);
    return env.Undefined();
}

// ---------------------------------------------------------------------------
// Custom P2P AllReduce bindings
// ---------------------------------------------------------------------------

static Napi::Value P2PEnablePeerAccess(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, peerDevice)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    int peer_device = info[1].As<Napi::Number>().Int32Value();
    int rc = glm_p2p_enable_peer_access(reinterpret_cast<GlmCtx*>(ctx_ptr), peer_device);
    return Napi::Number::New(env, rc);
}

static Napi::Value P2PCreateInstance(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (ctx, myRank, deviceIds[])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    int my_rank = info[1].As<Napi::Number>().Int32Value();
    Napi::Array devArr = info[2].As<Napi::Array>();
    int world_size = devArr.Length();
    std::vector<int> device_ids(world_size);
    for (int i = 0; i < world_size; ++i) {
        device_ids[i] = devArr.Get(i).As<Napi::Number>().Int32Value();
    }
    GlmP2PInstance* inst = glm_p2p_create_instance(reinterpret_cast<GlmCtx*>(ctx_ptr),
                                                    my_rank, world_size, device_ids.data());
    return Napi::Number::New(env, reinterpret_cast<uintptr_t>(inst));
}

static Napi::Value P2PDestroyInstance(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected (instance)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t inst_ptr = info[0].As<Napi::Number>().Int64Value();
    glm_p2p_destroy_instance(reinterpret_cast<GlmP2PInstance*>(inst_ptr));
    return env.Undefined();
}

static Napi::Value P2PGetFlagPtr(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uintptr_t inst_ptr = info[0].As<Napi::Number>().Int64Value();
    void* p = glm_p2p_get_flag_ptr(reinterpret_cast<GlmP2PInstance*>(inst_ptr));
    return Napi::Number::New(env, reinterpret_cast<uintptr_t>(p));
}

static Napi::Value P2PSetPeers(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (ctx, instance, flagPtrs[])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t inst_ptr = info[1].As<Napi::Number>().Int64Value();
    Napi::Array flagArr = info[2].As<Napi::Array>();
    int N = flagArr.Length();
    std::vector<int*>  flag_ptrs(N);
    for (int i = 0; i < N; ++i) {
        flag_ptrs[i] = reinterpret_cast<int*>(flagArr.Get(i).As<Napi::Number>().Int64Value());
    }
    glm_p2p_set_peers(reinterpret_cast<GlmCtx*>(ctx_ptr),
                      reinterpret_cast<GlmP2PInstance*>(inst_ptr),
                      flag_ptrs.data());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("p2PSetPeers failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value P2PAllGatherRowWrite(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 16) {
        Napi::TypeError::New(env, "Expected (ctx, localShard, p0..p7, output, N, shardDim1Bytes, fullDim1Bytes, outer, rank)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t local_shard = info[1].As<Napi::Number>().Int64Value();
    uintptr_t p[8];
    for (int i = 0; i < 8; i++) p[i] = info[2 + i].As<Napi::Number>().Int64Value();
    uintptr_t output_ptr = info[10].As<Napi::Number>().Int64Value();
    int N = info[11].As<Napi::Number>().Int32Value();
    int shard_dim1_bytes = info[12].As<Napi::Number>().Int32Value();
    int full_dim1_bytes = info[13].As<Napi::Number>().Int32Value();
    int outer = info[14].As<Napi::Number>().Int32Value();
    int rank = info[15].As<Napi::Number>().Int32Value();
    glm_p2p_allgather_row_write(reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<const void*>(local_shard),
        reinterpret_cast<const void*>(p[0]), reinterpret_cast<const void*>(p[1]),
        reinterpret_cast<const void*>(p[2]), reinterpret_cast<const void*>(p[3]),
        reinterpret_cast<const void*>(p[4]), reinterpret_cast<const void*>(p[5]),
        reinterpret_cast<const void*>(p[6]), reinterpret_cast<const void*>(p[7]),
        reinterpret_cast<void*>(output_ptr), N, shard_dim1_bytes, full_dim1_bytes, outer, rank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("p2PAllGatherRowWrite failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value P2PReduceScatterWrite(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 13) {
        Napi::TypeError::New(env, "Expected (ctx, localShard, p0..p7, N, chunkBytes, rank)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t local_shard = info[1].As<Napi::Number>().Int64Value();
    uintptr_t p[8];
    for (int i = 0; i < 8; i++) p[i] = info[2 + i].As<Napi::Number>().Int64Value();
    int N = info[10].As<Napi::Number>().Int32Value();
    int chunk_bytes = info[11].As<Napi::Number>().Int32Value();
    int rank = info[12].As<Napi::Number>().Int32Value();
    glm_p2p_reduce_scatter_write(reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<const void*>(local_shard),
        reinterpret_cast<void*>(p[0]), reinterpret_cast<void*>(p[1]),
        reinterpret_cast<void*>(p[2]), reinterpret_cast<void*>(p[3]),
        reinterpret_cast<void*>(p[4]), reinterpret_cast<void*>(p[5]),
        reinterpret_cast<void*>(p[6]), reinterpret_cast<void*>(p[7]),
        N, chunk_bytes, rank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("p2PReduceScatterWrite failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value P2PReduceGatherWrite(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 14) {
        Napi::TypeError::New(env, "Expected (ctx, staging, p0..p7, N, chunkLen, rank, dtype)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t staging = info[1].As<Napi::Number>().Int64Value();
    uintptr_t p[8];
    for (int i = 0; i < 8; i++) p[i] = info[2 + i].As<Napi::Number>().Int64Value();
    int N = info[10].As<Napi::Number>().Int32Value();
    int chunk_len = info[11].As<Napi::Number>().Int32Value();
    int rank = info[12].As<Napi::Number>().Int32Value();
    int dtype = info[13].As<Napi::Number>().Int32Value();
    glm_p2p_reduce_gather_write(reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<const void*>(staging),
        reinterpret_cast<void*>(p[0]), reinterpret_cast<void*>(p[1]),
        reinterpret_cast<void*>(p[2]), reinterpret_cast<void*>(p[3]),
        reinterpret_cast<void*>(p[4]), reinterpret_cast<void*>(p[5]),
        reinterpret_cast<void*>(p[6]), reinterpret_cast<void*>(p[7]),
        N, chunk_len, rank, dtype);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("p2PReduceGatherWrite failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value P2PBarrier(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, instance[, peerRank])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t inst_ptr = info[1].As<Napi::Number>().Int64Value();
    int peer_rank = (info.Length() >= 3) ? info[2].As<Napi::Number>().Int32Value() : -1;
    glm_p2p_barrier(reinterpret_cast<GlmCtx*>(ctx_ptr),
                    reinterpret_cast<GlmP2PInstance*>(inst_ptr),
                    peer_rank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("p2PBarrier failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value P2PArrive(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, instance[, peerRank])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t inst_ptr = info[1].As<Napi::Number>().Int64Value();
    int peer_rank = (info.Length() >= 3) ? info[2].As<Napi::Number>().Int32Value() : -1;
    glm_p2p_arrive(reinterpret_cast<GlmCtx*>(ctx_ptr),
                   reinterpret_cast<GlmP2PInstance*>(inst_ptr),
                   peer_rank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("p2PArrive failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value P2PWait(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, instance[, peerRank])").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t inst_ptr = info[1].As<Napi::Number>().Int64Value();
    int peer_rank = (info.Length() >= 3) ? info[2].As<Napi::Number>().Int32Value() : -1;
    glm_p2p_wait(reinterpret_cast<GlmCtx*>(ctx_ptr),
                 reinterpret_cast<GlmP2PInstance*>(inst_ptr),
                 peer_rank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("p2PWait failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Value RotateInputIds(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, output_ids, input_ids, qo_indptr, new_tokens, batch_size)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t output_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t input_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t indptr_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t new_tokens_ptr = info[4].As<Napi::Number>().Int64Value();
    int batch_size = info[5].As<Napi::Number>().Int32Value();
    glm_rotate_input_ids(reinterpret_cast<GlmCtx*>(ctx_ptr),
                          reinterpret_cast<int*>(output_ptr),
                          reinterpret_cast<const int*>(input_ptr),
                          reinterpret_cast<const int*>(indptr_ptr),
                          reinterpret_cast<const int*>(new_tokens_ptr),
                          batch_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Napi::Error::New(env, std::string("rotateInputIds failed: ") + cudaGetErrorString(err)).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

static Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
    exports.Set(Napi::String::New(env, "init"), Napi::Function::New(env, Init));
    exports.Set(Napi::String::New(env, "free"), Napi::Function::New(env, Free));
    exports.Set(Napi::String::New(env, "alloc"), Napi::Function::New(env, Alloc));
    exports.Set(Napi::String::New(env, "freeBuf"), Napi::Function::New(env, FreeBuf));
    exports.Set(Napi::String::New(env, "h2d"), Napi::Function::New(env, H2D));
    exports.Set(Napi::String::New(env, "writePointers"), Napi::Function::New(env, WritePointers));
    exports.Set(Napi::String::New(env, "d2h"), Napi::Function::New(env, D2H));
    exports.Set(Napi::String::New(env, "rmsnorm"), Napi::Function::New(env, Rmsnorm));
    exports.Set(Napi::String::New(env, "fusedAddRmsnorm"), Napi::Function::New(env, FusedAddRmsnorm));
    exports.Set(Napi::String::New(env, "fusedNormRope"), Napi::Function::New(env, FusedNormRope));
    exports.Set(Napi::String::New(env, "siluAndMul"), Napi::Function::New(env, SiluAndMul));
    exports.Set(Napi::String::New(env, "linear"), Napi::Function::New(env, Linear));
    exports.Set(Napi::String::New(env, "layernorm"), Napi::Function::New(env, Layernorm));
    exports.Set(Napi::String::New(env, "relu"), Napi::Function::New(env, Relu));
    exports.Set(Napi::String::New(env, "sigmoid"), Napi::Function::New(env, Sigmoid));
    exports.Set(Napi::String::New(env, "softmax"), Napi::Function::New(env, Softmax));
    exports.Set(Napi::String::New(env, "indexerScore"), Napi::Function::New(env, IndexerScore));
    exports.Set(Napi::String::New(env, "indexerScoreTopkPrefill"), Napi::Function::New(env, IndexerScoreTopkPrefill));
    exports.Set(Napi::String::New(env, "indexerScoreTopkV2"), Napi::Function::New(env, IndexerScoreTopkV2));
    exports.Set(Napi::String::New(env, "topkToSlots"), Napi::Function::New(env, TopkToSlots));
    exports.Set(Napi::String::New(env, "fill"), Napi::Function::New(env, Fill));
    exports.Set(Napi::String::New(env, "gather"), Napi::Function::New(env, Gather));
    exports.Set(Napi::String::New(env, "scatterScalar"), Napi::Function::New(env, ScatterScalar));
    exports.Set(Napi::String::New(env, "deinterleave"), Napi::Function::New(env, Deinterleave));
    exports.Set(Napi::String::New(env, "gatherPages"), Napi::Function::New(env, GatherPages));
    exports.Set(Napi::String::New(env, "maskedFill"), Napi::Function::New(env, MaskedFill));
    exports.Set(Napi::String::New(env, "indexAdd"), Napi::Function::New(env, IndexAdd));
    exports.Set(Napi::String::New(env, "rotaryEmbedding"), Napi::Function::New(env, RotaryEmbedding));
    exports.Set(Napi::String::New(env, "applyRotaryPosEmb"), Napi::Function::New(env, ApplyRotaryPosEmb));
    exports.Set(Napi::String::New(env, "applyRotaryPosEmbPartial"), Napi::Function::New(env, ApplyRotaryPosEmbPartial));
    exports.Set(Napi::String::New(env, "ropeTranspose"), Napi::Function::New(env, RopeTranspose));
    exports.Set(Napi::String::New(env, "mlaVExpand"), Napi::Function::New(env, MlaVExpand));
    exports.Set(Napi::String::New(env, "topk"), Napi::Function::New(env, Topk));
    exports.Set(Napi::String::New(env, "topkFromScores"), Napi::Function::New(env, TopkFromScores));
    exports.Set(Napi::String::New(env, "sortTopkByIndex"), Napi::Function::New(env, SortTopkByIndex));
    exports.Set(Napi::String::New(env, "bmm"), Napi::Function::New(env, Bmm));
    exports.Set(Napi::String::New(env, "scale"), Napi::Function::New(env, Scale));
    exports.Set(Napi::String::New(env, "sumPointers"), Napi::Function::New(env, SumPointers));
    exports.Set(Napi::String::New(env, "rmsNormPointers"), Napi::Function::New(env, RmsnormPointersSmem));
    exports.Set(Napi::String::New(env, "add"), Napi::Function::New(env, Add));
    exports.Set(Napi::String::New(env, "addBroadcast"), Napi::Function::New(env, AddBroadcast));
    exports.Set(Napi::String::New(env, "transpose4d"), Napi::Function::New(env, Transpose4d));
    exports.Set(Napi::String::New(env, "mul"), Napi::Function::New(env, Mul));
    exports.Set(Napi::String::New(env, "mulBroadcast"), Napi::Function::New(env, MulBroadcast));
    exports.Set(Napi::String::New(env, "reduceSum"), Napi::Function::New(env, ReduceSum));
    exports.Set(Napi::String::New(env, "rowNormalize"), Napi::Function::New(env, RowNormalize));
    exports.Set(Napi::String::New(env, "groupMaskMul"), Napi::Function::New(env, GroupMaskMul));
    exports.Set(Napi::String::New(env, "mulMatId"), Napi::Function::New(env, MulMatId));
    exports.Set(Napi::String::New(env, "nvfp4MulMatId"), Napi::Function::New(env, Nvfp4MulMatId));
    exports.Set(Napi::String::New(env, "scatterAddRows"), Napi::Function::New(env, ScatterAddRows));
    exports.Set(Napi::String::New(env, "mmaMoeWorkspaceSize"), Napi::Function::New(env, MmaMoeWorkspaceSize));
    exports.Set(Napi::String::New(env, "nvfp4MulMatIdGroupedMmaCoop"), Napi::Function::New(env, Nvfp4MulMatIdGroupedMmaCoop));
    exports.Set(Napi::String::New(env, "mmaMoeCoopWorkspaceSize"), Napi::Function::New(env, MmaMoeCoopWorkspaceSize));
    exports.Set(Napi::String::New(env, "mmaMoeCoopScatterWorkspaceSize"), Napi::Function::New(env, MmaMoeCoopScatterWorkspaceSize));
    exports.Set(Napi::String::New(env, "mmaMoeCoopGemmWorkspaceSize"), Napi::Function::New(env, MmaMoeCoopGemmWorkspaceSize));
    exports.Set(Napi::String::New(env, "mmaMoeCoopScatter"), Napi::Function::New(env, MmaMoeCoopScatter));
    exports.Set(Napi::String::New(env, "mmaMoeCoopGemm"), Napi::Function::New(env, MmaMoeCoopGemm));
    exports.Set(Napi::String::New(env, "bf16MulMatIdGroupedMma"), Napi::Function::New(env, Bf16MulMatIdGroupedMma));
    exports.Set(Napi::String::New(env, "indexSelect"), Napi::Function::New(env, IndexSelect));
    exports.Set(Napi::String::New(env, "arange"), Napi::Function::New(env, Arange));
    exports.Set(Napi::String::New(env, "max"), Napi::Function::New(env, Max));
    exports.Set(Napi::String::New(env, "memcpy"), Napi::Function::New(env, Memcpy));
    exports.Set(Napi::String::New(env, "kvCacheWrite"), Napi::Function::New(env, KvCacheWrite));
    exports.Set(Napi::String::New(env, "synchronize"), Napi::Function::New(env, Synchronize));
    exports.Set(Napi::String::New(env, "synchronizeStream"), Napi::Function::New(env, SynchronizeStream));
    exports.Set(Napi::String::New(env, "synchronizeAsync"), Napi::Function::New(env, SynchronizeAsync));
    exports.Set(Napi::String::New(env, "synchronizeStreamAsync"), Napi::Function::New(env, SynchronizeStreamAsync));
    exports.Set(Napi::String::New(env, "setStream"), Napi::Function::New(env, SetStream));
    exports.Set(Napi::String::New(env, "eventRecord"), Napi::Function::New(env, EventRecord));
    exports.Set(Napi::String::New(env, "streamWaitEvent"), Napi::Function::New(env, StreamWaitEvent));
    exports.Set(Napi::String::New(env, "mmapOpen"), Napi::Function::New(env, MmapOpen));
    exports.Set(Napi::String::New(env, "mmapLoadAsync"), Napi::Function::New(env, MmapLoadAsync));
    exports.Set(Napi::String::New(env, "memcpy2dHostToDeviceAsync"), Napi::Function::New(env, Memcpy2dHostToDeviceAsync));
    exports.Set(Napi::String::New(env, "mmapClose"), Napi::Function::New(env, MmapClose));
    exports.Set(Napi::String::New(env, "allocPinned"), Napi::Function::New(env, AllocPinned));
    exports.Set(Napi::String::New(env, "freePinned"), Napi::Function::New(env, FreePinned));
    exports.Set(Napi::String::New(env, "hostPointerToBuffer"), Napi::Function::New(env, HostPointerToBuffer));
    exports.Set(Napi::String::New(env, "batchDecodePlan"), Napi::Function::New(env, BatchDecodePlan));
    exports.Set(Napi::String::New(env, "batchDecodeRun"), Napi::Function::New(env, BatchDecodeRun));
    exports.Set(Napi::String::New(env, "batchPrefillPagedPlan"), Napi::Function::New(env, BatchPrefillPagedPlan));
    exports.Set(Napi::String::New(env, "batchPrefillPagedRun"), Napi::Function::New(env, BatchPrefillPagedRun));
    exports.Set(Napi::String::New(env, "batchPrefillRaggedPlan"), Napi::Function::New(env, BatchPrefillRaggedPlan));
    exports.Set(Napi::String::New(env, "batchPrefillRaggedRun"), Napi::Function::New(env, BatchPrefillRaggedRun));
    exports.Set(Napi::String::New(env, "mlaPrefillPlan"), Napi::Function::New(env, MlaPrefillPlan));
    exports.Set(Napi::String::New(env, "mlaPrefillRun"), Napi::Function::New(env, MlaPrefillRun));
    exports.Set(Napi::String::New(env, "mlaDecodePlan"), Napi::Function::New(env, MlaDecodePlan));
    exports.Set(Napi::String::New(env, "mlaDecodeRun"), Napi::Function::New(env, MlaDecodeRun));
    exports.Set(Napi::String::New(env, "mlaKvCacheAppend"), Napi::Function::New(env, MlaKvCacheAppend));
    exports.Set(Napi::String::New(env, "concatAndCacheDsMla"), Napi::Function::New(env, ConcatAndCacheDsMla));
    exports.Set(Napi::String::New(env, "sparseMlaPrefill"), Napi::Function::New(env, SparseMlaPrefill));
    exports.Set(Napi::String::New(env, "sparseMlaDecode"), Napi::Function::New(env, SparseMlaDecode));
    exports.Set(Napi::String::New(env, "gatherTopkCkv"), Napi::Function::New(env, GatherTopkCkv));
    exports.Set(Napi::String::New(env, "graphBeginCapture"), Napi::Function::New(env, GraphBeginCapture));
    exports.Set(Napi::String::New(env, "graphEndCapture"), Napi::Function::New(env, GraphEndCapture));
    exports.Set(Napi::String::New(env, "graphInstantiate"), Napi::Function::New(env, GraphInstantiate));
    exports.Set(Napi::String::New(env, "graphLaunch"), Napi::Function::New(env, GraphLaunch));
    exports.Set(Napi::String::New(env, "graphDestroy"), Napi::Function::New(env, GraphDestroy));
    exports.Set(Napi::String::New(env, "graphExecDestroy"), Napi::Function::New(env, GraphExecDestroy));
    exports.Set(Napi::String::New(env, "fp8LinearDecode"), Napi::Function::New(env, Fp8LinearDecode));
    exports.Set(Napi::String::New(env, "nvfp4LinearDecode"), Napi::Function::New(env, Nvfp4LinearDecode));
    exports.Set(Napi::String::New(env, "gdnRecurrentStep"), Napi::Function::New(env, GdnRecurrentStep));
    exports.Set(Napi::String::New(env, "gdnPrefill"), Napi::Function::New(env, GdnPrefill));
    exports.Set(Napi::String::New(env, "causalConv1d"), Napi::Function::New(env, CausalConv1d));
    exports.Set(Napi::String::New(env, "causalConv1dUpdate"), Napi::Function::New(env, CausalConv1dUpdate));
    exports.Set(Napi::String::New(env, "rmsnormGated"), Napi::Function::New(env, RmsnormGated));
    exports.Set(Napi::String::New(env, "gateSigmoidMul"), Napi::Function::New(env, GateSigmoidMul));
    exports.Set(Napi::String::New(env, "cpMergeTree"), Napi::Function::New(env, CpMergeTree));
    exports.Set(Napi::String::New(env, "cpMergeScatter"), Napi::Function::New(env, CpMergeScatter));
    exports.Set(Napi::String::New(env, "cpMergeLocal"), Napi::Function::New(env, CpMergeLocal));
    exports.Set(Napi::String::New(env, "cpCorrectAttnOut"), Napi::Function::New(env, CpCorrectAttnOut));
    exports.Set(Napi::String::New(env, "sampleBatch"), Napi::Function::New(env, SampleBatch));
    exports.Set(Napi::String::New(env, "rotateInputIds"), Napi::Function::New(env, RotateInputIds));
    exports.Set(Napi::String::New(env, "memcpy2d"), Napi::Function::New(env, Memcpy2d));
    exports.Set(Napi::String::New(env, "memcpyPeer"), Napi::Function::New(env, MemcpyPeer));
    exports.Set(Napi::String::New(env, "memcpy3dPeer"), Napi::Function::New(env, Memcpy3dPeer));
    exports.Set(Napi::String::New(env, "ncclUniqueId"), Napi::Function::New(env, NcclUniqueId));
    exports.Set(Napi::String::New(env, "ncclGroupStart"), Napi::Function::New(env, NcclGroupStart));
    exports.Set(Napi::String::New(env, "ncclGroupEnd"), Napi::Function::New(env, NcclGroupEnd));
    exports.Set(Napi::String::New(env, "ncclCommInitRank"), Napi::Function::New(env, NcclCommInitRank));
    exports.Set(Napi::String::New(env, "ncclCommInitAll"), Napi::Function::New(env, NcclCommInitAll));
    exports.Set(Napi::String::New(env, "ncclCommDestroy"), Napi::Function::New(env, NcclCommDestroy));
    exports.Set(Napi::String::New(env, "ncclAllReduce"), Napi::Function::New(env, NcclAllReduce));
    exports.Set(Napi::String::New(env, "ncclAllGather"), Napi::Function::New(env, NcclAllGather));
    exports.Set(Napi::String::New(env, "ncclSend"), Napi::Function::New(env, NcclSend));
    exports.Set(Napi::String::New(env, "ncclRecv"), Napi::Function::New(env, NcclRecv));
    exports.Set(Napi::String::New(env, "ncclReduceScatter"), Napi::Function::New(env, NcclReduceScatter));
    exports.Set(Napi::String::New(env, "p2pEnablePeerAccess"), Napi::Function::New(env, P2PEnablePeerAccess));
    exports.Set(Napi::String::New(env, "p2pCreateInstance"), Napi::Function::New(env, P2PCreateInstance));
    exports.Set(Napi::String::New(env, "p2pDestroyInstance"), Napi::Function::New(env, P2PDestroyInstance));
    exports.Set(Napi::String::New(env, "p2pGetFlagPtr"), Napi::Function::New(env, P2PGetFlagPtr));
    exports.Set(Napi::String::New(env, "p2pSetPeers"), Napi::Function::New(env, P2PSetPeers));
    exports.Set(Napi::String::New(env, "p2pAllGatherRowWrite"), Napi::Function::New(env, P2PAllGatherRowWrite));
    exports.Set(Napi::String::New(env, "p2pReduceScatterWrite"), Napi::Function::New(env, P2PReduceScatterWrite));
    exports.Set(Napi::String::New(env, "p2pReduceGatherWrite"), Napi::Function::New(env, P2PReduceGatherWrite));
    exports.Set(Napi::String::New(env, "p2pBarrier"), Napi::Function::New(env, P2PBarrier));
    exports.Set(Napi::String::New(env, "p2pArrive"), Napi::Function::New(env, P2PArrive));
    exports.Set(Napi::String::New(env, "p2pWait"), Napi::Function::New(env, P2PWait));
    return exports;
}

NODE_API_MODULE(glm, InitModule)
