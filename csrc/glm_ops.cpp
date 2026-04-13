#include <napi.h>
#include "glm_ops.h"
#include <cstdint>

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
    return env.Undefined();
}

static Napi::Value SiluAndMul(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
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
    return env.Undefined();
}

static Napi::Value Linear(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, weight, batch, n, k)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t wt_ptr = info[3].As<Napi::Number>().Int64Value();
    int batch = info[4].As<Napi::Number>().Int32Value();
    int n = info[5].As<Napi::Number>().Int32Value();
    int k = info[6].As<Napi::Number>().Int32Value();
    glm_linear(reinterpret_cast<GlmCtx*>(ctx_ptr),
               reinterpret_cast<void*>(out_ptr),
               reinterpret_cast<const void*>(in_ptr),
               reinterpret_cast<const void*>(wt_ptr),
               batch, n, k);
    return env.Undefined();
}

static Napi::Value Embedding(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
        Napi::TypeError::New(env, "Expected (ctx, out, table, ids, hidden, seq_len)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t table_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t ids_ptr = info[3].As<Napi::Number>().Int64Value();
    int hidden = info[4].As<Napi::Number>().Int32Value();
    int seq_len = info[5].As<Napi::Number>().Int32Value();
    glm_embedding(reinterpret_cast<GlmCtx*>(ctx_ptr),
                  reinterpret_cast<void*>(out_ptr),
                  reinterpret_cast<const void*>(table_ptr),
                  reinterpret_cast<const int*>(ids_ptr),
                  hidden, seq_len);
    return env.Undefined();
}

static Napi::Value Layernorm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
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
    return env.Undefined();
}

static Napi::Value Relu(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
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
    return env.Undefined();
}

static Napi::Value Sigmoid(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
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
    return env.Undefined();
}

static Napi::Value Softmax(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
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
    return env.Undefined();
}

static Napi::Value CausalMask(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (ctx, out, seq_len)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    int seq_len = info[2].As<Napi::Number>().Int32Value();
    glm_causal_mask(reinterpret_cast<GlmCtx*>(ctx_ptr),
                    reinterpret_cast<void*>(out_ptr), seq_len);
    return env.Undefined();
}

static Napi::Value Fill(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (ctx, out, value, n)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    float value = info[2].As<Napi::Number>().FloatValue();
    int n = info[3].As<Napi::Number>().Int32Value();
    glm_fill(reinterpret_cast<GlmCtx*>(ctx_ptr),
             reinterpret_cast<void*>(out_ptr), value, n);
    return env.Undefined();
}

static Napi::Value Gather(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, indices, k, in_dim, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t idx_ptr = info[3].As<Napi::Number>().Int64Value();
    int k = info[4].As<Napi::Number>().Int32Value();
    int in_dim = info[5].As<Napi::Number>().Int32Value();
    int batch = info[6].As<Napi::Number>().Int32Value();
    glm_gather(reinterpret_cast<GlmCtx*>(ctx_ptr),
               reinterpret_cast<void*>(out_ptr),
               reinterpret_cast<const void*>(in_ptr),
               reinterpret_cast<const int*>(idx_ptr),
               k, in_dim, batch);
    return env.Undefined();
}

static Napi::Value ScatterScalar(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
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
    return env.Undefined();
}

static Napi::Value CatLastDim(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, a, b, a_last_dim, b_last_dim, outer)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t a_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t b_ptr = info[3].As<Napi::Number>().Int64Value();
    int a_last_dim = info[4].As<Napi::Number>().Int32Value();
    int b_last_dim = info[5].As<Napi::Number>().Int32Value();
    int outer = info[6].As<Napi::Number>().Int32Value();
    glm_cat_last_dim(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(out_ptr),
                     reinterpret_cast<const void*>(a_ptr),
                     reinterpret_cast<const void*>(b_ptr),
                     a_last_dim, b_last_dim, outer);
    return env.Undefined();
}

static Napi::Value MaskedFill(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
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
    return env.Undefined();
}

static Napi::Value IndexAdd(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
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
    return env.Undefined();
}

static Napi::Value RotaryEmbedding(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7) {
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
    return env.Undefined();
}

static Napi::Value ApplyRotaryPosEmb(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 9) {
        Napi::TypeError::New(env, "Expected (ctx, out, x, cos, sin, rope_dim, n_heads, seq_len, batch, unsqueeze_dim)").ThrowAsJavaScriptException();
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
    glm_apply_rotary_pos_emb(reinterpret_cast<GlmCtx*>(ctx_ptr),
                             reinterpret_cast<void*>(out_ptr),
                             reinterpret_cast<const void*>(x_ptr),
                             reinterpret_cast<const void*>(cos_ptr),
                             reinterpret_cast<const void*>(sin_ptr),
                             rope_dim, n_heads, seq_len, batch, unsqueeze_dim);
    return env.Undefined();
}

static Napi::Value Topk(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out_values, out_indices, input, k, dim, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_vals_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t out_idxs_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[3].As<Napi::Number>().Int64Value();
    int k = info[4].As<Napi::Number>().Int32Value();
    int dim = info[5].As<Napi::Number>().Int32Value();
    int batch = info[6].As<Napi::Number>().Int32Value();
    glm_topk(reinterpret_cast<GlmCtx*>(ctx_ptr),
             reinterpret_cast<void*>(out_vals_ptr),
             reinterpret_cast<int*>(out_idxs_ptr),
             reinterpret_cast<const void*>(in_ptr),
             k, dim, batch);
    return env.Undefined();
}

static Napi::Value Bmm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 11) {
        Napi::TypeError::New(env, "Expected (ctx, C, A, B, alpha, beta, batch, M, N, K, transB)").ThrowAsJavaScriptException();
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
    int transB = info[10].As<Napi::Number>().Int32Value();
    glm_bmm(reinterpret_cast<GlmCtx*>(ctx_ptr),
            reinterpret_cast<void*>(c_ptr),
            reinterpret_cast<const void*>(a_ptr),
            reinterpret_cast<const void*>(b_ptr),
            alpha, beta, batch, M, N, K, transB);
    return env.Undefined();
}

static Napi::Value Scale(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4) {
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
    return env.Undefined();
}

static Napi::Value Add(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4) {
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
    return env.Undefined();
}

static Napi::Value ExpandDim1(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 8) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, dim1_out, dim1_in, seq_len, head_dim, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    int dim1_out = info[3].As<Napi::Number>().Int32Value();
    int dim1_in = info[4].As<Napi::Number>().Int32Value();
    int seq_len = info[5].As<Napi::Number>().Int32Value();
    int head_dim = info[6].As<Napi::Number>().Int32Value();
    int batch = info[7].As<Napi::Number>().Int32Value();
    glm_expand_dim1(reinterpret_cast<GlmCtx*>(ctx_ptr),
                    reinterpret_cast<void*>(out_ptr),
                    reinterpret_cast<const void*>(in_ptr),
                    dim1_out, dim1_in, seq_len, head_dim, batch);
    return env.Undefined();
}

static Napi::Value Transpose4d(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 10) {
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
    return env.Undefined();
}

static Napi::Value IndexSelect(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 6) {
        Napi::TypeError::New(env, "Expected (ctx, out, src, indices, dim, k)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t src_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t idx_ptr = info[3].As<Napi::Number>().Int64Value();
    int dim = info[4].As<Napi::Number>().Int32Value();
    int k = info[5].As<Napi::Number>().Int32Value();
    glm_index_select(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(out_ptr),
                     reinterpret_cast<const void*>(src_ptr),
                     reinterpret_cast<const void*>(idx_ptr),
                     dim, k);
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
    return env.Undefined();
}

static Napi::Value Argmax(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5) {
        Napi::TypeError::New(env, "Expected (ctx, out_index, input, dim, batch)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    int dim = info[3].As<Napi::Number>().Int32Value();
    int batch = info[4].As<Napi::Number>().Int32Value();
    glm_argmax(reinterpret_cast<GlmCtx*>(ctx_ptr),
               reinterpret_cast<int*>(out_ptr),
               reinterpret_cast<const void*>(in_ptr), dim, batch);
    return env.Undefined();
}

static Napi::Value Memcpy(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4) {
        Napi::TypeError::New(env, "Expected (ctx, dst, src, bytes)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t dst_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t src_ptr = info[2].As<Napi::Number>().Int64Value();
    size_t bytes = info[3].As<Napi::Number>().Int64Value();
    glm_memcpy(reinterpret_cast<GlmCtx*>(ctx_ptr),
               reinterpret_cast<void*>(dst_ptr),
               reinterpret_cast<const void*>(src_ptr), bytes);
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
    return env.Undefined();
}

static Napi::Value ExpandDim1Strided(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 9) {
        Napi::TypeError::New(env, "Expected (ctx, out, input, dim1_out, dim1_in, seq_len, head_dim, batch, head_stride)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t out_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t in_ptr = info[2].As<Napi::Number>().Int64Value();
    int dim1_out = info[3].As<Napi::Number>().Int32Value();
    int dim1_in = info[4].As<Napi::Number>().Int32Value();
    int seq_len = info[5].As<Napi::Number>().Int32Value();
    int head_dim = info[6].As<Napi::Number>().Int32Value();
    int batch = info[7].As<Napi::Number>().Int32Value();
    int head_stride = info[8].As<Napi::Number>().Int32Value();
    glm_expand_dim1_strided(reinterpret_cast<GlmCtx*>(ctx_ptr),
                            reinterpret_cast<void*>(out_ptr),
                            reinterpret_cast<const void*>(in_ptr),
                            dim1_out, dim1_in, seq_len, head_dim, batch, head_stride);
    return env.Undefined();
}

static Napi::Value FlashPrefill(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 19) {
        Napi::TypeError::New(env, "Expected (ctx, q, k, v, o, tmp, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, v_stride_n, v_stride_h, mask_mode, kv_layout, sm_scale)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t q_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t k_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t v_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t o_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t tmp_ptr = info[5].As<Napi::Number>().Int64Value();
    int qo_len = info[6].As<Napi::Number>().Int32Value();
    int kv_len = info[7].As<Napi::Number>().Int32Value();
    int num_qo_heads = info[8].As<Napi::Number>().Int32Value();
    int num_kv_heads = info[9].As<Napi::Number>().Int32Value();
    int head_dim = info[10].As<Napi::Number>().Int32Value();
    int q_stride_n = info[11].As<Napi::Number>().Int32Value();
    int q_stride_h = info[12].As<Napi::Number>().Int32Value();
    int kv_stride_n = info[13].As<Napi::Number>().Int32Value();
    int kv_stride_h = info[14].As<Napi::Number>().Int32Value();
    int v_stride_n = info[15].As<Napi::Number>().Int32Value();
    int v_stride_h = info[16].As<Napi::Number>().Int32Value();
    int mask_mode = info[17].As<Napi::Number>().Int32Value();
    int kv_layout = info[18].As<Napi::Number>().Int32Value();
    float sm_scale = info[19].As<Napi::Number>().FloatValue();
    glm_flash_prefill(reinterpret_cast<GlmCtx*>(ctx_ptr),
                      reinterpret_cast<void*>(q_ptr),
                      reinterpret_cast<void*>(k_ptr),
                      reinterpret_cast<void*>(v_ptr),
                      reinterpret_cast<void*>(o_ptr),
                      reinterpret_cast<void*>(tmp_ptr),
                      qo_len, kv_len,
                      num_qo_heads, num_kv_heads, head_dim,
                      q_stride_n, q_stride_h,
                      kv_stride_n, kv_stride_h,
                      v_stride_n, v_stride_h,
                      mask_mode, kv_layout, sm_scale);
    return env.Undefined();
}

static Napi::Value FlashDecode(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 14) {
        Napi::TypeError::New(env, "Expected (ctx, q, k, v, o, tmp, kv_len, num_qo_heads, num_kv_heads, head_dim, q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, sm_scale)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uintptr_t ctx_ptr = info[0].As<Napi::Number>().Int64Value();
    uintptr_t q_ptr = info[1].As<Napi::Number>().Int64Value();
    uintptr_t k_ptr = info[2].As<Napi::Number>().Int64Value();
    uintptr_t v_ptr = info[3].As<Napi::Number>().Int64Value();
    uintptr_t o_ptr = info[4].As<Napi::Number>().Int64Value();
    uintptr_t tmp_ptr = info[5].As<Napi::Number>().Int64Value();
    int kv_len = info[6].As<Napi::Number>().Int32Value();
    int num_qo_heads = info[7].As<Napi::Number>().Int32Value();
    int num_kv_heads = info[8].As<Napi::Number>().Int32Value();
    int head_dim = info[9].As<Napi::Number>().Int32Value();
    int q_stride_n = info[10].As<Napi::Number>().Int32Value();
    int q_stride_h = info[11].As<Napi::Number>().Int32Value();
    int kv_stride_n = info[12].As<Napi::Number>().Int32Value();
    int kv_stride_h = info[13].As<Napi::Number>().Int32Value();
    float sm_scale = info[14].As<Napi::Number>().FloatValue();
    glm_flash_decode(reinterpret_cast<GlmCtx*>(ctx_ptr),
                     reinterpret_cast<void*>(q_ptr),
                     reinterpret_cast<void*>(k_ptr),
                     reinterpret_cast<void*>(v_ptr),
                     reinterpret_cast<void*>(o_ptr),
                     reinterpret_cast<void*>(tmp_ptr),
                     kv_len,
                     num_qo_heads, num_kv_heads, head_dim,
                     q_stride_n, q_stride_h,
                     kv_stride_n, kv_stride_h,
                     sm_scale);
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

static Napi::Value MmapLoad(const Napi::CallbackInfo& info) {
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
    glm_mmap_load(reinterpret_cast<GlmCtx*>(ctx_ptr),
                  reinterpret_cast<void*>(gpu_dst),
                  reinterpret_cast<const void*>(mmap_ptr),
                  offset, nbytes);
    return env.Undefined();
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

static Napi::Value BatchDecodePlan(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 11) {
        Napi::TypeError::New(env, "Expected (ctx, float_ws, float_ws_size, int_ws, pinned_int_ws, int_ws_size, plan_info, indptr_h, batch_size, num_qo_heads, num_kv_heads, page_size)").ThrowAsJavaScriptException();
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
    uint32_t page_size = info[11].As<Napi::Number>().Uint32Value();
    glm_batch_decode_plan(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(float_ws), float_ws_size,
        reinterpret_cast<void*>(int_ws), reinterpret_cast<void*>(pinned_int_ws), int_ws_size,
        reinterpret_cast<int64_t*>(plan_info_ptr),
        reinterpret_cast<int32_t*>(indptr_h_ptr),
        batch_size, num_qo_heads, num_kv_heads, page_size);
    return env.Undefined();
}

static Napi::Value BatchDecodeRun(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 12) {
        Napi::TypeError::New(env, "Expected (ctx, q, o, k_data, v_data, indices, indptr_d, last_page_len, float_ws, int_ws, plan_info, num_qo_heads, num_kv_heads, head_dim, page_size, sm_scale)").ThrowAsJavaScriptException();
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
    uint32_t num_qo_heads = info[11].As<Napi::Number>().Uint32Value();
    uint32_t num_kv_heads = info[12].As<Napi::Number>().Uint32Value();
    uint32_t head_dim = info[13].As<Napi::Number>().Uint32Value();
    uint32_t page_size = info[14].As<Napi::Number>().Uint32Value();
    float sm_scale = info[15].As<Napi::Number>().FloatValue();
    glm_batch_decode_run(
        reinterpret_cast<GlmCtx*>(ctx_ptr),
        reinterpret_cast<void*>(q_ptr), reinterpret_cast<void*>(o_ptr),
        reinterpret_cast<void*>(k_data_ptr), reinterpret_cast<void*>(v_data_ptr),
        reinterpret_cast<int32_t*>(indices_ptr),
        reinterpret_cast<int32_t*>(indptr_d_ptr),
        reinterpret_cast<int32_t*>(last_page_len_ptr),
        reinterpret_cast<void*>(float_ws_ptr), reinterpret_cast<void*>(int_ws_ptr),
        reinterpret_cast<int64_t*>(plan_info_ptr),
        num_qo_heads, num_kv_heads, head_dim, page_size, sm_scale);
    return env.Undefined();
}

static Napi::Value BatchPrefillRaggedPlan(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 12) {
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
    return env.Undefined();
}

static Napi::Value BatchPrefillRaggedRun(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 18) {
        Napi::TypeError::New(env, "Expected (ctx, q, k, v, o, float_ws, int_ws, q_indptr_d, kv_indptr_d, plan_info, total_qo_rows, batch_size, num_qo_heads, num_kv_heads, head_dim, q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, mask_mode, sm_scale)").ThrowAsJavaScriptException();
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
    uint32_t q_stride_n = info[15].As<Napi::Number>().Uint32Value();
    uint32_t q_stride_h = info[16].As<Napi::Number>().Uint32Value();
    uint32_t kv_stride_n = info[17].As<Napi::Number>().Uint32Value();
    uint32_t kv_stride_h = info[18].As<Napi::Number>().Uint32Value();
    int mask_mode = info[19].As<Napi::Number>().Int32Value();
    float sm_scale = info[20].As<Napi::Number>().FloatValue();
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
        q_stride_n, q_stride_h, kv_stride_n, kv_stride_h,
        mask_mode, sm_scale);
    return env.Undefined();
}

static Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
    exports.Set(Napi::String::New(env, "init"), Napi::Function::New(env, Init));
    exports.Set(Napi::String::New(env, "free"), Napi::Function::New(env, Free));
    exports.Set(Napi::String::New(env, "alloc"), Napi::Function::New(env, Alloc));
    exports.Set(Napi::String::New(env, "freeBuf"), Napi::Function::New(env, FreeBuf));
    exports.Set(Napi::String::New(env, "h2d"), Napi::Function::New(env, H2D));
    exports.Set(Napi::String::New(env, "d2h"), Napi::Function::New(env, D2H));
    exports.Set(Napi::String::New(env, "rmsnorm"), Napi::Function::New(env, Rmsnorm));
    exports.Set(Napi::String::New(env, "siluAndMul"), Napi::Function::New(env, SiluAndMul));
    exports.Set(Napi::String::New(env, "linear"), Napi::Function::New(env, Linear));
    exports.Set(Napi::String::New(env, "embedding"), Napi::Function::New(env, Embedding));
    exports.Set(Napi::String::New(env, "layernorm"), Napi::Function::New(env, Layernorm));
    exports.Set(Napi::String::New(env, "relu"), Napi::Function::New(env, Relu));
    exports.Set(Napi::String::New(env, "sigmoid"), Napi::Function::New(env, Sigmoid));
    exports.Set(Napi::String::New(env, "softmax"), Napi::Function::New(env, Softmax));
    exports.Set(Napi::String::New(env, "causalMask"), Napi::Function::New(env, CausalMask));
    exports.Set(Napi::String::New(env, "fill"), Napi::Function::New(env, Fill));
    exports.Set(Napi::String::New(env, "gather"), Napi::Function::New(env, Gather));
    exports.Set(Napi::String::New(env, "scatterScalar"), Napi::Function::New(env, ScatterScalar));
    exports.Set(Napi::String::New(env, "catLastDim"), Napi::Function::New(env, CatLastDim));
    exports.Set(Napi::String::New(env, "maskedFill"), Napi::Function::New(env, MaskedFill));
    exports.Set(Napi::String::New(env, "indexAdd"), Napi::Function::New(env, IndexAdd));
    exports.Set(Napi::String::New(env, "rotaryEmbedding"), Napi::Function::New(env, RotaryEmbedding));
    exports.Set(Napi::String::New(env, "applyRotaryPosEmb"), Napi::Function::New(env, ApplyRotaryPosEmb));
    exports.Set(Napi::String::New(env, "topk"), Napi::Function::New(env, Topk));
    exports.Set(Napi::String::New(env, "bmm"), Napi::Function::New(env, Bmm));
    exports.Set(Napi::String::New(env, "scale"), Napi::Function::New(env, Scale));
    exports.Set(Napi::String::New(env, "add"), Napi::Function::New(env, Add));
    exports.Set(Napi::String::New(env, "expandDim1"), Napi::Function::New(env, ExpandDim1));
    exports.Set(Napi::String::New(env, "transpose4d"), Napi::Function::New(env, Transpose4d));
    exports.Set(Napi::String::New(env, "mul"), Napi::Function::New(env, Mul));
    exports.Set(Napi::String::New(env, "reduceSum"), Napi::Function::New(env, ReduceSum));
    exports.Set(Napi::String::New(env, "indexSelect"), Napi::Function::New(env, IndexSelect));
    exports.Set(Napi::String::New(env, "arange"), Napi::Function::New(env, Arange));
    exports.Set(Napi::String::New(env, "argmax"), Napi::Function::New(env, Argmax));
    exports.Set(Napi::String::New(env, "memcpy"), Napi::Function::New(env, Memcpy));
    exports.Set(Napi::String::New(env, "synchronize"), Napi::Function::New(env, Synchronize));
    exports.Set(Napi::String::New(env, "expandDim1Strided"), Napi::Function::New(env, ExpandDim1Strided));
    exports.Set(Napi::String::New(env, "flashPrefill"), Napi::Function::New(env, FlashPrefill));
    exports.Set(Napi::String::New(env, "flashDecode"), Napi::Function::New(env, FlashDecode));
    exports.Set(Napi::String::New(env, "mmapOpen"), Napi::Function::New(env, MmapOpen));
    exports.Set(Napi::String::New(env, "mmapLoad"), Napi::Function::New(env, MmapLoad));
    exports.Set(Napi::String::New(env, "mmapClose"), Napi::Function::New(env, MmapClose));
    exports.Set(Napi::String::New(env, "allocPinned"), Napi::Function::New(env, AllocPinned));
    exports.Set(Napi::String::New(env, "freePinned"), Napi::Function::New(env, FreePinned));
    exports.Set(Napi::String::New(env, "batchDecodePlan"), Napi::Function::New(env, BatchDecodePlan));
    exports.Set(Napi::String::New(env, "batchDecodeRun"), Napi::Function::New(env, BatchDecodeRun));
    exports.Set(Napi::String::New(env, "batchPrefillRaggedPlan"), Napi::Function::New(env, BatchPrefillRaggedPlan));
    exports.Set(Napi::String::New(env, "batchPrefillRaggedRun"), Napi::Function::New(env, BatchPrefillRaggedRun));
    return exports;
}

NODE_API_MODULE(glm, InitModule)
