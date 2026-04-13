import json
import math
import ctypes
import os
import pytest
import torch
from safetensors import safe_open
from helpers import ATOL, RTOL, GlmOps, get_model_path, has_model_cached

QWEN3_REPO = "Qwen/Qwen3-0.6B"

def load_qwen3_config():
    model_dir = get_model_path(QWEN3_REPO)
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)

def load_qwen3_weights(device="cpu"):
    model_dir = get_model_path(QWEN3_REPO)
    st_path = os.path.join(model_dir, "model.safetensors")
    weights = {}
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).to(device=device)
    return weights


class Qwen3Config:
    def __init__(self, d=None):
        self.hidden_size = d["hidden_size"]
        self.num_attention_heads = d["num_attention_heads"]
        self.num_key_value_heads = d["num_key_value_heads"]
        self.head_dim = d["head_dim"]
        self.intermediate_size = d["intermediate_size"]
        self.num_hidden_layers = d["num_hidden_layers"]
        self.rms_norm_eps = d["rms_norm_eps"]
        self.rope_theta = d["rope_theta"]
        self.vocab_size = d["vocab_size"]
        self.tie_word_embeddings = d.get("tie_word_embeddings", False)
        self.attention_bias = d.get("attention_bias", False)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def make_rope_inv_freq(head_dim, rope_theta, device):
    return (1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))).to(torch.bfloat16)

def make_rope_cos_sin(glm, device, head_dim, rope_theta, batch, seq_len):
    inv_freq = make_rope_inv_freq(head_dim, rope_theta, device)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1)
    cos_buf = torch.empty(batch, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    sin_buf = torch.empty(batch, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    glm.rotary_embedding(cos_buf, sin_buf, inv_freq, position_ids, head_dim // 2, batch, seq_len)
    return cos_buf, sin_buf

def rope_cos_sin_torch(head_dim, rope_theta, position_ids, device):
    inv_freq = (1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))).bfloat16()
    inv_freq_exp = inv_freq[None, :, None].float()
    pos_exp = position_ids[:, None, :].float()
    freqs = (inv_freq_exp @ pos_exp).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().bfloat16(), emb.sin().bfloat16()

def rotate_half_torch(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_torch(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half_torch(q) * sin)
    k_embed = (k * cos) + (rotate_half_torch(k) * sin)
    return q_embed, k_embed

def repeat_kv_torch(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------

def qwen3_rmsnorm_torch(x, weight, eps):
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)

def qwen3_mlp_torch(x, gate_proj_w, up_proj_w, down_proj_w):
    gate = torch.nn.functional.linear(x, gate_proj_w)
    up = torch.nn.functional.linear(x, up_proj_w)
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_proj_w)

def qwen3_attention_torch(hidden_states, cos, sin, causal_mask,
                           q_proj_w, k_proj_w, v_proj_w, o_proj_w,
                           q_norm_w, k_norm_w,
                           num_heads, num_kv_heads, head_dim, scaling, eps=1e-6):
    B, S, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    q = torch.nn.functional.linear(hidden_states, q_proj_w)
    q = q.view(B, S, num_heads, head_dim)
    q = qwen3_rmsnorm_torch(q, q_norm_w, eps)
    q = q.transpose(1, 2)

    k = torch.nn.functional.linear(hidden_states, k_proj_w)
    k = k.view(B, S, num_kv_heads, head_dim)
    k = qwen3_rmsnorm_torch(k, k_norm_w, eps)
    k = k.transpose(1, 2)

    v = torch.nn.functional.linear(hidden_states, v_proj_w)
    v = v.view(B, S, num_kv_heads, head_dim)
    v = v.transpose(1, 2)

    q, k = apply_rotary_pos_emb_torch(q, k, cos, sin)

    k = repeat_kv_torch(k, num_kv_groups)
    v = repeat_kv_torch(v, num_kv_groups)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
    if causal_mask is not None:
        attn_weights = attn_weights + causal_mask
    attn_weights = torch.nn.functional.softmax(attn_weights.float(), dim=-1).to(torch.bfloat16)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
    return torch.nn.functional.linear(attn_output, o_proj_w)

def qwen3_decoder_layer_torch(hidden_states, cos, sin, causal_mask, weights, layer_idx, cfg):
    B, S, _ = hidden_states.shape
    pfx = f"model.layers.{layer_idx}"

    residual = hidden_states
    hidden_states = qwen3_rmsnorm_torch(hidden_states, weights[f"{pfx}.input_layernorm.weight"], cfg.rms_norm_eps)

    attn_out = qwen3_attention_torch(
        hidden_states, cos, sin, causal_mask,
        weights[f"{pfx}.self_attn.q_proj.weight"],
        weights[f"{pfx}.self_attn.k_proj.weight"],
        weights[f"{pfx}.self_attn.v_proj.weight"],
        weights[f"{pfx}.self_attn.o_proj.weight"],
        weights[f"{pfx}.self_attn.q_norm.weight"],
        weights[f"{pfx}.self_attn.k_norm.weight"],
        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.scaling, cfg.rms_norm_eps)
    hidden_states = residual + attn_out

    residual = hidden_states
    hidden_states = qwen3_rmsnorm_torch(hidden_states, weights[f"{pfx}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
    mlp_out = qwen3_mlp_torch(hidden_states,
        weights[f"{pfx}.mlp.gate_proj.weight"],
        weights[f"{pfx}.mlp.up_proj.weight"],
        weights[f"{pfx}.mlp.down_proj.weight"])
    hidden_states = residual + mlp_out
    return hidden_states

def qwen3_model_torch(input_ids, weights, cfg):
    B, S = input_ids.shape
    device = input_ids.device

    hidden = torch.nn.functional.embedding(input_ids, weights["model.embed_tokens.weight"])

    position_ids = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0).expand(B, -1)
    cos, sin = rope_cos_sin_torch(cfg.head_dim, cfg.rope_theta, position_ids, device)

    causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=device, dtype=torch.bfloat16), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    for i in range(cfg.num_hidden_layers):
        hidden = qwen3_decoder_layer_torch(hidden, cos, sin, causal_mask, weights, i, cfg)

    hidden = qwen3_rmsnorm_torch(hidden, weights["model.norm.weight"], cfg.rms_norm_eps)
    logits = torch.nn.functional.linear(hidden, weights["lm_head.weight"])
    return logits


# ---------------------------------------------------------------------------
# CUDA implementation
# ---------------------------------------------------------------------------

def qwen3_attention_cuda(glm, device, hidden_states, cos, sin, causal_mask,
                          q_proj_w, k_proj_w, v_proj_w, o_proj_w,
                          q_norm_w, k_norm_w,
                          num_heads, num_kv_heads, head_dim, scaling, eps=1e-6):
    B, S, hidden_size = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    hidden_flat = hidden_states.reshape(B * S, hidden_size)

    q_out = torch.empty(B * S, num_heads * head_dim, dtype=torch.bfloat16, device=device)
    glm.linear(q_out, hidden_flat, q_proj_w, B * S, num_heads * head_dim, hidden_size)
    q_4d = q_out.reshape(B, S, num_heads, head_dim)
    q_normed = torch.empty_like(q_4d)
    glm.rmsnorm(q_normed.reshape(B * S * num_heads, head_dim),
                q_4d.reshape(B * S * num_heads, head_dim),
                q_norm_w, eps, head_dim, B * S * num_heads)
    q_t = torch.empty(B, num_heads, S, head_dim, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(q_t.reshape(-1), q_normed.reshape(-1),
                     B, S, num_heads, head_dim, 0, 2, 1, 3)

    k_out = torch.empty(B * S, num_kv_heads * head_dim, dtype=torch.bfloat16, device=device)
    glm.linear(k_out, hidden_flat, k_proj_w, B * S, num_kv_heads * head_dim, hidden_size)
    k_4d = k_out.reshape(B, S, num_kv_heads, head_dim)
    k_normed = torch.empty_like(k_4d)
    glm.rmsnorm(k_normed.reshape(B * S * num_kv_heads, head_dim),
                k_4d.reshape(B * S * num_kv_heads, head_dim),
                k_norm_w, eps, head_dim, B * S * num_kv_heads)
    k_t = torch.empty(B, num_kv_heads, S, head_dim, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(k_t.reshape(-1), k_normed.reshape(-1),
                     B, S, num_kv_heads, head_dim, 0, 2, 1, 3)

    v_out = torch.empty(B * S, num_kv_heads * head_dim, dtype=torch.bfloat16, device=device)
    glm.linear(v_out, hidden_flat, v_proj_w, B * S, num_kv_heads * head_dim, hidden_size)
    v_4d = v_out.reshape(B, S, num_kv_heads, head_dim)
    v_t = torch.empty(B, num_kv_heads, S, head_dim, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(v_t.reshape(-1), v_4d.reshape(-1),
                     B, S, num_kv_heads, head_dim, 0, 2, 1, 3)

    q_rope = torch.empty_like(q_t)
    glm.apply_rotary_pos_emb(q_rope, q_t, cos, sin, head_dim, num_heads, S, B, 1)
    k_rope = torch.empty_like(k_t)
    glm.apply_rotary_pos_emb(k_rope, k_t, cos, sin, head_dim, num_kv_heads, S, B, 1)

    if num_kv_groups > 1:
        k_expanded = torch.empty(B, num_heads, S, head_dim, dtype=torch.bfloat16, device=device)
        glm.expand_dim1(k_expanded.reshape(-1), k_rope.reshape(-1),
                        num_heads, num_kv_heads, S, head_dim, B)
        v_expanded = torch.empty(B, num_heads, S, head_dim, dtype=torch.bfloat16, device=device)
        glm.expand_dim1(v_expanded.reshape(-1), v_t.reshape(-1),
                        num_heads, num_kv_heads, S, head_dim, B)
    else:
        k_expanded = k_rope
        v_expanded = v_t

    attn_scores = torch.empty(B * num_heads, S, S, dtype=torch.bfloat16, device=device)
    glm.bmm(attn_scores, q_rope.reshape(B * num_heads, S, head_dim),
            k_expanded.reshape(B * num_heads, S, head_dim),
            scaling, 0.0, B * num_heads, S, S, head_dim, 1)

    if causal_mask is not None:
        mask_expanded = torch.empty(B, num_heads, S, S, dtype=torch.bfloat16, device=device)
        glm.expand_dim1(mask_expanded.reshape(-1), causal_mask.reshape(-1),
                        num_heads, 1, S, S, B)
        softmax_mask = mask_expanded.reshape(B * num_heads * S, S)
    else:
        softmax_mask = None

    glm.softmax(attn_scores.reshape(-1, S), attn_scores.reshape(-1, S),
                softmax_mask.reshape(-1, S) if softmax_mask is not None else None,
                S, B * num_heads * S)

    attn_out = torch.empty(B * num_heads, S, head_dim, dtype=torch.bfloat16, device=device)
    glm.bmm(attn_out, attn_scores.reshape(B * num_heads, S, S),
            v_expanded.reshape(B * num_heads, S, head_dim),
            1.0, 0.0, B * num_heads, S, head_dim, S, 0)

    attn_out_t = torch.empty(B, S, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(attn_out_t.reshape(-1), attn_out.reshape(-1),
                     B, num_heads, S, head_dim, 0, 2, 1, 3)

    output = torch.empty(B * S, hidden_size, dtype=torch.bfloat16, device=device)
    glm.linear(output, attn_out_t.reshape(B * S, num_heads * head_dim), o_proj_w,
               B * S, hidden_size, num_heads * head_dim)
    return output.reshape(B, S, hidden_size)

def qwen3_mlp_cuda(glm, device, x, gate_proj_w, up_proj_w, down_proj_w, hidden_size, intermediate_size):
    B, S, _ = x.shape
    x_flat = x.reshape(B * S, hidden_size)
    gate = torch.empty(B * S, intermediate_size, dtype=torch.bfloat16, device=device)
    up = torch.empty(B * S, intermediate_size, dtype=torch.bfloat16, device=device)
    glm.linear(gate, x_flat, gate_proj_w, B * S, intermediate_size, hidden_size)
    glm.linear(up, x_flat, up_proj_w, B * S, intermediate_size, hidden_size)
    silu_out = torch.empty(B * S, intermediate_size, dtype=torch.bfloat16, device=device)
    glm.silu_and_mul(silu_out, gate, up, intermediate_size, B * S)
    down = torch.empty(B * S, hidden_size, dtype=torch.bfloat16, device=device)
    glm.linear(down, silu_out, down_proj_w, B * S, hidden_size, intermediate_size)
    return down.reshape(B, S, hidden_size)

def qwen3_decoder_layer_cuda(glm, device, hidden_states, cos, sin, causal_mask, weights, layer_idx, cfg):
    B, S, hidden_size = hidden_states.shape
    pfx = f"model.layers.{layer_idx}"

    residual = hidden_states
    normed = torch.empty_like(hidden_states)
    glm.rmsnorm(normed.reshape(B * S, hidden_size),
                hidden_states.reshape(B * S, hidden_size),
                weights[f"{pfx}.input_layernorm.weight"],
                cfg.rms_norm_eps, hidden_size, B * S)

    attn_out = qwen3_attention_cuda(
        glm, device, normed, cos, sin, causal_mask,
        weights[f"{pfx}.self_attn.q_proj.weight"],
        weights[f"{pfx}.self_attn.k_proj.weight"],
        weights[f"{pfx}.self_attn.v_proj.weight"],
        weights[f"{pfx}.self_attn.o_proj.weight"],
        weights[f"{pfx}.self_attn.q_norm.weight"],
        weights[f"{pfx}.self_attn.k_norm.weight"],
        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.scaling, cfg.rms_norm_eps)

    hidden_states = torch.empty_like(residual)
    glm.add(hidden_states.reshape(-1), residual.reshape(-1), attn_out.reshape(-1), B * S * hidden_size)

    residual = hidden_states
    normed2 = torch.empty_like(hidden_states)
    glm.rmsnorm(normed2.reshape(B * S, hidden_size),
                hidden_states.reshape(B * S, hidden_size),
                weights[f"{pfx}.post_attention_layernorm.weight"],
                cfg.rms_norm_eps, hidden_size, B * S)

    mlp_out = qwen3_mlp_cuda(glm, device, normed2,
        weights[f"{pfx}.mlp.gate_proj.weight"],
        weights[f"{pfx}.mlp.up_proj.weight"],
        weights[f"{pfx}.mlp.down_proj.weight"],
        hidden_size, cfg.intermediate_size)

    hidden_states = torch.empty_like(residual)
    glm.add(hidden_states.reshape(-1), residual.reshape(-1), mlp_out.reshape(-1), B * S * hidden_size)
    return hidden_states

def qwen3_model_cuda(glm, device, input_ids, weights, cfg):
    B, S = input_ids.shape
    hidden_size = cfg.hidden_size

    hidden_flat = torch.empty(B * S, hidden_size, dtype=torch.bfloat16, device=device)
    ids_i32 = input_ids.to(torch.int32)
    glm.embedding(hidden_flat, weights["model.embed_tokens.weight"],
                  ids_i32.reshape(-1), hidden_size, B * S)
    hidden = hidden_flat.reshape(B, S, hidden_size)

    cos, sin = make_rope_cos_sin(glm, device, cfg.head_dim, cfg.rope_theta, B, S)

    causal_mask = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_mask, S)

    for i in range(cfg.num_hidden_layers):
        hidden = qwen3_decoder_layer_cuda(glm, device, hidden, cos, sin, causal_mask, weights, i, cfg)

    normed = torch.empty_like(hidden)
    glm.rmsnorm(normed.reshape(B * S, hidden_size),
                hidden.reshape(B * S, hidden_size),
                weights["model.norm.weight"],
                cfg.rms_norm_eps, hidden_size, B * S)

    logits = torch.empty(B * S, cfg.vocab_size, dtype=torch.bfloat16, device=device)
    glm.linear(logits, normed.reshape(B * S, hidden_size),
               weights["lm_head.weight"], B * S, cfg.vocab_size, hidden_size)
    return logits.reshape(B, S, cfg.vocab_size)


# ---------------------------------------------------------------------------
# Tests with small random weights
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qwen3_small():
    cfg = Qwen3Config({
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "vocab_size": 1024,
        "tie_word_embeddings": True,
        "attention_bias": False,
    })
    return cfg


def test_qwen3_rmsnorm(glm, device, qwen3_small):
    cfg = qwen3_small
    B, S, D = 2, 8, cfg.hidden_size
    torch.manual_seed(42)
    x = torch.randn(B, S, D, dtype=torch.bfloat16, device=device)
    w = torch.randn(D, dtype=torch.bfloat16, device=device)

    ref = qwen3_rmsnorm_torch(x, w, cfg.rms_norm_eps)
    out = torch.empty_like(x)
    glm.rmsnorm(out.reshape(B * S, D), x.reshape(B * S, D), w, cfg.rms_norm_eps, D, B * S)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3)


def test_qwen3_mlp(glm, device, qwen3_small):
    cfg = qwen3_small
    B, S = 1, 4
    torch.manual_seed(42)
    x = torch.randn(B, S, cfg.hidden_size, dtype=torch.bfloat16, device=device) * 0.1
    gate_w = torch.randn(cfg.intermediate_size, cfg.hidden_size, dtype=torch.bfloat16, device=device) * (cfg.hidden_size ** -0.5)
    up_w = torch.randn(cfg.intermediate_size, cfg.hidden_size, dtype=torch.bfloat16, device=device) * (cfg.hidden_size ** -0.5)
    down_w = torch.randn(cfg.hidden_size, cfg.intermediate_size, dtype=torch.bfloat16, device=device) * (cfg.intermediate_size ** -0.5)

    ref = qwen3_mlp_torch(x, gate_w, up_w, down_w)
    cuda = qwen3_mlp_cuda(glm, device, x, gate_w, up_w, down_w, cfg.hidden_size, cfg.intermediate_size)
    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_qwen3_attention(glm, device, qwen3_small):
    cfg = qwen3_small
    B, S = 1, 4
    torch.manual_seed(42)
    hidden = torch.randn(B, S, cfg.hidden_size, dtype=torch.bfloat16, device=device) * 0.1

    cos_cuda, sin_cuda = make_rope_cos_sin(glm, device, cfg.head_dim, cfg.rope_theta, B, S)

    position_ids = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0).expand(B, -1)
    cos_torch, sin_torch = rope_cos_sin_torch(cfg.head_dim, cfg.rope_theta, position_ids, device)

    causal_mask = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_mask, S)
    causal_mask_torch = causal_mask.unsqueeze(0).unsqueeze(0)

    scale = cfg.hidden_size ** -0.5
    q_proj_w = torch.randn(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, dtype=torch.bfloat16, device=device) * scale
    k_proj_w = torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size, dtype=torch.bfloat16, device=device) * scale
    v_proj_w = torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size, dtype=torch.bfloat16, device=device) * scale
    o_proj_w = torch.randn(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, dtype=torch.bfloat16, device=device) * scale
    q_norm_w = torch.ones(cfg.head_dim, dtype=torch.bfloat16, device=device)
    k_norm_w = torch.ones(cfg.head_dim, dtype=torch.bfloat16, device=device)

    ref = qwen3_attention_torch(hidden, cos_torch, sin_torch, causal_mask_torch,
                                 q_proj_w, k_proj_w, v_proj_w, o_proj_w,
                                 q_norm_w, k_norm_w,
                                 cfg.num_attention_heads, cfg.num_key_value_heads,
                                 cfg.head_dim, cfg.scaling, cfg.rms_norm_eps)
    cuda = qwen3_attention_cuda(glm, device, hidden, cos_cuda, sin_cuda, causal_mask,
                                 q_proj_w, k_proj_w, v_proj_w, o_proj_w,
                                 q_norm_w, k_norm_w,
                                 cfg.num_attention_heads, cfg.num_key_value_heads,
                                  cfg.head_dim, cfg.scaling, cfg.rms_norm_eps)
    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=0.5, rtol=0.1)


def test_qwen3_decoder_layer(glm, device, qwen3_small):
    cfg = qwen3_small
    B, S = 1, 4
    torch.manual_seed(42)
    hidden = torch.randn(B, S, cfg.hidden_size, dtype=torch.bfloat16, device=device) * 0.1

    cos_cuda, sin_cuda = make_rope_cos_sin(glm, device, cfg.head_dim, cfg.rope_theta, B, S)
    position_ids = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0).expand(B, -1)
    cos_torch, sin_torch = rope_cos_sin_torch(cfg.head_dim, cfg.rope_theta, position_ids, device)

    causal_mask = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_mask, S)
    causal_mask_torch = causal_mask.unsqueeze(0).unsqueeze(0)

    weights = {}
    layer_idx = 0
    pfx = f"model.layers.{layer_idx}"
    hs = cfg.hidden_size
    isz = cfg.intermediate_size
    for name, shape in [
        ("input_layernorm.weight", (hs,)),
        ("post_attention_layernorm.weight", (hs,)),
        ("self_attn.q_proj.weight", (cfg.num_attention_heads * cfg.head_dim, hs)),
        ("self_attn.k_proj.weight", (cfg.num_key_value_heads * cfg.head_dim, hs)),
        ("self_attn.v_proj.weight", (cfg.num_key_value_heads * cfg.head_dim, hs)),
        ("self_attn.o_proj.weight", (hs, cfg.num_attention_heads * cfg.head_dim)),
        ("self_attn.q_norm.weight", (cfg.head_dim,)),
        ("self_attn.k_norm.weight", (cfg.head_dim,)),
        ("mlp.gate_proj.weight", (isz, hs)),
        ("mlp.up_proj.weight", (isz, hs)),
        ("mlp.down_proj.weight", (hs, isz)),
    ]:
        fan_in = shape[1] if len(shape) == 2 else shape[0]
        weights[f"{pfx}.{name}"] = torch.randn(*shape, dtype=torch.bfloat16, device=device) * (fan_in ** -0.5)

    ref = qwen3_decoder_layer_torch(hidden, cos_torch, sin_torch, causal_mask_torch, weights, layer_idx, cfg)
    cuda = qwen3_decoder_layer_cuda(glm, device, hidden, cos_cuda, sin_cuda, causal_mask, weights, layer_idx, cfg)
    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=0.5, rtol=0.1)


# ---------------------------------------------------------------------------
# Tests with real Qwen3-0.6B weights
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_model_cached(QWEN3_REPO),
                    reason=f"{QWEN3_REPO} not in HF cache")
def test_qwen3_real_decoder_layer(glm, device):
    cfg = Qwen3Config(load_qwen3_config())
    weights = load_qwen3_weights(device)
    B, S = 1, 8
    torch.manual_seed(42)
    hidden = torch.randn(B, S, cfg.hidden_size, dtype=torch.bfloat16, device=device)

    cos_cuda, sin_cuda = make_rope_cos_sin(glm, device, cfg.head_dim, cfg.rope_theta, B, S)
    position_ids = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0).expand(B, -1)
    cos_torch, sin_torch = rope_cos_sin_torch(cfg.head_dim, cfg.rope_theta, position_ids, device)

    causal_mask = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_mask, S)
    causal_mask_torch = causal_mask.unsqueeze(0).unsqueeze(0)

    layer_idx = 0
    ref = qwen3_decoder_layer_torch(hidden, cos_torch, sin_torch, causal_mask_torch, weights, layer_idx, cfg)
    cuda = qwen3_decoder_layer_cuda(glm, device, hidden, cos_cuda, sin_cuda, causal_mask, weights, layer_idx, cfg)
    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=0.5, rtol=1e-1)


@pytest.mark.skipif(not has_model_cached(QWEN3_REPO),
                    reason=f"{QWEN3_REPO} not in HF cache")
def test_qwen3_real_forward(glm, device):
    cfg = Qwen3Config(load_qwen3_config())
    weights = load_qwen3_weights(device)

    input_ids = torch.tensor([[151643, 151644, 151645, 1, 2, 3, 4, 5]], dtype=torch.int64, device=device)

    ref_logits = qwen3_model_torch(input_ids, weights, cfg)
    cuda_logits = qwen3_model_cuda(glm, device, input_ids, weights, cfg)
    torch.testing.assert_close(cuda_logits.cpu(), ref_logits.cpu(), atol=1.0, rtol=1e-1)
