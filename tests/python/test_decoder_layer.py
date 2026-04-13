import torch
import pytest
from helpers import ATOL, RTOL
from test_attention import attention_forward_torch, attention_forward_cuda, _make_rotary_embed
from test_moe import moe_forward_torch, moe_forward_cuda


def decoder_layer_torch(hidden_states, cos, sin, attention_mask,
                        input_layernorm_w, post_attn_layernorm_w,
                        attn_weights, mlp_weights, mlp_type, eps=1e-5):
    residual = hidden_states
    h_f = residual.float()
    var = h_f.pow(2).mean(-1, keepdim=True)
    h = (h_f * torch.rsqrt(var + eps) * input_layernorm_w.float()).to(torch.bfloat16)

    attn_out, q_resid = attention_forward_torch(
        h, cos, sin, attention_mask, **attn_weights, eps=eps)

    hidden_states = residual + attn_out
    residual = hidden_states

    h_f = hidden_states.float()
    var = h_f.pow(2).mean(-1, keepdim=True)
    h = (h_f * torch.rsqrt(var + eps) * post_attn_layernorm_w.float()).to(torch.bfloat16)

    if mlp_type == "dense":
        mlp_out = _dense_mlp_torch(h, **mlp_weights)
    else:
        mlp_out = moe_forward_torch(h, **mlp_weights)

    return residual + mlp_out


def _dense_mlp_torch(h, gate_w, up_w, down_w, **kwargs):
    gate = torch.nn.functional.linear(h, gate_w)
    up = torch.nn.functional.linear(h, up_w)
    gate_up = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
    return torch.nn.functional.linear(gate_up, down_w)


def _dense_mlp_cuda(glm, device, h, gate_w, up_w, down_w, **kwargs):
    B, S, hidden = h.shape
    intermediate = gate_w.shape[0]

    gate = torch.empty(B * S, intermediate, dtype=torch.bfloat16, device=device)
    glm.linear(gate, h.reshape(B * S, hidden), gate_w, B * S, intermediate, hidden)
    up = torch.empty(B * S, intermediate, dtype=torch.bfloat16, device=device)
    glm.linear(up, h.reshape(B * S, hidden), up_w, B * S, intermediate, hidden)

    gate_up = torch.empty_like(gate)
    glm.silu_and_mul(gate_up.reshape(-1), gate.reshape(-1), up.reshape(-1),
                     intermediate, B * S)

    out = torch.empty(B * S, hidden, dtype=torch.bfloat16, device=device)
    glm.linear(out, gate_up, down_w, B * S, hidden, intermediate)
    return out.reshape(B, S, hidden)


def decoder_layer_cuda(glm, device, hidden_states, cos, sin, attention_mask,
                       input_layernorm_w, post_attn_layernorm_w,
                       attn_weights, mlp_weights, mlp_type, eps=1e-5):
    B, S, hidden = hidden_states.shape

    residual = hidden_states
    h = torch.empty_like(hidden_states)
    glm.rmsnorm(h.reshape(-1), hidden_states.reshape(-1), input_layernorm_w,
                eps, hidden, B * S)

    attn_out, q_resid = attention_forward_cuda(
        glm, device, h, cos, sin, attention_mask, **attn_weights, eps=eps)

    hidden_states = torch.empty_like(residual)
    glm.add(hidden_states.reshape(-1), residual.reshape(-1), attn_out.reshape(-1),
            B * S * hidden)

    residual = hidden_states
    h = torch.empty_like(hidden_states)
    glm.rmsnorm(h.reshape(-1), hidden_states.reshape(-1), post_attn_layernorm_w,
                eps, hidden, B * S)

    if mlp_type == "dense":
        mlp_out = _dense_mlp_cuda(glm, device, h, **mlp_weights, eps=eps)
    else:
        mlp_out = moe_forward_cuda(glm, device, h, **mlp_weights)

    out = torch.empty_like(residual)
    glm.add(out.reshape(-1), residual.reshape(-1), mlp_out.reshape(-1),
            B * S * hidden)
    return out


def test_decoder_layer_dense(glm, device):
    B, S = 1, 4
    hidden = 64
    num_heads = 2
    qk_nope_dim = 12
    qk_rope_dim = 4
    qk_head_dim = qk_nope_dim + qk_rope_dim
    v_head_dim = 16
    q_lora_rank = 16
    kv_lora_rank = 16
    eps = 1e-5
    intermediate = 128

    idx_n_heads = 2
    idx_head_dim = 8
    idx_qk_rope_dim = 4
    idx_topk = 2
    idx_softmax_scale = idx_head_dim ** -0.5
    idx_eps = 1e-6

    torch.manual_seed(42)
    hidden_states = torch.randn(B, S, hidden, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

    causal_2d = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_2d, S)
    attention_mask = causal_2d.unsqueeze(0).unsqueeze(0)

    input_layernorm_w = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    post_attn_layernorm_w = torch.randn(hidden, dtype=torch.bfloat16, device=device)

    q_a_proj_w = torch.randn(q_lora_rank, hidden, dtype=torch.bfloat16, device=device)
    q_a_layernorm_w = torch.randn(q_lora_rank, dtype=torch.bfloat16, device=device)
    q_b_proj_w = torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    kv_a_proj_with_mqa_w = torch.randn(kv_lora_rank + qk_rope_dim, hidden, dtype=torch.bfloat16, device=device)
    kv_a_layernorm_w = torch.randn(kv_lora_rank, dtype=torch.bfloat16, device=device)
    kv_b_proj_w = torch.randn(num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank, dtype=torch.bfloat16, device=device)
    o_proj_w = torch.randn(hidden, num_heads * v_head_dim, dtype=torch.bfloat16, device=device)

    idx_wq_b_w = torch.randn(idx_n_heads * idx_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    idx_wk_w = torch.randn(idx_head_dim, hidden, dtype=torch.bfloat16, device=device)
    idx_k_norm_w = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_k_norm_b = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_weights_proj_w = torch.randn(idx_n_heads, hidden, dtype=torch.bfloat16, device=device)

    attn_weights = dict(
        q_a_proj_w=q_a_proj_w, q_a_layernorm_w=q_a_layernorm_w,
        q_b_proj_w=q_b_proj_w,
        kv_a_proj_with_mqa_w=kv_a_proj_with_mqa_w, kv_a_layernorm_w=kv_a_layernorm_w,
        kv_b_proj_w=kv_b_proj_w, o_proj_w=o_proj_w,
        indexer_weights=dict(
            wq_b_w=idx_wq_b_w, wk_w=idx_wk_w,
            k_norm_w=idx_k_norm_w, k_norm_b=idx_k_norm_b,
            weights_proj_w=idx_weights_proj_w,
            n_heads=idx_n_heads, head_dim=idx_head_dim,
            qk_rope_dim=idx_qk_rope_dim, topk=idx_topk,
            softmax_scale=idx_softmax_scale, eps=idx_eps),
        num_heads=num_heads, qk_nope_dim=qk_nope_dim, qk_rope_dim=qk_rope_dim,
        v_head_dim=v_head_dim, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
        hidden_size=hidden)

    gate_w = torch.randn(intermediate, hidden, dtype=torch.bfloat16, device=device)
    up_w = torch.randn(intermediate, hidden, dtype=torch.bfloat16, device=device)
    down_w = torch.randn(hidden, intermediate, dtype=torch.bfloat16, device=device)

    mlp_weights = dict(gate_w=gate_w, up_w=up_w, down_w=down_w)

    ref = decoder_layer_torch(
        hidden_states, cos, sin, attention_mask,
        input_layernorm_w, post_attn_layernorm_w,
        attn_weights, mlp_weights, "dense", eps)

    cuda = decoder_layer_cuda(
        glm, device, hidden_states, cos, sin, attention_mask,
        input_layernorm_w, post_attn_layernorm_w,
        attn_weights, mlp_weights, "dense", eps)

    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=0.3, rtol=5e-2)


def test_decoder_layer_no_mask_dense(glm, device):
    B, S = 2, 4
    hidden = 64
    num_heads = 2
    qk_nope_dim = 12
    qk_rope_dim = 4
    qk_head_dim = qk_nope_dim + qk_rope_dim
    v_head_dim = 16
    q_lora_rank = 16
    kv_lora_rank = 16
    eps = 1e-5
    intermediate = 128

    idx_n_heads = 2
    idx_head_dim = 8
    idx_qk_rope_dim = 4
    idx_topk = 4
    idx_softmax_scale = idx_head_dim ** -0.5
    idx_eps = 1e-6

    torch.manual_seed(7)
    hidden_states = torch.randn(B, S, hidden, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

    input_layernorm_w = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    post_attn_layernorm_w = torch.randn(hidden, dtype=torch.bfloat16, device=device)

    q_a_proj_w = torch.randn(q_lora_rank, hidden, dtype=torch.bfloat16, device=device)
    q_a_layernorm_w = torch.randn(q_lora_rank, dtype=torch.bfloat16, device=device)
    q_b_proj_w = torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    kv_a_proj_with_mqa_w = torch.randn(kv_lora_rank + qk_rope_dim, hidden, dtype=torch.bfloat16, device=device)
    kv_a_layernorm_w = torch.randn(kv_lora_rank, dtype=torch.bfloat16, device=device)
    kv_b_proj_w = torch.randn(num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank, dtype=torch.bfloat16, device=device)
    o_proj_w = torch.randn(hidden, num_heads * v_head_dim, dtype=torch.bfloat16, device=device)

    idx_wq_b_w = torch.randn(idx_n_heads * idx_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    idx_wk_w = torch.randn(idx_head_dim, hidden, dtype=torch.bfloat16, device=device)
    idx_k_norm_w = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_k_norm_b = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_weights_proj_w = torch.randn(idx_n_heads, hidden, dtype=torch.bfloat16, device=device)

    attn_weights = dict(
        q_a_proj_w=q_a_proj_w, q_a_layernorm_w=q_a_layernorm_w,
        q_b_proj_w=q_b_proj_w,
        kv_a_proj_with_mqa_w=kv_a_proj_with_mqa_w, kv_a_layernorm_w=kv_a_layernorm_w,
        kv_b_proj_w=kv_b_proj_w, o_proj_w=o_proj_w,
        indexer_weights=dict(
            wq_b_w=idx_wq_b_w, wk_w=idx_wk_w,
            k_norm_w=idx_k_norm_w, k_norm_b=idx_k_norm_b,
            weights_proj_w=idx_weights_proj_w,
            n_heads=idx_n_heads, head_dim=idx_head_dim,
            qk_rope_dim=idx_qk_rope_dim, topk=idx_topk,
            softmax_scale=idx_softmax_scale, eps=idx_eps),
        num_heads=num_heads, qk_nope_dim=qk_nope_dim, qk_rope_dim=qk_rope_dim,
        v_head_dim=v_head_dim, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
        hidden_size=hidden)

    gate_w = torch.randn(intermediate, hidden, dtype=torch.bfloat16, device=device)
    up_w = torch.randn(intermediate, hidden, dtype=torch.bfloat16, device=device)
    down_w = torch.randn(hidden, intermediate, dtype=torch.bfloat16, device=device)

    mlp_weights = dict(gate_w=gate_w, up_w=up_w, down_w=down_w)

    ref = decoder_layer_torch(
        hidden_states, cos, sin, None,
        input_layernorm_w, post_attn_layernorm_w,
        attn_weights, mlp_weights, "dense", eps)

    cuda = decoder_layer_cuda(
        glm, device, hidden_states, cos, sin, None,
        input_layernorm_w, post_attn_layernorm_w,
        attn_weights, mlp_weights, "dense", eps)

    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=0.3, rtol=5e-2)


def test_decoder_layer_moe(glm, device):
    B, S = 1, 4
    hidden = 32
    num_heads = 2
    qk_nope_dim = 8
    qk_rope_dim = 4
    qk_head_dim = qk_nope_dim + qk_rope_dim
    v_head_dim = 8
    q_lora_rank = 16
    kv_lora_rank = 16
    eps = 1e-5

    num_experts = 4
    top_k = 2
    n_group = 2
    topk_group = 1
    moe_inter = 32
    norm_topk_prob = True
    routed_scaling_factor = 1.0

    idx_n_heads = 2
    idx_head_dim = 8
    idx_qk_rope_dim = 4
    idx_topk = 2
    idx_softmax_scale = idx_head_dim ** -0.5
    idx_eps = 1e-6

    torch.manual_seed(123)
    hidden_states = torch.randn(B, S, hidden, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

    causal_2d = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_2d, S)
    attention_mask = causal_2d.unsqueeze(0).unsqueeze(0)

    input_layernorm_w = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    post_attn_layernorm_w = torch.randn(hidden, dtype=torch.bfloat16, device=device)

    q_a_proj_w = torch.randn(q_lora_rank, hidden, dtype=torch.bfloat16, device=device)
    q_a_layernorm_w = torch.randn(q_lora_rank, dtype=torch.bfloat16, device=device)
    q_b_proj_w = torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    kv_a_proj_with_mqa_w = torch.randn(kv_lora_rank + qk_rope_dim, hidden, dtype=torch.bfloat16, device=device)
    kv_a_layernorm_w = torch.randn(kv_lora_rank, dtype=torch.bfloat16, device=device)
    kv_b_proj_w = torch.randn(num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank, dtype=torch.bfloat16, device=device)
    o_proj_w = torch.randn(hidden, num_heads * v_head_dim, dtype=torch.bfloat16, device=device)

    idx_wq_b_w = torch.randn(idx_n_heads * idx_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    idx_wk_w = torch.randn(idx_head_dim, hidden, dtype=torch.bfloat16, device=device)
    idx_k_norm_w = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_k_norm_b = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_weights_proj_w = torch.randn(idx_n_heads, hidden, dtype=torch.bfloat16, device=device)

    attn_weights = dict(
        q_a_proj_w=q_a_proj_w, q_a_layernorm_w=q_a_layernorm_w,
        q_b_proj_w=q_b_proj_w,
        kv_a_proj_with_mqa_w=kv_a_proj_with_mqa_w, kv_a_layernorm_w=kv_a_layernorm_w,
        kv_b_proj_w=kv_b_proj_w, o_proj_w=o_proj_w,
        indexer_weights=dict(
            wq_b_w=idx_wq_b_w, wk_w=idx_wk_w,
            k_norm_w=idx_k_norm_w, k_norm_b=idx_k_norm_b,
            weights_proj_w=idx_weights_proj_w,
            n_heads=idx_n_heads, head_dim=idx_head_dim,
            qk_rope_dim=idx_qk_rope_dim, topk=idx_topk,
            softmax_scale=idx_softmax_scale, eps=idx_eps),
        num_heads=num_heads, qk_nope_dim=qk_nope_dim, qk_rope_dim=qk_rope_dim,
        v_head_dim=v_head_dim, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
        hidden_size=hidden)

    gate_weight = torch.randn(num_experts, hidden, dtype=torch.bfloat16, device=device)
    e_score_correction_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)
    expert_gates = [torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    expert_ups = [torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    expert_downs = [torch.randn(hidden, moe_inter, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    shared_gate_w = torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device)
    shared_up_w = torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device)
    shared_down_w = torch.randn(hidden, moe_inter, dtype=torch.bfloat16, device=device)

    mlp_weights = dict(
        gate_weight=gate_weight, e_score_correction_bias=e_score_correction_bias,
        expert_gates=expert_gates, expert_ups=expert_ups, expert_downs=expert_downs,
        shared_gate_w=shared_gate_w, shared_up_w=shared_up_w, shared_down_w=shared_down_w,
        n_group=n_group, topk_group=topk_group, top_k=top_k, num_experts=num_experts,
        norm_topk_prob=norm_topk_prob, routed_scaling_factor=routed_scaling_factor,
        hidden=hidden, moe_inter=moe_inter)

    ref = decoder_layer_torch(
        hidden_states, cos, sin, attention_mask,
        input_layernorm_w, post_attn_layernorm_w,
        attn_weights, mlp_weights, "sparse", eps)

    cuda = decoder_layer_cuda(
        glm, device, hidden_states, cos, sin, attention_mask,
        input_layernorm_w, post_attn_layernorm_w,
        attn_weights, mlp_weights, "sparse", eps)

    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=0.5, rtol=1e-1)
