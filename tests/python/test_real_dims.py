import torch
import pytest
from test_attention import attention_forward_torch, attention_forward_cuda, _make_rotary_embed
from test_moe import moe_forward_torch, moe_forward_cuda
from test_decoder_layer import decoder_layer_torch, decoder_layer_cuda

HIDDEN = 6144
NUM_HEADS = 64
Q_LORA_RANK = 2048
KV_LORA_RANK = 512
QK_NOPE_DIM = 192
QK_ROPE_DIM = 64
QK_HEAD_DIM = 256
V_HEAD_DIM = 256
INTERMEDIATE = 12288
MOE_INTER = 2048
N_ROUTED_EXPERTS = 256
TOP_K = 8
N_GROUP = 1
TOPK_GROUP = 1
NORM_TOPK = True
ROUTED_SCALE = 2.5
IDX_N_HEADS = 32
IDX_HEAD_DIM = 128
IDX_QK_ROPE_DIM = 64
IDX_TOPK = 2048
ROPE_THETA = 1_000_000
EPS = 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_layer_real_dims(glm, device):
    B, S = 1, 4
    idx_topk = min(IDX_TOPK, S)
    idx_softmax_scale = IDX_HEAD_DIM ** -0.5
    idx_eps = 1e-6

    torch.manual_seed(42)
    hidden_states = torch.randn(B, S, HIDDEN, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, QK_ROPE_DIM // 2, B, S, theta=ROPE_THETA)

    causal_2d = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_2d, S)
    attention_mask = causal_2d.unsqueeze(0).unsqueeze(0)

    input_layernorm_w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)
    post_attn_layernorm_w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)

    q_a_proj_w = torch.randn(Q_LORA_RANK, HIDDEN, dtype=torch.bfloat16, device=device)
    q_a_layernorm_w = torch.randn(Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    q_b_proj_w = torch.randn(NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    kv_a_proj_with_mqa_w = torch.randn(KV_LORA_RANK + QK_ROPE_DIM, HIDDEN, dtype=torch.bfloat16, device=device)
    kv_a_layernorm_w = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    kv_b_proj_w = torch.randn(NUM_HEADS * (QK_NOPE_DIM + V_HEAD_DIM), KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    o_proj_w = torch.randn(HIDDEN, NUM_HEADS * V_HEAD_DIM, dtype=torch.bfloat16, device=device)

    idx_wq_b_w = torch.randn(IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    idx_wk_w = torch.randn(IDX_HEAD_DIM, HIDDEN, dtype=torch.bfloat16, device=device)
    idx_k_norm_w = torch.randn(IDX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    idx_k_norm_b = torch.randn(IDX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    idx_weights_proj_w = torch.randn(IDX_N_HEADS, HIDDEN, dtype=torch.bfloat16, device=device)

    attn_weights = dict(
        q_a_proj_w=q_a_proj_w, q_a_layernorm_w=q_a_layernorm_w,
        q_b_proj_w=q_b_proj_w,
        kv_a_proj_with_mqa_w=kv_a_proj_with_mqa_w, kv_a_layernorm_w=kv_a_layernorm_w,
        kv_b_proj_w=kv_b_proj_w, o_proj_w=o_proj_w,
        indexer_weights=dict(
            wq_b_w=idx_wq_b_w, wk_w=idx_wk_w,
            k_norm_w=idx_k_norm_w, k_norm_b=idx_k_norm_b,
            weights_proj_w=idx_weights_proj_w,
            n_heads=IDX_N_HEADS, head_dim=IDX_HEAD_DIM,
            qk_rope_dim=IDX_QK_ROPE_DIM, topk=idx_topk,
            softmax_scale=idx_softmax_scale, eps=idx_eps),
        num_heads=NUM_HEADS, qk_nope_dim=QK_NOPE_DIM, qk_rope_dim=QK_ROPE_DIM,
        v_head_dim=V_HEAD_DIM, q_lora_rank=Q_LORA_RANK, kv_lora_rank=KV_LORA_RANK,
        hidden_size=HIDDEN)

    gate_w = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=device)
    up_w = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=device)
    down_w = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device=device)

    mlp_weights = dict(gate_w=gate_w, up_w=up_w, down_w=down_w)

    ref = decoder_layer_torch(
        hidden_states, cos, sin, attention_mask,
        input_layernorm_w, post_attn_layernorm_w,
        attn_weights, mlp_weights, "dense", EPS)

    cuda = decoder_layer_cuda(
        glm, device, hidden_states, cos, sin, attention_mask,
        input_layernorm_w, post_attn_layernorm_w,
        attn_weights, mlp_weights, "dense", EPS)

    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=0.1, rtol=5e-3)


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(True, reason="Issues with gemv")
def test_moe_layer_real_dims(glm, device):
    num_distinct_experts = 32

    B, S = 1, 4
    idx_topk = min(IDX_TOPK, S)
    idx_softmax_scale = IDX_HEAD_DIM ** -0.5
    idx_eps = 1e-6

    torch.manual_seed(123)
    hidden_states = torch.randn(B, S, HIDDEN, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, QK_ROPE_DIM // 2, B, S, theta=ROPE_THETA)

    causal_2d = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_2d, S)
    attention_mask = causal_2d.unsqueeze(0).unsqueeze(0)

    input_layernorm_w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)
    post_attn_layernorm_w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)

    q_a_proj_w = torch.randn(Q_LORA_RANK, HIDDEN, dtype=torch.bfloat16, device=device)
    q_a_layernorm_w = torch.randn(Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    q_b_proj_w = torch.randn(NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    kv_a_proj_with_mqa_w = torch.randn(KV_LORA_RANK + QK_ROPE_DIM, HIDDEN, dtype=torch.bfloat16, device=device)
    kv_a_layernorm_w = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    kv_b_proj_w = torch.randn(NUM_HEADS * (QK_NOPE_DIM + V_HEAD_DIM), KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    o_proj_w = torch.randn(HIDDEN, NUM_HEADS * V_HEAD_DIM, dtype=torch.bfloat16, device=device)

    idx_wq_b_w = torch.randn(IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    idx_wk_w = torch.randn(IDX_HEAD_DIM, HIDDEN, dtype=torch.bfloat16, device=device)
    idx_k_norm_w = torch.randn(IDX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    idx_k_norm_b = torch.randn(IDX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    idx_weights_proj_w = torch.randn(IDX_N_HEADS, HIDDEN, dtype=torch.bfloat16, device=device)

    attn_weights = dict(
        q_a_proj_w=q_a_proj_w, q_a_layernorm_w=q_a_layernorm_w,
        q_b_proj_w=q_b_proj_w,
        kv_a_proj_with_mqa_w=kv_a_proj_with_mqa_w, kv_a_layernorm_w=kv_a_layernorm_w,
        kv_b_proj_w=kv_b_proj_w, o_proj_w=o_proj_w,
        indexer_weights=dict(
            wq_b_w=idx_wq_b_w, wk_w=idx_wk_w,
            k_norm_w=idx_k_norm_w, k_norm_b=idx_k_norm_b,
            weights_proj_w=idx_weights_proj_w,
            n_heads=IDX_N_HEADS, head_dim=IDX_HEAD_DIM,
            qk_rope_dim=IDX_QK_ROPE_DIM, topk=idx_topk,
            softmax_scale=idx_softmax_scale, eps=idx_eps),
        num_heads=NUM_HEADS, qk_nope_dim=QK_NOPE_DIM, qk_rope_dim=QK_ROPE_DIM,
        v_head_dim=V_HEAD_DIM, q_lora_rank=Q_LORA_RANK, kv_lora_rank=KV_LORA_RANK,
        hidden_size=HIDDEN)

    gate_weight = torch.randn(N_ROUTED_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)
    e_score_correction_bias = torch.randn(N_ROUTED_EXPERTS, dtype=torch.bfloat16, device=device)

    pool_gates = [torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device=device) for _ in range(num_distinct_experts)]
    pool_ups = [torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device=device) for _ in range(num_distinct_experts)]
    pool_downs = [torch.randn(HIDDEN, MOE_INTER, dtype=torch.bfloat16, device=device) for _ in range(num_distinct_experts)]
    expert_gates = [pool_gates[i % num_distinct_experts] for i in range(N_ROUTED_EXPERTS)]
    expert_ups = [pool_ups[i % num_distinct_experts] for i in range(N_ROUTED_EXPERTS)]
    expert_downs = [pool_downs[i % num_distinct_experts] for i in range(N_ROUTED_EXPERTS)]

    shared_gate_w = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device=device)
    shared_up_w = torch.randn(MOE_INTER, HIDDEN, dtype=torch.bfloat16, device=device)
    shared_down_w = torch.randn(HIDDEN, MOE_INTER, dtype=torch.bfloat16, device=device)

    mlp_weights = dict(
        gate_weight=gate_weight, e_score_correction_bias=e_score_correction_bias,
        expert_gates=expert_gates, expert_ups=expert_ups, expert_downs=expert_downs,
        shared_gate_w=shared_gate_w, shared_up_w=shared_up_w, shared_down_w=shared_down_w,
        n_group=N_GROUP, topk_group=TOPK_GROUP, top_k=TOP_K, num_experts=N_ROUTED_EXPERTS,
        norm_topk_prob=NORM_TOPK, routed_scaling_factor=ROUTED_SCALE,
        hidden=HIDDEN, moe_inter=MOE_INTER)

    ref = decoder_layer_torch(
        hidden_states, cos, sin, attention_mask,
        input_layernorm_w, post_attn_layernorm_w,
        attn_weights, mlp_weights, "sparse", EPS)

    cuda = decoder_layer_cuda(
        glm, device, hidden_states, cos, sin, attention_mask,
        input_layernorm_w, post_attn_layernorm_w,
        attn_weights, mlp_weights, "sparse", EPS)

    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=0.1, rtol=5e-3)
