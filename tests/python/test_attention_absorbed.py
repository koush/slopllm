import torch
import pytest
from helpers import ATOL, RTOL
from test_attention import attention_forward_torch, _make_rotary_embed
from test_indexer import indexer_forward_torch, indexer_forward_cuda, apply_rotary_pos_emb_torch


def attention_forward_absorbed_torch(hidden_states, cos, sin, attention_mask,
                                     q_a_proj_w, q_a_layernorm_w,
                                     q_b_proj_w,
                                     kv_a_proj_with_mqa_w, kv_a_layernorm_w,
                                     kv_b_proj_w,
                                     o_proj_w,
                                     indexer_weights,
                                     num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
                                     q_lora_rank, kv_lora_rank, hidden_size, eps=1e-5):
    B, S, _ = hidden_states.shape
    qk_head_dim = qk_nope_dim + qk_rope_dim
    scaling = qk_head_dim ** -0.5
    stride = qk_nope_dim + v_head_dim

    q_resid_raw = torch.nn.functional.linear(hidden_states, q_a_proj_w)
    q_resid_f = q_resid_raw.float()
    q_resid_var = q_resid_f.pow(2).mean(-1, keepdim=True)
    q_resid = (q_resid_f * torch.rsqrt(q_resid_var + eps) * q_a_layernorm_w.float()).to(torch.bfloat16)

    query = torch.nn.functional.linear(q_resid, q_b_proj_w)
    query = query.view(B, S, num_heads, qk_head_dim).transpose(1, 2).contiguous()
    q_nope = query[:, :, :, :qk_nope_dim].contiguous()
    q_pe = query[:, :, :, qk_nope_dim:].contiguous()
    q_pe = apply_rotary_pos_emb_torch(q_pe, cos, sin, unsqueeze_dim=1)

    compressed = torch.nn.functional.linear(hidden_states, kv_a_proj_with_mqa_w)
    k_compressed, k_pe = compressed.split([kv_lora_rank, qk_rope_dim], dim=-1)
    k_compressed_f = k_compressed.float()
    k_compressed_var = k_compressed_f.pow(2).mean(-1, keepdim=True)
    k_compressed_norm = (k_compressed_f * torch.rsqrt(k_compressed_var + eps) * kv_a_layernorm_w.float()).to(torch.bfloat16)

    W_k_nope = kv_b_proj_w.view(num_heads, stride, kv_lora_rank)[:, :qk_nope_dim, :].contiguous()
    q_absorbed = torch.einsum('bhsd,hdk->bhsk', q_nope, W_k_nope)

    k_pe = k_pe.view(B, 1, S, qk_rope_dim)
    k_pe = apply_rotary_pos_emb_torch(k_pe, cos, sin, unsqueeze_dim=1)
    k_pe_expanded = k_pe.expand(B, num_heads, S, qk_rope_dim)

    kv_c = k_compressed_norm.unsqueeze(1).expand(B, num_heads, S, kv_lora_rank)

    idx_attn_mask = attention_mask[:, 0, :, :] if attention_mask is not None and attention_mask.dim() == 4 else attention_mask
    _, topk_indices = indexer_forward_torch(
        hidden_states, q_resid, cos, sin, idx_attn_mask, **indexer_weights)

    total_len = S
    index_mask = torch.full((B, S, total_len), float('-inf'), device=hidden_states.device, dtype=torch.bfloat16)
    index_mask.scatter_(-1, topk_indices, 0.0)
    combined_mask = index_mask.unsqueeze(1)
    if attention_mask is not None:
        combined_mask = combined_mask + attention_mask[..., :total_len]
    combined_mask = combined_mask.expand(B, num_heads, S, total_len)

    attn_scores = torch.bmm(q_absorbed.reshape(B * num_heads, S, kv_lora_rank),
                              kv_c.reshape(B * num_heads, S, kv_lora_rank).transpose(1, 2)) * scaling
    attn_scores = attn_scores + torch.bmm(q_pe.reshape(B * num_heads, S, qk_rope_dim),
                                           k_pe_expanded.reshape(B * num_heads, S, qk_rope_dim).transpose(1, 2)) * scaling
    attn_scores = attn_scores + combined_mask.reshape(B * num_heads, S, total_len)
    attn_scores = torch.nn.functional.softmax(attn_scores.float(), dim=-1).to(torch.bfloat16)

    c = torch.bmm(attn_scores, kv_c.reshape(B * num_heads, S, kv_lora_rank))

    W_v = kv_b_proj_w.view(num_heads, stride, kv_lora_rank)[:, qk_nope_dim:, :].contiguous()
    output_per_head = torch.einsum('bhsk,hdk->bhsd', c.view(B, num_heads, S, kv_lora_rank), W_v)

    out = output_per_head.transpose(1, 2).reshape(B, S, num_heads * v_head_dim)
    output = torch.nn.functional.linear(out, o_proj_w)

    return output, q_resid


def attention_forward_absorbed_cuda(glm, device, hidden_states, cos, sin, attention_mask,
                                     q_a_proj_w, q_a_layernorm_w,
                                     q_b_proj_w,
                                     kv_a_proj_with_mqa_w, kv_a_layernorm_w,
                                     kv_b_proj_w,
                                     o_proj_w,
                                     indexer_weights,
                                     num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
                                     q_lora_rank, kv_lora_rank, hidden_size, eps=1e-5):
    B, S, _ = hidden_states.shape
    qk_head_dim = qk_nope_dim + qk_rope_dim
    scaling = qk_head_dim ** -0.5
    stride = qk_nope_dim + v_head_dim

    q_a_out = torch.empty(B * S, q_lora_rank, dtype=torch.bfloat16, device=device)
    glm.linear(q_a_out, hidden_states.reshape(B * S, hidden_size), q_a_proj_w, B * S, q_lora_rank, hidden_size)
    q_resid = torch.empty_like(q_a_out)
    glm.rmsnorm(q_resid, q_a_out, q_a_layernorm_w, eps, q_lora_rank, B * S)

    q_b_out = torch.empty(B * S, num_heads * qk_head_dim, dtype=torch.bfloat16, device=device)
    glm.linear(q_b_out, q_resid, q_b_proj_w, B * S, num_heads * qk_head_dim, q_lora_rank)
    query = q_b_out.reshape(B, S, num_heads, qk_head_dim).transpose(1, 2).contiguous()
    query = query.reshape(B * num_heads, S, qk_head_dim)

    q_nope_buf = query[:, :, :qk_nope_dim].contiguous()
    q_pe_buf = query[:, :, qk_nope_dim:].contiguous()
    q_pe_rope = torch.empty_like(q_pe_buf)
    glm.apply_rotary_pos_emb(q_pe_rope, q_pe_buf, cos, sin, qk_rope_dim, num_heads, S, B, 1)

    kv_a_out = torch.empty(B * S, kv_lora_rank + qk_rope_dim, dtype=torch.bfloat16, device=device)
    glm.linear(kv_a_out, hidden_states.reshape(B * S, hidden_size), kv_a_proj_with_mqa_w,
               B * S, kv_lora_rank + qk_rope_dim, hidden_size)
    k_compressed = kv_a_out[:, :kv_lora_rank].reshape(B, S, kv_lora_rank).contiguous()
    k_pe_raw = kv_a_out[:, kv_lora_rank:].reshape(B, S, qk_rope_dim).contiguous()

    k_compressed_norm = torch.empty_like(k_compressed)
    glm.rmsnorm(k_compressed_norm.reshape(B * S, kv_lora_rank),
                k_compressed.reshape(B * S, kv_lora_rank),
                kv_a_layernorm_w, eps, kv_lora_rank, B * S)

    q_absorbed = torch.empty(B * num_heads, S, kv_lora_rank, dtype=torch.bfloat16, device=device)
    W_k_nope = kv_b_proj_w.reshape(num_heads, stride, kv_lora_rank)[:, :qk_nope_dim, :].contiguous()
    q_nope_4d = q_nope_buf.reshape(B, num_heads, S, qk_nope_dim)
    q_nope_hb = torch.empty(num_heads, B, S, qk_nope_dim, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(q_nope_hb.reshape(-1), q_nope_4d.reshape(-1),
                     B, num_heads, S, qk_nope_dim, 1, 0, 2, 3)
    q_absorbed_hb = torch.empty(num_heads, B * S, kv_lora_rank, dtype=torch.bfloat16, device=device)
    glm.bmm(q_absorbed_hb, q_nope_hb.reshape(num_heads, B * S, qk_nope_dim),
             W_k_nope, 1.0, 0.0, num_heads, B * S, kv_lora_rank, qk_nope_dim, 0, 0)
    q_absorbed_4d = q_absorbed_hb.reshape(num_heads, B, S, kv_lora_rank)
    glm.transpose_4d(q_absorbed.reshape(-1), q_absorbed_4d.reshape(-1),
                     num_heads, B, S, kv_lora_rank, 1, 0, 2, 3)

    kv_c_4d = k_compressed_norm.reshape(B, 1, S, kv_lora_rank).contiguous()
    kv_c_expanded = torch.empty(B, num_heads, S, kv_lora_rank, dtype=torch.bfloat16, device=device)
    glm.expand_dim1(kv_c_expanded.reshape(-1), kv_c_4d.reshape(-1),
                     num_heads, 1, S, kv_lora_rank, B)
    kv_c_for_bmm = kv_c_expanded.reshape(B * num_heads, S, kv_lora_rank)

    k_pe_4d = k_pe_raw.reshape(B, 1, S, qk_rope_dim).contiguous()
    k_pe_rope = torch.empty_like(k_pe_4d)
    glm.apply_rotary_pos_emb(k_pe_rope, k_pe_4d, cos, sin, qk_rope_dim, 1, S, B, 1)
    k_pe_expanded = torch.empty(B, num_heads, S, qk_rope_dim, dtype=torch.bfloat16, device=device)
    glm.expand_dim1(k_pe_expanded.reshape(-1), k_pe_rope.reshape(-1),
                     num_heads, 1, S, qk_rope_dim, B)
    k_pe_for_bmm = k_pe_expanded.reshape(B * num_heads, S, qk_rope_dim)

    total_len = S

    idx_attn_mask = None
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            idx_attn_mask = attention_mask[:, 0, :, :].contiguous()
        else:
            idx_attn_mask = attention_mask.contiguous()
    _, topk_indices = indexer_forward_cuda(
        glm, device, hidden_states, q_resid.reshape(B, S, q_lora_rank),
        cos, sin, idx_attn_mask, **indexer_weights)

    topk_val = topk_indices.shape[-1]
    index_mask = torch.empty(B * S, total_len, dtype=torch.bfloat16, device=device)
    glm.fill(index_mask.reshape(-1), float('-inf'), B * S * total_len)
    glm.scatter_scalar(index_mask.reshape(-1), topk_indices.reshape(-1),
                       0.0, topk_val, total_len, B * S)
    index_mask = index_mask.reshape(B, 1, S, total_len)

    if attention_mask is not None:
        combined_mask = torch.empty_like(index_mask)
        glm.add(combined_mask.reshape(-1), index_mask.reshape(-1),
                attention_mask.reshape(-1), B * 1 * S * total_len)
    else:
        combined_mask = index_mask

    combined_mask_expanded = torch.empty(B, num_heads, S, total_len, dtype=torch.bfloat16, device=device)
    glm.expand_dim1(combined_mask_expanded.reshape(-1), combined_mask.reshape(-1),
                     num_heads, 1, S, total_len, B)

    attn_scores = torch.empty(B * num_heads, S, total_len, dtype=torch.bfloat16, device=device)
    glm.bmm(attn_scores, q_absorbed, kv_c_for_bmm, scaling, 0.0,
             B * num_heads, S, total_len, kv_lora_rank, 0, 1)
    glm.bmm(attn_scores, q_pe_rope, k_pe_for_bmm, scaling, 1.0,
             B * num_heads, S, total_len, qk_rope_dim, 0, 1)

    glm.add(attn_scores.reshape(-1), attn_scores.reshape(-1),
            combined_mask_expanded.reshape(-1), B * num_heads * S * total_len)

    softmax_mask_2d = combined_mask_expanded.reshape(B * num_heads, S, total_len)
    glm.softmax(attn_scores.reshape(-1, total_len), attn_scores.reshape(-1, total_len),
                softmax_mask_2d.reshape(-1, total_len), total_len, B * num_heads * S)

    c = torch.empty(B * num_heads, S, kv_lora_rank, dtype=torch.bfloat16, device=device)
    glm.bmm(c, attn_scores, kv_c_for_bmm, 1.0, 0.0,
             B * num_heads, S, kv_lora_rank, total_len, 0, 0)

    W_v = kv_b_proj_w.reshape(num_heads, stride, kv_lora_rank)[:, qk_nope_dim:, :].contiguous()
    c_4d = c.reshape(B, num_heads, S, kv_lora_rank)
    c_hb = torch.empty(num_heads, B, S, kv_lora_rank, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(c_hb.reshape(-1), c_4d.reshape(-1),
                     B, num_heads, S, kv_lora_rank, 1, 0, 2, 3)
    output_hb = torch.empty(num_heads, B * S, v_head_dim, dtype=torch.bfloat16, device=device)
    glm.bmm(output_hb, c_hb.reshape(num_heads, B * S, kv_lora_rank),
             W_v, 1.0, 0.0, num_heads, B * S, v_head_dim, kv_lora_rank, 0, 1)
    output_4d = output_hb.reshape(num_heads, B, S, v_head_dim)
    output_per_head = torch.empty(B, num_heads, S, v_head_dim, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(output_per_head.reshape(-1), output_4d.reshape(-1),
                     num_heads, B, S, v_head_dim, 1, 0, 2, 3)

    attn_out_t = torch.empty(B, S, num_heads, v_head_dim, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(attn_out_t.reshape(-1), output_per_head.reshape(-1),
                     B, num_heads, S, v_head_dim, 0, 2, 1, 3)
    attn_out_flat = attn_out_t.reshape(B * S, num_heads * v_head_dim)

    output = torch.empty(B * S, hidden_size, dtype=torch.bfloat16, device=device)
    glm.linear(output, attn_out_flat, o_proj_w, B * S, hidden_size, num_heads * v_head_dim)

    return output.reshape(B, S, hidden_size), q_resid.reshape(B, S, q_lora_rank)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_attention_absorbed_small(glm, device):
    B, S = 1, 4
    hidden_size = 64
    q_lora_rank = 16
    num_heads = 2
    qk_nope_dim = 12
    qk_rope_dim = 4
    qk_head_dim = qk_nope_dim + qk_rope_dim
    v_head_dim = 16
    kv_lora_rank = 16
    eps = 1e-5
    idx_n_heads = 2
    idx_head_dim = 8
    idx_qk_rope_dim = 4
    idx_topk = 2
    idx_softmax_scale = idx_head_dim ** -0.5
    idx_eps = 1e-6

    torch.manual_seed(42)
    hidden_states = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

    causal_2d = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_2d, S)
    attention_mask = causal_2d.unsqueeze(0).unsqueeze(0)

    q_a_proj_w = torch.randn(q_lora_rank, hidden_size, dtype=torch.bfloat16, device=device)
    q_a_layernorm_w = torch.randn(q_lora_rank, dtype=torch.bfloat16, device=device)
    q_b_proj_w = torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    kv_a_proj_with_mqa_w = torch.randn(kv_lora_rank + qk_rope_dim, hidden_size, dtype=torch.bfloat16, device=device)
    kv_a_layernorm_w = torch.randn(kv_lora_rank, dtype=torch.bfloat16, device=device)
    kv_b_proj_w = torch.randn(num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank, dtype=torch.bfloat16, device=device)
    o_proj_w = torch.randn(hidden_size, num_heads * v_head_dim, dtype=torch.bfloat16, device=device)

    idx_wq_b_w = torch.randn(idx_n_heads * idx_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    idx_wk_w = torch.randn(idx_head_dim, hidden_size, dtype=torch.bfloat16, device=device)
    idx_k_norm_w = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_k_norm_b = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_weights_proj_w = torch.randn(idx_n_heads, hidden_size, dtype=torch.bfloat16, device=device)

    indexer_weights = dict(
        wq_b_w=idx_wq_b_w, wk_w=idx_wk_w,
        k_norm_w=idx_k_norm_w, k_norm_b=idx_k_norm_b,
        weights_proj_w=idx_weights_proj_w,
        n_heads=idx_n_heads, head_dim=idx_head_dim,
        qk_rope_dim=idx_qk_rope_dim, topk=idx_topk,
        softmax_scale=idx_softmax_scale, eps=idx_eps)

    ref_output, ref_q_resid = attention_forward_absorbed_torch(
        hidden_states, cos, sin, attention_mask,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)

    cuda_output, cuda_q_resid = attention_forward_absorbed_cuda(
        glm, device, hidden_states, cos, sin, attention_mask,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)

    torch.testing.assert_close(cuda_output.cpu(), ref_output.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(cuda_q_resid.cpu(), ref_q_resid.cpu(), atol=5e-3, rtol=5e-3)

    exp_output, exp_q_resid = attention_forward_torch(
        hidden_states, cos, sin, attention_mask,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)
    torch.testing.assert_close(cuda_output.cpu(), exp_output.cpu(), atol=5.0, rtol=2e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_attention_absorbed_no_mask(glm, device):
    B, S = 2, 4
    hidden_size = 64
    q_lora_rank = 16
    num_heads = 2
    qk_nope_dim = 12
    qk_rope_dim = 4
    qk_head_dim = qk_nope_dim + qk_rope_dim
    v_head_dim = 16
    kv_lora_rank = 16
    eps = 1e-5
    idx_n_heads = 2
    idx_head_dim = 8
    idx_qk_rope_dim = 4
    idx_topk = 4
    idx_softmax_scale = idx_head_dim ** -0.5
    idx_eps = 1e-6

    torch.manual_seed(7)
    hidden_states = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

    q_a_proj_w = torch.randn(q_lora_rank, hidden_size, dtype=torch.bfloat16, device=device)
    q_a_layernorm_w = torch.randn(q_lora_rank, dtype=torch.bfloat16, device=device)
    q_b_proj_w = torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    kv_a_proj_with_mqa_w = torch.randn(kv_lora_rank + qk_rope_dim, hidden_size, dtype=torch.bfloat16, device=device)
    kv_a_layernorm_w = torch.randn(kv_lora_rank, dtype=torch.bfloat16, device=device)
    kv_b_proj_w = torch.randn(num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank, dtype=torch.bfloat16, device=device)
    o_proj_w = torch.randn(hidden_size, num_heads * v_head_dim, dtype=torch.bfloat16, device=device)

    idx_wq_b_w = torch.randn(idx_n_heads * idx_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    idx_wk_w = torch.randn(idx_head_dim, hidden_size, dtype=torch.bfloat16, device=device)
    idx_k_norm_w = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_k_norm_b = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_weights_proj_w = torch.randn(idx_n_heads, hidden_size, dtype=torch.bfloat16, device=device)

    indexer_weights = dict(
        wq_b_w=idx_wq_b_w, wk_w=idx_wk_w,
        k_norm_w=idx_k_norm_w, k_norm_b=idx_k_norm_b,
        weights_proj_w=idx_weights_proj_w,
        n_heads=idx_n_heads, head_dim=idx_head_dim,
        qk_rope_dim=idx_qk_rope_dim, topk=idx_topk,
        softmax_scale=idx_softmax_scale, eps=idx_eps)

    ref_output, ref_q_resid = attention_forward_absorbed_torch(
        hidden_states, cos, sin, None,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)

    cuda_output, cuda_q_resid = attention_forward_absorbed_cuda(
        glm, device, hidden_states, cos, sin, None,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)

    torch.testing.assert_close(cuda_output.cpu(), ref_output.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(cuda_q_resid.cpu(), ref_q_resid.cpu(), atol=5e-3, rtol=5e-3)

    exp_output, _ = attention_forward_torch(
        hidden_states, cos, sin, None,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)
    torch.testing.assert_close(cuda_output.cpu(), exp_output.cpu(), atol=5.0, rtol=2e-1)

    exp_output, exp_q_resid = attention_forward_torch(
        hidden_states, cos, sin, None,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)
    torch.testing.assert_close(cuda_output.cpu(), exp_output.cpu(), atol=0.5, rtol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_attention_absorbed_real_dims(glm, device):
    B, S = 1, 4
    hidden_size = 6144
    num_heads = 64
    qk_nope_dim = 192
    qk_rope_dim = 64
    qk_head_dim = qk_nope_dim + qk_rope_dim
    v_head_dim = 256
    q_lora_rank = 2048
    kv_lora_rank = 512
    eps = 1e-5
    idx_n_heads = 32
    idx_head_dim = 128
    idx_qk_rope_dim = 64
    idx_topk = min(2048, S)
    idx_softmax_scale = idx_head_dim ** -0.5
    idx_eps = 1e-6

    torch.manual_seed(42)
    hidden_states = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S, theta=1_000_000)

    causal_2d = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_2d, S)
    attention_mask = causal_2d.unsqueeze(0).unsqueeze(0)

    q_a_proj_w = torch.randn(q_lora_rank, hidden_size, dtype=torch.bfloat16, device=device)
    q_a_layernorm_w = torch.randn(q_lora_rank, dtype=torch.bfloat16, device=device)
    q_b_proj_w = torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    kv_a_proj_with_mqa_w = torch.randn(kv_lora_rank + qk_rope_dim, hidden_size, dtype=torch.bfloat16, device=device)
    kv_a_layernorm_w = torch.randn(kv_lora_rank, dtype=torch.bfloat16, device=device)
    kv_b_proj_w = torch.randn(num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank, dtype=torch.bfloat16, device=device)
    o_proj_w = torch.randn(hidden_size, num_heads * v_head_dim, dtype=torch.bfloat16, device=device)

    idx_wq_b_w = torch.randn(idx_n_heads * idx_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    idx_wk_w = torch.randn(idx_head_dim, hidden_size, dtype=torch.bfloat16, device=device)
    idx_k_norm_w = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_k_norm_b = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_weights_proj_w = torch.randn(idx_n_heads, hidden_size, dtype=torch.bfloat16, device=device)

    indexer_weights = dict(
        wq_b_w=idx_wq_b_w, wk_w=idx_wk_w,
        k_norm_w=idx_k_norm_w, k_norm_b=idx_k_norm_b,
        weights_proj_w=idx_weights_proj_w,
        n_heads=idx_n_heads, head_dim=idx_head_dim,
        qk_rope_dim=idx_qk_rope_dim, topk=idx_topk,
        softmax_scale=idx_softmax_scale, eps=idx_eps)

    ref_output, ref_q_resid = attention_forward_absorbed_torch(
        hidden_states, cos, sin, attention_mask,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)

    cuda_output, cuda_q_resid = attention_forward_absorbed_cuda(
        glm, device, hidden_states, cos, sin, attention_mask,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)

    torch.testing.assert_close(cuda_output.cpu(), ref_output.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(cuda_q_resid.cpu(), ref_q_resid.cpu(), atol=5e-3, rtol=5e-3)
