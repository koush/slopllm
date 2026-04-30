import torch
import pytest
from helpers import ATOL, RTOL
from test_indexer import (
    apply_rotary_pos_emb_torch,
    indexer_forward_torch,
    indexer_forward_cuda,
    _make_rotary_embed,
)


def attention_forward_torch(hidden_states, cos, sin, attention_mask,
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

    q_resid_raw = torch.nn.functional.linear(hidden_states, q_a_proj_w)
    q_resid_f = q_resid_raw.float()
    q_resid_var = q_resid_f.pow(2).mean(-1, keepdim=True)
    q_resid = (q_resid_f * torch.rsqrt(q_resid_var + eps) * q_a_layernorm_w.float()).to(torch.bfloat16)

    query = torch.nn.functional.linear(q_resid, q_b_proj_w)
    query = query.view(B, S, num_heads, qk_head_dim).transpose(1, 2)
    q_nope, q_pe = query.split([qk_nope_dim, qk_rope_dim], dim=-1)
    q_pe = apply_rotary_pos_emb_torch(q_pe, cos, sin, unsqueeze_dim=1)
    query = torch.cat([q_nope, q_pe], dim=-1)

    compressed = torch.nn.functional.linear(hidden_states, kv_a_proj_with_mqa_w)
    k_compressed, k_pe = compressed.split([kv_lora_rank, qk_rope_dim], dim=-1)
    k_compressed_f = k_compressed.float()
    k_compressed_var = k_compressed_f.pow(2).mean(-1, keepdim=True)
    k_compressed = (k_compressed_f * torch.rsqrt(k_compressed_var + eps) * kv_a_layernorm_w.float()).to(torch.bfloat16)

    kv_expanded = torch.nn.functional.linear(k_compressed, kv_b_proj_w)
    kv_expanded = kv_expanded.view(B, S, num_heads, qk_nope_dim + v_head_dim)
    k_nope, value = kv_expanded.split([qk_nope_dim, v_head_dim], dim=-1)
    k_nope = k_nope.transpose(1, 2)
    value = value.transpose(1, 2)

    k_pe = k_pe.view(B, 1, S, qk_rope_dim)
    k_pe = apply_rotary_pos_emb_torch(k_pe, cos, sin, unsqueeze_dim=1)
    k_pe = k_pe.expand(B, num_heads, S, qk_rope_dim).contiguous()

    key = torch.cat([k_nope, k_pe], dim=-1)

    idx_attn_mask = attention_mask[:, 0, :, :] if attention_mask is not None and attention_mask.dim() == 4 else attention_mask
    _, topk_indices = indexer_forward_torch(
        hidden_states, q_resid, cos, sin, idx_attn_mask, **indexer_weights)

    total_len = key.shape[2]
    index_mask = torch.full((B, S, total_len), float('-inf'), device=hidden_states.device, dtype=torch.bfloat16)
    index_mask.scatter_(-1, topk_indices, 0.0)
    combined_mask = index_mask.unsqueeze(1)
    if attention_mask is not None:
        combined_mask = combined_mask + attention_mask[..., :total_len]

    attn_w = (query @ key.transpose(2, 3)) * scaling
    attn_w = attn_w + combined_mask
    attn_w = torch.nn.functional.softmax(attn_w.float(), dim=-1).to(torch.bfloat16)
    out = (attn_w @ value).transpose(1, 2).reshape(B, S, -1)
    output = torch.nn.functional.linear(out, o_proj_w)

    return output, q_resid


def attention_forward_cuda(glm, device, hidden_states, cos, sin, attention_mask,
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
    num_kv_groups = 1

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
    query_full = torch.empty(B * num_heads, S, qk_head_dim, dtype=torch.bfloat16, device=device)
    glm.cat_last_dim(query_full.reshape(-1, qk_head_dim),
                     q_nope_buf.reshape(-1, qk_nope_dim),
                     q_pe_rope.reshape(-1, qk_rope_dim),
                     qk_nope_dim, qk_rope_dim, B * num_heads * S)

    kv_a_out = torch.empty(B * S, kv_lora_rank + qk_rope_dim, dtype=torch.bfloat16, device=device)
    glm.linear(kv_a_out, hidden_states.reshape(B * S, hidden_size), kv_a_proj_with_mqa_w,
               B * S, kv_lora_rank + qk_rope_dim, hidden_size)
    k_compressed = kv_a_out[:, :kv_lora_rank].reshape(B, S, kv_lora_rank).contiguous()
    k_pe_raw = kv_a_out[:, kv_lora_rank:].reshape(B, S, qk_rope_dim).contiguous()

    k_compressed_norm = torch.empty_like(k_compressed)
    glm.rmsnorm(k_compressed_norm.reshape(B * S, kv_lora_rank),
                k_compressed.reshape(B * S, kv_lora_rank),
                kv_a_layernorm_w, eps, kv_lora_rank, B * S)

    kv_b_out = torch.empty(B * S, num_heads * (qk_nope_dim + v_head_dim), dtype=torch.bfloat16, device=device)
    glm.linear(kv_b_out, k_compressed_norm.reshape(B * S, kv_lora_rank), kv_b_proj_w,
               B * S, num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank)
    kv_expanded = kv_b_out.reshape(B, S, num_heads, qk_nope_dim + v_head_dim)

    k_nope_3d = kv_expanded[:, :, :, :qk_nope_dim].contiguous()
    value_3d = kv_expanded[:, :, :, qk_nope_dim:].contiguous()

    k_nope_t = torch.empty(B, num_heads, S, qk_nope_dim, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(k_nope_t.reshape(-1), k_nope_3d.reshape(-1),
                     B, S, num_heads, qk_nope_dim, 0, 2, 1, 3)
    value_t = torch.empty(B, num_heads, S, v_head_dim, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(value_t.reshape(-1), value_3d.reshape(-1),
                     B, S, num_heads, v_head_dim, 0, 2, 1, 3)

    k_pe_4d = k_pe_raw.reshape(B, 1, S, qk_rope_dim).contiguous()
    k_pe_rope = torch.empty_like(k_pe_4d)
    glm.apply_rotary_pos_emb(k_pe_rope, k_pe_4d, cos, sin, qk_rope_dim, 1, S, B, 1)
    k_pe_expanded = torch.empty(B, num_heads, S, qk_rope_dim, dtype=torch.bfloat16, device=device)
    glm.expand_dim1(k_pe_expanded.reshape(-1), k_pe_rope.reshape(-1),
                     num_heads, 1, S, qk_rope_dim, B)

    key_full = torch.empty(B * num_heads, S, qk_head_dim, dtype=torch.bfloat16, device=device)
    glm.cat_last_dim(key_full.reshape(-1, qk_head_dim),
                     k_nope_t.reshape(-1, qk_nope_dim),
                     k_pe_expanded.reshape(-1, qk_rope_dim),
                     qk_nope_dim, qk_rope_dim, B * num_heads * S)

    idx_attn_mask = None
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            idx_attn_mask = attention_mask[:, 0, :, :].contiguous()
        else:
            idx_attn_mask = attention_mask.contiguous()
    _, topk_indices = indexer_forward_cuda(
        glm, device, hidden_states, q_resid.reshape(B, S, q_lora_rank),
        cos, sin, idx_attn_mask, **indexer_weights)

    total_len = S
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

    attn_w = torch.empty(B * num_heads, S, total_len, dtype=torch.bfloat16, device=device)
    glm.bmm(attn_w, query_full, key_full, scaling, 0.0, B * num_heads, S, total_len, qk_head_dim, 0, 1)
    glm.add(attn_w.reshape(-1), attn_w.reshape(-1), combined_mask_expanded.reshape(-1), B * num_heads * S * total_len)

    softmax_mask_2d = combined_mask_expanded.reshape(B * num_heads, S, total_len)
    glm.softmax(attn_w.reshape(-1, total_len), attn_w.reshape(-1, total_len),
                softmax_mask_2d.reshape(-1, total_len), total_len, B * num_heads * S)

    attn_out = torch.empty(B * num_heads, S, v_head_dim, dtype=torch.bfloat16, device=device)
    glm.bmm(attn_out, attn_w, value_t.reshape(B * num_heads, S, v_head_dim),
             1.0, 0.0, B * num_heads, S, v_head_dim, total_len, 0, 0)

    attn_out_t = torch.empty(B, S, num_heads, v_head_dim, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(attn_out_t.reshape(-1), attn_out.reshape(-1),
                     B, num_heads, S, v_head_dim, 0, 2, 1, 3)
    attn_out_flat = attn_out_t.reshape(B * S, num_heads * v_head_dim)

    output = torch.empty(B * S, hidden_size, dtype=torch.bfloat16, device=device)
    glm.linear(output, attn_out_flat, o_proj_w, B * S, hidden_size, num_heads * v_head_dim)

    return output.reshape(B, S, hidden_size), q_resid.reshape(B, S, q_lora_rank)


def test_attention_small(glm, device):
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

    ref_output, ref_q_resid = attention_forward_torch(
        hidden_states, cos, sin, attention_mask,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)

    cuda_output, cuda_q_resid = attention_forward_cuda(
        glm, device, hidden_states, cos, sin, attention_mask,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)

    torch.testing.assert_close(cuda_output.cpu(), ref_output.cpu(), atol=0.1, rtol=5e-3)
    torch.testing.assert_close(cuda_q_resid.cpu(), ref_q_resid.cpu(), atol=5e-3, rtol=5e-3)

def test_attention_no_mask(glm, device):
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

    ref_output, ref_q_resid = attention_forward_torch(
        hidden_states, cos, sin, None,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)

    cuda_output, cuda_q_resid = attention_forward_cuda(
        glm, device, hidden_states, cos, sin, None,
        q_a_proj_w, q_a_layernorm_w, q_b_proj_w,
        kv_a_proj_with_mqa_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
        indexer_weights,
        num_heads, qk_nope_dim, qk_rope_dim, v_head_dim,
        q_lora_rank, kv_lora_rank, hidden_size, eps)

    torch.testing.assert_close(cuda_output.cpu(), ref_output.cpu(), atol=0.1, rtol=5e-3)
    torch.testing.assert_close(cuda_q_resid.cpu(), ref_q_resid.cpu(), atol=5e-3, rtol=5e-3)
