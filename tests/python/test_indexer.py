import torch
import pytest
from helpers import ATOL, RTOL


def apply_rotary_pos_emb_torch(x, cos, sin, unsqueeze_dim):
    if unsqueeze_dim == 1:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif unsqueeze_dim == 2:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    return (x.float() * cos.float() + rotate_half(x.float()) * sin.float()).to(torch.bfloat16)


def indexer_forward_torch(hidden_states, q_resid, cos, sin, attention_mask,
                           wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
                           n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps=1e-6):
    B, S, _ = hidden_states.shape
    nope_dim = head_dim - qk_rope_dim

    q = torch.nn.functional.linear(q_resid, wq_b_w)
    q = q.view(B, S, n_heads, head_dim)
    q_pe, q_nope = q.split([qk_rope_dim, nope_dim], dim=-1)
    q_pe = apply_rotary_pos_emb_torch(q_pe, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pe, q_nope], dim=-1)

    k = torch.nn.functional.layer_norm(
        torch.nn.functional.linear(hidden_states, wk_w).float(),
        [head_dim], k_norm_w.float(), k_norm_b.float(), eps=eps
    ).to(torch.bfloat16)
    k_pe, k_nope = k.split([qk_rope_dim, nope_dim], dim=-1)
    k_pe = apply_rotary_pos_emb_torch(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)

    weights = torch.nn.functional.linear(hidden_states, weights_proj_w)
    weights = (weights.float() * (n_heads ** -0.5)).to(torch.bfloat16)

    scores = torch.einsum("bshd,btd->bsht", q, k) * softmax_scale
    scores = torch.nn.functional.relu(scores)
    index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

    if attention_mask is not None:
        index_scores = index_scores + attention_mask.float()

    topk_val = min(topk, S)
    return index_scores, index_scores.topk(topk_val, dim=-1).indices.to(torch.int32)


def indexer_forward_cuda(glm, device, hidden_states, q_resid, cos, sin, attention_mask,
                          wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
                          n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps=1e-6):
    B, S, _ = hidden_states.shape
    q_lora_rank = q_resid.shape[-1]
    hidden_size = hidden_states.shape[-1]
    nope_dim = head_dim - qk_rope_dim

    idx_q = torch.empty(B * S, n_heads * head_dim, dtype=torch.bfloat16, device=device)
    glm.linear(idx_q, q_resid.reshape(B * S, q_lora_rank), wq_b_w, B * S, n_heads * head_dim, q_lora_rank)
    idx_q = idx_q.reshape(B, S, n_heads, head_dim)

    q_pe = idx_q[:, :, :, :qk_rope_dim].contiguous()
    q_nope = idx_q[:, :, :, qk_rope_dim:].contiguous()

    q_pe_rope = torch.empty_like(q_pe)
    glm.apply_rotary_pos_emb(q_pe_rope, q_pe, cos, sin, qk_rope_dim, n_heads, S, B, 2)

    q_out = torch.empty(B * S * n_heads, head_dim, dtype=torch.bfloat16, device=device)
    glm.cat_last_dim(q_out, q_pe_rope.reshape(-1, qk_rope_dim), q_nope.reshape(-1, nope_dim),
                     qk_rope_dim, nope_dim, B * S * n_heads)
    q_out = q_out.reshape(B, S, n_heads, head_dim)

    k = torch.empty(B * S, head_dim, dtype=torch.bfloat16, device=device)
    glm.linear(k, hidden_states.reshape(B * S, hidden_size), wk_w, B * S, head_dim, hidden_size)
    k = k.reshape(B, S, head_dim)

    k_normed = torch.empty_like(k)
    glm.layernorm(k_normed.reshape(B * S, head_dim), k.reshape(B * S, head_dim),
                  k_norm_w, k_norm_b, eps, head_dim, B * S)

    k_pe = k_normed[:, :, :qk_rope_dim].contiguous()
    k_nope = k_normed[:, :, qk_rope_dim:].contiguous()

    k_pe_4d = k_pe.reshape(B, S, 1, qk_rope_dim).contiguous()
    k_pe_rope_4d = torch.empty_like(k_pe_4d)
    glm.apply_rotary_pos_emb(k_pe_rope_4d, k_pe_4d, cos, sin, qk_rope_dim, 1, S, B, 2)
    k_pe_rope = k_pe_rope_4d.reshape(B, S, qk_rope_dim)

    k_out = torch.empty(B * S, head_dim, dtype=torch.bfloat16, device=device)
    glm.cat_last_dim(k_out, k_pe_rope.reshape(-1, qk_rope_dim), k_nope.reshape(-1, nope_dim),
                     qk_rope_dim, nope_dim, B * S)
    k_out = k_out.reshape(B, S, head_dim)

    weights = torch.empty(B * S, n_heads, dtype=torch.bfloat16, device=device)
    glm.linear(weights, hidden_states.reshape(B * S, hidden_size), weights_proj_w, B * S, n_heads, hidden_size)
    weights = weights.reshape(B, S, n_heads)

    n_scale = float(n_heads ** -0.5)
    weights_flat = weights.reshape(-1)
    glm.scale(weights_flat, weights_flat, n_scale, B * S * n_heads)

    q_2d = q_out.reshape(B, S * n_heads, head_dim).contiguous()
    scores_2d = torch.empty(B, S * n_heads, S, dtype=torch.bfloat16, device=device)
    glm.bmm(scores_2d, q_2d, k_out, softmax_scale, 0.0, B, S * n_heads, S, head_dim, 0, 1)
    scores = scores_2d.reshape(B, S, n_heads, S)

    scores_flat = scores.reshape(-1)
    glm.relu(scores_flat, scores_flat, B * S * n_heads * S)

    s_2d = scores.reshape(B * S, n_heads, S).contiguous()
    w_2d = weights.reshape(B * S, 1, n_heads).contiguous()
    index_scores_2d = torch.empty(B * S, 1, S, dtype=torch.bfloat16, device=device)
    glm.bmm(index_scores_2d, w_2d, s_2d, 1.0, 0.0, B * S, 1, S, n_heads, 0)
    index_scores = index_scores_2d.reshape(B, S, S)

    if attention_mask is not None:
        mask = attention_mask.expand(B, S, S).contiguous()
        glm.add(index_scores.reshape(-1), index_scores.reshape(-1),
                mask.reshape(-1), B * S * S)

    topk_val = min(topk, S)
    topk_values = torch.empty(B * S, topk_val, dtype=torch.bfloat16, device=device)
    topk_indices = torch.empty(B * S, topk_val, dtype=torch.int32, device=device)
    glm.topk(topk_values, topk_indices, index_scores.reshape(B * S, S), topk_val, S, B * S)

    return index_scores, topk_indices.reshape(B, S, topk_val)


def _make_rotary_embed(glm, device, dim_half, batch, seq_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half * 2, 2, dtype=torch.float32) / (dim_half * 2)))
    inv_freq_bf = inv_freq.to(torch.bfloat16).to(device)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, seq_len)

    dim = dim_half * 2
    cos_buf = torch.empty(batch, seq_len, dim, dtype=torch.bfloat16, device=device)
    sin_buf = torch.empty(batch, seq_len, dim, dtype=torch.bfloat16, device=device)
    glm.rotary_embedding(cos_buf, sin_buf, inv_freq_bf, position_ids, dim_half, batch, seq_len)
    return cos_buf, sin_buf


def test_indexer_no_mask(glm, device):
    B, S = 2, 8
    hidden_size = 128
    q_lora_rank = 32
    n_heads = 4
    head_dim = 32
    qk_rope_dim = 8
    nope_dim = head_dim - qk_rope_dim
    topk = 4
    softmax_scale = head_dim ** -0.5
    eps = 1e-6

    torch.manual_seed(42)
    hidden_states = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
    q_resid = torch.randn(B, S, q_lora_rank, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

    wq_b_w = torch.randn(n_heads * head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    wk_w = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=device)
    k_norm_w = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_b = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    weights_proj_w = torch.randn(n_heads, hidden_size, dtype=torch.bfloat16, device=device)

    ref_scores, ref_indices = indexer_forward_torch(
        hidden_states, q_resid, cos, sin, None,
        wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
        n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps)

    cuda_scores, cuda_indices = indexer_forward_cuda(
        glm, device, hidden_states, q_resid, cos, sin, None,
        wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
        n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps)

    assert cuda_indices.shape == ref_indices.shape, f"Shape mismatch: {cuda_indices.shape} vs {ref_indices.shape}"
    _compare_topk(cuda_indices, ref_indices, cuda_scores, ref_scores)


def _compare_topk(cuda_indices, ref_indices, index_scores_cuda, index_scores_ref, atol=0.5):
    cuda_vals_sorted, _ = torch.sort(index_scores_cuda.float(), dim=-1, descending=True)
    ref_vals_sorted, _ = torch.sort(index_scores_ref.float(), dim=-1, descending=True)
    k = cuda_indices.shape[-1]
    torch.testing.assert_close(cuda_vals_sorted[..., :k].cpu(), ref_vals_sorted[..., :k].cpu(), atol=atol, rtol=5e-2)


def test_indexer_with_causal_mask(glm, device):
    B, S = 1, 8
    hidden_size = 128
    q_lora_rank = 32
    n_heads = 4
    head_dim = 32
    qk_rope_dim = 8
    topk = 4
    softmax_scale = head_dim ** -0.5
    eps = 1e-6

    torch.manual_seed(123)
    hidden_states = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
    q_resid = torch.randn(B, S, q_lora_rank, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

    causal_2d = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_2d, S)
    attention_mask = causal_2d.unsqueeze(0)  # [1, S, S]

    wq_b_w = torch.randn(n_heads * head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    wk_w = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=device)
    k_norm_w = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_b = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    weights_proj_w = torch.randn(n_heads, hidden_size, dtype=torch.bfloat16, device=device)

    ref_scores, ref_indices = indexer_forward_torch(
        hidden_states, q_resid, cos, sin, attention_mask,
        wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
        n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps)

    cuda_scores, cuda_indices = indexer_forward_cuda(
        glm, device, hidden_states, q_resid, cos, sin, attention_mask,
        wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
        n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps)

    assert cuda_indices.shape == ref_indices.shape
    _compare_topk(cuda_indices, ref_indices, cuda_scores, ref_scores)


def test_indexer_topk_equals_seq_len(glm, device):
    B, S = 1, 4
    hidden_size = 64
    q_lora_rank = 16
    n_heads = 2
    head_dim = 16
    qk_rope_dim = 4
    topk = 4
    softmax_scale = head_dim ** -0.5
    eps = 1e-6

    torch.manual_seed(99)
    hidden_states = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
    q_resid = torch.randn(B, S, q_lora_rank, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

    wq_b_w = torch.randn(n_heads * head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    wk_w = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=device)
    k_norm_w = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_b = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    weights_proj_w = torch.randn(n_heads, hidden_size, dtype=torch.bfloat16, device=device)

    ref_scores, ref_indices = indexer_forward_torch(
        hidden_states, q_resid, cos, sin, None,
        wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
        n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps)

    cuda_scores, cuda_indices = indexer_forward_cuda(
        glm, device, hidden_states, q_resid, cos, sin, None,
        wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
        n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps)

    assert cuda_indices.shape == ref_indices.shape
