import torch
import pytest
from helpers import ATOL, RTOL, pack_indexer_k, unpack_indexer_k


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


def apply_rotary_pos_emb_torch_interleaved(x, cos, sin, unsqueeze_dim):
    dim_half = cos.shape[-1] // 2
    cos_half = cos[..., :dim_half]
    sin_half = sin[..., :dim_half]
    if unsqueeze_dim == 1:
        cos_half = cos_half.unsqueeze(1)
        sin_half = sin_half.unsqueeze(1)
    elif unsqueeze_dim == 2:
        cos_half = cos_half.unsqueeze(2)
        sin_half = sin_half.unsqueeze(2)
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    o1 = x1 * cos_half.float() - x2 * sin_half.float()
    o2 = x2 * cos_half.float() + x1 * sin_half.float()
    return torch.stack((o1, o2), dim=-1).flatten(-2).to(torch.bfloat16)


def indexer_forward_torch(hidden_states, q_resid, cos, sin, attention_mask,
                           wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
                           n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps=1e-6):
    B, S, _ = hidden_states.shape
    nope_dim = head_dim - qk_rope_dim

    q = torch.nn.functional.linear(q_resid, wq_b_w)
    q = q.view(B, S, n_heads, head_dim)
    q_pe, q_nope = q.split([qk_rope_dim, nope_dim], dim=-1)
    q_pe = apply_rotary_pos_emb_torch_interleaved(q_pe, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pe, q_nope], dim=-1)

    k = torch.nn.functional.layer_norm(
        torch.nn.functional.linear(hidden_states, wk_w).float(),
        [head_dim], k_norm_w.float(), k_norm_b.float(), eps=eps
    ).to(torch.bfloat16)
    k_pe, k_nope = k.split([qk_rope_dim, nope_dim], dim=-1)
    k_pe = apply_rotary_pos_emb_torch_interleaved(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
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

    if nope_dim > 0:
        q_pe = idx_q[:, :, :, :qk_rope_dim].contiguous()
        q_nope = idx_q[:, :, :, qk_rope_dim:].contiguous()
        q_pe_rope = torch.empty_like(q_pe)
        glm.apply_rotary_pos_emb(q_pe_rope, q_pe, cos, sin, qk_rope_dim, n_heads, S, B, 2, interleaved=True)
        q_out = torch.empty(B * S * n_heads, head_dim, dtype=torch.bfloat16, device=device)
        glm.cat_last_dim(q_out, q_pe_rope.reshape(-1, qk_rope_dim), q_nope.reshape(-1, nope_dim),
                         qk_rope_dim, nope_dim, B * S * n_heads)
    else:
        q_out = torch.empty_like(idx_q)
        glm.apply_rotary_pos_emb(q_out, idx_q, cos, sin, qk_rope_dim, n_heads, S, B, 2, interleaved=True)
    q_out = q_out.reshape(B, S, n_heads, head_dim)

    k = torch.empty(B * S, head_dim, dtype=torch.bfloat16, device=device)
    glm.linear(k, hidden_states.reshape(B * S, hidden_size), wk_w, B * S, head_dim, hidden_size)
    k = k.reshape(B, S, head_dim)

    k_normed = torch.empty_like(k)
    glm.layernorm(k_normed.reshape(B * S, head_dim), k.reshape(B * S, head_dim),
                  k_norm_w, k_norm_b, eps, head_dim, B * S)

    if nope_dim > 0:
        k_pe = k_normed[:, :, :qk_rope_dim].contiguous()
        k_nope = k_normed[:, :, qk_rope_dim:].contiguous()
        k_pe_4d = k_pe.reshape(B, S, 1, qk_rope_dim).contiguous()
        k_pe_rope_4d = torch.empty_like(k_pe_4d)
        glm.apply_rotary_pos_emb(k_pe_rope_4d, k_pe_4d, cos, sin, qk_rope_dim, 1, S, B, 2, interleaved=True)
        k_pe_rope = k_pe_rope_4d.reshape(B, S, qk_rope_dim)
        k_out = torch.empty(B * S, head_dim, dtype=torch.bfloat16, device=device)
        glm.cat_last_dim(k_out, k_pe_rope.reshape(-1, qk_rope_dim), k_nope.reshape(-1, nope_dim),
                         qk_rope_dim, nope_dim, B * S)
    else:
        k_out = torch.empty_like(k_normed)
        k_pe_4d = k_normed.reshape(B, S, 1, qk_rope_dim).contiguous()
        glm.apply_rotary_pos_emb(k_out.reshape(B, S, 1, qk_rope_dim), k_pe_4d, cos, sin, qk_rope_dim, 1, S, B, 2, interleaved=True)
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
        glm, device, hidden_states, q_resid, cos, sin, None,
        wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
        n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps)

    assert cuda_indices.shape == ref_indices.shape


def test_indexer_nope_dim_zero(glm, device):
    """Test indexer with head_dim == qk_rope_dim (nope_dim=0), matching the test model config."""
    B, S = 1, 8
    hidden_size = 128
    q_lora_rank = 32
    n_heads = 4
    head_dim = 32
    qk_rope_dim = 32  # nope_dim = 0
    topk = 4
    softmax_scale = head_dim ** -0.5
    eps = 1e-6

    torch.manual_seed(77)
    hidden_states = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
    q_resid = torch.randn(B, S, q_lora_rank, dtype=torch.bfloat16, device=device)
    cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

    wq_b_w = torch.randn(n_heads * head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    wk_w = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=device)
    k_norm_w = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_b = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    weights_proj_w = torch.randn(n_heads, hidden_size, dtype=torch.bfloat16, device=device)

    causal_2d = torch.empty(S, S, dtype=torch.bfloat16, device=device)
    glm.causal_mask(causal_2d, S)
    attention_mask = causal_2d.unsqueeze(0)

    ref_scores, ref_indices = indexer_forward_torch(
        hidden_states, q_resid, cos, sin, attention_mask,
        wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
        n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps)

    cuda_scores, cuda_indices = indexer_forward_cuda(
        glm, device, hidden_states, q_resid, cos, sin, None,
        wq_b_w, wk_w, k_norm_w, k_norm_b, weights_proj_w,
        n_heads, head_dim, qk_rope_dim, topk, softmax_scale, eps)

    assert cuda_indices.shape == ref_indices.shape


# ---------------------------------------------------------------------------
# Fused indexer score kernel tests
# ---------------------------------------------------------------------------

def indexer_score_torch(q, k_paged, k_scale_paged, weights, page_indices, page_indptr,
                        last_page_len, qo_indptr, scale, page_size, max_kv_len, causal):
    """Pure torch reference for the fused indexer score kernel."""
    totalQ, n_heads, head_dim = q.shape
    if k_paged.dtype == torch.uint8:
        k_paged = unpack_indexer_k(k_paged, k_scale_paged)
    out = torch.full((totalQ, max_kv_len), float('-inf'), dtype=torch.bfloat16, device=q.device)

    for seq_idx in range(len(qo_indptr) - 1):
        q_start, q_end = int(qo_indptr[seq_idx]), int(qo_indptr[seq_idx + 1])
        num_queries = q_end - q_start

        page_start, page_end = int(page_indptr[seq_idx]), int(page_indptr[seq_idx + 1])
        num_pages = page_end - page_start
        kv_len = (num_pages - 1) * page_size + int(last_page_len[seq_idx]) if num_pages > 0 else 0
        prefix_len = max(0, kv_len - num_queries)

        # Gather K for this sequence
        k_parts = []
        for p in range(page_start, page_end):
            page_id = int(page_indices[p])
            is_last = (p == page_end - 1)
            tokens = int(last_page_len[seq_idx]) if is_last else page_size
            k_parts.append(k_paged[page_id, :tokens, :])
        k_seq = torch.cat(k_parts, dim=0)  # [kv_len, head_dim]

        for qi in range(num_queries):
            q_idx = q_start + qi
            causal_limit = prefix_len + qi if causal else kv_len - 1
            valid_k = min(kv_len, causal_limit + 1)
            if valid_k <= 0:
                continue
            # scores: [n_heads, valid_k]
            scores = torch.einsum('hd,td->ht', q[q_idx].float(), k_seq[:valid_k].float()) * scale
            scores = torch.clamp(scores, min=0.0)
            # index_score: [valid_k]
            index_score = torch.einsum('h,ht->t', weights[q_idx].float(), scores)
            out[q_idx, :valid_k] = index_score.to(torch.bfloat16)

    return out


def _run_indexer_score_test(glm, device, B, seq_lens, n_heads, head_dim, page_size, causal, seed=42):
    """Setup random paged KV, q, weights; compare fused kernel vs torch reference."""
    torch.manual_seed(seed)
    total_q = sum(seq_lens)
    max_kv_len = max(seq_lens)
    num_pages_total = sum((s + page_size - 1) // page_size for s in seq_lens)
    max_pages = num_pages_total + 16

    # Random paged K cache
    k_paged, k_scale_paged = pack_indexer_k(torch.randn(
        max_pages, page_size, head_dim, dtype=torch.bfloat16, device=device))
    # Random q and weights
    q = torch.randn(total_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(total_q, n_heads, dtype=torch.bfloat16, device=device)
    scale = head_dim ** -0.5

    # Build page indices and indptr
    page_indices_list = []
    page_indptr = [0]
    last_page_len_list = []
    page_id = 0
    for s in seq_lens:
        num_pages_s = (s + page_size - 1) // page_size
        page_indices_list.extend(range(page_id, page_id + num_pages_s))
        page_id += num_pages_s
        page_indptr.append(page_indptr[-1] + num_pages_s)
        last_page_len_list.append(s - (num_pages_s - 1) * page_size)

    # qo_indptr
    qo_indptr = [0]
    for s in seq_lens:
        qo_indptr.append(qo_indptr[-1] + s)

    page_indices_t = torch.tensor(page_indices_list, dtype=torch.int32, device=device)
    page_indptr_t = torch.tensor(page_indptr, dtype=torch.int32, device=device)
    last_page_len_t = torch.tensor(last_page_len_list, dtype=torch.int32, device=device)
    qo_indptr_t = torch.tensor(qo_indptr, dtype=torch.int32, device=device)

    # Torch reference
    ref_out = indexer_score_torch(
        q, k_paged, k_scale_paged, weights, page_indices_list, page_indptr,
        last_page_len_list, qo_indptr, scale, page_size, max_kv_len, causal)

    # CUDA kernel
    cuda_out = torch.full((total_q, max_kv_len), float('-inf'), dtype=torch.bfloat16, device=device)
    glm.indexer_score(cuda_out, q, k_paged, k_scale_paged, weights, page_indices_t, page_indptr_t,
                      last_page_len_t, qo_indptr_t, scale, total_q, n_heads, head_dim,
                      page_size, max_kv_len, causal)

    # Compare (only valid positions, not -inf padding)
    for seq_idx in range(B):
        q_start, q_end = qo_indptr[seq_idx], qo_indptr[seq_idx + 1]
        page_start, page_end = page_indptr[seq_idx], page_indptr[seq_idx + 1]
        num_pages = page_end - page_start
        kv_len = (num_pages - 1) * page_size + last_page_len_list[seq_idx] if num_pages > 0 else 0
        prefix_len = max(0, kv_len - (q_end - q_start))
        for qi in range(q_end - q_start):
            q_idx = q_start + qi
            causal_limit = prefix_len + qi if causal else kv_len - 1
            valid_k = min(kv_len, causal_limit + 1)
            if valid_k <= 0:
                continue
            ref_row = ref_out[q_idx, :valid_k].float()
            cuda_row = cuda_out[q_idx, :valid_k].float()
            torch.testing.assert_close(cuda_row.cpu(), ref_row.cpu(), atol=0.5, rtol=5e-2)

    # Verify -inf padding
    for seq_idx in range(B):
        q_start, q_end = qo_indptr[seq_idx], qo_indptr[seq_idx + 1]
        page_start, page_end = page_indptr[seq_idx], page_indptr[seq_idx + 1]
        num_pages = page_end - page_start
        kv_len = (num_pages - 1) * page_size + last_page_len_list[seq_idx] if num_pages > 0 else 0
        for qi in range(q_end - q_start):
            q_idx = q_start + qi
            if kv_len < max_kv_len:
                assert torch.all(cuda_out[q_idx, kv_len:].isneginf()), f"Row {q_idx} not -inf after kvLen={kv_len}"


def test_indexer_score_single_seq_causal(glm, device):
    _run_indexer_score_test(glm, device, B=1, seq_lens=[16], n_heads=4, head_dim=64,
                            page_size=16, causal=True, seed=42)


def test_indexer_score_single_seq_noncausal(glm, device):
    _run_indexer_score_test(glm, device, B=1, seq_lens=[16], n_heads=4, head_dim=64,
                            page_size=16, causal=False, seed=42)


def test_indexer_score_multi_page(glm, device):
    _run_indexer_score_test(glm, device, B=1, seq_lens=[48], n_heads=4, head_dim=64,
                            page_size=16, causal=True, seed=99)


def test_indexer_score_multi_seq(glm, device):
    _run_indexer_score_test(glm, device, B=3, seq_lens=[16, 32, 8], n_heads=4, head_dim=64,
                            page_size=16, causal=True, seed=123)


def test_indexer_score_uneven_pages(glm, device):
    """Last page partially filled (seq_len not multiple of page_size)."""
    _run_indexer_score_test(glm, device, B=2, seq_lens=[20, 33], n_heads=4, head_dim=64,
                            page_size=16, causal=True, seed=77)


def test_indexer_score_nope_dim_zero(glm, device):
    """head_dim == qk_rope_dim (test model config: 64==64)."""
    _run_indexer_score_test(glm, device, B=2, seq_lens=[16, 32], n_heads=4, head_dim=32,
                            page_size=16, causal=True, seed=55)


def test_indexer_score_decode_pattern(glm, device):
    """Simulate decode: each seq has 1 query token, kvLen > 1 (prefix)."""
    _run_indexer_score_test(glm, device, B=4, seq_lens=[1, 1, 1, 1], n_heads=4, head_dim=64,
                            page_size=16, causal=True, seed=88)


def test_indexer_score_chunked_prefill(glm, device):
    """Simulate chunked prefill: 1 seq with prefix (kvLen > numQueries)."""
    torch.manual_seed(42)
    n_heads, head_dim, page_size = 4, 64, 16
    total_q = 8  # chunk size
    kv_len = 48  # includes prefix
    max_kv_len = kv_len
    num_pages = (kv_len + page_size - 1) // page_size  # 3 pages
    max_pages = num_pages + 8

    k_paged, k_scale_paged = pack_indexer_k(torch.randn(
        max_pages, page_size, head_dim, dtype=torch.bfloat16, device=device))
    q = torch.randn(total_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(total_q, n_heads, dtype=torch.bfloat16, device=device)
    scale = head_dim ** -0.5

    page_indices = list(range(num_pages))
    page_indptr = [0, num_pages]
    last_page_len = [kv_len - (num_pages - 1) * page_size]  # 48 - 2*16 = 16
    qo_indptr = [0, total_q]  # 8 queries, all from seq 0

    page_indices_t = torch.tensor(page_indices, dtype=torch.int32, device=device)
    page_indptr_t = torch.tensor(page_indptr, dtype=torch.int32, device=device)
    last_page_len_t = torch.tensor(last_page_len, dtype=torch.int32, device=device)
    qo_indptr_t = torch.tensor(qo_indptr, dtype=torch.int32, device=device)

    ref_out = indexer_score_torch(
        q, k_paged, k_scale_paged, weights, page_indices, page_indptr,
        last_page_len, qo_indptr, scale, page_size, max_kv_len, True)

    cuda_out = torch.full((total_q, max_kv_len), float('-inf'), dtype=torch.bfloat16, device=device)
    glm.indexer_score(cuda_out, q, k_paged, k_scale_paged, weights, page_indices_t, page_indptr_t,
                      last_page_len_t, qo_indptr_t, scale, total_q, n_heads, head_dim,
                      page_size, max_kv_len, True)

    # prefix_len = 48 - 8 = 40; causal_limit for qi = 40 + qi
    for qi in range(total_q):
        causal_limit = 40 + qi
        valid_k = min(kv_len, causal_limit + 1)
        ref_row = ref_out[qi, :valid_k].float()
        cuda_row = cuda_out[qi, :valid_k].float()
        torch.testing.assert_close(cuda_row.cpu(), ref_row.cpu(), atol=0.5, rtol=5e-2)
        # Verify positions beyond causal limit are -inf
        if valid_k < max_kv_len:
            assert torch.all(cuda_out[qi, valid_k:].isneginf()), f"Row {qi} not -inf after causal_limit={causal_limit}"


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
