import torch
import pytest


def _make_rope_embeddings(glm, device, rope_dim, batch, seq_len, theta=10000.0):
    inv_freq = torch.zeros(rope_dim // 2, dtype=torch.bfloat16, device=device)
    for i in range(rope_dim // 2):
        inv_freq[i] = 1.0 / (theta ** (2.0 * i / rope_dim))
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    cos_out = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    sin_out = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    glm.rotary_embedding(cos_out, sin_out, inv_freq, position_ids, rope_dim // 2, batch, seq_len)
    return cos_out, sin_out


def _ref_token_major_rope(x, cos_out, sin_out, rope_dim, head_dim):
    # x: [batch*seq_len, n_heads, in_stride] token-major (unsqueeze_dim=2 layout).
    # Gathers the first head_dim dims of each head window (stride compaction)
    # and rotates the first rope_dim dims with neox-style RoPE.
    batch, seq_len = cos_out.shape[0], cos_out.shape[1]
    n_heads, in_stride = x.shape[1], x.shape[2]
    gathered = x.reshape(batch, seq_len, n_heads, in_stride)[..., :head_dim].clone()
    if rope_dim > 0:
        cos = cos_out.unsqueeze(2).float()
        sin = sin_out.unsqueeze(2).float()
        rot = gathered[..., :rope_dim].float()
        x1 = rot[..., : rope_dim // 2]
        x2 = rot[..., rope_dim // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        gathered[..., :rope_dim] = (rot * cos + rotated * sin).to(torch.bfloat16)
    return gathered.reshape(batch * seq_len, n_heads, head_dim)


def test_apply_rotary_pos_emb_token_major_no_rope(glm, device):
    batch, n_heads, seq_len, head_dim = 2, 4, 8, 64
    x = torch.randn(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)

    # cos/sin pointers are unused when rope_dim == 0
    out_cuda = torch.empty_like(x)
    glm.apply_rotary_pos_emb(out_cuda, x, x, x, 0, head_dim, n_heads, seq_len, batch, 2)

    torch.testing.assert_close(out_cuda.cpu(), x.cpu(), atol=0, rtol=0)


def test_apply_rotary_pos_emb_token_major_full_rope(glm, device):
    batch, n_heads, seq_len = 2, 4, 8
    rope_dim = 32
    head_dim = 32  # full RoPE: rope_dim == head_dim

    cos_out, sin_out = _make_rope_embeddings(glm, device, rope_dim, batch, seq_len)

    x = torch.randn(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    out_cuda = torch.empty_like(x)
    glm.apply_rotary_pos_emb(out_cuda, x, cos_out, sin_out, rope_dim, head_dim, n_heads, seq_len, batch, 2)

    ref = _ref_token_major_rope(x, cos_out, sin_out, rope_dim, head_dim)
    torch.testing.assert_close(out_cuda.cpu(), ref.cpu(), atol=2e-3, rtol=2e-3)


def test_apply_rotary_pos_emb_token_major_partial_rope(glm, device):
    batch, n_heads, seq_len = 1, 2, 4
    rope_dim = 8
    head_dim = 16

    cos_out, sin_out = _make_rope_embeddings(glm, device, rope_dim, batch, seq_len)

    x = torch.randn(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    out_cuda = torch.empty_like(x)
    glm.apply_rotary_pos_emb(out_cuda, x, cos_out, sin_out, rope_dim, head_dim, n_heads, seq_len, batch, 2)

    ref = _ref_token_major_rope(x, cos_out, sin_out, rope_dim, head_dim)
    torch.testing.assert_close(out_cuda.cpu(), ref.cpu(), atol=2e-3, rtol=2e-3)
    # Non-RoPE dims pass through unchanged.
    torch.testing.assert_close(out_cuda[..., rope_dim:].cpu(), x[..., rope_dim:].cpu(), atol=0, rtol=0)


def test_apply_rotary_pos_emb_token_major_stride_compaction(glm, device):
    batch, n_heads, seq_len = 2, 3, 5
    rope_dim = 16
    head_dim = 32
    in_stride = 48  # > head_dim: first head_dim dims of each head window are packed

    cos_out, sin_out = _make_rope_embeddings(glm, device, rope_dim, batch, seq_len)

    x = torch.randn(batch * seq_len, n_heads, in_stride, dtype=torch.bfloat16, device=device)
    out_cuda = torch.empty(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    glm.apply_rotary_pos_emb(out_cuda, x, cos_out, sin_out, rope_dim, head_dim, n_heads, seq_len, batch, 2, in_stride=in_stride)

    ref = _ref_token_major_rope(x, cos_out, sin_out, rope_dim, head_dim)
    torch.testing.assert_close(out_cuda.cpu(), ref.cpu(), atol=2e-3, rtol=2e-3)


def test_mla_v_expand(glm, device):
    batch, n_heads, seq_len = 2, 4, 3
    kv_lora_rank = 8
    v_head_dim = 6

    attn_out = torch.randn(batch * n_heads, seq_len, kv_lora_rank, dtype=torch.bfloat16, device=device)
    v_proj = torch.randn(n_heads * kv_lora_rank, v_head_dim, dtype=torch.bfloat16, device=device)

    result_cuda = torch.empty(batch * seq_len, n_heads * v_head_dim, dtype=torch.bfloat16, device=device)
    glm.mlaVExpand(result_cuda, attn_out, v_proj, kv_lora_rank, v_head_dim, n_heads, seq_len, batch)

    result_ref = torch.zeros(batch, seq_len, n_heads, v_head_dim, dtype=torch.float32, device=device)
    for b in range(batch):
        for s in range(seq_len):
            for h in range(n_heads):
                for j in range(v_head_dim):
                    sum_val = 0.0
                    for k in range(kv_lora_rank):
                        a = attn_out[(b * n_heads + h), s, k].float()
                        w = v_proj[h * kv_lora_rank + k, j].float()
                        sum_val += a * w
                    result_ref[b, s, h, j] = sum_val

    result_ref_flat = result_ref.reshape(batch * seq_len, n_heads * v_head_dim).bfloat16()
    torch.testing.assert_close(result_cuda.cpu(), result_ref_flat.cpu(), atol=1e-1, rtol=1e-1)
