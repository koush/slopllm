import torch
import pytest


def test_rope_transpose_no_rope(glm, device):
    batch, n_heads, seq_len, head_dim = 2, 4, 8, 64
    in_stride = head_dim
    x = torch.randn(batch * seq_len, n_heads * in_stride, dtype=torch.bfloat16, device=device)

    cos_dummy = torch.zeros(batch, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    sin_dummy = torch.zeros(batch, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    out_cuda = torch.empty(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    glm.ropeTranspose(out_cuda, x, cos_dummy, sin_dummy, 0, head_dim, n_heads, seq_len, batch, in_stride)

    x_ref = x.reshape(batch, seq_len, n_heads, head_dim).reshape(batch * seq_len, n_heads, head_dim)
    torch.testing.assert_close(out_cuda.cpu(), x_ref.cpu(), atol=0, rtol=0)


def test_rope_transpose_with_rope(glm, device):
    batch, n_heads, seq_len = 2, 4, 8
    rope_dim = 32
    head_dim = 32  # full RoPE: rope_dim == head_dim
    in_stride = head_dim

    inv_freq = torch.zeros(rope_dim // 2, dtype=torch.bfloat16, device=device)
    for i in range(rope_dim // 2):
        inv_freq[i] = 1.0 / (10000.0 ** (2.0 * i / rope_dim))
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    cos_out = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    sin_out = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    glm.rotary_embedding(cos_out, sin_out, inv_freq, position_ids, rope_dim // 2, batch, seq_len)

    x = torch.randn(batch * seq_len, n_heads * in_stride, dtype=torch.bfloat16, device=device)

    out_cuda = torch.empty(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    glm.ropeTranspose(out_cuda, x, cos_out, sin_out, rope_dim, head_dim, n_heads, seq_len, batch, in_stride)

    # Reference: transpose then apply_rotary_pos_emb (full rope_dim == head_dim)
    out_transpose = torch.empty(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    glm.ropeTranspose(out_transpose, x, cos_out, sin_out, 0, head_dim, n_heads, seq_len, batch, in_stride)
    out_rope = torch.empty(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    glm.apply_rotary_pos_emb(out_rope, out_transpose, cos_out, sin_out, rope_dim, n_heads, seq_len, batch, 0)

    torch.testing.assert_close(out_cuda.cpu(), out_rope.cpu(), atol=0, rtol=0)


def test_rope_transpose_partial_rope(glm, device):
    batch, n_heads, seq_len = 1, 2, 4
    rope_dim = 8
    head_dim = 16
    in_stride = head_dim

    inv_freq = torch.zeros(rope_dim // 2, dtype=torch.bfloat16, device=device)
    for i in range(rope_dim // 2):
        inv_freq[i] = 1.0 / (10000.0 ** (2.0 * i / rope_dim))
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    cos_out = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    sin_out = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    glm.rotary_embedding(cos_out, sin_out, inv_freq, position_ids, rope_dim // 2, batch, seq_len)

    x = torch.randn(batch * seq_len, n_heads * in_stride, dtype=torch.bfloat16, device=device)
    out_cuda = torch.empty(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    glm.ropeTranspose(out_cuda, x, cos_out, sin_out, rope_dim, head_dim, n_heads, seq_len, batch, in_stride)

    # Reference: transpose then apply_rotary_pos_emb_partial
    out_transpose = torch.empty(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    glm.ropeTranspose(out_transpose, x, cos_out, sin_out, 0, head_dim, n_heads, seq_len, batch, in_stride)
    out_rope = torch.empty(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    glm.apply_rotary_pos_emb_partial(out_rope, out_transpose, cos_out, sin_out, rope_dim, head_dim, n_heads, seq_len, batch, 0)

    torch.testing.assert_close(out_cuda.cpu(), out_rope.cpu(), atol=0, rtol=0)


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
