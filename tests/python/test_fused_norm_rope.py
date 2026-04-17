import torch
import pytest


def torch_rmsnorm(x, weight, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    return (weight.float() * x_f * inv_rms).to(torch.bfloat16)


def torch_apply_rope(x, cos, sin, rope_dim, head_dim):
    x_f = x.float()
    cos_f = cos.float()
    sin_f = sin.float()
    half = rope_dim // 2
    out = x_f.clone()
    out[..., :rope_dim] = x_f[..., :rope_dim] * cos_f + torch.cat([
        -x_f[..., half:rope_dim], x_f[..., :half]
    ], dim=-1) * sin_f
    return out.to(torch.bfloat16)


def torch_norm_rope(q_in, k_in, q_weight, k_weight, cos, sin, eps, rope_dim, head_dim, n_heads, n_kv, seq_len, batch):
    q_in_2d = q_in.reshape(batch * seq_len, n_heads * head_dim)
    k_in_2d = k_in.reshape(batch * seq_len, n_kv * head_dim)

    q_normed = torch_rmsnorm(q_in_2d.reshape(batch * seq_len * n_heads, head_dim), q_weight, eps)
    k_normed = torch_rmsnorm(k_in_2d.reshape(batch * seq_len * n_kv, head_dim), k_weight, eps)

    q_normed_4d = q_normed.reshape(batch, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
    k_normed_4d = k_normed.reshape(batch, seq_len, n_kv, head_dim).permute(0, 2, 1, 3)

    cos_3d = cos.reshape(batch, seq_len, rope_dim)
    sin_3d = sin.reshape(batch, seq_len, rope_dim)

    q_out = torch_apply_rope(q_normed_4d, cos_3d.unsqueeze(1), sin_3d.unsqueeze(1), rope_dim, head_dim)
    k_out = torch_apply_rope(k_normed_4d, cos_3d.unsqueeze(1), sin_3d.unsqueeze(1), rope_dim, head_dim)

    return q_out, k_out


@pytest.mark.parametrize("batch,seq_len,n_heads,n_kv,head_dim,rope_dim", [
    (1, 1, 16, 8, 128, 128),
    (4, 1, 16, 8, 128, 128),
    (1, 8, 16, 8, 128, 128),
    (2, 4, 16, 8, 128, 128),
    (1, 1, 8, 2, 256, 64),
    (2, 3, 8, 2, 256, 64),
])
def test_fused_norm_rope_vs_separate(glm, device, batch, seq_len, n_heads, n_kv, head_dim, rope_dim):
    q_in = torch.randn(batch, seq_len, n_heads * head_dim, dtype=torch.bfloat16, device=device)
    k_in = torch.randn(batch, seq_len, n_kv * head_dim, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)

    inv_freq = torch.randn(rope_dim // 2, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1)

    cos = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    sin = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    glm.rotary_embedding(cos, sin, inv_freq, position_ids, rope_dim // 2, batch, seq_len)

    q_out = torch.empty(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    k_out = torch.empty(batch, n_kv, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    eps = 1e-5

    glm.fused_norm_rope(q_out, q_in, q_weight, cos, sin, eps, rope_dim, head_dim, n_heads, seq_len, batch)
    glm.fused_norm_rope(k_out, k_in, k_weight, cos, sin, eps, rope_dim, head_dim, n_kv, seq_len, batch)

    ref_q, ref_k = torch_norm_rope(q_in, k_in, q_weight, k_weight, cos, sin, eps, rope_dim, head_dim, n_heads, n_kv, seq_len, batch)

    torch.testing.assert_close(q_out.cpu(), ref_q.cpu(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out.cpu(), ref_k.cpu(), atol=1e-2, rtol=1e-2)


def test_fused_norm_rope_partial_rope(glm, device):
    batch, seq_len, n_heads, n_kv = 2, 4, 8, 2
    head_dim, rope_dim = 256, 64
    eps = 1e-6

    q_in = torch.randn(batch, seq_len, n_heads * head_dim, dtype=torch.bfloat16, device=device)
    k_in = torch.randn(batch, seq_len, n_kv * head_dim, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)

    inv_freq = torch.randn(rope_dim // 2, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1)

    cos = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    sin = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    glm.rotary_embedding(cos, sin, inv_freq, position_ids, rope_dim // 2, batch, seq_len)

    q_out = torch.empty(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    k_out = torch.empty(batch, n_kv, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    glm.fused_norm_rope(q_out, q_in, q_weight, cos, sin, eps, rope_dim, head_dim, n_heads, seq_len, batch)
    glm.fused_norm_rope(k_out, k_in, k_weight, cos, sin, eps, rope_dim, head_dim, n_kv, seq_len, batch)

    ref_q, ref_k = torch_norm_rope(q_in, k_in, q_weight, k_weight, cos, sin, eps, rope_dim, head_dim, n_heads, n_kv, seq_len, batch)

    torch.testing.assert_close(q_out.cpu(), ref_q.cpu(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out.cpu(), ref_k.cpu(), atol=1e-2, rtol=1e-2)

    non_rope_q = q_out[..., rope_dim:]
    ref_non_rope_q = ref_q[..., rope_dim:]
    torch.testing.assert_close(non_rope_q.cpu(), ref_non_rope_q.cpu(), atol=1e-2, rtol=1e-2)
