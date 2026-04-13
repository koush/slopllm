import torch
import pytest


def _ref_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x.float() * cos.float() + rotated.float() * sin.float()).to(torch.bfloat16)


def _make_rope(cos_sin_shape, dim_half, seq_len, batch, device, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half * 2, 2, dtype=torch.float)[:dim_half].float() / (dim_half * 2)))
    inv_freq_bf = inv_freq.to(torch.bfloat16).to(device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1)
    inv = inv_freq[None, :, None].float().to(device).expand(batch, -1, 1)
    pos = position_ids[:, None, :].float()
    freqs = (inv @ pos).transpose(1, 2)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_emb = emb.cos().to(torch.bfloat16).to(device)
    sin_emb = emb.sin().to(torch.bfloat16).to(device)
    return cos_emb, sin_emb


def test_apply_rotary_pos_emb_unsqueeze1(glm, device):
    batch, n_heads, seq_len, rope_dim = 2, 4, 8, 16
    dim_half = rope_dim // 2
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device)

    x = torch.randn(batch, n_heads, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    x_flat = x.reshape(batch * n_heads, seq_len, rope_dim)
    out_flat = out.reshape(batch * n_heads, seq_len, rope_dim)

    glm.apply_rotary_pos_emb(out_flat, x_flat, cos_emb, sin_emb, rope_dim, n_heads, seq_len, batch, 1)

    ref = _ref_apply_rotary_pos_emb(x.cpu(), cos_emb.cpu(), sin_emb.cpu(), unsqueeze_dim=1)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=2e-3)


def test_apply_rotary_pos_emb_unsqueeze2(glm, device):
    batch, seq_len, n_heads, rope_dim = 2, 8, 4, 16
    dim_half = rope_dim // 2
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device)

    x = torch.randn(batch, seq_len, n_heads, rope_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    x_flat = x.reshape(batch * seq_len, n_heads, rope_dim)
    out_flat = out.reshape(batch * seq_len, n_heads, rope_dim)

    glm.apply_rotary_pos_emb(out_flat, x_flat, cos_emb, sin_emb, rope_dim, n_heads, seq_len, batch, 2)

    ref = _ref_apply_rotary_pos_emb(x.cpu(), cos_emb.cpu(), sin_emb.cpu(), unsqueeze_dim=2)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=2e-3)


def test_apply_rotary_pos_emb_position_zero(glm, device):
    batch, n_heads, seq_len, rope_dim = 1, 2, 4, 8
    dim_half = rope_dim // 2
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device)

    x = torch.randn(batch, n_heads, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    x_flat = x.reshape(batch * n_heads, seq_len, rope_dim)
    out_flat = out.reshape(batch * n_heads, seq_len, rope_dim)

    glm.apply_rotary_pos_emb(out_flat, x_flat, cos_emb, sin_emb, rope_dim, n_heads, seq_len, batch, 1)

    ref = _ref_apply_rotary_pos_emb(x.cpu(), cos_emb.cpu(), sin_emb.cpu(), unsqueeze_dim=1)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=2e-3)

    cos_pos0 = cos_emb[0, 0, :].cpu()
    assert (cos_pos0 == 1.0).all(), "cos(0) should be 1.0"


def test_apply_rotary_pos_emb_glm51_dims(glm, device):
    batch, n_heads, seq_len, rope_dim = 1, 128, 16, 64
    dim_half = rope_dim // 2
    theta = 1_000_000.0
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device, theta=theta)

    x = torch.randn(batch, n_heads, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    x_flat = x.reshape(batch * n_heads, seq_len, rope_dim)
    out_flat = out.reshape(batch * n_heads, seq_len, rope_dim)

    glm.apply_rotary_pos_emb(out_flat, x_flat, cos_emb, sin_emb, rope_dim, n_heads, seq_len, batch, 1)

    ref = _ref_apply_rotary_pos_emb(x.cpu(), cos_emb.cpu(), sin_emb.cpu(), unsqueeze_dim=1)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=2e-3)
