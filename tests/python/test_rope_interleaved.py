import torch
import pytest


@pytest.mark.parametrize("head_dim,n_heads", [(64, 8), (128, 4)])
@pytest.mark.parametrize("batch,seq_len", [(1, 4), (2, 5), (1, 512)])
@pytest.mark.parametrize("padding", [0, 32])
@pytest.mark.parametrize("interleaved", [False, True])
def test_apply_rotary_pos_emb_glm_shapes(glm, device, head_dim, n_heads,
                                       batch, seq_len, padding, interleaved):
    rope_dim = 64
    in_stride = head_dim + padding
    cos, sin = _make_rope((batch, seq_len, rope_dim), rope_dim // 2,
                         seq_len, batch, device)
    x = torch.randn(batch, seq_len, n_heads, in_stride,
                    dtype=torch.bfloat16, device=device)
    out = torch.empty(batch, seq_len, n_heads, head_dim,
                      dtype=torch.bfloat16, device=device)
    glm.apply_rotary_pos_emb(out, x, cos, sin, rope_dim, head_dim,
                            n_heads, seq_len, batch, 2,
                            interleaved=interleaved, in_stride=in_stride)
    ref = x[..., :head_dim].clone()
    rotate = (_ref_apply_rotary_pos_emb_interleaved if interleaved
              else _ref_apply_rotary_pos_emb_neox)
    ref[..., :rope_dim] = rotate(x[..., :rope_dim], cos, sin, unsqueeze_dim=2)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(out[..., rope_dim:], x[..., rope_dim:head_dim],
                               atol=0, rtol=0)


def _ref_apply_rotary_pos_emb_neox(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x.float() * cos.float() + rotated.float() * sin.float()).to(torch.bfloat16)


def _ref_apply_rotary_pos_emb_interleaved(x, cos, sin, unsqueeze_dim=1):
    dim_half = cos.shape[-1] // 2
    cos_half = cos[..., :dim_half].unsqueeze(unsqueeze_dim)
    sin_half = sin[..., :dim_half].unsqueeze(unsqueeze_dim)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    o1 = x1.float() * cos_half.float() - x2.float() * sin_half.float()
    o2 = x2.float() * cos_half.float() + x1.float() * sin_half.float()
    return torch.stack((o1, o2), dim=-1).flatten(-2).to(torch.bfloat16)


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


def test_apply_rotary_pos_emb_interleaved_basic(glm, device):
    batch, n_heads, seq_len, rope_dim = 2, 4, 8, 16
    dim_half = rope_dim // 2
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device)

    x = torch.randn(batch, n_heads, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    x_flat = x.reshape(batch * n_heads, seq_len, rope_dim)
    out_flat = out.reshape(batch * n_heads, seq_len, rope_dim)

    glm.apply_rotary_pos_emb(out_flat, x_flat, cos_emb, sin_emb, rope_dim, rope_dim, n_heads, seq_len, batch, 1, interleaved=True)

    ref = _ref_apply_rotary_pos_emb_interleaved(x.cpu(), cos_emb.cpu(), sin_emb.cpu(), unsqueeze_dim=1)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=2e-3)


def test_apply_rotary_pos_emb_interleaved_vs_neox_differ(glm, device):
    batch, n_heads, seq_len, rope_dim = 1, 2, 4, 16
    dim_half = rope_dim // 2
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device)

    x = torch.randn(batch, n_heads, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    out_neox = torch.empty_like(x)
    out_interleaved = torch.empty_like(x)

    x_flat = x.reshape(batch * n_heads, seq_len, rope_dim)

    glm.apply_rotary_pos_emb(out_neox.reshape(batch * n_heads, seq_len, rope_dim), x_flat, cos_emb, sin_emb, rope_dim, rope_dim, n_heads, seq_len, batch, 1, interleaved=False)
    glm.apply_rotary_pos_emb(out_interleaved.reshape(batch * n_heads, seq_len, rope_dim), x_flat, cos_emb, sin_emb, rope_dim, rope_dim, n_heads, seq_len, batch, 1, interleaved=True)

    assert not torch.equal(out_neox.cpu(), out_interleaved.cpu()), "Interleaved and non-interleaved RoPE should produce different outputs"


def test_apply_rotary_pos_emb_interleaved_position_zero(glm, device):
    batch, n_heads, seq_len, rope_dim = 1, 2, 4, 8
    dim_half = rope_dim // 2
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device)

    x = torch.randn(batch, n_heads, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    x_flat = x.reshape(batch * n_heads, seq_len, rope_dim)
    out_flat = out.reshape(batch * n_heads, seq_len, rope_dim)

    glm.apply_rotary_pos_emb(out_flat, x_flat, cos_emb, sin_emb, rope_dim, rope_dim, n_heads, seq_len, batch, 1, interleaved=True)

    ref = _ref_apply_rotary_pos_emb_interleaved(x.cpu(), cos_emb.cpu(), sin_emb.cpu(), unsqueeze_dim=1)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=2e-3)

    cos_pos0 = cos_emb[0, 0, :].cpu()
    assert (cos_pos0 == 1.0).all(), "cos(0) should be 1.0"


def test_apply_rotary_pos_emb_partial_interleaved(glm, device):
    batch, n_heads, seq_len, head_dim, rope_dim = 2, 4, 8, 64, 16
    dim_half = rope_dim // 2
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device)

    x = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    x_flat = x.reshape(batch * n_heads, seq_len, head_dim)
    out_flat = out.reshape(batch * n_heads, seq_len, head_dim)

    glm.apply_rotary_pos_emb(out_flat, x_flat, cos_emb, sin_emb,
                                       rope_dim, head_dim, n_heads, seq_len, batch, 1, interleaved=True)

    x_rot = x[..., :rope_dim]
    cos_half = cos_emb[..., :dim_half].unsqueeze(1)
    sin_half = sin_emb[..., :dim_half].unsqueeze(1)
    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    o1 = x1.float() * cos_half.float() - x2.float() * sin_half.float()
    o2 = x2.float() * cos_half.float() + x1.float() * sin_half.float()
    ref_rot = torch.stack((o1, o2), dim=-1).flatten(-2).to(torch.bfloat16)
    ref = x.clone()
    ref[..., :rope_dim] = ref_rot
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=2e-3, rtol=2e-3)

    nope_dims = out[..., rope_dim:].cpu()
    ref_nope = x[..., rope_dim:].cpu()
    assert torch.equal(nope_dims, ref_nope), "Non-RoPE dims should be unchanged"


def test_apply_rotary_pos_emb_interleaved_token_major(glm, device):
    batch, n_heads, seq_len = 2, 4, 8
    rope_dim = 32
    head_dim = 32
    dim_half = rope_dim // 2
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device)

    x = torch.randn(batch * seq_len, n_heads, head_dim, dtype=torch.bfloat16, device=device)

    out_cuda = torch.empty_like(x)
    glm.apply_rotary_pos_emb(out_cuda, x, cos_emb, sin_emb, rope_dim, head_dim, n_heads, seq_len, batch, 2, interleaved=True)

    x4 = x.reshape(batch, seq_len, n_heads, head_dim)
    ref = _ref_apply_rotary_pos_emb_interleaved(x4.cpu(), cos_emb.cpu(), sin_emb.cpu(), unsqueeze_dim=2)
    torch.testing.assert_close(out_cuda.reshape(batch, seq_len, n_heads, head_dim).cpu(), ref, atol=2e-3, rtol=2e-3)


def test_apply_rotary_pos_emb_interleaved_glm51_dims(glm, device):
    batch, n_heads, seq_len, rope_dim = 1, 128, 16, 64
    dim_half = rope_dim // 2
    theta = 1_000_000.0
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device, theta=theta)

    x = torch.randn(batch, n_heads, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    x_flat = x.reshape(batch * n_heads, seq_len, rope_dim)
    out_flat = out.reshape(batch * n_heads, seq_len, rope_dim)

    glm.apply_rotary_pos_emb(out_flat, x_flat, cos_emb, sin_emb, rope_dim, rope_dim, n_heads, seq_len, batch, 1, interleaved=True)

    ref = _ref_apply_rotary_pos_emb_interleaved(x.cpu(), cos_emb.cpu(), sin_emb.cpu(), unsqueeze_dim=1)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=2e-3)


def test_fused_norm_rope_interleaved(glm, device):
    batch, n_heads, seq_len = 1, 4, 8
    rope_dim = 32
    head_dim = 64
    in_stride = head_dim
    eps = 1e-5

    inv_freq = torch.zeros(rope_dim // 2, dtype=torch.bfloat16, device=device)
    for i in range(rope_dim // 2):
        inv_freq[i] = 1.0 / (10000.0 ** (2.0 * i / rope_dim))
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    cos_out = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    sin_out = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    glm.rotary_embedding(cos_out, sin_out, inv_freq, position_ids, rope_dim // 2, batch, seq_len)

    x = torch.randn(batch * seq_len, n_heads * in_stride, dtype=torch.bfloat16, device=device)
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)

    out_cuda = torch.empty(batch * n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    glm.fused_norm_rope(out_cuda, x, weight, cos_out, sin_out, eps, rope_dim, head_dim, n_heads, seq_len, batch, in_stride, interleaved=True)

    x_reshaped = x.reshape(batch, seq_len, n_heads, head_dim)
    normed = torch.zeros(batch, seq_len, n_heads, head_dim, dtype=torch.float32, device=device)
    for b in range(batch):
        for s in range(seq_len):
            for h in range(n_heads):
                vec = x_reshaped[b, s, h].float()
                rms = torch.rsqrt(vec.pow(2).mean() + eps)
                normed[b, s, h] = vec * rms * weight.float()

    dim_half = rope_dim // 2
    cos_half = cos_out[..., :dim_half]
    sin_half = sin_out[..., :dim_half]
    normed_rot = normed[..., :rope_dim]
    normed_pass = normed[..., rope_dim:]
    n1 = normed_rot[..., 0::2]
    n2 = normed_rot[..., 1::2]
    cos_exp = cos_half.unsqueeze(2).expand_as(n1)
    sin_exp = sin_half.unsqueeze(2).expand_as(n1)
    o1 = n1 * cos_exp - n2 * sin_exp
    o2 = n2 * cos_exp + n1 * sin_exp
    ref_rot = torch.stack((o1, o2), dim=-1).flatten(-2)
    ref = torch.cat([ref_rot, normed_pass], dim=-1).to(torch.bfloat16)
    ref_perm = ref.permute(0, 2, 1, 3).reshape(batch * n_heads, seq_len, head_dim)

    torch.testing.assert_close(out_cuda.cpu(), ref_perm.cpu(), atol=5e-3, rtol=5e-3)


def test_neox_still_works_after_interleaved_change(glm, device):
    batch, n_heads, seq_len, rope_dim = 2, 4, 8, 16
    dim_half = rope_dim // 2
    cos_emb, sin_emb = _make_rope((batch, seq_len, rope_dim), dim_half, seq_len, batch, device)

    x = torch.randn(batch, n_heads, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    x_flat = x.reshape(batch * n_heads, seq_len, rope_dim)
    out_flat = out.reshape(batch * n_heads, seq_len, rope_dim)

    glm.apply_rotary_pos_emb(out_flat, x_flat, cos_emb, sin_emb, rope_dim, rope_dim, n_heads, seq_len, batch, 1, interleaved=False)

    ref = _ref_apply_rotary_pos_emb_neox(x.cpu(), cos_emb.cpu(), sin_emb.cpu(), unsqueeze_dim=1)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=2e-3)
