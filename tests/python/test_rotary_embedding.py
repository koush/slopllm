import torch
import pytest
import math


def _ref_rotary_embedding(inv_freq, position_ids, dim_half):
    batch, seq_len = position_ids.shape
    inv = inv_freq[None, :, None].float().expand(batch, -1, 1)
    pos = position_ids[:, None, :].float()
    freqs = (inv @ pos).transpose(1, 2)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def test_rotary_embedding_basic(glm, device):
    dim_half = 16
    batch = 2
    seq_len = 8
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half * 2, 2, dtype=torch.float)[:dim_half].float() / (dim_half * 2)))
    inv_freq_bf = inv_freq.to(torch.bfloat16).to(device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1)

    dim = dim_half * 2
    cos_out = torch.empty(batch, seq_len, dim, dtype=torch.bfloat16, device=device)
    sin_out = torch.empty(batch, seq_len, dim, dtype=torch.bfloat16, device=device)

    glm.rotary_embedding(cos_out, sin_out, inv_freq_bf, position_ids, dim_half, batch, seq_len)

    ref_cos, ref_sin = _ref_rotary_embedding(inv_freq_bf.cpu(), position_ids.cpu(), dim_half)
    torch.testing.assert_close(cos_out.cpu(), ref_cos, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(sin_out.cpu(), ref_sin, atol=2e-3, rtol=2e-3)


def test_rotary_embedding_single_token(glm, device):
    dim_half = 8
    batch = 1
    seq_len = 1
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half * 2, 2, dtype=torch.float)[:dim_half].float() / (dim_half * 2)))
    inv_freq_bf = inv_freq.to(torch.bfloat16).to(device)
    position_ids = torch.tensor([[0]], dtype=torch.long, device=device)

    dim = dim_half * 2
    cos_out = torch.empty(batch, seq_len, dim, dtype=torch.bfloat16, device=device)
    sin_out = torch.empty(batch, seq_len, dim, dtype=torch.bfloat16, device=device)

    glm.rotary_embedding(cos_out, sin_out, inv_freq_bf, position_ids, dim_half, batch, seq_len)

    ref_cos, ref_sin = _ref_rotary_embedding(inv_freq_bf.cpu(), position_ids.cpu(), dim_half)
    torch.testing.assert_close(cos_out.cpu(), ref_cos, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(sin_out.cpu(), ref_sin, atol=2e-3, rtol=2e-3)

    assert (cos_out[0, 0, :] == 1.0).all(), "cos(0) should be 1"
    assert (sin_out[0, 0, :] == 0.0).all(), "sin(0) should be 0"


def test_rotary_embedding_glm51_dims(glm, device):
    dim_half = 32
    batch = 1
    seq_len = 16
    theta = 1_000_000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half * 2, 2, dtype=torch.float)[:dim_half].float() / (dim_half * 2)))
    inv_freq_bf = inv_freq.to(torch.bfloat16).to(device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    dim = dim_half * 2
    cos_out = torch.empty(batch, seq_len, dim, dtype=torch.bfloat16, device=device)
    sin_out = torch.empty(batch, seq_len, dim, dtype=torch.bfloat16, device=device)

    glm.rotary_embedding(cos_out, sin_out, inv_freq_bf, position_ids, dim_half, batch, seq_len)

    ref_cos, ref_sin = _ref_rotary_embedding(inv_freq_bf.cpu(), position_ids.cpu(), dim_half)
    torch.testing.assert_close(cos_out.cpu(), ref_cos, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(sin_out.cpu(), ref_sin, atol=2e-3, rtol=2e-3)
