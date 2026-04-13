import torch
import pytest


def _expand_ref(x, dim1_out):
    return x.repeat(1, dim1_out // x.shape[1], 1, 1).contiguous()


def test_expand_dim1_basic(glm, device):
    batch, dim1_in, seq_len, head_dim = 1, 1, 4, 8
    dim1_out = 4
    x = torch.randn(batch, dim1_in, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(batch, dim1_out, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    glm.expand_dim1(out, x, dim1_out, dim1_in, seq_len, head_dim, batch)
    expected = _expand_ref(x, dim1_out)
    assert torch.equal(out.cpu(), expected.cpu())


def test_expand_dim1_no_repeat(glm, device):
    batch, dim1_in, seq_len, head_dim = 1, 4, 4, 8
    dim1_out = 4
    x = torch.randn(batch, dim1_in, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(batch, dim1_out, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    glm.expand_dim1(out, x, dim1_out, dim1_in, seq_len, head_dim, batch)
    assert torch.equal(out.cpu(), x.cpu())


def test_expand_dim1_gqa_style(glm, device):
    batch, n_kv_heads, seq_len, head_dim = 1, 1, 8, 16
    n_heads = 4
    k_pe = torch.randn(batch, n_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    glm.expand_dim1(out, k_pe, n_heads, n_kv_heads, seq_len, head_dim, batch)
    expected = k_pe.expand(batch, n_heads, seq_len, head_dim).contiguous()
    assert torch.equal(out.cpu(), expected.cpu())


def test_expand_dim1_batch2(glm, device):
    batch, dim1_in, seq_len, head_dim = 2, 1, 4, 8
    dim1_out = 4
    x = torch.randn(batch, dim1_in, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(batch, dim1_out, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    glm.expand_dim1(out, x, dim1_out, dim1_in, seq_len, head_dim, batch)
    expected = _expand_ref(x, dim1_out)
    assert torch.equal(out.cpu(), expected.cpu())


def test_expand_dim1_large_repeat(glm, device):
    batch, dim1_in, seq_len, head_dim = 1, 1, 4, 16
    dim1_out = 128
    x = torch.randn(batch, dim1_in, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(batch, dim1_out, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    glm.expand_dim1(out, x, dim1_out, dim1_in, seq_len, head_dim, batch)
    expected = x.expand(batch, dim1_out, seq_len, head_dim).contiguous()
    assert torch.equal(out.cpu(), expected.cpu())
