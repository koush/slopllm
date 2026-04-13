import torch
import pytest


def test_masked_fill_basic(glm, device):
    n = 32
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    mask = torch.zeros(n, dtype=torch.bfloat16, device=device)
    mask[::2] = 1.0
    out = torch.empty_like(x)

    glm.masked_fill(out, x, mask, 0.0, n)

    ref = x.clone()
    ref[mask.to(torch.bool)] = 0.0
    assert torch.equal(out.cpu(), ref.cpu())


def test_masked_fill_neg_inf(glm, device):
    n = 16
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    mask = torch.zeros(n, dtype=torch.bfloat16, device=device)
    mask[5:10] = 1.0
    out = torch.empty_like(x)

    glm.masked_fill(out, x, mask, float('-inf'), n)

    ref = x.clone()
    ref[mask.to(torch.bool)] = float('-inf')
    for i in range(n):
        if mask[i].item() != 0:
            assert out[i].isinf() and out[i].item() < 0
        else:
            assert torch.equal(out[i:i+1].cpu(), ref[i:i+1].cpu())


def test_masked_fill_2d(glm, device):
    batch, dim = 4, 16
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    mask = torch.zeros(batch, dim, dtype=torch.bfloat16, device=device)
    mask[:, :8] = 1.0
    out = torch.empty_like(x)

    n = batch * dim
    glm.masked_fill(out, x, mask, -100.0, n)

    ref = x.clone()
    ref[mask.to(torch.bool)] = -100.0
    assert torch.equal(out.cpu(), ref.cpu())


def test_masked_fill_no_mask(glm, device):
    n = 16
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    mask = torch.zeros(n, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    glm.masked_fill(out, x, mask, 999.0, n)
    assert torch.equal(out.cpu(), x.cpu())


def test_masked_fill_all_masked(glm, device):
    n = 16
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    mask = torch.ones(n, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    glm.masked_fill(out, x, mask, 42.0, n)
    expected = torch.full((n,), 42.0, dtype=torch.bfloat16, device=device)
    assert torch.equal(out.cpu(), expected.cpu())
