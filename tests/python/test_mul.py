import torch
import pytest


def test_mul_random(glm, device):
    n = 256
    a = torch.randn(n, dtype=torch.bfloat16, device=device)
    b = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.mul(out, a, b, n)
    ref = (a * b).cpu()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-3)


def test_mul_zeros(glm, device):
    n = 128
    a = torch.randn(n, dtype=torch.bfloat16, device=device)
    b = torch.zeros(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.mul(out, a, b, n)
    assert torch.equal(out.cpu(), torch.zeros(n, dtype=torch.bfloat16))


def test_mul_ones(glm, device):
    n = 64
    a = torch.randn(n, dtype=torch.bfloat16, device=device)
    b = torch.ones(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.mul(out, a, b, n)
    assert torch.equal(out.cpu(), a.cpu())


def test_mul_2d(glm, device):
    rows, cols = 4, 32
    a = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
    b = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(a)
    glm.mul(out.reshape(-1), a.reshape(-1), b.reshape(-1), rows * cols)
    ref = (a * b).cpu()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-3)


def test_mul_scale_like(glm, device):
    n = 1024
    a = torch.randn(n, dtype=torch.bfloat16, device=device)
    scale_val = 0.5
    b = torch.full((n,), scale_val, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.mul(out, a, b, n)
    ref = (a * b).cpu()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-3)
