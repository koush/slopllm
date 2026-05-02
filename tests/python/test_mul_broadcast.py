import torch
import pytest


def test_mul_broadcast_basic(glm, device):
    rows, dim = 8, 256
    a = torch.randn(rows, dim, dtype=torch.bfloat16, device=device)
    b = torch.randn(dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dim, dtype=torch.bfloat16, device=device)
    glm.mul_broadcast(out, a, b, dim, rows)
    expected = a * b.unsqueeze(0)
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)


def test_mul_broadcast_ones(glm, device):
    rows, dim = 4, 128
    a = torch.randn(rows, dim, dtype=torch.bfloat16, device=device)
    b = torch.ones(dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dim, dtype=torch.bfloat16, device=device)
    glm.mul_broadcast(out, a, b, dim, rows)
    assert torch.equal(out.cpu(), a.cpu())


def test_mul_broadcast_zeros(glm, device):
    rows, dim = 6, 64
    a = torch.randn(rows, dim, dtype=torch.bfloat16, device=device)
    b = torch.zeros(dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dim, dtype=torch.bfloat16, device=device)
    glm.mul_broadcast(out, a, b, dim, rows)
    assert torch.equal(out.cpu(), torch.zeros(rows, dim, dtype=torch.bfloat16))
