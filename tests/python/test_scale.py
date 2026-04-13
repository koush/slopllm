import torch
import pytest


def test_scale_random(glm, device):
    n = 256
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.scale(out, x, 0.125, n)
    expected = x * 0.125
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)


def test_scale_one(glm, device):
    n = 128
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.scale(out, x, 1.0, n)
    assert torch.equal(out.cpu(), x.cpu())


def test_scale_zero(glm, device):
    n = 64
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.scale(out, x, 0.0, n)
    expected = torch.zeros(n, dtype=torch.bfloat16, device=device)
    assert torch.equal(out.cpu(), expected.cpu())


def test_scale_negative(glm, device):
    n = 128
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.scale(out, x, -2.0, n)
    expected = x * (-2.0)
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)


def test_scale_attention_head_dim(glm, device):
    n = 128 * 8 * 64
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    scale = 1.0 / (64 ** 0.5)
    glm.scale(out, x, scale, n)
    expected = x * scale
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)
