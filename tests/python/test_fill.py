import torch
import pytest


def test_fill_positive(glm, device):
    n = 256
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.fill(out, 3.14, n)
    expected = torch.full((n,), 3.14, dtype=torch.bfloat16, device=device)
    assert torch.equal(out.cpu(), expected.cpu())


def test_fill_zero(glm, device):
    n = 128
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.fill(out, 0.0, n)
    expected = torch.zeros(n, dtype=torch.bfloat16, device=device)
    assert torch.equal(out.cpu(), expected.cpu())


def test_fill_negative(glm, device):
    n = 64
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.fill(out, -42.5, n)
    expected = torch.full((n,), -42.5, dtype=torch.bfloat16, device=device)
    assert torch.equal(out.cpu(), expected.cpu())


def test_fill_neg_inf(glm, device):
    n = 32
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.fill(out, float('-inf'), n)
    expected = torch.full((n,), float('-inf'), dtype=torch.bfloat16, device=device)
    assert (out == expected).all(), "All values should be -inf"


def test_fill_large(glm, device):
    n = 4096
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.fill(out, 1.0, n)
    expected = torch.ones(n, dtype=torch.bfloat16, device=device)
    assert torch.equal(out.cpu(), expected.cpu())
