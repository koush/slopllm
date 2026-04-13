import torch
import pytest


def test_add_random(glm, device):
    n = 256
    a = torch.randn(n, dtype=torch.bfloat16, device=device)
    b = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.add(out, a, b, n)
    expected = a + b
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)


def test_add_zeros(glm, device):
    n = 128
    a = torch.randn(n, dtype=torch.bfloat16, device=device)
    b = torch.zeros(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.add(out, a, b, n)
    assert torch.equal(out.cpu(), a.cpu())


def test_add_negative(glm, device):
    n = 64
    a = torch.randn(n, dtype=torch.bfloat16, device=device)
    b = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.add(out, a, b, n)
    expected = a + b
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)


def test_add_large(glm, device):
    n = 4096
    a = torch.randn(n, dtype=torch.bfloat16, device=device)
    b = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.add(out, a, b, n)
    expected = a + b
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)


def test_add_attention_mask(glm, device):
    batch, heads, seq = 2, 4, 8
    n = batch * heads * seq * seq
    attn = torch.randn(batch * heads, seq, seq, dtype=torch.bfloat16, device=device).reshape(n)
    mask = torch.zeros(batch * heads, seq, seq, dtype=torch.bfloat16, device=device).reshape(n)
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    glm.add(out, attn, mask, n)
    assert torch.equal(out.cpu(), attn.cpu())
