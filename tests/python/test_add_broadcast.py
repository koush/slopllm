import torch
import pytest


def test_add_broadcast_basic(glm, device):
    rows, dim = 8, 256
    a = torch.randn(rows, dim, dtype=torch.bfloat16, device=device)
    b = torch.randn(dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dim, dtype=torch.bfloat16, device=device)
    glm.add_broadcast(out, a, b, dim, rows)
    expected = a + b.unsqueeze(0)
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)


def test_add_broadcast_single_row(glm, device):
    rows, dim = 1, 128
    a = torch.randn(rows, dim, dtype=torch.bfloat16, device=device)
    b = torch.randn(dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dim, dtype=torch.bfloat16, device=device)
    glm.add_broadcast(out, a, b, dim, rows)
    expected = a + b.unsqueeze(0)
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)


def test_add_broadcast_large(glm, device):
    rows, dim = 32, 2048
    a = torch.randn(rows, dim, dtype=torch.bfloat16, device=device)
    b = torch.randn(dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dim, dtype=torch.bfloat16, device=device)
    glm.add_broadcast(out, a, b, dim, rows)
    expected = a + b.unsqueeze(0)
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)


def test_add_broadcast_zeros_bias(glm, device):
    rows, dim = 6, 128
    a = torch.randn(rows, dim, dtype=torch.bfloat16, device=device)
    b = torch.zeros(dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dim, dtype=torch.bfloat16, device=device)
    glm.add_broadcast(out, a, b, dim, rows)
    assert torch.equal(out.cpu(), a.cpu())


def test_add_broadcast_moe_bias(glm, device):
    rows, num_experts = 8, 256
    gate = torch.randn(rows, num_experts, dtype=torch.bfloat16, device=device)
    bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, num_experts, dtype=torch.bfloat16, device=device)
    glm.add_broadcast(out, gate, bias, num_experts, rows)
    expected = gate + bias.unsqueeze(0)
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)
