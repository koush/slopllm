import torch
import pytest


def test_reduce_sum_basic(glm, device):
    rows, cols = 4, 32
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dtype=torch.bfloat16, device=device)
    glm.reduce_sum(out, x.reshape(-1), rows, cols)
    ref = x.sum(dim=-1).to(torch.bfloat16).cpu()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-3)


def test_reduce_sum_single_row(glm, device):
    rows, cols = 1, 64
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dtype=torch.bfloat16, device=device)
    glm.reduce_sum(out, x.reshape(-1), rows, cols)
    ref = x.sum(dim=-1).to(torch.bfloat16).cpu()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-3)


def test_reduce_sum_small_cols(glm, device):
    rows, cols = 8, 2
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dtype=torch.bfloat16, device=device)
    glm.reduce_sum(out, x.reshape(-1), rows, cols)
    ref = x.sum(dim=-1).to(torch.bfloat16).cpu()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-3)


def test_reduce_sum_zeros(glm, device):
    rows, cols = 4, 16
    x = torch.zeros(rows, cols, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, dtype=torch.bfloat16, device=device)
    glm.reduce_sum(out, x.reshape(-1), rows, cols)
    assert torch.equal(out.cpu(), torch.zeros(rows, dtype=torch.bfloat16))


def test_reduce_sum_positive(glm, device):
    rows, cols = 3, 8
    x = torch.rand(rows, cols, dtype=torch.bfloat16, device=device) + 0.5
    out = torch.empty(rows, dtype=torch.bfloat16, device=device)
    glm.reduce_sum(out, x.reshape(-1), rows, cols)
    ref = x.sum(dim=-1).to(torch.bfloat16).cpu()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-3)


def test_reduce_sum_moe_weights(glm, device):
    topk = 8
    num_tokens = 16
    topk_w = torch.rand(num_tokens, topk, dtype=torch.bfloat16, device=device)
    out = torch.empty(num_tokens, dtype=torch.bfloat16, device=device)
    glm.reduce_sum(out, topk_w.reshape(-1), num_tokens, topk)
    ref = topk_w.sum(dim=-1).to(torch.bfloat16).cpu()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-3, rtol=1e-3)
