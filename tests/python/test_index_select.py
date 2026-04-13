import torch
import pytest


def test_index_select_basic(glm, device):
    src_rows, dim = 8, 16
    src = torch.randn(src_rows, dim, dtype=torch.bfloat16, device=device)
    indices = torch.tensor([0, 3, 5, 7], dtype=torch.int32, device=device)
    k = indices.shape[0]
    out = torch.empty(k, dim, dtype=torch.bfloat16, device=device)
    glm.index_select(out.reshape(-1), src.reshape(-1), indices, dim, k)
    ref = src[indices.long()]
    assert torch.equal(out.cpu(), ref.cpu())


def test_index_select_single(glm, device):
    src_rows, dim = 4, 32
    src = torch.randn(src_rows, dim, dtype=torch.bfloat16, device=device)
    indices = torch.tensor([2], dtype=torch.int32, device=device)
    k = 1
    out = torch.empty(k, dim, dtype=torch.bfloat16, device=device)
    glm.index_select(out.reshape(-1), src.reshape(-1), indices, dim, k)
    ref = src[indices.long()]
    assert torch.equal(out.cpu(), ref.cpu())


def test_index_select_duplicate(glm, device):
    src_rows, dim = 6, 8
    src = torch.randn(src_rows, dim, dtype=torch.bfloat16, device=device)
    indices = torch.tensor([1, 1, 3, 3, 5], dtype=torch.int32, device=device)
    k = indices.shape[0]
    out = torch.empty(k, dim, dtype=torch.bfloat16, device=device)
    glm.index_select(out.reshape(-1), src.reshape(-1), indices, dim, k)
    ref = src[indices.long()]
    assert torch.equal(out.cpu(), ref.cpu())


def test_index_select_all_rows(glm, device):
    src_rows, dim = 4, 16
    src = torch.randn(src_rows, dim, dtype=torch.bfloat16, device=device)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    k = src_rows
    out = torch.empty(k, dim, dtype=torch.bfloat16, device=device)
    glm.index_select(out.reshape(-1), src.reshape(-1), indices, dim, k)
    assert torch.equal(out.cpu(), src.cpu())


def test_index_select_moe_style(glm, device):
    num_tokens, hidden = 16, 64
    src = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    token_idx = torch.tensor([0, 2, 5, 7, 11], dtype=torch.int32, device=device)
    k = token_idx.shape[0]
    out = torch.empty(k, hidden, dtype=torch.bfloat16, device=device)
    glm.index_select(out.reshape(-1), src.reshape(-1), token_idx, hidden, k)
    ref = src[token_idx.long()]
    assert torch.equal(out.cpu(), ref.cpu())
