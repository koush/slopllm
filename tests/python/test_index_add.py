import torch
import pytest


def test_index_add_basic(glm, device):
    n_indices = 4
    dim = 8
    out = torch.zeros(10, dim, dtype=torch.bfloat16, device=device)
    values = torch.randn(n_indices, dim, dtype=torch.bfloat16, device=device)
    indices = torch.tensor([0, 2, 5, 7], device=device, dtype=torch.int32)

    glm.index_add(out, indices, values, n_indices, dim)

    ref = torch.zeros(10, dim, dtype=torch.bfloat16, device=device)
    ref.index_add_(0, indices, values)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3)


def test_index_add_unique_indices(glm, device):
    dim = 16
    out = torch.randn(8, dim, dtype=torch.bfloat16, device=device)
    values = torch.randn(3, dim, dtype=torch.bfloat16, device=device)
    indices = torch.tensor([1, 4, 6], device=device, dtype=torch.int32)

    ref = out.clone()
    glm.index_add(out, indices, values, 3, dim)
    ref.index_add_(0, indices, values)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3)


def test_index_add_single_index(glm, device):
    dim = 4
    out = torch.zeros(5, dim, dtype=torch.bfloat16, device=device)
    values = torch.randn(1, dim, dtype=torch.bfloat16, device=device)
    indices = torch.tensor([3], device=device, dtype=torch.int32)

    glm.index_add(out, indices, values, 1, dim)

    ref = torch.zeros(5, dim, dtype=torch.bfloat16, device=device)
    ref[3] = values[0]
    assert torch.equal(out.cpu(), ref.cpu())


def test_index_add_large_dim(glm, device):
    n_indices = 8
    dim = 256
    out = torch.randn(32, dim, dtype=torch.bfloat16, device=device)
    values = torch.randn(n_indices, dim, dtype=torch.bfloat16, device=device)
    indices = torch.randperm(32, device=device, dtype=torch.int32)[:n_indices]

    ref = out.clone()
    glm.index_add(out, indices, values, n_indices, dim)
    ref.index_add_(0, indices, values)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3)
