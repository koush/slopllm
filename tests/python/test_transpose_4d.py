import torch
import pytest


def test_transpose_4d_swap_1_2(glm, device):
    B, S, H, D = 2, 4, 8, 16
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    out = torch.empty(B, H, S, D, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(out, x, B, S, H, D, 0, 2, 1, 3)
    expected = x.transpose(1, 2).contiguous()
    assert torch.equal(out.cpu(), expected.cpu())


def test_transpose_4d_identity(glm, device):
    B, S, H, D = 1, 4, 8, 16
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    out = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(out, x, B, S, H, D, 0, 1, 2, 3)
    assert torch.equal(out.cpu(), x.cpu())


def test_transpose_4d_swap_0_1(glm, device):
    B, S, H, D = 2, 3, 4, 8
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    out = torch.empty(S, B, H, D, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(out, x, B, S, H, D, 1, 0, 2, 3)
    expected = x.permute(1, 0, 2, 3).contiguous()
    assert torch.equal(out.cpu(), expected.cpu())


def test_transpose_4d_reverse_dims(glm, device):
    B, S, H, D = 1, 2, 3, 4
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    out = torch.empty(D, H, S, B, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(out, x, B, S, H, D, 3, 2, 1, 0)
    expected = x.permute(3, 2, 1, 0).contiguous()
    assert torch.equal(out.cpu(), expected.cpu())


def test_transpose_4d_attention_shape(glm, device):
    B, S, H, D = 1, 16, 128, 2624
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    out = torch.empty(B, H, S, D, dtype=torch.bfloat16, device=device)
    glm.transpose_4d(out, x, B, S, H, D, 0, 2, 1, 3)
    expected = x.transpose(1, 2).contiguous()
    assert torch.equal(out.cpu(), expected.cpu())
