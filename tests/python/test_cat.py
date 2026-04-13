import torch
import pytest


@pytest.mark.parametrize("outer,a_dim,b_dim", [(1, 4, 4), (2, 8, 12), (4, 16, 32)])
def test_cat_last_dim_basic(glm, device, outer, a_dim, b_dim):
    a = torch.randn(outer, a_dim, dtype=torch.bfloat16, device=device)
    b = torch.randn(outer, b_dim, dtype=torch.bfloat16, device=device)
    out = torch.empty(outer, a_dim + b_dim, dtype=torch.bfloat16, device=device)

    glm.cat_last_dim(out, a, b, a_dim, b_dim, outer)

    ref = torch.cat([a, b], dim=-1)
    assert torch.equal(out.cpu(), ref.cpu())


def test_cat_last_dim_3d(glm, device):
    B, H, S, d1, d2 = 2, 4, 8, 32, 64
    a = torch.randn(B, H, S, d1, dtype=torch.bfloat16, device=device)
    b = torch.randn(B, H, S, d2, dtype=torch.bfloat16, device=device)
    out = torch.empty(B, H, S, d1 + d2, dtype=torch.bfloat16, device=device)

    outer = B * H * S
    a_flat = a.reshape(outer, d1)
    b_flat = b.reshape(outer, d2)
    out_flat = out.reshape(outer, d1 + d2)

    glm.cat_last_dim(out_flat, a_flat, b_flat, d1, d2, outer)

    ref = torch.cat([a, b], dim=-1)
    assert torch.equal(out.cpu(), ref.cpu())


def test_cat_last_dim_one_empty(glm, device):
    outer = 3
    a = torch.randn(outer, 8, dtype=torch.bfloat16, device=device)
    b = torch.randn(outer, 0, dtype=torch.bfloat16, device=device)
    out = torch.empty(outer, 8, dtype=torch.bfloat16, device=device)

    a_flat = a.reshape(outer, 8)
    b_flat = b.reshape(outer, 0)
    out_flat = out.reshape(outer, 8)

    glm.cat_last_dim(out_flat, a_flat, b_flat, 8, 0, outer)
    assert torch.equal(out.cpu(), a.cpu())
