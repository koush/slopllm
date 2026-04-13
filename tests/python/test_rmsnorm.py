import torch
import pytest


def torch_rmsnorm(x, weight, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    return (weight.float() * x_f * inv_rms).to(torch.bfloat16)


@pytest.mark.parametrize("batch,dim", [(1, 64), (4, 128), (2, 6144), (8, 512)])
def test_rmsnorm_random(glm, device, batch, dim):
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    w = torch.randn(dim, dtype=torch.bfloat16, device=device)
    eps = 1e-5
    out = torch.empty_like(x)

    glm.rmsnorm(out, x, w, eps, dim, batch)
    ref = torch_rmsnorm(x, w, eps)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_rmsnorm_ones_weight(glm, device):
    x = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
    w = torch.ones(256, dtype=torch.bfloat16, device=device)
    eps = 1e-5
    out = torch.empty_like(x)

    glm.rmsnorm(out, x, w, eps, 256, 4)
    ref = torch_rmsnorm(x, w, eps)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_rmsnorm_small_values(glm, device):
    x = torch.randn(2, 128, dtype=torch.bfloat16, device=device) * 0.001
    w = torch.ones(128, dtype=torch.bfloat16, device=device)
    eps = 1e-5
    out = torch.empty_like(x)

    glm.rmsnorm(out, x, w, eps, 128, 2)
    ref = torch_rmsnorm(x, w, eps)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_rmsnorm_large_values(glm, device):
    x = torch.randn(2, 128, dtype=torch.bfloat16, device=device) * 100.0
    w = torch.randn(128, dtype=torch.bfloat16, device=device)
    eps = 1e-5
    out = torch.empty_like(x)

    glm.rmsnorm(out, x, w, eps, 128, 2)
    ref = torch_rmsnorm(x, w, eps)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_rmsnorm_batch1(glm, device):
    x = torch.randn(1, 6144, dtype=torch.bfloat16, device=device)
    w = torch.randn(6144, dtype=torch.bfloat16, device=device)
    eps = 1e-5
    out = torch.empty_like(x)

    glm.rmsnorm(out, x, w, eps, 6144, 1)
    ref = torch_rmsnorm(x, w, eps)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)
