import torch
import pytest


def torch_rmsnorm(x, weight, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    return (weight.float() * x_f * inv_rms).to(torch.bfloat16)


def torch_add_rmsnorm(input_a, input_b, weight, eps):
    residual = input_a + input_b
    normed = torch_rmsnorm(residual, weight, eps)
    return normed, residual


@pytest.mark.parametrize("batch,dim", [(1, 64), (4, 128), (2, 1024), (8, 512), (1, 6144)])
def test_fused_add_rmsnorm_random(glm, device, batch, dim):
    a = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    b = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    w = torch.randn(dim, dtype=torch.bfloat16, device=device)
    eps = 1e-5

    out = torch.empty_like(a)
    residual = torch.empty_like(a)

    glm.fused_add_rmsnorm(out, residual, a, b, w, eps, dim, batch)

    ref_normed, ref_residual = torch_add_rmsnorm(a, b, w, eps)
    torch.testing.assert_close(residual.cpu(), ref_residual.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(out.cpu(), ref_normed.cpu(), atol=1e-2, rtol=1e-2)


def test_fused_add_rmsnorm_ones_weight(glm, device):
    a = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
    b = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
    w = torch.ones(256, dtype=torch.bfloat16, device=device)
    eps = 1e-5

    out = torch.empty_like(a)
    residual = torch.empty_like(a)

    glm.fused_add_rmsnorm(out, residual, a, b, w, eps, 256, 4)

    ref_normed, ref_residual = torch_add_rmsnorm(a, b, w, eps)
    torch.testing.assert_close(residual.cpu(), ref_residual.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(out.cpu(), ref_normed.cpu(), atol=1e-2, rtol=1e-2)


def test_fused_add_rmsnorm_zero_b(glm, device):
    a = torch.randn(2, 128, dtype=torch.bfloat16, device=device)
    b = torch.zeros(2, 128, dtype=torch.bfloat16, device=device)
    w = torch.randn(128, dtype=torch.bfloat16, device=device)
    eps = 1e-5

    out = torch.empty_like(a)
    residual = torch.empty_like(a)

    glm.fused_add_rmsnorm(out, residual, a, b, w, eps, 128, 2)

    ref_normed = torch_rmsnorm(a, w, eps)
    ref_residual = a.clone()
    torch.testing.assert_close(residual.cpu(), ref_residual.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(out.cpu(), ref_normed.cpu(), atol=1e-2, rtol=1e-2)


def test_fused_add_rmsnorm_large_values(glm, device):
    a = torch.randn(2, 128, dtype=torch.bfloat16, device=device) * 100.0
    b = torch.randn(2, 128, dtype=torch.bfloat16, device=device) * 100.0
    w = torch.randn(128, dtype=torch.bfloat16, device=device)
    eps = 1e-5

    out = torch.empty_like(a)
    residual = torch.empty_like(a)

    glm.fused_add_rmsnorm(out, residual, a, b, w, eps, 128, 2)

    ref_normed, ref_residual = torch_add_rmsnorm(a, b, w, eps)
    torch.testing.assert_close(residual.cpu(), ref_residual.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(out.cpu(), ref_normed.cpu(), atol=1e-2, rtol=1e-2)


def test_fused_add_rmsnorm_matches_separate(glm, device):
    batch, dim = 4, 1024
    a = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    b = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    w = torch.randn(dim, dtype=torch.bfloat16, device=device)
    eps = 1e-6

    out_fused = torch.empty_like(a)
    residual_fused = torch.empty_like(a)
    glm.fused_add_rmsnorm(out_fused, residual_fused, a, b, w, eps, dim, batch)

    residual_sep = torch.empty_like(a)
    out_sep = torch.empty_like(a)
    glm.add(residual_sep, a, b, batch * dim)
    glm.rmsnorm(out_sep, residual_sep, w, eps, dim, batch)

    torch.testing.assert_close(residual_fused.cpu(), residual_sep.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(out_fused.cpu(), out_sep.cpu(), atol=2e-2, rtol=1e-2)
