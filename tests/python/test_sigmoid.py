import torch
import pytest


@pytest.mark.parametrize("n", [64, 256, 4096])
def test_sigmoid_random(glm, device, n):
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    glm.sigmoid(out, x, n)
    ref = torch.sigmoid(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_sigmoid_large_positive(glm, device):
    x = torch.randn(128, dtype=torch.bfloat16, device=device) + 10.0
    out = torch.empty_like(x)

    glm.sigmoid(out, x, 128)
    ref = torch.sigmoid(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)
    assert (out.cpu() > 0.99).all()


def test_sigmoid_large_negative(glm, device):
    x = torch.randn(128, dtype=torch.bfloat16, device=device) - 10.0
    out = torch.empty_like(x)

    glm.sigmoid(out, x, 128)
    ref = torch.sigmoid(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)
    assert (out.cpu() < 0.01).all()


def test_sigmoid_zero(glm, device):
    x = torch.zeros(64, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    glm.sigmoid(out, x, 64)
    ref = torch.sigmoid(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)
