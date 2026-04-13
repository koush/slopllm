import torch
import pytest


@pytest.mark.parametrize("n", [64, 256, 4096])
def test_relu_random(glm, device, n):
    x = torch.randn(n, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    glm.relu(out, x, n)
    ref = torch.relu(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3)


def test_relu_negative(glm, device):
    x = torch.randn(128, dtype=torch.bfloat16, device=device) - 5.0
    out = torch.empty_like(x)

    glm.relu(out, x, 128)
    ref = torch.relu(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3)


def test_relu_all_positive(glm, device):
    x = torch.abs(torch.randn(128, dtype=torch.bfloat16, device=device))
    out = torch.empty_like(x)

    glm.relu(out, x, 128)
    assert torch.equal(out.cpu(), x.cpu())


def test_relu_zeros(glm, device):
    x = torch.zeros(64, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    glm.relu(out, x, 64)
    assert (out.cpu() == 0).all()


def test_relu_2d(glm, device):
    x = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    glm.relu(out, x, 4 * 256)
    ref = torch.relu(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3)
