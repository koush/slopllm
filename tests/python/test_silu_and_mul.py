import torch
import pytest


def torch_silu_and_mul(gate, up):
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)


@pytest.mark.parametrize("batch,intermediate", [(1, 64), (4, 128), (2, 12288)])
def test_silu_and_mul_random(glm, device, batch, intermediate):
    gate = torch.randn(batch, intermediate, dtype=torch.bfloat16, device=device)
    up = torch.randn(batch, intermediate, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(gate)

    glm.silu_and_mul(out, gate, up, intermediate, batch)
    ref = torch_silu_and_mul(gate, up)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_silu_and_mul_batch1(glm, device):
    gate = torch.randn(1, 2048, dtype=torch.bfloat16, device=device)
    up = torch.randn(1, 2048, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(gate)

    glm.silu_and_mul(out, gate, up, 2048, 1)
    ref = torch_silu_and_mul(gate, up)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_silu_and_mul_negative(glm, device):
    gate = torch.randn(4, 256, dtype=torch.bfloat16, device=device) - 2.0
    up = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(gate)

    glm.silu_and_mul(out, gate, up, 256, 4)
    ref = torch_silu_and_mul(gate, up)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_silu_and_mul_zeros(glm, device):
    gate = torch.zeros(2, 128, dtype=torch.bfloat16, device=device)
    up = torch.randn(2, 128, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(gate)

    glm.silu_and_mul(out, gate, up, 128, 2)
    ref = torch_silu_and_mul(gate, up)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)
    assert (out.cpu().abs() < 1e-6).all(), "silu(0) * up should be ~0"
