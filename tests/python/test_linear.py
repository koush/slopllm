import torch
import pytest
from helpers import ATOL, RTOL


def torch_linear(x, weight):
    return torch.nn.functional.linear(x.float(), weight.float()).to(torch.bfloat16)


@pytest.mark.parametrize("batch,n,k", [(1, 64, 32), (4, 128, 64), (2, 2048, 6144)])
def test_linear_random(glm, device, batch, n, k):
    x = torch.randn(batch, k, dtype=torch.bfloat16, device=device)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
    out = torch.empty(batch, n, dtype=torch.bfloat16, device=device)

    glm.linear(out, x, w, batch, n, k)
    ref = torch_linear(x, w)
    tol = 0.5 if n * k > 1_000_000 else 5e-2
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=tol, rtol=1e-2)


def test_linear_batch1_large(glm, device):
    x = torch.randn(1, 6144, dtype=torch.bfloat16, device=device)
    w = torch.randn(2048, 6144, dtype=torch.bfloat16, device=device)
    out = torch.empty(1, 2048, dtype=torch.bfloat16, device=device)

    glm.linear(out, x, w, 1, 2048, 6144)
    ref = torch_linear(x, w)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0.5, rtol=0.1)


def test_linear_identity_weight(glm, device):
    n = 64
    x = torch.randn(2, n, dtype=torch.bfloat16, device=device)
    w = torch.eye(n, dtype=torch.bfloat16, device=device)
    out = torch.empty(2, n, dtype=torch.bfloat16, device=device)

    glm.linear(out, x, w, 2, n, n)
    torch.testing.assert_close(out.cpu(), x.cpu(), atol=ATOL, rtol=RTOL)


def test_linear_zero_weight(glm, device):
    x = torch.randn(2, 32, dtype=torch.bfloat16, device=device)
    w = torch.zeros(16, 32, dtype=torch.bfloat16, device=device)
    out = torch.empty(2, 16, dtype=torch.bfloat16, device=device)

    glm.linear(out, x, w, 2, 16, 32)
    assert (out.cpu().abs() < 1e-6).all(), "Zero weight should produce zero output"


@pytest.mark.parametrize("batch,n,k", [
    (15, 2048, 6144),
    (15, 768, 256),
    (15, 512, 6144),
    (15, 64, 6144),
    (15, 256, 2048),
    (15, 256, 768),
    (15, 192, 768),
    (8, 1024, 4096),
    (32, 512, 2048),
    (64, 256, 1024),
])
def test_linear_batch_gt1(glm, device, batch, n, k):
    x = torch.randn(batch, k, dtype=torch.bfloat16, device=device)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
    out = torch.empty(batch, n, dtype=torch.bfloat16, device=device)

    glm.linear(out, x, w, batch, n, k)
    ref = torch_linear(x, w)
    tol = 0.5 if n * k > 1_000_000 else 5e-2
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=tol, rtol=1e-2)
