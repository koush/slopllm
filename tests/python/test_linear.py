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
    (1, 384, 2048),  # Two-partition split-K.
    (2, 384, 5120),  # Four-partition split-K (router geometry).
    (1, 512, 128), (2, 513, 768), (1, 512, 5120),  # Multi-warp variants.
    (17, 512, 5120), (3, 33, 96),  # cuBLASLt, including output-stride tails.
    (1, 37, 97),  # Non-vector-aligned rows use BLAS rather than uint4 loads.
])
def test_linear_output_precision_and_dtype_cache(glm, device, batch, n, k):
    torch.manual_seed(712)
    x = torch.randn(batch, k, device=device).bfloat16()
    w = (torch.randn(n, k, device=device) / k**.5).bfloat16()
    expected = (x.double() @ w.double().T).float()
    # Identical geometry/workspace but different output descriptors must not
    # share an Lt cache entry. F32 must preserve the unrounded accumulator.
    for dtype in [torch.float32, torch.bfloat16, torch.float32]:
        out = torch.empty(batch, n, device=device, dtype=dtype)
        torch.cuda.synchronize(device)
        glm.linear(out, x, w, batch, n, k, output_f32=dtype == torch.float32)
        torch.cuda.synchronize(device)
        if dtype == torch.float32:
            torch.testing.assert_close(out, expected, atol=5e-6, rtol=5e-6)
            assert (out != out.bfloat16().float()).any(), "F32 output must not round through BF16"
        else:
            torch.testing.assert_close(out.float(), expected, atol=.008, rtol=.008)


