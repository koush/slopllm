import torch
import pytest
import ctypes


def torch_ref(peers, inputA, weight, eps):
    """Reference: FP32 accumulation (matches the kernel), BF16 round at store."""
    b = sum(p.float() for p in peers)
    s = inputA.float() + b
    residual = s.bfloat16() if inputA.dtype == torch.bfloat16 else s
    inv_rms = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    out = (weight.float() * s * inv_rms)
    out = out.bfloat16() if inputA.dtype == torch.bfloat16 else out
    return out, residual


def run_kernel(glm, peers, inputA, weight, eps, dtype=9):
    N = len(peers)
    dim = inputA.shape[1]
    numel = inputA.numel()
    out = torch.empty_like(inputA)
    residual = torch.empty_like(inputA)
    glm.rmsnorm_pointers_smem(
        [p.data_ptr() for p in peers], inputA, weight, out, residual,
        N, numel, dim, eps, dtype,
    )
    glm.synchronize()
    return out, residual


# Valid dims: dim % 512 == 0 and 8192 % dim == 0.
VALID_DIMS = [512, 1024, 2048, 4096, 8192]


@pytest.mark.parametrize("dim", VALID_DIMS)
@pytest.mark.parametrize("N", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("batch", [1, 3, 8, 10, 80])
def test_rmsnorm_pointers_smem_bf16(glm, device, batch, N, dim):
    peers = [torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
             for _ in range(N)]
    inputA = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(dim, dtype=torch.bfloat16, device=device)
    eps = 1e-5

    out, residual = run_kernel(glm, peers, inputA, weight, eps, dtype=9)
    ref_out, ref_res = torch_ref(peers, inputA, weight, eps)

    torch.testing.assert_close(residual.cpu(), ref_res.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(out.cpu(), ref_out.cpu(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dim", [512, 1024, 2048])
@pytest.mark.parametrize("N", [1, 2, 4, 8])
def test_rmsnorm_pointers_smem_f32(glm, device, N, dim):
    batch = 10
    peers = [torch.randn(batch, dim, dtype=torch.float32, device=device)
             for _ in range(N)]
    inputA = torch.randn(batch, dim, dtype=torch.float32, device=device)
    weight = torch.randn(dim, dtype=torch.float32, device=device)
    eps = 1e-6

    out, residual = run_kernel(glm, peers, inputA, weight, eps, dtype=7)
    ref_out, ref_res = torch_ref(peers, inputA, weight, eps)

    torch.testing.assert_close(residual.cpu(), ref_res.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out.cpu(), ref_out.cpu(), atol=1e-3, rtol=1e-3)


def test_rmsnorm_pointers_smem_matches_unfused(glm, device):
    """Fused kernel must match: reduce peers then call fused_add_rmsnorm."""
    batch, dim, N = 8, 1024, 4
    peers = [torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
             for _ in range(N)]
    inputA = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(dim, dtype=torch.bfloat16, device=device)
    eps = 1e-5

    out_fused, res_fused = run_kernel(glm, peers, inputA, weight, eps, dtype=9)

    b_reduced = sum(p.float() for p in peers).bfloat16()
    out_ref = torch.empty_like(inputA)
    res_ref = torch.empty_like(inputA)
    glm.fused_add_rmsnorm(out_ref, res_ref, inputA, b_reduced, weight, eps, dim, batch)
    glm.synchronize()

    # The unfused path rounds b_reduced to BF16 before adding inputA; the fused
    # kernel keeps the peer sum in FP32, so they differ by up to ~1 BF16 ULP.
    torch.testing.assert_close(res_fused.cpu(), res_ref.cpu(), atol=0.125, rtol=2e-2)
    torch.testing.assert_close(out_fused.cpu(), out_ref.cpu(), atol=2e-2, rtol=2e-2)


def test_rmsnorm_pointers_smem_large_values(glm, device):
    batch, dim, N = 4, 1024, 8
    peers = [torch.randn(batch, dim, dtype=torch.bfloat16, device=device) * 100.0
             for _ in range(N)]
    inputA = torch.randn(batch, dim, dtype=torch.bfloat16, device=device) * 100.0
    weight = torch.randn(dim, dtype=torch.bfloat16, device=device)
    eps = 1e-5

    out, residual = run_kernel(glm, peers, inputA, weight, eps, dtype=9)
    ref_out, ref_res = torch_ref(peers, inputA, weight, eps)

    torch.testing.assert_close(residual.cpu(), ref_res.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(out.cpu(), ref_out.cpu(), atol=2e-2, rtol=2e-2)


def test_rmsnorm_pointers_smem_single_peer(glm, device):
    """N=1: pure add+rmsnorm, no cross-peer accumulation."""
    batch, dim = 3, 1024
    peer = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    inputA = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(dim, dtype=torch.bfloat16, device=device)
    eps = 1e-5

    out, residual = run_kernel(glm, [peer], inputA, weight, eps, dtype=9)
    ref_out, ref_res = torch_ref([peer], inputA, weight, eps)

    torch.testing.assert_close(residual.cpu(), ref_res.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(out.cpu(), ref_out.cpu(), atol=1e-2, rtol=1e-2)
