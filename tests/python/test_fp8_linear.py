"""FP4-style CUDA-core FP8 linear: scale blocks, vector/tail paths and groups."""
import ctypes

import pytest
import torch


def native(glm):
    fn = glm.lib.glm_fp8_block_linear
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6
    return fn


@pytest.mark.parametrize("m,n,k,groups", [(9, 160, 1280, 1), (129, 256, 512, 1), (17, 384, 256, 3), (33, 160, 288, 1)])
def test_fp8_cublaslt_quantized_oracle(glm, device, m, n, k, groups):
    torch.manual_seed(528)
    x = torch.randn(m, groups * k, device=device).bfloat16()
    x[0] = 0
    w = torch.randn(n, k, device=device).to(torch.float8_e4m3fn)
    s = torch.randint(119, 131, (n // 32, k // 32), device=device, dtype=torch.uint8)
    blocks = x.double().reshape(m, groups, k // 32, 32)
    scale = 2. ** torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
    qx = ((blocks / scale).float().to(torch.float8_e4m3fn).double() * scale).reshape(m, groups, k)
    dw = w.double() * (2. ** (s.double() - 127)).repeat_interleave(32, 0).repeat_interleave(32, 1)
    expected = torch.einsum("mgk,gnk->mgn", qx, dw.reshape(groups, n // groups, k)).reshape(m, n).bfloat16()
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    pk = (k + 127) // 128 * 128
    q = torch.empty(m, pk, device=device, dtype=torch.uint8)
    xs = torch.empty(((m + 127) // 128 * 128, pk // 32), device=device, dtype=torch.uint8)
    ws = torch.empty((((n // groups + 127) // 128) * 128, pk // 32), device=device, dtype=torch.uint8)
    scratch = torch.empty(32 * 1024 * 1024 + (n // groups * pk if k != pk else 0), device=device, dtype=torch.uint8)
    fn = glm.lib.glm_fp8_linear_lt
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4
    torch.cuda.synchronize(device)
    assert fn(glm.ctx, *[t.data_ptr() for t in (out, x, w, s, q, xs, ws, scratch)], m, n, k, groups) == 0
    torch.cuda.synchronize(device)
    torch.testing.assert_close(out, expected, atol=.005, rtol=.008)


@pytest.mark.parametrize("block,kind", [(128, 0), (32, 1)])
@pytest.mark.parametrize("m,n,k,groups", [
    (1, 160, 5120, 1), (7, 512, 1280, 1), (33, 512, 288, 1), (1, 3200, 6144, 1),
    (3, 37, 95, 1), (2, 19, 513, 1), (4, 96, 64, 3), (1, 128, 32, 4),
    (17, 130, 512, 1), (2, 64, 1, 1),
])
def test_fp8_linear_fp64_oracle(glm, device, block, kind, m, n, k, groups):
    torch.manual_seed(2841)
    x = (torch.randn(m, groups * k, device=device) * .5).bfloat16()
    weight = (torch.randn(n, k, device=device) * .5).to(torch.float8_e4m3fn)
    shape = ((n + block - 1) // block, (k + block - 1) // block)
    if kind:
        scales = torch.randint(115, 130, shape, dtype=torch.uint8, device=device)
        decoded = 2. ** (scales.double() - 127)
    else:
        scales = (torch.rand(shape, device=device) * 2 + .01).bfloat16()
        decoded = scales.double()
    dense = weight.double() * decoded.repeat_interleave(block, 0).repeat_interleave(block, 1)[:n, :k]
    expected = torch.einsum("mgk,gnk->mgn", x.double().reshape(m, groups, k),
                            dense.reshape(groups, n // groups, k)).reshape(m, n).bfloat16()
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    assert native(glm)(glm.ctx, out.data_ptr(), x.data_ptr(), weight.data_ptr(), scales.data_ptr(),
                       m, n, k, block, kind, groups) == 0
    torch.cuda.synchronize(device)
    torch.testing.assert_close(out, expected, atol=.001, rtol=.008)


@pytest.mark.parametrize("exponent", [0, 1, 127, 254, 255])
def test_fp8_e8m0_extreme_scales(glm, device, exponent):
    weight = torch.ones(1, 32, device=device).to(torch.float8_e4m3fn)
    scales = torch.full((1, 1), exponent, dtype=torch.uint8, device=device)
    value = 2. ** (120 if exponent <= 1 else -120 if exponent == 254 else 0)
    x = torch.full((1, 32), value, dtype=torch.bfloat16, device=device)
    out = torch.empty(1, 1, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    assert native(glm)(glm.ctx, out.data_ptr(), x.data_ptr(), weight.data_ptr(), scales.data_ptr(),
                       1, 1, 32, 32, 1, 1) == 0
    torch.cuda.synchronize(device)
    if exponent == 255:
        assert out.isnan().all()
    else:
        assert out.item() == 32 * value * 2. ** (exponent - 127)


def test_fp8_all_finite_encodings(glm, device):
    # Every row isolates a code, including signed zero and FP8 subnormals.
    codes = torch.tensor([i for i in range(256) if i not in (127, 255)], device=device, dtype=torch.uint8)
    weight = codes[:, None].expand(-1, 32).contiguous()
    x = torch.ones(1, 32, device=device, dtype=torch.bfloat16)
    scales = torch.full((8, 1), 127, device=device, dtype=torch.uint8)
    out = torch.empty(1, 254, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    assert native(glm)(glm.ctx, out.data_ptr(), x.data_ptr(), weight.data_ptr(), scales.data_ptr(),
                       1, 254, 32, 32, 1, 1) == 0
    torch.cuda.synchronize(device)
    expected = (codes.view(torch.float8_e4m3fn).float() * 32).bfloat16()[None]
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_fp8_linear_invalid_geometry(glm):
    for args in [(0, 32, 32, 32, 1, 1), (1, 32, 0, 32, 1, 1),
                 (1, 32, 32, 32, 0, 1), (1, 32, 32, 128, 1, 1), (1, 32, 32, 32, 1, 3)]:
        assert native(glm)(glm.ctx, None, None, None, None, *args) != 0
