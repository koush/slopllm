"""Production projections, Engram gating, and packed MXFP4 expert kernels."""
import ctypes

import pytest
import torch


def test_engram_gate_matches_normalized_signed_sqrt(glm, device):
    torch.manual_seed(119)
    tokens, dim = 3, 5120
    x = torch.randn(tokens, 4, dim, device=device).bfloat16()
    kv = torch.randn(tokens, 5, dim, device=device).bfloat16()
    qw, kw = [torch.randn(4, dim, device=device).bfloat16() for _ in range(2)]
    x[0, 0] = 0
    kv[1, 1] = 0
    out = torch.empty_like(x)
    fn = glm.lib.glm_dsv41_engram_gate
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2 + [ctypes.c_float]
    torch.cuda.synchronize(device)
    fn(glm.ctx, out.data_ptr(), x.data_ptr(), kv.data_ptr(), qw.data_ptr(), kw.data_ptr(), tokens, dim, 1e-20)
    torch.cuda.synchronize(device)
    h, k = x.float(), kv[:, :4].float()
    dot = (h * qw.float() * kw.float() * k).sum(-1)
    dot *= torch.rsqrt(h.square().mean(-1) + 1e-20) * torch.rsqrt(k.square().mean(-1) + 1e-20) / dim**.5
    gate = torch.where(dot < 0, -1, 1) * dot.abs().clamp_min(1e-6).sqrt()
    expected = (h + gate.sigmoid()[..., None] * kv[:, 4:5].float()).bfloat16()
    torch.testing.assert_close(out, expected, atol=.008, rtol=.008)


@pytest.mark.parametrize("tokens,n,k", [(1, 288, 5120), (7, 288, 5120), (33, 5120, 288),
                                      (3, 37, 96), (17, 130, 512), (2, 64, 32)])
@pytest.mark.parametrize("top_k", [1, 6])
def test_mxfp4_shared_cuda_core_mul_mat_id(glm, device, tokens, n, k, top_k):
    """Independent unpack + FP64 dot oracle; all routes in one launch."""
    torch.manual_seed(419)
    experts = 4
    weight = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device=device)
    scales = torch.randint(115, 130, (experts, n, k // 32), dtype=torch.uint8, device=device)
    table = torch.tensor([[weight[e].data_ptr(), scales[e].data_ptr()] for e in range(experts)],
                         dtype=torch.int64, device=device)
    ids = torch.randint(0, experts, (tokens, 6), dtype=torch.int32, device=device)
    count = tokens * 6
    x = torch.randn(count // top_k, k, device=device).bfloat16()
    bits = torch.stack([weight & 15, weight >> 4], -1).reshape(experts, n, k)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=device)
    dense = lut[(bits & 7).long()] * torch.where(bits & 8 != 0, -1, 1)
    dense *= (2. ** (scales.float() - 127)).repeat_interleave(32, -1)
    fn = glm.lib.glm_mxfp4_mul_mat_id
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
    out = torch.empty(count, n, device=device, dtype=torch.bfloat16)
    expected = torch.bmm(dense[ids.flatten().long()].double(), x.repeat_interleave(top_k, 0).double().unsqueeze(-1)).squeeze(-1).bfloat16()
    torch.cuda.synchronize(device)
    assert fn(glm.ctx, out.data_ptr(), x.data_ptr(), table.data_ptr(), ids.data_ptr(), top_k, count, n, k) == 0
    torch.cuda.synchronize(device)
    torch.testing.assert_close(out, expected, atol=.002, rtol=.008)


@pytest.mark.parametrize("exponent", [0, 1, 127, 254, 255])
def test_mxfp4_scale_encoding_edges(glm, device, exponent):
    fn = glm.lib.glm_mxfp4_mul_mat_id
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
    weight = torch.full((1, 16), 0x22, dtype=torch.uint8, device=device)
    scale = torch.full((1, 1), exponent, dtype=torch.uint8, device=device)
    table = torch.tensor([weight.data_ptr(), scale.data_ptr()], dtype=torch.int64, device=device)
    ids = torch.zeros(1, 6, dtype=torch.int32, device=device)
    # Compensate the extreme exponent so the final result stays finite.
    xvalue = 2. ** (120 if exponent <= 1 else -120 if exponent == 254 else 0)
    x = torch.full((1, 32), xvalue, dtype=torch.bfloat16, device=device)
    out = torch.empty(1, 1, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize(device)
    assert fn(glm.ctx, out.data_ptr(), x.data_ptr(), table.data_ptr(), ids.data_ptr(), 1, 1, 1, 32) == 0
    torch.cuda.synchronize(device)
    if exponent == 255:
        assert out.isnan().all()
    else:
        expected = 32 * xvalue * 2. ** (exponent - 127)
        assert out.item() == expected


def test_mxfp4_mul_mat_id_rejects_invalid_geometry(glm):
    fn = glm.lib.glm_mxfp4_mul_mat_id
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
    for top_k, count, n, k in [(1, 0, 32, 32), (1, 1, 0, 32), (1, 1, 32, 16), (0, 1, 32, 32), (6, 5, 32, 32)]:
        assert fn(glm.ctx, None, None, None, None, top_k, count, n, k) != 0
