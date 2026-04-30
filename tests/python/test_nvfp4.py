import torch
import pytest
import ctypes
import numpy as np
from helpers import GlmOps

FP4_E2M1_MAX = 6.0
FP4_E2M1_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP8_E4M3_MAX = 448.0
GROUP_SIZE = 16


def compute_weight_scale_2(weight):
    amax = weight.abs().max().float().clamp(min=torch.finfo(torch.float32).tiny)
    return (amax / (FP4_E2M1_MAX * FP8_E4M3_MAX)).squeeze()


def compute_weight_scale(weight, scale_2):
    rows, cols = weight.shape
    assert cols % GROUP_SIZE == 0
    w = weight.reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    block_amax = w.abs().amax(dim=-1).float()
    per_block = block_amax / (FP4_E2M1_MAX * scale_2.float())
    per_block = torch.where(per_block == 0, torch.ones_like(per_block), per_block)
    return per_block.to(torch.float8_e4m3fn)


def _cast_fp4(x):
    device = x.device
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=device, dtype=x.dtype)
    sign = torch.signbit(x).long()
    abs_x = x.abs()
    idx = torch.searchsorted(bounds, abs_x)
    return (idx + (sign << 3)).to(torch.uint8)


def pack_fp4_to_uint8(x_fp4):
    m, n = x_fp4.shape
    assert n % 2 == 0
    flat = x_fp4.reshape(-1, 2)
    packed = (flat[:, 1].to(torch.uint8) << 4) | flat[:, 0].to(torch.uint8)
    return packed.reshape(m, n // 2)


def quantize_weight_nvfp4(w_bf16, scale_2_override=None):
    w = w_bf16.float()
    scale_2 = scale_2_override if scale_2_override is not None else compute_weight_scale_2(w)
    scale = compute_weight_scale(w, scale_2)
    rows, cols = w.shape
    s = scale.float().unsqueeze(-1)
    w_blocked = w.reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    scaled = w_blocked / (s * scale_2.float())
    scaled = scaled.reshape(rows, cols)
    fp4_indices = _cast_fp4(scaled)
    packed = pack_fp4_to_uint8(fp4_indices)
    return packed, scale, scale_2


def dequantize_nvfp4(packed, scale, scale_2, orig_shape):
    rows, cols = orig_shape
    low = (packed & 0x0F).to(torch.uint8)
    high = (packed >> 4).to(torch.uint8)
    unpacked = torch.stack([low, high], dim=-1).reshape(rows, cols)
    mag_idx = (unpacked & 0x07).long()
    sign = ((unpacked >> 3) & 1).float()
    lut = torch.tensor(FP4_E2M1_VALS, dtype=torch.float32, device=packed.device)
    vals = lut[mag_idx]
    vals = vals * (1.0 - 2.0 * sign)
    num_groups = cols // GROUP_SIZE
    vals = vals.reshape(rows, num_groups, GROUP_SIZE)
    s = scale.float().unsqueeze(-1)
    vals = vals * s * scale_2.float()
    return vals.reshape(rows, cols)


class TestNVFP4LinearDecode:

    @pytest.mark.parametrize("m,n,k", [
        (1, 128, 128),
        (1, 1024, 1024),
        (1, 2048, 6144),
        (1, 6144, 2048),
        (2, 128, 128),
        (4, 1024, 1024),
        (8, 2048, 6144),
        (4, 6144, 2048),
        (16, 1024, 1024),
        (32, 2048, 6144),
        (16, 6144, 2048),
        (64, 128, 128),
        (3, 1024, 1024),
    ])
    def test_nvfp4_decode_vs_reference(self, glm, device, m, n, k):
        torch.manual_seed(42)
        x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5
        w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.5

        packed, scale, scale_2 = quantize_weight_nvfp4(w_bf16.cpu())
        w_dequant = dequantize_nvfp4(packed, scale, scale_2, (n, k))
        ref_out = torch.nn.functional.linear(x_bf16.cpu().float(), w_dequant.float())

        num_k_groups = k // GROUP_SIZE

        fp4_w_gpu = glm.alloc(n * (k // 2))
        scale_fp8_gpu = glm.alloc(n * num_k_groups)
        scale2_f32_gpu = glm.alloc(4)
        out_gpu = glm.alloc(m * n * 2)

        fp4_w_bytes = packed.view(torch.uint8)
        glm.h2d(fp4_w_gpu, fp4_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * (k // 2))

        scale_fp8_bytes = scale.view(torch.uint8)
        glm.h2d(scale_fp8_gpu, scale_fp8_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * num_k_groups)

        scale_2_f32 = scale_2.float().contiguous()
        glm.h2d(scale2_f32_gpu, scale_2_f32.numpy().ctypes.data_as(ctypes.c_void_p), 4)

        glm.nvfp4_linear_decode(out_gpu, x_bf16.data_ptr(), fp4_w_gpu,
                                scale_fp8_gpu, scale2_f32_gpu, m, n, k)

        glm.synchronize()

        out_raw = torch.empty(m, n, dtype=torch.uint16, device='cpu')
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, m * n * 2)
        out_decode = out_raw.view(torch.bfloat16).float()

        for ptr in [fp4_w_gpu, scale_fp8_gpu, scale2_f32_gpu, out_gpu]:
            glm.free_buf(ptr)

        mean_err = (out_decode - ref_out).abs().mean().item()
        max_err = (out_decode - ref_out).abs().max().item()
        mean_abs_ref = ref_out.abs().mean().item()

        assert mean_err < mean_abs_ref * 0.15, f"Decode mean error {mean_err:.4f} > 15% of ref {mean_abs_ref:.4f}"
        assert max_err < mean_abs_ref * 3.0, f"Decode max error {max_err:.4f} > 3x ref {mean_abs_ref:.4f}"

    def test_nvfp4_decode_identity(self, glm, device):
        n, k = 128, 128
        x_bf16 = torch.randn(1, k, dtype=torch.bfloat16, device=device) * 0.5
        w_bf16 = torch.eye(n, k, dtype=torch.bfloat16, device=device)

        packed, scale, scale_2 = quantize_weight_nvfp4(w_bf16.cpu())
        w_dequant = dequantize_nvfp4(packed, scale, scale_2, (n, k))
        ref_out = torch.nn.functional.linear(x_bf16.cpu().float(), w_dequant.float())

        num_k_groups = k // GROUP_SIZE

        fp4_w_gpu = glm.alloc(n * (k // 2))
        scale_fp8_gpu = glm.alloc(n * num_k_groups)
        scale2_f32_gpu = glm.alloc(4)
        out_gpu = glm.alloc(1 * n * 2)

        fp4_w_bytes = packed.view(torch.uint8)
        glm.h2d(fp4_w_gpu, fp4_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * (k // 2))

        scale_fp8_bytes = scale.view(torch.uint8)
        glm.h2d(scale_fp8_gpu, scale_fp8_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * num_k_groups)

        scale_2_f32 = scale_2.float().contiguous()
        glm.h2d(scale2_f32_gpu, scale_2_f32.numpy().ctypes.data_as(ctypes.c_void_p), 4)

        glm.nvfp4_linear_decode(out_gpu, x_bf16.data_ptr(), fp4_w_gpu,
                                scale_fp8_gpu, scale2_f32_gpu, 1, n, k)

        glm.synchronize()

        out_raw = torch.empty(1, n, dtype=torch.uint16, device='cpu')
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, n * 2)
        out_bf16 = out_raw.view(torch.bfloat16)

        for ptr in [fp4_w_gpu, scale_fp8_gpu, scale2_f32_gpu, out_gpu]:
            glm.free_buf(ptr)

        diff = (out_bf16.float() - ref_out).abs().mean().item()
        assert diff < 0.35, f"Identity weight decode error too large: {diff}"

    def test_nvfp4_shared_scale_2(self, glm, device):
        n, k = 256, 256
        torch.manual_seed(99)
        gate_w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.5
        up_w = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.5

        gate_amax = gate_w.abs().max().float()
        up_amax = up_w.abs().max().float()
        shared_amax = torch.max(gate_amax, up_amax).clamp(min=torch.finfo(torch.float32).tiny)
        shared_scale_2 = (shared_amax / (FP4_E2M1_MAX * FP8_E4M3_MAX)).squeeze()

        gate_packed_s2, gate_scale_s2, gate_s2 = quantize_weight_nvfp4(gate_w.cpu(), scale_2_override=shared_scale_2.cpu())
        up_packed_s2, up_scale_s2, up_s2 = quantize_weight_nvfp4(up_w.cpu(), scale_2_override=shared_scale_2.cpu())

        assert torch.allclose(gate_s2, shared_scale_2.cpu()), "gate scale_2 should match shared"
        assert torch.allclose(up_s2, shared_scale_2.cpu()), "up scale_2 should match shared"

        gate_dequant = dequantize_nvfp4(gate_packed_s2, gate_scale_s2, gate_s2, (n, k))
        up_dequant = dequantize_nvfp4(up_packed_s2, up_scale_s2, up_s2, (n, k))

        gate_ref = torch.nn.functional.linear(
            torch.randn(1, k, dtype=torch.bfloat16).cpu().float(), gate_dequant.float())
        up_ref = torch.nn.functional.linear(
            torch.randn(1, k, dtype=torch.bfloat16).cpu().float(), up_dequant.float())

        assert not torch.isnan(gate_ref).any(), "Gate dequant with shared scale_2 contains NaN"
        assert not torch.isnan(up_ref).any(), "Up dequant with shared scale_2 contains NaN"
