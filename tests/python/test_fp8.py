import torch
import pytest
import ctypes
import numpy as np
from helpers import GlmOps

FP8_E4M3_MAX = 448.0


def blockwise_quantize_weight(w_bf16, block_size=128):
    n, k = w_bf16.shape
    assert n % block_size == 0 and k % block_size == 0
    n_groups = n // block_size
    k_groups = k // block_size
    w_float = w_bf16.float()
    w_blocks = w_float.reshape(n_groups, block_size, k_groups, block_size)
    w_amax = w_blocks.abs().amax(dim=-1).amax(dim=1)
    scales = w_amax / FP8_E4M3_MAX
    scales = scales.clamp(min=1e-12)
    inv_s = 1.0 / scales
    w_scaled = w_blocks * inv_s.unsqueeze(1).unsqueeze(-1)
    w_scaled = w_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    fp8_w = w_scaled.to(torch.float8_e4m3fn).reshape(n, k)
    return fp8_w, scales


class TestFP8LinearDecode:

    @pytest.mark.parametrize("m,n,k", [
        (1, 128, 128),
        (1, 1024, 1024),
        (1, 3072, 1024),
        (1, 1024, 3072),
        (2, 128, 128),
        (4, 1024, 1024),
        (8, 3072, 1024),
        (4, 1024, 3072),
        (16, 1024, 1024),
        (32, 3072, 1024),
        (16, 1024, 3072),
        (64, 128, 128),
        (17, 256, 256),
        (3, 1024, 1024),
    ])
    def test_fp8_decode_vs_reference(self, glm, device, m, n, k):
        torch.manual_seed(42)
        x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5
        w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.5

        ref_out = torch.nn.functional.linear(x_bf16.cpu().float(), w_bf16.cpu().float())

        num_groups_n = n // 128
        num_groups_k = k // 128

        fp8_w, weight_scales = blockwise_quantize_weight(w_bf16.cpu())
        weight_scales_bf16 = weight_scales.to(torch.bfloat16)

        fp8_w_gpu = glm.alloc(n * k)
        scales_w_bf16_gpu = glm.alloc(num_groups_n * num_groups_k * 2)
        out_gpu = glm.alloc(m * n * 2)

        fp8_w_bytes = fp8_w.view(torch.uint8)
        glm.h2d(fp8_w_gpu, fp8_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * k)
        scales_bf16_bytes = weight_scales_bf16.view(torch.uint16)
        glm.h2d(scales_w_bf16_gpu, scales_bf16_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p),
                num_groups_n * num_groups_k * 2)

        glm.fp8_linear_decode(out_gpu, x_bf16.data_ptr(), fp8_w_gpu,
                              scales_w_bf16_gpu, m, n, k)

        glm.synchronize()

        out_raw = torch.empty(m, n, dtype=torch.uint16, device='cpu')
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, m * n * 2)
        out_decode = out_raw.view(torch.bfloat16).float()

        for ptr in [fp8_w_gpu, scales_w_bf16_gpu, out_gpu]:
            glm.free_buf(ptr)

        mean_err = (out_decode - ref_out).abs().mean().item()
        max_err = (out_decode - ref_out).abs().max().item()
        mean_abs_ref = ref_out.abs().mean().item()

        assert mean_err < mean_abs_ref * 0.1, f"Decode mean error {mean_err:.4f} > 10% of ref {mean_abs_ref:.4f}"
        assert max_err < mean_abs_ref * 2.0, f"Decode max error {max_err:.4f} > 2x ref {mean_abs_ref:.4f}"

    def test_fp8_decode_identity(self, glm, device):
        n, k = 128, 128
        x_bf16 = torch.randn(1, k, dtype=torch.bfloat16, device=device) * 0.5
        w_bf16 = torch.eye(n, k, dtype=torch.bfloat16, device=device)

        ref_out = torch.nn.functional.linear(x_bf16.cpu().float(), w_bf16.cpu().float())

        fp8_w, weight_scales = blockwise_quantize_weight(w_bf16.cpu())
        weight_scales_bf16 = weight_scales.to(torch.bfloat16)

        fp8_w_gpu = glm.alloc(n * k)
        scales_w_bf16_gpu = glm.alloc(1 * 1 * 2)
        out_gpu = glm.alloc(1 * n * 2)

        fp8_w_bytes = fp8_w.view(torch.uint8)
        glm.h2d(fp8_w_gpu, fp8_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * k)
        scales_bf16_bytes = weight_scales_bf16.view(torch.uint16)
        glm.h2d(scales_w_bf16_gpu, scales_bf16_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), 2)

        glm.fp8_linear_decode(out_gpu, x_bf16.data_ptr(), fp8_w_gpu,
                              scales_w_bf16_gpu, 1, n, k)
        glm.synchronize()

        out_raw = torch.empty(1, n, dtype=torch.uint16, device='cpu')
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, n * 2)
        out_bf16 = out_raw.view(torch.bfloat16)

        for ptr in [fp8_w_gpu, scales_w_bf16_gpu, out_gpu]:
            glm.free_buf(ptr)

        diff = (out_bf16.float() - ref_out).abs().mean().item()
        assert diff < 0.3, f"Identity weight decode error too large: {diff}"
