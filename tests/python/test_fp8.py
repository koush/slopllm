import torch
import pytest
import ctypes
import numpy as np
from helpers import GlmOps

FP8_E4M3_MAX = 448.0
FP8_QUANT_GROUP_SIZE = 128


def per_token_group_quant_fp8_ref(input_bf16, group_size=128):
    m, k = input_bf16.shape
    num_groups_k = (k + group_size - 1) // group_size
    input_float = input_bf16.float()
    pad_k = num_groups_k * group_size
    if pad_k != k:
        padded = torch.nn.functional.pad(input_float, (0, pad_k - k))
    else:
        padded = input_float
    input_groups = padded.reshape(m, num_groups_k, group_size)
    amax = input_groups.abs().amax(dim=-1)
    scale = amax / FP8_E4M3_MAX
    scale = scale.clamp(min=1e-12)
    inv_scale = 1.0 / scale
    scaled = input_groups * inv_scale.unsqueeze(-1)
    scaled = scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    fp8_vals = scaled.to(torch.float8_e4m3fn)
    fp8_out = fp8_vals.reshape(m, pad_k)[:, :k].contiguous()
    scales_out = scale.reshape(m, num_groups_k)
    return fp8_out, scales_out


def fp8_dequantize_per_token_group(fp8_tensor, scales, group_size=128):
    m, k = fp8_tensor.shape
    num_groups_k = scales.shape[1]
    fp8_float = fp8_tensor.float()
    pad_k = num_groups_k * group_size
    if pad_k != k:
        fp8_float = torch.nn.functional.pad(fp8_float, (0, pad_k - k))
    fp8_groups = fp8_float.reshape(m, num_groups_k, group_size)
    dequant = fp8_groups * scales.unsqueeze(-1)
    return dequant.reshape(m, pad_k)[:, :k].to(torch.bfloat16)


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


def blockwise_dequantize_weight(fp8_w, scales, block_size=128):
    n, k = fp8_w.shape
    n_groups, k_groups = scales.shape
    fp8_float = fp8_w.float()
    fp8_blocks = fp8_float.reshape(n_groups, block_size, k_groups, block_size)
    dequant = fp8_blocks * scales.unsqueeze(1).unsqueeze(-1)
    return dequant.reshape(n, k).to(torch.bfloat16)


class TestFP8Quantize:

    @pytest.mark.parametrize("m,k", [
        (1, 128),
        (4, 256),
        (1, 1024),
        (2, 2048),
        (1, 6144),
    ])
    def test_fp8_quantize_roundtrip(self, glm, device, m, k):
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        num_groups = (k + 127) // 128

        fp8_gpu = glm.alloc(m * k)
        scales_gpu = glm.alloc(m * num_groups * 4)

        glm.fp8_quantize(fp8_gpu, scales_gpu, x.data_ptr(), m, k)
        glm.synchronize()

        fp8_host = torch.empty(m, k, dtype=torch.uint8, device='cpu')
        glm.d2h(fp8_host.numpy().ctypes.data_as(ctypes.c_void_p), fp8_gpu, m * k)

        scales_host = torch.empty(m, num_groups, dtype=torch.float32, device='cpu')
        glm.d2h(scales_host.numpy().ctypes.data_as(ctypes.c_void_p), scales_gpu, m * num_groups * 4)

        glm.free_buf(fp8_gpu)
        glm.free_buf(scales_gpu)

        fp8_ref, scales_ref = per_token_group_quant_fp8_ref(x.cpu())

        torch.testing.assert_close(scales_host, scales_ref, atol=1e-5, rtol=1e-5)

        fp8_view = fp8_host.view(torch.float8_e4m3fn)
        match_pct = (fp8_view == fp8_ref).float().mean().item()
        assert match_pct > 0.95, f"FP8 values match rate {match_pct:.4f} < 0.95"

        dequant_cuda = fp8_dequantize_per_token_group(fp8_view, scales_host)
        reconst_x = x.cpu().float()
        diff = (dequant_cuda.float() - reconst_x).abs().mean().item()
        assert diff < 0.1, f"Quantize roundtrip error too large: {diff}"

    def test_fp8_quantize_zero_input(self, glm, device):
        m, k = 2, 256
        x = torch.zeros(m, k, dtype=torch.bfloat16, device=device)
        num_groups = k // 128

        fp8_gpu = glm.alloc(m * k)
        scales_gpu = glm.alloc(m * num_groups * 4)

        glm.fp8_quantize(fp8_gpu, scales_gpu, x.data_ptr(), m, k)
        glm.synchronize()

        fp8_host = torch.empty(m, k, dtype=torch.uint8, device='cpu')
        glm.d2h(fp8_host.numpy().ctypes.data_as(ctypes.c_void_p), fp8_gpu, m * k)

        scales_host = torch.empty(m, num_groups, dtype=torch.float32, device='cpu')
        glm.d2h(scales_host.numpy().ctypes.data_as(ctypes.c_void_p), scales_gpu, m * num_groups * 4)

        glm.free_buf(fp8_gpu)
        glm.free_buf(scales_gpu)

        assert (fp8_host == 0).all(), "Zero input should produce zero FP8 output"
        assert (scales_host > 0).all(), "Scales should be positive (not zero)"


class TestFP8GEMM:

    @pytest.mark.parametrize("m,n,k", [
        (1, 128, 128),
        (4, 128, 256),
        (1, 1024, 2048),
        (2, 512, 1024),
    ])
    def test_fp8_gemm_vs_bf16(self, glm, device, m, n, k):
        torch.manual_seed(42)
        x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5
        w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.5

        ref_out = torch.nn.functional.linear(x_bf16.cpu().float(), w_bf16.cpu().float())

        fp8_x, act_scales = per_token_group_quant_fp8_ref(x_bf16.cpu())
        fp8_w, weight_scales = blockwise_quantize_weight(w_bf16.cpu())

        num_act_groups = k // 128
        n_groups_n = n // 128
        k_groups_k = k // 128
        workspace_size = glm.fp8_gemm_workspace_size(m, n, k)
        fp8_x_gpu = glm.alloc(m * k)
        scales_x_gpu = glm.alloc(m * num_act_groups * 4)
        fp8_w_gpu = glm.alloc(n * k)
        scales_w_gpu = glm.alloc(n_groups_n * k_groups_k * 4)
        out_gpu = glm.alloc(m * n * 2)
        workspace = glm.alloc(workspace_size)

        fp8_x_bytes = fp8_x.view(torch.uint8)
        glm.h2d(fp8_x_gpu, fp8_x_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), m * k)
        glm.h2d(scales_x_gpu, act_scales.cpu().numpy().ctypes.data_as(ctypes.c_void_p), m * num_act_groups * 4)

        fp8_w_bytes = fp8_w.view(torch.uint8)
        glm.h2d(fp8_w_gpu, fp8_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * k)
        glm.h2d(scales_w_gpu, weight_scales.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n_groups_n * k_groups_k * 4)

        glm.fp8_linear(out_gpu, fp8_x_gpu, scales_x_gpu, fp8_w_gpu, scales_w_gpu,
                       workspace, workspace_size, m, n, k)
        glm.synchronize()

        out_raw = torch.empty(m, n, dtype=torch.uint16, device='cpu')
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, m * n * 2)
        out_bf16 = out_raw.view(torch.bfloat16)

        for ptr in [fp8_x_gpu, scales_x_gpu, fp8_w_gpu, scales_w_gpu, out_gpu, workspace]:
            glm.free_buf(ptr)

        ref_float = ref_out
        cuda_float = out_bf16.float()
        mean_err = (ref_float - cuda_float).abs().mean().item()
        max_err = (ref_float - cuda_float).abs().max().item()

        mean_abs_ref = ref_float.abs().mean().item()
        assert mean_err < mean_abs_ref * 0.1, f"Mean error {mean_err:.4f} > 10% of mean abs ref {mean_abs_ref:.4f}"
        assert max_err < mean_abs_ref * 2.0, f"Max error {max_err:.4f} > 2x mean abs ref {mean_abs_ref:.4f}"


class TestFP8Linear:

    @pytest.mark.parametrize("m", [1, 4])
    def test_fp8_linear_pipeline(self, glm, device, m):
        n = 1024
        k = 1024

        torch.manual_seed(123)
        x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.3
        w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.3

        ref_out = torch.nn.functional.linear(x_bf16.cpu().float(), w_bf16.cpu().float())

        num_act_groups = k // 128
        n_groups_n = n // 128
        k_groups_k = k // 128

        fp8_x_gpu = glm.alloc(m * k)
        scales_x_gpu = glm.alloc(m * num_act_groups * 4)
        fp8_w_gpu = glm.alloc(n * k)
        scales_w_gpu = glm.alloc(n_groups_n * k_groups_k * 4)
        out_gpu = glm.alloc(m * n * 2)
        workspace_size = glm.fp8_gemm_workspace_size(m, n, k)
        workspace = glm.alloc(workspace_size)

        glm.fp8_quantize(fp8_x_gpu, scales_x_gpu, x_bf16.data_ptr(), m, k)

        fp8_w, weight_scales = blockwise_quantize_weight(w_bf16.cpu())
        fp8_w_bytes = fp8_w.view(torch.uint8)
        glm.h2d(fp8_w_gpu, fp8_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * k)
        glm.h2d(scales_w_gpu, weight_scales.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n_groups_n * k_groups_k * 4)

        glm.fp8_linear(out_gpu, fp8_x_gpu, scales_x_gpu, fp8_w_gpu, scales_w_gpu,
                       workspace, workspace_size, m, n, k)
        glm.synchronize()

        out_raw = torch.empty(m, n, dtype=torch.uint16, device='cpu')
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, m * n * 2)
        out_bf16 = out_raw.view(torch.bfloat16)

        for ptr in [fp8_x_gpu, scales_x_gpu, fp8_w_gpu, scales_w_gpu, out_gpu, workspace]:
            glm.free_buf(ptr)

        cuda_float = out_bf16.float()
        mean_err = (cuda_float - ref_out).abs().mean().item()
        max_err = (cuda_float - ref_out).abs().max().item()

        assert mean_err < 1.0, f"Mean error {mean_err:.4f} too large"
        assert max_err < 15.0, f"Max error {max_err:.4f} too large"

    def test_fp8_linear_identity_weight(self, glm, device):
        n = 128
        k = 128
        m = 2

        x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5
        w_bf16 = torch.eye(n, k, dtype=torch.bfloat16, device=device)

        num_act_groups = k // 128
        n_groups_n = n // 128
        k_groups_k = k // 128

        fp8_x_gpu = glm.alloc(m * k)
        scales_x_gpu = glm.alloc(m * num_act_groups * 4)
        fp8_w_gpu = glm.alloc(n * k)
        scales_w_gpu = glm.alloc(n_groups_n * k_groups_k * 4)
        out_gpu = glm.alloc(m * n * 2)
        workspace_size = glm.fp8_gemm_workspace_size(m, n, k)
        workspace = glm.alloc(workspace_size)

        glm.fp8_quantize(fp8_x_gpu, scales_x_gpu, x_bf16.data_ptr(), m, k)

        fp8_w, weight_scales = blockwise_quantize_weight(w_bf16.cpu())
        fp8_w_bytes = fp8_w.view(torch.uint8)
        glm.h2d(fp8_w_gpu, fp8_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * k)
        glm.h2d(scales_w_gpu, weight_scales.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n_groups_n * k_groups_k * 4)

        glm.fp8_linear(out_gpu, fp8_x_gpu, scales_x_gpu, fp8_w_gpu, scales_w_gpu,
                       workspace, workspace_size, m, n, k)
        glm.synchronize()

        out_raw = torch.empty(m, n, dtype=torch.uint16, device='cpu')
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, m * n * 2)
        out_bf16 = out_raw.view(torch.bfloat16)

        for ptr in [fp8_x_gpu, scales_x_gpu, fp8_w_gpu, scales_w_gpu, out_gpu, workspace]:
            glm.free_buf(ptr)

        ref_out = torch.nn.functional.linear(x_bf16.cpu().float(), w_bf16.cpu().float())
        diff = (out_bf16.float() - ref_out).abs().mean().item()
        assert diff < 0.5, f"Identity weight error too large: {diff}"


class TestFP8LinearDecode:

    @pytest.mark.parametrize("m,n,k", [
        (1, 128, 128),
        (1, 1024, 1024),
        (1, 3072, 1024),
        (1, 1024, 3072),
        (4, 1024, 1024),
        (8, 3072, 1024),
        (4, 1024, 3072),
    ])
    def test_fp8_decode_vs_reference(self, glm, device, m, n, k):
        torch.manual_seed(42)
        x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5
        w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.5

        ref_out = torch.nn.functional.linear(x_bf16.cpu().float(), w_bf16.cpu().float())

        num_groups_n = n // 128
        num_groups_k = k // 128

        fp8_w, weight_scales = blockwise_quantize_weight(w_bf16.cpu())
        weight_scales_f32 = weight_scales.float()

        fp8_w_gpu = glm.alloc(n * k)
        scales_w_f32_gpu = glm.alloc(num_groups_n * num_groups_k * 4)
        out_gpu = glm.alloc(m * n * 2)

        fp8_w_bytes = fp8_w.view(torch.uint8)
        glm.h2d(fp8_w_gpu, fp8_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * k)
        glm.h2d(scales_w_f32_gpu, weight_scales_f32.cpu().numpy().ctypes.data_as(ctypes.c_void_p),
                num_groups_n * num_groups_k * 4)

        glm.fp8_linear_decode(out_gpu, x_bf16.data_ptr(), fp8_w_gpu,
                              scales_w_f32_gpu, m, n, k)

        glm.synchronize()

        out_raw = torch.empty(m, n, dtype=torch.uint16, device='cpu')
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, m * n * 2)
        out_decode = out_raw.view(torch.bfloat16).float()

        for ptr in [fp8_w_gpu, scales_w_f32_gpu, out_gpu]:
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
        weight_scales_f32 = weight_scales.float()

        fp8_w_gpu = glm.alloc(n * k)
        scales_w_gpu = glm.alloc(1 * 1 * 4)
        out_gpu = glm.alloc(1 * n * 2)

        fp8_w_bytes = fp8_w.view(torch.uint8)
        glm.h2d(fp8_w_gpu, fp8_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * k)
        glm.h2d(scales_w_gpu, weight_scales_f32.cpu().numpy().ctypes.data_as(ctypes.c_void_p), 4)

        glm.fp8_linear_decode(out_gpu, x_bf16.data_ptr(), fp8_w_gpu,
                              scales_w_gpu, 1, n, k)
        glm.synchronize()

        out_raw = torch.empty(1, n, dtype=torch.uint16, device='cpu')
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, n * 2)
        out_bf16 = out_raw.view(torch.bfloat16)

        for ptr in [fp8_w_gpu, scales_w_gpu, out_gpu]:
            glm.free_buf(ptr)

        diff = (out_bf16.float() - ref_out).abs().mean().item()
        assert diff < 0.3, f"Identity weight decode error too large: {diff}"
