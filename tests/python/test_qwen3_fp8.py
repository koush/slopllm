import pytest
import torch
import numpy as np
from safetensors import safe_open
from helpers import GlmOps, get_model_path

FP8_E4M3_MAX = 448.0
FP8_BLOCK = 128


def has_fp8_model():
    try:
        get_model_path("Qwen/Qwen3-0.6B-FP8")
        return True
    except Exception:
        return False


requires_fp8_model = pytest.mark.skipif(
    not has_fp8_model(),
    reason="Qwen3-0.6B-FP8 model not cached"
)


def blockwise_dequantize_weight(fp8_w, scales, block_size=128):
    n, k = fp8_w.shape
    n_groups, k_groups = scales.shape
    fp8_float = fp8_w.float()
    fp8_blocks = fp8_float.reshape(n_groups, block_size, k_groups, block_size)
    dequant = fp8_blocks * scales.float().unsqueeze(1).unsqueeze(-1)
    return dequant.reshape(n, k).to(torch.bfloat16)


@pytest.fixture(scope="module")
def glm():
    ops = GlmOps()
    yield ops


@pytest.fixture(scope="module")
def device():
    return 0


class TestFP8WeightLoading:
    """Verify FP8 weight loading and dequantization against PyTorch reference."""

    @requires_fp8_model
    def test_weight_dequantize_single_layer(self, glm, device):
        model_dir = get_model_path("Qwen/Qwen3-0.6B-FP8")
        st_path = f"{model_dir}/model.safetensors"

        with safe_open(st_path, framework="pt", device="cpu") as f:
            fp8_w = f.get_tensor("model.layers.0.mlp.gate_proj.weight")
            scale_inv = f.get_tensor("model.layers.0.mlp.gate_proj.weight_scale_inv")

        dequant = blockwise_dequantize_weight(fp8_w, scale_inv)
        assert dequant.shape == fp8_w.shape
        assert not torch.isnan(dequant).any(), "Dequantized weights contain NaN"
        assert not torch.isinf(dequant).any(), "Dequantized weights contain Inf"
        assert dequant.abs().mean().item() > 0, "Dequantized weights are all zeros"
        assert dequant.abs().max().item() < 10, "Dequantized weights seem too large"

    @requires_fp8_model
    def test_scale_inv_dtype_and_shape(self, glm, device):
        model_dir = get_model_path("Qwen/Qwen3-0.6B-FP8")
        st_path = f"{model_dir}/model.safetensors"

        with safe_open(st_path, framework="pt", device="cpu") as f:
            scale_inv = f.get_tensor("model.layers.0.self_attn.q_proj.weight_scale_inv")

        assert scale_inv.dtype == torch.bfloat16, f"Expected BF16, got {scale_inv.dtype}"
        n, k = 2048, 1024
        assert scale_inv.shape == (n // FP8_BLOCK, k // FP8_BLOCK), \
            f"Expected shape ({n // FP8_BLOCK}, {k // FP8_BLOCK}), got {scale_inv.shape}"


class TestFP8ModelForward:
    """Test Qwen3FP8Model forward pass against PyTorch reference."""

    @requires_fp8_model
    def test_single_layer_linear(self, glm, device):
        """Test a single FP8 linear layer from actual model weights."""
        import ctypes
        from qwen3_fp8_model import Qwen3FP8Model

        model_dir = get_model_path("Qwen/Qwen3-0.6B-FP8")
        st_path = f"{model_dir}/model.safetensors"

        with safe_open(st_path, framework="pt", device="cpu") as f:
            fp8_w = f.get_tensor("model.layers.0.mlp.gate_proj.weight")
            scale_inv = f.get_tensor("model.layers.0.mlp.gate_proj.weight_scale_inv")

        w_bf16 = blockwise_dequantize_weight(fp8_w, scale_inv)
        n, k = w_bf16.shape
        assert n == 3072 and k == 1024, f"Unexpected shape: {n}, {k}"

        w_bf16_gpu = w_bf16.cuda()

        m = 1
        x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.3

        ref_out = torch.nn.functional.linear(x_bf16.cpu().float(), w_bf16.cpu().float())

        num_act_groups = k // FP8_BLOCK
        n_groups_n = n // FP8_BLOCK
        k_groups_k = k // FP8_BLOCK

        fp8_x_gpu = glm.alloc(m * k)
        scales_x_gpu = glm.alloc(m * num_act_groups * 4)
        fp8_w_gpu = glm.alloc(n * k)
        scales_w_gpu = glm.alloc(n_groups_n * k_groups_k * 4)
        out_gpu = glm.alloc(m * n * 2)
        workspace = glm.alloc(32 * 1024 * 1024)

        glm.fp8_quantize(fp8_x_gpu, scales_x_gpu, x_bf16.data_ptr(), m, k)

        fp8_w_bytes = fp8_w.view(torch.uint8)
        glm.h2d(fp8_w_gpu, fp8_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), n * k)

        scale_inv_f32 = scale_inv.float().contiguous()
        glm.h2d(scales_w_gpu, scale_inv_f32.cpu().numpy().ctypes.data_as(ctypes.c_void_p),
                 n_groups_n * k_groups_k * 4)

        glm.fp8_linear(out_gpu, fp8_x_gpu, scales_x_gpu, fp8_w_gpu, scales_w_gpu,
                        workspace, 32 * 1024 * 1024, m, n, k)
        glm.synchronize()

        out_raw = torch.empty(m, n, dtype=torch.uint16, device="cpu")
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, m * n * 2)
        out_bf16 = out_raw.view(torch.bfloat16)

        for ptr in [fp8_x_gpu, scales_x_gpu, fp8_w_gpu, scales_w_gpu, out_gpu, workspace]:
            glm.free_buf(ptr)

        cuda_float = out_bf16.float()
        mean_err = (cuda_float - ref_out).abs().mean().item()
        max_err = (cuda_float - ref_out).abs().max().item()
        mean_abs_ref = ref_out.abs().mean().item()

        assert mean_err < mean_abs_ref * 0.15, \
            f"Mean error {mean_err:.6f} > 15% of mean abs ref {mean_abs_ref:.6f}"
        assert max_err < mean_abs_ref * 5.0, \
            f"Max error {max_err:.6f} > 5x mean abs ref {mean_abs_ref:.6f}"

    @requires_fp8_model
    def test_model_load_and_forward(self, glm, device):
        """Load FP8 model and run a forward pass, checking output validity."""
        from qwen3_fp8_model import Qwen3FP8Model

        model = Qwen3FP8Model.from_pretrained(glm, "Qwen/Qwen3-0.6B-FP8",
                                                max_batch=1, max_seq_len=64)
        try:
            input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
            logits = model.forward(input_ids)

            assert not torch.isnan(logits).any(), "Output contains NaN"
            assert not torch.isinf(logits).any(), "Output contains Inf"
            assert logits.shape == (1, 5, 151936), f"Unexpected shape: {logits.shape}"

            last_logits = logits[0, -1]
            assert last_logits.abs().mean().item() > 0.01, "Logits seem too small"
            assert last_logits.abs().max().item() < 50, "Logits seem too large"

            sorted_indices = last_logits.argsort(descending=True)
            top5 = sorted_indices[:5].tolist()
            assert len(set(top5)) == 5, f"Top-5 tokens not diverse: {top5}"
        finally:
            model.free()

    @requires_fp8_model
    def test_fp8_vs_bf16_reference(self, glm, device):
        """Compare FP8 model output against PyTorch reference with dequantized weights."""
        from qwen3_fp8_model import Qwen3FP8Model

        model = Qwen3FP8Model.from_pretrained(glm, "Qwen/Qwen3-0.6B-FP8",
                                                max_batch=1, max_seq_len=8)
        try:
            torch.manual_seed(42)
            input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
            logits_fp8 = model.forward(input_ids)

            assert not torch.isnan(logits_fp8).any(), "FP8 output contains NaN"
            assert logits_fp8.shape == (1, 3, 151936), f"Unexpected shape: {logits_fp8.shape}"
        finally:
            model.free()
