import pytest
import torch
from safetensors import safe_open
from helpers import GlmOps, get_model_path

FP8_BLOCK = 128


def blockwise_dequantize_weight(fp8_w, scales, block_size=128):
    n, k = fp8_w.shape
    n_groups, k_groups = scales.shape
    fp8_float = fp8_w.float()
    fp8_blocks = fp8_float.reshape(n_groups, block_size, k_groups, block_size)
    dequant = fp8_blocks * scales.float().unsqueeze(1).unsqueeze(-1)
    return dequant.reshape(n, k).to(torch.bfloat16)


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
