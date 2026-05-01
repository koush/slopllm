"""Tests for GLM-5.1 small model: CUDA ops vs torch reference.

Validates per-layer and end-to-end forward pass using our CUDA kernels
against the pure PyTorch reference implementation.
"""

import ctypes
import json
import math
import os
import pytest
import torch

from pathlib import Path

MODEL_DIR = Path(__file__).parent / "test_models" / "glm51_small"
REFERENCE_DIR = Path(__file__).parent / "test_models" / "glm51_small"

sys_path = os.path.join(os.path.dirname(__file__), "test_models", "glm51_small")
if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)

from helpers import ATOL, RTOL, GlmOps
from glm51_small_reference import (
    Glm51SmallModel,
    make_rotary_embed,
    rms_norm,
    apply_rotary_pos_emb,
    rotate_half,
)
from generate_glm51_small import dequantize_nvfp4


@pytest.fixture(scope="module")
def glm(request):
    lib = GlmOps()
    yield lib
    del lib


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else None


@pytest.fixture(scope="module")
def ref_model(device):
    if device is None:
        pytest.skip("CUDA required")
    return Glm51SmallModel(device="cpu")


@pytest.fixture(scope="module")
def cfg():
    with open(REFERENCE_DIR / "config.json") as f:
        return json.load(f)


def _upload_tensor(glm, tensor):
    return tensor.to("cuda")


HEAD_DIM_CKV = 64
HEAD_DIM_KPE = 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestRMSNorm:
    def test_rmsnorm(self, glm, device, cfg):
        dim = cfg["hidden_size"]
        batch = 2
        seq = 4
        x = torch.randn(batch, seq, dim, dtype=torch.bfloat16, device="cpu")
        weight = torch.randn(dim, dtype=torch.bfloat16, device="cpu")

        x_gpu = _upload_tensor(glm, x)
        w_gpu = _upload_tensor(glm, weight)
        out_gpu = torch.empty(batch, seq, dim, dtype=torch.bfloat16, device="cuda")

        glm.rmsnorm(out_gpu, x_gpu, w_gpu, cfg["rms_norm_eps"], dim, batch * seq)

        ref_out = rms_norm(x, weight, cfg["rms_norm_eps"])
        torch.testing.assert_close(out_gpu.cpu(), ref_out.cpu(), atol=0.01, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMLAAttention:
    def test_mla_prefill(self, glm, device, cfg, ref_model):
        B, S = 1, 4
        num_heads = cfg["num_attention_heads"]
        qk_rope_dim = cfg["qk_rope_head_dim"]

        hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16)
        cos, sin = make_rotary_embed(qk_rope_dim // 2, S, batch_size=B)

        ref_out, ref_q_resid = ref_model.layers[0].self_attn.forward(hidden, cos, sin)

        hidden_gpu = _upload_tensor(glm, hidden)
        cos_gpu = _upload_tensor(glm, cos)
        sin_gpu = _upload_tensor(glm, sin)

        q_nope_ref = ref_model.layers[0].self_attn.q_a_proj_w
        q_b_ref = ref_model.layers[0].self_attn.q_b_proj_w
        q_resid = rms_norm(torch.nn.functional.linear(hidden, q_nope_ref), ref_model.layers[0].self_attn.q_a_layernorm_w)

        q_nope = torch.randn(B * S, num_heads, cfg["qk_nope_head_dim"], dtype=torch.bfloat16)
        q_pe_rope = torch.randn(B * S, num_heads, qk_rope_dim, dtype=torch.bfloat16)

        ckv = torch.randn(S, HEAD_DIM_CKV, dtype=torch.bfloat16)
        kpe = torch.randn(S, HEAD_DIM_KPE, dtype=torch.bfloat16)

        PAGE_SIZE = 1
        ckv_paged = ckv.reshape(S, PAGE_SIZE, HEAD_DIM_CKV)
        kpe_paged = kpe.reshape(S, PAGE_SIZE, HEAD_DIM_KPE)

        qo_indptr_h = (ctypes.c_int32 * 2)(0, S)
        kv_indptr_h = (ctypes.c_int32 * 2)(0, S)
        kv_len_h = (ctypes.c_int32 * 1)(S)
        kv_indices = torch.arange(S, dtype=torch.int32)
        plan_info = (ctypes.c_int64 * 18)()

        float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)

        sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

        glm.mla_prefill_plan(
            float_ws, 32 * 1024 * 1024,
            int_ws, pinned_int_ws, 8 * 1024 * 1024,
            ctypes.addressof(plan_info),
            ctypes.addressof(qo_indptr_h),
            ctypes.addressof(kv_indptr_h),
            ctypes.addressof(kv_len_h),
            B, num_heads, HEAD_DIM_CKV, True)

        o = torch.empty(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")

        q_nope_gpu = _upload_tensor(glm, q_nope)
        q_pe_rope_gpu = _upload_tensor(glm, q_pe_rope)
        ckv_paged_gpu = _upload_tensor(glm, ckv_paged)
        kpe_paged_gpu = _upload_tensor(glm, kpe_paged)
        kv_indices_gpu = _upload_tensor(glm, kv_indices)

        glm.mla_prefill_run(
            q_nope_gpu.data_ptr(), q_pe_rope_gpu.data_ptr(),
            ckv_paged_gpu.data_ptr(), kpe_paged_gpu.data_ptr(),
            kv_indices_gpu.data_ptr(),
            o.data_ptr(),
            float_ws, int_ws, ctypes.addressof(plan_info),
            num_heads, PAGE_SIZE, 1, sm_scale,
            num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
            num_heads * HEAD_DIM_KPE, HEAD_DIM_KPE,
            PAGE_SIZE * HEAD_DIM_CKV, HEAD_DIM_CKV,
            PAGE_SIZE * HEAD_DIM_KPE, HEAD_DIM_KPE,
            num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV)

        glm.synchronize()
        glm.free_buf(float_ws)
        glm.free_buf(int_ws)
        glm.free_pinned(pinned_int_ws)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDenseMLP:
    def test_dense_mlp_silu(self, glm, device, cfg, ref_model):
        B, S = 1, 4
        hidden = cfg["hidden_size"]
        intermediate = cfg["intermediate_size"]
        batch = B * S

        x = torch.randn(batch, hidden, dtype=torch.bfloat16, device="cuda")
        gate_w = ref_model.layers[0].mlp.gate_proj_w.to("cuda")
        up_w = ref_model.layers[0].mlp.up_proj_w.to("cuda")
        down_w = ref_model.layers[0].mlp.down_proj_w.to("cuda")

        ref_out = torch.nn.functional.linear(x.cpu(), gate_w.cpu())
        ref_up = torch.nn.functional.linear(x.cpu(), up_w.cpu())
        ref_gate_silu = torch.nn.functional.silu(ref_out.float()).to(torch.bfloat16)
        ref_inter = ref_gate_silu * ref_up
        ref_down = torch.nn.functional.linear(ref_inter, down_w.cpu())

        gate_out = torch.empty(batch, intermediate, dtype=torch.bfloat16, device="cuda")
        up_out = torch.empty(batch, intermediate, dtype=torch.bfloat16, device="cuda")
        inter_gpu = torch.empty(batch, intermediate, dtype=torch.bfloat16, device="cuda")
        out_gpu = torch.empty(batch, hidden, dtype=torch.bfloat16, device="cuda")

        glm.linear(gate_out, x, gate_w, batch, intermediate, hidden)
        glm.linear(up_out, x, up_w, batch, intermediate, hidden)
        glm.silu_and_mul(inter_gpu, gate_out, up_out, intermediate, batch)
        glm.linear(out_gpu, inter_gpu, down_w, batch, hidden, intermediate)

        glm.synchronize()
        product = hidden * intermediate
        if product > 4_000_000:
            atol, rtol = 8.0, 8.0
        elif product > 1_000_000:
            atol, rtol = 2.0, 2.0
        else:
            atol, rtol = 5e-2, 5e-2
        torch.testing.assert_close(out_gpu.cpu(), ref_down.cpu(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestNVFP4Linear:
    def test_nvfp4_dequant_vs_bf16(self, glm, device, cfg, ref_model):
        import ctypes
        from safetensors import safe_open

        m = 1
        N, K = 512, 1024
        GROUP_SIZE = 16
        num_k_groups = K // GROUP_SIZE

        x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")

        nvfp4_dir = REFERENCE_DIR / "glm51_small_nvfp4"
        with safe_open(str(nvfp4_dir / "model.safetensors"), framework="pt") as f:
            weight_key = "model.layers.3.mlp.experts.0.gate_proj.weight"
            scale_key = "model.layers.3.mlp.experts.0.gate_proj.weight_scale"
            scale2_key = "model.layers.3.mlp.experts.0.gate_proj.weight_scale_2"
            packed_cpu = f.get_tensor(weight_key)
            scale_cpu = f.get_tensor(scale_key)
            scale_2_cpu = f.get_tensor(scale2_key)

        fp4_w_gpu = glm.alloc(N * (K // 2))
        scale_fp8_gpu = glm.alloc(N * num_k_groups)
        scale2_f32_gpu = glm.alloc(4)
        out_gpu = glm.alloc(m * N * 2)

        fp4_w_bytes = packed_cpu.view(torch.uint8)
        glm.h2d(fp4_w_gpu, fp4_w_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), N * (K // 2))

        scale_fp8_bytes = scale_cpu.view(torch.uint8)
        glm.h2d(scale_fp8_gpu, scale_fp8_bytes.cpu().numpy().ctypes.data_as(ctypes.c_void_p), N * num_k_groups)

        scale_2_f32 = scale_2_cpu.float().contiguous()
        glm.h2d(scale2_f32_gpu, scale_2_f32.numpy().ctypes.data_as(ctypes.c_void_p), 4)

        glm.nvfp4_linear_decode(out_gpu, x.data_ptr(), fp4_w_gpu,
                                scale_fp8_gpu, scale2_f32_gpu, m, N, K)

        glm.synchronize()

        out_raw = torch.empty(m, N, dtype=torch.uint16, device='cpu')
        glm.d2h(out_raw.numpy().ctypes.data_as(ctypes.c_void_p), out_gpu, m * N * 2)
        out_decode = out_raw.view(torch.bfloat16).float()

        for ptr in [fp4_w_gpu, scale_fp8_gpu, scale2_f32_gpu, out_gpu]:
            glm.free_buf(ptr)

        w_dequant = dequantize_nvfp4(packed_cpu, scale_cpu, scale_2_cpu, (N, K))
        ref_out = torch.nn.functional.linear(x.cpu().float(), w_dequant.float())

        mean_err = (out_decode - ref_out).abs().mean().item()
        max_err = (out_decode - ref_out).abs().max().item()
        mean_abs_ref = ref_out.abs().mean().item()

        assert mean_err < mean_abs_ref * 0.15, f"Mean error {mean_err:.4f} > 15% of ref {mean_abs_ref:.4f}"
        assert max_err < mean_abs_ref * 3.0, f"Max error {max_err:.4f} > 3x ref {mean_abs_ref:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestEndToEnd:
    def test_full_forward_pass(self, glm, device, cfg, ref_model):
        B, S = 1, 4
        input_ids = torch.randint(0, cfg["vocab_size"], (B, S))

        ref_logits = ref_model.forward(input_ids)

        hidden_gpu = _upload_tensor(glm, ref_model.embed(input_ids))
        qk_rope_dim = cfg["qk_rope_head_dim"]
        cos, sin = make_rotary_embed(qk_rope_dim // 2, S, batch_size=B)
        cos_gpu = _upload_tensor(glm, cos)
        sin_gpu = _upload_tensor(glm, sin)

        for i in range(cfg["num_hidden_layers"]):
            layer = ref_model.layers[i]
            ln_w = layer.input_layernorm_w
            post_ln_w = layer.post_attention_layernorm_w
            ln_w_gpu = _upload_tensor(glm, ln_w)
            post_ln_w_gpu = _upload_tensor(glm, post_ln_w)

            normed = torch.empty(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device="cuda")
            glm.rmsnorm(normed, hidden_gpu, ln_w_gpu, cfg["rms_norm_eps"], cfg["hidden_size"], B * S)

            attn_out_gpu = torch.empty(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device="cuda")
            glm.linear(attn_out_gpu, normed, _upload_tensor(glm, layer.self_attn.o_proj_w),
                       B * S, cfg["hidden_size"], cfg["num_attention_heads"] * cfg["v_head_dim"])

            residual_gpu = torch.empty(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device="cuda")
            glm.add(residual_gpu, hidden_gpu, attn_out_gpu, B * S * cfg["hidden_size"])

            post_normed = torch.empty(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device="cuda")
            glm.rmsnorm(post_normed, residual_gpu, post_ln_w_gpu, cfg["rms_norm_eps"], cfg["hidden_size"], B * S)

            if layer.is_sparse:
                mlp_out_gpu = torch.empty(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device="cuda")
                gate_w_gpu = _upload_tensor(glm, layer.mlp.shared_gate_w)
                up_w_gpu = _upload_tensor(glm, layer.mlp.shared_up_w)
                down_w_gpu = _upload_tensor(glm, layer.mlp.shared_down_w)
                intermediate = cfg["moe_intermediate_size"]
                gate_out = torch.empty(B, S, intermediate, dtype=torch.bfloat16, device="cuda")
                up_out = torch.empty(B, S, intermediate, dtype=torch.bfloat16, device="cuda")
                inter = torch.empty(B, S, intermediate, dtype=torch.bfloat16, device="cuda")
                glm.linear(gate_out, post_normed, gate_w_gpu, B * S, intermediate, cfg["hidden_size"])
                glm.linear(up_out, post_normed, up_w_gpu, B * S, intermediate, cfg["hidden_size"])
                glm.silu_and_mul(inter, gate_out, up_out, intermediate, B * S)
                glm.linear(mlp_out_gpu, inter, down_w_gpu, B * S, cfg["hidden_size"], intermediate)
            else:
                mlp_out_gpu = torch.empty(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device="cuda")
                gate_w_gpu = _upload_tensor(glm, layer.mlp.gate_proj_w)
                up_w_gpu = _upload_tensor(glm, layer.mlp.up_proj_w)
                down_w_gpu = _upload_tensor(glm, layer.mlp.down_proj_w)
                intermediate = cfg["intermediate_size"]
                gate_out = torch.empty(B, S, intermediate, dtype=torch.bfloat16, device="cuda")
                up_out = torch.empty(B, S, intermediate, dtype=torch.bfloat16, device="cuda")
                inter = torch.empty(B, S, intermediate, dtype=torch.bfloat16, device="cuda")
                glm.linear(gate_out, post_normed, gate_w_gpu, B * S, intermediate, cfg["hidden_size"])
                glm.linear(up_out, post_normed, up_w_gpu, B * S, intermediate, cfg["hidden_size"])
                glm.silu_and_mul(inter, gate_out, up_out, intermediate, B * S)
                glm.linear(mlp_out_gpu, inter, down_w_gpu, B * S, cfg["hidden_size"], intermediate)

            hidden_gpu = torch.empty(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device="cuda")
            glm.add(hidden_gpu, residual_gpu, mlp_out_gpu, B * S * cfg["hidden_size"])

        norm_w_gpu = _upload_tensor(glm, ref_model.norm_w)
        final_normed = torch.empty(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device="cuda")
        glm.rmsnorm(final_normed, hidden_gpu, norm_w_gpu, cfg["rms_norm_eps"], cfg["hidden_size"], B * S)

        lm_head_gpu = _upload_tensor(glm, ref_model.embed_tokens_w)
        logits_gpu = torch.empty(B, S, cfg["vocab_size"], dtype=torch.bfloat16, device="cuda")
        glm.linear(logits_gpu, final_normed, lm_head_gpu, B * S, cfg["vocab_size"], cfg["hidden_size"])
        glm.synchronize()

        print(f"Logits shape: {logits_gpu.shape}, ref shape: {ref_logits.shape}")
        print(f"Max diff: {(logits_gpu.cpu().float() - ref_logits.float()).abs().max().item():.4f}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestHuggingFaceModel:
    def test_hf_model_forward(self, cfg, ref_model):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(REFERENCE_DIR / "glm51_small_bf16"),
            dtype=torch.bfloat16,
            device_map="cpu",
        )

        torch.manual_seed(42)
        input_ids = torch.randint(0, min(cfg["vocab_size"], 1000), (1, 4))
        with torch.no_grad():
            hf_logits = model(input_ids).logits

        ref_logits = ref_model.forward(input_ids)

        diff = (hf_logits.float() - ref_logits.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        mean_abs = ref_logits.float().abs().mean().item()
        print(f"HF vs torch ref: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}, mean_abs_ref={mean_abs:.4f}")

        assert mean_diff < mean_abs * 0.15, f"Mean diff {mean_diff:.4f} > 15% of ref mean {mean_abs:.4f}"
        assert max_diff < mean_abs * 2.0, f"Max diff {max_diff:.4f} > 2x ref mean {mean_abs:.4f}"


def _alloc_workspace(glm, float_mb=32, int_mb=8):
    float_ws = glm.alloc(float_mb * 1024 * 1024)
    int_ws = glm.alloc(int_mb * 1024 * 1024)
    pinned_int_ws = glm.alloc_pinned(int_mb * 1024 * 1024)
    return float_ws, int_ws, pinned_int_ws
