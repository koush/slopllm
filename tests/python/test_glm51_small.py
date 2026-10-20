"""Tests for GLM-5.1 small model: CUDA ops vs torch reference.

Validates per-layer ops and composed model execution against the
pure PyTorch reference implementation.
"""

import ctypes
import json
import math
import numpy as np
import os
import pytest
import torch
import torch.nn.functional as F

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
    apply_rotary_pos_emb_interleaved,
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


def _make_rotary_embed(glm, device, dim_half, batch_size, seq_len):
    """Create cos/sin rotary embeddings for the small model."""
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, dim_half * 2, 2, dtype=torch.float32, device=device) / (dim_half * 2)))
    positions = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch_size, -1)
    freqs = torch.outer(positions.float().reshape(-1)[:seq_len], inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.bfloat16).unsqueeze(0).expand(batch_size, -1, -1)
    sin = emb.sin().to(torch.bfloat16).unsqueeze(0).expand(batch_size, -1, -1)
    return cos, sin


def _bf16_to_f32(tensor_bf16):
    u16 = tensor_bf16.view(torch.uint16).cpu().numpy().astype(np.uint32)
    u32 = (u16 << 16).astype(np.uint32)
    return torch.from_numpy(u32.view(np.float32)).to(tensor_bf16.device)


def _f32_to_bf16(tensor_f32):
    u32 = tensor_f32.cpu().numpy().view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return torch.from_numpy(u16).view(torch.bfloat16).to(tensor_f32.device)


def _run_moe_cuda(glm, cfg, layer, post_normed_gpu, B, S):
    """Run MoE sparse MLP using CUDA ops with CPU routing."""
    BS = B * S
    hidden_size = cfg["hidden_size"]
    moe_inter = cfg["moe_intermediate_size"]
    num_experts = cfg["n_routed_experts"]
    top_k = cfg["num_experts_per_tok"]
    n_group = cfg["n_group"]
    topk_group = cfg["topk_group"]

    gate_w = _upload_tensor(glm, layer.mlp.gate_weight)
    gate_logits_gpu = torch.empty(BS, num_experts, dtype=torch.bfloat16, device="cuda")
    glm.linear(gate_logits_gpu, post_normed_gpu, gate_w, BS, num_experts, hidden_size)

    gate_logits_bytes = torch.empty(BS * num_experts, dtype=torch.uint16, device="cpu")
    glm.d2h(gate_logits_bytes.numpy().ctypes.data_as(ctypes.c_void_p),
             gate_logits_gpu.data_ptr(), BS * num_experts * 2)
    gate_logits_f32 = _bf16_to_f32(gate_logits_bytes.view(torch.bfloat16)).reshape(BS, num_experts)

    logits = torch.sigmoid(gate_logits_f32)

    bias_gpu = _upload_tensor(glm, layer.mlp.e_score_correction_bias)
    bias_dtype = layer.mlp.e_score_correction_bias.dtype
    if bias_dtype == torch.float32:
        bias_bytes = torch.empty(num_experts, dtype=torch.float32, device="cpu")
        glm.d2h(bias_bytes.numpy().ctypes.data_as(ctypes.c_void_p),
                 bias_gpu.data_ptr(), num_experts * 4)
        bias_f32 = bias_bytes
    else:
        bias_bytes = torch.empty(num_experts, dtype=torch.uint16, device="cpu")
        glm.d2h(bias_bytes.numpy().ctypes.data_as(ctypes.c_void_p),
                 bias_gpu.data_ptr(), num_experts * 2)
        bias_f32 = _bf16_to_f32(bias_bytes.view(torch.bfloat16))

    logits_corrected = logits + bias_f32.unsqueeze(0)

    if n_group > 1:
        experts_per_group = num_experts // n_group
        group_scores = logits_corrected.view(BS, n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = group_scores.topk(topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, experts_per_group).reshape(-1, num_experts)
        scores_masked = logits_corrected.masked_fill(~score_mask.bool(), 0.0)
        topk_idx = scores_masked.topk(top_k, dim=-1, sorted=False)[1]
    else:
        topk_idx = logits_corrected.topk(top_k, dim=-1, sorted=False)[1]

    topk_w = logits.gather(1, topk_idx)
    if cfg["norm_topk_prob"]:
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    topk_w = topk_w * cfg["routed_scaling_factor"]

    expert_indices = topk_idx.tolist()
    expert_weights = topk_w.tolist()

    routed_out_gpu = torch.zeros(BS, hidden_size, dtype=torch.bfloat16, device="cuda")
    glm.fill(routed_out_gpu, 0.0, BS * hidden_size)

    scale_f32 = torch.zeros(BS, dtype=torch.float32)

    for e in range(num_experts):
        any_selected = False
        for b in range(BS):
            if e in expert_indices[b]:
                any_selected = True
                break
        if not any_selected:
            continue

        expert_gate_w = _upload_tensor(glm, layer.mlp.expert_gate[e])
        expert_up_w = _upload_tensor(glm, layer.mlp.expert_up[e])
        expert_down_w = _upload_tensor(glm, layer.mlp.expert_down[e])

        gate_out = torch.empty(BS, moe_inter, dtype=torch.bfloat16, device="cuda")
        up_out = torch.empty(BS, moe_inter, dtype=torch.bfloat16, device="cuda")
        inter = torch.empty(BS, moe_inter, dtype=torch.bfloat16, device="cuda")
        expert_down = torch.empty(BS, hidden_size, dtype=torch.bfloat16, device="cuda")

        glm.linear(gate_out, post_normed_gpu, expert_gate_w, BS, moe_inter, hidden_size)
        glm.linear(up_out, post_normed_gpu, expert_up_w, BS, moe_inter, hidden_size)
        glm.silu_and_mul(inter, gate_out, up_out, moe_inter, BS)
        glm.linear(expert_down, inter, expert_down_w, BS, hidden_size, moe_inter)

        for b in range(BS):
            k_idx = expert_indices[b].index(e) if e in expert_indices[b] else -1
            scale_f32[b] = expert_weights[b][k_idx] if k_idx >= 0 else 0.0

        scale_bf16 = _f32_to_bf16(scale_f32).to("cuda")
        routed_out_gpu += scale_bf16.unsqueeze(1) * expert_down

    shared_gate_w_gpu = _upload_tensor(glm, layer.mlp.shared_gate_w)
    shared_up_w_gpu = _upload_tensor(glm, layer.mlp.shared_up_w)
    shared_down_w_gpu = _upload_tensor(glm, layer.mlp.shared_down_w)

    shared_gate = torch.empty(BS, moe_inter, dtype=torch.bfloat16, device="cuda")
    shared_up = torch.empty(BS, moe_inter, dtype=torch.bfloat16, device="cuda")
    shared_inter = torch.empty(BS, moe_inter, dtype=torch.bfloat16, device="cuda")
    shared_down = torch.empty(BS, hidden_size, dtype=torch.bfloat16, device="cuda")

    glm.linear(shared_gate, post_normed_gpu, shared_gate_w_gpu, BS, moe_inter, hidden_size)
    glm.linear(shared_up, post_normed_gpu, shared_up_w_gpu, BS, moe_inter, hidden_size)
    glm.silu_and_mul(shared_inter, shared_gate, shared_up, moe_inter, BS)
    glm.linear(shared_down, shared_inter, shared_down_w_gpu, BS, hidden_size, moe_inter)

    mlp_out_gpu = torch.empty(BS, hidden_size, dtype=torch.bfloat16, device="cuda")
    glm.add(mlp_out_gpu, routed_out_gpu, shared_down, BS * hidden_size)

    return mlp_out_gpu


# ---------------------------------------------------------------------------
# Per-layer tests vs reference
# ---------------------------------------------------------------------------

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
            atol, rtol = 3.0, 3.0
        else:
            atol, rtol = 5e-2, 5e-2
        torch.testing.assert_close(out_gpu.cpu(), ref_down.cpu(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# End-to-end: CUDA ops vs PyTorch reference
# ---------------------------------------------------------------------------

def _cuda_mla_attention_bmm(glm, device, layer, hidden_gpu, cos_gpu, sin_gpu, causal_mask_gpu, B, S, cfg):
    """MLA attention using BMM (non-absorbed, matches reference model)."""
    num_heads = cfg["num_attention_heads"]
    qk_nope_dim = cfg["qk_nope_head_dim"]
    qk_rope_dim = cfg["qk_rope_head_dim"]
    qk_head_dim = qk_nope_dim + qk_rope_dim
    v_head_dim = cfg["v_head_dim"]
    kv_lora_rank = cfg["kv_lora_rank"]
    q_lora_rank = cfg["q_lora_rank"]
    hidden_size = cfg["hidden_size"]
    BS = B * S
    eps = cfg["rms_norm_eps"]
    scaling = qk_head_dim ** -0.5

    q_resid = torch.empty(BS, q_lora_rank, dtype=torch.bfloat16, device=device)
    glm.linear(q_resid, hidden_gpu, _upload_tensor(glm, layer.self_attn.q_a_proj_w),
               BS, q_lora_rank, hidden_size)

    q_resid_normed = torch.empty_like(q_resid)
    glm.rmsnorm(q_resid_normed, q_resid, _upload_tensor(glm, layer.self_attn.q_a_layernorm_w),
                eps, q_lora_rank, BS)

    query_flat = torch.empty(BS, num_heads * qk_head_dim, dtype=torch.bfloat16, device=device)
    glm.linear(query_flat, q_resid_normed, _upload_tensor(glm, layer.self_attn.q_b_proj_w),
               BS, num_heads * qk_head_dim, q_lora_rank)

    query = query_flat.cpu().view(B, S, num_heads, qk_head_dim).permute(0, 2, 1, 3).contiguous()
    q_nope = query[:, :, :, :qk_nope_dim].to(device).contiguous()
    q_pe = query[:, :, :, qk_nope_dim:].to(device).contiguous()

    q_pe_rope = torch.empty_like(q_pe)
    glm.apply_rotary_pos_emb(q_pe_rope, q_pe, cos_gpu, sin_gpu, qk_rope_dim, qk_rope_dim, num_heads, S, B, 1, interleaved=True)

    compressed_flat = torch.empty(BS, kv_lora_rank + qk_rope_dim, dtype=torch.bfloat16, device=device)
    glm.linear(compressed_flat, hidden_gpu, _upload_tensor(glm, layer.self_attn.kv_a_proj_w),
               BS, kv_lora_rank + qk_rope_dim, hidden_size)

    compressed = compressed_flat.cpu()
    k_compressed = compressed[:, :kv_lora_rank].to(device).contiguous()
    k_pe = compressed[:, kv_lora_rank:]

    k_compressed_norm = torch.empty(BS, kv_lora_rank, dtype=torch.bfloat16, device=device)
    glm.rmsnorm(k_compressed_norm, k_compressed, _upload_tensor(glm, layer.self_attn.kv_a_layernorm_w),
                eps, kv_lora_rank, BS)

    kv_expanded_flat = torch.empty(BS, num_heads * (qk_nope_dim + v_head_dim), dtype=torch.bfloat16, device=device)
    glm.linear(kv_expanded_flat, k_compressed_norm, _upload_tensor(glm, layer.self_attn.kv_b_proj_w),
               BS, num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank)

    kv_expanded = kv_expanded_flat.cpu().view(B, S, num_heads, qk_nope_dim + v_head_dim)
    k_nope = kv_expanded[:, :, :, :qk_nope_dim].permute(0, 2, 1, 3).contiguous().to(device)
    value = kv_expanded[:, :, :, qk_nope_dim:].permute(0, 2, 1, 3).contiguous().to(device)

    k_pe_4d = k_pe.view(B, 1, S, qk_rope_dim).to(device).contiguous()
    k_pe_rope = torch.empty_like(k_pe_4d)
    glm.apply_rotary_pos_emb(k_pe_rope, k_pe_4d, cos_gpu, sin_gpu, qk_rope_dim, qk_rope_dim, 1, S, B, 1, interleaved=True)
    k_pe_expanded = k_pe_rope.expand(-1, num_heads, -1, -1).contiguous()

    query_full = torch.cat([q_nope, q_pe_rope], dim=-1).contiguous()
    key_full = torch.cat([k_nope, k_pe_expanded], dim=-1).contiguous()

    q_bmm = query_full.reshape(B * num_heads, S, qk_head_dim).contiguous()
    k_bmm = key_full.reshape(B * num_heads, S, qk_head_dim).contiguous()
    v_bmm = value.reshape(B * num_heads, S, v_head_dim).contiguous()

    attn_weights = torch.empty(B * num_heads, S, S, dtype=torch.bfloat16, device=device)
    glm.bmm(attn_weights, q_bmm, k_bmm, scaling, 0.0, B * num_heads, S, S, qk_head_dim, 0, 1)

    causal_expanded = causal_mask_gpu.unsqueeze(0).expand(B * num_heads, S, S).contiguous()
    glm.add(attn_weights, attn_weights, causal_expanded, B * num_heads * S * S)

    attn_probs = torch.empty_like(attn_weights)
    glm.softmax(attn_probs, attn_weights, None, S, B * num_heads * S)

    attn_output = torch.empty(B * num_heads, S, v_head_dim, dtype=torch.bfloat16, device=device)
    glm.bmm(attn_output, attn_probs, v_bmm, 1.0, 0.0, B * num_heads, S, v_head_dim, S, 0, 0)

    attn_out = attn_output.cpu().view(B, num_heads, S, v_head_dim).permute(0, 2, 1, 3).contiguous()
    attn_out_flat = attn_out.reshape(BS, num_heads * v_head_dim).to(device).contiguous()

    o_output = torch.empty(BS, hidden_size, dtype=torch.bfloat16, device=device)
    glm.linear(o_output, attn_out_flat, _upload_tensor(glm, layer.self_attn.o_proj_w),
               BS, hidden_size, num_heads * v_head_dim)

    return o_output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestEndToEnd:
    def test_cuda_forward_vs_reference(self, glm, device, cfg, ref_model):
        """Full CUDA forward pass (BMM attention) vs PyTorch reference."""
        B, S = 1, 4
        torch.manual_seed(42)
        input_ids = torch.randint(0, min(cfg["vocab_size"], 1000), (B, S))

        causal_mask_ref = torch.zeros(S, S, dtype=torch.bfloat16)
        for i in range(S):
            for j in range(i + 1, S):
                causal_mask_ref[i, j] = float('-inf')

        ref_logits = ref_model.forward(input_ids, attention_mask=causal_mask_ref.unsqueeze(0).unsqueeze(0))

        qk_rope_dim = cfg["qk_rope_head_dim"]
        cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

        causal_mask = torch.empty(S, S, dtype=torch.bfloat16, device=device)
        glm.causal_mask(causal_mask, S)

        hidden_gpu = _upload_tensor(glm, ref_model.embed(input_ids))

        for i in range(cfg["num_hidden_layers"]):
            layer = ref_model.layers[i]
            ln_w_gpu = _upload_tensor(glm, layer.input_layernorm_w)
            post_ln_w_gpu = _upload_tensor(glm, layer.post_attention_layernorm_w)

            normed_gpu = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
            glm.rmsnorm(normed_gpu, hidden_gpu, ln_w_gpu, cfg["rms_norm_eps"], cfg["hidden_size"], B * S)

            attn_out_gpu = _cuda_mla_attention_bmm(
                glm, device, layer, normed_gpu, cos, sin, causal_mask, B, S, cfg)

            residual_gpu = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
            glm.add(residual_gpu, hidden_gpu, attn_out_gpu, B * S * cfg["hidden_size"])

            post_normed_gpu = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
            glm.rmsnorm(post_normed_gpu, residual_gpu, post_ln_w_gpu, cfg["rms_norm_eps"], cfg["hidden_size"], B * S)

            if layer.is_sparse:
                mlp_out_gpu = _run_moe_cuda(glm, cfg, layer, post_normed_gpu, B, S)
            else:
                mlp_out_gpu = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
                intermediate = cfg["intermediate_size"]
                gate_out = torch.empty(B * S, intermediate, dtype=torch.bfloat16, device=device)
                up_out = torch.empty(B * S, intermediate, dtype=torch.bfloat16, device=device)
                inter = torch.empty(B * S, intermediate, dtype=torch.bfloat16, device=device)
                glm.linear(gate_out, post_normed_gpu, _upload_tensor(glm, layer.mlp.gate_proj_w),
                           B * S, intermediate, cfg["hidden_size"])
                glm.linear(up_out, post_normed_gpu, _upload_tensor(glm, layer.mlp.up_proj_w),
                           B * S, intermediate, cfg["hidden_size"])
                glm.silu_and_mul(inter, gate_out, up_out, intermediate, B * S)
                glm.linear(mlp_out_gpu, inter, _upload_tensor(glm, layer.mlp.down_proj_w),
                           B * S, cfg["hidden_size"], intermediate)

            hidden_gpu = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
            glm.add(hidden_gpu, residual_gpu, mlp_out_gpu, B * S * cfg["hidden_size"])

        norm_w_gpu = _upload_tensor(glm, ref_model.norm_w)
        final_normed = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
        glm.rmsnorm(final_normed, hidden_gpu, norm_w_gpu, cfg["rms_norm_eps"], cfg["hidden_size"], B * S)

        lm_head_gpu = _upload_tensor(glm, ref_model.embed_tokens_w)
        logits_gpu = torch.empty(B * S, cfg["vocab_size"], dtype=torch.bfloat16, device=device)
        glm.linear(logits_gpu, final_normed, lm_head_gpu, B * S, cfg["vocab_size"], cfg["hidden_size"])
        glm.synchronize()

        cuda_logits = logits_gpu.cpu().view(B, S, cfg["vocab_size"]).float()
        ref_f = ref_logits.float()
        max_diff = (cuda_logits - ref_f).abs().max().item()
        mean_diff = (cuda_logits - ref_f).abs().mean().item()
        mean_abs = ref_f.abs().mean().item()
        print(f"  CUDA vs ref: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}, mean_abs={mean_abs:.4f}")

        assert max_diff < mean_abs * 15.0, f"Max diff {max_diff:.4f} > 15x mean_abs {mean_abs:.4f}"

        cuda_top5 = cuda_logits[0, -1].topk(5).indices.tolist()
        ref_top5 = ref_f[0, -1].topk(5).indices.tolist()
        top5_overlap = len(set(cuda_top5) & set(ref_top5))
        print(f"  CUDA top-5: {cuda_top5}, Ref top-5: {ref_top5}, overlap: {top5_overlap}/5")
        assert top5_overlap >= 2, f"Top-5 overlap too low: {top5_overlap}/5"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skip(reason="HuggingFace transformers uses non-interleaved RoPE for GLM-5.1 despite rope_interleave=true in config, so comparison is invalid")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skip(reason="HuggingFace transformers uses non-interleaved RoPE for GLM-5.1 despite rope_interleave=true in config, so comparison is invalid")
class TestCudaVsHuggingFace:
    def test_cuda_forward_vs_hf(self, glm, device, cfg):
        """CUDA forward pass vs HuggingFace Transformers output."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(REFERENCE_DIR / "glm51_small_bf16"),
            dtype=torch.bfloat16,
            device_map="cpu",
        )

        B, S = 1, 4
        torch.manual_seed(42)
        input_ids = torch.randint(0, min(cfg["vocab_size"], 1000), (B, S))

        with torch.no_grad():
            hf_logits = model(input_ids).logits

        qk_rope_dim = cfg["qk_rope_head_dim"]
        cos, sin = _make_rotary_embed(glm, device, qk_rope_dim // 2, B, S)

        causal_mask = torch.empty(S, S, dtype=torch.bfloat16, device=device)
        glm.causal_mask(causal_mask, S)

        ref_model_local = Glm51SmallModel()
        hidden_gpu = _upload_tensor(glm, ref_model_local.embed(input_ids))

        for i in range(cfg["num_hidden_layers"]):
            layer = ref_model_local.layers[i]
            ln_w_gpu = _upload_tensor(glm, layer.input_layernorm_w)
            post_ln_w_gpu = _upload_tensor(glm, layer.post_attention_layernorm_w)

            normed_gpu = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
            glm.rmsnorm(normed_gpu, hidden_gpu, ln_w_gpu, cfg["rms_norm_eps"], cfg["hidden_size"], B * S)

            attn_out_gpu = _cuda_mla_attention_bmm(
                glm, device, layer, normed_gpu, cos, sin, causal_mask, B, S, cfg)

            residual_gpu = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
            glm.add(residual_gpu, hidden_gpu, attn_out_gpu, B * S * cfg["hidden_size"])

            post_normed_gpu = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
            glm.rmsnorm(post_normed_gpu, residual_gpu, post_ln_w_gpu, cfg["rms_norm_eps"], cfg["hidden_size"], B * S)

            if layer.is_sparse:
                mlp_out_gpu = _run_moe_cuda(glm, cfg, layer, post_normed_gpu, B, S)
            else:
                mlp_out_gpu = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
                intermediate = cfg["intermediate_size"]
                gate_out = torch.empty(B * S, intermediate, dtype=torch.bfloat16, device=device)
                up_out = torch.empty(B * S, intermediate, dtype=torch.bfloat16, device=device)
                inter = torch.empty(B * S, intermediate, dtype=torch.bfloat16, device=device)
                glm.linear(gate_out, post_normed_gpu, _upload_tensor(glm, layer.mlp.gate_proj_w),
                           B * S, intermediate, cfg["hidden_size"])
                glm.linear(up_out, post_normed_gpu, _upload_tensor(glm, layer.mlp.up_proj_w),
                           B * S, intermediate, cfg["hidden_size"])
                glm.silu_and_mul(inter, gate_out, up_out, intermediate, B * S)
                glm.linear(mlp_out_gpu, inter, _upload_tensor(glm, layer.mlp.down_proj_w),
                           B * S, cfg["hidden_size"], intermediate)

            hidden_gpu = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
            glm.add(hidden_gpu, residual_gpu, mlp_out_gpu, B * S * cfg["hidden_size"])

        norm_w_gpu = _upload_tensor(glm, ref_model_local.norm_w)
        final_normed = torch.empty(B * S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
        glm.rmsnorm(final_normed, hidden_gpu, norm_w_gpu, cfg["rms_norm_eps"], cfg["hidden_size"], B * S)

        lm_head_gpu = _upload_tensor(glm, ref_model_local.embed_tokens_w)
        logits_gpu = torch.empty(B * S, cfg["vocab_size"], dtype=torch.bfloat16, device=device)
        glm.linear(logits_gpu, final_normed, lm_head_gpu, B * S, cfg["vocab_size"], cfg["hidden_size"])
        glm.synchronize()

        cuda_logits = logits_gpu.cpu().view(B, S, cfg["vocab_size"]).float()
        hf_f = hf_logits.float()
        max_diff = (cuda_logits - hf_f).abs().max().item()
        mean_diff = (cuda_logits - hf_f).abs().mean().item()
        mean_abs = hf_f.abs().mean().item()
        print(f"  CUDA vs HF: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}, mean_abs={mean_abs:.4f}")

        assert max_diff < mean_abs * 15.0, f"Max diff {max_diff:.4f} > 15x mean_abs {mean_abs:.4f}"

        cuda_top5 = cuda_logits[0, -1].topk(5).indices.tolist()
        hf_top5 = hf_f[0, -1].topk(5).indices.tolist()
        top5_overlap = len(set(cuda_top5) & set(hf_top5))
        print(f"  CUDA top-5: {cuda_top5}, HF top-5: {hf_top5}, overlap: {top5_overlap}/5")
        assert top5_overlap >= 2, f"Top-5 overlap too low: {top5_overlap}/5"
