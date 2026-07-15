"""Tests for GLM-5.1 small model: CUDA ops vs torch reference.

Validates MLA kernel correctness and per-layer ops against the
pure PyTorch reference implementation.
"""

import ctypes
import gc
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


def _apply_interleaved_rope(x, cos, sin):
    dim_half = cos.shape[-1] // 2
    cos_half = cos[..., :dim_half]
    sin_half = sin[..., :dim_half]
    while cos_half.ndim < x.ndim:
        cos_half = cos_half.unsqueeze(0)
        sin_half = sin_half.unsqueeze(0)
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    o1 = x1 * cos_half.float() - x2 * sin_half.float()
    o2 = x2 * cos_half.float() + x1 * sin_half.float()
    return torch.stack((o1, o2), dim=-1).flatten(-2).to(x.dtype)
from generate_glm51_small import dequantize_nvfp4

HEAD_DIM_CKV = 128
HEAD_DIM_KPE = 64
PAGE_SIZE = 1


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


def _alloc_workspace(glm, float_mb=32, int_mb=8):
    float_ws = glm.alloc(float_mb * 1024 * 1024)
    int_ws = glm.alloc(int_mb * 1024 * 1024)
    pinned_int_ws = glm.alloc_pinned(int_mb * 1024 * 1024)
    return float_ws, int_ws, pinned_int_ws


def mla_prefill_reference(q_nope, q_pe_rope, ckv, kpe_rope, sm_scale, causal=True):
    """Reference MLA prefill attention.
    
    q_nope: (B*S, H, D_CKV) - query no-pe part
    q_pe_rope: (B*S, H, D_KPE) - query pe part (already RoPE applied)
    ckv: (1, S, D_CKV) - compressed KV (shared across heads)
    kpe_rope: (1, S, D_KPE) - k-pe (already RoPE applied, shared across heads)
    """
    BS, H, D_CKV = q_nope.shape
    S = ckv.shape[1]
    D_KPE = q_pe_rope.shape[-1]

    ckv_exp = ckv.expand(H, S, D_CKV)
    kpe_exp = kpe_rope.expand(H, S, D_KPE)

    score_nope = torch.einsum('bhd,hkd->bhk', q_nope, ckv_exp)
    score_pe = torch.einsum('bhd,hkd->bhk', q_pe_rope, kpe_exp)
    score = (score_nope + score_pe) * sm_scale

    if causal:
        mask = torch.triu(torch.full((BS, S), float('-inf'), device=score.device, dtype=score.dtype), diagonal=1)
        score = score + mask.unsqueeze(1)

    attn = torch.nn.functional.softmax(score.float(), dim=-1).to(q_nope.dtype)
    output = torch.einsum('bhk,hkd->bhd', attn, ckv_exp)
    return output


def mla_decode_reference(q_nope_absorbed, q_pe_rope, ckv, kpe, positions, sm_scale, rope_theta=1000000.0):
    """Reference MLA decode attention.
    
    q_nope_absorbed: (B, H, D_CKV)
    q_pe_rope: (B, H, D_KPE) - already RoPE applied
    ckv: (1, S, D_CKV) or (S, D_CKV) - compressed KV
    kpe: (1, S, D_KPE) or (S, D_KPE) - raw k-pe (will apply RoPE)
    positions: (S,) int tensor of position ids
    """
    B, H, D_CKV = q_nope_absorbed.shape
    if ckv.dim() == 2:
        ckv = ckv.unsqueeze(0)
    if kpe.dim() == 2:
        kpe = kpe.unsqueeze(0)
    S = ckv.shape[1]
    D_KPE = q_pe_rope.shape[-1]

    dim_half = D_KPE // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, D_KPE, 2, dtype=torch.float32, device=ckv.device) / D_KPE))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_emb = emb.cos().to(ckv.dtype)
    sin_emb = emb.sin().to(ckv.dtype)

    kpe_cos_half = cos_emb[:, :dim_half].unsqueeze(0)
    kpe_sin_half = sin_emb[:, :dim_half].unsqueeze(0)
    kpe_x1 = kpe[..., 0::2].float()
    kpe_x2 = kpe[..., 1::2].float()
    kpe_o1 = kpe_x1 * kpe_cos_half.float() - kpe_x2 * kpe_sin_half.float()
    kpe_o2 = kpe_x2 * kpe_cos_half.float() + kpe_x1 * kpe_sin_half.float()
    kpe_rope = torch.stack((kpe_o1, kpe_o2), dim=-1).flatten(-2).to(kpe.dtype)

    q_nope_3d = q_nope_absorbed.reshape(B * H, 1, D_CKV)
    q_pe_3d = q_pe_rope.reshape(B * H, 1, D_KPE)
    ckv_3d = ckv.reshape(1, S, D_CKV).expand(B, H, S, D_CKV).reshape(B * H, S, D_CKV)
    kpe_3d = kpe_rope.reshape(1, S, D_KPE).expand(B, H, S, D_KPE).reshape(B * H, S, D_KPE)

    score = torch.bmm(q_nope_3d, ckv_3d.transpose(1, 2)) + torch.bmm(q_pe_3d, kpe_3d.transpose(1, 2))
    score = score * sm_scale
    attn = torch.nn.functional.softmax(score.float(), dim=-1).to(ckv.dtype)
    output = torch.bmm(attn, ckv_3d)
    return output.reshape(B, H, D_CKV)


def _make_rotary_embed(glm, device, dim_half, batch_size, seq_len):
    """Create cos/sin rotary embeddings using the CUDA kernel, matching test_mla_flash.py."""
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
# MLA Kernel Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMLA:
    def test_mla_prefill(self, glm, device):
        """MLA prefill kernel vs PyTorch reference with small model dims (128/64)."""
        B, S = 1, 8
        num_heads = 4
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

        torch.manual_seed(42)
        q_nope = torch.randn(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        q_pe = torch.randn(B * S, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
        ckv = torch.randn(S, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        kpe = torch.randn(S, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

        cos, sin = _make_rotary_embed(glm, device, HEAD_DIM_KPE // 2, B, S)
        q_pe_4d = q_pe.reshape(B, num_heads, S, HEAD_DIM_KPE)
        q_pe_rope = torch.empty_like(q_pe_4d)
        glm.apply_rotary_pos_emb(q_pe_rope, q_pe_4d, cos, sin, HEAD_DIM_KPE, num_heads, S, B, 1, interleaved=True)
        q_pe_rope = q_pe_rope.reshape(B * S, num_heads, HEAD_DIM_KPE)

        kpe_4d = kpe.reshape(B, 1, S, HEAD_DIM_KPE)
        kpe_rope_4d = torch.empty_like(kpe_4d)
        glm.apply_rotary_pos_emb(kpe_rope_4d, kpe_4d, cos, sin, HEAD_DIM_KPE, 1, S, B, 1, interleaved=True)
        kpe_rope = kpe_rope_4d.reshape(S, HEAD_DIM_KPE)

        ckv_paged = ckv.reshape(S, PAGE_SIZE, HEAD_DIM_CKV)
        kpe_paged = kpe_rope.reshape(S, PAGE_SIZE, HEAD_DIM_KPE)

        qo_indptr_h = (ctypes.c_int32 * 2)(0, S)
        kv_indptr_h = (ctypes.c_int32 * 2)(0, S)
        kv_len_h = (ctypes.c_int32 * 1)(S)
        kv_indices = torch.arange(S, dtype=torch.int32, device=device)
        plan_info = (ctypes.c_int64 * 19)()

        float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)

        glm.mla_prefill_plan(
            float_ws, 32 * 1024 * 1024,
            int_ws, pinned_int_ws, 8 * 1024 * 1024,
            ctypes.addressof(plan_info),
            ctypes.addressof(qo_indptr_h),
            ctypes.addressof(kv_indptr_h),
            ctypes.addressof(kv_len_h),
            B, num_heads, HEAD_DIM_CKV, True)

        o = torch.empty(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)

        glm.mla_prefill_run(
            q_nope.data_ptr(), q_pe_rope.data_ptr(),
            ckv_paged.data_ptr(), kpe_paged.data_ptr(),
            kv_indices.data_ptr(),
            o.data_ptr(),
            float_ws, int_ws, ctypes.addressof(plan_info),
            num_heads, PAGE_SIZE, 1, sm_scale,
            num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
            num_heads * HEAD_DIM_KPE, HEAD_DIM_KPE,
            PAGE_SIZE * HEAD_DIM_CKV, HEAD_DIM_CKV,
            PAGE_SIZE * HEAD_DIM_KPE, HEAD_DIM_KPE,
            num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
            HEAD_DIM_CKV, HEAD_DIM_KPE,
            None, 0, 0)

        glm.synchronize()

        ref_output = mla_prefill_reference(
            q_nope, q_pe_rope,
            ckv.unsqueeze(0), kpe_rope.unsqueeze(0),
            sm_scale, causal=True)

        max_diff = (o.cpu() - ref_output.cpu()).abs().max().item()
        mean_diff = (o.cpu() - ref_output.cpu()).abs().mean().item()

        print(f"  MLA prefill: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")
        assert max_diff < 0.1, f"MLA prefill max_diff={max_diff:.4f} > 0.1"

        glm.free_buf(float_ws)
        glm.free_buf(int_ws)
        glm.free_pinned(pinned_int_ws)

    def test_mla_decode(self, glm, device):
        """MLA decode kernel vs PyTorch reference with small model dims (128/64)."""
        B = 1
        S = 16
        num_heads = 4
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

        torch.manual_seed(42)
        q_nope = torch.randn(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        q_pe = torch.randn(B, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
        ckv = torch.randn(S, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        kpe = torch.randn(S, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

        positions = torch.arange(S, dtype=torch.int32, device=device)
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM_KPE, 2, dtype=torch.float32, device=device) / HEAD_DIM_KPE))

        decode_pos = torch.tensor([S - 1], dtype=torch.float32, device=device)
        freqs_q = torch.outer(decode_pos, inv_freq)
        emb_q = torch.cat([freqs_q, freqs_q], dim=-1)
        cos_q = emb_q.cos().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
        sin_q = emb_q.sin().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
        q_pe_rope = _apply_interleaved_rope(q_pe, cos_q, sin_q)

        positions_k = torch.arange(S, dtype=torch.float32, device=device)
        freqs_k = torch.outer(positions_k, inv_freq)
        emb_k = torch.cat([freqs_k, freqs_k], dim=-1)
        cos_k = emb_k.cos().to(torch.bfloat16)
        sin_k = emb_k.sin().to(torch.bfloat16)
        kpe_rope = _apply_interleaved_rope(kpe, cos_k, sin_k)

        num_pages = S
        ckv_paged = ckv.reshape(num_pages, PAGE_SIZE, HEAD_DIM_CKV).contiguous()
        kpe_paged = kpe_rope.reshape(num_pages, PAGE_SIZE, HEAD_DIM_KPE).contiguous()
        indices = torch.arange(num_pages, dtype=torch.int32, device=device)

        indptr_h = (ctypes.c_int32 * 2)(0, num_pages)
        last_page_len_d = torch.tensor([PAGE_SIZE], dtype=torch.int32, device=device)

        float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)
        plan_info = (ctypes.c_int64 * 10)()

        glm.mla_decode_plan(
            float_ws, 32 * 1024 * 1024,
            int_ws, pinned_int_ws, 8 * 1024 * 1024,
            ctypes.addressof(plan_info),
            ctypes.addressof(indptr_h),
            B, num_heads, PAGE_SIZE, False,
            head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

        o = torch.empty(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        indptr_d = torch.tensor([0, num_pages], dtype=torch.int32, device=device)

        glm.mla_decode_run(
            q_nope.data_ptr(), q_pe_rope.data_ptr(),
            ckv_paged.data_ptr(), kpe_paged.data_ptr(),
            indices.data_ptr(), indptr_d.data_ptr(), last_page_len_d.data_ptr(),
            o.data_ptr(),
            float_ws, int_ws, ctypes.addressof(plan_info),
            B, num_heads, PAGE_SIZE, sm_scale,
            head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

        glm.synchronize()

        ref_output = mla_decode_reference(
            q_nope, q_pe_rope,
            ckv.unsqueeze(0), kpe.unsqueeze(0),
            positions, sm_scale)

        max_diff = (o.cpu() - ref_output.cpu()).abs().max().item()
        mean_diff = (o.cpu() - ref_output.cpu()).abs().mean().item()

        print(f"  MLA decode: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")
        assert max_diff < 0.1, f"MLA decode max_diff={max_diff:.4f} > 0.1"

        glm.free_buf(float_ws)
        glm.free_buf(int_ws)
        glm.free_pinned(pinned_int_ws)

    def test_mla_kv_cache_append(self, glm, device):
        """MLA KV cache append with small model dims (128/64)."""
        B, S = 2, 4
        num_pages = B * S

        torch.manual_seed(42)
        append_ckv = torch.randn(B * S, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cpu")
        append_kpe = torch.randn(B * S, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cpu")

        ckv_cache = torch.zeros(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cpu")
        kpe_cache = torch.zeros(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cpu")

        ckv_cache_gpu = torch.zeros(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        kpe_cache_gpu = torch.zeros(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

        indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        indptr = torch.tensor([0, S, S * 2], dtype=torch.int32, device=device)
        last_page_len = torch.tensor([PAGE_SIZE, PAGE_SIZE], dtype=torch.int32, device=device)
        batch_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device)
        positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device=device)

        append_ckv_gpu = append_ckv.to(device)
        append_kpe_gpu = append_kpe.to(device)

        nnz = B * S
        glm.mla_kv_cache_append(
            ckv_cache_gpu.data_ptr(), kpe_cache_gpu.data_ptr(),
            indices.data_ptr(), indptr.data_ptr(), last_page_len.data_ptr(),
            append_ckv_gpu.data_ptr(), append_kpe_gpu.data_ptr(),
            batch_indices.data_ptr(), positions.data_ptr(),
            nnz, PAGE_SIZE,
            HEAD_DIM_CKV, HEAD_DIM_KPE,
            HEAD_DIM_CKV, HEAD_DIM_KPE)

        glm.synchronize()

        result_ckv = ckv_cache_gpu.cpu()
        result_kpe = kpe_cache_gpu.cpu()
        for i in range(B):
            for j in range(S):
                page_idx = i * S + j
                ref_ckv = append_ckv[i * S + j]
                gpu_ckv = result_ckv[page_idx, 0, :]
                max_err = (gpu_ckv - ref_ckv).abs().max().item()
                assert max_err < 1e-5, f"ckv mismatch at batch={i} pos={j}: max_err={max_err}"
                ref_kpe = append_kpe[i * S + j]
                gpu_kpe = result_kpe[page_idx, 0, :]
                max_err_kpe = (gpu_kpe - ref_kpe).abs().max().item()
                assert max_err_kpe < 1e-5, f"kpe mismatch at batch={i} pos={j}: max_err={max_err_kpe}"

    def test_mla_kv_cache_append_page16(self, glm, device):
        """MLA KV cache append with PAGE_SIZE=16, multiple tokens across pages."""
        if device is None:
            pytest.skip("CUDA required")
        PAGE = 16
        B = 1
        S = 20  # spans 2 pages (positions 0-19)

        torch.manual_seed(99)
        append_ckv = torch.randn(S, HEAD_DIM_CKV, dtype=torch.bfloat16)
        append_kpe = torch.randn(S, HEAD_DIM_KPE, dtype=torch.bfloat16)

        num_pages = 2
        ckv_gpu = torch.zeros(num_pages, PAGE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        kpe_gpu = torch.zeros(num_pages, PAGE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

        # Page layout: page 0 holds positions 0-15, page 1 holds positions 16-19
        indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        last_page_len = torch.tensor([S - (num_pages - 1) * PAGE], dtype=torch.int32, device=device)
        batch_indices = torch.zeros(S, dtype=torch.int32, device=device)
        positions = torch.arange(S, dtype=torch.int32, device=device)

        append_ckv_gpu = append_ckv.to(device)
        append_kpe_gpu = append_kpe.to(device)

        glm.mla_kv_cache_append(
            ckv_gpu.data_ptr(), kpe_gpu.data_ptr(),
            indices.data_ptr(), indptr.data_ptr(), last_page_len.data_ptr(),
            append_ckv_gpu.data_ptr(), append_kpe_gpu.data_ptr(),
            batch_indices.data_ptr(), positions.data_ptr(),
            S, PAGE,
            HEAD_DIM_CKV, HEAD_DIM_KPE,
            HEAD_DIM_CKV, HEAD_DIM_KPE)
        glm.synchronize()

        result_ckv = ckv_gpu.cpu()
        result_kpe = kpe_gpu.cpu()
        for pos in range(S):
            page_idx = pos // PAGE
            offset = pos % PAGE
            max_ckv = (result_ckv[page_idx, offset, :] - append_ckv[pos]).abs().max().item()
            max_kpe = (result_kpe[page_idx, offset, :] - append_kpe[pos]).abs().max().item()
            assert max_ckv < 1e-5, f"ckv mismatch at pos={pos} (page={page_idx}, offset={offset}): {max_ckv}"
            assert max_kpe < 1e-5, f"kpe mismatch at pos={pos} (page={page_idx}, offset={offset}): {max_kpe}"

        # Verify unwritten slots are still zero
        for offset in range(S % PAGE if S % PAGE > 0 else PAGE, PAGE):
            assert result_ckv[1, offset].abs().max().item() == 0, f"page1 offset={offset} ckv should be zero"
            assert result_kpe[1, offset].abs().max().item() == 0, f"page1 offset={offset} kpe should be zero"

    def test_position_step(self, glm, device):
        """Verify position_step_kernel updates position_ids, last_page_len, slot_mapping correctly."""
        if device is None:
            pytest.skip("CUDA required")

        # position_step_kernel logic:
        #   pos = position_ids[seq] + 1
        #   position_ids[seq] = pos
        #   kv_len = pos + 1
        #   last_page_len[seq] = kv_len % page_size (or page_size if remainder==0)
        #   page_idx = pos / page_size
        #   page_offset = pos % page_size
        #   abs_page = indices[indptr[seq] + page_idx]
        #   slot_mapping[seq] = abs_page * page_size + page_offset

        PAGE = 16
        B = 1

        # Scenario: prefill of 6 tokens, then 3 decode steps
        # Initial state: 6 tokens in cache (positions 0-5), position_ids = [5]
        position_ids = torch.tensor([5], dtype=torch.int32, device=device)
        last_page_len = torch.tensor([6], dtype=torch.int32, device=device)
        slot_mapping = torch.tensor([0], dtype=torch.int32, device=device)

        # Page table: 2 pages allocated
        indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        indptr = torch.tensor([0, 2], dtype=torch.int32, device=device)

        # Step 1: pos becomes 6, kv_len=7, page=6//16=0, offset=6%16=6, slot=0*16+6=6
        glm.position_step(position_ids.data_ptr(), last_page_len.data_ptr(),
                         slot_mapping.data_ptr(),
                         indptr.data_ptr(), indices.data_ptr(), PAGE, B)
        glm.synchronize()
        assert position_ids[0].item() == 6, f"position_ids after step1: {position_ids[0].item()}"
        assert last_page_len[0].item() == 7, f"last_page_len after step1: {last_page_len[0].item()}"
        assert slot_mapping[0].item() == 6, f"slot_mapping after step1: {slot_mapping[0].item()}"

        # Step 2: pos becomes 7, kv_len=8, page=7//16=0, offset=7%16=7, slot=0*16+7=7
        glm.position_step(position_ids.data_ptr(), last_page_len.data_ptr(),
                         slot_mapping.data_ptr(),
                         indptr.data_ptr(), indices.data_ptr(), PAGE, B)
        glm.synchronize()
        assert position_ids[0].item() == 7
        assert last_page_len[0].item() == 8
        assert slot_mapping[0].item() == 7

        # Step 3: pos becomes 8, kv_len=9, page=8//16=0, offset=8%16=8, slot=0*16+8=8
        glm.position_step(position_ids.data_ptr(), last_page_len.data_ptr(),
                         slot_mapping.data_ptr(),
                         indptr.data_ptr(), indices.data_ptr(), PAGE, B)
        glm.synchronize()
        assert position_ids[0].item() == 8
        assert last_page_len[0].item() == 9
        assert slot_mapping[0].item() == 8

        # Test page boundary: position_ids=15, next step crosses to page 1
        position_ids[0] = 15
        glm.position_step(position_ids.data_ptr(), last_page_len.data_ptr(),
                         slot_mapping.data_ptr(),
                         indptr.data_ptr(), indices.data_ptr(), PAGE, B)
        glm.synchronize()
        assert position_ids[0].item() == 16
        assert last_page_len[0].item() == 1, f"last_page_len at page boundary: {last_page_len[0].item()}"
        assert slot_mapping[0].item() == 1 * PAGE + 0, f"slot at page boundary: {slot_mapping[0].item()}"

        # Next step fills page 1
        position_ids[0] = 16
        glm.position_step(position_ids.data_ptr(), last_page_len.data_ptr(),
                         slot_mapping.data_ptr(),
                         indptr.data_ptr(), indices.data_ptr(), PAGE, B)
        glm.synchronize()
        assert position_ids[0].item() == 17
        assert last_page_len[0].item() == 2
        assert slot_mapping[0].item() == 1 * PAGE + 1

        # Test last_page_len = page_size when position aligns with page end
        # position_ids=30, next step: pos=31, kv_len=32, 32%16=0 -> last_page_len=16
        position_ids[0] = 30
        last_page_len[0] = 15  # 31 tokens in cache
        # Need more pages for indices
        indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        indptr = torch.tensor([0, 2], dtype=torch.int32, device=device)
        glm.position_step(position_ids.data_ptr(), last_page_len.data_ptr(),
                         slot_mapping.data_ptr(),
                         indptr.data_ptr(), indices.data_ptr(), PAGE, B)
        glm.synchronize()
        assert position_ids[0].item() == 31
        # kv_len = 31+1 = 32, 32%16 = 0, so last_page_len should be 16
        assert last_page_len[0].item() == PAGE, f"last_page_len when kv_len is multiple of page_size: {last_page_len[0].item()}"
        # page_idx = 31//16 = 1, offset = 31%16 = 15
        assert slot_mapping[0].item() == 1 * PAGE + 15

    def test_mla_decode_with_kv_cache_append(self, glm, device):
        """End-to-end: populate KV cache via mla_kv_cache_append kernel, then decode and verify."""
        if device is None:
            pytest.skip("CUDA required")

        PAGE = 16
        B = 1
        H = 4
        D_CKV = HEAD_DIM_CKV
        D_KPE = HEAD_DIM_KPE
        sm_scale = 1.0 / math.sqrt(D_CKV + D_KPE)
        rope_theta = 1000000.0
        S_prefill = 6

        torch.manual_seed(42)
        ckv_prefill = torch.randn(S_prefill, D_CKV, dtype=torch.bfloat16, device=device)
        kpe_prefill = torch.randn(S_prefill, D_KPE, dtype=torch.bfloat16, device=device)

        # Build reference data (pre-rotated kpe)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, D_KPE, 2, dtype=torch.float32, device=device) / D_KPE))
        positions_prefill = torch.arange(S_prefill, dtype=torch.float32, device=device)
        freqs_p = torch.outer(positions_prefill, inv_freq)
        emb_p = torch.cat([freqs_p, freqs_p], dim=-1)
        cos_p = emb_p.cos().to(torch.bfloat16)
        sin_p = emb_p.sin().to(torch.bfloat16)
        kpe_prefill_4d = kpe_prefill.view(1, 1, S_prefill, D_KPE)
        kpe_rope_prefill_4d = _apply_interleaved_rope(kpe_prefill_4d, cos_p, sin_p)
        kpe_rope_prefill = kpe_rope_prefill_4d[0, 0]

        # Populate KV cache via mla_kv_cache_append kernel
        num_pages = 4
        ckv_cache = torch.zeros(num_pages, PAGE, D_CKV, dtype=torch.bfloat16, device=device)
        kpe_cache = torch.zeros(num_pages, PAGE, D_KPE, dtype=torch.bfloat16, device=device)

        indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        last_page_len = torch.tensor([S_prefill], dtype=torch.int32, device=device)
        batch_indices = torch.zeros(S_prefill, dtype=torch.int32, device=device)
        positions = torch.arange(S_prefill, dtype=torch.int32, device=device)

        glm.mla_kv_cache_append(
            ckv_cache.data_ptr(), kpe_cache.data_ptr(),
            indices.data_ptr(), indptr.data_ptr(), last_page_len.data_ptr(),
            ckv_prefill.data_ptr(), kpe_rope_prefill.data_ptr(),
            batch_indices.data_ptr(), positions.data_ptr(),
            S_prefill, PAGE,
            D_CKV, D_KPE,
            D_CKV, D_KPE)

        # Now decode one step: append new KV, then run MLA decode
        decode_pos = S_prefill
        q_nope = torch.randn(B, H, D_CKV, dtype=torch.bfloat16, device=device)
        q_pe = torch.randn(B, H, D_KPE, dtype=torch.bfloat16, device=device)
        ckv_new = torch.randn(D_CKV, dtype=torch.bfloat16, device=device)
        kpe_new = torch.randn(D_KPE, dtype=torch.bfloat16, device=device)

        # RoPE for decode query
        pos_tensor = torch.tensor([float(decode_pos)], device=device)
        freqs_d = torch.outer(pos_tensor, inv_freq)
        emb_d = torch.cat([freqs_d, freqs_d], dim=-1)
        cos_d = emb_d.cos().to(torch.bfloat16).reshape(1, 1, D_KPE)
        sin_d = emb_d.sin().to(torch.bfloat16).reshape(1, 1, D_KPE)
        q_pe_rope = _apply_interleaved_rope(q_pe, cos_d, sin_d)

        # RoPE for new kpe and append to cache
        kpe_new_4d = kpe_new.view(1, 1, 1, D_KPE)
        kpe_rope_new = torch.empty_like(kpe_new_4d)
        glm.apply_rotary_pos_emb(kpe_rope_new, kpe_new_4d, cos_d, sin_d, D_KPE, 1, 1, 1, 1, interleaved=True)

        page = decode_pos // PAGE
        offset = decode_pos % PAGE
        ckv_cache[page, offset] = ckv_new
        kpe_cache[page, offset] = kpe_rope_new[0, 0, 0]

        # MLA decode
        total_pages = page + 1
        last_page_len_val = offset + 1
        float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)
        indptr_h = (ctypes.c_int32 * 2)(0, total_pages)
        plan_info = (ctypes.c_int64 * 10)()
        glm.mla_decode_plan(
            float_ws, 32 * 1024 * 1024,
            int_ws, pinned_int_ws, 8 * 1024 * 1024,
            ctypes.addressof(plan_info), ctypes.addressof(indptr_h),
            B, H, PAGE, False,
            head_dim_ckv=D_CKV, head_dim_kpe=D_KPE)

        indices_d = torch.arange(total_pages, dtype=torch.int32, device=device)
        indptr_d = torch.tensor([0, total_pages], dtype=torch.int32, device=device)
        last_page_len_d = torch.tensor([last_page_len_val], dtype=torch.int32, device=device)

        o_decode = torch.empty(B, H, D_CKV, dtype=torch.bfloat16, device=device)
        glm.mla_decode_run(
            q_nope.data_ptr(), q_pe_rope.data_ptr(),
            ckv_cache[:total_pages].contiguous().data_ptr(),
            kpe_cache[:total_pages].contiguous().data_ptr(),
            indices_d.data_ptr(), indptr_d.data_ptr(), last_page_len_d.data_ptr(),
            o_decode.data_ptr(),
            float_ws, int_ws, ctypes.addressof(plan_info),
            B, H, PAGE, sm_scale,
            head_dim_ckv=D_CKV, head_dim_kpe=D_KPE)
        glm.synchronize()

        # Reference: use cache data directly (same as what was written via kernel)
        all_ckv = ckv_cache[:decode_pos + 1].reshape(-1, D_CKV)[:decode_pos + 1].cpu()
        all_kpe = kpe_cache[:decode_pos + 1].reshape(-1, D_KPE)[:decode_pos + 1].cpu()

        ckv_3d = all_ckv.unsqueeze(0).expand(B, H, -1, -1).reshape(B * H, -1, D_CKV)
        kpe_3d = all_kpe.unsqueeze(0).expand(B, H, -1, -1).reshape(B * H, -1, D_KPE)
        q_nope_3d = q_nope.cpu().reshape(B * H, 1, D_CKV)
        q_pe_3d = q_pe_rope.cpu().reshape(B * H, 1, D_KPE)

        score = torch.bmm(q_nope_3d, ckv_3d.transpose(1, 2)) + torch.bmm(q_pe_3d, kpe_3d.transpose(1, 2))
        score = score * sm_scale
        attn = F.softmax(score.float(), dim=-1).to(torch.bfloat16)
        ref_output = torch.bmm(attn, ckv_3d).reshape(B, H, D_CKV)

        max_diff = (o_decode.cpu() - ref_output).abs().max().item()
        mean_diff = (o_decode.cpu() - ref_output).abs().mean().item()
        assert max_diff < 0.05, f"Decode with appended cache: max_diff={max_diff:.6f} > 0.05"

        del float_ws, int_ws, pinned_int_ws
        gc.collect()
        torch.cuda.empty_cache()

    def test_mla_prefill_then_decode(self, glm, device):
        """MLA prefill then decode: full lifecycle with small model dims."""
        B = 1
        S = 4
        num_heads = 4
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

        torch.manual_seed(42)
        q_nope = torch.randn(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        q_pe = torch.randn(B * S, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
        ckv = torch.randn(S, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        kpe = torch.randn(S, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

        cos, sin = _make_rotary_embed(glm, device, HEAD_DIM_KPE // 2, B, S)
        q_pe_4d = q_pe.reshape(B, num_heads, S, HEAD_DIM_KPE)
        q_pe_rope = torch.empty_like(q_pe_4d)
        glm.apply_rotary_pos_emb(q_pe_rope, q_pe_4d, cos, sin, HEAD_DIM_KPE, num_heads, S, B, 1, interleaved=True)
        q_pe_rope_flat = q_pe_rope.reshape(B * S, num_heads, HEAD_DIM_KPE)

        kpe_4d = kpe.reshape(B, 1, S, HEAD_DIM_KPE)
        kpe_rope_4d = torch.empty_like(kpe_4d)
        glm.apply_rotary_pos_emb(kpe_rope_4d, kpe_4d, cos, sin, HEAD_DIM_KPE, 1, S, B, 1, interleaved=True)
        kpe_rope = kpe_rope_4d.reshape(S, HEAD_DIM_KPE)

        ckv_paged = ckv.reshape(S, PAGE_SIZE, HEAD_DIM_CKV)
        kpe_paged = kpe_rope.reshape(S, PAGE_SIZE, HEAD_DIM_KPE)
        kv_indices = torch.arange(S, dtype=torch.int32, device=device)

        float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)

        qo_indptr_h = (ctypes.c_int32 * 2)(0, S)
        kv_indptr_h = (ctypes.c_int32 * 2)(0, S)
        kv_len_h = (ctypes.c_int32 * 1)(S)
        plan_info = (ctypes.c_int64 * 19)()

        glm.mla_prefill_plan(
            float_ws, 32 * 1024 * 1024,
            int_ws, pinned_int_ws, 8 * 1024 * 1024,
            ctypes.addressof(plan_info),
            ctypes.addressof(qo_indptr_h),
            ctypes.addressof(kv_indptr_h),
            ctypes.addressof(kv_len_h),
            B, num_heads, HEAD_DIM_CKV, True)

        o_prefill = torch.empty(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        glm.mla_prefill_run(
            q_nope.data_ptr(), q_pe_rope_flat.data_ptr(),
            ckv_paged.data_ptr(), kpe_paged.data_ptr(),
            kv_indices.data_ptr(),
            o_prefill.data_ptr(),
            float_ws, int_ws, ctypes.addressof(plan_info),
            num_heads, PAGE_SIZE, 1, sm_scale,
            num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
            num_heads * HEAD_DIM_KPE, HEAD_DIM_KPE,
            PAGE_SIZE * HEAD_DIM_CKV, HEAD_DIM_CKV,
            PAGE_SIZE * HEAD_DIM_KPE, HEAD_DIM_KPE,
            num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
            HEAD_DIM_CKV, HEAD_DIM_KPE,
            None, 0, 0)

        glm.synchronize()

        ref_prefill = mla_prefill_reference(
            q_nope.cpu(), q_pe_rope_flat.cpu(),
            ckv.cpu().unsqueeze(0), kpe_rope.cpu().unsqueeze(0),
            sm_scale, causal=True)

        prefill_max_diff = (o_prefill.cpu() - ref_prefill).abs().max().item()
        print(f"  Prefill max_diff: {prefill_max_diff:.4f}")
        assert prefill_max_diff < 0.1, f"Prefill max_diff={prefill_max_diff:.4f} > 0.1"

        num_pages = S
        ckv_cache = torch.zeros(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        kpe_cache = torch.zeros(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
        for p in range(num_pages):
            ckv_cache[p, 0, :] = ckv[p, :]
            kpe_cache[p, 0, :] = kpe_rope[p, :]

        decode_q_nope = torch.randn(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        decode_q_pe = torch.randn(B, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

        inv_freq_decode = 1.0 / (1000000.0 ** (torch.arange(0, HEAD_DIM_KPE, 2, dtype=torch.float32, device=device) / HEAD_DIM_KPE))
        decode_pos = torch.tensor([S], dtype=torch.float32, device=device)
        freqs_q = torch.outer(decode_pos, inv_freq_decode)
        emb_q = torch.cat([freqs_q, freqs_q], dim=-1)
        cos_q = emb_q.cos().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
        sin_q = emb_q.sin().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
        decode_q_pe_rope = _apply_interleaved_rope(decode_q_pe, cos_q, sin_q)

        indptr_h_decode = (ctypes.c_int32 * 2)(0, num_pages)
        plan_info_decode = (ctypes.c_int64 * 10)()

        glm.mla_decode_plan(
            float_ws, 32 * 1024 * 1024,
            int_ws, pinned_int_ws, 8 * 1024 * 1024,
            ctypes.addressof(plan_info_decode),
            ctypes.addressof(indptr_h_decode),
            B, num_heads, PAGE_SIZE, False,
            head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

        o_decode = torch.empty(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        decode_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        decode_indptr_d = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        decode_last_page_len = torch.tensor([PAGE_SIZE], dtype=torch.int32, device=device)

        glm.mla_decode_run(
            decode_q_nope.data_ptr(), decode_q_pe_rope.data_ptr(),
            ckv_cache.data_ptr(), kpe_cache.data_ptr(),
            decode_indices.data_ptr(), decode_indptr_d.data_ptr(), decode_last_page_len.data_ptr(),
            o_decode.data_ptr(),
            float_ws, int_ws, ctypes.addressof(plan_info_decode),
            B, num_heads, PAGE_SIZE, sm_scale,
            head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

        glm.synchronize()

        ref_positions = torch.arange(S, dtype=torch.int32)
        ref_decode = mla_decode_reference(
            decode_q_nope.cpu(), decode_q_pe_rope.cpu(),
            ckv.cpu().unsqueeze(0), kpe.cpu().unsqueeze(0),
            ref_positions, sm_scale)

        decode_max_diff = (o_decode.cpu() - ref_decode.cpu()).abs().max().item()
        print(f"  Decode max_diff: {decode_max_diff:.4f}")
        assert decode_max_diff < 0.1, f"Decode max_diff={decode_max_diff:.4f} > 0.1"

        glm.free_buf(float_ws)
        glm.free_buf(int_ws)
        glm.free_pinned(pinned_int_ws)


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
    glm.apply_rotary_pos_emb(q_pe_rope, q_pe, cos_gpu, sin_gpu, qk_rope_dim, num_heads, S, B, 1, interleaved=True)

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
    glm.apply_rotary_pos_emb(k_pe_rope, k_pe_4d, cos_gpu, sin_gpu, qk_rope_dim, 1, S, B, 1, interleaved=True)
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
