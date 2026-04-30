#!/usr/bin/env python3
"""Generate GLM-5.1 small model checkpoints (BF16 and NVFP4) for testing.

Produces:
  glm51_small_bf16/model.safetensors       — all weights in BF16
  glm51_small_nvfp4/model.safetensors       — MLP weights in NVFP4 (U8 + FP8 scale + F32 scale_2)
  glm51_small_nvfp4/model.safetensors.index.json

Weight naming and NVFP4 format matches the real GLM-5.1 NVFP4 checkpoint exactly:
  - .weight        (uint8, two FP4 E2M1 values packed per byte, even=low nibble)
  - .weight_scale  (float8_e4m3fn, per-block scale for groups of 16)
  - .weight_scale_2 (float32, scalar global scale)
  - .input_scale   (float32, scalar activation scale)
  - gate_proj and up_proj share weight_scale_2 (NVIDIA modelopt convention)
"""

import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

DIR = Path(__file__).parent

FP4_E2M1_MAX = 6.0
FP4_E2M1_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP8_E4M3_MAX = 448.0
GROUP_SIZE = 16


def compute_weight_scale_2(weight: torch.Tensor) -> torch.Tensor:
    amax = weight.abs().max().float().clamp(min=torch.finfo(torch.float32).tiny)
    return (amax / (FP4_E2M1_MAX * FP8_E4M3_MAX)).squeeze()


def compute_weight_scale(weight: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    rows, cols = weight.shape
    w = weight.reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    block_amax = w.abs().amax(dim=-1).float()
    per_block = block_amax / (FP4_E2M1_MAX * scale_2.float())
    per_block = torch.where(per_block == 0, torch.ones_like(per_block), per_block)
    return per_block.to(torch.float8_e4m3fn)


def _cast_fp4(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=device, dtype=x.dtype)
    sign = torch.signbit(x).long()
    abs_x = x.abs()
    idx = torch.searchsorted(bounds, abs_x)
    return (idx + (sign << 3)).to(torch.uint8)


def pack_fp4_to_uint8(x_fp4: torch.Tensor) -> torch.Tensor:
    m, n = x_fp4.shape
    assert n % 2 == 0
    flat = x_fp4.reshape(-1, 2)
    packed = (flat[:, 1].to(torch.uint8) << 4) | flat[:, 0].to(torch.uint8)
    return packed.reshape(m, n // 2)


def quantize_weight_nvfp4(weight: torch.Tensor, scale_2_override: torch.Tensor = None):
    w = weight.float()
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
    m, half_n = packed.shape
    n = half_n * 2
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack([low, high], dim=-1).reshape(m, n).long()
    magnitude = unpacked & 0x07
    sign_bit = (unpacked >> 3) & 0x01
    fp4_vals = torch.tensor(FP4_E2M1_VALS, dtype=torch.float32, device=packed.device)
    signs = torch.tensor([1.0, -1.0], dtype=torch.float32, device=packed.device)
    dequant = fp4_vals[magnitude] * signs[sign_bit]
    rows, cols = orig_shape
    dequant = dequant.reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    scale_expanded = scale.float().unsqueeze(-1)
    dequant = dequant * scale_expanded * scale_2.float()
    return dequant.reshape(orig_shape)


class RMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float = 1e-5):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


def make_attn_weights(cfg, layer_idx, device):
    hidden = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    q_lora_rank = cfg["q_lora_rank"]
    kv_lora_rank = cfg["kv_lora_rank"]
    qk_nope_dim = cfg["qk_nope_head_dim"]
    qk_rope_dim = cfg["qk_rope_head_dim"]
    qk_head_dim = qk_nope_dim + qk_rope_dim
    v_head_dim = cfg["v_head_dim"]
    index_n_heads = cfg["index_n_heads"]
    index_head_dim = cfg["index_head_dim"]
    index_topk = cfg["index_topk"]
    pfx = f"model.layers.{layer_idx}.self_attn"

    weights = {}
    weights[f"{pfx}.q_a_proj.weight"] = torch.randn(q_lora_rank, hidden, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.q_a_layernorm.weight"] = torch.randn(q_lora_rank, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.q_b_proj.weight"] = torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.kv_a_proj_with_mqa.weight"] = torch.randn(kv_lora_rank + qk_rope_dim, hidden, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.kv_a_layernorm.weight"] = torch.randn(kv_lora_rank, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.kv_b_proj.weight"] = torch.randn(num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.o_proj.weight"] = torch.randn(hidden, num_heads * v_head_dim, dtype=torch.bfloat16, device=device)

    weights[f"{pfx}.indexer.wq_b.weight"] = torch.randn(index_n_heads * index_head_dim, q_lora_rank, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.indexer.wk.weight"] = torch.randn(index_head_dim, hidden, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.indexer.k_norm.weight"] = torch.randn(index_head_dim, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.indexer.k_norm.bias"] = torch.randn(index_head_dim, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.indexer.weights_proj.weight"] = torch.randn(index_n_heads, hidden, dtype=torch.bfloat16, device=device)

    return weights


def make_dense_mlp_weights(cfg, layer_idx, device):
    hidden = cfg["hidden_size"]
    intermediate = cfg["intermediate_size"]
    pfx = f"model.layers.{layer_idx}.mlp"

    weights = {}
    weights[f"{pfx}.gate_proj.weight"] = torch.randn(intermediate, hidden, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.up_proj.weight"] = torch.randn(intermediate, hidden, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.down_proj.weight"] = torch.randn(hidden, intermediate, dtype=torch.bfloat16, device=device)
    return weights


def make_moe_mlp_weights(cfg, layer_idx, device):
    hidden = cfg["hidden_size"]
    moe_inter = cfg["moe_intermediate_size"]
    n_experts = cfg["n_routed_experts"]
    pfx = f"model.layers.{layer_idx}.mlp"

    weights = {}
    weights[f"{pfx}.gate.weight"] = torch.randn(n_experts, hidden, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.gate.e_score_correction_bias"] = torch.randn(n_experts, dtype=torch.bfloat16, device=device)

    weights[f"{pfx}.shared_experts.gate_proj.weight"] = torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.shared_experts.up_proj.weight"] = torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device)
    weights[f"{pfx}.shared_experts.down_proj.weight"] = torch.randn(hidden, moe_inter, dtype=torch.bfloat16, device=device)

    for i in range(n_experts):
        epfx = f"{pfx}.experts.{i}"
        weights[f"{epfx}.gate_proj.weight"] = torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device)
        weights[f"{epfx}.up_proj.weight"] = torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device)
        weights[f"{epfx}.down_proj.weight"] = torch.randn(hidden, moe_inter, dtype=torch.bfloat16, device=device)

    return weights


def make_layer_weights(cfg, layer_idx, device):
    hidden = cfg["hidden_size"]
    weights = {}
    weights[f"model.layers.{layer_idx}.input_layernorm.weight"] = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = torch.randn(hidden, dtype=torch.bfloat16, device=device)

    weights.update(make_attn_weights(cfg, layer_idx, device))

    mlp_types = cfg.get("mlp_layer_types", ["dense"] * min(3, cfg["num_hidden_layers"]) + ["sparse"] * (cfg["num_hidden_layers"] - 3))
    first_k = sum(1 for t in mlp_types if t == "dense")
    if layer_idx < first_k:
        weights.update(make_dense_mlp_weights(cfg, layer_idx, device))
    else:
        weights.update(make_moe_mlp_weights(cfg, layer_idx, device))

    return weights


def quantize_mlp_nvfp4(bf16_weights, cfg, layer_idx, device):
    mlp_types = cfg.get("mlp_layer_types", ["dense"] * min(3, cfg["num_hidden_layers"]) + ["sparse"] * (cfg["num_hidden_layers"] - 3))
    first_k = sum(1 for t in mlp_types if t == "dense")
    if layer_idx < first_k:
        proj_names = ["gate_proj", "up_proj", "down_proj"]
        pfx = f"model.layers.{layer_idx}.mlp"
    else:
        proj_names = ["gate_proj", "up_proj", "down_proj"]
        nvfp4 = {}

        for expert_type in ["shared_experts"] + [f"experts.{i}" for i in range(cfg["n_routed_experts"])]:
            epfx = f"model.layers.{layer_idx}.mlp.{expert_type}"
            gate_key = f"{epfx}.gate_proj.weight"
            up_key = f"{epfx}.up_proj.weight"
            down_key = f"{epfx}.down_proj.weight"
            gate_w = bf16_weights[gate_key]
            up_w = bf16_weights[up_key]
            down_w = bf16_weights[down_key]

            gate_s2 = compute_weight_scale_2(gate_w)
            up_s2 = gate_s2.clone()
            down_s2 = compute_weight_scale_2(down_w)

            for proj_name, w, s2 in [("gate_proj", gate_w, gate_s2), ("up_proj", up_w, up_s2), ("down_proj", down_w, down_s2)]:
                packed, scale, _ = quantize_weight_nvfp4(w, scale_2_override=s2)
                nvfp4[f"{epfx}.{proj_name}.weight"] = packed
                nvfp4[f"{epfx}.{proj_name}.weight_scale"] = scale
                nvfp4[f"{epfx}.{proj_name}.weight_scale_2"] = s2
                nvfp4[f"{epfx}.{proj_name}.input_scale"] = torch.tensor(1.0, dtype=torch.float32, device=device)

        return nvfp4

    nvfp4 = {}
    pfx = f"model.layers.{layer_idx}.mlp"
    gate_key = f"{pfx}.gate_proj.weight"
    up_key = f"{pfx}.up_proj.weight"
    down_key = f"{pfx}.down_proj.weight"
    gate_w = bf16_weights[gate_key]
    up_w = bf16_weights[up_key]
    down_w = bf16_weights[down_key]

    gate_s2 = compute_weight_scale_2(gate_w)
    up_s2 = gate_s2.clone()
    down_s2 = compute_weight_scale_2(down_w)

    for proj_name, w, s2 in [("gate_proj", gate_w, gate_s2), ("up_proj", up_w, up_s2), ("down_proj", down_w, down_s2)]:
        packed, scale, _ = quantize_weight_nvfp4(w, scale_2_override=s2)
        nvfp4[f"{pfx}.{proj_name}.weight"] = packed
        nvfp4[f"{pfx}.{proj_name}.weight_scale"] = scale
        nvfp4[f"{pfx}.{proj_name}.weight_scale_2"] = s2
        nvfp4[f"{pfx}.{proj_name}.input_scale"] = torch.tensor(1.0, dtype=torch.float32, device=device)

    return nvfp4


def generate(cfg, seed=42, device="cpu"):
    torch.manual_seed(seed)
    num_layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]

    bf16_all = {}
    bf16_all["model.embed_tokens.weight"] = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device)
    bf16_all["model.norm.weight"] = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    if not cfg["tie_word_embeddings"]:
        bf16_all["lm_head.weight"] = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device)

    for i in range(num_layers):
        bf16_all.update(make_layer_weights(cfg, i, device))

    bf16_dir = DIR / "glm51_small_bf16"
    bf16_dir.mkdir(exist_ok=True)
    save_file(bf16_all, str(bf16_dir / "model.safetensors"))
    with open(bf16_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"BF16 checkpoint: {bf16_dir / 'model.safetensors'}")

    nvfp4_all = {}
    for name, tensor in bf16_all.items():
        if ".mlp." in name and ".weight" == name.split(".")[-1]:
            continue
        if ".mlp." in name and any(name.endswith(s) for s in [".weight_scale", ".weight_scale_2", ".input_scale"]):
            continue
        nvfp4_all[name] = tensor

    for i in range(num_layers):
        nvfp4_all.update(quantize_mlp_nvfp4(bf16_all, cfg, i, device))

    nvfp4_dir = DIR / "glm51_small_nvfp4"
    nvfp4_dir.mkdir(exist_ok=True)
    save_file(nvfp4_all, str(nvfp4_dir / "model.safetensors"))

    index = {"weight_map": {name: "model.safetensors" for name in nvfp4_all},
             "metadata": {"total_size": sum(t.nelement() * t.element_size() for t in nvfp4_all.values())}}
    with open(nvfp4_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    with open(nvfp4_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"NVFP4 checkpoint: {nvfp4_dir / 'model.safetensors'}")

    total_bf16 = sum(t.nelement() * t.element_size() for t in bf16_all.values())
    total_nvfp4 = sum(t.nelement() * t.element_size() for t in nvfp4_all.values())
    print(f"BF16 size: {total_bf16 / 1e6:.1f} MB")
    print(f"NVFP4 size: {total_nvfp4 / 1e6:.1f} MB")
    print(f"Compression ratio: {total_bf16 / total_nvfp4:.2f}x")

    return bf16_all, nvfp4_all


if __name__ == "__main__":
    with open(DIR / "config.json") as f:
        cfg = json.load(f)
    generate(cfg)
