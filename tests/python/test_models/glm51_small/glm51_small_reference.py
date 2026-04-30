"""Torch reference forward pass for GLM-5.1 small model.

Loads BF16 weights from safetensors and implements the full forward pass
using pure PyTorch operations (no CUDA-specific code). This serves as the
ground truth for verifying our CUDA kernel implementations.

Architecture: MLA attention + DSA Indexer (weights only, not used in forward) + MoE MLP.
"""

import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

DIR = Path(__file__).parent


def rms_norm(x, weight, eps=1e-5):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def make_rotary_embed(dim_half, seq_len, theta=1000000.0, device="cpu", batch_size=1):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half * 2, 2, dtype=torch.float32, device=device) / (dim_half * 2)))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.bfloat16).unsqueeze(0).expand(batch_size, -1, -1)
    sin = emb.sin().to(torch.bfloat16).unsqueeze(0).expand(batch_size, -1, -1)
    return cos, sin


class MLAAttention:
    def __init__(self, weights, layer_idx, cfg):
        self.layer_idx = layer_idx
        self.num_heads = cfg["num_attention_heads"]
        self.qk_nope_dim = cfg["qk_nope_head_dim"]
        self.qk_rope_dim = cfg["qk_rope_head_dim"]
        self.qk_head_dim = cfg.get("qk_head_dim", self.qk_nope_dim + self.qk_rope_dim)
        self.kv_lora_rank = cfg["kv_lora_rank"]
        self.v_head_dim = cfg["v_head_dim"]
        self.scaling = self.qk_head_dim ** -0.5
        pfx = f"model.layers.{layer_idx}.self_attn"

        self.q_a_proj_w = weights[f"{pfx}.q_a_proj.weight"]
        self.q_a_layernorm_w = weights[f"{pfx}.q_a_layernorm.weight"]
        self.q_b_proj_w = weights[f"{pfx}.q_b_proj.weight"]
        self.kv_a_proj_w = weights[f"{pfx}.kv_a_proj_with_mqa.weight"]
        self.kv_a_layernorm_w = weights[f"{pfx}.kv_a_layernorm.weight"]
        self.kv_b_proj_w = weights[f"{pfx}.kv_b_proj.weight"]
        self.o_proj_w = weights[f"{pfx}.o_proj.weight"]

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        B, S, _ = hidden_states.shape

        q_resid = rms_norm(F.linear(hidden_states, self.q_a_proj_w), self.q_a_layernorm_w)
        query = F.linear(q_resid, self.q_b_proj_w).view(B, S, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = query.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

        compressed = F.linear(hidden_states, self.kv_a_proj_w)
        k_compressed, k_pe = compressed.split([self.kv_lora_rank, self.qk_rope_dim], dim=-1)
        k_compressed = rms_norm(k_compressed, self.kv_a_layernorm_w)
        kv_expanded = F.linear(k_compressed, self.kv_b_proj_w).view(B, S, self.num_heads, self.qk_nope_dim + self.v_head_dim)
        k_nope, value = kv_expanded.split([self.qk_nope_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        value = value.transpose(1, 2)
        k_pe = k_pe.view(B, 1, S, self.qk_rope_dim)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

        query = torch.cat([q_nope, q_pe], dim=-1)
        key = torch.cat([k_nope, k_pe], dim=-1)

        attn_w = (query @ key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_w = attn_w + attention_mask
        attn_w = F.softmax(attn_w.float(), dim=-1).to(query.dtype)
        out = (attn_w @ value).transpose(1, 2).reshape(B, S, -1)
        out = F.linear(out, self.o_proj_w)
        return out, q_resid

    def forward_absorbed_decode(self, q_nope_absorbed, q_pe_rope, ckv, kpe, positions, sm_scale):
        B, H, D_CKV = q_nope_absorbed.shape
        S = ckv.shape[0]
        D_KPE = q_pe_rope.shape[-1]

        dim_half = D_KPE // 2
        rope_theta = 1000000.0
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, D_KPE, 2, dtype=torch.float32, device=ckv.device) / D_KPE))
        freqs = torch.outer(positions.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = emb.cos().to(ckv.dtype)
        sin_emb = emb.sin().to(ckv.dtype)

        kpe_rope = (kpe * cos_emb.unsqueeze(0)) + (rotate_half(kpe) * sin_emb.unsqueeze(0))

        q_nope_3d = q_nope_absorbed.reshape(B * H, 1, D_CKV)
        q_pe_3d = q_pe_rope.reshape(B * H, 1, D_KPE)
        ckv_3d = ckv.reshape(B, 1, S, D_CKV).expand(B, H, S, D_CKV).reshape(B * H, S, D_CKV)
        kpe_3d = kpe_rope.reshape(B, 1, S, D_KPE).expand(B, H, S, D_KPE).reshape(B * H, S, D_KPE)

        score = torch.bmm(q_nope_3d, ckv_3d.transpose(1, 2)) + torch.bmm(q_pe_3d, kpe_3d.transpose(1, 2))
        score = score * sm_scale
        attn = F.softmax(score.float(), dim=-1).to(ckv.dtype)
        output = torch.bmm(attn, ckv_3d)
        return output.reshape(B, H, D_CKV)


class DenseMLP:
    def __init__(self, weights, layer_idx, cfg):
        pfx = f"model.layers.{layer_idx}.mlp"
        self.gate_proj_w = weights[f"{pfx}.gate_proj.weight"]
        self.up_proj_w = weights[f"{pfx}.up_proj.weight"]
        self.down_proj_w = weights[f"{pfx}.down_proj.weight"]

    def forward(self, hidden_states):
        gate = F.silu(F.linear(hidden_states, self.gate_proj_w))
        up = F.linear(hidden_states, self.up_proj_w)
        return F.linear(gate * up, self.down_proj_w)


class MoEMLP:
    def __init__(self, weights, layer_idx, cfg):
        self.num_experts = cfg["n_routed_experts"]
        self.top_k = cfg["num_experts_per_tok"]
        self.n_group = cfg["n_group"]
        self.topk_group = cfg["topk_group"]
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]
        hidden = cfg["hidden_size"]
        pfx = f"model.layers.{layer_idx}.mlp"

        self.gate_weight = weights[f"{pfx}.gate.weight"]
        self.e_score_correction_bias = weights[f"{pfx}.gate.e_score_correction_bias"]

        self.expert_gate = [weights[f"{pfx}.experts.{i}.gate_proj.weight"] for i in range(self.num_experts)]
        self.expert_up = [weights[f"{pfx}.experts.{i}.up_proj.weight"] for i in range(self.num_experts)]
        self.expert_down = [weights[f"{pfx}.experts.{i}.down_proj.weight"] for i in range(self.num_experts)]

        self.shared_gate_w = weights[f"{pfx}.shared_experts.gate_proj.weight"]
        self.shared_up_w = weights[f"{pfx}.shared_experts.up_proj.weight"]
        self.shared_down_w = weights[f"{pfx}.shared_experts.down_proj.weight"]

    def route(self, hidden_states):
        x = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(x.float(), self.gate_weight.float()).sigmoid()
        logits_corrected = logits + self.e_score_correction_bias.float()
        if self.n_group > 1:
            group_scores = logits_corrected.view(-1, self.n_group, self.num_experts // self.n_group).topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = group_scores.topk(self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(-1, self.n_group, self.num_experts // self.n_group).reshape(-1, self.num_experts)
            scores_masked = logits_corrected.masked_fill(~score_mask.bool(), 0.0)
            topk_idx = scores_masked.topk(self.top_k, dim=-1, sorted=False)[1]
        else:
            topk_idx = logits_corrected.topk(self.top_k, dim=-1, sorted=False)[1]
        topk_w = logits.gather(1, topk_idx)
        if self.norm_topk_prob:
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        topk_w = topk_w * self.routed_scaling_factor
        return topk_idx, topk_w

    def forward(self, hidden_states):
        residual = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_w = self.route(hidden_states)
        h = hidden_states.view(-1, hidden_states.shape[-1])
        out = torch.zeros_like(h)
        expert_mask = F.one_hot(topk_idx, self.num_experts).permute(2, 1, 0)

        for i in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[i])
            if token_idx.numel() == 0:
                continue
            cur = h[token_idx]
            gate = F.silu(F.linear(cur, self.expert_gate[i]))
            up = F.linear(cur, self.expert_up[i])
            down = F.linear(gate * up, self.expert_down[i])
            out.index_add_(0, token_idx, (down * topk_w[token_idx, top_k_pos, None]).to(out.dtype))

        h = out.view(*orig_shape)
        shared_gate = F.silu(F.linear(residual, self.shared_gate_w))
        shared_up = F.linear(residual, self.shared_up_w)
        shared = F.linear(shared_gate * shared_up, self.shared_down_w)
        return h + shared


class DecoderLayer:
    def __init__(self, weights, layer_idx, cfg):
        self.input_layernorm_w = weights[f"model.layers.{layer_idx}.input_layernorm.weight"]
        self.post_attention_layernorm_w = weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
        self.self_attn = MLAAttention(weights, layer_idx, cfg)
        mlp_types = cfg.get("mlp_layer_types", ["dense"] * min(3, cfg["num_hidden_layers"]) + ["sparse"] * (cfg["num_hidden_layers"] - 3))
        first_k = sum(1 for t in mlp_types if t == "dense")
        self.is_sparse = layer_idx >= first_k
        if self.is_sparse:
            self.mlp = MoEMLP(weights, layer_idx, cfg)
        else:
            self.mlp = DenseMLP(weights, layer_idx, cfg)

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.input_layernorm_w)
        attn_out, q_resid = self.self_attn.forward(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.post_attention_layernorm_w)
        hidden_states = self.mlp.forward(hidden_states)
        return hidden_states + residual


class Glm51SmallModel:
    def __init__(self, checkpoint_dir=None, device="cpu"):
        if checkpoint_dir is None:
            checkpoint_dir = DIR / "glm51_small_bf16"
        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.device = device
        self.weights = {}
        shard_path = os.path.join(checkpoint_dir, "model.safetensors")
        with safe_open(shard_path, framework="pt") as f:
            for name in f.keys():
                self.weights[name] = f.get_tensor(name).to(device)

        self.embed_tokens_w = self.weights["model.embed_tokens.weight"]
        self.norm_w = self.weights["model.norm.weight"]
        self.tie_word_embeddings = self.cfg.get("tie_word_embeddings", False)
        if not self.tie_word_embeddings:
            self.lm_head_w = self.weights["lm_head.weight"]

        self.layers = []
        for i in range(self.cfg["num_hidden_layers"]):
            self.layers.append(DecoderLayer(self.weights, i, self.cfg))

    def embed(self, input_ids):
        return F.embedding(input_ids, self.embed_tokens_w)

    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape
        hidden_states = self.embed(input_ids)

        qk_rope_dim = self.cfg["qk_rope_head_dim"]
        cos, sin = make_rotary_embed(qk_rope_dim // 2, S, device=self.device, batch_size=B)

        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, cos, sin, attention_mask)

        hidden_states = rms_norm(hidden_states, self.norm_w)
        if self.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens_w)
        else:
            logits = F.linear(hidden_states, self.lm_head_w)
        return logits

    def forward_hidden_states(self, hidden_states, cos=None, sin=None, attention_mask=None):
        B, S, _ = hidden_states.shape
        if cos is None or sin is None:
            qk_rope_dim = self.cfg["qk_rope_head_dim"]
            cos, sin = make_rotary_embed(qk_rope_dim // 2, S, device=self.device, batch_size=B)

        for i, layer in enumerate(self.layers):
            hidden_states = layer.forward(hidden_states, cos, sin, attention_mask)

        hidden_states = rms_norm(hidden_states, self.norm_w)
        if self.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens_w)
        else:
            logits = F.linear(hidden_states, self.lm_head_w)
        return logits

    def forward_layer(self, hidden_states, layer_idx, cos=None, sin=None, attention_mask=None):
        B, S, _ = hidden_states.shape
        if cos is None or sin is None:
            qk_rope_dim = self.cfg["qk_rope_head_dim"]
            cos, sin = make_rotary_embed(qk_rope_dim // 2, S, device=self.device, batch_size=B)
        return self.layers[layer_idx].forward(hidden_states, cos, sin, attention_mask)
