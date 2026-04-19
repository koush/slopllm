from __future__ import annotations

import ctypes
import json
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from helpers import GlmOps, get_model_path
from flat_kv import FlatKVCache

BF16 = 2
I32 = 4
FLASH_TMP_SIZE = 32 * 1024 * 1024


def _f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    u32 = arr.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def _bf16_bytes_to_f32(data: bytes) -> np.ndarray:
    u16 = np.frombuffer(data, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _normalize_token_ids(input_ids: Any, tokenizer: Any) -> list[int]:
    if hasattr(input_ids, 'input_ids'):
        input_ids = input_ids['input_ids']
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, str):
        input_ids = tokenizer.encode(input_ids)
    if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return input_ids


class Qwen35Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    partial_rotary_factor: float
    mrope_section: list[int]
    rope_theta: float
    attn_output_gate: bool
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    layer_types: list[str]
    num_key_value_groups: int
    scaling: float
    num_full_attn_layers: int
    num_gdn_layers: int
    full_attn_layer_indices: list[int]
    full_attention_interval: int

    def __init__(self, d: dict[str, Any]) -> None:
        tc = d.get("text_config", d)
        self.hidden_size = tc["hidden_size"]
        self.intermediate_size = tc["intermediate_size"]
        self.num_hidden_layers = tc["num_hidden_layers"]
        self.rms_norm_eps = tc.get("rms_norm_eps", 1e-6)
        self.vocab_size = tc["vocab_size"]
        self.tie_word_embeddings = d.get("tie_word_embeddings", tc.get("tie_word_embeddings", False))
        self.num_attention_heads = tc["num_attention_heads"]
        self.num_key_value_heads = tc["num_key_value_heads"]
        self.head_dim = tc["head_dim"]
        self.partial_rotary_factor = tc.get("rope_parameters", {}).get("partial_rotary_factor", 1.0)
        self.mrope_section = tc.get("rope_parameters", {}).get("mrope_section", [])
        self.rope_theta = tc.get("rope_parameters", {}).get("rope_theta", 1000000)
        self.attn_output_gate = tc.get("attn_output_gate", False)
        self.linear_num_key_heads = tc.get("linear_num_key_heads", 16)
        self.linear_key_head_dim = tc.get("linear_key_head_dim", 128)
        self.linear_num_value_heads = tc.get("linear_num_value_heads", 16)
        self.linear_value_head_dim = tc.get("linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = tc.get("linear_conv_kernel_dim", 4)
        self.layer_types = tc.get("layer_types", [])
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.full_attention_interval = tc.get("full_attention_interval", 4)
        self.num_full_attn_layers = sum(1 for t in self.layer_types if t == "full_attention")
        self.num_gdn_layers = sum(1 for t in self.layer_types if t == "linear_attention")
        self.full_attn_layer_indices = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]


class Qwen35GdnState:
    def __init__(self, glm: GlmOps, cfg: Qwen35Config):
        self.glm = glm
        self.cfg = cfg
        lin_h = cfg.linear_num_key_heads
        lin_kd = cfg.linear_key_head_dim
        lin_vd = cfg.linear_value_head_dim
        conv_dim = lin_h * (lin_kd * 2 + lin_vd)
        kernel_size = cfg.linear_conv_kernel_dim
        self.conv_state_ptrs: list[int] = []
        self.recurrent_state_ptrs: list[int] = []
        for i in range(cfg.num_hidden_layers):
            if cfg.layer_types[i] == "linear_attention":
                cs = glm.alloc(conv_dim * (kernel_size - 1) * BF16)
                rs = glm.alloc(lin_h * lin_kd * lin_vd * 4)
                glm.fill(cs, 0.0, conv_dim * (kernel_size - 1))
                glm.fill(rs, 0.0, 2 * lin_h * lin_kd * lin_vd)
                self.conv_state_ptrs.append(cs)
                self.recurrent_state_ptrs.append(rs)
            else:
                self.conv_state_ptrs.append(0)
                self.recurrent_state_ptrs.append(0)

    def reset(self):
        cfg = self.cfg
        lin_h = cfg.linear_num_key_heads
        lin_kd = cfg.linear_key_head_dim
        lin_vd = cfg.linear_value_head_dim
        conv_dim = lin_h * (lin_kd * 2 + lin_vd)
        kernel_size = cfg.linear_conv_kernel_dim
        for i in range(cfg.num_hidden_layers):
            if cfg.layer_types[i] == "linear_attention":
                self.glm.fill(self.conv_state_ptrs[i], 0.0, conv_dim * (kernel_size - 1))
                self.glm.fill(self.recurrent_state_ptrs[i], 0.0, 2 * lin_h * lin_kd * lin_vd)

    def free(self):
        for p in self.conv_state_ptrs:
            if p:
                self.glm.free_buf(p)
        for p in self.recurrent_state_ptrs:
            if p:
                self.glm.free_buf(p)
        self.conv_state_ptrs = []
        self.recurrent_state_ptrs = []


class Qwen35Model:
    glm: GlmOps
    cfg: Qwen35Config
    weights: dict[str, int]
    max_batch: int
    max_seq_len: int
    device: int
    inv_freq: int
    _ws: dict[str, int]
    gdn_state: Qwen35GdnState

    def __init__(self, glm: GlmOps, config: Qwen35Config, weights: dict[str, int],
                 max_batch: int = 1, max_seq_len: int = 4096) -> None:
        self.glm = glm
        self.cfg = config
        self.weights = weights
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.device = glm.device

        rope_dim = int(config.head_dim * config.partial_rotary_factor)
        half_rope_dim = rope_dim // 2
        inv_freq_f32 = 1.0 / (
            config.rope_theta ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim)
        )
        inv_freq_bytes = _f32_to_bf16_bytes(inv_freq_f32)
        self.inv_freq = glm.alloc(len(inv_freq_bytes))
        glm.h2d(self.inv_freq, inv_freq_bytes)

        self._alloc_workspace(max_batch, max_seq_len)

    @classmethod
    def from_pretrained(cls, glm: GlmOps, repo_id: str = "Qwen/Qwen3.5-0.8B",
                        max_batch: int = 1, max_seq_len: int = 4096) -> Qwen35Model:
        model_dir = get_model_path(repo_id)
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        cfg = Qwen35Config(config)

        weights: dict[str, int] = {}
        prefix = "model.language_model."

        gemma_norm_suffixes = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "q_norm.weight",
            "k_norm.weight",
        )

        st_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
        for sf in st_files:
            st_path = os.path.join(model_dir, sf)
            with safe_open(st_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    weight_name = key[len(prefix):] if key.startswith(prefix) else key
                    if not weight_name.startswith("layers.") and not weight_name.startswith("embed_tokens") and not weight_name.startswith("norm.weight") and not weight_name.startswith("lm_head"):
                        continue

                    is_gemma_norm = (weight_name == "norm.weight" or
                                     any(weight_name.endswith(s) for s in gemma_norm_suffixes))

                    if is_gemma_norm:
                        t = (1.0 + t.float()).to(torch.bfloat16).contiguous()

                    if weight_name.endswith("A_log") or weight_name.endswith("dt_bias"):
                        t_f32 = t.float().contiguous()
                        nbytes = t_f32.numel() * 4
                        gpu_ptr = glm.alloc(nbytes)
                        glm.lib.glm_h2d(glm.ctx, ctypes.c_void_p(gpu_ptr), ctypes.c_void_p(t_f32.data_ptr()), nbytes)
                        weights[weight_name] = gpu_ptr
                    else:
                        t = t.to(torch.bfloat16).contiguous()
                        nbytes = t.numel() * BF16
                        gpu_ptr = glm.alloc(nbytes)
                        glm.lib.glm_h2d(glm.ctx, ctypes.c_void_p(gpu_ptr), ctypes.c_void_p(t.data_ptr()), nbytes)
                        weights[weight_name] = gpu_ptr

        if cfg.tie_word_embeddings and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["embed_tokens.weight"]

        return cls(glm, cfg, weights, max_batch, max_seq_len)

    def _alloc_workspace(self, B: int, S: int) -> None:
        cfg = self.cfg
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        inter = cfg.intermediate_size
        vs = cfg.vocab_size
        lin_h = cfg.linear_num_key_heads
        lin_kd = cfg.linear_key_head_dim
        lin_vd = cfg.linear_value_head_dim
        conv_dim = lin_h * (lin_kd * 2 + lin_vd)
        z_dim = lin_h * lin_vd
        q_total_dim = n_heads * hd
        BS = B * S
        glm = self.glm

        self._ws = {
            "hidden_a": glm.alloc(BS * hs * BF16),
            "hidden_b": glm.alloc(BS * hs * BF16),
            "normed": glm.alloc(BS * hs * BF16),
            "q_buf": glm.alloc(BS * q_total_dim * 2 * BF16),
            "k_buf": glm.alloc(BS * n_kv * hd * BF16),
            "v_buf": glm.alloc(BS * n_kv * hd * BF16),
            "q_normed": glm.alloc(BS * q_total_dim * BF16),
            "k_normed": glm.alloc(BS * n_kv * hd * BF16),
            "q_t": glm.alloc(B * n_heads * S * hd * BF16),
            "k_t": glm.alloc(B * n_kv * S * hd * BF16),
            "v_t": glm.alloc(B * n_kv * S * hd * BF16),
            "q_rope": glm.alloc(B * n_heads * S * hd * BF16),
            "k_rope": glm.alloc(B * n_kv * S * hd * BF16),
            "flash_out": glm.alloc(B * n_heads * S * hd * BF16),
            "flash_tmp": glm.alloc(FLASH_TMP_SIZE),
            "o_proj_buf": glm.alloc(BS * hs * BF16),
            "gate_proj_buf": glm.alloc(BS * inter * BF16),
            "up_buf": glm.alloc(BS * inter * BF16),
            "silu_buf": glm.alloc(BS * inter * BF16),
            "down_buf": glm.alloc(BS * hs * BF16),
            "cos": glm.alloc(B * S * hd * BF16),
            "sin": glm.alloc(B * S * hd * BF16),
            "position_ids": glm.alloc(B * S * I32),
            "logits_buf": glm.alloc(B * vs * BF16),
            "hidden_last": glm.alloc(B * hs * BF16),
            "last_idx": glm.alloc(B * I32),
            "argmax_idx": glm.alloc(B * I32),
            "input_ids_buf": glm.alloc(BS * I32),
            "decode_id": glm.alloc(I32),
            "gdn_qkv_linear": glm.alloc(BS * conv_dim * BF16),
            "gdn_conv_out": glm.alloc(S * conv_dim * BF16),
            "gdn_a": glm.alloc(BS * lin_h * BF16),
            "gdn_b": glm.alloc(BS * lin_h * BF16),
            "gdn_z": glm.alloc(BS * z_dim * BF16),
            "gdn_q": glm.alloc(S * lin_h * lin_kd * BF16),
            "gdn_k": glm.alloc(S * lin_h * lin_kd * BF16),
            "gdn_v": glm.alloc(S * lin_h * lin_vd * BF16),
            "gdn_out": glm.alloc(S * lin_h * lin_vd * BF16),
            "gdn_gated": glm.alloc(S * lin_h * lin_vd * BF16),
            "cu_seqlens": glm.alloc(2 * I32),
        }

    def free(self) -> None:
        glm = self.glm
        for ptr in self._ws.values():
            glm.free_buf(ptr)
        glm.free_buf(self.inv_freq)
        freed_ptrs = set()
        for ptr in self.weights.values():
            if isinstance(ptr, int) and ptr not in freed_ptrs:
                glm.free_buf(ptr)
                freed_ptrs.add(ptr)
        self._ws = {}
        self.weights = {}

    def create_gdn_state(self) -> Qwen35GdnState:
        return Qwen35GdnState(self.glm, self.cfg)

    def create_flat_kv_cache(self) -> FlatKVCache:
        return FlatKVCache(self.glm, self.cfg.num_key_value_heads, self.cfg.head_dim,
                           self.cfg.num_full_attn_layers, self.max_batch, self.max_seq_len)

    def __del__(self) -> None:
        if hasattr(self, '_ws') and self._ws:
            self.free()

    def _read_logits(self, ptr: int, count: int) -> np.ndarray:
        nbytes = count * BF16
        buf = ctypes.create_string_buffer(nbytes)
        self.glm.lib.glm_d2h(self.glm.ctx, buf, ctypes.c_void_p(ptr), nbytes)
        return np.frombuffer(buf.raw, dtype=np.uint16)

    def _final_norm_and_logits(self, count: int, src: Optional[int] = None) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        vs = cfg.vocab_size
        if src is None:
            src = self._ws["hidden_a"]
        glm.rmsnorm(self._ws["normed"], src,
                     self.weights["norm.weight"],
                     cfg.rms_norm_eps, hs, count)
        glm.linear(self._ws["logits_buf"], self._ws["normed"],
                    self.weights["lm_head.weight"],
                    count, vs, hs)

    def _extract_last_logits(self, count: int, last_indices_np: np.ndarray) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        glm.h2d(self._ws["last_idx"], last_indices_np.tobytes())
        glm.index_select(self._ws["hidden_last"], self._ws["hidden_a"],
                          self._ws["last_idx"], hs, count)
        self._final_norm_and_logits(count, src=self._ws["hidden_last"])

    def _full_attn_cache_idx(self, layer_idx: int) -> int:
        idx = 0
        for i in range(layer_idx):
            if self.cfg.layer_types[i] == "full_attention":
                idx += 1
        return idx

    def _gdn_layer_prefill(self, layer_idx: int, S: int, gdn_state: Qwen35GdnState) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        lin_h = cfg.linear_num_key_heads
        lin_kd = cfg.linear_key_head_dim
        lin_vd = cfg.linear_value_head_dim
        conv_dim = lin_h * (lin_kd * 2 + lin_vd)
        z_dim = lin_h * lin_vd
        pfx = f"layers.{layer_idx}.linear_attn"
        BS = S
        ws = self._ws

        glm.rmsnorm(ws["normed"], ws["hidden_a"],
                     self.weights[f"layers.{layer_idx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        glm.linear(ws["gdn_qkv_linear"], ws["normed"],
                    self.weights[f"{pfx}.in_proj_qkv.weight"],
                    BS, conv_dim, hs)
        glm.linear(ws["gdn_a"], ws["normed"],
                    self.weights[f"{pfx}.in_proj_a.weight"],
                    BS, lin_h, hs)
        glm.linear(ws["gdn_b"], ws["normed"],
                    self.weights[f"{pfx}.in_proj_b.weight"],
                    BS, lin_h, hs)
        glm.linear(ws["gdn_z"], ws["normed"],
                    self.weights[f"{pfx}.in_proj_z.weight"],
                    BS, z_dim, hs)

        conv_state = gdn_state.conv_state_ptrs[layer_idx]
        recurrent_state = gdn_state.recurrent_state_ptrs[layer_idx]
        kernel_size = cfg.linear_conv_kernel_dim

        glm.causal_conv1d(ws["gdn_conv_out"], conv_state,
                           ws["gdn_qkv_linear"],
                           self.weights[f"{pfx}.conv1d.weight"],
                           ws["cu_seqlens"], conv_dim, S, kernel_size,
                           batch_size=1, conv_state_stride=conv_dim * (kernel_size - 1),
                           ch_stride=1, seq_stride=conv_dim)

        glm.gdn_prefill(ws["gdn_out"], recurrent_state,
                         ws["gdn_conv_out"],
                         ws["gdn_a"], ws["gdn_b"],
                         self.weights[f"{pfx}.A_log"],
                         self.weights[f"{pfx}.dt_bias"],
                         ws["cu_seqlens"], S, lin_h, lin_kd, lin_vd,
                         1, lin_h * lin_kd * lin_vd,
                         qkv_ch_stride=1, qkv_seq_stride=conv_dim)

        glm.rmsnorm_gated(ws["gdn_gated"], ws["gdn_out"], ws["gdn_z"],
                           self.weights[f"{pfx}.norm.weight"],
                           cfg.rms_norm_eps, lin_vd, S * lin_h)

        glm.linear(ws["o_proj_buf"], ws["gdn_gated"],
                    self.weights[f"{pfx}.out_proj.weight"],
                    BS, hs, z_dim)

        glm.add(ws["hidden_b"], ws["hidden_a"], ws["o_proj_buf"], BS * hs)
        glm.rmsnorm(ws["normed"], ws["hidden_b"],
                     self.weights[f"layers.{layer_idx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)
        self._mlp(f"layers.{layer_idx}", BS)
        glm.add(ws["hidden_a"], ws["hidden_b"], ws["down_buf"], BS * hs)

    def _gdn_layer_decode(self, layer_idx: int, gdn_state: Qwen35GdnState) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        lin_h = cfg.linear_num_key_heads
        lin_kd = cfg.linear_key_head_dim
        lin_vd = cfg.linear_value_head_dim
        conv_dim = lin_h * (lin_kd * 2 + lin_vd)
        z_dim = lin_h * lin_vd
        pfx = f"layers.{layer_idx}.linear_attn"
        BS = 1
        ws = self._ws

        glm.rmsnorm(ws["normed"], ws["hidden_a"],
                     self.weights[f"layers.{layer_idx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        glm.linear(ws["gdn_qkv_linear"], ws["normed"],
                    self.weights[f"{pfx}.in_proj_qkv.weight"],
                    BS, conv_dim, hs)
        glm.linear(ws["gdn_a"], ws["normed"],
                    self.weights[f"{pfx}.in_proj_a.weight"],
                    BS, lin_h, hs)
        glm.linear(ws["gdn_b"], ws["normed"],
                    self.weights[f"{pfx}.in_proj_b.weight"],
                    BS, lin_h, hs)
        glm.linear(ws["gdn_z"], ws["normed"],
                    self.weights[f"{pfx}.in_proj_z.weight"],
                    BS, z_dim, hs)

        conv_state = gdn_state.conv_state_ptrs[layer_idx]
        recurrent_state = gdn_state.recurrent_state_ptrs[layer_idx]
        kernel_size = cfg.linear_conv_kernel_dim

        glm.causal_conv1d_update(ws["gdn_qkv_linear"], conv_state,
                                  ws["gdn_qkv_linear"],
                                  self.weights[f"{pfx}.conv1d.weight"],
                                  conv_dim, kernel_size)

        glm.gdn_recurrent_step(ws["gdn_out"], recurrent_state,
                                ws["gdn_qkv_linear"],
                                ws["gdn_a"], ws["gdn_b"],
                                self.weights[f"{pfx}.A_log"],
                                self.weights[f"{pfx}.dt_bias"],
                                lin_h, lin_kd, lin_vd, 1,
                                lin_h * lin_kd * lin_vd,
                                qkv_ch_stride=1, qkv_seq_stride=conv_dim)

        glm.rmsnorm_gated(ws["gdn_gated"], ws["gdn_out"], ws["gdn_z"],
                           self.weights[f"{pfx}.norm.weight"],
                           cfg.rms_norm_eps, lin_vd, lin_h)

        glm.linear(ws["o_proj_buf"], ws["gdn_gated"],
                    self.weights[f"{pfx}.out_proj.weight"],
                    BS, hs, z_dim)

        glm.add(ws["hidden_b"], ws["hidden_a"], ws["o_proj_buf"], BS * hs)
        glm.rmsnorm(ws["normed"], ws["hidden_b"],
                     self.weights[f"layers.{layer_idx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)
        self._mlp(f"layers.{layer_idx}", BS)
        glm.add(ws["hidden_a"], ws["hidden_b"], ws["down_buf"], BS * hs)

    def _full_attn_layer_prefill_flash(self, layer_idx: int, B: int, S: int, cache: FlatKVCache) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        max_S = self.max_seq_len
        BS = B * S
        cache_idx = self._full_attn_cache_idx(layer_idx)
        pfx = f"layers.{layer_idx}.self_attn"
        q_total_dim = n_heads * hd
        ws = self._ws

        glm.rmsnorm(ws["normed"], ws["hidden_a"],
                     self.weights[f"layers.{layer_idx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        glm.linear(ws["q_buf"], ws["normed"],
                    self.weights[f"{pfx}.q_proj.weight"],
                    BS, q_total_dim * 2, hs)

        glm.linear(ws["k_buf"], ws["normed"],
                    self.weights[f"{pfx}.k_proj.weight"],
                    BS, n_kv * hd, hs)
        glm.linear(ws["v_buf"], ws["normed"],
                    self.weights[f"{pfx}.v_proj.weight"],
                    BS, n_kv * hd, hs)

        rope_dim = int(hd * cfg.partial_rotary_factor)
        glm.fused_norm_rope(ws["q_rope"], ws["q_buf"],
                            self.weights[f"{pfx}.q_norm.weight"],
                            ws["cos"], ws["sin"],
                            cfg.rms_norm_eps, rope_dim, hd, n_heads, S, B, hd * 2)
        glm.fused_norm_rope(ws["k_rope"], ws["k_buf"],
                            self.weights[f"{pfx}.k_norm.weight"],
                            ws["cos"], ws["sin"],
                            cfg.rms_norm_eps, rope_dim, hd, n_kv, S, B)

        glm.transpose_4d(ws["v_t"], ws["v_buf"],
                          B, S, n_kv, hd, 0, 2, 1, 3)

        kv_stride_h = max_S * hd
        kv_stride_n = hd
        for h in range(n_kv):
            src_off = h * S * hd * BF16
            dst_off = (h * max_S * hd) * BF16
            glm.memcpy(cache.k_data[cache_idx] + dst_off,
                        ws["k_rope"] + src_off,
                        S * hd * BF16)
            glm.memcpy(cache.v_data[cache_idx] + dst_off,
                        ws["v_t"] + src_off,
                        S * hd * BF16)

        glm.flash_prefill(
            ws["q_rope"],
            cache.k_data[cache_idx],
            cache.v_data[cache_idx],
            ws["flash_out"],
            ws["flash_tmp"],
            S, S,
            n_heads, n_kv, hd,
            hd, S * hd,
            kv_stride_n, kv_stride_h,
            kv_stride_n, kv_stride_h,
            1, 1,
            cfg.scaling
        )

        if cfg.attn_output_gate:
            glm.gate_sigmoid_mul(ws["flash_out"], ws["q_buf"], BS, n_heads, hd)

        glm.linear(ws["o_proj_buf"], ws["flash_out"],
                    self.weights[f"{pfx}.o_proj.weight"],
                    BS, hs, n_heads * hd)

        glm.add(ws["hidden_b"], ws["hidden_a"], ws["o_proj_buf"], BS * hs)
        glm.rmsnorm(ws["normed"], ws["hidden_b"],
                     self.weights[f"layers.{layer_idx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)
        self._mlp(f"layers.{layer_idx}", BS)
        glm.add(ws["hidden_a"], ws["hidden_b"], ws["down_buf"], BS * hs)

    def _full_attn_layer_decode_flash(self, layer_idx: int, cached_len: int, cache: FlatKVCache) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        max_S = self.max_seq_len
        BS = 1
        S = 1
        total_len = cached_len + 1
        cache_idx = self._full_attn_cache_idx(layer_idx)
        pfx = f"layers.{layer_idx}.self_attn"
        q_total_dim = n_heads * hd
        ws = self._ws

        glm.rmsnorm(ws["normed"], ws["hidden_a"],
                     self.weights[f"layers.{layer_idx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        glm.linear(ws["q_buf"], ws["normed"],
                    self.weights[f"{pfx}.q_proj.weight"],
                    BS, q_total_dim * 2, hs)

        glm.linear(ws["k_buf"], ws["normed"],
                    self.weights[f"{pfx}.k_proj.weight"],
                    BS, n_kv * hd, hs)
        glm.linear(ws["v_buf"], ws["normed"],
                    self.weights[f"{pfx}.v_proj.weight"],
                    BS, n_kv * hd, hs)

        rope_dim = int(hd * cfg.partial_rotary_factor)
        glm.fused_norm_rope(ws["q_rope"], ws["q_buf"],
                            self.weights[f"{pfx}.q_norm.weight"],
                            ws["cos"], ws["sin"],
                            cfg.rms_norm_eps, rope_dim, hd, n_heads, S, BS, hd * 2)
        glm.fused_norm_rope(ws["k_rope"], ws["k_buf"],
                            self.weights[f"{pfx}.k_norm.weight"],
                            ws["cos"], ws["sin"],
                            cfg.rms_norm_eps, rope_dim, hd, n_kv, S, BS)

        kv_stride_h = max_S * hd
        kv_stride_n = hd
        for h in range(n_kv):
            src_off = h * hd * BF16
            dst_off = (h * max_S * hd + cached_len * hd) * BF16
            glm.memcpy(cache.k_data[cache_idx] + dst_off,
                        ws["k_rope"] + src_off,
                        hd * BF16)
            glm.memcpy(cache.v_data[cache_idx] + dst_off,
                        ws["v_t"] + src_off,
                        hd * BF16)

        glm.flash_decode(
            ws["q_rope"],
            cache.k_data[cache_idx],
            cache.v_data[cache_idx],
            ws["flash_out"],
            ws["flash_tmp"],
            total_len,
            n_heads, n_kv, hd,
            hd, S * hd,
            kv_stride_n, kv_stride_h,
            cfg.scaling
        )

        if cfg.attn_output_gate:
            glm.gate_sigmoid_mul(ws["flash_out"], ws["q_buf"], BS, n_heads, hd)

        glm.linear(ws["o_proj_buf"], ws["flash_out"],
                    self.weights[f"{pfx}.o_proj.weight"],
                    BS, hs, n_heads * hd)

        glm.add(ws["hidden_b"], ws["hidden_a"], ws["o_proj_buf"], BS * hs)
        glm.rmsnorm(ws["normed"], ws["hidden_b"],
                     self.weights[f"layers.{layer_idx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)
        self._mlp(f"layers.{layer_idx}", BS)
        glm.add(ws["hidden_a"], ws["hidden_b"], ws["down_buf"], BS * hs)

    def _mlp(self, pfx: str, BS: int) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        inter = cfg.intermediate_size
        ws = self._ws

        glm.linear(ws["gate_proj_buf"], ws["normed"],
                    self.weights[f"{pfx}.mlp.gate_proj.weight"],
                    BS, inter, hs)
        glm.linear(ws["up_buf"], ws["normed"],
                    self.weights[f"{pfx}.mlp.up_proj.weight"],
                    BS, inter, hs)
        glm.silu_and_mul(ws["silu_buf"], ws["gate_proj_buf"],
                          ws["up_buf"], inter, BS)
        glm.linear(ws["down_buf"], ws["silu_buf"],
                    self.weights[f"{pfx}.mlp.down_proj.weight"],
                    BS, hs, inter)

    def prefill(self, input_ids: torch.Tensor, cache: FlatKVCache, gdn_state: Qwen35GdnState) -> torch.Tensor:
        B, S = input_ids.shape
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        hd = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        vs = cfg.vocab_size
        BS = B * S

        assert B <= self.max_batch and S <= self.max_seq_len
        assert cache.cache_pos == 0

        gdn_state.reset()

        cu_seqlens_np = np.array([0, S], dtype=np.int32)
        glm.h2d(self._ws["cu_seqlens"], cu_seqlens_np.tobytes())

        ids_np = input_ids.cpu().numpy().astype(np.int32).flatten()
        glm.h2d(self._ws["input_ids_buf"], ids_np.tobytes())
        glm.embedding(self._ws["hidden_a"], self.weights["embed_tokens.weight"],
                       self._ws["input_ids_buf"], hs, BS)

        rope_dim = int(hd * cfg.partial_rotary_factor)
        glm.arange(self._ws["position_ids"], 0, 1, S)
        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              rope_dim // 2, B, S)

        for i in range(cfg.num_hidden_layers):
            if cfg.layer_types[i] == "linear_attention":
                self._gdn_layer_prefill(i, S, gdn_state)
            else:
                self._full_attn_layer_prefill_flash(i, B, S, cache)

        last_idx_np = np.array([S - 1], dtype=np.int32)
        self._extract_last_logits(1, last_idx_np)
        cache.cache_pos = S

        logits_u16 = self._read_logits(self._ws["logits_buf"], vs)
        logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(1, vs)
        return torch.from_numpy(logits_f32.copy())

    def _decode_token(self, token_id: int, cache: FlatKVCache, gdn_state: Qwen35GdnState) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        hd = cfg.head_dim
        vs = cfg.vocab_size
        B = 1
        S = 1
        BS = 1
        cached_len = cache.cache_pos

        assert cached_len > 0
        assert cached_len + S <= self.max_seq_len

        ids_np = np.array([token_id], dtype=np.int32)
        glm.h2d(self._ws["decode_id"], ids_np.tobytes())
        glm.embedding(self._ws["hidden_a"], self.weights["embed_tokens.weight"],
                       self._ws["decode_id"], hs, BS)

        rope_dim = int(hd * cfg.partial_rotary_factor)
        glm.arange(self._ws["position_ids"], cached_len, 0, 1)
        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              rope_dim // 2, B, S)

        for i in range(cfg.num_hidden_layers):
            if cfg.layer_types[i] == "linear_attention":
                self._gdn_layer_decode(i, gdn_state)
            else:
                self._full_attn_layer_decode_flash(i, cached_len, cache)

        self._final_norm_and_logits(BS)
        cache.cache_pos = cached_len + S

    def decode(self, input_ids: torch.Tensor, cache: FlatKVCache, gdn_state: Qwen35GdnState) -> torch.Tensor:
        B, S = input_ids.shape
        assert B == 1 and S == 1
        vs = self.cfg.vocab_size
        token_id = input_ids[0, 0].item()
        self._decode_token(token_id, cache, gdn_state)
        logits_u16 = self._read_logits(self._ws["logits_buf"], vs)
        logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(1, 1, vs)
        return torch.from_numpy(logits_f32.copy())

    def _argmax_logits(self, ptr: int, count: int) -> int:
        self.glm.argmax(self._ws["argmax_idx"], ptr, count)
        buf = ctypes.create_string_buffer(I32)
        self.glm.lib.glm_d2h(self.glm.ctx, buf, ctypes.c_void_p(self._ws["argmax_idx"]), I32)
        return int(np.frombuffer(buf.raw, dtype=np.int32)[0])

    def generate_tokens(self, input_ids: torch.Tensor, cache: FlatKVCache,
                        gdn_state: Qwen35GdnState,
                        max_new_tokens: int = 100,
                        eos_token_ids: Optional[set[int]] = None) -> Any:
        if eos_token_ids is None:
            eos_token_ids = {248044}
        B, S = input_ids.shape
        assert B == 1
        vs = self.cfg.vocab_size
        cache.reset()
        gdn_state.reset()
        logits = self.prefill(input_ids, cache, gdn_state)
        next_token = self._argmax_logits(self._ws["logits_buf"], vs)
        yield next_token
        for _ in range(max_new_tokens - 1):
            if next_token in eos_token_ids:
                break
            self._decode_token(next_token, cache, gdn_state)
            next_token = self._argmax_logits(self._ws["logits_buf"], vs)
            yield next_token


def torch_rmsnorm_gated(x, z, weight, eps):
    x_f = x.float()
    z_f = z.float()
    w_f = weight.float()
    rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = x_f * rms * w_f
    result = normed * torch.nn.functional.silu(z_f)
    return result.to(x.dtype)


def torch_gemma_rmsnorm(x, weight, eps):
    rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((1.0 + weight.float()) * x.float() * rms).to(x.dtype)


def torch_gdn_recurrent_step(q, k, v, beta, g, state, d_k, d_v):
    B, H, dk = q.shape
    scale = 1.0 / (dk ** 0.5)
    q_norm = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + 1e-8) * scale
    k_norm = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
    h = state.clone()
    h = h * g.unsqueeze(-1).unsqueeze(-1).exp()
    v_old = (h * k_norm.unsqueeze(-1)).sum(dim=2)
    delta = beta.unsqueeze(-1) * (v - v_old)
    h = h + k_norm.unsqueeze(-1) * delta.unsqueeze(-2)
    o = (h * q_norm.unsqueeze(-1)).sum(dim=2)
    return o, h


def torch_causal_conv1d(x, weight, kernel_size):
    B, C, S = x.shape
    out = torch.zeros_like(x)
    for c in range(C):
        for t in range(S):
            s = 0.0
            for kk in range(kernel_size):
                xt = t - (kernel_size - 1) + kk
                if xt >= 0:
                    s += weight[c, kk].item() * x[0, c, xt].item()
            out[0, c, t] = torch.nn.functional.silu(torch.tensor(s)).item()
    return out


class Qwen35TorchModel:
    def __init__(self, model_dir: str, device: torch.device):
        self.device = device
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        self.cfg = Qwen35Config(config)
        self.weights: dict[str, torch.Tensor] = {}
        prefix = "model.language_model."
        st_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
        for sf in st_files:
            st_path = os.path.join(model_dir, sf)
            with safe_open(st_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    weight_name = key[len(prefix):] if key.startswith(prefix) else key
                    if not weight_name.startswith("layers.") and not weight_name.startswith("embed_tokens") and not weight_name.startswith("norm.weight") and not weight_name.startswith("lm_head"):
                        continue
                    if weight_name.endswith("A_log"):
                        self.weights[weight_name] = t.float().to(device=device)
                    else:
                        self.weights[weight_name] = t.to(device=device, dtype=torch.bfloat16)

        if self.cfg.tie_word_embeddings and "lm_head.weight" not in self.weights:
            self.weights["lm_head.weight"] = self.weights["embed_tokens.weight"]

        rope_dim = int(self.cfg.head_dim * self.cfg.partial_rotary_factor)
        inv_freq = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
        self.inv_freq = inv_freq

    def _mrope_cos_sin(self, position_ids, head_dim):
        rope_dim = int(head_dim * self.cfg.partial_rotary_factor)
        inv_freq = self.inv_freq
        freqs = position_ids.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(torch.bfloat16)
        sin = emb.sin().to(torch.bfloat16)
        return cos, sin

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary(self, x, cos, sin):
        rope_dim = int(self.cfg.head_dim * self.cfg.partial_rotary_factor)
        x_rot = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        cos_exp = cos.unsqueeze(1)
        sin_exp = sin.unsqueeze(1)
        if x_rot.dim() == 3:
            pass
        elif x_rot.dim() == 2:
            cos_exp = cos
            sin_exp = sin
        x_rot = (x_rot.float() * cos_exp.float() + self._rotate_half(x_rot).float() * sin_exp.float()).to(torch.bfloat16)
        if x_pass.shape[-1] > 0:
            return torch.cat([x_rot, x_pass], dim=-1)
        return x_rot

    def _gdn_layer(self, hidden_states, layer_idx, conv_state, recurrent_state):
        cfg = self.cfg
        pfx = f"layers.{layer_idx}.linear_attn"
        lin_h = cfg.linear_num_key_heads
        lin_kd = cfg.linear_key_head_dim
        lin_vd = cfg.linear_value_head_dim
        conv_dim = lin_h * (lin_kd * 2 + lin_vd)
        z_dim = lin_h * lin_vd
        kernel_size = cfg.linear_conv_kernel_dim
        S = hidden_states.shape[0]
        device = self.device

        normed = torch_gemma_rmsnorm(hidden_states, self.weights[f"layers.{layer_idx}.input_layernorm.weight"], cfg.rms_norm_eps)

        mixed_qkv = F.linear(normed, self.weights[f"{pfx}.in_proj_qkv.weight"])
        a_raw = F.linear(normed, self.weights[f"{pfx}.in_proj_a.weight"])
        b_raw = F.linear(normed, self.weights[f"{pfx}.in_proj_b.weight"])
        z = F.linear(normed, self.weights[f"{pfx}.in_proj_z.weight"])

        mixed_qkv_t = mixed_qkv.t().contiguous()
        conv_weight = self.weights[f"{pfx}.conv1d.weight"].squeeze(1)
        conv_out_t = torch_causal_conv1d(mixed_qkv_t.unsqueeze(0), conv_weight, kernel_size)
        mixed_qkv = conv_out_t.squeeze(0).t().contiguous()

        q, k, v = mixed_qkv.split([lin_h * lin_kd, lin_h * lin_kd, lin_h * lin_vd], dim=-1)
        q = q.view(S, lin_h, lin_kd)
        k = k.view(S, lin_h, lin_kd)
        v = v.view(S, lin_h, lin_vd)
        z = z.view(S, lin_h, lin_vd)
        a_raw = a_raw.view(S, lin_h)
        b_raw = b_raw.view(S, lin_h)

        A_log = self.weights[f"{pfx}.A_log"].float()
        dt_bias = self.weights[f"{pfx}.dt_bias"].float()
        beta = torch.sigmoid(b_raw.float())
        g = -A_log.float().exp().unsqueeze(0) * F.softplus(a_raw.float() + dt_bias.float().unsqueeze(0))

        state = torch.zeros(lin_h, lin_kd, lin_vd, dtype=torch.float32, device=device)
        outputs = []
        for t in range(S):
            q_t = q[t:t+1].bfloat16().float()
            k_t = k[t:t+1].bfloat16().float()
            v_t = v[t:t+1].bfloat16().float()
            o_t, state = torch_gdn_recurrent_step(q_t, k_t, v_t, beta[t:t+1], g[t:t+1], state, lin_kd, lin_vd)
            outputs.append(o_t.squeeze(0))
        out = torch.stack(outputs)

        norm_weight = self.weights[f"{pfx}.norm.weight"]
        out_bf16 = out.to(torch.bfloat16)
        z_bf16 = z.to(torch.bfloat16)
        norm_w_bf16 = norm_weight.to(torch.bfloat16) if norm_weight.dtype != torch.bfloat16 else norm_weight
        out_gated = torch_rmsnorm_gated(out_bf16, z_bf16, norm_w_bf16, cfg.rms_norm_eps)

        out_2d = out_gated.reshape(S, -1)
        attn_out = F.linear(out_2d, self.weights[f"{pfx}.out_proj.weight"])

        residual = hidden_states + attn_out
        normed2 = torch_gemma_rmsnorm(residual, self.weights[f"layers.{layer_idx}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
        mlp_out = self._mlp(normed2, f"layers.{layer_idx}")
        return residual + mlp_out

    def _gdn_layer_decode(self, hidden_states, layer_idx, conv_state, recurrent_state):
        cfg = self.cfg
        pfx = f"layers.{layer_idx}.linear_attn"
        lin_h = cfg.linear_num_key_heads
        lin_kd = cfg.linear_key_head_dim
        lin_vd = cfg.linear_value_head_dim
        conv_dim = lin_h * (lin_kd * 2 + lin_vd)
        z_dim = lin_h * lin_vd
        kernel_size = cfg.linear_conv_kernel_dim

        normed = torch_gemma_rmsnorm(hidden_states, self.weights[f"layers.{layer_idx}.input_layernorm.weight"], cfg.rms_norm_eps)

        mixed_qkv = F.linear(normed, self.weights[f"{pfx}.in_proj_qkv.weight"])
        a_raw = F.linear(normed, self.weights[f"{pfx}.in_proj_a.weight"])
        b_raw = F.linear(normed, self.weights[f"{pfx}.in_proj_b.weight"])
        z = F.linear(normed, self.weights[f"{pfx}.in_proj_z.weight"])

        conv_weight = self.weights[f"{pfx}.conv1d.weight"].squeeze(1)

        x = mixed_qkv.squeeze(0)
        for c in range(conv_dim):
            conv_state[c] = torch.cat([conv_state[c, 1:], x[c:c+1]])
            s = conv_weight[c, -1].item() * x[c].item()
            for kk in range(kernel_size - 1):
                s += conv_weight[c, kk].item() * conv_state[c, kk].item()
            x[c] = torch.nn.functional.silu(torch.tensor(s)).to(torch.bfloat16)
        mixed_qkv = x.unsqueeze(0)

        q, k, v = mixed_qkv.squeeze(0).split([lin_h * lin_kd, lin_h * lin_kd, lin_h * lin_vd])
        q = q.view(lin_h, lin_kd).unsqueeze(0)
        k = k.view(lin_h, lin_kd).unsqueeze(0)
        v = v.view(lin_h, lin_vd).unsqueeze(0)
        z = z.view(1, lin_h, lin_vd)
        a_raw = a_raw.view(1, lin_h)
        b_raw = b_raw.view(1, lin_h)

        A_log = self.weights[f"{pfx}.A_log"].float()
        dt_bias = self.weights[f"{pfx}.dt_bias"].float()
        beta = torch.sigmoid(b_raw.float())
        g = -A_log.float().exp().unsqueeze(0) * F.softplus(a_raw.float() + dt_bias.float().unsqueeze(0))

        out, recurrent_state_new = torch_gdn_recurrent_step(q, k, v, beta, g, recurrent_state, lin_kd, lin_vd)
        recurrent_state.copy_(recurrent_state_new)

        norm_weight = self.weights[f"{pfx}.norm.weight"]
        out = torch_rmsnorm_gated(out, z, norm_weight, cfg.rms_norm_eps)

        out_2d = out.squeeze(0).reshape(1, -1)
        attn_out = F.linear(out_2d, self.weights[f"{pfx}.out_proj.weight"])

        residual = hidden_states + attn_out
        normed2 = torch_gemma_rmsnorm(residual, self.weights[f"layers.{layer_idx}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
        mlp_out = self._mlp(normed2, f"layers.{layer_idx}")
        return residual + mlp_out

    def _full_attn_layer(self, hidden_states, layer_idx, position_ids_cos, position_ids_sin):
        cfg = self.cfg
        pfx = f"layers.{layer_idx}.self_attn"
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        n_groups = cfg.num_key_value_groups
        S = hidden_states.shape[0]
        rope_dim = int(hd * cfg.partial_rotary_factor)

        normed = torch_gemma_rmsnorm(hidden_states, self.weights[f"layers.{layer_idx}.input_layernorm.weight"], cfg.rms_norm_eps)

        q_total_dim = n_heads * hd
        q_full = F.linear(normed, self.weights[f"{pfx}.q_proj.weight"])
        k = F.linear(normed, self.weights[f"{pfx}.k_proj.weight"])
        v = F.linear(normed, self.weights[f"{pfx}.v_proj.weight"])

        if cfg.attn_output_gate:
            q_full_3d = q_full.view(S, n_heads, hd * 2)
            q, gate = torch.chunk(q_full_3d, 2, dim=-1)
            q = q.reshape(S, n_heads * hd)
            gate = gate.reshape(S, n_heads * hd)
        else:
            q = q_full

        q_norm_weight = self.weights[f"{pfx}.q_norm.weight"]
        k_norm_weight = self.weights[f"{pfx}.k_norm.weight"]
        q = torch_gemma_rmsnorm(q.reshape(S, n_heads, hd), q_norm_weight, cfg.rms_norm_eps).reshape(S, n_heads * hd)
        k = torch_gemma_rmsnorm(k.reshape(S, n_kv, hd), k_norm_weight, cfg.rms_norm_eps).reshape(S, n_kv * hd)

        q = q.reshape(S, n_heads, hd)
        k = k.reshape(S, n_kv, hd)
        v = v.reshape(S, n_kv, hd)

        cos = position_ids_cos
        sin = position_ids_sin

        q_rope = self._apply_rotary(q, cos, sin)
        k_rope = self._apply_rotary(k, cos, sin)

        if n_groups > 1:
            k_expanded = k_rope.unsqueeze(2).expand(-1, -1, n_groups, -1).reshape(S, n_heads, hd)
            v_expanded = v.unsqueeze(2).expand(-1, -1, n_groups, -1).reshape(S, n_heads, hd)
        else:
            k_expanded = k_rope
            v_expanded = v

        scores = torch.matmul(q_rope.transpose(0, 1), k_expanded.transpose(0, 1).transpose(-2, -1)) * cfg.scaling
        mask = torch.triu(torch.full((S, S), float('-inf'), device=self.device), diagonal=1)
        scores = scores + mask.unsqueeze(0)
        attn_weights = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
        attn_output = torch.matmul(attn_weights, v_expanded.transpose(0, 1))

        if cfg.attn_output_gate:
            gate = gate.reshape(S, n_heads, hd)
            gate_sigmoid = torch.sigmoid(gate)
            attn_output = attn_output * gate_sigmoid.transpose(0, 1)

        attn_output = attn_output.transpose(0, 1).reshape(S, n_heads * hd)
        o = F.linear(attn_output, self.weights[f"{pfx}.o_proj.weight"])

        residual = hidden_states + o
        post_norm_w = self.weights[f"layers.{layer_idx}.post_attention_layernorm.weight"]
        post_normed = torch_gemma_rmsnorm(residual, post_norm_w, cfg.rms_norm_eps)
        mlp_out = self._mlp(post_normed, f"layers.{layer_idx}")
        return residual + mlp_out

    def _mlp(self, hidden_states, pfx):
        cfg = self.cfg
        gate = F.linear(hidden_states, self.weights[f"{pfx}.mlp.gate_proj.weight"])
        up = F.linear(hidden_states, self.weights[f"{pfx}.mlp.up_proj.weight"])
        silu_out = F.silu(gate) * up
        return F.linear(silu_out, self.weights[f"{pfx}.mlp.down_proj.weight"])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        BS = input_ids.shape[0]
        S = input_ids.shape[1] if input_ids.dim() > 1 else 1
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        cfg = self.cfg
        device = self.device

        hidden = F.embedding(input_ids, self.weights["embed_tokens.weight"])

        cos, sin = self._mrope_cos_sin(torch.arange(S, device=device), cfg.head_dim)

        for i in range(cfg.num_hidden_layers):
            if cfg.layer_types[i] == "linear_attention":
                hidden_2d = hidden.view(-1, cfg.hidden_size)
                hidden_2d = self._gdn_layer(hidden_2d, i, None, None)
                hidden = hidden_2d.view(BS, S, cfg.hidden_size)
            else:
                hidden_2d = hidden.view(-1, cfg.hidden_size)
                hidden_2d = self._full_attn_layer(hidden_2d, i, cos, sin)
                hidden = hidden_2d.view(BS, S, cfg.hidden_size)

        hidden = hidden.view(-1, cfg.hidden_size)
        hidden = torch_gemma_rmsnorm(hidden, self.weights["norm.weight"], cfg.rms_norm_eps)
        logits = F.linear(hidden, self.weights["lm_head.weight"])
        return logits.view(BS, S, cfg.vocab_size).float()
