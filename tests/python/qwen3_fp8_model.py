from __future__ import annotations

import ctypes
import json
import os
from typing import Any, Optional

import numpy as np
import torch
from safetensors import safe_open

from helpers import GlmOps, get_model_path
from qwen3_model import Qwen3Model, Qwen3Config, BF16, I32, _f32_to_bf16_bytes, _bf16_bytes_to_f32
from flat_kv import FlatKVCache

FP8 = 1  # bytes per float8
FP8_BLOCK = 128

FP8_LINEAR_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


class Qwen3FP8Model(Qwen3Model):

    @classmethod
    def from_pretrained(cls, glm: GlmOps, repo_id: str,
                        max_batch: int = 1, max_seq_len: int = 4096) -> Qwen3FP8Model:
        model_dir = get_model_path(repo_id)
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        cfg = Qwen3Config(config)

        weights: dict[str, int] = {}
        fp8_weights: dict[str, int] = {}
        weight_scales: dict[str, int] = {}

        st_path = os.path.join(model_dir, "model.safetensors")
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                is_scale = key.endswith(".weight_scale_inv")
                is_fp8_weight = any(key.endswith(sfx) for sfx in FP8_LINEAR_SUFFIXES)

                if is_scale:
                    scale_bf16 = t.to(torch.bfloat16).contiguous()
                    nbytes = scale_bf16.numel() * BF16
                    gpu_ptr = glm.alloc(nbytes)
                    glm.lib.glm_h2d(glm.ctx, ctypes.c_void_p(gpu_ptr),
                                     ctypes.c_void_p(scale_bf16.data_ptr()), nbytes)
                    weight_scales[key] = gpu_ptr
                elif is_fp8_weight:
                    fp8_bytes = t.view(torch.uint8).contiguous()
                    nbytes = fp8_bytes.numel() * FP8
                    gpu_ptr = glm.alloc(nbytes)
                    glm.lib.glm_h2d(glm.ctx, ctypes.c_void_p(gpu_ptr),
                                     ctypes.c_void_p(fp8_bytes.data_ptr()), nbytes)
                    fp8_weights[key] = gpu_ptr
                else:
                    t_bf16 = t.to(torch.bfloat16).contiguous()
                    nbytes = t_bf16.numel() * BF16
                    gpu_ptr = glm.alloc(nbytes)
                    glm.lib.glm_h2d(glm.ctx, ctypes.c_void_p(gpu_ptr),
                                     ctypes.c_void_p(t_bf16.data_ptr()), nbytes)
                    weights[key] = gpu_ptr

        if cfg.tie_word_embeddings and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

        model = cls(glm, cfg, weights, fp8_weights, weight_scales, max_batch, max_seq_len)
        return model

    def __init__(self, glm: GlmOps, config: Qwen3Config,
                 weights: dict[str, int],
                 fp8_weights: dict[str, int],
                 weight_scales: dict[str, int],
                 max_batch: int = 1, max_seq_len: int = 4096) -> None:
        self.glm = glm
        self.cfg = config
        self.weights = weights
        self.fp8_weights = fp8_weights
        self.weight_scales = weight_scales
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.device = glm.device

        half_dim = config.head_dim // 2
        inv_freq_f32 = 1.0 / (
            config.rope_theta ** (np.arange(0, config.head_dim, 2, dtype=np.float32) / config.head_dim)
        )
        inv_freq_bytes = _f32_to_bf16_bytes(inv_freq_f32)
        self.inv_freq = glm.alloc(len(inv_freq_bytes))
        glm.h2d(self.inv_freq, inv_freq_bytes)

        self._alloc_workspace(max_batch, max_seq_len)

    def _alloc_workspace(self, B: int, S: int) -> None:
        super()._alloc_workspace(B, S)

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
        for ptr in self.fp8_weights.values():
            if isinstance(ptr, int) and ptr not in freed_ptrs:
                glm.free_buf(ptr)
                freed_ptrs.add(ptr)
        for ptr in self.weight_scales.values():
            if isinstance(ptr, int) and ptr not in freed_ptrs:
                glm.free_buf(ptr)
                freed_ptrs.add(ptr)
        self._ws = {}
        self.weights = {}
        self.fp8_weights = {}
        self.weight_scales = {}

    def _fp8_linear(self, output: int, input_bf16: int,
                     weight_key: str, m: int, n: int, k: int) -> None:
        glm = self.glm
        fp8_w_ptr = self.fp8_weights[weight_key]
        scale_key = weight_key + "_scale_inv"
        ws_ptr = self.weight_scales[scale_key]

        glm.fp8_linear_decode(output, input_bf16, fp8_w_ptr, ws_ptr, m, n, k)

    def _compute_qkv(self, pfx: str, BS: int, B: int, S: int) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim

        self._fp8_linear(self._ws["q_buf"], self._ws["normed"],
                          f"{pfx}.self_attn.q_proj.weight", BS, n_heads * hd, hs)
        self._fp8_linear(self._ws["k_buf"], self._ws["normed"],
                          f"{pfx}.self_attn.k_proj.weight", BS, n_kv * hd, hs)
        self._fp8_linear(self._ws["v_buf"], self._ws["normed"],
                          f"{pfx}.self_attn.v_proj.weight", BS, n_kv * hd, hs)

        glm.rmsnorm(self._ws["q_normed"], self._ws["q_buf"],
                     self.weights[f"{pfx}.self_attn.q_norm.weight"],
                     cfg.rms_norm_eps, hd, BS * n_heads)

        glm.rmsnorm(self._ws["k_normed"], self._ws["k_buf"],
                     self.weights[f"{pfx}.self_attn.k_norm.weight"],
                     cfg.rms_norm_eps, hd, BS * n_kv)

        glm.transpose_4d(self._ws["q_t"], self._ws["q_normed"],
                          B, S, n_heads, hd, 0, 2, 1, 3)
        glm.transpose_4d(self._ws["k_t"], self._ws["k_normed"],
                          B, S, n_kv, hd, 0, 2, 1, 3)
        glm.transpose_4d(self._ws["v_t"], self._ws["v_buf"],
                          B, S, n_kv, hd, 0, 2, 1, 3)

        glm.apply_rotary_pos_emb(self._ws["q_rope"], self._ws["q_t"],
                                  self._ws["cos"], self._ws["sin"],
                                  hd, n_heads, S, B, 1)
        glm.apply_rotary_pos_emb(self._ws["k_rope"], self._ws["k_t"],
                                  self._ws["cos"], self._ws["sin"],
                                  hd, n_kv, S, B, 1)

    def _mlp(self, BS: int, pfx: str) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        inter = cfg.intermediate_size

        self._fp8_linear(self._ws["gate_buf"], self._ws["normed"],
                          f"{pfx}.mlp.gate_proj.weight", BS, inter, hs)
        self._fp8_linear(self._ws["up_buf"], self._ws["normed"],
                          f"{pfx}.mlp.up_proj.weight", BS, inter, hs)
        glm.silu_and_mul(self._ws["silu_buf"], self._ws["gate_buf"],
                          self._ws["up_buf"], inter, BS)
        self._fp8_linear(self._ws["down_buf"], self._ws["silu_buf"],
                          f"{pfx}.mlp.down_proj.weight", BS, hs, inter)

        glm.add(self._ws["hidden_a"], self._ws["hidden_b"],
                self._ws["down_buf"], BS * hs)

    def _attention(self, B: int, S: int, pfx: str) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        n_groups = cfg.num_key_value_groups
        BS = B * S

        self._compute_qkv(pfx, BS, B, S)

        if n_groups > 1:
            glm.expand_dim1(self._ws["k_expanded"], self._ws["k_rope"],
                            n_heads, n_kv, S, hd, B)
            glm.expand_dim1(self._ws["v_expanded"], self._ws["v_t"],
                            n_heads, n_kv, S, hd, B)
        else:
            glm.memcpy(self._ws["k_expanded"], self._ws["k_rope"], B * n_heads * S * hd * BF16)
            glm.memcpy(self._ws["v_expanded"], self._ws["v_t"], B * n_heads * S * hd * BF16)

        glm.bmm(self._ws["attn_scores"], self._ws["q_rope"], self._ws["k_expanded"],
                 cfg.scaling, 0.0, B * n_heads, S, S, hd, 0, 1)

        glm.expand_dim1(self._ws["mask_expanded"], self._ws["causal_mask"],
                        n_heads, 1, S, S, B)

        glm.softmax(self._ws["attn_scores"], self._ws["attn_scores"],
                     self._ws["mask_expanded"],
                     S, B * n_heads * S)

        glm.bmm(self._ws["attn_out"], self._ws["attn_scores"], self._ws["v_expanded"],
                 1.0, 0.0, B * n_heads, S, hd, S, 0, 0)

        glm.transpose_4d(self._ws["attn_out_t"], self._ws["attn_out"],
                          B, n_heads, S, hd, 0, 2, 1, 3)

        self._fp8_linear(self._ws["o_proj_buf"], self._ws["attn_out_t"],
                          f"{pfx}.self_attn.o_proj.weight", BS, hs, n_heads * hd)

    def _attention_prefill_flash(self, B: int, S: int, layer_idx: int,
                                  pfx: str, cache: FlatKVCache) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        max_S = self.max_seq_len
        BS = B * S

        self._compute_qkv(pfx, BS, B, S)

        self._write_kv_flat(layer_idx, S, cache)

        kv_stride_h = max_S * hd
        kv_stride_n = hd

        glm.flash_prefill(
            self._ws["q_rope"],
            cache.k_data[layer_idx],
            cache.v_data[layer_idx],
            self._ws["flash_out"],
            self._ws["flash_tmp"],
            S, S,
            n_heads, n_kv, hd,
            hd, S * hd,
            kv_stride_n, kv_stride_h,
            kv_stride_n, kv_stride_h,
            1, 1,
            cfg.scaling
        )

        self._fp8_linear(self._ws["o_proj_buf"], self._ws["flash_out"],
                          f"{pfx}.self_attn.o_proj.weight", BS, hs, n_heads * hd)

    def _attention_decode_flash(self, B: int, S: int, layer_idx: int,
                                 pfx: str, cached_len: int, cache: FlatKVCache) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        max_S = self.max_seq_len
        BS = B * S
        total_len = cached_len + S

        self._compute_qkv(pfx, BS, B, S)

        self._write_kv_flat(layer_idx, S, cache, offset=cached_len)

        kv_stride_h = max_S * hd
        kv_stride_n = hd

        glm.flash_decode(
            self._ws["q_rope"],
            cache.k_data[layer_idx],
            cache.v_data[layer_idx],
            self._ws["flash_out"],
            self._ws["flash_tmp"],
            total_len,
            n_heads, n_kv, hd,
            hd, S * hd,
            kv_stride_n, kv_stride_h,
            cfg.scaling
        )

        self._fp8_linear(self._ws["o_proj_buf"], self._ws["flash_out"],
                          f"{pfx}.self_attn.o_proj.weight", BS, hs, n_heads * hd)
