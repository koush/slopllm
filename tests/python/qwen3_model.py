from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from safetensors import safe_open

from helpers import GlmOps, get_model_path
from paged_kv import (
    PagedKVCache, WorkspaceBuffers,
    PAGE_SIZE, BATCH_FLOAT_WS_SIZE, BATCH_INT_WS_SIZE, BATCH_PINNED_INT_WS_SIZE,
)
from flat_kv import FlatKVCache

BF16 = 2  # bytes per bfloat16
I32 = 4   # bytes per int32
FLASH_TMP_SIZE = 32 * 1024 * 1024  # 32MB workspace for FlashInfer


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


class Qwen3Config:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    tie_word_embeddings: bool
    attention_bias: bool
    num_key_value_groups: int
    scaling: float

    def __init__(self, d: dict[str, Any]) -> None:
        self.hidden_size = d["hidden_size"]
        self.num_attention_heads = d["num_attention_heads"]
        self.num_key_value_heads = d["num_key_value_heads"]
        self.head_dim = d["head_dim"]
        self.intermediate_size = d["intermediate_size"]
        self.num_hidden_layers = d["num_hidden_layers"]
        self.rms_norm_eps = d["rms_norm_eps"]
        self.rope_theta = d["rope_theta"]
        self.vocab_size = d["vocab_size"]
        self.tie_word_embeddings = d.get("tie_word_embeddings", False)
        self.attention_bias = d.get("attention_bias", False)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5


@dataclass
class DecodeState:
    batch_size: int


@dataclass
class PrefillState:
    batch_size: int
    total_tokens: int
    seq_lens: list[int]
    page_allocs: list[tuple[int, int]]


class Qwen3Model:
    glm: GlmOps
    cfg: Qwen3Config
    weights: dict[str, int]
    max_batch: int
    max_seq_len: int
    device: int
    inv_freq: int
    _ws: dict[str, int]

    def __init__(self, glm: GlmOps, config: Qwen3Config, weights: dict[str, int],
                 max_batch: int = 1, max_seq_len: int = 4096) -> None:
        self.glm = glm
        self.cfg = config
        self.weights = weights
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

    @classmethod
    def from_pretrained(cls, glm: GlmOps, repo_id: str,
                        max_batch: int = 1, max_seq_len: int = 4096) -> Qwen3Model:
        model_dir = get_model_path(repo_id)
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        cfg = Qwen3Config(config)

        weights: dict[str, int] = {}
        st_path = os.path.join(model_dir, "model.safetensors")
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key).to(torch.bfloat16).contiguous()
                nbytes = t.numel() * BF16
                gpu_ptr = glm.alloc(nbytes)
                glm.lib.glm_h2d(glm.ctx, ctypes.c_void_p(gpu_ptr), ctypes.c_void_p(t.data_ptr()), nbytes)
                weights[key] = gpu_ptr

        if cfg.tie_word_embeddings and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

        return cls(glm, cfg, weights, max_batch, max_seq_len)

    def _alloc_workspace(self, B: int, S: int) -> None:
        cfg = self.cfg
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        inter = cfg.intermediate_size
        vs = cfg.vocab_size
        BS = B * S
        glm = self.glm

        self._ws = {
            "hidden_a": glm.alloc(BS * hs * BF16),
            "hidden_b": glm.alloc(BS * hs * BF16),
            "normed": glm.alloc(BS * hs * BF16),
            "q_buf": glm.alloc(BS * n_heads * hd * BF16),
            "k_buf": glm.alloc(BS * n_kv * hd * BF16),
            "v_buf": glm.alloc(BS * n_kv * hd * BF16),
            "q_normed": glm.alloc(BS * n_heads * hd * BF16),
            "k_normed": glm.alloc(BS * n_kv * hd * BF16),
            "q_t": glm.alloc(B * n_heads * S * hd * BF16),
            "k_t": glm.alloc(B * n_kv * S * hd * BF16),
            "v_t": glm.alloc(B * n_kv * S * hd * BF16),
            "q_rope": glm.alloc(B * n_heads * S * hd * BF16),
            "k_rope": glm.alloc(B * n_kv * S * hd * BF16),
            "k_expanded": glm.alloc(B * n_heads * S * hd * BF16),
            "v_expanded": glm.alloc(B * n_heads * S * hd * BF16),
            "attn_scores": glm.alloc(B * n_heads * S * S * BF16),
            "attn_out": glm.alloc(B * n_heads * S * hd * BF16),
            "attn_out_t": glm.alloc(B * n_heads * S * hd * BF16),
            "o_proj_buf": glm.alloc(BS * hs * BF16),
            "gate_buf": glm.alloc(BS * inter * BF16),
            "up_buf": glm.alloc(BS * inter * BF16),
            "silu_buf": glm.alloc(BS * inter * BF16),
            "down_buf": glm.alloc(BS * hs * BF16),
            "cos": glm.alloc(B * S * hd * BF16),
            "sin": glm.alloc(B * S * hd * BF16),
            "causal_mask": glm.alloc(S * S * BF16),
            "mask_expanded": glm.alloc(B * n_heads * S * S * BF16),
            "position_ids": glm.alloc(B * S * I32),
            "logits_buf": glm.alloc(B * vs * BF16),
            "hidden_last": glm.alloc(B * hs * BF16),
            "last_idx": glm.alloc(B * I32),
            "argmax_idx": glm.alloc(B * I32),
            "decode_id": glm.alloc(I32),
            "flash_out": glm.alloc(B * n_heads * S * hd * BF16),
            "flash_tmp": glm.alloc(FLASH_TMP_SIZE),
            "input_ids_buf": glm.alloc(BS * I32),
            "qo_indptr_d": glm.alloc((B + 1) * I32),
            "kv_indptr_d": glm.alloc((B + 1) * I32),
            "prefill_slot_mapping": glm.alloc(BS * I32),
        }

    def free(self) -> None:
        glm = self.glm
        for ptr in self._ws.values():
            glm.free_buf(ptr)
        glm.free_buf(self.inv_freq)
        for ptr in self.weights.values():
            glm.free_buf(ptr)
        self._ws = {}
        self.weights = {}

    def create_flat_kv_cache(self) -> FlatKVCache:
        return FlatKVCache(self.glm, self.cfg.num_key_value_heads, self.cfg.head_dim,
                           self.cfg.num_hidden_layers, self.max_batch, self.max_seq_len)

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
                     self.weights["model.norm.weight"],
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        vs = cfg.vocab_size
        BS = B * S

        assert B <= self.max_batch and S <= self.max_seq_len, \
            f"input (B={B}, S={S}) exceeds max (B={self.max_batch}, S={self.max_seq_len})"

        ids_np = input_ids.cpu().numpy().astype(np.int32).flatten()
        glm.h2d(self._ws["input_ids_buf"], ids_np.tobytes())
        glm.embedding(self._ws["hidden_a"], self.weights["model.embed_tokens.weight"],
                       self._ws["input_ids_buf"], hs, BS)

        glm.arange(self._ws["position_ids"], 0, 1, S)

        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              hd // 2, B, S)

        glm.causal_mask(self._ws["causal_mask"], S)

        for i in range(cfg.num_hidden_layers):
            pfx = f"model.layers.{i}"
            self._decoder_layer(B, S, pfx)

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights["model.norm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        logits_full = glm.alloc(BS * vs * BF16)
        glm.linear(logits_full, self._ws["normed"],
                    self.weights["lm_head.weight"],
                    BS, vs, hs)

        logits_count = BS * vs
        logits_u16 = self._read_logits(logits_full, logits_count)
        glm.free_buf(logits_full)
        logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(B, S, vs)
        return torch.from_numpy(logits_f32.copy())

    def prefill(self, input_ids: torch.Tensor, cache: FlatKVCache) -> torch.Tensor:
        B, S = input_ids.shape
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        vs = cfg.vocab_size
        BS = B * S

        assert B <= self.max_batch and S <= self.max_seq_len
        assert cache.cache_pos == 0, "Cache must be reset before prefill"

        ids_np = input_ids.cpu().numpy().astype(np.int32).flatten()
        glm.h2d(self._ws["input_ids_buf"], ids_np.tobytes())
        glm.embedding(self._ws["hidden_a"], self.weights["model.embed_tokens.weight"],
                       self._ws["input_ids_buf"], hs, BS)

        glm.arange(self._ws["position_ids"], 0, 1, S)

        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              hd // 2, B, S)

        glm.causal_mask(self._ws["causal_mask"], S)

        for i in range(cfg.num_hidden_layers):
            pfx = f"model.layers.{i}"
            self._decoder_layer_prefill_flash(B, S, i, pfx, cache)

        last_idx_np = np.array([S - 1], dtype=np.int32)
        self._extract_last_logits(1, last_idx_np)

        cache.cache_pos = S

        logits_u16 = self._read_logits(self._ws["logits_buf"], vs)
        logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(1, vs)
        return torch.from_numpy(logits_f32.copy())

    def _decode_token(self, token_id: int, cache: FlatKVCache) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        vs = cfg.vocab_size
        B = 1
        S = 1
        BS = 1
        cached_len = cache.cache_pos

        assert cached_len > 0, "Must prefill before decode"
        assert cached_len + S <= self.max_seq_len

        ids_np = np.array([token_id], dtype=np.int32)
        glm.h2d(self._ws["decode_id"], ids_np.tobytes())

        glm.embedding(self._ws["hidden_a"], self.weights["model.embed_tokens.weight"],
                       self._ws["decode_id"], hs, BS)

        glm.arange(self._ws["position_ids"], cached_len, 0, 1)

        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              hd // 2, B, S)

        for i in range(cfg.num_hidden_layers):
            pfx = f"model.layers.{i}"
            self._decoder_layer_decode_flash(B, S, i, pfx, cached_len, cache)

        self._final_norm_and_logits(BS)

        cache.cache_pos = cached_len + S

    def decode(self, input_ids: torch.Tensor, cache: FlatKVCache) -> torch.Tensor:
        B, S = input_ids.shape
        assert B == 1 and S == 1, "Decode only supports B=1, S=1"
        vs = self.cfg.vocab_size

        token_id = input_ids[0, 0].item()
        self._decode_token(token_id, cache)

        logits_u16 = self._read_logits(self._ws["logits_buf"], vs)
        logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(1, 1, vs)
        return torch.from_numpy(logits_f32.copy())

    def _argmax_logits(self, logits_ptr: int, count: int) -> int:
        self.glm.argmax(self._ws["argmax_idx"], logits_ptr, count)
        buf = ctypes.create_string_buffer(I32)
        self.glm.lib.glm_d2h(self.glm.ctx, buf, ctypes.c_void_p(self._ws["argmax_idx"]), I32)
        return int(np.frombuffer(buf.raw, dtype=np.int32)[0])

    def _argmax_logits_batch(self, batch_size: int) -> list[int]:
        vs = self.cfg.vocab_size
        self.glm.argmax(self._ws["argmax_idx"], self._ws["logits_buf"], vs, batch_size)
        buf = ctypes.create_string_buffer(batch_size * I32)
        self.glm.lib.glm_d2h(self.glm.ctx, buf, ctypes.c_void_p(self._ws["argmax_idx"]), batch_size * I32)
        return [int(x) for x in np.frombuffer(buf.raw, dtype=np.int32)]

    def generate(self, input_ids: torch.Tensor, cache: FlatKVCache,
                 max_new_tokens: int = 100, eos_token_ids: Optional[set[int]] = None) -> list[int]:
        return list(self.generate_tokens(input_ids, cache, max_new_tokens, eos_token_ids))

    def generate_tokens(self, input_ids: torch.Tensor, cache: FlatKVCache,
                        max_new_tokens: int = 100, eos_token_ids: Optional[set[int]] = None) -> Any:
        if eos_token_ids is None:
            eos_token_ids = {151645, 151643}

        B, S = input_ids.shape
        assert B == 1, "generate_tokens() only supports batch=1"

        cfg = self.cfg
        vs = cfg.vocab_size

        cache.reset()
        self.prefill(input_ids, cache)

        next_token = self._argmax_logits(self._ws["logits_buf"], vs)
        yield next_token

        for _ in range(max_new_tokens - 1):
            if next_token in eos_token_ids:
                break
            self._decode_token(next_token, cache)
            next_token = self._argmax_logits(self._ws["logits_buf"], vs)
            yield next_token

    def generate_text(self, prompt: str, tokenizer: Any, cache: FlatKVCache,
                      max_new_tokens: int = 100, eos_token_ids: Optional[set[int]] = None,
                      enable_thinking: bool = True) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )
        input_ids = _normalize_token_ids(input_ids, tokenizer)
        input_tensor = torch.tensor([input_ids], dtype=torch.int64)

        generated_ids = list(self.generate_tokens(
            input_tensor, cache, max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        ))
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_batch(self, input_ids_list: list[list[int]], ws: WorkspaceBuffers,
                       paged_kv: PagedKVCache, max_new_tokens: int = 100,
                       eos_token_ids: Optional[set[int]] = None) -> list[list[int]]:
        if eos_token_ids is None:
            eos_token_ids = {151645, 151643}

        batch_size = len(input_ids_list)
        cfg = self.cfg
        vs = cfg.vocab_size

        next_tokens = self.prefill_batch(input_ids_list, ws, paged_kv)

        generated: list[list[int]] = [[t] for t in next_tokens]
        finished = [t in eos_token_ids for t in next_tokens]

        for _ in range(max_new_tokens - 1):
            if all(finished):
                break

            token_ids = [next_tokens[i] for i in range(batch_size)]
            next_tokens = self.decode_batch(token_ids, ws, paged_kv)

            for i in range(batch_size):
                if finished[i]:
                    continue
                if next_tokens[i] in eos_token_ids:
                    finished[i] = True
                else:
                    generated[i].append(next_tokens[i])

        return generated

    def generate_text_batch(self, prompts: list[str], tokenizer: Any,
                            ws: WorkspaceBuffers, paged_kv: PagedKVCache,
                            max_new_tokens: int = 100, eos_token_ids: Optional[set[int]] = None,
                            enable_thinking: bool = True) -> list[str]:
        input_ids_list: list[list[int]] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                )
            ids = _normalize_token_ids(ids, tokenizer)
            input_ids_list.append(ids)

        generated_ids = self.generate_batch(
            input_ids_list, ws, paged_kv,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )

        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

    def _decoder_layer(self, B: int, S: int, pfx: str) -> None:
        cfg = self.cfg
        glm = self.glm
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, cfg.hidden_size, BS)

        self._attention(B, S, pfx)

        self._residual_and_mlp(pfx, BS)

    def _decoder_layer_prefill_flash(self, B: int, S: int, layer_idx: int,
                                      pfx: str, cache: FlatKVCache) -> None:
        cfg = self.cfg
        glm = self.glm
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, cfg.hidden_size, BS)

        self._attention_prefill_flash(B, S, layer_idx, pfx, cache)

        self._residual_and_mlp(pfx, BS)

    def _decoder_layer_decode_flash(self, B: int, S: int, layer_idx: int,
                                     pfx: str, cached_len: int, cache: FlatKVCache) -> None:
        cfg = self.cfg
        glm = self.glm
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, cfg.hidden_size, BS)

        self._attention_decode_flash(B, S, layer_idx, pfx, cached_len, cache)

        self._residual_and_mlp(pfx, BS)

    def _mlp(self, BS: int, pfx: str) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        inter = cfg.intermediate_size

        glm.linear(self._ws["gate_buf"], self._ws["normed"],
                    self.weights[f"{pfx}.mlp.gate_proj.weight"],
                    BS, inter, hs)
        glm.linear(self._ws["up_buf"], self._ws["normed"],
                    self.weights[f"{pfx}.mlp.up_proj.weight"],
                    BS, inter, hs)
        glm.silu_and_mul(self._ws["silu_buf"], self._ws["gate_buf"],
                          self._ws["up_buf"], inter, BS)
        glm.linear(self._ws["down_buf"], self._ws["silu_buf"],
                    self.weights[f"{pfx}.mlp.down_proj.weight"],
                    BS, hs, inter)

        glm.add(self._ws["hidden_a"], self._ws["hidden_b"],
                self._ws["down_buf"], BS * hs)

    def _residual_and_mlp(self, pfx: str, BS: int) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size

        glm.add(self._ws["hidden_b"], self._ws["hidden_a"],
                self._ws["o_proj_buf"], BS * hs)
        glm.rmsnorm(self._ws["normed"], self._ws["hidden_b"],
                     self.weights[f"{pfx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)
        self._mlp(BS, pfx)

    def _compute_qkv(self, pfx: str, BS: int, B: int, S: int) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim

        glm.linear(self._ws["q_buf"], self._ws["normed"],
                    self.weights[f"{pfx}.self_attn.q_proj.weight"],
                    BS, n_heads * hd, hs)
        glm.linear(self._ws["k_buf"], self._ws["normed"],
                    self.weights[f"{pfx}.self_attn.k_proj.weight"],
                    BS, n_kv * hd, hs)
        glm.linear(self._ws["v_buf"], self._ws["normed"],
                    self.weights[f"{pfx}.self_attn.v_proj.weight"],
                    BS, n_kv * hd, hs)

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

    def _write_kv_flat(self, layer_idx: int, S: int, cache: FlatKVCache,
                       offset: int = 0) -> None:
        cfg = self.cfg
        glm = self.glm
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        max_S = self.max_seq_len

        for h in range(n_kv):
            src_off = h * S * hd * BF16
            dst_off = (h * max_S * hd + offset * hd) * BF16
            glm.memcpy(cache.k_data[layer_idx] + dst_off,
                        self._ws["k_rope"] + src_off,
                        S * hd * BF16)
            glm.memcpy(cache.v_data[layer_idx] + dst_off,
                        self._ws["v_t"] + src_off,
                        S * hd * BF16)

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
                 cfg.scaling, 0.0, B * n_heads, S, S, hd, 1)

        glm.expand_dim1(self._ws["mask_expanded"], self._ws["causal_mask"],
                        n_heads, 1, S, S, B)

        glm.softmax(self._ws["attn_scores"], self._ws["attn_scores"],
                     self._ws["mask_expanded"],
                     S, B * n_heads * S)

        glm.bmm(self._ws["attn_out"], self._ws["attn_scores"], self._ws["v_expanded"],
                 1.0, 0.0, B * n_heads, S, hd, S, 0)

        glm.transpose_4d(self._ws["attn_out_t"], self._ws["attn_out"],
                          B, n_heads, S, hd, 0, 2, 1, 3)

        glm.linear(self._ws["o_proj_buf"], self._ws["attn_out_t"],
                    self.weights[f"{pfx}.self_attn.o_proj.weight"],
                    BS, hs, n_heads * hd)

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

        glm.linear(self._ws["o_proj_buf"], self._ws["flash_out"],
                    self.weights[f"{pfx}.self_attn.o_proj.weight"],
                    BS, hs, n_heads * hd)

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

        glm.linear(self._ws["o_proj_buf"], self._ws["flash_out"],
                    self.weights[f"{pfx}.self_attn.o_proj.weight"],
                    BS, hs, n_heads * hd)

    def prefill_batch_plan(self, input_ids_list: list[list[int]], ws: WorkspaceBuffers,
                           paged_kv: PagedKVCache) -> PrefillState:
        cfg = self.cfg
        glm = self.glm
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        batch_size = len(input_ids_list)

        paged_kv.reset(batch_size)

        seq_lens: list[int] = [len(ids) for ids in input_ids_list]
        total_tokens = sum(seq_lens)

        page_allocs: list[tuple[int, int]] = []
        for seq_idx, s in enumerate(seq_lens):
            start_page, num_pages = paged_kv.alloc_prefill_pages(seq_idx, s)
            page_allocs.append((start_page, num_pages))

        all_ids: list[int] = []
        for ids in input_ids_list:
            all_ids.extend(ids)
        ids_np = np.array(all_ids, dtype=np.int32)
        glm.h2d(self._ws["input_ids_buf"], ids_np.tobytes())

        qo_indptr = [0]
        kv_indptr = [0]
        for s in seq_lens:
            qo_indptr.append(qo_indptr[-1] + s)
            kv_indptr.append(kv_indptr[-1] + s)
        qo_indptr_np = np.array(qo_indptr, dtype=np.int32)
        kv_indptr_np = np.array(kv_indptr, dtype=np.int32)

        pos_ids: list[int] = []
        for s in seq_lens:
            pos_ids.extend(range(s))
        pos_ids_np = np.array(pos_ids, dtype=np.int32)
        glm.h2d(self._ws["position_ids"], pos_ids_np.tobytes())

        last_indices: list[int] = []
        offset = 0
        for s in seq_lens:
            last_indices.append(offset + s - 1)
            offset += s
        last_idx_np = np.array(last_indices, dtype=np.int32)
        glm.h2d(self._ws["last_idx"], last_idx_np.tobytes())

        glm.batch_prefill_ragged_plan(
            ws.float_ws, BATCH_FLOAT_WS_SIZE,
            ws.int_ws, ws.pinned_int_ws, BATCH_INT_WS_SIZE,
            ws.prefill_plan_info,
            qo_indptr_np.ctypes.data, kv_indptr_np.ctypes.data,
            total_tokens, batch_size,
            n_heads, n_kv, hd,
            1
        )

        glm.h2d(self._ws["qo_indptr_d"], qo_indptr_np.tobytes())
        glm.h2d(self._ws["kv_indptr_d"], kv_indptr_np.tobytes())

        return PrefillState(batch_size=batch_size, total_tokens=total_tokens,
                            seq_lens=seq_lens, page_allocs=page_allocs)

    def prefill_batch_forward(self, state: PrefillState, ws: WorkspaceBuffers,
                               paged_kv: PagedKVCache) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        batch_size = state.batch_size
        total_tokens = state.total_tokens
        seq_lens = state.seq_lens
        page_allocs = state.page_allocs

        glm.embedding(self._ws["hidden_a"], self.weights["model.embed_tokens.weight"],
                       self._ws["input_ids_buf"], hs, total_tokens)

        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              hd // 2, 1, total_tokens)

        for i in range(cfg.num_hidden_layers):
            pfx = f"model.layers.{i}"

            glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                         self.weights[f"{pfx}.input_layernorm.weight"],
                         cfg.rms_norm_eps, hs, total_tokens)

            self._compute_qkv(pfx, total_tokens, 1, total_tokens)

            for seq_idx, s in enumerate(seq_lens):
                start_page, num_pages = page_allocs[seq_idx]
                page_size = paged_kv.page_size
                seq_start = sum(seq_lens[:seq_idx])
                for h in range(n_kv):
                    for p in range(num_pages):
                        page_offset = (start_page + p) * n_kv * page_size * hd
                        kv_head_offset = page_offset + h * page_size * hd
                        token_start = p * page_size
                        token_count = min(page_size, s - p * page_size)
                        src_off = (h * total_tokens + seq_start + token_start) * hd * BF16
                        dst_off = kv_head_offset * BF16
                        copy_bytes = token_count * hd * BF16
                        glm.memcpy(paged_kv.k_data[i] + dst_off,
                                    self._ws["k_rope"] + src_off,
                                    copy_bytes)
                        glm.memcpy(paged_kv.v_data[i] + dst_off,
                                    self._ws["v_t"] + src_off,
                                    copy_bytes)

            q_stride_n = hd
            q_stride_h = total_tokens * hd
            kv_stride_n = hd
            kv_stride_h = total_tokens * hd

            glm.batch_prefill_ragged_run(
                self._ws["q_rope"], self._ws["k_rope"], self._ws["v_t"], self._ws["flash_out"],
                ws.float_ws, ws.int_ws,
                self._ws["qo_indptr_d"], self._ws["kv_indptr_d"],
                ws.prefill_plan_info,
                total_tokens, batch_size,
                n_heads, n_kv, hd,
                q_stride_n, q_stride_h,
                kv_stride_n, kv_stride_h,
                1, cfg.scaling
            )

            glm.linear(self._ws["o_proj_buf"], self._ws["flash_out"],
                         self.weights[f"{pfx}.self_attn.o_proj.weight"],
                         total_tokens, hs, n_heads * hd)

            self._residual_and_mlp(pfx, total_tokens)

        glm.index_select(self._ws["hidden_last"], self._ws["hidden_a"],
                          self._ws["last_idx"], hs, batch_size)
        self._final_norm_and_logits(batch_size, src=self._ws["hidden_last"])

        vs = cfg.vocab_size
        self.glm.argmax(self._ws["argmax_idx"], self._ws["logits_buf"], vs, batch_size)

    def prefill_batch_read(self, state: PrefillState) -> list[int]:
        batch_size = state.batch_size
        buf = ctypes.create_string_buffer(batch_size * I32)
        self.glm.lib.glm_d2h(self.glm.ctx, buf, ctypes.c_void_p(self._ws["argmax_idx"]), batch_size * I32)
        return [int(x) for x in np.frombuffer(buf.raw, dtype=np.int32)]

    def prefill_batch(self, input_ids_list: list[list[int]], ws: WorkspaceBuffers,
                      paged_kv: PagedKVCache) -> list[int]:
        state = self.prefill_batch_plan(input_ids_list, ws, paged_kv)
        self.prefill_batch_forward(state, ws, paged_kv)
        paged_kv.update_indptr()
        return self.prefill_batch_read(state)

    def prefill_batch_paged_plan(self, input_ids_list: list[list[int]], ws: WorkspaceBuffers,
                                  paged_kv: PagedKVCache) -> PrefillState:
        cfg = self.cfg
        glm = self.glm
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        page_size = paged_kv.page_size
        batch_size = len(input_ids_list)

        paged_kv.reset(batch_size)

        seq_lens: list[int] = [len(ids) for ids in input_ids_list]
        total_tokens = sum(seq_lens)

        page_allocs: list[tuple[int, int]] = []
        for seq_idx, s in enumerate(seq_lens):
            start_page, num_pages = paged_kv.alloc_prefill_pages(seq_idx, s)
            page_allocs.append((start_page, num_pages))

        all_ids: list[int] = []
        for ids in input_ids_list:
            all_ids.extend(ids)
        ids_np = np.array(all_ids, dtype=np.int32)
        glm.h2d(self._ws["input_ids_buf"], ids_np.tobytes())

        qo_indptr = [0]
        for s in seq_lens:
            qo_indptr.append(qo_indptr[-1] + s)
        qo_indptr_np = np.array(qo_indptr, dtype=np.int32)

        paged_kv.update_indptr()

        pos_ids: list[int] = []
        for s in seq_lens:
            pos_ids.extend(range(s))
        pos_ids_np = np.array(pos_ids, dtype=np.int32)
        glm.h2d(self._ws["position_ids"], pos_ids_np.tobytes())

        last_indices: list[int] = []
        offset = 0
        for s in seq_lens:
            last_indices.append(offset + s - 1)
            offset += s
        last_idx_np = np.array(last_indices, dtype=np.int32)
        glm.h2d(self._ws["last_idx"], last_idx_np.tobytes())

        slot_mapping: list[int] = []
        for seq_idx, s in enumerate(seq_lens):
            pages = paged_kv.seq_pages[seq_idx]
            for pos in range(s):
                page_idx_in_seq = pos // page_size
                offset_in_page = pos % page_size
                abs_page = pages[page_idx_in_seq]
                slot_mapping.append(abs_page * page_size + offset_in_page)
        slot_mapping_np = np.array(slot_mapping, dtype=np.int32)
        glm.h2d(self._ws["prefill_slot_mapping"], slot_mapping_np.tobytes())

        glm.batch_prefill_paged_plan(
            ws.float_ws, BATCH_FLOAT_WS_SIZE,
            ws.int_ws, ws.pinned_int_ws, BATCH_INT_WS_SIZE,
            ws.prefill_plan_info,
            qo_indptr_np.ctypes.data, paged_kv.indptr_h,
            total_tokens, batch_size,
            n_heads, n_kv, hd,
            page_size,
            1
        )

        glm.h2d(self._ws["qo_indptr_d"], qo_indptr_np.tobytes())

        return PrefillState(batch_size=batch_size, total_tokens=total_tokens,
                            seq_lens=seq_lens, page_allocs=page_allocs)

    def prefill_batch_paged_forward(self, state: PrefillState, ws: WorkspaceBuffers,
                                     paged_kv: PagedKVCache) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        page_size = paged_kv.page_size
        batch_size = state.batch_size
        total_tokens = state.total_tokens
        seq_lens = state.seq_lens
        page_allocs = state.page_allocs

        glm.embedding(self._ws["hidden_a"], self.weights["model.embed_tokens.weight"],
                       self._ws["input_ids_buf"], hs, total_tokens)

        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              hd // 2, 1, total_tokens)

        for i in range(cfg.num_hidden_layers):
            pfx = f"model.layers.{i}"

            glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                         self.weights[f"{pfx}.input_layernorm.weight"],
                         cfg.rms_norm_eps, hs, total_tokens)

            self._compute_qkv(pfx, total_tokens, 1, total_tokens)

            for seq_idx, s in enumerate(seq_lens):
                start_page, num_pages = page_allocs[seq_idx]
                seq_start = sum(seq_lens[:seq_idx])
                for h in range(n_kv):
                    for p in range(num_pages):
                        page_offset = (start_page + p) * n_kv * page_size * hd
                        kv_head_offset = page_offset + h * page_size * hd
                        token_start = p * page_size
                        token_count = min(page_size, s - p * page_size)
                        src_off = (h * total_tokens + seq_start + token_start) * hd * BF16
                        dst_off = kv_head_offset * BF16
                        copy_bytes = token_count * hd * BF16
                        glm.memcpy(paged_kv.k_data[i] + dst_off,
                                    self._ws["k_rope"] + src_off,
                                    copy_bytes)
                        glm.memcpy(paged_kv.v_data[i] + dst_off,
                                    self._ws["v_t"] + src_off,
                                    copy_bytes)

            q_stride_n = hd
            q_stride_h = total_tokens * hd

            glm.batch_prefill_paged_run(
                self._ws["q_rope"], self._ws["flash_out"],
                paged_kv.k_data[i], paged_kv.v_data[i],
                paged_kv.indices, paged_kv.indptr_d, paged_kv.last_page_len,
                ws.float_ws, ws.int_ws,
                self._ws["qo_indptr_d"],
                ws.prefill_plan_info,
                total_tokens, batch_size,
                n_heads, n_kv, hd,
                page_size,
                q_stride_n, q_stride_h,
                1, cfg.scaling
            )

            glm.linear(self._ws["o_proj_buf"], self._ws["flash_out"],
                         self.weights[f"{pfx}.self_attn.o_proj.weight"],
                         total_tokens, hs, n_heads * hd)

            self._residual_and_mlp(pfx, total_tokens)

        glm.index_select(self._ws["hidden_last"], self._ws["hidden_a"],
                          self._ws["last_idx"], hs, batch_size)
        self._final_norm_and_logits(batch_size, src=self._ws["hidden_last"])

        vs = cfg.vocab_size
        self.glm.argmax(self._ws["argmax_idx"], self._ws["logits_buf"], vs, batch_size)

    def prefill_batch_paged(self, input_ids_list: list[list[int]], ws: WorkspaceBuffers,
                             paged_kv: PagedKVCache) -> list[int]:
        state = self.prefill_batch_paged_plan(input_ids_list, ws, paged_kv)
        self.prefill_batch_paged_forward(state, ws, paged_kv)
        return self.prefill_batch_read(state)

    def prefill_batch_paged_append_plan(self, input_ids_list: list[list[int]], ws: WorkspaceBuffers,
                                         paged_kv: PagedKVCache) -> PrefillState:
        cfg = self.cfg
        glm = self.glm
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        page_size = paged_kv.page_size
        batch_size = len(input_ids_list)

        assert len(paged_kv.seq_pages) == batch_size, \
            f"prefill_batch_paged_append: paged_kv has {len(paged_kv.seq_pages)} sequences, expected {batch_size}"

        seq_lens: list[int] = [len(ids) for ids in input_ids_list]
        total_tokens = sum(seq_lens)
        start_pos = list(paged_kv.seq_kv_lens)

        page_allocs: list[tuple[int, int]] = []
        for seq_idx, s in enumerate(seq_lens):
            start_page, num_pages = paged_kv.alloc_append_pages(seq_idx, s)
            page_allocs.append((start_page, num_pages))

        all_ids: list[int] = []
        for ids in input_ids_list:
            all_ids.extend(ids)
        ids_np = np.array(all_ids, dtype=np.int32)
        glm.h2d(self._ws["input_ids_buf"], ids_np.tobytes())

        qo_indptr = [0]
        for s in seq_lens:
            qo_indptr.append(qo_indptr[-1] + s)
        qo_indptr_np = np.array(qo_indptr, dtype=np.int32)

        paged_kv.update_indptr()

        pos_ids: list[int] = []
        for seq_idx, s in enumerate(seq_lens):
            pos_ids.extend(range(start_pos[seq_idx], start_pos[seq_idx] + s))
        pos_ids_np = np.array(pos_ids, dtype=np.int32)
        glm.h2d(self._ws["position_ids"], pos_ids_np.tobytes())

        last_indices: list[int] = []
        offset = 0
        for s in seq_lens:
            last_indices.append(offset + s - 1)
            offset += s
        last_idx_np = np.array(last_indices, dtype=np.int32)
        glm.h2d(self._ws["last_idx"], last_idx_np.tobytes())

        glm.batch_prefill_paged_plan(
            ws.float_ws, BATCH_FLOAT_WS_SIZE,
            ws.int_ws, ws.pinned_int_ws, BATCH_INT_WS_SIZE,
            ws.prefill_plan_info,
            qo_indptr_np.ctypes.data, paged_kv.indptr_h,
            total_tokens, batch_size,
            n_heads, n_kv, hd,
            page_size,
            1
        )

        glm.h2d(self._ws["qo_indptr_d"], qo_indptr_np.tobytes())

        return PrefillState(batch_size=batch_size, total_tokens=total_tokens,
                            seq_lens=seq_lens, page_allocs=page_allocs)

    def prefill_batch_paged_append(self, input_ids_list: list[list[int]], ws: WorkspaceBuffers,
                                    paged_kv: PagedKVCache) -> list[int]:
        state = self.prefill_batch_paged_append_plan(input_ids_list, ws, paged_kv)
        self.prefill_batch_paged_forward(state, ws, paged_kv)
        return self.prefill_batch_read(state)

    def decode_batch_plan(self, token_ids_list: list[int], ws: WorkspaceBuffers,
                          paged_kv: PagedKVCache,
                          enable_cuda_graph: bool = False) -> DecodeState:
        cfg = self.cfg
        glm = self.glm
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        page_size = paged_kv.page_size
        batch_size = len(token_ids_list)

        write_locations: list[tuple[int, int]] = []
        for seq_idx in range(batch_size):
            abs_page, slot_in_page = paged_kv.alloc_decode_token(seq_idx)
            write_locations.append((abs_page, slot_in_page))

        paged_kv.update_indptr()
        paged_kv.update_slot_mapping(write_locations, page_size)

        ids_np = np.array(token_ids_list, dtype=np.int32)
        glm.h2d(self._ws["input_ids_buf"], ids_np.tobytes())

        pos_ids: list[int] = []
        for seq_idx in range(batch_size):
            pos_ids.append(paged_kv.seq_kv_lens[seq_idx] - 1)
        pos_ids_np = np.array(pos_ids, dtype=np.int32)
        glm.h2d(self._ws["position_ids"], pos_ids_np.tobytes())

        glm.batch_decode_plan(
            ws.float_ws, BATCH_FLOAT_WS_SIZE,
            ws.int_ws, ws.pinned_int_ws, BATCH_INT_WS_SIZE,
            ws.decode_plan_info,
            paged_kv.indptr_h,
            batch_size,
            n_heads, n_kv, page_size,
            enable_cuda_graph
        )

        return DecodeState(batch_size=batch_size)

    def decode_batch_forward(self, state: DecodeState, ws: WorkspaceBuffers,
                             paged_kv: PagedKVCache) -> None:
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        page_size = paged_kv.page_size
        batch_size = state.batch_size

        glm.embedding(self._ws["hidden_a"], self.weights["model.embed_tokens.weight"],
                       self._ws["input_ids_buf"], hs, batch_size)

        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              hd // 2, batch_size, 1)

        for i in range(cfg.num_hidden_layers):
            pfx = f"model.layers.{i}"

            glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                         self.weights[f"{pfx}.input_layernorm.weight"],
                         cfg.rms_norm_eps, hs, batch_size)

            self._compute_qkv(pfx, batch_size, batch_size, 1)

            glm.kv_cache_write(
                self._ws["k_rope"], self._ws["v_t"],
                paged_kv.k_data[i], paged_kv.v_data[i],
                paged_kv.slot_mapping,
                batch_size, n_kv, hd, page_size)

            glm.batch_decode_run(
                self._ws["q_rope"], self._ws["flash_out"],
                paged_kv.k_data[i], paged_kv.v_data[i],
                paged_kv.indices, paged_kv.indptr_d, paged_kv.last_page_len,
                ws.float_ws, ws.int_ws,
                ws.decode_plan_info,
                batch_size,
                n_heads, n_kv, hd, page_size, cfg.scaling
            )

            glm.linear(self._ws["o_proj_buf"], self._ws["flash_out"],
                         self.weights[f"{pfx}.self_attn.o_proj.weight"],
                          batch_size, hs, n_heads * hd)

            self._residual_and_mlp(pfx, batch_size)

        self._final_norm_and_logits(batch_size)

        vs = cfg.vocab_size
        self.glm.argmax(self._ws["argmax_idx"], self._ws["logits_buf"], vs, batch_size)

    def decode_batch_read(self, state: DecodeState) -> list[int]:
        batch_size = state.batch_size
        buf = ctypes.create_string_buffer(batch_size * I32)
        self.glm.lib.glm_d2h(self.glm.ctx, buf, ctypes.c_void_p(self._ws["argmax_idx"]), batch_size * I32)
        return [int(x) for x in np.frombuffer(buf.raw, dtype=np.int32)]

    def decode_batch(self, token_ids_list: list[int], ws: WorkspaceBuffers,
                     paged_kv: PagedKVCache) -> list[int]:
        state = self.decode_batch_plan(token_ids_list, ws, paged_kv)
        self.decode_batch_forward(state, ws, paged_kv)
        return self.decode_batch_read(state)
