import ctypes
import json
import os
import numpy as np
import torch
from safetensors import safe_open
from helpers import GlmOps, get_model_path
from paged_kv import (
    PagedKVCache, WorkspaceBuffers,
    PAGE_SIZE, BATCH_FLOAT_WS_SIZE, BATCH_INT_WS_SIZE, BATCH_PINNED_INT_WS_SIZE,
)

BF16 = 2  # bytes per bfloat16
I32 = 4   # bytes per int32
FLASH_TMP_SIZE = 32 * 1024 * 1024  # 32MB workspace for FlashInfer


def _f32_to_bf16_bytes(arr):
    u32 = arr.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def _bf16_bytes_to_f32(data):
    u16 = np.frombuffer(data, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _normalize_token_ids(input_ids, tokenizer):
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
    def __init__(self, d):
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


class Qwen3Model:
    def __init__(self, glm, config, weights, max_batch=1, max_seq_len=4096):
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
        self._alloc_cache(max_batch, max_seq_len)

    @classmethod
    def from_pretrained(cls, glm, repo_id, max_batch=1, max_seq_len=4096):
        model_dir = get_model_path(repo_id)
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        cfg = Qwen3Config(config)

        weights = {}
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

    def _alloc_workspace(self, B, S):
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
            "argmax_idx": glm.alloc(I32),
            "decode_id": glm.alloc(I32),
            "flash_out": glm.alloc(B * n_heads * S * hd * BF16),
            "flash_tmp": glm.alloc(FLASH_TMP_SIZE),
        }

    def _alloc_cache(self, B, S):
        cfg = self.cfg
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        n_layers = cfg.num_hidden_layers
        glm = self.glm

        self.k_cache = [glm.alloc(B * n_kv * S * hd * BF16) for _ in range(n_layers)]
        self.v_cache = [glm.alloc(B * n_kv * S * hd * BF16) for _ in range(n_layers)]
        self.cache_pos = 0

    def free(self):
        glm = self.glm
        for ptr in self._ws.values():
            glm.free_buf(ptr)
        for ptr in self.k_cache:
            glm.free_buf(ptr)
        for ptr in self.v_cache:
            glm.free_buf(ptr)
        glm.free_buf(self.inv_freq)
        for ptr in self.weights.values():
            glm.free_buf(ptr)
        self._ws = {}
        self.k_cache = []
        self.v_cache = []
        self.weights = {}

    def __del__(self):
        if hasattr(self, '_ws') and self._ws:
            self.free()

    def reset_cache(self):
        self.cache_pos = 0

    def _upload_ids(self, input_ids):
        B, S = input_ids.shape
        ids_np = input_ids.cpu().numpy().astype(np.int32).flatten()
        nbytes = ids_np.nbytes
        ptr = self.glm.alloc(nbytes)
        self.glm.h2d(ptr, ids_np.tobytes())
        return ptr, nbytes

    def _read_logits(self, ptr, count):
        nbytes = count * BF16
        buf = ctypes.create_string_buffer(nbytes)
        self.glm.lib.glm_d2h(self.glm.ctx, buf, ctypes.c_void_p(ptr), nbytes)
        return np.frombuffer(buf.raw, dtype=np.uint16)

    def _final_norm_and_logits(self, count, src=None):
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

    def _extract_last_logits(self, count, last_indices_np):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size

        glm.h2d(self._ws["last_idx"], last_indices_np.tobytes())
        glm.index_select(self._ws["hidden_last"], self._ws["hidden_a"],
                          self._ws["last_idx"], hs, count)
        self._final_norm_and_logits(count, src=self._ws["hidden_last"])

    def forward(self, input_ids):
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

        ids_ptr, ids_nbytes = self._upload_ids(input_ids)
        glm.embedding(self._ws["hidden_a"], self.weights["model.embed_tokens.weight"],
                       ids_ptr, hs, BS)
        glm.free_buf(ids_ptr)

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

    def prefill(self, input_ids):
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
        assert self.cache_pos == 0, "Cache must be reset before prefill"

        ids_ptr, ids_nbytes = self._upload_ids(input_ids)
        glm.embedding(self._ws["hidden_a"], self.weights["model.embed_tokens.weight"],
                       ids_ptr, hs, BS)
        glm.free_buf(ids_ptr)

        glm.arange(self._ws["position_ids"], 0, 1, S)

        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              hd // 2, B, S)

        glm.causal_mask(self._ws["causal_mask"], S)

        for i in range(cfg.num_hidden_layers):
            pfx = f"model.layers.{i}"
            self._decoder_layer_prefill_flash(B, S, i, pfx)

        last_idx_np = np.array([S - 1], dtype=np.int32)
        self._extract_last_logits(1, last_idx_np)

        self.cache_pos = S

        logits_u16 = self._read_logits(self._ws["logits_buf"], vs)
        logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(1, vs)
        return torch.from_numpy(logits_f32.copy())

    def _decode_token(self, token_id):
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
        cached_len = self.cache_pos

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
            self._decoder_layer_decode_flash(B, S, i, pfx, cached_len)

        self._final_norm_and_logits(BS)

        self.cache_pos = cached_len + S

    def decode(self, input_ids):
        B, S = input_ids.shape
        assert B == 1 and S == 1, "Decode only supports B=1, S=1"
        vs = self.cfg.vocab_size

        token_id = input_ids[0, 0].item()
        self._decode_token(token_id)

        logits_u16 = self._read_logits(self._ws["logits_buf"], vs)
        logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(1, 1, vs)
        return torch.from_numpy(logits_f32.copy())

    def _argmax_logits(self, logits_ptr, count):
        self.glm.argmax(self._ws["argmax_idx"], logits_ptr, count)
        buf = ctypes.create_string_buffer(I32)
        self.glm.lib.glm_d2h(self.glm.ctx, buf, ctypes.c_void_p(self._ws["argmax_idx"]), I32)
        return int(np.frombuffer(buf.raw, dtype=np.int32)[0])

    def generate(self, input_ids, max_new_tokens=100, eos_token_ids=None):
        if eos_token_ids is None:
            eos_token_ids = {151645, 151643}

        B, S = input_ids.shape
        assert B == 1, "generate() only supports batch=1"

        cfg = self.cfg
        vs = cfg.vocab_size

        self.reset_cache()
        self.prefill(input_ids)

        next_token = self._argmax_logits(self._ws["logits_buf"], vs)
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            if next_token in eos_token_ids:
                break
            self._decode_token(next_token)
            next_token = self._argmax_logits(self._ws["logits_buf"], vs)
            generated.append(next_token)

        return generated

    def generate_tokens(self, input_ids, max_new_tokens=100, eos_token_ids=None):
        if eos_token_ids is None:
            eos_token_ids = {151645, 151643}

        B, S = input_ids.shape
        assert B == 1, "generate_tokens() only supports batch=1"

        cfg = self.cfg
        vs = cfg.vocab_size

        self.reset_cache()
        self.prefill(input_ids)

        next_token = self._argmax_logits(self._ws["logits_buf"], vs)
        yield next_token

        for _ in range(max_new_tokens - 1):
            if next_token in eos_token_ids:
                break
            self._decode_token(next_token)
            next_token = self._argmax_logits(self._ws["logits_buf"], vs)
            yield next_token

    def generate_text(self, prompt, tokenizer, max_new_tokens=100,
                      eos_token_ids=None, enable_thinking=True):
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
            input_tensor, max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        ))
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_batch(self, input_ids_list, ws, paged_kv, max_new_tokens=100,
                       eos_token_ids=None):
        if eos_token_ids is None:
            eos_token_ids = {151645, 151643}

        batch_size = len(input_ids_list)
        cfg = self.cfg
        vs = cfg.vocab_size

        all_logits = self.prefill_batch(input_ids_list, ws, paged_kv)

        next_tokens = [logits.argmax().item() for logits in all_logits]
        generated = [[t] for t in next_tokens]
        finished = [t in eos_token_ids for t in next_tokens]

        for _ in range(max_new_tokens - 1):
            if all(finished):
                break

            token_ids = [next_tokens[i] for i in range(batch_size)]
            all_logits = self.decode_batch(token_ids, ws, paged_kv)

            for i in range(batch_size):
                if finished[i]:
                    continue
                next_tokens[i] = all_logits[i].argmax().item()
                if next_tokens[i] in eos_token_ids:
                    finished[i] = True
                else:
                    generated[i].append(next_tokens[i])

        return generated

    def generate_text_batch(self, prompts, tokenizer, ws, paged_kv,
                            max_new_tokens=100, eos_token_ids=None,
                            enable_thinking=True):
        input_ids_list = []
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

    def _decoder_layer(self, B, S, pfx):
        cfg = self.cfg
        glm = self.glm
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, cfg.hidden_size, BS)

        self._attention(B, S, pfx)

        self._residual_and_mlp(pfx, BS)

    def _decoder_layer_prefill_flash(self, B, S, layer_idx, pfx):
        cfg = self.cfg
        glm = self.glm
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, cfg.hidden_size, BS)

        self._attention_prefill_flash(B, S, layer_idx, pfx)

        self._residual_and_mlp(pfx, BS)

    def _decoder_layer_decode_flash(self, B, S, layer_idx, pfx, cached_len):
        cfg = self.cfg
        glm = self.glm
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, cfg.hidden_size, BS)

        self._attention_decode_flash(B, S, layer_idx, pfx, cached_len)

        self._residual_and_mlp(pfx, BS)

    def _mlp(self, BS, pfx):
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

    def _residual_and_mlp(self, pfx, BS):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size

        glm.add(self._ws["hidden_b"], self._ws["hidden_a"],
                self._ws["o_proj_buf"], BS * hs)
        glm.rmsnorm(self._ws["normed"], self._ws["hidden_b"],
                     self.weights[f"{pfx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)
        self._mlp(BS, pfx)

    def _compute_qkv(self, pfx, BS, B, S):
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

    def _write_kv_flat(self, layer_idx, S, offset=0):
        cfg = self.cfg
        glm = self.glm
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        max_S = self.max_seq_len

        for h in range(n_kv):
            src_off = h * S * hd * BF16
            dst_off = (h * max_S * hd + offset * hd) * BF16
            glm.memcpy(self.k_cache[layer_idx] + dst_off,
                        self._ws["k_rope"] + src_off,
                        S * hd * BF16)
            glm.memcpy(self.v_cache[layer_idx] + dst_off,
                        self._ws["v_t"] + src_off,
                        S * hd * BF16)

    def _attention(self, B, S, pfx):
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

    def _attention_prefill_flash(self, B, S, layer_idx, pfx):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        max_S = self.max_seq_len
        BS = B * S

        self._compute_qkv(pfx, BS, B, S)

        self._write_kv_flat(layer_idx, S)

        kv_stride_h = max_S * hd
        kv_stride_n = hd

        glm.flash_prefill(
            self._ws["q_rope"],
            self.k_cache[layer_idx],
            self.v_cache[layer_idx],
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

    def _attention_decode_flash(self, B, S, layer_idx, pfx, cached_len):
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

        self._write_kv_flat(layer_idx, S, offset=cached_len)

        kv_stride_h = max_S * hd
        kv_stride_n = hd

        glm.flash_decode(
            self._ws["q_rope"],
            self.k_cache[layer_idx],
            self.v_cache[layer_idx],
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

    def prefill_batch(self, input_ids_list, ws, paged_kv):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        vs = cfg.vocab_size
        batch_size = len(input_ids_list)

        paged_kv.reset(batch_size)

        seq_lens = [len(ids) for ids in input_ids_list]
        total_tokens = sum(seq_lens)

        page_allocs = []
        for seq_idx, s in enumerate(seq_lens):
            start_page, num_pages = paged_kv.alloc_prefill_pages(seq_idx, s)
            page_allocs.append((start_page, num_pages))

        all_ids = []
        for ids in input_ids_list:
            all_ids.extend(ids)
        ids_np = np.array(all_ids, dtype=np.int32)
        ids_ptr = glm.alloc(ids_np.nbytes)
        glm.h2d(ids_ptr, ids_np.tobytes())

        glm.embedding(self._ws["hidden_a"], self.weights["model.embed_tokens.weight"],
                       ids_ptr, hs, total_tokens)
        glm.free_buf(ids_ptr)

        qo_indptr = [0]
        kv_indptr = [0]
        for s in seq_lens:
            qo_indptr.append(qo_indptr[-1] + s)
            kv_indptr.append(kv_indptr[-1] + s)
        qo_indptr_np = np.array(qo_indptr, dtype=np.int32)
        kv_indptr_np = np.array(kv_indptr, dtype=np.int32)

        pos_ids = []
        for s in seq_lens:
            pos_ids.extend(range(s))
        pos_ids_np = np.array(pos_ids, dtype=np.int32)
        glm.h2d(self._ws["position_ids"], pos_ids_np.tobytes())

        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              hd // 2, 1, total_tokens)

        glm.batch_prefill_ragged_plan(
            ws.float_ws, BATCH_FLOAT_WS_SIZE,
            ws.int_ws, ws.pinned_int_ws, BATCH_INT_WS_SIZE,
            ws.prefill_plan_info,
            qo_indptr_np.ctypes.data, kv_indptr_np.ctypes.data,
            total_tokens, batch_size,
            n_heads, n_kv, hd,
            1
        )

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

            qo_indptr_d = glm.alloc((batch_size + 1) * I32)
            kv_indptr_d = glm.alloc((batch_size + 1) * I32)
            glm.h2d(qo_indptr_d, qo_indptr_np.tobytes())
            glm.h2d(kv_indptr_d, kv_indptr_np.tobytes())

            glm.batch_prefill_ragged_run(
                self._ws["q_rope"], self._ws["k_rope"], self._ws["v_t"], self._ws["flash_out"],
                ws.float_ws, ws.int_ws,
                qo_indptr_d, kv_indptr_d,
                ws.prefill_plan_info,
                total_tokens, batch_size,
                n_heads, n_kv, hd,
                q_stride_n, q_stride_h,
                kv_stride_n, kv_stride_h,
                1, cfg.scaling
            )

            glm.free_buf(qo_indptr_d)
            glm.free_buf(kv_indptr_d)

            glm.linear(self._ws["o_proj_buf"], self._ws["flash_out"],
                         self.weights[f"{pfx}.self_attn.o_proj.weight"],
                         total_tokens, hs, n_heads * hd)

            self._residual_and_mlp(pfx, total_tokens)

        last_indices = []
        offset = 0
        for s in seq_lens:
            last_indices.append(offset + s - 1)
            offset += s
        last_idx_np = np.array(last_indices, dtype=np.int32)
        self._extract_last_logits(batch_size, last_idx_np)

        paged_kv.update_indptr()

        all_logits = []
        for i in range(batch_size):
            logits_ptr = self._ws["logits_buf"] + i * vs * BF16
            logits_u16 = self._read_logits(logits_ptr, vs)
            logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(vs)
            all_logits.append(torch.from_numpy(logits_f32.copy()))

        return all_logits

    def decode_batch(self, token_ids_list, ws, paged_kv):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        vs = cfg.vocab_size
        page_size = paged_kv.page_size
        batch_size = len(token_ids_list)

        write_locations = []
        for seq_idx in range(batch_size):
            abs_page, slot_in_page = paged_kv.alloc_decode_token(seq_idx)
            write_locations.append((abs_page, slot_in_page))

        paged_kv.update_indptr()

        all_ids = token_ids_list
        ids_np = np.array(all_ids, dtype=np.int32)
        ids_ptr = glm.alloc(ids_np.nbytes)
        glm.h2d(ids_ptr, ids_np.tobytes())

        glm.embedding(self._ws["hidden_a"], self.weights["model.embed_tokens.weight"],
                       ids_ptr, hs, batch_size)
        glm.free_buf(ids_ptr)

        pos_ids = []
        for seq_idx in range(batch_size):
            pos_ids.append(paged_kv.seq_kv_lens[seq_idx] - 1)
        pos_ids_np = np.array(pos_ids, dtype=np.int32)
        glm.h2d(self._ws["position_ids"], pos_ids_np.tobytes())

        glm.rotary_embedding(self._ws["cos"], self._ws["sin"], self.inv_freq,
                              self._ws["position_ids"],
                              hd // 2, batch_size, 1)

        glm.batch_decode_plan(
            ws.float_ws, BATCH_FLOAT_WS_SIZE,
            ws.int_ws, ws.pinned_int_ws, BATCH_INT_WS_SIZE,
            ws.decode_plan_info,
            paged_kv.indptr_h,
            batch_size,
            n_heads, n_kv, page_size
        )

        for i in range(cfg.num_hidden_layers):
            pfx = f"model.layers.{i}"

            glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                         self.weights[f"{pfx}.input_layernorm.weight"],
                         cfg.rms_norm_eps, hs, batch_size)

            self._compute_qkv(pfx, batch_size, batch_size, 1)

            for seq_idx in range(batch_size):
                abs_page, slot_in_page = write_locations[seq_idx]
                for h in range(n_kv):
                    page_offset = abs_page * n_kv * page_size * hd
                    kv_head_offset = page_offset + h * page_size * hd
                    token_offset = kv_head_offset + slot_in_page * hd
                    src_off = (seq_idx * n_kv + h) * hd * BF16
                    glm.memcpy(paged_kv.k_data[i] + token_offset * BF16,
                                self._ws["k_rope"] + src_off,
                                hd * BF16)
                    glm.memcpy(paged_kv.v_data[i] + token_offset * BF16,
                                self._ws["v_t"] + src_off,
                                hd * BF16)

            glm.batch_decode_run(
                self._ws["q_rope"], self._ws["flash_out"],
                paged_kv.k_data[i], paged_kv.v_data[i],
                paged_kv.indices, paged_kv.indptr_d, paged_kv.last_page_len,
                ws.float_ws, ws.int_ws,
                ws.decode_plan_info,
                n_heads, n_kv, hd, page_size, cfg.scaling
            )

            glm.linear(self._ws["o_proj_buf"], self._ws["flash_out"],
                         self.weights[f"{pfx}.self_attn.o_proj.weight"],
                          batch_size, hs, n_heads * hd)

            self._residual_and_mlp(pfx, batch_size)

        self._final_norm_and_logits(batch_size)

        all_logits = []
        for seq_idx in range(batch_size):
            logits_ptr = self._ws["logits_buf"] + seq_idx * vs * BF16
            logits_u16 = self._read_logits(logits_ptr, vs)
            logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(vs)
            all_logits.append(torch.from_numpy(logits_f32.copy()))

        return all_logits
