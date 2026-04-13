import ctypes
import json
import os
import numpy as np
import torch
from safetensors import safe_open
from helpers import GlmOps, get_model_path

BF16 = 2  # bytes per bfloat16
I32 = 4   # bytes per int32
FLASH_TMP_SIZE = 32 * 1024 * 1024  # 32MB workspace for FlashInfer
BATCH_FLOAT_WS_SIZE = 128 * 1024 * 1024  # 128MB float workspace
BATCH_INT_WS_SIZE = 8 * 1024 * 1024  # 8MB int workspace (GPU)
BATCH_PINNED_INT_WS_SIZE = 8 * 1024 * 1024  # 8MB pinned host int workspace
PAGE_SIZE = 16
DECODE_PLAN_INFO_SIZE = 10  # int64s
PREFILL_PLAN_INFO_SIZE = 15  # int64s


def _f32_to_bf16_bytes(arr):
    u32 = arr.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def _bf16_bytes_to_f32(data):
    u16 = np.frombuffer(data, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


class PagedKVCache:
    def __init__(self, glm, n_kv, hd, n_layers, max_pages, page_size=PAGE_SIZE):
        self.glm = glm
        self.n_kv = n_kv
        self.hd = hd
        self.n_layers = n_layers
        self.max_pages = max_pages
        self.page_size = page_size
        self.page_stride = n_kv * page_size * hd * BF16
        self.k_data = [glm.alloc(max_pages * n_kv * page_size * hd * BF16) for _ in range(n_layers)]
        self.v_data = [glm.alloc(max_pages * n_kv * page_size * hd * BF16) for _ in range(n_layers)]
        self.indices = glm.alloc(max_pages * I32)
        self.indptr_d = None
        self.last_page_len = None
        self.indptr_h = None
        self.last_page_len_h = None
        self.num_pages_used = 0
        self.seq_page_counts = []
        self.seq_kv_lens = []

        indptr_np = np.arange(max_pages, dtype=np.int32)
        glm.h2d(self.indices, indptr_np.tobytes())

    def free(self):
        glm = self.glm
        for ptr in self.k_data:
            glm.free_buf(ptr)
        for ptr in self.v_data:
            glm.free_buf(ptr)
        glm.free_buf(self.indices)
        if self.indptr_d is not None:
            glm.free_buf(self.indptr_d)
        if self.last_page_len is not None:
            glm.free_buf(self.last_page_len)
        if self.indptr_h is not None:
            glm.free_pinned(self.indptr_h)
        if self.last_page_len_h is not None:
            glm.free_pinned(self.last_page_len_h)
        self.k_data = []
        self.v_data = []

    def reset(self, batch_size):
        self.num_pages_used = 0
        self.seq_page_counts = [0] * batch_size
        self.seq_kv_lens = [0] * batch_size

        if self.indptr_d is not None:
            self.glm.free_buf(self.indptr_d)
        if self.last_page_len is not None:
            self.glm.free_buf(self.last_page_len)
        if self.indptr_h is not None:
            self.glm.free_pinned(self.indptr_h)
        if self.last_page_len_h is not None:
            self.glm.free_pinned(self.last_page_len_h)

        self.indptr_d = self.glm.alloc((batch_size + 1) * I32)
        self.last_page_len = self.glm.alloc(batch_size * I32)
        self.indptr_h = self.glm.alloc_pinned((batch_size + 1) * I32)
        self.last_page_len_h = self.glm.alloc_pinned(batch_size * I32)

    def alloc_pages(self, seq_idx, num_tokens):
        page_size = self.page_size
        num_new_pages = (num_tokens + page_size - 1) // page_size
        start_page = self.num_pages_used
        self.num_pages_used += num_new_pages
        self.seq_page_counts[seq_idx] += num_new_pages
        self.seq_kv_lens[seq_idx] += num_tokens
        return start_page, num_new_pages

    def alloc_prefill_pages(self, seq_idx, seq_len):
        page_size = self.page_size
        num_pages = (seq_len + page_size - 1) // page_size
        start_page = self.num_pages_used
        self.num_pages_used += num_pages
        self.seq_page_counts[seq_idx] = num_pages
        self.seq_kv_lens[seq_idx] = seq_len
        return start_page, num_pages

    def alloc_decode_token(self, seq_idx):
        kv_len = self.seq_kv_lens[seq_idx]
        page_size = self.page_size
        page_idx_in_seq = kv_len // page_size
        if page_idx_in_seq >= self.seq_page_counts[seq_idx]:
            start_page = self.num_pages_used
            self.num_pages_used += 1
            self.seq_page_counts[seq_idx] += 1
        self.seq_kv_lens[seq_idx] = kv_len + 1
        abs_page = sum(self.seq_page_counts[:seq_idx]) + page_idx_in_seq
        slot_in_page = kv_len % page_size
        return abs_page, slot_in_page

    def update_indptr(self):
        batch_size = len(self.seq_page_counts)
        indptr = [0] * (batch_size + 1)
        for i in range(batch_size):
            indptr[i + 1] = indptr[i] + self.seq_page_counts[i]
        indptr_np = np.array(indptr, dtype=np.int32)
        last_page_len_list = []
        for i in range(batch_size):
            kv_len = self.seq_kv_lens[i]
            remainder = kv_len % self.page_size
            last_page_len_list.append(remainder if remainder != 0 else self.page_size if kv_len > 0 else 0)
        last_page_len_np = np.array(last_page_len_list, dtype=np.int32)
        ctypes.memset(self.indptr_h, 0, (batch_size + 1) * I32)
        ctypes.memset(self.last_page_len_h, 0, batch_size * I32)
        ctypes.memmove(self.indptr_h, indptr_np.tobytes(), (batch_size + 1) * I32)
        ctypes.memmove(self.last_page_len_h, last_page_len_np.tobytes(), batch_size * I32)
        self.glm.h2d(self.indptr_d, indptr_np.tobytes())
        self.glm.h2d(self.last_page_len, last_page_len_np.tobytes())


class WorkspaceBuffers:
    def __init__(self, glm):
        self.glm = glm
        self.float_ws = glm.alloc(BATCH_FLOAT_WS_SIZE)
        self.int_ws = glm.alloc(BATCH_INT_WS_SIZE)
        self.pinned_int_ws = glm.alloc_pinned(BATCH_PINNED_INT_WS_SIZE)
        self.decode_plan_info = glm.alloc_pinned(DECODE_PLAN_INFO_SIZE * 8)
        self.prefill_plan_info = glm.alloc_pinned(PREFILL_PLAN_INFO_SIZE * 8)

    def free(self):
        glm = self.glm
        glm.free_buf(self.float_ws)
        glm.free_buf(self.int_ws)
        glm.free_pinned(self.pinned_int_ws)
        glm.free_pinned(self.decode_plan_info)
        glm.free_pinned(self.prefill_plan_info)
        self.float_ws = 0
        self.int_ws = 0


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
            "logits_buf": glm.alloc(BS * vs * BF16),
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

        glm.linear(self._ws["logits_buf"], self._ws["normed"],
                    self.weights["lm_head.weight"],
                    BS, vs, hs)

        logits_count = BS * vs
        logits_u16 = self._read_logits(self._ws["logits_buf"], logits_count)
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

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights["model.norm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        glm.linear(self._ws["logits_buf"], self._ws["normed"],
                    self.weights["lm_head.weight"],
                    BS, vs, hs)

        self.cache_pos = S

        logits_count = BS * vs
        logits_u16 = self._read_logits(self._ws["logits_buf"], logits_count)
        logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(B, S, vs)
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

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights["model.norm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        glm.linear(self._ws["logits_buf"], self._ws["normed"],
                    self.weights["lm_head.weight"],
                    BS, vs, hs)

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

        last_logits_ptr = self._ws["logits_buf"] + (S - 1) * vs * BF16
        next_token = self._argmax_logits(last_logits_ptr, vs)
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

        last_logits_ptr = self._ws["logits_buf"] + (S - 1) * vs * BF16
        next_token = self._argmax_logits(last_logits_ptr, vs)
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

    def _decoder_layer(self, B, S, pfx):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        inter = cfg.intermediate_size
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        self._attention(B, S, pfx)

        glm.add(self._ws["hidden_b"], self._ws["hidden_a"],
                self._ws["o_proj_buf"], BS * hs)

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_b"],
                     self.weights[f"{pfx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        self._mlp(B, S, pfx)

    def _decoder_layer_prefill(self, B, S, layer_idx, pfx):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        inter = cfg.intermediate_size
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        self._attention_prefill(B, S, layer_idx, pfx)

        glm.add(self._ws["hidden_b"], self._ws["hidden_a"],
                self._ws["o_proj_buf"], BS * hs)

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_b"],
                     self.weights[f"{pfx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        self._mlp(B, S, pfx)

    def _decoder_layer_prefill_flash(self, B, S, layer_idx, pfx):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        inter = cfg.intermediate_size
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        self._attention_prefill_flash(B, S, layer_idx, pfx)

        glm.add(self._ws["hidden_b"], self._ws["hidden_a"],
                self._ws["o_proj_buf"], BS * hs)

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_b"],
                     self.weights[f"{pfx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        self._mlp(B, S, pfx)

    def _decoder_layer_decode(self, B, S, layer_idx, pfx, cached_len):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        inter = cfg.intermediate_size
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        self._attention_decode(B, S, layer_idx, pfx, cached_len)

        glm.add(self._ws["hidden_b"], self._ws["hidden_a"],
                self._ws["o_proj_buf"], BS * hs)

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_b"],
                     self.weights[f"{pfx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        self._mlp(B, S, pfx)

    def _decoder_layer_decode_flash(self, B, S, layer_idx, pfx, cached_len):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        inter = cfg.intermediate_size
        BS = B * S

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights[f"{pfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        self._attention_decode_flash(B, S, layer_idx, pfx, cached_len)

        glm.add(self._ws["hidden_b"], self._ws["hidden_a"],
                self._ws["o_proj_buf"], BS * hs)

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_b"],
                     self.weights[f"{pfx}.post_attention_layernorm.weight"],
                     cfg.rms_norm_eps, hs, BS)

        self._mlp(B, S, pfx)

    def _mlp(self, B, S, pfx):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        inter = cfg.intermediate_size
        BS = B * S

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

    def _attention(self, B, S, pfx):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        n_groups = cfg.num_key_value_groups
        BS = B * S

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

    def _attention_prefill(self, B, S, layer_idx, pfx):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        n_groups = cfg.num_key_value_groups
        max_S = self.max_seq_len
        BS = B * S

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

        for h in range(n_kv):
            src_off = h * S * hd * BF16
            dst_off = h * max_S * hd * BF16
            glm.memcpy(self.k_cache[layer_idx] + dst_off,
                        self._ws["k_rope"] + src_off,
                        S * hd * BF16)
            glm.memcpy(self.v_cache[layer_idx] + dst_off,
                        self._ws["v_t"] + src_off,
                        S * hd * BF16)

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

    def _attention_decode(self, B, S, layer_idx, pfx, cached_len):
        cfg = self.cfg
        glm = self.glm
        hs = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        n_groups = cfg.num_key_value_groups
        max_S = self.max_seq_len
        BS = B * S
        total_len = cached_len + S

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

        for h in range(n_kv):
            src_off = h * S * hd * BF16
            dst_off = (h * max_S * hd + cached_len * hd) * BF16
            glm.memcpy(self.k_cache[layer_idx] + dst_off,
                        self._ws["k_rope"] + src_off,
                        S * hd * BF16)
            glm.memcpy(self.v_cache[layer_idx] + dst_off,
                        self._ws["v_t"] + src_off,
                        S * hd * BF16)

        head_stride = max_S * hd

        if n_groups > 1:
            glm.expand_dim1_strided(self._ws["k_expanded"], self.k_cache[layer_idx],
                                    n_heads, n_kv, total_len, hd, B, head_stride)
            glm.expand_dim1_strided(self._ws["v_expanded"], self.v_cache[layer_idx],
                                    n_heads, n_kv, total_len, hd, B, head_stride)
        else:
            for h in range(n_kv):
                src_off = h * max_S * hd * BF16
                dst_off = h * total_len * hd * BF16
                glm.memcpy(self._ws["k_expanded"] + dst_off,
                            self.k_cache[layer_idx] + src_off,
                            total_len * hd * BF16)
                glm.memcpy(self._ws["v_expanded"] + dst_off,
                            self.v_cache[layer_idx] + src_off,
                            total_len * hd * BF16)

        glm.bmm(self._ws["attn_scores"], self._ws["q_rope"], self._ws["k_expanded"],
                 cfg.scaling, 0.0, B * n_heads, S, total_len, hd, 1)

        glm.softmax(self._ws["attn_scores"], self._ws["attn_scores"],
                     None,
                     total_len, B * n_heads * S)

        glm.bmm(self._ws["attn_out"], self._ws["attn_scores"], self._ws["v_expanded"],
                 1.0, 0.0, B * n_heads, S, hd, total_len, 0)

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

        for h in range(n_kv):
            src_off = h * S * hd * BF16
            dst_off = h * max_S * hd * BF16
            glm.memcpy(self.k_cache[layer_idx] + dst_off,
                        self._ws["k_rope"] + src_off,
                        S * hd * BF16)
            glm.memcpy(self.v_cache[layer_idx] + dst_off,
                        self._ws["v_t"] + src_off,
                        S * hd * BF16)

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

        for h in range(n_kv):
            src_off = h * S * hd * BF16
            dst_off = (h * max_S * hd + cached_len * hd) * BF16
            glm.memcpy(self.k_cache[layer_idx] + dst_off,
                        self._ws["k_rope"] + src_off,
                        S * hd * BF16)
            glm.memcpy(self.v_cache[layer_idx] + dst_off,
                        self._ws["v_t"] + src_off,
                        S * hd * BF16)

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

            glm.linear(self._ws["q_buf"], self._ws["normed"],
                         self.weights[f"{pfx}.self_attn.q_proj.weight"],
                         total_tokens, n_heads * hd, hs)
            glm.linear(self._ws["k_buf"], self._ws["normed"],
                         self.weights[f"{pfx}.self_attn.k_proj.weight"],
                         total_tokens, n_kv * hd, hs)
            glm.linear(self._ws["v_buf"], self._ws["normed"],
                         self.weights[f"{pfx}.self_attn.v_proj.weight"],
                         total_tokens, n_kv * hd, hs)

            glm.rmsnorm(self._ws["q_normed"], self._ws["q_buf"],
                         self.weights[f"{pfx}.self_attn.q_norm.weight"],
                         cfg.rms_norm_eps, hd, total_tokens * n_heads)
            glm.rmsnorm(self._ws["k_normed"], self._ws["k_buf"],
                         self.weights[f"{pfx}.self_attn.k_norm.weight"],
                         cfg.rms_norm_eps, hd, total_tokens * n_kv)

            glm.transpose_4d(self._ws["q_t"], self._ws["q_normed"],
                              1, total_tokens, n_heads, hd, 0, 2, 1, 3)
            glm.transpose_4d(self._ws["k_t"], self._ws["k_normed"],
                              1, total_tokens, n_kv, hd, 0, 2, 1, 3)
            glm.transpose_4d(self._ws["v_t"], self._ws["v_buf"],
                              1, total_tokens, n_kv, hd, 0, 2, 1, 3)

            glm.apply_rotary_pos_emb(self._ws["q_rope"], self._ws["q_t"],
                                      self._ws["cos"], self._ws["sin"],
                                      hd, n_heads, total_tokens, 1, 1)
            glm.apply_rotary_pos_emb(self._ws["k_rope"], self._ws["k_t"],
                                      self._ws["cos"], self._ws["sin"],
                                      hd, n_kv, total_tokens, 1, 1)

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

            glm.add(self._ws["hidden_b"], self._ws["hidden_a"],
                     self._ws["o_proj_buf"], total_tokens * hs)

            glm.rmsnorm(self._ws["normed"], self._ws["hidden_b"],
                         self.weights[f"{pfx}.post_attention_layernorm.weight"],
                         cfg.rms_norm_eps, hs, total_tokens)

            self._mlp(total_tokens, 1, pfx)

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights["model.norm.weight"],
                     cfg.rms_norm_eps, hs, total_tokens)

        glm.linear(self._ws["logits_buf"], self._ws["normed"],
                     self.weights["lm_head.weight"],
                     total_tokens, vs, hs)

        paged_kv.update_indptr()

        all_logits = []
        offset = 0
        for s in seq_lens:
            logits_ptr = self._ws["logits_buf"] + (offset + s - 1) * vs * BF16
            logits_u16 = self._read_logits(logits_ptr, vs)
            logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(vs)
            all_logits.append(torch.from_numpy(logits_f32.copy()))
            offset += s

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

            glm.linear(self._ws["q_buf"], self._ws["normed"],
                         self.weights[f"{pfx}.self_attn.q_proj.weight"],
                         batch_size, n_heads * hd, hs)
            glm.linear(self._ws["k_buf"], self._ws["normed"],
                         self.weights[f"{pfx}.self_attn.k_proj.weight"],
                         batch_size, n_kv * hd, hs)
            glm.linear(self._ws["v_buf"], self._ws["normed"],
                         self.weights[f"{pfx}.self_attn.v_proj.weight"],
                         batch_size, n_kv * hd, hs)

            glm.rmsnorm(self._ws["q_normed"], self._ws["q_buf"],
                         self.weights[f"{pfx}.self_attn.q_norm.weight"],
                         cfg.rms_norm_eps, hd, batch_size * n_heads)
            glm.rmsnorm(self._ws["k_normed"], self._ws["k_buf"],
                         self.weights[f"{pfx}.self_attn.k_norm.weight"],
                         cfg.rms_norm_eps, hd, batch_size * n_kv)

            glm.transpose_4d(self._ws["q_t"], self._ws["q_normed"],
                              batch_size, 1, n_heads, hd, 0, 2, 1, 3)
            glm.transpose_4d(self._ws["k_t"], self._ws["k_normed"],
                              batch_size, 1, n_kv, hd, 0, 2, 1, 3)
            glm.transpose_4d(self._ws["v_t"], self._ws["v_buf"],
                              batch_size, 1, n_kv, hd, 0, 2, 1, 3)

            glm.apply_rotary_pos_emb(self._ws["q_rope"], self._ws["q_t"],
                                      self._ws["cos"], self._ws["sin"],
                                      hd, n_heads, 1, batch_size, 1)
            glm.apply_rotary_pos_emb(self._ws["k_rope"], self._ws["k_t"],
                                      self._ws["cos"], self._ws["sin"],
                                      hd, n_kv, 1, batch_size, 1)

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

            glm.add(self._ws["hidden_b"], self._ws["hidden_a"],
                     self._ws["o_proj_buf"], batch_size * hs)

            glm.rmsnorm(self._ws["normed"], self._ws["hidden_b"],
                         self.weights[f"{pfx}.post_attention_layernorm.weight"],
                         cfg.rms_norm_eps, hs, batch_size)

            self._mlp(batch_size, 1, pfx)

        glm.rmsnorm(self._ws["normed"], self._ws["hidden_a"],
                     self.weights["model.norm.weight"],
                     cfg.rms_norm_eps, hs, batch_size)

        glm.linear(self._ws["logits_buf"], self._ws["normed"],
                     self.weights["lm_head.weight"],
                     batch_size, vs, hs)

        all_logits = []
        for seq_idx in range(batch_size):
            logits_ptr = self._ws["logits_buf"] + seq_idx * vs * BF16
            logits_u16 = self._read_logits(logits_ptr, vs)
            logits_f32 = _bf16_bytes_to_f32(logits_u16.tobytes()).reshape(vs)
            all_logits.append(torch.from_numpy(logits_f32.copy()))

        return all_logits
