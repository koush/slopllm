import ctypes
import gc
import json
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from helpers import GlmOps, has_model_cached, get_model_path
from qwen35_model import (
    Qwen35Config, Qwen35Model, Qwen35TorchModel,
    _f32_to_bf16_bytes, _bf16_bytes_to_f32,
    torch_gemma_rmsnorm, torch_rmsnorm_gated,
    torch_gdn_recurrent_step, torch_causal_conv1d,
)

QWEN35_REPO = "Qwen/Qwen3.5-0.8B"

BF16 = 2
F32 = 4

pytestmark = pytest.mark.skipif(
    not has_model_cached(QWEN35_REPO),
    reason=f"{QWEN35_REPO} not in HF cache"
)


def _upload_bf16(glm, tensor_np):
    bf16_bytes = _f32_to_bf16_bytes(tensor_np)
    ptr = glm.alloc(len(bf16_bytes))
    glm.h2d(ptr, bf16_bytes)
    return ptr


def _upload_f32(glm, tensor_np):
    data = tensor_np.astype(np.float32).tobytes()
    ptr = glm.alloc(len(data))
    glm.h2d(ptr, data)
    return ptr


def _download_bf16(glm, ptr, count):
    nbytes = count * BF16
    buf = ctypes.create_string_buffer(nbytes)
    glm.d2h(buf, ptr, nbytes)
    return _bf16_bytes_to_f32(buf.raw).copy()


def _download_f32(glm, ptr, count):
    nbytes = count * F32
    buf = ctypes.create_string_buffer(nbytes)
    glm.d2h(buf, ptr, nbytes)
    return np.frombuffer(buf.raw, dtype=np.float32).copy()


def _to_bf16(t):
    return t.to(torch.bfloat16).to(torch.float32)


@pytest.fixture(scope="module")
def glm():
    ops = GlmOps()
    yield ops
    ops.lib.glm_free(ops.ctx)
    ops.ctx = None


@pytest.fixture(scope="module")
def model_dir():
    return get_model_path(QWEN35_REPO)


@pytest.fixture(scope="module")
def cfg(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        return Qwen35Config(json.load(f))


@pytest.fixture(scope="module")
def qwen35_model(glm):
    model = Qwen35Model.from_pretrained(glm, QWEN35_REPO, max_batch=1, max_seq_len=64)
    yield model
    model.free()
    gc.collect(); torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def gdn_state(glm, qwen35_model):
    gs = qwen35_model.create_gdn_state()
    yield gs
    gs.free()


class TestGemmaRMSNorm:
    def test_gemma_vs_standard(self):
        dim = 128
        batch = 4
        eps = 1e-6
        torch.manual_seed(42)

        x = torch.randn(batch, dim, dtype=torch.bfloat16)
        weight = torch.randn(dim, dtype=torch.bfloat16)

        standard = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()
        gemma = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + weight.float())

        diff = (standard - gemma).abs().max().item()
        assert diff > 0.01, f"GemmaRMSNorm should differ from standard RMSNorm, but diff={diff}"


class TestGDNLayerPrefill:
    def test_gdn_layer_output(self, glm, cfg, model_dir):
        device = torch.device("cuda", glm.device)
        layer_idx = 0
        assert cfg.layer_types[layer_idx] == "linear_attention", "Layer 0 should be GDN"

        lin_h = cfg.linear_num_key_heads
        lin_kd = cfg.linear_key_head_dim
        lin_vd = cfg.linear_value_head_dim
        conv_dim = lin_h * (lin_kd * 2 + lin_vd)
        z_dim = lin_h * lin_vd
        kernel_size = cfg.linear_conv_kernel_dim
        hs = cfg.hidden_size
        S = 4

        layer_prefix = f"layers.{layer_idx}"
        sprefix = f"model.language_model.layers.{layer_idx}"
        weights_bf16 = {}
        weights_f32 = {}
        for sf in os.listdir(model_dir):
            if not sf.endswith('.safetensors'):
                continue
            with safe_open(os.path.join(model_dir, sf), framework="pt", device="cpu") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    if not (key.startswith(sprefix) or key.endswith("embed_tokens.weight")):
                        continue
                    wname = key[len("model.language_model."):] if key.startswith("model.language_model.") else key
                    if wname.endswith("A_log"):
                        weights_f32[wname] = t.float().to(device)
                    elif wname.endswith("norm.weight") and "linear_attn" in wname:
                        weights_f32[wname] = t.float().to(device)
                    elif wname.endswith("input_layernorm.weight") or wname.endswith("post_attention_layernorm.weight") or wname.endswith("q_norm.weight") or wname.endswith("k_norm.weight"):
                        weights_f32[wname] = t.float().to(device)
                    else:
                        weights_bf16[wname] = t.bfloat16().to(device)

        torch.manual_seed(42)
        hidden_input = torch.randn(S, hs, dtype=torch.bfloat16, device=device) * 0.1

        with torch.no_grad():
            norm_w = weights_f32[f"{layer_prefix}.input_layernorm.weight"]
            normed = torch_gemma_rmsnorm(hidden_input, norm_w, cfg.rms_norm_eps)

            mixed_qkv = F.linear(normed, weights_bf16[f"{layer_prefix}.linear_attn.in_proj_qkv.weight"])
            a_raw = F.linear(normed, weights_bf16[f"{layer_prefix}.linear_attn.in_proj_a.weight"])
            b_raw = F.linear(normed, weights_bf16[f"{layer_prefix}.linear_attn.in_proj_b.weight"])
            z = F.linear(normed, weights_bf16[f"{layer_prefix}.linear_attn.in_proj_z.weight"])

            mixed_qkv_t = mixed_qkv.t().contiguous()
            conv_w = weights_bf16[f"{layer_prefix}.linear_attn.conv1d.weight"].squeeze(1)
            conv_out_t = torch_causal_conv1d(mixed_qkv_t.unsqueeze(0), conv_w, kernel_size)
            mixed_qkv_conv = conv_out_t.squeeze(0).t().contiguous()

            q, k, v = mixed_qkv_conv.split([lin_h * lin_kd, lin_h * lin_kd, lin_h * lin_vd], dim=-1)
            q = q.view(S, lin_h, lin_kd)
            k = k.view(S, lin_h, lin_kd)
            v = v.view(S, lin_h, lin_vd)
            z = z.view(S, lin_h, lin_vd)
            a_raw = a_raw.view(S, lin_h)
            b_raw = b_raw.view(S, lin_h)

            A_log = weights_f32[f"{layer_prefix}.linear_attn.A_log"]
            dt_bias = weights_bf16[f"{layer_prefix}.linear_attn.dt_bias"].float()
            beta = torch.sigmoid(b_raw.float())
            g = -A_log.float().exp().unsqueeze(0) * F.softplus(a_raw.float() + dt_bias.float().unsqueeze(0))

            state = torch.zeros(lin_h, lin_kd, lin_vd, dtype=torch.float32, device=device)
            outputs = []
            for t_idx in range(S):
                q_t = _to_bf16(q[t_idx:t_idx+1])
                k_t = _to_bf16(k[t_idx:t_idx+1])
                v_t = _to_bf16(v[t_idx:t_idx+1])
                o_t, state = torch_gdn_recurrent_step(q_t, k_t, v_t, beta[t_idx:t_idx+1], g[t_idx:t_idx+1], state, lin_kd, lin_vd)
                outputs.append(o_t.squeeze(0))

            gdn_out = torch.stack(outputs).bfloat16()
            norm_weight = weights_f32[f"{layer_prefix}.linear_attn.norm.weight"]
            gated_out = torch_rmsnorm_gated(gdn_out, z, norm_weight, cfg.rms_norm_eps)

            out_2d = gated_out.reshape(S, -1)
            attn_out = F.linear(out_2d, weights_bf16[f"{layer_prefix}.linear_attn.out_proj.weight"])

            post_norm_w = weights_f32[f"{layer_prefix}.post_attention_layernorm.weight"]
            residual = hidden_input + attn_out
            post_normed = torch_gemma_rmsnorm(residual, post_norm_w, cfg.rms_norm_eps)

            gate = F.linear(post_normed, weights_bf16[f"{layer_prefix}.mlp.gate_proj.weight"])
            up = F.linear(post_normed, weights_bf16[f"{layer_prefix}.mlp.up_proj.weight"])
            silu_out = F.silu(gate) * up
            mlp_out = F.linear(silu_out, weights_bf16[f"{layer_prefix}.mlp.down_proj.weight"])

            ref_output = residual + mlp_out

        print(f"  GDN layer 0 output shape: {ref_output.shape}")
        print(f"  Output stats: mean={ref_output.float().mean():.4f}, std={ref_output.float().std():.4f}")
        print(f"  First few values: {ref_output[0, :5].float().tolist()}")


class TestFullAttentionLayer:
    def test_full_attn_layer_output(self, cfg, model_dir):
        device = torch.device("cuda:0")
        for layer_idx in cfg.full_attn_layer_indices[:1]:
            lin_h = cfg.linear_num_key_heads
            hs = cfg.hidden_size
            n_heads = cfg.num_attention_heads
            n_kv = cfg.num_key_value_heads
            hd = cfg.head_dim
            S = 4
            rope_dim = int(hd * cfg.partial_rotary_factor)

            prefix = f"model.language_model.layers.{layer_idx}"
            layer_prefix = f"layers.{layer_idx}"
            weights_bf16 = {}
            weights_f32 = {}
            for sf in os.listdir(model_dir):
                if not sf.endswith('.safetensors'):
                    continue
                with safe_open(os.path.join(model_dir, sf), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        t = f.get_tensor(key)
                        if not (key.startswith(prefix) or key.endswith("embed_tokens.weight")):
                            continue
                        wname = key[len("model.language_model."):] if key.startswith("model.language_model.") else key
                        if wname.endswith("q_norm.weight") or wname.endswith("k_norm.weight") or wname.endswith("input_layernorm.weight") or wname.endswith("post_attention_layernorm.weight"):
                            weights_f32[wname] = t.float().to(device)
                        else:
                            weights_bf16[wname] = t.bfloat16().to(device)

            torch.manual_seed(42)
            hidden_input = torch.randn(S, hs, dtype=torch.bfloat16, device=device) * 0.1

            with torch.no_grad():
                norm_w = weights_f32[f"{layer_prefix}.input_layernorm.weight"]
                norm_w_gemma = 1.0 + norm_w
                normed = torch_gemma_rmsnorm(hidden_input, norm_w_gemma, cfg.rms_norm_eps)

                q_total_dim = n_heads * hd
                q_full = F.linear(normed, weights_bf16[f"{layer_prefix}.self_attn.q_proj.weight"])
                k = F.linear(normed, weights_bf16[f"{layer_prefix}.self_attn.k_proj.weight"])
                v = F.linear(normed, weights_bf16[f"{layer_prefix}.self_attn.v_proj.weight"])

                q = q_full[:, :q_total_dim]
                gate = q_full[:, q_total_dim:]
                q = q.view(S, n_heads, hd)
                gate = gate.view(S, n_heads, hd)

                q_norm_w = weights_f32[f"{layer_prefix}.self_attn.q_norm.weight"]
                k_norm_w = weights_f32[f"{layer_prefix}.self_attn.k_norm.weight"]

                q_normed = torch_gemma_rmsnorm(q, q_norm_w, cfg.rms_norm_eps)
                k = k.view(S, n_kv, hd)
                k_normed = torch_gemma_rmsnorm(k, k_norm_w, cfg.rms_norm_eps)
                v = v.view(S, n_kv, hd)

                inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
                positions = torch.arange(S, dtype=torch.float32, device=device)
                freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
                emb = torch.cat([freqs, freqs], dim=-1)
                cos = emb.cos().to(torch.bfloat16)
                sin = emb.sin().to(torch.bfloat16)

                full_cos = torch.ones(S, hd, device=device, dtype=torch.bfloat16)
                full_sin = torch.zeros(S, hd, device=device, dtype=torch.bfloat16)
                full_cos[:, :rope_dim] = cos
                full_sin[:, :rope_dim] = sin

                def apply_rotary(x, cos, sin):
                    x1 = x[..., ::2]
                    x2 = x[..., 1::2]
                    c = cos[..., ::2]
                    s = sin[..., ::2]
                    o1 = x1 * c - x2 * s
                    o2 = x1 * s + x2 * c
                    return torch.stack([o1, o2], dim=-1).flatten(-2).to(torch.bfloat16)

                q_rope = apply_rotary(q_normed, full_cos.unsqueeze(1).expand(-1, n_heads, -1),
                                       full_sin.unsqueeze(1).expand(-1, n_heads, -1))
                k_rope = apply_rotary(k_normed, full_cos.unsqueeze(1).expand(-1, n_kv, -1),
                                       full_sin.unsqueeze(1).expand(-1, n_kv, -1))

                n_groups = n_heads // n_kv
                if n_groups > 1:
                    k_exp = k_rope.unsqueeze(2).expand(-1, -1, n_groups, -1).reshape(S, n_heads, hd)
                    v_exp = v.unsqueeze(2).expand(-1, -1, n_groups, -1).reshape(S, n_heads, hd)
                else:
                    k_exp = k_rope
                    v_exp = v

                q_h = q_rope.transpose(0, 1)
                k_h = k_exp.transpose(0, 1)
                v_h = v_exp.transpose(0, 1)
                scores = torch.matmul(q_h.float(), k_h.float().transpose(-2, -1)) * cfg.scaling
                mask = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1)
                scores = scores + mask.unsqueeze(0)
                attn_weights = torch.softmax(scores, dim=-1).to(torch.bfloat16)
                attn_output = torch.matmul(attn_weights, v_h)

                gate_sigmoid = torch.sigmoid(gate)
                attn_output = attn_output * gate_sigmoid.transpose(0, 1)

                attn_output = attn_output.transpose(0, 1).reshape(S, n_heads * hd)
                o = F.linear(attn_output, weights_bf16[f"{layer_prefix}.self_attn.o_proj.weight"])

                residual = hidden_input + o
                post_norm_w = weights_f32[f"{layer_prefix}.post_attention_layernorm.weight"]
                post_normed = torch_gemma_rmsnorm(residual, post_norm_w, cfg.rms_norm_eps)

                gate_proj = F.linear(post_normed, weights_bf16[f"{layer_prefix}.mlp.gate_proj.weight"])
                up_proj = F.linear(post_normed, weights_bf16[f"{layer_prefix}.mlp.up_proj.weight"])
                silu_out = F.silu(gate_proj) * up_proj
                mlp_out = F.linear(silu_out, weights_bf16[f"{layer_prefix}.mlp.down_proj.weight"])

                ref_output = residual + mlp_out

            print(f"  Full attn layer {layer_idx} output shape: {ref_output.shape}")
            print(f"  Output stats: mean={ref_output.float().mean():.4f}, std={ref_output.float().std():.4f}")


class TestQwen35ForwardVsTorch:
    def test_prefill_layer0_gdn(self, glm, qwen35_model, cfg):
        layer_idx = 0
        assert cfg.layer_types[layer_idx] == "linear_attention"
        device = torch.device("cuda", glm.device)
        hs = cfg.hidden_size
        S = 4
        lin_h = cfg.linear_num_key_heads
        lin_kd = cfg.linear_key_head_dim
        lin_vd = cfg.linear_value_head_dim
        conv_dim = lin_h * (lin_kd * 2 + lin_vd)
        z_dim = lin_h * lin_vd
        kernel_size = cfg.linear_conv_kernel_dim

        torch.manual_seed(42)
        hidden_np = np.random.randn(S, hs).astype(np.float32) * 0.1
        hidden_bf16 = _f32_to_bf16_bytes(hidden_np)

        hidden_gpu = _upload_bf16(glm, hidden_np)

        model = qwen35_model
        ws = model._ws

        pfx = f"layers.{layer_idx}.linear_attn"
        lpfx = f"layers.{layer_idx}"

        glm.rmsnorm(ws["normed"], hidden_gpu,
                     model.weights[f"{lpfx}.input_layernorm.weight"],
                     cfg.rms_norm_eps, hs, S)

        glm.linear(ws["gdn_qkv_linear"], ws["normed"],
                    model.weights[f"{pfx}.in_proj_qkv.weight"],
                    S, conv_dim, hs)
        glm.linear(ws["gdn_a"], ws["normed"],
                    model.weights[f"{pfx}.in_proj_a.weight"],
                    S, lin_h, hs)
        glm.linear(ws["gdn_b"], ws["normed"],
                    model.weights[f"{pfx}.in_proj_b.weight"],
                    S, lin_h, hs)
        glm.linear(ws["gdn_z"], ws["normed"],
                    model.weights[f"{pfx}.in_proj_z.weight"],
                    S, z_dim, hs)

        normed_np = _download_bf16(glm, ws["normed"], S * hs).reshape(S, hs)
        qkv_np = _download_bf16(glm, ws["gdn_qkv_linear"], S * conv_dim).reshape(S, conv_dim)

        print(f"  normed stats: mean={normed_np.mean():.4f}, std={normed_np.std():.4f}")
        print(f"  qkv stats: mean={qkv_np.mean():.4f}, std={qkv_np.std():.4f}")

        glm.free_buf(hidden_gpu)


class TestQwen35ModelForward:
    def test_prefill_single_token(self, glm, qwen35_model, gdn_state, cfg):
        model = qwen35_model
        device = torch.device("cuda", glm.device)
        cache = model.create_flat_kv_cache()

        input_ids = torch.tensor([[1]], dtype=torch.int64, device=device)
        logits = model.prefill(input_ids, cache, gdn_state)
        cache.reset()
        gdn_state.reset()

        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits stats: mean={logits.mean():.4f}, std={logits.std():.4f}")
        top5 = logits[0].topk(5)
        print(f"  Top-5 indices: {top5.indices.tolist()}, values: {top5.values.tolist()}")

        assert logits.shape == (1, cfg.vocab_size)
        assert not torch.isnan(logits).any(), "Logits contain NaN"
        assert not torch.isinf(logits).any(), "Logits contain Inf"

    def test_prefill_multi_token(self, glm, qwen35_model, gdn_state, cfg):
        model = qwen35_model
        device = torch.device("cuda", glm.device)
        cache = model.create_flat_kv_cache()

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64, device=device)
        logits = model.prefill(input_ids, cache, gdn_state)
        cache.reset()
        gdn_state.reset()

        print(f"  Logits shape: {logits.shape}")
        assert logits.shape == (1, cfg.vocab_size)
        assert not torch.isnan(logits).any(), "Logits contain NaN"

    def test_generate_tokens(self, glm, qwen35_model, gdn_state, cfg):
        model = qwen35_model
        device = torch.device("cuda", glm.device)
        cache = model.create_flat_kv_cache()

        input_ids = torch.tensor([[151643, 151644, 872]], dtype=torch.int64, device=device)
        tokens = list(model.generate_tokens(input_ids, cache, gdn_state, max_new_tokens=10))
        print(f"  Generated tokens: {tokens}")

        assert len(tokens) > 0, "No tokens generated"


class TestQwen35TorchReference:
    def test_torch_forward(self, model_dir, cfg):
        device = torch.device("cuda:0")
        torch_model = Qwen35TorchModel(model_dir, device)

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = torch_model.forward(input_ids)

        print(f"  PyTorch reference logits shape: {logits.shape}")
        print(f"  Logits stats: mean={logits.mean():.4f}, std={logits.std():.4f}")
        print(f"  Top-5: {logits[0, -1].topk(5)}")

        assert not torch.isnan(logits).any(), "PyTorch reference logits contain NaN"
        assert not torch.isinf(logits).any(), "PyTorch reference logits contain Inf"

    def test_torch_vs_cuda_prefill(self, glm, qwen35_model, gdn_state, model_dir, cfg):
        device = torch.device("cuda:0")
        torch_model = Qwen35TorchModel(model_dir, device)

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)

        with torch.no_grad():
            torch_logits = torch_model.forward(input_ids)

        cache = qwen35_model.create_flat_kv_cache()
        cuda_logits = qwen35_model.prefill(input_ids, cache, gdn_state)
        cache.reset()
        gdn_state.reset()

        max_diff = (cuda_logits - torch_logits.cpu()).abs().max().item()
        mean_diff = (cuda_logits - torch_logits.cpu()).abs().mean().item()
        print(f"  CUDA vs PyTorch: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")

        cuda_top5 = cuda_logits[0].topk(5).indices.tolist()
        torch_top5 = torch_logits[0, -1].topk(5).indices.tolist()
        print(f"  CUDA top-5: {cuda_top5}")
        print(f"  PyTorch top-5: {torch_top5}")

        assert not torch.isnan(cuda_logits).any(), "CUDA logits contain NaN"


def test_qwen35_generate_paris(glm, qwen35_model, gdn_state):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN35_REPO)
    model = qwen35_model
    cache = model.create_flat_kv_cache()

    try:
        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        tokens = list(model.generate_tokens(input_ids, cache, gdn_state, max_new_tokens=20))
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        assert len(tokens) > 0, "No tokens generated"
        assert "Paris" in text, f"Expected 'Paris' in generated text, got: {repr(text)}"
        print(f"  Generated {len(tokens)} tokens: {repr(text)}")
    finally:
        cache.free()
        gdn_state.reset()
