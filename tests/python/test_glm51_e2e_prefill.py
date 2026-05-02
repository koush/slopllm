"""End-to-end comparison: Node.js CUDA MLA prefill vs HuggingFace for GLM-5.1 small."""
import torch
import json
import ctypes
import math
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "test_models" / "glm51_small" / "glm51_small_bf16"
HF_REPO = "zai-org/GLM-5.1"
PAGE_SIZE = 16

import sys, os
sys_path = os.path.join(os.path.dirname(__file__), "test_models", "glm51_small")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from helpers import GlmOps
from glm51_small_reference import Glm51SmallModel, make_rotary_embed, rms_norm

def test_single_token_prefill():
    """Compare HF logits vs CUDA MLA prefill for a single token."""
    device = torch.device("cuda:0")
    glm = GlmOps(device_id=0)
    
    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map="cpu"
    )
    hf_model.eval()
    
    # Load reference model (same weights, pure PyTorch)
    ref_model = Glm51SmallModel()
    
    with open(MODEL_DIR / "config.json") as f:
        cfg = json.load(f)
    
    num_heads = cfg["num_attention_heads"]
    kv_lora_rank = cfg["kv_lora_rank"]
    qk_rope_dim = cfg["qk_rope_head_dim"]
    v_head_dim = cfg["v_head_dim"]
    hidden_size = cfg["hidden_size"]
    rms_norm_eps = cfg["rms_norm_eps"]
    num_layers = cfg["num_hidden_layers"]
    vocab_size = cfg["vocab_size"]
    scaling = 1.0 / math.sqrt(kv_lora_rank + qk_rope_dim)
    
    # Test with short sequences
    for S in [1, 2, 4]:
        torch.manual_seed(42)
        input_ids = torch.randint(0, min(vocab_size, 1000), (1, S))
        
        # HF forward
        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits.float()
        hf_argmax = hf_logits[0, -1].argmax().item()
        hf_top5 = hf_logits[0, -1].topk(5).indices.tolist()
        
        # CUDA forward using MLA paged prefill
        # Build embeddings
        hidden = ref_model.embed(input_ids)  # [1, S, hidden]
        hidden_gpu = torch.empty(S, hidden_size, dtype=torch.bfloat16, device=device)
        glm.h2d(hidden_gpu, hidden.reshape(S, hidden_size).contiguous())
        
        # Process layers
        qk_rope_half = qk_rope_dim // 2
        cos, sin = make_rotary_embed(glm, device, qk_rope_half, 1, S)
        
        for layer_idx in range(num_layers):
            layer = ref_model.layers[layer_idx]
            
            # RMSNorm
            normed_gpu = torch.empty(S, hidden_size, dtype=torch.bfloat16, device=device)
            ln_w_gpu = torch.empty(hidden_size, dtype=torch.bfloat16, device=device)
            glm.h2d(ln_w_gpu, layer.input_layernorm_w)
            glm.rmsnorm(normed_gpu, hidden_gpu, ln_w_gpu, rms_norm_eps, hidden_size, S)
            glm.synchronize()
            
            # MLA attention using paged prefill
            # q_a_proj, q_b_proj -> q_nope, q_pe
            q_a = torch.empty(S, cfg["q_lora_rank"], dtype=torch.bfloat16, device=device)
            glm.linear(q_a, normed_gpu, layer.q_a_proj_w.to(device), S, cfg["q_lora_rank"], hidden_size)
            
            q_a_normed = torch.empty(S, cfg["q_lora_rank"], dtype=torch.bfloat16, device=device)
            glm.rmsnorm(q_a_normed, q_a, layer.q_a_layernorm_w.to(device), rms_norm_eps, cfg["q_lora_rank"], S)
            
            q_b = torch.empty(S, num_heads * (kv_lora_rank + qk_rope_dim), dtype=torch.bfloat16, device=device)
            glm.linear(q_b, q_a_normed, layer.q_b_proj_w.to(device), S, num_heads * (kv_lora_rank + qk_rope_dim), cfg["q_lora_rank"])
            
            # Split q_b into q_nope and q_pe
            q_nope_flat = q_b[:, :num_heads * kv_lora_rank].contiguous()
            q_pe_flat = q_b[:, num_heads * kv_lora_rank:].contiguous()
            
            # ropeTranspose to BHD layout [B*S, H, D]
            q_nope = torch.empty(S, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)
            glm.ropeTranspose(q_nope, q_nope_flat, cos, sin, 0, kv_lora_rank, num_heads, S, 1, kv_lora_rank)
            
            q_pe = torch.empty(S, num_heads, qk_rope_dim, dtype=torch.bfloat16, device=device)
            glm.ropeTranspose(q_pe, q_pe_flat, cos, sin, qk_rope_dim, qk_rope_dim, num_heads, S, 1, qk_rope_dim)
            
            # ckv, k_pe
            ckv = torch.empty(S, kv_lora_rank, dtype=torch.bfloat16, device=device)
            glm.linear(ckv, normed_gpu, layer.ckv_proj_w.to(device), S, kv_lora_rank, hidden_size)
            
            ckv_normed = torch.empty(S, kv_lora_rank, dtype=torch.bfloat16, device=device)
            glm.rmsnorm(ckv_normed, ckv, layer.kv_a_layernorm_w.to(device), rms_norm_eps, kv_lora_rank, S)
            
            k_pe = torch.empty(S, qk_rope_dim, dtype=torch.bfloat16, device=device)
            glm.linear(k_pe, normed_gpu, layer.k_pe_proj_w.to(device), S, qk_rope_dim, hidden_size)
            
            # Apply RoPE to k_pe (nHeads=1)
            k_pe_4d = k_pe.reshape(1, 1, S, qk_rope_dim)
            k_pe_rope_4d = torch.empty_like(k_pe_4d)
            glm.apply_rotary_pos_emb(k_pe_rope_4d, k_pe_4d, cos, sin, qk_rope_dim, 1, S, 1, 1)
            k_pe_rope = k_pe_rope_4d.reshape(S, qk_rope_dim)
            
            # MLA paged prefill
            num_pages = math.ceil(S / PAGE_SIZE)
            ckv_paged = torch.zeros(num_pages, PAGE_SIZE, kv_lora_rank, dtype=torch.bfloat16, device=device)
            kpe_paged = torch.zeros(num_pages, PAGE_SIZE, qk_rope_dim, dtype=torch.bfloat16, device=device)
            for p in range(num_pages):
                start = p * PAGE_SIZE
                end = min(start + PAGE_SIZE, S)
                ckv_paged[p, :end-start] = ckv_normed[start:end]
                kpe_paged[p, :end-start] = k_pe_rope[start:end]
            
            kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
            qo_indptr_h = (ctypes.c_int32 * 2)(0, S)
            kv_indptr_h = (ctypes.c_int32 * 2)(0, num_pages)
            kv_len_h = (ctypes.c_int32 * 1)(S)
            plan_info = (ctypes.c_int64 * 18)()
            
            float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)
            
            glm.mla_prefill_plan(
                float_ws, 32 * 1024 * 1024,
                int_ws, pinned_int_ws, 8 * 1024 * 1024,
                ctypes.addressof(plan_info),
                ctypes.addressof(qo_indptr_h),
                ctypes.addressof(kv_indptr_h),
                ctypes.addressof(kv_len_h),
                1, num_heads, kv_lora_rank, True
            )
            
            attn_out = torch.empty(S, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)
            
            glm.mla_prefill_run(
                q_nope.data_ptr(), q_pe.data_ptr(),
                ckv_paged.data_ptr(), kpe_paged.data_ptr(),
                kv_indices.data_ptr(),
                attn_out.data_ptr(),
                float_ws, int_ws, ctypes.addressof(plan_info),
                num_heads, PAGE_SIZE, 1, scaling,
                num_heads * kv_lora_rank, kv_lora_rank,  # BHD strides
                num_heads * qk_rope_dim, qk_rope_dim,
                PAGE_SIZE * kv_lora_rank, kv_lora_rank,
                PAGE_SIZE * qk_rope_dim, qk_rope_dim,
                num_heads * kv_lora_rank, kv_lora_rank,  # BHD output strides
                kv_lora_rank, qk_rope_dim
            )
            
            glm.synchronize()
            glm.free_buf(float_ws)
            glm.free_buf(int_ws)
            glm.free_pinned(pinned_int_ws)
            
            # V expand
            v_proj = layer.v_proj_w.to(device)  # [num_heads * v_head_dim, kv_lora_rank]
            v_expanded = torch.empty(S, num_heads * v_head_dim, dtype=torch.bfloat16, device=device)
            glm.mlaVExpand(v_expanded, attn_out, v_proj, kv_lora_rank, v_head_dim, num_heads, S, 1)
            glm.synchronize()
            
            # o_proj
            o_proj = torch.empty(S, hidden_size, dtype=torch.bfloat16, device=device)
            glm.linear(o_proj, v_expanded, layer.o_proj_w.to(device), S, hidden_size, num_heads * v_head_dim)
            
            # Residual
            residual = torch.empty(S, hidden_size, dtype=torch.bfloat16, device=device)
            glm.add(residual, hidden_gpu, o_proj, S * hidden_size)
            
            # Post-attention norm + MLP
            post_normed = torch.empty(S, hidden_size, dtype=torch.bfloat16, device=device)
            post_ln_w = torch.empty(hidden_size, dtype=torch.bfloat16, device=device)
            glm.h2d(post_ln_w, layer.post_attention_layernorm_w)
            glm.rmsnorm(post_normed, residual, post_ln_w, rms_norm_eps, hidden_size, S)
            glm.synchronize()
            
            if layer.is_sparse:
                # MOE - use reference for simplicity
                post_cpu = post_normed.cpu()
                mlp_out = layer.mlp(post_cpu.unsqueeze(0)).squeeze(0).bfloat16()
                mlp_gpu = torch.empty(S, hidden_size, dtype=torch.bfloat16, device=device)
                glm.h2d(mlp_gpu, mlp_out)
            else:
                intermediate = cfg["intermediate_size"]
                gate_out = torch.empty(S, intermediate, dtype=torch.bfloat16, device=device)
                up_out = torch.empty(S, intermediate, dtype=torch.bfloat16, device=device)
                inter = torch.empty(S, intermediate, dtype=torch.bfloat16, device=device)
                glm.linear(gate_out, post_normed, layer.mlp.gate_proj_w.to(device), S, intermediate, hidden_size)
                glm.linear(up_out, post_normed, layer.mlp.up_proj_w.to(device), S, intermediate, hidden_size)
                glm.silu_and_mul(inter, gate_out, up_out, intermediate, S)
                mlp_gpu = torch.empty(S, hidden_size, dtype=torch.bfloat16, device=device)
                glm.linear(mlp_gpu, inter, layer.mlp.down_proj_w.to(device), S, hidden_size, intermediate)
            
            hidden_gpu = torch.empty(S, hidden_size, dtype=torch.bfloat16, device=device)
            glm.add(hidden_gpu, residual, mlp_gpu, S * hidden_size)
            glm.synchronize()
        
        # Final norm + lm_head
        norm_w_gpu = torch.empty(hidden_size, dtype=torch.bfloat16, device=device)
        glm.h2d(norm_w_gpu, ref_model.norm_w)
        final_normed = torch.empty(S, hidden_size, dtype=torch.bfloat16, device=device)
        glm.rmsnorm(final_normed, hidden_gpu, norm_w_gpu, rms_norm_eps, hidden_size, S)
        
        lm_head_gpu = torch.empty(hidden_size, vocab_size, dtype=torch.bfloat16, device=device)
        glm.h2d(lm_head_gpu, ref_model.embed_tokens_w)
        logits_gpu = torch.empty(S, vocab_size, dtype=torch.bfloat16, device=device)
        glm.linear(logits_gpu, final_normed, lm_head_gpu, S, vocab_size, hidden_size)
        glm.synchronize()
        
        cuda_logits = logits_gpu.cpu().view(1, S, vocab_size).float()
        cuda_argmax = cuda_logits[0, -1].argmax().item()
        cuda_top5 = cuda_logits[0, -1].topk(5).indices.tolist()
        
        max_diff = (cuda_logits - hf_logits).abs().max().item()
        mean_diff = (cuda_logits - hf_logits).abs().mean().item()
        mean_abs = hf_logits.abs().mean().item()
        
        match = "MATCH" if cuda_argmax == hf_argmax else "MISMATCH"
        print(f"S={S}: CUDA argmax={cuda_argmax} HF argmax={hf_argmax} [{match}]  max_diff={max_diff:.4f} mean_diff={mean_diff:.4f}")
        print(f"  CUDA top5: {cuda_top5}")
        print(f"  HF top5: {hf_top5}")
    
    glm.free()

def _alloc_workspace(glm):
    float_ws = glm.alloc_buf(128 * 1024 * 1024)
    int_ws = glm.alloc_buf(8 * 1024 * 1024)
    pinned_int_ws = glm.alloc_pinned(8 * 1024 * 1024)
    return float_ws, int_ws, pinned_int_ws

if __name__ == "__main__":
    test_single_token_prefill()
