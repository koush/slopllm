# GLM-5.1 NVFP4 Weight Reference

Source: `/mnt/storage/GLM-5.1-NVFP4-Fixed/` (80 safetensors shards)

## Overview

- **Total weights**: 233,345
- **Total size**: 461,793,880,328 bytes (430.08 GB)
- **Layers**: 79 total
  - Dense layers (0-2): [0, 1, 2]
  - MoE NVFP4 layers (3-77): layers 3-77 (75 layers)
  - MTP layer (78): Multi-Token Prediction draft layer (MoE, BF16)
- **Routed experts per MoE layer**: 256

### MTP Layer Structure

Layer 78 is the MTP (Multi-Token Prediction) layer used for speculative decoding.
It contains a full MoE transformer decoder plus MTP-specific projection weights:
- `eh_proj`: Linear projection combining embedded token + target model hidden state
- `enorm`: RMSNorm on embedded token before projection
- `hnorm`: RMSNorm on target model hidden state before projection
- `shared_head.norm`: RMSNorm before shared LM head
- Full decoder: self_attn (MLA + indexer), MoE MLP (256 routed + shared experts)

## Weight Name Patterns

Weights grouped by structural pattern. `N` = layer index, `E` = expert index.
`MTP` = MTP layer index (layer 78). Patterns marked **MTP** exist only in the MTP layer.
Some patterns have multiple shapes/dtypes across layers (e.g., NVFP4 vs BF16 for layer 78).

| Pattern | Shape | Dtype | Count | Total Size |
|---|---|---|---|---|
| `lm_head.weight` | [154880, 6144] | bfloat16 | 1 | 1.77 GB |
| `model.embed_tokens.weight` | [154880, 6144] | bfloat16 | 1 | 1.77 GB |
| `model.layers.MTP.eh_proj.weight` **MTP** | [6144, 12288] | bfloat16 | 1 | 144.00 MB |
| `model.layers.MTP.enorm.weight` **MTP** | [6144] | bfloat16 | 1 | 12.00 KB |
| `model.layers.MTP.hnorm.weight` **MTP** | [6144] | bfloat16 | 1 | 12.00 KB |
| `model.layers.MTP.input_layernorm.weight` **MTP** | [6144] | bfloat16 | 1 | 12.00 KB |
| `model.layers.MTP.mlp.experts.E.down_proj.weight` **MTP** | [6144, 2048] | bfloat16 | 256 | 6.00 GB |
| `model.layers.MTP.mlp.experts.E.gate_proj.weight` **MTP** | [2048, 6144] | bfloat16 | 256 | 6.00 GB |
| `model.layers.MTP.mlp.experts.E.up_proj.weight` **MTP** | [2048, 6144] | bfloat16 | 256 | 6.00 GB |
| `model.layers.MTP.mlp.gate.e_score_correction_bias` **MTP** | [256] | float32 | 1 | 1.00 KB |
| `model.layers.MTP.mlp.gate.weight` **MTP** | [256, 6144] | bfloat16 | 1 | 3.00 MB |
| `model.layers.MTP.mlp.shared_experts.down_proj.weight` **MTP** | [6144, 2048] | bfloat16 | 1 | 24.00 MB |
| `model.layers.MTP.mlp.shared_experts.gate_proj.weight` **MTP** | [2048, 6144] | bfloat16 | 1 | 24.00 MB |
| `model.layers.MTP.mlp.shared_experts.up_proj.weight` **MTP** | [2048, 6144] | bfloat16 | 1 | 24.00 MB |
| `model.layers.MTP.post_attention_layernorm.weight` **MTP** | [6144] | bfloat16 | 1 | 12.00 KB |
| `model.layers.MTP.self_attn.indexer.k_norm.bias` **MTP** | [128] | bfloat16 | 1 | 256 B |
| `model.layers.MTP.self_attn.indexer.k_norm.weight` **MTP** | [128] | bfloat16 | 1 | 256 B |
| `model.layers.MTP.self_attn.indexer.weights_proj.weight` **MTP** | [32, 6144] | bfloat16 | 1 | 384.00 KB |
| `model.layers.MTP.self_attn.indexer.wk.weight` **MTP** | [128, 6144] | bfloat16 | 1 | 1.50 MB |
| `model.layers.MTP.self_attn.indexer.wq_b.weight` **MTP** | [4096, 2048] | bfloat16 | 1 | 16.00 MB |
| `model.layers.MTP.self_attn.kv_a_layernorm.weight` **MTP** | [512] | bfloat16 | 1 | 1.00 KB |
| `model.layers.MTP.self_attn.kv_a_proj_with_mqa.weight` **MTP** | [576, 6144] | bfloat16 | 1 | 6.75 MB |
| `model.layers.MTP.self_attn.kv_b_proj.weight` **MTP** | [28672, 512] | bfloat16 | 1 | 28.00 MB |
| `model.layers.MTP.self_attn.o_proj.weight` **MTP** | [6144, 16384] | bfloat16 | 1 | 192.00 MB |
| `model.layers.MTP.self_attn.q_a_layernorm.weight` **MTP** | [2048] | bfloat16 | 1 | 4.00 KB |
| `model.layers.MTP.self_attn.q_a_proj.weight` **MTP** | [2048, 6144] | bfloat16 | 1 | 24.00 MB |
| `model.layers.MTP.self_attn.q_b_proj.weight` **MTP** | [16384, 2048] | bfloat16 | 1 | 64.00 MB |
| `model.layers.MTP.shared_head.norm.weight` **MTP** | [6144] | bfloat16 | 1 | 12.00 KB |
| `model.layers.N.input_layernorm.weight` | [6144] | bfloat16 | 78 | 936.00 KB |
| `model.layers.N.mlp.down_proj.weight` | [6144, 12288] | bfloat16 | 3 | 432.00 MB |
| `model.layers.N.mlp.experts.E.down_proj.input_scale` | [] | float32 | 19,200 | 75.00 KB |
| `model.layers.N.mlp.experts.E.down_proj.weight` | [6144, 1024] | uint8 | 19,200 | 112.50 GB |
| `model.layers.N.mlp.experts.E.down_proj.weight_scale` | [6144, 128] | float8_e4m3fn | 19,200 | 14.06 GB |
| `model.layers.N.mlp.experts.E.down_proj.weight_scale_2` | [] | float32 | 19,200 | 75.00 KB |
| `model.layers.N.mlp.experts.E.gate_proj.input_scale` | [] | float32 | 19,200 | 75.00 KB |
| `model.layers.N.mlp.experts.E.gate_proj.weight` | [2048, 3072] | uint8 | 19,200 | 112.50 GB |
| `model.layers.N.mlp.experts.E.gate_proj.weight_scale` | [2048, 384] | float8_e4m3fn | 19,200 | 14.06 GB |
| `model.layers.N.mlp.experts.E.gate_proj.weight_scale_2` | [] | float32 | 19,200 | 75.00 KB |
| `model.layers.N.mlp.experts.E.up_proj.input_scale` | [] | float32 | 19,200 | 75.00 KB |
| `model.layers.N.mlp.experts.E.up_proj.weight` | [2048, 3072] | uint8 | 19,200 | 112.50 GB |
| `model.layers.N.mlp.experts.E.up_proj.weight_scale` | [2048, 384] | float8_e4m3fn | 19,200 | 14.06 GB |
| `model.layers.N.mlp.experts.E.up_proj.weight_scale_2` | [] | float32 | 19,200 | 75.00 KB |
| `model.layers.N.mlp.gate.e_score_correction_bias` | [256] | float32 | 75 | 75.00 KB |
| `model.layers.N.mlp.gate.weight` | [256, 6144] | bfloat16 | 75 | 225.00 MB |
| `model.layers.N.mlp.gate_proj.weight` | [12288, 6144] | bfloat16 | 3 | 432.00 MB |
| `model.layers.N.mlp.shared_experts.down_proj.input_scale` | [] | float32 | 75 | 300 B |
| `model.layers.N.mlp.shared_experts.down_proj.weight` | [6144, 1024] | uint8 | 75 | 450.00 MB |
| `model.layers.N.mlp.shared_experts.down_proj.weight_scale` | [6144, 128] | float8_e4m3fn | 75 | 56.25 MB |
| `model.layers.N.mlp.shared_experts.down_proj.weight_scale_2` | [] | float32 | 75 | 300 B |
| `model.layers.N.mlp.shared_experts.gate_proj.input_scale` | [] | float32 | 75 | 300 B |
| `model.layers.N.mlp.shared_experts.gate_proj.weight` | [2048, 3072] | uint8 | 75 | 450.00 MB |
| `model.layers.N.mlp.shared_experts.gate_proj.weight_scale` | [2048, 384] | float8_e4m3fn | 75 | 56.25 MB |
| `model.layers.N.mlp.shared_experts.gate_proj.weight_scale_2` | [] | float32 | 75 | 300 B |
| `model.layers.N.mlp.shared_experts.up_proj.input_scale` | [] | float32 | 75 | 300 B |
| `model.layers.N.mlp.shared_experts.up_proj.weight` | [2048, 3072] | uint8 | 75 | 450.00 MB |
| `model.layers.N.mlp.shared_experts.up_proj.weight_scale` | [2048, 384] | float8_e4m3fn | 75 | 56.25 MB |
| `model.layers.N.mlp.shared_experts.up_proj.weight_scale_2` | [] | float32 | 75 | 300 B |
| `model.layers.N.mlp.up_proj.weight` | [12288, 6144] | bfloat16 | 3 | 432.00 MB |
| `model.layers.N.post_attention_layernorm.weight` | [6144] | bfloat16 | 78 | 936.00 KB |
| `model.layers.N.self_attn.indexer.k_norm.bias` | [128] | bfloat16 | 78 | 19.50 KB |
| `model.layers.N.self_attn.indexer.k_norm.weight` | [128] | bfloat16 | 78 | 19.50 KB |
| `model.layers.N.self_attn.indexer.weights_proj.weight` | [32, 6144] | bfloat16 | 78 | 29.25 MB |
| `model.layers.N.self_attn.indexer.wk.weight` | [128, 6144] | bfloat16 | 78 | 117.00 MB |
| `model.layers.N.self_attn.indexer.wq_b.weight` | [4096, 2048] | bfloat16 | 78 | 1.22 GB |
| `model.layers.N.self_attn.kv_a_layernorm.weight` | [512] | bfloat16 | 78 | 78.00 KB |
| `model.layers.N.self_attn.kv_a_proj_with_mqa.weight` | [576, 6144] | bfloat16 | 78 | 526.50 MB |
| `model.layers.N.self_attn.kv_b_proj.weight` | [28672, 512] | bfloat16 | 78 | 2.13 GB |
| `model.layers.N.self_attn.o_proj.weight` | [6144, 16384] | bfloat16 | 78 | 14.62 GB |
| `model.layers.N.self_attn.q_a_layernorm.weight` | [2048] | bfloat16 | 78 | 312.00 KB |
| `model.layers.N.self_attn.q_a_proj.weight` | [2048, 6144] | bfloat16 | 78 | 1.83 GB |
| `model.layers.N.self_attn.q_b_proj.weight` | [16384, 2048] | bfloat16 | 78 | 4.88 GB |
| `model.norm.weight` | [6144] | bfloat16 | 1 | 12.00 KB |

## Size Breakdown by Component

| Component | Size | % of Total |
|---|---|---|
| MoE Routed Experts | 379.69 GB | 88.3% |
| Attention (self_attn) | 25.34 GB | 5.9% |
| MTP MoE Routed Experts | 18.00 GB | 4.2% |
| LM Head | 1.77 GB | 0.4% |
| Embedding | 1.77 GB | 0.4% |
| MoE Shared Experts | 1.48 GB | 0.3% |
| Dense MLP | 1.27 GB | 0.3% |
| MTP Attention (self_attn) | 332.63 MB | 0.1% |
| MoE Router (gate) | 225.07 MB | 0.1% |
| MTP Other (eh_proj/enorm/hnorm) | 144.02 MB | 0.0% |
| MTP MoE Shared Experts | 72.00 MB | 0.0% |
| MTP MoE Router (gate) | 3.00 MB | 0.0% |
| LayerNorms | 1.83 MB | 0.0% |
| MTP LayerNorms | 24.00 KB | 0.0% |
| MTP Shared Head Norm | 12.00 KB | 0.0% |
| Final Norm | 12.00 KB | 0.0% |
