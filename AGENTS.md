# Project Conventions

## Reference Implementation

`vendor/streaming_nvfp4_quantize.py` is the **canonical reference** for the GLM-5.1 model architecture. All CUDA kernels and Python test references must match this file's implementation exactly. Key details:

- **Indexer uses LayerNorm (with bias)** for `k_norm`, while all other norms use RMSNorm
- **Indexer scoring uses ReLU** (not softmax) on scores before weighted sum
- **Main attention softmax** operates in float32 for numerical stability, then casts back to BF16
- **MoE dispatch**: per-expert Python loop (`for i in range(num_experts)`) — same pattern as our CUDA impl
- **Rope/nope ordering differs by module**:
  - Main attention: `[nope | pe]` — first qk_nope_dim=192 dims are non-positional, last qk_rope_dim=64 are positional
  - Indexer: `[pe | nope]` — first qk_rope_dim=64 dims are positional, remaining are non-positional
- **The quantization script creates FP4 from BF16**, but the safetensors checkpoint stores BF16 weights directly
- **Indexer shares `q_resid` with main attention**

## Build & Test

```bash
npm run build:all       # build everything
npm run test:python     # run all Python verification tests (158 tests)
```

Run Python tests directly:
```bash
cd tests/python && LD_LIBRARY_PATH=../../build:$LD_LIBRARY_PATH pytest -v .
```

## Precision Model

- All kernels operate on **BF16 with FP32 accumulation**, matching cuBLAS tensor core behavior
- Test references should match CUDA precision model: BF16 inputs for linear ops, float32 for fused ops (e.g. silu*mul)
- When increasing test tolerances, investigate implementation bugs first — don't mask real errors with loose tolerances
- MoE routing weight differences (~1 BF16 ULP) cause cascading errors proportional to output magnitude — this is expected BF16 behavior

## Model Details

### GLM-5.1
- Model: `zai-org/GLM-5.1`, cached in Python HF cache
- 78 layers (3 dense: 0-2, 75 MoE: 3-77), 256 routed experts (top-8)
- hidden_size=6144, moe_intermediate_size=2048, dense_intermediate=12288
- GPU: NVIDIA RTX PRO 6000 Blackwell (sm_120), PyTorch nightly required
- Build with `-gencode arch=compute_120,code=sm_120`

### Qwen3-0.6B
- Model: `Qwen/Qwen3-0.6B`, cached at `/mnt/storage/.cache/huggingface/`
- Standard GQA transformer (no MLA, no MoE, no DSA/Indexer)
- hidden_size=1024, num_attention_heads=16, num_key_value_heads=8, head_dim=128, GQA groups=2
- intermediate_size=3072, num_hidden_layers=28, vocab_size=151936
- rms_norm_eps=1e-6, rope_theta=1000000, attention_bias=false, tie_word_embeddings=true
- QK norm: RMSNorm on Q and K per head (before RoPE)
- SwiGLU MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`
- Reference: `vendor/modeling_qwen3.py`, `vendor/configuration_qwen3.py`
- Run Qwen3 tests with: `HF_HOME=/mnt/storage/.cache/huggingface pytest test_qwen3.py -v`

## FlashInfer Integration

FlashInfer's FA2 CUDA attention kernels are compiled into `libglm_ops.so` (Path B: C++ integration).

### Architecture
- `csrc/glm_flash.cu` — C wrappers calling FlashInfer's `SinglePrefillWithKVCacheDispatched` and `SingleDecodeWithKVCacheDispatched`
- Specialized for BF16, head_dim=128, PosEncodingMode::kNone, DefaultAttention variant (no custom mask/sliding window/logits_soft_cap/alibi)
- Prefill uses MaskMode::kCausal; Decode uses MaskMode::kNone
- KV cache is HND layout: `[n_kv, max_S, hd]` — FlashInfer reads directly from cache via stride parameters
- Q tensor layout after transpose is `[B, n_heads, S, hd]` (HND) — strides: q_stride_n=hd, q_stride_h=S*hd
- Output from FlashInfer is NHD: `[S, n_heads, hd]` = `[S, hidden]` — directly compatible with o_proj
- 32MB workspace buffer for split-KV path (rarely triggered for small sequences)

### Python bindings
- `helpers.py`: `flash_prefill()` and `flash_decode()` methods on `GlmOps`
- `qwen3_model.py`: `_attention_prefill_flash()` and `_attention_decode_flash()` methods
- Hot path (`prefill()` and `_decode_token()`) uses flash attention; `forward()` retains BMM-based attention for testing

### Memory savings (at S=4096)
Flash attention eliminates: attn_scores (512MB), mask_expanded (512MB), causal_mask (32MB), k/v_expanded (32MB), attn_out + attn_out_t (32MB) = ~1.1GB
Added: flash_out (16MB) + flash_tmp (32MB) = 48MB

## Python Code

- All Python code lives in `tests/python/` — verification only, not production
- Use `json.load()` for config parsing (no transformers dependency)
- Use `safetensors.safe_open()` for weight loading
