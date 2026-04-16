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
npm run test:python     # run all Python verification tests
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
- **Qwen3.5 BF16 vs FP32**: HuggingFace `AutoModelForCausalLM` uses FP32 computation. Our CUDA model uses BF16 throughout. Greedy decoding matches HuggingFace for ~9 tokens before diverging due to accumulated BF16 precision differences. Sampling (temperature/top-p/top-k/repetition penalty) mitigates this.
- Per-layer accuracy (synced input, CUDA vs PyTorch ref): GDN layers < 0.03 max_diff with 4-token input; full attention layers < 0.01 max_diff

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

### Qwen3.5-0.8B
- Model: `Qwen/Qwen3.5-0.8B`, cached at `/mnt/storage/.cache/huggingface/`
- 24 layers: 18 GDN (linear_attention) + 6 full_attention, every 4th layer is full attention
- hidden_size=1024, intermediate_size=3584, vocab_size=248320, rms_norm_eps=1e-6
- rope_theta=10000000 (in rope_parameters, not top-level)
- Full attention: 8 heads, 2 KV heads, head_dim=256, partial_rotary_factor=0.25 (64 RoPE dims), attn_output_gate=true
- GDN: 16 linear_key_heads, 16 linear_value_heads, linear_key_head_dim=128, linear_value_head_dim=128, conv_kernel_dim=4
- **Weight dtype note**: `A_log` is F32, `dt_bias` is BF16 in safetensors but must be uploaded as F32 (GDN kernels read both as float32)
- GemmaRMSNorm for layer norms (input_layernorm, post_attention_layernorm, q_norm, k_norm, final norm) — weight += 1 during loading
- Standard RMSNorm for GDN internal norm (linear_attn.norm.weight) — do NOT add +1; weight is F32 in safetensors but kernel reads as BF16
- GDN recurrent state is float32 (mamba_ssm_dtype: float32)
- Run: `npx tsx src/run_qwen3_chat.ts --qwen35`
- Sampling defaults: `--temperature 0.6 --top-p 0.95 --top-k 20 --repetition-penalty 1.1`; `--greedy` for argmax
- HuggingFace reference: `scratchpad/hf_qwen35_gen.py`

## FlashInfer Integration

FlashInfer's FA2 CUDA attention kernels are compiled into `libglm_ops.so` (Path B: C++ integration).

### Architecture
- `csrc/glm_flash.cu` — C wrappers calling FlashInfer's `SinglePrefillWithKVCacheDispatched` and `SingleDecodeWithKVCacheDispatched`
- Specialized for BF16, **head_dim=128 and head_dim=256** (256 added for Qwen3.5), PosEncodingMode::kNone, DefaultAttention variant
- Prefill uses MaskMode::kCausal; Decode uses MaskMode::kNone
- KV cache is HND layout: `[n_kv, max_S, hd]` — FlashInfer reads directly from cache via stride parameters
- Q tensor layout after transpose is `[B, n_heads, S, hd]` (HND) — strides: q_stride_n=hd, q_stride_h=S*hd
- Output from FlashInfer is NHD: `[S, n_heads, hd]` = `[S, hidden]` — directly compatible with o_proj
- 32MB workspace buffer for split-KV path (rarely triggered for small sequences)
- **Note**: Batch prefill/decode paths are hardcoded to head_dim=128 only (single prefill/decode supports both 128 and 256)

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
