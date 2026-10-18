# GLM-5.3-Flash support investigation

Investigated 2026-10-11. Scope: implementation assessment, without inference-engine changes or GPU qualification.

## Revised initial scope

### Implementation status

The first implementation adds `Glm53FlashModel`, text-only BF16/NVFP4 weight loading, TP weight layouts, and explicit forward/cache stubs. Select the checkpoint with `--model <owner/name>` or `--model-dir <directory>`; `model_type: "glm5_next"` selects this implementation. Vision and MTP tensors are skipped. The persistent loader uses the shared model-selection path.

`src/kda.ts` provides per-layer convolution/recurrent state and the complete KDA attention sublayer. Native primitives in `csrc/glm_flash_hybrid.cu` implement the 128-dimensional channel-wise KDA recurrence, sigmoid-gated RMSNorm, mHC pre/Sinkhorn collapse, FP32 post mixing, and stream expansion/final mean. These specialized operations live in `DeviceOps`, with shared shape/dtype/backend validation and implementations in `GlmOps`, `ParallelOps`, and `MetaOps`. They cover single-GPU, head-sharded TP for KDA, replicated TP residual streams for mHC, and metadata-only allocation. Ordinary linear projections remain tensor operations.

KDA currently uses a correctness-first recurrent scan for decode and packed/resumed prefill. Optimized chunked KDA prefill remains future performance work. Main-model forward, pooled indexer/MLA integration, and hybrid chat-cache orchestration still throw or remain unimplemented. The full production checkpoint has not been loaded end-to-end; loader tests use a synthetic sharded checkpoint with the real naming/dtype/layout contracts. GPU tests compare primitives with independent PyTorch math and exercise TP, chunk continuation, and graph replay.

The initial target is now **`local-inference-lab/GLM-5.3-Flash-NVFP4`**, pinned for inspection at `175ae8ce3b5af842b0d0140dbeb43e9cfc557c49`. **Phased prefill, MTP, and image/video support are deferred**; initial support is text-only, with ordinary chunked prefill. The original CSF-QAD checkpoint analysis below is retained as a later-target reference, not an initial implementation requirement.

The replacement config/index and representative safetensors headers confirm:

- Main decoder routed experts: NVFP4 with ordinary E4M3 `weight_scale` tensors; **no CSF decoder needed**.
- Attention and shared-expert weights: BF16. Verified KDA Q projection `[8192,4096]`, MLA KV-B `[32768,512]`, and shared expert gate `[2048,4096]`.
- MTP layer 45 routed experts: **MXFP8**, not W4A16 NVFP4. Verified E4M3 gate weight `[2048,4096]` and U8 UE8M0 scales `[2048,128]`. MXFP8 can be deferred with MTP for base-model bring-up; enabling MTP still needs an explicit MXFP8 execution/conversion path.
- Index tensor storage: 198,042,331,512 bytes (about 184.44 GiB), across 44 shards.
- Main experts still declare static NVFP4 activation quantization; the activation-scale fidelity question remains.

### Correction: FlashInfer already provides no-RoPE MLA

The vendored FlashInfer SM120 implementation **already supports `ModelType::GLM53_NOPE`**, including prefill and decode. The earlier investigation missed this existing implementation. A new MLA kernel family is not an initial requirement.

- `vendor/flashinfer/include/flashinfer/attention/sparse_mla_sm120/model/model_type.h:37`: 512 latent dimensions, zero rotary dimensions, arbitrary FP32 scales, **656-byte rows with 128 bytes of reserved padding**.
- `vendor/flashinfer/include/flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh:98`: no-RoPE traits.
- `vendor/flashinfer/csrc/sparse_mla_sm120_decode_dsv3_2.cu:208`: dedicated 32/64-head dispatch and runtime-head fallback; top-k width is a runtime argument.
- `vendor/flashinfer/csrc/sparse_mla_sm120_prefill.cu:606`: no-RoPE prefill dispatch.

Our wrapper must pass the correct model type, 512-wide queries, compatible cache packing, and `256^-0.5` softmax scale. Keep FlashInfer's 656-byte padded ABI when using these kernels; b12x's 528-byte compact record is a different backend contract. Validate/pad the checkpoint's 2051 selection capacity to the selected kernel's alignment requirements. The Python planner has 2176-wide calibration entries, but that is not evidence that the checkpoint itself selects a 128-token tail. Also, the current decode launcher's prequantized-Q (`Q_scales`) route is limited to GLM_NSA with split rotary Q; use the supported BF16-Q route initially.

### Routing source

The precision/selection reference is `Glm5NextTextTopkRouter.forward()` in `vendor/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py:157-182`, particularly line 159's explicit FP32 `F.linear`. Selection uses sigmoid scores plus correction bias, while mixture weights use the **unbiased** sigmoid scores, normalized and multiplied by 2.5. The config specifies one routing group, 288 experts, and top-8.

### Revised milestones

1. New config/weight mapping using ordinary NVFP4 scales and BF16 attention/shared experts.
2. KDA, mHC, clamped SwiGLU, 288-expert capacities, and FP32 routing.
3. Wire existing FlashInfer GLM53_NOPE MLA; implement pooled indexer and hybrid cache; establish text prefill/decode with TP.
4. Efficient chunked KDA prefill, graph capture, and request/prefix state ownership.
5. Validate text-only end-to-end generation and persistent model-loader integration. Hybrid CP remains a separate workstream.

Deferred: MTP (including its MXFP8 experts and speculative recurrent-state rollback), phased prefill, image/video support, and the original CSF-QAD checkpoint. Initial loading should skip layer 45's MTP tensors and `model.visual.*` vision tensors. Vision encoding, image/video preprocessing, multimodal embedding insertion, and multimodal API support are outside the initial scope. Consequently, native MXFP8 support is not required for the initial text decoder. Ordinary recurrent-state ownership and prefix continuation remain necessary even without speculation; prefix-cache reuse can be added after correctness-first generation with fresh state.

## Original CSF-QAD target assessment (superseded where noted above)

`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` needs a new hybrid model implementation with shared GLM components. It is substantially more work than adding DeltaNet to `Glm51Model`.

The major additions are:

1. Kimi Delta Attention (KDA), including efficient prefill and persistent recurrent state.
2. Four-stream manifold-constrained hyper-connections (mHC).
3. No-RoPE sparse MLA with a learned, four-token pooled indexer.
4. MXFP8 weights/activations and CSF decoding for NVFP4 expert scales.
5. 288-expert MoE support, clamped SwiGLU, and FP32 routing semantics.
6. Hybrid state lifecycle for serving, prefix reuse, and speculative verification.

The existing GLM sparse-MLA/MoE code and Qwen3.5 convolution/state code offer meaningful reuse, but neither model implements this architecture end-to-end. Multimodal inputs add a further vision/processor/server workstream.

## Sources and reproducibility

- [Target checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD), inspected revision `dec48abd33efa73c3bb7c95b74eee10cad34f9be`.
- [Original model configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json).
- Target config, model card, safetensors index, and representative safetensors headers were inspected. Header inspection used HTTP range requests, without downloading full weight shards.
- Transformers: `vendor/transformers`, revision `536ecc007387a50e77603bb5d92100e9b07514cc`.
- joninco/vLLM fork: `vendor/vllm`, revision `ce5f28ba31f22c07d5ea1205c8171034635c82ff`.
- b12x: `vendor/b12x`, revision `686450b72a5665ebeeb571ed5c1729b190a829da`.

Fetched the three existing vendor repositories. Transformers and b12x were checked out at the fetched main/master revisions; vLLM was retained at its existing GLM-aware fork revision (the fork's `origin/main` is older and lacks these model files). No new Python runtime dependency is implied by reading these references.

Useful reference files:

- `vendor/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py`: readable KDA, mHC, pooling, attention, MoE, and vision reference math.
- `vendor/vllm/vllm/models/glm5next/nvidia/{model,kda,attention,pooled_indexer,mtp,multimodal}.py`: serving implementation and checkpoint naming.
- `vendor/vllm/vllm/models/glm5next/model_state.py`: hybrid attention metadata/state integration.
- `vendor/b12x/b12x/sequence/{gdn_decode,kda_prefill}/`: optimized KDA references.
- `vendor/b12x/tests/sequence/{test_gdn_decode_kda,test_kda_prefill}.py`.
- `vendor/b12x/tests/norm/{test_mhc,test_residual_mhc}.py`.
- `vendor/b12x/tests/attention/test_glm_next_mla.py`: explicit GLM_NEXT no-RoPE sparse-MLA/cache contracts.
- `vendor/b12x/tests/gemm/{test_mxfp8_linear,test_gemm_mxfp8_linear}.py` and `tests/quantization/test_mxfp8.py`.

b12x uses Python/CuTe DSL and some Triton paths. Its presence in vendor does not make these kernels callable from our C++/CUDA addon. Porting math/kernels or providing an ahead-of-time build/launch integration is a separate engineering choice.

## Architecture comparison

| Property | Current production GLM-5.3 full | Flash target |
|---|---|---|
| Model class | GLM MoE DSA | `Glm5NextForConditionalGeneration` |
| Config layout | Flat text config | `text_config` + `vision_config` |
| Decoder layers | 78 | 45 |
| Hidden width | 6144 | 4096 |
| Attention schedule | Sparse MLA | 34 KDA + 11 sparse MLA |
| Sparse-MLA layers, zero-based | Throughout decoder | 3, 7, 11, …, 43 |
| Residual path | Ordinary residual | Four mHC streams around both attention and FFN |
| Routed/shared experts | 256 / 1 | 288 / 1 |
| Experts per token | 8 | 8 |
| First dense FFN layers | 3 | 3 |
| Expert intermediate width | 2048 | 2048 |
| Router scaling | 2.5 | 2.5 |
| Query LoRA rank | 2048 | 1536 |
| KV LoRA rank | 512 | 512 |
| MLA query non-rotary/rotary width | 192 / 64 | 256 / 0 |
| MLA value width | 256 | 256 |
| Index selection | Token top-k, selected cross-layer sharing | Learned pools of 4; all 11 MLA layers have their own indexer |
| Vocabulary | 154880 | 154880 |
| Configured maximum context | 1,048,576 | 1,048,576 |
| MTP | One layer | One layer, checkpoint layer 45, sparse MLA |

### KDA: related to Qwen GDN, but different math

Each KDA layer has 64 heads, key/value head dimension 128, and separate Q/K/V projections to width 8192. Each projection has a depthwise causal convolution of width 4 followed by SiLU. Q and K are L2-normalized.

The forget gate is low-rank: `f_a_proj: 4096 -> 128`, `f_b_proj: 128 -> 8192`. The target uses:

```
g[h, k] = -5 * sigmoid(exp(A_log[h]) * (f[h, k] + dt_bias[h, k]))
beta[h] = sigmoid(b_proj(x)[h])
S_decay = exp(g)[:, None] * S_previous
delta = beta * (v - k^T S_decay)
S = S_decay + k[:, None] * delta[None, :]
output = (q / sqrt(128))^T S
```

Decay is **per key channel**, not a single scalar per head. Output uses per-head RMSNorm followed by a **sigmoid** gate from `g_b_proj(g_a_proj(x))`, then the output projection.

Our `csrc/glm_gdn.cu` uses scalar per-head softplus decay and a different normalization epsilon (`1e-8`, versus reference KDA `1e-6`). The existing Qwen path must not simply be selected for these layers. Its convolution helpers, tensor-parallel head partitioning, FP32 recurrent storage, and packed-sequence conventions can be reused or generalized.

An initial recurrent CUDA implementation is relatively straightforward. Fast long-context prefill is a larger kernel task: the current GDN prefill loops through tokens and is not a substitute for a tuned chunked KDA algorithm.

### mHC changes the entire decoder residual path

Each token carries `[4, 4096]` hidden-state streams. Before each attention and FFN block:

- Flatten and RMS-normalize the streams, with FP32 coefficient computation.
- Project 16384 values to 24 coefficients: 4 pre, 4 post, and a 4x4 residual-mixing matrix.
- Apply learned scales/biases, sigmoid/softmax, and 20 Sinkhorn iterations.
- Collapse the four streams to the ordinary width-4096 block input.
- Combine the block output with the four residual streams using the post/mixing coefficients.

The final hyper-head collapses streams by an **unweighted mean**. This differs from generic DeepSeek-V4 hyper-head behavior. The checkpoint has `hc_attn_*` and `hc_ffn_*` tensors; its base/scale parameters are F32, while the large `*_fn` matrices are BF16. Blindly converting every F32 weight to BF16, as the current GLM loader usually does, loses intended precision.

This needs new tensor/native operations, a new forward-layer structure, and a review of phased-prefill activation storage. Four residual streams increase residual-buffer memory/bandwidth; attention and MLP inputs remain width 4096.

### Sparse MLA and pooled indexer

MLA still uses the familiar compressed 512-dimensional latent cache and factorized query/value projections, but **there is no rotary component**. The attention softmax scale remains `256^-0.5`, based on the unabsorbed query dimension, not `512^-0.5`.

The indexer forms learned, channel-wise weighted averages over each complete group of four keys. Weights come from an extra hidden-state projection plus a learned `[4,128]` positional parameter. It scores complete pools, selects at most `2048 / 4 = 512` pools, expands them to raw token indices, and appends the visible incomplete tail of 0–3 tokens. Output capacity is **2051**, including invalid padding. A pool is eligible only when its last token is causally visible.

Consequences:

- Add pooled-key storage and partial-pool state across prefill chunk boundaries and decode steps.
- Keep main MLA latents per original token: index pooling does **not** reduce the main KV cache by another factor of four.
- Preserve causal visibility, pool boundaries, and tail behavior for packed batches, prefix continuation, and speculation.
- Adapt selection widths, masks, cache metadata, and capture buckets.
- Wire the existing FlashInfer `GLM53_NOPE` sparse-MLA contract. `csrc/glm_sparse_mla.cu` currently fixes `ModelType::GLM_NSA`; FlashInfer's no-RoPE variant retains the 656-byte stride with reserved padding. The b12x GLM_NEXT reference instead uses a compact 528-byte record. These backend layouts must not be conflated.
- Review `Glm51Model.loadMlaWeight()`: it currently requires BF16 and creates rotary projections. The target's corresponding weights are MXFP8.
- Quantized `kv_b_proj` splitting/transposition/absorption needs special treatment. Weight scales are along the original input axis; they cannot simply be transposed as if they were BF16 weights. A BF16 dequantized representation for absorbed K/V weights is a viable first implementation, subject to numerical validation.

### FFN and routing differences

The broad sigmoid/noaux routing structure is reusable, but Flash requires:

- **288 experts**. `csrc/glm_moe_hybrid.cuh` sets `MOE_HYBRID_MAX_EXPERTS=256`; `csrc/glm_mma_moe_coop.cu` also contains 256-entry counters/template capacities and a hard-coded 256-expert launch. Audit every dispatch/workspace path, including decode, rather than changing one constant.
- **Clamped SwiGLU**: cap gate at +10, clamp up to [-10,+10], then `silu(gate) * up`. Apply to dense FFNs, routed experts, and shared experts, including fused kernels.
- **FP32 router computation**. Transformers explicitly computes router logits in float32. Our current `Tensor.validateMoeRoute()` accepts only BF16 logits/bias. Preserve routing precision through the new path and compare expert selections, not only final logits.

## Checkpoint and quantization work

The target index contains 186,381 stored tensors in 36 shards under `tensors/`, totaling 177,680,678,236 tensor bytes (about 165.48 GiB). This is checkpoint storage, not a GPU memory estimate. `ChatModel.loadWeights()` already follows the index into subdirectories.

Most text tensors are under `model.language_model.*`; vision tensors are under `model.visual.*`. Loading, layer-progress detection, CLI model selection, tokenizer configuration, and loader allocation-layout replay need the new model/prefix/config handling.

### MXFP8

Verified representative target headers:

- `layers.0.self_attn.q_proj.weight`: E4M3 `[8192,4096]`.
- Its `weight_scale`: U8 `[8192,128]`, encoding UE8M0 scale per 32 input weights.
- MLA `kv_b_proj.weight`: E4M3 `[32768,512]`, scales U8 `[32768,16]`.

The target recipes use dynamic MXFP8 input activations for attention and shared experts. Our existing E4M3 linear branch looks for `weight.name + "_scale_inv"` and calls the existing FP8 kernel; it does not implement this MXFP8 contract.

Required work: explicit quantization metadata/dispatch, UE8M0 handling, weight/scale TP slicing, and MXFP8 GEMV/GEMM plus activation quantization for native recipe execution. A load-time dequantized BF16 route can help establish architecture correctness, but omitting activation quantization changes the recipe and requires separate fidelity evaluation. It should not be reported as native MXFP8 support.

### NVFP4 and CSF

Main routed-expert values remain packed E2M1 NVFP4. Their E4M3 block scales are losslessly compressed using `byte-window4-fixed-stream-u24-exceptions/1`:

- `.weight_scale.nvfp4_csf_fixed`: U8 fixed stream.
- `.weight_scale.nvfp4_csf_exceptions`: U32 exceptions.
- `.weight_scale_2`: ordinary F32 global scale.
- `.input_scale`: ordinary F32 activation scale, sometimes in a different shard.

Example expert gate weight: U8 `[2048,2048]` representing logical `[2048,4096]`; fixed scale stream `[128,2064]`. These are encoded streams, not directly usable block-scale matrices.

There are 36,288 compressed scale matrices, all in main-decoder routed experts. Decode them losslessly at load time into the existing NVFP4 scale representation, then TP-shard/repack. Persistent compressed-on-GPU execution is a separate optimization, unnecessary for initial support. Expanded scales increase resident memory over checkpoint size. Global/input scales and MTP scales are not CSF-compressed.

**Unresolved:** the inspected vendor code and checkpoint files did not provide the named CSF codec implementation. The model card describes the contract, but byte-exact decoding should be based on the publisher's codec/specification and fixtures rather than inferred from tensor shapes. A search of the fetched vLLM `origin/dev/jovian-judgement` also found neither codec identifier. Acquiring/verifying the codec is an early dependency.

Our GLM loader discards `.input_scale`. Main target experts advertise static NVFP4 activation quantization; MTP experts explicitly advertise BF16 activations. Existing weight-only execution can reuse the decoded weights, but matching the main QAD inference recipe needs an explicit activation-quantization decision and quality tests. QAD itself is a training procedure, not an additional runtime layer.

### MTP and vision precisions

- Layer 45 routed experts: ordinary, uncompressed NVFP4 scales and W4A16 recipe.
- Vision attention: MXFP8.
- Vision MLP/merger: W4A16 NVFP4.
- Other vision tensors: predominantly BF16.

The target model card explicitly says GPU inference and multimodal quality evaluation have not been run for this checkpoint. Its byte-level conversion validation is not an end-to-end inference oracle.

## Serving and parallelism scope

### Hybrid state ownership

The cache must combine 11 decoder MLA caches with 34 recurrent matrices and convolution histories, plus pooled-indexer state. MTP adds another MLA/indexer cache when enabled.

FP32 KDA matrices alone cost `34 * 64 * 128 * 128 * 4 = 136 MiB` per sequence across all head shards. Three-token BF16 convolution histories add about 4.78 MiB per sequence; workspace, pool state, and speculative snapshots are additional. With head-wise TP, recurrent memory is divided across ranks.

`Qwen35GdnState` is useful scaffolding, but uses fixed batch-indexed arrays. `Qwen35ChatCache.prefixMatch()` resets all paged/recurrent state instead of restoring a prefix snapshot. Production support needs sequence-owned slots, reset/release, batching/reordering, and explicit snapshot/copy semantics. Sharing immutable KV pages alone does not restore a recurrent prefix state.

### TP versus CP

Head-wise TP is the natural starting point for KDA. Every rank processes tokens in order for its subset of heads. The 64 heads divide cleanly over the usual 2/4/8-GPU layouts.

Current interleaved-token CP is not directly applicable to recurrent attention: each token depends on the previous state, including tokens assigned to other ranks. Initial Flash support should use TP with CP disabled. Adding CP requires an explicit hybrid design: preserve ordered KDA execution, redistribute activations as needed for MLA, and handle four-token index pools spanning CP ranks. The existing independent partial-softmax merge does not solve the recurrent dependency.

### CUDA graphs, prefix reuse, and MTP

- Preserve stable addresses for convolution/recurrent/pool states and preplanned temporary capacity.
- Ensure plan+plan+run+run cannot advance or overwrite state before the corresponding run.
- Test chunked/phased prefill continuation at convolution and pool boundaries.
- Prefix reuse needs snapshots at chosen boundaries, with a memory/recompute policy.
- Speculative verification mutates all 34 target KDA states. Rejected tokens require restoring an accepted-prefix state or replaying from a checkpoint, including convolution and partial-pool histories. Truncating token counts/KV pages is insufficient.
- MTP layer 45 is a sparse-MLA block with ordinary residual handling; its checkpoint lacks the decoder mHC tensors. Reuse existing embedding/hidden projection and sampling machinery, but check hidden-state normalization, pooled selection reuse, and W4A16 expert semantics against the fork reference.

### Multimodal scope

Full `ForConditionalGeneration` support also requires the 24-layer width-1024 vision encoder, spatial/temporal patch embedding, vision positions/attention, merger to width 4096, and mixed-precision loading. Processor work includes dynamic image resizing/patch grids, video frame handling, image/video token expansion, and embedding insertion. API request handling must feed these embeddings to the text model.

Text-only support can be delivered first with an explicit model-loading policy for vision tensors. Model selection, chat templates, EOS handling, and text tokenization should still be checked against the target tokenizer assets rather than assuming identical behavior from the shared vocabulary size.

## Recommended implementation milestones

| Milestone | Deliverable | Relative effort |
|---|---|---|
| 1. Checkpoint contract | Pin revision; acquire CSF codec; decode fixtures; enumerate per-tensor formats; new config/weight mapping; memory budget | Medium, with external codec dependency |
| 2. Primitive correctness | KDA decode/reference prefill, sigmoid gated norm, mHC, clamped SwiGLU, 288-expert and FP32 routing support | Large |
| 3. Text forward | New hybrid model; no-RoPE MLA; pooled indexer; hybrid caches; eager prefill/decode; TP | Large |
| 4. Efficient native execution | Chunked KDA prefill; MXFP8 kernels; quantized MLA transformations; graph capture; workload tuning | Large |
| 5. Production state lifecycle | Multi-request slot ownership; prefix snapshots; phased prefill; MTP rollback; loader/restart validation | Large |
| 6. Multimodal | Vision encoder, mixed precisions, processor, embedding insertion, image/video API tests | Separate large workstream |
| 7. Hybrid CP | Explicit recurrent/MLA partitioning and redistribution design | Separate research/optimization workstream |

Some primitive and loading work can proceed independently. The critical dependencies are the CSF decoder and a trusted numerical reference for the target quantization recipe. This is a multi-workstream model port, not a small configuration patch; a calendar estimate before those dependencies and kernel reuse decisions would be unreliable.

Likely project touchpoints: a new model/config module, `model_cli.ts`, `run_model_loader.ts`, shared model loading, `tensor.ts`, `device_ops.ts`, `glm_ops.ts`, `parallel_ops.ts`, native addon bindings and CUDA kernels, `execution-workspace.ts`, hybrid cache/state ownership, capture integration, and speculative/prefix-cache orchestration. Extract reusable GLM operations instead of making the existing full-model class carry incompatible residual/cache assumptions.

## Verification gates for future implementation

1. **Storage:** byte-exact CSF reconstruction, UE8M0 decode, cross-shard scales, weight slicing, TP scale alignment, deterministic loader allocation replay.
2. **Primitives:** KDA sequence versus stepwise execution, resumed chunks, mHC pre/post/mixing numerics, clamping, all 288 router candidates including expert IDs 256–287.
3. **Indexer:** lengths 1–5, chunk/prefix boundaries inside pools, long contexts that actually require top-k selection, 2051-wide output, causal tail inclusion, packed batches.
4. **Model:** layer-by-layer hidden states/logits against pinned reference math; compare expert and pool selections as well as aggregate numerical error. Distinguish BF16-dequantized bring-up from native quantization fidelity.
5. **Serving:** reset/reuse and request reordering, eager versus graph replay, TP consistency, prefix continuation, multi-token verification with partial/full rejection, phased prefill, executor restart against resident weights.
6. **Performance:** measure actual prefill/decode/batch/context workloads only after correctness. No performance or target-checkpoint inference claims were established by this investigation.
