# DeepSeek V4.1 Flash implementation

Target checkpoint: `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF`, at
revision `90d665c19e1a9ff2097cc787515e370d86b616f0`.

## Supported operations

End-to-end text generation uses a GPU-resident forward pass with the resident TP
weights. `DeepseekV41Model` supports the ordinary `ChatModel` prefill/decode
interfaces, including batched and chunked inputs and CUDA-graph-captured decode.
Prefill runs eagerly. CP, vision, and pinned-host cache
offload are not enabled. Independent test-side numerical oracles check the
production kernels; there is no separate reference executor or CLI.

DSpark speculative decoding is available with `--mtp 4` (verification depths
1–5 are supported). The checkpoint's three draft layers always process all five
positions; only the requested proposal prefix is verified. Drafting is greedy
after the full-vocabulary rank-256 Markov correction. Sampled targets use exact
point-mass rejection sampling under the existing MTP sampling constraints.
Confidence-based adaptation and sampled drafting are not enabled.

The model collects mHC-mean inputs to backbone layers 37–39, projects their
concatenation, and writes context KV for all three draft layers. Every draft
query sees the same recent 128-token context plus the complete five-position
block. Draft context uses a bounded paged cache with absolute RoPE positions.
Generator startup replays at most 129 committed target tokens (the start is
aligned to a compressor pair); subsequent verification forwards update context
incrementally. Acceptance truncates both KV caches and selects immutable
compressor projection rows at the accepted boundary, without target replay.
Drafting, target verification and context updates share a CUDA graph; verification
indexer scratch uses padded KV-length buckets. Tests compare eager/captured
verification at identical batch/depth geometry, not single-token decode parity.

### Batched forward execution

`DeepseekV41Model.forwardPhased()` executes all tokens in an `ExecutionState`
layer by layer, yielding at layer boundaries. Dense/Engram/shared-expert
projections, normalization, routing, and selected-expert CUDA-core matrix products
operate on the combined token batch. Paged appends, index scoring, slot mapping,
and sparse attention consume all sequences together using per-token sequence IDs
and the plan's page-table snapshot. Decode uses the same batched forward path.
Alternating-chunk phased prefill is not implemented.

Tensor operations use `Tensor`, with allocation/native dispatch in `GlmTensor`
and shard orchestration in `ParallelTensor`. FP8 projections use `linear()` with
checkpoint-associated scales; BF16 projections request `outputType: "F32"` where
required. Grouped attention output uses `linear(..., { groups: 8 })`. MXFP4
experts use `mulMatId()` with flattened routing IDs; its backend caches expert
pointer tables. BF16 and F32 linear outputs share the native BF16 GEMV/Lt dispatch:
thread accumulators, shuffle reductions and shared partials remain FP32, with
output conversion only at the final store. Clipped activation and sqrt-softplus routing extend `siluAndMul`
and `moeRoute`. Specialized rotary, mHC, compressor, Engram and reduction kernels
are tensor methods. Paged cache/indexer/attention execution stays on `DeviceOps`.
Compositions and hash-constant setup stay in `deepseek_v41_model.ts`. There is no
separate execution-wrapper layer, DeepSeek parent interface or opcode dispatcher.

Local KV remains in ordinary paged storage; attention slots select the last 128
tokens. Four compressed-cache owners and their indexer keys use the same logical
64-input-token page IDs. Ratio-2 caches use the first 32 records per page. Prefix
sharing, copying, COW and truncation use the existing `PagedKVCache` lifecycle.
An odd resumed boundary reconditions the unfinished compressor pair with a
one-token overlap; full prefixes are not replayed. Engram uses three lookback
token IDs supplied by planning. Indexer selections/candidates have one row per
query; score scratch uses prefill lengths or captured decode KV-length buckets.

Compressor carry tensors are immutable inputs: each forward produces fresh
FP32 value/gate and position allocations, owned through `UsingHolder`. A
`CompressorCarry.viewClone()` preserves a boundary without copying or being
clobbered by later pairs. Planning reserves the outputs outside CUDA capture
and retains the preceding inputs per plan slot, including plan+plan+run+run.
Decode graph keys include all carry input/output addresses; released allocations
are recycled normally, while retained views prevent premature reuse. Ratio-1
compression has no carry. Prefix-cache reuse still uses the one-token overlap.

Batched GEMM/prefill attention and single-token GEMV/decode attention have
different rounding paths. Full-model hidden states and near-tied greedy choices
can differ with chunk geometry; eager-versus-graph comparisons use identical
prefill geometry. Primitive tests separately check causal masks, page boundaries,
hash history, routing, quantization, and per-token expert selection.

| Component | Implementation |
|---|---|
| MXFP4-CSF loading | `src/mxfp4_csf_codec.ts`, `src/csf_checkpoint.ts`: lossless CPU expansion at load time |
| Weight loading | `src/deepseek_v41_model.ts`: native FP8/MXFP4 payloads, FP32 mHC parameters, TP sharding, bounded Engram uploads |
| Persistent execution | `src/model_registry.ts`, `src/run_model_loader.ts`: CUDA IPC allocation replay |
| Sparse attention | `GlmOps.dsv4SparseAttention`: FlashInfer SM120 DSV4 single/dual-cache prefill and decode, sinks and valid lengths |
| Quantized linears | `csrc/glm_fp8.cu`: FP8 CUDA-core kernels for decode/small batches; cuBLASLt MXFP8 GEMM with temporary packed scales and quantized activations for larger batches. Routed MXFP4 uses the shared FP4 kernel in `csrc/glm_gemv.cu` |
| Delayed mHC | `Tensor.dsv41MhcPre`: parallel FP32 coefficient/split-K projection, warp-parallel 4x4 Sinkhorn, and previous-sublayer pre-mix |
| Engram | `src/deepseek_v41_engram.ts`: verified tokenizer map; backend n-gram hashing, tensor vocabulary-sharded embedding and gating |
| Compressor | Model-owned projection composition; `Tensor.dsv41Compress`: ratio-1/2 FP32 pooling and normalization with cache-owned pending-pair state |
| Text decoder | `src/deepseek_v41_model.ts`: complete batched forward and layer helpers, composing tensor and paged execution operations |
| Cache and selection | `src/deepseek_v41_cache.ts`: storage and prefix sharing; `GlmOps`/`ParallelOps`: paged appends, physical slot mapping and index scoring |
| Generation | Standard `ChatModel` interfaces and `src/openai-server.ts` |
| Chat formatting | `src/deepseek_v41_chat.ts`: V4.1 role markers, reasoning-effort prefix, thinking toggle, spaced-DSML function calls, tool-result history, and reasoning/content/tool output parsing; supplies the template absent from checkpoint metadata |

The FP8 kernel supports compile-time 32x32/E8M0 and 128x128/BF16 scale contracts,
vector loads, configurable rows per warp and small-output split-K. It replaces
the old FP8 GEMV/shared-memory GEMM implementation without adding FP8 support to
the FP4 kernels. DeepSeek uses it through eight local input rows, including its
grouped attention output projection. Larger batches use cuBLASLt MXFP8: checkpoint
32x32 E8M0 scales are temporarily repeated/packed into the MMA layout, and BF16
activations are dynamically quantized to E4M3 with per-row K32 E8M0 scales.
All preparation buffers and the 32-MiB cuBLASLt workspace are scoped temporaries.
K not divisible by 128 is padded in a temporary FP8 buffer; persistent checkpoint
weights/scales remain unchanged. Shapes not preserving 32-channel blocks use the
CUDA-core path. Activation quantization changes the numerical contract relative
to the previous BF16 expansion/GEMM path. Routed MXFP4
projections use the existing NVFP4 CUDA-core kernel with a compile-time E8M0 scale
policy: identical 16-weight work units and vector loads, one scale per two units,
and no global scale. They do not materialize token-specific BF16 weight matrices.
Expert-grouped tensor-core MXFP4 execution remains unimplemented.
Native MXFP4 A16/A8 implementations are available as references under
`vendor/b12x/b12x/moe/`; their packing/launch contracts still need integration.

Attention `wq_a`, `wkv`, `indexer.wq_b`, and Engram `wkv` weights and scales are
column-sharded at load time, and FP8 projection dispatch preserves that layout.
Gathering is owned by consumers: query/KV normalization, index scoring, and Engram
gating. Indexer queries remain sharded through RoPE and per-head quantization.
On TP8, each GPU stores and computes one eighth of these projection weights
instead of a complete replica.

The GPU executor keeps RoPE, routing, index selection, and intermediate outputs
on device. It uses the 584-byte DSV4 KV layout and FP8 index keys/queries; it does
not implement the alternative 528-byte all-dimension MXFP8 cache or MXFP4 indexer
cache. All six routes share one gate launch, one up launch, and one down launch
per layer/GPU. They select weight pointers on device and compute directly from
packed MXFP4 weights, with FP32 accumulation and BF16 outputs. The model uses
`swiGluMlpMoeReduce`, whose generic `scatterAddRows` reduction combines local
expert outputs with F64 routing weights into an F32 `[tokens, hidden]` tensor.
The shared expert's TP-partial output is included locally. One F32 collective
then completes the combined output before its final BF16 conversion. This uses
3.5x fewer MoE output payload bytes than separately reducing six BF16 routed
outputs and one BF16 shared output. Generic replicated shared inputs are included
on rank zero only, avoiding multiplication by the TP world size.
Routing uses FP32 sqrt-softplus scores, biased comparisons, normalization and
output weights, with lower-ID tie breaks. Bias affects selection only. Small
batches evaluate scores across a block before warp-local top-k; large batches
process four independent rows per block with register-resident scores. Target
routing uses 384 score threads through 512 tokens, 128 threads through 1,024,
then four rows per block. Draft routing uses 128 score threads through 2,048
tokens, then four rows per block. Index scoring
uses FP8 Tensor Core dot products with FP32 accumulation and weighted per-head
ReLU, followed by BF16 score rounding before candidate selection and top-k.
Small workloads use one query per CTA; larger workloads reuse a 64-key tile
across four queries, with independent compressed causal bounds for each query.
Mixed-sequence query tiles reload keys per query. Dispatch uses
query count and compressed key capacity; `SLOPLLM_DSV41_INDEXER_QUERY_TILE=1|4`
overrides it for kernel comparisons.

Index selection shares GLM's BF16 radix top-k kernels. At most 64 query rows use
the general split-radix dispatcher; larger batches use the prefill histogram
pipeline with deterministic gathering. Selection keeps the lowest logical
positions at cutoff ties; output follows GLM's threshold/position order rather
than descending score order. Invalid or masked scores leave `-1` output slots.
Preparation applies each query's compressed causal length and logical
eight-position candidate-block mask to a temporary score buffer. Layer 20
computes each block maximum once, selects 2047 non-latest blocks, and includes
the latest block unconditionally. Scratch is workspace-owned and graph-reusable;
the preparation pass refreshes padding and masks on every replay. FP8 scoring
still evaluates all causal keys before this selection-time masking.

mHC's projection and squared norm use FP32
split-K partials; inverse RMS is applied to the merged projection before coefficient
activation. Below 128 tokens the projection uses FP32 multiply/accumulate.
Larger batches use Tensor Cores with TF32-exact BF16 activations and FP32 weights
split into TF32 high and residual terms. Separate FP32 high/low accumulators are
merged every 32 K elements. Tiles cover 16 tokens below 2,048 tokens and 32 tokens
at larger sizes. `SLOPLLM_DSV41_MHC_PROJECTION=fp32|tf32` overrides the standalone
projection arithmetic for comparisons. Mixing coefficients, Sinkhorn, and four-stream collapse
use FP32 arithmetic; collapsed hidden states are rounded to BF16 only at output.
Tensor Core tiles use double-buffered 16-byte `cp.async` global-to-shared copies,
overlapping the next tile's transfer with current-tile computation. Unaligned
inputs or row widths use synchronous loads. `SLOPLLM_DSV41_MHC_PIPELINE=0|1`
controls asynchronous loading for comparisons without changing the arithmetic.
For fewer than 128 tokens, the native projection groups four coefficients per
CTA to reuse residual loads. When a following RMSNorm weight is supplied, this
stage also prepares the BF16-rounded collapse and its FP32 squared-norm partials;
the finalizer normalizes that prepared input while producing mixing coefficients.
The model enables this fused RMSNorm for small target and DSpark batches.
The decode Sinkhorn path uses butterfly row/column reductions and approximate
FP32 reciprocals, following b12x's native finalizer. This changes reduction and
rounding order and can change generated tokens and speculative acceptance.
`SLOPLLM_DSV41_MHC_DECODE_GROUPED=0|1`,
`SLOPLLM_DSV41_MHC_DECODE_NORM=0|1`, and
`SLOPLLM_DSV41_MHC_SINKHORN=0|1` independently control these paths.
For at most 32 tokens, `SLOPLLM_DSV41_MHC_POST_PRE=0|1` enables (default) or
disables fused post/pre preparation. Its 128-channel tiles compute all four
post-mixed residual streams, round each to BF16, and reuse them for four
projection coefficients plus collapse/norm preparation. Scratch uses one K
partial per 128 hidden channels. The finalizer produces the next coefficients
and normalizes the prepared input. The materialized residual remains available
for DSpark feature taps. Partial-sum block outputs are all-reduced before local
fusion, preserving the original communication and BF16 boundaries.
Prepared mixes are owned across phased yields. Target decode fuses 77 of 80
post operations; the two Engram boundaries and final output retain standalone
post-mixing. DSpark fuses five of six post operations. Larger batches use the
separate path because the fine-grained fused tiles regress at 64 rows.
Set all four mHC overrides to zero for the preceding unfused decode arithmetic
and organization (the fast-exponential policy below still applies).
No BF16 intermediate is introduced in the coefficient computation.
Project-owned CUDA exponential evaluations use fast single-precision intrinsics
(`__expf`, or approximate base-two exponentiation for CP correction), including
the clipped SwiGLU exponential. Other arithmetic in clipped SwiGLU retains its
existing precision. Fast exponentials can change routing and generation near
rounding/selection boundaries.
Indexer queries and keys use 132-byte records: 128 E4M3 bytes followed by one
power-of-two FP32 scale. With 2,048 pages, persistent replicated cache storage
is approximately 3.20 GiB/GPU. Index-score tests compare FP32-reference scores
after BF16 rounding, allowing reduction-order differences at rounding boundaries;
they do not require byte equality with a JavaScript FP64 accumulation.

### CUDA graph execution

Backend-owned expert pointer tables are prepared during eager warmup before capture;
the model initializes hash constants and expert-weight references after weights
and tokenizer loading, before its workspace freezes. Hash constants are named
model-owned tensors included in persistent layout validation/export. Executors
replay their allocations without uploading over the loader-owned data. Per-forward
preparation only updates execution-plan data such as lookback and page tables.
Token IDs, positions, page tables and lookback are device inputs; writes are
position-addressed so warmup/capture can repeat a step. Capture keys identify
stable cache/planning bindings and KV-length buckets. FP32 cuBLAS projections use caller-owned workspaces to
avoid graph allocation/free nodes. FP8 decoding uses the existing high-lane
conversion helper, including weights, Engram lookup, and indexer quantization.

The native graph backend explicitly calls `cudaGraphUpload()` after instantiation,
on the launch stream, without host synchronization. This works around an observed
first-launch host stall with the full TP8 forward graph; the implicit-upload
stall's root cause remains unresolved. No environment override is required.

## Generate text

The model's chat cache derives its sequence capacity from the logical page budget,
independently of prefill chunk size. Each chunk runs as a batched, layer-major
forward rather than interleaving two chunks of one sequence.

For an OpenAI-compatible server with captured decode, stop an existing executor before:

```bash
curl -X POST http://127.0.0.1:8099/fork \
  -H 'content-type: application/json' \
  -d '["src/openai-server.ts", "--host", "0.0.0.0", "--port", "8000", "--batch-size", "1", "--chunk-size", "128", "--max-pages", "2048", "--max-tokens", "64"]'
```

This configures a 131,072-token shared page budget and 128-token prefill chunks.
Shape-dependent FP8 GEMM performance and unfused expert execution still limit throughput.
Use `--no-cuda-graph` for eager GPU decode when comparing execution modes.
Allocation-stack tracing is opt-in with `SLOPLLM_TENSOR_ALLOCATION_STACKS=1`;
leave it disabled for throughput measurements.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF","messages":[{"role":"user","content":"What is the capital of France? Answer with only the city name."}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}'
```

Thinking defaults to enabled, with reasoning effort 75. Set `enable_thinking:false`
or `reasoning_effort:"none"` for direct answers. Supported reasoning efforts are
integers 1–100 or `low` (50), `high`/`xhigh` (75), and `max` (100).

## Resident weights and executor iteration

Start a resident loader:

```bash
NCCL_P2P_LEVEL=SYS NCCL_TOPO_FILE=/root/chat/vllm/topo_fixed.xml \
  npx tsx src/run_model_loader.ts \
  --arena 85 --gpus 0,4,5,7,1,2,3,6 \
  --model local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF
```

Executors use the existing `/fork`, `/spawn`, and `/restart` control endpoints.
An executor calls `loadModelRuntime(parseModelArgs(process.argv.slice(2)))` and
uses `DeepseekV41Model.tensors` with checkpoint-native names. The loader retains
the arena allocations after the executor exits. Model input/output experiments
belong in `scratchpad/`.

TP8 loading produced 93,418 resident text tensors and **59.249 GiB of logical
weight payload per GPU**, including expanded MXFP4 scales and replicated weights.
The 85-GiB arena reserves additional runtime space; `nvidia-smi` reports the arena
reservation rather than logical weight usage. Vision tensors are skipped; draft
tensors are loaded only when `--mtp` is enabled. Changing that flag requires
restarting the persistent loader.

Source payload accounting before runtime replication/padding:

| Category | GiB |
|---|---:|
| Backbone routed experts | 268.945 |
| Engram tables and projections | 189.127 |
| Other backbone text weights | 8.877 |
| Draft/MTP | 7.388 |
| Vision and other excluded tensors | 0.904 |

Packed MXFP4 `I8` tensors and E8M0 scale tensors are stored as opaque `U8`, retaining
their bytes. `sourceMetadata` records original shapes and dtypes. Engram tables
are column-sharded over contiguous vocabulary ranges, padded only at the end;
`engramRows` records unpadded sizes. Transfers are bounded to 64 MiB, and periodic
load/replay synchronization releases staged expanded scales.

## Numerical and state contracts

- DSV4 cache storage is `[pages,64,584]` bytes. Each page contains 64 consecutive
  576-byte data records (448 E4M3 bytes plus 64 BF16 rotary elements), **followed
  by** 64 eight-byte scale footers. Scales are not interleaved with token data.
- Attention indices are physical slots. Local/global visibility and causal masks
  must already be encoded in the indices/lengths. Returned LSE uses base-2 units;
  learned sinks use natural-log units.
- V4.1 mHC collapses the input with the previous sublayer's pre-mix. Omitting it
  selects stream zero for the initial embedding. The returned pre-mix belongs to
  the following sublayer; ordinary GLM mHC cannot be substituted unchanged.
- Engram token normalization is checked against the Rust-tokenizer reference
  map SHA-256. Hash multipliers are the published NumPy recipe evaluated for
  this checkpoint's layer IDs and compressed vocabulary. Explicit three-token
  lookback makes CPU hashing invariant to chunk boundaries.
- Compressor projections use BF16 weights/inputs with FP32 accumulation and output,
  including gate scores; no BF16 rounding occurs before pooling. Ratio 2
  saves one unfinished pair per sequence slot. `valid=1` marks emitted latents,
  `0` marks unfinished pairs, and `-1` reports a missing predecessor. Slot reuse,
  prefix restoration, and speculative rollback must restore/reset this state.
- Replicated caches are supported by the attention wrapper; context-parallel
  cache partitioning is not implemented.

## Validation

```bash
npm run build:all
npx tsx --test tests/test_deepseek_v41.ts tests/test_deepseek_v41_chat.ts tests/test_deepseek_v41_device.ts
npx tsx --test tests/test_csf_codec.ts tests/test_csf_checkpoint.ts
.venv/bin/python -m pytest -q \
  tests/python/test_deepseek_v41_attention.py \
  tests/python/test_deepseek_v41_quantization.py \
  tests/python/test_deepseek_v41_compressor.py \
  tests/python/test_deepseek_v41_oracles.py \
  tests/python/test_flash_hybrid.py
```

The checkpoint audit reconstructed all 46,080 compressed scale tensors and
matched their recorded source SHA-256 values. IPC executor validation compared
sampled expert weights/scales, projections, embeddings, and Engram shard-boundary
rows against source bytes. Replay also succeeded with tensor-payload reads
disabled. Resident-weight component execution covered mHC, query projections,
Engram lookup/injection, and an MXFP4 expert projection. Resident TP8 compressor
execution covers projection through pooling/normalization at layers 2 and 20.
For two synthetic input rows, checkpoint-weight projections matched CPU FP64
dot products with maximum absolute errors of 2.39e-8 and 1.48e-8, respectively.

Kernel tests compare sparse attention, dequantization, mHC, Engram gating, and
compression against independent PyTorch calculations. FP32-output projection tests
compare against FP64 dot products and reject intermediate BF16 rounding. They cover TP8 expert
shard geometry, dual-cache normalization, empty selections, runtime head counts,
and compression continuation across odd chunk boundaries. TypeScript tests cover
TP1/TP2 wrappers and Engram hashing against Rust-tokenizer/NumPy reference hashes.

Resident full-decoder checks:

- Raw prompt `The capital of France is` generated ` Paris.` followed by EOS;
  five input tokens and three generated tokens took 1.80 seconds in the TP8 run.
- A 543-token prompt crossed sliding-window rollover and the 512-candidate
  selection boundary, then answered `What is 2 + 2?` with `4`.
- `tests/test_deepseek_v41_generation.ts`, enabled with
  `SLOPLLM_TEST_DSV41_RESIDENT=1` and the loader's model arguments, validates
   batched/chunked prefill, actual graph instantiation/replay, eager-versus-captured
    decoded-token equality, and rollback reconditioning. Its France/Germany prompts select
   Paris/Berlin. A 120-token prefix followed by 12 decode steps verifies replay
   across the 128-token local-window rollover and page boundary, with identical
   eager/captured tokens and no new graph warmup at that boundary. A single
   256-token prefill chunk also selects Paris, exercising a chunk larger than
    the local attention window. Additional checks count forwarded tokens to prove
    active/staged prefix reuse, one-token odd-boundary reconditioning, pending-target
    preservation, changed batches, and one-token prefill chunks with separate overlap plans.
- A resident full-forward diagnostic compared hidden outputs at positions 0–9:
  all were bit-identical, including graph instantiation at position 3 and subsequent
  replay. These checks used the existing high-lane FP8 conversion workaround.
- A checkpoint layer-0 comparison against independent PyTorch dequantized-weight
  calculations measured relative RMS error 0.00403, cosine similarity 0.999992,
  and maximum absolute error 0.00390625. This is not full-model vendor-runtime
  numerical parity; activation quantization and TP reduction rounding differ.
- OpenAI chat requests returned `Paris` with thinking disabled, and `2 + 2 = 4`
  with separate `reasoning_content` at reasoning effort 1. Both terminated at EOS.
  Streaming chat returned `Berlin`, a `stop` finish reason, and `[DONE]`.
- With captured GPU decode enabled, both nonstreaming and repeated streaming
  requests returned `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` (29 completion tokens), with
  a `stop` finish reason and the streaming `[DONE]` marker.

## Remaining optimization and validation

1. Full-model/layerwise vendor comparisons, longer-context evaluation, and
   hierarchical candidate-selection evaluation beyond 16,384 positions.
2. Expert-grouped tensor-core MXFP4 and shape-specific FP8 prefill tuning; a custom
   MXFP8 kernel could consume compact checkpoint scales without temporary expansion.
3. CP, vision, host-cache offload, and more compact indexer storage.

## Paged-refactor serving check

With TP8, 2,048 replicated pages and 128-token chunks, an uncached 258-token chat
prompt returned Paris in 3.395 seconds (79.7 prefill tokens/s). Repeating the
request reused 256 cached tokens, forwarded only two prompt tokens and completed
in 0.279 seconds. Both generated two completion tokens and terminated normally.

After replacing token-specific MXFP4 weight expansion with the shared CUDA-core
kernel, the same uncached 258-token request measured 573.1 prefill tokens/s and
0.586 seconds end to end. Its cached repeat took 0.225 seconds. These are serving
smoke measurements at the same TP8/page/chunk settings, not a broad throughput
benchmark. Both returned Paris. The resident generation suite also passed eager/
captured equality, boundary reconditioning, and active/staged prefix-reuse checks.

The shared kernel is checked against independent unpacking plus FP64 dot products
for decode and batched inputs, including TP8's 288-channel down projection, partial
row tiles, all six routes, repeated experts and E8M0 encoding edge cases.

The separate FP8 kernels are checked against FP64 dot products for both scale
contracts, grouped inputs, partial rows/columns, every finite E4M3 encoding and
E8M0 exponent edge cases. Captured projection microbenchmarks on resident TP8
weights measured 1.8–5.7x faster single-row projections than BF16 expansion.
A 32-token steady-state captured decode comparison (after eight warmup steps)
measured 9.29 tokens/s with FP8 expansion versus 9.61 tokens/s with direct FP8.
These component speedups do not translate into proportional whole-model speedups.

## Historical session-based throughput check

On resident TP8 weights, with 1M context capacity and allocation-stack tracing
disabled, a 128-token prompt measured 7.48 tokens/s at chunk size 1, 50.14 tokens/s
at chunk size 32, and 58.75 tokens/s at chunk size 128. Each case includes initial
session setup; all selected ` Paris` for the final prompt suffix. Captured decode
after warmup measured 8.53 tokens/s over 16 steps. These are integration-path
measurements, not optimized quantized-kernel performance.

The OpenAI server with chunk size 128 processed a 1,017-token uncached prompt at
84.0 prefill tokens/s and returned `4` for `2+2` in 12.26 seconds end to end.
A subsequent 1,221-token request reused a 960-token prefix (reconstructed in
batched forwards), crossed the ratio-2 indexer's 512-key threshold, and streamed
`1, 2, 3, 4, 5, 6, 7, 8, 9, 10` with `[DONE]`.
