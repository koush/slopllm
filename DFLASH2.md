# Standalone DFlash2

`src/dflash2_model.ts` implements the BF16 `incoai/GLM-5.3-DFlash2`
checkpoint as a `ChatModel` subclass, without a target model or tokenizer.
It uses the shared `ChatModel.loadWeights()` / `loadTensor()` pipeline and
`loadModelRuntime()`. Chat generation (`forwardPhased()`, `createChatCache()`,
and the standalone EOS policy) is unsupported and throws; drafting uses the
dedicated APIs below.

## Computation

`forwardDflash2Phased(contextState, blockState, targetHidden, blockEmbeddings)`
returns a generator whose result is a BF16 tensor of shape
`[batch * 7, 6144]`: the seven final normalized prediction rows.

- `targetHidden` contains six BF16 `[contextTokens, 6144]` tensors, ordered
  according to `cfg.targetLayerIds` (5, 19, 33, 47, 61, 75).
- `contextState` is `ws.planPrefill(model, batch, contextLengths, cache)`.
- `blockState` is `model.planBlock(ws, cache)`, planned after the context.
- `blockEmbeddings` is BF16 `[batch * 8, 6144]`, containing one anchor embedding
  followed by seven mask embeddings per request. The standalone tests supply
  synthetic embeddings; no target embedding or LM head is loaded.
- The caller owns allocation tracking and disposes the returned tensor.

The context projection writes ordinary per-layer GQA K/V. The block forward
uses non-causal attention with a 2,048-token window, dynamic two-tap grouped
convolutions, and SwiGLU MLPs. Inverse RoPE frequencies remain FP32.
The checkpoint's default split-half RoPE layout is used unless explicitly
overridden in its configuration.

The context projection `fc.weight` stays replicated in weight storage. For up to
64 input rows, tensor-parallel execution narrows it to per-GPU output slices;
the following RMSNorm gathers those slices. Convolution coefficient and selector
projections remain replicated: their output gathers did not justify narrowing
in the measured TP8 workload.

`prepareContextPhased()` and `forwardBlockPhased()` are also exposed separately
for repeated drafting against an existing cache. The block K/V is temporary;
the caller must truncate those slots before appending more target-derived context.
The standalone APIs use normal paged storage with window masking and leave cache
management to their caller. The GLM decode generator described below bounds that
storage and handles speculative acceptance.

`selectCandidates(hidden, ids, logits, anchors)` applies the checkpoint's learned
selector to externally supplied top-16 candidates. Inputs are BF16 prediction
hidden states, I32 `[batch * 7, 16]` IDs, F32 or BF16 logits of the same shape, and I32
anchor IDs. It returns F32 `[batch * 7, 16, 16]` pairwise scores and I32
`[batch, 7]` greedy tokens. It does not sample or run target verification.

## Independent reference fixtures

Build before testing:

```bash
npm run build:all
.venv/bin/python -m pytest -q tests/python/test_dflash2.py
```

The fixture generator uses PyTorch linear algebra, explicit GQA head expansion,
FP32 attention softmax, and FP32 intermediate arithmetic for fused norms,
convolutions, and rotary operations. It does not call the native addon.

For a strict end-to-end test, create a small, well-conditioned six-layer GQA
checkpoint (the command refuses to overwrite an existing checkpoint):

```bash
.venv/bin/python tests/python/dflash2_reference.py \
  --make-test-model --model-dir scratchpad/dflash2-small-gqa \
  --context-lengths 2053,2061 --output scratchpad/dflash2-small-gqa-reference

npx tsx src/run_dflash2_test.ts --dflash2 \
  --model-dir scratchpad/dflash2-small-gqa --gpu 0 \
  --fixtures scratchpad/dflash2-small-gqa-reference --relative-tolerance 0.01
```

Use `--gpus 0,4,5,7,1,2,3,6` instead of `--gpu 0` for TP8. On the production
two-switch host, also set `NCCL_P2P_LEVEL=SYS` and
`NCCL_TOPO_FILE=/root/chat/vllm/topo_fixed.xml`.

For the real checkpoint, first download its `config.json` and `model.safetensors`
and set `DFLASH_MODEL_DIR` to their containing directory:

```bash
.venv/bin/python tests/python/dflash2_reference.py \
  --model-dir "$DFLASH_MODEL_DIR" --output scratchpad/dflash2-reference

npx tsx src/run_dflash2_test.ts --dflash2 \
  --model-dir "$DFLASH_MODEL_DIR" --gpu 0 \
  --fixtures scratchpad/dflash2-reference --layerwise
```

The real trained convolutions amplify rounding differences on unrelated random
features. `--layerwise` checks each layer against independent reference boundary
inputs and checks the convolution and selector separately. It also runs the full
forward, reports its accumulated numerical drift, and requires exact agreement
between eager and captured/replayed results. **It does not assert full-chain
reference parity in this mode.** The synthetic checkpoint exercises full-chain
parity with tighter tolerances. This validates arithmetic, not token acceptance
or target-model compatibility.

The default six iterations include three eager warmups, graph capture, and two
replays. `--eager` disables capture. The reported time includes context projection
and KV construction, the block forward, and the independent selector fixture;
it is not a steady-state speculative-decoding benchmark.

## Persistent loader

Load only the drafter into the resident arena:

```bash
NCCL_P2P_LEVEL=SYS NCCL_TOPO_FILE=/root/chat/vllm/topo_fixed.xml \
npx tsx src/run_model_loader.ts --dflash2 --arena 8 \
  --gpus 0,4,5,7,1,2,3,6 \
  src/run_dflash2_test.ts --fixtures scratchpad/dflash2-reference --layerwise
```

`--dflash2` defaults to the Hugging Face cache for `incoai/GLM-5.3-DFlash2`.
It can instead use `--model-dir`. It cannot be combined with target-model flags,
CP, or MTP. The ordinary loader CUDA IPC and allocation-layout validation apply.

Repeat the executor after editing the TypeScript model or test:

```bash
curl -N -X POST 'http://127.0.0.1:8099/restart?follow'
```

## GLM-owned drafter and combined capture

`--glm51 --dflash` loads the target followed by an owned `Dflash2Model` on the
same backend. `--dflash-model-dir <path-or-repo>` selects a different draft
checkpoint and also enables it. `--dflash2` remains the standalone draft mode.
The target exposes its drafter through `model.dflashModel` and frees it in
`model.free()`. Loader allocation replay and named-weight layout signatures
include both models. Changing the draft checkpoint requires restarting the loader.

`Glm51Model.forwardDflashPhased(targetState, contextState, blockState)` runs:

1. Target forward, retaining residual-output views from the six configured layers.
2. Draft context projection and KV writes from those features.
3. Block embedding through the target embedding table and the draft block forward.
4. Target LM-head projection, global top-16 selection, and the greedy draft selector.

Feature views are owned by scoped holders and released on completion, exceptions,
or generator cancellation. The returned `{ targetHidden, draftHidden, scores, tokens }`
tensors belong to the caller. No target final RMSNorm is applied to `draftHidden`.

Prepare all three plans and upload inputs **before** execution. The target may
use a decode or prefill plan; draft context uses a prefill plan with matching
per-sequence input-row counts, and the draft block uses `draft.planBlock()`.
Context and block share a draft KV cache and workspace. The target can use the
same execution workspace: each plan gets distinct metadata buffers.

Populate block inputs with `[anchor, mask, ...mask]` for each sequence, using
`draft.cfg.maskTokenId`. Anchors are caller-supplied, not sampled by this method.
Match draft context positions and committed cache prefixes to the target rows.
The method appends every supplied target row; acceptance filtering and removal
of temporary block KV remain caller responsibilities.

Capture with all three states in one `CaptureManager.execute()` call:

```ts
const execution = capture.execute({
  states: [targetState, contextState, blockState],
  inputs: {}, // Plan-slot input buffers are captured through the states.
  key: ["glm-dflash"],
}, () => {
  const forward = model.forwardDflashPhased(targetState, contextState, blockState);
  while (true) {
    const next = forward.next();
    if (next.done) return next.value;
  }
});
```

The integration test compares combined execution with separately run target and
draft forwards, closes generators early to check cleanup, and tests both prefill
and decode graph replay with changed anchor inputs at batch 2:

```bash
NCCL_P2P_LEVEL=SYS NCCL_TOPO_FILE=/root/chat/vllm/topo_fixed.xml \
npx tsx src/run_model_loader.ts --glm51 --dflash --arena 92 \
  --gpus 0,4,5,7,1,2,3,6 --cp \
  src/run_glm51_dflash_test.ts
```

Use `/restart?follow` for subsequent test iterations without reloading either model.

## Decode generator and OpenAI serving

`model.generateDflashDecode(ws, cache, executionManager?, samplingPolicy?, numDraftTokens?)`
consumes each sequence's pending target token and yields the same step contract
as MTP: `{ tokens, numAccepted, numDraftTokens, warmup }`. It proposes seven tokens,
verifies `[pending target, ...draft]` in one target prefill, commits the accepted
prefix, and publishes the target replacement/bonus token as the next pending input.

The optional experimental `numDraftTokens` argument verifies only the first 1–7
proposals (default 7). Drafting and candidate selection still run the full trained
eight-token block. Configure the sampling workspace with the requested verification
depth. Shortened multi-GPU verification currently supports batches of one or two;
larger shortened batches are rejected before cache mutation because batch-eight
validation stalled at graph launch. OpenAI serving defaults to depth seven; use
`--dflash-depth 4 --batch-size 2` to serve four-proposal verification. The server
configures warmup, sampling, and scheduler dispatch with that same depth.
The decode regression runner accepts `--depth 3` or `--depth 4` to exercise the
shortened path, including `--long-context`.

Each target chat cache owns a reusable, bounded draft cache. On generator startup
or restart after a batch change, the last 2,047 committed target tokens are replayed
against the earlier target KV to rebuild draft context. Reconditioning runs one
sequence at a time in workspace-sized chunks. Draft context plans are KV-write-only;
they do not run or plan an attention kernel. This also handles retained/shared
target prefixes without storing draft features in the prefix-cache tiers.

Steady-state execution captures drafting, target verification, and draft-context
updates together. Host readbacks are captured on alternate streams with explicit
joins, matching the MTP path. Rejected target
and draft KV are truncated before yielding; old full draft pages are evicted while
RoPE retains absolute target positions. Generator exit drains GPU work and releases
temporary draft pages. The scheduler closes the generator before changing batches.

The drafter remains greedy. For temperature/top-k/top-p target sampling, its
selected tokens are represented as point-mass proposal distributions and verified
by the existing rejection sampler. As with MTP sampling, effective top-k is limited
to 256 and repetition/presence penalties are unsupported. Greedy target verification
is the temperature-zero case. Verification uses prefill kernels, so near-tied BF16
logits may produce a different continuation from token-by-token decode.

The server selects DFlash when `--dflash` (or `--dflash-model-dir`) is present,
including warmup and live scheduler execution. It takes precedence over MTP if
MTP weights were also loaded; DFlash prefill does not run the MTP layer.

```bash
NCCL_P2P_LEVEL=SYS NCCL_TOPO_FILE=/root/chat/vllm/topo_fixed.xml \
npx tsx src/run_model_loader.ts --glm51 --dflash --arena 92 \
  --gpus 0,4,5,7,1,2,3,6 --cp \
  src/openai-server.ts --host 127.0.0.1 --port 8000 \
  --batch-size 2 --max-pages 128 --chunk-size 512
```

`vllm:spec_decode_num_*` metrics include DFlash proposal and acceptance counts.
The decode regression runner checks eager versus captured verification on identical
inputs, acceptance/history bookkeeping, generator restart, and batch compaction:

```bash
curl -N -X POST 'http://127.0.0.1:8099/fork?follow' \
  -H 'content-type: application/json' \
  -d '["src/run_glm51_dflash_decode_test.ts", "--long-context"]'
```

Stop the current executor before switching commands. `--long-context` exercises
the sliding-window boundary; omit it for the short-context test, or use `--eager`
to disable capture. Ordinary-decode prefix agreement is reported diagnostically.
