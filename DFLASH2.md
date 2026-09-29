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

`prepareContextPhased()` and `forwardBlockPhased()` are also exposed separately
for repeated drafting against an existing cache. The block K/V is temporary;
the caller must truncate those slots before appending more target-derived context.
This milestone uses normal paged storage with window masking, not a bounded ring
cache or a speculative generation scheduler.

`selectCandidates(hidden, ids, logits, anchors)` applies the checkpoint's learned
selector to externally supplied top-16 candidates. Inputs are BF16 prediction
hidden states, I32 `[batch * 7, 16]` IDs, F32 logits of the same shape, and I32
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
