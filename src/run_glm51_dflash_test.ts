import assert from "node:assert/strict";
import { CaptureManager } from "./capture-manager";
import { ExecutionWorkspace, type ExecutionState } from "./execution-workspace";
import { Glm51Model, type DflashForwardResult } from "./glm51_model";
import { freeModelRuntime, loadModelRuntime, parseModelArgs } from "./model_cli";
import { Tensor } from "./tensor";
import { DisposableSet } from "./using-holder";

function finish<T>(generator: Generator<void, T, void>): T {
  while (true) { const next = generator.next(); if (next.done) return next.value; }
}

function bytes(tensor: Tensor): Buffer {
  const result = Buffer.alloc(tensor.bytes);
  tensor.d2h(result);
  return result;
}

async function main() {
  const runtime = await loadModelRuntime(parseModelArgs(process.argv.slice(2)));
  try {
    const { model, ops } = runtime;
    assert(model instanceof Glm51Model && model.dflashModel);
    const draft = model.dflashModel;
    const batch = 2;
    using targetCache = model.createChatCache(16, batch, 32);
    using draftCache = draft.createCache(16, batch);
    using ws = new ExecutionWorkspace(ops, batch, 32);
    using capture = new CaptureManager(ops);
    targetCache.reset(batch);
    draftCache.reset(batch);

    for (const decode of [false, true]) {
      ws.clearTracking();
      const input = decode ? [[17], [29]] : [[1, 2, 3], [4, 5, 6, 7, 8]];
      const target: ExecutionState = decode
        ? ws.planDecode(model, batch, targetCache)
        : ws.planPrefill(model, batch, input.map(ids => ids.length), targetCache);
      target.setInput(input);
      const context = ws.planPrefill(draft, batch, input.map(ids => ids.length), draftCache);
      const block = draft.planBlock(ws, draftCache);
      const setBlock = (anchors: number[]) => block.setInput(anchors.map(id =>
        [id, ...Array(draft.cfg.blockSize - 1).fill(draft.cfg.maskTokenId)]));
      setBlock([31, 47]);

      // Independently run the target and draft APIs with retained layer outputs.
      // Compare the combined method to this split execution, including shared head selection.
      let expectedTarget: Buffer, expectedDraft: Buffer, expectedTokens: Buffer;
      {
        using owned = new DisposableSet();
        const features: Tensor[] = [];
        using hidden = finish(model["forwardTargetPhased"](target, undefined, (layer, _normed, residual) => {
          const slot = draft.cfg.targetLayerIds.indexOf(layer);
          if (slot >= 0) { features[slot] = residual.viewClone(); owned.add(features[slot]); }
        }));
        using embeddings = block.embedding(model.tensors.get("model.embed_tokens.weight")!);
        using prediction = finish(draft.forwardDflash2Phased(context, block, features, embeddings));
        using logits = prediction.linear(model.tensors.get("lm_head.weight")!);
        const candidates = logits.topk(draft.cfg.selectorTopK, model.cfg.vocabSize);
        using ids = candidates.indices; using values = candidates.values;
        using anchors = ws.alloc([batch], "I32");
        anchors.h2d(Buffer.from(new Int32Array([31, 47]).buffer));
        const selection = draft.selectCandidates(prediction, ids, values, anchors);
        using scores = selection.scores; using tokens = selection.tokens;
        await ops.synchronizeAsync();
        expectedTarget = bytes(hidden); expectedDraft = bytes(prediction); expectedTokens = bytes(tokens);
      }
      await ops.synchronizeAsync();
      ws.assertClear();

      let phases = 0;
      {
        const forward = model.forwardDflashPhased(target, context, block);
        let next = forward.next();
        while (!next.done) { phases++; next = forward.next(); }
        using owned = new DisposableSet();
        for (const tensor of Object.values(next.value)) owned.add(tensor);
        await ops.synchronizeAsync();
        assert.deepEqual(bytes(next.value.targetHidden), expectedTarget!);
        assert.deepEqual(bytes(next.value.draftHidden), expectedDraft!);
        assert.deepEqual(bytes(next.value.tokens), expectedTokens!);
        assert.deepEqual(next.value.tokens.shape, [batch, 7]);
      }
      // Closing the generator must release layer-output clones and suspended intermediates.
      for (const stop of [1, Math.floor(phases / 2), phases - 1]) {
        const forward = model.forwardDflashPhased(target, context, block);
        try { for (let i = 0; i < stop; i++) assert.equal(forward.next().done, false); }
        finally { forward.return(undefined!); }
        await ops.synchronizeAsync();
        ws.assertClear();
      }

      const states = [target, context, block];
      // Plan-slot input buffers are named metadata, captured through the states.
      const inputs = {};
      const key = ["glm-dflash", decode ? "decode" : "prefill"];
      const snapshots = (result: DflashForwardResult) => Object.fromEntries(
        Object.entries(result).map(([name, tensor]) => [name, bytes(tensor)]));
      let baseline: ReturnType<typeof snapshots>;
      for (let i = 0; i < 7; i++) {
        if (i === 5) {
          // Input addresses stay fixed, but replay must consume the changed anchor IDs.
          setBlock([53, 61]);
          using owned = new DisposableSet();
          const eager = finish(model.forwardDflashPhased(target, context, block));
          for (const tensor of Object.values(eager)) owned.add(tensor);
          await ops.synchronizeAsync();
          baseline = snapshots(eager);
        }
        using owned = new DisposableSet();
        const execution = capture.execute({ states, inputs, key }, (): DflashForwardResult =>
          finish(model.forwardDflashPhased(target, context, block)));
        for (const tensor of Object.values(execution.result)) owned.add(tensor);
        await ops.synchronizeAsync();
        if (i === 0) baseline = snapshots(execution.result);
        else assert.deepEqual(snapshots(execution.result), baseline!);
      }
      assert(capture.isStateCaptured({ states, inputs, key }), "Combined graph was not captured");
      ws.assertClear();
      // Draft blocks are temporary. Keep only the context before the next target step.
      for (const sequence of draftCache.sequences) sequence.truncate(sequence.allocLen - draft.cfg.blockSize);
      console.log(`GLM + DFlash PASS (${decode ? "decode" : "prefill"}, TP${ops.worldSize}, batch=2, one graph)`);
    }
  } finally { freeModelRuntime(runtime); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
