import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { CaptureManager } from "./capture-manager";
import { ExecutionWorkspace, type ExecutionState } from "./execution-workspace";
import { freeModelRuntime, loadModelRuntime, parseModelArgs } from "./model_cli";
import { Dflash2Model } from "./dflash2_model";
import { SafeTensorFile } from "./safetensors";
import { type Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

function finish<T>(generator: Generator<void, T, void>): T {
  while (true) { const next = generator.next(); if (next.done) return next.value; }
}

function numbers(t: Tensor): Float32Array {
  const bytes = Buffer.alloc(t.bytes);
  t.d2h(bytes);
  const out = new Float32Array(t.numElements);
  if (t.type === "I32") for (let i = 0; i < out.length; i++) out[i] = bytes.readInt32LE(i * 4);
  else if (t.type === "F32") for (let i = 0; i < out.length; i++) out[i] = bytes.readFloatLE(i * 4);
  else {
    const bits = new Uint32Array(out.buffer);
    for (let i = 0; i < bits.length; i++) bits[i] = bytes.readUInt16LE(i * 2) << 16;
  }
  return out;
}

function compare(name: string, actual: Float32Array, expected: Float32Array, relativeLimit = 0.025): void {
  assert.equal(actual.length, expected.length, `${name}: shape mismatch`);
  let squared = 0, energy = 0, maxError = 0, maxValue = 0;
  for (let i = 0; i < actual.length; i++) {
    assert(Number.isFinite(actual[i]) && Number.isFinite(expected[i]), `${name}: nonfinite at ${i}`);
    const e = actual[i] - expected[i];
    squared += e * e; energy += expected[i] * expected[i];
    maxError = Math.max(maxError, Math.abs(e)); maxValue = Math.max(maxValue, Math.abs(expected[i]));
  }
  const relative = Math.sqrt(squared / Math.max(energy, 1e-20));
  console.log(`${name}: relative-RMSE=${relative.toExponential(3)} max-error=${maxError.toExponential(3)}`);
  if (Number.isFinite(relativeLimit)) {
    assert(relative <= relativeLimit && maxError <= 0.12 * maxValue + 0.02, `${name}: reference mismatch`);
  }
}

async function main() {
  const argv = process.argv.slice(2);
  const arg = (name: string, fallback: string) => { const i = argv.indexOf(name); return i < 0 ? fallback : argv[i + 1]; };
  const fixtureDir = arg("--fixtures", "scratchpad/dflash2-reference");
  const iterations = Number(arg("--iterations", "6"));
  const relativeLimit = Number(arg("--relative-tolerance", "0.025"));
  const layerwise = argv.includes("--layerwise");
  assert(Number.isFinite(relativeLimit) && relativeLimit > 0);
  assert(Number.isInteger(iterations) && iterations >= 1);
  const modelArgs = parseModelArgs(argv);
  assert(modelArgs.useDflash2, "Standalone draft validation requires --dflash2");
  const runtime = await loadModelRuntime(modelArgs);
  try {
    const { model, ops } = runtime;
    assert(model instanceof Dflash2Model);
    const meta = JSON.parse(fs.readFileSync(path.join(fixtureDir, "case.json"), "utf8"));
    const lengths: number[] = meta.contextLengths;
    const offsets: number[] = meta.positionOffsets;
    const batch = lengths.length;
    assert.equal(meta.blockSize, model.cfg.blockSize);
    assert.equal(meta.modelHiddenSize, model.cfg.hiddenSize);
    using inputs = new WorkspaceBase(ops);
    const tensors = new Map<string, Tensor>();
    const source = SafeTensorFile.open(path.join(fixtureDir, "inputs.safetensors"));
    try {
      for (const name of source.tensorNames()) {
        const m = source.meta(name);
        const t = inputs.alloc(m.shape, m.dtype);
        t.h2d(source.readTensor(name));
        tensors.set(name, t);
      }
    } finally { source.close(); }
    const expected = new Map<string, Float32Array>();
    const reference = SafeTensorFile.open(path.join(fixtureDir, "expected.safetensors"));
    try {
      for (const name of reference.tensorNames()) {
        const bytes = reference.readTensor(name);
        expected.set(name, Float32Array.from({ length: bytes.length / 4 }, (_, i) => bytes.readFloatLE(i * 4)));
      }
    } finally { reference.close(); }
    const pages = lengths.reduce((n, length) => n + Math.ceil((length + model.cfg.blockSize) / 64), 0);
    using cache = model.createCache(pages, batch);
    using ws = new ExecutionWorkspace(ops, batch, Math.max(lengths.reduce((a, b) => a + b, 0), batch * model.cfg.blockSize));
    cache.reset(batch);
    const contextState = ws.planPrefill(model, batch, lengths, cache);
    const blockState = model.planBlock(ws, cache);
    const positions = (state: ExecutionState) => {
      state.positionIdsH.withPinnedBuffer(buf => {
        let row = 0;
        for (let b = 0; b < batch; b++) for (let t = 0; t < state.seqLens[b]; t++, row++) {
          buf.writeInt32LE(buf.readInt32LE(row * 4) + offsets[b], row * 4);
        }
      });
      state.uploadPlan();
    };
    positions(contextState); positions(blockState);
    const hidden = Array.from({ length: 6 }, (_, i) => tensors.get(`hidden.${i}`)!);
    const embeddings = tensors.get("embeddings")!;
    for (const phases of [1, model.cfg.numHiddenLayers + 2]) {
      const forward = model.forwardDflash2Phased(contextState, blockState, hidden, embeddings);
      try { for (let i = 0; i < phases; i++) assert.equal(forward.next().done, false); }
      finally { forward.return(undefined!); }
      await ops.synchronizeAsync();
      ws.assertClear();
      inputs.assertClear([...tensors.values()]);
    }
    let baseline: Float32Array;
    {
      using tracking = ws.startTracking();
      const checked = new Set<string>();
      const trace = (name: string, t: Tensor) => {
        ops.synchronize();
        compare(name, numbers(t), expected.get(name)!, relativeLimit);
        checked.add(name);
      };
      if (layerwise) {
        finish(model.prepareContextPhased(contextState, hidden, trace));
        const rope = blockState.rotaryEmbedding(model.invFreq);
        using cos = rope.cos; using sin = rope.sin;
        for (let i = 0; i < model.cfg.numHiddenLayers; i++) {
          const layer = finish(model.forwardLayerPhased(blockState, i, tensors.get(`layers.${i}.normalized_input`)!,
            tensors.get(`layers.${i}.residual`)!, cos, sin, trace));
          using normed = layer.normed; using residual = layer.residual;
        }
      }
      using result = finish(model.forwardDflash2Phased(contextState, blockState, hidden, embeddings, layerwise ? undefined : trace));
      await ops.synchronizeAsync();
      if (layerwise) {
        compare("full-forward accumulated BF16 drift (diagnostic)", numbers(result), expected.get("predictions")!, Infinity);
        checked.add("predictions");
      }
      baseline = numbers(result);
      for (let side = 0; side < 2; side++) {
        using out = tensors.get("conv.input")!.dflash2Conv(tensors.get("conv.coefficients")!,
          model.tensors.get("layers.0.attention_conv.base_kernel")!, model.cfg.blockSize, model.cfg.convGroupSize, side);
        compare(`conv.${side}`, numbers(out), expected.get(`conv.${side}`)!, 0.003);
        checked.add(`conv.${side}`);
      }
      const selected = model.selectCandidates(tensors.get("selector.hidden")!, tensors.get("selector.ids")!,
        tensors.get("selector.logits")!, tensors.get("selector.anchors")!);
      using scores = selected.scores; using tokens = selected.tokens;
      compare("selector.scores", numbers(scores), expected.get("selector.scores")!, 0.005);
      compare("selector.tokens", numbers(tokens), expected.get("selector.tokens")!, 0);
      checked.add("selector.scores"); checked.add("selector.tokens");
      assert.equal(checked.size, expected.size);
    }
    using capture = new CaptureManager(ops);
    const stableInputs: Record<string, Tensor> = Object.fromEntries(tensors);
    for (let i = 0; i < iterations; i++) {
      ws.clearTracking();
      const start = performance.now();
      const execution = capture.execute({ states: [contextState, blockState], inputs: stableInputs,
        key: argv.includes("--eager") ? [] : ["dflash2-test", ...lengths, ...offsets] }, retained => {
        const predictions = finish(model.forwardDflash2Phased(contextState, blockState,
          Array.from({ length: 6 }, (_, j) => retained[`hidden.${j}`]), retained.embeddings));
        const selected = model.selectCandidates(retained["selector.hidden"], retained["selector.ids"],
          retained["selector.logits"], retained["selector.anchors"]);
        return { predictions, ...selected };
      });
      using result = execution.result.predictions;
      using scores = execution.result.scores; using tokens = execution.result.tokens;
      await ops.synchronizeAsync();
      console.log(`iteration ${i}: ${(performance.now() - start).toFixed(3)} ms${execution.warmup ? " (warmup/capture)" : ""}`);
      if (!layerwise) compare("predictions", numbers(result), expected.get("predictions")!, relativeLimit);
      compare("eager/replay agreement", numbers(result), baseline!, 0);
      compare("selector scores/replay", numbers(scores), expected.get("selector.scores")!, 0.005);
      compare("selector tokens/replay", numbers(tokens), expected.get("selector.tokens")!, 0);
    }
    console.log(`DFlash2 standalone PASS (${layerwise ? "layerwise reference + full-forward replay" : "end-to-end reference + replay"}, TP${ops.worldSize}, batch=${batch}, contexts=${lengths})`);
  } finally { freeModelRuntime(runtime); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
