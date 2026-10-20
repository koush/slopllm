import assert from "node:assert/strict";
import { it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import { DeepseekV41Model, LayerHolders } from "../src/deepseek_v41_model";
import { DeepseekV41Cache } from "../src/deepseek_v41_cache";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { freeModelRuntime, loadModelRuntime, parseModelArgs } from "../src/model_cli";
import { GlmOps } from "../src/glm_ops";
import type { ExecutionState } from "../src/execution-workspace";
import type { DeepseekV41Config } from "../src/deepseek_v41_model";
import { PagedKVCache } from "../src/paged_kv";
import { WorkspaceBase } from "../src/workspace";
import { SamplingWorkspace } from "../src/sampling";
import { dsv41Route } from "./deepseek_v41_oracles";

it("DSpark top-3 routing matches independent sqrt-softplus selection and weights", {
  skip: process.env.SLOPLLM_TEST_DSV41_RESIDENT === "1",
}, () => {
  using ops = new GlmOps(0); using ws = new WorkspaceBase(ops);
  using logits = ws.alloc([2, 128], "F32"), bias = ws.alloc([128], "F32");
  const values = Float32Array.from({ length: 256 }, (_, i) => Math.sin(i * 0.17) * 12);
  const correction = Float32Array.from({ length: 128 }, (_, i) => Math.cos(i * 0.31));
  logits.h2d(Buffer.from(values.buffer)); bias.h2d(Buffer.from(correction.buffer));
  const routing = logits.moeRoute({ scoring: "sqrt_softplus", correctionBias: bias, numExpertsPerToken: 3, scalingFactor: 1.5, normalize: true });
  using ids = routing.indices; using stream = routing.normalizedWeightsStream; stream.streamWaitEvent(); using weights = stream.result;
  const selected = ids.readInt32LEArray(), bytes = Buffer.alloc(weights.bytes); weights.d2h(bytes);
  for (let b = 0; b < 2; b++) {
    const expected = dsv41Route(values.subarray(b * 128, (b + 1) * 128), correction);
    assert.deepEqual(selected.slice(b * 3, b * 3 + 3), expected.map(e => e.expert));
    assert.equal(weights.type, "F32");
    expected.forEach((e, d) => assert(Math.abs(bytes.readFloatLE((b * 3 + d) * 4) - e.weight) < 2e-7));
  }
});

it("DSpark accepts first-half projection rows without mutating the preceding carry", {
  skip: process.env.SLOPLLM_TEST_DSV41_RESIDENT === "1",
}, () => {
  using ops = new GlmOps(0);
  using model = new DeepseekV41Model(ops, { eosTokenId: 1 } as DeepseekV41Config);
  using cache = new DeepseekV41Cache(model, 2, 2, 128);
  using ws = new ExecutionWorkspace(ops, 2, 16);
  const seed = { cache, ws, slot: 0, batchSize: 2, totalTokens: 2, extras: new Map() } as unknown as ExecutionState;
  using first = new LayerHolders(seed);
  using retained = cache.compressorCarry.get(2)!.value.viewClone();
  const state = { cache, ws, slot: 1, batchSize: 2, totalTokens: 10, seqLens: [5, 5], seqKvLens: [8, 9],
    extras: new Map([["dsv41.retainCompressorRows", true]]) } as unknown as ExecutionState;
  using holders = new LayerHolders(state);
  // Both steps are planned before the first producer writes its carry.
  for (const output of first.compressorOutputs.values()) { output.pending.fill(99, output.pending.numElements); output.pendingPosition.fill(2, 2); }
  first[Symbol.dispose]();
  const values = Float32Array.from({ length: 10 * 1024 }, (_, i) => Math.floor(i / 1024));
  for (const rows of holders.compressorRows.values()) {
    using raw = ws.alloc([10, 1024], "F32");
    raw.h2d(Buffer.from(values.buffer));
    rows.replace(raw.viewClone());
  }
  cache.commitCompressor(state, holders, [0, 2]);
  for (const holder of cache.compressorCarry.values()) {
    assert.deepEqual(holder.value.pendingPosition.readInt32LEArray(), [2, 6]);
    const bytes = Buffer.alloc(holder.value.pending.bytes); holder.value.pending.d2h(bytes);
    const actual = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
    assert(actual.subarray(0, 1024).every(value => value === 99));
    assert(actual.subarray(1024).every(value => value === 7));
  }
  cache.commitCompressor(state, holders, [4, 0]);
  const rows = [...holders.compressorRows.values()].map(holder => holder.value);
  holders[Symbol.dispose]();
  assert(rows.every(row => row.disposed));
  for (const holder of cache.compressorCarry.values()) {
    assert.deepEqual(holder.value.pendingPosition.readInt32LEArray(), [6, 4]);
  }
  const bytes = Buffer.alloc(retained.pending.bytes); retained.pending.d2h(bytes);
  assert(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4).every(value => value === 99));
  assert.deepEqual(retained.pendingPosition.readInt32LEArray(), [2, 2]);
});

it("DSpark layer holders release per-step rows after captured verification while carry survives", {
  skip: process.env.SLOPLLM_TEST_DSV41_RESIDENT === "1",
}, () => {
  using ops = new GlmOps(0);
  using model = new DeepseekV41Model(ops, { eosTokenId: 1 } as DeepseekV41Config);
  using cache = new DeepseekV41Cache(model, 2, 1, 128);
  using ws = new WorkspaceBase(ops);
  using manager = new CaptureManager(ops);
  using raw = ws.alloc([2, 1024], "F32");
  using positions = ws.alloc([2], "I32");
  using cu = ws.alloc([2], "I32");
  using norm = ws.alloc([512], "BF16");
  cu.h2d(Buffer.from(Int32Array.of(0, 2).buffer)); norm.fill(1, 512);
  const seed = { cache, batchSize: 1, totalTokens: 2, extras: new Map() } as unknown as ExecutionState;
  {
    using holders = new LayerHolders(seed);
    for (const output of holders.compressorOutputs.values()) {
      output.pending.fill(0, output.pending.numElements);
      output.pendingPosition.fill(-1, 1);
    }
  }
  let forwardCalls = 0;
  for (let step = 0; step < 24; ++step) {
    const state = { cache, batchSize: 1, totalTokens: 2, seqLens: [2], seqKvLens: [step * 2 + 2],
      extras: new Map([["dsv41.retainCompressorRows", true]]) } as unknown as ExecutionState;
    using holders = new LayerHolders(state);
    raw.fill(step + 1, raw.numElements);
    positions.h2d(Buffer.from(Int32Array.of(step * 2, step * 2 + 1).buffer));
    const buffers = [...holders.compressorInputs.values(), ...holders.compressorOutputs.values()]
      .flatMap(carry => [carry.pending, carry.pendingPosition]);
    const execution = manager.execute({ states: [], inputs: { raw, positions, cu, norm },
      key: ["holder-carry", JSON.stringify(buffers.map(t => t.memoryRanges()))] }, input => {
      forwardCalls++;
      for (const [layer, output] of holders.compressorOutputs) {
        // Retain an intermediate produced inside the graph, not its input.
        using projected = input.raw.add(input.raw);
        holders.compressorRows.get(layer)!.replace(projected.viewClone());
        const previous = holders.compressorInputs.get(layer)!;
        const result = projected.dsv41Compress(input.positions, input.cu, input.norm, 2,
          previous.pending, previous.pendingPosition, 1e-20, output);
        using latent = result.latent;
        using valid = result.valid;
      }
      return { compressorRows: holders.detachCompressorRows() };
    });
    holders.replaceCompressorRows(execution.result.compressorRows);
    cache.commitCompressor(state, holders, [0]);
    const rows = [...holders.compressorRows.values()].map(holder => holder.value);
    holders[Symbol.dispose]();
    assert(rows.every(row => row.disposed));
    for (const holder of cache.compressorCarry.values()) {
      assert.deepEqual(holder.value.pendingPosition.readInt32LEArray(), [step * 2]);
      const bytes = Buffer.alloc(holder.value.pending.bytes); holder.value.pending.d2h(bytes);
      assert(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4).every(value => value === 2 * (step + 1)));
    }
    ws.assertClear([raw, positions, cu, norm]);
    cache.compressorWorkspace.assertClear([...cache.compressorCarry.values()]
      .flatMap(holder => [holder.value.pending, holder.value.pendingPosition]));
  }
  assert([...manager.captured.values()].some(graph => graph.graphExec !== null));
  assert(forwardCalls < 24, "verification must consume refreshed rows from replay without running the callback");
});

it("DSpark block queries share the historical window and see every draft position", {
  skip: process.env.SLOPLLM_TEST_DSV41_RESIDENT === "1",
}, () => {
  using ops = new GlmOps(0);
  using model = new DeepseekV41Model(ops, { eosTokenId: 1 } as DeepseekV41Config);
  using ws = new ExecutionWorkspace(ops, 2, 16);
  using cache = new PagedKVCache(ops, 1, 512, 3, 10, 2, 64, 512, 64, false, 128, [true, true, true], false, [], "dsv4");
  cache.reset(2); cache.allocAppendPages(0, 140); cache.allocAppendPages(1, 3);
  const state = { ws, batchSize: 2, totalTokens: 10, seqLens: [5, 5] } as ExecutionState;
  const plan = (model as any).planDspark(state, cache, "test", true);
  const actual: number[] = plan.slots.readInt32LEArray();
  for (let b = 0; b < 2; b++) {
    const start = b === 0 ? 140 : 3;
    const expected = Array.from({ length: 192 }, (_, i) => {
      const position = Math.max(0, start - 128) + i;
      return position < start + 5 ? cache.sequences[b].pages[Math.floor(position / 64)].id * 64 + position % 64 : -1;
    });
    for (let d = 0; d < 5; d++) assert.deepEqual(actual.slice((b * 5 + d) * 192, (b * 5 + d + 1) * 192), expected);
  }
});

it("DSpark resident draft/verify supports capture, accepted carry, restart and bounded context", {
  skip: process.env.SLOPLLM_TEST_DSV41_RESIDENT !== "1",
}, async () => {
  const runtime = await loadModelRuntime(parseModelArgs(process.argv.slice(2)));
  try {
    assert(runtime.model instanceof DeepseekV41Model && runtime.model.mtp);
    const model = runtime.model;
    using ws = new ExecutionWorkspace(runtime.ops, 2, 128);
    using manager = new CaptureManager(runtime.ops);
    using cache = model.createChatCache(32, 2, 128) as DeepseekV41Cache;
    const inputs = ["The capital of France is", "Write a short explanation of why the sky is blue."].map(text => model.tokenizer.encode(text));
    const prefill = async () => {
      cache.reset(2);
      let remaining = inputs.map(row => row.slice());
      while (remaining.some(row => row.length)) remaining = (await model.executePrefill(ws, cache, remaining, undefined, 16)).remainingInputIdsList;
    };
    await prefill();
    const run = async (capture: boolean) => {
      const results: { tokens: number[][]; accepted: number[] }[] = [];
      const generator = model.generateMtpDecode(ws, cache, 4, capture ? manager : undefined);
      try {
        for (let step = 0; step < 40; step++) {
          const result = await generator.next(); assert(!result.done);
          results.push({ tokens: result.value.tokens, accepted: result.value.numAccepted });
          for (let b = 0; b < 2; b++) {
            assert(result.value.numAccepted[b] >= 0 && result.value.numAccepted[b] <= 4);
            assert.equal(result.value.tokens[b].length, result.value.numAccepted[b] + 1);
            assert.equal(cache.sequences[b].reportedTokenCount(), cache.sequences[b].allocLen);
            assert(cache.draft!.sequences[b].allocLen <= 196);
            for (const carry of cache.compressorCarry.values()) {
              assert.equal(carry.value.pendingPosition.readInt32LEArray()[b], (cache.sequences[b].allocLen - 1) & ~1);
            }
          }
        }
      } finally { await generator.return(); }
      assert.equal(cache.draft!.sequences.length, 0);
      return results;
    };
    const captured = await run(true);
    assert([...manager.captured.values()].some(graph => graph.graphExec !== null), "DSpark must capture and replay");
    await prefill();
    const eager = await run(false);
    assert.deepEqual(captured, eager, "Captured proposals, accepted tokens and carry must match eager execution");
    assert(captured.some(step => step.accepted.some(count => count > 0)), "Trained DSpark should accept proposals");
    console.log("DSpark accepted", captured.reduce((total, step) => total + step.accepted.reduce((a, b) => a + b, 0), 0), "of", 40 * 2 * 4);
    const restart = model.generateMtpDecode(ws, cache, 4, manager);
    try { assert.equal((await restart.next()).done, false); } finally { await restart.return(); }
    const donor = cache.sequences[0].getTokenIds();
    let reused = [[], cache.prefixMatch(1, donor)];
    assert(cache.sequences[1].allocLen > 0, "DSpark restart should exercise a shared prefix");
    while (reused.some(row => row.length)) reused = (await model.executePrefill(ws, cache, reused, undefined, 32)).remainingInputIdsList;
    assert.deepEqual(cache.sequences[0].getTokenIds(), donor, "Reconditioning a shared recipient must not modify its donor");
    const shared = model.generateMtpDecode(ws, cache, 4, manager);
    try { assert.equal((await shared.next()).value!.tokens.length, 2); } finally { await shared.return(); }
    cache.stageSequence(0, 42);
    for (const depth of [1, 5]) {
      const compacted = model.generateMtpDecode(ws, cache, depth, manager);
      try {
        for (let i = 0; i < 20; i++) {
          const step = await compacted.next(); assert(!step.done);
          assert.equal(step.value.tokens.length, 1);
          assert.equal(step.value.numDraftTokens, depth);
        }
      } finally { await compacted.return(); }
    }
    using sampling = new SamplingWorkspace(runtime.ops, 5, model.cfg.vocabSize, 0, { maxBatchSize: 1, depth: 4 });
    sampling.updateMtpSampler([{ temperature: 0.8, topK: 16, topP: 0.95, repetitionPenalty: 1, presencePenalty: 0, repetitionPenaltyWindow: 0 }]);
    const stochastic = model.generateMtpDecode(ws, cache, 4, manager, sampling);
    try {
      for (let i = 0; i < 20; i++) {
        const step = await stochastic.next(); assert(!step.done);
        assert(step.value.numAccepted[0] >= 0 && step.value.numAccepted[0] <= 4);
        assert(step.value.tokens[0].every(token => token >= 0 && token < model.cfg.vocabSize));
      }
    } finally { await stochastic.return(); }

    // Both ratio-one and ratio-two indexers must actually discard keys. A
    // short prompt alone cannot detect score-dependent verification errors.
    using longCache = model.createChatCache(64, 1, 4096) as DeepseekV41Cache;
    using longWs = new ExecutionWorkspace(runtime.ops, 1, 128);
    using longManager = new CaptureManager(runtime.ops);
    const longPrompt = model.tokenizer.encode("A short note about geography.\n".repeat(300)).slice(0, 1100);
    assert.equal(longPrompt.length, 1100);
    const longRun = async (captured: boolean) => {
      longCache.reset(1);
      let remaining = [longPrompt.slice()];
      while (remaining[0].length) remaining = (await model.executePrefill(longWs, longCache, remaining, undefined, 128)).remainingInputIdsList;
      const output: { tokens: number[][]; accepted: number[] }[] = [];
      const generator = model.generateMtpDecode(longWs, longCache, 3, captured ? longManager : undefined);
      try {
        for (let step = 0; step < 20; ++step) {
          const result = await generator.next(); assert(!result.done);
          output.push({ tokens: result.value.tokens, accepted: result.value.numAccepted });
        }
      } finally { await generator.return(); }
      return output;
    };
    const longCaptured = await longRun(true);
    assert([...longManager.captured.values()].some(graph => graph.graphExec !== null));
    assert.deepEqual(await longRun(false), longCaptured, "Captured verification must match eager after compressed index top-k becomes active");
  } finally { freeModelRuntime(runtime); }
});
