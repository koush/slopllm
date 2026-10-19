import assert from "node:assert/strict";
import { it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import { DeepseekV41Model } from "../src/deepseek_v41_model";
import { DeepseekV41Cache } from "../src/deepseek_v41_cache";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { freeModelRuntime, loadModelRuntime, parseModelArgs } from "../src/model_cli";
import { GlmOps } from "../src/glm_ops";
import type { ExecutionState } from "../src/execution-workspace";
import type { DeepseekV41Config } from "../src/deepseek_v41_model";
import { PagedKVCache } from "../src/paged_kv";
import { WorkspaceBase } from "../src/workspace";
import { SamplingWorkspace } from "../src/sampling";

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
    const scores = Array.from(values.slice(b * 128, (b + 1) * 128), x => Math.fround(Math.sqrt(Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))))));
    const expected = Array.from({ length: 128 }, (_, i) => i).sort((a, b) => (scores[b] + correction[b]) - (scores[a] + correction[a]) || a - b).slice(0, 3);
    assert.deepEqual(selected.slice(b * 3, b * 3 + 3), expected);
    const total = expected.reduce((sum, i) => sum + scores[i], 0);
    expected.forEach((id, d) => assert(Math.abs(bytes.readDoubleLE((b * 3 + d) * 8) - 1.5 * scores[id] / total) < 1e-12));
  }
});

it("DSpark accepts first-half projection rows without mutating the preceding carry", {
  skip: process.env.SLOPLLM_TEST_DSV41_RESIDENT === "1",
}, () => {
  using ops = new GlmOps(0);
  using model = new DeepseekV41Model(ops, { eosTokenId: 1 } as DeepseekV41Config);
  using cache = new DeepseekV41Cache(model, 2, 2, 128);
  using ws = new ExecutionWorkspace(ops, 2, 16);
  const seed = { ws, slot: 0, batchSize: 2, totalTokens: 2, extras: new Map() } as ExecutionState;
  const first = cache.planCompressor(seed);
  for (const output of first.outputs.values()) { output.pending.fill(99, output.pending.numElements); output.pendingPosition.fill(2, 2); }
  using retained = cache.compressorCarry.get(2)!.value.viewClone();
  const state = { ws, slot: 1, batchSize: 2, totalTokens: 10, seqLens: [5, 5], seqKvLens: [8, 9],
    extras: new Map([["dsv41.retainCompressorRows", true]]) } as unknown as ExecutionState;
  const plan = cache.planCompressor(state);
  const values = Float32Array.from({ length: 10 * 1024 }, (_, i) => Math.floor(i / 1024));
  for (const rows of plan.rows.values()) rows.h2d(Buffer.from(values.buffer));
  cache.commitCompressor(state, [0, 2]);
  for (const holder of cache.compressorCarry.values()) {
    assert.deepEqual(holder.value.pendingPosition.readInt32LEArray(), [2, 6]);
    const bytes = Buffer.alloc(holder.value.pending.bytes); holder.value.pending.d2h(bytes);
    const actual = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
    assert(actual.subarray(0, 1024).every(value => value === 99));
    assert(actual.subarray(1024).every(value => value === 7));
  }
  cache.commitCompressor(state, [4, 0]);
  const bytes = Buffer.alloc(retained.pending.bytes); retained.pending.d2h(bytes);
  assert(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4).every(value => value === 99));
  assert.deepEqual(retained.pendingPosition.readInt32LEArray(), [2, 2]);
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
  } finally { freeModelRuntime(runtime); }
});
