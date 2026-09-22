import assert from "node:assert/strict";
import path from "node:path";
import { after, afterEach, before, beforeEach, describe, it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import type { ChatCache, ChatModel, TokenSelector, SamplingParams } from "../src/chat_model";
import type { DeviceOps } from "../src/device_ops";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { ParallelOps } from "../src/parallel_ops";
import { Tensor } from "../src/tensor";
import { PAGE_SIZE } from "../src/paged_kv";
import { SamplingWorkspace } from "../src/sampling";
import { UsingHolder } from "../src/using-holder";
import { WorkspaceBase } from "../src/workspace";
import { generateBatchTokens, generateStream } from "../src/run_qwen3_unified";
import { createAsyncQueue } from "@scrypted/deferred";
import { GenerationScheduler, type GenerationRequest, type ServerMetrics } from "../src/generation-scheduler";

const SMALL_MODEL_DIR = path.resolve(
  __dirname,
  "../tests/python/test_models/glm51_small/glm51_small_bf16",
);

function makeLongPrompt(length: number, prefix: number[] = [1, 2, 3]): number[] {
  return [...prefix, ...Array.from({ length: length - prefix.length }, (_, i) => 100 + i)];
}

function prefillTargets(cache: ChatCache): number[] {
  return cache.getPagedKV().sequences.map(sequence => {
    assert.notEqual(sequence.targetToken, undefined);
    return sequence.targetToken!;
  });
}

function chunkedPrefill(model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache, inputIds: number[], chunkSizes: number[]): number[] {
  let offset = 0;
  let logits: ReturnType<Tensor["argmax"]> | null = null;
  for (let i = 0; i < chunkSizes.length; i++) {
    const chunk = inputIds.slice(offset, offset + chunkSizes[i]);
    offset += chunkSizes[i];
    const state = ws.planPrefill(model, 1, [chunk.length], cache);
    state.setInput([chunk]);
    const hiddenStates = model.forward(state);
    if (i < chunkSizes.length - 1) {
      hiddenStates[Symbol.dispose]();
    } else {
      logits = state.computeLogits(hiddenStates, model);
      hiddenStates[Symbol.dispose]();
    }
  }
  using finalLogits = logits!;
  using argmaxOut = finalLogits.argmax();
  return argmaxOut.readInt32LEArray();
}

async function assertBatchedPrefillMatches(model: Glm51Model, glm: DeviceOps, ws: ExecutionWorkspace): Promise<void> {
  const prompts = [[1, 2, 3, 4, 5], [11, 12], [21, 22, 23, 24]];
  const run = async (chunked: boolean) => {
    using cache = model.createChatCache(32, 3);
    cache.reset(3);
    if (chunked) {
      assert.throws(() => model["planPhasedPrefill"](ws, cache, prompts), /batch size 1/);
      const plan = model.planChunkedPrefill(ws, cache, prompts);
      assert.deepEqual(plan.states.map(state => state.seqLens), [[5, 2, 4]]);
      for (const _ of plan.generator) {}
      await glm.synchronizeAsync();
      plan.reportTokens();
      ws.assertClear();
      ws.clearTracking();
    } else {
      let targets: number[];
      {
        const state = ws.planPrefill(model, prompts.length, prompts.map(ids => ids.length), cache);
        state.setInput(prompts);
        using hidden = model.forward(state);
        using logits = state.computeLogits(hidden, model);
        using selected = logits.argmax();
        targets = selected.readInt32LEArray();
      }
      await glm.synchronizeAsync();
      prompts.forEach((ids, batch) => cache.reportTokens(batch, ids, targets[batch]));
      ws.assertClear();
      ws.clearTracking();
    }
    assert.deepEqual(cache.getPagedKV().sequences.map(sequence => sequence.getTokenIds()), prompts);
    const output: number[][][] = [];
    for await (const step of model.generateDecode(ws, cache)) {
      output.push(step.tokens);
      if (output.length === 2) break;
    }
    ws.assertClear();
    return output;
  };
  assert.deepEqual(await run(true), await run(false));
}

async function prefillChunks(model: Glm51Model, ws: ExecutionWorkspace, cache: ChatCache,
  input: number[][], manager: CaptureManager, samplingPolicy?: TokenSelector): Promise<{ targetTokens: number[] }> {
  let remaining = input;
  while (remaining.some(ids => ids.length)) {
    remaining = (await model.executePrefill(ws, cache, remaining, manager, ws.maxSeqLen, samplingPolicy)).remainingInputIdsList;
  }
  return { targetTokens: cache.getPagedKV().sequences.map(sequence => sequence.targetToken!) };
}

describe("GLM-5.1 small model smoke test", () => {
  let glm: GlmOps;
  let model: Glm51Model;
  let ws: ExecutionWorkspace;

  beforeEach(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Glm51Model.fromPretrained(glm, SMALL_MODEL_DIR);
    ws = new ExecutionWorkspace(glm, 3, 128);
  });

  afterEach(() => {
    ws.free();
    model.free();
    glm.free();
  });

  it("loads model and checks config", () => {
    const cfg = model.cfg;
    assert.equal(cfg.numHiddenLayers, 8, "numHiddenLayers");
    assert.equal(cfg.hiddenSize, 1024, "hiddenSize");
    assert.equal(cfg.numAttentionHeads, 8, "numAttentionHeads");
    assert.equal(cfg.nRoutedExperts, 8, "nRoutedExperts");
    assert.equal(cfg.numExpertsPerTok, 4, "numExpertsPerTok");
    assert.equal(cfg.vocabSize, 154880, "vocabSize");
    assert.equal(cfg.kvLoraRank, 512, "kvLoraRank");
    assert.equal(cfg.qkRopeHeadDim, 64, "qkRopeHeadDim");
    assert.ok(cfg.tieWordEmbeddings, "tieWordEmbeddings");
  });

  it("forward pass produces logits of correct shape", () => {
    const cache = model.createChatCache(32);
    const inputIds = [1, 2, 3, 4];
    cache.reset(1);

    const firstTokens = ws.forwardEagerPrefill(model, [inputIds], cache);
    assert.equal(firstTokens.length, 1, "should produce 1 token for batch=1");
    assert.ok(
      firstTokens[0] >= 0 && firstTokens[0] < model.cfg.vocabSize,
      `token ${firstTokens[0]} out of vocab range [0, ${model.cfg.vocabSize})`,
    );

    cache.free();
  });

  it("planChunkedPrefill only interleaves single-sequence prompts", async () => {
    using prefillWs = new ExecutionWorkspace(glm, 2, 4097);
    using cache = model.createChatCache(128, 2, 4097);
    for (const lengths of [[4097], [2049, 2048], [4095, 1], [1, 4095], [2048, 2048], [4096, 0], [0, 4096]]) {
      cache.reset(lengths.length);
      const plan = model.planChunkedPrefill(prefillWs, cache, lengths.map(length => makeLongPrompt(length).slice(0, length)));
      assert.equal(plan.states.length, lengths.length === 1 ? 2 : 1);
      if (lengths.length === 1) {
        assert.equal(plan.states[0].totalTokens, Math.floor(lengths[0] / 2));
        assert.notEqual(plan.states[0].inputIdsBuf.data, plan.states[1].inputIdsBuf.data);
      } else {
        assert.deepEqual(plan.states[0].seqLens, lengths);
      }
      assert.deepEqual(cache.getPagedKV().sequences.map(sequence => sequence.allocLen), lengths,
        "planning appends each prompt exactly once");
      plan.generator.return([]);
      await glm.synchronizeAsync();
      prefillWs.clearTracking();
    }
    // Exercise both paths with small inputs alongside resident model weights.
    for (const lengths of [[9], [5, 4], [1, 8], [8, 1], [4, 5]]) {
      const inputs = lengths.map((length, index) => makeLongPrompt(length, [index + 1, 2, 3]).slice(0, length));
      const tail = inputs.map((_, index) => [7 + index]);
      cache.reset(inputs.length);
      prefillWs.forwardEagerPrefill(model, inputs, cache);
      prefillWs.clearTracking();
      const expected = prefillWs.forwardEagerPrefill(model, tail, cache);
      prefillWs.clearTracking();

      cache.reset(inputs.length);
      const plan = inputs.length === 1 ? model["planPhasedPrefill"](prefillWs, cache, inputs)
        : model.planChunkedPrefill(prefillWs, cache, inputs);
      assert.equal(plan.states.length, inputs.length === 1 ? 2 : 1);
      assert.deepEqual(cache.getPagedKV().sequences.map(sequence => sequence.allocLen), lengths,
        "planning appends each prompt exactly once");
      let phases = 0;
      while (true) {
        const step = plan.generator.next();
        if (step.done) {
          assert.equal(step.value, undefined);
          await glm.synchronizeAsync();
          break;
        }
        phases++;
      }
      assert.ok(phases > 0);
      plan.reportTokens();
      prefillWs.assertClear();
      prefillWs.clearTracking();

      await model.executePrefill(prefillWs, cache, tail);
      assert.deepEqual(prefillTargets(cache), expected);
      prefillWs.assertClear();
      prefillWs.clearTracking();
    }
  });

  it("executePrefill stays eager with a capture manager and reports plain inputs", async () => {
    using prefillWs = new ExecutionWorkspace(glm, 1, 128);
    using cache = model.createChatCache(32, 1);
    using manager = new CaptureManager(glm);
    for (const graphs of [false, true]) {
      for (let run = 0; run < 7; run++) {
        cache.reset(1);
        const input = [1, 2, 3, run + 4];
        const result = await model.executePrefill(prefillWs, cache, [input], graphs ? manager : undefined);
        assert.equal("nextTokens" in result, false);
        assert.deepEqual(cache.getPagedKV().sequences[0].getTokenIds(), input);
        assert.equal(result.warmup, false);
        assert.equal(manager.captured.size, 0);
      }
    }
  });

  it("plain generator consumers publish the prefill target exactly once", async () => {
    using cache = model.createChatCache(32, 1);
    const prompt = [1, 2, 3, 4];
    cache.reset(1);
    let current = ws.forwardEagerPrefill(model, [prompt], cache)[0];
    cache.reportTokens(0, prompt, current);
    const expected = [current];
    for (let i = 1; i < 6; i++) {
      const next = ws.forwardEagerDecode(model, [current], cache)[0];
      cache.reportTokens(0, [current], next);
      expected.push(next);
      current = next;
    }
    for (const graphs of [false, true]) {
      cache.reset(1);
      const output: number[] = [];
      for await (const token of generateStream(model, ws, glm, cache, prompt, 6, new Set(), undefined,
        graphs ? { graphExec: null, warmupRemaining: 3 } : undefined)) {
        output.push(token);
      }
      assert.deepEqual(output, expected);
      assert.deepEqual(cache.getPagedKV().sequences[0].getTokenIds(), [...prompt, ...output.slice(0, -1)]);
      assert.equal(cache.getPagedKV().sequences[0].targetToken, output.at(-1));
    }
    ws.assertClear();
  });

  it("plain chunked prefill returns selected inputs and unconsumed rows", async () => {
    using cache = model.createChatCache(32, 2);
    using prefillWs = new ExecutionWorkspace(glm, 2, 128);
    cache.reset(2);
    const inputs = [[1, 2, 3, 4, 5], [6, 7, 8]];
    const first = await model.executePrefill(prefillWs, cache, inputs, undefined, 4);
    assert.deepEqual(first.prefillInputIdsList, [[1, 2, 3, 4], []]);
    assert.deepEqual(first.remainingInputIdsList, [[5], [6, 7, 8]]);
    assert.deepEqual(cache.getPagedKV().sequences.map(s => s.getTokenIds()), [[1, 2, 3, 4], []]);
    const last = await model.executePrefill(prefillWs, cache, first.remainingInputIdsList, undefined, 4);
    assert.deepEqual(last.prefillInputIdsList, [[5], [6, 7, 8]]);
    assert.deepEqual(last.remainingInputIdsList, [[], []]);
    assert.deepEqual(cache.getPagedKV().sequences.map(s => s.getTokenIds()), inputs);
  });

  for (const phased of [false, true]) it(`prefill releases tensors when closed after its first yield (phased=${phased})`, async () => {
    using prefillWs = new ExecutionWorkspace(glm, 1, 128);
    using cache = model.createChatCache(32, 1, 128);
    cache.reset(1);
    const plan = phased
      ? model["planPhasedPrefill"](prefillWs, cache, [makeLongPrompt(9)])
      : model.planChunkedPrefill(prefillWs, cache, [makeLongPrompt(9)]);
    assert.equal(plan.generator.next().done, false);
    plan.generator.return([]);
    await glm.synchronizeAsync();
    prefillWs.assertClear();
  });

  it("decode step produces valid token after prefill", () => {
    const cache = model.createChatCache(32);
    const inputIds = [1, 2, 3];
    cache.reset(1);

    const firstToken = ws.forwardEagerPrefill(model, [inputIds], cache)[0];
    assert.ok(firstToken >= 0 && firstToken < model.cfg.vocabSize, "prefill token out of range");

    const secondToken = ws.forwardEagerDecode(model, [firstToken], cache)[0];
    assert.ok(secondToken >= 0 && secondToken < model.cfg.vocabSize, "decode token out of range");

    cache.free();
  });

  it("generates multiple tokens without crashing", () => {
    const cache = model.createChatCache(32);
    const inputIds = [1, 2, 3, 4, 5];
    cache.reset(1);

    const eosIds = model.eosIds;
    const maxNewTokens = 16;
    let token = ws.forwardEagerPrefill(model, [inputIds], cache)[0];
    const generated: number[] = [token];

    for (let i = 1; i < maxNewTokens && !eosIds.has(token); i++) {
      token = ws.forwardEagerDecode(model, [token], cache)[0];
      generated.push(token);
    }

    assert.ok(generated.length >= 2, `expected at least 2 tokens, got ${generated.length}`);
    for (const t of generated) {
      assert.ok(t >= 0 && t < model.cfg.vocabSize, `token ${t} out of range`);
    }

    cache.free();
  });

  it("handles batch size > 1", () => {
    const cache = model.createChatCache(64, 2);
    const inputIds1 = [1, 2, 3];
    const inputIds2 = [4, 5, 6];
    cache.reset(2);

    const tokens = ws.forwardEagerPrefill(model, [inputIds1, inputIds2], cache);
    assert.equal(tokens.length, 2, "should produce 2 tokens for batch=2");
    for (const t of tokens) {
      assert.ok(t >= 0 && t < model.cfg.vocabSize, `token ${t} out of range`);
    }

    cache.free();
  });

  it("chunked prefill: two halves", () => {
    using cache1 = model.createChatCache(32);
    using cache2 = model.createChatCache(32);
    const fullPrompt = [1, 2, 3, 4, 5, 6, 7];
    const mid = Math.floor(fullPrompt.length / 2);

    cache1.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], cache1);

    cache2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, cache2, fullPrompt, [mid, fullPrompt.length - mid]);

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Chunked prefill mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
  });

  it.skip("chunked prefill: uneven split", () => {
    using cache1 = model.createChatCache(32);
    using cache2 = model.createChatCache(32);
    const fullPrompt = [1, 2, 3, 4, 5, 6, 7, 8];

    cache1.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], cache1);

    cache2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, cache2, fullPrompt, [3, fullPrompt.length - 3]);

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Chunked prefill mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
  });

  it("chunked prefill + decode", () => {
    using cache1 = model.createChatCache(32);
    using cache2 = model.createChatCache(32);
    const fullPrompt = [1, 2, 3, 4, 5, 6, 7];
    const mid = Math.floor(fullPrompt.length / 2);

    cache1.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], cache1);
    const fullDecode = ws.forwardEagerDecode(model, [fullTokens[0]], cache1)[0];

    cache2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, cache2, fullPrompt, [mid, fullPrompt.length - mid]);
    const chunkedDecode = ws.forwardEagerDecode(model, [chunkedTokens[0]], cache2)[0];

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Chunked prefill token mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
    assert.equal(chunkedDecode, fullDecode,
      `Chunked decode token mismatch: chunked=${chunkedDecode}, full=${fullDecode}`);
  });

  it("batched chunked prefill matches an independent sequential forward", async () => {
    await assertBatchedPrefillMatches(model, glm, ws);
  });
});

describe("GLM-5.1 small model with context parallelism", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let glm2: GlmOps;
  let po: ParallelOps;
  let modelCp: Glm51Model;
  let wsCp: ExecutionWorkspace;

  const MAX_BATCH = 3;
  const MAX_SEQ_LEN = 128;

  before(async () => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    glm2 = new GlmOps(2);
    po = new ParallelOps([glm1, glm2]);
    modelCp = await Glm51Model.fromPretrained(po, SMALL_MODEL_DIR, true);
    wsCp = new ExecutionWorkspace(po, MAX_BATCH, MAX_SEQ_LEN);
  });

  after(() => {
    wsCp.free();
    modelCp.free();
    po.free();
    glm0.free(); glm1.free(); glm2.free();
  });

  it("prefill produces valid token", () => {
    using cache = modelCp.createChatCache(32);
    cache.reset(1);
    const inputIds = [1, 2, 3, 4, 5];
    const tokens = wsCp.forwardEagerPrefill(modelCp, [inputIds], cache);
    assert.equal(tokens.length, 1, "should produce 1 token for batch=1");
    assert.ok(tokens[0] >= 0 && tokens[0] < modelCp.cfg.vocabSize,
      `token ${tokens[0]} out of vocab range [0, ${modelCp.cfg.vocabSize})`);
  });

  it("prefill + decode produces valid tokens", () => {
    using cache = modelCp.createChatCache(32);
    cache.reset(1);
    const inputIds = [1, 2, 3];
    const firstToken = wsCp.forwardEagerPrefill(modelCp, [inputIds], cache)[0];
    assert.ok(firstToken >= 0 && firstToken < modelCp.cfg.vocabSize, "prefill token out of range");

    const secondToken = wsCp.forwardEagerDecode(modelCp, [firstToken], cache)[0];
    assert.ok(secondToken >= 0 && secondToken < modelCp.cfg.vocabSize, "decode token out of range");
  });

  it("chunked prefill + decode", () => {
    using cache1 = modelCp.createChatCache(32);
    using cache2 = modelCp.createChatCache(32);
    const fullPrompt = [1, 2, 3, 4, 5, 6, 7];
    const mid = Math.floor(fullPrompt.length / 2);

    cache1.reset(1);
    const fullTokens = wsCp.forwardEagerPrefill(modelCp, [fullPrompt], cache1);
    const fullDecode = wsCp.forwardEagerDecode(modelCp, [fullTokens[0]], cache1)[0];

    cache2.reset(1);
    const chunkedTokens = chunkedPrefill(modelCp, wsCp, cache2, fullPrompt, [mid, fullPrompt.length - mid]);
    const chunkedDecode = wsCp.forwardEagerDecode(modelCp, [chunkedTokens[0]], cache2)[0];

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Chunked prefill mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
    assert.equal(chunkedDecode, fullDecode,
      `Chunked decode mismatch: chunked=${chunkedDecode}, full=${fullDecode}`);
  });

  it("batched chunked prefill matches an independent sequential forward", async () => {
    await assertBatchedPrefillMatches(modelCp, po, wsCp);
  });
});

describe("GLM-5.1 small model phased MTP prefill", () => {
  let glm: GlmOps;
  let model: Glm51Model;
  let ws: ExecutionWorkspace;
  let captureManager: CaptureManager;

  // Releasing the device backend between cases also releases pooled scratch;
  // the resident loader leaves only a small amount of GPU memory for tests.
  beforeEach(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Glm51Model.fromPretrained(glm, SMALL_MODEL_DIR, false, true);
    ws = new ExecutionWorkspace(glm, 1, 128);
    captureManager = new CaptureManager(glm);
    captureManager.disabled = true;
  });

  afterEach(() => {
    captureManager[Symbol.dispose]();
    ws.free();
    model.free();
    glm.free();
  });

  it("splits a single sequence into balanced non-empty halves", () => {
    const input = [[1, 2, 3, 4, 5, 6, 7]];
    using cache = model.createChatCache(32, 1);
    cache.reset(1);
    const plan = model["planPhasedPrefill"](ws, cache, input);
    assert.deepEqual(plan.states.map(state => state.seqLens), [[3], [4]]);
    assert.deepEqual(plan.prefillInputIdsList, input);
    plan.generator.return([]);
    ws.clearTracking();
  });

  it("does not report an unstarted chunked MTP plan", () => {
    using cache = model.createChatCache(32, 1);
    cache.reset(1);
    const sequence = cache.getPagedKV().sequences[0];

    const unstarted = model.planChunkedPrefill(ws, cache, [[1, 2, 3, 4]]);
    assert.equal(sequence.allocLen, 4);
    unstarted.generator.return([]);
    assert.deepEqual(sequence.getTokenIds(), []);
    cache.reset(1);
    assert.equal(cache.getPagedKV().sequences[0].allocLen, 0);
    ws.resetPlanSlots();
  });

  for (const graphs of [false, true]) it(`stateful decode generator matches batched decode and restarts (graphs=${graphs})`, async () => {
    ws.free();
    ws = new ExecutionWorkspace(glm, 2, 128);
    const prompts = [makeLongPrompt(63), makeLongPrompt(61, [11, 12, 13])];
    const run = async (generator: boolean) => {
      using cache = model.createChatCache(32, 2);
      cache.reset(2);
      using manager = new CaptureManager(glm);
      manager.disabled = !graphs;
      const state = ws.planPrefill(model, 2, prompts.map(ids => ids.length), cache);
      state.setInput(prompts);
      let current: number[];
      {
        using hidden = model.forwardModel(state);
        using logits = state.computeLogits(hidden, model);
        using selected = logits.argmax();
        current = selected.readInt32LEArray();
      }
      prompts.forEach((ids, batch) => cache.reportTokens(batch, ids, current[batch]));
      ws.clearTracking();
      const output: number[][] = [];
      const histories = prompts.map(ids => [...ids]);
      let replayed = false;
      const record = (tokens: number[]) => {
        output.push([...tokens]);
        if (!generator) {
          current.forEach((token, batch) => cache.reportTokens(batch, [token]));
        }
        current.forEach((token, batch) => {
          histories[batch].push(token);
          assert.deepEqual(cache.getPagedKV().sequences[batch].getTokenIds(), histories[batch]);
        });
        current = [...tokens];
      };
      if (generator) {
        while (output.length < 12) {
          for await (const step of model.generateDecode(ws, cache, graphs ? manager : undefined,
            graphs ? { selectTarget: logits => logits.argmax(), captureKey: "test-selector" } : undefined)) {
            assert.ok(step.tokens.every(tokens => tokens.length === 1));
            assert.deepEqual(step.numAccepted, [0, 0]);
            assert.equal(step.numDraftTokens, 0);
            assert.deepEqual(step.draft, {
              targetTokens: step.tokens.map(tokens => tokens[0]), treeTokens: [[], []], topks: [],
            });
            record(step.tokens.map(tokens => tokens[0]));
            replayed ||= !step.warmup;
            // Mutating the returned array must not change the next decode input.
            step.tokens.forEach(tokens => tokens.fill(0));
            step.draft.targetTokens.fill(0);
            if (output.length % 6 === 0) break;
          }
          ws.assertClear();
          assert.equal(ws.staged.size, 0);
        }
        assert.ok(replayed);
      } else {
        for (let i = 0; i < 12; i++) {
          const decode = ws.planDecode(model, 2, cache, graphs);
          decode.setInput([current]);
          {
            using hidden = model.forwardModel(decode);
            using logits = decode.computeLogits(hidden, model);
            using selected = logits.argmax();
            record(selected.readInt32LEArray());
          }
          ws.clearTracking();
        }
      }
      cache.getPagedKV().sequences.forEach((sequence, batch) => {
        assert.equal(sequence.allocLen, prompts[batch].length + 12);
        assert.equal(sequence.getTokenIds().length, sequence.allocLen);
      });
      return output;
    };
    assert.deepEqual(await run(true), await run(false));
  });

  it("stateful decode generator preserves caller tensors on entry, resume, and break", async () => {
    using cache = model.createChatCache(32, 1);
    cache.reset(1);
    let targets = chunkedPrefill(model, ws, cache, [1, 2, 3], [3]);
    cache.reportTokens(0, [1, 2, 3], targets[0]);
    ws.assertClear();
    ws.clearTracking();
    {
      using callerTensor = ws.alloc([1], "I32");
      const generator = model.generateDecode(ws, cache);
      await assert.rejects(generator.next(), /assertClear/);
      assert.equal(callerTensor.disposed, false);
    }
    const generator = model.generateDecode(ws, cache);
    const first = await generator.next();
    assert.equal(first.done, false);
    targets = first.value!.tokens.map(tokens => tokens[0]);
    {
      using callerTensor = ws.alloc([1], "I32");
      await assert.rejects(generator.next(), /assertClear/);
      assert.equal(callerTensor.disposed, false);
    }
    let callerTensor: Tensor | undefined;
    try {
      for await (const step of model.generateDecode(ws, cache)) {
        callerTensor = ws.alloc([1], "I32");
        break;
      }
      assert.ok(callerTensor);
      assert.equal(callerTensor.disposed, false);
    } finally {
      callerTensor?.[Symbol.dispose]();
    }
    ws.assertClear();
  });

  it("generator consumers preserve streaming and batched token budgets", async () => {
    using cache = model.createChatCache(32, 1);
    const prompt = [1, 2, 3, 4];
    const expected = await generateBatchTokens(model, ws, cache, [prompt], 12, new Set());
    assert.equal(expected[0].length, 12);
    for (const graphs of [false, true]) {
      cache.reset(1);
      const tokens: number[] = [];
      for await (const token of generateStream(model, ws, glm, cache, prompt, 12, new Set(), undefined,
        graphs ? { graphExec: null, warmupRemaining: 3 } : undefined)) tokens.push(token);
      assert.deepEqual(tokens, expected[0]);
      assert.equal(cache.getPagedKV().sequences[0].allocLen, prompt.length + tokens.length - 1);
      ws.assertClear();
    }
    cache.reset(1);
    const single: number[] = [];
    for await (const token of generateStream(model, ws, glm, cache, prompt, 1, new Set(), undefined)) single.push(token);
    assert.deepEqual(single, expected[0].slice(0, 1));
    ws.assertClear();
  });

  it("generator consumers chunk ragged prompts and bootstrap plain and MTP streams", async () => {
    ws.free();
    ws = new ExecutionWorkspace(glm, 2, 8);
    using cache = model.createChatCache(32, 2);
    const prompts = [Array.from({ length: 21 }, (_, i) => i + 1), [7, 8, 9, 10]];
    const batch = await generateBatchTokens(model, ws, cache, prompts, 3, new Set());
    assert.deepEqual(batch.map(tokens => tokens.length), [3, 3]);
    for (const [index, sequence] of cache.getPagedKV().sequences.entries()) {
      assert.deepEqual(sequence.getTokenIds(), [...prompts[index], ...batch[index].slice(0, -1)]);
    }
    for (const mtp of [false, true]) {
      for (const graphs of [false, true]) {
        for (const budget of [0, 1, 9]) {
          cache.reset(1);
          const tokens: number[] = [];
          for await (const token of generateStream(model, ws, glm, cache, prompts[0], budget, new Set(), undefined,
            graphs ? { graphExec: null, warmupRemaining: 3 } : undefined, undefined, mtp, [1, 1])) {
            tokens.push(token);
          }
          assert.equal(tokens.length, budget);
          const history = cache.getPagedKV().sequences[0].getTokenIds();
          assert.deepEqual(history.slice(0, budget ? prompts[0].length + budget - 1 : 0),
            budget ? [...prompts[0], ...tokens.slice(0, -1)] : []);
          ws.assertClear();
        }
      }
    }
  });

  for (const mtp of [false, true]) it(`queued scheduler admits, drains, cancels, and returns to idle (mtp=${mtp})`, async () => {
    ws.free();
    if (!mtp) {
      // Exercise the actual non-MTP bootstrap, releasing pooled GPU memory before reload.
      captureManager[Symbol.dispose]();
      model.free();
      glm.free();
      glm = new GlmOps(parseInt(process.env.GLM_GPU ?? "0", 10));
      model = await Glm51Model.fromPretrained(glm, SMALL_MODEL_DIR, false, false);
      captureManager = new CaptureManager(glm);
    }
    captureManager.disabled = false;
    ws = new ExecutionWorkspace(glm, 2, 128);
    using cache = model.createChatCache(32, 2);
    using sampler = new SamplingWorkspace(glm, 6, model.cfg.vocabSize, 8);
    const requests = createAsyncQueue<GenerationRequest>();
    const metrics: ServerMetrics = {
      runningRequests: 0, generationTokensTotal: 0, promptTokensTotal: 0,
      specDecodeNumDraftsTotal: 0, specDecodeNumDraftTokensTotal: 0, specDecodeNumAcceptedTokensTotal: 0,
      mtpPhaseSeconds: new Map(), mtpPhaseCount: new Map(), requestSuccessTotal: 0,
      prefillTimeSecondsCount: 0, prefillTimeSecondsSum: 0,
    };
    const makeRequest = (id: string, maxTokens: number): GenerationRequest => ({
      id, maxTokens, inputIds: [1, 2, 3, 4], tokens: createAsyncQueue<number>(), generatedTokenCount: 0,
      finishReason: "stop", promptTokenCount: 4, cachedTokenCount: 0,
      prefillTokenCount: 4, prefillSeconds: 0,
      samplingParams: { temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, presencePenalty: 0, repetitionPenaltyWindow: 8 },
    });
    const scheduler = new GenerationScheduler({ requests, model, ws, cache, captureManager,
      samplingWorkspace: sampler, metrics, maxBatchSize: 2, chunkSize: 8,
      decodeLatencyMs: 0, topks: mtp ? [1, 1] : undefined });
    let stopped = false;
    const running = scheduler.run().finally(() => { stopped = true; });
    const long = makeRequest("long", 24);
    const short = makeRequest("short", 3);
    const cancelled = makeRequest("cancelled", 100);
    const abandoned = makeRequest("abandoned", 100);
    abandoned.tokens.end();
    let longSequence: ReturnType<typeof cache.getPagedKV>["sequences"][number] | undefined;
    try {
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(stopped, false, "scheduler waits for its first request");
      requests.submit(long);
      const output = await Promise.all([
        (async () => {
          const tokens: number[] = [];
          for await (const token of long.tokens.queue) {
            tokens.push(token);
            if (tokens.length === 1) {
              longSequence = cache.getPagedKV().sequences[0];
              requests.submit(abandoned);
              requests.submit(short);
              requests.submit(cancelled);
            }
          }
          return tokens;
        })(),
        (async () => {
          const tokens: number[] = [];
          for await (const token of short.tokens.queue) tokens.push(token);
          return tokens;
        })(),
        (async () => {
          for await (const token of cancelled.tokens.queue) return [token];
          return [];
        })(),
      ]);
      assert.deepEqual(output.map(tokens => tokens.length), [24, 3, 1]);
      assert.deepEqual(longSequence!.getTokenIds().slice(0, 27), [1, 2, 3, 4, ...output[0].slice(0, -1)],
        "batch restarts commit emitted tokens exactly once without skipping or duplicating history");
      assert.equal(abandoned.generatedTokenCount, 0);
      assert.equal(cancelled.tokens.submit(0), false);
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(stopped, false, "scheduler waits again after draining the batch");
      const next = makeRequest("next", 1);
      requests.submit(next);
      const final: number[] = [];
      for await (const token of next.tokens.queue) final.push(token);
      assert.equal(final.length, 1);
      const planPrefill = model.planChunkedPrefill;
      try {
        for (const selectedToken of [70, 71]) {
          model.planChunkedPrefill = function (workspace, chatCache, inputs, chunkSize, policy) {
            return planPrefill.call(this, workspace, chatCache, inputs, chunkSize, {
              selectTarget: logits => {
                const selected = policy ? policy.selectTarget(logits) : logits.argmax();
                selected.fill(selectedToken, selected.numElements);
                return selected;
              },
            });
          };
          const exact = makeRequest(`exact-prefix-${selectedToken}`, 1);
          exact.inputIds = Array.from({ length: PAGE_SIZE }, (_, i) => i + 100);
          exact.promptTokenCount = exact.inputIds.length;
          requests.submit(exact);
          const tokens: number[] = [];
          for await (const token of exact.tokens.queue) tokens.push(token);
          assert.deepEqual(tokens, [selectedToken], "a full prefix hit publishes a fresh prefill selection");
          if (selectedToken === 71) {
            assert.equal(exact.cachedTokenCount, PAGE_SIZE - 1);
            assert.equal(exact.prefillTokenCount, mtp ? 2 : 1);
          }
        }
      } finally {
        model.planChunkedPrefill = planPrefill;
      }
      try {
        model.planChunkedPrefill = () => { throw new Error("expected prefill failure"); };
        const failed = makeRequest("failed", 1);
        requests.submit(failed);
        await assert.rejects(async () => {
          for await (const token of failed.tokens.queue) assert.fail(`unexpected token ${token}`);
        }, /expected prefill failure/);
      } finally {
        model.planChunkedPrefill = planPrefill;
      }
      const recovered = makeRequest("recovered", 1);
      requests.submit(recovered);
      const recovery: number[] = [];
      for await (const token of recovered.tokens.queue) recovery.push(token);
      assert.equal(recovery.length, 1, "a recoverable request error does not stop the scheduler");
      const prefilling = makeRequest("prefilling", 3);
      prefilling.inputIds = Array.from({ length: 32 }, (_, i) => i + 10);
      prefilling.promptTokenCount = prefilling.inputIds.length;
      const joining = makeRequest("joining", 3);
      const replacement = makeRequest("replacement", 3);
      replacement.cachedTokenCount = -1;
      let joinedDuringPrefill = false;
      let replacedDuringPrefill = false;
      let submittedJoining = false;
      const originalForward = model.forwardPhased;
      const captureDisabled = captureManager.disabled;
      // This admission probe observes host forwards, which graph replay bypasses.
      captureManager.disabled = true;
      try {
        model.forwardPhased = function* (...args) {
          if (!submittedJoining && prefilling.inputIds.length < 32) {
            assert.ok(prefilling.inputIds.length > 1 && prefilling.inputIds.length < 32);
            submittedJoining = true;
            requests.submit(joining);
          } else if (!joinedDuringPrefill && metrics.runningRequests === 2) {
            assert.ok(prefilling.inputIds.length > 1, "admit another request before prefill finishes");
            joinedDuringPrefill = true;
            prefilling.tokens.end();
            requests.submit(replacement);
          } else if (joinedDuringPrefill && !replacedDuringPrefill && replacement.cachedTokenCount >= 0) {
            assert.equal(joining.generatedTokenCount, 1, "the completed row publishes its prefill target once");
            assert.equal(joining.inputIds.length, 0);
            replacedDuringPrefill = true;
          }
          return yield* originalForward.apply(this, args);
        };
        requests.submit(prefilling);
        const resumed = await Promise.all([joining, replacement].map(async request => {
          const tokens: number[] = [];
          for await (const token of request.tokens.queue) tokens.push(token);
          return tokens;
        }));
        assert.ok(joinedDuringPrefill);
        assert.ok(replacedDuringPrefill, "replace a cancelled prefill row on the next scheduler turn");
        assert.equal(prefilling.generatedTokenCount, 0);
        assert.deepEqual(resumed.map(tokens => tokens.length), [3, 3]);
        if (!mtp) {
          assert.deepEqual(resumed[0], output[1], "chunk completion preserves non-MTP output");
          assert.deepEqual(resumed[1], output[1]);
        }
        assert.deepEqual(joining.inputIds, []);
        assert.deepEqual(replacement.inputIds, []);
      } finally {
        model.forwardPhased = originalForward;
        captureManager.disabled = captureDisabled;
      }
    } finally {
      scheduler.stop();
      await running;
    }
    assert.equal(metrics.runningRequests, 0);
    assert.ok(captureManager.captured.size > 0, "decode uses the capture manager");
    assert.ok([...captureManager.captured.keys()].every(key => !key.startsWith("prefill")), "prefill never captures");
    ws.assertClear();
  });

  for (const { topks, graphs, restart } of [
    { topks: [1, 1, 1], graphs: true, restart: false },
    { topks: [1], graphs: false, restart: true },
    { topks: [2, 1], graphs: true, restart: false },
    { topks: [2, 1], graphs: false, restart: false },
  ]) it(`stateful MTP generator matches independent verification and cleans up (${topks}, graphs=${graphs}, restart=${restart})`, async () => {
    // Reconditioning changes forward shapes and therefore can change later
    // near-tied BF16 argmax decisions. Compare each verification against the
    // independent causal forward using the SAME linear draft. For branching,
    // compare eager execution with replay of the complete generator instead.
    const trace: { input: number[]; tokens: number[]; accepted: number; restart: boolean }[] = [];
    const run = async (generator: boolean) => {
      using cache = model.createChatCache(32, 1);
      cache.reset(1);
      using manager = new CaptureManager(glm);
      manager.disabled = !graphs || !generator;
      const prompt = makeLongPrompt(63);
      const draft0 = await prefillChunks(model, ws, cache, [prompt], manager);
      const output = [...draft0.targetTokens];
      if (generator || topks.some(k => k > 1)) {
        let replayed = false;
        while (output.length < 32) {
          let steps = 0;
          for await (const step of model.generateMtpDecode(ws, cache, topks, graphs ? manager : undefined)) {
            assert.equal(step.numDraftTokens, topks.length);
            assert.equal(step.tokens[0].length, step.numAccepted[0] + 1);
            const buf = ws.tensors.get(`glm51_mtp_decode_1_${topks.join("_")}_inputs_host`)!.readPinnedBuffer();
            trace.push({ input: Array.from({ length: buf.length / 4 }, (_, i) => buf.readInt32LE(i * 4)),
              tokens: [...step.tokens[0]], accepted: step.numAccepted[0], restart: steps === 0 });
            assert.equal(cache.getPagedKV().sequences[0].reportedTokenCount(), cache.getPagedKV().sequences[0].allocLen);
            output.push(...step.tokens[0]);
            assert.deepEqual(cache.getPagedKV().sequences[0].getTokenIds(), [...prompt, ...output.slice(0, -1)]);
            replayed ||= !step.warmup;
            steps++;
            if (output.length >= 32 || (restart && steps === 3)) break;
          }
          ws.assertClear();
          assert.equal(ws.staged.size, 0);
          assert.equal(cache.getPagedKV().sequences[0].allocLen, prompt.length + output.length - 1);
        }
        assert.ok(replayed, "the generator must reach graph replay");
      } else {
        for (const expected of trace) {
          if (expected.restart) {
            const sequence = cache.getPagedKV().sequences[0];
            const replayToken = sequence.getTokenIds().at(-1)!;
            sequence.truncate(sequence.allocLen - 1);
            const state = ws.planPrefill(model, 1, [1], cache);
            state.setInput([[replayToken]]);
            {
              using hidden = model.forwardModel(state);
              await glm.synchronizeAsync();
            }
            cache.reportTokens(0, [replayToken], expected.input[0]);
            ws.clearTracking();
          }
          {
            const originalLen = cache.getPagedKV().sequences[0].allocLen;
            const state = ws.planPrefill(model, 1, [expected.input.length], cache);
            state.setInput([expected.input]);
            {
              using hidden = model.forwardModel(state);
              using logits = state.computeLogits(hidden, model, true);
              using selected = logits.argmax();
              const predicted = selected.readInt32LEArray();
              let accepted = 0;
              while (accepted < topks.length && predicted[accepted] === expected.input[accepted + 1]) accepted++;
              assert.equal(accepted, expected.accepted);
              assert.deepEqual(predicted.slice(0, accepted + 1), expected.tokens);
            }
            cache.getPagedKV().sequences[0].truncate(originalLen + expected.accepted + 1);
            ws.assertClear();
            ws.clearTracking();
          }
          cache.reportTokens(0, expected.input.slice(0, expected.accepted + 1));
          output.push(...expected.tokens);
        }
      }
      return output.slice(0, 32);
    };
    assert.deepEqual(await run(true), await run(false));
  });

  it("stateful MTP generator supplies one shared sparse-slot row per branch in a batch", async () => {
    ws.free();
    ws = new ExecutionWorkspace(glm, 2, 128);
    const forwardMtp = model.forwardMtp.bind(model);
    model.forwardMtp = (state, hidden, slots, lengths) => {
      assert.equal(slots!.value.shape[0], state.totalTokens);
      assert.equal(lengths!.value.numElements, state.totalTokens);
      return forwardMtp(state, hidden, slots, lengths);
    };
    const run = async (graphs: boolean) => {
      using cache = model.createChatCache(32, 2);
      cache.reset(2);
      using manager = new CaptureManager(glm);
      manager.disabled = !graphs;
      const prompts = [makeLongPrompt(63), makeLongPrompt(31, [11, 12, 13])];
      await prefillChunks(model, ws, cache, prompts, manager);
      const result: number[][][] = [];
      for await (const step of model.generateMtpDecode(ws, cache, [2, 1], manager)) {
        result.push(step.tokens);
        if (result.length === 10) break;
      }
      ws.assertClear();
      return result;
    };
    // Exercise variable branch-commit sizes eagerly before recording graphs.
    const expected = await run(false);
    assert.deepEqual(await run(true), expected);
  });

  it("stateful MTP generator unwinds on a consumer exception and can recondition again", async () => {
    using cache = model.createChatCache(32, 1);
    cache.reset(1);
    const topks = [1, 1, 1];
    const prompt = [1, 2, 3, 4];
    await prefillChunks(model, ws, cache, [prompt], captureManager);
    const failure = new Error("consumer stopped");
    await assert.rejects(async () => {
      for await (const step of model.generateMtpDecode(ws, cache, topks, captureManager)) {
        throw failure;
      }
    }, error => error === failure);
    ws.assertClear();
    const oldSequence = cache.getPagedKV().sequences[0];
    for await (const step of model.generateMtpDecode(ws, cache, topks, captureManager)) {
      cache.getPagedKV().removeSequence(0);
      break;
    }
    ws.assertClear();
    assert.equal(oldSequence.allocLen, 0);
    assert.equal(oldSequence.pages.length, 0);
  });

  for (const temperature of [0, 0.7]) it(`stateful MTP generator supports device-side acceptance (temperature=${temperature})`, async () => {
    const topks = [1, 1, 1];
    using sampler = new SamplingWorkspace(glm, 4, model.cfg.vocabSize, 8,
      { maxBatchSize: 1, depth: topks.length, retainProposalsOnGpu: true });
    const params: SamplingParams = { temperature, topK: temperature ? 8 : 1, topP: 1,
      repetitionPenalty: 1, presencePenalty: 0, repetitionPenaltyWindow: 8 };
    const run = async (graphs: boolean) => {
      using cache = model.createChatCache(32, 1);
      cache.reset(1);
      using manager = new CaptureManager(glm);
      manager.disabled = !graphs;
      const prompt = [1, 2, 3, 4];
      const draft = await prefillChunks(model, ws, cache, [prompt], manager);
      for (const name of ["stepCounter", "draftStepCounter", "rejectionStepCounter"]) {
        const seed = Buffer.alloc(4);
        seed.writeUInt32LE(123456);
        sampler.tensors.get(name)!.h2d(seed);
      }
      sampler.updateMtpSampler([params]);
      sampler.updateSampler([params], [[]]);
      const output = [...draft.targetTokens];
      for await (const step of model.generateMtpDecode(ws, cache, topks, graphs ? manager : undefined, sampler)) {
        output.push(...step.tokens[0]);
        if (output.length >= 32) break;
      }
      ws.assertClear();
      return output.slice(0, 32);
    };
    assert.deepEqual(await run(true), await run(false));
  });

  for (const graphs of [false, true]) it(`executePrefill stays eager and initializes MTP decode (graphs=${graphs})`, async () => {
    using cache = model.createChatCache(32, 1);
    using manager = new CaptureManager(glm);
    manager.disabled = !graphs;
    for (let run = 0; run < 7; run++) {
      const prompt = [1, 2, 3, 4 + run];
      const generate = async (override: boolean) => {
        cache.reset(1);
        if (override) {
          const result = await model.executePrefill(ws, cache, [prompt], graphs ? manager : undefined);
          assert.equal("nextTokens" in result, false);
          assert.equal(result.warmup, false);
          assert.equal(manager.captured.size, 0);
          assert.deepEqual(cache.getPagedKV().sequences[0].getTokenIds(), prompt);
        } else {
          await model.executePrefill(ws, cache, [prompt]);
        }
        const output: number[] = [];
        let steps = 0;
        for await (const step of model.generateMtpDecode(ws, cache, [1, 1])) {
          output.push(...step.tokens[0]);
          if (++steps === 4) {
            break;
          }
        }
        ws.assertClear();
        return output;
      };
      assert.deepEqual(await generate(true), await generate(false));
    }
    assert.equal(manager.captured.size, 0);
  });

  it("GLM prefill conditions single-sequence phases and unphased batches", async () => {
    ws.free();
    ws = new ExecutionWorkspace(glm, 2, 128);
    using cache = model.createChatCache(32, 2);
    for (const lengths of [[9], [1, 8], [4, 5], [8, 1], [5, 4]]) {
      const prompts = lengths.map((length, batch) => Array.from({ length }, (_, i) => 10 + batch * 20 + i));
      const tail = prompts.map((_, batch) => [50 + batch]);
      cache.reset(prompts.length);
      await model.executePrefill(ws, cache, prompts);
      await model.executePrefill(ws, cache, tail);
      const expected = prefillTargets(cache);
      {
        cache.reset(prompts.length);
        const plan = prompts.length === 1 ? model["planPhasedPrefill"](ws, cache, prompts)
          : model.planChunkedPrefill(ws, cache, prompts);
        const rotations: number[][] = [];
        const forwardMtp = model.forwardMtp;
        model.forwardMtp = function (state, ...args) {
          rotations.push(state.input!.readInt32LEArray().slice(0, state.totalTokens));
          return forwardMtp.call(this, state, ...args);
        };
        try {
          let step = plan.generator.next();
          while (!step.done) {
            step = plan.generator.next();
          }
          assert.equal(step.value, undefined);
          await glm.synchronizeAsync();
          assert.deepEqual(cache.getPagedKV().sequences.map(sequence => sequence.getTokenIds()), prompts.map(() => []));
          plan.reportTokens();
        } finally {
          model.forwardMtp = forwardMtp;
          plan.generator.return([]);
        }
        ws.assertClear();
        ws.clearTracking();
        assert.deepEqual(cache.getPagedKV().sequences.map(sequence => sequence.getTokenIds()), prompts);
        {
          assert.equal(rotations.length, prompts.length === 1 ? 2 : 1);
          for (let part = 0; part < plan.states.length; part++) {
            let packed = 0;
            for (let batch = 0; batch < prompts.length; batch++) {
              const start = part ? plan.states[0].seqLens[batch] : 0;
              const length = plan.states[part].seqLens[batch];
              for (let i = 0; i < length; i++) {
                if (start + i + 1 < prompts[batch].length) {
                  assert.equal(rotations[part][packed + i], prompts[batch][start + i + 1]);
                }
              }
              packed += length;
            }
          }
        }
        await model.executePrefill(ws, cache, tail);
        assert.deepEqual(prefillTargets(cache), expected);
      }
    }
  });

  it("decode consumes the published target without modifying a shared source prefix", async () => {
    using cache = model.createChatCache(32, 2);
    for (const mtp of [false, true]) {
      for (const length of [1, 64, 65]) {
        cache.reset(1);
        const prompt = Array.from({ length }, (_, i) => i + 1);
        const result = await model.executePrefill(ws, cache, [prompt]);
        assert.equal("nextTokens" in result, false);
        const paged = cache.getPagedKV();
        const source = paged.sequences[0];
        const publishedTarget = 42;
        cache.reportTokens(0, [], publishedTarget);
        paged.copySequence(1, 0);
        paged.stageSequence(0, 123);
        const generator = mtp ? model.generateMtpDecode(ws, cache, [1, 1]) : model.generateDecode(ws, cache);
        try {
          const step = await generator.next();
          assert.equal(step.done, false);
          const tokens = step.value!.tokens[0];
          assert.deepEqual(paged.sequences[0].getTokenIds(), [...prompt, publishedTarget, ...tokens.slice(0, -1)]);
          assert.equal(paged.sequences[0].allocLen, length + tokens.length);
          assert.deepEqual(source.getTokenIds(), prompt, "replay must preserve the shared source prefix");
          assert.equal(source.allocLen, length);
        } finally {
          await generator.return();
        }
        paged.unstageSequence(123);
        ws.assertClear();
      }
    }
  });

  it("MTP planChunkedPrefill replays boundary tokens and reports only consumed input", async () => {
    ws.free();
    ws = new ExecutionWorkspace(glm, 2, 128);
    using cache = model.createChatCache(32, 2);
    cache.reset(1);
    const first = makeLongPrompt(65);
    const initial = await model.executePrefill(ws, cache, [first]);
    assert.equal("nextTokens" in initial, false);
    assert.deepEqual(cache.getPagedKV().sequences[0].getTokenIds(), first);

    // The existing sequence rewinds across a page boundary; the new row has no overlap.
    cache.getPagedKV().ensureSequence(1);
    const second = [[10, 11], [12, 13, 14]];
    const plan = model.planChunkedPrefill(ws, cache, second);
    assert.deepEqual(plan.states[0].seqLens, [3, 3]);
    assert.deepEqual(cache.getPagedKV().sequences.map(sequence => sequence.getTokenIds()), [first.slice(0, -1), []]);
    let step = plan.generator.next();
    while (!step.done) {
      step = plan.generator.next();
    }
    {
      assert.equal(step.value.length, 2);
      assert.ok(step.value.every(token => token !== undefined));
      await glm.synchronizeAsync();
      assert.deepEqual(cache.getPagedKV().sequences.map(sequence => sequence.getTokenIds()), [first.slice(0, -1), []],
        "GPU execution does not report tokens until the executor calls reportTokens");
      plan.reportTokens();
    }
    ws.assertClear();
    ws.clearTracking();
    const histories = [[...first, ...second[0]], second[1]];
    assert.deepEqual(cache.getPagedKV().sequences.map(sequence => sequence.getTokenIds()), histories);

    const third = [[15, 16], [17]];
    const result = await model.executePrefill(ws, cache, third);
    histories.forEach((history, i) => history.push(...third[i]));
    assert.deepEqual(cache.getPagedKV().sequences.map(sequence => sequence.getTokenIds()), histories);
    for await (const output of model.generateMtpDecode(ws, cache, [1, 1])) {
      assert.equal(output.numDraftTokens, 2);
      assert.deepEqual(cache.getPagedKV().sequences.map(sequence => sequence.getTokenIds()),
        histories.map((history, i) => [...history, result.targetTokens[i], ...output.tokens[i].slice(0, -1)]));
      break;
    }
    ws.assertClear();
  });

  it("chunked MTP prefill budgets overlap and preserves deferred sequences", async () => {
    ws.free();
    ws = new ExecutionWorkspace(glm, 2, 128);
    using cache = model.createChatCache(32, 2);
    cache.reset(2);
    const input = [[10, 11, 12, 13, 14], [20, 21, 22, 23]];
    const first = await model.executePrefill(ws, cache, input, undefined, 4, {
      selectTarget: logits => {
        const selected = logits.argmax();
        selected.fill(42, selected.numElements);
        return selected;
      },
    });
    assert.deepEqual(first.targetTokens, [42, undefined]);
    assert.equal(cache.getPagedKV().sequences[0].targetToken, 42);
    assert.deepEqual(first.prefillInputIdsList, [[10, 11, 12, 13], []]);
    assert.deepEqual(first.remainingInputIdsList, [[14], [20, 21, 22, 23]]);
    const second = await model.executePrefill(ws, cache, first.remainingInputIdsList, undefined, 4);
    assert.deepEqual(second.prefillInputIdsList, [[13, 14], [20, 21]]);
    assert.deepEqual(second.remainingInputIdsList, [[], [22, 23]]);
    assert.deepEqual(second.targetTokens, cache.getPagedKV().sequences.map(s => s.targetToken));
    assert.ok(second.targetTokens.every(token => token !== undefined));
    assert.deepEqual(cache.getPagedKV().sequences.map(s => s.getTokenIds()), [input[0], [20, 21]]);
    const third = await model.executePrefill(ws, cache, second.remainingInputIdsList, undefined, 4);
    assert.deepEqual(third.prefillInputIdsList, [[], [21, 22, 23]]);
    assert.deepEqual(third.remainingInputIdsList, [[], []]);
    assert.deepEqual(third.targetTokens, [undefined, cache.getPagedKV().sequences[1].targetToken]);
    assert.equal(cache.getPagedKV().sequences[0].targetToken, second.targetTokens[0]);
    assert.deepEqual(cache.getPagedKV().sequences.map(s => s.getTokenIds()), input);
    assert.throws(() => model.planChunkedPrefill(ws, cache, [[30], []], 1), /required overlap/);
    assert.deepEqual(cache.getPagedKV().sequences.map(s => s.getTokenIds()), input);
    const empty = await model.executePrefill(ws, cache, [[], []], undefined, 4);
    assert.deepEqual(empty.prefillInputIdsList, [[], []]);
    assert.deepEqual(empty.remainingInputIdsList, [[], []]);
    assert.deepEqual(empty.targetTokens, [undefined, undefined]);
    assert.deepEqual(cache.getPagedKV().sequences.map(s => s.getTokenIds()), input);
  });

  it("large batched MTP prefill stays unphased with cached prefixes", async () => {
    ws.free();
    ws = new ExecutionWorkspace(glm, 2, 8192);
    using cache = model.createChatCache(160, 2, 8192);
    cache.reset(2);
    // This is a planning-only regression; reserve prefix pages without a large forward.
    ws.planPrefill(model, 2, [1000, 1000], cache);
    cache.reportTokens(0, Array(1000).fill(7));
    cache.reportTokens(1, Array(1000).fill(8));
    await glm.synchronizeAsync();
    ws.clearTracking();
    const inputs = [Array(3000).fill(11), Array(2000).fill(12)];
    assert.throws(() => model["planPhasedPrefill"](ws, cache, inputs), /batch size 1/);
    const plan = model.planChunkedPrefill(ws, cache, inputs);
    assert.equal(plan.states.length, 1);
    assert.deepEqual(plan.states[0].seqLens, [3001, 2001]);
    assert.deepEqual(plan.remainingInputIdsList, [[], []]);
    plan.generator.return([]);
    await glm.synchronizeAsync();
    ws.assertClear();
    ws.clearTracking();
  });

  it("chunked MTP prefill defaults to 8192 execution rows including overlap", async () => {
    ws.free();
    ws = new ExecutionWorkspace(glm, 1, 8192);
    using cache = model.createChatCache(160, 1, 8192);
    const input = [Array(9000).fill(11) as number[]];
    for (const existing of [false, true]) {
      cache.reset(1);
      if (existing) await model.executePrefill(ws, cache, [[7]]);
      const plan = model.planChunkedPrefill(ws, cache, input);
      assert.deepEqual(plan.states.map(state => state.totalTokens), [4096, 4096]);
      assert.equal(plan.prefillInputIdsList[0].length, 8192);
      assert.equal(plan.prefillInputIdsList[0][0], existing ? 7 : 11);
      assert.equal(plan.remainingInputIdsList[0].length, existing ? 809 : 808);
      assert.deepEqual(input, [Array(9000).fill(11)]);
      plan.generator.return([]);
      await glm.synchronizeAsync();
      ws.assertClear();
      ws.clearTracking();
    }
  });

  it("matches sequential MTP chunks and final prefill", async () => {
    const inputA = [[1, 2, 3, 4]];
    const inputB = [[5, 6, 7, 8]];
    const finalInput = [[9, 10]];

    const run = async (phased: boolean) => {
      using cache = model.createChatCache(32, 1);
      cache.reset(1);
      if (phased) {
        const plan = model["planPhasedPrefill"](ws, cache, [[...inputA[0], ...inputB[0]]]);
        let result = plan.generator.next();
        while (!result.done) result = plan.generator.next();
        await glm.synchronizeAsync();
        plan.reportTokens();
        assert.deepEqual(result.value, [cache.getPagedKV().sequences[0].targetToken]);
        assert.notEqual(result.value[0], undefined);
        ws.assertClear();
        ws.clearTracking();
      } else {
        await model.executePrefill(ws, cache, inputA);
        await model.executePrefill(ws, cache, inputB);
      }
      return await prefillChunks(model, ws, cache, finalInput, captureManager);
    };

    const sequential = await run(false);
    const phased = await run(true);
    assert.deepStrictEqual(phased.targetTokens, sequential.targetTokens);
  });

  const greedy: SamplingParams = {
    temperature: 0, topK: 1, topP: 1,
    repetitionPenalty: 1, presencePenalty: 0, repetitionPenaltyWindow: 0,
  };

  const readBytes = (tensor: Tensor): Buffer => {
    const bytes = Buffer.alloc(tensor.bytes);
    tensor.d2h(bytes);
    glm.synchronize();
    return bytes;
  };

  it("linear speculative greedy matches token-comparison verification across generator iterations", async () => {
    const topks = [1, 1, 1];
    using target = new SamplingWorkspace(glm, topks.length + 1, model.cfg.vocabSize, 0, { maxBatchSize: 1, depth: topks.length, retainProposalsOnGpu: true });
    target.updateMtpSampler([greedy]);

    const run = async (linear: boolean) => {
      using manager = new CaptureManager(glm);
      manager.disabled = !linear;
      using cache = model.createChatCache(32, 1);
      cache.reset(1);
      const sequence = cache.getPagedKV().sequences[0];
      const prompt = [1, 2, 3, 4, 5, 6, 7, 8];
      target.updateSampler([greedy]);
      await prefillChunks(model, ws, cache, [prompt], manager, linear ? target : undefined);
      const outputs: { tokens: number[][]; numAccepted: number[] }[] = [];
      let originalAllocLen = sequence.allocLen;
      for await (const step of model.generateMtpDecode(ws, cache, topks, manager, linear ? target : undefined)) {
        assert.equal(sequence.allocLen, originalAllocLen + step.numAccepted[0] + 1);
        assert.equal(sequence.reportedTokenCount(), sequence.allocLen);
        outputs.push({ tokens: step.tokens, numAccepted: step.numAccepted });
        originalAllocLen = sequence.allocLen;
        if (outputs.length === 9) break;
      }
      if (linear) assert.ok([...manager.captured.values()].some(entry => entry.graphExec !== null && entry.capturedWorkspaces.has(target)));
      return outputs;
    };

    assert.deepStrictEqual(await run(true), await run(false));
  });

  for (const { graphs, retainProposalsOnGpu, force = true } of [
    { graphs: false, retainProposalsOnGpu: false }, { graphs: true, retainProposalsOnGpu: false },
    { graphs: false, retainProposalsOnGpu: true }, { graphs: true, retainProposalsOnGpu: true },
    { graphs: true, retainProposalsOnGpu: true, force: false },
  ]) it(`linear speculative stochastic rejection preserves target KV and the next MTP seed (${graphs ? "replay" : "eager"}, ${retainProposalsOnGpu ? "GPU" : "host"} q, ${force ? "forced" : "exact"})`, async t => {
    const topks = [1, 1, 1];
    const stochastic: SamplingParams = { ...greedy, temperature: 1, topK: 16, topP: 0.9 };
    using target = new SamplingWorkspace(glm, topks.length + 1, model.cfg.vocabSize, 8, { maxBatchSize: 1, depth: topks.length, retainProposalsOnGpu });
    using manager = new CaptureManager(glm);
    manager.disabled = !graphs;
    using controls = new WorkspaceBase(glm);
    const forcedPrefix = controls.alloc([topks.length], "I32", "forcedPrefix");
    const forcedCount = controls.alloc([1], "I32", "forcedCount");
    for (const name of ["stepCounter", "draftStepCounter", "rejectionStepCounter"]) {
      const seed = Buffer.alloc(4);
      seed.writeUInt32LE(123456);
      target.tensors.get(name)!.h2d(seed);
    }
    target.updateMtpSampler([stochastic]);
    target.updateSampler([stochastic], [[]]);
    using cache = model.createChatCache(32, 1);
    cache.reset(1);
    const sequence = cache.getPagedKV().sequences[0];
    const prompt = [1, 2, 3, 4, 5, 6, 7, 8];
    using reference = model.createChatCache(32, 1);
    reference.reset(1);
    using referenceWs = new ExecutionWorkspace(glm, 1, 128);
    const causalChunk = (tokens: number[], next: number, pad = true): Buffer => {
      using tracking = referenceWs.startTracking();
      const start = reference.getPagedKV().sequences[0].allocLen;
      // Match the verifier's four-row kernel dispatch without its custom mask,
      // rejected tokens, sampling, or commit machinery. Future zero rows cannot
      // condition the causal prefix; truncate them after the independent forward.
      const padding = Array(pad ? Math.max(0, topks.length + 1 - tokens.length) : 0).fill(0);
      const state = referenceWs.planPrefill(model, 1, [tokens.length + padding.length], reference);
      state.setInput([[...tokens, ...padding]]);
      using slots = new UsingHolder<Tensor>(undefined!);
      using lengths = new UsingHolder<Tensor>(undefined!);
      using hidden = model.forwardModel(state, slots, lengths);
      using shifted = referenceWs.alloc([tokens.length + padding.length], "I32");
      shifted.h2d(Buffer.from(new Int32Array([...tokens.slice(1), next, ...padding]).buffer));
      state.setInput(shifted);
      using mtp = model.forwardMtp(state, hidden, slots, lengths);
      using last = mtp.narrow(tokens.length - 1, 1);
      const result = readBytes(last);
      reference.getPagedKV().sequences[0].truncate(start + tokens.length);
      return result;
    };
    let comparedBytes = 0;
    const assertCaches = () => {
      const actual = cache.getPagedKV();
      const expected = reference.getPagedKV();
      assert.equal(actual.sequences[0].allocLen, expected.sequences[0].allocLen);
      for (const field of ["ckvData", "kData", "kScaleData"] as const) {
        actual[field].forEach((tensor, layer) => {
          if (!tensor) return;
          assert.equal(tensor.same(expected[field][layer]), false, "reference caches must not alias the tested caches");
          const rowBytes = tensor.bytes / (actual.maxPages * actual.pageSize);
          for (let page = 0; page < sequence.pages.length; page++) {
            using a = tensor.narrow(sequence.pages[page].id, 1);
            using b = expected[field][layer].narrow(expected.sequences[0].pages[page].id, 1);
            const bytes = Math.min(actual.pageSize, sequence.allocLen - page * actual.pageSize) * rowBytes;
            assert.deepEqual(readBytes(a).subarray(0, bytes), readBytes(b).subarray(0, bytes), `${field} layer=${layer} page=${page}`);
            comparedBytes += bytes;
          }
        });
      }
    };
    let retainedSeed: Tensor;
    const executionManager: import("../src/execution-manager").ExecutionManager = {
      get captureEnabled() { return manager.captureEnabled; },
      execute(options, fn) {
        retainedSeed = options.inputs.seed;
        return manager.execute(options, fn);
      },
    };
    const draft = await prefillChunks(model, ws, cache, [prompt], manager, target);
    causalChunk(prompt, 0);
    assertCaches();
    reference.getPagedKV().sequences[0].truncate(prompt.length - 1);
    causalChunk([prompt.at(-1)!], draft.targetTokens[0], false);

    let forcedAccepted = 0;
    let verifyRuns = 0;
    let draftRuns = 0;
    const forcedSampler: TokenSelector = {
      selectTarget: logits => target.selectTarget(logits),
      mtpEnabled: true,
      get mtpCaptureKey() { return force ? `forced-linear-rejection:${forcedAccepted}` : "exact-linear-rejection"; },
      prepareDraft: (batch, depth) => target.prepareDraft(batch, depth),
      sampleDraft: (logits, depth) => { draftRuns++; return target.sampleDraft(logits, depth); },
      prepareVerificationFromDevice: (draftTokens, batchSize) => {
        target.prepareVerificationFromDevice(draftTokens, batchSize);
        forcedPrefix.memcpy(draftTokens, topks.length * 4);
      },
      verify: logits => {
        verifyRuns++;
        const result = target.verify(logits);
        if (!force) return result;
        // Exercise the real stochastic verifier, then force commit depths without
        // falsifying q snapshots. This checks pipeline mechanics, not sampling law.
        using selected = target.sample(logits);
        result.tokens.memcpy(selected);
        if (forcedAccepted > 0) {
          using prefix = result.tokens.narrow(0, forcedAccepted);
          prefix.memcpy(forcedPrefix, forcedAccepted * 4);
        }
        result.numAccepted.memcpy(forcedCount);
        return result;
      },
    };

    const generator = model.generateMtpDecode(ws, cache, topks, executionManager, forcedSampler);
    let outstanding = draft.targetTokens[0];
    if (force) target.updateSampler(Array.from({ length: topks.length + 1 }, () => stochastic), Array.from({ length: topks.length + 1 }, () => []));
    try {
      for (let iteration = 0; iteration < 18; iteration++) {
        forcedAccepted = iteration % topks.length;
        forcedCount.h2d(Buffer.from(new Int32Array([forcedAccepted]).buffer));
        const originalAllocLen = sequence.allocLen;
        const history = sequence.getTokenIds();
        const next = await generator.next();
        assert.equal(next.done, false);
        const step = next.value!;
        if (force) assert.deepStrictEqual(step.numAccepted, [forcedAccepted]);
        assert.equal(step.numDraftTokens, topks.length);
        assert.equal(sequence.allocLen, originalAllocLen + step.numAccepted[0] + 1);
        assert.deepStrictEqual(sequence.getTokenIds(), [...history, outstanding, ...step.tokens[0].slice(0, -1)]);
        assert.equal(sequence.reportedTokenCount(), sequence.allocLen);
        const expectedSeed = causalChunk([outstanding, ...step.tokens[0].slice(0, -1)], step.tokens[0].at(-1)!);
        assertCaches();
        {
          using seed = retainedSeed!.narrow(step.numAccepted[0], 1);
          assert.deepEqual(readBytes(seed), expectedSeed, `retained MTP seed after acceptance=${step.numAccepted[0]}`);
        }
        assert.ok(bf16BytesToF32(expectedSeed).every(Number.isFinite));
        outstanding = step.tokens[0].at(-1)!;
        assert.equal(target.batchSize, force ? topks.length + 1 : 1);
        assert.equal(target.penaltyCount.readInt32LEArray()[0], force ? iteration + 1 : 1, "verification/draft must not advance ordinary target penalty history");
        target.assertClear();
      }
    } finally {
      await generator.return();
    }

    // Only consume the tested cache's outstanding token after all verify loops.
    const cachedNext = ws.forwardEagerDecode(model, [outstanding], cache);
    const freshNext = referenceWs.forwardEagerDecode(model, [outstanding], reference);
    assert.deepStrictEqual(cachedNext, freshNext, "target KV after rejection differs from fresh history");
    if (graphs) {
      assert.ok(verifyRuns < 18, "verification must replay without calling the JS sampler");
      assert.ok(draftRuns < 18 * topks.length, "draft must replay without calling the JS sampler");
      assert.ok([...manager.captured].some(([key, entry]) => key.startsWith("glm51-mtp-decode") && entry.graphExec !== null && entry.capturedWorkspaces.has(target)));
    }
    t.diagnostic(`${comparedBytes} committed cache bytes and 18 BF16 seeds equal exactly; verification JS runs=${verifyRuns}/18, draft JS runs=${draftRuns}/54`);
    glm.synchronize();
    ws.assertClear();
  });

  it("linear speculative workspace samples real decode logits with penalties through CaptureManager replay", t => {
    using sampling = new SamplingWorkspace(glm, 4, model.cfg.vocabSize, 8, { maxBatchSize: 1, depth: 3, retainProposalsOnGpu: true });
    using manager = new CaptureManager(glm);
    using cache = model.createChatCache(32, 1);
    using reference = model.createChatCache(32, 1);
    using referenceWs = new ExecutionWorkspace(glm, 1, 128);
    cache.reset(1);
    reference.reset(1);
    const prompt = [1, 2, 3, 4, 5, 6, 7, 8];
    let token = ws.forwardEagerPrefill(model, [prompt], cache)[0];
    assert.equal(referenceWs.forwardEagerPrefill(model, [prompt], reference)[0], token);
    const history = [...prompt, token];
    const seed = Buffer.alloc(4);
    seed.writeUInt32LE(123456);
    sampling.stepCounter.h2d(seed);
    const draftCounter = readBytes(sampling.draftStepCounter);
    let runs = 0;
    let changedWinners = 0;
    for (let iteration = 0; iteration < 8; iteration++) {
      const referenceBytes = (() => {
        using tracking = referenceWs.startTracking();
        const state = referenceWs.planDecode(model, 1, reference, true);
        state.setInput([[token]]);
        using hidden = model.forwardModel(state);
        using logits = state.computeLogits(hidden, model);
        return readBytes(logits);
      })();
      const values = bf16BytesToF32(referenceBytes);
      let rawBest = 0;
      values.forEach((value, index) => { if (value > values[rawBest]) rawBest = index; });
      // Prescribe a history containing the current winner so penalties are
      // exercised, rather than accidentally testing an unpenalized argmax.
      const penaltyHistory = [...history.slice(-7), rawBest];
      const seen = new Set(penaltyHistory);
      const params = { ...greedy, repetitionPenalty: 1.1 + iteration * 0.05, presencePenalty: 0.2 + iteration * 0.1 };
      sampling.updateSampler([params], [penaltyHistory]);
      const rep = Math.fround(params.repetitionPenalty), presence = Math.fround(params.presencePenalty);
      for (const id of seen) values[id] = Math.fround(Math.fround(values[id] < 0 ? values[id] * rep : values[id] / rep) - presence);
      let expected = 0;
      values.forEach((value, index) => {
        assert.ok(Number.isFinite(value));
        if (value > values[expected]) expected = index;
      });
      if (expected !== rawBest) changedWinners++;
      const state = ws.planDecode(model, 1, cache, true);
      state.setInput([[token]]);
      {
        const result = state.capture(manager, {}, () => {
          runs++;
          using hidden = model.forwardModel(state);
          const logits = state.computeLogits(hidden, model);
          return { logits, tokens: sampling.sample(logits) };
        }, ["real-model-penalty-sampling", sampling.captureKey]);
        using logits = result.logits;
        using selected = result.tokens;
        assert.deepEqual(readBytes(logits), referenceBytes, `decode logits iteration=${iteration}`);
        token = selected.readInt32LEArray()[0];
        assert.equal(token, expected, `CPU penalty oracle iteration=${iteration}`);
        assert.equal(sampling.penaltyCount.readInt32LEArray()[0], penaltyHistory.length + 1);
        const penaltyTokens = Array(8).fill(0);
        penaltyHistory.forEach((id, index) => { penaltyTokens[index] = id; });
        penaltyTokens[penaltyHistory.length % 8] = expected;
        assert.deepEqual(sampling.penaltyTokens.readInt32LEArray().slice(0, 8), penaltyTokens);
        assert.equal(readBytes(sampling.stepCounter).readUInt32LE(), 123456 + iteration + 1);
      }
      history.push(token);
      ws.clearTracking();
      sampling.assertClear();
    }
    assert.ok(runs < 8, "real model decode and sampling must replay");
    assert.ok(changedWinners > 0, "penalties must change at least one model output");
    assert.deepEqual(readBytes(sampling.draftStepCounter), draftCounter);
    assert.ok([...manager.captured.values()].some(entry => entry.graphExec !== null && entry.capturedWorkspaces.has(sampling) && entry.capturedWorkspaces.has(ws)));
    t.diagnostic(`8 full BF16 logits rows equal exactly; ${changedWinners} penalized winners changed, JS runs=${runs}/8; penalty ring and RNG counters match`);
  });
});
