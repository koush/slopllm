import assert from "node:assert/strict";
import path from "node:path";
import { after, afterEach, before, beforeEach, describe, it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import type { ChatCache, ChatModel, TokenSelector, SamplingParams } from "../src/chat_model";
import type { DeviceOps } from "../src/device_ops";
import { executePlan, ExecutionWorkspace } from "../src/execution-workspace";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { ParallelOps } from "../src/parallel_ops";
import { Tensor } from "../src/tensor";
import { PAGE_SIZE } from "../src/paged_kv";
import { PhasedPrefillRunner, splitRaggedInput } from "../src/phased-prefill";
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

function assertPhasedRaggedMatches(model: Glm51Model, glm: DeviceOps, ws: ExecutionWorkspace): void {
  const inputA = [[1, 2, 3], [11, 12]];
  const inputB = [[4, 5], [21, 22, 23, 24]];

  const run = (phased: boolean): { tokensA: number[]; tokensB: number[]; decode: number[] } => {
    using cache = model.createChatCache(32, 3);
    const pagedKV = cache.getPagedKV();
    cache.reset(3);
    const [sequence0, sequence1, sequence2] = pagedKV.sequences;

    // A runs rows 0/1. B then runs rows 0/2, matching the server chunker's
    // selected/deferred row behavior while preserving each Sequence object.
    pagedKV.stageSequence(2, 2);
    let tokensA: number[] = undefined!;
    let tokensB: number[] = undefined!;

    if (phased) {
      {
        const runner = new PhasedPrefillRunner(model, glm);
        const stateA = ws.planPrefill(model, 2, inputA.map(ids => ids.length), cache);
        stateA.setInput(inputA);
        sequence0.reportTokens(inputA[0]);
        sequence1.reportTokens(inputA[1]);

        pagedKV.stageSequence(1, 1);
        pagedKV.unstageSequence(2);
        const stateB = ws.planPrefill(model, 2, inputB.map(ids => ids.length), cache);
        stateB.setInput(inputB);
        sequence0.reportTokens(inputB[0]);
        sequence2.reportTokens(inputB[1]);

        runner.runPair(stateA, stateB, (pairedA, hiddenA, pairedB, hiddenB) => {
          using logitsA = pairedA.computeLogits(hiddenA, model);
          using logitsB = pairedB.computeLogits(hiddenB, model);
          using argmaxA = logitsA.argmax();
          using argmaxB = logitsB.argmax();
          tokensA = argmaxA.readInt32LEArray();
          tokensB = argmaxB.readInt32LEArray();
        });
      }
      glm.synchronize();
      ws.assertClear();
    } else {
      {
        const stateA = ws.planPrefill(model, 2, inputA.map(ids => ids.length), cache);
        stateA.setInput(inputA);
        using hiddenA = model.forward(stateA);
        using logitsA = stateA.computeLogits(hiddenA, model);
        using argmaxA = logitsA.argmax();
        tokensA = argmaxA.readInt32LEArray();
      }
      sequence0.reportTokens(inputA[0]);
      sequence1.reportTokens(inputA[1]);
      glm.synchronize();
      ws.assertClear();

      pagedKV.stageSequence(1, 1);
      pagedKV.unstageSequence(2);
      {
        const stateB = ws.planPrefill(model, 2, inputB.map(ids => ids.length), cache);
        stateB.setInput(inputB);
        using hiddenB = model.forward(stateB);
        using logitsB = stateB.computeLogits(hiddenB, model);
        using argmaxB = logitsB.argmax();
        tokensB = argmaxB.readInt32LEArray();
      }
      sequence0.reportTokens(inputB[0]);
      sequence2.reportTokens(inputB[1]);
      glm.synchronize();
      ws.assertClear();
    }

    // Restore the deferred A-only row. Active order is now 0/2/1.
    pagedKV.unstageSequence(1);
    const firstTokens = [tokensB[0], tokensB[1], tokensA[1]];
    sequence0.reportTokens([firstTokens[0]]);
    sequence2.reportTokens([firstTokens[1]]);
    sequence1.reportTokens([firstTokens[2]]);
    const decode = ws.forwardEagerDecode(model, firstTokens, cache);
    glm.synchronize();
    ws.assertClear();
    return { tokensA, tokensB, decode };
  };

  // The server warms this workspace before serving requests. Establish the
  // same allocator/recycle state before comparing the two execution modes.
  run(false);
  const sequential = run(false);
  const phased = run(true);
  assert.deepStrictEqual(phased.tokensA, sequential.tokensA, "A prefill tokens differ");
  assert.deepStrictEqual(phased.tokensB, sequential.tokensB, "B prefill tokens differ");
  assert.deepStrictEqual(phased.decode, sequential.decode, "decode tokens differ after phased prefill");
}

describe("GLM-5.1 small model smoke test", () => {
  let glm: GlmOps;
  let model: Glm51Model;
  let ws: ExecutionWorkspace;

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Glm51Model.fromPretrained(glm, SMALL_MODEL_DIR);
    ws = new ExecutionWorkspace(glm, 3, 128);
  });

  after(() => {
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

  it("phased prefill matches sequential prefill for different ragged row sets", () => {
    assertPhasedRaggedMatches(model, glm, ws);
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

  it("phased prefill matches sequential prefill for different ragged row sets", () => {
    assertPhasedRaggedMatches(modelCp, po, wsCp);
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
    const split = splitRaggedInput(input)!;
    assert.equal(split.inputA[0].length, 3);
    assert.equal(split.inputB[0].length, 4);
    assert.deepStrictEqual(split.nextA, split.inputB.map(ids => ids[0]));
    for (let i = 0; i < input.length; i++) {
      assert.ok(split.inputA[i].length > 0);
      assert.ok(split.inputB[i].length > 0);
      assert.deepStrictEqual([...split.inputA[i], ...split.inputB[i]], input[i]);
    }
    assert.equal(splitRaggedInput([[1], [2, 3]]), undefined);
    assert.equal(splitRaggedInput([[1, 2], [3, 4]]), undefined);
  });

  it("rolls back an unstarted phased MTP plan", () => {
    using cache = model.createChatCache(32, 1);
    cache.reset(1);
    const sequence = cache.getPagedKV().sequences[0];

    const unstarted = model.planPrefillMtpChunkPhased(ws, cache, [[1, 2, 3, 4]], [5]);
    assert.equal(sequence.allocLen, 4);
    unstarted[Symbol.dispose]();
    assert.equal(sequence.allocLen, 0);
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
      prompts.forEach((ids, batch) => cache.reportTokens(batch, [...ids, current[batch]]));
      ws.clearTracking();
      const output: number[][] = [];
      let replayed = false;
      const record = (tokens: number[]) => {
        output.push([...tokens]);
        current = [...tokens];
        tokens.forEach((token, batch) => cache.reportTokens(batch, [token]));
      };
      if (generator) {
        while (output.length < 12) {
          for await (const step of model.generateDecode(ws, cache, current, graphs ? manager : undefined,
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
        assert.equal(sequence.getTokenIds().length, sequence.allocLen + 1);
      });
      return output;
    };
    assert.deepEqual(await run(true), await run(false));
  });

  it("stateful decode generator preserves caller tensors on entry, resume, and break", async () => {
    using cache = model.createChatCache(32, 1);
    cache.reset(1);
    let targets = chunkedPrefill(model, ws, cache, [1, 2, 3], [3]);
    ws.assertClear();
    ws.clearTracking();
    {
      using callerTensor = ws.alloc([1], "I32");
      const generator = model.generateDecode(ws, cache, targets);
      await assert.rejects(generator.next(), /assertClear/);
      assert.equal(callerTensor.disposed, false);
    }
    const generator = model.generateDecode(ws, cache, targets);
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
      for await (const step of model.generateDecode(ws, cache, targets)) {
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

  for (const mtp of [false, true]) it(`queued scheduler admits, drains, cancels, and returns to idle (mtp=${mtp})`, async () => {
    ws.free();
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
      decodeLatencyMs: 0, phasedPrefill: false, topks: mtp ? [1, 1] : undefined });
    let stopped = false;
    const running = scheduler.run().finally(() => { stopped = true; });
    const long = makeRequest("long", 24);
    const short = makeRequest("short", 3);
    const cancelled = makeRequest("cancelled", 100);
    const abandoned = makeRequest("abandoned", 100);
    abandoned.tokens.end();
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
      assert.equal(abandoned.generatedTokenCount, 0);
      assert.equal(cancelled.tokens.submit(0), false);
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(stopped, false, "scheduler waits again after draining the batch");
      const next = makeRequest("next", 1);
      requests.submit(next);
      const final: number[] = [];
      for await (const token of next.tokens.queue) final.push(token);
      assert.equal(final.length, 1);
      const forward = model.forwardModel;
      try {
        model.forwardModel = () => { throw new Error("expected prefill failure"); };
        const failed = makeRequest("failed", 1);
        requests.submit(failed);
        await assert.rejects(async () => {
          for await (const token of failed.tokens.queue) assert.fail(`unexpected token ${token}`);
        }, /expected prefill failure/);
      } finally {
        model.forwardModel = forward;
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
      const originalForward = model.forwardModel;
      try {
        model.forwardModel = function (...args) {
          if (!submittedJoining) {
            assert.ok(prefilling.inputIds.length > 1 && prefilling.inputIds.length < 32);
            submittedJoining = true;
            requests.submit(joining);
          } else if (!joinedDuringPrefill && metrics.runningRequests === 2) {
            assert.ok(prefilling.inputIds.length > 1, "admit another request before prefill finishes");
            joinedDuringPrefill = true;
            prefilling.tokens.end();
            requests.submit(replacement);
          } else if (joinedDuringPrefill && !replacedDuringPrefill && replacement.cachedTokenCount >= 0) {
            assert.equal(joining.generatedTokenCount, mtp ? 0 : 1,
              "non-MTP publishes the first token as soon as its prompt fits a chunk");
            assert.equal(joining.inputIds.length, mtp ? 1 : 0);
            replacedDuringPrefill = true;
          }
          return originalForward.apply(this, args);
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
        model.forwardModel = originalForward;
      }
    } finally {
      scheduler.stop();
      await running;
    }
    assert.equal(metrics.runningRequests, 0);
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
    const trace: { input: number[]; tokens: number[]; accepted: number; bootstrap: boolean }[] = [];
    const run = async (generator: boolean) => {
      using cache = model.createChatCache(32, 1);
      cache.reset(1);
      using manager = new CaptureManager(glm);
      manager.disabled = !graphs || !generator;
      const prompt = makeLongPrompt(63);
      const draft0 = (await executePlan(manager, ws,
        model.planPrefillMtp(ws, cache, [prompt]))).result;
      cache.reportTokens(0, prompt);
      cache.reportTokens(0, draft0.targetTokens);
      const output = [...draft0.targetTokens];
      if (generator || topks.some(k => k > 1)) {
        let current = draft0.targetTokens;
        let replayed = false;
        while (output.length < 32) {
          let steps = 0;
          for await (const step of model.generateMtpDecode(ws, cache, current, topks, graphs ? manager : undefined)) {
            assert.ok(step.numDraftTokens === 0 || step.numDraftTokens === topks.length);
            assert.equal(step.tokens[0].length, step.numAccepted[0] + 1);
            const bootstrap = step.numDraftTokens === 0;
            const buf = ws.tensors.get(`glm51_mtp_decode_1_${topks.join("_")}_inputs_host`)!.readPinnedBuffer();
            trace.push({ input: bootstrap ? [...current] : Array.from({ length: buf.length / 4 }, (_, i) => buf.readInt32LE(i * 4)),
              tokens: [...step.tokens[0]], accepted: step.numAccepted[0], bootstrap });
            cache.reportTokens(0, step.tokens[0]);
            output.push(...step.tokens[0]);
            current = [step.tokens[0].at(-1)!];
            replayed ||= !step.warmup;
            if (output.length >= 32 || (restart && ++steps === 3)) break;
          }
          ws.assertClear();
          assert.equal(ws.staged.size, 0);
          assert.equal(cache.getPagedKV().sequences[0].allocLen, prompt.length + output.length - 1);
        }
        assert.ok(replayed, "the generator must reach graph replay");
      } else {
        for (const expected of trace) {
          if (expected.bootstrap) {
            const state = ws.planPrefill(model, 1, [1], cache);
            state.setInput([expected.input]);
            {
              using slots = new UsingHolder<Tensor>(undefined!);
              using lengths = new UsingHolder<Tensor>(undefined!);
              using hidden = model.forwardModel(state, slots, lengths);
              using logits = state.computeLogits(hidden, model);
              using selected = logits.argmax();
              assert.deepEqual(selected.readInt32LEArray(), expected.tokens);
              state.setInput([expected.tokens]);
              using mtpHidden = model.forwardMtp(state, hidden, slots, lengths);
              await glm.synchronizeAsync();
            }
            ws.clearTracking();
          } else {
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
          cache.reportTokens(0, expected.tokens);
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
      const initial = (await executePlan(manager, ws, model.planPrefillMtp(ws, cache, prompts))).result;
      prompts.forEach((ids, batch) => cache.reportTokens(batch, [...ids, initial.targetTokens[batch]]));
      const result: number[][][] = [];
      for await (const step of model.generateMtpDecode(ws, cache, initial.targetTokens, [2, 1], manager)) {
        step.tokens.forEach((tokens, batch) => cache.reportTokens(batch, tokens));
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
    const draft = (await executePlan(captureManager, ws,
      model.planPrefillMtp(ws, cache, [prompt]))).result;
    cache.reportTokens(0, prompt);
    cache.reportTokens(0, draft.targetTokens);
    let targets = draft.targetTokens;
    const failure = new Error("consumer stopped");
    await assert.rejects(async () => {
      for await (const step of model.generateMtpDecode(ws, cache, targets, topks, captureManager)) {
        cache.reportTokens(0, step.tokens[0]);
        targets = [step.tokens[0].at(-1)!];
        throw failure;
      }
    }, error => error === failure);
    ws.assertClear();
    const oldSequence = cache.getPagedKV().sequences[0];
    for await (const step of model.generateMtpDecode(ws, cache, targets, topks, captureManager)) {
      cache.reportTokens(0, step.tokens[0]);
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
      const draft = (await executePlan(manager, ws,
        model.planPrefillMtp(ws, cache, [prompt]))).result;
      cache.reportTokens(0, prompt);
      cache.reportTokens(0, draft.targetTokens);
      for (const name of ["stepCounter", "draftStepCounter", "rejectionStepCounter"]) {
        const seed = Buffer.alloc(4);
        seed.writeUInt32LE(123456);
        sampler.tensors.get(name)!.h2d(seed);
      }
      sampler.updateMtpSampler([params]);
      sampler.updateSampler([params], [[]]);
      const output = [...draft.targetTokens];
      for await (const step of model.generateMtpDecode(ws, cache, draft.targetTokens, topks, graphs ? manager : undefined, sampler)) {
        cache.reportTokens(0, step.tokens[0]);
        output.push(...step.tokens[0]);
        if (output.length >= 32) break;
      }
      ws.assertClear();
      return output.slice(0, 32);
    };
    assert.deepEqual(await run(true), await run(false));
  });

  it("matches sequential MTP chunks and final prefill", async () => {
    const inputA = [[1, 2, 3, 4]];
    const inputB = [[5, 6, 7, 8]];
    const nextA = [5];
    const nextB = [9];
    const finalInput = [[9, 10]];

    const run = async (phased: boolean) => {
      using cache = model.createChatCache(32, 1);
      cache.reset(1);
      const sequence = cache.getPagedKV().sequences[0];

      if (phased) {
        const runner = new PhasedPrefillRunner(model, glm);
        using planA = model.planPrefillMtpChunkPhased(ws, cache, inputA, nextA);
        using planB = model.planPrefillMtpChunkPhased(ws, cache, inputB, nextB);
        runner.runPlanPair(planA, planB);
        sequence.reportTokens(inputA[0]);
        sequence.reportTokens(inputB[0]);
        await glm.synchronizeAsync();
        ws.assertClear();
        ws.resetPlanSlots();
      } else {
        await executePlan(captureManager, ws, model.planPrefillMtpChunk(ws, cache, inputA, nextA));
        sequence.reportTokens(inputA[0]);
        await executePlan(captureManager, ws, model.planPrefillMtpChunk(ws, cache, inputB, nextB));
        sequence.reportTokens(inputB[0]);
      }

      const mtpInput = model.prepareMtpInput(cache, finalInput);
      return (await executePlan(
        captureManager,
        ws,
        model.planPrefillMtp(ws, cache, mtpInput),
      )).result;
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
      const draft = (await executePlan(manager, ws,
        model.planPrefillMtp(ws, cache, model.prepareMtpInput(cache, [prompt]),
          linear ? target : undefined))).result;
      sequence.reportTokens([...prompt, ...draft.targetTokens]);
      const outputs: { tokens: number[][]; numAccepted: number[] }[] = [];
      let originalAllocLen = sequence.allocLen;
      for await (const step of model.generateMtpDecode(ws, cache, draft.targetTokens, topks, manager, linear ? target : undefined)) {
        assert.equal(sequence.allocLen, originalAllocLen + step.numAccepted[0] + 1);
        sequence.reportTokens(step.tokens[0]);
        assert.equal(sequence.reportedTokenCount(), sequence.allocLen + 1);
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
    const draft = (await executePlan(manager, ws,
      model.planPrefillMtp(ws, cache, model.prepareMtpInput(cache, [prompt]),
        target))).result;
    sequence.reportTokens([...prompt, ...draft.targetTokens]);
    causalChunk(prompt, draft.targetTokens[0]);
    assertCaches();

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

    const generator = model.generateMtpDecode(ws, cache, draft.targetTokens, topks, executionManager, forcedSampler);
    let outstanding = draft.targetTokens[0];
    const bootstrap = await generator.next();
    assert.equal(bootstrap.done, false);
    const initial = bootstrap.value!.tokens[0];
    causalChunk([outstanding], initial[0], false);
    outstanding = initial[0];
    sequence.reportTokens(initial);
    assertCaches();
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
        sequence.reportTokens(step.tokens[0]);
        assert.deepStrictEqual(sequence.getTokenIds(), [...history, ...step.tokens[0]]);
        assert.equal(sequence.reportedTokenCount(), sequence.allocLen + 1);
        const expectedSeed = causalChunk([outstanding, ...step.tokens[0].slice(0, -1)], step.tokens[0].at(-1)!);
        assertCaches();
        {
          using seed = retainedSeed!.narrow(step.numAccepted[0], 1);
          assert.deepEqual(readBytes(seed), expectedSeed, `retained MTP seed after acceptance=${step.numAccepted[0]}`);
        }
        assert.ok(bf16BytesToF32(expectedSeed).every(Number.isFinite));
        outstanding = step.tokens[0].at(-1)!;
        assert.equal(target.batchSize, force ? topks.length + 1 : 1);
        assert.equal(target.penaltyCount.readInt32LEArray()[0], force ? iteration + 1 : 2, "verification/draft must not advance ordinary target penalty history");
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
        assert.equal(sampling.penaltyCount.readInt32LEArray()[0], seen.size + 1);
        const penaltyTokens = Array(8).fill(0);
        [...seen].forEach((id, index) => { penaltyTokens[index] = id; });
        penaltyTokens[seen.size % 8] = expected;
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
