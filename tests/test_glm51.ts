import assert from "node:assert/strict";
import path from "node:path";
import { after, before, describe, it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import type { ChatCache, ChatModel } from "../src/chat_model";
import type { DeviceOps } from "../src/device_ops";
import { executePlan, ExecutionWorkspace } from "../src/execution-workspace";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps } from "../src/glm_ops";
import { ParallelOps } from "../src/parallel_ops";
import { Tensor } from "../src/tensor";
import { PAGE_SIZE } from "../src/paged_kv";
import { PhasedPrefillRunner } from "../src/phased-prefill";

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
        using runner = new PhasedPrefillRunner(model, glm);
        const stateA = ws.planPrefill(model, 2, inputA.map(ids => ids.length), cache);
        stateA.setInput(inputA);
        assert.equal(runner.enqueue(stateA), false);
        sequence0.reportTokens(inputA[0]);
        sequence1.reportTokens(inputA[1]);

        pagedKV.stageSequence(1, 1);
        pagedKV.unstageSequence(2);
        const stateB = ws.planPrefill(model, 2, inputB.map(ids => ids.length), cache);
        stateB.setInput(inputB);
        sequence0.reportTokens(inputB[0]);
        sequence2.reportTokens(inputB[1]);

        assert.equal(runner.enqueue(stateB, (pairedA, hiddenA, pairedB, hiddenB) => {
          using logitsA = pairedA.computeLogits(hiddenA, model);
          using logitsB = pairedB.computeLogits(hiddenB, model);
          using argmaxA = logitsA.argmax();
          using argmaxB = logitsB.argmax();
          tokensA = argmaxA.readInt32LEArray();
          tokensB = argmaxB.readInt32LEArray();
        }), true);
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

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Glm51Model.fromPretrained(glm, SMALL_MODEL_DIR, false, true);
    ws = new ExecutionWorkspace(glm, 1, 128);
    captureManager = new CaptureManager(glm);
    captureManager.disabled = true;
  });

  after(() => {
    captureManager[Symbol.dispose]();
    ws.free();
    model.free();
    glm.free();
  });

  it("rolls back unstarted and parked phased MTP plans", async () => {
    using cache = model.createChatCache(32, 1);
    cache.reset(1);
    const sequence = cache.getPagedKV().sequences[0];

    const unstarted = model.planPrefillMtpChunkPhased(ws, cache, [[1, 2, 3, 4]], [5]);
    assert.equal(sequence.allocLen, 4);
    unstarted[Symbol.dispose]();
    assert.equal(sequence.allocLen, 0);
    ws.resetPlanSlots();

    const runner = new PhasedPrefillRunner(model, glm);
    const parked = model.planPrefillMtpChunkPhased(ws, cache, [[1, 2, 3, 4]], [5]);
    assert.equal(runner.enqueueGenerator(parked.state, parked.generator, undefined, parked), false);
    assert.equal(sequence.allocLen, 4);
    runner[Symbol.dispose]();
    assert.equal(sequence.allocLen, 0);
    await glm.synchronizeAsync();
    ws.assertClear();
    ws.resetPlanSlots();
  });

  it("matches sequential MTP chunks and final draft extension", async () => {
    const inputA = [[1, 2, 3, 4]];
    const inputB = [[5, 6, 7, 8]];
    const nextA = [5];
    const nextB = [9];
    const finalInput = [[9, 10]];
    const topks = [1, 1];

    const run = async (phased: boolean) => {
      using cache = model.createChatCache(32, 1);
      cache.reset(1);
      const sequence = cache.getPagedKV().sequences[0];

      if (phased) {
        using runner = new PhasedPrefillRunner(model, glm);
        const planA = model.planPrefillMtpChunkPhased(ws, cache, inputA, nextA);
        assert.equal(runner.enqueueGenerator(planA.state, planA.generator, undefined, planA), false);
        sequence.reportTokens(inputA[0]);

        const planB = model.planPrefillMtpChunkPhased(ws, cache, inputB, nextB);
        assert.equal(runner.enqueueGenerator(planB.state, planB.generator, undefined, planB), true);
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
        model.planPrefillMtpDraftExtend(ws, cache, mtpInput, topks),
      )).result;
    };

    const sequential = await run(false);
    const phased = await run(true);
    assert.deepStrictEqual(phased.targetTokens, sequential.targetTokens);
    assert.deepStrictEqual(phased.treeTokens, sequential.treeTokens);
  });
});
