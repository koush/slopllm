import assert from "node:assert/strict";
import path from "node:path";
import { after, afterEach, before, beforeEach, describe, it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import type { ChatCache, ChatModel, TokenSelector, SamplingParams } from "../src/chat_model";
import type { DeviceOps } from "../src/device_ops";
import { executePlan, ExecutionWorkspace, type ExecutionPlan } from "../src/execution-workspace";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { ParallelOps } from "../src/parallel_ops";
import { Tensor } from "../src/tensor";
import { PAGE_SIZE } from "../src/paged_kv";
import { PhasedPrefillRunner, splitRaggedInput } from "../src/phased-prefill";
import { SamplingWorkspace } from "../src/sampling";
import { UsingHolder } from "../src/using-holder";
import { WorkspaceBase } from "../src/workspace";

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
        model.planPrefillMtpDraftExtend(ws, cache, mtpInput, topks),
      )).result;
    };

    const sequential = await run(false);
    const phased = await run(true);
    assert.deepStrictEqual(phased.targetTokens, sequential.targetTokens);
    assert.deepStrictEqual(phased.treeTokens, sequential.treeTokens);
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

  it("linear speculative greedy matches the existing plan across verification iterations", async () => {
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
      let draft = (await executePlan(manager, ws,
        model.planPrefillMtpDraftExtend(ws, cache, model.prepareMtpInput(cache, [prompt]), topks,
          linear ? target : undefined))).result;
      sequence.reportTokens([...prompt, ...draft.targetTokens]);
      const outputs = [{ targetTokens: draft.targetTokens, treeTokens: draft.treeTokens, tokens: [] as number[][], numAccepted: [] as number[] }];
      for (let iteration = 0; iteration < 8; iteration++) {
        const originalAllocLen = sequence.allocLen;
        const step = (await executePlan(manager, ws,
          model.planTargetVerification(ws, cache, draft, linear ? target : undefined))).result;
        assert.deepStrictEqual(step.tokens[0], [...draft.treeTokens[0].slice(0, step.numAccepted[0]), step.draft.targetTokens[0]]);
        assert.equal(sequence.allocLen, originalAllocLen + step.numAccepted[0] + 1);
        sequence.reportTokens(step.tokens[0]);
        assert.equal(sequence.reportedTokenCount(), sequence.allocLen + 1);
        draft = step.draft;
        outputs.push({ targetTokens: draft.targetTokens, treeTokens: draft.treeTokens, tokens: step.tokens, numAccepted: step.numAccepted });
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
    const causalChunk = (tokens: number[], next: number): Buffer => {
      using tracking = referenceWs.startTracking();
      const start = reference.getPagedKV().sequences[0].allocLen;
      // Match the verifier's four-row kernel dispatch without its custom mask,
      // rejected tokens, sampling, or commit machinery. Future zero rows cannot
      // condition the causal prefix; truncate them after the independent forward.
      const padding = Array(Math.max(0, topks.length + 1 - tokens.length)).fill(0);
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
    let retainedSeed: Buffer;
    // Observe the retained verification row before the draft phase mutates seed.
    function* observeSeed<T>(plan: ExecutionPlan<T>): ExecutionPlan<T> {
      try {
        let step = plan.next();
        while (!step.done) {
          if (step.value.timingName === "draft") {
            using seed = step.value.inputs.seed.narrow(0, 1);
            retainedSeed = readBytes(seed);
          }
          step = plan.next(yield step.value);
        }
        return step.value;
      } finally {
        plan.return(undefined as never);
      }
    }
    let draft = (await executePlan(manager, ws,
      model.planPrefillMtpDraftExtend(ws, cache, model.prepareMtpInput(cache, [prompt]), topks,
        target))).result;
    if (force) target.updateSampler(Array.from({ length: topks.length + 1 }, () => stochastic), Array.from({ length: topks.length + 1 }, () => []));
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
      finishDraft: () => target.finishDraft(),
      prepareVerification: proposal => {
        target.prepareVerification(proposal);
        forcedPrefix.h2d(Buffer.from(new Int32Array(proposal.treeTokens[0]).buffer));
        forcedCount.h2d(Buffer.from(new Int32Array([forcedAccepted]).buffer));
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

    for (let iteration = 0; iteration < 18; iteration++) {
      forcedAccepted = iteration % topks.length;
      assert.ok(draft.proposal, "draft must retain real sampled q snapshots");
      const snapshots = [...draft.proposal.probabilities, ...draft.proposal.tokenIds].map(buf => Buffer.from(buf));
      const originalAllocLen = sequence.allocLen;
      const history = sequence.getTokenIds();
      const step = (await executePlan(manager, ws,
        observeSeed(model.planTargetVerification(ws, cache, draft, forcedSampler)))).result;
      if (force) assert.deepStrictEqual(step.numAccepted, [forcedAccepted]);
      assert.equal(step.numDraftTokens, topks.length);
      assert.deepStrictEqual(step.tokens[0], [...draft.treeTokens[0].slice(0, step.numAccepted[0]), step.draft.targetTokens[0]]);
      assert.equal(sequence.allocLen, originalAllocLen + step.numAccepted[0] + 1);
      sequence.reportTokens(step.tokens[0]);
      assert.deepStrictEqual(sequence.getTokenIds(), [...history, ...step.tokens[0]]);
      assert.equal(sequence.reportedTokenCount(), sequence.allocLen + 1);
      assert.deepStrictEqual([...draft.proposal.probabilities, ...draft.proposal.tokenIds], snapshots);
      if (draft.proposal.device) {
        assert.equal(step.draft.proposal!.device!.owner, draft.proposal.device.owner);
        assert.equal(step.draft.proposal!.device!.generation, draft.proposal.device.generation + 1);
      }
      const expectedSeed = causalChunk([draft.targetTokens[0], ...draft.treeTokens[0].slice(0, step.numAccepted[0])], step.draft.targetTokens[0]);
      assertCaches();
      assert.deepEqual(retainedSeed!, expectedSeed, `retained MTP seed after acceptance=${step.numAccepted[0]}`);
      assert.ok(bf16BytesToF32(expectedSeed).every(Number.isFinite));
      draft = step.draft;
      assert.equal(target.batchSize, force ? topks.length + 1 : 1);
      assert.equal(target.penaltyCount.readInt32LEArray()[0], force ? iteration + 1 : 1, "verification/draft must not advance ordinary target penalty history");
      target.assertClear();
    }

    // Only consume the tested cache's outstanding token after all verify loops.
    const cachedNext = ws.forwardEagerDecode(model, draft.targetTokens, cache);
    const freshNext = referenceWs.forwardEagerDecode(model, draft.targetTokens, reference);
    assert.deepStrictEqual(cachedNext, freshNext, "target KV after rejection differs from fresh history");
    if (graphs) {
      assert.ok(verifyRuns < 18, "verification must replay without calling the JS sampler");
      assert.ok(draftRuns < 18 * topks.length, "draft must replay without calling the JS sampler");
      for (const phase of ["glm51-mtp-verify", "glm51-mtp-draft"]) {
        assert.ok([...manager.captured].some(([key, entry]) => key.startsWith(phase) && entry.graphExec !== null && entry.capturedWorkspaces.has(target)));
      }
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
