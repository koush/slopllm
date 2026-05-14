import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { Qwen3Model } from "../src/qwen3_model";
import { Qwen35Model } from "../src/qwen35_model";
import { ExecutionWorkspace, PagedKVCache } from "../src/paged_kv";
import { Tensor } from "../src/tensor";
import { generateBatchTokens, generateTokens } from "./test_helper";
import { SamplingParams, makeSamplingParams, type ChatModel, type ChatCache } from "../src/chat_model";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const PROMPT1 = [151643, 151644, 151645, 1, 2, 3];
const PROMPT2 = [151643, 151644, 1, 2, 3, 4, 5];
const PROMPT_LONG1 = [151643, 151644, 151645, 1, 2, 3, 4, 5, 6, 7];
const PROMPT_LONG2 = [151643, 151644, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const PROMPT_GRAPH = [151643, 151644, 151645, 1, 2988, 279, 1716, 364];
const EOS_TOKEN_IDS = new Set([151645, 151643]);

describe("Qwen3-0.6B batch tests", () => {
  let glm: GlmOps;
  let model: Qwen3Model;
  let ws: ExecutionWorkspace;

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Qwen3Model.fromPretrained(glm, QWEN3_REPO, 4, 4096);
    ws = new ExecutionWorkspace(glm, 4, 4096);
  });

  after(() => {
    ws[Symbol.dispose]();
    model.free();
    glm.free();
  });

  function makePagedKV(maxBatch = 4, maxPages = 128): PagedKVCache {
    const cfg = model.cfg;
    return new PagedKVCache(glm, cfg.numKeyValueHeads, cfg.headDim, cfg.numHiddenLayers, maxPages, maxBatch);
  }

  it("batch prefill vs single prefill", () => {
    using pagedKV = makePagedKV();
    using singleKV = makePagedKV(1);
    pagedKV.reset(2);
    const batchTokens = ws.forwardEagerPrefill(model, [PROMPT1, PROMPT2], pagedKV);

    singleKV.reset(1);
    const singleToken1 = ws.forwardEagerPrefill(model, [PROMPT1], singleKV)[0];

    singleKV.reset(1);
    const singleToken2 = ws.forwardEagerPrefill(model, [PROMPT2], singleKV)[0];

    assert.equal(batchTokens[0], singleToken1,
      `Seq1 prefill token mismatch: batch=${batchTokens[0]}, single=${singleToken1}`);
    assert.equal(batchTokens[1], singleToken2,
      `Seq2 prefill token mismatch: batch=${batchTokens[1]}, single=${singleToken2}`);
  });

  it("batch prefill paged then decode", () => {
    using pagedKV = makePagedKV();
    pagedKV.reset(2);
    const batchTokens = ws.forwardEagerPrefill(model, [PROMPT1, PROMPT2], pagedKV);
    pagedKV.updateIndptr(ws);

    const decodeTokens = ws.forwardEagerDecode(model, batchTokens, pagedKV);

    assert.equal(typeof decodeTokens[0], "number", `Decode token 0 not a number: ${decodeTokens[0]}`);
    assert.equal(typeof decodeTokens[1], "number", `Decode token 1 not a number: ${decodeTokens[1]}`);
  });

  it("batch prefill append", () => {
    using pagedKV = makePagedKV(1, 256);
    using singleKV = makePagedKV(1, 256);
    const suffix = [4, 5, 6, 7];
    const fullPrompt = [...PROMPT1, ...suffix];

    pagedKV.reset(1);
    const tokensFull = ws.forwardEagerPrefill(model, [fullPrompt], pagedKV);

    pagedKV.reset(1);
    ws.forwardEagerPrefill(model, [PROMPT1], pagedKV);
    pagedKV.updateIndptr(ws);
    const tokensAppend = ws.forwardEagerPrefill(model, [suffix], pagedKV);

    singleKV.reset(1);
    const singleToken = ws.forwardEagerPrefill(model, [fullPrompt], singleKV)[0];

    assert.equal(tokensFull[0], singleToken,
      `Full paged prefill mismatch: paged=${tokensFull[0]}, single=${singleToken}`);
    assert.equal(tokensAppend[0], tokensFull[0],
      `Append prefill mismatch: append=${tokensAppend[0]}, full=${tokensFull[0]}`);
  });

  it("batch prefill truncate append", () => {
    using pagedKV = makePagedKV(1, 256);
    using singleKV = makePagedKV(1, 256);
    const suffix = [4, 5, 6, 7];
    const fullPrompt = [...PROMPT1, ...suffix];

    pagedKV.reset(1);
    ws.forwardEagerPrefill(model, [PROMPT1], pagedKV);
    pagedKV.updateIndptr(ws);
    ws.forwardEagerPrefill(model, [suffix], pagedKV);
    pagedKV.updateIndptr(ws);

    pagedKV.truncate(0, PROMPT1.length);
    pagedKV.updateIndptr(ws);
    const tokensTruncAppend = ws.forwardEagerPrefill(model, [suffix], pagedKV);

    using pagedKV2 = makePagedKV(1, 256);
    pagedKV2.reset(1);
    const tokensFull = ws.forwardEagerPrefill(model, [fullPrompt], pagedKV2);

    assert.equal(tokensTruncAppend[0], tokensFull[0],
      `Truncate+append mismatch: trunc_append=${tokensTruncAppend[0]}, full=${tokensFull[0]}`);

    singleKV.reset(1);
    const singleToken = ws.forwardEagerPrefill(model, [fullPrompt], singleKV)[0];
    assert.equal(tokensTruncAppend[0], singleToken,
      `Truncate+append vs single mismatch: trunc_append=${tokensTruncAppend[0]}, single=${singleToken}`);
  });

  it("batch decode vs single decode", () => {
    using pagedKV = makePagedKV();
    using singleKV = makePagedKV(1);
    using singleWs = new ExecutionWorkspace(glm, 1, 4096);
    pagedKV.reset(2);
    const batchTokens = ws.forwardEagerPrefill(model, [PROMPT1, PROMPT2], pagedKV);
    const token1 = batchTokens[0];
    const token2 = batchTokens[1];

    singleKV.reset(1);
    const singleFirst1 = singleWs.forwardEagerPrefill(model, [PROMPT1], singleKV)[0];
    const singleDecode1 = singleWs.forwardEagerDecode(model, [singleFirst1], singleKV)[0];

    singleKV.reset(1);
    const singleFirst2 = singleWs.forwardEagerPrefill(model, [PROMPT2], singleKV)[0];
    const singleDecode2 = singleWs.forwardEagerDecode(model, [singleFirst2], singleKV)[0];

    const batchDecodeTokens = ws.forwardEagerDecode(model, [token1, token2], pagedKV);

    assert.equal(batchDecodeTokens[0], singleDecode1,
      `Seq1 decode token mismatch: batch=${batchDecodeTokens[0]}, single=${singleDecode1}`);
    assert.equal(batchDecodeTokens[1], singleDecode2,
      `Seq2 decode token mismatch: batch=${batchDecodeTokens[1]}, single=${singleDecode2}`);
  });

  it("batch multi-step decode", () => {
    using pagedKV = makePagedKV();
    pagedKV.reset(2);
    const batchTokens = ws.forwardEagerPrefill(model, [PROMPT1, PROMPT2], pagedKV);
    pagedKV.updateIndptr(ws);

    let current = [batchTokens[0], batchTokens[1]];
    const numSteps = 5;

    for (let step = 0; step < numSteps; step++) {
      current = ws.forwardEagerDecode(model, current, pagedKV);
    }

    assert.equal(current.length, 2);
    assert.equal(typeof current[0], "number");
    assert.equal(typeof current[1], "number");
  });

  it("batch generate vs single generate", () => {
    using pagedKV = makePagedKV();
    using singleKV = makePagedKV(1, 256);
    const maxNewTokens = 20;
    const batchGenerated = generateBatchTokens(model, ws, pagedKV, [PROMPT_LONG1, PROMPT_LONG2], maxNewTokens, EOS_TOKEN_IDS);

    singleKV.reset(1);
    const single1 = [...generateTokens(model, ws, singleKV, PROMPT_LONG1, maxNewTokens, EOS_TOKEN_IDS)];

    singleKV.reset(1);
    const single2 = [...generateTokens(model, ws, singleKV, PROMPT_LONG2, maxNewTokens, EOS_TOKEN_IDS)];

    assert.ok(batchGenerated[0].length > 0, "Seq1 generated no tokens");
    assert.ok(batchGenerated[1].length > 0, "Seq2 generated no tokens");

    const match1 = batchGenerated[0].slice(0, 3).every((t: number, i: number) => t === single1[i]);
    const match2 = batchGenerated[1].slice(0, 3).every((t: number, i: number) => t === single2[i]);

    assert.ok(match1,
      `Seq1 first 3 tokens mismatch: batch=${batchGenerated[0].slice(0, 3)}, single=${single1.slice(0, 3)}`);
    assert.ok(match2,
      `Seq2 first 3 tokens mismatch: batch=${batchGenerated[1].slice(0, 3)}, single=${single2.slice(0, 3)}`);
  });

  it("cuda graph decode", () => {
    using gws = new ExecutionWorkspace(glm, 4, 4096);
    using pagedKV = makePagedKV();
    const prompt = PROMPT_GRAPH;

    pagedKV.reset(1);
    const tokens = gws.forwardEagerPrefill(model, [prompt], pagedKV);

    const stateRef = gws.planDecode(model, 1, pagedKV, true);
    stateRef.prepareInput([[tokens[0]]]);
    gws.decodeStep(stateRef, model);
    gws.forwardInput(stateRef);
    const hiddenStatesRef = model.forward(stateRef);
    const logitsRef = stateRef.computeLogits(hiddenStatesRef, model);
    using argmaxRef = logitsRef.argmax();
    const tokensRef = argmaxRef.readInt32LEArray();

    pagedKV.reset(1);
    const tokens2 = gws.forwardEagerPrefill(model, [prompt], pagedKV);
    const state = gws.planDecode(model, 1, pagedKV, true);
    state.prepareInput([[tokens2[0]]]);

    glm.graphBeginCapture();
    gws.decodeStep(state, model);
    gws.forwardInput(state);
    const captureHiddenStates = model.forward(state);
    const captureLogits = state.computeLogits(captureHiddenStates, model);
    const captureArgmax = captureLogits.argmax();
    const graph = glm.graphEndCapture();
    gws.freeze();
    const graphExec = glm.graphInstantiate(graph);
    glm.graphDestroy(graph);

    glm.graphLaunch(graphExec);
    glm.synchronize();

    const tokensReplay = captureArgmax.readInt32LEArray();
    assert.deepEqual(tokensReplay, tokensRef,
      `Graph replay mismatch: replay=${tokensReplay}, ref=${tokensRef}`);

    glm.graphExecDestroy(graphExec);
  });

  it("cuda graph multi-step decode", () => {
    using gws = new ExecutionWorkspace(glm, 4, 4096);
    using pagedKV = makePagedKV();
    const numSteps = 10;
    const prompt = PROMPT_GRAPH;

    pagedKV.reset(1);
    let tokens = gws.forwardEagerPrefill(model, [prompt], pagedKV);

    const refTokens: number[] = [];
    let current = tokens[0];
    for (let step = 0; step < numSteps; step++) {
      const state = gws.planDecode(model, 1, pagedKV, true);
      state.prepareInput([[current]]);
      gws.decodeStep(state, model);
      gws.forwardInput(state);
      const hiddenStates = model.forward(state);
      const lastLogits = state.computeLogits(hiddenStates, model);
      using argmaxResult = lastLogits.argmax();
      current = argmaxResult.readInt32LEArray()[0];
      refTokens.push(current);
    }

    pagedKV.reset(1);
    tokens = gws.forwardEagerPrefill(model, [prompt], pagedKV);
    current = tokens[0];

    let graphExec: number | null = null;
    let warmupRemaining = 3;
    let capturing = false;
    let captureArgmax: any = null;

    for (let step = 0; step < numSteps; step++) {
      const state = gws.planDecode(model, 1, pagedKV, true);
      state.prepareInput([[current]]);

      if (graphExec === null) {
        if (warmupRemaining === 0 && !capturing) {
          capturing = true;
          glm.graphBeginCapture();
        }

    gws.decodeStep(state, model);
        gws.forwardInput(state);
        const hiddenStates = model.forward(state);
        captureArgmax = state.computeLogits(hiddenStates, model).argmax();

        if (capturing) {
          const graph = glm.graphEndCapture();
          gws.freeze();
          graphExec = glm.graphInstantiate(graph);
          glm.graphDestroy(graph);
          capturing = false;
          warmupRemaining = 0;
        }

        if (warmupRemaining > 0) warmupRemaining--;
      }

      if (graphExec !== null) {
        glm.graphLaunch(graphExec);
        glm.synchronize();
      }

      current = captureArgmax.readInt32LEArray()[0];
      const expected = refTokens[step];
      assert.equal(current, expected,
        `Step ${step} mismatch: ${current} != ${expected}`);
    }

    if (graphExec !== null) {
      glm.graphExecDestroy(graphExec);
    }
  });

  it("batch sampling matches sequential sampling", () => {
    using pagedKV = model.createChatCache() as PagedKVCache;
    const greedy: SamplingParams = makeSamplingParams({
      temperature: 0, topP: 1.0, topK: 0,
      repetitionPenalty: 1.0, presencePenalty: 0, repetitionPenaltyWindow: 64,
    });
    const sampling: SamplingParams = makeSamplingParams({
      temperature: 0.8, topP: 0.95, topK: 20,
      repetitionPenalty: 1.05, presencePenalty: 0.0, repetitionPenaltyWindow: 64,
    });

    pagedKV.reset(1);
    const state = ws.planPrefill(model, 1, [PROMPT_GRAPH.length], pagedKV);
    state.prepareInput([PROMPT_GRAPH]);
    ws.forwardInput(state);
    const hiddenStates = model.forward(state);
    const logits = state.computeLogits(hiddenStates, model);
    using argmaxOut = logits.argmax();
    const tokens = argmaxOut.readInt32LEArray();
    pagedKV.updateIndptr(ws);

    const firstToken = tokens[0];
    const history = [...PROMPT_GRAPH, firstToken];

    const greedySingle = logits.sampleTokenGPU(greedy, history).readInt32LEArray()[0];

    const batchResults = logits.sampleBatchGPU([greedy, sampling], [history, history]).readInt32LEArray();

    assert.equal(batchResults[0], greedySingle,
      `Batch greedy[0] != sequential greedy: ${batchResults[0]} != ${greedySingle}`);
    assert.ok(Number.isInteger(batchResults[1]),
      `Sampling token should be integer: ${batchResults[1]}`);
    assert.ok(batchResults[1] >= 0 && batchResults[1] < model.cfg.vocabSize,
      `Sampling token out of range: ${batchResults[1]}`);
  });

  it("batch sampling with different histories", () => {
    using pagedKV = model.createChatCache(4) as PagedKVCache;
    const greedy: SamplingParams = makeSamplingParams({
      temperature: 0, topP: 1.0, topK: 0,
      repetitionPenalty: 1.0, presencePenalty: 0, repetitionPenaltyWindow: 64,
    });

    pagedKV.reset(2);
    const state = ws.planPrefill(model, 2, [PROMPT1.length, PROMPT2.length], pagedKV);
    state.prepareInput([PROMPT1, PROMPT2]);
    ws.forwardInput(state);
    const hiddenStates = model.forward(state);
    const logits = state.computeLogits(hiddenStates, model);
    using argmaxOut2 = logits.argmax();
    const tokens = argmaxOut2.readInt32LEArray();

    const history1 = [...PROMPT1, tokens[0]];
    const history2 = [...PROMPT2, tokens[1]];

    const batchResults = logits.sampleBatchGPU([greedy, greedy], [history1, history2]).readInt32LEArray();

    assert.equal(batchResults[0], tokens[0],
      `Batch greedy[0] != argmax: ${batchResults[0]} != ${tokens[0]}`);
    assert.equal(batchResults[1], tokens[1],
      `Batch greedy[1] != argmax: ${batchResults[1]} != ${tokens[1]}`);
  });

  function chunkedPrefill(model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache, inputIds: number[], chunkSizes: number[]): number[] {
    let offset = 0;
    let logits: ReturnType<Tensor["argmax"]> | null = null;
    for (let i = 0; i < chunkSizes.length; i++) {
      const chunk = inputIds.slice(offset, offset + chunkSizes[i]);
      offset += chunkSizes[i];
      const state = ws.planPrefill(model, 1, [chunk.length], cache);
      state.prepareInput([chunk]);
      ws.forwardInput(state);
      const hiddenStates = model.forward(state);
      if (i < chunkSizes.length - 1) {
        hiddenStates[Symbol.dispose]();
      } else {
        logits = state.computeLogits(hiddenStates, model);
        hiddenStates[Symbol.dispose]();
        state.finishPrefill();
      }
    }
    using argmaxOut = logits!.argmax();
    return argmaxOut.readInt32LEArray();
  }

  it("chunked prefill: two even halves", () => {
    using pagedKV = makePagedKV(1, 256);
    using pagedKV2 = makePagedKV(1, 256);
    const fullPrompt = PROMPT_LONG1;
    const mid = Math.floor(fullPrompt.length / 2);
    const firstHalf = fullPrompt.slice(0, mid);
    const secondHalf = fullPrompt.slice(mid);

    pagedKV.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], pagedKV);

    pagedKV2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, pagedKV2, fullPrompt, [firstHalf.length, secondHalf.length]);

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Chunked prefill mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
  });

  it("chunked prefill: uneven split", () => {
    using pagedKV = makePagedKV(1, 256);
    using pagedKV2 = makePagedKV(1, 256);
    const fullPrompt = PROMPT_LONG1;

    pagedKV.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], pagedKV);

    pagedKV2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, pagedKV2, fullPrompt, [3, fullPrompt.length - 3]);

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Chunked prefill mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
  });

  it("chunked prefill: three chunks", () => {
    using pagedKV = makePagedKV(1, 256);
    using pagedKV2 = makePagedKV(1, 256);
    const fullPrompt = PROMPT_LONG1;

    pagedKV.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], pagedKV);

    pagedKV2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, pagedKV2, fullPrompt, [3, 3, fullPrompt.length - 6]);

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Chunked prefill mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
  });

  it("chunked prefill + decode matches full prefill + decode", () => {
    using pagedKV = makePagedKV(1, 256);
    using pagedKV2 = makePagedKV(1, 256);
    const fullPrompt = PROMPT_LONG1;
    const mid = Math.floor(fullPrompt.length / 2);

    pagedKV.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], pagedKV);
    pagedKV.updateIndptr(ws);
    const fullDecode = ws.forwardEagerDecode(model, [fullTokens[0]], pagedKV)[0];

    pagedKV2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, pagedKV2, fullPrompt, [mid, fullPrompt.length - mid]);
    pagedKV2.updateIndptr(ws);
    const chunkedDecode = ws.forwardEagerDecode(model, [chunkedTokens[0]], pagedKV2)[0];

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Chunked prefill token mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
    assert.equal(chunkedDecode, fullDecode,
      `Chunked decode token mismatch: chunked=${chunkedDecode}, full=${fullDecode}`);
  });
});

describe("Qwen3.5-0.8B chunked prefill tests", () => {
  let glm: GlmOps;
  let model: Qwen35Model;
  let ws: ExecutionWorkspace;
  const PROMPT = [151643, 151644, 151645, 1, 2, 3, 4, 5, 6, 7];

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Qwen35Model.fromPretrained(glm, "Qwen/Qwen3.5-0.8B", 1, 128);
    ws = new ExecutionWorkspace(glm, 1, 128);
  });

  after(() => {
    ws[Symbol.dispose]();
    model.free();
    glm.free();
  });

  function chunkedPrefill(model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache, inputIds: number[], chunkSizes: number[]): number[] {
    let offset = 0;
    let logits: ReturnType<Tensor["argmax"]> | null = null;
    for (let i = 0; i < chunkSizes.length; i++) {
      const chunk = inputIds.slice(offset, offset + chunkSizes[i]);
      offset += chunkSizes[i];
      const state = ws.planPrefill(model, 1, [chunk.length], cache);
      state.prepareInput([chunk]);
      ws.forwardInput(state);
      const hiddenStates = model.forward(state);
      if (i < chunkSizes.length - 1) {
        hiddenStates[Symbol.dispose]();
      } else {
        logits = state.computeLogits(hiddenStates, model);
        hiddenStates[Symbol.dispose]();
        state.finishPrefill();
      }
    }
    using argmaxOut = logits!.argmax();
    return argmaxOut.readInt32LEArray();
  }

  it("chunked prefill: two even halves", () => {
    using cache1 = model.createChatCache(128);
    using cache2 = model.createChatCache(128);
    const fullPrompt = PROMPT;
    const mid = Math.floor(fullPrompt.length / 2);

    cache1.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], cache1);

    cache2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, cache2, fullPrompt, [mid, fullPrompt.length - mid]);

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Qwen3.5 chunked prefill mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
  });

  it("chunked prefill: uneven split", () => {
    using cache1 = model.createChatCache(128);
    using cache2 = model.createChatCache(128);
    const fullPrompt = PROMPT;

    cache1.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], cache1);

    cache2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, cache2, fullPrompt, [3, fullPrompt.length - 3]);

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Qwen3.5 chunked prefill mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
  });

  it("chunked prefill + decode matches full prefill + decode", () => {
    using cache1 = model.createChatCache(128);
    using cache2 = model.createChatCache(128);
    const fullPrompt = PROMPT;
    const mid = Math.floor(fullPrompt.length / 2);

    cache1.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], cache1);
    cache1.getPagedKV().updateIndptr(ws);
    const fullDecode = ws.forwardEagerDecode(model, [fullTokens[0]], cache1)[0];

    cache2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, cache2, fullPrompt, [mid, fullPrompt.length - mid]);
    cache2.getPagedKV().updateIndptr(ws);
    const chunkedDecode = ws.forwardEagerDecode(model, [chunkedTokens[0]], cache2)[0];

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Qwen3.5 chunked prefill token mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
    assert.equal(chunkedDecode, fullDecode,
      `Qwen3.5 chunked decode token mismatch: chunked=${chunkedDecode}, full=${fullDecode}`);
  });
});
