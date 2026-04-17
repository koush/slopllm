import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { Qwen3Model } from "../src/qwen3_model";
import { PagedKVCache, WorkspaceBuffers } from "../src/paged_kv";
import { generateBatchTokens, generateTokens } from "./test_helper";
import { SamplingParams, makeSamplingParams } from "../src/chat_model";

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
  let ws: WorkspaceBuffers;

  before(() => {
    process.env.CUDA_VISIBLE_DEVICES = process.env.GLM_GPU ?? "0";
    glm = new GlmOps(0);
    model = Qwen3Model.fromPretrained(glm, QWEN3_REPO, 4, 4096);
    ws = new WorkspaceBuffers(glm);
  });

  after(() => {
    ws.free();
    model.free();
  });

  function makePagedKV(maxBatch = 4, maxPages = 128): PagedKVCache {
    const cfg = model.cfg;
    return new PagedKVCache(glm, cfg.numKeyValueHeads, cfg.headDim, cfg.numHiddenLayers, maxPages, maxBatch);
  }

  it("batch prefill vs single prefill", () => {
    const pagedKV = makePagedKV();
    const singleKV = makePagedKV(1);
    try {
      pagedKV.reset(2);
      const batchTokens = model.forwardEager([PROMPT1, PROMPT2], ws, pagedKV);

      singleKV.reset(1);
      const singleToken1 = model.forwardEager([PROMPT1], ws, singleKV)[0];

      singleKV.reset(1);
      const singleToken2 = model.forwardEager([PROMPT2], ws, singleKV)[0];

      assert.equal(batchTokens[0], singleToken1,
        `Seq1 prefill token mismatch: batch=${batchTokens[0]}, single=${singleToken1}`);
      assert.equal(batchTokens[1], singleToken2,
        `Seq2 prefill token mismatch: batch=${batchTokens[1]}, single=${singleToken2}`);
    } finally {
      pagedKV.free();
      singleKV.free();
    }
  });

  it("batch prefill paged then decode", () => {
    const pagedKV = makePagedKV();
    try {
      pagedKV.reset(2);
      const batchTokens = model.forwardEager([PROMPT1, PROMPT2], ws, pagedKV);
      pagedKV.updateIndptr();

      const decodeTokens = model.forwardEagerDecode(batchTokens, ws, pagedKV);

      assert.equal(typeof decodeTokens[0], "number", `Decode token 0 not a number: ${decodeTokens[0]}`);
      assert.equal(typeof decodeTokens[1], "number", `Decode token 1 not a number: ${decodeTokens[1]}`);
    } finally {
      pagedKV.free();
    }
  });

  it("batch prefill append", () => {
    const pagedKV = makePagedKV(1, 256);
    const singleKV = makePagedKV(1, 256);
    try {
      const suffix = [4, 5, 6, 7];
      const fullPrompt = [...PROMPT1, ...suffix];

      pagedKV.reset(1);
      const tokensFull = model.forwardEager([fullPrompt], ws, pagedKV);

      pagedKV.reset(1);
      model.forwardEager([PROMPT1], ws, pagedKV);
      pagedKV.updateIndptr();
      const tokensAppend = model.forwardEager([suffix], ws, pagedKV);

      singleKV.reset(1);
      const singleToken = model.forwardEager([fullPrompt], ws, singleKV)[0];

      assert.equal(tokensFull[0], singleToken,
        `Full paged prefill mismatch: paged=${tokensFull[0]}, single=${singleToken}`);
      assert.equal(tokensAppend[0], tokensFull[0],
        `Append prefill mismatch: append=${tokensAppend[0]}, full=${tokensFull[0]}`);
    } finally {
      pagedKV.free();
      singleKV.free();
    }
  });

  it("batch prefill truncate append", () => {
    const pagedKV = makePagedKV(1, 256);
    const singleKV = makePagedKV(1, 256);
    try {
      const suffix = [4, 5, 6, 7];
      const fullPrompt = [...PROMPT1, ...suffix];

      pagedKV.reset(1);
      model.forwardEager([PROMPT1], ws, pagedKV);
      pagedKV.updateIndptr();
      model.forwardEager([suffix], ws, pagedKV);
      pagedKV.updateIndptr();

      pagedKV.truncate(0, PROMPT1.length);
      pagedKV.updateIndptr();
      const tokensTruncAppend = model.forwardEager([suffix], ws, pagedKV);

      const pagedKV2 = makePagedKV(1, 256);
      try {
        pagedKV2.reset(1);
        const tokensFull = model.forwardEager([fullPrompt], ws, pagedKV2);

        assert.equal(tokensTruncAppend[0], tokensFull[0],
          `Truncate+append mismatch: trunc_append=${tokensTruncAppend[0]}, full=${tokensFull[0]}`);
      } finally {
        pagedKV2.free();
      }

      singleKV.reset(1);
      const singleToken = model.forwardEager([fullPrompt], ws, singleKV)[0];
      assert.equal(tokensTruncAppend[0], singleToken,
        `Truncate+append vs single mismatch: trunc_append=${tokensTruncAppend[0]}, single=${singleToken}`);
    } finally {
      pagedKV.free();
      singleKV.free();
    }
  });

  it("batch decode vs single decode", () => {
    const pagedKV = makePagedKV();
    const singleKV = makePagedKV(1);
    try {
      pagedKV.reset(2);
      const batchTokens = model.forwardEager([PROMPT1, PROMPT2], ws, pagedKV);
      const token1 = batchTokens[0];
      const token2 = batchTokens[1];

      singleKV.reset(1);
      const singleFirst1 = model.forwardEager([PROMPT1], ws, singleKV)[0];
      const singleDecode1 = model.forwardEagerDecode([singleFirst1], ws, singleKV)[0];

      singleKV.reset(1);
      const singleFirst2 = model.forwardEager([PROMPT2], ws, singleKV)[0];
      const singleDecode2 = model.forwardEagerDecode([singleFirst2], ws, singleKV)[0];

      const batchDecodeTokens = model.forwardEagerDecode([token1, token2], ws, pagedKV);

      assert.equal(batchDecodeTokens[0], singleDecode1,
        `Seq1 decode token mismatch: batch=${batchDecodeTokens[0]}, single=${singleDecode1}`);
      assert.equal(batchDecodeTokens[1], singleDecode2,
        `Seq2 decode token mismatch: batch=${batchDecodeTokens[1]}, single=${singleDecode2}`);
    } finally {
      pagedKV.free();
      singleKV.free();
    }
  });

  it("batch multi-step decode", () => {
    const pagedKV = makePagedKV();
    try {
      pagedKV.reset(2);
      const batchTokens = model.forwardEager([PROMPT1, PROMPT2], ws, pagedKV);
      pagedKV.updateIndptr();

      let current = [batchTokens[0], batchTokens[1]];
      const numSteps = 5;

      for (let step = 0; step < numSteps; step++) {
        current = model.forwardEagerDecode(current, ws, pagedKV);
      }

      assert.equal(current.length, 2);
      assert.equal(typeof current[0], "number");
      assert.equal(typeof current[1], "number");
    } finally {
      pagedKV.free();
    }
  });

  it("batch generate vs single generate", () => {
    const pagedKV = makePagedKV();
    const singleKV = makePagedKV(1, 256);
    const maxNewTokens = 20;
    try {
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
    } finally {
      pagedKV.free();
      singleKV.free();
    }
  });

  it("cuda graph decode", () => {
    const pagedKV = makePagedKV();
    try {
      const prompt = PROMPT_GRAPH;

      pagedKV.reset(1);
      const tokens = model.forwardEager([prompt], ws, pagedKV);
      pagedKV.updateIndptr();

      const stateRef = model.planDecode([tokens[0]], ws, pagedKV, true);
      model.forwardDecode(stateRef, ws, pagedKV);
      const tokensRef = model.readDecode(stateRef);

      pagedKV.reset(1);
      const tokens2 = model.forwardEager([prompt], ws, pagedKV);
      pagedKV.updateIndptr();
      const state = model.planDecode([tokens2[0]], ws, pagedKV, true);

      glm.graphBeginCapture();
      model.forwardDecode(state, ws, pagedKV);
      const graph = glm.graphEndCapture();
      assert.ok(graph, "graph_end_capture returned null");
      const graphExec = glm.graphInstantiate(graph);
      assert.ok(graphExec, "graph_instantiate returned null");

      pagedKV.reset(1);
      const tokens3 = model.forwardEager([prompt], ws, pagedKV);
      pagedKV.updateIndptr();
      const state2 = model.planDecode([tokens3[0]], ws, pagedKV, true);

      glm.graphLaunch(graphExec);
      glm.synchronize();

      const tokensReplay = model.readDecode(state2);
      assert.deepEqual(tokensReplay, tokensRef,
        `Graph replay mismatch: replay=${tokensReplay}, ref=${tokensRef}`);

      glm.graphExecDestroy(graphExec);
      glm.graphDestroy(graph);
    } finally {
      pagedKV.free();
    }
  });

  it("cuda graph multi-step decode", () => {
    const pagedKV = makePagedKV();
    const numSteps = 10;
    try {
      const prompt = PROMPT_GRAPH;

      pagedKV.reset(1);
      let tokens = model.forwardEager([prompt], ws, pagedKV);
      pagedKV.updateIndptr();

      const refTokens: number[] = [];
      let current = tokens[0];
      for (let step = 0; step < numSteps; step++) {
        const state = model.planDecode([current], ws, pagedKV, true);
        model.forwardDecode(state, ws, pagedKV);
        current = model.readDecode(state)[0];
        refTokens.push(current);
      }

      pagedKV.reset(1);
      tokens = model.forwardEager([prompt], ws, pagedKV);
      pagedKV.updateIndptr();

      current = tokens[0];
      const warmupState = model.planDecode([current], ws, pagedKV, true);
      model.forwardDecode(warmupState, ws, pagedKV);
      current = model.readDecode(warmupState)[0];
      assert.equal(current, refTokens[0], `Warmup mismatch: ${current} != ${refTokens[0]}`);

      const state = model.planDecode([current], ws, pagedKV, true);
      glm.graphBeginCapture();
      model.forwardDecode(state, ws, pagedKV);
      const graph = glm.graphEndCapture();
      assert.ok(graph, "graph_end_capture returned null");
      const graphExec = glm.graphInstantiate(graph);
      assert.ok(graphExec, "graph_instantiate returned null");
      glm.graphDestroy(graph);

      glm.graphLaunch(graphExec);
      glm.synchronize();
      current = model.readDecode(state)[0];
      const graphTokens: number[] = [refTokens[0], current];
      assert.equal(current, refTokens[1], `Replay step 1 mismatch: ${current} != ${refTokens[1]}`);

      for (let step = 2; step < numSteps; step++) {
        const s = model.planDecode([current], ws, pagedKV, true);
        glm.graphLaunch(graphExec);
        glm.synchronize();
        current = model.readDecode(s)[0];
        graphTokens.push(current);
        assert.equal(current, refTokens[step],
          `Replay step ${step} mismatch: ${current} != ${refTokens[step]}`);
      }

      assert.deepEqual(graphTokens, refTokens,
        `Token sequence mismatch: graph=${graphTokens}, ref=${refTokens}`);

      glm.graphExecDestroy(graphExec);
    } finally {
      pagedKV.free();
    }
  });

  it("batch sampling matches sequential sampling", () => {
    const pagedKV = model.createChatCache() as PagedKVCache;
    try {
      const greedy: SamplingParams = makeSamplingParams({
        temperature: 0, topP: 1.0, topK: 0,
        repetitionPenalty: 1.0, presencePenalty: 0, repetitionPenaltyWindow: 64,
      });
      const sampling: SamplingParams = makeSamplingParams({
        temperature: 0.8, topP: 0.95, topK: 20,
        repetitionPenalty: 1.05, presencePenalty: 0.0, repetitionPenaltyWindow: 64,
      });

      pagedKV.reset(1);
      const tokens = model.forwardEager([PROMPT_GRAPH], ws, pagedKV);
      pagedKV.updateIndptr();

      const firstToken = tokens[0];
      const history = [...PROMPT_GRAPH, firstToken];

      const greedySingle = model.sampleTokenGPU(greedy, history);

      const batchResults = model.sampleBatchGPU([greedy, sampling], [history, history]);

      assert.equal(batchResults[0], greedySingle,
        `Batch greedy[0] != sequential greedy: ${batchResults[0]} != ${greedySingle}`);
      assert.ok(Number.isInteger(batchResults[1]),
        `Sampling token should be integer: ${batchResults[1]}`);
      assert.ok(batchResults[1] >= 0 && batchResults[1] < model.cfg.vocabSize,
        `Sampling token out of range: ${batchResults[1]}`);
    } finally {
      pagedKV.free();
    }
  });

  it("batch sampling with different histories", () => {
    const pagedKV = model.createChatCache(4) as PagedKVCache;
    try {
      const greedy: SamplingParams = makeSamplingParams({
        temperature: 0, topP: 1.0, topK: 0,
        repetitionPenalty: 1.0, presencePenalty: 0, repetitionPenaltyWindow: 64,
      });

      pagedKV.reset(2);
      const tokens = model.forwardEager([PROMPT1, PROMPT2], ws, pagedKV);

      const history1 = [...PROMPT1, tokens[0]];
      const history2 = [...PROMPT2, tokens[1]];

      const batchResults = model.sampleBatchGPU([greedy, greedy], [history1, history2]);

      assert.equal(batchResults[0], tokens[0],
        `Batch greedy[0] != argmax: ${batchResults[0]} != ${tokens[0]}`);
      assert.equal(batchResults[1], tokens[1],
        `Batch greedy[1] != argmax: ${batchResults[1]} != ${tokens[1]}`);
    } finally {
      pagedKV.free();
    }
  });
});
