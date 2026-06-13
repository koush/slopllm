import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";
import { SamplingParams, makeSamplingParams, type ChatCache, type ChatModel } from "../src/chat_model";
import { GlmOps } from "../src/glm_ops";
import { Qwen35Model } from "../src/qwen35_model";
import { Qwen3Model } from "../src/qwen3_model";
import { Tensor } from "../src/tensor";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { PagedKVCache } from "../src/paged_kv";
import { generateBatchTokens, generateTokens } from "./test_helper";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "../src/model_path";

import { PAGE_SIZE } from "../src/paged_kv";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";

function tokenizePrompt(tokenizer: any, prompt: string): number[] {
  const messages = [{ role: "user" as const, content: prompt }];
  const result = tokenizer.apply_chat_template(messages as any, {
    tokenize: true,
    add_generation_prompt: true,
    return_tensor: false,
    return_dict: true,
    tokenizer_kwargs: { enable_thinking: false },
  }) as { input_ids: number[] | number[][] };
  return (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
}

function makeLongPrompt(length: number, prefix: number[] = [151643, 151644, 151645]): number[] {
  return [...prefix, ...Array.from({ length: length - prefix.length }, (_, i) => 100 + i)];
}

describe("Qwen3-0.6B batch tests", () => {
  let glm: GlmOps;
  let model: Qwen3Model;
  let ws: ExecutionWorkspace;
  let tokenizer: any;
  let PROMPT1: number[];
  let PROMPT2: number[];
  let PROMPT_LONG1: number[];
  let PROMPT_LONG2: number[];
  let PROMPT_GRAPH: number[];
  const EOS_TOKEN_IDS = new Set([151645, 151643]);

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Qwen3Model.fromPretrained(glm, QWEN3_REPO);
    ws = new ExecutionWorkspace(glm, 4, 4096);
    tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(QWEN3_REPO), { local_files_only: true });
    PROMPT1 = tokenizePrompt(tokenizer, "Hi");
    PROMPT2 = tokenizePrompt(tokenizer, "Hello");
    PROMPT_LONG1 = tokenizePrompt(tokenizer, "What is the capital of France?");
    PROMPT_LONG2 = tokenizePrompt(tokenizer, "What is the capital of Japan?");
    PROMPT_GRAPH = tokenizePrompt(tokenizer, "The capital of France is");
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
    pagedKV.reportTokens(0, fullPrompt);

    pagedKV.reset(1);
    const suffix1 = pagedKV.prefixMatch(0, PROMPT1);
    assert.deepStrictEqual(suffix1, PROMPT1, "prefixMatch on empty cache should return full input");
    ws.forwardEagerPrefill(model, [suffix1], pagedKV);
    pagedKV.reportTokens(0, suffix1);
    pagedKV.updateIndptr(ws);
    const suffix2 = pagedKV.prefixMatch(0, fullPrompt);
    assert.deepStrictEqual(suffix2, suffix, `prefixMatch should return suffix after PROMPT1, got ${suffix2}`);
    ws.forwardEagerPrefill(model, [suffix2], pagedKV);
    pagedKV.reportTokens(0, suffix2);

    singleKV.reset(1);
    const singleToken = ws.forwardEagerPrefill(model, [fullPrompt], singleKV)[0];

    assert.equal(tokensFull[0], singleToken,
      `Full paged prefill mismatch: paged=${tokensFull[0]}, single=${singleToken}`);
    assert.equal(pagedKV.sequences[0].pages.length, 1, "should have 1 page for 6+4=10 tokens with pageSize=16");
  });

  it("batch prefill truncate append", () => {
    using pagedKV = makePagedKV(1, 256);
    using singleKV = makePagedKV(1, 256);
    const base = makeLongPrompt(PAGE_SIZE);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(1);
    ws.forwardEagerPrefill(model, [base], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws);
    let suffixA = pagedKV.prefixMatch(0, fullPrompt);
    assert.deepStrictEqual(suffixA, suffix, `prefixMatch after base should return suffix, got ${suffixA}`);
    ws.forwardEagerPrefill(model, [suffixA], pagedKV);
    pagedKV.reportTokens(0, suffixA);
    pagedKV.updateIndptr(ws);

    pagedKV.prefixMatch(0, base);
    pagedKV.updateIndptr(ws);
    const suffixB = pagedKV.prefixMatch(0, fullPrompt);
    assert.deepStrictEqual(suffixB, suffix, `prefixMatch after truncate should return suffix, got ${suffixB}`);
    pagedKV.updateIndptr(ws);
    const tokensTruncAppend = ws.forwardEagerPrefill(model, [suffixB], pagedKV);
    pagedKV.reportTokens(0, suffixB);

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
    stateRef.setInput([[tokens[0]]]);
    gws.positionStep(stateRef, model);
    const hiddenStatesRef = model.forward(stateRef);
    const logitsRef = stateRef.computeLogits(hiddenStatesRef, model);
    using argmaxRef = logitsRef.argmax();
    const tokensRef = argmaxRef.readInt32LEArray();

    pagedKV.reset(1);
    const tokens2 = gws.forwardEagerPrefill(model, [prompt], pagedKV);
    const state = gws.planDecode(model, 1, pagedKV, true);
    state.setInput([[tokens2[0]]]);

    glm.graphBeginCapture();
    gws.positionStep(state, model);
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
      state.setInput([[current]]);
      gws.positionStep(state, model);
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
      state.setInput([[current]]);

      if (graphExec === null) {
        if (warmupRemaining === 0 && !capturing) {
          capturing = true;
          glm.graphBeginCapture();
        }

    gws.positionStep(state, model);
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
    using pagedKV = model.createChatCache(256, 4) as PagedKVCache;
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
    state.setInput([PROMPT_GRAPH]);
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
    using pagedKV = model.createChatCache(4, 4) as PagedKVCache;
    const greedy: SamplingParams = makeSamplingParams({
      temperature: 0, topP: 1.0, topK: 0,
      repetitionPenalty: 1.0, presencePenalty: 0, repetitionPenaltyWindow: 64,
    });

    pagedKV.reset(2);
    const state = ws.planPrefill(model, 2, [PROMPT1.length, PROMPT2.length], pagedKV);
    state.setInput([PROMPT1, PROMPT2]);
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
      state.setInput([chunk]);
      const hiddenStates = model.forward(state);
      if (i < chunkSizes.length - 1) {
        hiddenStates[Symbol.dispose]();
      } else {
        logits = state.computeLogits(hiddenStates, model);
        hiddenStates[Symbol.dispose]();
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

  it("prefill after KV truncate: argmax at every position matches decode", () => {
    using pagedKV = makePagedKV(1, 256);
    const prompt = makeLongPrompt(PAGE_SIZE * 2);
    const numDecodeSteps = 8;

    // Step 1: Prefill prompt and decode N steps greedily to collect answer tokens
    pagedKV.reset(1);
    let current = ws.forwardEagerPrefill(model, [prompt], pagedKV)[0];
    pagedKV.reportTokens(0, prompt);
    pagedKV.updateIndptr(ws);

    const answerTokens: number[] = [current];
    for (let step = 0; step < numDecodeSteps; step++) {
      const result = ws.forwardEagerDecode(model, [current], pagedKV);
      pagedKV.reportTokens(0, [current]);
      pagedKV.updateIndptr(ws);
      current = result[0];
      answerTokens.push(current);
    }
    // answerTokens = [T0, T1, ..., Tn] where T0 is from prefill, T1..Tn from decode

    // Step 2: Truncate KV cache back to the prompt
    const suffix = pagedKV.prefixMatch(0, prompt);
    assert.deepStrictEqual(suffix, [],
      `prefixMatch should return empty suffix for exact prompt match, got length ${suffix.length}`);
    pagedKV.updateIndptr(ws);

    // Step 3: Prefill all answer tokens at once, get logits at every position
    const state = ws.planPrefill(model, 1, [answerTokens.length], pagedKV);
    state.setInput([answerTokens]);
    const hiddenStates = model.forward(state);
    const allLogits = state.computeLogits(hiddenStates, model, true);
    using argmaxResult = allLogits.argmax();
    const predictions = argmaxResult.readInt32LEArray();
    pagedKV.reportTokens(0, answerTokens);
    pagedKV.updateIndptr(ws);

    // Step 4: Each position i should predict answerTokens[i+1]
    // (position 0 predicts T1, position 1 predicts T2, etc.)
    const expected = answerTokens.slice(1);
    for (let i = 0; i < expected.length; i++) {
      assert.equal(predictions[i], expected[i],
        `Position ${i}: predicted ${predictions[i]} but decode produced ${expected[i]}`);
    }
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
    model = await Qwen35Model.fromPretrained(glm, "Qwen/Qwen3.5-0.8B");
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
      state.setInput([chunk]);
      const hiddenStates = model.forward(state);
      if (i < chunkSizes.length - 1) {
        hiddenStates[Symbol.dispose]();
      } else {
        logits = state.computeLogits(hiddenStates, model);
        hiddenStates[Symbol.dispose]();
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

describe("PagedKVCache prefix matching", () => {
  let glm: GlmOps;
  let model: Qwen3Model;
  let ws: ExecutionWorkspace;
  let PROMPT1: number[];

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Qwen3Model.fromPretrained(glm, QWEN3_REPO);
    ws = new ExecutionWorkspace(glm, 4, 4096);
    const tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(QWEN3_REPO), { local_files_only: true });
    PROMPT1 = tokenizePrompt(tokenizer, "Hi");
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

  it("self-match: prefix shorter than pageSize returns suffix", () => {
    using pagedKV = makePagedKV(1, 256);
    const prompt = PROMPT1;
    const suffix = [100, 101, 102, 103];
    const fullPrompt = [...prompt, ...suffix];

    pagedKV.reset(1);
    ws.forwardEagerPrefill(model, [prompt], pagedKV);
    pagedKV.reportTokens(0, prompt);
    pagedKV.updateIndptr(ws);

    const result = pagedKV.prefixMatch(0, fullPrompt);
    assert.deepStrictEqual(result, suffix, `self-match short prefix should return suffix, got ${result}`);
    assert.equal(pagedKV.sequences[0].pages.length, 1, "self-match should keep existing pages");
  });

  it("self-match: truncate longer cache returns suffix", () => {
    using pagedKV = makePagedKV(1, 256);
    const base = makeLongPrompt(PAGE_SIZE);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(1);
    ws.forwardEagerPrefill(model, [fullPrompt], pagedKV);
    pagedKV.reportTokens(0, fullPrompt);
    pagedKV.updateIndptr(ws);

    const result = pagedKV.prefixMatch(0, base);
    assert.deepStrictEqual(result, [], `truncate should return empty suffix, got ${result}`);
    assert.equal(pagedKV.sequences[0].pages.length, 1, "should keep 1 page after truncate");

    const suffixResult = pagedKV.prefixMatch(0, fullPrompt);
    assert.deepStrictEqual(suffixResult, suffix, `after truncate, prefixMatch should return suffix, got ${suffixResult}`);
  });

  it("cross-sequence: share full pages between sequences", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    const baseLen = PAGE_SIZE * 2;
    const base = makeLongPrompt(baseLen);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(2);
    ws2.forwardEagerPrefill(model, [base, []], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws2);

    const result = pagedKV.prefixMatch(1, fullPrompt);
    assert.deepStrictEqual(result, suffix, `cross-sequence prefix match should return suffix, got ${result}`);
    assert.equal(pagedKV.sequences[1].pages.length, 2, "seq1 should share 2 pages from seq0");
    assert.equal(pagedKV.sequences[1].allocLen, baseLen, `seq1 allocLen should be ${baseLen}, got ${pagedKV.sequences[1].allocLen}`);

    for (let i = 0; i < 2; i++) {
      assert.equal(pagedKV.sequences[0].pages[i].refs, 2, `page ${i} should have ref count 2`);
    }
  });

  it("cross-sequence: only full pages shared, partial page not shared", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    const baseLen = PAGE_SIZE + 4;
    const base = makeLongPrompt(baseLen);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(2);
    ws2.forwardEagerPrefill(model, [base, []], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws2);

    const result = pagedKV.prefixMatch(1, fullPrompt);
    assert.deepStrictEqual(result, fullPrompt.slice(PAGE_SIZE),
      `should share only full page, got suffix starting with ${result.slice(0, 3)}`);
    assert.equal(pagedKV.sequences[1].pages.length, 1, "seq1 should share 1 full page from seq0");
  });

  it("cross-sequence: page extending past match is not shared", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    const baseLen = PAGE_SIZE + 4;
    const base = makeLongPrompt(baseLen);
    const matchLen = PAGE_SIZE + 2;
    const matchInput = base.slice(0, matchLen);
    const suffix = [200, 201];
    const fullPrompt = [...matchInput, ...suffix];

    pagedKV.reset(2);
    ws2.forwardEagerPrefill(model, [base, []], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws2);

    const result = pagedKV.prefixMatch(1, fullPrompt);
    assert.deepStrictEqual(result, fullPrompt.slice(PAGE_SIZE),
      `page extending past match should not be shared, got suffix starting with ${result.slice(0, 3)}`);
    assert.equal(pagedKV.sequences[1].pages.length, 1, "seq1 should share only 1 page (page1 extends past match)");
  });

  it("self-match wins tie when cross-match has same token count", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    const shortPrefix = PROMPT1;
    const longerBase = makeLongPrompt(PAGE_SIZE + 2);

    pagedKV.reset(2);
    ws2.forwardEagerPrefill(model, [shortPrefix, longerBase], pagedKV);
    pagedKV.reportTokens(0, shortPrefix);
    pagedKV.reportTokens(1, longerBase);
    pagedKV.updateIndptr(ws2);

    const fullForSeq0 = [...shortPrefix, 999, 998];
    const result = pagedKV.prefixMatch(0, fullForSeq0);
    assert.deepStrictEqual(result, [999, 998],
      `self-match keeps pages when tied with cross-match, got ${result}`);
    assert.equal(pagedKV.sequences[0].pages.length, 1, "seq0 should keep its page");
  });

  it("no pages shared when all matching tokens in partial page", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    const base = makeLongPrompt(PAGE_SIZE + 2);

    pagedKV.reset(2);
    ws2.forwardEagerPrefill(model, [[], base], pagedKV);
    pagedKV.reportTokens(1, base);
    pagedKV.updateIndptr(ws2);

    const input = base.slice(0, 10);
    const result = pagedKV.prefixMatch(0, input);
    assert.deepStrictEqual(result, input,
      `full input returned when match falls in partial page, got ${result}`);
    assert.equal(pagedKV.sequences[0].pages.length, 0, "seq0 should be cleared");
  });

  it("cross-sequence sharing produces correct prefill output", () => {
    using pagedKV = makePagedKV(1, 256);
    using singleKV = makePagedKV(1, 256);
    const baseLen = PAGE_SIZE * 2;
    const base = makeLongPrompt(baseLen);
    const suffix = makeLongPrompt(8, [200]);
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(1);
    ws.forwardEagerPrefill(model, [base], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws);

    const sharedSuffix = pagedKV.prefixMatch(0, fullPrompt);
    ws.forwardEagerPrefill(model, [sharedSuffix], pagedKV);
    pagedKV.reportTokens(0, sharedSuffix);

    singleKV.reset(1);
    const singleTokens = ws.forwardEagerPrefill(model, [fullPrompt], singleKV);

    assert.equal(pagedKV.sequences[0].pages.length, singleKV.sequences[0].pages.length,
      "page count should match");
  });

  it("self-match: full cache match continues from partial page", () => {
    using pagedKV = makePagedKV(1, 256);
    using singleKV = makePagedKV(1, 256);
    const prompt = PROMPT1;
    const suffix = [100, 101, 102, 103];
    const fullPrompt = [...prompt, ...suffix];

    pagedKV.reset(1);
    ws.forwardEagerPrefill(model, [prompt], pagedKV);
    pagedKV.reportTokens(0, prompt);
    pagedKV.updateIndptr(ws);

    const pagesBefore = pagedKV.sequences[0].pages.length;
    const result = pagedKV.prefixMatch(0, fullPrompt);
    assert.deepStrictEqual(result, suffix, `full cache match should return suffix, got ${result}`);
    assert.equal(pagedKV.sequences[0].pages.length, pagesBefore, "no pages should be popped on full match");

    ws.forwardEagerPrefill(model, [suffix], pagedKV);
    pagedKV.reportTokens(0, suffix);

    singleKV.reset(1);
    ws.forwardEagerPrefill(model, [fullPrompt], singleKV);

    assert.equal(pagedKV.sequences[0].reportedTokenCount(), fullPrompt.length, "reportedTokenCount should equal full prompt length");
  });

  it("self-match: full cache match at page boundary", () => {
    using pagedKV = makePagedKV(1, 256);
    const base = makeLongPrompt(PAGE_SIZE * 2);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(1);
    ws.forwardEagerPrefill(model, [base], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws);

    const pagesBefore = pagedKV.sequences[0].pages.length;
    const result = pagedKV.prefixMatch(0, fullPrompt);
    assert.deepStrictEqual(result, suffix, `full cache match at boundary should return suffix, got ${result}`);
    assert.equal(pagedKV.sequences[0].pages.length, pagesBefore, "no pages should be popped on full match");
  });

  it("empty cache returns full input", () => {
    using pagedKV = makePagedKV(1, 256);
    pagedKV.reset(1);
    const result = pagedKV.prefixMatch(0, PROMPT1);
    assert.deepStrictEqual(result, PROMPT1, "empty cache should return full input");
  });

  it("truncate and re-extend produces correct decode", () => {
    using pagedKV = makePagedKV(1, 256);
    const base = makeLongPrompt(PAGE_SIZE);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(1);
    const fullTokens = ws.forwardEagerPrefill(model, [fullPrompt], pagedKV);
    pagedKV.reportTokens(0, fullPrompt);
    pagedKV.updateIndptr(ws);

    pagedKV.prefixMatch(0, base);
    pagedKV.updateIndptr(ws);

    const suffixResult = pagedKV.prefixMatch(0, fullPrompt);
    assert.deepStrictEqual(suffixResult, suffix, `after truncate, prefixMatch should return suffix`);
    ws.forwardEagerPrefill(model, [suffixResult], pagedKV);
    pagedKV.reportTokens(0, suffixResult);
    pagedKV.updateIndptr(ws);

    const decodeToken = ws.forwardEagerDecode(model, [fullTokens[0]], pagedKV)[0];
    assert.equal(typeof decodeToken, "number", "decode should produce a valid token");
  });

  it("cross-sequence copyPartial: copies partial page tokens and sets allocLen", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    const baseLen = PAGE_SIZE + 4;
    const base = makeLongPrompt(baseLen);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(2);
    ws2.forwardEagerPrefill(model, [base, []], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws2);

    const pageBytes = pagedKV.nKv * PAGE_SIZE * pagedKV.hd * 2;
    const srcPageId = pagedKV.sequences[0].pages[1].id;
    const dstPageId = pagedKV.availablePages[0];
    const srcOff = srcPageId * pageBytes;
    const dstOff = dstPageId * pageBytes;

    const preCopyK: Buffer[] = [];
    const preCopyV: Buffer[] = [];
    for (let layer = 0; layer < pagedKV.kData.length; layer++) {
      const kvBytes = pagedKV.maxPages * pageBytes;
      const kBuf = Buffer.alloc(kvBytes);
      const vBuf = Buffer.alloc(kvBytes);
      pagedKV.kData[layer].d2h(kBuf);
      pagedKV.vData[layer].d2h(vBuf);
      preCopyK.push(kBuf);
      preCopyV.push(vBuf);
    }

    const result = pagedKV.prefixMatch(1, fullPrompt, true);
    const expectedSuffixLen = fullPrompt.length - baseLen;
    assert.deepStrictEqual(result, suffix, `copyPartial should return suffix of ${expectedSuffixLen} tokens, got ${result.length} tokens`);
    assert.equal(pagedKV.sequences[1].pages.length, 2, "seq1 should have 2 pages (1 full + 1 partial copy)");
    assert.equal(pagedKV.sequences[1].allocLen, baseLen, `seq1 allocLen should be ${baseLen}, got ${pagedKV.sequences[1].allocLen}`);
    assert.equal(pagedKV.sequences[1].reportedTokenCount(), baseLen, `seq1 reportedTokenCount should be ${baseLen}, got ${pagedKV.sequences[1].reportedTokenCount()}`);
    const srcPartialPage = pagedKV.sequences[0].pages[1];
    const dstPartialPage = pagedKV.sequences[1].pages[1];
    assert.notEqual(dstPartialPage.id, srcPartialPage.id, "copied partial page should have a different page id");
    assert.equal(dstPartialPage.id, dstPageId, "copied partial page should use the expected available page");
    const lastPageTokens = dstPartialPage.tokenIds;
    assert.deepStrictEqual(lastPageTokens, base.slice(PAGE_SIZE),
      `partial page tokenIds should match source, got ${lastPageTokens}`);
    for (let layer = 0; layer < pagedKV.kData.length; layer++) {
      const kvBytes = pagedKV.maxPages * pageBytes;
      const kBuf = Buffer.alloc(kvBytes);
      const vBuf = Buffer.alloc(kvBytes);
      pagedKV.kData[layer].d2h(kBuf);
      pagedKV.vData[layer].d2h(vBuf);
      const srcKPage = kBuf.subarray(srcOff, srcOff + pageBytes);
      const srcVPage = vBuf.subarray(srcOff, srcOff + pageBytes);
      const preKDst = preCopyK[layer].subarray(dstOff, dstOff + pageBytes);
      const preVDst = preCopyV[layer].subarray(dstOff, dstOff + pageBytes);
      assert.ok(!preKDst.equals(srcKPage), `kData layer ${layer}: dst page should differ from source before copy`);
      assert.ok(!preVDst.equals(srcVPage), `vData layer ${layer}: dst page should differ from source before copy`);
      assert.deepStrictEqual(kBuf.subarray(dstOff, dstOff + pageBytes), srcKPage,
        `kData layer ${layer}: partial page data should match source after copy`);
      assert.deepStrictEqual(vBuf.subarray(dstOff, dstOff + pageBytes), srcVPage,
        `vData layer ${layer}: partial page data should match source after copy`);
    }
  });

  it("cross-sequence copyPartial: ref count unchanged on source pages", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    const baseLen = PAGE_SIZE + 4;
    const base = makeLongPrompt(baseLen);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(2);
    ws2.forwardEagerPrefill(model, [base, []], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws2);

    pagedKV.prefixMatch(1, fullPrompt, true);
    assert.equal(pagedKV.sequences[0].pages[0].refs, 2, "full page should have ref count 2 (shared)");
    assert.equal(pagedKV.sequences[0].pages[1].refs, 1, "source partial page should still have ref count 1 (not shared)");
    assert.equal(pagedKV.sequences[1].pages[1].refs, 1, "copied partial page should have ref count 1");
  });

  it("cross-sequence copyPartial=false: partial page not copied (original behavior)", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    const baseLen = PAGE_SIZE + 4;
    const base = makeLongPrompt(baseLen);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(2);
    ws2.forwardEagerPrefill(model, [base, []], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws2);

    const result = pagedKV.prefixMatch(1, fullPrompt, false);
    assert.deepStrictEqual(result, fullPrompt.slice(PAGE_SIZE),
      `copyPartial=false should only share full pages, got suffix starting with ${result.slice(0, 3)}`);
    assert.equal(pagedKV.sequences[1].pages.length, 1, "seq1 should share only 1 full page");
    assert.equal(pagedKV.sequences[1].allocLen, PAGE_SIZE, `seq1 allocLen should be ${PAGE_SIZE}`);
  });

  it("cross-sequence copyPartial: no partial page when match is page-aligned", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    const baseLen = PAGE_SIZE * 2;
    const base = makeLongPrompt(baseLen);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(2);
    ws2.forwardEagerPrefill(model, [base, []], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws2);

    const result = pagedKV.prefixMatch(1, fullPrompt, true);
    assert.deepStrictEqual(result, suffix, `page-aligned match should return suffix, got ${result}`);
    assert.equal(pagedKV.sequences[1].pages.length, 2, "seq1 should share 2 full pages");
    assert.equal(pagedKV.sequences[1].allocLen, baseLen, `seq1 allocLen should be ${baseLen}`);
  });

  it("cross-sequence copyPartial: subsequent reportTokens places tokens correctly", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    const baseLen = PAGE_SIZE + 4;
    const base = makeLongPrompt(baseLen);
    const suffix = [200, 201, 202, 203];
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(2);
    ws2.forwardEagerPrefill(model, [base, []], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws2);

    const result = pagedKV.prefixMatch(1, fullPrompt, true);
    assert.deepStrictEqual(result, suffix, `copyPartial should return suffix`);

    pagedKV.reportTokens(1, result);
    assert.equal(pagedKV.sequences[1].reportedTokenCount(), fullPrompt.length,
      `after reportTokens, reportedTokenCount should be ${fullPrompt.length}, got ${pagedKV.sequences[1].reportedTokenCount()}`);
    const allTokens = pagedKV.sequences[1].pages.map(p => p.tokenIds).flat();
    assert.deepStrictEqual(allTokens, fullPrompt, "all tokens should match full prompt after append");
  });

  it("cross-sequence sharing: prefill + decode matches full prefill + decode", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    using refKV = makePagedKV(1, 256);
    const baseLen = PAGE_SIZE * 2;
    const base = makeLongPrompt(baseLen);
    const suffix = makeLongPrompt(8, [200]);
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(2);
    const prefillTokens = ws2.forwardEagerPrefill(model, [base, []], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws2);

    const sharedSuffix = pagedKV.prefixMatch(1, fullPrompt);
    assert.deepStrictEqual(sharedSuffix, suffix, `prefixMatch should return suffix`);
    const suffixTokens = ws2.forwardEagerPrefill(model, [[], sharedSuffix], pagedKV);
    pagedKV.reportTokens(1, sharedSuffix);
    pagedKV.updateIndptr(ws2);

    refKV.reset(1);
    const refTokens = ws.forwardEagerPrefill(model, [fullPrompt], refKV);
    refKV.reportTokens(0, fullPrompt);
    refKV.updateIndptr(ws);

    const numDecodeSteps = 5;
    let lastSeq0 = prefillTokens[0];
    let lastSeq1 = suffixTokens[1];
    let lastRef = refTokens[0];

    const seq1Tokens: number[] = [];
    const refDecoded: number[] = [];

    for (let step = 0; step < numDecodeSteps; step++) {
      const decoded = ws2.forwardEagerDecode(model, [lastSeq0, lastSeq1], pagedKV);
      const refStep = ws.forwardEagerDecode(model, [lastRef], refKV);
      pagedKV.allocDecodeToken(0);
      pagedKV.allocDecodeToken(1);
      pagedKV.reportTokens(0, [decoded[0]]);
      pagedKV.reportTokens(1, [decoded[1]]);
      pagedKV.updateIndptr(ws2);
      refKV.allocDecodeToken(0);
      refKV.reportTokens(0, [refStep[0]]);
      refKV.updateIndptr(ws);
      lastSeq0 = decoded[0];
      lastSeq1 = decoded[1];
      lastRef = refStep[0];
      seq1Tokens.push(lastSeq1);
      refDecoded.push(lastRef);
    }

    assert.deepStrictEqual(seq1Tokens, refDecoded,
      `shared-page decode tokens should match reference: shared=${seq1Tokens}, ref=${refDecoded}`);
  });

  it("cross-sequence copyPartial: prefill + decode matches full prefill + decode", () => {
    using pagedKV = makePagedKV(2, 256);
    using ws2 = new ExecutionWorkspace(glm, 2, 4096);
    using refKV = makePagedKV(1, 256);
    const baseLen = PAGE_SIZE + 4;
    const base = makeLongPrompt(baseLen);
    const suffix = makeLongPrompt(8, [200]);
    const fullPrompt = [...base, ...suffix];

    pagedKV.reset(2);
    const prefillTokens = ws2.forwardEagerPrefill(model, [base, []], pagedKV);
    pagedKV.reportTokens(0, base);
    pagedKV.updateIndptr(ws2);

    const sharedSuffix = pagedKV.prefixMatch(1, fullPrompt, true);
    assert.deepStrictEqual(sharedSuffix, suffix, `prefixMatch with copyPartial should return suffix`);
    const suffixTokens = ws2.forwardEagerPrefill(model, [[], sharedSuffix], pagedKV);
    pagedKV.reportTokens(1, sharedSuffix);
    pagedKV.updateIndptr(ws2);

    refKV.reset(1);
    const refTokens = ws.forwardEagerPrefill(model, [fullPrompt], refKV);
    refKV.reportTokens(0, fullPrompt);
    refKV.updateIndptr(ws);

    const numDecodeSteps = 5;
    let lastSeq0 = prefillTokens[0];
    let lastSeq1 = suffixTokens[1];
    let lastRef = refTokens[0];

    const seq1Tokens: number[] = [];
    const refDecoded: number[] = [];

    for (let step = 0; step < numDecodeSteps; step++) {
      const decoded = ws2.forwardEagerDecode(model, [lastSeq0, lastSeq1], pagedKV);
      const refStep = ws.forwardEagerDecode(model, [lastRef], refKV);
      pagedKV.allocDecodeToken(0);
      pagedKV.allocDecodeToken(1);
      pagedKV.reportTokens(0, [decoded[0]]);
      pagedKV.reportTokens(1, [decoded[1]]);
      pagedKV.updateIndptr(ws2);
      refKV.allocDecodeToken(0);
      refKV.reportTokens(0, [refStep[0]]);
      refKV.updateIndptr(ws);
      lastSeq0 = decoded[0];
      lastSeq1 = decoded[1];
      lastRef = refStep[0];
      seq1Tokens.push(lastSeq1);
      refDecoded.push(lastRef);
    }

    assert.deepStrictEqual(seq1Tokens, refDecoded,
      `copyPartial decode tokens should match reference: copyPartial=${seq1Tokens}, ref=${refDecoded}`);
  });
});
