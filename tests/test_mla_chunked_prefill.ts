import assert from "node:assert/strict";
import path from "node:path";
import { after, before, describe, it } from "node:test";
import type { ChatCache, ChatModel } from "../src/chat_model";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps } from "../src/glm_ops";
import { Tensor } from "../src/tensor";

const SMALL_MODEL_DIR = path.resolve(
  __dirname,
  "../tests/python/test_models/glm51_small/glm51_small_bf16",
);

const PROMPT = [151644, 872, 198, 72357, 752, 264, 220, 16, 15, 15, 15, 3409, 3364, 151645, 198, 151644, 77091, 198];

function chunkedPrefill(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  inputIds: number[], chunkSizes: number[],
): number[] {
  let offset = 0;
  let logits: ReturnType<Tensor["argmax"]> | null = null;
  for (let i = 0; i < chunkSizes.length; i++) {
    const chunk = inputIds.slice(offset, offset + chunkSizes[i]);
    offset += chunkSizes[i];
    const state = ws.planPrefill(model, 1, [chunk.length], cache);
    state.setInput([chunk]);
    using hiddenStates = model.forward(state);
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

describe("GLM-5.1 MLA: 1-token chunked prefill vs full prefill + decode", () => {
  let glm: GlmOps;
  let model: Glm51Model;
  let ws: ExecutionWorkspace;

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Glm51Model.fromPretrained(glm, SMALL_MODEL_DIR);
    ws = new ExecutionWorkspace(glm, 2, 128);
  });

  after(() => {
    ws.free();
    model.free();
    glm.free();
  });

  it("1-token chunked prefill matches full prefill", () => {
    using cache1 = model.createChatCache(64);
    using cache2 = model.createChatCache(64);
    const prompt = PROMPT;
    const chunkSizes = prompt.map(() => 1);

    cache1.reset(1);
    const fullToken = ws.forwardEagerPrefill(model, [prompt], cache1)[0];

    cache2.reset(1);
    const chunkedToken = chunkedPrefill(model, ws, cache2, prompt, chunkSizes)[0];

    assert.equal(chunkedToken, fullToken,
      `1-token chunked prefill ${chunkedToken} != full prefill ${fullToken}`);
  });

  it("1-token chunked prefill + decode matches full prefill + decode", () => {
    using cache1 = model.createChatCache(64);
    using cache2 = model.createChatCache(64);
    const prompt = PROMPT;
    const chunkSizes = prompt.map(() => 1);

    cache1.reset(1);
    const fullToken = ws.forwardEagerPrefill(model, [prompt], cache1)[0];
    cache1.getPagedKV().updateIndptr(ws);
    const fullDecode = ws.forwardEagerDecode(model, [fullToken], cache1)[0];

    cache2.reset(1);
    const chunkedToken = chunkedPrefill(model, ws, cache2, prompt, chunkSizes)[0];
    cache2.getPagedKV().updateIndptr(ws);
    const chunkedDecode = ws.forwardEagerDecode(model, [chunkedToken], cache2)[0];

    assert.equal(chunkedToken, fullToken,
      `1-token chunked prefill token ${chunkedToken} != full prefill token ${fullToken}`);
    assert.equal(chunkedDecode, fullDecode,
      `1-token chunked decode ${chunkedDecode} != full decode ${fullDecode}`);
  });

  it("1-token chunked prefill emulates decode after prefill", () => {
    using cache1 = model.createChatCache(64);
    using cache2 = model.createChatCache(64);
    const prompt = PROMPT;
    const numDecodeSteps = 6;

    cache1.reset(1);
    const firstToken = ws.forwardEagerPrefill(model, [prompt], cache1)[0];
    cache1.getPagedKV().updateIndptr(ws);

    const decodeTokens: number[] = [firstToken];
    let current = firstToken;
    for (let step = 0; step < numDecodeSteps; step++) {
      current = ws.forwardEagerDecode(model, [current], cache1)[0];
      cache1.getPagedKV().updateIndptr(ws);
      decodeTokens.push(current);
    }

    cache2.reset(1);
    const chunkedFirstToken = ws.forwardEagerPrefill(model, [prompt], cache2)[0];
    assert.equal(chunkedFirstToken, firstToken,
      `prefill token mismatch: ${chunkedFirstToken} != ${firstToken}`);
    cache2.getPagedKV().updateIndptr(ws);

    const chunkedDecodeTokens: number[] = [chunkedFirstToken];
    let chunkedCurrent = chunkedFirstToken;
    for (let step = 0; step < numDecodeSteps; step++) {
      const state = ws.planPrefill(model, 1, [1], cache2);
      state.setInput([[chunkedCurrent]]);
      using hiddenStates = model.forward(state);
      using logits = state.computeLogits(hiddenStates, model);
      using argmaxOut = logits.argmax();
      chunkedCurrent = argmaxOut.readInt32LEArray()[0];
      cache2.getPagedKV().updateIndptr(ws);
      chunkedDecodeTokens.push(chunkedCurrent);

      assert.equal(chunkedCurrent, decodeTokens[step + 1],
        `Step ${step + 1}: 1-token prefill ${chunkedCurrent} != decode ${decodeTokens[step + 1]}`);
    }

    assert.deepStrictEqual(chunkedDecodeTokens, decodeTokens,
      `1-token prefill tokens ${chunkedDecodeTokens} != decode tokens ${decodeTokens}`);
  });

  it("1-token chunked prefill with longer prompt", () => {
    using cache1 = model.createChatCache(128);
    using cache2 = model.createChatCache(128);
    const prompt = PROMPT;
    const chunkSizes = prompt.map(() => 1);

    cache1.reset(1);
    const fullToken = ws.forwardEagerPrefill(model, [prompt], cache1)[0];
    cache1.getPagedKV().updateIndptr(ws);
    const fullDecode = ws.forwardEagerDecode(model, [fullToken], cache1)[0];

    cache2.reset(1);
    const chunkedToken = chunkedPrefill(model, ws, cache2, prompt, chunkSizes)[0];
    cache2.getPagedKV().updateIndptr(ws);
    const chunkedDecode = ws.forwardEagerDecode(model, [chunkedToken], cache2)[0];

    assert.equal(chunkedToken, fullToken,
      `1-token chunked prefill token ${chunkedToken} != full prefill token ${fullToken}`);
    assert.equal(chunkedDecode, fullDecode,
      `1-token chunked decode ${chunkedDecode} != full decode ${fullDecode}`);
  });
});
