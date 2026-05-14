import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { GlmOps } from "../src/glm_ops";
import { Glm51Model } from "../src/glm51_model";
import { ExecutionWorkspace, PagedKVCache } from "../src/paged_kv";
import { Tensor } from "../src/tensor";
import type { ChatModel, ChatCache } from "../src/chat_model";

const SMALL_MODEL_DIR = path.resolve(
  __dirname,
  "../tests/python/test_models/glm51_small/glm51_small_bf16",
);

describe("GLM-5.1 small model smoke test", () => {
  let glm: GlmOps;
  let model: Glm51Model;
  let ws: ExecutionWorkspace;

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Glm51Model.fromPretrained(glm, SMALL_MODEL_DIR, 2, 128);
    ws = new ExecutionWorkspace(glm, 2, 128);
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
    assert.equal(cfg.kvLoraRank, 128, "kvLoraRank");
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
    const cache = model.createChatCache(64);
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

  it("chunked prefill: uneven split", () => {
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
    cache1.getPagedKV().updateIndptr(ws);
    const fullDecode = ws.forwardEagerDecode(model, [fullTokens[0]], cache1)[0];

    cache2.reset(1);
    const chunkedTokens = chunkedPrefill(model, ws, cache2, fullPrompt, [mid, fullPrompt.length - mid]);
    cache2.getPagedKV().updateIndptr(ws);
    const chunkedDecode = ws.forwardEagerDecode(model, [chunkedTokens[0]], cache2)[0];

    assert.equal(chunkedTokens[0], fullTokens[0],
      `Chunked prefill token mismatch: chunked=${chunkedTokens[0]}, full=${fullTokens[0]}`);
    assert.equal(chunkedDecode, fullDecode,
      `Chunked decode token mismatch: chunked=${chunkedDecode}, full=${fullDecode}`);
  });
});
