import assert from "node:assert/strict";
import path from "node:path";
import { after, before, describe, it } from "node:test";
import { MaskMode } from "../src/device_ops";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { bf16BytesToF32, I32 } from "../src/glm_ops";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps } from "../src/glm_ops";

const SMALL_MODEL_DIR = path.resolve(__dirname, "../tests/python/test_models/glm51_small/glm51_small_bf16");

function i32Buf(data: Int32Array): Buffer {
  const buf = Buffer.alloc(data.length * 4);
  for (let i = 0; i < data.length; i++) buf.writeInt32LE(data[i], i * 4);
  return buf;
}

function buildCausalMask(qoLen: number): Uint8Array {
  const totalBits = qoLen * qoLen;
  const byteLen = Math.ceil(totalBits / 8);
  const data = new Uint8Array(byteLen);
  for (let q = 0; q < qoLen; q++) {
    for (let k = 0; k <= q; k++) {
      const bit = q * qoLen + k;
      data[bit >> 3] |= 1 << (bit & 7);
    }
  }
  return data;
}

function readLogits(logitsTensor: { d2h: (buf: Buffer) => void }, numTokens: number, vocabSize: number): Float32Array {
  const buf = Buffer.alloc(numTokens * vocabSize * 2);
  logitsTensor.d2h(buf);
  return bf16BytesToF32(buf);
}

describe("Token count comparison: same attention, different total tokens", () => {
  let glm: GlmOps;
  let model: Glm51Model;
  let ws: ExecutionWorkspace;

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Glm51Model.fromPretrained(glm, SMALL_MODEL_DIR);
    ws = new ExecutionWorkspace(glm, 2, 256);
  });

  after(() => {
    ws.free();
    model.free();
    glm.free();
  });

  it("8-token built-in causal vs 2-token built-in causal at position 1", () => {
    const vocabSize = model.cfg.vocabSize;

    // 8-token built-in causal
    using cache8 = model.createChatCache(128);
    cache8.reset(1);
    const state8 = ws.planPrefill(model, 1, [8], cache8);
    state8.setInput([[10, 20, 30, 40, 50, 60, 70, 80]]);
    using hidden8 = model.forward(state8);
    using logits8 = state8.computeLogits(hidden8, model, null);
    const f8 = readLogits(logits8, 8, vocabSize);

    // 2-token built-in causal
    using cache2 = model.createChatCache(128);
    cache2.reset(1);
    const state2 = ws.planPrefill(model, 1, [2], cache2);
    state2.setInput([[10, 20]]);
    using hidden2 = model.forward(state2);
    using logits2 = state2.computeLogits(hidden2, model, null);
    const f2 = readLogits(logits2, 2, vocabSize);

    // Position 1: token 20 attends to {0, 1} in both cases
    let maxDiff1 = 0;
    for (let v = 0; v < vocabSize; v++) {
      const diff = Math.abs(f8[1 * vocabSize + v] - f2[1 * vocabSize + v]);
      if (diff > maxDiff1) maxDiff1 = diff;
    }

    // Position 0: token 10 attends to {0} in both cases
    let maxDiff0 = 0;
    for (let v = 0; v < vocabSize; v++) {
      const diff = Math.abs(f8[0 * vocabSize + v] - f2[0 * vocabSize + v]);
      if (diff > maxDiff0) maxDiff0 = diff;
    }

    console.log(`  8-token causal vs 2-token causal:`);
    console.log(`    position 0 max logit diff: ${maxDiff0.toFixed(6)}`);
    console.log(`    position 1 max logit diff: ${maxDiff1.toFixed(6)}`);

    // These should match since the same tokens at the same positions produce the same output
    // regardless of total token count (causal mask ensures position 1 only attends to {0,1})
    assert.ok(maxDiff0 < 0.01, `position 0: 8-token vs 2-token causal diff ${maxDiff0} > 0.01`);
    assert.ok(maxDiff1 < 0.01, `position 1: 8-token vs 2-token causal diff ${maxDiff1} > 0.01`);
  });

  it("7-token built-in causal vs 2-token built-in causal at position 1", () => {
    const vocabSize = model.cfg.vocabSize;

    using cache7 = model.createChatCache(128);
    cache7.reset(1);
    const state7 = ws.planPrefill(model, 1, [7], cache7);
    state7.setInput([[10, 20, 30, 40, 50, 60, 70]]);
    using hidden7 = model.forward(state7);
    using logits7 = state7.computeLogits(hidden7, model, null);
    const f7 = readLogits(logits7, 7, vocabSize);

    using cache2 = model.createChatCache(128);
    cache2.reset(1);
    const state2 = ws.planPrefill(model, 1, [2], cache2);
    state2.setInput([[10, 20]]);
    using hidden2 = model.forward(state2);
    using logits2 = state2.computeLogits(hidden2, model, null);
    const f2 = readLogits(logits2, 2, vocabSize);

    let maxDiff1 = 0;
    for (let v = 0; v < vocabSize; v++) {
      const diff = Math.abs(f7[1 * vocabSize + v] - f2[1 * vocabSize + v]);
      if (diff > maxDiff1) maxDiff1 = diff;
    }

    console.log(`  7-token causal vs 2-token causal at position 1: max diff = ${maxDiff1.toFixed(6)}`);
    assert.ok(maxDiff1 < 0.01, `7-token vs 2-token causal diff ${maxDiff1} > 0.01`);
  });

  it("sweep: N-token causal vs 2-token causal at position 1", () => {
    const vocabSize = model.cfg.vocabSize;

    for (let n = 2; n <= 15; n++) {
      const tokens = Array.from({ length: n }, (_, i) => 10 + i * 10);

      using cacheN = model.createChatCache(128);
      cacheN.reset(1);
      const stateN = ws.planPrefill(model, 1, [n], cacheN);
      stateN.setInput([tokens]);
      using hiddenN = model.forward(stateN);
      using logitsN = stateN.computeLogits(hiddenN, model, null);
      const fN = readLogits(logitsN, n, vocabSize);

      using cache2 = model.createChatCache(128);
      cache2.reset(1);
      const state2 = ws.planPrefill(model, 1, [2], cache2);
      state2.setInput([[10, 20]]);
      using hidden2 = model.forward(state2);
      using logits2 = state2.computeLogits(hidden2, model, null);
      const f2 = readLogits(logits2, 2, vocabSize);

      let maxDiff = 0;
      for (let v = 0; v < vocabSize; v++) {
        const diff = Math.abs(fN[1 * vocabSize + v] - f2[1 * vocabSize + v]);
        if (diff > maxDiff) maxDiff = diff;
      }
      const status = maxDiff < 0.01 ? "PASS" : "FAIL";
      console.log(`  ${n}-token causal vs 2-token causal at position 1: max diff = ${maxDiff.toFixed(6)} ${status}`);
    }
  });
});
