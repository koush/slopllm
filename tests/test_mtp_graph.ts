import assert from "node:assert/strict";
import path from "node:path";
import { after, before, describe, it } from "node:test";
import { type ChatCache, ChatModel } from "../src/chat_model";
import { CaptureManager } from "../src/capture-manager";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps } from "../src/glm_ops";
import { MemcpyKind } from "../src/tensor";
import { ParallelOps } from "../src/parallel_ops";
import { Tensor } from "../src/tensor";
import { UsingHolder } from "../src/using-holder";
import { WorkspaceBase } from "../src/workspace";
import { mtpTreeDecode } from "../src/mtp";

const SMALL_MODEL_DIR = path.resolve(
  __dirname,
  "../tests/python/test_models/glm51_small/glm51_small_bf16",
);

const MAX_BATCH = 1;
const MAX_SEQ_LEN = 256;
const INPUT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

function greedySample(logits: Tensor, ws: WorkspaceBase): Tensor {
  using argmax = logits.argmax();
  const result = ws.alloc(argmax.shape, argmax.type);
  result.memcpy(argmax, argmax.bytes, MemcpyKind.DeviceToDevice);
  return result;
}

describe("MTP with CUDA graph capture: TP validation", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let po: ParallelOps;
  let modelTp: Glm51Model;
  let modelRef: Glm51Model;
  let wsTp: ExecutionWorkspace;
  let wsRef: ExecutionWorkspace;

  before(async () => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
    modelRef = await Glm51Model.fromPretrained(glm0, SMALL_MODEL_DIR, false, true);
    modelTp = await Glm51Model.fromPretrained(po, SMALL_MODEL_DIR, false, true);
    wsRef = new ExecutionWorkspace(glm0, MAX_BATCH, MAX_SEQ_LEN);
    wsTp = new ExecutionWorkspace(po, MAX_BATCH, MAX_SEQ_LEN);
  });

  after(() => {
    wsRef.free(); wsTp.free(); modelRef.free(); modelTp.free();
    po.free(); glm0.free(); glm1.free();
  });

  function runMtpLoop(model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache, nextn: number, maxSteps: number, label: string): number[] {
    const captureManager = new CaptureManager(ws.glm);
    const sampleWorkspace = new WorkspaceBase(ws.glm);
    let gpuSampleResult: Tensor | null = null;
    using targetHiddenStates = new UsingHolder<Tensor>(undefined!);

    const suffixIds = cache.prefixMatch(0, INPUT_IDS);
    {
      const state = ws.planPrefill(model, 1, [suffixIds.length], cache);
      state.setInput([suffixIds]);
      using hiddenStates = model.forward(state);
      using firstTokens = state.computeLogits(hiddenStates, model);
      using argmax = firstTokens.argmax();
      gpuSampleResult = sampleWorkspace.alloc(argmax.shape, argmax.type);
      gpuSampleResult.memcpy(argmax, argmax.bytes, MemcpyKind.DeviceToDevice);
      targetHiddenStates.replace(hiddenStates.slice(0, -1, 1).removeTracking());
    }

    const sampleResult = sampleWorkspace.allocPinned(gpuSampleResult!.shape, gpuSampleResult!.type);
    sampleResult.memcpy(gpuSampleResult!, gpuSampleResult!.bytes, MemcpyKind.DeviceToHost);
    ws.glm.synchronize();
    const firstToken = sampleResult.readPinnedBuffer().readInt32LE();
    cache.reportTokens(0, suffixIds);
    cache.reportTokens(0, [firstToken]);
    const output = [firstToken];

    for (let i = 1; i < maxSteps; i++) {
      if (model.forwardMtp && nextn > 0) {
        if (i === 1) {
          ws.inputIdsBuf.memcpy(gpuSampleResult!, gpuSampleResult!.bytes, MemcpyKind.DeviceToDevice);
        }
        sampleResult.memcpy(gpuSampleResult!, gpuSampleResult!.bytes, MemcpyKind.DeviceToHost);
        ws.glm.synchronize();
        const currentToken = sampleResult.readPinnedBuffer().readInt32LE();
        const result = mtpTreeDecode(
          captureManager, model, targetHiddenStates.value, ws, currentToken, Array(nextn).fill(2), cache,
        );
        for (const t of result) {
          cache.reportTokens(0, [t]);
          output.push(t);
        }
        continue;
      }

      const state = ws.planDecode(model, 1, cache, true);
      state.setInput(gpuSampleResult!);

      captureManager.run(() => {
        ws.positionStep(state, model);
        targetHiddenStates.replace(model.forward(state));
        using logits = state.computeLogits(targetHiddenStates.value, model);
        using argmax = logits.argmax();
        gpuSampleResult = gpuSampleResult || sampleWorkspace.alloc(argmax.shape, argmax.type);
        gpuSampleResult.memcpy(argmax, argmax.bytes, MemcpyKind.DeviceToDevice);
      }, ['decode']);

      sampleResult.memcpy(gpuSampleResult!, gpuSampleResult!.bytes, MemcpyKind.DeviceToHost);
      ws.glm.synchronize();
      const token = sampleResult.readPinnedBuffer().readInt32LE();
      cache.reportTokens(0, [token]);
      output.push(token);
    }

    captureManager[Symbol.dispose]();
    sampleWorkspace[Symbol.dispose]();
    console.log(`${label} tokens (${output.length}):`, output.slice(0, 20), output.length > 20 ? "..." : "");
    return output;
  }

  it("single-GPU MTP with CUDA graphs produces valid tokens across multiple steps", () => {
    using cache = modelRef.createChatCache(128);
    cache.reset(1);
    const nextn = modelRef.cfg.numNextNPredictLayers ?? 1;
    const tokens = runMtpLoop(modelRef, wsRef, cache, nextn, 10, "single-GPU");
    assert.ok(tokens.length >= 2, `expected at least 2 tokens, got ${tokens.length}`);
    for (const t of tokens) {
      assert.ok(t >= 0 && t < modelRef.cfg.vocabSize,
        `token ${t} out of range [0, ${modelRef.cfg.vocabSize})`);
    }
  });

  it("multi-GPU TP MTP with CUDA graphs produces valid tokens across multiple steps", () => {
    using cache = modelTp.createChatCache(128);
    cache.reset(1);
    const nextn = modelTp.cfg.numNextNPredictLayers ?? 1;
    const tokens = runMtpLoop(modelTp, wsTp, cache, nextn, 10, "multi-GPU-TP");
    assert.ok(tokens.length >= 2, `expected at least 2 tokens, got ${tokens.length}`);
    for (const t of tokens) {
      assert.ok(t >= 0 && t < modelTp.cfg.vocabSize,
        `token ${t} out of range [0, ${modelTp.cfg.vocabSize})`);
    }
  });

  it("single-GPU MTP with nextn=3 and CUDA graphs produces valid tokens", () => {
    using cache = modelRef.createChatCache(128);
    cache.reset(1);
    const tokens = runMtpLoop(modelRef, wsRef, cache, 3, 6, "single-GPU-nextn3");
    assert.ok(tokens.length >= 2, `expected at least 2 tokens, got ${tokens.length}`);
    for (const t of tokens) {
      assert.ok(t >= 0 && t < modelRef.cfg.vocabSize,
        `token ${t} out of range [0, ${modelRef.cfg.vocabSize})`);
    }
  });

  it("multi-GPU TP MTP with nextn=3 and CUDA graphs produces valid tokens", () => {
    using cache = modelTp.createChatCache(128);
    cache.reset(1);
    const tokens = runMtpLoop(modelTp, wsTp, cache, 3, 6, "multi-GPU-TP-nextn3");
    assert.ok(tokens.length >= 2, `expected at least 2 tokens, got ${tokens.length}`);
    for (const t of tokens) {
      assert.ok(t >= 0 && t < modelTp.cfg.vocabSize,
        `token ${t} out of range [0, ${modelTp.cfg.vocabSize})`);
    }
  });
});
