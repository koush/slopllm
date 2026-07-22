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

function runMtpTreeDecode(model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache, nextn: number, label: string): number[] {
  const firstTokens = ws.forwardEagerPrefill(model, [INPUT_IDS], cache);
  cache.reportTokens(0, INPUT_IDS);
  ws.updateIndptr(cache.getPagedKV());
  const firstToken = firstTokens[0];

  using captureManager = new CaptureManager(ws.glm);
  using sampleWorkspace = new WorkspaceBase(ws.glm);

  const state = ws.planDecode(model, 1, cache);
  state.setInput([[firstToken]]);
  ws.positionStep(state, model);

  using hiddenHolder = new UsingHolder<Tensor>(undefined!);
  using sharedSlots = new UsingHolder<Tensor>(undefined!);
  using sharedSlotsLength = new UsingHolder<Tensor>(undefined!);
  state.sharedSlots = sharedSlots;
  state.sharedSlotsLength = sharedSlotsLength;
  const hidden = model.forward(state);
  hiddenHolder.replace(hidden.removeTracking());

  using logits = state.computeLogits(hiddenHolder.value, model);
  using gpuSampleResult = greedySample(logits, sampleWorkspace);

  const currentTokenHost = sampleWorkspace.allocPinned(gpuSampleResult.shape, gpuSampleResult.type);
  currentTokenHost.memcpy(gpuSampleResult, gpuSampleResult.bytes, MemcpyKind.DeviceToHost);
  ws.glm.synchronize();
  const currentToken = currentTokenHost.readPinnedBuffer().readInt32LE();

  const { tokens: result } = mtpTreeDecode(
    captureManager, model, hiddenHolder.value, sharedSlots, sharedSlotsLength, ws, currentToken, Array(nextn).fill(2), cache,
  );

  for (const t of result) {
    assert.ok(t >= 0 && t < model.cfg.vocabSize,
      `${label}: token ${t} out of range [0, ${model.cfg.vocabSize})`);
  }

  return result;
}

describe("MTP tree decode: TP validation", () => {
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

  for (const nextn of [1, 3]) {
    it(`single-GPU MTP tree decode with nextn=${nextn} produces valid tokens`, () => {
      using cache = modelRef.createChatCache(128);
      cache.reset(1);
      runMtpTreeDecode(modelRef, wsRef, cache, nextn, `single-GPU-nextn${nextn}`);
    });

    it(`multi-GPU TP MTP tree decode with nextn=${nextn} produces valid tokens`, () => {
      using cache = modelTp.createChatCache(128);
      cache.reset(1);
      runMtpTreeDecode(modelTp, wsTp, cache, nextn, `multi-GPU-TP-nextn${nextn}`);
    });
  }
});
