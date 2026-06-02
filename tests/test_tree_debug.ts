import assert from "node:assert/strict";
import path from "node:path";
import { after, before, describe, it } from "node:test";
import type { ChatCache } from "../src/chat_model";
import { MaskMode } from "../src/device_ops";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { bf16BytesToF32, I32 } from "../src/glm_ops";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps } from "../src/glm_ops";
import { MemcpyKind } from "../src/tensor";

const SMALL_MODEL_DIR = path.resolve(__dirname, "../tests/python/test_models/glm51_small/glm51_small_bf16");
const MAX_BATCH = 2;
const MAX_SEQ_LEN = 256;

function treeDepth(node: number): number {
  let depth = 0;
  while (node > 0) { node = (node - 1) >> 1; depth++; }
  return depth;
}

function buildTreeMask(numNodes: number): { data: Uint8Array; indptr: Int32Array } {
  const totalBits = numNodes * numNodes;
  const byteLen = Math.ceil(totalBits / 8);
  const data = new Uint8Array(byteLen);
  for (let q = 0; q < numNodes; q++) {
    let cur = q;
    while (true) {
      const bit = q * numNodes + cur;
      data[bit >> 3] |= 1 << (bit & 7);
      if (cur === 0) break;
      cur = (cur - 1) >> 1;
    }
  }
  const indptr = new Int32Array(2);
  indptr[0] = 0;
  indptr[1] = byteLen;
  return { data, indptr };
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

function i32Buf(data: Int32Array): Buffer {
  const buf = Buffer.alloc(data.length * 4);
  for (let i = 0; i < data.length; i++) buf.writeInt32LE(data[i], i * 4);
  return buf;
}

describe("Tree prefill debug diagnostics", () => {
  let glm: GlmOps;
  let model: Glm51Model;
  let ws: ExecutionWorkspace;

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = await Glm51Model.fromPretrained(glm, SMALL_MODEL_DIR);
    ws = new ExecutionWorkspace(glm, MAX_BATCH, MAX_SEQ_LEN);
  });

  after(() => {
    ws.free();
    model.free();
    glm.free();
  });

  it("mask tensor survives model forward pass (8-node tree)", () => {
    const numNodes = 8;
    const { data: maskData, indptr: maskIndptr } = buildTreeMask(numNodes);
    const positionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) positionIds[i] = treeDepth(i);
    const tokens = [10, 20, 20, 30, 30, 30, 30, 40];

    const maskTensor = ws.alloc([maskData.length], "U8", "debug_mask");
    maskTensor.h2d(Buffer.from(maskData));
    const indptrTensor = ws.alloc([2], "I32", "debug_indptr");
    indptrTensor.h2d(i32Buf(maskIndptr));
    const posIdTensor = ws.alloc([numNodes], "I32", "debug_posids");
    posIdTensor.h2d(i32Buf(positionIds));

    using cache = model.createChatCache(128);
    cache.reset(1);

    const state = ws.planPrefill(model, 1, [numNodes], cache, {
      mask: maskTensor,
      indptr: indptrTensor,
      mode: MaskMode.CausalCustom,
      positionIds: posIdTensor,
    });
    state.setInput([tokens]);

    using hidden = model.forward(state);
    glm.synchronize();

    const readbackMask = Buffer.alloc(maskData.length);
    maskTensor.d2h(readbackMask);
    const readbackIndptr = Buffer.alloc(8);
    indptrTensor.d2h(readbackIndptr);

    console.log("  Original mask bytes:", Array.from(maskData));
    console.log("  Readback mask bytes:", Array.from(readbackMask));
    console.log("  Original indptr:", Array.from(maskIndptr));
    console.log("  Readback indptr:", [readbackIndptr.readInt32LE(0), readbackIndptr.readInt32LE(4)]);

    let maskMatch = true;
    for (let i = 0; i < maskData.length; i++) {
      if (maskData[i] !== readbackMask[i]) { maskMatch = false; break; }
    }
    assert.ok(maskMatch, "Mask tensor was corrupted during forward pass!");

    let indptrMatch = readbackIndptr.readInt32LE(0) === maskIndptr[0] &&
                      readbackIndptr.readInt32LE(4) === maskIndptr[1];
    assert.ok(indptrMatch, "Indptr tensor was corrupted during forward pass!");
  });

  it("position IDs tensor survives model forward pass (8-node tree)", () => {
    const numNodes = 8;
    const { data: maskData, indptr: maskIndptr } = buildTreeMask(numNodes);
    const positionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) positionIds[i] = treeDepth(i);
    const tokens = [10, 20, 20, 30, 30, 30, 30, 40];

    const maskTensor = ws.alloc([maskData.length], "U8", "debug_mask2");
    maskTensor.h2d(Buffer.from(maskData));
    const indptrTensor = ws.alloc([2], "I32", "debug_indptr2");
    indptrTensor.h2d(i32Buf(maskIndptr));
    const posIdTensor = ws.alloc([numNodes], "I32", "debug_posids2");
    posIdTensor.h2d(i32Buf(positionIds));

    using cache = model.createChatCache(128);
    cache.reset(1);

    const state = ws.planPrefill(model, 1, [numNodes], cache, {
      mask: maskTensor,
      indptr: indptrTensor,
      mode: MaskMode.CausalCustom,
      positionIds: posIdTensor,
    });
    state.setInput([tokens]);

    using hidden = model.forward(state);
    glm.synchronize();

    const readbackPosIds = Buffer.alloc(numNodes * 4);
    posIdTensor.d2h(readbackPosIds);
    const readbackPos = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) readbackPos[i] = readbackPosIds.readInt32LE(i * 4);

    console.log("  Original position IDs:", Array.from(positionIds));
    console.log("  Readback position IDs:", Array.from(readbackPos));
    console.log("  Workspace position IDs (sequential):", Array.from({ length: numNodes }, (_, i) => i));

    let posMatch = true;
    for (let i = 0; i < numNodes; i++) {
      if (positionIds[i] !== readbackPos[i]) { posMatch = false; break; }
    }
    assert.ok(posMatch, "Position IDs tensor was corrupted during forward pass!");
  });

  it("8-node tree vs 2-node causal: compare layer 0 CKV/KPE for nodes 0,1", () => {
    const numNodes = 8;
    const cfg = model.cfg;
    const kvLoraRank = cfg.kvLoraRank!;
    const qkRopeDim = cfg.qkRopeHeadDim!;
    const pageSize = 16;
    const vocabSize = cfg.vocabSize;

    const { data: maskData, indptr: maskIndptr } = buildTreeMask(numNodes);
    const positionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) positionIds[i] = treeDepth(i);
    const tokens = [10, 20, 20, 30, 30, 30, 30, 40];

    // Tree prefill
    {
      using cache = model.createChatCache(128);
      cache.reset(1);

      const maskTensor = ws.alloc([maskData.length], "U8", "kv_tree_mask");
      maskTensor.h2d(Buffer.from(maskData));
      const indptrTensor = ws.alloc([2], "I32", "kv_tree_indptr");
      indptrTensor.h2d(i32Buf(maskIndptr));
      const posIdTensor = ws.alloc([numNodes], "I32", "kv_tree_posids");
      posIdTensor.h2d(i32Buf(positionIds));

      const state = ws.planPrefill(model, 1, [numNodes], cache, {
        mask: maskTensor,
        indptr: indptrTensor,
        mode: MaskMode.CausalCustom,
        positionIds: posIdTensor,
      });
      state.setInput([tokens]);

      using hidden = model.forward(state);
      glm.synchronize();

      const treePagedKV = cache.getPagedKV();

      // Path prefill [10, 20] with positions [0, 1]
      using pathCache = model.createChatCache(128);
      pathCache.reset(1);

      const pathState = ws.planPrefill(model, 1, [2], pathCache);
      pathState.setInput([[10, 20]]);

      using pathHidden = model.forward(pathState);
      glm.synchronize();

      const pathPagedKV = pathCache.getPagedKV();

      // Compare CKV and KPE at layer 0 for slots 0 and 1
      const treeCkv = treePagedKV.ckvData[0];
      const pathCkv = pathPagedKV.ckvData[0];
      const treeKpe = treePagedKV.kpeData[0];
      const pathKpe = pathPagedKV.kpeData[0];

      const ckvBytesPerSlot = kvLoraRank * 2;
      const kpeBytesPerSlot = qkRopeDim * 2;

      const treeCkvBuf = Buffer.alloc(2 * ckvBytesPerSlot);
      const pathCkvBuf = Buffer.alloc(2 * ckvBytesPerSlot);
      const treeKpeBuf = Buffer.alloc(2 * kpeBytesPerSlot);
      const pathKpeBuf = Buffer.alloc(2 * kpeBytesPerSlot);

      treeCkv.d2h(treeCkvBuf);
      pathCkv.d2h(pathCkvBuf);
      treeKpe.d2h(treeKpeBuf);
      pathKpe.d2h(pathKpeBuf);

      for (let slot = 0; slot < 2; slot++) {
        let ckvMaxDiff = 0;
        for (let i = 0; i < kvLoraRank; i++) {
          const treeVal = bf16BytesToF32(treeCkvBuf.subarray(slot * ckvBytesPerSlot + i * 2, slot * ckvBytesPerSlot + i * 2 + 2))[0];
          const pathVal = bf16BytesToF32(pathCkvBuf.subarray(slot * ckvBytesPerSlot + i * 2, slot * ckvBytesPerSlot + i * 2 + 2))[0];
          const diff = Math.abs(treeVal - pathVal);
          if (diff > ckvMaxDiff) ckvMaxDiff = diff;
        }
        let kpeMaxDiff = 0;
        for (let i = 0; i < qkRopeDim; i++) {
          const treeVal = bf16BytesToF32(treeKpeBuf.subarray(slot * kpeBytesPerSlot + i * 2, slot * kpeBytesPerSlot + i * 2 + 2))[0];
          const pathVal = bf16BytesToF32(pathKpeBuf.subarray(slot * kpeBytesPerSlot + i * 2, slot * kpeBytesPerSlot + i * 2 + 2))[0];
          const diff = Math.abs(treeVal - pathVal);
          if (diff > kpeMaxDiff) kpeMaxDiff = diff;
        }
        console.log(`  Layer 0 slot ${slot}: CKV maxDiff=${ckvMaxDiff.toFixed(6)} KPE maxDiff=${kpeMaxDiff.toFixed(6)}`);
        assert.ok(ckvMaxDiff < 0.01, `Layer 0 slot ${slot}: CKV diff ${ckvMaxDiff} too large`);
        assert.ok(kpeMaxDiff < 0.01, `Layer 0 slot ${slot}: KPE diff ${kpeMaxDiff} too large`);
      }
    }
  });

  it("8-node tree CausalCustom vs 8-node causal-equivalent CausalCustom: mask is the only variable", () => {
    const numNodes = 8;
    const vocabSize = model.cfg.vocabSize;
    const tokens = [10, 20, 30, 40, 50, 60, 70, 80];

    // Tree mask
    const { data: treeMaskData, indptr: treeMaskIndptr } = buildTreeMask(numNodes);
    const treePositionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) treePositionIds[i] = treeDepth(i);

    // Causal-equivalent mask (lower triangular, same as built-in causal)
    const causalMaskData = buildCausalMask(numNodes);
    const causalMaskIndptr = new Int32Array([0, Math.ceil(numNodes * numNodes / 8)]);
    // For causal-equivalent, use same sequential position IDs as built-in causal
    const causalPositionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) causalPositionIds[i] = i;

    // Run with tree mask + depth position IDs
    {
      using cache = model.createChatCache(128);
      cache.reset(1);
      const maskTensor = ws.alloc([treeMaskData.length], "U8", "cmp_tree_mask");
      maskTensor.h2d(Buffer.from(treeMaskData));
      const indptrTensor = ws.alloc([2], "I32", "cmp_tree_indptr");
      indptrTensor.h2d(i32Buf(treeMaskIndptr));
      const posIdTensor = ws.alloc([numNodes], "I32", "cmp_tree_posids");
      posIdTensor.h2d(i32Buf(treePositionIds));

      const state = ws.planPrefill(model, 1, [numNodes], cache, {
        mask: maskTensor, indptr: indptrTensor, mode: MaskMode.CausalCustom, positionIds: posIdTensor,
      });
      state.setInput([tokens]);

      using hidden = model.forward(state);
      using logits = state.computeLogits(hidden, model, null);
      const treeLogits = Buffer.alloc(numNodes * vocabSize * 2);
      logits.d2h(treeLogits);
      const treeF32 = bf16BytesToF32(treeLogits);

      // Run with causal-equivalent mask + sequential position IDs
      using cache2 = model.createChatCache(128);
      cache2.reset(1);
      const causalMaskTensor = ws.alloc([causalMaskData.length], "U8", "cmp_causal_mask");
      causalMaskTensor.h2d(Buffer.from(causalMaskData));
      const causalIndptrTensor = ws.alloc([2], "I32", "cmp_causal_indptr");
      causalIndptrTensor.h2d(i32Buf(causalMaskIndptr));
      const causalPosIdTensor = ws.alloc([numNodes], "I32", "cmp_causal_posids");
      causalPosIdTensor.h2d(i32Buf(causalPositionIds));

      const state2 = ws.planPrefill(model, 1, [numNodes], cache2, {
        mask: causalMaskTensor, indptr: causalIndptrTensor, mode: MaskMode.CausalCustom, positionIds: causalPosIdTensor,
      });
      state2.setInput([tokens]);

      using hidden2 = model.forward(state2);
      using logits2 = state2.computeLogits(hidden2, model, null);
      const causalLogits = Buffer.alloc(numNodes * vocabSize * 2);
      logits2.d2h(causalLogits);
      const causalF32 = bf16BytesToF32(causalLogits);

      // Node 0 attends to {0} in both masks
      // Node 1 attends to {0,1} in tree mask AND {0,1} in causal mask
      // So node 0 and 1 should have the same output if the only difference is the mask
      // But position IDs differ: tree uses [0,1,1,2,...], causal uses [0,1,2,3,...]
      // Node 0 has same position (0) in both, node 1 has same position (1) in both
      // So node 0 and 1 logits should match
      for (let node = 0; node < 2; node++) {
        let maxDiff = 0;
        for (let v = 0; v < vocabSize; v++) {
          const diff = Math.abs(treeF32[node * vocabSize + v] - causalF32[node * vocabSize + v]);
          if (diff > maxDiff) maxDiff = diff;
        }
        console.log(`  Node ${node} (same KV, same position, same mask for this node): max logit diff = ${maxDiff.toFixed(6)}`);
      }

      // Also compare with built-in causal (no mask tensor)
      using cache3 = model.createChatCache(128);
      cache3.reset(1);
      const state3 = ws.planPrefill(model, 1, [numNodes], cache3);
      state3.setInput([tokens]);

      using hidden3 = model.forward(state3);
      using logits3 = state3.computeLogits(hidden3, model, null);
      const builtinLogits = Buffer.alloc(numNodes * vocabSize * 2);
      logits3.d2h(builtinLogits);
      const builtinF32 = bf16BytesToF32(builtinLogits);

      let causalVsBuiltinMaxDiff = 0;
      for (let i = 0; i < numNodes * vocabSize; i++) {
        const diff = Math.abs(causalF32[i] - builtinF32[i]);
        if (diff > causalVsBuiltinMaxDiff) causalVsBuiltinMaxDiff = diff;
      }
      console.log(`  CausalCustom (causal mask) vs built-in causal: max logit diff = ${causalVsBuiltinMaxDiff.toFixed(6)}`);

      // The causal-equivalent CausalCustom should match built-in causal
      assert.ok(causalVsBuiltinMaxDiff < 0.5, `CausalCustom causal mask vs built-in causal: max diff ${causalVsBuiltinMaxDiff} > 0.5`);
    }
  });

  it("8-node tree with sequential positions vs built-in causal: node 0,1 comparison", () => {
    const numNodes = 8;
    const vocabSize = model.cfg.vocabSize;
    const tokens = [10, 20, 30, 40, 50, 60, 70, 80];

    // Tree mask but with SEQUENTIAL position IDs [0,1,2,3,4,5,6,7]
    const { data: treeMaskData, indptr: treeMaskIndptr } = buildTreeMask(numNodes);
    const seqPositionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) seqPositionIds[i] = i;

    using cache = model.createChatCache(128);
    cache.reset(1);
    const maskTensor = ws.alloc([treeMaskData.length], "U8", "seq_tree_mask");
    maskTensor.h2d(Buffer.from(treeMaskData));
    const indptrTensor = ws.alloc([2], "I32", "seq_tree_indptr");
    indptrTensor.h2d(i32Buf(treeMaskIndptr));
    const posIdTensor = ws.alloc([numNodes], "I32", "seq_tree_posids");
    posIdTensor.h2d(i32Buf(seqPositionIds));

    const state = ws.planPrefill(model, 1, [numNodes], cache, {
      mask: maskTensor, indptr: indptrTensor, mode: MaskMode.CausalCustom, positionIds: posIdTensor,
    });
    state.setInput([tokens]);

    using hidden = model.forward(state);
    using logits = state.computeLogits(hidden, model, null);
    const treeLogits = Buffer.alloc(numNodes * vocabSize * 2);
    logits.d2h(treeLogits);
    const treeF32 = bf16BytesToF32(treeLogits);

    // Built-in causal with same tokens
    using cache2 = model.createChatCache(128);
    cache2.reset(1);
    const state2 = ws.planPrefill(model, 1, [numNodes], cache2);
    state2.setInput([tokens]);

    using hidden2 = model.forward(state2);
    using logits2 = state2.computeLogits(hidden2, model, null);
    const causalLogits = Buffer.alloc(numNodes * vocabSize * 2);
    logits2.d2h(causalLogits);
    const causalF32 = bf16BytesToF32(causalLogits);

    // Node 0: tree mask allows {0}, causal allows {0} — same
    // Node 1: tree mask allows {0,1}, causal allows {0,1} — same
    // Both use position IDs 0 and 1 respectively
    for (let node = 0; node < 2; node++) {
      let maxDiff = 0;
      for (let v = 0; v < vocabSize; v++) {
        const diff = Math.abs(treeF32[node * vocabSize + v] - causalF32[node * vocabSize + v]);
        if (diff > maxDiff) maxDiff = diff;
      }
      console.log(`  Node ${node} (tree+seq vs causal): max logit diff = ${maxDiff.toFixed(6)}`);
      assert.ok(maxDiff < 0.5, `Node ${node}: tree+seq vs causal diff ${maxDiff} > 0.5`);
    }
  });

  it("7-node tree vs 2-node causal: verify 7-node works at model level", () => {
    const numNodes = 7;
    const vocabSize = model.cfg.vocabSize;
    const tokens = [10, 20, 30, 40, 50, 60, 70];

    const { data: maskData, indptr: maskIndptr } = buildTreeMask(numNodes);
    const positionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) positionIds[i] = treeDepth(i);

    using cache = model.createChatCache(128);
    cache.reset(1);
    const maskTensor = ws.alloc([maskData.length], "U8", "7tree_mask");
    maskTensor.h2d(Buffer.from(maskData));
    const indptrTensor = ws.alloc([2], "I32", "7tree_indptr");
    indptrTensor.h2d(i32Buf(maskIndptr));
    const posIdTensor = ws.alloc([numNodes], "I32", "7tree_posids");
    posIdTensor.h2d(i32Buf(positionIds));

    const state = ws.planPrefill(model, 1, [numNodes], cache, {
      mask: maskTensor, indptr: indptrTensor, mode: MaskMode.CausalCustom, positionIds: posIdTensor,
    });
    state.setInput([tokens]);

    using hidden = model.forward(state);
    using logits = state.computeLogits(hidden, model, null);
    const treeLogits = Buffer.alloc(numNodes * vocabSize * 2);
    logits.d2h(treeLogits);
    const treeF32 = bf16BytesToF32(treeLogits);

    // Path [0, 1] with positions [0, 1]
    using cache2 = model.createChatCache(128);
    cache2.reset(1);
    const pathPosIds = ws.alloc([2], "I32", "7path_posids");
    pathPosIds.h2d(i32Buf(new Int32Array([0, 1])));
    const causalMask = buildCausalMask(2);
    const causalMaskIndptr = new Int32Array([0, Math.ceil(2 * 2 / 8)]);
    const causalMaskTensor = ws.alloc([causalMask.length], "U8", "7path_cmask");
    causalMaskTensor.h2d(Buffer.from(causalMask));
    const causalIndptrTensor = ws.alloc([2], "I32", "7path_cindptr");
    causalIndptrTensor.h2d(i32Buf(causalMaskIndptr));

    const state2 = ws.planPrefill(model, 1, [2], cache2, {
      mask: causalMaskTensor, indptr: causalIndptrTensor, mode: MaskMode.CausalCustom, positionIds: pathPosIds,
    });
    state2.setInput([[10, 20]]);

    using hidden2 = model.forward(state2);
    using logits2 = state2.computeLogits(hidden2, model, null);
    const pathLogits = Buffer.alloc(2 * vocabSize * 2);
    logits2.d2h(pathLogits);
    const pathF32 = bf16BytesToF32(pathLogits);

    // Compare node 1 in tree vs position 1 in path
    let maxDiff = 0;
    for (let v = 0; v < vocabSize; v++) {
      const diff = Math.abs(treeF32[1 * vocabSize + v] - pathF32[1 * vocabSize + v]);
      if (diff > maxDiff) maxDiff = diff;
    }
    console.log(`  7-node tree: node 1 vs path [0,1] position 1: max logit diff = ${maxDiff.toFixed(6)}`);
    assert.ok(maxDiff < 0.5, `7-node tree node 1 vs path: diff ${maxDiff} > 0.5`);
  });
});
