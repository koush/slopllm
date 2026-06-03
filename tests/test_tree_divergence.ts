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

function buildCausalSuffixMask(qoLen: number): Uint8Array {
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

function treeDepth(node: number): number {
  let depth = 0;
  while (node > 0) {
    node = (node - 1) >> 1;
    depth++;
  }
  return depth;
}

function i32Buf(data: Int32Array): Buffer {
  const buf = Buffer.alloc(data.length * 4);
  for (let i = 0; i < data.length; i++) buf.writeInt32LE(data[i], i * 4);
  return buf;
}

function readBf16Logits(tensor: { d2h: (buf: Buffer) => void }, numTokens: number, vocabSize: number): Float32Array {
  const buf = Buffer.alloc(numTokens * vocabSize * 2);
  tensor.d2h(buf);
  return bf16BytesToF32(buf);
}

function top1(logits: Float32Array, offset: number, vocabSize: number): number {
  let best = -1;
  let bestVal = -Infinity;
  for (let v = 0; v < vocabSize; v++) {
    if (logits[offset + v] > bestVal) { bestVal = logits[offset + v]; best = v; }
  }
  return best;
}

function compareLogits(treeLogits: Float32Array, pathLogits: Float32Array, treeNode: number, pathPos: number, vocabSize: number, label: string) {
  let maxDiff = 0;
  let top1Match = true;
  const treeTop = top1(treeLogits, treeNode * vocabSize, vocabSize);
  const pathTop = top1(pathLogits, pathPos * vocabSize, vocabSize);
  if (treeTop !== pathTop) top1Match = false;
  for (let v = 0; v < vocabSize; v++) {
    const diff = Math.abs(treeLogits[treeNode * vocabSize + v] - pathLogits[pathPos * vocabSize + v]);
    if (diff > maxDiff) maxDiff = diff;
  }
  return { maxDiff, top1Match };
}

interface PrefillResult {
  logits: Float32Array;
  cache: ChatCache;
  state: import("../src/execution-workspace").ExecutionState;
}

describe("Tree prefill divergence diagnosis", () => {
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

  function treePrefill(tokens: number[], positionIds: Int32Array, maskData: Uint8Array, maskIndptr: Int32Array, maskMode: MaskMode) {
    const numTokens = tokens.length;
    const vocabSize = model.cfg.vocabSize;
    using cache = model.createChatCache(128);
    cache.reset(1);

    const maskTensor = ws.alloc([maskData.length], "U8");
    maskTensor.h2d(Buffer.from(maskData));
    const indptrTensor = ws.alloc([2], "I32");
    indptrTensor.h2d(i32Buf(maskIndptr));
    const posIdTensor = ws.alloc([numTokens], "I32");
    posIdTensor.h2d(i32Buf(positionIds));

    const state = ws.planPrefill(model, 1, [numTokens], cache, {
      mask: maskTensor,
      indptr: indptrTensor,
      mode: maskMode,
      positionIds: posIdTensor,
    });
    state.setInput([tokens]);

    using hidden = model.forward(state);
    using logits = state.computeLogits(hidden, model, true);
    const logitsF32 = readBf16Logits(logits, numTokens, vocabSize);
    return { logits: logitsF32, cache, state };
  }

  function causalPrefill(tokens: number[], positionIds?: Int32Array) {
    const numTokens = tokens.length;
    const vocabSize = model.cfg.vocabSize;
    using cache = model.createChatCache(128);
    cache.reset(1);

    let state;
    if (positionIds) {
      const posIdTensor = ws.alloc([numTokens], "I32");
      posIdTensor.h2d(i32Buf(positionIds));
      const causalMask = buildCausalSuffixMask(numTokens);
      const causalIndptr = new Int32Array([0, Math.ceil(numTokens * numTokens / 8)]);
      const maskTensor = ws.alloc([causalMask.length], "U8");
      maskTensor.h2d(Buffer.from(causalMask));
      const indptrTensor = ws.alloc([2], "I32");
      indptrTensor.h2d(i32Buf(causalIndptr));
      state = ws.planPrefill(model, 1, [numTokens], cache, {
        mask: maskTensor,
        indptr: indptrTensor,
        mode: MaskMode.CausalCustom,
        positionIds: posIdTensor,
      });
    } else {
      state = ws.planPrefill(model, 1, [numTokens], cache);
    }
    state.setInput([tokens]);

    using hidden = model.forward(state);
    using logits = state.computeLogits(hidden, model, true);
    const logitsF32 = readBf16Logits(logits, numTokens, vocabSize);
    return { logits: logitsF32, cache, state };
  }

  // Test 1: Depth-3 tree (7 nodes) — this should match the ops-level test that passed
  it("depth-3 tree (7 nodes) logits match path-by-path causal", () => {
    const numNodes = 7;
    const tokens = [10, 20, 30, 40, 50, 60, 70];
    const vocabSize = model.cfg.vocabSize;

    const { data: maskData, indptr: maskIndptr } = buildTreeMask(numNodes);
    const positionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) positionIds[i] = treeDepth(i);

    const treeResult = treePrefill(tokens, positionIds, maskData, maskIndptr, MaskMode.CausalCustom);

    const paths = [[0,1,3], [0,1,4], [0,2,5], [0,2,6]];
    for (const pathIndices of paths) {
      const pathTokens = pathIndices.map(i => tokens[i]);
      const pathPositionIds = new Int32Array(pathIndices.map(i => treeDepth(i)));
      const pathResult = causalPrefill(pathTokens, pathPositionIds);

      for (let pos = 0; pos < pathIndices.length; pos++) {
        const treeNode = pathIndices[pos];
        const { maxDiff, top1Match } = compareLogits(treeResult.logits, pathResult.logits, treeNode, pos, vocabSize, "");
        assert.ok(top1Match, `depth-3 path ${pathIndices.join("->")} pos ${pos}: top1 mismatch (tree=${top1(treeResult.logits, treeNode*vocabSize, vocabSize)} path=${top1(pathResult.logits, pos*vocabSize, vocabSize)})`);
        assert.ok(maxDiff < 0.5, `depth-3 path ${pathIndices.join("->")} pos ${pos}: max logit diff ${maxDiff} exceeds 0.5`);
      }
    }
  });

  // Test 2: Same-tokens depth-4 tree — every path has the same token sequence [10, 20, 30, 40]
  // arranged by depth, so compare against a single causal prefill of [10, 20, 30, 40]
  it("same-tokens depth-4 tree: each depth gets same token, compare vs single causal prefill", () => {
    const numNodes = 15;
    const depthTokens = [10, 20, 30, 40]; // token for depth 0, 1, 2, 3
    const tokens = new Array(numNodes);
    for (let i = 0; i < numNodes; i++) tokens[i] = depthTokens[treeDepth(i)];
    const vocabSize = model.cfg.vocabSize;

    const { data: maskData, indptr: maskIndptr } = buildTreeMask(numNodes);
    const positionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) positionIds[i] = treeDepth(i);

    const treeResult = treePrefill(tokens, positionIds, maskData, maskIndptr, MaskMode.CausalCustom);

    // Reference: causal prefill of [10, 20, 30, 40] with position IDs [0, 1, 2, 3]
    const refTokens = depthTokens;
    const refPositionIds = new Int32Array([0, 1, 2, 3]);
    const refResult = causalPrefill(refTokens, refPositionIds);

    const paths = [[0,1,3,7], [0,1,3,8], [0,1,4,9], [0,1,4,10], [0,2,5,11], [0,2,5,12], [0,2,6,13], [0,2,6,14]];
    for (const pathIndices of paths) {
      for (let pos = 0; pos < pathIndices.length; pos++) {
        const treeNode = pathIndices[pos];
        const { maxDiff, top1Match } = compareLogits(treeResult.logits, refResult.logits, treeNode, pos, vocabSize, "");
        assert.ok(top1Match, `same-tokens path ${pathIndices.join("->")} pos ${pos} (tree-node ${treeNode}): top1 mismatch`);
        assert.ok(maxDiff < 0.5, `same-tokens path ${pathIndices.join("->")} pos ${pos} (tree-node ${treeNode}): max logit diff ${maxDiff} exceeds 0.5`);
      }
    }
  });

  // Test 3: Tree prefill with SEQUENTIAL position IDs [0,1,2,...,14] instead of depth-based.
  // If this passes, the bug is in depth-based position IDs.
  // If this fails, the bug is in the tree mask itself.
  it("depth-4 tree with sequential position IDs vs causal path", () => {
    const numNodes = 15;
    const tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150];
    const vocabSize = model.cfg.vocabSize;

    const { data: maskData, indptr: maskIndptr } = buildTreeMask(numNodes);
    // Use sequential position IDs instead of depth-based
    const seqPositionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) seqPositionIds[i] = i;

    const treeResult = treePrefill(tokens, seqPositionIds, maskData, maskIndptr, MaskMode.CausalCustom);

    // For path [0,1,3,7] with sequential IDs:
    // Tree node 0: pos 0, node 1: pos 1, node 3: pos 3, node 7: pos 7
    // Path prefill must use matching position IDs [0, 1, 3, 7]
    const pathIndices = [0, 1, 3, 7];
    const pathTokens = pathIndices.map(i => tokens[i]);
    const pathPositionIds = new Int32Array(pathIndices.map(i => i));
    const pathResult = causalPrefill(pathTokens, pathPositionIds);

    const posMaxDiffs: number[] = [];
    for (let pos = 0; pos < pathIndices.length; pos++) {
      const treeNode = pathIndices[pos];
      let posMaxDiff = 0;
      for (let v = 0; v < vocabSize; v++) {
        const diff = Math.abs(treeResult.logits[treeNode * vocabSize + v] - pathResult.logits[pos * vocabSize + v]);
        if (diff > posMaxDiff) posMaxDiff = diff;
      }
      posMaxDiffs.push(posMaxDiff);
    }
    console.log(`  sequential-pos path 0->1->3->7 per-pos max diffs: [${posMaxDiffs.map(d => d.toFixed(4)).join(", ")}]`);

    // Check if sequential IDs fix the issue (tolerance 0.5)
    for (let pos = 0; pos < pathIndices.length; pos++) {
      const treeNode = pathIndices[pos];
      const { maxDiff, top1Match } = compareLogits(treeResult.logits, pathResult.logits, treeNode, pos, vocabSize, "");
      assert.ok(top1Match, `seq-pos path 0->1->3->7 pos ${pos}: top1 mismatch`);
      assert.ok(maxDiff < 0.5, `seq-pos path 0->1->3->7 pos ${pos}: max logit diff ${maxDiff} exceeds 0.5`);
    }
  });

  // Test 4: Depth-4 tree with depth-based position IDs but CUSTOM mask mode (not CausalCustom).
  // This tests whether CausalCustom mask mode specifically is the issue.
  it("depth-4 tree with Custom mask mode (full mask) vs CausalCustom", () => {
    const numNodes = 15;
    const tokens = new Array(numNodes);
    const depthTokens = [10, 20, 30, 40];
    for (let i = 0; i < numNodes; i++) tokens[i] = depthTokens[treeDepth(i)];
    const vocabSize = model.cfg.vocabSize;

    const { data: maskData, indptr: maskIndptr } = buildTreeMask(numNodes);
    const positionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) positionIds[i] = treeDepth(i);

    // Run with CausalCustom (no prefix)
    const ccResult = treePrefill(tokens, positionIds, maskData, maskIndptr, MaskMode.CausalCustom);

    // Run with Custom (full mask = same mask data, but no prefix assumption)
    // For Custom mode with no prefix, we need the full mask to cover all tokens
    // The tree mask already covers all tokens, so it's the same mask data
    const customResult = treePrefill(tokens, positionIds, maskData, maskIndptr, MaskMode.Custom);

    // They should produce the same output since prefix_len=0 for CausalCustom
    let maxDiff = 0;
    for (let i = 0; i < numNodes * vocabSize; i++) {
      const diff = Math.abs(ccResult.logits[i] - customResult.logits[i]);
      if (diff > maxDiff) maxDiff = diff;
    }
    console.log(`  CausalCustom vs Custom max logit diff: ${maxDiff.toFixed(6)}`);
    assert.ok(maxDiff < 0.01, `CausalCustom vs Custom: max diff ${maxDiff} exceeds 0.01`);
  });

  // Test 5: 4-token flat tree (depth 2: root + 3 children) — minimal non-trivial tree
  it("4-token flat tree (root + 3 children) vs path causal", () => {
    // Tree: node 0 is root, nodes 1,2,3 are children of root
    // Mask: node 0 attends {0}, node 1 attends {0,1}, node 2 attends {0,2}, node 3 attends {0,3}
    const numNodes = 4;
    const tokens = [10, 20, 30, 40];
    const vocabSize = model.cfg.vocabSize;

    const maskData = new Uint8Array(Math.ceil(numNodes * numNodes / 8));
    for (let q = 0; q < numNodes; q++) {
      const bit0 = q * numNodes + 0;
      const bitSelf = q * numNodes + q;
      maskData[bit0 >> 3] |= 1 << (bit0 & 7);
      maskData[bitSelf >> 3] |= 1 << (bitSelf & 7);
    }
    const maskIndptr = new Int32Array([0, maskData.length]);

    const positionIds = new Int32Array([0, 1, 1, 1]);

    const treeResult = treePrefill(tokens, positionIds, maskData, maskIndptr, MaskMode.CausalCustom);

    const ref2Result = causalPrefill([10, 20], new Int32Array([0, 1]));
    const { maxDiff: diff1, top1Match: match1 } = compareLogits(treeResult.logits, ref2Result.logits, 1, 1, vocabSize, "");
    assert.ok(match1, `flat-tree node 1: top1 mismatch`);
    assert.ok(diff1 < 0.5, `flat-tree node 1: max logit diff ${diff1} exceeds 0.5`);

    const ref3Result = causalPrefill([10, 30], new Int32Array([0, 1]));
    const { maxDiff: diff2, top1Match: match2 } = compareLogits(treeResult.logits, ref3Result.logits, 2, 1, vocabSize, "");
    assert.ok(match2, `flat-tree node 2: top1 mismatch`);
    assert.ok(diff2 < 0.5, `flat-tree node 2: max logit diff ${diff2} exceeds 0.5`);

    const ref4Result = causalPrefill([10, 40], new Int32Array([0, 1]));
    const { maxDiff: diff3, top1Match: match3 } = compareLogits(treeResult.logits, ref4Result.logits, 3, 1, vocabSize, "");
    assert.ok(match3, `flat-tree node 3: top1 mismatch`);
    assert.ok(diff3 < 0.5, `flat-tree node 3: max logit diff ${diff3} exceeds 0.5`);
  });

  // Test 6: Binary tree sweep from 3 to 15 nodes — find divergence threshold
  it("binary tree size sweep: find divergence threshold", () => {
    const vocabSize = model.cfg.vocabSize;
    const depthTokens = [10, 20, 30, 40];

    for (let numNodes = 3; numNodes <= 15; numNodes++) {
      const tokens = new Array(numNodes);
      for (let i = 0; i < numNodes; i++) tokens[i] = depthTokens[Math.min(treeDepth(i), depthTokens.length - 1)];

      const { data: maskData, indptr: maskIndptr } = buildTreeMask(numNodes);
      const positionIds = new Int32Array(numNodes);
      for (let i = 0; i < numNodes; i++) positionIds[i] = treeDepth(i);

      const treeResult = treePrefill(tokens, positionIds, maskData, maskIndptr, MaskMode.CausalCustom);

      if (numNodes >= 2) {
        const refResult = causalPrefill([tokens[0], tokens[1]], new Int32Array([0, 1]));
        let maxDiff = 0;
        for (let v = 0; v < vocabSize; v++) {
          const diff = Math.abs(treeResult.logits[1 * vocabSize + v] - refResult.logits[1 * vocabSize + v]);
          if (diff > maxDiff) maxDiff = diff;
        }
        const status = maxDiff < 0.5 ? "PASS" : "FAIL";
        console.log(`  ${numNodes} nodes (pageSize=16): node 1 vs [0,1] causal maxDiff=${maxDiff.toFixed(4)} ${status}`);
        if (numNodes <= 7) {
          assert.ok(maxDiff < 0.5, `${numNodes} nodes: node 1 max diff ${maxDiff} exceeds 0.5`);
        }
      }
    }
  });

  // Test 6b: 8-node tree with pageSize=32 — does larger page size fix it?
  it("8-node tree with pageSize=32", () => {
    const numNodes = 8;
    const vocabSize = model.cfg.vocabSize;
    const tokens = [10, 20, 20, 30, 30, 30, 30, 40];

    const { data: maskData, indptr: maskIndptr } = buildTreeMask(numNodes);
    const positionIds = new Int32Array(numNodes);
    for (let i = 0; i < numNodes; i++) positionIds[i] = treeDepth(i);

    using cache = model.createChatCache(128, 1, 4096, 32);
    cache.reset(1);

    const maskTensor = ws.alloc([maskData.length], "U8");
    maskTensor.h2d(Buffer.from(maskData));
    const indptrTensor = ws.alloc([2], "I32");
    indptrTensor.h2d(i32Buf(maskIndptr));
    const posIdTensor = ws.alloc([numNodes], "I32");
    posIdTensor.h2d(i32Buf(positionIds));

    const state = ws.planPrefill(model, 1, [numNodes], cache, {
      mask: maskTensor,
      indptr: indptrTensor,
      mode: MaskMode.CausalCustom,
      positionIds: posIdTensor,
    });
    state.setInput([tokens]);

    using hidden = model.forward(state);
    using logitsTensor = state.computeLogits(hidden, model, true);
    const treeLogits = readBf16Logits(logitsTensor, numNodes, vocabSize);

    // Reference: causal prefill of [10, 20] with pageSize=32
    using refCache = model.createChatCache(128, 1, 4096, 32);
    refCache.reset(1);
    const refState = ws.planPrefill(model, 1, [2], refCache);
    refState.setInput([[tokens[0], tokens[1]]]);

    using refHidden = model.forward(refState);
    using refLogitsTensor = refState.computeLogits(refHidden, model, true);
    const refLogits = readBf16Logits(refLogitsTensor, 2, vocabSize);

    let maxDiff = 0;
    for (let v = 0; v < vocabSize; v++) {
      const diff = Math.abs(treeLogits[1 * vocabSize + v] - refLogits[1 * vocabSize + v]);
      if (diff > maxDiff) maxDiff = diff;
    }
    const status = maxDiff < 0.5 ? "PASS" : "FAIL";
    console.log(`  8 nodes (pageSize=32): node 1 vs [0,1] causal maxDiff=${maxDiff.toFixed(4)} ${status}`);
    assert.ok(maxDiff < 0.5, `8 nodes pageSize=32: node 1 max diff ${maxDiff} exceeds 0.5`);
  });

  // Test 8: Dump MLA plan info for 7 vs 8 tokens
  it("dump MLA plan info for 7 vs 8 tokens with tree mask", () => {
    const vocabSize = model.cfg.vocabSize;
    const nHeads = model.cfg.numAttentionHeads;

    for (const numTokens of [7, 8]) {
      const tokens = new Array(numTokens).fill(10);
      const { data: maskData, indptr: maskIndptr } = buildTreeMask(numTokens);
      const positionIds = new Int32Array(numTokens);
      for (let i = 0; i < numTokens; i++) positionIds[i] = treeDepth(i);

      using cache = model.createChatCache(128);
      cache.reset(1);

      const maskTensor = ws.alloc([maskData.length], "U8");
      maskTensor.h2d(Buffer.from(maskData));
      const indptrTensor = ws.alloc([2], "I32");
      indptrTensor.h2d(i32Buf(maskIndptr));
      const posIdTensor = ws.alloc([numTokens], "I32");
      posIdTensor.h2d(i32Buf(positionIds));

      const state = ws.planPrefill(model, 1, [numTokens], cache, {
        mask: maskTensor,
        indptr: indptrTensor,
        mode: MaskMode.CausalCustom,
        positionIds: posIdTensor,
      });

      // Read plan info
      const planBuf = Buffer.alloc(19 * 8);
      ws.mlaPrefillPlanInfo.d2h(planBuf);
      const planInfo = [];
      for (let i = 0; i < 19; i++) {
        planInfo.push(Number(planBuf.readBigInt64LE(i * 8)));
      }
      console.log(`  ${numTokens} tokens (tree, CausalCustom, nHeads=${nHeads}):`);
      console.log(`    num_blks_x (cluster_size)=${planInfo[0]} num_blks_y (num_clusters)=${planInfo[1]}`);
      console.log(`    planInfo[2..18]=${planInfo.slice(2).join(", ")}`);
      console.log(`    packed_qo_len=${numTokens * nHeads} avg_packed_qo_len=${numTokens * nHeads}`);

      // Also plan with built-in causal (no mask)
      using cache2 = model.createChatCache(128);
      cache2.reset(1);
      const state2 = ws.planPrefill(model, 1, [numTokens], cache2);
      ws.mlaPrefillPlanInfo.d2h(planBuf);
      const planInfo2 = [];
      for (let i = 0; i < 19; i++) {
        planInfo2.push(Number(planBuf.readBigInt64LE(i * 8)));
      }
      console.log(`  ${numTokens} tokens (causal, nHeads=${nHeads}):`);
      console.log(`    num_blks_x (cluster_size)=${planInfo2[0]} num_blks_y (num_clusters)=${planInfo2[1]}`);
      console.log(`    planInfo[2..18]=${planInfo2.slice(2).join(", ")}`);
    }
  });
});
