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
import { WorkspaceBase } from "../src/workspace";

const SMALL_MODEL_DIR = path.resolve(
  __dirname,
  "../tests/python/test_models/glm51_small/glm51_small_bf16",
);
const MAX_BATCH = 2;
const MAX_SEQ_LEN = 256;
const TREE_TOKENS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150];

const TREE_PARENTS = [
  [],
  [0],
  [0],
  [1],
  [1],
  [2],
  [2],
  [3],
  [3],
  [4],
  [4],
  [5],
  [5],
  [6],
  [6],
];

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
    const val = logits[offset + v];
    if (val > bestVal) {
      bestVal = val;
      best = v;
    }
  }
  return best;
}

describe("Model-level tree prefill vs sequential prefill", () => {
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

  it("CausalCustom with causal-equivalent mask matches built-in causal", () => {
    const seqLen = 15;
    const vocabSize = model.cfg.vocabSize;

    const causalMaskData = buildCausalSuffixMask(seqLen);
    const causalMaskIndptr = new Int32Array([0, Math.ceil(seqLen * seqLen / 8)]);

    using ccCache = model.createChatCache(128);
    ccCache.reset(1);

    const maskTensor = ws.alloc([causalMaskData.length], "U8", "test_cc_mask");
    maskTensor.h2d(Buffer.from(causalMaskData));
    const indptrTensor = ws.alloc([2], "I32", "test_cc_indptr");
    indptrTensor.h2d(i32Buf(causalMaskIndptr));

    const ccState = ws.planPrefill(model, 1, [seqLen], ccCache, {
      mask: maskTensor,
      indptr: indptrTensor,
      mode: MaskMode.CausalCustom,
    });
    ccState.setInput([TREE_TOKENS]);

    using ccHidden = model.forward(ccState);
    using ccLogits = ccState.computeLogits(ccHidden, model, null);
    const ccLogitsF32 = readBf16Logits(ccLogits, seqLen, vocabSize);

    using causalCache = model.createChatCache(128);
    causalCache.reset(1);

    const causalState = ws.planPrefill(model, 1, [seqLen], causalCache);
    causalState.setInput([TREE_TOKENS]);

    using causalHidden = model.forward(causalState);
    using causalLogits = causalState.computeLogits(causalHidden, model, null);
    const causalLogitsF32 = readBf16Logits(causalLogits, seqLen, vocabSize);

    let top1Mismatches = 0;
    let maxDiff = 0;
    for (let pos = 0; pos < seqLen; pos++) {
      const ccTop = top1(ccLogitsF32, pos * vocabSize, vocabSize);
      const causalTop = top1(causalLogitsF32, pos * vocabSize, vocabSize);
      if (ccTop !== causalTop) top1Mismatches++;
      for (let v = 0; v < vocabSize; v++) {
        const diff = Math.abs(ccLogitsF32[pos * vocabSize + v] - causalLogitsF32[pos * vocabSize + v]);
        if (diff > maxDiff) maxDiff = diff;
      }
    }

    assert.equal(top1Mismatches, 0, `${top1Mismatches} positions have different top-1 tokens between CausalCustom and built-in causal`);
    assert.ok(maxDiff < 0.5, `max logit diff ${maxDiff} between CausalCustom and built-in causal exceeds 0.5`);
  });

  it("tree prefill logits match path-by-path causal prefill logits", () => {
    const numTreeNodes = TREE_TOKENS.length;
    const vocabSize = model.cfg.vocabSize;

    const { data: treeMaskData, indptr: treeMaskIndptr } = buildTreeMask(numTreeNodes);
    const positionIds = new Int32Array(numTreeNodes);
    for (let i = 0; i < numTreeNodes; i++) {
      positionIds[i] = treeDepth(i);
    }

    using treeCache = model.createChatCache(128);
    treeCache.reset(1);

    const maskTensor = ws.alloc([treeMaskData.length], "U8", "test_tree_mask");
    maskTensor.h2d(Buffer.from(treeMaskData));
    const indptrTensor = ws.alloc([2], "I32", "test_tree_indptr");
    indptrTensor.h2d(i32Buf(treeMaskIndptr));

    const posIdTensor = ws.alloc([numTreeNodes], "I32", "test_tree_posids");
    posIdTensor.h2d(i32Buf(positionIds));

    const treeState = ws.planPrefill(model, 1, [numTreeNodes], treeCache, {
      mask: maskTensor,
      indptr: indptrTensor,
      mode: MaskMode.CausalCustom,
      positionIds: posIdTensor,
    });
    treeState.setInput([TREE_TOKENS]);

    using treeHidden = model.forward(treeState);
    using treeLogits = treeState.computeLogits(treeHidden, model, null);
    const treeLogitsF32 = readBf16Logits(treeLogits, numTreeNodes, vocabSize);

    const paths = [
      [0, 1, 3, 7],
      [0, 1, 3, 8],
      [0, 1, 4, 9],
      [0, 1, 4, 10],
      [0, 2, 5, 11],
      [0, 2, 5, 12],
      [0, 2, 6, 13],
      [0, 2, 6, 14],
    ];

    for (const pathIndices of paths) {
      const pathTokens = pathIndices.map(i => TREE_TOKENS[i]);
      const pathLen = pathTokens.length;

      using pathCache = model.createChatCache(128);
      pathCache.reset(1);

      const pathState = ws.planPrefill(model, 1, [pathLen], pathCache);
      pathState.setInput([pathTokens]);

      using pathHidden = model.forward(pathState);
      using pathLogits = pathState.computeLogits(pathHidden, model, null);
      const pathLogitsF32 = readBf16Logits(pathLogits, pathLen, vocabSize);

      let top1Mismatches = 0;
      let maxDiff = 0;
      let maxDiffPos = -1;
      let maxDiffVocab = -1;
      let maxDiffTreeVal = 0;
      let maxDiffPathVal = 0;
      const posMaxDiffs: number[] = [];
      for (let pos = 0; pos < pathLen; pos++) {
        const treeNode = pathIndices[pos];
        const treeTop = top1(treeLogitsF32, treeNode * vocabSize, vocabSize);
        const pathTop = top1(pathLogitsF32, pos * vocabSize, vocabSize);
        if (treeTop !== pathTop) top1Mismatches++;
        let posMaxDiff = 0;
        for (let v = 0; v < vocabSize; v++) {
          const diff = Math.abs(treeLogitsF32[treeNode * vocabSize + v] - pathLogitsF32[pos * vocabSize + v]);
          if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffPos = pos;
            maxDiffVocab = v;
            maxDiffTreeVal = treeLogitsF32[treeNode * vocabSize + v];
            maxDiffPathVal = pathLogitsF32[pos * vocabSize + v];
          }
          if (diff > posMaxDiff) posMaxDiff = diff;
        }
        posMaxDiffs.push(posMaxDiff);
      }

      console.log(`path ${pathIndices.join("->")}: per-position max diffs = [${posMaxDiffs.join(", ")}], global max diff ${maxDiff.toFixed(4)} at pos=${maxDiffPos} vocab=${maxDiffVocab} (tree=${maxDiffTreeVal.toFixed(4)} path=${maxDiffPathVal.toFixed(4)}), top1 mismatches=${top1Mismatches}`);
      assert.equal(top1Mismatches, 0, `path ${pathIndices.join("->")}: ${top1Mismatches}/${pathLen} positions have different top-1 tokens`);
      assert.ok(maxDiff < 0.5, `path ${pathIndices.join("->")}: max logit diff ${maxDiff} exceeds 0.5`);
    }
  });

  it("layer-by-layer KV cache comparison for tree vs path prefill", () => {
    const numTreeNodes = TREE_TOKENS.length;
    const cfg = model.cfg;
    const kvLoraRank = cfg.kvLoraRank!;
    const qkRopeDim = cfg.qkRopeHeadDim!;
    const pageSize = 16;

    const { data: treeMaskData, indptr: treeMaskIndptr } = buildTreeMask(numTreeNodes);
    const positionIds = new Int32Array(numTreeNodes);
    for (let i = 0; i < numTreeNodes; i++) {
      positionIds[i] = treeDepth(i);
    }

    using treeCache = model.createChatCache(128);
    treeCache.reset(1);

    const maskTensor = ws.alloc([treeMaskData.length], "U8", "test_kv_tree_mask");
    maskTensor.h2d(Buffer.from(treeMaskData));
    const indptrTensor = ws.alloc([2], "I32", "test_kv_tree_indptr");
    indptrTensor.h2d(i32Buf(treeMaskIndptr));
    const posIdTensor = ws.alloc([numTreeNodes], "I32", "test_kv_tree_posids");
    posIdTensor.h2d(i32Buf(positionIds));

    const treeState = ws.planPrefill(model, 1, [numTreeNodes], treeCache, {
      mask: maskTensor,
      indptr: indptrTensor,
      mode: MaskMode.CausalCustom,
      positionIds: posIdTensor,
    });
    treeState.setInput([TREE_TOKENS]);

    using treeHidden = model.forward(treeState);
    glm.synchronize();

    const treePagedKV = treeCache.getPagedKV();

    const pathIndices = [0, 1, 3, 7];
    const pathTokens = pathIndices.map(i => TREE_TOKENS[i]);
    const pathLen = pathTokens.length;

    using pathCache = model.createChatCache(128);
    pathCache.reset(1);

    const pathState = ws.planPrefill(model, 1, [pathLen], pathCache);
    pathState.setInput([pathTokens]);

    using pathHidden = model.forward(pathState);
    glm.synchronize();

    const pathPagedKV = pathCache.getPagedKV();

    for (let layer = 0; layer < cfg.numHiddenLayers; layer++) {
      const treeCkv = treePagedKV.ckvData[layer];
      const pathCkv = pathPagedKV.ckvData[layer];
      const treeKpe = treePagedKV.kpeData[layer];
      const pathKpe = pathPagedKV.kpeData[layer];

      const ckvBytesPerPage = pageSize * kvLoraRank * 2;
      const kpeBytesPerPage = pageSize * qkRopeDim * 2;

      const treeCkvBuf = Buffer.alloc(ckvBytesPerPage);
      const pathCkvBuf = Buffer.alloc(ckvBytesPerPage);
      const treeKpeBuf = Buffer.alloc(kpeBytesPerPage);
      const pathKpeBuf = Buffer.alloc(kpeBytesPerPage);

      treeCkv.d2h(treeCkvBuf);
      pathCkv.d2h(pathCkvBuf);
      treeKpe.d2h(treeKpeBuf);
      pathKpe.d2h(pathKpeBuf);

      for (let posInPath = 0; posInPath < pathLen; posInPath++) {
        const treeNodeSlot = pathIndices[posInPath];
        const pathSlot = posInPath;

        const treeCkvOffset = treeNodeSlot * kvLoraRank * 2;
        const pathCkvOffset = pathSlot * kvLoraRank * 2;
        const treeKpeOffset = treeNodeSlot * qkRopeDim * 2;
        const pathKpeOffset = pathSlot * qkRopeDim * 2;

        let ckvMaxDiff = 0;
        for (let i = 0; i < kvLoraRank; i++) {
          const treeVal = bf16BytesToF32(treeCkvBuf.subarray(treeCkvOffset + i * 2, treeCkvOffset + i * 2 + 2))[0];
          const pathVal = bf16BytesToF32(pathCkvBuf.subarray(pathCkvOffset + i * 2, pathCkvOffset + i * 2 + 2))[0];
          const diff = Math.abs(treeVal - pathVal);
          if (diff > ckvMaxDiff) ckvMaxDiff = diff;
        }

        let kpeMaxDiff = 0;
        for (let i = 0; i < qkRopeDim; i++) {
          const treeVal = bf16BytesToF32(treeKpeBuf.subarray(treeKpeOffset + i * 2, treeKpeOffset + i * 2 + 2))[0];
          const pathVal = bf16BytesToF32(pathKpeBuf.subarray(pathKpeOffset + i * 2, pathKpeOffset + i * 2 + 2))[0];
          const diff = Math.abs(treeVal - pathVal);
          if (diff > kpeMaxDiff) kpeMaxDiff = diff;
        }

        console.log(`  layer ${layer} path-pos ${posInPath} (tree-slot ${treeNodeSlot} vs path-slot ${pathSlot}): CKV maxDiff=${ckvMaxDiff.toFixed(6)} KPE maxDiff=${kpeMaxDiff.toFixed(6)}`);
      }
    }
  });
});
