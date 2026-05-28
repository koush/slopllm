import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { Tensor } from "../src/tensor";

const HEAD_DIM_CKV = 512;
const HEAD_DIM_KPE = 64;
const N_HEADS = 4;
const SM_SCALE = 1.0 / Math.sqrt(HEAD_DIM_CKV);

const I32 = 4;

function i32Buf(data: Int32Array): Buffer {
  const buf = Buffer.alloc(data.length * 4);
  for (let i = 0; i < data.length; i++) buf.writeInt32LE(data[i], i * 4);
  return buf;
}

function allocBf16(ws: WorkspaceBase, shape: number[]): Tensor {
  return ws.alloc(shape, "BF16");
}

function allocI32(ws: WorkspaceBase, shape: number[]): Tensor {
  return ws.alloc(shape, "I32");
}

function allocPinnedI32(ws: WorkspaceBase, shape: number[]): Tensor {
  return ws.allocPinned(shape, "I32");
}

function randomData(n: number): Float32Array {
  const f32 = new Float32Array(n);
  for (let i = 0; i < n; i++) f32[i] = (Math.random() * 2 - 1) * 0.5;
  return f32;
}

function packBitsLittleEndian(mask: boolean[], qoLen: number, kvLen: number): { data: Uint8Array; indptr: Int32Array } {
  const batchSize = 1;
  const totalBits = qoLen * kvLen;
  const byteLen = Math.ceil(totalBits / 8);
  const data = new Uint8Array(byteLen);

  for (let q = 0; q < qoLen; q++) {
    for (let k = 0; k < kvLen; k++) {
      const offset = q * kvLen + k;
      if (mask[offset]) {
        data[offset >> 3] |= 1 << (offset & 7);
      }
    }
  }

  const indptr = new Int32Array(batchSize + 1);
  indptr[0] = 0;
  indptr[1] = byteLen;

  return { data, indptr };
}

function buildTreeMask(parents: number[][], numTokens: number, kvLen: number): boolean[] {
  const mask = new Array<boolean>(numTokens * kvLen).fill(false);
  for (let q = 0; q < numTokens; q++) {
    mask[q * kvLen + q] = true;
    let cur = q;
    while (parents[cur].length > 0) {
      cur = parents[cur][0];
      mask[q * kvLen + cur] = true;
    }
  }
  return mask;
}

function buildCausalMask(qoLen: number, kvLen: number): boolean[] {
  const mask = new Array<boolean>(qoLen * kvLen).fill(false);
  const offset = kvLen - qoLen;
  for (let q = 0; q < qoLen; q++) {
    for (let k = 0; k <= offset + q; k++) {
      mask[q * kvLen + k] = true;
    }
  }
  return mask;
}

interface PrefillResult {
  o: Tensor;
  lse: Tensor;
}

function runMlaPrefill(
  glm: GlmOps,
  ws: WorkspaceBase,
  qNopeF32: Float32Array,
  qPeF32: Float32Array,
  ckvF32: Float32Array,
  kpeF32: Float32Array,
  seqLen: number,
  kvSeqLen: number,
  batchSize: number,
  nHeads: number,
  headDimCkv: number,
  headDimKpe: number,
  pageSize: number,
  maxPages: number,
  causal: boolean,
  customMaskData?: Uint8Array,
  maskIndptrData?: Int32Array,
): PrefillResult {
  const totalQTokens = batchSize * seqLen;
  const numPages = Math.ceil(kvSeqLen / pageSize);
  const lastPageLen = kvSeqLen % pageSize || pageSize;

  const qNope = allocBf16(ws, [1, totalQTokens, nHeads * headDimCkv]);
  const qPe = allocBf16(ws, [1, totalQTokens, nHeads * headDimKpe]);
  qNope.h2d(f32ToBf16Bytes(qNopeF32));
  qPe.h2d(f32ToBf16Bytes(qPeF32));

  const ckv = allocBf16(ws, [ckvF32.length]);
  const kpe = allocBf16(ws, [kpeF32.length]);
  ckv.h2d(f32ToBf16Bytes(ckvF32));
  kpe.h2d(f32ToBf16Bytes(kpeF32));

  const indices = allocI32(ws, [maxPages]);
  const indicesData = new Int32Array(maxPages);
  for (let i = 0; i < numPages; i++) indicesData[i] = i;
  indices.h2d(i32Buf(indicesData));

  const indptrArr = new Int32Array(batchSize + 1);
  indptrArr[0] = 0;
  indptrArr[1] = numPages;
  const indptrH = allocPinnedI32(ws, [batchSize + 1]);
  indptrH.h2d(i32Buf(indptrArr));

  const lastPageLenH = allocPinnedI32(ws, [batchSize]);
  lastPageLenH.h2d(i32Buf(new Int32Array([lastPageLen])));

  const floatWs = allocBf16(ws, [128 * 1024 * 1024 / 2]);
  const intWs = allocI32(ws, [8 * 1024 * 1024 / 4]);
  const pinnedIntWs = allocPinnedI32(ws, [8 * 1024 * 1024 / 4]);
  const planInfo = allocPinnedI32(ws, [19]);

  const kvLenH = allocPinnedI32(ws, [batchSize]);
  kvLenH.h2d(i32Buf(new Int32Array([kvSeqLen])));

  const qoIndptrH = allocPinnedI32(ws, [batchSize + 1]);
  qoIndptrH.h2d(i32Buf(new Int32Array([0, totalQTokens])));

  glm.mlaPrefillPlan(
    floatWs, 128 * 1024 * 1024,
    intWs, pinnedIntWs, 8 * 1024 * 1024,
    planInfo,
    qoIndptrH, indptrH, kvLenH, lastPageLenH,
    batchSize, nHeads, headDimCkv, causal,
    pageSize, [kvSeqLen],
  );

  const ckvStridePage = pageSize * headDimCkv;
  const kpeStridePage = pageSize * headDimKpe;
  const ckvStrideN = headDimCkv;
  const kpeStrideN = headDimKpe;
  const qNopeStrideN = nHeads * headDimCkv;
  const qNopeStrideH = headDimCkv;
  const qPeStrideN = nHeads * headDimKpe;
  const qPeStrideH = headDimKpe;
  const oStrideN = headDimCkv;
  const oStrideH = totalQTokens * headDimCkv;

  let customMask: Tensor | undefined;
  let maskIndptr: Tensor | undefined;
  let maskMode: number;

  if (customMaskData && maskIndptrData) {
    maskMode = 2;
    customMask = ws.alloc([customMaskData.length], "U8");
    customMask.h2d(Buffer.from(customMaskData));
    maskIndptr = ws.alloc([maskIndptrData.length], "I32");
    maskIndptr.h2d(i32Buf(maskIndptrData));
  } else {
    maskMode = causal ? 1 : 0;
  }

  const result = glm.mlaPrefillRun(
    qNope, qPe, ckv, kpe, indices,
    floatWs, intWs, planInfo,
    nHeads, pageSize, maskMode, SM_SCALE,
    qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH,
    ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN,
    oStrideN, oStrideH,
    headDimCkv, headDimKpe,
    false, 0, 0,
    customMask, maskIndptr,
  );
  glm.synchronize();

  return result;
}

function readOutput(o: Tensor, totalTokens: number, nHeads: number, headDim: number): Float32Array {
  const bytes = totalTokens * nHeads * headDim * 2;
  const buf = Buffer.alloc(bytes);
  o.d2h(buf);
  return bf16BytesToF32(buf);
}

function readLse(lseTensor: Tensor, totalTokens: number, nHeads: number): Float32Array {
  const buf = Buffer.alloc(totalTokens * nHeads * 4);
  lseTensor.d2h(buf);
  const f32 = new Float32Array(totalTokens * nHeads);
  for (let i = 0; i < totalTokens * nHeads; i++) f32[i] = buf.readFloatLE(i * 4);
  return f32;
}

describe("MLA custom mask", () => {
  let glm: GlmOps;
  let ws: WorkspaceBase;

  before(() => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    ws = new WorkspaceBase(glm);
  });

  after(() => {
    ws.free();
    glm.free();
  });

  it("tree mask: no crash and valid output", () => {
    const pageSize = 16;
    const numTokens = 15;
    const numPages = Math.ceil(numTokens / pageSize);
    const maxPages = numPages + 1;

    const parents: number[][] = [
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

    const mask = buildTreeMask(parents, numTokens, numTokens);
    const { data: packedMask, indptr: maskIndptr } = packBitsLittleEndian(mask, numTokens, numTokens);

    const qNopeF32 = randomData(numTokens * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(numTokens * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const { o, lse } = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      numTokens, numTokens, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      false,
      packedMask, maskIndptr,
    );

    const output = readOutput(o, numTokens, N_HEADS, HEAD_DIM_CKV);
    const lseData = readLse(lse, numTokens, N_HEADS);

    assert.equal(output.length, numTokens * N_HEADS * HEAD_DIM_CKV, "output length");
    assert.equal(lseData.length, numTokens * N_HEADS, "lse length");

    let hasFinite = false;
    for (let i = 0; i < lseData.length; i++) {
      if (isFinite(lseData[i])) {
        hasFinite = true;
      } else {
        assert.fail(`lse[${i}] is not finite: ${lseData[i]}`);
      }
    }
    assert.ok(hasFinite, "at least one finite LSE value");

    let hasNonZero = false;
    for (let i = 0; i < output.length; i++) {
      if (output[i] !== 0) hasNonZero = true;
    }
    assert.ok(hasNonZero, "output should not be all zeros");
  });

  it("tree mask: path output matches standalone causal prefill", () => {
    const pageSize = 16;
    const numTokens = 15;
    const numPages = Math.ceil(numTokens / pageSize);
    const maxPages = numPages + 1;

    const parents: number[][] = [
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

    const mask = buildTreeMask(parents, numTokens, numTokens);
    const { data: packedMask, indptr: maskIndptr } = packBitsLittleEndian(mask, numTokens, numTokens);

    const qNopeF32 = randomData(numTokens * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(numTokens * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const { o: treeO, lse: treeLse } = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      numTokens, numTokens, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      false,
      packedMask, maskIndptr,
    );

    const treeOutput = readOutput(treeO, numTokens, N_HEADS, HEAD_DIM_CKV);
    const treeLseArr = readLse(treeLse, numTokens, N_HEADS);

    const pathIndices = [0, 1, 3, 7];
    const pathLen = pathIndices.length;

    const qNopeRowBytes = N_HEADS * HEAD_DIM_CKV;
    const qPeRowBytes = N_HEADS * HEAD_DIM_KPE;
    const ckvRowBytes = HEAD_DIM_CKV;
    const kpeRowBytes = HEAD_DIM_KPE;

    const qNopePath = new Float32Array(pathLen * qNopeRowBytes);
    const qPePath = new Float32Array(pathLen * qPeRowBytes);
    for (let i = 0; i < pathLen; i++) {
      const srcIdx = pathIndices[i];
      for (let j = 0; j < qNopeRowBytes; j++) qNopePath[i * qNopeRowBytes + j] = qNopeF32[srcIdx * qNopeRowBytes + j];
      for (let j = 0; j < qPeRowBytes; j++) qPePath[i * qPeRowBytes + j] = qPeF32[srcIdx * qPeRowBytes + j];
    }

    const pathNumPages = Math.ceil(pathLen / pageSize);
    const pathMaxPages = pathNumPages + 1;
    const ckvPath = new Float32Array(pathNumPages * pageSize * HEAD_DIM_CKV);
    const kpePath = new Float32Array(pathNumPages * pageSize * HEAD_DIM_KPE);
    for (let i = 0; i < pathLen; i++) {
      const srcIdx = pathIndices[i];
      for (let j = 0; j < ckvRowBytes; j++) ckvPath[i * ckvRowBytes + j] = ckvF32[srcIdx * ckvRowBytes + j];
      for (let j = 0; j < kpeRowBytes; j++) kpePath[i * kpeRowBytes + j] = kpeF32[srcIdx * kpeRowBytes + j];
    }

    const { o: pathO, lse: pathLse } = runMlaPrefill(
      glm, ws, qNopePath, qPePath, ckvPath, kpePath,
      pathLen, pathLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, pathMaxPages,
      true,
    );

    const pathOutput = readOutput(pathO, pathLen, N_HEADS, HEAD_DIM_CKV);
    const pathLseArr = readLse(pathLse, pathLen, N_HEADS);

    let maxDiff = 0;
    for (let h = 0; h < N_HEADS; h++) {
      for (let d = 0; d < HEAD_DIM_CKV; d++) {
        const treeIdx = h * numTokens * HEAD_DIM_CKV + 7 * HEAD_DIM_CKV + d;
        const pathIdx = h * pathLen * HEAD_DIM_CKV + 3 * HEAD_DIM_CKV + d;
        const treeVal = treeOutput[treeIdx];
        const pathVal = pathOutput[pathIdx];
        const diff = Math.abs(treeVal - pathVal);
        if (diff > maxDiff) maxDiff = diff;
      }
    }

    let maxLseDiff = 0;
    for (let h = 0; h < N_HEADS; h++) {
      const treeVal = treeLseArr[7 * N_HEADS + h];
      const pathVal = pathLseArr[3 * N_HEADS + h];
      const diff = Math.abs(treeVal - pathVal);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    assert.ok(maxDiff < 0.05, `tree path output max diff ${maxDiff} exceeds tolerance 0.05`);
    assert.ok(maxLseDiff < 0.05, `tree path LSE max diff ${maxLseDiff} exceeds tolerance 0.05`);
  });

  it("causal-equivalent custom mask matches built-in causal", () => {
    const pageSize = 16;
    const seqLen = 7;
    const numPages = Math.ceil(seqLen / pageSize);
    const maxPages = numPages + 1;

    const causalMask = buildCausalMask(seqLen, seqLen);
    const { data: packedMask, indptr: maskIndptr } = packBitsLittleEndian(causalMask, seqLen, seqLen);

    const qNopeF32 = randomData(seqLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(seqLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const customResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, seqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      false,
      packedMask, maskIndptr,
    );

    const builtinResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, seqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
    );

    const customOutput = readOutput(customResult.o, seqLen, N_HEADS, HEAD_DIM_CKV);
    const builtinOutput = readOutput(builtinResult.o, seqLen, N_HEADS, HEAD_DIM_CKV);

    const customLse = readLse(customResult.lse, seqLen, N_HEADS);
    const builtinLse = readLse(builtinResult.lse, seqLen, N_HEADS);

    let maxDiff = 0;
    for (let i = 0; i < customOutput.length; i++) {
      const diff = Math.abs(customOutput[i] - builtinOutput[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    let maxLseDiff = 0;
    for (let i = 0; i < customLse.length; i++) {
      const diff = Math.abs(customLse[i] - builtinLse[i]);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    assert.ok(maxDiff < 0.05, `max output diff ${maxDiff} exceeds tolerance 0.05`);
    assert.ok(maxLseDiff < 0.05, `max LSE diff ${maxLseDiff} exceeds tolerance 0.05`);
  });
});
