import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { MaskMode } from "../src/device_ops";
import { WorkspaceBase } from "../src/workspace";
import { Tensor } from "../src/tensor";
import type { ExecutionState } from "../src/execution-workspace";

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

function buildCausalCustomSuffixMask(parents: number[][], numPrefillTokens: number): boolean[] {
  const mask = new Array<boolean>(numPrefillTokens * numPrefillTokens).fill(false);
  for (let q = 0; q < numPrefillTokens; q++) {
    mask[q * numPrefillTokens + q] = true;
    let cur = q;
    while (parents[cur].length > 0) {
      cur = parents[cur][0];
      mask[q * numPrefillTokens + cur] = true;
    }
  }
  return mask;
}

function buildFullMaskFromSuffix(suffixMask: boolean[], qoLen: number, kvLen: number): boolean[] {
  const prefixLen = kvLen - qoLen;
  const mask = new Array<boolean>(qoLen * kvLen).fill(false);
  for (let q = 0; q < qoLen; q++) {
    for (let k = 0; k < prefixLen; k++) {
      mask[q * kvLen + k] = true;
    }
    for (let k = 0; k < qoLen; k++) {
      mask[q * kvLen + prefixLen + k] = suffixMask[q * qoLen + k];
    }
  }
  return mask;
}

function buildCausalSuffixMask(qoLen: number): boolean[] {
  const mask = new Array<boolean>(qoLen * qoLen).fill(false);
  for (let q = 0; q < qoLen; q++) {
    for (let k = 0; k <= q; k++) {
      mask[q * qoLen + k] = true;
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
  maskModeOverride?: MaskMode,
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
  let maskMode: MaskMode;

  if (customMaskData && maskIndptrData) {
    maskMode = maskModeOverride ?? MaskMode.Custom;
    customMask = ws.alloc([customMaskData.length], "U8");
    customMask.h2d(Buffer.from(customMaskData));
    maskIndptr = ws.alloc([maskIndptrData.length], "I32");
    maskIndptr.h2d(i32Buf(maskIndptrData));
  } else {
    maskMode = causal ? MaskMode.Causal : MaskMode.None;
  }

  const execState = { batchSize, totalTokens: totalQTokens, cache: { getPagedKV: () => ({ pageSize }) } } as ExecutionState;
  const result = glm.mlaPrefillRun(
    execState,
    qNope, qPe, ckv, kpe, indices,
    floatWs, intWs, planInfo,
    nHeads, pageSize, maskMode, SM_SCALE,
    qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH,
    ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN,
    oStrideN, oStrideH,
    headDimCkv, headDimKpe,
    0, 0,
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

  it("causal-custom tree mask with prefix: matches full custom mask", () => {
    const pageSize = 16;
    const prefixLen = 32;
    const numPrefillTokens = 15;
    const qoLen = numPrefillTokens;
    const kvSeqLen = prefixLen + numPrefillTokens;
    const numPages = Math.ceil(kvSeqLen / pageSize);
    const maxPages = numPages + 1;

    const parents: number[][] = [
      [], [0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6],
    ];

    const suffixMask = buildCausalCustomSuffixMask(parents, numPrefillTokens);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, qoLen, qoLen);

    const fullMask = buildFullMaskFromSuffix(suffixMask, qoLen, kvSeqLen);
    const { data: packedFullMask, indptr: fullMaskIndptr } = packBitsLittleEndian(fullMask, qoLen, kvSeqLen);

    const qNopeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const ccResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const customResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      false,
      packedFullMask, fullMaskIndptr,
    );

    const ccOutput = readOutput(ccResult.o, qoLen, N_HEADS, HEAD_DIM_CKV);
    const customOutput = readOutput(customResult.o, qoLen, N_HEADS, HEAD_DIM_CKV);

    const ccLse = readLse(ccResult.lse, qoLen, N_HEADS);
    const customLse = readLse(customResult.lse, qoLen, N_HEADS);

    let maxDiff = 0;
    for (let i = 0; i < ccOutput.length; i++) {
      const diff = Math.abs(ccOutput[i] - customOutput[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    let maxLseDiff = 0;
    for (let i = 0; i < ccLse.length; i++) {
      const diff = Math.abs(ccLse[i] - customLse[i]);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    assert.ok(maxDiff < 0.05, `causal-custom vs full custom: max output diff ${maxDiff} exceeds tolerance 0.05`);
    assert.ok(maxLseDiff < 0.05, `causal-custom vs full custom: max LSE diff ${maxLseDiff} exceeds tolerance 0.05`);
  });

  it("causal-custom with causal suffix mask matches built-in causal", () => {
    const pageSize = 16;
    const prefixLen = 32;
    const numPrefillTokens = 7;
    const qoLen = numPrefillTokens;
    const kvSeqLen = prefixLen + numPrefillTokens;
    const numPages = Math.ceil(kvSeqLen / pageSize);
    const maxPages = numPages + 1;

    const suffixMask = buildCausalSuffixMask(qoLen);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, qoLen, qoLen);

    const qNopeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const ccResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const causalResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
    );

    const ccOutput = readOutput(ccResult.o, qoLen, N_HEADS, HEAD_DIM_CKV);
    const causalOutput = readOutput(causalResult.o, qoLen, N_HEADS, HEAD_DIM_CKV);

    const ccLse = readLse(ccResult.lse, qoLen, N_HEADS);
    const causalLse = readLse(causalResult.lse, qoLen, N_HEADS);

    let maxDiff = 0;
    for (let i = 0; i < ccOutput.length; i++) {
      const diff = Math.abs(ccOutput[i] - causalOutput[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    let maxLseDiff = 0;
    for (let i = 0; i < ccLse.length; i++) {
      const diff = Math.abs(ccLse[i] - causalLse[i]);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    assert.ok(maxDiff < 0.05, `causal-custom with causal suffix vs built-in causal: max output diff ${maxDiff} exceeds tolerance 0.05`);
    assert.ok(maxLseDiff < 0.05, `causal-custom with causal suffix vs built-in causal: max LSE diff ${maxLseDiff} exceeds tolerance 0.05`);
  });

  it("causal-custom tree mask: every position of every path matches standalone causal (prefix=32, 4-level tree)", () => {
    const pageSize = 16;
    const prefixLen = 32;
    const numPrefillTokens = 15;
    const qoLen = numPrefillTokens;
    const kvSeqLen = prefixLen + numPrefillTokens;
    const numPages = Math.ceil(kvSeqLen / pageSize);
    const maxPages = numPages + 1;

    const parents: number[][] = [
      [],
      [0], [0],
      [1], [1], [2], [2],
      [3], [3], [4], [4], [5], [5], [6], [6],
    ];

    const suffixMask = buildCausalCustomSuffixMask(parents, numPrefillTokens);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, qoLen, qoLen);

    const qNopeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const ccResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const ccOutput = readOutput(ccResult.o, qoLen, N_HEADS, HEAD_DIM_CKV);
    const ccLse = readLse(ccResult.lse, qoLen, N_HEADS);

    const qNopeRowBytes = N_HEADS * HEAD_DIM_CKV;
    const qPeRowBytes = N_HEADS * HEAD_DIM_KPE;
    const ckvRowBytes = HEAD_DIM_CKV;
    const kpeRowBytes = HEAD_DIM_KPE;

    const pathsToTest = [
      [0, 1, 3, 7],
      [0, 1, 3, 8],
      [0, 1, 4, 9],
      [0, 1, 4, 10],
      [0, 2, 5, 11],
      [0, 2, 5, 12],
      [0, 2, 6, 13],
      [0, 2, 6, 14],
    ];

    for (const pathIndices of pathsToTest) {
      const pathLen = pathIndices.length;
      const pathKvLen = prefixLen + pathLen;

      const qNopePath = new Float32Array(pathLen * qNopeRowBytes);
      const qPePath = new Float32Array(pathLen * qPeRowBytes);
      for (let i = 0; i < pathLen; i++) {
        const srcIdx = pathIndices[i];
        for (let j = 0; j < qNopeRowBytes; j++) qNopePath[i * qNopeRowBytes + j] = qNopeF32[srcIdx * qNopeRowBytes + j];
        for (let j = 0; j < qPeRowBytes; j++) qPePath[i * qPeRowBytes + j] = qPeF32[srcIdx * qPeRowBytes + j];
      }

      const pathNumPages = Math.ceil(pathKvLen / pageSize);
      const pathMaxPages = pathNumPages + 1;
      const ckvPath = new Float32Array(pathNumPages * pageSize * HEAD_DIM_CKV);
      const kpePath = new Float32Array(pathNumPages * pageSize * HEAD_DIM_KPE);
      for (let i = 0; i < prefixLen; i++) {
        for (let j = 0; j < ckvRowBytes; j++) ckvPath[i * ckvRowBytes + j] = ckvF32[i * ckvRowBytes + j];
        for (let j = 0; j < kpeRowBytes; j++) kpePath[i * kpeRowBytes + j] = kpeF32[i * kpeRowBytes + j];
      }
      for (let i = 0; i < pathLen; i++) {
        const srcIdx = prefixLen + pathIndices[i];
        for (let j = 0; j < ckvRowBytes; j++) ckvPath[(prefixLen + i) * ckvRowBytes + j] = ckvF32[srcIdx * ckvRowBytes + j];
        for (let j = 0; j < kpeRowBytes; j++) kpePath[(prefixLen + i) * kpeRowBytes + j] = kpeF32[srcIdx * kpeRowBytes + j];
      }

      const { o: pathO, lse: pathLse } = runMlaPrefill(
        glm, ws, qNopePath, qPePath, ckvPath, kpePath,
        pathLen, pathKvLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
        pageSize, pathMaxPages,
        true,
      );

      const pathOutput = readOutput(pathO, pathLen, N_HEADS, HEAD_DIM_CKV);
      const pathLseArr = readLse(pathLse, pathLen, N_HEADS);

      let maxDiff = 0;
      for (let pos = 0; pos < pathLen; pos++) {
        const treeNode = pathIndices[pos];
        for (let h = 0; h < N_HEADS; h++) {
          for (let d = 0; d < HEAD_DIM_CKV; d++) {
            const treeIdx = h * numPrefillTokens * HEAD_DIM_CKV + treeNode * HEAD_DIM_CKV + d;
            const pathIdx = h * pathLen * HEAD_DIM_CKV + pos * HEAD_DIM_CKV + d;
            const diff = Math.abs(ccOutput[treeIdx] - pathOutput[pathIdx]);
            if (diff > maxDiff) maxDiff = diff;
          }
        }
      }

      let maxLseDiff = 0;
      for (let pos = 0; pos < pathLen; pos++) {
        const treeNode = pathIndices[pos];
        for (let h = 0; h < N_HEADS; h++) {
          const treeVal = ccLse[treeNode * N_HEADS + h];
          const pathVal = pathLseArr[pos * N_HEADS + h];
          const diff = Math.abs(treeVal - pathVal);
          if (diff > maxLseDiff) maxLseDiff = diff;
        }
      }

      assert.ok(maxDiff < 0.05, `path ${pathIndices.join("->")} output max diff ${maxDiff} exceeds tolerance 0.05`);
      assert.ok(maxLseDiff < 0.05, `path ${pathIndices.join("->")} LSE max diff ${maxLseDiff} exceeds tolerance 0.05`);
    }
  });

  it("causal-custom with large prefix matches built-in causal", () => {
    const pageSize = 16;
    const prefixLen = 128;
    const numPrefillTokens = 7;
    const qoLen = numPrefillTokens;
    const kvSeqLen = prefixLen + numPrefillTokens;
    const numPages = Math.ceil(kvSeqLen / pageSize);
    const maxPages = numPages + 1;

    const suffixMask = buildCausalSuffixMask(qoLen);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, qoLen, qoLen);

    const qNopeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const ccResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const causalResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
    );

    const ccOutput = readOutput(ccResult.o, qoLen, N_HEADS, HEAD_DIM_CKV);
    const causalOutput = readOutput(causalResult.o, qoLen, N_HEADS, HEAD_DIM_CKV);

    const ccLse = readLse(ccResult.lse, qoLen, N_HEADS);
    const causalLse = readLse(causalResult.lse, qoLen, N_HEADS);

    let maxDiff = 0;
    for (let i = 0; i < ccOutput.length; i++) {
      const diff = Math.abs(ccOutput[i] - causalOutput[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    let maxLseDiff = 0;
    for (let i = 0; i < ccLse.length; i++) {
      const diff = Math.abs(ccLse[i] - causalLse[i]);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    assert.ok(maxDiff < 0.05, `large prefix causal-custom vs causal: max output diff ${maxDiff} exceeds tolerance 0.05`);
    assert.ok(maxLseDiff < 0.05, `large prefix causal-custom vs causal: max LSE diff ${maxLseDiff} exceeds tolerance 0.05`);
  });

  it("causal-custom no crash and valid output", () => {
    const pageSize = 16;
    const prefixLen = 32;
    const numPrefillTokens = 15;
    const qoLen = numPrefillTokens;
    const kvSeqLen = prefixLen + numPrefillTokens;
    const numPages = Math.ceil(kvSeqLen / pageSize);
    const maxPages = numPages + 1;

    const parents: number[][] = [
      [], [0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6],
    ];

    const suffixMask = buildCausalCustomSuffixMask(parents, numPrefillTokens);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, qoLen, qoLen);

    const qNopeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const { o, lse } = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const output = readOutput(o, qoLen, N_HEADS, HEAD_DIM_CKV);
    const lseData = readLse(lse, qoLen, N_HEADS);

    assert.equal(output.length, qoLen * N_HEADS * HEAD_DIM_CKV, "output length");
    assert.equal(lseData.length, qoLen * N_HEADS, "lse length");

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

  it("causal-custom with no prefix (prefix_len=0) matches built-in causal", () => {
    const pageSize = 16;
    const seqLen = 7;
    const kvSeqLen = seqLen;
    const numPages = Math.ceil(kvSeqLen / pageSize);
    const maxPages = numPages + 1;

    const suffixMask = buildCausalSuffixMask(seqLen);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, seqLen, seqLen);

    const qNopeF32 = randomData(seqLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(seqLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const ccResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const causalResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
    );

    const ccOutput = readOutput(ccResult.o, seqLen, N_HEADS, HEAD_DIM_CKV);
    const causalOutput = readOutput(causalResult.o, seqLen, N_HEADS, HEAD_DIM_CKV);

    const ccLse = readLse(ccResult.lse, seqLen, N_HEADS);
    const causalLse = readLse(causalResult.lse, seqLen, N_HEADS);

    let maxDiff = 0;
    for (let i = 0; i < ccOutput.length; i++) {
      const diff = Math.abs(ccOutput[i] - causalOutput[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    let maxLseDiff = 0;
    for (let i = 0; i < ccLse.length; i++) {
      const diff = Math.abs(ccLse[i] - causalLse[i]);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    assert.ok(maxDiff < 0.05, `no-prefix causal-custom vs causal: max output diff ${maxDiff} exceeds tolerance 0.05`);
    assert.ok(maxLseDiff < 0.05, `no-prefix causal-custom vs causal: max LSE diff ${maxLseDiff} exceeds tolerance 0.05`);
  });

  it("causal-custom single token (first logit self-attention) matches built-in causal", () => {
    const pageSize = 16;
    const seqLen = 1;
    const kvSeqLen = seqLen;
    const numPages = Math.ceil(kvSeqLen / pageSize);
    const maxPages = numPages + 1;

    const suffixMask = buildCausalSuffixMask(seqLen);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, seqLen, seqLen);

    const qNopeF32 = randomData(seqLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(seqLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const ccResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const causalResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
    );

    const ccOutput = readOutput(ccResult.o, seqLen, N_HEADS, HEAD_DIM_CKV);
    const causalOutput = readOutput(causalResult.o, seqLen, N_HEADS, HEAD_DIM_CKV);

    const ccLse = readLse(ccResult.lse, seqLen, N_HEADS);
    const causalLse = readLse(causalResult.lse, seqLen, N_HEADS);

    let maxDiff = 0;
    for (let i = 0; i < ccOutput.length; i++) {
      const diff = Math.abs(ccOutput[i] - causalOutput[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    let maxLseDiff = 0;
    for (let i = 0; i < ccLse.length; i++) {
      const diff = Math.abs(ccLse[i] - causalLse[i]);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    assert.ok(maxDiff < 0.01, `single-token causal-custom vs causal: max output diff ${maxDiff} exceeds tolerance 0.01`);
    assert.ok(maxLseDiff < 0.01, `single-token causal-custom vs causal: max LSE diff ${maxLseDiff} exceeds tolerance 0.01`);
  });

  it("causal-custom no-prefix with non-causal plan matches causal plan", () => {
    const pageSize = 16;
    const seqLen = 7;
    const kvSeqLen = seqLen;
    const numPages = Math.ceil(kvSeqLen / pageSize);
    const maxPages = numPages + 1;

    const suffixMask = buildCausalSuffixMask(seqLen);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, seqLen, seqLen);

    const qNopeF32 = randomData(seqLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(seqLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const ccCausalPlanResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const ccNonCausalPlanResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      false,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const ccCausalOutput = readOutput(ccCausalPlanResult.o, seqLen, N_HEADS, HEAD_DIM_CKV);
    const ccNonCausalOutput = readOutput(ccNonCausalPlanResult.o, seqLen, N_HEADS, HEAD_DIM_CKV);

    const ccCausalLse = readLse(ccCausalPlanResult.lse, seqLen, N_HEADS);
    const ccNonCausalLse = readLse(ccNonCausalPlanResult.lse, seqLen, N_HEADS);

    let maxDiff = 0;
    for (let i = 0; i < ccCausalOutput.length; i++) {
      const diff = Math.abs(ccCausalOutput[i] - ccNonCausalOutput[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    let maxLseDiff = 0;
    for (let i = 0; i < ccCausalLse.length; i++) {
      const diff = Math.abs(ccCausalLse[i] - ccNonCausalLse[i]);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    assert.ok(maxDiff < 0.05, `causal-custom: causal plan vs non-causal plan output diff ${maxDiff} exceeds tolerance 0.05`);
    assert.ok(maxLseDiff < 0.05, `causal-custom: causal plan vs non-causal plan LSE diff ${maxLseDiff} exceeds tolerance 0.05`);
  });

  it("causal-custom no-prefix with non-causal plan matches built-in causal", () => {
    const pageSize = 16;
    const seqLen = 7;
    const kvSeqLen = seqLen;
    const numPages = Math.ceil(kvSeqLen / pageSize);
    const maxPages = numPages + 1;

    const suffixMask = buildCausalSuffixMask(seqLen);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, seqLen, seqLen);

    const qNopeF32 = randomData(seqLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(seqLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    // CausalCustom with causal=FALSE plan (matches the production bug config)
    const ccResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      false,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    // Built-in causal with causal=TRUE plan
    const causalResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
    );

    const ccOutput = readOutput(ccResult.o, seqLen, N_HEADS, HEAD_DIM_CKV);
    const causalOutput = readOutput(causalResult.o, seqLen, N_HEADS, HEAD_DIM_CKV);

    const ccLse = readLse(ccResult.lse, seqLen, N_HEADS);
    const causalLse = readLse(causalResult.lse, seqLen, N_HEADS);

    let maxDiff = 0;
    for (let i = 0; i < ccOutput.length; i++) {
      const diff = Math.abs(ccOutput[i] - causalOutput[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    let maxLseDiff = 0;
    for (let i = 0; i < ccLse.length; i++) {
      const diff = Math.abs(ccLse[i] - causalLse[i]);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    assert.ok(maxDiff < 0.05, `causal-custom non-causal plan vs built-in causal: max output diff ${maxDiff} exceeds tolerance 0.05`);
    assert.ok(maxLseDiff < 0.05, `causal-custom non-causal plan vs built-in causal: max LSE diff ${maxLseDiff} exceeds tolerance 0.05`);
  });

  it("8-head causal-equivalent CausalCustom mask at 8 tokens matches built-in causal", () => {
    const N_HEADS_8 = 8;
    const pageSize = 16;
    const seqLen = 8;
    const numPages = Math.ceil(seqLen / pageSize);
    const maxPages = numPages + 1;

    const causalMask = buildCausalMask(seqLen, seqLen);
    const { data: packedMask, indptr: maskIndptr } = packBitsLittleEndian(causalMask, seqLen, seqLen);

    const qNopeF32 = randomData(seqLen * N_HEADS_8 * HEAD_DIM_CKV);
    const qPeF32 = randomData(seqLen * N_HEADS_8 * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const ccResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, seqLen, 1, N_HEADS_8, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedMask, maskIndptr,
      MaskMode.CausalCustom,
    );

    const causalResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, seqLen, 1, N_HEADS_8, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
    );

    const ccOutput = readOutput(ccResult.o, seqLen, N_HEADS_8, HEAD_DIM_CKV);
    const causalOutput = readOutput(causalResult.o, seqLen, N_HEADS_8, HEAD_DIM_CKV);

    const ccLse = readLse(ccResult.lse, seqLen, N_HEADS_8);
    const causalLse = readLse(causalResult.lse, seqLen, N_HEADS_8);

    let maxDiff = 0;
    for (let i = 0; i < ccOutput.length; i++) {
      const diff = Math.abs(ccOutput[i] - causalOutput[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    let maxLseDiff = 0;
    for (let i = 0; i < ccLse.length; i++) {
      const diff = Math.abs(ccLse[i] - causalLse[i]);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    console.log(`  8-head 8-token CausalCustom vs causal: maxOutputDiff=${maxDiff.toFixed(6)} maxLseDiff=${maxLseDiff.toFixed(6)}`);
    assert.ok(maxDiff < 0.05, `8-head 8-token CausalCustom vs causal: max output diff ${maxDiff} exceeds tolerance 0.05`);
    assert.ok(maxLseDiff < 0.05, `8-head 8-token CausalCustom vs causal: max LSE diff ${maxLseDiff} exceeds tolerance 0.05`);
  });

  it("8-head tree mask at 8 tokens: each path matches standalone causal", () => {
    const N_HEADS_8 = 8;
    const pageSize = 16;
    const numTokens = 8;
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
    ];

    const suffixMask = buildCausalCustomSuffixMask(parents, numTokens);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, numTokens, numTokens);

    const qNopeF32 = randomData(numTokens * N_HEADS_8 * HEAD_DIM_CKV);
    const qPeF32 = randomData(numTokens * N_HEADS_8 * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const ccResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      numTokens, numTokens, 1, N_HEADS_8, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const ccOutput = readOutput(ccResult.o, numTokens, N_HEADS_8, HEAD_DIM_CKV);
    const ccLse = readLse(ccResult.lse, numTokens, N_HEADS_8);

    const qNopeRowBytes = N_HEADS_8 * HEAD_DIM_CKV;
    const qPeRowBytes = N_HEADS_8 * HEAD_DIM_KPE;

    const pathsToTest = [
      [0, 1, 3, 7],
      [0, 1, 4],
      [0, 2, 5],
      [0, 2, 6],
    ];

    for (const pathIndices of pathsToTest) {
      const pathLen = pathIndices.length;
      const pathNumPages = Math.ceil(pathLen / pageSize);
      const pathMaxPages = pathNumPages + 1;

      const qNopePath = new Float32Array(pathLen * qNopeRowBytes);
      const qPePath = new Float32Array(pathLen * qPeRowBytes);
      for (let i = 0; i < pathLen; i++) {
        const srcIdx = pathIndices[i];
        for (let j = 0; j < qNopeRowBytes; j++) qNopePath[i * qNopeRowBytes + j] = qNopeF32[srcIdx * qNopeRowBytes + j];
        for (let j = 0; j < qPeRowBytes; j++) qPePath[i * qPeRowBytes + j] = qPeF32[srcIdx * qPeRowBytes + j];
      }

      const ckvPath = new Float32Array(pathNumPages * pageSize * HEAD_DIM_CKV);
      const kpePath = new Float32Array(pathNumPages * pageSize * HEAD_DIM_KPE);
      for (let i = 0; i < pathLen; i++) {
        const srcIdx = pathIndices[i];
        for (let j = 0; j < HEAD_DIM_CKV; j++) ckvPath[i * HEAD_DIM_CKV + j] = ckvF32[srcIdx * HEAD_DIM_CKV + j];
        for (let j = 0; j < HEAD_DIM_KPE; j++) kpePath[i * HEAD_DIM_KPE + j] = kpeF32[srcIdx * HEAD_DIM_KPE + j];
      }

      const { o: pathO, lse: pathLse } = runMlaPrefill(
        glm, ws, qNopePath, qPePath, ckvPath, kpePath,
        pathLen, pathLen, 1, N_HEADS_8, HEAD_DIM_CKV, HEAD_DIM_KPE,
        pageSize, pathMaxPages,
        true,
      );

      const pathOutput = readOutput(pathO, pathLen, N_HEADS_8, HEAD_DIM_CKV);
      const pathLseArr = readLse(pathLse, pathLen, N_HEADS_8);

      for (let pos = 0; pos < pathLen; pos++) {
        const treeNode = pathIndices[pos];
        let maxDiff = 0;
        for (let h = 0; h < N_HEADS_8; h++) {
          for (let d = 0; d < HEAD_DIM_CKV; d++) {
            const treeIdx = h * numTokens * HEAD_DIM_CKV + treeNode * HEAD_DIM_CKV + d;
            const pathIdx = h * pathLen * HEAD_DIM_CKV + pos * HEAD_DIM_CKV + d;
            const diff = Math.abs(ccOutput[treeIdx] - pathOutput[pathIdx]);
            if (diff > maxDiff) maxDiff = diff;
          }
        }
        let maxLseDiff = 0;
        for (let h = 0; h < N_HEADS_8; h++) {
          const treeVal = ccLse[treeNode * N_HEADS_8 + h];
          const pathVal = pathLseArr[pos * N_HEADS_8 + h];
          const diff = Math.abs(treeVal - pathVal);
          if (diff > maxLseDiff) maxLseDiff = diff;
        }
        console.log(`  path [${pathIndices}] pos ${pos} (node ${treeNode}): maxOutputDiff=${maxDiff.toFixed(6)} maxLseDiff=${maxLseDiff.toFixed(6)}`);
        assert.ok(maxDiff < 0.05, `8-head tree path [${pathIndices}] pos ${pos} node ${treeNode}: output diff ${maxDiff} exceeds 0.05`);
        assert.ok(maxLseDiff < 0.05, `8-head tree path [${pathIndices}] pos ${pos} node ${treeNode}: LSE diff ${maxLseDiff} exceeds 0.05`);
      }
    }
  });

  it("8-head 128-dim tree mask at 8 tokens: each path matches standalone causal", () => {
    const HEAD_DIM_CKV_128 = 128;
    const HEAD_DIM_KPE_64 = 64;
    const N_HEADS_8 = 8;
    const SM_SCALE_128 = 1.0 / Math.sqrt(HEAD_DIM_CKV_128);
    const pageSize = 16;
    const numTokens = 8;
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
    ];

    const suffixMask = buildCausalCustomSuffixMask(parents, numTokens);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, numTokens, numTokens);

    const qNopeF32 = randomData(numTokens * N_HEADS_8 * HEAD_DIM_CKV_128);
    const qPeF32 = randomData(numTokens * N_HEADS_8 * HEAD_DIM_KPE_64);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV_128);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE_64);

    const ccResult = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      numTokens, numTokens, 1, N_HEADS_8, HEAD_DIM_CKV_128, HEAD_DIM_KPE_64,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const ccOutput = readOutput(ccResult.o, numTokens, N_HEADS_8, HEAD_DIM_CKV_128);
    const ccLse = readLse(ccResult.lse, numTokens, N_HEADS_8);

    const qNopeRowBytes = N_HEADS_8 * HEAD_DIM_CKV_128;
    const qPeRowBytes = N_HEADS_8 * HEAD_DIM_KPE_64;

    const pathsToTest = [
      [0, 1, 3, 7],
      [0, 1, 4],
      [0, 2, 5],
      [0, 2, 6],
    ];

    for (const pathIndices of pathsToTest) {
      const pathLen = pathIndices.length;
      const pathNumPages = Math.ceil(pathLen / pageSize);
      const pathMaxPages = pathNumPages + 1;

      const qNopePath = new Float32Array(pathLen * qNopeRowBytes);
      const qPePath = new Float32Array(pathLen * qPeRowBytes);
      for (let i = 0; i < pathLen; i++) {
        const srcIdx = pathIndices[i];
        for (let j = 0; j < qNopeRowBytes; j++) qNopePath[i * qNopeRowBytes + j] = qNopeF32[srcIdx * qNopeRowBytes + j];
        for (let j = 0; j < qPeRowBytes; j++) qPePath[i * qPeRowBytes + j] = qPeF32[srcIdx * qPeRowBytes + j];
      }

      const ckvPath = new Float32Array(pathNumPages * pageSize * HEAD_DIM_CKV_128);
      const kpePath = new Float32Array(pathNumPages * pageSize * HEAD_DIM_KPE_64);
      for (let i = 0; i < pathLen; i++) {
        const srcIdx = pathIndices[i];
        for (let j = 0; j < HEAD_DIM_CKV_128; j++) ckvPath[i * HEAD_DIM_CKV_128 + j] = ckvF32[srcIdx * HEAD_DIM_CKV_128 + j];
        for (let j = 0; j < HEAD_DIM_KPE_64; j++) kpePath[i * HEAD_DIM_KPE_64 + j] = kpeF32[srcIdx * HEAD_DIM_KPE_64 + j];
      }

      const { o: pathO, lse: pathLse } = runMlaPrefill(
        glm, ws, qNopePath, qPePath, ckvPath, kpePath,
        pathLen, pathLen, 1, N_HEADS_8, HEAD_DIM_CKV_128, HEAD_DIM_KPE_64,
        pageSize, pathMaxPages,
        true,
      );

      const pathOutput = readOutput(pathO, pathLen, N_HEADS_8, HEAD_DIM_CKV_128);
      const pathLseArr = readLse(pathLse, pathLen, N_HEADS_8);

      for (let pos = 0; pos < pathLen; pos++) {
        const treeNode = pathIndices[pos];
        let maxDiff = 0;
        for (let h = 0; h < N_HEADS_8; h++) {
          for (let d = 0; d < HEAD_DIM_CKV_128; d++) {
            const treeIdx = h * numTokens * HEAD_DIM_CKV_128 + treeNode * HEAD_DIM_CKV_128 + d;
            const pathIdx = h * pathLen * HEAD_DIM_CKV_128 + pos * HEAD_DIM_CKV_128 + d;
            const diff = Math.abs(ccOutput[treeIdx] - pathOutput[pathIdx]);
            if (diff > maxDiff) maxDiff = diff;
          }
        }
        let maxLseDiff = 0;
        for (let h = 0; h < N_HEADS_8; h++) {
          const treeVal = ccLse[treeNode * N_HEADS_8 + h];
          const pathVal = pathLseArr[pos * N_HEADS_8 + h];
          const diff = Math.abs(treeVal - pathVal);
          if (diff > maxLseDiff) maxLseDiff = diff;
        }
        console.log(`  128-dim path [${pathIndices}] pos ${pos} (node ${treeNode}): maxOutputDiff=${maxDiff.toFixed(6)} maxLseDiff=${maxLseDiff.toFixed(6)}`);
        assert.ok(maxDiff < 0.05, `128-dim tree path [${pathIndices}] pos ${pos} node ${treeNode}: output diff ${maxDiff} exceeds 0.05`);
        assert.ok(maxLseDiff < 0.05, `128-dim tree path [${pathIndices}] pos ${pos} node ${treeNode}: LSE diff ${maxLseDiff} exceeds 0.05`);
      }
    }
  });
});
