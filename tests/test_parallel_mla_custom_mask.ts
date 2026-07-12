import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { MaskMode } from "../src/device_ops";
import { WorkspaceBase } from "../src/workspace";
import { Tensor } from "../src/tensor";
import type { ExecutionState } from "../src/execution-workspace";
import { TensorParallelism } from "../src/device_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";

const HEAD_DIM_CKV = 512;
const HEAD_DIM_KPE = 64;
const N_HEADS = 4;
const SM_SCALE = 1.0 / Math.sqrt(HEAD_DIM_CKV);

function i32Buf(data: Int32Array): Buffer {
  const buf = Buffer.alloc(data.length * 4);
  for (let i = 0; i < data.length; i++) buf.writeInt32LE(data[i], i * 4);
  return buf;
}

function allocBf16(ws: WorkspaceBase, shape: number[], parallelism?: TensorParallelism): Tensor {
  return ws.alloc(shape, "BF16", undefined, parallelism);
}

function allocI32(ws: WorkspaceBase, shape: number[], parallelism?: TensorParallelism): Tensor {
  return ws.alloc(shape, "I32", undefined, parallelism);
}

function allocPinnedI32(ws: WorkspaceBase, shape: number[], parallelism?: TensorParallelism): Tensor {
  return ws.allocPinned(shape, "I32", undefined, parallelism);
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

function runMlaPrefillParallel(
  po: ParallelOps,
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

  const qNope = allocBf16(ws, [1, totalQTokens, nHeads * headDimCkv], TensorParallelism.Replicated) as ParallelTensor;
  const qPe = allocBf16(ws, [1, totalQTokens, nHeads * headDimKpe], TensorParallelism.Replicated) as ParallelTensor;
  qNope.h2d(f32ToBf16Bytes(qNopeF32));
  qPe.h2d(f32ToBf16Bytes(qPeF32));

  const ckv = allocBf16(ws, [ckvF32.length], TensorParallelism.Replicated);
  const kpe = allocBf16(ws, [kpeF32.length], TensorParallelism.Replicated);
  ckv.h2d(f32ToBf16Bytes(ckvF32));
  kpe.h2d(f32ToBf16Bytes(kpeF32));

  const indices = allocI32(ws, [maxPages], TensorParallelism.Replicated);
  const indicesData = new Int32Array(maxPages);
  for (let i = 0; i < numPages; i++) indicesData[i] = i;
  indices.h2d(i32Buf(indicesData));

  const indptrArr = new Int32Array(batchSize + 1);
  indptrArr[0] = 0;
  indptrArr[1] = numPages;
  const indptrH = allocPinnedI32(ws, [batchSize + 1], TensorParallelism.Replicated);
  indptrH.h2d(i32Buf(indptrArr));

  const lastPageLenH = allocPinnedI32(ws, [batchSize], TensorParallelism.Replicated);
  lastPageLenH.h2d(i32Buf(new Int32Array([lastPageLen])));

  const floatWs = allocBf16(ws, [128 * 1024 * 1024 / 2], TensorParallelism.Replicated);
  const intWs = allocI32(ws, [8 * 1024 * 1024 / 4], TensorParallelism.Replicated);
  const pinnedIntWs = allocPinnedI32(ws, [8 * 1024 * 1024 / 4], TensorParallelism.Replicated);
  const planInfo = allocPinnedI32(ws, [19], TensorParallelism.Replicated);

  const kvLenH = allocPinnedI32(ws, [batchSize], TensorParallelism.Replicated);
  kvLenH.h2d(i32Buf(new Int32Array([kvSeqLen])));

  const qoIndptrH = allocPinnedI32(ws, [batchSize + 1], TensorParallelism.Replicated);
  qoIndptrH.h2d(i32Buf(new Int32Array([0, totalQTokens])));

  po.mlaPrefillPlan(
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
    maskMode = maskModeOverride ?? MaskMode.Custom;
    customMask = ws.alloc([customMaskData.length], "U8", undefined, TensorParallelism.Replicated);
    customMask.h2d(Buffer.from(customMaskData));
    maskIndptr = ws.alloc([maskIndptrData.length], "I32", undefined, TensorParallelism.Replicated);
    maskIndptr.h2d(i32Buf(maskIndptrData));
  } else {
    maskMode = causal ? MaskMode.Causal : MaskMode.None;
  }

  const execState = { batchSize, totalTokens: totalQTokens, cache: { getPagedKV: () => ({ pageSize }) } } as ExecutionState;
  const result = po.mlaPrefillRun(
    execState,
    qNope, qPe, ckv, kpe, indices,
    floatWs, intWs, planInfo,
    nHeads, pageSize, maskMode, SM_SCALE,
    qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH,
    ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN,
    oStrideN, oStrideH,
    headDimCkv, headDimKpe,
    undefined, undefined,
    customMask, maskIndptr,
  );
  po.synchronize();

  return result;
}

function runMlaPrefillRef(
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

  const qNope = ws.alloc([1, totalQTokens, nHeads * headDimCkv], "BF16");
  const qPe = ws.alloc([1, totalQTokens, nHeads * headDimKpe], "BF16");
  qNope.h2d(f32ToBf16Bytes(qNopeF32));
  qPe.h2d(f32ToBf16Bytes(qPeF32));

  const ckv = ws.alloc([ckvF32.length], "BF16");
  const kpe = ws.alloc([kpeF32.length], "BF16");
  ckv.h2d(f32ToBf16Bytes(ckvF32));
  kpe.h2d(f32ToBf16Bytes(kpeF32));

  const indices = ws.alloc([maxPages], "I32");
  const indicesData = new Int32Array(maxPages);
  for (let i = 0; i < numPages; i++) indicesData[i] = i;
  indices.h2d(i32Buf(indicesData));

  const indptrArr = new Int32Array(batchSize + 1);
  indptrArr[0] = 0;
  indptrArr[1] = numPages;
  const indptrH = ws.allocPinned([batchSize + 1], "I32");
  indptrH.h2d(i32Buf(indptrArr));

  const lastPageLenH = ws.allocPinned([batchSize], "I32");
  lastPageLenH.h2d(i32Buf(new Int32Array([lastPageLen])));

  const floatWs = ws.alloc([128 * 1024 * 1024 / 2], "BF16");
  const intWs = ws.alloc([8 * 1024 * 1024 / 4], "I32");
  const pinnedIntWs = ws.allocPinned([8 * 1024 * 1024 / 4], "I32");
  const planInfo = ws.allocPinned([19], "I32");

  const kvLenH = ws.allocPinned([batchSize], "I32");
  kvLenH.h2d(i32Buf(new Int32Array([kvSeqLen])));

  const qoIndptrH = ws.allocPinned([batchSize + 1], "I32");
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

describe("ParallelOps MLA custom mask", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let ref: GlmOps;
  let po: ParallelOps;
  let ws: WorkspaceBase;
  let refWs: WorkspaceBase;

  before(() => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    ref = new GlmOps(2);
    po = new ParallelOps([glm0, glm1]);
    ws = new WorkspaceBase(po);
    refWs = new WorkspaceBase(ref);
  });

  after(() => {
    refWs.free();
    ref.free();
    ws.free();
    po.free();
    glm0.free();
    glm1.free();
  });

  it("tree mask: no crash and valid output", () => {
    const pageSize = 16;
    const numTokens = 15;
    const numPages = Math.ceil(numTokens / pageSize);
    const maxPages = numPages + 1;
    const worldSize = 2;
    const effectiveNHeads = N_HEADS / worldSize;

    const parents: number[][] = [
      [], [0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6],
    ];

    const mask = buildTreeMask(parents, numTokens, numTokens);
    const { data: packedMask, indptr: maskIndptr } = packBitsLittleEndian(mask, numTokens, numTokens);

    const qNopeF32 = randomData(numTokens * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(numTokens * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const { o, lse } = runMlaPrefillParallel(
      po, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      numTokens, numTokens, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      false,
      packedMask, maskIndptr,
    );

    const output = readOutput(o, numTokens, effectiveNHeads, HEAD_DIM_CKV);
    const lseData = readLse(lse, numTokens, effectiveNHeads);

    assert.equal(output.length, numTokens * effectiveNHeads * HEAD_DIM_CKV, "output length");
    assert.equal(lseData.length, numTokens * effectiveNHeads, "lse length");

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
    const worldSize = 2;
    const effectiveNHeads = N_HEADS / worldSize;

    const parents: number[][] = [
      [], [0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6],
    ];

    const mask = buildTreeMask(parents, numTokens, numTokens);
    const { data: packedMask, indptr: maskIndptr } = packBitsLittleEndian(mask, numTokens, numTokens);

    const qNopeF32 = randomData(numTokens * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(numTokens * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const { o: treeO, lse: treeLse } = runMlaPrefillParallel(
      po, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      numTokens, numTokens, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      false,
      packedMask, maskIndptr,
    );

    const treeOutput = readOutput(treeO, numTokens, effectiveNHeads, HEAD_DIM_CKV);
    const treeLseArr = readLse(treeLse, numTokens, effectiveNHeads);

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

    const { o: pathO, lse: pathLse } = runMlaPrefillParallel(
      po, ws, qNopePath, qPePath, ckvPath, kpePath,
      pathLen, pathLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, pathMaxPages,
      true,
    );

    const pathOutput = readOutput(pathO, pathLen, effectiveNHeads, HEAD_DIM_CKV);
    const pathLseArr = readLse(pathLse, pathLen, effectiveNHeads);

    let maxDiff = 0;
    for (let h = 0; h < effectiveNHeads; h++) {
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
    for (let h = 0; h < effectiveNHeads; h++) {
      const treeVal = treeLseArr[7 * effectiveNHeads + h];
      const pathVal = pathLseArr[3 * effectiveNHeads + h];
      const diff = Math.abs(treeVal - pathVal);
      if (diff > maxLseDiff) maxLseDiff = diff;
    }

    assert.ok(maxDiff < 0.1, `tree path output max diff ${maxDiff} exceeds tolerance 0.1`);
    assert.ok(maxLseDiff < 0.15, `tree path LSE max diff ${maxLseDiff} exceeds tolerance 0.15`);
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

    const customResult = runMlaPrefillParallel(
      po, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, seqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      false,
      packedMask, maskIndptr,
    );

    const builtinResult = runMlaPrefillParallel(
      po, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, seqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
    );

    const worldSize = 2;
    const effectiveNHeads = N_HEADS / worldSize;

    const customOutput = readOutput(customResult.o, seqLen, effectiveNHeads, HEAD_DIM_CKV);
    const builtinOutput = readOutput(builtinResult.o, seqLen, effectiveNHeads, HEAD_DIM_CKV);

    const customLse = readLse(customResult.lse, seqLen, effectiveNHeads);
    const builtinLse = readLse(builtinResult.lse, seqLen, effectiveNHeads);

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
    const worldSize = 2;
    const effectiveNHeads = N_HEADS / worldSize;

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

    const ccResult = runMlaPrefillParallel(
      po, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const customResult = runMlaPrefillParallel(
      po, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      false,
      packedFullMask, fullMaskIndptr,
    );

    const ccOutput = readOutput(ccResult.o, qoLen, effectiveNHeads, HEAD_DIM_CKV);
    const customOutput = readOutput(customResult.o, qoLen, effectiveNHeads, HEAD_DIM_CKV);

    const ccLse = readLse(ccResult.lse, qoLen, effectiveNHeads);
    const customLse = readLse(customResult.lse, qoLen, effectiveNHeads);

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

    assert.ok(maxDiff < 0.1, `causal-custom vs full custom: max output diff ${maxDiff} exceeds tolerance 0.1`);
    assert.ok(maxLseDiff < 0.15, `causal-custom vs full custom: max LSE diff ${maxLseDiff} exceeds tolerance 0.15`);
  });

  it("causal-custom with causal suffix mask matches built-in causal", () => {
    const pageSize = 16;
    const prefixLen = 32;
    const numPrefillTokens = 7;
    const qoLen = numPrefillTokens;
    const kvSeqLen = prefixLen + numPrefillTokens;
    const numPages = Math.ceil(kvSeqLen / pageSize);
    const maxPages = numPages + 1;
    const worldSize = 2;
    const effectiveNHeads = N_HEADS / worldSize;

    const suffixMask = buildCausalSuffixMask(qoLen);
    const { data: packedSuffixMask, indptr: suffixMaskIndptr } = packBitsLittleEndian(suffixMask, qoLen, qoLen);

    const qNopeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(qoLen * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

    const ccResult = runMlaPrefillParallel(
      po, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
      packedSuffixMask, suffixMaskIndptr,
      MaskMode.CausalCustom,
    );

    const causalResult = runMlaPrefillParallel(
      po, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
      qoLen, kvSeqLen, 1, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      true,
    );

    const ccOutput = readOutput(ccResult.o, qoLen, effectiveNHeads, HEAD_DIM_CKV);
    const causalOutput = readOutput(causalResult.o, qoLen, effectiveNHeads, HEAD_DIM_CKV);

    const ccLse = readLse(ccResult.lse, qoLen, effectiveNHeads);
    const causalLse = readLse(causalResult.lse, qoLen, effectiveNHeads);

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
});
