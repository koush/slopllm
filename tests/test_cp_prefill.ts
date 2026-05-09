import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { Tensor } from "../src/tensor";

const HEAD_DIM_CKV = 512;
const HEAD_DIM_KPE = 64;
const N_HEADS = 4;
const V_HEAD_DIM = 128;
const SM_SCALE = 1.0 / Math.sqrt(HEAD_DIM_CKV);

function i32Buf(data: Int32Array): Buffer {
  const buf = Buffer.alloc(data.length * 4);
  for (let i = 0; i < data.length; i++) buf.writeInt32LE(data[i], i * 4);
  return buf;
}

function allocBf16(ws: WorkspaceBase, shape: number[]): Tensor {
  return ws.alloc(shape, "BF16");
}

function allocF32(ws: WorkspaceBase, shape: number[]): Tensor {
  return ws.alloc(shape, "F32");
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

function readLse(lseTensor: Tensor, totalTokens: number, nHeads: number): Float32Array {
  const buf = Buffer.alloc(totalTokens * nHeads * 4);
  lseTensor.d2h(buf);
  const f32 = new Float32Array(totalTokens * nHeads);
  for (let i = 0; i < totalTokens * nHeads; i++) f32[i] = buf.readFloatLE(i * 4);
  return f32;
}

interface PrefillResult {
  vOut: Tensor;
  lse: Tensor;
}

function runMlaPrefill(
  glm: GlmOps,
  ws: WorkspaceBase,
  qNopeF32: Float32Array,
  qPeF32: Float32Array,
  ckvF32: Float32Array,
  kpeF32: Float32Array,
  qSeqLen: number,
  kvSeqLen: number,
  batchSize: number,
  nHeads: number,
  headDimCkv: number,
  headDimKpe: number,
  pageSize: number,
  maxPages: number,
  ckvStridePage: number,
  kpeStridePage: number,
  cpWorldSize: number,
  cpRank: number,
  cpKvLensArr: Int32Array | null,
): PrefillResult {
  const totalQTokens = batchSize * qSeqLen;
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
  const indptrD = allocI32(ws, [batchSize + 1]);
  indptrD.h2d(i32Buf(indptrArr));

  const lastPageLenBuf = allocI32(ws, [batchSize]);
  lastPageLenBuf.h2d(i32Buf(new Int32Array([lastPageLen])));

  const floatWs = allocBf16(ws, [128 * 1024 * 1024 / 2]);
  const intWs = allocI32(ws, [8 * 1024 * 1024 / 4]);
  const pinnedIntWs = allocPinnedI32(ws, [8 * 1024 * 1024 / 4]);
  const planInfo = allocPinnedI32(ws, [18]);

  const kvLenH = allocPinnedI32(ws, [batchSize]);
  kvLenH.h2d(i32Buf(new Int32Array([kvSeqLen])));

  const qoIndptrH = allocPinnedI32(ws, [batchSize + 1]);
  qoIndptrH.h2d(i32Buf(new Int32Array([0, totalQTokens])));

  glm.mlaPrefillPlan(
    floatWs, 128 * 1024 * 1024,
    intWs, pinnedIntWs, 8 * 1024 * 1024,
    planInfo,
    qoIndptrH, indptrH, kvLenH,
    batchSize, nHeads, headDimCkv, true,
    cpWorldSize, cpRank,
  );

  const vOut = allocBf16(ws, [1, nHeads, totalQTokens, headDimCkv]);
  const lse = allocF32(ws, [totalQTokens, nHeads]);

  const ckvStrideN = headDimCkv;
  const kpeStrideN = headDimKpe;
  const qNopeStrideN = nHeads * headDimCkv;
  const qNopeStrideH = headDimCkv;
  const qPeStrideN = nHeads * headDimKpe;
  const qPeStrideH = headDimKpe;
  const oStrideN = headDimCkv;
  const oStrideH = totalQTokens * headDimCkv;

  let cpKvLenTensor: Tensor | null = null;
  if (cpWorldSize > 0 && cpKvLensArr) {
    const maxWorks = 16384;
    cpKvLenTensor = allocI32(ws, [maxWorks]);
    const cpKvLenData = new Int32Array(maxWorks);
    for (let i = 0; i < maxWorks; i++) cpKvLenData[i] = cpKvLensArr[i % batchSize];
    cpKvLenTensor.h2d(i32Buf(cpKvLenData));
  }

  glm.mlaPrefillRun(
    qNope, qPe, ckv, kpe, indices, vOut,
    floatWs, intWs, planInfo,
    nHeads, pageSize, 1, SM_SCALE,
    qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH,
    ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN,
    oStrideN, oStrideH,
    headDimCkv, headDimKpe,
    lse,
    cpWorldSize, cpRank, cpKvLenTensor,
  );
  glm.synchronize();

  return { vOut, lse };
}

function shardKV(
  ckvF32: Float32Array,
  kpeF32: Float32Array,
  seqLen: number,
  pageSize: number,
  worldSize: number,
  rank: number,
  headDimCkv: number,
  headDimKpe: number,
): { ckvShard: Float32Array; kpeShard: Float32Array; shardLen: number; maxShardPages: number } {
  const vPS = pageSize / worldSize;
  const shardLen = rank < (seqLen % worldSize) ? Math.ceil(seqLen / worldSize) : Math.floor(seqLen / worldSize);
  const shardPages = Math.ceil(shardLen / vPS);
  const maxPages = Math.ceil(seqLen / pageSize);
  const maxShardPages = Math.ceil(Math.ceil(seqLen / worldSize) / vPS);

  const ckvShard = new Float32Array(maxShardPages * vPS * headDimCkv);
  const kpeShard = new Float32Array(maxShardPages * vPS * headDimKpe);

  for (let localPos = 0; localPos < shardLen; localPos++) {
    const globalPos = localPos * worldSize + rank;
    const localPage = Math.floor(localPos / vPS);
    const localOffset = localPos % vPS;

    for (let d = 0; d < headDimCkv; d++) {
      ckvShard[(localPage * vPS + localOffset) * headDimCkv + d] = ckvF32[globalPos * headDimCkv + d];
    }
    for (let d = 0; d < headDimKpe; d++) {
      kpeShard[(localPage * vPS + localOffset) * headDimKpe + d] = kpeF32[globalPos * headDimKpe + d];
    }
  }

  return { ckvShard, kpeShard, shardLen, maxShardPages };
}

function runCpPrefillTest(
  glm: GlmOps,
  ws: WorkspaceBase,
  seqLen: number,
  batchSize: number,
  pageSize: number,
  worldSize: number,
): void {
  const totalTokens = batchSize * seqLen;
  const maxPages = Math.ceil(seqLen / pageSize) + 1;
  const vPS = pageSize / worldSize;

  const qNopeF32 = randomData(totalTokens * N_HEADS * HEAD_DIM_CKV);
  const qPeF32 = randomData(totalTokens * N_HEADS * HEAD_DIM_KPE);

  const numPages = Math.ceil(seqLen / pageSize);
  const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
  const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

  const vProjF32 = randomData(N_HEADS * V_HEAD_DIM * HEAD_DIM_CKV);
  const vProj = allocBf16(ws, [N_HEADS * V_HEAD_DIM, HEAD_DIM_CKV]);
  vProj.h2d(f32ToBf16Bytes(vProjF32));

  const baseline = runMlaPrefill(
    glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
    seqLen, seqLen, batchSize, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
    pageSize, maxPages,
    pageSize * HEAD_DIM_CKV, pageSize * HEAD_DIM_KPE,
    0, 0, null,
  );

  const baselineVExpanded = baseline.vOut.mlaVExpand(vProj, HEAD_DIM_CKV, V_HEAD_DIM, N_HEADS, seqLen, batchSize);
  const baselineVExpandedBuf = Buffer.alloc(totalTokens * N_HEADS * V_HEAD_DIM * 2);
  baselineVExpanded.d2h(baselineVExpandedBuf);
  const baselineVExpandedF32 = bf16BytesToF32(baselineVExpandedBuf);

  const shardVPtrs: number[] = [];
  const shardLsePtrs: number[] = [];
  const cpKvLens = new Int32Array(batchSize).fill(seqLen);

  for (let rank = 0; rank < worldSize; rank++) {
    const { ckvShard, kpeShard, shardLen, maxShardPages } = shardKV(
      ckvF32, kpeF32, seqLen, pageSize, worldSize, rank, HEAD_DIM_CKV, HEAD_DIM_KPE,
    );

    const ckvStridePageShard = vPS * HEAD_DIM_CKV;
    const kpeStridePageShard = vPS * HEAD_DIM_KPE;

    const result = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvShard, kpeShard,
      seqLen, shardLen, batchSize, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      vPS, maxShardPages,
      ckvStridePageShard, kpeStridePageShard,
      worldSize, rank, cpKvLens,
    );

    const shardVExpanded = result.vOut.mlaVExpand(vProj, HEAD_DIM_CKV, V_HEAD_DIM, N_HEADS, seqLen, batchSize);

    shardVPtrs.push(shardVExpanded.data);
    shardLsePtrs.push(result.lse.data);
  }

  const mergedVOut = allocBf16(ws, [totalTokens, N_HEADS * V_HEAD_DIM]);
  glm.contextParallelMerge(
    shardVPtrs, shardLsePtrs, worldSize,
    mergedVOut, null,
    totalTokens, N_HEADS, V_HEAD_DIM,
  );
  glm.synchronize();

  const mergedBuf = Buffer.alloc(totalTokens * N_HEADS * V_HEAD_DIM * 2);
  mergedVOut.d2h(mergedBuf);
  const mergedF32 = bf16BytesToF32(mergedBuf);

  let maxRelErr = 0;
  let maxAbsErr = 0;
  let maxRelErrForSignificant = 0;
  let errorCount = 0;
  const totalElems = totalTokens * N_HEADS * V_HEAD_DIM;
  const SIGNIFICANCE_THRESHOLD = 0.05;
  const ABS_TOL = 0.03;
  const REL_TOL = 0.10;

  for (let i = 0; i < totalElems; i++) {
    const merged = mergedF32[i];
    const expected = baselineVExpandedF32[i];
    const absErr = Math.abs(merged - expected);
    const relErr = absErr / Math.max(Math.abs(expected), 1e-6);
    if (absErr > ABS_TOL + REL_TOL * Math.abs(expected)) {
      errorCount++;
    }
    maxRelErr = Math.max(maxRelErr, relErr);
    maxAbsErr = Math.max(maxAbsErr, absErr);
    if (Math.abs(expected) > SIGNIFICANCE_THRESHOLD) {
      maxRelErrForSignificant = Math.max(maxRelErrForSignificant, relErr);
    }
  }

  console.log(`CP prefill (seqLen=${seqLen}, batch=${batchSize}, pageSize=${pageSize}, worldSize=${worldSize}): maxAbsErr=${maxAbsErr.toFixed(6)} maxRelErrForSignificant=${maxRelErrForSignificant.toFixed(6)} errors=${errorCount}/${totalElems}`);
  assert.ok(errorCount === 0, `${errorCount}/${totalElems} elements exceed tolerance (atol=${ABS_TOL}, rtol=${REL_TOL})`);
}

describe("CP MLA Prefill", () => {
  let glm: GlmOps;
  let ws: WorkspaceBase;

  before(() => {
    glm = new GlmOps(0);
    ws = new WorkspaceBase(glm);
  });

  after(() => {
    ws.free();
    glm.free();
  });

  it("2-shard CP prefill matches baseline (seqLen=32, batch=1, pageSize=16)", () => {
    runCpPrefillTest(glm, ws, 32, 1, 16, 2);
  });

  it("2-shard CP prefill matches baseline (seqLen=33, batch=1, pageSize=16) — odd seqLen", () => {
    runCpPrefillTest(glm, ws, 33, 1, 16, 2);
  });

  it("2-shard CP prefill matches baseline (seqLen=64, batch=1, pageSize=16) — longer seqLen", () => {
    runCpPrefillTest(glm, ws, 64, 1, 16, 2);
  });

  it("4-shard CP prefill matches baseline (seqLen=32, batch=1, pageSize=16)", () => {
    runCpPrefillTest(glm, ws, 32, 1, 16, 4);
  });

  it("2-shard CP prefill matches baseline (seqLen=64, batch=1, pageSize=16) — 4-shard", () => {
    runCpPrefillTest(glm, ws, 64, 1, 16, 4);
  });

  it("2-shard CP prefill matches baseline (seqLen=128, batch=1, pageSize=16)", () => {
    runCpPrefillTest(glm, ws, 128, 1, 16, 2);
  });
});
