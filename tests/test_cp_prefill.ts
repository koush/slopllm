import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { TensorParallelism } from "../src/device_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { PagedKVCache } from "../src/paged_kv";
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
  globalKvLen: number,
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
  const planInfo = allocPinnedI32(ws, [19]);

  const kvLenH = allocPinnedI32(ws, [batchSize]);
  kvLenH.h2d(i32Buf(new Int32Array([globalKvLen])));

  const lastPageLenH = allocPinnedI32(ws, [batchSize]);
  lastPageLenH.h2d(i32Buf(new Int32Array([lastPageLen])));

  const qoIndptrH = allocPinnedI32(ws, [batchSize + 1]);
  qoIndptrH.h2d(i32Buf(new Int32Array([0, totalQTokens])));

  glm.mlaPrefillPlan(
    floatWs, 128 * 1024 * 1024,
    intWs, pinnedIntWs, 8 * 1024 * 1024,
    planInfo,
    qoIndptrH, indptrH, kvLenH, lastPageLenH,
    batchSize, nHeads, headDimCkv, true,
    pageSize, [kvSeqLen],
    true, cpWorldSize, cpRank,
  );

  const ckvStrideN = headDimCkv;
  const kpeStrideN = headDimKpe;
  const qNopeStrideN = nHeads * headDimCkv;
  const qNopeStrideH = headDimCkv;
  const qPeStrideN = nHeads * headDimKpe;
  const qPeStrideH = headDimKpe;
  const oStrideN = headDimCkv;
  const oStrideH = totalQTokens * headDimCkv;

  const { o: vOut, lse } = glm.mlaPrefillRun(
    qNope, qPe, ckv, kpe, indices,
    floatWs, intWs, planInfo,
    nHeads, pageSize, 1, SM_SCALE,
    qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH,
    ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN,
    oStrideN, oStrideH,
    headDimCkv, headDimKpe,
    true, cpWorldSize, cpRank,
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
    seqLen, seqLen, seqLen, batchSize, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
    pageSize, maxPages,
    pageSize * HEAD_DIM_CKV, pageSize * HEAD_DIM_KPE,
    0, 0,
  );

  const baselineVExpanded = baseline.vOut.mlaVExpand(vProj, HEAD_DIM_CKV, V_HEAD_DIM, N_HEADS, seqLen, batchSize);
  const baselineVExpandedBuf = Buffer.alloc(totalTokens * N_HEADS * V_HEAD_DIM * 2);
  baselineVExpanded.d2h(baselineVExpandedBuf);
  const baselineVExpandedF32 = bf16BytesToF32(baselineVExpandedBuf);

  const shardVPtrs: number[] = [];
  const shardLsePtrs: number[] = [];

  for (let rank = 0; rank < worldSize; rank++) {
    const { ckvShard, kpeShard, shardLen, maxShardPages } = shardKV(
      ckvF32, kpeF32, seqLen, pageSize, worldSize, rank, HEAD_DIM_CKV, HEAD_DIM_KPE,
    );

    const ckvStridePageShard = vPS * HEAD_DIM_CKV;
    const kpeStridePageShard = vPS * HEAD_DIM_KPE;

    const result = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvShard, kpeShard,
      seqLen, shardLen, seqLen, batchSize, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      vPS, maxShardPages,
      ckvStridePageShard, kpeStridePageShard,
      worldSize, rank,
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

function runCpAppendPrefillTest(
  glm: GlmOps,
  ws: WorkspaceBase,
  prefixLen: number,
  newLen: number,
  batchSize: number,
  pageSize: number,
  worldSize: number,
): void {
  const totalKvLen = prefixLen + newLen;
  const totalQTokens = batchSize * newLen;
  const numPages = Math.ceil(totalKvLen / pageSize);
  const maxPages = numPages + 1;
  const vPS = pageSize / worldSize;

  const qNopeF32 = randomData(totalQTokens * N_HEADS * HEAD_DIM_CKV);
  const qPeF32 = randomData(totalQTokens * N_HEADS * HEAD_DIM_KPE);

  const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
  const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);

  const vProjF32 = randomData(N_HEADS * V_HEAD_DIM * HEAD_DIM_CKV);
  const vProj = allocBf16(ws, [N_HEADS * V_HEAD_DIM, HEAD_DIM_CKV]);
  vProj.h2d(f32ToBf16Bytes(vProjF32));

  const baseline = runMlaPrefill(
    glm, ws, qNopeF32, qPeF32, ckvF32, kpeF32,
    newLen, totalKvLen, totalKvLen, batchSize, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
    pageSize, maxPages,
    pageSize * HEAD_DIM_CKV, pageSize * HEAD_DIM_KPE,
    0, 0,
  );

  const baselineVExpanded = baseline.vOut.mlaVExpand(vProj, HEAD_DIM_CKV, V_HEAD_DIM, N_HEADS, newLen, batchSize);
  const baselineVExpandedBuf = Buffer.alloc(totalQTokens * N_HEADS * V_HEAD_DIM * 2);
  baselineVExpanded.d2h(baselineVExpandedBuf);
  const baselineVExpandedF32 = bf16BytesToF32(baselineVExpandedBuf);

  const shardVPtrs: number[] = [];
  const shardLsePtrs: number[] = [];

  for (let rank = 0; rank < worldSize; rank++) {
    const { ckvShard, kpeShard, shardLen, maxShardPages } = shardKV(
      ckvF32, kpeF32, totalKvLen, pageSize, worldSize, rank, HEAD_DIM_CKV, HEAD_DIM_KPE,
    );

    const ckvStridePageShard = vPS * HEAD_DIM_CKV;
    const kpeStridePageShard = vPS * HEAD_DIM_KPE;

    const result = runMlaPrefill(
      glm, ws, qNopeF32, qPeF32, ckvShard, kpeShard,
      newLen, shardLen, totalKvLen, batchSize, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      vPS, maxShardPages,
      ckvStridePageShard, kpeStridePageShard,
      worldSize, rank,
    );

    const shardVExpanded = result.vOut.mlaVExpand(vProj, HEAD_DIM_CKV, V_HEAD_DIM, N_HEADS, newLen, batchSize);

    shardVPtrs.push(shardVExpanded.data);
    shardLsePtrs.push(result.lse.data);
  }

  const mergedVOut = allocBf16(ws, [totalQTokens, N_HEADS * V_HEAD_DIM]);
  glm.contextParallelMerge(
    shardVPtrs, shardLsePtrs, worldSize,
    mergedVOut, null,
    totalQTokens, N_HEADS, V_HEAD_DIM,
  );
  glm.synchronize();

  const mergedBuf = Buffer.alloc(totalQTokens * N_HEADS * V_HEAD_DIM * 2);
  mergedVOut.d2h(mergedBuf);
  const mergedF32 = bf16BytesToF32(mergedBuf);

  let maxRelErr = 0;
  let maxAbsErr = 0;
  let maxRelErrForSignificant = 0;
  let errorCount = 0;
  const totalElems = totalQTokens * N_HEADS * V_HEAD_DIM;
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

  console.log(`CP append prefill (prefix=${prefixLen}, new=${newLen}, batch=${batchSize}, pageSize=${pageSize}, worldSize=${worldSize}): maxAbsErr=${maxAbsErr.toFixed(6)} maxRelErrForSignificant=${maxRelErrForSignificant.toFixed(6)} errors=${errorCount}/${totalElems}`);
  assert.ok(errorCount === 0, `${errorCount}/${totalElems} elements exceed tolerance (atol=${ABS_TOL}, rtol=${REL_TOL})`);
}

describe("CP MLA Append Prefill (qSeqLen < kvSeqLen)", () => {
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

  it("2-shard CP append prefill matches baseline (prefix=32, new=9, pageSize=16)", () => {
    runCpAppendPrefillTest(glm, ws, 32, 9, 1, 16, 2);
  });

  it("2-shard CP append prefill matches baseline (prefix=16, new=8, pageSize=16)", () => {
    runCpAppendPrefillTest(glm, ws, 16, 8, 1, 16, 2);
  });

  it("2-shard CP append prefill matches baseline (prefix=16, new=1, pageSize=16)", () => {
    runCpAppendPrefillTest(glm, ws, 16, 1, 1, 16, 2);
  });
});

function shardKVParallel(
  ckvF32: Float32Array,
  kpeF32: Float32Array,
  seqLen: number,
  pageSize: number,
  worldSize: number,
  rank: number,
  headDimCkv: number,
  headDimKpe: number,
): { ckvShard: Float32Array; kpeShard: Float32Array; shardLen: number } {
  const vPS = pageSize / worldSize;
  const shardLen = rank < (seqLen % worldSize) ? Math.ceil(seqLen / worldSize) : Math.floor(seqLen / worldSize);
  const numPages = Math.ceil(seqLen / pageSize);
  const shardPages = Math.ceil(shardLen / vPS);

  const ckvShard = new Float32Array(shardPages * vPS * headDimCkv);
  const kpeShard = new Float32Array(shardPages * vPS * headDimKpe);

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

  return { ckvShard, kpeShard, shardLen };
}

describe("CP MLA Prefill via ParallelOps + PagedKVCache", () => {
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

  function runParallelCpPrefillTest(
    seqLen: number,
    batchSize: number,
    pageSize: number,
  ): void {
    const worldSize = 2;
    const totalTokens = batchSize * seqLen;
    const numPages = Math.ceil(seqLen / pageSize);
    const maxPages = numPages + 1;
    const nLayers = 1;

    const qNopeF32 = randomData(totalTokens * N_HEADS * HEAD_DIM_CKV);
    const qPeF32 = randomData(totalTokens * N_HEADS * HEAD_DIM_KPE);
    const ckvF32 = randomData(numPages * pageSize * HEAD_DIM_CKV);
    const kpeF32 = randomData(numPages * pageSize * HEAD_DIM_KPE);
    const vProjF32 = randomData(N_HEADS * V_HEAD_DIM * HEAD_DIM_CKV);

    // --- Baseline: single-GPU, no CP ---
    const baseline = runMlaPrefill(
      ref, refWs, qNopeF32, qPeF32, ckvF32, kpeF32,
      seqLen, seqLen, seqLen, batchSize, N_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
      pageSize, maxPages,
      pageSize * HEAD_DIM_CKV, pageSize * HEAD_DIM_KPE,
      0, 0,
    );
    const vProjRef = refWs.alloc([N_HEADS * V_HEAD_DIM, HEAD_DIM_CKV], "BF16");
    vProjRef.h2d(f32ToBf16Bytes(vProjF32));
    const baselineVExpanded = baseline.vOut.mlaVExpand(vProjRef, HEAD_DIM_CKV, V_HEAD_DIM, N_HEADS, seqLen, batchSize);
    const baselineVExpandedBuf = Buffer.alloc(totalTokens * N_HEADS * V_HEAD_DIM * 2);
    baselineVExpanded.d2h(baselineVExpandedBuf);
    const baselineVExpandedF32 = bf16BytesToF32(baselineVExpandedBuf);

    // --- CP via ParallelOps ---
    // Create PagedKVCache with contextParallel=true (allocates Row-parallel ckv/kpe)
    const pagedKV = new PagedKVCache(po, 1, HEAD_DIM_CKV, nLayers, maxPages, batchSize, pageSize, HEAD_DIM_CKV, HEAD_DIM_KPE, true);

    // Setup PagedKVCache page allocation manually
    pagedKV.reset(batchSize);
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pagedKV.allocAppendPages(seqIdx, seqLen);
    }

    // Manually write indices, indptr, lastPageLen via host buffers
    const indicesData = new Int32Array(maxPages);
    for (let i = 0; i < numPages; i++) indicesData[i] = i;
    pagedKV.indicesH.h2d(i32Buf(indicesData));
    pagedKV.pagesDirtyHost = false;
    pagedKV.pagesDirtyDevice = true;
    pagedKV.indices.memcpy(pagedKV.indicesH, maxPages * 4, 1 /* HostToDevice */);

    // Fill each shard's ckv/kpe with interleaved data
    const ckvData = pagedKV.ckvData[0] as ParallelTensor;
    const kpeData = pagedKV.kpeData[0] as ParallelTensor;
    for (let rank = 0; rank < worldSize; rank++) {
      const { ckvShard, kpeShard } = shardKVParallel(
        ckvF32, kpeF32, seqLen, pageSize, worldSize, rank, HEAD_DIM_CKV, HEAD_DIM_KPE,
      );
      ckvData.shards[rank].h2d(f32ToBf16Bytes(ckvShard));
      kpeData.shards[rank].h2d(f32ToBf16Bytes(kpeShard));
    }

    // Allocate Q as Replicated (all shards need all Q heads for CP)
    const pQNope = ws.alloc([1, totalTokens, N_HEADS * HEAD_DIM_CKV], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const pQPe = ws.alloc([1, totalTokens, N_HEADS * HEAD_DIM_KPE], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pQNope.h2d(f32ToBf16Bytes(qNopeF32));
    pQPe.h2d(f32ToBf16Bytes(qPeF32));

    // Allocate v_proj as Replicated (for simplicity)
    const pVProj = ws.alloc([N_HEADS * V_HEAD_DIM, HEAD_DIM_CKV], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pVProj.h2d(f32ToBf16Bytes(vProjF32));

    // Allocate workspace tensors (Replicated — same on each GPU)
    const floatWs = ws.alloc([128 * 1024 * 1024 / 2], "BF16", undefined, TensorParallelism.Replicated);
    const intWs = ws.alloc([8 * 1024 * 1024 / 4], "I32", undefined, TensorParallelism.Replicated);
    const pinnedIntWs = ws.allocPinned([8 * 1024 * 1024 / 4], "I32", undefined, TensorParallelism.Replicated);
    const planInfo = ws.allocPinned([18], "I32", undefined, TensorParallelism.Replicated);

    // Setup indptr, kvLenH — Replicated, same on each shard
    // kvLenH always contains the GLOBAL KV length; kernel derives local shard length from cp_world_size/cp_rank
    const shardLen = Math.floor(seqLen / worldSize);
    const indptrH = ws.allocPinned([batchSize + 1], "I32", undefined, TensorParallelism.Replicated);
    const kvLenH = ws.allocPinned([batchSize], "I32", undefined, TensorParallelism.Replicated);
    const lastPageLenH = ws.allocPinned([batchSize], "I32", undefined, TensorParallelism.Replicated);
    const qoIndptrH = ws.allocPinned([batchSize + 1], "I32", undefined, TensorParallelism.Replicated);
    indptrH.h2d(i32Buf(new Int32Array([0, numPages])));
    kvLenH.h2d(i32Buf(new Int32Array([seqLen])));
    qoIndptrH.h2d(i32Buf(new Int32Array([0, totalTokens])));

    // Plan MLA prefill — ParallelOps injects cpWorldSize=2, cpRank=i
    po.mlaPrefillPlan(
      floatWs, 128 * 1024 * 1024,
      intWs, pinnedIntWs, 8 * 1024 * 1024,
      planInfo,
      qoIndptrH, indptrH, kvLenH, lastPageLenH,
      batchSize, N_HEADS, HEAD_DIM_CKV, true,
      pageSize, [seqLen],
      true,
    );

    // Run MLA prefill — ParallelOps adjusts strides for CP and AllGathers Row-parallel Q
    const ckvStridePage = pageSize * HEAD_DIM_CKV;
    const kpeStridePage = pageSize * HEAD_DIM_KPE;
    const { o: pOut, lse: pLse } = po.mlaPrefillRun(
      pQNope, pQPe, ckvData, kpeData, pagedKV.indices,
      floatWs, intWs, planInfo,
      N_HEADS, pageSize, 1, SM_SCALE,
      N_HEADS * HEAD_DIM_CKV, HEAD_DIM_CKV, N_HEADS * HEAD_DIM_KPE, HEAD_DIM_KPE,
      ckvStridePage, HEAD_DIM_CKV, kpeStridePage, HEAD_DIM_KPE,
      HEAD_DIM_CKV, totalTokens * HEAD_DIM_CKV,
      HEAD_DIM_CKV, HEAD_DIM_KPE,
      true,
    );
    po.synchronize();

    // mlaVExpand with LSE — triggers CP merge when contextParallel=true
    const pVExpanded = pOut.mlaVExpand(pVProj, HEAD_DIM_CKV, V_HEAD_DIM, N_HEADS, seqLen, batchSize, pLse) as ParallelTensor;
    po.synchronize();

    // Compare with baseline
    const mergedBuf = Buffer.alloc(totalTokens * N_HEADS * V_HEAD_DIM * 2);
    pVExpanded.d2h(mergedBuf);
    const mergedF32 = bf16BytesToF32(mergedBuf);

    let maxRelErr = 0;
    let maxAbsErr = 0;
    let errorCount = 0;
    const totalElems = totalTokens * N_HEADS * V_HEAD_DIM;
    const SIGNIFICANCE_THRESHOLD = 0.05;
    const ABS_TOL = 0.03;
    const REL_TOL = 0.10;
    let maxRelErrForSignificant = 0;

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

    console.log(`ParallelOps CP prefill (seqLen=${seqLen}, batch=${batchSize}, pageSize=${pageSize}): maxAbsErr=${maxAbsErr.toFixed(6)} maxRelErrForSignificant=${maxRelErrForSignificant.toFixed(6)} errors=${errorCount}/${totalElems}`);
    assert.ok(errorCount === 0, `${errorCount}/${totalElems} elements exceed tolerance (atol=${ABS_TOL}, rtol=${REL_TOL})`);

    // Cleanup
    baseline.vOut[Symbol.dispose]();
    baseline.lse[Symbol.dispose]();
    vProjRef[Symbol.dispose]();
    baselineVExpanded[Symbol.dispose]();
    pagedKV.free();
  }

  it("2-GPU CP prefill via ParallelOps matches baseline (seqLen=32, batch=1, pageSize=16)", () => {
    runParallelCpPrefillTest(32, 1, 16);
  });

  it("2-GPU CP prefill via ParallelOps matches baseline (seqLen=64, batch=1, pageSize=16)", () => {
    runParallelCpPrefillTest(64, 1, 16);
  });
});
