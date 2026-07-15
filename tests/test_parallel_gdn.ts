import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { TensorParallelism } from "../src/device_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import type { ExecutionState } from "../src/execution-workspace";

function shardQkvData(
  fullData: Float32Array, rows: number, numHeads: number, dK: number, dV: number,
  shardIdx: number, worldSize: number,
): Float32Array {
  const shardHeads = numHeads / worldSize;
  const fullConvDim = numHeads * (dK * 2 + dV);
  const shardConvDim = shardHeads * (dK * 2 + dV);
  const qSectionSize = numHeads * dK;
  const shardQSize = shardHeads * dK;
  const shardKSize = shardHeads * dK;
  const shardVSize = shardHeads * dV;
  const result = new Float32Array(rows * shardConvDim);
  for (let r = 0; r < rows; r++) {
    const fb = r * fullConvDim;
    const sb = r * shardConvDim;
    for (let i = 0; i < shardQSize; i++) result[sb + i] = fullData[fb + shardIdx * shardQSize + i];
    for (let i = 0; i < shardKSize; i++) result[sb + shardQSize + i] = fullData[fb + qSectionSize + shardIdx * shardKSize + i];
    for (let i = 0; i < shardVSize; i++) result[sb + shardQSize + shardKSize + i] = fullData[fb + qSectionSize + qSectionSize + shardIdx * shardVSize + i];
  }
  return result;
}

function shardGdnChannels(
  fullData: Float32Array,
  numBatches: number,
  numHeads: number, dK: number, dV: number,
  channelWidth: number,
  shardIdx: number, worldSize: number,
): Float32Array {
  const shardHeads = numHeads / worldSize;
  const shardConvDim = shardHeads * (dK * 2 + dV);
  const result = new Float32Array(numBatches * shardConvDim * channelWidth);
  for (let b = 0; b < numBatches; b++) {
    const fb = b * numHeads * (dK * 2 + dV) * channelWidth;
    const sb = b * shardConvDim * channelWidth;
    const qSize = shardHeads * dK * channelWidth;
    const kSize = shardHeads * dK * channelWidth;
    const vSize = shardHeads * dV * channelWidth;
    for (let i = 0; i < qSize; i++) result[sb + i] = fullData[fb + shardIdx * qSize + i];
    for (let i = 0; i < kSize; i++) result[sb + qSize + i] = fullData[fb + numHeads * dK * channelWidth + shardIdx * kSize + i];
    for (let i = 0; i < vSize; i++) result[sb + qSize + kSize + i] = fullData[fb + 2 * numHeads * dK * channelWidth + shardIdx * vSize + i];
  }
  return result;
}

function shardRowF32(
  fullData: Float32Array, rows: number, cols: number,
  shardIdx: number, worldSize: number,
): Float32Array {
  const shardCols = cols / worldSize;
  const result = new Float32Array(rows * shardCols);
  for (let r = 0; r < rows; r++) {
    const srcOff = r * cols + shardIdx * shardCols;
    const dstOff = r * shardCols;
    for (let i = 0; i < shardCols; i++) result[dstOff + i] = fullData[srcOff + i];
  }
  return result;
}

function gatherRowBf16(
  shard0Buf: Buffer, shard1Buf: Buffer, rows: number, fullCols: number, worldSize: number,
): Float32Array {
  const shardCols = fullCols / worldSize;
  const s0 = bf16BytesToF32(shard0Buf);
  const s1 = bf16BytesToF32(shard1Buf);
  const result = new Float32Array(rows * fullCols);
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < shardCols; i++) {
      result[r * fullCols + i] = s0[r * shardCols + i];
      result[r * fullCols + shardCols + i] = s1[r * shardCols + i];
    }
  }
  return result;
}

function gatherRowF32(
  shard0Buf: Buffer, shard1Buf: Buffer, rows: number, fullCols: number, worldSize: number,
): Float32Array {
  const shardCols = fullCols / worldSize;
  const s0 = new Float32Array(shard0Buf.buffer, shard0Buf.byteOffset, shard0Buf.length / 4);
  const s1 = new Float32Array(shard1Buf.buffer, shard1Buf.byteOffset, shard1Buf.length / 4);
  const result = new Float32Array(rows * fullCols);
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < shardCols; i++) {
      result[r * fullCols + i] = s0[r * shardCols + i];
      result[r * fullCols + shardCols + i] = s1[r * shardCols + i];
    }
  }
  return result;
}

describe("ParallelOps GDN recurrent step", () => {
  const numHeads = 4;
  const dK = 8;
  const dV = 8;
  const convDim = numHeads * (dK * 2 + dV);
  const zDim = numHeads * dV;
  const stateStride = numHeads * dK * dV;
  const batchSize = 2;
  const qkvChStride = 1;
  const qkvSeqStride = convDim;

  let glm0: GlmOps;
  let glm1: GlmOps;
  let po: ParallelOps;
  let ws: WorkspaceBase;
  let refGlm: GlmOps;
  let refWs: WorkspaceBase;

  before(() => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
    ws = new WorkspaceBase(po);
    refGlm = new GlmOps(0);
    refWs = new WorkspaceBase(refGlm);
  });

  after(() => {
    refWs.free();
    refGlm.free();
    ws.free();
    po.free();
    glm1.free();
    glm0.free();
  });

  it("gdnRecurrentStep matches single-GPU reference (Column A_log/dt_bias)", () => {
    const qkvF32 = new Float32Array(batchSize * convDim);
    const aRawF32 = new Float32Array(batchSize * numHeads);
    const bRawF32 = new Float32Array(batchSize * numHeads);
    const aLogF32 = new Float32Array(numHeads);
    const dtBiasF32 = new Float32Array(numHeads);
    const stateF32 = new Float32Array(batchSize * stateStride);

    for (let i = 0; i < qkvF32.length; i++) qkvF32[i] = Math.sin(i * 0.37) * 0.5;
    for (let i = 0; i < aRawF32.length; i++) aRawF32[i] = Math.cos(i * 0.73) * 0.3;
    for (let i = 0; i < bRawF32.length; i++) bRawF32[i] = Math.sin(i * 1.17) * 0.4;
    aLogF32[0] = -4.0; aLogF32[1] = -2.0; aLogF32[2] = 1.0; aLogF32[3] = 3.0;
    dtBiasF32[0] = -0.5; dtBiasF32[1] = 0.5; dtBiasF32[2] = 1.5; dtBiasF32[3] = 2.5;
    for (let i = 0; i < stateF32.length; i++) stateF32[i] = Math.cos(i * 0.19) * 0.1;

    const refState = refWs.alloc([batchSize, stateStride], "F32");
    const refQkv = refWs.alloc([batchSize, convDim], "BF16");
    const refARaw = refWs.alloc([batchSize, numHeads], "BF16");
    const refBRaw = refWs.alloc([batchSize, numHeads], "BF16");
    const refALog = refWs.alloc([numHeads], "F32");
    const refDtBias = refWs.alloc([numHeads], "F32");
    const refOutput = refWs.alloc([batchSize, zDim], "BF16");

    refState.h2d(Buffer.from(stateF32.buffer));
    refQkv.h2d(f32ToBf16Bytes(qkvF32));
    refARaw.h2d(f32ToBf16Bytes(aRawF32));
    refBRaw.h2d(f32ToBf16Bytes(bRawF32));
    refALog.h2d(Buffer.from(aLogF32.buffer));
    refDtBias.h2d(Buffer.from(dtBiasF32.buffer));
    refGlm.synchronize();

    refGlm.gdnRecurrentStep(
      { batchSize } as ExecutionState, refOutput, refState, refQkv, refARaw, refBRaw, refALog, refDtBias,
      numHeads, dK, dV, stateStride, qkvChStride, qkvSeqStride,
    );
    refGlm.synchronize();

    const refOutputBuf = Buffer.alloc(batchSize * zDim * 2);
    const refStateBuf = Buffer.alloc(batchSize * stateStride * 4);
    refOutput.d2h(refOutputBuf);
    refState.d2h(refStateBuf);
    const refOutputF32 = bf16BytesToF32(refOutputBuf);
    const refStateAfter = new Float32Array(refStateBuf.buffer, refStateBuf.byteOffset, batchSize * stateStride);

    refWs.free();

    const shardHeads = numHeads / 2;
    const shardConvDim = shardHeads * (dK * 2 + dV);
    const shardZDim = shardHeads * dV;
    const shardStateStride = shardHeads * dK * dV;

    const tpState = ws.alloc([batchSize, stateStride], "F32", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpQkv = ws.alloc([batchSize, convDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpARaw = ws.alloc([batchSize, numHeads], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpBRaw = ws.alloc([batchSize, numHeads], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpALog = ws.alloc([numHeads], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    const tpDtBias = ws.alloc([numHeads], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    const tpOutput = ws.alloc([batchSize, zDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;

    for (let s = 0; s < 2; s++) {
      const shardStateF32 = shardRowF32(stateF32, batchSize, stateStride, s, 2);
      const shardQkvF32 = shardQkvData(qkvF32, batchSize, numHeads, dK, dV, s, 2);
      const shardARawF32 = shardRowF32(aRawF32, batchSize, numHeads, s, 2);
      const shardBRawF32 = shardRowF32(bRawF32, batchSize, numHeads, s, 2);
      const shardALogF32 = aLogF32.slice(s * shardHeads, (s + 1) * shardHeads);
      const shardDtBiasF32 = dtBiasF32.slice(s * shardHeads, (s + 1) * shardHeads);

      tpState.shards[s].h2d(Buffer.from(shardStateF32.buffer));
      tpQkv.shards[s].h2d(f32ToBf16Bytes(shardQkvF32));
      tpARaw.shards[s].h2d(f32ToBf16Bytes(shardARawF32));
      tpBRaw.shards[s].h2d(f32ToBf16Bytes(shardBRawF32));
      tpALog.shards[s].h2d(Buffer.from(shardALogF32.buffer));
      tpDtBias.shards[s].h2d(Buffer.from(shardDtBiasF32.buffer));
    }
    po.synchronize();

    po.gdnRecurrentStep(
      { batchSize } as ExecutionState, tpOutput, tpState, tpQkv, tpARaw, tpBRaw, tpALog, tpDtBias,
      numHeads, dK, dV, stateStride, qkvChStride, qkvSeqStride,
    );
    po.synchronize();

    const tpOutputBuf0 = Buffer.alloc(batchSize * shardZDim * 2);
    const tpOutputBuf1 = Buffer.alloc(batchSize * shardZDim * 2);
    tpOutput.shards[0].d2h(tpOutputBuf0);
    tpOutput.shards[1].d2h(tpOutputBuf1);
    const tpOutputF32 = gatherRowBf16(tpOutputBuf0, tpOutputBuf1, batchSize, zDim, 2);

    const tpStateBuf0 = Buffer.alloc(batchSize * shardStateStride * 4);
    const tpStateBuf1 = Buffer.alloc(batchSize * shardStateStride * 4);
    tpState.shards[0].d2h(tpStateBuf0);
    tpState.shards[1].d2h(tpStateBuf1);
    const tpStateAfter = gatherRowF32(tpStateBuf0, tpStateBuf1, batchSize, stateStride, 2);

    let maxOutputDiff = 0;
    for (let i = 0; i < batchSize * zDim; i++) {
      const diff = Math.abs(tpOutputF32[i] - refOutputF32[i]);
      if (diff > maxOutputDiff) maxOutputDiff = diff;
    }

    let maxStateDiff = 0;
    for (let i = 0; i < batchSize * stateStride; i++) {
      const diff = Math.abs(tpStateAfter[i] - refStateAfter[i]);
      if (diff > maxStateDiff) maxStateDiff = diff;
    }

    assert.ok(maxOutputDiff < 0.01, `gdnRecurrentStep output max_diff=${maxOutputDiff} exceeds 0.01`);
    assert.ok(maxStateDiff < 0.001, `gdnRecurrentStep state max_diff=${maxStateDiff} exceeds 0.001`);
  });

  it("gdnRecurrentStep with Replicated A_log/dt_bias gives WRONG result (bug demonstration)", () => {
    const qkvF32 = new Float32Array(batchSize * convDim);
    const aRawF32 = new Float32Array(batchSize * numHeads);
    const bRawF32 = new Float32Array(batchSize * numHeads);
    const aLogF32 = new Float32Array(numHeads);
    const dtBiasF32 = new Float32Array(numHeads);
    const stateF32 = new Float32Array(batchSize * stateStride);

    for (let i = 0; i < qkvF32.length; i++) qkvF32[i] = Math.sin(i * 0.37) * 0.5;
    for (let i = 0; i < aRawF32.length; i++) aRawF32[i] = Math.cos(i * 0.73) * 0.3;
    for (let i = 0; i < bRawF32.length; i++) bRawF32[i] = Math.sin(i * 1.17) * 0.4;
    aLogF32[0] = -4.0; aLogF32[1] = -2.0; aLogF32[2] = 1.0; aLogF32[3] = 3.0;
    dtBiasF32[0] = -0.5; dtBiasF32[1] = 0.5; dtBiasF32[2] = 1.5; dtBiasF32[3] = 2.5;
    for (let i = 0; i < stateF32.length; i++) stateF32[i] = Math.cos(i * 0.19) * 0.1;

    const refState = refWs.alloc([batchSize, stateStride], "F32");
    const refQkv = refWs.alloc([batchSize, convDim], "BF16");
    const refARaw = refWs.alloc([batchSize, numHeads], "BF16");
    const refBRaw = refWs.alloc([batchSize, numHeads], "BF16");
    const refALog = refWs.alloc([numHeads], "F32");
    const refDtBias = refWs.alloc([numHeads], "F32");
    const refOutput = refWs.alloc([batchSize, zDim], "BF16");

    refState.h2d(Buffer.from(stateF32.buffer));
    refQkv.h2d(f32ToBf16Bytes(qkvF32));
    refARaw.h2d(f32ToBf16Bytes(aRawF32));
    refBRaw.h2d(f32ToBf16Bytes(bRawF32));
    refALog.h2d(Buffer.from(aLogF32.buffer));
    refDtBias.h2d(Buffer.from(dtBiasF32.buffer));
    refGlm.synchronize();

    refGlm.gdnRecurrentStep(
      { batchSize } as ExecutionState, refOutput, refState, refQkv, refARaw, refBRaw, refALog, refDtBias,
      numHeads, dK, dV, stateStride, qkvChStride, qkvSeqStride,
    );
    refGlm.synchronize();

    const refOutputBuf = Buffer.alloc(batchSize * zDim * 2);
    refOutput.d2h(refOutputBuf);
    const refOutputF32 = bf16BytesToF32(refOutputBuf);

    refWs.free();

    const shardHeads = numHeads / 2;
    const shardConvDim = shardHeads * (dK * 2 + dV);
    const shardZDim = shardHeads * dV;
    const shardStateStride = shardHeads * dK * dV;

    const tpState = ws.alloc([batchSize, stateStride], "F32", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpQkv = ws.alloc([batchSize, convDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpARaw = ws.alloc([batchSize, numHeads], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpBRaw = ws.alloc([batchSize, numHeads], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpALog = ws.alloc([numHeads], "F32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const tpDtBias = ws.alloc([numHeads], "F32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const tpOutput = ws.alloc([batchSize, zDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;

    for (let s = 0; s < 2; s++) {
      const shardStateF32 = shardRowF32(stateF32, batchSize, stateStride, s, 2);
      const shardQkvF32 = shardQkvData(qkvF32, batchSize, numHeads, dK, dV, s, 2);
      const shardARawF32 = shardRowF32(aRawF32, batchSize, numHeads, s, 2);
      const shardBRawF32 = shardRowF32(bRawF32, batchSize, numHeads, s, 2);

      tpState.shards[s].h2d(Buffer.from(shardStateF32.buffer));
      tpQkv.shards[s].h2d(f32ToBf16Bytes(shardQkvF32));
      tpARaw.shards[s].h2d(f32ToBf16Bytes(shardARawF32));
      tpBRaw.shards[s].h2d(f32ToBf16Bytes(shardBRawF32));
    }

    tpALog.h2d(Buffer.from(aLogF32.buffer));
    tpDtBias.h2d(Buffer.from(dtBiasF32.buffer));
    po.synchronize();

    po.gdnRecurrentStep(
      { batchSize } as ExecutionState, tpOutput, tpState, tpQkv, tpARaw, tpBRaw, tpALog, tpDtBias,
      numHeads, dK, dV, stateStride, qkvChStride, qkvSeqStride,
    );
    po.synchronize();

    const tpOutputBuf0 = Buffer.alloc(batchSize * shardZDim * 2);
    const tpOutputBuf1 = Buffer.alloc(batchSize * shardZDim * 2);
    tpOutput.shards[0].d2h(tpOutputBuf0);
    tpOutput.shards[1].d2h(tpOutputBuf1);
    const tpOutputF32 = gatherRowBf16(tpOutputBuf0, tpOutputBuf1, batchSize, zDim, 2);

    let maxDiff = 0;
    for (let i = 0; i < batchSize * zDim; i++) {
      const diff = Math.abs(tpOutputF32[i] - refOutputF32[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    assert.ok(maxDiff >= 0.01, `Replicated A_log/dt_bias should produce WRONG results (max_diff=${maxDiff}), but matched reference`);
  });
});

describe("ParallelOps GDN prefill", () => {
  const numHeads = 4;
  const dK = 8;
  const dV = 8;
  const convDim = numHeads * (dK * 2 + dV);
  const zDim = numHeads * dV;
  const stateStride = numHeads * dK * dV;
  const batchSize = 2;
  const seqLen = 4;
  const totalSeqLen = batchSize * seqLen;
  const qkvChStride = 1;
  const qkvSeqStride = convDim;

  let glm0: GlmOps;
  let glm1: GlmOps;
  let po: ParallelOps;
  let ws: WorkspaceBase;
  let refGlm: GlmOps;
  let refWs: WorkspaceBase;

  before(() => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
    ws = new WorkspaceBase(po);
    refGlm = new GlmOps(0);
    refWs = new WorkspaceBase(refGlm);
  });

  after(() => {
    refWs.free();
    refGlm.free();
    ws.free();
    po.free();
    glm1.free();
    glm0.free();
  });

  it("gdnPrefill matches single-GPU reference (Column A_log/dt_bias)", () => {
    const qkvF32 = new Float32Array(totalSeqLen * convDim);
    const aRawF32 = new Float32Array(totalSeqLen * numHeads);
    const bRawF32 = new Float32Array(totalSeqLen * numHeads);
    const aLogF32 = new Float32Array(numHeads);
    const dtBiasF32 = new Float32Array(numHeads);
    const stateF32 = new Float32Array(batchSize * stateStride);
    const cuSeqlens = new Int32Array([0, seqLen, totalSeqLen]);

    for (let i = 0; i < qkvF32.length; i++) qkvF32[i] = Math.sin(i * 0.41) * 0.3;
    for (let i = 0; i < aRawF32.length; i++) aRawF32[i] = Math.cos(i * 0.67) * 0.2;
    for (let i = 0; i < bRawF32.length; i++) bRawF32[i] = Math.sin(i * 1.31) * 0.3;
    aLogF32[0] = -4.0; aLogF32[1] = -2.0; aLogF32[2] = 1.0; aLogF32[3] = 3.0;
    dtBiasF32[0] = -0.5; dtBiasF32[1] = 0.5; dtBiasF32[2] = 1.5; dtBiasF32[3] = 2.5;
    for (let i = 0; i < stateF32.length; i++) stateF32[i] = 0;

    const refState = refWs.alloc([batchSize, stateStride], "F32");
    const refQkv = refWs.alloc([totalSeqLen, convDim], "BF16");
    const refARaw = refWs.alloc([totalSeqLen, numHeads], "BF16");
    const refBRaw = refWs.alloc([totalSeqLen, numHeads], "BF16");
    const refALog = refWs.alloc([numHeads], "F32");
    const refDtBias = refWs.alloc([numHeads], "F32");
    const refCuSeqlens = refWs.alloc([batchSize + 1], "I32");
    const refOutput = refWs.alloc([totalSeqLen, zDim], "BF16");

    refState.h2d(Buffer.from(stateF32.buffer));
    refQkv.h2d(f32ToBf16Bytes(qkvF32));
    refARaw.h2d(f32ToBf16Bytes(aRawF32));
    refBRaw.h2d(f32ToBf16Bytes(bRawF32));
    refALog.h2d(Buffer.from(aLogF32.buffer));
    refDtBias.h2d(Buffer.from(dtBiasF32.buffer));
    refCuSeqlens.h2d(Buffer.from(cuSeqlens.buffer));
    refGlm.synchronize();

    refGlm.gdnPrefill(
      { batchSize, totalTokens: totalSeqLen } as ExecutionState, refOutput, refState, refQkv, refARaw, refBRaw, refALog, refDtBias,
      refCuSeqlens, numHeads, dK, dV,
      stateStride, qkvChStride, qkvSeqStride,
    );
    refGlm.synchronize();

    const refOutputBuf = Buffer.alloc(totalSeqLen * zDim * 2);
    refOutput.d2h(refOutputBuf);
    const refOutputF32 = bf16BytesToF32(refOutputBuf);

    refWs.free();

    const shardHeads = numHeads / 2;
    const shardZDim = shardHeads * dV;

    const tpState = ws.alloc([batchSize, stateStride], "F32", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpQkv = ws.alloc([totalSeqLen, convDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpARaw = ws.alloc([totalSeqLen, numHeads], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpBRaw = ws.alloc([totalSeqLen, numHeads], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpALog = ws.alloc([numHeads], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    const tpDtBias = ws.alloc([numHeads], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    const tpCuSeqlens = ws.alloc([batchSize + 1], "I32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const tpOutput = ws.alloc([totalSeqLen, zDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;

    for (let s = 0; s < 2; s++) {
      const shardStateF32 = shardRowF32(stateF32, batchSize, stateStride, s, 2);
      const shardQkvF32 = shardQkvData(qkvF32, totalSeqLen, numHeads, dK, dV, s, 2);
      const shardARawF32 = shardRowF32(aRawF32, totalSeqLen, numHeads, s, 2);
      const shardBRawF32 = shardRowF32(bRawF32, totalSeqLen, numHeads, s, 2);
      const shardALogF32 = aLogF32.slice(s * shardHeads, (s + 1) * shardHeads);
      const shardDtBiasF32 = dtBiasF32.slice(s * shardHeads, (s + 1) * shardHeads);

      tpState.shards[s].h2d(Buffer.from(shardStateF32.buffer));
      tpQkv.shards[s].h2d(f32ToBf16Bytes(shardQkvF32));
      tpARaw.shards[s].h2d(f32ToBf16Bytes(shardARawF32));
      tpBRaw.shards[s].h2d(f32ToBf16Bytes(shardBRawF32));
      tpALog.shards[s].h2d(Buffer.from(shardALogF32.buffer));
      tpDtBias.shards[s].h2d(Buffer.from(shardDtBiasF32.buffer));
    }
    tpCuSeqlens.h2d(Buffer.from(cuSeqlens.buffer));
    po.synchronize();

    po.gdnPrefill(
      { batchSize, totalTokens: totalSeqLen } as ExecutionState, tpOutput, tpState, tpQkv, tpARaw, tpBRaw, tpALog, tpDtBias,
      tpCuSeqlens, numHeads, dK, dV,
      stateStride, qkvChStride, qkvSeqStride,
    );
    po.synchronize();

    const tpOutputBuf0 = Buffer.alloc(totalSeqLen * shardZDim * 2);
    const tpOutputBuf1 = Buffer.alloc(totalSeqLen * shardZDim * 2);
    tpOutput.shards[0].d2h(tpOutputBuf0);
    tpOutput.shards[1].d2h(tpOutputBuf1);
    const tpOutputF32 = gatherRowBf16(tpOutputBuf0, tpOutputBuf1, totalSeqLen, zDim, 2);

    let maxDiff = 0;
    for (let i = 0; i < totalSeqLen * zDim; i++) {
      const diff = Math.abs(tpOutputF32[i] - refOutputF32[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    assert.ok(maxDiff < 0.05, `gdnPrefill output max_diff=${maxDiff} exceeds 0.05`);
  });
});



describe("ParallelOps rmsnormGated", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let po: ParallelOps;
  let ws: WorkspaceBase;
  let refGlm: GlmOps;
  let refWs: WorkspaceBase;

  before(() => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
    ws = new WorkspaceBase(po);
    refGlm = new GlmOps(0);
    refWs = new WorkspaceBase(refGlm);
  });

  after(() => {
    refWs.free();
    refGlm.free();
    ws.free();
    po.free();
    glm1.free();
    glm0.free();
  });

  it("rmsnormGated on Row-parallel tensors matches single-GPU reference", () => {
    const seqLen = 2;
    const numHeads = 4;
    const dV = 8;
    const zDim = numHeads * dV;
    const batch = seqLen * numHeads;
    const dim = dV;
    const eps = 1e-6;

    const inputF32 = new Float32Array(seqLen * zDim);
    const gateF32 = new Float32Array(seqLen * zDim);
    const weightF32 = new Float32Array(dim);

    for (let i = 0; i < seqLen * zDim; i++) {
      inputF32[i] = Math.sin(i * 0.53) * 0.5;
      gateF32[i] = Math.cos(i * 0.37) * 0.3;
    }
    for (let i = 0; i < dim; i++) weightF32[i] = 0.8 + i * 0.02;

    const refInput = refWs.alloc([seqLen, zDim], "BF16");
    const refGate = refWs.alloc([seqLen, zDim], "BF16");
    const refWeight = refWs.alloc([dim], "BF16");
    const refOutput = refWs.alloc([seqLen, zDim], "BF16");

    refInput.h2d(f32ToBf16Bytes(inputF32));
    refGate.h2d(f32ToBf16Bytes(gateF32));
    refWeight.h2d(f32ToBf16Bytes(weightF32));
    refGlm.synchronize();

    refOutput.rmsnormGated(refInput, refGate, refWeight, eps);
    refGlm.synchronize();

    const refOutputBuf = Buffer.alloc(seqLen * zDim * 2);
    refOutput.d2h(refOutputBuf);
    const refOutputF32 = bf16BytesToF32(refOutputBuf);

    refWs.free();

    const tpInput = ws.alloc([seqLen, zDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpGate = ws.alloc([seqLen, zDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const tpWeight = ws.alloc([dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const tpOutput = ws.alloc([seqLen, zDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;

    tpInput.h2d(f32ToBf16Bytes(inputF32));
    tpGate.h2d(f32ToBf16Bytes(gateF32));
    tpWeight.h2d(f32ToBf16Bytes(weightF32));
    po.synchronize();

    tpOutput.rmsnormGated(tpInput, tpGate, tpWeight, eps);
    po.synchronize();

    const tpOutputBuf = Buffer.alloc(seqLen * zDim * 2);
    tpOutput.d2h(tpOutputBuf);
    const tpOutputF32 = bf16BytesToF32(tpOutputBuf);

    let maxDiff = 0;
    for (let i = 0; i < seqLen * zDim; i++) {
      const diff = Math.abs(tpOutputF32[i] - refOutputF32[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    assert.ok(maxDiff < 0.01, `rmsnormGated Row-parallel max_diff=${maxDiff} exceeds 0.01`);
  });
});
