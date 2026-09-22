import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { TensorParallelism } from "../src/device_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";

function shardColF32(
  fullData: Float32Array, rows: number, cols: number,
  shardIdx: number, worldSize: number,
): Float32Array {
  const shardCols = cols / worldSize;
  const result = new Float32Array(rows * shardCols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < shardCols; c++) {
      result[r * shardCols + c] = fullData[r * cols + shardIdx * shardCols + c];
    }
  }
  return result;
}

function gatherColBf16(
  shard0Buf: Buffer, shard1Buf: Buffer, rows: number, fullCols: number,
): Float32Array {
  const shardCols = fullCols / 2;
  const s0 = bf16BytesToF32(shard0Buf);
  const s1 = bf16BytesToF32(shard1Buf);
  const result = new Float32Array(rows * fullCols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < shardCols; c++) {
      result[r * fullCols + c] = s0[r * shardCols + c];
      result[r * fullCols + shardCols + c] = s1[r * shardCols + c];
    }
  }
  return result;
}

function shardColumnHeads(
  fullData: Float32Array,
  totalRows: number,
  nHeads: number,
  headDim: number,
  shardIdx: number,
  worldSize: number,
): Float32Array {
  const shardHeads = nHeads / worldSize;
  const shardRows = totalRows;
  const result = new Float32Array(shardRows * shardHeads * headDim);
  for (let r = 0; r < shardRows; r++) {
    for (let h = 0; h < shardHeads; h++) {
      for (let d = 0; d < headDim; d++) {
        const srcIdx = r * nHeads * headDim + (shardIdx * shardHeads + h) * headDim + d;
        const dstIdx = r * shardHeads * headDim + h * headDim + d;
        result[dstIdx] = fullData[srcIdx];
      }
    }
  }
  return result;
}

function gatherColumnHeads(
  shard0Buf: Buffer,
  shard1Buf: Buffer,
  totalRows: number,
  nHeads: number,
  headDim: number,
): Float32Array {
  const shardHeads = nHeads / 2;
  const s0 = bf16BytesToF32(shard0Buf);
  const s1 = bf16BytesToF32(shard1Buf);
  const result = new Float32Array(totalRows * nHeads * headDim);
  for (let r = 0; r < totalRows; r++) {
    for (let h = 0; h < shardHeads; h++) {
      for (let d = 0; d < headDim; d++) {
        result[r * nHeads * headDim + h * headDim + d] = s0[r * shardHeads * headDim + h * headDim + d];
        result[r * nHeads * headDim + (shardHeads + h) * headDim + d] = s1[r * shardHeads * headDim + h * headDim + d];
      }
    }
  }
  return result;
}

function gatherRopeTransposeOutput(
  shard0Buf: Buffer,
  shard1Buf: Buffer,
  batch: number,
  nHeads: number,
  seqLen: number,
  headDim: number,
): Float32Array {
  const shardHeads = nHeads / 2;
  const s0 = bf16BytesToF32(shard0Buf);
  const s1 = bf16BytesToF32(shard1Buf);
  const result = new Float32Array(batch * seqLen * nHeads * headDim);
  for (let b = 0; b < batch; b++) {
    for (let s = 0; s < seqLen; s++) {
      for (let h = 0; h < nHeads; h++) {
        const src = h < shardHeads ? s0 : s1;
        const shardH = h < shardHeads ? h : h - shardHeads;
        for (let d = 0; d < headDim; d++) {
          result[((b * seqLen + s) * nHeads + h) * headDim + d] =
            src[((b * seqLen + s) * shardHeads + shardH) * headDim + d];
        }
      }
    }
  }
  return result;
}

describe("ParallelOps.applyRotaryPosEmb token-major", () => {
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

  it("ropeDim=0 identity returns the input itself", () => {
    const batch = 2;
    const nHeads = 4;
    const seqLen = 3;
    const headDim = 16;
    const totalRows = batch * seqLen;

    const inputF32 = new Float32Array(totalRows * nHeads * headDim);
    for (let i = 0; i < inputF32.length; i++) inputF32[i] = (i % 13 - 6) * 0.1;

    const pInput = ws.alloc([totalRows, nHeads * headDim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pInput.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const pOut = pInput.applyRotaryPosEmb(undefined!, undefined!, 0, headDim, nHeads, seqLen, batch, 2) as ParallelTensor;

    assert.notEqual(pOut, pInput);
    assert.deepEqual(pOut.shape, [totalRows, nHeads * headDim]);

    const outBuf = Buffer.alloc(totalRows * nHeads * headDim * 2);
    pOut.d2h(outBuf);
    const outF32 = bf16BytesToF32(outBuf);
    const inF32 = bf16BytesToF32(f32ToBf16Bytes(inputF32));
    assert.deepEqual(outF32, inF32);

    pOut[Symbol.dispose]();
    pInput[Symbol.dispose]();
  });

  it("Row-parallel token-major with RoPE matches single-GPU", () => {
    const batch = 1;
    const nHeads = 4;
    const seqLen = 4;
    const headDim = 16;
    const ropeDim = 8;
    const dimHalf = ropeDim / 2;
    const totalRows = batch * seqLen;

    const inputF32 = new Float32Array(totalRows * nHeads * headDim);
    for (let i = 0; i < inputF32.length; i++) inputF32[i] = (i % 11 - 5) * 0.1;

    const positionIdsF32 = new Int32Array(batch * seqLen);
    for (let b = 0; b < batch; b++) {
      for (let s = 0; s < seqLen; s++) {
        positionIdsF32[b * seqLen + s] = s;
      }
    }
    const positionIdsBuf = Buffer.alloc(batch * seqLen * 4);
    for (let i = 0; i < batch * seqLen; i++) positionIdsBuf.writeInt32LE(positionIdsF32[i], i * 4);

    const invFreqF32 = new Float32Array(dimHalf);
    for (let i = 0; i < dimHalf; i++) invFreqF32[i] = 1.0 / (10000.0 ** (2.0 * i / ropeDim));

    const refInvFreq = refWs.alloc([dimHalf], "BF16");
    refInvFreq.h2d(f32ToBf16Bytes(invFreqF32));
    const refPosIds = refWs.alloc([batch * seqLen], "I32");
    refPosIds.h2d(positionIdsBuf);
    ref.synchronize();

    const { cos: refCos, sin: refSin } = refInvFreq.rotaryEmbedding(refPosIds, batch, seqLen);
    ref.synchronize();

    const refInput = refWs.alloc([totalRows, nHeads * headDim], "BF16");
    refInput.h2d(f32ToBf16Bytes(inputF32));
    ref.synchronize();

    const refRotated = refInput.applyRotaryPosEmb(refCos, refSin, ropeDim, headDim, nHeads, seqLen, batch, 2);
    const refOut = refRotated.reshape([batch * seqLen, nHeads, headDim]);
    ref.synchronize();
    const refBuf = Buffer.alloc(batch * nHeads * seqLen * headDim * 2);
    refOut.d2h(refBuf);
    const refF32 = bf16BytesToF32(refBuf);

    const pInvFreq = ws.alloc([dimHalf], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pInvFreq.h2d(f32ToBf16Bytes(invFreqF32));
    const pPosIds = ws.alloc([batch * seqLen], "I32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pPosIds.h2d(positionIdsBuf);
    po.synchronize();

    const { cos: pCos, sin: pSin } = pInvFreq.rotaryEmbedding(pPosIds, batch, seqLen);
    po.synchronize();

    const pInput = ws.alloc([totalRows, nHeads * headDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const shard0F32 = shardColumnHeads(inputF32, totalRows, nHeads, headDim, 0, 2);
    const shard1F32 = shardColumnHeads(inputF32, totalRows, nHeads, headDim, 1, 2);
    pInput.shard(0).h2d(f32ToBf16Bytes(shard0F32));
    pInput.shard(1).h2d(f32ToBf16Bytes(shard1F32));
    po.synchronize();

    const pRotated = pInput.applyRotaryPosEmb(pCos, pSin, ropeDim, headDim, nHeads, seqLen, batch, 2);
    const pOut = pRotated.reshape([batch * seqLen, nHeads, headDim]) as ParallelTensor;
    po.synchronize();

    assert.equal(pOut.parallelism, TensorParallelism.Row);
    assert.deepEqual(pOut.shape, [batch * seqLen, nHeads, headDim]);
    assert.deepEqual(pOut.shard(0).shape, [batch * seqLen, nHeads / 2, headDim]);

    const out0Buf = Buffer.alloc(batch * seqLen * nHeads / 2 * headDim * 2);
    const out1Buf = Buffer.alloc(batch * seqLen * nHeads / 2 * headDim * 2);
    pOut.shard(0).d2h(out0Buf);
    pOut.shard(1).d2h(out1Buf);
    const gatheredF32 = gatherRopeTransposeOutput(out0Buf, out1Buf, batch, nHeads, seqLen, headDim);

    for (let i = 0; i < refF32.length; i++) {
      const relErr = Math.abs(gatheredF32[i] - refF32[i]) / Math.max(Math.abs(refF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${refF32[i]}, got ${gatheredF32[i]} (relErr=${relErr})`);
    }

    refOut[Symbol.dispose]();
    refRotated[Symbol.dispose]();
    refInput[Symbol.dispose]();
    refCos[Symbol.dispose]();
    refSin[Symbol.dispose]();
    refInvFreq[Symbol.dispose]();
    refPosIds[Symbol.dispose]();
    pOut[Symbol.dispose]();
    pRotated[Symbol.dispose]();
    pInput[Symbol.dispose]();
    pCos[Symbol.dispose]();
    pSin[Symbol.dispose]();
    pInvFreq[Symbol.dispose]();
    pPosIds[Symbol.dispose]();
  });

  it("in_stride compaction gathers per-head windows", () => {
    const batch = 2;
    const nHeads = 4;
    const seqLen = 3;
    const headDim = 8;
    const inStride = 12;
    const totalRows = batch * seqLen;

    const inputF32 = new Float32Array(totalRows * nHeads * inStride);
    for (let i = 0; i < inputF32.length; i++) inputF32[i] = (i % 19 - 9) * 0.1;

    const pInput = ws.alloc([totalRows, nHeads * inStride], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pInput.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    // ropeDim=0 with inStride > headDim: the kernel runs as a pure gather.
    const pOut = pInput.applyRotaryPosEmb(undefined!, undefined!, 0, headDim, nHeads, seqLen, batch, 2, undefined, inStride) as ParallelTensor;
    po.synchronize();

    assert.deepEqual(pOut.shape, [totalRows, nHeads, headDim]);

    const outBuf = Buffer.alloc(totalRows * nHeads * headDim * 2);
    pOut.d2h(outBuf);
    const outF32 = bf16BytesToF32(outBuf);

    for (let t = 0; t < totalRows; t++) {
      for (let h = 0; h < nHeads; h++) {
        for (let d = 0; d < headDim; d++) {
          const expected = inputF32[t * nHeads * inStride + h * inStride + d];
          const got = outF32[(t * nHeads + h) * headDim + d];
          assert.ok(Math.abs(got - expected) < 1e-3, `t=${t} h=${h} d=${d}: expected ${expected}, got ${got}`);
        }
      }
    }

    pOut[Symbol.dispose]();
    pInput[Symbol.dispose]();
  });
});

describe("ParallelOps.applyRotaryPosEmb", () => {
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

  it("partial RoPE matches split/rotate/concat for packed indexer K rows", () => {
    const ropeDim = 64;
    for (const [batch, seqLen] of [[1, 7], [4, 1]]) {
      for (const headDim of [64, 128]) {
        for (const interleaved of [false, true]) {
          const totalRows = batch * seqLen;
          const inputBytes = f32ToBf16Bytes(Float32Array.from(
            { length: totalRows * headDim }, (_, i) => (i % 37 - 18) / 16,
          ));
          const cosBytes = f32ToBf16Bytes(Float32Array.from(
            { length: totalRows * ropeDim }, (_, i) => Math.cos(i / 31),
          ));
          const sinBytes = f32ToBf16Bytes(Float32Array.from(
            { length: totalRows * ropeDim }, (_, i) => Math.sin(i / 31),
          ));
          using input = ws.alloc([totalRows, headDim], "BF16") as ParallelTensor;
          using cos = ws.alloc([batch, seqLen, ropeDim], "BF16") as ParallelTensor;
          using sin = ws.alloc([batch, seqLen, ropeDim], "BF16") as ParallelTensor;
          input.h2d(inputBytes);
          cos.h2d(cosBytes);
          sin.h2d(sinBytes);
          using pe = input.slice(1, 0, ropeDim);
          using rotated = pe.applyRotaryPosEmb(cos, sin, ropeDim, ropeDim, 1, seqLen, batch, 1, interleaved);
          using nope = headDim > ropeDim ? input.slice(1, ropeDim, headDim - ropeDim) : undefined;
          using expected = (nope ? rotated.cat([nope], 1) : rotated.viewClone()) as ParallelTensor;
          using actual = input.applyRotaryPosEmb(cos, sin, ropeDim, headDim, 1, seqLen, batch, 1, interleaved) as ParallelTensor;
          po.synchronize();
          assert.deepEqual(actual.shape, input.shape);
          assert.equal(actual.parallelism, TensorParallelism.Replicated);
          for (let rank = 0; rank < 2; rank++) {
            const actualBytes = Buffer.alloc(inputBytes.length);
            const expectedBytes = Buffer.alloc(inputBytes.length);
            actual.shard(rank).d2h(actualBytes);
            expected.shard(rank).d2h(expectedBytes);
            assert.deepEqual(actualBytes, expectedBytes, `batch=${batch} seqLen=${seqLen} headDim=${headDim} interleaved=${interleaved} rank=${rank}`);
          }
        }
      }
    }
  });

  it("Row-parallel applyRotaryPosEmb matches single-GPU", () => {
    const batch = 2;
    const nHeads = 4;
    const seqLen = 4;
    const ropeDim = 8;
    const dimHalf = ropeDim / 2;

    const inputF32 = new Float32Array(batch * nHeads * seqLen * ropeDim);
    for (let i = 0; i < inputF32.length; i++) inputF32[i] = (i % 11 - 5) * 0.1;

    const positionIdsF32 = new Int32Array(batch * seqLen);
    for (let b = 0; b < batch; b++) {
      for (let s = 0; s < seqLen; s++) positionIdsF32[b * seqLen + s] = s;
    }
    const positionIdsBuf = Buffer.alloc(batch * seqLen * 4);
    for (let i = 0; i < batch * seqLen; i++) positionIdsBuf.writeInt32LE(positionIdsF32[i], i * 4);

    const invFreqF32 = new Float32Array(dimHalf);
    for (let i = 0; i < dimHalf; i++) invFreqF32[i] = 1.0 / (10000.0 ** (2.0 * i / ropeDim));

    const refInvFreq = refWs.alloc([dimHalf], "BF16");
    refInvFreq.h2d(f32ToBf16Bytes(invFreqF32));
    const refPosIds = refWs.alloc([batch * seqLen], "I32");
    refPosIds.h2d(positionIdsBuf);
    ref.synchronize();

    const { cos: refCos, sin: refSin } = refInvFreq.rotaryEmbedding(refPosIds, batch, seqLen);
    ref.synchronize();

    const refInput = refWs.alloc([batch, nHeads, seqLen, ropeDim], "BF16");
    refInput.h2d(f32ToBf16Bytes(inputF32));
    ref.synchronize();

    const refOut = refInput.applyRotaryPosEmb(refCos, refSin, ropeDim, ropeDim, nHeads, seqLen, batch, 1);
    ref.synchronize();
    const refBuf = Buffer.alloc(batch * nHeads * seqLen * ropeDim * 2);
    refOut.d2h(refBuf);
    const refF32 = bf16BytesToF32(refBuf);

    const pInvFreq = ws.alloc([dimHalf], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pInvFreq.h2d(f32ToBf16Bytes(invFreqF32));
    const pPosIds = ws.alloc([batch * seqLen], "I32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pPosIds.h2d(positionIdsBuf);
    po.synchronize();

    const { cos: pCos, sin: pSin } = pInvFreq.rotaryEmbedding(pPosIds, batch, seqLen);
    po.synchronize();

    const pInput = ws.alloc([batch * nHeads, seqLen, ropeDim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const shardHeads = nHeads / 2;
    const rowLen = seqLen * ropeDim;
    const shard0F32 = new Float32Array(batch * shardHeads * rowLen);
    const shard1F32 = new Float32Array(batch * shardHeads * rowLen);
    for (let b = 0; b < batch; b++) {
      for (let h = 0; h < shardHeads; h++) {
        const srcOff0 = ((b * nHeads + h) * rowLen);
        const srcOff1 = ((b * nHeads + shardHeads + h) * rowLen);
        const dstOff = (b * shardHeads + h) * rowLen;
        for (let j = 0; j < rowLen; j++) {
          shard0F32[dstOff + j] = inputF32[srcOff0 + j];
          shard1F32[dstOff + j] = inputF32[srcOff1 + j];
        }
      }
    }
    pInput.shard(0).h2d(f32ToBf16Bytes(shard0F32));
    pInput.shard(1).h2d(f32ToBf16Bytes(shard1F32));
    po.synchronize();

    const pOut = pInput.applyRotaryPosEmb(pCos, pSin, ropeDim, ropeDim, nHeads, seqLen, batch, 1) as ParallelTensor;
    po.synchronize();

    assert.equal(pOut.parallelism, TensorParallelism.Row);

    const shardSize = batch * shardHeads * seqLen * ropeDim;
    const out0Buf = Buffer.alloc(shardSize * 2);
    const out1Buf = Buffer.alloc(shardSize * 2);
    pOut.shard(0).d2h(out0Buf);
    pOut.shard(1).d2h(out1Buf);
    const out0F32 = bf16BytesToF32(out0Buf);
    const out1F32 = bf16BytesToF32(out1Buf);

    const gatheredF32 = new Float32Array(batch * nHeads * seqLen * ropeDim);
    for (let b = 0; b < batch; b++) {
      for (let h = 0; h < nHeads; h++) {
        const shardIdx = h < shardHeads ? 0 : 1;
        const hInShard = h < shardHeads ? h : h - shardHeads;
        const src = shardIdx === 0 ? out0F32 : out1F32;
        for (let s = 0; s < seqLen; s++) {
          for (let d = 0; d < ropeDim; d++) {
            const fullIdx = (b * nHeads + h) * seqLen * ropeDim + s * ropeDim + d;
            const shardIdx2 = (b * shardHeads + hInShard) * seqLen * ropeDim + s * ropeDim + d;
            gatheredF32[fullIdx] = src[shardIdx2];
          }
        }
      }
    }

    for (let i = 0; i < refF32.length; i++) {
      const relErr = Math.abs(gatheredF32[i] - refF32[i]) / Math.max(Math.abs(refF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${refF32[i]}, got ${gatheredF32[i]} (relErr=${relErr})`);
    }

    refOut[Symbol.dispose]();
    refInput[Symbol.dispose]();
    refCos[Symbol.dispose]();
    refSin[Symbol.dispose]();
    refInvFreq[Symbol.dispose]();
    refPosIds[Symbol.dispose]();
    pOut[Symbol.dispose]();
    pInput[Symbol.dispose]();
    pCos[Symbol.dispose]();
    pSin[Symbol.dispose]();
    pInvFreq[Symbol.dispose]();
    pPosIds[Symbol.dispose]();
  });

  it("Replicated applyRotaryPosEmb (nHeads=1) matches single-GPU", () => {
    const batch = 2;
    const nHeads = 1;
    const seqLen = 4;
    const ropeDim = 64;
    const dimHalf = ropeDim / 2;

    const inputF32 = new Float32Array(batch * nHeads * seqLen * ropeDim);
    for (let i = 0; i < inputF32.length; i++) inputF32[i] = (i % 19 - 9) * 0.05;

    const positionIdsF32 = new Int32Array(batch * seqLen);
    for (let b = 0; b < batch; b++) {
      for (let s = 0; s < seqLen; s++) positionIdsF32[b * seqLen + s] = s;
    }
    const positionIdsBuf = Buffer.alloc(batch * seqLen * 4);
    for (let i = 0; i < batch * seqLen; i++) positionIdsBuf.writeInt32LE(positionIdsF32[i], i * 4);

    const invFreqF32 = new Float32Array(dimHalf);
    for (let i = 0; i < dimHalf; i++) invFreqF32[i] = 1.0 / (10000.0 ** (2.0 * i / ropeDim));

    const refInvFreq = refWs.alloc([dimHalf], "BF16");
    refInvFreq.h2d(f32ToBf16Bytes(invFreqF32));
    const refPosIds = refWs.alloc([batch * seqLen], "I32");
    refPosIds.h2d(positionIdsBuf);
    ref.synchronize();

    const { cos: refCos, sin: refSin } = refInvFreq.rotaryEmbedding(refPosIds, batch, seqLen);
    ref.synchronize();

    const refInput = refWs.alloc([batch, nHeads, seqLen, ropeDim], "BF16");
    refInput.h2d(f32ToBf16Bytes(inputF32));
    ref.synchronize();

    const refOut = refInput.applyRotaryPosEmb(refCos, refSin, ropeDim, ropeDim, nHeads, seqLen, batch, 1);
    ref.synchronize();
    const refBuf = Buffer.alloc(batch * nHeads * seqLen * ropeDim * 2);
    refOut.d2h(refBuf);
    const refF32 = bf16BytesToF32(refBuf);

    const pInvFreq = ws.alloc([dimHalf], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pInvFreq.h2d(f32ToBf16Bytes(invFreqF32));
    const pPosIds = ws.alloc([batch * seqLen], "I32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pPosIds.h2d(positionIdsBuf);
    po.synchronize();

    const { cos: pCos, sin: pSin } = pInvFreq.rotaryEmbedding(pPosIds, batch, seqLen);
    po.synchronize();

    const pInput = ws.alloc([batch, nHeads, seqLen, ropeDim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pInput.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const pOut = pInput.applyRotaryPosEmb(pCos, pSin, ropeDim, ropeDim, nHeads, seqLen, batch, 1) as ParallelTensor;
    po.synchronize();

    assert.equal(pOut.parallelism, TensorParallelism.Replicated);

    const outBuf = Buffer.alloc(batch * nHeads * seqLen * ropeDim * 2);
    pOut.d2h(outBuf);
    const outF32 = bf16BytesToF32(outBuf);

    for (let i = 0; i < refF32.length; i++) {
      const relErr = Math.abs(outF32[i] - refF32[i]) / Math.max(Math.abs(refF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${refF32[i]}, got ${outF32[i]} (relErr=${relErr})`);
    }

    refOut[Symbol.dispose]();
    refInput[Symbol.dispose]();
    refCos[Symbol.dispose]();
    refSin[Symbol.dispose]();
    refInvFreq[Symbol.dispose]();
    refPosIds[Symbol.dispose]();
    pOut[Symbol.dispose]();
    pInput[Symbol.dispose]();
    pCos[Symbol.dispose]();
    pSin[Symbol.dispose]();
    pInvFreq[Symbol.dispose]();
    pPosIds[Symbol.dispose]();
  });
});

describe("ParallelOps.mlaVExpand", () => {
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

  function refMlaVExpand(
    attnOut: Float32Array, vProj: Float32Array,
    kvLoraRank: number, vHeadDim: number, nHeads: number,
    seqLen: number, batch: number,
  ): Float32Array {
    const BS = batch * seqLen;
    const result = new Float32Array(BS * nHeads * vHeadDim);
    for (let b = 0; b < batch; b++) {
      for (let s = 0; s < seqLen; s++) {
        for (let h = 0; h < nHeads; h++) {
          for (let j = 0; j < vHeadDim; j++) {
            let sum = 0;
            for (let k = 0; k < kvLoraRank; k++) {
              const a = attnOut[(b * nHeads + h) * seqLen * kvLoraRank + s * kvLoraRank + k];
              const w = vProj[(h * kvLoraRank + k) * vHeadDim + j];
              sum += a * w;
            }
            result[(b * seqLen + s) * nHeads * vHeadDim + h * vHeadDim + j] = sum;
          }
        }
      }
    }
    return result;
  }

  it("Row-parallel mlaVExpand matches single-GPU", () => {
    const batch = 1;
    const seqLen = 2;
    const nHeads = 4;
    const kvLoraRank = 8;
    const vHeadDim = 6;

    const attnOutF32 = new Float32Array(batch * nHeads * seqLen * kvLoraRank);
    for (let i = 0; i < attnOutF32.length; i++) attnOutF32[i] = (i % 7 - 3) * 0.1;

    const vProjF32 = new Float32Array(nHeads * vHeadDim * kvLoraRank);
    for (let i = 0; i < vProjF32.length; i++) vProjF32[i] = (i % 11 - 5) * 0.1;

    const expected = refMlaVExpand(attnOutF32, vProjF32, kvLoraRank, vHeadDim, nHeads, seqLen, batch);

    const refAttnOut = refWs.alloc([batch, nHeads, seqLen, kvLoraRank], "BF16");
    const refVProj = refWs.alloc([nHeads * kvLoraRank, vHeadDim], "BF16");
    refAttnOut.h2d(f32ToBf16Bytes(attnOutF32));
    refVProj.h2d(f32ToBf16Bytes(vProjF32));
    ref.synchronize();

    const refOut = refAttnOut.mlaVExpand(refVProj, seqLen, batch);
    ref.synchronize();
    const refBuf = Buffer.alloc(batch * seqLen * nHeads * vHeadDim * 2);
    refOut.d2h(refBuf);
    const refF32 = bf16BytesToF32(refBuf);

    for (let i = 0; i < expected.length; i++) {
      const relErr = Math.abs(refF32[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.1, `ref i=${i}: expected ${expected[i]}, got ${refF32[i]} (relErr=${relErr})`);
    }

    const pAttnOut = ws.alloc([batch, nHeads, seqLen, kvLoraRank], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const shardHeads = nHeads / 2;

    const attnShard0 = new Float32Array(batch * shardHeads * seqLen * kvLoraRank);
    const attnShard1 = new Float32Array(batch * shardHeads * seqLen * kvLoraRank);
    for (let b = 0; b < batch; b++) {
      for (let h = 0; h < shardHeads; h++) {
        for (let s = 0; s < seqLen; s++) {
          for (let k = 0; k < kvLoraRank; k++) {
            const srcIdx0 = (b * nHeads + h) * seqLen * kvLoraRank + s * kvLoraRank + k;
            const dstIdx = (b * shardHeads + h) * seqLen * kvLoraRank + s * kvLoraRank + k;
            attnShard0[dstIdx] = attnOutF32[srcIdx0];
            attnShard1[dstIdx] = attnOutF32[(b * nHeads + shardHeads + h) * seqLen * kvLoraRank + s * kvLoraRank + k];
          }
        }
      }
    }


    pAttnOut.shard(0).h2d(f32ToBf16Bytes(attnShard0));
    pAttnOut.shard(1).h2d(f32ToBf16Bytes(attnShard1));

    const pVProj = ws.alloc([nHeads * kvLoraRank, vHeadDim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pVProj.shard(0).h2d(f32ToBf16Bytes(vProjF32));
    pVProj.shard(1).h2d(f32ToBf16Bytes(vProjF32));
    po.synchronize();

    const pOut = pAttnOut.mlaVExpand(pVProj, seqLen, batch) as ParallelTensor;
    po.synchronize();

    assert.equal(pOut.parallelism, TensorParallelism.Row);

    const BS = batch * seqLen;
    const shardOutSize = BS * shardHeads * vHeadDim;
    const out0Buf = Buffer.alloc(shardOutSize * 2);
    const out1Buf = Buffer.alloc(shardOutSize * 2);
    pOut.shard(0).d2h(out0Buf);
    pOut.shard(1).d2h(out1Buf);
    const out0F32 = bf16BytesToF32(out0Buf);
    const out1F32 = bf16BytesToF32(out1Buf);

    const gatheredF32 = new Float32Array(BS * nHeads * vHeadDim);
    for (let bs = 0; bs < BS; bs++) {
      for (let h = 0; h < shardHeads; h++) {
        for (let j = 0; j < vHeadDim; j++) {
          gatheredF32[bs * nHeads * vHeadDim + h * vHeadDim + j] = out0F32[bs * shardHeads * vHeadDim + h * vHeadDim + j];
          gatheredF32[bs * nHeads * vHeadDim + (shardHeads + h) * vHeadDim + j] = out1F32[bs * shardHeads * vHeadDim + h * vHeadDim + j];
        }
      }
    }

    for (let i = 0; i < expected.length; i++) {
      const relErr = Math.abs(gatheredF32[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.15, `i=${i}: expected ${expected[i]}, got ${gatheredF32[i]} (relErr=${relErr})`);
    }

    refOut[Symbol.dispose]();
    refAttnOut[Symbol.dispose]();
    refVProj[Symbol.dispose]();
    pOut[Symbol.dispose]();
    pAttnOut[Symbol.dispose]();
    pVProj[Symbol.dispose]();
  });
});

describe("ParallelOps.bmm", () => {
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

  it("Column-parallel bmm with correct batch adjustment", () => {
    const batch = 4;
    const M = 1;
    const N = 16;
    const K = 8;

    const aF32 = new Float32Array(batch * M * K);
    const bF32 = new Float32Array(batch * K * N);
    for (let i = 0; i < aF32.length; i++) aF32[i] = (i % 7 - 3) * 0.1;
    for (let i = 0; i < bF32.length; i++) bF32[i] = (i % 11 - 5) * 0.1;

    const refA = refWs.alloc([batch * M, K], "BF16");
    const refB = refWs.alloc([batch * K, N], "BF16");
    refA.h2d(f32ToBf16Bytes(aF32));
    refB.h2d(f32ToBf16Bytes(bF32));
    ref.synchronize();

    const refOut = refA.bmm(refB, batch, M, N, K);
    ref.synchronize();
    const refBuf = Buffer.alloc(batch * M * N * 2);
    refOut.d2h(refBuf);
    const refF32 = bf16BytesToF32(refBuf);

    const shardBatch = batch / 2;
    const pA = ws.alloc([batch * M, K], "BF16", undefined, TensorParallelism.Column) as ParallelTensor;
    const pB = ws.alloc([batch * K, N], "BF16", undefined, TensorParallelism.Column) as ParallelTensor;

    const aShard0 = new Float32Array(shardBatch * M * K);
    const aShard1 = new Float32Array(shardBatch * M * K);
    const bShard0 = new Float32Array(shardBatch * K * N);
    const bShard1 = new Float32Array(shardBatch * K * N);
    for (let i = 0; i < shardBatch * M * K; i++) {
      aShard0[i] = aF32[i];
      aShard1[i] = aF32[shardBatch * M * K + i];
    }
    for (let i = 0; i < shardBatch * K * N; i++) {
      bShard0[i] = bF32[i];
      bShard1[i] = bF32[shardBatch * K * N + i];
    }

    pA.shard(0).h2d(f32ToBf16Bytes(aShard0));
    pA.shard(1).h2d(f32ToBf16Bytes(aShard1));
    pB.shard(0).h2d(f32ToBf16Bytes(bShard0));
    pB.shard(1).h2d(f32ToBf16Bytes(bShard1));
    po.synchronize();

    const pOut = pA.bmm(pB, batch, M, N, K) as ParallelTensor;
    po.synchronize();

    const out0Buf = Buffer.alloc(shardBatch * M * N * 2);
    const out1Buf = Buffer.alloc(shardBatch * M * N * 2);
    pOut.shard(0).d2h(out0Buf);
    pOut.shard(1).d2h(out1Buf);
    const out0F32 = bf16BytesToF32(out0Buf);
    const out1F32 = bf16BytesToF32(out1Buf);

    for (let b = 0; b < shardBatch; b++) {
      for (let j = 0; j < N; j++) {
        const refVal0 = refF32[b * N + j];
        const outVal0 = out0F32[b * N + j];
        const relErr0 = Math.abs(outVal0 - refVal0) / Math.max(Math.abs(refVal0), 1e-6);
        assert.ok(relErr0 < 0.05, `shard0 b=${b}, j=${j}: expected ${refVal0}, got ${outVal0}`);

        const refVal1 = refF32[(shardBatch + b) * N + j];
        const outVal1 = out1F32[b * N + j];
        const relErr1 = Math.abs(outVal1 - refVal1) / Math.max(Math.abs(refVal1), 1e-6);
        assert.ok(relErr1 < 0.05, `shard1 b=${b}, j=${j}: expected ${refVal1}, got ${outVal1}`);
      }
    }

    refOut[Symbol.dispose]();
    refA[Symbol.dispose]();
    refB[Symbol.dispose]();
    pOut[Symbol.dispose]();
    pA[Symbol.dispose]();
    pB[Symbol.dispose]();
  });
});

describe("ParallelOps.sigmoid", () => {
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

  it("Row-parallel sigmoid matches single-GPU", () => {
    const rows = 2;
    const cols = 8;
    const inputF32 = new Float32Array(rows * cols);
    for (let i = 0; i < rows * cols; i++) inputF32[i] = (i % 11 - 5) * 0.5;

    const refInput = refWs.alloc([rows, cols], "BF16");
    refInput.h2d(f32ToBf16Bytes(inputF32));
    ref.synchronize();

    const refOut = refInput.sigmoid();
    ref.synchronize();
    const refBuf = Buffer.alloc(rows * cols * 2);
    refOut.d2h(refBuf);
    const refF32 = bf16BytesToF32(refBuf);

    const pInput = ws.alloc([rows, cols], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    pInput.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const pOut = pInput.sigmoid() as ParallelTensor;
    po.synchronize();

    const outBuf = Buffer.alloc(rows * cols * 2);
    pOut.d2h(outBuf);
    const outF32 = bf16BytesToF32(outBuf);

    for (let i = 0; i < rows * cols; i++) {
      const relErr = Math.abs(outF32[i] - refF32[i]) / Math.max(Math.abs(refF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${refF32[i]}, got ${outF32[i]} (relErr=${relErr})`);
    }

    refOut[Symbol.dispose]();
    refInput[Symbol.dispose]();
    pOut[Symbol.dispose]();
    pInput[Symbol.dispose]();
  });
});

describe("ParallelOps.topk", () => {
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

  it("Replicated topk matches single-GPU", () => {
    const batch = 2;
    const dim = 16;
    const k = 4;

    const inputF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = Math.sin(i * 1.7) * 5;

    const refInput = refWs.alloc([batch, dim], "BF16");
    refInput.h2d(f32ToBf16Bytes(inputF32));
    ref.synchronize();

    const { values: refValues, indices: refIndices } = refInput.topk(k, dim);
    ref.synchronize();
    const refValuesBuf = Buffer.alloc(batch * k * 2);
    const refIndicesBuf = Buffer.alloc(batch * k * 4);
    refValues.d2h(refValuesBuf);
    refIndices.d2h(refIndicesBuf);
    const refValuesF32 = bf16BytesToF32(refValuesBuf);

    const pInput = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pInput.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const { values: pValues, indices: pIndices } = pInput.topk(k, dim);
    po.synchronize();

    assert.equal((pValues as ParallelTensor).parallelism, TensorParallelism.Replicated);
    assert.equal((pIndices as ParallelTensor).parallelism, TensorParallelism.Replicated);

    const pValuesBuf = Buffer.alloc(batch * k * 2);
    const pIndicesBuf = Buffer.alloc(batch * k * 4);
    pValues.d2h(pValuesBuf);
    pIndices.d2h(pIndicesBuf);
    const pValuesF32 = bf16BytesToF32(pValuesBuf);

    for (let b = 0; b < batch; b++) {
      for (let j = 0; j < k; j++) {
        const idx = b * k + j;
        const relErr = Math.abs(pValuesF32[idx] - refValuesF32[idx]) / Math.max(Math.abs(refValuesF32[idx]), 1e-6);
        assert.ok(relErr < 0.05, `batch=${b}, k=${j}: value expected ${refValuesF32[idx]}, got ${pValuesF32[idx]}`);
        assert.equal(pIndicesBuf.readInt32LE(idx * 4), refIndicesBuf.readInt32LE(idx * 4),
          `batch=${b}, k=${j}: index expected ${refIndicesBuf.readInt32LE(idx * 4)}, got ${pIndicesBuf.readInt32LE(idx * 4)}`);
      }
    }

    refValues[Symbol.dispose]();
    refIndices[Symbol.dispose]();
    refInput[Symbol.dispose]();
    pValues[Symbol.dispose]();
    pIndices[Symbol.dispose]();
    pInput[Symbol.dispose]();
  });
});

describe("ParallelOps.reduceSum", () => {
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

  it("Replicated reduceSum matches single-GPU", () => {
    const batch = 3;
    const dim = 8;

    const inputF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = (i % 13 - 6) * 0.1;

    const refInput = refWs.alloc([batch, dim], "BF16");
    refInput.h2d(f32ToBf16Bytes(inputF32));
    ref.synchronize();

    const refOut = refInput.reduceSum();
    ref.synchronize();
    const refBuf = Buffer.alloc(batch * 2);
    refOut.d2h(refBuf);
    const refF32 = bf16BytesToF32(refBuf);

    const pInput = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pInput.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const pOut = pInput.reduceSum() as ParallelTensor;
    po.synchronize();

    assert.equal(pOut.parallelism, TensorParallelism.Replicated);

    const outBuf = Buffer.alloc(batch * 2);
    pOut.d2h(outBuf);
    const outF32 = bf16BytesToF32(outBuf);

    for (let b = 0; b < batch; b++) {
      const relErr = Math.abs(outF32[b] - refF32[b]) / Math.max(Math.abs(refF32[b]), 1e-6);
      assert.ok(relErr < 0.05, `batch=${b}: expected ${refF32[b]}, got ${outF32[b]} (relErr=${relErr})`);
    }

    refOut[Symbol.dispose]();
    refInput[Symbol.dispose]();
    pOut[Symbol.dispose]();
    pInput[Symbol.dispose]();
  });

  it("Row-parallel reduceSum produces PartialSum and AllReduce matches single-GPU", () => {
    const batch = 4;
    const dim = 768;

    const inputF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = (i % 13 - 6) * 0.1;

    const refInput = refWs.alloc([batch, dim], "BF16");
    refInput.h2d(f32ToBf16Bytes(inputF32));
    ref.synchronize();

    const refOut = refInput.reduceSum();
    ref.synchronize();
    const refBuf = Buffer.alloc(batch * 2);
    refOut.d2h(refBuf);
    const refF32 = bf16BytesToF32(refBuf);

    const pInput = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    pInput.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const pOut = pInput.reduceSum() as ParallelTensor;
    po.synchronize();

    assert.equal(pOut.parallelism, TensorParallelism.PartialSum);

    pOut.allReduce();
    po.synchronize();

    const outBuf = Buffer.alloc(batch * 2);
    pOut.d2h(outBuf);
    const outF32 = bf16BytesToF32(outBuf);

    for (let b = 0; b < batch; b++) {
      const relErr = Math.abs(outF32[b] - refF32[b]) / Math.max(Math.abs(refF32[b]), 1e-6);
      assert.ok(relErr < 0.05, `batch=${b}: expected ${refF32[b]}, got ${outF32[b]} (relErr=${relErr})`);
    }

    refOut[Symbol.dispose]();
    refInput[Symbol.dispose]();
    pOut[Symbol.dispose]();
    pInput[Symbol.dispose]();
  });
});
