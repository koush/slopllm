import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import {
  GlmOps,
  f32ToBf16Bytes,
  bf16BytesToF32,
  BF16,
  MEMCPY_H2D,
  MEMCPY_D2D,
  MEMCPY_D2H,
  NCCL_UNIQUE_ID_BYTES,
  NCCL_BFLOAT16,
  NCCL_FLOAT32,
  NCCL_SUM,
} from "../src/glm_ops";

function bf16RoundTrip(values: Float32Array): { gpuPtr: number; count: number } {
  throw new Error("Use glm.alloc/h2d instead");
}

describe("memcpy2d", () => {
  let glm: GlmOps;

  before(() => {
    process.env.CUDA_VISIBLE_DEVICES = process.env.GLM_GPU ?? "0";
    glm = new GlmOps(0);
  });

  after(() => {
    glm.free();
  });

  it("extracts a column shard from a D2D 2D copy (BF16)", () => {
    const rows = 4;
    const cols = 8;
    const elemBytes = 2;
    const totalElems = rows * cols;
    const totalBytes = totalElems * elemBytes;

    const srcF32 = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) srcF32[i] = i;
    const srcBuf = f32ToBf16Bytes(srcF32);

    const srcGpu = glm.alloc(totalBytes);
    const dstGpu = glm.alloc(totalBytes);
    glm.h2d(srcGpu, srcBuf);

    const shardCols = 4;
    const shardOffset = 2;
    const width = shardCols * elemBytes;
    const spitch = cols * elemBytes;
    const dpitch = shardCols * elemBytes;

    glm.memcpy2d(dstGpu, dpitch, srcGpu + shardOffset * elemBytes, spitch, width, rows, MEMCPY_D2D);

    const dstBuf = Buffer.alloc(rows * shardCols * elemBytes);
    glm.d2h(dstBuf, dstGpu, rows * shardCols * elemBytes);
    glm.synchronize();

    const dstF32 = bf16BytesToF32(dstBuf);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < shardCols; c++) {
        const srcCol = shardOffset + c;
        const expected = srcF32[r * cols + srcCol];
        const actual = dstF32[r * shardCols + c];
        assert.ok(
          Math.abs(actual - expected) < 0.01,
          `row=${r} col=${c}: expected ${expected}, got ${actual}`
        );
      }
    }

    glm.freeBuf(srcGpu);
    glm.freeBuf(dstGpu);
  });

  it("copies full rows via 2D D2D copy (BF16)", () => {
    const rows = 3;
    const cols = 4;
    const elemBytes = 2;
    const totalBytes = rows * cols * elemBytes;

    const srcF32 = new Float32Array(rows * cols);
    for (let i = 0; i < rows * cols; i++) srcF32[i] = i * 1.5;
    const srcBuf = f32ToBf16Bytes(srcF32);

    const srcGpu = glm.alloc(totalBytes);
    const dstGpu = glm.alloc(totalBytes);

    glm.h2d(srcGpu, srcBuf);
    const pitch = cols * elemBytes;
    glm.memcpy2d(dstGpu, pitch, srcGpu, pitch, pitch, rows, MEMCPY_D2D);

    const dstBuf = Buffer.alloc(totalBytes);
    glm.d2h(dstBuf, dstGpu, totalBytes);
    glm.synchronize();

    const dstF32 = bf16BytesToF32(dstBuf);
    for (let i = 0; i < rows * cols; i++) {
      assert.ok(
        Math.abs(dstF32[i] - srcF32[i]) < 0.01,
        `idx=${i}: expected ${srcF32[i]}, got ${dstF32[i]}`
      );
    }

    glm.freeBuf(srcGpu);
    glm.freeBuf(dstGpu);
  });

  it("H2D 2D copy loads a column shard from host to device", () => {
    const rows = 4;
    const cols = 6;
    const elemBytes = 2;
    const shardCols = 3;
    const shardOffset = 1;

    const srcF32 = new Float32Array(rows * cols);
    for (let i = 0; i < rows * cols; i++) srcF32[i] = i + 0.5;

    const hostBuf = f32ToBf16Bytes(srcF32);
    const hostPtr = glm.allocPinned(hostBuf.length);
    glm.writePinned(hostPtr, hostBuf);

    const dstGpu = glm.alloc(rows * shardCols * elemBytes);
    const dpitch = shardCols * elemBytes;
    const spitch = cols * elemBytes;
    const width = shardCols * elemBytes;

    glm.memcpy2d(
      dstGpu, dpitch,
      hostPtr + shardOffset * elemBytes, spitch,
      width, rows, MEMCPY_H2D
    );

    const dstBuf = Buffer.alloc(rows * shardCols * elemBytes);
    glm.d2h(dstBuf, dstGpu, rows * shardCols * elemBytes);
    glm.synchronize();

    const dstF32 = bf16BytesToF32(dstBuf);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < shardCols; c++) {
        const srcCol = shardOffset + c;
        const expected = srcF32[r * cols + srcCol];
        const actual = dstF32[r * shardCols + c];
        assert.ok(
          Math.abs(actual - expected) < 0.01,
          `row=${r} col=${c}: expected ${expected}, got ${actual}`
        );
      }
    }

    glm.freeBuf(dstGpu);
    glm.freePinned(hostPtr);
  });
});

describe("NCCL single-rank", () => {
  let glm: GlmOps;

  before(() => {
    process.env.CUDA_VISIBLE_DEVICES = process.env.GLM_GPU ?? "0";
    glm = new GlmOps(0);
  });

  after(() => {
    glm.free();
  });

  it("generates a 128-byte unique ID", () => {
    const idBuf = Buffer.alloc(NCCL_UNIQUE_ID_BYTES);
    glm.native.ncclUniqueId(idBuf);
    const nonZero = idBuf.some(b => b !== 0);
    assert.ok(nonZero, "unique ID should not be all zeros");
  });

  it("initializes and destroys a single-rank communicator", () => {
    const idBuf = Buffer.alloc(NCCL_UNIQUE_ID_BYTES);
    glm.native.ncclUniqueId(idBuf);
    const idPtr = glm.allocPinned(NCCL_UNIQUE_ID_BYTES);
    glm.writePinned(idPtr, idBuf);

    const comm = glm.native.ncclCommInitRank(0, 1, idPtr);
    assert.ok(comm !== 0, "comm should not be null");

    glm.native.ncclCommDestroy(comm);
    glm.freePinned(idPtr);
  });

  it("allReduce with worldSize=1 is identity (BF16)", () => {
    const idBuf = Buffer.alloc(NCCL_UNIQUE_ID_BYTES);
    glm.native.ncclUniqueId(idBuf);
    const idPtr = glm.allocPinned(NCCL_UNIQUE_ID_BYTES);
    glm.writePinned(idPtr, idBuf);

    const comm = glm.native.ncclCommInitRank(0, 1, idPtr);

    const count = 16;
    const bytes = count * 2;
    const srcF32 = new Float32Array(count);
    for (let i = 0; i < count; i++) srcF32[i] = (i + 1) * 0.5;
    const srcBuf = f32ToBf16Bytes(srcF32);

    const sendGpu = glm.alloc(bytes);
    const recvGpu = glm.alloc(bytes);
    glm.h2d(sendGpu, srcBuf);

    glm.native.ncclAllReduce(comm, glm.ctx, sendGpu, recvGpu, count, NCCL_BFLOAT16, NCCL_SUM);
    glm.synchronize();

    const dstBuf = Buffer.alloc(bytes);
    glm.d2h(dstBuf, recvGpu, bytes);

    const dstF32 = bf16BytesToF32(dstBuf);
    for (let i = 0; i < count; i++) {
      assert.ok(
        Math.abs(dstF32[i] - srcF32[i]) < 0.01,
        `idx=${i}: expected ${srcF32[i]}, got ${dstF32[i]}`
      );
    }

    glm.native.ncclCommDestroy(comm);
    glm.freeBuf(sendGpu);
    glm.freeBuf(recvGpu);
    glm.freePinned(idPtr);
  });

  it("allReduce with worldSize=1 is identity (F32)", () => {
    const idBuf = Buffer.alloc(NCCL_UNIQUE_ID_BYTES);
    glm.native.ncclUniqueId(idBuf);
    const idPtr = glm.allocPinned(NCCL_UNIQUE_ID_BYTES);
    glm.writePinned(idPtr, idBuf);

    const comm = glm.native.ncclCommInitRank(0, 1, idPtr);

    const count = 16;
    const bytes = count * 4;
    const srcF32 = new Float32Array(count);
    for (let i = 0; i < count; i++) srcF32[i] = (i + 1) * 0.5;
    const srcBuf = Buffer.from(srcF32.buffer);

    const sendGpu = glm.alloc(bytes);
    const recvGpu = glm.alloc(bytes);
    glm.h2d(sendGpu, srcBuf);

    glm.native.ncclAllReduce(comm, glm.ctx, sendGpu, recvGpu, count, NCCL_FLOAT32, NCCL_SUM);
    glm.synchronize();

    const dstBuf = Buffer.alloc(bytes);
    glm.d2h(dstBuf, recvGpu, bytes);

    const dstF32 = new Float32Array(dstBuf.buffer, dstBuf.byteOffset, count);
    for (let i = 0; i < count; i++) {
      assert.ok(
        Math.abs(dstF32[i] - srcF32[i]) < 1e-6,
        `idx=${i}: expected ${srcF32[i]}, got ${dstF32[i]}`
      );
    }

    glm.native.ncclCommDestroy(comm);
    glm.freeBuf(sendGpu);
    glm.freeBuf(recvGpu);
    glm.freePinned(idPtr);
  });

  it("allGather with worldSize=1 is identity (BF16)", () => {
    const idBuf = Buffer.alloc(NCCL_UNIQUE_ID_BYTES);
    glm.native.ncclUniqueId(idBuf);
    const idPtr = glm.allocPinned(NCCL_UNIQUE_ID_BYTES);
    glm.writePinned(idPtr, idBuf);

    const comm = glm.native.ncclCommInitRank(0, 1, idPtr);

    const count = 16;
    const bytes = count * 2;
    const srcF32 = new Float32Array(count);
    for (let i = 0; i < count; i++) srcF32[i] = (i + 1) * 0.5;
    const srcBuf = f32ToBf16Bytes(srcF32);

    const sendGpu = glm.alloc(bytes);
    const recvGpu = glm.alloc(bytes);
    glm.h2d(sendGpu, srcBuf);

    glm.native.ncclAllGather(comm, glm.ctx, sendGpu, recvGpu, count, NCCL_BFLOAT16);
    glm.synchronize();

    const dstBuf = Buffer.alloc(bytes);
    glm.d2h(dstBuf, recvGpu, bytes);

    const dstF32 = bf16BytesToF32(dstBuf);
    for (let i = 0; i < count; i++) {
      assert.ok(
        Math.abs(dstF32[i] - srcF32[i]) < 0.01,
        `idx=${i}: expected ${srcF32[i]}, got ${dstF32[i]}`
      );
    }

    glm.native.ncclCommDestroy(comm);
    glm.freeBuf(sendGpu);
    glm.freeBuf(recvGpu);
    glm.freePinned(idPtr);
  });
});
