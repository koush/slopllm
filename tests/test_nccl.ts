import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import {
  GlmOps,
  GlmTensor,
  getNativeAddon,
  f32ToBf16Bytes,
  bf16BytesToF32,
  NCCL_UNIQUE_ID_BYTES,
  NCCL_BFLOAT16,
  NCCL_FLOAT32,
  NCCL_SUM,
} from "../src/glm_ops";
import { MemcpyKind } from "../src/tensor";
import { WorkspaceBase } from "../src/workspace";
import { Tensor } from "../src/tensor";

describe("memcpy2d", () => {
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

  it("extracts a column shard from a D2D 2D copy (BF16)", () => {
    const rows = 4;
    const cols = 8;
    const elemBytes = 2;
    const totalElems = rows * cols;
    const totalBytes = totalElems * elemBytes;

    const srcF32 = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) srcF32[i] = i;
    const srcBuf = f32ToBf16Bytes(srcF32);

    const srcGpu = ws.alloc([totalBytes], "U8");
    const dstGpu = ws.alloc([totalBytes], "U8");
    srcGpu.h2d(srcBuf);

    const shardCols = 4;
    const shardOffset = 2;
    const width = shardCols * elemBytes;
    const spitch = cols * elemBytes;
    const dpitch = shardCols * elemBytes;

    glm.memcpy2d(dstGpu.data, dpitch, srcGpu.data + shardOffset * elemBytes, spitch, width, rows, MemcpyKind.DeviceToDevice);

    const dstBuf = Buffer.alloc(rows * shardCols * elemBytes);
    dstGpu.d2h(dstBuf, rows * shardCols * elemBytes);
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
  });

  it("copies full rows via 2D D2D copy (BF16)", () => {
    const rows = 3;
    const cols = 4;
    const elemBytes = 2;
    const totalBytes = rows * cols * elemBytes;

    const srcF32 = new Float32Array(rows * cols);
    for (let i = 0; i < rows * cols; i++) srcF32[i] = i * 1.5;
    const srcBuf = f32ToBf16Bytes(srcF32);

    const srcGpu = ws.alloc([totalBytes], "U8");
    const dstGpu = ws.alloc([totalBytes], "U8");

    srcGpu.h2d(srcBuf);
    const pitch = cols * elemBytes;
    glm.memcpy2d(dstGpu.data, pitch, srcGpu.data, pitch, pitch, rows, MemcpyKind.DeviceToDevice);

    const dstBuf = Buffer.alloc(totalBytes);
    dstGpu.d2h(dstBuf, totalBytes);
    glm.synchronize();

    const dstF32 = bf16BytesToF32(dstBuf);
    for (let i = 0; i < rows * cols; i++) {
      assert.ok(
        Math.abs(dstF32[i] - srcF32[i]) < 0.01,
        `idx=${i}: expected ${srcF32[i]}, got ${dstF32[i]}`
      );
    }
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
    const hostTensor = new GlmTensor(ws, glm, hostPtr, hostBuf.length, [hostBuf.length], "U8", undefined, true, undefined);
    glm.writePinned(hostTensor, hostBuf);

    const dstGpu = ws.alloc([rows * shardCols * elemBytes], "U8");
    const dpitch = shardCols * elemBytes;
    const spitch = cols * elemBytes;
    const width = shardCols * elemBytes;

    glm.memcpy2d(
      dstGpu.data, dpitch,
      hostPtr + shardOffset * elemBytes, spitch,
      width, rows, MemcpyKind.HostToDevice
    );

    const dstBuf = Buffer.alloc(rows * shardCols * elemBytes);
    dstGpu.d2h(dstBuf, rows * shardCols * elemBytes);
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

    glm.freePinned(hostTensor);
  });
});

describe("NCCL single-rank", () => {
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

  it("generates a 128-byte unique ID", () => {
    const idBuf = Buffer.alloc(NCCL_UNIQUE_ID_BYTES);
    getNativeAddon().ncclUniqueId(idBuf);
    const nonZero = idBuf.some(b => b !== 0);
    assert.ok(nonZero, "unique ID should not be all zeros");
  });

  it("initializes and destroys a single-rank communicator", () => {
    const idBuf = Buffer.alloc(NCCL_UNIQUE_ID_BYTES);
    getNativeAddon().ncclUniqueId(idBuf);
    const idPtr = glm.allocPinned(NCCL_UNIQUE_ID_BYTES);
    const idTensor = new GlmTensor(ws, glm, idPtr, NCCL_UNIQUE_ID_BYTES, [NCCL_UNIQUE_ID_BYTES], "U8", undefined, true, undefined);
    glm.writePinned(idTensor, idBuf);

    const comm = getNativeAddon().ncclCommInitRank(0, 0, 1, idPtr);
    assert.ok(comm !== 0, "comm should not be null");

    getNativeAddon().ncclCommDestroy(comm);
    glm.freePinned(idTensor);
  });

  it("allReduce with worldSize=1 is identity (BF16)", () => {
    const idBuf = Buffer.alloc(NCCL_UNIQUE_ID_BYTES);
    getNativeAddon().ncclUniqueId(idBuf);
    const idPtr = glm.allocPinned(NCCL_UNIQUE_ID_BYTES);
    const idTensor = new GlmTensor(ws, glm, idPtr, NCCL_UNIQUE_ID_BYTES, [NCCL_UNIQUE_ID_BYTES], "U8", undefined, true, undefined);
    glm.writePinned(idTensor, idBuf);

    const comm = getNativeAddon().ncclCommInitRank(0, 0, 1, idPtr);

    const count = 16;
    const bytes = count * 2;
    const srcF32 = new Float32Array(count);
    for (let i = 0; i < count; i++) srcF32[i] = (i + 1) * 0.5;
    const srcBuf = f32ToBf16Bytes(srcF32);

    const sendGpu = ws.alloc([bytes], "U8");
    const recvGpu = ws.alloc([bytes], "U8");
    sendGpu.h2d(srcBuf);

    getNativeAddon().ncclAllReduce(comm, glm.ctx, sendGpu.data, recvGpu.data, count, NCCL_BFLOAT16, NCCL_SUM);
    glm.synchronize();

    const dstBuf = Buffer.alloc(bytes);
    recvGpu.d2h(dstBuf, bytes);

    const dstF32 = bf16BytesToF32(dstBuf);
    for (let i = 0; i < count; i++) {
      assert.ok(
        Math.abs(dstF32[i] - srcF32[i]) < 0.01,
        `idx=${i}: expected ${srcF32[i]}, got ${dstF32[i]}`
      );
    }

    getNativeAddon().ncclCommDestroy(comm);
    glm.freePinned(idTensor);
  });

  it("allReduce with worldSize=1 is identity (F32)", () => {
    const idBuf = Buffer.alloc(NCCL_UNIQUE_ID_BYTES);
    getNativeAddon().ncclUniqueId(idBuf);
    const idPtr = glm.allocPinned(NCCL_UNIQUE_ID_BYTES);
    const idTensor = new GlmTensor(ws, glm, idPtr, NCCL_UNIQUE_ID_BYTES, [NCCL_UNIQUE_ID_BYTES], "U8", undefined, true, undefined);
    glm.writePinned(idTensor, idBuf);

    const comm = getNativeAddon().ncclCommInitRank(0, 0, 1, idPtr);

    const count = 16;
    const bytes = count * 4;
    const srcF32 = new Float32Array(count);
    for (let i = 0; i < count; i++) srcF32[i] = (i + 1) * 0.5;
    const srcBuf = Buffer.from(srcF32.buffer);

    const sendGpu = ws.alloc([bytes], "U8");
    const recvGpu = ws.alloc([bytes], "U8");
    sendGpu.h2d(srcBuf);

    getNativeAddon().ncclAllReduce(comm, glm.ctx, sendGpu.data, recvGpu.data, count, NCCL_FLOAT32, NCCL_SUM);
    glm.synchronize();

    const dstBuf = Buffer.alloc(bytes);
    recvGpu.d2h(dstBuf, bytes);

    const dstF32 = new Float32Array(dstBuf.buffer, dstBuf.byteOffset, count);
    for (let i = 0; i < count; i++) {
      assert.ok(
        Math.abs(dstF32[i] - srcF32[i]) < 1e-6,
        `idx=${i}: expected ${srcF32[i]}, got ${dstF32[i]}`
      );
    }

    getNativeAddon().ncclCommDestroy(comm);
    glm.freePinned(idTensor);
  });

  it("allGather with worldSize=1 is identity (BF16)", () => {
    const idBuf = Buffer.alloc(NCCL_UNIQUE_ID_BYTES);
    getNativeAddon().ncclUniqueId(idBuf);
    const idPtr = glm.allocPinned(NCCL_UNIQUE_ID_BYTES);
    const idTensor = new GlmTensor(ws, glm, idPtr, NCCL_UNIQUE_ID_BYTES, [NCCL_UNIQUE_ID_BYTES], "U8", undefined, true, undefined);
    glm.writePinned(idTensor, idBuf);

    const comm = getNativeAddon().ncclCommInitRank(0, 0, 1, idPtr);

    const count = 16;
    const bytes = count * 2;
    const srcF32 = new Float32Array(count);
    for (let i = 0; i < count; i++) srcF32[i] = (i + 1) * 0.5;
    const srcBuf = f32ToBf16Bytes(srcF32);

    const sendGpu = ws.alloc([bytes], "U8");
    const recvGpu = ws.alloc([bytes], "U8");
    sendGpu.h2d(srcBuf);

    getNativeAddon().ncclAllGather(comm, glm.ctx, sendGpu.data, recvGpu.data, count, NCCL_BFLOAT16);
    glm.synchronize();

    const dstBuf = Buffer.alloc(bytes);
    recvGpu.d2h(dstBuf, bytes);

    const dstF32 = bf16BytesToF32(dstBuf);
    for (let i = 0; i < count; i++) {
      assert.ok(
        Math.abs(dstF32[i] - srcF32[i]) < 0.01,
        `idx=${i}: expected ${srcF32[i]}, got ${dstF32[i]}`
      );
    }

    getNativeAddon().ncclCommDestroy(comm);
    glm.freePinned(idTensor);
  });
});
