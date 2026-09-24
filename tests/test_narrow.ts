import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

describe("GlmTensor.narrow (single GPU)", () => {
  let ops: GlmOps;
  let ws: WorkspaceBase;

  before(() => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    ops = new GlmOps(deviceId);
    ws = new WorkspaceBase(ops);
  });

  after(() => {
    ws.free();
    ops.free();
  });

  it("narrow 1D I32 (skip first element)", () => {
    const t = ws.alloc([5], "I32");
    const data = new Int32Array([10, 20, 30, 40, 50]);
    t.h2d(Buffer.from(data.buffer));
    ops.synchronize();

    const view = t.narrow(1, 4);
    assert.deepEqual(view.shape, [4]);

    const buf = Buffer.alloc(4 * 4);
    view.d2h(buf);
    const result = new Int32Array(buf.buffer, buf.byteOffset, 4);
    assert.deepEqual([...result], [20, 30, 40, 50]);
  });

  it("narrow 1D I32 (last element)", () => {
    const t = ws.alloc([5], "I32");
    const data = new Int32Array([10, 20, 30, 40, 50]);
    t.h2d(Buffer.from(data.buffer));
    ops.synchronize();

    const view = t.narrow(4, 1);
    assert.deepEqual(view.shape, [1]);

    const buf = Buffer.alloc(4);
    view.d2h(buf);
    assert.deepEqual(buf.readInt32LE(0), 50);
  });

  it("narrow 1D I32 (negative start)", () => {
    const t = ws.alloc([5], "I32");
    const data = new Int32Array([10, 20, 30, 40, 50]);
    t.h2d(Buffer.from(data.buffer));
    ops.synchronize();

    const view = t.narrow(-2, 2);
    assert.deepEqual(view.shape, [2]);

    const buf = Buffer.alloc(2 * 4);
    view.d2h(buf);
    const result = new Int32Array(buf.buffer, buf.byteOffset, 2);
    assert.deepEqual([...result], [40, 50]);
  });

  it("narrow 2D F32 (skip first row)", () => {
    const rows = 4;
    const cols = 3;
    const t = ws.alloc([rows, cols], "F32");
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    t.h2d(Buffer.from(data.buffer));
    ops.synchronize();

    const view = t.narrow(1, 3);
    assert.deepEqual(view.shape, [3, 3]);

    const buf = Buffer.alloc(3 * 3 * 4);
    view.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 9);
    const expected = new Float32Array([4, 5, 6, 7, 8, 9, 10, 11, 12]);
    for (let i = 0; i < 9; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("narrow 2D F32 (middle rows)", () => {
    const t = ws.alloc([5, 4], "F32");
    const data = new Float32Array(5 * 4);
    for (let i = 0; i < data.length; i++) data[i] = i;
    t.h2d(Buffer.from(data.buffer));
    ops.synchronize();

    const view = t.narrow(2, 2);
    assert.deepEqual(view.shape, [2, 4]);

    const buf = Buffer.alloc(8 * 4);
    view.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 8);
    const expected = new Float32Array([8, 9, 10, 11, 12, 13, 14, 15]);
    for (let i = 0; i < 8; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("narrow is a view (modifying source affects result)", () => {
    const t = ws.alloc([4, 3], "F32");
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    t.h2d(Buffer.from(data.buffer));
    ops.synchronize();

    const view = t.narrow(1, 3);
    assert.deepEqual(view.shape, [3, 3]);

    const overwrite = new Float32Array([99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]);
    t.h2d(Buffer.from(overwrite.buffer));
    ops.synchronize();

    const buf = Buffer.alloc(9 * 4);
    view.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 9);
    const expected = new Float32Array([99, 99, 99, 99, 99, 99, 99, 99, 99]);
    for (let i = 0; i < 9; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("narrow 2D BF16", () => {
    const t = ws.alloc([4, 3], "BF16");
    const f32 = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    t.h2d(f32ToBf16Bytes(f32));
    ops.synchronize();

    const view = t.narrow(2, 2);
    assert.deepEqual(view.shape, [2, 3]);

    const buf = Buffer.alloc(6 * 2);
    view.d2h(buf);
    const result = bf16BytesToF32(buf);
    const expected = [7, 8, 9, 10, 11, 12];
    for (let i = 0; i < 6; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 0.01, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("narrow throws on start out of range", () => {
    const t = ws.alloc([4, 3], "F32");
    assert.throws(() => t.narrow(5, 1), /out of range/);
  });

  it("narrow throws on length exceeding dim 0", () => {
    const t = ws.alloc([4, 3], "F32");
    assert.throws(() => t.narrow(2, 3), /exceeds/);
  });

  it("narrow throws on non-positive length", () => {
    const t = ws.alloc([4, 3], "F32");
    assert.throws(() => t.narrow(0, 0), /positive/);
  });

  it("narrow throws on start out of range for 1D", () => {
    const t = ws.alloc([5], "I32");
    assert.throws(() => t.narrow(6, 1), /out of range/);
  });

  it("narrow 1D I32 gives correct view for qoIndptr pattern", () => {
    const batchSize = 3;

    const indptr = ws.alloc([batchSize + 1], "I32");
    const indptrData = new Int32Array([0, 3, 7, 10]);
    indptr.h2d(Buffer.from(indptrData.buffer));
    ops.synchronize();

    const indptrTail = indptr.narrow(1, batchSize);
    assert.deepEqual(indptrTail.shape, [batchSize]);

    const buf = Buffer.alloc(batchSize * 4);
    indptrTail.d2h(buf);
    const result = new Int32Array(buf.buffer, buf.byteOffset, batchSize);
    assert.deepEqual([...result], [3, 7, 10]);
  });
});
