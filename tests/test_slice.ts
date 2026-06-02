import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

describe("GlmTensor.slice (single GPU)", () => {
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

  it("slice 2D F32 along dim 0 (first rows)", () => {
    const t = ws.alloc([4, 3], "F32");
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    t.h2d(Buffer.from(data.buffer));

    const out = t.slice(0, 0, 2);
    assert.deepEqual(out.shape, [2, 3]);

    const buf = Buffer.alloc(6 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 6);
    const expected = new Float32Array([1, 2, 3, 4, 5, 6]);
    for (let i = 0; i < 6; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("slice 2D F32 along dim 0 (last row, negative start)", () => {
    const t = ws.alloc([4, 3], "F32");
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    t.h2d(Buffer.from(data.buffer));

    const out = t.slice(0, -1, 1);
    assert.deepEqual(out.shape, [1, 3]);

    const buf = Buffer.alloc(3 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 3);
    const expected = new Float32Array([10, 11, 12]);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("slice 2D F32 along dim 0 (middle rows)", () => {
    const t = ws.alloc([5, 4], "F32");
    const data = new Float32Array(5 * 4);
    for (let i = 0; i < data.length; i++) data[i] = i;
    t.h2d(Buffer.from(data.buffer));

    const out = t.slice(0, 2, 2);
    assert.deepEqual(out.shape, [2, 4]);

    const buf = Buffer.alloc(8 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 8);
    const expected = new Float32Array([8, 9, 10, 11, 12, 13, 14, 15]);
    for (let i = 0; i < 8; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("slice 2D F32 along dim 1 (columns)", () => {
    const t = ws.alloc([3, 5], "F32");
    const data = new Float32Array(3 * 5);
    for (let i = 0; i < data.length; i++) data[i] = i;
    t.h2d(Buffer.from(data.buffer));

    const out = t.slice(1, 1, 3);
    assert.deepEqual(out.shape, [3, 3]);

    const buf = Buffer.alloc(9 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 9);
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const expected = data[row * 5 + 1 + col];
        const actual = result[row * 3 + col];
        assert.ok(Math.abs(actual - expected) < 1e-6,
          `out[${row},${col}]: expected ${expected}, got ${actual}`);
      }
    }
  });

  it("slice 2D F32 along dim 1 (last column, negative start)", () => {
    const t = ws.alloc([2, 4], "F32");
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    t.h2d(Buffer.from(data.buffer));

    const out = t.slice(1, -1, 1);
    assert.deepEqual(out.shape, [2, 1]);

    const buf = Buffer.alloc(2 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 2);
    assert.ok(Math.abs(result[0] - 4) < 1e-6, `out[0]: expected 4, got ${result[0]}`);
    assert.ok(Math.abs(result[1] - 8) < 1e-6, `out[1]: expected 8, got ${result[1]}`);
  });

  it("slice 3D F32 along dim 1", () => {
    const t = ws.alloc([2, 4, 3], "F32");
    const data = new Float32Array(2 * 4 * 3);
    for (let i = 0; i < data.length; i++) data[i] = i;
    t.h2d(Buffer.from(data.buffer));

    const out = t.slice(1, 1, 2);
    assert.deepEqual(out.shape, [2, 2, 3]);

    const buf = Buffer.alloc(2 * 2 * 3 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 12);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        for (let k = 0; k < 3; k++) {
          const srcIdx = i * 4 * 3 + (1 + j) * 3 + k;
          const dstIdx = i * 2 * 3 + j * 3 + k;
          assert.ok(Math.abs(result[dstIdx] - data[srcIdx]) < 1e-6,
            `out[${i},${j},${k}]: expected ${data[srcIdx]}, got ${result[dstIdx]}`);
        }
      }
    }
  });

  it("slice 3D F32 along dim 2", () => {
    const t = ws.alloc([2, 3, 5], "F32");
    const data = new Float32Array(2 * 3 * 5);
    for (let i = 0; i < data.length; i++) data[i] = i;
    t.h2d(Buffer.from(data.buffer));

    const out = t.slice(2, 2, 3);
    assert.deepEqual(out.shape, [2, 3, 3]);

    const buf = Buffer.alloc(2 * 3 * 3 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 18);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 3; k++) {
          const srcIdx = i * 3 * 5 + j * 5 + (2 + k);
          const dstIdx = i * 3 * 3 + j * 3 + k;
          assert.ok(Math.abs(result[dstIdx] - data[srcIdx]) < 1e-6,
            `out[${i},${j},${k}]: expected ${data[srcIdx]}, got ${result[dstIdx]}`);
        }
      }
    }
  });

  it("slice 1D F32 along dim 0", () => {
    const t = ws.alloc([10], "F32");
    const data = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    t.h2d(Buffer.from(data.buffer));

    const out = t.slice(0, 3, 4);
    assert.deepEqual(out.shape, [4]);

    const buf = Buffer.alloc(4 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 4);
    const expected = new Float32Array([3, 4, 5, 6]);
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("slice BF16 along dim 0", () => {
    const t = ws.alloc([3, 4], "BF16");
    const f32 = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    t.h2d(f32ToBf16Bytes(f32));

    const out = t.slice(0, 1, 2);
    assert.deepEqual(out.shape, [2, 4]);

    const buf = Buffer.alloc(2 * 4 * 2);
    out.d2h(buf);
    const result = bf16BytesToF32(buf);
    const expected = [5, 6, 7, 8, 9, 10, 11, 12];
    for (let i = 0; i < 8; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 0.01, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("slice returns a copy (modifying source doesn't affect result)", () => {
    const t = ws.alloc([2, 3], "F32");
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    t.h2d(Buffer.from(data.buffer));

    const out = t.slice(0, 0, 2);
    assert.deepEqual(out.shape, [2, 3]);

    const overwrite = new Float32Array([99, 99, 99, 99, 99, 99]);
    t.h2d(Buffer.from(overwrite.buffer));

    const buf = Buffer.alloc(6 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 6);
    const expected = new Float32Array([1, 2, 3, 4, 5, 6]);
    for (let i = 0; i < 6; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("slice throws on invalid dim", () => {
    const t = ws.alloc([2, 3], "F32");
    assert.throws(() => t.slice(2, 0, 1), /out of range/);
    assert.throws(() => t.slice(-1, 0, 1), /out of range/);
  });

  it("slice throws on start out of range", () => {
    const t = ws.alloc([2, 3], "F32");
    assert.throws(() => t.slice(0, 5, 1), /out of range/);
  });

  it("slice throws on length exceeding dim", () => {
    const t = ws.alloc([2, 3], "F32");
    assert.throws(() => t.slice(1, 1, 3), /exceeds/);
  });

  it("slice throws on non-positive length", () => {
    const t = ws.alloc([2, 3], "F32");
    assert.throws(() => t.slice(0, 0, 0), /positive/);
  });

  it("slice with negative start wraps correctly", () => {
    const t = ws.alloc([5, 3], "F32");
    const data = new Float32Array(5 * 3);
    for (let i = 0; i < data.length; i++) data[i] = i;
    t.h2d(Buffer.from(data.buffer));

    const out = t.slice(0, -3, 2);
    assert.deepEqual(out.shape, [2, 3]);

    const buf = Buffer.alloc(6 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 6);
    const expected = new Float32Array([6, 7, 8, 9, 10, 11]);
    for (let i = 0; i < 6; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });
});
