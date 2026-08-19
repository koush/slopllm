import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { MemcpyKind } from "../src/enums";

describe("GlmTensor.cat (single GPU)", () => {
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

  it("cat 1D F32 along dim 0", () => {
    const a = ws.alloc([4], "F32");
    const b = ws.alloc([3], "F32");
    const aData = new Float32Array([1, 2, 3, 4]);
    const bData = new Float32Array([5, 6, 7]);
    a.h2d(Buffer.from(aData.buffer));
    b.h2d(Buffer.from(bData.buffer));

    const out = a.cat([b], 0);
    assert.deepEqual(out.shape, [7]);

    const buf = Buffer.alloc(7 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 7);
    const expected = new Float32Array([1, 2, 3, 4, 5, 6, 7]);
    for (let i = 0; i < 7; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("cat 2D F32 along dim 0", () => {
    const a = ws.alloc([2, 3], "F32");
    const b = ws.alloc([1, 3], "F32");
    const aData = new Float32Array([1, 2, 3, 4, 5, 6]);
    const bData = new Float32Array([7, 8, 9]);
    a.h2d(Buffer.from(aData.buffer));
    b.h2d(Buffer.from(bData.buffer));

    const out = a.cat([b], 0);
    assert.deepEqual(out.shape, [3, 3]);

    const buf = Buffer.alloc(9 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 9);
    const expected = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    for (let i = 0; i < 9; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("cat 2D F32 along dim 1", () => {
    const a = ws.alloc([2, 3], "F32");
    const b = ws.alloc([2, 2], "F32");
    const aData = new Float32Array([1, 2, 3, 4, 5, 6]);
    const bData = new Float32Array([10, 20, 30, 40]);
    a.h2d(Buffer.from(aData.buffer));
    b.h2d(Buffer.from(bData.buffer));

    const out = a.cat([b], 1);
    assert.deepEqual(out.shape, [2, 5]);

    const buf = Buffer.alloc(10 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 10);
    const expected = new Float32Array([1, 2, 3, 10, 20, 4, 5, 6, 30, 40]);
    for (let i = 0; i < 10; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("cat 3D F32 along dim 1", () => {
    const a = ws.alloc([2, 3, 4], "F32");
    const b = ws.alloc([2, 2, 4], "F32");
    const aData = new Float32Array(2 * 3 * 4);
    const bData = new Float32Array(2 * 2 * 4);
    for (let i = 0; i < aData.length; i++) aData[i] = i + 1;
    for (let i = 0; i < bData.length; i++) bData[i] = i + 100;
    a.h2d(Buffer.from(aData.buffer));
    b.h2d(Buffer.from(bData.buffer));

    const out = a.cat([b], 1);
    assert.deepEqual(out.shape, [2, 5, 4]);

    const buf = Buffer.alloc(2 * 5 * 4 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 2 * 5 * 4);

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 4; k++) {
          const srcIdx = i * 3 * 4 + j * 4 + k;
          const dstIdx = i * 5 * 4 + j * 4 + k;
          assert.ok(Math.abs(result[dstIdx] - aData[srcIdx]) < 1e-6,
            `a[${i},${j},${k}]: expected ${aData[srcIdx]}, got ${result[dstIdx]}`);
        }
      }
      for (let j = 0; j < 2; j++) {
        for (let k = 0; k < 4; k++) {
          const srcIdx = i * 2 * 4 + j * 4 + k;
          const dstIdx = i * 5 * 4 + (j + 3) * 4 + k;
          assert.ok(Math.abs(result[dstIdx] - bData[srcIdx]) < 1e-6,
            `b[${i},${j},${k}]: expected ${bData[srcIdx]}, got ${result[dstIdx]}`);
        }
      }
    }
  });

  it("cat three tensors along dim 0", () => {
    const a = ws.alloc([2, 4], "F32");
    const b = ws.alloc([1, 4], "F32");
    const c = ws.alloc([3, 4], "F32");
    const aData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const bData = new Float32Array([9, 10, 11, 12]);
    const cData = new Float32Array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]);
    a.h2d(Buffer.from(aData.buffer));
    b.h2d(Buffer.from(bData.buffer));
    c.h2d(Buffer.from(cData.buffer));

    const out = a.cat([b, c], 0);
    assert.deepEqual(out.shape, [6, 4]);

    const buf = Buffer.alloc(24 * 4);
    out.d2h(buf);
    const result = new Float32Array(buf.buffer, buf.byteOffset, 24);
    const expected = new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
      17, 18, 19, 20,
      21, 22, 23, 24,
    ]);
    for (let i = 0; i < 24; i++) {
      assert.ok(Math.abs(result[i] - expected[i]) < 1e-6, `out[${i}]: expected ${expected[i]}, got ${result[i]}`);
    }
  });

  it("cat BF16 tensors along dim 1", () => {
    const a = ws.alloc([3, 4], "BF16");
    const b = ws.alloc([3, 2], "BF16");
    const aF32 = new Float32Array(3 * 4);
    const bF32 = new Float32Array(3 * 2);
    for (let i = 0; i < aF32.length; i++) aF32[i] = i * 0.5;
    for (let i = 0; i < bF32.length; i++) bF32[i] = i * 1.5 + 100;
    a.h2d(f32ToBf16Bytes(aF32));
    b.h2d(f32ToBf16Bytes(bF32));

    const out = a.cat([b], 1);
    assert.deepEqual(out.shape, [3, 6]);

    const buf = Buffer.alloc(3 * 6 * 2);
    out.d2h(buf);
    const result = bf16BytesToF32(buf);

    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 4; col++) {
        const expected = aF32[row * 4 + col];
        const actual = result[row * 6 + col];
        assert.ok(Math.abs(actual - expected) < 0.01, `a[${row},${col}]: expected ${expected}, got ${actual}`);
      }
      for (let col = 0; col < 2; col++) {
        const expected = bF32[row * 2 + col];
        const actual = result[row * 6 + 4 + col];
        assert.ok(Math.abs(actual - expected) < 0.01, `b[${row},${col}]: expected ${expected}, got ${actual}`);
      }
    }
  });

  it("cat throws on mismatched non-cat dimensions", () => {
    const a = ws.alloc([2, 3], "F32");
    const b = ws.alloc([3, 3], "F32");
    assert.throws(() => a.cat([b], 1), /mismatch/);
  });

  it("cat throws on mismatched ndim", () => {
    const a = ws.alloc([2, 3], "F32");
    const b = ws.alloc([3], "F32");
    assert.throws(() => a.cat([b], 0), /2D.*1D|1D.*2D/);
  });
});
