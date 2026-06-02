import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

describe("GlmTensor.indexSelect (single GPU)", () => {
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

  it("selects two rows from a 4×8 BF16 tensor", () => {
    const rows = 4;
    const dim = 8;
    const k = 2;

    const srcF32 = new Float32Array(rows * dim);
    for (let i = 0; i < rows * dim; i++) srcF32[i] = i * 0.1;

    const indices = new Int32Array([1, 3]);
    const idxBuf = Buffer.alloc(k * 4);
    for (let i = 0; i < k; i++) idxBuf.writeInt32LE(indices[i], i * 4);

    const expected = new Float32Array(k * dim);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < dim; j++) {
        expected[i * dim + j] = srcF32[indices[i] * dim + j];
      }
    }

    const src = ws.alloc([rows, dim], "BF16");
    const idx = ws.alloc([k], "I32");
    src.h2d(f32ToBf16Bytes(srcF32));
    idx.h2d(idxBuf);
    glm.synchronize();

    using out = src.indexSelect(idx, k);
    glm.synchronize();

    const outBuf = Buffer.alloc(k * dim * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < k * dim; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("selects a single row (k=1)", () => {
    const rows = 6;
    const dim = 4;
    const k = 1;

    const srcF32 = new Float32Array(rows * dim);
    for (let i = 0; i < rows * dim; i++) srcF32[i] = i;

    const indices = new Int32Array([4]);
    const idxBuf = Buffer.alloc(k * 4);
    idxBuf.writeInt32LE(indices[0], 0);

    const src = ws.alloc([rows, dim], "BF16");
    const idx = ws.alloc([k], "I32");
    src.h2d(f32ToBf16Bytes(srcF32));
    idx.h2d(idxBuf);
    glm.synchronize();

    using out = src.indexSelect(idx, k);
    glm.synchronize();

    const outBuf = Buffer.alloc(k * dim * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let j = 0; j < dim; j++) {
      const expected = srcF32[4 * dim + j];
      const relErr = Math.abs(actual[j] - expected) / Math.max(Math.abs(expected), 1e-6);
      assert.ok(relErr < 0.05, `j=${j}: expected ${expected}, got ${actual[j]}`);
    }
  });

  it("selects all rows in reverse order", () => {
    const rows = 5;
    const dim = 3;
    const k = 5;

    const srcF32 = new Float32Array(rows * dim);
    for (let i = 0; i < rows * dim; i++) srcF32[i] = i;

    const indices = new Int32Array([4, 3, 2, 1, 0]);
    const idxBuf = Buffer.alloc(k * 4);
    for (let i = 0; i < k; i++) idxBuf.writeInt32LE(indices[i], i * 4);

    const expected = new Float32Array(k * dim);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < dim; j++) {
        expected[i * dim + j] = srcF32[indices[i] * dim + j];
      }
    }

    const src = ws.alloc([rows, dim], "BF16");
    const idx = ws.alloc([k], "I32");
    src.h2d(f32ToBf16Bytes(srcF32));
    idx.h2d(idxBuf);
    glm.synchronize();

    using out = src.indexSelect(idx, k);
    glm.synchronize();

    const outBuf = Buffer.alloc(k * dim * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < k * dim; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("selects with duplicate indices", () => {
    const rows = 4;
    const dim = 6;
    const k = 3;

    const srcF32 = new Float32Array(rows * dim);
    for (let i = 0; i < rows * dim; i++) srcF32[i] = i * 0.5;

    const indices = new Int32Array([2, 0, 2]);
    const idxBuf = Buffer.alloc(k * 4);
    for (let i = 0; i < k; i++) idxBuf.writeInt32LE(indices[i], i * 4);

    const expected = new Float32Array(k * dim);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < dim; j++) {
        expected[i * dim + j] = srcF32[indices[i] * dim + j];
      }
    }

    const src = ws.alloc([rows, dim], "BF16");
    const idx = ws.alloc([k], "I32");
    src.h2d(f32ToBf16Bytes(srcF32));
    idx.h2d(idxBuf);
    glm.synchronize();

    using out = src.indexSelect(idx, k);
    glm.synchronize();

    const outBuf = Buffer.alloc(k * dim * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < k * dim; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("selects from large dim (512)", () => {
    const rows = 8;
    const dim = 512;
    const k = 3;

    const srcF32 = new Float32Array(rows * dim);
    for (let i = 0; i < rows * dim; i++) srcF32[i] = (i % 100) * 0.01;

    const indices = new Int32Array([5, 0, 7]);
    const idxBuf = Buffer.alloc(k * 4);
    for (let i = 0; i < k; i++) idxBuf.writeInt32LE(indices[i], i * 4);

    const expected = new Float32Array(k * dim);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < dim; j++) {
        expected[i * dim + j] = srcF32[indices[i] * dim + j];
      }
    }

    const src = ws.alloc([rows, dim], "BF16");
    const idx = ws.alloc([k], "I32");
    src.h2d(f32ToBf16Bytes(srcF32));
    idx.h2d(idxBuf);
    glm.synchronize();

    using out = src.indexSelect(idx, k);
    glm.synchronize();

    const outBuf = Buffer.alloc(k * dim * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < k * dim; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("throws if indices are not I32", () => {
    const src = ws.alloc([4, 8], "BF16");
    const idx = ws.alloc([2], "BF16");
    assert.throws(() => src.indexSelect(idx, 2), /indices must be I32/);
  });

  it("throws if indices too small for batch", () => {
    const src = ws.alloc([4, 8], "BF16");
    const idx = ws.alloc([1], "I32");
    assert.throws(() => src.indexSelect(idx, 2), /insufficient for batch/);
  });
});
