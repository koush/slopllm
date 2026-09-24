import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

describe("GlmTensor.indexSelect (single GPU)", () => {
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
    ops.synchronize();

    using out = src.indexSelect(idx);
    ops.synchronize();

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
    ops.synchronize();

    using out = src.indexSelect(idx);
    ops.synchronize();

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
    ops.synchronize();

    using out = src.indexSelect(idx);
    ops.synchronize();

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
    ops.synchronize();

    using out = src.indexSelect(idx);
    ops.synchronize();

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
    ops.synchronize();

    using out = src.indexSelect(idx);
    ops.synchronize();

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
    assert.throws(() => src.indexSelect(idx), /indices must be I32/);
  });

  it("selects rows with negative offset (qoIndptr pattern)", () => {
    const batchSize = 3;
    const totalTokens = 10;
    const dim = 4;

    const indptr = ws.alloc([batchSize + 1], "I32");
    const indptrData = new Int32Array([0, 3, 7, 10]);
    indptr.h2d(Buffer.from(indptrData.buffer));
    ops.synchronize();

    const src = ws.alloc([totalTokens, dim], "BF16");
    const srcF32 = new Float32Array(totalTokens * dim);
    for (let i = 0; i < srcF32.length; i++) srcF32[i] = i * 0.1;
    src.h2d(f32ToBf16Bytes(srcF32));
    ops.synchronize();

    const indptrTail = indptr.narrow(1, batchSize);
    using selected = src.indexSelect(indptrTail, -1);
    ops.synchronize();

    const buf = Buffer.alloc(batchSize * dim * 2);
    selected.d2h(buf);
    const actual = bf16BytesToF32(buf);

    for (let i = 0; i < batchSize; i++) {
      const expectedRow = indptrData[i + 1] - 1;
      for (let j = 0; j < dim; j++) {
        const expected = srcF32[expectedRow * dim + j];
        const actualVal = actual[i * dim + j];
        const relErr = Math.abs(actualVal - expected) / Math.max(Math.abs(expected), 1e-6);
        assert.ok(relErr < 0.05, `row ${i} (src row ${expectedRow}), col ${j}: expected ${expected}, got ${actualVal}`);
      }
    }
  });

  it("selects rows with positive offset", () => {
    const rows = 4;
    const dim = 4;

    const indices = ws.alloc([2], "I32");
    const idxData = new Int32Array([0, 1]);
    indices.h2d(Buffer.from(idxData.buffer));

    const src = ws.alloc([rows, dim], "BF16");
    const srcF32 = new Float32Array(rows * dim);
    for (let i = 0; i < srcF32.length; i++) srcF32[i] = i;
    src.h2d(f32ToBf16Bytes(srcF32));
    ops.synchronize();

    using out = src.indexSelect(indices, 2);
    ops.synchronize();

    const buf = Buffer.alloc(2 * dim * 2);
    out.d2h(buf);
    const actual = bf16BytesToF32(buf);

    for (let j = 0; j < dim; j++) {
      const expected0 = srcF32[2 * dim + j];
      const expected1 = srcF32[3 * dim + j];
      assert.ok(Math.abs(actual[j] - expected0) < 0.05, `row 0 col ${j}: expected ${expected0}, got ${actual[j]}`);
      assert.ok(Math.abs(actual[dim + j] - expected1) < 0.05, `row 1 col ${j}: expected ${expected1}, got ${actual[dim + j]}`);
    }
  });
});
