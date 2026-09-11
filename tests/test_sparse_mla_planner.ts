import assert from "node:assert/strict";
import { it } from "node:test";
import { sparseMlaChunksPerBlock } from "../src/sparse-mla-planner";

it("plans from local work and query/head parallelism", () => {
  assert.equal(sparseMlaChunksPerBlock(4, 64, 2048, 188), 3);
  assert.equal(sparseMlaChunksPerBlock(8, 64, 2048, 188), 3);
  assert.equal(sparseMlaChunksPerBlock(4, 64, 256, 188), 1);
  assert.equal(sparseMlaChunksPerBlock(8, 64, 256, 188), 1);
  assert.equal(sparseMlaChunksPerBlock(4, 64, 512, 188), 1);
  assert.equal(sparseMlaChunksPerBlock(8, 64, 512, 188), 2);
  assert.equal(sparseMlaChunksPerBlock(4, 64, 1024, 188), 2);
  assert.equal(sparseMlaChunksPerBlock(4, 64, 0, 188), 1);
});

it("keeps enough work coverage for irregular lengths and device sizes", () => {
  for (const queries of [1, 4, 8, 16, 64]) {
    for (const heads of [8, 16, 64]) {
      for (const length of [1, 65, 193, 257, 1025, 2048]) {
        for (const sms of [48, 96, 188]) {
          const chunks = Math.ceil(length / 64);
          const cpb = sparseMlaChunksPerBlock(queries, heads, length, sms);
          assert.ok(Number.isInteger(cpb) && cpb >= 1 && cpb <= chunks);
          assert.ok(Math.ceil(chunks / cpb) * cpb * 64 >= length);
        }
      }
    }
  }
});
