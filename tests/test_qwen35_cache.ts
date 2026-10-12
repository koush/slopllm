import assert from "node:assert/strict";
import { it } from "node:test";
import { GlmOps } from "../src/glm_ops";
import { MetaOps } from "../src/meta_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { PagedKVCache } from "../src/paged_kv";
import { Qwen35ChatCache } from "../src/qwen35_model";
import type { Tensor } from "../src/tensor";

const cfg = {
  numKeyValueHeads: 2, headDim: 64, numFullAttnLayers: 1,
  layerTypes: ["linear_attention", "full_attention", "linear_attention"],
  linearNumKeyHeads: 4, linearKeyHeadDim: 128, linearValueHeadDim: 128, linearConvKernelDim: 4,
};

function assertZero(tensors: Tensor[]): void {
  for (const t of tensors) {
    const bytes = Buffer.alloc(t.bytes);
    t.d2h(bytes);
    assert.ok(bytes.equals(Buffer.alloc(t.bytes)), `nonzero recurrent state: ${t.name}`);
  }
}

for (const tp of [1, 2]) {
  it(`Qwen35 cache owns, resets, and frees per-layer recurrent tensors (TP${tp})`, () => {
    using gpu0 = new GlmOps(0);
    using gpu1 = tp === 2 ? new GlmOps(4) : undefined;
    using parallel = gpu1 ? new ParallelOps([gpu0, gpu1]) : undefined;
    const ops = parallel ?? gpu0;
    using cache = new Qwen35ChatCache(ops, cfg, 8, 2, 128);
    assert.ok(cache instanceof PagedKVCache);
    assert.equal(cache.getPagedKV(), cache);
    assert.equal(cache.convState.length, 3);
    assert.equal(cache.recurrentState.length, 3);
    assert.equal(cache.convState[1], undefined);
    assert.equal(cache.recurrentState[1], undefined);
    assert.equal(cache.kData.length, 1); // Full-attention indices remain compact.
    assert.deepEqual(cache.convState[0]!.shape, [2, 4 * 384 * 3]);
    assert.deepEqual(cache.recurrentState[0]!.shape, [2, 4 * 128 * 128]);
    const tensors = [...cache.convState, ...cache.recurrentState].filter(t => t !== undefined);
    const addresses = tensors.map(t => t.memoryRanges().map(range => range.data));
    for (const t of tensors) {
      assert.equal(t.workspace, cache);
      assert.equal(cache.tensors.get(t.name!), t);
      if (t instanceof ParallelTensor) {
        assert.deepEqual(t.shards[0].shape, [2, t.shape[1] / tp]);
      }
    }
    assertZero(tensors);
    cache.reset(2);
    cache.allocAppendPages(0, 3);
    cache.reportTokens(0, [1, 2, 3], 4);
    for (const t of tensors) t.fill(.25, t.numElements);
    cache.reset(1);
    assert.equal(cache.sequences.length, 1);
    assert.equal(cache.sequences[0].allocLen, 0);
    assert.equal(cache.availablePages.length, cache.maxPages);
    assert.deepEqual(tensors.map(t => t.memoryRanges().map(range => range.data)), addresses);
    assertZero(tensors); // Includes inactive capacity slots.

    cache.allocAppendPages(0, 3);
    cache.reportTokens(0, [1, 2, 3], 4);
    for (const t of tensors) t.fill(.5, t.numElements);
    assert.deepEqual(cache.getPagedKV().prefixMatch(0, [1, 2, 3, 4]), [1, 2, 3, 4]);
    assert.equal(cache.sequences[0].allocLen, 0);
    assertZero(tensors);
    assert.throws(() => cache.prefillBatchPlanHook(1, [257], 257), /exceeds max/);
    assert.throws(() => cache.reset(3), /exceeds maxBatch/);
    ops.synchronize();
    cache.free();
    assert.equal(cache.tensors.size, 0);
    assert.ok(tensors.every(t => t.memoryRanges().every(range => range.data === 0)));
  });
}

it("ordinary paged KV caches have no recurrent tensors", () => {
  using ops = new MetaOps();
  using cache = new PagedKVCache(ops, 2, 64, 2, 8, 2);
  assert.deepEqual(cache.convState, []);
  assert.deepEqual(cache.recurrentState, []);
  cache.reset(2);
  cache.allocAppendPages(0, 3);
  cache.reportTokens(0, [1, 2, 3], 4);
  cache.copySequence(1, 0);
  assert.deepEqual(cache.sequences[1].getTokenIds(), [1, 2, 3]);
  assert.equal(cache.sequences[1].targetToken, 4);
});
