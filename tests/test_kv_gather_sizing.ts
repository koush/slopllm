import assert from "node:assert/strict";
import { it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps } from "../src/glm_ops";
import { getNativeAddon } from "../src/native-addon";

function paddedCapacity(lengths: number[], worldSize = 8, maxPages = 8192, bucket = 0): number {
  const model = Object.create(Glm51Model.prototype);
  Object.assign(model, { ops: { worldSize } });
  return model.sparseMlaPaddedKvLen({
    batchSize: lengths.length,
    cache: { getPagedKV: () => ({
      contextParallel: worldSize > 1,
      pageSize: 64 * worldSize,
      maxPages,
      sequences: lengths.map(allocLen => ({ allocLen })),
    }) },
    getGraphVariantPaddedKvLen: () => bucket,
    // A phased gather must reserve the live lengths above, not this older plan.
    seqKvLens: lengths.map(length => Math.max(0, length - 4096)),
  });
}

it("KV gather padding covers every rank at aggregate page boundaries", () => {
  for (const [lengths, expected] of [
    [[129, 129, 129, 125], 1024],
    [[65537, 65537, 65537, 65533], 262656],
    [[100097, 100097, 100097, 100093], 400896],
    [[128, 128, 128, 128], 512],
    [[129, 129, 129, 124], 1024],
    [[129, 129, 129, 126], 1024],
    [[8193], 8704],
  ] as [number[], number][]) {
    const capacity = paddedCapacity(lengths);
    assert.equal(capacity, expected);
    for (let rank = 0; rank < 8; rank++) {
      const packedRows = lengths.reduce((sum, length) => sum + Math.max(0, Math.ceil((length - rank) / 8)), 0);
      assert.ok(packedRows <= capacity / 8, `rank ${rank}: ${packedRows} > ${capacity / 8}`);
    }
    assert.ok(capacity >= lengths.reduce((sum, length) => sum + length, 0));
  }
});

it("KV gather padding handles non-CP and duplicated prefix pages", () => {
  assert.equal(paddedCapacity([65, 63], 1), 128);
  // Both sequences can reference the same physical page. Padding must not be
  // clamped back to the single-page source capacity (128 global tokens).
  assert.equal(paddedCapacity([65, 63], 2, 1), 256);
  assert.equal(paddedCapacity([128, 128], 2, 1), 256);
});

it("captured KV padding is stable within a bucket and covers CP imbalance", () => {
  const previous = CaptureManager.capturing;
  CaptureManager.capturing = {} as any;
  try {
    assert.equal(paddedCapacity([129, 129, 129, 125], 8, 8192, 512), 1024);
    assert.equal(paddedCapacity([128, 128, 128, 128], 8, 8192, 512), 1024);
    // At full source capacity the batch-wide maximum is already sufficient.
    assert.equal(paddedCapacity([512, 512, 512, 512], 8, 1, 2048), 2048);
  } finally {
    CaptureManager.capturing = previous;
  }
});

it("page gather uses a replay-stable bound for shared page references", t => {
  const ops = Object.create(GlmOps.prototype) as GlmOps;
  Object.assign(ops, { ctx: 1 });
  const workspace = { alloc: () => ({ data: 256 }) };
  const metadata: any = { workspace, data: 256, numElements: 1 };
  const source: any = { data: 512, shape: [2, 64, 128], type: "U8" };
  const bounds: number[] = [];
  t.mock.method(getNativeAddon(), "gatherPages", (...args: any[]) => { bounds.push(args[6]); });
  const previous = CaptureManager.capturing;
  try {
    CaptureManager.capturing = undefined;
    ops.gatherPages(source, metadata, metadata, metadata, 2, 128, metadata, false);
    CaptureManager.capturing = {} as any;
    ops.gatherPages(source, metadata, metadata, metadata, 2, 128, metadata, false);
    assert.deepEqual(bounds, [1, 4]);
  } finally {
    CaptureManager.capturing = previous;
  }
});
