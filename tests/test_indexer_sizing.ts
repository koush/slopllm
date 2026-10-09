import assert from "node:assert/strict";
import { it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import { GlmOps } from "../src/glm_ops";
import { getNativeAddon } from "../src/native-addon";

// Exercise the production dispatcher without allocating the old 1.84 GiB
// score matrix. Native numerical coverage lives in test_indexer_flat.py.
for (const mode of ["flat", "cp", "captured", "direct"] as const) {
  it(`indexer sizing separates score rows from K storage (${mode})`, t => {
    const allocations: number[][] = [];
    const workspace = {
      alloc(shape: number[], type: string): any {
        allocations.push(shape);
        return tensor(shape, type);
      },
    };
    const tensor = (shape: number[], type: string): any => ({
      shape, type, workspace, data: 256, [Symbol.dispose]() {},
    });
    const ops = Object.create(GlmOps.prototype) as GlmOps;
    Object.assign(ops, { ctx: 1 });
    const direct = mode === "direct";
    const cp = mode === "cp";
    const rows = direct ? 60 : 957;
    const capacity = 1032192;
    let keyed = false;
    const state: any = {
      batchSize: 16,
      seqKvLens: [...Array(15).fill(64509), 64511],
      paddedKvLen: 1048576,
      getGraphVariantPaddedKvLen() { keyed = true; return this.paddedKvLen; },
      // Sizing must use the plan snapshot, not a cache advanced by later plans.
      getEagerKvLen() { throw new Error("live cache length read"); },
    };
    let nativeArgs: any[] | undefined;
    t.mock.method(getNativeAddon(), direct ? "indexerScoreTopkV2" : "indexerScoreTopkPrefill",
      (...args: any[]) => { nativeArgs = args; });
    const previousCapture = CaptureManager.capturing;
    if (mode === "captured") CaptureManager.capturing = {} as any;
    try {
      const metadata = tensor([17], "I32");
      ops.indexerTopk(state, tensor([rows, 32, 128], "U8"),
        tensor([capacity / 64, 64, 128], "U8"), tensor([capacity / 64, 64], "F32"),
        undefined, metadata, metadata, metadata, metadata, 1, 2048, false,
        0, undefined, undefined, undefined, cp ? 8 : 0, cp ? 7 : 0,
        metadata, cp ? undefined : metadata, tensor([rows, 32], "F32"));
      const expectedStride = direct || mode === "captured" ? capacity
        : cp ? 8063 : 64511;
      assert.deepEqual(allocations[2], [rows, expectedStride]);
      assert.equal(keyed, mode === "captured");
      assert.ok(nativeArgs);
      if (!direct) {
        assert.equal(nativeArgs[24], expectedStride);
        assert.equal(nativeArgs[34], capacity, "TMA must cover the full flat K buffer");
      }
    } finally {
      CaptureManager.capturing = previousCapture;
    }
  });
}
