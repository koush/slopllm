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
      viewClone() { return tensor(shape, type); },
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

for (const failure of ["score", "histogram", "unrelated", "captured", "exhausted"] as const) {
  it(`indexer prefill allocation recovery (${failure})`, t => {
    const ops = Object.create(GlmOps.prototype) as GlmOps;
    Object.assign(ops, { ctx: 1, arenaBase: 256, heap: { tryAlloc: () => undefined } });
    let allocationError: Error;
    try {
      (ops as any).allocDevice(1024);
      assert.fail("expected arena OOM");
    } catch (error) {
      allocationError = error as Error;
      assert.match(allocationError.message, /Arena heap OOM/);
    }
    const unrelated = new Error("unrelated failure");
    const live = new Set<object>();
    const calls: number[][] = [];
    const totalQ = 129;
    const start = 17;
    const stride = 512;
    const topk = 2;
    let failed = false;
    const tensor = (shape: number[], type: string, rowOffset = 0, owned = false): any => {
      const value = {
        shape, type, workspace, data: rowOffset,
        narrow(offset: number, rows: number) {
          assert.ok(offset >= 0 && rows > 0 && offset + rows <= shape[0]);
          return tensor([rows, ...shape.slice(1)], type, rowOffset + offset, true);
        },
        viewClone() { return tensor(shape, type, rowOffset, true); },
        memcpy(source: any) { assert.deepEqual(source.shape, shape); },
        [Symbol.dispose]() { live.delete(value); },
      };
      if (owned) live.add(value);
      return value;
    };
    const workspace = {
      alloc(shape: number[], type: string): any {
        const target = failure === "histogram" ? shape[1] === 1024 : type === "BF16" && shape[1] === stride;
        if (target && (shape[0] > 32 || failure === "exhausted")) {
          failed = true;
          throw failure === "unrelated" ? unrelated : allocationError;
        }
        return tensor(shape, type, 0, true);
      },
    };
    t.mock.method(getNativeAddon(), "indexerScoreTopkPrefill", (...args: any[]) => {
      const rows = args[12];
      const globalStart = args[18];
      assert.equal(args[3], globalStart - start, "Q must be sliced at the global query offset");
      assert.equal(args[33], globalStart - start, "effective weights must follow Q");
      calls.push([globalStart, rows]);
    });
    t.mock.method(getNativeAddon(), "indexerScoreTopkV2", () => assert.fail("microbatches must keep prefill kernels"));
    const previousCapture = CaptureManager.capturing;
    if (failure === "captured") CaptureManager.capturing = {} as any;
    try {
      const metadata = tensor([3], "I32");
      const state: any = { batchSize: 2, seqKvLens: [256, stride], getGraphVariantPaddedKvLen: () => stride };
      const run = () => ops.indexerTopk(state, tensor([totalQ, 32, 128], "U8"),
        tensor([8, 64, 128], "U8"), tensor([8, 64], "F32"), undefined,
        metadata, metadata, metadata, metadata, 1, topk, false, start,
        undefined, undefined, undefined, 0, 0, metadata, metadata, tensor([totalQ, 32], "F32"));
      if (failure === "unrelated" || failure === "captured" || failure === "exhausted") {
        assert.throws(run, error => error === (failure === "unrelated" ? unrelated : allocationError));
        assert.equal(calls.length, 0);
      } else {
        const result = run();
        assert.deepEqual(result.indices.shape, [totalQ, topk]);
        assert.deepEqual(result.values.shape, [totalQ, topk]);
        let next = start;
        for (const [offset, rows] of calls) {
          assert.equal(offset, next);
          assert.ok(rows <= 32);
          next += rows;
        }
        assert.equal(next, start + totalQ);
        assert.equal(live.size, 2, "only returned outputs may remain live");
        result.indices[Symbol.dispose]();
        result.values[Symbol.dispose]();
      }
      assert.ok(failed);
      assert.equal(live.size, 0, "failed attempts must release every allocated tensor");
    } finally {
      CaptureManager.capturing = previousCapture;
    }
  });
}
