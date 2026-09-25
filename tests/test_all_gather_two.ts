import assert from "node:assert/strict";
import { test } from "node:test";
import { GlmOps } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { TensorParallelism as TP } from "../src/device_ops";
import { WorkspaceBase } from "../src/workspace";

test("allGatherTwo: mixed layouts/dtypes, byte tails, stream waits, graph replay and fallback", () => {
  const devices = [new GlmOps(0), new GlmOps(1)];
  const ops = new ParallelOps(devices);
  const ws = new WorkspaceBase(ops);
  try {
    assert.ok(ops.p2pEnabled, "test requires direct P2P");
    const cases: [number[], TP, number[], TP][] = [
      [[4, 32, 128], TP.Row, [4, 32], TP.Row],
      [[3, 6], TP.Row, [5, 10], TP.Row],
      [[257, 34], TP.Row, [7, 6], TP.Row],
      // Combined payload exceeds the 512-block grid cap; exercise tile striding.
      [[1024, 1024], TP.Row, [512, 1024], TP.Row],
      [[4, 7], TP.Column, [3, 6], TP.Row],
      [[3, 6], TP.Row, [4, 5], TP.Column],
      [[4, 7], TP.Column, [6, 5], TP.Column],
      [[3, 6], TP.Replicated, [4, 5], TP.Column],
    ];
    for (const [shapeA, parA, shapeB, parB] of cases) {
      using tracking = ws.startTracking();
      using a = ws.alloc(shapeA, "U8", undefined, parA) as ParallelTensor;
      using b = ws.alloc(shapeB, "F32", undefined, parB) as ParallelTensor;
      const dataA = Buffer.from(Array.from({ length: a.numElements }, (_, i) => i % 251));
      const dataB = Buffer.from(Float32Array.from({ length: b.numElements }, (_, i) => i * 0.25 - 3).buffer);
      a.h2d(dataA);
      b.h2d(dataB);
      using stream = ops.withStream(() => ops.allGatherTwo(a, b, ws));
      stream.streamWaitEvent();
      using outA = stream.result[0];
      using outB = stream.result[1];
      ops.synchronize();
      for (const [out, expected] of [[outA, dataA], [outB, dataB]] as const) {
        assert.equal(out.parallelism, TP.Replicated);
        for (const shard of out.shards) {
          const actual = Buffer.alloc(expected.length);
          shard.d2h(actual);
          assert.deepEqual(actual, expected);
        }
      }
    }

    using a = ws.alloc([4, 32, 128], "U8", undefined, TP.Row) as ParallelTensor;
    using b = ws.alloc([4, 32], "F32", undefined, TP.Row) as ParallelTensor;
    a.fill(7, a.numElements);
    b.fill(9, b.numElements);
    assert.deepEqual(ops.allGatherTwo(undefined, undefined, ws), [undefined, undefined]);
    const [onlyA, missingB] = ops.allGatherTwo(a, undefined, ws);
    const [missingA, onlyB] = ops.allGatherTwo(undefined, b, ws);
    ops.synchronize();
    assert.equal(missingA, undefined);
    assert.equal(missingB, undefined);
    for (const shard of onlyA.shards) {
      const actual = Buffer.alloc(a.numElements);
      shard.d2h(actual);
      assert.deepEqual(actual, Buffer.alloc(a.numElements, 7));
    }
    for (const shard of onlyB.shards) {
      const actual = Buffer.alloc(b.bytes);
      shard.d2h(actual);
      for (let i = 0; i < b.numElements; i++) assert.equal(actual.readFloatLE(i * 4), 9);
    }
    onlyA[Symbol.dispose]();
    onlyB[Symbol.dispose]();
    for (let i = 0; i < 3; i++) {
      const warmup = ops.allGatherTwo(a, b, ws);
      warmup.forEach(t => t[Symbol.dispose]());
      ops.synchronize();
    }
    ops.graphBeginCapture();
    const outputs = ops.allGatherTwo(a, b, ws);
    const graph = ops.graphEndCapture();
    const exec = ops.graphInstantiate(graph);
    try {
      for (const value of [11, 23]) {
        a.fill(value, a.numElements);
        b.fill(value, b.numElements);
        ops.graphLaunch(exec);
        ops.synchronize();
        for (const shard of outputs[0].shards) {
          const actual = Buffer.alloc(a.numElements);
          shard.d2h(actual);
          assert.deepEqual(actual, Buffer.alloc(a.numElements, value));
        }
        for (const shard of outputs[1].shards) {
          const actual = Buffer.alloc(b.bytes);
          shard.d2h(actual);
          for (let i = 0; i < b.numElements; i++) assert.equal(actual.readFloatLE(i * 4), value);
        }
      }
    } finally {
      ops.graphExecDestroy(exec);
      ops.graphDestroy(graph);
      outputs.forEach(t => t[Symbol.dispose]());
    }
    ops.p2pEnabled = false;
    const fallback = ops.allGatherTwo(a, b, ws);
    ops.synchronize();
    for (const shard of fallback[0].shards) {
      const actual = Buffer.alloc(a.numElements);
      shard.d2h(actual);
      assert.deepEqual(actual, Buffer.alloc(a.numElements, 23));
    }
    for (const shard of fallback[1].shards) {
      const actual = Buffer.alloc(b.bytes);
      shard.d2h(actual);
      for (let i = 0; i < b.numElements; i++) assert.equal(actual.readFloatLE(i * 4), 23);
    }
    fallback.forEach(t => t[Symbol.dispose]());
    ops.p2pEnabled = true;
  } finally {
    ws.free();
    ops.free();
    devices.forEach(d => d.free());
  }
});
