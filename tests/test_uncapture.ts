import assert from "node:assert/strict";
import { it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import { MetaOps } from "../src/meta_ops";
import { ParallelOps } from "../src/parallel_ops";
import { Tensor } from "../src/tensor";
import { WorkspaceBase } from "../src/workspace";

it("replay reconciles split heaps before restoring multiple output addresses", (t) => {
  const ops = new MetaOps();
  const ws = new WorkspaceBase(ops);
  ws.getHeap(undefined).manage(4096, 1024);
  const originalA = ws.alloc([512], "U8");
  const originalB = ws.alloc([512], "U8");
  const a = originalA.capture();
  const b = originalB.capture();
  originalA[Symbol.dispose]();
  originalB[Symbol.dispose]();
  ws.synchronizeComplete();

  // Reproduce an output straddling two free pools after intervening work.
  const firstHalf = ws.alloc([256], "U8");
  const secondHalf = ws.alloc([256], "U8");
  const reusedB = ws.alloc([512], "U8");
  secondHalf[Symbol.dispose]();
  ws.synchronizeComplete();
  firstHalf[Symbol.dispose]();
  reusedB[Symbol.dispose]();
  assert.equal(ws.claimDevice(a.data, a.allocSize, null), false);

  const manager = new CaptureManager(ops);
  manager.captured.set("restore", {
    warmupSteps: 4, graphExec: 1, result: { a, b }, inputs: {},
    capturedWorkspaces: new Set([ws]),
  });
  t.mock.method(ws, "alloc", () => { throw new Error("restoration must not allocate"); });
  t.mock.method(ops, "graphLaunch", () => {
    assert.equal(ws.getHeap(0).contains(a.data, 256), true, "promotion must follow launch");
  });
  const restored = manager.run({}, () => { throw new Error("must replay"); }, ["restore"]) as { a: Tensor; b: Tensor };
  assert.equal(restored.a.data, a.data);
  assert.equal(restored.b.data, b.data);
  assert.notEqual(restored.a.data, restored.b.data);
  assert.equal(ws.getHeap(undefined).freeBytes, 0);
  restored.a[Symbol.dispose]();
  restored.b[Symbol.dispose]();
  assert.equal(ws.getHeap(0).freeBytes, 1024);
});

it("performs host bookkeeping after both the first captured launch and replay", (t) => {
  const ops = new MetaOps();
  const events: string[] = [];
  t.mock.method(console, "warn", () => {});
  t.mock.method(ops, "graphLaunch", () => events.push("launch"));
  t.mock.method(ops, "hostSynchronizeWorld", () => events.push("hostSync"));
  const manager = new CaptureManager(ops);
  for (let i = 0; i < 5; i++) manager.run({}, () => undefined, ["test"]);
  assert.deepEqual(events, ["launch", "hostSync", "launch", "hostSync"]);
});

it("host world bookkeeping promotes every shard's main heap but not other pools or pinned buffers", () => {
  // Use the real ParallelOps method without constructing GPU contexts.
  const devices = [new MetaOps(), new MetaOps()];
  const ops = Object.assign(Object.create(ParallelOps.prototype), {
    devices, synchronizeListeners: [],
  }) as ParallelOps;
  const workspaces = [new WorkspaceBase(ops), ...devices.map(device => new WorkspaceBase(device))];
  for (const ws of workspaces) {
    ws.getHeap(0).manage(4096, 256);
    ws.getHeap(undefined).manage(4352, 256);
    ws.getHeap(1).manage(8192, 256);
    const pinned = devices[0].wrapTensor(ws, 12288, 256, [256], "U8", true, undefined);
    ws.synchronizingHost.add(pinned);
  }
  ops.hostSynchronizeWorld();
  for (const ws of workspaces) {
    assert.equal(ws.getHeap(undefined).contains(4096, 512), true);
    assert.equal(ws.heapByKey.has(0), false);
    assert.equal(ws.getHeap(1).contains(8192, 256), true);
    assert.equal(ws.synchronizingHost.size, 1);
    assert.equal(ws.disposedHost.size, 0);
  }
});

it("uncapture fails without allocating when a free output straddles unreconciled heaps", (t) => {
  const ops = new MetaOps();
  const ws = new WorkspaceBase(ops);
  ws.getHeap(0).manage(4096, 256);
  ws.getHeap(undefined).manage(4352, 768);
  const captured = ops.wrapTensor(ws, 4096, 512, [512], "U8", false, undefined).capture();
  t.mock.method(ws, "alloc", () => { throw new Error("must not fall back to allocation"); });
  assert.throws(() => captured.uncapture(), /recorded allocation is unavailable.*bookkeeping invariant/);
  assert.equal(ws.getHeap(0).freeBytes, 256);
  assert.equal(ws.getHeap(undefined).freeBytes, 768);
  ops.hostSynchronizeWorld();
  using restored = captured.uncapture();
  assert.equal(restored.data, 4096);
});

it("uncapture preserves view offsets and rejects an unavailable backing allocation", (t) => {
  const ops = new MetaOps();
  const ws = new WorkspaceBase(ops);
  ws.getHeap(0).manage(4096, 1024);
  const root = ws.alloc([512], "U8");
  const view = ops.wrapTensor(ws, root.data + 256, 256, [256], "U8", false, root);
  const captured = view.capture();
  view[Symbol.dispose]();
  root[Symbol.dispose]();
  {
    using restored = captured.uncapture();
    assert.equal(restored.data, 4352);
    assert.equal(restored.view!.data, 4096);
    assert.equal(ws.getHeap(0).contains(4096, 512), false);
  }
  using occupied = ws.alloc([256], "U16");
  assert.equal(occupied.data, 4096);
  t.mock.method(ws, "alloc", () => { throw new Error("must not relocate a view's backing allocation"); });
  assert.throws(() => captured.uncapture(), /recorded allocation is unavailable/);
});

it("uncapture reuses a matching live owner without returning its range to the heap", () => {
  const ops = new MetaOps();
  const ws = new WorkspaceBase(ops);
  ws.getHeap(0).manage(4096, 512);
  const root = ws.alloc([512], "U8");
  const captured = root.capture();
  const restored = captured.uncapture();
  root[Symbol.dispose]();
  assert.equal(root.disposed, false);
  assert.equal(ws.getHeap(0).freeBytes, 0);
  restored[Symbol.dispose]();
  assert.equal(root.disposed, true);
  assert.equal(ws.getHeap(0).freeBytes, 512);
});
