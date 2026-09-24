import assert from "node:assert/strict";
import { it } from "node:test";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

async function fixture(run: (ops: GlmOps, ws: WorkspaceBase) => void | Promise<void>) {
  const ops = new GlmOps(Number(process.env.GLM_GPU ?? 0));
  const ws = new WorkspaceBase(ops);
  try {
    await run(ops, ws);
  } finally {
    ops.synchronize();
    ws.free();
    ops.free();
  }
}

for (const asynchronous of [false, true]) {
  it(`${asynchronous ? "async" : "sync"} synchronization retains device-wide cleanup and reserves live handles`, () => fixture(async (ops, ws) => {
    const root = ws.alloc([1024], "BF16");
    const first = ops.withStream(() => {
      using view = root.viewClone();
      using scratch = ws.alloc([1024], "BF16");
    });
    const second = ops.withStream(() => { using view = root.viewClone(); });
    root[Symbol.dispose]();
    assert.equal(root.disposed, false);
    if (asynchronous) {
      await ops.synchronizeAsync();
    } else {
      ops.synchronize();
    }
    assert.equal(root.disposed, true);
    assert.equal(ops.streamResources.has(0), false);
    for (const handle of [first, second]) {
      const resources = ops.streamResources.get(handle.streamId)!;
      assert.equal(resources.disposedViews.size, 0);
      assert.equal(resources.workspaces.size, 0);
      assert.deepEqual(resources.joined, [handle.streamId]);
      assert.equal(ops.normalPriorityStreams.includes(handle.streamId), false);
      // The handle and its event must still work after the device sync.
      handle.streamWaitEvent();
      handle[Symbol.dispose]();
    }
    assert.equal(ops.normalPriorityStreams.length + ops.highPriorityStreams.length, 63);
    assert.equal(new Set([...ops.normalPriorityStreams, ...ops.highPriorityStreams]).size, 63);
  }));
}

for (const highPriority of [false, true]) {
  it(`nested stream reuse respects requested priority ${highPriority}`, () => fixture((ops) => {
    using parent = ops.withStream(() => {
      let childId = -1;
      {
        using child = ops.withStream(!highPriority, () => {});
        childId = child.streamId;
        child.streamWaitEvent();
      }
      using requested = ops.withStream(highPriority, () => {});
      assert.equal(GlmOps.isHighPriorityStream(requested.streamId), highPriority);
      assert.notEqual(requested.streamId, childId);
      using reused = ops.withStream(!highPriority, () => {});
      assert.equal(reused.streamId, childId);
      requested.streamWaitEvent();
      reused.streamWaitEvent();
    });
    parent.streamWaitEvent();
  }));
}

it("device completion releases disposed views of live roots, but retains live views", () => fixture((ops, ws) => {
  const liveRoot = ws.alloc([1024], "BF16");
  const pendingView = liveRoot.viewClone();
  pendingView[Symbol.dispose]();
  const pendingRoot = ws.alloc([1024], "BF16");
  const liveView = pendingRoot.viewClone();
  pendingRoot[Symbol.dispose]();
  ops.synchronize();
  assert.equal(pendingView.disposed, true);
  assert.equal(liveRoot.disposed, false);
  assert.equal(liveRoot.views.size, 0);
  assert.equal(liveView.disposed, false);
  assert.equal(pendingRoot.disposed, false);
  assert.equal(pendingRoot.views.has(liveView), true);
  liveView[Symbol.dispose]();
  liveRoot[Symbol.dispose]();
  assert.equal(pendingRoot.disposed, true);
  ops.synchronize();
  assert.equal(ops.streamResources.has(0), false);
}));

it("stream-0 workspace bookkeeping is cleared after device synchronization", () => fixture((ops, ws) => {
  using stream = ops.withStream(() => { using tensor = ws.alloc([1024], "BF16"); });
  stream.streamWaitEvent();
  assert.equal(ops.getStreamResources(0).workspaces.has(ws), true);
  ops.synchronize();
  assert.equal(ops.streamResources.has(0), false);
  assert.equal(ops.streamResources.get(stream.streamId)!.workspaces.size, 0);
}));

it("explicit stream-0 synchronization retains device-wide workspace cleanup", () => fixture((ops, ws) => {
  using joined = ops.withStream(() => { using tensor = ws.alloc([1024], "BF16"); });
  joined.streamWaitEvent();
  using unrelated = ops.withStream(() => { using tensor = ws.alloc([2048], "BF16"); });
  ops.synchronize(0);
  assert.equal(ops.streamResources.has(0), false);
  assert.equal(ops.streamResources.get(unrelated.streamId)!.workspaces.size, 0);
}));

it("device synchronization returns disposed descendants without reusing live stream IDs", () => fixture((ops, ws) => {
  let childId = -1;
  using parent = ops.withStream(() => {
    using child = ops.withStream(() => { using tensor = ws.alloc([1024], "BF16"); });
    childId = child.streamId;
  });
  assert.equal(ops.normalPriorityStreams.includes(childId), false);
  ops.synchronize();
  assert.equal(ops.normalPriorityStreams.includes(childId), true);
  assert.equal(ops.normalPriorityStreams.includes(parent.streamId), false);
  assert.deepEqual(ops.streamResources.get(parent.streamId)!.joined, [parent.streamId]);
  parent.streamWaitEvent();
  assert.equal(ops.normalPriorityStreams.filter(id => id === childId).length, 1);
}));

it("stream handles use the same device-wide cleanup policy", () => fixture(async (ops, ws) => {
  const root = ws.alloc([1024], "BF16");
  using first = ops.withStream(() => { using view = root.viewClone(); });
  using second = ops.withStream(() => {
    using view = root.viewClone();
    using scratch = ws.alloc([1024], "BF16");
  });
  root[Symbol.dispose]();
  first.synchronize();
  // These streams contain no GPU reads: this checks the intentionally global
  // cleanup policy, not the safety of waiting for just one of several readers.
  assert.equal(root.disposed, true);
  assert.equal(root.views.size, 0);
  assert.equal(ops.streamResources.get(first.streamId)!.disposedViews.size, 0);
  assert.equal(ops.streamResources.get(second.streamId)!.disposedViews.size, 0);
  assert.equal(ops.streamResources.get(second.streamId)!.workspaces.size, 0);
  await ops.synchronizeAsync(second.streamId);
  assert.equal(root.disposed, true);
  assert.equal(ops.streamResources.get(second.streamId)!.disposedViews.size, 0);
}));

it("failed stream synchronization does not retire resources", () => fixture(async (ops, ws) => {
  const root = ws.alloc([1024], "BF16");
  using reader = ops.withStream(() => { using view = root.viewClone(); });
  root[Symbol.dispose]();
  assert.throws(() => ops.synchronize(999));
  await assert.rejects(ops.synchronizeAsync(-1));
  assert.equal(root.disposed, false);
  assert.equal(ops.streamResources.get(reader.streamId)!.disposedViews.size, 1);
  reader.streamWaitEvent();
  assert.equal(root.disposed, true);
}));

for (const mode of ["explicit-sync", "explicit-async", "default-sync", "default-async"]) {
  it(`${mode} synchronization waits for the selected alternate stream`, () => fixture(async (ops, ws) => {
    using tensor = ws.alloc([262144], "BF16");
    using host = ws.allocPinned([1], "BF16");
    host.withPinnedBuffer(buffer => buffer.writeUInt16LE(0));
    let graph = 0, exec = 0;
    using stream = ops.withStream(() => {
      tensor.fill(42, tensor.numElements);
      ops.graphBeginCapture();
      for (let i = 0; i < 128; i++) {
        tensor.fill(42, tensor.numElements);
      }
      graph = ops.graphEndCapture();
      exec = ops.graphInstantiate(graph);
      // Substantial queued GPU work, followed by a host copy on that stream.
      // Reading the pinned buffer after await adds no implicit CUDA wait.
      for (let i = 0; i < 128; i++) {
        ops.graphLaunch(exec);
      }
      host.memcpy(tensor, 2);
    });
    try {
      if (mode === "explicit-sync") {
        ops.synchronize(stream.streamId);
      } else if (mode === "explicit-async") {
        await ops.synchronizeAsync(stream.streamId);
      } else if (mode === "default-sync") {
        ops.pushStream(stream.streamId);
        try {
          ops.synchronize();
        } finally {
          ops.popStream(stream.streamId);
        }
      } else {
        ops.pushStream(stream.streamId);
        let completion: Promise<void>;
        try {
          completion = ops.synchronizeAsync();
        } finally {
          // The binding must snapshot ctx->active_stream before queuing its worker.
          ops.popStream(stream.streamId);
        }
        await completion;
      }
      assert.equal(bf16BytesToF32(host.readPinnedBuffer())[0], 42);
    } finally {
      ops.synchronize(stream.streamId);
      ops.graphExecDestroy(exec);
      ops.graphDestroy(graph);
    }
  }));
}
