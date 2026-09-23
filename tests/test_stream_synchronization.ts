import assert from "node:assert/strict";
import { it } from "node:test";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

async function fixture(run: (glm: GlmOps, ws: WorkspaceBase) => void | Promise<void>) {
  const glm = new GlmOps(Number(process.env.GLM_GPU ?? 0));
  const ws = new WorkspaceBase(glm);
  try {
    await run(glm, ws);
  } finally {
    glm.synchronize();
    ws.free();
    glm.free();
  }
}

for (const asynchronous of [false, true]) {
  it(`${asynchronous ? "async" : "sync"} synchronization retains device-wide cleanup and reserves live handles`, () => fixture(async (glm, ws) => {
    const root = ws.alloc([1024], "BF16");
    const first = glm.withStream(() => {
      using view = root.viewClone();
      using scratch = ws.alloc([1024], "BF16");
    });
    const second = glm.withStream(() => { using view = root.viewClone(); });
    root[Symbol.dispose]();
    assert.equal(root.disposed, false);
    if (asynchronous) {
      await glm.synchronizeAsync();
    } else {
      glm.synchronize();
    }
    assert.equal(root.disposed, true);
    assert.equal(glm.streamResources.has(0), false);
    for (const handle of [first, second]) {
      const resources = glm.streamResources.get(handle.streamId)!;
      assert.equal(resources.disposedViews.size, 0);
      assert.equal(resources.workspaces.size, 0);
      assert.deepEqual(resources.joined, [handle.streamId]);
      assert.equal(glm.normalPriorityStreams.includes(handle.streamId), false);
      // The handle and its event must still work after the device sync.
      handle.streamWaitEvent();
      handle[Symbol.dispose]();
    }
    assert.equal(glm.normalPriorityStreams.length + glm.highPriorityStreams.length, 63);
    assert.equal(new Set([...glm.normalPriorityStreams, ...glm.highPriorityStreams]).size, 63);
  }));
}

for (const highPriority of [false, true]) {
  it(`nested stream reuse respects requested priority ${highPriority}`, () => fixture((glm) => {
    using parent = glm.withStream(() => {
      let childId = -1;
      {
        using child = glm.withStream(!highPriority, () => {});
        childId = child.streamId;
        child.streamWaitEvent();
      }
      using requested = glm.withStream(highPriority, () => {});
      assert.equal(GlmOps.isHighPriorityStream(requested.streamId), highPriority);
      assert.notEqual(requested.streamId, childId);
      using reused = glm.withStream(!highPriority, () => {});
      assert.equal(reused.streamId, childId);
      requested.streamWaitEvent();
      reused.streamWaitEvent();
    });
    parent.streamWaitEvent();
  }));
}

it("device completion releases disposed views of live roots, but retains live views", () => fixture((glm, ws) => {
  const liveRoot = ws.alloc([1024], "BF16");
  const pendingView = liveRoot.viewClone();
  pendingView[Symbol.dispose]();
  const pendingRoot = ws.alloc([1024], "BF16");
  const liveView = pendingRoot.viewClone();
  pendingRoot[Symbol.dispose]();
  glm.synchronize();
  assert.equal(pendingView.disposed, true);
  assert.equal(liveRoot.disposed, false);
  assert.equal(liveRoot.views.size, 0);
  assert.equal(liveView.disposed, false);
  assert.equal(pendingRoot.disposed, false);
  assert.equal(pendingRoot.views.has(liveView), true);
  liveView[Symbol.dispose]();
  liveRoot[Symbol.dispose]();
  assert.equal(pendingRoot.disposed, true);
  glm.synchronize();
  assert.equal(glm.streamResources.has(0), false);
}));

it("stream-0 workspace bookkeeping is cleared after device synchronization", () => fixture((glm, ws) => {
  using stream = glm.withStream(() => { using tensor = ws.alloc([1024], "BF16"); });
  stream.streamWaitEvent();
  assert.equal(glm.getStreamResources(0).workspaces.has(ws), true);
  glm.synchronize();
  assert.equal(glm.streamResources.has(0), false);
  assert.equal(glm.streamResources.get(stream.streamId)!.workspaces.size, 0);
}));

it("explicit stream-0 synchronization retains device-wide workspace cleanup", () => fixture((glm, ws) => {
  using joined = glm.withStream(() => { using tensor = ws.alloc([1024], "BF16"); });
  joined.streamWaitEvent();
  using unrelated = glm.withStream(() => { using tensor = ws.alloc([2048], "BF16"); });
  glm.synchronize(0);
  assert.equal(glm.streamResources.has(0), false);
  assert.equal(glm.streamResources.get(unrelated.streamId)!.workspaces.size, 0);
}));

it("device synchronization returns disposed descendants without reusing live stream IDs", () => fixture((glm, ws) => {
  let childId = -1;
  using parent = glm.withStream(() => {
    using child = glm.withStream(() => { using tensor = ws.alloc([1024], "BF16"); });
    childId = child.streamId;
  });
  assert.equal(glm.normalPriorityStreams.includes(childId), false);
  glm.synchronize();
  assert.equal(glm.normalPriorityStreams.includes(childId), true);
  assert.equal(glm.normalPriorityStreams.includes(parent.streamId), false);
  assert.deepEqual(glm.streamResources.get(parent.streamId)!.joined, [parent.streamId]);
  parent.streamWaitEvent();
  assert.equal(glm.normalPriorityStreams.filter(id => id === childId).length, 1);
}));

it("stream handles use the same device-wide cleanup policy", () => fixture(async (glm, ws) => {
  const root = ws.alloc([1024], "BF16");
  using first = glm.withStream(() => { using view = root.viewClone(); });
  using second = glm.withStream(() => {
    using view = root.viewClone();
    using scratch = ws.alloc([1024], "BF16");
  });
  root[Symbol.dispose]();
  first.synchronize();
  // These streams contain no GPU reads: this checks the intentionally global
  // cleanup policy, not the safety of waiting for just one of several readers.
  assert.equal(root.disposed, true);
  assert.equal(root.views.size, 0);
  assert.equal(glm.streamResources.get(first.streamId)!.disposedViews.size, 0);
  assert.equal(glm.streamResources.get(second.streamId)!.disposedViews.size, 0);
  assert.equal(glm.streamResources.get(second.streamId)!.workspaces.size, 0);
  await glm.synchronizeAsync(second.streamId);
  assert.equal(root.disposed, true);
  assert.equal(glm.streamResources.get(second.streamId)!.disposedViews.size, 0);
}));

it("failed stream synchronization does not retire resources", () => fixture(async (glm, ws) => {
  const root = ws.alloc([1024], "BF16");
  using reader = glm.withStream(() => { using view = root.viewClone(); });
  root[Symbol.dispose]();
  assert.throws(() => glm.synchronize(999));
  await assert.rejects(glm.synchronizeAsync(-1));
  assert.equal(root.disposed, false);
  assert.equal(glm.streamResources.get(reader.streamId)!.disposedViews.size, 1);
  reader.streamWaitEvent();
  assert.equal(root.disposed, true);
}));

for (const mode of ["explicit-sync", "explicit-async", "default-sync", "default-async"]) {
  it(`${mode} synchronization waits for the selected alternate stream`, () => fixture(async (glm, ws) => {
    using tensor = ws.alloc([262144], "BF16");
    using host = ws.allocPinned([1], "BF16");
    host.withPinnedBuffer(buffer => buffer.writeUInt16LE(0));
    let graph = 0, exec = 0;
    using stream = glm.withStream(() => {
      tensor.fill(42, tensor.numElements);
      glm.graphBeginCapture();
      for (let i = 0; i < 128; i++) {
        tensor.fill(42, tensor.numElements);
      }
      graph = glm.graphEndCapture();
      exec = glm.graphInstantiate(graph);
      // Substantial queued GPU work, followed by a host copy on that stream.
      // Reading the pinned buffer after await adds no implicit CUDA wait.
      for (let i = 0; i < 128; i++) {
        glm.graphLaunch(exec);
      }
      host.memcpy(tensor, 2);
    });
    try {
      if (mode === "explicit-sync") {
        glm.synchronize(stream.streamId);
      } else if (mode === "explicit-async") {
        await glm.synchronizeAsync(stream.streamId);
      } else if (mode === "default-sync") {
        glm.pushStream(stream.streamId);
        try {
          glm.synchronize();
        } finally {
          glm.popStream(stream.streamId);
        }
      } else {
        glm.pushStream(stream.streamId);
        let completion: Promise<void>;
        try {
          completion = glm.synchronizeAsync();
        } finally {
          // The binding must snapshot ctx->active_stream before queuing its worker.
          glm.popStream(stream.streamId);
        }
        await completion;
      }
      assert.equal(bf16BytesToF32(host.readPinnedBuffer())[0], 42);
    } finally {
      glm.synchronize(stream.streamId);
      glm.graphExecDestroy(exec);
      glm.graphDestroy(graph);
    }
  }));
}
