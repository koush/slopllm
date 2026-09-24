import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { Tensor } from "../src/tensor";
import { WorkspaceBase } from "../src/workspace";

function fixture(run: (ops: GlmOps, ws: WorkspaceBase) => void) {
  const ops = new GlmOps(Number(process.env.GLM_GPU ?? 0));
  const ws = new WorkspaceBase(ops);
  try {
    run(ops, ws);
  } finally {
    ops.synchronize();
    ws.free();
    ops.free();
  }
}

function input(ws: WorkspaceBase) {
  const tensor = ws.alloc([1024], "I32");
  tensor.h2d(Buffer.from(new Int32Array(1024).fill(42).buffer));
  return tensor;
}

function checkOutput(tensor: Tensor, ops: GlmOps) {
  const buffer = Buffer.alloc(tensor.bytes);
  tensor.d2h(buffer);
  ops.synchronize();
  assert.deepEqual([...new Int32Array(buffer.buffer, buffer.byteOffset, 1024)], new Array(1024).fill(42));
}

for (const parentFirst of [false, true]) {
  it(`alternate view protects allocation (${parentFirst ? "parent" : "view"} disposed first)`, () => fixture((ops, ws) => {
    const root = input(ws);
    const pointer = root.data;
    const view = root.viewClone();
    if (parentFirst) {
      root[Symbol.dispose]();
    }
    using stream = ops.withStream(() => {
      using v = view;
      const output = ws.alloc(root.shape, root.type);
      output.memcpy(v);
      return output;
    });
    if (!parentFirst) {
      root[Symbol.dispose]();
    }
    // A second disposal on the parent stream must not move the first request
    // out of the reader stream's ordering domain.
    view[Symbol.dispose]();
    root[Symbol.dispose]();
    assert.equal(root.disposed, false);
    assert.equal(root.views.has(view), true);
    using replacement = ws.alloc([1024], "I32");
    assert.notEqual(replacement.data, pointer);
    replacement.h2d(Buffer.from(new Int32Array(1024).fill(7).buffer));
    stream.streamWaitEvent();
    assert.equal(root.disposed, true);
    assert.equal(view.disposed, true);
    using reused = ws.alloc([1024], "I32");
    assert.equal(reused.data, pointer);
    using output = stream.result;
    checkOutput(output, ops);
  }));
}

it("flattened nested views retain the root independently", () => fixture((ops, ws) => {
  const root = input(ws);
  const middle = root.narrow(0, 512);
  const leaf = middle.reshape([256, 2]);
  assert.equal(leaf.view, root);
  middle[Symbol.dispose]();
  root[Symbol.dispose]();
  assert.equal(root.disposed, false);
  assert.deepEqual([...root.views], [leaf]);
  using stream = ops.withStream(() => { leaf[Symbol.dispose](); });
  assert.equal(root.disposed, false);
  stream.streamWaitEvent();
  assert.equal(root.disposed, true);
  assert.equal(root.views.size, 0);
}));

it("all reader streams must join before the root is released", () => fixture((ops, ws) => {
  const root = input(ws);
  using first = ops.withStream(() => { using view = root.viewClone(); });
  using second = ops.withStream(() => { using view = root.viewClone(); });
  root[Symbol.dispose]();
  first.streamWaitEvent();
  assert.equal(root.disposed, false);
  assert.equal(root.views.size, 1);
  second.streamWaitEvent();
  assert.equal(root.disposed, true);
}));

it("alternate-to-alternate joins preserve protection until the parent joins", () => fixture((ops, ws) => {
  const root = input(ws);
  const pointer = root.data;
  using reader = ops.withStream(() => { using view = root.viewClone(); });
  root[Symbol.dispose]();
  using bridge = ops.withStream(() => { reader.streamWaitEvent(); });
  assert.equal(root.disposed, false);
  using replacement = ws.alloc([1024], "I32");
  assert.notEqual(replacement.data, pointer);
  bridge.streamWaitEvent();
  assert.equal(root.disposed, true);
}));

it("explicit tracking cleanup force-releases even pending roots", () => fixture((ops, ws) => {
  const tracking = ws.startTracking();
  const root = input(ws);
  using reader = ops.withStream(() => { using view = root.viewClone(); });
  root[Symbol.dispose]();
  assert.equal(root.disposed, false);
  assert.equal(root.views.size, 1);
  tracking[Symbol.dispose]();
  assert.equal(root.disposed, true);
  assert.equal(ws.tracked.size, 0);
  reader.streamWaitEvent();
  assert.equal(root.disposed, true);
  assert.equal(ops.getStreamResources(0).disposedViews.size, 0);
}));

it("handle disposal without an explicit join safely drains pending references", () => fixture((ops, ws) => {
  const root = input(ws);
  const reader = ops.withStream(() => {
    using view = root.viewClone();
    const output = ws.alloc(root.shape, root.type);
    output.memcpy(view);
    return output;
  });
  root[Symbol.dispose]();
  reader[Symbol.dispose]();
  assert.equal(root.disposed, true);
  using output = reader.result;
  checkOutput(output, ops);
}));

it("captured nested view restores its own pointer and shape", () => fixture((ops, ws) => {
  const root = input(ws);
  const middle = root.narrow(256, 512);
  const leaf = middle.reshape([256, 2]);
  const captured = leaf.capture();
  const pointer = leaf.data;
  leaf[Symbol.dispose]();
  middle[Symbol.dispose]();
  root[Symbol.dispose]();
  using restored = captured.uncapture();
  assert.equal(restored.data, pointer);
  assert.deepEqual(restored.shape, [256, 2]);
}));

for (const parentFirst of [false, true]) {
  it(`ordinary disposal avoids scanning unrelated pending clones (${parentFirst ? "parent" : "view"} first)`, () => fixture((ops, ws) => {
    const shared = input(ws);
    const pending = ops.getStreamResources(0).disposedViews;
    const originalIterator = pending[Symbol.iterator];
    let scans = 0;
    pending[Symbol.iterator] = () => {
      scans++;
      return originalIterator.call(pending);
    };
    try {
      for (let layer = 0; layer < 78; layer++) {
        using clone = shared.viewClone();
      }
      assert.equal(pending.size, 78);
      assert.equal(shared.views.size, 78);
      const root = input(ws);
      const view = root.viewClone();
      if (parentFirst) {
        root[Symbol.dispose]();
        view[Symbol.dispose]();
      } else {
        view[Symbol.dispose]();
        root[Symbol.dispose]();
      }
      assert.equal(root.disposed, true);
      assert.equal(view.disposed, true);
      assert.equal(pending.size, 78);
      assert.equal(shared.views.size, 78);
      shared[Symbol.dispose]();
      assert.equal(shared.disposed, true);
      assert.equal(pending.size, 0);
      assert.equal(scans, 0);
    } finally {
      pending[Symbol.iterator] = originalIterator;
    }
  }));
}
