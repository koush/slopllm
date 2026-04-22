import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { TensorParallelism } from "../src/device_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";

process.env.CUDA_VISIBLE_DEVICES = process.env.GLM_GPU ?? "0,1";

describe("ParallelOps construction", () => {
  it("accepts a single device", () => {
    const glm = new GlmOps(0);
    const po = new ParallelOps([glm]);
    assert.equal(po.worldSize, 1);
    assert.equal(po.devices.length, 1);
    glm.free();
  });

  it("accepts two devices", () => {
    const glm0 = new GlmOps(0);
    const glm1 = new GlmOps(1);
    const po = new ParallelOps([glm0, glm1]);
    assert.equal(po.worldSize, 2);
    assert.equal(po.devices.length, 2);
    glm0.free();
    glm1.free();
  });

  it("rejects empty device list", () => {
    assert.throws(() => new ParallelOps([] as GlmOps[]), /at least one device/);
  });

  it("rejects non-power-of-2 device count", () => {
    const glm0 = new GlmOps(0);
    const glm1 = new GlmOps(1);
    const glm2 = new GlmOps(0);
    assert.throws(() => new ParallelOps([glm0, glm1, glm2]), /power-of-2/);
    glm0.free();
    glm1.free();
    glm2.free();
  });
});

describe("ParallelOps shardShape", () => {
  let po: ParallelOps;
  let glm0: GlmOps;
  let glm1: GlmOps;

  before(() => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
  });

  after(() => {
    glm0.free();
    glm1.free();
  });

  it("column parallel splits dim 0", () => {
    assert.deepEqual(po.shardShape([8, 4], TensorParallelism.Column), [4, 4]);
    assert.deepEqual(po.shardShape([1024, 512], TensorParallelism.Column), [512, 512]);
  });

  it("row parallel splits dim 1", () => {
    assert.deepEqual(po.shardShape([4, 8], TensorParallelism.Row), [4, 4]);
    assert.deepEqual(po.shardShape([512, 1024], TensorParallelism.Row), [512, 512]);
  });

  it("replicated preserves shape", () => {
    assert.deepEqual(po.shardShape([4, 4], TensorParallelism.Replicated), [4, 4]);
    assert.deepEqual(po.shardShape([8, 16], TensorParallelism.Replicated), [8, 16]);
  });

  it("column parallel throws on indivisible dim 0", () => {
    assert.throws(() => po.shardShape([7, 4], TensorParallelism.Column), /not divisible/);
  });

  it("row parallel throws on indivisible dim 1", () => {
    assert.throws(() => po.shardShape([4, 7], TensorParallelism.Row), /not divisible/);
  });

  it("row parallel throws on 1D shape", () => {
    assert.throws(() => po.shardShape([4], TensorParallelism.Row), /2D/);
  });
});

describe("ParallelTensor allocation and properties", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let po: ParallelOps;
  let ws: WorkspaceBase;

  before(() => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
    ws = new WorkspaceBase(po);
  });

  after(() => {
    ws.free();
    glm0.free();
    glm1.free();
  });

  it("column parallel tensor has correct properties", () => {
    const pt = ws.alloc([8, 4], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    assert.equal(pt.parallelism, TensorParallelism.Column);
    assert.deepEqual(pt.fullShape, [8, 4]);
    assert.deepEqual(pt.shape, [8, 4]);
    assert.equal(pt.type, "F32");
    assert.equal(pt.data, 0);
    assert.equal(pt.shards.length, 2);
    assert.deepEqual(pt.shard(0).shape, [4, 4]);
    assert.deepEqual(pt.shard(1).shape, [4, 4]);
    assert.equal(pt.shard(0).type, "F32");
    assert.equal(pt.shard(1).type, "F32");
  });

  it("row parallel tensor has correct properties", () => {
    const pt = ws.alloc([4, 8], "F32", undefined, TensorParallelism.Row) as ParallelTensor;
    assert.equal(pt.parallelism, TensorParallelism.Row);
    assert.deepEqual(pt.fullShape, [4, 8]);
    assert.equal(pt.shards.length, 2);
    assert.deepEqual(pt.shard(0).shape, [4, 4]);
    assert.deepEqual(pt.shard(1).shape, [4, 4]);
  });

  it("replicated tensor has correct properties", () => {
    const pt = ws.alloc([4, 4], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    assert.equal(pt.parallelism, TensorParallelism.Replicated);
    assert.deepEqual(pt.fullShape, [4, 4]);
    assert.equal(pt.shards.length, 2);
    assert.deepEqual(pt.shard(0).shape, [4, 4]);
    assert.deepEqual(pt.shard(1).shape, [4, 4]);
    assert.equal(pt.shard(0).type, "BF16");
  });

  it("pinned parallel tensor has pinned shards", () => {
    const pt = ws.allocPinned([8, 4], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    assert.equal(pt.pinned, true);
    assert.equal(pt.shard(0).pinned, true);
    assert.equal(pt.shard(1).pinned, true);
  });

  it("shards are on different devices", () => {
    const pt = ws.alloc([8, 4], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    assert.notEqual(pt.shard(0).data, 0);
    assert.notEqual(pt.shard(1).data, 0);
    assert.notEqual(pt.shard(0).data, pt.shard(1).data);
  });
});

describe("ParallelTensor h2d/d2h round-trip", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let po: ParallelOps;
  let ws: WorkspaceBase;

  before(() => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
    ws = new WorkspaceBase(po);
  });

  after(() => {
    ws.free();
    glm0.free();
    glm1.free();
  });

  it("column parallel: write and read each shard (F32)", () => {
    const pt = ws.alloc([8, 4], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    const shardElems = 4 * 4;

    const shard0F32 = new Float32Array(shardElems);
    const shard1F32 = new Float32Array(shardElems);
    for (let i = 0; i < shardElems; i++) {
      shard0F32[i] = i;
      shard1F32[i] = i + shardElems;
    }

    pt.shard(0).h2d(Buffer.from(shard0F32.buffer));
    pt.shard(1).h2d(Buffer.from(shard1F32.buffer));

    glm0.synchronize();
    glm1.synchronize();

    const dst0Buf = Buffer.alloc(shardElems * 4);
    const dst1Buf = Buffer.alloc(shardElems * 4);
    pt.shard(0).d2h(dst0Buf);
    pt.shard(1).d2h(dst1Buf);

    const dst0F32 = new Float32Array(dst0Buf.buffer, dst0Buf.byteOffset, shardElems);
    const dst1F32 = new Float32Array(dst1Buf.buffer, dst1Buf.byteOffset, shardElems);

    for (let i = 0; i < shardElems; i++) {
      assert.ok(Math.abs(dst0F32[i] - shard0F32[i]) < 1e-6, `shard0[${i}]: expected ${shard0F32[i]}, got ${dst0F32[i]}`);
      assert.ok(Math.abs(dst1F32[i] - shard1F32[i]) < 1e-6, `shard1[${i}]: expected ${shard1F32[i]}, got ${dst1F32[i]}`);
    }
  });

  it("row parallel: write and read each shard (F32)", () => {
    const pt = ws.alloc([4, 8], "F32", undefined, TensorParallelism.Row) as ParallelTensor;
    const shardElems = 4 * 4;

    const shard0F32 = new Float32Array(shardElems);
    const shard1F32 = new Float32Array(shardElems);
    for (let i = 0; i < shardElems; i++) {
      shard0F32[i] = i * 1.5;
      shard1F32[i] = i * 2.5 + 100;
    }

    pt.shard(0).h2d(Buffer.from(shard0F32.buffer));
    pt.shard(1).h2d(Buffer.from(shard1F32.buffer));

    glm0.synchronize();
    glm1.synchronize();

    const dst0Buf = Buffer.alloc(shardElems * 4);
    const dst1Buf = Buffer.alloc(shardElems * 4);
    pt.shard(0).d2h(dst0Buf);
    pt.shard(1).d2h(dst1Buf);

    const dst0F32 = new Float32Array(dst0Buf.buffer, dst0Buf.byteOffset, shardElems);
    const dst1F32 = new Float32Array(dst1Buf.buffer, dst1Buf.byteOffset, shardElems);

    for (let i = 0; i < shardElems; i++) {
      assert.ok(Math.abs(dst0F32[i] - shard0F32[i]) < 1e-6, `shard0[${i}]: expected ${shard0F32[i]}, got ${dst0F32[i]}`);
      assert.ok(Math.abs(dst1F32[i] - shard1F32[i]) < 1e-6, `shard1[${i}]: expected ${shard1F32[i]}, got ${dst1F32[i]}`);
    }
  });

  it("replicated: write same data to both shards and verify (BF16)", () => {
    const pt = ws.alloc([4, 4], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const shardElems = 4 * 4;

    const srcF32 = new Float32Array(shardElems);
    for (let i = 0; i < shardElems; i++) srcF32[i] = i + 0.5;

    const shardBytes = shardElems * 2;
    const srcBuf = Buffer.alloc(shardBytes);
    for (let i = 0; i < shardElems; i++) {
      srcBuf.writeUInt16LE(srcF32[i] !== 0 ? ((new Uint32Array(new Float32Array([srcF32[i]]).buffer)[0]) >>> 16) : 0, i * 2);
    }

    pt.shard(0).h2d(srcBuf);
    pt.shard(1).h2d(srcBuf);

    glm0.synchronize();
    glm1.synchronize();

    const dst0Buf = Buffer.alloc(shardBytes);
    const dst1Buf = Buffer.alloc(shardBytes);
    pt.shard(0).d2h(dst0Buf);
    pt.shard(1).d2h(dst1Buf);

    assert.deepEqual(dst0Buf, srcBuf);
    assert.deepEqual(dst1Buf, srcBuf);
  });
});

describe("ParallelTensor disposal and recycling", () => {
  it("disposes shards into per-device workspaces", () => {
    const glm0 = new GlmOps(0);
    const glm1 = new GlmOps(1);
    const po = new ParallelOps([glm0, glm1]);
    const ws = new WorkspaceBase(po);

    const pt = ws.alloc([8, 4], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    const sws = po.shardWorkspacesFor(ws);

    assert.ok(ws.tracked.has(pt), "ParallelTensor should be in main workspace tracked");
    assert.equal(sws[0].tracked.size, 1, "shard 0 should be in device 0 workspace tracked");
    assert.equal(sws[1].tracked.size, 1, "shard 1 should be in device 1 workspace tracked");

    pt[Symbol.dispose]();

    assert.ok(!ws.tracked.has(pt), "ParallelTensor should be removed from main workspace tracked");
    assert.ok(!ws.disposed.has(pt), "ParallelTensor should NOT be in main workspace disposed");
    assert.equal(sws[0].disposed.size, 1, "shard 0 should be in device 0 workspace disposed");
    assert.equal(sws[1].disposed.size, 1, "shard 1 should be in device 1 workspace disposed");

    ws.free();
    glm0.free();
    glm1.free();
  });

  it("recycles shard buffers from per-device workspaces", () => {
    const glm0 = new GlmOps(0);
    const glm1 = new GlmOps(1);
    const po = new ParallelOps([glm0, glm1]);
    const ws = new WorkspaceBase(po);

    let s0_data: number;
    let s1_data: number;

    {
      using pt = ws.alloc([8, 4], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
      s0_data = pt.shard(0).data;
      s1_data = pt.shard(1).data;

      assert.notEqual(s0_data, 0, "shard 0 should have valid GPU pointer");
      assert.notEqual(s1_data, 0, "shard 1 should have valid GPU pointer");
    }

    const sws = po.shardWorkspacesFor(ws);
    assert.equal(sws[0].disposed.size, 1, "device 0 disposed should have 1 shard after scope exit");
    assert.equal(sws[1].disposed.size, 1, "device 1 disposed should have 1 shard after scope exit");

    const pt2 = ws.alloc([8, 4], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    assert.equal(pt2.shard(0).data, s0_data, "shard 0 buffer should be recycled");
    assert.equal(pt2.shard(1).data, s1_data, "shard 1 buffer should be recycled");

    ws.free();
    glm0.free();
    glm1.free();
  });

  it("using pattern auto-disposes ParallelTensor", () => {
    const glm0 = new GlmOps(0);
    const glm1 = new GlmOps(1);
    const po = new ParallelOps([glm0, glm1]);
    const ws = new WorkspaceBase(po);

    {
      using pt = ws.alloc([4, 8], "F32", undefined, TensorParallelism.Row) as ParallelTensor;
      assert.equal(pt.shards.length, 2);
      const sws = po.shardWorkspacesFor(ws);
      assert.equal(sws[0].tracked.size, 1);
      assert.equal(sws[1].tracked.size, 1);
    }

    const sws = po.shardWorkspacesFor(ws);
    assert.equal(sws[0].disposed.size, 1, "shard 0 should be recycled after using scope");
    assert.equal(sws[1].disposed.size, 1, "shard 1 should be recycled after using scope");

    ws.free();
    glm0.free();
    glm1.free();
  });

  it("different requesting workspaces get isolated shard pools", () => {
    const glm0 = new GlmOps(0);
    const glm1 = new GlmOps(1);
    const po = new ParallelOps([glm0, glm1]);
    const ws1 = new WorkspaceBase(po);
    const ws2 = new WorkspaceBase(po);

    const pt1 = ws1.alloc([8, 4], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    const pt2 = ws2.alloc([8, 4], "F32", undefined, TensorParallelism.Column) as ParallelTensor;

    const sws1 = po.shardWorkspacesFor(ws1);
    const sws2 = po.shardWorkspacesFor(ws2);

    assert.equal(sws1.length, 2, "ws1 should have 2 shard workspaces");
    assert.equal(sws2.length, 2, "ws2 should have 2 shard workspaces");
    assert.notEqual(sws1[0], sws2[0], "ws1 and ws2 should have different device 0 shard workspaces");
    assert.notEqual(sws1[1], sws2[1], "ws1 and ws2 should have different device 1 shard workspaces");

    assert.equal(sws1[0].tracked.size, 1, "ws1 device 0 should have 1 tracked shard");
    assert.equal(sws2[0].tracked.size, 1, "ws2 device 0 should have 1 tracked shard");

    pt1[Symbol.dispose]();
    assert.equal(sws1[0].disposed.size, 1, "ws1 device 0 should have 1 disposed after pt1 disposed");
    assert.equal(sws2[0].disposed.size, 0, "ws2 device 0 should have 0 disposed (unaffected)");

    pt2[Symbol.dispose]();
    assert.equal(sws2[0].disposed.size, 1, "ws2 device 0 should have 1 disposed after pt2 disposed");

    ws1.free();
    ws2.free();
    glm0.free();
    glm1.free();
  });
});
