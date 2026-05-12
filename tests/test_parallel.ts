import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32, NCCL_BFLOAT16, NCCL_FLOAT32, NCCL_SUM } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { TensorParallelism } from "../src/device_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";

describe("ParallelOps construction", () => {
  it("accepts a single device", () => {
    const glm = new GlmOps(0);
    const po = new ParallelOps([glm]);
    assert.equal(po.worldSize, 1);
    assert.equal(po.devices.length, 1);
    assert.equal(po.comms.length, 0);
    po.free();
    glm.free();
  });

  it("accepts two devices", () => {
    const glm0 = new GlmOps(0);
    const glm1 = new GlmOps(1);
    const po = new ParallelOps([glm0, glm1]);
    assert.equal(po.worldSize, 2);
    assert.equal(po.devices.length, 2);
    assert.equal(po.comms.length, 2);
    po.free();
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
    po.free();
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

  it("partial_sum preserves shape like replicated", () => {
    assert.deepEqual(po.shardShape([8, 16], TensorParallelism.PartialSum), [8, 16]);
    assert.deepEqual(po.shardShape([2, 4], TensorParallelism.PartialSum), [2, 4]);
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
    po.free();
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
    po.free();
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
    po.free();
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
    po.free();
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
    po.free();
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
    po.free();
    glm0.free();
    glm1.free();
  });
});

function refLinear(x: Float32Array, w: Float32Array, batch: number, n: number, k: number): Float32Array {
  const y = new Float32Array(batch * n);
  for (let b = 0; b < batch; b++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let i = 0; i < k; i++) {
        sum += x[b * k + i] * w[j * k + i];
      }
      y[b * n + j] = sum;
    }
  }
  return y;
}

describe("ParallelOps.linear", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("Column W + Replicated X → Row Y (BF16)", () => {
    const batch = 2;
    const n = 8;
    const k = 16;

    const weightF32 = new Float32Array(n * k);
    const inputF32 = new Float32Array(batch * k);
    for (let i = 0; i < n * k; i++) weightF32[i] = (i % 7 - 3) * 0.1;
    for (let i = 0; i < batch * k; i++) inputF32[i] = (i % 5 - 2) * 0.1;

    const expectedF32 = refLinear(inputF32, weightF32, batch, n, k);

    const weight = ws.alloc([n, k], "BF16", undefined, TensorParallelism.Column) as ParallelTensor;
    const input = ws.alloc([batch, k], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;

    weight.h2d(f32ToBf16Bytes(weightF32));
    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    using output = input.linear(weight, batch);
    assert.equal(output.parallelism, TensorParallelism.Row, "output should be Row-parallel");
    po.synchronize();

    const outputBuf = Buffer.alloc(batch * n * 2);
    output.d2h(outputBuf);
    const outputF32 = bf16BytesToF32(outputBuf);

    for (let i = 0; i < batch * n; i++) {
      const relErr = Math.abs(outputF32[i] - expectedF32[i]) / Math.max(Math.abs(expectedF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expectedF32[i]}, got ${outputF32[i]} (relErr=${relErr})`);
    }
  });

  it("Row W + Row X → PartialSum Y (BF16)", () => {
    const batch = 2;
    const n = 8;
    const k = 16;

    const weightF32 = new Float32Array(n * k);
    const inputF32 = new Float32Array(batch * k);
    for (let i = 0; i < n * k; i++) weightF32[i] = (i % 7 - 3) * 0.1;
    for (let i = 0; i < batch * k; i++) inputF32[i] = (i % 5 - 2) * 0.1;

    const expectedF32 = refLinear(inputF32, weightF32, batch, n, k);

    const weight = ws.alloc([n, k], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const input = ws.alloc([batch, k], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;

    weight.h2d(f32ToBf16Bytes(weightF32));
    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    using output = input.linear(weight, batch);
    assert.equal(output.parallelism, TensorParallelism.PartialSum, "output should be PartialSum");
    po.synchronize();

    const outputBuf = Buffer.alloc(batch * n * 2);
    output.d2h(outputBuf);
    const outputF32 = bf16BytesToF32(outputBuf);

    for (let i = 0; i < batch * n; i++) {
      const relErr = Math.abs(outputF32[i] - expectedF32[i]) / Math.max(Math.abs(expectedF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expectedF32[i]}, got ${outputF32[i]} (relErr=${relErr})`);
    }
  });

  it("Replicated W + Replicated X → Replicated Y (BF16)", () => {
    const batch = 2;
    const n = 8;
    const k = 16;

    const weightF32 = new Float32Array(n * k);
    const inputF32 = new Float32Array(batch * k);
    for (let i = 0; i < n * k; i++) weightF32[i] = (i % 7 - 3) * 0.1;
    for (let i = 0; i < batch * k; i++) inputF32[i] = (i % 5 - 2) * 0.1;

    const expectedF32 = refLinear(inputF32, weightF32, batch, n, k);

    const weight = ws.alloc([n, k], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const input = ws.alloc([batch, k], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;

    weight.h2d(f32ToBf16Bytes(weightF32));
    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    using output = input.linear(weight, batch);
    assert.equal(output.parallelism, TensorParallelism.Replicated, "output should be Replicated");
    po.synchronize();

    const outputBuf = Buffer.alloc(batch * n * 2);
    output.d2h(outputBuf);
    const outputF32 = bf16BytesToF32(outputBuf);

    for (let i = 0; i < batch * n; i++) {
      const relErr = Math.abs(outputF32[i] - expectedF32[i]) / Math.max(Math.abs(expectedF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expectedF32[i]}, got ${outputF32[i]} (relErr=${relErr})`);
    }
  });

  it("linear handles all parallelism combinations", () => {
    // Column weight + Replicated input → Row output
    const colW = ws.alloc([8, 16], "BF16", undefined, TensorParallelism.Column) as ParallelTensor;
    const repX = ws.alloc([2, 16], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const result = repX.linear(colW, 2);
    assert.equal((result as ParallelTensor).parallelism, TensorParallelism.Row);
  });

  it("linear rejects PartialSum weight with Row input", () => {
    const weight = ws.alloc([8, 16], "BF16", undefined, TensorParallelism.PartialSum) as ParallelTensor;
    const input = ws.alloc([2, 16], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;

    assert.throws(() => input.linear(weight, 2), /PartialSum weight requires Replicated input/);
  });
});

describe("ParallelTensor.allReduce", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("allReduce BF16 PartialSum → Replicated", () => {
    const rows = 4;
    const cols = 8;
    const totalElems = rows * cols;

    const pt = ws.alloc([rows, cols], "BF16", undefined, TensorParallelism.PartialSum) as ParallelTensor;
    assert.equal(pt.parallelism, TensorParallelism.PartialSum);

    const shard0F32 = new Float32Array(totalElems);
    const shard1F32 = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) {
      shard0F32[i] = (i % 7 - 3) * 0.1;
      shard1F32[i] = (i % 5 + 1) * 0.15;
    }

    pt.shard(0).h2d(f32ToBf16Bytes(shard0F32));
    pt.shard(1).h2d(f32ToBf16Bytes(shard1F32));
    po.synchronize();

    pt.allReduce();
    assert.equal(pt.parallelism, TensorParallelism.Replicated, "parallelism should be Replicated after allReduce");
    po.synchronize();

    const expected = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) {
      expected[i] = shard0F32[i] + shard1F32[i];
    }

    const dstBuf = Buffer.alloc(totalElems * 2);
    pt.d2h(dstBuf);
    const actual = bf16BytesToF32(dstBuf);

    for (let i = 0; i < totalElems; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("allReduce F32 PartialSum → Replicated", () => {
    const rows = 3;
    const cols = 6;
    const totalElems = rows * cols;

    const pt = ws.alloc([rows, cols], "F32", undefined, TensorParallelism.PartialSum) as ParallelTensor;

    const shard0F32 = new Float32Array(totalElems);
    const shard1F32 = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) {
      shard0F32[i] = (i + 1) * 0.5;
      shard1F32[i] = (i + 1) * -0.3;
    }

    pt.shard(0).h2d(Buffer.from(shard0F32.buffer));
    pt.shard(1).h2d(Buffer.from(shard1F32.buffer));
    po.synchronize();

    pt.allReduce();
    assert.equal(pt.parallelism, TensorParallelism.Replicated);
    po.synchronize();

    const expected = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) {
      expected[i] = shard0F32[i] + shard1F32[i];
    }

    const dstBuf = Buffer.alloc(totalElems * 4);
    pt.d2h(dstBuf);
    const actual = new Float32Array(dstBuf.buffer, dstBuf.byteOffset, totalElems);

    for (let i = 0; i < totalElems; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 1e-5, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("allReduce rejects non-PartialSum tensor", () => {
    const pt = ws.alloc([4, 4], "F32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    assert.throws(() => pt.allReduce(), /PartialSum/);
  });
});

describe("ParallelTensor.allGather", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let po: ParallelOps;
  let ws: WorkspaceBase;
  let ref: GlmOps;
  let refWs: WorkspaceBase;

  before(() => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
    ws = new WorkspaceBase(po);
    ref = new GlmOps(2);
    refWs = new WorkspaceBase(ref);
  });

  after(() => {
    refWs.free();
    ref.free();
    ws.free();
    po.free();
    glm0.free();
    glm1.free();
  });

  it("allGather Row BF16 → Replicated", () => {
    const rows = 4;
    const cols = 8;
    const totalElems = rows * cols;

    const fullF32 = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) fullF32[i] = (i % 11 - 5) * 0.1;

    const pt = ws.alloc([rows, cols], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    pt.h2d(f32ToBf16Bytes(fullF32));
    po.synchronize();

    const gathered = pt.allGather(ws);
    assert.equal(gathered.parallelism, TensorParallelism.Replicated);
    assert.equal(gathered.fullShape[0], rows);
    assert.equal(gathered.fullShape[1], cols);
    assert.notStrictEqual(gathered, pt, "allGather should return new tensor for Row");
    po.synchronize();

    const dstBuf = Buffer.alloc(totalElems * 2);
    gathered.d2h(dstBuf);
    const actual = bf16BytesToF32(dstBuf);

    for (let i = 0; i < totalElems; i++) {
      const relErr = Math.abs(actual[i] - fullF32[i]) / Math.max(Math.abs(fullF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${fullF32[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("allGather Column F32 → Replicated", () => {
    const rows = 8;
    const cols = 4;
    const totalElems = rows * cols;

    const fullF32 = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) fullF32[i] = (i + 1) * 0.25;

    const pt = ws.alloc([rows, cols], "F32", undefined, TensorParallelism.Column) as ParallelTensor;
    pt.h2d(Buffer.from(fullF32.buffer));
    po.synchronize();

    const gathered = pt.allGather(ws);
    assert.equal(gathered.parallelism, TensorParallelism.Replicated);
    assert.notStrictEqual(gathered, pt, "allGather should return new tensor for Column");
    po.synchronize();

    const dstBuf = Buffer.alloc(totalElems * 4);
    gathered.d2h(dstBuf);
    const actual = new Float32Array(dstBuf.buffer, dstBuf.byteOffset, totalElems);

    for (let i = 0; i < totalElems; i++) {
      const relErr = Math.abs(actual[i] - fullF32[i]) / Math.max(Math.abs(fullF32[i]), 1e-6);
      assert.ok(relErr < 1e-5, `i=${i}: expected ${fullF32[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("allGather Replicated returns same tensor", () => {
    const pt = ws.alloc([4, 4], "F32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const result = pt.allGather(ws);
    assert.strictEqual(result, pt, "allGather on Replicated should return same tensor");
  });

  it("allGather rejects PartialSum tensor", () => {
    const pt = ws.alloc([4, 4], "F32", undefined, TensorParallelism.PartialSum) as ParallelTensor;
    assert.throws(() => pt.allGather(ws), /PartialSum/);
  });
});

describe("ParallelTensor.all", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("all() on Replicated returns same tensor", () => {
    const pt = ws.alloc([4, 4], "F32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const result = pt.allGather(ws);
    assert.strictEqual(result, pt);
  });

  it("all() on PartialSum calls allReduce", () => {
    const totalElems = 4 * 4;
    const pt = ws.alloc([4, 4], "F32", undefined, TensorParallelism.PartialSum) as ParallelTensor;

    const shard0F32 = new Float32Array(totalElems);
    const shard1F32 = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) {
      shard0F32[i] = i * 0.1;
      shard1F32[i] = i * 0.2;
    }

    pt.shard(0).h2d(Buffer.from(shard0F32.buffer));
    pt.shard(1).h2d(Buffer.from(shard1F32.buffer));
    po.synchronize();

    const result = pt.allReduce();
    assert.strictEqual(result, pt, "allReduce() on PartialSum should return same tensor (allReduce is in-place)");
    assert.equal(pt.parallelism, TensorParallelism.Replicated, "parallelism should be Replicated after all()");
    po.synchronize();

    const expected = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) expected[i] = shard0F32[i] + shard1F32[i];

    const dstBuf = Buffer.alloc(totalElems * 4);
    pt.d2h(dstBuf);
    const actual = new Float32Array(dstBuf.buffer, dstBuf.byteOffset, totalElems);

    for (let i = 0; i < totalElems; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 1e-5, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("all() on Row calls allGather and returns new Replicated tensor", () => {
    const rows = 4;
    const cols = 8;
    const totalElems = rows * cols;

    const fullF32 = new Float32Array(totalElems);
    for (let i = 0; i < totalElems; i++) fullF32[i] = (i % 9 - 4) * 0.1;

    const pt = ws.alloc([rows, cols], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    pt.h2d(f32ToBf16Bytes(fullF32));
    po.synchronize();

    const result = pt.allGather(ws);
    assert.notStrictEqual(result, pt, "all() on Row should return new tensor");
    assert.equal(result.parallelism, TensorParallelism.Replicated);
    assert.equal(pt.parallelism, TensorParallelism.Row, "original tensor should still be Row");
    po.synchronize();

    const dstBuf = Buffer.alloc(totalElems * 2);
    result.d2h(dstBuf);
    const actual = bf16BytesToF32(dstBuf);

    for (let i = 0; i < totalElems; i++) {
      const relErr = Math.abs(actual[i] - fullF32[i]) / Math.max(Math.abs(fullF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${fullF32[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });
});

function refSiluAndMul(gate: Float32Array, up: Float32Array, intermediate: number, batch: number): Float32Array {
  const out = new Float32Array(batch * intermediate);
  for (let b = 0; b < batch; b++) {
    for (let j = 0; j < intermediate; j++) {
      const g = gate[b * intermediate + j];
      const u = up[b * intermediate + j];
      out[b * intermediate + j] = g / (1 + Math.exp(-g)) * u;
    }
  }
  return out;
}

function refRmsnorm(input: Float32Array, weight: Float32Array, eps: number, dim: number, batch: number): Float32Array {
  const out = new Float32Array(batch * dim);
  for (let b = 0; b < batch; b++) {
    let ss = 0;
    for (let j = 0; j < dim; j++) {
      const v = input[b * dim + j];
      ss += v * v;
    }
    const rms = Math.sqrt(ss / dim + eps);
    for (let j = 0; j < dim; j++) {
      out[b * dim + j] = (input[b * dim + j] / rms) * weight[j];
    }
  }
  return out;
}

function refFusedAddRmsnorm(inputA: Float32Array, inputB: Float32Array, weight: Float32Array, eps: number, dim: number, batch: number): { normed: Float32Array, residual: Float32Array } {
  const residual = new Float32Array(batch * dim);
  const normed = new Float32Array(batch * dim);
  for (let b = 0; b < batch; b++) {
    for (let j = 0; j < dim; j++) {
      residual[b * dim + j] = inputA[b * dim + j] + inputB[b * dim + j];
    }
    let ss = 0;
    for (let j = 0; j < dim; j++) {
      const v = residual[b * dim + j];
      ss += v * v;
    }
    const rms = Math.sqrt(ss / dim + eps);
    for (let j = 0; j < dim; j++) {
      normed[b * dim + j] = (residual[b * dim + j] / rms) * weight[j];
    }
  }
  return { normed, residual };
}

describe("ParallelOps.siluAndMul", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("siluAndMul on Row-parallel tensors (BF16)", () => {
    const batch = 2;
    const intermediate = 16;

    const gateF32 = new Float32Array(batch * intermediate);
    const upF32 = new Float32Array(batch * intermediate);
    for (let i = 0; i < batch * intermediate; i++) {
      gateF32[i] = (i % 7 - 3) * 0.1;
      upF32[i] = (i % 5 + 1) * 0.1;
    }
    const expected = refSiluAndMul(gateF32, upF32, intermediate, batch);

    const gate = ws.alloc([batch, intermediate], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const up = ws.alloc([batch, intermediate], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;

    gate.h2d(f32ToBf16Bytes(gateF32));
    up.h2d(f32ToBf16Bytes(upF32));
    po.synchronize();

    using out = gate.siluAndMul(up, intermediate, batch);
    po.synchronize();

    const outBuf = Buffer.alloc(batch * intermediate * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < batch * intermediate; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("siluAndMul on Replicated tensors (BF16)", () => {
    const batch = 2;
    const intermediate = 8;

    const gateF32 = new Float32Array(batch * intermediate);
    const upF32 = new Float32Array(batch * intermediate);
    for (let i = 0; i < batch * intermediate; i++) {
      gateF32[i] = (i % 9 - 4) * 0.1;
      upF32[i] = (i % 3 + 1) * 0.15;
    }
    const expected = refSiluAndMul(gateF32, upF32, intermediate, batch);

    const gate = ws.alloc([batch, intermediate], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const up = ws.alloc([batch, intermediate], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;

    gate.h2d(f32ToBf16Bytes(gateF32));
    up.h2d(f32ToBf16Bytes(upF32));
    po.synchronize();

    using out = gate.siluAndMul(up, intermediate, batch);
    po.synchronize();

    const outBuf = Buffer.alloc(batch * intermediate * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < batch * intermediate; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });
});

describe("ParallelOps.fill and arange", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("fill on Replicated BF16 tensor", () => {
    const n = 8;
    const pt = ws.alloc([n], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pt.fill(3.14, n);
    po.synchronize();

    const buf = Buffer.alloc(n * 2);
    pt.d2h(buf);
    const arr = bf16BytesToF32(buf);
    for (let i = 0; i < n; i++) {
      assert.ok(Math.abs(arr[i] - 3.14) < 0.02, `fill: i=${i}, expected ~3.14, got ${arr[i]}`);
    }
  });

  it("fill on Row-parallel BF16 tensor fills each shard", () => {
    const rows = 2;
    const cols = 8;
    const n = rows * cols;
    const pt = ws.alloc([rows, cols], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    pt.fill(2.5, n);
    po.synchronize();

    const buf = Buffer.alloc(n * 2);
    pt.d2h(buf);
    const arr = bf16BytesToF32(buf);
    for (let i = 0; i < n; i++) {
      assert.ok(Math.abs(arr[i] - 2.5) < 0.02, `fill Row: i=${i}, expected ~2.5, got ${arr[i]}`);
    }
  });

  it("arange on Replicated I32 tensor", () => {
    const count = 8;
    const pt = ws.alloc([count], "I32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    pt.arange(0, 1, count);
    po.synchronize();

    const buf = Buffer.alloc(count * 4);
    pt.d2h(buf);
    for (let i = 0; i < count; i++) {
      assert.equal(buf.readInt32LE(i * 4), i, `arange: i=${i}, expected ${i}`);
    }
  });
});

describe("ParallelOps.rmsnorm", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("rmsnorm on Replicated tensor (BF16)", () => {
    const batch = 2;
    const dim = 8;
    const eps = 1e-6;

    const inputF32 = new Float32Array(batch * dim);
    const weightF32 = new Float32Array(dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = (i % 7 - 3) * 0.1;
    for (let i = 0; i < dim; i++) weightF32[i] = 0.5 + i * 0.05;

    const expected = refRmsnorm(inputF32, weightF32, eps, dim, batch);

    const input = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const weight = ws.alloc([dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;

    input.h2d(f32ToBf16Bytes(inputF32));
    weight.h2d(f32ToBf16Bytes(weightF32));
    po.synchronize();

    using out = input.rmsnorm(weight, eps, dim, batch);
    po.synchronize();

    const outBuf = Buffer.alloc(batch * dim * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < batch * dim; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("rmsnorm auto-allReduces PartialSum input (BF16)", () => {
    const batch = 2;
    const dim = 8;
    const eps = 1e-6;

    const shard0F32 = new Float32Array(batch * dim);
    const shard1F32 = new Float32Array(batch * dim);
    const weightF32 = new Float32Array(dim);
    for (let i = 0; i < batch * dim; i++) {
      shard0F32[i] = (i % 7 - 3) * 0.05;
      shard1F32[i] = (i % 5 + 1) * 0.05;
    }
    for (let i = 0; i < dim; i++) weightF32[i] = 0.5 + i * 0.05;

    const inputF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = shard0F32[i] + shard1F32[i];
    const expected = refRmsnorm(inputF32, weightF32, eps, dim, batch);

    const input = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.PartialSum) as ParallelTensor;
    const weight = ws.alloc([dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;

    input.shard(0).h2d(f32ToBf16Bytes(shard0F32));
    input.shard(1).h2d(f32ToBf16Bytes(shard1F32));
    weight.h2d(f32ToBf16Bytes(weightF32));
    po.synchronize();

    assert.equal(input.parallelism, TensorParallelism.PartialSum);
    using out = input.rmsnorm(weight, eps, dim, batch);
    assert.equal(input.parallelism, TensorParallelism.Replicated, "rmsnorm should auto-allReduce PartialSum input");
    po.synchronize();

    const outBuf = Buffer.alloc(batch * dim * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < batch * dim; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });
});

describe("ParallelOps.fusedAddRmsnorm", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("fusedAddRmsnorm on Replicated tensors (BF16)", () => {
    const batch = 2;
    const dim = 8;
    const eps = 1e-6;

    const inputAF32 = new Float32Array(batch * dim);
    const inputBF32 = new Float32Array(batch * dim);
    const weightF32 = new Float32Array(dim);
    for (let i = 0; i < batch * dim; i++) {
      inputAF32[i] = (i % 7 - 3) * 0.1;
      inputBF32[i] = (i % 5 + 1) * 0.08;
    }
    for (let i = 0; i < dim; i++) weightF32[i] = 0.5 + i * 0.05;

    const expected = refFusedAddRmsnorm(inputAF32, inputBF32, weightF32, eps, dim, batch);

    const inputA = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const inputB = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const weight = ws.alloc([dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;

    inputA.h2d(f32ToBf16Bytes(inputAF32));
    inputB.h2d(f32ToBf16Bytes(inputBF32));
    weight.h2d(f32ToBf16Bytes(weightF32));
    po.synchronize();

    const { normed, residual } = inputA.fusedAddRmsnorm(inputB, weight, eps, dim, batch);
    po.synchronize();

    const normedBuf = Buffer.alloc(batch * dim * 2);
    const residualBuf = Buffer.alloc(batch * dim * 2);
    normed.d2h(normedBuf);
    residual.d2h(residualBuf);
    const actualNormed = bf16BytesToF32(normedBuf);
    const actualResidual = bf16BytesToF32(residualBuf);

    for (let i = 0; i < batch * dim; i++) {
      const nRelErr = Math.abs(actualNormed[i] - expected.normed[i]) / Math.max(Math.abs(expected.normed[i]), 1e-6);
      assert.ok(nRelErr < 0.05, `normed i=${i}: expected ${expected.normed[i]}, got ${actualNormed[i]} (relErr=${nRelErr})`);
      const rRelErr = Math.abs(actualResidual[i] - expected.residual[i]) / Math.max(Math.abs(expected.residual[i]), 1e-6);
      assert.ok(rRelErr < 0.05, `residual i=${i}: expected ${expected.residual[i]}, got ${actualResidual[i]} (relErr=${rRelErr})`);
    }
  });

  it("fusedAddRmsnorm auto-allReduces PartialSum inputA (BF16)", () => {
    const batch = 2;
    const dim = 8;
    const eps = 1e-6;

    const shard0F32 = new Float32Array(batch * dim);
    const shard1F32 = new Float32Array(batch * dim);
    const inputBF32 = new Float32Array(batch * dim);
    const weightF32 = new Float32Array(dim);
    for (let i = 0; i < batch * dim; i++) {
      shard0F32[i] = (i % 7 - 3) * 0.05;
      shard1F32[i] = (i % 5 + 1) * 0.05;
      inputBF32[i] = (i % 3 + 1) * 0.1;
    }
    for (let i = 0; i < dim; i++) weightF32[i] = 0.5 + i * 0.05;

    const inputAF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputAF32[i] = shard0F32[i] + shard1F32[i];
    const expected = refFusedAddRmsnorm(inputAF32, inputBF32, weightF32, eps, dim, batch);

    const inputA = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.PartialSum) as ParallelTensor;
    const inputB = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const weight = ws.alloc([dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;

    inputA.shard(0).h2d(f32ToBf16Bytes(shard0F32));
    inputA.shard(1).h2d(f32ToBf16Bytes(shard1F32));
    inputB.h2d(f32ToBf16Bytes(inputBF32));
    weight.h2d(f32ToBf16Bytes(weightF32));
    po.synchronize();

    assert.equal(inputA.parallelism, TensorParallelism.PartialSum);
    const { normed, residual } = inputA.fusedAddRmsnorm(inputB, weight, eps, dim, batch);
    assert.equal(inputA.parallelism, TensorParallelism.Replicated, "fusedAddRmsnorm should auto-allReduce PartialSum inputA");
    po.synchronize();

    const normedBuf = Buffer.alloc(batch * dim * 2);
    const residualBuf = Buffer.alloc(batch * dim * 2);
    normed.d2h(normedBuf);
    residual.d2h(residualBuf);
    const actualNormed = bf16BytesToF32(normedBuf);
    const actualResidual = bf16BytesToF32(residualBuf);

    for (let i = 0; i < batch * dim; i++) {
      const nRelErr = Math.abs(actualNormed[i] - expected.normed[i]) / Math.max(Math.abs(expected.normed[i]), 1e-6);
      assert.ok(nRelErr < 0.05, `normed i=${i}: expected ${expected.normed[i]}, got ${actualNormed[i]} (relErr=${nRelErr})`);
      const rRelErr = Math.abs(actualResidual[i] - expected.residual[i]) / Math.max(Math.abs(expected.residual[i]), 1e-6);
      assert.ok(rRelErr < 0.05, `residual i=${i}: expected ${expected.residual[i]}, got ${actualResidual[i]} (relErr=${rRelErr})`);
    }
  });
});

describe("ParallelOps.embedding", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("Row-parallel embedding lookup (BF16)", () => {
    const vocabSize = 8;
    const hidden = 16;
    const seqLen = 3;
    const ids = [2, 0, 5];

    const tableF32 = new Float32Array(vocabSize * hidden);
    for (let i = 0; i < vocabSize * hidden; i++) tableF32[i] = (i % 11 - 5) * 0.1;

    const expectedF32 = new Float32Array(seqLen * hidden);
    for (let s = 0; s < seqLen; s++) {
      for (let h = 0; h < hidden; h++) {
        expectedF32[s * hidden + h] = tableF32[ids[s] * hidden + h];
      }
    }

    const table = ws.alloc([vocabSize, hidden], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    const idsBuf = Buffer.alloc(seqLen * 4);
    for (let s = 0; s < seqLen; s++) idsBuf.writeInt32LE(ids[s], s * 4);

    const inputIds = ws.alloc([seqLen], "I32", undefined, TensorParallelism.Replicated) as ParallelTensor;

    table.h2d(f32ToBf16Bytes(tableF32));
    inputIds.h2d(idsBuf);
    po.synchronize();

    using out = table.embedding(inputIds, hidden, seqLen);
    po.synchronize();

    const outBuf = Buffer.alloc(seqLen * hidden * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < seqLen * hidden; i++) {
      const relErr = Math.abs(actual[i] - expectedF32[i]) / Math.max(Math.abs(expectedF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expectedF32[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });

  it("Replicated embedding lookup (BF16)", () => {
    const vocabSize = 4;
    const hidden = 8;
    const seqLen = 2;
    const ids = [1, 3];

    const tableF32 = new Float32Array(vocabSize * hidden);
    for (let i = 0; i < vocabSize * hidden; i++) tableF32[i] = i * 0.25;

    const expectedF32 = new Float32Array(seqLen * hidden);
    for (let s = 0; s < seqLen; s++) {
      for (let h = 0; h < hidden; h++) {
        expectedF32[s * hidden + h] = tableF32[ids[s] * hidden + h];
      }
    }

    const table = ws.alloc([vocabSize, hidden], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const idsBuf = Buffer.alloc(seqLen * 4);
    for (let s = 0; s < seqLen; s++) idsBuf.writeInt32LE(ids[s], s * 4);

    const inputIds = ws.alloc([seqLen], "I32", undefined, TensorParallelism.Replicated) as ParallelTensor;

    table.h2d(f32ToBf16Bytes(tableF32));
    inputIds.h2d(idsBuf);
    po.synchronize();

    using out = table.embedding(inputIds, hidden, seqLen);
    po.synchronize();

    const outBuf = Buffer.alloc(seqLen * hidden * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < seqLen * hidden; i++) {
      const relErr = Math.abs(actual[i] - expectedF32[i]) / Math.max(Math.abs(expectedF32[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expectedF32[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });
});

describe("ParallelOps.argmax", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("argmax on Replicated tensor (BF16)", () => {
    const batch = 2;
    const dim = 8;

    const inputF32 = new Float32Array(batch * dim);
    inputF32[3] = 10;
    inputF32[5 + dim] = 7;

    const input = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;

    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    using out = input.argmax();
    po.synchronize();

    const outBuf = Buffer.alloc(batch * 4);
    out.d2h(outBuf);
    assert.equal(outBuf.readInt32LE(0), 3, "batch 0 argmax should be 3");
    assert.equal(outBuf.readInt32LE(4), 5, "batch 1 argmax should be 5");
  });

  it("argmax auto-allGathers Row tensor (BF16)", () => {
    const batch = 2;
    const dim = 8;

    const inputF32 = new Float32Array(batch * dim);
    inputF32[3] = 10;
    inputF32[5 + dim] = 7;

    const input = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;

    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    using out = input.argmax();
    po.synchronize();

    const outBuf = Buffer.alloc(batch * 4);
    out.d2h(outBuf);
    assert.equal(outBuf.readInt32LE(0), 3, "batch 0 argmax should be 3 after allGather");
    assert.equal(outBuf.readInt32LE(4), 5, "batch 1 argmax should be 5 after allGather");
  });
});

describe("ParallelTensor.max", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("max on Replicated BF16 tensor", () => {
    const batch = 4;
    const dim = 8;

    const inputF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = (i % 13 - 6) * 0.5;
    inputF32[3] = 100;
    inputF32[dim + 7] = 99;
    inputF32[2 * dim + 1] = 98;
    inputF32[3 * dim + 5] = 97;

    const input = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const { values, indices } = input.max();
    po.synchronize();

    assert.equal((values as ParallelTensor).parallelism, TensorParallelism.Replicated);
    assert.equal((indices as ParallelTensor).parallelism, TensorParallelism.Replicated);

    const valuesBuf = Buffer.alloc(batch * 2);
    const indicesBuf = Buffer.alloc(batch * 4);
    values.d2h(valuesBuf);
    indices.d2h(indicesBuf);
    const valuesArr = bf16BytesToF32(valuesBuf);

    assert.ok(Math.abs(valuesArr[0] - 100) < 0.1, `batch 0 value: ${valuesArr[0]}`);
    assert.ok(Math.abs(valuesArr[1] - 99) < 0.1, `batch 1 value: ${valuesArr[1]}`);
    assert.ok(Math.abs(valuesArr[2] - 98) < 0.1, `batch 2 value: ${valuesArr[2]}`);
    assert.ok(Math.abs(valuesArr[3] - 97) < 0.1, `batch 3 value: ${valuesArr[3]}`);
    assert.equal(indicesBuf.readInt32LE(0), 3);
    assert.equal(indicesBuf.readInt32LE(4), 7);
    assert.equal(indicesBuf.readInt32LE(8), 1);
    assert.equal(indicesBuf.readInt32LE(12), 5);
  });

  it("max on Row-parallel BF16 tensor returns Replicated result", () => {
    const batch = 4;
    const dim = 8;

    const inputF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = (i % 13 - 6) * 0.5;
    inputF32[3] = 100;
    inputF32[dim + 7] = 99;
    inputF32[2 * dim + 1] = 98;
    inputF32[3 * dim + 5] = 97;

    const input = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const { values, indices } = input.max();
    po.synchronize();

    assert.equal((values as ParallelTensor).parallelism, TensorParallelism.Replicated, "Row max values should be Replicated");
    assert.equal((indices as ParallelTensor).parallelism, TensorParallelism.Replicated, "Row max indices should be Replicated");

    const valuesBuf = Buffer.alloc(batch * 2);
    const indicesBuf = Buffer.alloc(batch * 4);
    values.d2h(valuesBuf);
    indices.d2h(indicesBuf);
    const valuesArr = bf16BytesToF32(valuesBuf);

    assert.ok(Math.abs(valuesArr[0] - 100) < 0.1, `batch 0 value: ${valuesArr[0]}`);
    assert.ok(Math.abs(valuesArr[1] - 99) < 0.1, `batch 1 value: ${valuesArr[1]}`);
    assert.ok(Math.abs(valuesArr[2] - 98) < 0.1, `batch 2 value: ${valuesArr[2]}`);
    assert.ok(Math.abs(valuesArr[3] - 97) < 0.1, `batch 3 value: ${valuesArr[3]}`);
    assert.equal(indicesBuf.readInt32LE(0), 3);
    assert.equal(indicesBuf.readInt32LE(4), 7);
    assert.equal(indicesBuf.readInt32LE(8), 1);
    assert.equal(indicesBuf.readInt32LE(12), 5);
  });

  it("max on Column-parallel BF16 tensor returns Column result", () => {
    const batch = 4;
    const dim = 8;

    const inputF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = (i % 13 - 6) * 0.5;
    inputF32[3] = 100;
    inputF32[dim + 7] = 99;
    inputF32[2 * dim + 1] = 98;
    inputF32[3 * dim + 5] = 97;

    const input = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Column) as ParallelTensor;
    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const { values, indices } = input.max();
    po.synchronize();

    assert.equal((values as ParallelTensor).parallelism, TensorParallelism.Column, "Column max values should be Column");
    assert.equal((indices as ParallelTensor).parallelism, TensorParallelism.Column, "Column max indices should be Column");

    const valuesBuf = Buffer.alloc(batch * 2);
    const indicesBuf = Buffer.alloc(batch * 4);
    values.d2h(valuesBuf);
    indices.d2h(indicesBuf);
    const valuesArr = bf16BytesToF32(valuesBuf);

    assert.ok(Math.abs(valuesArr[0] - 100) < 0.1, `batch 0 value: ${valuesArr[0]}`);
    assert.ok(Math.abs(valuesArr[1] - 99) < 0.1, `batch 1 value: ${valuesArr[1]}`);
    assert.ok(Math.abs(valuesArr[2] - 98) < 0.1, `batch 2 value: ${valuesArr[2]}`);
    assert.ok(Math.abs(valuesArr[3] - 97) < 0.1, `batch 3 value: ${valuesArr[3]}`);
    assert.equal(indicesBuf.readInt32LE(0), 3);
    assert.equal(indicesBuf.readInt32LE(4), 7);
    assert.equal(indicesBuf.readInt32LE(8), 1);
    assert.equal(indicesBuf.readInt32LE(12), 5);
  });

  it("max with offset on Row-parallel BF16 tensor", () => {
    const batch = 2;
    const dim = 8;

    const inputF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = (i % 7 - 3) * 0.5;

    const input = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const offset = 100;
    const { values, indices } = input.max(offset);
    po.synchronize();

    const expectedIndices = new Int32Array(batch);
    const expectedValues = new Float32Array(batch);
    for (let b = 0; b < batch; b++) {
      let maxVal = -Infinity;
      let maxIdx = -1;
      for (let j = 0; j < dim; j++) {
        if (inputF32[b * dim + j] > maxVal) {
          maxVal = inputF32[b * dim + j];
          maxIdx = j;
        }
      }
      expectedValues[b] = maxVal;
      expectedIndices[b] = maxIdx + offset;
    }

    const valuesBuf = Buffer.alloc(batch * 2);
    const indicesBuf = Buffer.alloc(batch * 4);
    values.d2h(valuesBuf);
    indices.d2h(indicesBuf);
    const valuesArr = bf16BytesToF32(valuesBuf);

    for (let b = 0; b < batch; b++) {
      const relErr = Math.abs(valuesArr[b] - expectedValues[b]) / Math.max(Math.abs(expectedValues[b]), 1e-6);
      assert.ok(relErr < 0.05, `batch ${b} value: expected ${expectedValues[b]}, got ${valuesArr[b]}`);
      assert.equal(indicesBuf.readInt32LE(b * 4), expectedIndices[b], `batch ${b} index with offset`);
    }
  });

  it("max on Row-parallel BF16 matches single-GPU reference", () => {
    const batch = 3;
    const dim = 16;

    const inputF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = Math.sin(i * 0.7) * 10;

    const refGlm = new GlmOps(2);
    const refWs = new WorkspaceBase(refGlm);
    const refInput = refWs.alloc([batch, dim], "BF16");
    refInput.h2d(f32ToBf16Bytes(inputF32));
    refGlm.synchronize();

    const { values: refValues, indices: refIndices } = refInput.max();
    refGlm.synchronize();

    const refValuesBuf = Buffer.alloc(batch * 2);
    const refIndicesBuf = Buffer.alloc(batch * 4);
    refValues.d2h(refValuesBuf);
    refIndices.d2h(refIndicesBuf);
    const refValuesArr = bf16BytesToF32(refValuesBuf);

    const input = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Row) as ParallelTensor;
    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const { values, indices } = input.max();
    po.synchronize();

    const valuesBuf = Buffer.alloc(batch * 2);
    const indicesBuf = Buffer.alloc(batch * 4);
    values.d2h(valuesBuf);
    indices.d2h(indicesBuf);
    const valuesArr = bf16BytesToF32(valuesBuf);

    for (let b = 0; b < batch; b++) {
      const relErr = Math.abs(valuesArr[b] - refValuesArr[b]) / Math.max(Math.abs(refValuesArr[b]), 1e-6);
      assert.ok(relErr < 0.05, `batch ${b} value: ref=${refValuesArr[b]}, got=${valuesArr[b]}`);
      assert.equal(indicesBuf.readInt32LE(b * 4), refIndicesBuf.readInt32LE(b * 4), `batch ${b} index`);
    }

    refWs.free();
    refGlm.free();
  });

  it("max on Column-parallel BF16 matches single-GPU reference", () => {
    const batch = 4;
    const dim = 12;

    const inputF32 = new Float32Array(batch * dim);
    for (let i = 0; i < batch * dim; i++) inputF32[i] = Math.cos(i * 0.3) * 5;

    const refGlm = new GlmOps(2);
    const refWs = new WorkspaceBase(refGlm);
    const refInput = refWs.alloc([batch, dim], "BF16");
    refInput.h2d(f32ToBf16Bytes(inputF32));
    refGlm.synchronize();

    const { values: refValues, indices: refIndices } = refInput.max();
    refGlm.synchronize();

    const refValuesBuf = Buffer.alloc(batch * 2);
    const refIndicesBuf = Buffer.alloc(batch * 4);
    refValues.d2h(refValuesBuf);
    refIndices.d2h(refIndicesBuf);
    const refValuesArr = bf16BytesToF32(refValuesBuf);

    const input = ws.alloc([batch, dim], "BF16", undefined, TensorParallelism.Column) as ParallelTensor;
    input.h2d(f32ToBf16Bytes(inputF32));
    po.synchronize();

    const { values, indices } = input.max();
    po.synchronize();

    const valuesBuf = Buffer.alloc(batch * 2);
    const indicesBuf = Buffer.alloc(batch * 4);
    values.d2h(valuesBuf);
    indices.d2h(indicesBuf);
    const valuesArr = bf16BytesToF32(valuesBuf);

    for (let b = 0; b < batch; b++) {
      const relErr = Math.abs(valuesArr[b] - refValuesArr[b]) / Math.max(Math.abs(refValuesArr[b]), 1e-6);
      assert.ok(relErr < 0.05, `batch ${b} value: ref=${refValuesArr[b]}, got=${valuesArr[b]}`);
      assert.equal(indicesBuf.readInt32LE(b * 4), refIndicesBuf.readInt32LE(b * 4), `batch ${b} index`);
    }

    refWs.free();
    refGlm.free();
  });
});

describe("ParallelOps.indexSelect", () => {
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
    po.free();
    glm0.free();
    glm1.free();
  });

  it("indexSelect on Replicated tensors (BF16)", () => {
    const srcRows = 4;
    const srcDim = 8;
    const k = 2;
    const batch = k;

    const srcF32 = new Float32Array(srcRows * srcDim);
    for (let i = 0; i < srcRows * srcDim; i++) srcF32[i] = i * 0.1;

    const indices = new Int32Array([1, 3]);
    const indicesBuf = Buffer.alloc(k * 4);
    for (let i = 0; i < k; i++) indicesBuf.writeInt32LE(indices[i], i * 4);

    const expected = new Float32Array(k * srcDim);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < srcDim; j++) {
        expected[i * srcDim + j] = srcF32[indices[i] * srcDim + j];
      }
    }

    const src = ws.alloc([srcRows, srcDim], "BF16", undefined, TensorParallelism.Replicated) as ParallelTensor;
    const idx = ws.alloc([k], "I32", undefined, TensorParallelism.Replicated) as ParallelTensor;

    src.h2d(f32ToBf16Bytes(srcF32));
    idx.h2d(indicesBuf);
    po.synchronize();

    using out = src.indexSelect(idx, srcDim, batch);
    po.synchronize();

    const outBuf = Buffer.alloc(k * srcDim * 2);
    out.d2h(outBuf);
    const actual = bf16BytesToF32(outBuf);

    for (let i = 0; i < k * srcDim; i++) {
      const relErr = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
      assert.ok(relErr < 0.05, `i=${i}: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
    }
  });
});
