import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, getNativeAddon, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

const NUM_GPUS = Math.min(8, parseInt(process.env.GLM_GPUS || "2"));
const NCCL_BFLOAT16 = 9;

describe("P2P AllReduce Smem", () => {
  const devices: GlmOps[] = [];
  const addon = getNativeAddon();

  before(() => {
    for (let i = 0; i < NUM_GPUS; i++) {
      devices.push(new GlmOps(i));
    }
    for (const d of devices) {
      for (const d2 of devices) {
        if (d.device !== d2.device) {
          addon.p2pEnablePeerAccess(d.ctx, d2.device);
        }
      }
    }
  });

  after(() => {
    for (const d of devices) {
      d.free();
    }
  });

  function readBf16(ctx: number, data: number, count: number): Float32Array {
    const bytes = count * 2;
    const buf = Buffer.alloc(bytes);
    addon.d2h(ctx, buf, data, bytes);
    return bf16BytesToF32(buf);
  }

  it("2-GPU BF16 allreduce matches expected sum", () => {
    if (NUM_GPUS < 2) return;
    const worldSize = 2;
    const count = 512;

    const instances: number[] = [];
    const flagPtrs: number[] = [];
    const workspaces: WorkspaceBase[] = [];
    const inputPtrs: number[] = [];
    const outputPtrs: number[] = [];

    for (let rank = 0; rank < worldSize; rank++) {
      const ws = new WorkspaceBase(devices[rank]);
      workspaces.push(ws);
      const inst = addon.p2pCreateInstance(devices[rank].ctx, rank, worldSize);
      instances.push(inst);
      flagPtrs.push(addon.p2pGetFlagPtr(inst));
    }

    for (let rank = 0; rank < worldSize; rank++) {
      const values = new Float32Array(count);
      for (let i = 0; i < count; i++) values[i] = rank + 1;
      const buf = f32ToBf16Bytes(values);
      const inTensor = workspaces[rank].alloc([count * 2], "U8");
      inTensor.h2d(buf);
      inputPtrs.push(inTensor.data);

      const outTensor = workspaces[rank].alloc([count * 2], "U8");
      outputPtrs.push(outTensor.data);
    }

    const padPtrs = [...inputPtrs];
    while (padPtrs.length < 8) padPtrs.push(0);

    for (let rank = 0; rank < worldSize; rank++) {
      addon.p2pAllReduceSmem(
        devices[rank].ctx, instances[rank],
        padPtrs[0], padPtrs[1], padPtrs[2], padPtrs[3],
        padPtrs[4], padPtrs[5], padPtrs[6], padPtrs[7],
        outputPtrs[rank], worldSize, count, NCCL_BFLOAT16
      );
    }

    for (const d of devices) d.synchronize();

    const expectedSum = (worldSize * (worldSize + 1)) / 2;
    for (let rank = 0; rank < worldSize; rank++) {
      const result = readBf16(devices[rank].ctx, outputPtrs[rank], count);
      for (let i = 0; i < count; i++) {
        assert.ok(
          Math.abs(result[i] - expectedSum) < 0.05,
          `rank=${rank} idx=${i}: expected ${expectedSum}, got ${result[i]}`
        );
      }
    }

    for (let rank = 0; rank < worldSize; rank++) {
      addon.p2pDestroyInstance(instances[rank]);
      workspaces[rank].free();
    }
  });

  it("2-GPU BF16 allreduce with larger tensor", () => {
    if (NUM_GPUS < 2) return;
    const worldSize = 2;
    const count = 8192;

    const instances: number[] = [];
    const flagPtrs: number[] = [];
    const workspaces: WorkspaceBase[] = [];
    const inputPtrs: number[] = [];
    const outputPtrs: number[] = [];

    for (let rank = 0; rank < worldSize; rank++) {
      const ws = new WorkspaceBase(devices[rank]);
      workspaces.push(ws);
      const inst = addon.p2pCreateInstance(devices[rank].ctx, rank, worldSize);
      instances.push(inst);
      flagPtrs.push(addon.p2pGetFlagPtr(inst));
    }

    for (let rank = 0; rank < worldSize; rank++) {
      const values = new Float32Array(count);
      for (let i = 0; i < count; i++) values[i] = (rank + 1) * (i + 1) * 0.001;
      const buf = f32ToBf16Bytes(values);
      const inTensor = workspaces[rank].alloc([count * 2], "U8");
      inTensor.h2d(buf);
      inputPtrs.push(inTensor.data);

      const outTensor = workspaces[rank].alloc([count * 2], "U8");
      outputPtrs.push(outTensor.data);
    }

    const padPtrs = [...inputPtrs];
    while (padPtrs.length < 8) padPtrs.push(0);

    for (let rank = 0; rank < worldSize; rank++) {
      addon.p2pAllReduceSmem(
        devices[rank].ctx, instances[rank],
        padPtrs[0], padPtrs[1], padPtrs[2], padPtrs[3],
        padPtrs[4], padPtrs[5], padPtrs[6], padPtrs[7],
        outputPtrs[rank], worldSize, count, NCCL_BFLOAT16
      );
    }

    for (const d of devices) d.synchronize();

    for (let rank = 0; rank < worldSize; rank++) {
      const result = readBf16(devices[rank].ctx, outputPtrs[rank], count);
      for (let i = 0; i < count; i++) {
        const expected = (1 + 2) * (i + 1) * 0.001;
        assert.ok(
          Math.abs(result[i] - expected) < 0.05,
          `rank=${rank} idx=${i}: expected ${expected}, got ${result[i]}`
        );
      }
    }

    for (let rank = 0; rank < worldSize; rank++) {
      addon.p2pDestroyInstance(instances[rank]);
      workspaces[rank].free();
    }
  });
});
