import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import {
  GlmOps,
  getNativeAddon,
  f32ToBf16Bytes,
  bf16BytesToF32,
  NCCL_BFLOAT16,
  NCCL_FLOAT32,
  NCCL_SUM,
} from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { Tensor } from "../src/tensor";

const GPU_COUNTS = [2, 4, 8];

for (const numGpus of GPU_COUNTS) {
  describe(`NCCL multi-GPU (${numGpus} GPUs)`, () => {
    let devices: GlmOps[];
    let workspaces: WorkspaceBase[];
    let comms: number[];

    before(() => {
      const deviceIds = Array.from({ length: numGpus }, (_, i) => i);
      devices = deviceIds.map(id => new GlmOps(id));
      workspaces = devices.map(glm => new WorkspaceBase(glm));
      comms = getNativeAddon().ncclCommInitAll(deviceIds);
      assert.equal(comms.length, numGpus, "should return one comm per device");
    });

    after(() => {
      for (const comm of comms) {
        getNativeAddon().ncclCommDestroy(comm);
      }
      for (const ws of workspaces) ws.free();
      for (const glm of devices) glm.free();
    });

    it("allReduce BF16 sums across all ranks", () => {
      const count = 32;
      const bytes = count * 2;

      const sendGpus: Tensor[] = [];
      const recvGpus: Tensor[] = [];

      for (let rank = 0; rank < numGpus; rank++) {
        const srcF32 = new Float32Array(count);
        for (let i = 0; i < count; i++) srcF32[i] = (rank + 1) * (i + 1) * 0.01;
        const srcBuf = f32ToBf16Bytes(srcF32);

        const sendGpu = workspaces[rank].alloc([bytes], "U8");
        const recvGpu = workspaces[rank].alloc([bytes], "U8");
        sendGpu.h2d(srcBuf);

        sendGpus.push(sendGpu);
        recvGpus.push(recvGpu);
      }

      getNativeAddon().ncclGroupStart();
      for (let rank = 0; rank < numGpus; rank++) {
        getNativeAddon().ncclAllReduce(
          comms[rank], devices[rank].ctx,
          sendGpus[rank].data, recvGpus[rank].data,
          count, NCCL_BFLOAT16, NCCL_SUM
        );
      }
      getNativeAddon().ncclGroupEnd();

      for (let rank = 0; rank < numGpus; rank++) {
        devices[rank].synchronize();
      }

      const expected = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        for (let rank = 0; rank < numGpus; rank++) {
          expected[i] += (rank + 1) * (i + 1) * 0.01;
        }
      }

      for (let rank = 0; rank < numGpus; rank++) {
        const dstBuf = Buffer.alloc(bytes);
        recvGpus[rank].d2h(dstBuf, bytes);
        const dstF32 = bf16BytesToF32(dstBuf);
        for (let i = 0; i < count; i++) {
          const relErr = Math.abs(dstF32[i] - expected[i]) / Math.max(Math.abs(expected[i]), 1e-6);
          assert.ok(
            relErr < 0.02,
            `rank=${rank} idx=${i}: expected ${expected[i]}, got ${dstF32[i]} (relErr=${relErr})`
          );
        }
      }
    });

    it("allReduce F32 sums across all ranks", () => {
      const count = 16;
      const bytes = count * 4;

      const sendGpus: Tensor[] = [];
      const recvGpus: Tensor[] = [];

      for (let rank = 0; rank < numGpus; rank++) {
        const srcF32 = new Float32Array(count);
        for (let i = 0; i < count; i++) srcF32[i] = (rank + 1) * 0.1;
        const srcBuf = Buffer.from(srcF32.buffer);

        const sendGpu = workspaces[rank].alloc([bytes], "U8");
        const recvGpu = workspaces[rank].alloc([bytes], "U8");
        sendGpu.h2d(srcBuf);

        sendGpus.push(sendGpu);
        recvGpus.push(recvGpu);
      }

      getNativeAddon().ncclGroupStart();
      for (let rank = 0; rank < numGpus; rank++) {
        getNativeAddon().ncclAllReduce(
          comms[rank], devices[rank].ctx,
          sendGpus[rank].data, recvGpus[rank].data,
          count, NCCL_FLOAT32, NCCL_SUM
        );
      }
      getNativeAddon().ncclGroupEnd();

      for (let rank = 0; rank < numGpus; rank++) {
        devices[rank].synchronize();
      }

      const expected = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        for (let rank = 0; rank < numGpus; rank++) {
          expected[i] += (rank + 1) * 0.1;
        }
      }

      for (let rank = 0; rank < numGpus; rank++) {
        const dstBuf = Buffer.alloc(bytes);
        recvGpus[rank].d2h(dstBuf, bytes);
        const dstF32 = new Float32Array(dstBuf.buffer, dstBuf.byteOffset, count);
        for (let i = 0; i < count; i++) {
          assert.ok(
            Math.abs(dstF32[i] - expected[i]) < 1e-4,
            `rank=${rank} idx=${i}: expected ${expected[i]}, got ${dstF32[i]}`
          );
        }
      }
    });

    it("allGather BF16 collects all ranks", () => {
      const sendCount = 8;
      const recvCount = sendCount * numGpus;
      const sendBytes = sendCount * 2;
      const recvBytes = recvCount * 2;

      const sendGpus: Tensor[] = [];
      const recvGpus: Tensor[] = [];

      for (let rank = 0; rank < numGpus; rank++) {
        const srcF32 = new Float32Array(sendCount);
        for (let i = 0; i < sendCount; i++) srcF32[i] = rank + i;
        const srcBuf = f32ToBf16Bytes(srcF32);

        const sendGpu = workspaces[rank].alloc([sendBytes], "U8");
        const recvGpu = workspaces[rank].alloc([recvBytes], "U8");
        sendGpu.h2d(srcBuf);

        sendGpus.push(sendGpu);
        recvGpus.push(recvGpu);
      }

      getNativeAddon().ncclGroupStart();
      for (let rank = 0; rank < numGpus; rank++) {
        getNativeAddon().ncclAllGather(
          comms[rank], devices[rank].ctx,
          sendGpus[rank].data, recvGpus[rank].data,
          sendCount, NCCL_BFLOAT16
        );
      }
      getNativeAddon().ncclGroupEnd();

      for (let rank = 0; rank < numGpus; rank++) {
        devices[rank].synchronize();
      }

      for (let rank = 0; rank < numGpus; rank++) {
        const dstBuf = Buffer.alloc(recvBytes);
        recvGpus[rank].d2h(dstBuf, recvBytes);
        const dstF32 = bf16BytesToF32(dstBuf);
        for (let r = 0; r < numGpus; r++) {
          for (let i = 0; i < sendCount; i++) {
            const expected = r + i;
            const actual = dstF32[r * sendCount + i];
            assert.ok(
              Math.abs(actual - expected) < 0.01,
              `rank=${rank} srcRank=${r} idx=${i}: expected ${expected}, got ${actual}`
            );
          }
        }
      }
    });

    it("allGather F32 collects all ranks", () => {
      const sendCount = 8;
      const recvCount = sendCount * numGpus;
      const sendBytes = sendCount * 4;
      const recvBytes = recvCount * 4;

      const sendGpus: Tensor[] = [];
      const recvGpus: Tensor[] = [];

      for (let rank = 0; rank < numGpus; rank++) {
        const srcF32 = new Float32Array(sendCount);
        for (let i = 0; i < sendCount; i++) srcF32[i] = rank * 100 + i;
        const srcBuf = Buffer.from(srcF32.buffer);

        const sendGpu = workspaces[rank].alloc([sendBytes], "U8");
        const recvGpu = workspaces[rank].alloc([recvBytes], "U8");
        sendGpu.h2d(srcBuf);

        sendGpus.push(sendGpu);
        recvGpus.push(recvGpu);
      }

      getNativeAddon().ncclGroupStart();
      for (let rank = 0; rank < numGpus; rank++) {
        getNativeAddon().ncclAllGather(
          comms[rank], devices[rank].ctx,
          sendGpus[rank].data, recvGpus[rank].data,
          sendCount, NCCL_FLOAT32
        );
      }
      getNativeAddon().ncclGroupEnd();

      for (let rank = 0; rank < numGpus; rank++) {
        devices[rank].synchronize();
      }

      for (let rank = 0; rank < numGpus; rank++) {
        const dstBuf = Buffer.alloc(recvBytes);
        recvGpus[rank].d2h(dstBuf, recvBytes);
        const dstF32 = new Float32Array(dstBuf.buffer, dstBuf.byteOffset, recvCount);
        for (let r = 0; r < numGpus; r++) {
          for (let i = 0; i < sendCount; i++) {
            const expected = r * 100 + i;
            const actual = dstF32[r * sendCount + i];
            assert.ok(
              Math.abs(actual - expected) < 1e-6,
              `rank=${rank} srcRank=${r} idx=${i}: expected ${expected}, got ${actual}`
            );
          }
        }
      }
    });
  });
}
