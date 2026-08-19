import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import {
  GlmOps,
  f32ToBf16Bytes,
  bf16BytesToF32,
  NCCL_BFLOAT16,
  NCCL_FLOAT32,
} from "../src/glm_ops";
import { getNativeAddon } from "../src/native-addon";
import { WorkspaceBase } from "../src/workspace";
import { Tensor } from "../src/tensor";

const GPU_COUNTS = [2, 4, 8];

for (const numGpus of GPU_COUNTS) {
  describe(`NCCL Send/Recv (${numGpus} GPUs)`, () => {
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

    it("send/recv BF16 ring: each rank sends to next, receives from prev", () => {
      const count = 64;
      const bytes = count * 2;

      const sendGpus: Tensor[] = [];
      const recvGpus: Tensor[] = [];

      for (let rank = 0; rank < numGpus; rank++) {
        const srcF32 = new Float32Array(count);
        for (let i = 0; i < count; i++) srcF32[i] = rank * 1000 + i;
        const srcBuf = f32ToBf16Bytes(srcF32);

        const sendGpu = workspaces[rank].alloc([bytes], "U8");
        const recvGpu = workspaces[rank].alloc([bytes], "U8");
        sendGpu.h2d(srcBuf);

        sendGpus.push(sendGpu);
        recvGpus.push(recvGpu);
      }

      getNativeAddon().ncclGroupStart();
      for (let rank = 0; rank < numGpus; rank++) {
        const sendPeer = (rank + 1) % numGpus;
        const recvPeer = (rank - 1 + numGpus) % numGpus;
        getNativeAddon().ncclSend(comms[rank], devices[rank].ctx, sendGpus[rank].data, count, NCCL_BFLOAT16, sendPeer);
        getNativeAddon().ncclRecv(comms[rank], devices[rank].ctx, recvGpus[rank].data, count, NCCL_BFLOAT16, recvPeer);
      }
      getNativeAddon().ncclGroupEnd();

      for (let rank = 0; rank < numGpus; rank++) {
        devices[rank].synchronize();
      }

      for (let rank = 0; rank < numGpus; rank++) {
        const recvPeer = (rank - 1 + numGpus) % numGpus;
        const dstBuf = Buffer.alloc(bytes);
        recvGpus[rank].d2h(dstBuf, bytes);
        const dstF32 = bf16BytesToF32(dstBuf);
        const expectedF32 = new Float32Array(count);
        for (let i = 0; i < count; i++) expectedF32[i] = recvPeer * 1000 + i;
        const expectedBuf = f32ToBf16Bytes(expectedF32);
        const expectedBf16 = bf16BytesToF32(expectedBuf);
        for (let i = 0; i < count; i++) {
          assert.ok(
            Math.abs(dstF32[i] - expectedBf16[i]) < 0.01,
            `rank=${rank} idx=${i}: expected ${expectedBf16[i]}, got ${dstF32[i]}`
          );
        }
      }
    });

    it("send/recv F32 ring: each rank sends to next, receives from prev", () => {
      const count = 32;
      const bytes = count * 4;

      const sendGpus: Tensor[] = [];
      const recvGpus: Tensor[] = [];

      for (let rank = 0; rank < numGpus; rank++) {
        const srcF32 = new Float32Array(count);
        for (let i = 0; i < count; i++) srcF32[i] = rank * 100 + i * 0.5;
        const srcBuf = Buffer.from(srcF32.buffer);

        const sendGpu = workspaces[rank].alloc([bytes], "U8");
        const recvGpu = workspaces[rank].alloc([bytes], "U8");
        sendGpu.h2d(srcBuf);

        sendGpus.push(sendGpu);
        recvGpus.push(recvGpu);
      }

      getNativeAddon().ncclGroupStart();
      for (let rank = 0; rank < numGpus; rank++) {
        const sendPeer = (rank + 1) % numGpus;
        const recvPeer = (rank - 1 + numGpus) % numGpus;
        getNativeAddon().ncclSend(comms[rank], devices[rank].ctx, sendGpus[rank].data, count, NCCL_FLOAT32, sendPeer);
        getNativeAddon().ncclRecv(comms[rank], devices[rank].ctx, recvGpus[rank].data, count, NCCL_FLOAT32, recvPeer);
      }
      getNativeAddon().ncclGroupEnd();

      for (let rank = 0; rank < numGpus; rank++) {
        devices[rank].synchronize();
      }

      for (let rank = 0; rank < numGpus; rank++) {
        const recvPeer = (rank - 1 + numGpus) % numGpus;
        const dstBuf = Buffer.alloc(bytes);
        recvGpus[rank].d2h(dstBuf, bytes);
        const dstF32 = new Float32Array(dstBuf.buffer, dstBuf.byteOffset, count);
        for (let i = 0; i < count; i++) {
          const expected = recvPeer * 100 + i * 0.5;
          assert.ok(
            Math.abs(dstF32[i] - expected) < 1e-6,
            `rank=${rank} idx=${i}: expected ${expected}, got ${dstF32[i]}`
          );
        }
      }
    });

    it("send/recv butterfly: XOR peer exchange", () => {
      const count = 16;
      const bytes = count * 2;

      const sendGpus: Tensor[] = [];
      const recvGpus: Tensor[] = [];

      for (let rank = 0; rank < numGpus; rank++) {
        const srcF32 = new Float32Array(count);
        for (let i = 0; i < count; i++) srcF32[i] = rank * 10 + i;
        const srcBuf = f32ToBf16Bytes(srcF32);

        const sendGpu = workspaces[rank].alloc([bytes], "U8");
        const recvGpu = workspaces[rank].alloc([bytes], "U8");
        sendGpu.h2d(srcBuf);

        sendGpus.push(sendGpu);
        recvGpus.push(recvGpu);
      }

      for (let reduceHalf = numGpus / 2; reduceHalf >= 1; reduceHalf /= 2) {
        getNativeAddon().ncclGroupStart();
        for (let rank = 0; rank < numGpus; rank++) {
          const peer = rank ^ reduceHalf;
          getNativeAddon().ncclSend(comms[rank], devices[rank].ctx, sendGpus[rank].data, count, NCCL_BFLOAT16, peer);
          getNativeAddon().ncclRecv(comms[rank], devices[rank].ctx, recvGpus[rank].data, count, NCCL_BFLOAT16, peer);
        }
        getNativeAddon().ncclGroupEnd();

        for (let rank = 0; rank < numGpus; rank++) {
          devices[rank].synchronize();
        }

        for (let rank = 0; rank < numGpus; rank++) {
          const peer = rank ^ reduceHalf;
          const dstBuf = Buffer.alloc(bytes);
          recvGpus[rank].d2h(dstBuf, bytes);
          const dstF32 = bf16BytesToF32(dstBuf);
          for (let i = 0; i < count; i++) {
            const expected = peer * 10 + i;
            assert.ok(
              Math.abs(dstF32[i] - expected) < 0.01,
              `round half=${reduceHalf} rank=${rank} idx=${i}: expected ${expected}, got ${dstF32[i]}`
            );
          }
        }
      }
    });
  });
}
