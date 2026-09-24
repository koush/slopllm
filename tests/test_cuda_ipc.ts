import assert from "node:assert/strict";
import path from "node:path";
import { fork } from "node:child_process";
import { describe, it } from "node:test";
import { bf16BytesToF32, f32ToBf16Bytes, GlmOps } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

describe("CUDA IPC arena", () => {
  it("shares an owned arena with a separate process", async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    const arenaGb = 1 / 1024;
    const ops = new GlmOps(deviceId, undefined, arenaGb);
    const workspace = new WorkspaceBase(ops);

    try {
      const shared = workspace.alloc([4], "BF16", "shared");
      shared.h2d(f32ToBf16Bytes(new Float32Array([1, 2, 3, 4])));
      ops.synchronize();

      const child = fork(path.resolve("tests/fixtures/cuda_ipc_child.ts"), [String(deviceId)], {
        execArgv: ["--require", require.resolve("tsx/cjs")],
        stdio: ["ignore", "pipe", "pipe", "ipc"],
        env: {
          ...process.env,
          [`GLM_ARENA_IPC_HANDLE_${deviceId}`]: ops.exportArenaIpcHandle().toString("base64"),
        },
      });
      let stderr = "";
      child.stderr!.setEncoding("utf8");
      child.stderr!.on("data", chunk => { stderr += chunk; });

      const received = await new Promise<Buffer>((resolve, reject) => {
        let receivedMessage = false;
        child.once("error", reject);
        child.once("message", message => {
          receivedMessage = true;
          const values = (message as { received?: unknown }).received;
          if (typeof values !== "string") reject(new Error("CUDA IPC child returned an invalid response"));
          else resolve(Buffer.from(values, "base64"));
        });
        child.once("exit", (code, signal) => {
          if (code !== 0) reject(new Error(`CUDA IPC child exited with ${signal ?? code}: ${stderr}`));
          else if (!receivedMessage) reject(new Error("CUDA IPC child exited without a response"));
        });
      });

      assert.deepEqual([...bf16BytesToF32(received)], [1, 2, 3, 4]);
      const updated = Buffer.alloc(8);
      shared.d2h(updated);
      assert.deepEqual([...bf16BytesToF32(updated)], [2, 4, 6, 8]);
    } finally {
      workspace.free();
      ops.free();
      ops.free();
    }
  });

  it("uses translated IPC arena pointers for direct P2P AllReduce", async () => {
    const deviceIds = [0, 1];
    const arenaGb = 1 / 1024;
    const devices = deviceIds.map(deviceId => new GlmOps(deviceId, undefined, arenaGb));
    const workspaces = devices.map(device => new WorkspaceBase(device));

    try {
      const values = [
        new Float32Array([1, 2, 3, 4]),
        new Float32Array([10, 20, 30, 40]),
      ];
      for (let i = 0; i < devices.length; i++) {
        const shared = workspaces[i].alloc([4], "BF16", "shared");
        shared.h2d(f32ToBf16Bytes(values[i]));
        devices[i].synchronize();
      }

      const child = fork(path.resolve("tests/fixtures/cuda_ipc_p2p_child.ts"), deviceIds.map(String), {
        execArgv: ["--require", require.resolve("tsx/cjs")],
        stdio: ["ignore", "pipe", "pipe", "ipc"],
        env: {
          ...process.env,
          GLM_ARENA_IPC_HANDLE_0: devices[0].exportArenaIpcHandle().toString("base64"),
          GLM_ARENA_IPC_HANDLE_1: devices[1].exportArenaIpcHandle().toString("base64"),
        },
      });
      let stderr = "";
      child.stderr!.setEncoding("utf8");
      child.stderr!.on("data", chunk => { stderr += chunk; });

      const response = await new Promise<{ received: string[]; p2pEnabled: boolean }>((resolve, reject) => {
        let receivedMessage = false;
        child.once("error", reject);
        child.once("message", message => {
          receivedMessage = true;
          const response = message as { received?: unknown; p2pEnabled?: unknown };
          if (!Array.isArray(response.received) || response.received.some(value => typeof value !== "string") || typeof response.p2pEnabled !== "boolean") {
            reject(new Error("CUDA IPC P2P child returned an invalid response"));
          } else {
            resolve(response as { received: string[]; p2pEnabled: boolean });
          }
        });
        child.once("exit", (code, signal) => {
          if (code !== 0) reject(new Error(`CUDA IPC P2P child exited with ${signal ?? code}: ${stderr}`));
          else if (!receivedMessage) reject(new Error("CUDA IPC P2P child exited without a response"));
        });
      });

      assert.equal(response.p2pEnabled, true);
      for (const encoded of response.received) {
        assert.deepEqual([...bf16BytesToF32(Buffer.from(encoded, "base64"))], [11, 22, 33, 44]);
      }
    } finally {
      for (const workspace of workspaces) workspace.free();
      for (const device of devices) device.free();
    }
  });
});
