import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { TensorParallelism } from "../src/device_ops";
import { WorkspaceBase } from "../src/workspace";

function payload(bytes: number, seed: number): Buffer {
  return Buffer.from(Array.from({ length: bytes }, (_, i) => (i * 31 + (i >> 8) * 17 + seed) & 255));
}

function check(outputs: ParallelTensor[], hosts: Buffer[], worldSize: number): void {
  for (const [i, output] of outputs.entries()) {
    for (let rank = 0; rank < worldSize; rank++) {
      const shard = output.shards[rank];
      const actual = Buffer.alloc(shard.bytes);
      shard.d2h(actual);
      const expected = output.parallelism === TensorParallelism.Column
        ? hosts[i].subarray(rank * shard.bytes, (rank + 1) * shard.bytes)
        : hosts[i];
      assert.deepEqual(actual, expected, `tensor ${i}, rank ${rank}`);
    }
  }
}

for (const worldSize of [2, 8]) {
  for (const p2p of [true, false]) {
    it(`grouped redistribution TP${worldSize} ${p2p ? "P2P" : "NCCL"}, views and graph replay`, () => {
      const devices = Array.from({ length: worldSize }, (_, rank) => new GlmOps(rank));
      const ops = new ParallelOps(devices);
      const ws = new WorkspaceBase(ops);
      let graph: number | undefined;
      let exec: number | undefined;
      try {
        if (p2p) {
          assert.ok(ops.p2pEnabled);
        }
        ops.p2pEnabled = p2p;
        const inputs = [
          ws.alloc([worldSize, 64, 512], "BF16", undefined, TensorParallelism.Row),
          ws.alloc([worldSize, 64, 64], "BF16", undefined, TensorParallelism.Row),
          // Unaligned 15-byte source rows and multi-row owner slices.
          ws.alloc([worldSize * 3, worldSize * 5, 3], "U8", undefined, TensorParallelism.Row),
          ws.alloc([worldSize * 3, 7], "I32", undefined, TensorParallelism.Replicated),
          ws.alloc([worldSize * 3, 9], "U8", undefined, TensorParallelism.Column),
        ] as ParallelTensor[];
        const upload = (seed: number) => inputs.map((input, i) => {
          const host = payload(input.numElements * (input.type === "BF16" ? 2 : input.type === "I32" ? 4 : 1), seed + i);
          input.h2d(host);
          return host;
        });
        const run = () => {
          using stream = ops.withStream(() => ops.toColumnParallelMultiple(inputs, ws));
          stream.streamWaitEvent();
          return stream.result;
        };
        let hosts = upload(1);
        ops.synchronize();
        // Existing all-gather wrapper, including Column input, must still work.
        const gathered = ops.allGatherMultiple(inputs, ws);
        ops.synchronize();
        check(gathered, hosts, worldSize);
        for (const output of gathered) {
          output[Symbol.dispose]();
        }
        for (let i = 0; i < 3; i++) {
          const outputs = run();
          ops.synchronize();
          check(outputs, hosts, worldSize);
          assert.equal(outputs[3].shards[0].view, inputs[3].shards[0]);
          for (const output of outputs) {
            assert.equal(output.parallelism, TensorParallelism.Column);
            output[Symbol.dispose]();
          }
        }
        ops.graphBeginCapture();
        const outputs = run();
        graph = ops.graphEndCapture();
        exec = ops.graphInstantiate(graph);
        for (let iteration = 0; iteration < 5; iteration++) {
          hosts = upload(17 + iteration * 29);
          ops.synchronize();
          ops.graphLaunch(exec);
          ops.synchronize();
          check(outputs, hosts, worldSize);
        }
        ops.graphExecDestroy(exec);
        exec = undefined;
        ops.graphDestroy(graph);
        graph = undefined;
        // Results own their views even after callers release the original inputs.
        for (const input of inputs) {
          input[Symbol.dispose]();
        }
        check(outputs, hosts, worldSize);
        for (const output of outputs) {
          output[Symbol.dispose]();
        }
        using invalid = ws.alloc([worldSize + 1, 8], "U8") as ParallelTensor;
        assert.throws(() => ops.toColumnParallelMultiple([invalid], ws), /divisible/);
        assert.deepEqual(ops.toColumnParallelMultiple([], ws), []);
      } finally {
        if (exec !== undefined) {
          ops.graphExecDestroy(exec);
        }
        if (graph !== undefined) {
          ops.graphDestroy(graph);
        }
        ops.synchronize();
        ws.free();
        ops.free();
        for (const device of devices) {
          device.free();
        }
      }
    });
  }
}
