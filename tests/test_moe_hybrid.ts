import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { getNativeAddon } from "../src/native-addon";
import { Tensor } from "../src/tensor";

function bf16(values: number[]): Buffer {
  const out = Buffer.alloc(values.length * 2);
  const f32 = new Float32Array(1);
  const bits = new Uint32Array(f32.buffer);
  values.forEach((value, i) => {
    f32[0] = value;
    out.writeUInt16LE(((bits[0] + 0x7fff + ((bits[0] >>> 16) & 1)) >>> 16) & 0xffff, i * 2);
  });
  return out;
}

function read(tensor: Tensor, ops: GlmOps): Buffer {
  const result = Buffer.alloc(tensor.bytes);
  tensor.d2h(result);
  ops.synchronize();
  return result;
}

for (const [count, N, K] of [[1, 130, 128], [2, 130, 128], [4, 130, 6144], [8, 130, 128], [64, 130, 128], [64, 130, 6144], [512, 256, 6144], [512, 6144, 256]]) {
  it(`hybrid MoE partitions disjoint outputs and replays changing routes: ${count}x${N}x${K}`, () => {
    using ops = new GlmOps(0);
    using ws = new WorkspaceBase(ops);
    const addon = getNativeAddon();
    const experts = 256;
    // Down gathers a distinct activation per route, rather than per token.
    const topK = K === 256 ? 1 : Math.min(8, count);
    const inputs = Array.from({ length: count / topK * K }, (_, i) => ((i * 7) % 23 - 11) / 32);
    using input = ws.alloc([count / topK, K], "BF16");
    input.h2d(bf16(inputs));
    const packed = Array.from({ length: experts }, (_, e) => {
      const data = Buffer.alloc(N * K / 2);
      for (let i = 0; i < data.length; i++) {
        data[i] = ((i * 13 + e * 7) % 16) | (((i * 3 + e * 11 + 1) % 16) << 4);
      }
      return data;
    });
    const weights = packed.map(data => {
      const tensor = ws.alloc([N, K / 2], "U8");
      tensor.h2d(data);
      return tensor;
    });
    const scales = weights.map((_, e) => {
      const tensor = ws.alloc([N, K / 16], "F8_E4M3");
      tensor.h2d(Buffer.alloc(tensor.bytes, [0x30, 0x38, 0x40][e % 3]));
      return tensor;
    });
    const scales2 = weights.map((_, e) => {
      const tensor = ws.alloc([1], "F32");
      const bytes = Buffer.alloc(4);
      bytes.writeFloatLE([0.5, 1, 2][e % 3]);
      tensor.h2d(bytes);
      return tensor;
    });
    using wp = ws.alloc([experts], "I64");
    using sp = ws.alloc([experts], "I64");
    using s2p = ws.alloc([experts], "I64");
    wp.writePointers(weights);
    sp.writePointers(scales);
    s2p.writePointers(scales2);
    using ids = ws.alloc([count], "I32");
    using output = ws.alloc([count, N], "BF16");
    using reference = ws.alloc([count, N], "BF16");
    using plan = ws.allocRaw(addon.moeHybridWorkspaceSize(count, N));
    const args = [ops.ctx, output.data, input.data, wp.data, sp.data, s2p.data,
      ids.data, topK, count, N, K, plan.data] as const;
    const boundaryRoutes = [7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33]
      .flatMap((size, e) => new Array<number>(size).fill(e));
    const patterns = [
      Array.from({ length: count }, (_, i) => i < 9 ? 0 : i < 26 ? 1 : i < 33 ? 2 : 3 + (i - 33) % 253),
      Array.from({ length: count }, (_, i) => i % experts),
      Array.from({ length: count }, (_, i) => i % 8),
      // Maximum number of MMA tasks: every occupied expert has two rows.
      Array.from({ length: count }, (_, i) => Math.floor(i / 2)),
      Array.from({ length: count }, (_, i) => boundaryRoutes[i] ?? 12 + (i - boundaryRoutes.length) % 244),
    ];
    const poison = Buffer.alloc(output.bytes, 0xff);
    for (const threshold of [1, 2, 4, 8, 16, 513]) {
      const run = () => {
        addon.moeHybridPrepare(ops.ctx, ids.data, count, experts, threshold, plan.data);
        using mma = ops.withStream(true, () => {
          addon.moeHybridMma(...args);
          addon.moeHybridMmaReduce(ops.ctx, output.data, count, N, K, plan.data);
        });
        using cuda = ops.withStream(() => addon.moeHybridCuda(...args));
        mma.streamWaitEvent();
        cuda.streamWaitEvent();
      };
      ids.h2d(Buffer.from(new Int32Array(patterns[0]).buffer));
      run();
      ops.synchronize();
      ops.graphBeginCapture();
      run();
      const graph = ops.graphEndCapture();
      const exec = ops.graphInstantiate(graph);
      try {
        for (const routes of patterns) {
          ids.h2d(Buffer.from(new Int32Array(routes).buffer));
          // Every replay must overwrite its owned partials; no initialization
          // or values left behind by a previous route partition may be needed.
          plan.h2d(Buffer.alloc(plan.bytes, 0xff));
          output.h2d(poison);
          addon.nvfp4MulMatId(ops.ctx, reference.data, input.data, wp.data, sp.data, s2p.data,
            ids.data, topK, count, N, K);
          ops.graphLaunch(exec);
          const want = read(reference, ops);
          assert.deepEqual(read(output, ops), want, `threshold ${threshold}: graph result`);
          if (K === 256 && N === 6144) {
            using routing = ws.alloc([count], "BF16");
            routing.h2d(bf16(Array.from({ length: count }, (_, i) => (1 + (i * 3) % 8) / 16)));
            using combined = ws.alloc([count / 8, N], "BF16");
            using fused = ws.alloc([count / 8, N], "BF16");
            addon.scatterAddRows(ops.ctx, combined.data, output.data, routing.data, 8, N, count / 8, 0);
            addon.nvfp4MulMatIdReduce(ops.ctx, fused.data, input.data, wp.data, sp.data, s2p.data,
              ids.data, routing.data, count / 8);
            assert.deepEqual(read(combined, ops), read(fused, ops), "hybrid down plus weighted combine matches fused down");
          }
          if (K === 128) {
            const fp4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6];
            const expected: number[] = [];
            for (let route = 0; route < count; route++) {
              const e = routes[route];
              for (let n = 0; n < N; n++) {
                let sum = 0;
                for (let k = 0; k < K; k++) {
                  const byte = packed[e][n * K / 2 + Math.floor(k / 2)];
                  const w = fp4[(byte >> ((k % 2) * 4)) & 15];
                  sum += inputs[Math.floor(route / topK) * K + k] * w * [0.25, 1, 4][e % 3];
                }
                expected.push(sum);
              }
            }
            assert.deepEqual(want, bf16(expected), "independent FP4 CPU reference");
          }
          const occupancy = new Array(experts).fill(0);
          routes.forEach(e => occupancy[e]++);
          // Independently run each consumer against poison: no row may be
          // missing, written by both paths, or clobbered outside its partition.
          const ownership: boolean[][] = [];
          for (const mma of [false, true]) {
            output.h2d(poison);
            if (mma) {
              addon.moeHybridMma(...args);
              addon.moeHybridMmaReduce(ops.ctx, output.data, count, N, K, plan.data);
            }
            else {
              addon.moeHybridCuda(...args);
            }
            const actual = read(output, ops);
            const written: boolean[] = [];
            for (let route = 0; route < count; route++) {
              const start = route * N * 2;
              const end = start + N * 2;
              const owns = !actual.subarray(start, end).equals(poison.subarray(start, end));
              written.push(owns);
              assert.deepEqual(actual.subarray(start, end), (owns ? want : poison).subarray(start, end));
            }
            ownership.push(written);
          }
          const mmaRows = new Array(experts).fill(0);
          for (let route = 0; route < count; route++) {
            assert.notEqual(ownership[0][route], ownership[1][route], "exactly one consumer owns each row");
            if (ownership[1][route]) {
              mmaRows[routes[route]]++;
            }
          }
          for (let e = 0; e < experts; e++) {
            const rows = occupancy[e];
            const expectedMmaRows = rows < Math.max(2, threshold) ? 0 : rows;
            assert.equal(mmaRows[e], expectedMmaRows, `expert ${e}, occupancy ${rows}: correct partition`);
          }
        }
      }
      finally {
        ops.graphExecDestroy(exec);
        ops.graphDestroy(graph);
      }
    }
  });
}

it("hybrid MoE rejects unsupported dimensions before touching device pointers", () => {
  const addon = getNativeAddon();
  assert.throws(() => addon.moeHybridPrepare(0, 0, 513, 256, 8, 0), /1\.\.512/);
  assert.throws(() => addon.moeHybridPrepare(0, 0, 512, 257, 8, 0), /1\.\.256/);
  assert.throws(() => addon.moeHybridPrepare(0, 0, 512, 256, 0, 0), /minRows/);
  assert.throws(() => addon.moeHybridWorkspaceSize(513, 256), /dimensions/);
  assert.throws(() => addon.moeHybridMmaReduce(0, 0, 513, 256, 6144, 0), /dimensions/);
  for (const fn of [addon.moeHybridCuda, addon.moeHybridMma]) {
    assert.throws(() => fn(0, 0, 0, 0, 0, 0, 0, 8, 512, 256, 80, 0), /dimensions/);
  }
});
