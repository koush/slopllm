import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { TensorParallelism } from "../src/device_ops";
import { WorkspaceBase } from "../src/workspace";
import type { ExecutionState } from "../src/execution-workspace";

for (const mode of ["single", "tp", "cp"]) {
  for (const rows of [1, 8, 65]) {
    it(`factorized MLA Q ${mode}, rows=${rows}, matches two-stage CPU reference`, () => {
      const devices = Array.from({ length: mode === "single" ? 1 : 2 }, (_, rank) => new GlmOps(rank));
      const ops = mode === "single" ? devices[0] : new ParallelOps(devices);
      const ws = new WorkspaceBase(ops);
      try {
        const heads = 4, nope = 8, latent = 16, rank = 16, rope = 4;
        const weightPar = mode === "single" ? TensorParallelism.Replicated : TensorParallelism.Column;
        using x = ws.alloc([rows, rank], "BF16");
        using wq = ws.alloc([heads * nope, rank], "BF16", undefined, weightPar);
        using wk = ws.alloc([heads * nope, latent], "BF16", undefined, weightPar);
        using wp = ws.alloc([heads * rope, rank], "BF16", undefined, weightPar);
        using cos = ws.alloc([rows, rope], "BF16");
        using sin = ws.alloc([rows, rope], "BF16");
        using kv = ws.alloc([2, 8, latent], "BF16", undefined,
          mode === "cp" ? TensorParallelism.Row : TensorParallelism.Replicated);
        const xv = Float32Array.from({ length: rows * rank }, (_, i) => (i * 7 % 5) - 2);
        const qv = Float32Array.from({ length: heads * nope * rank }, (_, i) => (i * 3 + Math.floor(i / rank)) % 3 - 1);
        const kvs = Float32Array.from({ length: heads * nope * latent }, (_, i) => (i + Math.floor(i / latent)) % 3 - 1);
        const pv = Float32Array.from({ length: heads * rope * rank }, (_, i) => (i + Math.floor(i / rank)) % 3 - 1);
        x.h2d(f32ToBf16Bytes(xv));
        wq.h2d(f32ToBf16Bytes(qv));
        wk.h2d(f32ToBf16Bytes(kvs));
        wp.h2d(f32ToBf16Bytes(pv));
        cos.h2d(f32ToBf16Bytes(new Float32Array(rows * rope).fill(1)));
        sin.h2d(f32ToBf16Bytes(new Float32Array(rows * rope)));
        const expected = new Float32Array(rows * heads * latent);
        const expectedPe = new Float32Array(rows * heads * rope);
        for (let m = 0; m < rows; m++) {
          for (let h = 0; h < heads; h++) {
            for (let d = 0; d < nope; d++) {
              let q = 0;
              for (let k = 0; k < rank; k++) {
                q += xv[m * rank + k] * qv[(h * nope + d) * rank + k];
              }
              for (let v = 0; v < latent; v++) {
                expected[(m * heads + h) * latent + v] += q * kvs[(h * nope + d) * latent + v];
              }
            }
            for (let d = 0; d < rope; d++) {
              for (let k = 0; k < rank; k++) {
                expectedPe[(m * heads + h) * rope + d] += xv[m * rank + k] * pv[(h * rope + d) * rank + k];
              }
            }
          }
        }
        const state = { isDecode: rows === 1, cache: { getPagedKV: () => ({ sparseMode: false }) } } as unknown as ExecutionState;
        using absorbed = wk.bmm(wq, heads, latent, rank, nope, true, false);
        using stream = ops.withStream(() => ops.projectMlaQuery(state, kv.parallelism, x, wp, wq, wk, absorbed, cos, sin,
          rope, latent, heads, rows, 1, true));
        stream.streamWaitEvent();
        using q = stream.result.qAbsorbed;
        using pe = stream.result.qPe;
        ops.synchronize();
        for (const [tensor, reference, dim] of [[q, expected, latent], [pe, expectedPe, rope]] as const) {
          const shards = tensor instanceof ParallelTensor ? tensor.shards : [tensor];
          const sharded = mode === "tp";
          for (let gpu = 0; gpu < shards.length; gpu++) {
            const localHeads = sharded ? heads / 2 : heads;
            const local = new Float32Array(rows * localHeads * dim);
            for (let m = 0; m < rows; m++) {
              const start = (m * heads + (sharded ? gpu * localHeads : 0)) * dim;
              local.set(reference.subarray(start, start + localHeads * dim), m * localHeads * dim);
            }
            const actual = Buffer.alloc(shards[gpu].bytes);
            shards[gpu].d2h(actual);
            assert.deepEqual(actual, f32ToBf16Bytes(local));
          }
        }
      } finally {
        ops.synchronize();
        ws.free();
        if (ops instanceof ParallelOps) {
          ops.free();
        }
        for (const device of devices) {
          device.free();
        }
      }
    });
  }
}
