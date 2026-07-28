// Correctness of the push-based reduce-scatter allReduce (the default P2P path).
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { TensorParallelism } from "../src/device_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";

function runCase(ids: number[], rows: number, cols: number) {
  const world = ids.length;
  const glms = ids.map(d => new GlmOps(d));
  const po = new ParallelOps(glms);
  const ws = new WorkspaceBase(po);
  try {
    const total = rows * cols;
    // Round inputs to bf16 first so the reference matches what the kernel sums.
    const shards: Float32Array[] = [];
    for (let r = 0; r < world; r++) {
      const a = new Float32Array(total);
      for (let i = 0; i < total; i++) a[i] = Math.sin(i * 0.11 + r) * 0.5 + 1.0;
      shards.push(bf16BytesToF32(f32ToBf16Bytes(a)));
    }

    const pt = ws.alloc([rows, cols], "BF16", undefined, TensorParallelism.PartialSum) as ParallelTensor;
    for (let r = 0; r < world; r++) pt.shard(r).h2d(f32ToBf16Bytes(shards[r]));
    po.synchronize();
    pt.allReduce();
    po.synchronize();

    const expected = new Float32Array(total);
    for (let i = 0; i < total; i++) { let s = 0; for (let r = 0; r < world; r++) s += shards[r][i]; expected[i] = s; }

    let worst = 0;
    for (let r = 0; r < world; r++) {
      const buf = Buffer.alloc(total * 2);
      pt.shard(r).d2h(buf);
      const a = bf16BytesToF32(buf);
      for (let i = 0; i < total; i++)
        worst = Math.max(worst, Math.abs(a[i] - expected[i]) / Math.max(Math.abs(expected[i]), 0.5));
    }
    // bf16 output precision is ~2^-8 ≈ 0.0039; allow a small margin.
    assert.ok(worst < 0.02, `world=${world} rows=${rows} cols=${cols}: worstRelErr=${worst}`);
  } finally {
    ws.free();
    po.free();
    for (const g of glms) g.free();
  }
}

describe("push-based reduce-scatter allReduce", () => {
  it("world=4, int4 path", () => runCase([0, 1, 2, 4], 4, 768));
  it("world=8, aligned int4 path", () => runCase([0, 1, 2, 3, 4, 5, 6, 7], 1, 5120));
  it("world=8, misaligned chunk (byte fallback)", () => runCase([0, 1, 2, 3, 4, 5, 6, 7], 1, 5152));
  it("world=8, tiny (all byte fallback)", () => runCase([0, 1, 2, 3, 4, 5, 6, 7], 1, 64));
  it("world=8, larger multi-row", () => runCase([0, 1, 2, 3, 4, 5, 6, 7], 2, 4096));
});
