import assert from "node:assert/strict";
import { it } from "node:test";
import { TensorParallelism } from "../src/device_ops";
import { ParallelOps } from "../src/parallel_ops";

// Host-only dispatch test: emulate device indexer outputs to check query/weight
// slicing, global offsets, padding, and gather order independently of CUDA.
class HostTensor {
  parallelism = TensorParallelism.Replicated;
  workspace = workspace;
  constructor(public shape: number[], public type: string, public data = new Float64Array(shape.reduce((a, b) => a * b, 1))) {}
  get numElements() { return this.data.length; }
  narrow(offset: number, rows: number): HostTensor {
    assert.ok(rows > 0 && offset >= 0 && offset + rows <= this.shape[0]);
    const width = this.numElements / this.shape[0];
    return new HostTensor([rows, ...this.shape.slice(1)], this.type, this.data.subarray(offset * width, (offset + rows) * width));
  }
  fill(value: number) { this.data.fill(value); }
  memcpy(source: HostTensor) { this.data.set(source.data); }
  [Symbol.dispose]() {}
}

const workspace = {
  alloc: (shape: number[], type: string) => new HostTensor(shape, type),
};

class HostParallelTensor {
  workspace = workspace;
  constructor(public shards: HostTensor[], public shape: number[], public type: string, public parallelism = TensorParallelism.Replicated) {}
  viewClone() { return this; }
  allGather() {
    assert.equal(this.type, "I32", "only indices should be gathered");
    assert.equal(this.parallelism, TensorParallelism.Column);
    const localRows = this.shape[0] / this.shards.length;
    const width = this.shape[1];
    const data = new Float64Array(this.shape[0] * width);
    this.shards.forEach((shard, rank) => {
      assert.deepEqual(shard.shape, [localRows, width]);
      data.set(shard.data, rank * localRows * width);
    });
    return new HostParallelTensor(this.shards.map(() => new HostTensor(this.shape, this.type, data)), this.shape, this.type);
  }
  narrow(offset: number, rows: number) {
    assert.equal(this.parallelism, TensorParallelism.Replicated);
    return new HostParallelTensor(this.shards.map(shard => shard.narrow(offset, rows)), [rows, ...this.shape.slice(1)], this.type);
  }
  [Symbol.dispose]() {}
}

for (const totalQ of [7, 8, 9, 15, 16, 7435]) {
  it(`indexer query sharding preserves ${totalQ} real rows and global offsets`, () => {
    const W = 8;
    const globalStart = 17;
    const topk = 2;
    const calls: { rank: number; rows: number; start: number }[] = [];
    const ops = Object.create(ParallelOps.prototype) as ParallelOps;
    Object.assign(ops, {
      worldSize: W,
      devices: Array.from({ length: W }, (_, rank) => ({
        indexerTopk: (...args: any[]) => {
          const q = args[1] as HostTensor;
          const weights = args[4] as HostTensor;
          const start = args[12] as number;
          const rows = q.shape[0];
          calls.push({ rank, rows, start });
          const indices = workspace.alloc([rows, topk], "I32");
          const values = workspace.alloc([rows, topk], "BF16");
          for (let row = 0; row < rows; row++) {
            const originalRow = start - globalStart + row;
            assert.equal(q.data[row], originalRow + 1);
            assert.equal(weights.data[row], originalRow + 101);
            for (let k = 0; k < topk; k++) {
              indices.data[row * topk + k] = start + row + k;
              values.data[row * topk + k] = q.data[row] + weights.data[row] + k;
            }
          }
          return { indices, values };
        },
      })),
      wrapShards: (_ws: unknown, shards: HostTensor[], shape: number[], type: string, parallelism: TensorParallelism) =>
        new HostParallelTensor(shards, shape, type, parallelism),
    });
    const replicated = (shape: number[], type: string, base = 0) => new HostParallelTensor(
      Array.from({ length: W }, () => {
        const tensor = new HostTensor(shape, type);
        tensor.data.forEach((_, i) => { tensor.data[i] = base + i; });
        return tensor;
      }), shape, type,
    );
    const metadata = replicated([2], "I32");
    const state = {
      cache: { getPagedKV: () => ({ contextParallel: false }) },
      globalLastPageLen: metadata,
    };
    const result = ops.indexerTopk(
      state as any, replicated([totalQ, 1, 1], "BF16", 1) as any,
      replicated([1, 64, 1], "U8") as any, replicated([1, 64], "F32") as any,
      replicated([totalQ, 1], "BF16", 101) as any,
      metadata as any, metadata as any, metadata as any, metadata as any,
      1, topk, false, globalStart,
    );
    const sharded = totalQ >= W;
    assert.equal(calls.reduce((sum, call) => sum + call.rows, 0), sharded ? totalQ : totalQ * W);
    assert.ok(calls.every(call => call.rows <= (sharded ? Math.ceil(totalQ / W) : totalQ)));
    for (const [name, tensor] of Object.entries(result)) {
      const output = tensor as unknown as HostParallelTensor;
      if (name === "values" && sharded) {
        const localRows = Math.ceil(totalQ / W);
        assert.equal(output.parallelism, TensorParallelism.Column);
        assert.deepEqual(output.shape, [localRows * W, topk]);
        output.shards.forEach((shard, rank) => {
          assert.deepEqual(shard.shape, [localRows, topk]);
          for (let row = 0; row < localRows; row++) {
            const originalRow = rank * localRows + row;
            for (let k = 0; k < topk; k++) {
              assert.equal(shard.data[row * topk + k], originalRow < totalQ ? 2 * originalRow + 102 + k : 0);
            }
          }
        });
        continue;
      }
      assert.deepEqual(output.shape, [totalQ, topk]);
      for (const shard of output.shards) {
        for (let row = 0; row < totalQ; row++) {
          for (let k = 0; k < topk; k++) {
            assert.equal(shard.data[row * topk + k], name === "indices" ? globalStart + row + k : 2 * row + 102 + k);
          }
        }
      }
    }
  });
}
