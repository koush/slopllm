import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { TensorParallelism } from "../src/device_ops";
import { bf16BytesToF32, f32ToBf16Bytes, GlmOps } from "../src/glm_ops";
import { CP_TOPK_SORT, ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { Tensor } from "../src/tensor";
import { WorkspaceBase } from "../src/workspace";

describe("indexer owner top-k merge", () => {
  it("routes query rows to owners and restores replicated indices", () => {
    const devices = [0, 1, 2, 3, 4, 5, 6, 7].map(id => new GlmOps(id));
    const ops = new ParallelOps(devices);
    const ws = new WorkspaceBase(ops);
    try {
      const worldSize = devices.length;
      const totalQ = 16;
      const topk = 4;
      const shardWss = ops.getShardWorkspaces(ws);
      const valueShards: Tensor[] = [];
      const indexShards: Tensor[] = [];

      for (let rank = 0; rank < worldSize; rank++) {
        const values = new Float32Array(totalQ * topk);
        const indices = new Int32Array(totalQ * topk);
        for (let query = 0; query < totalQ; query++) {
          for (let k = 0; k < topk; k++) {
            const offset = query * topk + k;
            values[offset] = query * 100 + rank * 10 + k;
            indices[offset] = rank * 1000 + query * 10 + k;
          }
        }
        const valueShard = shardWss[rank].alloc([totalQ, topk], "BF16");
        const indexShard = shardWss[rank].alloc([totalQ, topk], "I32");
        valueShard.h2d(f32ToBf16Bytes(values));
        indexShard.h2d(Buffer.from(indices.buffer));
        valueShards.push(valueShard);
        indexShards.push(indexShard);
      }

      using localValues = ops.wrapShards(ws, valueShards, [totalQ, topk * worldSize], "BF16", TensorParallelism.Row);
      using localIndices = ops.wrapShards(ws, indexShards, [totalQ, topk * worldSize], "I32", TensorParallelism.Row);
      const merge = (ops as unknown as {
        tryMergeIndexerTopkByOwner(values: ParallelTensor, indices: ParallelTensor, queries: number, k: number): { values: Tensor; indices: Tensor } | undefined;
      }).tryMergeIndexerTopkByOwner(localValues, localIndices, totalQ, topk);
      assert.ok(merge, "owner merge was not available");
      using mergedValues = merge.values as ParallelTensor;
      using mergedIndices = merge.indices as ParallelTensor;
      ops.synchronize();

      assert.equal(mergedValues.parallelism, CP_TOPK_SORT ? TensorParallelism.Replicated : TensorParallelism.Column);
      assert.equal(mergedIndices.parallelism, TensorParallelism.Replicated);

      for (let rank = 0; rank < worldSize; rank++) {
        const ownerQ = totalQ / worldSize;
        const localQ = CP_TOPK_SORT ? totalQ : ownerQ;
        const queryStart = CP_TOPK_SORT ? 0 : rank * ownerQ;
        const valueBuffer = Buffer.alloc(localQ * topk * 2);
        mergedValues.shard(rank).d2h(valueBuffer);
        const values = bf16BytesToF32(valueBuffer);
        const indices = mergedIndices.shard(rank).readInt32LEArray();
        for (let localQuery = 0; localQuery < localQ; localQuery++) {
          const query = queryStart + localQuery;
          const actual = Array.from({ length: topk }, (_, k) => ({
            index: indices[query * topk + k],
            value: values[localQuery * topk + k],
          })).sort((a, b) => a.index - b.index);
          const expected = Array.from({ length: topk }, (_, k) => ({
            index: (worldSize - 1) * 1000 + query * 10 + k,
            value: bf16BytesToF32(f32ToBf16Bytes(Float32Array.of(query * 100 + (worldSize - 1) * 10 + k)))[0],
          }));
          assert.deepEqual(actual, expected);
        }
      }
    } finally {
      ws.free();
      ops.free();
      for (const device of devices) device.free();
    }
  });
});
