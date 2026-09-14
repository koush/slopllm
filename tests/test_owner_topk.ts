import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { CaptureManager } from "../src/capture-manager";
import { TensorParallelism } from "../src/device_ops";
import { bf16BytesToF32, f32ToBf16Bytes, GlmOps } from "../src/glm_ops";
import { CP_TOPK_SORT, ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { Tensor } from "../src/tensor";
import { WorkspaceBase } from "../src/workspace";

describe("indexer owner top-k merge", () => {
  const cases = [[16, 4], [1, 4], [4, 2048], [9, 2048]];
  cases.forEach(([totalQ, topk]) => it(`routes ${totalQ} query rows through eager and captured top-${topk}`, () => checkOwnerMerge(totalQ, topk)));
});

function checkOwnerMerge(totalQ: number, topk: number): void {
  const devices = [0, 1, 2, 3, 4, 5, 6, 7].map(id => new GlmOps(id));
  const ops = new ParallelOps(devices);
  const ws = new WorkspaceBase(ops);
  try {
    const worldSize = devices.length;
    const shardWss = ops.getShardWorkspaces(ws);
    const valueShards: Tensor[] = [];
    const indexShards: Tensor[] = [];

    for (let rank = 0; rank < worldSize; rank++) {
      const values = new Float32Array(totalQ * topk);
      const indices = new Int32Array(totalQ * topk);
      for (let query = 0; query < totalQ; query++) {
        for (let k = 0; k < topk; k++) {
          const offset = query * topk + k;
          values[offset] = rank * 16 + query / 32 + k / topk;
          indices[offset] = (rank * totalQ + query) * topk + k;
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
    using captureManager = new CaptureManager(ops);
    const inputs = { localValues, localIndices };
    for (let step = 0; step < 6; step++) {
      const result = captureManager.run(inputs, (_capturing, inputs) => {
        const merge = (ops as unknown as {
          tryMergeIndexerTopkByOwner(values: ParallelTensor, indices: ParallelTensor, queries: number, k: number): { values: Tensor; indices: Tensor } | undefined;
        }).tryMergeIndexerTopkByOwner(inputs.localValues, inputs.localIndices, totalQ, topk);
        assert.ok(merge, "owner merge was not available");
        return merge;
      }, ["owner-topk", totalQ, topk]) as { values: ParallelTensor; indices: ParallelTensor };
      using mergedValues = result.values;
      using mergedIndices = result.indices;
      ops.synchronize();

      assert.equal(mergedValues.parallelism, CP_TOPK_SORT ? TensorParallelism.Replicated : TensorParallelism.Column);
      assert.equal(mergedIndices.parallelism, TensorParallelism.Replicated);
      const ownerQ = Math.ceil(totalQ / worldSize);
      assert.deepEqual(mergedValues.shape, [CP_TOPK_SORT ? totalQ : ownerQ * worldSize, topk]);
      assert.deepEqual(mergedIndices.shape, [totalQ, topk]);

      for (let rank = 0; rank < worldSize; rank++) {
        assert.deepEqual(mergedIndices.shard(rank).shape, [totalQ, topk]);
        const localQ = CP_TOPK_SORT ? totalQ : Math.max(0, Math.min(ownerQ, totalQ - rank * ownerQ));
        const queryStart = CP_TOPK_SORT ? 0 : rank * ownerQ;
        const valueBuffer = Buffer.alloc(mergedValues.shard(rank).bytes);
        mergedValues.shard(rank).d2h(valueBuffer);
        const values = bf16BytesToF32(valueBuffer);
        const indices = mergedIndices.shard(rank).readInt32LEArray();
        // Even dummy owners must receive every real query's selected indices.
        for (let query = 0; query < totalQ; query++) {
          const actual = indices.slice(query * topk, (query + 1) * topk).sort((a, b) => a - b);
          const expected = Array.from({ length: topk }, (_, k) => ((worldSize - 1) * totalQ + query) * topk + k);
          assert.deepEqual(actual, expected);
        }
        for (let localQuery = 0; localQuery < localQ; localQuery++) {
          const query = queryStart + localQuery;
          const actual = Array.from({ length: topk }, (_, k) => ({
            index: indices[query * topk + k],
            value: values[localQuery * topk + k],
          })).sort((a, b) => a.index - b.index);
          const expected = Array.from({ length: topk }, (_, k) => ({
            index: ((worldSize - 1) * totalQ + query) * topk + k,
            value: bf16BytesToF32(f32ToBf16Bytes(Float32Array.of((worldSize - 1) * 16 + query / 32 + k / topk)))[0],
          }));
          assert.deepEqual(actual, expected);
        }
      }
    }
    assert.ok(captureManager.isCaptured(["owner-topk", totalQ, topk], inputs));
  } finally {
    ws.free();
    ops.free();
    for (const device of devices) device.free();
  }
}
