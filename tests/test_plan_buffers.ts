import assert from "node:assert/strict";
import { it } from "node:test";
import { GlmOps, GlmTensor } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { PagedKVCache } from "../src/paged_kv";
import type { ChatModel, ChatCache } from "../src/chat_model";
import type { Tensor } from "../src/tensor";

const shards = (tensor: Tensor): readonly Tensor[] => tensor instanceof ParallelTensor ? tensor.shards : [tensor];

for (const { parallel, mode } of [
  { parallel: false, mode: "standard" },
  { parallel: false, mode: "dense MLA" },
  { parallel: false, mode: "sparse MLA" },
  { parallel: true, mode: "sparse MLA" },
]) {
  it(`packed planning: ${mode}, ${parallel ? "TP2+CP" : "single GPU"}, slot isolation and replay across page growth`, () => {
    const devices = (parallel ? [0, 1] : [0]).map(id => new GlmOps(id));
    const ops = parallel ? new ParallelOps(devices) : devices[0];
    const ws = new ExecutionWorkspace(ops, 8, 256);
    const kvRank = mode === "standard" ? 0 : 128;
    const kv = new PagedKVCache(ops, 4, 128, 1, 32, 8, 64 / devices.length, kvRank, 64, parallel, mode === "sparse MLA" ? 128 : 0);
    const model = { cfg: { numAttentionHeads: 4, numKeyValueHeads: 4, headDim: 128, kvLoraRank: kvRank, qkRopeHeadDim: 64 } } as ChatModel;
    const cache = { getPagedKV: () => kv } as ChatCache;
    const originalMemcpy = GlmTensor.prototype.memcpy;
    let copies = 0;
    let bytes = 0;
    GlmTensor.prototype.memcpy = function (source, size, kind) {
      if (source.pinned && !this.pinned) {
        copies++;
        bytes += size ?? source.bytes;
      }
      return originalMemcpy.call(this, source, size, kind);
    };
    let graph: number | undefined;
    let exec: number | undefined;
    try {
      kv.reset(2);
      const prefill = ws.planPrefill(model, 2, [63, 63], cache);
      assert.equal(copies, devices.length);
      assert.equal(bytes, (3 * 126 + 6 * 2 + 3 + 2) * 4 * devices.length);
      assert.equal(prefill.positionIds.numElements, 126);
      assert.equal(prefill.indptrD.numElements, 3);
      assert.equal(prefill.indices.numElements, 2);
      for (const tensor of shards(prefill.indptrH)) {
        assert.equal(tensor.readPinnedBuffer().length, 3 * 4);
        tensor.withPinnedBuffer(buffer => assert.throws(() => buffer.writeInt32LE(0, 3 * 4), RangeError));
      }
      prefill.setInput([Array(63).fill(7), Array(63).fill(8)]);
      assert.throws(() => prefill.setInput([Array(127).fill(9)]), /planned token/);
      ops.synchronize();
      assert.deepEqual(shards(prefill.qoIndptrD)[0].readInt32LEArray(), [0, 63, 126]);
      assert.deepEqual(shards(prefill.kvLenD)[0].readInt32LEArray(), [63, 63]);
      ws.resetPlanSlots();

      const outputs = [0, 1].map(slot => ws.alloc([2], "I32", `positions${slot}`));
      const indptr = ws.alloc([3], "I32", "indptrOutput");
      let capturedPointers: number[][] | undefined;
      for (let iteration = 0; iteration < 4; iteration++) {
        copies = 0;
        const first = ws.planDecode(model, 2, cache, true);
        const second = ws.planDecode(model, 2, cache, true);
        assert.equal(copies, 2 * devices.length);
        assert.notEqual(shards(first.positionIds)[0].data, shards(second.positionIds)[0].data);
        assert.equal(first.indices.numElements, iteration === 0 ? 2 : 4);
        assert.equal(second.indices.numElements, 4);
        const pointers = [first, second].map(state => [state.positionIds, state.indptrD, state.indices, state.inputIdsBuf]
          .flatMap(tensor => shards(tensor).map(shard => shard.data)));
        if (capturedPointers) assert.deepEqual(pointers, capturedPointers);
        else capturedPointers = pointers;
        ops.synchronize();
        if (iteration === 0) {
          ops.graphBeginCapture();
          outputs[0].memcpy(first.positionIds);
          outputs[1].memcpy(second.positionIds);
          indptr.memcpy(first.indptrD);
          graph = ops.graphEndCapture();
          exec = ops.graphInstantiate(graph);
        }
        ops.graphLaunch(exec!);
        ops.synchronize();
        for (const output of shards(outputs[0])) assert.deepEqual(output.readInt32LEArray(), [63 + 2 * iteration, 63 + 2 * iteration]);
        for (const output of shards(outputs[1])) assert.deepEqual(output.readInt32LEArray(), [64 + 2 * iteration, 64 + 2 * iteration]);
        for (const output of shards(indptr)) assert.deepEqual(output.readInt32LEArray(), iteration === 0 ? [0, 1, 2] : [0, 2, 4]);
        for (const [rank, tensor] of shards(second.lastPageLen).entries()) {
          const length = 65 + 2 * iteration;
          const local = Math.ceil((length - rank) / devices.length);
          const expected = local % (64 / devices.length) || 64 / devices.length;
          assert.deepEqual(tensor.readInt32LEArray(), [expected, expected]);
        }
        for (const tensor of shards(second.globalLastPageLen)) assert.deepEqual(tensor.readInt32LEArray(), [1 + 2 * iteration, 1 + 2 * iteration]);
        ws.resetPlanSlots();
      }
      // Switch batch/token layouts in the same slot, then return to the
      // captured layout. Other graphs must not disturb its pointer bindings.
      kv.reset(8);
      const larger = ws.planPrefill(model, 8, Array(8).fill(4), cache);
      ops.synchronize();
      assert.equal(larger.inputIdsBuf.numElements, 32);
      assert.equal(larger.indices.numElements, 8);
      {
        using lastIdx = larger.lastIdx;
        for (const tensor of shards(lastIdx)) assert.deepEqual(tensor.readInt32LEArray(), [4, 8, 12, 16, 20, 24, 28, 32]);
      }
      ws.resetPlanSlots();
      kv.reset(2);
      const restored = ws.planDecode(model, 2, cache, true);
      ops.synchronize();
      assert.deepEqual([restored.positionIds, restored.indptrD, restored.indices, restored.inputIdsBuf]
        .flatMap(tensor => shards(tensor).map(shard => shard.data)), capturedPointers![0]);
    } finally {
      GlmTensor.prototype.memcpy = originalMemcpy;
      if (exec !== undefined) ops.graphExecDestroy(exec);
      if (graph !== undefined) ops.graphDestroy(graph);
      ws.free();
      kv.free();
      ops.free();
    }
  });
}
