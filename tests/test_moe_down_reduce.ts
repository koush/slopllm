import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { WorkspaceBase } from "../src/workspace";
import { TensorParallelism } from "../src/device_ops";
import { Tensor } from "../src/tensor";
import { getNativeAddon } from "../src/native-addon";

function nvfp4Weights(ws: WorkspaceBase, n: number, k: number, prefix: string, parallelism?: TensorParallelism) {
  return Array.from({ length: 8 }, (_, e) => {
    const name = `${prefix}.expert${e}.weight`;
    const weight = ws.alloc([n, k / 2], "U8", name, parallelism);
    weight.h2d(Buffer.alloc(n * k / 2, 0x22));
    ws.alloc([n, k / 16], "F8_E4M3", `${name}_weight_scale`, parallelism).h2d(Buffer.alloc(n * k / 16, 0x38));
    const one = Buffer.alloc(4); one.writeFloatLE(1);
    ws.alloc([1], "F32", `${name}_weight_scale_2`).h2d(one);
    return weight;
  });
}

function bytes(tensor: Tensor) {
  const data = Buffer.alloc(tensor.bytes);
  tensor.d2h(data);
  return data;
}

// Independent decomposed reference using the retained native combine kernel.
function combine(down: Tensor, routing: Tensor, topK: number, rows: number): Tensor {
  if (down instanceof ParallelTensor && routing instanceof ParallelTensor) {
    const shards = down.shards.map((shard, i) => combine(shard, routing.shards[i], topK, rows));
    return (down.workspace.ops as ParallelOps).wrapShards(down.workspace, shards, [rows, down.shape[1]], down.type, down.parallelism);
  }
  const out = down.workspace.alloc([rows, down.shape[1]], down.type);
  getNativeAddon().scatterAddRows((down.workspace.ops as GlmOps).ctx, out.data, down.data,
    routing.data, topK, down.shape[1], rows, 0);
  return out;
}

for (const worldSize of [1, 2]) {
  it(`swiGluMlpMoeReduce matches decomposed NVFP4 MLP on ${worldSize} GPU(s) with lazy local waits`, () => {
    using device0 = new GlmOps(0);
    using device1 = worldSize === 2 ? new GlmOps(1) : undefined;
    using parallel = device1 ? new ParallelOps([device0, device1]) : undefined;
    const ops = parallel ?? device0;
    using ws = new WorkspaceBase(ops);
    const intermediate = 256 * worldSize;
    const weights = {
      gate: nvfp4Weights(ws, intermediate, 6144, "gate", TensorParallelism.Column),
      up: nvfp4Weights(ws, intermediate, 6144, "up", TensorParallelism.Column),
      down: nvfp4Weights(ws, 6144, intermediate, "down", TensorParallelism.Row),
    };
    using input = ws.alloc([1, 6144], "BF16"); input.fill(1 / 6144, input.numElements);
    using ids = ws.alloc([8], "I32"); ids.h2d(Buffer.from(new Int32Array([0,1,2,3,4,5,6,7]).buffer));
    using expectedWeights = ws.alloc([8], "BF16"); expectedWeights.fill(0.125, 8);
    using down = input.swiGluMlpMoe(weights, ids, 8, 8, intermediate, 6144, "reference");
    using expected = combine(down, expectedWeights, 8, 1);

    using stream = ops.withStream(() => {
      const result = ws.alloc([1, 8], "BF16"); result.fill(0.125, 8); return result;
    });
    using routing = stream.result;
    const waits: (number | undefined)[] = [];
    const wait = stream.streamWaitEvent.bind(stream);
    stream.streamWaitEvent = () => { waits.push(undefined); wait(); };
    if (parallel) {
      parallel.devices.forEach((device, i) => {
        const localWait = device.streamWaitEvent.bind(device);
        device.streamWaitEvent = (destination, source) => {
          if (source === stream.streamId) waits.push(i);
          localWait(destination, source);
        };
      });
    }
    using actual = input.swiGluMlpMoeReduce({
      ...weights,
      normalizedWeightsStream: stream,
    }, ids, 8, 8, intermediate, 6144, "actual");
    assert.deepEqual(waits, worldSize === 1 ? [undefined] : [0, 1]);
    assert.equal(routing.disposed, false, "routing result remains caller-owned");
    assert.deepEqual(actual.shape, [1, 6144]);
    if (actual instanceof ParallelTensor && expected instanceof ParallelTensor) {
      assert.equal(actual.parallelism, TensorParallelism.PartialSum);
      actual.shards.forEach((shard, i) => assert.deepEqual(bytes(shard), bytes(expected.shards[i])));
    } else {
      assert.deepEqual(bytes(actual), bytes(expected));
    }
  });
}

it("swiGluMlpMoeReduce reads routing weights after the BF16 fallback MLP", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const makeWeights = (prefix: string, n: number, k: number) => Array.from({ length: 2 }, (_, i) => {
    const weight = ws.alloc([n, k], "BF16", `${prefix}.${i}`); weight.fill(0.125, weight.numElements); return weight;
  });
  const weights = { gate: makeWeights("gate", 16, 8), up: makeWeights("up", 16, 8), down: makeWeights("down", 8, 16) };
  using input = ws.alloc([2, 8], "BF16"); input.fill(1, input.numElements);
  using ids = ws.alloc([4], "I32"); ids.h2d(Buffer.from(new Int32Array([0,1,1,0]).buffer));
  using routing = ws.alloc([4], "BF16"); routing.fill(0.5, 4);
  using down = input.swiGluMlpMoe(weights, ids, 2, 4, 16, 8, "reference");
  using expected = combine(down, routing, 2, 2);
  const mlp = input.swiGluMlpMoe.bind(input);
  let mlpQueued = false;
  input.swiGluMlpMoe = (...args) => { const out = mlp(...args); mlpQueued = true; return out; };
  using stream = ops.withStream(() => routing);
  const wait = stream.streamWaitEvent.bind(stream);
  stream.streamWaitEvent = () => { assert.ok(mlpQueued); wait(); };
  using actual = input.swiGluMlpMoeReduce({
    ...weights,
    normalizedWeightsStream: stream,
  }, ids, 2, 4, 16, 8, "actual");
  assert.deepEqual(bytes(actual), bytes(expected));
});

it("native fused down rejects unsupported row counts before touching pointers", () => {
  for (const rows of [0, 33]) {
    assert.throws(() => getNativeAddon().nvfp4MulMatIdReduce(0, 0, 0, 0, 0, 0, 0, 0, rows), /1\.\.32 rows/);
  }
});
