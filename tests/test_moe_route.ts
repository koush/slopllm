import assert from "node:assert/strict";
import { it } from "node:test";
import { GlmOps, f32ToBf16Bytes } from "../src/glm_ops";
import { MetaOps } from "../src/meta_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { getNativeAddon } from "../src/native-addon";
import { Tensor, type MoeRoutingOptions } from "../src/tensor";
import { WorkspaceBase } from "../src/workspace";

function upload(tensor: Tensor, bias = false) {
  const values = Float32Array.from({ length: tensor.numElements }, (_, i) =>
    bias ? Math.sin(i * 3) * 0.3 : Math.sin(i * 0.13) * 4);
  tensor.h2d(f32ToBf16Bytes(values));
}

function read(tensor: Tensor) {
  const buffer = Buffer.alloc(tensor.bytes);
  tensor.d2h(buffer);
  tensor.workspace.glm.synchronize();
  return buffer;
}

for (const { rows, experts, topK, biased, disabled } of [
  { rows: 1, experts: 256, topK: 8, biased: true, disabled: false },
  { rows: 4, experts: 256, topK: 8, biased: true, disabled: false },
  { rows: 8, experts: 256, topK: 8, biased: true, disabled: false },
  { rows: 32, experts: 256, topK: 8, biased: true, disabled: false },
  { rows: 33, experts: 256, topK: 8, biased: true, disabled: false },
  { rows: 4, experts: 256, topK: 8, biased: false, disabled: false },
  { rows: 4, experts: 256, topK: 8, biased: true, disabled: true },
  { rows: 4, experts: 64, topK: 4, biased: true, disabled: false },
  { rows: 4, experts: 64, topK: 16, biased: true, disabled: false },
]) {
  for (const normalize of [false, true]) {
    it(`moeRoute exact default/backend match rows=${rows} experts=${experts} k=${topK} bias=${biased} disabled=${disabled} normalize=${normalize}`, () => {
      const glm = new GlmOps(Number(process.env.GLM_GPU ?? 0));
      const ws = new WorkspaceBase(glm);
      const native = getNativeAddon();
      const original = native.routeTop8;
      const previousEnv = process.env.GLM_ROUTING_FUSION;
      let fusedCalls = 0;
      native.routeTop8 = (...args) => {
        fusedCalls++;
        original(...args);
      };
      process.env.GLM_ROUTING_FUSION = disabled ? "0" : "1";
      try {
        {
          using logits = ws.alloc([rows, experts], "BF16");
          using bias = biased ? ws.alloc([experts], "BF16") : undefined;
          upload(logits);
          if (bias) {
            upload(bias, true);
          }
          const options: MoeRoutingOptions = {
            numExpertsPerToken: topK, correctionBias: bias, scalingFactor: 2.5, normalize,
          };
          const expected = Tensor.prototype.moeRoute.call(logits, options);
          using expectedIndices = expected.indices;
          using expectedStream = expected.normalizedWeightsStream;
          using expectedWeights = expectedStream.result;
          const actual = logits.moeRoute(options);
          using actualIndices = actual.indices;
          using actualStream = actual.normalizedWeightsStream;
          using actualWeights = actualStream.result;
          expected.normalizedWeightsStream.streamWaitEvent();
          actual.normalizedWeightsStream.streamWaitEvent();
          assert.deepEqual(read(actual.indices), read(expected.indices));
          assert.deepEqual(read(actual.normalizedWeightsStream.result), read(expected.normalizedWeightsStream.result));
          assert.equal(fusedCalls, rows <= 32 && experts === 256 && topK === 8 && biased && !disabled ? 1 : 0);
          assert.equal(Symbol.dispose in actual, false);
          actualStream[Symbol.dispose]();
          assert.equal(actualIndices.disposed, false);
          assert.equal(actualWeights.disposed, false);
        }
        glm.synchronize();
        assert.equal(ws.tracked.size, 0);
        assert.equal(glm.availableStreams.length, 63);
      } finally {
        native.routeTop8 = original;
        if (previousEnv === undefined) {
          delete process.env.GLM_ROUTING_FUSION;
        } else {
          process.env.GLM_ROUTING_FUSION = previousEnv;
        }
        ws.free();
        glm.free();
      }
    });
  }
}

it("MetaTensor inherits generic routing and releases all outputs", () => {
  const glm = new MetaOps();
  const ws = new WorkspaceBase(glm);
  try {
    {
      using logits = ws.alloc([7, 64], "BF16");
      const routing = logits.moeRoute({ numExpertsPerToken: 4, scalingFactor: 2.5, normalize: true });
      using indices = routing.indices;
      using stream = routing.normalizedWeightsStream;
      using weights = stream.result;
      assert.deepEqual(routing.indices.shape, [7, 4]);
      assert.equal(routing.indices.type, "I32");
      assert.deepEqual(routing.normalizedWeightsStream.result.shape, [7, 4]);
      assert.equal(routing.normalizedWeightsStream.result.type, "BF16");
    }
    assert.equal(ws.tracked.size, 0);
  } finally {
    ws.free();
    glm[Symbol.dispose]();
  }
});

for (const rows of [4, 33]) {
  it(`parallel moeRoute delegates per shard and owns independent stream results (rows=${rows})`, () => {
    const devices = [new GlmOps(0), new GlmOps(1)];
    const glm = new ParallelOps(devices);
    const ws = new WorkspaceBase(glm);
    try {
      {
        // Deliberately make per-device stream IDs differ for the fallback.
        using occupied = devices[0].withStream(() => undefined);
        using logits = ws.alloc([rows, 256], "BF16") as ParallelTensor;
        using bias = ws.alloc([256], "BF16") as ParallelTensor;
        upload(logits);
        upload(bias, true);
        const routing = logits.moeRoute({ numExpertsPerToken: 8, correctionBias: bias, scalingFactor: 2.5, normalize: true });
        using indices = routing.indices as ParallelTensor;
        using stream = routing.normalizedWeightsStream;
        using weights = stream.result as ParallelTensor;
        routing.normalizedWeightsStream.streamWaitEvent();
        assert.equal(routing.normalizedWeightsStream.shards?.length, 2);
        for (let rank = 0; rank < 2; rank++) {
          const expected = Tensor.prototype.moeRoute.call(logits.shards[rank], {
            numExpertsPerToken: 8, correctionBias: bias.shards[rank], scalingFactor: 2.5, normalize: true,
          });
          using expectedIndices = expected.indices;
          using expectedStream = expected.normalizedWeightsStream;
          using expectedWeights = expectedStream.result;
          expected.normalizedWeightsStream.streamWaitEvent();
          assert.deepEqual(read(indices.shards[rank]), read(expected.indices));
          assert.deepEqual(read(weights.shards[rank]), read(expected.normalizedWeightsStream.result));
        }
      }
      glm.synchronize();
      assert.equal(ws.tracked.size, 0);
      for (const device of devices) {
        assert.equal(device.availableStreams.length, 63);
      }
    } finally {
      ws.free();
      glm.free();
      for (const device of devices) {
        device.free();
      }
    }
  });
}
