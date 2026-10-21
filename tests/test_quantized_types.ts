import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { TensorParallelism } from "../src/device_ops";
import { WorkspaceBase } from "../src/workspace";
import { getNativeAddon } from "../src/native-addon";

it("BF16 conversion preserves shard layout and converts locally without collectives", () => {
  using gpu0 = new GlmOps(0);
  using gpu1 = new GlmOps(1);
  using ops = new ParallelOps([gpu0, gpu1]);
  using ws = new WorkspaceBase(ops);
  for (const layout of [TensorParallelism.PartialSum, TensorParallelism.Row,
    TensorParallelism.Column, TensorParallelism.Replicated]) {
    using input = ws.alloc([2, 256], "F32", undefined, layout) as ParallelTensor;
    // Distinguish local rounding from reduction-before-rounding (which gives 1).
    input.shards[0].fill(257, input.shards[0].numElements);
    input.shards[1].fill(-256, input.shards[1].numElements);
    using result = input.toType("BF16") as ParallelTensor;
    assert.equal(input.parallelism, layout);
    assert.equal(result.parallelism, layout);
    assert.deepEqual(result.shape, input.shape);
    for (let rank = 0; rank < 2; rank++) {
      const shard = result.shards[rank];
      assert.deepEqual(shard.shape, input.shards[rank].shape);
      const data = Buffer.alloc(shard.bytes);
      shard.d2h(data);
      assert.deepEqual(Array.from(bf16BytesToF32(data)), Array(shard.numElements).fill(rank ? -256 : 256));
      const source = new Float32Array(input.shards[rank].numElements);
      input.shards[rank].d2h(Buffer.from(source.buffer));
      assert.ok(source.every(value => value === (rank ? -256 : 257)));
    }
    using sameType = result.toType("BF16") as ParallelTensor;
    assert.equal(sameType.parallelism, layout);
  }
});

it("packed formats and E8M0 retain their bytes and dtype through TP gather", () => {
  using gpu0 = new GlmOps(0);
  using gpu1 = new GlmOps(1);
  using ops = new ParallelOps([gpu0, gpu1]);
  using ws = new WorkspaceBase(ops);
  for (const type of ["U8_NVFP4", "U8_MXFP4", "F8_E8M0"]) {
    for (const layout of [TensorParallelism.Column, TensorParallelism.Row]) {
      using tensor = ws.alloc([128, 64], type, undefined, layout);
      const bytes = Buffer.from(Array.from({ length: 128 * 64 }, (_, i) => (i * 13 + (i >> 6)) & 255));
      assert.equal(tensor.bytes, bytes.length);
      tensor.fill(0, tensor.numElements);
      const zero = Buffer.alloc(tensor.bytes);
      tensor.d2h(zero);
      assert.ok(zero.every(byte => byte === 0));
      tensor.h2d(bytes);
      using view = tensor.viewClone();
      using full = view.replicate();
      assert.equal(full.type, type);
      const actual = Buffer.alloc(full.bytes);
      full.d2h(actual);
      assert.deepEqual(actual, bytes);
    }
  }
});

it("linear dispatches NVFP4 and rejects opaque U8 and unsupported dense MXFP4", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  using input = ws.alloc([1, 128], "BF16");
  input.fill(1, 128);
  const weight = ws.alloc([128, 64], "U8_NVFP4", "weight");
  weight.h2d(Buffer.alloc(weight.bytes, 0x22)); // two E2M1 ones per byte
  ws.alloc([128, 8], "F8_E4M3", "weight_weight_scale").h2d(Buffer.alloc(128 * 8, 0x38));
  ws.alloc([1], "F32", "weight_weight_scale_2").fill(1, 1);
  using output = input.linear(weight);
  const actual = Buffer.alloc(output.bytes);
  output.d2h(actual);
  for (let i = 0; i < 128; i++) assert.equal(actual.readUInt16LE(i * 2), 0x4300); // BF16 128
  using opaque = ws.alloc([128, 128], "U8");
  assert.throws(() => input.linear(opaque), /weight type must be/);
  using mxfp4 = ws.alloc([128, 64], "U8_MXFP4");
  assert.throws(() => input.linear(mxfp4), /MXFP4 execution is not implemented/);
  using ids = ws.alloc([1], "I32");
  assert.throws(() => input.mulMatId([mxfp4], ids, 1, 1, 128, 128, "mx"), /Missing quantization scales/);
});

it("mulMatId dispatches all MXFP4 routes with associated scales", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const matrices = [0, 1].map(i => {
    const w = ws.alloc([64, 32], "U8_MXFP4", `expert${i}.weight`);
    w.h2d(Buffer.alloc(w.bytes, i ? 0x44 : 0x22));
    ws.alloc([64, 2], "F8_E8M0", `expert${i}.scale`).h2d(Buffer.alloc(128, 127));
    return w;
  });
  const experts = Array.from({ length: 384 }, (_, i) => matrices[i % 2]);
  using input = ws.alloc([2, 64], "BF16"); input.fill(1, input.numElements);
  using second = input.narrow(1, 1); second.fill(2, second.numElements);
  const routes = [0, 1, 2, 383, 250, 17, 33, 18, 27, 64, 9, 380];
  using ids = ws.alloc([12], "I32"); ids.h2d(Buffer.from(Int32Array.from(routes).buffer));
  using output = input.mulMatId(experts, ids, 6, 12, 64, 64, "experts");
  const bytes = Buffer.alloc(output.bytes); output.d2h(bytes);
  const values = bf16BytesToF32(bytes);
  using down = output.mulMatId(experts, ids, 1, 12, 64, 64, "experts");
  const downBytes = Buffer.alloc(down.bytes); down.d2h(downBytes);
  const downValues = bf16BytesToF32(downBytes);
  for (let row = 0; row < 12; row++) for (let n = 0; n < 64; n++) {
    const factor = routes[row] % 2 ? 2 : 1;
    assert.equal(values[row * 64 + n], 64 * (Math.floor(row / 6) + 1) * factor);
    assert.equal(downValues[row * 64 + n], values[row * 64 + n] * 64 * factor);
  }
});

it("grouped NVFP4 retains E4M3 and per-expert global scales", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const count = 513, N = 136, K = 512, experts = 3;
  const weights = [], scales = [], globalScales = [];
  for (let e = 0; e < experts; e++) {
    const w = ws.alloc([N, K / 2], "U8_NVFP4");
    w.h2d(Buffer.from(Array.from({ length: w.bytes }, (_, i) => (i * 17 + e * 37) & 255)));
    const s = ws.alloc([N, K / 16], "F8_E4M3");
    s.h2d(Buffer.from(Array.from({ length: s.bytes }, (_, i) => [0x30, 0x38, 0x40][(i + e) % 3])));
    const g = ws.alloc([1], "F32"); g.fill([0.5, 1, 2][e], 1);
    weights.push(w); scales.push(s); globalScales.push(g);
  }
  using wp = ws.alloc([experts], "I64"); wp.writePointers(weights);
  using sp = ws.alloc([experts], "I64"); sp.writePointers(scales);
  using gp = ws.alloc([experts], "I64"); gp.writePointers(globalScales);
  using input = ws.alloc([count, K], "BF16"); input.fill(1, input.numElements);
  using ids = ws.alloc([count], "I32");
  ids.h2d(Buffer.from(Int32Array.from({ length: count }, (_, i) => i % experts).buffer));
  using out = ws.alloc([count, N], "BF16");
  using reference = ws.alloc([count, N], "BF16");
  const addon = getNativeAddon();
  using scratch = ws.allocRaw(addon.mmaMoeCoopWorkspaceSize(count, N, K, experts));
  addon.nvfp4MulMatIdGroupedMmaCoop(ops.ctx, out.data, input.data, wp.data, sp.data, gp.data,
    ids.data, 1, count, N, K, experts, scratch.data);
  addon.nvfp4MulMatId(ops.ctx, reference.data, input.data, wp.data, sp.data, gp.data, ids.data, 1, count, N, K);
  const actual = Buffer.alloc(out.bytes), expected = Buffer.alloc(reference.bytes);
  out.d2h(actual); reference.d2h(expected);
  assert.deepEqual(actual, expected);
});

for (const K of [32, 64, 544, 5120]) {
  it(`grouped MXFP4 matches independent dequantized dot products, K=${K}`, () => {
    using ops = new GlmOps(0);
    using ws = new WorkspaceBase(ops);
    const N = 136, tokens = 131, topK = 6, count = tokens * topK;
    const levels = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
    const hostWeights: number[][] = [];
    const matrices = Array.from({ length: 3 }, (_, e) => {
      const packed = Buffer.alloc(N * K / 2), scales = Buffer.alloc(N * K / 32);
      const values = new Array<number>(N * K);
      for (let n = 0; n < N; n++) {
        for (let k = 0; k < K; k++) {
          const code = (k * 7 + n * 3 + e * 5 + (k >> 5)) & 15;
          const exponent = ((k >> 5) + n + e) % 7 - 4;
          packed[(n * K + k) >> 1] |= code << ((k & 1) * 4);
          scales[n * K / 32 + (k >> 5)] = exponent + 127;
          values[n * K + k] = levels[code & 7] * (code & 8 ? -1 : 1) * 2 ** exponent;
        }
      }
      const w = ws.alloc([N, K / 2], "U8_MXFP4", `grouped.${e}.weight`);
      w.h2d(packed);
      ws.alloc([N, K / 32], "F8_E8M0", `grouped.${e}.scale`).h2d(scales);
      hostWeights.push(values);
      return w;
    });
    // Sparse, imbalanced routing, including expert 383 and partial M/N tiles.
    const experts = Array.from({ length: 384 }, (_, e) => matrices[e % 3]);
    const routes = Array.from({ length: count }, (_, i) => [383, 0, 17, 1, 383, 2][i % 6]);
    using ids = ws.alloc([count], "I32");
    ids.h2d(Buffer.from(Int32Array.from(routes).buffer));
    const x = Buffer.alloc(tokens * K * 2);
    const inputValue = (row: number, k: number) => ((row + k * 3) % 5) - 2;
    const bf16 = [0xc000, 0xbf80, 0, 0x3f80, 0x4000];
    for (let row = 0; row < tokens; row++) for (let k = 0; k < K; k++) {
      x.writeUInt16LE(bf16[inputValue(row, k) + 2], (row * K + k) * 2);
    }
    using input = ws.alloc([tokens, K], "BF16"); input.h2d(x);
    using output = input.mulMatId(experts, ids, topK, count, N, K, "grouped");
    // Exercise the specialization explicitly as well, independent of dispatch thresholds.
    using pointers = ws.alloc([experts.length * 2], "I64");
    pointers.writePointers(experts.flatMap(w => [w, w.quantizationScales()]));
    const addon = getNativeAddon();
    using scratch = ws.allocRaw(addon.mmaMoeCoopWorkspaceSize(count, N, K, experts.length));
    using direct = ws.alloc([count, N], "BF16");
    const run = () => addon.nvfp4MulMatIdGroupedMmaCoop(ops.ctx, direct.data, input.data, pointers.data, pointers.data + 8,
      0, ids.data, topK, count, N, K, experts.length, scratch.data, true);
    run();
    const bytes = Buffer.alloc(output.bytes), directBytes = Buffer.alloc(direct.bytes);
    output.d2h(bytes); direct.d2h(directBytes);
    assert.deepEqual(directBytes, bytes);
    const actual = bf16BytesToF32(bytes);
    const rounding = new DataView(new ArrayBuffer(4));
    for (let row = 0; row < count; row++) for (let n = 0; n < N; n++) {
      let expected = 0;
      const w = hostWeights[routes[row] % 3];
      for (let k = 0; k < K; k++) expected += inputValue(Math.floor(row / topK), k) * w[n * K + k];
      rounding.setFloat32(0, expected, true);
      let bits = rounding.getUint32(0, true);
      bits = (bits + 0x7fff + ((bits >>> 16) & 1)) & 0xffff0000;
      rounding.setUint32(0, bits, true);
      assert.equal(actual[row * N + n], rounding.getFloat32(0, true), `route=${row} col=${n}`);
    }
    ops.graphBeginCapture();
    run();
    const graph = ops.graphEndCapture();
    const exec = ops.graphInstantiate(graph);
    try {
      for (const changed of [routes.map((_, i) => i % 384), routes.map(() => 383), routes]) {
        ids.h2d(Buffer.from(Int32Array.from(changed).buffer));
        direct.h2d(Buffer.alloc(direct.bytes, 0xff));
        scratch.h2d(Buffer.alloc(scratch.bytes, 0xff));
        addon.mxfp4MulMatId(ops.ctx, output.data, input.data, pointers.data, ids.data, topK, count, N, K);
        ops.graphLaunch(exec);
        output.d2h(bytes); direct.d2h(directBytes);
        assert.deepEqual(directBytes, bytes, "captured grouped output matches GEMV after route changes");
      }
    } finally {
      ops.graphExecDestroy(exec);
      ops.graphDestroy(graph);
    }
  });
}
