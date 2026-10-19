import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { TensorParallelism } from "../src/device_ops";
import { WorkspaceBase } from "../src/workspace";

it("BF16 conversion reduces F32 partial sums before rounding", () => {
  using gpu0 = new GlmOps(0);
  using gpu1 = new GlmOps(1);
  using ops = new ParallelOps([gpu0, gpu1]);
  using ws = new WorkspaceBase(ops);
  using partial = ws.alloc([1, 256], "F32", undefined, TensorParallelism.PartialSum) as ParallelTensor;
  // Individually rounding 257 and -256 to BF16 would incorrectly produce zero.
  partial.shards[0].fill(257, 256);
  partial.shards[1].fill(-256, 256);
  using result = partial.toType("BF16");
  assert.equal(result.parallelism, TensorParallelism.Replicated);
  const data = Buffer.alloc(result.bytes);
  result.d2h(data);
  assert.deepEqual(Array.from(bf16BytesToF32(data)), Array(256).fill(1));
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
