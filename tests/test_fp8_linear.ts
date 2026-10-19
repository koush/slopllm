import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { ParallelOps } from "../src/parallel_ops";
import { TensorParallelism as TP } from "../src/device_ops";

for (const world of [1, 2]) it(`BF16 linear preserves unrounded F32 output on TP${world}`, () => {
  using gpu0 = new GlmOps(0);
  using gpu1 = world === 2 ? new GlmOps(1) : undefined;
  using parallel = gpu1 ? new ParallelOps([gpu0, gpu1]) : undefined;
  using ws = new WorkspaceBase(parallel ?? gpu0);
  for (const layout of [TP.Replicated, TP.Column, TP.Row]) for (const rows of [1, 3]) {
    using weight = ws.alloc([32, 32], "BF16", undefined, layout);
    const data = Buffer.alloc(weight.bytes);
    for (let n = 0; n < 32; n++) {
      data.writeUInt16LE(0x3f80, n * 64); // 1
      data.writeUInt16LE(0x3b80, n * 64 + 32); // 1/256, in the other K shard
    }
    weight.h2d(data);
    using input = ws.alloc([rows, 32], "BF16", undefined, layout === TP.Row ? TP.Row : TP.Replicated);
    input.fill(1, input.numElements);
    using projected = input.linear(weight, { outputType: "F32" });
    using result = projected.replicate();
    assert.equal(result.type, "F32");
    const bytes = Buffer.alloc(result.bytes); result.d2h(bytes);
    for (let i = 0; i < rows * 32; i++) assert.equal(bytes.readFloatLE(i * 4), 1 + 1 / 256);
    using rounded = result.toType("BF16");
    const bf16 = Buffer.alloc(rounded.bytes); rounded.d2h(bf16);
    assert.ok(bf16BytesToF32(bf16).every(x => x === 1));
  }
});

it("legacy FP8 linear dispatch uses BF16 128x128 scales across block boundaries", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const weight = ws.alloc([129, 256], "F8_E4M3", "weight");
  weight.h2d(Buffer.alloc(weight.bytes, 0x38));
  const scales = ws.alloc([2, 2], "BF16", "weight_scale_inv");
  scales.h2d(Buffer.from(Uint16Array.from([0x3f00, 0x3f80, 0x4000, 0x4080]).buffer));
  for (const rows of [1, 3, 17]) {
    using input = ws.alloc([rows, 256], "BF16"); input.fill(1, input.numElements);
    using output = input.linear(weight);
    const bytes = Buffer.alloc(output.bytes); output.d2h(bytes);
    const values = bf16BytesToF32(bytes);
    for (let i = 0; i < values.length; i++) assert.equal(values[i], i % 129 < 128 ? 192 : 768);
  }
});

for (const world of [1, 2]) it(`grouped FP8 projections preserve token/group placement on TP${world}`, () => {
  using gpu0 = new GlmOps(0);
  using gpu1 = world === 2 ? new GlmOps(1) : undefined;
  using parallel = gpu1 ? new ParallelOps([gpu0, gpu1]) : undefined;
  const ops = parallel ?? gpu0;
  using ws = new WorkspaceBase(ops);
  const weight = ws.alloc([8192, 4096], "F8_E4M3", "projection.weight", TP.Column);
  weight.h2d(Buffer.alloc(weight.bytes, 0x38));
  const scales = ws.alloc([256, 128], "F8_E8M0", "projection.scale", TP.Column);
  scales.fill(127, scales.numElements);
  for (const rows of [3, 17]) {
    using input = ws.alloc([rows, 32768], "BF16", undefined, TP.Row);
    const data = Buffer.alloc(input.bytes), word = Buffer.alloc(4);
    for (let token = 0; token < rows; token++) for (let group = 0; group < 8; group++) {
      // Integers through 11 are exactly representable in E4M3 too; isolate
      // grouped addressing from the separately tested activation quantization.
      word.writeFloatLE(group + 1 + token % 4);
      const value = word.readUInt32LE() >>> 16;
      for (let k = 0; k < 4096; k++) data.writeUInt16LE(value, ((token * 8 + group) * 4096 + k) * 2);
    }
    input.h2d(data);
    using heads = input.reshape([rows, 64, 512]);
    using output = heads.linear(weight, { groups: 8 });
    using full = output.replicate();
    const raw = Buffer.alloc(full.bytes); full.d2h(raw);
    const values = bf16BytesToF32(raw);
    for (let i = 0; i < values.length; i++) {
      const token = Math.floor(i / 8192), group = Math.floor(i % 8192 / 1024);
      assert.equal(values[i], (group + 1 + token % 4) * 4096);
    }
  }
});
