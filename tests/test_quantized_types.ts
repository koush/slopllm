import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { ParallelOps } from "../src/parallel_ops";
import { TensorParallelism } from "../src/device_ops";
import { WorkspaceBase } from "../src/workspace";

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

it("linear dispatches explicit NVFP4 and rejects opaque U8 and unsupported MXFP4", () => {
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
  assert.throws(() => input.mulMatId([mxfp4], ids, 1, 1, 128, 128, "mx"), /unsupported weight type/);
});
