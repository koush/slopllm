import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";
import { GlmOps } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

describe("dtype-aware tensor fill", () => {
  let ops: GlmOps;
  let ws: WorkspaceBase;
  before(() => {
    ops = new GlmOps(Number(process.env.GLM_GPU ?? 0));
    ws = new WorkspaceBase(ops);
  });
  after(() => {
    ws.free();
    ops.free();
  });

  const cases: { type: string; value: number; encode: (b: Buffer) => void }[] = [
    { type: "BF16", value: 1.5, encode: b => b.writeUInt16LE(0x3fc0) },
    { type: "BF16", value: -Infinity, encode: b => b.writeUInt16LE(0xff80) },
    { type: "F16", value: -2.25, encode: b => b.writeUInt16LE(0xc080) },
    { type: "F32", value: 1.5, encode: b => b.writeFloatLE(1.5) },
    { type: "F64", value: 1 + 2 ** -40, encode: b => b.writeDoubleLE(1 + 2 ** -40) },
    { type: "I8", value: -73, encode: b => b.writeInt8(-73) },
    { type: "U8", value: 231, encode: b => b.writeUInt8(231) },
    { type: "I16", value: -12345, encode: b => b.writeInt16LE(-12345) },
    { type: "U16", value: 54321, encode: b => b.writeUInt16LE(54321) },
    { type: "I32", value: -0x1234567, encode: b => b.writeInt32LE(-0x1234567) },
    { type: "U32", value: 0x89abcdef, encode: b => b.writeUInt32LE(0x89abcdef) },
    { type: "I64", value: -(2 ** 40 + 123), encode: b => b.writeBigInt64LE(-(2n ** 40n + 123n)) },
    { type: "U64", value: 2 ** 40 + 123, encode: b => b.writeBigUInt64LE(2n ** 40n + 123n) },
    { type: "BOOL", value: 2, encode: b => b.writeUInt8(1) },
    { type: "F8_E4M3", value: 1.5, encode: b => b.writeUInt8(0x3c) },
    { type: "F8_E5M2", value: 1.5, encode: b => b.writeUInt8(0x3e) },
    { type: "C64", value: 1.5, encode: b => { b.writeFloatLE(1.5); b.writeFloatLE(0, 4); } },
  ];
  for (const { type, value, encode } of cases) {
    it(`fills ${type} with ${value} without touching adjacent elements`, () => {
      using tensor = ws.alloc([7], type);
      const poison = Buffer.alloc(tensor.bytes, 0xa5);
      tensor.h2d(poison);
      using view = tensor.narrow(1, 5);
      view.fill(value, 3);
      view.fill(0, 0);
      const actual = Buffer.alloc(tensor.bytes);
      tensor.d2h(actual);
      ops.synchronize();
      const expected = Buffer.from(poison);
      const width = tensor.bytes / tensor.numElements;
      for (let i = 1; i < 4; i++) encode(expected.subarray(i * width, (i + 1) * width));
      assert.deepEqual(actual, expected);

      tensor.fill(0, tensor.numElements);
      tensor.d2h(actual);
      ops.synchronize();
      assert.deepEqual(actual, Buffer.alloc(tensor.bytes));
    });
  }

  it("rejects invalid element counts before writing", () => {
    using tensor = ws.alloc([4], "I32");
    for (const n of [-1, 0.5, 5, NaN]) {
      assert.throws(() => tensor.fill(0, n), /invalid element count/);
    }
  });

  it("captures and replays I32 fill", () => {
    using tensor = ws.alloc([5], "I32");
    tensor.fill(0, 5);
    ops.synchronize();
    ops.graphBeginCapture();
    tensor.fill(-123456789, 5);
    const graph = ops.graphEndCapture();
    const exec = ops.graphInstantiate(graph);
    try {
      for (let i = 0; i < 2; i++) {
        tensor.fill(0, 5);
        ops.graphLaunch(exec);
        assert.deepEqual(tensor.readInt32LEArray(), Array(5).fill(-123456789));
      }
    } finally {
      ops.synchronize();
      ops.graphExecDestroy(exec);
      ops.graphDestroy(graph);
    }
  });
});
