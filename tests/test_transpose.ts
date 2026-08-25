import { after, before, describe, it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";

describe("GlmTensor.transpose4d", () => {
  let glm: GlmOps;
  let ws: WorkspaceBase;

  before(() => {
    glm = new GlmOps(parseInt(process.env.GLM_GPU ?? "0", 10));
    ws = new WorkspaceBase(glm);
  });

  after(() => {
    ws.free();
    glm.free();
  });

  it("swaps the first two dimensions of I32 tensors", () => {
    const d0 = 3;
    const d1 = 2;
    const inner = 5;
    using input = ws.alloc([d0, d1, inner, 1], "I32");
    const values = Int32Array.from({ length: d0 * d1 * inner }, (_, i) => i);
    input.h2d(Buffer.from(values.buffer));

    using output = input.transpose4d(d0, d1, inner, 1, 1, 0, 2, 3);
    const actual = output.readInt32LEArray();
    const expected: number[] = [];
    for (let i1 = 0; i1 < d1; i1++) {
      for (let i0 = 0; i0 < d0; i0++) {
        for (let k = 0; k < inner; k++) {
          expected.push((i0 * d1 + i1) * inner + k);
        }
      }
    }
    assert.deepEqual(actual, expected);
  });
});
