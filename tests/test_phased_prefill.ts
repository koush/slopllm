import assert from "node:assert/strict";
import { it } from "node:test";
import { splitRaggedInput } from "../src/phased-prefill";

it("skips phased splitting for multi-sequence or unsplittable inputs", () => {
  for (const input of [[], [[]], [[1]], [[1, 2], [3, 4]], [[1, 2, 3, 4], [5, 6]]]) {
    assert.equal(splitRaggedInput(input), undefined);
  }
});

it("preserves balanced singleton splits and the next token", () => {
  assert.deepEqual(splitRaggedInput([[1, 2]]), { inputA: [[1]], inputB: [[2]], nextA: [2] });
  assert.deepEqual(splitRaggedInput([[1, 2, 3, 4, 5]]), {
    inputA: [[1, 2]], inputB: [[3, 4, 5]], nextA: [3],
  });
});
