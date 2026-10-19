import assert from "node:assert/strict";
import { it } from "node:test";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { bf16Round, dsv41Rope, dsv41Route, dsv41SwiGlu, dsv41IndexQuantize, dsv41IndexScores, writeDsv4Cache } from "./deepseek_v41_oracles";

it("V4.1 routing, clipped activation, rotary and paged cache packing match the host oracle", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  using logits = ws.alloc([1, 384], "F32"), bias = ws.alloc([384], "F32");
  const l = Float32Array.from({ length: 384 }, (_, i) => Math.sin(i * 17) * 20);
  const b = Float32Array.from({ length: 384 }, (_, i) => Math.cos(i * 7));
  for (const underflow of [false, true]) {
    const input = underflow ? new Float32Array(384).fill(-1000) : l;
    logits.h2d(Buffer.from(input.buffer)); bias.h2d(Buffer.from(b.buffer));
    const route = logits.moeRoute({ scoring: "sqrt_softplus", correctionBias: bias, numExpertsPerToken: 6, scalingFactor: 1.5, normalize: true });
    using stream = route.normalizedWeightsStream;
    stream.streamWaitEvent();
    using ids = route.indices, weights = stream.result;
    const expected = dsv41Route(input, b), bytes = Buffer.alloc(48);
    weights.d2h(bytes);
    assert.deepEqual(ids.readInt32LEArray(), expected.map(e => e.expert));
    expected.forEach((e, i) => assert.ok(Math.abs(bytes.readDoubleLE(i * 8) - e.weight) < 1e-14));
  }
  const values = bf16BytesToF32(bf16Round(Float32Array.from({ length: 1024 }, (_, i) => Math.sin(i) * 15)));
  using input = ws.alloc([2, 512], "BF16"), position = ws.alloc([1], "I32");
  input.h2d(bf16Round(values));
  for (const p of [0, 37, 128, 513, 16000]) for (const compressed of [false, true]) {
    position.h2d(Buffer.from(Int32Array.of(p).buffer));
    using rotated = input.dsv41Rope(position, compressed);
    const bytes = Buffer.alloc(rotated.bytes); rotated.d2h(bytes);
    assert.deepEqual(bytes, bf16Round(dsv41Rope(values, p, compressed)));
  }
  using activated = input.siluAndMul(input, 10);
  const bytes = Buffer.alloc(activated.bytes); activated.d2h(bytes);
  assert.deepEqual(bytes, bf16Round(dsv41SwiGlu(values, values)));
  using cache = ws.alloc([2, 64, 584], "U8"); cache.fill(0, cache.numElements);
  using pages = ws.alloc([2], "I32"); pages.h2d(Buffer.from(Int32Array.of(0, 1).buffer));
  using row = input.narrow(0, 1);
  position.h2d(Buffer.from(Int32Array.of(65).buffer));
  ops.dsv4CacheAppend(cache, row, position, pages);
  const actual = Buffer.alloc(cache.bytes), expected = Buffer.alloc(cache.bytes);
  cache.d2h(actual); writeDsv4Cache(expected, 65, values.subarray(0, 512));
  assert.deepEqual(actual, expected);
});

it("V4.1 batched primitives preserve token offsets and causal paged visibility", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const tokens = 5;
  using logits = ws.alloc([tokens, 384], "F32"), bias = ws.alloc([384], "F32");
  const logitValues = Float32Array.from({ length: tokens * 384 }, (_, i) => Math.sin(i * 0.37) * 13);
  const biasValues = Float32Array.from({ length: 384 }, (_, i) => Math.cos(i * 0.13));
  logits.h2d(Buffer.from(logitValues.buffer)); bias.h2d(Buffer.from(biasValues.buffer));
  const route = logits.moeRoute({ scoring: "sqrt_softplus", correctionBias: bias, numExpertsPerToken: 6, scalingFactor: 1.5, normalize: true });
  using stream = route.normalizedWeightsStream;
  stream.streamWaitEvent();
  using ids = route.indices, weights = stream.result;
  const selected = ids.readInt32LEArray(), weightBytes = Buffer.alloc(weights.bytes);
  weights.d2h(weightBytes);
  for (let token = 0; token < tokens; token++) {
    const expected = dsv41Route(logitValues.subarray(token * 384, (token + 1) * 384), biasValues);
    assert.deepEqual(selected.slice(token * 6, token * 6 + 6), expected.map(e => e.expert));
    expected.forEach((e, slot) => assert.ok(Math.abs(weightBytes.readDoubleLE((token * 6 + slot) * 8) - e.weight) < 1e-14));
  }
  for (const [start, length] of [[0, 5], [125, 131], [256, 3]]) {
    const row = (position: number) => bf16BytesToF32(bf16Round(Float32Array.from({ length: 512 }, (_, d) => Math.sin(position * 0.3 + d * 0.1))));
    using positions = ws.alloc([length], "I32");
    positions.h2d(Buffer.from(Int32Array.from({ length }, (_, i) => start + i).buffer));
    using input = ws.alloc([length, 512], "BF16");
    input.h2d(Buffer.concat(Array.from({ length }, (_, i) => bf16Round(row(start + i)))));
    using rotated = input.dsv41Rope(positions, true);
    const rotatedBytes = Buffer.alloc(rotated.bytes); rotated.d2h(rotatedBytes);
    assert.deepEqual(rotatedBytes, Buffer.concat(Array.from({ length }, (_, i) => bf16Round(dsv41Rope(row(start + i), start + i, true)))));
    const pageCount = Math.ceil((start + length) / 64);
    using pages = ws.alloc([pageCount], "I32");
    pages.h2d(Buffer.from(Int32Array.from({ length: pageCount }, (_, i) => i).buffer));
    using cache = ws.alloc([pageCount, 64, 584], "U8"); cache.fill(0, cache.numElements);
    ops.dsv4CacheAppend(cache, input, positions, pages);
    const expected = Buffer.alloc(cache.bytes), actual = Buffer.alloc(cache.bytes);
    for (let i = 0; i < length; i++) writeDsv4Cache(expected, start + i, row(start + i));
    cache.d2h(actual); assert.deepEqual(actual, expected);
    using indices = ops.dsv4LocalSlots(positions, pages);
    const slots = indices.readInt32LEArray();
    for (let i = 0; i < length; i++) {
      const p = start + i, first = Math.max(0, p - 127);
      assert.deepEqual(slots.slice(i * 128, (i + 1) * 128), Array.from({ length: 128 }, (_, j) => first + j > p ? -1 : first + j));
    }
  }
});

it("V4.1 batched paged indexer masks future keys independently for each query", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const positions = [0, 3, 7], count = positions.length, capacity = 8;
  using p = ws.alloc([count], "I32"); p.h2d(Buffer.from(Int32Array.from(positions).buffer));
  using q = ws.alloc([count, 4096], "BF16"), quantized = ws.alloc(q.shape, "F32");
  const qValues = bf16BytesToF32(bf16Round(Float32Array.from({ length: count * 4096 }, (_, i) => Math.sin(i * 0.29))));
  q.h2d(bf16Round(qValues)); ops.dsv41IndexQuantize(quantized, q, p);
  const expectedQ = dsv41IndexQuantize(qValues), qBytes = Buffer.alloc(quantized.bytes);
  quantized.d2h(qBytes); assert.deepEqual(qBytes, Buffer.from(expectedQ.buffer));
  using keys = ws.alloc([1, 64, 128], "F32"), weights = ws.alloc([count, 32], "BF16");
  const k = Float32Array.from({ length: 64 * 128 }, (_, i) => Math.cos(i * 0.17));
  const w = bf16BytesToF32(bf16Round(Float32Array.from({ length: count * 32 }, (_, i) => Math.sin(i * 0.43))));
  keys.h2d(Buffer.from(k.buffer)); weights.h2d(bf16Round(w));
  using pages = ws.alloc([1], "I32"); pages.fill(0, 1);
  using scores = ops.dsv4IndexScores(quantized, keys, weights, p, pages, 1, capacity);
  const expected = new Float32Array(count * capacity).fill(-Infinity);
  positions.forEach((position, row) => expected.set(dsv41IndexScores(expectedQ.subarray(row * 4096, (row + 1) * 4096),
    Array.from({ length: position + 1 }, (_, j) => k.subarray(j * 128, (j + 1) * 128)), w.subarray(row * 32, (row + 1) * 32)), row * capacity));
  const bytes = Buffer.alloc(scores.bytes); scores.d2h(bytes);
  assert.deepEqual(bytes, Buffer.from(expected.buffer));
  using selected = ops.dsv41Select(scores, p, 1);
  assert.deepEqual(selected.readInt32LEArray(), positions.flatMap(position => Array.from({ length: 512 }, (_, i) => i <= position ? i : -1)));
});
