import assert from "node:assert/strict";
import { it } from "node:test";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { CaptureManager } from "../src/capture-manager";
import type { Tensor } from "../src/tensor";
import { bf16Round, dsv41Rope, dsv41Route, dsv41SwiGlu, dsv41IndexQuantize, dsv41IndexScores, writeDsv4Cache, packDsv41Index } from "./deepseek_v41_oracles";

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
    const expected = dsv41Route(input, b), bytes = Buffer.alloc(weights.bytes);
    weights.d2h(bytes);
    assert.deepEqual(ids.readInt32LEArray(), expected.map(e => e.expert));
    assert.equal(weights.type, "F32");
    expected.forEach((e, i) => assert.ok(Math.abs(bytes.readFloatLE(i * 4) - e.weight) < 2e-7));
  }
  const values = bf16BytesToF32(bf16Round(Float32Array.from({ length: 1024 }, (_, i) => Math.sin(i) * 15)));
  using input = ws.alloc([2, 512], "BF16"), position = ws.alloc([1], "I32");
  input.h2d(bf16Round(values));
  for (const p of [0, 37, 128, 513, 16000, 1048575]) {
    position.h2d(Buffer.from(Int32Array.of(p).buffer));
    using rotary = position.dsv41RotaryEmbedding();
    for (const compressed of [false, true]) for (const ratio of compressed ? [0, 1, 2] : [0]) for (const inverse of [false, true]) {
      using rotated = input.dsv41Rope(rotary, compressed, inverse, 512, ratio);
      const bytes = Buffer.alloc(rotated.bytes); rotated.d2h(bytes);
      const effectivePosition = ratio === 2 ? (Math.floor((p + 1) / 2) - 1) * 2 : p;
      assert.deepEqual(bytes, bf16Round(dsv41Rope(values, effectivePosition, compressed, inverse)));
    }
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

it("V4.1 FP32 routing preserves ties, unbiased weights and changing captured inputs", () => {
  using ops = new GlmOps(0);
  for (const experts of [128, 384]) for (const rows of [1, 5, 513, 1025, 2049, 8192]) {
    using ws = new WorkspaceBase(ops);
    using manager = new CaptureManager(ops);
    using logits = ws.alloc([rows, experts], "F32"), bias = ws.alloc([experts], "F32");
    const topK = experts === 384 ? 6 : 3;
    for (let pass = 0; pass < 6; pass++) {
      // The first bias vector distinguishes FP32 ties from FP64 ranking:
      // all additions to sqrt(softplus(0)) round to the same FP32 score.
      const correction = Float32Array.from({ length: experts }, (_, i) => pass === 0 ? i * 1e-12 : (i % 7 - 3) * .125);
      bias.h2d(Buffer.from(correction.buffer));
      const values = Float32Array.from({ length: rows * experts }, (_, i) => {
        // Equal scores, complete underflow, extreme positive inputs, and
        // ordinary logits, with different rows winning on graph replay.
        const row = Math.floor(i / experts), mode = (row + pass) % 4;
        return mode === 0 ? 0 : mode === 1 ? -1000 : mode === 2 ? 1000 + (i % experts) : Math.sin(i * .37 + pass) * 13;
      });
      logits.h2d(Buffer.from(values.buffer));
      const output = manager.run({ logits, bias }, (_capture, input) => {
        const route = input.logits.moeRoute({ scoring: "sqrt_softplus", correctionBias: input.bias,
          numExpertsPerToken: topK, scalingFactor: 1.5, normalize: true });
        using stream = route.normalizedWeightsStream; stream.streamWaitEvent();
        return { ids: route.indices, weights: stream.result };
      }, ["route"]) as { ids: Tensor; weights: Tensor };
      using ids = output.ids; using weights = output.weights;
      assert.equal(weights.type, "F32");
      const selected = ids.readInt32LEArray(), bytes = Buffer.alloc(weights.bytes); weights.d2h(bytes);
      for (let row = 0; row < rows; row++) {
        const expected = dsv41Route(values.subarray(row * experts, (row + 1) * experts), correction);
        assert.deepEqual(selected.slice(row * topK, (row + 1) * topK), expected.map(e => e.expert));
        expected.forEach((e, k) => assert(Math.abs(bytes.readFloatLE((row * topK + k) * 4) - e.weight) < 2e-7));
      }
    }
    assert([...manager.captured.values()].some(graph => graph.graphExec));
  }
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
    expected.forEach((e, slot) => assert.ok(Math.abs(weightBytes.readFloatLE((token * 6 + slot) * 4) - e.weight) < 2e-7));
  }
  for (const [start, length] of [[0, 5], [125, 131], [256, 3]]) {
    const row = (position: number) => bf16BytesToF32(bf16Round(Float32Array.from({ length: 512 }, (_, d) => Math.sin(position * 0.3 + d * 0.1))));
    using positions = ws.alloc([length], "I32");
    positions.h2d(Buffer.from(Int32Array.from({ length }, (_, i) => start + i).buffer));
    using input = ws.alloc([length, 512], "BF16");
    input.h2d(Buffer.concat(Array.from({ length }, (_, i) => bf16Round(row(start + i)))));
    using rotary = positions.dsv41RotaryEmbedding();
    using rotated = input.dsv41Rope(rotary, true);
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

it("V4.1 captured rotary tables refresh positions and preserve inverse ratio-2 rotation", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  using manager = new CaptureManager(ops);
  using input = ws.alloc([2, 128], "BF16"), positions = ws.alloc([2], "I32");
  const values = bf16BytesToF32(bf16Round(Float32Array.from({ length: 256 }, (_, i) => Math.sin(i * 0.17))));
  input.h2d(bf16Round(values));
  for (const start of [0, 63, 128, 15999, 65535, 1048574]) {
    positions.h2d(Buffer.from(Int32Array.of(start, start + 1).buffer));
    using output = manager.run({ input, positions }, (_capturing, retained) => {
      using stream = ops.withStream(() => retained.positions.dsv41RotaryEmbedding());
      using rotary = stream.result;
      stream.streamWaitEvent();
      return retained.input.dsv41Rope(rotary, true, true, 128, 2);
    }, ["dsv41.rotary"]) as Tensor;
    const bytes = Buffer.alloc(output.bytes); output.d2h(bytes);
    assert.deepEqual(bytes, Buffer.concat([0, 1].map(i => bf16Round(dsv41Rope(
      values.slice(i * 128, (i + 1) * 128), (Math.floor((start + i + 1) / 2) - 1) * 2, true, true, 128)))));
  }
  assert([...manager.captured.values()].some(graph => graph.graphExec !== null));
});

it("V4.1 batched paged indexer masks future keys independently for each query", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const positions = [0, 3, 7], count = positions.length, capacity = 8;
  using p = ws.alloc([count], "I32"); p.h2d(Buffer.from(Int32Array.from(positions).buffer));
  using q = ws.alloc([count, 4096], "BF16");
  const qValues = bf16BytesToF32(bf16Round(Float32Array.from({ length: count * 4096 }, (_, i) => Math.sin(i * 0.29))));
  q.h2d(bf16Round(qValues));
  using quantized = ops.dsv41IndexQuantize(q, p);
  assert.deepEqual(quantized.shape, [count, 32 * 132]);
  assert.equal(quantized.workspace, q.workspace);
  const expectedQ = dsv41IndexQuantize(qValues), qBytes = Buffer.alloc(quantized.bytes);
  quantized.d2h(qBytes); assert.deepEqual(qBytes, packDsv41Index(qValues));
  using keys = ws.alloc([1, 64, 132], "U8"), weights = ws.alloc([count, 32], "BF16");
  const k = dsv41IndexQuantize(Float32Array.from({ length: 64 * 128 }, (_, i) => Math.cos(i * 0.17)));
  const w = bf16BytesToF32(bf16Round(Float32Array.from({ length: count * 32 }, (_, i) => Math.sin(i * 0.43))));
  keys.h2d(packDsv41Index(k)); weights.h2d(bf16Round(w));
  using pages = ws.alloc([1], "I32"); pages.fill(0, 1);
  using scores = ops.dsv4IndexScores(quantized, keys, weights, p, pages, 1, capacity);
  const expected = new Float32Array(count * capacity).fill(-Infinity);
  positions.forEach((position, row) => expected.set(dsv41IndexScores(expectedQ.subarray(row * 4096, (row + 1) * 4096),
    Array.from({ length: position + 1 }, (_, j) => k.subarray(j * 128, (j + 1) * 128)), w.subarray(row * 32, (row + 1) * 32)), row * capacity));
  const bytes = Buffer.alloc(scores.bytes); scores.d2h(bytes);
  assert.deepEqual(bytes, bf16Round(expected));
  using selected = ops.dsv41Select(scores, p, 1);
  assert.deepEqual(selected.readInt32LEArray(), positions.flatMap(position => Array.from({ length: 512 }, (_, i) => i <= position ? i : -1)));
});

for (const rows of [1, 4, 64, 65, 128]) for (const ratio of [1, 2] as const) {
  it(`V4.1 FP8 scoring preserves paged causality and BF16 accuracy under replay, Q${rows}, ratio${ratio}`, () => {
    using ops = new GlmOps(0);
    using ws = new WorkspaceBase(ops);
    using manager = new CaptureManager(ops);
    const capacity = 1088, pageCount = 34, stride = 17 * ratio;
    const pageIds = Array.from({ length: 2 * stride }, (_, i) => 2 * stride - i - 1);
    using pages = ws.alloc([2, stride], "I32"); pages.h2d(Buffer.from(Int32Array.from(pageIds).buffer));
    using positions = ws.alloc([rows], "I32"), batches = ws.alloc([rows], "I32");
    // Query groups deliberately cross sequence boundaries inside a prefill tile.
    const batchIds = Array.from({ length: rows }, (_, i) => Math.floor(i / 7) % 2);
    batches.h2d(Buffer.from(Int32Array.from(batchIds).buffer));
    using keys = ws.alloc([pageCount * ratio, 64, 132], "U8");
    const keyValues = Array.from({ length: 2 }, (_, b) => dsv41IndexQuantize(Float32Array.from(
      { length: capacity * 128 }, (_, i) => Math.sin(i * 0.137 + b) * 2 ** (Math.floor(i / 128) % 7 - 3))));
    const packed = Buffer.alloc(keys.bytes);
    for (let b = 0; b < 2; ++b) {
      const data = packDsv41Index(keyValues[b]);
      for (let k = 0; k < capacity; ++k) {
        const slot = pageIds[b * stride + Math.floor(k * ratio / 64)] * 64 + k % (64 / ratio);
        data.copy(packed, slot * 132, k * 132, (k + 1) * 132);
      }
    }
    keys.h2d(packed);
    using query = ws.alloc([rows, 4096], "BF16"), weights = ws.alloc([rows, 32], "BF16");
    const w = bf16BytesToF32(bf16Round(Float32Array.from({ length: rows * 32 }, (_, i) => Math.cos(i * 0.47))));
    weights.h2d(bf16Round(w));
    for (let pass = 0; pass < 6; ++pass) {
      const counts = Array.from({ length: rows }, (_, i) => [0, 31, 32, 63, 64, 513, 1025, 1088][(i + pass * 3) % 8]);
      // Zero valid compressed keys is legal at position zero for ratio two.
      if (ratio === 1) for (let i = 0; i < rows; ++i) counts[i] = Math.max(1, counts[i]);
      positions.h2d(Buffer.from(Int32Array.from(counts, n => Math.max(0, n * ratio - 1)).buffer));
      const q = bf16BytesToF32(bf16Round(Float32Array.from({ length: rows * 4096 }, (_, i) => Math.sin(i * 0.29 + pass) * 2 ** (Math.floor(i / 128) % 5 - 2))));
      query.h2d(bf16Round(q));
      using result = manager.run({ query, keys, weights, positions, pages, batches }, (_capturing, input) => {
        using quantized = ops.dsv41IndexQuantize(input.query, input.positions);
        return ops.dsv4IndexScores(quantized, input.keys, input.weights, input.positions, input.pages, ratio, capacity, input.batches);
      }, ["dsv41.scores", rows, ratio]) as Tensor;
      const data = Buffer.alloc(result.bytes); result.d2h(data);
      const actual = bf16BytesToF32(data), qQuant = dsv41IndexQuantize(q);
      using selection = ops.dsv41Select(result, positions, ratio);
      const selectedIds = selection.readInt32LEArray();
      // Cover both sides of mixed-sequence tiles and the last, possibly partial tile.
      for (const row of new Set([0, Math.min(2, rows - 1), Math.min(3, rows - 1), Math.min(7, rows - 1), rows - 1])) {
        const expectedIds = Array.from({ length: counts[row] }, (_, k) => k);
        if (counts[row] > 512) expectedIds.sort((a, b) => actual[row * capacity + b] - actual[row * capacity + a] || a - b);
        assert.deepEqual(selectedIds.slice(row * 512, (row + 1) * 512),
          Array.from({ length: 512 }, (_, i) => expectedIds[i] ?? -1));
        const kv = keyValues[batchIds[row]];
        const reference = dsv41IndexScores(qQuant.subarray(row * 4096, (row + 1) * 4096),
          Array.from({ length: counts[row] }, (_, k) => kv.subarray(k * 128, (k + 1) * 128)), w.subarray(row * 32, (row + 1) * 32));
        const rounded = bf16BytesToF32(bf16Round(reference));
        for (let k = 0; k < capacity; ++k) {
          const got = actual[row * capacity + k];
          if (k >= counts[row]) assert.equal(got, -Infinity, `masked row=${row} key=${k}`);
          else {
            // FP32 reduction order can move a value across one BF16 rounding
            // boundary. Bound that difference, rather than requiring FP64 bits.
            const ulp = 2 ** (Math.floor(Math.log2(Math.max(Math.abs(reference[k]), 2 ** -126))) - 7);
            assert.ok(Math.abs(got - rounded[k]) <= ulp + 1e-5, `row=${row} key=${k}: ${got} vs ${rounded[k]}`);
          }
        }
      }
    }
    assert([...manager.captured.values()].some(graph => graph.graphExec !== null));
  });
}

it("V4.1 scoring handles an 8192-query prefill and refreshed causal bounds under capture", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  using manager = new CaptureManager(ops);
  const rows = 8192, capacity = 128;
  using input = ws.alloc([rows, 4096], "BF16"), positions = ws.alloc([rows], "I32");
  using keys = ws.alloc([2, 64, 132], "U8"), weights = ws.alloc([rows, 32], "BF16");
  using pages = ws.alloc([2], "I32"); pages.h2d(Buffer.from(Int32Array.of(1, 0).buffer));
  keys.h2d(packDsv41Index(new Float32Array(128 * 128).fill(1)));
  weights.fill(1, weights.numElements);
  for (let pass = 0; pass < 6; ++pass) {
    input.fill(pass === 1 ? -1 : 1, input.numElements);
    const p = Int32Array.from({ length: rows }, (_, i) => (i + pass * 31) % capacity);
    positions.h2d(Buffer.from(p.buffer));
    using output = manager.run({ input, positions, keys, weights, pages }, (_capture, retained) => {
      using q = ops.dsv41IndexQuantize(retained.input, retained.positions);
      return ops.dsv4IndexScores(q, retained.keys, retained.weights, retained.positions, retained.pages, 1, capacity);
    }, ["large-indexer"]) as Tensor;
    const data = Buffer.alloc(output.bytes); output.d2h(data);
    const values = bf16BytesToF32(data);
    for (let r = 0; r < rows; ++r) for (let k = 0; k < capacity; ++k)
      assert.equal(values[r * capacity + k], k <= p[r] ? (pass === 1 ? 0 : 64) : -Infinity);
  }
  assert([...manager.captured.values()].some(graph => graph.graphExec !== null));
});

it("V4.1 packed index cache addresses records beyond the signed 32-bit byte boundary", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const page = Math.ceil(2 ** 31 / (64 * 132));
  using keys = ws.alloc([page + 1, 64, 132], "U8");
  using key = ws.alloc([1, 128], "BF16"); key.fill(1, key.numElements);
  using positions = ws.alloc([1], "I32"); positions.fill(0, 1);
  using pages = ws.alloc([1], "I32"); pages.h2d(Buffer.from(Int32Array.of(page).buffer));
  ops.dsv4IndexAppend(keys, key, positions, pages, 1);
  using query = ws.alloc([1, 32 * 132], "U8"); query.h2d(packDsv41Index(new Float32Array(4096).fill(1)));
  using weights = ws.alloc([1, 32], "BF16"); weights.fill(1, weights.numElements);
  using scores = ops.dsv4IndexScores(query, keys, weights, positions, pages, 1, 1);
  const data = Buffer.alloc(scores.bytes); scores.d2h(data);
  assert.equal(bf16BytesToF32(data)[0], 64);
});

it("V4.1 BF16 selection respects ties at token and candidate-block cutoffs", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const count = 2051 * 8;
  using scores = ws.alloc([1, count], "BF16"), position = ws.alloc([1], "I32");
  position.h2d(Buffer.from(Int32Array.of(count - 1).buffer));
  const values = bf16BytesToF32(bf16Round(Float32Array.from({ length: count }, (_, i) => 100 + (i % 97) * 0.01)));
  scores.h2d(bf16Round(values));
  using candidates = ops.dsv41Candidates(scores, position);
  const maxima = Array.from({ length: count / 8 }, (_, b) => Math.max(...values.subarray(b * 8, b * 8 + 8)));
  const blocks = maxima.map((_, i) => i).slice(0, -1).sort((a, b) => maxima[b] - maxima[a] || a - b).slice(0, 2047);
  blocks.push(maxima.length - 1);
  const allowed = new Set(blocks);
  assert.deepEqual(candidates.readInt32LEArray(), maxima.map((_, b) => Number(allowed.has(b))));
  // A bounded score buffer must never make either selector read beyond it.
  position.h2d(Buffer.from(Int32Array.of(count + 1023).buffer));
  using boundedCandidates = ops.dsv41Candidates(scores, position);
  assert.deepEqual(boundedCandidates.readInt32LEArray(), candidates.readInt32LEArray());
  using selected = ops.dsv41Select(scores, position, 1, candidates);
  const expected = Array.from({ length: count }, (_, i) => i).filter(i => allowed.has(Math.floor(i / 8)))
    .sort((a, b) => values[b] - values[a] || a - b).slice(0, 512);
  assert.deepEqual(selected.readInt32LEArray(), expected);
});
