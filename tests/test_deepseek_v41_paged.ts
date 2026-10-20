import { it } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { ParallelOps } from "../src/parallel_ops";
import { WorkspaceBase } from "../src/workspace";
import { writeDsv4Cache } from "./deepseek_v41_oracles";
import type { Tensor } from "../src/tensor";
import { DeepseekV41Cache } from "../src/deepseek_v41_cache";
import type { DeepseekV41Model } from "../src/deepseek_v41_model";
import { CaptureManager } from "../src/capture-manager";

function bytes(t: Tensor): Buffer { const b = Buffer.alloc(t.bytes); t.d2h(b); return b; }

for (const world of [1, 2]) it(`DSV4 batches unequal sequences without cross-sequence pages or scores, TP${world}`, () => {
  using gpu0 = new GlmOps(0);
  using gpu1 = world === 2 ? new GlmOps(1) : undefined;
  using parallel = gpu1 ? new ParallelOps([gpu0, gpu1]) : undefined;
  const ops = parallel ?? gpu0;
  using ws = new WorkspaceBase(ops);
  const lengths = [67, 130], tokens = lengths[0] + lengths[1], pageIds = [[3, 1, 0], [5, 4, 2]];
  using positions = ws.alloc([tokens], "I32");
  positions.h2d(Buffer.from(Int32Array.from(lengths.flatMap(n => Array.from({ length: n }, (_, i) => i))).buffer));
  using batches = ws.alloc([tokens], "I32");
  batches.h2d(Buffer.from(Int32Array.from(lengths.flatMap((n, b) => Array(n).fill(b))).buffer));
  using pages = ws.alloc([2, 3], "I32"); pages.h2d(Buffer.from(Int32Array.from(pageIds.flat()).buffer));
  using input = ws.alloc([tokens, 512], "BF16");
  using keys = ws.alloc([tokens, 128], "BF16");
  for (const t of [input, keys]) {
    t.fill(1, t.numElements);
    using second = t.narrow(lengths[0], lengths[1]); second.fill(2, second.numElements);
  }
  using rawQuery = ws.alloc([tokens, 32 * 128], "BF16"); rawQuery.fill(1, rawQuery.numElements);
  using query = ops.dsv41IndexQuantize(rawQuery, positions);
  using weights = ws.alloc([tokens, 32], "BF16"); weights.fill(1, weights.numElements);
  for (const ratio of [1, 2] as const) {
    using cache = ws.alloc([6, 64, 584], "U8"); cache.fill(0, cache.numElements);
    using expected = ws.alloc(cache.shape, "U8"); expected.fill(0, expected.numElements);
    ops.dsv4CacheAppend(cache, input, positions, pages, ratio, batches);
    let offset = 0;
    for (let b = 0; b < 2; b++) {
      using row = pages.narrow(b, 1); using sequencePages = row.reshape([3]);
      using x = input.narrow(offset, lengths[b]); using p = positions.narrow(offset, lengths[b]);
      ops.dsv4CacheAppend(expected, x, p, sequencePages, ratio);
      offset += lengths[b];
    }
    assert.deepEqual(bytes(cache), bytes(expected));
    using keyCache = ws.alloc([6, 64, 132], "U8"); keyCache.fill(0, keyCache.numElements);
    ops.dsv4IndexAppend(keyCache, keys, positions, pages, ratio, batches);
    using scores = ops.dsv4IndexScores(query, keyCache, weights, positions, pages, ratio, 130, batches);
    const scored = bf16BytesToF32(bytes(scores));
    for (let t = 0; t < tokens; t++) for (let k = 0; k < 130; k++) {
      const b = t < lengths[0] ? 0 : 1, p = b ? t - lengths[0] : t;
      assert.equal(scored[t * 130 + k], k < Math.floor((p + 1) / ratio) ? 64 * (b + 1) : -Infinity);
    }
    using selected = ws.alloc([tokens, 4], "I32");
    selected.h2d(Buffer.from(Int32Array.from({ length: tokens * 4 }, (_, i) => [32, -1, 0, 64][i % 4]).buffer));
    using selectedLengths = ws.alloc([tokens], "I32");
    using mapped = ops.dsv4TopkToSlots(selected, positions, pages, ratio, batches, selectedLengths);
    const mappedIds = mapped.readInt32LEArray(), counts = selectedLengths.readInt32LEArray();
    for (let t = 0; t < tokens; ++t) {
      const b = t < lengths[0] ? 0 : 1, p = b ? t - lengths[0] : t;
      const valid = [32, 0, 64].filter(id => id < Math.floor((p + 1) / ratio));
      assert.equal(counts[t], valid.length);
      const expected = valid.map(id => pageIds[b][Math.floor(id * ratio / 64)] * 64 + id % (64 / ratio));
      assert.deepEqual(mappedIds.slice(t * 4, (t + 1) * 4), [...expected, ...Array(4 - valid.length).fill(-1)]);
    }
  }
  using localLengths = ws.alloc([tokens], "I32");
  using local = ops.dsv4LocalSlots(positions, pages, batches, localLengths);
  assert.deepEqual(localLengths.readInt32LEArray(), lengths.flatMap(n => Array.from({ length: n }, (_, i) => Math.min(i + 1, 128))));
  const slots = local.readInt32LEArray();
  for (let t = 0; t < tokens; t++) for (let col = 0; col < 128; col++) {
    const b = t < lengths[0] ? 0 : 1, p = b ? t - lengths[0] : t, k = Math.max(0, p - 127) + col;
    assert.equal(slots[t * 128 + col], k <= p ? pageIds[b][Math.floor(k / 64)] * 64 + k % 64 : -1);
  }
});

it("DSV4 pages preserve every cache plane across prefix sharing and partial-page copies", () => {
  using ops = new GlmOps(0);
  using cache = new DeepseekV41Cache({ ops } as unknown as DeepseekV41Model, 8, 2, 512);
  cache.ensureSequence(0);
  cache.allocAppendPages(0, 65);
  const history = Array.from({ length: 65 }, (_, i) => i + 10);
  cache.reportTokens(0, history, 1000);
  const original = cache.sequences[0];
  const planes = [...cache.ckvData, ...cache.kData.filter(Boolean)];
  for (const [i, plane] of planes.entries()) {
    using page = plane.narrow(original.pages[1].id, 1);
    page.h2d(Buffer.alloc(page.bytes, i + 1));
  }
  cache.copySequence(1, 0);
  const copied = cache.sequences[1];
  assert.equal(copied.pages[0], original.pages[0]);
  assert.notEqual(copied.pages[1], original.pages[1]);
  assert.deepEqual(copied.getTokenIds(), history);
  for (const [i, plane] of planes.entries()) {
    using page = plane.narrow(copied.pages[1].id, 1);
    assert.deepEqual(bytes(page), Buffer.alloc(page.bytes, i + 1));
  }
  copied.truncate(63);
  cache.allocAppendPages(1, 1);
  assert.notEqual(copied.pages[0], original.pages[0], "rewriting a shared page uses COW");
  assert.deepEqual(original.getTokenIds(), history);
});

it("Engram three-token lookback reproduces full-history hashes at every chunk boundary", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const values = [2, 4, 8, 3, 9, 10, 12];
  const ints = (values: number[]) => { const t = ws.alloc([values.length], "I32"); t.h2d(Buffer.from(Int32Array.from(values).buffer)); return t; };
  using token = ints(values);
  using map = ints(Array.from({ length: 20 }, (_, i) => i));
  using primes = ints(Array(48).fill(101));
  using offsets = ints(Array.from({ length: 48 }, (_, i) => i * 101));
  using initialLookback = ints([2, 2, 2]);
  using expected = ops.dsv41Hash(token, initialLookback, map, primes, offsets);
  const hashes = expected.readInt32LEArray();
  for (let start = 0; start < values.length; start++) {
    const length = values.length - start;
    using input = token.narrow(start, length);
    using lookback = ints(Array.from({ length: 3 }, (_, j) => values[start - 3 + j] ?? 2));
    using actual = ops.dsv41Hash(input, lookback, map, primes, offsets);
    const result = actual.readInt32LEArray();
    for (let layer = 0; layer < 2; layer++) assert.deepEqual(result.slice(layer * length * 24, (layer + 1) * length * 24),
      hashes.slice((layer * values.length + start) * 24, (layer + 1) * values.length * 24));
  }
});

for (const world of [1, 2]) it(`DSV4 paged writes, slots and scores agree across permuted pages, TP${world}`, () => {
  using gpu0 = new GlmOps(0);
  using gpu1 = world === 2 ? new GlmOps(1) : undefined;
  using parallel = gpu1 ? new ParallelOps([gpu0, gpu1]) : undefined;
  const ops = parallel ?? gpu0;
  using ws = new WorkspaceBase(ops);
  const tokens = 131, pageIds = [3, 1, 4];
  using positions = ws.alloc([tokens], "I32");
  positions.h2d(Buffer.from(Int32Array.from({ length: tokens }, (_, i) => i).buffer));
  using pages = ws.alloc([3], "I32"); pages.h2d(Buffer.from(Int32Array.from(pageIds).buffer));
  using input = ws.alloc([tokens, 512], "BF16"); input.fill(1, input.numElements);
  using keys = ws.alloc([tokens, 128], "BF16"); keys.fill(1, keys.numElements);
  using rawQuery = ws.alloc([tokens, 32 * 128], "BF16"); rawQuery.fill(1, rawQuery.numElements);
  using query = ops.dsv41IndexQuantize(rawQuery, positions);
  using weights = ws.alloc([tokens, 32], "BF16"); weights.fill(1, weights.numElements);
  const slot = (ordinal: number, ratio: number) => pageIds[Math.floor(ordinal * ratio / 64)] * 64 + ordinal % (64 / ratio);
  for (const ratio of [1, 2] as const) {
    using cache = ws.alloc([5, 64, 584], "U8"); cache.fill(0, cache.numElements);
    ops.dsv4CacheAppend(cache, input, positions, pages, ratio);
    const actual = bytes(cache), expected = Buffer.alloc(3 * 64 * 584);
    for (let ordinal = 0; ordinal < Math.floor(tokens / ratio); ordinal++) writeDsv4Cache(expected, ordinal, new Float32Array(512).fill(1));
    for (let ordinal = 0; ordinal < Math.floor(tokens / ratio); ordinal++) {
      const physical = slot(ordinal, ratio);
      const dst = Math.floor(physical / 64) * 64 * 584, src = Math.floor(ordinal / 64) * 64 * 584;
      assert.deepEqual(actual.subarray(dst + physical % 64 * 576, dst + physical % 64 * 576 + 576),
        expected.subarray(src + ordinal % 64 * 576, src + ordinal % 64 * 576 + 576));
      assert.deepEqual(actual.subarray(dst + 64 * 576 + physical % 64 * 8, dst + 64 * 576 + physical % 64 * 8 + 8),
        expected.subarray(src + 64 * 576 + ordinal % 64 * 8, src + 64 * 576 + ordinal % 64 * 8 + 8));
    }
    assert.ok(actual.subarray(0, 64 * 584).every(x => x === 0), "unreferenced page untouched");
    using selected = ws.alloc([tokens, 4], "I32");
    selected.h2d(Buffer.from(Int32Array.from({ length: tokens * 4 }, (_, i) => [-1, 0, 32, 64][i % 4]).buffer));
    using mappedLengths = ws.alloc([tokens], "I32");
    using mapped = ops.dsv4TopkToSlots(selected, positions, pages, ratio, undefined, mappedLengths);
    const indices = mapped.readInt32LEArray();
    const counts = mappedLengths.readInt32LEArray();
    for (let p = 0; p < tokens; p++) {
      const valid = [0, 32, 64].filter(ordinal => ordinal < Math.floor((p + 1) / ratio));
      assert.equal(counts[p], valid.length);
      assert.deepEqual(indices.slice(p * 4, (p + 1) * 4),
        [...valid.map(ordinal => slot(ordinal, ratio)), ...Array(4 - valid.length).fill(-1)]);
    }
    using keyCache = ws.alloc([5, 64, 132], "U8"); keyCache.fill(0, keyCache.numElements);
    ops.dsv4IndexAppend(keyCache, keys, positions, pages, ratio);
    using scores = ops.dsv4IndexScores(query, keyCache, weights, positions, pages, ratio, tokens);
    const scoreBytes = bf16BytesToF32(bytes(scores));
    for (let p = 0; p < tokens; p++) for (let k = 0; k < tokens; k++) {
      assert.equal(scoreBytes[p * tokens + k], k < Math.floor((p + 1) / ratio) ? 64 : -Infinity);
    }
  }
  using local = ops.dsv4LocalSlots(positions, pages);
  const indices = local.readInt32LEArray();
  for (let p = 0; p < tokens; p++) for (let j = 0; j < 128; j++) {
    const key = Math.max(0, p - 127) + j;
    assert.equal(indices[p * 128 + j], key <= p ? slot(key, 1) : -1);
  }
});

for (const ratio of [1, 2] as const) it(`DSV4 compact slots and lengths refresh under capture, ratio${ratio}`, () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  using manager = new CaptureManager(ops);
  const rows = 4, width = 513, pageIds = [[3, 1, 0], [5, 4, 2]];
  using positions = ws.alloc([rows], "I32");
  using pages = ws.alloc([2, 3], "I32");
  pages.h2d(Buffer.from(Int32Array.from(pageIds.flat()).buffer));
  using batches = ws.alloc([rows], "I32");
  batches.h2d(Buffer.from(Int32Array.from([0, 1, 0, 1]).buffer));
  using selected = ws.alloc([rows, width], "I32");
  using lengths = ws.alloc([rows], "I32");
  using localLengths = ws.alloc([rows], "I32");
  for (let pass = 0; pass < 4; ++pass) {
    const pos = Array.from({ length: rows }, (_, r) => [0, 31, 64, 129][(r + pass) % rows]);
    const ids = Int32Array.from({ length: rows * width }, (_, i) => (i + pass) % 3 === 0 ? -1 : (i * 17 + pass) % 150);
    positions.h2d(Buffer.from(Int32Array.from(pos).buffer));
    selected.h2d(Buffer.from(ids.buffer));
    using output = manager.run({ positions, pages, batches, selected, lengths, localLengths }, (_capture, input) => {
      using local = ops.dsv4LocalSlots(input.positions, input.pages, input.batches, input.localLengths);
      return ops.dsv4TopkToSlots(input.selected, input.positions, input.pages, ratio, input.batches, input.lengths);
    }, ["compact-slots", ratio]) as Tensor;
    const actual = output.readInt32LEArray(), counts = lengths.readInt32LEArray();
    assert.deepEqual(localLengths.readInt32LEArray(), pos.map(p => Math.min(p + 1, 128)));
    for (let r = 0; r < rows; ++r) {
      const valid = Array.from(ids.subarray(r * width, (r + 1) * width))
        .filter(id => id >= 0 && id < Math.floor((pos[r] + 1) / ratio));
      assert.equal(counts[r], valid.length);
      const expected = valid.map(id => pageIds[r % 2][Math.floor(id * ratio / 64)] * 64 + id % (64 / ratio));
      assert.deepEqual(actual.slice(r * width, (r + 1) * width), [...expected, ...Array(width - valid.length).fill(-1)]);
    }
  }
  assert([...manager.captured.values()].some(graph => graph.graphExec !== null));
});
