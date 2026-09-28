import test from "node:test";
import assert from "node:assert/strict";
import type { MemcpyBatchEntry } from "../src/device_ops";
import { Tensor } from "../src/tensor";
import { PAGE_SIZE, PagedKVCache, Sequence } from "../src/paged_kv";

function makeCache() {
  const copies: number[][] = [];
  // Exercise real cache bookkeeping without constructing GPU tensors/workspaces.
  const cache: PagedKVCache = Object.assign(Object.create(PagedKVCache.prototype), {
    pageSize: PAGE_SIZE,
    availablePages: Array.from({ length: 16 }, (_, i) => i),
    sequences: [],
    staging: new Map<number, Sequence>(),
    copyPage(src: number, dst: number) { copies.push([src, dst]); },
  });
  return { cache, copies };
}

const tokens = (length: number) => Array.from({ length }, (_, i) => i + 100);

for (const staged of [false, true]) {
  for (const copyPartial of [false, true]) {
    for (const allocLen of [63, 64, 127, 128]) {
      test(`pending target boundary: allocLen=${allocLen}, staged=${staged}, copyPartial=${copyPartial}`, () => {
        const { cache, copies } = makeCache();
        const source = cache.ensureSequence(0);
        cache.allocAppendPages(0, allocLen);
        const reported = tokens(allocLen + 1);
        cache.reportTokens(0, reported.slice(0, -1), reported.at(-1));
        const sourcePages = [...source.pages];
        const targetIndex = staged ? 0 : 1;
        if (staged) cache.stageSequence(0, 123);

        const input = [...reported, 999];
        const fullPages = Math.floor(allocLen / PAGE_SIZE);
        // Preserve the existing policy: copyPartial requires a full shared page.
        const reused = copyPartial && fullPages > 0 ? allocLen : fullPages * PAGE_SIZE;
        const suffix = cache.prefixMatch(targetIndex, input, copyPartial);
        const target = cache.sequences[targetIndex];
        assert.deepEqual(suffix, input.slice(reused));
        assert.equal(target.allocLen, reused);
        assert.deepEqual(target.getTokenIds(), input.slice(0, reused));
        assert.equal(target.pages.length, Math.ceil(reused / PAGE_SIZE));
        for (let i = 0; i < fullPages; i++) {
          assert.equal(target.pages[i], sourcePages[i]);
          assert.equal(sourcePages[i].refs, 2);
        }
        if (allocLen % PAGE_SIZE) assert.equal(sourcePages[fullPages].refs, 1, "partial page must not be shared");
        if (reused % PAGE_SIZE !== 0) {
          assert.deepEqual(copies, [[sourcePages[fullPages].id, target.pages[fullPages].id]]);
          assert.notEqual(target.pages[fullPages], sourcePages[fullPages]);
        } else {
          assert.deepEqual(copies, []);
        }

        cache.allocAppendPages(targetIndex, suffix.length);
        cache.reportTokens(targetIndex, suffix);
        assert.deepEqual(target.getTokenIds(), input, "suffix must not duplicate the source peek");
        assert.deepEqual(source.getTokenIds(), reported.slice(0, -1));
        assert.equal(source.allocLen, allocLen);
      });
    }
  }
}

test("candidate selection prefers materialized KV over an earlier matching pending target", () => {
  const { cache } = makeCache();
  const input = tokens(65);
  for (const index of [0, 1]) {
    cache.ensureSequence(index);
    cache.allocAppendPages(index, 63 + index);
    cache.reportTokens(index, input.slice(0, 63 + index), input[63 + index]);
  }
  assert.deepEqual(cache.prefixMatch(2, input), input.slice(64));
  assert.equal(cache.sequences[2].pages[0], cache.sequences[1].pages[0]);
  assert.equal(cache.sequences[0].pages[0].refs, 1);
});

for (const allocLen of [63, 64, 127, 128]) {
  test(`self-match leaves the pending prompt input unprocessed at allocLen=${allocLen}`, () => {
    const { cache, copies } = makeCache();
    const source = cache.ensureSequence(0);
    cache.allocAppendPages(0, allocLen);
    const reported = tokens(allocLen + 1);
    cache.reportTokens(0, reported.slice(0, -1), reported.at(-1));
    const pages = [...source.pages];
    assert.deepEqual(cache.prefixMatch(0, reported), reported.slice(-1));
    assert.deepEqual(cache.prefixMatch(0, [...reported, 999], true), [reported.at(-1), 999]);
    assert.equal(source.allocLen, allocLen);
    assert.deepEqual(source.getTokenIds(), reported.slice(0, -1));
    assert.equal(source.reportedTokenCount(), source.allocLen);
    pages.forEach((page, i) => {
      assert.equal(source.pages[i], page);
      assert.equal(page.refs, 1);
    });
    assert.deepEqual(copies, []);
  });
}

test("copyPartial trims metadata at a mismatch before the source KV boundary", () => {
  const { cache } = makeCache();
  const source = cache.ensureSequence(0);
  cache.allocAppendPages(0, 70);
  const reported = tokens(71);
  cache.reportTokens(0, reported.slice(0, -1), reported.at(-1));
  const input = [...reported.slice(0, 67), 999, 998];
  const suffix = cache.prefixMatch(1, input, true);
  assert.deepEqual(suffix, [999, 998]);
  assert.equal(cache.sequences[1].allocLen, 67);
  assert.deepEqual(cache.sequences[1].getTokenIds(), input.slice(0, 67));
  cache.allocAppendPages(1, suffix.length);
  cache.reportTokens(1, suffix);
  assert.deepEqual(cache.sequences[1].getTokenIds(), input);
  assert.deepEqual(source.getTokenIds(), reported.slice(0, -1));
});

test("self-match with a different pending target preserves the committed prefix", () => {
  const { cache } = makeCache();
  const source = cache.ensureSequence(0);
  cache.allocAppendPages(0, 127);
  const reported = tokens(128);
  cache.reportTokens(0, reported.slice(0, -1), reported.at(-1));
  const input = [...reported.slice(0, 127), 999];
  assert.deepEqual(cache.prefixMatch(0, input), [999]);
  assert.equal(source.allocLen, 127);
  assert.deepEqual(source.getTokenIds(), input.slice(0, 127));
});

// Shell caches for copyPrefixFrom: real bookkeeping methods plus fake layer
// tensors (shape/type only) so layerCachePairs/assertCacheLayoutCompatible and
// the batched-memcpy construction run without GPU allocations.
function makeCopyCache(overrides?: { bytesPerToken?: number }) {
  const bytesPerToken = overrides?.bytesPerToken ?? 656;
  const entries: MemcpyBatchEntry[] = [];
  const tensor = (shape: number[], type: string) => ({ shape: [...shape], type } as unknown as Tensor);
  const cache: PagedKVCache = Object.assign(Object.create(PagedKVCache.prototype), {
    pageSize: PAGE_SIZE,
    availablePages: Array.from({ length: 16 }, (_, i) => i),
    sequences: [],
    staging: new Map<number, Sequence>(),
    copyPage(_srcPageId: number, _dstPageId: number) {},
    nLayers: 1,
    nKv: 1,
    hd: 1,
    maxPages: 16,
    maxBatch: 4,
    contextParallel: false,
    sparseMode: true,
    bytesPerToken,
    kData: [tensor([16, PAGE_SIZE, 128], "U8")],
    kScaleData: [tensor([16, PAGE_SIZE], "F32")],
    vData: [],
    ckvData: [tensor([16, PAGE_SIZE, bytesPerToken], "U8")],
    kpeData: [],
    ops: { memcpyBatchAsync(copies: readonly MemcpyBatchEntry[]) { entries.push(...copies); } },
  });
  return { cache, entries };
}

// Decodes batched-memcpy entries to [srcPage, dstPage, pages, rowBytes] so
// tests can assert exactly which pages were copied for each cache tensor.
function pageCopies(entries: readonly MemcpyBatchEntry[]) {
  return entries.map(copy => {
    const rowBytes = Tensor.byteCount(copy.dst.shape.slice(1), copy.dst.type);
    return [(copy.srcOffset ?? 0) / rowBytes, (copy.dstOffset ?? 0) / rowBytes, copy.bytes / rowBytes, rowBytes];
  });
}

const K_ROW = PAGE_SIZE * 128;
const K_SCALE_ROW = PAGE_SIZE * 4;
const CKV_ROW = PAGE_SIZE * 656;

test("copyPrefixFrom copies full pages and returns the partial-page tail", () => {
  const { cache: src } = makeCopyCache();
  const { cache: dst, entries } = makeCopyCache();
  const srcSeq = src.ensureSequence(0);
  src.allocAppendPages(0, 130);
  src.reportTokens(0, tokens(130), 77);

  const suffix = dst.copyPrefixFrom(src, srcSeq, 0);
  const dstSeq = dst.sequences[0];
  assert.deepEqual(suffix, tokens(130).slice(128));
  assert.equal(dstSeq.allocLen, 128);
  assert.deepEqual(dstSeq.getTokenIds(), tokens(128));
  assert.deepEqual(dstSeq.pages.map(page => page.id), [0, 1]);
  assert.equal(dstSeq.targetToken, tokens(130)[128], "first uncopied token must be pending");
  assert.deepEqual(pageCopies(entries), [[0, 0, 2, K_ROW], [0, 0, 2, K_SCALE_ROW], [0, 0, 2, CKV_ROW]]);
  assert.equal(srcSeq.allocLen, 130);
  assert.deepEqual(srcSeq.getTokenIds(), tokens(130));
  assert.equal(srcSeq.targetToken, 77);

  // The caller finishes the restore by prefilling the tail and feeding the source's pending token.
  dst.allocAppendPages(0, suffix.length);
  dst.reportTokens(0, suffix, 77);
  assert.deepEqual(dstSeq.getTokenIds(), tokens(130));
  assert.equal(dstSeq.targetToken, 77);
});

test("copyPrefixFrom into a cache that already holds the sequence copies nothing", () => {
  const { cache: src } = makeCopyCache();
  const { cache: dst, entries } = makeCopyCache();
  const srcSeq = src.ensureSequence(0);
  src.allocAppendPages(0, 130);
  src.reportTokens(0, tokens(130), 77);
  const dstSeq = dst.ensureSequence(0);
  dst.allocAppendPages(0, 130);
  dst.reportTokens(0, tokens(130), 77);

  const suffix = dst.copyPrefixFrom(src, srcSeq, 0);
  assert.deepEqual(suffix, [tokens(130)[129]], "complete match returns its truncated final token");
  assert.equal(dstSeq.allocLen, 129);
  assert.deepEqual(dstSeq.getTokenIds(), tokens(129));
  assert.equal(dstSeq.targetToken, tokens(130)[129]);
  assert.deepEqual(entries, [], "complete self-restore must not copy pages");

  dst.allocAppendPages(0, suffix.length);
  dst.reportTokens(0, suffix, 77);
  assert.deepEqual(dstSeq.getTokenIds(), tokens(130));
  assert.equal(dstSeq.targetToken, 77);
});

test("copyPrefixFrom shares full pages with the destination's own cache", () => {
  const { cache: src } = makeCopyCache();
  const { cache: dst, entries } = makeCopyCache();
  const srcSeq = src.ensureSequence(0);
  src.allocAppendPages(0, 130);
  src.reportTokens(0, tokens(130));
  dst.ensureSequence(1);
  dst.allocAppendPages(1, 64);
  dst.reportTokens(1, tokens(64));

  const suffix = dst.copyPrefixFrom(src, srcSeq, 0);
  const dstSeq = dst.sequences[0];
  assert.deepEqual(suffix, tokens(130).slice(128));
  assert.equal(dstSeq.allocLen, 128);
  assert.deepEqual(dstSeq.getTokenIds(), tokens(128));
  assert.equal(dstSeq.targetToken, tokens(130)[128]);
  assert.equal(dstSeq.pages[0], dst.sequences[1].pages[0], "matched full page must be shared, not copied");
  assert.equal(dstSeq.pages[0].refs, 2);
  assert.deepEqual(pageCopies(entries), [[1, 1, 1, K_ROW], [1, 1, 1, K_SCALE_ROW], [1, 1, 1, CKV_ROW]]);
});

test("copyPrefixFrom leaves no suffix for a page-aligned sequence and keeps the pending token", () => {
  const { cache: src } = makeCopyCache();
  const { cache: dst, entries } = makeCopyCache();
  const srcSeq = src.ensureSequence(0);
  src.allocAppendPages(0, 128);
  src.reportTokens(0, tokens(128), 77);

  const suffix = dst.copyPrefixFrom(src, srcSeq, 0);
  const dstSeq = dst.sequences[0];
  assert.deepEqual(suffix, []);
  assert.equal(dstSeq.allocLen, 128);
  assert.deepEqual(dstSeq.getTokenIds(), tokens(128));
  assert.equal(dstSeq.targetToken, 77, "aligned restore leaves the source's pending token for decode");
  assert.deepEqual(pageCopies(entries), [[0, 0, 2, K_ROW], [0, 0, 2, K_SCALE_ROW], [0, 0, 2, CKV_ROW]]);
});

test("copyPrefixFrom from an empty sequence clears the destination", () => {
  const { cache: src } = makeCopyCache();
  const { cache: dst, entries } = makeCopyCache();
  const srcSeq = src.ensureSequence(0);
  dst.ensureSequence(0);
  dst.allocAppendPages(0, 64);
  dst.reportTokens(0, tokens(64));

  const suffix = dst.copyPrefixFrom(src, srcSeq, 0);
  assert.deepEqual(suffix, []);
  const dstSeq = dst.sequences[0];
  assert.equal(dstSeq.allocLen, 0);
  assert.equal(dstSeq.pages.length, 0);
  assert.deepEqual(dstSeq.getTokenIds(), []);
  assert.deepEqual(entries, []);
});

test("copyPrefixFrom rejects an incompatible cache layout", () => {
  const { cache: src } = makeCopyCache({ bytesPerToken: 512 });
  const { cache: dst } = makeCopyCache();
  const srcSeq = src.ensureSequence(0);
  src.allocAppendPages(0, 64);
  src.reportTokens(0, tokens(64));
  assert.throws(() => dst.copyPrefixFrom(src, srcSeq, 0), /KV packing mismatch/);
});
