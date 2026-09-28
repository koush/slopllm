import test from "node:test";
import assert from "node:assert/strict";
import type { DeviceOps, MemcpyBatchEntry } from "../src/device_ops";
import { PrefixTierPolicy } from "../src/generation-scheduler";
import { Tensor } from "../src/tensor";
import { PAGE_SIZE, PageAllocationError, PagedKVCache, Sequence } from "../src/paged_kv";

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

for (const donorKind of ["self", "active", "staged"]) {
  for (const length of [0, 1, 63, 64, 65, 128, 130]) {
    test(`exact match uses only full pages without allocations: ${donorKind}, length=${length}`, () => {
      const { cache, copies } = makeCache();
      cache.availablePages = Array.from({ length: Math.ceil(length / PAGE_SIZE) }, (_, i) => i);
      const source = cache.ensureSequence(0);
      const input = tokens(length);
      cache.allocAppendPages(0, length);
      cache.reportTokens(0, input, 999);
      const sourcePages = [...source.pages];
      let index = donorKind === "self" ? 0 : 1;
      if (donorKind === "staged") {
        cache.stageSequence(0, 123);
        index = 0;
      }
      cache.onPagePressure = () => assert.fail("prefix matching must not allocate or cause eviction");
      assert.equal(cache.availablePages.length, 0);
      const kept = Math.floor(Math.max(0, length - 1) / PAGE_SIZE) * PAGE_SIZE;
      assert.deepEqual(cache.prefixMatch(index, input), input.slice(kept));
      const matched = cache.sequences[index];
      assert.equal(matched.allocLen, kept);
      assert.deepEqual(matched.getTokenIds(), input.slice(0, kept));
      assert.equal(matched.targetToken, input[kept]);
      assert.deepEqual(copies, []);
      for (let i = 0; i < kept / PAGE_SIZE; i++) {
        assert.equal(matched.pages[i], sourcePages[i]);
        assert.equal(sourcePages[i].refs, donorKind === "self" ? 1 : 2);
      }
      if (donorKind !== "self") {
        assert.deepEqual(source.getTokenIds(), input);
        assert.equal(source.targetToken, 999);
      }
    });
  }
}

test("self-match releases partial and unreported reserved pages", () => {
  const { cache, copies } = makeCache();
  const seq = cache.ensureSequence(0);
  cache.allocAppendPages(0, 4 * PAGE_SIZE);
  cache.reportTokens(0, tokens(70));
  const suffix = cache.prefixMatch(0, tokens(80));
  assert.deepEqual(suffix, tokens(80).slice(PAGE_SIZE));
  assert.equal(seq.allocLen, PAGE_SIZE);
  assert.equal(seq.pages.length, 1);
  assert.equal(cache.availablePages.length, 15);
  assert.deepEqual(copies, []);
});

for (const staged of [false, true]) {
  for (const allocLen of [63, 64, 127, 128]) {
    test(`pending target boundary: allocLen=${allocLen}, staged=${staged}`, () => {
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
      const reused = fullPages * PAGE_SIZE;
      const suffix = cache.prefixMatch(targetIndex, input);
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
      assert.deepEqual(copies, []);

      cache.allocAppendPages(targetIndex, suffix.length);
      cache.reportTokens(targetIndex, suffix);
      assert.deepEqual(target.getTokenIds(), input, "suffix must not duplicate the source peek");
      assert.deepEqual(source.getTokenIds(), reported.slice(0, -1));
      assert.equal(source.allocLen, allocLen);
    });
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
    const reused = Math.floor(allocLen / PAGE_SIZE) * PAGE_SIZE;
    assert.deepEqual(cache.prefixMatch(0, reported), reported.slice(reused));
    assert.deepEqual(cache.prefixMatch(0, [...reported, 999]), [...reported.slice(reused), 999]);
    assert.equal(source.allocLen, reused);
    assert.deepEqual(source.getTokenIds(), reported.slice(0, reused));
    assert.equal(source.reportedTokenCount(), source.allocLen);
    pages.slice(0, reused / PAGE_SIZE).forEach((page, i) => {
      assert.equal(source.pages[i], page);
      assert.equal(page.refs, 1);
    });
    assert.deepEqual(copies, []);
  });
}

test("a mismatch within a page leaves the entire partial page for prefill", () => {
  const { cache } = makeCache();
  const source = cache.ensureSequence(0);
  cache.allocAppendPages(0, 70);
  const reported = tokens(71);
  cache.reportTokens(0, reported.slice(0, -1), reported.at(-1));
  const input = [...reported.slice(0, 67), 999, 998];
  const suffix = cache.prefixMatch(1, input);
  assert.deepEqual(suffix, input.slice(64));
  assert.equal(cache.sequences[1].allocLen, 64);
  assert.deepEqual(cache.sequences[1].getTokenIds(), input.slice(0, 64));
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
  assert.deepEqual(cache.prefixMatch(0, input), input.slice(64));
  assert.equal(source.allocLen, 64);
  assert.deepEqual(source.getTokenIds(), input.slice(0, 64));
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
  assert.deepEqual(suffix, tokens(130).slice(128), "complete match returns the partial-page tail");
  assert.equal(dstSeq.allocLen, 128);
  assert.deepEqual(dstSeq.getTokenIds(), tokens(128));
  assert.equal(dstSeq.targetToken, tokens(130)[128]);
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

test("repeated aligned restores share complete pages without final-token repriming", () => {
  const { cache: src } = makeCopyCache();
  const { cache: dst, entries } = makeCopyCache();
  const source = src.ensureSequence(0);
  src.allocAppendPages(0, 128);
  src.reportTokens(0, tokens(128), 77);
  // Exactly enough destination space for the first restore, with no spare
  // page for copy-on-write during subsequent self/cross-sequence matches.
  dst.availablePages = [0, 1];
  assert.deepEqual(dst.copyPrefixFrom(src, source, 0), []);
  entries.length = 0;
  for (const index of [0, 1, 2]) {
    assert.deepEqual(dst.copyPrefixFrom(src, source, index), []);
    assert.equal(dst.sequences[index].allocLen, 128);
    assert.deepEqual(dst.sequences[index].getTokenIds(), tokens(128));
    assert.equal(dst.sequences[index].targetToken, 77);
    assert.equal(dst.sequences[index].pages[0], dst.sequences[0].pages[0]);
    assert.equal(dst.sequences[index].pages[1], dst.sequences[0].pages[1]);
  }
  assert.deepEqual(entries, []);
  assert.equal(dst.sequences[0].pages[0].refs, 3);
  assert.equal(dst.sequences[0].pages[1].refs, 3);
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

test("copyPrefixFrom maxPages clamps the materialized page count", () => {
  const { cache: src } = makeCopyCache();
  const { cache: dst, entries } = makeCopyCache();
  const srcSeq = src.ensureSequence(0);
  src.allocAppendPages(0, 192);
  src.reportTokens(0, tokens(192), 77);

  const suffix = dst.copyPrefixFrom(src, srcSeq, 0, 2);
  const dstSeq = dst.sequences[0];
  assert.deepEqual(suffix, tokens(192).slice(128));
  assert.equal(dstSeq.allocLen, 128);
  assert.deepEqual(dstSeq.getTokenIds(), tokens(128));
  assert.equal(dstSeq.pages.length, 2, "the clamped third page must not be allocated");
  assert.equal(dstSeq.targetToken, tokens(192)[128]);
  assert.deepEqual(pageCopies(entries), [[0, 0, 2, K_ROW], [0, 0, 2, K_SCALE_ROW], [0, 0, 2, CKV_ROW]]);
});

// ---------------------------------------------------------------------------
// PrefixTierPolicy. Shell caches from makeCopyCache (page copies land in the
// per-cache entries array), plus a fake ops recording synchronizeAsync calls —
// the policy's only direct ops use in these tests.
// ---------------------------------------------------------------------------

function makePrefixOps() {
  const state = { syncs: 0 };
  const ops = {
    synchronizeAsync: async () => {
      state.syncs++;
    },
  } as unknown as DeviceOps;
  return { ops, state };
}

test("prefixMatch without a host tier passes through to the device cache", async () => {
  const { cache: gpu } = makeCopyCache();
  const { ops, state } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops);
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, 64);
  gpu.reportTokens(0, tokens(64));
  policy.retainSequence(0);
  const retained = gpu.staging.get(-1)!;
  assert.equal(retained.pages.length, 1);

  const prompt = [...tokens(64), 7, 8, 9];
  const suffixes = await policy.prefixMatch([prompt]);
  assert.deepEqual(suffixes, [[7, 8, 9]]);
  assert.equal(state.syncs, 0, "no host tier => no drain to await");
  const slot = gpu.sequences[0];
  assert.equal(slot.pages[0], retained.pages[0], "primed slot shares the retained page");
  assert.equal(retained.pages[0].refs, 2);
});

test("prefixMatch restores a host-tier prefix into an empty device cache", async () => {
  const { cache: host } = makeCopyCache();
  const { cache: gpu, entries } = makeCopyCache();
  const { ops, state } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);

  const original = host.ensureSequence(0);
  host.allocAppendPages(0, 130);
  host.reportTokens(0, tokens(130), 77);
  const prompt = [...tokens(130), 900];

  const suffixes = await policy.prefixMatch([prompt]);
  // Restore copies both full pages; the 2-token partial tail plus the new
  // token remain for prefill. The suffix derives from the prompt.
  assert.deepEqual(suffixes, [prompt.slice(128)]);
  assert.equal(state.syncs, 1, "host tier => one drain per invocation");

  const slot = gpu.sequences[0];
  assert.equal(slot.allocLen, 128);
  assert.deepEqual(slot.getTokenIds(), tokens(128));
  assert.equal(slot.targetToken, tokens(130)[128]);
  assert.deepEqual(pageCopies(entries), [[0, 0, 2, K_ROW], [0, 0, 2, K_SCALE_ROW], [0, 0, 2, CKV_ROW]]);

  // The staged original survives with its pages shared by the active host
  // prime slot; the offload/restore machinery never reads host pages out from
  // under an active slot.
  assert.equal(host.staging.size, 1, "only the original is staged");
  assert.equal(original.pages[0].refs, 2);
  assert.equal(host.sequences[0].pages[0], original.pages[0]);
  // 130 tokens reserve three host pages (two full + the partial); only the
  // device consumed exactly two.
  assert.equal(host.availablePages.length, 13);
  assert.equal(gpu.availablePages.length, 14);
});

test("prefixMatch stages superseded host slots instead of restoring them", async () => {
  const { cache: host } = makeCopyCache();
  const { cache: gpu, entries } = makeCopyCache();
  const { ops } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);

  // Device already holds two full pages of the prompt in a retained row.
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, 128);
  gpu.reportTokens(0, tokens(128));
  policy.retainSequence(0);
  const retained = gpu.staging.get(-1)!;

  // Host holds the same first two pages; the device match equals the host
  // match (both prime to 128 tokens), so the host slot is superseded.
  host.ensureSequence(0);
  host.allocAppendPages(0, 128);
  host.reportTokens(0, tokens(128));
  const prompt = [...tokens(200)];

  const suffixes = await policy.prefixMatch([prompt]);
  assert.deepEqual(suffixes, [prompt.slice(128)]);
  assert.deepEqual(entries, [], "equal matches must not copy");
  assert.equal(gpu.sequences[0].pages[0], retained.pages[0]);

  // Both the original and the superseded prime slot sit in host staging; no
  // active host slots remain.
  assert.equal(host.staging.size, 2);
  assert.equal(host.sequences.length, 0);
});

test("prefixMatch restores only the device-side delta past the device match", async () => {
  const { cache: host } = makeCopyCache();
  const { cache: gpu, entries } = makeCopyCache();
  const { ops } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);

  // Device: retained row with the first two pages of the prompt.
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, 128);
  gpu.reportTokens(0, tokens(128));
  policy.retainSequence(0);
  const retained = gpu.staging.get(-1)!;

  // Host: four full pages of the same prompt.
  host.ensureSequence(0);
  host.allocAppendPages(0, 256);
  host.reportTokens(0, tokens(256));
  const prompt = [...tokens(256), 700];

  const suffixes = await policy.prefixMatch([prompt]);
  assert.deepEqual(suffixes, [[700]]);

  const slot = gpu.sequences[0];
  assert.equal(slot.allocLen / PAGE_SIZE, 4);
  assert.deepEqual(slot.getTokenIds(), tokens(256));
  assert.equal(slot.pages[0], retained.pages[0], "pages the device already holds are shared");
  assert.equal(slot.pages[0].refs, 2);
  // Only the two delta pages cross tiers.
  assert.deepEqual(pageCopies(entries), [[2, 2, 2, K_ROW], [2, 2, 2, K_SCALE_ROW], [2, 2, 2, CKV_ROW]]);
  assert.equal(gpu.availablePages.length, 12, "two shared + two copied pages are consumed");
});

test("device eviction offloads a finished row into the host tier", () => {
  const { cache: host, entries: hostEntries } = makeCopyCache();
  const { cache: gpu } = makeCopyCache();
  const { ops } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);

  // A finished row (as retained by removeFinished) holding 12 pages that do
  // not match the incoming request's token space.
  const victimTokens = Array.from({ length: 12 * PAGE_SIZE }, (_, i) => i + 5000);
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, victimTokens.length);
  gpu.reportTokens(0, victimTokens);
  policy.retainSequence(0);
  assert.equal(gpu.availablePages.length, 4);

  // The next allocation needs more than the 4 remaining pages: the retained
  // row is evicted whole, offloaded to an appended, born-active host slot,
  // and its device pages return to the pool.
  gpu.ensureSequence(1);
  gpu.allocAppendPages(1, 6 * PAGE_SIZE);
  assert.equal(gpu.staging.size, 0, "victim left device staging");
  assert.equal(gpu.availablePages.length, 16 - 6);

  const offloaded = host.sequences[0];
  assert.deepEqual(offloaded.getTokenIds(), victimTokens);
  assert.equal(offloaded.pages.length, 12);
  assert.equal(host.sequences.length, 1, "offload destination appends and is born active");
  assert.equal(host.availablePages.length, 16 - 12);
  // The offload copied every victim page (D2H) as one contiguous run per
  // layer cache family.
  assert.deepEqual(pageCopies(hostEntries), [[0, 0, 12, K_ROW], [0, 0, 12, K_SCALE_ROW], [0, 0, 12, CKV_ROW]]);
});

test("device eviction drops a victim when the host tier cannot hold it", () => {
  const { cache: host } = makeCopyCache();
  const { cache: gpu } = makeCopyCache();
  const { ops } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);

  const victimTokens = Array.from({ length: 13 * PAGE_SIZE }, (_, i) => i + 5000);
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, victimTokens.length);
  gpu.reportTokens(0, victimTokens);
  policy.retainSequence(0);
  // Host tier starts with only 12 pages free: the 13-page victim must drop
  // without any host copy; the failed destination must be removed.
  host.availablePages.length = 12;

  gpu.ensureSequence(1);
  gpu.allocAppendPages(1, 16 * PAGE_SIZE);
  assert.equal(gpu.staging.size, 0);
  assert.equal(host.sequences.length, 0, "failed offload destination was removed");
  assert.equal(host.availablePages.length, 12);
});

test("device eviction reclaims host staging to make room for an offload", () => {
  const { cache: host } = makeCopyCache();
  const { cache: gpu } = makeCopyCache();
  const { ops } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);

  // Twelve pages of cold staged host content (a previously offloaded prefix).
  const coldTokens = Array.from({ length: 12 * PAGE_SIZE }, (_, i) => i + 5000);
  host.ensureSequence(0);
  host.allocAppendPages(0, coldTokens.length);
  host.reportTokens(0, coldTokens);
  host.stageSequence(0, 12345);
  assert.equal(host.availablePages.length, 4);

  // A 5-page device victim arrives but the host tier has only 4 free pages:
  // the allocator first reclaims the host tier's own staging — the cold
  // sequence is dropped (largest staged first, no exemptions), then the
  // offload proceeds into a fresh active slot.
  const victimTokens = Array.from({ length: 5 * PAGE_SIZE }, (_, i) => i + 7000);
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, victimTokens.length);
  gpu.reportTokens(0, victimTokens);
  policy.retainSequence(0);

  gpu.ensureSequence(1);
  gpu.allocAppendPages(1, 16 * PAGE_SIZE);
  assert.equal(gpu.staging.size, 0, "victim fully evicted");
  assert.equal(gpu.availablePages.length, 0, "the 16-page allocation consumed the freed pool");
  assert.equal(host.staging.size, 0, "cold staged content was reclaimed");
  assert.deepEqual(host.sequences[0].getTokenIds(), victimTokens);
  assert.equal(host.sequences[0].pages.length, 5);
  assert.equal(host.availablePages.length, 16 - 5);

  // Active host content is never reclaimed: with host staging empty and the
  // device pool exhausted, pressure surfaces as an allocation error rather
  // than dropping the born-active offload destination.
  gpu.ensureSequence(2);
  assert.throws(() => gpu.allocAppendPages(2, PAGE_SIZE), /need 1 pages, 0 available/);
  assert.equal(host.sequences.length, 1);
  assert.equal(host.sequences[0].pages.length, 5);
});

test("offload reuses a matching host prefix even with no free host pages", () => {
  const { cache: host, entries } = makeCopyCache();
  const { cache: gpu } = makeCopyCache();
  const { ops } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);
  host.ensureSequence(0);
  host.allocAppendPages(0, 16 * PAGE_SIZE);
  host.reportTokens(0, tokens(16 * PAGE_SIZE));
  const original = host.sequences[0];
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, 130);
  gpu.reportTokens(0, tokens(130));
  policy.retainSequence(0);

  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, 16 * PAGE_SIZE);
  assert.equal(gpu.staging.size, 0);
  assert.equal(host.sequences.length, 2);
  assert.deepEqual(host.sequences[1].getTokenIds(), tokens(128));
  assert.equal(host.sequences[1].pages[0], original.pages[0]);
  assert.equal(host.sequences[1].pages[1], original.pages[1]);
  assert.equal(original.pages[0].refs, 2);
  assert.equal(host.availablePages.length, 0);
  assert.deepEqual(entries, [], "the existing prefix needs no transfer");
});

test("failed host offload removes its slot and releases shared prefix refs", () => {
  const { cache: host, entries } = makeCopyCache();
  const { cache: gpu } = makeCopyCache();
  const { ops } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);
  host.ensureSequence(0);
  host.allocAppendPages(0, 16 * PAGE_SIZE);
  const history = [...tokens(PAGE_SIZE), ...Array(15 * PAGE_SIZE).fill(9000)];
  host.reportTokens(0, history);
  const original = host.sequences[0];
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, 3 * PAGE_SIZE);
  gpu.reportTokens(0, tokens(3 * PAGE_SIZE));
  policy.retainSequence(0);

  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, 16 * PAGE_SIZE);
  assert.equal(gpu.staging.size, 0, "the GPU victim is dropped despite host exhaustion");
  assert.deepEqual(host.sequences, [original]);
  assert.deepEqual(original.getTokenIds(), history);
  assert(original.pages.every(page => page.refs === 1), "failed destination must release its prefix refs");
  assert.equal(host.availablePages.length, 0);
  assert.deepEqual(entries, [], "reservation failure precedes copy submission");
});

test("host offload propagates errors other than page exhaustion", () => {
  const { cache: host } = makeCopyCache();
  const { cache: gpu } = makeCopyCache();
  const { ops } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);
  const failure = new Error("transfer failed");
  host.ops.memcpyBatchAsync = () => { throw failure; };
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, PAGE_SIZE);
  gpu.reportTokens(0, tokens(PAGE_SIZE));
  policy.retainSequence(0);
  gpu.ensureSequence(0);
  assert.throws(() => gpu.allocAppendPages(0, 16 * PAGE_SIZE), error => error === failure);
  assert.equal(gpu.staging.size, 1, "source remains alive for caller error handling");
});

test("restore uses page pressure instead of capping the prefix to free pages", async () => {
  const { cache: host } = makeCopyCache();
  const { cache: gpu } = makeCopyCache();
  const { ops } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);
  host.ensureSequence(0);
  host.allocAppendPages(0, 4 * PAGE_SIZE);
  host.reportTokens(0, tokens(4 * PAGE_SIZE));
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, 14 * PAGE_SIZE);
  gpu.reportTokens(0, Array(14 * PAGE_SIZE).fill(9000));
  policy.retainSequence(0);
  assert.equal(gpu.availablePages.length, 2);

  const prompt = [...tokens(4 * PAGE_SIZE), 777];
  assert.deepEqual(await policy.prefixMatch([prompt]), [[777]]);
  assert.deepEqual(gpu.sequences[0].getTokenIds(), tokens(4 * PAGE_SIZE));
  assert.equal(gpu.staging.size, 0);
  assert.equal(gpu.availablePages.length, 12);
  // The 14-page victim cannot fit beside the protected host restore source.
  assert.equal(host.sequences.length, 1, "failed offload did not leave an extra host slot");
});

test("restore page exhaustion bubbles up when protected GPU pages prevent admission", async () => {
  const { cache: host } = makeCopyCache();
  const { cache: gpu } = makeCopyCache();
  const { ops } = makePrefixOps();
  const policy = new PrefixTierPolicy(gpu, ops, host);
  host.ensureSequence(0);
  host.allocAppendPages(0, 4 * PAGE_SIZE);
  host.reportTokens(0, tokens(4 * PAGE_SIZE));
  gpu.ensureSequence(0);
  gpu.allocAppendPages(0, 15 * PAGE_SIZE);
  gpu.reportTokens(0, Array(15 * PAGE_SIZE).fill(9000));
  // A scheduler-staged active row is not an eviction candidate.
  gpu.stageSequence(0, 123);
  await assert.rejects(policy.prefixMatch([[...tokens(4 * PAGE_SIZE), 777]]), PageAllocationError);
  assert.equal(gpu.staging.get(123)!.pages.length, 15);
  assert.equal(gpu.availablePages.length, 1);
  assert.equal(gpu.sequences[0].allocLen, 0, "failed reservation did not allocate any pages");
});
