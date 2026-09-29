import assert from "node:assert/strict";
import test from "node:test";
import { PrefixTierPolicy } from "../src/generation-scheduler";
import { MetaOps } from "../src/meta_ops";
import { PagedKVCache } from "../src/paged_kv";

const tokens = (length: number) => Array.from({ length }, (_, i) => i + 100);

function makeCache(pageSize = 64, maxPages = 16) {
  const cache = new PagedKVCache(new MetaOps(), 1, 8, 1, maxPages, 4, pageSize);
  cache.copyPage = () => assert.fail("resume/match must not copy pages");
  return cache;
}

function seed(cache: PagedKVCache, index: number, history: number[]) {
  const sequence = cache.ensureSequence(index);
  cache.allocAppendPages(index, history.length);
  cache.reportTokens(index, history, 777);
  return sequence;
}

test("authorized resume can retain a prefix shorter than one page", () => {
  using cache = makeCache(512);
  const source = seed(cache, 0, tokens(3));
  const page = source.pages[0];
  cache.stageSequence(0, 123);
  assert.deepEqual(cache.prefixMatch(0, tokens(4), sequence => sequence === source), tokens(4).slice(3));
  assert.equal(cache.sequences[0], source);
  assert.equal(source.pages[0], page);
  assert.equal(page.refs, 1);
  assert.equal(cache.staging.size, 0);
});

for (const pageSize of [64, 512]) {
  for (const kind of ["self", "active", "staged"]) {
    test(`authorized ${kind} resume transfers a private partial page without allocation (pageSize=${pageSize})`, () => {
      using cache = makeCache(pageSize, 2);
      const history = tokens(2 * pageSize - 1);
      const source = seed(cache, 0, history);
      const pages = [...source.pages];
      cache.ensureSequence(1);
      const observer = cache.ensureSequence(2);
      observer.pushPage(pages[0]);
      let index = kind === "self" ? 0 : 1;
      let observerIndex = 2;
      if (kind === "staged") {
        cache.stageSequence(0, 123);
        index = 0;
        observerIndex = 1;
      }
      assert.equal(cache.availablePages.length, 0);
      cache.onPagePressure = () => assert.fail("resume must not request capacity");
      const input = [...history, 9000];
      assert.deepEqual(cache.prefixMatch(index, input, candidate => candidate === source), [9000]);
      assert.equal(cache.sequences[index], source);
      assert.deepEqual(source.pages, pages);
      assert.equal(pages[0].refs, 2, "full pages may remain shared");
      assert.equal(pages[1].refs, 1, "partial ownership must not be duplicated");
      assert.equal(source.targetToken, 9000, "use the prompt suffix, not the old sampled target");
      assert.equal(cache.sequences[observerIndex], observer, "other slot indices stay stable");
      if (kind === "active") {
        assert.equal(cache.sequences[0].pages.length, 0);
        assert.notEqual(cache.sequences[0], source);
      }
      if (kind === "staged") assert.equal(cache.staging.has(123), false);
      cache.allocAppendPages(index, 1);
      cache.reportTokens(index, [9000]);
      assert.deepEqual(source.getTokenIds(), input);
      assert.deepEqual(observer.getTokenIds(), history.slice(0, pageSize));
      assert.equal(cache.availablePages.length, 0);
    });
  }
}

for (const scenario of ["denied", "exact", "shorter", "mismatch", "uncommitted", "shared", "aligned"]) {
  test(`resume falls back to full pages for ${scenario} candidates`, () => {
    using cache = makeCache();
    const length = scenario === "aligned" ? 128 : 67;
    const history = tokens(length);
    const source = seed(cache, 0, history);
    cache.ensureSequence(1);
    if (scenario === "uncommitted") cache.allocAppendPages(0, 1);
    if (scenario === "shared") cache.ensureSequence(2).pushPage(source.pages[1], 3);
    let input = [...history, 9000];
    if (scenario === "exact") input = history;
    if (scenario === "shorter") input = history.slice(0, 66);
    if (scenario === "mismatch") input = [...history.slice(0, 66), 8000, 9000];
    let approvals = 0;
    const suffix = cache.prefixMatch(1, input, () => { approvals++; return scenario !== "denied"; });
    const kept = scenario === "aligned" ? 128 : 64;
    assert.deepEqual(suffix, input.slice(kept));
    assert.notEqual(cache.sequences[1], source);
    assert.equal(cache.sequences[0], source);
    assert.deepEqual(source.getTokenIds(), history);
    assert.equal(cache.sequences[1].allocLen, kept);
    assert.equal(approvals, scenario === "denied" ? 1 : 0, "only eligible partial candidates reach the callback");
  });
}

test("a longer full-page match wins without consuming a shorter approved donor", () => {
  using cache = makeCache();
  const donor = seed(cache, 0, tokens(67));
  const longer = seed(cache, 1, tokens(192));
  cache.stageSequence(0, 123);
  assert.deepEqual(cache.prefixMatch(1, tokens(193), () => true), tokens(193).slice(192));
  assert.equal(cache.staging.get(123), donor);
  assert.equal(donor.pages[1].refs, 1);
  assert.equal(cache.sequences[1].pages[2], longer.pages[2]);
});

test("resuming replaces an existing destination and releases its old pages", () => {
  using cache = makeCache();
  const source = seed(cache, 0, tokens(67));
  const old = seed(cache, 1, [9999]);
  const oldPage = old.pages[0];
  const available = cache.availablePages.length;
  cache.prefixMatch(1, tokens(68), sequence => sequence === source);
  assert.equal(cache.sequences[1], source);
  assert.equal(old.pages.length, 0);
  assert.equal(oldPage.refs, 0);
  assert.equal(cache.availablePages.length, available + 1);
});

test("a throwing permission callback leaves candidates and destination unchanged", () => {
  using cache = makeCache();
  const source = seed(cache, 0, tokens(67));
  const target = seed(cache, 1, [9999]);
  const available = [...cache.availablePages];
  const error = new Error("permission failure");
  assert.throws(() => cache.prefixMatch(1, tokens(68), () => { throw error; }), e => e === error);
  assert.equal(cache.sequences[0], source);
  assert.equal(cache.sequences[1], target);
  assert.deepEqual(source.getTokenIds(), tokens(67));
  assert.deepEqual(target.getTokenIds(), [9999]);
  assert.deepEqual(cache.availablePages, available);
});

for (const hostEnabled of [false, true]) {
  test(`policy consumes a retained donor only once per admission (host=${hostEnabled})`, async () => {
    using gpu = makeCache(512);
    using host = makeCache(512);
    const policy = new PrefixTierPolicy(gpu, gpu.ops, hostEnabled ? host : undefined);
    const donor = seed(gpu, 0, tokens(1023));
    policy.retainSequence(0);
    gpu.reset(2);
    const suffixes = await policy.prefixMatch([tokens(1024), tokens(1025)]);
    assert.deepEqual(suffixes, [tokens(1024).slice(1023), tokens(1025).slice(512)]);
    assert.equal(gpu.sequences[0], donor);
    assert.notEqual(gpu.sequences[1], donor);
    assert.equal(donor.pages[1].refs, 1);
    assert.equal(gpu.staging.size, 0);
    // Return both rows to retention; flushing must only process their new keys.
    policy.retainSequence(1);
    policy.retainSequence(0);
    await policy.flushGpuCache();
    assert.equal(gpu.availablePages.length, gpu.maxPages);
  });
}

test("policy never consumes a temporarily staged active request", async () => {
  using gpu = makeCache();
  const policy = new PrefixTierPolicy(gpu, gpu.ops);
  const protectedSeq = seed(gpu, 0, tokens(67));
  gpu.stageSequence(0, 123);
  gpu.reset(1);
  assert.deepEqual(await policy.prefixMatch([tokens(68)]), [tokens(68).slice(64)]);
  assert.equal(gpu.staging.get(123), protectedSeq);
  assert.deepEqual(protectedSeq.getTokenIds(), tokens(67));
  assert.equal(protectedSeq.pages[1].refs, 1);
  assert.notEqual(gpu.sequences[0], protectedSeq);
});
