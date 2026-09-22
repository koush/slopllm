import test from "node:test";
import assert from "node:assert/strict";
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
