import assert from "node:assert/strict";
import { it } from "node:test";
import { MetaOps } from "../src/meta_ops";
import { PAGE_SIZE, PagedKVCache } from "../src/paged_kv";

for (const length of [0, PAGE_SIZE - 1, PAGE_SIZE, PAGE_SIZE + 1]) {
  it(`reportTokens rejects unallocated input without changing history at length ${length}`, () => {
    using cache = new PagedKVCache(new MetaOps(), 1, 8, 1, 4, 1);
    cache.reset(1);
    const tokens = Array.from({ length }, (_, i) => i + 1);
    cache.allocAppendPages(0, length);
    cache.reportTokens(0, tokens, 777);
    const sequence = cache.sequences[0];
    const available = [...cache.availablePages];
    const pages = [...sequence.pages];
    assert.throws(() => cache.reportTokens(0, [777], 888), /exceed allocated region/);
    assert.deepEqual(sequence.getTokenIds(), tokens);
    assert.deepEqual(sequence.pages, pages);
    assert.deepEqual(cache.availablePages, available);
    assert.equal(sequence.allocLen, length);
    assert.equal(sequence.targetToken, 777);
  });
}

it("truncate discards reserved positions while preserving the pending target", () => {
  using cache = new PagedKVCache(new MetaOps(), 1, 8, 1, 4, 1);
  cache.reset(1);
  const tokens = Array.from({ length: PAGE_SIZE }, (_, i) => i + 1);
  cache.allocAppendPages(0, tokens.length);
  cache.reportTokens(0, tokens, 777);
  cache.allocAppendPages(0, 2);
  const sequence = cache.sequences[0];
  sequence.truncate(tokens.length);
  assert.equal(sequence.targetToken, 777);
  assert.deepEqual(sequence.getTokenIds(), tokens);
  assert.equal(sequence.allocLen, tokens.length);
});

for (const length of [PAGE_SIZE, PAGE_SIZE + 1, PAGE_SIZE + 3]) {
  it(`truncate trims reported history at length ${length} and permits re-reporting`, () => {
    using cache = new PagedKVCache(new MetaOps(), 1, 8, 1, 4, 1);
    cache.reset(1);
    const tokens = Array.from({ length }, (_, i) => i + 1);
    cache.allocAppendPages(0, length);
    cache.reportTokens(0, tokens, 777);
    const sequence = cache.sequences[0];
    assert.equal(sequence.targetToken, 777);
    sequence.truncate(length);
    assert.equal(sequence.targetToken, 777, "a no-op truncation preserves the pending token");
    sequence.truncate(length - 1);
    assert.equal(sequence.targetToken, tokens.at(-1));
    assert.equal(sequence.allocLen, length - 1);
    assert.deepEqual(sequence.getTokenIds(), tokens.slice(0, -1));
    cache.allocAppendPages(0, 2);
    cache.reportTokens(0, [tokens.at(-1)!, 999], 1000);
    assert.equal(sequence.targetToken, 1000);
    assert.deepEqual(sequence.getTokenIds(), [...tokens, 999]);
    assert.equal(sequence.reportedTokenCount(), sequence.allocLen);
  });
}

it("truncate trims the copied page's history without changing the shared source", () => {
  using cache = new PagedKVCache(new MetaOps(), 1, 8, 1, 4, 2);
  cache.reset(2);
  const tokens = Array.from({ length: PAGE_SIZE }, (_, i) => i + 1);
  cache.allocAppendPages(0, tokens.length);
  cache.reportTokens(0, tokens, 777);
  cache.copySequence(1, 0);
  assert.equal(cache.sequences[1].targetToken, 777);
  const original = cache.sequences[0].pages[0];
  assert.equal(original.refs, 2);
  cache.sequences[1].truncate(PAGE_SIZE - 1);
  assert.notEqual(cache.sequences[1].pages[0], original);
  assert.equal(original.refs, 1);
  assert.deepEqual(cache.sequences[0].getTokenIds(), tokens);
  assert.deepEqual(cache.sequences[1].getTokenIds(), tokens.slice(0, -1));
  assert.equal(cache.sequences[0].targetToken, 777);
  assert.equal(cache.sequences[1].targetToken, tokens.at(-1));
  cache.sequences[1].truncate(0);
  assert.equal(cache.sequences[1].targetToken, tokens[0]);
  cache.sequences[1].clear();
  assert.equal(cache.sequences[1].targetToken, undefined);
});

it("prefix sharing copies the target at the retained boundary", () => {
  using cache = new PagedKVCache(new MetaOps(), 1, 8, 1, 4, 2);
  cache.reset(2);
  const tokens = Array.from({ length: PAGE_SIZE + 3 }, (_, i) => i + 1);
  cache.allocAppendPages(0, tokens.length);
  cache.reportTokens(0, tokens, 999);
  assert.deepEqual(cache.prefixMatch(1, tokens), tokens.slice(PAGE_SIZE));
  assert.equal(cache.sequences[1].targetToken, tokens[PAGE_SIZE]);
  cache.sequences[1].clear();
  assert.deepEqual(cache.prefixMatch(1, tokens, true), tokens.slice(-1));
  assert.equal(cache.sequences[1].targetToken, tokens.at(-1));
  assert.equal(cache.sequences[1].allocLen, tokens.length - 1);
});

for (const donorKind of ["self", "active", "staged"]) {
  for (const extra of [0, 3]) {
    it(`full prefix match replays the last prompt token (${donorKind}, donor extra=${extra})`, () => {
      using cache = new PagedKVCache(new MetaOps(), 1, 8, 1, 4, 2);
      cache.reset(2);
      const prompt = Array.from({ length: PAGE_SIZE }, (_, i) => i + 1);
      const donorTokens = [...prompt, ...Array(extra).fill(888)];
      cache.allocAppendPages(0, donorTokens.length);
      cache.reportTokens(0, donorTokens, 999);
      const donor = cache.sequences[0];
      let index = donorKind === "self" ? 0 : 1;
      if (donorKind === "staged") {
        cache.stageSequence(0, 123);
        index = 0;
      }
      assert.deepEqual(cache.prefixMatch(index, prompt), prompt.slice(-1));
      const matched = cache.sequences[index];
      assert.equal(matched.allocLen, prompt.length - 1);
      assert.deepEqual(matched.getTokenIds(), prompt.slice(0, -1));
      assert.equal(matched.targetToken, prompt.at(-1));
      if (donorKind !== "self") {
        assert.deepEqual(donor.getTokenIds(), donorTokens);
        assert.equal(donor.targetToken, 999);
        assert.notEqual(matched.pages[0], donor.pages[0]);
      }
    });
  }
}
