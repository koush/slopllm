import test from "node:test";
import assert from "node:assert/strict";
import { Sequence, type Page, type SequenceCache } from "../src/paged_sequence";

// Sequence only needs pageSize (and page-recycling hooks) from the cache; stub
// it so this test runs without a GPU.
function makeCache(pageSize: number): SequenceCache {
  let nextPageId = 100;
  return {
    pageSize,
    availablePages: [] as number[],
    ensureAvailablePages(n: number) {
      for (let i = 0; i < n; i++) this.availablePages.push(nextPageId++);
    },
    copyPage() {},
  };
}

function makePage(id: number, tokenIds: number[]): Page {
  return { id, tokenIds, refs: 0 };
}

test("decode commit across a page edge does not overfill the tail page", () => {
  const cache = makeCache(4);
  const seq = new Sequence(cache);
  // Content [10,11,12] at positions 0..2, peek token 99 reported at position 3
  // (reported = allocLen + 1). Page 0's array is therefore full.
  seq.pushPage(makePage(0, [10, 11, 12, 99]), 3);
  // Verification commits 2 tokens: allocLen 3 -> 5, page 1 allocated.
  seq.pushPage(makePage(1, []), 2);
  // The 2 reported tokens (accepted + replacement) belong at positions 4..5,
  // i.e. page 1 — not appended to the already-full page 0.
  seq.reportTokens([7, 8]);
  assert.equal(seq.pages[0].tokenIds.length, 4);
  assert.deepEqual(seq.pages[1].tokenIds, [7, 8]);
  assert.deepEqual(seq.getTokenIds(), [10, 11, 12, 99, 7, 8]);
  assert.equal(seq.reportedTokenCount(), 6);
});

test("admission peek push at a page edge lands in the next page", () => {
  const cache = makeCache(4);
  const seq = new Sequence(cache);
  // Suffix exactly filled page 0 (allocLen 4, array full, no peek).
  seq.pushPage(makePage(0, [1, 2, 3, 4]), 4);
  seq.pushPage(makePage(1, []), 1);
  // First sampled token reported before decode: position 4 = page 1 offset 0.
  seq.reportTokens([42]);
  assert.equal(seq.pages[0].tokenIds.length, 4);
  assert.deepEqual(seq.pages[1].tokenIds, [42]);
});

test("normal mid-page and page-rolling appends are unchanged", () => {
  const cache = makeCache(4);
  const seq = new Sequence(cache);
  seq.pushPage(makePage(0, [1, 2, 3, 4]), 4);
  seq.pushPage(makePage(1, [5]), 1);
  seq.pushPage(makePage(2, []), 4);
  // allocLen 9, arrays: page0=4, page1=1, page2=0. Push 4 tokens starting at
  // pos 5 (page 1 offset 1): three fill page 1, the fourth rolls into page 2.
  seq.reportTokens([6, 7, 8, 9]);
  assert.deepEqual(seq.pages[1].tokenIds, [5, 6, 7, 8]);
  assert.deepEqual(seq.pages[2].tokenIds, [9]);
});

test("no page ever exceeds pageSize across many simulated decode steps", () => {
  const cache = makeCache(4);
  const seq = new Sequence(cache);
  const truth: number[] = [100, 101, 102, 103];
  // Page 0 holds 3 committed tokens + the reported-but-unprocessed peek (full).
  seq.pushPage(makePage(0, [100, 101, 102, 103]), 3);
  let contentLen = 3; // committed tokens; the peek sits at position 3
  let nextToken = 200;
  for (let step = 0; step < 50; step++) {
    const f = 2 + (step % 3); // committed tokens this step (accepted + replacement)
    // Attach empty pages to cover the committed region without advancing allocLen.
    while (seq.pages.length * 4 < contentLen + f) {
      seq.pushPage(makePage(seq.pages.length, []), 0);
    }
    // allocLen = committed region; the f reported tokens land at positions
    // (contentLen, contentLen + f] — the last one becomes the new peek.
    seq.allocLen = contentLen + f;
    const reported: number[] = [];
    for (let j = 0; j < f; j++) {
      reported.push(nextToken);
      truth.push(nextToken++);
    }
    seq.reportTokens(reported);
    contentLen += f;
    for (const page of seq.pages) {
      assert.ok(page.tokenIds.length <= 4, `page overfull: ${page.tokenIds.length}`);
    }
  }
  // Flatten stays content-correct: prefix matching against the true stream works.
  assert.deepEqual(seq.getTokenIds(), truth);
  assert.equal(seq.prefixMatch(truth), truth.length);
  assert.equal(seq.reportedTokenCount(), truth.length);
});