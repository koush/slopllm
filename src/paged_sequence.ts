export const PAGE_SIZE = 64;

function longestPrefix(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) return i;
  }
  return len;
}

// Pages are write-only until filled and ref-counted for cross-sequence sharing.
// Only full pages (pageSize tokens) can be shared — a partial page must never
// be shared because the receiving sequence would need to write suffix tokens into it.
// Empty pages (pre-allocated but unfilled) are never shared.
export interface Page {
  id: number;
  tokenIds: number[];
  refs: number;
}

// The subset of PagedKVCache that Sequence needs. PagedKVCache satisfies this
// structurally; keeping it minimal lets the sequence bookkeeping be unit-tested
// without a GPU.
export interface SequenceCache {
  readonly pageSize: number;
  availablePages: number[];
  ensureAvailablePages(requiredPages: number): void;
  copyPage(srcPageId: number, dstPageId: number): void;
}

// Ordered list of pages with ref-counted sharing. The page structure is:
// [full pages...] [optional partial page] [empty pages...]
// Content pages = ceil(allocLen / pageSize). Empty pages are pre-allocated slots
// beyond the content region. tokenIds must never be mutated after writing.
export class Sequence {
  pages: Page[] = [];
  allocLen = 0;

  constructor(public pagedKvCache: SequenceCache) {
  }

  get contentPages(): number {
    return this.allocLen > 0 ? Math.ceil(this.allocLen / this.pagedKvCache.pageSize) : 0;
  }

  pushPage(page: Page, pageLen = this.pagedKvCache.pageSize) {
    this.pages.push(page);
    this.allocLen += pageLen;
    page.refs++;
  }

  popPage() {
    const page = this.pages.pop()!;
    this.allocLen = Math.min(this.allocLen, this.pages.length * this.pagedKvCache.pageSize);
    page.refs--;
    if (!page.refs) {
      // unshift to return pages in order of allocation (FIFO) for better cache locality.
      this.pagedKvCache.availablePages.unshift(page.id);
    }
  }

  clear() {
    while (this.pages.length > 0) {
      this.popPage();
    }
  }

  truncate(newLen: number) {
    if (newLen < 0 || newLen > this.allocLen) {
      throw new Error(`truncate: newLen ${newLen} out of range [0, ${this.allocLen}]`);
    }
    if (newLen === this.allocLen) return;

    const pageSize = this.pagedKvCache.pageSize;
    this.allocLen = newLen;
    while (this.pages.length > 0 && newLen <= (this.pages.length - 1) * pageSize) {
      this.popPage();
    }

    if (newLen % pageSize === 0) return;

    // Appending after a non-page-aligned truncation overwrites the retained
    // page. Detach it first so another sequence sharing the full page keeps its
    // original KV contents.
    const pageIdx = Math.floor(newLen / pageSize);
    const page = this.pages[pageIdx];
    if (page.refs === 1) return;

    this.pagedKvCache.ensureAvailablePages(1);
    const newPageId = this.pagedKvCache.availablePages.shift();
    if (newPageId === undefined) {
      throw new Error("truncate: no available page for copy-on-write");
    }
    const newPage: Page = { id: newPageId, tokenIds: [...page.tokenIds], refs: 1 };
    this.pagedKvCache.copyPage(page.id, newPage.id);
    page.refs--;
    this.pages[pageIdx] = newPage;
  }

  // Returns the number of matching tokens at the start of this sequence and inputIds.
  prefixMatch(inputIds: number[]): number {
    const tokenIds = this.getTokenIds();
    return longestPrefix(tokenIds, inputIds);
  }

  getTokenIds(): number[] {
    return this.pages.map(p => p.tokenIds).flat();
  }

  // Number of tokens that have been reported via reportTokens (sum of page.tokenIds).
  // May be less than allocLen when pages have been allocated but tokens not yet
  // appended — e.g. after allocAppendPages/allocDecodeToken reserves space for a
  // prefill/decode that hasn't written its tokenIds yet. prefixMatch uses this
  // (not allocLen) because it can only compare against materialized tokens.
  reportedTokenCount(): number {
    let count = 0;
    for (const page of this.pages) count += page.tokenIds.length;
    return count;
  }

  reportTokens(tokenIds: number[]) {
    const pageSize = this.pagedKvCache.pageSize;
    // Pushes append at the flatten tail (sum of page arrays), which can be one
    // ahead of `allocLen - tokenIds.length` while a reported-but-unprocessed
    // peek token is present. Derive the starting page/offset from the array
    // state — a pos-derived offset rolls page boundaries one token late and
    // overfills pages, corrupting prefix sharing.
    let flattenLen = 0;
    for (const page of this.pages) flattenLen += page.tokenIds.length;
    if (flattenLen + tokenIds.length > this.allocLen + 1) {
      throw new Error(
        `reportTokens: ${tokenIds.length} tokens at flatten offset ${flattenLen} exceed allocated region (allocLen ${this.allocLen})`);
    }
    let currentPageIndex = Math.floor(flattenLen / pageSize);
    let offset = flattenLen % pageSize;
    for (let i = 0; i < tokenIds.length; i++) {
      if (!this.pages[currentPageIndex]) {
        // The final reported token can be the peek one slot past allocLen; if
        // that slot opens a new page, attach it now — the next
        // allocDecodeToken would have attached it anyway.
        this.pagedKvCache.ensureAvailablePages(1);
        const pageId = this.pagedKvCache.availablePages.shift();
        if (pageId === undefined) {
          throw new Error(`reportTokens: no available page at index ${currentPageIndex}`);
        }
        this.pushPage({ id: pageId, tokenIds: [], refs: 0 }, 0);
      }
      this.pages[currentPageIndex].tokenIds.push(tokenIds[i]);

      offset++;
      if (offset === pageSize) {
        offset = 0;
        currentPageIndex++;
      }
    }
  }
}