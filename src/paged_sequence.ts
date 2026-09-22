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
  /** Selected next input, not part of committed token history or KV. */
  targetToken?: number;

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
    const newLen = Math.min(this.allocLen, (this.pages.length - 1) * this.pagedKvCache.pageSize);
    const targetToken = this.tokenAfter(newLen);
    const page = this.pages.pop()!;
    this.allocLen = newLen;
    this.targetToken = targetToken;
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
    this.targetToken = undefined;
  }

  private tokenAfter(length: number): number | undefined {
    const pageSize = this.pagedKvCache.pageSize;
    const token = this.pages[Math.floor(length / pageSize)]?.tokenIds[length % pageSize];
    if (token !== undefined) {
      return token;
    }
    return length === this.reportedTokenCount() ? this.targetToken : undefined;
  }

  truncate(newLen: number) {
    if (newLen < 0 || newLen > this.allocLen) {
      throw new Error(`truncate: newLen ${newLen} out of range [0, ${this.allocLen}]`);
    }
    if (newLen === this.allocLen) return;

    const targetToken = this.tokenAfter(newLen);
    const pageSize = this.pagedKvCache.pageSize;
    this.allocLen = newLen;
    while (this.pages.length > 0 && newLen <= (this.pages.length - 1) * pageSize) {
      this.popPage();
    }
    this.targetToken = targetToken;

    if (newLen % pageSize === 0) return;

    // Appending after a non-page-aligned truncation overwrites the retained
    // page. Detach it first so another sequence sharing the full page keeps its
    // original KV contents.
    const pageIdx = Math.floor(newLen / pageSize);
    let page = this.pages[pageIdx];
    if (page.refs > 1) {
      this.pagedKvCache.ensureAvailablePages(1);
      const newPageId = this.pagedKvCache.availablePages.shift();
      if (newPageId === undefined) {
        throw new Error("truncate: no available page for copy-on-write");
      }
      const newPage: Page = { id: newPageId, tokenIds: [...page.tokenIds], refs: 1 };
      this.pagedKvCache.copyPage(page.id, newPage.id);
      page.refs--;
      this.pages[pageIdx] = page = newPage;
    }
    page.tokenIds.length = Math.min(page.tokenIds.length, newLen % pageSize);
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

  reportTokens(tokenIds: number[], targetToken?: number) {
    const pageSize = this.pagedKvCache.pageSize;
    // Append committed inputs after the reported history, within reserved space.
    const flattenLen = this.reportedTokenCount();
    if (flattenLen + tokenIds.length > this.allocLen) {
      throw new Error(
        `reportTokens: ${tokenIds.length} tokens at flatten offset ${flattenLen} exceed allocated region (allocLen ${this.allocLen})`);
    }
    let currentPageIndex = Math.floor(flattenLen / pageSize);
    let offset = flattenLen % pageSize;
    for (let i = 0; i < tokenIds.length; i++) {
      this.pages[currentPageIndex].tokenIds.push(tokenIds[i]);

      offset++;
      if (offset === pageSize) {
        offset = 0;
        currentPageIndex++;
      }
    }
    this.targetToken = targetToken;
  }
}
