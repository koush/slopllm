import { type ChatCache } from "./chat_model";
import { DeviceOps, TensorParallelism } from "./device_ops";
import { I32 } from "./glm_ops";
import { MemcpyKind, Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export const PAGE_SIZE = 16;

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

// Ordered list of pages with ref-counted sharing. The page structure is:
// [full pages...] [optional partial page] [empty pages...]
// Content pages = ceil(allocLen / pageSize). Empty pages are pre-allocated slots
// beyond the content region. tokenIds must never be mutated after writing.
export class Sequence {
  pages: Page[] = [];
  allocLen = 0;

  constructor(public pagedKvCache: PagedKVCache) {
  }

  get contentPages(): number {
    return this.allocLen > 0 ? Math.ceil(this.allocLen / this.pagedKvCache.pageSize) : 0;
  }

  pushPage(page: Page, pageLen = this.pagedKvCache.pageSize) {
    this.pages.push(page);
    this.allocLen += pageLen;
    page.refs++;
    this.pagedKvCache.pagesDirtyHost = true;
  }

  popPage() {
    const page = this.pages.pop()!;
    this.allocLen = Math.min(this.allocLen, this.pages.length * this.pagedKvCache.pageSize);
    page.refs--;
    if (!page.refs) {
      this.pagedKvCache.availablePages.push(page.id);
    }
    this.pagedKvCache.pagesDirtyHost = true;
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
    this.allocLen = newLen;
    const pageSize = this.pagedKvCache.pageSize;
    while (this.pages.length > 0 && newLen <= (this.pages.length - 1) * pageSize) {
      this.popPage();
    }
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
    let pos = this.allocLen - tokenIds.length;
    let currentPageIndex = Math.floor(pos / this.pagedKvCache.pageSize);
    let offset = pos % this.pagedKvCache.pageSize;
    for (let i = 0; i < tokenIds.length; i++) {
      if (!this.pages[currentPageIndex]) {
        throw new Error(`No page allocated at index ${currentPageIndex}, offset ${offset}`);
      }
      this.pages[currentPageIndex].tokenIds.push(tokenIds[i]);

      offset++;
      if (offset === this.pagedKvCache.pageSize) {
        offset = 0;
        currentPageIndex++;
      }
    }
  }
}

export class PagedKVCache extends WorkspaceBase implements ChatCache {
  readonly nKv: number;
  readonly hd: number;
  readonly nLayers: number;
  readonly maxPages: number;
  readonly maxBatch: number;
  readonly pageSize: number;
  readonly contextParallel: boolean;
  kData: Tensor[];
  vData: Tensor[];
  ckvData: Tensor[];
  kpeData: Tensor[];
  indices: Tensor;
  indicesH: Tensor;
  availablePages: number[];
  sequences: Sequence[];
  staging: Map<number, Sequence>;
  pagesDirtyHost: boolean;
  pagesDirtyDevice: boolean;
  positionIdsDirty: boolean;
  lastNumSequences: number;

  getPagedKV(): PagedKVCache { return this; }

  checkSequenceCount(): void {
    if (this.sequences.length !== this.lastNumSequences) {
      this.pagesDirtyHost = true;
      this.positionIdsDirty = true;
    }
    this.lastNumSequences = this.sequences.length;
  }

  constructor(glm: DeviceOps, nKv: number, hd: number, nLayers: number, maxPages: number, maxBatch: number, pageSize = PAGE_SIZE, kvLoraRank = 0, qkRopeDim = 0, contextParallel = false, indexHeadDim = 0) {
    super(glm);
    this.nKv = nKv;
    this.hd = hd;
    this.nLayers = nLayers;
    this.maxPages = maxPages;
    this.maxBatch = maxBatch;
    this.pageSize = pageSize;
    this.contextParallel = contextParallel;
    this.kData = [];
    this.vData = [];
    this.ckvData = [];
    this.kpeData = [];
    for (let i = 0; i < nLayers; i++) {
      if (kvLoraRank > 0) {
        this.ckvData.push(this.alloc([maxPages, pageSize, kvLoraRank], "BF16", undefined, contextParallel ? TensorParallelism.Row : undefined));
        this.kpeData.push(this.alloc([maxPages, pageSize, qkRopeDim], "BF16", undefined, contextParallel ? TensorParallelism.Row : undefined));
        if (indexHeadDim > 0) {
          this.kData.push(this.alloc([maxPages, pageSize, indexHeadDim], "BF16", undefined, contextParallel ? TensorParallelism.Row : undefined));
        }
      } else {
        this.kData.push(this.alloc([maxPages, nKv * pageSize * hd], "BF16", undefined, TensorParallelism.Row));
        this.vData.push(this.alloc([maxPages, nKv * pageSize * hd], "BF16", undefined, TensorParallelism.Row));
      }
    }
    this.indices = this.alloc([maxPages * maxBatch * I32], "I32", "indices");
    this.indicesH = this.allocPinned([maxPages * maxBatch], "I32", "indicesH");
    this.availablePages = Array.from({ length: maxPages }, (_, i) => i);
    this.sequences = [];
    this.staging = new Map();
    this.pagesDirtyHost = true;
    this.pagesDirtyDevice = true;
    this.positionIdsDirty = true;
    this.lastNumSequences = 0;
  }

  reset(batchSize: number): void {
    if (batchSize > this.maxBatch) {
      throw new Error(`batchSize ${batchSize} exceeds maxBatch ${this.maxBatch}`);
    }
    const stagedPageIds = new Set<number>();
    for (const seq of this.staging.values()) {
      for (const page of seq.pages) {
        stagedPageIds.add(page.id);
      }
    }
    this.availablePages = [];
    for (let i = 0; i < this.maxPages; i++) {
      if (!stagedPageIds.has(i)) {
        this.availablePages.push(i);
      }
    }
    this.sequences = Array.from({ length: batchSize }, () => new Sequence(this));
    this.pagesDirtyHost = true;
    this.positionIdsDirty = true;
  }

  stageSequence(seqIdx: number, stagingKey: number): void {
    if (seqIdx < 0 || seqIdx >= this.sequences.length) {
      throw new Error(`stageSequence: seqIdx ${seqIdx} out of range (${this.sequences.length} sequences)`);
    }
    if (this.staging.has(stagingKey)) {
      throw new Error(`stageSequence: staging key ${stagingKey} already in use`);
    }
    const seq = this.sequences[seqIdx];
    this.staging.set(stagingKey, seq);
    this.sequences.splice(seqIdx, 1);
    this.pagesDirtyHost = true;
    this.positionIdsDirty = true;
  }

  unstageSequence(stagingKey: number): Sequence {
    const seq = this.staging.get(stagingKey);
    if (!seq) {
      throw new Error(`unstageSequence: no staged sequence with key ${stagingKey}`);
    }
    this.staging.delete(stagingKey);
    this.sequences.push(seq);
    this.pagesDirtyHost = true;
    this.positionIdsDirty = true;
    return seq;
  }

  unstageAll(): void {
    const keys = [...this.staging.keys()];
    for (const key of keys) {
      this.unstageSequence(key);
    }
  }

  clearStaging(): void {
    for (const seq of this.staging.values()) {
      seq.clear();
    }
    this.staging.clear();
  }

  removeSequence(seqIdx: number): void {
    if (seqIdx < 0 || seqIdx >= this.sequences.length) {
      throw new Error(`removeSequence: seqIdx ${seqIdx} out of range (${this.sequences.length} sequences)`);
    }
    const seq = this.sequences[seqIdx];
    seq.clear();
    this.sequences.splice(seqIdx, 1);
    this.pagesDirtyHost = true;
    this.positionIdsDirty = true;
  }

  copySequence(dstSeqIdx: number, srcSeqIdx: number) {
    if (dstSeqIdx === srcSeqIdx)
      return;
    this.positionIdsDirty = true;
    const srcSeq = this.sequences[srcSeqIdx];
    const dstSeq = this.ensureSequence(dstSeqIdx);

    const keepPages = Math.floor(srcSeq.allocLen / this.pageSize);
    dstSeq.clear();
    for (let i = 0; i < keepPages; i++) {
      dstSeq.pushPage(srcSeq.pages[i], this.pageSize);
    }
    const partialPage = srcSeq.allocLen % this.pageSize !== 0;
    if (partialPage) {
      this.allocAppendPages(dstSeqIdx, this.pageSize);
      const srcPage = srcSeq.pages[keepPages];
      const dstPage = dstSeq.pages[keepPages];
      this.copyPage(srcPage.id, dstPage.id);
      dstPage.tokenIds.push(...srcPage.tokenIds);
      dstSeq.allocLen = srcSeq.allocLen;
    }
  }

  ensureSequence(seqIdx: number) {
    this.sequences[seqIdx] ??= new Sequence(this);
    return this.sequences[seqIdx];
  }

  // Finds the best prefix match across all sequences and returns the unmatched suffix.
  // Only full pages are kept/shared — the partial page is never shared because
  // the receiving sequence would write into it. Empty pages beyond the content
  // region are discarded. If the entire cache matches (self, no truncation needed),
  // returns the suffix immediately without touching pages.
  prefixMatch(seqIdx: number, inputIds: number[], copyPartial?: boolean): number[] {
    this.ensureSequence(seqIdx);
    this.positionIdsDirty = true;

    let bestSeqIdx = -1;
    let bestMatchTokens = 0;
    for (let i = 0; i < this.sequences.length; i++) {
      if (this.sequences[i].pages.length === 0) continue;
      const matchTokens = this.sequences[i].prefixMatch(inputIds);
      if (matchTokens > bestMatchTokens) {
        bestMatchTokens = matchTokens;
        bestSeqIdx = i;
      }
    }

    if (bestSeqIdx === seqIdx && bestMatchTokens === this.sequences[seqIdx].reportedTokenCount()) {
      return inputIds.slice(bestMatchTokens);
    }

    if (bestSeqIdx < 0 || bestMatchTokens === 0) {
      this.sequences[seqIdx].clear();
      return inputIds.slice();
    }

    // full pages can be shared, memcpy is needed for a partial page.
    const keepPages = Math.floor(bestMatchTokens / this.pageSize);

    if (bestSeqIdx === seqIdx) {
      while (this.sequences[seqIdx].pages.length > keepPages) {
        this.sequences[seqIdx].popPage();
      }
      return inputIds.slice(this.sequences[seqIdx].allocLen);
    }

    if (keepPages > 0) {
      const newSeq = new Sequence(this);
      for (let i = 0; i < keepPages; i++) {
        newSeq.pushPage(this.sequences[bestSeqIdx].pages[i], this.pageSize);
      }
      this.sequences[seqIdx].clear();
      this.sequences[seqIdx] = newSeq;
      const partialPage = bestMatchTokens % this.pageSize !== 0;
      if (copyPartial && partialPage) {
        this.allocAppendPages(seqIdx, this.pageSize);
        const srcPage = this.sequences[bestSeqIdx].pages[keepPages];
        const dstPage = this.sequences[seqIdx].pages[keepPages];
        this.copyPage(srcPage.id, dstPage.id);
        dstPage.tokenIds.push(...srcPage.tokenIds);
        this.sequences[seqIdx].allocLen = bestMatchTokens;
      }
      return inputIds.slice(this.sequences[seqIdx].allocLen);
    }

    this.sequences[seqIdx].clear();
    return inputIds.slice();
  }

  copyPage(srcPageId: number, dstPageId: number): void {
    for (let i = 0; i < this.kData.length; i++) {
      this.copyPageRow(this.kData[i], srcPageId, dstPageId);
      this.copyPageRow(this.vData[i], srcPageId, dstPageId);
    }
    for (let i = 0; i < this.ckvData.length; i++) {
      this.copyPageRow(this.ckvData[i], srcPageId, dstPageId);
      this.copyPageRow(this.kpeData[i], srcPageId, dstPageId);
    }
  }

  copyPageRow(tensor: Tensor, srcPageId: number, dstPageId: number): void {
    const rowBytes = tensor.shape.slice(1).reduce((a, b) => a * b, 1) * 2;
    tensor.memcpy2d(
      dstPageId * rowBytes, rowBytes,
      tensor, srcPageId * rowBytes,
      rowBytes, rowBytes, 1,
      MemcpyKind.DeviceToDevice,
    );
  }

  reportTokens(seqIdx: number, tokens: number[]): void {
    if (seqIdx >= this.sequences.length) throw new Error(`reportTokens: seqIdx ${seqIdx} out of range (${this.sequences.length} sequences)`);
    this.sequences[seqIdx].reportTokens(tokens);
  }

  pagesNeededForDecodeToken(seqIdx: number): number {
    const allocLen = this.sequences[seqIdx].allocLen;
    const pageIdxInSeq = Math.floor(allocLen / this.pageSize);
    return pageIdxInSeq >= this.sequences[seqIdx].pages.length ? 1 : 0;
  }

  pagesNeededForAppend(seqIdx: number, numNewTokens: number): number {
    const currentLen = this.sequences[seqIdx].allocLen;
    const currentPageCount = this.sequences[seqIdx].pages.length;
    const newPageCount = Math.ceil((currentLen + numNewTokens) / this.pageSize);
    return Math.max(0, newPageCount - currentPageCount);
  }

  allocAppendPages(seqIdx: number, numNewTokens: number): void {
    const sequence = this.sequences[seqIdx];
    const currentLen = sequence.allocLen;
    const currentPageCount = sequence.pages.length;
    const newTotalLen = currentLen + numNewTokens;
    const newPageCount = Math.ceil(newTotalLen / this.pageSize);
    const numNewPages = newPageCount - currentPageCount;
    if (numNewPages > this.availablePages.length) {
      throw new Error(`allocAppendPages: need ${numNewPages} pages, ${this.availablePages.length} available`);
    }
    for (let i = 0; i < numNewPages; i++) {
      const pageId = this.availablePages.shift()!;
      const page: Page = { id: pageId, tokenIds: [], refs: 0 };
      sequence.pushPage(page, 0);
    }
    sequence.allocLen = newTotalLen;
  }

  allocDecodeToken(seqIdx: number): void {
    this.allocAppendPages(seqIdx, 1);
  }
}
