import { type ChatCache } from "./chat_model";
import { DeviceOps, TensorParallelism } from "./device_ops";
import type { ExecutionWorkspace } from "./execution-workspace";
import { I32 } from "./glm_ops";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export const PAGE_SIZE = 16;

function longestPrefix(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) return i;
  }
  return len;
}

interface Page {
  id: number;
  tokenIds: number[];
  refs: number;
}

class Sequence {
  pages: Page[] = [];
  kvlen = 0;

  constructor(public pagedKvCache: PagedKVCache) {
  }

  pushPage(page: Page) {
    this.pages.push(page);
    this.kvlen += page.tokenIds.length;
    page.refs++;
  }

  popPage() {
    const page = this.pages.pop()!;
    this.kvlen -= page.tokenIds.length;
    page.refs--;
    if (!page.refs) {
      this.pagedKvCache.availablePages.push(page.id);
    }
  }

  // return the number of full pages that match
  prefixMatch(inputIds: number[]) {
    let bestPageIndex = 0;
    let bestPageLength = -1;

    const checkBest = (pageIndex: number, matchLen: number) => {
      if (matchLen > bestPageLength) {
        bestPageIndex = pageIndex;
        bestPageLength = matchLen;
      }
    }

    const tokenIds = this.pages.map(p => p.tokenIds).flat();
    const prefixLen = longestPrefix(tokenIds, inputIds);
    const pageIndex = Math.floor(prefixLen / this.pagedKvCache.pageSize);
    checkBest(pageIndex, prefixLen);
    return pageIndex;
  }

  slice(numPages: number) {
    const newSequence = new Sequence(this.pagedKvCache);
    for (let i = 0; i < numPages; i++) {
      const page = this.pages[i];
      newSequence.pages.push(page);
      page.refs++;
      newSequence.kvlen += page.tokenIds.length;
    }
    return newSequence;
  }

  appendTokens(tokenIds: number[]) {
    let currentPageIndex = Math.floor(this.kvlen / this.pagedKvCache.pageSize);
    let offset = this.kvlen % this.pagedKvCache.pageSize;
    while (tokenIds.length) {
      const tokenId = tokenIds.shift()!;
      if (!this.pages[currentPageIndex]) {
        throw new Error(`No page allocated for currentPageIndex ${currentPageIndex}, offset ${offset}`);
      }
      this.pages[currentPageIndex].tokenIds.push(tokenId);
      this.kvlen++;

      offset = (offset + 1) % this.pagedKvCache.pageSize;
      if (!offset) {
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
  seqPages: number[][];
  seqKvLens: number[];
  cachedTokenIds: number[][];
  pagesDirtyHost: boolean;
  pagesDirtyDevice: boolean;

  getPagedKV(): PagedKVCache { return this; }

  constructor(glm: DeviceOps, nKv: number, hd: number, nLayers: number, maxPages: number, maxBatch: number, pageSize = PAGE_SIZE, kvLoraRank = 0, qkRopeDim = 0, contextParallel = false) {
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
      } else {
        this.kData.push(this.alloc([maxPages, nKv * pageSize * hd], "BF16", undefined, TensorParallelism.Row));
        this.vData.push(this.alloc([maxPages, nKv * pageSize * hd], "BF16", undefined, TensorParallelism.Row));
      }
    }
    this.indices = this.alloc([maxPages * I32], "I32", "indices");
    this.indicesH = this.allocPinned([maxPages], "I32", "indicesH");
    this.availablePages = Array.from({ length: maxPages }, (_, i) => i);
    this.seqPages = [];
    this.seqKvLens = [];
    this.cachedTokenIds = [];
    this.pagesDirtyHost = true;
    this.pagesDirtyDevice = true;
  }

  reset(batchSize: number): void {
    if (batchSize > this.maxBatch) {
      throw new Error(`batchSize ${batchSize} exceeds maxBatch ${this.maxBatch}`);
    }
    this.availablePages = Array.from({ length: this.maxPages }, (_, i) => i);
    this.seqPages = Array.from({ length: batchSize }, () => []);
    this.seqKvLens = new Array(batchSize).fill(0);
    this.cachedTokenIds = Array.from({ length: batchSize }, () => []);
    this.pagesDirtyHost = true;
    this.pagesDirtyDevice = true;
  }

  prefixMatch(seqIdx: number, inputIds: number[]): number[] {
    if (seqIdx >= this.cachedTokenIds.length) {
      this.cachedTokenIds.length = seqIdx + 1;
      for (let i = 0; i <= seqIdx; i++) {
        if (!this.cachedTokenIds[i]) this.cachedTokenIds[i] = [];
      }
    }
    const cached = this.cachedTokenIds[seqIdx];
    const matchLen = cached.length > 0 ? longestPrefix(cached, inputIds) : 0;

    if (matchLen > 0 && matchLen < inputIds.length) {
      if (matchLen < cached.length) {
        this.truncate(seqIdx, matchLen);
      }
      this.cachedTokenIds[seqIdx] = cached.slice(0, matchLen);
      return inputIds.slice(matchLen);
    }

    if (this.seqPages.length > seqIdx) {
      this.truncate(seqIdx, 0);
    } else {
      this.reset(seqIdx + 1);
    }
    this.cachedTokenIds[seqIdx] = [];
    return inputIds.slice();
  }

  appendTokens(seqIdx: number, tokens: number[]): void {
    if (seqIdx >= this.cachedTokenIds.length) {
      this.cachedTokenIds[seqIdx] = [];
    }
    this.cachedTokenIds[seqIdx].push(...tokens);
  }

  truncate(seqIdx: number, newLen: number): void {
    if (newLen > this.seqKvLens[seqIdx]) {
      throw new Error(`truncate: newLen ${newLen} > current seqKvLens ${this.seqKvLens[seqIdx]}`);
    }
    if (newLen === 0) {
      const freedPages = this.seqPages[seqIdx];
      for (let i = freedPages.length - 1; i >= 0; i--) {
        this.availablePages.unshift(freedPages[i]);
      }
      this.seqPages[seqIdx] = [];
      this.seqKvLens[seqIdx] = 0;
      return;
    }
    const pageSize = this.pageSize;
    const newPageCount = Math.ceil(newLen / pageSize);
    const oldPageCount = this.seqPages[seqIdx].length;
    const freedPages = this.seqPages[seqIdx].slice(newPageCount);
    for (let i = freedPages.length - 1; i >= 0; i--) {
      this.availablePages.unshift(freedPages[i]);
    }
    this.seqPages[seqIdx] = this.seqPages[seqIdx].slice(0, newPageCount);
    this.seqKvLens[seqIdx] = newLen;
  }

  pagesNeededForDecodeToken(seqIdx: number): number {
    const kvLen = this.seqKvLens[seqIdx];
    const pageIdxInSeq = Math.floor(kvLen / this.pageSize);
    return pageIdxInSeq >= this.seqPages[seqIdx].length ? 1 : 0;
  }

  pagesNeededForAppend(seqIdx: number, numNewTokens: number): number {
    const currentLen = this.seqKvLens[seqIdx];
    const currentPageCount = this.seqPages[seqIdx].length;
    const newPageCount = Math.ceil((currentLen + numNewTokens) / this.pageSize);
    return Math.max(0, newPageCount - currentPageCount);
  }

  allocAppendPages(seqIdx: number, numNewTokens: number): [number, number] {
    const pageSize = this.pageSize;
    const currentLen = this.seqKvLens[seqIdx];
    const currentPageCount = this.seqPages[seqIdx].length;
    const newTotalLen = currentLen + numNewTokens;
    const newPageCount = Math.ceil(newTotalLen / pageSize);
    const numNewPages = newPageCount - currentPageCount;
    if (numNewPages > this.availablePages.length) {
      throw new Error(`allocAppendPages: need ${numNewPages} pages, ${this.availablePages.length} available`);
    }
    let startPage = -1;
    for (let i = 0; i < numNewPages; i++) {
      const page = this.availablePages.shift()!;
      if (i === 0) startPage = page;
      this.seqPages[seqIdx].push(page);
    }
    this.seqKvLens[seqIdx] = newTotalLen;
    if (numNewPages > 0) {
      this.pagesDirtyHost = true;
      this.pagesDirtyDevice = true;
    }
    return [startPage, numNewPages];
  }

  allocDecodeToken(seqIdx: number): [number, number] {
    const kvLen = this.seqKvLens[seqIdx];
    const pageSize = this.pageSize;
    const pageIdxInSeq = Math.floor(kvLen / pageSize);
    if (pageIdxInSeq >= this.seqPages[seqIdx].length) {
      if (this.availablePages.length === 0) {
        throw new Error(`allocDecodeToken: no pages available`);
      }
      const newPage = this.availablePages.shift()!;
      this.seqPages[seqIdx].push(newPage);
      this.pagesDirtyHost = true;
      this.pagesDirtyDevice = true;
    }
    this.seqKvLens[seqIdx] = kvLen + 1;
    const absPage = this.seqPages[seqIdx][pageIdxInSeq];
    const slotInPage = kvLen % pageSize;
    return [absPage, slotInPage];
  }

  updateIndptr(ws: ExecutionWorkspace): void {
    const batchSize = this.seqPages.length;
    ws.indptrH.withPinnedBuffer(buf => {
      buf.writeInt32LE(0, 0);
      let cumulative = 0;
      for (let i = 0; i < batchSize; i++) {
        cumulative += this.seqPages[i].length;
        buf.writeInt32LE(cumulative, (i + 1) * I32);
      }
    });

    this.indicesH.withPinnedBuffer(buf => {
      let indicesOff = 0;
      for (let i = 0; i < batchSize; i++) {
        for (const page of this.seqPages[i]) {
          buf.writeInt32LE(page, indicesOff * I32);
          indicesOff++;
        }
      }
    });

    ws.lastPageLenH.withPinnedBuffer(buf => {
      for (let i = 0; i < batchSize; i++) {
        const kvLen = this.seqKvLens[i];
        const remainder = kvLen % this.pageSize;
        buf.writeInt32LE(remainder !== 0 ? remainder : (kvLen > 0 ? this.pageSize : 0), i * I32);
      }
    });
  }
}
