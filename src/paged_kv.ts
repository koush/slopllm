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
  allocLen = 0;

  constructor(public pagedKvCache: PagedKVCache) {
  }

  pushPage(page: Page) {
    this.pages.push(page);
    this.allocLen += page.tokenIds.length;
    page.refs++;
  }

  popPage() {
    const page = this.pages.pop()!;
    this.allocLen -= page.tokenIds.length;
    page.refs--;
    if (!page.refs) {
      this.pagedKvCache.availablePages.push(page.id);
    }
  }

  clear() {
    while (this.pages.length > 0) {
      this.popPage();
    }
  }

  prefixMatch(inputIds: number[]): number {
    const tokenIds = this.pages.map(p => p.tokenIds).flat();
    const prefixLen = longestPrefix(tokenIds, inputIds);
    return Math.floor(prefixLen / this.pagedKvCache.pageSize);
  }

  slice(numPages: number): Sequence {
    const newSequence = new Sequence(this.pagedKvCache);
    for (let i = 0; i < numPages; i++) {
      newSequence.pushPage(this.pages[i]);
    }
    return newSequence;
  }

  appendTokens(tokenIds: number[]) {
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
    this.sequences = [];
    this.pagesDirtyHost = true;
    this.pagesDirtyDevice = true;
  }

  reset(batchSize: number): void {
    if (batchSize > this.maxBatch) {
      throw new Error(`batchSize ${batchSize} exceeds maxBatch ${this.maxBatch}`);
    }
    this.availablePages = Array.from({ length: this.maxPages }, (_, i) => i);
    this.sequences = Array.from({ length: batchSize }, () => new Sequence(this));
    this.pagesDirtyHost = true;
    this.pagesDirtyDevice = true;
  }

  prefixMatch(seqIdx: number, inputIds: number[]): number[] {
    while (this.sequences.length <= seqIdx) {
      this.sequences.push(new Sequence(this));
    }

    let bestSeqIdx = -1;
    let bestMatchPages = 0;
    for (let i = 0; i < this.sequences.length; i++) {
      if (this.sequences[i].pages.length === 0) continue;
      const matchPages = this.sequences[i].prefixMatch(inputIds);
      if (matchPages > bestMatchPages) {
        bestMatchPages = matchPages;
        bestSeqIdx = i;
      }
    }

    if (bestSeqIdx >= 0 && bestMatchPages > 0) {
      const bestSeq = this.sequences[bestSeqIdx];
      let matchTokens = 0;
      for (let i = 0; i < bestMatchPages; i++) {
        matchTokens += bestSeq.pages[i].tokenIds.length;
      }
      const newSeq = bestSeq.slice(bestMatchPages);
      this.sequences[seqIdx].clear();
      this.sequences[seqIdx] = newSeq;
      return inputIds.slice(matchTokens);
    }

    this.sequences[seqIdx].clear();
    return inputIds.slice();
  }

  appendTokens(seqIdx: number, tokens: number[]): void {
    while (this.sequences.length <= seqIdx) {
      this.sequences.push(new Sequence(this));
    }
    this.sequences[seqIdx].appendTokens(tokens);
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
      sequence.pushPage(page);
    }
    sequence.allocLen = newTotalLen;
    if (numNewPages > 0) {
      this.pagesDirtyHost = true;
      this.pagesDirtyDevice = true;
    }
  }

  allocDecodeToken(seqIdx: number): void {
    const sequence = this.sequences[seqIdx];
    const allocLen = sequence.allocLen;
    const pageIdxInSeq = Math.floor(allocLen / this.pageSize);
    if (pageIdxInSeq >= sequence.pages.length) {
      if (this.availablePages.length === 0) {
        throw new Error(`allocDecodeToken: no pages available`);
      }
      const pageId = this.availablePages.shift()!;
      const page: Page = { id: pageId, tokenIds: [], refs: 0 };
      sequence.pushPage(page);
      this.pagesDirtyHost = true;
      this.pagesDirtyDevice = true;
    }
    sequence.allocLen = allocLen + 1;
  }

  updateIndptr(ws: ExecutionWorkspace): void {
    const batchSize = this.sequences.length;
    ws.indptrH.withPinnedBuffer(buf => {
      buf.writeInt32LE(0, 0);
      let cumulative = 0;
      for (let i = 0; i < batchSize; i++) {
        cumulative += this.sequences[i].pages.length;
        buf.writeInt32LE(cumulative, (i + 1) * I32);
      }
    });

    this.indicesH.withPinnedBuffer(buf => {
      let indicesOff = 0;
      for (let i = 0; i < batchSize; i++) {
        for (const page of this.sequences[i].pages) {
          buf.writeInt32LE(page.id, indicesOff * I32);
          indicesOff++;
        }
      }
    });

    ws.lastPageLenH.withPinnedBuffer(buf => {
      for (let i = 0; i < batchSize; i++) {
        const allocLen = this.sequences[i].allocLen;
        const remainder = allocLen % this.pageSize;
        buf.writeInt32LE(remainder !== 0 ? remainder : (allocLen > 0 ? this.pageSize : 0), i * I32);
      }
    });
  }
}
