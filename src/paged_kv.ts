import { type ChatCache } from "./chat_model";
import { DeviceOps, TensorParallelism } from "./device_ops";
import { MemcpyKind } from "./enums";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";
import { BATCH_FLOAT_WS_SIZE } from "./execution-workspace";
import { PAGE_SIZE, Sequence, type Page } from "./paged_sequence";

export { PAGE_SIZE, Sequence } from "./paged_sequence";
export type { Page } from "./paged_sequence";

export class PagedKVCache extends WorkspaceBase implements ChatCache {
  readonly nKv: number;
  readonly hd: number;
  readonly nLayers: number;
  readonly maxPages: number;
  readonly maxBatch: number;
  readonly pageSize: number;
  readonly contextParallel: boolean;
  readonly sparseMode: boolean;
  readonly bytesPerToken: number;
  kData: Tensor[];
  kScaleData: Tensor[];
  vData: Tensor[];
  ckvData: Tensor[];
  kpeData: Tensor[];
  /** FlashInfer float workspace — shared across workspaces. Only allocated for non-sparse modes (MHA + dense MLA). */
  floatWs!: Tensor;
  availablePages: number[];
  sequences: Sequence[];
  staging: Map<number, Sequence>;
  onPagePressure?: (requiredPages: number) => void;

  getPagedKV(): PagedKVCache { return this; }

  constructor(glm: DeviceOps, nKv: number, hd: number, nLayers: number, maxPages: number, maxBatch: number, physicalPageSize = PAGE_SIZE, kvLoraRank = 0, qkRopeDim = 0, contextParallel = false, indexHeadDim = 0, sharedLayers: boolean[] = []) {
    super(glm);
    this.nKv = nKv;
    this.hd = hd;
    this.nLayers = nLayers;
    this.maxPages = maxPages;
    this.maxBatch = maxBatch;
    this.pageSize = contextParallel ? physicalPageSize * glm.worldSize : physicalPageSize;
    this.contextParallel = contextParallel;
    this.sparseMode = indexHeadDim > 0 && kvLoraRank > 0;
    this.bytesPerToken = kvLoraRank > 0
      ? kvLoraRank + (kvLoraRank / 128) * 4 + qkRopeDim * 2
      : 0;
    this.kData = [];
    this.kScaleData = [];
    this.vData = [];
    this.ckvData = [];
    this.kpeData = [];
    if (sharedLayers.length > 0 && sharedLayers.length < nLayers)
      throw new Error(`PagedKVCache: sharedLayers length ${sharedLayers.length} != nLayers ${nLayers}`);
    for (let i = 0; i < nLayers; i++) {
      if (kvLoraRank > 0) {
        if (this.sparseMode) {
          this.ckvData.push(this.alloc([maxPages, this.pageSize, this.bytesPerToken], "U8", "ckv" + i, contextParallel ? TensorParallelism.Row : undefined));
        } else {
          this.ckvData.push(this.alloc([maxPages, this.pageSize, kvLoraRank], "BF16", "ckv" + i, contextParallel ? TensorParallelism.Row : undefined));
          this.kpeData.push(this.alloc([maxPages, this.pageSize, qkRopeDim], "BF16", "kpe" + i, contextParallel ? TensorParallelism.Row : undefined));
        }
        if (indexHeadDim > 0) {
          if (sharedLayers[i]) {
            this.kData.push(undefined!);
            this.kScaleData.push(undefined!);
          } else {
            const parallelism = contextParallel ? TensorParallelism.Row : undefined;
            this.kData.push(this.alloc([maxPages, this.pageSize, indexHeadDim], "U8", "k" + i, parallelism));
            this.kScaleData.push(this.alloc([maxPages, this.pageSize], "F32", "kScale" + i, parallelism));
          }
        }
      } else {
        this.kData.push(this.alloc([maxPages, nKv * this.pageSize * hd], "BF16", "k" + i, TensorParallelism.Row));
        this.vData.push(this.alloc([maxPages, nKv * this.pageSize * hd], "BF16", "v" + i, TensorParallelism.Row));
      }
    }
    this.availablePages = Array.from({ length: maxPages }, (_, i) => i);
    this.sequences = [];
    this.staging = new Map();

    if (!this.sparseMode) {
      this.floatWs = this.alloc([BATCH_FLOAT_WS_SIZE], "U8", "floatWs");
    }

    this.freeze();
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
  }

  unstageSequence(stagingKey: number): Sequence {
    const seq = this.staging.get(stagingKey);
    if (!seq) {
      throw new Error(`unstageSequence: no staged sequence with key ${stagingKey}`);
    }
    this.staging.delete(stagingKey);
    this.sequences.push(seq);
    return seq;
  }

  removeStagedSequence(stagingKey: number): void {
    const seq = this.staging.get(stagingKey);
    if (!seq) {
      throw new Error(`removeStagedSequence: no staged sequence with key ${stagingKey}`);
    }
    seq.clear();
    this.staging.delete(stagingKey);
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
  }

  ensureAvailablePages(requiredPages: number): void {
    if (requiredPages > this.availablePages.length) {
      this.onPagePressure?.(requiredPages);
    }
  }

  copySequence(dstSeqIdx: number, srcSeqIdx: number) {
    if (dstSeqIdx === srcSeqIdx)
      return;
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

  // Finds the best prefix match across active and staged sequences and returns the unmatched suffix.
  // Only full pages are kept/shared — the partial page is never shared because
  // the receiving sequence would write into it. Empty pages beyond the content
  // region are discarded. If the entire cache matches (self, no truncation needed),
  // returns the suffix immediately without touching pages.
  prefixMatch(seqIdx: number, inputIds: number[], copyPartial?: boolean): number[] {
    const targetSeq = this.ensureSequence(seqIdx);

    let bestSeq: Sequence | undefined;
    let bestMatchTokens = 0;
    for (const sequence of [...this.sequences, ...this.staging.values()]) {
      if (sequence.pages.length === 0) continue;
      const matchTokens = sequence.prefixMatch(inputIds);
      if (matchTokens > bestMatchTokens) {
        bestMatchTokens = matchTokens;
        bestSeq = sequence;
      }
    }

    if (bestSeq === targetSeq && bestMatchTokens === targetSeq.reportedTokenCount()) {
      return inputIds.slice(bestMatchTokens);
    }

    if (!bestSeq || bestMatchTokens === 0) {
      targetSeq.clear();
      return inputIds.slice();
    }

    // full pages can be shared, memcpy is needed for a partial page.
    const keepPages = Math.floor(bestMatchTokens / this.pageSize);

    if (bestSeq === targetSeq) {
      while (targetSeq.pages.length > keepPages) {
        targetSeq.popPage();
      }
      return inputIds.slice(targetSeq.allocLen);
    }

    if (keepPages > 0) {
      const newSeq = new Sequence(this);
      for (let i = 0; i < keepPages; i++) {
        newSeq.pushPage(bestSeq.pages[i], this.pageSize);
      }
      targetSeq.clear();
      this.sequences[seqIdx] = newSeq;
      const partialPage = bestMatchTokens % this.pageSize !== 0;
      if (copyPartial && partialPage) {
        this.allocAppendPages(seqIdx, this.pageSize);
        const srcPage = bestSeq.pages[keepPages];
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
      if (this.kData[i]) this.copyPageRow(this.kData[i], srcPageId, dstPageId);
      if (this.kScaleData[i]) this.copyPageRow(this.kScaleData[i], srcPageId, dstPageId);
    }
    for (let i = 0; i < this.vData.length; i++) {
      this.copyPageRow(this.vData[i], srcPageId, dstPageId);
    }
    for (let i = 0; i < this.ckvData.length; i++) {
      this.copyPageRow(this.ckvData[i], srcPageId, dstPageId);
    }
    for (let i = 0; i < this.kpeData.length; i++) {
      this.copyPageRow(this.kpeData[i], srcPageId, dstPageId);
    }
  }

  copyPageRow(tensor: Tensor, srcPageId: number, dstPageId: number): void {
    const rowBytes = Tensor.byteCount(tensor.shape.slice(1), tensor.type);
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
    this.ensureAvailablePages(numNewPages);
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
