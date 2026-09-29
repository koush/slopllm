import { type ChatCache } from "./chat_model";
import { DeviceOps, TensorParallelism, type MemcpyBatchEntry } from "./device_ops";
import { MemcpyKind } from "./enums";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";
import { BATCH_FLOAT_WS_SIZE } from "./execution-workspace";
import { PAGE_SIZE, Sequence, type Page } from "./paged_sequence";

export { PAGE_SIZE, Sequence } from "./paged_sequence";
export type { Page } from "./paged_sequence";

export class PageAllocationError extends Error {
  constructor(requiredPages: number, availablePages: number) {
    super(`allocAppendPages: need ${requiredPages} pages, ${availablePages} available`);
    this.name = "PageAllocationError";
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
  readonly sparseMode: boolean;
  readonly pinned: boolean;
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

  constructor(ops: DeviceOps, nKv: number, hd: number, nLayers: number, maxPages: number, maxBatch: number, physicalPageSize = PAGE_SIZE, kvLoraRank = 0, qkRopeDim = 0, contextParallel = false, indexHeadDim = 0, sharedLayers: boolean[] = [], pinned = false) {
    super(ops);
    this.pinned = pinned;
    this.nKv = nKv;
    this.hd = hd;
    this.nLayers = nLayers;
    this.maxPages = maxPages;
    this.maxBatch = maxBatch;
    this.pageSize = contextParallel ? physicalPageSize * ops.worldSize : physicalPageSize;
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
      const alloc = (shape: number[], type: string, name: string, parallelism?: TensorParallelism): Tensor =>
        pinned ? this.allocPinned(shape, type, name, parallelism) : this.alloc(shape, type, name, parallelism);
      if (kvLoraRank > 0) {
        if (this.sparseMode) {
          this.ckvData.push(alloc([maxPages, this.pageSize, this.bytesPerToken], "U8", "ckv" + i, contextParallel ? TensorParallelism.Row : undefined));
        } else {
          this.ckvData.push(alloc([maxPages, this.pageSize, kvLoraRank], "BF16", "ckv" + i, contextParallel ? TensorParallelism.Row : undefined));
          this.kpeData.push(alloc([maxPages, this.pageSize, qkRopeDim], "BF16", "kpe" + i, contextParallel ? TensorParallelism.Row : undefined));
        }
        if (indexHeadDim > 0) {
          if (sharedLayers[i]) {
            this.kData.push(undefined!);
            this.kScaleData.push(undefined!);
          } else {
            const parallelism = contextParallel ? TensorParallelism.Row : undefined;
            this.kData.push(alloc([maxPages, this.pageSize, indexHeadDim], "U8", "k" + i, parallelism));
            this.kScaleData.push(alloc([maxPages, this.pageSize], "F32", "kScale" + i, parallelism));
          }
        }
      } else {
        this.kData.push(alloc([maxPages, nKv * this.pageSize * hd], "BF16", "k" + i, TensorParallelism.Row));
        this.vData.push(alloc([maxPages, nKv * this.pageSize * hd], "BF16", "v" + i, TensorParallelism.Row));
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
    dstSeq.targetToken = srcSeq.targetToken;
  }

  // Layer tensors [dst, src] for the layer's allocated caches (undefined
  // entries — shared-indexer layers — are skipped).
  private layerCachePairs(src: PagedKVCache, layer: number): [Tensor, Tensor][] {
    const pairs: [Tensor, Tensor][] = [];
    const add = (dst: Tensor | undefined, s: Tensor | undefined) => {
      if (dst && s) pairs.push([dst, s]);
    };
    add(this.kData[layer], src.kData[layer]);
    add(this.kScaleData[layer], src.kScaleData[layer]);
    add(this.vData[layer], src.vData[layer]);
    add(this.ckvData[layer], src.ckvData[layer]);
    add(this.kpeData[layer], src.kpeData[layer]);
    return pairs;
  }

  private assertCacheLayoutCompatible(src: PagedKVCache): void {
    if (src.pageSize !== this.pageSize) {
      throw new Error(`copyPrefixFrom: pageSize mismatch (dst ${this.pageSize}, src ${src.pageSize})`);
    }
    if (src.nLayers !== this.nLayers) {
      throw new Error(`copyPrefixFrom: nLayers mismatch (dst ${this.nLayers}, src ${src.nLayers})`);
    }
    if (src.sparseMode !== this.sparseMode || src.bytesPerToken !== this.bytesPerToken) {
      throw new Error("copyPrefixFrom: KV packing mismatch (sparse/dense or bytesPerToken)");
    }
    for (let i = 0; i < this.nLayers; i++) {
      const check = (dst: Tensor | undefined, s: Tensor | undefined, what: string) => {
        if (!!dst !== !!s) {
          throw new Error(`copyPrefixFrom: ${what}${i} present in only one cache`);
        }
        if (!dst || !s) return;
        const dstRow = Tensor.byteCount(dst.shape.slice(1), dst.type);
        const srcRow = Tensor.byteCount(s.shape.slice(1), s.type);
        if (dst.type !== s.type || dstRow !== srcRow) {
          throw new Error(`copyPrefixFrom: ${what}${i} layout mismatch (${dst.type} ${dstRow}B vs ${s.type} ${srcRow}B)`);
        }
      };
      check(this.kData[i], src.kData[i], "k");
      check(this.kScaleData[i], src.kScaleData[i], "kScale");
      check(this.vData[i], src.vData[i], "v");
      check(this.ckvData[i], src.ckvData[i], "ckv");
      check(this.kpeData[i], src.kpeData[i], "kpe");
    }
  }

  /**
   * Materializes the full-page prefix of a sequence from another cache into
   * dstSeqIdx and returns the uncopied suffix.
   *
   * The destination first reuses full pages from its own cache, without
   * reserving a final token for resampling. The remaining full pages are
   * copied in one batched CUDA submission issued from the source's stream
   * (posted writes; cross-device use needs the usual P2P
   * barrier before consuming). Reservation runs through the normal
   * page-allocation path (page pressure applies). Requires identical cache
   * layout: same pageSize, nLayers, and per-layer row layout.
   *
   * Returns the uncopied suffix — the partial-page tail — for the caller to
   * prefill through the normal chunked-prefill path, appending
   * srcSeq.targetToken to the last chunk to
   * continue the sequence exactly. When the suffix is empty (page-aligned
   * content), srcSeq.targetToken is already pending as the decode input.
   * Requires identical cache layout: same pageSize, nLayers, and per-layer row
   * layout.
   *
   * maxPages bounds the number of full pages materialized, regardless of how
   * much the source holds. This bounds the copy both by the caller's device
   * budget and — critically — by the caller's prompt, so a source whose
   * history runs past the prompt cannot inject foreign tokens into the
   * destination.
   */
  copyPrefixFrom(src: PagedKVCache, srcSeq: Sequence, dstSeqIdx: number, maxPages?: number): number[] {
    if (srcSeq.pagedKvCache !== src) {
      throw new Error("copyPrefixFrom: sequence does not belong to src cache");
    }
    this.assertCacheLayoutCompatible(src);
    const pageSize = this.pageSize;
    const srcTokens = srcSeq.getTokenIds();
    const fullPages = Math.min(Math.floor(srcTokens.length / pageSize), maxPages ?? Infinity);
    const matched = srcTokens.length - this.matchPrefix(dstSeqIdx, srcTokens, fullPages).length;
    const firstCopyPage = matched / pageSize;
    const dstSeq = this.ensureSequence(dstSeqIdx);
    if (fullPages > firstCopyPage) {
      this.allocAppendPages(dstSeqIdx, (fullPages - firstCopyPage) * pageSize);
    }

    // Every copied page is freshly allocated, so a shared page is never
    // overwritten. The partial tail is left for the caller to prefill.
    const entries: MemcpyBatchEntry[] = [];
    let runSrc = -2, runDst = -2, runPages = 0;
    const flushRun = () => {
      if (runPages === 0) return;
      for (let i = 0; i < this.nLayers; i++) {
        for (const [dstT, srcT] of this.layerCachePairs(src, i)) {
          const rowBytes = Tensor.byteCount(dstT.shape.slice(1), dstT.type);
          entries.push({
            dst: dstT,
            src: srcT,
            bytes: runPages * rowBytes,
            dstOffset: runDst * rowBytes,
            srcOffset: runSrc * rowBytes,
          });
        }
      }
      runPages = 0;
    };
    for (let p = firstCopyPage; p < fullPages; p++) {
      const sp = srcSeq.pages[p];
      const dp = dstSeq.pages[p];
      if (!sp || !dp) {
        throw new Error(`copyPrefixFrom: missing page ${p} (src ${!!sp}, dst ${!!dp})`);
      }
      if (sp.id === runSrc + runPages && dp.id === runDst + runPages) {
        runPages++;
      } else {
        flushRun();
        runSrc = sp.id;
        runDst = dp.id;
        runPages = 1;
      }
    }
    flushRun();
    if (entries.length) {
      this.ops.memcpyBatchAsync(entries);
    }

    // Commit the copied tokens (fresh pages have empty tokenIds) and leave
    // the first uncopied token pending as the next input — or, once the whole
    // history is materialized, the source's own pending decode input.
    const contentLen = fullPages * pageSize;
    const nextInput = contentLen < srcTokens.length ? srcTokens[contentLen] : srcSeq.targetToken;
    this.reportTokens(dstSeqIdx, srcTokens.slice(matched, contentLen), nextInput);
    return srcTokens.slice(contentLen);
  }

  ensureSequence(seqIdx: number) {
    this.sequences[seqIdx] ??= new Sequence(this);
    return this.sequences[seqIdx];
  }

  // Finds the best prefix match across active and staged sequences and returns the unmatched suffix.
  // Reuse only full pages, leaving at least one prompt token for prefill so
  // the caller selects a fresh output using its own sampling policy. An exact
  // page-aligned match replays the last full page. An optional pure permission
  // callback may authorize consuming an entire sequence when the prompt
  // strictly extends its committed history and its partial page is private.
  // Such a resume transfers ownership without copies/allocations; an active
  // donor slot is left empty (indices stay stable), or its staging entry is
  // removed. The callback may be consulted for candidates that do not win.
  // Without permission, matching remains full-page-only, even for self-matches.
  prefixMatch(seqIdx: number, inputIds: number[], allowResume?: (sequence: Sequence) => boolean): number[] {
    const maxPages = Math.floor(Math.max(0, inputIds.length - 1) / this.pageSize);
    return this.matchPrefix(seqIdx, inputIds, maxPages, allowResume);
  }

  private matchPrefix(seqIdx: number, inputIds: number[], maxPages: number, allowResume?: (sequence: Sequence) => boolean): number[] {
    const targetSeq = this.ensureSequence(seqIdx);

    let bestSeq: Sequence | undefined;
    let bestMatchTokens = 0;
    let bestResume = false;
    for (const sequence of [...this.sequences, ...this.staging.values()]) {
      if (sequence.pages.length === 0) continue;
      const commonTokens = Math.min(sequence.prefixMatch(inputIds), sequence.allocLen);
      let matchTokens = Math.min(Math.floor(commonTokens / this.pageSize), maxPages) * this.pageSize;
      const resume = !!allowResume
        && commonTokens === sequence.allocLen
        && commonTokens < inputIds.length
        && commonTokens % this.pageSize !== 0
        && sequence.reportedTokenCount() === sequence.allocLen
        && sequence.pages.length === Math.ceil(commonTokens / this.pageSize)
        && sequence.pages.at(-1)!.refs === 1
        && allowResume(sequence);
      if (resume) matchTokens = commonTokens;
      if (matchTokens > bestMatchTokens) {
        bestMatchTokens = matchTokens;
        bestSeq = sequence;
        bestResume = resume;
      }
    }

    if (!bestSeq || bestMatchTokens === 0) {
      targetSeq.clear();
      targetSeq.targetToken = inputIds[0];
      return inputIds.slice();
    }

    if (bestResume) {
      if (bestSeq !== targetSeq) {
        const sourceIndex = this.sequences.indexOf(bestSeq);
        if (sourceIndex !== -1) {
          this.sequences[sourceIndex] = new Sequence(this);
        } else {
          for (const [key, sequence] of this.staging) {
            if (sequence === bestSeq) {
              this.staging.delete(key);
              break;
            }
          }
        }
        targetSeq.clear();
        this.sequences[seqIdx] = bestSeq;
      }
      // The rendered prompt supplies the next input; the donor's sampled
      // targetToken is not committed KV and must not be injected here.
      bestSeq.targetToken = inputIds[bestMatchTokens];
      return inputIds.slice(bestMatchTokens);
    }

    const keepPages = bestMatchTokens / this.pageSize;
    const nextInput = inputIds[bestMatchTokens] ?? bestSeq.targetToken;

    if (bestSeq === targetSeq) {
      while (targetSeq.pages.length > keepPages) {
        targetSeq.popPage();
      }
    } else {
      const newSeq = new Sequence(this);
      for (let i = 0; i < keepPages; i++) {
        newSeq.pushPage(bestSeq.pages[i], this.pageSize);
      }
      targetSeq.clear();
      this.sequences[seqIdx] = newSeq;
    }
    this.sequences[seqIdx].targetToken = nextInput;
    return inputIds.slice(bestMatchTokens);
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
      tensor.pinned ? MemcpyKind.HostToHost : MemcpyKind.DeviceToDevice,
    );
  }

  reportTokens(seqIdx: number, tokens: number[], targetToken?: number): void {
    if (seqIdx >= this.sequences.length) throw new Error(`reportTokens: seqIdx ${seqIdx} out of range (${this.sequences.length} sequences)`);
    this.sequences[seqIdx].reportTokens(tokens, targetToken);
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
      throw new PageAllocationError(numNewPages, this.availablePages.length);
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
