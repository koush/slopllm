import { GlmOps, BF16, I32 } from "./glm_ops";
import { WorkspaceBase } from "./chat_model";
import type { ChatCache } from "./chat_model";
import { Tensor } from "./tensor";

export const BATCH_FLOAT_WS_SIZE = 128 * 1024 * 1024;
export const BATCH_INT_WS_SIZE = 8 * 1024 * 1024;
export const BATCH_PINNED_INT_WS_SIZE = 8 * 1024 * 1024;
export const PAGE_SIZE = 16;
export const DECODE_PLAN_INFO_SIZE = 10;
export const PREFILL_PLAN_INFO_SIZE = 15;

function longestPrefix(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) return i;
  }
  return len;
}

export class PagedKVCache extends WorkspaceBase implements ChatCache {
  private glm: GlmOps;
  readonly nKv: number;
  readonly hd: number;
  readonly nLayers: number;
  readonly maxPages: number;
  readonly maxBatch: number;
  readonly pageSize: number;
  kData: Tensor[];
  vData: Tensor[];
  indices: Tensor;
  indptrD: Tensor;
  lastPageLen: Tensor;
  indptrH: Tensor;
  lastPageLenH: Tensor;
  slotMapping: Tensor;
  slotMappingH: Tensor;
  numPagesUsed: number;
  seqPages: number[][];
  seqKvLens: number[];
  cachedTokenIds: number[][];

  constructor(glm: GlmOps, nKv: number, hd: number, nLayers: number, maxPages: number, maxBatch: number, pageSize = PAGE_SIZE) {
    super();
    this.glm = glm;
    this.nKv = nKv;
    this.hd = hd;
    this.nLayers = nLayers;
    this.maxPages = maxPages;
    this.maxBatch = maxBatch;
    this.pageSize = pageSize;
    this.kData = [];
    this.vData = [];
    for (let i = 0; i < nLayers; i++) {
      this.kData.push(this.alloc(glm, [maxPages * nKv * pageSize * hd * BF16], "U8"));
      this.vData.push(this.alloc(glm, [maxPages * nKv * pageSize * hd * BF16], "U8"));
    }
    this.indices = this.alloc(glm, [maxPages * I32], "I32", "indices");
    this.indptrD = this.alloc(glm, [(maxBatch + 1) * I32], "I32", "indptrD");
    this.lastPageLen = this.alloc(glm, [maxBatch * I32], "I32", "lastPageLen");
    this.indptrH = this.allocPinned(glm, [(maxBatch + 1) * I32], "I32", "indptrH");
    this.lastPageLenH = this.allocPinned(glm, [maxBatch * I32], "I32", "lastPageLenH");
    this.slotMapping = this.alloc(glm, [maxBatch * I32], "I32", "slotMapping");
    this.slotMappingH = this.allocPinned(glm, [maxBatch * I32], "I32", "slotMappingH");
    this.numPagesUsed = 0;
    this.seqPages = [];
    this.seqKvLens = [];
    this.cachedTokenIds = [];
  }

  reset(batchSize: number): void {
    if (batchSize > this.maxBatch) {
      throw new Error(`batchSize ${batchSize} exceeds maxBatch ${this.maxBatch}`);
    }
    this.numPagesUsed = 0;
    this.seqPages = Array.from({ length: batchSize }, () => []);
    this.seqKvLens = new Array(batchSize).fill(0);
    this.cachedTokenIds = Array.from({ length: batchSize }, () => []);
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
      this.seqPages[seqIdx] = [];
      this.seqKvLens[seqIdx] = 0;
      return;
    }
    const pageSize = this.pageSize;
    const newPageCount = Math.ceil(newLen / pageSize);
    const oldPageCount = this.seqPages[seqIdx].length;
    const removedPages = oldPageCount - newPageCount;
    if (removedPages > 0 && this.seqPages[seqIdx][oldPageCount - 1] === this.numPagesUsed - 1) {
      this.numPagesUsed -= removedPages;
    }
    this.seqPages[seqIdx] = this.seqPages[seqIdx].slice(0, newPageCount);
    this.seqKvLens[seqIdx] = newLen;
  }

  allocPrefillPages(seqIdx: number, seqLen: number): [number, number] {
    const pageSize = this.pageSize;
    const numPages = Math.ceil(seqLen / pageSize);
    const startPage = this.numPagesUsed;
    this.numPagesUsed += numPages;
    this.seqPages[seqIdx] = Array.from({ length: numPages }, (_, i) => startPage + i);
    this.seqKvLens[seqIdx] = seqLen;
    return [startPage, numPages];
  }

  allocAppendPages(seqIdx: number, numNewTokens: number): [number, number] {
    const pageSize = this.pageSize;
    const currentLen = this.seqKvLens[seqIdx];
    const currentPageCount = this.seqPages[seqIdx].length;
    const newTotalLen = currentLen + numNewTokens;
    const newPageCount = Math.ceil(newTotalLen / pageSize);
    const numNewPages = newPageCount - currentPageCount;
    const startPage = this.numPagesUsed;
    for (let i = 0; i < numNewPages; i++) {
      this.seqPages[seqIdx].push(startPage + i);
    }
    this.numPagesUsed += numNewPages;
    this.seqKvLens[seqIdx] = newTotalLen;
    return [startPage, numNewPages];
  }

  allocDecodeToken(seqIdx: number): [number, number] {
    const kvLen = this.seqKvLens[seqIdx];
    const pageSize = this.pageSize;
    const pageIdxInSeq = Math.floor(kvLen / pageSize);
    if (pageIdxInSeq >= this.seqPages[seqIdx].length) {
      const newPage = this.numPagesUsed;
      this.numPagesUsed += 1;
      this.seqPages[seqIdx].push(newPage);
    }
    this.seqKvLens[seqIdx] = kvLen + 1;
    const absPage = this.seqPages[seqIdx][pageIdxInSeq];
    const slotInPage = kvLen % pageSize;
    return [absPage, slotInPage];
  }

  updateIndptr(): void {
    const batchSize = this.seqPages.length;
    const indptr = new Array(batchSize + 1).fill(0);
    for (let i = 0; i < batchSize; i++) {
      indptr[i + 1] = indptr[i] + this.seqPages[i].length;
    }
    const allIndices: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      allIndices.push(...this.seqPages[i]);
    }
    const indicesBuf = Int32Array.from(allIndices);
    const indptrBuf = Int32Array.from(indptr);

    const lastPageLenList = new Array(batchSize);
    for (let i = 0; i < batchSize; i++) {
      const kvLen = this.seqKvLens[i];
      const remainder = kvLen % this.pageSize;
      lastPageLenList[i] = remainder !== 0 ? remainder : (kvLen > 0 ? this.pageSize : 0);
    }
    const lastPageLenBuf = Int32Array.from(lastPageLenList);

    this.glm.writePinned(this.indptrH.data, Buffer.from(indptrBuf.buffer, indptrBuf.byteOffset, indptrBuf.byteLength));
    this.glm.writePinned(this.lastPageLenH.data, Buffer.from(lastPageLenBuf.buffer, lastPageLenBuf.byteOffset, lastPageLenBuf.byteLength));
    this.glm.h2d(this.indices.data, Buffer.from(indicesBuf.buffer, indicesBuf.byteOffset, indicesBuf.byteLength));
    this.glm.h2d(this.indptrD.data, Buffer.from(indptrBuf.buffer, indptrBuf.byteOffset, indptrBuf.byteLength));
    this.glm.h2d(this.lastPageLen.data, Buffer.from(lastPageLenBuf.buffer, lastPageLenBuf.byteOffset, lastPageLenBuf.byteLength));
  }

  updateSlotMapping(writeLocations: [number, number][], pageSize: number): void {
    const batchSize = writeLocations.length;
    const slotMappingBuf = new Int32Array(batchSize);
    for (let i = 0; i < batchSize; i++) {
      const [absPage, slotInPage] = writeLocations[i];
      slotMappingBuf[i] = absPage * pageSize + slotInPage;
    }
    this.glm.writePinned(this.slotMappingH.data, Buffer.from(slotMappingBuf.buffer, slotMappingBuf.byteOffset, slotMappingBuf.byteLength));
    this.glm.h2d(this.slotMapping.data, Buffer.from(slotMappingBuf.buffer, slotMappingBuf.byteOffset, slotMappingBuf.byteLength));
  }
}
