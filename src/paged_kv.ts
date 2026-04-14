import { GlmOps, BF16, I32 } from "./glm_ops";

export const BATCH_FLOAT_WS_SIZE = 128 * 1024 * 1024;
export const BATCH_INT_WS_SIZE = 8 * 1024 * 1024;
export const BATCH_PINNED_INT_WS_SIZE = 8 * 1024 * 1024;
export const PAGE_SIZE = 16;
export const DECODE_PLAN_INFO_SIZE = 10;
export const PREFILL_PLAN_INFO_SIZE = 15;

export class PagedKVCache {
  private glm: GlmOps;
  readonly nKv: number;
  readonly hd: number;
  readonly nLayers: number;
  readonly maxPages: number;
  readonly maxBatch: number;
  readonly pageSize: number;
  kData: number[];
  vData: number[];
  indices: number;
  indptrD: number;
  lastPageLen: number;
  indptrH: number;
  lastPageLenH: number;
  numPagesUsed: number;
  seqPages: number[][];
  seqKvLens: number[];

  constructor(glm: GlmOps, nKv: number, hd: number, nLayers: number, maxPages: number, maxBatch: number, pageSize = PAGE_SIZE) {
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
      this.kData.push(glm.alloc(maxPages * nKv * pageSize * hd * BF16));
      this.vData.push(glm.alloc(maxPages * nKv * pageSize * hd * BF16));
    }
    this.indices = glm.alloc(maxPages * I32);
    this.indptrD = glm.alloc((maxBatch + 1) * I32);
    this.lastPageLen = glm.alloc(maxBatch * I32);
    this.indptrH = glm.allocPinned((maxBatch + 1) * I32);
    this.lastPageLenH = glm.allocPinned(maxBatch * I32);
    this.numPagesUsed = 0;
    this.seqPages = [];
    this.seqKvLens = [];
  }

  free(): void {
    const glm = this.glm;
    for (const ptr of this.kData) glm.freeBuf(ptr);
    for (const ptr of this.vData) glm.freeBuf(ptr);
    glm.freeBuf(this.indices);
    glm.freeBuf(this.indptrD);
    glm.freeBuf(this.lastPageLen);
    glm.freePinned(this.indptrH);
    glm.freePinned(this.lastPageLenH);
    this.kData = [];
    this.vData = [];
  }

  reset(batchSize: number): void {
    if (batchSize > this.maxBatch) {
      throw new Error(`batchSize ${batchSize} exceeds maxBatch ${this.maxBatch}`);
    }
    this.numPagesUsed = 0;
    this.seqPages = Array.from({ length: batchSize }, () => []);
    this.seqKvLens = new Array(batchSize).fill(0);
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

    this.glm.writePinned(this.indptrH, Buffer.from(indptrBuf.buffer, indptrBuf.byteOffset, indptrBuf.byteLength));
    this.glm.writePinned(this.lastPageLenH, Buffer.from(lastPageLenBuf.buffer, lastPageLenBuf.byteOffset, lastPageLenBuf.byteLength));
    this.glm.h2d(this.indices, Buffer.from(indicesBuf.buffer, indicesBuf.byteOffset, indicesBuf.byteLength));
    this.glm.h2d(this.indptrD, Buffer.from(indptrBuf.buffer, indptrBuf.byteOffset, indptrBuf.byteLength));
    this.glm.h2d(this.lastPageLen, Buffer.from(lastPageLenBuf.buffer, lastPageLenBuf.byteOffset, lastPageLenBuf.byteLength));
  }
}

export class WorkspaceBuffers {
  private glm: GlmOps;
  floatWs: number;
  intWs: number;
  pinnedIntWs: number;
  decodePlanInfo: number;
  prefillPlanInfo: number;

  constructor(glm: GlmOps) {
    this.glm = glm;
    this.floatWs = glm.alloc(BATCH_FLOAT_WS_SIZE);
    this.intWs = glm.alloc(BATCH_INT_WS_SIZE);
    this.pinnedIntWs = glm.allocPinned(BATCH_PINNED_INT_WS_SIZE);
    this.decodePlanInfo = glm.allocPinned(DECODE_PLAN_INFO_SIZE * 8);
    this.prefillPlanInfo = glm.allocPinned(PREFILL_PLAN_INFO_SIZE * 8);
  }

  free(): void {
    const glm = this.glm;
    glm.freeBuf(this.floatWs);
    glm.freeBuf(this.intWs);
    glm.freePinned(this.pinnedIntWs);
    glm.freePinned(this.decodePlanInfo);
    glm.freePinned(this.prefillPlanInfo);
    this.floatWs = 0;
    this.intWs = 0;
  }
}
