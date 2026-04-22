import type { ChatCache, ChatModel } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { BATCH_FLOAT_WS_SIZE, BATCH_INT_WS_SIZE, BATCH_PINNED_INT_WS_SIZE, BF16, I32 } from "./glm_ops";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export const PAGE_SIZE = 16;
export const DECODE_PLAN_INFO_SIZE = 10;
export const PREFILL_PLAN_INFO_SIZE = 15;

export interface BatchState {
  batchSize: number;
  totalTokens: number;
  seqLens: number[];
  readonly isDecode: boolean;
  readonly ws: ExecutionWorkspace;
  readonly cache: ChatCache;
}

function longestPrefix(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) return i;
  }
  return len;
}

export class ExecutionWorkspace extends WorkspaceBase {
  floatWs: Tensor;
  intWs: Tensor;
  pinnedIntWs: Tensor;
  decodePlanInfo: Tensor;
  prefillPlanInfo: Tensor;
  inputIdsBuf: Tensor;
  positionIds: Tensor;
  lastIdx: Tensor;
  qoIndptrD: Tensor;
  slotMapping: Tensor;
  indptrD: Tensor;
  indptrH: Tensor;
  lastPageLen: Tensor;
  lastPageLenH: Tensor;

  constructor(glm: DeviceOps, B: number, S: number) {
    super(glm);

    this.floatWs = this.alloc([BATCH_FLOAT_WS_SIZE], "U8", "floatWs");
    this.intWs = this.alloc([BATCH_INT_WS_SIZE], "U8", "intWs");
    this.pinnedIntWs = this.allocPinned([BATCH_PINNED_INT_WS_SIZE], "U8", "pinnedIntWs");
    this.decodePlanInfo = this.allocPinned([DECODE_PLAN_INFO_SIZE * 8], "U8", "decodePlanInfo");
    this.prefillPlanInfo = this.allocPinned([PREFILL_PLAN_INFO_SIZE * 8], "U8", "prefillPlanInfo");

    this.positionIds = this.alloc([B * S], "I32", "positionIds");
    this.lastIdx = this.alloc([B], "I32", "lastIdx");
    this.inputIdsBuf = this.alloc([B * S], "I32", "inputIdsBuf");
    this.qoIndptrD = this.alloc([B + 1], "I32", "qoIndptrD");
    this.slotMapping = this.alloc([B * S], "I32", "slotMapping");
    this.indptrD = this.alloc([(B + 1) * I32], "I32", "indptrD");
    this.indptrH = this.allocPinned([(B + 1) * I32], "I32", "indptrH");
    this.lastPageLen = this.alloc([B * I32], "I32", "lastPageLen");
    this.lastPageLenH = this.allocPinned([B * I32], "I32", "lastPageLenH");
  }

  flashDecode(query: Tensor, pagedKV: PagedKVCache, cacheIdx: number, batchSize: number, nHeads: number, nKv: number, hd: number, smScale: number): Tensor {
    const out = this.alloc([batchSize, nHeads, 1, hd], query.type, undefined, query.parallelism);
    this.glm.batchDecodeRun(
      query, out,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
      pagedKV.indices, this.indptrD, this.lastPageLen,
      this.floatWs, this.intWs,
      this.decodePlanInfo,
      batchSize, nHeads, nKv, hd, pagedKV.pageSize, smScale
    );
    return out;
  }

  flashPrefillPaged(query: Tensor, pagedKV: PagedKVCache, cacheIdx: number, totalTokens: number, batchSize: number, nHeads: number, nKv: number, hd: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): Tensor {
    const out = this.alloc([1, nHeads, totalTokens, hd], query.type, undefined, query.parallelism);
    this.glm.batchPrefillPagedRun(
      query, out,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
      pagedKV.indices, this.indptrD, this.lastPageLen,
      this.floatWs, this.intWs,
      this.qoIndptrD,
      this.prefillPlanInfo,
      totalTokens, batchSize, nHeads, nKv, hd, pagedKV.pageSize,
      qStrideN, qStrideH, maskMode, smScale
    );
    return out;
  }

  kvCacheWrite(kRope: Tensor, vBuf: Tensor, state: BatchState, cacheIdx: number, nKv: number, hd: number): void {
    const pagedKV = state.cache.getPagedKV();
    const BS = state.totalTokens;
    const kTokenStride = state.isDecode ? nKv * hd : hd;
    const kHeadStride = state.isDecode ? hd : BS * hd;
    const vTokenStride = nKv * hd;
    const vHeadStride = hd;
    this.glm.kvCacheWrite(
      kRope, vBuf,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
      this.slotMapping,
      BS, nKv, hd, pagedKV.pageSize,
      kTokenStride, kHeadStride, vTokenStride, vHeadStride
    );
  }

  plan(model: ChatModel, inputIdsList: number[][], cache: ChatCache, enableCudaGraph = false): BatchState {
    const pagedKV = cache.getPagedKV();
    const cfg = model.cfg;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const batchSize = inputIdsList.length;
    const seqLens = inputIdsList.map(ids => ids.length);
    const totalTokens = seqLens.reduce((a, b) => a + b, 0);
    const isDecode = seqLens.every(s => s === 1);

    if (enableCudaGraph && !isDecode) {
      throw new Error("enableCudaGraph requires all sequences to have length 1 (decode mode)");
    }

    if (pagedKV.seqPages.length !== batchSize) {
      throw new Error(`plan: pagedKV has ${pagedKV.seqPages.length} sequences, expected ${batchSize}`);
    }

    const allIds: number[] = [];
    for (const ids of inputIdsList) allIds.push(...ids);
    const idsBuf = Int32Array.from(allIds);
    this.inputIdsBuf.h2d(Buffer.from(idsBuf.buffer, idsBuf.byteOffset, idsBuf.byteLength));

    if (isDecode) {
      const writeLocations: [number, number][] = [];
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        writeLocations.push(pagedKV.allocDecodeToken(seqIdx));
      }

      pagedKV.updateIndptr(this);

      const slotMappingBuf = new Int32Array(batchSize);
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        const [absPage, slotInPage] = writeLocations[seqIdx];
        slotMappingBuf[seqIdx] = absPage * pageSize + slotInPage;
      }
      this.slotMapping.h2d(Buffer.from(slotMappingBuf.buffer, slotMappingBuf.byteOffset, slotMappingBuf.byteLength));

      const posIds = new Array(batchSize);
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        posIds[seqIdx] = pagedKV.seqKvLens[seqIdx] - 1;
      }
      const posIdsBuf = Int32Array.from(posIds);
      this.positionIds.h2d(Buffer.from(posIdsBuf.buffer, posIdsBuf.byteOffset, posIdsBuf.byteLength));

      this.glm.batchDecodePlan(
        this.floatWs, BATCH_FLOAT_WS_SIZE,
        this.intWs, this.pinnedIntWs, BATCH_INT_WS_SIZE,
        this.decodePlanInfo,
        this.indptrH,
        batchSize,
        nHeads, nKv, hd, pageSize,
        enableCudaGraph
      );

      return { batchSize, totalTokens, seqLens, isDecode: true, ws: this, cache };
    }

    const startPos = pagedKV.seqKvLens.slice();

    model.prefillBatchPlanHook(inputIdsList, seqLens, totalTokens, startPos, cache);

    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pagedKV.allocAppendPages(seqIdx, seqLens[seqIdx]);
    }

    const qoIndptr = [0];
    for (const s of seqLens) {
      qoIndptr.push(qoIndptr[qoIndptr.length - 1] + s);
    }
    const qoIndptrBuf = Int32Array.from(qoIndptr);

    pagedKV.updateIndptr(this);

    const posIds: number[] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      for (let p = 0; p < seqLens[seqIdx]; p++) {
        posIds.push(startPos[seqIdx] + p);
      }
    }
    const posIdsBuf = Int32Array.from(posIds);
    this.positionIds.h2d(Buffer.from(posIdsBuf.buffer, posIdsBuf.byteOffset, posIdsBuf.byteLength));

    const lastIndices: number[] = [];
    let offset = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      lastIndices.push(offset + seqLens[seqIdx] - 1);
      offset += seqLens[seqIdx];
    }
    const lastIdxBuf = Int32Array.from(lastIndices);
    this.lastIdx.h2d(Buffer.from(lastIdxBuf.buffer, lastIdxBuf.byteOffset, lastIdxBuf.byteLength));

    using qoIndptrHost = this.allocPinned([(batchSize + 1)], "I32");
    this.glm.writePinned(qoIndptrHost, Buffer.from(qoIndptrBuf.buffer, qoIndptrBuf.byteOffset, qoIndptrBuf.byteLength));

    this.glm.batchPrefillPagedPlan(
      this.floatWs, BATCH_FLOAT_WS_SIZE,
      this.intWs, this.pinnedIntWs, BATCH_INT_WS_SIZE,
      this.prefillPlanInfo,
      qoIndptrHost, this.indptrH,
      totalTokens, batchSize,
      nHeads, nKv, hd,
      pageSize,
      1
    );

    this.qoIndptrD.h2d(Buffer.from(qoIndptrBuf.buffer, qoIndptrBuf.byteOffset, qoIndptrBuf.byteLength));

    const slotMapping: number[] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      const pages = pagedKV.seqPages[seqIdx];
      for (let pos = 0; pos < seqLens[seqIdx]; pos++) {
        const kvPos = startPos[seqIdx] + pos;
        const pageIdxInSeq = Math.floor(kvPos / pagedKV.pageSize);
        const offsetInPage = kvPos % pagedKV.pageSize;
        const absPage = pages[pageIdxInSeq];
        slotMapping.push(absPage * pagedKV.pageSize + offsetInPage);
      }
    }
    const slotMappingBuf = Int32Array.from(slotMapping);
    this.slotMapping.h2d(Buffer.from(slotMappingBuf.buffer, slotMappingBuf.byteOffset, slotMappingBuf.byteLength));

    return { batchSize, totalTokens, seqLens, isDecode: false, ws: this, cache };
  }

  planDecode(model: ChatModel, tokenIdsList: number[], cache: ChatCache, enableCudaGraph = false): BatchState {
    return this.plan(model, tokenIdsList.map(t => [t]), cache, enableCudaGraph);
  }

  forwardEager(model: ChatModel, inputIdsList: number[][], cache: ChatCache): number[] {
    const state = this.plan(model, inputIdsList, cache);
    const logits = model.forward(state);
    using argmaxResult = logits.argmax();
    return argmaxResult.readInt32LE();
  }

  forwardDecode(model: ChatModel, state: BatchState): Tensor {
    return model.forward(state);
  }

  forwardEagerDecode(model: ChatModel, tokenIdsList: number[], cache: ChatCache): number[] {
    return this.forwardEager(model, tokenIdsList.map(t => [t]), cache);
  }
}

export class PagedKVCache extends WorkspaceBase implements ChatCache {
  readonly nKv: number;
  readonly hd: number;
  readonly nLayers: number;
  readonly maxPages: number;
  readonly maxBatch: number;
  readonly pageSize: number;
  kData: Tensor[];
  vData: Tensor[];
  indices: Tensor;
  numPagesUsed: number;
  seqPages: number[][];
  seqKvLens: number[];
  cachedTokenIds: number[][];

  getPagedKV(): PagedKVCache { return this; }

  constructor(glm: DeviceOps, nKv: number, hd: number, nLayers: number, maxPages: number, maxBatch: number, pageSize = PAGE_SIZE) {
    super(glm);
    this.nKv = nKv;
    this.hd = hd;
    this.nLayers = nLayers;
    this.maxPages = maxPages;
    this.maxBatch = maxBatch;
    this.pageSize = pageSize;
    this.kData = [];
    this.vData = [];
    for (let i = 0; i < nLayers; i++) {
      this.kData.push(this.alloc([maxPages * nKv * pageSize * hd * BF16], "U8"));
      this.vData.push(this.alloc([maxPages * nKv * pageSize * hd * BF16], "U8"));
    }
    this.indices = this.alloc([maxPages * I32], "I32", "indices");
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

  updateIndptr(ws: ExecutionWorkspace): void {
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

    this.glm.writePinned(ws.indptrH, Buffer.from(indptrBuf.buffer, indptrBuf.byteOffset, indptrBuf.byteLength));
    this.glm.writePinned(ws.lastPageLenH, Buffer.from(lastPageLenBuf.buffer, lastPageLenBuf.byteOffset, lastPageLenBuf.byteLength));
    this.indices.h2d(Buffer.from(indicesBuf.buffer, indicesBuf.byteOffset, indicesBuf.byteLength));
    ws.indptrD.h2d(Buffer.from(indptrBuf.buffer, indptrBuf.byteOffset, indptrBuf.byteLength));
    ws.lastPageLen.h2d(Buffer.from(lastPageLenBuf.buffer, lastPageLenBuf.byteOffset, lastPageLenBuf.byteLength));
  }
}
