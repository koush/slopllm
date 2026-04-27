import type { ChatCache, ChatModel } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { BATCH_FLOAT_WS_SIZE, BATCH_INT_WS_SIZE, BATCH_PINNED_INT_WS_SIZE, BF16, I32 } from "./glm_ops";
import { MemcpyKind } from "./tensor";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export const PAGE_SIZE = 16;
export const DECODE_PLAN_INFO_SIZE = 10;
export const PREFILL_PLAN_INFO_SIZE = 15;

export class ExecutionState {
  batchSize: number;
  totalTokens: number;
  seqLens: number[];
  decodeInput?: Tensor;
  readonly isDecode: boolean;
  readonly ws: ExecutionWorkspace;
  readonly cache: ChatCache;
  readonly qoIndptrHost?: Tensor;

  constructor(
    batchSize: number, totalTokens: number, seqLens: number[],
    isDecode: boolean, ws: ExecutionWorkspace, cache: ChatCache,
    qoIndptrHost?: Tensor,
  ) {
    this.batchSize = batchSize;
    this.totalTokens = totalTokens;
    this.seqLens = seqLens;
    this.isDecode = isDecode;
    this.ws = ws;
    this.cache = cache;
    this.qoIndptrHost = qoIndptrHost;
  }

  prepareInput(tokenIds: number[]|Tensor) {
    if (!this.isDecode)
      throw new Error("decodeInput should be null in prefill");

    if (tokenIds instanceof Tensor) {
      this.decodeInput = tokenIds;
    }
    else {
      const batchSize = tokenIds.length;
      this.ws.inputIdsBufH.withPinnedBuffer(buf => {
        for (let i = 0; i < batchSize; i++) {
          buf.writeInt32LE(tokenIds[i], i * I32);
        }
      });
      this.decodeInput = this.ws.inputIdsBufH;
    }
  }
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
  inputIdsBufH: Tensor;
  positionIds: Tensor;
  positionIdsH: Tensor;
  lastIdx: Tensor;
  qoIndptrD: Tensor;
  slotMapping: Tensor;
  slotMappingH: Tensor;
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
    this.positionIdsH = this.allocPinned([B], "I32", "positionIdsH");
    this.lastIdx = this.alloc([B], "I32", "lastIdx");
    this.inputIdsBuf = this.alloc([B * S], "I32", "inputIdsBuf");
    this.inputIdsBufH = this.allocPinned([B], "I32", "inputIdsBufH");
    this.qoIndptrD = this.alloc([B + 1], "I32", "qoIndptrD");
    this.slotMapping = this.alloc([B * S], "I32", "slotMapping");
    this.slotMappingH = this.allocPinned([B], "I32", "slotMappingH");
    this.indptrD = this.alloc([(B + 1) * I32], "I32", "indptrD");
    this.indptrH = this.allocPinned([(B + 1) * I32], "I32", "indptrH");
    this.lastPageLen = this.alloc([B * I32], "I32", "lastPageLen");
    this.lastPageLenH = this.allocPinned([B], "I32", "lastPageLenH");
  }

  forwardInput(state: ExecutionState): void {
    const pagedKV = state.cache.getPagedKV();
    const batchSize = state.batchSize;
    let decodeInput = state.decodeInput;

    pagedKV.indices.memcpy(pagedKV.indicesH, pagedKV.maxPages * I32, MemcpyKind.HostToDevice);
    this.indptrD.memcpy(this.indptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
    this.lastPageLen.memcpy(this.lastPageLenH, batchSize * I32, MemcpyKind.HostToDevice);

    if (state.isDecode) {
      if (!decodeInput) {
        throw new Error("decodeInput tensor is required for decode mode");
      }

      if (decodeInput.pinned) {
        this.inputIdsBuf.memcpy(decodeInput, batchSize * I32, MemcpyKind.HostToDevice);
      }
      this.slotMapping.memcpy(this.slotMappingH, batchSize * I32, MemcpyKind.HostToDevice);
      this.positionIds.memcpy(this.positionIdsH, batchSize * I32, MemcpyKind.HostToDevice);
    } else if (state.qoIndptrHost) {
      this.qoIndptrD.memcpy(state.qoIndptrHost, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
    }
  }

  decodeStep(state: ExecutionState): void {
    const pagedKV = state.cache.getPagedKV();
    const batchSize = state.batchSize;
    this.glm.decodeStep(
      this.positionIds, this.lastPageLen, this.slotMapping,
      this.indptrD, pagedKV.indices,
      pagedKV.pageSize, batchSize
    );
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

  kvCacheWrite(kRope: Tensor, vBuf: Tensor, state: ExecutionState, cacheIdx: number, nKv: number, hd: number): void {
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

  planDecode(model: ChatModel, batchSize: number, cache: ChatCache, enableCudaGraph = false): ExecutionState {
    const pagedKV = cache.getPagedKV();
    const cfg = model.cfg;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const seqLens = new Array(batchSize).fill(1) as number[];
    const totalTokens = batchSize;

    if (pagedKV.seqPages.length !== batchSize) {
      throw new Error(`planDecode: pagedKV has ${pagedKV.seqPages.length} sequences, expected ${batchSize}`);
    }

    const writeLocations: [number, number][] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      writeLocations.push(pagedKV.allocDecodeToken(seqIdx));
    }

    pagedKV.updateIndptr(this);

    this.slotMappingH.withPinnedBuffer(buf => {
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        const [absPage, slotInPage] = writeLocations[seqIdx];
        buf.writeInt32LE(absPage * pageSize + slotInPage, seqIdx * I32);
      }
    });

    this.positionIdsH.withPinnedBuffer(buf => {
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        buf.writeInt32LE(pagedKV.seqKvLens[seqIdx] - 1, seqIdx * I32);
      }
    });

    this.glm.batchDecodePlan(
      this.floatWs, BATCH_FLOAT_WS_SIZE,
      this.intWs, this.pinnedIntWs, BATCH_INT_WS_SIZE,
      this.decodePlanInfo,
      this.indptrH,
      batchSize,
      nHeads, nKv, hd, pageSize,
      enableCudaGraph
    );

    return new ExecutionState(batchSize, totalTokens, seqLens, true, this, cache);
  }

  planPrefill(model: ChatModel, inputIdsList: number[][], cache: ChatCache): ExecutionState {
    const pagedKV = cache.getPagedKV();
    const cfg = model.cfg;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const batchSize = inputIdsList.length;
    const seqLens = inputIdsList.map(ids => ids.length);
    const totalTokens = seqLens.reduce((a, b) => a + b, 0);

    if (pagedKV.seqPages.length !== batchSize) {
      throw new Error(`planPrefill: pagedKV has ${pagedKV.seqPages.length} sequences, expected ${batchSize}`);
    }

    const inputIdsBuf = Buffer.alloc(totalTokens * I32);
    let idsOff = 0;
    for (const ids of inputIdsList) {
      for (const id of ids) {
        inputIdsBuf.writeInt32LE(id, idsOff);
        idsOff += I32;
      }
    }
    this.inputIdsBuf.h2d(inputIdsBuf);

    const startPos = pagedKV.seqKvLens.slice();

    model.prefillBatchPlanHook(inputIdsList, seqLens, totalTokens, startPos, cache);

    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pagedKV.allocAppendPages(seqIdx, seqLens[seqIdx]);
    }

    const qoIndptrHost = this.allocPinned([(batchSize + 1)], "I32");
    qoIndptrHost.withPinnedBuffer(buf => {
      buf.writeInt32LE(0, 0);
      for (let i = 0; i < batchSize; i++) {
        buf.writeInt32LE(buf.readInt32LE(i * I32) + seqLens[i], (i + 1) * I32);
      }
    });

    pagedKV.updateIndptr(this);

    const positionIdsBuf = Buffer.alloc(totalTokens * I32);
    let posOff = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      for (let p = 0; p < seqLens[seqIdx]; p++) {
        positionIdsBuf.writeInt32LE(startPos[seqIdx] + p, posOff * I32);
        posOff++;
      }
    }
    this.positionIds.h2d(positionIdsBuf);

    const lastIdxBuf = Buffer.alloc(batchSize * I32);
    let lastOff = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      lastIdxBuf.writeInt32LE(lastOff + seqLens[seqIdx] - 1, seqIdx * I32);
      lastOff += seqLens[seqIdx];
    }
    this.lastIdx.h2d(lastIdxBuf);

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

    const slotMappingBuf = Buffer.alloc(totalTokens * I32);
    let slotOff = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      const pages = pagedKV.seqPages[seqIdx];
      for (let pos = 0; pos < seqLens[seqIdx]; pos++) {
        const kvPos = startPos[seqIdx] + pos;
        const pageIdxInSeq = Math.floor(kvPos / pagedKV.pageSize);
        const offsetInPage = kvPos % pagedKV.pageSize;
        const absPage = pages[pageIdxInSeq];
        slotMappingBuf.writeInt32LE(absPage * pagedKV.pageSize + offsetInPage, slotOff * I32);
        slotOff++;
      }
    }
    this.slotMapping.h2d(slotMappingBuf);

    return new ExecutionState(batchSize, totalTokens, seqLens, false, this, cache, qoIndptrHost);
  }

  forwardPrefill(model: ChatModel, inputIdsList: number[][], cache: ChatCache): Tensor {
    const state = this.planPrefill(model, inputIdsList, cache);
    this.forwardInput(state);
    return model.forward(state);
  }

  forwardEagerPrefill(model: ChatModel, inputIdsList: number[][], cache: ChatCache): number[] {
    const logits = this.forwardPrefill(model, inputIdsList, cache);
    using argmaxResult = logits.argmax();
    return argmaxResult.readInt32LE();
  }

  forwardDecode(model: ChatModel, state: ExecutionState): Tensor {
    this.forwardInput(state);
    return model.forward(state);
  }

  forwardEagerDecode(model: ChatModel, tokenIdsList: number[], cache: ChatCache): number[] {
    const state = this.planDecode(model, tokenIdsList.length, cache);
    state.prepareInput(tokenIdsList);
    this.forwardInput(state);
    const logits = model.forward(state);
    using argmaxResult = logits.argmax();
    return argmaxResult.readInt32LE();
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
  indicesH: Tensor;
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
    this.indicesH = this.allocPinned([maxPages], "I32", "indicesH");
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
