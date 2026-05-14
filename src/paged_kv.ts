import type { ChatCache, ChatModel } from "./chat_model";
import { DeviceOps, TensorParallelism } from "./device_ops";
import { BATCH_FLOAT_WS_SIZE, BATCH_INT_WS_SIZE, BATCH_PINNED_INT_WS_SIZE, I32 } from "./glm_ops";
import { MemcpyKind } from "./tensor";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export const PAGE_SIZE = 16;
export const DECODE_PLAN_INFO_SIZE = 10;
export const PREFILL_PLAN_INFO_SIZE = 15;
export const MLA_PREFILL_PLAN_INFO_SIZE = 18;
export const MLA_DECODE_PLAN_INFO_SIZE = 10;

export class ExecutionState {
  batchSize: number;
  totalTokens: number;
  seqLens: number[];
  input?: Tensor;
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

  kvCacheWrite(kRope: Tensor, vBuf: Tensor, cacheIdx: number, nKv: number, hd: number): void {
    const pagedKV = this.cache.getPagedKV();
    const BS = this.totalTokens;
    const kTokenStride = this.isDecode ? nKv * hd : hd;
    const kHeadStride = this.isDecode ? hd : BS * hd;
    const vTokenStride = nKv * hd;
    const vHeadStride = hd;
    this.ws.glm.kvCacheWrite(
      kRope, vBuf,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
      this.ws.slotMapping,
      BS, nKv, hd, pagedKV.pageSize,
      kTokenStride, kHeadStride, vTokenStride, vHeadStride
    );
  }

  mlaKvCacheAppend(appendCkv: Tensor, appendKpe: Tensor, cacheIdx: number, kvLoraRank: number, qkRopeDim: number): void {
    const pagedKV = this.cache.getPagedKV();
    const headDimCkv = kvLoraRank;
    const headDimKpe = qkRopeDim;
    const pageSize = pagedKV.pageSize;
    const nnz = this.isDecode ? this.batchSize : pagedKV.seqKvLens.reduce((a, b) => a + b, 0);
    const appendCkvStrideN = headDimCkv;
    const appendKpeStrideN = headDimKpe;
    this.ws.glm.mlaKvCacheAppend(
      pagedKV.ckvData[cacheIdx], pagedKV.kpeData[cacheIdx],
      pagedKV.indices, this.ws.indptrD, this.ws.lastPageLen,
      appendCkv, appendKpe,
      this.ws.mlaBatchIndices, this.ws.positionIds,
      nnz, pageSize, headDimCkv, headDimKpe,
      appendCkvStrideN, appendKpeStrideN,
      pagedKV.contextParallel,
    );
  }

  prepareInput(tokenIds: number[][]|Tensor) {
    if (tokenIds instanceof Tensor) {
      this.input = tokenIds;
    }
    else {
      this.ws.inputIdsBufH.withPinnedBuffer(buf => {
        let idsOff = 0;
        for (const ids of tokenIds) {
          for (const id of ids) {
            buf.writeInt32LE(id, idsOff);
            idsOff += I32;
          }
        }
      });
      this.input = this.ws.inputIdsBufH;
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
  /** GPU float workspace: written by FlashInfer plan, read by FlashInfer run. */
  floatWs: Tensor;
  /** GPU int workspace: written by FlashInfer plan, read by FlashInfer run. */
  intWs: Tensor;
  /** Pinned host int workspace: scratch space used internally by FlashInfer plan (read+write within plan call). */
  pinnedIntWs: Tensor;
  /** Pinned host buffer: written by batchDecodePlan, read by batchDecodeRun. */
  decodePlanInfo: Tensor;
  /** Pinned host buffer: written by batchPrefillPagedPlan, read by batchPrefillPagedRun. */
  prefillPlanInfo: Tensor;
  /** Pinned host buffer: written by mlaPrefillPlan, read by mlaPrefillRun. */
  mlaPrefillPlanInfo: Tensor;
  /** Pinned host buffer: written by mlaDecodePlan, read by mlaDecodeRun. */
  mlaDecodePlanInfo: Tensor;
  /** GPU buffer [B*S] of I32: written by host (h2d), read by embedding lookup. */
  inputIdsBuf: Tensor;
  /** Pinned host buffer [B] of I32: written by host, read via memcpy to inputIdsBuf. */
  inputIdsBufH: Tensor;
  /** GPU buffer [B*S] of I32: written by host (h2d), read by RoPE kernel. */
  positionIds: Tensor;
  /** Pinned host buffer [B*S] of I32: written by host, read via memcpy to positionIds. */
  positionIdsH: Tensor;
  /** GPU buffer [B] of I32: written by host (h2d), read to extract last-token logits per sequence (prefill). */
  lastIdx: Tensor;
  /** Pinned host buffer [B] of I32: written by host, read via memcpy to lastIdx. */
  lastIdxH: Tensor;
  /** GPU buffer [B+1] of I32: written by host via memcpy, read by FlashInfer prefill run. */
  qoIndptrD: Tensor;
  /** Pinned host buffer [B+1] of I32: written by host, read by MLA prefill plan and memcpy to qoIndptrD. */
  qoIndptrH: Tensor;
  /** GPU buffer [B*S] of I32: written by host (h2d) or memcpy, read by kvCacheWrite to scatter K/V into cache. */
  slotMapping: Tensor;
  /** Pinned host buffer [B*S] of I32: written by host, read via memcpy to slotMapping. */
  slotMappingH: Tensor;
  /** GPU buffer [B+1] of I32: written by host via memcpy, read by FlashInfer run (page indptr). */
  indptrD: Tensor;
  /** Pinned host buffer [B+1] of I32: written by updateIndptr, read by memcpy to indptrD and by FlashInfer plan. */
  indptrH: Tensor;
  /** GPU buffer [B] of I32: written by host via memcpy, read by FlashInfer run. */
  lastPageLen: Tensor;
  /** Pinned host buffer [B] of I32: written by updateIndptr, read via memcpy to lastPageLen. */
  lastPageLenH: Tensor;
  /** Pinned host buffer [B] of I32: KV lengths per batch entry, used by MLA prefill plan. */
  kvLenH: Tensor;
  /** GPU buffer [B*S] of I32: batch index per token for MLA KV cache append. */
  mlaBatchIndices: Tensor;
  /** Pinned host buffer [B*S] of I32: batch index per token for MLA KV cache append. */
  mlaBatchIndicesH: Tensor;

  constructor(glm: DeviceOps, B: number, S: number) {
    super(glm);

    this.floatWs = this.alloc([BATCH_FLOAT_WS_SIZE], "U8", "floatWs");
    this.intWs = this.alloc([BATCH_INT_WS_SIZE], "U8", "intWs");
    this.pinnedIntWs = this.allocPinned([BATCH_PINNED_INT_WS_SIZE], "U8", "pinnedIntWs");
    this.decodePlanInfo = this.allocPinned([DECODE_PLAN_INFO_SIZE * 8], "U8", "decodePlanInfo");
    this.prefillPlanInfo = this.allocPinned([PREFILL_PLAN_INFO_SIZE * 8], "U8", "prefillPlanInfo");
    this.mlaPrefillPlanInfo = this.allocPinned([MLA_PREFILL_PLAN_INFO_SIZE * 8], "U8", "mlaPrefillPlanInfo");
    this.mlaDecodePlanInfo = this.allocPinned([MLA_DECODE_PLAN_INFO_SIZE * 8], "U8", "mlaDecodePlanInfo");

    this.positionIds = this.alloc([B * S], "I32", "positionIds");
    this.positionIdsH = this.allocPinned([B * S], "I32", "positionIdsH");
    this.lastIdx = this.alloc([B], "I32", "lastIdx");
    this.lastIdxH = this.allocPinned([B], "I32", "lastIdxH");
    this.inputIdsBuf = this.alloc([B * S], "I32", "inputIdsBuf");
    this.inputIdsBufH = this.allocPinned([B * S], "I32", "inputIdsBufH");
    this.qoIndptrD = this.alloc([B + 1], "I32", "qoIndptrD");
    this.qoIndptrH = this.allocPinned([B + 1], "I32", "qoIndptrH");
    this.slotMapping = this.alloc([B * S], "I32", "slotMapping");
    this.slotMappingH = this.allocPinned([B * S], "I32", "slotMappingH");
    this.indptrD = this.alloc([(B + 1) * I32], "I32", "indptrD");
    this.indptrH = this.allocPinned([(B + 1) * I32], "I32", "indptrH");
    this.lastPageLen = this.alloc([B * I32], "I32", "lastPageLen");
    this.lastPageLenH = this.allocPinned([B], "I32", "lastPageLenH");
    this.kvLenH = this.allocPinned([B], "I32", "kvLenH");
    this.mlaBatchIndices = this.alloc([B * S], "I32", "mlaBatchIndices");
    this.mlaBatchIndicesH = this.allocPinned([B * S], "I32", "mlaBatchIndicesH");

    // Initialize mlaBatchIndices for decode: [0, 1, 2, ..., B-1]
    // This identity mapping never changes for decode; prefill overwrites it with
    // per-token batch indices in planPrefill.
    this.mlaBatchIndicesH.withPinnedBuffer(buf => {
      for (let i = 0; i < B; i++) buf.writeInt32LE(i, i * I32);
    });
    this.mlaBatchIndices.memcpy(this.mlaBatchIndicesH, B * I32, MemcpyKind.HostToDevice);
  }

  forwardInput(state: ExecutionState): void {
    const pagedKV = state.cache.getPagedKV();
    const batchSize = state.batchSize;
    let input = state.input;

    if (!input) {
      throw new Error("input tensor is required");
    }

    if (input.pinned) {
      const count = state.isDecode ? batchSize : state.totalTokens;
      this.inputIdsBuf.memcpy(input, count * I32, MemcpyKind.HostToDevice);
    }
  }

  decodeStep(state: ExecutionState, model: ChatModel): void {
    const pagedKV = state.cache.getPagedKV();
    const batchSize = state.batchSize;
    if (!model.cfg.kvLoraRank) {
      this.glm.decodeStep(
        this.positionIds, this.lastPageLen, this.slotMapping,
        this.indptrD, pagedKV.indices,
        pagedKV.pageSize, batchSize
      );
    } else {
      this.glm.mlaDecodeStep(
        this.positionIds, this.lastPageLen,
        this.indptrD,
        pagedKV.pageSize, batchSize,
        pagedKV.contextParallel
      );
    }
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

  mlaPrefillPaged(qNope: Tensor, qPe: Tensor, pagedKV: PagedKVCache, cacheIdx: number, totalTokens: number, batchSize: number, nHeads: number, kvLoraRank: number, qkRopeDim: number, smScale: number, contextParallel?: boolean): { o: Tensor, lse: Tensor } {
    const headDimCkv = kvLoraRank;
    const headDimKpe = qkRopeDim;
    const pageSize = pagedKV.pageSize;
    const qNopeStrideN = nHeads * headDimCkv;
    const qNopeStrideH = headDimCkv;
    const qPeStrideN = nHeads * headDimKpe;
    const qPeStrideH = headDimKpe;
    const ckvStridePage = pageSize * headDimCkv;
    const ckvStrideN = headDimCkv;
    const kpeStridePage = pageSize * headDimKpe;
    const kpeStrideN = headDimKpe;
    const oStrideN = headDimCkv;
    const oStrideH = totalTokens * headDimCkv;
    return this.glm.mlaPrefillRun(
      qNope, qPe, pagedKV.ckvData[cacheIdx], pagedKV.kpeData[cacheIdx],
      pagedKV.indices,
      this.floatWs, this.intWs,
      this.mlaPrefillPlanInfo,
      nHeads, pageSize, 1, smScale,
      qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH,
      ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN,
      oStrideN, oStrideH,
      headDimCkv, headDimKpe,
      contextParallel
    );
  }

  mlaDecodePaged(qNope: Tensor, qPe: Tensor, pagedKV: PagedKVCache, cacheIdx: number, batchSize: number, nHeads: number, kvLoraRank: number, qkRopeDim: number, smScale: number, contextParallel?: boolean): { o: Tensor, lse: Tensor } {
    const headDimCkv = kvLoraRank;
    const headDimKpe = qkRopeDim;
    return this.glm.mlaDecodeRun(
      qNope, qPe, pagedKV.ckvData[cacheIdx], pagedKV.kpeData[cacheIdx],
      pagedKV.indices, this.indptrD, this.lastPageLen,
      this.floatWs, this.intWs,
      this.mlaDecodePlanInfo,
      batchSize, nHeads, pagedKV.pageSize, smScale,
      headDimCkv, headDimKpe,
      contextParallel
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

    let decodePagesNeeded = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      decodePagesNeeded += pagedKV.pagesNeededForDecodeToken(seqIdx);
    }
    if (decodePagesNeeded > pagedKV.availablePages.length) {
      throw new Error(`planDecode: need ${decodePagesNeeded} pages, ${pagedKV.availablePages.length} available`);
    }

    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pagedKV.allocDecodeToken(seqIdx);
    }

    if (pagedKV.pagesDirtyHost) {
      pagedKV.updateIndptr(this);

      if (!cfg.kvLoraRank) {
        this.glm.batchDecodePlan(
          this.floatWs, BATCH_FLOAT_WS_SIZE,
          this.intWs, this.pinnedIntWs, BATCH_INT_WS_SIZE,
          this.decodePlanInfo,
          this.indptrH,
          batchSize,
          nHeads, nKv, hd, pageSize,
          enableCudaGraph
        );
      } else {
        this.glm.mlaDecodePlan(
          this.floatWs, 128 * 1024 * 1024,
          this.intWs, this.pinnedIntWs, 8 * 1024 * 1024,
          this.mlaDecodePlanInfo,
          this.indptrH, this.lastPageLenH,
          batchSize, model.cfg.numAttentionHeads, pagedKV.pageSize, enableCudaGraph,
          model.cfg.kvLoraRank!, model.cfg.qkRopeHeadDim!, pagedKV.contextParallel,
          undefined, undefined, pagedKV.seqKvLens
        );
      }
      pagedKV.pagesDirtyHost = false;
    }

    if (pagedKV.pagesDirtyDevice) {
      const usedPages = pagedKV.seqPages.reduce((sum, sp) => sum + sp.length, 0);
      pagedKV.indices.memcpy(pagedKV.indicesH, usedPages * I32, MemcpyKind.HostToDevice);
      this.indptrD.memcpy(this.indptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
      pagedKV.pagesDirtyDevice = false;
    }

    return new ExecutionState(batchSize, totalTokens, seqLens, true, this, cache);
  }

  planPrefill(model: ChatModel, batchSize: number, seqLens: number[], cache: ChatCache): ExecutionState {
    const pagedKV = cache.getPagedKV();
    const cfg = model.cfg;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const totalTokens = seqLens.reduce((a, b) => a + b, 0);

    if (pagedKV.seqPages.length !== batchSize) {
      throw new Error(`planPrefill: pagedKV has ${pagedKV.seqPages.length} sequences, expected ${batchSize}`);
    }

    const startPos = pagedKV.seqKvLens.slice();

    model.prefillBatchPlanHook(batchSize, seqLens, totalTokens, startPos, cache);

    let prefillPagesNeeded = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      prefillPagesNeeded += pagedKV.pagesNeededForAppend(seqIdx, seqLens[seqIdx]);
    }
    if (prefillPagesNeeded > pagedKV.availablePages.length) {
      throw new Error(`planPrefill: need ${prefillPagesNeeded} pages, ${pagedKV.availablePages.length} available`);
    }

    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pagedKV.allocAppendPages(seqIdx, seqLens[seqIdx]);
    }

    this.qoIndptrH.withPinnedBuffer(buf => {
      buf.writeInt32LE(0, 0);
      for (let i = 0; i < batchSize; i++) {
        buf.writeInt32LE(buf.readInt32LE(i * I32) + seqLens[i], (i + 1) * I32);
      }
    });

    pagedKV.updateIndptr(this);

    this.positionIdsH.withPinnedBuffer(buf => {
      let posOff = 0;
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        for (let p = 0; p < seqLens[seqIdx]; p++) {
          buf.writeInt32LE(startPos[seqIdx] + p, posOff * I32);
          posOff++;
        }
      }
    });
    this.positionIds.memcpy(this.positionIdsH, totalTokens * I32, MemcpyKind.HostToDevice);

    this.kvLenH.withPinnedBuffer(buf => {
      for (let i = 0; i < batchSize; i++) {
        buf.writeInt32LE(seqLens[i], i * I32);
      }
    });

    this.lastIdxH.withPinnedBuffer(buf => {
      let lastOff = 0;
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        buf.writeInt32LE(lastOff + seqLens[seqIdx] - 1, seqIdx * I32);
        lastOff += seqLens[seqIdx];
      }
    });
    this.lastIdx.memcpy(this.lastIdxH, batchSize * I32, MemcpyKind.HostToDevice);

    if (cfg.kvLoraRank) {
      this.glm.mlaPrefillPlan(
        this.floatWs, 128 * 1024 * 1024,
        this.intWs, this.pinnedIntWs, 8 * 1024 * 1024,
        this.mlaPrefillPlanInfo,
        this.qoIndptrH, this.indptrH,
        this.kvLenH, this.lastPageLenH,
        batchSize, nHeads, cfg.kvLoraRank!, true,
        pagedKV.pageSize, pagedKV.seqKvLens,
        pagedKV.contextParallel
      );
      this.mlaBatchIndicesH.withPinnedBuffer(buf => {
        let off = 0;
        for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
          for (let pos = 0; pos < seqLens[seqIdx]; pos++) {
            buf.writeInt32LE(seqIdx, off * I32);
            off++;
          }
        }
      });
      this.mlaBatchIndices.memcpy(this.mlaBatchIndicesH, totalTokens * I32, MemcpyKind.HostToDevice);
    } else {
      this.glm.batchPrefillPagedPlan(
        this.floatWs, BATCH_FLOAT_WS_SIZE,
        this.intWs, this.pinnedIntWs, BATCH_INT_WS_SIZE,
        this.prefillPlanInfo,
        this.qoIndptrH, this.indptrH,
        totalTokens, batchSize,
        nHeads, nKv, hd,
        pageSize,
        1
      );
      this.slotMappingH.withPinnedBuffer(buf => {
        let slotOff = 0;
        for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
          const pages = pagedKV.seqPages[seqIdx];
          for (let pos = 0; pos < seqLens[seqIdx]; pos++) {
            const kvPos = startPos[seqIdx] + pos;
            const pageIdxInSeq = Math.floor(kvPos / pagedKV.pageSize);
            const offsetInPage = kvPos % pagedKV.pageSize;
            const absPage = pages[pageIdxInSeq];
            buf.writeInt32LE(absPage * pagedKV.pageSize + offsetInPage, slotOff * I32);
            slotOff++;
          }
        }
      });
      this.slotMapping.memcpy(this.slotMappingH, totalTokens * I32, MemcpyKind.HostToDevice);
      this.qoIndptrD.memcpy(this.qoIndptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
    }

    const usedPages = pagedKV.seqPages.reduce((sum, sp) => sum + sp.length, 0);
    pagedKV.indices.memcpy(pagedKV.indicesH, usedPages * I32, MemcpyKind.HostToDevice);
    this.indptrD.memcpy(this.indptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
    this.lastPageLen.memcpy(this.lastPageLenH, batchSize * I32, MemcpyKind.HostToDevice);

    return new ExecutionState(batchSize, totalTokens, seqLens, false, this, cache, this.qoIndptrH);
  }

  forwardPrefill(model: ChatModel, inputIdsList: number[][], cache: ChatCache): Tensor {
    const batchSize = inputIdsList.length;
    const seqLens = inputIdsList.map(ids => ids.length);
    const state = this.planPrefill(model, batchSize, seqLens, cache);
    state.prepareInput(inputIdsList);
    this.forwardInput(state);
    const logits = model.forward(state);

    const pagedKV = state.cache.getPagedKV();
    this.positionIdsH.withPinnedBuffer(buf => {
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        buf.writeInt32LE(pagedKV.seqKvLens[seqIdx] - 1, seqIdx * I32);
      }
    });
    this.positionIds.memcpy(this.positionIdsH, batchSize * I32, MemcpyKind.HostToDevice);

    return logits;
  }

  forwardEagerPrefill(model: ChatModel, inputIdsList: number[][], cache: ChatCache): number[] {
    const logits = this.forwardPrefill(model, inputIdsList, cache);
    using argmaxResult = logits.argmax();
    return argmaxResult.readInt32LEArray();
  }

  forwardDecode(model: ChatModel, state: ExecutionState): Tensor {
    this.forwardInput(state);
    return model.forward(state);
  }

  forwardEagerDecode(model: ChatModel, tokenIdsList: number[], cache: ChatCache): number[] {
    const state = this.planDecode(model, tokenIdsList.length, cache);
    state.prepareInput([tokenIdsList]);
    this.decodeStep(state, model);
    this.forwardInput(state);
    const logits = model.forward(state);
    using argmaxResult = logits.argmax();
    return argmaxResult.readInt32LEArray();
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
    this.availablePages = Array.from({length: maxPages}, (_, i) => i);
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
    this.availablePages = Array.from({length: this.maxPages}, (_, i) => i);
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
