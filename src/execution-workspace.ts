import { ChatModel, type ChatCache } from "./chat_model";
import { DeviceOps, MaskMode } from "./device_ops";
import { I32 } from "./glm_ops";
import { type PagedKVCache } from "./paged_kv";
import { MemcpyKind, Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export const DECODE_PLAN_INFO_SIZE = 10;
export const PREFILL_PLAN_INFO_SIZE = 15;
export const MLA_PREFILL_PLAN_INFO_SIZE = 19;
export const MLA_DECODE_PLAN_INFO_SIZE = 10;
export const BATCH_FLOAT_WS_SIZE = 128 * 1024 * 1024;
export const BATCH_INT_WS_SIZE = 8 * 1024 * 1024;
export const BATCH_PINNED_INT_WS_SIZE = 8 * 1024 * 1024;

export class ExecutionState {
  input?: Tensor;

  constructor(
    public readonly batchSize: number, public readonly totalTokens: number, public readonly seqLens: number[],
    public readonly isDecode: boolean, public readonly ws: ExecutionWorkspace, public readonly cache: ChatCache,
    public readonly qoIndptrHost?: Tensor, public readonly customMask?: {
      indptr: Tensor;
      mask: Tensor;
      mode?: MaskMode;
      positionIds?: Tensor;
    },
  ) {
  }

  get lastIdx(): Tensor {
    return this.ws.qoIndptrD.narrow(1, this.batchSize);
  }

  computeLogits(hiddenStates: Tensor, model: ChatModel, allTokens: boolean = false): Tensor {
    const lmHead = model.tensors.get("lm_head.weight")!;
    const batchSize = this.batchSize;
    if (this.isDecode) {
      return hiddenStates.linear(lmHead, batchSize).removeTracking();
    }
    else if (allTokens) {
      return hiddenStates.linear(lmHead, this.totalTokens).removeTracking();
    }
    else {
      using lastIdxFromIndptr = this.lastIdx;
      using hiddenLast = hiddenStates.indexSelect(lastIdxFromIndptr, this.batchSize, -1);
      return hiddenLast.linear(lmHead, batchSize).removeTracking();
    }
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
    const nnz = this.isDecode ? this.batchSize : this.totalTokens;
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

  setInput(tokenIds: number[][] | Tensor) {
    if (tokenIds instanceof Tensor) {
      // if the tensor is not in the same workspace copy it into the workspace buffer.
      if (tokenIds.workspace !== this.ws) {
        this.input = this.ws.inputIdsBuf;
        this.input.memcpy(tokenIds, tokenIds.bytes, MemcpyKind.DeviceToDevice);
      }
      else {
        this.input = tokenIds;
      }
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
      this.input = this.ws.inputIdsBuf;
      this.input.memcpy(this.ws.inputIdsBufH, this.input.bytes, MemcpyKind.HostToDevice);
    }
  }
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
  /** Scratch GPU buffer [B*S] of I32: written by host (h2d), read by embedding lookup. */
  inputIdsBuf: Tensor;
  /** Scratch Pinned host buffer [B*S] of I32: written by host, read via memcpy to inputIdsBuf. */
  inputIdsBufH: Tensor;
  /** Whether the input buffer has been cleared. */
  inputCleared = false;
  /** GPU buffer [B*S] of I32: written by host (h2d), read by RoPE kernel. */
  positionIds: Tensor;
  /** Pinned host buffer [B*S] of I32: written by host, read via memcpy to positionIds. */
  positionIdsH: Tensor;
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
  lastDecodePagedKV: PagedKVCache | null;
  private tracking: Disposable & { [Symbol.dispose](): void } | null = null;

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
    this.lastDecodePagedKV = null;

    // Initialize mlaBatchIndices for decode: [0, 1, 2, ..., B-1]
    // This identity mapping never changes for decode; prefill overwrites it with
    // per-token batch indices in planPrefill.
    this.mlaBatchIndicesH.withPinnedBuffer(buf => {
      for (let i = 0; i < B; i++) buf.writeInt32LE(i, i * I32);
    });
    this.mlaBatchIndices.memcpy(this.mlaBatchIndicesH, B * I32, MemcpyKind.HostToDevice);
  }

  ensureInputCleared() {
    if (this.inputCleared) {
      return;
    }
    this.inputIdsBuf.fill(0, this.inputIdsBuf.numElements);
    this.inputCleared = true;
  }

  startTracking(keepExports = new Set<Tensor>()): Disposable & { [Symbol.dispose](): void } {
    if (this.tracking !== null) {
      throw new Error("startTracking already active");
    }
    if (this.tracked.size) {
      console.warn("startTracking was called with tensors already allocated, this may result in non-deterministic allocations."); 
    }
    for (const tensor of this.exported) {
      if (!keepExports.has(tensor)) {
        this.exported.delete(tensor);
        tensor[Symbol.dispose]();
      }
    }
    const ws = this;
    const tracker: Disposable & { [Symbol.dispose](): void } = {
      [Symbol.dispose]() {
        for (const tensor of ws.tracked) {
          tensor[Symbol.dispose]();
        }
        ws.tracked.clear();
        ws.tracking = null;
      },
    };
    this.tracking = tracker;
    return tracker;
  }

  positionStep(state: ExecutionState, model: ChatModel, steps = 1): void {
    const pagedKV = state.cache.getPagedKV();
    const batchSize = state.batchSize;
    if (!model.cfg.kvLoraRank) {
      this.glm.positionStep(
        this.positionIds, this.lastPageLen, this.slotMapping,
        this.indptrD, pagedKV.indices,
        pagedKV.pageSize, batchSize, steps
      );
    } else {
      this.glm.mlaPositionStep(
        this.positionIds, this.lastPageLen,
        this.indptrD,
        pagedKV.pageSize, batchSize,
        pagedKV.contextParallel,
        undefined, undefined, steps
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

  flashPrefillPaged(query: Tensor, pagedKV: PagedKVCache, cacheIdx: number, totalTokens: number, batchSize: number, nHeads: number, nKv: number, hd: number, qStrideN: number, qStrideH: number, maskMode: MaskMode, smScale: number): Tensor {
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

  mlaPrefillPaged(qNope: Tensor, qPe: Tensor, pagedKV: PagedKVCache, cacheIdx: number, totalTokens: number, batchSize: number, nHeads: number, kvLoraRank: number, qkRopeDim: number, smScale: number, contextParallel?: boolean, maskMode: MaskMode = MaskMode.Causal, customMask?: Tensor, maskIndptr?: Tensor): { o: Tensor, lse: Tensor } {
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
      nHeads, pageSize, maskMode, smScale,
      qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH,
      ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN,
      oStrideN, oStrideH,
      headDimCkv, headDimKpe,
      contextParallel, undefined, undefined,
      customMask, maskIndptr
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
    pagedKV.checkSequenceCount();
    const cfg = model.cfg;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const seqLens = new Array(batchSize).fill(1) as number[];
    const totalTokens = batchSize;

    if (pagedKV.sequences.length !== batchSize) {
      throw new Error(`planDecode: pagedKV has ${pagedKV.sequences.length} sequences, expected ${batchSize}`);
    }

    let decodePagesNeeded = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      decodePagesNeeded += pagedKV.pagesNeededForDecodeToken(seqIdx);
    }
    if (decodePagesNeeded > pagedKV.availablePages.length) {
      throw new Error(`planDecode: need ${decodePagesNeeded} pages, ${pagedKV.availablePages.length} available`);
    }

    if (pagedKV.positionIdsDirty || this.lastDecodePagedKV !== pagedKV) {
      this.positionIdsH.withPinnedBuffer(buf => {
        for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
          buf.writeInt32LE(pagedKV.sequences[seqIdx].allocLen - 1, seqIdx * I32);
        }
      });
      this.positionIds.memcpy(this.positionIdsH, batchSize * I32, MemcpyKind.HostToDevice);

      if (cfg.kvLoraRank) {
        this.mlaBatchIndicesH.withPinnedBuffer(buf => {
          for (let i = 0; i < batchSize; i++) buf.writeInt32LE(i, i * I32);
        });
        this.mlaBatchIndices.memcpy(this.mlaBatchIndicesH, batchSize * I32, MemcpyKind.HostToDevice);
      }

      pagedKV.positionIdsDirty = false;
      this.lastDecodePagedKV = pagedKV;
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
          this.floatWs, BATCH_FLOAT_WS_SIZE,
          this.intWs, this.pinnedIntWs, BATCH_INT_WS_SIZE,
          this.mlaDecodePlanInfo,
          this.indptrH, this.lastPageLenH,
          batchSize, model.cfg.numAttentionHeads, pagedKV.pageSize, enableCudaGraph,
          model.cfg.kvLoraRank!, model.cfg.qkRopeHeadDim!, pagedKV.contextParallel,
          undefined, undefined, pagedKV.sequences.map(s => s.allocLen)
        );
      }
      pagedKV.pagesDirtyHost = false;
      pagedKV.pagesDirtyDevice = true;
    }

    // could be rolled into above? keeping an explicit flag here since
    // there may be a case where only device needs update or device update is done later in graph?
    if (pagedKV.pagesDirtyDevice) {
      const usedPages = pagedKV.sequences.reduce((sum, s) => sum + s.contentPages, 0);
      pagedKV.indices.memcpy(pagedKV.indicesH, usedPages * I32, MemcpyKind.HostToDevice);
      this.indptrD.memcpy(this.indptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
      pagedKV.pagesDirtyDevice = false;
    }

    return new ExecutionState(batchSize, totalTokens, seqLens, true, this, cache);
  }

  planPrefill(model: ChatModel, batchSize: number, seqLens: number[], cache: ChatCache, customMask?: {
    indptr: Tensor;
    mask: Tensor;
    mode?: MaskMode;
    positionIds?: Tensor;
  }): ExecutionState {
    const pagedKV = cache.getPagedKV();
    pagedKV.checkSequenceCount();
    const cfg = model.cfg;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const totalTokens = seqLens.reduce((a, b) => a + b, 0);

    if (pagedKV.sequences.length !== batchSize) {
      throw new Error(`planPrefill: pagedKV has ${pagedKV.sequences.length} sequences, expected ${batchSize}`);
    }

    const startPos = pagedKV.sequences.map(s => s.allocLen);

    cache.prefillBatchPlanHook?.(batchSize, seqLens, totalTokens, startPos, cache);

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
    pagedKV.positionIdsDirty = true;

    this.kvLenH.withPinnedBuffer(buf => {
      for (let i = 0; i < batchSize; i++) {
        buf.writeInt32LE(pagedKV.sequences[i].allocLen, i * I32);
      }
    });

    if (cfg.kvLoraRank) {
      this.glm.mlaPrefillPlan(
        this.floatWs, BATCH_FLOAT_WS_SIZE,
        this.intWs, this.pinnedIntWs, BATCH_INT_WS_SIZE,
        this.mlaPrefillPlanInfo,
        this.qoIndptrH, this.indptrH,
        this.kvLenH, this.lastPageLenH,
        batchSize, nHeads, cfg.kvLoraRank!, !customMask || customMask.mode === MaskMode.CausalCustom  || customMask.mode === MaskMode.Causal,
        pagedKV.pageSize, pagedKV.sequences.map(s => s.allocLen),
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
      this.qoIndptrD.memcpy(this.qoIndptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
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
          const pages = pagedKV.sequences[seqIdx].pages;
          for (let pos = 0; pos < seqLens[seqIdx]; pos++) {
            const kvPos = startPos[seqIdx] + pos;
            const pageIdxInSeq = Math.floor(kvPos / pagedKV.pageSize);
            const offsetInPage = kvPos % pagedKV.pageSize;
            const absPage = pages[pageIdxInSeq].id;
            buf.writeInt32LE(absPage * pagedKV.pageSize + offsetInPage, slotOff * I32);
            slotOff++;
          }
        }
      });
      this.slotMapping.memcpy(this.slotMappingH, totalTokens * I32, MemcpyKind.HostToDevice);
      this.qoIndptrD.memcpy(this.qoIndptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
    }

    const usedPages = pagedKV.sequences.reduce((sum, s) => sum + s.contentPages, 0);
    pagedKV.indices.memcpy(pagedKV.indicesH, usedPages * I32, MemcpyKind.HostToDevice);
    this.indptrD.memcpy(this.indptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
    this.lastPageLen.memcpy(this.lastPageLenH, batchSize * I32, MemcpyKind.HostToDevice);

    return new ExecutionState(batchSize, totalTokens, seqLens, false, this, cache, this.qoIndptrH, customMask);
  }

  forwardPrefill(model: ChatModel, inputIdsList: number[][], cache: ChatCache): Tensor {
    const batchSize = inputIdsList.length;
    const seqLens = inputIdsList.map(ids => ids.length);
    const state = this.planPrefill(model, batchSize, seqLens, cache);
    state.setInput(inputIdsList);
    using hiddenStates = model.forward(state);
    const logits = state.computeLogits(hiddenStates, model);
    return logits;
  }

  forwardEagerPrefill(model: ChatModel, inputIdsList: number[][], cache: ChatCache): number[] {
    const logits = this.forwardPrefill(model, inputIdsList, cache);
    using argmaxResult = logits.argmax();
    return argmaxResult.readInt32LEArray();
  }

  forwardDecode(model: ChatModel, state: ExecutionState): Tensor {
    const hiddenStates = model.forward(state);
    return state.computeLogits(hiddenStates, model);
  }

  forwardEagerDecode(model: ChatModel, tokenIdsList: number[], cache: ChatCache): number[] {
    const state = this.planDecode(model, tokenIdsList.length, cache);
    state.setInput([tokenIdsList]);
    this.positionStep(state, model);
    const hiddenStates = model.forward(state);
    const logits = state.computeLogits(hiddenStates, model);
    using argmaxResult = logits.argmax();
    return argmaxResult.readInt32LEArray();
  }
}