import { type CaptureManager } from "./capture-manager";
import { ChatModel, type ChatCache } from "./chat_model";
import { DeviceOps, MaskMode } from "./device_ops";
import { MemcpyKind } from "./enums";
import { I32 } from "./glm_ops";
import { type PagedKVCache } from "./paged_kv";
import { Tensor } from "./tensor";
import { type TensorTree } from "./tensor-tree";
import { WorkspaceBase } from "./workspace";

export const DECODE_PLAN_INFO_SIZE = 10;
export const PREFILL_PLAN_INFO_SIZE = 15;
export const MLA_PREFILL_PLAN_INFO_SIZE = 19;
export const MLA_DECODE_PLAN_INFO_SIZE = 10;
export const BATCH_FLOAT_WS_SIZE = 128 * 1024 * 1024;
export const BATCH_INT_WS_SIZE = 8 * 1024 * 1024;
export const BATCH_PINNED_INT_WS_SIZE = 8 * 1024 * 1024;

// One persistent allocation pair per plan slot. Reuse matching views as well:
// rebuilding every per-GPU view and pinned Buffer would add hot-path GC work.
class StateBuffers {
  private readonly device: Tensor;
  private readonly host: Tensor;
  private readonly views: { offset: number; count: number; pair: [Tensor, Tensor] }[] = [];
  private viewIndex = 0;
  private offset = 0;

  constructor(ws: ExecutionWorkspace, slot: number, capacity: number) {
    this.device = ws.ensureAlloc([capacity], "I32", `stateBuffers:${slot}`);
    this.host = ws.ensureAllocPinned([capacity], "I32", `stateBuffersH:${slot}`);
  }

  reset(): void {
    this.viewIndex = 0;
    this.offset = 0;
  }

  private getViews(offset: number, count: number): [Tensor, Tensor] {
    if (offset + count > this.device.numElements) {
      throw new Error("Planning metadata exceeds workspace capacity");
    }
    const index = this.viewIndex++;
    const existing = this.views[index];
    if (existing?.offset === offset && existing.count === count
        && existing.pair.every(view => !view.disposed)) return existing.pair;
    if (existing) for (const view of existing.pair) view[Symbol.dispose]();
    const device = this.device.narrow(offset, count);
    const host = this.host.narrow(offset, count);
    // These views are slot-owned, not forward temporaries. In particular,
    // ParallelTensor.narrow registers its composite result for tracking.
    device.workspace.tracked.delete(device);
    host.workspace.tracked.delete(host);
    const pair: [Tensor, Tensor] = [device, host];
    this.views[index] = { offset, count, pair };
    return pair;
  }

  alloc(count: number): [Tensor, Tensor] {
    const pair = this.getViews(this.offset, count);
    this.offset += count;
    return pair;
  }

  upload(inputTokens: number): void {
    // Input IDs are supplied after planning, by setInput (possibly GPU-to-GPU).
    const [device, host] = this.getViews(inputTokens, this.offset - inputTokens);
    device.memcpy(host, device.bytes, MemcpyKind.HostToDevice);
  }
}

export class ExecutionState {
  input?: Tensor;
  extras = new Map<string, any>();
  paddedKvLenInvariant = true;
  /** Length snapshot; launch decisions must use getGraphVariantPaddedKvLen() or DeviceOps.getCaptureKeys(). */
  readonly paddedKvLen: number;

  // Per-state buffers (backed by persistent slot-suffixed allocations).
  // Each plan call gets a unique slot so that plan1+plan2+run1+run2 is safe:
  // host pinned writes are not stream-ordered, so sharing a single host buffer
  // across plans would race. Device buffers also need to be per-state because
  // plan N+1's async H2D would overwrite the device buffer before run N reads it.
  intWs!: Tensor;
  intWsH!: Tensor;
  decodePlanInfo!: Tensor;
  prefillPlanInfo!: Tensor;
  mlaPrefillPlanInfo!: Tensor;
  mlaDecodePlanInfo!: Tensor;
  inputIdsBuf!: Tensor;
  inputIdsBufH!: Tensor;
  positionIds!: Tensor;
  positionIdsH!: Tensor;
  qoIndptrD!: Tensor;
  qoIndptrH!: Tensor;
  slotMapping!: Tensor;
  slotMappingH!: Tensor;
  indptrD!: Tensor;
  indptrH!: Tensor;
  lastPageLen!: Tensor;
  lastPageLenH!: Tensor;
  globalLastPageLen!: Tensor;
  globalLastPageLenH!: Tensor;
  kvLenH!: Tensor;
  kvLenD!: Tensor;
  kvTokenIndptrH!: Tensor;
  kvTokenIndptrD!: Tensor;
  mlaBatchIndices!: Tensor;
  mlaBatchIndicesH!: Tensor;
  indices!: Tensor;
  indicesH!: Tensor;
  private readonly buffers: StateBuffers;

  constructor(
    public readonly model: ChatModel,
    public readonly batchSize: number, public readonly totalTokens: number, public readonly seqLens: number[],
    public readonly isDecode: boolean, public readonly ws: ExecutionWorkspace, public readonly cache: ChatCache,
    slot: number,
    public readonly customMask?: {
      indptr?: Tensor;
      mask?: Tensor;
      mode?: MaskMode;
      positionIds?: Tensor;
      maskKvLen?: Tensor;
    },
  ) {
    const totalKvLen = cache.getPagedKV().sequences.reduce((sum, s) => sum + s.allocLen, 0);
    this.paddedKvLen = ExecutionState.getPaddedKvLen(totalKvLen);

    const s = (name: string) => `${name}:${slot}`;
    // FlashInfer scratch and host-only plan records are not metadata uploads.
    this.intWs = ws.ensureAlloc([BATCH_INT_WS_SIZE], "U8", s("intWs"));
    this.intWsH = ws.ensureAllocPinned([BATCH_PINNED_INT_WS_SIZE], "U8", s("intWsH"));
    this.decodePlanInfo = ws.ensureAllocPinned([DECODE_PLAN_INFO_SIZE * 8], "U8", s("decodePlanInfo"));
    this.prefillPlanInfo = ws.ensureAllocPinned([PREFILL_PLAN_INFO_SIZE * 8], "U8", s("prefillPlanInfo"));
    this.mlaPrefillPlanInfo = ws.ensureAllocPinned([MLA_PREFILL_PLAN_INFO_SIZE * 8], "U8", s("mlaPrefillPlanInfo"));
    this.mlaDecodePlanInfo = ws.ensureAllocPinned([MLA_DECODE_PLAN_INFO_SIZE * 8], "U8", s("mlaDecodePlanInfo"));

    this.buffers = ws.getStateBuffers(slot, cache.getPagedKV().maxPages);
    [this.inputIdsBuf, this.inputIdsBufH] = this.buffers.alloc(totalTokens);
    [this.positionIds, this.positionIdsH] = this.buffers.alloc(totalTokens);
    [this.qoIndptrD, this.qoIndptrH] = this.buffers.alloc(batchSize + 1);
    [this.slotMapping, this.slotMappingH] = this.buffers.alloc(totalTokens);
    [this.indptrD, this.indptrH] = this.buffers.alloc(batchSize + 1);
    [this.lastPageLen, this.lastPageLenH] = this.buffers.alloc(batchSize);
    [this.globalLastPageLen, this.globalLastPageLenH] = this.buffers.alloc(batchSize);
    [this.kvLenD, this.kvLenH] = this.buffers.alloc(batchSize);
    [this.kvTokenIndptrD, this.kvTokenIndptrH] = this.buffers.alloc(batchSize + 1);
    [this.mlaBatchIndices, this.mlaBatchIndicesH] = this.buffers.alloc(totalTokens);
    // KV page count changes without changing the graph key. Keep this LAST so
    // it never shifts another pointer; batchSize/totalTokens are in the key.
    const usedPages = cache.getPagedKV().sequences.reduce((sum, sequence) => sum + sequence.contentPages, 0);
    [this.indices, this.indicesH] = this.buffers.alloc(usedPages);
  }

  uploadPlan(): void {
    this.buffers.upload(this.totalTokens);
  }

  private static getPaddedKvLen(totalKvLen: number): number {
    return Math.max(1024, 1 << Math.ceil(Math.log2(totalKvLen)));
  }

  get lastIdx(): Tensor {
    return this.qoIndptrD.narrow(1, this.batchSize);
  }

  computeLogits(hiddenStates: Tensor, model: ChatModel, allTokens: boolean = false): Tensor {
    const lmHead = model.tensors.get("lm_head.weight")!;
    if (this.isDecode) {
      return hiddenStates.linear(lmHead);
    }
    else if (allTokens) {
      return hiddenStates.linear(lmHead);
    }
    else if (this.seqLens.some(length => length === 0)) {
      // A deferred row has no final hidden state. Gather a valid dummy row;
      // its selection is ignored when the prefill plan reports consumed inputs.
      let end = 0;
      const indices = this.seqLens.map(length => { end += length; return Math.max(0, end - 1); });
      using rows = this.ws.alloc([this.batchSize], "I32");
      rows.h2d(Buffer.from(new Int32Array(indices).buffer));
      using hiddenLast = hiddenStates.indexSelect(rows);
      return hiddenLast.linear(lmHead);
    }
    else {
      using lastIdxFromIndptr = this.lastIdx;
      using hiddenLast = hiddenStates.indexSelect(lastIdxFromIndptr, -1);
      return hiddenLast.linear(lmHead);
    }
  }

  embedding(embedTable: Tensor): Tensor {
    using narrowed = this.input!.narrow(0, this.totalTokens);
    return embedTable.embedding(narrowed);
  }

  rotaryEmbedding(invFreq: Tensor): { cos: Tensor, sin: Tensor } {
    const B = this.isDecode ? this.batchSize : 1;
    const S = this.isDecode ? 1 : this.totalTokens;
    const posIds = this.customMask?.positionIds || this.positionIds;
    using narrowed = posIds.narrow(0, B * S);
    return invFreq.rotaryEmbedding(narrowed, B, S);
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
      this.slotMapping,
      BS, nKv, hd,
      kTokenStride, kHeadStride, vTokenStride, vHeadStride
    );
  }

  mlaKvCacheAppend(appendCkv: Tensor, appendKpe: Tensor, cacheIdx: number, kvLoraRank: number, qkRopeDim: number): { ckv: Tensor; kpe?: Tensor } {
    const pagedKV = this.cache.getPagedKV();
    if (pagedKV.sparseMode) {
      return {
        ckv: this.concatAndCacheDsMla(appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim),
      };
    } else {
      const ckv = pagedKV.ckvData[cacheIdx];
      const kpe = pagedKV.kpeData[cacheIdx];
      const nnz = this.isDecode ? this.batchSize : this.totalTokens;
      return this.ws.glm.mlaKvCacheAppend(
        this, cacheIdx,
        ckv, kpe,
        this.indices, this.indptrD, this.lastPageLen,
        appendCkv, appendKpe,
        this.mlaBatchIndices, this.positionIds,
        nnz, kvLoraRank, qkRopeDim,
        kvLoraRank, qkRopeDim,
      );
    }
  }

  concatAndCacheDsMla(appendCkv: Tensor, appendKpe: Tensor, cacheIdx: number, kvLoraRank: number, qkRopeDim: number): Tensor {
    const pagedKV = this.cache.getPagedKV();
    const ckv = pagedKV.ckvData[cacheIdx];
    const nnz = this.isDecode ? this.batchSize : this.totalTokens;
    return this.ws.glm.concatAndCacheDsMla(
      this, cacheIdx,
      ckv,
      appendCkv, appendKpe,
      this.indices, this.indptrD,
      this.mlaBatchIndices, this.positionIds,
      nnz, kvLoraRank, qkRopeDim,
      kvLoraRank, qkRopeDim,
    );
  }

  indexerKvCacheAppend(idxKOut: Tensor, cacheIdx: number, indexHeadDim: number): { kData: Tensor; kScaleData: Tensor } {
    const pagedKV = this.cache.getPagedKV();
    const kData = pagedKV.kData[cacheIdx];
    const kScaleData = pagedKV.kScaleData[cacheIdx];
    const nnz = this.isDecode ? this.batchSize : this.totalTokens;
    const cache = this.ws.glm.mlaKvCacheAppend(
      this, cacheIdx,
      kData, kScaleData,
      this.indices, this.indptrD, this.lastPageLen,
      idxKOut, null,
      this.mlaBatchIndices, this.positionIds,
      nnz, indexHeadDim, 0,
      indexHeadDim, 0
    );
    return { kData: cache.ckv, kScaleData: cache.kpe! };
  }

  // Run the indexer and return the raw top-k token positions (Replicated
  // [totalQ, topk]) and scores (Replicated [totalQ, topk] BF16). Slot
  // conversion is deferred to the GLM attention layer so the same top-k can be
  // reused across shared layers and mapped to whichever addressing
  // (flat/paged) each layer's CKV buffer requires.
  indexerTopk(idxQ: Tensor, kData: Tensor, kScaleData: Tensor, weights: Tensor, scale: number, topk: number): { values: Tensor, indices: Tensor } {
    const cm = (!this.isDecode && this.customMask?.mode === MaskMode.CausalCustom) ? this.customMask : undefined;
    return this.ws.glm.indexerTopk(
      this,
      idxQ, kData, kScaleData, weights,
      this.indices, this.indptrD, this.globalLastPageLen, this.qoIndptrD,
      scale, topk,
      this.isDecode,
      0,
      cm?.mask, cm?.indptr, cm?.maskKvLen,
    );
  }

  denseMla(qNope: Tensor, qPe: Tensor, ckv: Tensor, kpe: Tensor, smScale: number): { o: Tensor, lse: Tensor } {
    if (this.isDecode) {
      return this.ws.mlaDecodePaged(this, qNope, qPe, ckv, kpe, smScale);
    } else {
      return this.ws.mlaPrefillPaged(this, qNope, qPe, ckv, kpe, smScale, this.customMask?.mode ?? MaskMode.Causal, this.customMask?.mask, this.customMask?.indptr, this.customMask?.maskKvLen);
    }
  }

  sparseMla(qAbsorbed: Tensor, qPe: Tensor, ckv: Tensor, indices: Tensor, length: Tensor, topk: number, smScale: number, qAbsorbedScales?: Tensor): { o: Tensor, lse: Tensor } {
    if (this.isDecode) {
      const numSplits = Math.ceil(topk / 64);
      return this.ws.glm.sparseMlaDecode(
        this, qAbsorbed, qPe, ckv, indices,
        topk, numSplits,
        smScale, length, qAbsorbedScales,
      );
    } else {
      return this.ws.glm.sparseMlaPrefill(
        this, qAbsorbed, qPe, ckv, indices,
        topk,
        smScale, length,
        this.indptrD, this.lastPageLen, this.kvTokenIndptrD, qAbsorbedScales,
      );
    }
  }

  setInput(tokenIds: readonly number[][] | Tensor) {
    if (tokenIds instanceof Tensor) {
      if (tokenIds.bytes > this.inputIdsBuf.bytes) {
        throw new Error("setInput: input exceeds planned token count");
      }
      this.input = this.inputIdsBuf;
      if (tokenIds !== this.inputIdsBuf) {
        this.input.memcpy(tokenIds, tokenIds.bytes, MemcpyKind.DeviceToDevice);
      }
    }
    else {
      if (tokenIds.reduce((sum, ids) => sum + ids.length, 0) > this.totalTokens) {
        throw new Error("setInput: input exceeds planned token count");
      }
      this.inputIdsBufH.withPinnedBuffer(buf => {
        let idsOff = 0;
        for (const ids of tokenIds) {
          for (const id of ids) {
            buf.writeInt32LE(id, idsOff);
            idsOff += I32;
          }
        }
        const totalBytes = this.totalTokens * I32;
        if (idsOff < totalBytes)
          buf.fill(0, idsOff, totalBytes);
      });
      this.input = this.inputIdsBuf;
      this.input.memcpy(this.inputIdsBufH, this.totalTokens * I32, MemcpyKind.HostToDevice);
    }
  }

  /**
   * isCaptured for a graph spanning several states. See captureAll.
   */
  static isCaptured(captureManager: CaptureManager, states: readonly ExecutionState[], providedKeyParams: (string | number)[], inputs: { [name: string]: Tensor } = {}): boolean {
    return captureManager.isStateCaptured({ states, inputs, key: providedKeyParams });
  }

  /**
   * Capture one graph that replays several planned states back-to-back (plan+plan
   * +run+run). The states must be planned within a single ws.startTracking scope,
   * in the same order on every step: that is what gives each its own plan slot and
   * therefore stable device pointers for the buffers baked into the graph.
   *
   * The graph is length-variant if ANY of its states sizes buffers by padded kv
   * len, and the key then carries every state's bucket.
   */
  static captureAll<T, I extends { [name: string]: Tensor }>(captureManager: CaptureManager, states: readonly ExecutionState[], inputs: I, fn: (capturing: boolean, capturedInputs: I) => T, providedKeyParams: (string | number)[]): T {
    return captureManager.runStates({ states, inputs, key: providedKeyParams }, fn);
  }

  isCaptured(captureManager: CaptureManager, providedKeyParams: (string | number)[]): boolean {
    return ExecutionState.isCaptured(captureManager, [this], providedKeyParams);
  }

  getEagerKvLen(): number {
    return this.cache.getPagedKV().sequences.reduce((sum, sequence) => sum + sequence.allocLen, 0);
  }

  getGraphVariantPaddedKvLen() {
    this.paddedKvLenInvariant = false;
    return this.paddedKvLen;
  }

  capture<T, I extends { [name: string]: Tensor }>(captureManager: CaptureManager, inputs: I, fn: (capturing: boolean, capturedInputs: I) => T, providedKeyParams: (string | number)[]): T {
    return ExecutionState.captureAll(captureManager, [this], inputs, fn, providedKeyParams);
  }
}


export class ExecutionWorkspace extends WorkspaceBase {
  readonly maxBatch: number;
  readonly maxSeqLen: number;
  private planSlot = 0;
  private readonly stateBuffers: StateBuffers[] = [];

  constructor(glm: DeviceOps, B: number, S: number) {
    super(glm);
    this.maxBatch = B;
    this.maxSeqLen = S;
  }

  startTracking(keepExports = new Set<Tensor>()): Disposable & { [Symbol.dispose](): void } {
    this.planSlot = 0;
    const inner = super.startTracking(keepExports);
    const ws = this;
    return {
      [Symbol.dispose]() {
        ws.planSlot = 0;
        inner[Symbol.dispose]();
      },
    };
  }

  clearTracking(keep: TensorTree = undefined): void {
    super.clearTracking(keep);
    this.planSlot = 0;
  }

  resetPlanSlots(): void {
    if (this.tracking !== null || this.tracked.size !== 0 || this.staged.size !== 0) {
      throw new Error("resetPlanSlots requires a clear workspace");
    }
    this.planSlot = 0;
  }

  async withTrackingAsync<T>(keepExports = new Set<Tensor>(), fn: () => Promise<T>): Promise<T> {
    using _tracking = this.startTracking(keepExports);
    return await fn();
  }

  private nextPlanSlot(): number {
      return this.planSlot++;
  }

  getStateBuffers(slot: number, maxPages: number): StateBuffers {
    // Shared prefixes can repeat a physical page once per sequence.
    const capacity = 4 * this.maxBatch * this.maxSeqLen + 6 * this.maxBatch + 3
      + this.maxBatch * Math.max(this.maxSeqLen, maxPages);
    const buffers = this.stateBuffers[slot] ??= new StateBuffers(this, slot, capacity);
    buffers.reset();
    return buffers;
  }

  flashDecode(state: ExecutionState, query: Tensor, cacheIdx: number, nHeads: number, nKv: number, hd: number, smScale: number): Tensor {
    const pagedKV = state.cache.getPagedKV();
    const out = this.alloc([state.batchSize, nHeads, 1, hd], query.type, undefined, query.parallelism);
    this.glm.batchDecodeRun(
      state, query, out,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
      state.indices, state.indptrD, state.lastPageLen,
      pagedKV.floatWs, state.intWs,
      state.decodePlanInfo,
      nHeads, nKv, hd, smScale
    );
    return out;
  }

  batchPrefillRagged(state: ExecutionState, q: Tensor, k: Tensor, v: Tensor, nHeads: number, nKv: number, hd: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: MaskMode, smScale: number): Tensor {
    const pagedKV = state.cache.getPagedKV();
    const out = this.alloc([1, nHeads, state.totalTokens, hd], q.type, undefined, q.parallelism);
    this.glm.batchPrefillRaggedRun(
      state, q, k, v, out,
      pagedKV.floatWs, state.intWs,
      state.qoIndptrD, state.kvTokenIndptrD,
      state.prefillPlanInfo,
      nHeads, nKv, hd,
      qStrideN, qStrideH, kvStrideN, kvStrideH, vStrideN, vStrideH,
      maskMode, smScale
    );
    return out;
  }

  flashPrefillPaged(state: ExecutionState, query: Tensor, cacheIdx: number, nHeads: number, nKv: number, hd: number, qStrideN: number, qStrideH: number, maskMode: MaskMode, smScale: number): Tensor {
    const pagedKV = state.cache.getPagedKV();
    const out = this.alloc([1, nHeads, state.totalTokens, hd], query.type, undefined, query.parallelism);
    this.glm.batchPrefillPagedRun(
      state, query, out,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
      state.indices, state.indptrD, state.lastPageLen,
      pagedKV.floatWs, state.intWs,
      state.qoIndptrD,
      state.prefillPlanInfo,
      nHeads, nKv, hd,
      qStrideN, qStrideH, maskMode, smScale
    );
    return out;
  }

  mlaPrefillPaged(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, smScale: number, maskMode: MaskMode = MaskMode.Causal, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor): { o: Tensor, lse: Tensor } {
    const pagedKV = state.cache.getPagedKV();
    return this.glm.mlaPrefillRun(
      state, qNope, qPe, ckvData, kpeData,
      state.indices,
      pagedKV.floatWs, state.intWs,
      state.mlaPrefillPlanInfo,
      smScale, maskMode,
      undefined, undefined,
      customMask, maskIndptr, maskKvLen
    );
  }

  mlaDecodePaged(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, smScale: number): { o: Tensor, lse: Tensor } {
    const pagedKV = state.cache.getPagedKV();
    return this.glm.mlaDecodeRun(
      state, qNope, qPe, ckvData, kpeData,
      state.indices, state.indptrD, state.lastPageLen,
      pagedKV.floatWs, state.intWs,
      state.mlaDecodePlanInfo,
      smScale,
    );
  }

  updateIndptr(state: ExecutionState, pagedKV: PagedKVCache): void {
    const batchSize = pagedKV.sequences.length;
    state.indptrH.withPinnedBuffer(buf => {
      buf.writeInt32LE(0, 0);
      let cumulative = 0;
      for (let i = 0; i < batchSize; i++) {
        cumulative += pagedKV.sequences[i].contentPages;
        buf.writeInt32LE(cumulative, (i + 1) * I32);
      }
    });

    state.indicesH.withPinnedBuffer(buf => {
      let indicesOff = 0;
      for (let i = 0; i < batchSize; i++) {
        const contentPages = pagedKV.sequences[i].contentPages;
        for (let j = 0; j < contentPages; j++) {
          buf.writeInt32LE(pagedKV.sequences[i].pages[j].id, indicesOff * I32);
          indicesOff++;
        }
      }
    });

    state.lastPageLenH.withPinnedBuffer(buf => {
      for (let i = 0; i < batchSize; i++) {
        const allocLen = pagedKV.sequences[i].allocLen;
        const remainder = allocLen % pagedKV.pageSize;
        buf.writeInt32LE(remainder !== 0 ? remainder : (allocLen > 0 ? pagedKV.pageSize : 0), i * I32);
      }
    });
    state.globalLastPageLenH.withPinnedBuffer(buf => {
      for (let i = 0; i < batchSize; i++) {
        const allocLen = pagedKV.sequences[i].allocLen;
        const remainder = allocLen % pagedKV.pageSize;
        buf.writeInt32LE(remainder !== 0 ? remainder : (allocLen > 0 ? pagedKV.pageSize : 0), i * I32);
      }
    });
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

    if (pagedKV.sequences.length !== batchSize) {
      throw new Error(`planDecode: pagedKV has ${pagedKV.sequences.length} sequences, expected ${batchSize}`);
    }

    let decodePagesNeeded = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      decodePagesNeeded += pagedKV.pagesNeededForDecodeToken(seqIdx);
    }
    pagedKV.ensureAvailablePages(decodePagesNeeded);
    if (decodePagesNeeded > pagedKV.availablePages.length) {
      throw new Error(`planDecode: need ${decodePagesNeeded} pages, ${pagedKV.availablePages.length} available`);
    }

    const slot = this.nextPlanSlot();
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pagedKV.allocDecodeToken(seqIdx);
    }
    const state = new ExecutionState(model, batchSize, totalTokens, seqLens, true, this, cache, slot);

    state.positionIdsH.withPinnedBuffer(buf => {
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        buf.writeInt32LE(pagedKV.sequences[seqIdx].allocLen - 1, seqIdx * I32);
      }
    });

    state.kvTokenIndptrH.withPinnedBuffer(buf => {
      buf.writeInt32LE(0, 0);
      let cumulative = 0;
      for (let i = 0; i < batchSize; i++) {
        cumulative += pagedKV.sequences[i].allocLen;
        buf.writeInt32LE(cumulative, (i + 1) * I32);
      }
    });

    this.updateIndptr(state, pagedKV);

    if (cfg.kvLoraRank) {
      state.mlaBatchIndicesH.withPinnedBuffer(buf => {
        for (let i = 0; i < batchSize; i++) buf.writeInt32LE(i, i * I32);
      });
      state.qoIndptrH.withPinnedBuffer(buf => {
        for (let i = 0; i <= batchSize; i++) buf.writeInt32LE(i, i * I32);
      });
      const seqKvLens = pagedKV.sequences.map(sequence => sequence.allocLen);
      if (pagedKV.sparseMode) {
        this.glm.sparseMlaDecodePlan(
          state.lastPageLenH, batchSize, seqKvLens,
          pagedKV.pageSize, pagedKV.contextParallel,
        );
      } else {
        this.glm.mlaDecodePlan(
          pagedKV.floatWs, BATCH_FLOAT_WS_SIZE,
          state.intWs, state.intWsH, BATCH_INT_WS_SIZE,
          state.mlaDecodePlanInfo,
          state.indptrH, state.lastPageLenH,
          batchSize, model.cfg.numAttentionHeads, pagedKV.pageSize, enableCudaGraph,
          cfg.kvLoraRank, cfg.qkRopeHeadDim!, seqKvLens, pagedKV.contextParallel,
        );
      }
    } else {
      state.slotMappingH.withPinnedBuffer(buf => {
        for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
          const position = pagedKV.sequences[seqIdx].allocLen - 1;
          const pageIdx = Math.floor(position / pagedKV.pageSize);
          const pageOffset = position % pagedKV.pageSize;
          const absPage = pagedKV.sequences[seqIdx].pages[pageIdx].id;
          buf.writeInt32LE(absPage * pagedKV.pageSize + pageOffset, seqIdx * I32);
        }
      });
      this.glm.batchDecodePlan(
        pagedKV.floatWs, BATCH_FLOAT_WS_SIZE,
        state.intWs, state.intWsH, BATCH_INT_WS_SIZE,
        state.decodePlanInfo,
        state.indptrH,
        batchSize,
        nHeads, nKv, hd, pageSize,
        enableCudaGraph
      );
    }

    state.uploadPlan();

    return state;
  }

  planPrefill(model: ChatModel, batchSize: number, seqLens: number[], cache: ChatCache, customMask?: {
    indptr?: Tensor;
    mask?: Tensor;
    mode?: MaskMode;
    positionIds?: Tensor;
    maskKvLen?: Tensor;
  }): ExecutionState {
    const pagedKV = cache.getPagedKV();
    const cfg = model.cfg;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const totalTokens = seqLens.reduce((a, b) => a + b, 0);

    if (totalTokens > this.maxSeqLen) {
      throw new Error(`planPrefill: ${totalTokens} total tokens exceed workspace capacity ${this.maxSeqLen}`);
    }
    if (pagedKV.sequences.length !== batchSize) {
      throw new Error(`planPrefill: pagedKV has ${pagedKV.sequences.length} sequences, expected ${batchSize}`);
    }

    const startPos = cache.getPagedKV().sequences.map(s => s.allocLen);
    cache.prefillBatchPlanHook?.(batchSize, seqLens, totalTokens, startPos, cache);

    let prefillPagesNeeded = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      prefillPagesNeeded += pagedKV.pagesNeededForAppend(seqIdx, seqLens[seqIdx]);
    }
    pagedKV.ensureAvailablePages(prefillPagesNeeded);
    if (prefillPagesNeeded > pagedKV.availablePages.length) {
      throw new Error(`planPrefill: need ${prefillPagesNeeded} pages, ${pagedKV.availablePages.length} available`);
    }

    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pagedKV.allocAppendPages(seqIdx, seqLens[seqIdx]);
    }

    const slot = this.nextPlanSlot();
    const state = new ExecutionState(model, batchSize, totalTokens, seqLens, false, this, cache, slot, customMask);

    state.qoIndptrH.withPinnedBuffer(buf => {
      buf.writeInt32LE(0, 0);
      for (let i = 0; i < batchSize; i++) {
        buf.writeInt32LE(buf.readInt32LE(i * I32) + seqLens[i], (i + 1) * I32);
      }
    });

    this.updateIndptr(state, pagedKV);

    state.positionIdsH.withPinnedBuffer(buf => {
      let posOff = 0;
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        for (let p = 0; p < seqLens[seqIdx]; p++) {
          buf.writeInt32LE(startPos[seqIdx] + p, posOff * I32);
          posOff++;
        }
      }
    });

    state.kvLenH.withPinnedBuffer(buf => {
      for (let i = 0; i < batchSize; i++) {
        buf.writeInt32LE(pagedKV.sequences[i].allocLen, i * I32);
      }
    });

    state.kvTokenIndptrH.withPinnedBuffer(buf => {
      buf.writeInt32LE(0, 0);
      let cumulative = 0;
      for (let i = 0; i < batchSize; i++) {
        cumulative += pagedKV.sequences[i].allocLen;
        buf.writeInt32LE(cumulative, (i + 1) * I32);
      }
    });

    if (cfg.kvLoraRank) {
      if (!pagedKV.sparseMode) {
        this.glm.mlaPrefillPlan(
          pagedKV.floatWs, BATCH_FLOAT_WS_SIZE,
          state.intWs, state.intWsH, BATCH_INT_WS_SIZE,
          state.mlaPrefillPlanInfo,
          state.qoIndptrH, state.indptrH,
          state.kvLenH, state.lastPageLenH,
          batchSize, nHeads, cfg.kvLoraRank!, !customMask || (customMask.mode ?? MaskMode.Causal) === MaskMode.Causal || customMask.mode === MaskMode.CausalCustom,
          pagedKV.pageSize, pagedKV.sequences.map(s => s.allocLen),
          pagedKV.contextParallel
        );
      }
      else {
        // Sparse prefill has no kernel plan, but CP consumers still need
        // per-rank physical-page lengths rather than the global logical length.
        this.glm.sparseMlaDecodePlan(
          state.lastPageLenH, batchSize, pagedKV.sequences.map(s => s.allocLen),
          pagedKV.pageSize, pagedKV.contextParallel,
        );
      }
      state.mlaBatchIndicesH.withPinnedBuffer(buf => {
        let off = 0;
        for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
          for (let pos = 0; pos < seqLens[seqIdx]; pos++) {
            buf.writeInt32LE(seqIdx, off * I32);
            off++;
          }
        }
      });
    } else {
      this.glm.batchPrefillPagedPlan(
        pagedKV.floatWs, BATCH_FLOAT_WS_SIZE,
        state.intWs, state.intWsH, BATCH_INT_WS_SIZE,
        state.prefillPlanInfo,
        state.qoIndptrH, state.indptrH,
        totalTokens, batchSize,
        nHeads, nKv, hd,
        pageSize,
        1
      );
      state.slotMappingH.withPinnedBuffer(buf => {
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
    }

    state.uploadPlan();

    return state;
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
    using logits = this.forwardPrefill(model, inputIdsList, cache);
    using argmaxResult = logits.argmax();
    return argmaxResult.readInt32LEArray();
  }

  forwardDecode(model: ChatModel, state: ExecutionState): Tensor {
    using hiddenStates = model.forward(state);
    return state.computeLogits(hiddenStates, model);
  }

  forwardEagerDecode(model: ChatModel, tokenIdsList: number[], cache: ChatCache): number[] {
    const state = this.planDecode(model, tokenIdsList.length, cache);
    state.setInput([tokenIdsList]);
    using hiddenStates = model.forward(state);
    using logits = state.computeLogits(hiddenStates, model);
    using argmaxResult = logits.argmax();
    return argmaxResult.readInt32LEArray();
  }
}
