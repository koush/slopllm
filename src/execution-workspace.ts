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

export interface ExecutionPhase<T = TensorTree> {
  readonly states: readonly ExecutionState[];
  readonly inputs: { [name: string]: Tensor };
  readonly captureKey: readonly (string | number)[];
  run(inputs: { [name: string]: Tensor }): T;
}

export type ExecutionPlan<T> = Generator<ExecutionPhase, T, unknown>;

export interface ExecutionPlanResult<T> {
  result: T;
  warmup: boolean;
}

export function* executionPhase<T extends TensorTree>(phase: ExecutionPhase<T>): Generator<ExecutionPhase, T, unknown> {
  return (yield phase) as T;
}

export async function executePlan<T>(captureManager: CaptureManager, ws: ExecutionWorkspace, plan: ExecutionPlan<T>): Promise<ExecutionPlanResult<T>> {
  let warmup = false;
  let step = plan.next();

  try {
    while (!step.done) {
      const phase = step.value;
      const captureKey = [...phase.captureKey];

      if (!captureManager.disabled && captureKey.length > 0) {
        warmup ||= !ExecutionState.isCaptured(captureManager, phase.states, captureKey);
      }

      const phaseResult = ExecutionState.captureAll(
        captureManager,
        phase.states,
        phase.inputs,
        (_capturing, inputs) => phase.run(inputs),
        captureKey,
      );

      await captureManager.ops.synchronizeAsync();

      ws.clearTracking([phase.inputs, phaseResult as TensorTree]);
      step = plan.next(phaseResult);
    }

    ws.clearTracking(step.value as TensorTree);
    return { result: step.value, warmup };
  } finally {
    if (!step.done) {
      try {
        plan.return(undefined as never);
      } finally {
        ws.clearTracking();
      }
    }
  }
}

export class ExecutionState {
  input?: Tensor;
  paddedKvLenInvariant = true;
  private readonly paddedKvLen: number;

  // Per-state buffers (allocated via ensureAlloc with slot-suffixed names).
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

  constructor(
    public readonly model: ChatModel,
    public readonly batchSize: number, public readonly totalTokens: number, public readonly seqLens: number[],
    public readonly isDecode: boolean, public readonly ws: ExecutionWorkspace, public readonly cache: ChatCache,
    public readonly customMask?: {
      indptr: Tensor;
      mask: Tensor;
      mode?: MaskMode;
      positionIds?: Tensor;
      maskKvLen?: Tensor;
    },
  ) {
    const totalKvLen = cache.getPagedKV().sequences.reduce((sum, s) => sum + s.allocLen, 0);
    this.paddedKvLen = ExecutionState.getPaddedKvLen(totalKvLen);
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

  sparseMlaPrepareCache(slots: Tensor, ckvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, topk: Tensor | undefined, cacheIdx: number, kvLoraRank: number, qkRopeDim: number) {
    const nnz = this.isDecode ? this.batchSize : this.totalTokens;
    return this.ws.glm.sparseMlaPrepareCache(
      this, slots, cacheIdx,
      ckvCache, appendCkv, appendKpe,
      topk,
      this.indices, this.indptrD,
      this.mlaBatchIndices, this.positionIds,
      nnz, kvLoraRank, qkRopeDim,
      kvLoraRank, qkRopeDim
    );
  }

  mlaKvCacheAppend(appendCkv: Tensor, appendKpe: Tensor, cacheIdx: number, kvLoraRank: number, qkRopeDim: number): { ckv: Tensor; kpe?: Tensor } {
    const pagedKV = this.cache.getPagedKV();
    const ckv = pagedKV.ckvData[cacheIdx];
    const nnz = this.isDecode ? this.batchSize : this.totalTokens;
    if (pagedKV.sparseMode) {
      return {
        ckv: this.ws.glm.concatAndCacheDsMla(
          this, cacheIdx,
          ckv,
          appendCkv, appendKpe,
          this.indices, this.indptrD,
          this.mlaBatchIndices, this.positionIds,
          nnz, kvLoraRank, qkRopeDim,
          kvLoraRank, qkRopeDim,
        ),
      };
    } else {
      const kpe = pagedKV.kpeData[cacheIdx];
      return this.ws.glm.mlaKvCacheAppend(
        ckv, kpe,
        this.indices, this.indptrD, this.lastPageLen,
        appendCkv, appendKpe,
        this.mlaBatchIndices, this.positionIds,
        nnz, kvLoraRank, qkRopeDim,
        kvLoraRank, qkRopeDim,
      );
    }
  }

  indexerKvCacheAppend(idxKOut: Tensor, cacheIdx: number, indexHeadDim: number): Tensor {
    const pagedKV = this.cache.getPagedKV();
    const kData = pagedKV.kData[cacheIdx];
    const nnz = this.isDecode ? this.batchSize : this.totalTokens;
    const cache = this.ws.glm.mlaKvCacheAppend(
      kData, null,
      this.indices, this.indptrD, this.lastPageLen,
      idxKOut, null,
      this.mlaBatchIndices, this.positionIds,
      nnz, indexHeadDim, 0,
      indexHeadDim, 0
    );
    return cache.ckv;
  }

  // Run the indexer and return the raw top-k token positions (Replicated
  // [totalQ, topk]) and scores (Replicated [totalQ, topk] BF16). Slot
  // conversion is deferred to the GLM attention layer so the same top-k can be
  // reused across shared layers and mapped to whichever addressing
  // (flat/paged) each layer's CKV buffer requires.
  indexerTopk(idxQ: Tensor, kData: Tensor, weights: Tensor, scale: number, topk: number): { values: Tensor, indices: Tensor } {
    const cm = (!this.isDecode && this.customMask?.mode === MaskMode.CausalCustom) ? this.customMask : undefined;
    return this.ws.glm.indexerTopk(
      this,
      idxQ, kData, weights,
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
      return this.ws.mlaPrefillPaged(this, qNope, qPe, ckv, kpe, smScale, !this.customMask ? MaskMode.Causal : this.customMask.mode, this.customMask?.mask, this.customMask?.indptr, this.customMask?.maskKvLen);
    }
  }

  sparseMla(qAbsorbed: Tensor, qPe: Tensor, ckv: Tensor, indices: Tensor, length: Tensor, topk: number, smScale: number): { o: Tensor, lse: Tensor } {
    if (this.isDecode) {
      const numSplits = Math.ceil(topk / 64);
      return this.ws.glm.sparseMlaDecode(
        this, qAbsorbed, qPe, ckv, indices,
        topk, numSplits,
        smScale, 0, length,
      );
    } else {
      return this.ws.glm.sparseMlaPrefill(
        this, qAbsorbed, qPe, ckv, indices,
        topk,
        smScale, length,
        this.indptrD, this.lastPageLen, this.kvTokenIndptrD,
      );
    }
  }

  setInput(tokenIds: number[][] | Tensor) {
    if (tokenIds instanceof Tensor) {
      this.input = this.inputIdsBuf;
      if (tokenIds !== this.inputIdsBuf) {
        this.input.memcpy(tokenIds, tokenIds.bytes, MemcpyKind.DeviceToDevice);
      }
    }
    else {
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

  // Stable identity of a captured graph, known before execution. Padded dims
  // are NOT included here; they are appended to the effective key only once the
  // graph has been learned to size its buffers by them (see CaptureManager).
  private static baseKeyParams(states: readonly ExecutionState[], providedKeyParams: (string | number)[]): (string | number)[] {
    const keyParams = [...(providedKeyParams ?? [])];
    for (const state of states) {
      keyParams.push(`batchSize:${state.batchSize}`, `totalTokens:${state.totalTokens}`);
    }
    return keyParams;
  }

  // Effective capture key: base key + any padded dims this base graph is known
  // to be variant in. Length-invariant graphs collapse all KV-length buckets to
  // a single key (capture once, replay always); variant graphs (e.g. the CP
  // CKV-gather prefill) get a distinct key per bucket.
  //
  // Every state contributes its own bucket: the states in a multi-state graph
  // sit at kv lengths separated by a fixed offset, but power-of-2 bucketing is
  // lossy, so one state's bucket does not determine the others' near a boundary.
  private static effectiveKeyParams(captureManager: CaptureManager, states: readonly ExecutionState[], providedKeyParams: (string | number)[]): (string | number)[] {
    const keyParams = this.baseKeyParams(states, providedKeyParams);
    const variant = captureManager.getLengthVariant(keyParams.join(","));
    if (variant.kvLen) {
      for (const state of states) keyParams.push(`paddedKvLen:${state.paddedKvLen}`);
    }
    return keyParams;
  }

  /**
   * isCaptured for a graph spanning several states. See captureAll.
   */
  static isCaptured(captureManager: CaptureManager, states: readonly ExecutionState[], providedKeyParams: (string | number)[]): boolean {
    if (providedKeyParams.length === 0) return false;
    return captureManager.isCaptured(this.effectiveKeyParams(captureManager, states, providedKeyParams));
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
    if (providedKeyParams.length === 0) return fn(false, inputs);
    const baseKey = this.baseKeyParams(states, providedKeyParams).join(",");
    const keyParams = this.effectiveKeyParams(captureManager, states, providedKeyParams);
    return captureManager.run(inputs, (capturing, capturedInputs) => {
      const result = fn(capturing, capturedInputs);
      captureManager.recordLengthVariant(baseKey, states.some(s => !s.paddedKvLenInvariant));
      return result as TensorTree;
    }, keyParams) as T;
  }

  isCaptured(captureManager: CaptureManager, providedKeyParams: (string | number)[]): boolean {
    return ExecutionState.isCaptured(captureManager, [this], providedKeyParams);
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
  extras = new Map<string, any>();
  private planSlot = 0;

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

  async withTrackingAsync<T>(keepExports = new Set<Tensor>(), fn: () => Promise<T>): Promise<T> {
    using _tracking = this.startTracking(keepExports);
    return await fn();
  }

  private nextPlanSlot(): number {
      return this.planSlot++;
  }

  private allocStateBuffers(state: ExecutionState, slot: number, B: number, S: number): void {
    const s = (name: string) => `${name}:${slot}`;

    state.intWs = this.ensureAlloc([BATCH_INT_WS_SIZE], "U8", s("intWs"));
    state.intWsH = this.ensureAllocPinned([BATCH_PINNED_INT_WS_SIZE], "U8", s("intWsH"));
    state.decodePlanInfo = this.ensureAllocPinned([DECODE_PLAN_INFO_SIZE * 8], "U8", s("decodePlanInfo"));
    state.prefillPlanInfo = this.ensureAllocPinned([PREFILL_PLAN_INFO_SIZE * 8], "U8", s("prefillPlanInfo"));
    state.mlaPrefillPlanInfo = this.ensureAllocPinned([MLA_PREFILL_PLAN_INFO_SIZE * 8], "U8", s("mlaPrefillPlanInfo"));
    state.mlaDecodePlanInfo = this.ensureAllocPinned([MLA_DECODE_PLAN_INFO_SIZE * 8], "U8", s("mlaDecodePlanInfo"));

    state.positionIds = this.ensureAlloc([B * S], "I32", s("positionIds"));
    state.positionIdsH = this.ensureAllocPinned([B * S], "I32", s("positionIdsH"));
    state.inputIdsBuf = this.ensureAlloc([B * S], "I32", s("inputIdsBuf"));
    state.inputIdsBufH = this.ensureAllocPinned([B * S], "I32", s("inputIdsBufH"));
    state.qoIndptrD = this.ensureAlloc([B + 1], "I32", s("qoIndptrD"));
    state.qoIndptrH = this.ensureAllocPinned([B + 1], "I32", s("qoIndptrH"));
    state.slotMapping = this.ensureAlloc([B * S], "I32", s("slotMapping"));
    state.slotMappingH = this.ensureAllocPinned([B * S], "I32", s("slotMappingH"));
    state.indptrD = this.ensureAlloc([(B + 1) * I32], "I32", s("indptrD"));
    state.indptrH = this.ensureAllocPinned([(B + 1) * I32], "I32", s("indptrH"));
    state.lastPageLen = this.ensureAlloc([B * I32], "I32", s("lastPageLen"));
    state.lastPageLenH = this.ensureAllocPinned([B], "I32", s("lastPageLenH"));
    state.globalLastPageLen = this.ensureAlloc([B * I32], "I32", s("globalLastPageLen"));
    state.globalLastPageLenH = this.ensureAllocPinned([B], "I32", s("globalLastPageLenH"));
    state.kvLenH = this.ensureAllocPinned([B], "I32", s("kvLenH"));
    state.kvLenD = this.ensureAlloc([B], "I32", s("kvLenD"));
    state.kvTokenIndptrH = this.ensureAllocPinned([(B + 1) * I32], "I32", s("kvTokenIndptrH"));
    state.kvTokenIndptrD = this.ensureAlloc([(B + 1) * I32], "I32", s("kvTokenIndptrD"));
    state.mlaBatchIndices = this.ensureAlloc([B * S], "I32", s("mlaBatchIndices"));
    state.mlaBatchIndicesH = this.ensureAllocPinned([B * S], "I32", s("mlaBatchIndicesH"));
    state.indices = this.ensureAlloc([B * S], "I32", s("indices"));
    state.indicesH = this.ensureAllocPinned([B * S], "I32", s("indicesH"));
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
    const state = new ExecutionState(model, batchSize, totalTokens, seqLens, true, this, cache);
    this.allocStateBuffers(state, slot, this.maxBatch, this.maxSeqLen);

    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pagedKV.allocDecodeToken(seqIdx);
    }

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
      state.mlaBatchIndices.memcpy(state.mlaBatchIndicesH, batchSize * I32, MemcpyKind.HostToDevice);
      state.qoIndptrH.withPinnedBuffer(buf => {
        for (let i = 0; i <= batchSize; i++) buf.writeInt32LE(i, i * I32);
      });
      state.qoIndptrD.memcpy(state.qoIndptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
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

    const usedPages = pagedKV.sequences.reduce((sum, s) => sum + s.contentPages, 0);
    state.positionIds.memcpy(state.positionIdsH, batchSize * I32, MemcpyKind.HostToDevice);
    state.indices.memcpy(state.indicesH, usedPages * I32, MemcpyKind.HostToDevice);
    state.indptrD.memcpy(state.indptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
    state.lastPageLen.memcpy(state.lastPageLenH, batchSize * I32, MemcpyKind.HostToDevice);
    state.globalLastPageLen.memcpy(state.globalLastPageLenH, batchSize * I32, MemcpyKind.HostToDevice);
    state.kvTokenIndptrD.memcpy(state.kvTokenIndptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
    if (!cfg.kvLoraRank) {
      state.slotMapping.memcpy(state.slotMappingH, batchSize * I32, MemcpyKind.HostToDevice);
    }

    return state;
  }

  planPrefill(model: ChatModel, batchSize: number, seqLens: number[], cache: ChatCache, customMask?: {
    indptr: Tensor;
    mask: Tensor;
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
    const state = new ExecutionState(model, batchSize, totalTokens, seqLens, false, this, cache, customMask);
    this.allocStateBuffers(state, slot, this.maxBatch, this.maxSeqLen);

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
    state.positionIds.memcpy(state.positionIdsH, totalTokens * I32, MemcpyKind.HostToDevice);

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
          batchSize, nHeads, cfg.kvLoraRank!, !customMask || customMask.mode === MaskMode.CausalCustom || customMask.mode === MaskMode.Causal,
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
      state.mlaBatchIndices.memcpy(state.mlaBatchIndicesH, totalTokens * I32, MemcpyKind.HostToDevice);
      state.qoIndptrD.memcpy(state.qoIndptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
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
      state.slotMapping.memcpy(state.slotMappingH, totalTokens * I32, MemcpyKind.HostToDevice);
      state.qoIndptrD.memcpy(state.qoIndptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
    }

    const usedPages = pagedKV.sequences.reduce((sum, s) => sum + s.contentPages, 0);
    state.indices.memcpy(state.indicesH, usedPages * I32, MemcpyKind.HostToDevice);
    state.indptrD.memcpy(state.indptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);
    state.lastPageLen.memcpy(state.lastPageLenH, batchSize * I32, MemcpyKind.HostToDevice);
    state.globalLastPageLen.memcpy(state.globalLastPageLenH, batchSize * I32, MemcpyKind.HostToDevice);

    state.kvLenD.memcpy(state.kvLenH, batchSize * I32, MemcpyKind.HostToDevice);
    state.kvTokenIndptrD.memcpy(state.kvTokenIndptrH, (batchSize + 1) * I32, MemcpyKind.HostToDevice);

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
