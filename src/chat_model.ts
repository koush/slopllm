import { GlmOps, I32, BATCH_FLOAT_WS_SIZE, BATCH_INT_WS_SIZE, BATCH_PINNED_INT_WS_SIZE } from "./glm_ops";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv";
import { Tensor } from "./tensor";

export interface SamplingParams {
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  repetitionPenaltyWindow: number;
}

export interface ChatCache {
  reset(batchSize: number): void;
  free(): void;
  prefixMatch(seqIdx: number, inputIds: number[]): number[];
  appendTokens(seqIdx: number, tokens: number[]): void;
}

export interface BatchState {
  batchSize: number;
  totalTokens: number;
  seqLens: number[];
  pageAllocs: [number, number][];
  readonly isDecode: boolean;
}

export interface ChatModel {
  readonly eosIds: Set<number>;
  createChatCache(maxPages?: number): ChatCache;
  plan(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache, enableCudaGraph?: boolean): BatchState;
  forward(state: BatchState, ws: WorkspaceBuffers, cache: ChatCache): void;
  read(state: BatchState): number[];
  forwardEager(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache): number[];
  planDecode(tokenIdsList: number[], ws: WorkspaceBuffers, cache: ChatCache, enableCudaGraph?: boolean): BatchState;
  forwardDecode(state: BatchState, ws: WorkspaceBuffers, cache: ChatCache): void;
  readDecode(state: BatchState): number[];
  forwardEagerDecode(tokenIdsList: number[], ws: WorkspaceBuffers, cache: ChatCache): number[];
  sampleTokenGPU(params: SamplingParams, tokenHistory: number[]): number;
  free(): void;
}

export interface CommonModelConfig {
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  vocabSize: number;
}

export interface CommonModelWorkspace {
  argmaxIdx: Tensor;
  inputIdsBuf: Tensor;
  positionIds: Tensor;
  lastIdx: Tensor;
  qoIndptrD: Tensor;
  prefillSlotMapping: Tensor;
  logitsBuf: Tensor;
  sampleOutToken: Tensor;
  sampleTopkVals: Tensor;
  sampleTopkIdxs: Tensor;
  sampleWorkspace: Tensor;
  samplePenaltyTokens: Tensor;
}

export abstract class ChatModelBase implements ChatModel {
  abstract readonly eosIds: Set<number>;
  protected abstract readonly glm: GlmOps;
  protected abstract readonly cfg: CommonModelConfig;
  protected abstract readonly ws: CommonModelWorkspace;
  protected abstract readonly weights: Map<string, Tensor>;

  abstract createChatCache(maxPages?: number): ChatCache;
  abstract free(): void;

  protected abstract getPagedKV(cache: ChatCache): PagedKVCache;

  protected prefillBatchPlanHook(
    _inputIdsList: number[][], _seqLens: number[], _totalTokens: number,
    _startPos: number[], _cache: ChatCache,
  ): void {}

  plan(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache, enableCudaGraph = false): BatchState {
    const pagedKV = this.getPagedKV(cache);
    const cfg = this.cfg;
    const glm = this.glm;
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
    this.ws.inputIdsBuf.h2d(Buffer.from(idsBuf.buffer, idsBuf.byteOffset, idsBuf.byteLength));

    if (isDecode) {
      const writeLocations: [number, number][] = [];
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        writeLocations.push(pagedKV.allocDecodeToken(seqIdx));
      }

      pagedKV.updateIndptr();
      pagedKV.updateSlotMapping(writeLocations, pageSize);

      const posIds = new Array(batchSize);
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        posIds[seqIdx] = pagedKV.seqKvLens[seqIdx] - 1;
      }
      const posIdsBuf = Int32Array.from(posIds);
      this.ws.positionIds.h2d(Buffer.from(posIdsBuf.buffer, posIdsBuf.byteOffset, posIdsBuf.byteLength));

      glm.batchDecodePlan(
        ws.floatWs, BATCH_FLOAT_WS_SIZE,
        ws.intWs, ws.pinnedIntWs, BATCH_INT_WS_SIZE,
        ws.decodePlanInfo,
        pagedKV.indptrH,
        batchSize,
        nHeads, nKv, hd, pageSize,
        enableCudaGraph
      );

      return { batchSize, totalTokens, seqLens, pageAllocs: [], isDecode: true };
    }

    const startPos = pagedKV.seqKvLens.slice();

    this.prefillBatchPlanHook(inputIdsList, seqLens, totalTokens, startPos, cache);

    const pageAllocs: [number, number][] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pageAllocs.push(pagedKV.allocAppendPages(seqIdx, seqLens[seqIdx]));
    }

    const qoIndptr = [0];
    for (const s of seqLens) {
      qoIndptr.push(qoIndptr[qoIndptr.length - 1] + s);
    }
    const qoIndptrBuf = Int32Array.from(qoIndptr);

    pagedKV.updateIndptr();

    const posIds: number[] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      for (let p = 0; p < seqLens[seqIdx]; p++) {
        posIds.push(startPos[seqIdx] + p);
      }
    }
    const posIdsBuf = Int32Array.from(posIds);
    this.ws.positionIds.h2d(Buffer.from(posIdsBuf.buffer, posIdsBuf.byteOffset, posIdsBuf.byteLength));

    const lastIndices: number[] = [];
    let offset = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      lastIndices.push(offset + seqLens[seqIdx] - 1);
      offset += seqLens[seqIdx];
    }
    const lastIdxBuf = Int32Array.from(lastIndices);
    this.ws.lastIdx.h2d(Buffer.from(lastIdxBuf.buffer, lastIdxBuf.byteOffset, lastIdxBuf.byteLength));

    const qoIndptrHostPtr = glm.allocPinned((batchSize + 1) * I32);
    glm.writePinned(qoIndptrHostPtr, Buffer.from(qoIndptrBuf.buffer, qoIndptrBuf.byteOffset, qoIndptrBuf.byteLength));

    glm.batchPrefillPagedPlan(
      ws.floatWs, BATCH_FLOAT_WS_SIZE,
      ws.intWs, ws.pinnedIntWs, BATCH_INT_WS_SIZE,
      ws.prefillPlanInfo,
      qoIndptrHostPtr, pagedKV.indptrH,
      totalTokens, batchSize,
      nHeads, nKv, hd,
      pageSize,
      1
    );

    glm.freePinned(qoIndptrHostPtr);

    this.ws.qoIndptrD.h2d(Buffer.from(qoIndptrBuf.buffer, qoIndptrBuf.byteOffset, qoIndptrBuf.byteLength));

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
    this.ws.prefillSlotMapping.h2d(Buffer.from(slotMappingBuf.buffer, slotMappingBuf.byteOffset, slotMappingBuf.byteLength));

    return { batchSize, totalTokens, seqLens, pageAllocs, isDecode: false };
  }

  abstract forward(state: BatchState, ws: WorkspaceBuffers, cache: ChatCache): void;

  read(state: BatchState): number[] {
    const batchSize = state.batchSize;
    const buf = Buffer.alloc(batchSize * I32);
    this.ws.argmaxIdx.d2h(buf);
    const result: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      result.push(buf.readInt32LE(i * I32));
    }
    return result;
  }

  forwardEager(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache): number[] {
    const state = this.plan(inputIdsList, ws, cache);
    this.forward(state, ws, cache);
    return this.read(state);
  }

  planDecode(tokenIdsList: number[], ws: WorkspaceBuffers, cache: ChatCache, enableCudaGraph = false): BatchState {
    return this.plan(tokenIdsList.map(t => [t]), ws, cache, enableCudaGraph);
  }

  forwardDecode(state: BatchState, ws: WorkspaceBuffers, cache: ChatCache): void {
    this.forward(state, ws, cache);
  }

  readDecode(state: BatchState): number[] {
    return this.read(state);
  }

  forwardEagerDecode(tokenIdsList: number[], ws: WorkspaceBuffers, cache: ChatCache): number[] {
    return this.forwardEager(tokenIdsList.map(t => [t]), ws, cache);
  }

  protected readArgmax(ptr: Tensor | number, count: number): number {
    this.ws.argmaxIdx.argmax(ptr, count);
    const buf = Buffer.alloc(I32);
    this.ws.argmaxIdx.d2h(buf);
    return buf.readInt32LE(0);
  }

  protected readArgmaxBatch(batchSize: number): number[] {
    const vs = this.cfg.vocabSize;
    this.ws.argmaxIdx.argmax(this.ws.logitsBuf, vs, batchSize);
    const buf = Buffer.alloc(batchSize * I32);
    this.ws.argmaxIdx.d2h(buf);
    const result: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      result.push(buf.readInt32LE(i * I32));
    }
    return result;
  }

  sampleTokenGPU(params: SamplingParams, tokenHistory: number[]): number {
    const vs = this.cfg.vocabSize;
    const glm = this.glm;

    const hasRepPenalty = params.repetitionPenalty !== 1.0;
    const hasPresPenalty = params.presencePenalty !== 0;

    let numPenaltyTokens = 0;
    if (hasRepPenalty || hasPresPenalty) {
      const seen = new Set<number>();
      const start = Math.max(0, tokenHistory.length - params.repetitionPenaltyWindow);
      for (let i = start; i < tokenHistory.length; i++) seen.add(tokenHistory[i]);
      const penaltyBuf = Buffer.alloc(seen.size * I32);
      let offset = 0;
      for (const tid of seen) {
        if (tid < vs) {
          penaltyBuf.writeInt32LE(tid, offset);
          offset += I32;
          numPenaltyTokens++;
        }
      }
      if (numPenaltyTokens > 0) {
        this.ws.samplePenaltyTokens.h2d(penaltyBuf, numPenaltyTokens * I32);
      }
    }

    const randomVal = Math.random();
    const topK = params.topK > 0 ? params.topK : 0;
    const temperature = params.temperature > 0 ? params.temperature : 0;

    glm.sample(
      this.ws.sampleOutToken.data,
      this.ws.sampleTopkVals.data,
      this.ws.sampleTopkIdxs.data,
      this.ws.sampleWorkspace.data,
      this.ws.logitsBuf.data,
      this.ws.samplePenaltyTokens.data,
      vs,
      numPenaltyTokens,
      temperature,
      params.repetitionPenalty,
      params.presencePenalty,
      topK,
      params.topP,
      randomVal,
    );

    const buf = Buffer.alloc(I32);
    this.ws.sampleOutToken.d2h(buf);
    return buf.readInt32LE(0);
  }
}

export function makeSamplingParams(args: {
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  repetitionPenaltyWindow: number;
}): SamplingParams {
  return {
    temperature: args.temperature,
    topP: args.topP,
    topK: args.topK,
    repetitionPenalty: args.repetitionPenalty,
    presencePenalty: args.presencePenalty,
    repetitionPenaltyWindow: args.repetitionPenaltyWindow,
  };
}

export function needsSampling(sp: SamplingParams): boolean {
  return sp.temperature > 0 || sp.repetitionPenalty !== 1.0 || sp.presencePenalty !== 0 || sp.topK > 0 || sp.topP < 1.0;
}

export function samplingLabel(sp: SamplingParams): string {
  const parts: string[] = [];
  parts.push(`temp=${sp.temperature}`);
  if (sp.topP < 1.0) parts.push(`top_p=${sp.topP}`);
  if (sp.topK > 0) parts.push(`top_k=${sp.topK}`);
  if (sp.repetitionPenalty !== 1.0) parts.push(`rep_pen=${sp.repetitionPenalty}`);
  if (sp.presencePenalty !== 0) parts.push(`pres_pen=${sp.presencePenalty}`);
  return parts.join(" ");
}
