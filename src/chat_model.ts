import { BATCH_FLOAT_WS_SIZE, BATCH_INT_WS_SIZE, GlmOps, I32 } from "./glm_ops";
import { ExecutionWorkspace, PagedKVCache } from "./paged_kv";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

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
  readonly ws: ExecutionWorkspace;
  readonly cache: ChatCache;
}

export interface ChatModel {
  readonly eosIds: Set<number>;
  readonly vocabSize: number;
  createChatCache(maxPages?: number): ChatCache;
  plan(ws: ExecutionWorkspace, inputIdsList: number[][], cache: ChatCache, enableCudaGraph?: boolean): BatchState;
  forward(state: BatchState): Tensor;
  forwardEager(ws: ExecutionWorkspace, inputIdsList: number[][], cache: ChatCache): number[];
  planDecode(ws: ExecutionWorkspace, tokenIdsList: number[], cache: ChatCache, enableCudaGraph?: boolean): BatchState;
  forwardDecode(state: BatchState): Tensor;
  forwardEagerDecode(ws: ExecutionWorkspace, tokenIdsList: number[], cache: ChatCache): number[];
  free(): void;
}

export interface CommonModelConfig {
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  vocabSize: number;
}

export abstract class ChatModelBase extends WorkspaceBase implements ChatModel {
  abstract readonly eosIds: Set<number>;
  protected abstract readonly cfg: CommonModelConfig;
  get vocabSize(): number { return this.cfg.vocabSize; }

  protected constructor(glm: GlmOps) {
    super(glm);
  }

  abstract createChatCache(maxPages?: number): ChatCache;

  protected abstract getPagedKV(cache: ChatCache): PagedKVCache;

  protected prefillBatchPlanHook(
    _inputIdsList: number[][], _seqLens: number[], _totalTokens: number,
    _startPos: number[], _cache: ChatCache,
  ): void {}

  plan(ws: ExecutionWorkspace, inputIdsList: number[][], cache: ChatCache, enableCudaGraph = false): BatchState {
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
    ws.inputIdsBuf.h2d(Buffer.from(idsBuf.buffer, idsBuf.byteOffset, idsBuf.byteLength));

    if (isDecode) {
      const writeLocations: [number, number][] = [];
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        writeLocations.push(pagedKV.allocDecodeToken(seqIdx));
      }

      pagedKV.updateIndptr(ws);

      const slotMappingBuf = new Int32Array(batchSize);
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        const [absPage, slotInPage] = writeLocations[seqIdx];
        slotMappingBuf[seqIdx] = absPage * pageSize + slotInPage;
      }
      ws.slotMapping.h2d(Buffer.from(slotMappingBuf.buffer, slotMappingBuf.byteOffset, slotMappingBuf.byteLength));

      const posIds = new Array(batchSize);
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        posIds[seqIdx] = pagedKV.seqKvLens[seqIdx] - 1;
      }
      const posIdsBuf = Int32Array.from(posIds);
      ws.positionIds.h2d(Buffer.from(posIdsBuf.buffer, posIdsBuf.byteOffset, posIdsBuf.byteLength));

      glm.batchDecodePlan(
        ws.floatWs.data, BATCH_FLOAT_WS_SIZE,
        ws.intWs.data, ws.pinnedIntWs.data, BATCH_INT_WS_SIZE,
        ws.decodePlanInfo.data,
        ws.indptrH.data,
        batchSize,
        nHeads, nKv, hd, pageSize,
        enableCudaGraph
      );

      return { batchSize, totalTokens, seqLens, pageAllocs: [], isDecode: true, ws, cache };
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

    pagedKV.updateIndptr(ws);

    const posIds: number[] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      for (let p = 0; p < seqLens[seqIdx]; p++) {
        posIds.push(startPos[seqIdx] + p);
      }
    }
    const posIdsBuf = Int32Array.from(posIds);
    ws.positionIds.h2d(Buffer.from(posIdsBuf.buffer, posIdsBuf.byteOffset, posIdsBuf.byteLength));

    const lastIndices: number[] = [];
    let offset = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      lastIndices.push(offset + seqLens[seqIdx] - 1);
      offset += seqLens[seqIdx];
    }
    const lastIdxBuf = Int32Array.from(lastIndices);
    ws.lastIdx.h2d(Buffer.from(lastIdxBuf.buffer, lastIdxBuf.byteOffset, lastIdxBuf.byteLength));

    const qoIndptrHostPtr = glm.allocPinned((batchSize + 1) * I32);
    glm.writePinned(qoIndptrHostPtr, Buffer.from(qoIndptrBuf.buffer, qoIndptrBuf.byteOffset, qoIndptrBuf.byteLength));

    glm.batchPrefillPagedPlan(
      ws.floatWs.data, BATCH_FLOAT_WS_SIZE,
      ws.intWs.data, ws.pinnedIntWs.data, BATCH_INT_WS_SIZE,
      ws.prefillPlanInfo.data,
      qoIndptrHostPtr, ws.indptrH.data,
      totalTokens, batchSize,
      nHeads, nKv, hd,
      pageSize,
      1
    );

    glm.freePinned(qoIndptrHostPtr);

    ws.qoIndptrD.h2d(Buffer.from(qoIndptrBuf.buffer, qoIndptrBuf.byteOffset, qoIndptrBuf.byteLength));

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
    ws.slotMapping.h2d(Buffer.from(slotMappingBuf.buffer, slotMappingBuf.byteOffset, slotMappingBuf.byteLength));

    return { batchSize, totalTokens, seqLens, pageAllocs, isDecode: false, ws, cache };
  }

  abstract forward(state: BatchState): Tensor;

  forwardEager(ws: ExecutionWorkspace, inputIdsList: number[][], cache: ChatCache): number[] {
    const state = this.plan(ws, inputIdsList, cache);
    const logits = this.forward(state);
    using argmaxResult = logits.argmax();
    return argmaxResult.readInt32LE();
  }

  planDecode(ws: ExecutionWorkspace, tokenIdsList: number[], cache: ChatCache, enableCudaGraph = false): BatchState {
    return this.plan(ws, tokenIdsList.map(t => [t]), cache, enableCudaGraph);
  }

  forwardDecode(state: BatchState): Tensor {
    return this.forward(state);
  }

  forwardEagerDecode(ws: ExecutionWorkspace, tokenIdsList: number[], cache: ChatCache): number[] {
    return this.forwardEager(ws, tokenIdsList.map(t => [t]), cache);
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

export function samplingLabel(sp: SamplingParams): string {
  const parts: string[] = [];
  parts.push(`temp=${sp.temperature}`);
  if (sp.topP < 1.0) parts.push(`top_p=${sp.topP}`);
  if (sp.topK > 0) parts.push(`top_k=${sp.topK}`);
  if (sp.repetitionPenalty !== 1.0) parts.push(`rep_pen=${sp.repetitionPenalty}`);
  if (sp.presencePenalty !== 0) parts.push(`pres_pen=${sp.presencePenalty}`);
  return parts.join(" ");
}
