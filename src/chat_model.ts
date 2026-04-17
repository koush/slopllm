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

export interface DecodeState {
  batchSize: number;
}

export interface PrefillState {
  batchSize: number;
  totalTokens: number;
  seqLens: number[];
  pageAllocs: [number, number][];
}

export interface ChatModel {
  readonly eosIds: Set<number>;
  createChatCache(maxPages?: number): ChatCache;
  chatStream(
    inputIds: number[][],
    cache: ChatCache,
    ws: WorkspaceBuffers,
    maxNewTokens: number,
    eosIds?: Set<number>,
    sampling?: SamplingParams,
  ): Generator<number>;
  prefillBatch(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache): number[];
  decodeBatchPlan(tokenIdsList: number[], ws: WorkspaceBuffers, cache: ChatCache, enableCudaGraph?: boolean): DecodeState;
  decodeBatchForward(state: DecodeState, ws: WorkspaceBuffers, cache: ChatCache): void;
  decodeBatchRead(state: DecodeState): number[];
  generateBatch(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache, maxNewTokens?: number, eosTokenIds?: Set<number>): number[][];
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
  abstract prefillBatchPlan(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache): PrefillState;
  abstract prefillBatchForward(state: PrefillState, ws: WorkspaceBuffers, cache: ChatCache): void;
  abstract decodeBatchForward(state: DecodeState, ws: WorkspaceBuffers, cache: ChatCache): void;
  abstract free(): void;

  protected abstract getPagedKV(cache: ChatCache): PagedKVCache;

  *chatStream(
    inputIds: number[][], cache: ChatCache, ws: WorkspaceBuffers,
    maxNewTokens = 100, eosIds?: Set<number>, sampling?: SamplingParams,
  ): Generator<number> {
    yield* this.streamTokens(inputIds, ws, cache, maxNewTokens, eosIds ?? this.eosIds, sampling);
  }

  *streamTokens(
    inputIds: number[][], ws: WorkspaceBuffers, cache: ChatCache,
    maxNewTokens = 100, eosTokenIds?: Set<number>, sampling?: SamplingParams,
  ): Generator<number> {
    if (inputIds.length !== 1) throw new Error("streamTokens only supports batch=1");
    const effectiveEosIds = eosTokenIds ?? this.eosIds;
    cache.reset(1);
    const firstTokens = this.prefillBatch(inputIds, ws, cache);
    let nextToken = firstTokens[0];
    yield nextToken;

    const tokenHistory = [...inputIds[0], nextToken];

    for (let i = 0; i < maxNewTokens - 1; i++) {
      if (effectiveEosIds.has(nextToken)) break;

      const decodeTokens = this.decodeBatch([nextToken], ws, cache);
      nextToken = decodeTokens[0];

      if (sampling && needsSampling(sampling)) {
        nextToken = this.sampleTokenGPU(sampling, tokenHistory);
      }

      tokenHistory.push(nextToken);
      yield nextToken;
    }
  }

  generateTokens(
    inputIds: number[][], ws: WorkspaceBuffers, cache: ChatCache,
    maxNewTokens = 100, eosTokenIds?: Set<number>, sampling?: SamplingParams,
  ): number[] {
    return [...this.streamTokens(inputIds, ws, cache, maxNewTokens, eosTokenIds, sampling)];
  }

  generateBatch(
    inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache,
    maxNewTokens = 100, eosTokenIds?: Set<number>,
  ): number[][] {
    eosTokenIds = eosTokenIds ?? this.eosIds;
    const batchSize = inputIdsList.length;
    cache.reset(batchSize);
    const firstTokens = this.prefillBatch(inputIdsList, ws, cache);

    const nextTokens = [...firstTokens];
    const generated: number[][] = nextTokens.map(t => [t]);
    const finished = nextTokens.map(t => eosTokenIds.has(t));

    for (let step = 0; step < maxNewTokens - 1; step++) {
      if (finished.every(f => f)) break;

      const newTokens = this.decodeBatch(nextTokens, ws, cache);

      for (let i = 0; i < batchSize; i++) {
        nextTokens[i] = newTokens[i];
        if (!finished[i]) {
          if (eosTokenIds.has(newTokens[i])) {
            finished[i] = true;
          } else {
            generated[i].push(newTokens[i]);
          }
        }
      }
    }

    return generated;
  }

  prefill(inputIds: number[][], ws: WorkspaceBuffers, cache: ChatCache): number {
    return this.prefillBatch(inputIds, ws, cache)[0];
  }

  decode(tokenId: number, ws: WorkspaceBuffers, cache: ChatCache): number {
    return this.decodeBatch([tokenId], ws, cache)[0];
  }

  prefillBatch(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache): number[] {
    const state = this.prefillBatchPlan(inputIdsList, ws, cache);
    this.prefillBatchForward(state, ws, cache);
    return this.prefillBatchRead(state);
  }

  prefillBatchRead(state: PrefillState): number[] {
    const batchSize = state.batchSize;
    const buf = Buffer.alloc(batchSize * I32);
    this.ws.argmaxIdx.d2h(buf);
    const result: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      result.push(buf.readInt32LE(i * I32));
    }
    return result;
  }

  decodeBatch(tokenIdsList: number[], ws: WorkspaceBuffers, cache: ChatCache): number[] {
    const state = this.decodeBatchPlan(tokenIdsList, ws, cache);
    this.decodeBatchForward(state, ws, cache);
    return this.decodeBatchRead(state);
  }

  decodeBatchPlan(tokenIdsList: number[], ws: WorkspaceBuffers, cache: ChatCache, enableCudaGraph = false): DecodeState {
    const pagedKV = this.getPagedKV(cache);
    const cfg = this.cfg;
    const glm = this.glm;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const batchSize = tokenIdsList.length;

    const writeLocations: [number, number][] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      writeLocations.push(pagedKV.allocDecodeToken(seqIdx));
    }

    pagedKV.updateIndptr();
    pagedKV.updateSlotMapping(writeLocations, pageSize);

    const idsBuf = Int32Array.from(tokenIdsList);
    this.ws.inputIdsBuf.h2d(Buffer.from(idsBuf.buffer, idsBuf.byteOffset, idsBuf.byteLength));

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

    return { batchSize };
  }

  decodeBatchRead(state: DecodeState): number[] {
    const batchSize = state.batchSize;
    const buf = Buffer.alloc(batchSize * I32);
    this.ws.argmaxIdx.d2h(buf);
    const result: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      result.push(buf.readInt32LE(i * I32));
    }
    return result;
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

  protected sampleTokenGPU(params: SamplingParams, tokenHistory: number[]): number {
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
