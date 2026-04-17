import { WorkspaceBuffers } from "./paged_kv";

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
  free(): void;
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
