import { WorkspaceBuffers } from "./paged_kv";
import { PagedKVCache } from "./paged_kv";
import { Qwen35GdnState } from "./qwen35_gdn_state";

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
  free(): void;
}

export class Qwen35ChatCache implements ChatCache {
  constructor(
    public readonly pagedKV: PagedKVCache,
    public readonly gdnState: Qwen35GdnState,
  ) {}

  reset(batchSize: number): void {
    this.pagedKV.reset(batchSize);
    this.gdnState.reset();
  }

  free(): void {
    this.gdnState.free();
    this.pagedKV.free();
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


