import fs from "node:fs";
import path from "node:path";
import { DeviceOps } from "./device_ops";
import { mmapOpen, mmapClose } from "./glm_ops";
import { PagedKVCache, type BatchState } from "./paged_kv";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
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
  getPagedKV(): PagedKVCache;
  reset(batchSize: number): void;
  free(): void;
  prefixMatch(seqIdx: number, inputIds: number[]): number[];
  appendTokens(seqIdx: number, tokens: number[]): void;
}

export interface CommonModelConfig {
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  vocabSize: number;
}

export abstract class ChatModel extends WorkspaceBase {
  abstract readonly eosIds: Set<number>;
  abstract readonly cfg: CommonModelConfig;

  constructor(glm: DeviceOps) {
    super(glm);
  }

  abstract createChatCache(maxPages?: number): ChatCache;
  abstract forward(state: BatchState): Tensor;

  prefillBatchPlanHook(_inputIdsList: number[][], _seqLens: number[], _totalTokens: number, _startPos: number[], _cache: ChatCache): void {}

  protected abstract loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): void;  protected tieWeights(): void {}

  protected loadWeights(modelDir: string): void {
    const stFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.safetensors') || f.endsWith('.safetensors.json'));
    const shards: string[] = [];
    if (stFiles.some(f => f === 'model.safetensors')) {
      shards.push(path.join(modelDir, 'model.safetensors'));
    } else {
      const indexFile = stFiles.find(f => f.endsWith('.json'));
      if (indexFile) {
        const idx = JSON.parse(fs.readFileSync(path.join(modelDir, indexFile), 'utf-8'));
        for (const f of Object.keys(idx.weight_map ?? idx)) {
          if (f.endsWith('.safetensors') && !shards.includes(path.join(modelDir, f))) {
            shards.push(path.join(modelDir, f));
          }
        }
      } else {
        shards.push(...stFiles.filter(f => f.endsWith('.safetensors')).map(f => path.join(modelDir, f)));
      }
    }

    for (const stPath of shards) {
      const st = SafeTensorFile.open(stPath);
      const mmapPtr = mmapOpen(stPath);
      const fileSize = fs.statSync(stPath).size;

      for (const name of st.tensorNames()) {
        this.loadTensor(name, st.meta(name), st, mmapPtr);
      }

      this.glm.synchronize();
      st.close();
      mmapClose(mmapPtr, fileSize);
    }

    this.tieWeights();
    this.freeze();
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
