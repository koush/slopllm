import fs from "node:fs";
import path from "node:path";
import { DeviceOps } from "./device_ops";
import { f32ToBf16Bytes, mmapOpen, mmapClose } from "./glm_ops";
import { PagedKVCache, ExecutionState } from "./paged_kv";
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

export interface ChatCache extends Disposable {
  getPagedKV(): PagedKVCache;
  reset(batchSize: number): void;
  free(): void;
  prefixMatch(seqIdx: number, inputIds: number[]): number[];
  appendTokens(seqIdx: number, tokens: number[]): void;
}

export interface CommonModelConfig {
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  rmsNormEps: number;
  vocabSize: number;
  tieWordEmbeddings: boolean;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  ropeTheta: number;
  numKeyValueGroups: number;
  scaling: number;
  kvLoraRank?: number;
  qkRopeHeadDim?: number;
}

export abstract class ChatModel extends WorkspaceBase {
  abstract readonly eosIds: Set<number>;
  abstract readonly cfg: CommonModelConfig;

  constructor(glm: DeviceOps) {
    super(glm);
  }

  abstract createChatCache(maxPages?: number, contextParallel?: boolean): ChatCache;
  abstract forward(state: ExecutionState): Tensor;

  prefillBatchPlanHook(_inputIdsList: number[][], _seqLens: number[], _totalTokens: number, _startPos: number[], _cache: ChatCache): void {}

  protected initInvFreq(ropeDim: number, ropeTheta: number): Tensor {
    const halfDim = ropeDim / 2;
    const invFreqF32 = new Float32Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
      invFreqF32[i] = 1.0 / Math.pow(ropeTheta, (2 * i) / ropeDim);
    }
    const invFreq = this.alloc([halfDim], "BF16", "invFreq");
    invFreq.h2d(f32ToBf16Bytes(invFreqF32));
    return invFreq;
  }

  protected swiGluMlp(normed: Tensor, pfx: string, intermediateSize: number, BS: number): Tensor {
    using upStream = this.glm.withStream(() => normed.linear(this.tensors.get(`${pfx}.mlp.up_proj.weight`)!, BS));
    using upBuf = upStream.result;
    using gateBuf = normed.linear(this.tensors.get(`${pfx}.mlp.gate_proj.weight`)!, BS);
    upStream.streamWaitEvent();
    using siluBuf = gateBuf.siluAndMul(upBuf, intermediateSize, BS);
    return siluBuf.linear(this.tensors.get(`${pfx}.mlp.down_proj.weight`)!, BS);
  }

  protected computeLogits(normed: Tensor, state: ExecutionState): Tensor {
    const hs = this.cfg.hiddenSize;
    const ws = state.ws;
    const batchSize = state.batchSize;
    let logitsBuf: Tensor;
    if (state.isDecode) {
      logitsBuf = normed.linear(this.tensors.get("lm_head.weight")!, batchSize);
    } else {
      using hiddenLast = normed.indexSelect(ws.lastIdx, hs, batchSize);
      logitsBuf = hiddenLast.linear(this.tensors.get("lm_head.weight")!, batchSize);
    }
    return logitsBuf.removeTracking();
  }

  protected tieEmbeddingToLmHead(embedName: string): void {
    if (this.cfg.tieWordEmbeddings && !this.tensors.has("lm_head.weight")) {
      const embedTensor = this.tensors.get(embedName);
      if (embedTensor) this.tensors.set("lm_head.weight", embedTensor);
    }
  }

  protected abstract loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): Promise<void>;

  protected async loadWeights(modelDir: string): Promise<void> {
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

    const openShards = shards.map(stPath => {
      const st = SafeTensorFile.open(stPath);
      const mmapPtr = mmapOpen(stPath);
      const fileSize = fs.statSync(stPath).size;
      return { st, mmapPtr, fileSize };
    });

    for (const { st, mmapPtr } of openShards) {
      for (const name of st.tensorNames()) {
        console.log('Loading tensor', name);
        await this.loadTensor(name, st.meta(name), st, mmapPtr);
      }
    }

    this.glm.synchronize();

    for (const { st, mmapPtr, fileSize } of openShards) {
      st.close();
      mmapClose(mmapPtr, fileSize);
    }
  }

  protected async fromPretrained(modelDir: string): Promise<void> {
    await this.loadWeights(modelDir);
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
