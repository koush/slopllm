import { AutoTokenizer } from "@huggingface/transformers/tokenizers";
import fs from "node:fs";
import path from "node:path";
import { ChatModelParser, DefaultChatModelParser } from "./chat-model-parser";
import { DeviceOps } from "./device_ops";
import { type ExecutionPlan, ExecutionState, type ExecutionWorkspace } from "./execution-workspace";
import { f32ToBf16Bytes } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { mmapClose, mmapOpen } from "./native-addon";
import { PagedKVCache } from "./paged_kv";
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

export interface GenerationConfig {
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
}

export interface ChatCache extends Disposable {
  getPagedKV(): PagedKVCache;
  reset(batchSize: number): void;
  free(): void;
  prefixMatch(seqIdx: number, inputIds: number[]): number[];
  reportTokens(seqIdx: number, tokens: number[]): void;
  prefillBatchPlanHook?(_batchSize: number, _seqLens: number[], _totalTokens: number, _startPos: number[], _cache: ChatCache): void;
}

export interface MtpDraftBatch {
  targetTokens: number[];
  treeTokens: number[][];
  topks: readonly number[];
}

export interface MtpStepResult {
  draft: MtpDraftBatch;
  tokens: number[][];
  numAccepted: number[];
  numDraftTokens: number;
}

export interface CommonModelConfig {
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numNextNPredictLayers?: number;
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
  generationConfig?: GenerationConfig;
}

export function loadGenerationConfig(modelDir: string): GenerationConfig {
  const readJson = (filename: string): Record<string, unknown> => {
    const filePath = path.join(modelDir, filename);
    return fs.existsSync(filePath) ? JSON.parse(fs.readFileSync(filePath, "utf-8")) : {};
  };
  const raw = { ...readJson("config.json"), ...readJson("generation_config.json") } as Record<string, any>;
  return {
    temperature: typeof raw.temperature === "number" ? raw.temperature : undefined,
    topP: typeof raw.top_p === "number" ? raw.top_p : undefined,
    topK: typeof raw.top_k === "number" ? raw.top_k : undefined,
    repetitionPenalty: typeof raw.repetition_penalty === "number" ? raw.repetition_penalty : undefined,
  };
}

export type Tokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;

export interface ChatTemplateKwargs {
  enable_thinking?: boolean;
  [key: string]: unknown;
}

export abstract class ChatModel extends WorkspaceBase {
  abstract readonly eosIds: Set<number>;
  abstract readonly cfg: CommonModelConfig;
  tokenizer!: Tokenizer;

  constructor(glm: DeviceOps) {
    super(glm);
  }

  abstract createChatCache(maxPages?: number, maxBatch?: number, maxSeqLen?: number, pageSize?: number): ChatCache;
  abstract forwardModel(state: ExecutionState): Tensor;
  planPrefillMtpDraftExtend?(ws: ExecutionWorkspace, cache: ChatCache, inputIds: number[][], topks: readonly number[]): ExecutionPlan<MtpDraftBatch>;
  planTargetVerification?(ws: ExecutionWorkspace, cache: ChatCache, draft: MtpDraftBatch): ExecutionPlan<MtpStepResult>;

  prepareMtpInput(_cache: ChatCache, inputIdsList: number[][]): number[][] {
    return inputIdsList.map(inputIds => [...inputIds]);
  }

  createParser(_chatTemplateKwargs: ChatTemplateKwargs = {}): ChatModelParser {
    return new DefaultChatModelParser(this.tokenizer);
  }

  forward(state: ExecutionState): Tensor {
    using _tracker = state.ws.startTracking();
    return this.forwardModel(state).removeTracking();
  }

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

  protected swiGluMlpWeights(pfx: string): { gate: Tensor, up: Tensor, down: Tensor } {
    return {
      gate: this.tensors.get(`${pfx}.gate_proj.weight`)!,
      up: this.tensors.get(`${pfx}.up_proj.weight`)!,
      down: this.tensors.get(`${pfx}.down_proj.weight`)!,
    };
  }

  protected abstract loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): Promise<void>;

  protected async loadWeights(modelDir: string): Promise<void> {
    const allFiles = fs.readdirSync(modelDir);
    const shards: string[] = [];
    if (allFiles.includes('model.safetensors')) {
      shards.push(path.join(modelDir, 'model.safetensors'));
    } else {
      const indexFile = allFiles.find(f => f.endsWith('.index.json') || f.endsWith('.safetensors.json'));
      if (indexFile) {
        const idx = JSON.parse(fs.readFileSync(path.join(modelDir, indexFile), 'utf-8'));
        const weightMap = idx.weight_map ?? idx;
        for (const f of Object.values(weightMap)) {
          if (typeof f === 'string' && f.endsWith('.safetensors') && !shards.includes(path.join(modelDir, f))) {
            shards.push(path.join(modelDir, f));
          }
        }
      } else {
        shards.push(...allFiles.filter(f => f.endsWith('.safetensors')).map(f => path.join(modelDir, f)));
      }
    }

    const openShards = shards.map(stPath => {
      const st = SafeTensorFile.open(stPath);
      const mmapPtr = mmapOpen(stPath);
      const fileSize = fs.statSync(stPath).size;
      return { st, mmapPtr, fileSize };
    });

    const loadBatchSize = parseInt(process.env.GLM_LOAD_BATCH_SIZE ?? "8", 10);
    for (const { st, mmapPtr } of openShards) {
      const names = st.tensorNames();
      for (let i = 0; i < names.length; i += loadBatchSize) {
        await Promise.all(
          names.slice(i, i + loadBatchSize).map(name =>
            this.loadTensor(name, st.meta(name), st, mmapPtr)
          )
        );
      }
    }

    this.glm.synchronize();

    for (const { st, mmapPtr, fileSize } of openShards) {
      st.close();
      mmapClose(mmapPtr, fileSize);
    }
  }

  protected async fromPretrained(modelDir: string, tokenizerRepo: string): Promise<void> {
    await this.loadWeights(modelDir);
    const tokenizerDir = fs.existsSync(path.join(modelDir, "tokenizer_config.json"))
      ? modelDir
      : resolveModelPath(tokenizerRepo);
    this.tokenizer = await AutoTokenizer.from_pretrained(tokenizerDir, { local_files_only: true });

    // Served checkpoints may override the tokenizer repository's template.
    const chatTemplatePath = [modelDir, tokenizerDir]
      .map(dir => path.join(dir, "chat_template.jinja"))
      .find(candidate => fs.existsSync(candidate));
    if (chatTemplatePath) {
      // @huggingface/jinja requires bracket syntax for numeric member access.
      // Some upstream templates use Python/Jinja-style `content.0` instead.
      this.tokenizer.chat_template = fs.readFileSync(chatTemplatePath, "utf-8")
        .replace(/\.(\d+)\b/g, "[$1]");
    }
    this.cfg.generationConfig = loadGenerationConfig(modelDir);
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
