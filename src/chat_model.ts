import { AutoTokenizer } from "@huggingface/transformers/tokenizers";
import fs from "node:fs";
import path from "node:path";
import { ChatModelParser, DefaultChatModelParser } from "./chat-model-parser";
import { DeviceOps } from "./device_ops";
import { EagerExecution, type ExecutionManager } from "./execution-manager";
import { ExecutionState, type ExecutionWorkspace } from "./execution-workspace";
import { f32ToBf16Bytes } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { mmapClose, mmapOpen } from "./native-addon";
import { PagedKVCache } from "./paged_kv";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
import { Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";
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
  maxNewTokens?: number;
}

export interface ChatCache extends Disposable {
  getPagedKV(): PagedKVCache;
  reset(batchSize: number): void;
  free(): void;
  prefixMatch(seqIdx: number, inputIds: number[]): number[];
  reportTokens(seqIdx: number, tokens: number[], targetToken?: number): void;
  prefillBatchPlanHook?(_batchSize: number, _seqLens: number[], _totalTokens: number, _startPos: number[], _cache: ChatCache): void;
}

export interface ChunkedPrefillPlan {
  states: ExecutionState[];
  /** Returns selected targets in batch order; rows without input return undefined. */
  generator: Generator<void, (number | undefined)[], void>;
  /** Selected execution inputs, including any replayed overlap tokens. */
  prefillInputIdsList: number[][];
  /** Unconsumed caller input; adopt after successful execution and reporting. */
  remainingInputIdsList: number[][];
  /** Called by the executor after successful eager execution. */
  reportTokens(): void;
}

export interface MtpProposal {
  // Owned host snapshots: B buffers each, with depth-major [D, capacity] F32/I32 LE rows.
  // Both host arrays are empty in device mode; rows map to the owner's original draft batch.
  probabilities: Buffer[];
  tokenIds: Buffer[];
  capacity: number;
  device?: { owner: object; generation: number; rows: number[] };
}

export interface MtpDraftBatch {
  targetTokens: number[];
  treeTokens: number[][];
  topks: readonly number[];
  proposal?: MtpProposal;
}

export interface MtpStepResult {
  draft: MtpDraftBatch;
  tokens: number[][];
  numAccepted: number[];
  numDraftTokens: number;
}

export interface MtpDecodeStepResult extends Omit<MtpStepResult, "draft"> {
  /** Whether this is a graph warmup/capture step. */
  warmup: boolean;
}

export interface TokenSelector {
  selectTarget(logits: Tensor): Tensor;
  captureKey?: string | number;
  /** False/omitted retains token-comparison (greedy/tree) verification. */
  readonly mtpEnabled?: boolean;
  readonly mtpCaptureKey?: string | number;
  /** All MTP methods are required when mtpEnabled is true. */
  prepareDraft?(batchSize: number, depth: number): void;
  sampleDraft?(logits: Tensor, depth: number): Tensor;
  finishDraft?(): MtpProposal;
  prepareVerification?(draft: MtpDraftBatch): void;
  /** Capture-safe preparation for drafts generated on-device in the same graph. */
  prepareVerificationFromDevice?(draftTokens: Tensor, batchSize: number): void;
  verify?(logits: Tensor): { tokens: Tensor; numAccepted: Tensor };
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
  maxPositionEmbeddings?: number;
  generationConfig?: GenerationConfig;
}

export function loadMaxPositionEmbeddings(modelDir: string): number | undefined {
  const filePath = path.join(modelDir, "config.json");
  if (!fs.existsSync(filePath)) return undefined;
  const raw = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  const config = raw.text_config ?? raw;
  return typeof config.max_position_embeddings === "number" ? config.max_position_embeddings : undefined;
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
    maxNewTokens: typeof raw.max_new_tokens === "number" ? raw.max_new_tokens : undefined,
  };
}

export type Tokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;

export interface ChatTemplateKwargs {
  continue_final_message?: boolean;
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
  abstract forwardPhased(state: ExecutionState): Generator<void, Tensor, void>;

  protected runPhased<T>(generator: Generator<void, T, void>): T {
    while (true) {
      const iter = generator.next();
      if (iter.done) return iter.value;
    }
  }

  forwardModel(state: ExecutionState): Tensor {
    return this.runPhased(this.forwardPhased(state));
  }

  protected createChunkedPrefillPlan(states: ExecutionState[], cache: ChatCache,
    inputIdsList: readonly number[][], generator: Generator<void, (number | undefined)[], void>): ChunkedPrefillPlan {
    let targetTokens: (number | undefined)[] = [];
    const reportTokens = () => {
      inputIdsList.forEach((ids, index) => {
        if (ids.length) {
          cache.reportTokens(index, ids, targetTokens[index]);
        }
      });
    };
    return {
      states,
      reportTokens,
      generator: (function* () {
        const selected = yield* generator;
        targetTokens = inputIdsList.map((ids, index) => ids.length ? selected[index] : undefined);
        return targetTokens;
      })(),
      prefillInputIdsList: inputIdsList.map(ids => [...ids]),
      remainingInputIdsList: inputIdsList.map(() => []),
    };
  }

  protected planPhasedPrefill(ws: ExecutionWorkspace, cache: ChatCache,
    inputIdsList: readonly number[][],
    samplingPolicy: TokenSelector = { selectTarget: logits => logits.argmax() },
    forward: (state: ExecutionState, nextState?: ExecutionState) => Generator<void, Tensor, void>
      = state => this.forwardPhased(state)): ChunkedPrefillPlan {
    if (inputIdsList.length !== 1) {
      throw new Error("Phased prefill requires batch size 1");
    }
    let remainingA = Math.floor(inputIdsList.reduce((sum, ids) => sum + ids.length, 0) / 2);
    const inputA: number[][] = [];
    const inputB: number[][] = [];
    for (const ids of inputIdsList) {
      const take = Math.min(ids.length, remainingA);
      inputA.push(ids.slice(0, take));
      inputB.push(ids.slice(take));
      remainingA -= take;
    }
    // Zero-query rows preserve the original cache sequence indices in both plans.
    const stateA = ws.planPrefill(this, inputA.length, inputA.map(ids => ids.length), cache);
    stateA.setInput(inputA);
    const stateB = ws.planPrefill(this, inputB.length, inputB.map(ids => ids.length), cache);
    stateB.setInput(inputB);
    const self = this;
    const targetTokens: (number | undefined)[] = [];
    return this.createChunkedPrefillPlan([stateA, stateB], cache, inputIdsList, (function* () {
        const a = forward(stateA, stateB);
        const b = forward(stateB);
        using hiddenA = new UsingHolder<Tensor>(undefined!);
        using hiddenB = new UsingHolder<Tensor>(undefined!);
        const advance = (generator: Generator<void, Tensor, void>, hidden: UsingHolder<Tensor>) => {
          const result = generator.next();
          if (result.done && result.value) {
            hidden.replace(result.value);
          }
          return !!result.done;
        };
        let doneA = false;
        let doneB = false;
        try {
          // A stays one phase ahead so B can attend to A's newly written KV.
          doneA = advance(a, hiddenA);
          while (!doneA) {
            using streamB = self.glm.withStream(() => {
              if (!doneB) {
                doneB = advance(b, hiddenB);
              }
            });
            try {
              doneA = advance(a, hiddenA);
            } finally {
              streamB.streamWaitEvent();
            }
            yield;
          }
          while (!doneB) {
            doneB = advance(b, hiddenB);
            if (!doneB) {
              yield;
            }
          }
          if (hiddenB.value) {
            using logits = stateB.computeLogits(hiddenB.value, self);
            using selected = samplingPolicy.selectTarget(logits);
            targetTokens.push(...selected.readInt32LEArray());
          }
        } finally {
          try {
            if (!doneB) {
              b.return(undefined!);
            }
          } finally {
            if (!doneA) {
              a.return(undefined!);
            }
          }
        }
        return targetTokens;
      })());
  }

  protected selectPrefillChunk(inputIdsList: readonly number[][], chunkSize: number,
    overlapTokens: readonly (number | undefined)[] = []): {
      prefillInputIdsList: number[][];
      remainingInputIdsList: number[][];
    } {
    if (!Number.isSafeInteger(chunkSize) || chunkSize < 1) {
      throw new Error(`Invalid prefill chunk size: ${chunkSize}`);
    }
    let budget = chunkSize;
    const prefillInputIdsList: number[][] = [];
    const remainingInputIdsList: number[][] = [];
    for (const [index, ids] of inputIdsList.entries()) {
      const overlap = overlapTokens[index];
      const take = Math.min(ids.length, Math.max(0, budget - (overlap === undefined ? 0 : 1)));
      const input = ids.slice(0, take);
      if (take && overlap !== undefined) {
        input.unshift(overlap);
      }
      prefillInputIdsList.push(input);
      remainingInputIdsList.push(ids.slice(take));
      budget -= input.length;
    }
    if (budget === chunkSize && inputIdsList.some(ids => ids.length)) {
      throw new Error(`Prefill chunk size ${chunkSize} cannot fit input and its required overlap`);
    }
    return { prefillInputIdsList, remainingInputIdsList };
  }

  planChunkedPrefill(ws: ExecutionWorkspace, cache: ChatCache,
    inputIdsList: readonly number[][], chunkSize = 8192,
    samplingPolicy: TokenSelector = { selectTarget: logits => logits.argmax() }): ChunkedPrefillPlan {
    const chunk = this.selectPrefillChunk(inputIdsList, chunkSize);
    const plan = this.planPrefillChunk(ws, cache, chunk.prefillInputIdsList, samplingPolicy);
    plan.remainingInputIdsList = chunk.remainingInputIdsList;
    return plan;
  }

  protected planPrefillChunk(ws: ExecutionWorkspace, cache: ChatCache,
    inputIdsList: readonly number[][],
    samplingPolicy: TokenSelector = { selectTarget: logits => logits.argmax() }): ChunkedPrefillPlan {
    const batchSize = inputIdsList.length;
    const seqLens = inputIdsList.map(ids => ids.length);
    const totalTokens = seqLens.reduce((a, b) => a + b, 0);
    if (!totalTokens) {
      return this.createChunkedPrefillPlan([], cache, inputIdsList, (function* () { return []; })());
    }

    if (this.glm.worldSize > 1 && batchSize === 1 && totalTokens >= 4096 && process.env.GLM_PHASED_PREFILL !== "0") {
      return this.planPhasedPrefill(ws, cache, inputIdsList, samplingPolicy);
    }

    const state = ws.planPrefill(this, batchSize, seqLens, cache);
    state.setInput(inputIdsList);
    const self = this;
    return this.createChunkedPrefillPlan([state], cache, inputIdsList, (function* () {
        using hiddenStates = yield* self.forwardPhased(state);
        using logits = state.computeLogits(hiddenStates, self);
        using selected = samplingPolicy.selectTarget(logits);
        return selected.readInt32LEArray();
      })());
  }

  async executePrefill(ws: ExecutionWorkspace, cache: ChatCache,
    inputIdsList: readonly number[][],
    executionManager: ExecutionManager = new EagerExecution(), chunkSize = 8192,
    samplingPolicy: TokenSelector = { selectTarget: logits => logits.argmax() }): Promise<{
      warmup: boolean;
      targetTokens: (number | undefined)[];
      prefillInputIdsList: number[][];
      remainingInputIdsList: number[][];
    }> {
    ws.assertClear();
    ws.clearTracking();
    const plan = this.planChunkedPrefill(ws, cache, inputIdsList, chunkSize, samplingPolicy);
    const { states, generator } = plan;
    let warmup: boolean;
    let targetTokens: (number | undefined)[];
    {
      // Prefill is always eager, including single-state chunks.
      const execution = executionManager.execute({ states, inputs: {}, key: [] }, () => {
        try {
          while (true) {
            const iter = generator.next();
            if (iter.done) {
              return iter.value;
            }
          }
        }
        catch (e) {
          generator.return([]);
          throw e;
        }
      });

      warmup = execution.warmup;
      targetTokens = execution.result;
      await this.glm.synchronizeAsync();
      plan.reportTokens();
    }
    ws.assertClear();
    ws.clearTracking();
    return { warmup, targetTokens, prefillInputIdsList: plan.prefillInputIdsList, remainingInputIdsList: plan.remainingInputIdsList };
  }

  /** Read the caller-published pending token without modifying committed model state. */
  protected async prepareDecodeInput(_ws: ExecutionWorkspace, cache: ChatCache,
    _samplingPolicy: TokenSelector): Promise<number[]> {
    const sequences = cache.getPagedKV().sequences;
    if (!sequences.length || sequences.some(sequence => sequence.targetToken === undefined
      || sequence.reportedTokenCount() !== sequence.allocLen)) {
      throw new Error("Decode requires committed history and a pending target token for every sequence");
    }
    return sequences.map(sequence => sequence.targetToken!);
  }

  /** Consume the caller-published pending target and yield newly selected tokens. */
  async *generateDecode(
    ws: ExecutionWorkspace, cache: ChatCache,
    executionManager: ExecutionManager = new EagerExecution(),
    samplingPolicy: TokenSelector = { selectTarget: logits => logits.argmax() },
  ): AsyncGenerator<MtpStepResult & { warmup: boolean }, void, void> {
    const pagedKV = cache.getPagedKV();
    const sequences = pagedKV.sequences.slice();
    const batchSize = sequences.length;
    ws.assertClear();
    let nextTokens = await this.prepareDecodeInput(ws, cache, samplingPolicy);
    try {
      while (true) {
        ws.assertClear();
        ws.clearTracking();
        if (pagedKV.sequences.length !== batchSize || sequences.some((sequence, batch) =>
          pagedKV.sequences[batch] !== sequence)) {
          throw new Error("Decode batch changed; close and restart the generator");
        }
        const state = ws.planDecode(this, batchSize, cache, executionManager.captureEnabled);
        const inputTokens = nextTokens;
        state.setInput([nextTokens]);
        let warmup: boolean;
        {
          const captureKey = ["decode", samplingPolicy.captureKey ?? "greedy"];
          const execution = executionManager.execute({ states: [state], inputs: {}, key: captureKey }, () => {
            using hidden = this.forwardModel(state);
            using logits = state.computeLogits(hidden, this);
            return samplingPolicy.selectTarget(logits);
          });
          warmup = execution.warmup;
          using selected = execution.result;
          await this.glm.synchronizeAsync();
          nextTokens = selected.readInt32LEArray();
        }
        inputTokens.forEach((token, batch) => cache.reportTokens(batch, [token], nextTokens[batch]));
        ws.assertClear();
        ws.clearTracking();
        // Keep the next input independent of the caller's yielded array.
        yield {
          draft: { targetTokens: [...nextTokens], treeTokens: nextTokens.map(() => []), topks: [] },
          tokens: nextTokens.map(token => [token]),
          numAccepted: Array(batchSize).fill(0),
          numDraftTokens: 0,
          warmup,
        };
      }
    } finally {
      await this.glm.synchronizeAsync();
    }
  }

  generateMtpDecode?(ws: ExecutionWorkspace, cache: ChatCache, topks: readonly number[], executionManager?: ExecutionManager, samplingPolicy?: TokenSelector): AsyncGenerator<MtpDecodeStepResult, void, void>;

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

    await this.glm.synchronizeAsync();

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
    this.cfg.maxPositionEmbeddings = loadMaxPositionEmbeddings(modelDir);
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
