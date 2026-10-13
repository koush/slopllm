import { AutoTokenizer } from "@huggingface/transformers/tokenizers";
import fs from "node:fs";
import path from "node:path";
import { ChatModelParser, DefaultChatModelParser } from "./chat-model-parser";
import {
  openCheckpointSource,
  resolveCheckpointConfigDir,
  type CheckpointSource,
  type CheckpointTensor,
} from "./checkpoint_source";
import { DeviceOps, type StridedMmap } from "./device_ops";
import { EagerExecution, type ExecutionManager } from "./execution-manager";
import { ExecutionState, type ExecutionWorkspace } from "./execution-workspace";
import { f32ToBf16Bytes } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { mmapClose, mmapOpen } from "./native-addon";
import { PagedKVCache } from "./paged_kv";
import type { TensorMeta } from "./safetensors";
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
  indexTopk?: number;
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

/** Fix A (tail-chunk merge), CORRECTED per Codex consult #8
 *  (verdict-lowctx-20261013.md Q3a/Q3c/Q3f): a request left with an actual
 *  tail of at most this many tokens after the greedy take is absorbed into
 *  the current chunk ONLY through the caller-approved, capacity-guarded
 *  overflow path below. Inclusive endpoint (<= 256 merges, 257 does not).
 *  0 reproduces the pre-Fix-A splitting exactly (the fixture's split arm and
 *  the per-call reproduction hook). One shared bound: the PREFILL_CHUNK_TRACE
 *  integer mirror (GenerationScheduler.estimateRemainingPrefillChunks) runs
 *  the SAME pure policy — never diverge the two. */
export const PREFILL_TAIL_MERGE_MAX = 256;

/** Serving-path runtime override (arm toggle for the maintenance-window
 *  validation; RUNBOOK-window1 §7.2 arms 0/1/2 switch WITHOUT rebuilds).
 *  Read at chunk-plan build time — where ChatModel.selectPrefillChunk and the
 *  scheduler mirror resolve their tail ceiling — so one build serves every
 *  arm:
 *    SLOPLLM_TAIL_MERGE_MAX=0    arm 0 / reproduction hook: pre-Fix-A
 *                                splitting exactly (same as per-call
 *                                `tailMergeMax: 0`);
 *    SLOPLLM_TAIL_MERGE_MAX=256  arm 1 default (inclusive endpoint);
 *    unset                       PREFILL_TAIL_MERGE_MAX (shipped state).
 *  Values must be safe integers >= 0; anything else fails loudly at plan
 *  time. The selector and the mirror share this one resolver — never let the
 *  two diverge (verdict Q3c). */
export function prefillTailMergeMaxSetting(): number {
  const raw = process.env.SLOPLLM_TAIL_MERGE_MAX;
  if (raw === undefined || raw === "") {
    return PREFILL_TAIL_MERGE_MAX;
  }
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`Invalid SLOPLLM_TAIL_MERGE_MAX: ${raw}`);
  }
  return value;
}

/** The qualified execution-row budget the 256-token ceiling was sized against
 *  (verdict Q3f: 256/8192 ~= 3.1%; other chunk sizes are NOT qualified — with
 *  small C a fixed 256 would invalidate existing chunk-budget behavior). */
export const PREFILL_TAIL_MERGE_QUALIFIED_CHUNK_SIZE = 8192;

/** Phased-prefill row threshold — must match the actual planner predicates
 *  (planPrefillChunk in this file and the GLM override in glm51_model.ts,
 *  `total >= 4096`). Kept as a named anchor for the overflow eligibility rule;
 *  do not change one site without the others. */
export const PHASED_PREFILL_MIN_ROWS = 4096;

/** Inputs to the one shared tail-merge/overflow rule (selectPrefillTakes).
 *  Prediction (scheduler mirror) and actual selection must supply the SAME
 *  values for every field (verdict Q3c). */
export interface PrefillTailMergeOptions {
  /** Inclusive tail ceiling T; 0 disables all merging (pre-Fix-A behavior). */
  tailMergeMax?: number;
  /** Workspace capacity (ws.maxSeqLen): ceil(mergedRows/2) must fit in it. */
  maxSeqLen?: number;
  /** ORIGINAL batch length (inputIdsList.length) — EMPTY ROWS COUNT. A batch
   *  of [8198, []] has batchLength 2 and is never eligible (verdict Q3a/Q3c). */
  batchLength?: number;
  /** Caller-approved overflow: true only where the qualified GLM MTP planner
   *  will plan the merged chunk (glm51 planChunkedPrefill). Everything else —
   *  unphased, batched, other-model, unqualified chunk sizes — passes false or
   *  nothing and keeps today's greedy slicing. */
  mtp?: boolean;
  /** Chunk budget the ceiling is qualified for (default 8192). */
  qualifiedChunkSize?: number;
}

export interface PrefillTailMergeContext extends PrefillTailMergeOptions {
  /** C: per-chunk execution-token budget, prepended overlap included. */
  chunkSize: number;
}

/** The ONE overflow-eligibility rule (verdict Q3a "mandated first
 *  implementation"), applied identically by ChatModel.selectPrefillChunk and
 *  the scheduler mirror. Overflow is a caller-approved option ONLY for the
 *  qualified GLM single-request phased path:
 *    - original inputIdsList.length === 1 (batchLength; empty rows count),
 *    - the ACTUAL planner selects phased execution for the merged row
 *      (GLM MTP planner predicate: batch 1 + SLOPLLM_PHASED_PREFILL !== "0"
 *      + merged rows >= 4096 — see planPrefillChunk in glm51_model.ts),
 *    - ceil(mergedExecutionRows/2) <= ws.maxSeqLen (planPrefill rejects total
 *      execution rows above capacity; the phased halves must each fit),
 *    - the chunk budget is the qualified one (8192 rows; verdict Q3f).
 *  Decided BEFORE any MTP rewind; never retried on a mutated cache. */
export function prefillTailMergeOverflowEligible(ctx: PrefillTailMergeContext,
  mergedExecutionRows: number): boolean {
  return (ctx.tailMergeMax ?? prefillTailMergeMaxSetting()) > 0
    && (ctx.batchLength ?? -1) === 1
    && ctx.mtp === true
    && ctx.chunkSize === (ctx.qualifiedChunkSize ?? PREFILL_TAIL_MERGE_QUALIFIED_CHUNK_SIZE)
    && process.env.SLOPLLM_PHASED_PREFILL !== "0"
    && mergedExecutionRows >= PHASED_PREFILL_MIN_ROWS
    && Math.ceil(mergedExecutionRows / 2) <= (ctx.maxSeqLen ?? ctx.chunkSize);
}

/** Shared pure length-selection policy for ChatModel.selectPrefillChunk and
 *  GenerationScheduler.estimateRemainingPrefillChunks (verdict Q3c: one
 *  algorithm, not two). Returns the per-row greedy take, with the corrected
 *  bounded overflow applied where eligible. Executed rows per row =
 *  take + overlapCost (the prepended replay overlap); the shared budget counts
 *  them exactly as the pre-Fix-A slicing did. */
export function selectPrefillTakes(inputLengths: readonly number[],
  overlapCosts: readonly number[],
  ctx: PrefillTailMergeContext): number[] {
  if (!Number.isSafeInteger(ctx.chunkSize) || ctx.chunkSize < 1) {
    throw new Error(`Invalid prefill chunk size: ${ctx.chunkSize}`);
  }
  const tailMergeMax = ctx.tailMergeMax ?? prefillTailMergeMaxSetting();
  if (!Number.isSafeInteger(tailMergeMax) || tailMergeMax < 0) {
    throw new Error(`Invalid prefill tail merge max: ${tailMergeMax}`);
  }
  let budget = ctx.chunkSize;
  const takes: number[] = [];
  for (let i = 0; i < inputLengths.length; i++) {
    const length = inputLengths[i];
    const overlapCost = i < overlapCosts.length ? overlapCosts[i] : 0;
    let take = Math.min(length, Math.max(0, budget - overlapCost));
    // Corrected bounded merge (verdict Q3a): positive take, an ACTUAL tail
    // (0 < length - take <= T), and ONE global executed-row cap including the
    // prepended overlap: Eused + length + overlapCost <= C + T with
    // Eused = C - budget. No budget replenishment and no per-request T: a
    // negative budget after an accepted merge simply clamps later ordinary
    // takes to zero. take === 0 is deliberately NOT a merge shortcut — it
    // would bypass the no-progress throw and rewind a formerly deferred row.
    const remainder = length - take;
    const mergedExecutionRows = length + overlapCost;
    if (take > 0 && remainder > 0 && remainder <= tailMergeMax
      && (ctx.chunkSize - budget) + mergedExecutionRows <= ctx.chunkSize + tailMergeMax
      && prefillTailMergeOverflowEligible(ctx, mergedExecutionRows)) {
      take = length;
    }
    takes.push(take);
    budget -= take ? take + overlapCost : 0;
  }
  return takes;
}

export abstract class ChatModel extends WorkspaceBase {
  abstract readonly eosIds: Set<number>;
  abstract readonly cfg: CommonModelConfig;
  tokenizer!: Tokenizer;

  constructor(ops: DeviceOps) {
    super(ops);
  }

  abstract createChatCache(maxPages?: number, maxBatch?: number, maxSeqLen?: number, pageSize?: number, pinned?: boolean): ChatCache;
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
            using streamB = self.ops.withStream(() => {
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
    overlapTokens: readonly (number | undefined)[] = [],
    tailMerge: PrefillTailMergeOptions = {}): {
      prefillInputIdsList: number[][];
      remainingInputIdsList: number[][];
    } {
    const ctx: PrefillTailMergeContext = {
      ...tailMerge,
      chunkSize,
      batchLength: tailMerge.batchLength ?? inputIdsList.length,
    };
    // The one shared pure policy (selectPrefillTakes) owns the greedy take and
    // the corrected bounded overflow; this method only assembles slices. The
    // overflow eligibility (caller-approved GLM single-request phased path,
    // capacity guard) is decided HERE — before any MTP rewind in the caller —
    // and is never retried on a mutated cache (verdict Q3a).
    const takes = selectPrefillTakes(inputIdsList.map(ids => ids.length),
      overlapTokens.map(overlap => overlap === undefined ? 0 : 1), ctx);
    let budget = chunkSize;
    const prefillInputIdsList: number[][] = [];
    const remainingInputIdsList: number[][] = [];
    for (const [index, ids] of inputIdsList.entries()) {
      const overlap = overlapTokens[index];
      const take = takes[index];
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

    if (this.ops.worldSize > 1 && batchSize === 1 && totalTokens >= 4096 && process.env.SLOPLLM_PHASED_PREFILL !== "0") {
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
      await this.ops.synchronizeAsync();
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
          await this.ops.synchronizeAsync();
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
      await this.ops.synchronizeAsync();
    }
  }

  generateMtpDecode?(ws: ExecutionWorkspace, cache: ChatCache, numDraftTokens: number, executionManager?: ExecutionManager, samplingPolicy?: TokenSelector): AsyncGenerator<MtpDecodeStepResult, void, void>;

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

  protected abstract loadTensor(name: string, meta: TensorMeta, source: CheckpointTensor): Promise<void>;

  /**
   * Executor replay (SLOPLLM_SKIP_MMAP_LOAD) reconstructs allocation
   * layouts only: model loading must not read the checkpoint or move data.
   * mmapLoad already skips its transfer at the tensor layer; staged uploads
   * and source reads are guarded here (port plan §11.1). h2d itself is never
   * disabled — executors upload tokens and planning data while serving.
   */
  protected get loadReplay(): boolean {
    return !!process.env.SLOPLLM_SKIP_MMAP_LOAD;
  }

  /** h2d enqueues asynchronously; staging must survive until the next drain. */
  private pendingLoadBuffers: Buffer[] = [];

  protected uploadStaged(tensor: Tensor, bytes: Buffer): void {
    if (this.loadReplay) return;
    this.pendingLoadBuffers.push(bytes);
    tensor.h2d(bytes);
  }

  /** One transfer window at a time (§11.2): complete queued copies before reuse. */
  protected async drainLoadTransfers(): Promise<void> {
    await this.ops.synchronizeAsync();
    this.pendingLoadBuffers.length = 0;
  }

  /**
   * Load one tensor's logical bytes into a device tensor: direct file
   * mapping when the bytes are physically contiguous, bounded host staging
   * otherwise (e.g. CSF scale expansion). Multiple strided specs fill
   * disjoint destination regions of the same tensor.
   */
  protected async loadTensorBytes(tensor: Tensor, source: CheckpointTensor, strided?: StridedMmap): Promise<void> {
    await this.loadTensorStrided(tensor, source, strided ? [strided] : []);
  }

  protected async loadTensorStrided(tensor: Tensor, source: CheckpointTensor, specs: StridedMmap[]): Promise<void> {
    // Allocation-only replay: no source reads, no expansion, no transfers.
    if (this.loadReplay) return;
    const mapping = source.mapping();
    if (mapping) {
      for (const spec of specs) {
        await tensor.mmapLoad(mapping.mmapPtr, mapping.offset, tensor.bytes, spec);
      }
      if (!specs.length) {
        await tensor.mmapLoad(mapping.mmapPtr, mapping.offset, tensor.bytes);
      }
      return;
    }
    const bytes = source.read();
    if (!specs.length) {
      this.uploadStaged(tensor, bytes);
      return;
    }
    const out = Buffer.alloc(tensor.bytes);
    for (const spec of specs) {
      for (let r = 0; r < spec.height; r++) {
        const src = spec.srcOffset + r * spec.srcPitch;
        bytes.copy(out, spec.dstOffset + r * spec.dstPitch, src, src + spec.width);
      }
    }
    this.uploadStaged(tensor, out);
  }

  /** The checkpoint source opened by loadWeights (closed after loading). */
  protected checkpointSource?: CheckpointSource;

  protected async loadWeights(modelDir: string): Promise<void> {
    // Shard discovery and physical plane handling live behind the
    // checkpoint-source abstraction: safetensors roots and CSF containers
    // both expose logical source names with dtype/shape and bounded reads.
    const source = openCheckpointSource(modelDir, { open: mmapOpen, close: mmapClose });
    this.checkpointSource = source;
    const names = source.names();

    const loadBatchSize = parseInt(process.env.SLOPLLM_LOAD_BATCH_SIZE ?? "8", 10);
    const layerRe = /^model\.layers\.(\d+)\./;
    let totalTensors = 0;
    let totalBytes = 0;
    let maxLayer = -1;
    for (const name of names) {
      const meta = source.meta(name);
      totalTensors++;
      totalBytes += meta.dataOffsets[1] - meta.dataOffsets[0];
      const layerMatch = layerRe.exec(name);
      if (layerMatch) maxLayer = Math.max(maxLayer, Number(layerMatch[1]));
    }

    let loadedTensors = 0;
    let loadedBytes = 0;
    let currentLayer = -1;
    const startedMs = performance.now();
    const isTty = process.stderr.isTTY === true;
    let lastDrawMs = 0;
    let lastLineLength = 0;
    const giB = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
    let lastLoggedPercent = -1;
    const drawProgress = (force: boolean): void => {
      const now = performance.now();
      // Progress by bytes: tensor counts race through clusters of tiny
      // scale tensors (e.g. the 57.6k input_scale entries) while the
      // actual transfer barely moves.
      const fraction = totalBytes > 0 ? loadedBytes / totalBytes : 1;
      const layerInfo = currentLayer >= 0 ? ` layer ${currentLayer}/${maxLayer + 1}` : "";
      const status = `${loadedTensors}/${totalTensors} tensors${layerInfo} ${giB(loadedBytes)}/${giB(totalBytes)} ${((now - startedMs) / 1000).toFixed(1)}s`;
      if (isTty) {
        if (!force && now - lastDrawMs < 100) return;
        lastDrawMs = now;
        const width = 24;
        const filled = Math.min(width, Math.round(fraction * width));
        const bar = `${"=".repeat(filled)}${filled < width ? ">" : ""}${" ".repeat(Math.max(0, width - filled - (filled < width ? 1 : 0)))}`;
        const line = `Loading weights [${bar}] ${Math.floor(fraction * 100)}% ${status}`;
        process.stderr.write(`\r${line}${" ".repeat(Math.max(0, lastLineLength - line.length))}`);
        lastLineLength = line.length;
        return;
      }
      const percent = Math.floor(fraction * 100);
      if (!force && percent <= lastLoggedPercent) return;
      if (force && lastLoggedPercent >= 100) return;
      lastLoggedPercent = percent;
      console.log(`Loading weights: ${percent}% ${status}`);
    };

    drawProgress(false);

    try {
      for (let i = 0; i < names.length; i += loadBatchSize) {
        const batch = names.slice(i, i + loadBatchSize);
        await Promise.all(
          batch.map(name =>
            this.loadTensor(name, source.meta(name), source.tensor(name))
          )
        );
        for (const name of batch) {
          const meta = source.meta(name);
          loadedBytes += meta.dataOffsets[1] - meta.dataOffsets[0];
        }
        loadedTensors += batch.length;
        const layerMatch = layerRe.exec(batch[batch.length - 1]);
        if (layerMatch) currentLayer = Math.max(currentLayer, Number(layerMatch[1]));
        // Serialized completion (port plan §11.2): one transfer window at a
        // time. With SLOPLLM_LOAD_BATCH_SIZE=1 this is exactly one logical
        // tensor in flight; TP shard copies for that tensor may overlap.
        await this.drainLoadTransfers();
        drawProgress(false);
      }
    } finally {
      // Drain pending transfers before closing/unmapping the source, on
      // success and on failure (§11.2).
      try {
        await this.drainLoadTransfers();
      } catch {
        // the original load error wins
      }
      source.close();
    }

    drawProgress(true);
    if (isTty) process.stderr.write(`\r${" ".repeat(lastLineLength)}\r`);
    console.log(`Loaded ${totalTensors} weight tensors (${giB(totalBytes)}) in ${((performance.now() - startedMs) / 1000).toFixed(1)}s`);
  }

  protected async fromPretrained(modelDir: string, tokenizerRepo: string): Promise<void> {
    await this.loadWeights(modelDir);
    // CSF containers resolve model config, tokenizer, generation config and
    // the chat template from their metadata/ directory.
    const configDir = resolveCheckpointConfigDir(modelDir);
    const tokenizerDir = fs.existsSync(path.join(configDir, "tokenizer_config.json"))
      ? configDir
      : resolveModelPath(tokenizerRepo);
    this.tokenizer = await AutoTokenizer.from_pretrained(tokenizerDir, { local_files_only: true });

    // Served checkpoints may override the tokenizer repository's template.
    const chatTemplatePath = [configDir, tokenizerDir]
      .map(dir => path.join(dir, "chat_template.jinja"))
      .find(candidate => fs.existsSync(candidate));
    if (chatTemplatePath) {
      // @huggingface/jinja requires bracket syntax for numeric member access.
      // Some upstream templates use Python/Jinja-style `content.0` instead.
      this.tokenizer.chat_template = fs.readFileSync(chatTemplatePath, "utf-8")
        .replace(/\.(\d+)\b/g, "[$1]");
    }
    this.cfg.maxPositionEmbeddings = loadMaxPositionEmbeddings(configDir);
    this.cfg.generationConfig = loadGenerationConfig(configDir);
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
