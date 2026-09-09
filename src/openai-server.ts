import http from "node:http";
import crypto from "node:crypto";
import { CaptureManager } from "./capture-manager";
import { tokenizeContinuation } from "./chat-continuation";
import { type OutputParserEvent } from "./chat-model-parser";
import { ChatModel, ChatCache, ChatTemplateKwargs, loadGenerationConfig, loadMaxPositionEmbeddings, type MtpDraftBatch, SamplingParams, type TokenSelector, Tokenizer } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { executePlan, type ExecutionPhase, ExecutionWorkspace } from "./execution-workspace";
import { mtpTotalTreeNodes } from "./glm51_model";
import { SamplingWorkspace } from "./sampling";
import { createDeviceOps, loadModel, ModelCliArgs, parseModelArgs, resolveModelSelection } from "./model_cli";
import { ParallelOps } from "./parallel_ops";
import { PhasedPrefillRunner, splitRaggedInput } from "./phased-prefill";
import { Tensor } from "./tensor";

const PAGE_SIZE = 64;

interface ChatMessage {
  role: string;
  content: unknown;
  reasoning_content?: string;
  tool_calls?: Array<{
    id?: string;
    type?: string;
    function: { name: string; arguments: unknown };
  }>;
  [key: string]: unknown;
}

interface ServerArgs extends ModelCliArgs {
  port: number;
  host: string;
  chunkSize: number;
  batchSize: number;
  maxPages: number;
  maxTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  repetitionPenaltyWindow: number;
  decodeLatency: number;
  noCudaGraph: boolean;
  noMtp: boolean;
  phasedPrefill: boolean;
  mtpDraftTopk: number[];
}

function parseArgs(argv: string[]): ServerArgs {
  const args: ServerArgs = {
    ...parseModelArgs(argv),
    port: 8000,
    host: "0.0.0.0",
    chunkSize: 8192,
    batchSize: 8,
    maxPages: 0,
    maxTokens: 65536,
    temperature: 0.6,
    topP: 0.95,
    topK: 20,
    repetitionPenalty: 1.0,
    presencePenalty: 0,
    repetitionPenaltyWindow: 64,
    decodeLatency: 0,
    noCudaGraph: false,
    noMtp: false,
    phasedPrefill: false,
    mtpDraftTopk: [],
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--port" && i + 1 < argv.length) args.port = parseInt(argv[++i], 10);
    else if (a === "--host" && i + 1 < argv.length) args.host = argv[++i];
    else if (a === "--chunk-size" && i + 1 < argv.length) args.chunkSize = parseInt(argv[++i], 10);
    else if (a === "--batch-size" && i + 1 < argv.length) args.batchSize = parseInt(argv[++i], 10);
    else if (a === "--max-pages" && i + 1 < argv.length) args.maxPages = parseInt(argv[++i], 10);
    else if (a === "--max-tokens" && i + 1 < argv.length) args.maxTokens = parseInt(argv[++i], 10);
    else if (a === "--temperature" && i + 1 < argv.length) args.temperature = parseFloat(argv[++i]);
    else if (a === "--top-p" && i + 1 < argv.length) args.topP = parseFloat(argv[++i]);
    else if (a === "--top-k" && i + 1 < argv.length) args.topK = parseInt(argv[++i], 10);
    else if (a === "--repetition-penalty" && i + 1 < argv.length) args.repetitionPenalty = parseFloat(argv[++i]);
    else if (a === "--presence-penalty" && i + 1 < argv.length) args.presencePenalty = parseFloat(argv[++i]);
    else if (a === "--repetition-penalty-window" && i + 1 < argv.length) args.repetitionPenaltyWindow = parseInt(argv[++i], 10);
    else if (a === "--decode-latency" && i + 1 < argv.length) args.decodeLatency = parseInt(argv[++i], 10);
    else if (a === "--no-cuda-graph") args.noCudaGraph = true;
    else if (a === "--no-mtp") args.noMtp = true;
    else if (a === "--phased-prefill") args.phasedPrefill = true;
    else if (a === "--mtp-draft-topk" && i + 1 < argv.length) args.mtpDraftTopk = argv[++i].split(",").map(value => parseInt(value.trim(), 10));
    else if (a === "--help" || a === "-h") { printHelp(); process.exit(0); }
  }
  if (args.mtp && !args.noMtp && args.mtpDraftTopk.length === 0) args.mtpDraftTopk = [1, 1, 1];
  if (args.mtpDraftTopk.some(topk => !Number.isInteger(topk) || topk < 1)) {
    throw new Error(`Invalid --mtp-draft-topk: ${args.mtpDraftTopk.join(",")}`);
  }
  if (args.phasedPrefill && !args.useGlm51) {
    throw new Error("--phased-prefill currently requires --glm51");
  }
  if (args.maxPages === 0) {
    args.maxPages = args.batchSize * Math.ceil(args.chunkSize / PAGE_SIZE);
  }
  return args;
}

function printHelp() {
  console.log(`OpenAI-compatible server for Qwen3-0.6B

Usage: npx tsx src/openai-server.ts [options]

Options:
  --port <int>                  Server port (default: 8000)
  --host <string>               Server host (default: 0.0.0.0)
  --gpu <int>                   GPU device ID (default: 0)
  --gpus <list>                 GPU device IDs
  --arena <int>                 Arena size in GiB per GPU
  --chunk-size <int>            Maximum prefill chunk per sequence (default: 8192)
  --batch-size <int>            Maximum concurrent requests (default: 8)
  --max-pages <int>             KV cache pages (default: batch-size * ceil(chunk-size / 64))
  --max-tokens <int>            Default max completion tokens (default: 65536)
  --model-dir <string>          Model directory path (default: auto-detect from HF cache)
  --temperature <float>         Override model default sampling temperature
  --top-p <float>               Override model default top-p
  --top-k <int>                 Override model default top-k
  --repetition-penalty <float>  Override model default repetition penalty
  --presence-penalty <float>    Default presence penalty (default: 0)
  --repetition-penalty-window <int>  Repetition penalty window (default: 64)
  --decode-latency <int>        Artificial delay per decode step in ms (default: 0)
  --no-cuda-graph               Disable CUDA graph capture
  --mtp                         Enable MTP speculative decoding
  --no-mtp                      Disable MTP decoding for an MTP-loaded model
  --phased-prefill              Overlap pairs of intermediate GLM-5.1 prefill chunks
  --mtp-draft-topk <list>       MTP draft top-k per depth (default: 1,1,1)
  --help, -h                    Show this help message
`);
}

function tokenizeMessages(
  tokenizer: Tokenizer,
  messages: ChatMessage[],
  tools?: unknown[],
  chatTemplateKwargs: ChatTemplateKwargs = {},
): number[] {
  if (chatTemplateKwargs.continue_final_message !== undefined && typeof chatTemplateKwargs.continue_final_message !== "boolean") {
    throw new Error("continue_final_message must be a boolean");
  }
  if (chatTemplateKwargs.continue_final_message) {
    return tokenizeContinuation(tokenizer, messages, tools, chatTemplateKwargs);
  }
  try {
    const opts: any = {
      ...chatTemplateKwargs,
      tokenize: true,
      add_generation_prompt: true,
      return_tensor: false,
      return_dict: true,
    };
    if (tools !== undefined) opts.tools = tools;
    const result = tokenizer.apply_chat_template(messages as any, opts) as unknown as { input_ids: number[] | number[][] };
    return (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
  } catch (error) {
    // The basic fallback cannot represent tools, so failing visibly is safer
    // than prompting the model as though no tools were supplied.
    if (tools !== undefined || Object.keys(chatTemplateKwargs).length > 0) throw error;
    const text = messages.map(m => `<|${m.role}|>\n${m.content}`).join("\n") + "\n\n\n";
    return tokenizer.encode(text, { add_special_tokens: false });
  }
}

function normalizeMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages.map(message => {
    const normalized: ChatMessage = {
      ...message,
      content: message.content ?? "",
    };
    if (!message.tool_calls) return normalized;

    normalized.tool_calls = message.tool_calls.map(toolCall => {
      const args = toolCall.function.arguments;
      return {
        ...toolCall,
        function: {
          ...toolCall.function,
          arguments: typeof args === "string" ? JSON.parse(args) : args,
        },
      };
    });
    return normalized;
  });
}

interface ResponseToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface CompletionRequest {
  id: string;
  messages: ChatMessage[];
  tools?: unknown[];
  chatTemplateKwargs: ChatTemplateKwargs;
  inputIds: number[];
  maxTokens: number;
  samplingParams: SamplingParams;
  stream: boolean;
  created: number;
  stopSequences: string[];
  generatedIds: number[];
  generatedText: string;
  reasoningText: string;
  toolCalls: ResponseToolCall[];
  parser: ReturnType<ChatModel["createParser"]>;
  finished: boolean;
  finishReason: string;
  promptTokenCount: number;
  cachedTokenCount: number;
  prefillTokenCount: number;
  prefillSeconds: number;
  decodeStartedAt?: number;
  performanceLogged: boolean;
  onOutput: (events: OutputParserEvent[]) => void;
  onFinish: () => void;
  onError: (error: unknown) => void;
}

function processOutputEvents(req: CompletionRequest, events: OutputParserEvent[]): void {
  for (const event of events) {
    if (event.type === "reasoning_delta") {
      req.reasoningText += event.text;
    } else if (event.type === "content_delta") {
      req.generatedText += event.text;
    } else if (event.type === "tool_call") {
      req.toolCalls[event.index] = {
        id: `call_${crypto.randomBytes(12).toString("hex")}`,
        type: "function",
        function: { name: event.name, arguments: event.arguments },
      };
    } else {
      console.warn(`Output parser error: ${event.message}`);
      req.generatedText += event.raw;
    }
  }
  req.onOutput(events);
}

function processOutputToken(req: CompletionRequest, tokenId: number): void {
  processOutputEvents(req, req.parser.onToken(tokenId));
}

function finishOutput(req: CompletionRequest): void {
  processOutputEvents(req, req.parser.finish());
  if (req.toolCalls.length > 0) req.finishReason = "tool_calls";
}

function logRequestPerformance(req: CompletionRequest): void {
  if (req.performanceLogged) return;
  req.performanceLogged = true;
  const decodedTokens = Math.max(0, req.generatedIds.length - 1);
  const decodeSeconds = req.decodeStartedAt === undefined
    ? 0
    : (performance.now() - req.decodeStartedAt) / 1000;
  const prefillTps = req.prefillSeconds > 0 ? req.prefillTokenCount / req.prefillSeconds : 0;
  const decodeTps = decodeSeconds > 0 ? decodedTokens / decodeSeconds : 0;
  console.log(
    `Request ${req.id}: prompt_tokens=${req.promptTokenCount} cached_tokens=${req.cachedTokenCount}`
    + ` prefill_tokens=${req.prefillTokenCount} prefill_tokens_per_second=${prefillTps.toFixed(1)}`
    + ` output_tokens=${req.generatedIds.length} decode_tokens_per_second=${decodeTps.toFixed(1)}`,
  );
}

interface ActiveSequence {
  request: CompletionRequest;
  lastToken: number;
}

interface ServerMetrics {
  runningRequests: number;
  generationTokensTotal: number;
  promptTokensTotal: number;
  specDecodeNumDraftsTotal: number;
  specDecodeNumDraftTokensTotal: number;
  specDecodeNumAcceptedTokensTotal: number;
  mtpPhaseSeconds: Map<string, number>;
  mtpPhaseCount: Map<string, number>;
  requestSuccessTotal: number;
  prefillTimeSecondsCount: number;
  prefillTimeSecondsSum: number;
}

class StagedPrefixPolicy {
  private nextKey = -1;
  private readonly retainedKeys = new Set<number>();

  constructor(private readonly pagedKV: ReturnType<ChatCache["getPagedKV"]>) {
    pagedKV.onPagePressure = requiredPages => this.evictUntilAvailable(requiredPages);
  }

  retainSequence(seqIdx: number): void {
    const key = this.nextKey--;
    this.pagedKV.stageSequence(seqIdx, key);
    this.retainedKeys.add(key);
  }

  clearStaging(): void {
    this.pagedKV.clearStaging();
    this.retainedKeys.clear();
  }

  private evictUntilAvailable(requiredPages: number): void {
    while (this.pagedKV.availablePages.length < requiredPages && this.retainedKeys.size > 0) {
      let longestKey: number | undefined;
      let longestPages = 0;
      for (const key of this.retainedKeys) {
        const sequence = this.pagedKV.staging.get(key);
        if (!sequence) {
          this.retainedKeys.delete(key);
          continue;
        }
        if (sequence.pages.length === 0) {
          this.retainedKeys.delete(key);
          this.pagedKV.removeStagedSequence(key);
          continue;
        }
        if (sequence.pages.length > longestPages) {
          longestKey = key;
          longestPages = sequence.pages.length;
        }
      }
      if (longestKey === undefined) break;

      const sequence = this.pagedKV.staging.get(longestKey)!;
      sequence.popPage();
      if (sequence.pages.length === 0) {
        this.retainedKeys.delete(longestKey);
        this.pagedKV.removeStagedSequence(longestKey);
      }
    }
  }
}

interface PromptPrefillRow {
  request: CompletionRequest;
  inputIds: number[];
}

async function prefillPromptChunks(
  pagedKV: ReturnType<ChatCache["getPagedKV"]>,
  rows: PromptPrefillRow[],
  chunkSize: number,
  nextStagingKey: number,
  runChunk: (inputIds: number[][], nextTokens: number[]) => Promise<void>,
  onCancelled: (row: PromptPrefillRow) => void,
): Promise<number> {
  const removeCancelledRows = (): void => {
    for (let i = rows.length - 1; i >= 0; i--) {
      if (!rows[i].request.finished) continue;
      const [row] = rows.splice(i, 1);
      pagedKV.removeSequence(i);
      onCancelled(row);
    }
  };

  removeCancelledRows();
  // prepareMtpInput may prepend one overlap token per row. Keep enough room for
  // those tokens and cap every intermediate prefill by the workspace's total
  // token capacity, not by a per-row limit.
  const finalInputBudget = chunkSize - rows.length;
  if (finalInputBudget < 0) {
    throw new Error(`Prefill batch ${rows.length} exceeds chunk size ${chunkSize}`);
  }
  while (rows.reduce((sum, row) => sum + row.inputIds.length, 0) > finalInputBudget) {
    const staged = rows.map((row, index) => ({ row, sequence: pagedKV.sequences[index], key: nextStagingKey++ }));
    for (const entry of staged) pagedKV.stageSequence(0, entry.key);

    let budget = chunkSize;
    const chunked: { entry: typeof staged[number]; take: number }[] = [];
    const deferred: typeof staged = [];
    for (const entry of staged) {
      // Keep one token available as nextTokens for the MTP rotation.
      const take = Math.min(Math.max(0, entry.row.inputIds.length - 1), budget);
      if (take > 0) {
        pagedKV.unstageSequence(entry.key);
        chunked.push({ entry, take });
        budget -= take;
      } else {
        deferred.push(entry);
      }
    }
    if (chunked.length === 0) {
      throw new Error(`Unable to fit prefill batch within ${chunkSize} tokens`);
    }

    const inputIds = chunked.map(({ entry, take }) => entry.row.inputIds.slice(0, take));
    const nextTokens = chunked.map(({ entry, take }) => entry.row.inputIds[take]);
    await runChunk(inputIds, nextTokens);
    for (let i = 0; i < chunked.length; i++) {
      chunked[i].entry.sequence.reportTokens(inputIds[i]);
      const { entry, take } = chunked[i];
      entry.row.inputIds = entry.row.inputIds.slice(take);
    }

    for (const entry of deferred) pagedKV.unstageSequence(entry.key);
    rows.splice(0, rows.length, ...chunked.map(({ entry }) => entry.row), ...deferred.map(entry => entry.row));
    removeCancelledRows();
  }
  return nextStagingKey;
}

async function generateMtpBatches(
  model: ChatModel,
  ws: ExecutionWorkspace,
  cache: ChatCache,
  eosIds: Set<number>,
  pendingQueue: CompletionRequest[],
  maxBatchSize: number,
  decodeLatencyMs: number,
  metrics: ServerMetrics,
  captureManager: CaptureManager,
  samplingWorkspace: SamplingWorkspace,
  topks: readonly number[],
  chunkSize: number,
  stagedPrefixes: StagedPrefixPolicy,
  phasedPrefill: boolean,
): Promise<number> {
  if (!model.planPrefillMtpDraftExtend || !model.planTargetVerification) {
    throw new Error("The selected model does not support plan-based MTP decoding");
  }

  const pagedKV = cache.getPagedKV();
  const requests: CompletionRequest[] = [];
  const admitted = new Set<CompletionRequest>();
  const notified = new Set<CompletionRequest>();
  let draft: MtpDraftBatch | undefined;
  let nextStagingKey = 0;
  let admittedRequests = 0;
  const numVerificationTokens = mtpTotalTreeNodes(topks) + 1;
  const timing = process.env.GLM_MTP_TIMING === "1";
  const recordMtpPhase = (timingName: string, batchSize: number, elapsedSeconds: number): void => {
    const key = `${timingName}|${batchSize}`;
    metrics.mtpPhaseSeconds.set(key, (metrics.mtpPhaseSeconds.get(key) ?? 0) + elapsedSeconds);
    metrics.mtpPhaseCount.set(key, (metrics.mtpPhaseCount.get(key) ?? 0) + 1);
  };
  const observeMtpPhase = (phase: ExecutionPhase, elapsedSeconds: number): void => {
    if (!phase.timingName) return;
    recordMtpPhase(phase.timingName, phase.states[0]?.batchSize ?? 0, elapsedSeconds);
  };
  const selectTokens: TokenSelector = samplingWorkspace;
  const updateSamplingParams = (params: SamplingParams[]): void => {
    samplingWorkspace.updateSampler(params, params.map(() => []));
  };

  const finish = (req: CompletionRequest): void => {
    if (notified.has(req)) return;
    notified.add(req);
    admitted.delete(req);
    metrics.runningRequests--;
    metrics.requestSuccessTotal++;
    logRequestPerformance(req);
    try { req.onFinish(); } catch {}
  };

  const processToken = (req: CompletionRequest, token: number): void => {
    if (req.finished) return;
    req.generatedIds.push(token);
    metrics.generationTokensTotal++;
    const isEos = eosIds.has(token);
    if (!isEos) processOutputToken(req, token);
    if (isEos || req.generatedIds.length >= req.maxTokens) {
      req.finished = true;
      req.finishReason = isEos ? "stop" : "length";
    }
    if (!req.finished) checkStopSequences(req);
    if (req.finished) finish(req);
  };

  const removeFinishedRows = (): void => {
    if (!draft) return;
    const retained: number[] = [];
    for (let i = 0; i < requests.length; i++) {
      if (!requests[i].finished) retained.push(i);
    }
    for (let i = requests.length - 1; i >= 0; i--) {
      if (requests[i].finished) {
        finish(requests[i]);
        stagedPrefixes.retainSequence(i);
        requests.splice(i, 1);
      }
    }
    if (retained.length !== draft.targetTokens.length) {
      draft = {
        targetTokens: retained.map(index => draft!.targetTokens[index]),
        treeTokens: retained.map(index => draft!.treeTokens[index]),
        topks: draft.topks,
        ...(draft.proposal ? { proposal: {
          capacity: draft.proposal.capacity,
          probabilities: draft.proposal.device ? [] : retained.map(index => draft!.proposal!.probabilities[index]),
          tokenIds: draft.proposal.device ? [] : retained.map(index => draft!.proposal!.tokenIds[index]),
          ...(draft.proposal.device ? { device: {
            ...draft.proposal.device,
            rows: retained.map(index => draft!.proposal!.device!.rows[index]),
          } } : {}),
        } } : {}),
      };
    }
  };

  try {
    while (requests.length > 0 || pendingQueue.length > 0) {
      const newCount = Math.min(pendingQueue.length, maxBatchSize - requests.length);
      if (newCount > 0) {
        const newRequests = pendingQueue.splice(0, newCount);
        for (const req of newRequests) admitted.add(req);
        admittedRequests += newCount;
        metrics.runningRequests += newCount;
        metrics.promptTokensTotal += newRequests.reduce((sum, req) => sum + req.promptTokenCount, 0);

        // Recondition at a verification boundary. Existing rows retain their
        // committed KV and reprocess only their outstanding replacement token.
        const activeEntries = requests.map(request => ({ request, key: nextStagingKey++ }));
        for (const entry of activeEntries) pagedKV.stageSequence(0, entry.key);
        cache.reset(newCount);
        const suffixIds = newRequests.map((req, index) => cache.prefixMatch(index, req.inputIds));
        for (let i = 0; i < newRequests.length; i++) {
          const cachedTokens = newRequests[i].promptTokenCount - suffixIds[i].length;
          newRequests[i].cachedTokenCount = cachedTokens;
          // MTP reprocesses the final cached token to condition its extra layer.
          newRequests[i].prefillTokenCount = suffixIds[i].length + (cachedTokens > 0 ? 1 : 0);
        }
        const newRows = newRequests.map((request, index) => ({ request, inputIds: suffixIds[index] }));
        if (!model.planPrefillMtpChunk) throw new Error("The selected model does not support chunked MTP prefill");
        if (phasedPrefill && !model.planPrefillMtpChunkPhased) {
          throw new Error("The selected model does not support phased MTP prefill");
        }
        const prefillStart = performance.now();
        const phasedRunner = phasedPrefill ? new PhasedPrefillRunner(model, model.glm) : undefined;
        nextStagingKey = await prefillPromptChunks(
            pagedKV,
            newRows,
            chunkSize,
            nextStagingKey,
            async (inputIds, nextTokens) => {
              if (phasedRunner) {
                const split = splitRaggedInput(inputIds);
                if (split) {
                  const phaseStart = performance.now();
                  using planA = model.planPrefillMtpChunkPhased!(ws, cache, split.inputA, split.nextA);
                  using planB = model.planPrefillMtpChunkPhased!(ws, cache, split.inputB, nextTokens);
                  phasedRunner.runPlanPair(planA, planB);
                  await model.glm.synchronizeAsync();
                  const elapsedSeconds = (performance.now() - phaseStart) / 2000;
                  recordMtpPhase("prefill_chunk", planA.state.batchSize, elapsedSeconds);
                  recordMtpPhase("prefill_chunk", planB.state.batchSize, elapsedSeconds);
                  ws.assertClear();
                  ws.resetPlanSlots();
                  return;
                }
              }
              await executePlan(captureManager, ws, model.planPrefillMtpChunk!(ws, cache, inputIds, nextTokens), observeMtpPhase);
            },
            row => finish(row.request),
          );
        const retainedActive: CompletionRequest[] = [];
        for (const entry of activeEntries) {
          if (entry.request.finished) {
            pagedKV.removeStagedSequence(entry.key);
            finish(entry.request);
          } else {
            pagedKV.unstageSequence(entry.key);
            retainedActive.push(entry.request);
          }
        }
        requests.splice(0, requests.length, ...newRows.map(row => row.request), ...retainedActive);
        if (requests.length === 0) {
          draft = undefined;
          continue;
        }

        const mtpInputIds = model.prepareMtpInput(cache, [
          ...newRows.map(row => row.inputIds),
          ...Array.from({ length: retainedActive.length }, () => [] as number[]),
        ]);
        const targetParams = requests.map(req => req.samplingParams);
        updateSamplingParams(targetParams);
        if (samplingWorkspace.mtpEnabled) samplingWorkspace.updateMtpSampler(targetParams);
        draft = (await executePlan(
          captureManager,
          ws,
          model.planPrefillMtpDraftExtend(ws, cache, mtpInputIds, topks, selectTokens),
          observeMtpPhase,
        )).result;
        const prefillSeconds = (performance.now() - prefillStart) / 1000;
        metrics.prefillTimeSecondsCount += newCount;
        metrics.prefillTimeSecondsSum += prefillSeconds * newCount;
        const decodeStartedAt = performance.now();
        for (const req of newRequests) {
          req.prefillSeconds = prefillSeconds;
          req.decodeStartedAt = decodeStartedAt;
        }

        for (let i = 0; i < requests.length; i++) {
          if (i < newRows.length) pagedKV.reportTokens(i, newRows[i].inputIds);
          pagedKV.reportTokens(i, [draft.targetTokens[i]]);
          processToken(requests[i], draft.targetTokens[i]);
        }
        removeFinishedRows();
        continue;
      }

      const iterationBatchSize = requests.length;
      const iterationStart = timing ? performance.now() : 0;
      let phaseSeconds = 0;
      const verificationParams = requests.flatMap(req =>
        Array.from({ length: numVerificationTokens }, () => req.samplingParams));
      if (!samplingWorkspace.mtpEnabled) updateSamplingParams(verificationParams);
      if (samplingWorkspace.mtpEnabled) samplingWorkspace.updateMtpSampler(requests.map(req => req.samplingParams));
      const planStart = timing ? performance.now() : 0;
      const step = await executePlan(captureManager, ws, model.planTargetVerification(ws, cache, draft!, selectTokens), timing ? (phase, seconds) => {
        phaseSeconds += seconds;
        observeMtpPhase(phase, seconds);
      } : observeMtpPhase);
      const reportingStart = timing ? performance.now() : 0;
      draft = step.result.draft;
      metrics.specDecodeNumDraftsTotal += requests.length;
      metrics.specDecodeNumDraftTokensTotal += step.result.numDraftTokens * requests.length;
      metrics.specDecodeNumAcceptedTokensTotal += step.result.numAccepted.reduce((sum, count) => sum + count, 0);
      for (let i = 0; i < requests.length; i++) {
        const tokens = step.result.tokens[i];
        pagedKV.reportTokens(i, tokens);
        for (const token of tokens) {
          processToken(requests[i], token);
          if (requests[i].finished) break;
        }
      }
      removeFinishedRows();

      const yieldStart = timing ? performance.now() : 0;
      if (decodeLatencyMs > 0) {
        await new Promise(resolve => setTimeout(resolve, decodeLatencyMs));
      } else {
        await new Promise(resolve => setImmediate(resolve));
      }
      if (timing) {
        const end = performance.now();
        recordMtpPhase("iteration", iterationBatchSize, (end - iterationStart) / 1000);
        recordMtpPhase("sampling_params", iterationBatchSize, (planStart - iterationStart) / 1000);
        recordMtpPhase("planning_cleanup", iterationBatchSize, (reportingStart - planStart) / 1000 - phaseSeconds);
        recordMtpPhase("reporting", iterationBatchSize, (yieldStart - reportingStart) / 1000);
        recordMtpPhase("yield", iterationBatchSize, (end - yieldStart) / 1000);
      }
    }
  } catch (error) {
    ws.clearTracking();
    stagedPrefixes.clearStaging();
    cache.reset(0);
    for (const req of admitted) {
      req.finished = true;
      metrics.runningRequests = Math.max(0, metrics.runningRequests - 1);
      try { req.onError(error); } catch {}
    }
    throw error;
  }

  return admittedRequests;
}

async function generateContinuousBatch(
  model: ChatModel,
  ws: ExecutionWorkspace,
  glm: DeviceOps,
  cache: ChatCache,
  eosIds: Set<number>,
  pendingQueue: CompletionRequest[],
  maxBatchSize: number,
  decodeLatencyMs: number,
  metrics: ServerMetrics,
  captureManager: CaptureManager,
  samplingWorkspace: SamplingWorkspace,
  stagedPrefixes: StagedPrefixPolicy,
  phasedPrefill: boolean,
): Promise<number> {
  const pagedKV = cache.getPagedKV();
  const eosToken = [...eosIds][0];
  const active: ActiveSequence[] = [];
  let nextStagingKey = 0;
  let admittedRequests = 0;
  const admitted = new Set<CompletionRequest>();
  const finishCancelled = (req: CompletionRequest): void => {
    if (!admitted.delete(req)) return;
    metrics.runningRequests--;
    metrics.requestSuccessTotal++;
    logRequestPerformance(req);
    try { req.onFinish(); } catch {}
  };

  try {
    while (active.length > 0 || pendingQueue.length > 0) {
    // 1. Remove finished sequences (reverse order to avoid index shift)
    for (let i = active.length - 1; i >= 0; i--) {
      if (active[i].request.finished) {
        stagedPrefixes.retainSequence(i);
        finishCancelled(active[i].request);
        active.splice(i, 1);
      }
    }

    // 2. Prefill new requests if there's room
    const availableSlots = maxBatchSize - active.length;
    if (pendingQueue.length > 0 && availableSlots > 0) {
      const newCount = Math.min(pendingQueue.length, availableSlots);
      const newRequests = pendingQueue.splice(0, newCount);
      for (const req of newRequests) admitted.add(req);
      admittedRequests += newCount;
      metrics.runningRequests += newCount;

      for (const req of newRequests) metrics.promptTokensTotal += req.promptTokenCount;

      const activeEntries = active.map(sequence => ({ sequence, key: nextStagingKey++ }));
      for (const entry of activeEntries) pagedKV.stageSequence(0, entry.key);
      pagedKV.reset(newCount);
      const newRows = newRequests.map((request, index) => ({
        request,
        inputIds: cache.prefixMatch(index, request.inputIds),
      }));
      for (const row of newRows) {
        row.request.cachedTokenCount = row.request.promptTokenCount - row.inputIds.length;
        row.request.prefillTokenCount = row.inputIds.length;
      }

      // Prefill new requests
      const prefillStart = performance.now();
      const phasedRunner = phasedPrefill ? new PhasedPrefillRunner(model, glm) : undefined;
      nextStagingKey = await prefillPromptChunks(
          pagedKV,
          newRows,
          ws.maxSeqLen,
          nextStagingKey,
          async inputIds => {
            if (phasedRunner) {
              const split = splitRaggedInput(inputIds);
              if (split) {
                const stateA = ws.planPrefill(model, split.inputA.length, split.inputA.map(ids => ids.length), cache);
                stateA.setInput(split.inputA);
                const stateB = ws.planPrefill(model, split.inputB.length, split.inputB.map(ids => ids.length), cache);
                stateB.setInput(split.inputB);
                phasedRunner.runPair(stateA, stateB);
                await glm.synchronizeAsync();
                ws.assertClear();
                ws.resetPlanSlots();
                return;
              }
            }
            {
              using _logits = ws.forwardPrefill(model, inputIds, cache);
              await glm.synchronizeAsync();
            }
            ws.clearTracking();
          },
          row => finishCancelled(row.request),
        );
      const retainedActiveEntries: typeof activeEntries = [];
      for (const entry of activeEntries) {
        if (entry.sequence.request.finished) {
          pagedKV.removeStagedSequence(entry.key);
          finishCancelled(entry.sequence.request);
        } else {
          retainedActiveEntries.push(entry);
        }
      }
      if (newRows.length === 0) {
        for (const entry of retainedActiveEntries) pagedKV.unstageSequence(entry.key);
        const retainedActive = retainedActiveEntries.map(entry => entry.sequence);
        active.splice(0, active.length, ...retainedActive);
        continue;
      }
      const inputIdsList = newRows.map(row => row.inputIds);
      const firstTokens = ws.forwardEagerPrefill(model, inputIdsList, cache);
      await glm.synchronizeAsync();
      ws.clearTracking();
      const prefillSeconds = (performance.now() - prefillStart) / 1000;
      metrics.prefillTimeSecondsCount += newRows.length;
      metrics.prefillTimeSecondsSum += prefillSeconds * newRows.length;
      const decodeStartedAt = performance.now();
      for (const row of newRows) {
        row.request.prefillSeconds = prefillSeconds;
        row.request.decodeStartedAt = decodeStartedAt;
      }
      metrics.generationTokensTotal += firstTokens.length;

      // Report tokens and check for first-token EOS
      for (let i = 0; i < newRows.length; i++) {
        pagedKV.reportTokens(i, inputIdsList[i]);
        pagedKV.reportTokens(i, [firstTokens[i]]);
        newRows[i].request.generatedIds.push(firstTokens[i]);
        const isEos = eosIds.has(firstTokens[i]);
        if (!isEos) {
          processOutputToken(newRows[i].request, firstTokens[i]);
        }
        if (isEos || newRows[i].request.generatedIds.length >= newRows[i].request.maxTokens) {
          newRows[i].request.finished = true;
          newRows[i].request.finishReason = isEos ? "stop" : "length";
      }

        if (!newRows[i].request.finished) {
          checkStopSequences(newRows[i].request);
        }
      }

      for (const entry of retainedActiveEntries) pagedKV.unstageSequence(entry.key);
      const retainedActive = retainedActiveEntries.map(entry => entry.sequence);

      // Build new active list matching pagedKV sequence order: [new..., old...]
      const newActiveSequences: ActiveSequence[] = [];
      for (let i = 0; i < newRows.length; i++) {
        newActiveSequences.push({
          request: newRows[i].request,
          lastToken: firstTokens[i],
        });
      }
      newActiveSequences.push(...retainedActive);
      active.length = 0;
      active.push(...newActiveSequences);
    }

    // 3. Remove any newly-finished sequences (e.g. first-token EOS)
    for (let i = active.length - 1; i >= 0; i--) {
      if (active[i].request.finished) {
        stagedPrefixes.retainSequence(i);
        finishCancelled(active[i].request);
        active.splice(i, 1);
      }
    }

    if (active.length === 0) continue;

    // 4. Update sampling workspace for current batch composition
    const params = active.map(a => a.request.samplingParams);
    const tokenHistories = pagedKV.sequences.slice(0, active.length).map(s => s.getTokenIds());
    const useArgmax = params.every(param =>
      param.temperature <= 0 && param.repetitionPenalty === 1 && param.presencePenalty === 0,
    );
    if (!useArgmax) samplingWorkspace.updateSampler(params, tokenHistories);

    // 5. Decode one step
    const inputTokens = active.map(a => a.lastToken);
    const state = ws.planDecode(model, active.length, cache, true);
    state.setInput([inputTokens]);
    const decodeResult = state.capture(captureManager, {}, () => {
      using hiddenStates = model.forwardModel(state);
      using decodeLogits = state.computeLogits(hiddenStates, model);
      if (useArgmax) return decodeLogits.argmax();
      return samplingWorkspace.sample(decodeLogits);
    }, [useArgmax ? "openai-decode-argmax" : "openai-decode-sample", ...(useArgmax ? [] : [samplingWorkspace.captureKey])]);
    await glm.synchronizeAsync();
    let newTokens: number[];
    {
      using selectedTokens = decodeResult as Tensor;
      newTokens = selectedTokens.readInt32LEArray();
    }
    ws.clearTracking();
    metrics.generationTokensTotal += newTokens.length;

    // 6. Process decoded tokens
    for (let i = 0; i < active.length; i++) {
      const req = active[i].request;
      pagedKV.reportTokens(i, [newTokens[i]]);
      active[i].lastToken = newTokens[i];
      req.generatedIds.push(newTokens[i]);
      const isEos = eosIds.has(newTokens[i]);
      if (!isEos) {
        processOutputToken(req, newTokens[i]);
      }
      if (isEos || req.generatedIds.length >= req.maxTokens) {
        req.finished = true;
        req.finishReason = isEos ? "stop" : "length";
      }
      if (!req.finished) {
        checkStopSequences(req);
      }
    }

      if (decodeLatencyMs > 0) {
        await new Promise(resolve => setTimeout(resolve, decodeLatencyMs));
      } else {
        await new Promise(resolve => setImmediate(resolve));
      }
    }
  } catch (error) {
    ws.clearTracking();
    stagedPrefixes.clearStaging();
    for (const req of admitted) {
      req.finished = true;
      metrics.runningRequests = Math.max(0, metrics.runningRequests - 1);
      try { req.onError(error); } catch {}
    }
    throw error;
  }

  return admittedRequests;
}

function checkStopSequences(req: CompletionRequest): void {
  for (const stop of req.stopSequences) {
    if (req.generatedText.includes(stop)) {
      const idx = req.generatedText.indexOf(stop);
      req.generatedText = req.generatedText.slice(0, idx);
      req.finished = true;
      req.finishReason = "stop";
      break;
    }
  }
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk: Buffer) => { body += chunk.toString(); });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

function generateId(): string {
  return "chatcmpl-" + crypto.randomUUID().replace(/-/g, "").slice(0, 24);
}

function writeSSE(res: http.ServerResponse, data: object): void {
  try {
    if (!res.writableEnded) {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    }
  } catch {}
}

function writeSSEDone(res: http.ServerResponse): void {
  try {
    if (!res.writableEnded) {
      res.write("data: [DONE]\n\n");
    }
  } catch {}
}

function sendJSON(res: http.ServerResponse, statusCode: number, data: object): void {
  res.writeHead(statusCode, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}

function sendMetrics(
  res: http.ServerResponse,
  metrics: ServerMetrics,
  pendingRequests: number,
  cache: ChatCache,
  batchSize: number,
  chunkSize: number,
  maxModelLen: number,
): void {
  const pagedKV = cache.getPagedKV();
  const availableSlots = Math.max(0, batchSize - metrics.runningRequests);
  const waitingRequests = Math.max(0, pendingRequests - availableSlots);
  const maxTotalTokens = pagedKV.maxPages * pagedKV.pageSize;
  const kvUsage = pagedKV.maxPages === 0
    ? 0
    : (pagedKV.maxPages - pagedKV.availablePages.length) / pagedKV.maxPages;
  const mtpPhaseLines = Array.from(metrics.mtpPhaseSeconds, ([key, seconds]) => {
    const [phase, batchSize] = key.split("|");
    const labels = `{phase="${phase}",batch_size="${batchSize}"}`;
    return [
      `glm:mtp_phase_seconds_total${labels} ${seconds}`,
      `glm:mtp_phase_count_total${labels} ${metrics.mtpPhaseCount.get(key) ?? 0}`,
    ];
  }).flat();
  const lines = [
    "# HELP vllm:num_requests_running GLM.js requests admitted for model execution.",
    "# TYPE vllm:num_requests_running gauge",
    `vllm:num_requests_running ${metrics.runningRequests}`,
    "# HELP vllm:num_requests_waiting GLM.js requests blocked on admission capacity.",
    "# TYPE vllm:num_requests_waiting gauge",
    `vllm:num_requests_waiting ${waitingRequests}`,
    "# HELP vllm:generation_tokens_total GLM.js sampled completion tokens.",
    "# TYPE vllm:generation_tokens_total counter",
    `vllm:generation_tokens_total ${metrics.generationTokensTotal}`,
    "# HELP vllm:spec_decode_num_drafts_total GLM.js MTP draft sequences verified.",
    "# TYPE vllm:spec_decode_num_drafts_total counter",
    `vllm:spec_decode_num_drafts_total ${metrics.specDecodeNumDraftsTotal}`,
    "# HELP vllm:spec_decode_num_draft_tokens_total GLM.js MTP draft tokens proposed.",
    "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
    `vllm:spec_decode_num_draft_tokens_total ${metrics.specDecodeNumDraftTokensTotal}`,
    "# HELP vllm:spec_decode_num_accepted_tokens_total GLM.js MTP draft tokens accepted.",
    "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
    `vllm:spec_decode_num_accepted_tokens_total ${metrics.specDecodeNumAcceptedTokensTotal}`,
    "# HELP glm:mtp_phase_seconds_total GPU-complete wall time spent in MTP execution phases.",
    "# TYPE glm:mtp_phase_seconds_total counter",
    "# HELP glm:mtp_phase_count_total Completed MTP execution phases.",
    "# TYPE glm:mtp_phase_count_total counter",
    ...mtpPhaseLines,
    "# HELP vllm:prompt_tokens_total GLM.js prompt tokens admitted for prefill.",
    "# TYPE vllm:prompt_tokens_total counter",
    `vllm:prompt_tokens_total ${metrics.promptTokensTotal}`,
    "# HELP vllm:request_success_total GLM.js successfully completed requests.",
    "# TYPE vllm:request_success_total counter",
    `vllm:request_success_total ${metrics.requestSuccessTotal}`,
    "# HELP vllm:kv_cache_usage_perc GLM.js fraction of KV cache pages in use.",
    "# TYPE vllm:kv_cache_usage_perc gauge",
    `vllm:kv_cache_usage_perc ${kvUsage}`,
    "# HELP vllm:cache_config_info GLM.js paged KV cache configuration.",
    "# TYPE vllm:cache_config_info gauge",
    `vllm:cache_config_info{block_size="${pagedKV.pageSize}",num_gpu_blocks="${pagedKV.maxPages}",max_total_num_tokens="${maxTotalTokens}",cp_world_size="1"} 1`,
    "# HELP vllm:scheduler_config_info GLM.js request scheduler configuration.",
    "# TYPE vllm:scheduler_config_info gauge",
    `vllm:scheduler_config_info{max_num_seqs="${batchSize}",max_num_batched_tokens="${batchSize * chunkSize}",max_model_len="${maxModelLen}"} 1`,
    "# HELP vllm:request_prefill_time_seconds GLM.js request prefill duration.",
    "# TYPE vllm:request_prefill_time_seconds histogram",
    `vllm:request_prefill_time_seconds_bucket{le="+Inf"} ${metrics.prefillTimeSecondsCount}`,
    `vllm:request_prefill_time_seconds_sum ${metrics.prefillTimeSecondsSum}`,
    `vllm:request_prefill_time_seconds_count ${metrics.prefillTimeSecondsCount}`,
    "",
  ];
  res.writeHead(200, { "Content-Type": "text/plain; version=0.0.4; charset=utf-8" });
  res.end(lines.join("\n"));
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
  const args = parseArgs(argv);

  const { modelDir, repoId: modelName } = resolveModelSelection(args);

  console.log(`Loading model from ${modelDir}...`);
  const { glm, gpuDevices } = createDeviceOps(args);
  const model = await loadModel(glm, args, modelDir);
  const maxModelLen = model.cfg.maxPositionEmbeddings ?? loadMaxPositionEmbeddings(modelDir) ?? args.chunkSize;
  model.cfg.maxPositionEmbeddings = maxModelLen;
  const generationConfig = model.cfg.generationConfig ?? loadGenerationConfig(modelDir);
  model.cfg.generationConfig = generationConfig;
  if (!argv.includes("--temperature") && generationConfig.temperature !== undefined) args.temperature = generationConfig.temperature;
  if (!argv.includes("--top-p") && generationConfig.topP !== undefined) args.topP = generationConfig.topP;
  if (!argv.includes("--top-k") && generationConfig.topK !== undefined) args.topK = generationConfig.topK;
  if (!argv.includes("--repetition-penalty") && generationConfig.repetitionPenalty !== undefined) {
    args.repetitionPenalty = generationConfig.repetitionPenalty;
  }
  if (!argv.includes("--max-tokens") && generationConfig.maxNewTokens !== undefined) {
    args.maxTokens = generationConfig.maxNewTokens;
  }
  if (args.mtp && !args.noMtp && (!model.planPrefillMtpDraftExtend || !model.planTargetVerification)) {
    throw new Error("--mtp requires a model with plan-based MTP decoding support");
  }
  const cache = model.createChatCache(args.maxPages, args.batchSize, args.chunkSize);
  const ws = new ExecutionWorkspace(glm, args.batchSize, args.chunkSize);
  const captureManager = new CaptureManager(glm);
  captureManager.disabled = args.noCudaGraph;
  const mtpEnabled = args.mtp && !args.noMtp;
  const samplingWorkspace = new SamplingWorkspace(glm,
    args.batchSize * (mtpEnabled ? mtpTotalTreeNodes(args.mtpDraftTopk) + 1 : 1),
    model.cfg.vocabSize, args.repetitionPenaltyWindow,
    mtpEnabled && args.mtpDraftTopk.length > 0 && args.mtpDraftTopk.every(k => k === 1)
      ? { maxBatchSize: args.batchSize, depth: args.mtpDraftTopk.length, retainProposalsOnGpu: process.env.GLM_MTP_GPU_PROPOSALS !== "0" }
      : undefined);
  const tokenizer = model.tokenizer;
  const eosIds = model.eosIds;

  console.log(`Model loaded. chunk-size=${args.chunkSize} batch-size=${args.batchSize} max-pages=${args.maxPages} max-tokens=${args.maxTokens}`);
  console.log(`Generation defaults: temperature=${args.temperature} top-p=${args.topP} top-k=${args.topK} repetition-penalty=${args.repetitionPenalty}`);

  {
    console.log("Warming up...");
    const warmupIds = tokenizeMessages(tokenizer, [{ role: "user", content: "Hello" }]);
    cache.reset(1);
    using warmupSw = new SamplingWorkspace(glm, 1, model.cfg.vocabSize, args.repetitionPenaltyWindow);
    warmupSw.updateSampler([makeSamplingParamsHelper(args)], [warmupIds]);
    let lastToken: number;
    {
      using warmupLogits = ws.forwardPrefill(model, [warmupIds], cache);
      using warmupSampled = warmupSw.sample(warmupLogits);
      lastToken = warmupSampled.readInt32LEArray()[0];
    }
    cache.reportTokens(0, warmupIds);
    cache.reportTokens(0, [lastToken]);
    for (let i = 0; i < 3; i++) {
      const st = ws.planDecode(model, 1, cache);
      st.setInput([[lastToken]]);
      using hs = model.forward(st);
      using lg = st.computeLogits(hs, model);
      using ns = warmupSw.sample(lg);
      lastToken = ns.readInt32LEArray()[0];
      cache.reportTokens(0, [lastToken]);
    }
    await glm.synchronizeAsync();
    ws.clearTracking();
    cache.reset(1);
    console.log("Warmup complete.");
  }

  function makeSamplingParamsHelper(a: { temperature: number; topP: number; topK: number; repetitionPenalty: number; presencePenalty: number; repetitionPenaltyWindow: number }): SamplingParams {
    return {
      temperature: a.temperature,
      topP: a.topP,
      topK: a.topK,
      repetitionPenalty: a.repetitionPenalty,
      presencePenalty: a.presencePenalty,
      repetitionPenaltyWindow: a.repetitionPenaltyWindow,
    };
  }

  const pendingQueue: CompletionRequest[] = [];
  const stagedPrefixes = new StagedPrefixPolicy(cache.getPagedKV());
  const metrics: ServerMetrics = {
    runningRequests: 0,
    generationTokensTotal: 0,
    promptTokensTotal: 0,
    specDecodeNumDraftsTotal: 0,
    specDecodeNumDraftTokensTotal: 0,
    specDecodeNumAcceptedTokensTotal: 0,
    mtpPhaseSeconds: new Map(),
    mtpPhaseCount: new Map(),
    requestSuccessTotal: 0,
    prefillTimeSecondsCount: 0,
    prefillTimeSecondsSum: 0,
  };
  let busy = false;
  let queueStartScheduled = false;

  const isFatalCudaError = (error: unknown): boolean => {
    const message = error instanceof Error ? `${error.message}\n${error.stack ?? ""}` : String(error);
    return /illegal memory access|misaligned address|device-side assert|unspecified launch failure|CUDA context.*destroyed/i.test(message);
  };

  async function processQueue(): Promise<void> {
    if (busy) return;
    busy = true;
    try {
      const t0 = performance.now();
      let requestCount = 0;
      try {
        requestCount = args.mtp && !args.noMtp
          ? await generateMtpBatches(
            model,
            ws,
            cache,
            eosIds,
            pendingQueue,
            args.batchSize,
            args.decodeLatency,
            metrics,
            captureManager,
            samplingWorkspace,
            args.mtpDraftTopk,
            args.chunkSize,
            stagedPrefixes,
            args.phasedPrefill,
          )
          : await generateContinuousBatch(
            model,
            ws,
            glm,
            cache,
            eosIds,
            pendingQueue,
            args.batchSize,
            args.decodeLatency,
            metrics,
            captureManager,
            samplingWorkspace,
            stagedPrefixes,
            args.phasedPrefill,
          );
      } catch (err) {
        console.error("Continuous batch error:", err);
        if (isFatalCudaError(err)) {
          if (glm instanceof ParallelOps) console.error(glm.communicationDiagnostics());
          for (const req of pendingQueue.splice(0)) {
            req.finished = true;
            try { req.onError(err); } catch {}
          }
          // CUDA faults poison subsequent work. Exit this executor thread so
          // clients fail immediately and the persistent loader reports idle.
          setImmediate(() => process.exit(1));
        }
      }
      const elapsed = (performance.now() - t0) / 1000;
      if (elapsed > 0) {
        console.log(`Batch done: ${requestCount} req(s), ${elapsed.toFixed(2)}s`);
      }
    } finally {
      busy = false;
      if (pendingQueue.length > 0) scheduleQueueStart();
    }
  }

  function scheduleQueueStart(): void {
    if (busy || queueStartScheduled) return;
    queueStartScheduled = true;
    setImmediate(() => {
      queueStartScheduled = false;
      void processQueue();
    });
  }

  function enqueueRequest(req: CompletionRequest): void {
    pendingQueue.push(req);
    // Let all request callbacks already ready in this event-loop turn enqueue
    // before the first one starts an underfilled prefill batch.
    scheduleQueueStart();
  }

  function handleChatCompletions(req: http.IncomingMessage, res: http.ServerResponse): void {
    readBody(req).then(body => {
      let params: any;
      try {
        params = JSON.parse(body);
      } catch {
        sendJSON(res, 400, { error: { message: "Invalid JSON", type: "invalid_request_error" } });
        return;
      }

      if (params.model !== undefined && params.model !== modelName) {
        sendJSON(res, 404, {
          error: {
            message: `The model '${params.model}' does not exist`,
            type: "invalid_request_error",
            param: "model",
            code: "model_not_found",
          },
        });
        return;
      }

      const inputMessages = params.messages as ChatMessage[] | undefined;
      if (!inputMessages || !Array.isArray(inputMessages) || inputMessages.length === 0) {
        sendJSON(res, 400, { error: { message: "messages is required and must be a non-empty array", type: "invalid_request_error" } });
        return;
      }
      let messages: ChatMessage[];
      try {
        messages = normalizeMessages(inputMessages);
      } catch {
        sendJSON(res, 400, { error: { message: "tool call arguments must contain valid JSON", type: "invalid_request_error" } });
        return;
      }
      const tools = params.tools as unknown[] | undefined;
      if (tools !== undefined && !Array.isArray(tools)) {
        sendJSON(res, 400, { error: { message: "tools must be an array", type: "invalid_request_error" } });
        return;
      }
      if (
        params.chat_template_kwargs !== undefined &&
        (params.chat_template_kwargs === null || typeof params.chat_template_kwargs !== "object" || Array.isArray(params.chat_template_kwargs))
      ) {
        sendJSON(res, 400, { error: { message: "chat_template_kwargs must be an object", type: "invalid_request_error" } });
        return;
      }
      const chatTemplateKwargs: ChatTemplateKwargs = { ...(params.chat_template_kwargs ?? {}) };
      if (params.continue_final_message !== undefined) {
        chatTemplateKwargs.continue_final_message = params.continue_final_message;
      }
      if (params.enable_thinking !== undefined) {
        if (typeof params.enable_thinking !== "boolean") {
          sendJSON(res, 400, { error: { message: "enable_thinking must be a boolean", type: "invalid_request_error" } });
          return;
        }
        chatTemplateKwargs.enable_thinking = params.enable_thinking;
      }
      if (params.reasoning_effort !== undefined) {
        chatTemplateKwargs.reasoning_effort = params.reasoning_effort;
      }

      const stream = params.stream === true;
      const maxTokens = Math.max(1, Math.min(
        params.max_tokens ?? params.max_completion_tokens ?? args.maxTokens,
        maxModelLen - 1,
      ));
      const temperature = params.temperature ?? args.temperature;
      const topP = params.top_p ?? args.topP;
      const topK = params.top_k ?? args.topK;
      const repetitionPenalty = params.repetition_penalty ?? params.frequency_penalty ?? args.repetitionPenalty;
      const presencePenalty = params.presence_penalty ?? args.presencePenalty;

      if (samplingWorkspace.mtpEnabled) {
        const effectiveK = temperature <= 0 ? 1 : topK > 0 ? Math.min(topK, model.cfg.vocabSize) : 32;
        if (![temperature, topP, topK, repetitionPenalty, presencePenalty].every(Number.isFinite)
          || !Number.isInteger(topK) || !Number.isInteger(effectiveK)) {
          sendJSON(res, 400, { error: { message: "Linear MTP requires finite numeric sampling parameters and an integer top_k", type: "invalid_request_error" } });
          return;
        }
        if (topK > 256 || effectiveK > 256) {
          sendJSON(res, 400, { error: { message: "Linear MTP supports top_k <= 256 (effective top_k must not exceed 256)", type: "invalid_request_error", param: "top_k" } });
          return;
        }
      }

      let stopSequences: string[] = [];
      if (params.stop) {
        if (typeof params.stop === "string") stopSequences = [params.stop];
        else if (Array.isArray(params.stop)) stopSequences = params.stop.filter((s: any) => typeof s === "string");
      }

      const samplingParams: SamplingParams = {
        temperature: Math.max(0, temperature),
        topP,
        topK,
        repetitionPenalty,
        presencePenalty,
        repetitionPenaltyWindow: args.repetitionPenaltyWindow,
      };
      if (args.mtp && !args.noMtp && (samplingParams.repetitionPenalty !== 1 || samplingParams.presencePenalty !== 0)) {
        sendJSON(res, 400, {
          error: {
            message: "MTP decoding currently requires repetition_penalty=1 and presence_penalty=0",
            type: "invalid_request_error",
          },
        });
        return;
      }

      const id = generateId();
      const created = Math.floor(Date.now() / 1000);
      let inputIds: number[];
      let parser: ReturnType<ChatModel["createParser"]>;
      try {
        inputIds = tokenizeMessages(tokenizer, messages, tools, chatTemplateKwargs);
        parser = model.createParser(chatTemplateKwargs);
        if (chatTemplateKwargs.continue_final_message) {
          parser.continueFrom(tokenizer.encode(messages[messages.length - 1].content as string, { add_special_tokens: false }));
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to prepare chat prompt";
        sendJSON(res, 400, { error: { message, type: "invalid_request_error" } });
        return;
      }
      const maxPromptTokens = Math.max(1, maxModelLen - maxTokens);
      if (inputIds.length > maxPromptTokens) {
        inputIds.splice(0, inputIds.length - maxPromptTokens);
      }

      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role !== "user") continue;
        console.log(`Request ${id}: last_user_message=${JSON.stringify(messages[i].content)}`);
        break;
      }

      const completionReq: CompletionRequest = {
        id,
        messages,
        tools,
        chatTemplateKwargs,
        inputIds,
        maxTokens,
        samplingParams,
        stream,
        created,
        stopSequences,
        generatedIds: [],
        generatedText: "",
        reasoningText: "",
        toolCalls: [],
        parser,
        finished: false,
        finishReason: "stop",
        promptTokenCount: inputIds.length,
        cachedTokenCount: 0,
        prefillTokenCount: inputIds.length,
        prefillSeconds: 0,
        performanceLogged: false,
        onOutput: () => {},
        onFinish: () => {},
        onError: () => {},
      };

      if (stream) {
        const continuousUsage = params.stream_options?.continuous_usage_stats === true;
        const streamUsage = () => ({
          prompt_tokens: completionReq.promptTokenCount,
          completion_tokens: completionReq.generatedIds.length,
          total_tokens: completionReq.promptTokenCount + completionReq.generatedIds.length,
        });
        res.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
          "X-Accel-Buffering": "no",
        });

        writeSSE(res, {
          id,
          object: "chat.completion.chunk",
          created,
          model: modelName,
          choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
          ...(continuousUsage ? { usage: streamUsage() } : {}),
        });

        completionReq.onOutput = events => {
          let wroteDelta = false;
          for (const event of events) {
            let delta: Record<string, unknown> | undefined;
            if (event.type === "reasoning_delta") {
              delta = { reasoning_content: event.text };
            } else if (event.type === "content_delta") {
              delta = { content: event.text };
            } else if (event.type === "tool_call") {
              const toolCall = completionReq.toolCalls[event.index];
              delta = { tool_calls: [{ index: event.index, ...toolCall }] };
            } else if (event.raw) {
              delta = { content: event.raw };
            }

            if (delta) {
              wroteDelta = true;
              writeSSE(res, {
                id,
                object: "chat.completion.chunk",
                created,
                model: modelName,
                choices: [{ index: 0, delta, finish_reason: null }],
                ...(continuousUsage ? { usage: streamUsage() } : {}),
              });
            }
          }
          if (continuousUsage && !wroteDelta) {
            writeSSE(res, {
              id,
              object: "chat.completion.chunk",
              created,
              model: modelName,
              choices: [],
              usage: streamUsage(),
            });
          }
        };

        completionReq.onFinish = () => {
          finishOutput(completionReq);

          writeSSE(res, {
            id,
            object: "chat.completion.chunk",
            created,
            model: modelName,
            choices: [{ index: 0, delta: {}, finish_reason: completionReq.finishReason }],
            ...(continuousUsage ? { usage: streamUsage() } : {}),
          });

          if (params.stream_options?.include_usage) {
            writeSSE(res, {
              id,
              object: "chat.completion.chunk",
              created,
              model: modelName,
              choices: [],
              usage: {
                prompt_tokens: completionReq.promptTokenCount,
                completion_tokens: completionReq.generatedIds.length,
                total_tokens: completionReq.promptTokenCount + completionReq.generatedIds.length,
              },
            });
          }

          writeSSEDone(res);
          res.end();
        };
        completionReq.onError = error => {
          if (res.writableEnded) return;
          const message = error instanceof Error ? error.message : "Generation failed";
          writeSSE(res, { error: { message, type: "server_error" } });
          writeSSEDone(res);
          res.end();
        };

        res.on("close", () => {
          if (!res.writableEnded) completionReq.finished = true;
        });
      } else {
        completionReq.onFinish = () => {
          finishOutput(completionReq);
          const message: Record<string, unknown> = {
            role: "assistant",
            content: completionReq.toolCalls.length > 0 && !completionReq.generatedText
              ? null
              : completionReq.generatedText,
          };
          if (completionReq.reasoningText) message.reasoning_content = completionReq.reasoningText;
          if (completionReq.toolCalls.length > 0) message.tool_calls = completionReq.toolCalls;
          sendJSON(res, 200, {
            id,
            object: "chat.completion",
            created,
            model: modelName,
            choices: [{
              index: 0,
              message,
              finish_reason: completionReq.finishReason,
            }],
            usage: {
              prompt_tokens: completionReq.promptTokenCount,
              completion_tokens: completionReq.generatedIds.length,
              total_tokens: completionReq.promptTokenCount + completionReq.generatedIds.length,
            },
          });
        };
        completionReq.onError = error => {
          if (res.writableEnded) return;
          const message = error instanceof Error ? error.message : "Generation failed";
          sendJSON(res, 500, { error: { message, type: "server_error" } });
        };

        res.on("close", () => {
          if (!res.writableEnded) completionReq.finished = true;
        });
      }

      enqueueRequest(completionReq);
    }).catch(err => {
      console.error("Request handling error:", err);
      if (!res.headersSent) {
        sendJSON(res, 500, { error: { message: "Internal server error", type: "internal_error" } });
      }
    });
  }

  function handleTokenize(req: http.IncomingMessage, res: http.ServerResponse): void {
    readBody(req).then(body => {
      let params: any;
      try {
        params = JSON.parse(body);
      } catch {
        sendJSON(res, 400, { error: { message: "Invalid JSON", type: "invalid_request_error" } });
        return;
      }

      const inputMessages = params?.messages as ChatMessage[] | undefined;
      if (!inputMessages || !Array.isArray(inputMessages) || inputMessages.length === 0) {
        sendJSON(res, 400, { error: { message: "messages is required and must be a non-empty array", type: "invalid_request_error" } });
        return;
      }

      try {
        const messages = normalizeMessages(inputMessages);
        const tools = params.tools as unknown[] | undefined;
        if (tools !== undefined && !Array.isArray(tools)) {
          sendJSON(res, 400, { error: { message: "tools must be an array", type: "invalid_request_error" } });
          return;
        }
        const chatTemplateKwargs = params.chat_template_kwargs ?? {};
        if (chatTemplateKwargs === null || typeof chatTemplateKwargs !== "object" || Array.isArray(chatTemplateKwargs)) {
          sendJSON(res, 400, { error: { message: "chat_template_kwargs must be an object", type: "invalid_request_error" } });
          return;
        }
        const tokens = tokenizeMessages(tokenizer, messages, tools, {
          ...chatTemplateKwargs,
          ...(params.continue_final_message !== undefined ? { continue_final_message: params.continue_final_message } : {}),
        });
        sendJSON(res, 200, {
          count: tokens.length,
          max_model_len: maxModelLen,
          tokens,
          token_strs: null,
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : "Tokenization failed";
        sendJSON(res, 400, { error: { message, type: "invalid_request_error" } });
      }
    }).catch(err => {
      console.error("Tokenize request error:", err);
      if (!res.headersSent) {
        sendJSON(res, 500, { error: { message: "Internal server error", type: "internal_error" } });
      }
    });
  }

  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? "/", `http://${req.headers.host}`);

    if (req.method === "OPTIONS") {
      res.writeHead(200, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
      });
      res.end();
      return;
    }

    res.setHeader("Access-Control-Allow-Origin", "*");

    if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
      handleChatCompletions(req, res);
    } else if (req.method === "POST" && url.pathname === "/tokenize") {
      handleTokenize(req, res);
    } else if (req.method === "GET" && url.pathname === "/metrics") {
      sendMetrics(res, metrics, pendingQueue.length, cache, args.batchSize, args.chunkSize, maxModelLen);
    } else if (req.method === "GET" && url.pathname === "/v1/models") {
      const pagedKV = cache.getPagedKV();
      sendJSON(res, 200, {
        object: "list",
        data: [{
          id: modelName,
          object: "model",
          created: Math.floor(Date.now() / 1000),
          owned_by: "local",
          max_model_len: maxModelLen,
          max_running_requests: args.batchSize,
          max_total_num_tokens: pagedKV.maxPages * pagedKV.pageSize,
        }],
      });
    } else if (req.method === "GET" && url.pathname === "/version") {
      sendJSON(res, 200, { version: "glm.js" });
    } else if (req.method === "GET" && url.pathname === "/health") {
      sendJSON(res, 200, { status: "ok", model: modelName });
    } else {
      sendJSON(res, 404, { error: { message: "Not found", type: "not_found_error" } });
    }
  });

  server.listen(args.port, args.host, () => {
    console.log(`OpenAI-compatible server running at http://${args.host}:${args.port}`);
    console.log(`  POST /v1/chat/completions  - Chat completions (streaming & non-streaming)`);
    console.log(`  POST /tokenize             - Tokenize chat messages`);
    console.log(`  GET  /metrics              - Prometheus metrics`);
    console.log(`  GET  /v1/models            - List models`);
    console.log(`  GET  /version              - vLLM-compatible server version`);
    console.log(`  GET  /health               - Health check`);
    console.log(`  CUDA graphs: ${args.noCudaGraph ? "disabled" : "enabled"}`);
    console.log(`  MTP: ${args.mtp && !args.noMtp ? `enabled (draft top-k ${args.mtpDraftTopk.join(",")})` : "disabled"}`);
    if (samplingWorkspace.mtpEnabled) console.log(`  MTP proposals: ${samplingWorkspace.retainProposalsOnGpu ? "GPU-resident" : "host baseline"} (GLM_MTP_GPU_PROPOSALS=0 selects host baseline)`);
    console.log(`  Phased prefill: ${args.phasedPrefill ? "enabled" : "disabled"}`);
    console.log(`  Model: ${modelName}  |  GPU: ${args.gpus.join(",")}  |  chunk-size=${args.chunkSize}  |  batch-size=${args.batchSize}  |  max-pages=${args.maxPages}`);
  });

  let cleaningUp = false;
  const cleanup = async () => {
    if (cleaningUp) return;
    cleaningUp = true;
    await new Promise<void>(resolve => server.close(() => resolve()));
    while (busy) await new Promise(resolve => setTimeout(resolve, 10));
    glm.synchronize();
    captureManager[Symbol.dispose]();
    samplingWorkspace.free();
    cache.free();
    ws.free();
    model.free();
    if (glm instanceof ParallelOps) glm.free();
    for (const device of gpuDevices) device.free();
  };

  if (process.send) {
    process.on("message", message => {
      if ((message as { type?: string })?.type !== "shutdown") return;
      void cleanup().then(() => {
        process.send?.({ type: "stopped" }, () => process.disconnect?.());
      });
    });
  } else {
    process.once("SIGINT", () => void cleanup().then(() => process.exit(0)));
    process.once("SIGTERM", () => void cleanup().then(() => process.exit(0)));
  }
}

if (require.main === module) {
  main().catch(err => {
    console.error("Failed to start server:", err);
    process.exit(1);
  });
}
