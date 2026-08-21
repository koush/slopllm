import http from "node:http";
import crypto from "node:crypto";
import { parentPort } from "node:worker_threads";
import { CaptureManager } from "./capture-manager";
import { type OutputParserEvent } from "./chat-model-parser";
import { ChatModel, ChatCache, ChatTemplateKwargs, loadGenerationConfig, SamplingParams, Tokenizer } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { executePlan, ExecutionWorkspace } from "./execution-workspace";
import { SamplingWorkspace } from "./sampling";
import { createDeviceOps, loadModel, ModelCliArgs, parseModelArgs, resolveModelSelection } from "./model_cli";
import { ParallelOps } from "./parallel_ops";
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
  ctxSize: number;
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
  mtpDraftTopk: number[];
}

function parseArgs(argv: string[]): ServerArgs {
  const args: ServerArgs = {
    ...parseModelArgs(argv),
    port: 8000,
    host: "0.0.0.0",
    ctxSize: 4096,
    batchSize: 1,
    maxPages: 0,
    maxTokens: 512,
    temperature: 0.6,
    topP: 0.95,
    topK: 20,
    repetitionPenalty: 1.0,
    presencePenalty: 0,
    repetitionPenaltyWindow: 64,
    decodeLatency: 0,
    noCudaGraph: false,
    mtpDraftTopk: [],
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--port" && i + 1 < argv.length) args.port = parseInt(argv[++i], 10);
    else if (a === "--host" && i + 1 < argv.length) args.host = argv[++i];
    else if (a === "--ctx-size" && i + 1 < argv.length) args.ctxSize = parseInt(argv[++i], 10);
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
    else if (a === "--mtp-draft-topk" && i + 1 < argv.length) args.mtpDraftTopk = argv[++i].split(",").map(value => parseInt(value.trim(), 10));
    else if (a === "--help" || a === "-h") { printHelp(); process.exit(0); }
  }
  if (args.mtp && args.mtpDraftTopk.length === 0) args.mtpDraftTopk = [2, 2, 2];
  if (args.mtpDraftTopk.some(topk => !Number.isInteger(topk) || topk < 1)) {
    throw new Error(`Invalid --mtp-draft-topk: ${args.mtpDraftTopk.join(",")}`);
  }
  if (args.maxPages === 0) {
    args.maxPages = args.batchSize * Math.ceil(args.ctxSize / PAGE_SIZE);
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
  --ctx-size <int>              Context size / max sequence length (default: 4096)
  --batch-size <int>            Maximum concurrent requests (default: 1)
  --max-pages <int>             KV cache pages (default: batch-size * ceil(ctx-size / 16))
  --max-tokens <int>            Default max completion tokens (default: 512)
  --model-dir <string>          Model directory path (default: auto-detect from HF cache)
  --temperature <float>         Override model default sampling temperature
  --top-p <float>               Override model default top-p
  --top-k <int>                 Override model default top-k
  --repetition-penalty <float>  Override model default repetition penalty
  --presence-penalty <float>    Default presence penalty (default: 0)
  --repetition-penalty-window <int>  Repetition penalty window (default: 64)
  --decode-latency <int>        Artificial delay per decode step in ms (default: 0)
  --no-cuda-graph               Disable CUDA graph capture
  --mtp                         Enable greedy MTP speculative decoding
  --mtp-draft-topk <list>       MTP draft top-k per depth (default: 2,2,2)
  --help, -h                    Show this help message
`);
}

function tokenizeMessages(
  tokenizer: Tokenizer,
  messages: ChatMessage[],
  tools?: unknown[],
  chatTemplateKwargs: ChatTemplateKwargs = {},
): number[] {
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
  if (events.length > 0) req.onOutput(events);
}

function processOutputToken(req: CompletionRequest, tokenId: number): void {
  processOutputEvents(req, req.parser.onToken(tokenId));
}

function finishOutput(req: CompletionRequest): void {
  processOutputEvents(req, req.parser.finish());
  if (req.toolCalls.length > 0) req.finishReason = "tool_calls";
}

interface ActiveSequence {
  request: CompletionRequest;
  lastToken: number;
}

interface ServerMetrics {
  runningRequests: number;
  generationTokensTotal: number;
  promptTokensTotal: number;
  requestSuccessTotal: number;
  prefillTimeSecondsCount: number;
  prefillTimeSecondsSum: number;
}

async function generateMtpBatches(
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
  topks: readonly number[],
): Promise<number> {
  if (!model.planPrefillMtpDraftExtend || !model.planTargetVerification) {
    throw new Error("The selected model does not support plan-based MTP decoding");
  }

  const pagedKV = cache.getPagedKV();
  let admittedRequests = 0;

  while (pendingQueue.length > 0) {
    const cohort = pendingQueue.splice(0, maxBatchSize);
    const requests = [...cohort];
    const notified = new Set<CompletionRequest>();
    admittedRequests += requests.length;
    metrics.runningRequests += requests.length;

    const finish = (req: CompletionRequest): void => {
      if (notified.has(req)) return;
      notified.add(req);
      metrics.runningRequests--;
      metrics.requestSuccessTotal++;
      try { req.onFinish(); } catch {}
    };

    try {
      cache.reset(requests.length);
      const suffixIds = requests.map((req, index) => cache.prefixMatch(index, req.inputIds));
      const mtpInputIds = model.prepareMtpInput(cache, suffixIds);
      for (const req of requests) metrics.promptTokensTotal += req.promptTokenCount;

      const prefillStart = performance.now();
      let draft = (await executePlan(
        captureManager,
        ws,
        model.planPrefillMtpDraftExtend(ws, cache, mtpInputIds, topks),
      )).result;
      const prefillSeconds = (performance.now() - prefillStart) / 1000;
      metrics.prefillTimeSecondsCount += requests.length;
      metrics.prefillTimeSecondsSum += prefillSeconds * requests.length;

      const removeFinishedRows = (): void => {
        const retained: number[] = [];
        for (let i = 0; i < requests.length; i++) {
          if (!requests[i].finished) retained.push(i);
        }
        for (let i = requests.length - 1; i >= 0; i--) {
          if (requests[i].finished) {
            pagedKV.removeSequence(i);
            requests.splice(i, 1);
          }
        }
        if (retained.length !== draft.targetTokens.length) {
          draft = {
            targetTokens: retained.map(index => draft.targetTokens[index]),
            treeTokens: retained.map(index => draft.treeTokens[index]),
            topks: draft.topks,
          };
        }
      };

      for (let i = 0; i < requests.length; i++) {
        const req = requests[i];
        const token = draft.targetTokens[i];
        pagedKV.reportTokens(i, suffixIds[i]);
        pagedKV.reportTokens(i, [token]);
        if (!req.finished) {
          req.generatedIds.push(token);
          metrics.generationTokensTotal++;
          const isEos = eosIds.has(token);
          if (!isEos) processOutputToken(req, token);
          if (isEos || req.generatedIds.length >= req.maxTokens) {
            req.finished = true;
            req.finishReason = isEos ? "stop" : "length";
          }
          if (!req.finished) checkStopSequences(req);
        }
        if (req.finished) finish(req);
      }
      removeFinishedRows();

      while (requests.length > 0) {
        const step = await executePlan(captureManager, ws, model.planTargetVerification(ws, cache, draft));
        draft = step.result.draft;

        for (let i = 0; i < requests.length; i++) {
          const req = requests[i];
          const tokens = step.result.tokens[i];
          pagedKV.reportTokens(i, tokens);

          for (const token of tokens) {
            req.generatedIds.push(token);
            metrics.generationTokensTotal++;
            const isEos = eosIds.has(token);
            if (!isEos) processOutputToken(req, token);
            if (isEos || req.generatedIds.length >= req.maxTokens) {
              req.finished = true;
              req.finishReason = isEos ? "stop" : "length";
            }
            if (!req.finished) checkStopSequences(req);
            if (req.finished) {
              finish(req);
              break;
            }
          }
        }
        removeFinishedRows();

        if (decodeLatencyMs > 0) {
          await new Promise(resolve => setTimeout(resolve, decodeLatencyMs));
        } else {
          await new Promise(resolve => setImmediate(resolve));
        }
      }
    } catch (error) {
      ws.clearTracking();
      for (const req of cohort) {
        if (notified.has(req)) continue;
        notified.add(req);
        req.finished = true;
        metrics.runningRequests = Math.max(0, metrics.runningRequests - 1);
        try { req.onError(error); } catch {}
      }
      throw error;
    }
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
): Promise<number> {
  const pagedKV = cache.getPagedKV();
  const eosToken = [...eosIds][0];
  const active: ActiveSequence[] = [];
  let nextStagingKey = 0;
  let admittedRequests = 0;
  const admitted = new Set<CompletionRequest>();

  try {
    while (active.length > 0 || pendingQueue.length > 0) {
    // 1. Remove finished sequences (reverse order to avoid index shift)
    for (let i = active.length - 1; i >= 0; i--) {
      if (active[i].request.finished) {
        pagedKV.removeSequence(i);
        metrics.runningRequests--;
        metrics.requestSuccessTotal++;
        try { active[i].request.onFinish(); } catch {}
        admitted.delete(active[i].request);
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

      const inputIdsList = newRequests.map(req => req.inputIds);
      for (const req of newRequests) metrics.promptTokensTotal += req.promptTokenCount;

      if (active.length === 0) {
        // Fast path: no active sequences, no staging needed
        pagedKV.reset(newCount);
      } else {
        // Stage active sequences to preserve their KV cache
        for (let i = 0; i < active.length; i++) {
          pagedKV.stageSequence(0, nextStagingKey++);
        }
        pagedKV.reset(newCount);
      }

      // Prefill new requests
      const prefillStart = performance.now();
      const firstTokens = ws.forwardEagerPrefill(model, inputIdsList, cache);
      await glm.synchronizeAsync();
      ws.clearTracking();
      const prefillSeconds = (performance.now() - prefillStart) / 1000;
      metrics.prefillTimeSecondsCount += newCount;
      metrics.prefillTimeSecondsSum += prefillSeconds * newCount;
      metrics.generationTokensTotal += firstTokens.length;

      // Report tokens and check for first-token EOS
      for (let i = 0; i < newCount; i++) {
        pagedKV.reportTokens(i, inputIdsList[i]);
        pagedKV.reportTokens(i, [firstTokens[i]]);
        newRequests[i].generatedIds.push(firstTokens[i]);
        const isEos = eosIds.has(firstTokens[i]);
        if (!isEos) {
          processOutputToken(newRequests[i], firstTokens[i]);
        }
        if (isEos || newRequests[i].generatedIds.length >= newRequests[i].maxTokens) {
          newRequests[i].finished = true;
          newRequests[i].finishReason = isEos ? "stop" : "length";
      }

        if (!newRequests[i].finished) {
          checkStopSequences(newRequests[i]);
        }
      }

      // Unstage active sequences (if any)
      if (active.length > 0) {
        pagedKV.unstageAll();
      }

      // Build new active list matching pagedKV sequence order: [new..., old...]
      const newActiveSequences: ActiveSequence[] = [];
      for (let i = 0; i < newCount; i++) {
        newActiveSequences.push({
          request: newRequests[i],
          lastToken: firstTokens[i],
        });
      }
      newActiveSequences.push(...active);
      active.length = 0;
      active.push(...newActiveSequences);
    }

    // 3. Remove any newly-finished sequences (e.g. first-token EOS)
    for (let i = active.length - 1; i >= 0; i--) {
      if (active[i].request.finished) {
        pagedKV.removeSequence(i);
        metrics.runningRequests--;
        metrics.requestSuccessTotal++;
        try { active[i].request.onFinish(); } catch {}
        admitted.delete(active[i].request);
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
      samplingWorkspace.sample(decodeLogits);
      return undefined;
    }, [useArgmax ? "openai-decode-argmax" : "openai-decode-sample"]);
    await glm.synchronizeAsync();
    let newTokens: number[];
    if (useArgmax) {
      using argmaxResult = decodeResult as Tensor;
      newTokens = argmaxResult.readInt32LEArray();
    } else {
      newTokens = samplingWorkspace.outToken.readInt32LEArray();
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
  waitingRequests: number,
  cache: ChatCache,
): void {
  const pagedKV = cache.getPagedKV();
  const kvUsage = pagedKV.maxPages === 0
    ? 0
    : (pagedKV.maxPages - pagedKV.availablePages.length) / pagedKV.maxPages;
  const lines = [
    "# HELP vllm:num_requests_running GLM.js requests admitted for model execution.",
    "# TYPE vllm:num_requests_running gauge",
    `vllm:num_requests_running ${metrics.runningRequests}`,
    "# HELP vllm:num_requests_waiting GLM.js requests waiting for admission.",
    "# TYPE vllm:num_requests_waiting gauge",
    `vllm:num_requests_waiting ${waitingRequests}`,
    "# HELP vllm:generation_tokens_total GLM.js sampled completion tokens.",
    "# TYPE vllm:generation_tokens_total counter",
    `vllm:generation_tokens_total ${metrics.generationTokensTotal}`,
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
    `vllm:cache_config_info{block_size="${pagedKV.pageSize}",num_gpu_blocks="${pagedKV.maxPages}"} 1`,
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
  const generationConfig = model.cfg.generationConfig ?? loadGenerationConfig(modelDir);
  model.cfg.generationConfig = generationConfig;
  if (!argv.includes("--temperature") && generationConfig.temperature !== undefined) args.temperature = generationConfig.temperature;
  if (!argv.includes("--top-p") && generationConfig.topP !== undefined) args.topP = generationConfig.topP;
  if (!argv.includes("--top-k") && generationConfig.topK !== undefined) args.topK = generationConfig.topK;
  if (!argv.includes("--repetition-penalty") && generationConfig.repetitionPenalty !== undefined) {
    args.repetitionPenalty = generationConfig.repetitionPenalty;
  }
  if (args.mtp && (!model.planPrefillMtpDraftExtend || !model.planTargetVerification)) {
    throw new Error("--mtp requires a model with plan-based MTP decoding support");
  }
  const cache = model.createChatCache(args.maxPages, args.batchSize, args.ctxSize);
  const ws = new ExecutionWorkspace(glm, args.batchSize, args.ctxSize);
  const captureManager = new CaptureManager(glm);
  captureManager.disabled = args.noCudaGraph;
  const samplingWorkspace = new SamplingWorkspace(glm, args.batchSize, model.cfg.vocabSize, args.repetitionPenaltyWindow);
  const tokenizer = model.tokenizer;
  const eosIds = model.eosIds;

  console.log(`Model loaded. ctx-size=${args.ctxSize} batch-size=${args.batchSize} max-pages=${args.maxPages} max-tokens=${args.maxTokens}`);
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
      const warmupSampled = warmupSw.sample(warmupLogits);
      lastToken = warmupSampled.readInt32LEArray()[0];
    }
    cache.reportTokens(0, warmupIds);
    cache.reportTokens(0, [lastToken]);
    for (let i = 0; i < 3; i++) {
      const st = ws.planDecode(model, 1, cache);
      st.setInput([[lastToken]]);
      using hs = model.forward(st);
      using lg = st.computeLogits(hs, model);
      const ns = warmupSw.sample(lg);
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
  const metrics: ServerMetrics = {
    runningRequests: 0,
    generationTokensTotal: 0,
    promptTokensTotal: 0,
    requestSuccessTotal: 0,
    prefillTimeSecondsCount: 0,
    prefillTimeSecondsSum: 0,
  };
  let busy = false;

  async function processQueue(): Promise<void> {
    if (busy) return;
    busy = true;
    try {
      const t0 = performance.now();
      let requestCount = 0;
      try {
        requestCount = args.mtp
          ? await generateMtpBatches(
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
            args.mtpDraftTopk,
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
          );
      } catch (err) {
        console.error("Continuous batch error:", err);
      }
      const elapsed = (performance.now() - t0) / 1000;
      if (elapsed > 0) {
        console.log(`Batch done: ${requestCount} req(s), ${elapsed.toFixed(2)}s`);
      }
    } finally {
      busy = false;
      if (pendingQueue.length > 0) setImmediate(() => { void processQueue(); });
    }
  }

  function enqueueRequest(req: CompletionRequest): void {
    pendingQueue.push(req);
    processQueue();
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
        args.ctxSize,
      ));
      const temperature = params.temperature ?? args.temperature;
      const topP = params.top_p ?? args.topP;
      const topK = params.top_k ?? args.topK;
      const repetitionPenalty = params.repetition_penalty ?? params.frequency_penalty ?? args.repetitionPenalty;
      const presencePenalty = params.presence_penalty ?? args.presencePenalty;

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
      if (args.mtp && (samplingParams.temperature > 0 || samplingParams.repetitionPenalty !== 1 || samplingParams.presencePenalty !== 0)) {
        sendJSON(res, 400, {
          error: {
            message: "MTP decoding currently requires temperature=0, repetition_penalty=1, and presence_penalty=0",
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
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to prepare chat prompt";
        sendJSON(res, 400, { error: { message, type: "invalid_request_error" } });
        return;
      }
      if (inputIds.length > args.ctxSize - maxTokens) {
        inputIds.splice(0, inputIds.length - (args.ctxSize - maxTokens));
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
        onOutput: () => {},
        onFinish: () => {},
        onError: () => {},
      };

      if (stream) {
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
        });

        completionReq.onOutput = events => {
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
              writeSSE(res, {
                id,
                object: "chat.completion.chunk",
                created,
                model: modelName,
                choices: [{ index: 0, delta, finish_reason: null }],
              });
            }
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

        req.on("close", () => {
          completionReq.finished = true;
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

        req.on("close", () => {
          completionReq.finished = true;
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
        const tokens = tokenizeMessages(tokenizer, messages, tools, chatTemplateKwargs);
        sendJSON(res, 200, {
          count: tokens.length,
          max_model_len: args.ctxSize,
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
      sendMetrics(res, metrics, pendingQueue.length, cache);
    } else if (req.method === "GET" && url.pathname === "/v1/models") {
      sendJSON(res, 200, {
        object: "list",
        data: [{
          id: modelName,
          object: "model",
          created: Math.floor(Date.now() / 1000),
          owned_by: "local",
          max_model_len: args.ctxSize,
        }],
      });
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
    console.log(`  GET  /health               - Health check`);
    console.log(`  CUDA graphs: ${args.noCudaGraph ? "disabled" : "enabled"}`);
    console.log(`  MTP: ${args.mtp ? `enabled (draft top-k ${args.mtpDraftTopk.join(",")})` : "disabled"}`);
    console.log(`  Model: ${modelName}  |  GPU: ${args.gpus.join(",")}  |  ctx-size=${args.ctxSize}  |  batch-size=${args.batchSize}  |  max-pages=${args.maxPages}`);
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

  if (parentPort) {
    const port = parentPort;
    port.on("message", message => {
      if ((message as { type?: string })?.type !== "shutdown") return;
      void cleanup().then(() => {
        port.postMessage({ type: "stopped" });
        port.close();
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
