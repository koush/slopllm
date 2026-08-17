import http from "node:http";
import crypto from "node:crypto";
import { AutoTokenizer } from "@huggingface/transformers";
import fs from "node:fs";
import path from "node:path";
import { ChatModel, ChatCache, SamplingParams } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { ExecutionWorkspace } from "./execution-workspace";
import { GlmOps } from "./glm_ops";
import { Qwen3Model } from "./qwen3_model";
import { SamplingWorkspace } from "./tensor";
import { resolveModelPath } from "./model_path";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const MODEL_NAME = "qwen3-0.6b";
const PAGE_SIZE = 64;

interface ServerArgs {
  port: number;
  host: string;
  gpu: number;
  ctxSize: number;
  batchSize: number;
  maxPages: number;
  maxTokens: number;
  modelDir: string | undefined;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  repetitionPenaltyWindow: number;
  decodeLatency: number;
}

function parseArgs(argv: string[]): ServerArgs {
  const args: ServerArgs = {
    port: 8010,
    host: "0.0.0.0",
    gpu: parseInt(process.env.GLM_GPUS ?? process.env.GLM_GPU ?? "0", 10),
    ctxSize: 4096,
    batchSize: 1,
    maxPages: 0,
    maxTokens: 512,
    modelDir: undefined,
    temperature: 0.6,
    topP: 0.95,
    topK: 20,
    repetitionPenalty: 1.0,
    presencePenalty: 0,
    repetitionPenaltyWindow: 64,
    decodeLatency: 0,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--port" && i + 1 < argv.length) args.port = parseInt(argv[++i], 10);
    else if (a === "--host" && i + 1 < argv.length) args.host = argv[++i];
    else if (a === "--gpu" && i + 1 < argv.length) args.gpu = parseInt(argv[++i], 10);
    else if (a === "--ctx-size" && i + 1 < argv.length) args.ctxSize = parseInt(argv[++i], 10);
    else if (a === "--batch-size" && i + 1 < argv.length) args.batchSize = parseInt(argv[++i], 10);
    else if (a === "--max-pages" && i + 1 < argv.length) args.maxPages = parseInt(argv[++i], 10);
    else if (a === "--max-tokens" && i + 1 < argv.length) args.maxTokens = parseInt(argv[++i], 10);
    else if (a === "--model-dir" && i + 1 < argv.length) args.modelDir = argv[++i];
    else if (a === "--temperature" && i + 1 < argv.length) args.temperature = parseFloat(argv[++i]);
    else if (a === "--top-p" && i + 1 < argv.length) args.topP = parseFloat(argv[++i]);
    else if (a === "--top-k" && i + 1 < argv.length) args.topK = parseInt(argv[++i], 10);
    else if (a === "--repetition-penalty" && i + 1 < argv.length) args.repetitionPenalty = parseFloat(argv[++i]);
    else if (a === "--presence-penalty" && i + 1 < argv.length) args.presencePenalty = parseFloat(argv[++i]);
    else if (a === "--repetition-penalty-window" && i + 1 < argv.length) args.repetitionPenaltyWindow = parseInt(argv[++i], 10);
    else if (a === "--decode-latency" && i + 1 < argv.length) args.decodeLatency = parseInt(argv[++i], 10);
    else if (a === "--help" || a === "-h") { printHelp(); process.exit(0); }
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
  --ctx-size <int>              Context size / max sequence length (default: 4096)
  --batch-size <int>            Maximum concurrent requests (default: 1)
  --max-pages <int>             KV cache pages (default: batch-size * ceil(ctx-size / 16))
  --max-tokens <int>            Default max completion tokens (default: 512)
  --model-dir <string>          Model directory path (default: auto-detect from HF cache)
  --temperature <float>         Default sampling temperature (default: 0.6)
  --top-p <float>               Default top-p (default: 0.95)
  --top-k <int>                 Default top-k (default: 20)
  --repetition-penalty <float>  Default repetition penalty (default: 1.0)
  --presence-penalty <float>    Default presence penalty (default: 0)
  --repetition-penalty-window <int>  Repetition penalty window (default: 64)
  --decode-latency <int>        Artificial delay per decode step in ms (default: 0)
  --help, -h                    Show this help message
`);
}

function tokenizeMessages(
  tokenizer: any,
  messages: Array<{ role: string; content: string }>,
  chatTemplate?: string,
): number[] {
  try {
    const opts: any = {
      tokenize: true,
      add_generation_prompt: true,
      return_tensor: false,
      return_dict: true,
    };
    if (chatTemplate) opts.chat_template = chatTemplate;
    const result = tokenizer.apply_chat_template(messages, opts) as { input_ids: number[] | number[][] };
    return (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
  } catch {
    const text = messages.map(m => `<|${m.role}|>\n${m.content}`).join("\n") + "\n\n\n";
    return tokenizer.encode(text, { add_special_tokens: false });
  }
}

class TokenStreamDecoder {
  private tokenCache: number[] = [];
  private emittedText = "";

  push(tokenId: number, tokenizer: any, skipSpecialTokens: boolean): string {
    this.tokenCache.push(tokenId);
    const text = tokenizer.decode(this.tokenCache, { skip_special_tokens: skipSpecialTokens });

    let safeEnd = text.length;
    while (safeEnd > 0 && text.charCodeAt(safeEnd - 1) === 0xFFFD) {
      safeEnd--;
    }

    const safeText = text.slice(0, safeEnd);
    const delta = safeText.startsWith(this.emittedText)
      ? safeText.slice(this.emittedText.length)
      : this._diffFrom(safeText);
    this.emittedText = safeText;
    return delta;
  }

  flush(tokenizer: any, skipSpecialTokens: boolean): string {
    if (this.tokenCache.length === 0) return "";
    const text = tokenizer.decode(this.tokenCache, { skip_special_tokens: skipSpecialTokens });
    const delta = text.startsWith(this.emittedText)
      ? text.slice(this.emittedText.length)
      : this._diffFrom(text);
    this.tokenCache = [];
    this.emittedText = "";
    return delta;
  }

  private _diffFrom(text: string): string {
    let i = 0;
    while (i < this.emittedText.length && i < text.length && this.emittedText[i] === text[i]) {
      i++;
    }
    return text.slice(i);
  }
}

interface CompletionRequest {
  id: string;
  messages: Array<{ role: string; content: string }>;
  maxTokens: number;
  samplingParams: SamplingParams;
  stream: boolean;
  created: number;
  stopSequences: string[];
  generatedIds: number[];
  generatedText: string;
  decoder: TokenStreamDecoder;
  finished: boolean;
  finishReason: string;
  promptTokenCount: number;
  onToken: (tokenId: number, tokenText: string) => void;
  onFinish: () => void;
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

async function generateContinuousBatch(
  model: ChatModel,
  ws: ExecutionWorkspace,
  glm: DeviceOps,
  cache: ChatCache,
  tokenizer: any,
  chatTemplate: string | undefined,
  eosIds: Set<number>,
  pendingQueue: CompletionRequest[],
  maxSeqLen: number,
  maxBatchSize: number,
  decodeLatencyMs: number,
  metrics: ServerMetrics,
): Promise<void> {
  const pagedKV = cache.getPagedKV();
  const eosToken = [...eosIds][0];
  const active: ActiveSequence[] = [];
  let nextStagingKey = 0;
  using samplingWorkspace = new SamplingWorkspace(glm, maxBatchSize, model.cfg.vocabSize, 64);

  while (active.length > 0 || pendingQueue.length > 0) {
    // 1. Remove finished sequences (reverse order to avoid index shift)
    for (let i = active.length - 1; i >= 0; i--) {
      if (active[i].request.finished) {
        pagedKV.removeSequence(i);
        metrics.runningRequests--;
        metrics.requestSuccessTotal++;
        try { active[i].request.onFinish(); } catch {}
        active.splice(i, 1);
      }
    }

    // 2. Prefill new requests if there's room
    const availableSlots = maxBatchSize - active.length;
    if (pendingQueue.length > 0 && availableSlots > 0) {
      const newCount = Math.min(pendingQueue.length, availableSlots);
      const newRequests = pendingQueue.splice(0, newCount);
      metrics.runningRequests += newCount;

      const inputIdsList: number[][] = [];
      for (const req of newRequests) {
        const ids = tokenizeMessages(tokenizer, req.messages, chatTemplate);
        req.maxTokens = Math.max(1, req.maxTokens);
        if (ids.length > maxSeqLen - req.maxTokens) {
          ids.splice(0, ids.length - (maxSeqLen - req.maxTokens));
        }
        inputIdsList.push(ids);
        req.promptTokenCount = ids.length;
        metrics.promptTokensTotal += ids.length;
      }

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
          const chunk = newRequests[i].decoder.push(firstTokens[i], tokenizer, true);
          newRequests[i].generatedText += chunk;
          newRequests[i].onToken(firstTokens[i], chunk);
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
        active.splice(i, 1);
      }
    }

    if (active.length === 0) continue;

    // 4. Update sampling workspace for current batch composition
    const params = active.map(a => a.request.samplingParams);
    const tokenHistories = pagedKV.sequences.slice(0, active.length).map(s => s.getTokenIds());
    samplingWorkspace.updateSampler(params, tokenHistories);

    // 5. Decode one step
    const inputTokens = active.map(a => a.lastToken);
    const state = ws.planDecode(model, active.length, cache);
    state.setInput([inputTokens]);
    ws.positionStep(state, model);
    using hiddenStates = model.forward(state);
    using decodeLogits = state.computeLogits(hiddenStates, model);
    const newSampled = samplingWorkspace.sample(decodeLogits);
    const newTokens = newSampled.readInt32LEArray();
    metrics.generationTokensTotal += newTokens.length;

    // 6. Process decoded tokens
    for (let i = 0; i < active.length; i++) {
      const req = active[i].request;
      pagedKV.reportTokens(i, [newTokens[i]]);
      active[i].lastToken = newTokens[i];
      req.generatedIds.push(newTokens[i]);
      const isEos = eosIds.has(newTokens[i]);
      if (!isEos) {
        const chunk = req.decoder.push(newTokens[i], tokenizer, true);
        req.generatedText += chunk;
        req.onToken(newTokens[i], chunk);
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
}

async function generateBatch(
  model: ChatModel,
  ws: ExecutionWorkspace,
  glm: DeviceOps,
  cache: ChatCache,
  tokenizer: any,
  eosIds: Set<number>,
  requests: CompletionRequest[],
  maxSeqLen: number,
): Promise<void> {
  if (requests.length === 0) return;
  const batchSize = requests.length;
  for (const req of requests) {
    req.maxTokens = Math.max(1, req.maxTokens);
  }

  try {
    cache.reset(batchSize);

    const inputIdsList: number[][] = [];
    for (const req of requests) {
      const ids = tokenizeMessages(tokenizer, req.messages);
      const maxTok = req.maxTokens;
      if (ids.length > maxSeqLen - maxTok) {
        ids.splice(0, ids.length - (maxSeqLen - maxTok));
      }
      inputIdsList.push(ids);
      req.promptTokenCount = ids.length;
    }

    const samplingParams = requests.map(r => r.samplingParams);
    const maxWindow = Math.max(...samplingParams.map(p => p.repetitionPenaltyWindow));
    using samplingWorkspace = new SamplingWorkspace(
      glm, requests.length, model.cfg.vocabSize, maxWindow,
    );
    samplingWorkspace.updateSampler(samplingParams, inputIdsList);

    using prefillLogits = ws.forwardPrefill(model, inputIdsList, cache);
    const firstSampled = samplingWorkspace.sample(prefillLogits);
    const firstTokens = firstSampled.readInt32LEArray();

    for (let i = 0; i < batchSize; i++) {
      cache.reportTokens(i, inputIdsList[i]);
      cache.reportTokens(i, [firstTokens[i]]);
      requests[i].generatedIds.push(firstTokens[i]);
      const isEos = eosIds.has(firstTokens[i]);
      if (!isEos) {
        const chunk = requests[i].decoder.push(firstTokens[i], tokenizer, true);
        requests[i].generatedText += chunk;
        requests[i].onToken(firstTokens[i], chunk);
      }
      if (isEos || requests[i].generatedIds.length >= requests[i].maxTokens) {
        requests[i].finished = true;
        requests[i].finishReason = isEos ? "stop" : "length";
      }
      if (!requests[i].finished) {
        checkStopSequences(requests[i]);
      }
    }

    const finished = requests.map(r => r.finished);
    let lastTokens = [...firstTokens];
    const eosToken = [...eosIds][0];
    const maxSteps = Math.max(...requests.map(r => r.maxTokens));

    for (let step = 1; step < maxSteps; step++) {
      if (finished.every(f => f)) break;

      const inputTokens = lastTokens.map((t, i) => finished[i] ? eosToken : t);

      const state = ws.planDecode(model, batchSize, cache);
      state.setInput([inputTokens]);
      ws.positionStep(state, model);
      using hiddenStates = model.forward(state);
      using decodeLogits = state.computeLogits(hiddenStates, model);
      const newSampled = samplingWorkspace.sample(decodeLogits);
      const newTokens = newSampled.readInt32LEArray();

      for (let i = 0; i < batchSize; i++) {
        if (!finished[i]) {
          cache.reportTokens(i, [newTokens[i]]);
          requests[i].generatedIds.push(newTokens[i]);
          const isEos = eosIds.has(newTokens[i]);
          if (!isEos) {
            const chunk = requests[i].decoder.push(newTokens[i], tokenizer, true);
            requests[i].generatedText += chunk;
            requests[i].onToken(newTokens[i], chunk);
          }
          if (isEos || requests[i].generatedIds.length >= requests[i].maxTokens) {
            finished[i] = true;
            requests[i].finished = true;
            requests[i].finishReason = isEos ? "stop" : "length";
          }
          if (!finished[i]) {
            checkStopSequences(requests[i]);
            if (requests[i].finished) {
              finished[i] = true;
            }
          }
        } else {
          cache.reportTokens(i, [eosToken]);
        }
        lastTokens[i] = finished[i] ? eosToken : newTokens[i];
      }

      await new Promise(resolve => setImmediate(resolve));
    }

    for (const req of requests) {
      if (!req.finished) {
        req.finished = true;
        req.finishReason = "length";
      }
    }
  } finally {
    for (const req of requests) {
      try { req.onFinish(); } catch {}
    }
  }
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

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  const modelDir = args.modelDir ?? resolveModelPath(QWEN3_REPO);
  const tokenizerDir = args.modelDir && fs.existsSync(path.join(args.modelDir, "tokenizer_config.json"))
    ? args.modelDir : resolveModelPath(QWEN3_REPO);

  console.log(`Loading model from ${modelDir}...`);
  const glm = new GlmOps(args.gpu, undefined, undefined);
  const model = await Qwen3Model.fromPretrained(glm, modelDir);
  const cache = model.createChatCache(args.maxPages, args.batchSize, args.ctxSize);
  const ws = new ExecutionWorkspace(glm, args.batchSize, args.ctxSize);
  const tokenizer = await AutoTokenizer.from_pretrained(tokenizerDir, { local_files_only: true });
  const chatTemplatePath = path.join(tokenizerDir, "chat_template.jinja");
  const chatTemplate = fs.existsSync(chatTemplatePath) ? fs.readFileSync(chatTemplatePath, "utf-8") : undefined;
  const eosIds = model.eosIds;

  console.log(`Model loaded. ctx-size=${args.ctxSize} batch-size=${args.batchSize} max-pages=${args.maxPages} max-tokens=${args.maxTokens}`);

  {
    console.log("Warming up...");
    const warmupIds = tokenizeMessages(tokenizer, [{ role: "user", content: "Hello" }], chatTemplate);
    cache.reset(1);
    using warmupSw = new SamplingWorkspace(glm, 1, model.cfg.vocabSize, args.repetitionPenaltyWindow);
    warmupSw.updateSampler([makeSamplingParamsHelper(args)], [warmupIds]);
    using warmupLogits = ws.forwardPrefill(model, [warmupIds], cache);
    const warmupSampled = warmupSw.sample(warmupLogits);
    let lastToken = warmupSampled.readInt32LEArray()[0];
    cache.reportTokens(0, warmupIds);
    cache.reportTokens(0, [lastToken]);
    for (let i = 0; i < 3; i++) {
      const st = ws.planDecode(model, 1, cache);
      st.setInput([[lastToken]]);
      ws.positionStep(st, model);
      using hs = model.forward(st);
      using lg = st.computeLogits(hs, model);
      const ns = warmupSw.sample(lg);
      lastToken = ns.readInt32LEArray()[0];
      cache.reportTokens(0, [lastToken]);
    }
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
      let totalTokens = 0;
      let totalPrompt = 0;
      const requestCount = pendingQueue.length;
      try {
        await generateContinuousBatch(model, ws, glm, cache, tokenizer, chatTemplate, eosIds, pendingQueue, args.ctxSize, args.batchSize, args.decodeLatency, metrics);
      } catch (err) {
        console.error("Continuous batch error:", err);
      }
      for (let i = 0; i < requestCount; i++) {
        // tokens already counted per-request via onToken
      }
      const elapsed = (performance.now() - t0) / 1000;
      if (elapsed > 0) {
        console.log(`Batch done: ${requestCount} req(s), ${elapsed.toFixed(2)}s`);
      }
    } finally {
      busy = false;
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

      const messages = params.messages as Array<{ role: string; content: string }> | undefined;
      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        sendJSON(res, 400, { error: { message: "messages is required and must be a non-empty array", type: "invalid_request_error" } });
        return;
      }

      const stream = params.stream === true;
      const maxTokens = Math.min(
        params.max_tokens ?? params.max_completion_tokens ?? args.maxTokens,
        args.ctxSize,
      );
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

      const id = generateId();
      const created = Math.floor(Date.now() / 1000);

      const completionReq: CompletionRequest = {
        id,
        messages,
        maxTokens,
        samplingParams,
        stream,
        created,
        stopSequences,
        generatedIds: [],
        generatedText: "",
        decoder: new TokenStreamDecoder(),
        finished: false,
        finishReason: "stop",
        promptTokenCount: 0,
        onToken: () => {},
        onFinish: () => {},
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
          model: MODEL_NAME,
          choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
        });

        let lastSentLen = 0;
        completionReq.onToken = (_tokenId: number, _tokenText: string) => {
          if (completionReq.finished && completionReq.stopSequences.length > 0) return;
          const currentText = completionReq.generatedText;
          if (currentText.length > lastSentLen) {
            const delta = currentText.slice(lastSentLen);
            lastSentLen = currentText.length;
            writeSSE(res, {
              id,
              object: "chat.completion.chunk",
              created,
              model: MODEL_NAME,
              choices: [{ index: 0, delta: { content: delta }, finish_reason: null }],
            });
          }
        };

        completionReq.onFinish = () => {
          const remaining = completionReq.decoder.flush(tokenizer, true);
          if (remaining) completionReq.generatedText += remaining;
          const currentText = completionReq.generatedText;
          if (currentText.length > lastSentLen) {
            const delta = currentText.slice(lastSentLen);
            lastSentLen = currentText.length;
            writeSSE(res, {
              id,
              object: "chat.completion.chunk",
              created,
              model: MODEL_NAME,
              choices: [{ index: 0, delta: { content: delta }, finish_reason: null }],
            });
          }

          writeSSE(res, {
            id,
            object: "chat.completion.chunk",
            created,
            model: MODEL_NAME,
            choices: [{ index: 0, delta: {}, finish_reason: completionReq.finishReason }],
          });

          if (params.stream_options?.include_usage) {
            writeSSE(res, {
              id,
              object: "chat.completion.chunk",
              created,
              model: MODEL_NAME,
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

        req.on("close", () => {
          completionReq.finished = true;
        });
      } else {
        completionReq.onFinish = () => {
          const remaining = completionReq.decoder.flush(tokenizer, true);
          if (remaining) completionReq.generatedText += remaining;
          const content = completionReq.generatedText;
          sendJSON(res, 200, {
            id,
            object: "chat.completion",
            created,
            model: MODEL_NAME,
            choices: [{
              index: 0,
              message: { role: "assistant", content },
              finish_reason: completionReq.finishReason,
            }],
            usage: {
              prompt_tokens: completionReq.promptTokenCount,
              completion_tokens: completionReq.generatedIds.length,
              total_tokens: completionReq.promptTokenCount + completionReq.generatedIds.length,
            },
          });
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

      const messages = params?.messages as Array<{ role: string; content: string }> | undefined;
      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        sendJSON(res, 400, { error: { message: "messages is required and must be a non-empty array", type: "invalid_request_error" } });
        return;
      }

      try {
        const tokens = tokenizeMessages(tokenizer, messages, chatTemplate);
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
          id: MODEL_NAME,
          object: "model",
          created: Math.floor(Date.now() / 1000),
          owned_by: "local",
          max_model_len: args.ctxSize,
        }],
      });
    } else if (req.method === "GET" && url.pathname === "/health") {
      sendJSON(res, 200, { status: "ok", model: MODEL_NAME });
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
    console.log(`  Model: ${MODEL_NAME}  |  GPU: ${args.gpu}  |  ctx-size: ${args.ctxSize}  |  batch-size: ${args.batchSize}  |  max-pages: ${args.maxPages}`);
  });
}

main().catch(err => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
