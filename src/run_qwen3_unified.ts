import { GlmOps } from "./glm_ops";
import { Qwen3Model } from "./qwen3_model";
import { Qwen35Model } from "./qwen35_model";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv";
import { ChatModel, ChatCache, SamplingParams, makeSamplingParams, needsSampling, samplingLabel } from "./chat_model";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "./model_path";
import { createInterface } from "node:readline";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const QWEN3_FP8_REPO = "Qwen/Qwen3-0.6B-FP8";
const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";
const QWEN3_EOS = new Set([151645, 151643]);

interface TimingInfo {
  prefillMs: number;
  warmupMs: number[];
  captureMs: number;
  replayMs: number[];
}

interface CliArgs {
  gpu: number;
  maxNewTokens: number;
  maxSeqLen: number;
  warmupSteps: number;
  maxPages: number;
  maxBatch: number;
  noReset: boolean;
  thinking: boolean;
  prompt: string | undefined;
  useQwen35: boolean;
  useFp8: boolean;
  useBatch: boolean;
  useCudaGraph: boolean;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  repetitionPenaltyWindow: number;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    gpu: parseInt(process.env.GLM_GPU ?? "0", 10),
    maxNewTokens: 256,
    maxSeqLen: 4096,
    warmupSteps: 3,
    maxPages: 256,
    maxBatch: 4,
    noReset: true,
    thinking: true,
    prompt: undefined,
    useQwen35: false,
    useFp8: false,
    useBatch: false,
    useCudaGraph: false,
    temperature: 0.6,
    topP: 0.95,
    topK: 0,
    repetitionPenalty: 1.0,
    presencePenalty: 0,
    repetitionPenaltyWindow: 64,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--prompt" && i + 1 < argv.length) args.prompt = argv[++i];
    else if (a === "--max-new-tokens" && i + 1 < argv.length) args.maxNewTokens = parseInt(argv[++i], 10);
    else if (a === "--warmup-steps" && i + 1 < argv.length) args.warmupSteps = parseInt(argv[++i], 10);
    else if (a === "--gpu" && i + 1 < argv.length) args.gpu = parseInt(argv[++i], 10);
    else if (a === "--max-seq-len" && i + 1 < argv.length) args.maxSeqLen = parseInt(argv[++i], 10);
    else if (a === "--max-pages" && i + 1 < argv.length) args.maxPages = parseInt(argv[++i], 10);
    else if (a === "--max-batch" && i + 1 < argv.length) args.maxBatch = parseInt(argv[++i], 10);
    else if (a === "--no-kv-persist") args.noReset = false;
    else if (a === "--no-thinking") args.thinking = false;
    else if (a === "--qwen35") args.useQwen35 = true;
    else if (a === "--fp8") args.useFp8 = true;
    else if (a === "--batch") args.useBatch = true;
    else if (a === "--cuda-graph") args.useCudaGraph = true;
    else if (a === "--temperature" && i + 1 < argv.length) args.temperature = parseFloat(argv[++i]);
    else if (a === "--top-p" && i + 1 < argv.length) args.topP = parseFloat(argv[++i]);
    else if (a === "--top-k" && i + 1 < argv.length) args.topK = parseInt(argv[++i], 10);
    else if (a === "--presence-penalty" && i + 1 < argv.length) args.presencePenalty = parseFloat(argv[++i]);
    else if (a === "--repetition-penalty" && i + 1 < argv.length) args.repetitionPenalty = parseFloat(argv[++i]);
    else if (a === "--repetition-penalty-window" && i + 1 < argv.length) args.repetitionPenaltyWindow = parseInt(argv[++i], 10);
    else if (a === "--greedy") {
      args.temperature = 0;
      args.topK = 0;
      args.topP = 1.0;
      args.repetitionPenalty = 1.0;
      args.presencePenalty = 0;
    }
  }

  if (args.useQwen35 && args.useFp8) {
    console.error("Error: --fp8 is not supported with --qwen35");
    process.exit(1);
  }
  if (args.useBatch && args.useCudaGraph) {
    console.error("Error: --batch and --cuda-graph are mutually exclusive");
    process.exit(1);
  }

  if (args.useQwen35 && args.temperature > 0 && args.topP === 0.95 && args.topK === 0 && args.repetitionPenalty === 1.0 && args.presencePenalty === 0) {
    args.topK = 20;
    args.repetitionPenalty = 1.1;
  }

  return args;
}

function modelLabel(args: CliArgs): string {
  if (args.useQwen35) return "Qwen3.5-0.8B";
  return args.useFp8 ? "Qwen3-0.6B-FP8" : "Qwen3-0.6B";
}

function printTiming(timing: TimingInfo): void {
  if (timing.captureMs > 0) {
    const avg = timing.replayMs.length > 0
      ? timing.replayMs.reduce((a, b) => a + b, 0) / timing.replayMs.length
      : 0;
    console.log(`  [prefill ${timing.prefillMs.toFixed(0)}ms + capture ${timing.captureMs.toFixed(0)}ms + ${timing.warmupMs.length} warmup]${avg > 0 ? ` replay avg=${avg.toFixed(2)}ms (${(1000 / avg).toFixed(0)} tok/s)` : ""}`);
  }
}

// --- Qwen3 + CUDA Graph path (PagedKVCache, single batch) ---

function generateResponseCudaGraph(
  model: Qwen3Model, glm: GlmOps, ws: WorkspaceBuffers, cache: PagedKVCache,
  inputIds: number[], graphExec: number | null,
  maxNewTokens: number, warmupSteps: number, sp: SamplingParams | undefined,
): { tokens: number[]; graphExec: number | null; timing: TimingInfo; matchLen: number } {
  const timing: TimingInfo = { prefillMs: 0, warmupMs: [], captureMs: 0, replayMs: [] };
  const generatedTokens: number[] = [];

  const suffixIds = cache.prefixMatch(0, inputIds);
  cache.appendTokens(0, suffixIds);
  const matchLen = inputIds.length - suffixIds.length;

  const t0 = performance.now();
  const tokens = model.prefillBatch([suffixIds], ws, cache);
  cache.updateIndptr();
  timing.prefillMs = performance.now() - t0;

  let currentToken = tokens[0];
  generatedTokens.push(currentToken);
  cache.appendTokens(0, [currentToken]);

  const tokenHistory = [...inputIds, currentToken];
  let warmupRemaining = graphExec === null ? warmupSteps : 0;
  let capturing = false;

  for (let i = 1; i < maxNewTokens && !QWEN3_EOS.has(currentToken); i++) {
    const t = performance.now();
    const state = model.decodeBatchPlan([currentToken], ws, cache, true);

    if (graphExec !== null) {
      glm.graphLaunch(graphExec);
      glm.synchronize();
    } else {
      if (warmupRemaining === 0 && !capturing) {
        capturing = true;
        glm.graphBeginCapture();
      }
      model.decodeBatchForward(state, ws, cache);
      if (warmupRemaining === 0 && capturing) {
        const graph = glm.graphEndCapture();
        if (!graph) throw new Error("Graph capture failed");
        graphExec = glm.graphInstantiate(graph);
        if (!graphExec) throw new Error("Graph instantiation failed");
        glm.graphDestroy(graph);
      }
      warmupRemaining = Math.max(0, warmupRemaining - 1);
    }

    currentToken = model.decodeBatchRead(state)[0];

    if (capturing) {
      timing.captureMs = performance.now() - t;
      capturing = false;
    } else if (graphExec !== null) {
      timing.replayMs.push(performance.now() - t);
    } else {
      timing.warmupMs.push(performance.now() - t);
    }

    generatedTokens.push(currentToken);
    cache.appendTokens(0, [currentToken]);
    tokenHistory.push(currentToken);
  }

  return { tokens: generatedTokens, graphExec, timing, matchLen };
}

async function interactiveQwen3CudaGraph(
  model: Qwen3Model, glm: GlmOps, ws: WorkspaceBuffers, cache: PagedKVCache,
  tokenizer: any, args: CliArgs,
): Promise<void> {
  const sp = needsSampling(makeSamplingParams(args)) ? makeSamplingParams(args) : undefined;
  const messages: Array<{ role: string; content: string }> = [];
  let graphExec: number | null = null;

  console.log(`${modelLabel(args)}  |  GPU ${args.gpu}  |  max_seq_len=${args.maxSeqLen}  |  kv_persist=${args.noReset ? "on" : "off"}  |  cuda_graph=on`);
  if (sp) console.log(`Sampling: ${samplingLabel(sp)}`);
  console.log(`Graph capture: ${args.warmupSteps} warmup steps, max ${args.maxNewTokens} tokens/turn`);
  console.log("Type /quit to exit, /clear to reset conversation\n");

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const askLine = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    while (true) {
      const userInput = (await askLine("> ")).trim();
      if (!userInput) continue;
      if (userInput === "/quit") break;
      if (userInput === "/clear") {
        messages.length = 0;
        graphExec = null;
        cache.reset(1);
        console.log("Conversation cleared.\n");
        continue;
      }

      messages.push({ role: "user", content: userInput });
      const inputIds = tokenizeMessages(tokenizer, messages, args.thinking);

      if (inputIds.length > args.maxSeqLen - args.maxNewTokens) {
        console.log(`Warning: prompt (${inputIds.length} tokens) too long, truncating conversation`);
        while (inputIds.length > args.maxSeqLen - args.maxNewTokens && messages.length > 1) {
          messages.splice(1, 2);
          const retryIds = tokenizeMessages(tokenizer, messages, args.thinking);
          if (retryIds.length <= args.maxSeqLen - args.maxNewTokens) break;
        }
        if (messages.length === 1 && tokenizeMessages(tokenizer, messages, args.thinking).length > args.maxSeqLen - args.maxNewTokens) {
          console.log("Conversation too long even after truncation. Use /clear to reset.");
          messages.pop();
          continue;
        }
      }

      const result = generateResponseCudaGraph(
        model, glm, ws, cache, inputIds, graphExec,
        args.maxNewTokens, args.warmupSteps, sp,
      );

      graphExec = result.graphExec;

      const cachedLen = cache.cachedTokenIds[0]?.length ?? 0;
      if (cachedLen > 0) {
        if (result.matchLen > 0) {
          const suffixLen = inputIds.length - result.matchLen;
          console.log(`  [cache hit ${result.matchLen}/${cachedLen} tokens, appending ${suffixLen} new]`);
        } else {
          console.log("  [cache miss, full prefill]");
        }
      }

      const responseTokens = result.tokens.filter(t => !QWEN3_EOS.has(t));
      const responseText = tokenizer.decode(responseTokens, { skip_special_tokens: true });
      process.stdout.write(responseText + "\n\n");

      messages.push({ role: "assistant", content: responseText });
      printTiming(result.timing);
    }
  } finally {
    if (graphExec !== null) glm.graphExecDestroy(graphExec);
    cache.free();
    ws.free();
    model.free();
    rl.close();
  }
}

async function singlePromptQwen3CudaGraph(
  model: Qwen3Model, glm: GlmOps, ws: WorkspaceBuffers, cache: PagedKVCache,
  tokenizer: any, args: CliArgs,
): Promise<void> {
  const sp = needsSampling(makeSamplingParams(args)) ? makeSamplingParams(args) : undefined;
  const messages = [{ role: "user", content: args.prompt! }];
  const inputIds = tokenizeMessages(tokenizer, messages, args.thinking);

  console.log(`Prompt: ${args.prompt}`);
  console.log(`Tokens: ${inputIds.length}`);

  cache.reset(1);
  const result = generateResponseCudaGraph(
    model, glm, ws, cache, inputIds, null,
    args.maxNewTokens, args.warmupSteps, sp,
  );

  const responseTokens = result.tokens.filter(t => !QWEN3_EOS.has(t));
  const responseText = tokenizer.decode(responseTokens, { skip_special_tokens: true });

  console.log(`\nPrefill: ${result.timing.prefillMs.toFixed(1)}ms`);
  if (result.timing.captureMs > 0) console.log(`Graph capture: ${result.timing.captureMs.toFixed(1)}ms`);
  if (result.timing.warmupMs.length > 0) {
    const avg = result.timing.warmupMs.reduce((a, b) => a + b, 0) / result.timing.warmupMs.length;
    console.log(`Warmup decode: ${avg.toFixed(2)}ms avg (${result.timing.warmupMs.length} steps)`);
  }
  if (result.timing.replayMs.length > 0) {
    const avg = result.timing.replayMs.reduce((a, b) => a + b, 0) / result.timing.replayMs.length;
    const sorted = [...result.timing.replayMs].sort((a, b) => a - b);
    const p50 = sorted[Math.floor(sorted.length / 2)];
    console.log(`Graph replay: avg=${avg.toFixed(2)}ms  p50=${p50.toFixed(2)}ms  min=${Math.min(...result.timing.replayMs).toFixed(2)}ms  max=${Math.max(...result.timing.replayMs).toFixed(2)}ms  (${result.timing.replayMs.length} steps, ${Math.round(1000 / avg)} tok/s)`);
  }
  console.log(`\nResponse: ${responseText}`);

  if (result.graphExec !== null) glm.graphExecDestroy(result.graphExec);
  cache.free();
  ws.free();
  model.free();
}

// --- Qwen3 + Batch path ---

async function interactiveQwen3Batch(
  model: Qwen3Model, glm: GlmOps, ws: WorkspaceBuffers, cache: ChatCache,
  tokenizer: any, args: CliArgs,
): Promise<void> {
  const enableThinking = args.thinking;

  console.log(`${modelLabel(args)} batch  |  GPU ${args.gpu}  |  max_batch=${args.maxBatch}  |  max_seq_len=${args.maxSeqLen}  |  max_tokens=${args.maxNewTokens}`);
  console.log("Enter prompts one per line. Empty line to submit batch. /clear to reset, /q to quit.");

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const askLine = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    while (true) {
      const prompts: string[] = [];
      while (true) {
        const line = (await askLine(`\nPrompt ${prompts.length + 1} (empty=done): `)).trim();
        if (line === "") break;
        if (line.toLowerCase() === "quit" || line.toLowerCase() === "exit" || line.toLowerCase() === "/q") {
          rl.close();
          return;
        }
        if (line.toLowerCase() === "/clear") {
          prompts.length = 0;
          console.log("Batch cleared.");
          continue;
        }
        prompts.push(line);
      }

      if (prompts.length === 0) continue;
      if (prompts.length > args.maxBatch) {
        console.log(`Too many prompts (${prompts.length}), max batch is ${args.maxBatch}. Reducing to first ${args.maxBatch}.`);
        prompts.length = args.maxBatch;
      }

      const inputIdsList: number[][] = [];
      for (const prompt of prompts) {
        const messages = [{ role: "user" as const, content: prompt }];
        const result = tokenizer.apply_chat_template(messages as any, {
          tokenize: true,
          add_generation_prompt: true,
          return_tensor: false,
          return_dict: true,
          tokenizer_kwargs: { enable_thinking: enableThinking },
        }) as { input_ids: number[] | number[][] };
        const ids = (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
        inputIdsList.push(ids);
      }

      const maxPromptLen = Math.max(...inputIdsList.map(ids => ids.length));
      if (maxPromptLen > args.maxSeqLen) {
        console.log(`Longest prompt (${maxPromptLen} tokens) exceeds max_seq_len (${args.maxSeqLen}). Skipping.`);
        continue;
      }

      const start = Date.now();
      const generatedIds = model.generateBatch(inputIdsList, ws, cache, args.maxNewTokens, QWEN3_EOS);
      const elapsed = (Date.now() - start) / 1000;
      const totalTokens = generatedIds.reduce((sum, ids) => sum + ids.length, 0);

      for (let i = 0; i < prompts.length; i++) {
        const text = tokenizer.decode(generatedIds[i], { skip_special_tokens: true });
        console.log(`\n--- Response ${i + 1} ---`);
        console.log(text);
      }

      console.log(`\n  [${prompts.length} prompts, ${totalTokens} tokens, ${elapsed.toFixed(1)}s, ${(totalTokens / elapsed).toFixed(1)} tok/s]`);
    }
  } finally {
    cache.free();
    ws.free();
    model.free();
    rl.close();
  }
}

// --- Qwen3.5 + Batch path ---

async function interactiveQwen35Batch(
  model: Qwen35Model, cache: ChatCache, ws: WorkspaceBuffers,
  tokenizer: any, args: CliArgs,
): Promise<void> {
  const enableThinking = args.thinking;

  console.log(`${modelLabel(args)} batch  |  GPU ${args.gpu}  |  max_batch=${args.maxBatch}  |  max_seq_len=${args.maxSeqLen}  |  max_tokens=${args.maxNewTokens}`);
  console.log("Enter prompts one per line. Empty line to submit batch. /clear to reset, /q to quit.");

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const askLine = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    while (true) {
      const prompts: string[] = [];
      while (true) {
        const line = (await askLine(`\nPrompt ${prompts.length + 1} (empty=done): `)).trim();
        if (line === "") break;
        if (line.toLowerCase() === "quit" || line.toLowerCase() === "exit" || line.toLowerCase() === "/q") {
          rl.close();
          return;
        }
        if (line.toLowerCase() === "/clear") {
          prompts.length = 0;
          console.log("Batch cleared.");
          continue;
        }
        prompts.push(line);
      }

      if (prompts.length === 0) continue;
      if (prompts.length > args.maxBatch) {
        console.log(`Too many prompts (${prompts.length}), max batch is ${args.maxBatch}. Reducing to first ${args.maxBatch}.`);
        prompts.length = args.maxBatch;
      }

      const inputIdsList: number[][] = [];
      for (const prompt of prompts) {
        const messages = [{ role: "user" as const, content: prompt }];
        const ids = tokenizeMessages(tokenizer, messages, enableThinking);
        inputIdsList.push(ids);
      }

      const maxPromptLen = Math.max(...inputIdsList.map(ids => ids.length));
      if (maxPromptLen > args.maxSeqLen) {
        console.log(`Longest prompt (${maxPromptLen} tokens) exceeds max_seq_len (${args.maxSeqLen}). Skipping.`);
        continue;
      }

      cache.reset(prompts.length);

      const start = Date.now();
      const generatedIds = model.generateBatch(inputIdsList, ws, cache, args.maxNewTokens);
      const elapsed = (Date.now() - start) / 1000;
      const totalTokens = generatedIds.reduce((sum, ids) => sum + ids.length, 0);

      for (let i = 0; i < prompts.length; i++) {
        const text = tokenizer.decode(generatedIds[i], { skip_special_tokens: true });
        console.log(`\n--- Response ${i + 1} ---`);
        console.log(text);
      }

      console.log(`\n  [${prompts.length} prompts, ${totalTokens} tokens, ${elapsed.toFixed(1)}s, ${(totalTokens / elapsed).toFixed(1)} tok/s]`);
    }
  } finally {
    cache.free();
    ws.free();
    model.free();
    rl.close();
  }
}

function tokenizeMessages(
  tokenizer: any,
  messages: Array<{ role: string; content: string }>,
  enableThinking: boolean,
): number[] {
  try {
    const result = tokenizer.apply_chat_template(messages, {
      tokenize: true,
      add_generation_prompt: true,
      return_tensor: false,
      return_dict: true,
      tokenizer_kwargs: { enable_thinking: enableThinking },
    }) as { input_ids: number[] | number[][] };
    return (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
  } catch {
    const result = tokenizer.apply_chat_template(messages, {
      tokenize: true,
      add_generation_prompt: true,
      return_tensor: false,
      return_dict: true,
    }) as { input_ids: number[] | number[][] };
    return (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
  }
}

interface StreamArgs {
  maxSeqLen: number;
  maxNewTokens: number;
  thinking: boolean;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  repetitionPenaltyWindow: number;
}

async function interactiveStream(
  model: ChatModel, cache: ChatCache, ws: WorkspaceBuffers,
  tokenizer: any, args: StreamArgs,
): Promise<void> {
  const sp = needsSampling(makeSamplingParams(args)) ? makeSamplingParams(args) : undefined;
  const eosIds = model.eosIds;
  const messages: Array<{ role: string; content: string }> = [];

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const askLine = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    while (true) {
      const userInput = (await askLine("> ")).trim();
      if (!userInput) continue;
      if (userInput.toLowerCase() === "quit" || userInput.toLowerCase() === "exit" || userInput.toLowerCase() === "/q") break;
      if (userInput.toLowerCase() === "/clear") {
        messages.length = 0;
        cache.reset(1);
        console.log("Conversation cleared.\n");
        continue;
      }

      messages.push({ role: "user", content: userInput });
      const inputIds = tokenizeMessages(tokenizer, messages, args.thinking);

      if (inputIds.length > args.maxSeqLen - args.maxNewTokens) {
        console.log(`Warning: prompt (${inputIds.length} tokens) too long, truncating conversation`);
        while (inputIds.length > args.maxSeqLen - args.maxNewTokens && messages.length > 1) {
          messages.splice(1, 2);
          const retryIds = tokenizeMessages(tokenizer, messages, args.thinking);
          if (retryIds.length <= args.maxSeqLen - args.maxNewTokens) break;
        }
        if (messages.length === 1 && tokenizeMessages(tokenizer, messages, args.thinking).length > args.maxSeqLen - args.maxNewTokens) {
          console.log("Conversation too long even after truncation. Use /clear to reset.");
          messages.pop();
          continue;
        }
      }

      process.stdout.write("Assistant: ");
      const t0 = performance.now();
      let tokCount = 0;
      const generatedIds: number[] = [];

      for (const tokenId of model.chatStream([inputIds], cache, ws, args.maxNewTokens, eosIds, sp)) {
        generatedIds.push(tokenId);
        tokCount++;
        const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
        process.stdout.write(chunk);
        if (eosIds.has(tokenId)) break;
      }

      const elapsed = performance.now() - t0;
      console.log(`\n  [${tokCount} tokens, ${(elapsed / 1000).toFixed(1)}s, ${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s]`);

      const responseText = tokenizer.decode(generatedIds.filter(t => !eosIds.has(t)), { skip_special_tokens: true });
      messages.push({ role: "assistant", content: responseText });
    }
  } finally {
    cache.free();
    model.free();
    rl.close();
  }
}

async function singlePromptStream(
  model: ChatModel, cache: ChatCache, ws: WorkspaceBuffers,
  tokenizer: any, args: StreamArgs & { prompt: string },
): Promise<void> {
  const sp = needsSampling(makeSamplingParams(args)) ? makeSamplingParams(args) : undefined;
  const eosIds = model.eosIds;
  const messages = [{ role: "user", content: args.prompt }];
  const inputIds = tokenizeMessages(tokenizer, messages, args.thinking);

  console.log(`Prompt: ${args.prompt}`);
  console.log(`Tokens: ${inputIds.length}${sp ? "  |  " + samplingLabel(sp) : ""}`);

  process.stdout.write("\n");
  const t0 = performance.now();
  let tokCount = 0;
  const generatedIds: number[] = [];

  for (const tokenId of model.chatStream([inputIds], cache, ws, args.maxNewTokens, eosIds, sp)) {
    generatedIds.push(tokenId);
    tokCount++;
    const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
    process.stdout.write(chunk);
    if (eosIds.has(tokenId)) break;
  }

  const elapsed = performance.now() - t0;
  console.log(`\n\n${tokCount} tokens in ${elapsed.toFixed(1)}ms (${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s)`);

  cache.free();
  model.free();
}

function streamArgs(args: CliArgs): StreamArgs {
  return {
    maxSeqLen: args.maxSeqLen,
    maxNewTokens: args.maxNewTokens,
    thinking: args.thinking,
    temperature: args.temperature,
    topP: args.topP,
    topK: args.topK,
    repetitionPenalty: args.repetitionPenalty,
    presencePenalty: args.presencePenalty,
    repetitionPenaltyWindow: args.repetitionPenaltyWindow,
  };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  process.env.CUDA_VISIBLE_DEVICES = String(args.gpu);

  const glm = new GlmOps(0);

  if (args.useQwen35) {
    console.log(`Loading ${modelLabel(args)} on GPU ${args.gpu}...`);
    const maxBatch = args.useBatch ? args.maxBatch : 1;
    const model = Qwen35Model.fromPretrained(glm, QWEN35_REPO, maxBatch, args.maxSeqLen);
    const ws = new WorkspaceBuffers(glm);
    const cache = model.createChatCache(args.maxPages);

    const modelDir = resolveModelPath(QWEN35_REPO);
    const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });

    if (args.useBatch) {
      await interactiveQwen35Batch(model, cache, ws, tokenizer, args);
    } else {
      const sp = makeSamplingParams(args);
      console.log(`${modelLabel(args)}  |  GPU ${args.gpu}  |  max_seq_len=${args.maxSeqLen}  |  streaming`);
      console.log(`Max ${args.maxNewTokens} tokens/turn  |  ${samplingLabel(sp)}`);
      if (args.prompt) {
        console.log("Type /quit to exit, /clear to reset conversation\n");
      }

      if (args.prompt) {
        await singlePromptStream(model as ChatModel, cache, ws, tokenizer, { ...streamArgs(args), prompt: args.prompt });
      } else {
        await interactiveStream(model as ChatModel, cache, ws, tokenizer, streamArgs(args));
      }
    }
  } else {
    const repoId = args.useFp8 ? QWEN3_FP8_REPO : QWEN3_REPO;
    console.log(`Loading ${modelLabel(args)} on GPU ${args.gpu}...`);

    if (args.useBatch) {
      const model = Qwen3Model.fromPretrained(glm, repoId, args.maxBatch, args.maxSeqLen);
      const ws = new WorkspaceBuffers(glm);
      const cache = model.createChatCache(args.maxPages);

      const modelDir = resolveModelPath(repoId);
      const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });

      await interactiveQwen3Batch(model, glm, ws, cache, tokenizer, args);
    } else if (args.useCudaGraph) {
      const model = Qwen3Model.fromPretrained(glm, repoId, 1, args.maxSeqLen);
      const ws = new WorkspaceBuffers(glm);
      const cache = model.createChatCache(args.maxPages) as PagedKVCache;

      const modelDir = resolveModelPath(repoId);
      const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });

      if (args.prompt) {
        await singlePromptQwen3CudaGraph(model, glm, ws, cache, tokenizer, args);
      } else {
        await interactiveQwen3CudaGraph(model, glm, ws, cache, tokenizer, args);
      }
    } else {
      const model = Qwen3Model.fromPretrained(glm, repoId, 1, args.maxSeqLen);
      const ws = new WorkspaceBuffers(glm);
      const cache = model.createChatCache();

      const modelDir = resolveModelPath(repoId);
      const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });

      const sp = needsSampling(makeSamplingParams(args)) ? makeSamplingParams(args) : undefined;
      console.log(`${modelLabel(args)}  |  GPU ${args.gpu}  |  max_seq_len=${args.maxSeqLen}  |  streaming`);
      if (sp) console.log(`Sampling: ${samplingLabel(sp)}`);
      console.log(`Max ${args.maxNewTokens} tokens/turn`);
      console.log("Type a message to chat. /clear to reset, /quit to exit.\n");

      if (args.prompt) {
        await singlePromptStream(model as ChatModel, cache, ws, tokenizer, { ...streamArgs(args), prompt: args.prompt });
      } else {
        await interactiveStream(model as ChatModel, cache, ws, tokenizer, streamArgs(args));
      }
    }
  }
}

main().catch((err) => {
  console.error("Failed:", err);
  process.exit(1);
});
