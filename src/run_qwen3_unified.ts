import { GlmOps } from "./glm_ops";
import { Qwen3Model } from "./qwen3_model";
import { Qwen35Model } from "./qwen35_model";
import { WorkspaceBuffers } from "./paged_kv";
import { ChatModel, ChatCache, SamplingParams, makeSamplingParams, needsSampling, samplingLabel } from "./chat_model";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "./model_path";
import { createInterface } from "node:readline";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const QWEN3_FP8_REPO = "Qwen/Qwen3-0.6B-FP8";
const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";

export interface GraphState {
  graphExec: number | null;
  warmupRemaining: number;
}

interface CliArgs {
  gpu: number;
  maxNewTokens: number;
  maxSeqLen: number;
  warmupSteps: number;
  maxPages: number;
  maxBatch: number;
  noReset: boolean;
  prompt: string | undefined;
  useQwen35: boolean;
  useFp8: boolean;
  useBatch: boolean;
  noCudaGraph: boolean;
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
    prompt: undefined,
    useQwen35: false,
    useFp8: false,
    useBatch: false,
    noCudaGraph: false,
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
    else if (a === "--qwen35") args.useQwen35 = true;
    else if (a === "--fp8") args.useFp8 = true;
    else if (a === "--batch") args.useBatch = true;
    else if (a === "--no-cuda-graph") args.noCudaGraph = true;
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

// --- Generation primitives ---

export function* generateStream(
  model: ChatModel, glm: GlmOps, ws: WorkspaceBuffers, cache: ChatCache,
  inputIds: number[], maxNewTokens: number, eosIds: Set<number>,
  sampling: SamplingParams | undefined, graphState?: GraphState,
): Generator<number> {
  const suffixIds = cache.prefixMatch(0, inputIds);
  cache.appendTokens(0, suffixIds);

  const firstTokens = model.forwardEager([suffixIds], ws, cache);
  let currentToken = firstTokens[0];
  yield currentToken;
  cache.appendTokens(0, [currentToken]);

  const tokenHistory = [...inputIds, currentToken];
  const useGraph = graphState !== undefined;
  let capturing = false;

  for (let i = 1; i < maxNewTokens && !eosIds.has(currentToken); i++) {
    const state = model.planDecode([currentToken], ws, cache, useGraph);

    if (useGraph && graphState!.graphExec !== null) {
      glm.graphLaunch(graphState!.graphExec);
      glm.synchronize();
    } else {
      if (useGraph && graphState!.warmupRemaining === 0 && !capturing) {
        capturing = true;
        glm.graphBeginCapture();
      }

      model.decodeForward(state, ws, cache);

      if (capturing) {
        const graph = glm.graphEndCapture();
        if (!graph) throw new Error("Graph capture failed");
        graphState!.graphExec = glm.graphInstantiate(graph);
        if (!graphState!.graphExec) throw new Error("Graph instantiation failed");
        glm.graphDestroy(graph);
        capturing = false;
      }
      if (useGraph) graphState!.warmupRemaining = Math.max(0, graphState!.warmupRemaining - 1);
    }

    currentToken = model.decodeRead(state)[0];

    if (sampling && needsSampling(sampling)) {
      currentToken = model.sampleTokenGPU(sampling, tokenHistory);
    }

    cache.appendTokens(0, [currentToken]);
    tokenHistory.push(currentToken);
    yield currentToken;
  }
}

export function generateBatchTokens(
  model: ChatModel, ws: WorkspaceBuffers, cache: ChatCache,
  inputIdsList: number[][], maxNewTokens: number, eosIds: Set<number>,
): number[][] {
  const batchSize = inputIdsList.length;
  cache.reset(batchSize);
  const firstTokens = model.forwardEager(inputIdsList, ws, cache);

  const nextTokens = [...firstTokens];
  const generated: number[][] = nextTokens.map(t => [t]);
  const finished = nextTokens.map(t => eosIds.has(t));

  for (let step = 0; step < maxNewTokens - 1; step++) {
    if (finished.every(f => f)) break;

    const newTokens = model.decodeEager(nextTokens, ws, cache);

    for (let i = 0; i < batchSize; i++) {
      nextTokens[i] = newTokens[i];
      if (!finished[i]) {
        if (eosIds.has(newTokens[i])) {
          finished[i] = true;
        } else {
          generated[i].push(newTokens[i]);
        }
      }
    }
  }

  return generated;
}

// --- Interactive / single-prompt modes ---

async function interactiveChat(
  model: ChatModel, glm: GlmOps, ws: WorkspaceBuffers, cache: ChatCache,
  tokenizer: any, args: CliArgs, graphState?: GraphState,
): Promise<void> {
  const sp = needsSampling(makeSamplingParams(args)) ? makeSamplingParams(args) : undefined;
  const eosIds = model.eosIds;
  const messages: Array<{ role: string; content: string }> = [];

  console.log("Type /quit to exit, /clear to reset conversation\n");

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
        if (graphState) graphState.graphExec = null;
        cache.reset(1);
        console.log("Conversation cleared.\n");
        continue;
      }

      messages.push({ role: "user", content: userInput });
      const inputIds = tokenizeMessages(tokenizer, messages, true);

      if (inputIds.length > args.maxSeqLen - args.maxNewTokens) {
        console.log(`Warning: prompt (${inputIds.length} tokens) too long, truncating conversation`);
        while (inputIds.length > args.maxSeqLen - args.maxNewTokens && messages.length > 1) {
          messages.splice(1, 2);
          const retryIds = tokenizeMessages(tokenizer, messages, true);
          if (retryIds.length <= args.maxSeqLen - args.maxNewTokens) break;
        }
        if (messages.length === 1 && tokenizeMessages(tokenizer, messages, true).length > args.maxSeqLen - args.maxNewTokens) {
          console.log("Conversation too long even after truncation. Use /clear to reset.");
          messages.pop();
          continue;
        }
      }

      process.stdout.write("Assistant: ");
      const t0 = performance.now();
      let tokCount = 0;
      const generatedIds: number[] = [];

      for (const tokenId of generateStream(model, glm, ws, cache, inputIds, args.maxNewTokens, eosIds, sp, graphState)) {
        generatedIds.push(tokenId);
        tokCount++;
        const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
        process.stdout.write(chunk);
        if (eosIds.has(tokenId)) break;
      }

      const elapsed = performance.now() - t0;
      console.log(`\n  [${tokCount} tokens in ${(elapsed / 1000).toFixed(1)}s, ${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s]`);

      const responseText = tokenizer.decode(generatedIds.filter(t => !eosIds.has(t)), { skip_special_tokens: true });
      messages.push({ role: "assistant", content: responseText });
    }
  } finally {
    if (graphState?.graphExec) glm.graphExecDestroy(graphState.graphExec);
    cache.free();
    ws.free();
    model.free();
    rl.close();
  }
}

async function singlePrompt(
  model: ChatModel, glm: GlmOps, ws: WorkspaceBuffers, cache: ChatCache,
  tokenizer: any, args: CliArgs, graphState?: GraphState,
): Promise<void> {
  const sp = needsSampling(makeSamplingParams(args)) ? makeSamplingParams(args) : undefined;
  const eosIds = model.eosIds;
  const messages = [{ role: "user", content: args.prompt! }];
  const inputIds = tokenizeMessages(tokenizer, messages, true);

  console.log(`Prompt: ${args.prompt}`);
  console.log(`Tokens: ${inputIds.length}`);

  process.stdout.write("\n");
  const t0 = performance.now();
  let tokCount = 0;
  const generatedIds: number[] = [];

  for (const tokenId of generateStream(model, glm, ws, cache, inputIds, args.maxNewTokens, eosIds, sp, graphState)) {
    generatedIds.push(tokenId);
    tokCount++;
    const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
    process.stdout.write(chunk);
    if (eosIds.has(tokenId)) break;
  }

  const elapsed = performance.now() - t0;
  console.log(`\n\n${tokCount} tokens in ${elapsed.toFixed(1)}ms (${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s)`);

  if (graphState?.graphExec) glm.graphExecDestroy(graphState.graphExec);
  cache.free();
  ws.free();
  model.free();
}

// --- Batch mode ---

async function interactiveBatch(
  model: ChatModel, cache: ChatCache, ws: WorkspaceBuffers,
  tokenizer: any, args: CliArgs,
): Promise<void> {
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
        const ids = tokenizeMessages(tokenizer, messages, true);
        inputIdsList.push(ids);
      }

      const maxPromptLen = Math.max(...inputIdsList.map(ids => ids.length));
      if (maxPromptLen > args.maxSeqLen) {
        console.log(`Longest prompt (${maxPromptLen} tokens) exceeds max_seq_len (${args.maxSeqLen}). Skipping.`);
        continue;
      }

      const start = Date.now();
      const generatedIds = generateBatchTokens(model, ws, cache, inputIdsList, args.maxNewTokens, model.eosIds);
      const elapsed = (Date.now() - start) / 1000;
      const totalTokens = generatedIds.reduce((sum: number, ids: number[]) => sum + ids.length, 0);

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

// --- Main ---

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  process.env.CUDA_VISIBLE_DEVICES = String(args.gpu);

  const glm = new GlmOps(0);

  const repoId = args.useQwen35 ? QWEN35_REPO : (args.useFp8 ? QWEN3_FP8_REPO : QWEN3_REPO);
  const maxBatch = args.useBatch ? args.maxBatch : 1;

  console.log(`Loading ${modelLabel(args)} on GPU ${args.gpu}...`);
  const model: ChatModel = args.useQwen35
    ? Qwen35Model.fromPretrained(glm, QWEN35_REPO, maxBatch, args.maxSeqLen)
    : Qwen3Model.fromPretrained(glm, repoId, maxBatch, args.maxSeqLen);
  const ws = new WorkspaceBuffers(glm);
  const cache = model.createChatCache(args.maxPages);

  const modelDir = resolveModelPath(repoId);
  const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });

  const sp = makeSamplingParams(args);
  const samplingParts: string[] = [];
  if (sp.temperature > 0) samplingParts.push(`temp=${sp.temperature}`);
  if (sp.topP < 1.0) samplingParts.push(`top_p=${sp.topP}`);
  if (sp.topK > 0) samplingParts.push(`top_k=${sp.topK}`);
  if (sp.repetitionPenalty !== 1.0) samplingParts.push(`rep_pen=${sp.repetitionPenalty}`);
  if (sp.presencePenalty !== 0) samplingParts.push(`pres_pen=${sp.presencePenalty}`);
  const samplingStr = samplingParts.length > 0 ? samplingParts.join(" ") : "greedy";

  console.log(`${modelLabel(args)}  |  GPU ${args.gpu}  |  max_seq_len=${args.maxSeqLen}  |  max_tokens=${args.maxNewTokens}  |  ${args.useBatch ? `batch=${maxBatch}` : (args.noCudaGraph ? "cuda_graph=off" : `cuda_graph=on(warmup=${args.warmupSteps})`)}  |  ${samplingStr}`);

  if (args.useBatch) {
    await interactiveBatch(model, cache, ws, tokenizer, args);
  } else {
    const graphState = args.noCudaGraph ? undefined : { graphExec: null as number | null, warmupRemaining: args.warmupSteps };

    if (args.prompt) {
      await singlePrompt(model, glm, ws, cache, tokenizer, args, graphState);
    } else {
      await interactiveChat(model, glm, ws, cache, tokenizer, args, graphState);
    }
  }
}

main().catch((err) => {
  console.error("Failed:", err);
  process.exit(1);
});
