import { GlmOps } from "./glm_ops";
import { Qwen3Model } from "./qwen3_model";
import { Qwen35Model, SamplingParams } from "./qwen35_model";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv";
import { FlatKVCache } from "./flat_kv";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "./model_path";
import { createInterface } from "node:readline";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";
const QWEN3_EOS = new Set([151645, 151643]);
const QWEN35_EOS = new Set([248044]);

interface TimingInfo {
  prefillMs: number;
  warmupMs: number[];
  captureMs: number;
  replayMs: number[];
}

function tokenizeMessages(tokenizer: any, messages: Array<{ role: string; content: string }>, enableThinking: boolean): number[] {
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

function longestPrefix(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) return i;
  }
  return len;
}

function generateResponse(
  model: Qwen3Model, glm: GlmOps, ws: WorkspaceBuffers, pagedKV: PagedKVCache,
  inputIds: number[], cachedTokenIds: number[], graphExec: number | null, maxNewTokens: number, warmupSteps: number,
): { tokens: number[]; graphExec: number | null; timing: TimingInfo; cachedTokenIds: number[]; matchLen: number } {
  const timing: TimingInfo = { prefillMs: 0, warmupMs: [], captureMs: 0, replayMs: [] };
  const generatedTokens: number[] = [];

  let matchLen: number;
  if (cachedTokenIds.length > 0) {
    matchLen = longestPrefix(cachedTokenIds, inputIds);
  } else {
    matchLen = 0;
  }

  let suffixIds: number[];
  if (matchLen > 0 && matchLen < inputIds.length) {
    if (matchLen < cachedTokenIds.length) {
      pagedKV.truncate(0, matchLen);
    }
    suffixIds = inputIds.slice(matchLen);
  } else {
    matchLen = 0;
    suffixIds = inputIds;
    pagedKV.reset(1);
  }

  const t0 = performance.now();
  let tokens: number[];
  if (matchLen > 0) {
    tokens = model.prefillBatchAppend([suffixIds], ws, pagedKV);
  } else {
    tokens = model.prefillBatch([suffixIds], ws, pagedKV);
  }
  pagedKV.updateIndptr();
  timing.prefillMs = performance.now() - t0;

  let currentToken = tokens[0];
  generatedTokens.push(currentToken);

  if (QWEN3_EOS.has(currentToken)) {
    const newCachedTokenIds = inputIds.slice(0, matchLen).concat(suffixIds).concat(generatedTokens);
    return { tokens: generatedTokens, graphExec, timing, cachedTokenIds: newCachedTokenIds, matchLen };
  }

  if (graphExec === null) {
    for (let i = 0; i < warmupSteps; i++) {
      if (QWEN3_EOS.has(currentToken)) {
        const newCachedTokenIds = inputIds.slice(0, matchLen).concat(suffixIds).concat(generatedTokens);
        return { tokens: generatedTokens, graphExec: null, timing, cachedTokenIds: newCachedTokenIds, matchLen };
      }
      const t1 = performance.now();
      const state = model.decodeBatchPlan([currentToken], ws, pagedKV, true);
      model.decodeBatchForward(state, ws, pagedKV);
      currentToken = model.decodeBatchRead(state)[0];
      timing.warmupMs.push(performance.now() - t1);
      generatedTokens.push(currentToken);
    }

    if (QWEN3_EOS.has(currentToken)) {
      const newCachedTokenIds = inputIds.slice(0, matchLen).concat(suffixIds).concat(generatedTokens);
      return { tokens: generatedTokens, graphExec: null, timing, cachedTokenIds: newCachedTokenIds, matchLen };
    }

    const t2 = performance.now();
    const state = model.decodeBatchPlan([currentToken], ws, pagedKV, true);
    glm.graphBeginCapture();
    model.decodeBatchForward(state, ws, pagedKV);
    const graph = glm.graphEndCapture();
    if (!graph) throw new Error("Graph capture failed");
    graphExec = glm.graphInstantiate(graph);
    if (!graphExec) throw new Error("Graph instantiation failed");
    glm.graphDestroy(graph);
    timing.captureMs = performance.now() - t2;

    glm.graphLaunch(graphExec);
    glm.synchronize();
    currentToken = model.decodeBatchRead(state)[0];
    generatedTokens.push(currentToken);
  }

  const remaining = maxNewTokens - generatedTokens.length;
  for (let i = 0; i < remaining; i++) {
    if (QWEN3_EOS.has(currentToken)) break;
    const t3 = performance.now();
    const state = model.decodeBatchPlan([currentToken], ws, pagedKV, true);
    glm.graphLaunch(graphExec);
    glm.synchronize();
    currentToken = model.decodeBatchRead(state)[0];
    timing.replayMs.push(performance.now() - t3);
    generatedTokens.push(currentToken);
  }

  const newCachedTokenIds = inputIds.slice(0, matchLen).concat(suffixIds).concat(generatedTokens);
  return { tokens: generatedTokens, graphExec, timing, cachedTokenIds: newCachedTokenIds, matchLen };
}

async function interactiveChatQwen3(model: Qwen3Model, glm: GlmOps, ws: WorkspaceBuffers, pagedKV: PagedKVCache, tokenizer: any, args: any): Promise<void> {
  const messages: Array<{ role: string; content: string }> = [];
  let graphExec: number | null = null;
  let cachedTokenIds: number[] = [];

  console.log(`Qwen3-0.6B  |  GPU ${args.gpu}  |  max_seq_len=${args.maxSeqLen}  |  kv_persist=${args.noReset ? "on" : "off"}`);
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
        cachedTokenIds = [];
        graphExec = null;
        pagedKV.reset(1);
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

      const result = generateResponse(
        model, glm, ws, pagedKV, inputIds, cachedTokenIds, graphExec,
        args.maxNewTokens, args.warmupSteps,
      );

      graphExec = result.graphExec;
      cachedTokenIds = result.cachedTokenIds;

      if (cachedTokenIds.length > 0) {
        if (result.matchLen > 0) {
          const suffixLen = inputIds.length - result.matchLen;
          console.log(`  [cache hit ${result.matchLen}/${cachedTokenIds.length} tokens, appending ${suffixLen} new]`);
        } else {
          console.log("  [cache miss, full prefill]");
        }
      }

      const responseTokens = result.tokens.filter(t => !QWEN3_EOS.has(t));
      const responseText = tokenizer.decode(responseTokens, { skip_special_tokens: true });

      process.stdout.write(responseText + "\n\n");

      messages.push({ role: "assistant", content: responseText });

      if (result.timing.captureMs > 0) {
        const avg = result.timing.replayMs.length > 0
          ? result.timing.replayMs.reduce((a, b) => a + b, 0) / result.timing.replayMs.length
          : 0;
        console.log(`  [prefill ${result.timing.prefillMs.toFixed(0)}ms + capture ${result.timing.captureMs.toFixed(0)}ms + ${result.timing.warmupMs.length} warmup]${avg > 0 ? ` replay avg=${avg.toFixed(2)}ms (${(1000 / avg).toFixed(0)} tok/s)` : ""}`);
      }
    }
  } finally {
    if (graphExec !== null) {
      glm.graphExecDestroy(graphExec);
    }
    pagedKV.free();
    ws.free();
    model.free();
    rl.close();
  }
}

async function singlePromptQwen3(model: Qwen3Model, glm: GlmOps, ws: WorkspaceBuffers, pagedKV: PagedKVCache, tokenizer: any, args: any): Promise<void> {
  const messages = [{ role: "user", content: args.prompt }];
  const inputIds = tokenizeMessages(tokenizer, messages, args.thinking);

  console.log(`Prompt: ${args.prompt}`);
  console.log(`Tokens: ${inputIds.length}`);

  pagedKV.reset(1);
  const result = generateResponse(
    model, glm, ws, pagedKV, inputIds, [], null,
    args.maxNewTokens, args.warmupSteps,
  );

  const responseTokens = result.tokens.filter(t => !QWEN3_EOS.has(t));
  const responseText = tokenizer.decode(responseTokens, { skip_special_tokens: true });

  console.log(`\nPrefill: ${result.timing.prefillMs.toFixed(1)}ms`);
  if (result.timing.captureMs > 0) {
    console.log(`Graph capture: ${result.timing.captureMs.toFixed(1)}ms`);
  }
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

  if (result.graphExec !== null) {
    glm.graphExecDestroy(result.graphExec);
  }
  pagedKV.free();
  ws.free();
  model.free();
}

async function interactiveChatQwen35(model: Qwen35Model, cache: FlatKVCache, tokenizer: any, args: any): Promise<void> {
  const messages: Array<{ role: string; content: string }> = [];
  const sampling: SamplingParams = {
    temperature: args.temperature,
    topP: args.topP,
    repetitionPenalty: args.repetitionPenalty,
    repetitionPenaltyWindow: args.repetitionPenaltyWindow,
  };

  console.log(`Qwen3.5-0.8B  |  GPU ${args.gpu}  |  max_seq_len=${args.maxSeqLen}`);
  console.log(`Max ${args.maxNewTokens} tokens/turn  |  temp=${sampling.temperature} top_p=${sampling.topP} rep_pen=${sampling.repetitionPenalty}`);
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
        cache.reset();
        model.gdnState.reset();
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

      const t0 = performance.now();
      const generatedIds: number[] = [];
      for (const tokenId of model.streamTokens([inputIds], cache, args.maxNewTokens, QWEN35_EOS, sampling)) {
        generatedIds.push(tokenId);
        if (QWEN35_EOS.has(tokenId)) break;
      }
      const elapsed = performance.now() - t0;

      const responseTokens = generatedIds.filter(t => !QWEN35_EOS.has(t));
      const responseText = tokenizer.decode(responseTokens, { skip_special_tokens: true });

      process.stdout.write(responseText + "\n\n");

      const tokCount = generatedIds.length;
      console.log(`  [${tokCount} tokens, ${elapsed.toFixed(0)}ms total, ${(tokCount / (elapsed / 1000)).toFixed(0)} tok/s]`);

      messages.push({ role: "assistant", content: responseText });
    }
  } finally {
    cache.free();
    model.free();
    rl.close();
  }
}

async function singlePromptQwen35(model: Qwen35Model, cache: FlatKVCache, tokenizer: any, args: any): Promise<void> {
  const messages = [{ role: "user", content: args.prompt }];
  const inputIds = tokenizeMessages(tokenizer, messages, args.thinking);
  const sampling: SamplingParams = {
    temperature: args.temperature,
    topP: args.topP,
    repetitionPenalty: args.repetitionPenalty,
    repetitionPenaltyWindow: args.repetitionPenaltyWindow,
  };

  console.log(`Prompt: ${args.prompt}`);
  console.log(`Tokens: ${inputIds.length}  |  temp=${sampling.temperature} top_p=${sampling.topP} rep_pen=${sampling.repetitionPenalty}`);

  const t0 = performance.now();
  const generatedIds: number[] = [];
  for (const tokenId of model.streamTokens([inputIds], cache, args.maxNewTokens, QWEN35_EOS, sampling)) {
    generatedIds.push(tokenId);
    if (QWEN35_EOS.has(tokenId)) break;
  }
  const elapsed = performance.now() - t0;

  const responseTokens = generatedIds.filter(t => !QWEN35_EOS.has(t));
  const responseText = tokenizer.decode(responseTokens, { skip_special_tokens: true });

  const tokCount = generatedIds.length;
  console.log(`\n${tokCount} tokens in ${elapsed.toFixed(1)}ms (${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s)`);
  console.log(`\nResponse: ${responseText}`);

  cache.free();
  model.free();
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let maxNewTokens = 256;
  let warmupSteps = 3;
  let gpu: number | undefined;
  let maxSeqLen = 4096;
  let maxPages = 256;
  let noReset = true;
  let thinking = true;
  let prompt: string | undefined;
  let useQwen35 = false;
  let temperature = 0.6;
  let topP = 0.95;
  let repetitionPenalty = 1.1;
  let repetitionPenaltyWindow = 64;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--prompt" && i + 1 < args.length) {
      prompt = args[++i];
    } else if (args[i] === "--max-new-tokens" && i + 1 < args.length) {
      maxNewTokens = parseInt(args[++i], 10);
    } else if (args[i] === "--warmup-steps" && i + 1 < args.length) {
      warmupSteps = parseInt(args[++i], 10);
    } else if (args[i] === "--gpu" && i + 1 < args.length) {
      gpu = parseInt(args[++i], 10);
    } else if (args[i] === "--max-seq-len" && i + 1 < args.length) {
      maxSeqLen = parseInt(args[++i], 10);
    } else if (args[i] === "--max-pages" && i + 1 < args.length) {
      maxPages = parseInt(args[++i], 10);
    } else if (args[i] === "--no-kv-persist") {
      noReset = false;
    } else if (args[i] === "--no-thinking") {
      thinking = false;
    } else if (args[i] === "--qwen35") {
      useQwen35 = true;
    } else if (args[i] === "--temperature" && i + 1 < args.length) {
      temperature = parseFloat(args[++i]);
    } else if (args[i] === "--top-p" && i + 1 < args.length) {
      topP = parseFloat(args[++i]);
    } else if (args[i] === "--repetition-penalty" && i + 1 < args.length) {
      repetitionPenalty = parseFloat(args[++i]);
    } else if (args[i] === "--repetition-penalty-window" && i + 1 < args.length) {
      repetitionPenaltyWindow = parseInt(args[++i], 10);
    } else if (args[i] === "--greedy") {
      temperature = 0;
    }
  }

  const gpuId = gpu ?? parseInt(process.env.GLM_GPU ?? "0", 10);
  process.env.CUDA_VISIBLE_DEVICES = String(gpuId);

  const glm = new GlmOps(0);

  if (useQwen35) {
    console.log(`Loading Qwen3.5-0.8B on GPU ${gpuId}...`);
    const model = Qwen35Model.fromPretrained(glm, QWEN35_REPO, 1, maxSeqLen);
    const cache = model.createFlatKVCache();

    const modelDir = resolveModelPath(QWEN35_REPO);
    const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });

    const cliArgs = { gpu: gpuId, maxSeqLen, maxNewTokens, thinking, prompt, temperature, topP, repetitionPenalty, repetitionPenaltyWindow };

    if (prompt) {
      await singlePromptQwen35(model, cache, tokenizer, cliArgs);
    } else {
      await interactiveChatQwen35(model, cache, tokenizer, cliArgs);
    }
  } else {
    const model = Qwen3Model.fromPretrained(glm, QWEN3_REPO, 1, maxSeqLen);
    const cfg = (model as any).cfg;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const nLayers = cfg.numHiddenLayers;

    const modelDir = resolveModelPath(QWEN3_REPO);
    const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });
    const ws = new WorkspaceBuffers(glm);
    const pagedKV = new PagedKVCache(glm, nKv, hd, nLayers, maxPages, 1);

    const cliArgs = { gpu: gpuId, maxSeqLen, maxNewTokens, warmupSteps, noReset, thinking, prompt };

    if (prompt) {
      await singlePromptQwen3(model, glm, ws, pagedKV, tokenizer, cliArgs);
    } else {
      await interactiveChatQwen3(model, glm, ws, pagedKV, tokenizer, cliArgs);
    }
  }
}

main().catch((err) => {
  console.error("Failed:", err);
  process.exit(1);
});
