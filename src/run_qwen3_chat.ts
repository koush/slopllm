import { GlmOps } from "./glm_ops";
import { Qwen3Model } from "./qwen3_model";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "./model_path";
import { createInterface } from "node:readline";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const EOS_TOKEN_IDS = new Set([151645, 151643]);

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
    const ids = (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
    return ids;
  } catch {
    const result = tokenizer.apply_chat_template(messages, {
      tokenize: true,
      add_generation_prompt: true,
      return_tensor: false,
      return_dict: true,
    }) as { input_ids: number[] | number[][] };
    const ids = (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
    return ids;
  }
}

function isPrefix(prefix: number[], full: number[]): boolean {
  if (prefix.length > full.length) return false;
  for (let i = 0; i < prefix.length; i++) {
    if (prefix[i] !== full[i]) return false;
  }
  return true;
}

function generateResponse(
  model: Qwen3Model, glm: GlmOps, ws: WorkspaceBuffers, pagedKV: PagedKVCache,
  inputIds: number[], graphExec: number | null, maxNewTokens: number, warmupSteps: number,
  useAppend: boolean, previousTokens: number[],
): { tokens: number[]; graphExec: number | null; timing: TimingInfo; cachedTokens: number[] } {
  const timing: TimingInfo = { prefillMs: 0, warmupMs: [], captureMs: 0, replayMs: [] };
  const generatedTokens: number[] = [];

  // Compute suffix for append mode.
  // previousTokens = all tokens currently in KV cache (input + previously decoded).
  // On next turn, the tokenizer produces the full conversation including the
  // assistant response, so we diff against the cached token IDs.
  let suffixIds: number[];
  if (useAppend && previousTokens.length > 0 && isPrefix(previousTokens, inputIds)) {
    suffixIds = inputIds.slice(previousTokens.length);
  } else {
    suffixIds = inputIds;
    pagedKV.reset(1);
  }

  // Prefill
  const t0 = performance.now();
  let tokens: number[];
  if (useAppend && previousTokens.length > 0 && suffixIds !== inputIds) {
    tokens = model.prefillBatchAppend([suffixIds], ws, pagedKV);
  } else {
    if (!useAppend || previousTokens.length === 0) {
      pagedKV.reset(1);
    }
    tokens = model.prefillBatch([suffixIds], ws, pagedKV);
  }
  pagedKV.updateIndptr();
  timing.prefillMs = performance.now() - t0;

  let currentToken = tokens[0];
  generatedTokens.push(currentToken);

  if (EOS_TOKEN_IDS.has(currentToken)) {
    // cachedTokens = input (or suffix) + decoded tokens
    const cachedTokens = suffixIds.concat(generatedTokens);
    return { tokens: generatedTokens, graphExec, timing, cachedTokens };
  }

  // Warmup + capture (first call only)
  if (graphExec === null) {
    for (let i = 0; i < warmupSteps; i++) {
      if (EOS_TOKEN_IDS.has(currentToken)) {
        const cachedTokens = suffixIds.concat(generatedTokens);
        return { tokens: generatedTokens, graphExec: null, timing, cachedTokens };
      }
      const t1 = performance.now();
      const state = model.decodeBatchPlan([currentToken], ws, pagedKV, true);
      model.decodeBatchForward(state, ws, pagedKV);
      currentToken = model.decodeBatchRead(state)[0];
      timing.warmupMs.push(performance.now() - t1);
      generatedTokens.push(currentToken);
    }

    if (EOS_TOKEN_IDS.has(currentToken)) {
      const cachedTokens = suffixIds.concat(generatedTokens);
      return { tokens: generatedTokens, graphExec: null, timing, cachedTokens };
    }

    // Capture graph
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

    // Replay captured step (forward wasn't executed during capture)
    glm.graphLaunch(graphExec);
    glm.synchronize();
    currentToken = model.decodeBatchRead(state)[0];
    generatedTokens.push(currentToken);
  }

  // Decode loop with graph replay
  const remaining = maxNewTokens - generatedTokens.length;
  for (let i = 0; i < remaining; i++) {
    if (EOS_TOKEN_IDS.has(currentToken)) break;
    const t3 = performance.now();
    const state = model.decodeBatchPlan([currentToken], ws, pagedKV, true);
    glm.graphLaunch(graphExec);
    glm.synchronize();
    currentToken = model.decodeBatchRead(state)[0];
    timing.replayMs.push(performance.now() - t3);
    generatedTokens.push(currentToken);
  }

  // cachedTokens = all tokens in KV cache after this turn (suffix + decoded)
  const cachedTokens = suffixIds.concat(generatedTokens);
  return { tokens: generatedTokens, graphExec, timing, cachedTokens };
}

async function interactiveChat(model: Qwen3Model, glm: GlmOps, ws: WorkspaceBuffers, pagedKV: PagedKVCache, tokenizer: any, args: any): Promise<void> {
  const messages: Array<{ role: string; content: string }> = [];
  let graphExec: number | null = null;
  let previousTokens: number[] = [];

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
        previousTokens = [];
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
          messages.splice(1, 2); // remove oldest user+assistant pair
          const retryIds = tokenizeMessages(tokenizer, messages, args.thinking);
          if (retryIds.length <= args.maxSeqLen - args.maxNewTokens) break;
        }
        if (messages.length === 1 && tokenizeMessages(tokenizer, messages, args.thinking).length > args.maxSeqLen - args.maxNewTokens) {
          console.log("Conversation too long even after truncation. Use /clear to reset.");
          messages.pop();
          continue;
        }
      }

      // Prefix match check for KV cache persistence
      if (args.noReset && previousTokens.length > 0) {
        const prefixOk = isPrefix(previousTokens, inputIds);
        if (!prefixOk) {
          console.log("  [prefix mismatch, falling back to full prefill]");
          previousTokens = [];
          graphExec = null;
          pagedKV.reset(1);
        } else {
          const suffixLen = inputIds.length - previousTokens.length;
          console.log(`  [prefix match, appending ${suffixLen} tokens (${previousTokens.length} cached)]`);
        }
      }

      const result = generateResponse(
        model, glm, ws, pagedKV, inputIds, graphExec,
        args.maxNewTokens, args.warmupSteps,
        args.noReset, previousTokens,
      );

      graphExec = result.graphExec;
      previousTokens = result.cachedTokens;

      // Strip EOS tokens
      const responseTokens = result.tokens.filter(t => !EOS_TOKEN_IDS.has(t));
      const responseText = tokenizer.decode(responseTokens, { skip_special_tokens: true });

      process.stdout.write(responseText + "\n\n");

      // Add assistant response to history
      messages.push({ role: "assistant", content: responseText });

      // Print timing on first turn
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

async function singlePrompt(model: Qwen3Model, glm: GlmOps, ws: WorkspaceBuffers, pagedKV: PagedKVCache, tokenizer: any, args: any): Promise<void> {
  const messages = [{ role: "user" as const, content: args.prompt }];
  const inputIds = tokenizeMessages(tokenizer, messages, args.thinking);

  console.log(`Prompt: ${args.prompt}`);
  console.log(`Tokens: ${inputIds.length}`);

  pagedKV.reset(1);
  const result = generateResponse(
    model, glm, ws, pagedKV, inputIds, null,
    args.maxNewTokens, args.warmupSteps,
    args.noReset, [],
  );

  const responseTokens = result.tokens.filter(t => !EOS_TOKEN_IDS.has(t));
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
    }
  }

  const gpuId = gpu ?? parseInt(process.env.GLM_GPU ?? "0", 10);
  process.env.CUDA_VISIBLE_DEVICES = String(gpuId);

  const glm = new GlmOps(0);
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
    await singlePrompt(model, glm, ws, pagedKV, tokenizer, cliArgs);
  } else {
    await interactiveChat(model, glm, ws, pagedKV, tokenizer, cliArgs);
  }
}

main().catch((err) => {
  console.error("Failed:", err);
  process.exit(1);
});
