import { GlmOps } from "./glm_ops";
import { ParallelOps } from "./parallel_ops";
import { Qwen3Model } from "./qwen3_model";
import { Qwen35Model } from "./qwen35_model";
import { Glm51Model } from "./glm51_model";
import { ChatModel, ChatCache, SamplingParams, makeSamplingParams } from "./chat_model";
import { MemcpyKind, Tensor, SamplingWorkspace } from "./tensor";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "./model_path";
import { createInterface } from "node:readline";
import { ExecutionState, ExecutionWorkspace } from "./paged_kv";
import { DeviceOps } from "./device_ops";
import { WorkspaceBase } from "./workspace";
import { UsingHolder } from "./using-holder";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const QWEN3_FP8_REPO = "Qwen/Qwen3-0.6B-FP8";
const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";
const GLM51_REPO = "zai-org/GLM-5.1";

export interface GraphState {
  graphExec: number | null;
  warmupRemaining: number;
}

interface CliArgs {
  gpus: number[];
  maxNewTokens: number;
  maxSeqLen: number;
  warmupSteps: number;
  maxPages: number;
  maxBatch: number;
  noReset: boolean;
  prompt: string | undefined;
  useQwen35: boolean;
  useGlm51: boolean;
  useFp8: boolean;
  useBatch: boolean;
  noCudaGraph: boolean;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  repetitionPenaltyWindow: number;
  greedy: boolean;
}

function parseArgs(argv: string[]): CliArgs {
  const gpusEnv = process.env.GLM_GPUS ?? process.env.GLM_GPU ?? "0";
  const args: CliArgs = {
    gpus: gpusEnv.split(",").map(s => parseInt(s.trim(), 10)),
    maxNewTokens: 256,
    maxSeqLen: 4096,
    warmupSteps: 3,
    maxPages: 256,
    maxBatch: 4,
    noReset: true,
    prompt: undefined,
    useQwen35: false,
    useGlm51: false,
    useFp8: false,
    useBatch: false,
    noCudaGraph: false,
    temperature: 0.6,
    topP: 0.95,
    topK: 0,
    repetitionPenalty: 1.0,
    presencePenalty: 0,
    repetitionPenaltyWindow: 64,
    greedy: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--prompt" && i + 1 < argv.length) args.prompt = argv[++i];
    else if (a === "--max-new-tokens" && i + 1 < argv.length) args.maxNewTokens = parseInt(argv[++i], 10);
    else if (a === "--warmup-steps" && i + 1 < argv.length) args.warmupSteps = parseInt(argv[++i], 10);
    else if (a === "--gpus" && i + 1 < argv.length) args.gpus = argv[++i].split(",").map(s => parseInt(s.trim(), 10));
    else if (a === "--gpu" && i + 1 < argv.length) args.gpus = [parseInt(argv[++i], 10)];
    else if (a === "--max-seq-len" && i + 1 < argv.length) args.maxSeqLen = parseInt(argv[++i], 10);
    else if (a === "--max-pages" && i + 1 < argv.length) args.maxPages = parseInt(argv[++i], 10);
    else if (a === "--max-batch" && i + 1 < argv.length) args.maxBatch = parseInt(argv[++i], 10);
    else if (a === "--no-kv-persist") args.noReset = false;
    else if (a === "--qwen35") args.useQwen35 = true;
    else if (a === "--glm51") args.useGlm51 = true;
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
      args.greedy = true;
    }
  }

  if (args.useQwen35 && args.useFp8) {
    console.error("Error: --fp8 is not supported with --qwen35");
    process.exit(1);
  }
  if (args.useGlm51 && args.useFp8) {
    console.error("Error: --fp8 is not supported with --glm51");
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
  if (args.useGlm51) return "GLM-5.1";
  return args.useFp8 ? "Qwen3-0.6B-FP8" : "Qwen3-0.6B";
}

function tokenizeMessages(
  tokenizer: any,
  messages: Array<{ role: string; content: string }>,
): number[] {
  try {
    const result = tokenizer.apply_chat_template(messages, {
      tokenize: true,
      add_generation_prompt: true,
      return_tensor: false,
      return_dict: true,
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

export interface DecodeTiming {
  planMs: number;
  execMs: number;
  idleMs: number;
  warmupSteps: number;
  graphSteps: number;
  warmupTokPerSec: number;
}

export function* generateStream(
  model: ChatModel, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache,
  inputIds: number[], maxNewTokens: number, eosIds: Set<number>,
  sampling: SamplingParams | undefined, graphState?: GraphState,
  timing?: DecodeTiming,
): Generator<number> {
  const suffixIds = cache.prefixMatch(0, inputIds);
  cache.appendTokens(0, suffixIds);

  using sampleWorkspace = new WorkspaceBase(glm);
  const greedy = !sampling;
  let sampleResult: Tensor | null = null;
  let gpuSampleResult: Tensor | null = null;
  const sampledLogits = new UsingHolder<Tensor>(undefined!);
  using samplingWorkspace = sampling ? new SamplingWorkspace(glm, [sampling!], model.cfg.vocabSize, sampling!.repetitionPenaltyWindow, [inputIds]) : undefined;
  function doSample(logits: Tensor) {
    if (greedy) {
      sampledLogits.replace(logits.argmax());
    }
    else {
      sampledLogits.replace(samplingWorkspace!.sample(logits));
    }
    const argmaxValue = sampledLogits.value;
    gpuSampleResult ||= sampleWorkspace.alloc(argmaxValue.shape, argmaxValue.type);
    gpuSampleResult.memcpy(argmaxValue, argmaxValue.bytes, MemcpyKind.DeviceToDevice);
  }

  const tokenHistory = inputIds.slice();
  const sampleStream = new UsingHolder<ReturnType<typeof glm.withStream<void>>>(undefined!);

  function readSample() {
    sampleStream.replace(glm.withStream(() => {
      const argmaxValue = sampledLogits.value;
      sampleResult ||= sampleWorkspace.allocPinned(argmaxValue.shape, argmaxValue.type);
      sampleResult.memcpy(argmaxValue, argmaxValue.bytes, MemcpyKind.DeviceToHost);
    }));
  }

  {
    using firstTokens = ws.forwardPrefill(model, [suffixIds], cache);
    doSample(firstTokens);
    readSample();
    // the decode is pipelined and then the token is yielded
  }
  // cache.appendTokens(0, [currentToken]);

  const useGraph = graphState !== undefined;
  let capturing = false;

  let planMs = 0;
  let execMs = 0;
  let idleMs = 0;
  let warmupSteps = 0;
  let graphSteps = 0;
  let firstPostWarmupTime = 0;
  let lastTokenTime = 0;
  let postWarmupTokenCount = 0;
  let tAfterSync = 0;

  try {
    for (let i = 1; i < maxNewTokens; i++) {
      const tPlan = performance.now();
      const state = ws.planDecode(model, 1, cache, useGraph);
      planMs += performance.now() - tPlan;

      const tExec = performance.now();
      if (graphState?.graphExec == null) {
        ws.inputIdsBuf.memcpy(gpuSampleResult!, gpuSampleResult!.bytes, MemcpyKind.DeviceToDevice);
        state.prepareInput(gpuSampleResult!);

        if (useGraph && graphState!.warmupRemaining === 0 && !capturing) {
          capturing = true;
          glm.graphBeginCapture();
        }

        ws.decodeStep(state);
        ws.forwardInput(state);
        using logits = model.forward(state);
        doSample(logits);

        if (capturing) {
          const graph = glm.graphEndCapture();
          ws.freeze();
          graphState!.graphExec = glm.graphInstantiate(graph);
          glm.graphDestroy(graph);
          capturing = false;
        }
        if (useGraph) {
          graphState!.warmupRemaining = Math.max(0, graphState!.warmupRemaining - 1);
        }
        warmupSteps++;
      }

      // when cuda graph captures it is NOT executing. it must run again.
      // thats why it is not else if, the actual execution must run again after capture.
      if (graphState?.graphExec != null) {
        if (tAfterSync > 0) idleMs += performance.now() - tAfterSync;
        // only host pinned is automatically copied. if using a device pinned, must be explicitly copied.
        ws.inputIdsBuf.memcpy(gpuSampleResult!, gpuSampleResult!.bytes, MemcpyKind.DeviceToDevice);
        glm.graphLaunch(graphState!.graphExec);
        graphSteps++;
      }

      // sync on the previous token
      sampleStream.value.synchronize();
      const currentToken = sampleResult!.readPinnedBuffer().readInt32LE();
      // kick off next sample read
      readSample();

      execMs += performance.now() - tExec;
      tAfterSync = performance.now();
      cache.appendTokens(0, [currentToken]);
      tokenHistory.push(currentToken);

      // yield previous token
      yield currentToken;
      if (eosIds.has(currentToken))
        return;

      const isPostWarmupToken = !useGraph || (graphState?.graphExec !== null);
      if (isPostWarmupToken) {
        const now = performance.now();
        if (firstPostWarmupTime === 0) firstPostWarmupTime = now;
        lastTokenTime = now;
        postWarmupTokenCount++;
      }
    }
  } finally {
    if (timing) {
      timing.planMs = planMs;
      timing.execMs = execMs;
      timing.idleMs = idleMs;
      timing.warmupSteps = warmupSteps;
      timing.graphSteps = graphSteps;
      timing.warmupTokPerSec = (postWarmupTokenCount > 1 && firstPostWarmupTime > 0)
        ? postWarmupTokenCount / ((lastTokenTime - firstPostWarmupTime) / 1000)
        : 0;
    }
  }
}

export function generateBatchTokens(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  inputIdsList: number[][], maxNewTokens: number, eosIds: Set<number>,
): number[][] {
  const batchSize = inputIdsList.length;
  cache.reset(batchSize);
  const firstTokens = ws.forwardEagerPrefill(model, inputIdsList, cache);

  const nextTokens = [...firstTokens];
  const generated: number[][] = nextTokens.map(t => [t]);
  const finished = nextTokens.map(t => eosIds.has(t));

  for (let step = 0; step < maxNewTokens - 1; step++) {
    if (finished.every(f => f)) break;

    const newTokens = ws.forwardEagerDecode(model, nextTokens, cache);

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
  model: ChatModel, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache,
  tokenizer: any, args: CliArgs, graphState?: GraphState,
): Promise<void> {
  const sp = makeSamplingParams(args);
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
      const inputIds = tokenizeMessages(tokenizer, messages);

      if (inputIds.length > args.maxSeqLen - args.maxNewTokens) {
        console.log(`Warning: prompt (${inputIds.length} tokens) too long, truncating conversation`);
        while (inputIds.length > args.maxSeqLen - args.maxNewTokens && messages.length > 1) {
          messages.splice(1, 2);
          const retryIds = tokenizeMessages(tokenizer, messages);
          if (retryIds.length <= args.maxSeqLen - args.maxNewTokens) break;
        }
        if (messages.length === 1 && tokenizeMessages(tokenizer, messages).length > args.maxSeqLen - args.maxNewTokens) {
          console.log("Conversation too long even after truncation. Use /clear to reset.");
          messages.pop();
          continue;
        }
      }

      process.stdout.write("Assistant: ");
      const t0 = performance.now();
      let tokCount = 0;
      const generatedIds: number[] = [];
      const timing: DecodeTiming = { planMs: 0, execMs: 0, idleMs: 0, warmupSteps: 0, graphSteps: 0, warmupTokPerSec: 0 };

      for (const tokenId of generateStream(model, ws, glm, cache, inputIds, args.maxNewTokens, eosIds, sp, graphState, timing)) {
        generatedIds.push(tokenId);
        tokCount++;
        const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
        process.stdout.write(chunk);
        if (eosIds.has(tokenId)) break;
      }

      const elapsed = performance.now() - t0;
      console.log(`\n  [${tokCount} tokens in ${(elapsed / 1000).toFixed(1)}s, ${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s]`);
      console.log(`  timing: plan=${timing.planMs.toFixed(1)}ms exec=${timing.execMs.toFixed(1)}ms idle=${timing.idleMs.toFixed(1)}ms (warmup=${timing.warmupSteps} graph=${timing.graphSteps}) decode=${timing.warmupTokPerSec.toFixed(1)} tok/s`);

      const responseText = tokenizer.decode(generatedIds.filter(t => !eosIds.has(t)), { skip_special_tokens: true });
      messages.push({ role: "assistant", content: responseText });
    }
  } finally {
    if (graphState?.graphExec !== null && graphState?.graphExec !== undefined) glm.graphExecDestroy(graphState.graphExec);
    rl.close();
  }
}

async function singlePrompt(
  model: ChatModel, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache,
  tokenizer: any, args: CliArgs, graphState?: GraphState,
): Promise<void> {
  const sp = !args.greedy ? makeSamplingParams(args) : undefined;
  const eosIds = model.eosIds;
  const messages = [{ role: "user", content: args.prompt! }];
  const inputIds = tokenizeMessages(tokenizer, messages);

  console.log(`Prompt: ${args.prompt}`);
  console.log(`Tokens: ${inputIds.length}`);

  process.stdout.write("\n");
  const t0 = performance.now();
  let tokCount = 0;
  const generatedIds: number[] = [];
  const timing: DecodeTiming = { planMs: 0, execMs: 0, idleMs: 0, warmupSteps: 0, graphSteps: 0, warmupTokPerSec: 0 };

  for (const tokenId of generateStream(model, ws, glm, cache, inputIds, args.maxNewTokens, eosIds, sp, graphState, timing)) {
    generatedIds.push(tokenId);
    tokCount++;
    const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
    process.stdout.write(chunk);
    if (eosIds.has(tokenId)) break;
  }

  const elapsed = performance.now() - t0;
  console.log(`\n\n${tokCount} tokens in ${elapsed.toFixed(1)}ms (${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s)`);
  console.log(`timing: plan=${timing.planMs.toFixed(1)}ms exec=${timing.execMs.toFixed(1)}ms idle=${timing.idleMs.toFixed(1)}ms (warmup=${timing.warmupSteps} graph=${timing.graphSteps}) decode=${timing.warmupTokPerSec.toFixed(1)} tok/s`);

  if (graphState?.graphExec !== null && graphState?.graphExec !== undefined) glm.graphExecDestroy(graphState.graphExec);
}

// --- Batch mode ---

async function interactiveBatch(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
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
        const ids = tokenizeMessages(tokenizer, messages);
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
    rl.close();
  }
}

// --- Main ---

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  const gpuDevices = args.gpus.map(id => new GlmOps(id));
  const glm: DeviceOps = gpuDevices.length > 1
    ? new ParallelOps(gpuDevices)
    : gpuDevices[0];
  const gpuLabel = args.gpus.join(",");

  const repoId = args.useGlm51 ? GLM51_REPO
    : args.useQwen35 ? QWEN35_REPO
    : (args.useFp8 ? QWEN3_FP8_REPO : QWEN3_REPO);

  const model: ChatModel = args.useGlm51
    ? Glm51Model.fromPretrained(glm, GLM51_REPO, args.maxBatch, args.maxSeqLen)
    : args.useQwen35
    ? Qwen35Model.fromPretrained(glm, QWEN35_REPO, args.maxBatch, args.maxSeqLen)
    : Qwen3Model.fromPretrained(glm, repoId, args.maxBatch, args.maxSeqLen);
  const cache = model.createChatCache(args.maxPages);
  const ws = new ExecutionWorkspace(glm, args.maxBatch, args.maxSeqLen);

  const modelDir = resolveModelPath(repoId);
  const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });

  const sp = makeSamplingParams(args);
  const samplingParts: string[] = [];
  if (sp.temperature > 0) samplingParts.push(`temp=${sp.temperature}`);
  if (sp.topP < 1.0) samplingParts.push(`top_p=${sp.topP}`);
  if (sp.topK > 0) samplingParts.push(`top_k=${sp.topK}`);
  if (sp.repetitionPenalty !== 1.0) samplingParts.push(`rep_pen=${sp.repetitionPenalty}`);
  if (sp.presencePenalty !== 0) samplingParts.push(`pres_pen=${sp.presencePenalty}`);
  const samplingStr = !args.greedy ? samplingParts.join(" ") : "greedy";

  console.log(`${modelLabel(args)}  |  GPU${args.gpus.length > 1 ? "s" : ""} ${gpuLabel}  |  max_seq_len=${args.maxSeqLen}  |  max_tokens=${args.maxNewTokens}  |  ${args.useBatch ? `batch=${args.maxBatch}` : (args.noCudaGraph ? "cuda_graph=off" : `cuda_graph=on(warmup=${args.warmupSteps})`)}  |  ${samplingStr}`);

  const cleanup = () => {
    glm.synchronize();
    cache.free();
    ws.free();
    model.free();
    if (glm instanceof ParallelOps) glm.free();
    for (const d of gpuDevices) d.free();
  };

  if (args.useBatch) {
    await interactiveBatch(model, ws, cache, tokenizer, args);
  } else {
    const graphState = args.noCudaGraph ? undefined : { graphExec: null as number | null, warmupRemaining: args.warmupSteps };

    if (args.prompt) {
      await singlePrompt(model, ws, glm, cache, tokenizer, args, graphState);
    } else {
      await interactiveChat(model, ws, glm, cache, tokenizer, args, graphState);
    }
  }

  cleanup();
}

main().catch((err) => {
  console.error("Failed:", err);
  process.exit(1);
});
