import { createInterface } from "node:readline";
import { CaptureManager } from "./capture-manager";
import { ChatCache, ChatModel, SamplingParams, Tokenizer, makeSamplingParams, type TokenSelector } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { ExecutionWorkspace } from "./execution-workspace";
import { MetaOps } from "./meta_ops";
import { createDeviceOps, loadModel, modelLabel, ModelCliArgs, parseModelArgs, resolveModelSelection } from "./model_cli";
import { MtpStats } from "./mtp_stats";
import { ParallelOps } from "./parallel_ops";
import { SamplingWorkspace } from "./sampling";

export interface GraphState {
  graphExec: number | null;
  warmupRemaining: number;
}

interface CliArgs extends ModelCliArgs {
  maxNewTokens: number;
  maxSeqLen: number;
  warmupSteps: number;
  maxPages: number;
  maxBatch: number;
  noReset: boolean;
  prompt: string | undefined;
  useBatch: boolean;
  noCudaGraph: boolean;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  repetitionPenaltyWindow: number;
  greedy: boolean;
  stats: boolean;
  meta: boolean;
  mtpDraftTopk: number[];
}

class ExecutionResources implements Disposable {
  model?: ChatModel;
  cache?: ChatCache;
  ws?: ExecutionWorkspace;
  private disposed = false;

  constructor(
    private readonly glm: DeviceOps,
    private readonly gpuDevices: readonly DeviceOps[],
  ) { }

  [Symbol.dispose](): void {
    if (this.disposed) return;
    this.disposed = true;

    let cleanupError: unknown;
    const dispose = (fn: () => void) => {
      try {
        fn();
      } catch (error) {
        cleanupError ??= error;
      }
    };

    dispose(() => this.glm.synchronize());
    if (this.cache) dispose(() => this.cache![Symbol.dispose]());
    if (this.ws) dispose(() => this.ws![Symbol.dispose]());
    if (this.model) dispose(() => this.model![Symbol.dispose]());
    if (this.glm instanceof ParallelOps) dispose(() => this.glm[Symbol.dispose]());
    for (const device of this.gpuDevices) dispose(() => device[Symbol.dispose]());

    if (cleanupError) throw cleanupError;
  }
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    ...parseModelArgs(argv),
    maxNewTokens: 256,
    maxSeqLen: 4096,
    warmupSteps: 3,
    maxPages: 256,
    maxBatch: 4,
    noReset: true,
    prompt: undefined,
    useBatch: false,
    noCudaGraph: false,
    temperature: 0.6,
    topP: 0.95,
    topK: 0,
    repetitionPenalty: 1.0,
    presencePenalty: 0,
    repetitionPenaltyWindow: 64,
    greedy: false,
    stats: false,
    meta: false,
    mtpDraftTopk: [],
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--prompt" && i + 1 < argv.length) args.prompt = argv[++i];
    else if (a === "--max-new-tokens" && i + 1 < argv.length) args.maxNewTokens = parseInt(argv[++i], 10);
    else if (a === "--warmup-steps" && i + 1 < argv.length) args.warmupSteps = parseInt(argv[++i], 10);
    else if (a === "--max-seq-len" && i + 1 < argv.length) args.maxSeqLen = parseInt(argv[++i], 10);
    else if (a === "--max-pages" && i + 1 < argv.length) args.maxPages = parseInt(argv[++i], 10);
    else if (a === "--max-batch" && i + 1 < argv.length) args.maxBatch = parseInt(argv[++i], 10);
    else if (a === "--no-kv-persist") args.noReset = false;
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
    else if (a === "--stats") args.stats = true;
    else if (a === "--meta") args.meta = true;
    else if (a === "--mtp-draft-topk" && i + 1 < argv.length) args.mtpDraftTopk = argv[++i].split(",").map(s => parseInt(s.trim(), 10));
  }

  if (args.mtp && args.mtpDraftTopk.length === 0) {
    args.mtpDraftTopk = [1, 1, 1];
  }
  if (args.mtpDraftTopk.some(topk => !Number.isInteger(topk) || topk < 1)) {
    throw new Error(`Invalid --mtp-draft-topk: ${args.mtpDraftTopk.join(",")}`);
  }

  if (args.useQwen35 && args.temperature > 0 && args.topP === 0.95 && args.topK === 0 && args.repetitionPenalty === 1.0 && args.presencePenalty === 0) {
    args.topK = 20;
    args.repetitionPenalty = 1.1;
  }

  return args;
}

function tokenizeMessages(
  tokenizer: Tokenizer,
  messages: Array<{ role: string; content: string }>,
): number[] {
  try {
    const opts: any = {
      tokenize: true,
      add_generation_prompt: true,
      return_tensor: false,
      return_dict: true,
    };
    const result = tokenizer.apply_chat_template(messages, opts) as unknown as { input_ids: number[] | number[][] };
    return (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
  } catch {
    const text = messages.map(m => `<|${m.role}|>\n${m.content}`).join("\n") + "\n\n\n";
    return tokenizer.encode(text, { add_special_tokens: false });
  }
}

// --- Generation primitives ---

async function prefillChunks(model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  inputIdsList: number[][], captureManager?: CaptureManager, samplingPolicy?: TokenSelector): Promise<void> {
  let remaining = inputIdsList;
  while (remaining.some(ids => ids.length)) {
    const chunk = await model.executePrefill(ws, cache, remaining, captureManager, Math.min(8192, ws.maxSeqLen), samplingPolicy);
    remaining = chunk.remainingInputIdsList;
    await new Promise<void>(resolve => setImmediate(resolve));
  }
}

export interface DecodeTiming {
  planMs: number;
  execMs: number;
  idleMs: number;
  warmupSteps: number;
  graphSteps: number;
  warmupTokPerSec: number;
  mtpStats?: MtpStats;
}

async function* generateMtpStream(
  model: ChatModel, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache,
  inputIds: number[], maxNewTokens: number, eosIds: Set<number>,
  topks: readonly number[], graphState?: GraphState, timing?: DecodeTiming,
): AsyncGenerator<number> {
  if (!model.generateMtpDecode) {
    throw new Error("The selected model does not support MTP decoding");
  }

  using captureManager = new CaptureManager(glm);
  captureManager.disabled = graphState === undefined;
  const mtpStats = new MtpStats(topks.length);
  if (timing) {
    timing.mtpStats = mtpStats;
  }

  let execMs = 0;
  let warmupSteps = 0;
  let graphSteps = 0;
  let firstPostWarmupTime = 0;
  let lastTokenTime = 0;
  let postWarmupTokenCount = 0;

  try {
    const suffixIds = cache.prefixMatch(0, inputIds);
    await prefillChunks(model, ws, cache, [suffixIds], captureManager);
    const target = cache.getPagedKV().sequences[0].targetToken!;
    yield target;
    await new Promise<void>(resolve => setImmediate(resolve));
    let generated = 1;
    if (eosIds.has(target) || generated >= maxNewTokens) return;

    let started = performance.now();
    for await (const result of model.generateMtpDecode(ws, cache, topks, captureManager)) {
      const step = { result, warmup: result.warmup };
      execMs += performance.now() - started;

      if (!step.warmup) {
        mtpStats.observe(step.result.numDraftTokens, step.result.numAccepted[0]);
      }

      for (const token of step.result.tokens[0]) {
        const now = performance.now();
        if (step.warmup) {
          warmupSteps++;
          firstPostWarmupTime = 0;
          postWarmupTokenCount = 0;
        } else {
          graphSteps++;
          if (firstPostWarmupTime === 0) {
            firstPostWarmupTime = now;
          }
          lastTokenTime = now;
          postWarmupTokenCount++;
        }

        generated++;
        yield token;
        await new Promise<void>(resolve => setImmediate(resolve));
        if (eosIds.has(token) || generated >= maxNewTokens) {
          return;
        }
      }
      started = performance.now();
    }
  } finally {
    if (timing) {
      timing.planMs = 0;
      timing.execMs = execMs;
      timing.idleMs = 0;
      timing.warmupSteps = warmupSteps;
      timing.graphSteps = graphSteps;
      timing.warmupTokPerSec = postWarmupTokenCount > 1 && firstPostWarmupTime > 0
        ? postWarmupTokenCount / ((lastTokenTime - firstPostWarmupTime) / 1000)
        : 0;
    }
  }
}

export async function* generateStream(
  model: ChatModel, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache,
  inputIds: number[], maxNewTokens: number, eosIds: Set<number>,
  sampling: SamplingParams | undefined, graphState?: GraphState,
  timing?: DecodeTiming, mtp?: boolean, mtpDraftTopk?: number[],
): AsyncGenerator<number> {
  if (maxNewTokens <= 0) {
    return;
  }
  const topks = mtp && mtpDraftTopk && mtpDraftTopk.length > 0 &&
    model.generateMtpDecode
    ? mtpDraftTopk
    : [];
  if (topks.length > 0) {
    if (sampling) {
      throw new Error("Plan-based MTP decoding currently supports greedy sampling only");
    }
    yield* generateMtpStream(model, ws, glm, cache, inputIds, maxNewTokens, eosIds, topks, graphState, timing);
    return;
  }

  using samplingWorkspace = sampling ? new SamplingWorkspace(glm, 1, model.cfg.vocabSize, sampling!.repetitionPenaltyWindow) : undefined;
  if (samplingWorkspace) samplingWorkspace.updateSampler([sampling!], [inputIds]);

  using captureManager = new CaptureManager(glm);
  let currentToken: number;

  captureManager.disabled = graphState === undefined;

  const suffixIds = cache.prefixMatch(0, inputIds);
  await prefillChunks(model, ws, cache, [suffixIds], captureManager, samplingWorkspace);

  // Budget is in TOKENS, not loop iterations. One plain decode step emits one
  // token, but one MTP step emits 1 + (accepted drafts), so bounding the loop
  // counter overshoots by the mean acceptance length (~2.2x at topk 1,1,1).
  let generated = 0;
  let execMs = 0;
  let warmupSteps = 0;
  let graphSteps = 0;
  let firstPostWarmupTime = 0;
  let lastTokenTime = 0;
  let postWarmupTokenCount = 0;

  try {
    currentToken = cache.getPagedKV().sequences[0].targetToken!;
    yield currentToken;
    await new Promise<void>(resolve => setImmediate(resolve));
    generated++;
    if (eosIds.has(currentToken) || generated >= maxNewTokens) return;
    if (samplingWorkspace) samplingWorkspace.updateSampler([sampling!], [[...inputIds, currentToken]]);
    let started = performance.now();
    for await (const step of model.generateDecode(ws, cache, captureManager, samplingWorkspace)) {
      execMs += performance.now() - started;
      currentToken = step.tokens[0][0];
      if (step.warmup) {
        warmupSteps++;
        firstPostWarmupTime = 0;
        postWarmupTokenCount = 0;
      } else {
        graphSteps++;
        const now = performance.now();
        if (firstPostWarmupTime === 0) firstPostWarmupTime = now;
        lastTokenTime = now;
        postWarmupTokenCount++;
      }

      yield currentToken;
      await new Promise<void>(resolve => setImmediate(resolve));
      generated++;
      if (eosIds.has(currentToken) || generated >= maxNewTokens)
        return;
      started = performance.now();
    }
  } finally {
    if (timing) {
      timing.planMs = 0;
      timing.execMs = execMs;
      timing.idleMs = 0;
      timing.warmupSteps = warmupSteps;
      timing.graphSteps = graphSteps;
      timing.warmupTokPerSec = (postWarmupTokenCount > 1 && firstPostWarmupTime > 0)
        ? postWarmupTokenCount / ((lastTokenTime - firstPostWarmupTime) / 1000)
        : 0;
    }
  }
}

export async function generateBatchTokens(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  inputIdsList: number[][], maxNewTokens: number, eosIds: Set<number>,
): Promise<number[][]> {
  const batchSize = inputIdsList.length;
  cache.reset(batchSize);
  const generated: number[][] = inputIdsList.map(() => []);
  if (!batchSize || maxNewTokens <= 0) {
    return generated;
  }
  await prefillChunks(model, ws, cache, inputIdsList);
  const finished = cache.getPagedKV().sequences.map((sequence, index) => {
    const token = sequence.targetToken!;
    generated[index].push(token);
    return eosIds.has(token) || generated[index].length >= maxNewTokens;
  });
  if (finished.every(Boolean)) return generated;
  for await (const step of model.generateDecode(ws, cache)) {
    for (let i = 0; i < batchSize; i++) {
      if (!finished[i]) {
        const token = step.tokens[i][0];
        if (!eosIds.has(token) || !generated[i].length) {
          generated[i].push(token);
        }
        finished[i] = eosIds.has(token) || generated[i].length >= maxNewTokens;
      }
    }
    if (finished.every(Boolean)) {
      break;
    }
  }

  return generated;
}

// --- Interactive / single-prompt modes ---

async function interactiveChat(
  model: ChatModel, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache,
  args: CliArgs, graphState: GraphState | undefined,
): Promise<void> {
  const tokenizer = model.tokenizer;
  const sp = !args.greedy ? makeSamplingParams(args) : undefined;
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

      for await (const tokenId of generateStream(model, ws, glm, cache, inputIds, args.maxNewTokens, eosIds, sp, graphState, timing, args.mtp, args.mtpDraftTopk)) {
        generatedIds.push(tokenId);
        tokCount++;
        const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
        process.stdout.write(chunk);
        if (eosIds.has(tokenId)) break;
      }

      const elapsed = performance.now() - t0;
      console.log(`\n  [${tokCount} tokens in ${(elapsed / 1000).toFixed(1)}s, ${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s]`);
      console.log(`  timing: plan=${timing.planMs.toFixed(1)}ms exec=${timing.execMs.toFixed(1)}ms idle=${timing.idleMs.toFixed(1)}ms (warmup=${timing.warmupSteps} graph=${timing.graphSteps}) decode=${timing.warmupTokPerSec.toFixed(1)} tok/s`);
      if (timing.mtpStats) console.log(`  ${timing.mtpStats.log()}`);

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
  args: CliArgs, graphState: GraphState | undefined,
): Promise<void> {
  const tokenizer = model.tokenizer;
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

  for await (const tokenId of generateStream(model, ws, glm, cache, inputIds, args.maxNewTokens, eosIds, sp, graphState, timing, args.mtp, args.mtpDraftTopk)) {
    generatedIds.push(tokenId);
    tokCount++;
    const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
    process.stdout.write(chunk);
    if (eosIds.has(tokenId)) break;
  }

  const elapsed = performance.now() - t0;
  console.log(`\n\n${tokCount} tokens in ${elapsed.toFixed(1)}ms (${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s)`);
  console.log(`timing: plan=${timing.planMs.toFixed(1)}ms exec=${timing.execMs.toFixed(1)}ms idle=${timing.idleMs.toFixed(1)}ms (warmup=${timing.warmupSteps} graph=${timing.graphSteps}) decode=${timing.warmupTokPerSec.toFixed(1)} tok/s`);
  if (timing.mtpStats) console.log(timing.mtpStats.log());

  if (graphState?.graphExec !== null && graphState?.graphExec !== undefined) glm.graphExecDestroy(graphState.graphExec);
}

// --- Batch mode ---

async function interactiveBatch(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  args: CliArgs,
): Promise<void> {
  const tokenizer = model.tokenizer;
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
      const generatedIds = await generateBatchTokens(model, ws, cache, inputIdsList, args.maxNewTokens, model.eosIds);
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

export async function main(argv = process.argv.slice(2)): Promise<void> {
  Error.stackTraceLimit = 20;

  const args = parseArgs(argv);

  const { modelDir } = resolveModelSelection(args);

  if (args.meta) {
    const metaOps = new MetaOps();
    const model = await loadModel(metaOps, args, modelDir);
    const loadAllocs = metaOps.totalAllocs;
    const loadBytes = metaOps.totalBytes;

    const cache = model.createChatCache(args.maxPages, args.maxBatch, args.maxSeqLen);
    const ws = new ExecutionWorkspace(metaOps, args.maxBatch, args.maxSeqLen);
    const inputIds = [1, 2, 3, 4, 5];
    cache.reset(1);
    await prefillChunks(model, ws, cache, [inputIds]);
    const forwardAllocs = metaOps.totalAllocs;
    const forwardBytes = metaOps.totalBytes;

    const mb = (b: number) => (b / (1024 * 1024)).toFixed(1);
    console.log(`Meta mode: ${modelLabel(args)}`);
    console.log(`  model weight allocs: ${loadAllocs} (${mb(loadBytes)} MB)`);
    console.log(`  + workspace/cache allocs: ${forwardAllocs - loadAllocs} (${mb(forwardBytes - loadBytes)} MB)`);
    console.log(`  total allocs: ${forwardAllocs} (${mb(forwardBytes)} MB)`);
    return;
  }

  const { glm, gpuDevices } = createDeviceOps(args);
  using resources = new ExecutionResources(glm, gpuDevices);

  const gpuLabel = args.gpus.join(",");

  const model = await loadModel(glm, args, modelDir);
  resources.model = model;

  const cache = model.createChatCache(args.maxPages, args.maxBatch, args.maxSeqLen);
  resources.cache = cache;
  const ws = new ExecutionWorkspace(glm, args.maxBatch, args.maxSeqLen);
  resources.ws = ws;

  const sp = makeSamplingParams(args);
  const samplingParts: string[] = [];
  if (sp.temperature > 0) samplingParts.push(`temp=${sp.temperature}`);
  if (sp.topP < 1.0) samplingParts.push(`top_p=${sp.topP}`);
  if (sp.topK > 0) samplingParts.push(`top_k=${sp.topK}`);
  if (sp.repetitionPenalty !== 1.0) samplingParts.push(`rep_pen=${sp.repetitionPenalty}`);
  if (sp.presencePenalty !== 0) samplingParts.push(`pres_pen=${sp.presencePenalty}`);
  const samplingStr = !args.greedy ? samplingParts.join(" ") : "greedy";

  const arenaStr = args.arena ? `  |  arena=${args.arena}GB` : "";
  const mtpStr = args.mtp ? `  |  mtp=${args.mtpDraftTopk.join(',')}` : "";
  console.log(`${modelLabel(args)}  |  GPU${args.gpus.length > 1 ? "s" : ""} ${gpuLabel}  |  max_seq_len=${args.maxSeqLen}  |  max_tokens=${args.maxNewTokens}  |  ${args.useBatch ? `batch=${args.maxBatch}` : (args.noCudaGraph ? "cuda_graph=off" : `cuda_graph=on(warmup=${args.warmupSteps})`)}  |  ${samplingStr}${arenaStr}${mtpStr}`);

  if (args.useBatch) {
    await interactiveBatch(model, ws, cache, args);
  } else {
    const graphState = args.noCudaGraph ? undefined : { graphExec: null as number | null, warmupRemaining: args.warmupSteps };

    if (args.prompt) {
      await singlePrompt(model, ws, glm, cache, args, graphState);
    } else {
      await interactiveChat(model, ws, glm, cache, args, graphState);
    }
  }

}

// Only run as a CLI. This module also exports generateStream, and without the
// guard any `import { generateStream } from "./run_qwen3_unified"` would load a
// whole second model as a side effect of the import.
if (require.main === module) {
  main().catch((err) => {
    console.error("Failed:", err);
    process.exit(1);
  });
}
