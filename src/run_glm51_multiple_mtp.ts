import { CaptureManager } from "./capture-manager";
import fs from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { type ChatCache, type ChatModel, type Tokenizer } from "./chat_model";
import { type DeviceOps } from "./device_ops";
import { ExecutionWorkspace } from "./execution-workspace";
import { Glm51Model } from "./glm51_model";
import { type GlmOps } from "./glm_ops";
import { createDeviceOps, loadModel, type ModelCliArgs, parseModelArgs, resolveModelSelection } from "./model_cli";
import { MtpStats } from "./mtp_stats";
import { ParallelOps } from "./parallel_ops";
import { profilerStart, profilerStop } from "./native-addon";

const PROMPTS = [
  "tell me about india",
  "tell me a 1000 word story",
];
const PREFILL_CHUNK_SIZE = 8192;

interface Args extends ModelCliArgs {
  batchSize: number;
  maxNewTokens: number;
  maxSeqLen: number;
  maxPages: number;
  noCudaGraph: boolean;
  noMtp: boolean;
  contextLen?: number;
  file?: string;
  ignoreEos: boolean;
  instruction?: string;
  prompt?: string;
  warmupRuns: number;
  cooldownSeconds: number;
  profile: boolean;
}

function parseLength(value: string): number {
  const match = /^(\d+)([kKmM]?)$/.exec(value);
  if (!match) return NaN;
  const scale = match[2].toLowerCase() === "k" ? 1024 : match[2].toLowerCase() === "m" ? 1024 * 1024 : 1;
  return parseInt(match[1], 10) * scale;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    ...parseModelArgs(argv),
    batchSize: 2,
    maxNewTokens: 2000,
    maxSeqLen: 4096,
    maxPages: 256,
    noCudaGraph: false,
    noMtp: false,
    contextLen: undefined,
    file: undefined,
    ignoreEos: false,
    instruction: undefined,
    prompt: undefined,
    warmupRuns: 0,
    cooldownSeconds: 0,
    profile: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--batch-size" && i + 1 < argv.length) args.batchSize = parseInt(argv[++i], 10);
    else if (arg === "--max-new-tokens" && i + 1 < argv.length) args.maxNewTokens = parseInt(argv[++i], 10);
    else if (arg === "--max-seq-len" && i + 1 < argv.length) args.maxSeqLen = parseInt(argv[++i], 10);
    else if (arg === "--max-pages" && i + 1 < argv.length) args.maxPages = parseInt(argv[++i], 10);
    else if (arg === "--prompt" && i + 1 < argv.length) args.prompt = argv[++i];
    else if (arg === "--file" && i + 1 < argv.length) args.file = argv[++i];
    else if (arg === "--context-len" && i + 1 < argv.length) args.contextLen = parseLength(argv[++i]);
    else if (arg === "--instruction" && i + 1 < argv.length) args.instruction = argv[++i];
    else if (arg === "--no-cuda-graph") args.noCudaGraph = true;
    else if (arg === "--no-mtp") args.noMtp = true;
    else if (arg === "--ignore-eos") args.ignoreEos = true;
    else if (arg === "--warmup-runs") args.warmupRuns = Number(argv[++i]);
    else if (arg === "--cooldown-seconds") args.cooldownSeconds = Number(argv[++i]);
    else if (arg === "--profile") args.profile = true;
  }

  if (args.prompt && args.file) {
    throw new Error("--prompt and --file cannot be used together");
  }
  if (!Number.isSafeInteger(args.warmupRuns) || args.warmupRuns < 0) {
    throw new Error(`Invalid --warmup-runs: ${args.warmupRuns}`);
  }
  if (!Number.isFinite(args.cooldownSeconds) || args.cooldownSeconds < 0 || args.cooldownSeconds * 1000 > 2147483647) {
    throw new Error(`Invalid --cooldown-seconds: ${args.cooldownSeconds}`);
  }
  if (args.contextLen !== undefined && (!Number.isInteger(args.contextLen) || args.contextLen < 1)) {
    throw new Error(`Invalid --context-len: ${args.contextLen}`);
  }
  if (args.contextLen !== undefined && !args.file) {
    throw new Error("--context-len requires --file");
  }
  if (args.instruction !== undefined && !args.file) {
    throw new Error("--instruction requires --file");
  }
  const maxBatchSize = args.prompt || args.file ? 8 : PROMPTS.length;
  if (!Number.isInteger(args.batchSize) || args.batchSize < 1 || args.batchSize > maxBatchSize) {
    throw new Error(`--batch-size must be between 1 and ${maxBatchSize}`);
  }
  if (!Number.isInteger(args.maxNewTokens) || args.maxNewTokens < 1) {
    throw new Error(`Invalid --max-new-tokens: ${args.maxNewTokens}`);
  }
  if (!Number.isInteger(args.maxSeqLen) || args.maxSeqLen < 1) {
    throw new Error(`Invalid --max-seq-len: ${args.maxSeqLen}`);
  }
  if (!Number.isInteger(args.maxPages) || args.maxPages < 1) {
    throw new Error(`Invalid --max-pages: ${args.maxPages}`);
  }
  if (!args.useGlm51 || (args.mtp === 0 && !args.noMtp)) {
    throw new Error("run_glm51_multiple_mtp requires --glm51 and either --mtp or --no-mtp");
  }
  if (args.noMtp) {
    args.mtp = 0;
  }
  if (args.contextLen !== undefined) {
    args.maxSeqLen = Math.max(args.maxSeqLen, args.contextLen + args.maxNewTokens);
    const requiredPages = args.file && args.batchSize > 1
      ? Math.ceil(args.contextLen / 64) + args.batchSize * (Math.ceil(args.maxNewTokens / 64) + 1)
      : args.batchSize * Math.ceil((args.contextLen + args.maxNewTokens) / 64);
    args.maxPages = Math.max(args.maxPages, requiredPages);
  }

  return args;
}

function tokenizePrompt(tokenizer: Tokenizer, prompt: string): number[] {
  try {
    const result = tokenizer.apply_chat_template(
      [{ role: "user", content: prompt }],
      { tokenize: true, add_generation_prompt: true, return_tensor: false, return_dict: true } as any,
    ) as unknown as { input_ids: number[] | number[][] };
    return (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
  } catch {
    return tokenizer.encode(`<|user|>\n${prompt}\n\n\n`, { add_special_tokens: false });
  }
}

function buildFilePromptIds(tokenizer: Tokenizer, file: string, contextLen?: number, instruction?: string): number[] {
  const content = fs.readFileSync(file, "utf8");
  const promptFor = (source: string) => instruction ? `${source}\n\n${instruction}` : source;
  if (contextLen === undefined) return tokenizePrompt(tokenizer, promptFor(content));

  const contentIds = tokenizer.encode(content, { add_special_tokens: false });
  const fixedPromptLen = tokenizePrompt(tokenizer, promptFor("")).length;
  if (contentIds.length + fixedPromptLen < contextLen) {
    throw new Error(`${file} has approximately ${contentIds.length + fixedPromptLen} prompt tokens, fewer than --context-len ${contextLen}`);
  }

  const tokenizePrefix = (count: number) => {
    const prefix = tokenizer.decode(contentIds.slice(0, count), { skip_special_tokens: false });
    return tokenizePrompt(tokenizer, promptFor(prefix));
  };
  let contentTokenCount = Math.min(contentIds.length, Math.max(0, contextLen - fixedPromptLen));
  const tried = new Set<number>();
  for (let attempt = 0; attempt < 16; attempt++) {
    if (tried.has(contentTokenCount)) break;
    tried.add(contentTokenCount);
    const ids = tokenizePrefix(contentTokenCount);
    if (ids.length === contextLen) return ids;
    contentTokenCount = Math.max(0, Math.min(contentIds.length, contentTokenCount + contextLen - ids.length));
  }

  for (let delta = -64; delta <= 64; delta++) {
    const candidate = contentTokenCount + delta;
    if (candidate < 0 || candidate > contentIds.length || tried.has(candidate)) continue;
    const ids = tokenizePrefix(candidate);
    if (ids.length === contextLen) return ids;
  }
  throw new Error(`Unable to construct an exact ${contextLen}-token chat prompt from ${file}`);
}

function prepareInputs(tokenizer: Tokenizer, args: Args): { prompts: string[]; inputIds: number[][] } {
  if (args.file) {
    const ids = buildFilePromptIds(tokenizer, args.file, args.contextLen, args.instruction);
    const label = `${args.file} (${ids.length} prompt tokens)`;
    console.log(`File prompt: ${label}`);
    return {
      prompts: Array(args.batchSize).fill(label),
      inputIds: Array.from({ length: args.batchSize }, () => ids.slice()),
    };
  }
  const prompts = args.prompt ? Array(args.batchSize).fill(args.prompt) : PROMPTS.slice(0, args.batchSize);
  return { prompts, inputIds: prompts.map(prompt => tokenizePrompt(tokenizer, prompt)) };
}

function freeResources(model: ChatModel | undefined, cache: ChatCache | undefined, ws: ExecutionWorkspace | undefined, ops: DeviceOps, gpuDevices: GlmOps[]): void {
  let cleanupError: unknown;
  const dispose = (fn: () => void) => {
    try {
      fn();
    } catch (error) {
      cleanupError ??= error;
    }
  };

  dispose(() => ops.synchronize());
  if (cache) dispose(() => cache[Symbol.dispose]());
  if (ws) dispose(() => ws[Symbol.dispose]());
  if (model) dispose(() => model[Symbol.dispose]());
  if (ops instanceof ParallelOps) dispose(() => ops[Symbol.dispose]());
  for (const device of gpuDevices) dispose(() => device[Symbol.dispose]());
  if (cleanupError) throw cleanupError;
}

async function runBatch(model: Glm51Model, ws: ExecutionWorkspace, cache: ChatCache, args: Args, captureManager: CaptureManager): Promise<void> {
  const { prompts, inputIds } = prepareInputs(model.tokenizer, args);
  const longestPrompt = Math.max(...inputIds.map(ids => ids.length));
  if (longestPrompt + args.maxNewTokens > args.maxSeqLen) {
    throw new Error(`Prompt plus generation budget exceeds --max-seq-len (${longestPrompt} + ${args.maxNewTokens} > ${args.maxSeqLen})`);
  }

  const sharePrefill = args.file !== undefined && args.batchSize > 1;
  cache.reset(sharePrefill ? 1 : args.batchSize);

  if (args.file) {
    const prefillStarted = performance.now();
    let remaining = [inputIds[0]];
    while (remaining[0].length > PREFILL_CHUNK_SIZE) {
      const result = await model.executePrefill(ws, cache, remaining, captureManager, PREFILL_CHUNK_SIZE);
      remaining = result.remainingInputIdsList;
    }
    const offset = inputIds[0].length - remaining[0].length;
    console.log(`Prefilled ${offset} tokens in ${((performance.now() - prefillStarted) / 1000).toFixed(1)}s before final prefill.`);
  }

  const suffixIds = sharePrefill || args.file
    ? [cache.prefixMatch(0, inputIds[0])]
    : inputIds.map((ids, batch) => cache.prefixMatch(batch, ids));
  let remaining = suffixIds;
  while (remaining.some(ids => ids.length)) {
    const result = await model.executePrefill(ws, cache, remaining, captureManager, PREFILL_CHUNK_SIZE);
    remaining = result.remainingInputIdsList;
  }
  const generated: number[][] = Array.from({ length: args.batchSize }, () => []);
  const finished = Array(args.batchSize).fill(args.maxNewTokens === 0);
  let mtpStats = args.noMtp ? undefined : new MtpStats(args.mtp);
  let measurementStart = 0;
  let measurementEnd = 0;
  let postWarmupStepCount = 0;
  let postWarmupTokenCount = 0;

  if (sharePrefill) {
    for (let batch = 1; batch < args.batchSize; batch++) {
      cache.getPagedKV().copySequence(batch, 0);
    }
  }

  const started = performance.now();
  if (args.maxNewTokens > 0) {
    cache.getPagedKV().sequences.forEach((sequence, batch) => {
      const token = sequence.targetToken!;
      generated[batch].push(token);
      finished[batch] = (!args.ignoreEos && model.eosIds.has(token)) || generated[batch].length >= args.maxNewTokens;
    });
  }
  if (!finished.some(Boolean)) {
    const generator = args.noMtp
      ? model.generateDecode(ws, cache, captureManager)
      : model.generateMtpDecode(ws, cache, args.mtp, captureManager);
    for await (const step of generator) {
      const completedAt = performance.now();
      const { warmup, tokens: stepTokens, numAccepted, numDraftTokens } = step;
      if (warmup || measurementStart === 0) {
        // Anchor at completion, excluding initialization and every warmup/capture.
        // With no warmup, the first yielded step establishes the baseline.
        measurementStart = completedAt;
        measurementEnd = completedAt;
        postWarmupStepCount = 0;
        postWarmupTokenCount = 0;
        mtpStats = args.noMtp ? undefined : new MtpStats(args.mtp);
      } else {
        measurementEnd = completedAt;
        postWarmupStepCount++;
        // Count the complete executed step, including any final budget overshoot.
        postWarmupTokenCount += stepTokens.reduce((count, tokens) => count + tokens.length, 0);
        if (mtpStats) {
          for (const count of numAccepted) mtpStats.observe(numDraftTokens, count);
        }
      }

      for (let batch = 0; batch < args.batchSize; batch++) {
        for (const token of stepTokens[batch]) {
          generated[batch].push(token);

          if ((!args.ignoreEos && model.eosIds.has(token)) || generated[batch].length >= args.maxNewTokens) {
            finished[batch] = true;
            break;
          }
        }
      }
      if (finished.some(Boolean)) break;
    }
  }

  const elapsed = (performance.now() - started) / 1000;
  const measuredMs = measurementEnd - measurementStart;
  const decodeTokPerSec = measuredMs > 0
    ? postWarmupTokenCount * 1000 / measuredMs
    : 0;
  console.log(`Stopped when batch ${finished.findIndex(Boolean) + 1} completed after ${elapsed.toFixed(1)}s.`);
  for (let batch = 0; batch < args.batchSize; batch++) {
    const visibleTokens = generated[batch].filter(token => !model.eosIds.has(token));
    console.log(`\n--- Prompt ${batch + 1} ---\n${prompts[batch]}`);
    console.log(`\n--- Response ${batch + 1} (${visibleTokens.length} tokens) ---`);
    console.log(model.tokenizer.decode(visibleTokens, { skip_special_tokens: true }));
  }
  console.log(`\ndecode=${decodeTokPerSec.toFixed(1)} tok/s`);
  if (postWarmupStepCount > 0) {
    console.log(`Decode timing: ${(measuredMs / postWarmupStepCount).toFixed(4)} ms/step, ` +
      `${postWarmupStepCount} steps, ${measuredMs.toFixed(3)} ms, ${postWarmupTokenCount} tokens, ` +
      `${(postWarmupTokenCount / postWarmupStepCount).toFixed(4)} tokens/step ` +
      `(wall clock after last warmup; complete steps across batch)`);
  } else {
    console.log("Decode timing: no measured steps after the last warmup/baseline step");
  }
  if (mtpStats) {
    console.log(mtpStats.log() || "MTP metrics: no post-warmup drafts");
  }
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
  Error.stackTraceLimit = 20;
  const args = parseArgs(argv);
  const { modelDir } = resolveModelSelection(args);
  const { ops, gpuDevices } = createDeviceOps(args);
  let model: Glm51Model | undefined;
  let cache: ChatCache | undefined;
  let ws: ExecutionWorkspace | undefined;

  try {
    const loadedModel = await loadModel(ops, args, modelDir);
    if (!(loadedModel instanceof Glm51Model)) throw new Error("run_glm51_multiple_mtp requires a GLM-5.1 model");
    model = loadedModel;
    const workspaceSeqLen = args.file ? Math.min(args.maxSeqLen, PREFILL_CHUNK_SIZE + 1) : args.maxSeqLen;
    cache = model.createChatCache(args.maxPages, args.batchSize, workspaceSeqLen);
    ws = new ExecutionWorkspace(ops, args.batchSize, workspaceSeqLen);
    console.log(`GLM-5.1 batched ${args.noMtp ? "decode" : "MTP"}: batch=${args.batchSize}, max_tokens=${args.maxNewTokens}, depth=${args.noMtp ? "off" : args.mtp}, cuda_graph=${args.noCudaGraph ? "off" : "on"}`);
    using captureManager = new CaptureManager(ops);
    captureManager.disabled = args.noCudaGraph;
    for (let run = 0; run <= args.warmupRuns; run++) {
      const warmup = run < args.warmupRuns;
      const profiling = args.profile && !warmup;
      if (!warmup && args.cooldownSeconds > 0) {
        await ops.synchronizeAsync();
        console.log(`Cooling down for ${args.cooldownSeconds}s before the measured run (keeping cached graphs).`);
        await sleep(args.cooldownSeconds * 1000);
      }
      console.log(`\n=== ${warmup ? "Warmup" : "Measured"} run ${run + 1}/${args.warmupRuns + 1} (cached graphs: ${captureManager.captured.size}) ===`);
      await ops.synchronizeAsync();
      if (profiling) profilerStart();
      try {
        await runBatch(model, ws, cache, args, captureManager);
      } finally {
        // Drain every GPU before ending collection, including on a failed run.
        try { await ops.synchronizeAsync(); }
        finally { if (profiling) profilerStop(); }
      }
      ws.clearTracking();
      const capturedGraphs = [...captureManager.captured.values()].filter(entry => entry.graphExec !== null).length;
      console.log(`CUDA graphs after ${warmup ? "warmup" : "measured"} run: ${capturedGraphs} captured, ${captureManager.captured.size - capturedGraphs} warming up (currently cached logical graphs across GPUs).`);
    }
  } finally {
    freeResources(model, cache, ws, ops, gpuDevices);
  }
}

if (require.main === module) {
  main().catch(error => {
    console.error("Failed:", error);
    process.exit(1);
  });
}
