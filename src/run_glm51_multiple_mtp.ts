import { CaptureManager } from "./capture-manager";
import { type ChatCache, type ChatModel, type Tokenizer } from "./chat_model";
import { type DeviceOps } from "./device_ops";
import { executePlan, ExecutionWorkspace } from "./execution-workspace";
import { Glm51Model } from "./glm51_model";
import { type GlmOps } from "./glm_ops";
import { createDeviceOps, loadModel, type ModelCliArgs, parseModelArgs, resolveModelSelection } from "./model_cli";
import { MtpStats } from "./mtp";
import { ParallelOps } from "./parallel_ops";

const PROMPTS = [
  "tell me about india",
  "tell me a 1000 word story",
];

interface Args extends ModelCliArgs {
  batchSize: number;
  maxNewTokens: number;
  maxSeqLen: number;
  maxPages: number;
  mtpDraftTopk: number[];
  noCudaGraph: boolean;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    ...parseModelArgs(argv),
    batchSize: 2,
    maxNewTokens: 2000,
    maxSeqLen: 4096,
    maxPages: 256,
    mtpDraftTopk: [1, 1, 1],
    noCudaGraph: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--batch-size" && i + 1 < argv.length) args.batchSize = parseInt(argv[++i], 10);
    else if (arg === "--max-new-tokens" && i + 1 < argv.length) args.maxNewTokens = parseInt(argv[++i], 10);
    else if (arg === "--max-seq-len" && i + 1 < argv.length) args.maxSeqLen = parseInt(argv[++i], 10);
    else if (arg === "--max-pages" && i + 1 < argv.length) args.maxPages = parseInt(argv[++i], 10);
    else if (arg === "--mtp-draft-topk" && i + 1 < argv.length) {
      args.mtpDraftTopk = argv[++i].split(",").map(value => parseInt(value.trim(), 10));
    } else if (arg === "--no-cuda-graph") args.noCudaGraph = true;
  }

  if (!Number.isInteger(args.batchSize) || args.batchSize < 1 || args.batchSize > PROMPTS.length) {
    throw new Error(`--batch-size must be between 1 and ${PROMPTS.length}`);
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
  if (args.mtpDraftTopk.length === 0 || args.mtpDraftTopk.some(topk => !Number.isInteger(topk) || topk < 1)) {
    throw new Error(`Invalid --mtp-draft-topk: ${args.mtpDraftTopk.join(",")}`);
  }
  if (!args.useGlm51 || !args.mtp) {
    throw new Error("run_glm51_multiple_mtp requires --glm51 --mtp");
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

function freeResources(model: ChatModel | undefined, cache: ChatCache | undefined, ws: ExecutionWorkspace | undefined, glm: DeviceOps, gpuDevices: GlmOps[]): void {
  let cleanupError: unknown;
  const dispose = (fn: () => void) => {
    try {
      fn();
    } catch (error) {
      cleanupError ??= error;
    }
  };

  dispose(() => glm.synchronize());
  if (cache) dispose(() => cache[Symbol.dispose]());
  if (ws) dispose(() => ws[Symbol.dispose]());
  if (model) dispose(() => model[Symbol.dispose]());
  if (glm instanceof ParallelOps) dispose(() => glm[Symbol.dispose]());
  for (const device of gpuDevices) dispose(() => device[Symbol.dispose]());
  if (cleanupError) throw cleanupError;
}

async function runBatch(model: Glm51Model, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache, args: Args): Promise<void> {
  const prompts = PROMPTS.slice(0, args.batchSize);
  const inputIds = prompts.map(prompt => tokenizePrompt(model.tokenizer, prompt));
  const longestPrompt = Math.max(...inputIds.map(ids => ids.length));
  if (longestPrompt + args.maxNewTokens > args.maxSeqLen) {
    throw new Error(`Prompt plus generation budget exceeds --max-seq-len (${longestPrompt} + ${args.maxNewTokens} > ${args.maxSeqLen})`);
  }

  cache.reset(args.batchSize);
  const suffixIds = inputIds.map((ids, batch) => cache.prefixMatch(batch, ids));
  const mtpInputIds = model.prepareMtpInput(cache, suffixIds);

  using captureManager = new CaptureManager(glm);
  captureManager.disabled = args.noCudaGraph;

  let currentDraft = (await executePlan(captureManager, ws, model.planPrefillMtpDraftExtend(ws, cache, mtpInputIds, args.mtpDraftTopk))).result;
  const currentTokens = [...currentDraft.targetTokens];
  const generated = currentTokens.map(token => [token]);
  const finished = currentTokens.map(token => model.eosIds.has(token) || args.maxNewTokens === 1);
  const mtpStats = new MtpStats(args.mtpDraftTopk.length);
  let firstPostWarmupTime = 0;
  let lastTokenTime = 0;
  let postWarmupTokenCount = 0;

  for (let batch = 0; batch < args.batchSize; batch++) {
    cache.reportTokens(batch, suffixIds[batch]);
    cache.reportTokens(batch, [currentTokens[batch]]);
  }

  const started = performance.now();
  while (!finished.some(Boolean)) {
    const step = await executePlan(captureManager, ws, model.planTargetVerification(ws, cache, currentDraft));
    const warmup = step.warmup;
    currentDraft = step.result.draft;
    const stepTokens = step.result.tokens;
    const stepAccepted = step.result.numAccepted;
    const numDraftTokens = step.result.numDraftTokens;

    if (!warmup) {
      for (const count of stepAccepted) {
        mtpStats.observe(numDraftTokens, count);
      }
    }

    for (let batch = 0; batch < args.batchSize; batch++) {
      const reportedTokens: number[] = [];
      for (const token of stepTokens[batch]) {
        currentTokens[batch] = token;
        reportedTokens.push(token);
        generated[batch].push(token);

        const now = performance.now();
        if (firstPostWarmupTime === 0) firstPostWarmupTime = now;
        lastTokenTime = now;
        if (warmup) {
          firstPostWarmupTime = 0;
          postWarmupTokenCount = 0;
        } else {
          postWarmupTokenCount++;
        }

        if (model.eosIds.has(token) || generated[batch].length >= args.maxNewTokens) {
          finished[batch] = true;
          break;
        }
      }
      cache.reportTokens(batch, reportedTokens);
    }
  }

  const elapsed = (performance.now() - started) / 1000;
  const decodeTokPerSec = postWarmupTokenCount > 1 && firstPostWarmupTime > 0
    ? postWarmupTokenCount / ((lastTokenTime - firstPostWarmupTime) / 1000)
    : 0;
  console.log(`Stopped when batch ${finished.findIndex(Boolean) + 1} completed after ${elapsed.toFixed(1)}s.`);
  for (let batch = 0; batch < args.batchSize; batch++) {
    const visibleTokens = generated[batch].filter(token => !model.eosIds.has(token));
    console.log(`\n--- Prompt ${batch + 1} ---\n${prompts[batch]}`);
    console.log(`\n--- Response ${batch + 1} (${visibleTokens.length} tokens) ---`);
    console.log(model.tokenizer.decode(visibleTokens, { skip_special_tokens: true }));
  }
  console.log(`\ndecode=${decodeTokPerSec.toFixed(1)} tok/s`);
  console.log(mtpStats.log() || "MTP metrics: no post-warmup drafts");
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
  Error.stackTraceLimit = 20;
  const args = parseArgs(argv);
  const { modelDir } = resolveModelSelection(args);
  const { glm, gpuDevices } = createDeviceOps(args);
  let model: Glm51Model | undefined;
  let cache: ChatCache | undefined;
  let ws: ExecutionWorkspace | undefined;

  try {
    const loadedModel = await loadModel(glm, args, modelDir);
    if (!(loadedModel instanceof Glm51Model)) throw new Error("run_glm51_multiple_mtp requires a GLM-5.1 model");
    model = loadedModel;
    cache = model.createChatCache(args.maxPages, args.batchSize, args.maxSeqLen);
    ws = new ExecutionWorkspace(glm, args.batchSize, args.maxSeqLen);
    console.log(`GLM-5.1 batched MTP: batch=${args.batchSize}, max_tokens=${args.maxNewTokens}, topk=${args.mtpDraftTopk.join(",")}, cuda_graph=${args.noCudaGraph ? "off" : "on"}`);
    await runBatch(model, ws, glm, cache, args);
  } finally {
    freeResources(model, cache, ws, glm, gpuDevices);
  }
}

if (require.main === module) {
  main().catch(error => {
    console.error("Failed:", error);
    process.exit(1);
  });
}
