import { CaptureManager } from "./capture-manager";
import { type ChatCache, type ChatModel, type Tokenizer } from "./chat_model";
import { type DeviceOps } from "./device_ops";
import { MemcpyKind } from "./enums";
import { ExecutionWorkspace } from "./execution-workspace";
import { type GlmOps, I32 } from "./glm_ops";
import { createDeviceOps, loadModel, type ModelCliArgs, parseModelArgs, resolveModelSelection } from "./model_cli";
import { MtpStats, mtpTreeDecode } from "./mtp";
import { ParallelOps } from "./parallel_ops";
import { type Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";
import { WorkspaceBase } from "./workspace";

const PROMPTS = [
  "tell me a 1000 word story",
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

async function runBatch(model: ChatModel, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache, args: Args): Promise<void> {
  const prompts = PROMPTS.slice(0, args.batchSize);
  const inputIds = prompts.map(prompt => tokenizePrompt(model.tokenizer, prompt));
  const longestPrompt = Math.max(...inputIds.map(ids => ids.length));
  if (longestPrompt + args.maxNewTokens > args.maxSeqLen) {
    throw new Error(`Prompt plus generation budget exceeds --max-seq-len (${longestPrompt} + ${args.maxNewTokens} > ${args.maxSeqLen})`);
  }

  cache.reset(args.batchSize);
  const suffixIds = inputIds.map((ids, batch) => cache.prefixMatch(batch, ids));
  const mtpInputIds = model.prepareMtpInput(cache, suffixIds);
  const seqLens = mtpInputIds.map(ids => ids.length);

  using captureManager = new CaptureManager(glm);
  captureManager.disabled = args.noCudaGraph;
  using sampleWorkspace = new WorkspaceBase(glm);
  using mtpHiddenStates = new UsingHolder<Tensor>(undefined!);
  using sharedSlots = new UsingHolder<Tensor>(undefined!);
  using sharedSlotsLength = new UsingHolder<Tensor>(undefined!);

  const state = ws.planPrefill(model, args.batchSize, seqLens, cache);
  state.sharedSlots = sharedSlots;
  state.sharedSlotsLength = sharedSlotsLength;
  state.setInput(mtpInputIds);

  let gpuFirstTokens: Tensor | undefined;
  {
    using _tracker = state.ws.startTracking();
    const draftExtend = model.forwardMtpDraftExtend!(state, args.mtpDraftTopk, hiddenStates => {
      using logits = state.computeLogits(hiddenStates, model);
      using argmax = logits.argmax();
      gpuFirstTokens = sampleWorkspace.ensureAlloc([ws.maxBatch], "I32", "mtp_batch_first_tokens");
      gpuFirstTokens.memcpy(argmax, argmax.bytes, MemcpyKind.DeviceToDevice);
      return gpuFirstTokens.narrow(0, args.batchSize);
    });
    using _seed = draftExtend.mtpHiddenStates;
    using _token = draftExtend.token;

    let maxIntermediateWidth = 1;
    let width = 1;
    for (const topk of args.mtpDraftTopk.slice(0, -1)) {
      width *= topk;
      maxIntermediateWidth = Math.max(maxIntermediateWidth, width);
    }
    const scratch = ws.alloc([ws.maxBatch * maxIntermediateWidth, model.cfg.hiddenSize], _seed.type);
    scratch.memcpy(_seed, _seed.bytes, MemcpyKind.DeviceToDevice);
    mtpHiddenStates.replace(scratch.removeTracking());
  }

  const firstTokensHost = sampleWorkspace.ensureAllocPinned([ws.maxBatch], "I32", "mtp_batch_first_tokens_host");
  firstTokensHost.memcpy(gpuFirstTokens!, args.batchSize * I32, MemcpyKind.DeviceToHost);
  await glm.synchronizeAsync();
  const firstTokenBuffer = firstTokensHost.readPinnedBuffer();
  const currentTokens = Array.from({ length: args.batchSize }, (_, batch) => firstTokenBuffer.readInt32LE(batch * I32));
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
    const result = await mtpTreeDecode(
      captureManager,
      model,
      mtpHiddenStates.value,
      sharedSlots.value,
      sharedSlotsLength.value,
      ws,
      currentTokens,
      args.mtpDraftTopk,
      cache,
    );
    if (!result.warmup) {
      for (const accepted of result.numAccepted) {
        mtpStats.observe(result.numDraftTokens, accepted);
      }
    }

    for (let batch = 0; batch < args.batchSize; batch++) {
      for (const token of result.tokens[batch]) {
        currentTokens[batch] = token;
        cache.reportTokens(batch, [token]);
        generated[batch].push(token);

        const now = performance.now();
        if (firstPostWarmupTime === 0) firstPostWarmupTime = now;
        lastTokenTime = now;
        if (result.warmup) {
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
  let model: ChatModel | undefined;
  let cache: ChatCache | undefined;
  let ws: ExecutionWorkspace | undefined;

  try {
    model = await loadModel(glm, args, modelDir);
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
