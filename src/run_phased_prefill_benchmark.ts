import { CaptureManager } from "./capture-manager";
import { ChatCache } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { executePlan, ExecutionState, ExecutionWorkspace } from "./execution-workspace";
import { Glm51Model } from "./glm51_model";
import { freeModelRuntime, loadModelRuntime, parseModelArgs } from "./model_cli";
import { PhasedPrefillRunner, splitRaggedInput } from "./phased-prefill";
import { Tensor } from "./tensor";

interface BenchArgs {
  seqLen: number;
  chunkSize: number;
  warmupRuns: number;
  benchRuns: number;
  pageSize: number;
  maxPages?: number;
  mtpDraftTopk: number[];
}

function parseArgs(argv: string[]): BenchArgs {
  const args: BenchArgs = {
    seqLen: 128 * 1024,
    chunkSize: 4096,
    warmupRuns: 1,
    benchRuns: 1,
    pageSize: 64,
    mtpDraftTopk: [1, 1, 1],
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--seq-len" && i + 1 < argv.length) args.seqLen = parseInt(argv[++i], 10);
    else if (arg === "--chunk-size" && i + 1 < argv.length) args.chunkSize = parseInt(argv[++i], 10);
    else if (arg === "--warmup" && i + 1 < argv.length) args.warmupRuns = parseInt(argv[++i], 10);
    else if (arg === "--runs" && i + 1 < argv.length) args.benchRuns = parseInt(argv[++i], 10);
    else if (arg === "--page-size" && i + 1 < argv.length) args.pageSize = parseInt(argv[++i], 10);
    else if (arg === "--max-pages" && i + 1 < argv.length) args.maxPages = parseInt(argv[++i], 10);
    else if (arg === "--mtp-draft-topk" && i + 1 < argv.length) args.mtpDraftTopk = argv[++i].split(",").map(value => parseInt(value.trim(), 10));
    else if (arg === "--help") {
      console.log(`Usage: npx tsx src/run_model_loader.ts [model options] src/run_phased_prefill_benchmark.ts [options]
Options:
  --seq-len <n>      Total prefill length (default: 131072)
  --chunk-size <n>   Length of each A/B chunk (default: 4096)
  --warmup <n>       Warmup runs (default: 1)
  --runs <n>         Timed runs (default: 1)
  --page-size <n>    KV cache page size (default: 64)
  --max-pages <n>    KV cache page count
  --mtp-draft-topk <list>  MTP draft top-k per depth (default: 1,1,1)
  --help             Show this help`);
      process.exit(0);
    }
  }

  for (const [name, value] of Object.entries({
    "seq-len": args.seqLen,
    "chunk-size": args.chunkSize,
    warmup: args.warmupRuns,
    runs: args.benchRuns,
    "page-size": args.pageSize,
  })) {
    const minimum = name === "warmup" || name === "runs" ? 0 : 1;
    if (!Number.isInteger(value) || value < minimum) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (args.chunkSize >= args.seqLen) throw new Error("--chunk-size must leave at least two chunks");
  if (args.maxPages !== undefined && (!Number.isInteger(args.maxPages) || args.maxPages < 1)) {
    throw new Error(`Invalid --max-pages: ${args.maxPages}`);
  }
  if (args.mtpDraftTopk.length === 0 || args.mtpDraftTopk.some(topk => !Number.isInteger(topk) || topk < 1)) {
    throw new Error(`Invalid --mtp-draft-topk: ${args.mtpDraftTopk.join(",")}`);
  }
  return args;
}

function planChunk(model: Glm51Model, ws: ExecutionWorkspace, cache: ChatCache, tokenIds: number[]): ExecutionState {
  const state = ws.planPrefill(model, 1, [tokenIds.length], cache);
  state.setInput([tokenIds]);
  return state;
}

function runPhasedPair(
  model: Glm51Model,
  runner: PhasedPrefillRunner,
  ws: ExecutionWorkspace,
  cache: ChatCache,
  tokensA: number[],
  tokensB: number[],
): void {
  using _tracking = ws.startTracking();
  const stateA = planChunk(model, ws, cache, tokensA);
  const stateB = planChunk(model, ws, cache, tokensB);
  runner.runPair(stateA, stateB);
}

function runMtpPhasedPair(
  model: Glm51Model,
  runner: PhasedPrefillRunner,
  ws: ExecutionWorkspace,
  cache: ChatCache,
  tokensA: number[],
  tokensB: number[],
  nextToken: number,
): void {
  using planA = model.planPrefillMtpChunkPhased!(ws, cache, [tokensA], [tokensB[0]]);
  using planB = model.planPrefillMtpChunkPhased!(ws, cache, [tokensB], [nextToken]);
  runner.runPlanPair(planA, planB);
}

function runSequentialTail(
  model: Glm51Model,
  ws: ExecutionWorkspace,
  cache: ChatCache,
  tokenIds: number[],
): void {
  using _tracking = ws.startTracking();
  const state = planChunk(model, ws, cache, tokenIds);
  using hidden = model.forwardModel(state);
}

async function runPrefill(
  model: Glm51Model,
  glm: DeviceOps,
  ws: ExecutionWorkspace,
  cache: ChatCache,
  inputIds: readonly number[],
  chunkSize: number,
  runLabel: string,
  mtpDraftTopk?: readonly number[],
): Promise<void> {
  const runner = new PhasedPrefillRunner(model, glm);
  const pairBudget = chunkSize * 2;
  const finalInputBudget = mtpDraftTopk ? pairBudget - 1 : pairBudget;
  let offset = 0;
  let chunk = 0;
  while (mtpDraftTopk ? inputIds.length - offset > finalInputBudget : offset + chunkSize < inputIds.length) {
    const pairTokens = mtpDraftTopk
      ? Math.min(pairBudget, inputIds.length - offset - 1)
      : Math.min(pairBudget, inputIds.length - offset);
    const pairInput = inputIds.slice(offset, offset + pairTokens);
    const split = splitRaggedInput([pairInput]);
    if (!split) throw new Error(`Unable to split phased prefill pair of ${pairTokens} tokens`);
    const pairStart = performance.now();
    if (mtpDraftTopk) {
      runMtpPhasedPair(model, runner, ws, cache, split.inputA[0], split.inputB[0], inputIds[offset + pairTokens]);
    } else {
      runPhasedPair(model, runner, ws, cache, split.inputA[0], split.inputB[0]);
    }
    // The next pair reuses plan-slot pinned buffers. Their asynchronous copies
    // must complete before the host writes the next plans into those buffers.
    await glm.synchronizeAsync();
    if (mtpDraftTopk) {
      cache.reportTokens(0, pairInput);
      ws.assertClear();
    }
    ws.resetPlanSlots();
    const pairElapsed = performance.now() - pairStart;
    console.log(`  ${runLabel} chunks ${chunk + 1}-${chunk + 2}: ${pairTokens} tokens, ${pairElapsed.toFixed(0)}ms (${(pairTokens / (pairElapsed / 1000)).toFixed(0)} tok/s)`);
    offset += pairTokens;
    chunk += 2;
  }
  if (offset < inputIds.length) {
    const tailStart = performance.now();
    const tailInput = inputIds.slice(offset);
    if (mtpDraftTopk) {
      const mtpInput = model.prepareMtpInput(cache, [tailInput]);
      const captureManager = new CaptureManager(glm);
      captureManager.disabled = true;
      await executePlan(captureManager, ws, model.planPrefillMtpDraftExtend!(ws, cache, mtpInput, mtpDraftTopk));
    } else {
      runSequentialTail(model, ws, cache, tailInput);
      await glm.synchronizeAsync();
    }
    const tailElapsed = performance.now() - tailStart;
    const tailTokens = tailInput.length;
    console.log(`  ${runLabel} chunk ${chunk + 1}: ${tailTokens} tokens, ${tailElapsed.toFixed(0)}ms (${(tailTokens / (tailElapsed / 1000)).toFixed(0)} tok/s)`);
  }
  await glm.synchronizeAsync();
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const args = parseArgs(argv);
  const modelArgs = parseModelArgs(argv);
  if (!modelArgs.useGlm51) throw new Error("Phased prefill requires --glm51");

  const runtime = await loadModelRuntime(modelArgs);
  const model = runtime.model as Glm51Model;
  const { glm } = runtime;
  if (modelArgs.mtp && (!model.planPrefillMtpChunkPhased || !model.planPrefillMtpDraftExtend)) {
    throw new Error("MTP phased prefill requires phased chunk and draft-extend support");
  }
  const worldSize = modelArgs.gpus.length;
  const maxPages = args.maxPages
    ?? Math.ceil(args.seqLen / args.pageSize / (modelArgs.cp ? worldSize : 1)) + 64;
  const cache = model.createChatCache(maxPages, 1, args.seqLen + 1, args.pageSize);
  const ws = new ExecutionWorkspace(glm, 1, modelArgs.mtp ? args.chunkSize * 2 : args.chunkSize);
  const inputIds = new Array<number>(args.seqLen).fill(1);

  console.log(`GLM-5.1 Phased Prefill | GPUs ${modelArgs.gpus.join(",")} | seq_len=${args.seqLen} | chunk_size=${args.chunkSize} | cp=${modelArgs.cp} | mtp=${modelArgs.mtp ? args.mtpDraftTopk.join(",") : "off"}`);
  console.log(`Chunks: ${Math.ceil(args.seqLen / args.chunkSize)} | max_pages=${maxPages} | warmup=${args.warmupRuns} | runs=${args.benchRuns}`);

  try {
    for (let run = 0; run < args.warmupRuns + args.benchRuns; run++) {
      cache.reset(1);
      const warmup = run < args.warmupRuns;
      const label = warmup ? `warmup ${run + 1}` : `run ${run - args.warmupRuns + 1}`;
      const start = performance.now();
      await runPrefill(model, glm, ws, cache, inputIds, args.chunkSize, label, modelArgs.mtp ? args.mtpDraftTopk : undefined);
      const elapsed = performance.now() - start;
      const throughput = args.seqLen / (elapsed / 1000);
      console.log(`${label}: ${elapsed.toFixed(0)}ms (${throughput.toFixed(0)} tok/s)`);
    }
  } finally {
    glm.synchronize();
    cache.free();
    ws.free();
    freeModelRuntime(runtime);
  }
}

main().catch(error => {
  console.error("Failed:", error);
  process.exit(1);
});
