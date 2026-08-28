import { ChatCache } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { ExecutionState, ExecutionWorkspace } from "./execution-workspace";
import { Glm51Model } from "./glm51_model";
import { freeModelRuntime, loadModelRuntime, parseModelArgs } from "./model_cli";
import { Tensor } from "./tensor";

interface BenchArgs {
  seqLen: number;
  chunkSize: number;
  warmupRuns: number;
  benchRuns: number;
  pageSize: number;
  maxPages?: number;
}

function parseArgs(argv: string[]): BenchArgs {
  const args: BenchArgs = {
    seqLen: 128 * 1024,
    chunkSize: 4096,
    warmupRuns: 1,
    benchRuns: 1,
    pageSize: 64,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--seq-len" && i + 1 < argv.length) args.seqLen = parseInt(argv[++i], 10);
    else if (arg === "--chunk-size" && i + 1 < argv.length) args.chunkSize = parseInt(argv[++i], 10);
    else if (arg === "--warmup" && i + 1 < argv.length) args.warmupRuns = parseInt(argv[++i], 10);
    else if (arg === "--runs" && i + 1 < argv.length) args.benchRuns = parseInt(argv[++i], 10);
    else if (arg === "--page-size" && i + 1 < argv.length) args.pageSize = parseInt(argv[++i], 10);
    else if (arg === "--max-pages" && i + 1 < argv.length) args.maxPages = parseInt(argv[++i], 10);
    else if (arg === "--help") {
      console.log(`Usage: npx tsx src/run_model_loader.ts [model options] src/run_phased_prefill_benchmark.ts [options]
Options:
  --seq-len <n>      Total prefill length (default: 131072)
  --chunk-size <n>   Length of each A/B chunk (default: 4096)
  --warmup <n>       Warmup runs (default: 1)
  --runs <n>         Timed runs (default: 1)
  --page-size <n>    KV cache page size (default: 64)
  --max-pages <n>    KV cache page count
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
  return args;
}

function planChunk(model: Glm51Model, ws: ExecutionWorkspace, cache: ChatCache, tokenIds: number[]): ExecutionState {
  const state = ws.planPrefill(model, 1, [tokenIds.length], cache);
  state.setInput([tokenIds]);
  return state;
}

function closeGenerator(generator: Generator<void, Tensor, void>): void {
  generator.return(undefined as never);
}

function runPhasedPair(
  model: Glm51Model,
  glm: DeviceOps,
  ws: ExecutionWorkspace,
  cache: ChatCache,
  tokensA: number[],
  tokensB: number[],
): void {
  using _tracking = ws.startTracking();
  const stateA = planChunk(model, ws, cache, tokensA);
  const stateB = planChunk(model, ws, cache, tokensB);
  const generatorA = model.forwardPhased(stateA);
  const generatorB = model.forwardPhased(stateB);
  let resultA = generatorA.next();
  let resultB: IteratorResult<void, Tensor> | undefined;

  try {
    while (!resultA.done) {
      using streamB = glm.withStream(() => generatorB.next());
      try {
        resultA = generatorA.next();
      } finally {
        // Queue the join after A's phase. The next iteration forks from here.
        streamB.streamWaitEvent();
      }
      resultB = streamB.result;
      if (resultB.done) throw new Error("B completed before A");
    }

    resultB = generatorB.next();
    if (!resultB.done) throw new Error("B did not complete one phase after A");
    using hiddenA = resultA.value;
    using hiddenB = resultB.value;
  } finally {
    if (!resultA.done) closeGenerator(generatorA);
    if (!resultB?.done) closeGenerator(generatorB);
  }
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

function runPrefill(
  model: Glm51Model,
  glm: DeviceOps,
  ws: ExecutionWorkspace,
  cache: ChatCache,
  inputIds: readonly number[],
  chunkSize: number,
  runLabel: string,
): void {
  let offset = 0;
  let chunk = 0;
  while (offset + chunkSize < inputIds.length) {
    const aEnd = Math.min(offset + chunkSize, inputIds.length);
    const bEnd = Math.min(aEnd + chunkSize, inputIds.length);
    const pairStart = performance.now();
    runPhasedPair(
      model, glm, ws, cache,
      inputIds.slice(offset, aEnd),
      inputIds.slice(aEnd, bEnd),
    );
    // The next pair reuses plan-slot pinned buffers. Their asynchronous copies
    // must complete before the host writes the next plans into those buffers.
    glm.synchronize();
    const pairElapsed = performance.now() - pairStart;
    const pairTokens = bEnd - offset;
    console.log(`  ${runLabel} chunks ${chunk + 1}-${chunk + 2}: ${pairTokens} tokens, ${pairElapsed.toFixed(0)}ms (${(pairTokens / (pairElapsed / 1000)).toFixed(0)} tok/s)`);
    offset = bEnd;
    chunk += 2;
  }
  if (offset < inputIds.length) {
    const tailStart = performance.now();
    runSequentialTail(model, ws, cache, inputIds.slice(offset));
    glm.synchronize();
    const tailElapsed = performance.now() - tailStart;
    const tailTokens = inputIds.length - offset;
    console.log(`  ${runLabel} chunk ${chunk + 1}: ${tailTokens} tokens, ${tailElapsed.toFixed(0)}ms (${(tailTokens / (tailElapsed / 1000)).toFixed(0)} tok/s)`);
  }
  glm.synchronize();
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const args = parseArgs(argv);
  const modelArgs = parseModelArgs(argv);
  if (!modelArgs.useGlm51) throw new Error("Phased prefill requires --glm51");

  const runtime = await loadModelRuntime(modelArgs);
  const model = runtime.model as Glm51Model;
  const { glm } = runtime;
  const worldSize = modelArgs.gpus.length;
  const maxPages = args.maxPages
    ?? Math.ceil(args.seqLen / args.pageSize / (modelArgs.cp ? worldSize : 1)) + 64;
  const cache = model.createChatCache(maxPages, 1, args.seqLen + 1, args.pageSize);
  const ws = new ExecutionWorkspace(glm, 1, args.chunkSize);
  const inputIds = new Array<number>(args.seqLen).fill(1);

  console.log(`GLM-5.1 Phased Prefill | GPUs ${modelArgs.gpus.join(",")} | seq_len=${args.seqLen} | chunk_size=${args.chunkSize} | cp=${modelArgs.cp} | mtp=off`);
  console.log(`Chunks: ${Math.ceil(args.seqLen / args.chunkSize)} | max_pages=${maxPages} | warmup=${args.warmupRuns} | runs=${args.benchRuns}`);

  try {
    for (let run = 0; run < args.warmupRuns + args.benchRuns; run++) {
      cache.reset(1);
      const warmup = run < args.warmupRuns;
      const label = warmup ? `warmup ${run + 1}` : `run ${run - args.warmupRuns + 1}`;
      const start = performance.now();
      runPrefill(model, glm, ws, cache, inputIds, args.chunkSize, label);
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
