import type { ChatCache, ChatModel } from "./chat_model";
import { ExecutionWorkspace } from "./execution-workspace";
import { freeModelRuntime, loadModelRuntime, parseModelArgs } from "./model_cli";

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
    chunkSize: 8192,
    warmupRuns: 1,
    benchRuns: 1,
    pageSize: 64,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--seq-len" && i + 1 < argv.length) {
      args.seqLen = parseInt(argv[++i], 10);
    } else if (arg === "--chunk-size" && i + 1 < argv.length) {
      args.chunkSize = parseInt(argv[++i], 10);
    } else if (arg === "--warmup" && i + 1 < argv.length) {
      args.warmupRuns = parseInt(argv[++i], 10);
    } else if (arg === "--runs" && i + 1 < argv.length) {
      args.benchRuns = parseInt(argv[++i], 10);
    } else if (arg === "--page-size" && i + 1 < argv.length) {
      args.pageSize = parseInt(argv[++i], 10);
    } else if (arg === "--max-pages" && i + 1 < argv.length) {
      args.maxPages = parseInt(argv[++i], 10);
    } else if (arg === "--help") {
      console.log(`Usage: npx tsx src/run_model_loader.ts [model options] src/run_prefill_benchmark.ts [options]
Options:
  --seq-len <n>      Total prefill length (default: 131072)
  --chunk-size <n>   Execution tokens per call, including overlap (default: 8192)
  --warmup <n>       Warmup runs (default: 1)
  --runs <n>         Timed runs (default: 1)
  --page-size <n>    KV cache page size (default: 64)
  --max-pages <n>    KV cache page count
  --help            Show this help

Phased prefill is selected automatically for eligible chunks.
Set GLM_PHASED_PREFILL=0 to disable it.`);
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
    if (!Number.isInteger(value) || value < minimum) {
      throw new Error(`Invalid --${name}: ${value}`);
    }
  }
  if (args.maxPages !== undefined && (!Number.isInteger(args.maxPages) || args.maxPages < 1)) {
    throw new Error(`Invalid --max-pages: ${args.maxPages}`);
  }
  return args;
}

async function runPrefill(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  inputIds: readonly number[], chunkSize: number, label: string,
): Promise<void> {
  let chunk = 0;
  let remaining = [[...inputIds]];
  while (remaining[0].length) {
    const previousLength = remaining[0].length;
    const start = performance.now();
    // The model budgets both new input and overlap inside the requested chunk size.
    const result = await model.executePrefill(ws, cache, remaining, undefined, chunkSize);
    remaining = result.remainingInputIdsList;
    const consumed = previousLength - remaining[0].length;
    const elapsed = performance.now() - start;
    console.log(`  ${label} chunk ${++chunk}: ${consumed} new tokens, ${result.prefillInputIdsList[0].length} execution tokens, ${elapsed.toFixed(0)}ms (${(consumed / (elapsed / 1000)).toFixed(0)} tok/s)`);
  }
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const args = parseArgs(argv);
  const modelArgs = parseModelArgs(argv);
  if (!modelArgs.useGlm51) {
    throw new Error("Prefill benchmark requires --glm51");
  }

  const runtime = await loadModelRuntime(modelArgs);
  const { model, ops } = runtime;
  try {
    const worldSize = modelArgs.gpus.length;
    const maxPages = args.maxPages
      ?? Math.ceil(args.seqLen / args.pageSize / (modelArgs.cp ? worldSize : 1)) + 64;
    using cache = model.createChatCache(maxPages, 1, args.seqLen + 1, args.pageSize);
    using ws = new ExecutionWorkspace(ops, 1, Math.min(args.chunkSize, args.seqLen));
    const inputIds = new Array<number>(args.seqLen).fill(1);

    console.log(`GLM-5.1 Prefill | GPUs ${modelArgs.gpus.join(",")} | seq_len=${args.seqLen} | chunk_size=${args.chunkSize} | cp=${modelArgs.cp} | mtp=${modelArgs.mtp} | phased=${process.env.GLM_PHASED_PREFILL !== "0"}`);
    console.log(`max_pages=${maxPages} | warmup=${args.warmupRuns} | runs=${args.benchRuns}`);

    for (let run = 0; run < args.warmupRuns + args.benchRuns; run++) {
      cache.reset(1);
      const warmup = run < args.warmupRuns;
      const label = warmup ? `warmup ${run + 1}` : `run ${run - args.warmupRuns + 1}`;
      const start = performance.now();
      await runPrefill(model, ws, cache, inputIds, args.chunkSize, label);
      const elapsed = performance.now() - start;
      console.log(`${label}: ${elapsed.toFixed(0)}ms (${(args.seqLen / (elapsed / 1000)).toFixed(0)} tok/s)`);
    }
  } finally {
    freeModelRuntime(runtime);
  }
}

main().catch(error => {
  console.error("Failed:", error);
  process.exit(1);
});
